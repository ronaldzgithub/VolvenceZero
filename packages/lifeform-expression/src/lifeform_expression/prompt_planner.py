"""Structured PromptPlanner over kernel snapshot state.

Reads only what the kernel ResponseSynthesizer surface exposes:

* ``ResponseContext`` — regime, abstract action, alert count, switch gate,
  joint-loop schedule, primary reflection lesson / tension.
* ``ResponseAssemblySnapshot`` — knowledge / case / playbook / boundary
  signals + speech plan + continuum target position.

Produces a ``PromptPlan`` (frozen) that the ``GroundedResponseSynthesizer``
uses to:

* pick a section ordering (support-first / clarify-first / structure-first)
* allocate a response section budget
* state the turn intent in machine-readable form

The planner is **pure**: same inputs → same plan, no global state. It does
not read kernel internals; it does not write to any owner; it does not
import lifeform_core. A ``PromptPlan`` is a snapshot-shaped product-side
object, not a kernel snapshot — it lives in the lifeform layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

from lifeform_core import VitalsSnapshot
from volvence_zero.agent.response import ResponseContext
from volvence_zero.application.runtime import ResponseAssemblySnapshot
from volvence_zero.interlocutor import InterlocutorState
from volvence_zero.regime import ParticipationHint, ParticipationLevel


class SectionId(str, Enum):
    """Logical sections a response can include, in priority order.

    The planner picks a subset and an order based on regime / continuum.
    """

    ACKNOWLEDGE_PRESSURE = "acknowledge_pressure"
    REGIME_FRAME = "regime_frame"
    OPEN_LOOP_HANDOFF = "open_loop_handoff"
    CLARIFICATION = "clarification"
    NEXT_STEP = "next_step"
    BOUNDARY_DISCLAIMER = "boundary_disclaimer"
    REFLECTION_HOOK = "reflection_hook"
    CONTINUITY_NOTE = "continuity_note"


class TurnIntent(str, Enum):
    """Structured turn intents.

    The first six values mirror the kernel's ``ResponseAssemblySnapshot.expression_intent``
    so the planner can adopt whatever the kernel decided rather than re-deriving it
    from regime alone. ``JUDGMENT_PROCESS`` and ``REFER_OUT`` are still delegated to
    the base kernel renderer; the rest are rendered by ``GroundedResponseSynthesizer``.
    """

    SUPPORT_FIRST = "support-first"
    REPAIR_FIRST = "repair-first"
    STRUCTURE_FIRST = "structure-first"
    WARMTH_FIRST = "warmth-first"
    CLARIFY_FIRST = "clarify-first"
    DIRECT_ANSWER = "direct-answer"
    REFER_OUT = "refer-out"
    JUDGMENT_PROCESS = "judgment-process"


@dataclass(frozen=True)
class PromptPlan:
    """A read-only plan describing how a turn's response should be shaped."""

    intent: TurnIntent
    sections: tuple[SectionId, ...]
    section_budget: dict[SectionId, int]
    question_budget: int
    must_include_disclaimers: tuple[str, ...] = ()
    rationale_tags: tuple[str, ...] = field(default_factory=tuple)

    def has_section(self, section: SectionId) -> bool:
        return section in self.sections


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


_REGIME_DEFAULT_SECTIONS: dict[str, tuple[SectionId, ...]] = {
    "emotional_support": (
        SectionId.ACKNOWLEDGE_PRESSURE,
        SectionId.REGIME_FRAME,
        SectionId.NEXT_STEP,
        SectionId.CONTINUITY_NOTE,
    ),
    "repair_and_deescalation": (
        SectionId.ACKNOWLEDGE_PRESSURE,
        SectionId.REGIME_FRAME,
        SectionId.OPEN_LOOP_HANDOFF,
        SectionId.NEXT_STEP,
    ),
    "guided_exploration": (
        SectionId.REGIME_FRAME,
        SectionId.CLARIFICATION,
        SectionId.NEXT_STEP,
    ),
    "problem_solving": (
        SectionId.REGIME_FRAME,
        SectionId.NEXT_STEP,
        SectionId.OPEN_LOOP_HANDOFF,
    ),
    "acquaintance_building": (
        SectionId.REGIME_FRAME,
        SectionId.CONTINUITY_NOTE,
        SectionId.NEXT_STEP,
    ),
    "casual_social": (
        SectionId.REGIME_FRAME,
        SectionId.CONTINUITY_NOTE,
    ),
}


_DEFAULT_QUESTION_BUDGET_BY_INTENT: dict[TurnIntent, int] = {
    TurnIntent.SUPPORT_FIRST: 0,
    TurnIntent.REPAIR_FIRST: 0,
    TurnIntent.STRUCTURE_FIRST: 0,
    TurnIntent.WARMTH_FIRST: 1,
    TurnIntent.CLARIFY_FIRST: 1,
    TurnIntent.DIRECT_ANSWER: 0,
    TurnIntent.REFER_OUT: 0,
    TurnIntent.JUDGMENT_PROCESS: 0,
}


class PromptPlanner:
    """Pure planner. Stateless.

    Subclass to override defaults, but do not store mutable state — that
    would make the planner a hidden second owner of conversation strategy
    (R8 / SSOT).
    """

    def plan(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot | None,
        vitals: VitalsSnapshot | None = None,
        participation_hint: ParticipationHint | None = None,
        interlocutor_state: InterlocutorState | None = None,
    ) -> PromptPlan:
        """Build a frozen ``PromptPlan``.

        ``participation_hint`` (Gap 8) is a lifeform-side advisory
        originating in the ``regime`` snapshot. When present the
        planner applies the hint to filter sections that the
        regime says should stay out of this turn's prompt.

        ``interlocutor_state`` (Gap 9 slice 2c) is the 12-axis
        readout of the user's emotional / relational state. When
        present and ``readout_confidence >= 0.30`` the planner
        modulates section selection and question budget along
        continuous axes (warmth / resistance / pace pressure /
        directness / emotional weight / trust). NEVER via keyword
        matches over user text \u2014 the readout already aggregates
        kernel signals into the 12 typed axes. ``None`` and the
        cold-start case (low confidence) are no-ops, so this hook
        is fully backwards-compatible.
        """
        intent = self._pick_intent(context=context, assembly=assembly)
        sections = self._pick_sections(
            context=context, assembly=assembly, intent=intent, vitals=vitals
        )
        sections, hint_rationale = self._apply_participation_hint(
            sections=sections, participation_hint=participation_hint
        )
        sections, il_rationale = self._apply_interlocutor_state(
            sections=sections, interlocutor_state=interlocutor_state
        )
        budget = self._build_section_budget(sections=sections, intent=intent)
        question_budget = self._pick_question_budget(
            intent=intent,
            assembly=assembly,
            interlocutor_state=interlocutor_state,
        )
        disclaimers = (
            tuple(assembly.required_disclaimer_phrases) if assembly is not None else ()
        )
        rationale_tags = self._build_rationale_tags(
            context=context,
            assembly=assembly,
            intent=intent,
            vitals=vitals,
            participation_hint=participation_hint,
            hint_rationale=hint_rationale,
            interlocutor_state=interlocutor_state,
            il_rationale=il_rationale,
        )
        return PromptPlan(
            intent=intent,
            sections=tuple(sections),
            section_budget=budget,
            question_budget=question_budget,
            must_include_disclaimers=disclaimers,
            rationale_tags=rationale_tags,
        )

    # ------------------------------------------------------------------
    # Hooks (override in subclasses)
    # ------------------------------------------------------------------

    def _pick_intent(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot | None,
    ) -> TurnIntent:
        # The kernel's ``expression_intent`` is the canonical decision (it
        # already considered regime + response_mode + temporal + semantic
        # control). Adopt it directly when it lands in our enum so the
        # planner stays aligned with the kernel; only fall back to regime
        # heuristics when assembly is missing or the kernel emits a value
        # we do not yet recognise.
        if assembly is not None:
            try:
                return TurnIntent(assembly.expression_intent)
            except ValueError:
                pass

            mode = assembly.response_mode.value
            if mode == "refer-out":
                return TurnIntent.REFER_OUT
            if mode == "clarify":
                return TurnIntent.CLARIFY_FIRST
            ordering = assembly.ordering_driver
            if ordering in {"continuum-support-first", "continuum-support-clarify"}:
                return TurnIntent.SUPPORT_FIRST
            if ordering == "continuum-clarify-first":
                return TurnIntent.CLARIFY_FIRST
            if ordering == "continuum-structure-first":
                return TurnIntent.STRUCTURE_FIRST
            if assembly.continuum_target_position >= 0.70:
                return TurnIntent.SUPPORT_FIRST
            if assembly.continuum_target_position >= 0.52:
                return TurnIntent.CLARIFY_FIRST
            return TurnIntent.STRUCTURE_FIRST

        # No assembly available — fall back to regime defaults.
        if context.regime_id == "emotional_support":
            return TurnIntent.SUPPORT_FIRST
        if context.regime_id == "repair_and_deescalation":
            return TurnIntent.REPAIR_FIRST
        if context.regime_id == "guided_exploration":
            return TurnIntent.JUDGMENT_PROCESS
        if context.regime_id == "problem_solving":
            return TurnIntent.STRUCTURE_FIRST
        if context.regime_id == "acquaintance_building":
            return TurnIntent.WARMTH_FIRST
        if context.regime_id == "casual_social":
            return TurnIntent.DIRECT_ANSWER
        return TurnIntent.DIRECT_ANSWER

    def _pick_sections(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot | None,
        intent: TurnIntent,
        vitals: VitalsSnapshot | None = None,
    ) -> list[SectionId]:
        if intent is TurnIntent.JUDGMENT_PROCESS:
            return [
                SectionId.ACKNOWLEDGE_PRESSURE,
                SectionId.REGIME_FRAME,
                SectionId.NEXT_STEP,
            ]
        if intent is TurnIntent.REFER_OUT:
            return [
                SectionId.ACKNOWLEDGE_PRESSURE,
                SectionId.BOUNDARY_DISCLAIMER,
                SectionId.NEXT_STEP,
            ]
        if intent is TurnIntent.DIRECT_ANSWER:
            return [SectionId.REGIME_FRAME, SectionId.NEXT_STEP]

        regime_id = (
            (assembly.regime_id if assembly is not None else None)
            or context.regime_id
            or ""
        )
        base = list(_REGIME_DEFAULT_SECTIONS.get(regime_id, (SectionId.REGIME_FRAME, SectionId.NEXT_STEP)))

        if intent is TurnIntent.CLARIFY_FIRST and SectionId.CLARIFICATION not in base:
            base.insert(1, SectionId.CLARIFICATION)
        if intent is TurnIntent.SUPPORT_FIRST and SectionId.ACKNOWLEDGE_PRESSURE not in base:
            base.insert(0, SectionId.ACKNOWLEDGE_PRESSURE)
        if intent is TurnIntent.REPAIR_FIRST:
            if SectionId.ACKNOWLEDGE_PRESSURE not in base:
                base.insert(0, SectionId.ACKNOWLEDGE_PRESSURE)
            if SectionId.OPEN_LOOP_HANDOFF not in base:
                # Repair turns explicitly thread the prior rupture forward
                # rather than letting it drift into the open-loop bucket.
                base.append(SectionId.OPEN_LOOP_HANDOFF)
        if intent is TurnIntent.WARMTH_FIRST and SectionId.CONTINUITY_NOTE not in base:
            base.append(SectionId.CONTINUITY_NOTE)

        # Reflection hook is added when a primary lesson or tension is present.
        if (
            context.primary_reflection_lesson is not None
            or context.primary_reflection_tension is not None
            or context.reflection_writeback_applied
        ):
            if SectionId.REFLECTION_HOOK not in base:
                base.append(SectionId.REFLECTION_HOOK)

        # Boundary disclaimer if required.
        if assembly is not None and (
            assembly.refer_out_required
            or assembly.clarification_required
            or assembly.citation_mode == "required"
        ):
            if SectionId.BOUNDARY_DISCLAIMER not in base:
                base.append(SectionId.BOUNDARY_DISCLAIMER)

        # Vitals: when slow-scale PE has crossed the proactive threshold the
        # lifeform has been "missing" the user. Surface a continuity note so
        # the response acknowledges that gap rather than acting as if no
        # time has elapsed. We do not override the kernel's intent — only
        # add a section, never replace one.
        if vitals is not None and vitals.above_proactive_threshold:
            if SectionId.CONTINUITY_NOTE not in base:
                base.append(SectionId.CONTINUITY_NOTE)

        return base

    def _apply_participation_hint(
        self,
        *,
        sections: list[SectionId],
        participation_hint: ParticipationHint | None,
    ) -> tuple[list[SectionId], tuple[str, ...]]:
        """Drop sections the regime's participation hint marks SILENT.

        Mapping:

        * ``panorama_level == SILENT`` \u2014 drop ``CLARIFICATION``
          (probing the problem space is out of scope for this turn)
        * ``method_level == SILENT`` \u2014 drop ``REGIME_FRAME`` (no
          explicit regime-pose preamble needed)
        * ``task_level == SILENT`` \u2014 drop ``NEXT_STEP`` and
          ``OPEN_LOOP_HANDOFF`` (no task-progress pressure)

        The planner NEVER drops everything; if all dropping rules
        would leave the plan empty, the filter is skipped entirely
        and the rationale records that the hint was over-applied.
        This keeps pathological hint values from producing an
        unrenderable plan.
        """
        if participation_hint is None:
            return sections, ()
        drops: set[SectionId] = set()
        rationale: list[str] = []
        if participation_hint.panorama_level is ParticipationLevel.SILENT:
            drops.add(SectionId.CLARIFICATION)
            rationale.append("hint_dropped=clarification(panorama=silent)")
        if participation_hint.method_level is ParticipationLevel.SILENT:
            drops.add(SectionId.REGIME_FRAME)
            rationale.append("hint_dropped=regime_frame(method=silent)")
        if participation_hint.task_level is ParticipationLevel.SILENT:
            drops.update({SectionId.NEXT_STEP, SectionId.OPEN_LOOP_HANDOFF})
            rationale.append(
                "hint_dropped=next_step,open_loop_handoff(task=silent)"
            )
        if not drops:
            return sections, ()
        filtered = [s for s in sections if s not in drops]
        if not filtered:
            # Refuse to ship an empty plan; fall back to the unfiltered
            # base and record the over-application.
            return sections, (
                "hint_overapplied_skipped:"
                + ";".join(rationale),
            )
        return filtered, tuple(rationale)

    # ------------------------------------------------------------------
    # Gap 9 slice 2c: continuous-feature interlocutor-state modulation
    # ------------------------------------------------------------------
    # All thresholds below are calibrated against the InterlocutorState
    # axis ranges (``[0, 1]`` for most, ``[-1, 1]`` for ``trust_signal``).
    # These are NOT keyword thresholds \u2014 they're continuous-feature
    # cut-offs analogous to the participation_hint thresholds in
    # ``hint_readout``. The lattice is intentionally conservative
    # (only act on clearly-non-neutral readouts) so cold-start and
    # near-neutral states stay backwards-compatible with the
    # pre-slice-2c planner output.
    _IL_MIN_CONFIDENCE: ClassVar[float] = 0.30
    _IL_EMOTIONAL_HIGH: ClassVar[float] = 0.55
    _IL_RESISTANCE_HIGH: ClassVar[float] = 0.50
    _IL_RAPPORT_LOW: ClassVar[float] = 0.40
    _IL_PACE_HIGH: ClassVar[float] = 0.65
    _IL_DIRECTNESS_LOW: ClassVar[float] = 0.40
    _IL_TRUST_NEGATIVE: ClassVar[float] = -0.10
    _IL_ENGAGEMENT_FLOOR: ClassVar[float] = 0.30

    def _apply_interlocutor_state(
        self,
        *,
        sections: list[SectionId],
        interlocutor_state: InterlocutorState | None,
    ) -> tuple[list[SectionId], tuple[str, ...]]:
        """Modulate sections based on the 12-axis interlocutor readout.

        Returns ``(possibly_modified_sections, rationale_tags)``.
        Conservative behaviour:

        - never replaces an existing intent's section ordering
          wholesale; only adds / drops at well-defined edges;
        - refuses to ship an empty plan: if the drop set would
          empty the section list, the original list is restored
          and the rationale records the over-application;
        - reads ``readout_confidence`` first \u2014 cold-start sessions
          (no turn has produced kernel snapshots yet) get the
          historical no-modulation behaviour.

        Each decision rule is keyed to a single InterlocutorState
        axis, so future calibration is "tune one threshold". No
        rule reads user text or assembly text \u2014 the input is
        purely the typed readout.
        """
        if (
            interlocutor_state is None
            or interlocutor_state.readout_confidence < self._IL_MIN_CONFIDENCE
        ):
            return sections, ()
        s = interlocutor_state
        rationale: list[str] = []
        add: list[SectionId] = []
        drop: set[SectionId] = set()

        # Acknowledge pressure: any of (emotional weight high,
        # resistance high, trust dropped) merits an explicit
        # "I hear you" section at the front. We only ADD if the
        # section is not already there.
        ack_reasons: list[str] = []
        if s.emotional_weight >= self._IL_EMOTIONAL_HIGH:
            ack_reasons.append(f"emo={s.emotional_weight:.2f}")
        if s.resistance_level >= self._IL_RESISTANCE_HIGH:
            ack_reasons.append(f"res={s.resistance_level:.2f}")
        if s.trust_signal <= self._IL_TRUST_NEGATIVE:
            ack_reasons.append(f"trust={s.trust_signal:+.2f}")
        if ack_reasons and SectionId.ACKNOWLEDGE_PRESSURE not in sections:
            add.append(SectionId.ACKNOWLEDGE_PRESSURE)
            rationale.append("il_add=acknowledge_pressure(" + ",".join(ack_reasons) + ")")

        # Continuity note: cold rapport + non-zero engagement.
        # If the user is cold AND disengaged, leave them alone
        # (don't force-warm an exit).
        if (
            s.rapport_warmth <= self._IL_RAPPORT_LOW
            and s.engagement_intensity >= self._IL_ENGAGEMENT_FLOOR
            and SectionId.CONTINUITY_NOTE not in sections
        ):
            add.append(SectionId.CONTINUITY_NOTE)
            rationale.append(
                f"il_add=continuity_note(warmth={s.rapport_warmth:.2f},"
                f"engagement={s.engagement_intensity:.2f})"
            )

        # Drop probing / clarifying when emotional or indirect.
        # Same rule fires twice intentionally: emotional users
        # don't want to be quizzed; indirect users don't want
        # literal probes either.
        if (
            s.emotional_weight >= self._IL_EMOTIONAL_HIGH
            or s.directness <= self._IL_DIRECTNESS_LOW
        ):
            drop.add(SectionId.CLARIFICATION)
            rationale.append(
                f"il_drop=clarification(emo={s.emotional_weight:.2f},"
                f"dir={s.directness:.2f})"
            )

        # High pace pressure trims meta sections that slow the turn.
        if s.pace_pressure >= self._IL_PACE_HIGH:
            drop.update({SectionId.REFLECTION_HOOK, SectionId.OPEN_LOOP_HANDOFF})
            rationale.append(f"il_drop=meta(pace={s.pace_pressure:.2f})")

        if not add and not drop:
            return sections, ()

        new_sections = list(sections)
        for section in add:
            if section is SectionId.ACKNOWLEDGE_PRESSURE:
                new_sections.insert(0, section)
            else:
                new_sections.append(section)
        new_sections = [section for section in new_sections if section not in drop]

        if not new_sections:
            # Refuse to ship an empty plan; drop our modulation
            # entirely and record over-application.
            return sections, (
                "il_overapplied_skipped:" + ";".join(rationale),
            )
        return new_sections, tuple(rationale)

    def _build_section_budget(
        self,
        *,
        sections: list[SectionId],
        intent: TurnIntent,
    ) -> dict[SectionId, int]:
        # Approximate sentence budget per section. Keeps replies bounded and
        # gives the renderer a knob.
        default = 1
        budget: dict[SectionId, int] = {section: default for section in sections}
        if intent is TurnIntent.SUPPORT_FIRST and SectionId.ACKNOWLEDGE_PRESSURE in budget:
            budget[SectionId.ACKNOWLEDGE_PRESSURE] = 2
        if intent is TurnIntent.CLARIFY_FIRST and SectionId.CLARIFICATION in budget:
            budget[SectionId.CLARIFICATION] = 1
        if SectionId.NEXT_STEP in budget:
            budget[SectionId.NEXT_STEP] = max(budget[SectionId.NEXT_STEP], 1)
        return budget

    def _pick_question_budget(
        self,
        *,
        intent: TurnIntent,
        assembly: ResponseAssemblySnapshot | None,
        interlocutor_state: InterlocutorState | None = None,
    ) -> int:
        if assembly is not None:
            sp = assembly.speech_plan
            if sp is not None and isinstance(sp.question_budget, int):
                base = max(0, sp.question_budget)
            elif isinstance(assembly.max_questions, int):
                base = max(0, min(assembly.max_questions, 2))
            else:
                base = _DEFAULT_QUESTION_BUDGET_BY_INTENT.get(intent, 0)
        else:
            base = _DEFAULT_QUESTION_BUDGET_BY_INTENT.get(intent, 0)

        # Interlocutor-state cap: we never RAISE the kernel's budget;
        # we only LOWER it when the readout says questions would be
        # mis-timed. This keeps the kernel as the upper bound and
        # the planner as a downstream tone moderator (R8: the
        # kernel still owns the speech_plan; the planner is a
        # consumer).
        if (
            interlocutor_state is not None
            and interlocutor_state.readout_confidence >= self._IL_MIN_CONFIDENCE
        ):
            s = interlocutor_state
            if (
                s.emotional_weight >= self._IL_EMOTIONAL_HIGH
                or s.pace_pressure >= self._IL_PACE_HIGH
                or s.directness <= self._IL_DIRECTNESS_LOW
            ):
                base = 0
        return base

    def _build_rationale_tags(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot | None,
        intent: TurnIntent,
        vitals: VitalsSnapshot | None = None,
        participation_hint: ParticipationHint | None = None,
        hint_rationale: tuple[str, ...] = (),
        interlocutor_state: InterlocutorState | None = None,
        il_rationale: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        tags: list[str] = [f"intent={intent.value}"]
        regime_id = (
            (assembly.regime_id if assembly is not None else None)
            or context.regime_id
            or "none"
        )
        tags.append(f"regime={regime_id}")
        if context.regime_switched:
            tags.append("regime_switched")
        if context.temporal_is_switching:
            tags.append("temporal_switching")
        if assembly is not None and assembly.knowledge_hit_count:
            tags.append(f"knowledge_hits={assembly.knowledge_hit_count}")
        if assembly is not None and assembly.case_hit_count:
            tags.append(f"case_hits={assembly.case_hit_count}")
        if assembly is not None and assembly.playbook_rule_count:
            tags.append(f"playbook_rules={assembly.playbook_rule_count}")
        if vitals is not None and vitals.above_proactive_threshold:
            out_of_band = ",".join(
                d.name for d in vitals.drive_levels if d.out_of_band
            )
            tags.append(f"vitals_pressure={out_of_band}")
        # Gap 8: record participation-hint levels and any filter
        # application so operators can see why sections were dropped.
        if participation_hint is not None:
            tags.append(f"flow_kind={participation_hint.flow_kind.value}")
            tags.append(
                f"participation=panorama:{participation_hint.panorama_level.value},"
                f"method:{participation_hint.method_level.value},"
                f"task:{participation_hint.task_level.value}"
            )
            for rationale in hint_rationale:
                tags.append(rationale)
        # Gap 9 slice 2c: surface the interlocutor-state confidence
        # and rationale so operators can see when / why the planner
        # added or dropped sections in response to user-state.
        if (
            interlocutor_state is not None
            and interlocutor_state.readout_confidence >= self._IL_MIN_CONFIDENCE
        ):
            tags.append(
                f"interlocutor_conf={interlocutor_state.readout_confidence:.2f}"
            )
            for rationale in il_rationale:
                tags.append(rationale)
        return tuple(tags)
