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
from volvence_zero.social_cognition import (
    BeliefAboutOtherSnapshot,
    CommonGroundSnapshot,
    FeelingAboutOtherSnapshot,
    IntentAboutOtherSnapshot,
    PreferenceAboutOtherSnapshot,
)


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

    _REPAIR_ALPHA_MIN_CONFIDENCE: ClassVar[float] = 0.50

    def __init__(self, *, repair_alpha_enabled: bool = False) -> None:
        self._repair_alpha_enabled = bool(repair_alpha_enabled)

    @property
    def repair_alpha_enabled(self) -> bool:
        return self._repair_alpha_enabled

    def plan(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot | None,
        vitals: VitalsSnapshot | None = None,
        participation_hint: ParticipationHint | None = None,
        interlocutor_state: InterlocutorState | None = None,
        feeling_snapshot: FeelingAboutOtherSnapshot | None = None,
        common_ground_snapshot: CommonGroundSnapshot | None = None,
        belief_snapshot: BeliefAboutOtherSnapshot | None = None,
        intent_snapshot: IntentAboutOtherSnapshot | None = None,
        preference_snapshot: PreferenceAboutOtherSnapshot | None = None,
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

        ``feeling_snapshot`` (Phase 1 W1.D EQ-owner uplift) is the
        typed Theory-of-Mind FEELING readout for the active
        interlocutor. When present and the records carry a typed
        signal (records non-empty AND max-record confidence above the
        owner floor) the planner emits a ``feeling=observed(...)``
        rationale tag and may add ``ACKNOWLEDGE_PRESSURE`` when the
        FEELING owner reports a high control_signal but the 12-axis
        zones did not fire (i.e. typed feeling evidence outside the
        zone-bool view). NEVER via keyword matches; only the typed
        snapshot fields are read.
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
        sections, feeling_rationale = self._apply_feeling_snapshot(
            sections=sections, feeling_snapshot=feeling_snapshot
        )
        sections, common_ground_rationale = self._apply_common_ground_snapshot(
            sections=sections, common_ground_snapshot=common_ground_snapshot
        )
        tom_rationale = self._tom_rationale_tags(
            belief_snapshot=belief_snapshot,
            intent_snapshot=intent_snapshot,
            preference_snapshot=preference_snapshot,
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
            feeling_rationale=feeling_rationale,
            common_ground_rationale=common_ground_rationale,
            tom_rationale=tom_rationale,
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
        if self._repair_advisory_active(context):
            return TurnIntent.REPAIR_FIRST
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

        # No assembly available — fall back to a neutral intent rather
        # than hardcoding ``regime_id == "X"`` => intent (R14: regime is
        # not a prompt label; first-principles: do not encode by string
        # comparison what the system should learn). When upstream owners
        # have not published an assembly snapshot yet, the planner stays
        # minimal and lets downstream rendering carry the slack.
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
        if intent is TurnIntent.REPAIR_FIRST:
            return [
                SectionId.ACKNOWLEDGE_PRESSURE,
                SectionId.REGIME_FRAME,
                SectionId.OPEN_LOOP_HANDOFF,
                SectionId.NEXT_STEP,
            ]

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
    # Wave 2 SSOT cleanup: interlocutor-state modulation reads typed
    # zone booleans published by the InterlocutorStateModule owner
    # rather than re-applying numeric thresholds. The numeric
    # thresholds live ONCE in ``volvence_zero.interlocutor.contracts``
    # (``InterlocutorThresholds``) and the zone classification is
    # done ONCE in ``compute_zones``.
    # ------------------------------------------------------------------

    def _apply_interlocutor_state(
        self,
        *,
        sections: list[SectionId],
        interlocutor_state: InterlocutorState | None,
    ) -> tuple[list[SectionId], tuple[str, ...]]:
        """Modulate sections based on the typed zones in the readout.

        Returns ``(possibly_modified_sections, rationale_tags)``.
        Conservative behaviour:

        - never replaces an existing intent's section ordering
          wholesale; only adds / drops at well-defined edges;
        - refuses to ship an empty plan: if the drop set would
          empty the section list, the original list is restored
          and the rationale records the over-application;
        - cold-start (low confidence) sessions get all-False zones
          and produce a no-op; the historical readout_confidence
          gate is implicit in :func:`compute_zones`.

        Reads zone bools (``acknowledge_pressure_zone`` etc.); the
        underlying numeric thresholds are owned by the snapshot
        producer and are NOT re-applied here.
        """
        if interlocutor_state is None:
            return sections, ()
        s = interlocutor_state
        # When confidence is below the floor, ``compute_zones`` has
        # set every zone to ``False``. Short-circuit to avoid emitting
        # a confidence rationale entry when no zone fired.
        if not (
            s.acknowledge_pressure_zone
            or s.cold_rapport_zone
            or s.emotional_high_zone
            or s.low_directness_zone
            or s.pace_pressure_zone
        ):
            return sections, ()
        rationale: list[str] = []
        add: list[SectionId] = []
        drop: set[SectionId] = set()

        # Acknowledge pressure: composite zone fires when any of
        # (emotional weight high, resistance high, trust negative)
        # is true. Components are exposed individually for the
        # rationale tag so operators can see WHY it fired.
        if s.acknowledge_pressure_zone:
            ack_reasons: list[str] = []
            if s.emotional_high_zone:
                ack_reasons.append(f"emo={s.emotional_weight:.2f}")
            if s.resistance_high_zone:
                ack_reasons.append(f"res={s.resistance_level:.2f}")
            if s.trust_negative_zone:
                ack_reasons.append(f"trust={s.trust_signal:+.2f}")
            if ack_reasons and SectionId.ACKNOWLEDGE_PRESSURE not in sections:
                add.append(SectionId.ACKNOWLEDGE_PRESSURE)
                rationale.append(
                    "il_add=acknowledge_pressure(" + ",".join(ack_reasons) + ")"
                )

        # Continuity note: cold rapport + engaged. Cold + disengaged
        # is intentionally a no-op (don't force-warm an exit).
        if (
            s.cold_rapport_zone
            and SectionId.CONTINUITY_NOTE not in sections
        ):
            add.append(SectionId.CONTINUITY_NOTE)
            rationale.append(
                f"il_add=continuity_note(warmth={s.rapport_warmth:.2f},"
                f"engagement={s.engagement_intensity:.2f})"
            )

        # Drop probing / clarifying when emotional or indirect.
        if s.emotional_high_zone or s.low_directness_zone:
            drop.add(SectionId.CLARIFICATION)
            rationale.append(
                f"il_drop=clarification(emo={s.emotional_weight:.2f},"
                f"dir={s.directness:.2f})"
            )

        # High pace pressure trims meta sections that slow the turn.
        if s.pace_pressure_zone:
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
            return sections, (
                "il_overapplied_skipped:" + ";".join(rationale),
            )
        return new_sections, tuple(rationale)

    # ------------------------------------------------------------------
    # Phase 1 W1.D EQ-owner uplift: FEELING-about-other modulation.
    # Reads the typed Theory-of-Mind FEELING snapshot's records and
    # snapshot-level control_signal; never inspects user text.
    # ------------------------------------------------------------------

    _FEELING_RECORDS_MIN_CONFIDENCE: ClassVar[float] = 0.50
    _FEELING_ACK_PRESSURE_CONTROL_SIGNAL: ClassVar[float] = 0.40

    def _apply_feeling_snapshot(
        self,
        *,
        sections: list[SectionId],
        feeling_snapshot: FeelingAboutOtherSnapshot | None,
    ) -> tuple[list[SectionId], tuple[str, ...]]:
        """Modulate sections based on the typed FEELING snapshot.

        Returns ``(possibly_modified_sections, rationale_tags)``.
        Conservative behaviour, mirroring ``_apply_interlocutor_state``:

        * SHADOW or DISABLED wiring on the kernel side surfaces here as
          ``feeling_snapshot=None`` (lifeform provider returns ``None``
          when the slot is not in ``active_snapshots``); the planner
          treats that as a no-op.
        * Empty records OR all records below the typed confidence floor
          (the ToM owner's own ``min_proposal_confidence``, mirrored
          here as ``_FEELING_RECORDS_MIN_CONFIDENCE``) is also a no-op.
        * When non-empty AND control_signal is high enough, ADD
          ``ACKNOWLEDGE_PRESSURE`` so the planner reflects "we have a
          typed FEELING reading on the user this turn" even if the
          12-axis zone bools did not fire. We never DROP a section on
          feeling alone; the InterlocutorState owner remains the
          dominant section gate.
        """
        if feeling_snapshot is None:
            return sections, ()
        records = feeling_snapshot.records
        if not records:
            return sections, ()
        max_confidence = max(record.confidence for record in records)
        if max_confidence < self._FEELING_RECORDS_MIN_CONFIDENCE:
            return sections, ()
        control_signal = float(feeling_snapshot.control_signal)
        rationale: list[str] = [
            (
                f"feeling=observed(count={len(records)},"
                f"max_conf={max_confidence:.2f},sig={control_signal:.2f})"
            )
        ]
        new_sections = list(sections)
        if (
            control_signal >= self._FEELING_ACK_PRESSURE_CONTROL_SIGNAL
            and SectionId.ACKNOWLEDGE_PRESSURE not in new_sections
        ):
            new_sections.insert(0, SectionId.ACKNOWLEDGE_PRESSURE)
            rationale.append(
                "feeling_add=acknowledge_pressure("
                f"sig={control_signal:.2f})"
            )
        return new_sections, tuple(rationale)

    # ------------------------------------------------------------------
    # Phase 1 W1.F EQ-owner uplift: COMMON-GROUND modulation. Reads
    # the typed dyad atom set from the published CommonGroundSnapshot;
    # never inspects user text.
    # ------------------------------------------------------------------

    _COMMON_GROUND_ATOM_MIN_CONFIDENCE: ClassVar[float] = 0.50
    _COMMON_GROUND_CONTINUITY_DYAD_FLOOR: ClassVar[int] = 1

    def _apply_common_ground_snapshot(
        self,
        *,
        sections: list[SectionId],
        common_ground_snapshot: CommonGroundSnapshot | None,
    ) -> tuple[list[SectionId], tuple[str, ...]]:
        """Modulate sections based on the typed common-ground snapshot.

        When the ``CommonGroundModule`` reports at least one dyad atom
        whose ``confidence`` is above the typed floor, the planner
        emits a typed ``common_ground=observed(dyads=N,max_conf=X)``
        rationale tag and ADDs ``CONTINUITY_NOTE`` to the section list
        if it is not already present, marking that the response should
        thread shared dyad context (e.g. "as we mentioned earlier"
        framing). The planner never DROPS a section on common-ground
        evidence alone, and all signals come from typed atom fields;
        no user text is inspected.
        """
        if common_ground_snapshot is None:
            return sections, ()
        atoms = common_ground_snapshot.dyad_atoms
        if not atoms:
            return sections, ()
        confident_atoms = tuple(
            atom
            for atom in atoms
            if atom.confidence >= self._COMMON_GROUND_ATOM_MIN_CONFIDENCE
        )
        if len(confident_atoms) < self._COMMON_GROUND_CONTINUITY_DYAD_FLOOR:
            return sections, ()
        max_confidence = max(atom.confidence for atom in confident_atoms)
        rationale: list[str] = [
            (
                f"common_ground=observed(dyads={len(confident_atoms)},"
                f"max_conf={max_confidence:.2f})"
            )
        ]
        new_sections = list(sections)
        if SectionId.CONTINUITY_NOTE not in new_sections:
            new_sections.append(SectionId.CONTINUITY_NOTE)
            rationale.append(
                "common_ground_add=continuity_note("
                f"dyads={len(confident_atoms)})"
            )
        return new_sections, tuple(rationale)

    # ------------------------------------------------------------------
    # Phase 2 W2.A EQ-owner uplift: typed Theory-of-Mind rationale tags
    # for BELIEF / INTENT / PREFERENCE about-other owners. These three
    # snapshots influence FRAMING / EXPECTATION / STYLE downstream
    # without re-deriving sections (they are observation-only signals
    # the planner surfaces for evaluation / reflection / human audit).
    # Section additions remain governed by FEELING + InterlocutorState
    # so promotion stays minimally invasive on the section graph.
    # ------------------------------------------------------------------

    _TOM_RECORDS_MIN_CONFIDENCE: ClassVar[float] = 0.50

    def _tom_rationale_tags(
        self,
        *,
        belief_snapshot: BeliefAboutOtherSnapshot | None,
        intent_snapshot: IntentAboutOtherSnapshot | None,
        preference_snapshot: PreferenceAboutOtherSnapshot | None,
    ) -> tuple[str, ...]:
        """Surface typed observation tags for the three remaining
        ToM about-other owners.

        Each tag carries the record count, max confidence, and snapshot
        control_signal. Empty / low-confidence snapshots are no-ops.
        """
        tags: list[str] = []
        for prefix, snapshot in (
            ("framing=belief_observed", belief_snapshot),
            ("intent=expectation_observed", intent_snapshot),
            ("preference=style_observed", preference_snapshot),
        ):
            tag = self._tom_rationale_for(prefix=prefix, snapshot=snapshot)
            if tag is not None:
                tags.append(tag)
        return tuple(tags)

    def _tom_rationale_for(
        self,
        *,
        prefix: str,
        snapshot: BeliefAboutOtherSnapshot
        | IntentAboutOtherSnapshot
        | PreferenceAboutOtherSnapshot
        | None,
    ) -> str | None:
        """Build the rationale tag for one ToM about-other snapshot.

        All three about-other snapshot types share the
        ``records`` / ``control_signal`` shape (verified by the
        ``social_cognition`` contract dataclasses). We type-narrow on
        ``None`` and otherwise read fields directly; no ``getattr``
        defensive defaults to avoid masking schema drift.
        """
        if snapshot is None:
            return None
        if not snapshot.records:
            return None
        max_confidence = max(record.confidence for record in snapshot.records)
        if max_confidence < self._TOM_RECORDS_MIN_CONFIDENCE:
            return None
        control_signal = float(snapshot.control_signal)
        return (
            f"{prefix}(count={len(snapshot.records)},"
            f"max_conf={max_confidence:.2f},sig={control_signal:.2f})"
        )

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
        if intent is TurnIntent.REPAIR_FIRST:
            return 0
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
        # mis-timed. Reads typed zone bools (W2 SSOT), not raw
        # numeric thresholds.
        if interlocutor_state is not None:
            s = interlocutor_state
            if (
                s.emotional_high_zone
                or s.pace_pressure_zone
                or s.low_directness_zone
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
        feeling_rationale: tuple[str, ...] = (),
        common_ground_rationale: tuple[str, ...] = (),
        tom_rationale: tuple[str, ...] = (),
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
        repair_advisory = context.repair_advisory
        if self._repair_advisory_active(context) and repair_advisory is not None:
            tags.append(f"repair_alpha={repair_advisory.rupture_kind}")
            tags.append(f"repair_confidence={repair_advisory.confidence:.2f}")
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
        # Confidence floor is enforced inside ``compute_zones``; here
        # we surface the tag only when at least one zone fired (i.e.
        # we actually modulated something).
        if interlocutor_state is not None and (
            interlocutor_state.acknowledge_pressure_zone
            or interlocutor_state.cold_rapport_zone
            or interlocutor_state.emotional_high_zone
            or interlocutor_state.low_directness_zone
            or interlocutor_state.pace_pressure_zone
        ):
            tags.append(
                f"interlocutor_conf={interlocutor_state.readout_confidence:.2f}"
            )
            for rationale in il_rationale:
                tags.append(rationale)
        # Phase 1 W1.D EQ-owner uplift: surface FEELING-about-other
        # rationale tags emitted by ``_apply_feeling_snapshot``. The
        # tags are typed records-derived signals (count / max
        # confidence / control_signal); no user text is read.
        for rationale in feeling_rationale:
            tags.append(rationale)
        # Phase 1 W1.F EQ-owner uplift: surface common-ground rationale
        # tags emitted by ``_apply_common_ground_snapshot``. The tags
        # are typed atom-derived signals (dyad count, max confidence);
        # no user text is read.
        for rationale in common_ground_rationale:
            tags.append(rationale)
        # Phase 2 W2.A EQ-owner uplift: surface typed observation tags
        # for BELIEF / INTENT / PREFERENCE about-other ToM snapshots.
        # These are observation-only audit signals; they do NOT modify
        # sections (FEELING + InterlocutorState remain the dominant
        # section gates).
        for rationale in tom_rationale:
            tags.append(rationale)
        return tuple(tags)

    def _repair_advisory_active(self, context: ResponseContext) -> bool:
        advisory = context.repair_advisory
        return (
            self._repair_alpha_enabled
            and advisory is not None
            and advisory.confidence >= self._REPAIR_ALPHA_MIN_CONFIDENCE
        )
