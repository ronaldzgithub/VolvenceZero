"""GroundedResponseSynthesizer — the lifeform's voice.

Subclasses the kernel ``ResponseSynthesizer`` ABC. Composes a ``PromptPlan``
from the planner, then renders. The renderer is rule-based by default and
deterministic for replay; subclass and override ``_render`` to plug in an
LLM grounded by the same plan.

Design notes:

* The kernel's ``synthesize`` signature only exposes ``context`` and
  ``assembly``. We **do not** reach into kernel internals or ``BrainSession``
  to grab additional state — staying inside the contract is the whole point
  of this layer (R8).

* The base ``ResponseSynthesizer`` already does extensive heuristic
  templating; we delegate to it for fallback paths (REFER-OUT, judgment
  process, missing assembly) and only build our own structured renderer on
  top.

* The plan is exposed in ``AgentResponse.rationale`` as a structured tail
  so the lifeform-evolution layer can audit which plan drove each turn.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional

from lifeform_core import VitalsSnapshot
from volvence_zero.agent.response import (
    AgentResponse,
    ResponseContext,
    ResponseSynthesizer,
)
from volvence_zero.application.runtime import ResponseAssemblySnapshot
from volvence_zero.interlocutor import InterlocutorState
from volvence_zero.social_cognition import (
    BeliefAboutOtherSnapshot,
    CommonGroundSnapshot,
    FeelingAboutOtherSnapshot,
    IntentAboutOtherSnapshot,
    PreferenceAboutOtherSnapshot,
)

from lifeform_expression.prompt_planner import (
    PromptPlan,
    PromptPlanner,
    SectionId,
    TurnIntent,
)
from lifeform_expression.reflection_hints import (
    reflection_lesson_hint,
    reflection_tension_hint,
)


VitalsSnapshotProvider = Callable[[], VitalsSnapshot | None]
InterlocutorStateProvider = Callable[[], InterlocutorState | None]
FeelingAboutOtherProvider = Callable[[], FeelingAboutOtherSnapshot | None]
CommonGroundSnapshotProvider = Callable[[], CommonGroundSnapshot | None]
BeliefAboutOtherProvider = Callable[[], BeliefAboutOtherSnapshot | None]
IntentAboutOtherProvider = Callable[[], IntentAboutOtherSnapshot | None]
PreferenceAboutOtherProvider = Callable[[], PreferenceAboutOtherSnapshot | None]


class GroundedResponseSynthesizer(ResponseSynthesizer):
    """ResponseSynthesizer driven by a structured PromptPlan.

    Optional ``vitals_snapshot_provider`` is a zero-arg callable returning
    the current ``VitalsSnapshot`` (or ``None``) for the calling session.
    The lifeform layer wires a per-session provider via closure so that
    drives produced by ``VitalsModule`` reach the planner \u2014 which adds a
    ``CONTINUITY_NOTE`` section and a ``vitals_pressure=...`` rationale
    tag whenever the slow-scale PE is above threshold. Without a provider
    the synthesizer behaves exactly as before, so this is fully
    backwards-compatible.
    """

    def __init__(
        self,
        *,
        planner: PromptPlanner | None = None,
        vitals_snapshot_provider: VitalsSnapshotProvider | None = None,
        interlocutor_state_provider: InterlocutorStateProvider | None = None,
        feeling_about_other_provider: FeelingAboutOtherProvider | None = None,
        common_ground_snapshot_provider: CommonGroundSnapshotProvider | None = None,
        belief_about_other_provider: BeliefAboutOtherProvider | None = None,
        intent_about_other_provider: IntentAboutOtherProvider | None = None,
        preference_about_other_provider: PreferenceAboutOtherProvider | None = None,
        figure_bundle: object | None = None,
    ) -> None:
        self._planner = planner or PromptPlanner()
        self._vitals_snapshot_provider = vitals_snapshot_provider
        self._interlocutor_state_provider = interlocutor_state_provider
        self._feeling_about_other_provider = feeling_about_other_provider
        self._common_ground_snapshot_provider = common_ground_snapshot_provider
        self._belief_about_other_provider = belief_about_other_provider
        self._intent_about_other_provider = intent_about_other_provider
        self._preference_about_other_provider = preference_about_other_provider
        self._figure_bundle = figure_bundle

    @property
    def planner(self) -> PromptPlanner:
        return self._planner

    @property
    def vitals_snapshot_provider(self) -> VitalsSnapshotProvider | None:
        return self._vitals_snapshot_provider

    @property
    def interlocutor_state_provider(self) -> InterlocutorStateProvider | None:
        return self._interlocutor_state_provider

    @property
    def feeling_about_other_provider(self) -> FeelingAboutOtherProvider | None:
        return self._feeling_about_other_provider

    @property
    def common_ground_snapshot_provider(self) -> CommonGroundSnapshotProvider | None:
        return self._common_ground_snapshot_provider

    @property
    def belief_about_other_provider(self) -> BeliefAboutOtherProvider | None:
        return self._belief_about_other_provider

    @property
    def intent_about_other_provider(self) -> IntentAboutOtherProvider | None:
        return self._intent_about_other_provider

    @property
    def preference_about_other_provider(self) -> PreferenceAboutOtherProvider | None:
        return self._preference_about_other_provider

    @property
    def figure_bundle(self) -> object | None:
        """The optional :class:`FigureArtifactBundle` attached to this synthesizer.

        Returned by reference. The base ``GroundedResponseSynthesizer``
        does not consume the bundle on its own (it renders a rule-
        based response that does not call into the LLM); the bundle
        is preserved here so the lifeform-service / DLaaS-launcher
        adopt path can attach it via :meth:`with_figure_bundle` and
        clones produced by :meth:`_clone` carry it forward without
        the platform layer needing to re-attach.
        """
        return self._figure_bundle

    def with_figure_bundle(
        self, bundle: object | None
    ) -> "GroundedResponseSynthesizer":
        """Return a clone of this synthesizer carrying ``bundle``."""
        clone = self._clone()
        clone._figure_bundle = bundle  # noqa: SLF001 — internal reassignment
        return clone

    def _clone(
        self,
        *,
        vitals: VitalsSnapshotProvider | None = None,
        interlocutor: InterlocutorStateProvider | None = None,
        feeling: FeelingAboutOtherProvider | None = None,
        common_ground: CommonGroundSnapshotProvider | None = None,
        belief: BeliefAboutOtherProvider | None = None,
        intent: IntentAboutOtherProvider | None = None,
        preference: PreferenceAboutOtherProvider | None = None,
        replace_vitals: bool = False,
        replace_interlocutor: bool = False,
        replace_feeling: bool = False,
        replace_common_ground: bool = False,
        replace_belief: bool = False,
        replace_intent: bool = False,
        replace_preference: bool = False,
    ) -> "GroundedResponseSynthesizer":
        """Internal helper that constructs a new synthesizer preserving
        all existing providers except the ones explicitly replaced.

        Centralising the clone construction keeps the seven ``with_*``
        public methods symmetric and adding more providers in future
        waves only touches this method.
        """
        return GroundedResponseSynthesizer(
            planner=self._planner,
            vitals_snapshot_provider=(
                vitals if replace_vitals else self._vitals_snapshot_provider
            ),
            interlocutor_state_provider=(
                interlocutor if replace_interlocutor else self._interlocutor_state_provider
            ),
            feeling_about_other_provider=(
                feeling if replace_feeling else self._feeling_about_other_provider
            ),
            common_ground_snapshot_provider=(
                common_ground
                if replace_common_ground
                else self._common_ground_snapshot_provider
            ),
            belief_about_other_provider=(
                belief if replace_belief else self._belief_about_other_provider
            ),
            intent_about_other_provider=(
                intent if replace_intent else self._intent_about_other_provider
            ),
            preference_about_other_provider=(
                preference
                if replace_preference
                else self._preference_about_other_provider
            ),
            figure_bundle=self._figure_bundle,
        )

    def with_vitals_provider(
        self, provider: VitalsSnapshotProvider | None
    ) -> "GroundedResponseSynthesizer":
        """Return a clone of this synthesizer bound to the given vitals provider.

        The lifeform layer calls this once per ``LifeformSession`` so each
        session has its own synthesizer instance reading its own
        ``VitalsModule``. The original synthesizer (and its planner) are
        kept intact so the Brain-level default does not get mutated.
        Existing interlocutor / feeling / common-ground providers (if
        any) are preserved.
        """
        return self._clone(vitals=provider, replace_vitals=True)

    def with_interlocutor_provider(
        self, provider: InterlocutorStateProvider | None
    ) -> "GroundedResponseSynthesizer":
        """Return a clone bound to the given interlocutor-state provider.

        Mirrors ``with_vitals_provider``: the lifeform layer wires
        a per-session closure to ``LifeformSession.interlocutor_state``
        so the planner sees fresh 12-axis readouts every turn. The
        Brain-level default synthesizer is untouched. Other existing
        providers are preserved on the clone.
        """
        return self._clone(interlocutor=provider, replace_interlocutor=True)

    def with_feeling_about_other_provider(
        self, provider: FeelingAboutOtherProvider | None
    ) -> "GroundedResponseSynthesizer":
        """Return a clone bound to a typed ``feeling_about_other`` provider.

        Phase 1 W1.D of the EQ-owner uplift: the lifeform layer wires
        a per-session closure to ``LifeformSession.feeling_about_other``
        so the planner sees the latest typed Theory-of-Mind FEELING
        readout each turn. The closure returns ``None`` when the
        snapshot is missing (cold-start / SHADOW wiring); the planner
        treats that as a no-op.
        """
        return self._clone(feeling=provider, replace_feeling=True)

    def with_common_ground_provider(
        self, provider: CommonGroundSnapshotProvider | None
    ) -> "GroundedResponseSynthesizer":
        """Return a clone bound to a typed ``common_ground`` provider.

        Phase 1 W1.F of the EQ-owner uplift: the lifeform layer wires
        a per-session closure to ``LifeformSession.common_ground`` so
        the planner sees the typed dyad / group atoms each turn and
        can emit ``common_ground=observed(...)`` rationale tags +
        add a ``CONTINUITY_NOTE`` section when shared dyad context is
        available. ``None`` collapses to a no-op.
        """
        return self._clone(common_ground=provider, replace_common_ground=True)

    def with_belief_about_other_provider(
        self, provider: BeliefAboutOtherProvider | None
    ) -> "GroundedResponseSynthesizer":
        """Phase 2 W2.A: bind a typed ``belief_about_other`` provider.

        Records influence FRAMING (``framing=belief_observed(...)``);
        do not REPLACE the planner's intent.
        """
        return self._clone(belief=provider, replace_belief=True)

    def with_intent_about_other_provider(
        self, provider: IntentAboutOtherProvider | None
    ) -> "GroundedResponseSynthesizer":
        """Phase 2 W2.A: bind a typed ``intent_about_other`` provider.

        Records influence EXPECTATION ALIGNMENT
        (``intent=expectation_observed(...)``).
        """
        return self._clone(intent=provider, replace_intent=True)

    def with_preference_about_other_provider(
        self, provider: PreferenceAboutOtherProvider | None
    ) -> "GroundedResponseSynthesizer":
        """Phase 2 W2.A: bind a typed ``preference_about_other`` provider.

        Records influence STYLE / TONE selection
        (``preference=style_observed(...)``).
        """
        return self._clone(preference=provider, replace_preference=True)

    # Intents we delegate to the base kernel renderer. Judgment-process is
    # rendered locally now: the base template was too repetitive for
    # companion widening transcripts, while the grounded section renderer can
    # still surface the "show your reasoning" posture without collapsing all
    # such turns to one sentence.
    _DELEGATE_TO_BASE: frozenset[TurnIntent] = frozenset(
        {TurnIntent.REFER_OUT, TurnIntent.DIRECT_ANSWER}
    )

    def synthesize(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot | None = None,
    ) -> AgentResponse:
        vitals = self._read_vitals_snapshot()
        interlocutor_state = self._read_interlocutor_state()
        feeling_snapshot = self._read_feeling_about_other()
        common_ground_snapshot = self._read_common_ground_snapshot()
        belief_snapshot = self._read_belief_about_other()
        intent_snapshot = self._read_intent_about_other()
        preference_snapshot = self._read_preference_about_other()
        plan = self._planner.plan(
            context=context,
            assembly=assembly,
            vitals=vitals,
            interlocutor_state=interlocutor_state,
            feeling_snapshot=feeling_snapshot,
            common_ground_snapshot=common_ground_snapshot,
            belief_snapshot=belief_snapshot,
            intent_snapshot=intent_snapshot,
            preference_snapshot=preference_snapshot,
        )

        if plan.intent in self._DELEGATE_TO_BASE or assembly is None:
            base = super().synthesize(context=context, assembly=assembly)
            return _attach_plan_rationale(base, plan)

        section_tags: list[str] = []
        text = self._render(
            context=context,
            assembly=assembly,
            plan=plan,
            vitals=vitals,
            interlocutor_state=interlocutor_state,
            section_tags=section_tags,
        )
        if not text.strip():
            base = super().synthesize(context=context, assembly=assembly)
            return _attach_plan_rationale(base, plan)

        regime_id = assembly.regime_id if assembly is not None else context.regime_id
        regime_name = assembly.regime_name if assembly is not None else context.regime_name
        abstract_action = assembly.abstract_action if assembly is not None else context.abstract_action
        rationale, rationale_parts = _build_rationale(
            regime_name=regime_name,
            regime_id=regime_id,
            abstract_action=abstract_action,
            context=context,
            assembly=assembly,
            plan=plan,
            vitals=vitals,
        )
        merged_tags = _merge_rationale_tags(
            base_tags=(),
            plan=plan,
            extra_parts=tuple(rationale_parts),
            section_tags=tuple(section_tags),
        )
        return AgentResponse(
            text=text,
            regime_id=regime_id,
            abstract_action=abstract_action,
            rationale=rationale,
            rationale_tags=merged_tags,
        )

    def _read_vitals_snapshot(self) -> VitalsSnapshot | None:
        """Pull the current vitals snapshot (None when no provider bound)."""
        if self._vitals_snapshot_provider is None:
            return None
        return self._vitals_snapshot_provider()

    def _read_interlocutor_state(self) -> InterlocutorState | None:
        """Pull the current 12-axis interlocutor readout (None when unbound).

        Returning ``None`` is a normal cold-start signal: the
        planner treats it as "no modulation". Callers are expected
        to inspect ``readout_confidence`` themselves; here we just
        forward whatever the provider hands back without any
        interpretation \u2014 the planner is the policy owner.
        """
        if self._interlocutor_state_provider is None:
            return None
        return self._interlocutor_state_provider()

    def _read_feeling_about_other(self) -> FeelingAboutOtherSnapshot | None:
        """Pull the current FEELING-about-other ToM snapshot (None when unbound).

        Phase 1 W1.D: the lifeform-side provider returns the active
        ``feeling_about_other`` snapshot when ``feeling_about_other``
        is ACTIVE on the kernel turn. SHADOW / DISABLED / cold-start
        all collapse to ``None`` and the planner treats that as "no
        modulation". Records-level interpretation lives in the planner;
        this method only forwards.
        """
        if self._feeling_about_other_provider is None:
            return None
        return self._feeling_about_other_provider()

    def _read_common_ground_snapshot(self) -> CommonGroundSnapshot | None:
        """Pull the current ``common_ground`` snapshot (None when unbound).

        Phase 1 W1.F: the provider returns the active common-ground
        snapshot when the kernel publishes one. SHADOW / DISABLED /
        cold-start collapse to ``None``; the planner treats ``None``
        and an empty atom list identically as "no modulation".
        """
        if self._common_ground_snapshot_provider is None:
            return None
        return self._common_ground_snapshot_provider()

    def _read_belief_about_other(self) -> BeliefAboutOtherSnapshot | None:
        if self._belief_about_other_provider is None:
            return None
        return self._belief_about_other_provider()

    def _read_intent_about_other(self) -> IntentAboutOtherSnapshot | None:
        if self._intent_about_other_provider is None:
            return None
        return self._intent_about_other_provider()

    def _read_preference_about_other(self) -> PreferenceAboutOtherSnapshot | None:
        if self._preference_about_other_provider is None:
            return None
        return self._preference_about_other_provider()

    # ------------------------------------------------------------------
    # Rendering hooks — subclass to swap in an LLM
    # ------------------------------------------------------------------

    def _render(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot,
        plan: PromptPlan,
        vitals: VitalsSnapshot | None = None,
        interlocutor_state: InterlocutorState | None = None,
        section_tags: list[str] | None = None,
    ) -> str:
        sentences: list[str] = []
        for section in plan.sections:
            rendered = self._render_section(
                section=section,
                context=context,
                assembly=assembly,
                plan=plan,
                vitals=vitals,
                interlocutor_state=interlocutor_state,
                section_tags=section_tags,
            )
            if rendered:
                sentences.append(rendered)
        return " ".join(_dedupe_sentences(sentences)).strip()

    def _render_section(
        self,
        *,
        section: SectionId,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot,
        plan: PromptPlan,
        vitals: VitalsSnapshot | None = None,
        interlocutor_state: InterlocutorState | None = None,
        section_tags: list[str] | None = None,
    ) -> str:
        repair_alpha_active = _plan_uses_repair_alpha(plan)
        if section is SectionId.ACKNOWLEDGE_PRESSURE:
            text, variant = self._render_acknowledge(
                context=context,
                assembly=assembly,
                repair_alpha_active=repair_alpha_active,
                interlocutor_state=interlocutor_state,
            )
            if section_tags is not None and variant:
                section_tags.append(f"acknowledge_section={variant}")
            return text
        if section is SectionId.REGIME_FRAME:
            text, variant = self._render_regime_frame(
                context=context,
                assembly=assembly,
                repair_alpha_active=repair_alpha_active,
                interlocutor_state=interlocutor_state,
            )
            if section_tags is not None and variant:
                section_tags.append(f"regime_frame_section={variant}")
            return text
        if section is SectionId.OPEN_LOOP_HANDOFF:
            text, variant = self._render_open_loop(
                context=context,
                assembly=assembly,
                repair_alpha_active=repair_alpha_active,
            )
            if section_tags is not None and variant:
                section_tags.append(f"open_loop_section={variant}")
            return text
        if section is SectionId.CLARIFICATION:
            return self._render_clarification(context=context, assembly=assembly, plan=plan)
        if section is SectionId.NEXT_STEP:
            text, variant = self._render_next_step(
                context=context,
                assembly=assembly,
                repair_alpha_active=repair_alpha_active,
                interlocutor_state=interlocutor_state,
            )
            if section_tags is not None and variant:
                section_tags.append(f"next_step_section={variant}")
            return text
        if section is SectionId.BOUNDARY_DISCLAIMER:
            return self._render_boundary(context=context, assembly=assembly)
        if section is SectionId.REFLECTION_HOOK:
            return self._render_reflection(context=context)
        if section is SectionId.CONTINUITY_NOTE:
            return self._render_continuity(
                context=context, assembly=assembly, vitals=vitals
            )
        return ""

    # ------------------------------------------------------------------
    # Per-section text — keep deterministic so replay tests are stable
    # ------------------------------------------------------------------

    @staticmethod
    def _render_acknowledge(
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot,
        repair_alpha_active: bool = False,
        interlocutor_state: InterlocutorState | None = None,
    ) -> tuple[str, str]:
        # Variant order matters:
        # 1. repair_alpha advisory (cross-regime, takes priority)
        # 2. regime expression_brief.acknowledge_hint (W3 SSOT)
        # 3. interlocutor state zones (W2 SSOT)
        # 4. continuum-target fallback
        # 5. default
        if repair_alpha_active and context.repair_advisory is not None:
            # ``kind_label`` is the rupture_state-owned phrase (W3 SSOT);
            # fall back to the rupture_kind enum value only when running
            # against an older runtime that has not been updated to
            # populate the label field.
            advisory = context.repair_advisory
            label = advisory.kind_label or advisory.rupture_kind.replace("_", " ")
            return (
                f"I hear that this landed as {label}. I am going to pause "
                "the optimizing frame instead of pushing through it."
            ), "repair_alpha"
        brief_hint = context.regime_expression_brief.acknowledge_hint
        text = _ACKNOWLEDGE_TEMPLATES.get(brief_hint, "")
        if text:
            return text, brief_hint
        if _state_indicates_repair(interlocutor_state):
            return (
                "You are right to slow the pace; I do not want to turn you into a project."
            ), "interlocutor_repair"
        if _state_indicates_direct_task(interlocutor_state):
            return (
                "I can keep this practical and bounded without turning it into a full analysis."
            ), "interlocutor_direct_task"
        if _state_indicates_emotional_weight(interlocutor_state):
            return (
                "There is some weight here, so I want to stay close to what you are actually feeling."
            ), "interlocutor_emotional"
        if assembly.continuum_target_position >= 0.7:
            return (
                "I want to acknowledge the pressure here before we narrow anything."
            ), "continuum_high"
        return "I am not going to rush past what you just said.", "default"

    @staticmethod
    def _render_regime_frame(
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot,
        repair_alpha_active: bool = False,
        interlocutor_state: InterlocutorState | None = None,
    ) -> tuple[str, str]:
        if repair_alpha_active and context.repair_advisory is not None:
            return (
                "The repair is the frame now: name what went wrong, lower "
                "pressure, and make the next move reversible."
            ), "repair_alpha"
        # Guided-exploration is the only regime whose frame is
        # interlocutor-state-modulated; we surface variants by zone
        # before falling back to the brief's static hint.
        brief_hint = context.regime_expression_brief.frame_hint
        if brief_hint == "guided_exploration":
            if _state_indicates_repair(interlocutor_state):
                return (
                    "Let's reset the pace and make the next move feel chosen, not imposed."
                ), "guided_exploration_repair"
            if _state_indicates_direct_task(interlocutor_state):
                return (
                    "I will give one concrete step and keep the reasoning visible."
                ), "guided_exploration_direct"
            if _state_indicates_emotional_weight(interlocutor_state):
                return (
                    "We can sort the feeling first, then decide whether a step is needed."
                ), "guided_exploration_emotional"
        text = _FRAME_TEMPLATES.get(brief_hint, "")
        if text:
            return text, brief_hint
        return (
            "I will stay context-aware and keep usefulness and continuity in view."
        ), "default"

    @staticmethod
    def _render_open_loop(
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot,
        repair_alpha_active: bool = False,
    ) -> tuple[str, str]:
        if repair_alpha_active and context.repair_advisory is not None:
            return (
                "I will carry this rupture forward as something to repair, "
                "not as a task thread to solve."
            ), "repair_alpha"
        if assembly.case_hit_count or assembly.playbook_rule_count:
            return (
                "There is at least one thread we left open before; I will "
                "thread it forward instead of restarting from scratch."
            ), "case_or_playbook"
        return (
            "I will keep prior open threads in view rather than dropping them."
        ), "default"

    @staticmethod
    def _render_clarification(
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot,
        plan: PromptPlan,
    ) -> str:
        if plan.question_budget <= 0:
            return "I want to keep this bounded; if anything is missing I will hold off rather than over-commit."
        return "Before we go further, can you say a little more about what specifically would feel useful right now?"

    @staticmethod
    def _render_next_step(
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot,
        repair_alpha_active: bool = False,
        interlocutor_state: InterlocutorState | None = None,
    ) -> tuple[str, str]:
        if repair_alpha_active and context.repair_advisory is not None:
            return (
                "A reversible adjustment: I will stay with what you meant "
                "before offering structure, and you can stop or redirect me "
                "if I start turning it into a workflow again."
            ), "repair_alpha"
        brief_hint = context.regime_expression_brief.next_step_hint
        # Guided-exploration is the only next-step variant that
        # depends on interlocutor zone; surface that branch first.
        if brief_hint == "guided_exploration":
            if _state_indicates_repair(interlocutor_state):
                return (
                    "A good next move is to name what felt off, then choose one gentler thread to continue."
                ), "guided_exploration_repair"
            if _state_indicates_direct_task(interlocutor_state):
                return (
                    "One concrete next step: choose the smallest reversible action, then stop and reassess."
                ), "guided_exploration_direct"
            if _state_indicates_emotional_weight(interlocutor_state):
                return (
                    "Start by naming the heaviest part in one sentence; we can decide after that."
                ), "guided_exploration_emotional"
        text = _NEXT_STEP_TEMPLATES.get(brief_hint, "")
        if text:
            return text, brief_hint
        return (
            "From here we can take one small, concrete step that keeps things moving without forcing the pace."
        ), "default"

    @staticmethod
    def _render_boundary(
        *, context: ResponseContext, assembly: ResponseAssemblySnapshot
    ) -> str:
        if assembly.refer_out_required:
            return (
                "I should also flag that I am staying high-level here; for anything "
                "with material risk, please bring in a qualified professional."
            )
        if assembly.clarification_required:
            return "I will keep this bounded until one missing detail is clarified."
        if assembly.citation_mode == "required":
            return "I am keeping any factual reference clearly sourced and non-definitive."
        if assembly.required_disclaimer_phrases:
            joined = "; ".join(assembly.required_disclaimer_phrases)
            return f"Note: {joined}."
        return ""

    @staticmethod
    def _render_reflection(*, context: ResponseContext) -> str:
        if context.reflection_writeback_applied:
            return (
                f"I am also carrying forward {context.reflection_lesson_count} "
                "reflected lesson(s) from the slow loop."
            )
        if context.primary_reflection_lesson is not None:
            hint = reflection_lesson_hint(context.primary_reflection_lesson)
            if hint:
                return hint
        if context.primary_reflection_tension is not None:
            hint = reflection_tension_hint(context.primary_reflection_tension)
            if hint:
                return hint
        return ""

    @staticmethod
    def _render_continuity(
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot,
        vitals: VitalsSnapshot | None = None,
    ) -> str:
        # When the always-on drive layer is signalling pressure, the
        # continuity note acknowledges the elapsed silence directly rather
        # than the generic "carrying continuity" line. We name the most
        # affected drive so reflection / evaluation can audit which drive
        # actually drove the response shape.
        if vitals is not None and vitals.above_proactive_threshold:
            out_of_band = [d for d in vitals.drive_levels if d.out_of_band]
            if out_of_band:
                worst = max(out_of_band, key=lambda d: d.pe_contribution)
                return (
                    f"I want to acknowledge that some time has passed; my "
                    f"{worst.name.replace('_', ' ')} level dropped while we "
                    f"were quiet, and I am noticing it now rather than acting "
                    f"as if no gap occurred."
                )
        if assembly.prompt_residue_summary:
            return "I am also carrying continuity from our recent thread."
        return "I am keeping continuity in view across our turns."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dedupe_sentences(sentences: Iterable[str]) -> list[str]:
    """Remove duplicates while preserving order — defensive against the case
    where two adjacent sections render similar lines for the same regime.
    """
    seen: set[str] = set()
    out: list[str] = []
    for sentence in sentences:
        normalised = " ".join(sentence.split())
        if not normalised:
            continue
        if normalised in seen:
            continue
        seen.add(normalised)
        out.append(normalised)
    return out


_ACKNOWLEDGE_TEMPLATES: dict[str, str] = {
    "repair_regime": (
        "I want to slow this down so we can stay grounded together. "
        "I am not going to push past what just happened."
    ),
    "emotional_support": (
        "I am hearing weight in this and I want to stay with it before we move."
    ),
    "warmth_first": (
        "I want to honour what you just shared before I move toward anything practical."
    ),
}


_FRAME_TEMPLATES: dict[str, str] = {
    "emotional_support": "I will stay supportive first and not jump into solving.",
    "problem_solving": "I see a structured path here we can walk together.",
    "repair_and_deescalation": (
        "I will keep my responses steady and avoid escalating tone."
    ),
    "acquaintance_building": "I want to keep this warm rather than transactional.",
    "casual_social": "I can stay in steady, low-pressure mode.",
    "guided_exploration": (
        "I would rather explore this with you step by step than guess at one answer."
    ),
}


_NEXT_STEP_TEMPLATES: dict[str, str] = {
    "support_or_repair": (
        "We can stay here together, and when you are ready, name one "
        "small next step that would feel manageable."
    ),
    "problem_solving": (
        "Concretely, the smallest useful next step is to name the constraint that matters most."
    ),
    "guided_exploration": (
        "Pick one of the threads we just surfaced and we can go a little deeper on it."
    ),
    "acquaintance_building": (
        "We can move at whatever pace feels right; there is no rush."
    ),
    "casual_social": "We can keep things easy and continuous from here.",
}


def _state_indicates_repair(state: InterlocutorState | None) -> bool:
    """Wave 2: read the typed ``repair_zone`` bool the owner already
    classified, instead of re-applying numeric thresholds. Zone
    definitions live ONCE in
    ``volvence_zero.interlocutor.contracts.compute_zones``.
    """
    return state is not None and state.repair_zone


def _state_indicates_direct_task(state: InterlocutorState | None) -> bool:
    return state is not None and state.direct_task_zone


def _state_indicates_emotional_weight(state: InterlocutorState | None) -> bool:
    return state is not None and state.emotional_render_zone


def _plan_uses_repair_alpha(plan: PromptPlan) -> bool:
    return any(tag.startswith("repair_alpha=") for tag in plan.rationale_tags)


def _attach_plan_rationale(response: AgentResponse, plan: PromptPlan) -> AgentResponse:
    """Append the plan summary to a response rendered by the base kernel renderer.

    Used on the ``DELEGATE_TO_BASE`` path (refer-out / judgment-process /
    direct-answer) so even those responses carry the planner's rationale
    tags forward \u2014 most importantly ``vitals_pressure=...`` and any
    explicit cross-cutting tags the planner emitted. Without this, vitals
    state would only reach reflection / evaluation when the renderer is
    NOT the kernel's base, which makes the "always-on drive" signal
    silently disappear in roughly half of common turns.
    """
    rationale = response.rationale
    if rationale and not rationale.endswith("."):
        rationale += "."
    plan_tag = (
        f" Plan: intent={plan.intent.value}; "
        f"sections={','.join(s.value for s in plan.sections)}; "
        f"q={plan.question_budget}."
    )
    extra_tags = [
        tag for tag in plan.rationale_tags
        if tag.startswith("vitals_pressure=")
        or tag.startswith("interlocutor_conf=")
        or tag.startswith("repair_alpha=")
        or tag.startswith("repair_confidence=")
        or tag.startswith("il_")
    ]
    extra = (" " + " ".join(extra_tags) + ".") if extra_tags else ""
    merged_tags = _merge_rationale_tags(
        base_tags=response.rationale_tags,
        plan=plan,
        extra_parts=(),
        section_tags=(),
    )
    return AgentResponse(
        text=response.text,
        regime_id=response.regime_id,
        abstract_action=response.abstract_action,
        rationale=(rationale + plan_tag + extra).strip(),
        rationale_tags=merged_tags,
    )


def _merge_rationale_tags(
    *,
    base_tags: tuple[str, ...],
    plan: PromptPlan,
    extra_parts: tuple[str, ...],
    section_tags: tuple[str, ...],
) -> tuple[str, ...]:
    """Merge typed rationale tags from multiple sources, preserving order
    and deduplicating. Order priority: base (kernel) -> extra parts
    (synthesizer rationale_parts) -> plan (planner) -> section variant
    tags (renderer). Always emits a ``plan=intent:sections:q`` summary
    tag so downstream consumers can find the plan signature without
    parsing rationale text.
    """
    merged: list[str] = []
    seen: set[str] = set()

    def _add(tag: str) -> None:
        if not tag:
            return
        if tag in seen:
            return
        seen.add(tag)
        merged.append(tag)

    for tag in base_tags:
        _add(tag)
    for tag in extra_parts:
        _add(tag)
    for tag in plan.rationale_tags:
        _add(tag)
    for tag in section_tags:
        _add(tag)
    _add(
        "plan="
        f"intent:{plan.intent.value};"
        f"sections:{','.join(s.value for s in plan.sections)};"
        f"q:{plan.question_budget}"
    )
    return tuple(merged)


def _build_rationale(
    *,
    regime_name: str,
    regime_id: Optional[str],
    abstract_action: Optional[str],
    context: ResponseContext,
    assembly: ResponseAssemblySnapshot,
    plan: PromptPlan,
    vitals: VitalsSnapshot | None = None,
) -> tuple[str, list[str]]:
    parts: list[str] = [f"regime={regime_id or 'none'}"]
    if abstract_action:
        parts.append(f"temporal={abstract_action}")
    parts.append(f"switch_gate={context.temporal_switch_gate:.2f}")
    parts.append(f"intent={plan.intent.value}")
    parts.append("sections=" + ",".join(s.value for s in plan.sections))
    parts.append(f"q={plan.question_budget}")
    if assembly.knowledge_hit_count:
        parts.append(f"knowledge_hits={assembly.knowledge_hit_count}")
    if assembly.case_hit_count:
        parts.append(f"case_hits={assembly.case_hit_count}")
    if assembly.playbook_rule_count:
        parts.append(f"playbook_rules={assembly.playbook_rule_count}")
    parts.append(f"risk={assembly.risk_band.value}")
    if vitals is not None and vitals.above_proactive_threshold:
        out_of_band = ",".join(
            d.name for d in vitals.drive_levels if d.out_of_band
        )
        if out_of_band:
            parts.append(f"vitals_pressure={out_of_band}")
            parts.append(f"vitals_total_pe={vitals.total_pe:.2f}")
    # Forward Gap 9 slice 2c interlocutor-state tags so reflection /
    # evaluation can audit which user-state axes shaped this turn.
    for tag in plan.rationale_tags:
        if (
            tag.startswith("interlocutor_conf=")
            or tag.startswith("repair_alpha=")
            or tag.startswith("repair_confidence=")
            or tag.startswith("il_")
        ):
            parts.append(tag)
    rationale = (
        f"GroundedResponseSynthesizer from {regime_name}; "
        + ", ".join(parts)
        + "."
    )
    return rationale, parts
