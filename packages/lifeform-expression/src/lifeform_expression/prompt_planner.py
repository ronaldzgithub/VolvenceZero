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

from lifeform_core import VitalsSnapshot
from volvence_zero.agent.response import ResponseContext
from volvence_zero.application.runtime import ResponseAssemblySnapshot


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
    ) -> PromptPlan:
        intent = self._pick_intent(context=context, assembly=assembly)
        sections = self._pick_sections(
            context=context, assembly=assembly, intent=intent, vitals=vitals
        )
        budget = self._build_section_budget(sections=sections, intent=intent)
        question_budget = self._pick_question_budget(intent=intent, assembly=assembly)
        disclaimers = (
            tuple(assembly.required_disclaimer_phrases) if assembly is not None else ()
        )
        rationale_tags = self._build_rationale_tags(
            context=context, assembly=assembly, intent=intent, vitals=vitals
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
    ) -> int:
        if assembly is not None:
            sp = assembly.speech_plan
            if sp is not None and isinstance(sp.question_budget, int):
                return max(0, sp.question_budget)
            if isinstance(assembly.max_questions, int):
                return max(0, min(assembly.max_questions, 2))
        return _DEFAULT_QUESTION_BUDGET_BY_INTENT.get(intent, 0)

    def _build_rationale_tags(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot | None,
        intent: TurnIntent,
        vitals: VitalsSnapshot | None = None,
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
        return tuple(tags)
