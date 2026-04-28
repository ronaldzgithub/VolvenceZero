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

from collections.abc import Iterable
from typing import Optional

from volvence_zero.agent.response import (
    AgentResponse,
    ResponseContext,
    ResponseSynthesizer,
)
from volvence_zero.application.runtime import ResponseAssemblySnapshot

from lifeform_expression.prompt_planner import (
    PromptPlan,
    PromptPlanner,
    SectionId,
    TurnIntent,
)


class GroundedResponseSynthesizer(ResponseSynthesizer):
    """ResponseSynthesizer driven by a structured PromptPlan."""

    def __init__(self, *, planner: PromptPlanner | None = None) -> None:
        self._planner = planner or PromptPlanner()

    @property
    def planner(self) -> PromptPlanner:
        return self._planner

    # Intents we delegate to the base kernel renderer because the base
    # already produces strong, regime-faithful text for them (and tests pin
    # the judgment-process / refer-out shapes). Direct-answer also delegates
    # because there is no useful section structure to add on top of the
    # kernel's regime-tail templates.
    _DELEGATE_TO_BASE: frozenset[TurnIntent] = frozenset(
        {TurnIntent.REFER_OUT, TurnIntent.JUDGMENT_PROCESS, TurnIntent.DIRECT_ANSWER}
    )

    def synthesize(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot | None = None,
    ) -> AgentResponse:
        plan = self._planner.plan(context=context, assembly=assembly)

        if plan.intent in self._DELEGATE_TO_BASE or assembly is None:
            base = super().synthesize(context=context, assembly=assembly)
            return _attach_plan_rationale(base, plan)

        text = self._render(context=context, assembly=assembly, plan=plan)
        if not text.strip():
            base = super().synthesize(context=context, assembly=assembly)
            return _attach_plan_rationale(base, plan)

        regime_id = assembly.regime_id if assembly is not None else context.regime_id
        regime_name = assembly.regime_name if assembly is not None else context.regime_name
        abstract_action = assembly.abstract_action if assembly is not None else context.abstract_action
        rationale = _build_rationale(
            regime_name=regime_name,
            regime_id=regime_id,
            abstract_action=abstract_action,
            context=context,
            assembly=assembly,
            plan=plan,
        )
        return AgentResponse(
            text=text,
            regime_id=regime_id,
            abstract_action=abstract_action,
            rationale=rationale,
        )

    # ------------------------------------------------------------------
    # Rendering hooks — subclass to swap in an LLM
    # ------------------------------------------------------------------

    def _render(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot,
        plan: PromptPlan,
    ) -> str:
        sentences: list[str] = []
        for section in plan.sections:
            rendered = self._render_section(
                section=section, context=context, assembly=assembly, plan=plan
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
    ) -> str:
        if section is SectionId.ACKNOWLEDGE_PRESSURE:
            return self._render_acknowledge(context=context, assembly=assembly)
        if section is SectionId.REGIME_FRAME:
            return self._render_regime_frame(context=context, assembly=assembly)
        if section is SectionId.OPEN_LOOP_HANDOFF:
            return self._render_open_loop(context=context, assembly=assembly)
        if section is SectionId.CLARIFICATION:
            return self._render_clarification(context=context, assembly=assembly, plan=plan)
        if section is SectionId.NEXT_STEP:
            return self._render_next_step(context=context, assembly=assembly)
        if section is SectionId.BOUNDARY_DISCLAIMER:
            return self._render_boundary(context=context, assembly=assembly)
        if section is SectionId.REFLECTION_HOOK:
            return self._render_reflection(context=context)
        if section is SectionId.CONTINUITY_NOTE:
            return self._render_continuity(context=context, assembly=assembly)
        return ""

    # ------------------------------------------------------------------
    # Per-section text — keep deterministic so replay tests are stable
    # ------------------------------------------------------------------

    @staticmethod
    def _render_acknowledge(
        *, context: ResponseContext, assembly: ResponseAssemblySnapshot
    ) -> str:
        if context.regime_id == "repair_and_deescalation":
            return (
                "I want to slow this down so we can stay grounded together. "
                "I am not going to push past what just happened."
            )
        if context.regime_id == "emotional_support":
            return "I am hearing weight in this and I want to stay with it before we move."
        if assembly.continuum_target_position >= 0.7:
            return "I want to acknowledge the pressure here before we narrow anything."
        return "I am not going to rush past what you just said."

    @staticmethod
    def _render_regime_frame(
        *, context: ResponseContext, assembly: ResponseAssemblySnapshot
    ) -> str:
        regime = assembly.regime_id or context.regime_id or ""
        if regime == "emotional_support":
            return "I will stay supportive first and not jump into solving."
        if regime == "guided_exploration":
            return "I would rather explore this with you step by step than guess at one answer."
        if regime == "problem_solving":
            return "I see a structured path here we can walk together."
        if regime == "repair_and_deescalation":
            return "I will keep my responses steady and avoid escalating tone."
        if regime == "acquaintance_building":
            return "I want to keep this warm rather than transactional."
        if regime == "casual_social":
            return "I can stay in steady, low-pressure mode."
        return "I will stay context-aware and keep usefulness and continuity in view."

    @staticmethod
    def _render_open_loop(
        *, context: ResponseContext, assembly: ResponseAssemblySnapshot
    ) -> str:
        if assembly.case_hit_count or assembly.playbook_rule_count:
            return (
                "There is at least one thread we left open before; I will "
                "thread it forward instead of restarting from scratch."
            )
        return "I will keep prior open threads in view rather than dropping them."

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
        *, context: ResponseContext, assembly: ResponseAssemblySnapshot
    ) -> str:
        regime = assembly.regime_id or context.regime_id or ""
        if regime in {"emotional_support", "repair_and_deescalation"}:
            return "We can stay here together, and when you are ready, name one small next step that would feel manageable."
        if regime == "problem_solving":
            return "Concretely, the smallest useful next step is to name the constraint that matters most."
        if regime == "guided_exploration":
            return "Pick one of the threads we just surfaced and we can go a little deeper on it."
        return "From here we can take one small, concrete step that keeps things moving without forcing the pace."

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
            return f"I am also carrying forward {context.reflection_lesson_count} reflected lesson(s) from the slow loop."
        if context.primary_reflection_lesson is not None:
            return "I am letting the slower reflective layer shape this rather than treating it as a no-op."
        if context.primary_reflection_tension is not None:
            return "I am keeping an eye on the tension that is still open instead of smoothing past it."
        return ""

    @staticmethod
    def _render_continuity(
        *, context: ResponseContext, assembly: ResponseAssemblySnapshot
    ) -> str:
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


def _attach_plan_rationale(response: AgentResponse, plan: PromptPlan) -> AgentResponse:
    rationale = response.rationale
    if rationale and not rationale.endswith("."):
        rationale += "."
    plan_tag = f" Plan: intent={plan.intent.value}; sections={','.join(s.value for s in plan.sections)}; q={plan.question_budget}."
    return AgentResponse(
        text=response.text,
        regime_id=response.regime_id,
        abstract_action=response.abstract_action,
        rationale=(rationale + plan_tag).strip(),
    )


def _build_rationale(
    *,
    regime_name: str,
    regime_id: Optional[str],
    abstract_action: Optional[str],
    context: ResponseContext,
    assembly: ResponseAssemblySnapshot,
    plan: PromptPlan,
) -> str:
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
    return f"GroundedResponseSynthesizer from {regime_name}; " + ", ".join(parts) + "."
