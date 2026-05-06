"""LLM-backed grounded synthesizer.

Composes:

* The kernel's ``LLMResponseSynthesizer`` for the actual generation call
  (it already plumbs system prompt + chat messages + decoding constraints
  + control codes through the substrate runtime).
* Our ``PromptPlanner`` to add a structured plan on top, attached to the
  rationale field for downstream auditing.

This is the "real LLM" path of the lifeform expression layer. The plain
``GroundedResponseSynthesizer`` (in ``response_synthesizer.py``) is the
deterministic / template path used by the synthetic substrate and most
tests; this class is what a product wires up when the brain runs against
an actual HF runtime.

Why subclass ``LLMResponseSynthesizer`` rather than rebuild the prompt
plumbing here:

* The kernel already has ``build_system_prompt`` / ``build_chat_messages``
  that consume the assembly + context. Re-implementing that in the
  lifeform layer would duplicate kernel logic and become a hidden second
  owner of prompt assembly (R8 violation).
* Our value-add is the **plan** — section ordering / question budget /
  intent — exposed in a structured rationale tail. The plan is also
  available as ``synthesize.last_plan`` for product code that wants to
  surface the audit trail (e.g. evaluation, dashboards).

Construct with ``LifeformLLMResponseSynthesizer(runtime=...)`` and pass it
to ``Lifeform(response_synthesizer=...)``.
"""

from __future__ import annotations

from typing import Any

from volvence_zero.agent.response import (
    AgentResponse,
    LLMResponseSynthesizer,
    ResponseContext,
)
from volvence_zero.application.runtime import ResponseAssemblySnapshot

from lifeform_expression.prompt_planner import PromptPlan, PromptPlanner


class LifeformLLMResponseSynthesizer(LLMResponseSynthesizer):
    """LLM synthesizer that records a structured plan per turn."""

    def __init__(
        self,
        *,
        runtime: Any,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        planner: PromptPlanner | None = None,
    ) -> None:
        super().__init__(
            runtime=runtime,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        self._planner = planner or PromptPlanner()
        self._last_plan: PromptPlan | None = None

    @property
    def planner(self) -> PromptPlanner:
        return self._planner

    @property
    def last_plan(self) -> PromptPlan | None:
        """The most recent ``PromptPlan`` produced for inspection / audit.

        Returns ``None`` until the first ``synthesize`` call.
        """
        return self._last_plan

    def synthesize(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot | None = None,
    ) -> AgentResponse:
        plan = self._planner.plan(context=context, assembly=assembly)
        self._last_plan = plan
        response = super().synthesize(context=context, assembly=assembly)
        return _attach_plan_rationale(response, plan)


def _attach_plan_rationale(response: AgentResponse, plan: PromptPlan) -> AgentResponse:
    rationale = response.rationale or ""
    if rationale and not rationale.endswith("."):
        rationale += "."
    plan_tag = (
        f" Plan: intent={plan.intent.value};"
        f" sections={','.join(s.value for s in plan.sections)};"
        f" q={plan.question_budget}."
    )
    merged: list[str] = []
    seen: set[str] = set()
    for tag in tuple(response.rationale_tags) + tuple(plan.rationale_tags):
        if tag and tag not in seen:
            seen.add(tag)
            merged.append(tag)
    plan_summary_tag = (
        "plan="
        f"intent:{plan.intent.value};"
        f"sections:{','.join(s.value for s in plan.sections)};"
        f"q:{plan.question_budget}"
    )
    if plan_summary_tag not in seen:
        merged.append(plan_summary_tag)
    return AgentResponse(
        text=response.text,
        regime_id=response.regime_id,
        abstract_action=response.abstract_action,
        rationale=(rationale + plan_tag).strip(),
        rationale_tags=tuple(merged),
    )
