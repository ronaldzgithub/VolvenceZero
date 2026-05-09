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

from collections import deque
from dataclasses import replace
from typing import Any

from volvence_zero.agent.response import (
    AgentResponse,
    LLMResponseSynthesizer,
    ResponseContext,
)
from volvence_zero.application.runtime import ResponseAssemblySnapshot

from lifeform_expression.prompt_planner import PromptPlan, PromptPlanner


# Default conversation history budget. Six turns ≈ 12 chat messages,
# which is small enough not to blow up the substrate context window
# but large enough to give a 1-3B base model the raw cues a short
# follow-up like "然后呢" or a single-token user utterance needs.
_DEFAULT_HISTORY_TURNS = 6
# Per-message character cap used when injecting prior turns. Long
# assistant turns (e.g. structured replies) get trimmed in the middle
# so that the head and tail are both visible to the model.
_HISTORY_MESSAGE_CHAR_BUDGET = 600


class LifeformLLMResponseSynthesizer(LLMResponseSynthesizer):
    """LLM synthesizer that records a structured plan per turn.

    Per-instance state (``_history`` and ``_last_plan``) is **not**
    safe to share across concurrent sessions. ``Lifeform`` clones
    one synthesizer per session via :meth:`clone_for_session` so
    each conversation gets its own ring buffer.

    The optional ``figure_bundle`` parameter binds a frozen
    :class:`lifeform_domain_figure.FigureArtifactBundle` to this
    synthesizer. When set, the figure-vertical L1 / L3 / L4
    enforcement layers (style prior injection, grounded decoding,
    scope refusal) consume the bundle through the duck-typed
    contract surface — the synthesizer itself never imports
    ``lifeform_domain_figure``, so the wheel-boundary direction is
    preserved.
    """

    def __init__(
        self,
        *,
        runtime: Any,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        planner: PromptPlanner | None = None,
        history_turns: int = _DEFAULT_HISTORY_TURNS,
        figure_bundle: object | None = None,
    ) -> None:
        super().__init__(
            runtime=runtime,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        self._planner = planner or PromptPlanner()
        self._last_plan: PromptPlan | None = None
        self._history_turns = max(0, history_turns)
        self._history: deque[tuple[str, str]] = deque(maxlen=self._history_turns)
        self._figure_bundle = figure_bundle

    @property
    def planner(self) -> PromptPlanner:
        return self._planner

    @property
    def last_plan(self) -> PromptPlan | None:
        """The most recent ``PromptPlan`` produced for inspection / audit.

        Returns ``None`` until the first ``synthesize`` call.
        """
        return self._last_plan

    @property
    def history_turns(self) -> int:
        return self._history_turns

    @property
    def figure_bundle(self) -> object | None:
        return self._figure_bundle

    def with_figure_bundle(
        self, bundle: object | None
    ) -> "LifeformLLMResponseSynthesizer":
        """Return a clone bound to ``bundle`` (or unbound when ``None``).

        The figure bundle is shared by reference across the clone —
        it is frozen, so this is safe; new sessions get their own
        history ring buffer through :meth:`clone_for_session`.
        """

        clone = self.clone_for_session()
        clone._figure_bundle = bundle  # noqa: SLF001 — internal reassignment
        return clone

    def clone_for_session(self) -> "LifeformLLMResponseSynthesizer":
        """Return a session-local clone with an empty history buffer.

        ``runtime`` and ``planner`` are shared by reference (the
        substrate runtime is intentionally process-wide; the planner
        is stateless across turns). The figure bundle is also shared
        by reference because it is frozen. Only the ring buffer and
        the ``last_plan`` cache are independent so concurrent
        sessions do not see each other's turns.
        """

        clone = type(self)(
            runtime=self._runtime,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            planner=self._planner,
            history_turns=self._history_turns,
            figure_bundle=self._figure_bundle,
        )
        return clone

    def synthesize(
        self,
        *,
        context: ResponseContext,
        assembly: ResponseAssemblySnapshot | None = None,
    ) -> AgentResponse:
        plan = self._planner.plan(context=context, assembly=assembly)
        self._last_plan = plan
        history_turns = self._snapshot_history()
        if history_turns and not context.prior_turns:
            context = replace(context, prior_turns=history_turns)
        response = super().synthesize(context=context, assembly=assembly)
        self._record_turn(
            user_text=context.user_input,
            assistant_text=response.text,
        )
        return _attach_plan_rationale(response, plan)

    def _snapshot_history(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._history)

    def _record_turn(self, *, user_text: str, assistant_text: str) -> None:
        if self._history_turns <= 0:
            return
        if not user_text.strip() and not assistant_text.strip():
            return
        self._history.append(
            (
                _trim_history_message(user_text),
                _trim_history_message(assistant_text),
            )
        )


def _trim_history_message(text: str) -> str:
    compact = text.strip()
    if len(compact) <= _HISTORY_MESSAGE_CHAR_BUDGET:
        return compact
    head_budget = _HISTORY_MESSAGE_CHAR_BUDGET // 2
    tail_budget = _HISTORY_MESSAGE_CHAR_BUDGET - head_budget - 5
    return f"{compact[:head_budget]}…{compact[-tail_budget:]}"


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
