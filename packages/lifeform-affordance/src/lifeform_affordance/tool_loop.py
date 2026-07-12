"""Bounded lifeform-side automatic tool loop.

The loop deliberately lives in ``lifeform-affordance`` instead of the
``vz-*`` kernel. It advances cognition only through public
``LifeformSession.run_turn`` and ``BrainSession.submit_tool_result`` via
``AffordanceInvoker``.
"""

from __future__ import annotations

import json
import time
import uuid
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from lifeform_core.types import TurnTriggerKind

from lifeform_affordance.invoker import (
    AffordanceInvocationResult,
    AffordanceInvocationStatus,
    AffordanceInvoker,
)
from lifeform_affordance.registry import AffordanceRegistry


class ToolLoopStopReason(str, Enum):
    """Why a tool loop stopped."""

    FINAL_TEXT = "final_text"
    TOOL_CALLS_EMITTED = "tool_calls_emitted"
    ASYNC_TASK_HANDOFF = "async_task_handoff"
    CONFIRMATION_REQUIRED = "confirmation_required"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TOOL_BLOCKED = "tool_blocked"
    NO_TOOL_INTENT = "no_tool_intent"
    STOPPED_BY_CONVERSATION = "stopped_by_conversation"
    PAUSED_BY_CONVERSATION = "paused_by_conversation"


class ToolLoopDecision(str, Enum):
    """Typed conversational decision for the next loop action."""

    INVOKE_TOOL = "invoke_tool"
    FINAL_ANSWER = "final_answer"
    STOP = "stop"
    PAUSE = "pause"
    CONTINUE = "continue"


@dataclass(frozen=True)
class ToolLoopPolicy:
    """Per-request bounds for automatic tool execution."""

    max_tool_steps: int = 4
    max_wall_ms: int = 30_000
    allow_async_tasks: bool = True
    server_side_execution: bool = True
    require_user_confirmation: bool = False

    def __post_init__(self) -> None:
        if self.max_tool_steps < 0:
            raise ValueError("ToolLoopPolicy.max_tool_steps must be >= 0")
        if self.max_wall_ms <= 0:
            raise ValueError("ToolLoopPolicy.max_wall_ms must be > 0")


@dataclass(frozen=True)
class ToolCallIntent:
    """Typed request to invoke or emit one affordance call."""

    descriptor_name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    call_id: str = ""
    plan_ref: str | None = None
    source: str = "affordance_snapshot"

    def __post_init__(self) -> None:
        if not self.descriptor_name.strip():
            raise ValueError("ToolCallIntent.descriptor_name must be non-empty")
        if self.call_id and not self.call_id.strip():
            raise ValueError("ToolCallIntent.call_id must be non-empty when supplied")

    @property
    def stable_call_id(self) -> str:
        if self.call_id:
            return self.call_id
        payload = json.dumps(
            {
                "descriptor_name": self.descriptor_name,
                "parameters": dict(self.parameters),
                "plan_ref": self.plan_ref or "",
                "source": self.source,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return f"call_{uuid.uuid5(uuid.NAMESPACE_URL, payload).hex[:16]}"


@dataclass(frozen=True)
class ToolIntentProposalContext:
    """Inputs exposed to a conversational tool-intent proposer."""

    user_input: str
    latest_response_text: str
    descriptors: tuple[Any, ...]
    active_regime_id: str | None = None
    previous_steps: tuple[ToolLoopStep, ...] = ()


@dataclass(frozen=True)
class ToolIntentProposal:
    """Output from a conversational tool-intent proposer."""

    decision: ToolLoopDecision
    intent: ToolCallIntent | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.decision is ToolLoopDecision.INVOKE_TOOL and self.intent is None:
            raise ValueError("INVOKE_TOOL proposals must include intent")


ToolIntentProvider = Callable[[str], str | Awaitable[str]]


class ToolIntentProposer:
    """Protocol-like base for chat -> typed tool-loop decisions."""

    async def propose(self, context: ToolIntentProposalContext) -> ToolIntentProposal:
        return ToolIntentProposal(decision=ToolLoopDecision.FINAL_ANSWER)


class LLMToolIntentProposer(ToolIntentProposer):
    """LLM-backed proposer using centralized prompt and JSON schema files."""

    def __init__(self, provider: ToolIntentProvider) -> None:
        self._provider = provider

    async def propose(self, context: ToolIntentProposalContext) -> ToolIntentProposal:
        prompt = _render_tool_intent_prompt(context)
        raw = self._provider(prompt)
        text = await raw if inspect.isawaitable(raw) else raw
        return _parse_tool_intent_json(str(text), context=context)


@dataclass(frozen=True)
class ToolLoopStep:
    """Immutable audit record for one tool-loop step."""

    step_index: int
    intent: ToolCallIntent
    invocation: AffordanceInvocationResult | None
    status: str
    elapsed_ms: int


@dataclass(frozen=True)
class ToolLoopResult:
    """Result of one bounded tool loop."""

    final_turn_result: Any
    stop_reason: ToolLoopStopReason
    steps: tuple[ToolLoopStep, ...] = ()
    emitted_tool_calls: tuple[ToolCallIntent, ...] = ()
    async_task_ids: tuple[str, ...] = ()

    @property
    def response_text(self) -> str:
        response = self.final_turn_result.response
        return response.text


class ToolLoopOrchestrator:
    """Run bounded tool execution around ``LifeformSession.run_turn``."""

    def __init__(
        self,
        *,
        registry: AffordanceRegistry,
        invoker: AffordanceInvoker,
        policy: ToolLoopPolicy | None = None,
        contract_id: str | None = None,
        granted_consents: frozenset[str] = frozenset(),
        intent_proposer: ToolIntentProposer | None = None,
    ) -> None:
        self._registry = registry
        self._invoker = invoker
        self._policy = policy or ToolLoopPolicy()
        self._contract_id = contract_id
        self._granted_consents = granted_consents
        self._intent_proposer = intent_proposer or ToolIntentProposer()

    async def run(
        self,
        *,
        session: Any,
        user_input: str,
        initial_intents: Sequence[ToolCallIntent] = (),
        trigger_kind: TurnTriggerKind = TurnTriggerKind.USER_INPUT,
    ) -> ToolLoopResult:
        """Run a bounded server-side loop.

        ``initial_intents`` lets protocol adapters execute a structured
        tool request (for example an OpenAI assistant ``tool_calls`` entry)
        without parsing natural language. If no initial intent is provided,
        the orchestrator looks for a published affordance snapshot selection.
        """

        started = time.monotonic()
        turn_result = await session.run_turn(user_input, trigger_kind=trigger_kind)
        pending_intents = list(initial_intents)
        steps: list[ToolLoopStep] = []
        current_input = user_input

        while len(steps) < self._policy.max_tool_steps:
            if self._wall_elapsed_ms(started) >= self._policy.max_wall_ms:
                return ToolLoopResult(
                    final_turn_result=turn_result,
                    stop_reason=ToolLoopStopReason.BUDGET_EXHAUSTED,
                    steps=tuple(steps),
                )

            proposal = (
                ToolIntentProposal(
                    decision=ToolLoopDecision.INVOKE_TOOL,
                    intent=pending_intents.pop(0),
                    rationale="initial structured intent",
                )
                if pending_intents
                else await self._next_proposal(
                    session=session,
                    turn_result=turn_result,
                    user_input=current_input,
                    steps=tuple(steps),
                )
            )
            if proposal.decision is ToolLoopDecision.STOP:
                return ToolLoopResult(
                    final_turn_result=turn_result,
                    stop_reason=ToolLoopStopReason.STOPPED_BY_CONVERSATION,
                    steps=tuple(steps),
                )
            if proposal.decision is ToolLoopDecision.PAUSE:
                return ToolLoopResult(
                    final_turn_result=turn_result,
                    stop_reason=ToolLoopStopReason.PAUSED_BY_CONVERSATION,
                    steps=tuple(steps),
                )
            intent = proposal.intent
            if intent is None:
                reason = (
                    ToolLoopStopReason.FINAL_TEXT
                    if steps
                    else ToolLoopStopReason.NO_TOOL_INTENT
                )
                return ToolLoopResult(
                    final_turn_result=turn_result,
                    stop_reason=reason,
                    steps=tuple(steps),
                )

            if not self._policy.server_side_execution:
                return ToolLoopResult(
                    final_turn_result=turn_result,
                    stop_reason=ToolLoopStopReason.TOOL_CALLS_EMITTED,
                    steps=tuple(steps),
                    emitted_tool_calls=(intent,),
                )

            step_started = time.monotonic()
            invocation = await self._invoker.invoke(
                intent.descriptor_name,
                intent.parameters,
                active_regime_id=self._active_regime_id(turn_result),
                granted_consents=self._granted_consents,
                user_confirmed=not self._policy.require_user_confirmation,
                session=session.brain_session,
                event_id=intent.stable_call_id,
                action_id=f"{intent.descriptor_name}:{intent.stable_call_id}",
                plan_ref=intent.plan_ref,
                idempotency_key=intent.stable_call_id,
                contract_id=self._contract_id,
            )
            steps.append(
                ToolLoopStep(
                    step_index=len(steps) + 1,
                    intent=intent,
                    invocation=invocation,
                    status=invocation.status.value,
                    elapsed_ms=self._wall_elapsed_ms(step_started),
                )
            )
            if invocation.status is AffordanceInvocationStatus.PENDING_CONFIRMATION:
                return ToolLoopResult(
                    final_turn_result=turn_result,
                    stop_reason=ToolLoopStopReason.CONFIRMATION_REQUIRED,
                    steps=tuple(steps),
                )
            if invocation.status is AffordanceInvocationStatus.TASK_QUEUED:
                task_id = invocation.task_id
                return ToolLoopResult(
                    final_turn_result=turn_result,
                    stop_reason=ToolLoopStopReason.ASYNC_TASK_HANDOFF,
                    steps=tuple(steps),
                    async_task_ids=(task_id,) if task_id else (),
                )
            if invocation.status is not AffordanceInvocationStatus.SUCCEEDED:
                # Symmetric with the success path: drain any queued
                # semantic events (e.g. failed `execution_result` /
                # `open_loop`) inside the same interaction so vz
                # actually learns from a failed tool attempt instead
                # of waiting for the next user message.
                if invocation.status is AffordanceInvocationStatus.BACKEND_FAILED:
                    turn_result = await session.run_turn(
                        self._failure_continuation_text(intent, invocation),
                        trigger_kind=TurnTriggerKind.APPRENTICE,
                    )
                return ToolLoopResult(
                    final_turn_result=turn_result,
                    stop_reason=ToolLoopStopReason.TOOL_BLOCKED,
                    steps=tuple(steps),
                )
            current_input = self._continuation_text(intent)
            turn_result = await session.run_turn(
                current_input, trigger_kind=TurnTriggerKind.APPRENTICE
            )

        return ToolLoopResult(
            final_turn_result=turn_result,
            stop_reason=ToolLoopStopReason.BUDGET_EXHAUSTED,
            steps=tuple(steps),
        )

    def emit_tool_call(self, intent: ToolCallIntent) -> dict[str, Any]:
        """Render one intent as an OpenAI assistant ``tool_calls`` item."""

        descriptor = self._registry.get(intent.descriptor_name)
        parameters = json.dumps(dict(intent.parameters), ensure_ascii=False)
        return {
            "id": intent.stable_call_id,
            "type": "function",
            "function": {
                "name": descriptor.name,
                "arguments": parameters,
            },
        }

    async def _next_proposal(
        self,
        *,
        session: Any,
        turn_result: Any,
        user_input: str,
        steps: tuple[ToolLoopStep, ...],
    ) -> ToolIntentProposal:
        snapshot_intent = self._intent_from_latest_snapshots(session=session)
        if snapshot_intent is not None:
            return ToolIntentProposal(
                decision=ToolLoopDecision.INVOKE_TOOL,
                intent=snapshot_intent,
                rationale="affordance snapshot selected a parameterless tool",
            )
        descriptors = self._registry.list_for_session(contract_id=self._contract_id)
        return await self._intent_proposer.propose(
            ToolIntentProposalContext(
                user_input=user_input,
                latest_response_text=turn_result.response.text,
                descriptors=descriptors,
                active_regime_id=self._active_regime_id(turn_result),
                previous_steps=steps,
            )
        )

    def _intent_from_latest_snapshots(self, *, session: Any) -> ToolCallIntent | None:
        snapshots = session.latest_active_snapshots
        affordance_snapshot = snapshots.get("affordance")
        if affordance_snapshot is None:
            return None
        selected = affordance_snapshot.value.selected
        if selected is None or selected.is_blocked:
            return None
        descriptor = self._registry.get(selected.descriptor_name)
        required = tuple(descriptor.parameters_schema.get("required", ()))
        if required:
            # Parameter synthesis is a typed proposal source, not a guess.
            return None
        return ToolCallIntent(
            descriptor_name=selected.descriptor_name,
            parameters={},
            plan_ref=self._plan_ref_from_snapshots(session=session),
            source="affordance_snapshot",
        )

    @staticmethod
    def _active_regime_id(turn_result: Any) -> str | None:
        return turn_result.active_regime

    @staticmethod
    def _plan_ref_from_snapshots(*, session: Any) -> str | None:
        """Forward the PE-owner-issued pre-action prediction id (CP-10).

        The single ``prediction_error`` owner stamps ``prediction_id`` on the
        ``next_prediction`` it publishes; the tool loop only forwards that
        reference. When the PE snapshot is absent or carries no owner-issued
        id (bootstrap), we fall back to the latest closed temporal segment id
        so delayed segment credit still has an anchor, and finally to ``None``
        (= explicitly unknown lineage, never a fabricated id).
        """

        snapshots = session.latest_active_snapshots
        pe_snapshot = snapshots.get("prediction_error")
        if pe_snapshot is not None:
            prediction_id = pe_snapshot.value.next_prediction.prediction_id
            if prediction_id:
                return prediction_id
        temporal = snapshots.get("temporal_abstraction")
        if temporal is None:
            return None
        closed_segments = temporal.value.closed_segments
        if not closed_segments:
            return None
        return closed_segments[-1].segment_id

    @staticmethod
    def _continuation_text(intent: ToolCallIntent) -> str:
        return (
            "Continue the turn using the just-submitted tool result for "
            f"{intent.descriptor_name}."
        )

    @staticmethod
    def _failure_continuation_text(
        intent: ToolCallIntent,
        invocation: AffordanceInvocationResult,
    ) -> str:
        return (
            f"The tool {intent.descriptor_name!r} failed with "
            f"{invocation.error_class}: {invocation.error_detail}. "
            "Update the open loop / execution_result owner based on this "
            "failure and acknowledge to the user."
        )

    @staticmethod
    def _wall_elapsed_ms(started: float) -> int:
        return int((time.monotonic() - started) * 1000)


def _render_tool_intent_prompt(context: ToolIntentProposalContext) -> str:
    prompt_template = _read_package_text("prompts/tool_intent_proposal.md")
    schema = _read_package_text("schemas/tool_intent_proposal.json")
    descriptors = [
        {
            "name": descriptor.name,
            "description": descriptor.description,
            "when_to_use": descriptor.when_to_use,
            "when_not_to_use": descriptor.when_not_to_use,
            "parameters_schema": dict(descriptor.parameters_schema),
        }
        for descriptor in context.descriptors
    ]
    steps = [
        {
            "step_index": step.step_index,
            "tool": step.intent.descriptor_name,
            "status": step.status,
        }
        for step in context.previous_steps
    ]
    return prompt_template.format(
        schema=schema,
        user_input=context.user_input,
        latest_response_text=context.latest_response_text,
        active_regime_id=context.active_regime_id or "",
        descriptors=json.dumps(descriptors, ensure_ascii=False, sort_keys=True),
        previous_steps=json.dumps(steps, ensure_ascii=False, sort_keys=True),
    )


def _parse_tool_intent_json(
    text: str,
    *,
    context: ToolIntentProposalContext,
) -> ToolIntentProposal:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("tool intent proposer returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("tool intent proposer JSON must be an object")
    raw_decision = payload["decision"]
    decision = ToolLoopDecision(str(raw_decision))
    if decision is not ToolLoopDecision.INVOKE_TOOL:
        return ToolIntentProposal(
            decision=decision,
            rationale=str(payload.get("rationale", "")),
        )
    tool_name = payload["tool_name"]
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("invoke_tool proposal requires non-empty tool_name")
    available_names = {descriptor.name for descriptor in context.descriptors}
    if tool_name not in available_names:
        raise ValueError(f"tool intent proposer selected unavailable tool {tool_name!r}")
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("invoke_tool proposal parameters must be an object")
    call_id = payload.get("call_id", "")
    if call_id is not None and not isinstance(call_id, str):
        raise ValueError("invoke_tool proposal call_id must be a string")
    plan_ref = payload.get("plan_ref")
    if plan_ref is not None and not isinstance(plan_ref, str):
        raise ValueError("invoke_tool proposal plan_ref must be a string or null")
    return ToolIntentProposal(
        decision=ToolLoopDecision.INVOKE_TOOL,
        intent=ToolCallIntent(
            descriptor_name=tool_name,
            parameters=parameters,
            call_id=call_id or "",
            plan_ref=plan_ref,
            source="conversational_proposer",
        ),
        rationale=str(payload.get("rationale", "")),
    )


def _read_package_text(relative_path: str) -> str:
    path = Path(__file__).resolve().parent / relative_path
    return path.read_text(encoding="utf-8")


__all__ = [
    "ToolCallIntent",
    "LLMToolIntentProposer",
    "ToolIntentProposal",
    "ToolIntentProposalContext",
    "ToolIntentProposer",
    "ToolLoopOrchestrator",
    "ToolLoopDecision",
    "ToolLoopPolicy",
    "ToolLoopResult",
    "ToolLoopStep",
    "ToolLoopStopReason",
]
