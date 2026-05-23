"""Failure-path drain test for ToolLoopOrchestrator.

The orchestrator must run an extra ``session.run_turn(APPRENTICE)``
when the tool invocation returns ``BACKEND_FAILED`` so the
``execution_result`` / ``open_loop`` semantic events queued by
``submit_tool_result`` are drained inside the same interaction
instead of dangling until the next user message. This mirrors the
success-path behaviour and is what makes "the twin actually learns
from a failed tool" land in vz cognition.

The test stubs the ``AffordanceRegistry`` / ``AffordanceInvoker`` so
it can run in isolation from the heavyweight lifeform stack.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from lifeform_affordance.invoker import (
    AffordanceInvocationResult,
    AffordanceInvocationStatus,
)
from lifeform_affordance.tool_loop import (
    ToolCallIntent,
    ToolLoopOrchestrator,
    ToolLoopPolicy,
    ToolLoopStopReason,
)
from lifeform_core.types import TurnTriggerKind


@dataclass
class _StubResponse:
    text: str = ""


@dataclass
class _StubTurnResult:
    response: _StubResponse
    active_regime: str | None = None


class _StubSession:
    def __init__(self) -> None:
        self.run_turn_calls: list[tuple[str, TurnTriggerKind]] = []
        self.brain_session = object()

    async def run_turn(
        self,
        text: str,
        *,
        trigger_kind: TurnTriggerKind = TurnTriggerKind.USER_INPUT,
    ) -> _StubTurnResult:
        self.run_turn_calls.append((text, trigger_kind))
        return _StubTurnResult(response=_StubResponse(text=""))


class _StubRegistry:
    def get(self, name: str) -> Any:  # pragma: no cover - not used by failure path
        raise AssertionError(
            f"registry.get unexpectedly called for {name!r}"
        )


class _FailingInvoker:
    def __init__(self) -> None:
        self.invoke_calls: list[str] = []

    async def invoke(
        self,
        descriptor_name: str,
        parameters: Mapping[str, Any],
        **kwargs: Any,
    ) -> AffordanceInvocationResult:
        self.invoke_calls.append(descriptor_name)
        return AffordanceInvocationResult(
            descriptor_name=descriptor_name,
            status=AffordanceInvocationStatus.BACKEND_FAILED,
            error_class="ToolExecutionError",
            error_detail="simulated backend failure",
            tool_event_ids=("evt-1",),
        )


@pytest.mark.asyncio
async def test_tool_loop_drains_failed_invocation_within_same_interaction() -> None:
    session = _StubSession()
    invoker = _FailingInvoker()
    orchestrator = ToolLoopOrchestrator(
        registry=_StubRegistry(),
        invoker=invoker,
        policy=ToolLoopPolicy(server_side_execution=True),
    )

    result = await orchestrator.run(
        session=session,
        user_input="please attempt the failing tool",
        initial_intents=(
            ToolCallIntent(
                descriptor_name="vz-bundle.read_file",
                parameters={"path": "missing.txt"},
                call_id="call-fail-1",
                plan_ref="plan-1",
            ),
        ),
    )

    assert result.stop_reason is ToolLoopStopReason.TOOL_BLOCKED
    assert invoker.invoke_calls == ["vz-bundle.read_file"]

    # Two run_turn calls: the initial user turn AND the apprentice
    # drain turn introduced by the failure-path fix. Without the fix
    # we would see exactly one call.
    assert len(session.run_turn_calls) == 2, session.run_turn_calls
    initial_text, initial_kind = session.run_turn_calls[0]
    drain_text, drain_kind = session.run_turn_calls[1]
    assert initial_kind is TurnTriggerKind.USER_INPUT
    assert initial_text == "please attempt the failing tool"
    assert drain_kind is TurnTriggerKind.APPRENTICE
    assert "vz-bundle.read_file" in drain_text
    assert "BACKEND_FAILED" in drain_text or "failed" in drain_text


def test_module_runs_under_plain_asyncio() -> None:
    """Sanity check: confirms the async test above is genuinely awaited.

    Some CI environments mis-configure ``pytest-asyncio`` and silently
    drop coroutines on the floor. Re-running the same coroutine via
    ``asyncio.run`` here gives a fallback signal if that happens.
    """

    asyncio.run(
        test_tool_loop_drains_failed_invocation_within_same_interaction()
    )
