"""Packet A (long-horizon-closure) — affordance -> PE prediction_id lineage.

Three cases prove the new ``plan_ref`` parameter on
``AffordanceInvoker.invoke`` actually closes the loop from a caller-supplied
``plan_ref`` all the way to ``PredictionActionContext.prediction_id`` on
the next turn:

1. Back-compat: ``plan_ref=None`` keeps the legacy "no lineage" path
   (``action_context.prediction_id == ""``); existing tests must not
   regress.
2. Core: ``plan_ref="p-001"`` produces a non-empty ``prediction_id``
   on the next-turn PE action context, alongside ``environment_outcome_id``.
   Both fields point at the same ``EnvironmentOutcome`` lineage.
3. Failure path: a backend that raises still routes through
   ``submit_tool_result`` (BACKEND_FAILED branch), so the next-turn
   action context still carries the ``plan_ref`` for credit attribution.

These are pure lineage tests; they do not assert anything about PE
magnitudes or credit shape — those are Packet B's territory.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from lifeform_affordance import (
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceInvocationStatus,
    AffordanceInvoker,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceRegistry,
    AffordanceSafety,
)


_HINT = (
    "Use only inside the long-horizon-closure lineage test to prove "
    "plan_ref propagates through the kernel."
)


def _echo_descriptor(name: str = "echo_lineage") -> AffordanceDescriptor:
    return AffordanceDescriptor(
        name=name,
        kind=AffordanceKind.TOOL,
        version="0.1.0",
        display_name="Echo (lineage probe)",
        description="Echo affordance used by the Packet A lineage test.",
        when_to_use=_HINT,
        when_not_to_use=_HINT + " Not suitable for any other test.",
        parameters_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        output_schema={"type": "object"},
        cost_model=AffordanceCost(latency_class=AffordanceLatencyClass.INSTANT),
        safety_model=AffordanceSafety(),
    )


async def _echo_backend(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"echoed": parameters["message"]}


async def _failing_backend(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    raise RuntimeError("backend deliberately failed for lineage test")


def _build_invoker_and_session(*, descriptor_name: str, backend) -> tuple[Any, Any]:
    """Build a real lifeform session + an invoker bound to a single
    descriptor. Returns ``(session, invoker)``.
    """
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(
        session_id=f"affordance-lineage-{descriptor_name}"
    )
    registry = AffordanceRegistry()
    registry.register(_echo_descriptor(descriptor_name))
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend(descriptor_name, backend)
    return session, invoker


@pytest.mark.asyncio
async def test_invoke_without_plan_ref_keeps_action_context_prediction_id_empty() -> None:
    """Back-compat: callers that don't bind a prediction see no
    lineage on the next turn. This guards against accidental
    "prediction_id is always non-empty" regressions.
    """
    session, invoker = _build_invoker_and_session(
        descriptor_name="echo_lineage_a", backend=_echo_backend
    )
    await session.run_turn("turn 1: warm up the scene")

    result = await invoker.invoke(
        "echo_lineage_a",
        {"message": "no plan_ref here"},
        session=session.brain_session,
        event_id="lineage-no-plan",
    )
    assert result.status is AffordanceInvocationStatus.SUCCEEDED
    assert result.tool_event_ids, "tool result must reach the kernel"

    next_turn = await session.run_turn("turn 3: what's the action context?")
    pe_snapshot = next_turn.active_snapshots["prediction_error"].value
    # Lineage must be empty when caller did not supply plan_ref.
    assert pe_snapshot.action_context.prediction_id == ""
    # outcome id IS still populated (it's the call's own id, not
    # caller-bound). This proves the two fields are independent.
    assert pe_snapshot.action_context.environment_outcome_id != ""


@pytest.mark.asyncio
async def test_invoke_with_plan_ref_threads_to_next_turn_pe_action_context() -> None:
    """Core test: plan_ref="p-001" -> EnvironmentOutcome.prediction_id
    -> next-turn PE action_context.prediction_id.
    """
    session, invoker = _build_invoker_and_session(
        descriptor_name="echo_lineage_b", backend=_echo_backend
    )
    await session.run_turn("turn 1: warm up the scene")

    result = await invoker.invoke(
        "echo_lineage_b",
        {"message": "with plan_ref"},
        session=session.brain_session,
        event_id="lineage-with-plan",
        plan_ref="p-001",
    )
    assert result.status is AffordanceInvocationStatus.SUCCEEDED
    assert result.tool_event_ids, "successful invoke must populate tool_event_ids"

    next_turn = await session.run_turn("turn 3: confirm lineage")
    pe_snapshot = next_turn.active_snapshots["prediction_error"].value
    # Both fields must be populated, both must point to the same
    # outcome lineage.
    assert pe_snapshot.action_context.prediction_id == "p-001"
    assert pe_snapshot.action_context.environment_outcome_id == (
        "lineage-with-plan:outcome"
    )


@pytest.mark.asyncio
async def test_failed_invoke_with_plan_ref_still_carries_lineage_next_turn() -> None:
    """A backend that raises still routes through submit_tool_result
    (BACKEND_FAILED branch). The plan_ref must still arrive on the
    next-turn action context so credit can attribute the failure
    to the right caller-bound prediction.
    """
    session, invoker = _build_invoker_and_session(
        descriptor_name="echo_lineage_c", backend=_failing_backend
    )
    await session.run_turn("turn 1: warm up the scene")

    result = await invoker.invoke(
        "echo_lineage_c",
        {"message": "this call will fail"},
        session=session.brain_session,
        event_id="lineage-failing",
        plan_ref="p-002",
    )
    assert result.status is AffordanceInvocationStatus.BACKEND_FAILED
    # ran_backend == True for BACKEND_FAILED, so submit_tool_result
    # was called and tool_event_ids is non-empty.
    assert result.tool_event_ids, (
        "BACKEND_FAILED must still route through submit_tool_result"
    )

    next_turn = await session.run_turn("turn 3: confirm lineage on failure")
    pe_snapshot = next_turn.active_snapshots["prediction_error"].value
    assert pe_snapshot.action_context.prediction_id == "p-002"
    assert pe_snapshot.action_context.environment_outcome_id == (
        "lineage-failing:outcome"
    )
