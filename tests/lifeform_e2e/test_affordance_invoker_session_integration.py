"""End-to-end: AffordanceInvoker drives a real BrainSession.

Validates that when the invoker feeds its result through
``session.submit_tool_result``, the kernel's semantic-state owners
actually see it: ``execution_result`` gets a completed record,
and the ``tool_event_ids`` returned by the invoker are the
kernel's semantic proposal IDs.

This is the slice-2a proof that the affordance pipeline isn't a
pure-Python parallel universe \u2014 it plugs into the same canonical
tool-result bus that product code has been using all along
(``semantic_events_from_tool_result`` \u2192
``ToolResultSemanticAdapter`` \u2192 ``execution_result``).
"""

from __future__ import annotations

from typing import Any, Mapping

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
    "Use when the lifeform needs a tiny echo of the supplied message "
    "to prove tool-call plumbing is working."
)


def _echo_descriptor() -> AffordanceDescriptor:
    return AffordanceDescriptor(
        name="echo",
        kind=AffordanceKind.TOOL,
        version="0.1.0",
        display_name="Echo",
        description="Test affordance that echoes its message parameter.",
        when_to_use=_HINT,
        when_not_to_use=_HINT + " Not suitable for production use.",
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
    return {"echoed": parameters["message"], "length": len(parameters["message"])}


async def test_invoker_feeds_completion_into_real_brain_session() -> None:
    """A successful invocation should produce execution_result
    records visible in the kernel's next snapshot.
    """
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="invoker-e2e-success")
    # Run one turn to open the scene so the brain is fully warmed.
    await session.run_turn("hello")

    registry = AffordanceRegistry()
    registry.register(_echo_descriptor())
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("echo", _echo_backend)

    result = await invoker.invoke(
        "echo",
        {"message": "hi from the affordance pipeline"},
        session=session.brain_session,
        event_id="affordance-echo-1",
    )

    assert result.status is AffordanceInvocationStatus.SUCCEEDED
    assert result.payload == {
        "echoed": "hi from the affordance pipeline",
        "length": 31,
    }
    # tool_event_ids MUST be non-empty when the session was wired.
    assert result.tool_event_ids, (
        "Expected invoker to return semantic event ids from "
        "submit_tool_result; got empty tuple."
    )
    # Run another turn so the kernel processes the queued tool event
    # and the execution_result owner reflects it.
    await session.run_turn("what happened?")
    execution_snap = session.latest_active_snapshots.get("execution_result")
    assert execution_snap is not None, (
        "execution_result snapshot missing from the latest turn; "
        "tool-result bus wiring is broken."
    )
    completed = execution_snap.value.completed_actions
    assert any(
        "affordance-echo-1" in record.record_id for record in completed
    ), (
        f"Expected the invoker's event_id 'affordance-echo-1' to appear "
        f"in execution_result.completed_actions; got ids: "
        f"{[r.record_id for r in completed]!r}"
    )


async def test_invoker_failure_feeds_failure_event_into_brain_session() -> None:
    """A failed backend should produce a failed tool-result event
    that lands in execution_result.failed_actions.
    """
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="invoker-e2e-failure")
    await session.run_turn("hello")

    async def _failing_backend(_p):
        raise RuntimeError("backend deliberately broke")

    registry = AffordanceRegistry()
    registry.register(_echo_descriptor())
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("echo", _failing_backend)

    result = await invoker.invoke(
        "echo",
        {"message": "trigger failure"},
        session=session.brain_session,
        event_id="affordance-echo-fail",
    )
    assert result.status is AffordanceInvocationStatus.BACKEND_FAILED
    assert result.error_class == "RuntimeError"
    assert result.tool_event_ids  # even failures go on the bus

    # Next turn: the failed tool event should land in the
    # execution_result owner's failed_actions bucket.
    await session.run_turn("did it work?")
    execution_snap = session.latest_active_snapshots.get("execution_result")
    assert execution_snap is not None
    failed = execution_snap.value.failed_actions
    assert any(
        "affordance-echo-fail" in record.record_id for record in failed
    ), (
        f"Expected failed invocation to appear in "
        f"execution_result.failed_actions; got ids: "
        f"{[r.record_id for r in failed]!r}"
    )


async def test_invoker_boundary_denial_does_not_touch_brain_session() -> None:
    """When the boundary policy vetoes, nothing should be queued on
    the session's tool-result bus. The kernel must not see any
    trace of the attempted call.
    """
    from lifeform_domain_emogpt import build_companion_lifeform

    lifeform = build_companion_lifeform()
    session = lifeform.create_session(session_id="invoker-e2e-denied")
    await session.run_turn("hello")

    # Descriptor with a confirmation gate; caller forgets to confirm.
    denied_descriptor = AffordanceDescriptor(
        name="irreversible_op",
        kind=AffordanceKind.TOOL,
        version="0.1.0",
        display_name="Irreversible Op",
        description="A dangerous op that must be confirmed first.",
        when_to_use=_HINT,
        when_not_to_use=_HINT + " Requires explicit confirmation gate.",
        parameters_schema={"type": "object"},
        output_schema={"type": "object"},
        cost_model=AffordanceCost(latency_class=AffordanceLatencyClass.INSTANT),
        safety_model=AffordanceSafety(requires_user_confirmation=True, irreversible=True),
    )
    registry = AffordanceRegistry()
    registry.register(denied_descriptor)
    invoker = AffordanceInvoker(registry=registry)
    invoker.register_backend("irreversible_op", _echo_backend)

    result = await invoker.invoke(
        "irreversible_op",
        {},
        session=session.brain_session,
        event_id="affordance-denied",
        user_confirmed=False,
    )
    assert result.status is AffordanceInvocationStatus.DENIED_BY_BOUNDARY
    assert result.error_class == "confirmation_required"
    assert result.tool_event_ids == (), (
        "Denied invocations must NOT enqueue kernel tool events; the "
        "denial happens before the backend call, before the session "
        "wiring, so tool_event_ids stays empty."
    )

    # Run another turn and verify the denied event_id never appears
    # in execution_result. We check by id, not by total count, because
    # the kernel's default NoOp semantic-proposal runtime adds an
    # observe record per turn regardless of tool activity \u2014 so
    # total attempted_actions naturally grows.
    await session.run_turn("anything new?")
    execution_snap = session.latest_active_snapshots.get("execution_result")
    if execution_snap is not None:
        all_ids = [r.record_id for r in execution_snap.value.attempted_actions]
        assert not any("affordance-denied" in rid for rid in all_ids), (
            f"Denied invocation's event_id leaked into execution_result: "
            f"{all_ids!r}"
        )
