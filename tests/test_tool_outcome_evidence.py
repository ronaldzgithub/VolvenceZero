from __future__ import annotations

import asyncio

from volvence_zero.agent.dialogue_outcome_producers import (
    TOOL_OUTCOME_CONFIDENCE,
    tool_outcome_evidence_from_environment_outcome,
)
from volvence_zero.brain import Brain, BrainConfig
from volvence_zero.dialogue_trace import (
    DialogueOutcomeEvidenceSource,
    DialogueOutcomeKind,
)
from volvence_zero.environment import EnvironmentEventKind, EnvironmentOutcome


def _build_outcome(status: str, *, detail: str = "detail", prediction_id: str | None = None) -> EnvironmentOutcome:
    return EnvironmentOutcome(
        outcome_id="tool:1:outcome",
        event_id="tool:1",
        outcome_kind=EnvironmentEventKind.TOOL_RESULT,
        action_id="action:1",
        status=status,
        summary="tool completed",
        detail=detail,
        confidence=0.8,
        prediction_id=prediction_id,
    )


def test_tool_outcome_producer_maps_typed_status_to_outcome_kind() -> None:
    success = tool_outcome_evidence_from_environment_outcome(
        environment_outcome=_build_outcome("succeeded", prediction_id="plan-1"),
        tool_name="read_file",
    )
    failure = tool_outcome_evidence_from_environment_outcome(
        environment_outcome=_build_outcome("failed"),
        tool_name="read_file",
    )
    deferred = tool_outcome_evidence_from_environment_outcome(
        environment_outcome=_build_outcome("pending_confirmation"),
        tool_name="read_file",
    )
    unknown = tool_outcome_evidence_from_environment_outcome(
        environment_outcome=_build_outcome("weird_undocumented_status"),
        tool_name="read_file",
    )

    assert success and success[0].outcome_kind is DialogueOutcomeKind.CONTINUED
    assert success[0].confidence == 0.8 * TOOL_OUTCOME_CONFIDENCE
    assert success[0].source is DialogueOutcomeEvidenceSource.OWNER_SNAPSHOT
    assert "tool_prediction:plan-1" in success[0].evidence_refs
    assert failure and failure[0].outcome_kind is DialogueOutcomeKind.REJECTED
    assert deferred and deferred[0].outcome_kind is DialogueOutcomeKind.DEFERRED
    assert unknown == ()


def test_tool_outcome_producer_does_not_read_detail_text() -> None:
    evidence = tool_outcome_evidence_from_environment_outcome(
        environment_outcome=_build_outcome(
            "succeeded",
            detail="raw text full of failed and rejected words",
        ),
        tool_name="read_file",
    )

    assert evidence and evidence[0].outcome_kind is DialogueOutcomeKind.CONTINUED
    assert all(
        "raw text" not in ref for ref in evidence[0].evidence_refs
    )
    assert "raw text" not in evidence[0].description


def test_brain_submit_tool_result_attaches_evidence_to_last_trace() -> None:
    brain = Brain(BrainConfig(rare_heavy_enabled=False))
    session = brain.create_session(session_id="tool-evidence-session")

    asyncio.run(session.run_turn_async("First turn that records a dialogue trace."))
    initial_snapshot = session.runner.dialogue_trace_snapshot
    assert initial_snapshot is not None
    initial_resolved = len(initial_snapshot.resolved_outcomes)

    session.submit_tool_result(
        event_id="tool:1",
        tool_name="read_file",
        action_id="action:1",
        status="succeeded",
        summary="tool completed",
        detail="detail body that should not be parsed",
        confidence=0.9,
        plan_ref="plan-1",
    )

    snapshot = session.runner.dialogue_trace_snapshot
    assert snapshot is not None
    assert len(snapshot.resolved_outcomes) == initial_resolved + 1
    last_resolved = snapshot.resolved_outcomes[-1]
    assert last_resolved.kind is DialogueOutcomeKind.CONTINUED
    assert any(
        evidence.evidence_id == "tool_outcome:tool:1:outcome"
        for evidence in last_resolved.structured_evidence
    )


def test_brain_submit_tool_result_links_next_turn_prediction_context() -> None:
    brain = Brain(BrainConfig(rare_heavy_enabled=False))
    session = brain.create_session(session_id="tool-outcome-pe-context")

    asyncio.run(session.run_turn_async("First turn that records a prediction."))
    session.submit_tool_result(
        event_id="tool:prediction",
        tool_name="read_file",
        action_id="action:prediction",
        status="succeeded",
        summary="tool completed",
        detail="detail body",
        plan_ref="plan-1",
    )
    result = asyncio.run(session.run_turn_async("Continue after tool result."))

    prediction_snapshot = result.active_snapshots["prediction_error"].value
    assert prediction_snapshot.action_context.environment_outcome_id == "tool:prediction:outcome"
    replay = session.runner.export_snapshot_replay_artifact()
    action_replay = replay["action_replay"]
    assert action_replay["action_context"]["environment_outcome_id"] == "tool:prediction:outcome"
    assert action_replay["prediction_error"]["turn_index"] >= 1
    assert "credit_records" in action_replay


def test_brain_submit_tool_result_failure_marks_trace_rejected() -> None:
    brain = Brain(BrainConfig(rare_heavy_enabled=False))
    session = brain.create_session(session_id="tool-evidence-failure-session")

    asyncio.run(session.run_turn_async("First turn that records a dialogue trace."))
    session.submit_tool_result(
        event_id="tool:fail",
        tool_name="read_file",
        action_id="action:fail",
        status="failed",
        summary="tool failed",
        detail="ignored detail",
    )

    snapshot = session.runner.dialogue_trace_snapshot
    assert snapshot is not None
    last_resolved = snapshot.resolved_outcomes[-1]
    assert last_resolved.kind is DialogueOutcomeKind.REJECTED
