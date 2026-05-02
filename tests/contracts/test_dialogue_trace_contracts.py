"""Contract tests for dialogue trace replay shapes."""

from __future__ import annotations

import dataclasses

import pytest

from volvence_zero.dialogue_trace import (
    DialogueActionKind,
    DialogueActionTrace,
    DialogueOutcomeEvidence,
    DialogueOutcomeEvidenceSource,
    DialogueOutcomeKind,
    DialogueOutcomeResolution,
    DialogueOutcomeTrace,
    DialogueResolutionStatus,
    DialogueTraceSnapshot,
    build_unknown_dialogue_outcome,
)
from volvence_zero.environment import build_primary_environment_frame


def test_dialogue_trace_enums_are_stable() -> None:
    assert set(DialogueActionKind) == {DialogueActionKind.ASSISTANT_RESPONSE}
    assert set(DialogueOutcomeKind) == {
        DialogueOutcomeKind.UNKNOWN,
        DialogueOutcomeKind.CONTINUED,
        DialogueOutcomeKind.CLARIFIED,
        DialogueOutcomeKind.CORRECTED,
        DialogueOutcomeKind.REJECTED,
        DialogueOutcomeKind.SCENE_CLOSED,
        DialogueOutcomeKind.DEFERRED,
    }
    assert set(DialogueResolutionStatus) == {
        DialogueResolutionStatus.PENDING,
        DialogueResolutionStatus.RESOLVED,
        DialogueResolutionStatus.STALE,
    }


def test_dialogue_action_trace_is_frozen_and_machine_readable() -> None:
    outcome = build_unknown_dialogue_outcome(
        previous_trace_id="trace-1",
        observed_trace_id="trace-1",
        observed_turn_index=1,
    )
    trace = DialogueActionTrace(
        trace_id="trace-1",
        event_id="event-1",
        wave_id="wave-1",
        turn_index=1,
        action_kind=DialogueActionKind.ASSISTANT_RESPONSE,
        environment_frame=build_primary_environment_frame(),
        environment_event_kind="user_input",
        environment_trigger_kind="user_input",
        active_regime="support",
        active_abstract_action="clarify",
        response_rationale="machine-readable rationale snapshot",
        prediction_id="prediction-1",
        outcome=outcome,
        response_text_hash="hash-1",
    )

    assert dataclasses.is_dataclass(trace)
    assert trace.__dataclass_params__.frozen
    assert trace.outcome.kind is DialogueOutcomeKind.UNKNOWN


def test_dialogue_outcome_evidence_is_structured_and_bounded() -> None:
    evidence = DialogueOutcomeEvidence(
        evidence_id="evidence-1",
        source=DialogueOutcomeEvidenceSource.EVALUATION,
        source_owner="EvaluationModule",
        outcome_kind=DialogueOutcomeKind.CLARIFIED,
        confidence=0.8,
        evidence_refs=("evaluation:clarification_resolved",),
        description="Evaluation readout resolved the clarification outcome.",
    )

    assert dataclasses.is_dataclass(evidence)
    assert evidence.__dataclass_params__.frozen
    assert evidence.outcome_kind is DialogueOutcomeKind.CLARIFIED

    with pytest.raises(ValueError, match="confidence"):
        DialogueOutcomeEvidence(
            evidence_id="evidence-2",
            source=DialogueOutcomeEvidenceSource.EVALUATION,
            source_owner="EvaluationModule",
            outcome_kind=DialogueOutcomeKind.CLARIFIED,
            confidence=1.5,
        )

    with pytest.raises(ValueError, match="source_owner"):
        DialogueOutcomeEvidence(
            evidence_id="evidence-3",
            source=DialogueOutcomeEvidenceSource.OWNER_SNAPSHOT,
            source_owner=" ",
            outcome_kind=DialogueOutcomeKind.CORRECTED,
            confidence=0.7,
        )


def test_dialogue_trace_contracts_reject_empty_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="trace_id"):
        DialogueActionTrace(
            trace_id=" ",
            event_id="event-1",
            wave_id="wave-1",
            turn_index=1,
            action_kind=DialogueActionKind.ASSISTANT_RESPONSE,
            environment_frame=build_primary_environment_frame(),
            environment_event_kind="user_input",
            environment_trigger_kind="user_input",
            active_regime=None,
            active_abstract_action=None,
            response_rationale="",
            prediction_id=None,
            outcome=build_unknown_dialogue_outcome(
                previous_trace_id="trace-1",
                observed_trace_id="trace-1",
                observed_turn_index=1,
            ),
        )

    outcome = DialogueOutcomeTrace(
        outcome_id="outcome-1",
        previous_trace_id="trace-1",
        observed_trace_id="trace-2",
        observed_turn_index=2,
        kind=DialogueOutcomeKind.UNKNOWN,
        evidence_refs=("evidence-1",),
        structured_evidence=(
            DialogueOutcomeEvidence(
                evidence_id="structured-1",
                source=DialogueOutcomeEvidenceSource.EVALUATION,
                source_owner="EvaluationModule",
                outcome_kind=DialogueOutcomeKind.CONTINUED,
                confidence=0.75,
            ),
        ),
    )
    trace = DialogueActionTrace(
        trace_id="trace-1",
        event_id="event-1",
        wave_id="wave-1",
        turn_index=1,
        action_kind=DialogueActionKind.ASSISTANT_RESPONSE,
        environment_frame=build_primary_environment_frame(),
        environment_event_kind="user_input",
        environment_trigger_kind="user_input",
        active_regime=None,
        active_abstract_action=None,
        response_rationale="",
        prediction_id=None,
        outcome=outcome,
    )

    with pytest.raises(ValueError, match="trace_id"):
        DialogueTraceSnapshot(
            traces=(trace, trace),
            unresolved_trace_ids=("trace-1",),
            resolved_outcomes=(outcome,),
            description="duplicate trace ids",
        )


def test_dialogue_outcome_resolution_is_explicit() -> None:
    outcome = build_unknown_dialogue_outcome(
        previous_trace_id="trace-1",
        observed_trace_id="trace-2",
        observed_turn_index=2,
        prediction_error_refs=("pe:1",),
    )
    resolution = DialogueOutcomeResolution(
        previous_trace_id="trace-1",
        observed_trace_id="trace-2",
        status=DialogueResolutionStatus.RESOLVED,
        outcome=outcome,
        description="resolved through prediction_error snapshot",
    )

    assert resolution.status is DialogueResolutionStatus.RESOLVED
    assert resolution.outcome.kind is DialogueOutcomeKind.UNKNOWN
