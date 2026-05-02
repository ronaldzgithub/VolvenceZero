from __future__ import annotations

from volvence_zero.agent.dialogue_trace import DialogueTraceStore
from volvence_zero.dialogue_trace import (
    DialogueOutcomeEvidence,
    DialogueOutcomeEvidenceSource,
    DialogueOutcomeKind,
    DialogueResolutionStatus,
)
from volvence_zero.environment import build_user_input_environment_event
from volvence_zero.prediction import ActualOutcome, PredictedOutcome, PredictionError


def test_dialogue_trace_resolver_does_not_keyword_classify_text() -> None:
    store = DialogueTraceStore()
    first_event = build_user_input_environment_event(
        event_id="event-1",
        user_input="I accept everything.",
        scene_id="scene-1",
        timestamp_ms=1,
    )
    second_event = build_user_input_environment_event(
        event_id="event-2",
        user_input="No, that is wrong and I reject it.",
        scene_id="scene-1",
        timestamp_ms=2,
    )

    first_trace, first_resolution = store.record_action(
        session_id="session",
        wave_id="wave-1",
        turn_index=1,
        environment_event=first_event,
        active_regime="support",
        active_abstract_action="clarify",
        response_text="first response",
        response_rationale="first rationale",
        next_prediction=PredictedOutcome(1, 2, 0.5, 0.5, 0.5, 0.5, 0.8, "pred"),
        evaluated_prediction=None,
        actual_outcome=None,
        prediction_error=None,
    )
    second_trace, second_resolution = store.record_action(
        session_id="session",
        wave_id="wave-2",
        turn_index=2,
        environment_event=second_event,
        active_regime="support",
        active_abstract_action="repair",
        response_text="second response",
        response_rationale="second rationale",
        next_prediction=PredictedOutcome(2, 3, 0.5, 0.5, 0.5, 0.5, 0.8, "pred"),
        evaluated_prediction=PredictedOutcome(1, 2, 0.5, 0.5, 0.5, 0.5, 0.8, "pred"),
        actual_outcome=ActualOutcome(2, 0.2, 0.2, 0.2, 0.2, "actual"),
        prediction_error=PredictionError(0.3, 0.3, 0.3, 0.3, 1.2, -0.3, "pe"),
    )

    assert first_resolution is None
    assert second_resolution is not None
    assert second_resolution.status is DialogueResolutionStatus.RESOLVED
    assert second_resolution.outcome.kind is DialogueOutcomeKind.UNKNOWN
    assert second_resolution.previous_trace_id == first_trace.trace_id
    assert second_resolution.observed_trace_id == second_trace.trace_id


def test_dialogue_trace_uses_only_structured_evidence_for_richer_outcome() -> None:
    store = DialogueTraceStore()
    first_event = build_user_input_environment_event(
        event_id="event-1",
        user_input="This raw text says reject but must not decide outcome.",
        scene_id="scene-1",
        timestamp_ms=1,
    )
    second_event = build_user_input_environment_event(
        event_id="event-2",
        user_input="This raw text says continue but structured evidence wins.",
        scene_id="scene-1",
        timestamp_ms=2,
    )

    first_trace, _ = store.record_action(
        session_id="session",
        wave_id="wave-1",
        turn_index=1,
        environment_event=first_event,
        active_regime="support",
        active_abstract_action="clarify",
        response_text="first response",
        response_rationale="first rationale",
        next_prediction=PredictedOutcome(1, 2, 0.5, 0.5, 0.5, 0.5, 0.8, "pred"),
        evaluated_prediction=None,
        actual_outcome=None,
        prediction_error=None,
    )
    _, resolution = store.record_action(
        session_id="session",
        wave_id="wave-2",
        turn_index=2,
        environment_event=second_event,
        active_regime="support",
        active_abstract_action="repair",
        response_text="second response",
        response_rationale="second rationale",
        next_prediction=PredictedOutcome(2, 3, 0.5, 0.5, 0.5, 0.5, 0.8, "pred"),
        evaluated_prediction=PredictedOutcome(1, 2, 0.5, 0.5, 0.5, 0.5, 0.8, "pred"),
        actual_outcome=ActualOutcome(2, 0.2, 0.2, 0.2, 0.2, "actual"),
        prediction_error=PredictionError(0.3, 0.3, 0.3, 0.3, 1.2, -0.3, "pe"),
        outcome_evidence=(
            DialogueOutcomeEvidence(
                evidence_id="eval:corrected",
                source=DialogueOutcomeEvidenceSource.EVALUATION,
                source_owner="EvaluationModule",
                outcome_kind=DialogueOutcomeKind.CORRECTED,
                confidence=0.8,
                evidence_refs=("evaluation:dialogue_correction",),
            ),
        ),
    )

    assert resolution is not None
    assert resolution.previous_trace_id == first_trace.trace_id
    assert resolution.outcome.kind is DialogueOutcomeKind.CORRECTED
    assert resolution.outcome.structured_evidence[0].evidence_id == "eval:corrected"
    assert "This raw text says" not in resolution.outcome.description


def test_dialogue_trace_export_is_replay_safe() -> None:
    store = DialogueTraceStore()
    event = build_user_input_environment_event(
        event_id="event-1",
        user_input="raw text should not appear in artifact",
        scene_id="scene-1",
        timestamp_ms=1,
    )
    trace, _ = store.record_action(
        session_id="session",
        wave_id="wave-1",
        turn_index=1,
        environment_event=event,
        active_regime="support",
        active_abstract_action="clarify",
        response_text="assistant raw response",
        response_rationale="public rationale",
        next_prediction=None,
        evaluated_prediction=None,
        actual_outcome=None,
        prediction_error=None,
    )

    artifact = store.export_replay_artifact()
    turn = artifact["turns"][0]
    assert turn["trace_id"] == trace.trace_id
    assert turn["response_text_hash"]
    assert "assistant raw response" not in str(artifact)
