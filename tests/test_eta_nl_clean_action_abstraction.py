from __future__ import annotations

from dataclasses import fields

from volvence_zero.agent import AgentSessionRunner
from volvence_zero.credit.gate import derive_segment_closure_credit_records
from volvence_zero.environment import EnvironmentEventKind, EnvironmentOutcome
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.temporal import (
    ControllerState,
    TemporalAbstractionSnapshot,
    TemporalSegmentClosure,
)


def test_no_action_outcome_trace_runtime_slot_is_declared() -> None:
    assert not hasattr(AgentSessionRunner, "action_outcome_trace_snapshot")


def test_environment_outcome_contains_only_observable_extension_fields() -> None:
    field_names = {field.name for field in fields(EnvironmentOutcome)}

    assert {"latency_ms", "monetary_cost", "reversibility", "environment_state_delta_kind"} <= field_names
    assert "trust_delta" not in field_names
    assert "common_ground_delta" not in field_names
    assert "commitment_progress_delta" not in field_names
    assert "information_gain" not in field_names


def test_segment_closure_context_is_temporal_owned() -> None:
    closure = TemporalSegmentClosure(
        segment_id="segment-1",
        open_turn_index=1,
        close_turn_index=3,
        abstract_action_id="clarify-before-act",
        z_t_digest=(0.1, 0.2),
        beta_open_digest=0.8,
        beta_close_digest=0.9,
        description="closed by beta_t switch",
    )
    snapshot = TemporalAbstractionSnapshot(
        controller_state=ControllerState(
            code=(0.1, 0.2),
            code_dim=2,
            switch_gate=0.9,
            is_switching=True,
            steps_since_switch=0,
        ),
        active_abstract_action="new-action",
        controller_params_hash="hash",
        description="temporal snapshot",
        closed_segments=(closure,),
    )

    assert snapshot.closed_segments[0].segment_id == "segment-1"
    assert snapshot.closed_segments[0].abstract_action_id == "clarify-before-act"


def test_segment_credit_is_derived_from_prediction_error_snapshot_only() -> None:
    context = PredictionActionContext(
        segment_id="segment-1",
        abstract_action_id="clarify-before-act",
        z_t_digest=(0.1, 0.2),
        environment_event_id="env-1",
    )
    pe_snapshot = PredictionErrorSnapshot(
        evaluated_prediction=None,
        actual_outcome=ActualOutcome(
            observed_turn_index=2,
            task_progress=0.5,
            relationship_delta=0.5,
            regime_stability=0.5,
            action_payoff=0.2,
            description="actual",
            action_context=context,
        ),
        next_prediction=PredictedOutcome(
            source_turn_index=2,
            target_turn_index=3,
            predicted_task_progress=0.5,
            predicted_relationship_delta=0.5,
            predicted_regime_stability=0.5,
            predicted_action_payoff=0.5,
            confidence=0.5,
            description="predicted",
            action_context=context,
        ),
        error=PredictionError(
            task_error=0.0,
            relationship_error=0.0,
            regime_error=0.0,
            action_error=-0.3,
            magnitude=0.3,
            signed_reward=-0.3,
            description="action segment underperformed",
        ),
        turn_index=2,
        bootstrap=False,
        description="pe with action context",
        action_context=context,
    )
    temporal_snapshot = TemporalAbstractionSnapshot(
        controller_state=ControllerState(
            code=(0.1, 0.2),
            code_dim=2,
            switch_gate=0.9,
            is_switching=True,
            steps_since_switch=0,
        ),
        active_abstract_action="new-action",
        controller_params_hash="hash",
        description="temporal snapshot",
        closed_segments=(
            TemporalSegmentClosure(
                segment_id="segment-1",
                open_turn_index=1,
                close_turn_index=2,
                abstract_action_id="clarify-before-act",
                z_t_digest=(0.1, 0.2),
                beta_open_digest=0.8,
                beta_close_digest=0.9,
            ),
        ),
    )

    records = derive_segment_closure_credit_records(
        prediction_error_snapshot=pe_snapshot,
        temporal_snapshot=temporal_snapshot,
        timestamp_ms=10,
    )

    assert len(records) == 1
    assert records[0].level == "abstract_action_segment"
    assert records[0].source_event == "segment:segment-1"
    assert "clarify-before-act" in records[0].context


def test_snapshot_replay_export_uses_existing_snapshots() -> None:
    runner = AgentSessionRunner(session_id="snapshot-replay-test")
    artifact = runner.export_snapshot_replay_artifact()

    assert artifact["session_id"] == "snapshot-replay-test:context-1"
    assert artifact["snapshot_count"] == 0
    assert "action_replay" in artifact
    assert "dialogue_trace" in artifact
    assert "trace-specific runtime schema" in artifact["description"]


def test_environment_outcome_observable_defaults() -> None:
    outcome = EnvironmentOutcome(
        outcome_id="out-1",
        event_id="evt-1",
        outcome_kind=EnvironmentEventKind.TOOL_RESULT,
        action_id="act-1",
        status="succeeded",
        summary="done",
        detail="detail",
    )

    assert outcome.latency_ms is None
    assert outcome.monetary_cost == 0.0
    assert outcome.reversibility == "reversible"
    assert outcome.environment_state_delta_kind == "none"
