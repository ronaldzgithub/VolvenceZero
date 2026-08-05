from __future__ import annotations

import hashlib
import struct
from types import SimpleNamespace

import pytest

from companion_bench.prediction_research import (
    parse_msc_full_runtime_context,
)
from lifeform_service.msc_runtime_collector import (
    MSC_REQUIRED_RUNTIME_SLOTS,
    build_msc_runtime_context_payload,
)
from volvence_zero.agent.response import RuntimeContextEvidence
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.runtime import Snapshot
from volvence_zero.steering_contracts import (
    STEERING_CONDITION_BELIEF_SLOT,
    STEERING_GATE_DECISION_SLOT,
    STEERING_INTERVENTION_SLOT,
    SteeringConditionBelief,
    SteeringGateAction,
    SteeringGateDecision,
    SteeringIntervention,
    SteeringResidualContext,
)
from volvence_zero.substrate import (
    SubstrateFingerprint,
    SyntheticOpenWeightResidualRuntime,
    publish_runtime_capture_representation,
)


def _result() -> SimpleNamespace:
    runtime = SyntheticOpenWeightResidualRuntime(model_id="frozen-model")
    snapshot = publish_runtime_capture_representation(
        sample_id="runtime-observation",
        source_sha256="b" * 64,
        capture=runtime.capture(source_text="transient test input"),
        model_fingerprint=SubstrateFingerprint(
            model_id="frozen-model",
            version="snapshot-v1",
            weights_sha256="a" * 64,
        ),
        runtime_origin="hf-local",
    )
    evidence = RuntimeContextEvidence(
        representation=snapshot,
        input_token_count=12,
        output_token_count=3,
        generation_latency_ms=4.0,
    )
    return SimpleNamespace(
        response=SimpleNamespace(runtime_context_evidence=evidence),
        acceptance_passed=True,
        active_snapshots={slot: object() for slot in MSC_REQUIRED_RUNTIME_SLOTS},
        shadow_snapshots={},
        event_count=14,
        substrate_fallback_active=False,
        substrate_model_id="frozen-model",
        substrate_runtime_origin="hf-local",
        active_speaker_id="speaker_2",
    )


def test_service_projection_attests_complete_runtime_without_raw_text() -> None:
    payload = build_msc_runtime_context_payload(
        result=_result(),
        assembly=SimpleNamespace(control_code=(0.1, 0.2, 0.3)),
        turn_latency_ms=6.5,
    )

    parsed = parse_msc_full_runtime_context(payload, sample_id="sample-1")
    assert parsed.model_id == "frozen-model"
    assert parsed.runtime_origin == "hf-local"
    assert parsed.input_token_count == 12
    assert parsed.output_token_count == 3
    assert parsed.generation_latency_ms == 4.0
    assert parsed.latency_ms == 6.5
    assert payload["raw_text_retained"] is False
    assert "transient test input" not in repr(payload)


def test_service_projection_rejects_partial_dag_and_bad_timing() -> None:
    result = _result()
    result.active_snapshots.pop("credit")
    with pytest.raises(RuntimeError, match="missing slots"):
        build_msc_runtime_context_payload(
            result=result,
            assembly=SimpleNamespace(control_code=(0.1, 0.2, 0.3)),
            turn_latency_ms=6.5,
        )

    with pytest.raises(RuntimeError, match="below generation latency"):
        build_msc_runtime_context_payload(
            result=_result(),
            assembly=SimpleNamespace(control_code=(0.1, 0.2, 0.3)),
            turn_latency_ms=3.0,
        )


def _snapshot(slot: str, owner: str, value: object) -> Snapshot[object]:
    return Snapshot(slot, owner, 1, 1, value)


def _prediction_error() -> PredictionErrorSnapshot:
    predicted = PredictedOutcome(
        source_turn_index=1,
        target_turn_index=2,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.0,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.0,
        confidence=0.8,
        description="Prediction fixture.",
    )
    return PredictionErrorSnapshot(
        evaluated_prediction=predicted,
        actual_outcome=ActualOutcome(
            observed_turn_index=2,
            task_progress=0.6,
            relationship_delta=0.1,
            regime_stability=0.5,
            action_payoff=0.2,
            description="Outcome fixture.",
        ),
        next_prediction=PredictedOutcome(
            source_turn_index=2,
            target_turn_index=3,
            predicted_task_progress=0.5,
            predicted_relationship_delta=0.0,
            predicted_regime_stability=0.5,
            predicted_action_payoff=0.0,
            confidence=0.8,
            description="Next prediction fixture.",
        ),
        error=PredictionError(
            task_error=0.1,
            relationship_error=0.1,
            regime_error=0.0,
            action_error=0.0,
            magnitude=0.4,
            signed_reward=0.1,
            description="PE fixture.",
        ),
        turn_index=2,
        bootstrap=False,
        description="Prediction-error fixture.",
    )


def _residual_context(*, conditioned: bool) -> SteeringResidualContext:
    values = (0.6, 0.8)
    return SteeringResidualContext(
        source_sha256="b" * 64,
        layer_indices=(0,),
        activation_widths=(2,),
        values=values,
        values_sha256=hashlib.sha256(struct.pack("!2d", *values)).hexdigest(),
        conditioned=conditioned,
    )


def test_service_projection_includes_complete_text_free_steering_ablation() -> None:
    result = _result()
    belief = SteeringConditionBelief(
        belief_label="relationship",
        belief_index=0,
        belief_margin=0.8,
        fresh_belief_label="relationship",
        fresh_belief_index=0,
        fresh_margin=0.9,
        belief_disagrees_fresh=False,
        staleness_proxy=0.1,
        base_action_entropy=0.2,
        reader_artifact_id="reader-v1",
        source_model_id="frozen-model",
        source_layer_index=0,
        source_residual_norm=1.0,
        description="Belief fixture.",
    )
    observations = (
        ("belief_margin", 0.8),
        ("prediction_error_magnitude", 0.1),
    )
    gate = SteeringGateDecision(
        decision_id="gate-v1:decision:1",
        action=SteeringGateAction.STEER,
        steer_probability=1.0,
        observations=observations,
        policy_artifact_id="gate-v1",
        policy_version=1,
        terminal_credit_pending=True,
        decision_mode="frozen-policy-argmax",
        description="Always-steer collector fixture.",
    )
    intervention = SteeringIntervention(
        action=SteeringGateAction.STEER,
        source_model_id="frozen-model",
        source_model_weights_sha256="a" * 64,
        layer_index=0,
        residual_delta=(0.1, 0.0),
        residual_norm=1.0,
        control_norm=0.1,
        control_norm_cap=0.25,
        executor_artifact_id="executor-v1",
        reader_artifact_id="reader-v1",
        gate_policy_version=1,
        zero_code_noop=False,
        application_mode="shadow-preview",
        shadow_hook_executed=True,
        runtime_backend="transformers-direct-steering:frozen-model",
        downstream_effect=(0.1, 0.0, 0.0),
        description="Intervention fixture.",
        noop_context=_residual_context(conditioned=False),
        action_context=_residual_context(conditioned=True),
        shadow_hook_latency_ms=0.1,
        sensor_off_action_context=_residual_context(conditioned=True),
        sensor_off_executor_artifact_id="executor-unconditional-v1",
        sensor_off_control_norm=0.1,
        sensor_off_shadow_hook_latency_ms=0.1,
    )
    result.active_snapshots["prediction_error"] = _snapshot(
        "prediction_error", "PredictionErrorModule", _prediction_error()
    )
    result.shadow_snapshots = {
        STEERING_CONDITION_BELIEF_SLOT: _snapshot(
            STEERING_CONDITION_BELIEF_SLOT, "SteeringSensorModule", belief
        ),
        STEERING_GATE_DECISION_SLOT: _snapshot(
            STEERING_GATE_DECISION_SLOT, "SteeringGateModule", gate
        ),
        STEERING_INTERVENTION_SLOT: _snapshot(
            STEERING_INTERVENTION_SLOT, "SteeringExecutorModule", intervention
        ),
    }

    payload = build_msc_runtime_context_payload(
        result=result,
        assembly=SimpleNamespace(control_code=(0.1, 0.2, 0.3)),
        turn_latency_ms=6.5,
    )
    steering = payload["steering_shadow"]

    assert steering["decision_id"] == gate.decision_id
    assert steering["sensor_off_executor_artifact_id"] == (
        "executor-unconditional-v1"
    )
    assert steering["raw_text_retained"] is False
    assert steering["evaluation_writeback_allowed"] is False
