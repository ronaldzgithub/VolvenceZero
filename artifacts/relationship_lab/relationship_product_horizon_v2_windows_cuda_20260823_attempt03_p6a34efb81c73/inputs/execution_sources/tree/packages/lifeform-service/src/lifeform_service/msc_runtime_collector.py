"""Evidence-only projection for the MSC complete-runtime context arm."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math

from volvence_zero.agent.session import AgentTurnResult
from volvence_zero.application.types import ResponseAssemblySnapshot
from volvence_zero.prediction import PredictionErrorSnapshot
from volvence_zero.steering_contracts import (
    STEERING_CONDITION_BELIEF_SLOT,
    STEERING_GATE_DECISION_SLOT,
    STEERING_INTERVENTION_SLOT,
    SteeringConditionBelief,
    SteeringGateAction,
    SteeringGateDecision,
    SteeringIntervention,
)


MSC_RUNTIME_CONTEXT_SCHEMA_VERSION = "msc-full-runtime-context.v1"
MSC_REQUIRED_RUNTIME_SLOTS = (
    "substrate",
    "memory",
    "prediction_error",
    "credit",
    "world_temporal",
    "self_temporal",
    "temporal_abstraction",
    "evaluation",
    "response_assembly",
)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_msc_runtime_context_payload(
    *,
    result: AgentTurnResult,
    assembly: ResponseAssemblySnapshot,
    turn_latency_ms: float,
) -> dict[str, object]:
    """Publish one conditioned substrate context after the full DAG turn."""

    evidence = result.response.runtime_context_evidence
    if evidence is None:
        raise RuntimeError("MSC collector turn lacks runtime context evidence")
    if not result.acceptance_passed:
        raise RuntimeError(
            "MSC collector cannot attest a turn that failed runtime acceptance"
        )
    active_slots = tuple(sorted(result.active_snapshots))
    shadow_slots = tuple(sorted(result.shadow_snapshots))
    published_slots = frozenset((*active_slots, *shadow_slots))
    missing = tuple(
        slot for slot in MSC_REQUIRED_RUNTIME_SLOTS if slot not in published_slots
    )
    if missing:
        raise RuntimeError(f"MSC complete runtime is missing slots: {missing!r}")
    if result.event_count < len(MSC_REQUIRED_RUNTIME_SLOTS):
        raise RuntimeError("MSC complete runtime propagate event count is too small")
    if result.substrate_fallback_active:
        raise RuntimeError("MSC complete runtime cannot use substrate fallback")
    if result.substrate_model_id is None or result.substrate_runtime_origin is None:
        raise RuntimeError("MSC complete runtime lacks substrate identity")
    if (
        not math.isfinite(turn_latency_ms)
        or turn_latency_ms < evidence.generation_latency_ms
    ):
        raise RuntimeError(
            "MSC complete runtime turn latency is below generation latency"
        )

    snapshot = evidence.representation
    if len(snapshot.representations) != 1:
        raise RuntimeError("MSC runtime context must contain exactly one row")
    row = snapshot.representations[0]
    temporal_n_z = len(assembly.control_code)
    if temporal_n_z < 1:
        raise RuntimeError("MSC runtime context lacks temporal controller code")
    slot_surface = {
        "active": active_slots,
        "shadow": shadow_slots,
    }
    payload: dict[str, object] = {
        "schema_version": MSC_RUNTIME_CONTEXT_SCHEMA_VERSION,
        "collector_path": (
            "lifeform-service->LifeformSession.run_turn->"
            "AgentSessionRunner.run_turn->run_final_wiring_turn->propagate"
        ),
        "volvence_full_stack": True,
        "acceptance_passed": True,
        "propagate_event_count": result.event_count,
        "required_runtime_slots": MSC_REQUIRED_RUNTIME_SLOTS,
        "runtime_slot_surface_sha256": _sha256_json(slot_surface),
        "active_slot_names": active_slots,
        "shadow_slot_names": shadow_slots,
        "active_speaker_id": result.active_speaker_id,
        "temporal_n_z": temporal_n_z,
        "temporal_code_sha256": _sha256_json(assembly.control_code),
        "substrate_model_id": result.substrate_model_id,
        "substrate_runtime_origin": result.substrate_runtime_origin,
        "substrate_fallback_active": False,
        "context_lineage": asdict(snapshot.lineage),
        "context_representation": {
            "sample_id": row.sample_id,
            "source_sha256": row.source_sha256,
            "values": row.values,
            "values_sha256": row.values_sha256,
        },
        "input_token_count": evidence.input_token_count,
        "output_token_count": evidence.output_token_count,
        "total_token_count": (
            evidence.input_token_count + evidence.output_token_count
        ),
        "generation_latency_ms": evidence.generation_latency_ms,
        "end_to_end_latency_ms": turn_latency_ms,
        "raw_text_retained": False,
        "evaluation_writeback_allowed": False,
    }
    steering_snapshots = {
        slot: result.shadow_snapshots.get(slot)
        or result.active_snapshots.get(slot)
        for slot in (
            STEERING_CONDITION_BELIEF_SLOT,
            STEERING_GATE_DECISION_SLOT,
            STEERING_INTERVENTION_SLOT,
        )
    }
    present = tuple(
        slot for slot, steering_snapshot in steering_snapshots.items()
        if steering_snapshot is not None
    )
    if present:
        if len(present) != len(steering_snapshots):
            raise RuntimeError(
                "MSC steering collector received a partial owner chain: "
                f"{present!r}"
            )
        belief_snapshot = steering_snapshots[STEERING_CONDITION_BELIEF_SLOT]
        gate_snapshot = steering_snapshots[STEERING_GATE_DECISION_SLOT]
        intervention_snapshot = steering_snapshots[STEERING_INTERVENTION_SLOT]
        if (
            belief_snapshot is None
            or gate_snapshot is None
            or intervention_snapshot is None
        ):  # pragma: no cover - guarded above
            raise RuntimeError("MSC steering snapshots disappeared")
        belief = belief_snapshot.value
        gate = gate_snapshot.value
        intervention = intervention_snapshot.value
        prediction = result.active_snapshots["prediction_error"].value
        if not isinstance(belief, SteeringConditionBelief):
            raise TypeError("MSC steering belief snapshot has the wrong value type")
        if not isinstance(gate, SteeringGateDecision):
            raise TypeError("MSC steering gate snapshot has the wrong value type")
        if not isinstance(intervention, SteeringIntervention):
            raise TypeError("MSC steering intervention has the wrong value type")
        if not isinstance(prediction, PredictionErrorSnapshot):
            raise TypeError("MSC steering collector requires PredictionErrorSnapshot")
        if gate.action is not SteeringGateAction.STEER:
            raise RuntimeError(
                "MSC steering counterfactual collection requires explicit "
                "always-steer SHADOW gate"
            )
        if (
            intervention.action is not SteeringGateAction.STEER
            or not intervention.shadow_hook_executed
            or intervention.noop_context is None
            or intervention.action_context is None
        ):
            raise RuntimeError(
                "MSC steering collector lacks matched noop/steer contexts"
            )
        payload["steering_shadow"] = {
            "schema_version": "msc-steering-shadow-context.v1",
            "belief": asdict(belief),
            "gate_observations": gate.observations,
            "gate_policy_artifact_id": gate.policy_artifact_id,
            "gate_policy_version": gate.policy_version,
            "decision_id": gate.decision_id,
            "prediction_error_magnitude": prediction.error.magnitude,
            "noop_context": asdict(intervention.noop_context),
            "action_context": asdict(intervention.action_context),
            "executor_artifact_id": intervention.executor_artifact_id,
            "reader_artifact_id": intervention.reader_artifact_id,
            "source_model_id": intervention.source_model_id,
            "source_model_weights_sha256": (
                intervention.source_model_weights_sha256
            ),
            "layer_index": intervention.layer_index,
            "residual_norm": intervention.residual_norm,
            "control_norm": intervention.control_norm,
            "control_norm_cap": intervention.control_norm_cap,
            "free_bias_present": False,
            "zero_code_strict_noop": True,
            "shadow_hook_latency_ms": intervention.shadow_hook_latency_ms,
            "shadow_hook_executed": True,
            "sensor_off_action_context": (
                asdict(intervention.sensor_off_action_context)
                if intervention.sensor_off_action_context is not None
                else None
            ),
            "sensor_off_executor_artifact_id": (
                intervention.sensor_off_executor_artifact_id
            ),
            "sensor_off_control_norm": intervention.sensor_off_control_norm,
            "sensor_off_shadow_hook_latency_ms": (
                intervention.sensor_off_shadow_hook_latency_ms
            ),
            "raw_text_retained": False,
            "evaluation_writeback_allowed": False,
        }
    return payload


__all__ = (
    "MSC_REQUIRED_RUNTIME_SLOTS",
    "MSC_RUNTIME_CONTEXT_SCHEMA_VERSION",
    "build_msc_runtime_context_payload",
)
