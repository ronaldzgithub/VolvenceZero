"""Substrate-owned bounded residual steering executor RuntimeModule."""

from __future__ import annotations

import hashlib
import math
import struct
import time
from typing import Mapping

from volvence_zero.runtime import (
    ContractViolationError,
    RuntimeModule,
    RuntimePlaceholderValue,
    Snapshot,
    WiringLevel,
)
from volvence_zero.steering_contracts import (
    STEERING_CONDITION_BELIEF_SLOT,
    STEERING_GATE_DECISION_SLOT,
    STEERING_INTERVENTION_SLOT,
    SteeringConditionBelief,
    SteeringExecutorArtifact,
    SteeringGateAction,
    SteeringGateDecision,
    SteeringIntervention,
    SteeringResidualContext,
)
from volvence_zero.substrate.adapter import SubstrateSnapshot
from volvence_zero.substrate.residual_interfaces import OpenWeightResidualRuntime


class SteeringExecutorModule(RuntimeModule[SteeringIntervention]):
    """Apply the frozen rank-r multiplicative operator under a gate decision."""

    slot_name = STEERING_INTERVENTION_SLOT
    owner = "SteeringExecutorModule"
    value_type = SteeringIntervention
    dependencies = (
        "substrate",
        STEERING_CONDITION_BELIEF_SLOT,
        STEERING_GATE_DECISION_SLOT,
    )
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        artifact: SteeringExecutorArtifact,
        runtime: OpenWeightResidualRuntime | None = None,
        source_text: str = "",
        apply_shadow_hook: bool = True,
        sensor_off_artifact: SteeringExecutorArtifact | None = None,
        ungated_action: SteeringGateAction | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._artifact = artifact
        self._runtime = runtime
        self._source_text = source_text
        self._apply_shadow_hook = bool(apply_shadow_hook)
        self._sensor_off_artifact = sensor_off_artifact
        self._ungated_action = ungated_action
        if self.wiring_level is WiringLevel.SHADOW and self._apply_shadow_hook:
            if runtime is None:
                raise ValueError("SHADOW steering hook requires a residual runtime")
            if not source_text.strip():
                raise ValueError("SHADOW steering hook requires source_text")
        if runtime is not None and runtime.model_id != artifact.model_id:
            raise ValueError(
                "steering executor runtime model mismatch: "
                f"{runtime.model_id!r} != {artifact.model_id!r}"
            )
        if sensor_off_artifact is not None:
            if self.wiring_level is not WiringLevel.SHADOW:
                raise ValueError("sensor-off executor is evidence-only SHADOW")
            if (
                sensor_off_artifact.model_id != artifact.model_id
                or sensor_off_artifact.model_weights_sha256
                != artifact.model_weights_sha256
                or sensor_off_artifact.reader_artifact_id
                != artifact.reader_artifact_id
                or sensor_off_artifact.layer_index != artifact.layer_index
                or sensor_off_artifact.residual_width != artifact.residual_width
                or sensor_off_artifact.rank != artifact.rank
                or sensor_off_artifact.class_labels != artifact.class_labels
                or len(set(sensor_off_artifact.condition_codes)) != 1
            ):
                raise ValueError("sensor-off executor artifact is not matched")

    @property
    def artifact(self) -> SteeringExecutorArtifact:
        return self._artifact

    def _resolve_action(self, value: object) -> tuple[SteeringGateAction, int]:
        if isinstance(value, SteeringGateDecision):
            return value.action, value.policy_version
        if isinstance(value, RuntimePlaceholderValue):
            if self._ungated_action is None:
                raise ContractViolationError(
                    "steering executor cannot consume a missing/inactive gate "
                    "without an explicit promotion ungated_action"
                )
            return self._ungated_action, 1
        raise TypeError("steering executor requires SteeringGateDecision")

    def _residual(
        self,
        *,
        substrate: SubstrateSnapshot,
    ) -> tuple[float, ...]:
        artifact = self._artifact
        if not substrate.is_frozen:
            raise ValueError("steering executor requires a frozen substrate")
        if substrate.model_id != artifact.model_id:
            raise ValueError("steering executor substrate model drift")
        activations = tuple(
            activation.activation
            for activation in substrate.residual_activations
            if activation.layer_index == artifact.layer_index
        )
        if len(activations) != 1:
            raise ValueError(
                "steering executor requires exactly one source residual at "
                f"layer {artifact.layer_index}, got {len(activations)}"
            )
        if len(activations[0]) != artifact.residual_width:
            raise ValueError("steering executor residual width drift")
        return activations[0]

    def _compute_delta(
        self,
        *,
        artifact: SteeringExecutorArtifact,
        residual: tuple[float, ...],
        belief_index: int,
        action: SteeringGateAction,
    ) -> tuple[tuple[float, ...], float, float, float]:
        residual_norm = math.sqrt(sum(value * value for value in residual))
        if not math.isfinite(residual_norm) or residual_norm <= 0.0:
            raise ValueError("steering executor source residual norm must be positive")
        cap = artifact.control_norm_cap_ratio * residual_norm
        if action is SteeringGateAction.NOOP:
            return (
                tuple(0.0 for _ in range(artifact.residual_width)),
                residual_norm,
                0.0,
                cap,
            )
        if belief_index >= len(artifact.class_labels):
            raise ValueError("steering belief index exceeds executor code rows")
        projected = tuple(
            sum(
                residual[row] * artifact.v_factors[row][column]
                for row in range(artifact.residual_width)
            )
            for column in range(artifact.rank)
        )
        gated = tuple(
            math.tanh(artifact.condition_codes[belief_index][column])
            * projected[column]
            for column in range(artifact.rank)
        )
        raw_delta = tuple(
            sum(
                artifact.u_factors[row][column] * gated[column]
                for column in range(artifact.rank)
            )
            for row in range(artifact.residual_width)
        )
        raw_norm = math.sqrt(sum(value * value for value in raw_delta))
        if not math.isfinite(raw_norm):
            raise ValueError("steering executor produced a non-finite delta")
        scale = min(1.0, cap / raw_norm) if raw_norm > 0.0 else 0.0
        delta = tuple(value * scale for value in raw_delta)
        control_norm = raw_norm * scale
        return delta, residual_norm, control_norm, cap

    def _context(
        self,
        *,
        substrate: SubstrateSnapshot,
        conditioned: bool,
    ) -> SteeringResidualContext:
        activations = tuple(
            sorted(substrate.residual_activations, key=lambda row: row.layer_index)
        )
        if not activations:
            raise ValueError("steering context requires residual activations")
        layer_indices = tuple(row.layer_index for row in activations)
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError("steering context residual layers must be unique")
        widths = tuple(len(row.activation) for row in activations)
        flat = tuple(value for row in activations for value in row.activation)
        norm = math.sqrt(sum(value * value for value in flat))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("steering context residual norm must be positive")
        values = tuple(value / norm for value in flat)
        values_sha256 = hashlib.sha256(
            struct.pack(f"!{len(values)}d", *values)
        ).hexdigest()
        return SteeringResidualContext(
            source_sha256=hashlib.sha256(
                self._source_text.encode("utf-8")
            ).hexdigest(),
            layer_indices=layer_indices,
            activation_widths=widths,
            values=values,
            values_sha256=values_sha256,
            conditioned=conditioned,
        )

    def _execute(
        self,
        *,
        substrate: SubstrateSnapshot,
        belief: SteeringConditionBelief,
        gate_value: object,
    ) -> SteeringIntervention:
        artifact = self._artifact
        if belief.reader_artifact_id != artifact.reader_artifact_id:
            raise ValueError("steering reader/executor artifact lineage mismatch")
        if belief.belief_label != artifact.class_labels[belief.belief_index]:
            raise ValueError("steering belief label/index drift")
        action, policy_version = self._resolve_action(gate_value)
        residual = self._residual(substrate=substrate)
        delta, residual_norm, control_norm, cap = self._compute_delta(
            artifact=artifact,
            residual=residual,
            belief_index=belief.belief_index,
            action=action,
        )
        application_mode = "active-pending"
        hook_executed = False
        runtime_backend = "not-applied"
        downstream_effect: tuple[float, ...] = ()
        shadow_hook_latency_ms = 0.0
        noop_context = self._context(substrate=substrate, conditioned=False)
        action_context: SteeringResidualContext | None = None
        sensor_off_action_context: SteeringResidualContext | None = None
        sensor_off_control_norm = 0.0
        sensor_off_hook_latency_ms = 0.0
        if self.wiring_level is WiringLevel.SHADOW:
            application_mode = "shadow-compute-only"
            if action is SteeringGateAction.NOOP:
                application_mode = "shadow-noop"
                action_context = noop_context
            elif self._apply_shadow_hook:
                if self._runtime is None:
                    raise RuntimeError("shadow steering runtime disappeared")
                hook_started = time.perf_counter()
                application = self._runtime.apply_direct_residual_delta(
                    source_text=self._source_text,
                    substrate_snapshot=substrate,
                    layer_index=artifact.layer_index,
                    residual_delta=delta,
                )
                shadow_hook_latency_ms = (
                    time.perf_counter() - hook_started
                ) * 1000.0
                application_mode = "shadow-preview"
                hook_executed = True
                runtime_backend = application.backend_name
                downstream_effect = application.downstream_effect
                action_context = self._context(
                    substrate=application.applied_snapshot,
                    conditioned=True,
                )
                if self._sensor_off_artifact is not None:
                    (
                        sensor_off_delta,
                        _,
                        sensor_off_control_norm,
                        sensor_off_cap,
                    ) = self._compute_delta(
                        artifact=self._sensor_off_artifact,
                        residual=residual,
                        belief_index=0,
                        action=SteeringGateAction.STEER,
                    )
                    if not math.isclose(
                        sensor_off_cap, cap, rel_tol=1e-9, abs_tol=1e-9
                    ):
                        raise ValueError("sensor-off steering norm cap drift")
                    sensor_off_started = time.perf_counter()
                    sensor_off_application = (
                        self._runtime.apply_direct_residual_delta(
                            source_text=self._source_text,
                            substrate_snapshot=substrate,
                            layer_index=self._sensor_off_artifact.layer_index,
                            residual_delta=sensor_off_delta,
                        )
                    )
                    sensor_off_hook_latency_ms = (
                        time.perf_counter() - sensor_off_started
                    ) * 1000.0
                    sensor_off_action_context = self._context(
                        substrate=sensor_off_application.applied_snapshot,
                        conditioned=True,
                    )
        return SteeringIntervention(
            action=action,
            source_model_id=artifact.model_id,
            source_model_weights_sha256=artifact.model_weights_sha256,
            layer_index=artifact.layer_index,
            residual_delta=delta,
            residual_norm=residual_norm,
            control_norm=control_norm,
            control_norm_cap=cap,
            executor_artifact_id=artifact.artifact_id,
            reader_artifact_id=artifact.reader_artifact_id,
            gate_policy_version=policy_version,
            zero_code_noop=(
                action is SteeringGateAction.NOOP and control_norm <= 1e-12
            ),
            application_mode=application_mode,
            shadow_hook_executed=hook_executed,
            runtime_backend=runtime_backend,
            downstream_effect=downstream_effect,
            description=(
                "Frozen rank-r multiplicative residual intervention; no free "
                "bias and norm capped against the source residual."
            ),
            noop_context=noop_context,
            action_context=action_context,
            shadow_hook_latency_ms=shadow_hook_latency_ms,
            sensor_off_action_context=sensor_off_action_context,
            sensor_off_executor_artifact_id=(
                self._sensor_off_artifact.artifact_id
                if sensor_off_action_context is not None
                else ""
            ),
            sensor_off_control_norm=sensor_off_control_norm,
            sensor_off_shadow_hook_latency_ms=sensor_off_hook_latency_ms,
        )

    async def process(
        self,
        upstream: Mapping[str, Snapshot[object]],
    ) -> Snapshot[SteeringIntervention]:
        substrate_snapshot = upstream["substrate"]
        belief_snapshot = upstream[STEERING_CONDITION_BELIEF_SLOT]
        gate_snapshot = upstream[STEERING_GATE_DECISION_SLOT]
        if not isinstance(substrate_snapshot.value, SubstrateSnapshot):
            raise TypeError("steering executor requires SubstrateSnapshot")
        if not isinstance(belief_snapshot.value, SteeringConditionBelief):
            raise TypeError("steering executor requires SteeringConditionBelief")
        return self.publish(
            self._execute(
                substrate=substrate_snapshot.value,
                belief=belief_snapshot.value,
                gate_value=gate_snapshot.value,
            )
        )

    async def process_standalone(
        self,
        **kwargs: object,
    ) -> Snapshot[SteeringIntervention]:
        substrate = kwargs.get("substrate")
        belief = kwargs.get("belief")
        gate = kwargs.get("gate")
        if not isinstance(substrate, SubstrateSnapshot):
            raise TypeError("process_standalone requires substrate")
        if not isinstance(belief, SteeringConditionBelief):
            raise TypeError("process_standalone requires belief")
        return self.publish(
            self._execute(substrate=substrate, belief=belief, gate_value=gate)
        )


__all__ = ("SteeringExecutorModule",)
