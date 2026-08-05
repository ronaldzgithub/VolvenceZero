"""B3 steering-specific SHADOW -> ACTIVE promotion evidence and gate.

This is deliberately separate from ``learned_active_gate``: steering has
gate-off and sensor-off controls, not ETA-off/PE-off backends.  The evaluator
is pure and never flips runtime wiring; deployment changes one ordered
``FinalRolloutConfig`` field only after consuming its immutable verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import statistics

from volvence_zero.agent.dialogue_steering_evidence import (
    DialogueSteeringEffect,
    DialogueSteeringReport,
    DialogueSteeringTraceDataset,
)
from volvence_zero.steering_contracts import SteeringGateArtifact
from volvence_zero.steering_gate import SteeringGateModule


STEERING_PROMOTION_SCHEMA_VERSION = "steering-promotion-evidence.v1"


class SteeringComponent(str, Enum):
    SENSOR = "steering_sensor"
    EXECUTOR = "steering_executor"
    GATE = "steering_gate"


STEERING_PROMOTION_ORDER = (
    SteeringComponent.SENSOR,
    SteeringComponent.EXECUTOR,
    SteeringComponent.GATE,
)


@dataclass(frozen=True)
class SteeringPromotionThresholds:
    min_real_trace_turns: int = 500
    min_informative_axes: int = 2
    informative_std_floor: float = 0.01
    min_relative_validation_improvement: float = 0.15
    min_absolute_validation_improvement: float = 0.02
    min_gate_off_effect: float = 0.0
    min_sensor_off_effect: float = 0.0
    min_executor_on_effect: float = 0.0
    min_checkpoint_round_trips: int = 1
    max_p95_shadow_overhead_ratio: float = 1.0
    max_p95_end_to_end_latency_ms: float = 15_000.0


@dataclass(frozen=True)
class SteeringValidationAxis:
    name: str
    baseline_arm: str
    baseline_error: float
    learned_error: float
    target_std: float
    relative_improvement: float
    absolute_improvement: float
    informative: bool
    passed: bool


@dataclass(frozen=True)
class SteeringPromotionEvidence:
    schema_version: str
    preregistration_sha256: str
    c3_preregistration_sha256: str
    c3_report_sha256: str
    trace_sha256: str
    bundle_sha256: str
    real_trace_turns: int
    validation_axes: tuple[SteeringValidationAxis, ...]
    gate_off_vs_noop: DialogueSteeringEffect
    gate_off_vs_always_on: DialogueSteeringEffect
    executor_on_vs_noop: DialogueSteeringEffect
    sensor_off_conditional_advantage: DialogueSteeringEffect
    checkpoint_round_trips_verified: int
    checkpoint_json_round_trip_verified: bool
    p95_shadow_overhead_ratio: float
    p95_end_to_end_latency_ms: float
    safety_gate_ok: bool
    runtime_acceptance_all_passed: bool
    c3_admitted: bool
    reader_artifact_id: str
    executor_artifact_id: str
    sensor_off_executor_artifact_id: str
    candidate_gate_artifact: SteeringGateArtifact
    free_bias_present: bool
    zero_code_strict_noop: bool
    raw_text_retained: bool
    evaluation_writeback_allowed: bool
    production_default_changed: bool
    description: str

    def __post_init__(self) -> None:
        if self.schema_version != STEERING_PROMOTION_SCHEMA_VERSION:
            raise ValueError("steering promotion evidence schema is unsupported")
        for field_name, value in (
            ("preregistration_sha256", self.preregistration_sha256),
            ("c3_preregistration_sha256", self.c3_preregistration_sha256),
            ("c3_report_sha256", self.c3_report_sha256),
            ("trace_sha256", self.trace_sha256),
            ("bundle_sha256", self.bundle_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"steering promotion {field_name} is invalid")
        if self.real_trace_turns < 0 or self.checkpoint_round_trips_verified < 0:
            raise ValueError("steering promotion evidence counts are invalid")
        if not self.validation_axes:
            raise ValueError("steering promotion validation axes are empty")
        if not all(
            value.strip()
            for value in (
                self.reader_artifact_id,
                self.executor_artifact_id,
                self.sensor_off_executor_artifact_id,
                self.description,
            )
        ):
            raise ValueError("steering promotion artifact lineage is incomplete")
        for value in (
            self.p95_shadow_overhead_ratio,
            self.p95_end_to_end_latency_ms,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("steering promotion latency evidence is invalid")


@dataclass(frozen=True)
class SteeringComponentVerdict:
    component: SteeringComponent
    eligible: bool
    missing_gates: tuple[str, ...]
    rollback: str
    description: str


@dataclass(frozen=True)
class SteeringPromotionVerdict:
    eligible_prefix: tuple[SteeringComponent, ...]
    component_verdicts: tuple[SteeringComponentVerdict, ...]
    sensor_executor_active_authorized: bool
    gate_active_authorized: bool
    activation_order: tuple[SteeringComponent, ...]
    rollback_order: tuple[SteeringComponent, ...]
    blocking_reasons: tuple[str, ...]
    description: str


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _steer(artifact: SteeringGateArtifact, observations: tuple[tuple[str, float], ...]) -> bool:
    if tuple(name for name, _ in observations) != artifact.feature_names:
        raise ValueError("promotion gate observation schema drift")
    logits = tuple(
        artifact.bias[action]
        + sum(
            value * artifact.weights[index][action]
            for index, (_, value) in enumerate(observations)
        )
        for action in range(2)
    )
    return logits[1] > logits[0]


def _clustered_effect(
    *,
    dataset: DialogueSteeringTraceDataset,
    effects: tuple[float, ...],
    seed: int,
    resamples: int,
) -> DialogueSteeringEffect:
    rows = dataset.validation_rows
    if len(rows) != len(effects):
        raise ValueError("promotion clustered effect length drift")
    by_cluster: dict[str, list[float]] = {}
    for row, effect in zip(rows, effects, strict=True):
        by_cluster.setdefault(row.cluster_id, []).append(effect)
    import random

    rng = random.Random(seed)
    clusters = tuple(sorted(by_cluster))
    estimates = []
    for _ in range(resamples):
        sampled = tuple(clusters[rng.randrange(len(clusters))] for _ in clusters)
        estimates.append(
            statistics.fmean(
                value for cluster in sampled for value in by_cluster[cluster]
            )
        )
    estimates.sort()
    return DialogueSteeringEffect(
        mean=statistics.fmean(effects),
        ci_lower=estimates[max(0, int(0.025 * len(estimates)))],
        ci_upper=estimates[min(len(estimates) - 1, int(0.975 * len(estimates)))],
        cluster_count=len(clusters),
        row_count=len(rows),
    )


def verify_gate_checkpoint_round_trip(
    *,
    artifact: SteeringGateArtifact,
    observations: tuple[tuple[str, float], ...],
) -> tuple[int, bool]:
    gate = SteeringGateModule(
        artifact=artifact,
        learning_enabled=False,
    )
    before = gate.export_checkpoint(checkpoint_id="b3-rollback-drill")
    encoded = before.to_json()
    decoded = type(before).from_json(encoded)
    if decoded != before:
        raise RuntimeError("steering gate JSON checkpoint round-trip drift")
    gate.replay_observations(observations)
    gate.restore_checkpoint(decoded)
    after = gate.export_checkpoint(checkpoint_id="b3-rollback-drill")
    if after != before:
        raise RuntimeError("steering gate restore did not reproduce preimage")
    return 1, True


def build_steering_promotion_evidence(
    *,
    dataset: DialogueSteeringTraceDataset,
    c3_report: DialogueSteeringReport,
    preregistration_sha256: str,
    c3_report_sha256: str,
    trace_sha256: str,
    bundle_sha256: str,
    thresholds: SteeringPromotionThresholds | None = None,
    bootstrap_resamples: int = 5000,
) -> SteeringPromotionEvidence:
    active_thresholds = thresholds or SteeringPromotionThresholds()
    selected_point = min(
        c3_report.seed_points,
        key=lambda point: (point.selection_train_loss, point.seed),
    )
    artifact = selected_point.selected_gate_artifact
    rows = dataset.validation_rows
    flags = tuple(_steer(artifact, row.observations) for row in rows)
    noop_mse = tuple(row.normalized_noop_loss for row in rows)
    action_mse = tuple(row.normalized_steer_loss for row in rows)
    learned_mse = tuple(
        action if flag else noop
        for action, noop, flag in zip(action_mse, noop_mse, flags, strict=True)
    )
    noop_cosine_error = tuple(
        1.0 - row.terminal_prediction_error.noop_mean_cosine_similarity
        for row in rows
    )
    action_cosine_error = tuple(
        1.0 - row.terminal_prediction_error.action_mean_cosine_similarity
        for row in rows
    )
    learned_cosine_error = tuple(
        action if flag else noop
        for action, noop, flag in zip(
            action_cosine_error, noop_cosine_error, flags, strict=True
        )
    )

    def axis(
        name: str,
        train_noop_values: tuple[float, ...],
        train_action_values: tuple[float, ...],
        noop_values: tuple[float, ...],
        action_values: tuple[float, ...],
        learned_values: tuple[float, ...],
    ) -> SteeringValidationAxis:
        noop_train = statistics.fmean(train_noop_values)
        action_train = statistics.fmean(train_action_values)
        baseline_arm = "noop" if noop_train <= action_train else "always_on"
        baseline_values = noop_values if baseline_arm == "noop" else action_values
        baseline = statistics.fmean(baseline_values)
        learned = statistics.fmean(learned_values)
        target_std = statistics.pstdev(baseline_values)
        relative = (baseline - learned) / max(baseline, 1e-12)
        absolute = baseline - learned
        informative = target_std >= active_thresholds.informative_std_floor
        return SteeringValidationAxis(
            name=name,
            baseline_arm=baseline_arm,
            baseline_error=baseline,
            learned_error=learned,
            target_std=target_std,
            relative_improvement=relative,
            absolute_improvement=absolute,
            informative=informative,
            passed=(
                informative
                and (
                    relative
                    >= active_thresholds.min_relative_validation_improvement
                    or absolute
                    >= active_thresholds.min_absolute_validation_improvement
                )
            ),
        )

    train_noop_mse = tuple(
        row.normalized_noop_loss for row in dataset.train_rows
    )
    train_action_mse = tuple(
        row.normalized_steer_loss for row in dataset.train_rows
    )
    train_noop_cosine_error = tuple(
        1.0 - row.terminal_prediction_error.noop_mean_cosine_similarity
        for row in dataset.train_rows
    )
    train_action_cosine_error = tuple(
        1.0 - row.terminal_prediction_error.action_mean_cosine_similarity
        for row in dataset.train_rows
    )
    axes = (
        axis(
            "normalized_n_plus_one_mse",
            train_noop_mse,
            train_action_mse,
            noop_mse,
            action_mse,
            learned_mse,
        ),
        axis(
            "n_plus_one_cosine_error",
            train_noop_cosine_error,
            train_action_cosine_error,
            noop_cosine_error,
            action_cosine_error,
            learned_cosine_error,
        ),
    )
    sensor_off_losses = []
    conditional_losses = []
    for row in rows:
        if row.sensor_off_mean_squared_error is None:
            raise ValueError("B3 sensor-off counterfactual is missing")
        denominator = max(
            row.terminal_prediction_error.action_mean_squared_error,
            row.terminal_prediction_error.noop_mean_squared_error,
            row.sensor_off_mean_squared_error,
            1e-12,
        )
        sensor_off_losses.append(row.sensor_off_mean_squared_error / denominator)
        conditional_losses.append(
            row.terminal_prediction_error.action_mean_squared_error / denominator
        )
    sensor_effect = _clustered_effect(
        dataset=dataset,
        effects=tuple(
            sensor - conditional
            for sensor, conditional in zip(
                sensor_off_losses, conditional_losses, strict=True
            )
        ),
        seed=20260805,
        resamples=bootstrap_resamples,
    )
    executor_effect = _clustered_effect(
        dataset=dataset,
        effects=tuple(
            noop - action
            for noop, action in zip(noop_mse, action_mse, strict=True)
        ),
        seed=20260806,
        resamples=bootstrap_resamples,
    )
    checkpoint_count, json_round_trip = verify_gate_checkpoint_round_trip(
        artifact=artifact,
        observations=rows[0].observations,
    )
    overhead_ratios = tuple(
        row.shadow_hook_latency_ms
        / max(row.end_to_end_latency_ms - row.shadow_hook_latency_ms, 1e-9)
        for row in rows
    )
    return SteeringPromotionEvidence(
        schema_version=STEERING_PROMOTION_SCHEMA_VERSION,
        preregistration_sha256=preregistration_sha256,
        c3_preregistration_sha256=c3_report.preregistration_sha256,
        c3_report_sha256=c3_report_sha256,
        trace_sha256=trace_sha256,
        bundle_sha256=bundle_sha256,
        real_trace_turns=len(rows),
        validation_axes=axes,
        gate_off_vs_noop=selected_point.gain_vs_noop,
        gate_off_vs_always_on=selected_point.gain_vs_always_on,
        executor_on_vs_noop=executor_effect,
        sensor_off_conditional_advantage=sensor_effect,
        checkpoint_round_trips_verified=checkpoint_count,
        checkpoint_json_round_trip_verified=json_round_trip,
        p95_shadow_overhead_ratio=_percentile(overhead_ratios, 0.95),
        p95_end_to_end_latency_ms=_percentile(
            tuple(row.end_to_end_latency_ms for row in rows), 0.95
        ),
        safety_gate_ok=(
            not c3_report.free_bias_present
            and c3_report.zero_code_strict_noop
            and not c3_report.raw_text_retained
            and not c3_report.evaluation_writeback_allowed
        ),
        runtime_acceptance_all_passed=all(
            row.shadow_owner_chain_complete and row.shadow_hook_executed
            for row in rows
        ),
        c3_admitted=c3_report.admission.admitted,
        reader_artifact_id=rows[0].reader_artifact_id,
        executor_artifact_id=rows[0].executor_artifact_id,
        sensor_off_executor_artifact_id=(
            rows[0].sensor_off_executor_artifact_id
        ),
        candidate_gate_artifact=artifact,
        free_bias_present=c3_report.free_bias_present,
        zero_code_strict_noop=c3_report.zero_code_strict_noop,
        raw_text_retained=c3_report.raw_text_retained,
        evaluation_writeback_allowed=c3_report.evaluation_writeback_allowed,
        production_default_changed=False,
        description=(
            "Steering-specific real-soak, validation, gate-off, sensor-off, "
            "rollback, latency, and safety evidence."
        ),
    )


def evaluate_steering_promotion(
    evidence: SteeringPromotionEvidence,
    *,
    thresholds: SteeringPromotionThresholds | None = None,
) -> SteeringPromotionVerdict:
    active = thresholds or SteeringPromotionThresholds()
    informative = tuple(axis for axis in evidence.validation_axes if axis.informative)
    shared_missing = []
    if evidence.real_trace_turns < active.min_real_trace_turns:
        shared_missing.append("real_trace")
    if len(informative) < active.min_informative_axes or not all(
        axis.passed for axis in informative
    ):
        shared_missing.append("validation")
    if evidence.checkpoint_round_trips_verified < active.min_checkpoint_round_trips:
        shared_missing.append("rollback_drill")
    if not evidence.checkpoint_json_round_trip_verified:
        shared_missing.append("checkpoint_json_round_trip")
    if (
        evidence.p95_shadow_overhead_ratio
        > active.max_p95_shadow_overhead_ratio
        or evidence.p95_end_to_end_latency_ms
        > active.max_p95_end_to_end_latency_ms
    ):
        shared_missing.append("latency_slo")
    if not evidence.safety_gate_ok or not evidence.runtime_acceptance_all_passed:
        shared_missing.append("safety")
    if evidence.raw_text_retained or evidence.evaluation_writeback_allowed:
        shared_missing.append("r12_privacy")
    if evidence.free_bias_present or not evidence.zero_code_strict_noop:
        shared_missing.append("executor_structure")
    sensor_missing = list(shared_missing)
    sensor_effect = evidence.sensor_off_conditional_advantage
    if (
        sensor_effect.mean <= active.min_sensor_off_effect
        or sensor_effect.ci_lower <= 0.0
    ):
        sensor_missing.append("sensor_off")
    sensor_ok = not sensor_missing
    executor_missing = list(shared_missing)
    if not sensor_ok:
        executor_missing.append("prior_sensor_active")
    executor_effect = evidence.executor_on_vs_noop
    if (
        executor_effect.mean <= active.min_executor_on_effect
        or executor_effect.ci_lower <= 0.0
    ):
        executor_missing.append("executor_on_noop")
    executor_ok = not executor_missing
    gate_missing = list(shared_missing)
    if not executor_ok:
        gate_missing.append("prior_executor_active")
    for name, effect in (
        ("gate_off_noop", evidence.gate_off_vs_noop),
        ("gate_off_always_on", evidence.gate_off_vs_always_on),
    ):
        if effect.mean <= active.min_gate_off_effect or effect.ci_lower <= 0.0:
            gate_missing.append(name)
    if not evidence.c3_admitted:
        gate_missing.append("c3_admission")
    gate_ok = not gate_missing
    verdicts = (
        SteeringComponentVerdict(
            component=SteeringComponent.SENSOR,
            eligible=sensor_ok,
            missing_gates=tuple(sensor_missing),
            rollback="VZ_STEERING_SENSOR=shadow",
            description="First ordered steering promotion component.",
        ),
        SteeringComponentVerdict(
            component=SteeringComponent.EXECUTOR,
            eligible=executor_ok,
            missing_gates=tuple(executor_missing),
            rollback="VZ_STEERING_EXECUTOR=shadow",
            description="Second ordered steering promotion component.",
        ),
        SteeringComponentVerdict(
            component=SteeringComponent.GATE,
            eligible=gate_ok,
            missing_gates=tuple(gate_missing),
            rollback="VZ_STEERING_GATE=shadow",
            description="Final ordered steering promotion component.",
        ),
    )
    prefix = []
    for verdict in verdicts:
        if not verdict.eligible:
            break
        prefix.append(verdict.component)
    blocking = tuple(
        f"{verdict.component.value}:{reason}"
        for verdict in verdicts
        for reason in verdict.missing_gates
    )
    return SteeringPromotionVerdict(
        eligible_prefix=tuple(prefix),
        component_verdicts=verdicts,
        sensor_executor_active_authorized=(
            tuple(prefix[:2])
            == (SteeringComponent.SENSOR, SteeringComponent.EXECUTOR)
        ),
        gate_active_authorized=(
            tuple(prefix) == STEERING_PROMOTION_ORDER
        ),
        activation_order=STEERING_PROMOTION_ORDER,
        rollback_order=tuple(reversed(STEERING_PROMOTION_ORDER)),
        blocking_reasons=blocking,
        description=(
            "Steering ACTIVE prefix authorized in sensor->executor->gate order."
            if prefix
            else "Steering remains SHADOW; no ordered prefix passed every gate."
        ),
    )


__all__ = (
    "STEERING_PROMOTION_ORDER",
    "STEERING_PROMOTION_SCHEMA_VERSION",
    "SteeringComponent",
    "SteeringComponentVerdict",
    "SteeringPromotionEvidence",
    "SteeringPromotionThresholds",
    "SteeringPromotionVerdict",
    "SteeringValidationAxis",
    "build_steering_promotion_evidence",
    "evaluate_steering_promotion",
    "verify_gate_checkpoint_round_trip",
)
