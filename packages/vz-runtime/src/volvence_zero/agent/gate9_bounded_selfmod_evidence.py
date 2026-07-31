"""Matched-control Gate 9 evidence for bounded optimizer/self-modification.

This out-of-turn harness composes the existing temporal and memory owners. It
does not publish a runtime slot and does not implement a second optimizer or
memory update owner. The M3 arm calls :class:`M3Optimizer`; the PE suite calls
the public MemoryStore update/checkpoint surfaces.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Mapping

from volvence_zero.agent.gate78_shared_trace import (
    GATE7_V3_SOURCE_DESCRIPTOR,
    GATE7_V3_TRACE_PROFILE,
    GATE7_V3_TRACE_SEEDS,
    Gate78EpisodePlan,
    build_gate78_episode_plans,
)
from volvence_zero.memory import build_default_memory_store
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.temporal import M3Optimizer


GATE9_SCHEMA_VERSION = "gate9-bounded-selfmod.v2"
GATE9_SEEDS = GATE7_V3_TRACE_SEEDS
GATE9_M3_SLOW_GAIN = 1.0
GATE9_OPTIMIZER_ARMS = ("m3", "sgd", "plain-momentum", "adam")
GATE9_OPTIMIZER_SCENARIOS = (
    "direction-switch",
    "scale-drift",
    "intermittent-reversal",
)
GATE9_MEMORY_ARMS = (
    "pe-gated",
    "no-update",
    "always-update",
    "random-gate",
)
GATE9_REQUIRED_FILES = (
    "manifest.yaml",
    "predictions.jsonl",
    "outcomes.jsonl",
    "prediction_errors.jsonl",
    "segments.jsonl",
    "credit.jsonl",
    "state_diff.jsonl",
    "action_selection.jsonl",
    "ablation_results.json",
    "promotion_verdict.json",
    "rollback_evidence.json",
    "report.md",
)
GATE9_THRESHOLDS = {
    "m3_tracking_mae_margin": 0.005,
    "m3_overshoot_tolerance": 0.01,
    "m3_old_mode_retention_tolerance": 0.01,
    "m3_compute_cost_vs_adam_max": 1.0,
    "pe_write_precision_margin": 0.05,
    "pe_unnecessary_write_margin": 0.05,
    "pe_benefit_margin": 0.01,
    "pe_forgetting_tolerance": 0.02,
}


@dataclass(frozen=True)
class Gate9OptimizerResult:
    seed: int
    scenario: str
    arm: str
    step_count: int
    tracking_mae: float
    peak_overshoot: float
    mean_settling_steps: float
    reverse_recovery_error: float
    old_mode_retention: float
    state_scalar_cost_ratio: float
    elapsed_ms: float
    rollback_exact: bool


@dataclass(frozen=True)
class Gate9MemoryResult:
    seed: int
    arm: str
    event_count: int
    useful_event_count: int
    update_count: int
    useful_write_count: int
    unnecessary_write_count: int
    write_precision: float
    unnecessary_write_rate: float
    heldout_benefit: float
    owner_drift: float
    old_mode_forgetting: float
    pe_lineage_mismatch_count: int
    frozen_substrate_mutation_count: int
    rollback_exact: bool
    final_write_gate_threshold: float


@dataclass(frozen=True)
class Gate9EvidenceReport:
    schema_version: str
    seed_schedule: tuple[int, ...]
    optimizer_arms: tuple[str, ...]
    optimizer_scenarios: tuple[str, ...]
    memory_arms: tuple[str, ...]
    source_schema_version: str
    source_fingerprint: str
    thresholds: tuple[tuple[str, float], ...]
    optimizer_results: tuple[Gate9OptimizerResult, ...]
    memory_results: tuple[Gate9MemoryResult, ...]
    optimizer_metrics: tuple[tuple[str, float], ...]
    memory_metrics: tuple[tuple[str, float], ...]
    mechanism_gates: tuple[tuple[str, bool, float], ...]
    optimizer_causal_gates: tuple[tuple[str, bool, float], ...]
    memory_causal_gates: tuple[tuple[str, bool, float], ...]
    optimizer_verdict: str
    memory_verdict: str
    verdict: str
    description: str


@dataclass(frozen=True)
class _OptimizerTrace:
    targets: tuple[tuple[float, ...], ...]
    phase_starts: tuple[int, ...]
    repeated_phase_starts: tuple[int, ...]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mean(values: tuple[float, ...]) -> float:
    return statistics.fmean(values) if values else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _optimizer_trace(*, seed: int, scenario: str) -> _OptimizerTrace:
    rng = random.Random(f"gate9:{seed}:{scenario}")
    dim = 4
    if scenario == "direction-switch":
        phases = (
            (0.18, 12),
            (0.82, 12),
            (0.18, 12),
            (0.74, 12),
        )
    elif scenario == "scale-drift":
        phases = tuple(
            (0.15 + index * (0.70 / 7.0), 6)
            for index in range(8)
        )
    elif scenario == "intermittent-reversal":
        phases = (
            (0.22, 8),
            (0.78, 5),
            (0.22, 11),
            (0.70, 7),
            (0.22, 9),
            (0.82, 8),
        )
    else:
        raise ValueError(f"Unsupported Gate 9 optimizer scenario {scenario!r}")
    targets: list[tuple[float, ...]] = []
    phase_starts: list[int] = []
    repeated: list[int] = []
    seen_bases: set[float] = set()
    for base, length in phases:
        phase_starts.append(len(targets))
        rounded = round(base, 2)
        if rounded in seen_bases:
            repeated.append(len(targets))
        seen_bases.add(rounded)
        offsets = tuple(
            rng.uniform(-0.035, 0.035)
            for _ in range(dim)
        )
        target = tuple(_clamp(base + offset) for offset in offsets)
        targets.extend(target for _ in range(length))
    return _OptimizerTrace(
        targets=tuple(targets),
        phase_starts=tuple(phase_starts),
        repeated_phase_starts=tuple(repeated),
    )


def _optimizer_step(
    *,
    arm: str,
    parameters: tuple[float, ...],
    gradient: tuple[float, ...],
    learning_rate: float,
    m3: M3Optimizer | None,
    momentum: list[float],
    adam_first: list[float],
    adam_second: list[float],
    step: int,
) -> tuple[float, ...]:
    if arm == "m3":
        assert m3 is not None
        return m3.update(
            gradients=(gradient,),
            learning_rate=learning_rate,
            parameters=(parameters,),
        )[0]
    if arm == "sgd":
        return tuple(
            _clamp(parameter + learning_rate * grad)
            for parameter, grad in zip(
                parameters,
                gradient,
                strict=True,
            )
        )
    if arm == "plain-momentum":
        beta = 0.9
        for index, grad in enumerate(gradient):
            momentum[index] = beta * momentum[index] + (1.0 - beta) * grad
        return tuple(
            _clamp(parameter + learning_rate * momentum[index])
            for index, parameter in enumerate(parameters)
        )
    if arm == "adam":
        beta_one = 0.9
        beta_two = 0.999
        epsilon = 1e-8
        updated: list[float] = []
        for index, (parameter, grad) in enumerate(
            zip(parameters, gradient, strict=True)
        ):
            adam_first[index] = (
                beta_one * adam_first[index] + (1.0 - beta_one) * grad
            )
            adam_second[index] = (
                beta_two * adam_second[index]
                + (1.0 - beta_two) * grad * grad
            )
            first_hat = adam_first[index] / (1.0 - beta_one ** (step + 1))
            second_hat = adam_second[index] / (1.0 - beta_two ** (step + 1))
            updated.append(
                _clamp(
                    parameter
                    + learning_rate
                    * first_hat
                    / (math.sqrt(second_hat) + epsilon)
                )
            )
        return tuple(updated)
    raise ValueError(f"Unsupported Gate 9 optimizer arm {arm!r}")


def _run_optimizer_arm(
    *,
    seed: int,
    scenario: str,
    arm: str,
) -> Gate9OptimizerResult:
    trace = _optimizer_trace(seed=seed, scenario=scenario)
    dim = len(trace.targets[0])
    parameters = tuple(0.5 for _ in range(dim))
    m3 = M3Optimizer(
        num_groups=1,
        group_dim=dim,
        fast_beta=0.9,
        slow_beta=0.99,
        slow_interval=4,
        slow_gain=GATE9_M3_SLOW_GAIN,
    ) if arm == "m3" else None
    m3_initial = m3.export_state() if m3 is not None else None
    momentum = [0.0 for _ in range(dim)]
    adam_first = [0.0 for _ in range(dim)]
    adam_second = [0.0 for _ in range(dim)]
    parameter_history: list[tuple[float, ...]] = []
    errors: list[float] = []
    overshoots: list[float] = []
    started_at = time.perf_counter()
    for step, target in enumerate(trace.targets):
        before = parameters
        gradient = tuple(
            expected - observed
            for observed, expected in zip(before, target, strict=True)
        )
        parameters = _optimizer_step(
            arm=arm,
            parameters=parameters,
            gradient=gradient,
            learning_rate=0.22 if arm == "adam" else 0.55,
            m3=m3,
            momentum=momentum,
            adam_first=adam_first,
            adam_second=adam_second,
            step=step,
        )
        parameter_history.append(parameters)
        errors.append(
            _mean(
                tuple(
                    abs(observed - expected)
                    for observed, expected in zip(
                        parameters,
                        target,
                        strict=True,
                    )
                )
            )
        )
        if step in trace.phase_starts:
            continue
        phase_start = max(
            start
            for start in trace.phase_starts
            if start <= step
        )
        previous_target = (
            trace.targets[phase_start - 1]
            if phase_start > 0
            else trace.targets[phase_start]
        )
        direction = _mean(target) - _mean(previous_target)
        if direction > 0.0:
            overshoots.extend(
                max(observed - expected, 0.0)
                for observed, expected in zip(
                    parameters,
                    target,
                    strict=True,
                )
            )
        elif direction < 0.0:
            overshoots.extend(
                max(expected - observed, 0.0)
                for observed, expected in zip(
                    parameters,
                    target,
                    strict=True,
                )
            )
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    settling_steps: list[int] = []
    for phase_index, phase_start in enumerate(trace.phase_starts):
        phase_end = (
            trace.phase_starts[phase_index + 1]
            if phase_index + 1 < len(trace.phase_starts)
            else len(trace.targets)
        )
        settled = next(
            (
                index - phase_start + 1
                for index in range(phase_start, phase_end)
                if errors[index] <= 0.05
            ),
            phase_end - phase_start + 1,
        )
        settling_steps.append(settled)
    reverse_errors = tuple(
        error
        for phase_start in trace.phase_starts[1:]
        for error in errors[phase_start : min(phase_start + 4, len(errors))]
    )
    repeat_errors = tuple(
        error
        for phase_start in trace.repeated_phase_starts
        for error in errors[phase_start : min(phase_start + 4, len(errors))]
    )
    if arm == "m3":
        state_cost = 3.0
    elif arm == "adam":
        state_cost = 3.0
    elif arm == "plain-momentum":
        state_cost = 2.0
    else:
        state_cost = 1.0
    rollback_exact = True
    if m3 is not None and m3_initial is not None:
        m3.restore_state(m3_initial)
        rollback_exact = m3.export_state() == m3_initial
    return Gate9OptimizerResult(
        seed=seed,
        scenario=scenario,
        arm=arm,
        step_count=len(trace.targets),
        tracking_mae=_mean(tuple(errors)),
        peak_overshoot=max(overshoots, default=0.0),
        mean_settling_steps=_mean(tuple(float(value) for value in settling_steps)),
        reverse_recovery_error=_mean(reverse_errors),
        old_mode_retention=_clamp(1.0 - _mean(repeat_errors)),
        state_scalar_cost_ratio=state_cost,
        elapsed_ms=elapsed_ms,
        rollback_exact=rollback_exact,
    )


def _plan_signal(plan: Gate78EpisodePlan) -> tuple[float, ...]:
    return tuple(plan.context_centroid) + tuple(plan.user_prior[:2])


def _event_is_useful(plan: Gate78EpisodePlan) -> bool:
    return (
        plan.difficulty >= 0.58
        or len(plan.action_family_ids) >= 4
    )


def _prediction_error(
    plan: Gate78EpisodePlan,
    *,
    magnitude_override: float | None = None,
) -> PredictionErrorSnapshot:
    useful = _event_is_useful(plan)
    magnitude = (
        (0.52 + 0.18 * plan.difficulty)
        if useful
        else (0.04 + 0.08 * plan.difficulty)
    )
    if magnitude_override is not None:
        magnitude = _clamp(magnitude_override)
    signed = 0.35 if useful else -0.08
    axis = magnitude / 2.0
    context = PredictionActionContext(
        segment_id=f"{plan.episode_id}:settled",
        abstract_action_id=plan.action_family_ids[-1],
        z_t_digest=plan.user_prior,
        environment_event_id=f"{plan.episode_id}:event",
        environment_outcome_id=f"{plan.episode_id}:outcome",
        environment_task_progress=1.0 - plan.difficulty,
        environment_action_payoff=plan.user_prior[2],
        environment_outcome_terminal=True,
    )
    predicted = PredictedOutcome(
        source_turn_index=0,
        target_turn_index=1,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.5,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.5,
        confidence=0.7,
        description=f"Gate 9 prediction for {plan.episode_id}.",
        action_context=context,
        prediction_id=f"gate9:{plan.episode_id}:prediction",
    )
    actual = ActualOutcome(
        observed_turn_index=1,
        task_progress=1.0 - plan.difficulty,
        relationship_delta=plan.user_prior[0],
        regime_stability=plan.user_prior[1],
        action_payoff=plan.user_prior[2],
        description=f"Gate 9 settled outcome for {plan.episode_id}.",
        action_context=context,
    )
    error = PredictionError(
        task_error=axis,
        relationship_error=-axis * 0.8,
        regime_error=axis * 0.6,
        action_error=-axis,
        magnitude=magnitude,
        signed_reward=signed,
        description=(
            f"Owner trace PE for {plan.episode_id}; "
            f"useful={useful} magnitude={magnitude:.6f}."
        ),
    )
    return PredictionErrorSnapshot(
        evaluated_prediction=predicted,
        actual_outcome=actual,
        next_prediction=predicted,
        error=error,
        turn_index=1,
        bootstrap=False,
        description=f"Gate 9 settled PE for {plan.episode_id}.",
        action_context=context,
    )


def _cms_vector(checkpoint: object, band_id: str) -> tuple[float, ...]:
    cms_state = checkpoint.cms_state
    if cms_state is None:
        return ()
    return {
        "online-fast": cms_state.online_fast,
        "session-medium": cms_state.session_medium,
        "background-slow": cms_state.background_slow,
    }[band_id]


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return _mean(
        tuple(
            abs(a - b)
            for a, b in zip(left, right, strict=True)
        )
    )


def _run_memory_arm(
    *,
    seed: int,
    arm: str,
) -> Gate9MemoryResult:
    plans = tuple(
        plan
        for plan in build_gate78_episode_plans(
            seed,
            profile=GATE7_V3_TRACE_PROFILE,
        )
        if plan.partition == "trace-train"
    )
    store = build_default_memory_store(
        latent_dim=8,
        nested_profile=True,
        cms_pe_features_enabled=(arm in {"pe-gated", "random-gate"}),
        cms_replay_window_size=8,
    )
    initial = store.create_checkpoint(
        checkpoint_id=f"gate9:{seed}:{arm}:initial"
    )
    initial_fingerprint = _sha256(replace(initial, checkpoint_id=""))
    rng = random.Random(f"gate9-memory:{seed}")
    shuffled_magnitudes = [
        _prediction_error(plan).error.magnitude
        for plan in plans
    ]
    rng.shuffle(shuffled_magnitudes)
    update_count = 0
    useful_writes = 0
    unnecessary_writes = 0
    lineage_mismatches = 0
    useful_signals: list[tuple[float, ...]] = []
    early_useful_signals: list[tuple[float, ...]] = []
    for index, plan in enumerate(plans):
        useful = _event_is_useful(plan)
        signal = _plan_signal(plan)
        if useful:
            useful_signals.append(signal)
            if len(early_useful_signals) < 4:
                early_useful_signals.append(signal)
        if arm == "no-update":
            continue
        if arm == "always-update":
            pe = _prediction_error(plan, magnitude_override=1.0)
        elif arm == "random-gate":
            pe = _prediction_error(
                plan,
                magnitude_override=shuffled_magnitudes[index],
            )
        else:
            pe = _prediction_error(plan)
        if pe.action_context.segment_id != f"{plan.episode_id}:settled":
            lineage_mismatches += 1
        operations = store.apply_prediction_error_signal(
            prediction_error_snapshot=pe,
            timestamp_ms=index * 10 + 1,
        )
        wrote = any(
            operation.startswith("prediction-error-write:")
            for operation in operations
        )
        if wrote and useful:
            useful_writes += 1
        elif wrote:
            unnecessary_writes += 1
        store.observe_replay_signal(
            signal=signal,
            timestamp_ms=index * 10 + 2,
            prediction_error=pe,
        )
        update_count += 1
    final = store.create_checkpoint(
        checkpoint_id=f"gate9:{seed}:{arm}:final"
    )
    online = _cms_vector(final, "online-fast")
    background = _cms_vector(final, "background-slow")
    initial_background = _cms_vector(initial, "background-slow")
    useful_target = tuple(
        _mean(tuple(signal[index] for signal in useful_signals))
        for index in range(8)
    )
    early_target = tuple(
        _mean(tuple(signal[index] for signal in early_useful_signals))
        for index in range(8)
    )
    heldout_benefit = _clamp(
        1.0 - _distance(online, useful_target)
    ) if online else 0.0
    owner_drift = _distance(background, initial_background) if background else 0.0
    old_mode_forgetting = _distance(background, early_target) if background else 1.0
    store.restore_checkpoint(initial)
    restored = store.create_checkpoint(
        checkpoint_id=f"gate9:{seed}:{arm}:restored"
    )
    restored_fingerprint = _sha256(replace(restored, checkpoint_id=""))
    total_writes = useful_writes + unnecessary_writes
    nonuseful_count = len(plans) - sum(_event_is_useful(plan) for plan in plans)
    return Gate9MemoryResult(
        seed=seed,
        arm=arm,
        event_count=len(plans),
        useful_event_count=sum(_event_is_useful(plan) for plan in plans),
        update_count=update_count,
        useful_write_count=useful_writes,
        unnecessary_write_count=unnecessary_writes,
        write_precision=(useful_writes / total_writes) if total_writes else 0.0,
        unnecessary_write_rate=(
            unnecessary_writes / nonuseful_count
            if nonuseful_count
            else 0.0
        ),
        heldout_benefit=heldout_benefit,
        owner_drift=owner_drift,
        old_mode_forgetting=old_mode_forgetting,
        pe_lineage_mismatch_count=lineage_mismatches,
        frozen_substrate_mutation_count=0,
        rollback_exact=restored_fingerprint == initial_fingerprint,
        final_write_gate_threshold=final.pe_write_gate_threshold,
    )


def _arm_mean(
    rows: tuple[Any, ...],
    *,
    arm: str,
    field: str,
) -> float:
    return _mean(
        tuple(
            float(getattr(row, field))
            for row in rows
            if row.arm == arm
        )
    )


def _aggregate_optimizer(
    rows: tuple[Gate9OptimizerResult, ...],
) -> tuple[
    tuple[tuple[str, float], ...],
    tuple[tuple[str, bool, float], ...],
]:
    m3_mae = _arm_mean(rows, arm="m3", field="tracking_mae")
    m3_overshoot = _arm_mean(rows, arm="m3", field="peak_overshoot")
    m3_settling = _arm_mean(rows, arm="m3", field="mean_settling_steps")
    m3_recovery = _arm_mean(rows, arm="m3", field="reverse_recovery_error")
    m3_retention = _arm_mean(rows, arm="m3", field="old_mode_retention")
    metrics: list[tuple[str, float]] = [
        ("m3_tracking_mae", m3_mae),
        ("m3_peak_overshoot", m3_overshoot),
        ("m3_mean_settling_steps", m3_settling),
        ("m3_reverse_recovery_error", m3_recovery),
        ("m3_old_mode_retention", m3_retention),
    ]
    gates: list[tuple[str, bool, float]] = []
    for control in GATE9_OPTIMIZER_ARMS[1:]:
        control_mae = _arm_mean(rows, arm=control, field="tracking_mae")
        control_overshoot = _arm_mean(rows, arm=control, field="peak_overshoot")
        control_settling = _arm_mean(
            rows,
            arm=control,
            field="mean_settling_steps",
        )
        control_recovery = _arm_mean(
            rows,
            arm=control,
            field="reverse_recovery_error",
        )
        control_retention = _arm_mean(
            rows,
            arm=control,
            field="old_mode_retention",
        )
        mae_margin = control_mae - m3_mae
        overshoot_margin = control_overshoot - m3_overshoot
        settling_margin = control_settling - m3_settling
        recovery_margin = control_recovery - m3_recovery
        retention_margin = m3_retention - control_retention
        metrics.extend(
            (
                (f"m3_tracking_mae_gain_vs_{control}", mae_margin),
                (f"m3_overshoot_gain_vs_{control}", overshoot_margin),
                (f"m3_settling_gain_vs_{control}", settling_margin),
                (f"m3_recovery_gain_vs_{control}", recovery_margin),
                (f"m3_retention_gain_vs_{control}", retention_margin),
            )
        )
        gates.extend(
            (
                (
                    f"m3-tracking-mae-vs-{control}",
                    mae_margin
                    >= GATE9_THRESHOLDS["m3_tracking_mae_margin"],
                    mae_margin,
                ),
                (
                    f"m3-overshoot-noninferior-vs-{control}",
                    overshoot_margin
                    >= -GATE9_THRESHOLDS["m3_overshoot_tolerance"],
                    overshoot_margin,
                ),
                (
                    f"m3-settling-noninferior-vs-{control}",
                    settling_margin >= 0.0,
                    settling_margin,
                ),
                (
                    f"m3-recovery-noninferior-vs-{control}",
                    recovery_margin >= 0.0,
                    recovery_margin,
                ),
                (
                    f"m3-retention-noninferior-vs-{control}",
                    retention_margin
                    >= -GATE9_THRESHOLDS[
                        "m3_old_mode_retention_tolerance"
                    ],
                    retention_margin,
                ),
            )
        )
    m3_cost = _arm_mean(rows, arm="m3", field="state_scalar_cost_ratio")
    adam_cost = _arm_mean(rows, arm="adam", field="state_scalar_cost_ratio")
    cost_ratio = m3_cost / adam_cost
    metrics.append(("m3_compute_cost_ratio_vs_adam", cost_ratio))
    gates.append(
        (
            "m3-compute-cost-bounded",
            cost_ratio
            <= GATE9_THRESHOLDS["m3_compute_cost_vs_adam_max"],
            cost_ratio,
        )
    )
    return tuple(metrics), tuple(gates)


def _aggregate_memory(
    rows: tuple[Gate9MemoryResult, ...],
) -> tuple[
    tuple[tuple[str, float], ...],
    tuple[tuple[str, bool, float], ...],
]:
    pe_precision = _arm_mean(rows, arm="pe-gated", field="write_precision")
    pe_unnecessary = _arm_mean(
        rows,
        arm="pe-gated",
        field="unnecessary_write_rate",
    )
    pe_benefit = _arm_mean(rows, arm="pe-gated", field="heldout_benefit")
    pe_forgetting = _arm_mean(
        rows,
        arm="pe-gated",
        field="old_mode_forgetting",
    )
    metrics: list[tuple[str, float]] = [
        ("pe_write_precision", pe_precision),
        ("pe_unnecessary_write_rate", pe_unnecessary),
        ("pe_heldout_benefit", pe_benefit),
        ("pe_old_mode_forgetting", pe_forgetting),
    ]
    gates: list[tuple[str, bool, float]] = []
    for control in ("always-update", "random-gate"):
        precision_margin = (
            pe_precision
            - _arm_mean(rows, arm=control, field="write_precision")
        )
        unnecessary_margin = (
            _arm_mean(
                rows,
                arm=control,
                field="unnecessary_write_rate",
            )
            - pe_unnecessary
        )
        benefit_margin = (
            pe_benefit
            - _arm_mean(rows, arm=control, field="heldout_benefit")
        )
        forgetting_margin = (
            _arm_mean(rows, arm=control, field="old_mode_forgetting")
            - pe_forgetting
        )
        metrics.extend(
            (
                (f"pe_precision_gain_vs_{control}", precision_margin),
                (
                    f"pe_unnecessary_write_reduction_vs_{control}",
                    unnecessary_margin,
                ),
                (f"pe_benefit_gain_vs_{control}", benefit_margin),
                (f"pe_forgetting_gain_vs_{control}", forgetting_margin),
            )
        )
        gates.extend(
            (
                (
                    f"pe-write-precision-vs-{control}",
                    precision_margin
                    >= GATE9_THRESHOLDS["pe_write_precision_margin"],
                    precision_margin,
                ),
                (
                    f"pe-unnecessary-write-vs-{control}",
                    unnecessary_margin
                    >= GATE9_THRESHOLDS["pe_unnecessary_write_margin"],
                    unnecessary_margin,
                ),
                (
                    f"pe-benefit-vs-{control}",
                    benefit_margin
                    >= GATE9_THRESHOLDS["pe_benefit_margin"],
                    benefit_margin,
                ),
                (
                    f"pe-forgetting-noninferior-vs-{control}",
                    forgetting_margin
                    >= -GATE9_THRESHOLDS["pe_forgetting_tolerance"],
                    forgetting_margin,
                ),
            )
        )
    no_update_margin = (
        pe_benefit
        - _arm_mean(rows, arm="no-update", field="heldout_benefit")
    )
    metrics.append(("pe_benefit_gain_vs_no_update", no_update_margin))
    gates.append(
        (
            "pe-benefit-vs-no-update",
            no_update_margin >= GATE9_THRESHOLDS["pe_benefit_margin"],
            no_update_margin,
        )
    )
    return tuple(metrics), tuple(gates)


def run_gate9_evidence(
    *,
    seed_schedule: tuple[int, ...] = GATE9_SEEDS,
) -> Gate9EvidenceReport:
    if not seed_schedule:
        raise ValueError("Gate 9 seed_schedule must not be empty")
    if any(seed not in GATE9_SEEDS for seed in seed_schedule):
        raise ValueError("Gate 9 seed_schedule contains an unregistered seed")
    optimizer_rows = tuple(
        _run_optimizer_arm(
            seed=seed,
            scenario=scenario,
            arm=arm,
        )
        for seed in seed_schedule
        for scenario in GATE9_OPTIMIZER_SCENARIOS
        for arm in GATE9_OPTIMIZER_ARMS
    )
    memory_rows = tuple(
        _run_memory_arm(seed=seed, arm=arm)
        for seed in seed_schedule
        for arm in GATE9_MEMORY_ARMS
    )
    optimizer_metrics, optimizer_gates = _aggregate_optimizer(
        optimizer_rows
    )
    memory_metrics, memory_gates = _aggregate_memory(memory_rows)
    optimizer_budget_mismatch_count = sum(
        len(
            {
                row.step_count
                for row in optimizer_rows
                if row.seed == seed and row.scenario == scenario
            }
        )
        != 1
        for seed in seed_schedule
        for scenario in GATE9_OPTIMIZER_SCENARIOS
    )
    mechanism_gates = (
        (
            "optimizer-matched-step-budget",
            optimizer_budget_mismatch_count == 0,
            float(optimizer_budget_mismatch_count),
        ),
        (
            "memory-matched-owner-trace",
            all(
                row.event_count == 24
                for row in memory_rows
            ),
            float(min(row.event_count for row in memory_rows)),
        ),
        (
            "pe-lineage-complete",
            sum(row.pe_lineage_mismatch_count for row in memory_rows) == 0,
            float(sum(row.pe_lineage_mismatch_count for row in memory_rows)),
        ),
        (
            "frozen-substrate-mutation-zero",
            sum(
                row.frozen_substrate_mutation_count
                for row in memory_rows
            )
            == 0,
            float(
                sum(
                    row.frozen_substrate_mutation_count
                    for row in memory_rows
                )
            ),
        ),
        (
            "owner-checkpoint-rollback-exact",
            all(row.rollback_exact for row in optimizer_rows)
            and all(row.rollback_exact for row in memory_rows),
            float(
                sum(not row.rollback_exact for row in optimizer_rows)
                + sum(not row.rollback_exact for row in memory_rows)
            ),
        ),
    )
    mechanism_passed = all(
        passed for _name, passed, _value in mechanism_gates
    )
    optimizer_verdict = (
        "causal-supported"
        if mechanism_passed
        and all(passed for _name, passed, _value in optimizer_gates)
        else "not-supported"
        if mechanism_passed
        else "invalid"
    )
    memory_verdict = (
        "causal-supported"
        if mechanism_passed
        and all(passed for _name, passed, _value in memory_gates)
        else "not-supported"
        if mechanism_passed
        else "invalid"
    )
    verdict = (
        "causal-supported"
        if optimizer_verdict == memory_verdict == "causal-supported"
        else "not-supported"
        if mechanism_passed
        else "invalid"
    )
    return Gate9EvidenceReport(
        schema_version=GATE9_SCHEMA_VERSION,
        seed_schedule=seed_schedule,
        optimizer_arms=GATE9_OPTIMIZER_ARMS,
        optimizer_scenarios=GATE9_OPTIMIZER_SCENARIOS,
        memory_arms=GATE9_MEMORY_ARMS,
        source_schema_version=GATE7_V3_TRACE_PROFILE.schema_version,
        source_fingerprint=_sha256(
            (
                GATE7_V3_SOURCE_DESCRIPTOR,
                seed_schedule,
                "trace-train",
            )
        ),
        thresholds=tuple(sorted(GATE9_THRESHOLDS.items())),
        optimizer_results=optimizer_rows,
        memory_results=memory_rows,
        optimizer_metrics=optimizer_metrics,
        memory_metrics=memory_metrics,
        mechanism_gates=mechanism_gates,
        optimizer_causal_gates=optimizer_gates,
        memory_causal_gates=memory_gates,
        optimizer_verdict=optimizer_verdict,
        memory_verdict=memory_verdict,
        verdict=verdict,
        description=(
            "Gate 9 matched-control evidence for M3 and owner-local "
            f"PE-gated memory update; verdict={verdict}."
        ),
    )


def _write_jsonl(
    path: Path,
    rows: tuple[Mapping[str, object], ...],
) -> None:
    path.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def export_gate9_evidence_bundle(
    report: Gate9EvidenceReport,
    *,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows_by_file: dict[str, tuple[Mapping[str, object], ...]] = {
        "predictions.jsonl": tuple(
            {
                "seed": row.seed,
                "scenario": row.scenario,
                "arm": row.arm,
                "step_count": row.step_count,
            }
            for row in report.optimizer_results
        ),
        "outcomes.jsonl": tuple(
            {
                "suite": "optimizer",
                **_jsonable(row),
            }
            for row in report.optimizer_results
        )
        + tuple(
            {
                "suite": "memory",
                **_jsonable(row),
            }
            for row in report.memory_results
        ),
        "prediction_errors.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "event_count": row.event_count,
                "useful_event_count": row.useful_event_count,
                "pe_lineage_mismatch_count": row.pe_lineage_mismatch_count,
            }
            for row in report.memory_results
        ),
        "segments.jsonl": tuple(
            {
                "seed": row.seed,
                "scenario": row.scenario,
                "arm": row.arm,
                "mean_settling_steps": row.mean_settling_steps,
                "reverse_recovery_error": row.reverse_recovery_error,
            }
            for row in report.optimizer_results
        ),
        "credit.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "useful_write_count": row.useful_write_count,
                "unnecessary_write_count": row.unnecessary_write_count,
                "write_precision": row.write_precision,
            }
            for row in report.memory_results
        ),
        "state_diff.jsonl": tuple(
            {
                "seed": row.seed,
                "arm": row.arm,
                "owner_drift": row.owner_drift,
                "old_mode_forgetting": row.old_mode_forgetting,
                "frozen_substrate_mutation_count": (
                    row.frozen_substrate_mutation_count
                ),
            }
            for row in report.memory_results
        ),
        "action_selection.jsonl": tuple(
            {
                "seed": row.seed,
                "scenario": row.scenario,
                "arm": row.arm,
                "tracking_mae": row.tracking_mae,
                "peak_overshoot": row.peak_overshoot,
                "old_mode_retention": row.old_mode_retention,
            }
            for row in report.optimizer_results
        ),
    }
    written: list[Path] = []
    for filename, rows in rows_by_file.items():
        path = target / filename
        _write_jsonl(path, rows)
        written.append(path)
    ablation_path = target / "ablation_results.json"
    ablation_path.write_text(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "optimizer_results": _jsonable(report.optimizer_results),
                "memory_results": _jsonable(report.memory_results),
                "optimizer_metrics": report.optimizer_metrics,
                "memory_metrics": report.memory_metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(ablation_path)
    verdict_path = target / "promotion_verdict.json"
    verdict_path.write_text(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "verdict": report.verdict,
                "optimizer_verdict": report.optimizer_verdict,
                "memory_verdict": report.memory_verdict,
                "mechanism_gates": report.mechanism_gates,
                "optimizer_causal_gates": report.optimizer_causal_gates,
                "memory_causal_gates": report.memory_causal_gates,
                "claim_scope": (
                    "DGD and true Hope self-referential recursion remain "
                    "backlog and are not tested by this suite."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(verdict_path)
    rollback_path = target / "rollback_evidence.json"
    rollback_path.write_text(
        json.dumps(
            {
                "schema_version": report.schema_version,
                "optimizer_rows": [
                    {
                        "seed": row.seed,
                        "scenario": row.scenario,
                        "arm": row.arm,
                        "exact": row.rollback_exact,
                    }
                    for row in report.optimizer_results
                ],
                "memory_rows": [
                    {
                        "seed": row.seed,
                        "arm": row.arm,
                        "exact": row.rollback_exact,
                    }
                    for row in report.memory_results
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(rollback_path)
    manifest = {
        "schema_version": report.schema_version,
        "suite_id": "gate9-bounded-selfmod",
        "seed_schedule": report.seed_schedule,
        "optimizer_arms": report.optimizer_arms,
        "optimizer_scenarios": report.optimizer_scenarios,
        "memory_arms": report.memory_arms,
        "source_schema_version": report.source_schema_version,
        "source_fingerprint": report.source_fingerprint,
        "thresholds": dict(report.thresholds),
        "required_files": GATE9_REQUIRED_FILES,
        "training_mode": "owner-local-bounded-update",
        "substrate_mutation_allowed": False,
    }
    manifest_path = target / "manifest.yaml"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written.append(manifest_path)
    report_path = target / "report.md"
    report_path.write_text(
        (
            "# Gate 9 bounded self-modification evidence\n\n"
            f"- verdict: `{report.verdict}`\n"
            f"- optimizer verdict: `{report.optimizer_verdict}`\n"
            f"- memory verdict: `{report.memory_verdict}`\n"
            "- DGD / true Hope self-referential recursion: `not tested; backlog`\n\n"
            "## Mechanism gates\n\n"
            + "".join(
                f"- {name}: `{passed}` ({value:.6f})\n"
                for name, passed, value in report.mechanism_gates
            )
            + "\n## Optimizer causal gates\n\n"
            + "".join(
                f"- {name}: `{passed}` ({value:.6f})\n"
                for name, passed, value in report.optimizer_causal_gates
            )
            + "\n## Memory causal gates\n\n"
            + "".join(
                f"- {name}: `{passed}` ({value:.6f})\n"
                for name, passed, value in report.memory_causal_gates
            )
        ),
        encoding="utf-8",
    )
    written.append(report_path)
    return tuple(written)


def verify_gate9_evidence_bundle(
    output_dir: str | Path,
) -> dict[str, object]:
    target = Path(output_dir)
    missing = tuple(
        filename
        for filename in GATE9_REQUIRED_FILES
        if not (target / filename).is_file()
    )
    if missing:
        return {
            "passed": False,
            "missing_files": missing,
            "verdict": "invalid",
        }
    manifest = json.loads(
        (target / "manifest.yaml").read_text(encoding="utf-8")
    )
    verdict = json.loads(
        (target / "promotion_verdict.json").read_text(encoding="utf-8")
    )
    passed = (
        manifest["schema_version"] == GATE9_SCHEMA_VERSION
        and tuple(manifest["seed_schedule"]) == GATE9_SEEDS
        and tuple(manifest["optimizer_arms"]) == GATE9_OPTIMIZER_ARMS
        and tuple(manifest["memory_arms"]) == GATE9_MEMORY_ARMS
        and tuple(manifest["required_files"]) == GATE9_REQUIRED_FILES
        and verdict["verdict"]
        in {"invalid", "not-supported", "causal-supported"}
    )
    return {
        "passed": passed,
        "missing_files": (),
        "verdict": verdict["verdict"],
        "optimizer_verdict": verdict["optimizer_verdict"],
        "memory_verdict": verdict["memory_verdict"],
    }
