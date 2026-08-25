from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from random import Random
import json
import math
import statistics

from volvence_zero.agent.eta_proof_benchmark import (
    ETAOpenWeightRuntimeConfig,
    ETAProofCase,
    _build_eta_open_weight_runtime,
    _build_sandbox,
    _calibrate_case_for_real_snapshots,
    _profile_config,
    _runtime_capture_snapshot,
    _snapshot_from_step,
    _validate_eta_open_weight_runtime,
    build_default_eta_proof_environment,
    default_eta_proof_cases,
)
from volvence_zero.internal_rl import HierarchicalRouteSpec, ZRollout
from volvence_zero.memory import Track
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    ExpertActionTarget,
    OpenWeightResidualRuntime,
    SubstrateSnapshot,
    build_training_trace,
)
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerParameterStore,
    MetacontrollerSSLTrainer,
    SSLBatchTrainingReport,
    build_training_trace_from_substrate_snapshots,
)


_TRAINING_MODES = ("ssl-rl-alternating", "rl-only")


@dataclass(frozen=True)
class SegmentCreditEvent:
    run_seed: int
    case_id: str
    split: str
    event_id: str
    assignment_reason: str
    subgoal_id: str
    causal_start_step: int
    causal_end_step: int
    observed_outcome_step: int
    observation_lag: int
    eta_credit_start_step: int
    eta_credit_end_step: int
    eta_credit_steps: tuple[int, ...]
    turn_credit_steps: tuple[int, ...]
    true_credit_steps: tuple[int, ...]
    true_family_id: str
    eta_family_id: str
    turn_family_id: str
    outcome_value: float
    eta_precision: float
    eta_recall: float
    eta_f1: float
    eta_false_credit_rate: float
    turn_precision: float
    turn_recall: float
    turn_f1: float
    turn_false_credit_rate: float


@dataclass(frozen=True)
class SegmentBoundaryCase:
    run_seed: int
    case_id: str
    split: str
    true_boundaries: tuple[int, ...]
    beta_boundaries: tuple[int, ...]
    active_family_ids: tuple[str, ...]
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class SegmentCreditRunMetrics:
    run_seed: int
    event_count: int
    heldout_event_count: int
    eta_credit_f1: float
    turn_credit_f1: float
    credit_f1_delta: float
    eta_false_credit_rate: float
    turn_false_credit_rate: float
    false_credit_reduction: float
    eta_family_assignment_accuracy: float
    turn_family_assignment_accuracy: float
    family_assignment_delta: float
    eta_heldout_pe: float
    turn_heldout_pe: float
    eta_pe_reduction_rate: float
    turn_pe_reduction_rate: float
    pe_reduction_rate_delta: float
    segment_boundary_f1: float
    active_family_count: int
    learned_family_bank_count: int
    beta_boundary_count: int
    true_boundary_count: int
    training_cycle_count: int
    ssl_trajectory_count: int
    ssl_trained_step_count: int
    ssl_prediction_loss_mean: float
    ssl_kl_loss_mean: float
    ssl_total_loss_mean: float
    ssl_switch_frequency_mean: float
    ssl_switch_frequency_final: float
    ssl_switch_probability_mean: float
    ssl_switch_probability_final: float
    ssl_switch_rate_loss_mean: float
    ssl_gate_choice_loss_mean: float
    ssl_keep_prediction_loss_mean: float
    ssl_switch_prediction_loss_mean: float
    ssl_target_variance_mean: float
    ssl_action_boundary_f1_final: float
    ssl_boundary_switch_probability_final: float
    ssl_continuation_switch_probability_final: float
    ssl_boundary_switch_preference_final: float
    ssl_continuation_switch_preference_final: float
    ssl_switch_threshold_final: float
    runtime_switch_threshold_final: float
    ssl_supervision_targets: tuple[str, ...]
    ssl_expert_action_supervision: bool
    ssl_optimizer_final_step: int
    ssl_optimizer_reuse_count: int
    ssl_parameter_change_count: int
    ssl_writeback_count: int


@dataclass(frozen=True)
class MetricInterval:
    metric_name: str
    mean: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class SegmentCreditEvidenceReport:
    schema_version: str
    backend_label: str
    model_id: str
    device: str
    runtime_origin: str
    fallback_active: bool
    seed_schedule: tuple[int, ...]
    controller_initialization_seed: int
    experience_seed_semantics: str
    training_mode: str
    training_cycles: int
    ssl_updates_per_cycle: int
    controller_dim: int
    ssl_alpha: float
    switch_prior: float
    switch_rate_weight: float
    switch_binary_weight: float
    switch_group_weight: float
    proposal_prediction_weight: float
    gate_choice_weight: float
    gate_choice_temperature: float
    prediction_horizon: int
    distortion_target: str
    ssl_supervision_target: str
    expert_action_supervision: bool
    outcome_target: str
    family_truth_source: str
    family_mapping_fit_split: str
    causal_family_manifold_projection: bool
    rollout_replacement_mode: str
    temporal_fast_prior_enabled: bool
    episode_recurrent_state_isolated: bool
    run_metrics: tuple[SegmentCreditRunMetrics, ...]
    metric_intervals: tuple[MetricInterval, ...]
    claim_status: str
    retain_gates: tuple[tuple[str, bool, float], ...]
    events: tuple[SegmentCreditEvent, ...]
    boundary_cases: tuple[SegmentBoundaryCase, ...]
    description: str


def _mean(values: tuple[float, ...]) -> float:
    return statistics.fmean(values) if values else 0.0


def _expert_action_observation_bundle(
    case: ETAProofCase,
    *,
    open_weight_runtime: OpenWeightResidualRuntime | None,
) -> tuple[
    tuple[SubstrateSnapshot, ...],
    tuple[str, ...],
    tuple[ExpertActionTarget, ...],
]:
    """Render one frozen observation per expert controller-action phase."""

    environment = build_default_eta_proof_environment()
    route = HierarchicalRouteSpec(
        case_id=case.case_id,
        split=case.split,
        source_text=case.source_text,
        waypoints=case.route_signature,
        split_detail=case.split_detail,
        description=case.description,
    )
    target_ids = {
        transition.target_id for transition in environment.transitions
    }
    action_vocabulary = tuple(
        location.location_id
        for location in environment.locations
        if location.location_id in target_ids
    )
    state = environment.reset(route)
    observation_texts: list[str] = []
    expert_targets: list[ExpertActionTarget] = []
    for target_id in route.waypoints[1:]:
        observation = environment.observe(state)
        target_location = environment.location(target_id)
        phase_count = (
            max(target_location.min_persistence, 1)
            if target_location.is_objective
            else 1
        )
        action_index = action_vocabulary.index(target_id)
        action_values = tuple(
            1.0 if index == action_index else 0.0
            for index in range(len(action_vocabulary))
        )
        for phase_index in range(phase_count):
            observation_texts.append(
                "Task context: "
                f"{case.source_text}. Available transitions: "
                f"{', '.join(observation.available_targets)}. Current location: "
                f"{observation.current_location_id}. Remaining route: "
                f"{' -> '.join(observation.remaining_route)}."
            )
            expert_targets.append(
                ExpertActionTarget(
                    action_id=f"move:{target_id}",
                    values=action_values,
                    source="eta-proof-environment-demonstration",
                    description=(
                        f"Expert transition from {state.current_location_id} "
                        f"to {target_id}; phase {phase_index + 1}/{phase_count}."
                    ),
                )
            )
        state = environment.step(state, target_id=target_id).next_state
    if len(observation_texts) < 2:
        raise ValueError(
            f"ETA expert route {case.case_id!r} must publish at least two phases."
        )
    if open_weight_runtime is None:
        snapshots = tuple(
            _snapshot_from_step(
                f"{case.case_id}:expert:{step_index}",
                build_training_trace(
                    trace_id=f"{case.case_id}:expert:{step_index}",
                    source_text=source_text,
                ).steps[-1],
            )
            for step_index, source_text in enumerate(observation_texts)
        )
    else:
        snapshots = tuple(
            _runtime_capture_snapshot(
                runtime=open_weight_runtime,
                case=case,
                source_text=source_text,
                step_index=step_index,
                total_steps=len(observation_texts),
            )
            for step_index, source_text in enumerate(observation_texts)
        )
    normalized_snapshots = tuple(
        replace(
            snapshot,
            residual_sequence=(),
            description=(
                f"{snapshot.description} Normalized to one environment-level "
                "residual step for SSL/runtime temporal parity."
            ),
        )
        for snapshot in snapshots
    )
    return (
        normalized_snapshots,
        tuple(observation_texts),
        tuple(expert_targets),
    )


def _safe_f1(predicted: set[int], expected: set[int]) -> tuple[float, float, float, float]:
    if not predicted and not expected:
        return (1.0, 1.0, 1.0, 0.0)
    overlap = len(predicted & expected)
    precision = overlap / max(len(predicted), 1)
    recall = overlap / max(len(expected), 1)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    false_credit_rate = len(predicted - expected) / max(len(predicted), 1)
    return (precision, recall, f1, false_credit_rate)


def _boundary_f1(
    *,
    predicted: tuple[int, ...],
    expected: tuple[int, ...],
    tolerance: int = 1,
) -> tuple[float, float, float]:
    remaining = list(expected)
    matched = 0
    for candidate in predicted:
        matches = tuple(
            (abs(candidate - target), index)
            for index, target in enumerate(remaining)
            if abs(candidate - target) <= tolerance
        )
        if not matches:
            continue
        _, matched_index = min(matches)
        remaining.pop(matched_index)
        matched += 1
    precision = matched / max(len(predicted), 1)
    recall = matched / max(len(expected), 1)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return (precision, recall, f1)


def _abstract_action_windows(rollout: ZRollout) -> tuple[tuple[int, int], ...]:
    if not rollout.transitions:
        return ()
    windows: list[tuple[int, int]] = []
    start = 0
    for index, transition in enumerate(rollout.transitions):
        if index > 0 and transition.controller_state.is_switching:
            windows.append((start, index - 1))
            start = index
    windows.append((start, len(rollout.transitions) - 1))
    return tuple(windows)


def _dominant_family(rollout: ZRollout, steps: tuple[int, ...]) -> str:
    counts: dict[str, int] = {}
    for step in steps:
        family_id = rollout.transitions[step].active_family_id or "unassigned"
        counts[family_id] = counts.get(family_id, 0) + 1
    if not counts:
        return "unassigned"
    return min(counts, key=lambda family_id: (-counts[family_id], family_id))


def _dominant_expert_action(
    expert_targets: tuple[ExpertActionTarget, ...],
    steps: tuple[int, ...],
) -> str:
    counts: dict[str, int] = {}
    for step in steps:
        action_id = expert_targets[step].action_id
        counts[action_id] = counts.get(action_id, 0) + 1
    if not counts:
        return "unassigned"
    return min(counts, key=lambda action_id: (-counts[action_id], action_id))


def _event_rows(
    *,
    rollout: ZRollout,
    case: ETAProofCase,
    seed: int,
    max_observation_lag: int,
    expert_targets: tuple[ExpertActionTarget, ...],
) -> tuple[SegmentCreditEvent, ...]:
    if len(expert_targets) != len(rollout.transitions):
        raise ValueError(
            "Event evaluation requires one expert action target per rollout "
            f"transition; got {len(expert_targets)} targets for "
            f"{len(rollout.transitions)} transitions."
        )
    windows = _abstract_action_windows(rollout)
    rows: list[SegmentCreditEvent] = []
    for assignment_index, assignment in enumerate(rollout.delayed_credit_assignments):
        if assignment.reason != "subgoal-complete":
            continue
        causal_start = max(0, assignment.start_step)
        causal_end = min(len(rollout.transitions) - 1, assignment.end_step)
        if causal_end < causal_start or causal_end >= len(rollout.transitions) - 1:
            continue
        available_lag = len(rollout.transitions) - 1 - causal_end
        lag_cap = max(1, min(max_observation_lag, available_lag))
        lag = 1 + ((seed + assignment_index) % lag_cap)
        observed_step = causal_end + lag
        overlapping = tuple(
            (start, end)
            for start, end in windows
            if end >= causal_start and start <= causal_end
        )
        if overlapping:
            eta_start = min(start for start, _ in overlapping)
            eta_end = max(end for _, end in overlapping)
        else:
            eta_start, eta_end = causal_start, causal_end
        true_steps = tuple(range(causal_start, causal_end + 1))
        eta_steps = tuple(range(eta_start, eta_end + 1))
        turn_steps = (observed_step,)
        eta_precision, eta_recall, eta_f1, eta_false = _safe_f1(
            set(eta_steps), set(true_steps)
        )
        turn_precision, turn_recall, turn_f1, turn_false = _safe_f1(
            set(turn_steps), set(true_steps)
        )
        true_family = _dominant_expert_action(expert_targets, true_steps)
        outcome_value = max(-1.0, min(1.0, assignment.completion_margin))
        rows.append(
            SegmentCreditEvent(
                run_seed=seed,
                case_id=case.case_id,
                split=case.split,
                event_id=f"{seed}:{case.case_id}:{assignment_index}",
                assignment_reason=assignment.reason,
                subgoal_id=assignment.subgoal_id or "",
                causal_start_step=causal_start,
                causal_end_step=causal_end,
                observed_outcome_step=observed_step,
                observation_lag=lag,
                eta_credit_start_step=eta_start,
                eta_credit_end_step=eta_end,
                eta_credit_steps=eta_steps,
                turn_credit_steps=turn_steps,
                true_credit_steps=true_steps,
                true_family_id=true_family,
                eta_family_id=_dominant_family(rollout, eta_steps),
                turn_family_id=_dominant_family(rollout, turn_steps),
                outcome_value=outcome_value,
                eta_precision=eta_precision,
                eta_recall=eta_recall,
                eta_f1=eta_f1,
                eta_false_credit_rate=eta_false,
                turn_precision=turn_precision,
                turn_recall=turn_recall,
                turn_f1=turn_f1,
                turn_false_credit_rate=turn_false,
            )
        )
    return tuple(rows)


def _boundary_row(
    *,
    rollout: ZRollout,
    case: ETAProofCase,
    seed: int,
    expert_targets: tuple[ExpertActionTarget, ...],
) -> SegmentBoundaryCase:
    if len(expert_targets) != len(rollout.transitions):
        raise ValueError(
            "Boundary evaluation requires one expert action target per rollout "
            f"transition; got {len(expert_targets)} targets for "
            f"{len(rollout.transitions)} transitions."
        )
    true_boundaries = tuple(
        index
        for index in range(1, len(expert_targets))
        if expert_targets[index].action_id
        != expert_targets[index - 1].action_id
    )
    beta_boundaries = tuple(
        transition.step_index
        for transition in rollout.transitions[1:]
        if transition.controller_state.is_switching
    )
    active_family_ids = tuple(
        sorted(
            {
                transition.active_family_id
                for transition in rollout.transitions
                if transition.active_family_id not in {None, "unassigned"}
            }
        )
    )
    precision, recall, f1 = _boundary_f1(
        predicted=beta_boundaries,
        expected=true_boundaries,
    )
    return SegmentBoundaryCase(
        run_seed=seed,
        case_id=case.case_id,
        split=case.split,
        true_boundaries=true_boundaries,
        beta_boundaries=beta_boundaries,
        active_family_ids=active_family_ids,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _fit_family_values(
    events: tuple[SegmentCreditEvent, ...],
    *,
    family_field: str,
) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for event in events:
        family_id = str(getattr(event, family_field))
        if family_id == "unassigned":
            continue
        values.setdefault(family_id, []).append(event.outcome_value)
    return {family_id: _mean(tuple(outcomes)) for family_id, outcomes in values.items()}


def _fit_family_action_mapping(
    events: tuple[SegmentCreditEvent, ...],
    *,
    family_field: str,
) -> dict[str, str]:
    counts: dict[str, dict[str, int]] = {}
    for event in events:
        family_id = str(getattr(event, family_field))
        if family_id == "unassigned":
            continue
        action_counts = counts.setdefault(family_id, {})
        action_counts[event.true_family_id] = (
            action_counts.get(event.true_family_id, 0) + 1
        )
    return {
        family_id: min(
            action_counts,
            key=lambda action_id: (-action_counts[action_id], action_id),
        )
        for family_id, action_counts in counts.items()
    }


def _family_action_accuracy(
    events: tuple[SegmentCreditEvent, ...],
    *,
    family_field: str,
    mapping: dict[str, str],
) -> float:
    return _mean(
        tuple(
            1.0
            if mapping.get(str(getattr(event, family_field)))
            == event.true_family_id
            else 0.0
            for event in events
        )
    )


def _prediction_metrics(
    *,
    train_events: tuple[SegmentCreditEvent, ...],
    heldout_events: tuple[SegmentCreditEvent, ...],
    family_field: str,
) -> tuple[float, float]:
    family_values = _fit_family_values(train_events, family_field=family_field)
    if not heldout_events:
        return (0.0, 0.0)
    errors = tuple(
        abs(
            event.outcome_value
            - family_values.get(str(getattr(event, family_field)), 0.0)
        )
        for event in heldout_events
    )
    baseline_errors = tuple(abs(event.outcome_value) for event in heldout_events)
    mean_error = _mean(errors)
    baseline_error = _mean(baseline_errors)
    reduction = (
        (baseline_error - mean_error) / baseline_error
        if baseline_error > 1e-12
        else 0.0
    )
    return (mean_error, reduction)


def _run_metrics(
    *,
    seed: int,
    events: tuple[SegmentCreditEvent, ...],
    boundaries: tuple[SegmentBoundaryCase, ...],
    training_cycle_count: int,
    ssl_reports: tuple[SSLBatchTrainingReport, ...],
    runtime_switch_threshold: float,
    learned_family_bank_count: int,
) -> SegmentCreditRunMetrics:
    heldout_events = tuple(event for event in events if event.split == "heldout")
    train_events = tuple(event for event in events if event.split == "train")
    metric_events = heldout_events
    eta_pe, eta_reduction = _prediction_metrics(
        train_events=train_events,
        heldout_events=metric_events,
        family_field="eta_family_id",
    )
    turn_pe, turn_reduction = _prediction_metrics(
        train_events=train_events,
        heldout_events=metric_events,
        family_field="turn_family_id",
    )
    boundary_eval = tuple(row for row in boundaries if row.split != "train")
    eta_f1 = _mean(tuple(event.eta_f1 for event in metric_events))
    turn_f1 = _mean(tuple(event.turn_f1 for event in metric_events))
    eta_false = _mean(tuple(event.eta_false_credit_rate for event in metric_events))
    turn_false = _mean(tuple(event.turn_false_credit_rate for event in metric_events))
    eta_family = _family_action_accuracy(
        metric_events,
        family_field="eta_family_id",
        mapping=_fit_family_action_mapping(
            train_events,
            family_field="eta_family_id",
        ),
    )
    turn_family = _family_action_accuracy(
        metric_events,
        family_field="turn_family_id",
        mapping=_fit_family_action_mapping(
            train_events,
            family_field="turn_family_id",
        ),
    )
    return SegmentCreditRunMetrics(
        run_seed=seed,
        event_count=len(events),
        heldout_event_count=len(heldout_events),
        eta_credit_f1=eta_f1,
        turn_credit_f1=turn_f1,
        credit_f1_delta=eta_f1 - turn_f1,
        eta_false_credit_rate=eta_false,
        turn_false_credit_rate=turn_false,
        false_credit_reduction=turn_false - eta_false,
        eta_family_assignment_accuracy=eta_family,
        turn_family_assignment_accuracy=turn_family,
        family_assignment_delta=eta_family - turn_family,
        eta_heldout_pe=eta_pe,
        turn_heldout_pe=turn_pe,
        eta_pe_reduction_rate=eta_reduction,
        turn_pe_reduction_rate=turn_reduction,
        pe_reduction_rate_delta=eta_reduction - turn_reduction,
        segment_boundary_f1=_mean(tuple(row.f1 for row in boundary_eval)),
        beta_boundary_count=sum(len(row.beta_boundaries) for row in boundaries),
        true_boundary_count=sum(len(row.true_boundaries) for row in boundaries),
        active_family_count=len(
            {
                family_id
                for row in boundaries
                for family_id in row.active_family_ids
            }
        ),
        learned_family_bank_count=learned_family_bank_count,
        training_cycle_count=training_cycle_count,
        ssl_trajectory_count=sum(
            report.trajectory_count for report in ssl_reports
        ),
        ssl_trained_step_count=sum(
            report.trained_step_count for report in ssl_reports
        ),
        ssl_prediction_loss_mean=_mean(
            tuple(report.torch_prediction_loss for report in ssl_reports)
        ),
        ssl_kl_loss_mean=_mean(
            tuple(report.torch_kl_loss for report in ssl_reports)
        ),
        ssl_total_loss_mean=_mean(
            tuple(report.torch_total_loss for report in ssl_reports)
        ),
        ssl_switch_frequency_mean=_mean(
            tuple(report.torch_binary_switch_ratio for report in ssl_reports)
        ),
        ssl_switch_frequency_final=(
            ssl_reports[-1].torch_binary_switch_ratio
            if ssl_reports
            else 0.0
        ),
        ssl_switch_probability_mean=_mean(
            tuple(
                report.torch_mean_switch_probability
                for report in ssl_reports
            )
        ),
        ssl_switch_probability_final=(
            ssl_reports[-1].torch_mean_switch_probability
            if ssl_reports
            else 0.0
        ),
        ssl_switch_rate_loss_mean=_mean(
            tuple(report.torch_switch_rate_loss for report in ssl_reports)
        ),
        ssl_gate_choice_loss_mean=_mean(
            tuple(report.torch_gate_choice_loss for report in ssl_reports)
        ),
        ssl_keep_prediction_loss_mean=_mean(
            tuple(report.torch_keep_prediction_loss for report in ssl_reports)
        ),
        ssl_switch_prediction_loss_mean=_mean(
            tuple(
                report.torch_switch_prediction_loss
                for report in ssl_reports
            )
        ),
        ssl_target_variance_mean=_mean(
            tuple(report.torch_target_variance for report in ssl_reports)
        ),
        ssl_action_boundary_f1_final=(
            ssl_reports[-1].torch_expert_action_boundary_f1
            if ssl_reports
            else 0.0
        ),
        ssl_boundary_switch_probability_final=(
            ssl_reports[-1].torch_boundary_switch_probability
            if ssl_reports
            else 0.0
        ),
        ssl_continuation_switch_probability_final=(
            ssl_reports[-1].torch_continuation_switch_probability
            if ssl_reports
            else 0.0
        ),
        ssl_boundary_switch_preference_final=(
            ssl_reports[-1].torch_boundary_switch_preference
            if ssl_reports
            else 0.0
        ),
        ssl_continuation_switch_preference_final=(
            ssl_reports[-1].torch_continuation_switch_preference
            if ssl_reports
            else 0.0
        ),
        ssl_switch_threshold_final=(
            ssl_reports[-1].torch_switch_threshold_after
            if ssl_reports
            else 0.0
        ),
        runtime_switch_threshold_final=runtime_switch_threshold,
        ssl_supervision_targets=tuple(
            sorted(
                {
                    report.torch_supervision_target
                    for report in ssl_reports
                }
            )
        ),
        ssl_expert_action_supervision=(
            bool(ssl_reports)
            and all(
                report.torch_expert_action_supervision
                for report in ssl_reports
            )
        ),
        ssl_optimizer_final_step=max(
            (report.torch_optimizer_step for report in ssl_reports),
            default=0,
        ),
        ssl_optimizer_reuse_count=sum(
            1 for report in ssl_reports if report.torch_optimizer_state_reused
        ),
        ssl_parameter_change_count=sum(
            report.torch_parameters_changed for report in ssl_reports
        ),
        ssl_writeback_count=sum(
            1 for report in ssl_reports if report.torch_wrote_back
        ),
    )


def _bootstrap_interval(
    values: tuple[float, ...],
    *,
    seed: int = 1701,
    samples: int = 2000,
) -> tuple[float, float]:
    if len(values) <= 1:
        value = values[0] if values else 0.0
        return (value, value)
    random = Random(seed)
    means = sorted(
        _mean(tuple(values[random.randrange(len(values))] for _ in values))
        for _ in range(samples)
    )
    low_index = int(0.025 * (samples - 1))
    high_index = int(0.975 * (samples - 1))
    return (means[low_index], means[high_index])


def _metric_intervals(
    run_metrics: tuple[SegmentCreditRunMetrics, ...],
) -> tuple[MetricInterval, ...]:
    names = (
        "eta_credit_f1",
        "turn_credit_f1",
        "credit_f1_delta",
        "eta_false_credit_rate",
        "turn_false_credit_rate",
        "false_credit_reduction",
        "eta_family_assignment_accuracy",
        "turn_family_assignment_accuracy",
        "family_assignment_delta",
        "eta_heldout_pe",
        "turn_heldout_pe",
        "eta_pe_reduction_rate",
        "turn_pe_reduction_rate",
        "pe_reduction_rate_delta",
        "segment_boundary_f1",
    )
    intervals: list[MetricInterval] = []
    for index, name in enumerate(names):
        values = tuple(float(getattr(row, name)) for row in run_metrics)
        ci_low, ci_high = _bootstrap_interval(values, seed=1701 + index)
        intervals.append(
            MetricInterval(
                metric_name=name,
                mean=_mean(values),
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
    return tuple(intervals)


def run_eta_segment_credit_evidence(
    *,
    seed_schedule: tuple[int, ...] = tuple(range(5)),
    backend_label: str = "transformers-open-weight",
    open_weight_config: ETAOpenWeightRuntimeConfig | None = None,
    max_observation_lag: int = 3,
    training_mode: str = "ssl-rl-alternating",
    training_cycles: int = 3,
    ssl_updates_per_cycle: int = 1,
    controller_dim: int = 16,
    ssl_alpha: float = 0.1,
    switch_prior: float = 0.10,
    switch_rate_weight: float = 0.05,
    switch_binary_weight: float = 0.01,
    switch_group_weight: float = 0.01,
    proposal_prediction_weight: float = 0.50,
    gate_choice_weight: float = 1.0,
    gate_choice_temperature: float = 0.02,
    prediction_horizon: int = 3,
    distortion_target: str = "innovation",
) -> SegmentCreditEvidenceReport:
    if not seed_schedule:
        raise ValueError("seed_schedule must contain at least one seed")
    if backend_label not in {"transformers-open-weight", "trace"}:
        raise ValueError(f"Unsupported backend_label {backend_label!r}")
    if training_mode not in _TRAINING_MODES:
        raise ValueError(
            f"Unsupported training_mode {training_mode!r}; expected one of {_TRAINING_MODES}."
        )
    if training_cycles < 1:
        raise ValueError("training_cycles must be at least 1")
    if ssl_updates_per_cycle < 1:
        raise ValueError("ssl_updates_per_cycle must be at least 1")
    if controller_dim != 3 and controller_dim < 4:
        raise ValueError("controller_dim must be 3 or at least 4")
    if training_mode == "ssl-rl-alternating" and controller_dim <= 3:
        raise ValueError(
            "ssl-rl-alternating requires controller_dim >= 4 for ACTIVE autograd writeback"
        )
    if not math.isfinite(ssl_alpha) or ssl_alpha < 0.0:
        raise ValueError("ssl_alpha must be a finite non-negative value")
    if not math.isfinite(switch_prior) or not 0.0 < switch_prior < 1.0:
        raise ValueError("switch_prior must be finite and strictly between 0 and 1")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (
            switch_rate_weight,
            switch_binary_weight,
            switch_group_weight,
            proposal_prediction_weight,
            gate_choice_weight,
        )
    ):
        raise ValueError("switch loss weights must be finite and non-negative")
    if prediction_horizon < 1:
        raise ValueError("prediction_horizon must be at least 1")
    if not math.isfinite(gate_choice_temperature) or gate_choice_temperature <= 0.0:
        raise ValueError("gate_choice_temperature must be finite and positive")
    if distortion_target not in {"absolute", "innovation"}:
        raise ValueError(
            "distortion_target must be 'absolute' or 'innovation'"
        )
    active_config = open_weight_config or ETAOpenWeightRuntimeConfig(device="mps")
    runtime: OpenWeightResidualRuntime | None = None
    if backend_label == "transformers-open-weight":
        runtime = _build_eta_open_weight_runtime(active_config)
        _validate_eta_open_weight_runtime(runtime=runtime, config=active_config)

    all_events: list[SegmentCreditEvent] = []
    all_boundaries: list[SegmentBoundaryCase] = []
    all_run_metrics: list[SegmentCreditRunMetrics] = []
    cases = default_eta_proof_cases()
    case_inputs: dict[
        str,
        tuple[
            ETAProofCase,
            tuple[SubstrateSnapshot, ...],
            tuple[str, ...],
            tuple[ExpertActionTarget, ...],
        ],
    ] = {}
    for case in cases:
        snapshots, source_texts, expert_targets = _expert_action_observation_bundle(
            case,
            open_weight_runtime=runtime,
        )
        calibrated_case = _calibrate_case_for_real_snapshots(
            case,
            snapshots=snapshots,
            enabled=backend_label == "transformers-open-weight",
        )
        case_inputs[case.case_id] = (
            calibrated_case,
            snapshots,
            source_texts,
            expert_targets,
        )

    effective_training_cycles = (
        training_cycles if training_mode == "ssl-rl-alternating" else 1
    )
    for seed in seed_schedule:
        random = Random(seed)
        ordered_cases = tuple(
            sorted(cases, key=lambda case: (random.random(), case.case_id))
        )
        profile = _profile_config("full-internal-rl")
        initial_policy = FullLearnedTemporalPolicy(
            parameter_store=MetacontrollerParameterStore(
                n_z=controller_dim,
                # The adapter/metacontroller checkpoint is shared. Evidence
                # seeds vary experience order and delayed-observation lag,
                # not the deployed model's initialization.
                initialization_seed=42,
            )
        )
        sandbox = _build_sandbox(
            profile=profile,
            backend_label=backend_label,
            bootstrap_snapshot=initial_policy.export_rare_heavy_snapshot(),
            open_weight_runtime=runtime,
        )
        trainer = (
            MetacontrollerSSLTrainer(
                n_z=controller_dim,
                alpha=ssl_alpha,
                ssl_backend=WiringLevel.ACTIVE,
                switch_prior=switch_prior,
                switch_rate_weight=switch_rate_weight,
                switch_binary_weight=switch_binary_weight,
                switch_group_weight=switch_group_weight,
                proposal_prediction_weight=proposal_prediction_weight,
                gate_choice_weight=gate_choice_weight,
                gate_choice_temperature=gate_choice_temperature,
                prediction_horizon=prediction_horizon,
                distortion_target=distortion_target,
            )
            if training_mode == "ssl-rl-alternating"
            else None
        )
        ssl_reports: list[SSLBatchTrainingReport] = []
        train_cases = tuple(item for item in ordered_cases if item.split == "train")
        ssl_cases = tuple(sorted(train_cases, key=lambda case: case.case_id))
        for cycle_index in range(effective_training_cycles):
            if trainer is not None:
                sandbox.policy.parameter_store.set_learning_phase(
                    "ssl", structure_frozen=False
                )
                ssl_traces = []
                for case in ssl_cases:
                    (
                        _rollout_case,
                        snapshots,
                        _source_texts,
                        expert_targets,
                    ) = case_inputs[case.case_id]
                    ssl_traces.append(
                        build_training_trace_from_substrate_snapshots(
                            trace_id=(
                                f"segment-credit:{seed}:{case.case_id}:"
                                f"ssl:{cycle_index}"
                            ),
                            source_text=case.source_text,
                            snapshots=snapshots,
                            expert_action_targets=expert_targets,
                        )
                    )
                for ssl_update_index in range(ssl_updates_per_cycle):
                    ssl_reports.append(
                        trainer.optimize_batch(
                            policy=sandbox.policy,
                            traces=tuple(ssl_traces),
                            batch_id=(
                                f"segment-credit:{seed}:ssl-batch:"
                                f"{cycle_index}:{ssl_update_index}"
                            ),
                        )
                    )

            train_rollouts: list[ZRollout] = []
            for case in train_cases:
                (
                    rollout_case,
                    snapshots,
                    source_texts,
                    _expert_targets,
                ) = case_inputs[case.case_id]
                sandbox.policy.parameter_store.set_learning_phase(
                    "rl", structure_frozen=True
                )
                train_rollouts.append(
                    sandbox.rollout(
                        rollout_id=(
                            f"segment-credit:{seed}:{case.case_id}:"
                            f"train:{cycle_index}"
                        ),
                        substrate_steps=snapshots,
                        track=Track.SHARED,
                        replacement_mode="causal",
                        proof_episode=rollout_case.proof_episode,
                        source_text_by_step=source_texts,
                    )
                )
            if train_rollouts:
                sandbox.optimize(tuple(train_rollouts))
                sandbox.ingest_temporal_fast_prior(
                    tuple(train_rollouts),
                    enabled=False,
                )
                sandbox.policy.parameter_store.calibrate_beta_threshold(
                    tuple(
                        transition.controller_state.switch_gate
                        for rollout in train_rollouts
                        for transition in rollout.transitions[1:]
                    ),
                    target_rate=switch_prior,
                )

        for report in ssl_reports:
            if report.trained_step_count <= 0:
                raise RuntimeError(
                    f"SSL batch {report.batch_id!r} trained zero steps."
                )
            if not all(
                math.isfinite(value)
                for value in (
                    report.torch_prediction_loss,
                    report.torch_kl_loss,
                    report.torch_total_loss,
                    report.torch_switch_rate_loss,
                )
            ):
                raise RuntimeError(
                    f"SSL batch {report.batch_id!r} produced non-finite loss."
                )
            if not report.torch_wrote_back:
                raise RuntimeError(
                    f"ACTIVE SSL batch {report.batch_id!r} did not write back."
                )

        seed_events: list[SegmentCreditEvent] = []
        seed_boundaries: list[SegmentBoundaryCase] = []
        for case in ordered_cases:
            (
                rollout_case,
                snapshots,
                source_texts,
                expert_targets,
            ) = case_inputs[case.case_id]
            sandbox.policy.parameter_store.set_learning_phase(
                "rl", structure_frozen=True
            )
            rollout = sandbox.rollout(
                rollout_id=f"segment-credit:{seed}:{case.case_id}:eval",
                substrate_steps=snapshots,
                track=Track.SHARED,
                replacement_mode="causal",
                proof_episode=rollout_case.proof_episode,
                source_text_by_step=source_texts,
            )
            seed_events.extend(
                _event_rows(
                    rollout=rollout,
                    case=case,
                    seed=seed,
                    max_observation_lag=max_observation_lag,
                    expert_targets=expert_targets,
                )
            )
            seed_boundaries.append(
                _boundary_row(
                    rollout=rollout,
                    case=case,
                    seed=seed,
                    expert_targets=expert_targets,
                )
            )
        event_tuple = tuple(seed_events)
        boundary_tuple = tuple(seed_boundaries)
        all_events.extend(event_tuple)
        all_boundaries.extend(boundary_tuple)
        all_run_metrics.append(
            _run_metrics(
                seed=seed,
                events=event_tuple,
                boundaries=boundary_tuple,
                training_cycle_count=effective_training_cycles,
                ssl_reports=tuple(ssl_reports),
                runtime_switch_threshold=(
                    sandbox.policy.parameter_store.beta_threshold
                ),
                learned_family_bank_count=len(
                    sandbox.policy.parameter_store.action_families
                ),
            )
        )

    run_metrics = tuple(all_run_metrics)
    intervals = _metric_intervals(run_metrics)
    interval_map = {interval.metric_name: interval for interval in intervals}
    mechanism_gates = (
        (
            "ssl-uses-expert-action-targets",
            all(
                row.ssl_expert_action_supervision
                and row.ssl_supervision_targets
                == ("expert-action-vector",)
                for row in run_metrics
            ),
            float(
                min(
                    int(row.ssl_expert_action_supervision)
                    for row in run_metrics
                )
            ),
        ),
        (
            "ssl-consumes-real-residual-trajectories",
            (
                backend_label == "transformers-open-weight"
                and runtime is not None
                and not bool(getattr(runtime, "fallback_active", False))
                and all(row.ssl_trained_step_count > 0 for row in run_metrics)
            ),
            float(min(row.ssl_trained_step_count for row in run_metrics)),
        ),
        (
            "heldout-delayed-events-observed",
            all(row.heldout_event_count > 0 for row in run_metrics),
            float(min(row.heldout_event_count for row in run_metrics)),
        ),
        (
            "ssl-switch-rate-nondegenerate",
            all(
                0.02 <= row.ssl_switch_frequency_final <= 0.80
                for row in run_metrics
            ),
            float(min(row.ssl_switch_frequency_final for row in run_metrics)),
        ),
        (
            "ssl-active-backend-writes-live-store",
            all(row.ssl_writeback_count > 0 for row in run_metrics),
            float(min(row.ssl_writeback_count for row in run_metrics)),
        ),
        (
            "ssl-persistent-optimizer-reused",
            all(
                row.ssl_optimizer_final_step == effective_training_cycles
                * ssl_updates_per_cycle
                and row.ssl_optimizer_reuse_count
                == max(
                    effective_training_cycles * ssl_updates_per_cycle - 1,
                    0,
                )
                for row in run_metrics
            ),
            float(min(row.ssl_optimizer_reuse_count for row in run_metrics)),
        ),
        (
            "multiple-action-families-emerge",
            all(row.active_family_count >= 2 for row in run_metrics),
            float(min(row.active_family_count for row in run_metrics)),
        ),
    )
    outcome_gates = (
        (
            "segment-credit-f1-beats-turn",
            interval_map["credit_f1_delta"].ci_low > 0.0,
            interval_map["credit_f1_delta"].ci_low,
        ),
        (
            "segment-credit-reduces-false-credit",
            interval_map["false_credit_reduction"].ci_low > 0.0,
            interval_map["false_credit_reduction"].ci_low,
        ),
        (
            "segment-family-assignment-beats-turn",
            interval_map["family_assignment_delta"].ci_low > 0.0,
            interval_map["family_assignment_delta"].ci_low,
        ),
        (
            "segment-credit-reduces-heldout-pe",
            interval_map["pe_reduction_rate_delta"].ci_low > 0.0,
            interval_map["pe_reduction_rate_delta"].ci_low,
        ),
        (
            "beta-boundaries-track-subgoals",
            interval_map["segment_boundary_f1"].ci_low >= 0.50,
            interval_map["segment_boundary_f1"].ci_low,
        ),
    )
    gates = mechanism_gates + outcome_gates
    all_positive_means = all(
        interval_map[name].mean > 0.0
        for name in (
            "credit_f1_delta",
            "false_credit_reduction",
            "family_assignment_delta",
            "pe_reduction_rate_delta",
        )
    )
    mechanism_ready = all(passed for _, passed, _ in mechanism_gates)
    if len(seed_schedule) >= 5 and all(passed for _, passed, _ in gates):
        claim_status = "retain"
    elif mechanism_ready and all_positive_means:
        claim_status = "weak"
    else:
        claim_status = "fail"
    return SegmentCreditEvidenceReport(
        schema_version="eta-segment-credit-evidence.v13",
        backend_label=backend_label,
        model_id=active_config.model_id if runtime is not None else "trace",
        device=active_config.device if runtime is not None else "cpu",
        runtime_origin=(
            str(getattr(runtime, "runtime_origin", "unknown"))
            if runtime is not None
            else "trace"
        ),
        fallback_active=(
            bool(getattr(runtime, "fallback_active", False))
            if runtime is not None
            else False
        ),
        seed_schedule=seed_schedule,
        controller_initialization_seed=42,
        experience_seed_semantics=(
            "fixed-shared-checkpoint; seed varies episode order and "
            "delayed-observation lag"
        ),
        training_mode=training_mode,
        training_cycles=effective_training_cycles,
        ssl_updates_per_cycle=(
            ssl_updates_per_cycle
            if training_mode == "ssl-rl-alternating"
            else 0
        ),
        controller_dim=controller_dim,
        ssl_alpha=ssl_alpha,
        switch_prior=switch_prior,
        switch_rate_weight=switch_rate_weight,
        switch_binary_weight=switch_binary_weight,
        switch_group_weight=switch_group_weight,
        proposal_prediction_weight=proposal_prediction_weight,
        gate_choice_weight=gate_choice_weight,
        gate_choice_temperature=gate_choice_temperature,
        prediction_horizon=prediction_horizon,
        distortion_target=distortion_target,
        ssl_supervision_target=(
            "expert-action-vector"
            if training_mode == "ssl-rl-alternating"
            else "none"
        ),
        expert_action_supervision=all(
            row.ssl_expert_action_supervision for row in run_metrics
        ),
        outcome_target="observed-alignment-minus-nominal-completion-threshold",
        family_truth_source="environment-expert-action-target",
        family_mapping_fit_split="train-only",
        causal_family_manifold_projection=(
            training_mode == "ssl-rl-alternating"
        ),
        rollout_replacement_mode="causal",
        temporal_fast_prior_enabled=False,
        episode_recurrent_state_isolated=True,
        run_metrics=run_metrics,
        metric_intervals=intervals,
        claim_status=claim_status,
        retain_gates=gates,
        events=tuple(all_events),
        boundary_cases=tuple(all_boundaries),
        description=(
            "Matched delayed-outcome credit experiment with owner-side residual "
            f"training mode={training_mode}. Both credit arms consume the same "
            "frozen-substrate rollouts; only the credit assignment unit differs. "
            "The causal rollout keeps the learned switch unit as beta owner; "
            "Internal RL supplies z candidates without replacing beta boundaries. "
            "Boundary, completion, reward, and outcome labels remain "
            "evaluation-only and never supervise beta. Train traces carry "
            "environment-demonstrated action vectors with provenance; eval and "
            "held-out rollouts receive observations but no action targets. "
            "Discovered family IDs are aligned to environment expert actions "
            "using train events only; held-out events remain read-only. "
            "All runs start from one fixed shared-controller initialization; "
            "experience seeds vary case order and delayed-observation lag. "
            "Held-out PE predicts observed completion alignment relative to "
            "the environment's nominal threshold, which remains stable when "
            "backend-specific runtime completion thresholds are calibrated."
        ),
    )


def _write_jsonl(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def export_eta_segment_credit_evidence(
    report: SegmentCreditEvidenceReport,
    *,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / "report.json"
    report_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    event_rows = tuple(asdict(event) for event in report.events)
    boundary_rows = tuple(asdict(row) for row in report.boundary_cases)
    prediction_rows: list[dict[str, object]] = []
    for seed in report.seed_schedule:
        seed_events = tuple(event for event in report.events if event.run_seed == seed)
        train_events = tuple(event for event in seed_events if event.split == "train")
        eta_values = _fit_family_values(
            train_events,
            family_field="eta_family_id",
        )
        turn_values = _fit_family_values(
            train_events,
            family_field="turn_family_id",
        )
        for event in seed_events:
            eta_prediction = eta_values.get(event.eta_family_id, 0.0)
            turn_prediction = turn_values.get(event.turn_family_id, 0.0)
            prediction_rows.append(
                {
                    "event_id": event.event_id,
                    "run_seed": event.run_seed,
                    "case_id": event.case_id,
                    "split": event.split,
                    "eta_family_id": event.eta_family_id,
                    "turn_family_id": event.turn_family_id,
                    "true_family_id": event.true_family_id,
                    "raw_prior_prediction": 0.0,
                    "eta_predicted_outcome": eta_prediction,
                    "turn_predicted_outcome": turn_prediction,
                    "eta_absolute_error": abs(
                        event.outcome_value - eta_prediction
                    ),
                    "turn_absolute_error": abs(
                        event.outcome_value - turn_prediction
                    ),
                }
            )
    outcome_rows = tuple(
        {
            "event_id": event["event_id"],
            "observed_outcome_step": event["observed_outcome_step"],
            "observation_lag": event["observation_lag"],
            "outcome_value": event["outcome_value"],
            "subgoal_id": event["subgoal_id"],
        }
        for event in event_rows
    )
    credit_rows = tuple(
        {
            "event_id": event["event_id"],
            "true_credit_steps": event["true_credit_steps"],
            "eta_credit_steps": event["eta_credit_steps"],
            "turn_credit_steps": event["turn_credit_steps"],
            "eta_f1": event["eta_f1"],
            "turn_f1": event["turn_f1"],
            "eta_false_credit_rate": event["eta_false_credit_rate"],
            "turn_false_credit_rate": event["turn_false_credit_rate"],
        }
        for event in event_rows
    )
    paths = (
        report_path,
        target / "predictions.jsonl",
        target / "outcomes.jsonl",
        target / "segments.jsonl",
        target / "credit.jsonl",
        target / "report.md",
    )
    _write_jsonl(paths[1], tuple(prediction_rows))
    _write_jsonl(paths[2], outcome_rows)
    _write_jsonl(paths[3], boundary_rows)
    _write_jsonl(paths[4], credit_rows)
    interval_map = {
        interval.metric_name: interval
        for interval in report.metric_intervals
    }
    mean_families = _mean(
        tuple(float(row.active_family_count) for row in report.run_metrics)
    )
    beta_boundaries = sum(row.beta_boundary_count for row in report.run_metrics)
    true_boundaries = sum(row.true_boundary_count for row in report.run_metrics)
    paths[5].write_text(
        "\n".join(
            (
                "# ETA Segment Credit Evidence",
                "",
                f"- Verdict: `{report.claim_status}`",
                f"- Backend: `{report.backend_label}`",
                f"- Model: `{report.model_id}` on `{report.device}`",
                f"- Runtime origin: `{report.runtime_origin}`",
                f"- Seeds: `{report.seed_schedule}`",
                f"- Training mode: `{report.training_mode}`",
                f"- Training cycles: `{report.training_cycles}`",
                f"- SSL updates per cycle: `{report.ssl_updates_per_cycle}`",
                f"- Controller dim: `{report.controller_dim}`",
                f"- SSL alpha: `{report.ssl_alpha}`",
                f"- Switch prior: `{report.switch_prior}`",
                f"- Switch rate weight: `{report.switch_rate_weight}`",
                f"- Switch binary weight: `{report.switch_binary_weight}`",
                f"- Switch group weight: `{report.switch_group_weight}`",
                (
                    "- Proposal prediction / gate choice weight: "
                    f"`{report.proposal_prediction_weight}` / "
                    f"`{report.gate_choice_weight}`"
                ),
                (
                    "- Gate choice temperature: "
                    f"`{report.gate_choice_temperature}`"
                ),
                f"- Prediction horizon: `{report.prediction_horizon}`",
                f"- Distortion target: `{report.distortion_target}`",
                f"- SSL supervision target: `{report.ssl_supervision_target}`",
                (
                    "- Expert action supervision: "
                    f"`{report.expert_action_supervision}`"
                ),
                f"- Family truth source: `{report.family_truth_source}`",
                f"- Family mapping fit split: `{report.family_mapping_fit_split}`",
                (
                    "- Causal family manifold projection: "
                    f"`{report.causal_family_manifold_projection}`"
                ),
                f"- Rollout replacement mode: `{report.rollout_replacement_mode}`",
                (
                    "- Temporal fast prior enabled: "
                    f"`{report.temporal_fast_prior_enabled}`"
                ),
                (
                    "- Episode recurrent state isolated: "
                    f"`{report.episode_recurrent_state_isolated}`"
                ),
                "",
                "## Matched-control results",
                "",
                (
                    "- Credit F1 delta: "
                    f"{interval_map['credit_f1_delta'].mean:.4f} "
                    f"[{interval_map['credit_f1_delta'].ci_low:.4f}, "
                    f"{interval_map['credit_f1_delta'].ci_high:.4f}]"
                ),
                (
                    "- False-credit reduction: "
                    f"{interval_map['false_credit_reduction'].mean:.4f} "
                    f"[{interval_map['false_credit_reduction'].ci_low:.4f}, "
                    f"{interval_map['false_credit_reduction'].ci_high:.4f}]"
                ),
                (
                    "- Family-assignment delta: "
                    f"{interval_map['family_assignment_delta'].mean:.4f} "
                    f"[{interval_map['family_assignment_delta'].ci_low:.4f}, "
                    f"{interval_map['family_assignment_delta'].ci_high:.4f}]"
                ),
                (
                    "- Held-out PE reduction delta: "
                    f"{interval_map['pe_reduction_rate_delta'].mean:.4f} "
                    f"[{interval_map['pe_reduction_rate_delta'].ci_low:.4f}, "
                    f"{interval_map['pe_reduction_rate_delta'].ci_high:.4f}]"
                ),
                (
                    "- Segment-boundary F1: "
                    f"{interval_map['segment_boundary_f1'].mean:.4f} "
                    f"[{interval_map['segment_boundary_f1'].ci_low:.4f}, "
                    f"{interval_map['segment_boundary_f1'].ci_high:.4f}]"
                ),
                "",
                "## Mechanism diagnosis",
                "",
                f"- Mean active family count per run: {mean_families:.2f}",
                f"- Beta boundaries observed: {beta_boundaries}",
                f"- Ground-truth subgoal boundaries: {true_boundaries}",
                (
                    "- Held-out delayed events: "
                    f"{sum(row.heldout_event_count for row in report.run_metrics)}"
                ),
                (
                    "- Mean SSL trained steps per run: "
                    f"{_mean(tuple(float(row.ssl_trained_step_count) for row in report.run_metrics)):.2f}"
                ),
                (
                    "- Mean SSL prediction loss: "
                    f"{_mean(tuple(row.ssl_prediction_loss_mean for row in report.run_metrics)):.6f}"
                ),
                (
                    "- Mean SSL KL loss: "
                    f"{_mean(tuple(row.ssl_kl_loss_mean for row in report.run_metrics)):.6f}"
                ),
                (
                    "- Mean SSL switch frequency: "
                    f"{_mean(tuple(row.ssl_switch_frequency_mean for row in report.run_metrics)):.6f}"
                ),
                (
                    "- Final SSL switch frequency: "
                    f"{_mean(tuple(row.ssl_switch_frequency_final for row in report.run_metrics)):.6f}"
                ),
                (
                    "- Mean SSL switch probability: "
                    f"{_mean(tuple(row.ssl_switch_probability_mean for row in report.run_metrics)):.6f}"
                ),
                (
                    "- Final SSL switch probability: "
                    f"{_mean(tuple(row.ssl_switch_probability_final for row in report.run_metrics)):.6f}"
                ),
                (
                    "- Mean SSL switch-rate loss: "
                    f"{_mean(tuple(row.ssl_switch_rate_loss_mean for row in report.run_metrics)):.6f}"
                ),
                (
                    "- Mean SSL gate-choice loss: "
                    f"{_mean(tuple(row.ssl_gate_choice_loss_mean for row in report.run_metrics)):.6f}"
                ),
                (
                    "- Mean keep/switch counterfactual loss: "
                    f"{_mean(tuple(row.ssl_keep_prediction_loss_mean for row in report.run_metrics)):.6f}"
                    " / "
                    f"{_mean(tuple(row.ssl_switch_prediction_loss_mean for row in report.run_metrics)):.6f}"
                ),
                (
                    "- Mean SSL target variance: "
                    f"{_mean(tuple(row.ssl_target_variance_mean for row in report.run_metrics)):.6f}"
                ),
                (
                    "- Final SSL expert-action boundary F1: "
                    f"{_mean(tuple(row.ssl_action_boundary_f1_final for row in report.run_metrics)):.6f}"
                ),
                (
                    "- Final SSL boundary/continuation switch probability: "
                    f"{_mean(tuple(row.ssl_boundary_switch_probability_final for row in report.run_metrics)):.6f}"
                    " / "
                    f"{_mean(tuple(row.ssl_continuation_switch_probability_final for row in report.run_metrics)):.6f}"
                ),
                (
                    "- Final calibrated switch threshold: "
                    f"{_mean(tuple(row.ssl_switch_threshold_final for row in report.run_metrics)):.6f}"
                ),
                (
                    "- Final causal-runtime switch threshold: "
                    f"{_mean(tuple(row.runtime_switch_threshold_final for row in report.run_metrics)):.6f}"
                ),
                (
                    "- Persistent optimizer final step/reuse count: "
                    f"{min(row.ssl_optimizer_final_step for row in report.run_metrics)}"
                    " / "
                    f"{min(row.ssl_optimizer_reuse_count for row in report.run_metrics)}"
                ),
                (
                    "- SSL ACTIVE writebacks: "
                    f"{sum(row.ssl_writeback_count for row in report.run_metrics)}"
                ),
                "",
                (
                    "This report is an evaluation readout. It does not write back "
                    "to prediction-error, credit, or temporal owners."
                ),
                "",
            )
        ),
        encoding="utf-8",
    )
    return paths
