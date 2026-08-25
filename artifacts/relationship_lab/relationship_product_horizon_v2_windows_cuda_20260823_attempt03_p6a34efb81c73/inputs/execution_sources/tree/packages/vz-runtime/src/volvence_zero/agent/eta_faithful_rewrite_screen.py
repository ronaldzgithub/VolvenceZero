"""Branch-B screen for a more faithful ETA controller parameterization.

This lane is a new claim after the sealed Stage-3 ``kill-eta`` verdict.  It
does not reinterpret that verdict.  The screen changes exactly the three
instrument surfaces identified by P1/S2:

* cumulative causal-prefix residuals at the real injection layer are consumed
  at their full hidden width by a learnable encoder projection;
* steering is ``U_t e_t`` with a low-rank, z-conditioned matrix and no
  additive bias; and
* active-subgoal boundaries are evaluation-only oracle labels, never an SSL
  target.

The only training objective remains ETA Eq.3: expert-action NLL through the
frozen model plus ``alpha * KL``.  Passing this directional screen merely
admits a separately preregistered authoritative sweep; it cannot promote a
runtime path or revise prior evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import math
import statistics
import time
from typing import Protocol

from volvence_zero.agent.eta_proof_benchmark import (
    ETAProofCase,
    ETAProofCorpus,
    _runtime_capture_snapshot,
)
from volvence_zero.agent.eta_rate_distortion_evidence import (
    OBSERVATION_PROTOCOL_V4,
    _action_options,
    _rate_distortion_observation_texts,
    eta_stage2_probe_rows,
)
from volvence_zero.substrate import OpenWeightResidualRuntime, TrainingTrace
from volvence_zero.temporal import (
    MetacontrollerParameterStore,
    build_training_trace_from_substrate_snapshots,
)
from volvence_zero.temporal.metacontroller_components import (
    POSTERIOR_PARAMETERIZATION_SMOOTH,
    RATE_GATING_SWITCH,
)
from volvence_zero.temporal.torch_store_ssl import (
    CURRENT_OBSERVATION_LEARNED_PROJECTION,
    STEERED_CONTROL_CYCLIC_PERMUTED_Z,
    STEERED_CONTROL_ZERO_Z,
    STEERING_PARAMETERIZATION_LOW_RANK_MULTIPLICATIVE,
    FaithfulSteeringAttestation,
    StoreSSLEvaluationReport,
    StoreSSLTrainingSession,
)


ETA_FAITHFUL_REWRITE_SCREEN_SCHEMA_VERSION = (
    "eta-faithful-rewrite-screen-evidence.v1"
)
FAITHFUL_ACTION_PROMPT_SUFFIX = "\nNext move:"


@dataclass(frozen=True)
class FaithfulETAScreenThresholds:
    max_alpha_rate_spearman: float = -0.50
    min_rate_span: float = 0.05
    min_primary_zero_z_penalty: float = 0.02
    min_primary_permuted_z_penalty: float = 0.02
    min_seed_positive_fraction: float = 1.0
    min_primary_boundary_probability_contrast: float = 0.02
    min_primary_oracle_boundary_f1: float = 0.20


@dataclass(frozen=True)
class FaithfulETAScreenPoint:
    alpha: float
    seed: int
    train_rate: float
    heldout_rate: float
    train_distortion: float
    heldout_distortion: float
    heldout_zero_z_distortion: float
    heldout_permuted_z_distortion: float
    zero_z_penalty: float
    permuted_z_penalty: float
    train_oracle_boundary_f1: float
    heldout_oracle_boundary_f1: float
    heldout_boundary_switch_probability: float
    heldout_continuation_switch_probability: float
    heldout_boundary_probability_contrast: float
    heldout_hard_switch_frequency: float
    optimizer_steps: int
    final_total_loss: float
    final_grad_norm: float
    controller_input_width: int
    residual_width: int
    steering_rank: int
    free_bias_present: bool
    zero_code_strict_noop: bool
    input_projection_parameter_count: int
    input_projection_parameters_changed: int
    low_rank_parameter_count: int
    low_rank_parameters_changed: int
    wall_seconds: float


@dataclass(frozen=True)
class FaithfulETAScreenAggregate:
    alpha: float
    seed_count: int
    train_rate_mean: float
    heldout_rate_mean: float
    heldout_distortion_mean: float
    heldout_distortion_std: float
    zero_z_penalty_mean: float
    permuted_z_penalty_mean: float
    zero_z_positive_seed_fraction: float
    permuted_z_positive_seed_fraction: float
    oracle_boundary_f1_mean: float
    boundary_probability_contrast_mean: float


@dataclass(frozen=True)
class FaithfulETAScreenAdmission:
    admitted_for_authoritative_sweep: bool
    condition_structural_integrity: bool
    condition_rate_axis: bool
    condition_zero_z_causality: bool
    condition_permuted_z_causality: bool
    condition_boundary_alignment: bool
    failed_conditions: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class FaithfulETAScreenReport:
    schema_version: str
    claim_scope: str
    model_id: str
    model_source: str
    device: str
    runtime_origin: str
    observation_protocol: str
    observation_surface: str
    corpus_seed: int
    source_train_route_count: int
    source_heldout_route_count: int
    screen_train_route_count: int
    screen_heldout_route_count: int
    train_step_count: int
    heldout_step_count: int
    injection_layer_index: int
    residual_width: int
    n_z: int
    steering_rank: int
    steering_parameterization: str
    current_observation_mode: str
    free_bias_present: bool
    control_norm_ratio: float
    control_norm_cap: float
    probe_hidden_norm: float
    alpha_grid: tuple[float, ...]
    primary_alpha: float
    seed_schedule: tuple[int, ...]
    updates_per_run: int
    learning_rate: float
    switch_threshold: float
    posterior_parameterization: str
    rate_gating: str
    gate_mode: str
    max_observed_source_tokens: int
    scorer_max_length: int
    truncated_row_count: int
    thresholds: FaithfulETAScreenThresholds
    points: tuple[FaithfulETAScreenPoint, ...]
    aggregates: tuple[FaithfulETAScreenAggregate, ...]
    alpha_rate_spearman: float
    rate_span: float
    admission: FaithfulETAScreenAdmission
    substrate_trainable_parameter_count: int
    production_wiring_changed: bool
    feedback_to_learning: bool
    description: str


@dataclass(frozen=True)
class _FaithfulTraceBundle:
    traces: tuple[TrainingTrace, ...]
    boundary_labels: dict[str, tuple[float, ...]]


class FaithfulETAScreenPointCache(Protocol):
    def load_point(
        self, *, alpha: float, seed: int
    ) -> FaithfulETAScreenPoint | None: ...

    def store_point(self, point: FaithfulETAScreenPoint) -> None: ...


def _mean(values: tuple[float, ...]) -> float:
    return statistics.fmean(values) if values else 0.0


def _spearman(xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0

    def rank(values: tuple[float, ...]) -> tuple[float, ...]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        ranks = [0.0] * len(values)
        index = 0
        while index < len(order):
            end = index
            while (
                end + 1 < len(order)
                and values[order[end + 1]] == values[order[index]]
            ):
                end += 1
            average = (index + end) / 2.0
            for position in range(index, end + 1):
                ranks[order[position]] = average
            index = end + 1
        return tuple(ranks)

    left = rank(xs)
    right = rank(ys)
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return numerator / (left_norm * right_norm)


def _build_faithful_trace_bundle(
    *,
    cases: tuple[ETAProofCase, ...],
    corpus: ETAProofCorpus,
    runtime: OpenWeightResidualRuntime,
    split_label: str,
    injection_layer_index: int,
    residual_width: int,
    progress: Callable[[str], None] | None,
) -> _FaithfulTraceBundle:
    probe_rows, class_ids = eta_stage2_probe_rows(
        cases,
        environment=corpus.environment,
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    rows_by_case: dict[str, list[object]] = {}
    for row in probe_rows:
        rows_by_case.setdefault(row.case_id, []).append(row)

    traces: list[TrainingTrace] = []
    boundary_labels: dict[str, tuple[float, ...]] = {}
    captured = 0
    total_rows = len(probe_rows)
    for case in cases:
        case_rows = tuple(
            sorted(rows_by_case.get(case.case_id, ()), key=lambda row: row.step_index)
        )
        if len(case_rows) < 2:
            raise RuntimeError(
                f"Faithful ETA case {case.case_id!r} has fewer than two "
                "causal-prefix rows."
            )
        _single_texts, targets, subgoals = _rate_distortion_observation_texts(
            case,
            environment=corpus.environment,
            protocol_version=OBSERVATION_PROTOCOL_V4,
        )
        snapshots = []
        aligned_targets = []
        aligned_subgoals: list[str] = []
        aligned_texts = []
        for row in case_rows:
            subgoal = subgoals[row.step_index]
            if subgoal is None or class_ids[row.subgoal_label] != subgoal:
                raise RuntimeError(
                    "Faithful ETA cumulative-prefix/subgoal lineage mismatch "
                    f"for {case.case_id!r} step {row.step_index}."
                )
            scored_prefix = row.observation_text + FAITHFUL_ACTION_PROMPT_SUFFIX
            snapshot = _runtime_capture_snapshot(
                runtime=runtime,
                case=case,
                source_text=scored_prefix,
                step_index=row.step_index,
                total_steps=len(case_rows),
            )
            activations = snapshot.residual_activations
            if (
                len(activations) != 1
                or activations[0].layer_index != injection_layer_index
                or len(activations[0].activation) != residual_width
            ):
                raise RuntimeError(
                    "Faithful ETA requires one exact full-width residual at "
                    f"layer {injection_layer_index}; case={case.case_id!r} "
                    f"step={row.step_index}."
                )
            snapshots.append(
                replace(
                    snapshot,
                    residual_sequence=(),
                    description=(
                        f"{snapshot.description} Faithful ETA cumulative-prefix "
                        "single environment step."
                    ),
                )
            )
            aligned_targets.append(targets[row.step_index])
            aligned_subgoals.append(subgoal)
            aligned_texts.append(scored_prefix)
            captured += 1
            if progress is not None and (
                captured == total_rows or captured % 32 == 0
            ):
                progress(
                    f"faithful ETA capture {split_label}: "
                    f"{captured}/{total_rows}"
                )
        trace_id = f"faithful-eta:{split_label}:{case.case_id}"
        trace = build_training_trace_from_substrate_snapshots(
            trace_id=trace_id,
            source_text=case.source_text,
            snapshots=tuple(snapshots),
            expert_action_targets=tuple(aligned_targets),
            observation_texts=tuple(aligned_texts),
        )
        traces.append(trace)
        boundary_labels[trace_id] = tuple(
            float(aligned_subgoals[index] != aligned_subgoals[index - 1])
            for index in range(1, len(aligned_subgoals))
        )
    return _FaithfulTraceBundle(
        traces=tuple(traces),
        boundary_labels=boundary_labels,
    )


def _run_cell(
    *,
    alpha: float,
    seed: int,
    n_z: int,
    residual_width: int,
    steering_rank: int,
    scorer,
    train_bundle: _FaithfulTraceBundle,
    heldout_bundle: _FaithfulTraceBundle,
    updates_per_run: int,
    learning_rate: float,
    switch_threshold: float,
    progress: Callable[[str], None] | None,
) -> FaithfulETAScreenPoint:
    started = time.perf_counter()
    store = MetacontrollerParameterStore(
        n_z=n_z,
        n_input=residual_width,
        initialization_seed=seed,
    )
    session = StoreSSLTrainingSession(
        n_z=n_z,
        alpha=alpha,
        learning_rate=learning_rate,
        switch_rate_weight=0.0,
        switch_binary_weight=0.0,
        switch_group_weight=0.0,
        proposal_prediction_weight=0.0,
        gate_choice_weight=0.0,
        action_scorer=scorer,
        reparam_seed=seed * 1_000_003 + 17,
        posterior_parameterization=POSTERIOR_PARAMETERIZATION_SMOOTH,
        rate_gating=RATE_GATING_SWITCH,
        steered_gate_mode="hard-st",
        steering_parameterization=(
            STEERING_PARAMETERIZATION_LOW_RANK_MULTIPLICATIVE
        ),
        steering_rank=steering_rank,
        current_observation_mode=CURRENT_OBSERVATION_LEARNED_PROJECTION,
    )
    final_report = None
    for update_index in range(updates_per_run):
        final_report = session.train_batch(
            store=store,
            traces=train_bundle.traces,
            batch_id=f"faithful:a{alpha}:s{seed}:u{update_index}",
            switch_threshold=switch_threshold,
            write_back=False,
        )
        if not math.isfinite(final_report.total_loss):
            raise RuntimeError(
                "Non-finite faithful ETA loss at "
                f"alpha={alpha} seed={seed} update={update_index}."
            )
        if progress is not None and (
            update_index + 1 == updates_per_run
            or (update_index + 1) % 10 == 0
        ):
            progress(
                f"faithful ETA alpha={alpha} seed={seed} "
                f"update={update_index + 1}/{updates_per_run}"
            )
    if final_report is None:
        raise RuntimeError("Faithful ETA screen needs at least one update.")

    train_eval: StoreSSLEvaluationReport = session.evaluate_batch(
        store=store,
        traces=train_bundle.traces,
        batch_id=f"faithful:a{alpha}:s{seed}:train",
        switch_threshold=switch_threshold,
        boundary_labels=train_bundle.boundary_labels,
    )
    heldout_eval: StoreSSLEvaluationReport = session.evaluate_batch(
        store=store,
        traces=heldout_bundle.traces,
        batch_id=f"faithful:a{alpha}:s{seed}:heldout",
        switch_threshold=switch_threshold,
        boundary_labels=heldout_bundle.boundary_labels,
    )
    zero_eval = session.evaluate_batch(
        store=store,
        traces=heldout_bundle.traces,
        batch_id=f"faithful:a{alpha}:s{seed}:zero-z",
        switch_threshold=switch_threshold,
        control_ablation=STEERED_CONTROL_ZERO_Z,
        boundary_labels=heldout_bundle.boundary_labels,
    )
    permuted_eval = session.evaluate_batch(
        store=store,
        traces=heldout_bundle.traces,
        batch_id=f"faithful:a{alpha}:s{seed}:permuted-z",
        switch_threshold=switch_threshold,
        control_ablation=STEERED_CONTROL_CYCLIC_PERMUTED_Z,
        boundary_labels=heldout_bundle.boundary_labels,
    )
    attestation: FaithfulSteeringAttestation = (
        session.faithful_steering_attestation()
    )
    return FaithfulETAScreenPoint(
        alpha=alpha,
        seed=seed,
        train_rate=train_eval.kl_rate,
        heldout_rate=heldout_eval.kl_rate,
        train_distortion=train_eval.distortion,
        heldout_distortion=heldout_eval.distortion,
        heldout_zero_z_distortion=zero_eval.distortion,
        heldout_permuted_z_distortion=permuted_eval.distortion,
        zero_z_penalty=zero_eval.distortion - heldout_eval.distortion,
        permuted_z_penalty=(
            permuted_eval.distortion - heldout_eval.distortion
        ),
        train_oracle_boundary_f1=train_eval.boundary_f1,
        heldout_oracle_boundary_f1=heldout_eval.boundary_f1,
        heldout_boundary_switch_probability=(
            heldout_eval.boundary_switch_probability
        ),
        heldout_continuation_switch_probability=(
            heldout_eval.continuation_switch_probability
        ),
        heldout_boundary_probability_contrast=(
            heldout_eval.boundary_switch_probability
            - heldout_eval.continuation_switch_probability
        ),
        heldout_hard_switch_frequency=heldout_eval.hard_switch_frequency,
        optimizer_steps=session.optimizer_step,
        final_total_loss=final_report.total_loss,
        final_grad_norm=final_report.grad_norm,
        controller_input_width=attestation.controller_input_width,
        residual_width=attestation.residual_width,
        steering_rank=attestation.steering_rank,
        free_bias_present=attestation.free_bias_present,
        zero_code_strict_noop=attestation.zero_code_strict_noop,
        input_projection_parameter_count=(
            attestation.input_projection_parameter_count
        ),
        input_projection_parameters_changed=(
            attestation.input_projection_parameters_changed
        ),
        low_rank_parameter_count=attestation.low_rank_parameter_count,
        low_rank_parameters_changed=(
            attestation.low_rank_parameters_changed
        ),
        wall_seconds=time.perf_counter() - started,
    )


def _aggregate_points(
    points: tuple[FaithfulETAScreenPoint, ...],
    *,
    alpha_grid: tuple[float, ...],
) -> tuple[FaithfulETAScreenAggregate, ...]:
    aggregates = []
    for alpha in alpha_grid:
        rows = tuple(point for point in points if point.alpha == alpha)
        if not rows:
            raise RuntimeError(f"Faithful ETA alpha {alpha} has no points.")
        distortions = tuple(row.heldout_distortion for row in rows)
        aggregates.append(
            FaithfulETAScreenAggregate(
                alpha=alpha,
                seed_count=len(rows),
                train_rate_mean=_mean(tuple(row.train_rate for row in rows)),
                heldout_rate_mean=_mean(
                    tuple(row.heldout_rate for row in rows)
                ),
                heldout_distortion_mean=_mean(distortions),
                heldout_distortion_std=(
                    statistics.pstdev(distortions)
                    if len(distortions) > 1
                    else 0.0
                ),
                zero_z_penalty_mean=_mean(
                    tuple(row.zero_z_penalty for row in rows)
                ),
                permuted_z_penalty_mean=_mean(
                    tuple(row.permuted_z_penalty for row in rows)
                ),
                zero_z_positive_seed_fraction=(
                    sum(row.zero_z_penalty > 0.0 for row in rows) / len(rows)
                ),
                permuted_z_positive_seed_fraction=(
                    sum(row.permuted_z_penalty > 0.0 for row in rows)
                    / len(rows)
                ),
                oracle_boundary_f1_mean=_mean(
                    tuple(row.heldout_oracle_boundary_f1 for row in rows)
                ),
                boundary_probability_contrast_mean=_mean(
                    tuple(
                        row.heldout_boundary_probability_contrast
                        for row in rows
                    )
                ),
            )
        )
    return tuple(aggregates)


def assess_faithful_eta_screen(
    *,
    points: tuple[FaithfulETAScreenPoint, ...],
    aggregates: tuple[FaithfulETAScreenAggregate, ...],
    primary_alpha: float,
    alpha_rate_spearman: float,
    rate_span: float,
    residual_width: int,
    steering_rank: int,
    thresholds: FaithfulETAScreenThresholds,
) -> FaithfulETAScreenAdmission:
    primary = next(row for row in aggregates if row.alpha == primary_alpha)
    conditions = {
        "structural-integrity": all(
            point.controller_input_width == residual_width
            and point.residual_width == residual_width
            and point.steering_rank == steering_rank
            and not point.free_bias_present
            and point.zero_code_strict_noop
            and point.input_projection_parameters_changed > 0
            and point.low_rank_parameters_changed > 0
            for point in points
        ),
        "rate-axis": (
            alpha_rate_spearman <= thresholds.max_alpha_rate_spearman
            and rate_span >= thresholds.min_rate_span
        ),
        "zero-z-causality": (
            primary.zero_z_penalty_mean
            >= thresholds.min_primary_zero_z_penalty
            and primary.zero_z_positive_seed_fraction
            >= thresholds.min_seed_positive_fraction
        ),
        "permuted-z-causality": (
            primary.permuted_z_penalty_mean
            >= thresholds.min_primary_permuted_z_penalty
            and primary.permuted_z_positive_seed_fraction
            >= thresholds.min_seed_positive_fraction
        ),
        "oracle-boundary-alignment": (
            primary.boundary_probability_contrast_mean
            >= thresholds.min_primary_boundary_probability_contrast
            and primary.oracle_boundary_f1_mean
            >= thresholds.min_primary_oracle_boundary_f1
        ),
    }
    failed = tuple(name for name, passed in conditions.items() if not passed)
    return FaithfulETAScreenAdmission(
        admitted_for_authoritative_sweep=not failed,
        condition_structural_integrity=conditions["structural-integrity"],
        condition_rate_axis=conditions["rate-axis"],
        condition_zero_z_causality=conditions["zero-z-causality"],
        condition_permuted_z_causality=conditions["permuted-z-causality"],
        condition_boundary_alignment=conditions["oracle-boundary-alignment"],
        failed_conditions=failed,
        description=(
            "Faithful ETA rewrite admitted for a separate authoritative sweep."
            if not failed
            else "Faithful ETA rewrite screen blocked: " + ", ".join(failed)
        ),
    )


def run_eta_faithful_rewrite_screen(
    *,
    corpus: ETAProofCorpus,
    runtime: OpenWeightResidualRuntime,
    model_source: str,
    device: str,
    screen_train_route_count: int = 16,
    screen_heldout_route_count: int = 8,
    alpha_grid: tuple[float, ...] = (0.03, 0.30, 3.00),
    primary_alpha: float = 0.30,
    seed_schedule: tuple[int, ...] = (0, 1),
    updates_per_run: int = 40,
    learning_rate: float = 0.005,
    switch_threshold: float = 0.55,
    n_z: int = 16,
    injection_layer_index: int = 20,
    residual_width: int = 896,
    steering_rank: int = 8,
    control_norm_ratio: float = 0.25,
    scorer_max_length: int = 768,
    max_observed_source_tokens: int = 0,
    point_cache: FaithfulETAScreenPointCache | None = None,
    thresholds: FaithfulETAScreenThresholds | None = None,
    progress: Callable[[str], None] | None = None,
) -> FaithfulETAScreenReport:
    if tuple(sorted(set(alpha_grid))) != alpha_grid or len(alpha_grid) < 3:
        raise ValueError("Faithful ETA alpha_grid must be sorted, unique, and >=3.")
    if primary_alpha not in alpha_grid:
        raise ValueError("Faithful ETA primary_alpha must be in alpha_grid.")
    if not seed_schedule or len(set(seed_schedule)) != len(seed_schedule):
        raise ValueError("Faithful ETA seed_schedule must be non-empty and unique.")
    if updates_per_run < 1 or learning_rate <= 0.0:
        raise ValueError("Faithful ETA updates and learning rate must be positive.")
    if n_z <= 3 or residual_width <= n_z:
        raise ValueError("Faithful ETA requires low-dimensional z and wide residuals.")
    if not 1 <= steering_rank <= n_z:
        raise ValueError("Faithful ETA steering_rank must be in [1, n_z].")
    if (
        max_observed_source_tokens < 1
        or max_observed_source_tokens > scorer_max_length
    ):
        raise ValueError(
            "Faithful ETA token preflight must prove every source fits."
        )
    if not 1 <= screen_train_route_count <= len(corpus.train_cases):
        raise ValueError("Faithful ETA train route screen count is invalid.")
    if not 1 <= screen_heldout_route_count <= len(corpus.heldout_cases):
        raise ValueError("Faithful ETA heldout route screen count is invalid.")

    train_cases = corpus.train_cases[:screen_train_route_count]
    heldout_cases = corpus.heldout_cases[:screen_heldout_route_count]
    train_bundle = _build_faithful_trace_bundle(
        cases=train_cases,
        corpus=corpus,
        runtime=runtime,
        split_label="train",
        injection_layer_index=injection_layer_index,
        residual_width=residual_width,
        progress=progress,
    )
    heldout_bundle = _build_faithful_trace_bundle(
        cases=heldout_cases,
        corpus=corpus,
        runtime=runtime,
        split_label="heldout",
        injection_layer_index=injection_layer_index,
        residual_width=residual_width,
        progress=progress,
    )
    probe_texts = tuple(
        step.observation_text
        for trace in train_bundle.traces
        for step in trace.steps
    )[:16]
    scorer = runtime.build_steered_action_scorer(
        action_options=_action_options(corpus.environment),
        injection_layer_index=injection_layer_index,
        prompt_suffix="",
        max_length=scorer_max_length,
        control_norm_ratio=control_norm_ratio,
        probe_texts=probe_texts,
        joint_training=False,
        prefix_cache=True,
    )
    if scorer.hidden_size != residual_width:
        raise RuntimeError("Faithful ETA scorer residual width drifted.")
    substrate_trainable_parameter_count = sum(
        int(parameter.numel()) for parameter in scorer.trainable_parameters()
    )
    if substrate_trainable_parameter_count:
        raise RuntimeError("Faithful ETA screen requires a frozen substrate.")

    points: list[FaithfulETAScreenPoint] = []
    for alpha in alpha_grid:
        for seed in seed_schedule:
            cached = (
                point_cache.load_point(alpha=alpha, seed=seed)
                if point_cache is not None
                else None
            )
            if cached is not None:
                points.append(cached)
                if progress is not None:
                    progress(f"faithful ETA resume alpha={alpha} seed={seed}")
                continue
            point = _run_cell(
                alpha=alpha,
                seed=seed,
                n_z=n_z,
                residual_width=residual_width,
                steering_rank=steering_rank,
                scorer=scorer,
                train_bundle=train_bundle,
                heldout_bundle=heldout_bundle,
                updates_per_run=updates_per_run,
                learning_rate=learning_rate,
                switch_threshold=switch_threshold,
                progress=progress,
            )
            if point_cache is not None:
                point_cache.store_point(point)
            points.append(point)
    scorer.clear_prefix_cache()

    ordered_points = tuple(sorted(points, key=lambda row: (row.alpha, row.seed)))
    aggregates = _aggregate_points(ordered_points, alpha_grid=alpha_grid)
    rates = tuple(row.train_rate_mean for row in aggregates)
    alpha_rate_spearman = _spearman(alpha_grid, rates)
    rate_span = max(rates) - min(rates)
    active_thresholds = thresholds or FaithfulETAScreenThresholds()
    admission = assess_faithful_eta_screen(
        points=ordered_points,
        aggregates=aggregates,
        primary_alpha=primary_alpha,
        alpha_rate_spearman=alpha_rate_spearman,
        rate_span=rate_span,
        residual_width=residual_width,
        steering_rank=steering_rank,
        thresholds=active_thresholds,
    )
    return FaithfulETAScreenReport(
        schema_version=ETA_FAITHFUL_REWRITE_SCREEN_SCHEMA_VERSION,
        claim_scope="faithful-eta-rewrite-directional-screen",
        model_id=runtime.model_id,
        model_source=model_source,
        device=device,
        runtime_origin=str(runtime.runtime_origin),
        observation_protocol=OBSERVATION_PROTOCOL_V4,
        observation_surface="stage2-v4-cumulative-causal-prefix",
        corpus_seed=corpus.seed,
        source_train_route_count=len(corpus.train_cases),
        source_heldout_route_count=len(corpus.heldout_cases),
        screen_train_route_count=len(train_cases),
        screen_heldout_route_count=len(heldout_cases),
        train_step_count=sum(len(trace.steps) for trace in train_bundle.traces),
        heldout_step_count=sum(
            len(trace.steps) for trace in heldout_bundle.traces
        ),
        injection_layer_index=injection_layer_index,
        residual_width=residual_width,
        n_z=n_z,
        steering_rank=steering_rank,
        steering_parameterization=(
            STEERING_PARAMETERIZATION_LOW_RANK_MULTIPLICATIVE
        ),
        current_observation_mode=CURRENT_OBSERVATION_LEARNED_PROJECTION,
        free_bias_present=False,
        control_norm_ratio=control_norm_ratio,
        control_norm_cap=scorer.control_norm_cap,
        probe_hidden_norm=scorer.probe_hidden_norm,
        alpha_grid=alpha_grid,
        primary_alpha=primary_alpha,
        seed_schedule=seed_schedule,
        updates_per_run=updates_per_run,
        learning_rate=learning_rate,
        switch_threshold=switch_threshold,
        posterior_parameterization=POSTERIOR_PARAMETERIZATION_SMOOTH,
        rate_gating=RATE_GATING_SWITCH,
        gate_mode="hard-st",
        max_observed_source_tokens=max_observed_source_tokens,
        scorer_max_length=scorer_max_length,
        truncated_row_count=0,
        thresholds=active_thresholds,
        points=ordered_points,
        aggregates=aggregates,
        alpha_rate_spearman=alpha_rate_spearman,
        rate_span=rate_span,
        admission=admission,
        substrate_trainable_parameter_count=substrate_trainable_parameter_count,
        production_wiring_changed=False,
        feedback_to_learning=False,
        description=(
            "New-claim Branch-B screen with full-width learned entry, no-bias "
            "low-rank U_t·e, and cumulative causal-prefix parity; admitted="
            f"{admission.admitted_for_authoritative_sweep}."
        ),
    )


__all__ = [
    "ETA_FAITHFUL_REWRITE_SCREEN_SCHEMA_VERSION",
    "FAITHFUL_ACTION_PROMPT_SUFFIX",
    "FaithfulETAScreenAdmission",
    "FaithfulETAScreenAggregate",
    "FaithfulETAScreenPoint",
    "FaithfulETAScreenPointCache",
    "FaithfulETAScreenReport",
    "FaithfulETAScreenThresholds",
    "assess_faithful_eta_screen",
    "run_eta_faithful_rewrite_screen",
]
