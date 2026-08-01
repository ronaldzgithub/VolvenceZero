"""ETA rate-distortion criterion (paper-faithful Eq.3 instrument).

The ETA paper gives one falsifiable criterion for "a frozen substrate plus a
metacontroller can discover temporal abstractions": sweep the KL weight
``alpha`` and plot the (rate, distortion) curve. With a frozen substrate a
near-vertical gap should appear — a small rate increase buys a large
distortion improvement — and subgoal-aligned switching lives inside that
gap. Joint training of the substrate makes the gap disappear (the model
degenerates to switching once at the start).

This harness is the first executable version of that criterion in this
repository. It became runnable only after four deviations from Eq.3 were
fixed (differentiable steered forward, through-model action NLL distortion,
genuine reparameterized sampling, switch regularizers zeroed):

- **Distortion** = expert-action NLL through the *steered frozen model*
  (``TransformersSteeredActionScorer``), not a decoder regression.
- **Rate** = mean per-dimension posterior KL of ``z_t``.
- Loss has exactly two terms: distortion + alpha * KL. Beta boundary F1 and
  switch frequency are evaluation readouts only.
- The joint arm (non-frozen upper blocks) is a mandatory validity control:
  if the frozen and joint arms produce indistinguishable curves the
  instrument is invalid and no thesis conclusion may be drawn.

Observation design: unlike the legacy segment-credit bundle, the
rate-distortion observations are *partially observable* — they do NOT
include the remaining route. The legacy prompt leaked the next expert action
verbatim ("Remaining route: alpha -> ..."), which lets the frozen model read
the answer and leaves nothing for ``z_t`` to carry. Here the route identity
only appears as the abstract task-context sentence, so the controller must
compress "which route am I on / which segment is active" into ``z_t`` —
exactly the temporal abstraction the criterion tests.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import statistics
import time
from typing import Protocol

from volvence_zero.agent.eta_proof_benchmark import (
    ETAOpenWeightRuntimeConfig,
    ETAProofCase,
    _build_eta_open_weight_runtime,
    _runtime_capture_snapshot,
    _validate_eta_open_weight_runtime,
    build_default_eta_proof_environment,
    default_eta_proof_cases,
)
from volvence_zero.internal_rl import HierarchicalRouteSpec
from volvence_zero.substrate import (
    ExpertActionTarget,
    OpenWeightResidualRuntime,
    SteeredActionOption,
    SubstrateSnapshot,
    TrainingTrace,
)
from volvence_zero.temporal import (
    MetacontrollerParameterStore,
    build_training_trace_from_substrate_snapshots,
)
from volvence_zero.temporal.torch_store_ssl import (
    StoreSSLEvaluationReport,
    StoreSSLTrainingSession,
)

RATE_DISTORTION_SCHEMA_VERSION = "eta-rate-distortion-evidence.v1"

_ARMS = ("frozen", "joint")


@dataclass(frozen=True)
class RateDistortionPoint:
    """One trained (arm, alpha, seed) run's final readouts."""

    arm: str
    alpha: float
    seed: int
    train_rate: float
    train_distortion: float
    heldout_rate: float
    heldout_distortion: float
    baseline_train_distortion: float
    baseline_heldout_distortion: float
    mean_switch_probability: float
    hard_switch_frequency: float
    train_boundary_f1: float
    heldout_boundary_f1: float
    optimizer_steps: int
    final_total_loss: float
    final_grad_norm: float
    wall_seconds: float
    description: str = ""


@dataclass(frozen=True)
class RateDistortionCurvePoint:
    """Cross-seed aggregate for one (arm, alpha) grid cell."""

    arm: str
    alpha: float
    rate_mean: float
    rate_std: float
    distortion_mean: float
    distortion_std: float
    heldout_distortion_mean: float
    boundary_f1_mean: float
    switch_frequency_mean: float
    seed_count: int


@dataclass(frozen=True)
class GapAssessment:
    """Detection of the paper's near-vertical rate-distortion gap.

    A gap is declared when, on the rate-ordered aggregate curve, one
    adjacent segment carries at least ``drop_share_threshold`` of the total
    distortion improvement while consuming at most ``rate_share_threshold``
    of the total rate span, and the total improvement clears the cross-seed
    noise floor (``noise_multiple`` x pooled distortion std).
    """

    arm: str
    gap_detected: bool
    distortion_span: float
    rate_span: float
    noise_scale: float
    max_drop: float
    max_drop_share: float
    max_drop_rate_share: float
    gap_low_alpha: float
    gap_high_alpha: float
    drop_share_threshold: float
    rate_share_threshold: float
    noise_multiple: float
    boundary_f1_gap_region: float
    boundary_f1_outside_gap: float
    description: str = ""


@dataclass(frozen=True)
class RateDistortionEvidenceReport:
    schema_version: str
    model_id: str
    device: str
    runtime_origin: str
    fallback_active: bool
    injection_layer_index: int
    control_norm_cap: float
    probe_hidden_norm: float
    n_z: int
    alpha_grid: tuple[float, ...]
    seed_schedule: tuple[int, ...]
    updates_per_run: int
    learning_rate: float
    substrate_learning_rate: float
    switch_threshold: float
    arms: tuple[str, ...]
    observation_protocol: str
    action_vocabulary: tuple[str, ...]
    train_case_ids: tuple[str, ...]
    heldout_case_ids: tuple[str, ...]
    train_step_count: int
    heldout_step_count: int
    points: tuple[RateDistortionPoint, ...]
    curves: tuple[RateDistortionCurvePoint, ...]
    gaps: tuple[GapAssessment, ...]
    arms_distinguishable: bool
    arm_separation: float
    arm_separation_threshold: float
    verdict: str
    verdict_reason: str
    description: str


def _mean(values: tuple[float, ...]) -> float:
    return statistics.fmean(values) if values else 0.0


def _std(values: tuple[float, ...]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _action_options(
    environment,
) -> tuple[SteeredActionOption, ...]:
    target_ids = {
        transition.target_id for transition in environment.transitions
    }
    return tuple(
        SteeredActionOption(
            action_id=f"move:{location.location_id}",
            surface_text=location.location_id,
        )
        for location in environment.locations
        if location.location_id in target_ids
    )


def _rate_distortion_observation_bundle(
    case: ETAProofCase,
    *,
    open_weight_runtime: OpenWeightResidualRuntime,
) -> tuple[
    tuple[SubstrateSnapshot, ...],
    tuple[str, ...],
    tuple[ExpertActionTarget, ...],
]:
    """Partially observable expert demonstration for one proof route.

    One observation per expert controller-action phase, WITHOUT the
    remaining-route leak: the expert action is not readable from any single
    observation, so the temporal abstraction (active route segment) is the
    only channel that can reduce action uncertainty.
    """

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
        completed = (
            ", ".join(observation.completed_objective_ids)
            if observation.completed_objective_ids
            else "none"
        )
        for phase_index in range(phase_count):
            observation_texts.append(
                "Task context: "
                f"{case.source_text}. Current location: "
                f"{observation.current_location_id}. Available transitions: "
                f"{', '.join(observation.available_targets)}. Completed "
                f"objectives: {completed}. Phase {phase_index + 1} of "
                f"{phase_count}."
            )
            expert_targets.append(
                ExpertActionTarget(
                    action_id=f"move:{target_id}",
                    values=action_values,
                    source="eta-rate-distortion-demonstration",
                    description=(
                        f"Expert transition from {state.current_location_id} "
                        f"to {target_id}; phase {phase_index + 1}/{phase_count}."
                    ),
                )
            )
        state = environment.step(state, target_id=target_id).next_state
    if len(observation_texts) < 2:
        raise ValueError(
            f"Rate-distortion route {case.case_id!r} must publish at least "
            "two phases."
        )
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
                "residual step for rate-distortion SSL."
            ),
        )
        for snapshot in snapshots
    )
    return (
        normalized_snapshots,
        tuple(observation_texts),
        tuple(expert_targets),
    )


def _build_traces(
    cases: tuple[ETAProofCase, ...],
    *,
    runtime: OpenWeightResidualRuntime,
    label: str,
) -> tuple[TrainingTrace, ...]:
    traces: list[TrainingTrace] = []
    for case in cases:
        snapshots, texts, targets = _rate_distortion_observation_bundle(
            case, open_weight_runtime=runtime
        )
        traces.append(
            build_training_trace_from_substrate_snapshots(
                trace_id=f"rate-distortion:{label}:{case.case_id}",
                source_text=case.source_text,
                snapshots=snapshots,
                expert_action_targets=targets,
                observation_texts=texts,
            )
        )
    return tuple(traces)


def _trace_scorer_rows(
    traces: tuple[TrainingTrace, ...],
    scorer,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    texts: list[str] = []
    indices: list[int] = []
    for trace in traces:
        for step in trace.steps:
            if step.expert_action_target is None:
                raise ValueError(
                    f"Trace {trace.trace_id!r} step {step.step} lacks an "
                    "expert action target."
                )
            texts.append(step.observation_text)
            indices.append(
                scorer.action_index(step.expert_action_target.action_id)
            )
    return (tuple(texts), tuple(indices))


def _baseline_distortion(
    traces: tuple[TrainingTrace, ...],
    scorer,
) -> float:
    texts, indices = _trace_scorer_rows(traces, scorer)
    values = scorer.baseline_action_nll(
        source_texts=texts, action_indices=indices
    )
    return _mean(tuple(values))


def _run_single(
    *,
    arm: str,
    alpha: float,
    seed: int,
    n_z: int,
    scorer,
    train_traces: tuple[TrainingTrace, ...],
    heldout_traces: tuple[TrainingTrace, ...],
    updates_per_run: int,
    learning_rate: float,
    substrate_learning_rate: float,
    switch_threshold: float,
    baseline_train: float,
    baseline_heldout: float,
) -> RateDistortionPoint:
    start = time.perf_counter()
    store = MetacontrollerParameterStore(
        n_z=n_z,
        initialization_seed=seed,
    )
    session = StoreSSLTrainingSession(
        n_z=n_z,
        alpha=alpha,
        learning_rate=learning_rate,
        substrate_learning_rate=substrate_learning_rate,
        switch_rate_weight=0.0,
        switch_binary_weight=0.0,
        switch_group_weight=0.0,
        proposal_prediction_weight=0.0,
        gate_choice_weight=0.0,
        action_scorer=scorer,
        reparam_seed=seed * 1_000_003 + 17,
    )
    final_report = None
    for update_index in range(updates_per_run):
        final_report = session.train_batch(
            store=store,
            traces=train_traces,
            batch_id=f"rd:{arm}:a{alpha}:s{seed}:u{update_index}",
            switch_threshold=switch_threshold,
            write_back=False,
        )
        if not math.isfinite(final_report.total_loss):
            raise RuntimeError(
                f"Non-finite SSL loss in arm={arm} alpha={alpha} seed={seed} "
                f"update={update_index}."
            )
    if final_report is None:
        raise RuntimeError("updates_per_run must be at least 1.")
    train_eval: StoreSSLEvaluationReport = session.evaluate_batch(
        store=store,
        traces=train_traces,
        batch_id=f"rd:{arm}:a{alpha}:s{seed}:train-eval",
        switch_threshold=switch_threshold,
    )
    heldout_eval: StoreSSLEvaluationReport = session.evaluate_batch(
        store=store,
        traces=heldout_traces,
        batch_id=f"rd:{arm}:a{alpha}:s{seed}:heldout-eval",
        switch_threshold=switch_threshold,
    )
    return RateDistortionPoint(
        arm=arm,
        alpha=alpha,
        seed=seed,
        train_rate=train_eval.kl_rate,
        train_distortion=train_eval.distortion,
        heldout_rate=heldout_eval.kl_rate,
        heldout_distortion=heldout_eval.distortion,
        baseline_train_distortion=baseline_train,
        baseline_heldout_distortion=baseline_heldout,
        mean_switch_probability=train_eval.mean_switch_probability,
        hard_switch_frequency=train_eval.hard_switch_frequency,
        train_boundary_f1=train_eval.boundary_f1,
        heldout_boundary_f1=heldout_eval.boundary_f1,
        optimizer_steps=session.optimizer_step,
        final_total_loss=final_report.total_loss,
        final_grad_norm=final_report.grad_norm,
        wall_seconds=time.perf_counter() - start,
        description=(
            f"arm={arm} alpha={alpha} seed={seed} "
            f"rate={train_eval.kl_rate:.4f} "
            f"distortion={train_eval.distortion:.4f}"
        ),
    )


def _aggregate_curve(
    points: tuple[RateDistortionPoint, ...],
    *,
    arm: str,
    alpha_grid: tuple[float, ...],
) -> tuple[RateDistortionCurvePoint, ...]:
    rows: list[RateDistortionCurvePoint] = []
    for alpha in alpha_grid:
        cell = tuple(
            point
            for point in points
            if point.arm == arm and point.alpha == alpha
        )
        if not cell:
            raise RuntimeError(
                f"Missing sweep cell arm={arm} alpha={alpha}."
            )
        rows.append(
            RateDistortionCurvePoint(
                arm=arm,
                alpha=alpha,
                rate_mean=_mean(tuple(p.train_rate for p in cell)),
                rate_std=_std(tuple(p.train_rate for p in cell)),
                distortion_mean=_mean(
                    tuple(p.train_distortion for p in cell)
                ),
                distortion_std=_std(
                    tuple(p.train_distortion for p in cell)
                ),
                heldout_distortion_mean=_mean(
                    tuple(p.heldout_distortion for p in cell)
                ),
                boundary_f1_mean=_mean(
                    tuple(p.train_boundary_f1 for p in cell)
                ),
                switch_frequency_mean=_mean(
                    tuple(p.hard_switch_frequency for p in cell)
                ),
                seed_count=len(cell),
            )
        )
    return tuple(rows)


def assess_gap(
    curve: tuple[RateDistortionCurvePoint, ...],
    *,
    arm: str,
    drop_share_threshold: float = 0.5,
    rate_share_threshold: float = 0.25,
    noise_multiple: float = 2.0,
) -> GapAssessment:
    """Detect the paper's near-vertical segment on the rate-ordered curve."""

    ordered = tuple(sorted(curve, key=lambda row: row.rate_mean))
    if len(ordered) < 3:
        raise ValueError("Gap assessment requires at least three grid cells.")
    distortions = tuple(row.distortion_mean for row in ordered)
    rates = tuple(row.rate_mean for row in ordered)
    distortion_span = max(distortions) - min(distortions)
    rate_span = max(rates) - min(rates)
    noise_scale = _mean(tuple(row.distortion_std for row in ordered))
    best_drop = 0.0
    best_drop_share = 0.0
    best_rate_share = 1.0
    gap_low_alpha = 0.0
    gap_high_alpha = 0.0
    for left, right in zip(ordered, ordered[1:], strict=False):
        drop = left.distortion_mean - right.distortion_mean
        if drop <= best_drop:
            continue
        best_drop = drop
        best_drop_share = (
            drop / distortion_span if distortion_span > 1e-12 else 0.0
        )
        best_rate_share = (
            (right.rate_mean - left.rate_mean) / rate_span
            if rate_span > 1e-12
            else 1.0
        )
        gap_low_alpha = left.alpha
        gap_high_alpha = right.alpha
    gap_detected = (
        distortion_span > noise_multiple * max(noise_scale, 1e-9)
        and best_drop_share >= drop_share_threshold
        and best_rate_share <= rate_share_threshold
    )
    in_gap = {gap_low_alpha, gap_high_alpha} if gap_detected else set()
    gap_f1 = _mean(
        tuple(
            row.boundary_f1_mean for row in ordered if row.alpha in in_gap
        )
    )
    outside_f1 = _mean(
        tuple(
            row.boundary_f1_mean
            for row in ordered
            if row.alpha not in in_gap
        )
    )
    return GapAssessment(
        arm=arm,
        gap_detected=gap_detected,
        distortion_span=distortion_span,
        rate_span=rate_span,
        noise_scale=noise_scale,
        max_drop=best_drop,
        max_drop_share=best_drop_share,
        max_drop_rate_share=best_rate_share,
        gap_low_alpha=gap_low_alpha,
        gap_high_alpha=gap_high_alpha,
        drop_share_threshold=drop_share_threshold,
        rate_share_threshold=rate_share_threshold,
        noise_multiple=noise_multiple,
        boundary_f1_gap_region=gap_f1,
        boundary_f1_outside_gap=outside_f1,
        description=(
            f"arm={arm} gap_detected={gap_detected} "
            f"max_drop={best_drop:.4f} ({best_drop_share:.2%} of span) over "
            f"{best_rate_share:.2%} of the rate span between "
            f"alpha={gap_low_alpha} and alpha={gap_high_alpha}."
        ),
    )


@dataclass(frozen=True)
class RateDistortionAdjudication:
    """The retain/kill decision derived purely from the aggregate curves.

    Kept separate from the sweep so an auditor can recompute the verdict
    from ``curves.json`` and ``gap_assessments.json`` alone, without the
    model or an accelerator.
    """

    arms_distinguishable: bool
    arm_separation: float
    arm_separation_threshold: float
    verdict: str
    verdict_reason: str


def adjudicate_rate_distortion(
    curves: tuple[RateDistortionCurvePoint, ...],
    gaps: tuple[GapAssessment, ...],
    *,
    arms: tuple[str, ...],
) -> RateDistortionAdjudication:
    """Map aggregate curves and gap assessments onto the frozen verdict set."""

    gap_by_arm = {gap.arm: gap for gap in gaps}
    pooled_std = _mean(tuple(row.distortion_std for row in curves))
    arm_separation_threshold = max(2.0 * pooled_std, 0.02)
    if set(arms) == set(_ARMS):
        frozen_curve = tuple(row for row in curves if row.arm == "frozen")
        joint_curve = tuple(row for row in curves if row.arm == "joint")
        arm_separation = _arm_separation(frozen_curve, joint_curve)
        arms_distinguishable = arm_separation >= arm_separation_threshold
    else:
        arm_separation = 0.0
        arms_distinguishable = False

    frozen_gap = gap_by_arm.get("frozen")
    joint_gap = gap_by_arm.get("joint")
    if set(arms) != set(_ARMS):
        verdict = "incomplete-sweep"
        verdict_reason = (
            "Both arms are required for the validity control; ran only "
            f"{arms}."
        )
    elif not arms_distinguishable:
        verdict = "instrument-invalid"
        verdict_reason = (
            "Frozen and joint curves are indistinguishable "
            f"(max separation {arm_separation:.4f} < threshold "
            f"{arm_separation_threshold:.4f}); no thesis conclusion may be "
            "drawn this round."
        )
    elif (
        frozen_gap is not None
        and joint_gap is not None
        and frozen_gap.gap_detected
        and not joint_gap.gap_detected
    ):
        if (
            frozen_gap.boundary_f1_gap_region
            > frozen_gap.boundary_f1_outside_gap
        ):
            verdict = "retain-eta"
            verdict_reason = (
                "Frozen arm shows the predicted gap, the joint arm does "
                "not, and beta boundary F1 is higher inside the gap region."
            )
        else:
            verdict = "retain-weak"
            verdict_reason = (
                "Frozen arm shows the predicted gap and the joint arm does "
                "not, but beta boundary F1 inside the gap region is not "
                "higher than outside."
            )
    elif frozen_gap is not None and not frozen_gap.gap_detected:
        verdict = "kill-eta"
        verdict_reason = (
            "The instrument passed the joint-arm validity control but the "
            "frozen arm shows no rate-distortion gap across the alpha grid."
        )
    else:
        verdict = "inconclusive-joint-arm-gap"
        verdict_reason = (
            "The joint arm also shows a gap, contradicting the paper's "
            "prediction; the instrument is suspect."
        )
    return RateDistortionAdjudication(
        arms_distinguishable=arms_distinguishable,
        arm_separation=arm_separation,
        arm_separation_threshold=arm_separation_threshold,
        verdict=verdict,
        verdict_reason=verdict_reason,
    )


def _arm_separation(
    frozen_curve: tuple[RateDistortionCurvePoint, ...],
    joint_curve: tuple[RateDistortionCurvePoint, ...],
) -> float:
    by_alpha = {row.alpha: row for row in joint_curve}
    gaps = tuple(
        abs(row.distortion_mean - by_alpha[row.alpha].distortion_mean)
        for row in frozen_curve
    )
    return max(gaps) if gaps else 0.0


class RateDistortionPointCache(Protocol):
    """Per-cell resume journal owned by the caller, not by this module.

    A sweep is 2 arms x |alpha grid| x |seeds| independently trained cells,
    each costing minutes of exclusive accelerator time. Without a journal an
    interrupt discards the whole sweep, which in practice pressures the
    operator into shortening the grid.
    """

    def load_point(
        self, *, arm: str, alpha: float, seed: int
    ) -> RateDistortionPoint | None: ...

    def store_point(self, point: RateDistortionPoint) -> None: ...


def run_eta_rate_distortion_evidence(
    *,
    alpha_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
    seed_schedule: tuple[int, ...] = (0, 1, 2),
    n_z: int = 16,
    updates_per_run: int = 40,
    learning_rate: float = 0.02,
    substrate_learning_rate: float = 1e-4,
    switch_threshold: float = 0.55,
    open_weight_config: ETAOpenWeightRuntimeConfig | None = None,
    arms: tuple[str, ...] = _ARMS,
    point_cache: RateDistortionPointCache | None = None,
) -> RateDistortionEvidenceReport:
    if len(alpha_grid) < 3:
        raise ValueError("alpha_grid needs at least three values.")
    if len(set(alpha_grid)) != len(alpha_grid):
        raise ValueError("alpha_grid values must be unique.")
    if not seed_schedule:
        raise ValueError("seed_schedule must contain at least one seed.")
    unknown_arms = set(arms) - set(_ARMS)
    if unknown_arms or not arms:
        raise ValueError(f"arms must be a subset of {_ARMS}, got {arms!r}.")
    config = open_weight_config or ETAOpenWeightRuntimeConfig(device="mps")
    if config.model_dtype != "float32":
        # fp16 master weights overflow under Adam in the joint arm, and the
        # arm comparison must not be confounded by dtype; force fp32.
        config = replace(config, model_dtype="float32")
    runtime = _build_eta_open_weight_runtime(config)
    _validate_eta_open_weight_runtime(runtime=runtime, config=config)

    environment = build_default_eta_proof_environment()
    options = _action_options(environment)
    cases = default_eta_proof_cases()
    train_cases = tuple(case for case in cases if case.split == "train")
    heldout_cases = tuple(case for case in cases if case.split != "train")
    if not train_cases or not heldout_cases:
        raise RuntimeError(
            "Rate-distortion sweep needs both train and heldout proof cases."
        )
    train_traces = _build_traces(train_cases, runtime=runtime, label="train")
    heldout_traces = _build_traces(
        heldout_cases, runtime=runtime, label="heldout"
    )
    train_step_count = sum(len(trace.steps) for trace in train_traces)
    heldout_step_count = sum(len(trace.steps) for trace in heldout_traces)

    points: list[RateDistortionPoint] = []
    frozen_scorer = None
    injection_layer_index = -1
    control_norm_cap = 0.0
    probe_hidden_norm = 0.0
    for arm in arms:
        joint = arm == "joint"
        scorer = runtime.build_steered_action_scorer(
            action_options=options,
            joint_training=joint,
        )
        if not joint:
            frozen_scorer = scorer
        injection_layer_index = scorer.injection_layer_index
        control_norm_cap = scorer.control_norm_cap
        probe_hidden_norm = scorer.probe_hidden_norm
        try:
            for alpha in alpha_grid:
                for seed in seed_schedule:
                    if point_cache is not None:
                        cached = point_cache.load_point(
                            arm=arm, alpha=alpha, seed=seed
                        )
                        if cached is not None:
                            points.append(cached)
                            continue
                    if joint:
                        # Every cell starts from pristine upper blocks, so a
                        # resumed sweep is identical to an uninterrupted one.
                        scorer.reset_joint_parameters()
                    baseline_train = _baseline_distortion(
                        train_traces, scorer
                    )
                    baseline_heldout = _baseline_distortion(
                        heldout_traces, scorer
                    )
                    point = _run_single(
                        arm=arm,
                        alpha=alpha,
                        seed=seed,
                        n_z=n_z,
                        scorer=scorer,
                        train_traces=train_traces,
                        heldout_traces=heldout_traces,
                        updates_per_run=updates_per_run,
                        learning_rate=learning_rate,
                        substrate_learning_rate=substrate_learning_rate,
                        switch_threshold=switch_threshold,
                        baseline_train=baseline_train,
                        baseline_heldout=baseline_heldout,
                    )
                    if point_cache is not None:
                        point_cache.store_point(point)
                    points.append(point)
        finally:
            if joint:
                # Never leave the shared frozen runtime dirty, even if a
                # sweep cell raised.
                scorer.restore_and_freeze()
    del frozen_scorer

    point_tuple = tuple(points)
    curves: list[RateDistortionCurvePoint] = []
    gaps: list[GapAssessment] = []
    for arm in arms:
        arm_curve = _aggregate_curve(
            point_tuple, arm=arm, alpha_grid=alpha_grid
        )
        curves.extend(arm_curve)
        gaps.append(assess_gap(arm_curve, arm=arm))

    adjudication = adjudicate_rate_distortion(
        tuple(curves), tuple(gaps), arms=arms
    )

    return RateDistortionEvidenceReport(
        schema_version=RATE_DISTORTION_SCHEMA_VERSION,
        model_id=runtime.model_id,
        device=config.device,
        runtime_origin=str(runtime.runtime_origin),
        fallback_active=bool(runtime.fallback_active),
        injection_layer_index=injection_layer_index,
        control_norm_cap=control_norm_cap,
        probe_hidden_norm=probe_hidden_norm,
        n_z=n_z,
        alpha_grid=alpha_grid,
        seed_schedule=seed_schedule,
        updates_per_run=updates_per_run,
        learning_rate=learning_rate,
        substrate_learning_rate=substrate_learning_rate,
        switch_threshold=switch_threshold,
        arms=arms,
        observation_protocol=(
            "partially-observable-no-remaining-route.v1"
        ),
        action_vocabulary=tuple(
            option.action_id for option in options
        ),
        train_case_ids=tuple(case.case_id for case in train_cases),
        heldout_case_ids=tuple(case.case_id for case in heldout_cases),
        train_step_count=train_step_count,
        heldout_step_count=heldout_step_count,
        points=point_tuple,
        curves=tuple(curves),
        gaps=tuple(gaps),
        arms_distinguishable=adjudication.arms_distinguishable,
        arm_separation=adjudication.arm_separation,
        arm_separation_threshold=adjudication.arm_separation_threshold,
        verdict=adjudication.verdict,
        verdict_reason=adjudication.verdict_reason,
        description=(
            "ETA Eq.3 rate-distortion criterion: "
            f"{len(arms)} arms x {len(alpha_grid)} alphas x "
            f"{len(seed_schedule)} seeds, {updates_per_run} updates per "
            f"run, verdict={adjudication.verdict}."
        ),
    )
