"""S2 no-bias causal steering over the frozen S1 residual axes.

The experiment is intentionally read-only: no controller is fit, no bias is
added, and no model parameter receives a gradient.  On held-out route prefixes
it compares the target active-subgoal axis against its sign reversal, zero
control, and three deterministic class-axis mismatches.  Effects are averaged
within route before bootstrap inference so repeated steps are not treated as
independent samples.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics
from typing import Any

from volvence_zero.agent.eta_proof_benchmark import ETAProbeRow, ETAProofCorpus
from volvence_zero.agent.eta_rate_distortion_evidence import (
    OBSERVATION_PROTOCOL_V4,
    _action_options,
    _rate_distortion_observation_texts,
    eta_stage2_probe_rows,
)
from volvence_zero.substrate import (
    FrozenResidualReadoutArtifact,
    OpenWeightResidualRuntime,
)


ETA_S2_CAUSAL_STEERING_SCHEMA_VERSION = "eta-s2-causal-steering-evidence.v1"


@dataclass(frozen=True)
class S2CausalSteeringThresholds:
    min_plus_vs_noop_nll_gain: float = 0.02
    min_plus_vs_minus_nll_contrast: float = 0.04
    min_plus_vs_shuffled_nll_contrast: float = 0.02
    min_route_win_rate: float = 0.65
    require_bootstrap_lower_positive: bool = True


@dataclass(frozen=True)
class S2EffectEstimate:
    mean: float
    ci_lower: float
    ci_upper: float
    route_count: int


@dataclass(frozen=True)
class S2ScaleMetrics:
    scale_fraction: float
    control_norm: float
    noop_mean_nll: float
    plus_mean_nll: float
    minus_mean_nll: float
    shuffled_mean_nll: float
    plus_vs_noop: S2EffectEstimate
    plus_vs_minus: S2EffectEstimate
    plus_vs_shuffled: S2EffectEstimate
    plus_vs_noop_route_win_rate: float
    plus_vs_minus_route_win_rate: float
    plus_vs_shuffled_route_win_rate: float
    plus_vs_noop_row_win_rate: float
    row_count: int


@dataclass(frozen=True)
class S2SteeringPoint:
    sample_id: str
    case_id: str
    step_index: int
    active_subgoal: str
    expert_action_id: str
    scale_fraction: float
    noop_nll: float
    plus_nll: float
    minus_nll: float
    shuffled_nll: float
    shuffled_axis_class_ids: tuple[str, ...]
    shuffled_axis_nlls: tuple[float, ...]


@dataclass(frozen=True)
class S2CausalSteeringAdmission:
    admitted: bool
    condition_plus_vs_noop: bool
    condition_plus_vs_minus: bool
    condition_plus_vs_shuffled: bool
    condition_route_win_rate: bool
    condition_bootstrap: bool
    failed_conditions: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class S2CausalSteeringReport:
    schema_version: str
    claim_scope: str
    source_s1_artifact_id: str
    model_id: str
    model_source: str
    device: str
    observation_protocol: str
    corpus_seed: int
    heldout_route_count: int
    heldout_row_count: int
    injection_layer_index: int
    hidden_size: int
    control_norm_ratio: float
    control_norm_cap: float
    probe_hidden_norm: float
    probe_train_row_count: int
    max_observed_source_tokens: int
    scorer_max_length: int
    truncated_row_count: int
    scale_fractions: tuple[float, ...]
    primary_scale_fraction: float
    shuffled_class_shifts: tuple[int, ...]
    bootstrap_seed: int
    bootstrap_resamples: int
    bootstrap_confidence: float
    thresholds: S2CausalSteeringThresholds
    scale_metrics: tuple[S2ScaleMetrics, ...]
    primary_admission: S2CausalSteeringAdmission
    trainable_parameter_count: int
    free_bias_present: bool
    production_wiring_changed: bool
    feedback_to_learning: bool
    description: str = ""


@dataclass(frozen=True)
class _SteeringRow:
    probe: ETAProbeRow
    active_subgoal: str
    expert_action_id: str

    @property
    def sample_id(self) -> str:
        return (
            f"{self.probe.split}:{self.probe.case_id}:step-{self.probe.step_index}"
        )


def _heldout_steering_rows(
    corpus: ETAProofCorpus,
) -> tuple[tuple[_SteeringRow, ...], tuple[str, ...]]:
    probe_rows, class_ids = eta_stage2_probe_rows(
        corpus.heldout_cases,
        environment=corpus.environment,
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    target_map: dict[tuple[str, int], tuple[str, str]] = {}
    for case in corpus.heldout_cases:
        _texts, targets, subgoals = _rate_distortion_observation_texts(
            case,
            environment=corpus.environment,
            protocol_version=OBSERVATION_PROTOCOL_V4,
        )
        for step_index, (target, subgoal) in enumerate(
            zip(targets, subgoals, strict=True)
        ):
            if subgoal is None:
                continue
            target_map[(case.case_id, step_index)] = (
                subgoal,
                target.action_id,
            )
    rows: list[_SteeringRow] = []
    for probe in probe_rows:
        try:
            subgoal, action_id = target_map[(probe.case_id, probe.step_index)]
        except KeyError:
            raise RuntimeError(
                "S2 probe/action lineage is incomplete for "
                f"{probe.case_id!r} step {probe.step_index}"
            ) from None
        if class_ids[probe.subgoal_label] != subgoal:
            raise RuntimeError(
                "S2 probe label and active-subgoal owner disagree for "
                f"{probe.case_id!r} step {probe.step_index}"
            )
        rows.append(
            _SteeringRow(
                probe=probe,
                active_subgoal=subgoal,
                expert_action_id=action_id,
            )
        )
    if not rows or len({row.sample_id for row in rows}) != len(rows):
        raise RuntimeError("S2 heldout steering rows must be non-empty and unique")
    return tuple(rows), class_ids


def _route_values(
    points: tuple[S2SteeringPoint, ...],
    *,
    value: Any,
) -> tuple[float, ...]:
    grouped: dict[str, list[float]] = {}
    for point in points:
        grouped.setdefault(point.case_id, []).append(float(value(point)))
    return tuple(
        statistics.fmean(grouped[case_id]) for case_id in sorted(grouped)
    )


def _bootstrap_effect(
    route_effects: tuple[float, ...],
    *,
    seed: int,
    resamples: int,
    confidence: float,
) -> S2EffectEstimate:
    if not route_effects:
        raise ValueError("S2 bootstrap requires route effects")
    if resamples < 100:
        raise ValueError("S2 bootstrap requires at least 100 resamples")
    if not 0.0 < confidence < 1.0:
        raise ValueError("S2 bootstrap confidence must be in (0, 1)")
    rng = random.Random(seed)
    count = len(route_effects)
    draws = sorted(
        statistics.fmean(route_effects[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    lower_index = min(max(int(math.floor(tail * resamples)), 0), resamples - 1)
    upper_index = min(
        max(int(math.ceil((1.0 - tail) * resamples)) - 1, 0),
        resamples - 1,
    )
    return S2EffectEstimate(
        mean=statistics.fmean(route_effects),
        ci_lower=draws[lower_index],
        ci_upper=draws[upper_index],
        route_count=count,
    )


def _scale_metrics(
    *,
    points: tuple[S2SteeringPoint, ...],
    control_norm: float,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
) -> S2ScaleMetrics:
    if not points:
        raise ValueError("S2 scale metrics require points")
    scale = points[0].scale_fraction
    if any(point.scale_fraction != scale for point in points):
        raise ValueError("S2 scale metrics received mixed scale fractions")
    plus_noop_rows = tuple(point.noop_nll - point.plus_nll for point in points)
    plus_noop_routes = _route_values(
        points,
        value=lambda point: point.noop_nll - point.plus_nll,
    )
    plus_minus_routes = _route_values(
        points,
        value=lambda point: point.minus_nll - point.plus_nll,
    )
    plus_shuffle_routes = _route_values(
        points,
        value=lambda point: point.shuffled_nll - point.plus_nll,
    )
    noop_routes = _route_values(points, value=lambda point: point.noop_nll)
    plus_routes = _route_values(points, value=lambda point: point.plus_nll)
    minus_routes = _route_values(points, value=lambda point: point.minus_nll)
    shuffle_routes = _route_values(points, value=lambda point: point.shuffled_nll)
    return S2ScaleMetrics(
        scale_fraction=scale,
        control_norm=control_norm,
        noop_mean_nll=statistics.fmean(noop_routes),
        plus_mean_nll=statistics.fmean(plus_routes),
        minus_mean_nll=statistics.fmean(minus_routes),
        shuffled_mean_nll=statistics.fmean(shuffle_routes),
        plus_vs_noop=_bootstrap_effect(
            plus_noop_routes,
            seed=bootstrap_seed + 11,
            resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
        ),
        plus_vs_minus=_bootstrap_effect(
            plus_minus_routes,
            seed=bootstrap_seed + 23,
            resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
        ),
        plus_vs_shuffled=_bootstrap_effect(
            plus_shuffle_routes,
            seed=bootstrap_seed + 37,
            resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
        ),
        plus_vs_noop_route_win_rate=(
            sum(value > 0.0 for value in plus_noop_routes) / len(plus_noop_routes)
        ),
        plus_vs_minus_route_win_rate=(
            sum(value > 0.0 for value in plus_minus_routes) / len(plus_minus_routes)
        ),
        plus_vs_shuffled_route_win_rate=(
            sum(value > 0.0 for value in plus_shuffle_routes)
            / len(plus_shuffle_routes)
        ),
        plus_vs_noop_row_win_rate=(
            sum(value > 0.0 for value in plus_noop_rows) / len(plus_noop_rows)
        ),
        row_count=len(points),
    )


def assess_s2_causal_steering(
    *,
    primary: S2ScaleMetrics,
    thresholds: S2CausalSteeringThresholds,
) -> S2CausalSteeringAdmission:
    if thresholds.min_route_win_rate <= 0.5 or thresholds.min_route_win_rate > 1.0:
        raise ValueError("S2 min_route_win_rate must be in (0.5, 1]")
    conditions = {
        "plus-vs-noop-effect": (
            primary.plus_vs_noop.mean >= thresholds.min_plus_vs_noop_nll_gain
        ),
        "plus-vs-minus-effect": (
            primary.plus_vs_minus.mean
            >= thresholds.min_plus_vs_minus_nll_contrast
        ),
        "plus-vs-shuffled-effect": (
            primary.plus_vs_shuffled.mean
            >= thresholds.min_plus_vs_shuffled_nll_contrast
        ),
        "route-win-rate": (
            primary.plus_vs_noop_route_win_rate >= thresholds.min_route_win_rate
            and primary.plus_vs_minus_route_win_rate >= thresholds.min_route_win_rate
            and primary.plus_vs_shuffled_route_win_rate
            >= thresholds.min_route_win_rate
        ),
        "bootstrap-lower-positive": (
            not thresholds.require_bootstrap_lower_positive
            or (
                primary.plus_vs_noop.ci_lower > 0.0
                and primary.plus_vs_minus.ci_lower > 0.0
                and primary.plus_vs_shuffled.ci_lower > 0.0
            )
        ),
    }
    failed = tuple(name for name, passed in conditions.items() if not passed)
    return S2CausalSteeringAdmission(
        admitted=not failed,
        condition_plus_vs_noop=conditions["plus-vs-noop-effect"],
        condition_plus_vs_minus=conditions["plus-vs-minus-effect"],
        condition_plus_vs_shuffled=conditions["plus-vs-shuffled-effect"],
        condition_route_win_rate=conditions["route-win-rate"],
        condition_bootstrap=conditions["bootstrap-lower-positive"],
        failed_conditions=failed,
        description=(
            "S2 target residual axes show preregistered causal action steering."
            if not failed
            else "S2 causal steering blocked: " + ", ".join(failed)
        ),
    )


def run_eta_s2_causal_steering(
    *,
    corpus: ETAProofCorpus,
    runtime: OpenWeightResidualRuntime,
    artifact: FrozenResidualReadoutArtifact,
    model_source: str,
    device: str,
    scale_fractions: tuple[float, ...] = (0.25, 0.50, 1.00),
    primary_scale_fraction: float = 0.50,
    shuffled_class_shifts: tuple[int, ...] = (1, 3, 5),
    control_norm_ratio: float = 0.25,
    scorer_max_length: int = 768,
    max_observed_source_tokens: int = 0,
    probe_train_row_count: int = 16,
    batch_size: int = 32,
    bootstrap_seed: int = 20260804,
    bootstrap_resamples: int = 5000,
    bootstrap_confidence: float = 0.95,
    thresholds: S2CausalSteeringThresholds | None = None,
    progress: Any | None = None,
) -> tuple[S2CausalSteeringReport, tuple[S2SteeringPoint, ...]]:
    if not scale_fractions or tuple(sorted(set(scale_fractions))) != scale_fractions:
        raise ValueError("S2 scale_fractions must be unique and sorted")
    if any(value <= 0.0 or value > 1.0 for value in scale_fractions):
        raise ValueError("S2 scale fractions must be in (0, 1]")
    if primary_scale_fraction not in scale_fractions:
        raise ValueError("S2 primary scale must be in scale_fractions")
    if batch_size < 1 or probe_train_row_count < 1:
        raise ValueError("S2 batch and probe sizes must be positive")
    if (
        max_observed_source_tokens < 1
        or max_observed_source_tokens > scorer_max_length
    ):
        raise ValueError(
            "S2 token-budget preflight must prove every scored source fits"
        )
    if artifact.layer_indices != (20,) or artifact.representation_dim != 896:
        raise ValueError("S2 requires the admitted layer20/full-width896 S1 artifact")
    if artifact.model_fingerprint.model_id != runtime.model_id:
        raise ValueError("S2 artifact/runtime model id mismatch")
    class_count = len(artifact.class_ids)
    normalized_shifts = tuple(shift % class_count for shift in shuffled_class_shifts)
    if (
        not normalized_shifts
        or 0 in normalized_shifts
        or len(set(normalized_shifts)) != len(normalized_shifts)
    ):
        raise ValueError("S2 shuffled class shifts must be unique derangements")

    rows, class_ids = _heldout_steering_rows(corpus)
    if class_ids != artifact.class_ids:
        raise ValueError("S2 corpus class ids differ from the frozen S1 artifact")
    train_rows, train_class_ids = eta_stage2_probe_rows(
        corpus.train_cases,
        environment=corpus.environment,
        protocol_version=OBSERVATION_PROTOCOL_V4,
    )
    if train_class_ids != class_ids or len(train_rows) < probe_train_row_count:
        raise RuntimeError("S2 train-only norm probe corpus is invalid")

    scorer = runtime.build_steered_action_scorer(
        action_options=_action_options(corpus.environment),
        injection_layer_index=20,
        max_length=scorer_max_length,
        control_norm_ratio=control_norm_ratio,
        probe_texts=tuple(
            row.observation_text for row in train_rows[:probe_train_row_count]
        ),
        joint_training=False,
        prefix_cache=True,
    )
    if scorer.hidden_size != artifact.representation_dim:
        raise RuntimeError("S2 scorer hidden size differs from the S1 artifact")
    if scorer.trainable_parameters():
        raise RuntimeError("S2 frozen scorer unexpectedly exposes trainable parameters")

    import torch

    class_position = {
        class_id: index for index, class_id in enumerate(artifact.class_ids)
    }
    points: list[S2SteeringPoint] = []
    total_batches = math.ceil(len(rows) / batch_size)
    for batch_index, start in enumerate(range(0, len(rows), batch_size)):
        batch = rows[start : start + batch_size]
        source_texts = tuple(row.probe.observation_text for row in batch)
        action_indices = tuple(
            scorer.action_index(row.expert_action_id) for row in batch
        )
        noop_nll = scorer.baseline_action_nll(
            source_texts=source_texts,
            action_indices=action_indices,
        )
        target_axes = torch.tensor(
            [artifact.axis_for(row.active_subgoal) for row in batch],
            dtype=torch.float32,
        )
        for scale in scale_fractions:
            control_norm = scorer.control_norm_cap * scale
            plus_delta = target_axes * control_norm
            plus_nll = scorer.controlled_action_nll(
                source_texts=source_texts,
                control_deltas=plus_delta,
                action_indices=action_indices,
            )
            minus_nll = scorer.controlled_action_nll(
                source_texts=source_texts,
                control_deltas=-plus_delta,
                action_indices=action_indices,
            )
            shuffled_ids_by_shift: list[tuple[str, ...]] = []
            shuffled_nll_by_shift: list[tuple[float, ...]] = []
            for shift in normalized_shifts:
                shuffled_ids = tuple(
                    artifact.class_ids[
                        (class_position[row.active_subgoal] + shift) % class_count
                    ]
                    for row in batch
                )
                shuffled_delta = torch.tensor(
                    [artifact.axis_for(class_id) for class_id in shuffled_ids],
                    dtype=torch.float32,
                ) * control_norm
                shuffled_ids_by_shift.append(shuffled_ids)
                shuffled_nll_by_shift.append(
                    scorer.controlled_action_nll(
                        source_texts=source_texts,
                        control_deltas=shuffled_delta,
                        action_indices=action_indices,
                    )
                )
            for row_index, row in enumerate(batch):
                per_shift_nll = tuple(
                    values[row_index] for values in shuffled_nll_by_shift
                )
                points.append(
                    S2SteeringPoint(
                        sample_id=row.sample_id,
                        case_id=row.probe.case_id,
                        step_index=row.probe.step_index,
                        active_subgoal=row.active_subgoal,
                        expert_action_id=row.expert_action_id,
                        scale_fraction=scale,
                        noop_nll=noop_nll[row_index],
                        plus_nll=plus_nll[row_index],
                        minus_nll=minus_nll[row_index],
                        shuffled_nll=statistics.fmean(per_shift_nll),
                        shuffled_axis_class_ids=tuple(
                            values[row_index] for values in shuffled_ids_by_shift
                        ),
                        shuffled_axis_nlls=per_shift_nll,
                    )
                )
        scorer.clear_prefix_cache()
        if progress is not None:
            progress(
                f"S2 batch={batch_index + 1}/{total_batches} "
                f"rows={min(start + batch_size, len(rows))}/{len(rows)}"
            )

    ordered_points = tuple(
        sorted(points, key=lambda point: (point.scale_fraction, point.sample_id))
    )
    metrics = tuple(
        _scale_metrics(
            points=tuple(
                point
                for point in ordered_points
                if point.scale_fraction == scale
            ),
            control_norm=scorer.control_norm_cap * scale,
            bootstrap_seed=bootstrap_seed + scale_index * 1000,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_confidence=bootstrap_confidence,
        )
        for scale_index, scale in enumerate(scale_fractions)
    )
    primary = next(
        row for row in metrics if row.scale_fraction == primary_scale_fraction
    )
    active_thresholds = thresholds or S2CausalSteeringThresholds()
    admission = assess_s2_causal_steering(
        primary=primary,
        thresholds=active_thresholds,
    )
    report = S2CausalSteeringReport(
        schema_version=ETA_S2_CAUSAL_STEERING_SCHEMA_VERSION,
        claim_scope="s2-heldout-axis-causal-steering",
        source_s1_artifact_id=artifact.artifact_id,
        model_id=runtime.model_id,
        model_source=model_source,
        device=device,
        observation_protocol=OBSERVATION_PROTOCOL_V4,
        corpus_seed=corpus.seed,
        heldout_route_count=corpus.heldout_route_count,
        heldout_row_count=len(rows),
        injection_layer_index=scorer.injection_layer_index,
        hidden_size=scorer.hidden_size,
        control_norm_ratio=control_norm_ratio,
        control_norm_cap=scorer.control_norm_cap,
        probe_hidden_norm=scorer.probe_hidden_norm,
        probe_train_row_count=probe_train_row_count,
        max_observed_source_tokens=max_observed_source_tokens,
        scorer_max_length=scorer_max_length,
        truncated_row_count=0,
        scale_fractions=scale_fractions,
        primary_scale_fraction=primary_scale_fraction,
        shuffled_class_shifts=normalized_shifts,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_confidence=bootstrap_confidence,
        thresholds=active_thresholds,
        scale_metrics=metrics,
        primary_admission=admission,
        trainable_parameter_count=0,
        free_bias_present=False,
        production_wiring_changed=False,
        feedback_to_learning=False,
        description=(
            "Matched heldout no-bias steering with target/sign/noop/shuffled "
            f"controls; S2 causal admission={admission.admitted}."
        ),
    )
    return report, ordered_points


__all__ = [
    "ETA_S2_CAUSAL_STEERING_SCHEMA_VERSION",
    "S2CausalSteeringAdmission",
    "S2CausalSteeringReport",
    "S2CausalSteeringThresholds",
    "S2EffectEstimate",
    "S2ScaleMetrics",
    "S2SteeringPoint",
    "assess_s2_causal_steering",
    "run_eta_s2_causal_steering",
]
