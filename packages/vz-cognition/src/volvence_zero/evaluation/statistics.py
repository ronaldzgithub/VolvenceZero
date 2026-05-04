"""Evaluation statistical helpers.

Clamping, sample-standard-deviation, percentile, confidence-interval
summary, and pairwise-metric-effect computation used throughout
:class:`EvaluationBackbone` scoring and paper-suite runners.

Slice S.2 (2026-05-04): extracted from ``evaluation/backbone.py``.
"""

from __future__ import annotations

import math
from typing import Mapping

from volvence_zero.evaluation.types import (
    MetricIntervalSummary,
    PairwiseMetricEffect,
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sample_std(values: tuple[float, ...]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    clamped = max(0.0, min(1.0, percentile))
    position = (len(ordered) - 1) * clamped
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def build_metric_interval_summary(
    *,
    metric_name: str,
    values: tuple[float, ...],
    ci_level: float = 0.95,
) -> MetricIntervalSummary:
    sample_count = len(values)
    if sample_count == 0:
        return MetricIntervalSummary(
            metric_name=metric_name,
            sample_count=0,
            mean=0.0,
            std=0.0,
            stderr=0.0,
            ci_low=0.0,
            ci_high=0.0,
            min_value=0.0,
            max_value=0.0,
            description=f"{metric_name} has no samples.",
        )
    mean = sum(values) / sample_count
    std = _sample_std(values)
    stderr = std / math.sqrt(sample_count) if sample_count > 0 else 0.0
    tail_mass = (1.0 - ci_level) / 2.0
    ci_low = _percentile(values, tail_mass)
    ci_high = _percentile(values, 1.0 - tail_mass)
    return MetricIntervalSummary(
        metric_name=metric_name,
        sample_count=sample_count,
        mean=mean,
        std=std,
        stderr=stderr,
        ci_low=ci_low,
        ci_high=ci_high,
        min_value=min(values),
        max_value=max(values),
        description=(
            f"{metric_name} summarized over {sample_count} samples "
            f"with mean={mean:.3f}, std={std:.3f}, ci=[{ci_low:.3f}, {ci_high:.3f}]."
        ),
    )


def build_metric_interval_summaries(
    *,
    metric_samples: Mapping[str, tuple[float, ...]],
    ci_level: float = 0.95,
) -> tuple[MetricIntervalSummary, ...]:
    return tuple(
        build_metric_interval_summary(
            metric_name=metric_name,
            values=values,
            ci_level=ci_level,
        )
        for metric_name, values in metric_samples.items()
    )


def build_pairwise_metric_effect(
    *,
    metric_name: str,
    candidate_label: str,
    control_label: str,
    candidate_values: tuple[float, ...],
    control_values: tuple[float, ...],
    ci_level: float = 0.95,
) -> PairwiseMetricEffect:
    sample_count = min(len(candidate_values), len(control_values))
    if sample_count == 0:
        return PairwiseMetricEffect(
            metric_name=metric_name,
            candidate_label=candidate_label,
            control_label=control_label,
            sample_count=0,
            mean_delta=0.0,
            std_delta=0.0,
            stderr_delta=0.0,
            ci_low=0.0,
            ci_high=0.0,
            effect_size=0.0,
            description=f"{metric_name} has no aligned samples for pairwise comparison.",
        )
    deltas = tuple(
        candidate_value - control_value
        for candidate_value, control_value in zip(
            candidate_values[:sample_count],
            control_values[:sample_count],
            strict=True,
        )
    )
    delta_summary = build_metric_interval_summary(
        metric_name=metric_name,
        values=deltas,
        ci_level=ci_level,
    )
    pooled_std = math.sqrt(
        max(
            (_sample_std(candidate_values[:sample_count]) ** 2 + _sample_std(control_values[:sample_count]) ** 2)
            / 2.0,
            0.0,
        )
    )
    effect_size = delta_summary.mean / pooled_std if pooled_std > 1e-8 else 0.0
    return PairwiseMetricEffect(
        metric_name=metric_name,
        candidate_label=candidate_label,
        control_label=control_label,
        sample_count=sample_count,
        mean_delta=delta_summary.mean,
        std_delta=delta_summary.std,
        stderr_delta=delta_summary.stderr,
        ci_low=delta_summary.ci_low,
        ci_high=delta_summary.ci_high,
        effect_size=effect_size,
        description=(
            f"{candidate_label} vs {control_label} on {metric_name} "
            f"has mean_delta={delta_summary.mean:.3f} and effect_size={effect_size:.3f}."
        ),
    )
