from __future__ import annotations

import math
from dataclasses import dataclass

from volvence_zero.substrate import SubstrateSnapshot, SurfaceKind, feature_signal_value


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mean_abs(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def _project_signal(seed: tuple[float, ...], *, dim: int) -> tuple[float, ...]:
    if dim <= 0:
        return ()
    if not seed:
        return tuple(0.0 for _ in range(dim))
    return tuple(_clamp(seed[index % len(seed)]) for index in range(dim))


@dataclass(frozen=True)
class RuntimeBackboneEvidence:
    signal: tuple[float, ...]
    signal_norm: float
    signal_quality: float
    runtime_strength: float
    hook_coverage: float
    fallback_active: float
    residual_stream_active: float
    sequence_density: float
    activation_density: float
    semantic_residual_weight: float
    description: str = ""


def build_runtime_backbone_evidence(
    *,
    substrate_snapshot: SubstrateSnapshot | None,
    dim: int,
) -> RuntimeBackboneEvidence:
    if substrate_snapshot is None:
        zero_signal = tuple(0.0 for _ in range(max(dim, 0)))
        return RuntimeBackboneEvidence(
            signal=zero_signal,
            signal_norm=0.0,
            signal_quality=0.0,
            runtime_strength=0.0,
            hook_coverage=0.0,
            fallback_active=0.0,
            residual_stream_active=0.0,
            sequence_density=0.0,
            activation_density=0.0,
            semantic_residual_weight=0.0,
            description="Runtime backbone evidence missing because substrate snapshot is absent.",
        )

    residual_means = tuple(
        _mean(tuple(abs(value) for value in activation.activation))
        for activation in substrate_snapshot.residual_activations
        if activation.activation
    )
    feature_means = tuple(
        _mean(tuple(abs(value) for value in feature.values))
        for feature in substrate_snapshot.feature_surface
        if feature.values
    )
    residual_average = _clamp(_mean(residual_means))
    residual_peak = _clamp(max(residual_means, default=0.0))
    residual_spread = _clamp(
        residual_peak - _clamp(min(residual_means, default=residual_peak))
    )
    feature_average = _clamp(_mean(feature_means))
    feature_peak = _clamp(max(feature_means, default=0.0))
    residual_stream_active = float(
        substrate_snapshot.surface_kind is SurfaceKind.RESIDUAL_STREAM
        or bool(substrate_snapshot.residual_activations)
    )
    sequence_density = _clamp(len(substrate_snapshot.residual_sequence) / 6.0)
    activation_density = _clamp(len(substrate_snapshot.residual_activations) / 12.0)
    hook_coverage = _clamp(feature_signal_value(substrate_snapshot.feature_surface, name="hook_layer_coverage"))
    fallback_active = _clamp(feature_signal_value(substrate_snapshot.feature_surface, name="fallback_active"))
    semantic_residual_weight = _clamp(
        feature_signal_value(substrate_snapshot.feature_surface, name="semantic_residual_weight")
    )
    top_logit_margin = _clamp(feature_signal_value(substrate_snapshot.feature_surface, name="top_logit_margin"))
    top_logit_entropy = _clamp(feature_signal_value(substrate_snapshot.feature_surface, name="top_logit_entropy"))
    task_pull = _clamp(feature_signal_value(substrate_snapshot.feature_surface, name="semantic_task_pull"))
    support_pull = _clamp(feature_signal_value(substrate_snapshot.feature_surface, name="semantic_support_pull"))
    repair_pull = _clamp(feature_signal_value(substrate_snapshot.feature_surface, name="semantic_repair_pull"))
    semantic_drive = _clamp(_mean((task_pull, support_pull, repair_pull)))
    runtime_strength = _clamp(
        residual_stream_active * 0.24
        + sequence_density * 0.14
        + activation_density * 0.14
        + hook_coverage * 0.22
        + (1.0 - fallback_active) * 0.16
        + semantic_residual_weight * 0.10
    )
    signal_quality = _clamp(
        runtime_strength * 0.65
        + residual_peak * 0.10
        + top_logit_margin * 0.15
        + (1.0 - top_logit_entropy) * 0.10
    )
    seed = (
        residual_average,
        residual_peak,
        residual_spread,
        feature_average,
        feature_peak,
        sequence_density,
        activation_density,
        hook_coverage,
        1.0 - fallback_active,
        semantic_residual_weight,
        semantic_drive,
        top_logit_margin,
        1.0 - top_logit_entropy,
    )
    signal = _project_signal(seed, dim=dim)
    return RuntimeBackboneEvidence(
        signal=signal,
        signal_norm=_mean_abs(signal),
        signal_quality=signal_quality,
        runtime_strength=runtime_strength,
        hook_coverage=hook_coverage,
        fallback_active=fallback_active,
        residual_stream_active=residual_stream_active,
        sequence_density=sequence_density,
        activation_density=activation_density,
        semantic_residual_weight=semantic_residual_weight,
        description=(
            f"Runtime backbone evidence quality={signal_quality:.3f}, strength={runtime_strength:.3f}, "
            f"hook_coverage={hook_coverage:.3f}, residual_stream={residual_stream_active:.0f}."
        ),
    )


def cosine_alignment(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    if length <= 0:
        return 0.0
    left_trimmed = left[:length]
    right_trimmed = right[:length]
    left_norm = math.sqrt(sum(value * value for value in left_trimmed))
    right_norm = math.sqrt(sum(value * value for value in right_trimmed))
    if left_norm <= 1e-6 or right_norm <= 1e-6:
        return 0.0
    return sum(l * r for l, r in zip(left_trimmed, right_trimmed, strict=True)) / (left_norm * right_norm)
