"""Substrate residual-backend helper functions.

Small numerical / semantic helpers shared by the synthetic runtime,
the Hugging Face runtime, and the intervention backends. No class
state, no side effects; all pure functions.

Slice S.3 (2026-05-04): extracted from the previous monolithic
``residual_backend.py``.
"""

from __future__ import annotations

from dataclasses import replace

import hashlib
import math
import random
from typing import Any, Sequence

from volvence_zero.substrate.adapter import (
    ResidualActivation,
    SubstrateSnapshot,
)

from volvence_zero.substrate.residual_contracts import (
    LocalSubstrateRuntimeMode,
    SubstrateDeltaAdapterLayer,
    SubstrateFallbackMode,
    SubstrateOnlineFastCheckpoint,
    SubstrateRareHeavyCheckpoint,
)


def _clamp_signed(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _summarize_activations(residual_activations: tuple[ResidualActivation, ...]) -> tuple[float, float, float]:
    if not residual_activations:
        return (0.0, 0.0, 0.0)
    values = [
        sum(activation.activation) / len(activation.activation)
        for activation in residual_activations
        if activation.activation
    ]
    if not values:
        return (0.0, 0.0, 0.0)
    average = sum(values) / len(values)
    maximum = max(values)
    spread = maximum - min(values)
    return (_clamp_unit(average), _clamp_unit(maximum), _clamp_unit(spread))


def _summarize_real_activations(residual_activations: tuple[ResidualActivation, ...]) -> tuple[float, float, float]:
    flat_values = tuple(
        value
        for activation in residual_activations
        for value in activation.activation
    )
    if not flat_values:
        return (0.0, 0.0, 0.0)
    mean_abs = sum(abs(value) for value in flat_values) / len(flat_values)
    max_abs = max(abs(value) for value in flat_values)
    signed_mean = sum(flat_values) / len(flat_values)
    return (
        math.tanh(mean_abs),
        math.tanh(max_abs),
        math.tanh(signed_mean),
    )


def _normalized_entropy(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    epsilon = 1e-9
    entropy = -sum(value * math.log(max(value, epsilon)) for value in values if value > 0.0)
    max_entropy = math.log(max(len(values), 1))
    if max_entropy <= epsilon:
        return 0.0
    return _clamp_unit(entropy / max_entropy)


def _softmax_probabilities(values: Sequence[float], *, temperature: float = 1.0) -> tuple[float, ...]:
    if not values:
        return ()
    safe_temperature = max(temperature, 1e-6)
    shifted = [value / safe_temperature for value in values]
    maximum = max(shifted)
    exponentials = [math.exp(value - maximum) for value in shifted]
    total = sum(exponentials)
    if total <= 1e-9:
        return tuple(1.0 / len(values) for _ in values)
    return tuple(value / total for value in exponentials)


def _semantic_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    ascii_buffer: list[str] = []
    lowered = text.lower()
    compact = "".join(char for char in lowered if not char.isspace())
    for char in lowered:
        if char.isascii() and char.isalnum():
            ascii_buffer.append(char)
            continue
        if ascii_buffer:
            tokens.append("".join(ascii_buffer))
            ascii_buffer.clear()
        if not char.isspace():
            tokens.append(char)
    if ascii_buffer:
        tokens.append("".join(ascii_buffer))
    for width in (2, 3, 4, 5):
        tokens.extend(compact[index : index + width] for index in range(max(len(compact) - width + 1, 0)))
    return tuple(tokens)


def _normalize_vector(values: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-9:
        return tuple(0.0 for _ in values)
    return tuple(value / norm for value in values)


def _hashed_semantic_embedding(text: str, *, dim: int = 24) -> tuple[float, ...]:
    tokens = _semantic_tokens(text)
    if not tokens:
        return tuple(0.0 for _ in range(dim))
    vector = [0.0 for _ in range(dim)]
    for token in tokens:
        token_weight = max(len(token), 1)
        seed = sum((index + 1) * ord(char) for index, char in enumerate(token))
        for bucket in range(dim):
            phase = (seed + (bucket + 1) * 131) % 1543
            vector[bucket] += math.sin(phase * 0.013) / token_weight
            vector[bucket] += math.cos(phase * 0.007) / (token_weight * 1.7)
    return _normalize_vector(vector)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


RARE_HEAVY_ANCHOR_ORDER = ("task", "support", "repair", "exploration", "directive")


def _mean_feature_value(substrates: Sequence[SubstrateSnapshot], *, name: str) -> float:
    values: list[float] = []
    for substrate in substrates:
        for feature in substrate.feature_surface:
            if feature.name == name and feature.values:
                values.append(float(sum(feature.values) / len(feature.values)))
    if not values:
        return 0.0
    return sum(values) / len(values)


def _flatten_substrate_batches(
    substrate_steps_per_trace: tuple[tuple[SubstrateSnapshot, ...], ...],
) -> tuple[SubstrateSnapshot, ...]:
    return tuple(snapshot for batch in substrate_steps_per_trace for snapshot in batch)


def _mean_sequence_length(substrates: Sequence[SubstrateSnapshot]) -> float:
    if not substrates:
        return 0.0
    return sum(
        max(len(substrate.residual_sequence), 1)
        for substrate in substrates
    ) / len(substrates)


def _mean_residual_magnitude(substrates: Sequence[SubstrateSnapshot]) -> float:
    values: list[float] = []
    for substrate in substrates:
        if substrate.residual_activations:
            values.extend(
                abs(value)
                for activation in substrate.residual_activations
                for value in activation.activation
            )
        else:
            values.extend(
                abs(value)
                for feature in substrate.feature_surface
                for value in feature.values
            )
    if not values:
        return 0.0
    return _clamp_unit(sum(values) / len(values))


def _derive_anchor_bias(substrates: Sequence[SubstrateSnapshot]) -> tuple[float, ...]:
    means = tuple(
        _mean_feature_value(substrates, name=f"semantic_{anchor}_pull")
        for anchor in RARE_HEAVY_ANCHOR_ORDER
    )
    if not any(abs(value) > 1e-8 for value in means):
        return tuple(0.0 for _ in RARE_HEAVY_ANCHOR_ORDER)
    center = sum(means) / len(means)
    return tuple(
        max(-0.2, min(0.2, (value - center) * 0.5))
        for value in means
    )


def _normalize_semantic_weights(*, text_weight: float, residual_weight: float) -> tuple[float, float]:
    text = max(0.05, min(0.95, text_weight))
    residual = max(0.05, min(0.95, residual_weight))
    total = text + residual
    if total <= 1e-8:
        return (0.5, 0.5)
    return (text / total, residual / total)


def _mean_abs_delta(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def _clamp_delta_vector(values: Sequence[float], *, limit: float = 0.20) -> tuple[float, ...]:
    return tuple(max(-limit, min(limit, float(value))) for value in values)


def _adapter_parameter_count(adapter_layers: Sequence[SubstrateDeltaAdapterLayer]) -> int:
    return sum(len(layer.delta_vector) for layer in adapter_layers)


def _build_compatibility_fingerprint(
    *,
    model_id: str,
    runtime_origin: str,
    hidden_size: int,
    layer_indices: Sequence[int],
    training_mode: str,
) -> str:
    raw = "|".join(
        (
            model_id,
            runtime_origin,
            str(hidden_size),
            ",".join(str(index) for index in layer_indices),
            training_mode,
        )
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{training_mode}:{digest}"


def _checkpoint_with_adapter_payload(
    checkpoint: SubstrateRareHeavyCheckpoint,
    *,
    training_mode: str,
    compatibility_fingerprint: str,
    adapter_scale: float,
    adapter_training_loss: float,
    adapter_layers: Sequence[SubstrateDeltaAdapterLayer],
    description: str,
) -> SubstrateRareHeavyCheckpoint:
    layer_tuple = tuple(adapter_layers)
    return replace(
        checkpoint,
        checkpoint_version=2,
        training_mode=training_mode,
        compatibility_fingerprint=compatibility_fingerprint,
        adapter_scale=max(0.0, adapter_scale),
        adapter_parameter_count=_adapter_parameter_count(layer_tuple),
        adapter_training_loss=max(0.0, adapter_training_loss),
        adapter_layers=layer_tuple,
        description=description,
    )


def _derive_rare_heavy_checkpoint(
    *,
    checkpoint_id: str,
    model_id: str,
    runtime_origin: str,
    current_control_scale: float,
    default_text_weight: float,
    default_residual_weight: float,
    previous_update_count: int,
    substrate_steps_per_trace: tuple[tuple[SubstrateSnapshot, ...], ...],
) -> SubstrateRareHeavyCheckpoint:
    flattened = _flatten_substrate_batches(substrate_steps_per_trace)
    if not flattened:
        text_weight, residual_weight = _normalize_semantic_weights(
            text_weight=default_text_weight,
            residual_weight=default_residual_weight,
        )
        return SubstrateRareHeavyCheckpoint(
            checkpoint_id=checkpoint_id,
            model_id=model_id,
            runtime_origin=runtime_origin,
            control_scale=max(0.04, min(0.30, current_control_scale)),
            semantic_text_weight=text_weight,
            semantic_residual_weight=residual_weight,
            semantic_anchor_bias=tuple(0.0 for _ in RARE_HEAVY_ANCHOR_ORDER),
            update_count=previous_update_count,
            source_batch_count=0,
            mean_sequence_length=0.0,
            mean_residual_magnitude=0.0,
            description=(
                f"Rare-heavy substrate checkpoint for {model_id} exported without offline substrate batches; "
                f"state kept unchanged."
            ),
        )
    mean_seq_len = _mean_sequence_length(flattened)
    mean_residual = _mean_residual_magnitude(flattened)
    hook_coverage = _mean_feature_value(flattened, name="hook_layer_coverage")
    residual_signal = max(
        mean_residual,
        _mean_feature_value(flattened, name="residual_mean_abs"),
    )
    sequence_signal = _clamp_unit(mean_seq_len / 10.0)
    residual_target = (
        default_residual_weight * 0.60
        + residual_signal * 0.22
        + sequence_signal * 0.10
        + hook_coverage * 0.08
    )
    text_weight, residual_weight = _normalize_semantic_weights(
        text_weight=1.0 - residual_target,
        residual_weight=residual_target,
    )
    target_control_scale = max(
        0.04,
        min(
            0.30,
            current_control_scale * 0.60
            + (0.08 + residual_signal * 0.12 + hook_coverage * 0.05 + sequence_signal * 0.03) * 0.40,
        ),
    )
    anchor_bias = _derive_anchor_bias(flattened)
    return SubstrateRareHeavyCheckpoint(
        checkpoint_id=checkpoint_id,
        model_id=model_id,
        runtime_origin=runtime_origin,
        control_scale=target_control_scale,
        semantic_text_weight=text_weight,
        semantic_residual_weight=residual_weight,
        semantic_anchor_bias=anchor_bias,
        update_count=previous_update_count + 1,
        source_batch_count=len(substrate_steps_per_trace),
        mean_sequence_length=mean_seq_len,
        mean_residual_magnitude=mean_residual,
        description=(
            f"Rare-heavy substrate checkpoint for {model_id}: batches={len(substrate_steps_per_trace)}, "
            f"seq_len={mean_seq_len:.2f}, residual={mean_residual:.3f}, hook_coverage={hook_coverage:.3f}, "
            f"control_scale={target_control_scale:.3f}."
        ),
    )


SEMANTIC_ANCHOR_BANK: dict[str, tuple[str, ...]] = {
    "task": (
        "Break the work into concrete steps, choose one path, and justify the tradeoff.",
        "I need a decision, an execution order, and the clearest next action.",
        "把任务拆开 选定路线 给我明确顺序和取舍理由",
        "直接做决策 然后安排执行顺序 不要停在泛泛安抚",
    ),
    "support": (
        "Stay with me first and help me feel safer before solving anything.",
        "Offer warmth, reassurance, and a gentle next step without pressure.",
        "Please be supportive first, go gently, and do not rush me into a solution.",
        "先陪我稳住 情绪支持 温和一点 给我一个不逼迫的下一步",
    ),
    "repair": (
        "Repair trust after friction, de-escalate the interaction, and reduce rupture before widening scope.",
        "We need to stabilize the relationship frame after conflict or misunderstanding.",
        "先修复关系和信任 降低冲突张力 稳住局面 再继续推进",
    ),
    "exploration": (
        "Explore the space gradually, narrow options, and reason step by step.",
        "Guide the conversation through uncertainty without rushing to closure.",
        "一起探索 逐步收窄 先看可能性 再决定方向",
    ),
    "directive": (
        "Skip the nuance, cut through ambiguity, and tell me the one decision right now.",
        "Do not cushion this, make the call, and state the top priority immediately.",
        "别讲铺垫 直接拍板 告诉我唯一最优先项和放弃理由",
        "不要温和过渡 现在就下判断 直接给结论",
    ),
}


def _anchor_profile_bank(*, dim: int) -> dict[str, tuple[float, ...]]:
    profiles: dict[str, tuple[float, ...]] = {}
    for name, texts in SEMANTIC_ANCHOR_BANK.items():
        accum = [0.0 for _ in range(dim)]
        for text in texts:
            embedding = _hashed_semantic_embedding(text, dim=dim)
            for index, value in enumerate(embedding):
                accum[index] += value
        profiles[name] = _normalize_vector(tuple(value / max(len(texts), 1) for value in accum))
    return profiles


def resolve_substrate_fallback_mode(
    *,
    fallback_mode: SubstrateFallbackMode | str | None = None,
    fallback_to_builtin: bool | None = None,
) -> SubstrateFallbackMode:
    if fallback_mode is not None:
        return SubstrateFallbackMode(fallback_mode)
    if fallback_to_builtin is None:
        return SubstrateFallbackMode.ALLOW_BUILTIN
    return SubstrateFallbackMode.ALLOW_BUILTIN if fallback_to_builtin else SubstrateFallbackMode.DENY


def resolve_local_runtime_mode(
    *,
    runtime_mode: LocalSubstrateRuntimeMode | str | None = None,
    local_files_only: bool = False,
    fallback_mode: SubstrateFallbackMode | str | None = None,
    fallback_to_builtin: bool | None = None,
) -> LocalSubstrateRuntimeMode | None:
    if runtime_mode is not None:
        return LocalSubstrateRuntimeMode(runtime_mode)
    resolved_fallback = resolve_substrate_fallback_mode(
        fallback_mode=fallback_mode,
        fallback_to_builtin=fallback_to_builtin,
    )
    if local_files_only and resolved_fallback is SubstrateFallbackMode.DENY:
        return LocalSubstrateRuntimeMode.STRICT_LOCAL
    if local_files_only and resolved_fallback is SubstrateFallbackMode.ALLOW_BUILTIN:
        return LocalSubstrateRuntimeMode.PREFER_LOCAL
    return None


