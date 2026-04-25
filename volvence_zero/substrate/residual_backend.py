from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import importlib
import math
from typing import Callable, Iterable, Sequence

from volvence_zero.substrate.adapter import (
    FeatureSignal,
    ResidualActivation,
    ResidualSequenceStep,
    ResidualStreamSubstrateAdapter,
    SubstrateSnapshot,
)

if False:
    from volvence_zero.agent.response import GenerationConstraints


@dataclass(frozen=True)
class TraceStep:
    step: int
    token: str
    feature_surface: tuple[FeatureSignal, ...]
    residual_activations: tuple[ResidualActivation, ...]


@dataclass(frozen=True)
class TrainingTrace:
    trace_id: str
    source_text: str
    steps: tuple[TraceStep, ...]


@dataclass(frozen=True)
class ResidualControlApplication:
    applied_snapshot: SubstrateSnapshot
    downstream_effect: tuple[float, ...]
    control_energy: float
    backend_name: str
    description: str


@dataclass(frozen=True)
class OpenWeightRuntimeCapture:
    token_logits: tuple[float, ...]
    feature_surface: tuple[FeatureSignal, ...]
    residual_activations: tuple[ResidualActivation, ...]
    residual_sequence: tuple[ResidualSequenceStep, ...]
    description: str


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_count: int
    capture: OpenWeightRuntimeCapture | None
    description: str


@dataclass(frozen=True)
class HookLayerCalibrationCase:
    layer_indices: tuple[int, ...]
    hook_layer_coverage: float
    residual_sequence_length: int
    semantic_separation: float
    signal_quality: float
    runtime_origin: str
    description: str


@dataclass(frozen=True)
class HookLayerCalibrationReport:
    model_id: str
    source_text: str
    cases: tuple[HookLayerCalibrationCase, ...]
    recommended_layers: tuple[int, ...]
    description: str


@dataclass(frozen=True)
class LocalModelCompatibilityReport:
    model_id: str
    local_tokenizer_available: bool
    local_model_available: bool
    strict_local_runtime_available: bool
    error_type: str | None
    error_message: str
    description: str


@dataclass(frozen=True)
class SubstrateDeltaAdapterLayer:
    layer_index: int
    delta_vector: tuple[float, ...]
    mean_abs_delta: float
    description: str


@dataclass(frozen=True)
class SubstrateRareHeavyCheckpoint:
    checkpoint_id: str
    model_id: str
    runtime_origin: str
    control_scale: float
    semantic_text_weight: float
    semantic_residual_weight: float
    semantic_anchor_bias: tuple[float, ...]
    update_count: int
    source_batch_count: int
    mean_sequence_length: float
    mean_residual_magnitude: float
    description: str
    checkpoint_version: int = 1
    training_mode: str = "bounded-state-v1"
    compatibility_fingerprint: str = ""
    adapter_scale: float = 0.0
    adapter_parameter_count: int = 0
    adapter_training_loss: float = 0.0
    adapter_layers: tuple[SubstrateDeltaAdapterLayer, ...] = ()


@dataclass(frozen=True)
class SubstrateOnlineFastCheckpoint:
    checkpoint_id: str
    model_id: str
    runtime_origin: str
    delta_scale: float
    update_count: int
    source_wave_id: str
    source_turn_index: int
    gate: str
    optimizer_state_norm: float
    parameter_change_rate: float
    description: str
    checkpoint_version: int = 1
    training_mode: str = "online-fast-delta-v1"
    compatibility_fingerprint: str = ""
    adapter_parameter_count: int = 0
    adapter_layers: tuple[SubstrateDeltaAdapterLayer, ...] = ()
    fast_state_hash: str = ""
    source_fast_state_hash: str = ""
    fast_memory_signal: tuple[float, ...] = ()
    optimizer_state_description: str = ""


class SubstrateFallbackMode(str, Enum):
    ALLOW_BUILTIN = "allow-builtin"
    DENY = "deny"


class LocalSubstrateRuntimeMode(str, Enum):
    STRICT_LOCAL = "strict-local"
    PREFER_LOCAL = "prefer-local"
    BUILTIN_ONLY = "builtin-only"


@dataclass
class HashingWhitespaceTokenizer:
    """Minimal local tokenizer for bundled tiny transformers runtimes."""

    vocab_size: int = 256
    _token_to_id: dict[str, int] = field(default_factory=lambda: {"<empty>": 1}, init=False, repr=False)
    _id_to_token: dict[int, str] = field(default_factory=lambda: {1: "<empty>"}, init=False, repr=False)

    def __call__(self, text: str, *, return_tensors: str, truncation: bool, max_length: int):
        del truncation
        if return_tensors != "pt":
            raise ValueError("HashingWhitespaceTokenizer expects return_tensors='pt'.")
        torch = importlib.import_module("torch")
        tokens = tuple(part for part in text.split() if part.strip())[:max_length] or ("<empty>",)
        token_ids = [self._resolve_token_id(token) for token in tokens]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def convert_ids_to_tokens(self, token_ids: tuple[int, ...]) -> tuple[str, ...]:
        return tuple(self._id_to_token.get(token_id, f"<tok:{token_id}>") for token_id in token_ids)

    def _resolve_token_id(self, token: str) -> int:
        existing = self._token_to_id.get(token)
        if existing is not None:
            return existing
        next_id = (len(self._token_to_id) % max(self.vocab_size - 1, 1)) + 1
        self._token_to_id[token] = next_id
        self._id_to_token[next_id] = token
        return next_id


class OpenWeightResidualRuntime(ABC):
    """Hook-ready runtime contract for frozen open-weight residual access."""

    model_id: str
    is_frozen: bool
    runtime_origin: str = "unknown"
    supports_live_substrate_mutation: bool = False
    supports_offline_substrate_training: bool = False

    @abstractmethod
    def capture(self, *, source_text: str) -> OpenWeightRuntimeCapture:
        """Capture a frozen-model residual snapshot for the given source text."""

    @abstractmethod
    def apply_control(
        self,
        *,
        source_text: str,
        substrate_snapshot: SubstrateSnapshot,
        applied_control: tuple[float, ...],
        track_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> ResidualControlApplication:
        """Apply bounded residual intervention through the runtime."""

    def generate(
        self,
        *,
        prompt: str,
        system_context: str = "",
        chat_messages: tuple[tuple[str, str], ...] = (),
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        control_parameters: tuple[float, ...] = (),
        control_scale: float = 0.0,
        generation_constraints: "GenerationConstraints | None" = None,
    ) -> GenerationResult:
        """Generate text using the underlying model.

        Subclasses that hold a real model override this to run
        autoregressive decoding.  The default implementation returns
        a placeholder so synthetic runtimes remain functional.
        """
        del generation_constraints
        return GenerationResult(
            text=f"[generation not supported by {self.model_id}]",
            token_count=0,
            capture=None,
            description=f"{self.model_id} does not support generation",
        )

    @property
    def fallback_active(self) -> bool:
        return self.runtime_origin in {"builtin-fallback", "synthetic-open-weight"}

    @property
    def capture_source(self) -> str:
        return "real" if not self.fallback_active else "fallback"

    @property
    def experimental_live_mutation_enabled(self) -> bool:
        return self.supports_live_substrate_mutation

    @property
    def live_mutation_mode(self) -> str:
        return "experimental-live" if self.experimental_live_mutation_enabled else "frozen-review-only"

    def require_live_substrate_mutation(self, *, operation: str) -> None:
        if not self.supports_live_substrate_mutation:
            raise RuntimeError(
                f"{type(self).__name__} blocks {operation} because the default frozen-substrate doctrine "
                "does not allow live substrate mutation. Keep the runtime on capture/control/generate only, "
                "or opt into explicit experimental live mutation."
            )

    def require_offline_substrate_training(self, *, operation: str) -> None:
        if not self.supports_offline_substrate_training:
            raise RuntimeError(
                f"{type(self).__name__} blocks {operation} because this runtime is not marked as an "
                "offline substrate-artifact owner. Clone the runtime for rare-heavy/offline training first."
            )

    def require_substrate_artifact_import(self, *, operation: str) -> None:
        if self.supports_live_substrate_mutation or self.supports_offline_substrate_training:
            return
        raise RuntimeError(
            f"{type(self).__name__} blocks {operation} because importing substrate artifacts into the live "
            "runtime is disabled under the frozen-substrate doctrine. Use an offline clone or explicit "
            "experimental live mutation mode."
        )

    def export_rare_heavy_state(self, *, checkpoint_id: str | None = None) -> SubstrateRareHeavyCheckpoint:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement rare-heavy substrate export."
        )

    def import_rare_heavy_state(self, checkpoint: SubstrateRareHeavyCheckpoint) -> tuple[str, ...]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement rare-heavy substrate import."
        )

    def restore_rare_heavy_state(self, checkpoint: SubstrateRareHeavyCheckpoint) -> tuple[str, ...]:
        return self.import_rare_heavy_state(checkpoint)

    def train_rare_heavy(
        self,
        *,
        traces: tuple[TrainingTrace, ...] = (),
        substrate_steps_per_trace: tuple[tuple[SubstrateSnapshot, ...], ...],
        checkpoint_id: str | None = None,
    ) -> SubstrateRareHeavyCheckpoint:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement rare-heavy substrate training."
        )

    def clone_for_rare_heavy(self) -> "OpenWeightResidualRuntime":
        raise NotImplementedError(
            f"{type(self).__name__} does not implement rare-heavy runtime cloning."
        )

    def export_online_fast_state(
        self,
        *,
        checkpoint_id: str | None = None,
    ) -> SubstrateOnlineFastCheckpoint:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement online-fast substrate export."
        )

    def apply_online_fast_state(self, checkpoint: SubstrateOnlineFastCheckpoint) -> tuple[str, ...]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement online-fast substrate apply."
        )

    def restore_online_fast_state(self, checkpoint: SubstrateOnlineFastCheckpoint) -> tuple[str, ...]:
        return self.apply_online_fast_state(checkpoint)


class ResidualInterventionBackend(ABC):
    """Bounded owner-side backend for residual control application."""

    name: str

    @abstractmethod
    def apply_control(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        applied_control: tuple[float, ...],
        track_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> ResidualControlApplication:
        """Apply bounded residual control and return the resulting effect."""


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


class TraceResidualInterventionBackend(ResidualInterventionBackend):
    """Executable trace-backed backend approximating residual intervention."""

    name = "trace-residual-backend"

    def apply_control(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        applied_control: tuple[float, ...],
        track_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> ResidualControlApplication:
        sequence = substrate_snapshot.residual_sequence
        if not sequence:
            sequence = (
                ResidualSequenceStep(
                    step=max((activation.step for activation in substrate_snapshot.residual_activations), default=0),
                    token="<runtime-step>",
                    feature_surface=substrate_snapshot.feature_surface,
                    residual_activations=substrate_snapshot.residual_activations,
                    description="Synthesized single-step residual sequence.",
                ),
            )
        latest_before = _summarize_activations(sequence[-1].residual_activations)
        modified_steps: list[ResidualSequenceStep] = []
        for step in sequence:
            modified_activations: list[ResidualActivation] = []
            for activation in step.residual_activations:
                activation_values = tuple(
                    _clamp_unit(
                        value
                        + value * applied_control[min(index, len(applied_control) - 1)] * 0.35
                        + track_scale[min(index, len(track_scale) - 1)] * 0.05
                    )
                    for index, value in enumerate(activation.activation)
                )
                modified_activations.append(
                    ResidualActivation(
                        layer_index=activation.layer_index,
                        activation=activation_values,
                        step=activation.step,
                    )
                )
            modified_steps.append(
                ResidualSequenceStep(
                    step=step.step,
                    token=step.token,
                    feature_surface=step.feature_surface,
                    residual_activations=tuple(modified_activations),
                    description=f"{step.description} residual_control_applied",
                )
            )
        latest_after = _summarize_activations(modified_steps[-1].residual_activations)
        downstream_effect = tuple(
            _clamp_signed(
                (latest_after[index] - latest_before[index]) * 1.5
                + applied_control[index] * track_scale[index] * 0.25
            )
            for index in range(3)
        )
        control_energy = sum(abs(value) for value in applied_control) / max(len(applied_control), 1)
        applied_snapshot = SubstrateSnapshot(
            model_id=substrate_snapshot.model_id,
            is_frozen=substrate_snapshot.is_frozen,
            surface_kind=substrate_snapshot.surface_kind,
            token_logits=substrate_snapshot.token_logits,
            feature_surface=substrate_snapshot.feature_surface,
            residual_activations=modified_steps[-1].residual_activations,
            residual_sequence=tuple(modified_steps),
            unavailable_fields=substrate_snapshot.unavailable_fields,
            description=(
                f"{substrate_snapshot.description} Applied residual control "
                f"{tuple(round(value, 3) for value in applied_control)}."
            ),
        )
        return ResidualControlApplication(
            applied_snapshot=applied_snapshot,
            downstream_effect=downstream_effect,
            control_energy=control_energy,
            backend_name=self.name,
            description=(
                f"{self.name} energy={control_energy:.3f} "
                f"effect={tuple(round(value, 3) for value in downstream_effect)}."
            ),
        )


class NoOpResidualInterventionBackend(ResidualInterventionBackend):
    """Baseline backend that preserves the frozen substrate unchanged."""

    name = "noop-residual-backend"

    def apply_control(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        applied_control: tuple[float, ...],
        track_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> ResidualControlApplication:
        del applied_control
        del track_scale
        return ResidualControlApplication(
            applied_snapshot=substrate_snapshot,
            downstream_effect=(0.0, 0.0, 0.0),
            control_energy=0.0,
            backend_name=self.name,
            description="noop-residual-backend preserved the frozen substrate unchanged.",
        )


class OpenWeightResidualInterventionBackend(ResidualInterventionBackend):
    """Backend that delegates intervention to an open-weight runtime."""

    def __init__(self, *, runtime: OpenWeightResidualRuntime, source_text: str) -> None:
        self._runtime = runtime
        self._source_text = source_text
        self.name = f"open-weight:{runtime.model_id}"

    def apply_control(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        applied_control: tuple[float, ...],
        track_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> ResidualControlApplication:
        result = self._runtime.apply_control(
            source_text=self._source_text,
            substrate_snapshot=substrate_snapshot,
            applied_control=applied_control,
            track_scale=track_scale,
        )
        return ResidualControlApplication(
            applied_snapshot=result.applied_snapshot,
            downstream_effect=result.downstream_effect,
            control_energy=result.control_energy,
            backend_name=self.name,
            description=result.description,
        )


def apply_residual_control(
    *,
    substrate_snapshot: SubstrateSnapshot,
    applied_control: tuple[float, ...],
    track_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> ResidualControlApplication:
    return TraceResidualInterventionBackend().apply_control(
        substrate_snapshot=substrate_snapshot,
        applied_control=applied_control,
        track_scale=track_scale,
    )


class SyntheticOpenWeightResidualRuntime(OpenWeightResidualRuntime):
    """Hook-shaped frozen runtime backed by the existing residual simulator."""

    def __init__(
        self,
        *,
        model_id: str = "synthetic-open-weight-runtime",
        allow_live_substrate_mutation: bool = False,
        allow_offline_substrate_training: bool = False,
    ) -> None:
        self.model_id = model_id
        self.is_frozen = True
        self.runtime_origin = "synthetic-open-weight"
        self.supports_live_substrate_mutation = allow_live_substrate_mutation
        self.supports_offline_substrate_training = allow_offline_substrate_training
        self._rare_heavy_control_scale = 0.12
        self._rare_heavy_semantic_text_weight = 0.85
        self._rare_heavy_semantic_residual_weight = 0.15
        self._rare_heavy_anchor_bias = tuple(0.0 for _ in RARE_HEAVY_ANCHOR_ORDER)
        self._rare_heavy_update_count = 0
        self._rare_heavy_adapter_scale = 0.0
        self._rare_heavy_adapter_layers: dict[int, tuple[float, ...]] = {}
        self._rare_heavy_activation_width = 0
        self._rare_heavy_layer_indices: tuple[int, ...] = ()
        self._online_fast_delta_scale = 0.0
        self._online_fast_update_count = 0
        self._online_fast_optimizer_state_norm = 0.0
        self._online_fast_parameter_change_rate = 0.0
        self._online_fast_adapter_layers: dict[int, tuple[float, ...]] = {}
        self._online_fast_state_hash = ""
        self._online_fast_source_state_hash = ""
        self._online_fast_signal: tuple[float, ...] = ()
        self._online_fast_optimizer_state_description = ""

    def capture(self, *, source_text: str) -> OpenWeightRuntimeCapture:
        trace = build_training_trace(trace_id=f"{self.model_id}:capture", source_text=source_text)
        self._remember_trace_shape(trace)
        latest_step = trace.steps[-1]
        adapted_residuals = self._apply_adapter_to_residuals(latest_step.residual_activations)
        feature_surface = latest_step.feature_surface + (
            FeatureSignal(
                name="semantic_text_weight",
                values=(self._rare_heavy_semantic_text_weight,),
                source="synthetic-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_residual_weight",
                values=(self._rare_heavy_semantic_residual_weight,),
                source="synthetic-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_rare_heavy_update_count",
                values=(_clamp_unit(self._rare_heavy_update_count / 10.0),),
                source="synthetic-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_delta_parameter_count",
                values=(_clamp_unit(_adapter_parameter_count(self._export_adapter_layers()) / 64.0),),
                source="synthetic-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_online_fast_update_count",
                values=(_clamp_unit(self._online_fast_update_count / 10.0),),
                source="synthetic-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_online_fast_delta_parameter_count",
                values=(_clamp_unit(_adapter_parameter_count(self._export_online_fast_layers()) / 64.0),),
                source="synthetic-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_online_fast_parameter_change_rate",
                values=(_clamp_unit(self._online_fast_parameter_change_rate),),
                source="synthetic-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_online_fast_experimental_mode",
                values=(1.0 if self.experimental_live_mutation_enabled else 0.0,),
                source="synthetic-open-weight-semantic",
            ),
        )
        residual_sequence = tuple(
            ResidualSequenceStep(
                step=step.step,
                token=step.token,
                feature_surface=step.feature_surface,
                residual_activations=self._apply_adapter_to_residuals(step.residual_activations),
                description=f"Synthetic hook token '{step.token}' at step {step.step}.",
            )
            for step in trace.steps
        )
        return OpenWeightRuntimeCapture(
            token_logits=tuple(
                min(sum(feature.values) / max(len(feature.values), 1), 1.0)
                for feature in feature_surface
            ),
            feature_surface=feature_surface,
            residual_activations=adapted_residuals,
            residual_sequence=residual_sequence,
            description=(
                f"Synthetic frozen open-weight capture for len={len(source_text)} "
                f"rare_heavy_updates={self._rare_heavy_update_count} "
                f"adapter_params={_adapter_parameter_count(self._export_adapter_layers())} "
                f"online_fast_updates={self._online_fast_update_count} "
                f"online_fast_params={_adapter_parameter_count(self._export_online_fast_layers())} "
                f"live_mode={self.live_mutation_mode}."
            ),
        )

    def apply_control(
        self,
        *,
        source_text: str,
        substrate_snapshot: SubstrateSnapshot,
        applied_control: tuple[float, ...],
        track_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> ResidualControlApplication:
        del source_text
        scale_ratio = max(0.25, min(2.0, self._rare_heavy_control_scale / 0.12))
        scaled_control = tuple(value * scale_ratio for value in applied_control)
        result = TraceResidualInterventionBackend().apply_control(
            substrate_snapshot=substrate_snapshot,
            applied_control=scaled_control,
            track_scale=track_scale,
        )
        return ResidualControlApplication(
            applied_snapshot=result.applied_snapshot,
            downstream_effect=result.downstream_effect,
            control_energy=result.control_energy,
            backend_name=f"synthetic-open-weight:{self.model_id}",
            description=(
                f"Synthetic open-weight runtime delegated residual intervention. "
                f"{result.description}"
            ),
        )

    def export_rare_heavy_state(self, *, checkpoint_id: str | None = None) -> SubstrateRareHeavyCheckpoint:
        adapter_layers = self._export_adapter_layers()
        hidden_size = self._rare_heavy_activation_width
        layer_indices = self._rare_heavy_layer_indices
        training_mode = "adapter-delta-v2" if adapter_layers else "bounded-state-v1"
        return SubstrateRareHeavyCheckpoint(
            checkpoint_id=checkpoint_id or f"{self.model_id}:rare-heavy",
            model_id=self.model_id,
            runtime_origin=self.runtime_origin,
            control_scale=self._rare_heavy_control_scale,
            semantic_text_weight=self._rare_heavy_semantic_text_weight,
            semantic_residual_weight=self._rare_heavy_semantic_residual_weight,
            semantic_anchor_bias=self._rare_heavy_anchor_bias,
            update_count=self._rare_heavy_update_count,
            source_batch_count=0,
            mean_sequence_length=0.0,
            mean_residual_magnitude=0.0,
            description=(
                f"Synthetic rare-heavy checkpoint for {self.model_id} "
                f"updates={self._rare_heavy_update_count}."
            ),
            checkpoint_version=2 if adapter_layers else 1,
            training_mode=training_mode,
            compatibility_fingerprint=(
                _build_compatibility_fingerprint(
                    model_id=self.model_id,
                    runtime_origin=self.runtime_origin,
                    hidden_size=hidden_size,
                    layer_indices=layer_indices,
                    training_mode=training_mode,
                )
                if hidden_size > 0 and layer_indices
                else ""
            ),
            adapter_scale=self._rare_heavy_adapter_scale,
            adapter_parameter_count=_adapter_parameter_count(adapter_layers),
            adapter_training_loss=0.0,
            adapter_layers=adapter_layers,
        )

    def import_rare_heavy_state(self, checkpoint: SubstrateRareHeavyCheckpoint) -> tuple[str, ...]:
        self.require_substrate_artifact_import(operation="import_rare_heavy_state()")
        if checkpoint.model_id != self.model_id:
            raise ValueError(
                f"Synthetic runtime {self.model_id!r} cannot import checkpoint for {checkpoint.model_id!r}."
            )
        text_weight, residual_weight = _normalize_semantic_weights(
            text_weight=checkpoint.semantic_text_weight,
            residual_weight=checkpoint.semantic_residual_weight,
        )
        self._rare_heavy_control_scale = max(0.04, min(0.30, checkpoint.control_scale))
        self._rare_heavy_semantic_text_weight = text_weight
        self._rare_heavy_semantic_residual_weight = residual_weight
        self._rare_heavy_anchor_bias = tuple(
            max(-0.2, min(0.2, value))
            for value in checkpoint.semantic_anchor_bias[: len(RARE_HEAVY_ANCHOR_ORDER)]
        )
        if len(self._rare_heavy_anchor_bias) < len(RARE_HEAVY_ANCHOR_ORDER):
            self._rare_heavy_anchor_bias = self._rare_heavy_anchor_bias + tuple(
                0.0 for _ in range(len(RARE_HEAVY_ANCHOR_ORDER) - len(self._rare_heavy_anchor_bias))
            )
        self._rare_heavy_update_count = max(0, checkpoint.update_count)
        self._rare_heavy_adapter_scale = max(0.0, checkpoint.adapter_scale)
        self._rare_heavy_adapter_layers = {
            layer.layer_index: _clamp_delta_vector(layer.delta_vector)
            for layer in checkpoint.adapter_layers
        }
        if checkpoint.adapter_layers:
            self._rare_heavy_activation_width = len(checkpoint.adapter_layers[0].delta_vector)
            self._rare_heavy_layer_indices = tuple(layer.layer_index for layer in checkpoint.adapter_layers)
        return ("rare-heavy:substrate-import",)

    def restore_rare_heavy_state(self, checkpoint: SubstrateRareHeavyCheckpoint) -> tuple[str, ...]:
        self.import_rare_heavy_state(checkpoint)
        return ("rare-heavy:substrate-rollback",)

    def train_rare_heavy(
        self,
        *,
        traces: tuple[TrainingTrace, ...] = (),
        substrate_steps_per_trace: tuple[tuple[SubstrateSnapshot, ...], ...],
        checkpoint_id: str | None = None,
    ) -> SubstrateRareHeavyCheckpoint:
        self.require_offline_substrate_training(operation="train_rare_heavy()")
        checkpoint = _derive_rare_heavy_checkpoint(
            checkpoint_id=checkpoint_id or f"{self.model_id}:rare-heavy-trained",
            model_id=self.model_id,
            runtime_origin=self.runtime_origin,
            current_control_scale=self._rare_heavy_control_scale,
            default_text_weight=self._rare_heavy_semantic_text_weight,
            default_residual_weight=self._rare_heavy_semantic_residual_weight,
            previous_update_count=self._rare_heavy_update_count,
            substrate_steps_per_trace=substrate_steps_per_trace,
        )
        adapter_layers, training_loss = self._train_adapter_deltas(
            traces=traces,
            substrate_steps_per_trace=substrate_steps_per_trace,
        )
        if not adapter_layers:
            return checkpoint
        return _checkpoint_with_adapter_payload(
            checkpoint,
            training_mode="adapter-delta-v2",
            compatibility_fingerprint=_build_compatibility_fingerprint(
                model_id=self.model_id,
                runtime_origin=self.runtime_origin,
                hidden_size=self._rare_heavy_activation_width,
                layer_indices=self._rare_heavy_layer_indices,
                training_mode="adapter-delta-v2",
            ),
            adapter_scale=1.0,
            adapter_training_loss=training_loss,
            adapter_layers=adapter_layers,
            description=(
                f"{checkpoint.description} Synthetic adapter-delta payload "
                f"layers={len(adapter_layers)} loss={training_loss:.4f}."
            ),
        )

    def clone_for_rare_heavy(self) -> "OpenWeightResidualRuntime":
        cloned = SyntheticOpenWeightResidualRuntime(
            model_id=self.model_id,
            allow_offline_substrate_training=True,
        )
        cloned.import_rare_heavy_state(self.export_rare_heavy_state())
        return cloned

    def export_online_fast_state(
        self,
        *,
        checkpoint_id: str | None = None,
    ) -> SubstrateOnlineFastCheckpoint:
        self.require_live_substrate_mutation(operation="export_online_fast_state()")
        adapter_layers = self._export_online_fast_layers()
        return SubstrateOnlineFastCheckpoint(
            checkpoint_id=checkpoint_id or f"{self.model_id}:online-fast",
            model_id=self.model_id,
            runtime_origin=self.runtime_origin,
            delta_scale=self._online_fast_delta_scale,
            update_count=self._online_fast_update_count,
            source_wave_id="runtime",
            source_turn_index=self._online_fast_update_count,
            gate="online",
            optimizer_state_norm=self._online_fast_optimizer_state_norm,
            parameter_change_rate=self._online_fast_parameter_change_rate,
            description=(
                f"Synthetic online-fast checkpoint for {self.model_id} "
                f"updates={self._online_fast_update_count}."
            ),
            compatibility_fingerprint=(
                _build_compatibility_fingerprint(
                    model_id=self.model_id,
                    runtime_origin=self.runtime_origin,
                    hidden_size=self._rare_heavy_activation_width,
                    layer_indices=self._rare_heavy_layer_indices,
                    training_mode="online-fast-delta-v1",
                )
                if self._rare_heavy_activation_width > 0 and self._rare_heavy_layer_indices
                else ""
            ),
            adapter_parameter_count=_adapter_parameter_count(adapter_layers),
            adapter_layers=adapter_layers,
            fast_state_hash=self._online_fast_state_hash,
            source_fast_state_hash=self._online_fast_source_state_hash,
            fast_memory_signal=self._online_fast_signal,
            optimizer_state_description=self._online_fast_optimizer_state_description,
        )

    def apply_online_fast_state(self, checkpoint: SubstrateOnlineFastCheckpoint) -> tuple[str, ...]:
        self.require_live_substrate_mutation(operation="apply_online_fast_state()")
        if checkpoint.model_id != self.model_id:
            raise ValueError(
                f"Synthetic runtime {self.model_id!r} cannot import online-fast checkpoint for "
                f"{checkpoint.model_id!r}."
            )
        self._online_fast_delta_scale = max(0.0, min(0.18, checkpoint.delta_scale))
        self._online_fast_update_count = max(0, checkpoint.update_count)
        self._online_fast_optimizer_state_norm = _clamp_unit(checkpoint.optimizer_state_norm)
        self._online_fast_parameter_change_rate = _clamp_unit(checkpoint.parameter_change_rate)
        self._online_fast_state_hash = checkpoint.fast_state_hash
        self._online_fast_source_state_hash = checkpoint.source_fast_state_hash
        self._online_fast_signal = checkpoint.fast_memory_signal
        self._online_fast_optimizer_state_description = checkpoint.optimizer_state_description
        self._online_fast_adapter_layers = {
            layer.layer_index: _clamp_delta_vector(layer.delta_vector, limit=0.12)
            for layer in checkpoint.adapter_layers
        }
        if checkpoint.adapter_layers:
            self._rare_heavy_activation_width = len(checkpoint.adapter_layers[0].delta_vector)
            self._rare_heavy_layer_indices = tuple(layer.layer_index for layer in checkpoint.adapter_layers)
        return ("online-fast:substrate-import",)

    def restore_online_fast_state(self, checkpoint: SubstrateOnlineFastCheckpoint) -> tuple[str, ...]:
        self.apply_online_fast_state(checkpoint)
        return ("online-fast:substrate-rollback",)

    def _remember_trace_shape(self, trace: TrainingTrace) -> None:
        layer_indices = tuple(
            sorted(
                {
                    activation.layer_index
                    for step in trace.steps
                    for activation in step.residual_activations
                }
            )
        )
        if layer_indices:
            self._rare_heavy_layer_indices = layer_indices
        for step in trace.steps:
            for activation in step.residual_activations:
                if activation.activation:
                    self._rare_heavy_activation_width = len(activation.activation)
                    return

    def _apply_adapter_to_residuals(
        self,
        residual_activations: tuple[ResidualActivation, ...],
    ) -> tuple[ResidualActivation, ...]:
        if (
            not self._rare_heavy_adapter_layers
            and not self._online_fast_adapter_layers
        ) or (
            self._rare_heavy_adapter_scale <= 0.0
            and self._online_fast_delta_scale <= 0.0
        ):
            return residual_activations
        adapted: list[ResidualActivation] = []
        for activation in residual_activations:
            rare_heavy_delta = self._rare_heavy_adapter_layers.get(activation.layer_index)
            online_fast_delta = self._online_fast_adapter_layers.get(activation.layer_index)
            if (rare_heavy_delta is None and online_fast_delta is None) or not activation.activation:
                adapted.append(activation)
                continue
            values = tuple(
                _clamp_signed(
                    base
                    + (
                        (rare_heavy_delta[index] * self._rare_heavy_adapter_scale)
                        if rare_heavy_delta is not None
                        else 0.0
                    )
                    + (
                        (online_fast_delta[index] * self._online_fast_delta_scale)
                        if online_fast_delta is not None
                        else 0.0
                    )
                )
                for index, base in enumerate(activation.activation)
            )
            adapted.append(
                ResidualActivation(
                    layer_index=activation.layer_index,
                    activation=values,
                    step=activation.step,
                )
            )
        return tuple(adapted)

    def _export_adapter_layers(self) -> tuple[SubstrateDeltaAdapterLayer, ...]:
        return tuple(
            SubstrateDeltaAdapterLayer(
                layer_index=layer_index,
                delta_vector=delta_vector,
                mean_abs_delta=_mean_abs_delta(delta_vector),
                description=(
                    f"Synthetic adapter delta for layer {layer_index} "
                    f"width={len(delta_vector)}."
                ),
            )
            for layer_index, delta_vector in sorted(self._rare_heavy_adapter_layers.items())
        )

    def _export_online_fast_layers(self) -> tuple[SubstrateDeltaAdapterLayer, ...]:
        return tuple(
            SubstrateDeltaAdapterLayer(
                layer_index=layer_index,
                delta_vector=delta_vector,
                mean_abs_delta=_mean_abs_delta(delta_vector),
                description=f"Synthetic online-fast delta for layer {layer_index}.",
            )
            for layer_index, delta_vector in sorted(self._online_fast_adapter_layers.items())
        )

    def _train_adapter_deltas(
        self,
        *,
        traces: tuple[TrainingTrace, ...],
        substrate_steps_per_trace: tuple[tuple[SubstrateSnapshot, ...], ...],
    ) -> tuple[tuple[SubstrateDeltaAdapterLayer, ...], float]:
        if not traces or not substrate_steps_per_trace:
            return ((), 0.0)
        torch = importlib.import_module("torch")
        target_vectors = self._target_layer_vectors(substrate_steps_per_trace=substrate_steps_per_trace)
        base_vectors = self._trace_layer_vectors(traces=traces)
        common_layers = tuple(layer for layer in self._rare_heavy_layer_indices if layer in target_vectors and layer in base_vectors)
        if not common_layers or self._rare_heavy_activation_width <= 0:
            return ((), 0.0)
        parameters = {
            layer: torch.tensor(
                self._rare_heavy_adapter_layers.get(
                    layer,
                    tuple(0.0 for _ in range(self._rare_heavy_activation_width)),
                ),
                dtype=torch.float32,
                requires_grad=True,
            )
            for layer in common_layers
        }
        optimizer = torch.optim.Adam(tuple(parameters.values()), lr=0.08)
        final_loss = 0.0
        for _ in range(12):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, dtype=torch.float32)
            for layer in common_layers:
                base = torch.tensor(base_vectors[layer], dtype=torch.float32)
                target = torch.tensor(target_vectors[layer], dtype=torch.float32)
                predicted = base + parameters[layer]
                total_loss = total_loss + torch.mean((predicted - target) ** 2)
                total_loss = total_loss + torch.mean(parameters[layer] ** 2) * 0.01
            total_loss.backward()
            optimizer.step()
            final_loss = float(total_loss.detach().item())
        adapter_layers = tuple(
            SubstrateDeltaAdapterLayer(
                layer_index=layer,
                delta_vector=_clamp_delta_vector(parameters[layer].detach().tolist(), limit=0.18),
                mean_abs_delta=_mean_abs_delta(parameters[layer].detach().tolist()),
                description=f"Synthetic trained adapter delta for layer {layer}.",
            )
            for layer in common_layers
        )
        return (adapter_layers, final_loss)

    def _target_layer_vectors(
        self,
        *,
        substrate_steps_per_trace: tuple[tuple[SubstrateSnapshot, ...], ...],
    ) -> dict[int, tuple[float, ...]]:
        values: dict[int, list[tuple[float, ...]]] = {}
        for batch in substrate_steps_per_trace:
            for snapshot in batch:
                for activation in snapshot.residual_activations:
                    if activation.activation:
                        values.setdefault(activation.layer_index, []).append(activation.activation)
        return {
            layer: tuple(
                sum(vector[index] for vector in vectors) / len(vectors)
                for index in range(len(vectors[0]))
            )
            for layer, vectors in values.items()
            if vectors
        }

    def _trace_layer_vectors(
        self,
        *,
        traces: tuple[TrainingTrace, ...],
    ) -> dict[int, tuple[float, ...]]:
        values: dict[int, list[tuple[float, ...]]] = {}
        for trace in traces:
            self._remember_trace_shape(trace)
            for step in trace.steps:
                for activation in step.residual_activations:
                    if activation.activation:
                        values.setdefault(activation.layer_index, []).append(activation.activation)
        return {
            layer: tuple(
                sum(vector[index] for vector in vectors) / len(vectors)
                for index in range(len(vectors[0]))
            )
            for layer, vectors in values.items()
            if vectors
        }


class TransformersOpenWeightResidualRuntime(OpenWeightResidualRuntime):
    """Frozen HF runtime with real middle-layer capture and intervention hooks."""

    def __init__(
        self,
        *,
        model_id: str,
        pretrained_source: str | None = None,
        device: str = "cpu",
        model: object | None = None,
        tokenizer: object | None = None,
        max_length: int = 64,
        top_k_logits: int = 8,
        activation_width: int = 8,
        layer_indices: tuple[int, ...] | None = None,
        hook_layer_selection: str = "middle",
        control_scale: float = 0.12,
        local_files_only: bool = False,
        runtime_origin: str = "hf-pretrained",
        allow_live_substrate_mutation: bool = False,
        allow_offline_substrate_training: bool = False,
    ) -> None:
        self._torch = importlib.import_module("torch")
        self._transformers = importlib.import_module("transformers")
        self.model_id = model_id
        self._pretrained_source = pretrained_source or model_id
        self.is_frozen = True
        self.supports_live_substrate_mutation = allow_live_substrate_mutation
        self.supports_offline_substrate_training = allow_offline_substrate_training
        self._device = self._resolve_device(device=device)
        self._max_length = max(1, max_length)
        self._top_k_logits = max(1, top_k_logits)
        self._activation_width = max(1, activation_width)
        self._control_scale = max(0.0, control_scale)
        self._runtime_origin = runtime_origin
        self.runtime_origin = runtime_origin
        self._tokenizer = tokenizer or self._load_tokenizer(
            model_id=self._pretrained_source,
            local_files_only=local_files_only,
        )
        self._model = model or self._transformers.AutoModelForCausalLM.from_pretrained(
            self._pretrained_source,
            local_files_only=local_files_only,
        )
        self._prepare_model()
        self._block_modules = self._resolve_transformer_blocks()
        self._layer_indices = self._normalize_layer_indices(
            requested=layer_indices,
            block_count=len(self._block_modules),
            hook_layer_selection=hook_layer_selection,
        )
        self._hidden_size = self._resolve_hidden_size()
        self._model_family = self._resolve_model_family()
        self._control_basis = self._build_control_basis(hidden_size=self._hidden_size)
        self._semantic_projection_dim = 24
        self._semantic_basis = self._build_semantic_basis(
            hidden_size=self._hidden_size,
            projection_dim=self._semantic_projection_dim,
        )
        self._semantic_anchor_profiles = _anchor_profile_bank(dim=self._semantic_projection_dim)
        base_text_weight, base_residual_weight = self._base_semantic_weights()
        self._rare_heavy_control_scale = self._control_scale
        self._rare_heavy_semantic_text_weight = base_text_weight
        self._rare_heavy_semantic_residual_weight = base_residual_weight
        self._rare_heavy_anchor_bias = tuple(0.0 for _ in RARE_HEAVY_ANCHOR_ORDER)
        self._rare_heavy_update_count = 0
        self._rare_heavy_adapter_scale = 0.0
        self._rare_heavy_adapter_deltas: dict[int, object] = {}
        self._online_fast_delta_scale = 0.0
        self._online_fast_update_count = 0
        self._online_fast_optimizer_state_norm = 0.0
        self._online_fast_parameter_change_rate = 0.0
        self._online_fast_adapter_deltas: dict[int, object] = {}
        self._online_fast_state_hash = ""
        self._online_fast_source_state_hash = ""
        self._online_fast_signal: tuple[float, ...] = ()
        self._online_fast_optimizer_state_description = ""

    def capture(self, *, source_text: str) -> OpenWeightRuntimeCapture:
        return self._capture_with_hooks(source_text=source_text)

    def apply_control(
        self,
        *,
        source_text: str,
        substrate_snapshot: SubstrateSnapshot,
        applied_control: tuple[float, ...],
        track_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> ResidualControlApplication:
        after_capture = self._capture_with_hooks(
            source_text=source_text,
            applied_control=applied_control,
            track_scale=track_scale,
        )
        before_summary = _summarize_real_activations(substrate_snapshot.residual_activations)
        after_summary = _summarize_real_activations(after_capture.residual_activations)
        logit_before = max(substrate_snapshot.token_logits, default=0.0)
        logit_after = max(after_capture.token_logits, default=0.0)
        downstream_effect = (
            _clamp_signed(after_summary[0] - before_summary[0]),
            _clamp_signed(after_summary[1] - before_summary[1]),
            _clamp_signed((logit_after - logit_before) + after_summary[2] - before_summary[2]),
        )
        control_energy = sum(abs(value) for value in applied_control) / max(len(applied_control), 1)
        applied_snapshot = SubstrateSnapshot(
            model_id=self.model_id,
            is_frozen=self.is_frozen,
            surface_kind=substrate_snapshot.surface_kind,
            token_logits=after_capture.token_logits,
            feature_surface=after_capture.feature_surface,
            residual_activations=after_capture.residual_activations,
            residual_sequence=after_capture.residual_sequence,
            unavailable_fields=substrate_snapshot.unavailable_fields,
            description=(
                f"{after_capture.description} Applied transformers residual control "
                f"{tuple(round(value, 3) for value in applied_control)}."
            ),
        )
        return ResidualControlApplication(
            applied_snapshot=applied_snapshot,
            downstream_effect=downstream_effect,
            control_energy=control_energy,
            backend_name=f"transformers-open-weight:{self.model_id}",
            description=(
                f"transformers-open-weight:{self.model_id} device={self._device} "
                f"layers={self._layer_indices} effect={tuple(round(value, 3) for value in downstream_effect)}."
            ),
        )

    def generate(
        self,
        *,
        prompt: str,
        system_context: str = "",
        chat_messages: tuple[tuple[str, str], ...] = (),
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        control_parameters: tuple[float, ...] = (),
        control_scale: float = 0.0,
        generation_constraints: "GenerationConstraints | None" = None,
    ) -> GenerationResult:
        effective_max_new_tokens = max_new_tokens
        effective_temperature = temperature
        effective_repetition_penalty = 1.08
        effective_top_p = 1.0
        if generation_constraints is not None:
            if generation_constraints.answer_depth_limit == "high-level-only":
                effective_max_new_tokens = min(effective_max_new_tokens, 96)
            elif generation_constraints.answer_depth_limit == "support-first":
                effective_max_new_tokens = min(effective_max_new_tokens, 128)
            if generation_constraints.response_mode in {"clarify", "refer-out"}:
                effective_temperature = min(effective_temperature, 0.45)
            (
                effective_max_new_tokens,
                effective_temperature,
                effective_repetition_penalty,
                effective_top_p,
            ) = self._apply_continuum_generation_controls(
                max_new_tokens=effective_max_new_tokens,
                temperature=effective_temperature,
                repetition_penalty=effective_repetition_penalty,
                top_p=effective_top_p,
                constraints=generation_constraints,
            )
        effective_prompt, model_inputs = self._build_generation_inputs(
            prompt=prompt,
            system_context=system_context,
            chat_messages=chat_messages,
        )
        input_ids = model_inputs["input_ids"]
        prompt_length = int(input_ids.shape[-1])

        control_delta = None
        if control_parameters and control_scale > 0:
            control_delta = self._build_control_delta(
                applied_control=control_parameters,
                track_scale=(control_scale, control_scale, control_scale),
            )
        captured_layers: dict[int, object] = {}
        hooks = [
            self._block_modules[layer_index].register_forward_hook(
                self._make_capture_hook(
                    layer_index=layer_index,
                    captured_layers=captured_layers,
                    control_delta=control_delta,
                )
            )
            for layer_index in self._layer_indices
        ]
        try:
            with self._torch.no_grad():
                generate_kwargs: dict[str, object] = {
                    "max_new_tokens": effective_max_new_tokens,
                    "do_sample": effective_temperature > 0,
                    "pad_token_id": getattr(self._tokenizer, "eos_token_id", 0) or 0,
                    "eos_token_id": self._generation_eos_token_id(),
                    "repetition_penalty": effective_repetition_penalty,
                }
                if effective_temperature > 0:
                    generate_kwargs["temperature"] = effective_temperature
                    if effective_top_p < 0.999:
                        generate_kwargs["top_p"] = effective_top_p
                output_ids = self._model.generate(**model_inputs, **generate_kwargs)
        finally:
            for hook in hooks:
                hook.remove()

        new_token_ids = output_ids[0, prompt_length:]
        generated_text = self._decode_generated_text(token_ids=new_token_ids)
        if generation_constraints is not None:
            generated_text = self._apply_generation_constraints(
                text=generated_text,
                constraints=generation_constraints,
            )
        token_count = int(new_token_ids.shape[0])

        capture = None
        if captured_layers:
            try:
                logits_pass = self._model(output_ids[:, :prompt_length])
                logits = self._extract_logits(outputs=logits_pass)
                capture = self._build_runtime_capture(
                    source_text=effective_prompt,
                    input_ids=input_ids,
                    logits=logits,
                    captured_layers=captured_layers,
                    control_applied=control_delta is not None,
                )
            except Exception:
                pass

        return GenerationResult(
            text=generated_text,
            token_count=token_count,
            capture=capture,
            description=(
                f"Generated {token_count} tokens from {self.model_id} "
                f"device={self._device} temp={effective_temperature} "
                f"profile={generation_constraints.decoding_profile if generation_constraints is not None else 'balanced'} "
                f"control={'on' if control_delta is not None else 'off'}"
            ),
        )

    def _apply_generation_constraints(
        self,
        *,
        text: str,
        constraints: "GenerationConstraints",
    ) -> str:
        compact = text.strip()
        if not compact:
            return compact
        if constraints.max_questions <= 0:
            question_count = 0
            truncated_chars: list[str] = []
            for char in compact:
                if char == "?":
                    question_count += 1
                    if question_count > constraints.max_questions:
                        continue
                truncated_chars.append(char)
            compact = "".join(truncated_chars)
        for phrase in constraints.required_disclaimer_phrases:
            if phrase and phrase not in compact:
                compact = f"{compact} {phrase}".strip()
        if constraints.ordering_driver in {"continuum-support-first", "continuum-support-clarify"}:
            compact = self._support_first_trim(compact)
        elif constraints.ordering_driver == "continuum-structure-first":
            compact = self._structure_first_trim(compact)
        if constraints.ordering_bias and len(compact.split()) > 80:
            compact = compact[:320].rstrip()
        return compact

    def _apply_continuum_generation_controls(
        self,
        *,
        max_new_tokens: int,
        temperature: float,
        repetition_penalty: float,
        top_p: float,
        constraints: "GenerationConstraints",
    ) -> tuple[int, float, float, float]:
        target = constraints.continuum_target_position
        effective_max_new_tokens = max_new_tokens
        effective_temperature = temperature
        effective_repetition_penalty = repetition_penalty
        effective_top_p = top_p
        if constraints.decoding_profile == "support-first":
            effective_max_new_tokens = min(effective_max_new_tokens, 112)
            effective_temperature = min(effective_temperature, 0.42)
            effective_repetition_penalty = max(effective_repetition_penalty, 1.04)
            effective_top_p = min(effective_top_p, 0.92)
        elif constraints.decoding_profile == "clarify-first":
            effective_max_new_tokens = min(effective_max_new_tokens, 84)
            effective_temperature = min(effective_temperature, 0.34)
            effective_repetition_penalty = max(effective_repetition_penalty, 1.06)
            effective_top_p = min(effective_top_p, 0.82)
        elif constraints.decoding_profile == "structure-first":
            effective_max_new_tokens = min(effective_max_new_tokens, 176)
            effective_temperature = min(effective_temperature, 0.28)
            effective_repetition_penalty = max(effective_repetition_penalty, 1.10)
            effective_top_p = min(effective_top_p, 0.74)

        if target >= 0.75:
            effective_temperature = min(effective_temperature, 0.40)
            effective_max_new_tokens = min(effective_max_new_tokens, 104)
        elif target < 0.42:
            effective_max_new_tokens = min(effective_max_new_tokens, 168)
            effective_repetition_penalty = max(effective_repetition_penalty, 1.09)

        return (
            effective_max_new_tokens,
            effective_temperature,
            effective_repetition_penalty,
            effective_top_p,
        )

    def _support_first_trim(self, text: str) -> str:
        sentences = [part.strip() for part in text.replace("\n", " ").split(".") if part.strip()]
        if not sentences:
            return text
        compact = ". ".join(sentences[:2]).strip()
        if not text.strip().endswith("?") and compact and not compact.endswith("."):
            compact += "."
        if len(compact) > 160:
            compact = compact[:160].rstrip(". ").rstrip()
        return compact

    def _structure_first_trim(self, text: str) -> str:
        compact = text.strip()
        if not compact:
            return compact
        return compact[:420].rstrip()

    def _build_generation_inputs(
        self,
        *,
        prompt: str,
        system_context: str,
        chat_messages: tuple[tuple[str, str], ...],
    ) -> tuple[str, dict[str, object]]:
        source_text = self._chat_messages_to_source_text(
            prompt=prompt,
            system_context=system_context,
            chat_messages=chat_messages,
        )
        if chat_messages:
            apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
            if callable(apply_chat_template):
                chat_payload = [
                    {
                        "role": role,
                        "content": content,
                    }
                    for role, content in chat_messages
                ]
                try:
                    encoded = apply_chat_template(
                        chat_payload,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                    )
                except TypeError:
                    encoded = None
                if encoded is not None:
                    return source_text, self._prepare_model_inputs(encoded=encoded)
                rendered = apply_chat_template(
                    chat_payload,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered.strip(), self._tokenize(source_text=rendered.strip())
            fallback_sections = [f"{role.upper()}:\n{content}" for role, content in chat_messages if content.strip()]
            fallback_sections.append("ASSISTANT:\n")
            rendered_fallback = "\n\n".join(fallback_sections).strip()
            if rendered_fallback:
                return rendered_fallback, self._tokenize(source_text=rendered_fallback)
        return source_text, self._tokenize(source_text=source_text)

    def _chat_messages_to_source_text(
        self,
        *,
        prompt: str,
        system_context: str,
        chat_messages: tuple[tuple[str, str], ...],
    ) -> str:
        if chat_messages:
            rendered_messages = [
                f"{role}: {content}"
                for role, content in chat_messages
                if content.strip()
            ]
            return "\n".join(rendered_messages).strip() or "<empty>"
        full_prompt = f"{system_context}\n{prompt}".strip() if system_context else prompt.strip()
        return full_prompt or "<empty>"

    def _prepare_model_inputs(self, *, encoded) -> dict[str, object]:
        model_inputs: dict[str, object] = {}
        for key, value in encoded.items():
            if isinstance(value, self._torch.Tensor):
                model_inputs[key] = value.to(self._device)
            else:
                model_inputs[key] = value
        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, self._torch.Tensor):
            raise ValueError(f"Transformers runtime '{self.model_id}' chat template did not return tensor input_ids.")
        return model_inputs

    def _generation_eos_token_id(self) -> int | list[int]:
        token_ids: list[int] = []
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int) and eos_token_id >= 0:
            token_ids.append(eos_token_id)
        elif isinstance(eos_token_id, (list, tuple)):
            token_ids.extend(token_id for token_id in eos_token_id if isinstance(token_id, int) and token_id >= 0)
        convert_tokens_to_ids = getattr(self._tokenizer, "convert_tokens_to_ids", None)
        if callable(convert_tokens_to_ids):
            for token in ("<|im_end|>", "<|eot_id|>"):
                token_id = convert_tokens_to_ids(token)
                if isinstance(token_id, int) and token_id >= 0:
                    token_ids.append(token_id)
        unique_ids = list(dict.fromkeys(token_ids))
        if not unique_ids:
            return 0
        if len(unique_ids) == 1:
            return unique_ids[0]
        return unique_ids

    def _load_tokenizer(self, *, model_id: str, local_files_only: bool):
        try:
            return self._transformers.AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=local_files_only,
            )
        except Exception as first_exc:
            if not local_files_only:
                raise
            try:
                return self._transformers.AutoTokenizer.from_pretrained(
                    model_id,
                    local_files_only=True,
                    use_fast=False,
                )
            except Exception:
                raise first_exc

    def _resolve_device(self, *, device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if self._torch.cuda.is_available() else "cpu"

    def _prepare_model(self) -> None:
        self._model.to(self._device)
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    def _resolve_transformer_blocks(self) -> tuple[object, ...]:
        candidate_paths = (
            ("model", "layers"),
            ("language_model", "model", "layers"),
            ("model", "decoder", "layers"),
            ("decoder", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
            ("transformer", "blocks"),
            ("backbone", "layers"),
            ("layers",),
        )
        for path in candidate_paths:
            resolved = self._resolve_module_path(path)
            if resolved is not None:
                return resolved
        raise NotImplementedError(
            f"Transformers runtime '{self.model_id}' could not resolve transformer blocks for hook capture."
        )

    def _normalize_layer_indices(
        self,
        *,
        requested: tuple[int, ...] | None,
        block_count: int,
        hook_layer_selection: str = "middle",
    ) -> tuple[int, ...]:
        if block_count <= 0:
            raise ValueError(f"Transformers runtime '{self.model_id}' has no hookable transformer blocks.")
        if requested is not None:
            normalized = tuple(sorted({index for index in requested if 0 <= index < block_count}))
            if not normalized:
                raise ValueError(f"Transformers runtime '{self.model_id}' received no valid hook layer indices.")
            return normalized
        if hook_layer_selection == "all":
            return tuple(range(block_count))
        if hook_layer_selection != "middle":
            raise ValueError(
                f"Unsupported hook_layer_selection {hook_layer_selection!r}; expected 'middle' or 'all'."
            )
        if block_count <= 3:
            return tuple(range(block_count))
        middle = block_count // 2
        return tuple(sorted({middle - 1, middle, min(block_count - 1, middle + 1)}))

    def _resolve_hidden_size(self) -> int:
        try:
            return int(self._model.config.hidden_size)
        except AttributeError:
            pass
        try:
            return int(self._model.config.n_embd)
        except AttributeError:
            pass
        try:
            return int(self._model.config.d_model)
        except AttributeError as exc:
            raise AttributeError(
                f"Transformers runtime '{self.model_id}' could not resolve hidden size from model config."
            ) from exc

    def _resolve_model_family(self) -> str:
        model_type = getattr(self._model.config, "model_type", None)
        if isinstance(model_type, str) and model_type:
            return model_type
        return type(self._model).__name__

    def _resolve_module_path(self, path: tuple[str, ...]) -> tuple[object, ...] | None:
        current = self._model
        for segment in path:
            try:
                current = getattr(current, segment)
            except AttributeError:
                return None
        return self._as_module_tuple(current)

    def _as_module_tuple(self, container: object) -> tuple[object, ...] | None:
        try:
            resolved = tuple(container)  # type: ignore[arg-type]
        except TypeError:
            return None
        return resolved if resolved else None

    def _build_control_basis(self, *, hidden_size: int):
        positions = self._torch.arange(hidden_size, dtype=self._torch.float32)
        rows = []
        for factor in (1.0, 2.0, 3.0):
            row = self._torch.sin((positions + 1.0) * 0.173 * factor) + self._torch.cos(
                (positions + 1.0) * 0.117 * (factor + 1.0)
            )
            row = row / row.norm().clamp_min(1e-6)
            rows.append(row)
        return self._torch.stack(rows, dim=0).to(self._device)

    def _build_semantic_basis(self, *, hidden_size: int, projection_dim: int):
        positions = self._torch.arange(hidden_size, dtype=self._torch.float32)
        rows = []
        for factor in range(1, projection_dim + 1):
            row = self._torch.sin((positions + 1.0) * 0.071 * factor) + self._torch.cos(
                (positions + 1.0) * 0.043 * (factor + 1.0)
            )
            row = row / row.norm().clamp_min(1e-6)
            rows.append(row)
        return self._torch.stack(rows, dim=0).to(self._device)

    def _base_semantic_weights(self) -> tuple[float, float]:
        if self._runtime_origin == "builtin-fallback":
            return (0.9, 0.1)
        return (0.55, 0.45)

    def _semantic_profile_from_capture(self, *, source_text: str, captured_layers: dict[int, object]) -> tuple[float, ...]:
        text_profile = _hashed_semantic_embedding(source_text, dim=self._semantic_projection_dim)
        residual_profile = self._residual_semantic_profile(captured_layers=captured_layers)
        text_weight, residual_weight = _normalize_semantic_weights(
            text_weight=self._rare_heavy_semantic_text_weight,
            residual_weight=self._rare_heavy_semantic_residual_weight,
        )
        combined = tuple(
            text_value * text_weight + residual_value * residual_weight
            for text_value, residual_value in zip(text_profile, residual_profile, strict=True)
        )
        return _normalize_vector(combined)

    def _residual_semantic_profile(self, *, captured_layers: dict[int, object]) -> tuple[float, ...]:
        stacked = self._torch.stack(
            [captured_layers[layer_index][0].to(self._device, dtype=self._torch.float32) for layer_index in self._layer_indices],
            dim=0,
        )
        mean_hidden = stacked.mean(dim=(0, 1))
        tail_hidden = stacked[:, -1, :].mean(dim=0)
        dispersion_hidden = stacked.std(dim=1).mean(dim=0) if stacked.shape[1] > 1 else self._torch.zeros_like(mean_hidden)
        composite = mean_hidden * 0.55 + tail_hidden * 0.30 + dispersion_hidden * 0.15
        projected = self._semantic_basis.to(dtype=self._torch.float32) @ composite.to(dtype=self._torch.float32)
        norm = projected.norm().clamp_min(1e-6)
        normalized = (projected / norm).detach().cpu().tolist()
        return tuple(float(value) for value in normalized)

    def _semantic_feature_surface(
        self,
        *,
        source_text: str,
        captured_layers: dict[int, object],
    ) -> tuple[FeatureSignal, ...]:
        profile = self._semantic_profile_from_capture(
            source_text=source_text,
            captured_layers=captured_layers,
        )
        similarities = {
            name: _cosine_similarity(profile, anchor_profile)
            for name, anchor_profile in self._semantic_anchor_profiles.items()
        }
        centered_similarities = {
            name: similarities[name] - (sum(similarities.values()) / max(len(similarities), 1))
            for name in similarities
        }
        distribution = {
            name: probability
            for name, probability in zip(
                similarities.keys(),
                _softmax_probabilities(tuple(centered_similarities.values()), temperature=0.22),
                strict=True,
            )
        }

        def relative_pull(target_name: str) -> float:
            target = similarities[target_name]
            others = [value for name, value in similarities.items() if name != target_name]
            runner_up = max(others) if others else 0.0
            absolute = _clamp_unit((target + 1.0) / 2.0)
            margin = _clamp_unit(0.5 + (target - runner_up) * 3.2)
            return _clamp_unit(
                distribution[target_name] * 0.65
                + margin * 0.25
                + absolute * 0.10
            )

        raw_task_pull = relative_pull("task")
        raw_support_pull = relative_pull("support")
        raw_repair_pull = relative_pull("repair")
        raw_exploration_pull = relative_pull("exploration")
        raw_directive_pull = relative_pull("directive")
        raw_task_pull = _clamp_unit(raw_task_pull + self._rare_heavy_anchor_bias[0])
        raw_support_pull = _clamp_unit(raw_support_pull + self._rare_heavy_anchor_bias[1])
        raw_repair_pull = _clamp_unit(raw_repair_pull + self._rare_heavy_anchor_bias[2])
        raw_exploration_pull = _clamp_unit(raw_exploration_pull + self._rare_heavy_anchor_bias[3])
        raw_directive_pull = _clamp_unit(raw_directive_pull + self._rare_heavy_anchor_bias[4])
        semantic_task_pull = _clamp_unit(raw_task_pull * 0.35 + raw_directive_pull * 0.65)
        semantic_support_pull = _clamp_unit(raw_support_pull * 0.75 + raw_repair_pull * 0.25)
        semantic_repair_pull = _clamp_unit(raw_repair_pull * 0.80 + raw_support_pull * 0.20)
        text_weight, residual_weight = _normalize_semantic_weights(
            text_weight=self._rare_heavy_semantic_text_weight,
            residual_weight=self._rare_heavy_semantic_residual_weight,
        )

        return (
            FeatureSignal(
                name="semantic_task_pull",
                values=(semantic_task_pull,),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_support_pull",
                values=(semantic_support_pull,),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_repair_pull",
                values=(semantic_repair_pull,),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_exploration_pull",
                values=(raw_exploration_pull,),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_directive_pull",
                values=(raw_directive_pull,),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_text_weight",
                values=(text_weight,),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_residual_weight",
                values=(residual_weight,),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_rare_heavy_update_count",
                values=(_clamp_unit(self._rare_heavy_update_count / 10.0),),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_delta_parameter_count",
                values=(_clamp_unit(len(self._rare_heavy_adapter_deltas) * self._hidden_size / 512.0),),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_online_fast_update_count",
                values=(_clamp_unit(self._online_fast_update_count / 10.0),),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_online_fast_delta_parameter_count",
                values=(_clamp_unit(len(self._online_fast_adapter_deltas) * self._hidden_size / 512.0),),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_online_fast_parameter_change_rate",
                values=(_clamp_unit(self._online_fast_parameter_change_rate),),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="substrate_online_fast_experimental_mode",
                values=(1.0 if self.experimental_live_mutation_enabled else 0.0,),
                source="transformers-open-weight-semantic",
            ),
        )

    def _capture_with_hooks(
        self,
        *,
        source_text: str,
        applied_control: tuple[float, ...] | None = None,
        track_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> OpenWeightRuntimeCapture:
        effective_source = source_text.strip() or "<empty>"
        model_inputs = self._tokenize(source_text=effective_source)
        input_ids = model_inputs["input_ids"]
        captured_layers: dict[int, object] = {}
        control_delta = None
        if applied_control is not None:
            control_delta = self._build_control_delta(applied_control=applied_control, track_scale=track_scale)
        hooks = [
            self._block_modules[layer_index].register_forward_hook(
                self._make_capture_hook(
                    layer_index=layer_index,
                    captured_layers=captured_layers,
                    control_delta=control_delta,
                )
            )
            for layer_index in self._layer_indices
        ]
        try:
            with self._torch.no_grad():
                outputs = self._model(**model_inputs)
        finally:
            for hook in hooks:
                hook.remove()
        logits = self._extract_logits(outputs=outputs)
        return self._build_runtime_capture(
            source_text=effective_source,
            input_ids=input_ids,
            logits=logits,
            captured_layers=captured_layers,
            control_applied=applied_control is not None,
        )

    def _capture_hidden_state_means(
        self,
        *,
        source_text: str,
    ) -> dict[int, object]:
        effective_source = source_text.strip() or "<empty>"
        model_inputs = self._tokenize(source_text=effective_source)
        captured_layers: dict[int, object] = {}

        def make_hook(layer_index: int):
            def hook(module, args, output):
                del module
                del args
                captured_layers[layer_index] = self._extract_hidden_tensor(output=output).detach().cpu()
                return None

            return hook

        hooks = [
            self._block_modules[layer_index].register_forward_hook(make_hook(layer_index))
            for layer_index in self._layer_indices
        ]
        try:
            with self._torch.no_grad():
                self._model(**model_inputs)
        finally:
            for hook in hooks:
                hook.remove()
        return {
            layer_index: captured_layers[layer_index][0].to(self._device, dtype=self._torch.float32).mean(dim=0)
            for layer_index in self._layer_indices
            if layer_index in captured_layers
        }

    def _target_semantic_profile_tensor(
        self,
        *,
        substrates: Sequence[SubstrateSnapshot],
        source_text: str,
    ):
        feature_names = {
            "task": "semantic_task_pull",
            "support": "semantic_support_pull",
            "repair": "semantic_repair_pull",
            "exploration": "semantic_exploration_pull",
            "directive": "semantic_directive_pull",
        }
        weights = tuple(_mean_feature_value(substrates, name=feature_names[anchor]) for anchor in RARE_HEAVY_ANCHOR_ORDER)
        if any(weight > 1e-6 for weight in weights):
            target = self._torch.zeros(self._semantic_projection_dim, dtype=self._torch.float32, device=self._device)
            for anchor, weight in zip(RARE_HEAVY_ANCHOR_ORDER, weights, strict=True):
                anchor_profile = self._torch.tensor(
                    self._semantic_anchor_profiles[anchor],
                    dtype=self._torch.float32,
                    device=self._device,
                )
                target = target + anchor_profile * float(weight)
            return target / target.norm().clamp_min(1e-6)
        return self._torch.tensor(
            _hashed_semantic_embedding(source_text, dim=self._semantic_projection_dim),
            dtype=self._torch.float32,
            device=self._device,
        )

    def _target_residual_tensor(
        self,
        *,
        substrates: Sequence[SubstrateSnapshot],
    ):
        return self._torch.tensor(
            max(
                _mean_residual_magnitude(substrates),
                _mean_feature_value(substrates, name="residual_mean_abs"),
            ),
            dtype=self._torch.float32,
            device=self._device,
        )

    def _train_adapter_deltas(
        self,
        *,
        traces: tuple[TrainingTrace, ...],
        substrate_steps_per_trace: tuple[tuple[SubstrateSnapshot, ...], ...],
    ) -> tuple[tuple[SubstrateDeltaAdapterLayer, ...], float]:
        if not traces or not substrate_steps_per_trace:
            return ((), 0.0)
        paired = tuple(zip(traces, substrate_steps_per_trace))
        if not paired:
            return ((), 0.0)
        parameters = {
            layer_index: self._torch.nn.Parameter(
                self._rare_heavy_adapter_deltas.get(
                    layer_index,
                    self._torch.zeros(self._hidden_size, dtype=self._torch.float32, device=self._device),
                ).detach().clone().to(self._device, dtype=self._torch.float32)
            )
            for layer_index in self._layer_indices
        }
        optimizer = self._torch.optim.Adam(tuple(parameters.values()), lr=0.03)
        final_loss = 0.0
        for _ in range(max(4, min(12, len(paired) * 2))):
            optimizer.zero_grad()
            total_loss = self._torch.tensor(0.0, dtype=self._torch.float32, device=self._device)
            for trace, batch in paired:
                base_means = self._capture_hidden_state_means(source_text=trace.source_text)
                available_layers = tuple(layer for layer in self._layer_indices if layer in base_means)
                if not available_layers:
                    continue
                predicted_layers = self._torch.stack(
                    tuple(base_means[layer] + parameters[layer] for layer in available_layers),
                    dim=0,
                )
                composite = predicted_layers.mean(dim=0)
                projected = self._semantic_basis.to(dtype=self._torch.float32) @ composite.to(dtype=self._torch.float32)
                predicted_profile = projected / projected.norm().clamp_min(1e-6)
                target_profile = self._target_semantic_profile_tensor(
                    substrates=batch,
                    source_text=trace.source_text,
                )
                predicted_residual = self._torch.tanh(predicted_layers.abs().mean())
                target_residual = self._target_residual_tensor(substrates=batch)
                total_loss = total_loss + self._torch.mean((predicted_profile - target_profile) ** 2)
                total_loss = total_loss + (predicted_residual - target_residual) ** 2 * 0.15
            total_loss = total_loss / max(len(paired), 1)
            total_loss = total_loss + sum(parameter.pow(2).mean() for parameter in parameters.values()) * 0.002
            total_loss.backward()
            optimizer.step()
            final_loss = float(total_loss.detach().item())
        adapter_layers = tuple(
            SubstrateDeltaAdapterLayer(
                layer_index=layer_index,
                delta_vector=_clamp_delta_vector(parameters[layer_index].detach().cpu().tolist(), limit=0.18),
                mean_abs_delta=_mean_abs_delta(parameters[layer_index].detach().cpu().tolist()),
                description=(
                    f"Transformers trained adapter delta for layer {layer_index} "
                    f"hidden={self._hidden_size}."
                ),
            )
            for layer_index in self._layer_indices
        )
        return (adapter_layers, final_loss)

    def _tokenize(self, *, source_text: str) -> dict[str, object]:
        encoded = self._tokenizer(
            source_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        model_inputs: dict[str, object] = {}
        for key, value in encoded.items():
            if isinstance(value, self._torch.Tensor):
                model_inputs[key] = value.to(self._device)
            else:
                model_inputs[key] = value
        input_ids = model_inputs.get("input_ids")
        if not isinstance(input_ids, self._torch.Tensor):
            raise ValueError(f"Transformers runtime '{self.model_id}' tokenizer did not return tensor input_ids.")
        return model_inputs

    def _build_control_delta(
        self,
        *,
        applied_control: tuple[float, ...],
        track_scale: tuple[float, ...],
    ):
        coeffs = []
        for index in range(3):
            coeffs.append(
                float(applied_control[min(index, len(applied_control) - 1)])
                * float(track_scale[min(index, len(track_scale) - 1)])
            )
        control_vector = self._torch.tensor(coeffs, dtype=self._torch.float32, device=self._device)
        delta = control_vector @ self._control_basis
        return delta * self._rare_heavy_control_scale

    def export_rare_heavy_state(self, *, checkpoint_id: str | None = None) -> SubstrateRareHeavyCheckpoint:
        adapter_layers = self._export_adapter_layers()
        training_mode = "adapter-delta-v2" if adapter_layers else "bounded-state-v1"
        return SubstrateRareHeavyCheckpoint(
            checkpoint_id=checkpoint_id or f"{self.model_id}:rare-heavy",
            model_id=self.model_id,
            runtime_origin=self.runtime_origin,
            control_scale=self._rare_heavy_control_scale,
            semantic_text_weight=self._rare_heavy_semantic_text_weight,
            semantic_residual_weight=self._rare_heavy_semantic_residual_weight,
            semantic_anchor_bias=self._rare_heavy_anchor_bias,
            update_count=self._rare_heavy_update_count,
            source_batch_count=0,
            mean_sequence_length=0.0,
            mean_residual_magnitude=0.0,
            description=(
                f"Transformers rare-heavy checkpoint for {self.model_id} "
                f"updates={self._rare_heavy_update_count}."
            ),
            checkpoint_version=2 if adapter_layers else 1,
            training_mode=training_mode,
            compatibility_fingerprint=_build_compatibility_fingerprint(
                model_id=self.model_id,
                runtime_origin=self.runtime_origin,
                hidden_size=self._hidden_size,
                layer_indices=self._layer_indices,
                training_mode=training_mode,
            ),
            adapter_scale=self._rare_heavy_adapter_scale,
            adapter_parameter_count=_adapter_parameter_count(adapter_layers),
            adapter_training_loss=0.0,
            adapter_layers=adapter_layers,
        )

    def import_rare_heavy_state(self, checkpoint: SubstrateRareHeavyCheckpoint) -> tuple[str, ...]:
        self.require_substrate_artifact_import(operation="import_rare_heavy_state()")
        if checkpoint.model_id != self.model_id:
            raise ValueError(
                f"Transformers runtime {self.model_id!r} cannot import checkpoint for {checkpoint.model_id!r}."
            )
        if checkpoint.compatibility_fingerprint and checkpoint.training_mode != "bounded-state-v1":
            expected = _build_compatibility_fingerprint(
                model_id=self.model_id,
                runtime_origin=self.runtime_origin,
                hidden_size=self._hidden_size,
                layer_indices=self._layer_indices,
                training_mode=checkpoint.training_mode,
            )
            if checkpoint.compatibility_fingerprint != expected:
                raise ValueError(
                    f"Checkpoint fingerprint {checkpoint.compatibility_fingerprint!r} does not match runtime {expected!r}."
                )
        text_weight, residual_weight = _normalize_semantic_weights(
            text_weight=checkpoint.semantic_text_weight,
            residual_weight=checkpoint.semantic_residual_weight,
        )
        self._rare_heavy_control_scale = max(0.04, min(0.30, checkpoint.control_scale))
        self._rare_heavy_semantic_text_weight = text_weight
        self._rare_heavy_semantic_residual_weight = residual_weight
        anchor_bias = tuple(
            max(-0.2, min(0.2, value))
            for value in checkpoint.semantic_anchor_bias[: len(RARE_HEAVY_ANCHOR_ORDER)]
        )
        if len(anchor_bias) < len(RARE_HEAVY_ANCHOR_ORDER):
            anchor_bias = anchor_bias + tuple(
                0.0 for _ in range(len(RARE_HEAVY_ANCHOR_ORDER) - len(anchor_bias))
            )
        self._rare_heavy_anchor_bias = anchor_bias
        self._rare_heavy_update_count = max(0, checkpoint.update_count)
        self._rare_heavy_adapter_scale = max(0.0, checkpoint.adapter_scale)
        self._rare_heavy_adapter_deltas = {
            layer.layer_index: self._torch.tensor(
                _clamp_delta_vector(layer.delta_vector),
                dtype=self._torch.float32,
                device=self._device,
            )
            for layer in checkpoint.adapter_layers
        }
        return ("rare-heavy:substrate-import",)

    def restore_rare_heavy_state(self, checkpoint: SubstrateRareHeavyCheckpoint) -> tuple[str, ...]:
        self.import_rare_heavy_state(checkpoint)
        return ("rare-heavy:substrate-rollback",)

    def train_rare_heavy(
        self,
        *,
        traces: tuple[TrainingTrace, ...] = (),
        substrate_steps_per_trace: tuple[tuple[SubstrateSnapshot, ...], ...],
        checkpoint_id: str | None = None,
    ) -> SubstrateRareHeavyCheckpoint:
        self.require_offline_substrate_training(operation="train_rare_heavy()")
        base_text_weight, base_residual_weight = self._base_semantic_weights()
        checkpoint = _derive_rare_heavy_checkpoint(
            checkpoint_id=checkpoint_id or f"{self.model_id}:rare-heavy-trained",
            model_id=self.model_id,
            runtime_origin=self.runtime_origin,
            current_control_scale=self._rare_heavy_control_scale,
            default_text_weight=base_text_weight,
            default_residual_weight=base_residual_weight,
            previous_update_count=self._rare_heavy_update_count,
            substrate_steps_per_trace=substrate_steps_per_trace,
        )
        adapter_layers, training_loss = self._train_adapter_deltas(
            traces=traces,
            substrate_steps_per_trace=substrate_steps_per_trace,
        )
        if not adapter_layers:
            return checkpoint
        return _checkpoint_with_adapter_payload(
            checkpoint,
            training_mode="adapter-delta-v2",
            compatibility_fingerprint=_build_compatibility_fingerprint(
                model_id=self.model_id,
                runtime_origin=self.runtime_origin,
                hidden_size=self._hidden_size,
                layer_indices=self._layer_indices,
                training_mode="adapter-delta-v2",
            ),
            adapter_scale=1.0,
            adapter_training_loss=training_loss,
            adapter_layers=adapter_layers,
            description=(
                f"{checkpoint.description} Adapter-delta payload "
                f"layers={len(adapter_layers)} loss={training_loss:.4f}."
            ),
        )

    def clone_for_rare_heavy(self) -> "OpenWeightResidualRuntime":
        cloned = TransformersOpenWeightResidualRuntime(
            model_id=self.model_id,
            pretrained_source=self._pretrained_source,
            device=self._device,
            model=self._model,
            tokenizer=self._tokenizer,
            max_length=self._max_length,
            top_k_logits=self._top_k_logits,
            activation_width=self._activation_width,
            layer_indices=self._layer_indices,
            control_scale=self._control_scale,
            runtime_origin=self._runtime_origin,
            allow_offline_substrate_training=True,
        )
        cloned.import_rare_heavy_state(self.export_rare_heavy_state())
        return cloned

    def export_online_fast_state(
        self,
        *,
        checkpoint_id: str | None = None,
    ) -> SubstrateOnlineFastCheckpoint:
        self.require_live_substrate_mutation(operation="export_online_fast_state()")
        adapter_layers = self._export_online_fast_layers()
        return SubstrateOnlineFastCheckpoint(
            checkpoint_id=checkpoint_id or f"{self.model_id}:online-fast",
            model_id=self.model_id,
            runtime_origin=self.runtime_origin,
            delta_scale=self._online_fast_delta_scale,
            update_count=self._online_fast_update_count,
            source_wave_id="runtime",
            source_turn_index=self._online_fast_update_count,
            gate="online",
            optimizer_state_norm=self._online_fast_optimizer_state_norm,
            parameter_change_rate=self._online_fast_parameter_change_rate,
            description=(
                f"Transformers online-fast checkpoint for {self.model_id} "
                f"updates={self._online_fast_update_count}."
            ),
            compatibility_fingerprint=_build_compatibility_fingerprint(
                model_id=self.model_id,
                runtime_origin=self.runtime_origin,
                hidden_size=self._hidden_size,
                layer_indices=self._layer_indices,
                training_mode="online-fast-delta-v1",
            ),
            adapter_parameter_count=_adapter_parameter_count(adapter_layers),
            adapter_layers=adapter_layers,
            fast_state_hash=self._online_fast_state_hash,
            source_fast_state_hash=self._online_fast_source_state_hash,
            fast_memory_signal=self._online_fast_signal,
            optimizer_state_description=self._online_fast_optimizer_state_description,
        )

    def apply_online_fast_state(self, checkpoint: SubstrateOnlineFastCheckpoint) -> tuple[str, ...]:
        self.require_live_substrate_mutation(operation="apply_online_fast_state()")
        if checkpoint.model_id != self.model_id:
            raise ValueError(
                f"Transformers runtime {self.model_id!r} cannot import online-fast checkpoint for "
                f"{checkpoint.model_id!r}."
            )
        if checkpoint.compatibility_fingerprint:
            expected = _build_compatibility_fingerprint(
                model_id=self.model_id,
                runtime_origin=self.runtime_origin,
                hidden_size=self._hidden_size,
                layer_indices=self._layer_indices,
                training_mode=checkpoint.training_mode,
            )
            if checkpoint.compatibility_fingerprint != expected:
                raise ValueError(
                    f"Online-fast checkpoint fingerprint {checkpoint.compatibility_fingerprint!r} "
                    f"does not match runtime {expected!r}."
                )
        self._online_fast_delta_scale = max(0.0, min(0.18, checkpoint.delta_scale))
        self._online_fast_update_count = max(0, checkpoint.update_count)
        self._online_fast_optimizer_state_norm = _clamp_unit(checkpoint.optimizer_state_norm)
        self._online_fast_parameter_change_rate = _clamp_unit(checkpoint.parameter_change_rate)
        self._online_fast_state_hash = checkpoint.fast_state_hash
        self._online_fast_source_state_hash = checkpoint.source_fast_state_hash
        self._online_fast_signal = checkpoint.fast_memory_signal
        self._online_fast_optimizer_state_description = checkpoint.optimizer_state_description
        self._online_fast_adapter_deltas = {
            layer.layer_index: self._torch.tensor(
                _clamp_delta_vector(
                    (
                        layer.delta_vector[: self._hidden_size]
                        + tuple(0.0 for _ in range(max(self._hidden_size - len(layer.delta_vector), 0)))
                    ),
                    limit=0.12,
                ),
                dtype=self._torch.float32,
                device=self._device,
            )
            for layer in checkpoint.adapter_layers
        }
        return ("online-fast:substrate-import",)

    def restore_online_fast_state(self, checkpoint: SubstrateOnlineFastCheckpoint) -> tuple[str, ...]:
        self.apply_online_fast_state(checkpoint)
        return ("online-fast:substrate-rollback",)

    def _adapter_delta_for_layer(self, *, layer_index: int):
        rare_heavy_delta = self._rare_heavy_adapter_deltas.get(layer_index)
        online_fast_delta = self._online_fast_adapter_deltas.get(layer_index)
        if rare_heavy_delta is None and online_fast_delta is None:
            return None
        combined = None
        if rare_heavy_delta is not None and self._rare_heavy_adapter_scale > 0.0:
            combined = rare_heavy_delta * self._rare_heavy_adapter_scale
        if online_fast_delta is not None and self._online_fast_delta_scale > 0.0:
            scaled_online_fast = online_fast_delta * self._online_fast_delta_scale
            combined = scaled_online_fast if combined is None else combined + scaled_online_fast
        return combined

    def _export_adapter_layers(self) -> tuple[SubstrateDeltaAdapterLayer, ...]:
        return tuple(
            SubstrateDeltaAdapterLayer(
                layer_index=layer_index,
                delta_vector=_clamp_delta_vector(delta.detach().cpu().tolist(), limit=0.18),
                mean_abs_delta=_mean_abs_delta(delta.detach().cpu().tolist()),
                description=(
                    f"Transformers adapter delta for layer {layer_index} "
                    f"hidden={self._hidden_size}."
                ),
            )
            for layer_index, delta in sorted(self._rare_heavy_adapter_deltas.items())
        )

    def _export_online_fast_layers(self) -> tuple[SubstrateDeltaAdapterLayer, ...]:
        return tuple(
            SubstrateDeltaAdapterLayer(
                layer_index=layer_index,
                delta_vector=_clamp_delta_vector(delta.detach().cpu().tolist(), limit=0.12),
                mean_abs_delta=_mean_abs_delta(delta.detach().cpu().tolist()),
                description=(
                    f"Transformers online-fast delta for layer {layer_index} "
                    f"hidden={self._hidden_size}."
                ),
            )
            for layer_index, delta in sorted(self._online_fast_adapter_deltas.items())
        )

    def _make_capture_hook(
        self,
        *,
        layer_index: int,
        captured_layers: dict[int, object],
        control_delta,
    ):
        def hook(module, args, output):
            del module
            del args
            hidden = self._extract_hidden_tensor(output=output)
            adapter_delta = self._adapter_delta_for_layer(layer_index=layer_index)
            if adapter_delta is None and control_delta is None:
                captured_layers[layer_index] = hidden.detach().cpu()
                return None
            adjusted = hidden
            if adapter_delta is not None:
                adjusted = adjusted + adapter_delta.view(1, 1, -1).to(dtype=hidden.dtype)
            if control_delta is not None:
                adjusted = adjusted + control_delta.view(1, 1, -1).to(dtype=hidden.dtype)
            captured_layers[layer_index] = adjusted.detach().cpu()
            if isinstance(output, tuple):
                return (adjusted, *output[1:])
            return adjusted

        return hook

    def _extract_hidden_tensor(self, *, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        if not isinstance(hidden, self._torch.Tensor):
            raise TypeError(f"Transformers runtime '{self.model_id}' hook output was not tensor-shaped.")
        return hidden

    def _extract_logits(self, *, outputs):
        try:
            logits = outputs.logits
        except AttributeError:
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                raise TypeError(f"Transformers runtime '{self.model_id}' outputs did not expose logits.")
        if not isinstance(logits, self._torch.Tensor):
            raise TypeError(f"Transformers runtime '{self.model_id}' logits were not tensor-shaped.")
        return logits.detach().cpu()

    def _decode_tokens(self, *, input_ids) -> tuple[str, ...]:
        token_ids = tuple(int(token_id) for token_id in input_ids[0].tolist())
        try:
            raw_tokens = tuple(self._tokenizer.convert_ids_to_tokens(token_ids))
        except AttributeError:
            raw_tokens = tuple(str(token_id) for token_id in token_ids)
        normalized = []
        for index, token in enumerate(raw_tokens):
            cleaned = token.replace("Ġ", "").replace("▁", "").strip()
            normalized.append(cleaned or f"<tok:{token_ids[index]}>")
        return tuple(normalized)

    def _decode_generated_text(self, *, token_ids) -> str:
        try:
            return str(self._tokenizer.decode(token_ids, skip_special_tokens=True)).strip()
        except AttributeError:
            pass
        flattened_ids = tuple(int(token_id) for token_id in token_ids.tolist())
        try:
            tokens = tuple(self._tokenizer.convert_ids_to_tokens(flattened_ids))
        except AttributeError:
            tokens = tuple(str(token_id) for token_id in flattened_ids)
        cleaned_tokens = [
            token.replace("Ġ", " ").replace("▁", " ").strip()
            for token in tokens
            if token.strip()
        ]
        return " ".join(cleaned_tokens).strip()

    def _build_runtime_capture(
        self,
        *,
        source_text: str,
        input_ids,
        logits,
        captured_layers: dict[int, object],
        control_applied: bool,
    ) -> OpenWeightRuntimeCapture:
        if not captured_layers:
            raise RuntimeError(f"Transformers runtime '{self.model_id}' did not record any hooked activations.")
        tokens = self._decode_tokens(input_ids=input_ids)
        step_count = len(tokens)
        last_logits = logits[0, -1]
        probabilities = self._torch.softmax(last_logits, dim=-1)
        top_k = min(self._top_k_logits, int(probabilities.shape[-1]))
        top_values, _ = self._torch.topk(probabilities, k=top_k)
        token_logits = tuple(float(value) for value in top_values.tolist())
        top_entropy = _normalized_entropy(token_logits)
        top_margin = _clamp_unit(token_logits[0] - token_logits[1]) if len(token_logits) > 1 else _clamp_unit(
            token_logits[0] if token_logits else 0.0
        )
        hook_coverage = _clamp_unit(len(self._layer_indices) / max(len(self._block_modules), 1))
        residual_sequence: list[ResidualSequenceStep] = []
        for step_index, token in enumerate(tokens):
            step_residuals = tuple(
                ResidualActivation(
                    layer_index=layer_index,
                    activation=self._tensor_to_activation_tuple(captured_layers[layer_index][0, step_index, :]),
                    step=step_index,
                )
                for layer_index in self._layer_indices
            )
            step_summary = _summarize_real_activations(step_residuals)
            step_features = (
                FeatureSignal(
                    name="residual_mean_abs",
                    values=(step_summary[0],),
                    source="transformers-open-weight",
                    layer_hint=self._layer_indices[0],
                ),
                FeatureSignal(
                    name="residual_peak_abs",
                    values=(step_summary[1],),
                    source="transformers-open-weight",
                    layer_hint=self._layer_indices[-1],
                ),
                FeatureSignal(
                    name="sequence_progress",
                    values=((step_index + 1) / max(step_count, 1),),
                    source="transformers-open-weight",
                ),
                FeatureSignal(
                    name="hook_layer_coverage",
                    values=(hook_coverage,),
                    source="transformers-open-weight",
                ),
            )
            residual_sequence.append(
                ResidualSequenceStep(
                    step=step_index,
                    token=token,
                    feature_surface=step_features,
                    residual_activations=step_residuals,
                    description=(
                        f"Transformers hook capture for token '{token}' on layers {self._layer_indices}"
                        f"{' with control' if control_applied else ''}."
                    ),
                )
            )
        latest_activations = residual_sequence[-1].residual_activations
        latest_summary = _summarize_real_activations(latest_activations)
        feature_surface = (
            FeatureSignal(
                name="residual_mean_abs",
                values=(latest_summary[0],),
                source="transformers-open-weight",
                layer_hint=self._layer_indices[0],
            ),
            FeatureSignal(
                name="residual_peak_abs",
                values=(latest_summary[1],),
                source="transformers-open-weight",
                layer_hint=self._layer_indices[-1],
            ),
            FeatureSignal(
                name="top_logit_confidence",
                values=(max(token_logits, default=0.0),),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="top_logit_entropy",
                values=(top_entropy,),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="top_logit_margin",
                values=(top_margin,),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="residual_signed_mean",
                values=(latest_summary[2],),
                source="transformers-open-weight",
                layer_hint=self._layer_indices[-1],
            ),
            FeatureSignal(
                name="hook_layer_coverage",
                values=(hook_coverage,),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="hidden_size_scale",
                values=(_clamp_unit(self._hidden_size / 4096.0),),
                source="transformers-open-weight",
            ),
            FeatureSignal(
                name="fallback_active",
                values=(1.0 if self._runtime_origin == "builtin-fallback" else 0.0,),
                source="transformers-open-weight",
            ),
        ) + self._semantic_feature_surface(
            source_text=source_text,
            captured_layers=captured_layers,
        )
        return OpenWeightRuntimeCapture(
            token_logits=token_logits,
            feature_surface=feature_surface,
            residual_activations=latest_activations,
            residual_sequence=tuple(residual_sequence),
            description=(
                f"Transformers open-weight capture model={self.model_id} device={self._device} "
                f"family={self._model_family} origin={self._runtime_origin} "
                f"tokens={len(tokens)} layers={self._layer_indices} source_len={len(source_text)} "
                f"live_mode={self.live_mutation_mode}."
            ),
        )

    def _tensor_to_activation_tuple(self, tensor) -> tuple[float, ...]:
        values = tensor.detach().cpu().tolist()
        return tuple(float(value) for value in values[: self._activation_width])


def build_builtin_transformers_runtime(
    *,
    model_id: str = "builtin-transformers-runtime",
    device: str = "cpu",
    tokenizer: object | None = None,
    layer_indices: tuple[int, ...] | None = None,
    hook_layer_selection: str = "middle",
    allow_live_substrate_mutation: bool = False,
) -> TransformersOpenWeightResidualRuntime:
    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=256,
                n_positions=64,
                n_ctx=64,
                n_embd=48,
                n_layer=4,
                n_head=4,
            )
        )
    
    return TransformersOpenWeightResidualRuntime(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer or HashingWhitespaceTokenizer(vocab_size=256),
        device=device,
        layer_indices=layer_indices,
        hook_layer_selection=hook_layer_selection,
        activation_width=8,
        top_k_logits=8,
        runtime_origin="builtin-fallback",
        allow_live_substrate_mutation=allow_live_substrate_mutation,
    )


def build_transformers_runtime_with_fallback(
    *,
    model_id: str,
    model_source: str | None = None,
    device: str = "auto",
    layer_indices: tuple[int, ...] | None = None,
    hook_layer_selection: str = "middle",
    local_files_only: bool = False,
    fallback_to_builtin: bool | None = None,
    fallback_mode: SubstrateFallbackMode | str | None = None,
    runtime_mode: LocalSubstrateRuntimeMode | str | None = None,
    builtin_model_id: str = "builtin-transformers-runtime",
    allow_live_substrate_mutation: bool = False,
) -> TransformersOpenWeightResidualRuntime:
    resolved_runtime_mode = resolve_local_runtime_mode(
        runtime_mode=runtime_mode,
        local_files_only=local_files_only,
        fallback_mode=fallback_mode,
        fallback_to_builtin=fallback_to_builtin,
    )
    if resolved_runtime_mode is LocalSubstrateRuntimeMode.BUILTIN_ONLY:
        return build_builtin_transformers_runtime(
            model_id=builtin_model_id,
            device=device,
            layer_indices=layer_indices,
            hook_layer_selection=hook_layer_selection,
            allow_live_substrate_mutation=allow_live_substrate_mutation,
        )
    resolved_mode = resolve_substrate_fallback_mode(
        fallback_mode=fallback_mode,
        fallback_to_builtin=fallback_to_builtin,
    )
    effective_local_files_only = local_files_only
    effective_runtime_origin = "hf-local" if local_files_only else "hf-pretrained"
    if resolved_runtime_mode is LocalSubstrateRuntimeMode.STRICT_LOCAL:
        effective_local_files_only = True
        resolved_mode = SubstrateFallbackMode.DENY
        effective_runtime_origin = "hf-local"
    elif resolved_runtime_mode is LocalSubstrateRuntimeMode.PREFER_LOCAL:
        effective_local_files_only = True
        resolved_mode = SubstrateFallbackMode.ALLOW_BUILTIN
        effective_runtime_origin = "hf-local"
    try:
        return TransformersOpenWeightResidualRuntime(
            model_id=model_id,
            pretrained_source=model_source or model_id,
            device=device,
            layer_indices=layer_indices,
            hook_layer_selection=hook_layer_selection,
            local_files_only=effective_local_files_only,
            runtime_origin=effective_runtime_origin,
            allow_live_substrate_mutation=allow_live_substrate_mutation,
        )
    except Exception as exc:
        if resolved_mode is not SubstrateFallbackMode.ALLOW_BUILTIN or not _is_transformers_runtime_fallback_error(exc):
            raise
        return build_builtin_transformers_runtime(
            model_id=builtin_model_id,
            device=device,
            layer_indices=layer_indices,
            hook_layer_selection=hook_layer_selection,
            allow_live_substrate_mutation=allow_live_substrate_mutation,
        )


def run_hook_layer_calibration(
    *,
    model_id: str,
    source_text: str,
    runtime_builder: Callable[[tuple[int, ...]], OpenWeightResidualRuntime],
    layer_index_sets: tuple[tuple[int, ...], ...],
) -> HookLayerCalibrationReport:
    cases: list[HookLayerCalibrationCase] = []
    for layer_indices in layer_index_sets:
        runtime = runtime_builder(layer_indices)
        capture = runtime.capture(source_text=source_text)
        feature_map = {signal.name: signal.values[0] for signal in capture.feature_surface if signal.values}
        task_pull = feature_map.get("semantic_task_pull", 0.0)
        support_pull = feature_map.get("semantic_support_pull", 0.0)
        repair_pull = feature_map.get("semantic_repair_pull", 0.0)
        directive_pull = feature_map.get("semantic_directive_pull", 0.0)
        exploration_pull = feature_map.get("semantic_exploration_pull", 0.0)
        hook_coverage = feature_map.get("hook_layer_coverage", 0.0)
        fallback_active = feature_map.get("fallback_active", 0.0)
        semantic_separation = _clamp_unit(
            max(task_pull, support_pull, repair_pull, directive_pull, exploration_pull)
            - min(task_pull, support_pull, repair_pull, directive_pull, exploration_pull)
        )
        signal_quality = _clamp_unit(
            hook_coverage * 0.35
            + (1.0 - fallback_active) * 0.25
            + feature_map.get("top_logit_margin", 0.0) * 0.15
            + (1.0 - feature_map.get("top_logit_entropy", 0.0)) * 0.10
            + semantic_separation * 0.15
        )
        cases.append(
            HookLayerCalibrationCase(
                layer_indices=layer_indices,
                hook_layer_coverage=round(hook_coverage, 4),
                residual_sequence_length=len(capture.residual_sequence),
                semantic_separation=round(semantic_separation, 4),
                signal_quality=round(signal_quality, 4),
                runtime_origin=getattr(runtime, "runtime_origin", "unknown"),
                description=capture.description,
            )
        )
    ranked = sorted(
        cases,
        key=lambda item: (
            item.signal_quality,
            item.semantic_separation,
            item.hook_layer_coverage,
            item.residual_sequence_length,
        ),
        reverse=True,
    )
    recommended_layers = ranked[0].layer_indices if ranked else ()
    return HookLayerCalibrationReport(
        model_id=model_id,
        source_text=source_text,
        cases=tuple(cases),
        recommended_layers=recommended_layers,
        description=(
            f"Hook layer calibration for {model_id} over {len(cases)} cases; "
            f"recommended_layers={recommended_layers}."
        ),
    )
    resolved_mode = resolve_substrate_fallback_mode(
        fallback_mode=fallback_mode,
        fallback_to_builtin=fallback_to_builtin,
    )
    effective_local_files_only = local_files_only
    effective_runtime_origin = "hf-local" if local_files_only else "hf-pretrained"
    if resolved_runtime_mode is LocalSubstrateRuntimeMode.STRICT_LOCAL:
        effective_local_files_only = True
        resolved_mode = SubstrateFallbackMode.DENY
        effective_runtime_origin = "hf-local"
    elif resolved_runtime_mode is LocalSubstrateRuntimeMode.PREFER_LOCAL:
        effective_local_files_only = True
        resolved_mode = SubstrateFallbackMode.ALLOW_BUILTIN
        effective_runtime_origin = "hf-local"
    try:
        return TransformersOpenWeightResidualRuntime(
            model_id=model_id,
            device=device,
            local_files_only=effective_local_files_only,
            runtime_origin=effective_runtime_origin,
        )
    except Exception as exc:
        if resolved_mode is not SubstrateFallbackMode.ALLOW_BUILTIN or not _is_transformers_runtime_fallback_error(exc):
            raise
        return build_builtin_transformers_runtime(
            model_id=builtin_model_id,
            device=device,
        )


def _is_transformers_runtime_fallback_error(exc: Exception) -> bool:
    if isinstance(exc, (OSError, ValueError, RuntimeError, TimeoutError)):
        return True
    module_name = type(exc).__module__
    class_name = type(exc).__name__
    if module_name.startswith("httpx") and class_name.endswith("Timeout"):
        return True
    if module_name.startswith("requests") and class_name in {"ReadTimeout", "ConnectTimeout", "Timeout"}:
        return True
    if module_name.startswith("huggingface_hub.errors"):
        return True
    return False


def probe_local_model_compatibility(
    *,
    model_id: str,
    model_source: str | None = None,
    device: str = "cpu",
) -> LocalModelCompatibilityReport:
    transformers = importlib.import_module("transformers")
    load_source = model_source or model_id
    local_tokenizer_available = False
    local_model_available = False
    strict_local_runtime_available = False
    error_type: str | None = None
    error_message = ""
    try:
        try:
            transformers.AutoTokenizer.from_pretrained(
                load_source,
                local_files_only=True,
            )
            local_tokenizer_available = True
        except Exception:
            transformers.AutoTokenizer.from_pretrained(
                load_source,
                local_files_only=True,
                use_fast=False,
            )
            local_tokenizer_available = True
        transformers.AutoModelForCausalLM.from_pretrained(
            load_source,
            local_files_only=True,
        )
        local_model_available = True
        runtime = build_transformers_runtime_with_fallback(
            model_id=model_id,
            model_source=model_source,
            device=device,
            local_files_only=True,
            runtime_mode=LocalSubstrateRuntimeMode.STRICT_LOCAL,
        )
        runtime.capture(source_text="local compatibility probe")
        strict_local_runtime_available = True
        description = (
            f"Local model compatibility OK for {model_id}: tokenizer/model/runtime all available."
        )
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
        description = (
            f"Local model compatibility probe failed for {model_id}: "
            f"{error_type}: {error_message}"
        )
    return LocalModelCompatibilityReport(
        model_id=model_id,
        local_tokenizer_available=local_tokenizer_available,
        local_model_available=local_model_available,
        strict_local_runtime_available=strict_local_runtime_available,
        error_type=error_type,
        error_message=error_message,
        description=description,
    )


def _normalized_token_value(token: str) -> float:
    if not token:
        return 0.0
    return min(sum(ord(ch) for ch in token) / (len(token) * 128.0), 1.0)


def build_training_trace(
    *,
    trace_id: str,
    source_text: str,
    layer_count: int = 3,
) -> TrainingTrace:
    tokens = tuple(part for part in source_text.split() if part.strip()) or ("<empty>",)
    steps: list[TraceStep] = []
    for step_index, token in enumerate(tokens):
        base_value = _normalized_token_value(token)
        feature_surface = (
            FeatureSignal(
                name="token_value",
                values=(base_value,),
                source="residual-sim",
                layer_hint=0,
            ),
            FeatureSignal(
                name="token_length",
                values=(min(len(token) / 12.0, 1.0),),
                source="residual-sim",
                layer_hint=1,
            ),
        )
        residual_activations = tuple(
            ResidualActivation(
                layer_index=layer_index,
                activation=(
                    min(base_value + layer_index * 0.1, 1.0),
                    min((step_index + 1) / max(len(tokens), 1), 1.0),
                    min(len(token) / 10.0, 1.0),
                ),
                step=step_index,
            )
            for layer_index in range(layer_count)
        )
        steps.append(
            TraceStep(
                step=step_index,
                token=token,
                feature_surface=feature_surface,
                residual_activations=residual_activations,
            )
        )
    return TrainingTrace(trace_id=trace_id, source_text=source_text, steps=tuple(steps))


class TrainingTraceDataset:
    """Minimal replay dataset for ETA/NL stage-two experiments."""

    def __init__(self, traces: Iterable[TrainingTrace] = ()) -> None:
        self._traces = list(traces)

    @property
    def traces(self) -> tuple[TrainingTrace, ...]:
        return tuple(self._traces)

    def add_trace(self, trace: TrainingTrace) -> None:
        self._traces.append(trace)

    def latest(self) -> TrainingTrace:
        if not self._traces:
            raise ValueError("TrainingTraceDataset is empty.")
        return self._traces[-1]


class SimulatedResidualSubstrateAdapter(ResidualStreamSubstrateAdapter):
    """Executable residual backend for stage-two ETA experiments."""

    def __init__(self, *, trace: TrainingTrace, model_id: str = "residual-sim-backend") -> None:
        self._trace = trace
        latest_step = trace.steps[-1]
        super().__init__(
            model_id=model_id,
            residual_activations=latest_step.residual_activations,
            residual_sequence=tuple(
                ResidualSequenceStep(
                    step=step.step,
                    token=step.token,
                    feature_surface=step.feature_surface,
                    residual_activations=step.residual_activations,
                    description=f"Trace token '{step.token}' at step {step.step}.",
                )
                for step in trace.steps
            ),
            token_logits=tuple(
                min(sum(feature.values) / max(len(feature.values), 1), 1.0)
                for feature in latest_step.feature_surface
            ),
            is_frozen=True,
            feature_surface=latest_step.feature_surface,
        )

    @property
    def trace(self) -> TrainingTrace:
        return self._trace

    async def capture(self, *, source_text: str | None = None) -> SubstrateSnapshot:
        return await super().capture(source_text=source_text or self._trace.source_text)
