from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import importlib
import math
from typing import Iterable, Sequence

from volvence_zero.substrate.adapter import (
    FeatureSignal,
    ResidualActivation,
    ResidualSequenceStep,
    ResidualStreamSubstrateAdapter,
    SubstrateSnapshot,
)


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


class SubstrateFallbackMode(str, Enum):
    ALLOW_BUILTIN = "allow-builtin"
    DENY = "deny"


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


SEMANTIC_ANCHOR_BANK: dict[str, tuple[str, ...]] = {
    "task": (
        "Break the work into concrete steps and decide the best execution order.",
        "Give me a direct plan, clear prioritization, and tradeoff reasoning.",
        "把任务拆开 直接判断先做什么 给我明确顺序和执行步骤",
        "别安抚了 直接判断优先级 现在就给我明确取舍",
    ),
    "support": (
        "Stay with me first, help me feel steadier before solving anything.",
        "Offer warmth, reassurance, and emotional support without rushing.",
        "先陪我稳住 情绪支持 温和一点 不要急着给方案",
    ),
    "repair": (
        "Repair trust, de-escalate pressure, and reduce tension before widening scope.",
        "Slow the interaction down and stabilize the relationship frame.",
        "先修复关系 降低张力 稳住局面 再继续推进",
    ),
    "exploration": (
        "Explore the space gradually, narrow options, and reason step by step.",
        "Guide the conversation through uncertainty without rushing to closure.",
        "一起探索 逐步收窄 先看可能性 再决定方向",
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

    def __init__(self, *, model_id: str = "synthetic-open-weight-runtime") -> None:
        self.model_id = model_id
        self.is_frozen = True

    def capture(self, *, source_text: str) -> OpenWeightRuntimeCapture:
        trace = build_training_trace(trace_id=f"{self.model_id}:capture", source_text=source_text)
        latest_step = trace.steps[-1]
        return OpenWeightRuntimeCapture(
            token_logits=tuple(
                min(sum(feature.values) / max(len(feature.values), 1), 1.0)
                for feature in latest_step.feature_surface
            ),
            feature_surface=latest_step.feature_surface,
            residual_activations=latest_step.residual_activations,
            residual_sequence=tuple(
                ResidualSequenceStep(
                    step=step.step,
                    token=step.token,
                    feature_surface=step.feature_surface,
                    residual_activations=step.residual_activations,
                    description=f"Synthetic hook token '{step.token}' at step {step.step}.",
                )
                for step in trace.steps
            ),
            description=f"Synthetic frozen open-weight capture for len={len(source_text)}.",
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
        result = TraceResidualInterventionBackend().apply_control(
            substrate_snapshot=substrate_snapshot,
            applied_control=applied_control,
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


class TransformersOpenWeightResidualRuntime(OpenWeightResidualRuntime):
    """Frozen HF runtime with real middle-layer capture and intervention hooks."""

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cpu",
        model: object | None = None,
        tokenizer: object | None = None,
        max_length: int = 64,
        top_k_logits: int = 8,
        activation_width: int = 8,
        layer_indices: tuple[int, ...] | None = None,
        control_scale: float = 0.12,
        local_files_only: bool = False,
        runtime_origin: str = "hf-pretrained",
    ) -> None:
        self._torch = importlib.import_module("torch")
        self._transformers = importlib.import_module("transformers")
        self.model_id = model_id
        self.is_frozen = True
        self._device = self._resolve_device(device=device)
        self._max_length = max(1, max_length)
        self._top_k_logits = max(1, top_k_logits)
        self._activation_width = max(1, activation_width)
        self._control_scale = max(0.0, control_scale)
        self._runtime_origin = runtime_origin
        self._tokenizer = tokenizer or self._transformers.AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        self._model = model or self._transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        self._prepare_model()
        self._block_modules = self._resolve_transformer_blocks()
        self._layer_indices = self._normalize_layer_indices(
            requested=layer_indices,
            block_count=len(self._block_modules),
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
    ) -> tuple[int, ...]:
        if block_count <= 0:
            raise ValueError(f"Transformers runtime '{self.model_id}' has no hookable transformer blocks.")
        if requested is not None:
            normalized = tuple(sorted({index for index in requested if 0 <= index < block_count}))
            if not normalized:
                raise ValueError(f"Transformers runtime '{self.model_id}' received no valid hook layer indices.")
            return normalized
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

    def _semantic_profile_from_capture(self, *, source_text: str, captured_layers: dict[int, object]) -> tuple[float, ...]:
        text_profile = _hashed_semantic_embedding(source_text, dim=self._semantic_projection_dim)
        residual_profile = self._residual_semantic_profile(captured_layers=captured_layers)
        if self._runtime_origin == "builtin-fallback":
            text_weight = 0.9
            residual_weight = 0.1
        else:
            text_weight = 0.55
            residual_weight = 0.45
        combined = tuple(
            text_value * text_weight + residual_value * residual_weight
            for text_value, residual_value in zip(text_profile, residual_profile, strict=True)
        )
        return _normalize_vector(combined)

    def _residual_semantic_profile(self, *, captured_layers: dict[int, object]) -> tuple[float, ...]:
        stacked = self._torch.stack(
            [captured_layers[layer_index][0].to(self._device) for layer_index in self._layer_indices],
            dim=0,
        )
        mean_hidden = stacked.mean(dim=(0, 1))
        tail_hidden = stacked[:, -1, :].mean(dim=0)
        dispersion_hidden = stacked.std(dim=1).mean(dim=0) if stacked.shape[1] > 1 else self._torch.zeros_like(mean_hidden)
        composite = mean_hidden * 0.55 + tail_hidden * 0.30 + dispersion_hidden * 0.15
        projected = self._semantic_basis @ composite
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

        def relative_pull(target_name: str) -> float:
            target = similarities[target_name]
            others = [value for name, value in similarities.items() if name != target_name]
            other_mean = sum(others) / len(others) if others else 0.0
            absolute = _clamp_unit((target + 1.0) / 2.0)
            contrast = _clamp_unit(0.5 + (target - other_mean) * 1.4)
            return _clamp_unit(absolute * 0.35 + contrast * 0.65)

        return (
            FeatureSignal(
                name="semantic_task_pull",
                values=(relative_pull("task"),),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_support_pull",
                values=(relative_pull("support"),),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_repair_pull",
                values=(relative_pull("repair"),),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_exploration_pull",
                values=(relative_pull("exploration"),),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_text_weight",
                values=((0.9 if self._runtime_origin == "builtin-fallback" else 0.55),),
                source="transformers-open-weight-semantic",
            ),
            FeatureSignal(
                name="semantic_residual_weight",
                values=((0.1 if self._runtime_origin == "builtin-fallback" else 0.45),),
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
        return delta * self._control_scale

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
            if control_delta is None:
                captured_layers[layer_index] = hidden.detach().cpu()
                return None
            adjusted = hidden + control_delta.view(1, 1, -1).to(dtype=hidden.dtype)
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
                f"tokens={len(tokens)} layers={self._layer_indices} source_len={len(source_text)}."
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
        layer_indices=layer_indices or (1, 2),
        activation_width=8,
        top_k_logits=8,
        runtime_origin="builtin-fallback",
    )


def build_transformers_runtime_with_fallback(
    *,
    model_id: str,
    device: str = "auto",
    local_files_only: bool = False,
    fallback_to_builtin: bool | None = None,
    fallback_mode: SubstrateFallbackMode | str | None = None,
    builtin_model_id: str = "builtin-transformers-runtime",
) -> TransformersOpenWeightResidualRuntime:
    resolved_mode = resolve_substrate_fallback_mode(
        fallback_mode=fallback_mode,
        fallback_to_builtin=fallback_to_builtin,
    )
    try:
        return TransformersOpenWeightResidualRuntime(
            model_id=model_id,
            device=device,
            local_files_only=local_files_only,
            runtime_origin="hf-pretrained",
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
