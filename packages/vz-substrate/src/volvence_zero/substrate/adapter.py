from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import TYPE_CHECKING, Any, Mapping

from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel

if TYPE_CHECKING:
    from volvence_zero.substrate.residual_backend import OpenWeightResidualRuntime


class SurfaceKind(str, Enum):
    PLACEHOLDER = "placeholder"
    FEATURE_SURFACE = "feature-surface"
    RESIDUAL_STREAM = "residual-stream"


@dataclass(frozen=True)
class ResidualActivation:
    layer_index: int
    activation: tuple[float, ...]
    step: int


@dataclass(frozen=True)
class FeatureSignal:
    name: str
    values: tuple[float, ...]
    source: str
    layer_hint: int | None = None


@dataclass(frozen=True)
class ResidualSequenceStep:
    step: int
    token: str
    feature_surface: tuple[FeatureSignal, ...]
    residual_activations: tuple[ResidualActivation, ...]
    description: str


@dataclass(frozen=True)
class UnavailableField:
    field_name: str
    reason: str
    detail: str


@dataclass(frozen=True)
class SubstrateSnapshot:
    model_id: str
    is_frozen: bool
    surface_kind: SurfaceKind
    token_logits: tuple[float, ...]
    feature_surface: tuple[FeatureSignal, ...]
    residual_activations: tuple[ResidualActivation, ...]
    residual_sequence: tuple[ResidualSequenceStep, ...]
    unavailable_fields: tuple[UnavailableField, ...]
    description: str


def feature_signal_map(feature_surface: tuple[FeatureSignal, ...]) -> dict[str, tuple[float, ...]]:
    return {signal.name: signal.values for signal in feature_surface}


def feature_signal_value(
    feature_surface: tuple[FeatureSignal, ...],
    *,
    name: str,
    index: int = 0,
    default: float = 0.0,
) -> float:
    values = feature_signal_map(feature_surface).get(name)
    if values is None or index >= len(values):
        return default
    return float(values[index])


class SubstrateAdapter(ABC):
    """Adapts the current model surface into the public substrate contract."""

    model_id: str
    is_frozen: bool
    surface_kind: SurfaceKind

    @abstractmethod
    async def capture(self, *, source_text: str | None = None) -> SubstrateSnapshot:
        """Capture the currently available substrate state."""


class PlaceholderSubstrateAdapter(SubstrateAdapter):
    """Most conservative adapter when no feature surface is available yet."""

    def __init__(self, *, model_id: str, detail: str = "No substrate surface is available yet.") -> None:
        self.model_id = model_id
        self.is_frozen = True
        self.surface_kind = SurfaceKind.PLACEHOLDER
        self._detail = detail

    async def capture(self, *, source_text: str | None = None) -> SubstrateSnapshot:
        return SubstrateSnapshot(
            model_id=self.model_id,
            is_frozen=self.is_frozen,
            surface_kind=self.surface_kind,
            token_logits=(),
            feature_surface=(),
            residual_activations=(),
            residual_sequence=(),
            unavailable_fields=(
                UnavailableField(
                    field_name="feature_surface",
                    reason="placeholder-adapter",
                    detail=self._detail,
                ),
                UnavailableField(
                    field_name="residual_activations",
                    reason="placeholder-adapter",
                    detail="Residual-stream hooks are not available in the current substrate mode.",
                ),
                UnavailableField(
                    field_name="residual_sequence",
                    reason="placeholder-adapter",
                    detail="Residual-step sequence is not available in the current substrate mode.",
                ),
            ),
            description="Placeholder substrate snapshot with no live model surface.",
        )


class FeatureSurfaceSubstrateAdapter(SubstrateAdapter):
    """
    Current stable adapter for a model surface that can expose coarse features
    and optional logits, but not full residual-stream activations.
    """

    def __init__(
        self,
        *,
        model_id: str,
        feature_surface: tuple[FeatureSignal, ...],
        token_logits: tuple[float, ...] = (),
        is_frozen: bool = True,
    ) -> None:
        self.model_id = model_id
        self.is_frozen = is_frozen
        self.surface_kind = SurfaceKind.FEATURE_SURFACE
        self._feature_surface = feature_surface
        self._token_logits = token_logits

    async def capture(self, *, source_text: str | None = None) -> SubstrateSnapshot:
        source_detail = "Feature-surface substrate snapshot."
        if source_text:
            source_detail = f"Feature-surface substrate snapshot for source_text len={len(source_text)}."
        return SubstrateSnapshot(
            model_id=self.model_id,
            is_frozen=self.is_frozen,
            surface_kind=self.surface_kind,
            token_logits=self._token_logits,
            feature_surface=self._feature_surface,
            residual_activations=(),
            residual_sequence=(),
            unavailable_fields=(
                UnavailableField(
                    field_name="residual_activations",
                    reason="feature-surface-adapter",
                    detail="Current substrate adapter does not expose residual-stream activations.",
                ),
                UnavailableField(
                    field_name="residual_sequence",
                    reason="feature-surface-adapter",
                    detail="Current substrate adapter does not expose residual-step sequences.",
                ),
            ),
            description=source_detail,
        )


def _semantic_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    ascii_buffer: list[str] = []
    compact = "".join(char for char in text.lower() if not char.isspace())
    for char in text.lower():
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
    tokens.extend(compact[index : index + 2] for index in range(len(compact) - 1))
    return tuple(tokens)


def _semantic_embedding(text: str, *, dim: int = 256) -> tuple[float, ...]:
    tokens = _semantic_tokens(text)
    if not tokens:
        return tuple(0.0 for _ in range(dim))
    vector = [0.0 for _ in range(dim)]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-6:
        return tuple(0.0 for _ in range(dim))
    return tuple(value / norm for value in vector)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _softmax(values: tuple[float, ...], *, temperature: float = 0.18) -> tuple[float, ...]:
    if not values:
        return ()
    scale = max(temperature, 1e-6)
    shifted = tuple((value - max(values)) / scale for value in values)
    exp_values = tuple(math.exp(value) for value in shifted)
    total = sum(exp_values)
    if total <= 1e-12:
        return tuple(1.0 / len(values) for _ in values)
    return tuple(value / total for value in exp_values)


_SEMANTIC_PROTOTYPES: Mapping[str, tuple[float, ...]] = {
    "task": _semantic_embedding(
        "concrete problem solving troubleshoot edit revise check compare debug fix "
        "configuration steps next action code implementation python script error index "
        "jwt verification signature library dashboard reporting operations espresso "
        "machine grind puck meeting email draft phrase"
    ),
    "support": _semantic_embedding(
        "emotional support feeling overwhelmed sad hurt scared vulnerable reassurance "
        "steady presence warmth trust comfort stuck heavy tense heard low mood"
    ),
    "repair": _semantic_embedding(
        "repair rupture cold procedural blame trust broken start over apologize "
        "felt dismissed restore safety"
    ),
    "exploration": _semantic_embedding(
        "think through options uncertainty decide explore tradeoff not sure clarify "
        "life decision direction"
    ),
    "directive": _semantic_embedding(
        "do this write change fix check tweak concrete answer direct instruction "
        "specific next step"
    ),
}
_SOCIAL_PROTOTYPE = _semantic_embedding(
    "casual social chat hello introduction getting to know shared interest pet "
    "weekend hobby friendly light conversation acquaintance"
)


def semantic_feature_surface_from_text(
    source_text: str,
    *,
    source: str = "substrate-semantic-surface",
    fallback_active: float = 1.0,
) -> tuple[FeatureSignal, ...]:
    """Return the public semantic pull surface for text-only fallback captures.

    This is a substrate-owned fallback for environments without a real
    residual feature surface. It publishes the same semantic pull names as
    the open-weight path, so downstream cognition consumes one contract.
    """

    embedding = _semantic_embedding(source_text)
    similarities = {
        name: _cosine_similarity(embedding, prototype)
        for name, prototype in _SEMANTIC_PROTOTYPES.items()
    }
    mean_similarity = sum(similarities.values()) / max(len(similarities), 1)
    centered = tuple(similarities[name] - mean_similarity for name in _SEMANTIC_PROTOTYPES)
    distribution = dict(zip(_SEMANTIC_PROTOTYPES, _softmax(centered), strict=True))
    pulls: dict[str, float] = {}
    for name in _SEMANTIC_PROTOTYPES:
        target = similarities[name]
        runner_up = max(
            value for other_name, value in similarities.items() if other_name != name
        )
        margin = _clamp_unit(0.5 + (target - runner_up) * 4.0)
        pulls[name] = _clamp_unit(distribution[name] * 0.78 + margin * 0.22)
    social_absolute = _clamp_unit(
        (_cosine_similarity(embedding, _SOCIAL_PROTOTYPE) + 1.0) / 2.0
    )
    social_pull = _clamp_unit(max(social_absolute - 0.50, 0.0) * 2.0)

    return (
        FeatureSignal(name="semantic_task_pull", values=(pulls["task"],), source=source),
        FeatureSignal(name="semantic_support_pull", values=(pulls["support"],), source=source),
        FeatureSignal(name="semantic_repair_pull", values=(pulls["repair"],), source=source),
        FeatureSignal(name="semantic_exploration_pull", values=(pulls["exploration"],), source=source),
        FeatureSignal(name="semantic_directive_pull", values=(pulls["directive"],), source=source),
        FeatureSignal(name="semantic_social_pull", values=(social_pull,), source=source),
        FeatureSignal(name="semantic_surface_active", values=(1.0,), source=source),
        FeatureSignal(name="fallback_active", values=(_clamp_unit(fallback_active),), source=source),
    )


class SemanticFeatureSurfaceSubstrateAdapter(FeatureSurfaceSubstrateAdapter):
    """Text-only semantic feature surface for fallback substrate captures."""

    def __init__(
        self,
        *,
        source_text: str,
        model_id: str = "semantic-feature-surface",
        fallback_active: float = 1.0,
    ) -> None:
        super().__init__(
            model_id=model_id,
            feature_surface=semantic_feature_surface_from_text(
                source_text,
                fallback_active=fallback_active,
            ),
            token_logits=(),
            is_frozen=True,
        )


class ResidualStreamSubstrateAdapter(SubstrateAdapter):
    """Future adapter for open-weight models with explicit residual-stream hooks."""

    def __init__(
        self,
        *,
        model_id: str,
        residual_activations: tuple[ResidualActivation, ...],
        residual_sequence: tuple[ResidualSequenceStep, ...] = (),
        token_logits: tuple[float, ...] = (),
        is_frozen: bool = True,
        feature_surface: tuple[FeatureSignal, ...] = (),
    ) -> None:
        self.model_id = model_id
        self.is_frozen = is_frozen
        self.surface_kind = SurfaceKind.RESIDUAL_STREAM
        self._residual_activations = residual_activations
        self._residual_sequence = residual_sequence
        self._token_logits = token_logits
        self._feature_surface = feature_surface

    async def capture(self, *, source_text: str | None = None) -> SubstrateSnapshot:
        residual_sequence = self._residual_sequence
        if not residual_sequence and self._residual_activations:
            residual_sequence = (
                ResidualSequenceStep(
                    step=max((activation.step for activation in self._residual_activations), default=0),
                    token=source_text or "<runtime-step>",
                    feature_surface=self._feature_surface,
                    residual_activations=self._residual_activations,
                    description="Single-step residual fallback sequence.",
                ),
            )
        return SubstrateSnapshot(
            model_id=self.model_id,
            is_frozen=self.is_frozen,
            surface_kind=self.surface_kind,
            token_logits=self._token_logits,
            feature_surface=self._feature_surface,
            residual_activations=self._residual_activations,
            residual_sequence=residual_sequence,
            unavailable_fields=(),
            description="Residual-stream substrate snapshot.",
        )


class OpenWeightResidualStreamSubstrateAdapter(SubstrateAdapter):
    """Hook-ready adapter backed by a frozen open-weight residual runtime."""

    def __init__(self, *, runtime: "OpenWeightResidualRuntime", default_source_text: str | None = None) -> None:
        self._runtime = runtime
        self._default_source_text = default_source_text
        self.model_id = runtime.model_id
        self.is_frozen = runtime.is_frozen
        self.surface_kind = SurfaceKind.RESIDUAL_STREAM

    async def capture(self, *, source_text: str | None = None) -> SubstrateSnapshot:
        effective_source_text = source_text or self._default_source_text
        if effective_source_text is None:
            raise ValueError("OpenWeightResidualStreamSubstrateAdapter requires source_text.")
        capture = self._runtime.capture(source_text=effective_source_text)
        runtime_origin = getattr(self._runtime, "runtime_origin", "unknown")
        fallback_active = 1 if getattr(self._runtime, "fallback_active", False) else 0
        capture_source = getattr(self._runtime, "capture_source", "unknown")
        description = (
            f"{capture.description} runtime_origin={runtime_origin} "
            f"capture_source={capture_source} fallback_active={fallback_active} "
            f"residual_sequence_len={len(capture.residual_sequence)}."
        )
        return SubstrateSnapshot(
            model_id=self.model_id,
            is_frozen=self.is_frozen,
            surface_kind=self.surface_kind,
            token_logits=capture.token_logits,
            feature_surface=capture.feature_surface,
            residual_activations=capture.residual_activations,
            residual_sequence=capture.residual_sequence,
            unavailable_fields=(),
            description=description,
        )


class SubstrateModule(RuntimeModule[SubstrateSnapshot]):
    slot_name = "substrate"
    owner = "SubstrateModule"
    value_type = SubstrateSnapshot
    dependencies = ()
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        adapter: SubstrateAdapter,
        source_text: str | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._adapter = adapter
        self._source_text = source_text

    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[SubstrateSnapshot]:
        return self.publish(await self._adapter.capture(source_text=self._source_text))

    async def process_standalone(self, **kwargs: Any) -> Snapshot[SubstrateSnapshot]:
        source_text = kwargs.get("source_text", self._source_text)
        return self.publish(await self._adapter.capture(source_text=source_text))
