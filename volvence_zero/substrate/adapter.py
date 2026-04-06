from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
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
        return SubstrateSnapshot(
            model_id=self.model_id,
            is_frozen=self.is_frozen,
            surface_kind=self.surface_kind,
            token_logits=capture.token_logits,
            feature_surface=capture.feature_surface,
            residual_activations=capture.residual_activations,
            residual_sequence=capture.residual_sequence,
            unavailable_fields=(),
            description=capture.description,
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
