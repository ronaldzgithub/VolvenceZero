from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel


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
            unavailable_fields=(
                UnavailableField(
                    field_name="residual_activations",
                    reason="feature-surface-adapter",
                    detail="Current substrate adapter does not expose residual-stream activations.",
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
        token_logits: tuple[float, ...] = (),
        is_frozen: bool = True,
        feature_surface: tuple[FeatureSignal, ...] = (),
    ) -> None:
        self.model_id = model_id
        self.is_frozen = is_frozen
        self.surface_kind = SurfaceKind.RESIDUAL_STREAM
        self._residual_activations = residual_activations
        self._token_logits = token_logits
        self._feature_surface = feature_surface

    async def capture(self, *, source_text: str | None = None) -> SubstrateSnapshot:
        return SubstrateSnapshot(
            model_id=self.model_id,
            is_frozen=self.is_frozen,
            surface_kind=self.surface_kind,
            token_logits=self._token_logits,
            feature_surface=self._feature_surface,
            residual_activations=self._residual_activations,
            unavailable_fields=(),
            description="Residual-stream substrate snapshot.",
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
