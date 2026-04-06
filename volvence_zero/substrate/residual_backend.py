from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from volvence_zero.substrate.adapter import (
    FeatureSignal,
    ResidualActivation,
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
