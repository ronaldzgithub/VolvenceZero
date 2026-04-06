from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
from typing import Iterable

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
    """Import-guarded runtime placeholder for a true HF hook implementation."""

    def __init__(self, *, model_id: str, device: str = "cpu") -> None:
        importlib.import_module("torch")
        importlib.import_module("transformers")
        self.model_id = model_id
        self.is_frozen = True
        self._device = device

    def capture(self, *, source_text: str) -> OpenWeightRuntimeCapture:
        raise NotImplementedError(
            f"Transformers runtime '{self.model_id}' is import-ready on device={self._device}, "
            "but forward hook capture is not wired yet."
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
        del substrate_snapshot
        del applied_control
        del track_scale
        raise NotImplementedError(
            f"Transformers runtime '{self.model_id}' is import-ready on device={self._device}, "
            "but residual intervention hooks are not wired yet."
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
