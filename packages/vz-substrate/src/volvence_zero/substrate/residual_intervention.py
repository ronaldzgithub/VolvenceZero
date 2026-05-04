"""Residual-stream intervention backends.

Three concrete :class:`ResidualInterventionBackend` implementations:

* :class:`TraceResidualInterventionBackend` — records
  :class:`ResidualControlApplication` entries on a shared list for
  training-trace replay, without touching residuals in place.
* :class:`NoOpResidualInterventionBackend` — no-op fallback used when a
  concrete HF runtime is not available.
* :class:`OpenWeightResidualInterventionBackend` — forwards to the
  live runtime hook set to modify residuals on the HF forward pass.

``apply_residual_control`` is the tiny dispatch helper that runtimes
call to route a decoder control vector through the chosen backend.

Slice S.3 (2026-05-04): extracted from the previous monolithic
``residual_backend.py``.
"""

from __future__ import annotations

from typing import Any, Sequence

from volvence_zero.substrate.adapter import (
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateSnapshot,
)

from volvence_zero.substrate.residual_contracts import ResidualControlApplication
from volvence_zero.substrate.residual_helpers import (
    _clamp_signed,
    _clamp_unit,
    _summarize_activations,
)
from volvence_zero.substrate.residual_interfaces import (
    OpenWeightResidualRuntime,
    ResidualInterventionBackend,
)


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


