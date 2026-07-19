"""``AntSubstrateAdapter`` — publishes the frozen sensorimotor surface as a
public :class:`SubstrateSnapshot`.

The adapter is bound to a mutable :class:`AntSenseHolder` that the
:class:`~volvence_ant.runtime.ant_session.AntSession` refreshes with the live
world observation + navigator state before each kernel turn. This is the ONLY
coupling point between the embodiment and the kernel; everything downstream
(temporal / memory / cognition) consumes the snapshot, exactly as it does for
the LLM substrate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from volvence_zero.substrate import (
    FeatureSignal,
    ResidualActivation,
    ResidualSequenceStep,
    SubstrateAdapter,
    SubstrateSnapshot,
    SurfaceKind,
)

from volvence_ant.env.ant_world import WorldObservation
from volvence_ant.substrate.navigator import NavigatorState
from volvence_ant.substrate.sense_encode import (
    SENSE_CHANNELS,
    sense_encode,
    sense_to_drives,
)

_MODEL_ID = "digital-ant-v0"


@dataclass
class AntSenseHolder:
    """Mutable holder the session refreshes each tick; adapters read it live."""

    observation: WorldObservation
    navigator_state: NavigatorState
    turn_command_scale: float
    step: int = 0

    def update(
        self,
        *,
        observation: WorldObservation,
        navigator_state: NavigatorState,
        step: int,
    ) -> None:
        self.observation = observation
        self.navigator_state = navigator_state
        self.step = step


class AntSubstrateAdapter(SubstrateAdapter):
    """Frozen non-language substrate adapter for the digital ant."""

    def __init__(self, holder: AntSenseHolder) -> None:
        self._holder = holder
        self.model_id = _MODEL_ID
        self.is_frozen = True
        self.surface_kind = SurfaceKind.RESIDUAL_STREAM

    async def capture(self, *, source_text: str | None = None) -> SubstrateSnapshot:
        holder = self._holder
        vector = sense_encode(
            holder.observation,
            holder.navigator_state,
            turn_command_scale=holder.turn_command_scale,
        )
        drives = sense_to_drives(holder.observation, holder.navigator_state)
        feature_surface = self._build_feature_surface(vector, drives)
        activation = ResidualActivation(
            layer_index=0,
            activation=tuple(float(v) for v in vector),
            step=holder.step,
        )
        sequence = (
            ResidualSequenceStep(
                step=holder.step,
                token="<ant-sense>",
                feature_surface=feature_surface,
                residual_activations=(activation,),
                description="digital-ant sensorimotor residual step",
            ),
        )
        return SubstrateSnapshot(
            model_id=self.model_id,
            is_frozen=self.is_frozen,
            surface_kind=self.surface_kind,
            token_logits=(),
            feature_surface=feature_surface,
            residual_activations=(activation,),
            residual_sequence=sequence,
            unavailable_fields=(),
            description=(
                f"digital-ant substrate step={holder.step} "
                f"carrying={holder.observation.carrying_food} "
                f"alarm={holder.observation.alarm:.2f}"
            ),
        )

    @staticmethod
    def _build_feature_surface(
        vector: np.ndarray, drives: "object"
    ) -> tuple[FeatureSignal, ...]:
        src = _MODEL_ID
        signals: list[FeatureSignal] = [
            # Generic drive names consumed by the kernel PE owner unchanged.
            FeatureSignal(name="semantic_task_pull", values=(drives.forage_pull,), source=src),
            FeatureSignal(name="semantic_support_pull", values=(drives.homing_pull,), source=src),
            FeatureSignal(name="semantic_repair_pull", values=(drives.alarm_pull,), source=src),
            FeatureSignal(
                name="semantic_exploration_pull", values=(drives.explore_pull,), source=src
            ),
            FeatureSignal(name="semantic_directive_pull", values=(drives.commit_pull,), source=src),
            # Markers the residual consumers expect.
            FeatureSignal(name="semantic_surface_active", values=(1.0,), source=src),
            FeatureSignal(name="fallback_active", values=(0.0,), source=src),
            FeatureSignal(name="hook_layer_coverage", values=(1.0,), source=src),
            FeatureSignal(name="semantic_residual_weight", values=(1.0,), source=src),
        ]
        # Embodiment-native channels (used by our own metrics / actuator).
        for name, value in zip(SENSE_CHANNELS, vector, strict=True):
            signals.append(
                FeatureSignal(name=f"ant_{name}", values=(float(value),), source=src)
            )
        return tuple(signals)
