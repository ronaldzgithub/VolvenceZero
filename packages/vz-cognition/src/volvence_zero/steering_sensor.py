"""Unique runtime owner for frozen residual steering-condition beliefs."""

from __future__ import annotations

import math
from typing import Mapping

from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.steering_contracts import (
    STEERING_CONDITION_BELIEF_SLOT,
    SteeringConditionBelief,
    SteeringReaderArtifact,
)
from volvence_zero.substrate import SubstrateSnapshot


def _normalized_margin(scores: tuple[float, ...]) -> float:
    ordered = sorted(scores, reverse=True)
    gap = max(ordered[0] - ordered[1], 0.0)
    return min(1.0, max(0.0, 1.0 - math.exp(-gap)))


def _normalized_entropy(probabilities: tuple[float, ...]) -> float:
    if len(probabilities) < 2:
        return 0.0
    total = sum(max(value, 0.0) for value in probabilities)
    if total <= 0.0:
        return 0.0
    normalized = tuple(max(value, 0.0) / total for value in probabilities)
    entropy = -sum(value * math.log(value) for value in normalized if value > 0.0)
    return min(1.0, max(0.0, entropy / math.log(len(normalized))))


class SteeringSensorModule(RuntimeModule[SteeringConditionBelief]):
    """Interpret one substrate-owned residual with one frozen ridge reader."""

    slot_name = STEERING_CONDITION_BELIEF_SLOT
    owner = "SteeringSensorModule"
    value_type = SteeringConditionBelief
    dependencies = ("substrate",)
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        artifact: SteeringReaderArtifact,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._artifact = artifact
        self._previous_index: int | None = None
        self._previous_margin = 0.0

    @property
    def artifact(self) -> SteeringReaderArtifact:
        return self._artifact

    def reset_history(self) -> None:
        self._previous_index = None
        self._previous_margin = 0.0

    def _read(self, substrate: SubstrateSnapshot) -> SteeringConditionBelief:
        artifact = self._artifact
        if not substrate.is_frozen:
            raise ValueError("steering sensor requires a frozen substrate")
        if substrate.model_id != artifact.model_id:
            raise ValueError(
                "steering reader model mismatch: "
                f"{substrate.model_id!r} != {artifact.model_id!r}"
            )
        activations = tuple(
            activation
            for activation in substrate.residual_activations
            if activation.layer_index == artifact.layer_index
        )
        if len(activations) != 1:
            raise ValueError(
                "steering sensor requires exactly one residual activation at "
                f"layer {artifact.layer_index}, got {len(activations)}"
            )
        residual = activations[0].activation
        if len(residual) != artifact.residual_width:
            raise ValueError(
                "steering sensor residual width mismatch: "
                f"{len(residual)} != {artifact.residual_width}"
            )
        residual_norm = math.sqrt(sum(value * value for value in residual))
        if not math.isfinite(residual_norm) or residual_norm <= 0.0:
            raise ValueError("steering sensor residual norm must be positive")
        standardized = tuple(
            (value - artifact.feature_mean[index])
            / artifact.feature_scale[index]
            for index, value in enumerate(residual)
        )
        scores = tuple(
            sum(
                standardized[row] * artifact.weights[row][column]
                for row in range(artifact.residual_width)
            )
            for column in range(len(artifact.class_labels))
        )
        fresh_index = max(range(len(scores)), key=scores.__getitem__)
        fresh_margin = _normalized_margin(scores)
        belief_index = (
            fresh_index if self._previous_index is None else self._previous_index
        )
        belief_margin = (
            fresh_margin
            if self._previous_index is None
            else self._previous_margin
        )
        disagreement = belief_index != fresh_index
        staleness = (
            1.0
            if disagreement
            else min(1.0, max(0.0, 1.0 - min(belief_margin, fresh_margin)))
        )
        value = SteeringConditionBelief(
            belief_label=artifact.class_labels[belief_index],
            belief_index=belief_index,
            belief_margin=belief_margin,
            fresh_belief_label=artifact.class_labels[fresh_index],
            fresh_belief_index=fresh_index,
            fresh_margin=fresh_margin,
            belief_disagrees_fresh=disagreement,
            staleness_proxy=staleness,
            base_action_entropy=_normalized_entropy(substrate.token_logits),
            reader_artifact_id=artifact.artifact_id,
            source_model_id=substrate.model_id,
            source_layer_index=artifact.layer_index,
            source_residual_norm=residual_norm,
            description=(
                "Frozen residual reader belief; lagged belief and fresh read "
                "are both published for gate observability."
            ),
        )
        self._previous_index = fresh_index
        self._previous_margin = fresh_margin
        return value

    async def process(
        self,
        upstream: Mapping[str, Snapshot[object]],
    ) -> Snapshot[SteeringConditionBelief]:
        substrate_snapshot = upstream["substrate"]
        if not isinstance(substrate_snapshot.value, SubstrateSnapshot):
            raise TypeError("steering sensor requires a SubstrateSnapshot")
        return self.publish(self._read(substrate_snapshot.value))

    async def process_standalone(
        self,
        **kwargs: object,
    ) -> Snapshot[SteeringConditionBelief]:
        substrate = kwargs.get("substrate")
        if not isinstance(substrate, SubstrateSnapshot):
            raise TypeError("process_standalone requires substrate=SubstrateSnapshot")
        return self.publish(self._read(substrate))


__all__ = ("SteeringSensorModule",)
