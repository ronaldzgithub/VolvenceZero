from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping

from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel
from volvence_zero.substrate import FeatureSignal, SubstrateSnapshot


class TemporalImplementationMode(str, Enum):
    PLACEHOLDER = "placeholder"
    HEURISTIC = "heuristic"
    LEARNED_LITE = "learned-lite"
    FULL_LEARNED = "full-learned"


@dataclass(frozen=True)
class ControllerState:
    code: tuple[float, ...]
    code_dim: int
    switch_gate: float
    is_switching: bool
    steps_since_switch: int


@dataclass(frozen=True)
class TemporalAbstractionSnapshot:
    controller_state: ControllerState
    active_abstract_action: str
    controller_params_hash: str
    description: str


@dataclass(frozen=True)
class TemporalStep:
    controller_state: ControllerState
    active_abstract_action: str
    controller_params_hash: str
    description: str


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _hash_payload(payload: object) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _feature_signature(feature_surface: tuple[FeatureSignal, ...]) -> tuple[str, ...]:
    return tuple(feature.name for feature in feature_surface[:4])


def _code_from_feature_surface(feature_surface: tuple[FeatureSignal, ...]) -> tuple[float, ...]:
    if not feature_surface:
        return (0.0, 0.0, 0.0)
    magnitudes = [sum(feature.values) / len(feature.values) for feature in feature_surface if feature.values]
    if not magnitudes:
        return (0.0, 0.0, 0.0)
    average = sum(magnitudes) / len(magnitudes)
    maximum = max(magnitudes)
    spread = maximum - min(magnitudes)
    return (_clamp(average), _clamp(maximum), _clamp(spread))


def _abstract_action_from_code(code: tuple[float, ...], switch_gate: float) -> str:
    average, maximum, spread = code
    if switch_gate > 0.7:
        return "refresh-controller-context"
    if spread > 0.35:
        return "focus-dominant-signal"
    if average < 0.2 and maximum < 0.25:
        return "hold-low-signal-context"
    return "stabilize-current-controller"


class TemporalPolicy(ABC):
    """Common interface for placeholder, heuristic, and future learned policies."""

    mode: TemporalImplementationMode

    @abstractmethod
    def step(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        previous_snapshot: TemporalAbstractionSnapshot | None,
    ) -> TemporalStep:
        """Produce the next temporal abstraction state."""


class PlaceholderTemporalPolicy(TemporalPolicy):
    mode = TemporalImplementationMode.PLACEHOLDER

    def step(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        previous_snapshot: TemporalAbstractionSnapshot | None,
    ) -> TemporalStep:
        steps_since_switch = (
            previous_snapshot.controller_state.steps_since_switch + 1
            if previous_snapshot is not None
            else 0
        )
        controller_state = ControllerState(
            code=(0.0, 0.0, 0.0),
            code_dim=3,
            switch_gate=0.0,
            is_switching=False,
            steps_since_switch=steps_since_switch,
        )
        params_hash = _hash_payload(
            {
                "mode": self.mode.value,
                "model_id": substrate_snapshot.model_id,
            }
        )
        return TemporalStep(
            controller_state=controller_state,
            active_abstract_action="placeholder-controller",
            controller_params_hash=params_hash,
            description="Placeholder temporal controller with no active switching.",
        )


class HeuristicTemporalPolicy(TemporalPolicy):
    mode = TemporalImplementationMode.HEURISTIC

    def __init__(self) -> None:
        self._previous_feature_signature: tuple[str, ...] = ()

    def step(
        self,
        *,
        substrate_snapshot: SubstrateSnapshot,
        previous_snapshot: TemporalAbstractionSnapshot | None,
    ) -> TemporalStep:
        feature_signature = _feature_signature(substrate_snapshot.feature_surface)
        code = _code_from_feature_surface(substrate_snapshot.feature_surface)
        previous_steps = 0
        if previous_snapshot is not None:
            previous_steps = previous_snapshot.controller_state.steps_since_switch

        signature_changed = feature_signature != self._previous_feature_signature
        switch_gate = 0.85 if signature_changed and feature_signature else 0.15
        is_switching = switch_gate > 0.7
        steps_since_switch = 0 if is_switching else previous_steps + 1
        active_action = _abstract_action_from_code(code, switch_gate)
        signature_suffix = "|".join(feature_signature) if feature_signature else "no-feature-signal"
        controller_state = ControllerState(
            code=code,
            code_dim=len(code),
            switch_gate=switch_gate,
            is_switching=is_switching,
            steps_since_switch=steps_since_switch,
        )
        params_hash = _hash_payload(
            {
                "mode": self.mode.value,
                "feature_signature": feature_signature,
                "code": code,
            }
        )
        step = TemporalStep(
            controller_state=controller_state,
            active_abstract_action=f"{active_action}:{signature_suffix}",
            controller_params_hash=params_hash,
            description=(
                f"Heuristic temporal controller mode={self.mode.value}, "
                f"switch_gate={switch_gate:.2f}, feature_signature={signature_suffix}."
            ),
        )
        self._previous_feature_signature = feature_signature
        return step


class TemporalModule(RuntimeModule[TemporalAbstractionSnapshot]):
    slot_name = "temporal_abstraction"
    owner = "TemporalModule"
    value_type = TemporalAbstractionSnapshot
    dependencies = ("substrate",)
    default_wiring_level = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        policy: TemporalPolicy | None = None,
        wiring_level: WiringLevel | None = None,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        self._policy = policy or HeuristicTemporalPolicy()
        self._previous_snapshot: TemporalAbstractionSnapshot | None = None

    @property
    def policy(self) -> TemporalPolicy:
        return self._policy

    async def process(
        self,
        upstream: Mapping[str, Snapshot[Any]],
    ) -> Snapshot[TemporalAbstractionSnapshot]:
        substrate_snapshot = upstream["substrate"]
        substrate_value = substrate_snapshot.value
        if not isinstance(substrate_value, SubstrateSnapshot):
            step = PlaceholderTemporalPolicy().step(
                substrate_snapshot=SubstrateSnapshot(
                    model_id="runtime-placeholder",
                    is_frozen=True,
                    surface_kind=getattr(substrate_value, "surface_kind", "placeholder"),
                    token_logits=(),
                    feature_surface=(),
                    residual_activations=(),
                    unavailable_fields=(),
                    description="Runtime placeholder substrate value.",
                ),
                previous_snapshot=self._previous_snapshot,
            )
        else:
            step = self._policy.step(
                substrate_snapshot=substrate_value,
                previous_snapshot=self._previous_snapshot,
            )

        snapshot_value = TemporalAbstractionSnapshot(
            controller_state=step.controller_state,
            active_abstract_action=step.active_abstract_action,
            controller_params_hash=step.controller_params_hash,
            description=step.description,
        )
        self._previous_snapshot = snapshot_value
        return self.publish(snapshot_value)

    async def process_standalone(self, **kwargs: Any) -> Snapshot[TemporalAbstractionSnapshot]:
        substrate_snapshot = kwargs.get("substrate_snapshot")
        if not isinstance(substrate_snapshot, SubstrateSnapshot):
            raise TypeError("substrate_snapshot must be a SubstrateSnapshot.")
        step = self._policy.step(
            substrate_snapshot=substrate_snapshot,
            previous_snapshot=self._previous_snapshot,
        )
        snapshot_value = TemporalAbstractionSnapshot(
            controller_state=step.controller_state,
            active_abstract_action=step.active_abstract_action,
            controller_params_hash=step.controller_params_hash,
            description=step.description,
        )
        self._previous_snapshot = snapshot_value
        return self.publish(snapshot_value)
