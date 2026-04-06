from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.substrate import SubstrateSnapshot


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class CMSBandState:
    name: str
    vector: tuple[float, ...]
    last_update_ms: int
    cadence_interval: int
    observations_since_update: int
    pending_signal: tuple[float, ...]
    learning_rate: float = 0.0
    momentum: tuple[float, ...] = ()
    anti_forgetting_strength: float = 0.0


@dataclass(frozen=True)
class CMSState:
    online_fast: CMSBandState
    session_medium: CMSBandState
    background_slow: CMSBandState
    total_observations: int
    total_reflections: int
    description: str


@dataclass(frozen=True)
class CMSCheckpointState:
    online_fast: tuple[float, ...]
    session_medium: tuple[float, ...]
    background_slow: tuple[float, ...]
    last_update_ms: int
    total_observations: int
    total_reflections: int
    session_observations_since_update: int
    background_observations_since_update: int
    session_pending_signal: tuple[float, ...]
    background_pending_signal: tuple[float, ...]


class CMSMemoryCore:
    """Multi-timescale memory core with gradient-style updates.

    Each band has its own learning rate and momentum. Anti-forgetting
    backflow prevents catastrophic overwriting of slow bands.
    """

    def __init__(
        self,
        *,
        dim: int = 3,
        session_cadence: int = 2,
        background_cadence: int = 4,
        online_lr: float = 0.65,
        session_lr: float = 0.3,
        background_lr: float = 0.1,
        momentum_beta: float = 0.9,
        anti_forgetting: float = 0.1,
    ) -> None:
        self._dim = dim
        self._session_cadence = max(session_cadence, 1)
        self._background_cadence = max(background_cadence, 1)
        self._online_lr = online_lr
        self._session_lr = session_lr
        self._background_lr = background_lr
        self._momentum_beta = momentum_beta
        self._anti_forgetting = anti_forgetting
        self._online_fast = tuple(0.0 for _ in range(dim))
        self._session_medium = tuple(0.0 for _ in range(dim))
        self._background_slow = tuple(0.0 for _ in range(dim))
        self._online_momentum = tuple(0.0 for _ in range(dim))
        self._session_momentum = tuple(0.0 for _ in range(dim))
        self._background_momentum = tuple(0.0 for _ in range(dim))
        self._session_observations_since_update = 0
        self._background_observations_since_update = 0
        self._session_pending_signal = tuple(0.0 for _ in range(dim))
        self._background_pending_signal = tuple(0.0 for _ in range(dim))
        self._total_observations = 0
        self._total_reflections = 0
        self._last_update_ms = 0

    @property
    def dim(self) -> int:
        return self._dim

    def observe_substrate(self, *, substrate_snapshot: SubstrateSnapshot | None, timestamp_ms: int) -> None:
        signal = self._signal_from_substrate(substrate_snapshot)
        self._online_fast, self._online_momentum = self._gradient_update(
            current=self._online_fast,
            target=signal,
            momentum=self._online_momentum,
            lr=self._online_lr,
        )
        self._total_observations += 1
        (
            self._session_medium,
            self._session_pending_signal,
            self._session_observations_since_update,
            self._session_momentum,
        ) = self._integrate_signal_gradient(
            current_vector=self._session_medium,
            pending_signal=self._session_pending_signal,
            observations_since_update=self._session_observations_since_update,
            momentum=self._session_momentum,
            signal=signal,
            lr=self._session_lr,
            cadence_interval=self._session_cadence,
        )
        (
            self._background_slow,
            self._background_pending_signal,
            self._background_observations_since_update,
            self._background_momentum,
        ) = self._integrate_signal_gradient(
            current_vector=self._background_slow,
            pending_signal=self._background_pending_signal,
            observations_since_update=self._background_observations_since_update,
            momentum=self._background_momentum,
            signal=signal,
            lr=self._background_lr,
            cadence_interval=self._background_cadence,
        )
        if self._anti_forgetting > 0:
            self._apply_anti_forgetting()
        self._last_update_ms = timestamp_ms

    def reflect_lessons(self, *, lesson_count: int, timestamp_ms: int) -> None:
        lesson_signal = tuple(_clamp(lesson_count / (index + 3)) for index in range(self._dim))
        self._total_reflections += 1
        self._session_medium, self._session_momentum = self._gradient_update(
            current=self._session_medium,
            target=lesson_signal,
            momentum=self._session_momentum,
            lr=self._session_lr * 0.83,
        )
        (
            self._background_slow,
            self._background_pending_signal,
            self._background_observations_since_update,
            self._background_momentum,
        ) = self._integrate_signal_gradient(
            current_vector=self._background_slow,
            pending_signal=self._background_pending_signal,
            observations_since_update=self._background_observations_since_update,
            momentum=self._background_momentum,
            signal=lesson_signal,
            lr=self._background_lr * 2.0,
            cadence_interval=max(self._background_cadence - 1, 1),
        )
        self._last_update_ms = timestamp_ms

    def observe_encoder_feedback(
        self,
        *,
        encoder_signal: tuple[float, ...],
        timestamp_ms: int,
    ) -> None:
        """Accept metacontroller encoder output as an additional observation."""
        if len(encoder_signal) != self._dim:
            projected = tuple(encoder_signal[i % len(encoder_signal)] for i in range(self._dim)) if encoder_signal else tuple(0.0 for _ in range(self._dim))
        else:
            projected = encoder_signal
        self._online_fast, self._online_momentum = self._gradient_update(
            current=self._online_fast,
            target=projected,
            momentum=self._online_momentum,
            lr=self._online_lr * 0.3,
        )
        (
            self._session_medium,
            self._session_pending_signal,
            self._session_observations_since_update,
            self._session_momentum,
        ) = self._integrate_signal_gradient(
            current_vector=self._session_medium,
            pending_signal=self._session_pending_signal,
            observations_since_update=self._session_observations_since_update,
            momentum=self._session_momentum,
            signal=projected,
            lr=self._session_lr * 0.33,
            cadence_interval=self._session_cadence,
        )
        self._last_update_ms = timestamp_ms

    def export_state(self) -> CMSCheckpointState:
        return CMSCheckpointState(
            online_fast=self._online_fast,
            session_medium=self._session_medium,
            background_slow=self._background_slow,
            last_update_ms=self._last_update_ms,
            total_observations=self._total_observations,
            total_reflections=self._total_reflections,
            session_observations_since_update=self._session_observations_since_update,
            background_observations_since_update=self._background_observations_since_update,
            session_pending_signal=self._session_pending_signal,
            background_pending_signal=self._background_pending_signal,
        )

    def restore_state(
        self,
        state: CMSCheckpointState,
    ) -> None:
        self._online_fast = state.online_fast
        self._session_medium = state.session_medium
        self._background_slow = state.background_slow
        self._last_update_ms = state.last_update_ms
        self._total_observations = state.total_observations
        self._total_reflections = state.total_reflections
        self._session_observations_since_update = state.session_observations_since_update
        self._background_observations_since_update = state.background_observations_since_update
        self._session_pending_signal = state.session_pending_signal
        self._background_pending_signal = state.background_pending_signal

    def snapshot(self) -> CMSState:
        return CMSState(
            online_fast=CMSBandState(
                name="online-fast",
                vector=self._online_fast,
                last_update_ms=self._last_update_ms,
                cadence_interval=1,
                observations_since_update=0,
                pending_signal=tuple(0.0 for _ in range(self._dim)),
                learning_rate=self._online_lr,
                momentum=self._online_momentum,
                anti_forgetting_strength=self._anti_forgetting,
            ),
            session_medium=CMSBandState(
                name="session-medium",
                vector=self._session_medium,
                last_update_ms=self._last_update_ms,
                cadence_interval=self._session_cadence,
                observations_since_update=self._session_observations_since_update,
                pending_signal=self._session_pending_signal,
                learning_rate=self._session_lr,
                momentum=self._session_momentum,
                anti_forgetting_strength=self._anti_forgetting,
            ),
            background_slow=CMSBandState(
                name="background-slow",
                vector=self._background_slow,
                last_update_ms=self._last_update_ms,
                cadence_interval=self._background_cadence,
                observations_since_update=self._background_observations_since_update,
                pending_signal=self._background_pending_signal,
                learning_rate=self._background_lr,
                momentum=self._background_momentum,
                anti_forgetting_strength=self._anti_forgetting,
            ),
            total_observations=self._total_observations,
            total_reflections=self._total_reflections,
            description=(
                f"CMS core dim={self._dim} with gradient updates, "
                f"online_lr={self._online_lr}, session_lr={self._session_lr}, "
                f"bg_lr={self._background_lr}, anti_forgetting={self._anti_forgetting}."
            ),
        )

    def _signal_from_substrate(self, substrate_snapshot: SubstrateSnapshot | None) -> tuple[float, ...]:
        if substrate_snapshot is None:
            return tuple(0.0 for _ in range(self._dim))
        if substrate_snapshot.residual_activations:
            values = [
                sum(activation.activation) / len(activation.activation)
                for activation in substrate_snapshot.residual_activations
                if activation.activation
            ]
        else:
            values = [
                sum(feature.values) / len(feature.values)
                for feature in substrate_snapshot.feature_surface
                if feature.values
            ]
        if not values:
            return tuple(0.0 for _ in range(self._dim))
        base = tuple(values[index % len(values)] for index in range(self._dim))
        return tuple(_clamp(value) for value in base)

    def _blend(
        self,
        previous: tuple[float, ...],
        current: tuple[float, ...],
        *,
        rate: float,
    ) -> tuple[float, ...]:
        return tuple(
            _clamp(previous_value * (1.0 - rate) + current_value * rate)
            for previous_value, current_value in zip(previous, current)
        )

    def _gradient_update(
        self,
        *,
        current: tuple[float, ...],
        target: tuple[float, ...],
        momentum: tuple[float, ...],
        lr: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Gradient-style update: compute error → update momentum → apply."""
        error = tuple(target[i] - current[i] for i in range(self._dim))
        new_momentum = tuple(
            self._momentum_beta * momentum[i] + (1.0 - self._momentum_beta) * error[i]
            for i in range(self._dim)
        )
        updated = tuple(
            _clamp(current[i] + lr * new_momentum[i])
            for i in range(self._dim)
        )
        return updated, new_momentum

    def _apply_anti_forgetting(self) -> None:
        """Backflow from slow to fast: prevent fast bands from drifting
        too far from consolidated slow knowledge."""
        strength = self._anti_forgetting
        self._online_fast = tuple(
            _clamp(
                self._online_fast[i]
                + strength * (self._background_slow[i] - self._online_fast[i]) * 0.1
            )
            for i in range(self._dim)
        )
        self._session_medium = tuple(
            _clamp(
                self._session_medium[i]
                + strength * (self._background_slow[i] - self._session_medium[i]) * 0.05
            )
            for i in range(self._dim)
        )

    def _integrate_signal_gradient(
        self,
        *,
        current_vector: tuple[float, ...],
        pending_signal: tuple[float, ...],
        observations_since_update: int,
        momentum: tuple[float, ...],
        signal: tuple[float, ...],
        lr: float,
        cadence_interval: int,
    ) -> tuple[tuple[float, ...], tuple[float, ...], int, tuple[float, ...]]:
        """Cadence-gated gradient update for medium/slow bands."""
        next_count = observations_since_update + 1
        next_pending = tuple(
            _clamp((pending_signal[index] * observations_since_update + signal[index]) / next_count)
            for index in range(self._dim)
        )
        if next_count < cadence_interval:
            return (current_vector, next_pending, next_count, momentum)
        updated, new_momentum = self._gradient_update(
            current=current_vector,
            target=next_pending,
            momentum=momentum,
            lr=lr,
        )
        return (updated, tuple(0.0 for _ in range(self._dim)), 0, new_momentum)
