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
    """Minimal multi-timescale memory core backing the explicit memory owner."""

    def __init__(
        self,
        *,
        dim: int = 3,
        session_cadence: int = 2,
        background_cadence: int = 4,
    ) -> None:
        self._dim = dim
        self._session_cadence = max(session_cadence, 1)
        self._background_cadence = max(background_cadence, 1)
        self._online_fast = tuple(0.0 for _ in range(dim))
        self._session_medium = tuple(0.0 for _ in range(dim))
        self._background_slow = tuple(0.0 for _ in range(dim))
        self._session_observations_since_update = 0
        self._background_observations_since_update = 0
        self._session_pending_signal = tuple(0.0 for _ in range(dim))
        self._background_pending_signal = tuple(0.0 for _ in range(dim))
        self._total_observations = 0
        self._total_reflections = 0
        self._last_update_ms = 0

    def observe_substrate(self, *, substrate_snapshot: SubstrateSnapshot | None, timestamp_ms: int) -> None:
        signal = self._signal_from_substrate(substrate_snapshot)
        self._online_fast = self._blend(self._online_fast, signal, rate=0.65)
        self._total_observations += 1
        (
            self._session_medium,
            self._session_pending_signal,
            self._session_observations_since_update,
        ) = self._integrate_signal(
            current_vector=self._session_medium,
            pending_signal=self._session_pending_signal,
            observations_since_update=self._session_observations_since_update,
            signal=signal,
            rate=0.3,
            cadence_interval=self._session_cadence,
        )
        (
            self._background_slow,
            self._background_pending_signal,
            self._background_observations_since_update,
        ) = self._integrate_signal(
            current_vector=self._background_slow,
            pending_signal=self._background_pending_signal,
            observations_since_update=self._background_observations_since_update,
            signal=signal,
            rate=0.1,
            cadence_interval=self._background_cadence,
        )
        self._last_update_ms = timestamp_ms

    def reflect_lessons(self, *, lesson_count: int, timestamp_ms: int) -> None:
        lesson_signal = tuple(_clamp(lesson_count / (index + 3)) for index in range(self._dim))
        self._total_reflections += 1
        self._session_medium = self._blend(self._session_medium, lesson_signal, rate=0.25)
        (
            self._background_slow,
            self._background_pending_signal,
            self._background_observations_since_update,
        ) = self._integrate_signal(
            current_vector=self._background_slow,
            pending_signal=self._background_pending_signal,
            observations_since_update=self._background_observations_since_update,
            signal=lesson_signal,
            rate=0.2,
            cadence_interval=max(self._background_cadence - 1, 1),
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
            ),
            session_medium=CMSBandState(
                name="session-medium",
                vector=self._session_medium,
                last_update_ms=self._last_update_ms,
                cadence_interval=self._session_cadence,
                observations_since_update=self._session_observations_since_update,
                pending_signal=self._session_pending_signal,
            ),
            background_slow=CMSBandState(
                name="background-slow",
                vector=self._background_slow,
                last_update_ms=self._last_update_ms,
                cadence_interval=self._background_cadence,
                observations_since_update=self._background_observations_since_update,
                pending_signal=self._background_pending_signal,
            ),
            total_observations=self._total_observations,
            total_reflections=self._total_reflections,
            description=(
                "CMS memory core with machine-readable online-fast, session-medium, "
                "and background-slow bands plus cadence gating."
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

    def _integrate_signal(
        self,
        *,
        current_vector: tuple[float, ...],
        pending_signal: tuple[float, ...],
        observations_since_update: int,
        signal: tuple[float, ...],
        rate: float,
        cadence_interval: int,
    ) -> tuple[tuple[float, ...], tuple[float, ...], int]:
        next_count = observations_since_update + 1
        next_pending = tuple(
            _clamp((pending_signal[index] * observations_since_update + signal[index]) / next_count)
            for index in range(self._dim)
        )
        if next_count < cadence_interval:
            return (current_vector, next_pending, next_count)
        next_vector = self._blend(current_vector, next_pending, rate=rate)
        return (next_vector, tuple(0.0 for _ in range(self._dim)), 0)
