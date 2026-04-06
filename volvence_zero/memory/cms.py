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


@dataclass(frozen=True)
class CMSState:
    online_fast: CMSBandState
    session_medium: CMSBandState
    background_slow: CMSBandState
    description: str


class CMSMemoryCore:
    """Minimal multi-timescale memory core backing the explicit memory owner."""

    def __init__(self, *, dim: int = 3) -> None:
        self._dim = dim
        self._online_fast = tuple(0.0 for _ in range(dim))
        self._session_medium = tuple(0.0 for _ in range(dim))
        self._background_slow = tuple(0.0 for _ in range(dim))
        self._last_update_ms = 0

    def observe_substrate(self, *, substrate_snapshot: SubstrateSnapshot | None, timestamp_ms: int) -> None:
        signal = self._signal_from_substrate(substrate_snapshot)
        self._online_fast = self._blend(self._online_fast, signal, rate=0.65)
        self._session_medium = self._blend(self._session_medium, signal, rate=0.3)
        self._background_slow = self._blend(self._background_slow, signal, rate=0.1)
        self._last_update_ms = timestamp_ms

    def reflect_lessons(self, *, lesson_count: int, timestamp_ms: int) -> None:
        lesson_signal = tuple(_clamp(lesson_count / (index + 3)) for index in range(self._dim))
        self._session_medium = self._blend(self._session_medium, lesson_signal, rate=0.25)
        self._background_slow = self._blend(self._background_slow, lesson_signal, rate=0.2)
        self._last_update_ms = timestamp_ms

    def export_state(self) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], int]:
        return (
            self._online_fast,
            self._session_medium,
            self._background_slow,
            self._last_update_ms,
        )

    def restore_state(
        self,
        state: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], int],
    ) -> None:
        self._online_fast, self._session_medium, self._background_slow, self._last_update_ms = state

    def snapshot(self) -> CMSState:
        return CMSState(
            online_fast=CMSBandState(
                name="online-fast",
                vector=self._online_fast,
                last_update_ms=self._last_update_ms,
            ),
            session_medium=CMSBandState(
                name="session-medium",
                vector=self._session_medium,
                last_update_ms=self._last_update_ms,
            ),
            background_slow=CMSBandState(
                name="background-slow",
                vector=self._background_slow,
                last_update_ms=self._last_update_ms,
            ),
            description=(
                "CMS memory core with online-fast, session-medium, and background-slow bands."
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
