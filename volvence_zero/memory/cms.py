from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from volvence_zero.substrate import SubstrateSnapshot


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _matvec(mat: list[float], vec: list[float] | tuple[float, ...], rows: int, cols: int) -> list[float]:
    return [
        sum(mat[i * cols + j] * vec[j] for j in range(cols))
        for i in range(rows)
    ]


def _init_weight(size: int, scale: float = 0.01) -> list[float]:
    return [
        scale * (((i * 2654435761 + 17) % 65537) / 32768.5 - 1.0)
        for i in range(size)
    ]


class CMSBandMLP:
    """2-layer residual MLP knowledge store: y = x + W1 @ tanh(W2 @ x).

    W1 initialized to zero so the MLP starts as an identity map.
    An internal state vector evolves toward targets (like vector mode)
    while the MLP weights provide additional nonlinear capacity.
    """

    def __init__(
        self,
        *,
        d_in: int = 16,
        d_hidden: int = 32,
        learning_rate: float = 0.1,
        momentum_beta: float = 0.9,
    ) -> None:
        self._d_in = d_in
        self._d_hidden = d_hidden
        self._lr = learning_rate
        self._momentum_beta = momentum_beta

        self._state = [0.0] * d_in
        self._state_momentum = [0.0] * d_in

        self._w2 = _init_weight(d_hidden * d_in, 0.01)
        self._w1 = [0.0] * (d_in * d_hidden)
        self._w2_momentum = [0.0] * (d_hidden * d_in)
        self._w1_momentum = [0.0] * (d_in * d_hidden)

    @property
    def d_in(self) -> int:
        return self._d_in

    @property
    def d_hidden(self) -> int:
        return self._d_hidden

    def forward(self, x: tuple[float, ...] | list[float]) -> tuple[float, ...]:
        h = _matvec(self._w2, x, self._d_hidden, self._d_in)
        a = [math.tanh(v) for v in h]
        residual = _matvec(self._w1, a, self._d_in, self._d_hidden)
        return tuple(_clamp(x[i] + residual[i]) for i in range(self._d_in))

    def representation_vector(self) -> tuple[float, ...]:
        return self.forward(self._state)

    def update(self, *, target: tuple[float, ...]) -> None:
        x = self._state
        d_in = self._d_in
        d_hidden = self._d_hidden
        beta = self._momentum_beta
        lr = self._lr

        h = _matvec(self._w2, x, d_hidden, d_in)
        a = [math.tanh(v) for v in h]
        residual = _matvec(self._w1, a, d_in, d_hidden)
        y = [x[i] + residual[i] for i in range(d_in)]

        dy = [y[i] - target[i] for i in range(d_in)]

        grad_w1 = [dy[i] * a[j] for i in range(d_in) for j in range(d_hidden)]

        grad_a = [
            sum(dy[i] * self._w1[i * d_hidden + j] for i in range(d_in))
            for j in range(d_hidden)
        ]
        grad_h = [grad_a[j] * (1.0 - a[j] * a[j]) for j in range(d_hidden)]

        grad_w2 = [grad_h[j] * x[k] for j in range(d_hidden) for k in range(d_in)]

        w1_len = d_in * d_hidden
        for i in range(w1_len):
            self._w1_momentum[i] = beta * self._w1_momentum[i] + (1.0 - beta) * grad_w1[i]
            self._w1[i] -= lr * self._w1_momentum[i]

        w2_len = d_hidden * d_in
        for i in range(w2_len):
            self._w2_momentum[i] = beta * self._w2_momentum[i] + (1.0 - beta) * grad_w2[i]
            self._w2[i] -= lr * self._w2_momentum[i]

        for i in range(d_in):
            error = target[i] - x[i]
            self._state_momentum[i] = beta * self._state_momentum[i] + (1.0 - beta) * error
            self._state[i] = _clamp(x[i] + lr * self._state_momentum[i])

    def export_params(self) -> tuple[tuple[float, ...], ...]:
        return (
            tuple(self._state),
            tuple(self._state_momentum),
            tuple(self._w2),
            tuple(self._w1),
            tuple(self._w2_momentum),
            tuple(self._w1_momentum),
        )

    def load_representation(self, vector: tuple[float, ...]) -> None:
        projected = tuple(
            vector[i % len(vector)] if vector else 0.0
            for i in range(self._d_in)
        )
        self._state = [float(_clamp(value)) for value in projected]
        self._state_momentum = [0.0] * self._d_in

    def restore_params(self, params: tuple[tuple[float, ...], ...]) -> None:
        if len(params) != 6:
            raise ValueError(f"Expected 6 param groups, got {len(params)}")
        self._state = list(params[0])
        self._state_momentum = list(params[1])
        self._w2 = list(params[2])
        self._w1 = list(params[3])
        self._w2_momentum = list(params[4])
        self._w1_momentum = list(params[5])

    def parameter_count(self) -> int:
        return self._d_in + self._d_hidden * self._d_in + self._d_in * self._d_hidden

    def mix_from(self, other: CMSBandMLP, *, strength: float, factor: float) -> None:
        for i in range(self._d_in):
            self._state[i] = _clamp(
                self._state[i] + strength * (other._state[i] - self._state[i]) * factor
            )
        for i in range(len(self._w1)):
            self._w1[i] += strength * (other._w1[i] - self._w1[i]) * factor
        for i in range(len(self._w2)):
            self._w2[i] += strength * (other._w2[i] - self._w2[i]) * factor


class CMSVariant(str, Enum):
    SEQUENTIAL = "sequential"
    INDEPENDENT = "independent"


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
    mode: str = "vector"
    mlp_param_count: int = 0


@dataclass(frozen=True)
class CMSState:
    online_fast: CMSBandState
    session_medium: CMSBandState
    background_slow: CMSBandState
    total_observations: int
    total_reflections: int
    description: str
    variant: str = "sequential"


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
    mode: str = "vector"
    mlp_params: tuple[tuple[tuple[float, ...], ...], ...] = ()


class CMSMemoryCore:
    """Multi-timescale memory core with gradient-style updates.

    Each band has its own learning rate and momentum. Anti-forgetting
    backflow prevents catastrophic overwriting of slow bands.

    Supports two modes:
    - ``"vector"``: fixed-dim vector per band (original behavior)
    - ``"mlp"``: 2-layer residual MLP per band (higher capacity)
    """

    def __init__(
        self,
        *,
        dim: int = 3,
        mode: str = "vector",
        d_in: int = 16,
        d_hidden: int = 32,
        variant: str = "sequential",
        session_cadence: int = 2,
        background_cadence: int = 4,
        online_lr: float = 0.65,
        session_lr: float = 0.3,
        background_lr: float = 0.1,
        momentum_beta: float = 0.9,
        anti_forgetting: float = 0.1,
    ) -> None:
        if mode not in ("vector", "mlp"):
            raise ValueError(f"mode must be 'vector' or 'mlp', got {mode!r}")
        self._mode = mode
        self._variant = CMSVariant(variant)
        self._session_cadence = max(session_cadence, 1)
        self._background_cadence = max(background_cadence, 1)
        self._online_lr = online_lr
        self._session_lr = session_lr
        self._background_lr = background_lr
        self._momentum_beta = momentum_beta
        self._anti_forgetting = anti_forgetting
        self._total_observations = 0
        self._total_reflections = 0
        self._last_update_ms = 0
        self._session_observations_since_update = 0
        self._background_observations_since_update = 0

        if mode == "mlp":
            self._dim = d_in
            self._d_hidden = d_hidden
            self._online_mlp = CMSBandMLP(
                d_in=d_in, d_hidden=d_hidden,
                learning_rate=online_lr, momentum_beta=momentum_beta,
            )
            self._session_mlp = CMSBandMLP(
                d_in=d_in, d_hidden=d_hidden,
                learning_rate=session_lr, momentum_beta=momentum_beta,
            )
            self._background_mlp = CMSBandMLP(
                d_in=d_in, d_hidden=d_hidden,
                learning_rate=background_lr, momentum_beta=momentum_beta,
            )
            self._session_pending_signal = tuple(0.0 for _ in range(d_in))
            self._background_pending_signal = tuple(0.0 for _ in range(d_in))
        else:
            self._dim = dim
            self._d_hidden = 0
            self._online_fast = tuple(0.0 for _ in range(dim))
            self._session_medium = tuple(0.0 for _ in range(dim))
            self._background_slow = tuple(0.0 for _ in range(dim))
            self._online_momentum = tuple(0.0 for _ in range(dim))
            self._session_momentum = tuple(0.0 for _ in range(dim))
            self._background_momentum = tuple(0.0 for _ in range(dim))
            self._session_pending_signal = tuple(0.0 for _ in range(dim))
            self._background_pending_signal = tuple(0.0 for _ in range(dim))

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # observe_substrate
    # ------------------------------------------------------------------

    def observe_substrate(self, *, substrate_snapshot: SubstrateSnapshot | None, timestamp_ms: int) -> None:
        signal = self._signal_from_substrate(substrate_snapshot)
        self._total_observations += 1

        if self._mode == "mlp":
            self._online_mlp.update(target=signal)
            online_signal = self._online_mlp.representation_vector()
            if self._variant is CMSVariant.INDEPENDENT:
                session_signal = signal
                background_signal = signal
            else:
                session_signal = online_signal
                background_signal = self._session_mlp.representation_vector()

            self._session_pending_signal, self._session_observations_since_update = (
                self._integrate_signal_mlp(
                    mlp=self._session_mlp,
                    pending_signal=self._session_pending_signal,
                    observations_since_update=self._session_observations_since_update,
                    signal=session_signal,
                    cadence_interval=self._session_cadence,
                )
            )
            self._background_pending_signal, self._background_observations_since_update = (
                self._integrate_signal_mlp(
                    mlp=self._background_mlp,
                    pending_signal=self._background_pending_signal,
                    observations_since_update=self._background_observations_since_update,
                    signal=background_signal,
                    cadence_interval=self._background_cadence,
                )
            )

            if self._anti_forgetting > 0:
                self._apply_anti_forgetting_mlp()
        else:
            self._online_fast, self._online_momentum = self._gradient_update(
                current=self._online_fast,
                target=signal,
                momentum=self._online_momentum,
                lr=self._online_lr,
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

    # ------------------------------------------------------------------
    # reflect_lessons
    # ------------------------------------------------------------------

    def reflect_lessons(self, *, lesson_count: int, timestamp_ms: int) -> None:
        lesson_signal = tuple(_clamp(lesson_count / (index + 3)) for index in range(self._dim))
        self._total_reflections += 1

        if self._mode == "mlp":
            self._session_mlp.update(target=lesson_signal)
            background_signal = (
                lesson_signal
                if self._variant is CMSVariant.INDEPENDENT
                else self._session_mlp.representation_vector()
            )
            self._background_pending_signal, self._background_observations_since_update = (
                self._integrate_signal_mlp(
                    mlp=self._background_mlp,
                    pending_signal=self._background_pending_signal,
                    observations_since_update=self._background_observations_since_update,
                    signal=background_signal,
                    cadence_interval=max(self._background_cadence - 1, 1),
                )
            )
        else:
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

    # ------------------------------------------------------------------
    # observe_encoder_feedback
    # ------------------------------------------------------------------

    def observe_encoder_feedback(
        self,
        *,
        encoder_signal: tuple[float, ...],
        timestamp_ms: int,
    ) -> None:
        """Accept metacontroller encoder output as an additional observation."""
        if len(encoder_signal) != self._dim:
            projected: tuple[float, ...] = (
                tuple(encoder_signal[i % len(encoder_signal)] for i in range(self._dim))
                if encoder_signal
                else tuple(0.0 for _ in range(self._dim))
            )
        else:
            projected = encoder_signal

        if self._mode == "mlp":
            self._online_mlp.update(target=projected)
            session_signal = (
                projected
                if self._variant is CMSVariant.INDEPENDENT
                else self._online_mlp.representation_vector()
            )
            self._session_pending_signal, self._session_observations_since_update = (
                self._integrate_signal_mlp(
                    mlp=self._session_mlp,
                    pending_signal=self._session_pending_signal,
                    observations_since_update=self._session_observations_since_update,
                    signal=session_signal,
                    cadence_interval=self._session_cadence,
                )
            )
        else:
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

    # ------------------------------------------------------------------
    # observe_family_signal (MLP mode only)
    # ------------------------------------------------------------------

    def observe_family_signal(
        self,
        *,
        family_centroid: tuple[float, ...],
        family_stability: float,
        timestamp_ms: int,
    ) -> None:
        """Accept action-family observation to enrich session-medium band."""
        if self._mode != "mlp":
            return
        if len(family_centroid) != self._dim:
            projected = tuple(
                family_centroid[i % len(family_centroid)] if family_centroid else 0.0
                for i in range(self._dim)
            )
        else:
            projected = family_centroid
        weighted = tuple(_clamp(projected[i] * _clamp(family_stability)) for i in range(self._dim))
        self._session_pending_signal, self._session_observations_since_update = (
            self._integrate_signal_mlp(
                mlp=self._session_mlp,
                pending_signal=self._session_pending_signal,
                observations_since_update=self._session_observations_since_update,
                signal=weighted,
                cadence_interval=self._session_cadence,
            )
        )
        self._last_update_ms = timestamp_ms

    # ------------------------------------------------------------------
    # export / restore / snapshot
    # ------------------------------------------------------------------

    def export_state(self) -> CMSCheckpointState:
        if self._mode == "mlp":
            return CMSCheckpointState(
                online_fast=self._online_mlp.representation_vector(),
                session_medium=self._session_mlp.representation_vector(),
                background_slow=self._background_mlp.representation_vector(),
                last_update_ms=self._last_update_ms,
                total_observations=self._total_observations,
                total_reflections=self._total_reflections,
                session_observations_since_update=self._session_observations_since_update,
                background_observations_since_update=self._background_observations_since_update,
                session_pending_signal=self._session_pending_signal,
                background_pending_signal=self._background_pending_signal,
                mode="mlp",
                mlp_params=(
                    self._online_mlp.export_params(),
                    self._session_mlp.export_params(),
                    self._background_mlp.export_params(),
                ),
            )
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

    def restore_state(self, state: CMSCheckpointState) -> None:
        self._last_update_ms = state.last_update_ms
        self._total_observations = state.total_observations
        self._total_reflections = state.total_reflections
        self._session_observations_since_update = state.session_observations_since_update
        self._background_observations_since_update = state.background_observations_since_update
        self._session_pending_signal = state.session_pending_signal
        self._background_pending_signal = state.background_pending_signal

        if self._mode == "mlp":
            if state.mode == "mlp" and state.mlp_params:
                self._online_mlp.restore_params(state.mlp_params[0])
                self._session_mlp.restore_params(state.mlp_params[1])
                self._background_mlp.restore_params(state.mlp_params[2])
            else:
                self._online_mlp.load_representation(state.online_fast)
                self._session_mlp.load_representation(state.session_medium)
                self._background_mlp.load_representation(state.background_slow)
        elif self._mode == "vector":
            self._online_fast = state.online_fast
            self._session_medium = state.session_medium
            self._background_slow = state.background_slow

    def snapshot(self) -> CMSState:
        if self._mode == "mlp":
            return self._snapshot_mlp()
        return self._snapshot_vector()

    def _snapshot_vector(self) -> CMSState:
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

    def _snapshot_mlp(self) -> CMSState:
        online_rep = self._online_mlp.representation_vector()
        session_rep = self._session_mlp.representation_vector()
        bg_rep = self._background_mlp.representation_vector()
        pc = self._online_mlp.parameter_count()
        return CMSState(
            online_fast=CMSBandState(
                name="online-fast",
                vector=online_rep,
                last_update_ms=self._last_update_ms,
                cadence_interval=1,
                observations_since_update=0,
                pending_signal=tuple(0.0 for _ in range(self._dim)),
                learning_rate=self._online_lr,
                momentum=tuple(self._online_mlp._state_momentum),
                anti_forgetting_strength=self._anti_forgetting,
                mode="mlp",
                mlp_param_count=pc,
            ),
            session_medium=CMSBandState(
                name="session-medium",
                vector=session_rep,
                last_update_ms=self._last_update_ms,
                cadence_interval=self._session_cadence,
                observations_since_update=self._session_observations_since_update,
                pending_signal=self._session_pending_signal,
                learning_rate=self._session_lr,
                momentum=tuple(self._session_mlp._state_momentum),
                anti_forgetting_strength=self._anti_forgetting,
                mode="mlp",
                mlp_param_count=pc,
            ),
            background_slow=CMSBandState(
                name="background-slow",
                vector=bg_rep,
                last_update_ms=self._last_update_ms,
                cadence_interval=self._background_cadence,
                observations_since_update=self._background_observations_since_update,
                pending_signal=self._background_pending_signal,
                learning_rate=self._background_lr,
                momentum=tuple(self._background_mlp._state_momentum),
                anti_forgetting_strength=self._anti_forgetting,
                mode="mlp",
                mlp_param_count=pc,
            ),
            total_observations=self._total_observations,
            total_reflections=self._total_reflections,
            description=(
                f"CMS core mode=mlp d_in={self._dim} d_hidden={self._d_hidden} "
                f"variant={self._variant.value} params/band={pc}, "
                f"online_lr={self._online_lr}, session_lr={self._session_lr}, "
                f"bg_lr={self._background_lr}, anti_forgetting={self._anti_forgetting}."
            ),
            variant=self._variant.value,
        )

    # ------------------------------------------------------------------
    # internal helpers — signal extraction
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # internal helpers — vector mode
    # ------------------------------------------------------------------

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
        """Gradient-style update: compute error -> update momentum -> apply."""
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
        """Backflow from slow to fast (vector mode)."""
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
        """Cadence-gated gradient update for medium/slow bands (vector mode)."""
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

    # ------------------------------------------------------------------
    # internal helpers — MLP mode
    # ------------------------------------------------------------------

    def _integrate_signal_mlp(
        self,
        *,
        mlp: CMSBandMLP,
        pending_signal: tuple[float, ...],
        observations_since_update: int,
        signal: tuple[float, ...],
        cadence_interval: int,
    ) -> tuple[tuple[float, ...], int]:
        """Cadence-gated MLP update for medium/slow bands."""
        next_count = observations_since_update + 1
        next_pending = tuple(
            _clamp((pending_signal[i] * observations_since_update + signal[i]) / next_count)
            for i in range(self._dim)
        )
        if next_count < cadence_interval:
            return (next_pending, next_count)
        mlp.update(target=next_pending)
        return (tuple(0.0 for _ in range(self._dim)), 0)

    def _apply_anti_forgetting_mlp(self) -> None:
        """Backflow from slow to fast (MLP mode)."""
        self._online_mlp.mix_from(self._background_mlp, strength=self._anti_forgetting, factor=0.1)
        self._session_mlp.mix_from(self._background_mlp, strength=self._anti_forgetting, factor=0.05)
