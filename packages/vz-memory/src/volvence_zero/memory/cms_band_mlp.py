"""2-layer residual MLP knowledge store used by CMS bands."""

from __future__ import annotations

import math

from volvence_zero.memory.cms_math import _clamp, _init_weight, _matvec


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

    def update(
        self,
        *,
        target: tuple[float, ...],
        lr_scale: float = 1.0,
        momentum_gate: float = 1.0,
    ) -> None:
        self._apply_single_target_update(
            target=target,
            lr_scale=lr_scale,
            momentum_gate=momentum_gate,
        )

    def update_with_replay(
        self,
        *,
        targets: tuple[tuple[float, ...], ...],
        weights: tuple[float, ...],
        lr_scale: float = 1.0,
        momentum_gate: float = 1.0,
    ) -> None:
        """ATLAS-style joint optimization step over a replay window.

        The joint loss

            L = sum_k w_k * || y(x) - t_k ||^2

        has gradient ``2 * (y - sum_k w_k * t_k)`` whenever the weights are
        normalized to sum to 1, which is identical (up to the constant 2 that
        is absorbed into the learning rate) to the single-target gradient on
        the weighted-average target. We therefore run **one** SGD step on
        the weighted-average target. This keeps ``update_with_replay`` with
        a single normalized target bit-equal to the legacy ``update``
        method, which the SHADOW vs ACTIVE protocol relies on.

        See ``docs/specs/cms-atlas-titans-uplift.md`` §3.
        """
        if not targets:
            return
        if len(targets) != len(weights):
            raise ValueError(
                f"replay targets length {len(targets)} != weights length {len(weights)}"
            )
        d_in = self._d_in
        weight_total = sum(max(0.0, weight) for weight in weights)
        if weight_total <= 1e-9:
            return
        averaged = [0.0] * d_in
        for target, weight in zip(targets, weights, strict=True):
            normalized_weight = max(0.0, weight) / weight_total
            if normalized_weight <= 0.0:
                continue
            for i in range(d_in):
                averaged[i] += normalized_weight * target[i]
        self._apply_single_target_update(
            target=tuple(averaged),
            lr_scale=lr_scale,
            momentum_gate=momentum_gate,
        )

    def _apply_single_target_update(
        self,
        *,
        target: tuple[float, ...],
        lr_scale: float,
        momentum_gate: float,
    ) -> None:
        x = self._state
        d_in = self._d_in
        d_hidden = self._d_hidden
        beta = _clamp(self._momentum_beta * max(0.0, momentum_gate))
        lr = self._lr * max(0.0, lr_scale)

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
        rate = _clamp(strength * factor)
        for i in range(self._d_in):
            self._state[i] = _clamp(
                self._state[i] + rate * (other._state[i] - self._state[i])
            )
        for i in range(len(self._w1)):
            self._w1[i] += rate * (other._w1[i] - self._w1[i])
        for i in range(len(self._w2)):
            self._w2[i] += rate * (other._w2[i] - self._w2[i])
