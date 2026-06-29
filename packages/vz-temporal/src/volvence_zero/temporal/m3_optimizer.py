"""M3-style dual-timescale momentum optimizer (NL Appendix A.6).

Fast momentum m^(1) updates every step.
Slow momentum m^(2) aggregates every `slow_interval` steps, acting as
optimizer-level "reflection" that can feed into CMS session-medium band.
"""

from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class M3OptimizerState:
    fast_momentum: tuple[tuple[float, ...], ...]
    slow_momentum: tuple[tuple[float, ...], ...]
    step_count: int
    slow_update_count: int
    num_groups: int
    group_dim: int
    fast_beta: float
    slow_beta: float
    slow_interval: int
    mean_fast_norm: float
    mean_slow_norm: float
    description: str


class M3Optimizer:
    """Dual-timescale momentum for parameter groups.

    Each parameter group is a tuple of floats (e.g., one row of a weight
    matrix). Fast momentum updates every step; slow momentum aggregates
    every `slow_interval` steps.
    """

    def __init__(
        self,
        *,
        num_groups: int,
        group_dim: int,
        fast_beta: float = 0.9,
        slow_beta: float = 0.99,
        slow_interval: int = 4,
    ) -> None:
        self._num_groups = num_groups
        self._group_dim = group_dim
        self._fast_beta = fast_beta
        self._slow_beta = slow_beta
        self._slow_interval = max(slow_interval, 1)
        self._fast_momentum: list[list[float]] = [
            [0.0] * group_dim for _ in range(num_groups)
        ]
        self._slow_momentum: list[list[float]] = [
            [0.0] * group_dim for _ in range(num_groups)
        ]
        self._step_count = 0
        self._slow_update_count = 0

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def slow_update_count(self) -> int:
        return self._slow_update_count

    def update(
        self,
        *,
        gradients: tuple[tuple[float, ...], ...],
        learning_rate: float,
        parameters: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[float, ...], ...]:
        """Apply one M3 update step and return the new parameters."""
        updated: list[tuple[float, ...]] = []
        for group_idx in range(self._num_groups):
            grad = gradients[group_idx]
            param = parameters[group_idx]
            fast = self._fast_momentum[group_idx]
            for dim_idx in range(self._group_dim):
                fast[dim_idx] = (
                    self._fast_beta * fast[dim_idx]
                    + (1.0 - self._fast_beta) * grad[dim_idx]
                )
            effective_grad = tuple(fast[d] for d in range(self._group_dim))
            new_param = tuple(
                _clamp(param[d] + learning_rate * effective_grad[d])
                for d in range(self._group_dim)
            )
            updated.append(new_param)

        self._step_count += 1

        if self._step_count % self._slow_interval == 0:
            for group_idx in range(self._num_groups):
                fast = self._fast_momentum[group_idx]
                slow = self._slow_momentum[group_idx]
                for dim_idx in range(self._group_dim):
                    slow[dim_idx] = (
                        self._slow_beta * slow[dim_idx]
                        + (1.0 - self._slow_beta) * fast[dim_idx]
                    )
            self._slow_update_count += 1

        return tuple(updated)

    def slow_momentum_signal(self) -> tuple[float, ...]:
        """Return aggregated slow momentum as a single signal vector.

        Useful for feeding into CMS session-medium band.
        """
        if self._num_groups == 0:
            return ()
        result: list[float] = [0.0] * self._group_dim
        for group_idx in range(self._num_groups):
            for dim_idx in range(self._group_dim):
                result[dim_idx] += abs(self._slow_momentum[group_idx][dim_idx])
        scale = 1.0 / self._num_groups
        return tuple(_clamp(v * scale) for v in result)

    def export_state(self) -> M3OptimizerState:
        mean_fast_norm = (
            sum(abs(value) for row in self._fast_momentum for value in row)
            / max(self._num_groups * self._group_dim, 1)
        )
        mean_slow_norm = (
            sum(abs(value) for row in self._slow_momentum for value in row)
            / max(self._num_groups * self._group_dim, 1)
        )
        return M3OptimizerState(
            fast_momentum=tuple(tuple(row) for row in self._fast_momentum),
            slow_momentum=tuple(tuple(row) for row in self._slow_momentum),
            step_count=self._step_count,
            slow_update_count=self._slow_update_count,
            num_groups=self._num_groups,
            group_dim=self._group_dim,
            fast_beta=self._fast_beta,
            slow_beta=self._slow_beta,
            slow_interval=self._slow_interval,
            mean_fast_norm=mean_fast_norm,
            mean_slow_norm=mean_slow_norm,
            description=(
                f"M3 optimizer step={self._step_count}, "
                f"slow_updates={self._slow_update_count}, "
                f"groups={self._num_groups}, dim={self._group_dim}, "
                f"fast_beta={self._fast_beta}, slow_beta={self._slow_beta}, "
                f"slow_interval={self._slow_interval}, "
                f"mean_fast_norm={mean_fast_norm:.3f}, mean_slow_norm={mean_slow_norm:.3f}."
            ),
        )

    def restore_state(self, state: M3OptimizerState) -> None:
        for group_idx in range(min(self._num_groups, len(state.fast_momentum))):
            for dim_idx in range(min(self._group_dim, len(state.fast_momentum[group_idx]))):
                self._fast_momentum[group_idx][dim_idx] = state.fast_momentum[group_idx][dim_idx]
                self._slow_momentum[group_idx][dim_idx] = state.slow_momentum[group_idx][dim_idx]
        self._step_count = state.step_count
        self._slow_update_count = state.slow_update_count


class DeltaMomentumOptimizer:
    """Delta-momentum with gradient-dependent weight decay (NL Figure 4).

    NL observes that the "delta momentum finds the solution faster, mainly due
    to its gradient-dependent weight decay that helps the momentum term to
    decay or stop when it is needed." We implement that as a momentum whose
    retention shrinks when the new gradient *opposes* the accumulated momentum
    (an overshoot signal) and, more mildly, when the gradient magnitude is
    large. This lets the momentum brake near the target instead of oscillating.

    Convention matches :class:`M3Optimizer` (``param += lr * momentum`` with
    clamping), so the two optimizers are drop-in comparable in proofs.
    """

    def __init__(
        self,
        *,
        num_groups: int,
        group_dim: int,
        beta: float = 0.9,
        reverse_decay: float = 0.9,
        grad_decay: float = 0.1,
    ) -> None:
        self._num_groups = num_groups
        self._group_dim = group_dim
        self._beta = beta
        self._reverse_decay = reverse_decay
        self._grad_decay = grad_decay
        self._momentum: list[list[float]] = [[0.0] * group_dim for _ in range(num_groups)]
        self._step_count = 0

    @property
    def step_count(self) -> int:
        return self._step_count

    def update(
        self,
        *,
        gradients: tuple[tuple[float, ...], ...],
        learning_rate: float,
        parameters: tuple[tuple[float, ...], ...],
        clamp_output: bool = True,
    ) -> tuple[tuple[float, ...], ...]:
        updated: list[tuple[float, ...]] = []
        for group_idx in range(self._num_groups):
            grad = gradients[group_idx]
            param = parameters[group_idx]
            mom = self._momentum[group_idx]
            new_param = []
            for d in range(self._group_dim):
                g = grad[d]
                m_prev = mom[d]
                # gradient-dependent weight decay on the momentum term:
                # strong decay when the gradient reverses the momentum
                # (overshoot), mild decay scaling with gradient magnitude.
                reversing = 1.0 if (g * m_prev) < 0.0 else 0.0
                decay = _clamp(self._reverse_decay * reversing + self._grad_decay * abs(g))
                retention = self._beta * (1.0 - decay)
                m_new = retention * m_prev + (1.0 - self._beta) * g
                mom[d] = m_new
                value = param[d] + learning_rate * m_new
                new_param.append(_clamp(value) if clamp_output else value)
            updated.append(tuple(new_param))
        self._step_count += 1
        return tuple(updated)

    def momentum_norm(self) -> float:
        total = sum(abs(v) for row in self._momentum for v in row)
        return total / max(self._num_groups * self._group_dim, 1)


@dataclass(frozen=True)
class MomentumOvershootComparison:
    target: float
    plain_peak_overshoot: float
    delta_peak_overshoot: float
    plain_final_error: float
    delta_final_error: float
    delta_brakes_overshoot: bool
    description: str = ""


def compare_momentum_overshoot(
    *,
    target: float = 0.8,
    learning_rate: float = 0.6,
    steps: int = 40,
) -> MomentumOvershootComparison:
    """NL Figure 4 style proof: delta-momentum overshoots less than plain.

    Both optimizers track a scalar param toward ``target`` with an
    aggressive learning rate that makes plain momentum overshoot and
    oscillate. The gradient-dependent weight decay should brake the
    delta-momentum run, yielding a smaller peak overshoot.
    """

    delta = DeltaMomentumOptimizer(num_groups=1, group_dim=1)

    def run_plain(*, beta: float = 0.9) -> tuple[float, float]:
        # Plain momentum (unclamped) for a fair overshoot comparison.
        param = 0.0
        m = 0.0
        peak = 0.0
        for _ in range(steps):
            g = target - param
            m = beta * m + (1.0 - beta) * g
            param = param + learning_rate * m
            peak = max(peak, param - target)
        return peak, abs(param - target)

    def run_delta() -> tuple[float, float]:
        param = (0.0,)
        peak = 0.0
        for _ in range(steps):
            grad = ((target - param[0]),)
            out = delta.update(
                gradients=(grad,), learning_rate=learning_rate,
                parameters=(param,), clamp_output=False,
            )
            param = out[0]
            peak = max(peak, param[0] - target)
        return peak, abs(param[0] - target)

    plain_peak, plain_final = run_plain()
    delta_peak, delta_final = run_delta()
    return MomentumOvershootComparison(
        target=target,
        plain_peak_overshoot=plain_peak,
        delta_peak_overshoot=delta_peak,
        plain_final_error=plain_final,
        delta_final_error=delta_final,
        delta_brakes_overshoot=delta_peak < plain_peak,
        description=(
            f"delta-momentum overshoot {delta_peak:.3f} vs plain {plain_peak:.3f}; "
            f"final err delta={delta_final:.3f} plain={plain_final:.3f}"
        ),
    )
