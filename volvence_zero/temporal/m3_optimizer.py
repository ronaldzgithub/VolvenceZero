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
        return M3OptimizerState(
            fast_momentum=tuple(tuple(row) for row in self._fast_momentum),
            slow_momentum=tuple(tuple(row) for row in self._slow_momentum),
            step_count=self._step_count,
            slow_update_count=self._slow_update_count,
            description=(
                f"M3 optimizer step={self._step_count}, "
                f"slow_updates={self._slow_update_count}, "
                f"fast_beta={self._fast_beta}, slow_beta={self._slow_beta}, "
                f"slow_interval={self._slow_interval}."
            ),
        )

    def restore_state(self, state: M3OptimizerState) -> None:
        for group_idx in range(min(self._num_groups, len(state.fast_momentum))):
            for dim_idx in range(min(self._group_dim, len(state.fast_momentum[group_idx]))):
                self._fast_momentum[group_idx][dim_idx] = state.fast_momentum[group_idx][dim_idx]
                self._slow_momentum[group_idx][dim_idx] = state.slow_momentum[group_idx][dim_idx]
        self._step_count = state.step_count
        self._slow_update_count = state.slow_update_count
