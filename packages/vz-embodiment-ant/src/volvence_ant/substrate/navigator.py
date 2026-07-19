"""Body-side ring-attractor heading + path-integration home vector.

This is the digital analogue of the central complex: a *frozen*, non-learning
dead-reckoning system attached to the body. It integrates efference copies of
the ant's own turn/step commands into:

- an internal heading estimate ``h_hat`` (ring attractor state), and
- a home-vector accumulator (running displacement from the nest).

It never reads the world's ground-truth *position*, so its homing accuracy is a
genuine path-integration measurement. A small proprioceptive noise term makes
the accumulation realistic.

Optionally it fuses a noisy *absolute heading* observation — the digital
analogue of the sky-polarization compass that desert ants and the AntBot robot
(Dupeyroux 2019) both rely on. This is a body sensor, not a position readout:
it supplies an absolute bearing reference that keeps the heading estimate from
drifting as a random walk. Without this channel a pure efference-copy integrator
cannot match the AntBot ~0.5%/journey scale, because heading error accumulates
as sqrt(N); with an AntBot-class compass (~0.4 deg) it can. The fusion is a
complementary filter with gain ``compass_gain``; ``compass_gain=0`` disables it
and recovers pure dead reckoning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


def wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi]."""

    wrapped = (angle + math.pi) % TWO_PI - math.pi
    return wrapped if wrapped > -math.pi else math.pi


@dataclass(frozen=True)
class NavigatorState:
    h_hat: float  # heading estimate, world frame (radians)
    home_dx: float  # estimated vector FROM current pos TO nest (x)
    home_dy: float  # estimated vector FROM current pos TO nest (y)

    @property
    def home_distance(self) -> float:
        return math.hypot(self.home_dx, self.home_dy)

    @property
    def home_bearing(self) -> float:
        return math.atan2(self.home_dy, self.home_dx)


class AntNavigator:
    """Frozen dead-reckoning integrator (ring attractor + path integration)."""

    def __init__(
        self,
        *,
        step_size: float,
        heading_noise: float = 0.01,
        step_noise: float = 0.01,
        compass_gain: float = 0.0,
        compass_noise: float = 0.007,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= compass_gain <= 1.0:
            raise ValueError("compass_gain must lie in [0, 1]")
        if compass_noise < 0.0:
            raise ValueError("compass_noise must be non-negative")
        self._step_size = step_size
        self._heading_noise = heading_noise
        self._step_noise = step_noise
        self._compass_gain = compass_gain
        self._compass_noise = compass_noise
        self._rng = np.random.default_rng(seed)
        self._h_hat = 0.0
        # home vector points from current position back to the nest; starts at 0
        self._home_dx = 0.0
        self._home_dy = 0.0

    def reset(self, *, initial_heading: float = 0.0) -> None:
        self._h_hat = initial_heading % TWO_PI
        self._home_dx = 0.0
        self._home_dy = 0.0

    def sync_to(self, *, x: float, y: float, heading: float, nest: tuple[float, float]) -> None:
        """Set the estimate to a known true pose (perfect PI).

        Used by the scripted-route familiarity experiment, where path-integration
        error is not under study and we want a consistent egocentric home
        channel along a fixed route.
        """

        self._h_hat = heading % TWO_PI
        self._home_dx = nest[0] - x
        self._home_dy = nest[1] - y

    @property
    def state(self) -> NavigatorState:
        return NavigatorState(h_hat=self._h_hat, home_dx=self._home_dx, home_dy=self._home_dy)

    def update(
        self,
        *,
        turn_command: float,
        step_command: float,
        true_heading: float | None = None,
    ) -> NavigatorState:
        """Integrate one efference copy. Returns the post-move state.

        The ant only knows the commands it issued (plus small proprioceptive
        noise); it never sees the true position. When ``compass_gain > 0`` and a
        ``true_heading`` is supplied, the integrator additionally fuses a noisy
        absolute-heading observation (sky-compass analogue) via a complementary
        filter — an absolute bearing reference, not a position readout.
        """

        noisy_turn = turn_command + float(self._rng.normal(0.0, self._heading_noise))
        self._h_hat = (self._h_hat + noisy_turn) % TWO_PI
        if self._compass_gain > 0.0 and true_heading is not None:
            compass = true_heading + float(self._rng.normal(0.0, self._compass_noise))
            self._h_hat = (
                self._h_hat + self._compass_gain * wrap_angle(compass - self._h_hat)
            ) % TWO_PI
        noisy_step = max(0.0, step_command + float(self._rng.normal(0.0, self._step_noise)))
        move_x = noisy_step * math.cos(self._h_hat)
        move_y = noisy_step * math.sin(self._h_hat)
        # moving by (move_x, move_y) shifts the nest-relative vector by -(move)
        self._home_dx -= move_x
        self._home_dy -= move_y
        return self.state

    def egocentric_home(self) -> tuple[float, float]:
        """Home direction in the body's egocentric frame (cos, sin of bearing-h_hat)."""

        rel = wrap_angle(self.home_bearing_minus_heading())
        return (math.cos(rel), math.sin(rel))

    def home_bearing_minus_heading(self) -> float:
        return math.atan2(self._home_dy, self._home_dx) - self._h_hat
