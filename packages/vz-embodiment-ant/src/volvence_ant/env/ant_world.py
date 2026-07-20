"""Digital-ant 2D world: continuous positions + a smooth food odour field.

Pure numpy, zero ML. The world owns the *true* body pose and the food field;
it exposes only a :class:`WorldObservation` (antenna samples + proprioception +
discrete contact flags). Ground-truth geometry (true home bearing / distance)
is exposed on the observation ONLY under ``eval_*`` fields, which
``sense_encode`` deliberately does not read, so the controller never gets a
free home vector.

Food odour field (feasibility doc 3.1):

    I_food(x) = sum_k A_k * exp(-||x - p_k|| / lambda_food)

Homing is by path integration in the controller, not by reading the field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class FoodSource:
    x: float
    y: float
    strength: float = 1.0
    decay: float = 6.0
    radius: float = 1.5
    remaining: float = float("inf")


@dataclass(frozen=True)
class AxisAlignedObstacle:
    """Immutable environment-owned rectangular obstacle."""

    obstacle_id: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def __post_init__(self) -> None:
        bounds = (self.min_x, self.max_x, self.min_y, self.max_y)
        if not self.obstacle_id:
            raise ValueError("obstacle_id must be non-empty")
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError("obstacle bounds must be finite")
        if self.min_x >= self.max_x or self.min_y >= self.max_y:
            raise ValueError(
                "obstacle bounds must satisfy min_x < max_x and min_y < max_y"
            )

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def strictly_contains(self, x: float, y: float) -> bool:
        return self.min_x < x < self.max_x and self.min_y < y < self.max_y

    def penetration_depth(self, x: float, y: float) -> float:
        if not self.strictly_contains(x, y):
            return 0.0
        return min(
            x - self.min_x,
            self.max_x - x,
            y - self.min_y,
            self.max_y - y,
        )


@dataclass(frozen=True)
class MotorDistortionProfile:
    """Hidden actuator transfer function, optionally switching once at runtime."""

    turn_gain: float = 1.0
    turn_bias: float = 0.0
    switch_tick: int | None = None
    switched_turn_gain: float = 1.0
    switched_turn_bias: float = 0.0

    def at_tick(self, tick: int) -> tuple[float, float]:
        if self.switch_tick is not None and tick >= self.switch_tick:
            return (self.switched_turn_gain, self.switched_turn_bias)
        return (self.turn_gain, self.turn_bias)


@dataclass(frozen=True)
class AntBody:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0  # true heading in radians, world frame
    carrying_food: bool = False

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class WorldObservation:
    # --- sensory channels (fed through sense_encode) ---
    food_left: float
    food_right: float
    home_pher_left: float
    home_pher_right: float
    trail_pher_left: float
    trail_pher_right: float
    obstacle_left: float
    obstacle_right: float
    obstacle_contact: bool
    last_turn_command: float
    # --- discrete contact / drive flags ---
    carrying_food: bool
    at_nest: bool
    at_food: bool
    food_center: float
    alarm: float  # panic channel (predator etc.); 0 unless triggered
    # --- eval-only ground truth (NOT read by sense_encode) ---
    eval_home_bearing: float
    eval_home_distance: float
    eval_true_heading: float


@dataclass(frozen=True)
class WorldTransitionEvidence:
    """Observable facts produced by exactly one body action."""

    transition_id: str
    body_id: int
    tick: int
    action_sequence: int
    picked_up: bool
    delivered: bool
    carrying_before: bool
    carrying_after: bool
    commanded_turn: float
    applied_turn: float
    commanded_step: float
    applied_step: float
    blocked_by_obstacle: bool
    obstacle_id: str | None


@dataclass
class AntWorldConfig:
    nest_x: float = 0.0
    nest_y: float = 0.0
    nest_radius: float = 1.0
    antenna_offset_deg: float = 30.0
    antenna_reach: float = 0.6
    step_size: float = 0.4
    max_turn_rate: float = math.radians(45.0)
    food_pickup_radius: float = 1.2
    seed: int = 0
    # Empty = identity actuator. One profile broadcasts to every body; otherwise
    # the tuple must contain exactly one immutable profile per body.
    motor_distortions: tuple[MotorDistortionProfile, ...] = ()


class AntWorld:
    """Single-body continuous 2D foraging world with a smooth food field.

    Multi-body colony use (Phase 1) shares one world instance across bodies;
    each body is addressed by an integer id via :meth:`observe` / :meth:`act`.
    Phase 0 uses a single implicit body (id 0).
    """

    def __init__(
        self,
        *,
        config: AntWorldConfig | None = None,
        food_sources: tuple[FoodSource, ...] = (),
        obstacles: tuple[AxisAlignedObstacle, ...] = (),
        n_bodies: int = 1,
    ) -> None:
        self.config = config or AntWorldConfig()
        self._validate_motor_distortions(
            self.config.motor_distortions, n_bodies=max(1, n_bodies)
        )
        self._motor_distortions = tuple(self.config.motor_distortions)
        self._rng = np.random.default_rng(self.config.seed)
        self._food: list[FoodSource] = list(food_sources) or [
            FoodSource(x=8.0, y=0.0, strength=1.0, decay=6.0)
        ]
        self._obstacles = self._validated_obstacles(obstacles)
        self._bodies: list[AntBody] = [
            self._spawn_body() for _ in range(max(1, n_bodies))
        ]
        self._last_turn: list[float] = [0.0 for _ in self._bodies]
        self._last_obstacle_contact: list[bool] = [False for _ in self._bodies]
        self._alarm: list[float] = [0.0 for _ in self._bodies]
        self._last_transition: list[WorldTransitionEvidence | None] = [
            None for _ in self._bodies
        ]
        self._action_sequence = 0
        self.tick: int = 0
        # metrics
        self.food_delivered: int = 0
        self.food_pickups: int = 0

    # -- construction helpers -------------------------------------------------
    def _spawn_body(self) -> AntBody:
        heading = float(self._rng.uniform(0.0, TWO_PI))
        return AntBody(
            x=self.config.nest_x,
            y=self.config.nest_y,
            heading=heading,
            carrying_food=False,
        )

    @property
    def nest(self) -> tuple[float, float]:
        return (self.config.nest_x, self.config.nest_y)

    @property
    def n_bodies(self) -> int:
        return len(self._bodies)

    def body(self, body_id: int = 0) -> AntBody:
        return self._bodies[body_id]

    def food_sources(self) -> tuple[FoodSource, ...]:
        return tuple(self._food)

    def obstacles(self) -> tuple[AxisAlignedObstacle, ...]:
        return self._obstacles

    def last_transition(self, body_id: int = 0) -> WorldTransitionEvidence:
        transition = self._last_transition[body_id]
        if transition is None:
            raise RuntimeError(f"body {body_id} has not acted yet")
        return transition

    # -- food field -----------------------------------------------------------
    def food_intensity(self, x: float, y: float) -> float:
        total = 0.0
        for src in self._food:
            if src.remaining <= 0.0:
                continue
            dist = math.hypot(x - src.x, y - src.y)
            total += src.strength * math.exp(-dist / max(src.decay, 1e-6))
        return total

    # -- observation ----------------------------------------------------------
    def observe(self, body_id: int = 0) -> WorldObservation:
        body = self._bodies[body_id]
        cfg = self.config
        phi = math.radians(cfg.antenna_offset_deg)
        r = cfg.antenna_reach
        lx = body.x + r * math.cos(body.heading + phi)
        ly = body.y + r * math.sin(body.heading + phi)
        rx = body.x + r * math.cos(body.heading - phi)
        ry = body.y + r * math.sin(body.heading - phi)

        home_dx = cfg.nest_x - body.x
        home_dy = cfg.nest_y - body.y
        home_distance = math.hypot(home_dx, home_dy)
        home_bearing = math.atan2(home_dy, home_dx)

        at_nest = home_distance <= cfg.nest_radius
        at_food, food_center = self._food_contact(body)

        pher_l = self._pheromone_samples(lx, ly)
        pher_r = self._pheromone_samples(rx, ry)

        return WorldObservation(
            food_left=self.food_intensity(lx, ly),
            food_right=self.food_intensity(rx, ry),
            home_pher_left=pher_l[0],
            home_pher_right=pher_r[0],
            trail_pher_left=pher_l[1],
            trail_pher_right=pher_r[1],
            obstacle_left=self._obstacle_sample(lx, ly),
            obstacle_right=self._obstacle_sample(rx, ry),
            obstacle_contact=self._last_obstacle_contact[body_id],
            last_turn_command=self._last_turn[body_id],
            carrying_food=body.carrying_food,
            at_nest=at_nest,
            at_food=at_food,
            food_center=food_center,
            alarm=self._alarm[body_id],
            eval_home_bearing=home_bearing,
            eval_home_distance=home_distance,
            eval_true_heading=body.heading,
        )

    def _food_contact(self, body: AntBody) -> tuple[bool, float]:
        cfg = self.config
        center = self.food_intensity(body.x, body.y)
        for src in self._food:
            if src.remaining <= 0.0:
                continue
            if math.hypot(body.x - src.x, body.y - src.y) <= cfg.food_pickup_radius:
                return True, center
        return False, center

    def _pheromone_samples(self, x: float, y: float) -> tuple[float, float]:
        """Base world has no pheromone field (Phase 0). Overridden in colony."""

        return (0.0, 0.0)

    def pheromone_metrics(self) -> tuple[float, float | None]:
        """Published trail mass/entropy; a plain world has no shared bus."""

        return 0.0, None

    def _obstacle_sample(self, x: float, y: float) -> float:
        return 1.0 if any(obstacle.contains(x, y) for obstacle in self._obstacles) else 0.0

    # -- action ---------------------------------------------------------------
    def act(
        self,
        *,
        turn_command: float,
        step_command: float,
        body_id: int = 0,
    ) -> WorldObservation:
        """Apply a motor command to one body and return the next observation."""

        cfg = self.config
        body = self._bodies[body_id]
        commanded_turn = float(
            np.clip(turn_command, -cfg.max_turn_rate, cfg.max_turn_rate)
        )
        gain, bias = self._motor_distortion(body_id).at_tick(self.tick)
        applied_turn = float(
            np.clip(
                commanded_turn * gain + bias,
                -cfg.max_turn_rate,
                cfg.max_turn_rate,
            )
        )
        new_heading = (body.heading + applied_turn) % TWO_PI
        commanded_step = float(np.clip(step_command, 0.0, cfg.step_size))
        target_x = body.x + commanded_step * math.cos(new_heading)
        target_y = body.y + commanded_step * math.sin(new_heading)
        new_x, new_y, obstacle_id = self._resolve_obstacle_motion(
            start_x=body.x,
            start_y=body.y,
            target_x=target_x,
            target_y=target_y,
        )
        applied_step = math.hypot(new_x - body.x, new_y - body.y)
        blocked_by_obstacle = obstacle_id is not None
        new_body = replace(body, x=new_x, y=new_y, heading=new_heading)

        new_body = self._resolve_contacts(new_body, body_id)
        self._action_sequence += 1
        self._last_transition[body_id] = WorldTransitionEvidence(
            transition_id=f"ant:{body_id}:action:{self._action_sequence}",
            body_id=body_id,
            tick=self.tick,
            action_sequence=self._action_sequence,
            picked_up=not body.carrying_food and new_body.carrying_food,
            delivered=body.carrying_food and not new_body.carrying_food,
            carrying_before=body.carrying_food,
            carrying_after=new_body.carrying_food,
            commanded_turn=commanded_turn,
            applied_turn=applied_turn,
            commanded_step=commanded_step,
            applied_step=applied_step,
            blocked_by_obstacle=blocked_by_obstacle,
            obstacle_id=obstacle_id,
        )
        self._bodies[body_id] = new_body
        # Efference copy only: the controller knows what it commanded, not the
        # hidden actuator transfer. Applied turn remains audit-only evidence.
        self._last_turn[body_id] = commanded_turn
        self._last_obstacle_contact[body_id] = blocked_by_obstacle
        self._on_body_moved(body_id, new_body)
        if body_id == self.n_bodies - 1:
            self.tick += 1
            self._decay_alarm()
            self._on_round_complete()
        return self.observe(body_id)

    def _motor_distortion(self, body_id: int) -> MotorDistortionProfile:
        profiles = self._motor_distortions
        if not profiles:
            return MotorDistortionProfile()
        return profiles[0] if len(profiles) == 1 else profiles[body_id]

    def _resolve_obstacle_motion(
        self,
        *,
        start_x: float,
        start_y: float,
        target_x: float,
        target_y: float,
    ) -> tuple[float, float, str | None]:
        earliest_fraction = 1.0
        obstacle_id: str | None = None
        for obstacle in self._obstacles:
            # A newly activated obstacle never teleports a body that is already
            # inside it. The body may leave, but cannot re-enter afterwards.
            if obstacle.strictly_contains(start_x, start_y):
                if (
                    obstacle.strictly_contains(target_x, target_y)
                    and obstacle.penetration_depth(target_x, target_y)
                    >= obstacle.penetration_depth(start_x, start_y)
                ):
                    return start_x, start_y, obstacle.obstacle_id
                continue
            if obstacle.contains(start_x, start_y):
                probe_fraction = 1e-9
                probe_x = start_x + (target_x - start_x) * probe_fraction
                probe_y = start_y + (target_y - start_y) * probe_fraction
                if not obstacle.strictly_contains(probe_x, probe_y):
                    continue
            fraction = self._segment_obstacle_entry_fraction(
                start_x=start_x,
                start_y=start_y,
                target_x=target_x,
                target_y=target_y,
                obstacle=obstacle,
            )
            if fraction is not None and fraction < earliest_fraction:
                earliest_fraction = fraction
                obstacle_id = obstacle.obstacle_id
        if obstacle_id is None:
            return target_x, target_y, None
        safe_fraction = max(0.0, earliest_fraction - 1e-9)
        return (
            start_x + (target_x - start_x) * safe_fraction,
            start_y + (target_y - start_y) * safe_fraction,
            obstacle_id,
        )

    @staticmethod
    def _segment_obstacle_entry_fraction(
        *,
        start_x: float,
        start_y: float,
        target_x: float,
        target_y: float,
        obstacle: AxisAlignedObstacle,
    ) -> float | None:
        """Return the first segment fraction entering an axis-aligned box."""

        entry = 0.0
        exit_ = 1.0
        for origin, delta, lower, upper in (
            (start_x, target_x - start_x, obstacle.min_x, obstacle.max_x),
            (start_y, target_y - start_y, obstacle.min_y, obstacle.max_y),
        ):
            if abs(delta) <= 1e-15:
                if origin < lower or origin > upper:
                    return None
                continue
            first = (lower - origin) / delta
            second = (upper - origin) / delta
            if first > second:
                first, second = second, first
            entry = max(entry, first)
            exit_ = min(exit_, second)
            if entry > exit_:
                return None
        if exit_ < 0.0 or entry > 1.0:
            return None
        return max(0.0, entry)

    @staticmethod
    def _validate_motor_distortions(
        profiles: tuple[MotorDistortionProfile, ...], *, n_bodies: int
    ) -> None:
        if profiles and len(profiles) not in {1, n_bodies}:
            raise ValueError(
                "motor_distortions must be empty, a single broadcast profile, "
                f"or one profile per body; got {len(profiles)} profiles for "
                f"{n_bodies} bodies"
            )

    def set_motor_distortion(
        self,
        profile: MotorDistortionProfile,
        *,
        body_id: int | None = None,
    ) -> None:
        """Replace the hidden actuator transfer at an environment boundary.

        This owner API is intended for controlled perturbations.  The profile
        remains absent from observations; agents can only observe its physical
        consequences through the normal outcome/PE path.
        """

        if body_id is None:
            self._motor_distortions = (profile,)
            return
        if body_id < 0 or body_id >= self.n_bodies:
            raise IndexError(f"body_id {body_id} outside [0, {self.n_bodies})")
        per_body = [self._motor_distortion(index) for index in range(self.n_bodies)]
        per_body[body_id] = profile
        self._motor_distortions = tuple(per_body)

    def _resolve_contacts(self, body: AntBody, body_id: int) -> AntBody:
        cfg = self.config
        # deposit at nest
        if body.carrying_food:
            if math.hypot(body.x - cfg.nest_x, body.y - cfg.nest_y) <= cfg.nest_radius:
                self.food_delivered += 1
                return replace(body, carrying_food=False)
            return body
        # pick up at a food source
        for index, src in enumerate(self._food):
            if src.remaining <= 0.0:
                continue
            if math.hypot(body.x - src.x, body.y - src.y) <= cfg.food_pickup_radius:
                self.food_pickups += 1
                if src.remaining != float("inf"):
                    self._food[index] = replace(src, remaining=src.remaining - 1.0)
                return replace(body, carrying_food=True)
        return body

    def _on_body_moved(self, body_id: int, body: AntBody) -> None:
        """Hook for colony subclasses to deposit pheromone. No-op in Phase 0."""

    def _on_round_complete(self) -> None:
        """Hook fired after the last body moves each round. No-op in Phase 0."""

    # -- perturbations (for demos / matched-control) --------------------------
    def move_food(self, *, index: int = 0, x: float, y: float) -> None:
        self._food[index] = replace(self._food[index], x=x, y=y)

    def set_food_sources(self, food_sources: tuple[FoodSource, ...]) -> None:
        self._food = list(food_sources)

    def set_obstacles(self, obstacles: tuple[AxisAlignedObstacle, ...]) -> None:
        """Atomically replace the environment-owned obstacle geometry."""

        self._obstacles = self._validated_obstacles(obstacles)

    @staticmethod
    def _validated_obstacles(
        obstacles: tuple[AxisAlignedObstacle, ...],
    ) -> tuple[AxisAlignedObstacle, ...]:
        obstacle_ids = tuple(obstacle.obstacle_id for obstacle in obstacles)
        if len(set(obstacle_ids)) != len(obstacle_ids):
            raise ValueError("obstacle_id values must be unique")
        return tuple(obstacles)

    def trigger_alarm(self, *, body_id: int | None = None, magnitude: float = 1.0) -> None:
        if body_id is None:
            self._alarm = [magnitude for _ in self._alarm]
        else:
            self._alarm[body_id] = magnitude

    def _decay_alarm(self) -> None:
        self._alarm = [max(0.0, value - 0.25) for value in self._alarm]

    def reset_body(self, body_id: int = 0) -> None:
        self._bodies[body_id] = self._spawn_body()
        self._last_turn[body_id] = 0.0
        self._last_obstacle_contact[body_id] = False
        self._alarm[body_id] = 0.0
        self._last_transition[body_id] = None

    def set_body_pose(
        self,
        *,
        x: float,
        y: float,
        heading: float,
        carrying_food: bool | None = None,
        body_id: int = 0,
    ) -> None:
        """Teleport a body (used by the scripted-route familiarity experiment)."""

        body = self._bodies[body_id]
        carry = body.carrying_food if carrying_food is None else carrying_food
        self._bodies[body_id] = AntBody(x=x, y=y, heading=heading % TWO_PI, carrying_food=carry)
