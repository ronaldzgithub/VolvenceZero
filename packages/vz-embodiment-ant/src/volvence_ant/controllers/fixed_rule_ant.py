"""``FixedRuleAnt`` — a classic hardcoded finite-state-machine forager.

This is the *negative* control for the whole project: a competent but brittle
ant built the traditional way (explicit ``if situation then behaviour`` rules,
ACO / NetLogo style). It shares the SAME frozen substrate (world + navigator +
sense/motor functions) as the kernel ant, but the policy is hand-written
instead of learned.

It serves three roles:
1. a fast, deterministic outbound driver for the path-integration homing
   experiment (the navigator's homing precision is substrate-level, so it does
   not need the learned kernel to be measured);
2. the "hardcoded" arm in the matched-control study (Workstream E);
3. the "scripted ant" side of the emergent-vs-scripted demo (Workstream G2).

By design it uses keyword-free numeric rules (gradient following, path
integration) — the point of the demo is that these rules break under
perturbations the author did not foresee, whereas the learned kernel adapts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from volvence_ant.env.ant_world import AntWorld, WorldObservation
from volvence_ant.substrate.navigator import AntNavigator, NavigatorState, wrap_angle


@dataclass
class FixedRuleConfig:
    seed: int = 0
    heading_noise: float = 0.01
    step_noise: float = 0.01
    explore_jitter: float = 0.35  # random turn when nothing to sense
    gradient_gain: float = 6.0
    food_sense_threshold: float = 0.05  # only ascend when food is genuinely near
    trail_gain: float = 12.0
    trail_follow_threshold: float = 1e-5
    panic_flee_speed: float = 1.0
    # Caste tendency (Phase 2). 0.5 is neutral and preserves Phase 1 behaviour.
    # High -> explorer (roams wider, ignores trail more); low -> patroller /
    # exploiter (hugs the trail). This value is SET by the offline caste
    # reprogramming artifact, not chosen at runtime.
    exploration_bias: float = 0.5


@dataclass(frozen=True)
class FixedRuleStep:
    tick: int
    x: float
    y: float
    carrying_food: bool
    at_food: bool
    at_nest: bool
    mode: str
    turn_command: float
    step_command: float
    homing_direction_error: float


class FixedRuleAnt:
    """Hand-written FSM forager sharing the frozen ant substrate."""

    def __init__(
        self,
        world: AntWorld,
        *,
        config: FixedRuleConfig | None = None,
        body_id: int = 0,
    ) -> None:
        self.world = world
        self.config = config or FixedRuleConfig()
        self._body_id = body_id
        self._rng = np.random.default_rng(self.config.seed)
        self.navigator = AntNavigator(
            step_size=world.config.step_size,
            heading_noise=self.config.heading_noise,
            step_noise=self.config.step_noise,
            seed=self.config.seed,
        )
        obs = world.observe(body_id)
        self.navigator.reset(initial_heading=obs.eval_true_heading)
        self._last_obs = obs
        self.trajectory: list[FixedRuleStep] = []

    def _decide(self, obs: WorldObservation, nav: NavigatorState) -> tuple[float, float, str]:
        cfg = self.config
        max_turn = self.world.config.max_turn_rate
        step = self.world.config.step_size

        # 1) panic reflex: alarm overrides everything (flee away from nest-less;
        #    here: flee in current heading fast). Hardcoded safety branch.
        if obs.alarm > 0.5:
            return 0.0, step * cfg.panic_flee_speed, "panic-flee"

        # 2) carrying food -> steer toward the path-integration home vector.
        if obs.carrying_food:
            rel = wrap_angle(nav.home_bearing - nav.h_hat)
            turn = float(np.clip(rel, -max_turn, max_turn))
            return turn, step, "homebound"

        # 3) strong local food gradient -> ascend it (differential steering).
        #    Only when food is genuinely near; otherwise the trail must guide.
        grad = obs.food_left - obs.food_right
        if obs.food_center > cfg.food_sense_threshold:
            turn = float(np.clip(grad * cfg.gradient_gain, -max_turn, max_turn))
            return turn, step, "gradient-ascent"

        # 3b) stigmergy: no food sensed -> follow the trail pheromone gradient
        #     from the shared bus (this is what makes the colony converge).
        #     Explorers (high bias) probabilistically ignore the trail to roam.
        trail_sum = obs.trail_pher_left + obs.trail_pher_right
        trail_grad = obs.trail_pher_left - obs.trail_pher_right
        skip_trail_prob = max(0.0, cfg.exploration_bias - 0.5) * 2.0 * 0.7
        if trail_sum > cfg.trail_follow_threshold and (
            skip_trail_prob <= 0.0 or self._rng.random() >= skip_trail_prob
        ):
            turn = float(np.clip(trail_grad * cfg.trail_gain, -max_turn, max_turn))
            return turn, step, "trail-follow"

        # 4) nothing to sense -> random-walk exploration (wider for explorers).
        jitter = cfg.explore_jitter * (0.6 + 0.8 * cfg.exploration_bias)
        turn = float(self._rng.uniform(-jitter, jitter))
        turn = float(np.clip(turn, -max_turn, max_turn))
        return turn, step, "explore"

    def step(self) -> FixedRuleStep:
        obs = self._last_obs
        nav_state = self.navigator.state
        turn, step_cmd, mode = self._decide(obs, nav_state)
        nav_state = self.navigator.update(turn_command=turn, step_command=step_cmd)
        obs = self.world.act(
            turn_command=turn, step_command=step_cmd, body_id=self._body_id
        )
        self._last_obs = obs
        body = self.world.body(self._body_id)
        estimated = nav_state.home_bearing
        dir_err = abs((estimated - obs.eval_home_bearing + math.pi) % (2.0 * math.pi) - math.pi)
        record = FixedRuleStep(
            tick=self.world.tick,
            x=body.x,
            y=body.y,
            carrying_food=obs.carrying_food,
            at_food=obs.at_food,
            at_nest=obs.at_nest,
            mode=mode,
            turn_command=turn,
            step_command=step_cmd,
            homing_direction_error=dir_err,
        )
        self.trajectory.append(record)
        return record

    def run(self, ticks: int) -> list[FixedRuleStep]:
        for _ in range(ticks):
            self.step()
        return self.trajectory
