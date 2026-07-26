"""``RandomAnt`` — random-motor lower bound (no learning, no rules).

The floor baseline for the matched-control study: it issues random bounded
turns and constant steps, sharing the same frozen substrate. Any competent
controller (learned or hand-written) must beat it.

"Same frozen substrate" is a load-bearing claim, not decoration: ``random`` is
a first-class arm of the P2-B confirmatory matrix
(``research/ant/05_ecology_p0_p1_p2_plan.md`` §5.4) and of both matched-control
lanes, so its body sensors must be the ones every other arm runs
(``docs/specs/digital-ant-embodiment.md`` §3). Until 2026-07-27 this controller
built a compass-less navigator and integrated the efference copy BEFORE the
world acted, which is a different (and, for path integration, strictly easier)
body than the kernel / scripted / E2E-RL arms.
"""

from __future__ import annotations

import numpy as np

from volvence_ant.controllers.e2e_rl_ant import (
    SHARED_COMPASS_GAIN,
    SHARED_COMPASS_NOISE,
    SHARED_HEADING_NOISE,
    SHARED_STEP_NOISE,
)
from volvence_ant.env.ant_world import AntWorld
from volvence_ant.substrate.navigator import AntNavigator


class RandomAnt:
    def __init__(self, world: AntWorld, *, seed: int = 0, body_id: int = 0) -> None:
        self.world = world
        self._body_id = body_id
        self._rng = np.random.default_rng(seed)
        # Frozen body sensors shared with AntSession / FixedRuleAnt / E2ERLAnt.
        self.navigator = AntNavigator(
            step_size=world.config.step_size,
            heading_noise=SHARED_HEADING_NOISE,
            step_noise=SHARED_STEP_NOISE,
            compass_gain=SHARED_COMPASS_GAIN,
            compass_noise=SHARED_COMPASS_NOISE,
            seed=seed,
        )
        self.navigator.reset(initial_heading=world.observe(body_id).eval_true_heading)
        self.positions: list[tuple[float, float]] = []

    def step(self) -> None:
        max_turn = self.world.config.max_turn_rate
        step = self.world.config.step_size
        turn = float(self._rng.uniform(-max_turn, max_turn))
        # Act first, then integrate: the sky compass reads the POST-move
        # absolute heading, so declaring ``compass_gain`` without feeding
        # ``true_heading`` back would leave this arm on pure dead reckoning.
        observation = self.world.act(
            turn_command=turn, step_command=step, body_id=self._body_id
        )
        self.navigator.update(
            turn_command=turn,
            step_command=step,
            true_heading=observation.eval_true_heading,
        )
        body = self.world.body(self._body_id)
        self.positions.append((body.x, body.y))

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.step()
