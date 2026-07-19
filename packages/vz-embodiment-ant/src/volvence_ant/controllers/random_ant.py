"""``RandomAnt`` — random-motor lower bound (no learning, no rules).

The floor baseline for the matched-control study: it issues random bounded
turns and constant steps, sharing the same frozen substrate. Any competent
controller (learned or hand-written) must beat it.
"""

from __future__ import annotations

import numpy as np

from volvence_ant.env.ant_world import AntWorld
from volvence_ant.substrate.navigator import AntNavigator


class RandomAnt:
    def __init__(self, world: AntWorld, *, seed: int = 0, body_id: int = 0) -> None:
        self.world = world
        self._body_id = body_id
        self._rng = np.random.default_rng(seed)
        self.navigator = AntNavigator(step_size=world.config.step_size, seed=seed)
        self.navigator.reset(initial_heading=world.observe(body_id).eval_true_heading)
        self.positions: list[tuple[float, float]] = []

    def step(self) -> None:
        max_turn = self.world.config.max_turn_rate
        step = self.world.config.step_size
        turn = float(self._rng.uniform(-max_turn, max_turn))
        self.navigator.update(turn_command=turn, step_command=step)
        self.world.act(turn_command=turn, step_command=step, body_id=self._body_id)
        body = self.world.body(self._body_id)
        self.positions.append((body.x, body.y))

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.step()
