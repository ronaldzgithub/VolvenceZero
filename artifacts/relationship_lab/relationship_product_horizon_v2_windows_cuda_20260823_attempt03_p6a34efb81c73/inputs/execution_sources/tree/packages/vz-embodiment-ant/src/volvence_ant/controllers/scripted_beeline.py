"""``ScriptedBeelineAnt`` — a hardcoded waypoint forager (the brittle arm).

This is the caricature of the traditional approach the whole project argues
against: the forager route is *baked in* at authoring time as a fixed waypoint
(the coordinate where food was when the programmer wrote the script). The ant
beelines to that waypoint, grabs whatever is there, and beelines home. It never
senses the odour field and never adapts.

It is competent — even optimal — as long as the world matches the author's
assumption. The moment food *moves* (a perturbation the author did not foresee)
the baked-in waypoint points at empty ground and delivery collapses to zero.
That failure, contrasted with the stigmergic colony that re-forms its trail via
decay + redeposit, is the emergent-vs-hardcoded demo (Workstream G2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from volvence_ant.env.ant_world import AntWorld, TWO_PI


@dataclass(frozen=True)
class BeelineStep:
    tick: int
    x: float
    y: float
    carrying_food: bool
    mode: str


class ScriptedBeelineAnt:
    """Hardcoded waypoint follower sharing the frozen ant world/plant."""

    def __init__(
        self,
        world: AntWorld,
        *,
        food_waypoint: tuple[float, float],
        body_id: int = 0,
    ) -> None:
        self.world = world
        self._body_id = body_id
        # The route is baked in at authoring time: this is the whole point.
        self._waypoint = food_waypoint
        self.trajectory: list[BeelineStep] = []

    def _steer_towards(self, tx: float, ty: float) -> float:
        body = self.world.body(self._body_id)
        desired = math.atan2(ty - body.y, tx - body.x)
        rel = (desired - body.heading + math.pi) % TWO_PI - math.pi
        return float(np.clip(rel, -self.world.config.max_turn_rate, self.world.config.max_turn_rate))

    def step(self) -> BeelineStep:
        cfg = self.world.config
        body = self.world.body(self._body_id)
        if body.carrying_food:
            target = (cfg.nest_x, cfg.nest_y)
            mode = "return-to-nest"
        else:
            target = self._waypoint
            mode = "beeline-to-baked-waypoint"
        turn = self._steer_towards(*target)
        obs = self.world.act(turn_command=turn, step_command=cfg.step_size, body_id=self._body_id)
        record = BeelineStep(
            tick=self.world.tick,
            x=self.world.body(self._body_id).x,
            y=self.world.body(self._body_id).y,
            carrying_food=obs.carrying_food,
            mode=mode,
        )
        self.trajectory.append(record)
        return record

    def run(self, ticks: int) -> list[BeelineStep]:
        for _ in range(ticks):
            self.step()
        return self.trajectory
