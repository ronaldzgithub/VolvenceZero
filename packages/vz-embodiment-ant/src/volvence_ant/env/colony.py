"""``ColonyWorld`` — a multi-body world wired to the pheromone snapshot bus.

Individuals sense the pheromone field only through the published immutable
snapshot (via the inherited antenna sampling) and deposit only through additive
events. Depositing is a property of the BODY state (outbound bodies lay home
pheromone; food-carrying bodies lay trail pheromone), not of any inter-ant
call: no ant ever references another ant. Collective foraging convergence is
therefore pure stigmergy through the shared external memory.
"""

from __future__ import annotations

from volvence_ant.env.ant_world import AntBody, AntWorld, AntWorldConfig, FoodSource
from volvence_ant.env.pheromone_field import PheromoneBus, PheromoneField


class ColonyWorld(AntWorld):
    def __init__(
        self,
        *,
        config: AntWorldConfig | None = None,
        food_sources: tuple[FoodSource, ...] = (),
        n_bodies: int = 8,
        bus: PheromoneBus | None = None,
    ) -> None:
        self._bus = bus or PheromoneBus()
        super().__init__(config=config, food_sources=food_sources, n_bodies=n_bodies)

    @property
    def pheromone(self) -> PheromoneField:
        return self._bus.snapshot

    @property
    def bus(self) -> PheromoneBus:
        return self._bus

    def _pheromone_samples(self, x: float, y: float) -> tuple[float, float]:
        # read the CURRENT published immutable snapshot (SSOT read path)
        return self._bus.snapshot.sample(x, y)

    def _on_body_moved(self, body_id: int, body: AntBody) -> None:
        # additive deposit event; outbound bodies mark the way home, carrying
        # bodies mark the trail to food. No overwrite, no inter-ant call.
        if body.carrying_food:
            self._bus.deposit(x=body.x, y=body.y, trail_amount=1.0)
        else:
            self._bus.deposit(x=body.x, y=body.y, home_amount=1.0)

    def _on_round_complete(self) -> None:
        # aggregate deposits + decay -> publish next immutable snapshot
        self._bus.advance()
