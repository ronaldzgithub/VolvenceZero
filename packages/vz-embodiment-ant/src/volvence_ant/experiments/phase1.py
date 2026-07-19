"""Phase 1 experiment: stigmergic collective foraging via the snapshot bus.

Runs a colony of simple foragers and shows that the shared pheromone snapshot
bus (externalised memory) drives collective convergence: with the bus, a
nest<->food trail corridor self-organises and delivery accelerates; without it,
the same ants forage independently and deliver less. This is the digital
analogue of the Bayesian-superorganism / stigmergy result, and it exercises the
SSOT snapshot bus (individuals communicate only through published snapshots).
"""

from __future__ import annotations

from dataclasses import dataclass

from volvence_ant.controllers.fixed_rule_ant import FixedRuleAnt, FixedRuleConfig
from volvence_ant.env.ant_world import AntWorld, AntWorldConfig, FoodSource
from volvence_ant.env.colony import ColonyWorld
from volvence_ant.env.pheromone_field import PheromoneBus
from volvence_ant.runtime import AntSessionConfig, KernelColonyRunner


@dataclass(frozen=True)
class ColonyForagingResult:
    with_bus_delivered: int
    without_bus_delivered: int
    with_bus_curve: tuple[int, ...]
    without_bus_curve: tuple[int, ...]
    with_bus_trail_mass: float
    trail_follow_events: int
    n_ants: int
    rounds: int
    bus_helps: bool
    description: str


@dataclass(frozen=True)
class KernelColonyArm:
    arm: str
    bus_enabled: bool
    delivered: int
    pickups: int
    trail_sense_events: int
    delivery_curve: tuple[int, ...]
    post_relocation_delivered: int


@dataclass(frozen=True)
class KernelColonyReport:
    seed: int
    n_ants: int
    rounds: int
    arms: tuple[KernelColonyArm, ...]
    learned_bus_effect: int


# Tuned so food is NOT smellable across the corridor (steep field), forcing
# exploration + trail-sharing. Emergent recruitment is stochastic: when scouts
# find food early a strong trail forms (big lift); when they don't, no trail and
# no advantage — exactly how real stigmergy behaves.
_FOOD = FoodSource(x=5.0, y=0.0, strength=2.5, decay=1.1, radius=2.0)
_ANT_CFG = dict(food_sense_threshold=0.18, trail_gain=14.0)


def _world_config(seed: int) -> AntWorldConfig:
    return AntWorldConfig(seed=seed, antenna_offset_deg=45.0, antenna_reach=1.3)


def _run_colony(*, world: AntWorld, n_ants: int, rounds: int, seed: int) -> tuple[list[int], int]:
    ants = [
        FixedRuleAnt(world, config=FixedRuleConfig(seed=seed * 100 + i, **_ANT_CFG), body_id=i)
        for i in range(n_ants)
    ]
    delivered_curve: list[int] = []
    trail_follow_events = 0
    for _ in range(rounds):
        for ant in ants:
            record = ant.step()
            trail_follow_events += record.mode == "trail-follow"
        delivered_curve.append(world.food_delivered)
    return delivered_curve, trail_follow_events


def colony_foraging_experiment(
    *,
    n_ants: int = 20,
    rounds: int = 700,
    seed: int = 0,
) -> ColonyForagingResult:
    food = (_FOOD,)

    colony_world = ColonyWorld(
        config=_world_config(seed),
        food_sources=food,
        n_bodies=n_ants,
        bus=PheromoneBus(decay=0.008, deposit_amount=2.5, cell_size=1.0),
    )
    with_curve, trail_follow_events = _run_colony(
        world=colony_world, n_ants=n_ants, rounds=rounds, seed=seed
    )
    trail_mass = colony_world.bus.total_mass()[1]
    plain_world = AntWorld(
        config=_world_config(seed), food_sources=food, n_bodies=n_ants
    )
    without_curve, _ = _run_colony(
        world=plain_world, n_ants=n_ants, rounds=rounds, seed=seed
    )
    with_delivered = colony_world.food_delivered
    without_delivered = plain_world.food_delivered
    return ColonyForagingResult(
        with_bus_delivered=with_delivered,
        without_bus_delivered=without_delivered,
        with_bus_curve=tuple(with_curve),
        without_bus_curve=tuple(without_curve),
        with_bus_trail_mass=trail_mass,
        trail_follow_events=trail_follow_events,
        n_ants=n_ants,
        rounds=rounds,
        bus_helps=with_delivered >= without_delivered,
        description=(
            f"colony foraging ({n_ants} ants, {rounds} rounds): "
            f"with_bus delivered={with_delivered}, without_bus delivered={without_delivered}, "
            f"trail_mass={trail_mass:.1f}, trail_follow_events={trail_follow_events}, "
            f"bus_helps={with_delivered >= without_delivered}"
        ),
    )


async def _run_kernel_colony_arm(
    *,
    seed: int,
    n_ants: int,
    rounds: int,
    bus_enabled: bool,
    base_config: AntSessionConfig,
) -> KernelColonyArm:
    world: AntWorld
    if bus_enabled:
        world = ColonyWorld(
            config=_world_config(seed),
            food_sources=(_FOOD,),
            n_bodies=n_ants,
            bus=PheromoneBus(decay=0.008, deposit_amount=2.5, cell_size=1.0),
        )
    else:
        world = AntWorld(
            config=_world_config(seed),
            food_sources=(_FOOD,),
            n_bodies=n_ants,
        )
    runner = KernelColonyRunner(world, base_config=base_config)
    curve: list[int] = []
    trail_events = 0
    delivered_at_relocation = 0
    for round_index in range(rounds):
        if round_index == rounds // 2:
            delivered_at_relocation = world.food_delivered
            world.move_food(x=-5.0, y=2.0)
        record = await runner.step_round()
        curve.append(record.delivered)
        trail_events += record.trail_sense_events
    return KernelColonyArm(
        arm="learned",
        bus_enabled=bus_enabled,
        delivered=world.food_delivered,
        pickups=world.food_pickups,
        trail_sense_events=trail_events,
        delivery_curve=tuple(curve),
        post_relocation_delivered=world.food_delivered - delivered_at_relocation,
    )


async def kernel_colony_foraging_experiment(
    *,
    n_ants: int = 4,
    rounds: int = 40,
    seed: int = 0,
    session_config: AntSessionConfig | None = None,
) -> KernelColonyReport:
    """Formal VZ colony lane; FixedRule remains a separate baseline."""

    base = session_config or AntSessionConfig(
        temporal_latent_dim=16,
        session_id=f"kernel-colony:{seed}",
        seed=seed,
    )
    with_bus = await _run_kernel_colony_arm(
        seed=seed,
        n_ants=n_ants,
        rounds=rounds,
        bus_enabled=True,
        base_config=base,
    )
    without_bus = await _run_kernel_colony_arm(
        seed=seed,
        n_ants=n_ants,
        rounds=rounds,
        bus_enabled=False,
        base_config=base,
    )
    return KernelColonyReport(
        seed=seed,
        n_ants=n_ants,
        rounds=rounds,
        arms=(with_bus, without_bus),
        learned_bus_effect=with_bus.delivered - without_bus.delivered,
    )
