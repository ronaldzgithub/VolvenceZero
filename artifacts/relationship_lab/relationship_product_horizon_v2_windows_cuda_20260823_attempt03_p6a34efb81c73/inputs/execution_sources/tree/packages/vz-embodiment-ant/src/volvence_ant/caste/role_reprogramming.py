"""Offline caste (role) reprogramming — rare-heavy analogue.

Given an :class:`EnvironmentPressure`, an OFFLINE search evaluates candidate
role distributions (what fraction of the colony are explorers vs patrollers) by
simulating short foraging episodes, and returns the yield-maximising
:class:`CasteProfile`. Nothing is hardcoded: the role mix that comes out is
whatever the environment rewards, so it shifts systematically with pressure
(scarce/steep food -> more explorers; abundant/predator-heavy -> more
patrollers). This is the R2 middle tier: a slow, offline reprogramming that
sets bounded init params the online controller then lives within.

Runtime code must NOT call :func:`reprogram_castes` — it is a heavy offline
step (like a rare-heavy artifact refresh), guarded by ``allow_offline=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from volvence_ant.controllers.fixed_rule_ant import FixedRuleAnt, FixedRuleConfig
from volvence_ant.env.ant_world import AntWorldConfig, FoodSource
from volvence_ant.env.colony import ColonyWorld
from volvence_ant.env.pheromone_field import PheromoneBus


@dataclass(frozen=True)
class EnvironmentPressure:
    """Immutable description of the environment the colony is reprogrammed for."""

    food_distance: float = 5.0
    food_strength: float = 2.5
    food_decay: float = 1.1  # smaller -> steeper -> harder to find -> exploration pays
    food_radius: float = 2.0
    predation: float = 0.0  # [0,1]; higher penalises wide roaming (explorers)
    label: str = "baseline"


@dataclass(frozen=True)
class CasteProfile:
    """Immutable, offline-produced role-assignment artifact (rare-heavy analogue)."""

    schema_version: int
    pressure_label: str
    explorer_fraction: float
    n_individuals: int
    exploration_bias_by_individual: tuple[float, ...]
    expected_yield: float
    provenance: str = "offline-caste-reprogramming"

    def config_for(self, individual: int, *, base: FixedRuleConfig | None = None) -> FixedRuleConfig:
        bias = self.exploration_bias_by_individual[individual]
        base = base or FixedRuleConfig(seed=individual)
        return FixedRuleConfig(
            seed=base.seed,
            heading_noise=base.heading_noise,
            step_noise=base.step_noise,
            explore_jitter=base.explore_jitter,
            gradient_gain=base.gradient_gain,
            food_sense_threshold=base.food_sense_threshold,
            trail_gain=base.trail_gain,
            trail_follow_threshold=base.trail_follow_threshold,
            panic_flee_speed=base.panic_flee_speed,
            exploration_bias=bias,
        )


@dataclass(frozen=True)
class ReprogrammingResult:
    profiles: tuple[CasteProfile, ...]
    role_shift_monotone: bool
    description: str
    yield_grid: dict[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)


_EXPLORER_BIAS = 0.95
_PATROLLER_BIAS = 0.15


def _assign_biases(*, explorer_fraction: float, n: int, rng: np.random.Generator) -> tuple[float, ...]:
    n_explorers = int(round(explorer_fraction * n))
    biases = [_EXPLORER_BIAS] * n_explorers + [_PATROLLER_BIAS] * (n - n_explorers)
    rng.shuffle(biases)
    return tuple(float(b) for b in biases)


def _evaluate_yield(
    *,
    pressure: EnvironmentPressure,
    explorer_fraction: float,
    n_individuals: int,
    rounds: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    biases = _assign_biases(explorer_fraction=explorer_fraction, n=n_individuals, rng=rng)
    world = ColonyWorld(
        config=AntWorldConfig(seed=seed, antenna_offset_deg=45.0, antenna_reach=1.3),
        food_sources=(
            FoodSource(
                x=pressure.food_distance,
                y=0.0,
                strength=pressure.food_strength,
                decay=pressure.food_decay,
                radius=pressure.food_radius,
            ),
        ),
        n_bodies=n_individuals,
        bus=PheromoneBus(decay=0.008, deposit_amount=2.5, cell_size=1.0),
    )
    ants = [
        FixedRuleAnt(
            world,
            config=FixedRuleConfig(
                seed=seed * 100 + i,
                food_sense_threshold=0.18,
                trail_gain=14.0,
                exploration_bias=biases[i],
            ),
            body_id=i,
        )
        for i in range(n_individuals)
    ]
    biases_by_body = {i: ants[i].config.exploration_bias for i in range(n_individuals)}
    roam_cost = 0.0
    for _ in range(rounds):
        for body_id, ant in enumerate(ants):
            record = ant.step()
            # predation risk scales with how much this individual roams: an
            # explorer (high bias) far from the nest takes more risk than a
            # patroller. So more explorers -> more risk -> fewer explorers wins.
            if pressure.predation > 0.0:
                dist = (record.x ** 2 + record.y ** 2) ** 0.5
                roam_cost += (
                    pressure.predation
                    * biases_by_body[body_id]
                    * 0.012
                    * max(0.0, dist - 3.0)
                )
    return float(world.food_delivered) - roam_cost


def reprogram_castes(
    *,
    pressures: tuple[EnvironmentPressure, ...],
    n_individuals: int = 16,
    fraction_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    rounds: int = 500,
    seed: int = 0,
    n_seeds: int = 2,
    allow_offline: bool = False,
) -> ReprogrammingResult:
    """Offline: for each pressure, pick the yield-maximising explorer fraction.

    ``allow_offline`` must be True — a guard mirroring rare-heavy artifact
    refresh, which the online-fast runtime is forbidden to trigger.
    """

    if not allow_offline:
        raise RuntimeError(
            "reprogram_castes is an OFFLINE rare-heavy step; runtime must not "
            "trigger it. Pass allow_offline=True from an offline pipeline."
        )

    profiles: list[CasteProfile] = []
    yield_grid: dict[str, tuple[tuple[float, float], ...]] = {}
    for pressure in pressures:
        scored: list[tuple[float, float]] = []
        for fraction in fraction_grid:
            # average over a couple of seeds to reduce discovery noise
            yields = [
                _evaluate_yield(
                    pressure=pressure,
                    explorer_fraction=fraction,
                    n_individuals=n_individuals,
                    rounds=rounds,
                    seed=seed + s,
                )
                for s in range(max(1, n_seeds))
            ]
            scored.append((fraction, float(np.mean(yields))))
        yield_grid[pressure.label] = tuple(scored)
        best_fraction, best_yield = max(scored, key=lambda item: item[1])
        rng = np.random.default_rng(seed + 777)
        profiles.append(
            CasteProfile(
                schema_version=1,
                pressure_label=pressure.label,
                explorer_fraction=best_fraction,
                n_individuals=n_individuals,
                exploration_bias_by_individual=_assign_biases(
                    explorer_fraction=best_fraction, n=n_individuals, rng=rng
                ),
                expected_yield=best_yield,
            )
        )

    # systematic shift check: scarcer / steeper food should not REDUCE the
    # chosen explorer fraction relative to abundant food.
    by_label = {p.pressure_label: p.explorer_fraction for p in profiles}
    monotone = True
    if "abundant" in by_label and "scarce" in by_label:
        monotone = by_label["scarce"] >= by_label["abundant"]

    return ReprogrammingResult(
        profiles=tuple(profiles),
        role_shift_monotone=monotone,
        description=(
            "caste reprogramming explorer_fraction by pressure: "
            + ", ".join(f"{p.pressure_label}={p.explorer_fraction:.2f}(yield={p.expected_yield:.1f})" for p in profiles)
        ),
        yield_grid=yield_grid,
    )
