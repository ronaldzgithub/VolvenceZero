"""Workstream G2 — emergent/adaptive vs hardcoded script under perturbation.

Two colonies forage the same world. Half-way through, the food is *relocated*
(a perturbation neither controller was told about):

- **Emergent/adaptive arm**: foragers that *sense* the odour field and ascend
  its gradient (plus path-integration homing). When food moves they simply
  smell the new source and adapt — delivery *keeps going*.
- **Hardcoded arm**: ``ScriptedBeelineAnt``s whose route is baked to the old
  food coordinate. Before the move they are near-optimal; after it they keep
  marching to empty ground and delivery *collapses to exactly zero*.

The contrast is the whole argument of the project made visible: behaviour that
adapts from sensing degrades gracefully; behaviour hand-wired to the author's
assumptions shatters on the first unforeseen change. (The efficient-but-brittle
hardcoded arm even out-delivers the adaptive one *while its assumptions hold* —
which is exactly why hardcoding is so tempting, and so dangerous.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from volvence_ant.controllers.fixed_rule_ant import FixedRuleAnt, FixedRuleConfig
from volvence_ant.controllers.scripted_beeline import ScriptedBeelineAnt
from volvence_ant.controllers.random_ant import RandomAnt
from volvence_ant.env.ant_world import AntWorld, AntWorldConfig, FoodSource
from volvence_ant.runtime import AntSessionConfig, KernelColonyRunner
from volvence_ant.viz.render import save_line_overlay

_FOOD_A = (6.0, 0.0)
_FOOD_B = (-4.0, 4.0)


def _world_config(seed: int) -> AntWorldConfig:
    return AntWorldConfig(seed=seed, antenna_offset_deg=30.0, antenna_reach=0.9)


@dataclass(frozen=True)
class PerturbationArm:
    label: str
    delivered_before: int
    delivered_after: int  # deliveries in the final window AFTER relocation
    curve: tuple[int, ...]
    tracks: tuple[tuple[tuple[float, float], ...], ...]


@dataclass(frozen=True)
class PerturbationReport:
    relocate_at: int
    total_rounds: int
    emergent: PerturbationArm
    hardcoded: PerturbationArm
    emergent_recovered: bool
    hardcoded_collapsed: bool
    description: str
    figure_path: str | None


@dataclass(frozen=True)
class FormalPerturbationReport:
    relocate_at: int
    total_rounds: int
    arms: tuple[PerturbationArm, ...]
    learned_recovered: bool
    figure_path: str | None


def _food(at: tuple[float, float]) -> FoodSource:
    return FoodSource(x=at[0], y=at[1], strength=1.5, decay=5.0, radius=1.5)


def _run_emergent(*, n_ants: int, rounds: int, relocate_at: int, seed: int) -> PerturbationArm:
    world = AntWorld(
        config=_world_config(seed), food_sources=(_food(_FOOD_A),), n_bodies=n_ants
    )
    ants = [
        FixedRuleAnt(
            world,
            config=FixedRuleConfig(seed=seed * 100 + i, food_sense_threshold=0.02, gradient_gain=6.0),
            body_id=i,
        )
        for i in range(n_ants)
    ]
    return _drive(world, ants, rounds=rounds, relocate_at=relocate_at, label="adaptive (senses odour)")


def _run_hardcoded(*, n_ants: int, rounds: int, relocate_at: int, seed: int) -> PerturbationArm:
    world = AntWorld(
        config=_world_config(seed), food_sources=(_food(_FOOD_A),), n_bodies=n_ants
    )
    ants = [ScriptedBeelineAnt(world, food_waypoint=_FOOD_A, body_id=i) for i in range(n_ants)]
    return _drive(world, ants, rounds=rounds, relocate_at=relocate_at, label="hardcoded (baked route)")


def _run_random(*, n_ants: int, rounds: int, relocate_at: int, seed: int) -> PerturbationArm:
    world = AntWorld(
        config=_world_config(seed), food_sources=(_food(_FOOD_A),), n_bodies=n_ants
    )
    ants = [RandomAnt(world, seed=seed * 100 + i, body_id=i) for i in range(n_ants)]
    return _drive(world, ants, rounds=rounds, relocate_at=relocate_at, label="random")


def _drive(world, ants, *, rounds: int, relocate_at: int, label: str) -> PerturbationArm:
    curve: list[int] = []
    tracks: list[list[tuple[float, float]]] = [[] for _ in ants]
    delivered_before = 0
    for r in range(rounds):
        if r == relocate_at:
            delivered_before = world.food_delivered
            world.move_food(index=0, x=_FOOD_B[0], y=_FOOD_B[1])
        for i, ant in enumerate(ants):
            ant.step()
            body = world.body(i)
            tracks[i].append((body.x, body.y))
        curve.append(world.food_delivered)
    # Measure recovery in a FINAL window only, so deliveries from food already
    # in-flight at the moment of relocation do not count as post-move foraging.
    final_window = max(1, (rounds - relocate_at) // 2)
    idx = rounds - final_window - 1
    delivered_after = curve[-1] - curve[idx]
    return PerturbationArm(
        label=label,
        delivered_before=delivered_before,
        delivered_after=delivered_after,
        curve=tuple(curve),
        tracks=tuple(tuple(t) for t in tracks),
    )


def run_perturbation_demo(
    *,
    n_ants: int = 16,
    rounds: int = 1200,
    relocate_at: int = 600,
    seed: int = 0,
    figure_path: Path | None = None,
) -> PerturbationReport:
    emergent = _run_emergent(n_ants=n_ants, rounds=rounds, relocate_at=relocate_at, seed=seed)
    hardcoded = _run_hardcoded(n_ants=n_ants, rounds=rounds, relocate_at=relocate_at, seed=seed)

    emergent_recovered = emergent.delivered_after > 0
    hardcoded_collapsed = hardcoded.delivered_after == 0

    fig_out: str | None = None
    if figure_path is not None:
        rounds_axis = list(range(1, rounds + 1))
        reloc_marker = [relocate_at, relocate_at]
        top = max(emergent.curve[-1], hardcoded.curve[-1], 1)
        saved = save_line_overlay(
            series=[
                {"x": rounds_axis, "y": list(hardcoded.curve),
                 "label": "hardcoded (baked route)", "style": "-"},
                {"x": rounds_axis, "y": list(emergent.curve),
                 "label": "adaptive (senses odour)", "style": "-"},
                {"x": reloc_marker, "y": [0, top],
                 "label": "food relocated", "style": "k--"},
            ],
            x_label="round",
            y_label="cumulative food delivered",
            title="G2 under perturbation: hardcoded flatlines, adaptive keeps delivering",
            out_path=figure_path,
        )
        fig_out = str(saved) if saved is not None else None

    return PerturbationReport(
        relocate_at=relocate_at,
        total_rounds=rounds,
        emergent=emergent,
        hardcoded=hardcoded,
        emergent_recovered=emergent_recovered,
        hardcoded_collapsed=hardcoded_collapsed,
        description=(
            f"food relocated at round {relocate_at}: "
            f"adaptive delivered_after={emergent.delivered_after} (recovered={emergent_recovered}); "
            f"hardcoded delivered_after={hardcoded.delivered_after} (collapsed={hardcoded_collapsed})"
        ),
        figure_path=fig_out,
    )


async def run_formal_perturbation_demo(
    *,
    session_config: AntSessionConfig,
    n_ants: int = 4,
    rounds: int = 120,
    relocate_at: int = 60,
    seed: int = 0,
    figure_path: Path | None = None,
) -> FormalPerturbationReport:
    """Formal G2: trained AntSession against matched rule/script/random arms."""

    world = AntWorld(
        config=_world_config(seed),
        food_sources=(_food(_FOOD_A),),
        n_bodies=n_ants,
    )
    runner = KernelColonyRunner(world, base_config=session_config)
    curve: list[int] = []
    tracks: list[list[tuple[float, float]]] = [[] for _ in range(n_ants)]
    delivered_before = 0
    for round_index in range(rounds):
        if round_index == relocate_at:
            delivered_before = world.food_delivered
            world.move_food(x=_FOOD_B[0], y=_FOOD_B[1])
        await runner.step_round()
        for body_id in range(n_ants):
            tracks[body_id].append(world.body(body_id).position)
        curve.append(world.food_delivered)
    final_window = max(1, (rounds - relocate_at) // 2)
    learned = PerturbationArm(
        label="trained AntSession",
        delivered_before=delivered_before,
        delivered_after=curve[-1] - curve[rounds - final_window - 1],
        curve=tuple(curve),
        tracks=tuple(tuple(track) for track in tracks),
    )
    fixed = _run_emergent(
        n_ants=n_ants,
        rounds=rounds,
        relocate_at=relocate_at,
        seed=seed,
    )
    fixed = PerturbationArm(
        label="FixedRule",
        delivered_before=fixed.delivered_before,
        delivered_after=fixed.delivered_after,
        curve=fixed.curve,
        tracks=fixed.tracks,
    )
    scripted = _run_hardcoded(
        n_ants=n_ants,
        rounds=rounds,
        relocate_at=relocate_at,
        seed=seed,
    )
    random = _run_random(
        n_ants=n_ants,
        rounds=rounds,
        relocate_at=relocate_at,
        seed=seed,
    )
    arms = (learned, fixed, scripted, random)
    figure_out = None
    if figure_path is not None:
        saved = save_line_overlay(
            series=[
                {
                    "x": list(range(1, rounds + 1)),
                    "y": list(arm.curve),
                    "label": arm.label,
                    "style": "-",
                }
                for arm in arms
            ],
            x_label="round",
            y_label="cumulative food delivered",
            title="Formal G2: learned vs matched controls under relocation",
            out_path=figure_path,
        )
        figure_out = str(saved) if saved is not None else None
    return FormalPerturbationReport(
        relocate_at=relocate_at,
        total_rounds=rounds,
        arms=arms,
        learned_recovered=learned.delivered_after > 0,
        figure_path=figure_out,
    )
