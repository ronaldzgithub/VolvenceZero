"""Ant-context behavioural matched-control (Workstream E).

Runs several controllers on ONE shared environment configuration + seed and
reports directional behavioural metrics. Honest scope: at the toy tick budgets
used here, the authoritative "learning is real" evidence is the latent proof
(``latent_proofs``); this behavioural comparison is a directional, visual
complement (it also feeds the emergent-vs-scripted demo, Workstream G2).

Boundary-clean arms (expressible via the vz-runtime facade only):

- ``learned``   — full kernel ant (PE drive on, temporal ACTIVE)
- ``pe_off``    — kernel ant with ``external_prediction_error_drive=False``
- ``fixed_rule``— hand-written FSM forager
- ``random``    — random-motor floor

Additional schedule-gated arms (``no_optimize`` / ``eta_off``) require a
``JointLoopSchedule`` object, which the embodiment package must not import; the
evidence lane script passes one in via ``extra_kernel_arms``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from volvence_ant.controllers.e2e_rl_ant import E2ERLAnt, PPOConfig
from volvence_ant.controllers.fixed_rule_ant import FixedRuleAnt, FixedRuleConfig
from volvence_ant.controllers.random_ant import RandomAnt
from volvence_ant.env.ant_world import AntWorld, AntWorldConfig, FoodSource
from volvence_ant.runtime.ant_session import AntSession, AntSessionConfig


@dataclass(frozen=True)
class ArmMetrics:
    arm: str
    ticks: int
    food_delivered: int
    food_pickups: int
    mean_food_experienced: float
    max_food_experienced: float
    max_distance_from_nest: float
    final_distance_from_nest: float
    held_out_success: bool


@dataclass(frozen=True)
class MatchedControlReport:
    ticks: int
    seed: int
    arms: tuple[ArmMetrics, ...]
    learned_beats_random_food: bool
    description: str


@dataclass(frozen=True)
class ArmAggregate:
    arm: str
    seeds: tuple[int, ...]
    mean_delivered: float
    delivery_ci95: tuple[float, float]
    held_out_success_rate: float


@dataclass(frozen=True)
class MultiSeedMatchedControlReport:
    reports: tuple[MatchedControlReport, ...]
    aggregates: tuple[ArmAggregate, ...]
    learned_minus_no_optimize: float | None


def _default_world(seed: int) -> AntWorld:
    """Held-out evaluation map shared by every arm."""

    return AntWorld(
        config=AntWorldConfig(seed=seed),
        food_sources=(FoodSource(x=0.0, y=6.0, strength=1.0, decay=5.0),),
    )


def _training_world(seed: int) -> AntWorld:
    return AntWorld(
        config=AntWorldConfig(seed=seed),
        food_sources=(FoodSource(x=6.0, y=0.0, strength=1.0, decay=5.0),),
    )


def _metrics_from_positions(
    *, arm: str, world: AntWorld, positions: list[tuple[float, float]], ticks: int
) -> ArmMetrics:
    nest = world.nest
    foods = [world.food_intensity(x, y) for (x, y) in positions] or [0.0]
    distances = [math.hypot(x - nest[0], y - nest[1]) for (x, y) in positions] or [0.0]
    final = distances[-1] if distances else 0.0
    return ArmMetrics(
        arm=arm,
        ticks=ticks,
        food_delivered=world.food_delivered,
        food_pickups=world.food_pickups,
        mean_food_experienced=float(sum(foods) / len(foods)),
        max_food_experienced=float(max(foods)),
        max_distance_from_nest=float(max(distances)),
        final_distance_from_nest=float(final),
        held_out_success=world.food_delivered > 0,
    )


async def _run_kernel_arm(
    *, arm: str, seed: int, ticks: int, session_config: AntSessionConfig
) -> ArmMetrics:
    world = _default_world(seed)
    session = AntSession(world, config=session_config)
    records = await session.run(ticks)
    positions = [(r.x, r.y) for r in records]
    return _metrics_from_positions(arm=arm, world=world, positions=positions, ticks=ticks)


def _run_fixed_rule_arm(*, seed: int, ticks: int) -> ArmMetrics:
    world = _default_world(seed)
    ant = FixedRuleAnt(world, config=FixedRuleConfig(seed=seed))
    records = ant.run(ticks)
    positions = [(r.x, r.y) for r in records]
    return _metrics_from_positions(arm="fixed_rule", world=world, positions=positions, ticks=ticks)


def _run_random_arm(*, seed: int, ticks: int) -> ArmMetrics:
    world = _default_world(seed)
    ant = RandomAnt(world, seed=seed)
    ant.run(ticks)
    return _metrics_from_positions(arm="random", world=world, positions=ant.positions, ticks=ticks)


def _run_e2e_arm(
    *,
    seed: int,
    ticks: int,
    train_episodes: int,
    train_ticks: int,
) -> ArmMetrics:
    policy = E2ERLAnt(seed=seed)
    policy.train(
        world_factory=_training_world,
        seed=seed,
        config=PPOConfig(
            episodes=train_episodes,
            ticks_per_episode=train_ticks,
        ),
    )
    world = _default_world(seed)
    evaluation = policy.evaluate(world=world, ticks=ticks, seed=seed)
    return _metrics_from_positions(
        arm="e2e_rl",
        world=world,
        positions=list(evaluation.positions),
        ticks=ticks,
    )


async def run_behavioral_matched_control(
    *,
    ticks: int = 60,
    seed: int = 0,
    temporal_latent_dim: int = 4,
    learned_config: AntSessionConfig | None = None,
    pe_off_config: AntSessionConfig | None = None,
    extra_kernel_arms: dict[str, AntSessionConfig] | None = None,
    include_e2e_rl: bool = False,
    e2e_train_episodes: int = 4,
    e2e_train_ticks: int = 64,
) -> MatchedControlReport:
    """Run the behavioural arms on a shared env/seed and collect metrics."""

    arms: list[ArmMetrics] = []

    learned_cfg = learned_config or AntSessionConfig(
        temporal_latent_dim=temporal_latent_dim,
        seed=seed,
        external_prediction_error_drive=True,
    )
    arms.append(await _run_kernel_arm(arm="learned", seed=seed, ticks=ticks, session_config=learned_cfg))

    pe_off_cfg = pe_off_config or AntSessionConfig(
        temporal_latent_dim=temporal_latent_dim,
        seed=seed,
        external_prediction_error_drive=False,
    )
    arms.append(await _run_kernel_arm(arm="pe_off", seed=seed, ticks=ticks, session_config=pe_off_cfg))

    for arm_name, cfg in (extra_kernel_arms or {}).items():
        arms.append(
            await _run_kernel_arm(arm=arm_name, seed=seed, ticks=ticks, session_config=cfg)
        )

    arms.append(_run_fixed_rule_arm(seed=seed, ticks=ticks))
    if include_e2e_rl:
        arms.append(
            _run_e2e_arm(
                seed=seed,
                ticks=ticks,
                train_episodes=e2e_train_episodes,
                train_ticks=e2e_train_ticks,
            )
        )
    arms.append(_run_random_arm(seed=seed, ticks=ticks))

    by_arm = {arm.arm: arm for arm in arms}
    learned_food = by_arm["learned"].mean_food_experienced
    random_food = by_arm["random"].mean_food_experienced
    return MatchedControlReport(
        ticks=ticks,
        seed=seed,
        arms=tuple(arms),
        learned_beats_random_food=learned_food >= random_food,
        description=(
            "behavioural matched-control ("
            + ", ".join(f"{a.arm}:deliver={a.food_delivered},food={a.mean_food_experienced:.3f}" for a in arms)
            + ")"
        ),
    )


def _bootstrap_ci(values: tuple[float, ...], *, seed: int) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap values must be non-empty")
    if len(values) == 1:
        return values[0], values[0]
    rng = np.random.default_rng(seed)
    samples = np.asarray(values, dtype=float)
    means = np.asarray(
        [
            rng.choice(samples, size=len(samples), replace=True).mean()
            for _ in range(2000)
        ]
    )
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


async def run_multiseed_matched_control(
    *,
    seeds: tuple[int, ...],
    ticks: int,
    temporal_latent_dim: int,
    kernel_arm_factory: Callable[[int, int], dict[str, AntSessionConfig]],
    learned_config_factory: Callable[[int, int], AntSessionConfig] | None = None,
    pe_off_config_factory: Callable[[int, int], AntSessionConfig] | None = None,
    include_e2e_rl: bool = True,
) -> MultiSeedMatchedControlReport:
    """Run a caller-defined fair kernel matrix over a frozen seed schedule."""

    if len(seeds) < 1:
        raise ValueError("seeds must be non-empty")
    reports: list[MatchedControlReport] = []
    for seed in seeds:
        reports.append(
            await run_behavioral_matched_control(
                ticks=ticks,
                seed=seed,
                temporal_latent_dim=temporal_latent_dim,
                learned_config=(
                    learned_config_factory(seed, temporal_latent_dim)
                    if learned_config_factory is not None
                    else None
                ),
                pe_off_config=(
                    pe_off_config_factory(seed, temporal_latent_dim)
                    if pe_off_config_factory is not None
                    else None
                ),
                extra_kernel_arms=kernel_arm_factory(seed, temporal_latent_dim),
                include_e2e_rl=include_e2e_rl,
            )
        )
    arm_names = tuple(arm.arm for arm in reports[0].arms)
    aggregates: list[ArmAggregate] = []
    for arm_name in arm_names:
        per_seed = tuple(
            next(arm for arm in report.arms if arm.arm == arm_name)
            for report in reports
        )
        deliveries = tuple(float(arm.food_delivered) for arm in per_seed)
        aggregates.append(
            ArmAggregate(
                arm=arm_name,
                seeds=seeds,
                mean_delivered=float(np.mean(deliveries)),
                delivery_ci95=_bootstrap_ci(deliveries, seed=seeds[0]),
                held_out_success_rate=float(
                    np.mean([arm.held_out_success for arm in per_seed])
                ),
            )
        )
    by_name = {aggregate.arm: aggregate for aggregate in aggregates}
    effect = None
    if "no_optimize" in by_name:
        effect = (
            by_name["learned"].mean_delivered
            - by_name["no_optimize"].mean_delivered
        )
    return MultiSeedMatchedControlReport(
        reports=tuple(reports),
        aggregates=tuple(aggregates),
        learned_minus_no_optimize=effect,
    )
