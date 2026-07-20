"""Offline ecology curriculum over the real ant runtime.

No action labels or steering hints are supplied.  Every episode runs the same
``AntSession`` sense -> temporal -> action -> outcome -> PE -> credit path used
by the live app, then carries owner-exported checkpoints into the next world.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from volvence_ant.env.ant_world import AntWorld, AntWorldConfig
from volvence_ant.env.colony import ColonyWorld
from volvence_ant.env.world_objects import (
    BurningMatch,
    ButterSource,
    WoodStick,
    WorldObject,
)
from volvence_ant.evidence.runtime_profile import (
    ant_runtime_replay_rollout_config,
)
from volvence_ant.runtime import (
    AntLearningCheckpoint,
    AntObjectiveKind,
    AntSenseSchema,
    AntSessionConfig,
    KernelColonyRunner,
)


ECOLOGY_CURRICULUM_SCHEMA_VERSION = "digital-ant-ecology-curriculum.v1"
ECOLOGY_REQUIRED_GATE_NAMES = (
    "policy_changed",
    "no_optimize_policy_stable",
    "butter_pickup_gain",
    "butter_delivery_present",
    "wood_stick_avoidance",
    "burning_match_avoidance",
    "composite_performance",
    "runtime_replay_lineage",
)


class EcologyStage(str, Enum):
    BUTTER = "butter"
    WOOD_STICK = "wood_stick"
    BURNING_MATCH = "burning_match"
    COMPOSITE = "composite"


@dataclass(frozen=True)
class EcologyCurriculumConfig:
    n_ants: int = 8
    temporal_latent_dim: int = 16
    stage_rounds: int = 80
    stage_episodes: int = 3
    heldout_rounds: int = 120
    heldout_seeds: tuple[int, ...] = (101, 211, 307, 401, 503)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_ants < 1:
            raise ValueError("n_ants must be >= 1")
        if self.temporal_latent_dim < 3:
            raise ValueError("temporal_latent_dim must be >= 3")
        if self.stage_rounds < 1 or self.stage_episodes < 1:
            raise ValueError("stage budgets must be >= 1")
        if self.heldout_rounds < 1 or not self.heldout_seeds:
            raise ValueError("heldout budget and seeds must be non-empty")


@dataclass(frozen=True)
class EcologyArmMetrics:
    arm: str
    stage: EcologyStage
    seed: int
    pickups: int
    deliveries: int
    obstacle_contacts: int
    harmful_heat_ticks: int
    heat_entries: int
    heat_escapes: int
    applied_distance: float
    replay_captured: int
    replay_settled: int
    replay_lineage_matches: int
    replay_pending_captures: int
    replay_staged_rollouts: int
    replay_drop_count: int
    replay_settlement_coverage: float
    replay_lineage_coverage: float


@dataclass(frozen=True)
class EcologyGate:
    name: str
    passed: bool
    observed: str
    threshold: str


@dataclass(frozen=True)
class EcologyCheckpointReport:
    schema_version: str
    config: EcologyCurriculumConfig
    initial_policy_fingerprints: tuple[str, ...]
    learned_policy_fingerprints: tuple[str, ...]
    no_optimize_policy_fingerprints: tuple[str, ...]
    learned_metrics: tuple[EcologyArmMetrics, ...]
    cold_metrics: tuple[EcologyArmMetrics, ...]
    no_optimize_metrics: tuple[EcologyArmMetrics, ...]
    gates: tuple[EcologyGate, ...]
    verdict: str
    diagnostic_breakpoints: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EcologyCheckpointCandidate:
    checkpoints: tuple[AntLearningCheckpoint, ...]
    checkpoint_archives: tuple[bytes, ...]
    report: EcologyCheckpointReport


def _scene_objects(
    *,
    stage: EcologyStage,
    seed: int,
    heldout: bool,
) -> tuple[WorldObject, ...]:
    rng = np.random.default_rng(seed + (1_000_003 if heldout else 0))
    bearing = float(rng.uniform(-math.pi, math.pi))
    distance = float(rng.uniform(3.0, 4.4))
    food_x = math.cos(bearing) * distance
    food_y = math.sin(bearing) * distance
    objects: list[WorldObject] = [
        ButterSource(
            object_id="butter",
            x=food_x,
            y=food_y,
            strength=2.2,
            decay=2.4,
            radius=1.1,
        )
    ]
    if stage in {EcologyStage.WOOD_STICK, EcologyStage.COMPOSITE}:
        centre_x = food_x * 0.55
        centre_y = food_y * 0.55
        perpendicular = bearing + math.pi / 2.0
        half_length = 1.35
        objects.append(
            WoodStick(
                object_id="wood-stick",
                start_x=centre_x + math.cos(perpendicular) * half_length,
                start_y=centre_y + math.sin(perpendicular) * half_length,
                end_x=centre_x - math.cos(perpendicular) * half_length,
                end_y=centre_y - math.sin(perpendicular) * half_length,
                radius=0.2,
            )
        )
    if stage in {EcologyStage.BURNING_MATCH, EcologyStage.COMPOSITE}:
        offset_sign = -1.0 if seed % 2 else 1.0
        match_bearing = bearing + offset_sign * 0.28
        match_distance = distance * 0.48
        objects.append(
            BurningMatch(
                object_id="burning-match",
                x=math.cos(match_bearing) * match_distance,
                y=math.sin(match_bearing) * match_distance,
                angle=bearing,
                heat_strength=1.0,
                heat_decay=1.5,
                harm_threshold=0.55,
            )
        )
    return tuple(objects)


def _world(
    *,
    config: EcologyCurriculumConfig,
    stage: EcologyStage,
    seed: int,
    heldout: bool,
) -> ColonyWorld:
    return ColonyWorld(
        config=AntWorldConfig(
            seed=seed + (2_000_003 if heldout else 17),
            antenna_offset_deg=45.0,
            antenna_reach=0.9,
        ),
        world_objects=_scene_objects(stage=stage, seed=seed, heldout=heldout),
        n_bodies=config.n_ants,
    )


def _session_config(
    *,
    config: EcologyCurriculumConfig,
    seed: int,
    session_id: str,
    optimize: bool,
) -> AntSessionConfig:
    return AntSessionConfig(
        temporal_latent_dim=config.temporal_latent_dim,
        session_id=session_id,
        seed=seed,
        rollout_config=ant_runtime_replay_rollout_config(enable_sparse_exploration=True),
        joint_apply_writeback=True,
        joint_apply_policy_optimization=optimize,
        objective=AntObjectiveKind.ECOLOGY,
        sense_schema=AntSenseSchema.ECOLOGY_V2,
    )


def _policy_fingerprints(
    checkpoints: tuple[AntLearningCheckpoint, ...],
) -> tuple[str, ...]:
    return tuple(checkpoint.policy_fingerprint for checkpoint in checkpoints)


async def _train_arm(
    *,
    config: EcologyCurriculumConfig,
    initial: tuple[AntLearningCheckpoint, ...],
    arm: str,
    optimize: bool,
) -> tuple[tuple[AntLearningCheckpoint, ...], tuple[bytes, ...]]:
    checkpoints = initial
    stages = (
        EcologyStage.BUTTER,
        EcologyStage.WOOD_STICK,
        EcologyStage.BURNING_MATCH,
        EcologyStage.COMPOSITE,
    )
    for stage_index, stage in enumerate(stages):
        for episode in range(config.stage_episodes):
            episode_seed = config.seed + stage_index * 10_000 + episode * 101
            runner = KernelColonyRunner(
                _world(
                    config=config,
                    stage=stage,
                    seed=episode_seed,
                    heldout=False,
                ),
                base_config=_session_config(
                    config=config,
                    seed=episode_seed,
                    session_id=(f"ecology:{arm}:{stage.value}:episode:{episode}"),
                    optimize=optimize,
                ),
            )
            runner.restore_learning_checkpoints(checkpoints)
            await runner.run(config.stage_rounds)
            checkpoints = runner.export_learning_checkpoints(
                checkpoint_prefix=(f"ecology:{arm}:{stage.value}:episode:{episode}:trained"),
                include_runtime_replay=False,
            )
    archives = runner.export_learning_checkpoint_archives(
        checkpoint_prefix=f"ecology:{arm}:trained",
    )
    return checkpoints, archives


def _latest_replay_counts(
    runner: KernelColonyRunner,
) -> tuple[int, int, int, int, int, int]:
    if not runner.rounds:
        return 0, 0, 0, 0, 0, 0
    records = runner.rounds[-1].ant_steps
    return (
        sum(item.runtime_replay_captured for item in records),
        sum(item.runtime_replay_settled for item in records),
        sum(item.runtime_replay_lineage_matches for item in records),
        sum(item.runtime_replay_pending_captures for item in records),
        sum(item.runtime_replay_staged_rollouts for item in records),
        sum(len(item.runtime_replay_drop_reasons) for item in records),
    )


async def _evaluate_arm(
    *,
    config: EcologyCurriculumConfig,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    arm: str,
    stage: EcologyStage,
    seed: int,
) -> EcologyArmMetrics:
    world = _world(
        config=config,
        stage=stage,
        seed=seed,
        heldout=True,
    )
    runner = KernelColonyRunner(
        world,
        base_config=_session_config(
            config=config,
            seed=seed,
            session_id=f"ecology:{arm}:heldout:{stage.value}:{seed}",
            optimize=False,
        ),
    )
    runner.restore_learning_checkpoints(checkpoints)
    await runner.run(config.heldout_rounds)
    records = tuple(step for round_record in runner.rounds for step in round_record.ant_steps)
    captured, settled, lineage, pending, staged, drop_count = _latest_replay_counts(runner)
    eligible_captures = captured - pending
    if eligible_captures < 0 or settled > eligible_captures:
        raise RuntimeError("runtime replay counters violate eligible settlement ordering")
    return EcologyArmMetrics(
        arm=arm,
        stage=stage,
        seed=seed,
        pickups=world.food_pickups,
        deliveries=world.food_delivered,
        obstacle_contacts=sum(item.obstacle_contact for item in records),
        harmful_heat_ticks=sum(item.heat_harmful for item in records),
        heat_entries=sum(item.entered_harmful_heat for item in records),
        heat_escapes=sum(item.escaped_harmful_heat for item in records),
        applied_distance=sum(item.applied_step for item in records),
        replay_captured=captured,
        replay_settled=settled,
        replay_lineage_matches=lineage,
        replay_pending_captures=pending,
        replay_staged_rollouts=staged,
        replay_drop_count=drop_count,
        replay_settlement_coverage=(settled / eligible_captures if eligible_captures else 0.0),
        replay_lineage_coverage=(lineage / settled if settled else 0.0),
    )


def _stage_metrics(
    metrics: tuple[EcologyArmMetrics, ...],
    stage: EcologyStage,
) -> tuple[EcologyArmMetrics, ...]:
    return tuple(item for item in metrics if item.stage is stage)


def _build_gates(
    *,
    initial: tuple[AntLearningCheckpoint, ...],
    learned: tuple[AntLearningCheckpoint, ...],
    no_optimize: tuple[AntLearningCheckpoint, ...],
    learned_metrics: tuple[EcologyArmMetrics, ...],
    cold_metrics: tuple[EcologyArmMetrics, ...],
    no_optimize_metrics: tuple[EcologyArmMetrics, ...],
) -> tuple[EcologyGate, ...]:
    evaluation_schedule = tuple((item.stage, item.seed) for item in learned_metrics)
    if (
        tuple((item.stage, item.seed) for item in cold_metrics) != evaluation_schedule
        or tuple((item.stage, item.seed) for item in no_optimize_metrics) != evaluation_schedule
    ):
        raise ValueError("held-out arm stage/seed schedules must align")

    learned_butter = _stage_metrics(learned_metrics, EcologyStage.BUTTER)
    cold_butter = _stage_metrics(cold_metrics, EcologyStage.BUTTER)
    no_opt_butter = _stage_metrics(no_optimize_metrics, EcologyStage.BUTTER)
    learned_stick = _stage_metrics(learned_metrics, EcologyStage.WOOD_STICK)
    no_opt_stick = _stage_metrics(no_optimize_metrics, EcologyStage.WOOD_STICK)
    learned_match = _stage_metrics(learned_metrics, EcologyStage.BURNING_MATCH)
    no_opt_match = _stage_metrics(no_optimize_metrics, EcologyStage.BURNING_MATCH)
    learned_composite = _stage_metrics(learned_metrics, EcologyStage.COMPOSITE)
    cold_composite = _stage_metrics(cold_metrics, EcologyStage.COMPOSITE)
    no_opt_composite = _stage_metrics(no_optimize_metrics, EcologyStage.COMPOSITE)
    expected_seeds = tuple(item.seed for item in learned_butter)
    learned_stage_groups = (
        learned_butter,
        learned_stick,
        learned_match,
        learned_composite,
    )
    if not expected_seeds or any(
        tuple(item.seed for item in group) != expected_seeds for group in learned_stage_groups
    ):
        raise ValueError("each learned held-out stage must use the same non-empty seed schedule")

    pickup_gain = all(
        learned_item.pickups > max(cold_item.pickups, no_opt_item.pickups)
        for learned_item, cold_item, no_opt_item in zip(
            learned_butter,
            cold_butter,
            no_opt_butter,
            strict=True,
        )
    )
    delivery_present = all(item.deliveries >= 1 for item in learned_butter)
    stick_avoidance = all(
        learned_item.pickups > 0 and learned_item.obstacle_contacts < no_opt_item.obstacle_contacts
        for learned_item, no_opt_item in zip(
            learned_stick,
            no_opt_stick,
            strict=True,
        )
    )
    match_avoidance = all(
        learned_item.heat_escapes > 0 and learned_item.harmful_heat_ticks < no_opt_item.harmful_heat_ticks
        for learned_item, no_opt_item in zip(
            learned_match,
            no_opt_match,
            strict=True,
        )
    )
    composite_performance = all(
        learned_item.deliveries >= 1
        and learned_item.pickups > max(cold_item.pickups, no_opt_item.pickups)
        and learned_item.obstacle_contacts <= no_opt_item.obstacle_contacts
        and learned_item.harmful_heat_ticks <= no_opt_item.harmful_heat_ticks
        for learned_item, cold_item, no_opt_item in zip(
            learned_composite,
            cold_composite,
            no_opt_composite,
            strict=True,
        )
    )
    replay_ok = all(
        item.replay_settlement_coverage >= 0.99 and item.replay_lineage_coverage >= 0.99 and item.replay_drop_count == 0
        for item in learned_metrics
    )

    return (
        EcologyGate(
            name="policy_changed",
            passed=_policy_fingerprints(learned) != _policy_fingerprints(initial),
            observed=str(_policy_fingerprints(learned)),
            threshold="learned policy fingerprint differs from initial",
        ),
        EcologyGate(
            name="no_optimize_policy_stable",
            passed=_policy_fingerprints(no_optimize) == _policy_fingerprints(initial),
            observed=str(_policy_fingerprints(no_optimize)),
            threshold="no-optimize policy fingerprint equals initial",
        ),
        EcologyGate(
            name="butter_pickup_gain",
            passed=pickup_gain,
            observed=(
                f"learned={sum(item.pickups for item in learned_butter)}, "
                f"cold={sum(item.pickups for item in cold_butter)}, "
                f"no_opt={sum(item.pickups for item in no_opt_butter)}"
            ),
            threshold="butter-only pickups exceed both controls per seed",
        ),
        EcologyGate(
            name="butter_delivery_present",
            passed=delivery_present,
            observed=str(tuple(item.deliveries for item in learned_butter)),
            threshold="butter-only delivery in every held-out seed",
        ),
        EcologyGate(
            name="wood_stick_avoidance",
            passed=stick_avoidance,
            observed=(
                f"learned={sum(item.obstacle_contacts for item in learned_stick)}, "
                f"no_opt={sum(item.obstacle_contacts for item in no_opt_stick)}"
            ),
            threshold=("stick held-out contacts lower than no-optimize with pickup per seed"),
        ),
        EcologyGate(
            name="burning_match_avoidance",
            passed=match_avoidance,
            observed=(
                f"heat learned={sum(item.harmful_heat_ticks for item in learned_match)}, "
                f"no_opt={sum(item.harmful_heat_ticks for item in no_opt_match)}; "
                f"escapes={sum(item.heat_escapes for item in learned_match)}"
            ),
            threshold=("match held-out escape present and harmful exposure lower per seed"),
        ),
        EcologyGate(
            name="composite_performance",
            passed=composite_performance,
            observed=str(
                tuple(
                    (
                        item.pickups,
                        item.deliveries,
                        item.obstacle_contacts,
                        item.harmful_heat_ticks,
                    )
                    for item in learned_composite
                )
            ),
            threshold=("composite delivery and pickup gain with hazards no worse than no-optimize per seed"),
        ),
        EcologyGate(
            name="runtime_replay_lineage",
            passed=replay_ok,
            observed=str(
                tuple(
                    (
                        item.stage.value,
                        item.seed,
                        item.replay_settlement_coverage,
                        item.replay_lineage_coverage,
                        item.replay_pending_captures,
                        item.replay_drop_count,
                    )
                    for item in learned_metrics
                )
            ),
            threshold=("eligible settlement and lineage coverage >= 0.99 with no drops per stage/seed"),
        ),
    )


async def train_and_evaluate_ecology_checkpoint(
    config: EcologyCurriculumConfig,
) -> EcologyCheckpointCandidate:
    bootstrap_world: AntWorld = _world(
        config=config,
        stage=EcologyStage.COMPOSITE,
        seed=config.seed,
        heldout=False,
    )
    bootstrap_runner = KernelColonyRunner(
        bootstrap_world,
        base_config=_session_config(
            config=config,
            seed=config.seed,
            session_id="ecology:shared-initial",
            optimize=True,
        ),
    )
    initial = bootstrap_runner.export_learning_checkpoints(
        checkpoint_prefix="ecology:shared-initial",
        include_runtime_replay=False,
    )
    learned, learned_archives = await _train_arm(
        config=config,
        initial=initial,
        arm="learned",
        optimize=True,
    )
    no_optimize, _ = await _train_arm(
        config=config,
        initial=initial,
        arm="no_optimize",
        optimize=False,
    )
    heldout_stages = (
        EcologyStage.BUTTER,
        EcologyStage.WOOD_STICK,
        EcologyStage.BURNING_MATCH,
        EcologyStage.COMPOSITE,
    )
    learned_metrics = tuple(
        [
            await _evaluate_arm(
                config=config,
                checkpoints=learned,
                arm="learned",
                stage=stage,
                seed=seed,
            )
            for stage in heldout_stages
            for seed in config.heldout_seeds
        ]
    )
    cold_metrics = tuple(
        [
            await _evaluate_arm(
                config=config,
                checkpoints=initial,
                arm="cold",
                stage=stage,
                seed=seed,
            )
            for stage in heldout_stages
            for seed in config.heldout_seeds
        ]
    )
    no_optimize_metrics = tuple(
        [
            await _evaluate_arm(
                config=config,
                checkpoints=no_optimize,
                arm="no_optimize",
                stage=stage,
                seed=seed,
            )
            for stage in heldout_stages
            for seed in config.heldout_seeds
        ]
    )
    gates = _build_gates(
        initial=initial,
        learned=learned,
        no_optimize=no_optimize,
        learned_metrics=learned_metrics,
        cold_metrics=cold_metrics,
        no_optimize_metrics=no_optimize_metrics,
    )
    breakpoints = tuple(gate.name for gate in gates if not gate.passed)
    verdict = "PASS" if not breakpoints else "BLOCK"
    report = EcologyCheckpointReport(
        schema_version=ECOLOGY_CURRICULUM_SCHEMA_VERSION,
        config=config,
        initial_policy_fingerprints=_policy_fingerprints(initial),
        learned_policy_fingerprints=_policy_fingerprints(learned),
        no_optimize_policy_fingerprints=_policy_fingerprints(no_optimize),
        learned_metrics=learned_metrics,
        cold_metrics=cold_metrics,
        no_optimize_metrics=no_optimize_metrics,
        gates=gates,
        verdict=verdict,
        diagnostic_breakpoints=breakpoints,
        description=(
            "PASS: ecology checkpoint cleared all frozen held-out gates"
            if verdict == "PASS"
            else "BLOCK: " + ", ".join(breakpoints)
        ),
    )
    return EcologyCheckpointCandidate(
        checkpoints=learned,
        checkpoint_archives=learned_archives,
        report=report,
    )


__all__ = [
    "ECOLOGY_CURRICULUM_SCHEMA_VERSION",
    "ECOLOGY_REQUIRED_GATE_NAMES",
    "EcologyArmMetrics",
    "EcologyCheckpointCandidate",
    "EcologyCheckpointReport",
    "EcologyCurriculumConfig",
    "EcologyGate",
    "EcologyStage",
    "train_and_evaluate_ecology_checkpoint",
]
