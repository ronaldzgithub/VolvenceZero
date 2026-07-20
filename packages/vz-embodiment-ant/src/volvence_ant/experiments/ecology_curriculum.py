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
) -> tuple[AntLearningCheckpoint, ...]:
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
                checkpoint_prefix=(f"ecology:{arm}:{stage.value}:episode:{episode}:trained")
            )
    return checkpoints


def _latest_replay_counts(
    runner: KernelColonyRunner,
) -> tuple[int, int, int]:
    if not runner.rounds:
        return 0, 0, 0
    records = runner.rounds[-1].ant_steps
    return (
        sum(item.runtime_replay_captured for item in records),
        sum(item.runtime_replay_settled for item in records),
        sum(item.runtime_replay_lineage_matches for item in records),
    )


async def _evaluate_arm(
    *,
    config: EcologyCurriculumConfig,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    arm: str,
    seed: int,
) -> EcologyArmMetrics:
    world = _world(
        config=config,
        stage=EcologyStage.COMPOSITE,
        seed=seed,
        heldout=True,
    )
    runner = KernelColonyRunner(
        world,
        base_config=_session_config(
            config=config,
            seed=seed,
            session_id=f"ecology:{arm}:heldout:{seed}",
            optimize=False,
        ),
    )
    runner.restore_learning_checkpoints(checkpoints)
    await runner.run(config.heldout_rounds)
    records = tuple(step for round_record in runner.rounds for step in round_record.ant_steps)
    captured, settled, lineage = _latest_replay_counts(runner)
    return EcologyArmMetrics(
        arm=arm,
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
        replay_settlement_coverage=(settled / captured if captured else 0.0),
        replay_lineage_coverage=(lineage / settled if settled else 0.0),
    )


def _build_gates(
    *,
    initial: tuple[AntLearningCheckpoint, ...],
    learned: tuple[AntLearningCheckpoint, ...],
    no_optimize: tuple[AntLearningCheckpoint, ...],
    learned_metrics: tuple[EcologyArmMetrics, ...],
    cold_metrics: tuple[EcologyArmMetrics, ...],
    no_optimize_metrics: tuple[EcologyArmMetrics, ...],
) -> tuple[EcologyGate, ...]:
    seed_schedule = tuple(item.seed for item in learned_metrics)
    if (
        tuple(item.seed for item in cold_metrics) != seed_schedule
        or tuple(item.seed for item in no_optimize_metrics) != seed_schedule
    ):
        raise ValueError("held-out arm seed schedules must align")
    learned_pickups = float(sum(item.pickups for item in learned_metrics))
    cold_pickups = float(sum(item.pickups for item in cold_metrics))
    no_opt_pickups = float(sum(item.pickups for item in no_optimize_metrics))
    learned_deliveries = float(sum(item.deliveries for item in learned_metrics))
    learned_contacts = float(sum(item.obstacle_contacts for item in learned_metrics))
    no_opt_contacts = float(sum(item.obstacle_contacts for item in no_optimize_metrics))
    learned_heat = float(sum(item.harmful_heat_ticks for item in learned_metrics))
    no_opt_heat = float(sum(item.harmful_heat_ticks for item in no_optimize_metrics))
    learned_escapes = float(sum(item.heat_escapes for item in learned_metrics))
    replay_ok = all(
        item.replay_settlement_coverage >= 0.99 and item.replay_lineage_coverage >= 0.99 for item in learned_metrics
    )
    pickup_gain = all(
        learned_item.pickups > max(cold_item.pickups, no_opt_item.pickups)
        for learned_item, cold_item, no_opt_item in zip(
            learned_metrics,
            cold_metrics,
            no_optimize_metrics,
            strict=True,
        )
    )
    delivery_present = all(item.deliveries >= 1 for item in learned_metrics)
    stick_avoidance = all(
        learned_item.pickups > 0 and learned_item.obstacle_contacts < no_opt_item.obstacle_contacts
        for learned_item, no_opt_item in zip(
            learned_metrics,
            no_optimize_metrics,
            strict=True,
        )
    )
    match_avoidance = all(
        learned_item.heat_escapes > 0
        and learned_item.harmful_heat_ticks < no_opt_item.harmful_heat_ticks
        for learned_item, no_opt_item in zip(
            learned_metrics,
            no_optimize_metrics,
            strict=True,
        )
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
            observed=(f"learned={learned_pickups:.0f}, cold={cold_pickups:.0f}, no_opt={no_opt_pickups:.0f}"),
            threshold="learned pickups strictly exceed both controls per seed",
        ),
        EcologyGate(
            name="butter_delivery_present",
            passed=delivery_present,
            observed=f"{learned_deliveries:.0f}",
            threshold="at least one held-out delivery in every seed",
        ),
        EcologyGate(
            name="wood_stick_avoidance",
            passed=stick_avoidance,
            observed=(f"learned={learned_contacts:.0f}, no_opt={no_opt_contacts:.0f}"),
            threshold="learned contacts lower than no-optimize with pickup per seed",
        ),
        EcologyGate(
            name="burning_match_avoidance",
            passed=match_avoidance,
            observed=(f"heat learned={learned_heat:.0f}, no_opt={no_opt_heat:.0f}; escapes={learned_escapes:.0f}"),
            threshold="at least one escape and less harmful exposure per seed",
        ),
        EcologyGate(
            name="runtime_replay_lineage",
            passed=replay_ok,
            observed=str(
                tuple(
                    (
                        item.replay_settlement_coverage,
                        item.replay_lineage_coverage,
                    )
                    for item in learned_metrics
                )
            ),
            threshold="settlement and lineage coverage >= 0.99 per seed",
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
    initial = bootstrap_runner.export_learning_checkpoints(checkpoint_prefix="ecology:shared-initial")
    learned = await _train_arm(
        config=config,
        initial=initial,
        arm="learned",
        optimize=True,
    )
    no_optimize = await _train_arm(
        config=config,
        initial=initial,
        arm="no_optimize",
        optimize=False,
    )
    learned_metrics = tuple(
        [
            await _evaluate_arm(
                config=config,
                checkpoints=learned,
                arm="learned",
                seed=seed,
            )
            for seed in config.heldout_seeds
        ]
    )
    cold_metrics = tuple(
        [
            await _evaluate_arm(
                config=config,
                checkpoints=initial,
                arm="cold",
                seed=seed,
            )
            for seed in config.heldout_seeds
        ]
    )
    no_optimize_metrics = tuple(
        [
            await _evaluate_arm(
                config=config,
                checkpoints=no_optimize,
                arm="no_optimize",
                seed=seed,
            )
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
    return EcologyCheckpointCandidate(checkpoints=learned, report=report)


__all__ = [
    "ECOLOGY_CURRICULUM_SCHEMA_VERSION",
    "EcologyArmMetrics",
    "EcologyCheckpointCandidate",
    "EcologyCheckpointReport",
    "EcologyCurriculumConfig",
    "EcologyGate",
    "EcologyStage",
    "train_and_evaluate_ecology_checkpoint",
]
