"""Offline ecology curriculum over the real ant runtime.

No action labels or steering hints are supplied.  Every episode runs the same
``AntSession`` sense -> temporal -> action -> outcome -> PE -> credit path used
by the live app, then carries owner-exported checkpoints into the next world.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable

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
from volvence_ant.experiments.ecology_probe import (
    EcologyActionProbe,
    run_ecology_action_probes,
)
from volvence_ant.runtime import (
    AntLearningCheckpoint,
    AntObjectiveKind,
    AntSenseSchema,
    AntStepRecord,
    AntSessionConfig,
    KernelColonyRunner,
)


ECOLOGY_CURRICULUM_SCHEMA_VERSION = "digital-ant-ecology-curriculum.v2"
ECOLOGY_REQUIRED_GATE_NAMES = (
    "training_event_coverage",
    "paired_action_sensitivity",
    "policy_changed",
    "no_optimize_policy_stable",
    "butter_pickup_gain",
    "butter_delivery_present",
    "wood_stick_navigation",
    "burning_match_route_avoidance",
    "burning_match_forced_escape",
    "composite_performance",
    "matched_ablation_advantage",
    "runtime_replay_lineage",
)


class EcologyStage(str, Enum):
    BUTTER = "butter"
    WOOD_STICK = "wood_stick"
    BURNING_MATCH = "burning_match"
    COMPOSITE = "composite"


class EcologyTrainingTier(str, Enum):
    NEAR = "near"
    MEDIUM = "medium"
    FAR = "far"


class EcologyDataSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    HELDOUT = "heldout"


class EcologyEvaluationScenario(str, Enum):
    BUTTER_ONLY = "butter_only"
    WOOD_STICK_ROUTE = "wood_stick_route"
    HEAT_ROUTE_AVOIDANCE = "heat_route_avoidance"
    HEAT_FORCED_ESCAPE = "heat_forced_escape"
    COMPOSITE = "composite"


@dataclass(frozen=True)
class EcologyCurriculumConfig:
    n_ants: int = 8
    temporal_latent_dim: int = 16
    stage_rounds: int = 80
    stage_episodes: int = 4
    mastery_min_episodes: int = 3
    mastery_min_pickups: int = 2
    mastery_min_obstacle_contacts: int = 2
    mastery_min_heat_events: int = 2
    interleave_every: int = 2
    validation_rounds: int = 80
    validation_seeds: tuple[int, ...] = (43, 59)
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
        if not 1 <= self.mastery_min_episodes <= self.stage_episodes:
            raise ValueError(
                "mastery_min_episodes must be within stage episode budget"
            )
        if (
            self.mastery_min_pickups < 1
            or self.mastery_min_obstacle_contacts < 1
            or self.mastery_min_heat_events < 1
        ):
            raise ValueError("mastery event thresholds must be >= 1")
        if self.interleave_every < 1:
            raise ValueError("interleave_every must be >= 1")
        if self.validation_rounds < 1 or not self.validation_seeds:
            raise ValueError(
                "validation budget and seeds must be non-empty"
            )
        if self.heldout_rounds < 1 or not self.heldout_seeds:
            raise ValueError("heldout budget and seeds must be non-empty")
        if set(self.validation_seeds).intersection(self.heldout_seeds):
            raise ValueError(
                "validation and held-out seeds must be disjoint"
            )


@dataclass(frozen=True)
class EcologyArmMetrics:
    arm: str
    data_split: EcologyDataSplit
    stage: EcologyStage
    scenario: EcologyEvaluationScenario
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
    nonzero_ecology_payoffs: int
    activated_sense_channels: tuple[str, ...]
    first_pickup_tick: int | None
    first_obstacle_contact_tick: int | None
    first_heat_entry_tick: int | None
    first_heat_escape_tick: int | None
    minimum_food_distance: float | None
    minimum_obstacle_distance: float | None
    minimum_heat_distance: float | None
    switch_count: int
    mean_persistence_steps: float
    closed_segment_count: int
    longest_segment_length: int
    mean_absolute_turn_delta: float
    policy_fingerprint_stable: bool
    temporal_fingerprint_stable: bool


@dataclass(frozen=True)
class EcologyTrainingEpisodePlan:
    stage: EcologyStage
    tier: EcologyTrainingTier
    seed: int
    episode_index: int
    interleaved: bool
    forced_escape: bool


@dataclass(frozen=True)
class EcologyTrainingEpisodeReport:
    arm: str
    plan: EcologyTrainingEpisodePlan
    pickups: int
    deliveries: int
    obstacle_contacts: int
    heat_entries: int
    heat_escapes: int
    nonzero_ecology_payoffs: int
    activated_sense_channels: tuple[str, ...]
    minimum_food_distance: float | None
    minimum_obstacle_distance: float | None
    minimum_heat_distance: float | None
    switch_count: int
    mean_persistence_steps: float
    closed_segment_count: int
    longest_segment_length: int
    policy_fingerprints_before: tuple[str, ...]
    policy_fingerprints_after: tuple[str, ...]


@dataclass(frozen=True)
class EcologyStageMastery:
    stage: EcologyStage
    reached: bool
    primary_episodes: int
    pickups: int
    obstacle_contacts: int
    heat_entries: int
    heat_escapes: int
    threshold: str


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
    training_schedule: tuple[EcologyTrainingEpisodePlan, ...]
    learned_training: tuple[EcologyTrainingEpisodeReport, ...]
    no_optimize_training: tuple[EcologyTrainingEpisodeReport, ...]
    valence_off_training: tuple[EcologyTrainingEpisodeReport, ...]
    segment_credit_off_training: tuple[
        EcologyTrainingEpisodeReport,
        ...,
    ]
    learned_mastery: tuple[EcologyStageMastery, ...]
    action_probes: tuple[EcologyActionProbe, ...]
    validation_metrics: tuple[EcologyArmMetrics, ...]
    learned_metrics: tuple[EcologyArmMetrics, ...]
    cold_metrics: tuple[EcologyArmMetrics, ...]
    no_optimize_metrics: tuple[EcologyArmMetrics, ...]
    valence_off_metrics: tuple[EcologyArmMetrics, ...]
    segment_credit_off_metrics: tuple[EcologyArmMetrics, ...]
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
    data_split: EcologyDataSplit,
    tier: EcologyTrainingTier,
) -> tuple[WorldObject, ...]:
    split_offset = {
        EcologyDataSplit.TRAIN: 0,
        EcologyDataSplit.VALIDATION: 1_000_003,
        EcologyDataSplit.HELDOUT: 2_000_003,
    }[data_split]
    rng = np.random.default_rng(seed + split_offset)
    bearing = float(rng.uniform(-math.pi, math.pi))
    distance_bounds = {
        EcologyTrainingTier.NEAR: (1.35, 1.75),
        EcologyTrainingTier.MEDIUM: (2.1, 2.9),
        EcologyTrainingTier.FAR: (3.0, 4.4),
    }[tier]
    distance = float(rng.uniform(*distance_bounds))
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
        centre_distance = max(1.15, distance * 0.55)
        centre_x = math.cos(bearing) * centre_distance
        centre_y = math.sin(bearing) * centre_distance
        perpendicular = bearing + math.pi / 2.0
        half_length = min(1.35, max(0.8, distance * 0.35))
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
        match_distance = max(1.15, distance * 0.48)
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
    data_split: EcologyDataSplit,
    tier: EcologyTrainingTier,
    forced_escape: bool = False,
) -> ColonyWorld:
    split_offset = {
        EcologyDataSplit.TRAIN: 17,
        EcologyDataSplit.VALIDATION: 3_000_017,
        EcologyDataSplit.HELDOUT: 4_000_017,
    }[data_split]
    objects = _scene_objects(
        stage=stage,
        seed=seed,
        data_split=data_split,
        tier=tier,
    )
    world = ColonyWorld(
        config=AntWorldConfig(
            seed=seed + split_offset,
            antenna_offset_deg=45.0,
            antenna_reach=0.9,
        ),
        world_objects=objects,
        n_bodies=config.n_ants,
    )
    if forced_escape:
        matches = tuple(
            item for item in objects if isinstance(item, BurningMatch)
        )
        if len(matches) != 1:
            raise RuntimeError(
                "forced escape scene requires exactly one burning match"
            )
        match = matches[0]
        spawn_radius = max(0.05, match.harm_radius * 0.35)
        for body_id in range(config.n_ants):
            angle = (
                (body_id + 1) * 2.399963229728653
                + seed * 0.017
            )
            world.set_body_pose(
                body_id=body_id,
                x=match.x + math.cos(angle) * spawn_radius,
                y=match.y + math.sin(angle) * spawn_radius,
                heading=(angle + math.pi / 2.0),
            )
    return world


def _session_config(
    *,
    config: EcologyCurriculumConfig,
    seed: int,
    session_id: str,
    optimize: bool,
    local_valence_enabled: bool = True,
    segment_credit_enabled: bool = True,
    learning_enabled: bool = True,
    sparse_exploration_enabled: bool = True,
) -> AntSessionConfig:
    return AntSessionConfig(
        temporal_latent_dim=config.temporal_latent_dim,
        session_id=session_id,
        seed=seed,
        rollout_config=ant_runtime_replay_rollout_config(
            enable_sparse_exploration=sparse_exploration_enabled,
            enable_segment_credit=segment_credit_enabled,
        ),
        joint_apply_writeback=True,
        joint_apply_policy_optimization=optimize,
        joint_learning_enabled=learning_enabled,
        objective=AntObjectiveKind.ECOLOGY,
        sense_schema=AntSenseSchema.ECOLOGY_V2,
        ecology_local_valence_enabled=local_valence_enabled,
    )


def _policy_fingerprints(
    checkpoints: tuple[AntLearningCheckpoint, ...],
) -> tuple[str, ...]:
    return tuple(checkpoint.policy_fingerprint for checkpoint in checkpoints)


def _flatten_records(
    runner: KernelColonyRunner,
) -> tuple[AntStepRecord, ...]:
    return tuple(
        step
        for round_record in runner.rounds
        for step in round_record.ant_steps
    )


def _minimum_optional(
    values: tuple[float | None, ...],
) -> float | None:
    finite = tuple(value for value in values if value is not None)
    return min(finite) if finite else None


def _activated_sense_channels(
    records: tuple[AntStepRecord, ...],
) -> tuple[str, ...]:
    active: set[str] = set()
    for record in records:
        for name, value in record.sense_activation:
            if abs(value) > 1e-12:
                active.add(name)
    return tuple(sorted(active))


def _mean_absolute_turn_delta(
    records: tuple[AntStepRecord, ...],
) -> float:
    if len(records) < 2:
        return 0.0
    deltas = tuple(
        abs(current.command.turn_command - previous.command.turn_command)
        for previous, current in zip(
            records,
            records[1:],
            strict=True,
        )
    )
    return sum(deltas) / len(deltas)


def _first_tick(
    records: tuple[AntStepRecord, ...],
    *,
    predicate: Callable[[AntStepRecord], bool],
) -> int | None:
    for record in records:
        if predicate(record):
            return int(record.tick)
    return None


def _tier_for_episode(episode_index: int) -> EcologyTrainingTier:
    if episode_index == 0:
        return EcologyTrainingTier.NEAR
    if episode_index == 1:
        return EcologyTrainingTier.MEDIUM
    return EcologyTrainingTier.FAR


def _mastery_reached(
    *,
    config: EcologyCurriculumConfig,
    stage: EcologyStage,
    primary_episodes: int,
    pickups: int,
    obstacle_contacts: int,
    heat_entries: int,
    heat_escapes: int,
) -> bool:
    if primary_episodes < config.mastery_min_episodes:
        return False
    if stage is EcologyStage.BUTTER:
        return pickups >= config.mastery_min_pickups
    if stage is EcologyStage.WOOD_STICK:
        return (
            pickups >= config.mastery_min_pickups
            and obstacle_contacts
            >= config.mastery_min_obstacle_contacts
        )
    if stage is EcologyStage.BURNING_MATCH:
        return (
            heat_entries >= config.mastery_min_heat_events
            and heat_escapes >= config.mastery_min_heat_events
        )
    return (
        pickups >= config.mastery_min_pickups
        and (
            obstacle_contacts > 0
            or heat_entries > 0
            or heat_escapes > 0
        )
    )


def _mastery_threshold(
    config: EcologyCurriculumConfig,
    stage: EcologyStage,
) -> str:
    if stage is EcologyStage.BUTTER:
        return f"pickups>={config.mastery_min_pickups}"
    if stage is EcologyStage.WOOD_STICK:
        return (
            f"pickups>={config.mastery_min_pickups}, "
            "contacts>="
            f"{config.mastery_min_obstacle_contacts}"
        )
    if stage is EcologyStage.BURNING_MATCH:
        return (
            f"entries>={config.mastery_min_heat_events}, "
            f"escapes>={config.mastery_min_heat_events}"
        )
    return (
        f"pickups>={config.mastery_min_pickups} and hazard sample present"
    )


async def _run_training_episode(
    *,
    config: EcologyCurriculumConfig,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    arm: str,
    optimize: bool,
    local_valence_enabled: bool,
    segment_credit_enabled: bool,
    plan: EcologyTrainingEpisodePlan,
) -> tuple[
    KernelColonyRunner,
    tuple[AntLearningCheckpoint, ...],
    EcologyTrainingEpisodeReport,
]:
    runner = KernelColonyRunner(
        _world(
            config=config,
            stage=plan.stage,
            seed=plan.seed,
            data_split=EcologyDataSplit.TRAIN,
            tier=plan.tier,
            forced_escape=plan.forced_escape,
        ),
        base_config=_session_config(
            config=config,
            seed=plan.seed,
            session_id=(
                f"ecology:{arm}:{plan.stage.value}:"
                f"{plan.tier.value}:episode:{plan.episode_index}"
            ),
            optimize=optimize,
            local_valence_enabled=local_valence_enabled,
            segment_credit_enabled=segment_credit_enabled,
        ),
    )
    runner.restore_learning_checkpoints(checkpoints)
    before = _policy_fingerprints(checkpoints)
    await runner.run(config.stage_rounds)
    trained = runner.export_learning_checkpoints(
        checkpoint_prefix=(
            f"ecology:{arm}:{plan.stage.value}:{plan.tier.value}:"
            f"episode:{plan.episode_index}:trained"
        ),
        include_runtime_replay=False,
    )
    records = _flatten_records(runner)
    report = EcologyTrainingEpisodeReport(
        arm=arm,
        plan=plan,
        pickups=runner.world.food_pickups,
        deliveries=runner.world.food_delivered,
        obstacle_contacts=sum(
            int(item.obstacle_contact) for item in records
        ),
        heat_entries=sum(
            int(item.entered_harmful_heat) for item in records
        ),
        heat_escapes=sum(
            int(item.escaped_harmful_heat) for item in records
        ),
        nonzero_ecology_payoffs=sum(
            abs(item.ecology_action_payoff) > 1e-12
            for item in records
        ),
        activated_sense_channels=_activated_sense_channels(records),
        minimum_food_distance=_minimum_optional(
            tuple(item.nearest_food_distance for item in records)
        ),
        minimum_obstacle_distance=_minimum_optional(
            tuple(item.nearest_obstacle_distance for item in records)
        ),
        minimum_heat_distance=_minimum_optional(
            tuple(item.nearest_heat_distance for item in records)
        ),
        switch_count=sum(int(item.is_switching) for item in records),
        mean_persistence_steps=(
            sum(item.steps_since_switch for item in records)
            / len(records)
            if records
            else 0.0
        ),
        closed_segment_count=max(
            (
                item.runtime_closed_segments
                for item in records
            ),
            default=0,
        ),
        longest_segment_length=max(
            (
                item.runtime_longest_segment_length
                for item in records
            ),
            default=0,
        ),
        policy_fingerprints_before=before,
        policy_fingerprints_after=_policy_fingerprints(trained),
    )
    return runner, trained, report


async def _train_arm(
    *,
    config: EcologyCurriculumConfig,
    initial: tuple[AntLearningCheckpoint, ...],
    arm: str,
    optimize: bool,
    local_valence_enabled: bool,
    segment_credit_enabled: bool,
    schedule: tuple[EcologyTrainingEpisodePlan, ...] | None = None,
) -> tuple[
    tuple[AntLearningCheckpoint, ...],
    tuple[bytes, ...],
    tuple[EcologyTrainingEpisodePlan, ...],
    tuple[EcologyTrainingEpisodeReport, ...],
    tuple[EcologyStageMastery, ...],
]:
    checkpoints = initial
    stages = (
        EcologyStage.BUTTER,
        EcologyStage.WOOD_STICK,
        EcologyStage.BURNING_MATCH,
        EcologyStage.COMPOSITE,
    )
    reports: list[EcologyTrainingEpisodeReport] = []
    plans: list[EcologyTrainingEpisodePlan] = []
    mastery: list[EcologyStageMastery] = []
    runner: KernelColonyRunner | None = None
    if schedule is not None:
        for plan in schedule:
            runner, checkpoints, report = await _run_training_episode(
                config=config,
                checkpoints=checkpoints,
                arm=arm,
                optimize=optimize,
                local_valence_enabled=local_valence_enabled,
                segment_credit_enabled=segment_credit_enabled,
                plan=plan,
            )
            plans.append(plan)
            reports.append(report)
    else:
        mastered_stages: list[EcologyStage] = []
        for stage_index, stage in enumerate(stages):
            pickups = 0
            obstacle_contacts = 0
            heat_entries = 0
            heat_escapes = 0
            primary_episodes = 0
            reached = False
            for episode in range(config.stage_episodes):
                plan = EcologyTrainingEpisodePlan(
                    stage=stage,
                    tier=_tier_for_episode(episode),
                    seed=(
                        config.seed
                        + stage_index * 10_000
                        + episode * 101
                    ),
                    episode_index=episode,
                    interleaved=False,
                    forced_escape=(
                        stage is EcologyStage.BURNING_MATCH
                        and episode == 0
                    ),
                )
                runner, checkpoints, report = (
                    await _run_training_episode(
                        config=config,
                        checkpoints=checkpoints,
                        arm=arm,
                        optimize=optimize,
                        local_valence_enabled=(
                            local_valence_enabled
                        ),
                        segment_credit_enabled=(
                            segment_credit_enabled
                        ),
                        plan=plan,
                    )
                )
                plans.append(plan)
                reports.append(report)
                primary_episodes += 1
                pickups += report.pickups
                obstacle_contacts += report.obstacle_contacts
                heat_entries += report.heat_entries
                heat_escapes += report.heat_escapes
                reached = _mastery_reached(
                    config=config,
                    stage=stage,
                    primary_episodes=primary_episodes,
                    pickups=pickups,
                    obstacle_contacts=obstacle_contacts,
                    heat_entries=heat_entries,
                    heat_escapes=heat_escapes,
                )
                if (
                    mastered_stages
                    and primary_episodes % config.interleave_every == 0
                ):
                    replay_stage = mastered_stages[
                        (primary_episodes - 1)
                        % len(mastered_stages)
                    ]
                    replay_plan = EcologyTrainingEpisodePlan(
                        stage=replay_stage,
                        tier=EcologyTrainingTier.MEDIUM,
                        seed=(
                            config.seed
                            + 500_000
                            + stage_index * 10_000
                            + episode * 101
                        ),
                        episode_index=episode,
                        interleaved=True,
                        forced_escape=False,
                    )
                    runner, checkpoints, replay_report = (
                        await _run_training_episode(
                            config=config,
                            checkpoints=checkpoints,
                            arm=arm,
                            optimize=optimize,
                            local_valence_enabled=(
                                local_valence_enabled
                            ),
                            segment_credit_enabled=(
                                segment_credit_enabled
                            ),
                            plan=replay_plan,
                        )
                    )
                    plans.append(replay_plan)
                    reports.append(replay_report)
                if reached:
                    break
            mastery.append(
                EcologyStageMastery(
                    stage=stage,
                    reached=reached,
                    primary_episodes=primary_episodes,
                    pickups=pickups,
                    obstacle_contacts=obstacle_contacts,
                    heat_entries=heat_entries,
                    heat_escapes=heat_escapes,
                    threshold=_mastery_threshold(config, stage),
                )
            )
            if reached:
                mastered_stages.append(stage)
    if runner is None:
        raise RuntimeError("ecology training schedule must not be empty")
    archives = runner.export_learning_checkpoint_archives(
        checkpoint_prefix=f"ecology:{arm}:trained",
    )
    return (
        checkpoints,
        archives,
        tuple(plans),
        tuple(reports),
        tuple(mastery),
    )


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
    data_split: EcologyDataSplit,
    scenario: EcologyEvaluationScenario,
    seed: int,
) -> EcologyArmMetrics:
    scenario_stage = {
        EcologyEvaluationScenario.BUTTER_ONLY: EcologyStage.BUTTER,
        EcologyEvaluationScenario.WOOD_STICK_ROUTE: (
            EcologyStage.WOOD_STICK
        ),
        EcologyEvaluationScenario.HEAT_ROUTE_AVOIDANCE: (
            EcologyStage.BURNING_MATCH
        ),
        EcologyEvaluationScenario.HEAT_FORCED_ESCAPE: (
            EcologyStage.BURNING_MATCH
        ),
        EcologyEvaluationScenario.COMPOSITE: EcologyStage.COMPOSITE,
    }[scenario]
    forced_escape = (
        scenario is EcologyEvaluationScenario.HEAT_FORCED_ESCAPE
    )
    world = _world(
        config=config,
        stage=scenario_stage,
        seed=seed,
        data_split=data_split,
        tier=EcologyTrainingTier.FAR,
        forced_escape=forced_escape,
    )
    runner = KernelColonyRunner(
        world,
        base_config=_session_config(
            config=config,
            seed=seed,
            session_id=(
                f"ecology:{arm}:{data_split.value}:"
                f"{scenario.value}:{seed}"
            ),
            optimize=False,
            learning_enabled=False,
            sparse_exploration_enabled=False,
        ),
    )
    runner.restore_learning_checkpoints(checkpoints)
    rounds = (
        config.validation_rounds
        if data_split is EcologyDataSplit.VALIDATION
        else config.heldout_rounds
    )
    await runner.run(rounds)
    records = _flatten_records(runner)
    after = runner.export_learning_checkpoints(
        checkpoint_prefix=(
            f"ecology:{arm}:{data_split.value}:{scenario.value}:"
            f"{seed}:frozen-check"
        ),
        include_runtime_replay=False,
    )
    captured, settled, lineage, pending, staged, drop_count = _latest_replay_counts(runner)
    eligible_captures = captured - pending
    if eligible_captures < 0 or settled > eligible_captures:
        raise RuntimeError("runtime replay counters violate eligible settlement ordering")
    return EcologyArmMetrics(
        arm=arm,
        data_split=data_split,
        stage=scenario_stage,
        scenario=scenario,
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
        nonzero_ecology_payoffs=sum(
            abs(item.ecology_action_payoff) > 1e-12
            for item in records
        ),
        activated_sense_channels=_activated_sense_channels(records),
        first_pickup_tick=_first_tick(
            records,
            predicate=lambda item: item.at_food,
        ),
        first_obstacle_contact_tick=_first_tick(
            records,
            predicate=lambda item: item.obstacle_contact,
        ),
        first_heat_entry_tick=_first_tick(
            records,
            predicate=lambda item: item.entered_harmful_heat,
        ),
        first_heat_escape_tick=_first_tick(
            records,
            predicate=lambda item: item.escaped_harmful_heat,
        ),
        minimum_food_distance=_minimum_optional(
            tuple(item.nearest_food_distance for item in records)
        ),
        minimum_obstacle_distance=_minimum_optional(
            tuple(item.nearest_obstacle_distance for item in records)
        ),
        minimum_heat_distance=_minimum_optional(
            tuple(item.nearest_heat_distance for item in records)
        ),
        switch_count=sum(int(item.is_switching) for item in records),
        mean_persistence_steps=(
            sum(item.steps_since_switch for item in records)
            / len(records)
            if records
            else 0.0
        ),
        closed_segment_count=max(
            (
                item.runtime_closed_segments
                for item in records
            ),
            default=0,
        ),
        longest_segment_length=max(
            (
                item.runtime_longest_segment_length
                for item in records
            ),
            default=0,
        ),
        mean_absolute_turn_delta=_mean_absolute_turn_delta(records),
        policy_fingerprint_stable=(
            _policy_fingerprints(after)
            == _policy_fingerprints(checkpoints)
        ),
        temporal_fingerprint_stable=(
            tuple(item.temporal_fingerprint for item in after)
            == tuple(
                item.temporal_fingerprint for item in checkpoints
            )
        ),
    )


def _scenario_metrics(
    metrics: tuple[EcologyArmMetrics, ...],
    scenario: EcologyEvaluationScenario,
) -> tuple[EcologyArmMetrics, ...]:
    return tuple(
        item for item in metrics if item.scenario is scenario
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
