"""Matched dynamic-colony benchmark with isolated regime-shift episodes.

The environment owns every perturbation. Controllers receive only the normal
local observation and immutable pheromone snapshot; benchmark readouts are
evaluation-only and never enter reward or the runtime learning path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from volvence_ant.controllers.fixed_rule_ant import FixedRuleAnt, FixedRuleConfig
from volvence_ant.controllers.random_ant import RandomAnt
from volvence_ant.env.ant_world import (
    AntWorld,
    AntWorldConfig,
    AxisAlignedObstacle,
    FoodSource,
    MotorDistortionProfile,
)
from volvence_ant.env.colony import ColonyWorld
from volvence_ant.env.pheromone_field import PheromoneBus
from volvence_ant.runtime import (
    AntLearningCheckpoint,
    AntSessionConfig,
    AntStepRecord,
    KernelColonyRunner,
)


class DynamicPerturbationKind(str, Enum):
    OBSTACLE_BLOCK = "obstacle_block"
    FOOD_RELOCATION = "food_relocation"
    MOTOR_BIAS = "motor_bias"


class DynamicColonyArmKind(str, Enum):
    LEARNED_BUS = "learned_bus"
    LEARNED_NO_BUS = "learned_no_bus"
    NO_OPTIMIZE_BUS = "no_optimize_bus"
    PE_OFF_BUS = "pe_off_bus"
    FIXED_RULE_BUS = "fixed_rule_bus"
    FIXED_RULE_NO_BUS = "fixed_rule_no_bus"
    RANDOM_NO_BUS = "random_no_bus"


@dataclass(frozen=True)
class DynamicColonyConfig:
    n_ants: int = 4
    training_rounds: int = 20
    pre_shift_rounds: int = 20
    post_shift_rounds: int = 40
    recovery_window: int = 10
    temporal_latent_dim: int = 16

    def __post_init__(self) -> None:
        if self.n_ants < 1:
            raise ValueError("n_ants must be >= 1")
        if self.training_rounds < 0:
            raise ValueError("training_rounds must be >= 0")
        if self.pre_shift_rounds < 1 or self.post_shift_rounds < 1:
            raise ValueError("pre_shift_rounds and post_shift_rounds must be >= 1")
        if self.recovery_window < 1:
            raise ValueError("recovery_window must be >= 1")
        if self.recovery_window > min(
            self.pre_shift_rounds, self.post_shift_rounds
        ):
            raise ValueError(
                "recovery_window must fit both pre-shift and post-shift phases"
            )
        if self.temporal_latent_dim < 2:
            raise ValueError("temporal_latent_dim must be >= 2")


@dataclass(frozen=True)
class RuntimeReplayCoverage:
    captured: int
    settled: int
    transitions: int
    lineage_matches: int
    settlement_coverage: float
    lineage_coverage: float
    drop_reasons: tuple[str, ...]


@dataclass(frozen=True)
class DynamicColonyArm:
    arm: str
    controller_kind: str
    bus_enabled: bool
    perturbation: str
    n_ants: int
    training_rounds: int
    pre_shift_rounds: int
    post_shift_rounds: int
    training_delivered: int
    training_pickups: int
    delivered: int
    pickups: int
    pre_shift_delivered: int
    post_shift_delivered: int
    pre_shift_pickups: int
    post_shift_pickups: int
    delivery_curve: tuple[int, ...]
    pickup_curve: tuple[int, ...]
    pre_shift_throughput_per_1k_actions: float
    post_shift_throughput_per_1k_actions: float
    post_shift_delivery_auc: float
    recovery_rounds: int | None
    oracle_delivery_shortfall: float
    total_applied_distance: float
    route_stretch: float | None
    obstacle_contacts: int
    shift_overlap_count: int
    trail_sense_events: int
    trail_mass: float
    trail_entropy: float | None
    trail_mass_curve: tuple[float, ...]
    trail_entropy_curve: tuple[float | None, ...]
    initial_checkpoint_fingerprints: tuple[str, ...]
    trained_policy_fingerprints: tuple[str, ...]
    shift_policy_fingerprints: tuple[str, ...]
    adaptation_start_policy_fingerprints: tuple[str, ...]
    final_policy_fingerprints: tuple[str, ...]
    policy_parameters_changed: bool
    post_shift_policy_parameters_changed: bool
    runtime_replay_captured: int
    runtime_replay_settled: int
    runtime_replay_transitions: int
    runtime_replay_lineage_matches: int
    runtime_replay_lineage_coverage: float
    runtime_replay_active: bool
    runtime_replay_per_ant: tuple[RuntimeReplayCoverage, ...]
    post_shift_runtime_replay_per_ant: tuple[RuntimeReplayCoverage, ...]
    diagnostic_breakpoint: str
    description: str


@dataclass(frozen=True)
class DynamicColonySeedReport:
    seed: int
    training_world_seed: int
    evaluation_world_seed: int
    perturbation: str
    config: DynamicColonyConfig
    arms: tuple[DynamicColonyArm, ...]


@dataclass(frozen=True)
class DynamicColonyEffect:
    effect_name: str
    values: tuple[float, ...]
    mean: float
    ci95: tuple[float, float]


@dataclass(frozen=True)
class DynamicColonyGate:
    gate_name: str
    passed: bool
    observed: float
    threshold: str


@dataclass(frozen=True)
class DynamicColonyAggregateReport:
    perturbation: str
    seeds: tuple[int, ...]
    seed_reports: tuple[DynamicColonySeedReport, ...]
    effects: tuple[DynamicColonyEffect, ...]
    gates: tuple[DynamicColonyGate, ...]
    verdict: str
    description: str


_TRAINING_FOOD = FoodSource(
    x=5.0,
    y=0.0,
    strength=2.5,
    decay=1.1,
    radius=2.0,
)
_EVALUATION_FOOD = FoodSource(
    x=0.0,
    y=5.0,
    strength=2.5,
    decay=1.1,
    radius=2.0,
)
_RELOCATED_FOOD = (-5.0, 0.0)
_BLOCKING_OBSTACLE = AxisAlignedObstacle(
    obstacle_id="regime-shift-barrier",
    min_x=-2.0,
    max_x=2.0,
    min_y=2.7,
    max_y=2.9,
)
_RUNTIME_REPLAY_TRACK_COUNT = 2


def _world_config(seed: int) -> AntWorldConfig:
    return AntWorldConfig(
        seed=seed,
        antenna_offset_deg=45.0,
        antenna_reach=1.3,
    )


def _training_world_seed(seed: int) -> int:
    return seed * 2 + 17


def _evaluation_world_seed(seed: int) -> int:
    return seed * 2 + 1_000_003


def _controller_seed(seed: int) -> int:
    return seed * 100


def _make_world(
    *,
    seed: int,
    n_ants: int,
    bus_enabled: bool,
    food_source: FoodSource,
) -> AntWorld:
    if bus_enabled:
        return ColonyWorld(
            config=_world_config(seed),
            food_sources=(food_source,),
            n_bodies=n_ants,
            bus=PheromoneBus(
                decay=0.008,
                deposit_amount=2.5,
                cell_size=1.0,
            ),
        )
    return AntWorld(
        config=_world_config(seed),
        food_sources=(food_source,),
        n_bodies=n_ants,
    )


def _apply_perturbation(
    world: AntWorld,
    *,
    perturbation: DynamicPerturbationKind,
) -> int:
    if perturbation is DynamicPerturbationKind.OBSTACLE_BLOCK:
        overlap_count = sum(
            _BLOCKING_OBSTACLE.contains(body.x, body.y)
            for body in (world.body(body_id) for body_id in range(world.n_bodies))
        )
        world.set_obstacles((_BLOCKING_OBSTACLE,))
        return overlap_count
    if perturbation is DynamicPerturbationKind.FOOD_RELOCATION:
        overlap_count = sum(
            math.hypot(
                world.body(body_id).x - _RELOCATED_FOOD[0],
                world.body(body_id).y - _RELOCATED_FOOD[1],
            )
            <= world.config.food_pickup_radius
            for body_id in range(world.n_bodies)
        )
        world.move_food(x=_RELOCATED_FOOD[0], y=_RELOCATED_FOOD[1])
        return overlap_count
    if perturbation is DynamicPerturbationKind.MOTOR_BIAS:
        for body_id in range(0, world.n_bodies, 2):
            world.set_motor_distortion(
                MotorDistortionProfile(turn_gain=1.0, turn_bias=0.18),
                body_id=body_id,
            )
        return 0
    raise ValueError(f"unsupported perturbation: {perturbation!r}")


def _phase_throughput(*, delivered: int, n_ants: int, rounds: int) -> float:
    return 1000.0 * delivered / (n_ants * rounds)


def _runtime_replay_coverage(
    record: AntStepRecord,
    *,
    previous: AntStepRecord | None = None,
) -> RuntimeReplayCoverage:
    previous_captured = (
        previous.runtime_replay_captured if previous is not None else 0
    )
    previous_settled = (
        previous.runtime_replay_settled if previous is not None else 0
    )
    previous_transitions = (
        previous.runtime_replay_transitions if previous is not None else 0
    )
    previous_lineage = (
        previous.runtime_replay_lineage_matches if previous is not None else 0
    )
    captured = record.runtime_replay_captured - previous_captured
    settled = record.runtime_replay_settled - previous_settled
    transitions = record.runtime_replay_transitions - previous_transitions
    lineage_matches = (
        record.runtime_replay_lineage_matches - previous_lineage
    )
    return RuntimeReplayCoverage(
        captured=captured,
        settled=settled,
        transitions=transitions,
        lineage_matches=lineage_matches,
        settlement_coverage=(
            settled / captured if captured > 0 else 0.0
        ),
        lineage_coverage=(
            lineage_matches / settled if settled > 0 else 0.0
        ),
        drop_reasons=record.runtime_replay_drop_reasons,
    )


def _recovery_rounds(
    *,
    curve: tuple[int, ...],
    pre_shift_rounds: int,
    post_shift_rounds: int,
    recovery_window: int,
    n_ants: int,
) -> int | None:
    pre_deliveries = curve[pre_shift_rounds - 1]
    baseline = _phase_throughput(
        delivered=pre_deliveries,
        n_ants=n_ants,
        rounds=pre_shift_rounds,
    )
    if baseline <= 0.0:
        return None
    target = 0.8 * baseline
    for post_end in range(recovery_window, post_shift_rounds + 1):
        absolute_end = pre_shift_rounds + post_end
        absolute_start = absolute_end - recovery_window
        deliveries = curve[absolute_end - 1] - curve[absolute_start - 1]
        throughput = _phase_throughput(
            delivered=deliveries,
            n_ants=n_ants,
            rounds=recovery_window,
        )
        if throughput >= target:
            return post_end
    return None


def _oracle_delivery_shortfall(
    *,
    world: AntWorld,
    post_curve: tuple[int, ...],
    post_shift_rounds: int,
    n_ants: int,
) -> float:
    food = world.food_sources()[0]
    nest_x, nest_y = world.nest
    center_distance = math.hypot(food.x - nest_x, food.y - nest_y)
    one_way_contact_distance = max(
        0.0,
        center_distance
        - world.config.nest_radius
        - world.config.food_pickup_radius,
    )
    lower_bound_round_trip = 2.0 * one_way_contact_distance
    oracle_interval = max(
        1,
        math.ceil(lower_bound_round_trip / world.config.step_size),
    )
    optimistic_oracle_deliveries = (
        n_ants * post_shift_rounds / oracle_interval
    )
    actual_deliveries = post_curve[-1] if post_curve else 0
    return max(
        0.0,
        optimistic_oracle_deliveries - actual_deliveries,
    ) / (n_ants * post_shift_rounds)


def _build_arm(
    *,
    arm: DynamicColonyArmKind,
    controller_kind: str,
    bus_enabled: bool,
    perturbation: DynamicPerturbationKind,
    config: DynamicColonyConfig,
    world: AntWorld,
    training_delivered: int,
    training_pickups: int,
    delivery_curve: tuple[int, ...],
    pickup_curve: tuple[int, ...],
    obstacle_contacts: int,
    shift_overlap_count: int,
    trail_sense_events: int,
    trail_mass_curve: tuple[float, ...],
    trail_entropy_curve: tuple[float | None, ...],
    total_applied_distance: float,
    initial_checkpoints: tuple[AntLearningCheckpoint, ...] = (),
    trained_checkpoints: tuple[AntLearningCheckpoint, ...] = (),
    shift_checkpoints: tuple[AntLearningCheckpoint, ...] = (),
    adaptation_start_checkpoints: tuple[AntLearningCheckpoint, ...] = (),
    final_checkpoints: tuple[AntLearningCheckpoint, ...] = (),
    runtime_replay_captured: int = 0,
    runtime_replay_settled: int = 0,
    runtime_replay_transitions: int = 0,
    runtime_replay_lineage_matches: int = 0,
    runtime_replay_active: bool = False,
    runtime_replay_per_ant: tuple[RuntimeReplayCoverage, ...] = (),
    post_shift_runtime_replay_per_ant: tuple[RuntimeReplayCoverage, ...] = (),
) -> DynamicColonyArm:
    pre_delivered = delivery_curve[config.pre_shift_rounds - 1]
    pre_pickups = pickup_curve[config.pre_shift_rounds - 1]
    post_delivered = world.food_delivered - pre_delivered
    post_pickups = world.food_pickups - pre_pickups
    post_curve = tuple(
        delivered - pre_delivered
        for delivered in delivery_curve[config.pre_shift_rounds :]
    )
    trail_mass = trail_mass_curve[-1]
    trail_entropy = trail_entropy_curve[-1]
    initial_fingerprints = tuple(
        checkpoint.fingerprint for checkpoint in initial_checkpoints
    )
    trained_policy_fingerprints = tuple(
        checkpoint.policy_fingerprint for checkpoint in trained_checkpoints
    )
    shift_policy_fingerprints = tuple(
        checkpoint.policy_fingerprint for checkpoint in shift_checkpoints
    )
    adaptation_start_policy_fingerprints = tuple(
        checkpoint.policy_fingerprint
        for checkpoint in adaptation_start_checkpoints
    )
    final_policy_fingerprints = tuple(
        checkpoint.policy_fingerprint for checkpoint in final_checkpoints
    )
    policy_changed = bool(
        initial_checkpoints
        and final_checkpoints
        and any(
            initial.policy_fingerprint != final.policy_fingerprint
            for initial, final in zip(
                initial_checkpoints, final_checkpoints, strict=True
            )
        )
    )
    post_shift_policy_changed = bool(
        adaptation_start_checkpoints
        and final_checkpoints
        and any(
            start.policy_fingerprint != final.policy_fingerprint
            for start, final in zip(
                adaptation_start_checkpoints,
                final_checkpoints,
                strict=True,
            )
        )
    )
    lineage_coverage = (
        runtime_replay_lineage_matches / runtime_replay_settled
        if runtime_replay_settled > 0
        else 0.0
    )
    if pre_pickups == 0:
        breakpoint = "static-qualification-no-food-contact"
    elif controller_kind == "kernel" and runtime_replay_settled == 0:
        breakpoint = "runtime-replay-not-settled"
    elif (
        controller_kind == "kernel"
        and runtime_replay_lineage_matches < runtime_replay_settled
    ):
        breakpoint = "runtime-replay-lineage-incomplete"
    elif post_pickups == 0:
        breakpoint = "post-shift-no-food-contact"
    else:
        breakpoint = "observable-success"
    recovery = _recovery_rounds(
        curve=delivery_curve,
        pre_shift_rounds=config.pre_shift_rounds,
        post_shift_rounds=config.post_shift_rounds,
        recovery_window=config.recovery_window,
        n_ants=config.n_ants,
    )
    post_auc = (
        float(sum(post_curve))
        / (config.n_ants * config.post_shift_rounds)
    )
    oracle_shortfall = _oracle_delivery_shortfall(
        world=world,
        post_curve=post_curve,
        post_shift_rounds=config.post_shift_rounds,
        n_ants=config.n_ants,
    )
    pre_throughput = _phase_throughput(
        delivered=pre_delivered,
        n_ants=config.n_ants,
        rounds=config.pre_shift_rounds,
    )
    post_throughput = _phase_throughput(
        delivered=post_delivered,
        n_ants=config.n_ants,
        rounds=config.post_shift_rounds,
    )
    nest_x, nest_y = world.nest
    initial_round_trip = 2.0 * max(
        0.0,
        math.hypot(
            _EVALUATION_FOOD.x - nest_x,
            _EVALUATION_FOOD.y - nest_y,
        )
        - world.config.nest_radius
        - world.config.food_pickup_radius,
    )
    final_food = world.food_sources()[0]
    post_round_trip = 2.0 * max(
        0.0,
        math.hypot(
            final_food.x - nest_x,
            final_food.y - nest_y,
        )
        - world.config.nest_radius
        - world.config.food_pickup_radius,
    )
    minimum_delivery_distance = (
        pre_delivered * initial_round_trip
        + post_delivered * post_round_trip
    )
    route_stretch = (
        total_applied_distance / minimum_delivery_distance
        if minimum_delivery_distance > 0.0
        else None
    )
    return DynamicColonyArm(
        arm=arm.value,
        controller_kind=controller_kind,
        bus_enabled=bus_enabled,
        perturbation=perturbation.value,
        n_ants=config.n_ants,
        training_rounds=config.training_rounds,
        pre_shift_rounds=config.pre_shift_rounds,
        post_shift_rounds=config.post_shift_rounds,
        training_delivered=training_delivered,
        training_pickups=training_pickups,
        delivered=world.food_delivered,
        pickups=world.food_pickups,
        pre_shift_delivered=pre_delivered,
        post_shift_delivered=post_delivered,
        pre_shift_pickups=pre_pickups,
        post_shift_pickups=post_pickups,
        delivery_curve=delivery_curve,
        pickup_curve=pickup_curve,
        pre_shift_throughput_per_1k_actions=pre_throughput,
        post_shift_throughput_per_1k_actions=post_throughput,
        post_shift_delivery_auc=post_auc,
        recovery_rounds=recovery,
        oracle_delivery_shortfall=oracle_shortfall,
        total_applied_distance=total_applied_distance,
        route_stretch=route_stretch,
        obstacle_contacts=obstacle_contacts,
        shift_overlap_count=shift_overlap_count,
        trail_sense_events=trail_sense_events,
        trail_mass=trail_mass,
        trail_entropy=trail_entropy,
        trail_mass_curve=trail_mass_curve,
        trail_entropy_curve=trail_entropy_curve,
        initial_checkpoint_fingerprints=initial_fingerprints,
        trained_policy_fingerprints=trained_policy_fingerprints,
        shift_policy_fingerprints=shift_policy_fingerprints,
        adaptation_start_policy_fingerprints=(
            adaptation_start_policy_fingerprints
        ),
        final_policy_fingerprints=final_policy_fingerprints,
        policy_parameters_changed=policy_changed,
        post_shift_policy_parameters_changed=post_shift_policy_changed,
        runtime_replay_captured=runtime_replay_captured,
        runtime_replay_settled=runtime_replay_settled,
        runtime_replay_transitions=runtime_replay_transitions,
        runtime_replay_lineage_matches=runtime_replay_lineage_matches,
        runtime_replay_lineage_coverage=lineage_coverage,
        runtime_replay_active=runtime_replay_active,
        runtime_replay_per_ant=runtime_replay_per_ant,
        post_shift_runtime_replay_per_ant=(
            post_shift_runtime_replay_per_ant
        ),
        diagnostic_breakpoint=breakpoint,
        description=(
            f"{arm.value}: pre={pre_throughput:.4f}/1k-actions, "
            f"post={post_throughput:.4f}/1k-actions, "
            f"recovery={recovery}, oracle_shortfall={oracle_shortfall:.4f}"
        ),
    )


async def _run_kernel_arm(
    *,
    arm: DynamicColonyArmKind,
    seed: int,
    perturbation: DynamicPerturbationKind,
    config: DynamicColonyConfig,
    session_config: AntSessionConfig,
    bus_enabled: bool,
    initial_checkpoints: tuple[AntLearningCheckpoint, ...],
) -> DynamicColonyArm:
    training_world = _make_world(
        seed=_training_world_seed(seed),
        n_ants=config.n_ants,
        bus_enabled=bus_enabled,
        food_source=_TRAINING_FOOD,
    )
    training_runner = KernelColonyRunner(
        training_world,
        base_config=replace(
            session_config,
            session_id=f"dynamic-colony:{arm.value}:{seed}:train",
            seed=_controller_seed(seed),
        ),
    )
    training_runner.restore_learning_checkpoints(initial_checkpoints)
    if config.training_rounds:
        await training_runner.run(config.training_rounds)
    trained_checkpoints = training_runner.export_learning_checkpoints(
        checkpoint_prefix=f"dynamic-colony:{arm.value}:{seed}:trained",
        include_runtime_replay=False,
    )

    world = _make_world(
        seed=_evaluation_world_seed(seed),
        n_ants=config.n_ants,
        bus_enabled=bus_enabled,
        food_source=_EVALUATION_FOOD,
    )
    runner = KernelColonyRunner(
        world,
        base_config=replace(
            session_config,
            session_id=f"dynamic-colony:{arm.value}:{seed}:eval",
            seed=_controller_seed(seed),
        ),
    )
    runner.restore_learning_checkpoints(trained_checkpoints)
    await runner.run(config.pre_shift_rounds)
    shift_checkpoints = runner.export_learning_checkpoints(
        checkpoint_prefix=f"dynamic-colony:{arm.value}:{seed}:shift"
    )
    shift_overlap_count = _apply_perturbation(
        world,
        perturbation=perturbation,
    )
    # Runtime replay settles action t on turn t+1. The first post-shift turn
    # therefore acts as an explicit latency boundary: it settles the final
    # pre-shift outcome, executes post action 1, then exposes a policy
    # checkpoint uncontaminated by any post outcome. Adaptation evidence starts
    # from this checkpoint and covers post outcomes 1..N-1.
    await runner.run(1)
    adaptation_start_checkpoints = runner.export_learning_checkpoints(
        checkpoint_prefix=(
            f"dynamic-colony:{arm.value}:{seed}:adaptation-start"
        )
    )
    if config.post_shift_rounds > 1:
        await runner.run(config.post_shift_rounds - 1)
    final_checkpoints = runner.export_learning_checkpoints(
        checkpoint_prefix=f"dynamic-colony:{arm.value}:{seed}:final"
    )
    rounds = tuple(runner.rounds)
    latest_records = rounds[-1].ant_steps if rounds else ()
    adaptation_start_records = (
        rounds[config.pre_shift_rounds].ant_steps if rounds else ()
    )
    return _build_arm(
        arm=arm,
        controller_kind="kernel",
        bus_enabled=bus_enabled,
        perturbation=perturbation,
        config=config,
        world=world,
        training_delivered=training_world.food_delivered,
        training_pickups=training_world.food_pickups,
        delivery_curve=tuple(record.delivered for record in rounds),
        pickup_curve=tuple(record.pickups for record in rounds),
        obstacle_contacts=sum(record.obstacle_contacts for record in rounds),
        shift_overlap_count=shift_overlap_count,
        trail_sense_events=sum(record.trail_sense_events for record in rounds),
        trail_mass_curve=tuple(record.trail_mass for record in rounds),
        trail_entropy_curve=tuple(record.trail_entropy for record in rounds),
        total_applied_distance=sum(
            step.applied_step
            for record in rounds
            for step in record.ant_steps
        ),
        initial_checkpoints=initial_checkpoints,
        trained_checkpoints=trained_checkpoints,
        shift_checkpoints=shift_checkpoints,
        adaptation_start_checkpoints=adaptation_start_checkpoints,
        final_checkpoints=final_checkpoints,
        runtime_replay_captured=sum(
            record.runtime_replay_captured for record in latest_records
        ),
        runtime_replay_settled=sum(
            record.runtime_replay_settled for record in latest_records
        ),
        runtime_replay_transitions=sum(
            record.runtime_replay_transitions for record in latest_records
        ),
        runtime_replay_lineage_matches=sum(
            record.runtime_replay_lineage_matches for record in latest_records
        ),
        runtime_replay_active=bool(latest_records)
        and all(
            dict(record.backend_wiring)["internal_rl_runtime_replay"]
            == "active"
            for record in latest_records
        ),
        runtime_replay_per_ant=tuple(
            _runtime_replay_coverage(record)
            for record in latest_records
        ),
        post_shift_runtime_replay_per_ant=tuple(
            _runtime_replay_coverage(record, previous=start_record)
            for record, start_record in zip(
                latest_records,
                adaptation_start_records,
                strict=True,
            )
        ),
    )


def _run_fixed_rule_arm(
    *,
    arm: DynamicColonyArmKind,
    seed: int,
    perturbation: DynamicPerturbationKind,
    config: DynamicColonyConfig,
    bus_enabled: bool,
) -> DynamicColonyArm:
    world = _make_world(
        seed=_evaluation_world_seed(seed),
        n_ants=config.n_ants,
        bus_enabled=bus_enabled,
        food_source=_EVALUATION_FOOD,
    )
    ants = tuple(
        FixedRuleAnt(
            world,
            config=FixedRuleConfig(
                seed=_controller_seed(seed) + body_id,
                food_sense_threshold=0.18,
                trail_gain=14.0,
            ),
            body_id=body_id,
        )
        for body_id in range(config.n_ants)
    )
    delivery_curve: list[int] = []
    pickup_curve: list[int] = []
    obstacle_contacts = 0
    trail_sense_events = 0
    total_applied_distance = 0.0
    trail_mass_curve: list[float] = []
    trail_entropy_curve: list[float | None] = []
    shift_overlap_count = 0
    total_rounds = config.pre_shift_rounds + config.post_shift_rounds
    for round_index in range(total_rounds):
        if round_index == config.pre_shift_rounds:
            shift_overlap_count = _apply_perturbation(
                world,
                perturbation=perturbation,
            )
        for body_id, ant in enumerate(ants):
            step = ant.step()
            trail_sense_events += step.mode == "trail-follow"
            transition = world.last_transition(body_id)
            obstacle_contacts += transition.blocked_by_obstacle
            total_applied_distance += transition.applied_step
        delivery_curve.append(world.food_delivered)
        pickup_curve.append(world.food_pickups)
        trail_mass, trail_entropy = world.pheromone_metrics()
        trail_mass_curve.append(trail_mass)
        trail_entropy_curve.append(trail_entropy)
    return _build_arm(
        arm=arm,
        controller_kind="fixed_rule",
        bus_enabled=bus_enabled,
        perturbation=perturbation,
        config=config,
        world=world,
        training_delivered=0,
        training_pickups=0,
        delivery_curve=tuple(delivery_curve),
        pickup_curve=tuple(pickup_curve),
        obstacle_contacts=obstacle_contacts,
        shift_overlap_count=shift_overlap_count,
        trail_sense_events=trail_sense_events,
        trail_mass_curve=tuple(trail_mass_curve),
        trail_entropy_curve=tuple(trail_entropy_curve),
        total_applied_distance=total_applied_distance,
    )


def _run_random_arm(
    *,
    seed: int,
    perturbation: DynamicPerturbationKind,
    config: DynamicColonyConfig,
) -> DynamicColonyArm:
    world = _make_world(
        seed=_evaluation_world_seed(seed),
        n_ants=config.n_ants,
        bus_enabled=False,
        food_source=_EVALUATION_FOOD,
    )
    ants = tuple(
        RandomAnt(
            world,
            seed=_controller_seed(seed) + body_id,
            body_id=body_id,
        )
        for body_id in range(config.n_ants)
    )
    delivery_curve: list[int] = []
    pickup_curve: list[int] = []
    obstacle_contacts = 0
    total_applied_distance = 0.0
    trail_mass_curve: list[float] = []
    trail_entropy_curve: list[float | None] = []
    shift_overlap_count = 0
    total_rounds = config.pre_shift_rounds + config.post_shift_rounds
    for round_index in range(total_rounds):
        if round_index == config.pre_shift_rounds:
            shift_overlap_count = _apply_perturbation(
                world,
                perturbation=perturbation,
            )
        for body_id, ant in enumerate(ants):
            ant.step()
            transition = world.last_transition(body_id)
            obstacle_contacts += transition.blocked_by_obstacle
            total_applied_distance += transition.applied_step
        delivery_curve.append(world.food_delivered)
        pickup_curve.append(world.food_pickups)
        trail_mass, trail_entropy = world.pheromone_metrics()
        trail_mass_curve.append(trail_mass)
        trail_entropy_curve.append(trail_entropy)
    return _build_arm(
        arm=DynamicColonyArmKind.RANDOM_NO_BUS,
        controller_kind="random",
        bus_enabled=False,
        perturbation=perturbation,
        config=config,
        world=world,
        training_delivered=0,
        training_pickups=0,
        delivery_curve=tuple(delivery_curve),
        pickup_curve=tuple(pickup_curve),
        obstacle_contacts=obstacle_contacts,
        shift_overlap_count=shift_overlap_count,
        trail_sense_events=0,
        trail_mass_curve=tuple(trail_mass_curve),
        trail_entropy_curve=tuple(trail_entropy_curve),
        total_applied_distance=total_applied_distance,
    )


async def run_dynamic_colony_seed(
    *,
    seed: int,
    perturbation: DynamicPerturbationKind,
    config: DynamicColonyConfig,
    learned_config: AntSessionConfig,
    no_optimize_config: AntSessionConfig,
    pe_off_config: AntSessionConfig,
) -> DynamicColonySeedReport:
    """Run one paired seed unit across the frozen v1 arm matrix."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if learned_config.temporal_latent_dim != config.temporal_latent_dim:
        raise ValueError("learned_config temporal_latent_dim mismatch")
    for name, arm_config in (
        ("no_optimize", no_optimize_config),
        ("pe_off", pe_off_config),
    ):
        if arm_config.temporal_latent_dim != config.temporal_latent_dim:
            raise ValueError(f"{name}_config temporal_latent_dim mismatch")

    bootstrap_world = _make_world(
        seed=_training_world_seed(seed),
        n_ants=config.n_ants,
        bus_enabled=False,
        food_source=_TRAINING_FOOD,
    )
    bootstrap_runner = KernelColonyRunner(
        bootstrap_world,
        base_config=replace(
            learned_config,
            session_id=f"dynamic-colony:{seed}:shared-initial",
            seed=_controller_seed(seed),
        ),
    )
    initial_checkpoints = bootstrap_runner.export_learning_checkpoints(
        checkpoint_prefix=f"dynamic-colony:{seed}:shared-initial",
        include_runtime_replay=False,
    )

    arms = [
        await _run_kernel_arm(
            arm=DynamicColonyArmKind.LEARNED_BUS,
            seed=seed,
            perturbation=perturbation,
            config=config,
            session_config=learned_config,
            bus_enabled=True,
            initial_checkpoints=initial_checkpoints,
        ),
        await _run_kernel_arm(
            arm=DynamicColonyArmKind.LEARNED_NO_BUS,
            seed=seed,
            perturbation=perturbation,
            config=config,
            session_config=learned_config,
            bus_enabled=False,
            initial_checkpoints=initial_checkpoints,
        ),
        await _run_kernel_arm(
            arm=DynamicColonyArmKind.NO_OPTIMIZE_BUS,
            seed=seed,
            perturbation=perturbation,
            config=config,
            session_config=no_optimize_config,
            bus_enabled=True,
            initial_checkpoints=initial_checkpoints,
        ),
        await _run_kernel_arm(
            arm=DynamicColonyArmKind.PE_OFF_BUS,
            seed=seed,
            perturbation=perturbation,
            config=config,
            session_config=pe_off_config,
            bus_enabled=True,
            initial_checkpoints=initial_checkpoints,
        ),
        _run_fixed_rule_arm(
            arm=DynamicColonyArmKind.FIXED_RULE_BUS,
            seed=seed,
            perturbation=perturbation,
            config=config,
            bus_enabled=True,
        ),
        _run_fixed_rule_arm(
            arm=DynamicColonyArmKind.FIXED_RULE_NO_BUS,
            seed=seed,
            perturbation=perturbation,
            config=config,
            bus_enabled=False,
        ),
        _run_random_arm(
            seed=seed,
            perturbation=perturbation,
            config=config,
        ),
    ]
    return DynamicColonySeedReport(
        seed=seed,
        training_world_seed=_training_world_seed(seed),
        evaluation_world_seed=_evaluation_world_seed(seed),
        perturbation=perturbation.value,
        config=config,
        arms=tuple(arms),
    )


def _bootstrap_effect(
    *,
    effect_name: str,
    values: tuple[float, ...],
    bootstrap_seed: int,
) -> DynamicColonyEffect:
    if not values:
        raise ValueError("effect values must be non-empty")
    samples = np.asarray(values, dtype=float)
    if len(values) == 1:
        ci = (float(samples[0]), float(samples[0]))
    else:
        rng = np.random.default_rng(bootstrap_seed)
        means = np.asarray(
            [
                rng.choice(samples, size=len(samples), replace=True).mean()
                for _ in range(4000)
            ],
            dtype=float,
        )
        ci = (
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        )
    return DynamicColonyEffect(
        effect_name=effect_name,
        values=values,
        mean=float(samples.mean()),
        ci95=ci,
    )


def aggregate_dynamic_colony_reports(
    reports: tuple[DynamicColonySeedReport, ...],
    *,
    seed_order: tuple[int, ...],
) -> DynamicColonyAggregateReport:
    """Aggregate paired seed reports and apply the frozen v1 gates."""

    if len(seed_order) < 1:
        raise ValueError("seed_order must be non-empty")
    if len(set(seed_order)) != len(seed_order):
        raise ValueError("seed_order must contain distinct seeds")
    if any(seed < 0 for seed in seed_order):
        raise ValueError("seed_order must contain non-negative seeds")
    by_seed = {report.seed: report for report in reports}
    if len(by_seed) != len(reports):
        raise ValueError("dynamic colony reports contain duplicate seeds")
    if set(by_seed) != set(seed_order):
        raise ValueError("dynamic colony report seeds do not match seed_order")
    ordered = tuple(by_seed[seed] for seed in seed_order)
    perturbations = {report.perturbation for report in ordered}
    if len(perturbations) != 1:
        raise ValueError("cannot aggregate different perturbation kinds")
    benchmark_config = ordered[0].config
    if any(report.config != benchmark_config for report in ordered[1:]):
        raise ValueError("dynamic colony reports do not share one config")

    expected_arms = {arm.value for arm in DynamicColonyArmKind}
    arms_by_seed: list[dict[str, DynamicColonyArm]] = []
    for report in ordered:
        if (
            report.training_world_seed != _training_world_seed(report.seed)
            or report.evaluation_world_seed
            != _evaluation_world_seed(report.seed)
        ):
            raise ValueError(
                f"seed {report.seed} world-seed derivation mismatch"
            )
        arm_map = {arm.arm: arm for arm in report.arms}
        if set(arm_map) != expected_arms:
            raise ValueError(
                f"seed {report.seed} arm mismatch: "
                f"actual={tuple(sorted(arm_map))}"
            )
        for arm in arm_map.values():
            if (
                arm.n_ants != benchmark_config.n_ants
                or arm.training_rounds != benchmark_config.training_rounds
                or arm.pre_shift_rounds != benchmark_config.pre_shift_rounds
                or arm.post_shift_rounds != benchmark_config.post_shift_rounds
            ):
                raise ValueError(
                    f"seed {report.seed} arm {arm.arm} config mismatch"
                )
        arms_by_seed.append(arm_map)

    def paired_difference(
        treatment: DynamicColonyArmKind,
        control: DynamicColonyArmKind,
        attribute: str,
    ) -> tuple[float, ...]:
        return tuple(
            float(getattr(arms[treatment.value], attribute))
            - float(getattr(arms[control.value], attribute))
            for arms in arms_by_seed
        )

    def relative_improvement(
        treatment: DynamicColonyArmKind,
        control: DynamicColonyArmKind,
        attribute: str,
        *,
        lower_is_better: bool,
    ) -> tuple[float, ...]:
        values: list[float] = []
        for arms in arms_by_seed:
            treatment_value = float(getattr(arms[treatment.value], attribute))
            control_value = float(getattr(arms[control.value], attribute))
            if abs(control_value) <= 1e-12:
                values.append(0.0)
                continue
            denominator = abs(control_value)
            numerator = (
                control_value - treatment_value
                if lower_is_better
                else treatment_value - control_value
            )
            values.append(numerator / denominator)
        return tuple(values)

    post_attribute = "post_shift_throughput_per_1k_actions"
    effects = (
        _bootstrap_effect(
            effect_name="learned_bus_minus_learned_no_bus_post_throughput",
            values=paired_difference(
                DynamicColonyArmKind.LEARNED_BUS,
                DynamicColonyArmKind.LEARNED_NO_BUS,
                post_attribute,
            ),
            bootstrap_seed=seed_order[0] + 101,
        ),
        _bootstrap_effect(
            effect_name="learned_minus_no_optimize_post_throughput",
            values=paired_difference(
                DynamicColonyArmKind.LEARNED_BUS,
                DynamicColonyArmKind.NO_OPTIMIZE_BUS,
                post_attribute,
            ),
            bootstrap_seed=seed_order[0] + 102,
        ),
        _bootstrap_effect(
            effect_name="learned_minus_no_optimize_pre_throughput",
            values=paired_difference(
                DynamicColonyArmKind.LEARNED_BUS,
                DynamicColonyArmKind.NO_OPTIMIZE_BUS,
                "pre_shift_throughput_per_1k_actions",
            ),
            bootstrap_seed=seed_order[0] + 107,
        ),
        _bootstrap_effect(
            effect_name="learned_minus_pe_off_post_throughput",
            values=paired_difference(
                DynamicColonyArmKind.LEARNED_BUS,
                DynamicColonyArmKind.PE_OFF_BUS,
                post_attribute,
            ),
            bootstrap_seed=seed_order[0] + 103,
        ),
        _bootstrap_effect(
            effect_name=(
                "learned_vs_fixed_post_throughput_relative_to_fixed_pre"
            ),
            values=tuple(
                (
                    arms[
                        DynamicColonyArmKind.LEARNED_BUS.value
                    ].post_shift_throughput_per_1k_actions
                    - arms[
                        DynamicColonyArmKind.FIXED_RULE_BUS.value
                    ].post_shift_throughput_per_1k_actions
                )
                / (
                    arms[
                        DynamicColonyArmKind.FIXED_RULE_BUS.value
                    ].pre_shift_throughput_per_1k_actions
                )
                if arms[
                    DynamicColonyArmKind.FIXED_RULE_BUS.value
                ].pre_shift_throughput_per_1k_actions
                > 1e-12
                else 0.0
                for arms in arms_by_seed
            ),
            bootstrap_seed=seed_order[0] + 104,
        ),
        _bootstrap_effect(
            effect_name="learned_vs_fixed_recovery_relative",
            values=tuple(
                (
                    (
                        arms[
                            DynamicColonyArmKind.FIXED_RULE_BUS.value
                        ].recovery_rounds
                        or ordered[index].config.post_shift_rounds + 1
                    )
                    - (
                        arms[
                            DynamicColonyArmKind.LEARNED_BUS.value
                        ].recovery_rounds
                        or ordered[index].config.post_shift_rounds + 1
                    )
                )
                / max(
                    (
                        arms[
                            DynamicColonyArmKind.FIXED_RULE_BUS.value
                        ].recovery_rounds
                        or ordered[index].config.post_shift_rounds + 1
                    ),
                    1,
                )
                for index, arms in enumerate(arms_by_seed)
            ),
            bootstrap_seed=seed_order[0] + 105,
        ),
        _bootstrap_effect(
            effect_name="learned_vs_fixed_shortfall_relative",
            values=relative_improvement(
                DynamicColonyArmKind.LEARNED_BUS,
                DynamicColonyArmKind.FIXED_RULE_BUS,
                "oracle_delivery_shortfall",
                lower_is_better=True,
            ),
            bootstrap_seed=seed_order[0] + 106,
        ),
    )
    effect_by_name = {effect.effect_name: effect for effect in effects}

    learned_arms = tuple(
        arms[DynamicColonyArmKind.LEARNED_BUS.value] for arms in arms_by_seed
    )
    fixed_arms = tuple(
        arms[DynamicColonyArmKind.FIXED_RULE_BUS.value] for arms in arms_by_seed
    )
    no_optimize_arms = tuple(
        arms[DynamicColonyArmKind.NO_OPTIMIZE_BUS.value]
        for arms in arms_by_seed
    )
    kernel_arms = tuple(
        arm
        for arms in arms_by_seed
        for arm in arms.values()
        if arm.controller_kind == "kernel"
    )
    learned_pre = float(
        np.mean(
            [arm.pre_shift_throughput_per_1k_actions for arm in learned_arms]
        )
    )
    fixed_pre = float(
        np.mean(
            [arm.pre_shift_throughput_per_1k_actions for arm in fixed_arms]
        )
    )
    no_optimize_pre = float(
        np.mean(
            [
                arm.pre_shift_throughput_per_1k_actions
                for arm in no_optimize_arms
            ]
        )
    )
    static_ratio = learned_pre / max(fixed_pre, 1e-12)
    pickup_seed_rate = float(
        np.mean([arm.pre_shift_pickups > 0 for arm in learned_arms])
    )
    fixed_pre_pickup_seed_rate = float(
        np.mean([arm.pre_shift_pickups > 0 for arm in fixed_arms])
    )
    replay_slices = tuple(
        coverage
        for arm in kernel_arms
        for coverage in (
            *arm.runtime_replay_per_ant,
            *arm.post_shift_runtime_replay_per_ant,
        )
    )
    replay_coverage = min(
        (
            min(
                coverage.settlement_coverage,
                coverage.lineage_coverage,
            )
            for coverage in replay_slices
        ),
        default=0.0,
    )

    def replay_slice_valid(
        coverage: RuntimeReplayCoverage,
        *,
        expected_captured: int,
        expected_settled: int,
        expected_transitions: int,
    ) -> bool:
        return bool(
            expected_transitions > 0
            and coverage.captured == expected_captured
            and coverage.settled == expected_settled
            and coverage.transitions == expected_transitions
            and coverage.lineage_matches == expected_transitions
            and coverage.transitions == coverage.settled
            and 0
            <= coverage.lineage_matches
            <= coverage.settled
            <= coverage.captured
            and coverage.settlement_coverage >= 0.99
            and coverage.lineage_coverage >= 0.99
            and not coverage.drop_reasons
        )

    expected_full_captured = _RUNTIME_REPLAY_TRACK_COUNT * (
        benchmark_config.pre_shift_rounds
        + benchmark_config.post_shift_rounds
        - 1
    )
    expected_full_settled = _RUNTIME_REPLAY_TRACK_COUNT * (
        benchmark_config.pre_shift_rounds
        + benchmark_config.post_shift_rounds
        - 2
    )
    expected_post_transitions = _RUNTIME_REPLAY_TRACK_COUNT * (
        benchmark_config.post_shift_rounds - 1
    )

    replay_integrity = float(
        bool(kernel_arms)
        and all(
            arm.runtime_replay_active
            and len(arm.runtime_replay_per_ant)
            == benchmark_config.n_ants
            and len(arm.post_shift_runtime_replay_per_ant)
            == benchmark_config.n_ants
            and all(
                replay_slice_valid(
                    coverage,
                    expected_captured=expected_full_captured,
                    expected_settled=expected_full_settled,
                    expected_transitions=expected_full_settled,
                )
                for coverage in arm.runtime_replay_per_ant
            )
            and all(
                replay_slice_valid(
                    coverage,
                    expected_captured=expected_post_transitions,
                    expected_settled=expected_post_transitions,
                    expected_transitions=expected_post_transitions,
                )
                for coverage in arm.post_shift_runtime_replay_per_ant
            )
            for arm in kernel_arms
        )
    )
    checkpoint_cardinality = float(
        all(
            len(arm.initial_checkpoint_fingerprints)
            == benchmark_config.n_ants
            and len(arm.trained_policy_fingerprints)
            == benchmark_config.n_ants
            and len(arm.shift_policy_fingerprints)
            == benchmark_config.n_ants
            and len(arm.adaptation_start_policy_fingerprints)
            == benchmark_config.n_ants
            and len(arm.final_policy_fingerprints)
            == benchmark_config.n_ants
            for arm in kernel_arms
        )
    )
    initial_alignment = float(
        all(
            len(
                {
                    tuple(arm.initial_checkpoint_fingerprints)
                    for arm in arms.values()
                    if arm.controller_kind == "kernel"
                }
            )
            == 1
            for arms in arms_by_seed
        )
    )
    learned_policy_diverged = float(
        all(arm.policy_parameters_changed for arm in learned_arms)
    )
    no_optimize_policy_stable = float(
        all(not arm.policy_parameters_changed for arm in no_optimize_arms)
    )
    learned_post_shift_policy_diverged = float(
        all(arm.post_shift_policy_parameters_changed for arm in learned_arms)
    )
    no_optimize_post_shift_policy_stable = float(
        all(
            not arm.post_shift_policy_parameters_changed
            for arm in no_optimize_arms
        )
    )
    no_shift_overlap = float(
        all(
            arm.shift_overlap_count == 0
            for arms in arms_by_seed
            for arm in arms.values()
        )
    )

    bus_effect = effect_by_name[
        "learned_bus_minus_learned_no_bus_post_throughput"
    ]
    no_opt_effect = effect_by_name[
        "learned_minus_no_optimize_post_throughput"
    ]
    no_opt_pre_effect = effect_by_name[
        "learned_minus_no_optimize_pre_throughput"
    ]
    pe_effect = effect_by_name["learned_minus_pe_off_post_throughput"]
    throughput_relative = effect_by_name[
        "learned_vs_fixed_post_throughput_relative_to_fixed_pre"
    ]
    recovery_relative = effect_by_name["learned_vs_fixed_recovery_relative"]
    shortfall_relative = effect_by_name[
        "learned_vs_fixed_shortfall_relative"
    ]
    pre_alignment_tolerance = max(
        0.02,
        0.10 * max(abs(learned_pre), abs(no_optimize_pre), 1.0),
    )

    gates = (
        DynamicColonyGate(
            gate_name="formal_seed_count",
            passed=len(seed_order) >= 10,
            observed=float(len(seed_order)),
            threshold=">=10",
        ),
        DynamicColonyGate(
            gate_name="formal_colony_size",
            passed=benchmark_config.n_ants >= 8,
            observed=float(benchmark_config.n_ants),
            threshold=">=8",
        ),
        DynamicColonyGate(
            gate_name="formal_training_budget",
            passed=benchmark_config.training_rounds >= 200,
            observed=float(benchmark_config.training_rounds),
            threshold=">=200 rounds/ant",
        ),
        DynamicColonyGate(
            gate_name="formal_pre_shift_budget",
            passed=benchmark_config.pre_shift_rounds >= 50,
            observed=float(benchmark_config.pre_shift_rounds),
            threshold=">=50 rounds",
        ),
        DynamicColonyGate(
            gate_name="formal_post_shift_budget",
            passed=benchmark_config.post_shift_rounds >= 100,
            observed=float(benchmark_config.post_shift_rounds),
            threshold=">=100 rounds",
        ),
        DynamicColonyGate(
            gate_name="formal_recovery_window",
            passed=benchmark_config.recovery_window >= 20,
            observed=float(benchmark_config.recovery_window),
            threshold=">=20 rounds",
        ),
        DynamicColonyGate(
            gate_name="static_throughput_comparability",
            passed=0.8 <= static_ratio <= 1.2,
            observed=static_ratio,
            threshold="0.8<=learned/fixed<=1.2",
        ),
        DynamicColonyGate(
            gate_name="static_pickup_seed_rate",
            passed=pickup_seed_rate >= 0.8,
            observed=pickup_seed_rate,
            threshold=">=0.8",
        ),
        DynamicColonyGate(
            gate_name="fixed_rule_pre_shift_competency",
            passed=fixed_pre_pickup_seed_rate >= 0.8,
            observed=fixed_pre_pickup_seed_rate,
            threshold="pickup in >=0.8 of seeds",
        ),
        DynamicColonyGate(
            gate_name="learned_bus_causal_effect",
            passed=bus_effect.mean >= 0.02 and bus_effect.ci95[0] > 0.0,
            observed=bus_effect.mean,
            threshold="mean>=0.02 and ci95.low>0",
        ),
        DynamicColonyGate(
            gate_name="policy_optimization_lifecycle_effect",
            passed=no_opt_effect.mean >= 0.02 and no_opt_effect.ci95[0] > 0.0,
            observed=no_opt_effect.mean,
            threshold="mean>=0.02 and ci95.low>0",
        ),
        DynamicColonyGate(
            gate_name="pre_shift_learned_no_optimize_alignment",
            passed=(
                abs(no_opt_pre_effect.mean) <= pre_alignment_tolerance
                and no_opt_pre_effect.ci95[0] <= 0.0
                <= no_opt_pre_effect.ci95[1]
            ),
            observed=no_opt_pre_effect.mean,
            threshold=(
                f"abs(mean)<={pre_alignment_tolerance:.6f} and ci95 spans 0"
            ),
        ),
        DynamicColonyGate(
            gate_name="prediction_error_causal_effect",
            passed=pe_effect.mean >= 0.02 and pe_effect.ci95[0] > 0.0,
            observed=pe_effect.mean,
            threshold="mean>=0.02 and ci95.low>0",
        ),
        DynamicColonyGate(
            gate_name="post_shift_throughput_advantage",
            passed=(
                throughput_relative.mean >= 0.10
                and throughput_relative.ci95[0] > 0.0
            ),
            observed=throughput_relative.mean,
            threshold="relative>=0.10 and ci95.low>0",
        ),
        DynamicColonyGate(
            gate_name="recovery_time_advantage",
            passed=(
                recovery_relative.mean >= 0.20
                and recovery_relative.ci95[0] > 0.0
            ),
            observed=recovery_relative.mean,
            threshold="relative>=0.20 and ci95.low>0",
        ),
        DynamicColonyGate(
            gate_name="oracle_delivery_shortfall_advantage",
            passed=(
                shortfall_relative.mean >= 0.15
                and shortfall_relative.ci95[0] > 0.0
            ),
            observed=shortfall_relative.mean,
            threshold="relative>=0.15 and ci95.low>0",
        ),
        DynamicColonyGate(
            gate_name="runtime_replay_lineage",
            passed=bool(replay_integrity),
            observed=replay_coverage,
            threshold=(
                "every kernel arm/ant full+post slice ACTIVE, transitions>0, "
                "0<=matches<=settled<=captured, both coverages>=0.99, no drops"
            ),
        ),
        DynamicColonyGate(
            gate_name="checkpoint_cardinality",
            passed=bool(checkpoint_cardinality),
            observed=checkpoint_cardinality,
            threshold="one owner checkpoint per ant",
        ),
        DynamicColonyGate(
            gate_name="initial_checkpoint_alignment",
            passed=bool(initial_alignment),
            observed=initial_alignment,
            threshold="==1",
        ),
        DynamicColonyGate(
            gate_name="learned_policy_diverged",
            passed=bool(learned_policy_diverged),
            observed=learned_policy_diverged,
            threshold="==1",
        ),
        DynamicColonyGate(
            gate_name="no_optimize_policy_stable",
            passed=bool(no_optimize_policy_stable),
            observed=no_optimize_policy_stable,
            threshold="==1",
        ),
        DynamicColonyGate(
            gate_name="learned_post_shift_policy_diverged",
            passed=bool(learned_post_shift_policy_diverged),
            observed=learned_post_shift_policy_diverged,
            threshold="==1",
        ),
        DynamicColonyGate(
            gate_name="no_optimize_post_shift_policy_stable",
            passed=bool(no_optimize_post_shift_policy_stable),
            observed=no_optimize_post_shift_policy_stable,
            threshold="==1",
        ),
        DynamicColonyGate(
            gate_name="shift_has_no_body_overlap",
            passed=bool(no_shift_overlap),
            observed=no_shift_overlap,
            threshold="==1",
        ),
    )
    verdict = "PASS" if all(gate.passed for gate in gates) else "BLOCK"
    perturbation = next(iter(perturbations))
    return DynamicColonyAggregateReport(
        perturbation=perturbation,
        seeds=seed_order,
        seed_reports=ordered,
        effects=effects,
        gates=gates,
        verdict=verdict,
        description=(
            f"dynamic colony {perturbation}: verdict={verdict}, "
            f"static_ratio={static_ratio:.4f}, "
            f"bus_effect={bus_effect.mean:.4f}, "
            f"post_shift_vs_fixed={throughput_relative.mean:.4f}"
        ),
    )


__all__ = [
    "DynamicColonyAggregateReport",
    "DynamicColonyArm",
    "DynamicColonyArmKind",
    "DynamicColonyConfig",
    "DynamicColonyEffect",
    "DynamicColonyGate",
    "DynamicColonySeedReport",
    "DynamicPerturbationKind",
    "RuntimeReplayCoverage",
    "aggregate_dynamic_colony_reports",
    "run_dynamic_colony_seed",
]
