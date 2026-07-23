"""Paired ecology probes for sensor reachability and learned action sensitivity."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from volvence_ant.env import (
    AntWorld,
    AntWorldConfig,
    BurningMatch,
    ButterSource,
    WoodStick,
)
from volvence_ant.env.world_objects import WorldObject
from volvence_ant.runtime import (
    AntLearningCheckpoint,
    AntObjectiveKind,
    AntSenseSchema,
    AntSession,
    AntSessionConfig,
)
from volvence_ant.evidence.runtime_profile import (
    ant_runtime_replay_rollout_config,
)


class EcologyProbeKind(str, Enum):
    FOOD = "food"
    OBSTACLE = "obstacle"
    HEAT = "heat"
    HOME = "home"


@dataclass(frozen=True)
class EcologyActionProbe:
    kind: EcologyProbeKind
    left_sensor_pair: tuple[float, float]
    right_sensor_pair: tuple[float, float]
    left_code: tuple[float, ...]
    right_code: tuple[float, ...]
    left_turn: float
    right_turn: float
    code_l1_delta: float
    turn_delta: float
    input_reachable: bool
    action_sensitive: bool
    left_action_head_residual: tuple[float, ...] = ()
    right_action_head_residual: tuple[float, ...] = ()
    left_action_head_update_step: int = 0
    right_action_head_update_step: int = 0
    target_aligned: bool = True


@dataclass(frozen=True)
class EcologyCheckpointActionProbe:
    """Per-body action-chain evidence bound to one learning checkpoint."""

    body_id: int
    checkpoint_id: str
    policy_fingerprint: str
    temporal_learning_fingerprint: str
    probes: tuple[EcologyActionProbe, ...]


def _paired_objects(
    kind: EcologyProbeKind,
) -> tuple[WorldObject, WorldObject]:
    if kind is EcologyProbeKind.FOOD:
        return (
            ButterSource(object_id="probe-butter", x=0.6, y=0.35),
            ButterSource(object_id="probe-butter", x=0.6, y=-0.35),
        )
    if kind is EcologyProbeKind.OBSTACLE:
        return (
            WoodStick(
                object_id="probe-stick",
                start_x=0.5,
                start_y=0.15,
                end_x=0.8,
                end_y=0.45,
            ),
            WoodStick(
                object_id="probe-stick",
                start_x=0.5,
                start_y=-0.15,
                end_x=0.8,
                end_y=-0.45,
            ),
        )
    if kind is EcologyProbeKind.HEAT:
        return (
            BurningMatch(
                object_id="probe-match",
                x=0.6,
                y=0.35,
                heat_decay=0.8,
            ),
            BurningMatch(
                object_id="probe-match",
                x=0.6,
                y=-0.35,
                heat_decay=0.8,
            ),
        )
    if kind is EcologyProbeKind.HOME:
        # Both lanes see identical geometry; only carrying state differs.
        shared = ButterSource(
            object_id="probe-home-butter",
            x=8.0,
            y=0.0,
        )
        return (shared, shared)
    raise ValueError(f"unsupported ecology probe kind: {kind!r}")


def _sensor_pair(
    *,
    kind: EcologyProbeKind,
    observation: object,
) -> tuple[float, float]:
    from volvence_ant.env.ant_world import WorldObservation

    if not isinstance(observation, WorldObservation):
        raise TypeError(
            "ecology probe observation must be a WorldObservation, "
            f"got {type(observation).__name__}"
        )
    if kind is EcologyProbeKind.FOOD:
        return (observation.food_left, observation.food_right)
    if kind is EcologyProbeKind.OBSTACLE:
        return (observation.obstacle_left, observation.obstacle_right)
    if kind is EcologyProbeKind.HEAT:
        return (observation.heat_left, observation.heat_right)
    if kind is EcologyProbeKind.HOME:
        return (
            float(observation.carrying_food),
            observation.eval_home_distance,
        )
    raise ValueError(f"unsupported ecology probe kind: {kind!r}")


async def run_ecology_action_probes(
    *,
    temporal_latent_dim: int,
    seed: int,
    checkpoint: AntLearningCheckpoint | None = None,
    code_delta_threshold: float = 1e-8,
    turn_delta_threshold: float = 1e-8,
) -> tuple[EcologyActionProbe, ...]:
    probes: list[EcologyActionProbe] = []
    for kind in EcologyProbeKind:
        objects = _paired_objects(kind)
        paired_records: list[
            tuple[
                tuple[float, float],
                tuple[float, ...],
                float,
                tuple[float, ...],
                int,
            ]
        ] = []
        for side_index, world_object in enumerate(objects):
            world = AntWorld(
                config=AntWorldConfig(
                    seed=seed,
                    step_size=0.4,
                ),
                world_objects=(world_object,),
            )
            session = AntSession(
                world,
                config=AntSessionConfig(
                    temporal_latent_dim=temporal_latent_dim,
                    session_id=(
                        f"ecology-probe:{kind.value}:side:{side_index}:seed:{seed}"
                    ),
                    seed=seed,
                    heading_noise=0.0,
                    step_noise=0.0,
                    rollout_config=(
                        ant_runtime_replay_rollout_config(
                            enable_sparse_exploration=False,
                        )
                    ),
                    objective=AntObjectiveKind.ECOLOGY,
                    sense_schema=AntSenseSchema.ECOLOGY_V2,
                ),
            )
            if checkpoint is not None:
                session.restore_learning_checkpoint(checkpoint)
            if kind is EcologyProbeKind.HOME:
                for _ in range(5):
                    session.navigator.update(
                        turn_command=0.0,
                        step_command=0.4,
                        true_heading=0.0,
                    )
                session.navigator.update(
                    turn_command=math.pi / 2.0,
                    step_command=0.0,
                    true_heading=math.pi / 2.0,
                )
                world.set_body_pose(
                    x=2.0,
                    y=0.0,
                    heading=math.pi / 2.0,
                    carrying_food=bool(side_index),
                )
            else:
                world.set_body_pose(x=0.0, y=0.0, heading=0.0)
            observation = world.observe()
            record = await session.step()
            paired_records.append(
                (
                    _sensor_pair(kind=kind, observation=observation),
                    record.code,
                    record.command.turn_command,
                    record.causal_action_head_residual,
                    record.causal_action_head_update_step,
                )
            )
        left, right = paired_records
        code_l1_delta = sum(
            abs(left_value - right_value)
            for left_value, right_value in zip(
                left[1],
                right[1],
                strict=True,
            )
        )
        turn_delta = abs(left[2] - right[2])
        input_reachable = left[0] != right[0] and code_l1_delta > code_delta_threshold
        target_aligned = True
        if kind is EcologyProbeKind.HOME:
            # At (2, 0), heading north, home lies to the left (+turn).
            # The carrying lane is the right-hand member of this pair.
            target_aligned = right[2] > turn_delta_threshold
        probes.append(
            EcologyActionProbe(
                kind=kind,
                left_sensor_pair=left[0],
                right_sensor_pair=right[0],
                left_code=left[1],
                right_code=right[1],
                left_turn=left[2],
                right_turn=right[2],
                code_l1_delta=code_l1_delta,
                turn_delta=turn_delta,
                input_reachable=input_reachable,
                action_sensitive=input_reachable
                and turn_delta > turn_delta_threshold,
                left_action_head_residual=left[3],
                right_action_head_residual=right[3],
                left_action_head_update_step=left[4],
                right_action_head_update_step=right[4],
                target_aligned=target_aligned,
            )
        )
    return tuple(probes)


async def run_ecology_checkpoint_action_probes(
    *,
    temporal_latent_dim: int,
    seed: int,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    code_delta_threshold: float = 1e-8,
    turn_delta_threshold: float = 1e-4,
) -> tuple[EcologyCheckpointActionProbe, ...]:
    """Run deterministic paired probes for every isolated colony body."""

    reports: list[EcologyCheckpointActionProbe] = []
    for body_id, checkpoint in enumerate(checkpoints):
        probes = await run_ecology_action_probes(
            temporal_latent_dim=temporal_latent_dim,
            seed=seed,
            checkpoint=checkpoint,
            code_delta_threshold=code_delta_threshold,
            turn_delta_threshold=turn_delta_threshold,
        )
        reports.append(
            EcologyCheckpointActionProbe(
                body_id=body_id,
                checkpoint_id=checkpoint.checkpoint_id,
                policy_fingerprint=checkpoint.policy_fingerprint,
                temporal_learning_fingerprint=(
                    checkpoint.temporal_learning_fingerprint
                ),
                probes=probes,
            )
        )
    return tuple(reports)


__all__ = [
    "EcologyActionProbe",
    "EcologyCheckpointActionProbe",
    "EcologyProbeKind",
    "run_ecology_checkpoint_action_probes",
    "run_ecology_action_probes",
]
