"""Paired ecology probes for sensor reachability and learned action sensitivity."""

from __future__ import annotations

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
            tuple[tuple[float, float], tuple[float, ...], float]
        ] = []
        for side_index, world_object in enumerate(objects):
            world = AntWorld(
                config=AntWorldConfig(
                    seed=seed,
                    step_size=0.4,
                ),
                world_objects=(world_object,),
            )
            world.set_body_pose(x=0.0, y=0.0, heading=0.0)
            observation = world.observe()
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
            record = await session.step()
            paired_records.append(
                (
                    _sensor_pair(kind=kind, observation=observation),
                    record.code,
                    record.command.turn_command,
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
            )
        )
    return tuple(probes)


__all__ = [
    "EcologyActionProbe",
    "EcologyProbeKind",
    "run_ecology_action_probes",
]
