"""Kernel-driven colony runner with one isolated runtime owner per ant."""

from __future__ import annotations

from dataclasses import dataclass

from volvence_ant.env.ant_world import AntWorld
from volvence_ant.runtime.ant_session import (
    AntLearningCheckpoint,
    AntSession,
    AntSessionConfig,
    AntStepRecord,
)


@dataclass(frozen=True)
class ColonyRoundRecord:
    round_index: int
    delivered: int
    pickups: int
    trail_sense_events: int
    obstacle_contacts: int
    trail_mass: float
    trail_entropy: float | None
    ant_steps: tuple[AntStepRecord, ...]


class KernelColonyRunner:
    """Drive independent AntSessions against one environment snapshot bus."""

    def __init__(
        self,
        world: AntWorld,
        *,
        base_config: AntSessionConfig,
    ) -> None:
        self.world = world
        self.sessions = tuple(
            AntSession(
                world,
                config=AntSessionConfig(
                    temporal_latent_dim=base_config.temporal_latent_dim,
                    session_id=f"{base_config.session_id}:body:{body_id}",
                    seed=base_config.seed + body_id,
                    heading_noise=base_config.heading_noise,
                    step_noise=base_config.step_noise,
                    compass_gain=base_config.compass_gain,
                    compass_noise=base_config.compass_noise,
                    code_gain=base_config.code_gain,
                    rollout_config=base_config.rollout_config,
                    external_prediction_error_drive=(base_config.external_prediction_error_drive),
                    rare_heavy_enabled=base_config.rare_heavy_enabled,
                    joint_schedule=base_config.joint_schedule,
                    joint_apply_writeback=base_config.joint_apply_writeback,
                    joint_apply_policy_optimization=(base_config.joint_apply_policy_optimization),
                    temporal_policy=base_config.temporal_policy,
                    allow_live_substrate_mutation=(base_config.allow_live_substrate_mutation),
                    objective=base_config.objective,
                    sense_schema=base_config.sense_schema,
                ),
                body_id=body_id,
            )
            for body_id in range(world.n_bodies)
        )
        self.rounds: list[ColonyRoundRecord] = []

    def export_learning_checkpoints(self, *, checkpoint_prefix: str) -> tuple[AntLearningCheckpoint, ...]:
        return tuple(
            session.export_learning_checkpoint(checkpoint_id=f"{checkpoint_prefix}:body:{body_id}")
            for body_id, session in enumerate(self.sessions)
        )

    def restore_learning_checkpoints(self, checkpoints: tuple[AntLearningCheckpoint, ...]) -> None:
        if len(checkpoints) != len(self.sessions):
            raise ValueError(
                f"colony checkpoint count mismatch: expected={len(self.sessions)}, actual={len(checkpoints)}"
            )
        for session, checkpoint in zip(self.sessions, checkpoints, strict=True):
            session.restore_learning_checkpoint(checkpoint)

    async def step_round(self) -> ColonyRoundRecord:
        steps: list[AntStepRecord] = []
        trail_sense_events = 0
        for session in self.sessions:
            observation = session.holder.observation
            if abs(observation.trail_pher_left - observation.trail_pher_right) > 1e-8:
                trail_sense_events += 1
            steps.append(await session.step())
        trail_mass, trail_entropy = self.world.pheromone_metrics()
        record = ColonyRoundRecord(
            round_index=len(self.rounds),
            delivered=self.world.food_delivered,
            pickups=self.world.food_pickups,
            trail_sense_events=trail_sense_events,
            obstacle_contacts=sum(step.obstacle_contact for step in steps),
            trail_mass=trail_mass,
            trail_entropy=trail_entropy,
            ant_steps=tuple(steps),
        )
        self.rounds.append(record)
        return record

    async def run(self, rounds: int) -> tuple[ColonyRoundRecord, ...]:
        for _ in range(rounds):
            await self.step_round()
        return tuple(self.rounds)


__all__ = ["ColonyRoundRecord", "KernelColonyRunner"]
