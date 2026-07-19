"""``AntSession`` — the closed sense->think->act loop over the kernel.

Each ``step`` is one environment tick:

1. the substrate adapter (bound to the live sense holder) publishes the current
   sensorimotor snapshot inside ``AgentSessionRunner.run_turn``;
2. the kernel produces a controller code ``z_t`` (temporal owner, ACTIVE);
3. ``AntActuator`` + ``motor_decode`` turn ``z_t`` into a bounded command;
4. the navigator integrates the efference copy (path integration);
5. the world applies the command and returns the next observation;
6. the holder is refreshed for the next tick.

The kernel (``vz-temporal`` / ``vz-memory`` / ``vz-cognition``) is reused
verbatim through the ``vz-runtime`` facade — this module imports NONE of those
internals directly (enforced by ``tests/test_import_boundaries.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from volvence_zero.agent.session import AgentSessionRunner, AgentTurnResult
from volvence_zero.environment import (
    EnvironmentEventKind,
    EnvironmentMeasurement,
    EnvironmentOutcome,
)
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.runtime import stable_value_hash
from volvence_zero.substrate import SyntheticOpenWeightResidualRuntime
from volvence_zero.temporal_types import TemporalAbstractionSnapshot

from volvence_ant.env.ant_world import AntWorld, WorldObservation
from volvence_ant.substrate.ant_actuator import AntActuator, AntMotorCommand
from volvence_ant.substrate.ant_adapter import AntSenseHolder, AntSubstrateAdapter
from volvence_ant.substrate.navigator import AntNavigator, NavigatorState


class AntSessionError(RuntimeError):
    """The kernel turn did not expose the state the embodiment requires."""


@dataclass(frozen=True)
class AntLearningCheckpoint:
    """Opaque owner-exported state used for fair ant train/eval branches.

    The embodiment never reconstructs kernel state.  Each owner exports and
    restores its own immutable checkpoint through the runtime facade.
    """

    checkpoint_id: str
    temporal_state: object
    memory_state: object
    fingerprint: str


@dataclass
class AntSessionConfig:
    temporal_latent_dim: int = 4
    session_id: str = "digital-ant"
    seed: int = 0
    heading_noise: float = 0.01
    step_noise: float = 0.01
    code_gain: float = 4.0
    rollout_config: FinalRolloutConfig | None = None
    # Learning knobs exposed on the vz-runtime facade (used by matched-control
    # arms). ``external_prediction_error_drive=False`` -> PE-off arm.
    external_prediction_error_drive: bool = True
    rare_heavy_enabled: bool = False
    # Opaque ``JointLoopSchedule`` passthrough. The embodiment package must not
    # import that vz-temporal-internal type (import-boundary rule), so callers
    # in orchestration scripts construct it and pass it here as an object.
    joint_schedule: object | None = None
    joint_apply_writeback: bool = False
    temporal_policy: object | None = None
    allow_live_substrate_mutation: bool = False


@dataclass(frozen=True)
class AntStepRecord:
    tick: int
    x: float
    y: float
    true_heading: float
    est_heading: float
    carrying_food: bool
    at_food: bool
    at_nest: bool
    command: AntMotorCommand
    code: tuple[float, ...]
    switch_gate: float
    abstract_action: str
    homing_direction_error: float
    homing_distance_error: float
    environment_outcome_id: str
    prediction_id: str


class AntSession:
    """Reuses the kernel ``AgentSessionRunner`` to drive a digital ant body."""

    def __init__(
        self,
        world: AntWorld,
        *,
        config: AntSessionConfig | None = None,
        body_id: int = 0,
    ) -> None:
        self.world = world
        self.config = config or AntSessionConfig()
        self._body_id = body_id
        self.navigator = AntNavigator(
            step_size=world.config.step_size,
            heading_noise=self.config.heading_noise,
            step_noise=self.config.step_noise,
            seed=self.config.seed,
        )
        observation = world.observe(body_id)
        self.navigator.reset(initial_heading=observation.eval_true_heading)
        self.holder = AntSenseHolder(
            observation=observation,
            navigator_state=self.navigator.state,
            turn_command_scale=world.config.max_turn_rate,
            step=world.tick,
        )
        self.actuator = AntActuator(
            max_turn_rate=world.config.max_turn_rate,
            step_size=world.config.step_size,
            code_gain=self.config.code_gain,
        )
        runner_kwargs: dict = dict(
            session_id=self.config.session_id,
            config=self.config.rollout_config or FinalRolloutConfig(),
            temporal_latent_dim=self.config.temporal_latent_dim,
            substrate_adapter_factory=self._adapter_factory,
            default_residual_runtime=SyntheticOpenWeightResidualRuntime(
                model_id="digital-ant-fallback-runtime",
                allow_live_substrate_mutation=(
                    self.config.allow_live_substrate_mutation
                ),
            ),
            rare_heavy_enabled=self.config.rare_heavy_enabled,
            external_prediction_error_drive=self.config.external_prediction_error_drive,
            joint_apply_writeback=self.config.joint_apply_writeback,
            allow_live_substrate_mutation=self.config.allow_live_substrate_mutation,
        )
        if self.config.joint_schedule is not None:
            runner_kwargs["joint_schedule"] = self.config.joint_schedule
        if self.config.temporal_policy is not None:
            runner_kwargs["temporal_policy"] = self.config.temporal_policy
        self.runner = AgentSessionRunner(**runner_kwargs)
        self.trajectory: list[AntStepRecord] = []

    def export_learning_checkpoint(self, *, checkpoint_id: str) -> AntLearningCheckpoint:
        temporal_state = (
            self.runner.world_temporal_policy.parameter_store.export_parameter_snapshot()
        )
        memory_state = self.runner.memory_store.create_checkpoint(
            checkpoint_id=f"{checkpoint_id}:memory"
        )
        fingerprint = stable_value_hash((temporal_state, memory_state))
        return AntLearningCheckpoint(
            checkpoint_id=checkpoint_id,
            temporal_state=temporal_state,
            memory_state=memory_state,
            fingerprint=fingerprint,
        )

    def restore_learning_checkpoint(self, checkpoint: AntLearningCheckpoint) -> None:
        world_store = self.runner.world_temporal_policy.parameter_store
        world_store.restore_parameter_snapshot(checkpoint.temporal_state)
        self_store = self.runner.self_temporal_policy.parameter_store
        if self_store is not world_store:
            self_store.restore_parameter_snapshot(checkpoint.temporal_state)
        self.runner.memory_store.restore_checkpoint(checkpoint.memory_state)
        restored = self.export_learning_checkpoint(
            checkpoint_id=checkpoint.checkpoint_id
        )
        if restored.fingerprint != checkpoint.fingerprint:
            raise AntSessionError(
                "learning checkpoint restore changed owner-exported state: "
                f"expected={checkpoint.fingerprint}, actual={restored.fingerprint}"
            )

    def _adapter_factory(self, user_input: str, turn_index: int) -> AntSubstrateAdapter:
        return AntSubstrateAdapter(self.holder)

    @staticmethod
    def _read_code(result: AgentTurnResult) -> tuple[tuple[float, ...], float, str]:
        snapshot = result.active_snapshots.get("temporal_abstraction")
        if snapshot is None:
            raise AntSessionError(
                "temporal_abstraction snapshot missing from active chain; "
                "the temporal owner must be ACTIVE for the ant to move."
            )
        value = snapshot.value
        if not isinstance(value, TemporalAbstractionSnapshot):
            raise AntSessionError(
                f"temporal_abstraction slot carried {type(value)!r}, expected "
                "TemporalAbstractionSnapshot."
            )
        controller = value.controller_state
        return tuple(controller.code), float(controller.switch_gate), value.active_abstract_action

    async def step(self) -> AntStepRecord:
        result = await self.runner.run_turn(f"ant-tick-{self.world.tick}")
        code, switch_gate, abstract_action = self._read_code(result)
        command = self.actuator.plan(code, alarm=self.holder.observation.alarm)
        nav_state = self.navigator.update(
            turn_command=command.turn_command, step_command=command.step_command
        )
        observation = self.world.act(
            turn_command=command.turn_command,
            step_command=command.step_command,
            body_id=self._body_id,
        )
        transition = self.world.last_transition(self._body_id)
        prediction_id = (
            result.next_prediction.prediction_id
            if result.next_prediction is not None
            else ""
        )
        environment_outcome = self._environment_outcome(
            transition_id=transition.transition_id,
            delivered=transition.delivered,
            picked_up=transition.picked_up,
            prediction_id=prediction_id,
        )
        self.runner.submit_environment_outcome(environment_outcome)
        body = self.world.body(self._body_id)
        record = AntStepRecord(
            tick=self.world.tick,
            x=body.x,
            y=body.y,
            true_heading=body.heading,
            est_heading=nav_state.h_hat,
            carrying_food=observation.carrying_food,
            at_food=observation.at_food,
            at_nest=observation.at_nest,
            command=command,
            code=code,
            switch_gate=switch_gate,
            abstract_action=abstract_action,
            homing_direction_error=self._homing_direction_error(observation, nav_state),
            homing_distance_error=abs(nav_state.home_distance - observation.eval_home_distance),
            environment_outcome_id=environment_outcome.outcome_id,
            prediction_id=prediction_id,
        )
        self.holder.update(
            observation=observation, navigator_state=nav_state, step=self.world.tick
        )
        self.trajectory.append(record)
        return record

    @staticmethod
    def _environment_outcome(
        *,
        transition_id: str,
        delivered: bool,
        picked_up: bool,
        prediction_id: str,
    ) -> EnvironmentOutcome:
        status = "delivered" if delivered else ("picked_up" if picked_up else "moved")
        measurement = (
            EnvironmentMeasurement(
                task_progress=1.0,
                action_payoff=1.0,
                terminal=True,
            )
            if delivered
            else None
        )
        return EnvironmentOutcome(
            outcome_id=f"{transition_id}:outcome",
            event_id=transition_id,
            outcome_kind=EnvironmentEventKind.SCENE_EVENT,
            action_id=transition_id,
            status=status,
            summary=f"digital-ant transition {status}",
            detail="observable AntWorld pickup/delivery transition",
            prediction_id=prediction_id or None,
            evidence=(f"ant_transition:{transition_id}",),
            environment_state_delta_kind=status,
            measurement=measurement,
        )

    @staticmethod
    def _homing_direction_error(
        observation: WorldObservation, nav_state: NavigatorState
    ) -> float:
        estimated = nav_state.home_bearing
        true_bearing = observation.eval_home_bearing
        diff = (estimated - true_bearing + math.pi) % (2.0 * math.pi) - math.pi
        return abs(diff)

    async def run(self, ticks: int) -> list[AntStepRecord]:
        for _ in range(ticks):
            await self.step()
        return self.trajectory
