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
from enum import Enum

from volvence_zero.agent.session import (
    AgentLearningCheckpoint,
    AgentSessionRunner,
    AgentTurnResult,
)
from volvence_zero.environment import (
    EnvironmentEventKind,
    EnvironmentMeasurement,
    EnvironmentOutcome,
)
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.substrate import (
    SubstrateSnapshot,
    SyntheticOpenWeightResidualRuntime,
)
from volvence_zero.temporal_types import TemporalAbstractionSnapshot

from volvence_ant.env.ant_world import (
    AntWorld,
    WorldObservation,
    WorldTransitionEvidence,
)
from volvence_ant.substrate.ant_actuator import AntActuator, AntMotorCommand
from volvence_ant.substrate.ant_adapter import AntSenseHolder, AntSubstrateAdapter
from volvence_ant.substrate.navigator import AntNavigator, NavigatorState
from volvence_ant.substrate.sense_encode import AntSenseSchema, sense_channels


class AntSessionError(RuntimeError):
    """The kernel turn did not expose the state the embodiment requires."""


AntLearningCheckpoint = AgentLearningCheckpoint


class AntObjectiveKind(str, Enum):
    FORAGING = "foraging"
    HEADING_STABILITY = "heading_stability"
    ECOLOGY = "ecology"


@dataclass
class AntSessionConfig:
    temporal_latent_dim: int = 4
    session_id: str = "digital-ant"
    seed: int = 0
    heading_noise: float = 0.01
    step_noise: float = 0.01
    # Sky-compass (absolute-heading) fusion — a body sensor shared by all
    # navigation, matching AntBot's celestial compass. compass_gain=0 recovers
    # pure dead reckoning.
    compass_gain: float = 0.85
    compass_noise: float = 0.007
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
    # Matched-control gate for Internal-RL policy/critic persistence. False
    # still runs the same SSL, rollout and optimizer evidence, then restores the
    # post-SSL/pre-RL checkpoint so no reward-driven policy update accumulates.
    joint_apply_policy_optimization: bool = True
    joint_learning_enabled: bool = True
    temporal_policy: object | None = None
    allow_live_substrate_mutation: bool = False
    objective: AntObjectiveKind = AntObjectiveKind.FORAGING
    sense_schema: AntSenseSchema = AntSenseSchema.V1
    ecology_local_valence_enabled: bool = True


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
    applied_turn: float
    applied_step: float
    obstacle_contact: bool
    code: tuple[float, ...]
    switch_gate: float
    is_switching: bool
    steps_since_switch: int
    closed_segment_count: int
    abstract_action: str
    homing_direction_error: float
    homing_distance_error: float
    heading_stability_error: float
    motor_execution_error: float
    environment_outcome_id: str
    prediction_id: str
    pe_magnitude: float
    pe_bootstrap: bool
    signed_reward: float
    cumulative_credit: float
    memory_entries_total: int
    memory_pending_promotions: int
    memory_pending_decays: int
    cms_total_observations: int
    bounded_writeback_applied: bool
    joint_schedule_action: str
    writeback_operations: tuple[str, ...]
    backend_wiring: tuple[tuple[str, str], ...]
    runtime_replay_captured: int
    runtime_replay_settled: int
    runtime_replay_transitions: int
    runtime_replay_lineage_matches: int
    runtime_replay_drop_reasons: tuple[str, ...]
    runtime_replay_pending_captures: int
    runtime_replay_staged_rollouts: int
    heat_center: float
    heat_harmful: bool
    entered_harmful_heat: bool
    escaped_harmful_heat: bool
    ecology_action_payoff: float
    local_food_delta: float
    local_home_delta: float
    local_cooling_delta: float
    runtime_open_segment_transitions: int
    runtime_closed_segments: int
    runtime_longest_segment_length: int
    sense_activation: tuple[tuple[str, float], ...]
    nearest_food_distance: float | None
    nearest_obstacle_distance: float | None
    nearest_heat_distance: float | None
    causal_action_head_residual: tuple[float, ...] = ()
    causal_action_head_wiring: str = "disabled"
    causal_action_head_update_step: int = 0
    track_switch_gates: tuple[tuple[str, float], ...] = ()
    runtime_last_segment_close_reason: str = ""
    runtime_segment_close_reason_counts: tuple[tuple[str, int], ...] = ()
    fast_prior_strength: float = 0.0
    fast_prior_switch_pressure_delta: float = 0.0
    prediction_error_switch_pressure_delta: float = 0.0
    body_id: int = 0
    picked_up: bool = False
    delivered: bool = False


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
            compass_gain=self.config.compass_gain,
            compass_noise=self.config.compass_noise,
            seed=self.config.seed,
        )
        observation = world.observe(body_id)
        self._heading_stability_target = observation.eval_true_heading
        self._previous_heading_stability_error = 0.0
        self.navigator.reset(initial_heading=observation.eval_true_heading)
        self.holder = AntSenseHolder(
            observation=observation,
            navigator_state=self.navigator.state,
            turn_command_scale=world.config.max_turn_rate,
            sense_schema=self.config.sense_schema,
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
            runtime_exploration_context=f"seed:{self.config.seed}",
            substrate_adapter_factory=self._adapter_factory,
            default_residual_runtime=SyntheticOpenWeightResidualRuntime(
                model_id="digital-ant-fallback-runtime",
                allow_live_substrate_mutation=(self.config.allow_live_substrate_mutation),
            ),
            rare_heavy_enabled=self.config.rare_heavy_enabled,
            external_prediction_error_drive=self.config.external_prediction_error_drive,
            joint_apply_writeback=self.config.joint_apply_writeback,
            joint_apply_policy_optimization=(self.config.joint_apply_policy_optimization),
            joint_learning_enabled=self.config.joint_learning_enabled,
            allow_live_substrate_mutation=self.config.allow_live_substrate_mutation,
        )
        if self.config.temporal_policy is None:
            runner_kwargs["temporal_input_dim"] = len(
                sense_channels(self.config.sense_schema)
            )
        if self.config.joint_schedule is not None:
            runner_kwargs["joint_schedule"] = self.config.joint_schedule
        if self.config.temporal_policy is not None:
            runner_kwargs["temporal_policy"] = self.config.temporal_policy
        self.runner = AgentSessionRunner(**runner_kwargs)
        self.trajectory: list[AntStepRecord] = []

    def export_learning_checkpoint(
        self,
        *,
        checkpoint_id: str,
        include_runtime_replay: bool = True,
    ) -> AgentLearningCheckpoint:
        return self.runner.export_learning_checkpoint(
            checkpoint_id=checkpoint_id,
            include_runtime_replay=include_runtime_replay,
        )

    def restore_learning_checkpoint(self, checkpoint: AgentLearningCheckpoint) -> None:
        self.runner.restore_learning_checkpoint(checkpoint)

    def export_learning_checkpoint_archive(
        self,
        *,
        checkpoint_id: str,
    ) -> bytes:
        return self.runner.export_learning_checkpoint_archive(
            checkpoint_id=checkpoint_id,
        )

    def restore_learning_checkpoint_archive(
        self,
        archive: bytes,
        *,
        expected_state_fingerprint: str | None = None,
    ) -> None:
        self.runner.restore_learning_checkpoint_archive(
            archive,
            expected_state_fingerprint=expected_state_fingerprint,
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
                f"temporal_abstraction slot carried {type(value)!r}, expected TemporalAbstractionSnapshot."
            )
        controller = value.controller_state
        return tuple(controller.code), float(controller.switch_gate), value.active_abstract_action

    async def step(self) -> AntStepRecord:
        # Refresh at the action boundary so every colony member consumes the
        # same currently published round snapshot. Keeping only the post-action
        # observation would make the last body uniquely see the next bus tick.
        navigator_before = self.navigator.state
        self.holder.update(
            observation=self.world.observe(self._body_id),
            navigator_state=navigator_before,
            step=self.world.tick,
        )
        result = await self.runner.run_turn(f"ant-tick-{self.world.tick}")
        code, switch_gate, abstract_action = self._read_code(result)
        command = self.actuator.plan(code, alarm=self.holder.observation.alarm)
        observation = self.world.act(
            turn_command=command.turn_command,
            step_command=command.step_command,
            body_id=self._body_id,
        )
        # The navigator integrates the efference copy and fuses the noisy
        # sky-compass reading of the post-move absolute heading.
        nav_state = self.navigator.update(
            turn_command=command.turn_command,
            step_command=command.step_command,
            true_heading=observation.eval_true_heading,
        )
        home_progress = (
            navigator_before.home_distance - nav_state.home_distance
        )
        transition = self.world.last_transition(self._body_id)
        prediction_id = result.next_prediction.prediction_id if result.next_prediction is not None else ""
        environment_outcome = self._environment_outcome(
            transition=transition,
            observation=observation,
            prediction_id=prediction_id,
            home_progress=home_progress,
        )
        self.runner.submit_environment_outcome(environment_outcome)
        environment_measurement = environment_outcome.measurement
        prediction_error = result.prediction_error
        temporal_snapshot = result.active_snapshots["temporal_abstraction"].value
        prediction_error_snapshot = result.active_snapshots.get("prediction_error")
        credit_snapshot = result.active_snapshots.get("credit")
        memory_snapshot = result.active_snapshots.get("memory")
        cumulative_credit = (
            sum(value for _, value in credit_snapshot.value.cumulative_credit_by_level)
            if credit_snapshot is not None
            else 0.0
        )
        memory_value = memory_snapshot.value if memory_snapshot is not None else None
        memory_entries_total = (
            sum(value for _, value in memory_value.total_entries_by_stratum) if memory_value is not None else 0
        )
        cms_total_observations = (
            memory_value.cms_state.total_observations
            if memory_value is not None and memory_value.cms_state is not None
            else 0
        )
        rollout = self.runner.rollout_config
        replay = result.runtime_replay_report
        body = self.world.body(self._body_id)
        ecology_diagnostics = self.world.ecology_diagnostics(
            self._body_id
        )
        substrate_snapshot = result.active_snapshots["substrate"].value
        if not isinstance(substrate_snapshot, SubstrateSnapshot):
            raise AntSessionError(
                "substrate slot must carry SubstrateSnapshot for ant "
                "diagnostics"
            )
        if not substrate_snapshot.residual_activations:
            raise AntSessionError(
                "ant substrate must publish residual activation diagnostics"
            )
        channels = sense_channels(self.config.sense_schema)
        activation = (
            substrate_snapshot.residual_activations[-1].activation
        )
        if len(channels) != len(activation):
            raise AntSessionError(
                "ant sense diagnostic width mismatch: "
                f"channels={len(channels)}, activation={len(activation)}"
            )
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
            applied_turn=transition.applied_turn,
            applied_step=transition.applied_step,
            obstacle_contact=transition.blocked_by_obstacle,
            code=code,
            switch_gate=switch_gate,
            is_switching=temporal_snapshot.controller_state.is_switching,
            steps_since_switch=temporal_snapshot.controller_state.steps_since_switch,
            closed_segment_count=len(temporal_snapshot.closed_segments),
            abstract_action=abstract_action,
            homing_direction_error=self._homing_direction_error(observation, nav_state),
            homing_distance_error=abs(nav_state.home_distance - observation.eval_home_distance),
            heading_stability_error=self._heading_stability_error(observation),
            motor_execution_error=abs(transition.applied_turn - transition.commanded_turn),
            environment_outcome_id=environment_outcome.outcome_id,
            prediction_id=prediction_id,
            pe_magnitude=prediction_error.magnitude if prediction_error is not None else 0.0,
            pe_bootstrap=(prediction_error_snapshot.value.bootstrap if prediction_error_snapshot is not None else True),
            signed_reward=(prediction_error.signed_reward if prediction_error is not None else 0.0),
            cumulative_credit=float(cumulative_credit),
            memory_entries_total=memory_entries_total,
            memory_pending_promotions=(memory_value.pending_promotions if memory_value is not None else 0),
            memory_pending_decays=(memory_value.pending_decays if memory_value is not None else 0),
            cms_total_observations=cms_total_observations,
            bounded_writeback_applied=result.bounded_writeback_applied,
            joint_schedule_action=result.joint_schedule_action,
            writeback_operations=result.writeback_operations,
            backend_wiring=(
                ("temporal_runtime_backend", rollout.temporal_runtime_backend.value),
                ("temporal_ssl_backend", rollout.temporal_ssl_backend.value),
                ("internal_rl_backend", rollout.internal_rl_backend.value),
                (
                    "internal_rl_runtime_replay",
                    rollout.internal_rl_runtime_replay.value,
                ),
                (
                    "internal_rl_runtime_segment_credit",
                    rollout.internal_rl_runtime_segment_credit.value,
                ),
                (
                    "internal_rl_causal_action_head",
                    rollout.internal_rl_causal_action_head.value,
                ),
                (
                    "prediction_error_temporal_switch",
                    rollout.prediction_error_temporal_switch.value,
                ),
                (
                    "internal_rl_runtime_exploration_strength",
                    f"{rollout.internal_rl_runtime_exploration_strength:.6f}",
                ),
                ("cms_torch_backend", rollout.cms_torch_backend.value),
            ),
            runtime_replay_captured=(replay.captured_count if replay is not None else 0),
            runtime_replay_settled=(replay.settled_count if replay is not None else 0),
            runtime_replay_transitions=(replay.transition_count if replay is not None else 0),
            runtime_replay_lineage_matches=(replay.lineage_match_count if replay is not None else 0),
            runtime_replay_drop_reasons=(replay.drop_reasons if replay is not None else ()),
            runtime_replay_pending_captures=(replay.pending_capture_count if replay is not None else 0),
            runtime_replay_staged_rollouts=(replay.staged_rollout_count if replay is not None else 0),
            heat_center=observation.heat_center,
            heat_harmful=observation.heat_harmful,
            entered_harmful_heat=transition.entered_harmful_heat,
            escaped_harmful_heat=transition.escaped_harmful_heat,
            ecology_action_payoff=(
                environment_measurement.action_payoff
                if environment_measurement is not None
                else 0.0
            ),
            local_food_delta=(
                transition.local_food_signal_after
                - transition.local_food_signal_before
            ),
            local_home_delta=(
                home_progress
            ),
            local_cooling_delta=(
                transition.heat_load_before
                - transition.heat_load_after
            ),
            runtime_open_segment_transitions=(
                replay.open_segment_transition_count
                if replay is not None
                else 0
            ),
            runtime_closed_segments=(
                replay.closed_segment_count if replay is not None else 0
            ),
            runtime_longest_segment_length=(
                replay.longest_segment_length if replay is not None else 0
            ),
            sense_activation=tuple(
                zip(channels, activation, strict=True)
            ),
            nearest_food_distance=(
                ecology_diagnostics.nearest_food_distance
            ),
            nearest_obstacle_distance=(
                ecology_diagnostics.nearest_obstacle_distance
            ),
            nearest_heat_distance=(
                ecology_diagnostics.nearest_heat_distance
            ),
            causal_action_head_residual=(
                result.metacontroller_state.causal_action_head_residual
                if result.metacontroller_state is not None
                else ()
            ),
            causal_action_head_wiring=(
                result.metacontroller_state.causal_action_head_wiring
                if result.metacontroller_state is not None
                else "disabled"
            ),
            causal_action_head_update_step=(
                result.metacontroller_state.causal_action_head_update_step
                if result.metacontroller_state is not None
                else 0
            ),
            track_switch_gates=(
                result.metacontroller_state.track_switch_gates
                if result.metacontroller_state is not None
                else ()
            ),
            runtime_last_segment_close_reason=(
                replay.last_segment_close_reason
                if replay is not None
                else ""
            ),
            runtime_segment_close_reason_counts=(
                replay.segment_close_reason_counts
                if replay is not None
                else ()
            ),
            fast_prior_strength=(
                result.metacontroller_state.fast_prior_strength
                if result.metacontroller_state is not None
                else 0.0
            ),
            fast_prior_switch_pressure_delta=(
                result.metacontroller_state.fast_prior_switch_pressure_delta
                if result.metacontroller_state is not None
                else 0.0
            ),
            prediction_error_switch_pressure_delta=(
                result.metacontroller_state.prediction_error_switch_pressure_delta
                if result.metacontroller_state is not None
                else 0.0
            ),
            body_id=self._body_id,
            picked_up=transition.picked_up,
            delivered=transition.delivered,
        )
        self.holder.update(observation=observation, navigator_state=nav_state, step=self.world.tick)
        self.trajectory.append(record)
        return record

    def _environment_outcome(
        self,
        *,
        transition: WorldTransitionEvidence,
        observation: WorldObservation,
        prediction_id: str,
        home_progress: float = 0.0,
    ) -> EnvironmentOutcome:
        if self.config.objective is AntObjectiveKind.HEADING_STABILITY:
            heading_error = self._heading_stability_error(observation)
            task_progress = max(0.0, min(1.0, 1.0 - heading_error))
            payoff = max(
                -1.0,
                min(1.0, self._previous_heading_stability_error - heading_error),
            )
            self._previous_heading_stability_error = heading_error
            return EnvironmentOutcome(
                outcome_id=f"{transition.transition_id}:outcome",
                event_id=transition.transition_id,
                outcome_kind=EnvironmentEventKind.SCENE_EVENT,
                action_id=transition.transition_id,
                status="heading_stability_observed",
                summary="digital-ant heading stability transition",
                detail="observable sky-compass heading deviation",
                prediction_id=prediction_id or None,
                evidence=(f"ant_transition:{transition.transition_id}",),
                environment_state_delta_kind="heading_stability",
                measurement=EnvironmentMeasurement(
                    task_progress=task_progress,
                    action_payoff=payoff,
                    terminal=False,
                ),
            )

        if self.config.objective is AntObjectiveKind.ECOLOGY:
            return self._ecology_environment_outcome(
                transition=transition,
                prediction_id=prediction_id,
                home_progress=home_progress,
            )

        # Observable task facts (NOT rewards; the PE owner compares them with
        # prior predictions). Foraging has exactly two genuine, discrete,
        # observable milestones: picking food up (carrying False->True) and
        # delivering it home (terminal). We publish a partial task_progress on
        # pickup and full on delivery. We deliberately do NOT emit any
        # distance-to-food shaping: a continuous "closer = better" signal would
        # hand the controller the gradient-following answer the FSM hardcodes,
        # which is exactly the skill the controller is supposed to LEARN.
        status = "delivered" if transition.delivered else ("picked_up" if transition.picked_up else "moved")
        if transition.delivered:
            measurement = EnvironmentMeasurement(task_progress=1.0, action_payoff=1.0, terminal=True)
        elif transition.picked_up:
            measurement = EnvironmentMeasurement(task_progress=0.5, action_payoff=0.5, terminal=False)
        else:
            measurement = None
        return EnvironmentOutcome(
            outcome_id=f"{transition.transition_id}:outcome",
            event_id=transition.transition_id,
            outcome_kind=EnvironmentEventKind.SCENE_EVENT,
            action_id=transition.transition_id,
            status=status,
            summary=f"digital-ant transition {status}",
            detail="observable AntWorld pickup/delivery milestone",
            prediction_id=prediction_id or None,
            evidence=(f"ant_transition:{transition.transition_id}",),
            environment_state_delta_kind=status,
            measurement=measurement,
        )

    def _ecology_environment_outcome(
        self,
        *,
        transition: WorldTransitionEvidence,
        prediction_id: str,
        home_progress: float = 0.0,
    ) -> EnvironmentOutcome:
        """Publish owner-authored local valence without a steering direction."""

        facts: list[str] = []
        task_progress: float | None = None
        payoff = 0.0
        valenced = False
        terminal = False
        if transition.delivered:
            facts.append("delivered")
            task_progress = 1.0
            payoff += 1.0
            terminal = True
            valenced = True
        elif transition.picked_up:
            facts.append("picked_up")
            task_progress = 0.5
            payoff += 0.5
            valenced = True
        if transition.blocked_by_obstacle:
            # Wood sticks are neutral physical geometry: contact stays an
            # observable fact but must never contribute payoff/valence.
            facts.append("obstacle_contact")
        if transition.entered_harmful_heat:
            facts.append("heat_exposure_started")
            payoff -= 1.0
            valenced = True
        elif transition.heat_harmful_after:
            facts.append("heat_exposure_continued")
            payoff -= 0.4
            valenced = True
        elif transition.escaped_harmful_heat:
            facts.append("heat_exposure_ended")
            payoff += 0.35
            valenced = True
        food_delta = (
            transition.local_food_signal_after
            - transition.local_food_signal_before
        )
        home_delta = home_progress
        cooling_delta = transition.heat_load_before - transition.heat_load_after
        local_valence = 0.0
        if self.config.ecology_local_valence_enabled:
            if not transition.carrying_before and not transition.picked_up:
                local_valence += 0.45 * math.tanh(food_delta)
            if transition.carrying_before and not transition.delivered:
                local_valence += 0.45 * math.tanh(home_delta)
            local_valence += 0.7 * math.tanh(cooling_delta)
            payoff += local_valence
            if abs(local_valence) > 1e-9:
                valenced = True
                if not facts:
                    facts.append("local_valence")
        status = "+".join(facts) if facts else "moved"
        measurement = (
            EnvironmentMeasurement(
                task_progress=task_progress,
                action_payoff=max(-1.0, min(1.0, payoff)),
                terminal=terminal,
            )
            if valenced
            else None
        )
        return EnvironmentOutcome(
            outcome_id=f"{transition.transition_id}:outcome",
            event_id=transition.transition_id,
            outcome_kind=EnvironmentEventKind.SCENE_EVENT,
            action_id=transition.transition_id,
            status=status,
            summary=f"digital-ant ecology transition {status}",
            detail=(
                "owner-authored pickup/delivery, contact, thermal threshold and "
                "bounded local food/path-integration-home/heat deltas; no "
                "coordinates, target "
                "direction or steering recommendation"
            ),
            prediction_id=prediction_id or None,
            evidence=(
                f"ant_transition:{transition.transition_id}",
                f"local_food_delta:{food_delta:.9f}",
                f"local_home_delta:{home_delta:.9f}",
                f"local_cooling_delta:{cooling_delta:.9f}",
            ),
            environment_state_delta_kind=status,
            measurement=measurement,
        )

    def _heading_stability_error(self, observation: WorldObservation) -> float:
        error = abs(
            (observation.eval_true_heading - self._heading_stability_target + math.pi) % (2.0 * math.pi) - math.pi
        )
        return error / math.pi

    @staticmethod
    def _homing_direction_error(observation: WorldObservation, nav_state: NavigatorState) -> float:
        estimated = nav_state.home_bearing
        true_bearing = observation.eval_home_bearing
        diff = (estimated - true_bearing + math.pi) % (2.0 * math.pi) - math.pi
        return abs(diff)

    async def run(self, ticks: int) -> list[AntStepRecord]:
        for _ in range(ticks):
            await self.step()
        return self.trajectory
