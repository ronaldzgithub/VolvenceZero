from __future__ import annotations

from dataclasses import replace
import math

import pytest

from volvence_ant.env import AntWorld, AntWorldConfig, MotorDistortionProfile
from volvence_ant.runtime import (
    AntObjectiveKind,
    AntSession,
    AntSessionConfig,
    AntStepRecord,
)
from volvence_zero.credit import CreditRecord, CreditSnapshot
from volvence_zero.environment import (
    EnvironmentEventKind,
    EnvironmentMeasurement,
    EnvironmentOutcome,
)
from volvence_zero.integration import FinalRolloutConfig
from volvence_zero.internal_rl import (
    InternalRLSandbox,
    RuntimeReplayLineageError,
    ZRollout,
    runtime_replay_policy_distribution,
)
from volvence_zero.joint_loop import ETANLJointLoop, JointLoopSchedule
from volvence_zero.memory import Track
from volvence_zero.prediction import (
    ActualOutcome,
    PredictedOutcome,
    PredictionActionContext,
    PredictionError,
    PredictionErrorSnapshot,
)
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate import (
    ResidualSequenceStep,
    SubstrateSnapshot,
    SurfaceKind,
    build_training_trace,
)
from volvence_zero.temporal import (
    FullLearnedTemporalPolicy,
    MetacontrollerParameterStore,
    TemporalAbstractionSnapshot,
    clone_full_learned_temporal_policy,
)


def _substrate(source_text: str) -> SubstrateSnapshot:
    trace = build_training_trace(
        trace_id=f"runtime-replay:{source_text}",
        source_text=source_text,
    )
    step = trace.steps[-1]
    return SubstrateSnapshot(
        model_id=trace.trace_id,
        is_frozen=True,
        surface_kind=SurfaceKind.RESIDUAL_STREAM,
        token_logits=(0.1, 0.2),
        feature_surface=step.feature_surface,
        residual_activations=step.residual_activations,
        residual_sequence=(
            ResidualSequenceStep(
                step=step.step,
                token=step.token,
                feature_surface=step.feature_surface,
                residual_activations=step.residual_activations,
                description=f"runtime replay token {step.token}",
            ),
        ),
        unavailable_fields=(),
        description=f"runtime replay substrate {source_text}",
    )


def _runtime_action(
    policy: FullLearnedTemporalPolicy,
    substrate: SubstrateSnapshot,
) -> tuple[TemporalAbstractionSnapshot, object]:
    step = policy.step(
        substrate_snapshot=substrate,
        previous_snapshot=None,
    )
    temporal = TemporalAbstractionSnapshot(
        controller_state=step.controller_state,
        active_abstract_action=step.active_abstract_action,
        controller_params_hash=step.controller_params_hash,
        description=step.description,
        action_family_version=step.action_family_version,
    )
    return temporal, policy.export_runtime_state()


def test_causal_action_head_wiring_is_reversible_and_shadow_is_noop() -> None:
    source_store = MetacontrollerParameterStore(n_z=4)
    source_head = source_store.causal_action_head_parameters(
        track=Track.WORLD
    )
    source_store.restore_causal_action_head_parameters(
        replace(
            source_head,
            bias=(0.35, -0.25, 0.2, -0.15),
            update_step=1,
        )
    )
    snapshot = source_store.export_parameter_snapshot()

    def configured_policy(
        wiring_level: WiringLevel,
    ) -> FullLearnedTemporalPolicy:
        store = MetacontrollerParameterStore(n_z=4)
        store.restore_parameter_snapshot(snapshot)
        policy = FullLearnedTemporalPolicy(parameter_store=store)
        policy.set_causal_action_head(
            wiring_level=wiring_level,
            track=Track.WORLD,
            strength=0.5,
        )
        return policy

    substrate = _substrate("paired causal action head observation")
    disabled = configured_policy(WiringLevel.DISABLED)
    shadow = configured_policy(WiringLevel.SHADOW)
    active = configured_policy(WiringLevel.ACTIVE)

    disabled_step = disabled.step(
        substrate_snapshot=substrate,
        previous_snapshot=None,
    )
    shadow_step = shadow.step(
        substrate_snapshot=substrate,
        previous_snapshot=None,
    )
    active_step = active.step(
        substrate_snapshot=substrate,
        previous_snapshot=None,
    )

    assert shadow_step.controller_state.code == (
        disabled_step.controller_state.code
    )
    assert any(
        abs(value) > 1e-9
        for value in shadow.export_runtime_state().causal_action_head_residual
    )
    assert active_step.controller_state.code != (
        disabled_step.controller_state.code
    )


def _outcome(*, prediction_id: str, outcome_id: str = "outcome-1") -> EnvironmentOutcome:
    return EnvironmentOutcome(
        outcome_id=outcome_id,
        event_id="event-1",
        outcome_kind=EnvironmentEventKind.SCENE_EVENT,
        action_id="action-1",
        status="observed",
        summary="observable runtime result",
        detail="environment fact",
        prediction_id=prediction_id,
        measurement=EnvironmentMeasurement(
            task_progress=0.7,
            action_payoff=0.2,
            terminal=False,
        ),
    )


def _prediction_error(
    *,
    prediction_id: str,
    outcome_id: str = "outcome-1",
    next_prediction_id: str = "prediction-next",
) -> PredictionErrorSnapshot:
    context = PredictionActionContext(
        segment_id="segment-1",
        abstract_action_id="runtime-action",
        z_t_digest=(0.1, 0.2),
        environment_outcome_id=outcome_id,
        prediction_id=prediction_id,
    )
    evaluated = PredictedOutcome(
        source_turn_index=1,
        target_turn_index=2,
        predicted_task_progress=0.5,
        predicted_relationship_delta=0.0,
        predicted_regime_stability=0.5,
        predicted_action_payoff=0.0,
        confidence=0.8,
        description="captured prediction",
        prediction_id=prediction_id,
    )
    return PredictionErrorSnapshot(
        evaluated_prediction=evaluated,
        actual_outcome=ActualOutcome(
            observed_turn_index=2,
            task_progress=0.7,
            relationship_delta=0.0,
            regime_stability=0.5,
            action_payoff=0.2,
            description="realized environment result",
            action_context=context,
        ),
        next_prediction=replace(
            evaluated,
            source_turn_index=2,
            target_turn_index=3,
            prediction_id=next_prediction_id,
        ),
        error=PredictionError(
            task_error=-0.2,
            relationship_error=0.0,
            regime_error=0.0,
            action_error=-0.2,
            magnitude=0.2,
            signed_reward=0.25,
            description="PE-derived reward",
        ),
        turn_index=2,
        bootstrap=False,
        description="matched runtime PE",
        action_context=context,
    )


def _credit() -> CreditSnapshot:
    return CreditSnapshot(
        recent_credits=(
            CreditRecord(
                record_id="credit-1",
                level="abstract_action_segment",
                track=Track.SHARED,
                source_event="segment:segment-1",
                credit_value=0.5,
                context="matched PE segment",
                timestamp_ms=2,
            ),
        ),
        recent_modifications=(),
        cumulative_credit_by_level=(("abstract_action_segment", 0.5),),
    )


def _sandbox_capture() -> tuple[
    InternalRLSandbox,
    MetacontrollerParameterStore,
    object,
]:
    store = MetacontrollerParameterStore(n_z=4)
    store.track_weights[Track.WORLD] = (0.4, 0.3, 0.2, 0.1)
    policy = FullLearnedTemporalPolicy(parameter_store=store)
    policy.set_runtime_track_modulation(0.3)
    sandbox = InternalRLSandbox(policy=policy)
    substrate = _substrate("initial heading observation")
    temporal, runtime_state = _runtime_action(policy, substrate)
    capture = sandbox.capture_runtime_action(
        turn_index=1,
        track=Track.WORLD,
        prediction_id="prediction-1",
        substrate_snapshot=substrate,
        temporal_snapshot=temporal,
        runtime_state=runtime_state,
    )
    return sandbox, store, capture


def test_runtime_replay_default_is_disabled() -> None:
    assert (
        FinalRolloutConfig().internal_rl_runtime_replay
        is WiringLevel.DISABLED
    )


async def test_disabled_runtime_replay_keeps_synthetic_rollout() -> None:
    loop = ETANLJointLoop()
    trace = build_training_trace(
        trace_id="disabled-synthetic",
        source_text="historical synthetic rollback lane",
    )

    report = await loop.run_cycle(cycle_index=1, trace=trace)

    assert report.backend_name != "waiting-for-runtime-replay"
    assert report.runtime_replay_report is not None
    assert report.runtime_replay_report.wiring_level == "disabled"
    assert report.runtime_replay_report.transition_source == "synthetic"


async def test_shadow_runtime_replay_collects_without_training_staging() -> None:
    session = AntSession(
        AntWorld(config=AntWorldConfig(seed=5)),
        config=AntSessionConfig(
            temporal_latent_dim=4,
            seed=5,
            rollout_config=FinalRolloutConfig(
                internal_rl_runtime_replay=WiringLevel.SHADOW,
                internal_rl_runtime_modulation_strength=0.3,
            ),
            joint_schedule=JointLoopSchedule(
                ssl_interval=0,
                rl_interval=1,
            ),
        ),
    )

    await session.run(3)
    report = session.runner._joint_loop.latest_runtime_replay_report

    assert report.wiring_level == "shadow"
    assert report.captured_count == 6
    assert report.settled_count == 4
    assert report.lineage_match_count == 4
    assert report.staged_rollout_count == 0


def test_runner_rejects_runtime_replay_wiring_mismatch() -> None:
    loop = ETANLJointLoop(
        internal_rl_runtime_replay=WiringLevel.ACTIVE,
    )

    with pytest.raises(ValueError, match="runtime replay wiring mismatch"):
        from volvence_zero.agent import AgentSessionRunner

        AgentSessionRunner(
            config=FinalRolloutConfig(
                internal_rl_runtime_replay=WiringLevel.DISABLED,
            ),
            joint_loop=loop,
        )


def test_ant_matched_arms_share_runtime_replay_with_true_no_optimize() -> None:
    from scripts.run_ant_matched_control import (
        _learned_config,
        _pe_off_config,
        _schedule_gated_arms,
    )

    learned = _learned_config(3, 4)
    no_optimize = _schedule_gated_arms(seed=3, n_z=4)["no_optimize"]
    pe_off = _pe_off_config(3, 4)

    assert learned.rollout_config == no_optimize.rollout_config
    assert learned.rollout_config == pe_off.rollout_config
    assert (
        learned.rollout_config.internal_rl_runtime_replay
        is WiringLevel.ACTIVE
    )
    assert learned.joint_apply_writeback is True
    assert no_optimize.joint_apply_writeback is True
    assert learned.joint_apply_policy_optimization is True
    assert no_optimize.joint_apply_policy_optimization is False
    assert pe_off.external_prediction_error_drive is False


def test_runtime_replay_settlement_contains_real_action_and_next_substrate_effect() -> None:
    from volvence_zero.internal_rl.sandbox import _surface_signature

    sandbox, _, capture = _sandbox_capture()
    next_substrate = _substrate("changed compass and motor observation")
    settlement = sandbox.settle_runtime_action(
        next_substrate_snapshot=next_substrate,
        environment_outcome=_outcome(prediction_id="prediction-1"),
        prediction_error_snapshot=_prediction_error(
            prediction_id="prediction-1"
        ),
        credit_snapshot=_credit(),
    )

    assert settlement.lineage_matched is True
    assert settlement.rollout is not None
    transition = settlement.rollout.transitions[0]
    expected_next = _surface_signature(next_substrate, 4)
    assert transition.transition_source == "runtime-replay"
    assert transition.policy_action == capture.temporal_snapshot.controller_state.code
    assert transition.observation_signature == capture.observation_signature
    assert transition.downstream_effect == pytest.approx(
        tuple(
            expected_next[index] - capture.observation_signature[index]
            for index in range(4)
        )
    )
    assert transition.reward == pytest.approx(0.31)
    assert transition.lineage_matched is True


def test_causal_action_head_optimizes_from_runtime_replay_and_rolls_back() -> None:
    store = MetacontrollerParameterStore(n_z=4)
    store.track_weights[Track.WORLD] = (0.4, 0.3, 0.2, 0.1)
    policy = FullLearnedTemporalPolicy(parameter_store=store)
    policy.set_runtime_track_modulation(0.3)
    policy.set_causal_action_head(
        wiring_level=WiringLevel.ACTIVE,
        track=Track.WORLD,
        strength=0.35,
    )
    sandbox = InternalRLSandbox(policy=policy)
    substrate = _substrate("causal action head replay observation")
    temporal, runtime_state = _runtime_action(policy, substrate)
    sandbox.capture_runtime_action(
        turn_index=1,
        track=Track.WORLD,
        prediction_id="prediction-1",
        substrate_snapshot=substrate,
        temporal_snapshot=temporal,
        runtime_state=runtime_state,
    )
    settlement = sandbox.settle_runtime_action(
        next_substrate_snapshot=_substrate(
            "causal action head changed observation"
        ),
        environment_outcome=_outcome(prediction_id="prediction-1"),
        prediction_error_snapshot=_prediction_error(
            prediction_id="prediction-1"
        ),
        credit_snapshot=_credit(),
    )
    assert settlement.rollout is not None
    checkpoint = sandbox.create_checkpoint(
        checkpoint_id="causal-action-head-before"
    )
    before = store.causal_action_head_parameters(track=Track.WORLD)

    report = sandbox.optimize(settlement.rollout)
    after = store.causal_action_head_parameters(track=Track.WORLD)

    assert report.parameter_change_norm > 0.0
    assert after != before
    assert after.update_step > before.update_step

    sandbox.restore_checkpoint(checkpoint)

    assert store.causal_action_head_parameters(track=Track.WORLD) == before


def test_runtime_replay_lineage_mismatch_fails_loudly() -> None:
    sandbox, _, _ = _sandbox_capture()

    with pytest.raises(RuntimeReplayLineageError, match="lineage mismatch"):
        sandbox.settle_runtime_action(
            next_substrate_snapshot=_substrate("next observation"),
            environment_outcome=_outcome(prediction_id="wrong-prediction"),
            prediction_error_snapshot=_prediction_error(
                prediction_id="prediction-1"
            ),
            credit_snapshot=_credit(),
        )


def test_runtime_replay_checkpoint_round_trips_pending_capture() -> None:
    sandbox, _, capture = _sandbox_capture()
    checkpoint = sandbox.create_checkpoint(
        checkpoint_id="runtime-replay",
        include_runtime_replay=True,
    )
    sandbox.settle_runtime_action(
        next_substrate_snapshot=_substrate("dropped observation"),
        environment_outcome=None,
        prediction_error_snapshot=None,
        credit_snapshot=None,
    )

    sandbox.restore_checkpoint(checkpoint)

    assert sandbox.runtime_replay_checkpoint.pending_capture == capture
    assert sandbox.runtime_replay_checkpoint.dropped_count == 0


def test_joint_checkpoint_round_trips_staged_and_pending_runtime_replay() -> None:
    world_policy = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=4)
    )
    self_policy = clone_full_learned_temporal_policy(world_policy)
    world_policy.set_runtime_track_modulation(0.3)
    self_policy.set_runtime_track_modulation(0.3)
    loop = ETANLJointLoop(
        world_policy=world_policy,
        self_policy=self_policy,
        internal_rl_runtime_replay=WiringLevel.ACTIVE,
    )
    first_substrate = _substrate("checkpoint first observation")
    first_world, first_world_state = _runtime_action(
        world_policy,
        first_substrate,
    )
    first_self, first_self_state = _runtime_action(
        self_policy,
        first_substrate,
    )
    loop.observe_runtime_transition(
        turn_index=1,
        substrate_snapshot=first_substrate,
        world_temporal_snapshot=first_world,
        self_temporal_snapshot=first_self,
        world_runtime_state=first_world_state,
        self_runtime_state=first_self_state,
        environment_outcome=_outcome(prediction_id="bootstrap"),
        prediction_error_snapshot=_prediction_error(
            prediction_id="bootstrap",
            next_prediction_id="prediction-1",
        ),
        credit_snapshot=_credit(),
    )

    second_substrate = _substrate("checkpoint second observation")
    second_world, second_world_state = _runtime_action(
        world_policy,
        second_substrate,
    )
    second_self, second_self_state = _runtime_action(
        self_policy,
        second_substrate,
    )
    loop.observe_runtime_transition(
        turn_index=2,
        substrate_snapshot=second_substrate,
        world_temporal_snapshot=second_world,
        self_temporal_snapshot=second_self,
        world_runtime_state=second_world_state,
        self_runtime_state=second_self_state,
        environment_outcome=_outcome(prediction_id="prediction-1"),
        prediction_error_snapshot=_prediction_error(
            prediction_id="prediction-1",
            next_prediction_id="prediction-2",
        ),
        credit_snapshot=_credit(),
    )
    checkpoint = loop.create_learning_checkpoint(
        checkpoint_id="joint-runtime-replay"
    )

    assert checkpoint.pending_task_rollouts is not None
    assert len(checkpoint.pending_task_rollouts) == 1
    assert checkpoint.pending_relationship_rollouts is not None
    assert len(checkpoint.pending_relationship_rollouts) == 1
    assert checkpoint.world_policy_checkpoint.runtime_replay is not None
    assert (
        checkpoint.world_policy_checkpoint.runtime_replay.pending_capture
        is not None
    )
    assert checkpoint.runtime_replay_report is not None
    assert checkpoint.runtime_replay_report.transition_count == 2

    loop.restore_learning_checkpoint(checkpoint)

    restored = loop.latest_runtime_replay_report
    assert restored.transition_count == 2
    assert restored.staged_rollout_count == 2


def test_runtime_replay_aggregates_real_transitions_by_beta_segment() -> None:
    sandbox, _, _ = _sandbox_capture()
    settlement = sandbox.settle_runtime_action(
        next_substrate_snapshot=_substrate("segment next observation"),
        environment_outcome=_outcome(prediction_id="prediction-1"),
        prediction_error_snapshot=_prediction_error(
            prediction_id="prediction-1"
        ),
        credit_snapshot=_credit(),
    )
    assert settlement.rollout is not None
    base_transition = settlement.rollout.transitions[0]
    world_policy = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=4)
    )
    loop = ETANLJointLoop(
        world_policy=world_policy,
        self_policy=clone_full_learned_temporal_policy(world_policy),
        internal_rl_runtime_replay=WiringLevel.ACTIVE,
        runtime_replay_segment_credit=WiringLevel.ACTIVE,
        runtime_replay_segment_max_steps=8,
    )

    def paired_rollouts(
        *,
        turn_index: int,
        switching: bool,
        milestone: bool = False,
    ) -> tuple[ZRollout, ZRollout]:
        controller = replace(
            base_transition.controller_state,
            is_switching=switching,
        )
        world_transition = replace(
            base_transition,
            track=Track.WORLD,
            controller_state=controller,
            runtime_turn_index=turn_index,
            runtime_milestone=milestone,
        )
        self_transition = replace(
            world_transition,
            track=Track.SELF,
        )
        return (
            replace(
                settlement.rollout,
                rollout_id=f"world-{turn_index}",
                track=Track.WORLD,
                transitions=(world_transition,),
            ),
            replace(
                settlement.rollout,
                rollout_id=f"self-{turn_index}",
                track=Track.SELF,
                transitions=(self_transition,),
            ),
        )

    for turn_index in (1, 2):
        world_rollout, self_rollout = paired_rollouts(
            turn_index=turn_index,
            switching=False,
        )
        loop._stage_runtime_segment_pair(
            world_rollout=world_rollout,
            self_rollout=self_rollout,
        )
    assert loop.latest_runtime_replay_report.open_segment_transition_count == 2
    assert loop.latest_runtime_replay_report.closed_segment_count == 0

    world_rollout, self_rollout = paired_rollouts(
        turn_index=3,
        switching=True,
    )
    loop._stage_runtime_segment_pair(
        world_rollout=world_rollout,
        self_rollout=self_rollout,
    )
    assert len(loop._pending_task_rollouts) == 1
    assert tuple(
        item.step_index
        for item in loop._pending_task_rollouts[0].transitions
    ) == (0, 1)
    assert loop.latest_runtime_replay_report.open_segment_transition_count == 1

    world_rollout, self_rollout = paired_rollouts(
        turn_index=4,
        switching=False,
        milestone=True,
    )
    loop._stage_runtime_segment_pair(
        world_rollout=world_rollout,
        self_rollout=self_rollout,
    )
    report = loop.latest_runtime_replay_report
    assert report.open_segment_transition_count == 0
    assert report.closed_segment_count == 2
    assert report.longest_segment_length == 2


def test_joint_transfer_checkpoint_omits_episode_local_runtime_replay() -> None:
    world_policy = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=4)
    )
    self_policy = clone_full_learned_temporal_policy(world_policy)
    loop = ETANLJointLoop(
        world_policy=world_policy,
        self_policy=self_policy,
        internal_rl_runtime_replay=WiringLevel.ACTIVE,
    )
    substrate = _substrate("transfer checkpoint observation")
    world_temporal, world_state = _runtime_action(world_policy, substrate)
    self_temporal, self_state = _runtime_action(self_policy, substrate)
    loop.observe_runtime_transition(
        turn_index=1,
        substrate_snapshot=substrate,
        world_temporal_snapshot=world_temporal,
        self_temporal_snapshot=self_temporal,
        world_runtime_state=world_state,
        self_runtime_state=self_state,
        environment_outcome=_outcome(prediction_id="bootstrap"),
        prediction_error_snapshot=_prediction_error(
            prediction_id="bootstrap",
            next_prediction_id="prediction-1",
        ),
        credit_snapshot=_credit(),
    )

    checkpoint = loop.create_learning_checkpoint(
        checkpoint_id="cross-episode-transfer",
        include_runtime_replay=False,
    )

    assert checkpoint.pending_task_rollouts == ()
    assert checkpoint.pending_relationship_rollouts == ()
    assert checkpoint.runtime_replay_report is None
    assert checkpoint.world_policy_checkpoint.runtime_replay is None
    assert checkpoint.self_policy_checkpoint.runtime_replay is None

    operations = loop.restore_learning_checkpoint(checkpoint)
    restored = loop.latest_runtime_replay_report
    assert "runtime-replay:episode-transfer-reset" in operations
    assert restored.captured_count == 0
    assert restored.settled_count == 0
    assert restored.transition_count == 0
    assert restored.pending_capture_count == 0
    assert restored.staged_rollout_count == 0
    assert restored.drop_reasons == ()


def test_runtime_replay_distribution_matches_captured_likelihood() -> None:
    sandbox, store, capture = _sandbox_capture()
    track_weights = store.track_weights[Track.WORLD]
    mean, std = runtime_replay_policy_distribution(
        base_mean=capture.runtime_base_mean,
        base_std=capture.runtime_base_std,
        previous_code=capture.previous_code,
        beta_t=capture.runtime_beta_t,
        track_weights=track_weights,
        other_track_sum=capture.runtime_other_track_sum,
        modulation_strength=0.3,
    )
    expected_log_prob = sum(
        -0.5
        * (
            ((action - mean_value) ** 2) / max(std_value**2, 1e-6)
            + math.log(2.0 * math.pi * max(std_value**2, 1e-6))
        )
        for action, mean_value, std_value in zip(
            capture.policy_action,
            mean,
            std,
            strict=True,
        )
    )
    n_z = len(capture.policy_action)
    gains = tuple(
        max(
            0.5,
            min(
                1.5,
                1.0
                + 0.3
                * (
                    (
                        track_weights[index]
                        + capture.runtime_other_track_sum[index]
                    )
                    / 3.0
                    * n_z
                    - 1.0
                ),
            ),
        )
        for index in range(n_z)
    )
    expected_sampled_candidate = tuple(
        max(
            0.0,
            min(
                1.0,
                (
                    max(
                        0.0,
                        min(
                            1.0,
                            capture.runtime_base_mean[index]
                            + capture.runtime_base_std[index]
                            * capture.policy_noise[index]
                            * 0.5,
                        ),
                    )
                )
                * gains[index],
            ),
        )
        for index in range(n_z)
    )
    expected_action = tuple(
        capture.runtime_beta_t[index] * expected_sampled_candidate[index]
        + (1.0 - capture.runtime_beta_t[index])
        * capture.previous_code[index]
        for index in range(n_z)
    )

    assert mean == pytest.approx(capture.policy_mean)
    assert std == pytest.approx(capture.policy_std)
    assert capture.runtime_state.z_tilde == pytest.approx(
        expected_sampled_candidate
    )
    assert capture.policy_action == pytest.approx(expected_action)
    assert expected_log_prob == pytest.approx(capture.log_prob)
    assert sandbox.runtime_replay_checkpoint.pending_capture == capture


def test_torch_runtime_likelihood_matches_pure_capture() -> None:
    pytest.importorskip("torch")
    from volvence_zero.internal_rl.torch_causal_ppo import (
        torch_causal_ppo_update,
    )

    sandbox, store, _ = _sandbox_capture()
    settlement = sandbox.settle_runtime_action(
        next_substrate_snapshot=_substrate("torch next observation"),
        environment_outcome=_outcome(prediction_id="prediction-1"),
        prediction_error_snapshot=_prediction_error(
            prediction_id="prediction-1"
        ),
        credit_snapshot=_credit(),
    )
    assert settlement.rollout is not None
    transition = replace(
        settlement.rollout.transitions[0],
        advantage_estimate=0.25,
        return_estimate=0.25,
    )
    report = torch_causal_ppo_update(
        parameter_store=store,
        value_weights=sandbox.causal_policy._value_weights,
        value_bias=sandbox.causal_policy._value_bias,
        track=Track.WORLD,
        transitions=(transition,),
        n_z=4,
        write_back=False,
        ppo_epochs=1,
        learning_rate=0.0,
        runtime_track_modulation_strength=0.3,
    )

    assert report.approx_kl == pytest.approx(0.0, abs=1e-10)


async def test_active_runtime_replay_never_calls_synthetic_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = FullLearnedTemporalPolicy(
        parameter_store=MetacontrollerParameterStore(n_z=4)
    )
    policy.set_runtime_track_modulation(0.3)
    loop = ETANLJointLoop(
        world_policy=policy,
        self_policy=clone_full_learned_temporal_policy(policy),
        internal_rl_runtime_replay=WiringLevel.ACTIVE,
    )

    def forbidden_rollout(**_kwargs: object) -> object:
        raise AssertionError("ACTIVE runtime replay called synthetic rollout")

    monkeypatch.setattr(loop._world_sandbox, "rollout", forbidden_rollout)
    monkeypatch.setattr(loop._self_sandbox, "rollout", forbidden_rollout)
    trace = build_training_trace(
        trace_id="active-no-fallback",
        source_text="real runtime only",
    )

    report = await loop.run_cycle(cycle_index=1, trace=trace)

    assert report.backend_name == "waiting-for-runtime-replay"
    assert (
        report.optimization_summary
        == "task_adv=0.000, rel_adv=0.000"
    )
    assert report.runtime_replay_report is not None
    assert report.runtime_replay_report.transition_source == "runtime-replay"


async def test_no_optimize_does_not_persist_runtime_replay_policy_update() -> None:
    async def run_arm(
        apply_policy_optimization: bool,
    ) -> tuple[tuple[float, ...], tuple[float, ...], AntStepRecord]:
        session = AntSession(
            AntWorld(
                config=AntWorldConfig(
                    seed=7,
                    motor_distortions=(
                        MotorDistortionProfile(turn_bias=0.18),
                    ),
                )
            ),
            config=AntSessionConfig(
                temporal_latent_dim=4,
                seed=7,
                objective=AntObjectiveKind.HEADING_STABILITY,
                rollout_config=FinalRolloutConfig(
                    internal_rl_runtime_replay=WiringLevel.ACTIVE,
                    internal_rl_runtime_modulation_strength=0.3,
                ),
                joint_schedule=JointLoopSchedule(
                    ssl_interval=0,
                    rl_interval=1,
                ),
                joint_apply_policy_optimization=apply_policy_optimization,
            ),
        )
        store = (
            session.runner._joint_loop.world_temporal_policy.parameter_store
        )
        before = tuple(store.track_weights[Track.WORLD])
        records = await session.run(5)
        after = tuple(store.track_weights[Track.WORLD])
        return before, after, records[-1]

    learned_before, learned_after, learned_record = await run_arm(True)
    no_opt_before, no_opt_after, no_opt_record = await run_arm(False)

    assert learned_after != learned_before
    assert no_opt_after == no_opt_before
    assert learned_after != no_opt_after
    assert learned_record.runtime_replay_transitions > 0
    assert (
        learned_record.runtime_replay_lineage_matches
        == learned_record.runtime_replay_transitions
    )
    assert no_opt_record.runtime_replay_transitions > 0
