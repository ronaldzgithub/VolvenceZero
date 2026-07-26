"""Typed reward eligibility + PE-drive/outcome-payoff split for runtime replay.

Covers the two reward-seam defects closed by convergence package W3-a
(docs/specs/prediction-error-loop.md):

* the optimizer silently consumed a substrate-derived realized payoff on ticks
  where the environment published nothing;
* ``external_prediction_error_drive=False`` ("PE-off") also removed the
  environment-published payoff, so the matched arm trained on an
  identically-zero reward stream.

Every test that asserts a fixed number also asserts the rollback: the default
contract must reproduce the historical value exactly.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from volvence_zero.credit import CreditRecord, CreditSnapshot
from volvence_zero.environment import (
    EnvironmentEventKind,
    EnvironmentMeasurement,
    EnvironmentOutcome,
)
from volvence_zero.internal_rl.sandbox import (
    InternalRLSandbox,
    RuntimeReplayLatentBoundContractError,
    RuntimeReplayRewardEligibility,
    RuntimeReplayRewardEligibilityError,
    RuntimeReplayRewardEligibilityReason,
    runtime_replay_policy_distribution,
)
from volvence_zero.joint_loop.runtime import ETANLJointLoop
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


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _substrate(source_text: str) -> SubstrateSnapshot:
    trace = build_training_trace(
        trace_id=f"reward-eligibility:{source_text}",
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
                description=f"reward eligibility token {step.token}",
            ),
        ),
        unavailable_fields=(),
        description=f"reward eligibility substrate {source_text}",
    )


def _temporal_and_runtime_state(
    policy: FullLearnedTemporalPolicy,
    substrate: SubstrateSnapshot,
) -> tuple[TemporalAbstractionSnapshot, object]:
    step = policy.step(substrate_snapshot=substrate, previous_snapshot=None)
    temporal = TemporalAbstractionSnapshot(
        controller_state=step.controller_state,
        active_abstract_action=step.active_abstract_action,
        controller_params_hash=step.controller_params_hash,
        description=step.description,
        action_family_version=step.action_family_version,
    )
    return temporal, policy.export_runtime_state()


def _outcome(
    *,
    prediction_id: str = "prediction-1",
    outcome_id: str = "outcome-1",
    measurement: EnvironmentMeasurement | None = None,
) -> EnvironmentOutcome:
    """An environment outcome; ``measurement=None`` models a silent tick."""

    return EnvironmentOutcome(
        outcome_id=outcome_id,
        event_id="event-1",
        outcome_kind=EnvironmentEventKind.SCENE_EVENT,
        action_id="action-1",
        status="observed",
        summary="observable runtime result",
        detail="environment fact",
        prediction_id=prediction_id,
        measurement=measurement,
    )


def _prediction_error(
    *,
    prediction_id: str = "prediction-1",
    outcome_id: str = "outcome-1",
    next_prediction_id: str = "prediction-next",
    action_payoff: float = 0.2,
    signed_reward: float = 0.25,
    magnitude: float = 0.2,
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
            # On a measurement-free tick the PE owner never overrides this
            # axis, so it stays the internally synthesized blend of substrate
            # feature signals and evaluation-family signals. That is exactly
            # the quantity strict eligibility must refuse.
            action_payoff=action_payoff,
            description="realized outcome",
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
            magnitude=magnitude,
            signed_reward=signed_reward,
            description="PE residual",
        ),
        turn_index=2,
        bootstrap=False,
        description="matched runtime PE",
        action_context=context,
    )


def _credit(credit_value: float = 0.5) -> CreditSnapshot:
    return CreditSnapshot(
        recent_credits=(
            CreditRecord(
                record_id="credit-1",
                level="abstract_action_segment",
                track=Track.SHARED,
                source_event="segment:segment-1",
                credit_value=credit_value,
                context="matched PE segment",
                timestamp_ms=2,
            ),
        ),
        recent_modifications=(),
        cumulative_credit_by_level=(("abstract_action_segment", 0.5),),
    )


def _captured_sandbox() -> InternalRLSandbox:
    store = MetacontrollerParameterStore(n_z=4)
    store.track_weights[Track.WORLD] = (0.4, 0.3, 0.2, 0.1)
    policy = FullLearnedTemporalPolicy(parameter_store=store)
    policy.set_runtime_track_modulation(0.3)
    sandbox = InternalRLSandbox(policy=policy)
    substrate = _substrate("initial heading observation")
    temporal, runtime_state = _temporal_and_runtime_state(policy, substrate)
    sandbox.capture_runtime_action(
        turn_index=1,
        track=Track.WORLD,
        prediction_id="prediction-1",
        substrate_snapshot=substrate,
        temporal_snapshot=temporal,
        runtime_state=runtime_state,
    )
    return sandbox


def _settle(sandbox: InternalRLSandbox, **kwargs):
    return sandbox.settle_runtime_action(
        next_substrate_snapshot=_substrate("next observation"),
        environment_outcome=kwargs.pop("environment_outcome", _outcome()),
        prediction_error_snapshot=kwargs.pop(
            "prediction_error_snapshot", _prediction_error()
        ),
        credit_snapshot=kwargs.pop("credit_snapshot", _credit()),
        **kwargs,
    )


_MEASURED = EnvironmentMeasurement(
    task_progress=0.7,
    action_payoff=0.2,
    terminal=False,
)
_PROGRESS_ONLY = EnvironmentMeasurement(
    task_progress=0.7,
    action_payoff=None,
    terminal=False,
)


# --------------------------------------------------------------------------
# defect 1: measurement-free ticks must not smuggle a substrate reward
# --------------------------------------------------------------------------


def test_default_eligibility_reproduces_todays_measurement_free_value() -> None:
    """Rollback proof: the default contract keeps the historical numbers.

    Measurement is ``None``, so the PE owner's synthesized action axis (0.42)
    is what the optimizer historically consumed, plus the PE-derived segment
    bonus 0.5 * 0.1 * (1 + 0.2) = 0.06.
    """

    settlement = _settle(
        _captured_sandbox(),
        environment_outcome=_outcome(measurement=None),
        prediction_error_snapshot=_prediction_error(action_payoff=0.42),
    )

    assert settlement.rollout is not None
    transition = settlement.rollout.transitions[0]
    assert transition.raw_reward == pytest.approx(0.42)
    assert transition.reward == pytest.approx(0.48)
    assert settlement.realized_action_payoff == pytest.approx(0.42)
    assert settlement.segment_bonus == pytest.approx(0.06)
    assert settlement.reward == pytest.approx(0.48)
    assert settlement.reward_eligible is True
    assert (
        settlement.reward_eligibility
        == RuntimeReplayRewardEligibility.ANY_SETTLED_OUTCOME.value
    )
    assert (
        settlement.reward_eligibility_reason
        == RuntimeReplayRewardEligibilityReason.ELIGIBLE.value
    )


def test_strict_eligibility_zeroes_measurement_free_reward_and_tags_it() -> None:
    settlement = _settle(
        _captured_sandbox(),
        environment_outcome=_outcome(measurement=None),
        prediction_error_snapshot=_prediction_error(action_payoff=0.42),
        reward_eligibility=(
            RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY
        ),
    )

    assert settlement.rollout is not None
    transition = settlement.rollout.transitions[0]
    # Realized payoff AND the segment bonus are zero: the bonus must not
    # smuggle the same quantity back in through PE-derived credit.
    assert settlement.realized_action_payoff == 0.0
    assert settlement.segment_bonus == 0.0
    assert settlement.reward == 0.0
    assert transition.reward == 0.0
    assert transition.raw_reward == 0.0
    assert "abstract_action_credit" not in dict(transition.reward_components)
    # The transition still settles with full lineage and dynamics.
    assert settlement.lineage_matched is True
    assert transition.lineage_matched is True
    assert transition.transition_source == "runtime-replay"
    # ...and it is tagged so an auditor can count eligible vs ineligible.
    assert settlement.reward_eligible is False
    assert transition.runtime_reward_eligible is False
    assert (
        transition.runtime_reward_eligibility_reason
        == RuntimeReplayRewardEligibilityReason.NO_ENVIRONMENT_MEASUREMENT.value
    )


def test_strict_eligibility_keeps_environment_measured_reward() -> None:
    settlement = _settle(
        _captured_sandbox(),
        environment_outcome=_outcome(measurement=_MEASURED),
        prediction_error_snapshot=_prediction_error(action_payoff=0.2),
        reward_eligibility=(
            RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY
        ),
    )

    assert settlement.reward_eligible is True
    assert settlement.realized_action_payoff == pytest.approx(0.2)
    assert settlement.segment_bonus == pytest.approx(0.06)
    assert settlement.reward == pytest.approx(0.26)
    assert (
        settlement.reward_eligibility_reason
        == RuntimeReplayRewardEligibilityReason.ELIGIBLE.value
    )


def test_strict_eligibility_rejects_measurement_without_action_payoff() -> None:
    """A milestone that publishes progress but no payoff earns no reward."""

    settlement = _settle(
        _captured_sandbox(),
        environment_outcome=_outcome(measurement=_PROGRESS_ONLY),
        prediction_error_snapshot=_prediction_error(action_payoff=0.42),
        reward_eligibility=(
            RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY
        ),
    )

    assert settlement.reward_eligible is False
    assert settlement.reward == 0.0
    assert settlement.reward_eligibility_reason == (
        RuntimeReplayRewardEligibilityReason
        .NO_ENVIRONMENT_ACTION_PAYOFF.value
    )
    assert settlement.rollout is not None
    # The milestone flag is unchanged: eligibility gates reward, not lineage.
    assert settlement.rollout.transitions[0].runtime_milestone is True


def test_fully_blocked_move_under_strict_eligibility_yields_zero_reward() -> None:
    """The neutral-contact case from docs/specs/digital-ant-embodiment.md §4.

    A tick that by contract carries no payoff must not still be strictly
    positive because the substrate axis happens to be high.
    """

    sandbox = _captured_sandbox()
    default = _settle(
        sandbox,
        environment_outcome=_outcome(measurement=None),
        prediction_error_snapshot=_prediction_error(
            action_payoff=0.9, signed_reward=0.0
        ),
    )
    assert default.reward > 0.0

    strict = _settle(
        _captured_sandbox(),
        environment_outcome=_outcome(measurement=None),
        prediction_error_snapshot=_prediction_error(
            action_payoff=0.9, signed_reward=0.0
        ),
        reward_eligibility=(
            RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY
        ),
    )
    assert strict.reward == 0.0


def test_pe_residual_stays_a_diagnostic_and_never_becomes_reward() -> None:
    settlement = _settle(
        _captured_sandbox(),
        environment_outcome=_outcome(measurement=None),
        prediction_error_snapshot=_prediction_error(
            action_payoff=0.42, signed_reward=-0.7
        ),
        reward_eligibility=(
            RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY
        ),
    )

    assert settlement.rollout is not None
    components = dict(settlement.rollout.transitions[0].reward_components)
    assert components["prediction_error_residual"] == pytest.approx(-0.7)
    assert components["realized_action_payoff"] == 0.0
    assert settlement.reward == 0.0


# --------------------------------------------------------------------------
# defect 2: PE-off must not mean reward-off
# --------------------------------------------------------------------------


def test_pe_off_under_default_eligibility_is_the_exact_historical_rollback() -> None:
    """A domain that does not declare the new contract keeps today's zeros."""

    settlement = _settle(
        _captured_sandbox(),
        environment_outcome=_outcome(measurement=_MEASURED),
        prediction_error_reward_enabled=False,
    )

    assert settlement.realized_action_payoff == 0.0
    assert settlement.segment_bonus == 0.0
    assert settlement.reward == 0.0


def test_pe_off_keeps_environment_payoff_under_strict_eligibility() -> None:
    """PE-off removes PE-derived credit; the environment payoff survives."""

    pe_on = _settle(
        _captured_sandbox(),
        environment_outcome=_outcome(measurement=_MEASURED),
        reward_eligibility=(
            RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY
        ),
    )
    pe_off = _settle(
        _captured_sandbox(),
        environment_outcome=_outcome(measurement=_MEASURED),
        prediction_error_reward_enabled=False,
        reward_eligibility=(
            RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY
        ),
    )

    assert pe_on.segment_bonus == pytest.approx(0.06)
    assert pe_off.segment_bonus == 0.0
    assert pe_off.realized_action_payoff == pytest.approx(
        pe_on.realized_action_payoff
    )
    assert pe_off.realized_action_payoff == pytest.approx(0.2)
    assert pe_off.reward == pytest.approx(0.2)


def test_explicit_outcome_payoff_contract_overrides_the_derivation() -> None:
    forced_on = _settle(
        _captured_sandbox(),
        environment_outcome=_outcome(measurement=_MEASURED),
        prediction_error_reward_enabled=False,
        outcome_payoff_reward_enabled=True,
    )
    forced_off = _settle(
        _captured_sandbox(),
        environment_outcome=_outcome(measurement=_MEASURED),
        outcome_payoff_reward_enabled=False,
        reward_eligibility=(
            RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY
        ),
    )

    # (b) on while (a) is off, under the DEFAULT eligibility contract.
    assert forced_on.realized_action_payoff == pytest.approx(0.2)
    assert forced_on.segment_bonus == 0.0
    # (b) off while (a) is on, under STRICT eligibility.
    assert forced_off.realized_action_payoff == 0.0
    assert forced_off.segment_bonus == pytest.approx(0.06)


def test_joint_loop_pe_off_switch_pressure_is_independent_of_payoff() -> None:
    """(a) still owns the temporal switch; (b) no longer rides on it."""

    def loop_with(pe_drive: bool) -> ETANLJointLoop:
        return ETANLJointLoop(
            prediction_error_temporal_switch=WiringLevel.ACTIVE,
            runtime_replay_prediction_error_enabled=pe_drive,
            runtime_replay_reward_eligibility=(
                RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY
            ),
        )

    pe_on = loop_with(True)
    pe_off = loop_with(False)
    signals = {"prediction_error_magnitude": 1.5}
    pe_on.set_external_learning_signals(signals)
    pe_off.set_external_learning_signals({})

    on_pressure = (
        pe_on.world_temporal_policy.parameter_store
        .prediction_error_switch_pressure_delta()
    )
    off_pressure = (
        pe_off.world_temporal_policy.parameter_store
        .prediction_error_switch_pressure_delta()
    )
    assert on_pressure > 0.0
    assert off_pressure == 0.0
    # ...while both arms still declare the environment payoff reachable.
    assert (
        pe_off.runtime_replay_reward_eligibility
        is RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY
    )


# --------------------------------------------------------------------------
# published reward stream
# --------------------------------------------------------------------------


def _replay_loop(
    *,
    eligibility: RuntimeReplayRewardEligibility,
    pe_drive: bool = True,
) -> ETANLJointLoop:
    world_store = MetacontrollerParameterStore(n_z=4)
    world_store.track_weights[Track.WORLD] = (0.4, 0.3, 0.2, 0.1)
    world_policy = FullLearnedTemporalPolicy(parameter_store=world_store)
    world_policy.set_runtime_track_modulation(0.3)
    self_policy = clone_full_learned_temporal_policy(world_policy)
    return ETANLJointLoop(
        world_policy=world_policy,
        self_policy=self_policy,
        internal_rl_runtime_replay=WiringLevel.ACTIVE,
        runtime_replay_prediction_error_enabled=pe_drive,
        runtime_replay_reward_eligibility=eligibility,
    )


def _drive_turns(
    loop: ETANLJointLoop,
    *,
    measurements: tuple[EnvironmentMeasurement | None, ...],
) -> None:
    """Drive the owner entry point over a real prediction-id chain.

    The owner captures turn ``t`` under ``next_prediction.prediction_id`` and
    settles it on turn ``t + 1``, so each turn's PE snapshot must evaluate the
    previous turn's prediction id.

    The PE snapshot reproduces the production wiring: ``final_wiring`` forwards
    ``measurement.action_payoff`` into ``PredictionActionContext`` and
    ``prediction/error.py`` overrides its synthesized action axis with it, so a
    measured tick publishes the environment's number and a silent tick keeps
    the synthesized 0.42. The strict lane refuses to settle a fixture where
    those two disagree (see
    ``test_strict_eligibility_raises_when_the_paid_axis_is_not_the_gated_one``).
    """

    world_policy = loop.world_temporal_policy
    self_policy = loop.self_temporal_policy
    for turn, measurement in enumerate(measurements, start=1):
        substrate = _substrate(f"replay turn {turn}")
        world_temporal, world_state = _temporal_and_runtime_state(
            world_policy, substrate
        )
        self_temporal, self_state = _temporal_and_runtime_state(
            self_policy, substrate
        )
        settled_prediction_id = f"prediction-{turn - 1}"
        outcome_id = f"outcome-{turn}"
        loop.observe_runtime_transition(
            turn_index=turn,
            substrate_snapshot=substrate,
            world_temporal_snapshot=world_temporal,
            self_temporal_snapshot=self_temporal,
            world_runtime_state=world_state,
            self_runtime_state=self_state,
            environment_outcome=_outcome(
                prediction_id=settled_prediction_id,
                outcome_id=outcome_id,
                measurement=measurement,
            ),
            prediction_error_snapshot=_prediction_error(
                prediction_id=settled_prediction_id,
                outcome_id=outcome_id,
                next_prediction_id=f"prediction-{turn}",
                action_payoff=(
                    0.42
                    if measurement is None or measurement.action_payoff is None
                    else measurement.action_payoff
                ),
            ),
            credit_snapshot=_credit(),
        )


def _staged_batch_transitions(loop: ETANLJointLoop) -> tuple:
    """The transitions the optimizer will actually consume.

    Reaches the owner's staging buffers directly: the joint loop publishes
    only counts today, and this test's whole point is that the published
    stream equals the batch.
    """

    return tuple(
        transition
        for rollout in (
            *loop._pending_task_rollouts,
            *loop._pending_relationship_rollouts,
        )
        for transition in rollout.transitions
    )


def test_published_reward_stream_matches_the_staged_batch() -> None:
    loop = _replay_loop(
        eligibility=RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY
    )
    _drive_turns(loop, measurements=(None, None, _MEASURED))

    stream = loop.latest_runtime_replay_reward_stream
    batch_transitions = _staged_batch_transitions(loop)

    assert batch_transitions, "runtime replay must stage a real batch"
    assert stream.settled_transition_count == len(batch_transitions)
    assert stream.reward_sum == pytest.approx(
        sum(transition.reward for transition in batch_transitions)
    )
    assert stream.realized_action_payoff_sum == pytest.approx(
        sum(transition.raw_reward for transition in batch_transitions)
    )
    assert stream.eligible_transition_count == sum(
        int(transition.runtime_reward_eligible)
        for transition in batch_transitions
    )
    assert stream.nonzero_reward_transition_count == sum(
        int(abs(transition.reward) > 1e-12)
        for transition in batch_transitions
    )
    # Two settlements per settled turn (world + self); turns 2 and 3 settle.
    assert stream.settled_transition_count == 4
    assert stream.eligible_transition_count == 2
    assert stream.ineligible_transition_count == 2
    assert dict(stream.eligibility_reason_counts) == {
        RuntimeReplayRewardEligibilityReason.ELIGIBLE.value: 2,
        (
            RuntimeReplayRewardEligibilityReason
            .NO_ENVIRONMENT_MEASUREMENT.value
        ): 2,
    }
    assert stream.nonzero_reward_transition_count == 2
    assert (
        stream.eligibility_contract
        == RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY.value
    )
    assert stream.outcome_payoff_reward == "derived-from-eligibility"


def test_published_reward_stream_default_contract_counts_everything() -> None:
    loop = _replay_loop(
        eligibility=RuntimeReplayRewardEligibility.ANY_SETTLED_OUTCOME
    )
    _drive_turns(loop, measurements=(None, None, _MEASURED))

    stream = loop.latest_runtime_replay_reward_stream
    batch_transitions = _staged_batch_transitions(loop)

    assert stream.settled_transition_count == len(batch_transitions) == 4
    assert stream.eligible_transition_count == 4
    assert stream.ineligible_transition_count == 0
    assert stream.nonzero_reward_transition_count == 4
    # Turn 2 settles against a silent tick (the PE owner's synthesized 0.42
    # is what the default contract pays); turn 3 settles against the measured
    # tick (0.2). Two settlements each, world + self.
    assert stream.realized_action_payoff_sum == pytest.approx(
        2 * 0.42 + 2 * 0.2
    )
    assert stream.reward_sum == pytest.approx(
        sum(transition.reward for transition in batch_transitions)
    )


# --------------------------------------------------------------------------
# contract validation + rollback surface
# --------------------------------------------------------------------------


def test_kernel_defaults_are_the_historical_lane() -> None:
    loop = ETANLJointLoop()
    assert (
        loop.runtime_replay_reward_eligibility
        is RuntimeReplayRewardEligibility.ANY_SETTLED_OUTCOME
    )
    assert loop.runtime_replay_outcome_payoff_reward is None
    assert loop.runtime_replay_latent_unit_clamp is False


def test_joint_loop_rejects_an_untyped_eligibility_declaration() -> None:
    with pytest.raises(TypeError, match="RuntimeReplayRewardEligibility"):
        ETANLJointLoop(
            runtime_replay_reward_eligibility="environment-measured-only",
        )


def test_joint_loop_rejects_a_non_bool_outcome_payoff_declaration() -> None:
    with pytest.raises(TypeError, match="must be None or bool"):
        ETANLJointLoop(runtime_replay_outcome_payoff_reward="yes")


# --------------------------------------------------------------------------
# the gate must read the field the payout pays (B2)
# --------------------------------------------------------------------------


def _diverging_pe() -> PredictionErrorSnapshot:
    """A PE snapshot whose action axis is NOT the environment's measurement.

    Reachable today without touching the kernel: any domain that assembles its
    own ``PredictionActionContext`` (or leaves ``environment_action_payoff``
    unset) keeps the PE owner's synthesized action axis, while the environment
    still publishes a measurement. Historically the eligibility gate read the
    measurement (0.2) and the payout paid the synthesized axis (0.9).
    """

    return _prediction_error(action_payoff=0.9)


def test_strict_eligibility_raises_when_the_paid_axis_is_not_the_gated_one() -> None:
    with pytest.raises(Exception) as excinfo:
        _settle(
            _captured_sandbox(),
            environment_outcome=_outcome(measurement=_MEASURED),
            prediction_error_snapshot=_diverging_pe(),
            reward_eligibility=(
                RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY
            ),
        )

    assert isinstance(excinfo.value, RuntimeReplayRewardEligibilityError)
    message = str(excinfo.value)
    # The message must name both readings so the seam is diagnosable.
    assert "measurement.action_payoff=0.2" in message
    assert "prediction_error.actual_outcome.action_payoff=0.9" in message


def test_gate_payout_invariant_holds_even_when_the_payoff_is_ablated_off() -> None:
    """A broken wiring must not lie dormant behind an ablation switch.

    ``outcome_payoff_reward_enabled=False`` zeroes the paid value, so a check
    placed only on the payout expression would stay silent here and start
    paying the wrong quantity the moment the arm is switched back on.
    """

    with pytest.raises(RuntimeReplayRewardEligibilityError):
        _settle(
            _captured_sandbox(),
            environment_outcome=_outcome(measurement=_MEASURED),
            prediction_error_snapshot=_diverging_pe(),
            outcome_payoff_reward_enabled=False,
            reward_eligibility=(
                RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY
            ),
        )


def test_default_eligibility_is_unaffected_by_the_divergence_check() -> None:
    """Rollback proof: a domain declaring nothing keeps the historical value.

    The same divergent inputs settle silently and pay the PE owner's axis,
    exactly as before Wave 3.
    """

    settlement = _settle(
        _captured_sandbox(),
        environment_outcome=_outcome(measurement=_MEASURED),
        prediction_error_snapshot=_diverging_pe(),
    )

    assert settlement.realized_action_payoff == pytest.approx(0.9)
    assert settlement.reward == pytest.approx(0.96)
    assert settlement.reward_eligible is True


def test_ineligible_transition_never_reaches_the_divergence_check() -> None:
    """No measurement means nothing was authorized; there is nothing to police."""

    settlement = _settle(
        _captured_sandbox(),
        environment_outcome=_outcome(measurement=None),
        prediction_error_snapshot=_diverging_pe(),
        reward_eligibility=(
            RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY
        ),
    )

    assert settlement.reward_eligible is False
    assert settlement.reward == 0.0


# --------------------------------------------------------------------------
# both replay lanes must reconstruct on the same latent bound (B1)
# --------------------------------------------------------------------------


def _torch_report_stub():
    from volvence_zero.internal_rl.torch_causal_ppo import TorchPPOReport

    return TorchPPOReport(
        backend="recorder",
        transition_count=0,
        policy_loss=0.0,
        value_loss=0.0,
        approx_kl=0.0,
        clip_fraction=0.0,
        entropy=0.0,
        parameters_changed=0,
        parameter_change_rate=0.0,
        wrote_back=False,
    )


def _settled_runtime_rollout(*, latent_unit_clamp: bool):
    """One real runtime-replay rollout plus the sandbox that produced it."""

    store = MetacontrollerParameterStore(n_z=4)
    store.track_weights[Track.WORLD] = (0.4, 0.3, 0.2, 0.1)
    policy = FullLearnedTemporalPolicy(parameter_store=store)
    policy.set_runtime_track_modulation(0.3)
    sandbox = InternalRLSandbox(
        policy=policy,
        latent_unit_clamp=latent_unit_clamp,
        rl_backend=WiringLevel.SHADOW,
    )
    substrate = _substrate("initial heading observation")
    temporal, runtime_state = _temporal_and_runtime_state(policy, substrate)
    sandbox.capture_runtime_action(
        turn_index=1,
        track=Track.WORLD,
        prediction_id="prediction-1",
        substrate_snapshot=substrate,
        temporal_snapshot=temporal,
        runtime_state=runtime_state,
    )
    settlement = sandbox.settle_runtime_action(
        next_substrate_snapshot=_substrate("next observation"),
        environment_outcome=_outcome(measurement=_MEASURED),
        prediction_error_snapshot=_prediction_error(),
        credit_snapshot=_credit(),
    )
    assert settlement.rollout is not None
    return sandbox, settlement.rollout


def _optimize_recording_both_lanes(monkeypatch, *, latent_unit_clamp: bool):
    """Run one optimizer batch, recording what each replay lane was given."""

    import functools

    from volvence_zero.internal_rl import sandbox as sandbox_module
    from volvence_zero.internal_rl import torch_causal_ppo as torch_module

    sandbox, rollout = _settled_runtime_rollout(
        latent_unit_clamp=latent_unit_clamp
    )
    pure_lane: list[bool] = []
    torch_lane: list[object] = []

    real_distribution = sandbox_module.runtime_replay_policy_distribution

    def recording_distribution(**kwargs):
        pure_lane.append(kwargs.get("latent_unit_clamp"))
        return real_distribution(**kwargs)

    @functools.wraps(torch_module.torch_causal_ppo_update)
    def recording_torch_update(**kwargs):
        torch_lane.append(kwargs.get("latent_unit_clamp", "<omitted>"))
        return _torch_report_stub()

    monkeypatch.setattr(
        sandbox_module,
        "runtime_replay_policy_distribution",
        recording_distribution,
    )
    monkeypatch.setattr(sandbox_module, "is_torch_available", lambda: True)
    monkeypatch.setattr(
        torch_module, "torch_causal_ppo_update", recording_torch_update
    )

    sandbox._causal_policy.optimize(rollout=rollout)
    return pure_lane, torch_lane


@pytest.mark.parametrize("latent_unit_clamp", [False, True])
def test_both_replay_lanes_receive_the_same_latent_unit_clamp(
    monkeypatch, latent_unit_clamp: bool
) -> None:
    """The declared bound reaches the torch lane, not only the pure lane.

    Before this fix the production call site never forwarded the kwarg, so a
    domain declaring ``latent_unit_clamp=True`` with an ACTIVE torch backend
    had the pure lane reconstruct on ``[0, 1]`` and the torch lane -- the
    authoritative writer -- reconstruct on ``[-1, 1]`` for the same batch.
    """

    pure_lane, torch_lane = _optimize_recording_both_lanes(
        monkeypatch, latent_unit_clamp=latent_unit_clamp
    )

    assert pure_lane, "the pure replay lane must have reconstructed the batch"
    assert torch_lane, "the torch lane must have been reached"
    assert set(pure_lane) == {latent_unit_clamp}
    assert set(torch_lane) == {latent_unit_clamp}
    assert set(pure_lane) == set(torch_lane)


def test_latent_bound_contract_rejects_a_torch_call_that_drops_the_kwarg() -> None:
    """Negative control: the guard is what makes the forwarding non-optional."""

    from volvence_zero.internal_rl.sandbox import (
        assert_runtime_replay_latent_bounds_agree,
    )
    from volvence_zero.internal_rl.torch_causal_ppo import (
        torch_causal_ppo_update,
    )

    payload = dict(
        parameter_store=MetacontrollerParameterStore(n_z=4),
        value_weights={},
        value_bias={},
        track=Track.WORLD,
        transitions=(),
        n_z=4,
    )

    with pytest.raises(RuntimeReplayLatentBoundContractError, match="omits"):
        assert_runtime_replay_latent_bounds_agree(
            torch_update=torch_causal_ppo_update,
            call_kwargs=payload,
            latent_unit_clamp=True,
        )

    with pytest.raises(RuntimeReplayLatentBoundContractError, match="violated"):
        assert_runtime_replay_latent_bounds_agree(
            torch_update=torch_causal_ppo_update,
            call_kwargs={**payload, "latent_unit_clamp": False},
            latent_unit_clamp=True,
        )

    # Agreement passes and reports the owner's bound.
    from volvence_zero.temporal.interface import LATENT_CODE_BOUNDS

    assert (
        assert_runtime_replay_latent_bounds_agree(
            torch_update=torch_causal_ppo_update,
            call_kwargs={**payload, "latent_unit_clamp": True},
            latent_unit_clamp=True,
        )
        == LATENT_CODE_BOUNDS
    )
    assert (
        assert_runtime_replay_latent_bounds_agree(
            torch_update=torch_causal_ppo_update,
            call_kwargs={**payload, "latent_unit_clamp": False},
            latent_unit_clamp=False,
        )
        == (-1.0, 1.0)
    )


def test_sandbox_unit_clamp_is_the_temporal_owners_bound() -> None:
    """The unit branch is derived from the owner, not re-declared here."""

    from volvence_zero.internal_rl import sandbox as sandbox_module
    from volvence_zero.temporal.interface import LATENT_CODE_BOUNDS

    lower, upper = LATENT_CODE_BOUNDS
    assert sandbox_module._clamp_unit(upper + 5.0) == upper
    assert sandbox_module._clamp_unit(lower - 5.0) == lower
    assert sandbox_module.resolve_latent_code_bounds(
        latent_unit_clamp=True
    ) == LATENT_CODE_BOUNDS
    # ...and the two lanes' resolvers agree member for member.
    from volvence_zero.internal_rl.torch_causal_ppo import (
        resolve_latent_code_bounds as torch_resolve,
    )

    for declaration in (False, True):
        assert sandbox_module.resolve_latent_code_bounds(
            latent_unit_clamp=declaration
        ) == torch_resolve(latent_unit_clamp=declaration)


# --------------------------------------------------------------------------
# latent clamp convention
# --------------------------------------------------------------------------


def _distribution_kwargs() -> dict:
    return dict(
        base_mean=(0.10, 0.10, 0.10, 0.10),
        base_std=(0.20, 0.20, 0.20, 0.20),
        previous_code=(0.05, 0.05, 0.05, 0.05),
        beta_t=1.0,
        track_weights=(0.25, 0.25, 0.25, 0.25),
        other_track_sum=(0.50, 0.50, 0.50, 0.50),
        modulation_strength=0.3,
        # A head residual large enough to push the candidate mean below the
        # live plant's [0, 1] latent floor.
        action_head_residual=(-0.9, -0.9, -0.9, -0.9),
    )


def test_latent_unit_clamp_is_opt_in_and_default_keeps_the_signed_bound() -> None:
    signed_mean, signed_std = runtime_replay_policy_distribution(
        **_distribution_kwargs()
    )
    unit_mean, unit_std = runtime_replay_policy_distribution(
        **_distribution_kwargs(),
        latent_unit_clamp=True,
    )

    assert min(signed_mean) < 0.0, "default must stay the signed rollback"
    assert min(unit_mean) == 0.0
    assert max(unit_mean) <= 1.0
    # The clamp only bounds the mean; the posterior spread is untouched.
    assert unit_std == signed_std


def test_latent_unit_clamp_does_not_touch_reward_or_advantage_bounds() -> None:
    """Rewards stay signed under the unit-clamp declaration."""

    store = MetacontrollerParameterStore(n_z=4)
    store.track_weights[Track.WORLD] = (0.4, 0.3, 0.2, 0.1)
    policy = FullLearnedTemporalPolicy(parameter_store=store)
    policy.set_runtime_track_modulation(0.3)
    sandbox = InternalRLSandbox(policy=policy, latent_unit_clamp=True)
    substrate = _substrate("initial heading observation")
    temporal, runtime_state = _temporal_and_runtime_state(policy, substrate)
    sandbox.capture_runtime_action(
        turn_index=1,
        track=Track.WORLD,
        prediction_id="prediction-1",
        substrate_snapshot=substrate,
        temporal_snapshot=temporal,
        runtime_state=runtime_state,
    )

    settlement = sandbox.settle_runtime_action(
        next_substrate_snapshot=_substrate("next observation"),
        environment_outcome=_outcome(
            measurement=EnvironmentMeasurement(
                task_progress=0.1,
                action_payoff=-0.6,
                terminal=False,
            )
        ),
        prediction_error_snapshot=_prediction_error(action_payoff=-0.6),
        credit_snapshot=_credit(credit_value=-0.5),
        reward_eligibility=(
            RuntimeReplayRewardEligibility.ENVIRONMENT_MEASURED_ONLY
        ),
    )

    assert settlement.realized_action_payoff == pytest.approx(-0.6)
    assert settlement.segment_bonus == pytest.approx(-0.06)
    assert settlement.reward == pytest.approx(-0.66)
