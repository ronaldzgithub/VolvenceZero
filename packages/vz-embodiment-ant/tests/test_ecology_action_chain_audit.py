"""P0-A action-chain audit regression tests (plan 05 s3.2).

These lock the three properties whose absence made the committed v1 PASS
vacuous: the shared-initial checkpoint is gated on input reachability only, a
retention floor may never be derived from a zero baseline, and a head that
emits no turn cannot satisfy the sign-consistency gate.
"""

from __future__ import annotations

import math

import pytest

from volvence_ant.experiments.ecology_mechanism_audit import (
    ECOLOGY_AUDIT_BACKEND_PARITY_EXERCISE_STEPS,
    _GATE_MODE_INPUT_REACHABILITY,
    _GATE_MODE_POST_TRAINING,
    _PRIMARY_PROBE_SEED_OFFSET,
    _action_head_updates,
    _backend_lane_availability,
    _evaluate_action_snapshot,
    _lane_coverage_failure,
    _lane_wiring_mismatch,
    _lateral_bias,
    _max_distribution_delta,
    _probe_checkpoints,
    _retention_floor,
    _sign_consistency,
    _turn_sign,
    _update_retention_baselines,
    EcologyMechanismAuditConfig,
    EcologyMechanismAuditError,
)
from volvence_ant.experiments.ecology_probe import (
    ECOLOGY_PROBE_LANE_WIRING_KEYS,
    ECOLOGY_PROBE_NEAR_NULL_SPACE_ALIGNMENT,
    EcologyActionProbe,
    EcologyBackendExecutionEvidence,
    EcologyCheckpointActionProbe,
    EcologyProbeBackendLane,
    EcologyProbeKind,
    ecology_probe_lane_declared_active_backends,
    ecology_probe_lane_expected_wiring,
    run_ecology_action_probes,
)
from volvence_ant.env import AntWorld, AntWorldConfig, ButterSource
from volvence_ant.runtime import (
    AntObjectiveKind,
    AntSenseSchema,
    AntSession,
    AntSessionConfig,
)


def _probe(
    kind: EcologyProbeKind,
    *,
    left_turn: float,
    right_turn: float,
    reachable: bool = True,
    update_step: int = 0,
    residual: tuple[float, ...] = (0.0,),
) -> EcologyActionProbe:
    return EcologyActionProbe(
        kind=kind,
        left_sensor_pair=(1.0, 0.0),
        right_sensor_pair=(0.0, 1.0),
        left_code=(0.1, 0.2),
        right_code=(0.2, 0.1),
        left_turn=left_turn,
        right_turn=right_turn,
        code_l1_delta=0.2,
        turn_delta=abs(left_turn - right_turn),
        input_reachable=reachable,
        action_sensitive=abs(left_turn - right_turn) > 1e-4,
        left_action_head_residual=residual,
        right_action_head_residual=residual,
        left_action_head_update_step=update_step,
        right_action_head_update_step=update_step,
    )


def _body(
    body_id: int,
    *,
    left_turn: float,
    right_turn: float,
    reachable: bool = True,
    update_step: int = 0,
    residual: tuple[float, ...] = (0.0,),
    policy_fingerprint: str = "learned",
) -> EcologyCheckpointActionProbe:
    return EcologyCheckpointActionProbe(
        body_id=body_id,
        checkpoint_id=f"cp-{body_id}",
        policy_fingerprint=policy_fingerprint,
        temporal_learning_fingerprint="temporal",
        probes=tuple(
            _probe(
                kind,
                left_turn=left_turn,
                right_turn=right_turn,
                reachable=reachable,
                update_step=update_step,
                residual=residual,
            )
            for kind in EcologyProbeKind
        ),
    )


def _config() -> EcologyMechanismAuditConfig:
    return EcologyMechanismAuditConfig(
        n_ants=1,
        temporal_latent_dim=4,
        episode_rounds=1,
        episodes_per_stage=1,
        evaluation_rounds=3,
        seed=5,
    )


def test_shared_initial_is_gated_on_input_reachability_only() -> None:
    """A cold exclusive-steering head is exactly zero BY DESIGN.

    Applying the post-training turn gate to the shared-initial checkpoint is
    block-by-construction, which is why the v1/v2 audit could only "pass" it
    by rolling every trained checkpoint back to the cold one.
    """

    config = _config()
    cold = (_body(0, left_turn=0.0, right_turn=0.0),)

    reachability = _evaluate_action_snapshot(
        config=config,
        arm="shared-initial",
        label="shared-initial",
        stage="initial",
        tier="initial",
        episode_index=-1,
        gate_mode=_GATE_MODE_INPUT_REACHABILITY,
        body_reports=cold,
        retention_baselines={},
    )
    post_training = _evaluate_action_snapshot(
        config=config,
        arm="learned",
        label="final",
        stage="final",
        tier="final",
        episode_index=0,
        gate_mode=_GATE_MODE_POST_TRAINING,
        body_reports=cold,
        retention_baselines={},
    )

    assert reachability.passed
    assert reachability.failures == ()
    assert not post_training.passed
    assert any("turn-delta" in item for item in post_training.failures)


def test_unreachable_input_fails_even_the_cold_gate() -> None:
    config = _config()
    snapshot = _evaluate_action_snapshot(
        config=config,
        arm="shared-initial",
        label="shared-initial",
        stage="initial",
        tier="initial",
        episode_index=-1,
        gate_mode=_GATE_MODE_INPUT_REACHABILITY,
        body_reports=(_body(0, left_turn=0.0, right_turn=0.0, reachable=False),),
        retention_baselines={},
    )

    assert not snapshot.passed
    assert all("input-unreachable" in item for item in snapshot.failures)


def test_retention_floor_refuses_a_zero_baseline() -> None:
    config = _config()

    with pytest.raises(EcologyMechanismAuditError) as excinfo:
        _retention_floor(
            baseline_turn_delta=0.0,
            config=config,
            context="unit",
        )

    assert "vacuous" in str(excinfo.value)
    assert _retention_floor(
        baseline_turn_delta=0.4,
        config=config,
        context="unit",
    ) == pytest.approx(0.1)


def test_retention_baselines_only_track_acquired_sensitivity() -> None:
    config = _config()
    baselines: dict[tuple[int, str], float] = {}

    _update_retention_baselines(
        baselines=baselines,
        body_reports=(_body(0, left_turn=0.0, right_turn=0.0),),
        config=config,
    )
    assert baselines == {}

    _update_retention_baselines(
        baselines=baselines,
        body_reports=(_body(0, left_turn=0.3, right_turn=-0.3),),
        config=config,
    )
    assert baselines[(0, "food")] == pytest.approx(0.6)

    # A later, weaker checkpoint must not lower the peak the arm reached.
    _update_retention_baselines(
        baselines=baselines,
        body_reports=(_body(0, left_turn=0.1, right_turn=-0.1),),
        config=config,
    )
    assert baselines[(0, "food")] == pytest.approx(0.6)


def test_retention_collapse_after_acquisition_fails() -> None:
    config = _config()
    snapshot = _evaluate_action_snapshot(
        config=config,
        arm="learned",
        label="final",
        stage="final",
        tier="final",
        episode_index=1,
        gate_mode=_GATE_MODE_POST_TRAINING,
        # 0.02 total delta against a 0.6 peak is 3.3%, far below the frozen
        # 25% retention floor.
        body_reports=(_body(0, left_turn=0.01, right_turn=-0.01),),
        retention_baselines={(0, "food"): 0.6, (0, "heat"): 0.6},
    )

    assert not snapshot.passed
    assert any("retention" in item for item in snapshot.failures)


def test_turn_sign_treats_sub_threshold_turns_as_no_sign() -> None:
    assert _turn_sign(1e-9, threshold=1e-4) == 0
    assert _turn_sign(-1e-9, threshold=1e-4) == 0
    assert _turn_sign(0.3, threshold=1e-4) == 1
    assert _turn_sign(-0.3, threshold=1e-4) == -1


def _cold_checkpoint(*, temporal_latent_dim: int, seed: int):
    world = AntWorld(
        config=AntWorldConfig(seed=seed, step_size=0.4),
        world_objects=(ButterSource(object_id="probe", x=0.6, y=0.35),),
    )
    session = AntSession(
        world,
        config=AntSessionConfig(
            temporal_latent_dim=temporal_latent_dim,
            session_id=f"test:p0a:cold:{seed}",
            seed=seed,
            objective=AntObjectiveKind.ECOLOGY,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
        ),
    )
    return session.export_learning_checkpoint(
        checkpoint_id=f"test:p0a:cold:{seed}",
        include_runtime_replay=False,
    )


async def test_sign_consistency_rejects_a_head_with_no_direction() -> None:
    """Zero turns repeat perfectly; that is not a consistent direction."""

    config = _config()
    cold = _cold_checkpoint(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=3,
    )
    first_repeat = await _probe_checkpoints(
        config=config,
        checkpoints=(cold,),
        seed_offset=_PRIMARY_PROBE_SEED_OFFSET,
    )

    results, seeds = await _sign_consistency(
        config=config,
        checkpoints=(cold,),
        first_repeat=first_repeat,
    )

    assert len(seeds) == config.sign_repeat_count
    assert len(set(seeds)) == config.sign_repeat_count
    assert {item.kind for item in results} == {"food", "heat"}
    assert all(item.left_turn_signs == (0, 0, 0) for item in results)
    assert all(item.right_turn_signs == (0, 0, 0) for item in results)
    assert not any(item.consistent for item in results)


def test_lateral_bias_flags_a_colony_wide_same_direction_turn() -> None:
    config = _config()

    biased = _lateral_bias(
        config=config,
        body_reports=tuple(
            # Both antennae drive the same +0.3 turn: a fixed bias, not
            # steering.
            _body(body_id, left_turn=0.3, right_turn=0.3)
            for body_id in range(4)
        ),
    )
    steering = _lateral_bias(
        config=config,
        body_reports=tuple(
            _body(body_id, left_turn=0.3, right_turn=-0.3)
            for body_id in range(4)
        ),
    )

    assert all(item.systematic_same_direction for item in biased)
    assert not any(item.systematic_same_direction for item in steering)
    assert steering[0].mean_contrast == pytest.approx(0.3)


def test_action_head_update_reads_owner_published_evidence() -> None:
    initial = _FakeCheckpoint("cold")
    learned_changed = _FakeCheckpoint("trained")

    updated = _action_head_updates(
        initial=(initial,),
        learned=(learned_changed,),
        body_reports=(
            _body(0, left_turn=0.2, right_turn=-0.2, update_step=7),
        ),
    )
    never_updated = _action_head_updates(
        initial=(initial,),
        learned=(_FakeCheckpoint("cold"),),
        body_reports=(
            _body(0, left_turn=0.0, right_turn=0.0, update_step=0),
        ),
    )
    nan_update = _action_head_updates(
        initial=(initial,),
        learned=(learned_changed,),
        body_reports=(
            _body(
                0,
                left_turn=0.2,
                right_turn=-0.2,
                update_step=7,
                residual=(math.nan,),
            ),
        ),
    )

    assert updated[0].passed
    assert updated[0].update_step == 7
    assert not never_updated[0].passed
    assert not never_updated[0].policy_fingerprint_changed
    assert not nan_update[0].passed
    assert not nan_update[0].residual_finite


class _FakeCheckpoint:
    """Minimal stand-in exposing only the owner-published fingerprint."""

    def __init__(self, policy_fingerprint: str) -> None:
        self.policy_fingerprint = policy_fingerprint


def test_backend_lane_availability_is_explicit() -> None:
    for lane in EcologyProbeBackendLane:
        available, reason = _backend_lane_availability(lane)
        assert available is (reason == "")


async def test_probe_publishes_the_wiring_it_actually_ran_on() -> None:
    """plan 05:123 -- a parity lane must prove which backend it measured.

    The probe publishes the session's OWN backend wiring, so a lane whose
    rollout config silently failed to reach the session can be rejected
    instead of contributing a green parity result.
    """

    for lane in EcologyProbeBackendLane:
        probes = await run_ecology_action_probes(
            temporal_latent_dim=4,
            seed=17,
            backend_lane=lane,
        )
        expected = ecology_probe_lane_expected_wiring(lane)

        assert expected
        assert tuple(name for name, _ in expected) == (
            ECOLOGY_PROBE_LANE_WIRING_KEYS
        )
        assert all(
            probe.observed_backend_wiring == expected for probe in probes
        )


async def test_lane_wiring_mismatch_is_reported_not_swallowed() -> None:
    config = _config()
    honest = _body(0, left_turn=0.2, right_turn=-0.2)
    mismatched = EcologyCheckpointActionProbe(
        body_id=0,
        checkpoint_id="cp-0",
        policy_fingerprint="learned",
        temporal_learning_fingerprint="temporal",
        probes=honest.probes,
        observed_backend_wiring=(("temporal_runtime_backend", "disabled"),),
    )

    assert _lane_wiring_mismatch(
        lane=EcologyProbeBackendLane.RUNTIME,
        body_reports=(mismatched,),
    )
    assert config.backend_parity_tolerance > 0.0


def _execution(
    *,
    runtime: str = "disabled",
    ssl: str = "disabled",
    internal_rl: str = "disabled",
    ssl_trained_steps: int = 0,
    ssl_torch_backends: tuple[str, ...] = ("disabled",),
    internal_rl_report_published: bool = True,
    internal_rl_torch_backends: tuple[str, ...] = ("disabled",),
    internal_rl_torch_wrote_back: bool = False,
) -> EcologyBackendExecutionEvidence:
    return EcologyBackendExecutionEvidence(
        exercise_steps=ECOLOGY_AUDIT_BACKEND_PARITY_EXERCISE_STEPS,
        temporal_runtime_backend_applied=runtime,
        ssl_backend_applied=ssl,
        internal_rl_backend_applied=internal_rl,
        ssl_trained_steps=ssl_trained_steps,
        ssl_report_published=True,
        ssl_torch_backends=ssl_torch_backends,
        ssl_torch_parameters_changed=0,
        ssl_torch_wrote_back=False,
        internal_rl_report_published=internal_rl_report_published,
        internal_rl_torch_backends=internal_rl_torch_backends,
        internal_rl_torch_parameters_changed=31,
        internal_rl_torch_wrote_back=internal_rl_torch_wrote_back,
    )


def test_lane_coverage_is_decided_only_by_owner_published_execution() -> None:
    """plan 05:123 -- coverage asks "did this backend run", nothing else.

    The v3 audit answered it with "do this lane's numbers differ from the
    reference", which declared a correct, exactly-agreeing backend to be
    unevaluated. Nothing in this function can see the reference at all.
    """

    fully_exercised = _execution(
        ssl="active",
        internal_rl="active",
        ssl_trained_steps=6,
        ssl_torch_backends=("active",),
        internal_rl_torch_backends=("active",),
        internal_rl_torch_wrote_back=True,
    )

    assert (
        _lane_coverage_failure(
            lane=EcologyProbeBackendLane.TORCH,
            execution=fully_exercised,
        )
        == ""
    )
    assert (
        _lane_coverage_failure(
            lane=EcologyProbeBackendLane.PURE,
            execution=_execution(),
        )
        == ""
    )
    assert (
        _lane_coverage_failure(
            lane=EcologyProbeBackendLane.RUNTIME,
            execution=_execution(runtime="active"),
        )
        == ""
    )

    # The SSL trainer never trained: the ant's residual trace is shorter than
    # two steps, so the torch SSL path cannot have executed.
    never_trained = _lane_coverage_failure(
        lane=EcologyProbeBackendLane.TORCH,
        execution=_execution(
            ssl="active",
            internal_rl="active",
            ssl_trained_steps=0,
            internal_rl_torch_backends=("active",),
            internal_rl_torch_wrote_back=True,
        ),
    )
    assert "trained_steps=0" in never_trained

    # No full optimization cycle ran in the exercise budget.
    no_cycle = _lane_coverage_failure(
        lane=EcologyProbeBackendLane.TORCH,
        execution=_execution(
            ssl="active",
            internal_rl="active",
            ssl_trained_steps=6,
            ssl_torch_backends=("active",),
            internal_rl_report_published=False,
            internal_rl_torch_backends=(),
        ),
    )
    assert "never published an optimization report" in no_cycle

    # The reference lane may not secretly engage an accelerated backend.
    contaminated = _lane_coverage_failure(
        lane=EcologyProbeBackendLane.PURE,
        execution=_execution(runtime="active"),
    )
    assert "pure reference lane" in contaminated


def test_max_distribution_delta_counts_a_missing_family_at_full_mass() -> None:
    assert _max_distribution_delta(
        (("a", 0.5), ("b", 0.5)),
        (("a", 0.5), ("b", 0.5)),
    ) == pytest.approx(0.0)
    assert _max_distribution_delta(
        (("a", 1.0),),
        (("a", 0.5), ("b", 0.5)),
    ) == pytest.approx(0.5)
    with pytest.raises(EcologyMechanismAuditError):
        _max_distribution_delta((("a", 0.5), ("a", 0.5)), (("a", 1.0),))


async def test_the_exercise_budget_is_what_makes_the_torch_lane_run() -> None:
    """The reviewer's root cause, pinned in both directions.

    A single no-optimize probe step can never reach the joint loop's first
    full optimization cycle, so ``internal_rl_backend`` never executes and the
    lane looks bit-identical to the pure reference for a reason that says
    nothing about the backend. With the frozen exercise budget it really runs
    and really writes back.
    """

    lane = EcologyProbeBackendLane.TORCH
    assert ecology_probe_lane_declared_active_backends(lane) == (
        "temporal_ssl_backend",
        "internal_rl_backend",
    )

    unexercised = await run_ecology_action_probes(
        temporal_latent_dim=4,
        seed=17,
        backend_lane=lane,
        exercise_steps=0,
    )
    exercised = await run_ecology_action_probes(
        temporal_latent_dim=4,
        seed=17,
        backend_lane=lane,
        exercise_steps=ECOLOGY_AUDIT_BACKEND_PARITY_EXERCISE_STEPS,
    )

    cold = unexercised[0].backend_execution
    warm = exercised[0].backend_execution
    assert cold is not None
    assert warm is not None
    assert cold.internal_rl_backend_applied == "active"
    assert not cold.internal_rl_report_published
    assert warm.internal_rl_report_published
    assert warm.internal_rl_torch_backends == ("active",)
    assert warm.internal_rl_torch_wrote_back
    assert warm.internal_rl_torch_parameters_changed > 0
    # ...and the honest remaining gap: the SSL lane still never trains.
    assert warm.ssl_backend_applied == "active"
    assert warm.ssl_trained_steps == 0
    assert "trained_steps=0" in _lane_coverage_failure(
        lane=lane,
        execution=warm,
    )


async def test_probe_publishes_sense_hidden_state_and_null_space_diagnostic() -> None:
    probes = await run_ecology_action_probes(
        temporal_latent_dim=4,
        seed=17,
        backend_lane=EcologyProbeBackendLane.PURE,
    )

    by_kind = {item.kind: item for item in probes}
    food = by_kind[EcologyProbeKind.FOOD]
    # plan 05:123 compares the final code, the ACTION DISTRIBUTION and the
    # turn; the probe has to publish the last two, not just the code.
    assert food.left_abstract_action
    assert food.right_abstract_action
    assert food.left_action_distribution
    assert sum(mass for _name, mass in food.left_action_distribution) == (
        pytest.approx(1.0)
    )
    assert sum(mass for _name, mass in food.right_action_distribution) == (
        pytest.approx(1.0)
    )
    assert len(food.left_sense) == 19
    assert len(food.right_sense) == 19
    assert tuple(name for name, _ in food.left_sense) == tuple(
        name for name, _ in food.right_sense
    )
    assert food.left_posterior_hidden is not None
    assert food.right_posterior_hidden is not None
    assert food.left_posterior_hidden.dim == 4
    assert food.posterior_hidden_l1_delta >= 0.0
    assert len(food.motor_readout_gradient) == len(food.left_code)
    assert food.latent_delta_l2_norm > 0.0
    assert 0.0 <= food.null_space_alignment <= 1.0 + 1e-9
    assert food.in_motor_near_null_space is (
        food.null_space_alignment < ECOLOGY_PROBE_NEAR_NULL_SPACE_ALIGNMENT
    )
    assert food.backend_lane == "pure"
    assert all(item.backend_lane == "pure" for item in probes)
