"""P0 mechanism-audit regression tests.

The previous version of this file accepted ``verdict in {"PASS", "BLOCK"}``,
which cannot fail.  Both directions are exercised here: a constructed clean
evidence set must produce PASS, a constructed defective one must produce BLOCK,
and the real (tiny) audit run must name its own breakpoints.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest

from volvence_ant.evidence.provenance import (
    AntArtifactExistsError,
    AntRunProvenance,
    verify_ant_artifact_manifest,
)
from volvence_ant.evidence.runtime_profile import (
    ant_runtime_replay_rollout_config,
)
from volvence_ant.experiments.ecology_mechanism_audit import (
    ECOLOGY_AUDIT_BACKEND_PARITY_EXERCISE_STEPS,
    ECOLOGY_AUDIT_BACKEND_PARITY_TOLERANCE,
    ECOLOGY_AUDIT_BODY_PASS_RATIO,
    ECOLOGY_AUDIT_DECLARED_GAPS,
    ECOLOGY_AUDIT_CODE_DELTA_THRESHOLD,
    ECOLOGY_AUDIT_NEGATIVE_CONTROL_SWITCH_RATE_CEILING,
    ECOLOGY_AUDIT_RETENTION_RATIO,
    ECOLOGY_AUDIT_SEGMENT_CREDIT_PARITY_TOLERANCE,
    ECOLOGY_AUDIT_SIGN_REPEAT_COUNT,
    ECOLOGY_AUDIT_SWITCH_LOCALIZATION_WINDOW,
    ECOLOGY_AUDIT_TIMEOUT_CLOSURE_RATIO_CEILING,
    ECOLOGY_AUDIT_TURN_DELTA_THRESHOLD,
    ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION,
    EcologyActionChainSnapshot,
    EcologyActionHeadUpdate,
    EcologyBackendParityLane,
    EcologyBoundaryLocalization,
    EcologyDeclaredGap,
    EcologyDiagnosticSurface,
    EcologyFrozenEvaluationAudit,
    EcologyLateralBias,
    EcologyMechanismAuditConfig,
    EcologyMechanismGate,
    EcologySegmentCreditParity,
    EcologySignConsistency,
    EcologyTemporalSwitchAudit,
    EcologyTemporalTick,
    EcologyTransitionTrace,
    build_ecology_mechanism_gates,
    ecology_mechanism_audit_seed_schedule,
    run_ecology_mechanism_audit,
)
from volvence_ant.experiments.ecology_probe import (
    EcologyBackendExecutionEvidence,
    EcologyCheckpointActionProbe,
    EcologyProbeKind,
    run_ecology_checkpoint_action_probes,
)
from volvence_ant.experiments.ecology_mechanism_audit import (
    ECOLOGY_AUDIT_FROZEN_EVALUATION_CASES,
)
from volvence_ant.experiments.ecology_curriculum import (
    EcologyCurriculumConfig,
    EcologyDataSplit,
    EcologyStage,
    EcologyTrainingTier,
    _session_config,
    _world,
)
from volvence_ant.runtime import KernelColonyRunner
from volvence_zero.joint_loop import ETANLJointLoop
from volvence_zero.runtime import WiringLevel


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DRIVER_PATH = _REPO_ROOT / "scripts" / "audit_ant_ecology_mechanisms.py"

_EXPECTED_GATE_NAMES = (
    "action_chain_input_reachability",
    "action_chain_final_sensitivity",
    "action_chain_sign_consistency",
    "action_chain_lateral_bias",
    "action_head_update_applied",
    "action_chain_no_rollback",
    "backend_lane_coverage",
    "backend_parity",
    "no_optimize_policy_stable",
    "temporal_positive_control",
    "temporal_negative_control",
    "temporal_segment_closure",
    "segment_credit_parity",
    "frozen_evaluation",
)

# The P0 gates the system genuinely does NOT satisfy at ``_audit_config()``
# (the smallest runnable budget), recorded so the audit's own verdict is
# regression-locked instead of self-fulfilling:
#   * the action head emits no turn (turn_delta ~ 1e-17) and reports
#     update_step=0, so it has no direction to be sign-consistent about, and
#     the learned policy fingerprint never leaves the shared-initial one --
#     at this budget that is partly a budget artifact and only the full-budget
#     run can say whether training would ever move it;
#   * all three backend lanes are now exercised and remain inside the frozen
#     parity band, while both frozen-evaluation cases keep every tracked owner
#     stable; those former breakpoints are intentionally absent below;
#   * temporal segments still never close from a beta switch, so the audit has
#     not yet demonstrated learned temporal-boundary formation.
_RECORDED_P0_BREAKPOINTS = (
    "action_chain_final_sensitivity",
    "action_chain_sign_consistency",
    "action_head_update_applied",
    "action_chain_no_rollback",
    "temporal_segment_closure",
)


def _audit_config() -> EcologyMechanismAuditConfig:
    return EcologyMechanismAuditConfig(
        n_ants=1,
        temporal_latent_dim=4,
        episode_rounds=1,
        episodes_per_stage=1,
        evaluation_rounds=3,
        seed=5,
    )


def _clean_snapshot(name: str) -> EcologyActionChainSnapshot:
    return EcologyActionChainSnapshot(
        arm="learned",
        label=name,
        stage=name,
        tier=name,
        episode_index=0,
        gate_mode="post-training",
        body_reports=(),
        required_body_passes=1,
        passing_bodies=1,
        passed=True,
        failures=(),
    )


def _clean_trace(
    *,
    label: str,
    switch_ticks: tuple[int, ...],
    close_reasons: tuple[tuple[str, int], ...],
    timeout_ratio: float,
    localized: bool,
    ticks: tuple[EcologyTemporalTick, ...] = (),
) -> EcologyTransitionTrace:
    return EcologyTransitionTrace(
        label=label,
        segment_credit_enabled=True,
        checkpoint_id="cp",
        ticks=ticks,
        boundary_localizations=(
            EcologyBoundaryLocalization(
                phase="safe_to_harmful",
                boundary_tick=19,
                nearest_switch_tick=19 if localized else -1,
                distance=0 if localized else 99,
                localized=localized,
            ),
        ),
        switch_ticks=switch_ticks,
        switch_rate=len(switch_ticks) / 28.0,
        close_reason_counts=close_reasons,
        closed_segment_count=sum(count for _reason, count in close_reasons),
        timeout_closure_ratio=timeout_ratio,
        pickups=1,
        deliveries=1,
        harmful_ticks=4,
    )


def _clean_temporal(
    *,
    negative_switch_rate: float = 0.0,
    parity_passed: bool = True,
    localized: bool = True,
    close_reasons: tuple[tuple[str, int], ...] = (
        ("beta-switch", 6),
        ("environment-milestone", 3),
    ),
    timeout_ratio: float = 0.0,
    ticks: tuple[EcologyTemporalTick, ...] = (),
) -> EcologyTemporalSwitchAudit:
    negative = EcologyTransitionTrace(
        label="negative-control",
        segment_credit_enabled=True,
        checkpoint_id="cp",
        ticks=ticks,
        boundary_localizations=(),
        switch_ticks=(),
        switch_rate=negative_switch_rate,
        close_reason_counts=(),
        closed_segment_count=0,
        timeout_closure_ratio=0.0,
        pickups=0,
        deliveries=0,
        harmful_ticks=0,
    )
    return EcologyTemporalSwitchAudit(
        positive_control=_clean_trace(
            label="positive-control",
            switch_ticks=(19, 21),
            close_reasons=close_reasons,
            timeout_ratio=timeout_ratio,
            localized=localized,
            ticks=ticks,
        ),
        negative_control=negative,
        segment_credit_off_control=_clean_trace(
            label="segment-credit-off",
            switch_ticks=(19,),
            close_reasons=(),
            timeout_ratio=0.0,
            localized=localized,
            ticks=ticks,
        ),
        parity=EcologySegmentCreditParity(
            tolerance=ECOLOGY_AUDIT_SEGMENT_CREDIT_PARITY_TOLERANCE,
            max_sense_delta=0.0,
            max_turn_delta=0.0,
            max_step_delta=0.0,
            lineage_aligned=parity_passed,
            lineage_differences=(),
            first_misaligned_tick=-1,
            passed=parity_passed,
        ),
    )


def _clean_frozen(*, passed: bool = True) -> EcologyFrozenEvaluationAudit:
    return EcologyFrozenEvaluationAudit(
        scenario="butter_only",
        seed=307,
        rounds=3,
        gated_owner_names=("joint-loop/policy",),
        allowed_owner_names=(),
        policy_stable=passed,
        temporal_learning_stable=True,
        unstable_owner_names=() if passed else ("credit",),
        first_differences=(),
        block_reason="" if passed else "credit moved",
        replay_settlement_coverage=1.0,
        replay_lineage_coverage=1.0,
        replay_drop_count=0,
        passed=passed,
    )


def _execution(
    *,
    runtime: str,
    ssl: str,
    internal_rl: str,
    ssl_trained_steps: int = 4,
) -> EcologyBackendExecutionEvidence:
    """Owner-published evidence for a lane whose declared backends all ran."""

    return EcologyBackendExecutionEvidence(
        exercise_steps=ECOLOGY_AUDIT_BACKEND_PARITY_EXERCISE_STEPS,
        temporal_runtime_backend_applied=runtime,
        ssl_backend_applied=ssl,
        internal_rl_backend_applied=internal_rl,
        ssl_trained_steps=ssl_trained_steps,
        ssl_report_published=True,
        ssl_torch_backends=("active",) if ssl == "active" else ("disabled",),
        ssl_torch_parameters_changed=7 if ssl == "active" else 0,
        ssl_torch_wrote_back=ssl == "active",
        internal_rl_report_published=True,
        internal_rl_torch_backends=(
            ("active",) if internal_rl == "active" else ("disabled",)
        ),
        internal_rl_torch_parameters_changed=(
            31 if internal_rl == "active" else 0
        ),
        internal_rl_torch_wrote_back=internal_rl == "active",
    )


_LANE_APPLIED_WIRING = {
    "pure": ("disabled", "disabled", "disabled"),
    "runtime": ("active", "disabled", "disabled"),
    "torch": ("disabled", "active", "active"),
}
_LANE_DECLARED_ACTIVE = {
    "pure": (),
    "runtime": ("temporal_runtime_backend",),
    "torch": ("temporal_ssl_backend", "internal_rl_backend"),
}


def _clean_lane(
    *,
    lane: str,
    code_delta: float,
    execution: EcologyBackendExecutionEvidence | None = None,
    action_distribution_delta: float = 0.0,
) -> EcologyBackendParityLane:
    runtime, ssl, internal_rl = _LANE_APPLIED_WIRING[lane]
    resolved = execution or _execution(
        runtime=runtime,
        ssl=ssl,
        internal_rl=internal_rl,
    )
    worst = max(code_delta, action_distribution_delta)
    return EcologyBackendParityLane(
        lane=lane,
        measured=True,
        covered=True,
        not_measured_reason="",
        not_covered_reason="",
        declared_active_backends=_LANE_DECLARED_ACTIVE[lane],
        max_code_delta=code_delta,
        max_turn_delta=0.0,
        max_step_delta=0.0,
        max_action_head_residual_delta=0.0,
        max_action_distribution_delta=action_distribution_delta,
        abstract_actions_agree=True,
        within_tolerance=worst <= ECOLOGY_AUDIT_BACKEND_PARITY_TOLERANCE,
        observed_backend_wiring=(("internal_rl_backend", lane),),
        distinguishable_from_reference=worst > 0.0,
        backend_execution=resolved,
    )


def _clean_gate_inputs(config: EcologyMechanismAuditConfig) -> dict[str, object]:
    return {
        "config": config,
        "initial_snapshot": _clean_snapshot("shared-initial"),
        "final_learned_snapshot": _clean_snapshot("final"),
        "sign_consistency": (
            EcologySignConsistency(
                body_id=0,
                kind="food",
                probe_seeds=(1, 2, 3),
                left_turn_signs=(1, 1, 1),
                right_turn_signs=(-1, -1, -1),
                consistent=True,
            ),
        ),
        "lateral_bias": (
            EcologyLateralBias(
                kind="food",
                body_count=1,
                mean_common_mode=0.0,
                mean_contrast=0.3,
                systematic_same_direction=False,
            ),
        ),
        "action_head_updates": (
            EcologyActionHeadUpdate(
                body_id=0,
                update_step=5,
                residual_finite=True,
                policy_fingerprint_changed=True,
                passed=True,
            ),
        ),
        "backend_parity": tuple(
            _clean_lane(lane=lane, code_delta=0.0 if lane == "pure" else 1e-5)
            for lane in ("pure", "runtime", "torch")
        ),
        "rollback_episodes": (),
        "action_probe_guard_enabled": False,
        "gated_checkpoint_is_post_training": True,
        "no_optimize_stable": True,
        "temporal_switch": _clean_temporal(),
        "frozen_evaluations": (_clean_frozen(),),
    }


def test_prediction_error_bridge_adds_only_active_switch_pressure() -> None:
    active = ETANLJointLoop(
        prediction_error_temporal_switch=WiringLevel.ACTIVE,
        prediction_error_temporal_switch_strength=0.35,
        prediction_error_temporal_switch_floor=0.5,
    )
    active.set_external_learning_signals(
        {"prediction_error_magnitude": 1.0}
    )

    disabled = ETANLJointLoop(
        prediction_error_temporal_switch=WiringLevel.DISABLED,
    )
    disabled.set_external_learning_signals(
        {"prediction_error_magnitude": 1.0}
    )

    assert (
        active.world_temporal_policy.parameter_store
        .prediction_error_switch_pressure_delta()
        > 0.0
    )
    assert (
        disabled.world_temporal_policy.parameter_store
        .prediction_error_switch_pressure_delta()
        == 0.0
    )


def test_ant_milestone_boundary_replaces_pe_magnitude_event_detector() -> None:
    """The ant closes segments on typed pickup/delivery milestones, not PE.

    The v30 replay measurement (scripts/measure_ant_pe_boundary_margin.py ->
    research/ant/results/.partials/pe_boundary_margin.v30.json) found the
    routine PE distribution sits ON any plausible floor (p50 0.508, 68% of
    ticks above 0.45) while natural medium pickups settle at 0.32 on the next
    tick, so NO magnitude floor separates pickup events from routine
    prediction error. The profile therefore keeps PE additive-only at the
    generic default floor and delegates boundaries to the owner-declared
    ``EnvironmentMeasurement.discrete_milestone`` channel.
    """

    profile = ant_runtime_replay_rollout_config(
        enable_sparse_exploration=False,
    )
    # No ant-specific floor calibration survives: the additive prior uses the
    # kernel default, and the boundary channel is the typed milestone.
    assert profile.prediction_error_temporal_switch_floor == pytest.approx(0.5)
    assert (
        profile.environment_milestone_temporal_switch is WiringLevel.ACTIVE
    )

    loop = ETANLJointLoop(
        prediction_error_temporal_switch=WiringLevel.ACTIVE,
        prediction_error_temporal_switch_strength=(
            profile.prediction_error_temporal_switch_strength
        ),
        prediction_error_temporal_switch_floor=(
            profile.prediction_error_temporal_switch_floor
        ),
        environment_milestone_temporal_switch=(
            profile.environment_milestone_temporal_switch
        ),
    )
    store = loop.world_temporal_policy.parameter_store

    # Routine PE around and above the retired 0.45 "calibration" (0.4789 was
    # the pre-pickup approach outcome, i.e. a routine value) never requests
    # a boundary, no matter how large the magnitude.
    for routine_magnitude in (0.44, 0.4789, 0.701, 1.0):
        loop.set_external_learning_signals(
            {"prediction_error_magnitude": routine_magnitude}
        )
        assert store.external_boundary_requested() is False

    # A typed pickup/delivery milestone requests the boundary even when the
    # settled PE is the LOW value natural medium pickups actually produce.
    loop.set_external_learning_signals(
        {
            "prediction_error_magnitude": 0.32,
            "environment_milestone_boundary": 1.0,
        }
    )
    assert store.external_boundary_requested() is True

    # And the next quiet turn clears it: the request is turn-scoped.
    loop.set_external_learning_signals(
        {"prediction_error_magnitude": 0.32}
    )
    assert store.external_boundary_requested() is False


def test_p0_thresholds_are_frozen_in_the_config_contract() -> None:
    """plan 05 s2.1 -- written into schema/test before the first new result."""

    assert ECOLOGY_AUDIT_CODE_DELTA_THRESHOLD == 1e-8
    assert ECOLOGY_AUDIT_TURN_DELTA_THRESHOLD == 1e-4
    assert ECOLOGY_AUDIT_RETENTION_RATIO == 0.25
    assert ECOLOGY_AUDIT_BODY_PASS_RATIO == 0.8
    assert ECOLOGY_AUDIT_NEGATIVE_CONTROL_SWITCH_RATE_CEILING == 0.2
    assert ECOLOGY_AUDIT_TIMEOUT_CLOSURE_RATIO_CEILING == 1.0
    assert ECOLOGY_AUDIT_SWITCH_LOCALIZATION_WINDOW == 4
    assert ECOLOGY_AUDIT_BACKEND_PARITY_TOLERANCE == 1e-3
    assert ECOLOGY_AUDIT_SEGMENT_CREDIT_PARITY_TOLERANCE == 1e-6
    assert ECOLOGY_AUDIT_SIGN_REPEAT_COUNT == 3

    defaults = EcologyMechanismAuditConfig()
    assert defaults.code_delta_threshold == ECOLOGY_AUDIT_CODE_DELTA_THRESHOLD
    assert defaults.turn_delta_threshold == ECOLOGY_AUDIT_TURN_DELTA_THRESHOLD
    assert defaults.retention_ratio_threshold == ECOLOGY_AUDIT_RETENTION_RATIO
    assert defaults.body_pass_ratio == ECOLOGY_AUDIT_BODY_PASS_RATIO

    for field_name, value in (
        ("code_delta_threshold", 1e-6),
        ("turn_delta_threshold", 1e-9),
        ("retention_ratio_threshold", 0.05),
        ("body_pass_ratio", 0.5),
        ("negative_control_switch_rate_ceiling", 0.9),
        ("timeout_closure_ratio_ceiling", 2.0),
        ("switch_localization_window", 50),
        ("backend_parity_tolerance", 1.0),
        ("segment_credit_parity_tolerance", 1.0),
        ("sign_repeat_count", 1),
    ):
        with pytest.raises(ValueError):
            EcologyMechanismAuditConfig(**{field_name: value})


def test_gate_builder_passes_on_clean_evidence() -> None:
    gates = build_ecology_mechanism_gates(**_clean_gate_inputs(_audit_config()))

    assert tuple(gate.name for gate in gates) == _EXPECTED_GATE_NAMES
    assert all(gate.passed for gate in gates)


@pytest.mark.parametrize(
    ("override_key", "override_value", "expected_gate"),
    [
        (
            "rollback_episodes",
            ("learned:butter:near:episode:0",),
            "action_chain_no_rollback",
        ),
        ("no_optimize_stable", False, "no_optimize_policy_stable"),
        (
            "frozen_evaluations",
            None,
            "frozen_evaluation",
        ),
    ],
)
def test_gate_builder_blocks_on_defective_evidence(
    override_key: str,
    override_value: object,
    expected_gate: str,
) -> None:
    inputs = _clean_gate_inputs(_audit_config())
    if override_key == "frozen_evaluations":
        inputs[override_key] = (_clean_frozen(passed=False),)
    else:
        inputs[override_key] = override_value

    gates = build_ecology_mechanism_gates(**inputs)
    failed = tuple(gate.name for gate in gates if not gate.passed)

    assert failed == (expected_gate,)


def test_gate_builder_blocks_on_an_unimportable_backend_lane() -> None:
    """An unimportable backend fails BOTH questions, and says so separately."""

    inputs = _clean_gate_inputs(_audit_config())
    inputs["backend_parity"] = (
        EcologyBackendParityLane(
            lane="torch",
            measured=False,
            covered=False,
            not_measured_reason="torch is not importable in this process",
            not_covered_reason="torch is not importable in this process",
            declared_active_backends=(
                "temporal_ssl_backend",
                "internal_rl_backend",
            ),
            max_code_delta=None,
            max_turn_delta=None,
            max_step_delta=None,
            max_action_head_residual_delta=None,
            max_action_distribution_delta=None,
            abstract_actions_agree=False,
            within_tolerance=False,
        ),
    )

    gates = build_ecology_mechanism_gates(**inputs)

    assert tuple(gate.name for gate in gates if not gate.passed) == (
        "backend_lane_coverage",
        "backend_parity",
    )


def test_a_correct_backend_that_agrees_exactly_passes_both_gates() -> None:
    """The v3 admissible band was ``0 < delta <= 1e-3``; that was the defect.

    A torch lane that runs its declared backends and then agrees with the pure
    reference bit-for-bit is the IDEAL outcome.  v3 declared it "not
    evaluated" because it was indistinguishable from the reference and BLOCKed
    on it, which meant no correct backend could ever clear the gate.  Coverage
    now comes from the owner's own execution evidence, so exact agreement
    passes.
    """

    inputs = _clean_gate_inputs(_audit_config())
    inputs["backend_parity"] = tuple(
        _clean_lane(lane=lane, code_delta=0.0)
        for lane in ("pure", "runtime", "torch")
    )

    gates = build_ecology_mechanism_gates(**inputs)

    torch_lane = inputs["backend_parity"][2]
    assert torch_lane.max_code_delta == 0.0
    assert torch_lane.distinguishable_from_reference is False
    assert all(gate.passed for gate in gates)


def test_coverage_and_parity_fail_independently() -> None:
    """The two questions must be separately answerable.

    A lane that ran but disagrees fails parity only; a lane that agrees but
    never executed fails coverage only.  Fusing them is what made an exactly
    correct backend unable to pass.
    """

    disagreeing = _clean_gate_inputs(_audit_config())
    disagreeing["backend_parity"] = (
        _clean_lane(lane="pure", code_delta=0.0),
        _clean_lane(lane="runtime", code_delta=0.0),
        _clean_lane(lane="torch", code_delta=7.4e-3),
    )
    unexercised = _clean_gate_inputs(_audit_config())
    unexercised["backend_parity"] = (
        _clean_lane(lane="pure", code_delta=0.0),
        _clean_lane(lane="runtime", code_delta=0.0),
        EcologyBackendParityLane(
            lane="torch",
            measured=True,
            covered=False,
            not_measured_reason="",
            not_covered_reason=(
                "temporal_ssl_backend is active but the SSL trainer reported "
                "trained_steps=0"
            ),
            declared_active_backends=(
                "temporal_ssl_backend",
                "internal_rl_backend",
            ),
            max_code_delta=0.0,
            max_turn_delta=0.0,
            max_step_delta=0.0,
            max_action_head_residual_delta=0.0,
            max_action_distribution_delta=0.0,
            abstract_actions_agree=True,
            within_tolerance=True,
        ),
    )

    assert tuple(
        gate.name
        for gate in build_ecology_mechanism_gates(**disagreeing)
        if not gate.passed
    ) == ("backend_parity",)
    assert tuple(
        gate.name
        for gate in build_ecology_mechanism_gates(**unexercised)
        if not gate.passed
    ) == ("backend_lane_coverage",)


def test_action_distribution_disagreement_fails_parity() -> None:
    """plan 05:123 compares final code, ACTION DISTRIBUTION and turn.

    A lane whose code and turn match but which discovered a different action
    family distribution has not reproduced the controller's behaviour.
    """

    inputs = _clean_gate_inputs(_audit_config())
    inputs["backend_parity"] = (
        _clean_lane(lane="pure", code_delta=0.0),
        _clean_lane(
            lane="runtime",
            code_delta=0.0,
            action_distribution_delta=0.5,
        ),
        _clean_lane(lane="torch", code_delta=0.0),
    )

    gates = build_ecology_mechanism_gates(**inputs)

    assert tuple(gate.name for gate in gates if not gate.passed) == (
        "backend_parity",
    )


def test_backend_parity_exercise_budget_is_frozen_downward() -> None:
    """A shorter probe never reaches the first full optimization cycle."""

    assert ECOLOGY_AUDIT_BACKEND_PARITY_EXERCISE_STEPS == 6
    assert (
        EcologyMechanismAuditConfig().backend_parity_exercise_steps
        == ECOLOGY_AUDIT_BACKEND_PARITY_EXERCISE_STEPS
    )
    with pytest.raises(ValueError):
        EcologyMechanismAuditConfig(backend_parity_exercise_steps=1)
    # Raising it only gives the backends more opportunity to run, so it stays
    # open upward: this is a probe budget, not an acceptance threshold.
    assert (
        EcologyMechanismAuditConfig(
            backend_parity_exercise_steps=12
        ).backend_parity_exercise_steps
        == 12
    )


def test_declared_gaps_are_published_with_a_plan_reference() -> None:
    """A gap that lives only in a code comment is invisible to a reviewer."""

    assert ECOLOGY_AUDIT_DECLARED_GAPS
    for gap in ECOLOGY_AUDIT_DECLARED_GAPS:
        assert gap.plan_reference.startswith("research/ant/05_")
        assert gap.requirement.strip()
        assert len(gap.status.strip()) > 60
        assert gap.owner.strip()
    references = {gap.plan_reference for gap in ECOLOGY_AUDIT_DECLARED_GAPS}
    # The retention-baseline deviation and the SSL histogram provenance gap
    # remain declared until their respective owners close them.
    assert "research/ant/05_ecology_p0_p1_p2_plan.md:121" in references
    assert "research/ant/05_ecology_p0_p1_p2_plan.md:150" in references


def test_gate_builder_blocks_when_every_segment_closes_on_timeout() -> None:
    inputs = _clean_gate_inputs(_audit_config())
    inputs["temporal_switch"] = _clean_temporal(
        close_reasons=(("bounded-horizon", 4),),
        timeout_ratio=1.0,
    )

    gates = build_ecology_mechanism_gates(**inputs)

    assert tuple(gate.name for gate in gates if not gate.passed) == (
        "temporal_segment_closure",
    )


def test_gate_builder_blocks_on_a_chattering_negative_control() -> None:
    inputs = _clean_gate_inputs(_audit_config())
    inputs["temporal_switch"] = _clean_temporal(negative_switch_rate=0.5)

    gates = build_ecology_mechanism_gates(**inputs)

    assert tuple(gate.name for gate in gates if not gate.passed) == (
        "temporal_negative_control",
    )


async def test_checkpoint_action_probe_binds_every_body() -> None:
    config = EcologyCurriculumConfig(
        n_ants=2,
        temporal_latent_dim=4,
        stage_rounds=1,
        stage_episodes=1,
        mastery_min_episodes=1,
        validation_rounds=1,
        validation_seeds=(13,),
        heldout_rounds=1,
        heldout_seeds=(19,),
        seed=3,
    )
    runner = KernelColonyRunner(
        _world(
            config=config,
            stage=EcologyStage.COMPOSITE,
            seed=3,
            data_split=EcologyDataSplit.TRAIN,
            tier=EcologyTrainingTier.NEAR,
        ),
        base_config=_session_config(
            config=config,
            seed=3,
            session_id="test:p0:probe",
            optimize=True,
        ),
    )
    checkpoints = runner.export_learning_checkpoints(
        checkpoint_prefix="test:p0:probe",
        include_runtime_replay=False,
    )
    reports = await run_ecology_checkpoint_action_probes(
        temporal_latent_dim=4,
        seed=700_003,
        checkpoints=checkpoints,
    )

    assert tuple(item.body_id for item in reports) == (0, 1)
    assert all(item.probe_seed == 700_003 for item in reports)
    assert all(len(item.probes) == 4 for item in reports)
    assert all(
        {probe.kind for probe in item.probes}
        == {
            EcologyProbeKind.FOOD,
            EcologyProbeKind.OBSTACLE,
            EcologyProbeKind.HEAT,
            EcologyProbeKind.HOME,
        }
        for item in reports
    )
    assert all(item.policy_fingerprint for item in reports)
    assert all(
        len(probe.left_sense) == 19
        for item in reports
        for probe in item.probes
    )


async def test_p0_audit_runs_without_rollback_and_names_its_breakpoints() -> None:
    report = await run_ecology_mechanism_audit(_audit_config())

    assert report.schema_version == ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION
    assert report.schema_version.endswith(".v4")
    assert tuple(gate.name for gate in report.gates) == _EXPECTED_GATE_NAMES
    # The audit must NOT run the per-episode rollback guard: with it enabled
    # every gated snapshot is a restored cold checkpoint.
    assert report.action_probe_guard_enabled is False
    assert report.rollback_episodes == ()
    assert not any(
        item.action_chain_rollback_applied for item in report.segment_telemetry
    )
    assert report.initial_snapshot.gate_mode == "input-reachability"
    assert report.final_learned_snapshot.gate_mode == "post-training"
    assert report.action_chain_snapshots
    assert report.segment_telemetry
    assert len(report.frozen_evaluations) == len(
        ECOLOGY_AUDIT_FROZEN_EVALUATION_CASES
    )
    assert tuple(item.seed for item in report.frozen_evaluations) == (307, 101)
    assert all(
        item.replay_settlement_coverage == 1.0
        and item.replay_lineage_coverage == 1.0
        for item in report.frozen_evaluations
    )
    # The recorded, reproducible P0 outcome on this tree. This is deliberately
    # an equality assertion and not ``verdict in {"PASS", "BLOCK"}``: the
    # latter cannot fail, which is how the vacuous v1 PASS survived review.
    # Fixing any of these mechanisms must UPDATE this list (plan 05 s2.1
    # allows tightening by fixing the implementation, never relaxing).
    assert report.verdict == "BLOCK"
    assert report.diagnostic_breakpoints == _RECORDED_P0_BREAKPOINTS
    assert (report.verdict == "BLOCK") == bool(report.diagnostic_breakpoints)
    assert set(report.diagnostic_breakpoints) <= set(_EXPECTED_GATE_NAMES)
    assert report.to_dict()["schema_version"] == (
        ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION
    )
    # The driver serialises exactly this payload with ``allow_nan=False``, so
    # a report carrying a bare ``nan`` would abort the run with no artifact at
    # all. ``_backend_parity`` therefore publishes ``None`` for an unmeasured
    # lane rather than a non-standard NaN token.
    round_tripped = json.loads(
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    assert round_tripped["verdict"] == "BLOCK"
    assert round_tripped["schema_version"] == (
        ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION
    )
    # Distinct training CHECKPOINTS must reach the gate: the committed v1
    # artifact carried a single repeated fingerprint because every candidate
    # update had been rolled back. ``len(...) >= 1`` cannot detect a repeated
    # fingerprint, so the assertion is on distinctness.
    learned_snapshots = tuple(
        snapshot
        for snapshot in report.action_chain_snapshots
        if snapshot.arm == "learned" and snapshot.label != "final"
    )
    assert len(learned_snapshots) > 1
    learned_checkpoint_ids = {
        body.checkpoint_id
        for snapshot in learned_snapshots
        for body in snapshot.body_reports
    }
    assert len(learned_checkpoint_ids) == len(learned_snapshots) * len(
        learned_snapshots[0].body_reports
    )
    initial_checkpoint_ids = {
        body.checkpoint_id for body in report.initial_snapshot.body_reports
    }
    assert initial_checkpoint_ids
    assert not (learned_checkpoint_ids & initial_checkpoint_ids)
    # The policy fingerprint is the falsifiable half. At this budget the
    # optimizer does NOT move it -- which is precisely why
    # ``action_chain_no_rollback`` and ``action_head_update_applied`` are in
    # the recorded breakpoints. Assert the two statements agree rather than
    # asserting something that cannot fail.
    learned_fingerprints = {
        body.policy_fingerprint
        for snapshot in learned_snapshots
        for body in snapshot.body_reports
    }
    initial_fingerprints = {
        body.policy_fingerprint
        for body in report.initial_snapshot.body_reports
    }
    assert learned_fingerprints
    assert initial_fingerprints
    gated_checkpoint_is_trained = not (
        learned_fingerprints <= initial_fingerprints
    )
    assert gated_checkpoint_is_trained == (
        "action_chain_no_rollback" not in report.diagnostic_breakpoints
    )
    # plan 05:130's bisect trigger: the per-episode learned series gates
    # nothing, so the first failing episode has to be named explicitly.
    assert report.first_failing_learned_episode == next(
        (
            f"{snapshot.arm}:{snapshot.label}"
            for snapshot in learned_snapshots
            if not snapshot.passed
        ),
        "",
    )
    assert any(
        surface.name.startswith("action_chain_snapshots[learned")
        and not surface.gated
        for surface in report.diagnostic_surfaces
    )
    assert report.declared_gaps == ECOLOGY_AUDIT_DECLARED_GAPS
    # Every parity lane must publish either real numbers or an explicit
    # ``None``; a NaN would break the canonical artifact.
    for lane in report.backend_parity:
        assert lane.backend_execution is not None
        if lane.measured:
            assert lane.max_code_delta is not None
            assert lane.max_action_distribution_delta is not None
        else:
            assert lane.max_code_delta is None
    torch_lane = next(
        lane for lane in report.backend_parity if lane.lane == "torch"
    )
    # Both torch owners must execute.  This protects against silently restoring
    # the old trained_steps=0 state while still comparing deterministic output.
    assert torch_lane.backend_execution.internal_rl_torch_wrote_back
    assert torch_lane.backend_execution.ssl_trained_steps > 0
    assert "active" in torch_lane.backend_execution.ssl_torch_backends
    assert torch_lane.covered
    assert torch_lane.not_covered_reason == ""
    # The transition artifact still derives its histogram locally instead of
    # carrying the SSL owner's SwitchGateStats.  That provenance gap remains
    # declared, but it is not evidence that the backend failed to execute.
    ssl_gap = next(
        gap
        for gap in report.declared_gaps
        if gap.plan_reference.endswith(":150")
    )
    assert ssl_gap.gate_failing is False
    for trace in (
        report.temporal_switch.positive_control,
        report.temporal_switch.negative_control,
        report.temporal_switch.segment_credit_off_control,
    ):
        assert sum(trace.world_beta_histogram) == len(trace.ticks)
        assert sum(trace.self_beta_histogram) == len(trace.ticks)
        assert trace.switch_parameters_before is not None
        assert trace.switch_parameters_after is not None


# ---------------------------------------------------------------------------
# Driver contract (honest failure surface)
# ---------------------------------------------------------------------------


def _stub_tick(index: int) -> EcologyTemporalTick:
    return EcologyTemporalTick(
        tick=index,
        phase="cruise",
        world_beta=0.4,
        self_beta=0.3,
        world_beta_threshold=0.55,
        self_beta_threshold=0.55,
        world_beta_binary=0,
        self_beta_binary=0,
        track_switch_gates=(("world", 0.4), ("self", 0.3)),
        binary_switch=False,
        fast_prior_switch_pressure=0.0,
        prediction_error_switch_pressure=0.0,
        external_switch_pressure=0.0,
        steps_since_switch=index,
        open_segment_transitions=0,
        closed_segments=0,
        segment_closed_this_tick=False,
        last_segment_close_reason="",
        carrying_food=False,
        heat_harmful=False,
        picked_up=False,
        delivered=False,
    )


def _stub_body() -> EcologyCheckpointActionProbe:
    return EcologyCheckpointActionProbe(
        body_id=0,
        checkpoint_id="stub:cp:0",
        policy_fingerprint="stub-policy",
        temporal_learning_fingerprint="stub-temporal",
        probes=(),
    )


@dataclass(frozen=True)
class _StubReport:
    """Exactly the surface the driver reads, and nothing else."""

    verdict: str
    diagnostic_breakpoints: tuple[str, ...]
    schema_version: str = ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION

    @property
    def description(self) -> str:
        return f"{self.verdict}: stub"

    @property
    def gates(self) -> tuple[EcologyMechanismGate, ...]:
        return (
            EcologyMechanismGate(
                name="frozen_evaluation",
                passed=self.verdict == "PASS",
                observed="stub | observed",
                threshold="stub threshold",
            ),
        )

    @property
    def final_learned_snapshot(self) -> EcologyActionChainSnapshot:
        return EcologyActionChainSnapshot(
            arm="learned",
            label="final",
            stage="final",
            tier="final",
            episode_index=0,
            gate_mode="post-training",
            body_reports=(_stub_body(),),
            required_body_passes=1,
            passing_bodies=1,
            passed=True,
            failures=(),
        )

    @property
    def temporal_switch(self) -> EcologyTemporalSwitchAudit:
        return _clean_temporal(ticks=(_stub_tick(0), _stub_tick(1)))

    first_failing_learned_episode: str = ""
    diagnostic_surfaces: tuple[EcologyDiagnosticSurface, ...] = ()
    declared_gaps: tuple[EcologyDeclaredGap, ...] = field(
        default=ECOLOGY_AUDIT_DECLARED_GAPS
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION,
            "verdict": self.verdict,
        }


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "audit_ant_ecology_mechanisms_under_test",
        _DRIVER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load audit driver from {_DRIVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_driver_artifact_name_matches_the_schema_version() -> None:
    driver = _load_driver()

    assert driver.ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION == (
        ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION
    )
    assert driver._ARTIFACT_NAME == "ecology_mechanism_audit.v4.json"
    # The frozen v1 BLOCK/PASS artifact must never be overwritten by a run
    # that emits a different shape.
    assert driver._ARTIFACT_NAME != "ecology_mechanism_audit.v1.json"
    # plan 05 s2.3: the bundle lives under p0/<run-id>/, not at a fixed path
    # that the next run would silently destroy.
    assert driver._RESULTS_ROOT == Path(
        "research/ant/results/ecology_recovery/p0"
    )


def test_driver_run_id_is_new_per_run() -> None:
    driver = _load_driver()
    config = _audit_config()

    first = driver._default_run_id(config)
    other = driver._default_run_id(
        EcologyMechanismAuditConfig(
            n_ants=2,
            temporal_latent_dim=4,
            episode_rounds=1,
            episodes_per_stage=1,
            evaluation_rounds=3,
            seed=5,
        )
    )

    assert first.endswith(driver.stable_json_digest(asdict(config))[:8])
    assert "seed5" in first
    # A different configuration must not land in the same run-id directory.
    assert first.rsplit("-", 1)[-1] != other.rsplit("-", 1)[-1]


def test_default_audit_seed_namespaces_are_disjoint() -> None:
    training, layout = ecology_mechanism_audit_seed_schedule(
        EcologyMechanismAuditConfig()
    )

    assert training[:3] == (1_000_003, 1_000_104, 1_000_205)
    assert layout == (43, 101, 307)
    assert set(training).isdisjoint(layout)


def _driver_args(driver, tmp_path: Path, run_id: str):
    import argparse

    return argparse.Namespace(
        n_ants=1,
        temporal_latent_dim=4,
        episode_rounds=1,
        episodes_per_stage=1,
        evaluation_rounds=3,
        seed=5,
        device=None,
        run_id=run_id,
        results_root=Path("scratch"),
        overwrite=False,
    )


def _stub_provenance(driver, monkeypatch: pytest.MonkeyPatch) -> None:
    def _collect(**kwargs):
        return AntRunProvenance(
            git_sha="0" * 40,
            git_branch="stub",
            working_tree_dirty=True,
            python_version="3.13.5",
            platform="stub-platform",
            dependency_versions=(("numpy", "2.5.1"),),
            seed_schedule=tuple(kwargs["seeds"]),
            config_digest="stub-digest",
            model_fingerprint=kwargs["model_fingerprint"],
            training_seeds=tuple(kwargs["training_seeds"]),
            layout_seeds=tuple(kwargs["layout_seeds"]),
        )

    monkeypatch.setattr(driver, "collect_ant_provenance", _collect)


async def test_driver_preflights_seed_namespaces_before_running_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _load_driver()
    monkeypatch.setattr(driver, "_REPO_ROOT", tmp_path)
    audit_called = False

    def _reject(_config):
        raise RuntimeError("seed namespaces overlap")

    async def _audit(_config):
        nonlocal audit_called
        audit_called = True
        return _StubReport("PASS", ())

    monkeypatch.setattr(
        driver,
        "ecology_mechanism_audit_seed_schedule",
        _reject,
    )
    monkeypatch.setattr(driver, "run_ecology_mechanism_audit", _audit)

    with pytest.raises(RuntimeError, match="seed namespaces overlap"):
        await driver._run(_driver_args(driver, tmp_path, "seed-preflight"))
    assert audit_called is False


async def test_driver_emits_the_full_bundle_and_honours_the_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plan 05 s2.3: JSON + manifest + Markdown + raw trace, per run-id."""

    driver = _load_driver()
    monkeypatch.setattr(driver, "_REPO_ROOT", tmp_path)
    _stub_provenance(driver, monkeypatch)

    async def _blocked(_config):
        return _StubReport("BLOCK", ("frozen_evaluation",))

    async def _passed(_config):
        return _StubReport("PASS", ())

    monkeypatch.setattr(driver, "run_ecology_mechanism_audit", _blocked)
    assert await driver._run(_driver_args(driver, tmp_path, "run-block")) == 1

    monkeypatch.setattr(driver, "run_ecology_mechanism_audit", _passed)
    assert await driver._run(_driver_args(driver, tmp_path, "run-pass")) == 0

    run_directory = tmp_path / "scratch" / "run-block"
    artifact = run_directory / driver._ARTIFACT_NAME
    manifest = run_directory / "ecology_mechanism_audit.v4.manifest.json"
    summary = run_directory / "summary.md"
    raw_trace = run_directory / "raw" / "temporal_ticks.jsonl"
    assert artifact.is_file()
    assert manifest.is_file()
    assert summary.is_file()
    assert raw_trace.is_file()

    written = json.loads(artifact.read_text(encoding="utf-8"))
    assert written["schema_version"] == (
        ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION
    )
    assert written["run_id"] == "run-block"
    assert written["provenance"]["training_seeds"]
    assert written["provenance"]["layout_seeds"]
    assert written["provenance"]["model_fingerprint"]
    # The Markdown summary and raw trace are bound to the artifact by digest,
    # so a bundle whose sidecars were edited afterwards is detectable.
    bundle = {entry["path"]: entry for entry in written["bundle_files"]}
    assert len(bundle) == 2
    for path in (summary, raw_trace):
        relative = str(path.relative_to(tmp_path))
        assert bundle[relative]["size_bytes"] == path.stat().st_size

    # The manifest binds the artifact bytes and verifies against them.
    verify_ant_artifact_manifest(manifest_path=manifest, repo_root=tmp_path)

    lines = raw_trace.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6
    assert {json.loads(line)["trace"] for line in lines} == {
        "positive-control",
        "negative-control",
        "segment-credit-off",
    }
    assert "BLOCK" in summary.read_text(encoding="utf-8")


async def test_driver_refuses_to_destroy_an_existing_block_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plan 05 s2.1 -- a run must never overwrite an existing artifact.

    The refusal is raised BEFORE the audit runs, so a name collision costs no
    training budget.
    """

    driver = _load_driver()
    monkeypatch.setattr(driver, "_REPO_ROOT", tmp_path)
    _stub_provenance(driver, monkeypatch)
    runs = 0

    async def _blocked(_config):
        nonlocal runs
        runs += 1
        return _StubReport("BLOCK", ("frozen_evaluation",))

    monkeypatch.setattr(driver, "run_ecology_mechanism_audit", _blocked)
    assert await driver._run(_driver_args(driver, tmp_path, "collide")) == 1
    assert runs == 1

    with pytest.raises(AntArtifactExistsError) as excinfo:
        await driver._run(_driver_args(driver, tmp_path, "collide"))

    assert "BLOCK" in str(excinfo.value)
    # The budget was never spent on the refused run.
    assert runs == 1
