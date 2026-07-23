"""P0 mechanism-audit regression tests."""

from __future__ import annotations

from volvence_ant.experiments.ecology_mechanism_audit import (
    ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION,
    EcologyMechanismAuditConfig,
    run_ecology_mechanism_audit,
)
from volvence_ant.experiments.ecology_probe import (
    EcologyProbeKind,
    run_ecology_checkpoint_action_probes,
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


async def test_p0_audit_reports_honest_gates_and_owner_differences() -> None:
    report = await run_ecology_mechanism_audit(
        EcologyMechanismAuditConfig(
            n_ants=1,
            temporal_latent_dim=4,
            episode_rounds=1,
            episodes_per_stage=1,
            evaluation_rounds=3,
            seed=5,
        )
    )

    assert report.schema_version == ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION
    assert report.verdict in {"PASS", "BLOCK"}
    assert tuple(gate.name for gate in report.gates) == (
        "action_chain",
        "no_optimize_policy_stable",
        "temporal_switch_and_closure",
        "frozen_evaluation",
    )
    assert report.action_chain_snapshots
    assert report.segment_telemetry
    assert len(report.frozen_evaluations) == 2
    assert all(
        item.replay_settlement_coverage == 1.0
        and item.replay_lineage_coverage == 1.0
        for item in report.frozen_evaluations
    )
    assert report.to_dict()["schema_version"] == (
        ECOLOGY_MECHANISM_AUDIT_SCHEMA_VERSION
    )
