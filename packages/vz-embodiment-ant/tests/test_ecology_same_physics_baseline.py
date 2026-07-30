"""Frozen contract tests for the v31 same-physics preregistration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from volvence_zero.runtime import WiringLevel

from volvence_ant.experiments.ecology_curriculum import _session_config
from volvence_ant.experiments.ecology_same_physics_baseline import (
    ECOLOGY_SAME_PHYSICS_BASELINE_SCHEMA_VERSION,
    EcologySamePhysicsBaselinePacketError,
    build_ecology_same_physics_baseline_packet,
    formal_same_physics_baseline_config,
    validate_ecology_same_physics_baseline_packet,
)
from volvence_ant.experiments.ecology_p1 import _curriculum_config
from volvence_ant.experiments.ecology_same_physics_run import (
    evaluate_same_physics_station1,
)


_ROOT = Path(__file__).resolve().parents[3]


def _packet() -> dict:
    return build_ecology_same_physics_baseline_packet(
        repo_root=_ROOT,
        seed=0,
    )


def _clone(value: dict) -> dict:
    return json.loads(json.dumps(value))


def test_packet_binds_same_schedule_and_exactly_one_causal_difference() -> None:
    packet = _packet()

    assert (
        packet["schema_version"]
        == ECOLOGY_SAME_PHYSICS_BASELINE_SCHEMA_VERSION
    )
    assert packet["status"] == "PREREGISTERED"
    assert packet["schedule"]["full_episode_count"] == 55
    assert len(packet["schedule"]["rows"]) == 55
    assert packet["schedule"]["blocks"][0]["episode_start_inclusive"] == 0
    assert packet["schedule"]["blocks"][-1]["episode_end_exclusive"] == 55
    assert packet["arms"]["allowed_differences"] == [
        {
            "field": "environment_milestone_temporal_switch",
            "candidate": "active",
            "control": "disabled",
        }
    ]
    assert packet["arms"]["shared_initial_checkpoint"] is True
    assert packet["historical_baselines"]["decision_use"] == "EXCLUDED"
    assert (
        packet["execution_contract"]["code_tree_binding"]["file_count"] > 100
    )
    assert len(
        packet["execution_contract"]["code_tree_binding"]["sha256"]
    ) == 64


def test_packet_freezes_decisions_before_any_result_is_read() -> None:
    packet = _packet()
    thresholds = packet["thresholds"]

    assert thresholds["station1"]["candidate_aggregate_pickup_ratio_min"] == 0.8
    assert thresholds["station1"]["deliveries_role"] == (
        "DESCRIPTIVE_SPARSE_OBSERVATION"
    )
    assert (
        thresholds["station2"][
            "candidate_medium_deliveries_must_exceed_control"
        ]
        is True
    )
    assert packet["decision_protocol"]["no_posthoc_threshold_changes"] is True
    assert (
        packet["execution_contract"]["old_v31_journal_resume_forbidden"]
        is True
    )


def test_curriculum_threads_the_only_baseline_wiring_lever() -> None:
    curriculum = _curriculum_config(
        formal_same_physics_baseline_config(seed=0)
    )
    active = _session_config(
        config=curriculum,
        seed=0,
        session_id="test:same-physics:active",
        optimize=True,
    )
    control = _session_config(
        config=curriculum,
        seed=0,
        session_id="test:same-physics:control",
        optimize=True,
        environment_milestone_switch_enabled=False,
    )

    assert (
        active.rollout_config.environment_milestone_temporal_switch
        is WiringLevel.ACTIVE
    )
    assert (
        control.rollout_config.environment_milestone_temporal_switch
        is WiringLevel.DISABLED
    )
    assert {
        key
        for key, value in active.rollout_config.__dict__.items()
        if value != control.rollout_config.__dict__[key]
    } == {"environment_milestone_temporal_switch"}


def test_validator_rejects_threshold_or_causal_arm_tampering() -> None:
    threshold_tamper = _clone(_packet())
    threshold_tamper["thresholds"]["station1"][
        "candidate_aggregate_pickup_ratio_min"
    ] = 0.5
    with pytest.raises(
        EcologySamePhysicsBaselinePacketError,
        match="differs from the frozen executable contract",
    ):
        validate_ecology_same_physics_baseline_packet(
            threshold_tamper,
            repo_root=_ROOT,
        )

    arm_tamper = _clone(_packet())
    arm_tamper["arms"]["control"]["rollout_config"][
        "prediction_error_temporal_switch"
    ] = "disabled"
    with pytest.raises(
        EcologySamePhysicsBaselinePacketError,
        match="differs from the frozen executable contract",
    ):
        validate_ecology_same_physics_baseline_packet(
            arm_tamper,
            repo_root=_ROOT,
        )


def test_validator_rejects_bound_source_drift() -> None:
    packet = _clone(_packet())
    packet["execution_contract"]["source_bindings"][0]["sha256"] = "0" * 64

    with pytest.raises(
        EcologySamePhysicsBaselinePacketError,
        match="differs from the frozen executable contract",
    ):
        validate_ecology_same_physics_baseline_packet(
            packet,
            repo_root=_ROOT,
        )


def test_station1_verdict_uses_preregistered_pickup_and_structure_gates() -> None:
    packet = _packet()
    control = [
        {"pickups": 1, "deliveries": 100}
        for _ in range(20)
    ]
    candidate = [
        {"pickups": 1, "deliveries": 0}
        for _ in range(20)
    ]
    structural = [{"passed": True} for _ in range(8)]

    result = evaluate_same_physics_station1(
        packet=packet,
        candidate_reports=candidate,
        control_reports=control,
        structural_lanes=structural,
    )

    assert result["verdict"] == "GO"
    assert result["candidate_to_control_pickup_ratio"] == 1.0
    assert result["next_episode_authorized"] == 20

    regressed = [dict(row) for row in candidate]
    for index in range(5):
        regressed[index]["pickups"] = 0
    blocked = evaluate_same_physics_station1(
        packet=packet,
        candidate_reports=regressed,
        control_reports=control,
        structural_lanes=structural,
    )
    assert blocked["verdict"] == "BLOCK"
    assert blocked["gates"]["no_candidate_zero_block"] is False
    assert blocked["next_episode_authorized"] is None
