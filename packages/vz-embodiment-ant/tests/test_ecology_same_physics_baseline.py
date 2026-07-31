"""Frozen contract tests for the v31 same-physics preregistration."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from volvence_zero.runtime import WiringLevel

from volvence_ant.experiments import ecology_same_physics_run
from volvence_ant.experiments.ecology_curriculum import (
    EcologyTrainingEpisodeReport,
    _session_config,
)
from volvence_ant.experiments.ecology_same_physics_baseline import (
    ECOLOGY_SAME_PHYSICS_BASELINE_SCHEMA_VERSION,
    EcologySamePhysicsBaselinePacketError,
    build_ecology_same_physics_baseline_packet,
    formal_same_physics_baseline_config,
    validate_ecology_same_physics_baseline_packet,
)
from volvence_ant.experiments.ecology_same_physics_review import (
    ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_STATION1_PACKET_SCHEMA_VERSION,
    EcologySamePhysicsAlignmentReviewPacketError,
    build_ecology_same_physics_alignment_review_packet,
    validate_ecology_same_physics_alignment_review_packet,
)
from volvence_ant.experiments.ecology_p1 import _curriculum_config
from volvence_ant.experiments.ecology_same_physics_run import (
    evaluate_same_physics_alignment_review,
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


def _review_packet(station1_packet: dict) -> dict:
    return build_ecology_same_physics_alignment_review_packet(
        repo_root=_ROOT,
        station1_packet=station1_packet,
        station1_preregistration_sha256=hashlib.sha256(
            b"frozen-station1-preregistration"
        ).hexdigest(),
    )


def _historical_review_station1_packet() -> dict:
    packet = _clone(_packet())
    packet["schema_version"] = (
        ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_STATION1_PACKET_SCHEMA_VERSION
    )
    packet["thresholds"]["station1"][
        "food_alignment_review_authorized"
    ] = True
    return packet


def test_packet_binds_same_schedule_and_exactly_one_causal_difference() -> None:
    packet = _packet()

    assert (
        packet["schema_version"]
        == ECOLOGY_SAME_PHYSICS_BASELINE_SCHEMA_VERSION
    )
    assert packet["status"] == "PREREGISTERED"
    assert packet["experiment_generation"] == "station1-v4"
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
    for arm in ("candidate", "control"):
        rollout = packet["arms"][arm]["rollout_config"]
        assert (
            rollout[
                "internal_rl_causal_action_head_formation_protection"
            ]
            == "active"
        )
        assert (
            rollout[
                "internal_rl_causal_action_head_formation_max_update_steps"
            ]
            == 160
        )
        assert (
            rollout[
                "internal_rl_causal_action_head_formation_conflict_scale"
            ]
            == 0.25
        )
    assert (
        packet["arms"]["candidate"]["rollout_config"][
            "temporal_post_switch_min_dwell"
        ]
        == "active"
    )
    assert (
        packet["thresholds"]["station1"][
            "post_switch_min_dwell_actions"
        ]
        == 4
    )
    assert packet["historical_baselines"]["decision_use"] == "EXCLUDED"
    assert packet["reopening_basis"]["threshold_change"] == "NONE"
    assert packet["reopening_basis"]["seed_only_rerun"] is False
    assert packet["reopening_basis"]["l1b_precheck"]["status"] == (
        "PRECHECK_PASS"
    )
    assert packet["authorization"] == {
        "station1_run_authorized": True,
        "station1_max_episode_end_exclusive": 20,
        "alignment_review_authorized": False,
        "station2_authorized_before_station1_go": False,
        "p1_authorized_before_station2_go": False,
        "p2_authorized_before_station2_go": False,
    }
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
        thresholds["station1"]["food_alignment_review_authorized"]
        is False
    )
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


def test_validator_can_audit_old_packet_semantics_without_current_source() -> None:
    packet = _clone(_packet())
    packet["execution_contract"]["source_bindings"][0]["sha256"] = "0" * 64
    packet["execution_contract"]["code_tree_binding"]["sha256"] = "1" * 64

    validate_ecology_same_physics_baseline_packet(
        packet,
        repo_root=_ROOT,
        check_source_bindings=False,
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
    aligned = [
        {
            "input_reachable": True,
            "action_sensitive": True,
            "target_aligned": True,
        }
        for _ in range(4)
    ]

    result = evaluate_same_physics_station1(
        packet=packet,
        candidate_reports=candidate,
        control_reports=control,
        structural_lanes=structural,
        food_alignment_rows=aligned,
    )

    assert result["verdict"] == "GO"
    assert result["candidate_to_control_pickup_ratio"] == 1.0
    assert result["next_episode_authorized"] == 20
    assert result["food_alignment_status"] == "DIRECT_STATION2"

    regressed = [dict(row) for row in candidate]
    for index in range(5):
        regressed[index]["pickups"] = 0
    blocked = evaluate_same_physics_station1(
        packet=packet,
        candidate_reports=regressed,
        control_reports=control,
        structural_lanes=structural,
        food_alignment_rows=aligned,
    )
    assert blocked["verdict"] == "BLOCK"
    assert blocked["gates"]["no_candidate_zero_block"] is False
    assert blocked["next_episode_authorized"] is None


def test_station1_v4_blocks_alignment_failure_without_review() -> None:
    packet = _packet()
    reports = [
        {"pickups": 1, "deliveries": 0}
        for _ in range(20)
    ]
    structural = [{"passed": True} for _ in range(8)]
    unaligned = [
        {
            "input_reachable": True,
            "action_sensitive": False,
            "target_aligned": False,
        }
        for _ in range(4)
    ]

    result = evaluate_same_physics_station1(
        packet=packet,
        candidate_reports=reports,
        control_reports=reports,
        structural_lanes=structural,
        food_alignment_rows=unaligned,
    )

    assert result["verdict"] == "BLOCK"
    assert result["next_episode_authorized"] is None
    assert result["alignment_review_authorized"] is False
    assert result["food_alignment_status"] == "BLOCKED_BY_ALIGNMENT"
    assert result["gates"]["food_alignment_4_of_4"] is False


def test_alignment_review_replays_frozen_rows_and_reprobes_once() -> None:
    packet = _historical_review_station1_packet()
    station1 = {
        "verdict": "GO",
        "alignment_review_authorized": True,
        "next_episode_authorized": None,
        "aligned_food_bodies": 1,
    }
    review_rows = packet["schedule"]["rows"][:5]
    aligned = [
        {
            "input_reachable": True,
            "action_sensitive": True,
            "target_aligned": True,
        }
        for _ in range(4)
    ]

    passed = evaluate_same_physics_alignment_review(
        packet=packet,
        station1_evaluation=station1,
        review_schedule_rows=review_rows,
        food_alignment_rows=aligned,
    )

    assert passed["verdict"] == "GO"
    assert passed["next_episode_authorized"] == 20
    assert passed["pre_review_aligned_food_bodies"] == 1
    assert passed["post_review_aligned_food_bodies"] == 4

    failed = evaluate_same_physics_alignment_review(
        packet=packet,
        station1_evaluation=station1,
        review_schedule_rows=review_rows,
        food_alignment_rows=[
            {
                "input_reachable": True,
                "action_sensitive": True,
                "target_aligned": body_id < 3,
            }
            for body_id in range(4)
        ],
    )
    assert failed["verdict"] == "BLOCK"
    assert failed["next_episode_authorized"] is None

    tampered_rows = _clone({"rows": review_rows})["rows"]
    tampered_rows[0]["seed"] += 1
    with pytest.raises(ValueError, match="without seed or config changes"):
        evaluate_same_physics_alignment_review(
            packet=packet,
            station1_evaluation=station1,
            review_schedule_rows=tampered_rows,
            food_alignment_rows=aligned,
        )


def test_alignment_review_packet_binds_source_schedule_and_authority() -> None:
    station1_packet = _historical_review_station1_packet()
    station1_sha256 = hashlib.sha256(
        b"frozen-station1-preregistration"
    ).hexdigest()
    review_packet = _review_packet(station1_packet)

    assert review_packet["review_schedule"]["rows"] == (
        station1_packet["schedule"]["rows"][:5]
    )
    assert review_packet["review_schedule"]["episode_count"] == 5
    assert review_packet["probe"]["attempt_count"] == 1
    assert review_packet["authorization"][
        "additional_training_after_failure_forbidden"
    ]
    validate_ecology_same_physics_alignment_review_packet(
        review_packet,
        repo_root=_ROOT,
        station1_packet=station1_packet,
        station1_preregistration_sha256=station1_sha256,
    )

    tampered = _clone(review_packet)
    tampered["review_schedule"]["rows"][0]["seed"] += 1
    with pytest.raises(
        EcologySamePhysicsAlignmentReviewPacketError,
        match="differs from the frozen executable contract",
    ):
        validate_ecology_same_physics_alignment_review_packet(
            tampered,
            repo_root=_ROOT,
            station1_packet=station1_packet,
            station1_preregistration_sha256=station1_sha256,
        )


def test_alignment_review_runner_commits_five_rows_then_reprobes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    packet = _historical_review_station1_packet()
    station1_progress = tmp_path / "station1"
    review_progress = tmp_path / "review"
    station1_progress.mkdir()
    review_state: dict | None = None
    saved_reports: list[dict] = []

    def fake_load_progress(*, progress_dir: Path, arm: str, **_: object):
        if progress_dir == station1_progress:
            assert arm == "learned"
            return {
                "completed_training_episodes": 20,
                "checkpoint_sha256": "a" * 64,
            }
        return review_state

    def fake_save_progress(
        *,
        completed_training_episodes: int,
        training_complete: bool,
        **_: object,
    ) -> None:
        nonlocal review_state
        review_state = {
            "completed_training_episodes": completed_training_episodes,
            "training_complete": training_complete,
            "checkpoint_sha256": (
                f"{completed_training_episodes:064x}"
            ),
        }

    async def fake_train_arm(*, schedule, episode_callback, **_: object):
        for schedule_index, plan in enumerate(schedule):
            report = EcologyTrainingEpisodeReport(
                arm="learned_alignment_review",
                plan=plan,
                pickups=1,
                deliveries=0,
                obstacle_contacts=0,
                heat_entries=0,
                heat_escapes=0,
                nonzero_ecology_payoffs=1,
                activated_sense_channels=(),
                minimum_food_distance=0.0,
                minimum_obstacle_distance=None,
                minimum_heat_distance=None,
                switch_count=0,
                mean_persistence_steps=0.0,
                closed_segment_count=0,
                longest_segment_length=0,
                policy_fingerprints_before=(),
                policy_fingerprints_after=(),
                memory_entries_evicted=0,
                rounds=49,
                milestone_round_budget=49,
                milestone_samplable=True,
            )
            episode_callback(
                schedule_index,
                object(),
                ("checkpoint",),
                report,
            )
        return ("checkpoint",), (), schedule, (), ()

    async def fake_action_probes(**_: object):
        return ()

    monkeypatch.setattr(
        ecology_same_physics_run,
        "_load_arm_progress",
        fake_load_progress,
    )
    monkeypatch.setattr(
        ecology_same_physics_run,
        "_save_arm_progress",
        fake_save_progress,
    )
    monkeypatch.setattr(
        ecology_same_physics_run,
        "_load_arm_reports",
        lambda **_: saved_reports,
    )
    monkeypatch.setattr(
        ecology_same_physics_run,
        "_save_arm_reports",
        lambda **_: None,
    )
    monkeypatch.setattr(
        ecology_same_physics_run,
        "_read_progress_archive",
        lambda **_: (b"archive",),
    )
    monkeypatch.setattr(
        ecology_same_physics_run,
        "_hydrate_progress_checkpoints",
        lambda **_: ("checkpoint",),
    )
    monkeypatch.setattr(
        ecology_same_physics_run,
        "_bind_alignment_review_progress",
        lambda **_: None,
    )
    monkeypatch.setattr(
        ecology_same_physics_run,
        "_train_arm",
        fake_train_arm,
    )
    monkeypatch.setattr(
        ecology_same_physics_run,
        "run_ecology_checkpoint_action_probes",
        fake_action_probes,
    )
    monkeypatch.setattr(
        ecology_same_physics_run,
        "_food_alignment_rows",
        lambda _: [
            {
                "input_reachable": True,
                "action_sensitive": True,
                "target_aligned": True,
            }
            for _ in range(4)
        ],
    )

    result = asyncio.run(
        ecology_same_physics_run.run_ecology_same_physics_alignment_review(
            packet=packet,
            preregistration_sha256="b" * 64,
            review_preregistration_sha256="c" * 64,
            station1_evaluation={
                "verdict": "GO",
                "alignment_review_authorized": True,
                "next_episode_authorized": None,
                "aligned_food_bodies": 0,
            },
            station1_progress_dir=station1_progress,
            review_progress_dir=review_progress,
        )
    )

    assert result["verdict"] == "GO"
    assert result["next_episode_authorized"] == 20
    assert len(result["review_episode_reports"]) == 5
    assert [row["plan"]["seed"] for row in saved_reports] == [
        10000,
        10101,
        10202,
        10303,
        10404,
    ]
