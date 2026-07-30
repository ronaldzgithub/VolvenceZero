"""Execute the preregistered same-physics v31 station-1 matched control."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from volvence_ant.experiments.ecology_curriculum import (
    EcologyDataSplit,
    EcologyStage,
    EcologyTrainingTier,
    _session_config,
    _train_arm,
    _world,
)
from volvence_ant.experiments.ecology_p1 import (
    EcologyP1Config,
    EcologyP1ProgressPaused,
    _atomic_write,
    _curriculum_config,
    _fixed_schedule,
    _hydrate_progress_checkpoints,
    _json_ready,
    _load_arm_progress,
    _read_progress_archive,
    _save_arm_progress,
    _schedule_digest,
    _stable_json_bytes,
)
from volvence_ant.experiments.ecology_probe import (
    run_ecology_checkpoint_post_pickup_uturn_probes,
)
from volvence_ant.experiments.ecology_same_physics_baseline import (
    ECOLOGY_SAME_PHYSICS_CANDIDATE_ARM,
    ECOLOGY_SAME_PHYSICS_CONTROL_ARM,
    ECOLOGY_SAME_PHYSICS_STATION1_EPISODES,
)
from volvence_ant.runtime import AntLearningCheckpoint, KernelColonyRunner


ECOLOGY_SAME_PHYSICS_PROGRESS_SCHEMA_VERSION = (
    "digital-ant-ecology-same-physics-progress.v1"
)
ECOLOGY_SAME_PHYSICS_STATION1_REPORT_SCHEMA_VERSION = (
    "digital-ant-ecology-same-physics-station1.v1"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _arm_report_path(progress_dir: Path, arm: str) -> Path:
    return progress_dir / f"{arm}.station-reports.json"


def _load_arm_reports(
    *,
    progress_dir: Path,
    arm: str,
    preregistration_sha256: str,
    completed: int,
) -> list[dict[str, Any]]:
    path = _arm_report_path(progress_dir, arm)
    if not path.exists():
        if completed:
            raise ValueError(
                f"same-physics report journal is missing for {arm}"
            )
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": ECOLOGY_SAME_PHYSICS_PROGRESS_SCHEMA_VERSION,
        "arm": arm,
        "preregistration_sha256": preregistration_sha256,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"same-physics report journal mismatch for {arm}: "
                f"field={field}, expected={value!r}, "
                f"actual={payload.get(field)!r}"
            )
    rows = payload.get("episodes")
    if not isinstance(rows, list):
        raise ValueError(
            f"same-physics report journal episodes must be a list for {arm}"
        )
    if len(rows) < completed:
        raise ValueError(
            f"same-physics report journal trails checkpoint for {arm}: "
            f"reports={len(rows)}, completed={completed}"
        )
    # Report bytes are committed before the checkpoint pointer. A process may
    # die in between; only the prefix backed by the owner checkpoint survives.
    return [dict(item) for item in rows[:completed]]


def _save_arm_reports(
    *,
    progress_dir: Path,
    arm: str,
    preregistration_sha256: str,
    reports: list[dict[str, Any]],
) -> None:
    _atomic_write(
        _arm_report_path(progress_dir, arm),
        _stable_json_bytes(
            {
                "schema_version": (
                    ECOLOGY_SAME_PHYSICS_PROGRESS_SCHEMA_VERSION
                ),
                "arm": arm,
                "preregistration_sha256": preregistration_sha256,
                "episodes": reports,
            }
        ),
    )


def _block_metrics(
    reports: list[dict[str, Any]],
    *,
    start: int,
    stop: int,
) -> dict[str, int]:
    rows = reports[start:stop]
    if len(rows) != stop - start:
        raise ValueError(
            f"station report block is incomplete: [{start}, {stop})"
        )
    return {
        "pickups": sum(int(row["pickups"]) for row in rows),
        "deliveries": sum(int(row["deliveries"]) for row in rows),
    }


def evaluate_same_physics_station1(
    *,
    packet: Mapping[str, Any],
    candidate_reports: list[dict[str, Any]],
    control_reports: list[dict[str, Any]],
    structural_lanes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply only the thresholds frozen in the preregistration packet."""

    thresholds = packet["thresholds"]["station1"]
    station_blocks = packet["schedule"]["blocks"][:4]
    candidate_blocks: dict[str, dict[str, int]] = {}
    control_blocks: dict[str, dict[str, int]] = {}
    for block in station_blocks:
        name = str(block["name"])
        start = int(block["episode_start_inclusive"])
        stop = int(block["episode_end_exclusive"])
        candidate_blocks[name] = _block_metrics(
            candidate_reports,
            start=start,
            stop=stop,
        )
        control_blocks[name] = _block_metrics(
            control_reports,
            start=start,
            stop=stop,
        )
    candidate_total = sum(
        block["pickups"] for block in candidate_blocks.values()
    )
    control_total = sum(
        block["pickups"] for block in control_blocks.values()
    )
    pickup_ratio = (
        candidate_total / control_total
        if control_total
        else None
    )
    minimum_control = int(
        thresholds["minimum_control_pickups_per_physical_block"]
    )
    control_signal_gate = all(
        block["pickups"] >= minimum_control
        for block in control_blocks.values()
    )
    pickup_noninferiority_gate = (
        pickup_ratio is not None
        and pickup_ratio
        >= float(thresholds["candidate_aggregate_pickup_ratio_min"])
    )
    zero_block_gate = all(
        candidate_blocks[name]["pickups"] > 0
        or control_blocks[name]["pickups"] == 0
        for name in candidate_blocks
    )
    structural_gate = (
        bool(structural_lanes)
        and sum(bool(row["passed"]) for row in structural_lanes)
        / len(structural_lanes)
        >= float(thresholds["candidate_post_pickup_switch_rate_min"])
    )
    gates = {
        "control_signal": control_signal_gate,
        "pickup_noninferiority": pickup_noninferiority_gate,
        "no_candidate_zero_block": zero_block_gate,
        "typed_milestone_structure": structural_gate,
    }
    return {
        "candidate_blocks": candidate_blocks,
        "control_blocks": control_blocks,
        "candidate_total_pickups": candidate_total,
        "control_total_pickups": control_total,
        "candidate_to_control_pickup_ratio": pickup_ratio,
        "structural_lanes": structural_lanes,
        "gates": gates,
        "verdict": "GO" if all(gates.values()) else "BLOCK",
        "next_episode_authorized": (
            ECOLOGY_SAME_PHYSICS_STATION1_EPISODES
            if all(gates.values())
            else None
        ),
    }


def _structural_lane_rows(probes: tuple[Any, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probe in probes:
        for lane in probe.lanes:
            passed = (
                lane.picked_up
                and lane.post_pickup_switch_observed
                and lane.first_post_pickup_switch_step is not None
                and lane.first_post_pickup_switch_step <= 2
                and lane.post_switch_family_survival_actions >= 3
                and lane.policy_fingerprint_stable
                and lane.temporal_learning_fingerprint_stable
            )
            rows.append(
                {
                    "body_id": probe.body_id,
                    "side": lane.side,
                    "picked_up": lane.picked_up,
                    "post_pickup_switch_observed": (
                        lane.post_pickup_switch_observed
                    ),
                    "first_post_pickup_switch_step": (
                        lane.first_post_pickup_switch_step
                    ),
                    "post_switch_family_survival_actions": (
                        lane.post_switch_family_survival_actions
                    ),
                    "policy_fingerprint_stable": (
                        lane.policy_fingerprint_stable
                    ),
                    "temporal_learning_fingerprint_stable": (
                        lane.temporal_learning_fingerprint_stable
                    ),
                    "passed": passed,
                }
            )
    return rows


async def run_ecology_same_physics_station1(
    *,
    packet: Mapping[str, Any],
    preregistration_sha256: str,
    progress_dir: Path,
    max_new_work_items: int | None = None,
) -> dict[str, Any]:
    """Run/resume both matched arms through episode 19 and grade station 1."""

    if max_new_work_items is not None and max_new_work_items < 1:
        raise ValueError("max_new_work_items must be positive")
    config = EcologyP1Config(**dict(packet["formal_config"]))
    curriculum = _curriculum_config(config)
    schedule = _fixed_schedule(config)
    schedule_sha256 = _schedule_digest(schedule)
    if schedule_sha256 != packet["schedule"]["full_sha256"]:
        raise ValueError("same-physics schedule drifted after preregistration")
    station_schedule = schedule[:ECOLOGY_SAME_PHYSICS_STATION1_EPISODES]
    progress_dir.mkdir(parents=True, exist_ok=True)
    bootstrap = KernelColonyRunner(
        _world(
            config=curriculum,
            stage=EcologyStage.COMPOSITE,
            seed=config.seed,
            data_split=EcologyDataSplit.TRAIN,
            tier=EcologyTrainingTier.NEAR,
        ),
        base_config=_session_config(
            config=curriculum,
            seed=config.seed,
            session_id="ecology:same-physics:shared-initial",
            optimize=True,
        ),
    )
    initial = bootstrap.export_learning_checkpoints(
        checkpoint_prefix="ecology:same-physics:shared-initial",
        include_runtime_replay=False,
    )
    completed_work_items = 0
    final_checkpoints: dict[str, tuple[AntLearningCheckpoint, ...]] = {}
    reports_by_arm: dict[str, list[dict[str, Any]]] = {}

    def make_save_episode(
        *,
        active_arm: str,
        active_reports: list[dict[str, Any]],
    ):
        def save_episode(
            schedule_index: int,
            runner: KernelColonyRunner,
            _checkpoints: tuple[AntLearningCheckpoint, ...],
            report: Any,
        ) -> None:
            nonlocal completed_work_items
            if schedule_index != len(active_reports):
                raise ValueError(
                    f"same-physics report order mismatch for {active_arm}: "
                    f"index={schedule_index}, reports={len(active_reports)}"
                )
            active_reports.append(_json_ready(asdict(report)))
            # Commit evidence first. If the process dies before the owner
            # checkpoint pointer advances, resume truncates this extra suffix.
            _save_arm_reports(
                progress_dir=progress_dir,
                arm=active_arm,
                preregistration_sha256=preregistration_sha256,
                reports=active_reports,
            )
            _save_arm_progress(
                progress_dir=progress_dir,
                arm=active_arm,
                config=config,
                schedule_sha256=schedule_sha256,
                completed_training_episodes=schedule_index + 1,
                runner=runner,
                # This checkpoint is intentionally resumable by the full P1
                # runner after station1 GO; 20/55 is not training-complete.
                training_complete=False,
                last_episode_report=report,
            )
            completed_work_items += 1
            if (
                max_new_work_items is not None
                and completed_work_items >= max_new_work_items
            ):
                raise EcologyP1ProgressPaused(
                    completed_work_items=completed_work_items
                )

        return save_episode

    for arm, milestone_enabled in (
        (ECOLOGY_SAME_PHYSICS_CONTROL_ARM, False),
        (ECOLOGY_SAME_PHYSICS_CANDIDATE_ARM, True),
    ):
        state = _load_arm_progress(
            progress_dir=progress_dir,
            arm=arm,
            config=config,
            schedule_sha256=schedule_sha256,
        )
        completed = (
            int(state["completed_training_episodes"])
            if state is not None
            else 0
        )
        if completed > ECOLOGY_SAME_PHYSICS_STATION1_EPISODES:
            raise ValueError(
                f"same-physics arm {arm} exceeds station1 boundary"
            )
        reports = _load_arm_reports(
            progress_dir=progress_dir,
            arm=arm,
            preregistration_sha256=preregistration_sha256,
            completed=completed,
        )
        checkpoints = initial
        if state is not None:
            checkpoints = _hydrate_progress_checkpoints(
                config=config,
                curriculum=curriculum,
                archives=_read_progress_archive(
                    progress_dir=progress_dir,
                    state=state,
                    config=config,
                ),
                arm=arm,
            )

        if completed < ECOLOGY_SAME_PHYSICS_STATION1_EPISODES:
            checkpoints, _, _, _, _ = await _train_arm(
                config=curriculum,
                initial=checkpoints,
                arm=arm,
                optimize=True,
                local_valence_enabled=True,
                segment_credit_enabled=True,
                environment_milestone_switch_enabled=milestone_enabled,
                schedule=station_schedule,
                schedule_start_index=completed,
                episode_callback=make_save_episode(
                    active_arm=arm,
                    active_reports=reports,
                ),
            )
        final_checkpoints[arm] = checkpoints
        reports_by_arm[arm] = reports

    probes = await run_ecology_checkpoint_post_pickup_uturn_probes(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=config.seed + 700_003,
        checkpoints=final_checkpoints[ECOLOGY_SAME_PHYSICS_CANDIDATE_ARM],
    )
    evaluation = evaluate_same_physics_station1(
        packet=packet,
        candidate_reports=reports_by_arm[
            ECOLOGY_SAME_PHYSICS_CANDIDATE_ARM
        ],
        control_reports=reports_by_arm[ECOLOGY_SAME_PHYSICS_CONTROL_ARM],
        structural_lanes=_structural_lane_rows(probes),
    )
    return {
        "schema_version": (
            ECOLOGY_SAME_PHYSICS_STATION1_REPORT_SCHEMA_VERSION
        ),
        "preregistration_sha256": preregistration_sha256,
        "schedule_sha256": schedule_sha256,
        "completed_episode_count_per_arm": (
            ECOLOGY_SAME_PHYSICS_STATION1_EPISODES
        ),
        **evaluation,
    }


__all__ = [
    "ECOLOGY_SAME_PHYSICS_PROGRESS_SCHEMA_VERSION",
    "ECOLOGY_SAME_PHYSICS_STATION1_REPORT_SCHEMA_VERSION",
    "evaluate_same_physics_station1",
    "run_ecology_same_physics_station1",
]
