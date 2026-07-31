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
    EcologyProbeKind,
    run_ecology_checkpoint_action_probes,
    run_ecology_checkpoint_post_pickup_uturn_probes,
)
from volvence_ant.experiments.ecology_same_physics_baseline import (
    ECOLOGY_SAME_PHYSICS_CANDIDATE_ARM,
    ECOLOGY_SAME_PHYSICS_CONTROL_ARM,
    ECOLOGY_SAME_PHYSICS_STATION1_EPISODES,
)
from volvence_ant.runtime import AntLearningCheckpoint, KernelColonyRunner


ECOLOGY_SAME_PHYSICS_PROGRESS_SCHEMA_VERSION = (
    "digital-ant-ecology-same-physics-progress.v2"
)
ECOLOGY_SAME_PHYSICS_STATION1_REPORT_SCHEMA_VERSION = (
    "digital-ant-ecology-same-physics-station1.v3"
)
ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_REPORT_SCHEMA_VERSION = (
    "digital-ant-ecology-same-physics-alignment-review.v1"
)
ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_BINDING_SCHEMA_VERSION = (
    "digital-ant-ecology-same-physics-alignment-review-binding.v1"
)
_ALIGNMENT_REVIEW_ARM = "learned_alignment_review"


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


def _bind_alignment_review_progress(
    *,
    progress_dir: Path,
    preregistration_sha256: str,
    review_preregistration_sha256: str,
    station1_checkpoint_sha256: str,
    review_schedule_sha256: str,
) -> None:
    path = progress_dir / "alignment-review-binding.json"
    expected = {
        "schema_version": (
            ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_BINDING_SCHEMA_VERSION
        ),
        "preregistration_sha256": preregistration_sha256,
        "review_preregistration_sha256": (
            review_preregistration_sha256
        ),
        "station1_checkpoint_sha256": station1_checkpoint_sha256,
        "review_schedule_sha256": review_schedule_sha256,
    }
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(
                "alignment review progress is bound to a different station1 "
                "checkpoint, preregistration, or review schedule"
            )
        return
    unexpected = tuple(
        item.name
        for item in progress_dir.iterdir()
        if item.name != ".writer.lock"
    )
    if unexpected:
        raise ValueError(
            "alignment review requires a new empty progress directory; "
            f"found {unexpected!r}"
        )
    _atomic_write(path, _stable_json_bytes(expected))


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
    food_alignment_rows: list[dict[str, Any]],
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
    required_aligned_bodies = int(
        thresholds["candidate_food_alignment_direct_station2_bodies"]
    )
    if len(food_alignment_rows) != required_aligned_bodies:
        raise ValueError(
            "station1 food-alignment probe must publish exactly one row per "
            f"body: expected={required_aligned_bodies}, "
            f"actual={len(food_alignment_rows)}"
        )
    aligned_food_bodies = sum(
        bool(row["input_reachable"])
        and bool(row["action_sensitive"])
        and bool(row["target_aligned"])
        for row in food_alignment_rows
    )
    causal_gates_passed = all(gates.values())
    direct_station2 = (
        causal_gates_passed
        and aligned_food_bodies >= required_aligned_bodies
    )
    review_permitted = bool(
        thresholds.get("food_alignment_review_authorized", True)
    )
    if not review_permitted:
        gates["food_alignment_4_of_4"] = direct_station2
    station1_passed = all(gates.values())
    alignment_review_authorized = (
        review_permitted and causal_gates_passed and not direct_station2
    )
    return {
        "candidate_blocks": candidate_blocks,
        "control_blocks": control_blocks,
        "candidate_total_pickups": candidate_total,
        "control_total_pickups": control_total,
        "candidate_to_control_pickup_ratio": pickup_ratio,
        "structural_lanes": structural_lanes,
        "food_alignment_rows": food_alignment_rows,
        "aligned_food_bodies": aligned_food_bodies,
        "required_aligned_food_bodies": required_aligned_bodies,
        "food_alignment_status": (
            "DIRECT_STATION2"
            if direct_station2
            else (
                "REVIEW_REQUIRED"
                if alignment_review_authorized
                else (
                    "BLOCKED_BY_ALIGNMENT"
                    if causal_gates_passed
                    else "BLOCKED_BY_STATION1"
                )
            )
        ),
        "alignment_review_authorized": alignment_review_authorized,
        "gates": gates,
        "verdict": "GO" if station1_passed else "BLOCK",
        "next_episode_authorized": (
            ECOLOGY_SAME_PHYSICS_STATION1_EPISODES
            if direct_station2
            else None
        ),
    }


def _aligned_food_body_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        bool(row["input_reachable"])
        and bool(row["action_sensitive"])
        and bool(row["target_aligned"])
        for row in rows
    )


def evaluate_same_physics_alignment_review(
    *,
    packet: Mapping[str, Any],
    station1_evaluation: Mapping[str, Any],
    review_schedule_rows: list[dict[str, Any]],
    food_alignment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Grade the single review path frozen before station-1 results exist."""

    thresholds = packet["thresholds"]["station1"]
    if thresholds.get("food_alignment_review_authorized", True) is not True:
        raise ValueError(
            "alignment review is forbidden by this station1 preregistration"
        )
    review_episode_count = int(
        thresholds["food_alignment_review_episode_count"]
    )
    expected_review_rows = [
        dict(row)
        for row in packet["schedule"]["rows"][:review_episode_count]
    ]
    if review_schedule_rows != expected_review_rows:
        raise ValueError(
            "alignment review must replay the preregistered station schedule "
            f"rows [0, {review_episode_count}) without seed or config changes"
        )
    if (
        station1_evaluation.get("verdict") != "GO"
        or station1_evaluation.get("alignment_review_authorized") is not True
        or station1_evaluation.get("next_episode_authorized") is not None
    ):
        raise ValueError(
            "alignment review requires a causal station1 GO that explicitly "
            "authorizes review and does not authorize episode 20"
        )
    required_bodies = int(
        thresholds["food_alignment_review_reprobe_required_bodies"]
    )
    if int(station1_evaluation["aligned_food_bodies"]) >= required_bodies:
        raise ValueError(
            "alignment review is forbidden when station1 already meets the "
            "direct station2 alignment threshold"
        )
    if len(food_alignment_rows) != required_bodies:
        raise ValueError(
            "alignment review re-probe must publish exactly one row per "
            f"body: expected={required_bodies}, "
            f"actual={len(food_alignment_rows)}"
        )
    aligned_food_bodies = _aligned_food_body_count(food_alignment_rows)
    passed = aligned_food_bodies >= required_bodies
    return {
        "schema_version": (
            ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_REPORT_SCHEMA_VERSION
        ),
        "review_episode_count": review_episode_count,
        "review_schedule_rows": review_schedule_rows,
        "pre_review_aligned_food_bodies": int(
            station1_evaluation["aligned_food_bodies"]
        ),
        "food_alignment_rows": food_alignment_rows,
        "post_review_aligned_food_bodies": aligned_food_bodies,
        "required_aligned_food_bodies": required_bodies,
        "verdict": "GO" if passed else "BLOCK",
        "next_episode_authorized": (
            ECOLOGY_SAME_PHYSICS_STATION1_EPISODES if passed else None
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


def _food_alignment_rows(probes: tuple[Any, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probe in probes:
        food = tuple(
            item
            for item in probe.probes
            if item.kind is EcologyProbeKind.FOOD
        )
        if len(food) != 1:
            raise ValueError(
                "station1 food-alignment probe must publish exactly one food "
                f"row per body, got body={probe.body_id}, rows={len(food)}"
            )
        item = food[0]
        rows.append(
            {
                "body_id": probe.body_id,
                "input_reachable": item.input_reachable,
                "action_sensitive": item.action_sensitive,
                "target_aligned": item.target_aligned,
                "left_turn": item.left_turn,
                "right_turn": item.right_turn,
                "policy_fingerprint": probe.policy_fingerprint,
                "temporal_learning_fingerprint": (
                    probe.temporal_learning_fingerprint
                ),
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
    action_probes = await run_ecology_checkpoint_action_probes(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=(
            config.seed
            + int(
                packet["thresholds"]["station1"][
                    "food_alignment_probe_seed_offset"
                ]
            )
        ),
        checkpoints=final_checkpoints[ECOLOGY_SAME_PHYSICS_CANDIDATE_ARM],
    )
    evaluation = evaluate_same_physics_station1(
        packet=packet,
        candidate_reports=reports_by_arm[
            ECOLOGY_SAME_PHYSICS_CANDIDATE_ARM
        ],
        control_reports=reports_by_arm[ECOLOGY_SAME_PHYSICS_CONTROL_ARM],
        structural_lanes=_structural_lane_rows(probes),
        food_alignment_rows=_food_alignment_rows(action_probes),
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


async def run_ecology_same_physics_alignment_review(
    *,
    packet: Mapping[str, Any],
    preregistration_sha256: str,
    review_preregistration_sha256: str,
    station1_evaluation: Mapping[str, Any],
    station1_progress_dir: Path,
    review_progress_dir: Path,
    max_new_work_items: int | None = None,
) -> dict[str, Any]:
    """Run exactly five frozen butter-near reviews and one re-probe."""

    if max_new_work_items is not None and max_new_work_items < 1:
        raise ValueError("max_new_work_items must be positive")
    config = EcologyP1Config(**dict(packet["formal_config"]))
    curriculum = _curriculum_config(config)
    schedule = _fixed_schedule(config)
    schedule_sha256 = _schedule_digest(schedule)
    if schedule_sha256 != packet["schedule"]["full_sha256"]:
        raise ValueError("same-physics schedule drifted after preregistration")
    thresholds = packet["thresholds"]["station1"]
    review_episode_count = int(
        thresholds["food_alignment_review_episode_count"]
    )
    review_schedule = schedule[:review_episode_count]
    review_schedule_rows = [
        _json_ready(asdict(item)) for item in review_schedule
    ]
    expected_review_rows = [
        dict(row)
        for row in packet["schedule"]["rows"][:review_episode_count]
    ]
    if review_schedule_rows != expected_review_rows:
        raise ValueError(
            "alignment review schedule drifted from preregistered rows"
        )
    # Preflight the authorization before hydrating or spending rollout budget.
    evaluate_same_physics_alignment_review(
        packet=packet,
        station1_evaluation=station1_evaluation,
        review_schedule_rows=review_schedule_rows,
        food_alignment_rows=[
            {
                "input_reachable": True,
                "action_sensitive": True,
                "target_aligned": True,
            }
            for _ in range(
                int(
                    thresholds[
                        "food_alignment_review_reprobe_required_bodies"
                    ]
                )
            )
        ],
    )
    station1_state = _load_arm_progress(
        progress_dir=station1_progress_dir,
        arm=ECOLOGY_SAME_PHYSICS_CANDIDATE_ARM,
        config=config,
        schedule_sha256=schedule_sha256,
    )
    if station1_state is None:
        raise ValueError("station1 candidate checkpoint is missing")
    completed_station1 = int(
        station1_state["completed_training_episodes"]
    )
    if completed_station1 != ECOLOGY_SAME_PHYSICS_STATION1_EPISODES:
        raise ValueError(
            "alignment review requires the exact station1 candidate "
            f"checkpoint: completed={completed_station1}"
        )
    station1_checkpoint_sha256 = str(
        station1_state["checkpoint_sha256"]
    )
    review_schedule_sha256 = _schedule_digest(review_schedule)
    review_progress_dir.mkdir(parents=True, exist_ok=True)
    _bind_alignment_review_progress(
        progress_dir=review_progress_dir,
        preregistration_sha256=preregistration_sha256,
        review_preregistration_sha256=(
            review_preregistration_sha256
        ),
        station1_checkpoint_sha256=station1_checkpoint_sha256,
        review_schedule_sha256=review_schedule_sha256,
    )
    review_state = _load_arm_progress(
        progress_dir=review_progress_dir,
        arm=_ALIGNMENT_REVIEW_ARM,
        config=config,
        schedule_sha256=review_schedule_sha256,
    )
    completed_review = (
        int(review_state["completed_training_episodes"])
        if review_state is not None
        else 0
    )
    if completed_review > review_episode_count:
        raise ValueError("alignment review progress exceeds frozen budget")
    reports = _load_arm_reports(
        progress_dir=review_progress_dir,
        arm=_ALIGNMENT_REVIEW_ARM,
        preregistration_sha256=preregistration_sha256,
        completed=completed_review,
    )
    if review_state is None:
        checkpoints = _hydrate_progress_checkpoints(
            config=config,
            curriculum=curriculum,
            archives=_read_progress_archive(
                progress_dir=station1_progress_dir,
                state=station1_state,
                config=config,
            ),
            arm=ECOLOGY_SAME_PHYSICS_CANDIDATE_ARM,
        )
    else:
        checkpoints = _hydrate_progress_checkpoints(
            config=config,
            curriculum=curriculum,
            archives=_read_progress_archive(
                progress_dir=review_progress_dir,
                state=review_state,
                config=config,
            ),
            arm=_ALIGNMENT_REVIEW_ARM,
        )

    def save_episode(
        schedule_index: int,
        runner: KernelColonyRunner,
        _checkpoints: tuple[AntLearningCheckpoint, ...],
        report: Any,
    ) -> None:
        if schedule_index != len(reports):
            raise ValueError(
                "alignment review report order mismatch: "
                f"index={schedule_index}, reports={len(reports)}"
            )
        reports.append(_json_ready(asdict(report)))
        _save_arm_reports(
            progress_dir=review_progress_dir,
            arm=_ALIGNMENT_REVIEW_ARM,
            preregistration_sha256=preregistration_sha256,
            reports=reports,
        )
        completed_count = schedule_index + 1
        _save_arm_progress(
            progress_dir=review_progress_dir,
            arm=_ALIGNMENT_REVIEW_ARM,
            config=config,
            schedule_sha256=review_schedule_sha256,
            completed_training_episodes=completed_count,
            runner=runner,
            training_complete=completed_count == review_episode_count,
            last_episode_report=report,
        )
        if (
            max_new_work_items is not None
            and completed_count - completed_review >= max_new_work_items
        ):
            raise EcologyP1ProgressPaused(
                completed_work_items=completed_count - completed_review
            )

    if completed_review < review_episode_count:
        checkpoints, _, _, _, _ = await _train_arm(
            config=curriculum,
            initial=checkpoints,
            arm=_ALIGNMENT_REVIEW_ARM,
            optimize=True,
            local_valence_enabled=True,
            segment_credit_enabled=True,
            environment_milestone_switch_enabled=True,
            schedule=review_schedule,
            schedule_start_index=completed_review,
            episode_callback=save_episode,
        )
    final_state = _load_arm_progress(
        progress_dir=review_progress_dir,
        arm=_ALIGNMENT_REVIEW_ARM,
        config=config,
        schedule_sha256=review_schedule_sha256,
    )
    if (
        final_state is None
        or int(final_state["completed_training_episodes"])
        != review_episode_count
        or final_state.get("training_complete") is not True
    ):
        raise RuntimeError(
            "alignment review checkpoint is incomplete before re-probe"
        )
    action_probes = await run_ecology_checkpoint_action_probes(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=(
            config.seed
            + int(thresholds["food_alignment_probe_seed_offset"])
        ),
        checkpoints=checkpoints,
    )
    evaluation = evaluate_same_physics_alignment_review(
        packet=packet,
        station1_evaluation=station1_evaluation,
        review_schedule_rows=review_schedule_rows,
        food_alignment_rows=_food_alignment_rows(action_probes),
    )
    return {
        "preregistration_sha256": preregistration_sha256,
        "review_preregistration_sha256": (
            review_preregistration_sha256
        ),
        "station1_checkpoint_sha256": station1_checkpoint_sha256,
        "review_checkpoint_sha256": str(
            final_state["checkpoint_sha256"]
        ),
        "review_schedule_sha256": review_schedule_sha256,
        "review_episode_reports": reports,
        **evaluation,
    }


__all__ = [
    "ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_BINDING_SCHEMA_VERSION",
    "ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_REPORT_SCHEMA_VERSION",
    "ECOLOGY_SAME_PHYSICS_PROGRESS_SCHEMA_VERSION",
    "ECOLOGY_SAME_PHYSICS_STATION1_REPORT_SCHEMA_VERSION",
    "evaluate_same_physics_alignment_review",
    "evaluate_same_physics_station1",
    "run_ecology_same_physics_alignment_review",
    "run_ecology_same_physics_station1",
]
