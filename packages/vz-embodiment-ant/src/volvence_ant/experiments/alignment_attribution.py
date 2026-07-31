"""Read-only attribution for the ecology food-alignment formation failure.

The attribution owner consumes immutable station/review reports plus owner-
exported learning checkpoint archives.  It never resumes training and never
writes into an ecology journal.  The resulting artifact distinguishes three
hypotheses from the five-lever evidence plan:

* H1: one body settled into a different learned action-head state;
* H2: that body received less food exposure than its peers;
* H3: food and carrying updates may interfere on the shared action head.

Per-tick gradient magnitude and gradient cosine were not published by the v28
journal contract.  The artifact therefore reports that limitation explicitly
instead of reconstructing hidden owner state from text or inventing a proxy.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from volvence_ant.experiments.ecology_p1 import (
    EcologyP1Config,
    _curriculum_config,
    _fixed_schedule,
    _hydrate_progress_checkpoints,
    _load_arm_progress,
    _read_progress_archive,
    _schedule_digest,
)
from volvence_ant.experiments.ecology_probe import (
    EcologyProbeKind,
    run_ecology_checkpoint_action_probes,
)
from volvence_ant.runtime import AntLearningCheckpoint


ALIGNMENT_FORMATION_ATTRIBUTION_SCHEMA_VERSION = (
    "alignment-formation-attribution.v1"
)
ALIGNMENT_ATTRIBUTION_PROBE_SEEDS = (
    700_003,
    700_004,
    700_005,
    700_006,
    700_007,
)
ALIGNMENT_TURN_THRESHOLD = 1e-4


class AlignmentFormationAttributionError(ValueError):
    """The frozen attribution inputs violate their declared contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _source_path(path: Path, *, source_root: Path | None) -> str:
    resolved = path.resolve()
    if source_root is None:
        return resolved.as_posix()
    try:
        return resolved.relative_to(source_root.resolve()).as_posix()
    except ValueError as exc:
        raise AlignmentFormationAttributionError(
            f"formal source path is outside source_root: {resolved}"
        ) from exc


def _json_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AlignmentFormationAttributionError(
            f"expected JSON object at {path}"
        )
    return payload


def _alignment_rows(report: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_rows = report.get("food_alignment_rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != 4:
        raise AlignmentFormationAttributionError(
            "alignment report must contain exactly four body rows"
        )
    rows: list[dict[str, Any]] = []
    for expected_body_id, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise AlignmentFormationAttributionError(
                "alignment row must be a JSON object"
            )
        body_id = raw.get("body_id")
        if body_id != expected_body_id:
            raise AlignmentFormationAttributionError(
                "alignment body ids must be ordered 0..3: "
                f"expected={expected_body_id}, actual={body_id!r}"
            )
        left_turn = raw.get("left_turn")
        right_turn = raw.get("right_turn")
        target_aligned = raw.get("target_aligned")
        if (
            isinstance(left_turn, bool)
            or not isinstance(left_turn, (int, float))
            or isinstance(right_turn, bool)
            or not isinstance(right_turn, (int, float))
            or not isinstance(target_aligned, bool)
        ):
            raise AlignmentFormationAttributionError(
                f"alignment row {body_id} has invalid typed fields"
            )
        signed_min_turn = min(float(left_turn), -float(right_turn))
        rows.append(
            {
                "body_id": body_id,
                "left_turn": float(left_turn),
                "right_turn": float(right_turn),
                "target_aligned": target_aligned,
                "signed_min_turn": signed_min_turn,
                "margin_over_threshold": (
                    signed_min_turn - ALIGNMENT_TURN_THRESHOLD
                ),
                "distance_to_pass": max(
                    ALIGNMENT_TURN_THRESHOLD - signed_min_turn,
                    0.0,
                ),
            }
        )
    return tuple(rows)


def _episode_rows(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list):
        raise AlignmentFormationAttributionError(
            "station report journal must publish an episodes array"
        )
    episodes: list[Mapping[str, Any]] = []
    for raw in raw_episodes:
        if not isinstance(raw, dict):
            raise AlignmentFormationAttributionError(
                "station report episode must be a JSON object"
            )
        episodes.append(raw)
    return tuple(episodes)


def summarize_body_food_exposure(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Aggregate owner-published butter-near contact evidence per body."""

    counts = {
        body_id: {
            "body_id": body_id,
            "butter_near_episode_rows": 0,
            "encountered_food_events": 0,
            "pickup_events": 0,
            "delivery_events": 0,
            "applied_distance_sum": 0.0,
        }
        for body_id in range(4)
    }
    for episode in _episode_rows(payload):
        lineage = episode.get("body_lineage")
        if not isinstance(lineage, list) or len(lineage) != 4:
            raise AlignmentFormationAttributionError(
                "each station episode must publish four body-lineage rows"
            )
        for raw in lineage:
            if not isinstance(raw, dict):
                raise AlignmentFormationAttributionError(
                    "body-lineage row must be a JSON object"
                )
            body_id = raw.get("body_id")
            if body_id not in counts:
                raise AlignmentFormationAttributionError(
                    f"unexpected body id in station report: {body_id!r}"
                )
            if raw.get("stage") != "butter" or raw.get("tier") != "near":
                continue
            row = counts[body_id]
            row["butter_near_episode_rows"] += 1
            row["encountered_food_events"] += int(
                raw.get("encountered_food") is True
            )
            row["pickup_events"] += int(raw.get("picked_up") is True)
            row["delivery_events"] += int(raw.get("delivered") is True)
            applied_distance = raw.get("applied_distance")
            if (
                isinstance(applied_distance, bool)
                or not isinstance(applied_distance, (int, float))
            ):
                raise AlignmentFormationAttributionError(
                    "body-lineage applied_distance must be numeric"
                )
            row["applied_distance_sum"] += float(applied_distance)
    result: list[dict[str, Any]] = []
    for body_id in range(4):
        row = counts[body_id]
        episode_count = int(row["butter_near_episode_rows"])
        if episode_count < 1:
            raise AlignmentFormationAttributionError(
                f"body {body_id} has no butter-near exposure rows"
            )
        result.append(
            {
                **row,
                "mean_applied_distance": (
                    float(row["applied_distance_sum"]) / episode_count
                ),
            }
        )
    return tuple(result)


def _load_checkpoints(
    *,
    progress_dir: Path,
    arm: str,
    config: EcologyP1Config,
    review: bool,
) -> tuple[tuple[AntLearningCheckpoint, ...], dict[str, Any]]:
    schedule = _fixed_schedule(config)
    if review:
        schedule = schedule[:5]
    state = _load_arm_progress(
        progress_dir=progress_dir,
        arm=arm,
        config=config,
        schedule_sha256=_schedule_digest(schedule),
    )
    if state is None:
        raise AlignmentFormationAttributionError(
            f"missing frozen progress for arm={arm!r} under {progress_dir}"
        )
    checkpoints = _hydrate_progress_checkpoints(
        config=config,
        curriculum=_curriculum_config(config),
        archives=_read_progress_archive(
            progress_dir=progress_dir,
            state=state,
            config=config,
        ),
        arm=arm,
    )
    if len(checkpoints) != 4:
        raise AlignmentFormationAttributionError(
            f"attribution requires four body checkpoints, got {len(checkpoints)}"
        )
    return checkpoints, state


def _world_action_head_vector(
    checkpoint: AntLearningCheckpoint,
) -> tuple[float, ...]:
    joint_state = checkpoint.joint_loop_state
    snapshot = joint_state.world_policy_checkpoint.metacontroller_snapshot
    heads = tuple(
        head for head in snapshot.causal_action_heads
        if head.track.value == "world"
    )
    if len(heads) != 1:
        raise AlignmentFormationAttributionError(
            "checkpoint must publish exactly one world causal action head"
        )
    head = heads[0]
    return (
        tuple(value for row in head.input_factors for value in row)
        + tuple(value for row in head.output_factors for value in row)
        + tuple(head.bias)
    )


def _l2(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise AlignmentFormationAttributionError(
            "action-head vectors have inconsistent dimensions"
        )
    return math.sqrt(
        sum((a - b) * (a - b) for a, b in zip(left, right, strict=True))
    )


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _centroid(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        raise AlignmentFormationAttributionError(
            "cannot build an empty action-head centroid"
        )
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise AlignmentFormationAttributionError(
            "action-head vectors have inconsistent dimensions"
        )
    return tuple(
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(width)
    )


def action_head_geometry(
    checkpoints: Sequence[AntLearningCheckpoint],
    *,
    aligned_body_ids: Sequence[int],
) -> dict[str, Any]:
    """Publish distances without leaking mutable owner references."""

    vectors = tuple(_world_action_head_vector(checkpoint) for checkpoint in checkpoints)
    if not aligned_body_ids:
        raise AlignmentFormationAttributionError(
            "action-head geometry requires at least one aligned body"
        )
    centroid = _centroid(tuple(vectors[index] for index in aligned_body_ids))
    pairwise = tuple(
        tuple(_l2(left, right) for right in vectors)
        for left in vectors
    )
    rows = []
    for body_id, (checkpoint, vector) in enumerate(
        zip(checkpoints, vectors, strict=True)
    ):
        policy_parameters = tuple(
            item
            for item in checkpoint.joint_loop_state.world_policy_checkpoint.parameters_by_track
            if item.track.value == "world"
        )
        if len(policy_parameters) != 1:
            raise AlignmentFormationAttributionError(
                "checkpoint must publish exactly one world policy row"
            )
        encoded = json.dumps(
            list(vector),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        rows.append(
            {
                "body_id": body_id,
                "vector_dimension": len(vector),
                "vector_sha256": _sha256_bytes(encoded),
                "vector_l2_norm": _norm(vector),
                "distance_to_aligned_centroid": _l2(vector, centroid),
                "world_policy_update_step": policy_parameters[0].update_step,
            }
        )
    return {
        "aligned_body_ids": list(aligned_body_ids),
        "per_body": rows,
        "pairwise_l2": [list(row) for row in pairwise],
    }


def _combine_exposure(
    station: Sequence[Mapping[str, Any]],
    review: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if len(station) != 4 or len(review) != 4:
        raise AlignmentFormationAttributionError(
            "exposure combination requires four station and four review rows"
        )
    rows = []
    for body_id, (before, after) in enumerate(
        zip(station, review, strict=True)
    ):
        if before["body_id"] != body_id or after["body_id"] != body_id:
            raise AlignmentFormationAttributionError(
                "exposure body ids must be ordered 0..3"
            )
        rows.append(
            {
                "body_id": body_id,
                "station1": dict(before),
                "review": dict(after),
                "combined_encountered_food_events": (
                    int(before["encountered_food_events"])
                    + int(after["encountered_food_events"])
                ),
                "combined_pickup_events": (
                    int(before["pickup_events"])
                    + int(after["pickup_events"])
                ),
            }
        )
    return tuple(rows)


async def _probe_seed_matrix(
    *,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    temporal_latent_dim: int,
) -> tuple[dict[str, Any], ...]:
    matrix = []
    for seed in ALIGNMENT_ATTRIBUTION_PROBE_SEEDS:
        probes = await run_ecology_checkpoint_action_probes(
            temporal_latent_dim=temporal_latent_dim,
            seed=seed,
            checkpoints=checkpoints,
            turn_delta_threshold=ALIGNMENT_TURN_THRESHOLD,
        )
        rows = []
        for probe in probes:
            food = tuple(
                item
                for item in probe.probes
                if item.kind is EcologyProbeKind.FOOD
            )
            if len(food) != 1:
                raise AlignmentFormationAttributionError(
                    "probe must publish one food row per body"
                )
            item = food[0]
            signed_min_turn = min(item.left_turn, -item.right_turn)
            rows.append(
                {
                    "body_id": probe.body_id,
                    "left_turn": item.left_turn,
                    "right_turn": item.right_turn,
                    "target_aligned": item.target_aligned,
                    "signed_min_turn": signed_min_turn,
                    "margin_over_threshold": (
                        signed_min_turn - ALIGNMENT_TURN_THRESHOLD
                    ),
                }
            )
        matrix.append({"probe_seed": seed, "rows": rows})
    return tuple(matrix)


def _assert_primary_probe_reproduces_report(
    *,
    reported_rows: Sequence[Mapping[str, Any]],
    probe_matrix: Sequence[Mapping[str, Any]],
) -> None:
    if not probe_matrix or probe_matrix[0]["probe_seed"] != 700_003:
        raise AlignmentFormationAttributionError(
            "primary attribution probe seed must reproduce seed 700003"
        )
    observed_rows = probe_matrix[0]["rows"]
    if not isinstance(observed_rows, list) or len(observed_rows) != 4:
        raise AlignmentFormationAttributionError(
            "primary probe must publish four rows"
        )
    for reported, observed in zip(reported_rows, observed_rows, strict=True):
        if reported["body_id"] != observed["body_id"]:
            raise AlignmentFormationAttributionError(
                "primary probe body ids differ from report"
            )
        if (
            not math.isclose(
                float(reported["left_turn"]),
                float(observed["left_turn"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(reported["right_turn"]),
                float(observed["right_turn"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or reported["target_aligned"] != observed["target_aligned"]
        ):
            raise AlignmentFormationAttributionError(
                "current owner probe does not reproduce the immutable report"
            )


def _per_body_by_id(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(row["body_id"]): row for row in rows}


def classify_alignment_hypotheses(
    *,
    station_rows: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    combined_exposure: Sequence[Mapping[str, Any]],
    station_geometry: Mapping[str, Any],
    review_geometry: Mapping[str, Any],
    station_probe_matrix: Sequence[Mapping[str, Any]],
    review_probe_matrix: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen L1-A attribution decision without changing a gate."""

    failed_station = tuple(
        int(row["body_id"])
        for row in station_rows
        if row["target_aligned"] is False
    )
    failed_review = tuple(
        int(row["body_id"])
        for row in review_rows
        if row["target_aligned"] is False
    )
    if len(failed_station) != 1 or failed_station != failed_review:
        raise AlignmentFormationAttributionError(
            "L1-A requires one stable failed body across station1 and review"
        )
    failed_body_id = failed_review[0]
    aligned_body_ids = tuple(
        int(row["body_id"])
        for row in review_rows
        if row["target_aligned"] is True
    )
    all_probe_rows = []
    for matrix in (station_probe_matrix, review_probe_matrix):
        for seed_result in matrix:
            rows = seed_result["rows"]
            if not isinstance(rows, list):
                raise AlignmentFormationAttributionError(
                    "probe seed result rows must be a list"
                )
            all_probe_rows.extend(
                row for row in rows if row["body_id"] == failed_body_id
            )
    failure_stable = bool(all_probe_rows) and all(
        row["target_aligned"] is False for row in all_probe_rows
    )
    measurement_noise_kill_triggered = not failure_stable

    exposure_by_id = _per_body_by_id(combined_exposure)
    failed_exposure = exposure_by_id[failed_body_id]
    aligned_exposure = tuple(exposure_by_id[index] for index in aligned_body_ids)
    failed_has_exposure_scarcity = (
        int(failed_exposure["combined_encountered_food_events"])
        < min(
            int(row["combined_encountered_food_events"])
            for row in aligned_exposure
        )
        or int(failed_exposure["combined_pickup_events"])
        < min(int(row["combined_pickup_events"]) for row in aligned_exposure)
    )

    station_geometry_by_id = _per_body_by_id(station_geometry["per_body"])
    review_geometry_by_id = _per_body_by_id(review_geometry["per_body"])
    failed_review_distance = float(
        review_geometry_by_id[failed_body_id]["distance_to_aligned_centroid"]
    )
    max_aligned_review_distance = max(
        float(row["distance_to_aligned_centroid"])
        for body_id, row in review_geometry_by_id.items()
        if body_id in aligned_body_ids
    )
    parameter_outlier = failed_review_distance > max_aligned_review_distance

    station_by_id = _per_body_by_id(station_rows)
    review_by_id = _per_body_by_id(review_rows)
    failed_margin_change = (
        float(review_by_id[failed_body_id]["margin_over_threshold"])
        - float(station_by_id[failed_body_id]["margin_over_threshold"])
    )
    butter_near_review_improved_margin = failed_margin_change > 0.0

    h1_supported = (
        failure_stable
        and parameter_outlier
        and not failed_has_exposure_scarcity
    )
    selected_hypothesis = (
        "H1_learning_state_divergence"
        if h1_supported and not measurement_noise_kill_triggered
        else "INCONCLUSIVE_REDESIGN_REQUIRED"
    )
    return {
        "failed_body_id": failed_body_id,
        "failed_body_stable_across_station_and_review": True,
        "failure_stable_under_probe_seed_perturbation": failure_stable,
        "measurement_noise_kill_triggered": measurement_noise_kill_triggered,
        "selected_hypothesis": selected_hypothesis,
        "hypotheses": {
            "H1_learning_state_divergence": {
                "verdict": "supported" if h1_supported else "inconclusive",
                "failed_body_distance_to_aligned_centroid": (
                    failed_review_distance
                ),
                "max_aligned_body_distance_to_aligned_centroid": (
                    max_aligned_review_distance
                ),
                "failed_body_is_parameter_outlier": parameter_outlier,
                "failed_body_world_policy_update_step_station1": (
                    station_geometry_by_id[failed_body_id][
                        "world_policy_update_step"
                    ]
                ),
                "failed_body_world_policy_update_step_review": (
                    review_geometry_by_id[failed_body_id][
                        "world_policy_update_step"
                    ]
                ),
            },
            "H2_curriculum_exposure_imbalance": {
                "verdict": (
                    "supported"
                    if failed_has_exposure_scarcity
                    else "not-supported"
                ),
                "failed_body_has_exposure_scarcity": (
                    failed_has_exposure_scarcity
                ),
                "failed_body_combined_encountered_food_events": (
                    failed_exposure["combined_encountered_food_events"]
                ),
                "failed_body_combined_pickup_events": (
                    failed_exposure["combined_pickup_events"]
                ),
            },
            "H3_gradient_interference": {
                "verdict": "inconclusive",
                "selection_role": (
                    "not-selected; direct per-update gradient evidence "
                    "unavailable"
                ),
                "failed_body_margin_change_during_butter_near_review": (
                    failed_margin_change
                ),
                "butter_near_review_improved_alignment_margin": (
                    butter_near_review_improved_margin
                ),
                "direct_per_update_gradient_cosine_available": False,
            },
        },
    }


async def build_alignment_formation_attribution(
    *,
    station1_report_path: Path,
    review_report_path: Path,
    station1_progress_dir: Path,
    review_progress_dir: Path,
    seed: int = 0,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Build one deterministic L1-A attribution artifact."""

    station1_report = _json_payload(station1_report_path)
    review_report = _json_payload(review_report_path)
    if station1_report.get("verdict") != "GO":
        raise AlignmentFormationAttributionError(
            "L1-A source station1 report must have verdict=GO"
        )
    if review_report.get("verdict") != "BLOCK":
        raise AlignmentFormationAttributionError(
            "L1-A source review report must have verdict=BLOCK"
        )
    station_rows = _alignment_rows(station1_report)
    review_rows = _alignment_rows(review_report)
    config = EcologyP1Config(seed=seed)
    station_checkpoints, station_state = _load_checkpoints(
        progress_dir=station1_progress_dir,
        arm="learned",
        config=config,
        review=False,
    )
    review_checkpoints, review_state = _load_checkpoints(
        progress_dir=review_progress_dir,
        arm="learned_alignment_review",
        config=config,
        review=True,
    )
    if (
        review_report.get("station1_checkpoint_sha256")
        != station_state["checkpoint_sha256"]
    ):
        raise AlignmentFormationAttributionError(
            "station1 checkpoint digest differs from review provenance"
        )
    if (
        review_report.get("review_checkpoint_sha256")
        != review_state["checkpoint_sha256"]
    ):
        raise AlignmentFormationAttributionError(
            "review checkpoint digest differs from review provenance"
        )
    station_reports_path = station1_progress_dir / "learned.station-reports.json"
    review_reports_path = (
        review_progress_dir / "learned_alignment_review.station-reports.json"
    )
    station_reports = _json_payload(station_reports_path)
    review_reports = _json_payload(review_reports_path)
    station_exposure = summarize_body_food_exposure(station_reports)
    review_exposure = summarize_body_food_exposure(review_reports)
    combined_exposure = _combine_exposure(station_exposure, review_exposure)
    aligned_body_ids = tuple(
        int(row["body_id"])
        for row in review_rows
        if row["target_aligned"] is True
    )
    station_geometry = action_head_geometry(
        station_checkpoints,
        aligned_body_ids=aligned_body_ids,
    )
    review_geometry = action_head_geometry(
        review_checkpoints,
        aligned_body_ids=aligned_body_ids,
    )
    station_probe_matrix = await _probe_seed_matrix(
        checkpoints=station_checkpoints,
        temporal_latent_dim=config.temporal_latent_dim,
    )
    review_probe_matrix = await _probe_seed_matrix(
        checkpoints=review_checkpoints,
        temporal_latent_dim=config.temporal_latent_dim,
    )
    _assert_primary_probe_reproduces_report(
        reported_rows=station_rows,
        probe_matrix=station_probe_matrix,
    )
    _assert_primary_probe_reproduces_report(
        reported_rows=review_rows,
        probe_matrix=review_probe_matrix,
    )
    decision = classify_alignment_hypotheses(
        station_rows=station_rows,
        review_rows=review_rows,
        combined_exposure=combined_exposure,
        station_geometry=station_geometry,
        review_geometry=review_geometry,
        station_probe_matrix=station_probe_matrix,
        review_probe_matrix=review_probe_matrix,
    )
    next_package = (
        "L1-B temporal-owner alignment-formation protection; default "
        "DISABLED with byte-equivalent rollback"
        if decision["selected_hypothesis"] == "H1_learning_state_divergence"
        else "alignment-gate semantic redesign under a new schema"
    )
    return {
        "schema_version": ALIGNMENT_FORMATION_ATTRIBUTION_SCHEMA_VERSION,
        "status": "ATTRIBUTION_COMPLETE",
        "read_only": True,
        "training_or_journal_write_performed": False,
        "thresholds": {
            "food_turn_alignment_threshold": ALIGNMENT_TURN_THRESHOLD,
            "probe_seeds": list(ALIGNMENT_ATTRIBUTION_PROBE_SEEDS),
            "measurement_noise_kill_rule": (
                "failed body becomes target_aligned on any frozen seed "
                "perturbation"
            ),
        },
        "sources": {
            "station1_report": {
                "path": _source_path(
                    station1_report_path,
                    source_root=source_root,
                ),
                "sha256": _sha256_file(station1_report_path),
            },
            "review_report": {
                "path": _source_path(
                    review_report_path,
                    source_root=source_root,
                ),
                "sha256": _sha256_file(review_report_path),
            },
            "station1_checkpoint": {
                "arm": "learned",
                "completed_training_episodes": station_state[
                    "completed_training_episodes"
                ],
                "sha256": station_state["checkpoint_sha256"],
            },
            "review_checkpoint": {
                "arm": "learned_alignment_review",
                "completed_training_episodes": review_state[
                    "completed_training_episodes"
                ],
                "sha256": review_state["checkpoint_sha256"],
            },
            "station1_episode_journal": {
                "sha256": _sha256_file(station_reports_path),
                "episode_count": len(_episode_rows(station_reports)),
            },
            "review_episode_journal": {
                "sha256": _sha256_file(review_reports_path),
                "episode_count": len(_episode_rows(review_reports)),
            },
        },
        "station1_alignment": list(station_rows),
        "review_alignment": list(review_rows),
        "body_food_exposure": list(combined_exposure),
        "station1_action_head_geometry": station_geometry,
        "review_action_head_geometry": review_geometry,
        "station1_probe_seed_matrix": list(station_probe_matrix),
        "review_probe_seed_matrix": list(review_probe_matrix),
        "decision": {
            **decision,
            "next_package": next_package,
            "station2_remains_unauthorized": True,
        },
        "limitations": [
            (
                "v28 station reports publish body-level food encounter and "
                "pickup events, not per-tick food-gradient magnitude"
            ),
            (
                "v28 checkpoints do not publish per-update food-vs-carrying "
                "gradient cosine; the frozen butter-near review direction is "
                "diagnostic only and cannot decide H3"
            ),
            (
                "this attribution chooses the next owner mechanism package; "
                "it does not authorize station2, P1, P2, or a gate promotion"
            ),
        ],
    }


__all__ = [
    "ALIGNMENT_ATTRIBUTION_PROBE_SEEDS",
    "ALIGNMENT_FORMATION_ATTRIBUTION_SCHEMA_VERSION",
    "ALIGNMENT_TURN_THRESHOLD",
    "AlignmentFormationAttributionError",
    "action_head_geometry",
    "build_alignment_formation_attribution",
    "classify_alignment_hypotheses",
    "summarize_body_food_exposure",
]
