"""Preregister the bounded food-alignment review before station-1 is read."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from volvence_ant.experiments.ecology_same_physics_baseline import (
    _code_tree_binding,
    _file_binding,
    _sha256,
    _stable_json_bytes,
)
from volvence_ant.experiments.ecology_same_physics_run import (
    ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_REPORT_SCHEMA_VERSION,
)


ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_PREREGISTRATION_SCHEMA_VERSION = (
    "digital-ant-ecology-same-physics-alignment-review-preregistration.v1"
)
ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_STATION1_PACKET_SCHEMA_VERSION = (
    "digital-ant-ecology-same-physics-baseline-preregistration.v2"
)
ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_STATION1_REPORT_SCHEMA_VERSION = (
    "digital-ant-ecology-same-physics-station1.v2"
)

_SOURCE_PATHS = (
    (
        "packages/vz-embodiment-ant/src/volvence_ant/experiments/"
        "ecology_curriculum.py"
    ),
    "packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_p1.py",
    (
        "packages/vz-embodiment-ant/src/volvence_ant/experiments/"
        "ecology_probe.py"
    ),
    (
        "packages/vz-embodiment-ant/src/volvence_ant/experiments/"
        "ecology_same_physics_baseline.py"
    ),
    (
        "packages/vz-embodiment-ant/src/volvence_ant/experiments/"
        "ecology_same_physics_review.py"
    ),
    (
        "packages/vz-embodiment-ant/src/volvence_ant/experiments/"
        "ecology_same_physics_run.py"
    ),
    "packages/vz-embodiment-ant/src/volvence_ant/runtime/ant_session.py",
    "packages/vz-runtime/src/volvence_zero/agent/session.py",
    "packages/vz-runtime/src/volvence_zero/integration/final_wiring.py",
    "packages/vz-temporal/src/volvence_zero/temporal/interface.py",
    "scripts/preregister_ant_ecology_same_physics_alignment_review.py",
    "scripts/run_ant_ecology_same_physics_alignment_review.py",
)


class EcologySamePhysicsAlignmentReviewPacketError(ValueError):
    """The alignment-review packet drifted from its frozen contract."""


def build_ecology_same_physics_alignment_review_packet(
    *,
    repo_root: Path,
    station1_packet: Mapping[str, Any],
    station1_preregistration_sha256: str,
) -> dict[str, Any]:
    """Bind the one allowed review path without reading station-1 results."""

    if (
        station1_packet.get("schema_version")
        != ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_STATION1_PACKET_SCHEMA_VERSION
    ):
        raise EcologySamePhysicsAlignmentReviewPacketError(
            "alignment review requires the v2 station1 preregistration"
        )
    if len(station1_preregistration_sha256) != 64:
        raise EcologySamePhysicsAlignmentReviewPacketError(
            "station1 preregistration SHA256 must contain 64 hex characters"
        )
    thresholds = station1_packet["thresholds"]["station1"]
    review_episode_count = int(
        thresholds["food_alignment_review_episode_count"]
    )
    review_rows = [
        dict(row)
        for row in station1_packet["schedule"]["rows"][
            :review_episode_count
        ]
    ]
    resolved_root = repo_root.resolve()
    return {
        "schema_version": (
            ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_PREREGISTRATION_SCHEMA_VERSION
        ),
        "status": "PREREGISTERED",
        "station1_preregistration": {
            "schema_version": (
                ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_STATION1_PACKET_SCHEMA_VERSION
            ),
            "sha256": station1_preregistration_sha256,
        },
        "accepted_station1_report_schema_version": (
            ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_STATION1_REPORT_SCHEMA_VERSION
        ),
        "output_report_schema_version": (
            ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_REPORT_SCHEMA_VERSION
        ),
        "formal_config": dict(station1_packet["formal_config"]),
        "review_schedule": {
            "episode_count": review_episode_count,
            "rows": review_rows,
            "sha256": _sha256(_stable_json_bytes(review_rows)),
            "source": "station1_preregistration.schedule.rows[0:5]",
        },
        "probe": {
            "seed_offset": int(
                thresholds["food_alignment_probe_seed_offset"]
            ),
            "required_aligned_bodies": int(
                thresholds[
                    "food_alignment_review_reprobe_required_bodies"
                ]
            ),
            "attempt_count": 1,
        },
        "authorization": {
            "station1_verdict": "GO",
            "station1_alignment_review_authorized": True,
            "station1_next_episode_authorized": None,
            "review_pass_next_episode_authorized": 20,
            "review_fail_verdict": "BLOCK",
            "additional_training_after_failure_forbidden": True,
        },
        "execution_contract": {
            "device": "cpu",
            "numeric_precision": "float64",
            "new_empty_progress_directory_required": True,
            "single_writer_lock_required": True,
            "station1_candidate_checkpoint_digest_required": True,
            "source_bindings": [
                _file_binding(
                    repo_root=resolved_root,
                    relative_path=relative_path,
                )
                for relative_path in _SOURCE_PATHS
            ],
            "code_tree_binding": _code_tree_binding(
                repo_root=resolved_root
            ),
        },
    }


def validate_ecology_same_physics_alignment_review_packet(
    packet: Mapping[str, Any],
    *,
    repo_root: Path,
    station1_packet: Mapping[str, Any],
    station1_preregistration_sha256: str,
) -> None:
    """Fail before rollout if inputs or executable review source drift."""

    expected = build_ecology_same_physics_alignment_review_packet(
        repo_root=repo_root,
        station1_packet=station1_packet,
        station1_preregistration_sha256=(
            station1_preregistration_sha256
        ),
    )
    if dict(packet) != expected:
        raise EcologySamePhysicsAlignmentReviewPacketError(
            "alignment-review packet differs from the frozen executable "
            "contract"
        )


__all__ = [
    "ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_PREREGISTRATION_SCHEMA_VERSION",
    "ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_STATION1_PACKET_SCHEMA_VERSION",
    "ECOLOGY_SAME_PHYSICS_ALIGNMENT_REVIEW_STATION1_REPORT_SCHEMA_VERSION",
    "EcologySamePhysicsAlignmentReviewPacketError",
    "build_ecology_same_physics_alignment_review_packet",
    "validate_ecology_same_physics_alignment_review_packet",
]
