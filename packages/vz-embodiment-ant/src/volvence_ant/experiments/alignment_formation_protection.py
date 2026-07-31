"""Frozen-checkpoint precheck for L1-B alignment formation protection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from volvence_zero.runtime import WiringLevel

from volvence_ant.evidence.runtime_profile import (
    ANT_CAUSAL_ACTION_HEAD_FORMATION_CONFLICT_SCALE,
    ANT_CAUSAL_ACTION_HEAD_FORMATION_MAX_UPDATE_STEPS,
    ant_runtime_replay_rollout_config,
)
from volvence_ant.experiments.alignment_attribution import (
    ALIGNMENT_TURN_THRESHOLD,
    AlignmentFormationAttributionError,
    _load_checkpoints,
)
from volvence_ant.experiments.ecology_p1 import EcologyP1Config
from volvence_ant.experiments.ecology_probe import (
    EcologyCheckpointActionProbe,
    EcologyProbeKind,
    run_ecology_checkpoint_action_probes,
)
from volvence_ant.substrate import AntSenseSchema


ALIGNMENT_FORMATION_PROTECTION_PRECHECK_SCHEMA_VERSION = (
    "alignment-formation-protection-precheck.v1"
)


class AlignmentFormationProtectionPrecheckError(ValueError):
    """The L1-B precheck inputs or rollback comparison are invalid."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_path(path: Path, *, source_root: Path | None) -> str:
    resolved = path.resolve()
    if source_root is None:
        return str(resolved)
    try:
        return str(resolved.relative_to(source_root.resolve()))
    except ValueError:
        return str(resolved)


def _stable_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _probe_payload(
    reports: tuple[EcologyCheckpointActionProbe, ...],
) -> list[dict[str, Any]]:
    return [asdict(report) for report in reports]


def _food_rows(
    reports: tuple[EcologyCheckpointActionProbe, ...],
) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        food = tuple(
            probe
            for probe in report.probes
            if probe.kind is EcologyProbeKind.FOOD
        )
        if len(food) != 1:
            raise AlignmentFormationProtectionPrecheckError(
                "each body precheck must publish exactly one food probe"
            )
        probe = food[0]
        rows.append(
            {
                "body_id": report.body_id,
                "left_turn": probe.left_turn,
                "right_turn": probe.right_turn,
                "target_aligned": probe.target_aligned,
                "action_head_update_step": (
                    probe.left_action_head_update_step
                ),
            }
        )
    return rows


async def build_alignment_formation_protection_precheck(
    *,
    review_report_path: Path,
    review_progress_dir: Path,
    seed: int = 0,
    probe_seed: int = 700_003,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Prove ACTIVE L1-B wiring is forward-neutral on the frozen v28 head."""

    try:
        review_report = json.loads(
            review_report_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise AlignmentFormationProtectionPrecheckError(
            f"cannot read frozen review report {review_report_path}"
        ) from error
    if review_report.get("verdict") != "BLOCK":
        raise AlignmentFormationProtectionPrecheckError(
            "L1-B precheck requires the immutable BLOCK review report"
        )
    config = EcologyP1Config(seed=seed)
    try:
        checkpoints, progress_state = _load_checkpoints(
            progress_dir=review_progress_dir,
            arm="learned_alignment_review",
            config=config,
            review=True,
        )
    except AlignmentFormationAttributionError as error:
        raise AlignmentFormationProtectionPrecheckError(str(error)) from error
    checkpoint_sha256 = progress_state["checkpoint_sha256"]
    if review_report.get("review_checkpoint_sha256") != checkpoint_sha256:
        raise AlignmentFormationProtectionPrecheckError(
            "review checkpoint digest differs from frozen report provenance"
        )

    active_config = ant_runtime_replay_rollout_config(
        enable_sparse_exploration=False,
        sense_schema=AntSenseSchema.ECOLOGY_V2,
    )
    if (
        active_config.internal_rl_causal_action_head_formation_protection
        is not WiringLevel.ACTIVE
        or active_config
        .internal_rl_causal_action_head_formation_max_update_steps
        != ANT_CAUSAL_ACTION_HEAD_FORMATION_MAX_UPDATE_STEPS
        or active_config
        .internal_rl_causal_action_head_formation_conflict_scale
        != ANT_CAUSAL_ACTION_HEAD_FORMATION_CONFLICT_SCALE
    ):
        raise AlignmentFormationProtectionPrecheckError(
            "Digital Ant profile does not declare the frozen L1-B mechanism"
        )
    disabled_config = replace(
        active_config,
        internal_rl_causal_action_head_formation_protection=(
            WiringLevel.DISABLED
        ),
        internal_rl_causal_action_head_formation_max_update_steps=0,
        internal_rl_causal_action_head_formation_conflict_scale=1.0,
    )
    disabled_reports = await run_ecology_checkpoint_action_probes(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=probe_seed,
        checkpoints=checkpoints,
        turn_delta_threshold=ALIGNMENT_TURN_THRESHOLD,
        rollout_config=disabled_config,
        learning_enabled=False,
    )
    active_reports = await run_ecology_checkpoint_action_probes(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=probe_seed,
        checkpoints=checkpoints,
        turn_delta_threshold=ALIGNMENT_TURN_THRESHOLD,
        rollout_config=active_config,
        learning_enabled=False,
    )
    disabled_payload = _probe_payload(disabled_reports)
    active_payload = _probe_payload(active_reports)
    if active_payload != disabled_payload:
        raise AlignmentFormationProtectionPrecheckError(
            "formation protection changed frozen checkpoint forward output"
        )
    food_rows = _food_rows(active_reports)
    max_update_step = max(
        int(row["action_head_update_step"]) for row in food_rows
    )
    if max_update_step >= ANT_CAUSAL_ACTION_HEAD_FORMATION_MAX_UPDATE_STEPS:
        raise AlignmentFormationProtectionPrecheckError(
            "frozen failed checkpoint falls outside the declared formation "
            "window"
        )
    probe_digest = _stable_digest(active_payload)
    return {
        "schema_version": (
            ALIGNMENT_FORMATION_PROTECTION_PRECHECK_SCHEMA_VERSION
        ),
        "status": "PRECHECK_PASS",
        "read_only": True,
        "training_or_journal_write_performed": False,
        "source": {
            "review_report": {
                "path": _source_path(
                    review_report_path,
                    source_root=source_root,
                ),
                "sha256": _sha256_file(review_report_path),
            },
            "review_checkpoint_sha256": checkpoint_sha256,
        },
        "mechanism": {
            "owner": "vz-temporal causal action-head optimizer",
            "wiring": "active",
            "max_update_steps": (
                ANT_CAUSAL_ACTION_HEAD_FORMATION_MAX_UPDATE_STEPS
            ),
            "conflict_scale": (
                ANT_CAUSAL_ACTION_HEAD_FORMATION_CONFLICT_SCALE
            ),
            "domain_semantics_consumed": False,
            "rollback": {
                "wiring": "disabled",
                "max_update_steps": 0,
                "conflict_scale": 1.0,
            },
        },
        "probe": {
            "seed": probe_seed,
            "learning_writes_enabled": False,
            "active_digest": probe_digest,
            "disabled_digest": probe_digest,
            "byte_equivalent_forward": True,
            "food_rows": food_rows,
            "max_frozen_action_head_update_step": max_update_step,
        },
        "decision": {
            "l1b_mechanism_precheck": "PASS",
            "l1c_preregistration_may_be_created": True,
            "station_run_authorized": False,
            "station2_remains_unauthorized": True,
        },
        "limitations": [
            (
                "frozen forward equivalence proves rollback and checkpoint "
                "compatibility, not learned uplift"
            ),
            (
                "a fresh L1-C preregistration and empty journal are required "
                "before any station claim"
            ),
        ],
    }


__all__ = [
    "ALIGNMENT_FORMATION_PROTECTION_PRECHECK_SCHEMA_VERSION",
    "AlignmentFormationProtectionPrecheckError",
    "build_alignment_formation_protection_precheck",
]
