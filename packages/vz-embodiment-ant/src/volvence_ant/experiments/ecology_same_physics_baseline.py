"""Preregister the matched same-physics baseline for the v31 restart.

The old v31 station-1 journal was stopped against a historical threshold whose
physical curriculum no longer matched.  This packet removes that ambiguity:
both arms fork from one initial checkpoint, replay one frozen schedule, and
differ only in the typed environment-milestone temporal-switch wiring.

This module creates and validates the preregistration.  It deliberately does
not inspect results or choose thresholds after a run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

from volvence_zero.runtime import WiringLevel

from volvence_ant.evidence.runtime_profile import (
    ant_runtime_replay_rollout_config,
)
from volvence_ant.experiments.ecology_curriculum import (
    ECOLOGY_CURRICULUM_SCHEMA_VERSION,
)
from volvence_ant.experiments.ecology_p1 import (
    ECOLOGY_P1_FORMAL_MIN_ANTS,
    ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS,
    ECOLOGY_P1_FORMAL_MIN_LAYOUTS_PER_TIER,
    ECOLOGY_P1_FORMAL_MIN_TRAINING_ROUNDS,
    ECOLOGY_P1_PROGRESS_SCHEMA_VERSION,
    ECOLOGY_P1_SCHEMA_VERSION,
    EcologyP1Config,
    _curriculum_config,
    _fixed_schedule,
    _schedule_digest,
)
from volvence_ant.experiments.ecology_probe import (
    ECOLOGY_POST_PICKUP_MIN_FAMILY_PERSISTENCE_ACTIONS,
    ECOLOGY_POST_PICKUP_UTURN_MAX_SWITCH_LATENCY,
    ECOLOGY_POST_PICKUP_UTURN_MIN_CONSECUTIVE_APPROACH_STEPS,
    ECOLOGY_POST_PICKUP_UTURN_MIN_NET_PROGRESS,
)
from volvence_ant.substrate import AntSenseSchema


ECOLOGY_SAME_PHYSICS_BASELINE_SCHEMA_VERSION = (
    "digital-ant-ecology-same-physics-baseline-preregistration.v1"
)
ECOLOGY_SAME_PHYSICS_CANDIDATE_ARM = "learned"
ECOLOGY_SAME_PHYSICS_CONTROL_ARM = "typed_milestone_disabled"
ECOLOGY_SAME_PHYSICS_STATION1_EPISODES = 20
ECOLOGY_SAME_PHYSICS_STATION2_EPISODES = 30
ECOLOGY_SAME_PHYSICS_STATION3_EPISODES = 55
ECOLOGY_SAME_PHYSICS_PICKUP_NONINFERIORITY_RATIO = 0.8

_CAUSAL_FIELD = "environment_milestone_temporal_switch"
_SOURCE_PATHS = (
    "packages/vz-embodiment-ant/src/volvence_ant/evidence/runtime_profile.py",
    (
        "packages/vz-embodiment-ant/src/volvence_ant/experiments/"
        "ecology_curriculum.py"
    ),
    "packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_p1.py",
    (
        "packages/vz-embodiment-ant/src/volvence_ant/experiments/"
        "ecology_same_physics_baseline.py"
    ),
    (
        "packages/vz-embodiment-ant/src/volvence_ant/experiments/"
        "ecology_same_physics_run.py"
    ),
    "packages/vz-embodiment-ant/src/volvence_ant/runtime/ant_session.py",
    "packages/vz-runtime/src/volvence_zero/agent/response.py",
    "packages/vz-runtime/src/volvence_zero/agent/session.py",
    "packages/vz-runtime/src/volvence_zero/agent/session_observation.py",
    "packages/vz-runtime/src/volvence_zero/integration/final_wiring.py",
    "packages/vz-temporal/src/volvence_zero/joint_loop/runtime.py",
    "scripts/run_ant_ecology_same_physics_station1.py",
)


class EcologySamePhysicsBaselinePacketError(ValueError):
    """The preregistration does not match its frozen executable contract."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_ready(item) for item in sorted(value, key=repr)]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    return value


def _stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_binding(*, repo_root: Path, relative_path: str) -> dict[str, Any]:
    path = repo_root / relative_path
    payload = path.read_bytes()
    return {
        "path": relative_path,
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
    }


def _code_tree_binding(*, repo_root: Path) -> dict[str, Any]:
    paths = {
        path
        for root_name in ("packages", "scripts")
        for path in (repo_root / root_name).rglob("*.py")
        if "__pycache__" not in path.parts
    }
    paths.update(repo_root.rglob("pyproject.toml"))
    lock_path = repo_root / "uv.lock"
    if lock_path.is_file():
        paths.add(lock_path)
    digest = hashlib.sha256()
    total_size = 0
    for path in sorted(paths):
        relative = str(path.relative_to(repo_root))
        payload = path.read_bytes()
        total_size += len(payload)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return {
        "scope": "packages/**/*.py,scripts/**/*.py,**/pyproject.toml,uv.lock",
        "file_count": len(paths),
        "total_size_bytes": total_size,
        "sha256": digest.hexdigest(),
    }


def _rollout_payload(*, milestone_enabled: bool) -> dict[str, Any]:
    rollout = ant_runtime_replay_rollout_config(
        enable_sparse_exploration=True,
        enable_segment_credit=True,
        enable_prediction_error_switch=True,
        enable_environment_milestone_switch=milestone_enabled,
        sense_schema=AntSenseSchema.ECOLOGY_V2,
    )
    return {
        field.name: _json_ready(getattr(rollout, field.name))
        for field in fields(rollout)
    }


def _rollout_differences(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    keys = tuple(sorted(set(left).union(right)))
    return tuple(
        {
            "field": key,
            "candidate": left.get(key),
            "control": right.get(key),
        }
        for key in keys
        if left.get(key) != right.get(key)
    )


def _schedule_rows(config: EcologyP1Config) -> list[dict[str, Any]]:
    return _json_ready(
        [asdict(item) for item in _fixed_schedule(config)]
    )


def _station_blocks() -> list[dict[str, Any]]:
    return [
        {
            "name": "butter_near",
            "episode_start_inclusive": 0,
            "episode_end_exclusive": 5,
        },
        {
            "name": "burning_near",
            "episode_start_inclusive": 5,
            "episode_end_exclusive": 10,
        },
        {
            "name": "composite_near",
            "episode_start_inclusive": 10,
            "episode_end_exclusive": 15,
        },
        {
            "name": "forced_return",
            "episode_start_inclusive": 15,
            "episode_end_exclusive": 20,
        },
        {
            "name": "forced_approach",
            "episode_start_inclusive": 20,
            "episode_end_exclusive": 25,
        },
        {
            "name": "butter_medium",
            "episode_start_inclusive": 25,
            "episode_end_exclusive": 30,
        },
        {
            "name": "station3_remainder",
            "episode_start_inclusive": 30,
            "episode_end_exclusive": 55,
        },
    ]


def formal_same_physics_baseline_config(
    *,
    seed: int = 0,
) -> EcologyP1Config:
    """The only config accepted by a formal same-physics packet."""

    return EcologyP1Config(
        n_ants=ECOLOGY_P1_FORMAL_MIN_ANTS,
        temporal_latent_dim=16,
        training_rounds=ECOLOGY_P1_FORMAL_MIN_TRAINING_ROUNDS,
        evaluation_rounds=ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS,
        layouts_per_tier=ECOLOGY_P1_FORMAL_MIN_LAYOUTS_PER_TIER,
        seed=seed,
    )


def build_ecology_same_physics_baseline_packet(
    *,
    repo_root: Path,
    seed: int = 0,
) -> dict[str, Any]:
    """Build the deterministic preregistration before any result is read."""

    resolved_root = repo_root.resolve()
    config = formal_same_physics_baseline_config(seed=seed)
    curriculum = _curriculum_config(config)
    schedule = _fixed_schedule(config)
    schedule_rows = _schedule_rows(config)
    candidate_rollout = _rollout_payload(milestone_enabled=True)
    control_rollout = _rollout_payload(milestone_enabled=False)
    differences = _rollout_differences(candidate_rollout, control_rollout)
    if differences != (
        {
            "field": _CAUSAL_FIELD,
            "candidate": WiringLevel.ACTIVE.value,
            "control": WiringLevel.DISABLED.value,
        },
    ):
        raise EcologySamePhysicsBaselinePacketError(
            "same-physics arms must differ only in "
            f"{_CAUSAL_FIELD!r}; observed {differences!r}"
        )

    source_bindings = [
        _file_binding(repo_root=resolved_root, relative_path=path)
        for path in _SOURCE_PATHS
    ]
    matched_candidate = dict(candidate_rollout)
    matched_control = dict(control_rollout)
    matched_candidate.pop(_CAUSAL_FIELD)
    matched_control.pop(_CAUSAL_FIELD)
    matched_digest = _sha256(_stable_json_bytes(matched_candidate))
    if matched_digest != _sha256(_stable_json_bytes(matched_control)):
        raise EcologySamePhysicsBaselinePacketError(
            "matched rollout fields do not have identical digests"
        )

    return {
        "schema_version": ECOLOGY_SAME_PHYSICS_BASELINE_SCHEMA_VERSION,
        "status": "PREREGISTERED",
        "claim": (
            "Under the current physical curriculum, enabling the typed "
            "pickup/delivery milestone boundary is non-inferior on station-1 "
            "pickup reachability and improves the station-2 medium "
            "carrying-home outcome relative to an otherwise identical "
            "milestone-disabled control."
        ),
        "historical_baselines": {
            "decision_use": "EXCLUDED",
            "excluded_generations": ["v24", "v30", "old_v31_station1"],
            "reason": (
                "Historical journals do not provide a preregistered, "
                "same-source, same-physics causal control for this restart."
            ),
        },
        "schemas": {
            "curriculum": ECOLOGY_CURRICULUM_SCHEMA_VERSION,
            "p1_report": ECOLOGY_P1_SCHEMA_VERSION,
            "p1_progress": ECOLOGY_P1_PROGRESS_SCHEMA_VERSION,
        },
        "formal_config": _json_ready(asdict(config)),
        "physical_curriculum": _json_ready(asdict(curriculum)),
        "schedule": {
            "full_episode_count": len(schedule),
            "full_sha256": _schedule_digest(schedule),
            "station1_sha256": _sha256(
                _stable_json_bytes(
                    schedule_rows[:ECOLOGY_SAME_PHYSICS_STATION1_EPISODES]
                )
            ),
            "station2_prefix_sha256": _sha256(
                _stable_json_bytes(
                    schedule_rows[:ECOLOGY_SAME_PHYSICS_STATION2_EPISODES]
                )
            ),
            "rows": schedule_rows,
            "blocks": _station_blocks(),
        },
        "arms": {
            "shared_initial_checkpoint": True,
            "execution_order": [
                ECOLOGY_SAME_PHYSICS_CONTROL_ARM,
                ECOLOGY_SAME_PHYSICS_CANDIDATE_ARM,
            ],
            "candidate": {
                "name": ECOLOGY_SAME_PHYSICS_CANDIDATE_ARM,
                "rollout_config": candidate_rollout,
            },
            "control": {
                "name": ECOLOGY_SAME_PHYSICS_CONTROL_ARM,
                "rollout_config": control_rollout,
            },
            "allowed_differences": list(differences),
            "matched_fields_sha256": matched_digest,
        },
        "thresholds": {
            "station1": {
                "episode_end_exclusive": (
                    ECOLOGY_SAME_PHYSICS_STATION1_EPISODES
                ),
                "minimum_control_pickups_per_physical_block": 1,
                "candidate_aggregate_pickup_ratio_min": (
                    ECOLOGY_SAME_PHYSICS_PICKUP_NONINFERIORITY_RATIO
                ),
                "candidate_zero_pickup_block_forbidden_when_control_nonzero": (
                    True
                ),
                "deliveries_role": "DESCRIPTIVE_SPARSE_OBSERVATION",
                "candidate_post_pickup_switch_rate_min": 1.0,
                "candidate_switch_latency_actions_max": (
                    ECOLOGY_POST_PICKUP_UTURN_MAX_SWITCH_LATENCY
                ),
                "candidate_family_persistence_actions_min": (
                    ECOLOGY_POST_PICKUP_MIN_FAMILY_PERSISTENCE_ACTIONS
                ),
            },
            "station2": {
                "episode_end_exclusive": (
                    ECOLOGY_SAME_PHYSICS_STATION2_EPISODES
                ),
                "minimum_control_medium_pickups": 1,
                "candidate_medium_pickup_ratio_min": (
                    ECOLOGY_SAME_PHYSICS_PICKUP_NONINFERIORITY_RATIO
                ),
                "candidate_medium_deliveries_must_exceed_control": True,
                "candidate_carrying_home_alignment_must_be_positive": True,
                "candidate_uturn_net_progress_min": (
                    ECOLOGY_POST_PICKUP_UTURN_MIN_NET_PROGRESS
                ),
                "candidate_uturn_consecutive_approach_steps_min": (
                    ECOLOGY_POST_PICKUP_UTURN_MIN_CONSECUTIVE_APPROACH_STEPS
                ),
            },
            "station3": {
                "episode_end_exclusive": (
                    ECOLOGY_SAME_PHYSICS_STATION3_EPISODES
                ),
                "candidate_matched_block_pickup_ratio_min": (
                    ECOLOGY_SAME_PHYSICS_PICKUP_NONINFERIORITY_RATIO
                ),
                "far_results_role": "DESCRIPTIVE_D3",
            },
        },
        "decision_protocol": {
            "station1_go": (
                "All station1 thresholds pass; otherwise BLOCK and do not "
                "run episode 20."
            ),
            "station2_go": (
                "All station2 thresholds pass; otherwise BLOCK and do not "
                "run episode 30."
            ),
            "final_pass": (
                "Station1 and station2 are GO, station3 regression threshold "
                "passes, and the existing formal P1 gates pass."
            ),
            "no_posthoc_threshold_changes": True,
        },
        "execution_contract": {
            "device": "cpu",
            "numeric_precision": "float64",
            "new_empty_progress_directory_required": True,
            "old_v31_journal_resume_forbidden": True,
            "single_writer_lock_required": True,
            "source_bindings": source_bindings,
            "code_tree_binding": _code_tree_binding(
                repo_root=resolved_root
            ),
        },
    }


def validate_ecology_same_physics_baseline_packet(
    packet: Mapping[str, Any],
    *,
    repo_root: Path,
    check_source_bindings: bool = True,
) -> None:
    """Fail loudly on threshold, schedule, causal-arm or source drift."""

    if not isinstance(packet, Mapping):
        raise EcologySamePhysicsBaselinePacketError(
            "same-physics preregistration must be a JSON object"
        )
    seed_raw = packet.get("formal_config", {}).get("seed")
    if isinstance(seed_raw, bool) or not isinstance(seed_raw, int):
        raise EcologySamePhysicsBaselinePacketError(
            "formal_config.seed must be an integer"
        )
    expected = build_ecology_same_physics_baseline_packet(
        repo_root=repo_root,
        seed=seed_raw,
    )
    comparable = dict(packet)
    if not check_source_bindings:
        comparable = json.loads(json.dumps(packet))
        comparable["execution_contract"]["source_bindings"] = (
            expected["execution_contract"]["source_bindings"]
        )
    if comparable != expected:
        raise EcologySamePhysicsBaselinePacketError(
            "same-physics preregistration differs from the frozen executable "
            "contract; create a new version before running"
        )


__all__ = [
    "ECOLOGY_SAME_PHYSICS_BASELINE_SCHEMA_VERSION",
    "ECOLOGY_SAME_PHYSICS_CANDIDATE_ARM",
    "ECOLOGY_SAME_PHYSICS_CONTROL_ARM",
    "ECOLOGY_SAME_PHYSICS_PICKUP_NONINFERIORITY_RATIO",
    "ECOLOGY_SAME_PHYSICS_STATION1_EPISODES",
    "ECOLOGY_SAME_PHYSICS_STATION2_EPISODES",
    "ECOLOGY_SAME_PHYSICS_STATION3_EPISODES",
    "EcologySamePhysicsBaselinePacketError",
    "build_ecology_same_physics_baseline_packet",
    "formal_same_physics_baseline_config",
    "validate_ecology_same_physics_baseline_packet",
]
