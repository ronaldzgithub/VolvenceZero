"""Preregister the station1-v4 matched same-physics baseline.

The old v31 station-1 journal was stopped against a historical threshold whose
physical curriculum no longer matched.  This packet removes that ambiguity:
both arms fork from one initial checkpoint, replay one frozen schedule, and
differ only in the typed environment-milestone temporal-switch wiring.

The v4 generation is a legal mechanism-change restart: both arms carry the
L1-B temporal-owner formation guard, while the causal contrast remains the
typed milestone wiring alone.  It creates and validates the preregistration
without inspecting new results or choosing thresholds after a run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

from volvence_zero.runtime import WiringLevel

from volvence_ant.evidence.runtime_profile import (
    ANT_CAUSAL_ACTION_HEAD_FORMATION_CONFLICT_SCALE,
    ANT_CAUSAL_ACTION_HEAD_FORMATION_MAX_UPDATE_STEPS,
    ANT_TEMPORAL_POST_SWITCH_MIN_DWELL_ACTIONS,
    ant_runtime_replay_rollout_config,
)
from volvence_ant.experiments.alignment_formation_protection import (
    ALIGNMENT_FORMATION_PROTECTION_PRECHECK_SCHEMA_VERSION,
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
    "digital-ant-ecology-same-physics-baseline-preregistration.v3"
)
ECOLOGY_SAME_PHYSICS_EXPERIMENT_GENERATION = "station1-v4"
ECOLOGY_SAME_PHYSICS_CANDIDATE_ARM = "learned"
ECOLOGY_SAME_PHYSICS_CONTROL_ARM = "typed_milestone_disabled"
ECOLOGY_SAME_PHYSICS_STATION1_EPISODES = 20
ECOLOGY_SAME_PHYSICS_STATION2_EPISODES = 30
ECOLOGY_SAME_PHYSICS_STATION3_EPISODES = 55
ECOLOGY_SAME_PHYSICS_PICKUP_NONINFERIORITY_RATIO = 0.8

_CAUSAL_FIELD = "environment_milestone_temporal_switch"
_FORMATION_PRECHECK_PATH = (
    "research/ant/results/ecology_recovery/same_physics_baseline/"
    "alignment_formation_protection_precheck.v1.json"
)
_SOURCE_PATHS = (
    "packages/vz-embodiment-ant/src/volvence_ant/evidence/runtime_profile.py",
    (
        "packages/vz-embodiment-ant/src/volvence_ant/experiments/"
        "ecology_curriculum.py"
    ),
    "packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_p1.py",
    (
        "packages/vz-embodiment-ant/src/volvence_ant/experiments/"
        "alignment_formation_protection.py"
    ),
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
    "packages/vz-temporal/src/volvence_zero/internal_rl/sandbox.py",
    (
        "packages/vz-temporal/src/volvence_zero/internal_rl/"
        "torch_causal_ppo.py"
    ),
    "packages/vz-temporal/src/volvence_zero/temporal/interface.py",
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


def _formation_precheck_binding(*, repo_root: Path) -> dict[str, Any]:
    path = repo_root / _FORMATION_PRECHECK_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EcologySamePhysicsBaselinePacketError(
            "station1-v4 requires the frozen L1-B formation precheck"
        ) from error
    required = {
        "schema_version": (
            ALIGNMENT_FORMATION_PROTECTION_PRECHECK_SCHEMA_VERSION
        ),
        "status": "PRECHECK_PASS",
        "read_only": True,
        "training_or_journal_write_performed": False,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise EcologySamePhysicsBaselinePacketError(
                "L1-B formation precheck is not eligible for station1-v4: "
                f"field={field}, expected={expected!r}, "
                f"actual={payload.get(field)!r}"
            )
    decision = payload.get("decision")
    if not isinstance(decision, Mapping) or (
        decision.get("l1c_preregistration_may_be_created") is not True
        or decision.get("station_run_authorized") is not False
        or decision.get("station2_remains_unauthorized") is not True
    ):
        raise EcologySamePhysicsBaselinePacketError(
            "L1-B formation precheck does not authorize L1-C preregistration"
        )
    probe = payload.get("probe")
    if not isinstance(probe, Mapping) or (
        probe.get("learning_writes_enabled") is not False
        or probe.get("active_digest") != probe.get("disabled_digest")
        or probe.get("byte_equivalent_forward") is not True
    ):
        raise EcologySamePhysicsBaselinePacketError(
            "L1-B formation precheck lacks no-write rollback equivalence"
        )
    return {
        **_file_binding(
            repo_root=repo_root,
            relative_path=_FORMATION_PRECHECK_PATH,
        ),
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "active_digest": probe["active_digest"],
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
    expected_formation = {
        "internal_rl_causal_action_head_formation_protection": (
            WiringLevel.ACTIVE.value
        ),
        "internal_rl_causal_action_head_formation_max_update_steps": (
            ANT_CAUSAL_ACTION_HEAD_FORMATION_MAX_UPDATE_STEPS
        ),
        "internal_rl_causal_action_head_formation_conflict_scale": (
            ANT_CAUSAL_ACTION_HEAD_FORMATION_CONFLICT_SCALE
        ),
    }
    for arm_name, rollout in (
        (ECOLOGY_SAME_PHYSICS_CANDIDATE_ARM, candidate_rollout),
        (ECOLOGY_SAME_PHYSICS_CONTROL_ARM, control_rollout),
    ):
        observed = {
            field: rollout.get(field) for field in expected_formation
        }
        if observed != expected_formation:
            raise EcologySamePhysicsBaselinePacketError(
                "station1-v4 requires the frozen L1-B formation profile in "
                f"both arms; arm={arm_name!r}, observed={observed!r}"
            )
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
        "experiment_generation": (
            ECOLOGY_SAME_PHYSICS_EXPERIMENT_GENERATION
        ),
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
        "reopening_basis": {
            "prior_station1_generation": "station1-v3",
            "prior_verdict": "BLOCK_ALIGNMENT_3_OF_4",
            "mechanism_change": (
                "vz-temporal causal action-head formation protection"
            ),
            "formation_profile": expected_formation,
            "l1b_precheck": _formation_precheck_binding(
                repo_root=resolved_root
            ),
            "threshold_change": "NONE",
            "seed_only_rerun": False,
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
                "family_persistence_counting": (
                    "The switch action counts as action 1; any later beta "
                    "switch ends survival even when the selected family label "
                    "is unchanged."
                ),
                "post_switch_min_dwell_actions": (
                    ANT_TEMPORAL_POST_SWITCH_MIN_DWELL_ACTIONS
                ),
                "candidate_food_alignment_direct_station2_bodies": (
                    config.n_ants
                ),
                "food_alignment_review_authorized": False,
                "food_alignment_probe_seed_offset": 700_003,
                "food_alignment_review_episode_count": 5,
                "food_alignment_review_stage": "butter-near",
                "food_alignment_review_reprobe_required_bodies": (
                    config.n_ants
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
                "All station1 thresholds, including food alignment 4/4, "
                "pass; otherwise BLOCK and do not run episode 20."
            ),
            "station1_alignment": (
                "Station1-v4 requires 4/4 aligned food bodies at the frozen "
                "20-episode checkpoint. The prior five-episode alignment "
                "review path is exhausted and forbidden for this generation."
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
        "authorization": {
            "station1_run_authorized": True,
            "station1_max_episode_end_exclusive": (
                ECOLOGY_SAME_PHYSICS_STATION1_EPISODES
            ),
            "alignment_review_authorized": False,
            "station2_authorized_before_station1_go": False,
            "p1_authorized_before_station2_go": False,
            "p2_authorized_before_station2_go": False,
        },
        "execution_contract": {
            "device": "cpu",
            "numeric_precision": "float64",
            "isolated_source_snapshot_required": True,
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
        comparable["execution_contract"]["code_tree_binding"] = (
            expected["execution_contract"]["code_tree_binding"]
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
    "ECOLOGY_SAME_PHYSICS_EXPERIMENT_GENERATION",
    "ECOLOGY_SAME_PHYSICS_PICKUP_NONINFERIORITY_RATIO",
    "ECOLOGY_SAME_PHYSICS_STATION1_EPISODES",
    "ECOLOGY_SAME_PHYSICS_STATION2_EPISODES",
    "ECOLOGY_SAME_PHYSICS_STATION3_EPISODES",
    "EcologySamePhysicsBaselinePacketError",
    "build_ecology_same_physics_baseline_packet",
    "formal_same_physics_baseline_config",
    "validate_ecology_same_physics_baseline_packet",
]
