"""Model-free Product Horizon source-v3 campaign-input admission.

The source and reactive-environment owners already exist in
``lifeform-domain-emogpt``.  This module is only the offline admission owner:
it freezes their public and sealed projections, materializes every
action-conditioned outcome branch before a future campaign, and derives a
fail-closed development-tier verdict.  It does not execute a campaign.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
from collections.abc import Mapping, Sequence
from lifeform_domain_emogpt.lab.contracts import canonical_json, sha256_json
from lifeform_domain_emogpt.lab.environment import REACTIVE_ENVIRONMENT_VERSION
from lifeform_domain_emogpt.lab.relationship_product_pilot_source import (
    RelationshipProductPilotEvaluatorBundle,
)
from lifeform_domain_emogpt.lab.relationship_product_pilot_source_v2 import (
    RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3,
    build_relationship_product_pilot_environment,
    build_relationship_product_pilot_evaluator_bundle,
    build_relationship_product_pilot_public_view,
    load_relationship_product_pilot_source_protocol,
    relationship_product_pilot_source_protocol_path,
)
from lifeform_domain_emogpt.relationship_action_contracts import RELATIONSHIP_ACTIONS


SOURCE_ADMISSION_PROTOCOL_SCHEMA_VERSION = (
    "relationship-product-source-campaign-admission-protocol.v1"
)
SOURCE_ADMISSION_COMMITMENTS_SCHEMA_VERSION = (
    "relationship-product-source-action-counterfactual-commitments.v1"
)
SOURCE_ADMISSION_ROOT_MANIFEST_SCHEMA_VERSION = (
    "relationship-product-source-admission-root-manifest.v1"
)
SOURCE_ADMISSION_COMPARISON_SCHEMA_VERSION = (
    "relationship-product-source-admission-comparison.v1"
)
SOURCE_ADMISSION_MANIFEST_SCHEMA_VERSION = (
    "relationship-product-source-campaign-admission-manifest.v1"
)

_PROTOCOL_FILENAME = "relationship_product_source_v3_campaign_admission_v1.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_MATERIALIZATION_FILES = (
    "manifest.json",
    "protocol.json",
    "public/source_plan.json",
    "sealed/action_counterfactual_commitments.json",
    "sealed/evaluator_bundle.json",
)
_CLAIM_CEILING = {
    "campaign_execution_authorized": False,
    "campaign_runtime_order_verified": False,
    "formal_evidence_authorized": False,
    "integrated_horizon_authorized": False,
    "appendable_effect": False,
    "readable_effect": False,
    "learnable_effect": False,
    "steerable_effect": False,
    "four_able_complete": False,
    "human_validation_complete": False,
    "production_active": False,
    "fresh_process_independence_proven": False,
    "model_output_count": 0,
}


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def relationship_product_source_admission_protocol_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "protocols" / _PROTOCOL_FILENAME


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return (canonical_json(payload) + "\n").encode("utf-8")


def _object_pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def _parse_json_bytes(raw: bytes, *, source: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_pairs_no_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return value


def _read_json(path: pathlib.Path, *, source: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{source} must be a regular non-symlink file")
    return _parse_json_bytes(path.read_bytes(), source=source)


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    source: str,
) -> None:
    missing = sorted(expected - set(payload))
    extra = sorted(set(payload) - expected)
    if missing or extra:
        raise ValueError(
            f"{source} fields drifted; missing={missing}, extra={extra}"
        )


def _require_sha256(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return text


def _safe_repo_path(value: object, field_name: str) -> pathlib.PurePosixPath:
    text = _text(value, field_name)
    path = pathlib.PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ValueError(f"{field_name} must be a normalized repo-relative POSIX path")
    return path


def _validate_claim_ceiling(claims: Mapping[str, object]) -> None:
    expected = {"campaign_input_admission_may_be_derived", *_CLAIM_CEILING}
    _exact_keys(claims, expected, source="protocol.claims")
    if _boolean(
        claims["campaign_input_admission_may_be_derived"],
        "claims.campaign_input_admission_may_be_derived",
    ) is not True:
        raise ValueError("protocol must allow only mechanical campaign input admission")
    for key, expected_value in _CLAIM_CEILING.items():
        if claims[key] != expected_value:
            raise ValueError(f"protocol claim ceiling drifted at {key}")


def _validate_direct_execution_closure(
    rows: Sequence[object],
    *,
    repo_root: pathlib.Path,
) -> None:
    if len(rows) != 9:
        raise ValueError("direct execution closure must contain exactly nine pinned files")
    observed: set[str] = set()
    for index, item in enumerate(rows):
        row = _mapping(item, f"direct_execution_closure[{index}]")
        _exact_keys(
            row,
            {"path", "raw_bytes", "raw_sha256"},
            source=f"direct_execution_closure[{index}]",
        )
        relative = _safe_repo_path(row["path"], f"direct_execution_closure[{index}].path")
        if relative.as_posix() in observed:
            raise ValueError("direct execution closure contains a duplicate path")
        observed.add(relative.as_posix())
        path = repo_root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"direct execution closure path is not a regular file: {relative}")
        raw = path.read_bytes()
        if len(raw) != _integer(row["raw_bytes"], f"closure[{index}].raw_bytes"):
            raise ValueError(f"direct execution closure byte count drifted: {relative}")
        if _sha256_bytes(raw) != _require_sha256(
            row["raw_sha256"], f"closure[{index}].raw_sha256"
        ):
            raise ValueError(f"direct execution closure SHA-256 drifted: {relative}")


def load_relationship_product_source_admission_protocol(
    protocol_path: pathlib.Path | None = None,
) -> tuple[dict[str, object], str]:
    """Load and validate the frozen development-tier admission protocol."""

    path = pathlib.Path(protocol_path or relationship_product_source_admission_protocol_path())
    raw = path.read_bytes()
    payload = _parse_json_bytes(raw, source="source admission protocol")
    _exact_keys(
        payload,
        {
            "schema_version",
            "evidence_tier",
            "owner",
            "source",
            "direct_execution_closure",
            "inventory",
            "randomness",
            "replay",
            "claims",
            "claim_boundary",
        },
        source="source admission protocol",
    )
    if payload["schema_version"] != SOURCE_ADMISSION_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("source admission protocol schema drifted")
    if payload["evidence_tier"] != "development":
        raise ValueError("source admission is development-tier only")
    if payload["owner"] != "lifeform_evolution.relationship_product_source_admission":
        raise ValueError("source admission owner drifted")
    _text(payload["claim_boundary"], "claim_boundary")

    source = _mapping(payload["source"], "source")
    _exact_keys(
        source,
        {
            "schema_version",
            "protocol_raw_sha256",
            "protocol_raw_bytes",
            "protocol_sha256",
            "public_plan_sha256",
            "sealed_bundle_sha256",
        },
        source="source",
    )
    if source["schema_version"] != RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3:
        raise ValueError("source admission must pin source-v3")
    source_protocol_path = relationship_product_pilot_source_protocol_path(
        RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3
    )
    source_raw = source_protocol_path.read_bytes()
    if len(source_raw) != _integer(source["protocol_raw_bytes"], "source.protocol_raw_bytes"):
        raise ValueError("source-v3 protocol byte count drifted")
    if _sha256_bytes(source_raw) != _require_sha256(
        source["protocol_raw_sha256"], "source.protocol_raw_sha256"
    ):
        raise ValueError("source-v3 protocol raw SHA-256 drifted")

    source_owner = load_relationship_product_pilot_source_protocol(source_protocol_path)
    public = build_relationship_product_pilot_public_view(source_owner)
    evaluator = build_relationship_product_pilot_evaluator_bundle(source_owner)
    if source_owner.protocol_sha256 != _require_sha256(
        source["protocol_sha256"], "source.protocol_sha256"
    ):
        raise ValueError("source-v3 canonical protocol identity drifted")
    if public.public_plan_sha256 != _require_sha256(
        source["public_plan_sha256"], "source.public_plan_sha256"
    ):
        raise ValueError("source-v3 public plan identity drifted")
    if evaluator.sealed_bundle_sha256 != _require_sha256(
        source["sealed_bundle_sha256"], "source.sealed_bundle_sha256"
    ):
        raise ValueError("source-v3 sealed bundle identity drifted")

    closure = _sequence(payload["direct_execution_closure"], "direct_execution_closure")
    _validate_direct_execution_closure(closure, repo_root=_repo_root())

    inventory = _mapping(payload["inventory"], "inventory")
    _exact_keys(
        inventory,
        {
            "subject_count",
            "onboarding_session_count",
            "decision_count",
            "action_order",
            "action_counterfactual_commitment_count",
        },
        source="inventory",
    )
    action_order = tuple(
        _text(item, f"inventory.action_order[{index}]")
        for index, item in enumerate(_sequence(inventory["action_order"], "inventory.action_order"))
    )
    expected_action_order = tuple(action.value for action in RELATIONSHIP_ACTIONS)
    expected_counts = (8, 32, 192, len(RELATIONSHIP_ACTIONS) * 192)
    observed_counts = (
        _integer(inventory["subject_count"], "inventory.subject_count"),
        _integer(inventory["onboarding_session_count"], "inventory.onboarding_session_count"),
        _integer(inventory["decision_count"], "inventory.decision_count"),
        _integer(
            inventory["action_counterfactual_commitment_count"],
            "inventory.action_counterfactual_commitment_count",
        ),
    )
    if action_order != expected_action_order or observed_counts != expected_counts:
        raise ValueError("source admission inventory contract drifted")

    randomness = _mapping(payload["randomness"], "randomness")
    _exact_keys(
        randomness,
        {
            "seed_owner",
            "selected_action_is_in_draw_hash",
            "common_random_number_design",
            "arm_identity_affects_source_or_environment_seed",
            "all_action_branches_sealed_before_campaign",
        },
        source="randomness",
    )
    if randomness != {
        "seed_owner": "sealed evaluator decision environment_seed",
        "selected_action_is_in_draw_hash": True,
        "common_random_number_design": False,
        "arm_identity_affects_source_or_environment_seed": False,
        "all_action_branches_sealed_before_campaign": True,
    }:
        raise ValueError("source admission randomness contract drifted")

    replay = _mapping(payload["replay"], "replay")
    if replay != {
        "cooperative_materializer_process_count": 2,
        "cooperative_comparison_process_count": 1,
        "byte_exact_required": True,
        "process_independence_security_claim": False,
    }:
        raise ValueError("source admission replay contract drifted")
    _validate_claim_ceiling(_mapping(payload["claims"], "claims"))
    return payload, sha256_json(payload)


def _evaluator_payload(bundle: RelationshipProductPilotEvaluatorBundle) -> dict[str, object]:
    return {
        "schema_version": bundle.schema_version,
        "protocol_sha256": bundle.protocol_sha256,
        "cohort_id": bundle.cohort_id,
        "onboarding_sessions": [item.__dict__ for item in bundle.onboarding_sessions],
        "decision_sessions": [item.__dict__ for item in bundle.decision_sessions],
        "preferred_action_probabilities": list(bundle.preferred_action_probabilities),
        "nonpreferred_stay_probabilities": list(bundle.nonpreferred_stay_probabilities),
        "nonpreferred_space_probabilities": list(bundle.nonpreferred_space_probabilities),
        "neutral_noop_probabilities": list(bundle.neutral_noop_probabilities),
        "evaluation_or_judge_feedback_to_learning": (
            bundle.evaluation_or_judge_feedback_to_learning
        ),
    }


def build_relationship_product_source_action_commitments(
    evaluator: RelationshipProductPilotEvaluatorBundle,
) -> dict[str, object]:
    """Seal all action-conditioned branches with the existing environment owner."""

    environments = {
        subject_id: build_relationship_product_pilot_environment(
            evaluator,
            subject_id=subject_id,
        )
        for subject_id in sorted({item.subject_id for item in evaluator.decision_sessions})
    }
    commitments: list[dict[str, object]] = []
    for decision in evaluator.decision_sessions:
        environment = environments[decision.subject_id]
        for action in RELATIONSHIP_ACTIONS:
            settled = environment.settle(
                scene_id=decision.scene_id,
                decision_id=decision.decision_id,
                action=action,
                seed=decision.environment_seed,
            )
            distribution = environment.distribution_for(
                scene_id=decision.scene_id,
                action=action,
            )
            if distribution != settled.outcome_distribution:
                raise ValueError("environment settlement distribution drifted")
            core: dict[str, object] = {
                "subject_id": decision.subject_id,
                "world_clone_id": decision.world_clone_id,
                "session_id": decision.session_id,
                "decision_id": decision.decision_id,
                "decision_index": decision.decision_index,
                "scene_id": decision.scene_id,
                "phase_id": decision.phase_id,
                "stage_id": decision.stage_id,
                "domain_id": decision.domain_id,
                "condition_id": decision.condition_id,
                "policy_id": decision.policy_id,
                "preferred_action_id": decision.preferred_action_id,
                "environment_seed": decision.environment_seed,
                "selected_action_id": action.value,
                "outcome_distribution": distribution.to_payload(),
                "deterministic_draw": settled.deterministic_draw,
                "typed_outcome_id": settled.typed_outcome.value,
                "rendered_user_reaction": settled.rendered_user_reaction,
                "environment_evidence_ref": settled.environment_evidence_ref,
                "environment_version": settled.environment_version,
            }
            commitments.append({"commitment_id": sha256_json(core), **core})
    expected_count = len(evaluator.decision_sessions) * len(RELATIONSHIP_ACTIONS)
    if len(commitments) != expected_count:
        raise ValueError("action commitment count drifted")
    coverage = {
        (item["decision_id"], item["selected_action_id"])
        for item in commitments
    }
    if len(coverage) != expected_count:
        raise ValueError("action commitment coverage is not one row per decision and action")
    return {
        "schema_version": SOURCE_ADMISSION_COMMITMENTS_SCHEMA_VERSION,
        "source_protocol_sha256": evaluator.protocol_sha256,
        "sealed_evaluator_bundle_sha256": evaluator.sealed_bundle_sha256,
        "environment_version": REACTIVE_ENVIRONMENT_VERSION,
        "randomness_contract": {
            "seed_owner": "sealed evaluator decision environment_seed",
            "selected_action_is_in_draw_hash": True,
            "common_random_number_design": False,
            "arm_identity_affects_source_or_environment_seed": False,
        },
        "action_order": [action.value for action in RELATIONSHIP_ACTIONS],
        "decision_count": len(evaluator.decision_sessions),
        "commitment_count": len(commitments),
        "commitments": commitments,
    }


def _file_entries(files: Mapping[str, bytes]) -> list[dict[str, object]]:
    return [
        {
            "path": path,
            "raw_bytes": len(raw),
            "raw_sha256": _sha256_bytes(raw),
        }
        for path, raw in sorted(files.items())
    ]


def build_relationship_product_source_admission_materialization(
    protocol_path: pathlib.Path | None = None,
) -> dict[str, bytes]:
    """Build one deterministic, location-independent admission root in memory."""

    path = pathlib.Path(protocol_path or relationship_product_source_admission_protocol_path())
    protocol, protocol_id = load_relationship_product_source_admission_protocol(path)
    source_owner = load_relationship_product_pilot_source_protocol()
    public = build_relationship_product_pilot_public_view(source_owner)
    evaluator = build_relationship_product_pilot_evaluator_bundle(source_owner)
    commitments = build_relationship_product_source_action_commitments(evaluator)
    source = _mapping(protocol["source"], "source")
    if public.public_plan_sha256 != source["public_plan_sha256"]:
        raise ValueError("materialized public plan drifted from protocol")
    if evaluator.sealed_bundle_sha256 != source["sealed_bundle_sha256"]:
        raise ValueError("materialized sealed evaluator drifted from protocol")
    files: dict[str, bytes] = {
        "protocol.json": path.read_bytes(),
        "public/source_plan.json": _canonical_bytes(public.to_sut_payload()),
        "sealed/evaluator_bundle.json": _canonical_bytes(_evaluator_payload(evaluator)),
        "sealed/action_counterfactual_commitments.json": _canonical_bytes(commitments),
    }
    manifest_core: dict[str, object] = {
        "schema_version": SOURCE_ADMISSION_ROOT_MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "source_protocol_sha256": evaluator.protocol_sha256,
        "public_plan_sha256": public.public_plan_sha256,
        "sealed_bundle_sha256": evaluator.sealed_bundle_sha256,
        "subject_count": len(public.subjects),
        "onboarding_session_count": len(evaluator.onboarding_sessions),
        "decision_count": len(evaluator.decision_sessions),
        "action_counterfactual_commitment_count": commitments["commitment_count"],
        "files": _file_entries(files),
        "claims": {
            "materialization_complete": True,
            "campaign_input_admitted": False,
            **_CLAIM_CEILING,
        },
    }
    manifest = {"artifact_id": sha256_json(manifest_core), **manifest_core}
    files["manifest.json"] = _canonical_bytes(manifest)
    return files


def _write_create_only(path: pathlib.Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def materialize_relationship_product_source_admission(
    output_dir: pathlib.Path,
    *,
    protocol_path: pathlib.Path | None = None,
) -> dict[str, object]:
    """Create one deterministic root; an existing path is never overwritten."""

    root = pathlib.Path(output_dir)
    if root.exists():
        raise FileExistsError(f"source admission root is create-only: {root}")
    root.mkdir(parents=True, exist_ok=False)
    files = build_relationship_product_source_admission_materialization(protocol_path)
    for relative, raw in sorted(files.items()):
        _write_create_only(root.joinpath(*pathlib.PurePosixPath(relative).parts), raw)
    return _parse_json_bytes(files["manifest.json"], source="materialized manifest")


def _regular_file_map(root: pathlib.Path) -> dict[str, bytes]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("source admission root must be a regular directory")
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"source admission contains a non-regular file: {path}")
        relative = path.relative_to(root).as_posix()
        files[relative] = path.read_bytes()
    return files


def validate_relationship_product_source_admission_materialization(
    output_dir: pathlib.Path,
    *,
    expected_protocol_id: str,
    protocol_path: pathlib.Path | None = None,
) -> dict[str, object]:
    """Rebuild one root and require byte identity for every file."""

    protocol_id = _require_sha256(expected_protocol_id, "expected_protocol_id")
    expected = build_relationship_product_source_admission_materialization(protocol_path)
    actual = _regular_file_map(pathlib.Path(output_dir))
    if tuple(sorted(actual)) != _MATERIALIZATION_FILES:
        raise ValueError("source admission materialization file inventory drifted")
    if actual != expected:
        for path in sorted(set(actual) | set(expected)):
            if actual.get(path) != expected.get(path):
                raise ValueError(f"source admission materialization byte drifted: {path}")
        raise ValueError("source admission materialization drifted")
    manifest = _parse_json_bytes(actual["manifest.json"], source="root manifest")
    if manifest["protocol_id"] != protocol_id:
        raise ValueError("source admission root protocol identity drifted")
    return manifest


def _materialization_tree_sha256(files: Mapping[str, bytes]) -> str:
    return sha256_json(_file_entries(files))


def build_relationship_product_source_admission_comparison(
    replay_a: pathlib.Path,
    replay_b: pathlib.Path,
    *,
    expected_protocol_id: str,
    worker_a_pid: int,
    worker_b_pid: int,
    comparator_pid: int | None = None,
) -> dict[str, object]:
    """Validate two roots and derive the third-process comparison receipt."""

    pid_c = os.getpid() if comparator_pid is None else comparator_pid
    pids = (worker_a_pid, worker_b_pid, pid_c)
    if any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in pids):
        raise ValueError("source admission process ids must be positive integers")
    if len(set(pids)) != 3:
        raise ValueError("source admission requires two workers and one distinct comparator")
    manifest_a = validate_relationship_product_source_admission_materialization(
        replay_a,
        expected_protocol_id=expected_protocol_id,
    )
    manifest_b = validate_relationship_product_source_admission_materialization(
        replay_b,
        expected_protocol_id=expected_protocol_id,
    )
    files_a = _regular_file_map(pathlib.Path(replay_a))
    files_b = _regular_file_map(pathlib.Path(replay_b))
    if files_a != files_b:
        raise ValueError("independent source admission materializations are not byte exact")
    if manifest_a["artifact_id"] != manifest_b["artifact_id"]:
        raise ValueError("source admission materialization artifact identities differ")
    return {
        "schema_version": SOURCE_ADMISSION_COMPARISON_SCHEMA_VERSION,
        "protocol_id": _require_sha256(expected_protocol_id, "expected_protocol_id"),
        "worker_a_pid": worker_a_pid,
        "worker_b_pid": worker_b_pid,
        "comparator_pid": pid_c,
        "reported_distinct_process_count": 3,
        "process_ids_self_reported": True,
        "process_independence_proven": False,
        "materialization_file_count": len(files_a),
        "materialization_artifact_id": manifest_a["artifact_id"],
        "materialization_tree_sha256": _materialization_tree_sha256(files_a),
        "byte_exact": True,
        "status": "campaign_input_admitted_execution_not_authorized",
        "claims": {
            "campaign_input_admitted": True,
            **_CLAIM_CEILING,
        },
        "process_receipt_boundary": (
            "PIDs and exit success are local development receipts, not an OS security "
            "boundary or transferable proof of process independence."
        ),
    }


def write_relationship_product_source_admission_comparison(
    output_path: pathlib.Path,
    replay_a: pathlib.Path,
    replay_b: pathlib.Path,
    *,
    expected_protocol_id: str,
    worker_a_pid: int,
    worker_b_pid: int,
) -> dict[str, object]:
    receipt = build_relationship_product_source_admission_comparison(
        replay_a,
        replay_b,
        expected_protocol_id=expected_protocol_id,
        worker_a_pid=worker_a_pid,
        worker_b_pid=worker_b_pid,
    )
    _write_create_only(pathlib.Path(output_path), _canonical_bytes(receipt))
    return receipt


def _validate_comparison_receipt(
    payload: Mapping[str, object],
    *,
    replay_a: pathlib.Path,
    replay_b: pathlib.Path,
    expected_protocol_id: str,
) -> None:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "worker_a_pid",
        "worker_b_pid",
        "comparator_pid",
        "reported_distinct_process_count",
        "process_ids_self_reported",
        "process_independence_proven",
        "materialization_file_count",
        "materialization_artifact_id",
        "materialization_tree_sha256",
        "byte_exact",
        "status",
        "claims",
        "process_receipt_boundary",
    }
    _exact_keys(payload, expected_keys, source="comparison receipt")
    if payload["schema_version"] != SOURCE_ADMISSION_COMPARISON_SCHEMA_VERSION:
        raise ValueError("comparison receipt schema drifted")
    expected = build_relationship_product_source_admission_comparison(
        replay_a,
        replay_b,
        expected_protocol_id=expected_protocol_id,
        worker_a_pid=_integer(payload["worker_a_pid"], "worker_a_pid"),
        worker_b_pid=_integer(payload["worker_b_pid"], "worker_b_pid"),
        comparator_pid=_integer(payload["comparator_pid"], "comparator_pid"),
    )
    if dict(payload) != expected:
        raise ValueError("comparison receipt content drifted")


def finalize_relationship_product_source_admission(
    output_dir: pathlib.Path,
    *,
    implementation_git_commit: str,
) -> dict[str, object]:
    """Derive the outer create-only manifest after the third-process receipt."""

    root = pathlib.Path(output_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("source campaign admission manifest is create-only")
    commit = _text(implementation_git_commit, "implementation_git_commit")
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("implementation_git_commit must be a lowercase 40-hex commit")
    protocol, protocol_id = load_relationship_product_source_admission_protocol()
    comparison_path = root / "comparison.json"
    comparison = _read_json(comparison_path, source="comparison receipt")
    _validate_comparison_receipt(
        comparison,
        replay_a=root / "replay_a",
        replay_b=root / "replay_b",
        expected_protocol_id=protocol_id,
    )
    files = _regular_file_map(root)
    if "manifest.json" in files:
        raise ValueError("outer manifest must not exist before finalization")
    manifest_core: dict[str, object] = {
        "schema_version": SOURCE_ADMISSION_MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "implementation_git_commit": commit,
        "comparison_raw_sha256": _sha256_bytes(comparison_path.read_bytes()),
        "materialization_artifact_id": comparison["materialization_artifact_id"],
        "subject_count": _mapping(protocol["inventory"], "inventory")["subject_count"],
        "onboarding_session_count": _mapping(protocol["inventory"], "inventory")[
            "onboarding_session_count"
        ],
        "decision_count": _mapping(protocol["inventory"], "inventory")["decision_count"],
        "action_counterfactual_commitment_count": _mapping(
            protocol["inventory"], "inventory"
        )["action_counterfactual_commitment_count"],
        "files": _file_entries(files),
        "status": comparison["status"],
        "claims": comparison["claims"],
        "claim_boundary": protocol["claim_boundary"],
    }
    manifest = {"artifact_id": sha256_json(manifest_core), **manifest_core}
    _write_create_only(manifest_path, _canonical_bytes(manifest))
    return manifest


def validate_relationship_product_source_admission(
    output_dir: pathlib.Path,
    *,
    expected_protocol_id: str,
) -> dict[str, object]:
    """Validate the full two-worker/one-comparator admission artifact."""

    root = pathlib.Path(output_dir)
    protocol, protocol_id = load_relationship_product_source_admission_protocol()
    if protocol_id != _require_sha256(expected_protocol_id, "expected_protocol_id"):
        raise ValueError("source admission external expected protocol identity drifted")
    validate_relationship_product_source_admission_materialization(
        root / "replay_a",
        expected_protocol_id=protocol_id,
    )
    validate_relationship_product_source_admission_materialization(
        root / "replay_b",
        expected_protocol_id=protocol_id,
    )
    comparison_path = root / "comparison.json"
    comparison = _read_json(comparison_path, source="comparison receipt")
    _validate_comparison_receipt(
        comparison,
        replay_a=root / "replay_a",
        replay_b=root / "replay_b",
        expected_protocol_id=protocol_id,
    )
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path, source="source admission manifest")
    _exact_keys(
        manifest,
        {
            "artifact_id",
            "schema_version",
            "protocol_id",
            "implementation_git_commit",
            "comparison_raw_sha256",
            "materialization_artifact_id",
            "subject_count",
            "onboarding_session_count",
            "decision_count",
            "action_counterfactual_commitment_count",
            "files",
            "status",
            "claims",
            "claim_boundary",
        },
        source="source admission manifest",
    )
    if manifest["schema_version"] != SOURCE_ADMISSION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("source admission manifest schema drifted")
    if manifest["protocol_id"] != protocol_id:
        raise ValueError("source admission manifest protocol identity drifted")
    if _GIT_COMMIT.fullmatch(
        _text(manifest["implementation_git_commit"], "implementation_git_commit")
    ) is None:
        raise ValueError("source admission implementation commit is invalid")
    files_without_manifest = _regular_file_map(root)
    del files_without_manifest["manifest.json"]
    manifest_core = {
        key: value for key, value in manifest.items() if key != "artifact_id"
    }
    expected_core = {
        "schema_version": SOURCE_ADMISSION_MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "implementation_git_commit": manifest["implementation_git_commit"],
        "comparison_raw_sha256": _sha256_bytes(comparison_path.read_bytes()),
        "materialization_artifact_id": comparison["materialization_artifact_id"],
        "subject_count": _mapping(protocol["inventory"], "inventory")["subject_count"],
        "onboarding_session_count": _mapping(protocol["inventory"], "inventory")[
            "onboarding_session_count"
        ],
        "decision_count": _mapping(protocol["inventory"], "inventory")["decision_count"],
        "action_counterfactual_commitment_count": _mapping(
            protocol["inventory"], "inventory"
        )["action_counterfactual_commitment_count"],
        "files": _file_entries(files_without_manifest),
        "status": "campaign_input_admitted_execution_not_authorized",
        "claims": comparison["claims"],
        "claim_boundary": protocol["claim_boundary"],
    }
    if manifest_core != expected_core:
        raise ValueError("source admission manifest content drifted")
    if manifest["artifact_id"] != sha256_json(expected_core):
        raise ValueError("source admission manifest artifact identity drifted")
    return manifest


__all__ = [
    "SOURCE_ADMISSION_COMMITMENTS_SCHEMA_VERSION",
    "SOURCE_ADMISSION_COMPARISON_SCHEMA_VERSION",
    "SOURCE_ADMISSION_MANIFEST_SCHEMA_VERSION",
    "SOURCE_ADMISSION_PROTOCOL_SCHEMA_VERSION",
    "SOURCE_ADMISSION_ROOT_MANIFEST_SCHEMA_VERSION",
    "build_relationship_product_source_action_commitments",
    "build_relationship_product_source_admission_comparison",
    "build_relationship_product_source_admission_materialization",
    "finalize_relationship_product_source_admission",
    "load_relationship_product_source_admission_protocol",
    "materialize_relationship_product_source_admission",
    "relationship_product_source_admission_protocol_path",
    "validate_relationship_product_source_admission",
    "validate_relationship_product_source_admission_materialization",
    "write_relationship_product_source_admission_comparison",
]
