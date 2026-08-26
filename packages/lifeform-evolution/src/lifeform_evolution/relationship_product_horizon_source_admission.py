"""Development admission for the 112-root Product Horizon source-v4.

The source and reactive-environment owners live in ``lifeform-domain-emogpt``.
This module only persists their public/sealed projections and every canonical
action-conditioned outcome before a future campaign.  Admission is create-only
and model-free; it does not execute a campaign or establish an ability effect.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence

from lifeform_domain_emogpt.lab.contracts import canonical_json, sha256_json
from lifeform_domain_emogpt.lab.environment import (
    REACTIVE_ENVIRONMENT_VERSION,
    ReactiveRelationshipEnvironment,
)
from lifeform_domain_emogpt.lab.relationship_product_horizon_source_v4 import (
    RELATIONSHIP_PRODUCT_HORIZON_SOURCE_SCHEMA_VERSION,
    HorizonEvaluatorDecisionSession,
    RelationshipProductHorizonEvaluatorBundle,
    build_relationship_product_horizon_environment,
    build_relationship_product_horizon_evaluator_bundle,
    build_relationship_product_horizon_public_view,
    load_relationship_product_horizon_source_protocol,
    relationship_product_horizon_source_protocol_path,
)
from lifeform_domain_emogpt.relationship_action_contracts import (
    RELATIONSHIP_ACTIONS,
    RelationshipAction,
)


HORIZON_SOURCE_ADMISSION_PROTOCOL_SCHEMA_VERSION = (
    "relationship-product-horizon-source-admission-protocol.v1"
)
HORIZON_SOURCE_ADMISSION_COMMITMENTS_SCHEMA_VERSION = (
    "relationship-product-horizon-action-counterfactual-commitments.v1"
)
HORIZON_SOURCE_ADMISSION_MANIFEST_SCHEMA_VERSION = (
    "relationship-product-horizon-source-admission-manifest.v1"
)

_PROTOCOL_FILENAME = "relationship_product_horizon_source_v4_campaign_admission_v1.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_MATERIALIZATION_FILES = (
    "manifest.json",
    "protocol.json",
    "public/source_plan.json",
    "sealed/action_counterfactual_commitment_index.json",
    "sealed/evaluator_bundle.json",
    "source/source_protocol.json",
)
_EXPECTED_CLOSURE_PATHS = (
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/relationship_product_horizon_source_v4.py",
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab_protocols/relationship_product_horizon_source_v4.json",
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/environment.py",
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/dataset.py",
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/contracts.py",
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/relationship_action_contracts.py",
    "packages/vz-contracts/src/volvence_zero/dialogue_trace.py",
)
_CLAIM_CEILING = {
    "campaign_execution_authorized": False,
    "campaign_runtime_order_verified": False,
    "collection_forced_action_schedule_frozen": False,
    "reader_input_materialized": False,
    "reader_qualified": False,
    "theta0_materialized": False,
    "model_execution_authorized": False,
    "cuda_execution_authorized": False,
    "formal_evidence_authorized": False,
    "unseen_single_axis_evidence": False,
    "integrated_horizon_authorized": False,
    "appendable_effect": False,
    "readable_effect": False,
    "learnable_effect": False,
    "steerable_effect": False,
    "four_able_complete": False,
    "human_validation_complete": False,
    "human_sample_claimed": False,
    "production_active": False,
    "process_independence_proven": False,
    "source_v3_admission_inherited": False,
    "common_random_number_design": False,
    "direction_variance_icc_or_cost_estimated": False,
    "model_output_count": 0,
    "cuda_invocation_count": 0,
}
_ADMISSION_TRUE_CLAIMS = {
    "materialization_complete": True,
    "pre_manifest_byte_rebuild_verified": True,
    "source_identity_verified": True,
    "public_evaluator_join_verified": True,
    "cross_root_inventory_verified": True,
    "action_branch_coverage_complete": True,
    "deterministic_environment_replay_verified": True,
    "campaign_input_admitted": True,
}
_COMMITMENT_PREIMAGE_FIELDS = (
    "source_protocol_id",
    "sealed_evaluator_bundle_sha256",
    "subject_id",
    "dataset_fingerprint",
    "decision_id",
    "scene_id",
    "environment_seed",
    "selected_action_id",
    "outcome_distribution",
    "deterministic_draw",
    "typed_outcome_id",
    "rendered_user_reaction",
    "environment_evidence_ref",
    "environment_version",
)


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def relationship_product_horizon_source_admission_protocol_path() -> pathlib.Path:
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


def _exact_keys(payload: Mapping[str, object], expected: set[str], *, source: str) -> None:
    missing = sorted(expected - set(payload))
    extra = sorted(set(payload) - expected)
    if missing or extra:
        raise ValueError(f"{source} fields drifted; missing={missing}, extra={extra}")


def _require_sha256(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return text


def _require_git_commit(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    if _GIT_COMMIT.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a lowercase 40-hex commit")
    return text


def _safe_repo_path(value: object, field_name: str) -> pathlib.PurePosixPath:
    text = _text(value, field_name)
    path = pathlib.PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ValueError(f"{field_name} must be a normalized repo-relative POSIX path")
    return path


def _validate_direct_execution_closure(rows: Sequence[object]) -> None:
    if len(rows) != len(_EXPECTED_CLOSURE_PATHS):
        raise ValueError("source-v4 direct execution closure inventory drifted")
    observed_paths: list[str] = []
    for index, item in enumerate(rows):
        row = _mapping(item, f"direct_execution_closure[{index}]")
        _exact_keys(
            row,
            {"path", "raw_bytes", "raw_sha256"},
            source=f"direct_execution_closure[{index}]",
        )
        relative = _safe_repo_path(row["path"], f"direct_execution_closure[{index}].path")
        observed_paths.append(relative.as_posix())
        path = _repo_root().joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"direct execution closure path is not a regular file: {relative}")
        raw = path.read_bytes()
        if len(raw) != _integer(row["raw_bytes"], f"closure[{index}].raw_bytes"):
            raise ValueError(f"direct execution closure byte count drifted: {relative}")
        if _sha256_bytes(raw) != _require_sha256(
            row["raw_sha256"], f"closure[{index}].raw_sha256"
        ):
            raise ValueError(f"direct execution closure SHA-256 drifted: {relative}")
    if tuple(observed_paths) != _EXPECTED_CLOSURE_PATHS:
        raise ValueError("source-v4 direct execution closure path order drifted")


def _validate_claim_ceiling(claims: Mapping[str, object]) -> None:
    _exact_keys(
        claims,
        {"campaign_input_admission_may_be_derived", *_CLAIM_CEILING},
        source="protocol.claims",
    )
    if _boolean(
        claims["campaign_input_admission_may_be_derived"],
        "claims.campaign_input_admission_may_be_derived",
    ) is not True:
        raise ValueError("protocol must allow mechanical campaign-input admission")
    for key, expected in _CLAIM_CEILING.items():
        if claims[key] != expected:
            raise ValueError(f"protocol claim ceiling drifted at {key}")


def load_relationship_product_horizon_source_admission_protocol(
    protocol_path: pathlib.Path | None = None,
) -> tuple[dict[str, object], str]:
    """Load and validate the frozen model-free source-v4 admission protocol."""

    path = pathlib.Path(
        protocol_path or relationship_product_horizon_source_admission_protocol_path()
    )
    raw = path.read_bytes()
    if b"\r" in raw:
        raise ValueError("source-v4 admission protocol must be LF-only")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("source-v4 admission protocol must end in exactly one LF")
    payload = _parse_json_bytes(raw, source="source-v4 admission protocol")
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
        source="source-v4 admission protocol",
    )
    if payload["schema_version"] != HORIZON_SOURCE_ADMISSION_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("source-v4 admission protocol schema drifted")
    if payload["evidence_tier"] != "development":
        raise ValueError("source-v4 admission is development-tier only")
    if payload["owner"] != "lifeform_evolution.relationship_product_horizon_source_admission":
        raise ValueError("source-v4 admission owner drifted")
    _text(payload["claim_boundary"], "claim_boundary")

    source = _mapping(payload["source"], "source")
    _exact_keys(
        source,
        {
            "schema_version",
            "protocol_raw_sha256",
            "protocol_raw_bytes",
            "protocol_id",
            "public_plan_sha256",
            "sealed_bundle_sha256",
        },
        source="source",
    )
    if source["schema_version"] != RELATIONSHIP_PRODUCT_HORIZON_SOURCE_SCHEMA_VERSION:
        raise ValueError("admission protocol must pin source-v4")
    source_path = relationship_product_horizon_source_protocol_path()
    source_raw = source_path.read_bytes()
    if len(source_raw) != _integer(source["protocol_raw_bytes"], "source.protocol_raw_bytes"):
        raise ValueError("source-v4 protocol byte count drifted")
    if _sha256_bytes(source_raw) != _require_sha256(
        source["protocol_raw_sha256"], "source.protocol_raw_sha256"
    ):
        raise ValueError("source-v4 protocol raw SHA-256 drifted")
    source_protocol = load_relationship_product_horizon_source_protocol(source_path)
    public = build_relationship_product_horizon_public_view(source_protocol)
    evaluator = build_relationship_product_horizon_evaluator_bundle(source_protocol)
    if source_protocol.protocol_id != _require_sha256(source["protocol_id"], "source.protocol_id"):
        raise ValueError("source-v4 canonical protocol identity drifted")
    if public.public_plan_sha256 != _require_sha256(
        source["public_plan_sha256"], "source.public_plan_sha256"
    ):
        raise ValueError("source-v4 public plan identity drifted")
    if evaluator.sealed_bundle_sha256 != _require_sha256(
        source["sealed_bundle_sha256"], "source.sealed_bundle_sha256"
    ):
        raise ValueError("source-v4 sealed bundle identity drifted")

    _validate_direct_execution_closure(
        _sequence(payload["direct_execution_closure"], "direct_execution_closure")
    )
    inventory = _mapping(payload["inventory"], "inventory")
    _exact_keys(
        inventory,
        {
            "root_count",
            "onboarding_session_count",
            "collection_decision_count",
            "evaluation_decision_count",
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
    expected_counts = (112, 448, 896, 4_480, 5_376, 16_128)
    observed_counts = (
        _integer(inventory["root_count"], "inventory.root_count"),
        _integer(inventory["onboarding_session_count"], "inventory.onboarding_session_count"),
        _integer(inventory["collection_decision_count"], "inventory.collection_decision_count"),
        _integer(inventory["evaluation_decision_count"], "inventory.evaluation_decision_count"),
        _integer(inventory["decision_count"], "inventory.decision_count"),
        _integer(
            inventory["action_counterfactual_commitment_count"],
            "inventory.action_counterfactual_commitment_count",
        ),
    )
    if action_order != tuple(action.value for action in RELATIONSHIP_ACTIONS):
        raise ValueError("source-v4 admission action order drifted")
    if observed_counts != expected_counts:
        raise ValueError("source-v4 admission inventory contract drifted")

    randomness = _mapping(payload["randomness"], "randomness")
    if randomness != {
        "seed_owner": "sealed evaluator decision environment_seed",
        "selected_action_is_in_draw_hash": True,
        "common_random_number_design": False,
        "arm_identity_affects_source_or_environment_seed": False,
        "all_action_branches_sealed_before_campaign": True,
    }:
        raise ValueError("source-v4 admission randomness contract drifted")
    replay = _mapping(payload["replay"], "replay")
    if replay != {
        "materialization_count": 1,
        "pre_manifest_full_rebuild_required": True,
        "consumer_validate_existing_required": True,
        "byte_exact_required": True,
        "process_independence_security_claim": False,
    }:
        raise ValueError("source-v4 admission replay contract drifted")
    _validate_claim_ceiling(_mapping(payload["claims"], "claims"))
    return payload, sha256_json(payload)


def _build_action_commitment(
    evaluator: RelationshipProductHorizonEvaluatorBundle,
    decision: HorizonEvaluatorDecisionSession,
    environment: ReactiveRelationshipEnvironment,
    action: RelationshipAction,
    *,
    sealed_bundle_sha256: str,
) -> dict[str, object]:
    distribution = environment.distribution_for(scene_id=decision.scene_id, action=action)
    settled = environment.settle(
        scene_id=decision.scene_id,
        decision_id=decision.decision_id,
        action=action,
        seed=decision.environment_seed,
    )
    if settled.outcome_distribution != distribution or settled.selected_action is not action:
        raise ValueError("source-v4 environment settlement drifted from selected action")
    preimage: dict[str, object] = {
        "source_protocol_id": evaluator.protocol_id,
        "sealed_evaluator_bundle_sha256": sealed_bundle_sha256,
        "subject_id": decision.subject_id,
        "dataset_fingerprint": environment.dataset_fingerprint,
        "decision_id": decision.decision_id,
        "scene_id": decision.scene_id,
        "environment_seed": decision.environment_seed,
        "selected_action_id": action.value,
        "outcome_distribution": distribution.to_payload(),
        "deterministic_draw": settled.deterministic_draw,
        "typed_outcome_id": settled.typed_outcome.value,
        "rendered_user_reaction": settled.rendered_user_reaction,
        "environment_evidence_ref": settled.environment_evidence_ref,
        "environment_version": settled.environment_version,
    }
    if tuple(preimage) != _COMMITMENT_PREIMAGE_FIELDS:
        raise ValueError("source-v4 action commitment preimage schema drifted")
    return {"commitment_id": sha256_json(preimage), "preimage": preimage}


def build_relationship_product_horizon_source_action_commitment(
    evaluator: RelationshipProductHorizonEvaluatorBundle,
    *,
    subject_id: str,
    decision_id: str,
    action: RelationshipAction,
) -> dict[str, object]:
    """Rebuild one selected-action commitment with its owner-defined preimage."""

    if not isinstance(action, RelationshipAction):
        raise TypeError("action must be a RelationshipAction")
    decisions = evaluator.sessions_for(_text(subject_id, "subject_id"))
    matches = tuple(item for item in decisions if item.decision_id == _text(decision_id, "decision_id"))
    if len(matches) != 1:
        raise KeyError(decision_id)
    environment = build_relationship_product_horizon_environment(
        evaluator,
        subject_id=subject_id,
    )
    return _build_action_commitment(
        evaluator,
        matches[0],
        environment,
        action,
        sealed_bundle_sha256=evaluator.sealed_bundle_sha256,
    )


def build_relationship_product_horizon_source_action_commitments(
    evaluator: RelationshipProductHorizonEvaluatorBundle,
) -> dict[str, object]:
    """Materialize all 5,376-by-3 digests through the existing environment owner."""

    environments = {
        manifest.subject_id: build_relationship_product_horizon_environment(
            evaluator,
            subject_id=manifest.subject_id,
        )
        for manifest in evaluator.root_manifests
    }
    decision_commitments: list[dict[str, object]] = []
    evidence_refs: set[str] = set()
    commitment_ids: set[str] = set()
    coverage: set[tuple[str, str]] = set()
    sealed_bundle_sha256 = evaluator.sealed_bundle_sha256
    for decision in evaluator.decision_sessions:
        environment = environments[decision.subject_id]
        branches: list[dict[str, str]] = []
        for action in RELATIONSHIP_ACTIONS:
            commitment = _build_action_commitment(
                evaluator,
                decision,
                environment,
                action,
                sealed_bundle_sha256=sealed_bundle_sha256,
            )
            preimage = _mapping(commitment["preimage"], "commitment.preimage")
            evidence_ref = _text(
                preimage["environment_evidence_ref"],
                "commitment.preimage.environment_evidence_ref",
            )
            if evidence_ref in evidence_refs:
                raise ValueError("source-v4 action branches must have unique evidence refs")
            evidence_refs.add(evidence_ref)
            commitment_id = _require_sha256(
                commitment["commitment_id"],
                "commitment.commitment_id",
            )
            if commitment_id in commitment_ids:
                raise ValueError("source-v4 action commitment identities must be unique")
            commitment_ids.add(commitment_id)
            branch_key = (decision.decision_id, action.value)
            if branch_key in coverage:
                raise ValueError("source-v4 action commitment coverage contains duplicates")
            coverage.add(branch_key)
            branches.append(
                {
                    "selected_action_id": action.value,
                    "commitment_id": commitment_id,
                }
            )
        decision_commitments.append(
            {
                "decision_id": decision.decision_id,
                "branches": branches,
            }
        )
    expected_count = len(evaluator.decision_sessions) * len(RELATIONSHIP_ACTIONS)
    if len(commitment_ids) != expected_count or len(coverage) != expected_count:
        raise ValueError("source-v4 action commitment coverage drifted")
    return {
        "schema_version": HORIZON_SOURCE_ADMISSION_COMMITMENTS_SCHEMA_VERSION,
        "source_protocol_id": evaluator.protocol_id,
        "sealed_evaluator_bundle_sha256": sealed_bundle_sha256,
        "environment_version": REACTIVE_ENVIRONMENT_VERSION,
        "randomness_contract": {
            "seed_owner": "sealed evaluator decision environment_seed",
            "selected_action_is_in_draw_hash": True,
            "common_random_number_design": False,
            "arm_identity_affects_source_or_environment_seed": False,
        },
        "commitment_hash_algorithm": "sha256_canonical_json_v1",
        "commitment_preimage_fields": list(_COMMITMENT_PREIMAGE_FIELDS),
        "action_order": [action.value for action in RELATIONSHIP_ACTIONS],
        "decision_count": len(evaluator.decision_sessions),
        "commitment_count": len(commitment_ids),
        "decision_branch_commitments": decision_commitments,
    }


def _file_entries(files: Mapping[str, bytes]) -> list[dict[str, object]]:
    return [
        {"path": path, "raw_bytes": len(raw), "raw_sha256": _sha256_bytes(raw)}
        for path, raw in sorted(files.items())
    ]


def build_relationship_product_horizon_source_admission_materialization(
    *,
    implementation_git_commit: str,
    protocol_path: pathlib.Path | None = None,
) -> dict[str, bytes]:
    """Build one deterministic and location-independent admission root in memory."""

    commit = _require_git_commit(implementation_git_commit, "implementation_git_commit")
    path = pathlib.Path(
        protocol_path or relationship_product_horizon_source_admission_protocol_path()
    )
    protocol, protocol_id = load_relationship_product_horizon_source_admission_protocol(path)
    source_protocol = load_relationship_product_horizon_source_protocol()
    public = build_relationship_product_horizon_public_view(source_protocol)
    evaluator = build_relationship_product_horizon_evaluator_bundle(source_protocol)
    commitments = build_relationship_product_horizon_source_action_commitments(evaluator)
    source = _mapping(protocol["source"], "source")
    if public.public_plan_sha256 != source["public_plan_sha256"]:
        raise ValueError("materialized source-v4 public plan drifted from protocol")
    if evaluator.sealed_bundle_sha256 != source["sealed_bundle_sha256"]:
        raise ValueError("materialized source-v4 evaluator drifted from protocol")
    files: dict[str, bytes] = {
        "protocol.json": path.read_bytes(),
        "source/source_protocol.json": relationship_product_horizon_source_protocol_path().read_bytes(),
        "public/source_plan.json": _canonical_bytes(public.to_sut_payload()),
        "sealed/evaluator_bundle.json": _canonical_bytes(evaluator.to_payload()),
        "sealed/action_counterfactual_commitment_index.json": _canonical_bytes(commitments),
    }
    inventory = _mapping(protocol["inventory"], "inventory")
    root_coverage = [item.to_payload() for item in evaluator.root_manifests]
    decision_coverage = [
        {
            "subject_id": item.subject_id,
            "decision_id": item.decision_id,
            "decision_index": item.decision_index,
            "scene_id": item.scene_id,
            "segment_id": item.segment_id,
        }
        for item in evaluator.decision_sessions
    ]
    branch_coverage: list[list[str]] = []
    for decision_index, item in enumerate(
        _sequence(commitments["decision_branch_commitments"], "decision_branch_commitments")
    ):
        decision_commitment = _mapping(item, f"decision_branch_commitments[{decision_index}]")
        decision_id = _text(
            decision_commitment["decision_id"],
            f"decision_branch_commitments[{decision_index}].decision_id",
        )
        for branch_index, branch in enumerate(
            _sequence(decision_commitment["branches"], f"branches[{decision_index}]")
        ):
            branch_payload = _mapping(branch, f"branches[{decision_index}][{branch_index}]")
            branch_coverage.append(
                [
                    decision_id,
                    _text(
                        branch_payload["selected_action_id"],
                        f"branches[{decision_index}][{branch_index}].selected_action_id",
                    ),
                ]
            )
    segment_counts = Counter(item.segment_id for item in evaluator.decision_sessions)
    manifest_core: dict[str, object] = {
        "schema_version": HORIZON_SOURCE_ADMISSION_MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "implementation_git_commit": commit,
        "source_protocol_id": evaluator.protocol_id,
        "public_plan_sha256": public.public_plan_sha256,
        "sealed_bundle_sha256": evaluator.sealed_bundle_sha256,
        "root_count": inventory["root_count"],
        "onboarding_session_count": inventory["onboarding_session_count"],
        "collection_decision_count": inventory["collection_decision_count"],
        "evaluation_decision_count": inventory["evaluation_decision_count"],
        "decision_count": inventory["decision_count"],
        "action_counterfactual_commitment_count": commitments["commitment_count"],
        "segment_decision_counts": dict(sorted(segment_counts.items())),
        "root_coverage_sha256": sha256_json(root_coverage),
        "decision_coverage_sha256": sha256_json(decision_coverage),
        "action_branch_coverage_sha256": sha256_json(branch_coverage),
        "files": _file_entries(files),
        "status": "campaign_input_admitted_execution_not_authorized",
        "claims": {**_ADMISSION_TRUE_CLAIMS, **_CLAIM_CEILING},
        "claim_boundary": protocol["claim_boundary"],
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


def materialize_relationship_product_horizon_source_admission(
    output_dir: pathlib.Path,
    *,
    implementation_git_commit: str,
    protocol_path: pathlib.Path | None = None,
) -> dict[str, object]:
    """Persist one create-only admission root."""

    root = pathlib.Path(output_dir)
    if root.exists():
        raise FileExistsError(f"source-v4 admission root is create-only: {root}")
    root.mkdir(parents=True, exist_ok=False)
    files = build_relationship_product_horizon_source_admission_materialization(
        implementation_git_commit=implementation_git_commit,
        protocol_path=protocol_path,
    )
    for relative, raw in sorted(files.items()):
        if relative == "manifest.json":
            continue
        _write_create_only(root.joinpath(*pathlib.PurePosixPath(relative).parts), raw)
    rebuilt = build_relationship_product_horizon_source_admission_materialization(
        implementation_git_commit=implementation_git_commit,
        protocol_path=protocol_path,
    )
    if rebuilt != files:
        raise ValueError("source-v4 admission pre-manifest semantic rebuild drifted")
    persisted = _regular_file_map(root)
    expected_pre_manifest = {
        relative: raw for relative, raw in files.items() if relative != "manifest.json"
    }
    if persisted != expected_pre_manifest:
        raise ValueError("source-v4 admission pre-manifest persisted bytes drifted")
    _write_create_only(root / "manifest.json", files["manifest.json"])
    return _parse_json_bytes(files["manifest.json"], source="source-v4 admission manifest")


def _regular_file_map(root: pathlib.Path) -> dict[str, bytes]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("source-v4 admission root must be a regular directory")
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"source-v4 admission contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"source-v4 admission contains a non-regular file: {path}")
        files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def validate_relationship_product_horizon_source_admission(
    output_dir: pathlib.Path,
    *,
    expected_protocol_id: str,
    expected_artifact_id: str,
    protocol_path: pathlib.Path | None = None,
) -> dict[str, object]:
    """Rebuild the complete root and require byte identity for every file."""

    required_protocol_id = _require_sha256(expected_protocol_id, "expected_protocol_id")
    required_artifact_id = _require_sha256(expected_artifact_id, "expected_artifact_id")
    path = pathlib.Path(
        protocol_path or relationship_product_horizon_source_admission_protocol_path()
    )
    _, protocol_id = load_relationship_product_horizon_source_admission_protocol(path)
    if protocol_id != required_protocol_id:
        raise ValueError("source-v4 admission external expected protocol identity drifted")
    root = pathlib.Path(output_dir)
    actual = _regular_file_map(root)
    if tuple(sorted(actual)) != _MATERIALIZATION_FILES:
        raise ValueError("source-v4 admission file inventory drifted")
    manifest = _read_json(root / "manifest.json", source="source-v4 admission manifest")
    if manifest.get("artifact_id") != required_artifact_id:
        raise ValueError("source-v4 admission external expected artifact identity drifted")
    implementation_commit = _require_git_commit(
        manifest.get("implementation_git_commit"),
        "manifest.implementation_git_commit",
    )
    expected = build_relationship_product_horizon_source_admission_materialization(
        implementation_git_commit=implementation_commit,
        protocol_path=path,
    )
    if actual != expected:
        for relative in sorted(set(actual) | set(expected)):
            if actual.get(relative) != expected.get(relative):
                raise ValueError(f"source-v4 admission byte drifted: {relative}")
        raise ValueError("source-v4 admission materialization drifted")
    if manifest.get("protocol_id") != protocol_id:
        raise ValueError("source-v4 admission manifest protocol identity drifted")
    return manifest


__all__ = [
    "HORIZON_SOURCE_ADMISSION_COMMITMENTS_SCHEMA_VERSION",
    "HORIZON_SOURCE_ADMISSION_MANIFEST_SCHEMA_VERSION",
    "HORIZON_SOURCE_ADMISSION_PROTOCOL_SCHEMA_VERSION",
    "build_relationship_product_horizon_source_action_commitment",
    "build_relationship_product_horizon_source_action_commitments",
    "build_relationship_product_horizon_source_admission_materialization",
    "load_relationship_product_horizon_source_admission_protocol",
    "materialize_relationship_product_horizon_source_admission",
    "relationship_product_horizon_source_admission_protocol_path",
    "validate_relationship_product_horizon_source_admission",
]
