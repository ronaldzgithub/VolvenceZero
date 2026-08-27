"""Development admission for the identity-disjoint Product Horizon source-v5.

The public/sealed source and reactive settlement remain owned by
``lifeform-domain-emogpt``.  This module only freezes their exact projections,
all action-conditioned branch commitments, and a closed reader-input exclusion
registry before a future campaign.  It is model-free and does not authorize a
campaign or establish any four-able effect.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
import pathlib
import re
import stat

from lifeform_domain_emogpt.lab.contracts import canonical_json, sha256_json
from lifeform_domain_emogpt.lab.relationship_product_horizon_source_v4 import (
    RELATIONSHIP_PRODUCT_HORIZON_EVALUATOR_SCHEMA_VERSION,
    RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_VIEW_SCHEMA_VERSION,
)
from lifeform_domain_emogpt.lab.relationship_product_horizon_source_v5 import (
    RELATIONSHIP_PRODUCT_HORIZON_SOURCE_V5_SCHEMA_VERSION,
    build_relationship_product_horizon_source_v5_projections,
    load_relationship_product_horizon_source_v5_protocol,
    relationship_product_horizon_source_v5_protocol_path,
    relationship_product_horizon_source_v5_reader_text_inventory,
)
from lifeform_domain_emogpt.relationship_action_contracts import RELATIONSHIP_ACTIONS

from lifeform_evolution.relationship_lab_product_model_adapters import (
    PrecomputedPublicEmbeddingTable,
)
from lifeform_evolution.relationship_product_horizon_source_admission import (
    HORIZON_SOURCE_ADMISSION_COMMITMENTS_SCHEMA_VERSION,
    build_relationship_product_horizon_source_action_commitments,
)


HORIZON_SOURCE_V5_ADMISSION_PROTOCOL_SCHEMA_VERSION = (
    "relationship-product-horizon-source-v5-admission-protocol.v1"
)
HORIZON_SOURCE_V5_ADMISSION_INVENTORY_SCHEMA_VERSION = (
    "relationship-product-horizon-source-v5-exact-disjoint-inventory.v1"
)
HORIZON_SOURCE_V5_ADMISSION_MANIFEST_SCHEMA_VERSION = (
    "relationship-product-horizon-source-v5-admission-manifest.v1"
)

_PROTOCOL_FILENAME = "relationship_product_horizon_source_v5_campaign_admission_v1.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_MATERIALIZATION_FILES = (
    "lineage/exact_disjoint_inventory.json",
    "manifest.json",
    "protocol.json",
    "public/source_plan.json",
    "sealed/action_counterfactual_commitment_index.json",
    "sealed/evaluator_bundle.json",
    "source/source_protocol.json",
)
_EXPECTED_CLOSURE_PATHS = (
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/relationship_product_horizon_source_v5.py",
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab_protocols/relationship_product_horizon_source_v5.json",
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/relationship_product_horizon_source_v4.py",
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab_protocols/relationship_product_horizon_source_v4.json",
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/environment.py",
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/dataset.py",
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/contracts.py",
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/relationship_action_contracts.py",
    "packages/vz-contracts/src/volvence_zero/dialogue_trace.py",
    "packages/lifeform-evolution/src/lifeform_evolution/relationship_lab_product_model_adapters.py",
    "packages/lifeform-evolution/src/lifeform_evolution/relationship_lab_product_baselines.py",
    "packages/lifeform-evolution/src/lifeform_evolution/relationship_product_horizon_source_admission.py",
)
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
_CLAIM_CEILING = {
    "campaign_protocol_frozen": False,
    "campaign_materialized": False,
    "campaign_execution_authorized": False,
    "campaign_runtime_order_verified": False,
    "preaction_durable_before_branch_verified": False,
    "actual_delivered_action_join_verified": False,
    "postaction_barrier_verified": False,
    "forecast_runtime_arm_blinding_verified": False,
    "forecast_runtime_scope_blinding_verified": False,
    "forecast_runtime_order_blinding_verified": False,
    "source_v5_embedding_table_materialized": False,
    "reader_fit_count": 0,
    "reader_qualified": False,
    "theta_handoff_materialized": False,
    "geometric_reachability_established": False,
    "credit_achievability_established": False,
    "treatment_reachability_admitted": False,
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
    "semantic_novelty_established": False,
    "dgp_independence_established": False,
    "statistical_independence_established": False,
    "process_independence_proven": False,
    "common_random_number_design": False,
    "a1_reactive_source_evidence_inherited": False,
    "a2_msc_budget_evidence_inherited": False,
    "model_output_count": 0,
    "cuda_invocation_count": 0,
    "rehearsal_execution_count": 0,
}
_ADMISSION_TRUE_CLAIMS = {
    "materialization_complete": True,
    "pre_manifest_byte_rebuild_verified": True,
    "source_v5_identity_verified": True,
    "public_evaluator_join_verified": True,
    "cross_root_inventory_verified": True,
    "source_v3_exact_disjoint_verified": True,
    "source_v4_exact_disjoint_verified": True,
    "frozen_adaptive_reader_registry_exact_disjoint_verified": True,
    "qualification_v5_subset_of_development_table_verified": True,
    "action_branch_coverage_complete": True,
    "deterministic_environment_replay_verified": True,
    "campaign_input_admitted": True,
}


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def relationship_product_horizon_source_v5_admission_protocol_path() -> pathlib.Path:
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
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"{source} contains non-finite JSON number: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return value


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
        raise ValueError("source-v5 direct execution closure inventory drifted")
    observed_paths: list[str] = []
    for index, item in enumerate(rows):
        row = _mapping(item, f"direct_execution_closure[{index}]")
        _exact_keys(
            row,
            {"path", "raw_bytes", "raw_sha256"},
            source=f"direct_execution_closure[{index}]",
        )
        relative = _safe_repo_path(row["path"], f"closure[{index}].path")
        observed_paths.append(relative.as_posix())
        path = _repo_root().joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"direct execution closure is not a regular file: {relative}")
        raw = path.read_bytes()
        if len(raw) != _integer(row["raw_bytes"], f"closure[{index}].raw_bytes"):
            raise ValueError(f"direct execution closure byte count drifted: {relative}")
        if _sha256_bytes(raw) != _require_sha256(
            row["raw_sha256"], f"closure[{index}].raw_sha256"
        ):
            raise ValueError(f"direct execution closure SHA-256 drifted: {relative}")
    if tuple(observed_paths) != _EXPECTED_CLOSURE_PATHS:
        raise ValueError("source-v5 direct execution closure path order drifted")


def _validate_reader_inventory_pin(
    row: Mapping[str, object],
    *,
    source: str,
    extra_keys: set[str],
) -> None:
    _exact_keys(
        row,
        {
            "reader_text_unique_count",
            "reader_text_inventory_sha256",
            *extra_keys,
        },
        source=source,
    )
    if _integer(row["reader_text_unique_count"], f"{source}.reader_text_unique_count") <= 0:
        raise ValueError(f"{source} reader inventory must be non-empty")
    _require_sha256(
        row["reader_text_inventory_sha256"],
        f"{source}.reader_text_inventory_sha256",
    )


def _validate_exclusion_inputs(value: object) -> None:
    inputs = _mapping(value, "exclusion_inputs")
    _exact_keys(
        inputs,
        {
            "source_v3_admission",
            "source_v4_admission",
            "adaptive_reader_input_registry",
        },
        source="exclusion_inputs",
    )
    for key in ("source_v3_admission", "source_v4_admission"):
        row = _mapping(inputs[key], f"exclusion_inputs.{key}")
        _validate_reader_inventory_pin(
            row,
            source=f"exclusion_inputs.{key}",
            extra_keys={
                "admission_protocol_id",
                "admission_artifact_id",
                "public_content_id",
                "public_raw_sha256",
                "reader_text_occurrence_count",
            },
        )
        for name in (
            "admission_protocol_id",
            "admission_artifact_id",
            "public_content_id",
            "public_raw_sha256",
        ):
            _require_sha256(row[name], f"exclusion_inputs.{key}.{name}")
        if _integer(
            row["reader_text_occurrence_count"],
            f"exclusion_inputs.{key}.reader_text_occurrence_count",
        ) <= 0:
            raise ValueError(f"exclusion_inputs.{key} occurrence count must be positive")

    registry = _mapping(
        inputs["adaptive_reader_input_registry"],
        "exclusion_inputs.adaptive_reader_input_registry",
    )
    _exact_keys(
        registry,
        {
            "scope",
            "future_input_requires_new_protocol",
            "source_v3_public_is_subset_of_development_reader_table",
            "source_v4_public_is_subset_of_development_reader_table",
            "development_reader_table",
            "attempt03_table",
            "qualification_v5_table_lineage_only",
            "union_unique_count",
            "union_inventory_sha256",
        },
        source="adaptive_reader_input_registry",
    )
    if registry["scope"] != "closed_product_horizon_registry_at_protocol_freeze":
        raise ValueError("adaptive reader registry scope drifted")
    for key in (
        "future_input_requires_new_protocol",
        "source_v3_public_is_subset_of_development_reader_table",
        "source_v4_public_is_subset_of_development_reader_table",
    ):
        if _boolean(registry[key], f"adaptive_reader_input_registry.{key}") is not True:
            raise ValueError(f"adaptive reader registry must require {key}")
    table_keys = {
        "development_reader_table": {
            "package_protocol_id",
            "package_artifact_id",
            "table_artifact_id",
            "table_raw_sha256",
        },
        "attempt03_table": {
            "campaign_protocol_id",
            "reobservation_artifact_id",
            "reobservation_raw_sha256",
            "table_artifact_id",
            "table_raw_sha256",
        },
        "qualification_v5_table_lineage_only": {
            "qualification_protocol_id",
            "execution_protocol_id",
            "prediction_manifest_artifact_id",
            "prediction_manifest_raw_sha256",
            "table_artifact_id",
            "table_raw_sha256",
            "exact_subset_of_development_reader_table",
        },
    }
    for key, extra in table_keys.items():
        row = _mapping(registry[key], f"adaptive_reader_input_registry.{key}")
        _validate_reader_inventory_pin(
            row,
            source=f"adaptive_reader_input_registry.{key}",
            extra_keys=extra,
        )
        for name in extra - {"exact_subset_of_development_reader_table"}:
            _require_sha256(row[name], f"adaptive_reader_input_registry.{key}.{name}")
        if "exact_subset_of_development_reader_table" in extra and _boolean(
            row["exact_subset_of_development_reader_table"],
            "qualification_v5_table_lineage_only.exact_subset_of_development_reader_table",
        ) is not True:
            raise ValueError("qualification-v5 table must be a development-table subset")
    if _integer(registry["union_unique_count"], "registry.union_unique_count") <= 0:
        raise ValueError("adaptive reader registry union must be non-empty")
    _require_sha256(registry["union_inventory_sha256"], "registry.union_inventory_sha256")


def _validate_claim_ceiling(claims: Mapping[str, object]) -> None:
    _exact_keys(
        claims,
        {
            "campaign_input_admission_may_be_derived",
            "exact_disjoint_admission_may_be_derived",
            *_CLAIM_CEILING,
        },
        source="protocol.claims",
    )
    for key in (
        "campaign_input_admission_may_be_derived",
        "exact_disjoint_admission_may_be_derived",
    ):
        if _boolean(claims[key], f"claims.{key}") is not True:
            raise ValueError(f"protocol must allow mechanical {key}")
    for key, expected in _CLAIM_CEILING.items():
        if claims[key] != expected:
            raise ValueError(f"protocol claim ceiling drifted at {key}")


def load_relationship_product_horizon_source_v5_admission_protocol(
    protocol_path: pathlib.Path | None = None,
) -> tuple[dict[str, object], str]:
    """Load and validate the frozen model-free source-v5 admission protocol."""

    path = pathlib.Path(
        protocol_path or relationship_product_horizon_source_v5_admission_protocol_path()
    )
    raw = path.read_bytes()
    if b"\r" in raw:
        raise ValueError("source-v5 admission protocol must be LF-only")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("source-v5 admission protocol must end in exactly one LF")
    payload = _parse_json_bytes(raw, source="source-v5 admission protocol")
    _exact_keys(
        payload,
        {
            "schema_version",
            "evidence_tier",
            "owner",
            "source",
            "exclusion_inputs",
            "direct_execution_closure",
            "inventory",
            "randomness",
            "replay",
            "claims",
            "claim_boundary",
        },
        source="source-v5 admission protocol",
    )
    if payload["schema_version"] != HORIZON_SOURCE_V5_ADMISSION_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("source-v5 admission protocol schema drifted")
    if payload["evidence_tier"] != "development":
        raise ValueError("source-v5 admission is development-tier only")
    if payload["owner"] != (
        "lifeform_evolution.relationship_product_horizon_source_v5_admission"
    ):
        raise ValueError("source-v5 admission owner drifted")
    _text(payload["claim_boundary"], "claim_boundary")

    source = _mapping(payload["source"], "source")
    _exact_keys(
        source,
        {
            "schema_version",
            "protocol_raw_sha256",
            "protocol_raw_bytes",
            "protocol_id",
            "public_view_schema_version",
            "evaluator_schema_version",
            "public_plan_sha256",
            "sealed_bundle_sha256",
            "reader_text_occurrence_count",
            "reader_text_unique_count",
            "reader_text_inventory_sha256",
        },
        source="source",
    )
    if source["schema_version"] != RELATIONSHIP_PRODUCT_HORIZON_SOURCE_V5_SCHEMA_VERSION:
        raise ValueError("admission protocol must pin source-v5")
    if source["public_view_schema_version"] != (
        RELATIONSHIP_PRODUCT_HORIZON_PUBLIC_VIEW_SCHEMA_VERSION
    ):
        raise ValueError("source-v5 public view schema drifted")
    if source["evaluator_schema_version"] != (
        RELATIONSHIP_PRODUCT_HORIZON_EVALUATOR_SCHEMA_VERSION
    ):
        raise ValueError("source-v5 evaluator schema drifted")
    source_path = relationship_product_horizon_source_v5_protocol_path()
    source_raw = source_path.read_bytes()
    if len(source_raw) != _integer(source["protocol_raw_bytes"], "source.protocol_raw_bytes"):
        raise ValueError("source-v5 protocol byte count drifted")
    if _sha256_bytes(source_raw) != _require_sha256(
        source["protocol_raw_sha256"], "source.protocol_raw_sha256"
    ):
        raise ValueError("source-v5 protocol raw SHA-256 drifted")
    source_protocol = load_relationship_product_horizon_source_v5_protocol(source_path)
    public, evaluator = build_relationship_product_horizon_source_v5_projections(
        source_protocol
    )
    inventory_rows = relationship_product_horizon_source_v5_reader_text_inventory(
        public,
        protocol=source_protocol,
    )
    occurrence_count = sum(
        len(root.onboarding_sessions) + len(root.decision_sessions)
        for root in public.roots
    )
    observed_source = {
        "protocol_id": source_protocol.protocol_id,
        "public_plan_sha256": public.public_plan_sha256,
        "sealed_bundle_sha256": evaluator.sealed_bundle_sha256,
        "reader_text_occurrence_count": occurrence_count,
        "reader_text_unique_count": len(inventory_rows),
        "reader_text_inventory_sha256": sha256_json(inventory_rows),
    }
    for key, observed in observed_source.items():
        if source[key] != observed:
            raise ValueError(f"source-v5 identity or inventory drifted at {key}")

    _validate_exclusion_inputs(payload["exclusion_inputs"])
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
    counts = (
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
    if counts != (112, 448, 896, 4_480, 5_376, 16_128):
        raise ValueError("source-v5 admission inventory contract drifted")
    action_order = tuple(
        _text(item, f"inventory.action_order[{index}]")
        for index, item in enumerate(
            _sequence(inventory["action_order"], "inventory.action_order")
        )
    )
    if action_order != tuple(action.value for action in RELATIONSHIP_ACTIONS):
        raise ValueError("source-v5 admission action order drifted")
    randomness = _mapping(payload["randomness"], "randomness")
    if randomness != {
        "seed_owner": "sealed evaluator decision environment_seed",
        "selected_action_is_in_draw_hash": True,
        "common_random_number_design": False,
        "arm_identity_affects_source_or_environment_seed": False,
        "all_action_branches_sealed_before_campaign": True,
    }:
        raise ValueError("source-v5 admission randomness contract drifted")
    replay = _mapping(payload["replay"], "replay")
    if replay != {
        "materialization_count": 1,
        "pre_manifest_full_rebuild_required": True,
        "consumer_validate_existing_required": True,
        "external_expected_protocol_and_artifact_ids_required": True,
        "byte_exact_required": True,
        "read_only_validation_required": True,
        "process_independence_security_claim": False,
    }:
        raise ValueError("source-v5 admission replay contract drifted")
    _validate_claim_ceiling(_mapping(payload["claims"], "claims"))
    return payload, sha256_json(payload)


def _is_reparse_point(st: os.stat_result) -> bool:
    if os.name != "nt":
        return False
    return bool(st.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _regular_file_map(root: pathlib.Path) -> dict[str, bytes]:
    if not os.path.lexists(root):
        raise ValueError(f"artifact root does not exist: {root}")
    root_stat = root.lstat()
    if _is_reparse_point(root_stat) or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("artifact root must be a regular non-reparse directory")
    files: dict[str, bytes] = {}

    def visit(directory: pathlib.Path) -> None:
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda item: item.name)
        for entry in ordered:
            path = pathlib.Path(entry.path)
            path_stat = path.lstat()
            if entry.is_symlink() or _is_reparse_point(path_stat):
                raise ValueError(f"artifact contains a symlink or reparse point: {path}")
            if stat.S_ISDIR(path_stat.st_mode):
                visit(path)
                continue
            if not stat.S_ISREG(path_stat.st_mode):
                raise ValueError(f"artifact contains a non-regular file: {path}")
            if path_stat.st_nlink != 1:
                raise ValueError(f"artifact contains a hard-linked file: {path}")
            relative = path.relative_to(root).as_posix()
            if relative in files:
                raise ValueError(f"artifact contains a case-colliding path: {relative}")
            files[relative] = path.read_bytes()

    visit(root)
    return files


def _verify_manifest_envelope(
    root: pathlib.Path,
    *,
    expected_protocol_id: str,
    expected_artifact_id: str,
    expected_schema_version: str,
    source: str,
) -> tuple[dict[str, object], dict[str, bytes]]:
    files = _regular_file_map(pathlib.Path(root))
    raw_manifest = files.get("manifest.json")
    if raw_manifest is None:
        raise ValueError(f"{source} is missing manifest.json")
    manifest = _parse_json_bytes(raw_manifest, source=f"{source} manifest")
    if manifest.get("schema_version") != expected_schema_version:
        raise ValueError(f"{source} manifest schema drifted")
    if manifest.get("protocol_id") != _require_sha256(
        expected_protocol_id,
        f"{source}.expected_protocol_id",
    ):
        raise ValueError(f"{source} protocol identity drifted")
    if manifest.get("artifact_id") != _require_sha256(
        expected_artifact_id,
        f"{source}.expected_artifact_id",
    ):
        raise ValueError(f"{source} artifact identity drifted")
    core = {key: value for key, value in manifest.items() if key != "artifact_id"}
    if manifest["artifact_id"] != sha256_json(core):
        raise ValueError(f"{source} manifest content identity drifted")
    rows = _sequence(manifest.get("files"), f"{source}.files")
    expected_files: set[str] = set()
    for index, item in enumerate(rows):
        row = _mapping(item, f"{source}.files[{index}]")
        _exact_keys(
            row,
            {"path", "raw_bytes", "raw_sha256"},
            source=f"{source}.files[{index}]",
        )
        relative = _safe_repo_path(row["path"], f"{source}.files[{index}].path").as_posix()
        if relative in expected_files:
            raise ValueError(f"{source} manifest file paths must be unique")
        expected_files.add(relative)
        raw = files.get(relative)
        if raw is None:
            raise ValueError(f"{source} manifest file is missing: {relative}")
        if len(raw) != _integer(row["raw_bytes"], f"{source}.files[{index}].raw_bytes"):
            raise ValueError(f"{source} file byte count drifted: {relative}")
        if _sha256_bytes(raw) != _require_sha256(
            row["raw_sha256"], f"{source}.files[{index}].raw_sha256"
        ):
            raise ValueError(f"{source} file SHA-256 drifted: {relative}")
    if set(files) != {"manifest.json", *expected_files}:
        raise ValueError(f"{source} envelope contains missing or extra files")
    return manifest, files


def _reader_inventory_rows(texts: Sequence[str]) -> tuple[tuple[str, str], ...]:
    by_digest: dict[str, str] = {}
    for text in texts:
        value = _text(text, "reader text")
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        existing = by_digest.setdefault(digest, value)
        if existing != value:
            raise RuntimeError("SHA-256 collision in reader-text inventory")
    return tuple(sorted(by_digest.items()))


def _public_reader_texts(
    payload: Mapping[str, object],
    *,
    root_key: str,
    source: str,
) -> tuple[str, ...]:
    roots = _sequence(payload.get(root_key), f"{source}.{root_key}")
    texts: list[str] = []
    for root_index, value in enumerate(roots):
        root = _mapping(value, f"{source}.{root_key}[{root_index}]")
        for key, text_key in (
            ("onboarding_sessions", "user_utterance"),
            ("decision_sessions", "current_input"),
        ):
            for row_index, raw in enumerate(
                _sequence(root.get(key), f"{source}.{root_key}[{root_index}].{key}")
            ):
                row = _mapping(raw, f"{source}.{key}[{row_index}]")
                texts.append(_text(row.get(text_key), f"{source}.{key}.{text_key}"))
    return tuple(texts)


def _verify_inventory(
    *,
    texts: Sequence[str],
    pin: Mapping[str, object],
    occurrence_field: str | None,
    source: str,
) -> tuple[tuple[str, str], ...]:
    rows = _reader_inventory_rows(texts)
    if occurrence_field is not None and len(texts) != _integer(
        pin[occurrence_field], f"{source}.{occurrence_field}"
    ):
        raise ValueError(f"{source} reader occurrence count drifted")
    if len(rows) != _integer(
        pin["reader_text_unique_count"], f"{source}.reader_text_unique_count"
    ):
        raise ValueError(f"{source} reader unique count drifted")
    if sha256_json(rows) != _require_sha256(
        pin["reader_text_inventory_sha256"],
        f"{source}.reader_text_inventory_sha256",
    ):
        raise ValueError(f"{source} reader inventory identity drifted")
    return rows


def _load_pinned_embedding_table(
    path: pathlib.Path,
    *,
    pin: Mapping[str, object],
    source: str,
) -> tuple[PrecomputedPublicEmbeddingTable, tuple[tuple[str, str], ...]]:
    target = pathlib.Path(path)
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"{source} must be a regular non-symlink file")
    target_stat = target.lstat()
    if _is_reparse_point(target_stat) or target_stat.st_nlink != 1:
        raise ValueError(f"{source} must not be a reparse point or hard link")
    raw = target.read_bytes()
    if _sha256_bytes(raw) != _require_sha256(
        pin["table_raw_sha256"], f"{source}.table_raw_sha256"
    ):
        raise ValueError(f"{source} raw SHA-256 drifted")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source} must be strict UTF-8") from exc
    table = PrecomputedPublicEmbeddingTable.from_json(decoded)
    if table.artifact_id != _require_sha256(
        pin["table_artifact_id"], f"{source}.table_artifact_id"
    ):
        raise ValueError(f"{source} artifact identity drifted")
    rows = _verify_inventory(
        texts=tuple(record.text for record in table.records),
        pin=pin,
        occurrence_field=None,
        source=source,
    )
    return table, rows


def _load_pinned_content_receipt(
    path: pathlib.Path,
    *,
    pin: Mapping[str, object],
    raw_pin_field: str,
    artifact_pin_field: str,
    source: str,
) -> dict[str, object]:
    target = pathlib.Path(path)
    if not target.is_file() or target.is_symlink():
        raise ValueError(f"{source} must be a regular non-symlink file")
    target_stat = target.lstat()
    if _is_reparse_point(target_stat) or target_stat.st_nlink != 1:
        raise ValueError(f"{source} must not be a reparse point or hard link")
    raw = target.read_bytes()
    if _sha256_bytes(raw) != _require_sha256(
        pin[raw_pin_field], f"{source}.{raw_pin_field}"
    ):
        raise ValueError(f"{source} raw SHA-256 drifted")
    receipt = _parse_json_bytes(raw, source=source)
    expected_artifact = _require_sha256(
        pin[artifact_pin_field], f"{source}.{artifact_pin_field}"
    )
    if receipt.get("artifact_id") != expected_artifact:
        raise ValueError(f"{source} artifact identity drifted")
    core = {key: value for key, value in receipt.items() if key != "artifact_id"}
    if sha256_json(core) != expected_artifact:
        raise ValueError(f"{source} content identity drifted")
    return receipt


def _inventory_section(
    rows: tuple[tuple[str, str], ...],
    *,
    occurrence_count: int,
) -> dict[str, object]:
    return {
        "reader_text_occurrence_count": occurrence_count,
        "reader_text_unique_count": len(rows),
        "reader_text_inventory_sha256": sha256_json(rows),
        "rows": [list(item) for item in rows],
    }


def _build_exact_disjoint_inventory(
    *,
    protocol: Mapping[str, object],
    protocol_id: str,
    source_v5_rows: tuple[tuple[str, str], ...],
    source_v3_admission_root: pathlib.Path,
    source_v4_admission_root: pathlib.Path,
    development_reader_root: pathlib.Path,
    attempt03_embedding_table_path: pathlib.Path,
    attempt03_reobservation_path: pathlib.Path,
    qualification_v5_embedding_table_path: pathlib.Path,
) -> dict[str, object]:
    inputs = _mapping(protocol["exclusion_inputs"], "exclusion_inputs")
    v3_pin = _mapping(inputs["source_v3_admission"], "source_v3_admission")
    _, v3_files = _verify_manifest_envelope(
        pathlib.Path(source_v3_admission_root),
        expected_protocol_id=_text(v3_pin["admission_protocol_id"], "v3 protocol id"),
        expected_artifact_id=_text(v3_pin["admission_artifact_id"], "v3 artifact id"),
        expected_schema_version=(
            "relationship-product-source-campaign-admission-manifest.v1"
        ),
        source="source-v3 admission",
    )
    v3_public_raw = v3_files["replay_a/public/source_plan.json"]
    if _sha256_bytes(v3_public_raw) != v3_pin["public_raw_sha256"]:
        raise ValueError("source-v3 public raw SHA-256 drifted")
    v3_public = _parse_json_bytes(v3_public_raw, source="source-v3 public plan")
    if sha256_json(v3_public) != v3_pin["public_content_id"]:
        raise ValueError("source-v3 public content identity drifted")
    v3_texts = _public_reader_texts(
        v3_public,
        root_key="subjects",
        source="source-v3 public plan",
    )
    v3_rows = _verify_inventory(
        texts=v3_texts,
        pin=v3_pin,
        occurrence_field="reader_text_occurrence_count",
        source="source-v3",
    )

    v4_pin = _mapping(inputs["source_v4_admission"], "source_v4_admission")
    _, v4_files = _verify_manifest_envelope(
        pathlib.Path(source_v4_admission_root),
        expected_protocol_id=_text(v4_pin["admission_protocol_id"], "v4 protocol id"),
        expected_artifact_id=_text(v4_pin["admission_artifact_id"], "v4 artifact id"),
        expected_schema_version="relationship-product-horizon-source-admission-manifest.v1",
        source="source-v4 admission",
    )
    v4_public_raw = v4_files["public/source_plan.json"]
    if _sha256_bytes(v4_public_raw) != v4_pin["public_raw_sha256"]:
        raise ValueError("source-v4 public raw SHA-256 drifted")
    v4_public = _parse_json_bytes(v4_public_raw, source="source-v4 public plan")
    if sha256_json(v4_public) != v4_pin["public_content_id"]:
        raise ValueError("source-v4 public content identity drifted")
    v4_texts = _public_reader_texts(
        v4_public,
        root_key="roots",
        source="source-v4 public plan",
    )
    v4_rows = _verify_inventory(
        texts=v4_texts,
        pin=v4_pin,
        occurrence_field="reader_text_occurrence_count",
        source="source-v4",
    )

    registry = _mapping(
        inputs["adaptive_reader_input_registry"],
        "adaptive_reader_input_registry",
    )
    development_pin = _mapping(
        registry["development_reader_table"],
        "development_reader_table",
    )
    _verify_manifest_envelope(
        pathlib.Path(development_reader_root),
        expected_protocol_id=_text(
            development_pin["package_protocol_id"], "development protocol id"
        ),
        expected_artifact_id=_text(
            development_pin["package_artifact_id"], "development artifact id"
        ),
        expected_schema_version=(
            "relationship-product-horizon-development-reader-manifest.v1"
        ),
        source="development reader",
    )
    _, development_rows = _load_pinned_embedding_table(
        pathlib.Path(development_reader_root) / "embedding_table.json",
        pin=development_pin,
        source="development reader table",
    )
    attempt03_pin = _mapping(registry["attempt03_table"], "attempt03_table")
    attempt03_receipt = _load_pinned_content_receipt(
        pathlib.Path(attempt03_reobservation_path),
        pin=attempt03_pin,
        raw_pin_field="reobservation_raw_sha256",
        artifact_pin_field="reobservation_artifact_id",
        source="attempt03 table reobservation receipt",
    )
    if attempt03_receipt.get("protocol_id") != attempt03_pin["campaign_protocol_id"]:
        raise ValueError("attempt03 reobservation campaign protocol drifted")
    if attempt03_receipt.get("table_artifact_id") != attempt03_pin["table_artifact_id"]:
        raise ValueError("attempt03 reobservation table artifact drifted")
    if attempt03_receipt.get("table_raw_sha256") != attempt03_pin["table_raw_sha256"]:
        raise ValueError("attempt03 reobservation table raw identity drifted")
    _, attempt03_rows = _load_pinned_embedding_table(
        pathlib.Path(attempt03_embedding_table_path),
        pin=attempt03_pin,
        source="attempt03 table",
    )
    qualification_pin = _mapping(
        registry["qualification_v5_table_lineage_only"],
        "qualification_v5_table_lineage_only",
    )
    qualification_path = pathlib.Path(qualification_v5_embedding_table_path)
    qualification_manifest = _load_pinned_content_receipt(
        qualification_path.parent / "manifest.json",
        pin=qualification_pin,
        raw_pin_field="prediction_manifest_raw_sha256",
        artifact_pin_field="prediction_manifest_artifact_id",
        source="qualification-v5 prediction manifest",
    )
    if qualification_manifest.get("protocol_id") != qualification_pin[
        "qualification_protocol_id"
    ]:
        raise ValueError("qualification-v5 manifest protocol drifted")
    if qualification_manifest.get("execution_protocol_id") != qualification_pin[
        "execution_protocol_id"
    ]:
        raise ValueError("qualification-v5 execution protocol drifted")
    qualification_file_rows = [
        _mapping(value, "qualification-v5 manifest file")
        for value in _sequence(
            qualification_manifest.get("files"),
            "qualification-v5 manifest files",
        )
        if _mapping(value, "qualification-v5 manifest file").get("path")
        == "embedding_table.json"
    ]
    if len(qualification_file_rows) != 1:
        raise ValueError("qualification-v5 manifest table file lineage drifted")
    qualification_file = qualification_file_rows[0]
    if qualification_file.get("artifact_id") != qualification_pin["table_artifact_id"]:
        raise ValueError("qualification-v5 manifest table artifact drifted")
    if qualification_file.get("raw_sha256") != qualification_pin["table_raw_sha256"]:
        raise ValueError("qualification-v5 manifest table raw identity drifted")
    _, qualification_rows = _load_pinned_embedding_table(
        qualification_path,
        pin=qualification_pin,
        source="qualification-v5 table",
    )

    row_sets = {
        "source_v5": {text for _, text in source_v5_rows},
        "source_v3": {text for _, text in v3_rows},
        "source_v4": {text for _, text in v4_rows},
        "development_reader_table": {text for _, text in development_rows},
        "attempt03_table": {text for _, text in attempt03_rows},
        "qualification_v5_table": {text for _, text in qualification_rows},
    }
    if not row_sets["source_v3"] <= row_sets["development_reader_table"]:
        raise ValueError("source-v3 public inventory is not a development-table subset")
    if not row_sets["source_v4"] <= row_sets["development_reader_table"]:
        raise ValueError("source-v4 public inventory is not a development-table subset")
    if not row_sets["qualification_v5_table"] <= row_sets["development_reader_table"]:
        raise ValueError("qualification-v5 inventory is not a development-table subset")
    adaptive_union_rows = _reader_inventory_rows(
        tuple(
            row_sets["development_reader_table"] | row_sets["attempt03_table"]
        )
    )
    if len(adaptive_union_rows) != _integer(
        registry["union_unique_count"], "registry.union_unique_count"
    ):
        raise ValueError("adaptive reader registry union count drifted")
    if sha256_json(adaptive_union_rows) != registry["union_inventory_sha256"]:
        raise ValueError("adaptive reader registry union identity drifted")
    overlap_counts = {
        "source_v5_with_source_v3": len(
            row_sets["source_v5"] & row_sets["source_v3"]
        ),
        "source_v5_with_source_v4": len(
            row_sets["source_v5"] & row_sets["source_v4"]
        ),
        "source_v5_with_development_reader_table": len(
            row_sets["source_v5"] & row_sets["development_reader_table"]
        ),
        "source_v5_with_attempt03_table": len(
            row_sets["source_v5"] & row_sets["attempt03_table"]
        ),
        "source_v5_with_qualification_v5_table": len(
            row_sets["source_v5"] & row_sets["qualification_v5_table"]
        ),
        "source_v5_with_adaptive_reader_union": len(
            row_sets["source_v5"] & {text for _, text in adaptive_union_rows}
        ),
    }
    if any(overlap_counts.values()):
        raise ValueError(f"source-v5 reader text overlap detected: {overlap_counts}")
    core: dict[str, object] = {
        "schema_version": HORIZON_SOURCE_V5_ADMISSION_INVENTORY_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "registry_scope": registry["scope"],
        "future_input_requires_new_protocol": registry[
            "future_input_requires_new_protocol"
        ],
        "source_v5": _inventory_section(
            source_v5_rows,
            occurrence_count=_integer(
                _mapping(protocol["source"], "source")[
                    "reader_text_occurrence_count"
                ],
                "source.reader_text_occurrence_count",
            ),
        ),
        "source_v3": _inventory_section(v3_rows, occurrence_count=len(v3_texts)),
        "source_v4": _inventory_section(v4_rows, occurrence_count=len(v4_texts)),
        "development_reader_table": _inventory_section(
            development_rows,
            occurrence_count=len(development_rows),
        ),
        "attempt03_table": _inventory_section(
            attempt03_rows,
            occurrence_count=len(attempt03_rows),
        ),
        "qualification_v5_table_lineage_only": _inventory_section(
            qualification_rows,
            occurrence_count=len(qualification_rows),
        ),
        "adaptive_reader_union": {
            "members": ["development_reader_table", "attempt03_table"],
            **_inventory_section(
                adaptive_union_rows,
                occurrence_count=len(adaptive_union_rows),
            ),
        },
        "subset_relations": {
            "source_v3_public_is_subset_of_development_reader_table": True,
            "source_v4_public_is_subset_of_development_reader_table": True,
            "qualification_v5_is_subset_of_development_reader_table": True,
        },
        "full_string_overlap_counts": overlap_counts,
        "semantic_novelty_established": False,
        "dgp_independence_established": False,
        "statistical_independence_established": False,
    }
    return {"artifact_id": sha256_json(core), **core}


def _file_entries(files: Mapping[str, bytes]) -> list[dict[str, object]]:
    return [
        {"path": path, "raw_bytes": len(raw), "raw_sha256": _sha256_bytes(raw)}
        for path, raw in sorted(files.items())
    ]


def build_relationship_product_horizon_source_v5_admission_materialization(
    *,
    implementation_git_commit: str,
    source_v3_admission_root: pathlib.Path,
    source_v4_admission_root: pathlib.Path,
    development_reader_root: pathlib.Path,
    attempt03_embedding_table_path: pathlib.Path,
    attempt03_reobservation_path: pathlib.Path,
    qualification_v5_embedding_table_path: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> dict[str, bytes]:
    """Build one deterministic and location-independent admission root in memory."""

    commit = _require_git_commit(implementation_git_commit, "implementation_git_commit")
    path = pathlib.Path(
        protocol_path or relationship_product_horizon_source_v5_admission_protocol_path()
    )
    protocol, protocol_id = load_relationship_product_horizon_source_v5_admission_protocol(
        path
    )
    source_protocol = load_relationship_product_horizon_source_v5_protocol()
    public, evaluator = build_relationship_product_horizon_source_v5_projections(
        source_protocol
    )
    source_v5_rows = relationship_product_horizon_source_v5_reader_text_inventory(
        public,
        protocol=source_protocol,
    )
    disjoint_inventory = _build_exact_disjoint_inventory(
        protocol=protocol,
        protocol_id=protocol_id,
        source_v5_rows=source_v5_rows,
        source_v3_admission_root=source_v3_admission_root,
        source_v4_admission_root=source_v4_admission_root,
        development_reader_root=development_reader_root,
        attempt03_embedding_table_path=attempt03_embedding_table_path,
        attempt03_reobservation_path=attempt03_reobservation_path,
        qualification_v5_embedding_table_path=qualification_v5_embedding_table_path,
    )
    commitments = build_relationship_product_horizon_source_action_commitments(
        evaluator
    )
    if commitments.get("schema_version") != (
        HORIZON_SOURCE_ADMISSION_COMMITMENTS_SCHEMA_VERSION
    ):
        raise ValueError("source-v5 action commitment schema drifted")
    if commitments.get("source_protocol_id") != source_protocol.protocol_id:
        raise ValueError("source-v5 action commitments used another source protocol")
    if commitments.get("sealed_evaluator_bundle_sha256") != (
        evaluator.sealed_bundle_sha256
    ):
        raise ValueError("source-v5 action commitments used another evaluator bundle")
    if tuple(commitments.get("commitment_preimage_fields", ())) != (
        _COMMITMENT_PREIMAGE_FIELDS
    ):
        raise ValueError("source-v5 action commitment preimage schema drifted")
    source = _mapping(protocol["source"], "source")
    if public.public_plan_sha256 != source["public_plan_sha256"]:
        raise ValueError("source-v5 public plan drifted from admission protocol")
    if evaluator.sealed_bundle_sha256 != source["sealed_bundle_sha256"]:
        raise ValueError("source-v5 evaluator drifted from admission protocol")
    files: dict[str, bytes] = {
        "protocol.json": path.read_bytes(),
        "source/source_protocol.json": (
            relationship_product_horizon_source_v5_protocol_path().read_bytes()
        ),
        "public/source_plan.json": _canonical_bytes(public.to_sut_payload()),
        "sealed/evaluator_bundle.json": _canonical_bytes(evaluator.to_payload()),
        "sealed/action_counterfactual_commitment_index.json": _canonical_bytes(
            commitments
        ),
        "lineage/exact_disjoint_inventory.json": _canonical_bytes(disjoint_inventory),
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
    branch_coverage = [
        [
            _text(item["decision_id"], "decision commitment id"),
            _text(branch["selected_action_id"], "commitment action id"),
        ]
        for item_value in _sequence(
            commitments["decision_branch_commitments"],
            "decision_branch_commitments",
        )
        for item in [_mapping(item_value, "decision commitment")]
        for branch_value in _sequence(item["branches"], "decision branches")
        for branch in [_mapping(branch_value, "decision branch")]
    ]
    segment_counts = Counter(item.segment_id for item in evaluator.decision_sessions)
    manifest_core: dict[str, object] = {
        "schema_version": HORIZON_SOURCE_V5_ADMISSION_MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "implementation_git_commit": commit,
        "source_protocol_id": evaluator.protocol_id,
        "public_plan_sha256": public.public_plan_sha256,
        "sealed_bundle_sha256": evaluator.sealed_bundle_sha256,
        "reader_text_inventory_artifact_id": disjoint_inventory["artifact_id"],
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
        "status": "source_v5_campaign_input_admitted_execution_not_authorized",
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


def materialize_relationship_product_horizon_source_v5_admission(
    output_dir: pathlib.Path,
    *,
    implementation_git_commit: str,
    source_v3_admission_root: pathlib.Path,
    source_v4_admission_root: pathlib.Path,
    development_reader_root: pathlib.Path,
    attempt03_embedding_table_path: pathlib.Path,
    attempt03_reobservation_path: pathlib.Path,
    qualification_v5_embedding_table_path: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> dict[str, object]:
    """Persist one create-only, manifest-last source-v5 admission root."""

    root = pathlib.Path(output_dir)
    if os.path.lexists(root):
        raise FileExistsError(f"source-v5 admission root is create-only: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    kwargs = {
        "implementation_git_commit": implementation_git_commit,
        "source_v3_admission_root": source_v3_admission_root,
        "source_v4_admission_root": source_v4_admission_root,
        "development_reader_root": development_reader_root,
        "attempt03_embedding_table_path": attempt03_embedding_table_path,
        "attempt03_reobservation_path": attempt03_reobservation_path,
        "qualification_v5_embedding_table_path": qualification_v5_embedding_table_path,
        "protocol_path": protocol_path,
    }
    files = build_relationship_product_horizon_source_v5_admission_materialization(
        **kwargs
    )
    for relative, raw in sorted(files.items()):
        if relative == "manifest.json":
            continue
        _write_create_only(root.joinpath(*pathlib.PurePosixPath(relative).parts), raw)
    rebuilt = build_relationship_product_horizon_source_v5_admission_materialization(
        **kwargs
    )
    if rebuilt != files:
        raise ValueError("source-v5 admission pre-manifest semantic rebuild drifted")
    persisted = _regular_file_map(root)
    expected_pre_manifest = {
        relative: raw for relative, raw in files.items() if relative != "manifest.json"
    }
    if persisted != expected_pre_manifest:
        raise ValueError("source-v5 admission pre-manifest persisted bytes drifted")
    _write_create_only(root / "manifest.json", files["manifest.json"])
    return _parse_json_bytes(
        files["manifest.json"],
        source="source-v5 admission manifest",
    )


def _metadata_snapshot(paths: Sequence[pathlib.Path]) -> tuple[tuple[str, int, int], ...]:
    rows: list[tuple[str, int, int]] = []
    for root in paths:
        target = pathlib.Path(root)
        if target.is_dir():
            _regular_file_map(target)
            candidates = [
                pathlib.Path(entry.path)
                for directory, _names, _files in os.walk(target)
                for entry in os.scandir(directory)
                if entry.is_file(follow_symlinks=False)
            ]
        else:
            candidates = [target]
        for candidate in candidates:
            item = candidate.lstat()
            rows.append((str(candidate.resolve()), item.st_size, item.st_mtime_ns))
    return tuple(sorted(rows))


def validate_relationship_product_horizon_source_v5_admission(
    output_dir: pathlib.Path,
    *,
    expected_protocol_id: str,
    expected_artifact_id: str,
    source_v3_admission_root: pathlib.Path,
    source_v4_admission_root: pathlib.Path,
    development_reader_root: pathlib.Path,
    attempt03_embedding_table_path: pathlib.Path,
    attempt03_reobservation_path: pathlib.Path,
    qualification_v5_embedding_table_path: pathlib.Path,
    protocol_path: pathlib.Path | None = None,
) -> dict[str, object]:
    """Read-only full rebuild requiring external protocol and artifact identities."""

    required_protocol = _require_sha256(expected_protocol_id, "expected_protocol_id")
    required_artifact = _require_sha256(expected_artifact_id, "expected_artifact_id")
    path = pathlib.Path(
        protocol_path or relationship_product_horizon_source_v5_admission_protocol_path()
    )
    _, protocol_id = load_relationship_product_horizon_source_v5_admission_protocol(path)
    if protocol_id != required_protocol:
        raise ValueError("source-v5 admission external protocol identity drifted")
    root = pathlib.Path(output_dir)
    actual = _regular_file_map(root)
    if tuple(sorted(actual)) != tuple(sorted(_MATERIALIZATION_FILES)):
        raise ValueError("source-v5 admission file inventory drifted")
    manifest = _parse_json_bytes(
        actual["manifest.json"],
        source="source-v5 admission manifest",
    )
    if manifest.get("artifact_id") != required_artifact:
        raise ValueError("source-v5 admission external artifact identity drifted")
    watched = (
        root,
        pathlib.Path(source_v3_admission_root),
        pathlib.Path(source_v4_admission_root),
        pathlib.Path(development_reader_root),
        pathlib.Path(attempt03_embedding_table_path),
        pathlib.Path(attempt03_reobservation_path),
        pathlib.Path(qualification_v5_embedding_table_path),
        pathlib.Path(qualification_v5_embedding_table_path).parent / "manifest.json",
    )
    before = _metadata_snapshot(watched)
    implementation_commit = _require_git_commit(
        manifest.get("implementation_git_commit"),
        "manifest.implementation_git_commit",
    )
    expected = build_relationship_product_horizon_source_v5_admission_materialization(
        implementation_git_commit=implementation_commit,
        source_v3_admission_root=source_v3_admission_root,
        source_v4_admission_root=source_v4_admission_root,
        development_reader_root=development_reader_root,
        attempt03_embedding_table_path=attempt03_embedding_table_path,
        attempt03_reobservation_path=attempt03_reobservation_path,
        qualification_v5_embedding_table_path=qualification_v5_embedding_table_path,
        protocol_path=path,
    )
    after = _metadata_snapshot(watched)
    if after != before:
        raise ValueError("source-v5 admission read-only validation changed input metadata")
    if actual != expected:
        for relative in sorted(set(actual) | set(expected)):
            if actual.get(relative) != expected.get(relative):
                raise ValueError(f"source-v5 admission byte drifted: {relative}")
        raise ValueError("source-v5 admission materialization drifted")
    if manifest.get("protocol_id") != protocol_id:
        raise ValueError("source-v5 admission manifest protocol identity drifted")
    return manifest


__all__ = [
    "HORIZON_SOURCE_V5_ADMISSION_INVENTORY_SCHEMA_VERSION",
    "HORIZON_SOURCE_V5_ADMISSION_MANIFEST_SCHEMA_VERSION",
    "HORIZON_SOURCE_V5_ADMISSION_PROTOCOL_SCHEMA_VERSION",
    "build_relationship_product_horizon_source_v5_admission_materialization",
    "load_relationship_product_horizon_source_v5_admission_protocol",
    "materialize_relationship_product_horizon_source_v5_admission",
    "relationship_product_horizon_source_v5_admission_protocol_path",
    "validate_relationship_product_horizon_source_v5_admission",
]
