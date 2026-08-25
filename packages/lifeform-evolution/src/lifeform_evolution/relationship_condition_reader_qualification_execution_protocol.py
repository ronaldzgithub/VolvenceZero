"""Static freeze and admission validators for reader qualification execution.

This module deliberately has no execution entry point and imports neither
``torch`` nor ``sentence_transformers``.  It freezes exact local Python source,
the complete load-relevant BGE-M3 snapshot, the already-created model-free
qualification preflight, and the honest process/claim boundary that a later
formal CLI must satisfy.  A public-anchor receipt is accepted only when its
content address is supplied independently by the caller.

The full preflight-tree validator opens sealed files.  It is therefore an
offline freeze-time tool and must not be called by the prediction-stage parent
before its prediction ledger is committed.
"""

from __future__ import annotations

import base64
import binascii
import csv
from dataclasses import dataclass
import datetime as dt
import hashlib
import importlib.metadata
import io
import os
import pathlib
import platform
import re
import struct
import stat
import subprocess
import sys
from typing import Mapping

from volvence_zero.canonical_json import canonical_json_bytes, strict_json_loads


RELATIONSHIP_READER_EXECUTION_SOURCE_TREE_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-execution-source-tree.v1"
)
RELATIONSHIP_READER_BGE_SNAPSHOT_TREE_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-bge-snapshot-tree.v1"
)
RELATIONSHIP_READER_EXECUTION_PREFLIGHT_BINDING_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-execution-preflight-binding.v1"
)
RELATIONSHIP_READER_EXECUTION_PROTOCOL_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-execution-protocol.v1"
)
RELATIONSHIP_READER_EXECUTION_PUBLIC_ANCHOR_RECEIPT_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-execution-public-anchor-receipt.v1"
)
RELATIONSHIP_READER_EXECUTION_RUNTIME_IDENTITY_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-execution-runtime-identity.v1"
)
RELATIONSHIP_READER_EXECUTION_INTEGRITY_RECEIPT_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-execution-integrity-receipt.v1"
)

BGE_M3_MODEL_ID = "BAAI/bge-m3"
BGE_M3_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3_WEIGHTS_SHA256 = "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"
BGE_M3_SENTENCE_TRANSFORMERS_VERSION = "5.6.0"

DEFAULT_EXECUTION_CLI_RELATIVE_PATH = "scripts/run_relationship_condition_reader_qualification_execution.py"

_SOURCE_COVERAGE_PATTERNS = (
    "packages/*/src/**/*.py",
    DEFAULT_EXECUTION_CLI_RELATIVE_PATH,
)
_SOURCE_PATH_CONTRACT = "repo_relative_posix_utf8_ordinal_casefold_unique.v1"
_SOURCE_CONTENT_CONTRACT = "exact_raw_bytes_no_eol_normalization.v1"
_SOURCE_LINK_CONTRACT = "regular_nlink_one_non_symlink_non_reparse.v1"

_BGE_EXPECTED_PATHS = frozenset(
    {
        "1_Pooling/config.json",
        "README.md",
        "config.json",
        "config_sentence_transformers.json",
        "modules.json",
        "pytorch_model.bin",
        "sentence_bert_config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)

_PREFLIGHT_SCHEMA_VERSION = "relationship-condition-reader-qualification-protocol.v1"
_PREFLIGHT_MANIFEST_SCHEMA_VERSION = "relationship-condition-reader-qualification-preflight-manifest.v1"
_PREFLIGHT_FILES = (
    ("manifest.json", _PREFLIGHT_MANIFEST_SCHEMA_VERSION, True),
    ("protocol.json", _PREFLIGHT_SCHEMA_VERSION, False),
    (
        "public/predictor_request.json",
        "relationship-condition-reader-qualification-predictor-request.v1",
        True,
    ),
    (
        "public/public_corpus.json",
        "relationship-condition-reader-qualification-public-corpus.v1",
        True,
    ),
    (
        "public/publication_request.json",
        "relationship-condition-reader-qualification-publication-request.v1",
        True,
    ),
    (
        "sealed/challenge_labels.json",
        "relationship-condition-reader-qualification-challenge-labels.v1",
        True,
    ),
    (
        "sealed/condition_training_labels.json",
        "relationship-condition-reader-qualification-training-labels.v1",
        True,
    ),
    (
        "sealed/group_split.json",
        "relationship-condition-reader-qualification-group-split.v1",
        True,
    ),
)
_PREFLIGHT_MANIFEST_RECEIPT_PATHS = frozenset(
    path for path, _schema, _artifact in _PREFLIGHT_FILES if path != "manifest.json"
)

_EXECUTION_ORDER = (
    "validate_protocol_source_model_runtime_anchor_and_absent_output_root",
    "launch_predictor_1_suspended_assign_job_resume_and_wait_empty",
    "revalidate_source_and_snapshot_tree",
    "launch_predictor_2_suspended_assign_job_resume_and_wait_empty",
    "verify_embedding_reader_and_ledger_outputs_byte_exact",
    "ledger_xplusb_write_flush_fsync_same_descriptor_readback",
    "ledger_close_reopen_identity_revalidation",
    "commit_receipt_write_fsync_close_reopen",
    "create_scoring_request",
    "launch_fresh_model_free_scorer_suspended_assign_job_resume",
    "validate_scoring_request",
    "validate_commit_receipt",
    "revalidate_prediction_ledger",
    "open_challenge_labels",
    "open_group_split",
    "compute_score",
    "report_write_fsync_close_reopen",
    "wait_scorer_job_empty",
    "validate_final_manifest",
)

_PROCESS_FIREWALL = {
    "predictor_process_count": 2,
    "predictors_fresh": True,
    "predictors_sequential_after_full_exit": True,
    "scorer_process_count": 1,
    "scorer_fresh": True,
    "scorer_model_free": True,
    "predictor_receives_challenge_labels": False,
    "predictor_receives_sealed_paths": False,
    "reviewed_source_visibility_recorded": True,
    "empty_environment_plus_exact_allowlist": True,
    "extra_environment_keys_allowed": False,
    "ledger_commit_before_label_open": True,
    "windows_job": {
        "create_suspended": True,
        "assign_before_resume": True,
        "kill_on_job_close": True,
        "active_process_limit": 1,
        "active_processes_zero_before_stage_transition": True,
        "fallback_allowed": False,
    },
    "os_security_boundary": False,
    "malicious_same_user_process_confinement": False,
    "filesystem_isolation": False,
    "network_isolation_at_os_layer": False,
    "windows_directory_entry_durability_attested": False,
}

_QUALIFICATION_GATES = {
    "live_embeddings_per_predictor": 228,
    "training_count": 4,
    "challenge_row_count": 224,
    "challenge_count_per_class": 112,
    "challenge_group_count": 28,
    "rows_per_challenge_group": 8,
    "required_correct_rows": 224,
    "required_correct_groups": 28,
    "minimum_normalized_margin_hex": (0.01).hex(),
    "tie_policy": "fail",
    "fresh_bge_process_count": 2,
    "fresh_bge_exact_vector_reobservation_required": True,
    "reader_artifact_rederived_from_reobservation_required": True,
    "three_deterministic_outputs_byte_exact": True,
    "prediction_ledger_fsync_before_label_release": True,
    "statistical_independence_claim": False,
}

_CLAIMS = {
    "condition_reader_exact_source_qualified": False,
    "campaign_execution_admitted": False,
    "appendable_product_effect": False,
    "readable_product_effect": False,
    "learnable_product_effect": False,
    "steerable_product_effect": False,
    "four_able_complete": False,
    "formal_evidence_authorized": False,
    "human_validation_complete": False,
    "production_active": False,
}

_REQUIRED_DISTRIBUTIONS = frozenset({"huggingface-hub", "sentence-transformers", "torch", "transformers"})

_INTEGRITY_PHASES = (
    "post_anchor_pre_execution",
    "pre_prediction_child_1",
    "post_prediction_child_1",
    "pre_prediction_child_2",
    "post_prediction_child_2",
    "pre_scorer",
    "post_scorer",
    "final_validation",
)


@dataclass(frozen=True)
class RelationshipConditionReaderQualificationIntegrityGuard:
    """Callable phase guard that reobserves every frozen execution input."""

    execution_protocol: Mapping[str, object]
    expected_execution_protocol_id: str
    repository_root: pathlib.Path
    bge_snapshot_root: pathlib.Path

    def __call__(
        self,
        *,
        phase: str,
        previous_integrity_receipt_artifact_id: str | None,
    ) -> Mapping[str, object]:
        return build_relationship_condition_reader_qualification_integrity_receipt(
            execution_protocol=self.execution_protocol,
            expected_execution_protocol_id=self.expected_execution_protocol_id,
            repository_root=self.repository_root,
            bge_snapshot_root=self.bge_snapshot_root,
            phase=phase,
            previous_integrity_receipt_artifact_id=(previous_integrity_receipt_artifact_id),
        )


def build_relationship_condition_reader_qualification_runtime_identity() -> Mapping[str, object]:
    """Reobserve Python, GPU/driver, and four distributions without model import."""

    _assert_integrity_observer_model_free()
    core = {
        "schema_version": RELATIONSHIP_READER_EXECUTION_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "platform": "windows",
        "gpu": _observe_gpu_identity(),
        "python": _observe_python_identity(),
        "distributions": [_observe_distribution_pin(lookup_name) for lookup_name in sorted(_REQUIRED_DISTRIBUTIONS)],
    }
    payload = _with_artifact_id(core)
    _validate_runtime_identity(payload)
    _assert_integrity_observer_model_free()
    return payload


def validate_relationship_condition_reader_qualification_runtime_identity(
    payload: Mapping[str, object],
    *,
    reobserve_current_runtime: bool = False,
) -> str:
    """Validate runtime pins and optionally compare with a fresh observation."""

    runtime = _mapping(payload, "runtime identity")
    _validate_runtime_identity(runtime)
    if reobserve_current_runtime:
        observed = build_relationship_condition_reader_qualification_runtime_identity()
        if runtime != observed:
            raise ValueError("runtime identity does not match the current exact host runtime")
    return _digest(runtime["artifact_id"], "runtime identity artifact_id")


def relationship_condition_reader_qualification_integrity_guard(
    *,
    execution_protocol: Mapping[str, object],
    expected_execution_protocol_id: str,
    repository_root: pathlib.Path,
    bge_snapshot_root: pathlib.Path,
) -> RelationshipConditionReaderQualificationIntegrityGuard:
    """Freeze guard inputs and return the callable used at each formal phase."""

    protocol_id = validate_relationship_condition_reader_qualification_execution_protocol(
        execution_protocol,
        expected_protocol_id=expected_execution_protocol_id,
    )
    frozen_payload = _parse_json_object(
        canonical_json_bytes(dict(execution_protocol)),
        source="execution protocol guard copy",
        max_bytes=8_000_000,
    )
    return RelationshipConditionReaderQualificationIntegrityGuard(
        execution_protocol=frozen_payload,
        expected_execution_protocol_id=protocol_id,
        repository_root=pathlib.Path(repository_root).absolute(),
        bge_snapshot_root=pathlib.Path(bge_snapshot_root).absolute(),
    )


def build_relationship_condition_reader_qualification_integrity_receipt(
    *,
    execution_protocol: Mapping[str, object],
    expected_execution_protocol_id: str,
    repository_root: pathlib.Path,
    bge_snapshot_root: pathlib.Path,
    phase: str,
    previous_integrity_receipt_artifact_id: str | None,
) -> Mapping[str, object]:
    """Rebuild all integrity domains and emit a deterministic chained receipt."""

    _assert_integrity_observer_model_free()
    if phase not in _INTEGRITY_PHASES:
        raise ValueError(f"unknown qualification integrity phase: {phase}")
    phase_ordinal = _INTEGRITY_PHASES.index(phase)
    if phase_ordinal == 0:
        if previous_integrity_receipt_artifact_id is not None:
            raise ValueError("first integrity phase must not claim a previous receipt")
        previous_id = None
    else:
        previous_id = _digest(
            previous_integrity_receipt_artifact_id,
            "previous integrity receipt artifact id",
        )
    protocol = _mapping(execution_protocol, "execution protocol")
    protocol_id = validate_relationship_condition_reader_qualification_execution_protocol(
        protocol,
        expected_protocol_id=expected_execution_protocol_id,
    )
    observed_source = build_relationship_condition_reader_execution_source_tree_manifest(
        repository_root=repository_root
    )
    frozen_source = _mapping(protocol["execution_source_tree"], "frozen source tree")
    if observed_source != frozen_source:
        raise ValueError("integrity guard observed execution source-tree drift")
    observed_bge = build_bge_m3_snapshot_tree_manifest(snapshot_root=bge_snapshot_root)
    frozen_bge = _mapping(protocol["bge_snapshot_tree"], "frozen BGE tree")
    if observed_bge != frozen_bge:
        raise ValueError("integrity guard observed BGE snapshot-tree drift")
    observed_runtime = build_relationship_condition_reader_qualification_runtime_identity()
    frozen_runtime = _mapping(protocol["runtime_identity"], "frozen runtime identity")
    if observed_runtime != frozen_runtime:
        raise ValueError("integrity guard observed host runtime drift")
    _assert_integrity_observer_model_free()
    return _with_artifact_id(
        {
            "schema_version": RELATIONSHIP_READER_EXECUTION_INTEGRITY_RECEIPT_SCHEMA_VERSION,
            "execution_protocol_id": protocol_id,
            "phase": phase,
            "phase_ordinal": phase_ordinal,
            "previous_integrity_receipt_artifact_id": previous_id,
            "source_tree_artifact_id": observed_source["artifact_id"],
            "source_tree_entry_count": observed_source["entry_count"],
            "bge_snapshot_tree_artifact_id": observed_bge["artifact_id"],
            "bge_snapshot_entry_count": observed_bge["entry_count"],
            "runtime_identity_artifact_id": observed_runtime["artifact_id"],
            "source_tree_exact": True,
            "bge_snapshot_tree_exact": True,
            "runtime_identity_exact": True,
            "observer_model_or_cuda_execution_used": False,
            "torch_imported": False,
            "sentence_transformers_imported": False,
            "os_security_boundary": False,
            "windows_directory_entry_durability_attested": False,
        }
    )


def build_relationship_condition_reader_execution_source_tree_manifest(
    *,
    repository_root: pathlib.Path,
    execution_cli_relative_path: str = DEFAULT_EXECUTION_CLI_RELATIVE_PATH,
) -> Mapping[str, object]:
    """Return a content-addressed manifest for all repository runtime Python."""

    root = _absolute_directory(repository_root, "repository_root")
    cli_relative = _relative_posix_path(execution_cli_relative_path, "execution CLI path")
    if cli_relative != DEFAULT_EXECUTION_CLI_RELATIVE_PATH:
        raise ValueError("execution CLI path must remain the frozen qualification entrypoint")

    source_paths = _enumerate_repository_python_sources(root)
    cli_path = root / pathlib.PurePosixPath(cli_relative)
    if not os.path.lexists(cli_path):
        raise FileNotFoundError(f"execution CLI is absent: {cli_path}")
    source_paths.append(cli_path)

    entries = _file_entries(root=root, paths=source_paths, field_name="execution source")
    core = {
        "schema_version": RELATIONSHIP_READER_EXECUTION_SOURCE_TREE_SCHEMA_VERSION,
        "coverage_patterns": list(_SOURCE_COVERAGE_PATTERNS),
        "path_contract": _SOURCE_PATH_CONTRACT,
        "content_contract": _SOURCE_CONTENT_CONTRACT,
        "link_contract": _SOURCE_LINK_CONTRACT,
        "entries": entries,
        "entry_count": len(entries),
        "total_raw_bytes": sum(_nonnegative_integer(row["raw_bytes"], "source bytes") for row in entries),
    }
    manifest = _with_artifact_id(core)
    _validate_source_tree_shape(manifest)
    return manifest


def validate_relationship_condition_reader_execution_source_tree_manifest(
    payload: Mapping[str, object],
    *,
    repository_root: pathlib.Path | None = None,
    execution_cli_relative_path: str = DEFAULT_EXECUTION_CLI_RELATIVE_PATH,
) -> str:
    """Validate manifest shape/content address and optionally reobserve disk."""

    manifest = _mapping(payload, "execution source tree")
    _validate_source_tree_shape(manifest)
    if repository_root is not None:
        observed = build_relationship_condition_reader_execution_source_tree_manifest(
            repository_root=repository_root,
            execution_cli_relative_path=execution_cli_relative_path,
        )
        if manifest != observed:
            raise ValueError("execution source tree does not match the current exact repository tree")
    return _digest(manifest["artifact_id"], "execution source tree artifact_id")


def build_bge_m3_snapshot_tree_manifest(
    *,
    snapshot_root: pathlib.Path,
    model_id: str = BGE_M3_MODEL_ID,
    model_revision: str = BGE_M3_MODEL_REVISION,
) -> Mapping[str, object]:
    """Hash the exact 11-file BGE-M3 snapshot without importing model code."""

    if model_id != BGE_M3_MODEL_ID or model_revision != BGE_M3_MODEL_REVISION:
        raise ValueError("BGE-M3 model identity must remain exact")
    root = _absolute_directory(snapshot_root, "BGE snapshot root")
    paths = _enumerate_tree_files(root, field_name="BGE snapshot")
    entries = _file_entries(root=root, paths=paths, field_name="BGE snapshot")
    observed_paths = {str(row["path"]) for row in entries}
    if observed_paths != _BGE_EXPECTED_PATHS:
        missing = sorted(_BGE_EXPECTED_PATHS - observed_paths)
        extra = sorted(observed_paths - _BGE_EXPECTED_PATHS)
        raise ValueError(f"BGE snapshot file set mismatch; missing={missing}, extra={extra}")
    core = {
        "schema_version": RELATIONSHIP_READER_BGE_SNAPSHOT_TREE_SCHEMA_VERSION,
        "model_id": model_id,
        "model_revision": model_revision,
        "entries": entries,
        "entry_count": len(entries),
    }
    manifest = _with_artifact_id(core)
    _validate_bge_tree_shape(manifest)
    return manifest


def validate_bge_m3_snapshot_tree_manifest(
    payload: Mapping[str, object],
    *,
    snapshot_root: pathlib.Path | None = None,
) -> str:
    """Validate a complete BGE tree and optionally reobserve the snapshot."""

    manifest = _mapping(payload, "BGE snapshot tree")
    _validate_bge_tree_shape(manifest)
    if snapshot_root is not None:
        observed = build_bge_m3_snapshot_tree_manifest(snapshot_root=snapshot_root)
        if manifest != observed:
            raise ValueError("BGE snapshot tree does not match the current exact snapshot")
    return _digest(manifest["artifact_id"], "BGE snapshot tree artifact_id")


def build_relationship_condition_reader_execution_preflight_binding(
    *,
    preflight_root: pathlib.Path,
    expected_qualification_protocol_id: str,
) -> Mapping[str, object]:
    """Freeze all eight preflight files; this offline helper opens sealed files."""

    root = _absolute_directory(preflight_root, "qualification preflight root")
    expected_protocol_id = _digest(
        expected_qualification_protocol_id,
        "expected qualification protocol id",
    )
    rows: list[dict[str, object]] = []
    payloads: dict[str, Mapping[str, object]] = {}
    for relative_path, schema_version, has_artifact_id in _PREFLIGHT_FILES:
        path = root / pathlib.PurePosixPath(relative_path)
        raw = _read_stable_regular_file(path, root=root, field_name=f"preflight {relative_path}")
        parsed = _parse_json_object(raw, source=relative_path, max_bytes=4_000_000)
        if parsed.get("schema_version") != schema_version:
            raise ValueError(f"preflight schema mismatch: {relative_path}")
        artifact_id: str | None = None
        if has_artifact_id:
            _validate_canonical_artifact(parsed, raw=raw, field_name=relative_path)
            artifact_id = _digest(parsed["artifact_id"], f"{relative_path} artifact_id")
        elif relative_path == "protocol.json":
            observed_id = _sha256_json(parsed)
            if observed_id != expected_protocol_id:
                raise ValueError("qualification preflight protocol id mismatch")
        if relative_path not in {"protocol.json", "manifest.json"}:
            if parsed.get("protocol_id") != expected_protocol_id:
                raise ValueError(f"preflight protocol lineage mismatch: {relative_path}")
        payloads[relative_path] = parsed
        rows.append(
            {
                "path": relative_path,
                "schema_version": schema_version,
                "artifact_id": artifact_id,
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "raw_bytes": len(raw),
            }
        )

    _validate_preflight_manifest_against_files(
        payloads["manifest.json"],
        qualification_protocol_id=expected_protocol_id,
        rows=rows,
    )
    core = {
        "schema_version": RELATIONSHIP_READER_EXECUTION_PREFLIGHT_BINDING_SCHEMA_VERSION,
        "qualification_protocol_id": expected_protocol_id,
        "files": rows,
        "file_count": len(rows),
    }
    binding = _with_artifact_id(core)
    _validate_preflight_binding_shape(binding)
    return binding


def validate_relationship_condition_reader_execution_preflight_binding(
    payload: Mapping[str, object],
    *,
    preflight_root: pathlib.Path | None = None,
) -> str:
    """Validate binding shape and optionally reobserve all eight preflight files."""

    binding = _mapping(payload, "preflight binding")
    _validate_preflight_binding_shape(binding)
    if preflight_root is not None:
        observed = build_relationship_condition_reader_execution_preflight_binding(
            preflight_root=preflight_root,
            expected_qualification_protocol_id=_digest(
                binding["qualification_protocol_id"],
                "qualification protocol id",
            ),
        )
        if binding != observed:
            raise ValueError("preflight binding does not match the current exact preflight tree")
    return _digest(binding["artifact_id"], "preflight binding artifact_id")


def build_relationship_condition_reader_qualification_execution_protocol(
    *,
    preflight_binding: Mapping[str, object],
    source_tree_manifest: Mapping[str, object],
    bge_snapshot_tree_manifest: Mapping[str, object],
    runtime_identity: Mapping[str, object],
    proposed_execution_root: pathlib.Path,
    anchor_receipt_relative_path: str,
) -> Mapping[str, object]:
    """Compose the static protocol payload; it does not authorize execution."""

    _validate_preflight_binding_shape(_mapping(preflight_binding, "preflight binding"))
    _validate_source_tree_shape(_mapping(source_tree_manifest, "source tree"))
    _validate_bge_tree_shape(_mapping(bge_snapshot_tree_manifest, "BGE tree"))
    _validate_runtime_identity(_mapping(runtime_identity, "runtime identity"))
    execution_root = _absolute_windows_path_text(proposed_execution_root, "proposed execution root")
    receipt_path = _relative_posix_path(anchor_receipt_relative_path, "anchor receipt path")
    payload = {
        "schema_version": RELATIONSHIP_READER_EXECUTION_PROTOCOL_SCHEMA_VERSION,
        "evidence_role": "exact_source_reader_development_admission_only",
        "qualification_preflight": dict(preflight_binding),
        "execution_source_tree": dict(source_tree_manifest),
        "bge_snapshot_tree": dict(bge_snapshot_tree_manifest),
        "runtime_identity": dict(runtime_identity),
        "proposed_execution_root": execution_root,
        "external_public_anchor": _external_anchor_contract(receipt_path),
        "process_firewall": _deep_mutable_copy(_PROCESS_FIREWALL),
        "execution_order": list(_EXECUTION_ORDER),
        "qualification_gates": dict(_QUALIFICATION_GATES),
        "claims": dict(_CLAIMS),
    }
    _validate_execution_protocol_shape(payload)
    return payload


def relationship_condition_reader_qualification_execution_protocol_id(
    payload: Mapping[str, object],
) -> str:
    """Return the semantic protocol ID after strict shape validation."""

    protocol = _mapping(payload, "execution protocol")
    _validate_execution_protocol_shape(protocol)
    return _sha256_json(protocol)


def validate_relationship_condition_reader_qualification_execution_protocol(
    payload: Mapping[str, object],
    *,
    expected_protocol_id: str,
    repository_root: pathlib.Path | None = None,
    preflight_root: pathlib.Path | None = None,
    bge_snapshot_root: pathlib.Path | None = None,
) -> str:
    """Validate protocol ID and optional exact local source/model/preflight state."""

    protocol = _mapping(payload, "execution protocol")
    _validate_execution_protocol_shape(protocol)
    observed_id = _sha256_json(protocol)
    if observed_id != _digest(expected_protocol_id, "expected execution protocol id"):
        raise ValueError("execution protocol id does not match the external expected id")
    validate_relationship_condition_reader_execution_source_tree_manifest(
        _mapping(protocol["execution_source_tree"], "execution source tree"),
        repository_root=repository_root,
    )
    validate_bge_m3_snapshot_tree_manifest(
        _mapping(protocol["bge_snapshot_tree"], "BGE snapshot tree"),
        snapshot_root=bge_snapshot_root,
    )
    validate_relationship_condition_reader_execution_preflight_binding(
        _mapping(protocol["qualification_preflight"], "preflight binding"),
        preflight_root=preflight_root,
    )
    return observed_id


def load_relationship_condition_reader_qualification_execution_protocol(
    path: pathlib.Path,
    *,
    expected_protocol_id: str,
) -> tuple[Mapping[str, object], bytes]:
    """Load raw protocol bytes and require an independently supplied ID."""

    source = pathlib.Path(path)
    root = source.absolute().parent
    raw = _read_stable_regular_file(source.absolute(), root=root, field_name="execution protocol")
    payload = _parse_json_object(raw, source=str(source), max_bytes=8_000_000)
    validate_relationship_condition_reader_qualification_execution_protocol(
        payload,
        expected_protocol_id=expected_protocol_id,
    )
    return payload, raw


def build_relationship_condition_reader_qualification_public_anchor_receipt(
    *,
    execution_protocol_payload: Mapping[str, object],
    execution_protocol_raw: bytes,
    expected_execution_protocol_id: str,
    expected_execution_root: pathlib.Path,
    gist_owner: str,
    gist_id: str,
    gist_url: str,
    filename: str,
    public: bool,
    history_version: str,
    history_revision_count: int,
    first_revision: bool,
    created_at: str,
    updated_at: str,
    api_raw_url: str,
    revision_raw_url: str,
    observation_transport: str,
    observed_at_utc: str,
    observed_protocol_raw: bytes,
) -> Mapping[str, object]:
    """Build, but never authorize, a first-revision public-anchor receipt.

    All observation fields must come from an unauthenticated GitHub API/raw
    reobservation performed by the caller.  This function does no network or
    filesystem writes.  Its returned ``artifact_id`` remains data, not
    authority; validation and execution still require that ID through an
    independent channel.
    """

    protocol = _mapping(execution_protocol_payload, "execution protocol")
    protocol_id = validate_relationship_condition_reader_qualification_execution_protocol(
        protocol,
        expected_protocol_id=expected_execution_protocol_id,
    )
    protocol_raw_sha256 = _validate_execution_protocol_raw_binding(
        protocol,
        execution_protocol_raw,
    )
    if not isinstance(observed_protocol_raw, bytes):
        raise TypeError("observed protocol raw must be bytes")
    if observed_protocol_raw != execution_protocol_raw:
        raise ValueError("public anchor observation must exactly match protocol raw bytes")

    anchor = _mapping(protocol["external_public_anchor"], "external public anchor")
    owner = _text(gist_owner, "gist_owner")
    observed_filename = _text(filename, "filename")
    if owner != anchor["gist_owner"] or observed_filename != anchor["filename"]:
        raise ValueError("public anchor owner or filename drifted")
    observed_gist_id = _hex_text(gist_id, "gist_id", lengths={32})
    observed_history_version = _hex_text(
        history_version,
        "history_version",
        lengths={40},
    )
    canonical_gist_url = f"https://gist.github.com/{owner}/{observed_gist_id}"
    if gist_url != canonical_gist_url:
        raise ValueError("public anchor gist_url is not canonical")
    raw_prefix = f"https://gist.githubusercontent.com/{owner}/{observed_gist_id}/raw/"
    if not api_raw_url.startswith(raw_prefix):
        raise ValueError("public anchor API raw URL does not bind the Gist")
    canonical_revision_raw_url = f"{raw_prefix}{observed_history_version}/{observed_filename}"
    if revision_raw_url != canonical_revision_raw_url:
        raise ValueError("public anchor revision raw URL is not canonical")
    if public is not True:
        raise ValueError("public anchor observation must report public visibility")
    if _integer(history_revision_count, "history_revision_count") != 1 or first_revision is not True:
        raise ValueError("public anchor must contain exactly one first revision")
    created = _github_utc_timestamp(created_at, "created_at")
    updated = _github_utc_timestamp(updated_at, "updated_at")
    if created != updated:
        raise ValueError("public anchor must be an unchanged first revision")
    observed_at = _github_utc_timestamp(observed_at_utc, "observed_at_utc")
    if observation_transport != "unauthenticated_github_rest_api_and_raw_http":
        raise ValueError("public anchor requires independent unauthenticated observation")

    expected_root = _absolute_windows_path_text(
        expected_execution_root,
        "expected execution root",
    )
    if protocol["proposed_execution_root"] != expected_root:
        raise ValueError("public anchor execution-root lineage mismatch")
    if pathlib.Path(expected_execution_root).exists():
        raise FileExistsError("public anchor admission requires an absent execution root")

    receipt = _with_artifact_id(
        {
            "schema_version": (RELATIONSHIP_READER_EXECUTION_PUBLIC_ANCHOR_RECEIPT_SCHEMA_VERSION),
            "execution_protocol_id": protocol_id,
            "protocol_raw_sha256": protocol_raw_sha256,
            "protocol_raw_bytes": len(execution_protocol_raw),
            "gist_owner": owner,
            "gist_id": observed_gist_id,
            "gist_url": canonical_gist_url,
            "filename": observed_filename,
            "public": True,
            "history_version": observed_history_version,
            "history_revision_count": 1,
            "first_revision": True,
            "created_at": created,
            "updated_at": updated,
            "api_raw_url": api_raw_url,
            "revision_raw_url": canonical_revision_raw_url,
            "observation_transport": observation_transport,
            "observed_at_utc": observed_at,
            "observed_raw_sha256": hashlib.sha256(observed_protocol_raw).hexdigest(),
            "observed_raw_bytes": len(observed_protocol_raw),
            "exact_protocol_raw_match": True,
            "execution_root": expected_root,
            "execution_root_existed_at_observation": False,
            "model_output_count_at_observation": 0,
            "qualification_report_existed_at_observation": False,
        }
    )
    _validate_anchor_receipt_shape(receipt)
    return receipt


def validate_relationship_condition_reader_qualification_public_anchor_receipt(
    receipt_payload: Mapping[str, object],
    *,
    expected_receipt_artifact_id: str,
    execution_protocol_payload: Mapping[str, object],
    execution_protocol_raw: bytes,
    expected_execution_protocol_id: str,
    expected_execution_root: pathlib.Path,
) -> str:
    """Validate a first-revision public anchor before execution root creation."""

    protocol = _mapping(execution_protocol_payload, "execution protocol")
    protocol_id = validate_relationship_condition_reader_qualification_execution_protocol(
        protocol,
        expected_protocol_id=expected_execution_protocol_id,
    )
    receipt = _mapping(receipt_payload, "public anchor receipt")
    _validate_anchor_receipt_shape(receipt)
    expected_receipt_id = _digest(
        expected_receipt_artifact_id,
        "external expected anchor receipt artifact id",
    )
    if receipt["artifact_id"] != expected_receipt_id:
        raise ValueError("public anchor receipt does not match external expected artifact id")
    anchor = _mapping(protocol["external_public_anchor"], "external public anchor")
    expected_root = _absolute_windows_path_text(expected_execution_root, "expected execution root")
    if protocol["proposed_execution_root"] != expected_root or receipt["execution_root"] != expected_root:
        raise ValueError("public anchor execution-root lineage mismatch")
    if pathlib.Path(expected_execution_root).exists():
        raise FileExistsError("public anchor admission requires an absent execution root")
    raw_sha256 = _validate_execution_protocol_raw_binding(
        protocol,
        execution_protocol_raw,
    )
    if (
        receipt["execution_protocol_id"] != protocol_id
        or receipt["protocol_raw_sha256"] != raw_sha256
        or receipt["observed_raw_sha256"] != raw_sha256
        or receipt["protocol_raw_bytes"] != len(execution_protocol_raw)
        or receipt["observed_raw_bytes"] != len(execution_protocol_raw)
    ):
        raise ValueError("public anchor did not observe the exact execution protocol raw bytes")
    if receipt["gist_owner"] != anchor["gist_owner"] or receipt["filename"] != anchor["filename"]:
        raise ValueError("public anchor owner or filename drifted")
    gist_id = _hex_text(receipt["gist_id"], "gist_id", lengths={32})
    owner = _text(receipt["gist_owner"], "gist_owner")
    if receipt["gist_url"] != f"https://gist.github.com/{owner}/{gist_id}":
        raise ValueError("public anchor gist_url is not canonical")
    history_version = _hex_text(receipt["history_version"], "history_version", lengths={40})
    raw_prefix = f"https://gist.githubusercontent.com/{owner}/{gist_id}/raw/"
    if not _text(receipt["api_raw_url"], "api_raw_url").startswith(raw_prefix):
        raise ValueError("public anchor API raw URL does not bind the Gist")
    if receipt["revision_raw_url"] != (f"{raw_prefix}{history_version}/{receipt['filename']}"):
        raise ValueError("public anchor revision raw URL is not canonical")
    return expected_receipt_id


def _validate_source_tree_shape(payload: Mapping[str, object]) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "coverage_patterns",
            "path_contract",
            "content_contract",
            "link_contract",
            "entries",
            "entry_count",
            "total_raw_bytes",
            "artifact_id",
        },
        "execution source tree",
    )
    if payload["schema_version"] != RELATIONSHIP_READER_EXECUTION_SOURCE_TREE_SCHEMA_VERSION:
        raise ValueError("execution source-tree schema drifted")
    if payload["coverage_patterns"] != list(_SOURCE_COVERAGE_PATTERNS):
        raise ValueError("execution source-tree coverage drifted")
    if (
        payload["path_contract"] != _SOURCE_PATH_CONTRACT
        or payload["content_contract"] != _SOURCE_CONTENT_CONTRACT
        or payload["link_contract"] != _SOURCE_LINK_CONTRACT
    ):
        raise ValueError("execution source-tree canonicalization contract drifted")
    entries = _validate_file_entry_array(payload["entries"], field_name="execution source entries")
    entry_paths = {str(row["path"]) for row in entries}
    if DEFAULT_EXECUTION_CLI_RELATIVE_PATH not in entry_paths:
        raise ValueError("execution source tree does not contain the final CLI")
    invalid_paths = sorted(
        path
        for path in entry_paths
        if path != DEFAULT_EXECUTION_CLI_RELATIVE_PATH and not _is_package_python_source_path(path)
    )
    if invalid_paths:
        raise ValueError(f"execution source tree contains paths outside its exact coverage: {invalid_paths}")
    if _positive_integer(payload["entry_count"], "source entry_count") != len(entries):
        raise ValueError("execution source entry_count mismatch")
    total = sum(_nonnegative_integer(row["raw_bytes"], "source raw_bytes") for row in entries)
    if _nonnegative_integer(payload["total_raw_bytes"], "source total_raw_bytes") != total:
        raise ValueError("execution source total_raw_bytes mismatch")
    _validate_artifact_id(payload, "execution source tree")


def _validate_bge_tree_shape(payload: Mapping[str, object]) -> None:
    _exact_keys(
        payload,
        {"schema_version", "model_id", "model_revision", "entries", "entry_count", "artifact_id"},
        "BGE snapshot tree",
    )
    if (
        payload["schema_version"] != RELATIONSHIP_READER_BGE_SNAPSHOT_TREE_SCHEMA_VERSION
        or payload["model_id"] != BGE_M3_MODEL_ID
        or payload["model_revision"] != BGE_M3_MODEL_REVISION
    ):
        raise ValueError("BGE snapshot tree identity drifted")
    entries = _validate_file_entry_array(payload["entries"], field_name="BGE entries")
    if {str(row["path"]) for row in entries} != _BGE_EXPECTED_PATHS:
        raise ValueError("BGE snapshot tree must contain the exact 11-file set")
    if _positive_integer(payload["entry_count"], "BGE entry_count") != len(entries):
        raise ValueError("BGE snapshot entry_count mismatch")
    weight_row = next(row for row in entries if row["path"] == "pytorch_model.bin")
    if weight_row["raw_sha256"] != BGE_M3_WEIGHTS_SHA256:
        raise ValueError("BGE snapshot root weight digest drifted")
    _validate_artifact_id(payload, "BGE snapshot tree")


def _validate_preflight_binding_shape(payload: Mapping[str, object]) -> None:
    _exact_keys(
        payload,
        {"schema_version", "qualification_protocol_id", "files", "file_count", "artifact_id"},
        "preflight binding",
    )
    if payload["schema_version"] != RELATIONSHIP_READER_EXECUTION_PREFLIGHT_BINDING_SCHEMA_VERSION:
        raise ValueError("preflight binding schema drifted")
    _digest(payload["qualification_protocol_id"], "qualification protocol id")
    files = payload["files"]
    if not isinstance(files, list):
        raise ValueError("preflight binding files must be an array")
    if len(files) != len(_PREFLIGHT_FILES) or payload["file_count"] != len(_PREFLIGHT_FILES):
        raise ValueError("preflight binding must contain all eight files")
    expected_by_path = {path: (schema, has_artifact) for path, schema, has_artifact in _PREFLIGHT_FILES}
    observed_paths: list[str] = []
    for index, raw_row in enumerate(files):
        row = _mapping(raw_row, f"preflight binding file {index}")
        _exact_keys(
            row,
            {"path", "schema_version", "artifact_id", "raw_sha256", "raw_bytes"},
            f"preflight binding file {index}",
        )
        path = _relative_posix_path(row["path"], "preflight binding path")
        if path not in expected_by_path:
            raise ValueError(f"unexpected preflight binding path: {path}")
        expected_schema, has_artifact = expected_by_path[path]
        if row["schema_version"] != expected_schema:
            raise ValueError(f"preflight binding schema mismatch: {path}")
        if has_artifact:
            _digest(row["artifact_id"], f"{path} artifact_id")
        elif row["artifact_id"] is not None:
            raise ValueError("preflight protocol must have a null artifact_id")
        _digest(row["raw_sha256"], f"{path} raw_sha256")
        _positive_integer(row["raw_bytes"], f"{path} raw_bytes")
        observed_paths.append(path)
    if observed_paths != sorted(observed_paths, key=lambda value: value.encode("utf-8")):
        raise ValueError("preflight binding files must use UTF-8 ordinal path order")
    if len(set(observed_paths)) != len(observed_paths):
        raise ValueError("preflight binding paths must be unique")
    _validate_artifact_id(payload, "preflight binding")


def _validate_execution_protocol_shape(payload: Mapping[str, object]) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "evidence_role",
            "qualification_preflight",
            "execution_source_tree",
            "bge_snapshot_tree",
            "runtime_identity",
            "proposed_execution_root",
            "external_public_anchor",
            "process_firewall",
            "execution_order",
            "qualification_gates",
            "claims",
        },
        "execution protocol",
    )
    if payload["schema_version"] != RELATIONSHIP_READER_EXECUTION_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("execution protocol schema drifted")
    if payload["evidence_role"] != "exact_source_reader_development_admission_only":
        raise ValueError("execution protocol evidence role is overbroad")
    _validate_preflight_binding_shape(_mapping(payload["qualification_preflight"], "preflight binding"))
    _validate_source_tree_shape(_mapping(payload["execution_source_tree"], "source tree"))
    _validate_bge_tree_shape(_mapping(payload["bge_snapshot_tree"], "BGE tree"))
    _validate_runtime_identity(_mapping(payload["runtime_identity"], "runtime identity"))
    _absolute_windows_path_text(payload["proposed_execution_root"], "proposed execution root")
    _validate_external_anchor_contract(_mapping(payload["external_public_anchor"], "external public anchor"))
    if payload["process_firewall"] != _PROCESS_FIREWALL:
        raise ValueError("execution process firewall is incomplete or overclaims security")
    if payload["execution_order"] != list(_EXECUTION_ORDER):
        raise ValueError("execution/unseal order drifted")
    if payload["qualification_gates"] != _QUALIFICATION_GATES:
        raise ValueError("qualification gates drifted")
    if payload["claims"] != _CLAIMS:
        raise ValueError("execution protocol claim ceiling drifted")


def _observe_gpu_identity() -> Mapping[str, object]:
    if os.name != "nt":
        raise RuntimeError("formal reader qualification GPU identity requires Windows")
    system_root = pathlib.Path(os.environ.get("SystemRoot", r"C:\Windows"))
    nvidia_smi = system_root / "System32/nvidia-smi.exe"
    nvcuda = system_root / "System32/nvcuda.dll"
    environment = {
        "PATH": str(system_root / "System32"),
        "ProgramFiles": os.environ.get("ProgramFiles", r"C:\Program Files"),
        "SystemRoot": str(system_root),
        "WINDIR": str(system_root),
    }
    query = subprocess.run(
        [
            str(nvidia_smi),
            "--query-gpu=index,uuid,name,pci.bus_id,driver_version,vbios_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=environment,
        timeout=30,
    )
    rows = list(csv.reader(io.StringIO(query.stdout)))
    if len(rows) != 1 or len(rows[0]) != 8:
        raise ValueError("reader qualification requires exactly one observable GPU row")
    values = [value.strip() for value in rows[0]]
    full = subprocess.run(
        [str(nvidia_smi)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=environment,
        timeout=30,
    )
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", full.stdout)
    if match is None:
        raise ValueError("nvidia-smi output did not expose its CUDA maximum version")
    try:
        index = int(values[0])
        memory_total_mib = int(values[6])
    except ValueError as exc:
        raise ValueError("nvidia-smi returned a non-integer index or memory size") from exc
    return {
        "index": index,
        "uuid": _text(values[1], "GPU UUID"),
        "name": _text(values[2], "GPU name"),
        "pci_bus_id": _text(values[3], "GPU PCI bus id"),
        "driver_version": _text(values[4], "GPU driver version"),
        "vbios_version": _text(values[5], "GPU VBIOS version"),
        "memory_total_mib": memory_total_mib,
        "compute_capability": _text(values[7], "GPU compute capability"),
        "nvidia_smi_cuda_max_version": match.group(1),
        "nvidia_smi_binary": _absolute_file_pin(nvidia_smi, field_name="nvidia-smi"),
        "nvcuda_binary": _absolute_file_pin(nvcuda, field_name="nvcuda"),
    }


def _observe_python_identity() -> Mapping[str, object]:
    executable = pathlib.Path(sys.executable)
    prefix = pathlib.Path(sys.base_prefix)
    dll_names = ("python3.dll", f"python{sys.version_info.major}{sys.version_info.minor}.dll")
    executable_pin = _absolute_file_pin(executable, field_name="Python executable")
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "version_full": sys.version,
        "architecture": platform.machine(),
        "pointer_bits": struct.calcsize("P") * 8,
        "platform_version": platform.platform(),
        "executable": executable_pin["path"],
        "executable_raw_sha256": executable_pin["raw_sha256"],
        "executable_raw_bytes": executable_pin["raw_bytes"],
        "runtime_dlls": [
            _absolute_file_pin(prefix / name, field_name=f"Python runtime DLL {name}") for name in dll_names
        ],
    }


def _observe_distribution_pin(lookup_name: str) -> Mapping[str, object]:
    distribution = importlib.metadata.distribution(lookup_name)
    files = distribution.files
    if files is None:
        raise ValueError(f"distribution {lookup_name} does not publish a RECORD file list")
    metadata_candidates = sorted(
        (item for item in files if item.as_posix().endswith(".dist-info/METADATA")),
        key=lambda item: item.as_posix(),
    )
    if len(metadata_candidates) != 1:
        raise ValueError(f"distribution {lookup_name} has ambiguous dist-info metadata")
    dist_info_relative = pathlib.PurePosixPath(metadata_candidates[0].as_posix()).parent
    dist_info_path = pathlib.Path(os.path.abspath(distribution.locate_file(str(dist_info_relative))))
    _assert_directory_without_reparse(
        dist_info_path,
        root=dist_info_path,
        field_name=f"{lookup_name} dist-info",
    )
    metadata_pin = _relative_file_pin(
        dist_info_path / "METADATA",
        root=dist_info_path,
        field_name=f"{lookup_name} METADATA",
    )
    record_path = dist_info_path / "RECORD"
    record_pin = _relative_file_pin(
        record_path,
        root=dist_info_path,
        field_name=f"{lookup_name} RECORD",
    )
    wheel_pin = _relative_file_pin(
        dist_info_path / "WHEEL",
        root=dist_info_path,
        field_name=f"{lookup_name} WHEEL",
    )
    record_raw = _read_stable_regular_file(
        record_path,
        root=dist_info_path,
        field_name=f"{lookup_name} RECORD",
        require_single_hardlink=False,
    )
    site_packages_root = pathlib.Path(os.path.abspath(distribution.locate_file(".")))
    environment_root = pathlib.Path(os.path.abspath(sys.prefix))
    verification = _verify_record_entries(
        record_raw,
        record_base=site_packages_root,
        environment_root=environment_root,
        field_name=lookup_name,
    )
    return {
        "lookup_name": lookup_name,
        "distribution_name": _text(distribution.metadata["Name"], f"{lookup_name} distribution name"),
        "version": _text(distribution.version, f"{lookup_name} version"),
        "dist_info_path": str(dist_info_path),
        "metadata": metadata_pin,
        "record": record_pin,
        "wheel": wheel_pin,
        "record_entry_count": verification["record_entry_count"],
        "record_hashed_entry_count": verification["record_hashed_entry_count"],
        "record_hashed_entries_verified_at_freeze": True,
        "record_unhashed_non_pyc_paths": verification["record_unhashed_non_pyc_paths"],
    }


def _verify_record_entries(
    record_raw: bytes,
    *,
    record_base: pathlib.Path,
    environment_root: pathlib.Path,
    field_name: str,
) -> Mapping[str, object]:
    try:
        text = record_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field_name} RECORD is not UTF-8") from exc
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows:
        raise ValueError(f"{field_name} RECORD is empty")
    hashed_count = 0
    unhashed_non_pyc: list[str] = []
    observed_paths: list[str] = []
    root = pathlib.Path(os.path.abspath(environment_root))
    base = pathlib.Path(os.path.abspath(record_base))
    for index, row in enumerate(rows):
        if len(row) != 3:
            raise ValueError(f"{field_name} RECORD row {index} must contain three fields")
        relative_path, hash_field, size_field = row
        relative = _record_relative_posix_path(
            relative_path,
            f"{field_name} RECORD path {index}",
        )
        observed_paths.append(relative)
        candidate = pathlib.Path(os.path.abspath(base / pathlib.PurePosixPath(relative)))
        _relative_to_root(candidate, root=root, field_name=f"{field_name} RECORD entry {relative}")
        if not hash_field:
            if size_field:
                raise ValueError(f"{field_name} unhashed RECORD row unexpectedly declares a size")
            if not relative.endswith(".pyc"):
                unhashed_non_pyc.append(relative)
            continue
        if not size_field:
            raise ValueError(f"{field_name} hashed RECORD row is missing a size")
        try:
            expected_size = int(size_field)
        except ValueError as exc:
            raise ValueError(f"{field_name} RECORD row has an invalid size") from exc
        if expected_size < 0:
            raise ValueError(f"{field_name} RECORD row has a negative size")
        algorithm, separator, encoded = hash_field.partition("=")
        if separator != "=" or algorithm != "sha256" or not encoded:
            raise ValueError(f"{field_name} RECORD permits only sha256 hashes")
        try:
            expected_digest = base64.b64decode(
                encoded + "=" * (-len(encoded) % 4),
                altchars=b"-_",
                validate=True,
            ).hex()
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"{field_name} RECORD row has invalid URL-safe base64") from exc
        if len(expected_digest) != 64:
            raise ValueError(f"{field_name} RECORD row does not contain a SHA-256 digest")
        actual_digest, actual_size = _hash_stable_regular_file(
            candidate,
            root=root,
            field_name=f"{field_name} RECORD entry {relative}",
            require_single_hardlink=False,
        )
        if actual_digest != expected_digest or actual_size != expected_size:
            raise ValueError(f"{field_name} RECORD entry identity mismatch: {relative}")
        hashed_count += 1
    if hashed_count < 1:
        raise ValueError(f"{field_name} RECORD contains no hashed entries")
    if len(observed_paths) != len(set(observed_paths)):
        raise ValueError(f"{field_name} RECORD contains duplicate paths")
    if len(observed_paths) != len({path.casefold() for path in observed_paths}):
        raise ValueError(f"{field_name} RECORD contains a casefold path collision")
    if len(unhashed_non_pyc) != 1 or not unhashed_non_pyc[0].endswith(".dist-info/RECORD"):
        raise ValueError(f"{field_name} RECORD has unexpected unhashed non-pyc paths")
    return {
        "record_entry_count": len(rows),
        "record_hashed_entry_count": hashed_count,
        "record_unhashed_non_pyc_paths": unhashed_non_pyc,
    }


def _absolute_file_pin(path: pathlib.Path, *, field_name: str) -> dict[str, object]:
    candidate = pathlib.Path(os.path.abspath(path))
    digest, raw_bytes = _hash_stable_regular_file(
        candidate,
        root=candidate.parent,
        field_name=field_name,
        require_single_hardlink=False,
    )
    return {"path": str(candidate), "raw_sha256": digest, "raw_bytes": raw_bytes}


def _relative_file_pin(
    path: pathlib.Path,
    *,
    root: pathlib.Path,
    field_name: str,
) -> dict[str, object]:
    candidate = pathlib.Path(os.path.abspath(path))
    digest, raw_bytes = _hash_stable_regular_file(
        candidate,
        root=root,
        field_name=field_name,
        require_single_hardlink=False,
    )
    return {
        "path": _relative_to_root(candidate, root=root, field_name=field_name).as_posix(),
        "raw_sha256": digest,
        "raw_bytes": raw_bytes,
    }


def _assert_integrity_observer_model_free() -> None:
    forbidden = sorted(
        name
        for name in sys.modules
        if name == "torch"
        or name.startswith("torch.")
        or name == "sentence_transformers"
        or name.startswith("sentence_transformers.")
    )
    if forbidden:
        raise RuntimeError(
            f"qualification integrity observation must run in a model-free process; loaded={forbidden[:8]}"
        )


def _validate_runtime_identity(payload: Mapping[str, object]) -> None:
    _exact_keys(
        payload,
        {"schema_version", "platform", "gpu", "python", "distributions", "artifact_id"},
        "runtime identity",
    )
    if payload["schema_version"] != RELATIONSHIP_READER_EXECUTION_RUNTIME_IDENTITY_SCHEMA_VERSION:
        raise ValueError("runtime identity schema drifted")
    if payload["platform"] != "windows":
        raise ValueError("formal reader qualification runtime must be Windows")
    gpu = _mapping(payload["gpu"], "runtime GPU")
    _exact_keys(
        gpu,
        {
            "index",
            "uuid",
            "name",
            "pci_bus_id",
            "driver_version",
            "vbios_version",
            "memory_total_mib",
            "compute_capability",
            "nvidia_smi_cuda_max_version",
            "nvidia_smi_binary",
            "nvcuda_binary",
        },
        "runtime GPU",
    )
    _nonnegative_integer(gpu["index"], "GPU index")
    for name in (
        "uuid",
        "name",
        "pci_bus_id",
        "driver_version",
        "vbios_version",
        "compute_capability",
        "nvidia_smi_cuda_max_version",
    ):
        _text(gpu[name], f"GPU {name}")
    _positive_integer(gpu["memory_total_mib"], "GPU memory_total_mib")
    _validate_absolute_file_pin(_mapping(gpu["nvidia_smi_binary"], "nvidia-smi pin"), "nvidia-smi pin")
    _validate_absolute_file_pin(_mapping(gpu["nvcuda_binary"], "nvcuda pin"), "nvcuda pin")

    python = _mapping(payload["python"], "runtime Python")
    _exact_keys(
        python,
        {
            "implementation",
            "version",
            "version_full",
            "architecture",
            "pointer_bits",
            "platform_version",
            "executable",
            "executable_raw_sha256",
            "executable_raw_bytes",
            "runtime_dlls",
        },
        "runtime Python",
    )
    if python["implementation"] != "CPython" or python["pointer_bits"] != 64:
        raise ValueError("runtime Python must be 64-bit CPython")
    for name in ("version", "version_full", "architecture", "platform_version"):
        _text(python[name], f"Python {name}")
    _absolute_windows_path_text(python["executable"], "Python executable")
    _digest(python["executable_raw_sha256"], "Python executable raw_sha256")
    _positive_integer(python["executable_raw_bytes"], "Python executable raw_bytes")
    runtime_dlls = python["runtime_dlls"]
    if not isinstance(runtime_dlls, list) or len(runtime_dlls) < 2:
        raise ValueError("runtime Python must pin at least python3 and python311 DLLs")
    dll_paths = []
    for index, value in enumerate(runtime_dlls):
        pin = _mapping(value, f"Python DLL pin {index}")
        _validate_absolute_file_pin(pin, f"Python DLL pin {index}")
        dll_paths.append(str(pin["path"]).casefold())
    if len(dll_paths) != len(set(dll_paths)):
        raise ValueError("runtime Python DLL pins must be unique")

    distributions = payload["distributions"]
    if not isinstance(distributions, list):
        raise ValueError("runtime distributions must be an array")
    lookup_names: list[str] = []
    for index, value in enumerate(distributions):
        distribution = _mapping(value, f"runtime distribution {index}")
        _validate_distribution_pin(distribution, index=index)
        lookup_names.append(str(distribution["lookup_name"]))
    if lookup_names != sorted(_REQUIRED_DISTRIBUTIONS):
        raise ValueError("runtime must pin exactly the four load-critical distributions in canonical order")
    sentence_transformers = next(
        item for item in distributions if isinstance(item, dict) and item.get("lookup_name") == "sentence-transformers"
    )
    if sentence_transformers["version"] != BGE_M3_SENTENCE_TRANSFORMERS_VERSION:
        raise ValueError("sentence-transformers version disagrees with the BGE contract")
    _validate_artifact_id(payload, "runtime identity")


def _validate_distribution_pin(payload: Mapping[str, object], *, index: int) -> None:
    _exact_keys(
        payload,
        {
            "lookup_name",
            "distribution_name",
            "version",
            "dist_info_path",
            "metadata",
            "record",
            "wheel",
            "record_entry_count",
            "record_hashed_entry_count",
            "record_hashed_entries_verified_at_freeze",
            "record_unhashed_non_pyc_paths",
        },
        f"runtime distribution {index}",
    )
    lookup_name = _text(payload["lookup_name"], "distribution lookup_name")
    if lookup_name not in _REQUIRED_DISTRIBUTIONS:
        raise ValueError(f"unexpected runtime distribution: {lookup_name}")
    _text(payload["distribution_name"], "distribution_name")
    _text(payload["version"], "distribution version")
    _absolute_windows_path_text(payload["dist_info_path"], "distribution dist_info_path")
    for name in ("metadata", "record", "wheel"):
        _validate_relative_file_pin(_mapping(payload[name], f"distribution {name}"), f"distribution {name}")
    total = _positive_integer(payload["record_entry_count"], "RECORD entry count")
    hashed = _positive_integer(payload["record_hashed_entry_count"], "RECORD hashed entry count")
    if hashed > total:
        raise ValueError("RECORD hashed entry count cannot exceed total entries")
    if payload["record_hashed_entries_verified_at_freeze"] is not True:
        raise ValueError("all hashed RECORD entries must be verified at freeze time")
    unhashed = payload["record_unhashed_non_pyc_paths"]
    if not isinstance(unhashed, list) or len(unhashed) != 1:
        raise ValueError("only the RECORD file itself may be unhashed and non-pyc")
    record_self = _relative_posix_path(unhashed[0], "unhashed RECORD self path")
    if not record_self.endswith(".dist-info/RECORD"):
        raise ValueError("unhashed non-pyc distribution path must be RECORD itself")


def _validate_external_anchor_contract(payload: Mapping[str, object]) -> None:
    _exact_keys(
        payload,
        {
            "required",
            "receipt_schema_version",
            "receipt_path",
            "gist_owner",
            "filename",
            "visibility",
            "new_anchor_required",
            "existing_product_horizon_anchor_accepted",
            "exact_protocol_raw_match_required",
            "single_first_revision_required",
            "created_equals_updated_required",
            "execution_root_absent_at_observation_required",
            "model_output_count_at_observation_required",
            "expected_receipt_artifact_id_must_be_external",
            "protocol_alone_authorizes_execution",
        },
        "external public anchor",
    )
    receipt_path = _relative_posix_path(payload["receipt_path"], "anchor receipt path")
    if not receipt_path.startswith("artifacts/relationship_lab/"):
        raise ValueError("anchor receipt must live under artifacts/relationship_lab")
    expected = _external_anchor_contract(receipt_path)
    if payload != expected:
        raise ValueError("external public-anchor admission contract drifted")


def _external_anchor_contract(receipt_path: str) -> dict[str, object]:
    return {
        "required": True,
        "receipt_schema_version": RELATIONSHIP_READER_EXECUTION_PUBLIC_ANCHOR_RECEIPT_SCHEMA_VERSION,
        "receipt_path": receipt_path,
        "gist_owner": "ronaldzgithub",
        "filename": "relationship_condition_reader_qualification_execution_v1.json",
        "visibility": "public",
        "new_anchor_required": True,
        "existing_product_horizon_anchor_accepted": False,
        "exact_protocol_raw_match_required": True,
        "single_first_revision_required": True,
        "created_equals_updated_required": True,
        "execution_root_absent_at_observation_required": True,
        "model_output_count_at_observation_required": 0,
        "expected_receipt_artifact_id_must_be_external": True,
        "protocol_alone_authorizes_execution": False,
    }


def _validate_anchor_receipt_shape(payload: Mapping[str, object]) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "execution_protocol_id",
            "protocol_raw_sha256",
            "protocol_raw_bytes",
            "gist_owner",
            "gist_id",
            "gist_url",
            "filename",
            "public",
            "history_version",
            "history_revision_count",
            "first_revision",
            "created_at",
            "updated_at",
            "api_raw_url",
            "revision_raw_url",
            "observation_transport",
            "observed_at_utc",
            "observed_raw_sha256",
            "observed_raw_bytes",
            "exact_protocol_raw_match",
            "execution_root",
            "execution_root_existed_at_observation",
            "model_output_count_at_observation",
            "qualification_report_existed_at_observation",
            "artifact_id",
        },
        "public anchor receipt",
    )
    if payload["schema_version"] != RELATIONSHIP_READER_EXECUTION_PUBLIC_ANCHOR_RECEIPT_SCHEMA_VERSION:
        raise ValueError("public anchor receipt schema drifted")
    for name in ("execution_protocol_id", "protocol_raw_sha256", "observed_raw_sha256"):
        _digest(payload[name], f"anchor {name}")
    _positive_integer(payload["protocol_raw_bytes"], "anchor protocol_raw_bytes")
    _positive_integer(payload["observed_raw_bytes"], "anchor observed_raw_bytes")
    _text(payload["gist_owner"], "anchor gist_owner")
    _hex_text(payload["gist_id"], "anchor gist_id", lengths={32})
    _hex_text(payload["history_version"], "anchor history_version", lengths={40})
    for name in (
        "gist_url",
        "filename",
        "api_raw_url",
        "revision_raw_url",
        "execution_root",
    ):
        _text(payload[name], f"anchor {name}")
    _github_utc_timestamp(payload["created_at"], "anchor created_at")
    _github_utc_timestamp(payload["updated_at"], "anchor updated_at")
    _github_utc_timestamp(payload["observed_at_utc"], "anchor observed_at_utc")
    if payload["created_at"] != payload["updated_at"]:
        raise ValueError("public anchor must be an unchanged first revision")
    if (
        _integer(payload["history_revision_count"], "anchor history_revision_count") != 1
        or payload["first_revision"] is not True
    ):
        raise ValueError("public anchor must contain exactly one first revision")
    if payload["public"] is not True or payload["exact_protocol_raw_match"] is not True:
        raise ValueError("public anchor must be public and an exact protocol raw match")
    if payload["observation_transport"] != "unauthenticated_github_rest_api_and_raw_http":
        raise ValueError("public anchor requires independent unauthenticated observation")
    if payload["execution_root_existed_at_observation"] is not False:
        raise ValueError("public anchor must precede execution-root creation")
    if (
        _nonnegative_integer(
            payload["model_output_count_at_observation"],
            "anchor model_output_count_at_observation",
        )
        != 0
    ):
        raise ValueError("public anchor must precede all model output")
    if payload["qualification_report_existed_at_observation"] is not False:
        raise ValueError("public anchor must precede the qualification report")
    _validate_artifact_id(payload, "public anchor receipt")


def _validate_preflight_manifest_against_files(
    manifest: Mapping[str, object],
    *,
    qualification_protocol_id: str,
    rows: list[dict[str, object]],
) -> None:
    _exact_keys(
        manifest,
        {
            "schema_version",
            "protocol_id",
            "files",
            "file_count",
            "model_output_count",
            "external_public_anchor_created",
            "qualification_execution_authorized",
            "artifact_id",
        },
        "preflight manifest",
    )
    if manifest["protocol_id"] != qualification_protocol_id:
        raise ValueError("preflight manifest protocol lineage mismatch")
    if (
        manifest["file_count"] != 7
        or manifest["model_output_count"] != 0
        or manifest["external_public_anchor_created"] is not False
        or manifest["qualification_execution_authorized"] is not False
    ):
        raise ValueError("preflight manifest honesty boundary drifted")
    receipts = manifest["files"]
    if not isinstance(receipts, list) or len(receipts) != 7:
        raise ValueError("preflight manifest must contain seven file receipts")
    row_by_path = {str(row["path"]): row for row in rows}
    observed_paths: set[str] = set()
    for index, value in enumerate(receipts):
        receipt = _mapping(value, f"preflight manifest receipt {index}")
        _exact_keys(receipt, {"path", "raw_sha256", "raw_bytes", "artifact_id"}, "preflight receipt")
        path = _relative_posix_path(receipt["path"], "preflight receipt path")
        if path not in _PREFLIGHT_MANIFEST_RECEIPT_PATHS or path in observed_paths:
            raise ValueError(f"preflight manifest receipt path drifted: {path}")
        row = row_by_path[path]
        if (
            receipt["raw_sha256"] != row["raw_sha256"]
            or receipt["raw_bytes"] != row["raw_bytes"]
            or receipt["artifact_id"] != row["artifact_id"]
        ):
            raise ValueError(f"preflight manifest receipt identity mismatch: {path}")
        observed_paths.add(path)
    if observed_paths != _PREFLIGHT_MANIFEST_RECEIPT_PATHS:
        raise ValueError("preflight manifest receipt set is incomplete")


def _validate_file_entry_array(value: object, *, field_name: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty array")
    rows: list[Mapping[str, object]] = []
    paths: list[str] = []
    for index, raw_row in enumerate(value):
        row = _mapping(raw_row, f"{field_name}[{index}]")
        _exact_keys(row, {"path", "raw_sha256", "raw_bytes"}, f"{field_name}[{index}]")
        path = _relative_posix_path(row["path"], f"{field_name}[{index}].path")
        _digest(row["raw_sha256"], f"{field_name}[{index}].raw_sha256")
        _nonnegative_integer(row["raw_bytes"], f"{field_name}[{index}].raw_bytes")
        rows.append(row)
        paths.append(path)
    if paths != sorted(paths, key=lambda item: item.encode("utf-8")):
        raise ValueError(f"{field_name} must use UTF-8 ordinal path order")
    if len(paths) != len(set(paths)):
        raise ValueError(f"{field_name} contains duplicate paths")
    folded = [path.casefold() for path in paths]
    if len(folded) != len(set(folded)):
        raise ValueError(f"{field_name} contains a casefold path collision")
    return rows


def _enumerate_repository_python_sources(root: pathlib.Path) -> list[pathlib.Path]:
    packages_root = root / "packages"
    _assert_directory_without_reparse(packages_root, root=root, field_name="packages root")
    paths: list[pathlib.Path] = []
    for package_root in sorted(packages_root.iterdir(), key=lambda path: path.name.encode("utf-8")):
        src_root = package_root / "src"
        if not os.path.lexists(src_root):
            continue
        _assert_directory_without_reparse(src_root, root=root, field_name="package source root")
        paths.extend(_enumerate_python_files(src_root, repository_root=root))
    if not paths:
        raise ValueError("repository source-tree coverage found no Python files")
    return paths


def _enumerate_python_files(src_root: pathlib.Path, *, repository_root: pathlib.Path) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for current, dirnames, filenames in os.walk(src_root, topdown=True, followlinks=False):
        current_path = pathlib.Path(current)
        _assert_directory_without_reparse(current_path, root=repository_root, field_name="source directory")
        for dirname in tuple(dirnames):
            _assert_directory_without_reparse(
                current_path / dirname,
                root=repository_root,
                field_name="source directory",
            )
        for filename in filenames:
            if filename.endswith(".py"):
                paths.append(current_path / filename)
    return paths


def _enumerate_tree_files(root: pathlib.Path, *, field_name: str) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = pathlib.Path(current)
        _assert_directory_without_reparse(current_path, root=root, field_name=f"{field_name} directory")
        for dirname in tuple(dirnames):
            _assert_directory_without_reparse(
                current_path / dirname,
                root=root,
                field_name=f"{field_name} directory",
            )
        paths.extend(current_path / filename for filename in filenames)
    return paths


def _file_entries(
    *,
    root: pathlib.Path,
    paths: list[pathlib.Path],
    field_name: str,
) -> list[dict[str, object]]:
    relative_and_path = [
        (
            _relative_to_root(path, root=root, field_name=field_name).as_posix(),
            path,
        )
        for path in paths
    ]
    relative_and_path.sort(key=lambda pair: pair[0].encode("utf-8"))
    relative_paths = [relative for relative, _path in relative_and_path]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError(f"{field_name} contains duplicate paths")
    if len({path.casefold() for path in relative_paths}) != len(relative_paths):
        raise ValueError(f"{field_name} contains a casefold path collision")
    entries: list[dict[str, object]] = []
    for relative, path in relative_and_path:
        raw_sha256, raw_bytes = _hash_stable_regular_file(
            path,
            root=root,
            field_name=f"{field_name} {relative}",
        )
        entries.append(
            {
                "path": relative,
                "raw_sha256": raw_sha256,
                "raw_bytes": raw_bytes,
            }
        )
    return entries


def _hash_stable_regular_file(
    path: pathlib.Path,
    *,
    root: pathlib.Path,
    field_name: str,
    require_single_hardlink: bool = True,
) -> tuple[str, int]:
    candidate, before = _validated_regular_file_before_read(
        path,
        root=root,
        field_name=field_name,
        require_single_hardlink=require_single_hardlink,
    )
    digest = hashlib.sha256()
    raw_bytes = 0
    with candidate.open("rb") as handle:
        during = os.fstat(handle.fileno())
        while True:
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            raw_bytes += len(chunk)
    after = os.lstat(candidate)
    _validate_stable_file_identity(before, during=during, after=after, field_name=field_name)
    if raw_bytes != before.st_size:
        raise ValueError(f"{field_name} byte count changed while being hashed")
    return digest.hexdigest(), raw_bytes


def _read_stable_regular_file(
    path: pathlib.Path,
    *,
    root: pathlib.Path,
    field_name: str,
    require_single_hardlink: bool = True,
) -> bytes:
    candidate, before = _validated_regular_file_before_read(
        path,
        root=root,
        field_name=field_name,
        require_single_hardlink=require_single_hardlink,
    )
    with candidate.open("rb") as handle:
        during = os.fstat(handle.fileno())
        chunks: list[bytes] = []
        while True:
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    after = os.lstat(candidate)
    _validate_stable_file_identity(before, during=during, after=after, field_name=field_name)
    return b"".join(chunks)


def _validated_regular_file_before_read(
    path: pathlib.Path,
    *,
    root: pathlib.Path,
    field_name: str,
    require_single_hardlink: bool,
) -> tuple[pathlib.Path, os.stat_result]:
    candidate = pathlib.Path(path).absolute()
    _relative_to_root(candidate, root=root, field_name=field_name)
    _reject_reparse_components(candidate, root=root, field_name=field_name)
    if not os.path.lexists(candidate):
        raise FileNotFoundError(f"{field_name} is absent: {candidate}")
    before = os.lstat(candidate)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{field_name} must be a regular file")
    if require_single_hardlink and before.st_nlink != 1:
        raise ValueError(f"{field_name} must have exactly one hard link")
    return candidate, before


def _validate_stable_file_identity(
    before: os.stat_result,
    *,
    during: os.stat_result,
    after: os.stat_result,
    field_name: str,
) -> None:
    if _file_identity(before) != _file_identity(during) or _file_identity(during) != _file_identity(after):
        raise ValueError(f"{field_name} changed identity while being read")


def _assert_directory_without_reparse(
    path: pathlib.Path,
    *,
    root: pathlib.Path,
    field_name: str,
) -> None:
    candidate = pathlib.Path(path).absolute()
    _relative_to_root(candidate, root=root, field_name=field_name)
    if not os.path.lexists(candidate):
        raise FileNotFoundError(f"{field_name} is absent: {candidate}")
    value = os.lstat(candidate)
    if stat.S_ISLNK(value.st_mode) or _is_reparse(value):
        raise ValueError(f"{field_name} must not be a symlink or reparse point: {candidate}")
    if not stat.S_ISDIR(value.st_mode):
        raise ValueError(f"{field_name} must be a directory: {candidate}")


def _reject_reparse_components(
    path: pathlib.Path,
    *,
    root: pathlib.Path,
    field_name: str,
) -> None:
    relative = _relative_to_root(path, root=root, field_name=field_name)
    candidate = root
    _assert_one_component_not_reparse(candidate, field_name=field_name)
    for part in relative.parts:
        candidate = candidate / part
        if os.path.lexists(candidate):
            _assert_one_component_not_reparse(candidate, field_name=field_name)


def _assert_one_component_not_reparse(path: pathlib.Path, *, field_name: str) -> None:
    value = os.lstat(path)
    if stat.S_ISLNK(value.st_mode) or _is_reparse(value):
        raise ValueError(f"{field_name} traverses a symlink or reparse point: {path}")


def _is_reparse(value: os.stat_result) -> bool:
    return bool(os.name == "nt" and getattr(value, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_nlink


def _validate_canonical_artifact(
    payload: Mapping[str, object],
    *,
    raw: bytes,
    field_name: str,
) -> None:
    _validate_artifact_id(payload, field_name)
    if raw != canonical_json_bytes(dict(payload)) + b"\n":
        raise ValueError(f"{field_name} is not canonical LF-terminated JSON")


def _validate_artifact_id(payload: Mapping[str, object], field_name: str) -> None:
    artifact_id = _digest(payload.get("artifact_id"), f"{field_name} artifact_id")
    core = {key: value for key, value in payload.items() if key != "artifact_id"}
    if artifact_id != _sha256_json(core):
        raise ValueError(f"{field_name} artifact_id mismatch")


def _with_artifact_id(core: Mapping[str, object]) -> dict[str, object]:
    if "artifact_id" in core:
        raise ValueError("artifact core must not contain artifact_id")
    return {**dict(core), "artifact_id": _sha256_json(core)}


def _sha256_json(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()


def _parse_json_object(raw: bytes, *, source: str, max_bytes: int) -> Mapping[str, object]:
    parsed = strict_json_loads(raw, max_bytes=max_bytes)
    return _mapping(parsed, source)


def _validate_execution_protocol_raw_binding(
    protocol: Mapping[str, object],
    raw: bytes,
) -> str:
    if not isinstance(raw, bytes):
        raise TypeError("execution protocol raw must be bytes")
    parsed = _parse_json_object(
        raw,
        source="execution protocol raw",
        max_bytes=8_000_000,
    )
    if parsed != protocol:
        raise ValueError("execution protocol raw bytes do not encode the supplied protocol")
    return hashlib.sha256(raw).hexdigest()


def _github_utc_timestamp(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", text) is None:
        raise ValueError(f"{field_name} must be a canonical whole-second UTC timestamp")
    try:
        parsed = dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid UTC timestamp") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != text:
        raise ValueError(f"{field_name} must be a canonical whole-second UTC timestamp")
    return text


def _validate_absolute_file_pin(payload: Mapping[str, object], field_name: str) -> None:
    _exact_keys(payload, {"path", "raw_sha256", "raw_bytes"}, field_name)
    _absolute_windows_path_text(payload["path"], f"{field_name} path")
    _digest(payload["raw_sha256"], f"{field_name} raw_sha256")
    _positive_integer(payload["raw_bytes"], f"{field_name} raw_bytes")


def _validate_relative_file_pin(payload: Mapping[str, object], field_name: str) -> None:
    _exact_keys(payload, {"path", "raw_sha256", "raw_bytes"}, field_name)
    _relative_posix_path(payload["path"], f"{field_name} path")
    _digest(payload["raw_sha256"], f"{field_name} raw_sha256")
    _positive_integer(payload["raw_bytes"], f"{field_name} raw_bytes")


def _absolute_directory(value: pathlib.Path, field_name: str) -> pathlib.Path:
    path = pathlib.Path(value).absolute()
    if not path.is_absolute() or not os.path.lexists(path):
        raise FileNotFoundError(f"{field_name} is absent: {path}")
    _assert_directory_without_reparse(path, root=path, field_name=field_name)
    return path


def _absolute_windows_path_text(value: object, field_name: str) -> str:
    text = str(value) if isinstance(value, pathlib.Path) else _text(value, field_name)
    path = pathlib.PureWindowsPath(text)
    if not path.is_absolute() or not path.drive or str(path) != text:
        raise ValueError(f"{field_name} must be a canonical absolute Windows path")
    return text


def _relative_to_root(path: pathlib.Path, *, root: pathlib.Path, field_name: str) -> pathlib.PurePath:
    candidate = pathlib.Path(path).absolute()
    base = pathlib.Path(root).absolute()
    try:
        return candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"{field_name} must remain within its declared root") from exc


def _relative_posix_path(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    path = pathlib.PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text or "\\" in text:
        raise ValueError(f"{field_name} must be a canonical relative POSIX path")
    return text


def _record_relative_posix_path(value: object, field_name: str) -> str:
    """Validate a wheel RECORD path while retaining safe parent traversal.

    Wheels installed on Windows can legitimately record executables as paths
    such as ``../../Scripts/tool.exe``.  Resolution is still confined to
    ``sys.prefix`` by the caller after joining against the distribution base.
    """

    text = _text(value, field_name)
    path = pathlib.PurePosixPath(text)
    parent_prefix_open = True
    parent_traversal_is_prefix = True
    for part in path.parts:
        if part == "..":
            if not parent_prefix_open:
                parent_traversal_is_prefix = False
        else:
            parent_prefix_open = False
    if (
        path.is_absolute()
        or path.as_posix() != text
        or "\\" in text
        or text in {".", ".."}
        or any(part == "." for part in path.parts)
        or not parent_traversal_is_prefix
    ):
        raise ValueError(f"{field_name} must be a canonical relative POSIX RECORD path")
    return text


def _is_package_python_source_path(value: str) -> bool:
    path = pathlib.PurePosixPath(value)
    return len(path.parts) >= 4 and path.parts[0] == "packages" and path.parts[2] == "src" and value.endswith(".py")


def _deep_mutable_copy(value: object) -> object:
    if isinstance(value, dict):
        return {key: _deep_mutable_copy(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_deep_mutable_copy(item) for item in value]
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _exact_keys(payload: Mapping[str, object], expected: set[str], field_name: str) -> None:
    missing = sorted(expected - set(payload))
    extra = sorted(set(payload) - expected)
    if missing or extra:
        raise ValueError(f"{field_name} keys mismatch; missing={missing}, extra={extra}")


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value


def _digest(value: object, field_name: str) -> str:
    return _hex_text(value, field_name, lengths={64})


def _hex_text(value: object, field_name: str, *, lengths: set[int]) -> str:
    text = _text(value, field_name)
    if len(text) not in lengths or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be lowercase hexadecimal with length in {sorted(lengths)}")
    return text


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _positive_integer(value: object, field_name: str) -> int:
    result = _integer(value, field_name)
    if result < 1:
        raise ValueError(f"{field_name} must be positive")
    return result


def _nonnegative_integer(value: object, field_name: str) -> int:
    result = _integer(value, field_name)
    if result < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return result


__all__ = [
    "BGE_M3_MODEL_ID",
    "BGE_M3_MODEL_REVISION",
    "DEFAULT_EXECUTION_CLI_RELATIVE_PATH",
    "RELATIONSHIP_READER_BGE_SNAPSHOT_TREE_SCHEMA_VERSION",
    "RELATIONSHIP_READER_EXECUTION_PREFLIGHT_BINDING_SCHEMA_VERSION",
    "RELATIONSHIP_READER_EXECUTION_PROTOCOL_SCHEMA_VERSION",
    "RELATIONSHIP_READER_EXECUTION_PUBLIC_ANCHOR_RECEIPT_SCHEMA_VERSION",
    "RELATIONSHIP_READER_EXECUTION_RUNTIME_IDENTITY_SCHEMA_VERSION",
    "RELATIONSHIP_READER_EXECUTION_SOURCE_TREE_SCHEMA_VERSION",
    "RELATIONSHIP_READER_EXECUTION_INTEGRITY_RECEIPT_SCHEMA_VERSION",
    "RelationshipConditionReaderQualificationIntegrityGuard",
    "build_bge_m3_snapshot_tree_manifest",
    "build_relationship_condition_reader_execution_preflight_binding",
    "build_relationship_condition_reader_execution_source_tree_manifest",
    "build_relationship_condition_reader_qualification_execution_protocol",
    "build_relationship_condition_reader_qualification_integrity_receipt",
    "build_relationship_condition_reader_qualification_public_anchor_receipt",
    "build_relationship_condition_reader_qualification_runtime_identity",
    "load_relationship_condition_reader_qualification_execution_protocol",
    "relationship_condition_reader_qualification_execution_protocol_id",
    "relationship_condition_reader_qualification_integrity_guard",
    "validate_bge_m3_snapshot_tree_manifest",
    "validate_relationship_condition_reader_execution_preflight_binding",
    "validate_relationship_condition_reader_execution_source_tree_manifest",
    "validate_relationship_condition_reader_qualification_execution_protocol",
    "validate_relationship_condition_reader_qualification_public_anchor_receipt",
    "validate_relationship_condition_reader_qualification_runtime_identity",
]
