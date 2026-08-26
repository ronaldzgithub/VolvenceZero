#!/usr/bin/env python3
"""Materialize the model-free, post-hoc attempt03 action-divergence audit.

This script is deliberately independent from the Product Horizon runner.  It
reads only the immutable attempt03 evidence tree, identifies each arm record
by ``(arm_id, subject_scope, decision_id)``, and matches records across arms on
``(subject_scope, decision_id)``.  It never treats ``credit_record_id`` as a
unique cross-arm key.  The audit does not revise the frozen attempt03 verdict
or authorize a new causal claim.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from typing import Callable, Mapping, Sequence
import uuid


sys.dont_write_bytecode = True


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RELATIONSHIP_LAB_ARTIFACT_ROOT = _REPO_ROOT / "artifacts" / "relationship_lab"
_FULL_ARM = "volvence_full"
_COMPARATORS = (
    "appendable_frozen_onboarding",
    "readable_unnamed_legacy",
)
_AUDITED_ARMS = (_FULL_ARM, *_COMPARATORS)
_EXPECTED_PROTOCOL_ID = "6a34efb81c7313595314693aef0a6bf8596582273808830ed2d36f5155ce8099"
_EXPECTED_REPORT_ARTIFACT_ID = "49bc11d614fe51f3e10e21bfe9e8d3fc9834760a0288e92bbc1b4606b432472e"
_EXPECTED_MANIFEST_ARTIFACT_ID = "e95d2396d2612668f88e47c9689cea6e3488bf41553ab3341fcf2b49253334ea"
_EXPECTED_MANIFEST_RAW_SHA256 = "e9ac26e39bf248aa7325640fe7909e9c8c849dc3e59af77df2017eef3cfca964"
_EXPECTED_PROTOCOL_RAW_SHA256 = "089e50b3eb515d851513a94d660f840520c4d8b7d90157d8a6990b5de61d8712"
_EXPECTED_REPORT_RAW_SHA256 = "11462006a89c0b19bff9e36ac5e72ccdabc4c1f0837fec6fa6f5f61856a21881"
_EXPECTED_SOURCE_PROTOCOL_ID = "048b73d4a412b4444fb469be0d9daa6d2a26e9920c743804da8f36dc331691ae"
_EXPECTED_SOURCE_PUBLIC_PLAN_ID = "93474269cb5b9d066e68253d6f2e51fbc0d3bf3b6a7fe2a748b140d136bb812b"
_EXPECTED_SOURCE_SEALED_BUNDLE_ID = "d502b78364dcb7024b229f4bb10c0cddb002488c3a360edd7aa0932c345d8b5a"
_EXPECTED_SOURCE_PUBLIC_RAW_SHA256 = "267aabc81e2f4d8127833541e0496febfb16a2cd5001adba9ec26c29a0ee4c09"
_EXPECTED_SOURCE_SEALED_RAW_SHA256 = "421a7f21f2a55fa4ee44cac33347c61e3aaf574e0acab431a05b98d64a1b515d"
_EXPECTED_VERDICT = "typed_control_product_horizon_executed_effect_not_observed"
_EXPECTED_READER_ARTIFACT_SCHEMA_VERSION = "relationship-condition-reader-artifact.v1"
_EXPECTED_FORECAST_SNAPSHOT_SCHEMA_VERSION = "preference-action-forecast-snapshot.v1"
_EXPECTED_GATE_ARTIFACT_ID = "relationship-action-gate-zero-init"
_EXPECTED_GATE_ARTIFACT_VERSION = 1
_POST_HOC_READER_AUTHORITY_ROOT = (
    _RELATIONSHIP_LAB_ARTIFACT_ROOT / "relationship_condition_reader_qualification_preflight_v4_20260825_p723796027a64"
)
_POST_HOC_READER_AUTHORITY_ROOT_RELATIVE = (
    "artifacts/relationship_lab/relationship_condition_reader_qualification_preflight_v4_20260825_p723796027a64"
)
_EXPECTED_POST_HOC_READER_AUTHORITY_MANIFEST_ARTIFACT_ID = (
    "7dd323f022521f31a49d2a786524fa8fbc5bb641377bd998b21bb75aa2cc9e89"
)
_EXPECTED_POST_HOC_READER_AUTHORITY_MANIFEST_RAW_SHA256 = (
    "ee6d5abadf02fb3dfe44f57b2fb241a8db7a7bb7d006e87f1528247e283904b6"
)
_EXPECTED_POST_HOC_READER_AUTHORITY_MANIFEST_RAW_BYTES = 1839
_EXPECTED_POST_HOC_READER_AUTHORITY_PROTOCOL_ID = "723796027a64a627f8f858e4499d5956ad43d7c45bbc20e20f7b04fd197c8e6b"
_EXPECTED_POST_HOC_READER_AUTHORITY_PROTOCOL_RAW_SHA256 = (
    "fe4ef1efad7b03121ee1ee4f956c2dfc50cbf9dc66449f335221620f8c120dce"
)
_EXPECTED_POST_HOC_READER_AUTHORITY_PROTOCOL_RAW_BYTES = 5658
_OWNER_HISTORY_AUDIT_ROOT = (
    _RELATIONSHIP_LAB_ARTIFACT_ROOT
    / "relationship_product_horizon_v2_attempt03_owner_history_audit_20260825_p6a34efb81c73"
)
_OWNER_HISTORY_AUDIT_ROOT_RELATIVE = (
    "artifacts/relationship_lab/relationship_product_horizon_v2_attempt03_owner_history_audit_20260825_p6a34efb81c73"
)
_EXPECTED_OWNER_HISTORY_MANIFEST_ARTIFACT_ID = "2973f3a9198714ae37c8404179c716b654bab12b3233fc04a25f864d86edd0f8"
_EXPECTED_OWNER_HISTORY_MANIFEST_RAW_SHA256 = "fc09d48e5d80a7160051d63fd5c9d9269925f7491792031713864362ec4a9337"
_EXPECTED_OWNER_HISTORY_MANIFEST_RAW_BYTES = 905
_EXPECTED_OWNER_HISTORY_REPORT_ARTIFACT_ID = "ff44bec452e9e9391c972ef8664a9631aa65958038ed3515afb9ad43c85a997f"
_EXPECTED_OWNER_HISTORY_REPORT_RAW_SHA256 = "340a116ebb6b62f1c3e5f9a4773bd6556dcf1ff1cce2e994dbb108f7b3525297"
_EXPECTED_OWNER_HISTORY_REPORT_RAW_BYTES = 34313
_EXPECTED_OWNER_HISTORY_ROWS_ARTIFACT_ID = "d7d845e2d189d74962b3d1ec51037c4ebf69ef36dd9e543aa5639d569a1aa97e"
_EXPECTED_OWNER_HISTORY_ROWS_RAW_SHA256 = "b879f850123f62c0b2f2999a0922776ef9307e1c2026a74ca94e9e64dd3318e0"
_EXPECTED_OWNER_HISTORY_ROWS_RAW_BYTES = 4526789
_RECORD_IDENTITY_FIELDS = ["arm_id", "subject_scope", "decision_id"]
_CROSS_ARM_MATCH_FIELDS = ["subject_scope", "decision_id"]
_EXPECTED_ACTION_IDS = frozenset(
    {
        "neutral_noop",
        "respect_space_with_return_option",
        "stay_present_without_probe",
    }
)
_CONDITION_TO_READER_LABEL = {
    "agency_under_override": "agency_displacement",
    "connection_under_exclusion": "belonging_erasure",
}
_POSITIVE_OUTCOME_IDS = frozenset({"helped", "felt_heard"})
_SAFETY_NEGATIVE_OUTCOME_IDS = frozenset({"missed", "over_directive"})
_EXPECTED_OUTCOME_IDS = _POSITIVE_OUTCOME_IDS | _SAFETY_NEGATIVE_OUTCOME_IDS
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OUTPUT_FILES = (
    "pair_rows.json",
    "unique_decision_rows.json",
    "report.json",
    "manifest.json",
)


class AuditContractError(ValueError):
    """Raised when immutable evidence or an audit invariant is violated."""


def _fail(message: str) -> None:
    raise AuditContractError(message)


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AuditContractError(f"payload is not canonical-JSON encodable: {exc}") from exc
    return rendered.encode("utf-8")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _artifact_id(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("artifact_id", None)
    return _sha256_bytes(_canonical_json_bytes(unsigned))


def _with_artifact_id(payload: Mapping[str, object]) -> dict[str, object]:
    unsigned = dict(payload)
    if "artifact_id" in unsigned:
        _fail("artifact payload must not predeclare artifact_id")
    return {**unsigned, "artifact_id": _artifact_id(unsigned)}


def _with_row_id(payload: Mapping[str, object]) -> dict[str, object]:
    unsigned = dict(payload)
    if "row_id" in unsigned:
        _fail("row payload must not predeclare row_id")
    return {**unsigned, "row_id": _sha256_bytes(_canonical_json_bytes(unsigned))}


def _reject_constant(value: str) -> object:
    raise AuditContractError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AuditContractError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        text = raw.decode("utf-8", errors="strict")
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditContractError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        _fail(f"{label} must contain one JSON object")
    return parsed


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        _fail(f"{label} must be an array")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    if not math.isfinite(float(value)):
        _fail(f"{label} must be finite")
    return float(value)


def _bounded_number(
    value: object,
    label: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    numeric = _number(value, label)
    if numeric < minimum or numeric > maximum:
        _fail(f"{label} must be within [{minimum}, {maximum}]")
    return numeric


def _text_array(value: object, label: str, *, allow_empty: bool = False) -> list[str]:
    values = _array(value, label)
    if not allow_empty and not values:
        _fail(f"{label} must not be empty")
    return [_text(item, f"{label}[{index}]") for index, item in enumerate(values)]


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{label} must be a boolean")
    return value


def _sha256(value: object, label: str) -> str:
    text = _text(value, label)
    if _SHA256_RE.fullmatch(text) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return text


def _require_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        _fail(f"{label} mismatch: observed={observed!r}, expected={expected!r}")


def _require_artifact_identity(payload: Mapping[str, object], *, label: str) -> str:
    observed = _sha256(payload.get("artifact_id"), f"{label}.artifact_id")
    expected = _artifact_id(payload)
    _require_equal(observed, expected, f"{label}.artifact_id")
    return observed


def _repo_relative(path: Path, *, label: str) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise AuditContractError(f"{label} must be inside repository root: {resolved}") from exc
    return relative.as_posix()


def _audit_output_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    allowed_root = _RELATIONSHIP_LAB_ARTIFACT_ROOT.resolve()
    try:
        relative = resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise AuditContractError(f"{label} must be inside relationship-lab artifact root: {resolved}") from exc
    if not relative.parts:
        _fail(f"{label} must be a child of relationship-lab artifact root")
    return resolved


def _audit_output_label(path: Path, *, label: str) -> str:
    resolved = _audit_output_path(path, label=label)
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        # Tests may replace the allowed root with a pytest-owned temporary
        # directory.  Production CLI execution keeps the immutable constant.
        return resolved.relative_to(_RELATIONSHIP_LAB_ARTIFACT_ROOT.resolve()).as_posix()


def _join_contract(*, include_reason: bool = False) -> dict[str, object]:
    contract: dict[str, object] = {
        "record_identity_fields": list(_RECORD_IDENTITY_FIELDS),
        "cross_arm_match_fields": list(_CROSS_ARM_MATCH_FIELDS),
        "credit_record_id_is_unique_join_key": False,
    }
    if include_reason:
        contract["reason"] = "credit_record_id is intentionally matched across arms and is not globally unique"
    return contract


def _normalized_member_path(value: object, *, label: str) -> str:
    text = _text(value, label)
    candidate = PurePosixPath(text)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        _fail(f"{label} must be a normalized relative POSIX path: {text!r}")
    if candidate.as_posix() != text or "\\" in text:
        _fail(f"{label} is not canonical POSIX form: {text!r}")
    return text


def _load_post_hoc_reader_crosswalk_authority() -> dict[str, object]:
    """Load the post-attempt preflight only as a frozen semantic authority.

    This authority was frozen after attempt03.  It can authenticate the label
    crosswalk used by this diagnosis, but it cannot retroactively preregister
    an analysis or qualify the reader that ran in attempt03.
    """

    root = _POST_HOC_READER_AUTHORITY_ROOT.resolve()
    if not root.is_dir():
        _fail(f"post-hoc reader authority root does not exist: {root}")
    _require_equal(
        _repo_relative(root, label="post-hoc reader authority root"),
        _POST_HOC_READER_AUTHORITY_ROOT_RELATIVE,
        "post-hoc reader authority root",
    )
    manifest_path = root / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    _require_equal(
        len(manifest_raw),
        _EXPECTED_POST_HOC_READER_AUTHORITY_MANIFEST_RAW_BYTES,
        "post-hoc reader authority manifest byte count",
    )
    _require_equal(
        _sha256_bytes(manifest_raw),
        _EXPECTED_POST_HOC_READER_AUTHORITY_MANIFEST_RAW_SHA256,
        "post-hoc reader authority manifest raw SHA-256",
    )
    manifest = _strict_json_object(manifest_raw, label="post-hoc reader authority manifest")
    _require_artifact_identity(manifest, label="post-hoc reader authority manifest")
    _require_equal(
        manifest.get("artifact_id"),
        _EXPECTED_POST_HOC_READER_AUTHORITY_MANIFEST_ARTIFACT_ID,
        "post-hoc reader authority manifest artifact_id",
    )
    _require_equal(
        manifest.get("schema_version"),
        "relationship-condition-reader-qualification-preflight-manifest.v1",
        "post-hoc reader authority manifest schema_version",
    )
    _require_equal(
        manifest.get("protocol_id"),
        _EXPECTED_POST_HOC_READER_AUTHORITY_PROTOCOL_ID,
        "post-hoc reader authority manifest protocol_id",
    )
    for field in (
        "external_public_anchor_created",
        "qualification_execution_authorized",
    ):
        _require_equal(manifest.get(field), False, f"post-hoc reader authority manifest {field}")
    _require_equal(manifest.get("model_output_count"), 0, "post-hoc reader authority model output count")

    entries = _array(manifest.get("files"), "post-hoc reader authority manifest.files")
    protocol_entries = [
        _object(item, "post-hoc reader authority manifest file")
        for item in entries
        if _object(item, "post-hoc reader authority manifest file").get("path") == "protocol.json"
    ]
    _require_equal(len(protocol_entries), 1, "post-hoc reader authority protocol entry count")
    protocol_entry = protocol_entries[0]
    _require_equal(
        protocol_entry.get("artifact_id"),
        None,
        "post-hoc reader authority protocol manifest artifact_id",
    )
    _require_equal(
        protocol_entry.get("raw_bytes"),
        _EXPECTED_POST_HOC_READER_AUTHORITY_PROTOCOL_RAW_BYTES,
        "post-hoc reader authority protocol manifest byte count",
    )
    _require_equal(
        protocol_entry.get("raw_sha256"),
        _EXPECTED_POST_HOC_READER_AUTHORITY_PROTOCOL_RAW_SHA256,
        "post-hoc reader authority protocol manifest raw SHA-256",
    )
    protocol_path = root / "protocol.json"
    protocol_raw = protocol_path.read_bytes()
    _require_equal(
        len(protocol_raw),
        _EXPECTED_POST_HOC_READER_AUTHORITY_PROTOCOL_RAW_BYTES,
        "post-hoc reader authority protocol byte count",
    )
    _require_equal(
        _sha256_bytes(protocol_raw),
        _EXPECTED_POST_HOC_READER_AUTHORITY_PROTOCOL_RAW_SHA256,
        "post-hoc reader authority protocol raw SHA-256",
    )
    protocol = _strict_json_object(protocol_raw, label="post-hoc reader authority protocol")
    _require_equal(
        protocol.get("schema_version"),
        "relationship-condition-reader-qualification-protocol.v2",
        "post-hoc reader authority protocol schema_version",
    )
    _require_equal(
        _sha256_bytes(_canonical_json_bytes(protocol)),
        _EXPECTED_POST_HOC_READER_AUTHORITY_PROTOCOL_ID,
        "post-hoc reader authority protocol canonical identity",
    )
    _require_equal(
        protocol.get("label_crosswalk"),
        _CONDITION_TO_READER_LABEL,
        "post-hoc reader authority label crosswalk",
    )
    claims = _object(protocol.get("claims"), "post-hoc reader authority claims")
    for field in (
        "campaign_execution_admitted",
        "condition_reader_qualified",
        "formal_evidence_authorized",
        "four_able_complete",
        "readable_product_effect",
    ):
        _require_equal(claims.get(field), False, f"post-hoc reader authority claim {field}")
    execution = _object(protocol.get("execution"), "post-hoc reader authority execution")
    _require_equal(
        execution.get("qualification_execution_authorized"),
        False,
        "post-hoc reader authority execution authorization",
    )
    _require_equal(execution.get("model_output_count"), 0, "post-hoc reader authority execution model outputs")
    return {
        "authority_role": "post_hoc_semantic_crosswalk_only",
        "authority_timing": "frozen_after_attempt03",
        "pre_registered_attempt03_analysis": False,
        "reader_qualification_result": False,
        "qualification_execution_authorized": False,
        "root": _POST_HOC_READER_AUTHORITY_ROOT_RELATIVE,
        "manifest": {
            "path": f"{_POST_HOC_READER_AUTHORITY_ROOT_RELATIVE}/manifest.json",
            "bytes": len(manifest_raw),
            "raw_sha256": _sha256_bytes(manifest_raw),
            "artifact_id": manifest["artifact_id"],
            "schema_version": manifest["schema_version"],
        },
        "protocol": {
            "path": f"{_POST_HOC_READER_AUTHORITY_ROOT_RELATIVE}/protocol.json",
            "bytes": len(protocol_raw),
            "raw_sha256": _sha256_bytes(protocol_raw),
            "protocol_id": _EXPECTED_POST_HOC_READER_AUTHORITY_PROTOCOL_ID,
            "schema_version": protocol["schema_version"],
        },
        "label_crosswalk": deepcopy(_CONDITION_TO_READER_LABEL),
    }


def _load_owner_history_audit_authority() -> dict[str, object]:
    """Strictly bind the separately materialized post-hoc owner audit."""

    root = _OWNER_HISTORY_AUDIT_ROOT.resolve()
    if not root.is_dir():
        _fail(f"owner-history audit root does not exist: {root}")
    _require_equal(
        _repo_relative(root, label="owner-history audit root"),
        _OWNER_HISTORY_AUDIT_ROOT_RELATIVE,
        "owner-history audit root",
    )
    _require_equal(
        sorted(path.name for path in root.iterdir()),
        ["manifest.json", "report.json", "rows.json"],
        "owner-history audit exact file set",
    )
    expected_files = {
        "report.json": {
            "bytes": _EXPECTED_OWNER_HISTORY_REPORT_RAW_BYTES,
            "raw_sha256": _EXPECTED_OWNER_HISTORY_REPORT_RAW_SHA256,
            "artifact_id": _EXPECTED_OWNER_HISTORY_REPORT_ARTIFACT_ID,
        },
        "rows.json": {
            "bytes": _EXPECTED_OWNER_HISTORY_ROWS_RAW_BYTES,
            "raw_sha256": _EXPECTED_OWNER_HISTORY_ROWS_RAW_SHA256,
            "artifact_id": _EXPECTED_OWNER_HISTORY_ROWS_ARTIFACT_ID,
        },
    }

    manifest_path = root / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    _require_equal(
        len(manifest_raw),
        _EXPECTED_OWNER_HISTORY_MANIFEST_RAW_BYTES,
        "owner-history manifest byte count",
    )
    _require_equal(
        _sha256_bytes(manifest_raw),
        _EXPECTED_OWNER_HISTORY_MANIFEST_RAW_SHA256,
        "owner-history manifest raw SHA-256",
    )
    manifest = _strict_json_object(manifest_raw, label="owner-history manifest")
    _require_artifact_identity(manifest, label="owner-history manifest")
    _require_equal(
        manifest.get("artifact_id"),
        _EXPECTED_OWNER_HISTORY_MANIFEST_ARTIFACT_ID,
        "owner-history manifest artifact_id",
    )
    _require_equal(
        manifest.get("schema_version"),
        "relationship-product-horizon-attempt03-owner-history-audit-manifest.v1",
        "owner-history manifest schema_version",
    )
    _require_equal(manifest.get("manifest_written_last"), True, "owner-history manifest order")
    _require_equal(
        manifest.get("report_artifact_id"),
        _EXPECTED_OWNER_HISTORY_REPORT_ARTIFACT_ID,
        "owner-history manifest report artifact_id",
    )
    _require_equal(
        manifest.get("source_attempt_manifest_artifact_id"),
        _EXPECTED_MANIFEST_ARTIFACT_ID,
        "owner-history source attempt manifest artifact_id",
    )
    _require_equal(
        manifest.get("source_attempt_manifest_raw_sha256"),
        _EXPECTED_MANIFEST_RAW_SHA256,
        "owner-history source attempt manifest raw SHA-256",
    )
    manifest_entries = {
        _normalized_member_path(
            _object(item, "owner-history manifest file").get("path"),
            label="owner-history manifest file path",
        ): _object(item, "owner-history manifest file")
        for item in _array(manifest.get("files"), "owner-history manifest files")
    }
    _require_equal(
        sorted(manifest_entries),
        sorted(expected_files),
        "owner-history manifest output set",
    )

    loaded: dict[str, tuple[dict[str, object], bytes]] = {}
    for name, expected in expected_files.items():
        raw = (root / name).read_bytes()
        _require_equal(len(raw), expected["bytes"], f"owner-history {name} byte count")
        _require_equal(
            _sha256_bytes(raw),
            expected["raw_sha256"],
            f"owner-history {name} raw SHA-256",
        )
        payload = _strict_json_object(raw, label=f"owner-history {name}")
        _require_artifact_identity(payload, label=f"owner-history {name}")
        _require_equal(
            payload.get("artifact_id"),
            expected["artifact_id"],
            f"owner-history {name} artifact_id",
        )
        entry = manifest_entries[name]
        _require_equal(entry.get("bytes"), len(raw), f"owner-history manifest {name} bytes")
        _require_equal(
            entry.get("raw_sha256"),
            _sha256_bytes(raw),
            f"owner-history manifest {name} raw SHA-256",
        )
        _require_equal(
            entry.get("artifact_id"),
            payload.get("artifact_id"),
            f"owner-history manifest {name} artifact_id",
        )
        loaded[name] = (payload, raw)

    report, report_raw = loaded["report.json"]
    rows, rows_raw = loaded["rows.json"]
    _require_equal(
        report.get("schema_version"),
        "relationship-product-horizon-attempt03-owner-history-audit-report.v1",
        "owner-history report schema_version",
    )
    _require_equal(
        rows.get("schema_version"),
        "relationship-product-horizon-attempt03-owner-history-rows.v1",
        "owner-history rows schema_version",
    )
    _require_equal(
        report.get("audit_scope"),
        "post_hoc_owner_history_hard_window_contribution",
        "owner-history report audit_scope",
    )
    _require_equal(rows.get("audit_scope"), report.get("audit_scope"), "owner-history rows scope")
    _require_equal(rows.get("source_protocol_id"), _EXPECTED_PROTOCOL_ID, "owner-history rows protocol")
    _require_equal(rows.get("row_count"), 36, "owner-history rows row count")
    _require_equal(len(_array(rows.get("rows"), "owner-history rows")), 36, "owner-history row array")
    report_rows = _object(
        _object(report.get("outputs"), "owner-history report outputs").get("rows"),
        "owner-history report rows output",
    )
    _require_equal(
        report_rows,
        {
            "path": "rows.json",
            "bytes": len(rows_raw),
            "raw_sha256": _sha256_bytes(rows_raw),
            "artifact_id": rows["artifact_id"],
            "row_count": 36,
        },
        "owner-history report rows output",
    )
    source_attempt = _object(report.get("source_attempt"), "owner-history report source attempt")
    for field, expected in {
        "manifest_artifact_id": _EXPECTED_MANIFEST_ARTIFACT_ID,
        "manifest_raw_sha256": _EXPECTED_MANIFEST_RAW_SHA256,
        "protocol_id": _EXPECTED_PROTOCOL_ID,
        "protocol_raw_sha256": _EXPECTED_PROTOCOL_RAW_SHA256,
        "report_artifact_id": _EXPECTED_REPORT_ARTIFACT_ID,
        "report_raw_sha256": _EXPECTED_REPORT_RAW_SHA256,
        "frozen_verdict": _EXPECTED_VERDICT,
    }.items():
        _require_equal(source_attempt.get(field), expected, f"owner-history source attempt {field}")
    owner_contract = _object(report.get("owner_contract"), "owner-history owner contract")
    _require_equal(owner_contract.get("hard_window_size"), 12, "owner-history hard window size")
    _require_equal(
        owner_contract.get("social_record_store_role"),
        "carrier_only",
        "owner-history carrier role",
    )
    _require_equal(
        owner_contract.get("unique_semantic_owner"),
        "PreferenceAboutOtherModule/preference_about_other",
        "owner-history unique semantic owner",
    )
    boundaries = _object(report.get("honest_boundaries"), "owner-history honest boundaries")
    for field, expected in {
        "mechanical_history_and_hard_window_contribution_replay_only": True,
        "post_hoc_diagnosis": True,
        "pre_registered_confirmatory_analysis": False,
        "product_causal_effect_established": False,
        "learnable_capability_established": False,
        "human_product_validation": False,
        "production_active": False,
        "reader_error_attribution_authorized": False,
    }.items():
        _require_equal(boundaries.get(field), expected, f"owner-history honest boundary {field}")
    frozen = _object(report.get("frozen_judgment_preserved"), "owner-history frozen judgment")
    _require_equal(frozen.get("verdict"), _EXPECTED_VERDICT, "owner-history frozen verdict")
    for field in (
        "attempt03_files_modified",
        "formal_evidence_authorized",
        "four_able_complete",
        "single_axis_contrast_claim_authorized",
    ):
        _require_equal(frozen.get(field), False, f"owner-history frozen judgment {field}")
    return {
        "authority_role": "external_post_hoc_owner_semantic_audit",
        "materialized": True,
        "bound": True,
        "causal_attribution_authorized": False,
        "root": _OWNER_HISTORY_AUDIT_ROOT_RELATIVE,
        "manifest": {
            "path": f"{_OWNER_HISTORY_AUDIT_ROOT_RELATIVE}/manifest.json",
            "bytes": len(manifest_raw),
            "raw_sha256": _sha256_bytes(manifest_raw),
            "artifact_id": manifest["artifact_id"],
            "schema_version": manifest["schema_version"],
        },
        "report": {
            "path": f"{_OWNER_HISTORY_AUDIT_ROOT_RELATIVE}/report.json",
            "bytes": len(report_raw),
            "raw_sha256": _sha256_bytes(report_raw),
            "artifact_id": report["artifact_id"],
            "schema_version": report["schema_version"],
        },
        "rows": {
            "path": f"{_OWNER_HISTORY_AUDIT_ROOT_RELATIVE}/rows.json",
            "bytes": len(rows_raw),
            "raw_sha256": _sha256_bytes(rows_raw),
            "artifact_id": rows["artifact_id"],
            "schema_version": rows["schema_version"],
            "row_count": rows["row_count"],
        },
    }


class _AttemptReader:
    """Read manifest-bound members from exactly one pinned attempt03 root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        if not self.root.is_dir():
            _fail(f"attempt root does not exist: {self.root}")
        self.root_relative = _repo_relative(self.root, label="attempt root")
        manifest_path = self.root / "manifest.json"
        manifest_raw = manifest_path.read_bytes()
        manifest_sha256 = _sha256_bytes(manifest_raw)
        _require_equal(
            manifest_sha256,
            _EXPECTED_MANIFEST_RAW_SHA256,
            "attempt manifest raw SHA-256",
        )
        self.manifest_raw = manifest_raw
        self.manifest_raw_sha256 = manifest_sha256
        self.manifest = _strict_json_object(manifest_raw, label="attempt manifest")
        _require_artifact_identity(self.manifest, label="attempt manifest")
        _require_equal(
            self.manifest.get("artifact_id"),
            _EXPECTED_MANIFEST_ARTIFACT_ID,
            "attempt manifest artifact_id",
        )
        _require_equal(
            self.manifest.get("schema_version"),
            "relationship-product-horizon-manifest.v1",
            "attempt manifest schema_version",
        )
        _require_equal(
            self.manifest.get("protocol_id"),
            _EXPECTED_PROTOCOL_ID,
            "attempt manifest protocol_id",
        )
        _require_equal(
            self.manifest.get("report_artifact_id"),
            _EXPECTED_REPORT_ARTIFACT_ID,
            "attempt manifest report_artifact_id",
        )
        _require_equal(
            self.manifest.get("manifest_written_last"),
            True,
            "attempt manifest manifest_written_last",
        )
        rows = _array(self.manifest.get("files"), "attempt manifest.files")
        entries: dict[str, dict[str, object]] = {}
        for index, item in enumerate(rows):
            row = _object(item, f"attempt manifest.files[{index}]")
            path = _normalized_member_path(
                row.get("path"),
                label=f"attempt manifest.files[{index}].path",
            )
            if path in entries:
                _fail(f"duplicate path in attempt manifest: {path}")
            _sha256(row.get("sha256"), f"attempt manifest.files[{index}].sha256")
            size = _integer(row.get("bytes"), f"attempt manifest.files[{index}].bytes")
            if size < 0:
                _fail(f"negative byte count in attempt manifest: {path}")
            entries[path] = row
        self.entries = entries

    def load(self, relative_path: str, *, verify_artifact_id: bool = True) -> dict[str, object]:
        path = _normalized_member_path(relative_path, label="attempt member path")
        entry = self.entries.get(path)
        if entry is None:
            _fail(f"attempt member is not bound by manifest: {path}")
        full_path = self.root.joinpath(*PurePosixPath(path).parts)
        if not full_path.is_file():
            _fail(f"attempt member is missing: {path}")
        raw = full_path.read_bytes()
        _require_equal(len(raw), entry["bytes"], f"{path} byte count")
        _require_equal(_sha256_bytes(raw), entry["sha256"], f"{path} raw SHA-256")
        payload = _strict_json_object(raw, label=path)
        if verify_artifact_id and "artifact_id" in payload:
            _require_artifact_identity(payload, label=path)
        return payload

    def reference(self, relative_path: str, payload: Mapping[str, object]) -> dict[str, object]:
        path = _normalized_member_path(relative_path, label="source reference path")
        entry = self.entries.get(path)
        if entry is None:
            _fail(f"source reference is not bound by manifest: {path}")
        reference: dict[str, object] = {
            "path": path,
            "bytes": entry["bytes"],
            "raw_sha256": entry["sha256"],
            "schema_version": payload.get("schema_version"),
        }
        artifact_id = payload.get("artifact_id")
        if artifact_id is not None:
            reference["artifact_id"] = _sha256(artifact_id, f"{path}.artifact_id")
        return reference


def _validate_protocol_and_report(
    reader: _AttemptReader,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, tuple[int, int]],
    dict[str, object],
    dict[str, str],
]:
    protocol = reader.load("protocol.json", verify_artifact_id=False)
    report = reader.load("report.json")
    _require_equal(
        reader.entries["protocol.json"]["sha256"],
        _EXPECTED_PROTOCOL_RAW_SHA256,
        "protocol raw SHA-256",
    )
    _require_equal(
        reader.entries["report.json"]["sha256"],
        _EXPECTED_REPORT_RAW_SHA256,
        "report raw SHA-256",
    )
    _require_equal(
        protocol.get("schema_version"),
        "relationship-product-horizon-campaign.v2",
        "protocol schema_version",
    )
    _require_equal(
        _sha256_bytes(_canonical_json_bytes(protocol)),
        _EXPECTED_PROTOCOL_ID,
        "protocol canonical identity",
    )
    _require_equal(report.get("protocol_id"), _EXPECTED_PROTOCOL_ID, "report protocol_id")
    _require_equal(
        report.get("artifact_id"),
        _EXPECTED_REPORT_ARTIFACT_ID,
        "report artifact_id",
    )
    _require_equal(report.get("verdict"), _EXPECTED_VERDICT, "report verdict")
    for field in (
        "formal_evidence_authorized",
        "four_able_complete",
        "human_product_validation",
        "production_active",
        "residual_steerable",
        "single_axis_contrast_claim_authorized",
        "thesis_validated",
        "user_visible_generation",
    ):
        _require_equal(report.get(field), False, f"report {field}")
    execution = _object(protocol.get("execution"), "protocol.execution")
    named_reader = _object(execution.get("named_condition_reader"), "protocol named condition reader")
    reader_artifact = _object(named_reader.get("artifact"), "protocol named reader artifact")
    reader_artifact_id = _require_artifact_identity(reader_artifact, label="protocol named reader artifact")
    _require_equal(
        reader_artifact.get("schema_version"),
        _EXPECTED_READER_ARTIFACT_SCHEMA_VERSION,
        "protocol named reader artifact schema_version",
    )
    mechanism_evidence = _object(report.get("mechanism_evidence"), "report mechanism_evidence")
    _require_equal(
        mechanism_evidence.get("reader_artifact_ids"),
        [reader_artifact_id],
        "report reader artifact identities",
    )
    report_source_bundle_id = _sha256(
        report.get("execution_source_bundle_artifact_id"),
        "report execution source bundle artifact_id",
    )
    protocol_source_tree = _object(
        execution.get("local_execution_source_tree"),
        "protocol local execution source tree",
    )
    report_source_tree = _object(
        report.get("local_execution_source_tree"),
        "report local execution source tree",
    )
    _require_equal(
        protocol_source_tree.get("schema_version"),
        "relationship-product-local-execution-source-tree.v1",
        "protocol local execution source tree schema_version",
    )
    source_tree_sha256 = _sha256(
        protocol_source_tree.get("tree_sha256"),
        "protocol local execution source tree SHA-256",
    )
    for field in ("tree_sha256", "file_count", "canonical_bytes"):
        _require_equal(
            report_source_tree.get(field),
            protocol_source_tree.get(field),
            f"report local execution source tree {field}",
        )

    execution_source_bundle_path = "inputs/execution_sources/bundle.json"
    execution_source_bundle = reader.load(execution_source_bundle_path)
    _require_equal(
        execution_source_bundle.get("schema_version"),
        "relationship-product-execution-source-bundle.v2",
        "execution source bundle schema_version",
    )
    _require_equal(
        execution_source_bundle.get("artifact_id"),
        report_source_bundle_id,
        "execution source bundle artifact_id",
    )
    bundled_source_tree = _object(
        execution_source_bundle.get("local_execution_source_tree"),
        "execution source bundle local source tree",
    )
    _require_artifact_identity(
        bundled_source_tree,
        label="execution source bundle local source tree",
    )
    for field in (
        "schema_version",
        "selector",
        "entrypoints",
        "resource_paths",
        "active_protocol_path",
        "canonicalization",
        "tree_sha256",
        "file_count",
        "canonical_bytes",
    ):
        _require_equal(
            bundled_source_tree.get(field),
            protocol_source_tree.get(field),
            f"execution source bundle local source tree {field}",
        )
    active_protocol_resource = _object(
        execution_source_bundle.get("active_protocol_resource"),
        "execution source bundle active protocol resource",
    )
    _require_equal(
        active_protocol_resource.get("schema_version"),
        "relationship-product-active-protocol-resource.v1",
        "execution source bundle active protocol resource schema_version",
    )
    _require_equal(
        active_protocol_resource.get("protocol_id"),
        _EXPECTED_PROTOCOL_ID,
        "execution source bundle active protocol resource protocol_id",
    )
    _require_equal(
        active_protocol_resource.get("raw_sha256"),
        _EXPECTED_PROTOCOL_RAW_SHA256,
        "execution source bundle active protocol resource raw SHA-256",
    )
    worker_repository_path = "scripts/run_relationship_lab_product_horizon.py"
    bundled_files = _array(
        bundled_source_tree.get("files"),
        "execution source bundle local source tree files",
    )
    worker_entries = [
        _object(item, "execution source bundle worker entry")
        for item in bundled_files
        if _object(item, "execution source bundle worker entry").get("repository_path") == worker_repository_path
    ]
    _require_equal(len(worker_entries), 1, "execution source bundle worker entry count")
    worker_entry = worker_entries[0]
    worker_raw_sha256 = _sha256(
        worker_entry.get("raw_sha256"),
        "execution source bundle worker raw SHA-256",
    )
    worker_member_path = _normalized_member_path(
        worker_entry.get("path"),
        label="execution source bundle worker member path",
    )
    worker_manifest_entry = reader.entries.get(worker_member_path)
    if worker_manifest_entry is None:
        _fail("execution source bundle worker is not bound by attempt manifest")
    _require_equal(
        worker_manifest_entry.get("sha256"),
        worker_raw_sha256,
        "attempt manifest worker raw SHA-256",
    )
    source = _object(protocol.get("source"), "protocol.source")
    _require_equal(
        source.get("source_protocol_id"),
        _EXPECTED_SOURCE_PROTOCOL_ID,
        "protocol source.source_protocol_id",
    )
    _require_equal(
        source.get("public_plan_sha256"),
        _EXPECTED_SOURCE_PUBLIC_PLAN_ID,
        "protocol source.public_plan_sha256",
    )
    _require_equal(
        source.get("sealed_evaluator_bundle_sha256"),
        _EXPECTED_SOURCE_SEALED_BUNDLE_ID,
        "protocol source.sealed_evaluator_bundle_sha256",
    )
    _require_equal(source.get("subject_count"), 8, "protocol source.subject_count")
    _require_equal(
        source.get("decision_sessions_per_subject"),
        24,
        "protocol source.decision_sessions_per_subject",
    )
    analysis = _object(protocol.get("analysis"), "protocol.analysis")
    _require_equal(
        analysis.get("analysis_unit"),
        "subject_world_clone",
        "protocol analysis.analysis_unit",
    )
    _require_equal(
        analysis.get("primary_window_decision_indices"),
        [12, 23],
        "protocol primary window",
    )
    raw_windows = _object(
        analysis.get("horizon_segment_windows"),
        "protocol analysis.horizon_segment_windows",
    )
    windows: dict[str, tuple[int, int]] = {}
    covered: list[int] = []
    for segment, raw_bounds in raw_windows.items():
        bounds = _array(raw_bounds, f"segment window {segment}")
        if len(bounds) != 2:
            _fail(f"segment window {segment} must contain inclusive start/end")
        start = _integer(bounds[0], f"segment window {segment} start")
        end = _integer(bounds[1], f"segment window {segment} end")
        if start > end:
            _fail(f"segment window {segment} has descending bounds")
        windows[segment] = (start, end)
        covered.extend(range(start, end + 1))
    _require_equal(sorted(covered), list(range(12, 24)), "horizon segment partition")
    public_path = "source/public/public_plan.json"
    sealed_path = "source/sealed/evaluator_bundle.json"
    public_plan = reader.load(public_path, verify_artifact_id=False)
    sealed_bundle = reader.load(sealed_path, verify_artifact_id=False)
    _require_equal(
        reader.entries[public_path]["sha256"],
        _EXPECTED_SOURCE_PUBLIC_RAW_SHA256,
        "source public plan raw SHA-256",
    )
    _require_equal(
        reader.entries[sealed_path]["sha256"],
        _EXPECTED_SOURCE_SEALED_RAW_SHA256,
        "source sealed bundle raw SHA-256",
    )
    _require_equal(
        public_plan.get("protocol_sha256"),
        _EXPECTED_SOURCE_PROTOCOL_ID,
        "source public plan protocol identity",
    )
    _require_equal(
        sealed_bundle.get("protocol_sha256"),
        _EXPECTED_SOURCE_PROTOCOL_ID,
        "source sealed bundle protocol identity",
    )
    _require_equal(
        _sha256_bytes(_canonical_json_bytes(public_plan)),
        _EXPECTED_SOURCE_PUBLIC_PLAN_ID,
        "source public plan canonical identity",
    )
    unsigned_sealed = dict(sealed_bundle)
    observed_sealed_identity = _sha256(
        unsigned_sealed.pop("sealed_bundle_sha256", None),
        "source sealed bundle self identity",
    )
    _require_equal(
        observed_sealed_identity,
        _EXPECTED_SOURCE_SEALED_BUNDLE_ID,
        "source sealed bundle declared identity",
    )
    _require_equal(
        _sha256_bytes(_canonical_json_bytes(unsigned_sealed)),
        _EXPECTED_SOURCE_SEALED_BUNDLE_ID,
        "source sealed bundle canonical identity",
    )
    _require_equal(
        sealed_bundle.get("evaluation_or_judge_feedback_to_learning"),
        False,
        "source sealed evaluator feedback boundary",
    )
    _require_equal(report.get("source_protocol_id"), _EXPECTED_SOURCE_PROTOCOL_ID, "report source ID")
    _require_equal(
        report.get("public_plan_sha256"),
        _EXPECTED_SOURCE_PUBLIC_PLAN_ID,
        "report source public plan ID",
    )
    _require_equal(
        report.get("sealed_bundle_sha256"),
        _EXPECTED_SOURCE_SEALED_BUNDLE_ID,
        "report source sealed bundle ID",
    )
    source_provenance = {
        "source_protocol_id": _EXPECTED_SOURCE_PROTOCOL_ID,
        "public_plan_id": _EXPECTED_SOURCE_PUBLIC_PLAN_ID,
        "sealed_bundle_id": _EXPECTED_SOURCE_SEALED_BUNDLE_ID,
        "public_plan": reader.reference(public_path, public_plan),
        "sealed_bundle": reader.reference(sealed_path, sealed_bundle),
        "execution_source_bundle": reader.reference(
            execution_source_bundle_path,
            execution_source_bundle,
        ),
    }
    runtime_contract = {
        "condition_reader_artifact_id": reader_artifact_id,
        "execution_source_bundle_artifact_id": report_source_bundle_id,
        "local_execution_source_tree_sha256": source_tree_sha256,
        "worker_script_raw_sha256": worker_raw_sha256,
    }
    return protocol, report, windows, source_provenance, runtime_contract


def _source_lineage_summary(
    pre: Mapping[str, object],
    post: Mapping[str, object],
    *,
    expected_source_bundle_artifact_id: str,
    expected_source_tree_sha256: str,
    expected_worker_script_raw_sha256: str,
) -> dict[str, object]:
    pre_lineage = _object(pre.get("execution_source_lineage"), "preaction execution_source_lineage")
    post_lineage = _object(post.get("execution_source_lineage"), "postaction execution_source_lineage")
    pre_lineage_id = _require_artifact_identity(pre_lineage, label="preaction execution_source_lineage")
    post_lineage_id = _require_artifact_identity(post_lineage, label="postaction execution_source_lineage")
    _require_equal(post_lineage_id, pre_lineage_id, "execution lineage artifact_id")
    _require_equal(
        pre_lineage.get("schema_version"),
        "relationship-product-worker-source-lineage.v1",
        "execution lineage schema_version",
    )
    _require_equal(
        pre_lineage.get("execution_source_bundle_artifact_id"),
        expected_source_bundle_artifact_id,
        "execution lineage source bundle artifact_id",
    )
    _require_equal(
        pre_lineage.get("local_execution_source_tree_sha256"),
        expected_source_tree_sha256,
        "execution lineage source tree SHA-256",
    )
    _require_equal(
        pre_lineage.get("worker_script_raw_sha256"),
        expected_worker_script_raw_sha256,
        "execution lineage worker script SHA-256",
    )
    _require_equal(
        pre_lineage.get("worker_script_repository_path"),
        "scripts/run_relationship_lab_product_horizon.py",
        "execution lineage worker script path",
    )
    fields = (
        "artifact_id",
        "execution_source_bundle_artifact_id",
        "local_execution_source_tree_sha256",
        "schema_version",
        "volvence_zero_namespace_search_locations",
        "worker_script_raw_sha256",
        "worker_script_repository_path",
    )
    for field in fields:
        _require_equal(post_lineage.get(field), pre_lineage.get(field), f"execution lineage {field}")
    return {field: deepcopy(pre_lineage.get(field)) for field in fields}


def _validate_forecast_payload(
    forecast: Mapping[str, object],
    *,
    arm_id: str,
    expected_reader_artifact_id: str,
) -> tuple[str, str, list[str], list[str]]:
    forecast_id = _text(forecast.get("forecast_id"), "preaction forecast forecast_id")
    _text(forecast.get("decision_id"), "preaction forecast decision_id")
    _text(forecast.get("session_scope"), "preaction forecast session_scope")
    _text(forecast.get("interlocutor_id"), "preaction forecast interlocutor_id")
    issued_turn = _integer(forecast.get("issued_turn"), "preaction forecast issued_turn")
    if issued_turn < 0:
        _fail("preaction forecast issued_turn must be non-negative")
    _bounded_number(
        forecast.get("confidence"),
        "preaction forecast confidence",
        minimum=0.0,
        maximum=1.0,
    )
    evidence = _text_array(forecast.get("evidence"), "preaction forecast evidence")
    source_record_ids = _text_array(
        forecast.get("source_record_ids"),
        "preaction forecast source_record_ids",
    )

    predictions = _array(
        forecast.get("candidate_predictions"),
        "preaction forecast candidate_predictions",
    )
    _require_equal(len(predictions), len(_EXPECTED_ACTION_IDS), "forecast candidate action count")
    candidate_action_ids: set[str] = set()
    for candidate_index, item in enumerate(predictions):
        candidate = _object(item, f"forecast candidate_predictions[{candidate_index}]")
        action_id = _text(
            candidate.get("action_id"),
            f"forecast candidate_predictions[{candidate_index}].action_id",
        )
        if action_id in candidate_action_ids:
            _fail(f"forecast repeats candidate action_id {action_id!r}")
        candidate_action_ids.add(action_id)
        outcomes = _array(
            candidate.get("outcomes"),
            f"forecast candidate_predictions[{candidate_index}].outcomes",
        )
        outcome_probabilities: dict[str, float] = {}
        for outcome_index, outcome_item in enumerate(outcomes):
            outcome = _object(
                outcome_item,
                f"forecast candidate_predictions[{candidate_index}].outcomes[{outcome_index}]",
            )
            outcome_id = _text(
                outcome.get("outcome_id"),
                f"forecast candidate_predictions[{candidate_index}].outcome_id",
            )
            if outcome_id in outcome_probabilities:
                _fail(f"forecast candidate {action_id!r} repeats outcome {outcome_id!r}")
            outcome_probabilities[outcome_id] = _bounded_number(
                outcome.get("probability"),
                f"forecast candidate {action_id!r} outcome {outcome_id!r} probability",
                minimum=0.0,
                maximum=1.0,
            )
        _require_equal(
            set(outcome_probabilities),
            set(_EXPECTED_OUTCOME_IDS),
            f"forecast candidate {action_id!r} outcome set",
        )
        if not math.isclose(
            sum(outcome_probabilities.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            _fail(f"forecast candidate {action_id!r} probabilities must sum to one")
    _require_equal(candidate_action_ids, set(_EXPECTED_ACTION_IDS), "forecast candidate action set")
    recommended_action_id = _text(
        forecast.get("recommended_action_id"),
        "preaction forecast recommended_action_id",
    )
    if recommended_action_id not in candidate_action_ids:
        _fail("preaction forecast recommends an action absent from candidate_predictions")

    readout_value = forecast.get("condition_readout")
    if arm_id == "readable_unnamed_legacy":
        _require_equal(readout_value, None, "unnamed reader condition_readout")
    else:
        readout = _object(readout_value, "named reader condition_readout")
        _require_equal(
            readout.get("reader_artifact_id"),
            expected_reader_artifact_id,
            "named reader artifact_id",
        )
        observed_label = _text(readout.get("condition_label"), "named reader condition_label")
        if observed_label not in _CONDITION_TO_READER_LABEL.values():
            _fail(f"named reader emitted an unregistered condition label: {observed_label!r}")
        _sha256(readout.get("source_observation_sha256"), "named reader source observation SHA-256")
        _bounded_number(
            readout.get("confidence"),
            "named reader confidence",
            minimum=0.0,
            maximum=1.0,
        )
        normalized_margin = _bounded_number(
            readout.get("normalized_margin"),
            "named reader normalized margin",
            minimum=0.0,
            maximum=1.0,
        )
        candidate_scores = _array(readout.get("candidate_scores"), "named reader candidate_scores")
        _require_equal(
            len(candidate_scores),
            len(_CONDITION_TO_READER_LABEL),
            "named reader candidate score count",
        )
        scores: dict[str, float] = {}
        for score_index, item in enumerate(candidate_scores):
            score = _object(item, f"named reader candidate_scores[{score_index}]")
            label = _text(
                score.get("label"),
                f"named reader candidate_scores[{score_index}].label",
            )
            if label in scores:
                _fail(f"named reader repeats candidate label {label!r}")
            scores[label] = _bounded_number(
                score.get("score"),
                f"named reader candidate score {label!r}",
                minimum=-1.0,
                maximum=1.0,
            )
        _require_equal(
            set(scores),
            set(_CONDITION_TO_READER_LABEL.values()),
            "named reader candidate label set",
        )
        ranked_scores = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        _require_equal(observed_label, ranked_scores[0][0], "named reader top-1 label")
        expected_margin = max(0.0, min(1.0, (ranked_scores[0][1] - ranked_scores[1][1]) / 2.0))
        if not math.isclose(normalized_margin, expected_margin, rel_tol=0.0, abs_tol=1e-15):
            _fail("named reader normalized_margin does not match candidate scores")
    return forecast_id, recommended_action_id, source_record_ids, evidence


def _mechanism_payload(
    pre: Mapping[str, object],
    post: Mapping[str, object],
    *,
    arm_id: str,
    expected_reader_artifact_id: str,
    expected_source_bundle_artifact_id: str,
    expected_source_tree_sha256: str,
    expected_worker_script_raw_sha256: str,
) -> dict[str, object]:
    frozen_forecast = _object(pre.get("frozen_forecast"), "preaction frozen_forecast")
    _require_artifact_identity(frozen_forecast, label="preaction frozen_forecast")
    _require_equal(
        frozen_forecast.get("schema_version"),
        _EXPECTED_FORECAST_SNAPSHOT_SCHEMA_VERSION,
        "preaction frozen_forecast schema_version",
    )
    forecast = _object(frozen_forecast.get("forecast"), "preaction frozen_forecast.forecast")
    forecast_sha256 = _sha256(pre.get("forecast_sha256"), "preaction forecast_sha256")
    _require_equal(
        forecast_sha256,
        _sha256_bytes(_canonical_json_bytes(forecast)),
        "preaction forecast canonical SHA-256",
    )
    forecast_id, recommended_action_id, source_record_ids, forecast_evidence = _validate_forecast_payload(
        forecast,
        arm_id=arm_id,
        expected_reader_artifact_id=expected_reader_artifact_id,
    )
    gate_decision = _object(pre.get("gate_decision"), "preaction gate_decision")
    _require_equal(gate_decision.get("artifact_id"), _EXPECTED_GATE_ARTIFACT_ID, "gate artifact_id")
    _require_equal(
        gate_decision.get("artifact_version"),
        _EXPECTED_GATE_ARTIFACT_VERSION,
        "gate artifact_version",
    )
    _require_equal(gate_decision.get("mode"), "learned", "gate mode")
    _require_equal(gate_decision.get("evaluator_only"), False, "gate evaluator_only")
    _require_equal(gate_decision.get("forecast_id"), forecast_id, "gate forecast_id")
    _require_equal(
        gate_decision.get("recommended_action_id"),
        recommended_action_id,
        "gate recommended_action_id",
    )
    _require_equal(
        _text_array(gate_decision.get("evidence_refs"), "gate evidence_refs"),
        [*source_record_ids, *forecast_evidence],
        "gate evidence_refs",
    )
    features = _array(gate_decision.get("features"), "gate features")
    _require_equal(len(features), 5, "gate feature count")
    for feature_index, value in enumerate(features):
        _number(value, f"gate features[{feature_index}]")
    _text_array(gate_decision.get("rationale_codes"), "gate rationale_codes")
    _bounded_number(
        gate_decision.get("steer_probability"),
        "gate steer_probability",
        minimum=0.0,
        maximum=1.0,
    )
    gate_action = _text(gate_decision.get("gate_action"), "gate gate_action")
    if gate_action not in {"noop", "steer"}:
        _fail(f"gate gate_action is unsupported: {gate_action!r}")
    selected_action_id = _text(
        gate_decision.get("selected_action_id"),
        "gate selected_action_id",
    )
    expected_selected_action_id = recommended_action_id if gate_action == "steer" else "neutral_noop"
    _require_equal(selected_action_id, expected_selected_action_id, "gate bounded action selection")
    gate_update_before = _integer(pre.get("gate_update_count_before"), "gate update count before")
    if gate_update_before < 0:
        _fail("gate update count before must be non-negative")
    _require_equal(gate_decision.get("update_count"), gate_update_before, "gate decision update_count")
    _require_equal(_boolean(pre.get("owner_loaded"), "owner_loaded"), True, "owner_loaded")
    _sha256(pre.get("pre_owner_snapshot_sha256"), "pre owner snapshot SHA-256")
    _sha256(post.get("post_owner_snapshot_sha256"), "post owner snapshot SHA-256")
    _require_equal(
        _boolean(post.get("credit_applied_to_gate"), "credit_applied_to_gate"),
        True,
        "audited arm credit_applied_to_gate",
    )
    _require_equal(
        post.get("evaluator_or_judge_feedback_received"),
        False,
        "audited arm evaluator/judge feedback",
    )
    _text(post.get("credit_record_id"), "credit_record_id")
    credit_value_hex = _text(post.get("credit_value_hex"), "credit_value_hex")
    try:
        credit_value = float.fromhex(credit_value_hex)
    except ValueError as exc:
        raise AuditContractError("credit_value_hex must encode a finite float") from exc
    if not math.isfinite(credit_value):
        _fail("credit_value_hex must encode a finite float")
    _text(post.get("settlement_id"), "settlement_id")
    _sha256(post.get("settlement_payload_sha256"), "settlement payload SHA-256")
    _sha256(post.get("social_prediction_error_snapshot_sha256"), "social PE snapshot SHA-256")
    return {
        "forecast": {
            "artifact_id": frozen_forecast["artifact_id"],
            "schema_version": frozen_forecast.get("schema_version"),
            "forecast_sha256": forecast_sha256,
            "payload": deepcopy(forecast),
        },
        "gate": {
            "decision": deepcopy(gate_decision),
            "update_count_before": gate_update_before,
            "update_count_after": post.get("gate_update_count_after"),
        },
        "owner": {
            "loaded": pre.get("owner_loaded"),
            "pre_snapshot_sha256": pre.get("pre_owner_snapshot_sha256"),
            "post_snapshot_sha256": post.get("post_owner_snapshot_sha256"),
            "receipt_scope": "hash_only",
            "semantic_delta_materialized_in_divergence_rows": False,
            "causal_attribution_authorized": False,
        },
        "credit": {
            "applied_to_gate": post.get("credit_applied_to_gate"),
            "credit_record_id": post.get("credit_record_id"),
            "credit_value_hex": post.get("credit_value_hex"),
            "evaluator_or_judge_feedback_received": post.get("evaluator_or_judge_feedback_received"),
            "settlement_id": post.get("settlement_id"),
            "settlement_payload_sha256": post.get("settlement_payload_sha256"),
            "social_prediction_error_snapshot_sha256": post.get("social_prediction_error_snapshot_sha256"),
        },
        "execution_source_lineage": _source_lineage_summary(
            pre,
            post,
            expected_source_bundle_artifact_id=expected_source_bundle_artifact_id,
            expected_source_tree_sha256=expected_source_tree_sha256,
            expected_worker_script_raw_sha256=expected_worker_script_raw_sha256,
        ),
    }


def _validate_document_link(
    *,
    record: Mapping[str, object],
    field_prefix: str,
    path: str,
    payload: Mapping[str, object],
    reader: _AttemptReader,
) -> dict[str, object]:
    _require_equal(record.get(f"{field_prefix}_path"), path, f"chain {field_prefix}_path")
    reference = reader.reference(path, payload)
    _require_equal(
        record.get(f"{field_prefix}_sha256"),
        reference["raw_sha256"],
        f"chain {field_prefix}_sha256",
    )
    artifact_field = f"{field_prefix}_artifact_id"
    if artifact_field in record:
        _require_equal(
            record.get(artifact_field),
            reference.get("artifact_id"),
            f"chain {artifact_field}",
        )
    return reference


def _load_arm_records(
    reader: _AttemptReader,
    *,
    subject_scope: str,
    arm_id: str,
    runtime_contract: Mapping[str, str],
) -> tuple[dict[str, object], dict[tuple[str, str], dict[str, object]]]:
    arm_prefix = f"chains/{subject_scope}/{arm_id}"
    chain_path = f"{arm_prefix}/chain.json"
    chain = reader.load(chain_path)
    _require_equal(
        chain.get("schema_version"),
        "relationship-product-typed-chain.v1",
        f"{chain_path} schema_version",
    )
    _require_equal(chain.get("arm_id"), arm_id, f"{chain_path} arm_id")
    _require_equal(chain.get("subject_scope"), subject_scope, f"{chain_path} subject_scope")
    world_clone_id = _sha256(chain.get("world_clone_id"), f"{chain_path} world_clone_id")
    decisions = _array(chain.get("decisions"), f"{chain_path} decisions")
    _require_equal(len(decisions), 24, f"{chain_path} decision count")
    result: dict[tuple[str, str], dict[str, object]] = {}
    for expected_index, item in enumerate(decisions):
        record = _object(item, f"{chain_path} decisions[{expected_index}]")
        decision_index = _integer(
            record.get("decision_index"),
            f"{chain_path} decisions[{expected_index}].decision_index",
        )
        _require_equal(decision_index, expected_index, f"{chain_path} decision ordering")
        decision_id = _text(record.get("decision_id"), f"{chain_path} decision_id")
        join_key = (subject_scope, decision_id)
        if join_key in result:
            _fail(f"duplicate (subject_scope, decision_id) in {chain_path}: {join_key}")

        expected_stem = f"{arm_prefix}/receipts/decision-{decision_index:02d}"
        pre_path = f"{expected_stem}.preaction.json"
        post_path = f"{expected_stem}.postaction.json"
        sealed_path = f"{arm_prefix}/sealed/decision-{decision_index:02d}.json"
        request_path = f"{arm_prefix}/requests/decision-{decision_index:02d}.json"
        pre = reader.load(pre_path)
        post = reader.load(post_path)
        sealed = reader.load(sealed_path)
        request = reader.load(request_path)
        for payload, expected_schema, label in (
            (pre, "relationship-product-preaction-receipt.v2", pre_path),
            (post, "relationship-product-postaction-receipt.v2", post_path),
            (sealed, "relationship-product-sealed-decision.v1", sealed_path),
            (request, "relationship-product-worker-request.v1", request_path),
        ):
            _require_equal(payload.get("schema_version"), expected_schema, f"{label} schema_version")
        _require_equal(request.get("protocol_id"), _EXPECTED_PROTOCOL_ID, f"{request_path} protocol_id")
        _require_equal(request.get("arm_id"), arm_id, f"{request_path} arm_id")
        _require_equal(request.get("subject_scope"), subject_scope, f"{request_path} subject_scope")
        _require_equal(request.get("world_clone_id"), world_clone_id, f"{request_path} world_clone_id")
        _require_equal(request.get("operation"), "decision_handshake", f"{request_path} operation")
        _require_equal(request.get("gate_mode"), "learned", f"{request_path} gate_mode")
        _require_equal(
            request.get("apply_credit_to_gate"),
            True,
            f"{request_path} apply_credit_to_gate",
        )
        expected_named_reader = (
            "legacy_unnamed_semantic_similarity"
            if arm_id == "readable_unnamed_legacy"
            else "prototype_named_condition_readout"
        )
        _require_equal(
            request.get("named_reader"),
            expected_named_reader,
            f"{request_path} named_reader",
        )
        _require_equal(
            request.get("execution_source_bundle_artifact_id"),
            runtime_contract["execution_source_bundle_artifact_id"],
            f"{request_path} execution source bundle artifact_id",
        )
        _require_equal(
            request.get("local_execution_source_tree_sha256"),
            runtime_contract["local_execution_source_tree_sha256"],
            f"{request_path} local execution source tree SHA-256",
        )
        source_documents = {
            "chain": reader.reference(chain_path, chain),
            "request": _validate_document_link(
                record=record,
                field_prefix="request",
                path=request_path,
                payload=request,
                reader=reader,
            ),
            "preaction": _validate_document_link(
                record=record,
                field_prefix="preaction_receipt",
                path=pre_path,
                payload=pre,
                reader=reader,
            ),
            "postaction": _validate_document_link(
                record=record,
                field_prefix="postaction_receipt",
                path=post_path,
                payload=post,
                reader=reader,
            ),
            "sealed": _validate_document_link(
                record=record,
                field_prefix="sealed_record",
                path=sealed_path,
                payload=sealed,
                reader=reader,
            ),
        }

        forecast_wrapper = _object(pre.get("frozen_forecast"), f"{pre_path} frozen_forecast")
        forecast = _object(forecast_wrapper.get("forecast"), f"{pre_path} forecast")
        gate = _object(pre.get("gate_decision"), f"{pre_path} gate_decision")
        _require_equal(forecast.get("decision_id"), decision_id, f"{pre_path} forecast decision_id")
        _require_equal(forecast.get("session_scope"), subject_scope, f"{pre_path} session_scope")
        _require_equal(gate.get("decision_id"), decision_id, f"{pre_path} gate decision_id")
        _require_equal(pre.get("forecast_id"), forecast.get("forecast_id"), f"{pre_path} forecast_id")
        _require_equal(post.get("forecast_id"), forecast.get("forecast_id"), f"{post_path} forecast_id")
        _require_equal(
            post.get("preaction_artifact_id"),
            pre.get("artifact_id"),
            f"{post_path} preaction_artifact_id",
        )
        _require_equal(post.get("request_artifact_id"), request.get("artifact_id"), f"{post_path} request")
        _require_equal(pre.get("request_artifact_id"), request.get("artifact_id"), f"{pre_path} request")
        selected_action = _text(record.get("selected_action_id"), f"{chain_path} selected action")
        typed_outcome = _text(record.get("typed_outcome_id"), f"{chain_path} typed outcome")
        _require_equal(pre.get("selected_action_id"), selected_action, f"{pre_path} selected action")
        _require_equal(gate.get("selected_action_id"), selected_action, f"{pre_path} gate selected action")
        _require_equal(sealed.get("selected_action_id"), selected_action, f"{sealed_path} selected action")
        _require_equal(post.get("typed_outcome_id"), typed_outcome, f"{post_path} typed outcome")
        _require_equal(sealed.get("typed_outcome_id"), typed_outcome, f"{sealed_path} typed outcome")
        _require_equal(sealed.get("decision_id"), decision_id, f"{sealed_path} decision_id")
        _require_equal(sealed.get("decision_index"), decision_index, f"{sealed_path} decision_index")
        _require_equal(sealed.get("world_clone_id"), world_clone_id, f"{sealed_path} world_clone_id")
        _require_equal(record.get("world_clone_id"), world_clone_id, f"{chain_path} world_clone_id")
        _require_equal(
            pre.get("recommended_action_id"),
            forecast.get("recommended_action_id"),
            f"{pre_path} recommended action",
        )
        _require_equal(
            gate.get("recommended_action_id"),
            forecast.get("recommended_action_id"),
            f"{pre_path} gate recommended action",
        )
        positive = typed_outcome in _POSITIVE_OUTCOME_IDS
        safety_negative = typed_outcome in _SAFETY_NEGATIVE_OUTCOME_IDS
        if positive == safety_negative:
            _fail(f"{chain_path} outcome is not exactly one of positive/safety-negative: {typed_outcome}")
        _require_equal(record.get("positive_outcome"), positive, f"{chain_path} positive outcome")
        credit_applied = _boolean(post.get("credit_applied_to_gate"), f"{post_path} credit applied")
        update_before = _integer(pre.get("gate_update_count_before"), f"{pre_path} gate update before")
        update_after = _integer(post.get("gate_update_count_after"), f"{post_path} gate update after")
        _require_equal(update_after - update_before, int(credit_applied), f"{post_path} gate update increment")
        _require_equal(
            post.get("evaluator_or_judge_feedback_received"),
            False,
            f"{post_path} evaluator leakage",
        )
        _sha256(pre.get("pre_owner_snapshot_sha256"), f"{pre_path} pre owner SHA")
        _sha256(post.get("post_owner_snapshot_sha256"), f"{post_path} post owner SHA")
        _sha256(
            post.get("social_prediction_error_snapshot_sha256"),
            f"{post_path} PE snapshot SHA",
        )

        condition_id = _text(sealed.get("condition_id"), f"{sealed_path} condition_id")
        if condition_id not in _CONDITION_TO_READER_LABEL:
            _fail(f"{sealed_path} has unknown reader truth condition: {condition_id}")
        truth = {
            field: deepcopy(sealed.get(field))
            for field in (
                "condition_id",
                "decision_id",
                "decision_index",
                "domain_id",
                "environment_seed",
                "phase_id",
                "policy_id",
                "preferred_action_id",
                "public_correction_target_session_id",
                "scene_id",
                "session_id",
                "stage_id",
                "subject_id",
                "subject_seed",
                "world_clone_id",
            )
        }
        result[join_key] = {
            "arm_id": arm_id,
            "subject_scope": subject_scope,
            "world_clone_id": world_clone_id,
            "decision_id": decision_id,
            "decision_index": decision_index,
            "truth": truth,
            "selected_action_id": selected_action,
            "typed_outcome_id": typed_outcome,
            "positive_outcome": positive,
            "safety_negative_outcome": safety_negative,
            "preferred_action_match": record.get("preferred_action_match"),
            "mechanism": _mechanism_payload(
                pre,
                post,
                arm_id=arm_id,
                expected_reader_artifact_id=runtime_contract["condition_reader_artifact_id"],
                expected_source_bundle_artifact_id=(runtime_contract["execution_source_bundle_artifact_id"]),
                expected_source_tree_sha256=runtime_contract["local_execution_source_tree_sha256"],
                expected_worker_script_raw_sha256=runtime_contract["worker_script_raw_sha256"],
            ),
            "source_documents": source_documents,
        }
    return chain, result


def _reader_diagnostic(record: Mapping[str, object]) -> dict[str, object]:
    truth = _object(record.get("truth"), "record truth")
    condition_id = _text(truth.get("condition_id"), "truth condition_id")
    expected = _CONDITION_TO_READER_LABEL[condition_id]
    mechanism = _object(record.get("mechanism"), "record mechanism")
    forecast = _object(mechanism.get("forecast"), "record mechanism forecast")
    forecast_payload = _object(forecast.get("payload"), "record forecast payload")
    readout_value = forecast_payload.get("condition_readout")
    if readout_value is None:
        return {
            "condition_id": condition_id,
            "expected_reader_label": expected,
            "named_readout_present": False,
            "observed_reader_label": None,
            "reader_truth_match": None,
        }
    readout = _object(readout_value, "record condition_readout")
    observed = _text(readout.get("condition_label"), "condition_readout condition_label")
    return {
        "condition_id": condition_id,
        "expected_reader_label": expected,
        "named_readout_present": True,
        "observed_reader_label": observed,
        "reader_truth_match": observed == expected,
        "confidence": readout.get("confidence"),
        "normalized_margin": readout.get("normalized_margin"),
        "candidate_scores": deepcopy(readout.get("candidate_scores")),
        "reader_artifact_id": readout.get("reader_artifact_id"),
        "source_observation_sha256": readout.get("source_observation_sha256"),
    }


def _segment_for_index(index: int, windows: Mapping[str, tuple[int, int]]) -> str | None:
    matching = [name for name, (start, end) in windows.items() if start <= index <= end]
    if len(matching) > 1:
        _fail(f"decision index {index} belongs to multiple horizon segments")
    return matching[0] if matching else None


def _validate_cross_arm_truth(
    full: Mapping[str, object],
    comparator: Mapping[str, object],
    *,
    comparator_id: str,
) -> None:
    _require_equal(full.get("subject_scope"), comparator.get("subject_scope"), "join subject_scope")
    _require_equal(full.get("decision_id"), comparator.get("decision_id"), "join decision_id")
    _require_equal(full.get("decision_index"), comparator.get("decision_index"), "join decision_index")
    _require_equal(full.get("world_clone_id"), comparator.get("world_clone_id"), "join world_clone_id")
    _require_equal(full.get("truth"), comparator.get("truth"), f"{comparator_id} sealed exogenous truth")


def _pair_comparison(
    full: Mapping[str, object],
    comparator: Mapping[str, object],
) -> dict[str, object]:
    full_positive = _boolean(full.get("positive_outcome"), "full positive_outcome")
    comparator_positive = _boolean(
        comparator.get("positive_outcome"),
        "comparator positive_outcome",
    )
    full_safety = _boolean(full.get("safety_negative_outcome"), "full safety negative")
    comparator_safety = _boolean(
        comparator.get("safety_negative_outcome"),
        "comparator safety negative",
    )
    _require_equal(full_safety, not full_positive, "full safety complement")
    _require_equal(comparator_safety, not comparator_positive, "comparator safety complement")
    return {
        "action_discordant": full.get("selected_action_id") != comparator.get("selected_action_id"),
        "typed_outcome_id_discordant": full.get("typed_outcome_id") != comparator.get("typed_outcome_id"),
        "positive_outcome_discordant": full_positive != comparator_positive,
        "full_only_positive": full_positive and not comparator_positive,
        "comparator_only_positive": comparator_positive and not full_positive,
        "positive_outcome_net_numerator": int(full_positive) - int(comparator_positive),
        "full_safety_negative_increase_numerator": int(full_safety) - int(comparator_safety),
        "safety_is_exact_complement_of_positive": True,
    }


def _aggregate_pairs(
    pairs: Sequence[Mapping[str, object]],
    *,
    include: Callable[[Mapping[str, object]], bool],
) -> dict[str, object]:
    selected = [pair for pair in pairs if include(pair)]
    if not selected:
        _fail("aggregate window contains zero matched decisions")
    action_discordant = 0
    typed_outcome_discordant = 0
    outcome_discordant = 0
    full_only = 0
    comparator_only = 0
    full_positive = 0
    comparator_positive = 0
    full_safety = 0
    comparator_safety = 0
    by_world: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for pair in selected:
        comparison = _object(pair.get("comparison"), "pair comparison")
        action_discordant += int(_boolean(comparison.get("action_discordant"), "action discordance"))
        typed_outcome_discordant += int(
            _boolean(comparison.get("typed_outcome_id_discordant"), "typed outcome discordance")
        )
        outcome_discordant += int(
            _boolean(comparison.get("positive_outcome_discordant"), "positive outcome discordance")
        )
        full_only += int(_boolean(comparison.get("full_only_positive"), "full-only positive"))
        comparator_only += int(_boolean(comparison.get("comparator_only_positive"), "comparator-only positive"))
        full = _object(pair.get("full"), "pair full")
        comparator = _object(pair.get("comparator_arm"), "pair comparator arm")
        full_positive += int(_boolean(full.get("positive_outcome"), "full positive"))
        comparator_positive += int(_boolean(comparator.get("positive_outcome"), "comparator positive"))
        full_safety += int(_boolean(full.get("safety_negative_outcome"), "full safety"))
        comparator_safety += int(_boolean(comparator.get("safety_negative_outcome"), "comparator safety"))
        world = _sha256(pair.get("world_clone_id"), "pair world_clone_id")
        by_world[world].append(pair)
    denominator = len(selected)
    net = full_only - comparator_only
    _require_equal(net, full_positive - comparator_positive, "aggregate positive net identity")
    safety_net = full_safety - comparator_safety
    _require_equal(safety_net, -net, "aggregate safety complement identity")
    world_rows: list[dict[str, object]] = []
    directions: Counter[str] = Counter()
    for world in sorted(by_world):
        world_pairs = by_world[world]
        world_full = sum(int(_object(pair.get("full"), "world full")["positive_outcome"]) for pair in world_pairs)
        world_comparator = sum(
            int(_object(pair.get("comparator_arm"), "world comparator")["positive_outcome"]) for pair in world_pairs
        )
        world_net = world_full - world_comparator
        direction = "positive" if world_net > 0 else "negative" if world_net < 0 else "tie"
        directions[direction] += 1
        subject_scopes = sorted({_text(pair.get("subject_scope"), "pair subject_scope") for pair in world_pairs})
        _require_equal(len(subject_scopes), 1, f"world {world} subject scope cardinality")
        world_rows.append(
            {
                "world_clone_id": world,
                "subject_scope": subject_scopes[0],
                "matched_decision_count": len(world_pairs),
                "full_positive_outcome_count": world_full,
                "comparator_positive_outcome_count": world_comparator,
                "net_numerator": world_net,
                "direction": direction,
            }
        )
    return {
        "matched_world_count": len(by_world),
        "matched_decision_count": denominator,
        "action_discordant_count": action_discordant,
        "typed_outcome_id_discordant_count": typed_outcome_discordant,
        "positive_outcome_discordant_count": outcome_discordant,
        "full_only_positive_count": full_only,
        "comparator_only_positive_count": comparator_only,
        "positive_outcome_net": {"numerator": net, "denominator": denominator},
        "full_safety_negative_increase": {
            "numerator": safety_net,
            "denominator": denominator,
        },
        "world_direction_counts": {
            "positive": directions["positive"],
            "tie": directions["tie"],
            "negative": directions["negative"],
        },
        "world_directions": world_rows,
    }


def _record_for_row(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "arm_join_key": {
            "arm_id": record.get("arm_id"),
            "subject_scope": record.get("subject_scope"),
            "decision_id": record.get("decision_id"),
        },
        "selected_action_id": record.get("selected_action_id"),
        "typed_outcome_id": record.get("typed_outcome_id"),
        "positive_outcome": record.get("positive_outcome"),
        "safety_negative_outcome": record.get("safety_negative_outcome"),
        "preferred_action_match": record.get("preferred_action_match"),
        "reader_truth": _reader_diagnostic(record),
        "mechanism": deepcopy(record.get("mechanism")),
        "source_documents": deepcopy(record.get("source_documents")),
    }


def _build_all_pairs(
    records_by_arm: Mapping[str, Mapping[tuple[str, str], Mapping[str, object]]],
    windows: Mapping[str, tuple[int, int]],
) -> dict[str, list[dict[str, object]]]:
    full_records = records_by_arm[_FULL_ARM]
    all_pairs: dict[str, list[dict[str, object]]] = {}
    for comparator_id in _COMPARATORS:
        comparator_records = records_by_arm[comparator_id]
        _require_equal(
            sorted(comparator_records),
            sorted(full_records),
            f"{comparator_id} matched join-key set",
        )
        pairs: list[dict[str, object]] = []
        for join_key in sorted(
            full_records,
            key=lambda key: (key[0], _integer(full_records[key]["decision_index"], "decision index")),
        ):
            full = full_records[join_key]
            comparator = comparator_records[join_key]
            _validate_cross_arm_truth(full, comparator, comparator_id=comparator_id)
            decision_index = _integer(full.get("decision_index"), "full decision_index")
            pairs.append(
                {
                    "comparator": comparator_id,
                    "subject_scope": full.get("subject_scope"),
                    "world_clone_id": full.get("world_clone_id"),
                    "decision_id": full.get("decision_id"),
                    "decision_index": decision_index,
                    "primary_window": 12 <= decision_index <= 23,
                    "horizon_segment": _segment_for_index(decision_index, windows),
                    "truth": deepcopy(full.get("truth")),
                    "full": _record_for_row(full),
                    "comparator_arm": _record_for_row(comparator),
                    "comparison": _pair_comparison(full, comparator),
                }
            )
        _require_equal(len(pairs), 192, f"{comparator_id} matched decision denominator")
        all_pairs[comparator_id] = pairs
    return all_pairs


def _frozen_report_comparison(report: Mapping[str, object], comparator: str) -> Mapping[str, object]:
    rows = _array(report.get("paired_comparisons"), "report paired_comparisons")
    matches = [
        _object(row, "report paired comparison")
        for row in rows
        if _object(row, "report paired comparison").get("comparator") == comparator
    ]
    if len(matches) != 1:
        _fail(f"report must contain exactly one paired comparison for {comparator}")
    return matches[0]


def _aggregate_rate(
    aggregate: Mapping[str, object],
    *,
    field: str,
    label: str,
) -> float:
    fraction = _object(aggregate.get(field), label)
    numerator = _integer(fraction.get("numerator"), f"{label} numerator")
    denominator = _integer(fraction.get("denominator"), f"{label} denominator")
    if denominator <= 0:
        _fail(f"{label} denominator must be positive")
    return numerator / denominator


def _require_close(observed: object, expected: float, *, label: str) -> None:
    numeric = _number(observed, label)
    if not math.isclose(numeric, expected, rel_tol=0.0, abs_tol=1e-15):
        _fail(f"{label} mismatch: observed={numeric}, recomputed={expected}")


def _unique_comparator_row(
    report: Mapping[str, object],
    *,
    field: str,
    comparator: str,
) -> Mapping[str, object]:
    rows = _array(report.get(field), f"report {field}")
    matches = [
        _object(row, f"report {field} row")
        for row in rows
        if _object(row, f"report {field} row").get("comparator") == comparator
    ]
    if len(matches) != 1:
        _fail(f"report {field} must contain exactly one row for {comparator}")
    return matches[0]


def _validate_report_aggregates(
    report: Mapping[str, object],
    comparator: str,
    all_window: Mapping[str, object],
    primary_window: Mapping[str, object],
    segments: Mapping[str, Mapping[str, object]],
    windows: Mapping[str, tuple[int, int]],
) -> int:
    aggregate_validation_count = 0
    mechanism = _object(report.get("mechanism_evidence"), "report mechanism_evidence")
    divergences = _array(
        mechanism.get("action_divergence_vs_full"),
        "report mechanism action divergences",
    )
    matches = [
        _object(row, "report action divergence")
        for row in divergences
        if _object(row, "report action divergence").get("comparator") == comparator
    ]
    if len(matches) != 1:
        _fail(f"report must contain exactly one action-divergence row for {comparator}")
    _require_equal(
        matches[0].get("matched_decision_count"),
        all_window.get("matched_decision_count"),
        f"{comparator} report action denominator",
    )
    _require_equal(
        matches[0].get("action_divergence_count"),
        all_window.get("action_discordant_count"),
        f"{comparator} report action divergence",
    )
    frozen = _frozen_report_comparison(report, comparator)
    _require_equal(frozen.get("status"), "observed", f"{comparator} frozen paired status")
    _require_close(
        frozen.get("mean_paired_effect"),
        _aggregate_rate(
            primary_window,
            field="positive_outcome_net",
            label="primary positive outcome net",
        ),
        label=f"{comparator} frozen paired effect",
    )
    aggregate_validation_count += 1
    primary_world_directions = _object(
        primary_window.get("world_direction_counts"),
        "primary world direction counts",
    )
    _require_equal(
        frozen.get("subjects_with_positive_effect"),
        primary_world_directions.get("positive"),
        f"{comparator} frozen subjects_with_positive_effect",
    )
    aggregate_validation_count += 1

    safety = _unique_comparator_row(
        report,
        field="safety_noninferiority_comparisons",
        comparator=comparator,
    )
    _require_equal(safety.get("status"), "observed", f"{comparator} frozen safety status")
    _require_close(
        safety.get("mean_full_safety_rate_increase"),
        _aggregate_rate(
            primary_window,
            field="full_safety_negative_increase",
            label="primary safety-negative increase",
        ),
        label=f"{comparator} frozen safety increase",
    )
    aggregate_validation_count += 1

    horizon = _unique_comparator_row(
        report,
        field="horizon_segment_comparisons",
        comparator=comparator,
    )
    _require_equal(horizon.get("status"), "observed", f"{comparator} frozen horizon status")
    frozen_segments = _array(horizon.get("segments"), f"{comparator} frozen horizon segments")
    _require_equal(len(frozen_segments), len(windows), f"{comparator} frozen horizon segment count")
    observed_segment_names: list[str] = []
    for item in frozen_segments:
        frozen_segment = _object(item, f"{comparator} frozen horizon segment")
        segment = _text(frozen_segment.get("segment"), f"{comparator} frozen horizon segment name")
        if segment in observed_segment_names:
            _fail(f"{comparator} frozen report repeats horizon segment {segment!r}")
        if segment not in windows or segment not in segments:
            _fail(f"{comparator} frozen report has unknown horizon segment {segment!r}")
        observed_segment_names.append(segment)
        start, end = windows[segment]
        _require_equal(
            frozen_segment.get("decision_indices"),
            [start, end],
            f"{comparator} frozen {segment} decision indices",
        )
        aggregate = segments[segment]
        _require_close(
            frozen_segment.get("mean_paired_effect"),
            _aggregate_rate(
                aggregate,
                field="positive_outcome_net",
                label=f"{comparator} {segment} positive outcome net",
            ),
            label=f"{comparator} frozen {segment} paired effect",
        )
        aggregate_validation_count += 1
        directions = _object(
            aggregate.get("world_direction_counts"),
            f"{comparator} {segment} world direction counts",
        )
        _require_equal(
            frozen_segment.get("subjects_with_positive_effect"),
            directions.get("positive"),
            f"{comparator} frozen {segment} subjects_with_positive_effect",
        )
        aggregate_validation_count += 1
    _require_equal(
        sorted(observed_segment_names),
        sorted(windows),
        f"{comparator} frozen horizon segment name set",
    )
    _require_equal(
        aggregate_validation_count,
        13,
        f"{comparator} frozen aggregate validation count",
    )
    return aggregate_validation_count


def _reader_summary(
    records: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[str, object]:
    prediction_counts: Counter[str] = Counter()
    truth_counts: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    correct = 0
    primary_correct = 0
    primary_total = 0
    margins: list[float] = []
    conditions_by_truth_match: dict[bool, set[str]] = {True: set(), False: set()}
    for record in records.values():
        diagnostic = _reader_diagnostic(record)
        _require_equal(diagnostic.get("named_readout_present"), True, "full named reader presence")
        observed = _text(diagnostic.get("observed_reader_label"), "observed reader label")
        expected = _text(diagnostic.get("expected_reader_label"), "expected reader label")
        condition = _text(diagnostic.get("condition_id"), "reader truth condition")
        prediction_counts[observed] += 1
        truth_counts[condition] += 1
        confusion[(expected, observed)] += 1
        is_correct = _boolean(diagnostic.get("reader_truth_match"), "reader truth match")
        conditions_by_truth_match[is_correct].add(condition)
        correct += int(is_correct)
        decision_index = _integer(record.get("decision_index"), "reader decision index")
        if 12 <= decision_index <= 23:
            primary_total += 1
            primary_correct += int(is_correct)
        margins.append(_number(diagnostic.get("normalized_margin"), "reader normalized margin"))
    correct_conditions = sorted(conditions_by_truth_match[True])
    incorrect_conditions = sorted(conditions_by_truth_match[False])
    _require_equal(
        correct_conditions,
        ["agency_under_override"],
        "reader-correct condition partition",
    )
    _require_equal(
        incorrect_conditions,
        ["connection_under_exclusion"],
        "reader-incorrect condition partition",
    )
    return {
        "crosswalk": dict(sorted(_CONDITION_TO_READER_LABEL.items())),
        "matched_decision_count": len(records),
        "correct_count": correct,
        "incorrect_count": len(records) - correct,
        "primary_window_count": primary_total,
        "primary_window_correct_count": primary_correct,
        "truth_condition_counts": dict(sorted(truth_counts.items())),
        "observed_reader_label_counts": dict(sorted(prediction_counts.items())),
        "expected_to_observed_confusion": [
            {
                "expected_reader_label": expected,
                "observed_reader_label": observed,
                "count": count,
            }
            for (expected, observed), count in sorted(confusion.items())
        ],
        "all_named_outputs_collapsed_to_one_label": len(prediction_counts) == 1,
        "collapsed_label": next(iter(prediction_counts)) if len(prediction_counts) == 1 else None,
        "reader_truth_partition_perfectly_confounded_with_condition_id": True,
        "reader_truth_correct_condition_ids": correct_conditions,
        "reader_truth_incorrect_condition_ids": incorrect_conditions,
        "reader_error_causal_attribution_authorized": False,
        "normalized_margin_below_qualification_floor_count": sum(1 for margin in margins if margin < 0.01),
        "diagnostic_scope": ("post_hoc attempt03 static reader-truth crosswalk; not a reader qualification result"),
    }


def _summary_for_comparator(
    comparator: str,
    pairs: Sequence[Mapping[str, object]],
    windows: Mapping[str, tuple[int, int]],
) -> dict[str, object]:
    all_window = _aggregate_pairs(pairs, include=lambda _pair: True)
    primary = _aggregate_pairs(pairs, include=lambda pair: bool(pair["primary_window"]))
    segments = {
        segment: _aggregate_pairs(
            pairs,
            include=lambda pair, name=segment: pair.get("horizon_segment") == name,
        )
        for segment in windows
    }
    reader_truth_partitions: dict[str, object] = {}
    for name, expected_match in (("reader_truth_correct", True), ("reader_truth_incorrect", False)):
        reader_truth_partitions[name] = {
            "all_window": _aggregate_pairs(
                pairs,
                include=lambda pair, match=expected_match: (
                    _object(
                        _object(pair["full"], "pair full").get("reader_truth"),
                        "full reader truth",
                    ).get("reader_truth_match")
                    is match
                ),
            ),
            "primary_window": _aggregate_pairs(
                pairs,
                include=lambda pair, match=expected_match: (
                    bool(pair["primary_window"])
                    and _object(
                        _object(pair["full"], "pair full").get("reader_truth"),
                        "full reader truth",
                    ).get("reader_truth_match")
                    is match
                ),
            ),
        }
    return {
        "comparator": comparator,
        "all_24_decisions": all_window,
        "primary_window_decisions_12_through_23": primary,
        "horizon_segments": segments,
        "full_named_reader_truth_partitions": reader_truth_partitions,
    }


def _build_pair_rows(
    all_pairs: Mapping[str, Sequence[Mapping[str, object]]],
    primary_world_directions: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for comparator in _COMPARATORS:
        for pair in all_pairs[comparator]:
            comparison = _object(pair.get("comparison"), "pair comparison")
            if not comparison.get("action_discordant"):
                continue
            row = deepcopy(dict(pair))
            world = _sha256(row.get("world_clone_id"), "pair row world")
            row["world_primary_direction"] = deepcopy(primary_world_directions[comparator][world])
            row["join_contract"] = _join_contract()
            rows.append(_with_row_id(row))
    return rows


def _build_unique_rows(pair_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in pair_rows:
        key = (
            _text(row.get("subject_scope"), "pair row subject_scope"),
            _text(row.get("decision_id"), "pair row decision_id"),
        )
        grouped[key].append(row)
    rows: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1])):
        members = sorted(grouped[key], key=lambda row: _COMPARATORS.index(_text(row["comparator"], "comparator")))
        first = members[0]
        for member in members[1:]:
            for field in (
                "subject_scope",
                "world_clone_id",
                "decision_id",
                "decision_index",
                "primary_window",
                "horizon_segment",
                "truth",
                "full",
            ):
                _require_equal(member.get(field), first.get(field), f"unique-decision {key} field {field}")
        unique = {
            field: deepcopy(first.get(field))
            for field in (
                "subject_scope",
                "world_clone_id",
                "decision_id",
                "decision_index",
                "primary_window",
                "horizon_segment",
                "truth",
                "full",
            )
        }
        unique["divergent_comparator_count"] = len(members)
        unique["divergent_comparators"] = [member["comparator"] for member in members]
        unique["comparisons"] = [
            {
                "pair_row_id": member["row_id"],
                "comparator": member["comparator"],
                "comparator_arm": deepcopy(member["comparator_arm"]),
                "comparison": deepcopy(member["comparison"]),
                "world_primary_direction": deepcopy(member["world_primary_direction"]),
            }
            for member in members
        ]
        unique["join_contract"] = deepcopy(first["join_contract"])
        rows.append(_with_row_id(unique))
    rows.sort(key=lambda row: (str(row["subject_scope"]), int(row["decision_index"])))
    return rows


def _artifact_document(schema_version: str, **payload: object) -> tuple[dict[str, object], bytes]:
    document = _with_artifact_id({"schema_version": schema_version, **payload})
    return document, _canonical_json_bytes(document) + b"\n"


def build_audit_documents(attempt_root: Path) -> dict[str, bytes]:
    """Recompute every audit output in memory from the immutable attempt tree."""

    reader = _AttemptReader(attempt_root)
    _protocol, report, windows, source_provenance, runtime_contract = _validate_protocol_and_report(reader)
    post_hoc_reader_authority = _load_post_hoc_reader_crosswalk_authority()
    owner_history_authority = _load_owner_history_audit_authority()
    subject_scopes = sorted(path.name for path in (reader.root / "chains").iterdir() if path.is_dir())
    _require_equal(len(subject_scopes), 8, "attempt subject scope count")
    for subject_scope in subject_scopes:
        _sha256(subject_scope, "subject_scope")

    records_by_arm: dict[str, dict[tuple[str, str], dict[str, object]]] = {arm: {} for arm in _AUDITED_ARMS}
    chain_references: list[dict[str, object]] = []
    for subject_scope in subject_scopes:
        expected_world: str | None = None
        for arm in _AUDITED_ARMS:
            chain, records = _load_arm_records(
                reader,
                subject_scope=subject_scope,
                arm_id=arm,
                runtime_contract=runtime_contract,
            )
            world = _sha256(chain.get("world_clone_id"), f"{arm} chain world_clone_id")
            if expected_world is None:
                expected_world = world
            _require_equal(world, expected_world, f"{subject_scope} cross-arm world_clone_id")
            overlap = set(records_by_arm[arm]).intersection(records)
            if overlap:
                _fail(f"duplicate join keys while loading arm {arm}: {sorted(overlap)[:3]}")
            records_by_arm[arm].update(records)
            chain_path = f"chains/{subject_scope}/{arm}/chain.json"
            chain_references.append(reader.reference(chain_path, chain))
    for arm in _AUDITED_ARMS:
        _require_equal(len(records_by_arm[arm]), 192, f"{arm} decision count")
    execution_lineage_artifact_ids = {
        _sha256(
            _object(
                _object(record.get("mechanism"), "record mechanism").get("execution_source_lineage"),
                "record execution source lineage",
            ).get("artifact_id"),
            "record execution source lineage artifact_id",
        )
        for records in records_by_arm.values()
        for record in records.values()
    }
    _require_equal(
        len(execution_lineage_artifact_ids),
        1,
        "audited execution lineage artifact identity cardinality",
    )
    execution_lineage_artifact_id = next(iter(execution_lineage_artifact_ids))

    all_pairs = _build_all_pairs(records_by_arm, windows)
    comparator_summaries: dict[str, dict[str, object]] = {}
    primary_world_directions: dict[str, dict[str, dict[str, object]]] = {}
    frozen_aggregate_validation_count = 0
    for comparator in _COMPARATORS:
        summary = _summary_for_comparator(comparator, all_pairs[comparator], windows)
        comparator_summaries[comparator] = summary
        frozen_aggregate_validation_count += _validate_report_aggregates(
            report,
            comparator,
            _object(summary["all_24_decisions"], "all-window summary"),
            _object(summary["primary_window_decisions_12_through_23"], "primary summary"),
            _object(summary["horizon_segments"], "horizon segment summary"),
            windows,
        )
        primary_rows = _array(
            _object(summary["primary_window_decisions_12_through_23"], "primary summary").get("world_directions"),
            "primary world directions",
        )
        primary_world_directions[comparator] = {
            _sha256(_object(row, "primary world direction").get("world_clone_id"), "world id"): _object(
                row,
                "primary world direction",
            )
            for row in primary_rows
        }
    _require_equal(
        frozen_aggregate_validation_count,
        26,
        "all frozen report aggregate validation count",
    )

    pair_rows = _build_pair_rows(all_pairs, primary_world_directions)
    unique_rows = _build_unique_rows(pair_rows)
    counts_by_comparator = Counter(_text(row["comparator"], "pair comparator") for row in pair_rows)
    _require_equal(counts_by_comparator, Counter({arm: 36 for arm in _COMPARATORS}), "pair-row counts")
    _require_equal(len(pair_rows), 72, "pair-row total")
    _require_equal(len(unique_rows), 50, "unique-decision-row total")
    divergence_sets = {
        comparator: {
            (_text(row["subject_scope"], "pair subject"), _text(row["decision_id"], "pair decision"))
            for row in pair_rows
            if row["comparator"] == comparator
        }
        for comparator in _COMPARATORS
    }
    intersection = divergence_sets[_COMPARATORS[0]] & divergence_sets[_COMPARATORS[1]]
    only_first = divergence_sets[_COMPARATORS[0]] - divergence_sets[_COMPARATORS[1]]
    only_second = divergence_sets[_COMPARATORS[1]] - divergence_sets[_COMPARATORS[0]]
    union = divergence_sets[_COMPARATORS[0]] | divergence_sets[_COMPARATORS[1]]
    _require_equal(
        (len(intersection), len(only_first), len(only_second), len(union)),
        (22, 14, 14, 50),
        "divergence-set overlap",
    )

    reader_summary = _reader_summary(records_by_arm[_FULL_ARM])
    _require_equal(reader_summary["correct_count"], 96, "full reader correct count")
    _require_equal(reader_summary["incorrect_count"], 96, "full reader incorrect count")
    _require_equal(
        reader_summary["observed_reader_label_counts"],
        {"agency_displacement": 192},
        "full reader collapsed output",
    )
    reader_summary["post_hoc_crosswalk_authority"] = post_hoc_reader_authority

    pair_document, pair_raw = _artifact_document(
        "relationship-product-horizon-attempt03-divergence-pair-rows.v1",
        audit_scope="post_hoc_model_free_mechanism_diagnosis",
        source_protocol_id=_EXPECTED_PROTOCOL_ID,
        join_contract=_join_contract(),
        row_count=len(pair_rows),
        rows=pair_rows,
    )
    unique_document, unique_raw = _artifact_document(
        "relationship-product-horizon-attempt03-divergence-unique-decision-rows.v1",
        audit_scope="post_hoc_model_free_mechanism_diagnosis",
        source_protocol_id=_EXPECTED_PROTOCOL_ID,
        join_contract=_join_contract(),
        row_count=len(unique_rows),
        rows=unique_rows,
    )

    report_payload: dict[str, object] = {
        "audit_scope": "post_hoc_model_free_mechanism_diagnosis",
        "source_attempt": {
            "attempt_root": reader.root_relative,
            "manifest_raw_sha256": reader.manifest_raw_sha256,
            "manifest_artifact_id": _EXPECTED_MANIFEST_ARTIFACT_ID,
            "protocol_id": _EXPECTED_PROTOCOL_ID,
            "protocol_raw_sha256": _EXPECTED_PROTOCOL_RAW_SHA256,
            "report_artifact_id": _EXPECTED_REPORT_ARTIFACT_ID,
            "report_raw_sha256": _EXPECTED_REPORT_RAW_SHA256,
            "source_protocol_id": _EXPECTED_SOURCE_PROTOCOL_ID,
            "source_public_plan_id": _EXPECTED_SOURCE_PUBLIC_PLAN_ID,
            "source_sealed_bundle_id": _EXPECTED_SOURCE_SEALED_BUNDLE_ID,
            "source_public_and_sealed": source_provenance,
            "runtime_contract": {
                **runtime_contract,
                "execution_lineage_artifact_id": execution_lineage_artifact_id,
            },
            "frozen_verdict": report.get("verdict"),
        },
        "join_contract": _join_contract(include_reason=True),
        "denominators": {
            "matched_world_count": 8,
            "decisions_per_world_all_window": 24,
            "matched_pair_decisions_per_comparator_all_window": 192,
            "decisions_per_world_primary_window": 12,
            "matched_pair_decisions_per_comparator_primary_window": 96,
            "pair_rows_action_discordant_only": 72,
            "unique_action_divergent_decisions": 50,
        },
        "divergence_set_overlap": {
            "intersection_count": len(intersection),
            f"{_COMPARATORS[0]}_only_count": len(only_first),
            f"{_COMPARATORS[1]}_only_count": len(only_second),
            "union_count": len(union),
            "pair_row_count": len(pair_rows),
        },
        "comparators": [comparator_summaries[arm] for arm in _COMPARATORS],
        "frozen_report_recomputation": {
            "aggregate_values_per_comparator": 13,
            "aggregate_values_total": frozen_aggregate_validation_count,
            "all_recomputed_values_match": True,
        },
        "reader_truth_crosswalk_diagnostic": reader_summary,
        "owner_history_diagnostic_authority": owner_history_authority,
        "safety_complement_diagnostic": {
            "positive_outcome_ids": sorted(_POSITIVE_OUTCOME_IDS),
            "safety_negative_outcome_ids": sorted(_SAFETY_NEGATIVE_OUTCOME_IDS),
            "all_audited_arm_decisions_exactly_one_of_positive_or_safety_negative": True,
            "audited_arm_decision_count": 576,
            "safety_difference_is_negative_positive_difference_for_every_aggregate": True,
            "independent_safety_evidence": False,
            "explanation": (
                "Within this typed-outcome support, safety-negative is the exact complement of "
                "positive outcome; the safety percentage repeats the same discordant events."
            ),
        },
        "outputs": {
            "pair_rows": {
                "path": "pair_rows.json",
                "artifact_id": pair_document["artifact_id"],
                "raw_sha256": _sha256_bytes(pair_raw),
                "bytes": len(pair_raw),
                "row_count": len(pair_rows),
            },
            "unique_decision_rows": {
                "path": "unique_decision_rows.json",
                "artifact_id": unique_document["artifact_id"],
                "raw_sha256": _sha256_bytes(unique_raw),
                "bytes": len(unique_raw),
                "row_count": len(unique_rows),
            },
        },
        "frozen_judgment_preserved": {
            "verdict": report.get("verdict"),
            "product_stage_two_effect_observed": report.get("product_stage_two_effect_observed"),
            "four_able_complete": False,
            "formal_evidence_authorized": False,
            "single_axis_contrast_claim_authorized": False,
            "attempt03_files_modified": False,
        },
        "honest_boundaries": {
            "post_hoc_diagnosis": True,
            "pre_registered_confirmatory_analysis": False,
            "action_divergence_is_post_treatment_mechanism_support_not_itt_estimand": True,
            "original_matched_decision_denominators_remain_authoritative": True,
            "reader_truth_partition_perfectly_confounded_with_condition_id": True,
            "reader_error_causal_attribution_authorized": False,
            "reader_crosswalk_authority_is_post_hoc": True,
            "reader_static_accuracy_proves_dynamic_writeback": False,
            "owner_snapshot_receipts_are_hash_only": True,
            "owner_semantic_delta_materialized_in_divergence_rows": False,
            "external_owner_semantic_audit_materialized": True,
            "external_owner_semantic_audit_bound": True,
            "owner_state_causal_attribution_authorized": False,
            "owner_history_artifact_bound": True,
            "human_product_validation": False,
            "production_active": False,
        },
        "source_chain_references": sorted(
            chain_references,
            key=lambda row: _text(row["path"], "chain reference path"),
        ),
        "implementation": {
            "model_output_count": 0,
            "cuda_required": False,
            "network_required": False,
            "evaluation_or_judge_used_as_learning_signal": False,
        },
    }
    report_document, report_raw = _artifact_document(
        "relationship-product-horizon-attempt03-divergence-audit-report.v1",
        **report_payload,
    )
    manifest_document, manifest_raw = _artifact_document(
        "relationship-product-horizon-attempt03-divergence-audit-manifest.v1",
        manifest_written_last=True,
        source_attempt_manifest_raw_sha256=reader.manifest_raw_sha256,
        report_artifact_id=report_document["artifact_id"],
        files=[
            {
                "path": "pair_rows.json",
                "bytes": len(pair_raw),
                "raw_sha256": _sha256_bytes(pair_raw),
                "artifact_id": pair_document["artifact_id"],
            },
            {
                "path": "unique_decision_rows.json",
                "bytes": len(unique_raw),
                "raw_sha256": _sha256_bytes(unique_raw),
                "artifact_id": unique_document["artifact_id"],
            },
            {
                "path": "report.json",
                "bytes": len(report_raw),
                "raw_sha256": _sha256_bytes(report_raw),
                "artifact_id": report_document["artifact_id"],
            },
        ],
    )
    return {
        "pair_rows.json": pair_raw,
        "unique_decision_rows.json": unique_raw,
        "report.json": report_raw,
        "manifest.json": manifest_raw,
    }


def _write_documents_create_only(output_dir: Path, documents: Mapping[str, bytes]) -> None:
    output = _audit_output_path(output_dir, label="audit output directory")
    if output.exists():
        _fail(f"refusing to overwrite existing audit output: {output}")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    if temp.exists():
        _fail(f"unexpected temporary output collision: {temp}")
    temp.mkdir()
    try:
        for name in _OUTPUT_FILES:
            raw = documents.get(name)
            if raw is None:
                _fail(f"missing generated output document: {name}")
            path = temp / name
            with path.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            _require_equal(path.read_bytes(), raw, f"fresh audit output {name}")
        if output.exists():
            _fail(f"audit output appeared concurrently: {output}")
        temp.rename(output)
    except BaseException:
        if temp.exists() and temp.parent == parent and temp.name.startswith(f".{output.name}.tmp-"):
            shutil.rmtree(temp)
        raise


def materialize(attempt_root: Path, output_dir: Path) -> dict[str, object]:
    documents = build_audit_documents(attempt_root)
    _write_documents_create_only(output_dir, documents)
    report = _strict_json_object(documents["report.json"], label="generated report")
    manifest = _strict_json_object(documents["manifest.json"], label="generated manifest")
    return {
        "output_dir": _audit_output_label(output_dir, label="audit output directory"),
        "report_artifact_id": report["artifact_id"],
        "manifest_artifact_id": manifest["artifact_id"],
        "pair_row_count": 72,
        "unique_decision_row_count": 50,
    }


def validate_existing(attempt_root: Path, output_dir: Path) -> dict[str, object]:
    expected = build_audit_documents(attempt_root)
    output = _audit_output_path(output_dir, label="audit output directory")
    if not output.is_dir():
        _fail(f"audit output directory does not exist: {output}")
    observed_names = sorted(path.name for path in output.iterdir())
    _require_equal(observed_names, sorted(_OUTPUT_FILES), "audit output exact file set")
    for name in _OUTPUT_FILES:
        raw = (output / name).read_bytes()
        _require_equal(raw, expected[name], f"audit output byte-exact replay {name}")
        payload = _strict_json_object(raw, label=f"audit output {name}")
        _require_artifact_identity(payload, label=f"audit output {name}")
    report = _strict_json_object(expected["report.json"], label="expected audit report")
    manifest = _strict_json_object(expected["manifest.json"], label="expected audit manifest")
    return {
        "output_dir": _audit_output_label(output, label="audit output directory"),
        "report_artifact_id": report["artifact_id"],
        "manifest_artifact_id": manifest["artifact_id"],
        "byte_exact_replay": True,
        "pair_row_count": 72,
        "unique_decision_row_count": 50,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model-free post-hoc divergence audit for the pinned Product Horizon attempt03",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("materialize", "validate-existing"):
        command = commands.add_parser(name)
        command.add_argument("--attempt-root", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        result = materialize(args.attempt_root, args.output_dir)
    elif args.command == "validate-existing":
        result = validate_existing(args.attempt_root, args.output_dir)
    else:  # pragma: no cover - argparse enforces the command set.
        _fail(f"unsupported command: {args.command}")
    print(_canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
