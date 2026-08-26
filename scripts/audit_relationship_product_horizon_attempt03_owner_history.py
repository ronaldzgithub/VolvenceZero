#!/usr/bin/env python3
"""Replay attempt03 owner history and hard-window forecast contributions.

This is a model-free, post-hoc diagnostic.  It reads only manifest-bound
attempt03 artifacts and the attempt's frozen public embedding table.  It does
not import the current runtime, run a model, revise the frozen verdict, or
authorize a Learnable/product-causal claim.
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
from typing import Mapping, Sequence
import uuid


sys.dont_write_bytecode = True


_REPO_ROOT = Path(__file__).resolve().parents[1]
_FULL_ARM = "volvence_full"
_FROZEN_ARM = "appendable_frozen_onboarding"
_ARMS = (_FULL_ARM, _FROZEN_ARM)
_RECORD_WINDOW = 12
_PRIOR_COUNT = 1.0
_EVIDENCE_WEIGHT = 4.0
_INTERLOCUTOR_ID = "primary"
_POSITIVE_OUTCOME_IDS = frozenset({"helped", "felt_heard"})
_NEGATIVE_OUTCOME_IDS = frozenset({"missed", "over_directive"})
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
_EMBEDDING_TABLE_PATH = "inputs/public_embedding_table.json"
_EXPECTED_EMBEDDING_TABLE_ID = "75ef759e82a1490dd0a8ce7402ce2e77e156ca957196caa44705386b0a12af57"
_EXPECTED_EMBEDDING_TABLE_RAW_SHA256 = "894b508f4cfaebea24f66076b17ba8bf1ea812ac61ef36605b6cda8b2fad32fc"
_EXPECTED_READER_ARTIFACT_ID = "0f4ba01d0f356980ca3dcf6947a7c0580ed825f97943d86c78477b3e10e0a6a6"
_BGE_MODEL_ID = "BAAI/bge-m3"
_BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_BGE_WEIGHTS_SHA256 = "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"
_BGE_RUNTIME_VERSION = "5.6.0"
_READER_TEMPERATURE = 0.05
_PROTOTYPES = (
    (
        "agency_displacement",
        "别人越过当事人的确认，替其表达、选择、承诺或作决定，使当事人失去发言权、决定权以及按自己节奏回应的空间。",
    ),
    (
        "belonging_erasure",
        "当事人被共同经历、邀请、名单、记忆或关系网络遗漏和排除，因而感觉自己在关系中的位置消失、没有被算在其中。",
    ),
)
_CONDITION_TO_READER_LABEL = {
    "agency_under_override": "agency_displacement",
    "connection_under_exclusion": "belonging_erasure",
}
_SOURCE_PATHS = {
    "record_store": ("inputs/execution_sources/tree/packages/vz-cognition/src/volvence_zero/social/record_store.py"),
    "tom": ("inputs/execution_sources/tree/packages/vz-cognition/src/volvence_zero/social/tom.py"),
    "relationship_forecast": (
        "inputs/execution_sources/tree/packages/lifeform-domain-emogpt/src/"
        "lifeform_domain_emogpt/relationship_forecast.py"
    ),
    "condition_reader": (
        "inputs/execution_sources/tree/packages/lifeform-domain-emogpt/src/"
        "lifeform_domain_emogpt/relationship_condition_reader.py"
    ),
    "product_horizon": (
        "inputs/execution_sources/tree/packages/lifeform-evolution/src/"
        "lifeform_evolution/relationship_lab_product_horizon.py"
    ),
    "model_adapters": (
        "inputs/execution_sources/tree/packages/lifeform-evolution/src/"
        "lifeform_evolution/relationship_lab_product_model_adapters.py"
    ),
}
_OUTPUT_FILES = ("rows.json", "report.json", "manifest.json")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OWNER_VERSION_RE = re.compile(r"owner_hydration__social_record_store_v([1-9][0-9]*)\.json")


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


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_json(payload: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(payload))


def _artifact_id(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("artifact_id", None)
    return _sha256_json(unsigned)


def _with_artifact_id(payload: Mapping[str, object]) -> dict[str, object]:
    if "artifact_id" in payload:
        _fail("artifact payload must not predeclare artifact_id")
    unsigned = dict(payload)
    return {**unsigned, "artifact_id": _artifact_id(unsigned)}


def _with_row_id(payload: Mapping[str, object]) -> dict[str, object]:
    if "row_id" in payload:
        _fail("row payload must not predeclare row_id")
    unsigned = dict(payload)
    return {**unsigned, "row_id": _sha256_json(unsigned)}


def _reject_constant(value: str) -> object:
    raise AuditContractError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditContractError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} must be one JSON object")
    return value


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
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{label} must be boolean")
    return value


def _sha256(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256_RE.fullmatch(result) is None:
        _fail(f"{label} must be a lowercase SHA-256")
    return result


def _require_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        _fail(f"{label} mismatch: observed={observed!r}, expected={expected!r}")


def _require_exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        _fail(
            f"{label} keys mismatch: missing={sorted(expected.difference(value))!r}, "
            f"extra={sorted(set(value).difference(expected))!r}"
        )


def _require_artifact_identity(payload: Mapping[str, object], *, label: str) -> str:
    observed = _sha256(payload.get("artifact_id"), f"{label}.artifact_id")
    _require_equal(observed, _artifact_id(payload), f"{label}.artifact_id")
    return observed


def _repo_relative(path: Path, *, label: str) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError as exc:
        raise AuditContractError(f"{label} must be inside repository: {resolved}") from exc


def _normalized_member_path(value: object, *, label: str) -> str:
    result = _text(value, label)
    candidate = PurePosixPath(result)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        _fail(f"{label} must be a normalized relative POSIX path")
    if candidate.as_posix() != result or "\\" in result:
        _fail(f"{label} must be canonical POSIX form: {result!r}")
    return result


def _float_payload(value: object, label: str) -> dict[str, object]:
    number = _number(value, label)
    return {"value": number, "hex": number.hex()}


def hard_window(values: Sequence[object], *, size: int = _RECORD_WINDOW) -> tuple[object, ...]:
    """Apply the frozen source-order window without interpreting contents."""

    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        _fail("hard-window size must be a positive integer")
    return tuple(values)[-size:]


class _AttemptReader:
    """Read only files cryptographically bound by one attempt manifest."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        if not self.root.is_dir():
            _fail(f"attempt root does not exist: {self.root}")
        self.root_relative = _repo_relative(self.root, label="attempt root")
        raw = (self.root / "manifest.json").read_bytes()
        _require_equal(_sha256_bytes(raw), _EXPECTED_MANIFEST_RAW_SHA256, "manifest raw SHA-256")
        self.manifest = _strict_json_object(raw, label="attempt manifest")
        _require_artifact_identity(self.manifest, label="attempt manifest")
        _require_equal(
            self.manifest.get("artifact_id"),
            _EXPECTED_MANIFEST_ARTIFACT_ID,
            "attempt manifest artifact_id",
        )
        _require_equal(self.manifest.get("protocol_id"), _EXPECTED_PROTOCOL_ID, "manifest protocol_id")
        rows = _array(self.manifest.get("files"), "attempt manifest.files")
        self.entries: dict[str, dict[str, object]] = {}
        for index, value in enumerate(rows):
            entry = _object(value, f"attempt manifest.files[{index}]")
            path = _normalized_member_path(entry.get("path"), label="manifest file path")
            if path in self.entries:
                _fail(f"duplicate attempt manifest member: {path}")
            _sha256(entry.get("sha256"), f"manifest {path} sha256")
            if _integer(entry.get("bytes"), f"manifest {path} bytes") < 0:
                _fail(f"manifest {path} byte count must be non-negative")
            self.entries[path] = entry

    def raw(self, relative_path: str) -> bytes:
        path = _normalized_member_path(relative_path, label="attempt member path")
        entry = self.entries.get(path)
        if entry is None:
            _fail(f"attempt member is not manifest-bound: {path}")
        full = self.root.joinpath(*PurePosixPath(path).parts)
        if not full.is_file():
            _fail(f"attempt member is missing: {path}")
        raw = full.read_bytes()
        _require_equal(len(raw), entry["bytes"], f"{path} bytes")
        _require_equal(_sha256_bytes(raw), entry["sha256"], f"{path} raw SHA-256")
        return raw

    def load(self, relative_path: str, *, verify_artifact_id: bool = True) -> dict[str, object]:
        payload = _strict_json_object(self.raw(relative_path), label=relative_path)
        if verify_artifact_id and "artifact_id" in payload:
            _require_artifact_identity(payload, label=relative_path)
        return payload

    def reference(
        self,
        relative_path: str,
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        path = _normalized_member_path(relative_path, label="source reference path")
        self.raw(path)
        entry = self.entries[path]
        result: dict[str, object] = {
            "path": path,
            "bytes": entry["bytes"],
            "raw_sha256": entry["sha256"],
        }
        if payload is not None:
            if "schema_version" in payload:
                result["schema_version"] = payload["schema_version"]
            if "artifact_id" in payload:
                result["artifact_id"] = _sha256(payload["artifact_id"], f"{path} artifact_id")
        return result


def _reader_artifact_payload() -> dict[str, object]:
    payload = {
        "schema_version": "relationship-condition-reader-artifact.v1",
        "embedding_model_id": f"{_BGE_MODEL_ID}@revision:{_BGE_REVISION}",
        "embedding_weights_sha256": _BGE_WEIGHTS_SHA256,
        "semantic_similarity": "cosine",
        "softmax_temperature": _READER_TEMPERATURE,
        "prototypes": [
            {
                "label": label,
                "summary": summary,
                "summary_sha256": _sha256_text(summary),
            }
            for label, summary in _PROTOTYPES
        ],
    }
    _require_equal(_sha256_json(payload), _EXPECTED_READER_ARTIFACT_ID, "reader artifact identity")
    return payload


def _load_embedding_table(
    reader: _AttemptReader,
) -> tuple[dict[str, tuple[float, ...]], dict[str, object], dict[str, object]]:
    raw = reader.raw(_EMBEDDING_TABLE_PATH)
    _require_equal(_sha256_bytes(raw), _EXPECTED_EMBEDDING_TABLE_RAW_SHA256, "embedding table raw SHA")
    payload = _strict_json_object(raw, label="public embedding table")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "source_embedder_name",
            "source_model_id",
            "source_model_revision",
            "embedding_width",
            "records",
            "artifact_id",
        },
        "public embedding table",
    )
    _require_equal(
        payload.get("schema_version"),
        "relationship-product-public-embedding-table.v2",
        "embedding table schema",
    )
    _require_equal(payload.get("artifact_id"), _EXPECTED_EMBEDDING_TABLE_ID, "embedding table ID")
    _require_artifact_identity(payload, label="public embedding table")
    _require_equal(payload.get("source_model_id"), _BGE_MODEL_ID, "embedding source model")
    _require_equal(payload.get("source_model_revision"), _BGE_REVISION, "embedding revision")
    source_name = _text(payload.get("source_embedder_name"), "embedding source name")
    for token in (_BGE_MODEL_ID, _BGE_REVISION, _BGE_WEIGHTS_SHA256, _BGE_RUNTIME_VERSION):
        if token not in source_name:
            _fail(f"embedding source identity is missing frozen token {token}")
    width = _integer(payload.get("embedding_width"), "embedding width")
    if width < 2:
        _fail("embedding width must be at least two")
    vectors: dict[str, tuple[float, ...]] = {}
    prior_order: tuple[str, str] | None = None
    for index, value in enumerate(_array(payload.get("records"), "embedding records")):
        record = _object(value, f"embedding records[{index}]")
        _require_exact_keys(
            record,
            {"schema_version", "text", "text_sha256", "embedding_hex", "artifact_id"},
            f"embedding records[{index}]",
        )
        _require_equal(
            record.get("schema_version"),
            "relationship-product-public-embedding-record.v1",
            "embedding record schema",
        )
        _require_artifact_identity(record, label=f"embedding records[{index}]")
        text = _text(record.get("text"), "embedding text")
        digest = _sha256(record.get("text_sha256"), "embedding text SHA")
        _require_equal(digest, _sha256_text(text), "embedding text SHA")
        ordering = (digest, text)
        if prior_order is not None and ordering <= prior_order:
            _fail("embedding records are not in strict canonical digest/text order")
        prior_order = ordering
        encoded = _array(record.get("embedding_hex"), "embedding vector")
        if len(encoded) != width:
            _fail("embedding vector width mismatch")
        vector: list[float] = []
        for component_index, raw_component in enumerate(encoded):
            component = _text(raw_component, f"embedding component {component_index}")
            try:
                number = float.fromhex(component)
            except ValueError as exc:
                raise AuditContractError("embedding component is not a hexadecimal float") from exc
            if not math.isfinite(number) or number.hex() != component:
                _fail("embedding component is not finite canonical float.hex()")
            vector.append(number)
        if digest in vectors:
            _fail(f"duplicate embedding text digest: {digest}")
        vectors[digest] = tuple(vector)
    if raw != _canonical_json_bytes(payload) + b"\n":
        _fail("public embedding table is not exact canonical JSON plus LF")
    reader_payload = _reader_artifact_payload()
    for _label, summary in _PROTOTYPES:
        if _sha256_text(summary) not in vectors:
            _fail("public embedding table is missing a frozen reader prototype")
    return vectors, payload, reader_payload


def _vector_for_text(
    vectors: Mapping[str, tuple[float, ...]],
    text: str,
) -> tuple[float, ...]:
    digest = _sha256_text(text)
    try:
        raw = vectors[digest]
    except KeyError as exc:
        raise AuditContractError(f"public embedding table has no exact text row: {digest}") from exc
    norm = math.sqrt(math.fsum(value * value for value in raw))
    if norm <= 1e-12:
        _fail(f"public embedding vector has non-positive norm: {digest}")
    return tuple(value / norm for value in raw)


def _condition_readout(
    text: str,
    vectors: Mapping[str, tuple[float, ...]],
) -> dict[str, object]:
    vector = _vector_for_text(vectors, text)
    scores: list[tuple[str, float]] = []
    for label, summary in _PROTOTYPES:
        prototype = _vector_for_text(vectors, summary)
        if len(vector) != len(prototype):
            _fail("condition reader embedding width mismatch")
        score = math.fsum(a * b for a, b in zip(vector, prototype, strict=True))
        scores.append((label, max(-1.0, min(1.0, score))))
    ordered = sorted(scores, key=lambda item: item[1], reverse=True)
    top_label, top_score = ordered[0]
    normalized_margin = min(1.0, max(0.0, (top_score - ordered[1][1]) / 2.0))
    maximum = max(score for _, score in scores)
    exponentials = tuple(math.exp((score - maximum) / _READER_TEMPERATURE) for _, score in scores)
    top_index = next(index for index, (label, _) in enumerate(scores) if label == top_label)
    confidence = exponentials[top_index] / math.fsum(exponentials)
    return {
        "condition_label": top_label,
        "confidence": confidence,
        "normalized_margin": normalized_margin,
        "candidate_scores": [{"label": label, "score": score} for label, score in scores],
        "reader_artifact_id": _EXPECTED_READER_ARTIFACT_ID,
        "source_observation_sha256": _sha256_text(text),
    }


def require_active_outcome_pairing(
    records: Sequence[Mapping[str, object]],
    outcomes: Sequence[Mapping[str, object]],
    *,
    interlocutor_id: str = _INTERLOCUTOR_ID,
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    """Mirror TOM eligibility, then close its record/outcome pairing gap."""

    eligible_records = tuple(
        record
        for record in records
        if record.get("interlocutor_id") == interlocutor_id and record.get("status") == "active"
    )
    eligible_outcomes = tuple(outcome for outcome in outcomes if outcome.get("interlocutor_id") == interlocutor_id)
    record_ids = tuple(_text(item.get("record_id"), "eligible record_id") for item in eligible_records)
    outcome_ids = tuple(_text(item.get("evidence_id"), "eligible outcome evidence_id") for item in eligible_outcomes)
    if len(set(record_ids)) != len(record_ids):
        _fail("eligible ACTIVE record IDs must be unique")
    if len(set(outcome_ids)) != len(outcome_ids):
        _fail("eligible outcome evidence IDs must be unique")
    _require_equal(outcome_ids, record_ids, "eligible ACTIVE record/outcome ordered exact join")
    return eligible_records, eligible_outcomes


def recompute_forecast_proposal(
    *,
    current_observation: str,
    candidate_action_ids: Sequence[str],
    outcome_ids: Sequence[str],
    records: Sequence[Mapping[str, object]],
    outcomes: Sequence[Mapping[str, object]],
    vectors: Mapping[str, tuple[float, ...]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Recompute the frozen named-reader similarity-squared proposal."""

    eligible_records, eligible_outcomes = require_active_outcome_pairing(records, outcomes)
    if not eligible_outcomes:
        _fail("owner-history audit requires at least one eligible outcome")
    action_surface = tuple(_text(value, "candidate action") for value in candidate_action_ids)
    outcome_surface = tuple(_text(value, "candidate outcome") for value in outcome_ids)
    if len(action_surface) < 2 or len(set(action_surface)) != len(action_surface):
        _fail("candidate action surface must be unique and contain at least two values")
    if len(outcome_surface) < 2 or len(set(outcome_surface)) != len(outcome_surface):
        _fail("candidate outcome surface must be unique and contain at least two values")
    record_by_id = {_text(item.get("record_id"), "record ID"): item for item in eligible_records}
    current_readout = _condition_readout(current_observation, vectors)
    weighted: list[tuple[Mapping[str, object], float, float, dict[str, object]]] = []
    contributions: list[dict[str, object]] = []
    for outcome in eligible_outcomes:
        evidence_id = _text(outcome.get("evidence_id"), "outcome evidence_id")
        record = record_by_id[evidence_id]
        action_id = _text(outcome.get("action_id"), "outcome action_id")
        outcome_id = _text(outcome.get("observed_outcome_id"), "observed outcome_id")
        if action_id not in action_surface or outcome_id not in outcome_surface:
            _fail("persisted outcome is outside the requested action/outcome surface")
        observation = _text(outcome.get("observation_summary"), "outcome observation_summary")
        prior_readout = _condition_readout(observation, vectors)
        similarity = (
            0.0
            if current_readout["condition_label"] != prior_readout["condition_label"]
            else min(
                _number(current_readout["confidence"], "current reader confidence"),
                _number(prior_readout["confidence"], "prior reader confidence"),
            )
        )
        similarity_squared = float(similarity) ** 2
        mass = _EVIDENCE_WEIGHT * similarity_squared
        weighted.append((outcome, similarity, similarity_squared, prior_readout))
        contributions.append(
            {
                "record_id": evidence_id,
                "action_id": action_id,
                "observed_outcome_id": outcome_id,
                "source_turn": _integer(outcome.get("source_turn"), "outcome source_turn"),
                "record_status": record.get("status"),
                "record_confidence": _float_payload(record.get("confidence"), "record confidence"),
                "record_prediction_error_refs": deepcopy(record.get("prediction_error_refs")),
                "observation_summary_sha256": _sha256_text(observation),
                "prior_named_readout": prior_readout,
                "semantic_similarity": similarity,
                "similarity_hex": similarity.hex(),
                "similarity_squared": similarity_squared,
                "similarity_squared_hex": similarity_squared.hex(),
                "action_outcome_weighted_mass": mass,
                "action_outcome_weighted_mass_hex": mass.hex(),
            }
        )
    candidate_predictions: list[dict[str, object]] = []
    positive_mass_by_action: dict[str, float] = {}
    for action_id in action_surface:
        counts = {outcome_id: _PRIOR_COUNT for outcome_id in outcome_surface}
        for item, _similarity, weight, _readout in weighted:
            if item["action_id"] == action_id:
                counts[_text(item["observed_outcome_id"], "weighted outcome ID")] += _EVIDENCE_WEIGHT * weight
        total = math.fsum(counts.values())
        probabilities = [
            {"outcome_id": outcome_id, "probability": counts[outcome_id] / total} for outcome_id in outcome_surface
        ]
        candidate_predictions.append({"action_id": action_id, "outcomes": probabilities})
        positive_mass_by_action[action_id] = math.fsum(
            _number(item["probability"], "candidate probability")
            for item in probabilities
            if item["outcome_id"] in _POSITIVE_OUTCOME_IDS
        )
    ranked = sorted(
        action_surface,
        key=lambda action_id: (-positive_mass_by_action[action_id], action_surface.index(action_id)),
    )
    margin = positive_mass_by_action[ranked[0]] - positive_mass_by_action[ranked[1]]
    support = max(weight for _item, _similarity, weight, _readout in weighted)
    confidence = max(0.0, min(1.0, 0.5 + 0.5 * support * margin))
    source_record_ids = [
        _text(item.get("evidence_id"), "sorted evidence ID")
        for item, _similarity, _weight, _readout in sorted(
            weighted,
            key=lambda row: (-row[2], _integer(row[0].get("source_turn"), "source turn")),
        )
    ]
    return (
        {
            "candidate_predictions": candidate_predictions,
            "recommended_action_id": ranked[0],
            "confidence": confidence,
            "source_record_ids": source_record_ids,
            "current_condition_readout": current_readout,
        },
        contributions,
    )


def recompute_proposal_from_session(
    *,
    session: Mapping[str, object],
    outcome_ids: Sequence[str],
    records: Sequence[Mapping[str, object]],
    outcomes: Sequence[Mapping[str, object]],
    vectors: Mapping[str, tuple[float, ...]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Select only runtime-consumed session fields; gap metadata is excluded."""

    return recompute_forecast_proposal(
        current_observation=_text(session.get("current_input"), "session current_input"),
        candidate_action_ids=_array(session.get("candidate_action_ids"), "session candidate_action_ids"),
        outcome_ids=outcome_ids,
        records=records,
        outcomes=outcomes,
        vectors=vectors,
    )


def _validate_attempt_authority(
    reader: _AttemptReader,
) -> tuple[dict[str, object], dict[str, object], dict[str, tuple[int, int]], dict[str, object]]:
    protocol = reader.load("protocol.json", verify_artifact_id=False)
    report = reader.load("report.json")
    _require_equal(reader.entries["protocol.json"]["sha256"], _EXPECTED_PROTOCOL_RAW_SHA256, "protocol raw SHA")
    _require_equal(reader.entries["report.json"]["sha256"], _EXPECTED_REPORT_RAW_SHA256, "report raw SHA")
    _require_equal(_sha256_json(protocol), _EXPECTED_PROTOCOL_ID, "protocol canonical identity")
    _require_equal(protocol.get("schema_version"), "relationship-product-horizon-campaign.v2", "protocol schema")
    _require_equal(report.get("artifact_id"), _EXPECTED_REPORT_ARTIFACT_ID, "report artifact ID")
    _require_equal(report.get("protocol_id"), _EXPECTED_PROTOCOL_ID, "report protocol ID")
    _require_equal(report.get("verdict"), _EXPECTED_VERDICT, "attempt03 frozen verdict")
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
    source = _object(protocol.get("source"), "protocol.source")
    _require_equal(source.get("source_protocol_id"), _EXPECTED_SOURCE_PROTOCOL_ID, "source protocol ID")
    _require_equal(source.get("public_plan_sha256"), _EXPECTED_SOURCE_PUBLIC_PLAN_ID, "source public ID")
    _require_equal(
        source.get("sealed_evaluator_bundle_sha256"),
        _EXPECTED_SOURCE_SEALED_BUNDLE_ID,
        "source sealed ID",
    )
    _require_equal(source.get("subject_count"), 8, "source subject count")
    _require_equal(source.get("decision_sessions_per_subject"), 24, "source decisions per subject")
    analysis = _object(protocol.get("analysis"), "protocol.analysis")
    _require_equal(analysis.get("primary_window_decision_indices"), [12, 23], "primary window")
    windows: dict[str, tuple[int, int]] = {}
    covered: list[int] = []
    for name, value in _object(analysis.get("horizon_segment_windows"), "segment windows").items():
        bounds = _array(value, f"segment {name}")
        if len(bounds) != 2:
            _fail(f"segment {name} must contain two inclusive bounds")
        start = _integer(bounds[0], f"segment {name} start")
        end = _integer(bounds[1], f"segment {name} end")
        if start > end:
            _fail(f"segment {name} bounds descend")
        windows[name] = (start, end)
        covered.extend(range(start, end + 1))
    _require_equal(sorted(covered), list(range(12, 24)), "horizon segment partition")
    public_path = "source/public/public_plan.json"
    sealed_path = "source/sealed/evaluator_bundle.json"
    public = reader.load(public_path, verify_artifact_id=False)
    sealed = reader.load(sealed_path, verify_artifact_id=False)
    _require_equal(reader.entries[public_path]["sha256"], _EXPECTED_SOURCE_PUBLIC_RAW_SHA256, "public raw SHA")
    _require_equal(reader.entries[sealed_path]["sha256"], _EXPECTED_SOURCE_SEALED_RAW_SHA256, "sealed raw SHA")
    _require_equal(_sha256_json(public), _EXPECTED_SOURCE_PUBLIC_PLAN_ID, "public canonical ID")
    unsigned_sealed = dict(sealed)
    _require_equal(
        unsigned_sealed.pop("sealed_bundle_sha256", None), _EXPECTED_SOURCE_SEALED_BUNDLE_ID, "sealed declared ID"
    )
    _require_equal(_sha256_json(unsigned_sealed), _EXPECTED_SOURCE_SEALED_BUNDLE_ID, "sealed canonical ID")
    _require_equal(sealed.get("evaluation_or_judge_feedback_to_learning"), False, "sealed learning firewall")
    _require_equal(report.get("source_protocol_id"), _EXPECTED_SOURCE_PROTOCOL_ID, "report source protocol")
    _require_equal(report.get("public_plan_sha256"), _EXPECTED_SOURCE_PUBLIC_PLAN_ID, "report public ID")
    _require_equal(report.get("sealed_bundle_sha256"), _EXPECTED_SOURCE_SEALED_BUNDLE_ID, "report sealed ID")
    provenance = {
        "source_protocol_id": _EXPECTED_SOURCE_PROTOCOL_ID,
        "public_plan_id": _EXPECTED_SOURCE_PUBLIC_PLAN_ID,
        "sealed_bundle_id": _EXPECTED_SOURCE_SEALED_BUNDLE_ID,
        "public_plan": reader.reference(public_path, public),
        "sealed_bundle": reader.reference(sealed_path, sealed),
    }
    return protocol, report, windows, provenance


def _segment_for_index(index: int, windows: Mapping[str, tuple[int, int]]) -> str | None:
    matches = [name for name, (start, end) in windows.items() if start <= index <= end]
    if len(matches) > 1:
        _fail(f"decision index {index} belongs to multiple segments")
    return matches[0] if matches else None


def _validate_record(record: Mapping[str, object], *, label: str) -> None:
    _require_exact_keys(
        record,
        {
            "confidence",
            "detail",
            "evidence",
            "interlocutor_id",
            "kind",
            "prediction_error_refs",
            "record_id",
            "source_turn",
            "status",
            "summary",
        },
        label,
    )
    confidence = _number(record.get("confidence"), f"{label}.confidence")
    if not 0.0 <= confidence <= 1.0:
        _fail(f"{label}.confidence must be in [0,1]")
    for field in ("detail", "evidence", "interlocutor_id", "kind", "record_id", "status", "summary"):
        _text(record.get(field), f"{label}.{field}")
    _integer(record.get("source_turn"), f"{label}.source_turn")
    refs = _array(record.get("prediction_error_refs"), f"{label}.prediction_error_refs")
    for index, value in enumerate(refs):
        _text(value, f"{label}.prediction_error_refs[{index}]")


def _validate_outcome(outcome: Mapping[str, object], *, label: str) -> None:
    _require_exact_keys(
        outcome,
        {
            "action_id",
            "evidence_id",
            "evidence_refs",
            "interlocutor_id",
            "observation_summary",
            "observed_outcome_id",
            "reaction_summary",
            "source_turn",
        },
        label,
    )
    for field in (
        "action_id",
        "evidence_id",
        "interlocutor_id",
        "observation_summary",
        "observed_outcome_id",
        "reaction_summary",
    ):
        _text(outcome.get(field), f"{label}.{field}")
    _integer(outcome.get("source_turn"), f"{label}.source_turn")
    refs = _array(outcome.get("evidence_refs"), f"{label}.evidence_refs")
    for index, value in enumerate(refs):
        _text(value, f"{label}.evidence_refs[{index}]")


def _owner_state_summary(
    *,
    reader: _AttemptReader,
    path: str,
    expected_version: int,
) -> dict[str, object]:
    filename = PurePosixPath(path).name
    match = _OWNER_VERSION_RE.fullmatch(filename)
    if match is None:
        _fail(f"owner hydration filename is not canonical: {filename}")
    _require_equal(int(match.group(1)), expected_version, f"{path} persistence version")
    document = reader.load(path, verify_artifact_id=False)
    _require_exact_keys(
        document,
        {"description", "owner_name", "payload", "schema_version"},
        f"owner hydration {path}",
    )
    _text(document.get("description"), f"{path}.description")
    _require_equal(document.get("owner_name"), "social_record_store", f"{path}.owner_name")
    _require_equal(document.get("schema_version"), 4, f"{path}.schema_version")
    payload = _object(document.get("payload"), f"{path}.payload")
    _require_exact_keys(
        payload,
        {
            "common_ground",
            "group_durability",
            "group_regimes",
            "preference_action_forecasts",
            "preference_action_outcome_mutation_receipts",
            "preference_action_outcomes",
            "preference_forecast_settlements",
            "tom_records",
        },
        f"{path}.payload",
    )
    for field in ("common_ground", "group_durability", "group_regimes", "tom_records"):
        _object(payload.get(field), f"{path}.payload.{field}")
    for field in (
        "preference_action_forecasts",
        "preference_action_outcome_mutation_receipts",
        "preference_action_outcomes",
        "preference_forecast_settlements",
    ):
        _array(payload.get(field), f"{path}.payload.{field}")
    tom_records = _object(payload["tom_records"], f"{path}.payload.tom_records")
    records = tuple(
        _object(value, f"{path} preference record[{index}]")
        for index, value in enumerate(_array(tom_records.get("preference_about_other"), f"{path} preference records"))
    )
    outcomes = tuple(
        _object(value, f"{path} preference outcome[{index}]")
        for index, value in enumerate(_array(payload["preference_action_outcomes"], f"{path} preference outcomes"))
    )
    if len(records) > _RECORD_WINDOW or len(outcomes) > _RECORD_WINDOW:
        _fail(f"{path} exceeds the frozen {_RECORD_WINDOW}-record window")
    for index, record in enumerate(records):
        _validate_record(record, label=f"{path} records[{index}]")
    for index, outcome in enumerate(outcomes):
        _validate_outcome(outcome, label=f"{path} outcomes[{index}]")
    require_active_outcome_pairing(records, outcomes)
    record_ids = tuple(_text(item["record_id"], "owner record ID") for item in records)
    outcome_ids = tuple(_text(item["evidence_id"], "owner outcome ID") for item in outcomes)
    reference = reader.reference(path, document)
    reference.update(
        {
            "persistence_version": expected_version,
            "owner_name": "social_record_store",
            "semantic_owner": "PreferenceAboutOtherModule/preference_about_other",
            "carrier_only": True,
            "payload_sha256": _sha256_json(payload),
        }
    )
    return {
        "path": path,
        "version": expected_version,
        "document": document,
        "payload": payload,
        "payload_sha256": _sha256_json(payload),
        "records": records,
        "outcomes": outcomes,
        "record_ids": record_ids,
        "outcome_ids": outcome_ids,
        "reference": reference,
    }


def _record_projection(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "record_id": record["record_id"],
        "status": record["status"],
        "confidence": _float_payload(record["confidence"], "record confidence"),
        "prediction_error_refs": deepcopy(record["prediction_error_refs"]),
        "source_turn": record["source_turn"],
        "kind": record["kind"],
        "interlocutor_id": record["interlocutor_id"],
        "evidence": record["evidence"],
        "summary_sha256": _sha256_text(_text(record["summary"], "record summary")),
        "detail_sha256": _sha256_text(_text(record["detail"], "record detail")),
        "canonical_value_sha256": _sha256_json(record),
    }


def _outcome_projection(outcome: Mapping[str, object]) -> dict[str, object]:
    return {
        "evidence_id": outcome["evidence_id"],
        "action_id": outcome["action_id"],
        "observed_outcome_id": outcome["observed_outcome_id"],
        "source_turn": outcome["source_turn"],
        "interlocutor_id": outcome["interlocutor_id"],
        "evidence_refs": deepcopy(outcome["evidence_refs"]),
        "observation_summary_sha256": _sha256_text(
            _text(outcome["observation_summary"], "outcome observation summary")
        ),
        "reaction_summary_sha256": _sha256_text(_text(outcome["reaction_summary"], "outcome reaction summary")),
        "canonical_value_sha256": _sha256_json(outcome),
    }


def _public_owner_state(state: Mapping[str, object]) -> dict[str, object]:
    records = state["records"]
    outcomes = state["outcomes"]
    assert isinstance(records, tuple) and isinstance(outcomes, tuple)
    payload = _object(state["payload"], "owner state payload")
    return {
        "hydration": deepcopy(state["reference"]),
        "hard_window_size": _RECORD_WINDOW,
        "ordered_record_ids": list(state["record_ids"]),
        "ordered_outcome_evidence_ids": list(state["outcome_ids"]),
        "record_count": len(records),
        "outcome_count": len(outcomes),
        "records": [_record_projection(record) for record in records],
        "outcomes": [_outcome_projection(outcome) for outcome in outcomes],
        "active_record_outcome_pairing_exact": True,
        "component_sha256": {
            field: _sha256_json(payload[field])
            for field in (
                "common_ground",
                "group_durability",
                "group_regimes",
                "preference_action_forecasts",
                "preference_action_outcome_mutation_receipts",
                "preference_action_outcomes",
                "preference_forecast_settlements",
                "tom_records",
            )
        },
        "preference_action_forecast_count": len(_array(payload["preference_action_forecasts"], "owner forecasts")),
        "preference_forecast_settlement_count": len(
            _array(payload["preference_forecast_settlements"], "owner settlements")
        ),
        "mutation_receipt_count": len(
            _array(payload["preference_action_outcome_mutation_receipts"], "owner mutations")
        ),
    }


def _load_chain_summaries(
    reader: _AttemptReader,
    *,
    subject_scope: str,
    arm_id: str,
) -> tuple[dict[str, object], dict[tuple[str, str], dict[str, object]]]:
    path = f"chains/{subject_scope}/{arm_id}/chain.json"
    chain = reader.load(path)
    _require_equal(chain.get("subject_scope"), subject_scope, f"{path} subject_scope")
    _require_equal(chain.get("arm_id"), arm_id, f"{path} arm_id")
    world_clone_id = _sha256(chain.get("world_clone_id"), f"{path} world_clone_id")
    result: dict[tuple[str, str], dict[str, object]] = {}
    decisions = _array(chain.get("decisions"), f"{path}.decisions")
    _require_equal(len(decisions), 24, f"{path} decision count")
    for expected_index, value in enumerate(decisions):
        record = _object(value, f"{path}.decisions[{expected_index}]")
        index = _integer(record.get("decision_index"), f"{path} decision index")
        _require_equal(index, expected_index, f"{path} decision order")
        decision_id = _text(record.get("decision_id"), f"{path} decision ID")
        selected_action = _text(record.get("selected_action_id"), f"{path} selected action")
        typed_outcome = _text(record.get("typed_outcome_id"), f"{path} typed outcome")
        if typed_outcome not in _POSITIVE_OUTCOME_IDS | _NEGATIVE_OUTCOME_IDS:
            _fail(f"{path} decision {index} has unknown typed outcome {typed_outcome!r}")
        positive = typed_outcome in _POSITIVE_OUTCOME_IDS
        _require_equal(record.get("positive_outcome"), positive, f"{path} positive outcome")
        _require_equal(record.get("world_clone_id"), world_clone_id, f"{path} decision world")
        key = (subject_scope, decision_id)
        if key in result:
            _fail(f"duplicate chain join key: {key!r}")
        result[key] = {
            "subject_scope": subject_scope,
            "arm_id": arm_id,
            "world_clone_id": world_clone_id,
            "decision_id": decision_id,
            "decision_index": index,
            "selected_action_id": selected_action,
            "typed_outcome_id": typed_outcome,
            "positive_outcome": positive,
            "preferred_action_match": record.get("preferred_action_match"),
            "chain_record": record,
            "chain_path": path,
        }
    return chain, result


def _validate_chain_link(
    *,
    reader: _AttemptReader,
    chain_record: Mapping[str, object],
    field_prefix: str,
    expected_path: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    _require_equal(
        chain_record.get(f"{field_prefix}_path"),
        expected_path,
        f"chain {field_prefix}_path",
    )
    reference = reader.reference(expected_path, payload)
    _require_equal(
        chain_record.get(f"{field_prefix}_sha256"),
        reference["raw_sha256"],
        f"chain {field_prefix}_sha256",
    )
    artifact_field = f"{field_prefix}_artifact_id"
    if artifact_field in chain_record:
        _require_equal(
            chain_record.get(artifact_field),
            reference.get("artifact_id"),
            f"chain {artifact_field}",
        )
    return reference


def _source_lineage_projection(
    *,
    reader: _AttemptReader,
    pre: Mapping[str, object],
    post: Mapping[str, object],
) -> dict[str, object]:
    pre_lineage = _object(pre.get("execution_source_lineage"), "preaction source lineage")
    post_lineage = _object(post.get("execution_source_lineage"), "postaction source lineage")
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
        _require_equal(post_lineage.get(field), pre_lineage.get(field), f"source lineage {field}")
    critical = {
        _text(_object(value, "critical source row").get("repository_path"), "critical source path"): _sha256(
            _object(value, "critical source row").get("raw_sha256"), "critical source SHA"
        )
        for value in _array(pre_lineage.get("critical_module_origins"), "critical module origins")
    }
    for key in ("record_store", "relationship_forecast", "condition_reader", "product_horizon"):
        path = _SOURCE_PATHS[key]
        repository_path = path.removeprefix("inputs/execution_sources/tree/")
        _require_equal(
            critical.get(repository_path),
            reader.entries[path]["sha256"],
            f"critical source lineage {repository_path}",
        )
    return {field: deepcopy(pre_lineage.get(field)) for field in fields}


def _owner_transition(
    *,
    pre_state: Mapping[str, object],
    post_state: Mapping[str, object],
    current_evidence_id: str,
) -> dict[str, object]:
    pre_record_ids = tuple(pre_state["record_ids"])
    post_record_ids = tuple(post_state["record_ids"])
    pre_outcome_ids = tuple(pre_state["outcome_ids"])
    post_outcome_ids = tuple(post_state["outcome_ids"])
    expected_record_ids = hard_window((*pre_record_ids, current_evidence_id))
    expected_outcome_ids = hard_window((*pre_outcome_ids, current_evidence_id))
    _require_equal(post_record_ids, expected_record_ids, "post record IDs from source-order hard window")
    _require_equal(post_outcome_ids, expected_outcome_ids, "post outcome IDs from source-order hard window")
    evicted_records = list(pre_record_ids[: max(0, len(pre_record_ids) + 1 - _RECORD_WINDOW)])
    evicted_outcomes = list(pre_outcome_ids[: max(0, len(pre_outcome_ids) + 1 - _RECORD_WINDOW)])
    _require_equal(
        [value for value in pre_record_ids if value not in set(post_record_ids)],
        evicted_records,
        "observed record eviction",
    )
    _require_equal(
        [value for value in pre_outcome_ids if value not in set(post_outcome_ids)],
        evicted_outcomes,
        "observed outcome eviction",
    )
    pre_records = {_text(item["record_id"], "pre record ID"): item for item in pre_state["records"]}
    post_records = {_text(item["record_id"], "post record ID"): item for item in post_state["records"]}
    pre_outcomes = {_text(item["evidence_id"], "pre outcome ID"): item for item in pre_state["outcomes"]}
    post_outcomes = {_text(item["evidence_id"], "post outcome ID"): item for item in post_state["outcomes"]}
    record_mutations = []
    for record_id in post_record_ids:
        if record_id in pre_records and pre_records[record_id] != post_records[record_id]:
            changed = sorted(
                field
                for field in set(pre_records[record_id]) | set(post_records[record_id])
                if pre_records[record_id].get(field) != post_records[record_id].get(field)
            )
            record_mutations.append({"record_id": record_id, "changed_fields": changed})
    outcome_mutations = []
    for evidence_id in post_outcome_ids:
        if evidence_id in pre_outcomes and pre_outcomes[evidence_id] != post_outcomes[evidence_id]:
            changed = sorted(
                field
                for field in set(pre_outcomes[evidence_id]) | set(post_outcomes[evidence_id])
                if pre_outcomes[evidence_id].get(field) != post_outcomes[evidence_id].get(field)
            )
            outcome_mutations.append({"evidence_id": evidence_id, "changed_fields": changed})
    return {
        "hard_window_size": _RECORD_WINDOW,
        "source_order_is_only_eviction_order": True,
        "pre_record_count": len(pre_record_ids),
        "post_record_count": len(post_record_ids),
        "appended_record_id": current_evidence_id,
        "evicted_record_ids": evicted_records,
        "evicted_outcome_evidence_ids": evicted_outcomes,
        "expected_post_record_ids": list(expected_record_ids),
        "expected_post_outcome_evidence_ids": list(expected_outcome_ids),
        "observed_post_matches_hard_window": True,
        "retained_record_mutations": record_mutations,
        "retained_outcome_mutations": outcome_mutations,
    }


def _recompute_and_verify_forecast(
    *,
    request: Mapping[str, object],
    pre: Mapping[str, object],
    pre_state: Mapping[str, object],
    vectors: Mapping[str, tuple[float, ...]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    session = _object(request.get("session"), "decision request session")
    frozen = _object(pre.get("frozen_forecast"), "preaction frozen_forecast")
    _require_artifact_identity(frozen, label="preaction frozen forecast")
    observed = _object(frozen.get("forecast"), "frozen forecast payload")
    observed_candidates = _array(observed.get("candidate_predictions"), "forecast candidates")
    expected_actions = tuple(
        _text(_object(value, "forecast candidate").get("action_id"), "forecast action ID")
        for value in observed_candidates
    )
    request_actions = tuple(
        _text(value, "request candidate action")
        for value in _array(session.get("candidate_action_ids"), "request candidate actions")
    )
    _require_equal(expected_actions, request_actions, "request/forecast ordered action surface")
    outcome_vocabularies: list[tuple[str, ...]] = []
    for value in observed_candidates:
        candidate = _object(value, "forecast candidate")
        outcome_vocabularies.append(
            tuple(
                _text(_object(item, "forecast outcome").get("outcome_id"), "forecast outcome ID")
                for item in _array(candidate.get("outcomes"), "forecast candidate outcomes")
            )
        )
    if not outcome_vocabularies or len(set(outcome_vocabularies)) != 1:
        _fail("forecast candidates do not share one ordered outcome surface")
    outcome_ids = outcome_vocabularies[0]
    proposal, contributions = recompute_proposal_from_session(
        session=session,
        outcome_ids=outcome_ids,
        records=pre_state["records"],
        outcomes=pre_state["outcomes"],
        vectors=vectors,
    )
    decision_id = _text(session.get("decision_id"), "session decision ID")
    decision_index = _integer(session.get("decision_index"), "session decision index")
    issued_turn = 4 + decision_index * 2
    subject_scope = _text(request.get("subject_scope"), "request subject scope")
    observation_ref = f"public-decision:{_sha256_json(session)}"
    expected_forecast = {
        "forecast_id": f"preference_about_other:{decision_id}:forecast:{issued_turn}",
        "decision_id": decision_id,
        "interlocutor_id": _INTERLOCUTOR_ID,
        "candidate_predictions": proposal["candidate_predictions"],
        "recommended_action_id": proposal["recommended_action_id"],
        "confidence": proposal["confidence"],
        "source_record_ids": proposal["source_record_ids"],
        "issued_turn": issued_turn,
        "evidence": [
            f"typed_observation:{observation_ref}",
            "runtime:relationship-p2-bounded-forecast.v1",
            f"typed_owner_evidence_count:{len(pre_state['outcomes'])}",
            "semantic_similarity_only:no_text_routing",
            f"condition_reader:{_EXPECTED_READER_ARTIFACT_ID}",
            f"condition_label:{proposal['current_condition_readout']['condition_label']}",
        ],
        "session_scope": subject_scope,
        "condition_readout": proposal["current_condition_readout"],
    }
    expected_raw = _canonical_json_bytes(expected_forecast)
    observed_raw = _canonical_json_bytes(observed)
    _require_equal(observed_raw, expected_raw, "recomputed frozen forecast canonical payload bytes")
    _require_equal(pre.get("forecast_sha256"), _sha256_bytes(expected_raw), "receipt forecast SHA")
    _require_equal(pre.get("forecast_id"), expected_forecast["forecast_id"], "receipt forecast ID")
    _require_equal(
        pre.get("recommended_action_id"),
        expected_forecast["recommended_action_id"],
        "receipt recommendation",
    )
    expected_envelope_id = _sha256_json(
        {"schema_version": "preference-action-forecast-snapshot.v1", "forecast": expected_forecast}
    )
    _require_equal(frozen.get("artifact_id"), expected_envelope_id, "frozen forecast envelope ID")
    _require_equal(pre.get("semantic_table_artifact_id"), _EXPECTED_EMBEDDING_TABLE_ID, "receipt table ID")
    _require_equal(
        pre.get("semantic_similarity_formula"),
        "prototype_cosine_named_condition_then_same_label_confidence",
        "receipt semantic formula",
    )
    _require_equal(pre.get("model_output_count"), 0, "receipt model output count")
    return expected_forecast, contributions


def _detailed_arm_decision(
    *,
    reader: _AttemptReader,
    summary: Mapping[str, object],
    vectors: Mapping[str, tuple[float, ...]],
) -> dict[str, object]:
    subject_scope = _text(summary.get("subject_scope"), "summary subject scope")
    arm_id = _text(summary.get("arm_id"), "summary arm ID")
    if arm_id not in _ARMS:
        _fail(f"unsupported owner-history arm: {arm_id}")
    decision_id = _text(summary.get("decision_id"), "summary decision ID")
    decision_index = _integer(summary.get("decision_index"), "summary decision index")
    arm_prefix = f"chains/{subject_scope}/{arm_id}"
    stem = f"decision-{decision_index:02d}"
    request_path = f"{arm_prefix}/requests/{stem}.json"
    pre_path = f"{arm_prefix}/receipts/{stem}.preaction.json"
    post_path = f"{arm_prefix}/receipts/{stem}.postaction.json"
    sealed_path = f"{arm_prefix}/sealed/{stem}.json"
    request = reader.load(request_path)
    pre = reader.load(pre_path)
    post = reader.load(post_path)
    sealed = reader.load(sealed_path)
    chain_record = _object(summary.get("chain_record"), "chain decision record")
    documents = {
        "chain": reader.reference(_text(summary.get("chain_path"), "chain path")),
        "request": _validate_chain_link(
            reader=reader,
            chain_record=chain_record,
            field_prefix="request",
            expected_path=request_path,
            payload=request,
        ),
        "preaction": _validate_chain_link(
            reader=reader,
            chain_record=chain_record,
            field_prefix="preaction_receipt",
            expected_path=pre_path,
            payload=pre,
        ),
        "postaction": _validate_chain_link(
            reader=reader,
            chain_record=chain_record,
            field_prefix="postaction_receipt",
            expected_path=post_path,
            payload=post,
        ),
        "sealed": _validate_chain_link(
            reader=reader,
            chain_record=chain_record,
            field_prefix="sealed_record",
            expected_path=sealed_path,
            payload=sealed,
        ),
    }
    _require_equal(request.get("subject_scope"), subject_scope, f"{request_path} subject")
    _require_equal(request.get("arm_id"), arm_id, f"{request_path} arm")
    session = _object(request.get("session"), f"{request_path}.session")
    _require_equal(session.get("decision_id"), decision_id, f"{request_path} decision ID")
    _require_equal(session.get("decision_index"), decision_index, f"{request_path} decision index")
    _require_equal(pre.get("request_artifact_id"), request.get("artifact_id"), f"{pre_path} request")
    _require_equal(post.get("request_artifact_id"), request.get("artifact_id"), f"{post_path} request")
    _require_equal(post.get("preaction_artifact_id"), pre.get("artifact_id"), f"{post_path} preaction")
    _require_equal(sealed.get("decision_id"), decision_id, f"{sealed_path} decision ID")
    _require_equal(sealed.get("decision_index"), decision_index, f"{sealed_path} decision index")
    _require_equal(sealed.get("world_clone_id"), summary.get("world_clone_id"), f"{sealed_path} world")
    state_root = _normalized_member_path(request.get("state_root"), label=f"{request_path} state_root")
    if arm_id == _FULL_ARM:
        expected_state_root = f"{arm_prefix}/state/owner"
        pre_version = 4 + decision_index
        post_version = 5 + decision_index
    else:
        expected_state_root = f"{arm_prefix}/frozen_owner_sessions/{stem}"
        pre_version = 4
        post_version = 5
    _require_equal(state_root, expected_state_root, f"{request_path} state root")
    pre_state_path = f"{state_root}/owner_hydration__social_record_store_v{pre_version}.json"
    post_state_path = f"{state_root}/owner_hydration__social_record_store_v{post_version}.json"
    pre_state = _owner_state_summary(
        reader=reader,
        path=pre_state_path,
        expected_version=pre_version,
    )
    post_state = _owner_state_summary(
        reader=reader,
        path=post_state_path,
        expected_version=post_version,
    )
    _require_equal(pre.get("owner_loaded"), True, f"{pre_path} owner loaded")
    _require_equal(
        pre.get("pre_owner_snapshot_sha256"),
        pre_state["payload_sha256"],
        f"{pre_path} pre owner payload hash",
    )
    _require_equal(
        post.get("post_owner_snapshot_sha256"),
        post_state["payload_sha256"],
        f"{post_path} post owner payload hash",
    )
    selected_action = _text(summary.get("selected_action_id"), "summary selected action")
    typed_outcome = _text(summary.get("typed_outcome_id"), "summary typed outcome")
    _require_equal(pre.get("selected_action_id"), selected_action, f"{pre_path} selected action")
    gate = _object(pre.get("gate_decision"), f"{pre_path} gate decision")
    _require_equal(gate.get("selected_action_id"), selected_action, f"{pre_path} gate selected action")
    _require_equal(post.get("typed_outcome_id"), typed_outcome, f"{post_path} typed outcome")
    _require_equal(sealed.get("selected_action_id"), selected_action, f"{sealed_path} selected action")
    _require_equal(sealed.get("typed_outcome_id"), typed_outcome, f"{sealed_path} typed outcome")
    current_evidence_id = f"relationship-product-outcome:{decision_id}"
    transition = _owner_transition(
        pre_state=pre_state,
        post_state=post_state,
        current_evidence_id=current_evidence_id,
    )
    current_records = [item for item in post_state["records"] if item.get("record_id") == current_evidence_id]
    current_outcomes = [item for item in post_state["outcomes"] if item.get("evidence_id") == current_evidence_id]
    _require_equal(len(current_records), 1, "post owner current record cardinality")
    _require_equal(len(current_outcomes), 1, "post owner current outcome cardinality")
    _require_equal(current_outcomes[0].get("action_id"), selected_action, "current persisted action")
    _require_equal(current_outcomes[0].get("observed_outcome_id"), typed_outcome, "current persisted outcome")
    forecast, contributions = _recompute_and_verify_forecast(
        request=request,
        pre=pre,
        pre_state=pre_state,
        vectors=vectors,
    )
    _require_equal(gate.get("recommended_action_id"), forecast["recommended_action_id"], "gate recommendation")
    _require_equal(pre.get("recommended_action_id"), forecast["recommended_action_id"], "preaction recommendation")
    _require_equal(post.get("forecast_id"), forecast["forecast_id"], "postaction forecast ID")
    _require_equal(post.get("evaluator_or_judge_feedback_received"), False, "postaction evaluator firewall")
    _sha256(post.get("social_prediction_error_snapshot_sha256"), "postaction PE snapshot SHA")
    source_lineage = _source_lineage_projection(reader=reader, pre=pre, post=post)
    condition_id = _text(sealed.get("condition_id"), f"{sealed_path} condition ID")
    if condition_id not in _CONDITION_TO_READER_LABEL:
        _fail(f"unknown sealed condition ID: {condition_id}")
    readout = _object(forecast.get("condition_readout"), "verified condition readout")
    observed_reader_label = _text(readout.get("condition_label"), "observed reader label")
    current_observation = _text(session.get("current_input"), "current observation")
    gate_probability = _number(gate.get("steer_probability"), "gate steer probability")
    return {
        "unit": {
            "world_clone_id": summary["world_clone_id"],
            "arm_id": arm_id,
            "decision_index": decision_index,
        },
        "record_identity": {
            "arm_id": arm_id,
            "subject_scope": subject_scope,
            "decision_id": decision_id,
        },
        "source_documents": documents,
        "execution_source_lineage": source_lineage,
        "owner_contract": {
            "unique_semantic_owner": "PreferenceAboutOtherModule/preference_about_other",
            "social_record_store_role": "carrier_only",
            "pre_owner_loaded": True,
        },
        "pre_owner": _public_owner_state(pre_state),
        "post_owner": _public_owner_state(post_state),
        "owner_transition": transition,
        "current_observation": {
            "text": current_observation,
            "sha256": _sha256_text(current_observation),
            "virtual_day": session.get("virtual_day"),
            "public_context_chunk_sha256": _sha256_text(
                _text(session.get("public_context_chunk"), "public context chunk")
            ),
        },
        "reader_truth_crosswalk": {
            "sealed_condition_id": condition_id,
            "expected_reader_label": _CONDITION_TO_READER_LABEL[condition_id],
            "observed_reader_label": observed_reader_label,
            "reader_truth_match": observed_reader_label == _CONDITION_TO_READER_LABEL[condition_id],
            "sealed_truth_used_by_recomputation": False,
        },
        "sealed_exogenous_truth": {
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
        },
        "named_readout": deepcopy(readout),
        "forecast": {
            "payload": forecast,
            "forecast_sha256": pre["forecast_sha256"],
            "frozen_envelope_artifact_id": _object(pre["frozen_forecast"], "frozen forecast")["artifact_id"],
            "canonical_payload_byte_exact_with_receipt": True,
            "numeric_values_bit_exact_with_receipt": True,
            "prior_count": _float_payload(_PRIOR_COUNT, "prior count"),
            "evidence_weight": _float_payload(_EVIDENCE_WEIGHT, "evidence weight"),
            "per_record_contributions": contributions,
        },
        "recommendation": forecast["recommended_action_id"],
        "gate": {
            "gate_action": gate.get("gate_action"),
            "selected_action_id": gate.get("selected_action_id"),
            "steer_probability": gate_probability,
            "steer_probability_hex": gate_probability.hex(),
            "threshold": 0.5,
            "threshold_distance": abs(gate_probability - 0.5),
            "threshold_distance_hex": abs(gate_probability - 0.5).hex(),
            "update_count_before": pre.get("gate_update_count_before"),
            "update_count_after": post.get("gate_update_count_after"),
        },
        "selected_action_id": selected_action,
        "typed_outcome_id": typed_outcome,
        "positive_outcome": typed_outcome in _POSITIVE_OUTCOME_IDS,
        "safety_negative_outcome": typed_outcome in _NEGATIVE_OUTCOME_IDS,
        "preferred_action_id": sealed.get("preferred_action_id"),
        "preferred_action_match": summary.get("preferred_action_match"),
        "postaction_mechanism_refs": {
            "credit_record_id": post.get("credit_record_id"),
            "credit_applied_to_gate": post.get("credit_applied_to_gate"),
            "social_prediction_error_snapshot_sha256": post.get("social_prediction_error_snapshot_sha256"),
            "settlement_id": post.get("settlement_id"),
            "settlement_payload_sha256": post.get("settlement_payload_sha256"),
        },
        "weight_contract": {
            "formula": "semantic_similarity ** 2",
            "confidence_used_by_forecast": False,
            "confidence_field_scope": "OtherMindRecord.confidence",
            "named_reader_confidence_used_by_similarity": True,
            "prediction_error_refs_used_by_forecast": False,
            "elapsed_time_used_by_forecast": False,
            "virtual_day_used_by_forecast": False,
            "public_context_chunk_used_by_forecast": False,
            "decay_used_by_forecast": False,
            "source_turn_used_as_continuous_weight": False,
            "source_turn_used_only_for_equal_weight_ordering": True,
        },
    }


def _probability_map(arm: Mapping[str, object]) -> dict[tuple[str, str], float]:
    forecast = _object(_object(arm.get("forecast"), "arm forecast").get("payload"), "forecast payload")
    result: dict[tuple[str, str], float] = {}
    for candidate_value in _array(forecast.get("candidate_predictions"), "candidate predictions"):
        candidate = _object(candidate_value, "candidate prediction")
        action_id = _text(candidate.get("action_id"), "candidate action ID")
        for outcome_value in _array(candidate.get("outcomes"), "candidate outcomes"):
            outcome = _object(outcome_value, "candidate outcome")
            key = (action_id, _text(outcome.get("outcome_id"), "candidate outcome ID"))
            if key in result:
                _fail(f"duplicate forecast probability cell: {key!r}")
            result[key] = _number(outcome.get("probability"), "forecast probability")
    return result


def _state_id_delta(full_ids: Sequence[object], frozen_ids: Sequence[object]) -> dict[str, object]:
    full = tuple(_text(value, "full state ID") for value in full_ids)
    frozen = tuple(_text(value, "frozen state ID") for value in frozen_ids)
    return {
        "ordered_equal": full == frozen,
        "full_only_ids": [value for value in full if value not in set(frozen)],
        "frozen_only_ids": [value for value in frozen if value not in set(full)],
        "ordered_full_ids": list(full),
        "ordered_frozen_ids": list(frozen),
    }


def _compare_arms(full: Mapping[str, object], frozen: Mapping[str, object]) -> dict[str, object]:
    full_probabilities = _probability_map(full)
    frozen_probabilities = _probability_map(frozen)
    _require_equal(sorted(full_probabilities), sorted(frozen_probabilities), "forecast probability surface")
    probability_deltas = []
    for action_id, outcome_id in sorted(full_probabilities):
        delta = full_probabilities[(action_id, outcome_id)] - frozen_probabilities[(action_id, outcome_id)]
        probability_deltas.append(
            {
                "action_id": action_id,
                "outcome_id": outcome_id,
                "full_minus_frozen": delta,
                "full_minus_frozen_hex": delta.hex(),
            }
        )
    full_pre = _object(full.get("pre_owner"), "full pre owner")
    frozen_pre = _object(frozen.get("pre_owner"), "frozen pre owner")
    full_post = _object(full.get("post_owner"), "full post owner")
    frozen_post = _object(frozen.get("post_owner"), "frozen post owner")
    full_gate = _object(full.get("gate"), "full gate")
    frozen_gate = _object(frozen.get("gate"), "frozen gate")
    recommendation_diff = full.get("recommendation") != frozen.get("recommendation")
    gate_action_diff = full_gate.get("gate_action") != frozen_gate.get("gate_action")
    if recommendation_diff and gate_action_diff:
        descriptive_pattern = "recommendation_different_gate_action_different"
    elif recommendation_diff:
        descriptive_pattern = "recommendation_different_gate_action_same"
    elif gate_action_diff:
        descriptive_pattern = "recommendation_same_gate_action_different"
    else:
        descriptive_pattern = "recommendation_same_gate_action_same"
    full_positive = _boolean(full.get("positive_outcome"), "full positive outcome")
    frozen_positive = _boolean(frozen.get("positive_outcome"), "frozen positive outcome")
    full_components = _object(full_pre.get("component_sha256"), "full pre component hashes")
    frozen_components = _object(frozen_pre.get("component_sha256"), "frozen pre component hashes")
    _require_equal(sorted(full_components), sorted(frozen_components), "owner component surface")
    return {
        "pre_owner_payload_equal": _object(full_pre["hydration"], "full hydration").get("payload_sha256")
        == _object(frozen_pre["hydration"], "frozen hydration").get("payload_sha256"),
        "post_owner_payload_equal": _object(full_post["hydration"], "full hydration").get("payload_sha256")
        == _object(frozen_post["hydration"], "frozen hydration").get("payload_sha256"),
        "pre_owner_component_discordance": {
            field: full_components[field] != frozen_components[field] for field in sorted(full_components)
        },
        "pre_record_state_delta": _state_id_delta(
            _array(full_pre.get("ordered_record_ids"), "full pre record IDs"),
            _array(frozen_pre.get("ordered_record_ids"), "frozen pre record IDs"),
        ),
        "pre_outcome_state_delta": _state_id_delta(
            _array(full_pre.get("ordered_outcome_evidence_ids"), "full pre outcome IDs"),
            _array(frozen_pre.get("ordered_outcome_evidence_ids"), "frozen pre outcome IDs"),
        ),
        "post_record_state_delta": _state_id_delta(
            _array(full_post.get("ordered_record_ids"), "full post record IDs"),
            _array(frozen_post.get("ordered_record_ids"), "frozen post record IDs"),
        ),
        "post_outcome_state_delta": _state_id_delta(
            _array(full_post.get("ordered_outcome_evidence_ids"), "full post outcome IDs"),
            _array(frozen_post.get("ordered_outcome_evidence_ids"), "frozen post outcome IDs"),
        ),
        "forecast_probability_deltas": probability_deltas,
        "recommendation_discordant": recommendation_diff,
        "gate_action_discordant": gate_action_diff,
        "selected_action_discordant": full.get("selected_action_id") != frozen.get("selected_action_id"),
        "typed_outcome_discordant": full.get("typed_outcome_id") != frozen.get("typed_outcome_id"),
        "positive_outcome_discordant": full_positive != frozen_positive,
        "positive_outcome_net_numerator": int(full_positive) - int(frozen_positive),
        "descriptive_mixed_path_pattern": descriptive_pattern,
        "causal_category_assigned": False,
    }


def _source_contract_references(reader: _AttemptReader) -> dict[str, object]:
    required_tokens = {
        "record_store": (
            "_RECORD_WINDOW = 12",
            "[-_RECORD_WINDOW:]",
            "bounded_records = records[-_RECORD_WINDOW:]",
            "bounded_outcomes = action_outcomes[-_RECORD_WINDOW:]",
        ),
        "tom": (
            "record.status is OtherMindRecordStatus.ACTIVE",
            "eligible_action_outcomes",
        ),
        "relationship_forecast": (
            "float(similarity) ** 2",
            "self._evidence_weight * weight",
            "math.fsum(counts.values())",
            "pair[0].source_turn",
        ),
        "condition_reader": (
            "return min(left_readout.confidence, right_readout.confidence)",
            "softmax_temperature",
            "_cosine(vector, prototype_vector)",
        ),
        "product_horizon": (
            "prototype_cosine_named_condition_then_same_label_confidence",
            "sha256_json(snapshot.payload)",
        ),
        "model_adapters": (
            "float.fromhex(encoded)",
            "value.hex() != encoded",
            "public embedding table must use exact canonical JSON bytes",
        ),
    }
    line_anchors = {
        "record_store": [77, 358, 464, 497, 561, 562, 563, 564],
        "tom": [
            655,
            656,
            657,
            658,
            659,
            660,
            661,
            662,
            663,
            664,
            665,
            666,
            667,
            668,
            669,
            670,
            671,
            672,
            673,
            674,
            675,
            676,
            677,
            678,
            679,
            680,
            681,
        ],
        "relationship_forecast": [53, 70, 79, 150],
        "condition_reader": [116, 238],
        "product_horizon": [5622, 5666, 5669, 5676],
        "model_adapters": [464, 608, 780, 800],
    }
    result: dict[str, object] = {}
    for key, path in _SOURCE_PATHS.items():
        raw = reader.raw(path)
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise AuditContractError(f"frozen source is not UTF-8: {path}") from exc
        for token in required_tokens[key]:
            if token not in text:
                _fail(f"frozen source contract token missing from {path}: {token!r}")
        reference = reader.reference(path)
        reference["relevant_line_anchors"] = line_anchors[key]
        result[key] = reference
    return result


def _world_primary_directions(
    full_records: Mapping[tuple[str, str], Mapping[str, object]],
    frozen_records: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    by_subject: dict[str, list[tuple[Mapping[str, object], Mapping[str, object]]]] = defaultdict(list)
    for key, full in full_records.items():
        frozen = frozen_records[key]
        index = _integer(full.get("decision_index"), "world direction decision index")
        if 12 <= index <= 23:
            by_subject[key[0]].append((full, frozen))
    result: dict[str, dict[str, object]] = {}
    for subject_scope in sorted(by_subject):
        pairs = by_subject[subject_scope]
        _require_equal(len(pairs), 12, f"{subject_scope} primary-window decision count")
        full_positive = sum(int(_boolean(full["positive_outcome"], "full positive")) for full, _ in pairs)
        frozen_positive = sum(int(_boolean(frozen["positive_outcome"], "frozen positive")) for _, frozen in pairs)
        net = full_positive - frozen_positive
        direction = "positive" if net > 0 else "negative" if net < 0 else "tie"
        worlds = {str(full["world_clone_id"]) for full, _ in pairs}
        _require_equal(len(worlds), 1, f"{subject_scope} world cardinality")
        result[subject_scope] = {
            "subject_scope": subject_scope,
            "world_clone_id": next(iter(worlds)),
            "matched_decision_count": 12,
            "full_positive_outcome_count": full_positive,
            "frozen_positive_outcome_count": frozen_positive,
            "full_minus_frozen_net_numerator": net,
            "direction": direction,
        }
    _require_equal(len(result), 8, "primary-window matched world count")
    return result


def _full_reader_confounding_diagnostic(
    reader: _AttemptReader,
    full_records: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[str, object]:
    observed_counts: Counter[str] = Counter()
    condition_counts: Counter[str] = Counter()
    correct_keys: set[tuple[str, str]] = set()
    agency_keys: set[tuple[str, str]] = set()
    for key, summary in full_records.items():
        subject_scope, _decision_id = key
        index = _integer(summary.get("decision_index"), "reader diagnostic decision index")
        prefix = f"chains/{subject_scope}/{_FULL_ARM}"
        pre = reader.load(f"{prefix}/receipts/decision-{index:02d}.preaction.json")
        sealed = reader.load(f"{prefix}/sealed/decision-{index:02d}.json")
        forecast = _object(
            _object(pre.get("frozen_forecast"), "reader diagnostic frozen forecast").get("forecast"),
            "reader diagnostic forecast",
        )
        readout = _object(forecast.get("condition_readout"), "reader diagnostic readout")
        observed = _text(readout.get("condition_label"), "reader diagnostic observed label")
        condition = _text(sealed.get("condition_id"), "reader diagnostic condition")
        expected = _CONDITION_TO_READER_LABEL.get(condition)
        if expected is None:
            _fail(f"reader diagnostic found unknown sealed condition: {condition}")
        observed_counts[observed] += 1
        condition_counts[condition] += 1
        if observed == expected:
            correct_keys.add(key)
        if condition == "agency_under_override":
            agency_keys.add(key)
    _require_equal(observed_counts, Counter({"agency_displacement": 192}), "full reader collapsed output")
    _require_equal(
        condition_counts,
        Counter({"agency_under_override": 96, "connection_under_exclusion": 96}),
        "full condition balance",
    )
    _require_equal(correct_keys, agency_keys, "reader correctness/agency condition partition")
    return {
        "decision_denominator": 192,
        "observed_reader_label_counts": dict(sorted(observed_counts.items())),
        "sealed_condition_counts": dict(sorted(condition_counts.items())),
        "correct_count": len(correct_keys),
        "incorrect_count": 192 - len(correct_keys),
        "correct_partition_equals_agency_condition_partition": True,
        "incorrect_partition_equals_connection_condition_partition": True,
        "perfect_condition_confounding": True,
        "reader_error_attribution_authorized": False,
        "explanation": (
            "The frozen reader emitted agency_displacement for every decision, so reader-correctness "
            "is exactly confounded with the sealed condition. Net outcome differences cannot be "
            "attributed to reader error from this partition."
        ),
    }


def _artifact_document(schema_version: str, **payload: object) -> tuple[dict[str, object], bytes]:
    document = _with_artifact_id({"schema_version": schema_version, **payload})
    return document, _canonical_json_bytes(document) + b"\n"


def build_audit_documents(attempt_root: Path) -> dict[str, bytes]:
    """Build every output in memory, failing before any write on drift."""

    reader = _AttemptReader(attempt_root)
    _protocol, frozen_report, windows, source_provenance = _validate_attempt_authority(reader)
    vectors, embedding_table, reader_artifact = _load_embedding_table(reader)
    source_contracts = _source_contract_references(reader)
    audit_script_path = Path(__file__).resolve()
    audit_script_raw = audit_script_path.read_bytes()
    audit_script_reference = {
        "path": _repo_relative(audit_script_path, label="audit script"),
        "bytes": len(audit_script_raw),
        "raw_sha256": _sha256_bytes(audit_script_raw),
        "standard_library_only": True,
    }
    subject_scopes = sorted(path.name for path in (reader.root / "chains").iterdir() if path.is_dir())
    _require_equal(len(subject_scopes), 8, "attempt subject scope count")
    for subject_scope in subject_scopes:
        _sha256(subject_scope, "subject scope")

    chains: dict[tuple[str, str], dict[str, object]] = {}
    records_by_arm: dict[str, dict[tuple[str, str], dict[str, object]]] = {arm_id: {} for arm_id in _ARMS}
    chain_references: list[dict[str, object]] = []
    for subject_scope in subject_scopes:
        expected_world: str | None = None
        for arm_id in _ARMS:
            chain, records = _load_chain_summaries(
                reader,
                subject_scope=subject_scope,
                arm_id=arm_id,
            )
            world = _sha256(chain.get("world_clone_id"), "chain world clone ID")
            if expected_world is None:
                expected_world = world
            _require_equal(world, expected_world, f"{subject_scope} cross-arm world")
            overlap = set(records_by_arm[arm_id]).intersection(records)
            if overlap:
                _fail(f"duplicate arm record keys: {sorted(overlap)[:3]!r}")
            records_by_arm[arm_id].update(records)
            chain_key = (subject_scope, arm_id)
            chains[chain_key] = chain
            chain_references.append(reader.reference(f"chains/{subject_scope}/{arm_id}/chain.json", chain))
    for arm_id in _ARMS:
        _require_equal(len(records_by_arm[arm_id]), 192, f"{arm_id} matched decision denominator")
    full_records = records_by_arm[_FULL_ARM]
    frozen_records = records_by_arm[_FROZEN_ARM]
    _require_equal(sorted(full_records), sorted(frozen_records), "cross-arm match-key set")
    world_directions = _world_primary_directions(full_records, frozen_records)

    divergent_keys: list[tuple[str, str]] = []
    for key in sorted(
        full_records,
        key=lambda item: (item[0], _integer(full_records[item]["decision_index"], "decision index")),
    ):
        full = full_records[key]
        frozen = frozen_records[key]
        _require_equal(full.get("world_clone_id"), frozen.get("world_clone_id"), f"{key} world")
        _require_equal(full.get("decision_index"), frozen.get("decision_index"), f"{key} decision index")
        if full.get("selected_action_id") != frozen.get("selected_action_id"):
            divergent_keys.append(key)
    _require_equal(len(divergent_keys), 36, "full-vs-frozen action-divergence count")

    rows: list[dict[str, object]] = []
    for key in divergent_keys:
        full_summary = full_records[key]
        frozen_summary = frozen_records[key]
        full = _detailed_arm_decision(reader=reader, summary=full_summary, vectors=vectors)
        frozen = _detailed_arm_decision(reader=reader, summary=frozen_summary, vectors=vectors)
        _require_equal(full.get("sealed_exogenous_truth"), frozen.get("sealed_exogenous_truth"), f"{key} truth")
        _require_equal(full.get("current_observation"), frozen.get("current_observation"), f"{key} public observation")
        _require_equal(full.get("named_readout"), frozen.get("named_readout"), f"{key} named readout")
        comparison = _compare_arms(full, frozen)
        _require_equal(comparison.get("selected_action_discordant"), True, f"{key} action divergence")
        index = _integer(full_summary.get("decision_index"), "row decision index")
        row = {
            "match": {
                "cross_arm_match_fields": ["subject_scope", "decision_id"],
                "subject_scope": key[0],
                "decision_id": key[1],
                "world_clone_id": full_summary["world_clone_id"],
                "decision_index": index,
                "world_clone_and_decision_index_validated": True,
            },
            "record_identity_fields": ["arm_id", "subject_scope", "decision_id"],
            "credit_record_id_is_cross_arm_join_key": False,
            "primary_window": 12 <= index <= 23,
            "horizon_segment": _segment_for_index(index, windows),
            "world_primary_window_direction": deepcopy(world_directions[key[0]]),
            "volvence_full": full,
            "appendable_frozen_onboarding": frozen,
            "full_minus_frozen": comparison,
            "diagnostic_scope": "post_hoc_owner_history_hard_window_contribution",
        }
        rows.append(_with_row_id(row))
    _require_equal(len(rows), 36, "materialized owner-history row count")

    pattern_rows: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "net": 0})
    component_counts: Counter[str] = Counter()
    full_near_threshold = 0
    frozen_near_threshold = 0
    used_text_digests: set[str] = set()
    eviction_counts: Counter[str] = Counter()
    for row in rows:
        comparison = _object(row.get("full_minus_frozen"), "row comparison")
        pattern = _text(comparison.get("descriptive_mixed_path_pattern"), "mixed path pattern")
        pattern_rows[pattern]["count"] += 1
        pattern_rows[pattern]["net"] += _integer(
            comparison.get("positive_outcome_net_numerator"),
            "row outcome net",
        )
        for component, discordant in _object(
            comparison.get("pre_owner_component_discordance"),
            "owner component discordance",
        ).items():
            component_counts[component] += int(_boolean(discordant, f"component {component}"))
        for arm_id in _ARMS:
            arm = _object(row.get(arm_id), f"row {arm_id}")
            gate = _object(arm.get("gate"), f"{arm_id} gate")
            distance = _number(gate.get("threshold_distance"), f"{arm_id} threshold distance")
            if arm_id == _FULL_ARM and distance < 0.05:
                full_near_threshold += 1
            if arm_id == _FROZEN_ARM and distance < 0.05:
                frozen_near_threshold += 1
            observation = _object(arm.get("current_observation"), f"{arm_id} observation")
            used_text_digests.add(_sha256(observation.get("sha256"), "observation digest"))
            forecast = _object(arm.get("forecast"), f"{arm_id} forecast")
            for contribution_value in _array(
                forecast.get("per_record_contributions"),
                f"{arm_id} contributions",
            ):
                contribution = _object(contribution_value, "forecast contribution")
                used_text_digests.add(
                    _sha256(
                        contribution.get("observation_summary_sha256"),
                        "contribution observation digest",
                    )
                )
            transition = _object(arm.get("owner_transition"), f"{arm_id} owner transition")
            if _array(transition.get("evicted_record_ids"), f"{arm_id} evicted IDs"):
                eviction_counts[arm_id] += 1
    expected_patterns = {
        "recommendation_same_gate_action_different": {"count": 8, "net": -6},
        "recommendation_different_gate_action_same": {"count": 17, "net": -2},
        "recommendation_different_gate_action_different": {"count": 11, "net": -4},
    }
    _require_equal(dict(pattern_rows), expected_patterns, "mixed recommendation/gate path cross-check")
    _require_equal(full_near_threshold, 29, "full near-threshold divergence count")
    expected_components = {
        "common_ground": 0,
        "group_durability": 0,
        "group_regimes": 0,
        "preference_action_forecasts": 0,
        "preference_action_outcome_mutation_receipts": 0,
        "preference_action_outcomes": 36,
        "preference_forecast_settlements": 36,
        "tom_records": 36,
    }
    _require_equal(dict(sorted(component_counts.items())), expected_components, "owner component delta counts")

    reader_diagnostic = _full_reader_confounding_diagnostic(reader, full_records)
    final_owner_states: list[dict[str, object]] = []
    for subject_scope in subject_scopes:
        path = f"chains/{subject_scope}/{_FULL_ARM}/state/owner/owner_hydration__social_record_store_v28.json"
        state = _owner_state_summary(reader=reader, path=path, expected_version=28)
        _require_equal(len(state["records"]), 12, f"{subject_scope} final record count")
        _require_equal(len(state["outcomes"]), 12, f"{subject_scope} final outcome count")
        mutations = _array(
            _object(state["payload"], "final owner payload").get("preference_action_outcome_mutation_receipts"),
            "final mutation receipts",
        )
        _require_equal(len(mutations), 0, f"{subject_scope} final mutation receipt count")
        final_owner_states.append(
            {
                "subject_scope": subject_scope,
                "hydration": deepcopy(state["reference"]),
                "record_count": 12,
                "outcome_count": 12,
                "mutation_receipt_count": 0,
                "ordered_record_ids": list(state["record_ids"]),
                "ordered_outcome_evidence_ids": list(state["outcome_ids"]),
            }
        )

    rows_document, rows_raw = _artifact_document(
        "relationship-product-horizon-attempt03-owner-history-rows.v1",
        audit_scope="post_hoc_owner_history_hard_window_contribution",
        source_protocol_id=_EXPECTED_PROTOCOL_ID,
        record_identity_fields=["arm_id", "subject_scope", "decision_id"],
        cross_arm_match_fields=["subject_scope", "decision_id"],
        credit_record_id_is_cross_arm_join_key=False,
        row_count=len(rows),
        rows=rows,
    )
    report_payload: dict[str, object] = {
        "audit_scope": "post_hoc_owner_history_hard_window_contribution",
        "source_attempt": {
            "attempt_root": reader.root_relative,
            "manifest_artifact_id": _EXPECTED_MANIFEST_ARTIFACT_ID,
            "manifest_raw_sha256": _EXPECTED_MANIFEST_RAW_SHA256,
            "protocol_id": _EXPECTED_PROTOCOL_ID,
            "protocol_raw_sha256": _EXPECTED_PROTOCOL_RAW_SHA256,
            "report_artifact_id": _EXPECTED_REPORT_ARTIFACT_ID,
            "report_raw_sha256": _EXPECTED_REPORT_RAW_SHA256,
            "frozen_verdict": frozen_report.get("verdict"),
            "source_public_and_sealed": source_provenance,
        },
        "owner_contract": {
            "unique_semantic_owner": "PreferenceAboutOtherModule/preference_about_other",
            "social_record_store_role": "carrier_only",
            "hard_window_size": _RECORD_WINDOW,
            "eviction_order": "source_order_last_12",
        },
        "join_contract": {
            "record_identity_fields": ["arm_id", "subject_scope", "decision_id"],
            "cross_arm_match_fields": ["subject_scope", "decision_id"],
            "world_clone_and_decision_index_validated": True,
            "credit_record_id_is_cross_arm_join_key": False,
        },
        "denominators": {
            "matched_world_count": 8,
            "decisions_per_world": 24,
            "matched_decision_count": 192,
            "full_vs_frozen_action_divergent_decision_count": 36,
            "materialized_arm_decision_count": 72,
            "exact_forecast_recomputation_count": 72,
            "primary_window_decisions_per_world": 12,
            "primary_window_matched_decision_count": 96,
            "primary_window_divergence_row_count": sum(int(row["primary_window"]) for row in rows),
        },
        "horizon_segment_divergence_counts": dict(sorted(Counter(str(row["horizon_segment"]) for row in rows).items())),
        "world_primary_window_directions": [world_directions[key] for key in sorted(world_directions)],
        "mechanical_replay": {
            "public_embedding_table": {
                **reader.reference(_EMBEDDING_TABLE_PATH, embedding_table),
                "artifact_id": _EXPECTED_EMBEDDING_TABLE_ID,
                "embedding_width": embedding_table["embedding_width"],
                "record_count": len(_array(embedding_table["records"], "embedding records")),
                "all_audited_texts_present": all(digest in vectors for digest in used_text_digests),
                "audited_distinct_text_count": len(used_text_digests),
            },
            "condition_reader_artifact": {
                **reader_artifact,
                "artifact_id": _EXPECTED_READER_ARTIFACT_ID,
            },
            "forecast_formula": "semantic_similarity ** 2",
            "prior_count": _float_payload(_PRIOR_COUNT, "prior count"),
            "evidence_weight": _float_payload(_EVIDENCE_WEIGHT, "evidence weight"),
            "receipt_canonical_payload_byte_exact_count": 72,
            "receipt_numeric_bit_exact_count": 72,
            "model_output_count": 0,
            "cuda_used": False,
            "network_used": False,
        },
        "non_load_bearing_fields": {
            "confidence_used_by_forecast": False,
            "confidence_field_scope": "OtherMindRecord.confidence",
            "elapsed_time_used_by_forecast": False,
            "virtual_day_used_by_forecast": False,
            "public_context_chunk_used_by_forecast": False,
            "prediction_error_refs_used_by_forecast": False,
            "named_reader_confidence_used_by_similarity": True,
            "source_turn_used_only_for_equal_weight_ordering": True,
        },
        "owner_component_discordance_counts_over_36_rows": expected_components,
        "hard_window_eviction_row_counts": {arm_id: eviction_counts[arm_id] for arm_id in _ARMS},
        "descriptive_mixed_path_crosscheck": {
            "patterns": expected_patterns,
            "full_gate_probability_distance_from_0_5_below_0_05_count": full_near_threshold,
            "frozen_gate_probability_distance_from_0_5_below_0_05_count": frozen_near_threshold,
            "causal_category_assigned": False,
            "explanation": (
                "Recommendation and gate-action differences co-occur in multiple patterns. These "
                "descriptive partitions do not identify a causal owner of the outcome difference."
            ),
        },
        "reader_truth_crosswalk": reader_diagnostic,
        "final_full_owner_states": final_owner_states,
        "source_contracts": source_contracts,
        "audit_implementation": audit_script_reference,
        "source_chain_references": sorted(
            chain_references,
            key=lambda value: _text(value["path"], "chain reference path"),
        ),
        "frozen_judgment_preserved": {
            "verdict": frozen_report.get("verdict"),
            "four_able_complete": False,
            "formal_evidence_authorized": False,
            "single_axis_contrast_claim_authorized": False,
            "attempt03_files_modified": False,
        },
        "honest_boundaries": {
            "post_hoc_diagnosis": True,
            "pre_registered_confirmatory_analysis": False,
            "mechanical_history_and_hard_window_contribution_replay_only": True,
            "product_causal_effect_established": False,
            "learnable_capability_established": False,
            "reader_error_attribution_authorized": False,
            "human_product_validation": False,
            "production_active": False,
            "claim_ceiling": (
                "The pinned owner histories and named-reader similarity-squared forecasts can be "
                "replayed exactly for the 36 full-vs-frozen action divergences. This does not "
                "identify a product causal effect or establish Learnable."
            ),
        },
        "outputs": {
            "rows": {
                "path": "rows.json",
                "artifact_id": rows_document["artifact_id"],
                "raw_sha256": _sha256_bytes(rows_raw),
                "bytes": len(rows_raw),
                "row_count": len(rows),
            }
        },
    }
    report_document, report_raw = _artifact_document(
        "relationship-product-horizon-attempt03-owner-history-audit-report.v1",
        **report_payload,
    )
    manifest_document, manifest_raw = _artifact_document(
        "relationship-product-horizon-attempt03-owner-history-audit-manifest.v1",
        manifest_written_last=True,
        source_attempt_manifest_artifact_id=_EXPECTED_MANIFEST_ARTIFACT_ID,
        source_attempt_manifest_raw_sha256=_EXPECTED_MANIFEST_RAW_SHA256,
        report_artifact_id=report_document["artifact_id"],
        files=[
            {
                "path": "rows.json",
                "bytes": len(rows_raw),
                "raw_sha256": _sha256_bytes(rows_raw),
                "artifact_id": rows_document["artifact_id"],
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
        "rows.json": rows_raw,
        "report.json": report_raw,
        "manifest.json": manifest_raw,
    }


def _write_documents_create_only(output_dir: Path, documents: Mapping[str, bytes]) -> None:
    output = output_dir.resolve()
    _repo_relative(output, label="audit output directory")
    if output.exists():
        _fail(f"refusing to overwrite existing audit output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    if temporary.exists():
        _fail(f"temporary output collision: {temporary}")
    temporary.mkdir()
    try:
        for name in _OUTPUT_FILES:
            raw = documents.get(name)
            if raw is None:
                _fail(f"missing generated output: {name}")
            path = temporary / name
            with path.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            _require_equal(path.read_bytes(), raw, f"fresh output {name}")
        if output.exists():
            _fail(f"audit output appeared concurrently: {output}")
        temporary.rename(output)
    except BaseException:
        if (
            temporary.exists()
            and temporary.parent == output.parent
            and temporary.name.startswith(f".{output.name}.tmp-")
        ):
            shutil.rmtree(temporary)
        raise


def materialize(attempt_root: Path, output_dir: Path) -> dict[str, object]:
    documents = build_audit_documents(attempt_root)
    _write_documents_create_only(output_dir, documents)
    report = _strict_json_object(documents["report.json"], label="generated report")
    manifest = _strict_json_object(documents["manifest.json"], label="generated manifest")
    return {
        "output_dir": _repo_relative(output_dir, label="audit output directory"),
        "report_artifact_id": report["artifact_id"],
        "manifest_artifact_id": manifest["artifact_id"],
        "row_count": 36,
        "recomputed_arm_decision_count": 72,
    }


def validate_existing(attempt_root: Path, output_dir: Path) -> dict[str, object]:
    expected = build_audit_documents(attempt_root)
    output = output_dir.resolve()
    _repo_relative(output, label="audit output directory")
    if not output.is_dir():
        _fail(f"audit output directory does not exist: {output}")
    observed_names = sorted(path.name for path in output.iterdir())
    _require_equal(observed_names, sorted(_OUTPUT_FILES), "audit output exact file set")
    for name in _OUTPUT_FILES:
        observed = (output / name).read_bytes()
        _require_equal(observed, expected[name], f"audit byte-exact replay {name}")
        payload = _strict_json_object(observed, label=f"audit output {name}")
        _require_artifact_identity(payload, label=f"audit output {name}")
    report = _strict_json_object(expected["report.json"], label="expected report")
    manifest = _strict_json_object(expected["manifest.json"], label="expected manifest")
    return {
        "output_dir": _repo_relative(output, label="audit output directory"),
        "report_artifact_id": report["artifact_id"],
        "manifest_artifact_id": manifest["artifact_id"],
        "byte_exact_replay": True,
        "row_count": 36,
        "recomputed_arm_decision_count": 72,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model-free attempt03 owner-history and hard-window contribution audit",
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
