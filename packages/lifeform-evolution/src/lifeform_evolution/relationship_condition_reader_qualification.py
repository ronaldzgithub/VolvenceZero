"""Model-free preregistration inputs for the relationship reader challenge.

This module prepares a physically separated qualification preflight.  The
future predictor projection contains only an opaque item id and exact public
text.  Source position, group membership, and evaluator labels remain sealed
until after predictions have been persisted by a later execution package.

The current protocol is deliberately *not* an execution authorization.  It
freezes the source split, semantic model, reader solver, and promotion gates so
the full CUDA executor can be frozen and externally anchored in a subsequent
convergence package without changing the scientific question.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
import pathlib
import stat
from typing import Mapping

from lifeform_domain_emogpt.lab.contracts import canonical_json, sha256_json
from lifeform_domain_emogpt.lab.relationship_product_pilot_source import (
    RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION as RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V1,
    RelationshipProductPilotEvaluatorBundle,
    RelationshipProductPilotPublicView,
    build_relationship_product_pilot_evaluator_bundle as build_legacy_product_pilot_evaluator_bundle,
    build_relationship_product_pilot_public_view as build_legacy_product_pilot_public_view,
    load_relationship_product_pilot_source_protocol as load_legacy_product_pilot_source_protocol,
    relationship_product_pilot_source_protocol_path as legacy_product_pilot_source_protocol_path,
)
from lifeform_domain_emogpt.lab.relationship_product_pilot_source_v2 import (
    RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V2,
    RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3,
    archived_relationship_product_pilot_source_v2_protocol_path,
    build_relationship_product_pilot_evaluator_bundle as build_independent_product_pilot_evaluator_bundle,
    build_relationship_product_pilot_public_view as build_independent_product_pilot_public_view,
    load_archived_relationship_product_pilot_source_v2_protocol,
    load_relationship_product_pilot_source_protocol as load_independent_product_pilot_source_protocol,
    relationship_product_pilot_source_protocol_path as independent_product_pilot_source_protocol_path,
)
from lifeform_domain_emogpt.relationship_condition_reader import (
    RELATIONSHIP_CONDITION_LINEAR_READER_SCHEMA_VERSION,
    RELATIONSHIP_CONDITION_LINEAR_SOLVER,
    RELATIONSHIP_CONDITION_LINEAR_SOLVER_VERSION,
)

from lifeform_evolution.relationship_lab_product_model_adapters import (
    BGE_M3_MODEL_ID,
    BGE_M3_MODEL_REVISION,
    BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    BGE_M3_WEIGHT_BYTES_SHA256,
)


RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V1 = (
    "relationship-condition-reader-qualification-protocol.v1"
)
RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V2 = (
    "relationship-condition-reader-qualification-protocol.v2"
)
RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION = (
    RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V2
)
RELATIONSHIP_READER_QUALIFICATION_PUBLIC_CORPUS_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-public-corpus.v1"
)
RELATIONSHIP_READER_QUALIFICATION_PREDICTOR_REQUEST_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-predictor-request.v1"
)
RELATIONSHIP_READER_QUALIFICATION_TRAINING_LABELS_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-training-labels.v1"
)
RELATIONSHIP_READER_QUALIFICATION_CHALLENGE_LABELS_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-challenge-labels.v1"
)
RELATIONSHIP_READER_QUALIFICATION_GROUP_SPLIT_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-group-split.v1"
)
RELATIONSHIP_READER_QUALIFICATION_PUBLICATION_REQUEST_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-publication-request.v1"
)
RELATIONSHIP_READER_QUALIFICATION_MANIFEST_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-preflight-manifest.v1"
)

_TRAINING_COUNT = 4
_CHALLENGE_COUNT = 224
_CHALLENGE_GROUP_COUNT = 28
_ROWS_PER_CHALLENGE_GROUP = 8
_TRAINING_COUNT_PER_CLASS = 2
_CHALLENGE_COUNT_PER_CLASS = 112
# These labels deliberately match the existing Product Horizon named-reader
# contract.  The linear reader is label-agnostic; its unit-test fixture names
# are not a second product taxonomy.
_READER_LABELS = ("agency_displacement", "belonging_erasure")
_CONDITION_TO_READER_LABEL = {
    "agency_under_override": "agency_displacement",
    "connection_under_exclusion": "belonging_erasure",
}
_PUBLIC_FORBIDDEN_KEYS = frozenset(
    {
        "action_id",
        "active_policy_mode",
        "arm_id",
        "cohort_id",
        "condition_id",
        "condition_label",
        "decision_index",
        "decision_id",
        "domain_id",
        "environment_seed",
        "expected_label",
        "group_id",
        "observed_outcome_id",
        "outcome_id",
        "phase_id",
        "policy_id",
        "preferred_action_id",
        "scene_id",
        "session_id",
        "source_position",
        "stage_id",
        "subject_index",
        "surface_kind",
        "world_clone_id",
    }
)
_PREFLIGHT_SOURCE_PATHS_V1 = frozenset(
    {
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/"
        "relationship_product_pilot_source.py",
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab_protocols/"
        "relationship_product_pilot_source_v1.json",
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab_protocols/"
        "relationship_product_pilot_source_v2.json",
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/"
        "relationship_condition_reader.py",
        "packages/lifeform-evolution/src/lifeform_evolution/"
        "relationship_condition_reader_qualification.py",
        "packages/lifeform-evolution/src/lifeform_evolution/"
        "relationship_lab_product_model_adapters.py",
        "scripts/run_relationship_condition_reader_qualification.py",
    }
)
_PREFLIGHT_SOURCE_PATHS_V2 = frozenset(
    {
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/"
        "relationship_product_pilot_source.py",
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab/"
        "relationship_product_pilot_source_v2.py",
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab_protocols/"
        "relationship_product_pilot_source_v1.json",
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab_protocols/"
        "relationship_product_pilot_source_v2.json",
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/lab_protocols/"
        "relationship_product_pilot_source_v3.json",
        "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/"
        "relationship_condition_reader.py",
        "packages/lifeform-evolution/src/lifeform_evolution/"
        "relationship_condition_reader_qualification.py",
        "packages/lifeform-evolution/src/lifeform_evolution/"
        "relationship_lab_product_model_adapters.py",
        "scripts/run_relationship_condition_reader_qualification.py",
    }
)
_QUALIFICATION_PROTOCOL_FILENAMES = {
    RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V1: (
        "relationship_condition_reader_qualification_v1.json"
    ),
    RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V2: (
        "relationship_condition_reader_qualification_v2.json"
    ),
}
_PREFLIGHT_SOURCE_PATHS_BY_PROTOCOL = {
    RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V1: _PREFLIGHT_SOURCE_PATHS_V1,
    RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V2: _PREFLIGHT_SOURCE_PATHS_V2,
}
_PREPARED_RELATIVE_PATHS = (
    "protocol.json",
    "public/public_corpus.json",
    "public/predictor_request.json",
    "public/publication_request.json",
    "sealed/condition_training_labels.json",
    "sealed/challenge_labels.json",
    "sealed/group_split.json",
)


def _preflight_source_paths(schema_version: str) -> frozenset[str]:
    try:
        return _PREFLIGHT_SOURCE_PATHS_BY_PROTOCOL[schema_version]
    except KeyError as exc:
        raise ValueError(f"unregistered qualification protocol schema: {schema_version!r}") from exc


@dataclass(frozen=True)
class QualificationSourcePin:
    schema_version: str
    raw_sha256: str
    raw_bytes: int
    protocol_id: str
    public_plan_id: str
    sealed_bundle_id: str


@dataclass(frozen=True)
class RelationshipConditionReaderQualificationProtocol:
    """Strict immutable view over one packaged preregistration payload."""

    schema_version: str
    canonical_payload: str
    raw_sha256: str
    raw_bytes: int
    training_source: QualificationSourcePin
    challenge_source: QualificationSourcePin
    preflight_source_sha256s: tuple[tuple[str, str], ...]

    @property
    def protocol_id(self) -> str:
        return hashlib.sha256(self.canonical_payload.encode("utf-8")).hexdigest()

    def to_payload(self) -> dict[str, object]:
        payload = json.loads(self.canonical_payload)
        if not isinstance(payload, dict):  # pragma: no cover - constructor invariant
            raise RuntimeError("qualification protocol canonical payload drifted")
        return payload


@dataclass(frozen=True)
class OpaqueQualificationTextInput:
    """Only payload shape a future prediction child may receive."""

    item_id: str
    text: str
    text_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.item_id, "item_id")
        _require_text(self.text, "text")
        _require_sha256(self.text_sha256, "text_sha256")
        if self.text_sha256 != _sha256_text(self.text):
            raise ValueError("qualification text_sha256 does not match exact text")

    def to_payload(self) -> dict[str, str]:
        return {
            "item_id": self.item_id,
            "text": self.text,
            "text_sha256": self.text_sha256,
        }


@dataclass(frozen=True)
class PreparedQualificationArtifacts:
    public_corpus: Mapping[str, object]
    predictor_request: Mapping[str, object]
    training_labels: Mapping[str, object]
    challenge_labels: Mapping[str, object]
    group_split: Mapping[str, object]
    publication_request: Mapping[str, object]


def relationship_condition_reader_qualification_protocol_path(
    schema_version: str = RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION,
) -> pathlib.Path:
    try:
        filename = _QUALIFICATION_PROTOCOL_FILENAMES[schema_version]
    except KeyError as exc:
        raise ValueError(f"unregistered qualification protocol schema: {schema_version!r}") from exc
    return pathlib.Path(__file__).resolve().parent / "protocols" / filename


def load_relationship_condition_reader_qualification_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipConditionReaderQualificationProtocol:
    source = pathlib.Path(path or relationship_condition_reader_qualification_protocol_path())
    raw_bytes = source.read_bytes()
    if b"\r" in raw_bytes:
        raise ValueError("qualification protocol must use LF-only UTF-8 bytes")
    if not raw_bytes.endswith(b"\n"):
        raise ValueError("qualification protocol must end with one LF")
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("qualification protocol must be exact UTF-8") from exc
    payload = _parse_unique_json(text, "qualification protocol")
    _validate_protocol_payload(payload)
    sources = _mapping(payload["sources"], "sources")
    execution = _mapping(payload["execution"], "execution")
    raw_preflight_sources = _mapping(
        execution["preflight_source_sha256s"],
        "execution.preflight_source_sha256s",
    )
    protocol = RelationshipConditionReaderQualificationProtocol(
        schema_version=_text(payload["schema_version"], "schema_version"),
        canonical_payload=canonical_json(payload),
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        raw_bytes=len(raw_bytes),
        training_source=_source_pin(
            _mapping(sources["training"], "sources.training")
        ),
        challenge_source=_source_pin(
            _mapping(sources["challenge"], "sources.challenge")
        ),
        preflight_source_sha256s=tuple(
            sorted(
                (
                    _relative_path(key, "execution source path"),
                    _digest(value, "execution source sha256"),
                )
                for key, value in raw_preflight_sources.items()
            )
        ),
    )
    return protocol


def prepare_relationship_condition_reader_qualification_preflight(
    *,
    preflight_root: pathlib.Path,
    proposed_execution_root: pathlib.Path,
) -> Mapping[str, object]:
    """Create zero-model publication inputs without touching the execution root."""

    root = pathlib.Path(preflight_root).resolve()
    proposed = pathlib.Path(proposed_execution_root).resolve()
    if root.exists():
        raise FileExistsError(f"qualification preflight root already exists: {root}")
    if proposed.exists():
        raise FileExistsError(
            f"proposed qualification execution root must not exist before anchor: {proposed}"
        )
    if root == proposed or root in proposed.parents or proposed in root.parents:
        raise ValueError("preflight and proposed execution roots must be disjoint")

    protocol_source = relationship_condition_reader_qualification_protocol_path()
    protocol = load_relationship_condition_reader_qualification_protocol(protocol_source)
    _validate_preflight_source_pins(protocol)
    artifacts = _build_prepared_artifacts(protocol, proposed_execution_root=proposed)

    root.mkdir(parents=True)
    payloads = {
        "public/public_corpus.json": artifacts.public_corpus,
        "public/predictor_request.json": artifacts.predictor_request,
        "public/publication_request.json": artifacts.publication_request,
        "sealed/condition_training_labels.json": artifacts.training_labels,
        "sealed/challenge_labels.json": artifacts.challenge_labels,
        "sealed/group_split.json": artifacts.group_split,
    }
    _write_bytes_create_only(root / "protocol.json", protocol_source.read_bytes())
    for relative_path, payload in payloads.items():
        _write_json_create_only(root / relative_path, payload)
    manifest = _build_manifest(root=root, protocol=protocol)
    _write_json_create_only(root / "manifest.json", manifest)
    return validate_relationship_condition_reader_qualification_preflight(
        preflight_root=root,
        expected_protocol_id=protocol.protocol_id,
        expected_publication_request_artifact_id=_digest(
            artifacts.publication_request["artifact_id"],
            "publication_request artifact_id",
        ),
        expected_proposed_execution_root=proposed,
    )


def validate_relationship_condition_reader_qualification_preflight(
    *,
    preflight_root: pathlib.Path,
    expected_protocol_id: str,
    expected_publication_request_artifact_id: str,
    expected_proposed_execution_root: pathlib.Path,
) -> Mapping[str, object]:
    """Recompute a prepared preflight without importing or loading a model."""

    expected = _digest(expected_protocol_id, "expected_protocol_id")
    expected_publication = _digest(
        expected_publication_request_artifact_id,
        "expected_publication_request_artifact_id",
    )
    expected_proposed = pathlib.Path(expected_proposed_execution_root).resolve()
    root = pathlib.Path(preflight_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"qualification preflight root does not exist: {root}")
    protocol_path = root / "protocol.json"
    protocol = load_relationship_condition_reader_qualification_protocol(protocol_path)
    packaged = relationship_condition_reader_qualification_protocol_path(protocol.schema_version)
    if protocol_path.read_bytes() != packaged.read_bytes():
        raise ValueError("preflight protocol differs from packaged preregistration")
    if protocol.protocol_id != expected:
        raise ValueError("external expected qualification protocol id mismatch")
    _validate_preflight_source_pins(protocol)

    expected_paths = {*_PREPARED_RELATIVE_PATHS, "manifest.json"}
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("qualification preflight contains missing or extra files")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("qualification preflight rejects symlinks")
        if path.is_file() and path.stat().st_nlink != 1:
            raise ValueError("qualification preflight rejects hard-linked files")

    publication_request = _load_canonical_artifact(
        root / "public/publication_request.json",
        RELATIONSHIP_READER_QUALIFICATION_PUBLICATION_REQUEST_SCHEMA_VERSION,
    )
    if publication_request["artifact_id"] != expected_publication:
        raise ValueError("external expected publication request artifact id mismatch")
    proposed = pathlib.Path(
        _text(publication_request["proposed_execution_root"], "proposed_execution_root")
    ).resolve()
    if proposed != expected_proposed:
        raise ValueError("external expected proposed execution root mismatch")
    if proposed.exists():
        raise ValueError("proposed qualification execution root now exists before anchor")
    if root == proposed or root in proposed.parents or proposed in root.parents:
        raise ValueError("preflight and proposed execution roots must remain disjoint")
    recomputed = _build_prepared_artifacts(protocol, proposed_execution_root=proposed)
    expected_payloads = {
        "public/public_corpus.json": recomputed.public_corpus,
        "public/predictor_request.json": recomputed.predictor_request,
        "public/publication_request.json": recomputed.publication_request,
        "sealed/condition_training_labels.json": recomputed.training_labels,
        "sealed/challenge_labels.json": recomputed.challenge_labels,
        "sealed/group_split.json": recomputed.group_split,
    }
    for relative_path, payload in expected_payloads.items():
        actual = _load_canonical_artifact(root / relative_path, _text(payload["schema_version"], "schema_version"))
        if actual != payload:
            raise ValueError(f"qualification preflight artifact drifted: {relative_path}")

    manifest = _load_canonical_artifact(
        root / "manifest.json",
        RELATIONSHIP_READER_QUALIFICATION_MANIFEST_SCHEMA_VERSION,
    )
    if manifest != _build_manifest(root=root, protocol=protocol):
        raise ValueError("qualification preflight manifest/file tree drifted")
    return {
        "schema_version": "relationship-condition-reader-qualification-preflight-validation.v1",
        "protocol_id": protocol.protocol_id,
        "protocol_raw_sha256": protocol.raw_sha256,
        "protocol_raw_bytes": protocol.raw_bytes,
        "public_corpus_artifact_id": recomputed.public_corpus["artifact_id"],
        "predictor_request_artifact_id": recomputed.predictor_request["artifact_id"],
        "training_labels_artifact_id": recomputed.training_labels["artifact_id"],
        "challenge_labels_artifact_id": recomputed.challenge_labels["artifact_id"],
        "group_split_artifact_id": recomputed.group_split["artifact_id"],
        "publication_request_artifact_id": recomputed.publication_request["artifact_id"],
        "training_input_count": _TRAINING_COUNT,
        "challenge_input_count": _CHALLENGE_COUNT,
        "challenge_group_count": _CHALLENGE_GROUP_COUNT,
        "proposed_execution_root": str(proposed),
        "proposed_execution_root_exists": False,
        "model_or_cuda_used": False,
        "external_public_anchor_created": False,
        "qualification_execution_authorized": False,
        "condition_reader_qualified": False,
        "campaign_execution_admitted": False,
        "readable_product_effect": False,
        "four_able_complete": False,
        "formal_evidence_authorized": False,
    }


def _build_prepared_artifacts(
    protocol: RelationshipConditionReaderQualificationProtocol,
    *,
    proposed_execution_root: pathlib.Path,
) -> PreparedQualificationArtifacts:
    if protocol.training_source.schema_version != RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V1:
        raise ValueError("qualification training source must remain the legacy source-v1 owner")
    training_protocol = load_legacy_product_pilot_source_protocol()
    training_public = build_legacy_product_pilot_public_view(training_protocol)
    training_evaluator = build_legacy_product_pilot_evaluator_bundle(training_protocol)
    if protocol.challenge_source.schema_version == RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V2:
        challenge_protocol = load_archived_relationship_product_pilot_source_v2_protocol()
    elif protocol.challenge_source.schema_version == RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3:
        challenge_protocol = load_independent_product_pilot_source_protocol()
    else:
        raise ValueError("qualification challenge source is not an owned independent revision")
    challenge_public = build_independent_product_pilot_public_view(challenge_protocol)
    challenge_evaluator = build_independent_product_pilot_evaluator_bundle(challenge_protocol)
    _validate_source_pin(
        protocol.training_source,
        schema_version=RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V1,
        public=training_public,
        evaluator=training_evaluator,
    )
    _validate_source_pin(
        protocol.challenge_source,
        schema_version=protocol.challenge_source.schema_version,
        public=challenge_public,
        evaluator=challenge_evaluator,
    )

    training_inputs, training_label_rows = _training_rows(
        protocol,
        public=training_public,
        evaluator=training_evaluator,
    )
    challenge_inputs, challenge_label_rows = _challenge_rows(
        protocol,
        public=challenge_public,
        evaluator=challenge_evaluator,
    )
    training_texts = {item.text_sha256 for item in training_inputs}
    challenge_texts = {item.text_sha256 for item in challenge_inputs}
    if training_texts & challenge_texts:
        raise ValueError("qualification training/challenge exact text overlap")

    public_corpus = _with_artifact_id(
        {
            "schema_version": RELATIONSHIP_READER_QUALIFICATION_PUBLIC_CORPUS_SCHEMA_VERSION,
            "protocol_id": protocol.protocol_id,
            "training_inputs": [item.to_payload() for item in training_inputs],
            "challenge_inputs": [item.to_payload() for item in challenge_inputs],
            "training_input_count": len(training_inputs),
            "challenge_input_count": len(challenge_inputs),
            "exact_text_overlap_count": 0,
            "predictor_projection": "opaque_item_id_exact_text_and_text_sha256_only",
        }
    )
    _assert_public_firewall(public_corpus)
    predictor_request = _with_artifact_id(
        {
            "schema_version": RELATIONSHIP_READER_QUALIFICATION_PREDICTOR_REQUEST_SCHEMA_VERSION,
            "protocol_id": protocol.protocol_id,
            "public_corpus_artifact_id": public_corpus["artifact_id"],
            "challenge_inputs": [item.to_payload() for item in challenge_inputs],
            "challenge_input_count": len(challenge_inputs),
        }
    )
    _assert_public_firewall(predictor_request)
    _validate_predictor_request_surface(predictor_request)
    training_labels = _with_artifact_id(
        {
            "schema_version": RELATIONSHIP_READER_QUALIFICATION_TRAINING_LABELS_SCHEMA_VERSION,
            "protocol_id": protocol.protocol_id,
            "public_corpus_artifact_id": public_corpus["artifact_id"],
            "rows": list(training_label_rows),
            "row_count": len(training_label_rows),
            "labels": list(_READER_LABELS),
            "condition_only": True,
            "action_outcome_pe_credit_evaluation_present": False,
        }
    )
    challenge_labels = _with_artifact_id(
        {
            "schema_version": RELATIONSHIP_READER_QUALIFICATION_CHALLENGE_LABELS_SCHEMA_VERSION,
            "protocol_id": protocol.protocol_id,
            "public_corpus_artifact_id": public_corpus["artifact_id"],
            "rows": list(challenge_label_rows),
            "row_count": len(challenge_label_rows),
            "label_release_condition": "prediction_ledger_create_only_fsynced",
        }
    )
    group_rows: list[dict[str, object]] = []
    by_group: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in challenge_label_rows:
        by_group[_text(row["group_id"], "group_id")].append(row)
    for group_id, rows in sorted(by_group.items()):
        labels = {_text(row["condition_label"], "condition_label") for row in rows}
        if len(labels) != 1:
            raise ValueError("qualification challenge group maps to multiple labels")
        group_rows.append(
            {
                "group_id": group_id,
                "item_ids": sorted(_text(row["item_id"], "item_id") for row in rows),
                "row_count": len(rows),
                "condition_label": next(iter(labels)),
            }
        )
    group_split = _with_artifact_id(
        {
            "schema_version": RELATIONSHIP_READER_QUALIFICATION_GROUP_SPLIT_SCHEMA_VERSION,
            "protocol_id": protocol.protocol_id,
            "training_item_ids": [item.item_id for item in training_inputs],
            "challenge_item_ids": [item.item_id for item in challenge_inputs],
            "challenge_groups": group_rows,
            "challenge_group_count": len(group_rows),
            "rows_per_challenge_group": _ROWS_PER_CHALLENGE_GROUP,
            "training_challenge_text_overlap_count": 0,
            "statistical_independence_claim": False,
            "grouping_owner": "qualification_preflight",
            "grouping_contract": (
                "surface_kind_and_source_position_across_voice_variants.v1"
            ),
            "group_level_evaluation_unit_count": _CHALLENGE_GROUP_COUNT,
        }
    )
    publication_request = _with_artifact_id(
        {
            "schema_version": RELATIONSHIP_READER_QUALIFICATION_PUBLICATION_REQUEST_SCHEMA_VERSION,
            "protocol_id": protocol.protocol_id,
            "protocol_filename": relationship_condition_reader_qualification_protocol_path(
                protocol.schema_version
            ).name,
            "protocol_raw_sha256": protocol.raw_sha256,
            "protocol_raw_bytes": protocol.raw_bytes,
            "public_corpus_artifact_id": public_corpus["artifact_id"],
            "predictor_request_artifact_id": predictor_request["artifact_id"],
            "training_labels_artifact_id": training_labels["artifact_id"],
            "challenge_labels_artifact_id": challenge_labels["artifact_id"],
            "group_split_artifact_id": group_split["artifact_id"],
            "proposed_execution_root": str(proposed_execution_root),
            "proposed_execution_root_exists_at_prepare": False,
            "external_observation_required": True,
            "requested_publication_visibility": "public",
            "public_gist_created": False,
            "qualification_execution_authorized": False,
        }
    )
    _validate_artifact_counts(
        training_inputs=training_inputs,
        challenge_inputs=challenge_inputs,
        training_label_rows=training_label_rows,
        challenge_label_rows=challenge_label_rows,
        group_rows=tuple(group_rows),
    )
    return PreparedQualificationArtifacts(
        public_corpus=public_corpus,
        predictor_request=predictor_request,
        training_labels=training_labels,
        challenge_labels=challenge_labels,
        group_split=group_split,
        publication_request=publication_request,
    )


def _training_rows(
    protocol: RelationshipConditionReaderQualificationProtocol,
    *,
    public: RelationshipProductPilotPublicView,
    evaluator: RelationshipProductPilotEvaluatorBundle,
) -> tuple[tuple[OpaqueQualificationTextInput, ...], tuple[Mapping[str, object], ...]]:
    evaluator_by_session = {row.session_id: row for row in evaluator.onboarding_sessions}
    by_text: dict[str, tuple[OpaqueQualificationTextInput, dict[str, object]]] = {}
    for subject in public.subjects:
        for source_position, session in enumerate(subject.onboarding_sessions):
            truth = evaluator_by_session[session.session_id]
            label = _CONDITION_TO_READER_LABEL[truth.condition_id]
            text_sha256 = _sha256_text(session.user_utterance)
            item = OpaqueQualificationTextInput(
                item_id=sha256_json(
                    {
                        "schema_version": "relationship-reader-qualification-item.v1",
                        "source_protocol_id": protocol.training_source.protocol_id,
                        "text_sha256": text_sha256,
                    }
                ),
                text=session.user_utterance,
                text_sha256=text_sha256,
            )
            label_row = {
                "item_id": item.item_id,
                "text_sha256": item.text_sha256,
                "condition_label": label,
                "source_position": source_position,
            }
            existing = by_text.get(text_sha256)
            if existing is not None and existing != (item, label_row):
                raise ValueError("deduplicated training text has inconsistent label/position")
            by_text[text_sha256] = (item, label_row)
    ordered = tuple(value for _, value in sorted(by_text.items()))
    return (
        tuple(sorted((item for item, _ in ordered), key=lambda item: item.item_id)),
        tuple(sorted((row for _, row in ordered), key=lambda row: _text(row["item_id"], "item_id"))),
    )


def _challenge_rows(
    protocol: RelationshipConditionReaderQualificationProtocol,
    *,
    public: RelationshipProductPilotPublicView,
    evaluator: RelationshipProductPilotEvaluatorBundle,
) -> tuple[tuple[OpaqueQualificationTextInput, ...], tuple[Mapping[str, object], ...]]:
    onboarding_truth = {row.session_id: row for row in evaluator.onboarding_sessions}
    decision_truth = {row.session_id: row for row in evaluator.decision_sessions}
    inputs: list[OpaqueQualificationTextInput] = []
    labels: list[dict[str, object]] = []
    for subject_index, subject in enumerate(public.subjects):
        surfaces = (
            *(
                ("onboarding", index, session.session_id, session.user_utterance)
                for index, session in enumerate(subject.onboarding_sessions)
            ),
            *(
                ("decision", index, session.session_id, session.current_input)
                for index, session in enumerate(subject.decision_sessions)
            ),
        )
        for surface_kind, source_position, session_id, text in surfaces:
            truth = (
                onboarding_truth[session_id]
                if surface_kind == "onboarding"
                else decision_truth[session_id]
            )
            text_sha256 = _sha256_text(text)
            item = OpaqueQualificationTextInput(
                item_id=sha256_json(
                    {
                        "schema_version": "relationship-reader-qualification-item.v1",
                        "source_protocol_id": protocol.challenge_source.protocol_id,
                        "text_sha256": text_sha256,
                    }
                ),
                text=text,
                text_sha256=text_sha256,
            )
            group_id = sha256_json(
                {
                    "schema_version": "relationship-reader-qualification-group.v1",
                    "source_protocol_id": protocol.challenge_source.protocol_id,
                    "surface_kind": surface_kind,
                    "source_position": source_position,
                }
            )
            inputs.append(item)
            labels.append(
                {
                    "item_id": item.item_id,
                    "text_sha256": item.text_sha256,
                    "condition_label": _CONDITION_TO_READER_LABEL[truth.condition_id],
                    "group_id": group_id,
                    "subject_index": subject_index,
                    "surface_kind": surface_kind,
                    "source_position": source_position,
                    "source_session_id": session_id,
                }
            )
    if len({item.item_id for item in inputs}) != len(inputs):
        raise ValueError("qualification challenge opaque item ids must be unique")
    return (
        tuple(sorted(inputs, key=lambda item: item.item_id)),
        tuple(sorted(labels, key=lambda row: _text(row["item_id"], "item_id"))),
    )


def _validate_artifact_counts(
    *,
    training_inputs: tuple[OpaqueQualificationTextInput, ...],
    challenge_inputs: tuple[OpaqueQualificationTextInput, ...],
    training_label_rows: tuple[Mapping[str, object], ...],
    challenge_label_rows: tuple[Mapping[str, object], ...],
    group_rows: tuple[Mapping[str, object], ...],
) -> None:
    if len(training_inputs) != _TRAINING_COUNT or len(training_label_rows) != _TRAINING_COUNT:
        raise ValueError("qualification requires four unique v1 training utterances")
    if len(challenge_inputs) != _CHALLENGE_COUNT or len(challenge_label_rows) != _CHALLENGE_COUNT:
        raise ValueError("qualification requires 224 source-v2 challenge rows")
    training_counts = Counter(_text(row["condition_label"], "condition_label") for row in training_label_rows)
    challenge_counts = Counter(_text(row["condition_label"], "condition_label") for row in challenge_label_rows)
    if training_counts != Counter({label: _TRAINING_COUNT_PER_CLASS for label in _READER_LABELS}):
        raise ValueError("qualification training labels must be balanced 2/2")
    if challenge_counts != Counter({label: _CHALLENGE_COUNT_PER_CLASS for label in _READER_LABELS}):
        raise ValueError("qualification challenge labels must be balanced 112/112")
    if len(group_rows) != _CHALLENGE_GROUP_COUNT or any(
        _integer(row["row_count"], "group row_count") != _ROWS_PER_CHALLENGE_GROUP
        for row in group_rows
    ):
        raise ValueError("qualification challenge requires 28 groups of eight voice variants")


def _validate_source_pin(
    pin: QualificationSourcePin,
    *,
    schema_version: str,
    public: RelationshipProductPilotPublicView,
    evaluator: RelationshipProductPilotEvaluatorBundle,
) -> None:
    if schema_version == RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V1:
        path = legacy_product_pilot_source_protocol_path()
        loaded = load_legacy_product_pilot_source_protocol()
    elif schema_version == RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V2:
        path = archived_relationship_product_pilot_source_v2_protocol_path()
        loaded = load_archived_relationship_product_pilot_source_v2_protocol()
    elif schema_version == RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3:
        path = independent_product_pilot_source_protocol_path(schema_version)
        loaded = load_independent_product_pilot_source_protocol()
    else:
        raise ValueError(f"qualification source schema is not registered: {schema_version!r}")
    raw = path.read_bytes()
    actual = QualificationSourcePin(
        schema_version=schema_version,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        raw_bytes=len(raw),
        protocol_id=loaded.protocol_sha256,
        public_plan_id=public.public_plan_sha256,
        sealed_bundle_id=evaluator.sealed_bundle_sha256,
    )
    if actual != pin:
        raise ValueError(f"qualification source pin drifted for {schema_version}")


def _validate_preflight_source_pins(
    protocol: RelationshipConditionReaderQualificationProtocol,
) -> None:
    repo = _repo_root().resolve()
    expected_paths = _preflight_source_paths(protocol.schema_version)
    actual_paths = {path for path, _ in protocol.preflight_source_sha256s}
    if actual_paths != expected_paths:
        raise ValueError("qualification preflight source pin set drifted")
    for relative_path, expected_sha256 in protocol.preflight_source_sha256s:
        path = repo / pathlib.PurePosixPath(relative_path)
        _reject_reparse_components(path, stop=repo, field_name="preflight source")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"qualification execution source missing or unsafe: {relative_path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(repo):
            raise ValueError(
                f"qualification preflight source escapes repository: {relative_path}"
            )
        if path.stat().st_nlink != 1:
            raise ValueError(
                f"qualification preflight source must not be hard linked: {relative_path}"
            )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError(f"qualification execution source drifted: {relative_path}")


def _validate_protocol_payload(payload: Mapping[str, object]) -> None:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "evidence_role",
            "sources",
            "label_crosswalk",
            "semantic_model",
            "reader",
            "qualification_gates",
            "execution",
            "claims",
        },
        "qualification protocol",
    )
    schema_version = _text(payload["schema_version"], "schema_version")
    if schema_version not in _QUALIFICATION_PROTOCOL_FILENAMES:
        raise ValueError("qualification protocol schema mismatch")
    if payload["evidence_role"] != "exact_source_reader_development_admission_only":
        raise ValueError("qualification protocol evidence role drifted")
    sources = _mapping(payload["sources"], "sources")
    _require_exact_keys(sources, {"training", "challenge"}, "sources")
    for name in ("training", "challenge"):
        raw = _mapping(sources[name], f"sources.{name}")
        _require_exact_keys(
            raw,
            {
                "schema_version",
                "raw_sha256",
                "raw_bytes",
                "protocol_id",
                "public_plan_id",
                "sealed_bundle_id",
                "selection",
            },
            f"sources.{name}",
        )
        _source_pin(raw)
    if _text(_mapping(sources["training"], "sources.training")["selection"], "training selection") != (
        "deduplicated_v1_onboarding_user_utterance_only"
    ):
        raise ValueError("qualification training selection drifted")
    if _text(_mapping(sources["challenge"], "sources.challenge")["selection"], "challenge selection") != (
        "all_v2_onboarding_user_utterance_and_decision_current_input"
    ):
        raise ValueError("qualification challenge selection drifted")
    training_pin = _source_pin(_mapping(sources["training"], "sources.training"))
    challenge_pin = _source_pin(_mapping(sources["challenge"], "sources.challenge"))
    expected_challenge_schema = (
        RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V2
        if schema_version == RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION_V1
        else RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V3
    )
    if training_pin.schema_version != RELATIONSHIP_PRODUCT_PILOT_SOURCE_SCHEMA_VERSION_V1:
        raise ValueError("qualification protocol must train from legacy source-v1")
    if challenge_pin.schema_version != expected_challenge_schema:
        raise ValueError("qualification protocol challenge source revision drifted")

    crosswalk = _mapping(payload["label_crosswalk"], "label_crosswalk")
    if crosswalk != _CONDITION_TO_READER_LABEL:
        raise ValueError("qualification condition-to-reader label crosswalk drifted")
    semantic = _mapping(payload["semantic_model"], "semantic_model")
    _positive_integer(semantic.get("embedding_width"), "semantic embedding_width")
    _boolean(semantic.get("network_allowed"), "semantic network_allowed")
    _boolean(semantic.get("stub_allowed"), "semantic stub_allowed")
    expected_semantic = {
        "model_id": BGE_M3_MODEL_ID,
        "model_revision": BGE_M3_MODEL_REVISION,
        "weights_sha256": BGE_M3_WEIGHT_BYTES_SHA256,
        "sentence_transformers_version": BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
        "embedding_width": 1024,
        "device": "cuda",
        "network_allowed": False,
        "stub_allowed": False,
    }
    if semantic != expected_semantic:
        raise ValueError("qualification semantic model contract drifted")
    reader = _mapping(payload["reader"], "reader")
    _boolean(reader.get("runtime_fit_allowed"), "reader runtime_fit_allowed")
    expected_reader = {
        "schema_version": RELATIONSHIP_CONDITION_LINEAR_READER_SCHEMA_VERSION,
        "solver": RELATIONSHIP_CONDITION_LINEAR_SOLVER,
        "solver_version": RELATIONSHIP_CONDITION_LINEAR_SOLVER_VERSION,
        "labels": list(_READER_LABELS),
        "score_definition": (
            "clamp_-1_1(cosine(unit_input,unit_centroid)+bias).v1"
        ),
        "normalized_margin_definition": (
            "clamp_0_1((top_score-second_score)/2).v1"
        ),
        "runtime_fit_allowed": False,
    }
    if reader != expected_reader:
        raise ValueError("qualification reader contract drifted")
    gates = _mapping(payload["qualification_gates"], "qualification_gates")
    for field_name in (
        "training_unique_count",
        "training_count_per_class",
        "challenge_row_count",
        "challenge_count_per_class",
        "challenge_group_count",
        "rows_per_challenge_group",
        "required_correct_rows",
        "required_correct_groups",
        "fresh_bge_process_count",
    ):
        _positive_integer(gates.get(field_name), f"qualification gate {field_name}")
    margin = gates.get("minimum_normalized_margin")
    if not isinstance(margin, float) or margin != 0.01:
        raise ValueError(
            "qualification gate minimum_normalized_margin must be float 0.01"
        )
    for field_name in (
        "top1_expected_required",
        "fresh_bge_exact_vector_reobservation_required",
        "reader_artifact_rederived_from_reobservation_required",
        "prediction_ledger_fsync_before_label_release",
        "statistical_independence_claim",
    ):
        _boolean(gates.get(field_name), f"qualification gate {field_name}")
    expected_gates = {
        "training_unique_count": _TRAINING_COUNT,
        "training_count_per_class": _TRAINING_COUNT_PER_CLASS,
        "challenge_row_count": _CHALLENGE_COUNT,
        "challenge_count_per_class": _CHALLENGE_COUNT_PER_CLASS,
        "challenge_group_count": _CHALLENGE_GROUP_COUNT,
        "rows_per_challenge_group": _ROWS_PER_CHALLENGE_GROUP,
        "required_correct_rows": _CHALLENGE_COUNT,
        "required_correct_groups": _CHALLENGE_GROUP_COUNT,
        "group_correctness_rule": "all_eight_rows_correct_and_margin_gate",
        "challenge_group_assignment": (
            "qualification_owned_surface_kind_plus_source_position.v1"
        ),
        "top1_expected_required": True,
        "minimum_normalized_margin": 0.01,
        "minimum_normalized_margin_comparator": ">=",
        "tie_policy": "fail",
        "fresh_bge_process_count": 2,
        "fresh_bge_exact_vector_reobservation_required": True,
        "reader_artifact_rederived_from_reobservation_required": True,
        "prediction_ledger_fsync_before_label_release": True,
        "statistical_independence_claim": False,
    }
    if gates != expected_gates:
        raise ValueError("qualification promotion gates drifted")
    execution = _mapping(payload["execution"], "execution")
    _require_exact_keys(
        execution,
        {
            "external_public_anchor_required",
            "prediction_request_surface",
            "future_execution_source_pinning_required",
            "future_transitive_execution_closure_required",
            "qualification_execution_authorized",
            "model_output_count",
            "preflight_source_sha256s",
            "preflight_source_scope",
            "prediction_child_repo_access_allowed",
            "prediction_child_sealed_labels_access_before_ledger_fsync",
            "prediction_process_sees_challenge_labels",
            "process_firewall_security_claim",
            "scoring_process_model_free",
        },
        "execution",
    )
    for field_name in (
        "external_public_anchor_required",
        "future_execution_source_pinning_required",
        "future_transitive_execution_closure_required",
        "qualification_execution_authorized",
        "prediction_child_repo_access_allowed",
        "prediction_child_sealed_labels_access_before_ledger_fsync",
        "prediction_process_sees_challenge_labels",
        "process_firewall_security_claim",
        "scoring_process_model_free",
    ):
        _boolean(execution.get(field_name), f"qualification execution {field_name}")
    if _integer(execution.get("model_output_count"), "model_output_count") != 0:
        raise ValueError("qualification preflight model_output_count must be zero")
    raw_preflight_sources = _mapping(
        execution.get("preflight_source_sha256s"),
        "execution.preflight_source_sha256s",
    )
    if set(raw_preflight_sources) != _preflight_source_paths(schema_version):
        raise ValueError("qualification preflight source pin key set drifted")
    for source_path, source_digest in raw_preflight_sources.items():
        _relative_path(source_path, "qualification preflight source path")
        _digest(source_digest, "qualification preflight source digest")
    if {
        "external_public_anchor_required": execution["external_public_anchor_required"],
        "prediction_request_surface": execution["prediction_request_surface"],
        "future_execution_source_pinning_required": execution[
            "future_execution_source_pinning_required"
        ],
        "future_transitive_execution_closure_required": execution[
            "future_transitive_execution_closure_required"
        ],
        "qualification_execution_authorized": execution[
            "qualification_execution_authorized"
        ],
        "model_output_count": execution["model_output_count"],
        "preflight_source_scope": execution["preflight_source_scope"],
        "prediction_child_repo_access_allowed": execution[
            "prediction_child_repo_access_allowed"
        ],
        "prediction_child_sealed_labels_access_before_ledger_fsync": execution[
            "prediction_child_sealed_labels_access_before_ledger_fsync"
        ],
        "prediction_process_sees_challenge_labels": execution[
            "prediction_process_sees_challenge_labels"
        ],
        "process_firewall_security_claim": execution[
            "process_firewall_security_claim"
        ],
        "scoring_process_model_free": execution["scoring_process_model_free"],
    } != {
        "external_public_anchor_required": True,
        "prediction_request_surface": "opaque_item_id_exact_text_and_text_sha256_only",
        "future_execution_source_pinning_required": True,
        "future_transitive_execution_closure_required": True,
        "qualification_execution_authorized": False,
        "model_output_count": 0,
        "preflight_source_scope": "qualification_preflight_direct_sources_only",
        "prediction_child_repo_access_allowed": False,
        "prediction_child_sealed_labels_access_before_ledger_fsync": False,
        "prediction_process_sees_challenge_labels": False,
        "process_firewall_security_claim": False,
        "scoring_process_model_free": True,
    }:
        raise ValueError("qualification execution boundary drifted")
    claims = _mapping(payload["claims"], "claims")
    for field_name, value in claims.items():
        _boolean(value, f"qualification claim {field_name}")
    expected_claims = {
        "condition_reader_qualified": False,
        "campaign_execution_admitted": False,
        "readable_product_effect": False,
        "four_able_complete": False,
        "formal_evidence_authorized": False,
    }
    if claims != expected_claims:
        raise ValueError("qualification preflight honesty claims drifted")


def _source_pin(payload: Mapping[str, object]) -> QualificationSourcePin:
    return QualificationSourcePin(
        schema_version=_text(payload["schema_version"], "source schema_version"),
        raw_sha256=_digest(payload["raw_sha256"], "source raw_sha256"),
        raw_bytes=_positive_integer(payload["raw_bytes"], "source raw_bytes"),
        protocol_id=_digest(payload["protocol_id"], "source protocol_id"),
        public_plan_id=_digest(payload["public_plan_id"], "source public_plan_id"),
        sealed_bundle_id=_digest(payload["sealed_bundle_id"], "source sealed_bundle_id"),
    )


def _build_manifest(
    *,
    root: pathlib.Path,
    protocol: RelationshipConditionReaderQualificationProtocol,
) -> Mapping[str, object]:
    files = []
    for relative_path in _PREPARED_RELATIVE_PATHS:
        path = root / pathlib.PurePosixPath(relative_path)
        raw = path.read_bytes()
        payload = None if relative_path == "protocol.json" else _load_canonical_artifact(
            path,
            _text(_parse_unique_json(raw.decode("utf-8"), relative_path)["schema_version"], "schema_version"),
        )
        files.append(
            {
                "path": relative_path,
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "raw_bytes": len(raw),
                "artifact_id": None if payload is None else payload["artifact_id"],
            }
        )
    return _with_artifact_id(
        {
            "schema_version": RELATIONSHIP_READER_QUALIFICATION_MANIFEST_SCHEMA_VERSION,
            "protocol_id": protocol.protocol_id,
            "files": files,
            "file_count": len(files),
            "model_output_count": 0,
            "external_public_anchor_created": False,
            "qualification_execution_authorized": False,
        }
    )


def _assert_public_firewall(payload: object) -> None:
    if isinstance(payload, dict):
        leaked = sorted(set(payload) & _PUBLIC_FORBIDDEN_KEYS)
        if leaked:
            raise ValueError(f"qualification predictor surface leaked sealed keys: {leaked}")
        for value in payload.values():
            _assert_public_firewall(value)
    elif isinstance(payload, list):
        for value in payload:
            _assert_public_firewall(value)


def _validate_predictor_request_surface(payload: Mapping[str, object]) -> None:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "protocol_id",
            "public_corpus_artifact_id",
            "challenge_inputs",
            "challenge_input_count",
            "artifact_id",
        },
        "qualification predictor request",
    )
    rows = payload["challenge_inputs"]
    if not isinstance(rows, list):
        raise ValueError("qualification predictor challenge_inputs must be an array")
    if len(rows) != _CHALLENGE_COUNT:
        raise ValueError("qualification predictor request must contain 224 rows")
    for row in rows:
        mapped = _mapping(row, "qualification predictor challenge row")
        _require_exact_keys(
            mapped,
            {"item_id", "text", "text_sha256"},
            "qualification predictor challenge row",
        )
def _with_artifact_id(core: Mapping[str, object]) -> dict[str, object]:
    mutable = dict(core)
    return {**mutable, "artifact_id": sha256_json(mutable)}


def _load_canonical_artifact(path: pathlib.Path, schema_version: str) -> Mapping[str, object]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"qualification artifact is not UTF-8: {path}") from exc
    payload = _parse_unique_json(text, str(path))
    if payload.get("schema_version") != schema_version:
        raise ValueError(f"qualification artifact schema mismatch: {path}")
    artifact_id = _digest(payload.get("artifact_id"), "artifact_id")
    core = {key: value for key, value in payload.items() if key != "artifact_id"}
    if artifact_id != sha256_json(core):
        raise ValueError(f"qualification artifact_id mismatch: {path}")
    if raw != (canonical_json(payload) + "\n").encode("utf-8"):
        raise ValueError(f"qualification artifact is not canonical JSON: {path}")
    return payload


def _write_json_create_only(path: pathlib.Path, payload: Mapping[str, object]) -> None:
    _write_bytes_create_only(path, (canonical_json(payload) + "\n").encode("utf-8"))


def _write_bytes_create_only(path: pathlib.Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _parse_unique_json(text: str, source: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{source} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> object:
        raise ValueError(f"{source} contains non-finite JSON number: {value}")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must be a JSON object")
    return payload


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def _reject_reparse_components(
    path: pathlib.Path,
    *,
    stop: pathlib.Path,
    field_name: str,
) -> None:
    if path != stop and stop not in path.parents:
        raise ValueError(f"{field_name} must remain within repository")
    for candidate in (path, *path.parents):
        if not os.path.lexists(candidate):
            if candidate == stop:
                break
            continue
        if candidate.is_symlink():
            raise ValueError(f"{field_name} traverses symlink: {candidate}")
        if os.name == "nt":
            attributes = os.lstat(candidate).st_file_attributes
            if attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(f"{field_name} traverses reparse point: {candidate}")
        if candidate == stop:
            break


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _relative_path(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    path = pathlib.PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or text != path.as_posix():
        raise ValueError(f"{field_name} must be a canonical relative POSIX path")
    return text


def _digest(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_sha256(value: object, field_name: str) -> None:
    _digest(value, field_name)


def _require_text(value: object, field_name: str) -> None:
    _text(value, field_name)


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _positive_integer(value: object, field_name: str) -> int:
    result = _integer(value, field_name)
    if result < 1:
        raise ValueError(f"{field_name} must be positive")
    return result


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    field_name: str,
) -> None:
    missing = sorted(expected - set(payload))
    extra = sorted(set(payload) - expected)
    if missing or extra:
        raise ValueError(f"{field_name} keys mismatch; missing={missing}, extra={extra}")


__all__ = [
    "OpaqueQualificationTextInput",
    "PreparedQualificationArtifacts",
    "QualificationSourcePin",
    "RELATIONSHIP_READER_QUALIFICATION_PROTOCOL_SCHEMA_VERSION",
    "RelationshipConditionReaderQualificationProtocol",
    "load_relationship_condition_reader_qualification_protocol",
    "prepare_relationship_condition_reader_qualification_preflight",
    "relationship_condition_reader_qualification_protocol_path",
    "validate_relationship_condition_reader_qualification_preflight",
]
