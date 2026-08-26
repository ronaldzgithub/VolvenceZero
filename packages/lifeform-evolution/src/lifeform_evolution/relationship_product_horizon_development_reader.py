"""Development-only frozen reader input for Product Horizon.

This lane deliberately consumes only the four public training texts and their
condition-only labels from the committed v6 preflight.  It does not open the
challenge-label artifact, score held-out rows, or inherit formal qualification
authority.  The output is one content-addressed reader bundle that can be
frozen before any source-v4 campaign outcome exists and replayed without a
model during calibration and campaign execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import pathlib
import re
from typing import Mapping, Protocol

from lifeform_domain_emogpt.lab.contracts import canonical_json, sha256_json
from lifeform_domain_emogpt.relationship_condition_reader import (
    RELATIONSHIP_CONDITION_LINEAR_READER_SCHEMA_VERSION,
    RELATIONSHIP_CONDITION_LINEAR_SOLVER,
    RELATIONSHIP_CONDITION_LINEAR_SOLVER_VERSION,
    FrozenLinearRelationshipConditionReaderArtifact,
    LabeledRelationshipConditionEmbeddingRow,
    build_frozen_linear_relationship_condition_reader_artifact,
)

from lifeform_evolution.relationship_lab_product_model_adapters import (
    BGE_M3_MODEL_ID,
    BGE_M3_MODEL_REVISION,
    BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    BGE_M3_WEIGHT_BYTES_SHA256,
    PrecomputedPublicEmbeddingRecord,
    PrecomputedPublicEmbeddingTable,
    bge_m3_weight_pinned_embedder_identity,
    bge_m3_public_semantic_embedder,
)


DEVELOPMENT_READER_PROTOCOL_SCHEMA_VERSION = (
    "relationship-product-horizon-development-reader-protocol.v1"
)
DEVELOPMENT_READER_TRAINING_SCHEMA_VERSION = (
    "relationship-product-horizon-development-reader-training.v1"
)
DEVELOPMENT_READER_MANIFEST_SCHEMA_VERSION = (
    "relationship-product-horizon-development-reader-manifest.v1"
)

_PROTOCOL_FILENAME = "relationship_product_horizon_development_reader_v1.json"
_EXPECTED_OUTPUT_FILES = frozenset(
    {
        "protocol.json",
        "training_inputs.json",
        "embedding_table.json",
        "reader_artifact.json",
        "manifest.json",
    }
)
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class _PinnedEmbedder(Protocol):
    name: str
    model_source: str
    model_revision: str | None
    weights_sha256: str | None
    sentence_transformers_version: str | None

    def embed(self, text: str) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class RelationshipProductHorizonDevelopmentReaderProtocol:
    payload: Mapping[str, object]
    protocol_id: str
    raw_sha256: str
    raw_bytes: int

    @property
    def training_source(self) -> Mapping[str, object]:
        return _mapping(self.payload["training_source"], "training_source")

    @property
    def semantic_model(self) -> Mapping[str, object]:
        return _mapping(self.payload["semantic_model"], "semantic_model")

    @property
    def campaign_source(self) -> Mapping[str, object]:
        return _mapping(self.payload["campaign_source"], "campaign_source")

    @property
    def reader(self) -> Mapping[str, object]:
        return _mapping(self.payload["reader"], "reader")


def relationship_product_horizon_development_reader_protocol_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "protocols" / _PROTOCOL_FILENAME


def load_relationship_product_horizon_development_reader_protocol(
    path: pathlib.Path | None = None,
) -> RelationshipProductHorizonDevelopmentReaderProtocol:
    source = pathlib.Path(
        path or relationship_product_horizon_development_reader_protocol_path()
    )
    raw = source.read_bytes()
    payload = _parse_json(raw, source=str(source))
    _exact_keys(
        payload,
        {
            "schema_version",
            "evidence_tier",
            "purpose",
            "training_source",
            "campaign_source",
            "semantic_model",
            "reader",
            "claims",
            "claim_boundary",
        },
        "development reader protocol",
    )
    if payload["schema_version"] != DEVELOPMENT_READER_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("development reader protocol schema drifted")
    if payload["evidence_tier"] != "development":
        raise ValueError("development reader evidence tier drifted")
    if payload["purpose"] != (
        "materialize_one_unqualified_frozen_linear_reader_for_"
        "product_horizon_development_inputs"
    ):
        raise ValueError("development reader purpose drifted")

    training = _mapping(payload["training_source"], "training_source")
    _exact_keys(
        training,
        {
            "preflight_protocol_id",
            "preflight_manifest_artifact_id",
            "required_files",
            "group_split_artifact_id_lineage_only",
            "training_item_ids",
            "challenge_label_files_consumed",
            "preflight_label_free_challenge_text_count",
        },
        "development reader training_source",
    )
    _digest(training["preflight_protocol_id"], "preflight_protocol_id")
    _digest(
        training["preflight_manifest_artifact_id"],
        "preflight_manifest_artifact_id",
    )
    _digest(
        training["group_split_artifact_id_lineage_only"],
        "group_split_artifact_id_lineage_only",
    )
    required = _list(training["required_files"], "required_files")
    pins = tuple(_source_pin(item) for item in required)
    if tuple(item["relative_path"] for item in pins) != (
        "public/public_corpus.json",
        "sealed/condition_training_labels.json",
    ):
        raise ValueError("development reader may consume only corpus and training labels")
    item_ids = _text_tuple(training["training_item_ids"], "training_item_ids")
    if len(item_ids) != 4 or len(set(item_ids)) != 4:
        raise ValueError("development reader requires four unique training item ids")
    for item_id in item_ids:
        _digest(item_id, "training item id")
    if training["challenge_label_files_consumed"] != []:
        raise ValueError("development reader cannot consume challenge-label files")
    if training["preflight_label_free_challenge_text_count"] != 224:
        raise ValueError("preflight label-free challenge text count drifted")

    campaign = _mapping(payload["campaign_source"], "campaign_source")
    _exact_keys(
        campaign,
        {
            "admission_protocol_id",
            "admission_artifact_id",
            "source_protocol_id",
            "public_plan_sha256",
            "public_plan_relative_path",
            "public_plan_schema_version",
            "public_plan_raw_sha256",
            "root_count",
            "reader_text_occurrence_count",
            "reader_text_unique_count",
            "combined_public_text_unique_count",
            "sealed_files_consumed",
        },
        "development reader campaign_source",
    )
    for field_name in (
        "admission_protocol_id",
        "admission_artifact_id",
        "source_protocol_id",
        "public_plan_sha256",
        "public_plan_raw_sha256",
    ):
        _digest(campaign[field_name], f"campaign_source.{field_name}")
    if campaign["public_plan_relative_path"] != "public/source_plan.json":
        raise ValueError("campaign public plan path drifted")
    if campaign["public_plan_schema_version"] != (
        "relationship-product-horizon-public-view.v4"
    ):
        raise ValueError("campaign public plan schema drifted")
    expected_counts = (112, 5824, 1881, 2109)
    actual_counts = tuple(
        _integer(campaign[field_name], f"campaign_source.{field_name}")
        for field_name in (
            "root_count",
            "reader_text_occurrence_count",
            "reader_text_unique_count",
            "combined_public_text_unique_count",
        )
    )
    if actual_counts != expected_counts:
        raise ValueError("campaign public reader inventory drifted")
    if campaign["sealed_files_consumed"] != []:
        raise ValueError("development reader cannot consume sealed source-v4 files")

    semantic = _mapping(payload["semantic_model"], "semantic_model")
    expected_semantic = {
        "model_id": BGE_M3_MODEL_ID,
        "model_revision": BGE_M3_MODEL_REVISION,
        "weights_sha256": BGE_M3_WEIGHT_BYTES_SHA256,
        "sentence_transformers_version": BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
        "embedding_width": 1024,
        "device": "cuda",
        "network_allowed": False,
    }
    if semantic != expected_semantic:
        raise ValueError("development reader semantic model identity drifted")

    reader = _mapping(payload["reader"], "reader")
    expected_reader = {
        "schema_version": RELATIONSHIP_CONDITION_LINEAR_READER_SCHEMA_VERSION,
        "solver": RELATIONSHIP_CONDITION_LINEAR_SOLVER,
        "solver_version": RELATIONSHIP_CONDITION_LINEAR_SOLVER_VERSION,
        "labels": ["agency_displacement", "belonging_erasure"],
        "runtime_fit_allowed": False,
    }
    if reader != expected_reader:
        raise ValueError("development reader contract drifted")

    claims = _mapping(payload["claims"], "claims")
    expected_claims = {
        "development_unqualified_reader_input": True,
        "condition_reader_qualified": False,
        "readable_effect": False,
        "campaign_execution_authorized": False,
        "formal_evidence_authorized": False,
        "integrated_horizon_authorized": False,
        "production_active": False,
    }
    if claims != expected_claims:
        raise ValueError("development reader claim boundary drifted")
    _text(payload["claim_boundary"], "claim_boundary")
    return RelationshipProductHorizonDevelopmentReaderProtocol(
        payload=payload,
        protocol_id=sha256_json(payload),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        raw_bytes=len(raw),
    )


def materialize_relationship_product_horizon_development_reader(
    *,
    preflight_root: pathlib.Path,
    source_v4_admission_root: pathlib.Path,
    output_dir: pathlib.Path,
    implementation_git_commit: str,
) -> Mapping[str, object]:
    """Embed all 2,109 unique public inputs and write the development bundle."""

    protocol = load_relationship_product_horizon_development_reader_protocol()
    semantic = protocol.semantic_model
    embedder = bge_m3_public_semantic_embedder(
        device=_text(semantic["device"], "semantic_model.device"),
        model_revision=_text(
            semantic["model_revision"], "semantic_model.model_revision"
        ),
        weights_sha256=_digest(
            semantic["weights_sha256"], "semantic_model.weights_sha256"
        ),
        sentence_transformers_version=_text(
            semantic["sentence_transformers_version"],
            "semantic_model.sentence_transformers_version",
        ),
    )
    return _materialize_with_embedder(
        protocol=protocol,
        preflight_root=pathlib.Path(preflight_root),
        source_v4_admission_root=pathlib.Path(source_v4_admission_root),
        output_dir=pathlib.Path(output_dir),
        implementation_git_commit=implementation_git_commit,
        embedder=embedder,
    )


def _materialize_with_embedder(
    *,
    protocol: RelationshipProductHorizonDevelopmentReaderProtocol,
    preflight_root: pathlib.Path,
    source_v4_admission_root: pathlib.Path,
    output_dir: pathlib.Path,
    implementation_git_commit: str,
    embedder: _PinnedEmbedder,
) -> Mapping[str, object]:
    commit = _git_commit(implementation_git_commit)
    root = pathlib.Path(output_dir).resolve()
    source_root = pathlib.Path(preflight_root).resolve()
    campaign_root = pathlib.Path(source_v4_admission_root).resolve()
    if root.exists():
        raise FileExistsError(f"development reader output already exists: {root}")
    for upstream_root in (source_root, campaign_root):
        if (
            root == upstream_root
            or root in upstream_root.parents
            or upstream_root in root.parents
        ):
            raise ValueError("development reader output and source roots must be disjoint")
    _validate_embedder_identity(embedder, protocol)

    training = _build_training_projection(protocol, source_root)
    public_texts = _build_public_text_inventory(
        protocol=protocol,
        preflight_root=source_root,
        source_v4_admission_root=campaign_root,
    )
    records: list[PrecomputedPublicEmbeddingRecord] = []
    vector_by_text_sha256: dict[str, tuple[float, ...]] = {}
    for text_sha256, text in public_texts:
        vector = _embedding(
            embedder.embed(text),
            expected_width=_integer(
                protocol.semantic_model["embedding_width"],
                "semantic_model.embedding_width",
            ),
        )
        vector_by_text_sha256[text_sha256] = vector
        records.append(
            PrecomputedPublicEmbeddingRecord(
                text=text,
                embedding_hex=tuple(value.hex() for value in vector),
            )
        )
    records.sort(key=lambda item: (item.text_sha256, item.text))
    table = PrecomputedPublicEmbeddingTable(
        source_embedder_name=embedder.name,
        embedding_width=_integer(
            protocol.semantic_model["embedding_width"],
            "semantic_model.embedding_width",
        ),
        records=tuple(records),
    )
    training_raw = _artifact_bytes(training)
    reader = _build_reader(
        protocol=protocol,
        training=training,
        training_raw=training_raw,
        table=table,
        vector_by_text_sha256=vector_by_text_sha256,
    )

    root.mkdir(parents=True)
    protocol_raw = relationship_product_horizon_development_reader_protocol_path().read_bytes()
    outputs = {
        "protocol.json": protocol_raw,
        "training_inputs.json": training_raw,
        "embedding_table.json": table.to_json().encode("utf-8"),
        "reader_artifact.json": reader.to_json_bytes(),
    }
    for relative_path, raw in outputs.items():
        _write_create_only(root / relative_path, raw)
    manifest = _manifest(
        root=root,
        protocol=protocol,
        implementation_git_commit=commit,
        training=training,
        table=table,
        reader=reader,
    )
    _write_create_only(root / "manifest.json", _artifact_bytes(manifest))
    return validate_relationship_product_horizon_development_reader(
        preflight_root=source_root,
        source_v4_admission_root=campaign_root,
        output_dir=root,
        expected_protocol_id=protocol.protocol_id,
        expected_artifact_id=_digest(manifest["artifact_id"], "manifest artifact_id"),
    )


def validate_relationship_product_horizon_development_reader(
    *,
    preflight_root: pathlib.Path,
    source_v4_admission_root: pathlib.Path,
    output_dir: pathlib.Path,
    expected_protocol_id: str,
    expected_artifact_id: str,
) -> Mapping[str, object]:
    """Model-free replay of the training join, table, reader, and manifest."""

    expected_protocol = _digest(expected_protocol_id, "expected_protocol_id")
    expected_artifact = _digest(expected_artifact_id, "expected_artifact_id")
    protocol = load_relationship_product_horizon_development_reader_protocol()
    if protocol.protocol_id != expected_protocol:
        raise ValueError("external expected development reader protocol id mismatch")
    root = pathlib.Path(output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"development reader output does not exist: {root}")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != _EXPECTED_OUTPUT_FILES:
        raise ValueError("development reader output contains missing or extra files")
    if (root / "protocol.json").read_bytes() != (
        relationship_product_horizon_development_reader_protocol_path().read_bytes()
    ):
        raise ValueError("development reader output protocol drifted")

    training = _load_artifact(
        root / "training_inputs.json",
        DEVELOPMENT_READER_TRAINING_SCHEMA_VERSION,
    )
    expected_training = _build_training_projection(
        protocol,
        pathlib.Path(preflight_root).resolve(),
    )
    if training != expected_training:
        raise ValueError("development reader training projection drifted")
    table = PrecomputedPublicEmbeddingTable.from_json(
        (root / "embedding_table.json").read_bytes().decode("utf-8")
    )
    _validate_table(
        protocol=protocol,
        training=training,
        table=table,
        public_texts=_build_public_text_inventory(
            protocol=protocol,
            preflight_root=pathlib.Path(preflight_root).resolve(),
            source_v4_admission_root=pathlib.Path(
                source_v4_admission_root
            ).resolve(),
        ),
    )
    reader = FrozenLinearRelationshipConditionReaderArtifact.from_json(
        (root / "reader_artifact.json").read_bytes()
    )
    vector_by_text_sha256 = {
        item.text_sha256: item.embedding for item in table.records
    }
    rebuilt = _build_reader(
        protocol=protocol,
        training=training,
        training_raw=(root / "training_inputs.json").read_bytes(),
        table=table,
        vector_by_text_sha256=vector_by_text_sha256,
    )
    if reader != rebuilt:
        raise ValueError("development reader artifact differs from model-free rebuild")

    manifest = _load_artifact(
        root / "manifest.json",
        DEVELOPMENT_READER_MANIFEST_SCHEMA_VERSION,
    )
    commit = _git_commit(manifest["implementation_git_commit"])
    expected_manifest = _manifest(
        root=root,
        protocol=protocol,
        implementation_git_commit=commit,
        training=training,
        table=table,
        reader=reader,
    )
    if manifest != expected_manifest:
        raise ValueError("development reader manifest drifted")
    if manifest["artifact_id"] != expected_artifact:
        raise ValueError("external expected development reader artifact id mismatch")
    return manifest


def _build_training_projection(
    protocol: RelationshipProductHorizonDevelopmentReaderProtocol,
    preflight_root: pathlib.Path,
) -> dict[str, object]:
    source_root = pathlib.Path(preflight_root).resolve()
    pins = {
        item["relative_path"]: item
        for item in (
            _source_pin(value)
            for value in _list(
                protocol.training_source["required_files"],
                "required_files",
            )
        )
    }
    corpus = _load_pinned_source(source_root, pins["public/public_corpus.json"])
    labels = _load_pinned_source(
        source_root,
        pins["sealed/condition_training_labels.json"],
    )
    raw_inputs = _list(corpus["training_inputs"], "public corpus training_inputs")
    raw_labels = _list(labels["rows"], "condition training label rows")
    inputs_by_id: dict[str, Mapping[str, object]] = {}
    for value in raw_inputs:
        item = _mapping(value, "public training input")
        item_id = _digest(item["item_id"], "public training item_id")
        if item_id in inputs_by_id:
            raise ValueError("public training item ids must be unique")
        text = _text(item["text"], "public training text")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != item["text_sha256"]:
            raise ValueError("public training text sha256 mismatch")
        inputs_by_id[item_id] = item
    labels_by_id: dict[str, Mapping[str, object]] = {}
    for value in raw_labels:
        item = _mapping(value, "condition training label")
        item_id = _digest(item["item_id"], "condition training item_id")
        if item_id in labels_by_id:
            raise ValueError("condition training item ids must be unique")
        labels_by_id[item_id] = item
    expected_ids = set(
        _text_tuple(protocol.training_source["training_item_ids"], "training_item_ids")
    )
    if set(inputs_by_id) != expected_ids or set(labels_by_id) != expected_ids:
        raise ValueError("development reader training item inventory drifted")

    rows: list[dict[str, object]] = []
    for item_id in expected_ids:
        public = inputs_by_id[item_id]
        label = labels_by_id[item_id]
        if public["text_sha256"] != label["text_sha256"]:
            raise ValueError("training public/label text lineage mismatch")
        rows.append(
            {
                "item_id": item_id,
                "source_position": _integer(
                    label["source_position"], "training source_position"
                ),
                "condition_label": _text(
                    label["condition_label"], "training condition_label"
                ),
                "text": _text(public["text"], "training text"),
                "text_sha256": _digest(
                    public["text_sha256"], "training text_sha256"
                ),
            }
        )
    rows.sort(key=lambda item: _integer(item["source_position"], "source_position"))
    if tuple(item["source_position"] for item in rows) != tuple(range(4)):
        raise ValueError("development reader source positions must be contiguous")
    expected_labels = set(_text_tuple(protocol.reader["labels"], "reader labels"))
    if {item["condition_label"] for item in rows} != expected_labels:
        raise ValueError("development reader training labels drifted")
    core = {
        "schema_version": DEVELOPMENT_READER_TRAINING_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "preflight_protocol_id": protocol.training_source[
            "preflight_protocol_id"
        ],
        "preflight_manifest_artifact_id_lineage_only": protocol.training_source[
            "preflight_manifest_artifact_id"
        ],
        "public_corpus_artifact_id": corpus["artifact_id"],
        "training_labels_artifact_id": labels["artifact_id"],
        "group_split_artifact_id_lineage_only": protocol.training_source[
            "group_split_artifact_id_lineage_only"
        ],
        "challenge_label_file_read_count": 0,
        "rows": rows,
    }
    return _with_artifact_id(core)


def _build_public_text_inventory(
    *,
    protocol: RelationshipProductHorizonDevelopmentReaderProtocol,
    preflight_root: pathlib.Path,
    source_v4_admission_root: pathlib.Path,
) -> tuple[tuple[str, str], ...]:
    pins = {
        item["relative_path"]: item
        for item in (
            _source_pin(value)
            for value in _list(
                protocol.training_source["required_files"],
                "required_files",
            )
        )
    }
    corpus = _load_pinned_source(
        pathlib.Path(preflight_root).resolve(),
        pins["public/public_corpus.json"],
    )
    training_inputs = _list(
        corpus["training_inputs"],
        "public corpus training_inputs",
    )
    calibration_inputs = _list(
        corpus["challenge_inputs"],
        "public corpus challenge_inputs",
    )
    if len(training_inputs) != 4 or len(calibration_inputs) != 224:
        raise ValueError("source-v3 public reader inventory count drifted")

    campaign = protocol.campaign_source
    campaign_root = pathlib.Path(source_v4_admission_root).resolve()
    admission_manifest = _load_artifact(
        campaign_root / "manifest.json",
        "relationship-product-horizon-source-admission-manifest.v1",
    )
    if admission_manifest["protocol_id"] != campaign["admission_protocol_id"]:
        raise ValueError("source-v4 admission protocol identity drifted")
    if admission_manifest["artifact_id"] != campaign["admission_artifact_id"]:
        raise ValueError("source-v4 admission artifact identity drifted")
    if admission_manifest["public_plan_sha256"] != campaign["public_plan_sha256"]:
        raise ValueError("source-v4 admission public plan identity drifted")
    public_plan_path = campaign_root / _text(
        campaign["public_plan_relative_path"],
        "campaign_source.public_plan_relative_path",
    )
    public_plan_raw = public_plan_path.read_bytes()
    if hashlib.sha256(public_plan_raw).hexdigest() != campaign[
        "public_plan_raw_sha256"
    ]:
        raise ValueError("source-v4 public plan raw bytes drifted")
    public_plan = _parse_json(public_plan_raw, source=str(public_plan_path))
    if public_plan.get("schema_version") != campaign["public_plan_schema_version"]:
        raise ValueError("source-v4 public plan schema drifted")
    if public_plan.get("protocol_id") != campaign["source_protocol_id"]:
        raise ValueError("source-v4 source protocol identity drifted")
    if sha256_json(public_plan) != campaign["public_plan_sha256"]:
        raise ValueError("source-v4 public plan canonical identity drifted")
    roots = _list(public_plan["roots"], "source-v4 public roots")
    if len(roots) != campaign["root_count"]:
        raise ValueError("source-v4 root count drifted")

    by_digest: dict[str, str] = {}

    def add_text(text: str, *, expected_sha256: object | None = None) -> None:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if expected_sha256 is not None and digest != _digest(
            expected_sha256,
            "public text sha256",
        ):
            raise ValueError("public reader text sha256 drifted")
        existing = by_digest.get(digest)
        if existing is not None and existing != text:
            raise RuntimeError("sha256 collision in development reader text inventory")
        by_digest[digest] = text

    for value in (*training_inputs, *calibration_inputs):
        item = _mapping(value, "source-v3 public reader input")
        add_text(
            _text(item["text"], "source-v3 public reader text"),
            expected_sha256=item["text_sha256"],
        )

    source_v4_occurrences = 0
    source_v4_unique: set[str] = set()
    for root in roots:
        item = _mapping(root, "source-v4 public root")
        onboarding = _list(
            item["onboarding_sessions"],
            "source-v4 onboarding sessions",
        )
        decisions = _list(
            item["decision_sessions"],
            "source-v4 decision sessions",
        )
        if len(onboarding) != 4 or len(decisions) != 48:
            raise ValueError("source-v4 per-root reader inventory drifted")
        for session in onboarding:
            text = _text(
                _mapping(session, "source-v4 onboarding")["user_utterance"],
                "source-v4 onboarding user_utterance",
            )
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            source_v4_unique.add(digest)
            source_v4_occurrences += 1
            add_text(text)
        for session in decisions:
            text = _text(
                _mapping(session, "source-v4 decision")["current_input"],
                "source-v4 decision current_input",
            )
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            source_v4_unique.add(digest)
            source_v4_occurrences += 1
            add_text(text)
    if source_v4_occurrences != campaign["reader_text_occurrence_count"]:
        raise ValueError("source-v4 reader text occurrence count drifted")
    if len(source_v4_unique) != campaign["reader_text_unique_count"]:
        raise ValueError("source-v4 unique reader text count drifted")
    if len(by_digest) != campaign["combined_public_text_unique_count"]:
        raise ValueError("combined public reader text count drifted")
    return tuple(sorted(by_digest.items()))


def _build_reader(
    *,
    protocol: RelationshipProductHorizonDevelopmentReaderProtocol,
    training: Mapping[str, object],
    training_raw: bytes,
    table: PrecomputedPublicEmbeddingTable,
    vector_by_text_sha256: Mapping[str, tuple[float, ...]],
) -> FrozenLinearRelationshipConditionReaderArtifact:
    rows = tuple(
        LabeledRelationshipConditionEmbeddingRow(
            example_id=_digest(item["item_id"], "training item_id"),
            condition_label=_text(
                item["condition_label"], "training condition_label"
            ),
            embedding_hex=tuple(
                value.hex()
                for value in vector_by_text_sha256[
                    _digest(item["text_sha256"], "training text_sha256")
                ]
            ),
        )
        for item in (
            _mapping(value, "training row")
            for value in _list(training["rows"], "training rows")
        )
    )
    return build_frozen_linear_relationship_condition_reader_artifact(
        embedding_model_id=table.source_model_id,
        embedding_model_revision=_text(
            table.source_model_revision,
            "embedding table source_model_revision",
        ),
        embedding_weights_sha256=_digest(
            table.source_weights_sha256,
            "embedding table source_weights_sha256",
        ),
        embedding_runtime_version=_text(
            table.source_sentence_transformers_version,
            "embedding table runtime version",
        ),
        embedding_width=table.embedding_width,
        labels=_text_tuple(protocol.reader["labels"], "reader labels"),
        condition_training_corpus_artifact_id=_digest(
            training["artifact_id"], "training artifact_id"
        ),
        condition_training_corpus_raw_sha256=hashlib.sha256(training_raw).hexdigest(),
        group_split_artifact_id=_digest(
            protocol.training_source["group_split_artifact_id_lineage_only"],
            "group_split_artifact_id_lineage_only",
        ),
        rows=rows,
    )


def _validate_table(
    *,
    protocol: RelationshipProductHorizonDevelopmentReaderProtocol,
    training: Mapping[str, object],
    table: PrecomputedPublicEmbeddingTable,
    public_texts: tuple[tuple[str, str], ...],
) -> None:
    semantic = protocol.semantic_model
    identity = (
        table.source_model_id,
        table.source_model_revision,
        table.source_weights_sha256,
        table.source_sentence_transformers_version,
        table.embedding_width,
    )
    expected = (
        semantic["model_id"],
        semantic["model_revision"],
        semantic["weights_sha256"],
        semantic["sentence_transformers_version"],
        semantic["embedding_width"],
    )
    if identity != expected:
        raise ValueError("development reader embedding table identity drifted")
    expected_texts = {text for _digest_value, text in public_texts}
    if {item.text for item in table.records} != expected_texts:
        raise ValueError("development reader embedding table text inventory drifted")
    training_texts = {
        _text(item["text"], "training text")
        for item in (
            _mapping(value, "training row")
            for value in _list(training["rows"], "training rows")
        )
    }
    if not training_texts.issubset(expected_texts):
        raise ValueError("development reader table lost a training text")


def _validate_embedder_identity(
    embedder: _PinnedEmbedder,
    protocol: RelationshipProductHorizonDevelopmentReaderProtocol,
) -> None:
    semantic = protocol.semantic_model
    identity = (
        embedder.model_source,
        embedder.model_revision,
        embedder.weights_sha256,
        embedder.sentence_transformers_version,
    )
    expected = (
        semantic["model_id"],
        semantic["model_revision"],
        semantic["weights_sha256"],
        semantic["sentence_transformers_version"],
    )
    if identity != expected:
        raise ValueError("development reader embedder identity drifted")
    expected_name = bge_m3_weight_pinned_embedder_identity(
        model_revision=_text(
            semantic["model_revision"],
            "semantic_model.model_revision",
        ),
        weights_sha256=_digest(
            semantic["weights_sha256"],
            "semantic_model.weights_sha256",
        ),
        sentence_transformers_version=_text(
            semantic["sentence_transformers_version"],
            "semantic_model.sentence_transformers_version",
        ),
        identity_kind="model-adapter-v2",
    )
    if embedder.name != expected_name:
        raise ValueError("development reader embedder canonical name drifted")


def _manifest(
    *,
    root: pathlib.Path,
    protocol: RelationshipProductHorizonDevelopmentReaderProtocol,
    implementation_git_commit: str,
    training: Mapping[str, object],
    table: PrecomputedPublicEmbeddingTable,
    reader: FrozenLinearRelationshipConditionReaderArtifact,
) -> dict[str, object]:
    files = []
    for relative_path in sorted(_EXPECTED_OUTPUT_FILES - {"manifest.json"}):
        raw = (root / relative_path).read_bytes()
        files.append(
            {
                "path": relative_path,
                "raw_bytes": len(raw),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    core = {
        "schema_version": DEVELOPMENT_READER_MANIFEST_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "protocol_raw_sha256": protocol.raw_sha256,
        "implementation_git_commit": implementation_git_commit,
        "preflight_protocol_id": protocol.training_source[
            "preflight_protocol_id"
        ],
        "preflight_manifest_artifact_id_lineage_only": protocol.training_source[
            "preflight_manifest_artifact_id"
        ],
        "training_inputs_artifact_id": training["artifact_id"],
        "embedding_table_artifact_id": table.artifact_id,
        "reader_artifact_id": reader.artifact_id,
        "training_input_count": 4,
        "preflight_label_free_challenge_text_count": protocol.training_source[
            "preflight_label_free_challenge_text_count"
        ],
        "source_v4_reader_text_occurrence_count": protocol.campaign_source[
            "reader_text_occurrence_count"
        ],
        "source_v4_reader_text_unique_count": protocol.campaign_source[
            "reader_text_unique_count"
        ],
        "embedding_table_record_count": len(table.records),
        "embedding_call_count": len(table.records),
        "challenge_label_file_read_count": 0,
        "source_v4_sealed_file_read_count": 0,
        "requested_device": protocol.semantic_model["device"],
        "files": files,
        "status": "development_unqualified_reader_materialized",
        "claims": protocol.payload["claims"],
        "claim_boundary": protocol.payload["claim_boundary"],
    }
    return _with_artifact_id(core)


def _load_pinned_source(
    root: pathlib.Path,
    pin: Mapping[str, object],
) -> Mapping[str, object]:
    path = root / _text(pin["relative_path"], "source relative_path")
    if not path.is_file():
        raise FileNotFoundError(f"development reader source file is missing: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["raw_sha256"]:
        raise ValueError(f"development reader source raw bytes drifted: {path}")
    payload = _parse_json(raw, source=str(path))
    if payload.get("schema_version") != pin["schema_version"]:
        raise ValueError(f"development reader source schema drifted: {path}")
    if payload.get("artifact_id") != pin["artifact_id"]:
        raise ValueError(f"development reader source artifact id drifted: {path}")
    _validate_artifact_id(payload, source=str(path))
    return payload


def _source_pin(value: object) -> Mapping[str, object]:
    pin = _mapping(value, "source pin")
    _exact_keys(
        pin,
        {"relative_path", "schema_version", "artifact_id", "raw_sha256"},
        "development reader source pin",
    )
    _text(pin["relative_path"], "source pin relative_path")
    _text(pin["schema_version"], "source pin schema_version")
    _digest(pin["artifact_id"], "source pin artifact_id")
    _digest(pin["raw_sha256"], "source pin raw_sha256")
    return pin


def _embedding(value: object, *, expected_width: int) -> tuple[float, ...]:
    if not isinstance(value, tuple) or len(value) != expected_width:
        raise ValueError("development reader embedding width drifted")
    vector = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in vector):
        raise ValueError("development reader embedding must be finite")
    if math.sqrt(math.fsum(item * item for item in vector)) <= 1e-12:
        raise ValueError("development reader embedding norm must be positive")
    return vector


def _load_artifact(path: pathlib.Path, schema_version: str) -> dict[str, object]:
    payload = _parse_json(path.read_bytes(), source=str(path))
    if payload.get("schema_version") != schema_version:
        raise ValueError(f"artifact schema drifted: {path}")
    _validate_artifact_id(payload, source=str(path))
    if path.read_bytes() != _artifact_bytes(payload):
        raise ValueError(f"artifact is not canonical LF-terminated JSON: {path}")
    return payload


def _validate_artifact_id(payload: Mapping[str, object], *, source: str) -> None:
    artifact_id = _digest(payload.get("artifact_id"), f"{source} artifact_id")
    core = {key: value for key, value in payload.items() if key != "artifact_id"}
    if artifact_id != sha256_json(core):
        raise ValueError(f"content-addressed artifact id mismatch: {source}")


def _with_artifact_id(core: Mapping[str, object]) -> dict[str, object]:
    return {"artifact_id": sha256_json(core), **core}


def _artifact_bytes(payload: Mapping[str, object]) -> bytes:
    return (canonical_json(payload) + "\n").encode("utf-8")


def _write_create_only(path: pathlib.Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != raw:
        raise RuntimeError(f"development reader create-only readback failed: {path}")


def _parse_json(raw: bytes, *, source: str) -> dict[str, object]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"JSON must be UTF-8: {source}") from exc

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {source}: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> object:
        raise ValueError(f"non-finite JSON number in {source}: {value}")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return payload


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    return tuple(
        _text(item, f"{field_name}[{index}]")
        for index, item in enumerate(_list(value, field_name))
    )


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _digest(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return text


def _git_commit(value: object) -> str:
    commit = _text(value, "implementation_git_commit")
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("implementation_git_commit must be a full lowercase git commit")
    return commit


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    source: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{source} keys drifted: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


__all__ = [
    "DEVELOPMENT_READER_MANIFEST_SCHEMA_VERSION",
    "DEVELOPMENT_READER_PROTOCOL_SCHEMA_VERSION",
    "DEVELOPMENT_READER_TRAINING_SCHEMA_VERSION",
    "RelationshipProductHorizonDevelopmentReaderProtocol",
    "load_relationship_product_horizon_development_reader_protocol",
    "materialize_relationship_product_horizon_development_reader",
    "relationship_product_horizon_development_reader_protocol_path",
    "validate_relationship_product_horizon_development_reader",
]
