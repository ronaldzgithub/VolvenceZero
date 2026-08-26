"""Fresh-process predictor for the frozen relationship-condition reader.

The worker consumes only one externally pinned child request, four labelled
development texts, and the existing opaque 224-row predictor request.  It does
not import the qualification preflight, source generator, campaign executor,
or challenge labels.  Every input is strict canonical JSON and every output is
written create-only, flushed, fsynced, and read back before success is
reported.

The public entry point always constructs the pinned BGE-M3 adapter on CUDA.
The private core accepts an embedder factory solely so model-free unit tests can
exercise the complete artifact path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
import pathlib
import platform
import sys
from typing import Callable, Mapping, Protocol

from lifeform_domain_emogpt.relationship_condition_reader import (
    RELATIONSHIP_CONDITION_LINEAR_READER_SCHEMA_VERSION,
    RELATIONSHIP_CONDITION_LINEAR_SOLVER,
    RELATIONSHIP_CONDITION_LINEAR_SOLVER_VERSION,
    FrozenLinearRelationshipConditionReaderArtifact,
    FrozenLinearRelationshipConditionReaderRuntime,
    LabeledRelationshipConditionEmbeddingRow,
    build_frozen_linear_relationship_condition_reader_artifact,
)
from volvence_zero.social_cognition import (
    relationship_condition_readout_from_payload,
    relationship_condition_readout_to_payload,
)

from lifeform_evolution.relationship_lab_product_model_adapters import (
    BGE_M3_MODEL_ID,
    BGE_M3_MODEL_REVISION,
    BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
    BGE_M3_WEIGHT_BYTES_SHA256,
    PrecomputedPublicEmbeddingRecord,
    PrecomputedPublicEmbeddingTable,
    PrecomputedPublicSemanticEmbedder,
    bge_m3_public_semantic_embedder,
)
from lifeform_evolution.relationship_condition_reader_qualification_runtime_binding import (
    snapshot_file_backed_module_origins,
)


RELATIONSHIP_READER_PREDICTION_CHILD_REQUEST_SCHEMA_VERSION = (
    "relationship-condition-reader-prediction-child-request.v1"
)
RELATIONSHIP_READER_PREDICTION_TRAINING_CORPUS_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-training-corpus.v1"
)
RELATIONSHIP_READER_PREDICTION_LEDGER_SCHEMA_VERSION = "relationship-condition-reader-prediction-ledger.v1"
RELATIONSHIP_READER_PREDICTION_PROCESS_ATTESTATION_SCHEMA_VERSION = (
    "relationship-condition-reader-prediction-process-attestation.v5"
)
RELATIONSHIP_READER_PREDICTION_MANIFEST_SCHEMA_VERSION = "relationship-condition-reader-prediction-manifest.v1"
RELATIONSHIP_READER_PREDICTOR_REQUEST_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-predictor-request.v1"
)

_TRAINING_COUNT = 4
_CHALLENGE_COUNT = 224
_LIVE_EMBEDDING_COUNT = _TRAINING_COUNT + _CHALLENGE_COUNT
_FORMAL_EMBEDDING_WIDTH = 1024
_OUTPUT_PATHS = (
    "embedding_table.json",
    "reader_artifact.json",
    "prediction_ledger.json",
    "process_attestation.json",
    "manifest.json",
)
_DETERMINISTIC_OUTPUT_PATHS = _OUTPUT_PATHS[:3]
_FORBIDDEN_MODULE_NAMES = (
    "lifeform_domain_emogpt.lab.relationship_product_pilot_source",
    "lifeform_evolution.relationship_condition_reader_qualification",
    "lifeform_evolution.relationship_lab_product_horizon",
)
_ENVIRONMENT_PROJECTION_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HUB_OFFLINE",
    "KMP_DUPLICATE_LIB_OK",
    "KMP_INIT_AT_FORK",
    "PYTHONHASHSEED",
    "PYTHONPATH",
    "PYTHONPYCACHEPREFIX",
    "PYTHONSAFEPATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUTF8",
    "TOKENIZERS_PARALLELISM",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRANSFORMERS_OFFLINE",
)


class _FrozenQualificationEmbedder(Protocol):
    name: str
    model_source: str
    model_revision: str
    weights_sha256: str
    sentence_transformers_version: str

    def embed(self, text: str) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class _SemanticModelSpec:
    model_id: str
    model_revision: str
    weights_sha256: str
    sentence_transformers_version: str
    embedding_width: int
    device: str
    network_allowed: bool
    stub_allowed: bool


@dataclass(frozen=True)
class _ChildRequest:
    protocol_id: str
    execution_protocol_id: str
    artifact_id: str
    public_corpus_artifact_id: str
    training_corpus_artifact_id: str
    training_corpus_raw_sha256: str
    training_corpus_raw_bytes: int
    predictor_request_artifact_id: str
    predictor_request_raw_sha256: str
    predictor_request_raw_bytes: int
    group_split_artifact_id: str
    labels: tuple[str, ...]
    semantic_model: _SemanticModelSpec


@dataclass(frozen=True)
class _OpaqueTextRow:
    item_id: str
    text: str
    text_sha256: str
    condition_label: str | None = None


@dataclass(frozen=True)
class _LoadedArtifact:
    payload: Mapping[str, object]
    raw: bytes
    raw_sha256: str


EmbedderFactory = Callable[[_SemanticModelSpec], _FrozenQualificationEmbedder]


def run_relationship_condition_reader_prediction_child(
    *,
    child_request_path: pathlib.Path,
    expected_child_request_artifact_id: str,
    training_corpus_path: pathlib.Path,
    predictor_request_path: pathlib.Path,
    output_root: pathlib.Path,
    run_ordinal: int,
    run_nonce: str,
    bge_snapshot_path: pathlib.Path | None = None,
) -> Mapping[str, object]:
    """Run the formal pinned BGE-M3 CUDA prediction child.

    This is the only supported production/CLI entry point.  It intentionally
    has no model-factory parameter, so test doubles cannot enter a formal run.
    """

    loaded_at_entry = _forbidden_module_presence()
    if any(loaded_at_entry.values()):
        present = sorted(name for name, loaded in loaded_at_entry.items() if loaded)
        raise RuntimeError(f"formal prediction child must start without forbidden modules: {present}")
    snapshot_path = None
    if bge_snapshot_path is not None:
        snapshot_path = pathlib.Path(bge_snapshot_path)

    def formal_factory(spec: _SemanticModelSpec) -> _FrozenQualificationEmbedder:
        return bge_m3_public_semantic_embedder(
            device="cuda",
            model_revision=spec.model_revision,
            weights_sha256=spec.weights_sha256,
            sentence_transformers_version=spec.sentence_transformers_version,
            snapshot_path=snapshot_path,
        )

    return _run_relationship_condition_reader_prediction_child_core(
        child_request_path=child_request_path,
        expected_child_request_artifact_id=expected_child_request_artifact_id,
        training_corpus_path=training_corpus_path,
        predictor_request_path=predictor_request_path,
        output_root=output_root,
        run_ordinal=run_ordinal,
        run_nonce=run_nonce,
        embedder_factory=formal_factory,
        require_formal_bge_cuda=True,
        forbidden_at_entry=loaded_at_entry,
    )


def _run_relationship_condition_reader_prediction_child_core(
    *,
    child_request_path: pathlib.Path,
    expected_child_request_artifact_id: str,
    training_corpus_path: pathlib.Path,
    predictor_request_path: pathlib.Path,
    output_root: pathlib.Path,
    run_ordinal: int,
    run_nonce: str,
    embedder_factory: EmbedderFactory,
    require_formal_bge_cuda: bool = False,
    forbidden_at_entry: Mapping[str, bool] | None = None,
) -> Mapping[str, object]:
    """Execute one child; dependency injection is private and test-only."""

    _require_sha256(expected_child_request_artifact_id, "expected child request id")
    _require_sha256(run_nonce, "run_nonce")
    if isinstance(run_ordinal, bool) or run_ordinal not in {1, 2}:
        raise ValueError("run_ordinal must be exactly 1 or 2")
    if not callable(embedder_factory):
        raise TypeError("embedder_factory must be callable")

    root = pathlib.Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(f"prediction child output root already exists: {root}")
    entry_observation = dict(forbidden_at_entry if forbidden_at_entry is not None else _forbidden_module_presence())

    child_loaded = _load_canonical_artifact(
        pathlib.Path(child_request_path),
        RELATIONSHIP_READER_PREDICTION_CHILD_REQUEST_SCHEMA_VERSION,
    )
    child_request = _parse_child_request(child_loaded.payload)
    if child_request.artifact_id != expected_child_request_artifact_id:
        raise ValueError("external expected child request artifact id mismatch")
    if require_formal_bge_cuda:
        _validate_formal_bge_cuda(child_request.semantic_model)

    training_loaded = _load_canonical_artifact(
        pathlib.Path(training_corpus_path),
        RELATIONSHIP_READER_PREDICTION_TRAINING_CORPUS_SCHEMA_VERSION,
    )
    predictor_loaded = _load_canonical_artifact(
        pathlib.Path(predictor_request_path),
        RELATIONSHIP_READER_PREDICTOR_REQUEST_SCHEMA_VERSION,
    )
    _validate_source_file_pin(
        training_loaded,
        expected_artifact_id=child_request.training_corpus_artifact_id,
        expected_raw_sha256=child_request.training_corpus_raw_sha256,
        expected_raw_bytes=child_request.training_corpus_raw_bytes,
        field_name="training corpus",
    )
    _validate_source_file_pin(
        predictor_loaded,
        expected_artifact_id=child_request.predictor_request_artifact_id,
        expected_raw_sha256=child_request.predictor_request_raw_sha256,
        expected_raw_bytes=child_request.predictor_request_raw_bytes,
        field_name="predictor request",
    )
    training_rows = _parse_training_corpus(
        training_loaded.payload,
        request=child_request,
    )
    challenge_rows = _parse_predictor_request(
        predictor_loaded.payload,
        request=child_request,
    )
    _validate_joint_inputs(training_rows, challenge_rows)

    embedder = embedder_factory(child_request.semantic_model)
    _validate_embedder_identity(embedder, child_request.semantic_model)
    table, live_embedding_count = _build_live_embedding_table(
        embedder=embedder,
        model=child_request.semantic_model,
        rows=tuple(sorted((*training_rows, *challenge_rows), key=lambda row: row.item_id)),
    )
    if live_embedding_count != _LIVE_EMBEDDING_COUNT:
        raise RuntimeError("prediction child did not perform exactly 228 live embeddings")

    records_by_text_sha256 = {record.text_sha256: record for record in table.records}
    labelled_rows = tuple(
        LabeledRelationshipConditionEmbeddingRow(
            example_id=row.item_id,
            condition_label=_required_label(row),
            embedding_hex=records_by_text_sha256[row.text_sha256].embedding_hex,
        )
        for row in training_rows
    )
    reader_artifact = build_frozen_linear_relationship_condition_reader_artifact(
        embedding_model_id=child_request.semantic_model.model_id,
        embedding_model_revision=child_request.semantic_model.model_revision,
        embedding_weights_sha256=child_request.semantic_model.weights_sha256,
        embedding_runtime_version=(child_request.semantic_model.sentence_transformers_version),
        embedding_width=child_request.semantic_model.embedding_width,
        labels=child_request.labels,
        condition_training_corpus_artifact_id=(child_request.training_corpus_artifact_id),
        condition_training_corpus_raw_sha256=(child_request.training_corpus_raw_sha256),
        group_split_artifact_id=child_request.group_split_artifact_id,
        rows=labelled_rows,
    )
    frozen_embedder = PrecomputedPublicSemanticEmbedder(table)
    reader_runtime = FrozenLinearRelationshipConditionReaderRuntime(
        artifact=reader_artifact,
        embedder=frozen_embedder,
    )
    prediction_ledger = _build_prediction_ledger(
        request=child_request,
        predictor_request_artifact_id=child_request.predictor_request_artifact_id,
        embedding_table=table,
        reader_artifact=reader_artifact,
        reader_runtime=reader_runtime,
        challenge_rows=challenge_rows,
    )

    exit_observation = _forbidden_module_presence()
    imported_by_worker = {
        name: bool(exit_observation[name] and not entry_observation.get(name, False))
        for name in _FORBIDDEN_MODULE_NAMES
    }
    if any(imported_by_worker.values()):
        imported = sorted(name for name, loaded in imported_by_worker.items() if loaded)
        raise RuntimeError(f"prediction child imported forbidden modules: {imported}")
    if require_formal_bge_cuda and any(exit_observation.values()):
        present = sorted(name for name, loaded in exit_observation.items() if loaded)
        raise RuntimeError(f"formal prediction child observed forbidden modules before output: {present}")

    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    deterministic = _write_deterministic_outputs(
        root=root,
        table=table,
        reader_artifact=reader_artifact,
        prediction_ledger=prediction_ledger,
    )
    attestation = _with_artifact_id(
        {
            "schema_version": (RELATIONSHIP_READER_PREDICTION_PROCESS_ATTESTATION_SCHEMA_VERSION),
            "protocol_id": child_request.protocol_id,
            "execution_protocol_id": child_request.execution_protocol_id,
            "child_request_artifact_id": child_request.artifact_id,
            "run_ordinal": run_ordinal,
            "run_nonce": run_nonce,
            "process_id": os.getpid(),
            "parent_process_id": os.getppid(),
            "python_executable": sys.executable,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "argv": list(sys.argv),
            "interpreter_flags": {
                "safe_path": sys.flags.safe_path,
                "no_site": sys.flags.no_site,
                "dont_write_bytecode": sys.flags.dont_write_bytecode,
                "utf8_mode": sys.flags.utf8_mode,
                "isolated": sys.flags.isolated,
                "ignore_environment": sys.flags.ignore_environment,
                "stdout_write_through": bool(sys.stdout.write_through),
                "stderr_write_through": bool(sys.stderr.write_through),
            },
            "pycache_prefix": sys.pycache_prefix,
            "working_directory": os.getcwd(),
            "sys_path": list(sys.path),
            "bootstrap_import_roots": _bootstrap_import_roots(),
            "environment_contract": {
                "schema_version": ("relationship-condition-reader-prediction-environment.v4"),
                "projected_keys": list(_ENVIRONMENT_PROJECTION_KEYS),
                "all_environment_values_hashed": True,
                "unlisted_environment_variables_recorded": True,
                "key_name_canonicalization": "windows_uppercase",
                "complete_environment_observation_scope": "cpython_visible_mapping",
                "raw_win32_environment_block_attested": False,
            },
            "environment_projection": {key: os.environ.get(key) for key in _ENVIRONMENT_PROJECTION_KEYS},
            "environment_key_names": sorted(os.environ),
            "environment_value_sha256s": {
                key: hashlib.sha256(value.encode("utf-8")).hexdigest() for key, value in sorted(os.environ.items())
            },
            "loaded_file_backed_module_origins": snapshot_file_backed_module_origins(sys.modules),
            "volvence_zero_namespace_search_locations": (_volvence_zero_namespace_search_locations()),
            "embedder_factory_kind": ("formal_bge_m3_cuda" if require_formal_bge_cuda else "test_injected_nonformal"),
            "model": _semantic_model_payload(child_request.semantic_model),
            "live_embedding_call_count": live_embedding_count,
            "training_embedding_count": _TRAINING_COUNT,
            "challenge_embedding_count": _CHALLENGE_COUNT,
            "prediction_ledger_fsync_completed": True,
            "forbidden_module_observations": [
                {
                    "module_name": name,
                    "loaded_at_worker_entry": bool(entry_observation.get(name, False)),
                    "loaded_at_worker_exit": exit_observation[name],
                    "imported_by_worker": imported_by_worker[name],
                }
                for name in _FORBIDDEN_MODULE_NAMES
            ],
            "deterministic_outputs": list(deterministic),
            "os_security_boundary": False,
        }
    )
    attestation_raw = _canonical_artifact_bytes(attestation)
    _write_bytes_create_only(root / "process_attestation.json", attestation_raw)
    loaded_attestation = _load_canonical_artifact(
        root / "process_attestation.json",
        RELATIONSHIP_READER_PREDICTION_PROCESS_ATTESTATION_SCHEMA_VERSION,
    )
    if loaded_attestation.payload != attestation:
        raise RuntimeError("process attestation failed canonical readback")

    manifest_entries = (
        *deterministic,
        _file_receipt(
            path="process_attestation.json",
            raw=attestation_raw,
            artifact_id=_artifact_id(attestation),
        ),
    )
    manifest = _with_artifact_id(
        {
            "schema_version": RELATIONSHIP_READER_PREDICTION_MANIFEST_SCHEMA_VERSION,
            "protocol_id": child_request.protocol_id,
            "execution_protocol_id": child_request.execution_protocol_id,
            "child_request_artifact_id": child_request.artifact_id,
            "files": list(manifest_entries),
            "file_count": len(manifest_entries),
            "deterministic_file_paths": list(_DETERMINISTIC_OUTPUT_PATHS),
            "prediction_ledger_fsync_completed": True,
        }
    )
    manifest_raw = _canonical_artifact_bytes(manifest)
    _write_bytes_create_only(root / "manifest.json", manifest_raw)
    loaded_manifest = _load_canonical_artifact(
        root / "manifest.json",
        RELATIONSHIP_READER_PREDICTION_MANIFEST_SCHEMA_VERSION,
    )
    if loaded_manifest.payload != manifest:
        raise RuntimeError("prediction child manifest failed canonical readback")
    actual_paths = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_paths != set(_OUTPUT_PATHS):
        raise RuntimeError("prediction child output root has missing or extra files")

    return {
        "schema_version": "relationship-condition-reader-prediction-child-result.v1",
        "protocol_id": child_request.protocol_id,
        "execution_protocol_id": child_request.execution_protocol_id,
        "child_request_artifact_id": child_request.artifact_id,
        "embedding_table_artifact_id": table.artifact_id,
        "reader_artifact_id": reader_artifact.artifact_id,
        "prediction_ledger_artifact_id": _artifact_id(prediction_ledger),
        "process_attestation_artifact_id": _artifact_id(attestation),
        "manifest_artifact_id": _artifact_id(manifest),
        "live_embedding_call_count": live_embedding_count,
        "prediction_count": len(challenge_rows),
        "run_ordinal": run_ordinal,
        "run_nonce": run_nonce,
        "process_id": os.getpid(),
        "output_root": str(root),
        "formal_bge_cuda": require_formal_bge_cuda,
        "challenge_labels_used": False,
        "qualification_scored": False,
    }


def _parse_child_request(payload: Mapping[str, object]) -> _ChildRequest:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "protocol_id",
            "execution_protocol_id",
            "public_corpus_artifact_id",
            "training_corpus_artifact_id",
            "training_corpus_raw_sha256",
            "training_corpus_raw_bytes",
            "predictor_request_artifact_id",
            "predictor_request_raw_sha256",
            "predictor_request_raw_bytes",
            "group_split_artifact_id",
            "semantic_model",
            "reader",
            "required_live_embedding_count",
            "artifact_id",
        },
        "prediction child request",
    )
    if payload["schema_version"] != RELATIONSHIP_READER_PREDICTION_CHILD_REQUEST_SCHEMA_VERSION:
        raise ValueError("prediction child request schema mismatch")
    if (
        _positive_integer(
            payload["required_live_embedding_count"],
            "required_live_embedding_count",
        )
        != _LIVE_EMBEDDING_COUNT
    ):
        raise ValueError("prediction child request must require 228 live embeddings")
    model_payload = _mapping(payload["semantic_model"], "semantic_model")
    _require_exact_keys(
        model_payload,
        {
            "model_id",
            "model_revision",
            "weights_sha256",
            "sentence_transformers_version",
            "embedding_width",
            "device",
            "network_allowed",
            "stub_allowed",
        },
        "semantic_model",
    )
    if _boolean(model_payload["network_allowed"], "network_allowed"):
        raise ValueError("prediction child semantic model must forbid network access")
    if _boolean(model_payload["stub_allowed"], "stub_allowed"):
        raise ValueError("prediction child semantic model must forbid stubs")
    model = _SemanticModelSpec(
        model_id=_text(model_payload["model_id"], "model_id"),
        model_revision=_text(model_payload["model_revision"], "model_revision"),
        weights_sha256=_digest(model_payload["weights_sha256"], "weights_sha256"),
        sentence_transformers_version=_text(
            model_payload["sentence_transformers_version"],
            "sentence_transformers_version",
        ),
        embedding_width=_positive_integer(
            model_payload["embedding_width"],
            "embedding_width",
        ),
        device=_text(model_payload["device"], "device"),
        network_allowed=False,
        stub_allowed=False,
    )
    if model.embedding_width < 2:
        raise ValueError("prediction child embedding_width must be at least two")

    reader_payload = _mapping(payload["reader"], "reader")
    _require_exact_keys(
        reader_payload,
        {"schema_version", "solver", "solver_version", "labels"},
        "reader",
    )
    if reader_payload["schema_version"] != RELATIONSHIP_CONDITION_LINEAR_READER_SCHEMA_VERSION:
        raise ValueError("prediction child reader schema mismatch")
    if reader_payload["solver"] != RELATIONSHIP_CONDITION_LINEAR_SOLVER:
        raise ValueError("prediction child reader solver mismatch")
    if reader_payload["solver_version"] != RELATIONSHIP_CONDITION_LINEAR_SOLVER_VERSION:
        raise ValueError("prediction child reader solver version mismatch")
    labels = _labels(reader_payload["labels"], "reader.labels")
    artifact_id = _digest(payload["artifact_id"], "artifact_id")
    if artifact_id != _artifact_id(payload):
        raise ValueError("prediction child request artifact_id mismatch")
    return _ChildRequest(
        protocol_id=_digest(payload["protocol_id"], "protocol_id"),
        execution_protocol_id=_digest(
            payload["execution_protocol_id"],
            "execution_protocol_id",
        ),
        artifact_id=artifact_id,
        public_corpus_artifact_id=_digest(
            payload["public_corpus_artifact_id"],
            "public_corpus_artifact_id",
        ),
        training_corpus_artifact_id=_digest(
            payload["training_corpus_artifact_id"],
            "training_corpus_artifact_id",
        ),
        training_corpus_raw_sha256=_digest(
            payload["training_corpus_raw_sha256"],
            "training_corpus_raw_sha256",
        ),
        training_corpus_raw_bytes=_positive_integer(
            payload["training_corpus_raw_bytes"],
            "training_corpus_raw_bytes",
        ),
        predictor_request_artifact_id=_digest(
            payload["predictor_request_artifact_id"],
            "predictor_request_artifact_id",
        ),
        predictor_request_raw_sha256=_digest(
            payload["predictor_request_raw_sha256"],
            "predictor_request_raw_sha256",
        ),
        predictor_request_raw_bytes=_positive_integer(
            payload["predictor_request_raw_bytes"],
            "predictor_request_raw_bytes",
        ),
        group_split_artifact_id=_digest(
            payload["group_split_artifact_id"],
            "group_split_artifact_id",
        ),
        labels=labels,
        semantic_model=model,
    )


def _parse_training_corpus(
    payload: Mapping[str, object],
    *,
    request: _ChildRequest,
) -> tuple[_OpaqueTextRow, ...]:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "protocol_id",
            "public_corpus_artifact_id",
            "labels",
            "rows",
            "row_count",
            "condition_only",
            "artifact_id",
        },
        "training corpus",
    )
    if payload["protocol_id"] != request.protocol_id:
        raise ValueError("training corpus protocol_id mismatch")
    if payload["public_corpus_artifact_id"] != request.public_corpus_artifact_id:
        raise ValueError("training corpus public_corpus_artifact_id mismatch")
    if _labels(payload["labels"], "training corpus labels") != request.labels:
        raise ValueError("training corpus labels mismatch")
    if _boolean(payload["condition_only"], "condition_only") is not True:
        raise ValueError("training corpus must be condition-only")
    rows = _list(payload["rows"], "training corpus rows")
    if _positive_integer(payload["row_count"], "training corpus row_count") != len(rows):
        raise ValueError("training corpus row_count mismatch")
    if len(rows) != _TRAINING_COUNT:
        raise ValueError("prediction child requires exactly four training rows")
    parsed = tuple(_parse_text_row(row, labelled=True) for row in rows)
    _require_canonical_row_order(parsed, "training corpus")
    counts = Counter(_required_label(row) for row in parsed)
    if counts != Counter({label: 2 for label in request.labels}):
        raise ValueError("training corpus labels must be balanced two per class")
    if any(_required_label(row) not in request.labels for row in parsed):
        raise ValueError("training corpus contains an unknown label")
    return parsed


def _parse_predictor_request(
    payload: Mapping[str, object],
    *,
    request: _ChildRequest,
) -> tuple[_OpaqueTextRow, ...]:
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
        "predictor request",
    )
    if payload["protocol_id"] != request.protocol_id:
        raise ValueError("predictor request protocol_id mismatch")
    if payload["public_corpus_artifact_id"] != request.public_corpus_artifact_id:
        raise ValueError("predictor request public_corpus_artifact_id mismatch")
    rows = _list(payload["challenge_inputs"], "challenge_inputs")
    if _positive_integer(
        payload["challenge_input_count"],
        "challenge_input_count",
    ) != len(rows):
        raise ValueError("predictor request challenge_input_count mismatch")
    if len(rows) != _CHALLENGE_COUNT:
        raise ValueError("prediction child requires exactly 224 challenge rows")
    parsed = tuple(_parse_text_row(row, labelled=False) for row in rows)
    _require_canonical_row_order(parsed, "predictor request")
    return parsed


def _parse_text_row(payload: object, *, labelled: bool) -> _OpaqueTextRow:
    mapped = _mapping(payload, "prediction text row")
    expected = {"item_id", "text", "text_sha256"}
    if labelled:
        expected.add("condition_label")
    _require_exact_keys(mapped, expected, "prediction text row")
    text = _text(mapped["text"], "text")
    text_sha256 = _digest(mapped["text_sha256"], "text_sha256")
    if text_sha256 != _sha256_text(text):
        raise ValueError("prediction text row text_sha256 mismatch")
    return _OpaqueTextRow(
        item_id=_digest(mapped["item_id"], "item_id"),
        text=text,
        text_sha256=text_sha256,
        condition_label=(_text(mapped["condition_label"], "condition_label") if labelled else None),
    )


def _validate_joint_inputs(
    training_rows: tuple[_OpaqueTextRow, ...],
    challenge_rows: tuple[_OpaqueTextRow, ...],
) -> None:
    all_rows = (*training_rows, *challenge_rows)
    item_ids = tuple(row.item_id for row in all_rows)
    text_sha256s = tuple(row.text_sha256 for row in all_rows)
    if len(set(item_ids)) != _LIVE_EMBEDDING_COUNT:
        raise ValueError("prediction child item ids must be unique across all 228 rows")
    if len(set(text_sha256s)) != _LIVE_EMBEDDING_COUNT:
        raise ValueError("prediction child texts must be unique across all 228 rows")


def _build_live_embedding_table(
    *,
    embedder: _FrozenQualificationEmbedder,
    model: _SemanticModelSpec,
    rows: tuple[_OpaqueTextRow, ...],
) -> tuple[PrecomputedPublicEmbeddingTable, int]:
    records: list[PrecomputedPublicEmbeddingRecord] = []
    live_count = 0
    for row in rows:
        vector = embedder.embed(row.text)
        live_count += 1
        if not isinstance(vector, tuple) or len(vector) != model.embedding_width:
            raise ValueError("live relationship embedding width mismatch")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in vector
        ):
            raise ValueError("live relationship embedding must be finite numeric values")
        records.append(
            PrecomputedPublicEmbeddingRecord(
                text=row.text,
                embedding_hex=tuple(_canonical_float_hex(value) for value in vector),
            )
        )
    table = PrecomputedPublicEmbeddingTable(
        source_embedder_name=embedder.name,
        embedding_width=model.embedding_width,
        records=tuple(sorted(records, key=lambda item: (item.text_sha256, item.text))),
    )
    return table, live_count


def _build_prediction_ledger(
    *,
    request: _ChildRequest,
    predictor_request_artifact_id: str,
    embedding_table: PrecomputedPublicEmbeddingTable,
    reader_artifact: FrozenLinearRelationshipConditionReaderArtifact,
    reader_runtime: FrozenLinearRelationshipConditionReaderRuntime,
    challenge_rows: tuple[_OpaqueTextRow, ...],
) -> Mapping[str, object]:
    rows: list[dict[str, object]] = []
    for item in challenge_rows:
        readout = reader_runtime.read_condition(item.text)
        numeric = relationship_condition_readout_to_payload(readout)
        if numeric["source_observation_sha256"] != item.text_sha256:
            raise RuntimeError("reader readout source hash drifted from predictor input")
        candidate_scores = _list(numeric["candidate_scores"], "candidate_scores")
        rows.append(
            {
                "item_id": item.item_id,
                "text_sha256": item.text_sha256,
                "condition_label": _text(
                    numeric["condition_label"],
                    "condition_label",
                ),
                "confidence_hex": _canonical_float_hex(numeric["confidence"]),
                "normalized_margin_hex": _canonical_float_hex(numeric["normalized_margin"]),
                "candidate_scores": [
                    {
                        "label": _text(
                            _mapping(score, "candidate score")["label"],
                            "candidate score label",
                        ),
                        "score_hex": _canonical_float_hex(_mapping(score, "candidate score")["score"]),
                    }
                    for score in candidate_scores
                ],
                "reader_artifact_id": _text(
                    numeric["reader_artifact_id"],
                    "reader_artifact_id",
                ),
                "source_observation_sha256": _text(
                    numeric["source_observation_sha256"],
                    "source_observation_sha256",
                ),
            }
        )
    ledger = _with_artifact_id(
        {
            "schema_version": RELATIONSHIP_READER_PREDICTION_LEDGER_SCHEMA_VERSION,
            "protocol_id": request.protocol_id,
            "execution_protocol_id": request.execution_protocol_id,
            "child_request_artifact_id": request.artifact_id,
            "predictor_request_artifact_id": predictor_request_artifact_id,
            "embedding_table_artifact_id": embedding_table.artifact_id,
            "reader_artifact_id": reader_artifact.artifact_id,
            "rows": rows,
            "row_count": len(rows),
            "challenge_labels_present": False,
            "qualification_scored": False,
        }
    )
    _validate_prediction_ledger(
        ledger,
        request=request,
        challenge_rows=challenge_rows,
        embedding_table_artifact_id=embedding_table.artifact_id,
        reader_artifact_id=reader_artifact.artifact_id,
    )
    return ledger


def _validate_prediction_ledger(
    payload: Mapping[str, object],
    *,
    request: _ChildRequest,
    challenge_rows: tuple[_OpaqueTextRow, ...],
    embedding_table_artifact_id: str,
    reader_artifact_id: str,
) -> None:
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "protocol_id",
            "execution_protocol_id",
            "child_request_artifact_id",
            "predictor_request_artifact_id",
            "embedding_table_artifact_id",
            "reader_artifact_id",
            "rows",
            "row_count",
            "challenge_labels_present",
            "qualification_scored",
            "artifact_id",
        },
        "prediction ledger",
    )
    if payload["schema_version"] != RELATIONSHIP_READER_PREDICTION_LEDGER_SCHEMA_VERSION:
        raise ValueError("prediction ledger schema mismatch")
    _digest(payload["artifact_id"], "prediction ledger artifact_id")
    _digest(payload["child_request_artifact_id"], "child_request_artifact_id")
    _digest(payload["predictor_request_artifact_id"], "predictor_request_artifact_id")
    _digest(payload["embedding_table_artifact_id"], "embedding_table_artifact_id")
    _digest(payload["reader_artifact_id"], "reader_artifact_id")
    if payload["protocol_id"] != request.protocol_id:
        raise ValueError("prediction ledger protocol_id mismatch")
    if payload["execution_protocol_id"] != request.execution_protocol_id:
        raise ValueError("prediction ledger execution_protocol_id mismatch")
    _digest(payload["execution_protocol_id"], "execution_protocol_id")
    if payload["child_request_artifact_id"] != request.artifact_id:
        raise ValueError("prediction ledger child request mismatch")
    if payload["predictor_request_artifact_id"] != request.predictor_request_artifact_id:
        raise ValueError("prediction ledger predictor request mismatch")
    if payload["embedding_table_artifact_id"] != embedding_table_artifact_id:
        raise ValueError("prediction ledger embedding table mismatch")
    if payload["reader_artifact_id"] != reader_artifact_id:
        raise ValueError("prediction ledger reader artifact mismatch")
    if _boolean(payload["challenge_labels_present"], "challenge_labels_present"):
        raise ValueError("prediction ledger must not contain challenge labels")
    if _boolean(payload["qualification_scored"], "qualification_scored"):
        raise ValueError("prediction ledger must remain unscored")
    raw_rows = _list(payload["rows"], "prediction ledger rows")
    if _positive_integer(payload["row_count"], "row_count") != _CHALLENGE_COUNT:
        raise ValueError("prediction ledger must contain 224 rows")
    if len(raw_rows) != _CHALLENGE_COUNT:
        raise ValueError("prediction ledger row array must contain 224 rows")
    expected_by_id = {row.item_id: row for row in challenge_rows}
    observed_ids: list[str] = []
    for raw_row in raw_rows:
        row = _mapping(raw_row, "prediction ledger row")
        _require_exact_keys(
            row,
            {
                "item_id",
                "text_sha256",
                "condition_label",
                "confidence_hex",
                "normalized_margin_hex",
                "candidate_scores",
                "reader_artifact_id",
                "source_observation_sha256",
            },
            "prediction ledger row",
        )
        item_id = _digest(row["item_id"], "item_id")
        if item_id not in expected_by_id:
            raise ValueError("prediction ledger contains an unknown item id")
        expected = expected_by_id[item_id]
        if row["text_sha256"] != expected.text_sha256:
            raise ValueError("prediction ledger text hash mismatch")
        candidate_scores = tuple(
            (
                _text(
                    _candidate_score(score)["label"],
                    "score label",
                ),
                _decode_canonical_float_hex(
                    _candidate_score(score)["score_hex"],
                    "score_hex",
                ),
            )
            for score in _list(row["candidate_scores"], "candidate_scores")
        )
        if tuple(label for label, _ in candidate_scores) != request.labels:
            raise ValueError("prediction ledger candidate labels mismatch")
        numeric_payload = {
            "condition_label": row["condition_label"],
            "confidence": _decode_canonical_float_hex(
                row["confidence_hex"],
                "confidence_hex",
            ),
            "normalized_margin": _decode_canonical_float_hex(
                row["normalized_margin_hex"],
                "normalized_margin_hex",
            ),
            "candidate_scores": [{"label": label, "score": score} for label, score in candidate_scores],
            "reader_artifact_id": row["reader_artifact_id"],
            "source_observation_sha256": row["source_observation_sha256"],
        }
        readout = relationship_condition_readout_from_payload(numeric_payload)
        if readout.reader_artifact_id != reader_artifact_id:
            raise ValueError("prediction ledger row reader artifact mismatch")
        if readout.source_observation_sha256 != expected.text_sha256:
            raise ValueError("prediction ledger row source observation mismatch")
        observed_ids.append(item_id)
    if observed_ids != sorted(expected_by_id):
        raise ValueError("prediction ledger rows must use canonical item_id order")
    if _artifact_id(payload) != payload["artifact_id"]:
        raise ValueError("prediction ledger artifact_id mismatch")


def _candidate_score(value: object) -> Mapping[str, object]:
    payload = _mapping(value, "candidate score")
    _require_exact_keys(payload, {"label", "score_hex"}, "candidate score")
    return payload


def _write_deterministic_outputs(
    *,
    root: pathlib.Path,
    table: PrecomputedPublicEmbeddingTable,
    reader_artifact: FrozenLinearRelationshipConditionReaderArtifact,
    prediction_ledger: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    table_raw = table.to_json().encode("utf-8")
    reader_raw = reader_artifact.to_json_bytes()
    ledger_raw = _canonical_artifact_bytes(prediction_ledger)
    values = (
        ("embedding_table.json", table_raw, table.artifact_id),
        ("reader_artifact.json", reader_raw, reader_artifact.artifact_id),
        ("prediction_ledger.json", ledger_raw, _artifact_id(prediction_ledger)),
    )
    receipts: list[Mapping[str, object]] = []
    for relative_path, raw, artifact_id in values:
        target = root / relative_path
        _write_bytes_create_only(target, raw)
        if relative_path == "embedding_table.json":
            loaded_table = PrecomputedPublicEmbeddingTable.from_json(target.read_bytes().decode("utf-8"))
            if loaded_table != table:
                raise RuntimeError("embedding table failed canonical readback")
        elif relative_path == "reader_artifact.json":
            loaded_reader = FrozenLinearRelationshipConditionReaderArtifact.from_json(target.read_bytes())
            if loaded_reader != reader_artifact:
                raise RuntimeError("reader artifact failed canonical readback")
        else:
            loaded_ledger = _load_canonical_artifact(
                target,
                RELATIONSHIP_READER_PREDICTION_LEDGER_SCHEMA_VERSION,
            )
            if loaded_ledger.payload != prediction_ledger:
                raise RuntimeError("prediction ledger failed canonical readback")
        receipts.append(_file_receipt(path=relative_path, raw=raw, artifact_id=artifact_id))
    return tuple(receipts)


def _validate_embedder_identity(
    embedder: _FrozenQualificationEmbedder,
    spec: _SemanticModelSpec,
) -> None:
    identity = (
        embedder.model_source,
        embedder.model_revision,
        embedder.weights_sha256,
        embedder.sentence_transformers_version,
    )
    expected = (
        spec.model_id,
        spec.model_revision,
        spec.weights_sha256,
        spec.sentence_transformers_version,
    )
    if identity != expected:
        raise ValueError("prediction child embedder identity mismatch")
    _text(embedder.name, "embedder.name")


def _validate_formal_bge_cuda(spec: _SemanticModelSpec) -> None:
    expected = (
        BGE_M3_MODEL_ID,
        BGE_M3_MODEL_REVISION,
        BGE_M3_WEIGHT_BYTES_SHA256,
        BGE_M3_SENTENCE_TRANSFORMERS_VERSION,
        _FORMAL_EMBEDDING_WIDTH,
        "cuda",
        False,
        False,
    )
    actual = (
        spec.model_id,
        spec.model_revision,
        spec.weights_sha256,
        spec.sentence_transformers_version,
        spec.embedding_width,
        spec.device,
        spec.network_allowed,
        spec.stub_allowed,
    )
    if actual != expected:
        raise ValueError("formal prediction child requires exact pinned BGE-M3 CUDA identity")


def _validate_source_file_pin(
    loaded: _LoadedArtifact,
    *,
    expected_artifact_id: str,
    expected_raw_sha256: str,
    expected_raw_bytes: int,
    field_name: str,
) -> None:
    if loaded.payload["artifact_id"] != expected_artifact_id:
        raise ValueError(f"{field_name} artifact_id pin mismatch")
    if loaded.raw_sha256 != expected_raw_sha256:
        raise ValueError(f"{field_name} raw sha256 pin mismatch")
    if len(loaded.raw) != expected_raw_bytes:
        raise ValueError(f"{field_name} raw byte count mismatch")


def _load_canonical_artifact(
    path: pathlib.Path,
    expected_schema_version: str,
) -> _LoadedArtifact:
    source = pathlib.Path(path)
    if source.is_symlink():
        raise ValueError(f"canonical artifact path must not be a symlink: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"canonical artifact does not exist: {source}")
    if source.stat().st_nlink != 1:
        raise ValueError(f"canonical artifact must not be hard-linked: {source}")
    raw = source.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"canonical artifact must be exact UTF-8: {source}") from exc
    payload = _parse_unique_json(text, str(source))
    if payload.get("schema_version") != expected_schema_version:
        raise ValueError(f"canonical artifact schema mismatch: {source}")
    artifact_id = _digest(payload.get("artifact_id"), "artifact_id")
    if artifact_id != _artifact_id(payload):
        raise ValueError(f"canonical artifact id mismatch: {source}")
    if raw != _canonical_artifact_bytes(payload):
        raise ValueError(f"artifact is not canonical LF-terminated JSON: {source}")
    return _LoadedArtifact(
        payload=payload,
        raw=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _write_bytes_create_only(path: pathlib.Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != raw:
        raise RuntimeError(f"create-only artifact readback mismatch: {path}")


def _file_receipt(
    *,
    path: str,
    raw: bytes,
    artifact_id: str,
) -> Mapping[str, object]:
    return {
        "path": path,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes": len(raw),
        "artifact_id": artifact_id,
    }


def _semantic_model_payload(spec: _SemanticModelSpec) -> Mapping[str, object]:
    return {
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "weights_sha256": spec.weights_sha256,
        "sentence_transformers_version": spec.sentence_transformers_version,
        "embedding_width": spec.embedding_width,
        "device": spec.device,
        "network_allowed": spec.network_allowed,
        "stub_allowed": spec.stub_allowed,
    }


def _forbidden_module_presence() -> dict[str, bool]:
    return {name: name in sys.modules for name in _FORBIDDEN_MODULE_NAMES}


def _bootstrap_import_roots() -> list[str]:
    value = os.environ.get("PYTHONPATH")
    if value is None:
        return []
    roots = value.split(os.pathsep)
    if any(not root for root in roots):
        raise RuntimeError("prediction child PYTHONPATH contains an empty import root")
    return roots


def _volvence_zero_namespace_search_locations() -> list[str]:
    module = sys.modules.get("volvence_zero")
    if module is None:
        raise RuntimeError("volvence_zero namespace is absent from the prediction child")
    module_spec = vars(module).get("__spec__")
    locations = None if module_spec is None else module_spec.submodule_search_locations
    if locations is None:
        raise RuntimeError("volvence_zero is not a namespace/package in the prediction child")
    return [str(pathlib.Path(value).resolve()) for value in locations]


def _required_label(row: _OpaqueTextRow) -> str:
    if row.condition_label is None:
        raise ValueError("training row is missing condition_label")
    return row.condition_label


def _require_canonical_row_order(
    rows: tuple[_OpaqueTextRow, ...],
    field_name: str,
) -> None:
    item_ids = tuple(row.item_id for row in rows)
    if len(set(item_ids)) != len(item_ids):
        raise ValueError(f"{field_name} item ids must be unique")
    if item_ids != tuple(sorted(item_ids)):
        raise ValueError(f"{field_name} rows must use canonical item_id order")


def _canonical_float_hex(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("canonical float value must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("canonical float value must be finite")
    if numeric == 0.0:
        numeric = 0.0
    return numeric.hex()


def _decode_canonical_float_hex(value: object, field_name: str) -> float:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a canonical float hex string")
    try:
        numeric = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a canonical float hex string") from exc
    if not math.isfinite(numeric) or _canonical_float_hex(numeric) != value:
        raise ValueError(f"{field_name} must be finite canonical float hex")
    return numeric


def _with_artifact_id(core: Mapping[str, object]) -> dict[str, object]:
    payload = dict(core)
    if "artifact_id" in payload:
        raise ValueError("artifact core must not predefine artifact_id")
    return {**payload, "artifact_id": _artifact_id(payload)}


def _artifact_id(payload: Mapping[str, object]) -> str:
    core = {key: value for key, value in payload.items() if key != "artifact_id"}
    return hashlib.sha256(_canonical_json(core).encode("utf-8")).hexdigest()


def _canonical_artifact_bytes(payload: Mapping[str, object]) -> bytes:
    return (_canonical_json(payload) + "\n").encode("utf-8")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


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


def _labels(value: object, field_name: str) -> tuple[str, ...]:
    values = _list(value, field_name)
    labels = tuple(_text(item, field_name) for item in values)
    if len(labels) != 2 or len(set(labels)) != 2:
        raise ValueError(f"{field_name} must contain exactly two unique labels")
    if labels != tuple(sorted(labels, key=lambda item: item.encode("utf-8"))):
        raise ValueError(f"{field_name} must use strict UTF-8 byte order")
    return labels


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be a JSON object with string keys")
    return value


def _list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be canonical non-empty text")
    return value


def _digest(value: object, field_name: str) -> str:
    _require_sha256(value, field_name)
    assert isinstance(value, str)
    return value


def _require_sha256(value: object, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    field_name: str,
) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{field_name} keys mismatch; missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "RELATIONSHIP_READER_PREDICTION_CHILD_REQUEST_SCHEMA_VERSION",
    "RELATIONSHIP_READER_PREDICTION_LEDGER_SCHEMA_VERSION",
    "RELATIONSHIP_READER_PREDICTION_MANIFEST_SCHEMA_VERSION",
    "RELATIONSHIP_READER_PREDICTION_PROCESS_ATTESTATION_SCHEMA_VERSION",
    "RELATIONSHIP_READER_PREDICTION_TRAINING_CORPUS_SCHEMA_VERSION",
    "RELATIONSHIP_READER_PREDICTOR_REQUEST_SCHEMA_VERSION",
    "run_relationship_condition_reader_prediction_child",
]
