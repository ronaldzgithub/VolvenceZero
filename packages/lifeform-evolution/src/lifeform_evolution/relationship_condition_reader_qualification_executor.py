"""Source-separated parent executor for relationship-reader qualification.

The parent is deliberately narrower than the preflight validator and the
model-free scorer.  Before the prediction ledger is durably committed it may
open only the preflight manifest, publication request, public corpus, opaque
predictor request, and four condition-only training labels.  In particular,
it never opens the challenge-label or group-split files.  Their immutable
identities are copied from the already content-addressed preflight manifest
into a scoring request only after two fresh prediction children agree byte for
byte and the ledger commit receipt has been closed and reopened successfully.

This is a reviewed process-order firewall, not an operating-system security
boundary.  The Windows launcher uses a fresh process and a kill-on-close Job
Object, while every published artifact continues to state that filesystem and
directory-entry durability are not provided by that boundary.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
import pathlib
import stat
import subprocess
import sys
import threading
from typing import Callable, Mapping, Protocol

from volvence_zero.canonical_json import canonical_json_bytes, strict_json_loads

from lifeform_evolution.relationship_condition_reader_qualification_runtime_binding import (
    QualificationChildImportBinding,
    build_qualification_child_import_binding,
    controlled_child_path,
    expected_child_sys_path,
    validate_child_file_backed_module_origin_attestation,
)


RELATIONSHIP_READER_PREDICTION_LEDGER_COMMIT_SCHEMA_VERSION = (
    "relationship-condition-reader-prediction-ledger-commit.v1"
)
RELATIONSHIP_READER_SCORING_REQUEST_SCHEMA_VERSION = "relationship-condition-reader-qualification-scoring-request.v1"

_PREFLIGHT_MANIFEST_SCHEMA_VERSION = "relationship-condition-reader-qualification-preflight-manifest.v1"
_PUBLICATION_REQUEST_SCHEMA_VERSION = "relationship-condition-reader-qualification-publication-request.v1"
_PUBLIC_CORPUS_SCHEMA_VERSION = "relationship-condition-reader-qualification-public-corpus.v1"
_PREDICTOR_REQUEST_SCHEMA_VERSION = "relationship-condition-reader-qualification-predictor-request.v1"
_TRAINING_LABELS_SCHEMA_VERSION = "relationship-condition-reader-qualification-training-labels.v1"
_TRAINING_CORPUS_SCHEMA_VERSION = "relationship-condition-reader-qualification-training-corpus.v1"
_CHILD_REQUEST_SCHEMA_VERSION = "relationship-condition-reader-prediction-child-request.v1"
_PREDICTION_LEDGER_SCHEMA_VERSION = "relationship-condition-reader-prediction-ledger.v1"
_PREDICTION_ATTESTATION_SCHEMA_VERSION = "relationship-condition-reader-prediction-process-attestation.v3"
_PREDICTION_MANIFEST_SCHEMA_VERSION = "relationship-condition-reader-prediction-manifest.v1"
_EMBEDDING_TABLE_SCHEMA_VERSION = "relationship-product-public-embedding-table.v2"
_READER_ARTIFACT_SCHEMA_VERSION = "relationship-condition-reader-artifact.v2"

_BGE_M3_MODEL_ID = "BAAI/bge-m3"
_BGE_M3_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_BGE_M3_WEIGHT_BYTES_SHA256 = "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"
_BGE_M3_SENTENCE_TRANSFORMERS_VERSION = "5.6.0"
_BGE_M3_EMBEDDING_WIDTH = 1024
_READER_SOLVER = "unit_normalized_class_centroid_linear"
_READER_SOLVER_VERSION = "relationship-condition-centroid-solver.v1"
_LABELS = ("agency_displacement", "belonging_erasure")

_TRAINING_COUNT = 4
_CHALLENGE_COUNT = 224
_LIVE_EMBEDDING_COUNT = _TRAINING_COUNT + _CHALLENGE_COUNT
_PREFLIGHT_RELATIVE_PATHS = (
    "protocol.json",
    "public/public_corpus.json",
    "public/predictor_request.json",
    "public/publication_request.json",
    "sealed/condition_training_labels.json",
    "sealed/challenge_labels.json",
    "sealed/group_split.json",
)
_PARENT_OPENABLE_PREFLIGHT_PATHS = frozenset(
    {
        "manifest.json",
        "public/publication_request.json",
        "public/public_corpus.json",
        "public/predictor_request.json",
        "sealed/condition_training_labels.json",
    }
)
_DETERMINISTIC_OUTPUT_PATHS = (
    "embedding_table.json",
    "reader_artifact.json",
    "prediction_ledger.json",
)
_CHILD_OUTPUT_PATHS = (
    *_DETERMINISTIC_OUTPUT_PATHS,
    "process_attestation.json",
    "manifest.json",
)
_FORBIDDEN_WORKER_MODULES = frozenset(
    {
        "lifeform_domain_emogpt.lab.relationship_product_pilot_source",
        "lifeform_evolution.relationship_condition_reader_qualification",
        "lifeform_evolution.relationship_lab_product_horizon",
    }
)
_MAX_SMALL_ARTIFACT_BYTES = 2_000_000
_MAX_PUBLIC_CORPUS_BYTES = 2_000_000
_MAX_CHILD_OUTPUT_BYTES = 64_000_000
_MAX_LEDGER_BYTES = 4_000_000
_MAX_CAPTURED_STREAM_PREFIX_BYTES = 65_536
_FORMAL_ENVIRONMENT_ALLOWLIST = (
    "APPDATA",
    "LOCALAPPDATA",
    "SystemDrive",
    "SystemRoot",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
)
_FORMAL_ENVIRONMENT_FIXED = {
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_HUB_OFFLINE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "PYTHONUTF8": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}
_PREDICTION_ENVIRONMENT_PROJECTION_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HUB_OFFLINE",
    "PYTHONHASHSEED",
    "PYTHONPATH",
    "PYTHONPYCACHEPREFIX",
    "PYTHONSAFEPATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUTF8",
    "TRANSFORMERS_OFFLINE",
)
_PREDICTION_REQUIRED_REPOSITORY_MODULES = frozenset(
    {
        "lifeform_domain_emogpt.relationship_condition_reader",
        "lifeform_evolution.relationship_condition_reader_qualification_predictor",
        "lifeform_evolution.relationship_condition_reader_qualification_runtime_binding",
        "lifeform_evolution.relationship_lab_product_model_adapters",
        "volvence_zero.social_cognition",
    }
)
_CREATE_SUSPENDED = 0x00000004
_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_CREATE_NO_WINDOW = 0x08000000
_FORMAL_CREATION_FLAGS = _CREATE_SUSPENDED | _EXTENDED_STARTUPINFO_PRESENT | _CREATE_NO_WINDOW
_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_FORMAL_JOB_LIMIT_FLAGS = _JOB_OBJECT_LIMIT_ACTIVE_PROCESS | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
_PREDICTION_CHILD_BOOTSTRAP = """import json
import pathlib
import sys

expected_import_roots = json.loads(sys.argv[1])
if not isinstance(expected_import_roots, list) or not all(
    isinstance(value, str) and value for value in expected_import_roots
):
    raise RuntimeError("prediction bootstrap import roots are invalid")
if sys.path[: len(expected_import_roots)] != expected_import_roots:
    raise RuntimeError("prediction bootstrap import roots differ from interpreter sys.path")

from lifeform_evolution.relationship_condition_reader_qualification_predictor import (
    run_relationship_condition_reader_prediction_child,
)

run_relationship_condition_reader_prediction_child(
    child_request_path=pathlib.Path(sys.argv[2]),
    expected_child_request_artifact_id=sys.argv[3],
    training_corpus_path=pathlib.Path(sys.argv[4]),
    predictor_request_path=pathlib.Path(sys.argv[5]),
    output_root=pathlib.Path(sys.argv[6]),
    run_ordinal=int(sys.argv[7]),
    run_nonce=sys.argv[8],
    bge_snapshot_path=pathlib.Path(sys.argv[9]),
)
"""


@dataclass(frozen=True)
class _LoadedArtifact:
    payload: Mapping[str, object]
    raw: bytes
    raw_sha256: str


@dataclass(frozen=True)
class _PredictionChildLaunchSpec:
    child_request_path: pathlib.Path
    expected_child_request_artifact_id: str
    training_corpus_path: pathlib.Path
    predictor_request_path: pathlib.Path
    output_root: pathlib.Path
    capsule_root: pathlib.Path
    run_ordinal: int
    run_nonce: str
    bge_snapshot_path: pathlib.Path | None
    import_binding: QualificationChildImportBinding
    pycache_prefix: pathlib.Path


@dataclass(frozen=True)
class _EnvironmentValueReceipt:
    key: str
    value_sha256: str
    value_utf8_bytes: int


@dataclass(frozen=True)
class _BoundedStreamCapture:
    raw_sha256: str
    total_bytes: int
    retained_prefix: bytes
    retained_prefix_sha256: str
    retained_prefix_bytes: int
    prefix_truncated: bool


@dataclass(frozen=True)
class _PredictionChildLaunchResult:
    process_id: int
    process_argv: tuple[str, ...]
    exit_code: int | None
    process_exited: bool
    job_object_empty: bool
    environment_contract_id: str
    environment_projection: tuple[_EnvironmentValueReceipt, ...]
    creation_flags: int
    shell: bool
    close_fds: bool
    process_created_suspended: bool
    job_assigned_before_resume: bool
    initial_thread_resume_previous_count: int
    job_limit_flags: int
    job_active_process_limit: int
    stdout_capture: _BoundedStreamCapture
    stderr_capture: _BoundedStreamCapture
    source_capsule_used: bool
    repository_import_path_used: bool
    bge_snapshot_tree_verified: bool


@dataclass(frozen=True)
class _VerifiedPredictionChild:
    process_id: int
    manifest_artifact_id: str
    attestation_artifact_id: str
    embedding_table_artifact_id: str
    reader_artifact_id: str
    prediction_ledger_artifact_id: str
    deterministic_raw: tuple[bytes, bytes, bytes]
    ledger_payload: Mapping[str, object]


class _PredictionChildLauncher(Protocol):
    def __call__(
        self,
        spec: _PredictionChildLaunchSpec,
    ) -> _PredictionChildLaunchResult: ...


_ArtifactLoader = Callable[..., _LoadedArtifact]


class _IntegrityGuard(Protocol):
    def __call__(
        self,
        *,
        phase: str,
        previous_integrity_receipt_artifact_id: str | None,
    ) -> Mapping[str, object]: ...


def execute_relationship_condition_reader_qualification_prediction_stage(
    *,
    preflight_root: pathlib.Path,
    execution_root: pathlib.Path,
    expected_qualification_protocol_id: str,
    expected_preflight_manifest_artifact_id: str,
    expected_publication_request_artifact_id: str,
    execution_protocol_id: str,
    run_nonce: str,
    integrity_guard: _IntegrityGuard,
    previous_integrity_receipt_artifact_id: str,
    expected_source_tree_artifact_id: str,
    expected_bge_snapshot_tree_artifact_id: str,
    expected_runtime_identity_artifact_id: str,
    repository_root: pathlib.Path,
    repository_source_roots: tuple[pathlib.Path, ...],
    frozen_source_entries: Mapping[str, Mapping[str, object]],
    frozen_site_packages_root: pathlib.Path,
    bge_snapshot_path: pathlib.Path | None = None,
    python_executable: pathlib.Path | None = None,
    child_timeout_seconds: int = 7_200,
) -> Mapping[str, object]:
    """Run the fresh-process mechanism and freeze the scorer handoff.

    No scorer is launched here.  A later CLI must start the model-free scorer
    in a separate process with the returned scoring-request path.  This entry
    point does not validate an external execution-protocol publication anchor;
    a bare ``execution_protocol_id`` is lineage, not authorization.  Therefore
    the final CLI must validate that anchor before calling this function, and
    direct invocation must never be reported as an authorized qualification.
    """

    import_binding = build_qualification_child_import_binding(
        python_executable=pathlib.Path(python_executable or sys.executable),
        repository_root=repository_root,
        repository_source_roots=repository_source_roots,
        frozen_source_entries=frozen_source_entries,
        frozen_site_packages_root=frozen_site_packages_root,
    )
    if (
        isinstance(child_timeout_seconds, bool)
        or not isinstance(child_timeout_seconds, int)
        or child_timeout_seconds < 1
    ):
        raise ValueError("child_timeout_seconds must be a positive integer")
    if bge_snapshot_path is None:
        raise ValueError("formal prediction requires an explicit pinned BGE snapshot path")
    if not callable(integrity_guard):
        raise TypeError("formal prediction requires an integrity_guard callback")
    resolved_snapshot = pathlib.Path(bge_snapshot_path).resolve()
    if not resolved_snapshot.is_dir() or resolved_snapshot.is_symlink():
        raise ValueError("formal BGE snapshot path must be a non-symlink directory")

    def launcher(spec: _PredictionChildLaunchSpec) -> _PredictionChildLaunchResult:
        return _launch_windows_fresh_prediction_subprocess(
            spec,
            timeout_seconds=child_timeout_seconds,
        )

    return _execute_relationship_condition_reader_qualification_prediction_stage_core(
        preflight_root=preflight_root,
        execution_root=execution_root,
        expected_qualification_protocol_id=expected_qualification_protocol_id,
        expected_preflight_manifest_artifact_id=(expected_preflight_manifest_artifact_id),
        expected_publication_request_artifact_id=(expected_publication_request_artifact_id),
        execution_protocol_id=execution_protocol_id,
        run_nonce=run_nonce,
        bge_snapshot_path=resolved_snapshot,
        integrity_guard=integrity_guard,
        previous_integrity_receipt_artifact_id=(previous_integrity_receipt_artifact_id),
        expected_source_tree_artifact_id=expected_source_tree_artifact_id,
        expected_bge_snapshot_tree_artifact_id=(expected_bge_snapshot_tree_artifact_id),
        expected_runtime_identity_artifact_id=(expected_runtime_identity_artifact_id),
        import_binding=import_binding,
        launcher=launcher,
        artifact_loader=_load_canonical_artifact,
    )


def _execute_relationship_condition_reader_qualification_prediction_stage_core(
    *,
    preflight_root: pathlib.Path,
    execution_root: pathlib.Path,
    expected_qualification_protocol_id: str,
    expected_preflight_manifest_artifact_id: str,
    expected_publication_request_artifact_id: str,
    execution_protocol_id: str,
    run_nonce: str,
    bge_snapshot_path: pathlib.Path | None,
    integrity_guard: _IntegrityGuard,
    previous_integrity_receipt_artifact_id: str,
    expected_source_tree_artifact_id: str,
    expected_bge_snapshot_tree_artifact_id: str,
    expected_runtime_identity_artifact_id: str,
    import_binding: QualificationChildImportBinding,
    launcher: _PredictionChildLauncher,
    artifact_loader: _ArtifactLoader,
) -> Mapping[str, object]:
    """Private injectable core used by model-free orchestration tests."""

    qualification_protocol_id = _digest(
        expected_qualification_protocol_id,
        "expected_qualification_protocol_id",
    )
    expected_manifest_id = _digest(
        expected_preflight_manifest_artifact_id,
        "expected_preflight_manifest_artifact_id",
    )
    expected_publication_id = _digest(
        expected_publication_request_artifact_id,
        "expected_publication_request_artifact_id",
    )
    execution_id = _digest(execution_protocol_id, "execution_protocol_id")
    parent_run_nonce = _digest(run_nonce, "run_nonce")
    expected_integrity_ids = {
        "source_tree_artifact_id": _digest(
            expected_source_tree_artifact_id,
            "expected_source_tree_artifact_id",
        ),
        "bge_snapshot_tree_artifact_id": _digest(
            expected_bge_snapshot_tree_artifact_id,
            "expected_bge_snapshot_tree_artifact_id",
        ),
        "runtime_identity_artifact_id": _digest(
            expected_runtime_identity_artifact_id,
            "expected_runtime_identity_artifact_id",
        ),
    }
    initial_integrity_receipt_id = _digest(
        previous_integrity_receipt_artifact_id,
        "previous_integrity_receipt_artifact_id",
    )
    if execution_id == qualification_protocol_id:
        raise ValueError("execution protocol must be distinct from qualification protocol")
    if not isinstance(import_binding, QualificationChildImportBinding):
        raise TypeError("formal prediction requires a frozen import binding")
    if not callable(integrity_guard) or not callable(launcher) or not callable(artifact_loader):
        raise TypeError("integrity_guard, launcher, and artifact_loader must be callable")

    preflight = pathlib.Path(preflight_root).resolve()
    root = pathlib.Path(execution_root).resolve()
    if not preflight.is_dir():
        raise FileNotFoundError(f"qualification preflight root is absent: {preflight}")
    if root.exists():
        raise FileExistsError(f"qualification execution root already exists: {root}")
    if root == preflight or root in preflight.parents or preflight in root.parents:
        raise ValueError("preflight and execution roots must be physically disjoint")

    # Deliberately do not call the full preflight validator.  That validator
    # opens challenge labels and group split, which is forbidden in this phase.
    manifest_loaded = artifact_loader(
        preflight / "manifest.json",
        expected_schema_version=_PREFLIGHT_MANIFEST_SCHEMA_VERSION,
        max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
    )
    manifest = manifest_loaded.payload
    if manifest["artifact_id"] != expected_manifest_id:
        raise ValueError("external preflight manifest artifact id mismatch")
    manifest_files = _validate_preflight_manifest(
        manifest,
        qualification_protocol_id=qualification_protocol_id,
    )

    publication_loaded = artifact_loader(
        preflight / "public/publication_request.json",
        expected_schema_version=_PUBLICATION_REQUEST_SCHEMA_VERSION,
        max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
    )
    _validate_manifest_file_receipt(
        manifest_files["public/publication_request.json"],
        publication_loaded,
        "publication request",
    )
    publication = publication_loaded.payload
    _validate_publication_request(
        publication,
        qualification_protocol_id=qualification_protocol_id,
        expected_artifact_id=expected_publication_id,
        expected_execution_root=root,
    )

    public_corpus_loaded = artifact_loader(
        preflight / "public/public_corpus.json",
        expected_schema_version=_PUBLIC_CORPUS_SCHEMA_VERSION,
        max_bytes=_MAX_PUBLIC_CORPUS_BYTES,
    )
    predictor_loaded = artifact_loader(
        preflight / "public/predictor_request.json",
        expected_schema_version=_PREDICTOR_REQUEST_SCHEMA_VERSION,
        max_bytes=_MAX_PUBLIC_CORPUS_BYTES,
    )
    training_labels_loaded = artifact_loader(
        preflight / "sealed/condition_training_labels.json",
        expected_schema_version=_TRAINING_LABELS_SCHEMA_VERSION,
        max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
    )
    for relative_path, loaded, field_name in (
        ("public/public_corpus.json", public_corpus_loaded, "public corpus"),
        ("public/predictor_request.json", predictor_loaded, "predictor request"),
        (
            "sealed/condition_training_labels.json",
            training_labels_loaded,
            "training labels",
        ),
    ):
        _validate_manifest_file_receipt(
            manifest_files[relative_path],
            loaded,
            field_name,
        )
    _validate_publication_lineage(
        publication,
        public_corpus=public_corpus_loaded.payload,
        predictor_request=predictor_loaded.payload,
        training_labels=training_labels_loaded.payload,
        manifest_files=manifest_files,
    )

    training_corpus, predictor_rows = _build_prediction_capsule_inputs(
        public_corpus=public_corpus_loaded.payload,
        predictor_request=predictor_loaded.payload,
        training_labels=training_labels_loaded.payload,
        qualification_protocol_id=qualification_protocol_id,
    )
    training_raw = _canonical_artifact_bytes(training_corpus)
    child_request = _build_child_request(
        qualification_protocol_id=qualification_protocol_id,
        execution_protocol_id=execution_id,
        public_corpus_artifact_id=_digest(
            public_corpus_loaded.payload["artifact_id"],
            "public corpus artifact_id",
        ),
        training_corpus=training_corpus,
        training_raw=training_raw,
        predictor_request=predictor_loaded.payload,
        predictor_raw=predictor_loaded.raw,
        group_split_artifact_id=_digest(
            publication["group_split_artifact_id"],
            "group split artifact_id",
        ),
    )
    child_request_raw = _canonical_artifact_bytes(child_request)

    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    capsule = root / "predictor_capsule"
    capsule.mkdir()
    training_path = capsule / "training_corpus.json"
    child_request_path = capsule / "child_request.json"
    predictor_path = capsule / "predictor_request.json"
    _write_bytes_create_only(training_path, training_raw)
    _write_bytes_create_only(child_request_path, child_request_raw)
    _write_bytes_create_only(predictor_path, predictor_loaded.raw)

    launch_specs = tuple(
        _PredictionChildLaunchSpec(
            child_request_path=child_request_path,
            expected_child_request_artifact_id=_digest(
                child_request["artifact_id"],
                "child request artifact_id",
            ),
            training_corpus_path=training_path,
            predictor_request_path=predictor_path,
            output_root=root / "prediction_runs" / f"run-{ordinal}",
            capsule_root=capsule,
            run_ordinal=ordinal,
            run_nonce=hashlib.sha256(f"{parent_run_nonce}:prediction-child:{ordinal}".encode("utf-8")).hexdigest(),
            bge_snapshot_path=(None if bge_snapshot_path is None else pathlib.Path(bge_snapshot_path).resolve()),
            import_binding=import_binding,
            pycache_prefix=capsule / f"pycache-run-{ordinal}",
        )
        for ordinal in (1, 2)
    )

    launch_results: list[_PredictionChildLaunchResult] = []
    integrity_receipts: list[Mapping[str, object]] = []
    previous_integrity_id = initial_integrity_receipt_id
    for spec in launch_specs:
        before_phase = f"pre_prediction_child_{spec.run_ordinal}"
        after_phase = f"post_prediction_child_{spec.run_ordinal}"
        before_receipt = _run_integrity_guard(
            integrity_guard,
            phase=before_phase,
            execution_protocol_id=execution_id,
            expected_integrity_ids=expected_integrity_ids,
            previous_integrity_receipt_artifact_id=previous_integrity_id,
        )
        _write_artifact_create_only(
            root / "integrity_receipts" / f"{before_phase}.json",
            before_receipt,
        )
        integrity_receipts.append(before_receipt)
        previous_integrity_id = _digest(
            before_receipt["artifact_id"],
            "pre-child integrity receipt artifact_id",
        )
        _validate_integrity_receipt_consistency(integrity_receipts)
        try:
            result = launcher(spec)
        finally:
            after_receipt = _run_integrity_guard(
                integrity_guard,
                phase=after_phase,
                execution_protocol_id=execution_id,
                expected_integrity_ids=expected_integrity_ids,
                previous_integrity_receipt_artifact_id=previous_integrity_id,
            )
            _write_artifact_create_only(
                root / "integrity_receipts" / f"{after_phase}.json",
                after_receipt,
            )
            integrity_receipts.append(after_receipt)
            previous_integrity_id = _digest(
                after_receipt["artifact_id"],
                "post-child integrity receipt artifact_id",
            )
            _validate_integrity_receipt_consistency(integrity_receipts)
        _validate_launch_result(result, spec=spec)
        launch_results.append(result)
    _validate_integrity_receipt_sequence(
        integrity_receipts,
        initial_previous_integrity_receipt_artifact_id=(initial_integrity_receipt_id),
    )
    if len({result.process_id for result in launch_results}) != 2:
        raise RuntimeError("prediction children must have distinct process ids")

    launcher_attestation = _with_artifact_id(
        {
            "schema_version": ("relationship-condition-reader-qualification-launcher-attestation.v1"),
            "qualification_protocol_id": qualification_protocol_id,
            "execution_protocol_id": execution_id,
            "child_request_artifact_id": child_request["artifact_id"],
            "run_nonce": parent_run_nonce,
            "runs": [
                _launcher_run_attestation_payload(spec=spec, result=result)
                for spec, result in zip(launch_specs, launch_results, strict=True)
            ],
            "run_count": 2,
            "integrity_receipts": list(integrity_receipts),
            "integrity_receipt_count": 4,
            "integrity_phase_order": [
                "pre_prediction_child_1",
                "post_prediction_child_1",
                "pre_prediction_child_2",
                "post_prediction_child_2",
            ],
            "previous_integrity_receipt_artifact_id": (initial_integrity_receipt_id),
            "last_integrity_receipt_artifact_id": previous_integrity_id,
            "expected_integrity_artifact_ids": expected_integrity_ids,
            "processes_created_suspended": True,
            "job_assigned_before_initial_thread_resume": True,
            "job_kill_on_close": True,
            "job_active_process_limit": 1,
            "shell": False,
            "close_fds": True,
            "environment_built_from_empty_allowlist": True,
            "source_capsule_used": False,
            "repository_import_path_used": True,
            "bge_snapshot_tree_verified_by_launcher": False,
            "external_execution_anchor_verified": False,
            "qualification_execution_authorized": False,
            "os_security_boundary": False,
            "windows_directory_entry_durability_attested": False,
        }
    )
    launcher_attestation_path = root / "launcher_attestation.json"
    _write_artifact_create_only(launcher_attestation_path, launcher_attestation)

    # Only after both children have exited and both Job Objects are empty may
    # the parent inspect any prediction output.
    verified = tuple(
        _verify_prediction_child_output(
            spec=spec,
            launch_result=result,
            child_request=child_request,
            training_corpus=training_corpus,
            predictor_rows=predictor_rows,
            artifact_loader=artifact_loader,
        )
        for spec, result in zip(launch_specs, launch_results, strict=True)
    )
    first, second = verified
    if first.deterministic_raw != second.deterministic_raw:
        raise RuntimeError("two prediction children did not produce byte-exact deterministic outputs")
    if (
        first.manifest_artifact_id == second.manifest_artifact_id
        or first.attestation_artifact_id == second.attestation_artifact_id
    ):
        raise RuntimeError("fresh prediction run manifests/attestations must be distinct")

    commit_root = root / "commit"
    committed_ledger_path = commit_root / "prediction_ledger.json"
    ledger_write = _write_bytes_create_only(
        committed_ledger_path,
        first.deterministic_raw[2],
    )
    committed_ledger = artifact_loader(
        committed_ledger_path,
        expected_schema_version=_PREDICTION_LEDGER_SCHEMA_VERSION,
        max_bytes=_MAX_LEDGER_BYTES,
    )
    if committed_ledger.raw != first.deterministic_raw[2] or committed_ledger.payload != first.ledger_payload:
        raise RuntimeError("committed prediction ledger changed after closed reopen")

    commit_receipt = _with_artifact_id(
        {
            "schema_version": (RELATIONSHIP_READER_PREDICTION_LEDGER_COMMIT_SCHEMA_VERSION),
            "qualification_protocol_id": qualification_protocol_id,
            "execution_protocol_id": execution_id,
            "child_request_artifact_id": child_request["artifact_id"],
            "predictor_request_artifact_id": predictor_loaded.payload["artifact_id"],
            "prediction_ledger_artifact_id": committed_ledger.payload["artifact_id"],
            "prediction_ledger_raw_sha256": committed_ledger.raw_sha256,
            "prediction_ledger_raw_bytes": len(committed_ledger.raw),
            "prediction_run_manifest_artifact_ids": [item.manifest_artifact_id for item in verified],
            "prediction_run_attestation_artifact_ids": [item.attestation_artifact_id for item in verified],
            "fresh_process_count": 2,
            "predictor_processes_exited": True,
            "predictor_job_objects_empty": True,
            "embedding_tables_byte_exact": True,
            "reader_artifacts_byte_exact": True,
            "prediction_ledgers_byte_exact": True,
            "ledger_file_fsync_completed": ledger_write["fsync_completed"],
            "ledger_same_descriptor_readback": ledger_write["same_descriptor_readback"],
            "ledger_closed_reopen_readback": ledger_write["closed_reopen_readback"],
            "windows_directory_entry_durability_attested": False,
        }
    )
    commit_receipt_path = commit_root / "commit_receipt.json"
    _write_artifact_create_only(commit_receipt_path, commit_receipt)

    # This independent load is the release gate.  A failed or tampered receipt
    # exits before the scoring request (and therefore before any label path is
    # handed to a scorer) exists.
    reopened_commit = artifact_loader(
        commit_receipt_path,
        expected_schema_version=(RELATIONSHIP_READER_PREDICTION_LEDGER_COMMIT_SCHEMA_VERSION),
        max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
    )
    _validate_reopened_commit_receipt(
        reopened_commit.payload,
        expected=commit_receipt,
    )

    challenge_receipt = manifest_files["sealed/challenge_labels.json"]
    group_receipt = manifest_files["sealed/group_split.json"]
    scoring_request = _with_artifact_id(
        {
            "schema_version": RELATIONSHIP_READER_SCORING_REQUEST_SCHEMA_VERSION,
            "qualification_protocol_id": qualification_protocol_id,
            "execution_protocol_id": execution_id,
            "run_nonce": parent_run_nonce,
            "prediction_ledger_path": str(committed_ledger_path.resolve()),
            "prediction_ledger_artifact_id": committed_ledger.payload["artifact_id"],
            "prediction_ledger_raw_sha256": committed_ledger.raw_sha256,
            "prediction_ledger_raw_bytes": len(committed_ledger.raw),
            "commit_receipt_path": str(commit_receipt_path.resolve()),
            "commit_receipt_artifact_id": reopened_commit.payload["artifact_id"],
            "challenge_labels_path": str((preflight / "sealed/challenge_labels.json").resolve()),
            "challenge_labels_artifact_id": challenge_receipt["artifact_id"],
            "challenge_labels_raw_sha256": challenge_receipt["raw_sha256"],
            "challenge_labels_raw_bytes": challenge_receipt["raw_bytes"],
            "group_split_path": str((preflight / "sealed/group_split.json").resolve()),
            "group_split_artifact_id": group_receipt["artifact_id"],
            "group_split_raw_sha256": group_receipt["raw_sha256"],
            "group_split_raw_bytes": group_receipt["raw_bytes"],
            "minimum_normalized_margin_hex": (0.01).hex(),
        }
    )
    scoring_request_path = root / "scoring_request.json"
    _write_artifact_create_only(scoring_request_path, scoring_request)

    return {
        "schema_version": ("relationship-condition-reader-qualification-executor-result.v1"),
        "qualification_protocol_id": qualification_protocol_id,
        "execution_protocol_id": execution_id,
        "preflight_manifest_artifact_id": expected_manifest_id,
        "publication_request_artifact_id": expected_publication_id,
        "launcher_attestation_artifact_id": launcher_attestation["artifact_id"],
        "last_integrity_receipt_artifact_id": previous_integrity_id,
        "training_corpus_artifact_id": training_corpus["artifact_id"],
        "child_request_artifact_id": child_request["artifact_id"],
        "prediction_ledger_artifact_id": committed_ledger.payload["artifact_id"],
        "commit_receipt_artifact_id": reopened_commit.payload["artifact_id"],
        "scoring_request_artifact_id": scoring_request["artifact_id"],
        "scoring_request_path": str(scoring_request_path.resolve()),
        "fresh_process_count": 2,
        "predictor_processes_exited": True,
        "predictor_job_objects_empty": True,
        "deterministic_outputs_byte_exact": True,
        "parent_opened_challenge_labels": False,
        "parent_opened_group_split": False,
        "scorer_launched": False,
        "os_security_boundary": False,
        "windows_directory_entry_durability_attested": False,
        "qualification_scored": False,
        "external_execution_anchor_verified": False,
        "qualification_execution_authorized": False,
        "reader_development_admission": False,
        "readable_claim_proven": False,
        "four_able_claim_proven": False,
    }


def _build_prediction_capsule_inputs(
    *,
    public_corpus: Mapping[str, object],
    predictor_request: Mapping[str, object],
    training_labels: Mapping[str, object],
    qualification_protocol_id: str,
) -> tuple[Mapping[str, object], tuple[Mapping[str, str], ...]]:
    _exact_keys(
        public_corpus,
        {
            "schema_version",
            "protocol_id",
            "training_inputs",
            "challenge_inputs",
            "training_input_count",
            "challenge_input_count",
            "exact_text_overlap_count",
            "predictor_projection",
            "artifact_id",
        },
        "public corpus",
    )
    if (
        public_corpus["protocol_id"] != qualification_protocol_id
        or _positive_integer(
            public_corpus["training_input_count"],
            "training_input_count",
        )
        != _TRAINING_COUNT
        or _positive_integer(
            public_corpus["challenge_input_count"],
            "challenge_input_count",
        )
        != _CHALLENGE_COUNT
        or _nonnegative_integer(
            public_corpus["exact_text_overlap_count"],
            "exact_text_overlap_count",
        )
        != 0
        or public_corpus["predictor_projection"] != "opaque_item_id_exact_text_and_text_sha256_only"
    ):
        raise ValueError("public corpus count, protocol, or projection drifted")
    training_inputs = tuple(
        _parse_opaque_text_row(value, f"training input {index}")
        for index, value in enumerate(_list(public_corpus["training_inputs"], "training_inputs"))
    )
    challenge_inputs = tuple(
        _parse_opaque_text_row(value, f"challenge input {index}")
        for index, value in enumerate(_list(public_corpus["challenge_inputs"], "challenge_inputs"))
    )
    _validate_canonical_text_rows(training_inputs, _TRAINING_COUNT, "training inputs")
    _validate_canonical_text_rows(challenge_inputs, _CHALLENGE_COUNT, "challenge inputs")
    if {row["item_id"] for row in training_inputs} & {row["item_id"] for row in challenge_inputs}:
        raise ValueError("training and challenge item ids overlap")
    if {row["text_sha256"] for row in training_inputs} & {row["text_sha256"] for row in challenge_inputs}:
        raise ValueError("training and challenge exact texts overlap")

    _exact_keys(
        predictor_request,
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
    predictor_rows = tuple(
        _parse_opaque_text_row(value, f"predictor challenge {index}")
        for index, value in enumerate(_list(predictor_request["challenge_inputs"], "predictor challenge_inputs"))
    )
    if (
        predictor_request["protocol_id"] != qualification_protocol_id
        or predictor_request["public_corpus_artifact_id"] != public_corpus["artifact_id"]
        or _positive_integer(
            predictor_request["challenge_input_count"],
            "predictor challenge_input_count",
        )
        != _CHALLENGE_COUNT
        or predictor_rows != challenge_inputs
    ):
        raise ValueError("predictor request is not the exact public challenge projection")

    _exact_keys(
        training_labels,
        {
            "schema_version",
            "protocol_id",
            "public_corpus_artifact_id",
            "rows",
            "row_count",
            "labels",
            "condition_only",
            "action_outcome_pe_credit_evaluation_present",
            "artifact_id",
        },
        "training labels",
    )
    raw_labels = _list(training_labels["labels"], "training labels labels")
    if tuple(raw_labels) != _LABELS:
        raise ValueError("training label taxonomy drifted")
    if (
        training_labels["protocol_id"] != qualification_protocol_id
        or training_labels["public_corpus_artifact_id"] != public_corpus["artifact_id"]
        or _positive_integer(training_labels["row_count"], "training row_count") != _TRAINING_COUNT
        or training_labels["condition_only"] is not True
        or training_labels["action_outcome_pe_credit_evaluation_present"] is not False
    ):
        raise ValueError("training labels lineage or condition-only boundary drifted")
    label_rows: dict[str, Mapping[str, object]] = {}
    source_positions: set[int] = set()
    counts: Counter[str] = Counter()
    for index, value in enumerate(_list(training_labels["rows"], "training rows")):
        row = _mapping(value, f"training label row {index}")
        _exact_keys(
            row,
            {"item_id", "text_sha256", "condition_label", "source_position"},
            f"training label row {index}",
        )
        item_id = _digest(row["item_id"], "training label item_id")
        if item_id in label_rows:
            raise ValueError("training label item ids must be unique")
        label = _text(row["condition_label"], "condition_label")
        if label not in _LABELS:
            raise ValueError("training label is outside the frozen taxonomy")
        position = _nonnegative_integer(row["source_position"], "source_position")
        source_positions.add(position)
        counts[label] += 1
        label_rows[item_id] = row
    if (
        source_positions != set(range(_TRAINING_COUNT))
        or counts != Counter({label: 2 for label in _LABELS})
        or tuple(label_rows) != tuple(sorted(label_rows))
    ):
        raise ValueError("training labels must remain canonical, balanced, and complete")

    training_rows: list[dict[str, str]] = []
    for public_row in training_inputs:
        label_row = label_rows.get(public_row["item_id"])
        if label_row is None:
            raise ValueError("training public input has no condition label")
        if label_row["text_sha256"] != public_row["text_sha256"]:
            raise ValueError("training label text hash differs from public input")
        training_rows.append(
            {
                **public_row,
                "condition_label": _text(
                    label_row["condition_label"],
                    "condition_label",
                ),
            }
        )
    if set(label_rows) != {row["item_id"] for row in training_inputs}:
        raise ValueError("training labels and public inputs do not form an exact join")
    training_corpus = _with_artifact_id(
        {
            "schema_version": _TRAINING_CORPUS_SCHEMA_VERSION,
            "protocol_id": qualification_protocol_id,
            "public_corpus_artifact_id": public_corpus["artifact_id"],
            "labels": list(_LABELS),
            "rows": training_rows,
            "row_count": _TRAINING_COUNT,
            "condition_only": True,
        }
    )
    return training_corpus, predictor_rows


def _build_child_request(
    *,
    qualification_protocol_id: str,
    execution_protocol_id: str,
    public_corpus_artifact_id: str,
    training_corpus: Mapping[str, object],
    training_raw: bytes,
    predictor_request: Mapping[str, object],
    predictor_raw: bytes,
    group_split_artifact_id: str,
) -> Mapping[str, object]:
    return _with_artifact_id(
        {
            "schema_version": _CHILD_REQUEST_SCHEMA_VERSION,
            "protocol_id": qualification_protocol_id,
            "execution_protocol_id": execution_protocol_id,
            "public_corpus_artifact_id": public_corpus_artifact_id,
            "training_corpus_artifact_id": training_corpus["artifact_id"],
            "training_corpus_raw_sha256": hashlib.sha256(training_raw).hexdigest(),
            "training_corpus_raw_bytes": len(training_raw),
            "predictor_request_artifact_id": predictor_request["artifact_id"],
            "predictor_request_raw_sha256": hashlib.sha256(predictor_raw).hexdigest(),
            "predictor_request_raw_bytes": len(predictor_raw),
            "group_split_artifact_id": group_split_artifact_id,
            "semantic_model": {
                "model_id": _BGE_M3_MODEL_ID,
                "model_revision": _BGE_M3_MODEL_REVISION,
                "weights_sha256": _BGE_M3_WEIGHT_BYTES_SHA256,
                "sentence_transformers_version": (_BGE_M3_SENTENCE_TRANSFORMERS_VERSION),
                "embedding_width": _BGE_M3_EMBEDDING_WIDTH,
                "device": "cuda",
                "network_allowed": False,
                "stub_allowed": False,
            },
            "reader": {
                "schema_version": _READER_ARTIFACT_SCHEMA_VERSION,
                "solver": _READER_SOLVER,
                "solver_version": _READER_SOLVER_VERSION,
                "labels": list(_LABELS),
            },
            "required_live_embedding_count": _LIVE_EMBEDDING_COUNT,
        }
    )


def _verify_prediction_child_output(
    *,
    spec: _PredictionChildLaunchSpec,
    launch_result: _PredictionChildLaunchResult,
    child_request: Mapping[str, object],
    training_corpus: Mapping[str, object],
    predictor_rows: tuple[Mapping[str, str], ...],
    artifact_loader: _ArtifactLoader,
) -> _VerifiedPredictionChild:
    root = spec.output_root.resolve()
    loaded = {
        "embedding_table.json": artifact_loader(
            root / "embedding_table.json",
            expected_schema_version=_EMBEDDING_TABLE_SCHEMA_VERSION,
            max_bytes=_MAX_CHILD_OUTPUT_BYTES,
        ),
        "reader_artifact.json": artifact_loader(
            root / "reader_artifact.json",
            expected_schema_version=_READER_ARTIFACT_SCHEMA_VERSION,
            max_bytes=_MAX_CHILD_OUTPUT_BYTES,
        ),
        "prediction_ledger.json": artifact_loader(
            root / "prediction_ledger.json",
            expected_schema_version=_PREDICTION_LEDGER_SCHEMA_VERSION,
            max_bytes=_MAX_LEDGER_BYTES,
        ),
        "process_attestation.json": artifact_loader(
            root / "process_attestation.json",
            expected_schema_version=_PREDICTION_ATTESTATION_SCHEMA_VERSION,
            max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
        ),
        "manifest.json": artifact_loader(
            root / "manifest.json",
            expected_schema_version=_PREDICTION_MANIFEST_SCHEMA_VERSION,
            max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
        ),
    }
    table = loaded["embedding_table.json"].payload
    reader = loaded["reader_artifact.json"].payload
    ledger = loaded["prediction_ledger.json"].payload
    attestation = loaded["process_attestation.json"].payload
    manifest = loaded["manifest.json"].payload
    parsed_table = _validate_embedding_table(
        table,
        child_request=child_request,
        training_corpus=training_corpus,
        predictor_rows=predictor_rows,
    )
    parsed_reader = _validate_reader_artifact(
        reader,
        child_request=child_request,
        training_corpus=training_corpus,
    )
    _validate_prediction_ledger(
        ledger,
        child_request=child_request,
        predictor_rows=predictor_rows,
        embedding_table_artifact_id=_digest(table["artifact_id"], "table artifact_id"),
        reader_artifact_id=_digest(reader["artifact_id"], "reader artifact_id"),
        parsed_embedding_table=parsed_table,
        parsed_reader_artifact=parsed_reader,
    )
    _validate_prediction_attestation(
        attestation,
        spec=spec,
        launch_result=launch_result,
        child_request=child_request,
    )
    _validate_prediction_manifest(
        manifest,
        spec=spec,
        child_request=child_request,
        loaded=loaded,
    )
    return _VerifiedPredictionChild(
        process_id=launch_result.process_id,
        manifest_artifact_id=_digest(manifest["artifact_id"], "manifest artifact_id"),
        attestation_artifact_id=_digest(
            attestation["artifact_id"],
            "attestation artifact_id",
        ),
        embedding_table_artifact_id=_digest(table["artifact_id"], "table artifact_id"),
        reader_artifact_id=_digest(reader["artifact_id"], "reader artifact_id"),
        prediction_ledger_artifact_id=_digest(
            ledger["artifact_id"],
            "prediction ledger artifact_id",
        ),
        deterministic_raw=tuple(loaded[path].raw for path in _DETERMINISTIC_OUTPUT_PATHS),
        ledger_payload=ledger,
    )


def _validate_embedding_table(
    payload: Mapping[str, object],
    *,
    child_request: Mapping[str, object],
    training_corpus: Mapping[str, object],
    predictor_rows: tuple[Mapping[str, str], ...],
) -> object:
    from lifeform_evolution.relationship_lab_product_model_adapters import (
        PrecomputedPublicEmbeddingTable,
    )

    parsed = PrecomputedPublicEmbeddingTable.from_payload(dict(payload))
    model = _mapping(child_request["semantic_model"], "semantic_model")
    if (
        parsed.source_model_id != model["model_id"]
        or parsed.source_model_revision != model["model_revision"]
        or parsed.source_weights_sha256 != model["weights_sha256"]
        or parsed.source_sentence_transformers_version != model["sentence_transformers_version"]
        or parsed.embedding_width != model["embedding_width"]
    ):
        raise ValueError("prediction embedding table model identity drifted")
    training_rows = _list(training_corpus["rows"], "training corpus rows")
    expected_text_sha256s = {
        _digest(
            _mapping(row, "training row")["text_sha256"],
            "training text_sha256",
        )
        for row in training_rows
    } | {row["text_sha256"] for row in predictor_rows}
    if (
        len(parsed.records) != _LIVE_EMBEDDING_COUNT
        or {record.text_sha256 for record in parsed.records} != expected_text_sha256s
    ):
        raise ValueError("prediction embedding table must contain 228 records")
    return parsed


def _validate_reader_artifact(
    payload: Mapping[str, object],
    *,
    child_request: Mapping[str, object],
    training_corpus: Mapping[str, object],
) -> object:
    from lifeform_domain_emogpt.relationship_condition_reader import (
        FrozenLinearRelationshipConditionReaderArtifact,
    )

    parsed = FrozenLinearRelationshipConditionReaderArtifact.from_payload(dict(payload))
    model = _mapping(child_request["semantic_model"], "semantic_model")
    expected = {
        "embedding_model_id": model["model_id"],
        "embedding_model_revision": model["model_revision"],
        "embedding_weights_sha256": model["weights_sha256"],
        "embedding_runtime_version": model["sentence_transformers_version"],
        "embedding_width": model["embedding_width"],
        "labels": list(_LABELS),
        "condition_training_corpus_artifact_id": child_request["training_corpus_artifact_id"],
        "condition_training_corpus_raw_sha256": child_request["training_corpus_raw_sha256"],
        "group_split_artifact_id": child_request["group_split_artifact_id"],
        "solver": _READER_SOLVER,
        "solver_version": _READER_SOLVER_VERSION,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("prediction reader artifact lineage or solver drifted")
    training_by_label: dict[str, list[str]] = {label: [] for label in _LABELS}
    for value in _list(training_corpus["rows"], "training corpus rows"):
        row = _mapping(value, "training corpus row")
        training_by_label[_text(row["condition_label"], "training label")].append(
            _digest(row["item_id"], "training item_id")
        )
    expected_example_hashes = {
        label: _sha256_json({"example_ids": sorted(item_ids)}) for label, item_ids in training_by_label.items()
    }
    if any(
        parameter.example_count != 2 or parameter.example_ids_sha256 != expected_example_hashes[parameter.label]
        for parameter in parsed.class_parameters
    ):
        raise ValueError("prediction reader training example lineage drifted")
    return parsed


def _validate_prediction_ledger(
    payload: Mapping[str, object],
    *,
    child_request: Mapping[str, object],
    predictor_rows: tuple[Mapping[str, str], ...],
    embedding_table_artifact_id: str,
    reader_artifact_id: str,
    parsed_embedding_table: object,
    parsed_reader_artifact: object,
) -> None:
    from lifeform_domain_emogpt.relationship_condition_reader import (
        FrozenLinearRelationshipConditionReaderArtifact,
        FrozenLinearRelationshipConditionReaderRuntime,
    )
    from lifeform_evolution.relationship_lab_product_model_adapters import (
        PrecomputedPublicEmbeddingTable,
        PrecomputedPublicSemanticEmbedder,
    )
    from volvence_zero.social_cognition import (
        relationship_condition_readout_from_payload,
        relationship_condition_readout_to_payload,
    )

    if not isinstance(parsed_embedding_table, PrecomputedPublicEmbeddingTable):
        raise TypeError("parsed embedding table has an unsupported type")
    if not isinstance(
        parsed_reader_artifact,
        FrozenLinearRelationshipConditionReaderArtifact,
    ):
        raise TypeError("parsed reader artifact has an unsupported type")
    replay_runtime = FrozenLinearRelationshipConditionReaderRuntime(
        artifact=parsed_reader_artifact,
        embedder=PrecomputedPublicSemanticEmbedder(parsed_embedding_table),
    )
    _exact_keys(
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
    if (
        payload["protocol_id"] != child_request["protocol_id"]
        or payload["execution_protocol_id"] != child_request["execution_protocol_id"]
        or payload["child_request_artifact_id"] != child_request["artifact_id"]
        or payload["predictor_request_artifact_id"] != child_request["predictor_request_artifact_id"]
        or payload["embedding_table_artifact_id"] != embedding_table_artifact_id
        or payload["reader_artifact_id"] != reader_artifact_id
        or _positive_integer(payload["row_count"], "prediction row_count") != _CHALLENGE_COUNT
        or payload["challenge_labels_present"] is not False
        or payload["qualification_scored"] is not False
    ):
        raise ValueError("prediction ledger identity, count, or firewall drifted")
    expected_by_id = {row["item_id"]: row for row in predictor_rows}
    observed_ids: list[str] = []
    for index, value in enumerate(_list(payload["rows"], "prediction rows")):
        row = _mapping(value, f"prediction row {index}")
        _exact_keys(
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
            f"prediction row {index}",
        )
        item_id = _digest(row["item_id"], "prediction item_id")
        expected = expected_by_id.get(item_id)
        if expected is None:
            raise ValueError("prediction ledger contains an unknown item")
        if (
            row["text_sha256"] != expected["text_sha256"]
            or row["source_observation_sha256"] != expected["text_sha256"]
            or row["reader_artifact_id"] != reader_artifact_id
            or row["condition_label"] not in _LABELS
        ):
            raise ValueError("prediction row lineage or label drifted")
        supplied_confidence = _canonical_hex_float(
            row["confidence_hex"],
            "confidence_hex",
        )
        supplied_margin = _canonical_hex_float(
            row["normalized_margin_hex"],
            "normalized_margin_hex",
        )
        candidates = _list(row["candidate_scores"], "candidate_scores")
        if len(candidates) != 2:
            raise ValueError("prediction row must contain exactly two candidates")
        candidate_labels: list[str] = []
        supplied_candidate_scores: list[dict[str, object]] = []
        for candidate_value in candidates:
            candidate = _mapping(candidate_value, "candidate score")
            _exact_keys(candidate, {"label", "score_hex"}, "candidate score")
            candidate_labels.append(_text(candidate["label"], "candidate label"))
            supplied_candidate_scores.append(
                {
                    "label": candidate["label"],
                    "score": _canonical_hex_float(
                        candidate["score_hex"],
                        "score_hex",
                    ),
                }
            )
        if tuple(candidate_labels) != _LABELS:
            raise ValueError("prediction candidate label order drifted")
        relationship_condition_readout_from_payload(
            {
                "condition_label": row["condition_label"],
                "confidence": supplied_confidence,
                "normalized_margin": supplied_margin,
                "candidate_scores": supplied_candidate_scores,
                "reader_artifact_id": row["reader_artifact_id"],
                "source_observation_sha256": row["source_observation_sha256"],
            }
        )
        replay = relationship_condition_readout_to_payload(replay_runtime.read_condition(expected["text"]))
        expected_row = {
            "item_id": item_id,
            "text_sha256": expected["text_sha256"],
            "condition_label": replay["condition_label"],
            "confidence_hex": _canonical_float_hex(replay["confidence"]),
            "normalized_margin_hex": _canonical_float_hex(replay["normalized_margin"]),
            "candidate_scores": [
                {
                    "label": _mapping(score, "replay candidate score")["label"],
                    "score_hex": _canonical_float_hex(_mapping(score, "replay candidate score")["score"]),
                }
                for score in _list(replay["candidate_scores"], "replay candidates")
            ],
            "reader_artifact_id": replay["reader_artifact_id"],
            "source_observation_sha256": replay["source_observation_sha256"],
        }
        if row != expected_row:
            raise ValueError("prediction ledger row is not a byte-exact model-free reader replay")
        observed_ids.append(item_id)
    if observed_ids != sorted(expected_by_id):
        raise ValueError("prediction ledger rows must use canonical item-id order")


def _validate_prediction_attestation(
    payload: Mapping[str, object],
    *,
    spec: _PredictionChildLaunchSpec,
    launch_result: _PredictionChildLaunchResult,
    child_request: Mapping[str, object],
) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "protocol_id",
            "execution_protocol_id",
            "child_request_artifact_id",
            "run_ordinal",
            "run_nonce",
            "process_id",
            "parent_process_id",
            "python_executable",
            "python_implementation",
            "python_version",
            "argv",
            "interpreter_flags",
            "pycache_prefix",
            "working_directory",
            "sys_path",
            "bootstrap_import_roots",
            "environment_contract",
            "environment_projection",
            "environment_key_names",
            "environment_value_sha256s",
            "loaded_file_backed_module_origins",
            "volvence_zero_namespace_search_locations",
            "embedder_factory_kind",
            "model",
            "live_embedding_call_count",
            "training_embedding_count",
            "challenge_embedding_count",
            "prediction_ledger_fsync_completed",
            "forbidden_module_observations",
            "deterministic_outputs",
            "os_security_boundary",
            "artifact_id",
        },
        "prediction process attestation",
    )
    required = {
        "schema_version": _PREDICTION_ATTESTATION_SCHEMA_VERSION,
        "protocol_id": child_request["protocol_id"],
        "execution_protocol_id": child_request["execution_protocol_id"],
        "child_request_artifact_id": child_request["artifact_id"],
        "run_ordinal": spec.run_ordinal,
        "run_nonce": spec.run_nonce,
        "process_id": launch_result.process_id,
        "embedder_factory_kind": "formal_bge_m3_cuda",
        "model": child_request["semantic_model"],
        "live_embedding_call_count": _LIVE_EMBEDDING_COUNT,
        "training_embedding_count": _TRAINING_COUNT,
        "challenge_embedding_count": _CHALLENGE_COUNT,
        "prediction_ledger_fsync_completed": True,
        "os_security_boundary": False,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("prediction process attestation lineage or honesty drifted")
    if _positive_integer(payload["parent_process_id"], "prediction parent_process_id") == launch_result.process_id:
        raise ValueError("prediction child parent_process_id must differ from its own process_id")
    binding = spec.import_binding
    child_executable = pathlib.Path(_text(payload["python_executable"], "prediction python_executable"))
    if (
        not child_executable.is_absolute()
        or child_executable.resolve(strict=True) != binding.python_executable
        or str(child_executable) != str(binding.python_executable)
    ):
        raise ValueError("prediction child Python executable differs from the frozen runtime")
    if payload["python_implementation"] != "CPython":
        raise ValueError("prediction child must use CPython")
    python_version = _text(payload["python_version"], "prediction python_version")
    if payload["argv"] != _prediction_child_sys_argv(spec):
        raise ValueError("prediction child sys.argv differs from the frozen launcher argv")
    flags = _mapping(payload["interpreter_flags"], "prediction interpreter_flags")
    if (
        set(flags)
        != {
            "safe_path",
            "no_site",
            "dont_write_bytecode",
            "utf8_mode",
            "isolated",
            "ignore_environment",
            "stdout_write_through",
            "stderr_write_through",
        }
        or flags["safe_path"] is not True
        or flags["stdout_write_through"] is not True
        or flags["stderr_write_through"] is not True
        or type(flags["no_site"]) is not int
        or flags["no_site"] != 1
        or type(flags["dont_write_bytecode"]) is not int
        or flags["dont_write_bytecode"] != 1
        or type(flags["utf8_mode"]) is not int
        or flags["utf8_mode"] != 1
        or type(flags["isolated"]) is not int
        or flags["isolated"] != 0
        or type(flags["ignore_environment"]) is not int
        or flags["ignore_environment"] != 0
    ):
        raise ValueError("prediction child interpreter flags drifted")
    if payload["pycache_prefix"] != str(spec.pycache_prefix):
        raise ValueError("prediction child pycache prefix drifted")
    if (
        pathlib.Path(_text(payload.get("working_directory"), "prediction working_directory")).resolve()
        != spec.capsule_root.resolve()
    ):
        raise ValueError("prediction child working directory escaped the capsule")
    expected_import_roots = [str(path) for path in binding.import_roots]
    if payload["bootstrap_import_roots"] != expected_import_roots:
        raise ValueError("prediction child bootstrap import roots drifted")
    expected_sys_path = list(
        expected_child_sys_path(
            binding,
            python_version=python_version,
        )
    )
    if payload["sys_path"] != expected_sys_path:
        raise ValueError("prediction child sys.path differs from the complete frozen path")
    environment_contract = _mapping(
        payload.get("environment_contract"),
        "prediction environment_contract",
    )
    _exact_keys(
        environment_contract,
        {
            "schema_version",
            "projected_keys",
            "all_environment_values_hashed",
            "unlisted_environment_variables_recorded",
        },
        "prediction environment_contract",
    )
    expected_projected_keys = list(_PREDICTION_ENVIRONMENT_PROJECTION_KEYS)
    if (
        environment_contract["schema_version"] != "relationship-condition-reader-prediction-environment.v2"
        or environment_contract["projected_keys"] != expected_projected_keys
        or environment_contract["all_environment_values_hashed"] is not True
        or environment_contract["unlisted_environment_variables_recorded"] is not True
    ):
        raise ValueError("prediction child environment contract drifted")
    child_environment = _mapping(
        payload.get("environment_projection"),
        "prediction environment_projection",
    )
    if set(child_environment) != set(expected_projected_keys):
        raise ValueError("prediction child environment projection key set drifted")
    launcher_environment = {item.key: item for item in launch_result.environment_projection}
    for key in expected_projected_keys:
        child_value = child_environment[key]
        launcher_receipt = launcher_environment.get(key)
        if child_value is None:
            if launcher_receipt is not None:
                raise ValueError("prediction child omitted a launcher-projected environment value")
            continue
        value = _text(child_value, f"prediction environment {key}")
        if (
            launcher_receipt is None
            or launcher_receipt.value_sha256 != hashlib.sha256(value.encode("utf-8")).hexdigest()
            or launcher_receipt.value_utf8_bytes != len(value.encode("utf-8"))
        ):
            raise ValueError("prediction child environment value differs from launcher projection")
    expected_python_path = os.pathsep.join(expected_import_roots)
    if child_environment["PYTHONPATH"] != expected_python_path:
        raise ValueError("prediction child controlled PYTHONPATH drifted")
    if child_environment["PYTHONPYCACHEPREFIX"] != str(spec.pycache_prefix):
        raise ValueError("prediction child PYTHONPYCACHEPREFIX drifted")
    environment_key_names = _list(
        payload["environment_key_names"],
        "prediction environment_key_names",
    )
    expected_environment_key_names = list(launcher_environment)
    if environment_key_names != expected_environment_key_names:
        raise ValueError("prediction child complete environment key set drifted")
    environment_value_sha256s = _mapping(
        payload["environment_value_sha256s"],
        "prediction environment_value_sha256s",
    )
    if set(environment_value_sha256s) != set(expected_environment_key_names):
        raise ValueError("prediction child environment hash key set drifted")
    for key in expected_environment_key_names:
        if environment_value_sha256s[key] != launcher_environment[key].value_sha256:
            raise ValueError("prediction child complete environment value hash drifted")
    validate_child_file_backed_module_origin_attestation(
        loaded_module_origins=payload["loaded_file_backed_module_origins"],
        volvence_zero_namespace_search_locations=(payload["volvence_zero_namespace_search_locations"]),
        binding=binding,
        required_module_names=_PREDICTION_REQUIRED_REPOSITORY_MODULES,
    )
    observations = _list(
        payload.get("forbidden_module_observations"),
        "forbidden_module_observations",
    )
    if {item.get("module_name") for item in observations if isinstance(item, dict)} != (_FORBIDDEN_WORKER_MODULES):
        raise ValueError("prediction forbidden-module observation set drifted")
    for value in observations:
        observation = _mapping(value, "forbidden module observation")
        if any(
            observation.get(field_name) is not False
            for field_name in (
                "loaded_at_worker_entry",
                "loaded_at_worker_exit",
                "imported_by_worker",
            )
        ):
            raise ValueError("prediction child observed a forbidden module")


def _validate_prediction_manifest(
    payload: Mapping[str, object],
    *,
    spec: _PredictionChildLaunchSpec,
    child_request: Mapping[str, object],
    loaded: Mapping[str, _LoadedArtifact],
) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "protocol_id",
            "execution_protocol_id",
            "child_request_artifact_id",
            "files",
            "file_count",
            "deterministic_file_paths",
            "prediction_ledger_fsync_completed",
            "artifact_id",
        },
        "prediction manifest",
    )
    if (
        payload["protocol_id"] != child_request["protocol_id"]
        or payload["execution_protocol_id"] != child_request["execution_protocol_id"]
        or payload["child_request_artifact_id"] != child_request["artifact_id"]
        or payload["deterministic_file_paths"] != list(_DETERMINISTIC_OUTPUT_PATHS)
        or payload["prediction_ledger_fsync_completed"] is not True
    ):
        raise ValueError("prediction manifest lineage or deterministic contract drifted")
    files = _list(payload["files"], "prediction manifest files")
    if _positive_integer(payload["file_count"], "prediction file_count") != 4:
        raise ValueError("prediction manifest must bind four pre-manifest files")
    receipts: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(files):
        receipt = _mapping(value, f"prediction manifest file {index}")
        _exact_keys(
            receipt,
            {"path", "raw_sha256", "raw_bytes", "artifact_id"},
            f"prediction manifest file {index}",
        )
        path = _text(receipt["path"], "prediction manifest path")
        if path in receipts:
            raise ValueError("prediction manifest file paths must be unique")
        receipts[path] = receipt
    expected_paths = {*_DETERMINISTIC_OUTPUT_PATHS, "process_attestation.json"}
    if set(receipts) != expected_paths:
        raise ValueError("prediction manifest file set drifted")
    for relative_path in expected_paths:
        _validate_manifest_file_receipt(
            receipts[relative_path],
            loaded[relative_path],
            f"prediction output {relative_path}",
        )
    attestation_outputs = loaded["process_attestation.json"].payload.get("deterministic_outputs")
    if attestation_outputs != [receipts[path] for path in _DETERMINISTIC_OUTPUT_PATHS]:
        raise ValueError("prediction attestation and manifest receipts disagree")
    if spec.output_root.resolve() != spec.output_root:
        raise ValueError("prediction child output root must be absolute")


def _validate_launch_result(
    result: _PredictionChildLaunchResult,
    *,
    spec: _PredictionChildLaunchSpec,
) -> None:
    if not isinstance(result, _PredictionChildLaunchResult):
        raise TypeError("prediction launcher returned an unsupported result")
    if isinstance(result.process_id, bool) or result.process_id < 1:
        raise RuntimeError("prediction child process id is invalid")
    if result.process_argv != _prediction_child_process_argv(spec):
        raise RuntimeError("prediction child process argv drifted")
    if result.process_exited is not True:
        raise RuntimeError(f"prediction child {spec.run_ordinal} did not fully exit")
    if result.exit_code != 0:
        diagnostic = result.stderr_capture.retained_prefix.decode(
            "utf-8",
            errors="replace",
        )[:1_000]
        raise RuntimeError(
            f"prediction child {spec.run_ordinal} failed with exit code "
            f"{result.exit_code}; bounded stderr prefix={diagnostic!r}"
        )
    if result.job_object_empty is not True:
        raise RuntimeError(f"prediction child {spec.run_ordinal} Job Object is not empty")
    if os.path.lexists(spec.pycache_prefix):
        raise RuntimeError("prediction child pycache prefix was materialized")
    if (
        result.creation_flags != _FORMAL_CREATION_FLAGS
        or result.shell is not False
        or result.close_fds is not True
        or result.process_created_suspended is not True
        or result.job_assigned_before_resume is not True
        or result.initial_thread_resume_previous_count != 1
        or result.job_limit_flags != _FORMAL_JOB_LIMIT_FLAGS
        or result.job_active_process_limit != 1
    ):
        raise RuntimeError(f"prediction child {spec.run_ordinal} process containment contract drifted")
    _digest(result.environment_contract_id, "environment_contract_id")
    environment_keys = tuple(item.key for item in result.environment_projection)
    if environment_keys != tuple(sorted(environment_keys)) or len(set(environment_keys)) != len(environment_keys):
        raise RuntimeError("prediction child environment projection is not canonical")
    for item in result.environment_projection:
        _text(item.key, "environment key")
        _digest(item.value_sha256, "environment value_sha256")
        _nonnegative_integer(item.value_utf8_bytes, "environment value_utf8_bytes")
    if result.environment_contract_id != _environment_contract_id(result.environment_projection):
        raise RuntimeError("prediction child environment contract id mismatch")
    for capture_name, capture in (
        ("stdout", result.stdout_capture),
        ("stderr", result.stderr_capture),
    ):
        _validate_stream_capture(capture, field_name=capture_name)
    if (
        result.source_capsule_used is not False
        or result.repository_import_path_used is not True
        or result.bge_snapshot_tree_verified is not False
    ):
        raise RuntimeError("prediction launcher source-boundary honesty drifted")


def _launcher_run_attestation_payload(
    *,
    spec: _PredictionChildLaunchSpec,
    result: _PredictionChildLaunchResult,
) -> Mapping[str, object]:
    return {
        "run_ordinal": spec.run_ordinal,
        "run_nonce": spec.run_nonce,
        "process_id": result.process_id,
        "process_argv": list(result.process_argv),
        "exit_code": result.exit_code,
        "process_exited": result.process_exited,
        "job_object_empty": result.job_object_empty,
        "environment_contract_id": result.environment_contract_id,
        "environment_projection": [
            {
                "key": item.key,
                "value_sha256": item.value_sha256,
                "value_utf8_bytes": item.value_utf8_bytes,
            }
            for item in result.environment_projection
        ],
        "creation_flags": result.creation_flags,
        "shell": result.shell,
        "close_fds": result.close_fds,
        "process_created_suspended": result.process_created_suspended,
        "job_assigned_before_resume": result.job_assigned_before_resume,
        "initial_thread_resume_previous_count": (result.initial_thread_resume_previous_count),
        "job_limit_flags": result.job_limit_flags,
        "job_active_process_limit": result.job_active_process_limit,
        "stdout_capture": _stream_capture_payload(result.stdout_capture),
        "stderr_capture": _stream_capture_payload(result.stderr_capture),
        "source_capsule_used": result.source_capsule_used,
        "repository_import_path_used": result.repository_import_path_used,
        "bge_snapshot_path_sha256": hashlib.sha256(str(spec.bge_snapshot_path).encode("utf-8")).hexdigest(),
        "bge_snapshot_tree_verified": result.bge_snapshot_tree_verified,
    }


def _run_integrity_guard(
    guard: _IntegrityGuard,
    *,
    phase: str,
    execution_protocol_id: str,
    expected_integrity_ids: Mapping[str, str],
    previous_integrity_receipt_artifact_id: str,
) -> Mapping[str, object]:
    previous_id = _digest(
        previous_integrity_receipt_artifact_id,
        "previous integrity receipt artifact_id",
    )
    receipt = _mapping(
        guard(
            phase=phase,
            previous_integrity_receipt_artifact_id=previous_id,
        ),
        f"integrity receipt {phase}",
    )
    _exact_keys(
        receipt,
        {
            "schema_version",
            "execution_protocol_id",
            "phase",
            "phase_ordinal",
            "previous_integrity_receipt_artifact_id",
            "source_tree_artifact_id",
            "source_tree_entry_count",
            "bge_snapshot_tree_artifact_id",
            "bge_snapshot_entry_count",
            "runtime_identity_artifact_id",
            "source_tree_exact",
            "bge_snapshot_tree_exact",
            "runtime_identity_exact",
            "observer_model_or_cuda_execution_used",
            "torch_imported",
            "sentence_transformers_imported",
            "os_security_boundary",
            "windows_directory_entry_durability_attested",
            "artifact_id",
        },
        f"integrity receipt {phase}",
    )
    if (
        receipt["schema_version"] != "relationship-condition-reader-qualification-execution-integrity-receipt.v1"
        or receipt["execution_protocol_id"] != execution_protocol_id
        or receipt["phase"] != phase
        or receipt["phase_ordinal"]
        != {
            "pre_prediction_child_1": 1,
            "post_prediction_child_1": 2,
            "pre_prediction_child_2": 3,
            "post_prediction_child_2": 4,
        }[phase]
        or receipt["previous_integrity_receipt_artifact_id"] != previous_id
        or receipt["source_tree_exact"] is not True
        or receipt["bge_snapshot_tree_exact"] is not True
        or receipt["runtime_identity_exact"] is not True
        or receipt["observer_model_or_cuda_execution_used"] is not False
        or receipt["torch_imported"] is not False
        or receipt["sentence_transformers_imported"] is not False
        or receipt["os_security_boundary"] is not False
        or receipt["windows_directory_entry_durability_attested"] is not False
    ):
        raise ValueError(f"integrity receipt {phase} lineage or verdict drifted")
    _positive_integer(receipt["source_tree_entry_count"], "source tree entry count")
    _positive_integer(receipt["bge_snapshot_entry_count"], "BGE entry count")
    for field_name in (
        "source_tree_artifact_id",
        "bge_snapshot_tree_artifact_id",
        "runtime_identity_artifact_id",
        "artifact_id",
    ):
        _digest(receipt[field_name], f"integrity receipt {field_name}")
    for field_name, expected_id in expected_integrity_ids.items():
        if receipt[field_name] != expected_id:
            raise ValueError(f"integrity receipt {phase} differs from expected {field_name}")
    if receipt["artifact_id"] != _artifact_id(receipt):
        raise ValueError(f"integrity receipt {phase} content address mismatch")
    return dict(receipt)


def _validate_integrity_receipt_sequence(
    receipts: list[Mapping[str, object]],
    *,
    initial_previous_integrity_receipt_artifact_id: str,
) -> None:
    expected_phases = (
        "pre_prediction_child_1",
        "post_prediction_child_1",
        "pre_prediction_child_2",
        "post_prediction_child_2",
    )
    if tuple(receipt["phase"] for receipt in receipts) != expected_phases:
        raise ValueError("prediction integrity receipt phase order drifted")
    _validate_integrity_receipt_consistency(receipts)
    previous_id = initial_previous_integrity_receipt_artifact_id
    for receipt in receipts:
        if receipt["previous_integrity_receipt_artifact_id"] != previous_id:
            raise ValueError("prediction integrity receipt chain drifted")
        previous_id = _digest(receipt["artifact_id"], "integrity artifact_id")


def _validate_integrity_receipt_consistency(
    receipts: list[Mapping[str, object]],
) -> None:
    for field_name in (
        "source_tree_artifact_id",
        "bge_snapshot_tree_artifact_id",
        "runtime_identity_artifact_id",
    ):
        if len({receipt[field_name] for receipt in receipts}) != 1:
            raise ValueError(f"prediction integrity guard observed {field_name} drift")
    artifact_ids = tuple(receipt["artifact_id"] for receipt in receipts)
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("prediction integrity receipts must be phase-distinct")


def _stream_capture_payload(capture: _BoundedStreamCapture) -> Mapping[str, object]:
    return {
        "raw_sha256": capture.raw_sha256,
        "total_bytes": capture.total_bytes,
        "retained_prefix_sha256": capture.retained_prefix_sha256,
        "retained_prefix_bytes": capture.retained_prefix_bytes,
        "prefix_truncated": capture.prefix_truncated,
    }


def _validate_stream_capture(
    capture: _BoundedStreamCapture,
    *,
    field_name: str,
) -> None:
    _digest(capture.raw_sha256, f"{field_name} raw_sha256")
    _digest(capture.retained_prefix_sha256, f"{field_name} retained_prefix_sha256")
    _nonnegative_integer(capture.total_bytes, f"{field_name} total_bytes")
    _nonnegative_integer(
        capture.retained_prefix_bytes,
        f"{field_name} retained_prefix_bytes",
    )
    if (
        capture.retained_prefix_bytes != len(capture.retained_prefix)
        or capture.retained_prefix_bytes > _MAX_CAPTURED_STREAM_PREFIX_BYTES
        or capture.retained_prefix_sha256 != hashlib.sha256(capture.retained_prefix).hexdigest()
        or capture.prefix_truncated != (capture.total_bytes > capture.retained_prefix_bytes)
    ):
        raise RuntimeError(f"prediction child {field_name} bounded capture drifted")


def _validate_preflight_manifest(
    payload: Mapping[str, object],
    *,
    qualification_protocol_id: str,
) -> Mapping[str, Mapping[str, object]]:
    _exact_keys(
        payload,
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
    if (
        payload["protocol_id"] != qualification_protocol_id
        or _positive_integer(payload["file_count"], "preflight file_count") != 7
        or _nonnegative_integer(payload["model_output_count"], "model_output_count") != 0
        or payload["external_public_anchor_created"] is not False
        or payload["qualification_execution_authorized"] is not False
    ):
        raise ValueError("preflight manifest honesty or lineage drifted")
    files = _list(payload["files"], "preflight manifest files")
    receipts: dict[str, Mapping[str, object]] = {}
    observed_paths: list[str] = []
    for index, value in enumerate(files):
        receipt = _mapping(value, f"preflight manifest file {index}")
        _exact_keys(
            receipt,
            {"path", "raw_sha256", "raw_bytes", "artifact_id"},
            f"preflight manifest file {index}",
        )
        path = _text(receipt["path"], "preflight manifest path")
        if path in receipts:
            raise ValueError("preflight manifest paths must be unique")
        _digest(receipt["raw_sha256"], "preflight raw_sha256")
        _positive_integer(receipt["raw_bytes"], "preflight raw_bytes")
        if path == "protocol.json":
            if receipt["artifact_id"] is not None:
                raise ValueError("raw protocol manifest entry must have null artifact_id")
        else:
            _digest(receipt["artifact_id"], "preflight artifact_id")
        receipts[path] = receipt
        observed_paths.append(path)
    if tuple(observed_paths) != _PREFLIGHT_RELATIVE_PATHS:
        raise ValueError("preflight manifest path set/order drifted")
    return receipts


def _validate_publication_request(
    payload: Mapping[str, object],
    *,
    qualification_protocol_id: str,
    expected_artifact_id: str,
    expected_execution_root: pathlib.Path,
) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "protocol_id",
            "protocol_filename",
            "protocol_raw_sha256",
            "protocol_raw_bytes",
            "public_corpus_artifact_id",
            "predictor_request_artifact_id",
            "training_labels_artifact_id",
            "challenge_labels_artifact_id",
            "group_split_artifact_id",
            "proposed_execution_root",
            "proposed_execution_root_exists_at_prepare",
            "external_observation_required",
            "requested_publication_visibility",
            "public_gist_created",
            "qualification_execution_authorized",
            "artifact_id",
        },
        "publication request",
    )
    if payload["artifact_id"] != expected_artifact_id:
        raise ValueError("external publication request artifact id mismatch")
    if payload["protocol_id"] != qualification_protocol_id:
        raise ValueError("publication request qualification protocol mismatch")
    for field_name in (
        "protocol_raw_sha256",
        "public_corpus_artifact_id",
        "predictor_request_artifact_id",
        "training_labels_artifact_id",
        "challenge_labels_artifact_id",
        "group_split_artifact_id",
    ):
        _digest(payload[field_name], field_name)
    proposed_text = _text(payload["proposed_execution_root"], "proposed_execution_root")
    if proposed_text != str(expected_execution_root):
        raise ValueError("publication request proposed execution root text mismatch")
    proposed = pathlib.Path(proposed_text).resolve()
    if proposed != expected_execution_root:
        raise ValueError("publication request proposed execution root mismatch")
    if (
        payload["proposed_execution_root_exists_at_prepare"] is not False
        or payload["external_observation_required"] is not True
        or payload["requested_publication_visibility"] != "public"
        or payload["public_gist_created"] is not False
        or payload["qualification_execution_authorized"] is not False
    ):
        raise ValueError("publication request preflight honesty boundary drifted")


def _validate_publication_lineage(
    publication: Mapping[str, object],
    *,
    public_corpus: Mapping[str, object],
    predictor_request: Mapping[str, object],
    training_labels: Mapping[str, object],
    manifest_files: Mapping[str, Mapping[str, object]],
) -> None:
    joins = (
        (
            publication["public_corpus_artifact_id"],
            public_corpus["artifact_id"],
            "public corpus",
        ),
        (
            publication["predictor_request_artifact_id"],
            predictor_request["artifact_id"],
            "predictor request",
        ),
        (
            publication["training_labels_artifact_id"],
            training_labels["artifact_id"],
            "training labels",
        ),
        (
            publication["challenge_labels_artifact_id"],
            manifest_files["sealed/challenge_labels.json"]["artifact_id"],
            "challenge labels",
        ),
        (
            publication["group_split_artifact_id"],
            manifest_files["sealed/group_split.json"]["artifact_id"],
            "group split",
        ),
    )
    for left, right, field_name in joins:
        if left != right:
            raise ValueError(f"publication request {field_name} lineage mismatch")


def _validate_manifest_file_receipt(
    receipt: Mapping[str, object],
    loaded: _LoadedArtifact,
    field_name: str,
) -> None:
    if (
        receipt["artifact_id"] != loaded.payload["artifact_id"]
        or receipt["raw_sha256"] != loaded.raw_sha256
        or receipt["raw_bytes"] != len(loaded.raw)
    ):
        raise ValueError(f"{field_name} manifest receipt mismatch")


def _validate_reopened_commit_receipt(
    payload: Mapping[str, object],
    *,
    expected: Mapping[str, object],
) -> None:
    expected_keys = {
        "schema_version",
        "qualification_protocol_id",
        "execution_protocol_id",
        "child_request_artifact_id",
        "predictor_request_artifact_id",
        "prediction_ledger_artifact_id",
        "prediction_ledger_raw_sha256",
        "prediction_ledger_raw_bytes",
        "prediction_run_manifest_artifact_ids",
        "prediction_run_attestation_artifact_ids",
        "fresh_process_count",
        "predictor_processes_exited",
        "predictor_job_objects_empty",
        "embedding_tables_byte_exact",
        "reader_artifacts_byte_exact",
        "prediction_ledgers_byte_exact",
        "ledger_file_fsync_completed",
        "ledger_same_descriptor_readback",
        "ledger_closed_reopen_readback",
        "windows_directory_entry_durability_attested",
        "artifact_id",
    }
    _exact_keys(payload, expected_keys, "reopened commit receipt")
    if payload != expected:
        raise ValueError("prediction ledger commit receipt changed after reopen")
    for field_name in (
        "predictor_processes_exited",
        "predictor_job_objects_empty",
        "embedding_tables_byte_exact",
        "reader_artifacts_byte_exact",
        "prediction_ledgers_byte_exact",
        "ledger_file_fsync_completed",
        "ledger_same_descriptor_readback",
        "ledger_closed_reopen_readback",
    ):
        if payload[field_name] is not True:
            raise ValueError(f"commit receipt requires {field_name}=true")
    if payload["windows_directory_entry_durability_attested"] is not False:
        raise ValueError("commit receipt must not claim directory-entry durability")


def _load_canonical_artifact(
    path: pathlib.Path,
    *,
    expected_schema_version: str,
    max_bytes: int,
) -> _LoadedArtifact:
    source = pathlib.Path(path)
    if source.is_symlink():
        raise ValueError(f"qualification artifact must not be a symlink: {source}")
    before = source.stat(follow_symlinks=False)
    if before.st_nlink != 1:
        raise ValueError(f"qualification artifact must not be hard linked: {source}")
    attributes = getattr(before, "st_file_attributes", 0)
    if os.name == "nt" and attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"qualification artifact must not be a reparse point: {source}")
    with source.open("rb") as handle:
        during = os.fstat(handle.fileno())
        raw = handle.read(max_bytes + 1)
    after = source.stat(follow_symlinks=False)
    if len(raw) > max_bytes:
        raise ValueError(f"qualification artifact exceeds byte bound: {source}")
    if _file_identity(before) != _file_identity(during) or _file_identity(during) != _file_identity(after):
        raise ValueError(f"qualification artifact identity changed while reading: {source}")
    parsed = strict_json_loads(raw, max_bytes=max_bytes)
    payload = _mapping(parsed, str(source))
    if payload.get("schema_version") != expected_schema_version:
        raise ValueError(f"qualification artifact schema mismatch: {source}")
    if raw != _canonical_artifact_bytes(payload):
        raise ValueError(f"qualification artifact is not canonical JSON: {source}")
    artifact_id = _digest(payload.get("artifact_id"), f"{source} artifact_id")
    if artifact_id != _artifact_id(payload):
        raise ValueError(f"qualification artifact content address mismatch: {source}")
    return _LoadedArtifact(
        payload=payload,
        raw=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _write_artifact_create_only(
    path: pathlib.Path,
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    return _write_bytes_create_only(path, _canonical_artifact_bytes(payload))


def _write_bytes_create_only(
    path: pathlib.Path,
    raw: bytes,
) -> Mapping[str, object]:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x+b") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
        handle.seek(0)
        same_descriptor = handle.read() == raw
        if not same_descriptor:
            raise RuntimeError(f"same-descriptor artifact readback failed: {target}")
    with target.open("rb") as handle:
        closed_reopen = handle.read() == raw
    if not closed_reopen:
        raise RuntimeError(f"closed-reopen artifact readback failed: {target}")
    return {
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes": len(raw),
        "fsync_completed": True,
        "same_descriptor_readback": same_descriptor,
        "closed_reopen_readback": closed_reopen,
        "windows_directory_entry_durability_attested": False,
    }


def _launch_windows_fresh_prediction_subprocess(
    spec: _PredictionChildLaunchSpec,
    *,
    timeout_seconds: int,
) -> _PredictionChildLaunchResult:
    """Start suspended, assign a one-process Job, then resume exactly once."""

    if os.name != "nt":
        raise RuntimeError("formal relationship-reader prediction requires Windows")
    if spec.bge_snapshot_path is None:
        raise ValueError("formal prediction child requires an explicit BGE snapshot")
    if os.path.lexists(spec.pycache_prefix):
        raise FileExistsError(f"formal prediction pycache prefix must be absent before launch: {spec.pycache_prefix}")
    import _winapi
    import ctypes
    from ctypes import wintypes
    import msvcrt

    class _JobObjectBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JobObjectExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobObjectBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _JobObjectBasicAccountingInformation(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_longlong),
            ("TotalKernelTime", ctypes.c_longlong),
            ("ThisPeriodTotalUserTime", ctypes.c_longlong),
            ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    environment, environment_projection, environment_contract_id = _build_formal_child_environment(
        os.environ,
        import_binding=spec.import_binding,
        pycache_prefix=spec.pycache_prefix,
    )
    argv = list(_prediction_child_process_argv(spec))
    python_executable = spec.import_binding.python_executable

    stdout_read, stdout_write = os.pipe()
    stderr_read, stderr_write = os.pipe()
    stdin_fd = os.open(os.devnull, os.O_RDONLY)
    process_handle = None
    thread_handle = None
    job = None
    stdout_drain: _BoundedPipeDrain | None = None
    stderr_drain: _BoundedPipeDrain | None = None
    write_fds_open = True
    initial_resume_count = -1
    process_wait_completed = False
    process_assigned_to_job = False
    try:
        for fd in (stdin_fd, stdout_write, stderr_write):
            os.set_inheritable(fd, True)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESTDHANDLES
        startupinfo.hStdInput = msvcrt.get_osfhandle(stdin_fd)
        startupinfo.hStdOutput = msvcrt.get_osfhandle(stdout_write)
        startupinfo.hStdError = msvcrt.get_osfhandle(stderr_write)
        startupinfo.lpAttributeList = {
            "handle_list": [
                startupinfo.hStdInput,
                startupinfo.hStdOutput,
                startupinfo.hStdError,
            ]
        }

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        limits = _JobObjectExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = _FORMAL_JOB_LIMIT_FLAGS
        limits.BasicLimitInformation.ActiveProcessLimit = 1
        if not kernel32.SetInformationJobObject(
            job,
            9,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")

        process_handle, thread_handle, process_id, _thread_id = _winapi.CreateProcess(
            str(python_executable),
            subprocess.list2cmdline(argv),
            None,
            None,
            True,
            _FORMAL_CREATION_FLAGS,
            environment,
            str(spec.capsule_root),
            startupinfo,
        )
        if not kernel32.AssignProcessToJobObject(
            job,
            wintypes.HANDLE(int(process_handle)),
        ):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        process_assigned_to_job = True

        os.close(stdin_fd)
        stdin_fd = -1
        os.close(stdout_write)
        os.close(stderr_write)
        write_fds_open = False
        stdout_drain = _BoundedPipeDrain(stdout_read)
        stderr_drain = _BoundedPipeDrain(stderr_read)
        stdout_read = -1
        stderr_read = -1
        stdout_drain.start()
        stderr_drain.start()

        initial_resume_count = int(kernel32.ResumeThread(wintypes.HANDLE(int(thread_handle))))
        if initial_resume_count == 0xFFFFFFFF:
            raise OSError(ctypes.get_last_error(), "ResumeThread failed")
        if initial_resume_count != 1:
            raise RuntimeError("fresh prediction child initial suspension count must be exactly one")
        _winapi.CloseHandle(thread_handle)
        thread_handle = None

        wait_result = _winapi.WaitForSingleObject(
            process_handle,
            timeout_seconds * 1_000,
        )
        if wait_result == 258:  # WAIT_TIMEOUT
            if not kernel32.TerminateJobObject(job, 1):
                raise OSError(ctypes.get_last_error(), "TerminateJobObject failed")
            reap_result = _winapi.WaitForSingleObject(process_handle, 30_000)
            process_wait_completed = reap_result == 0
            raise TimeoutError(f"prediction child {spec.run_ordinal} exceeded timeout")
        if wait_result != 0:  # WAIT_OBJECT_0
            raise OSError(f"prediction child wait failed with code {wait_result}")
        process_wait_completed = True
        exit_code = int(_winapi.GetExitCodeProcess(process_handle))
        accounting = _JobObjectBasicAccountingInformation()
        returned = wintypes.DWORD()
        if not kernel32.QueryInformationJobObject(
            job,
            1,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            ctypes.byref(returned),
        ):
            raise OSError(ctypes.get_last_error(), "QueryInformationJobObject failed")
        stdout_capture = stdout_drain.finish()
        stderr_capture = stderr_drain.finish()
        return _PredictionChildLaunchResult(
            process_id=int(process_id),
            process_argv=tuple(argv),
            exit_code=exit_code,
            process_exited=True,
            job_object_empty=accounting.ActiveProcesses == 0,
            environment_contract_id=environment_contract_id,
            environment_projection=environment_projection,
            creation_flags=_FORMAL_CREATION_FLAGS,
            shell=False,
            close_fds=True,
            process_created_suspended=True,
            job_assigned_before_resume=True,
            initial_thread_resume_previous_count=initial_resume_count,
            job_limit_flags=_FORMAL_JOB_LIMIT_FLAGS,
            job_active_process_limit=1,
            stdout_capture=stdout_capture,
            stderr_capture=stderr_capture,
            source_capsule_used=False,
            repository_import_path_used=True,
            bge_snapshot_tree_verified=False,
        )
    finally:
        active_error = sys.exception()
        cleanup_errors: list[BaseException] = []
        if process_handle is not None and not process_wait_completed:
            if process_assigned_to_job:
                if job and not kernel32.TerminateJobObject(job, 1):
                    cleanup_errors.append(
                        OSError(
                            ctypes.get_last_error(),
                            "TerminateJobObject failed",
                        )
                    )
                    if not kernel32.TerminateProcess(
                        wintypes.HANDLE(int(process_handle)),
                        1,
                    ):
                        cleanup_errors.append(
                            OSError(
                                ctypes.get_last_error(),
                                "fallback TerminateProcess failed",
                            )
                        )
            elif not kernel32.TerminateProcess(
                wintypes.HANDLE(int(process_handle)),
                1,
            ):
                cleanup_errors.append(OSError(ctypes.get_last_error(), "TerminateProcess failed"))
            try:
                reap_result = _winapi.WaitForSingleObject(process_handle, 30_000)
                if reap_result != 0:
                    cleanup_errors.append(RuntimeError("prediction child could not be reaped after termination"))
                else:
                    process_wait_completed = True
            except BaseException as exc:  # pragma: no cover - Win32 cleanup boundary
                cleanup_errors.append(exc)
        if job:
            if not kernel32.CloseHandle(job):
                cleanup_errors.append(OSError(ctypes.get_last_error(), "CloseHandle(job) failed"))
        if thread_handle is not None:
            try:
                _winapi.CloseHandle(thread_handle)
            except BaseException as exc:  # pragma: no cover - Win32 cleanup boundary
                cleanup_errors.append(exc)
        if process_handle is not None:
            try:
                _winapi.CloseHandle(process_handle)
            except BaseException as exc:  # pragma: no cover - Win32 cleanup boundary
                cleanup_errors.append(exc)
        if stdin_fd >= 0:
            try:
                os.close(stdin_fd)
            except OSError as exc:  # pragma: no cover - Win32 cleanup boundary
                cleanup_errors.append(exc)
        if write_fds_open:
            for fd in (stdout_write, stderr_write):
                try:
                    os.close(fd)
                except OSError as exc:  # pragma: no cover - Win32 cleanup boundary
                    cleanup_errors.append(exc)
        for fd in (stdout_read, stderr_read):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError as exc:  # pragma: no cover - Win32 cleanup boundary
                    cleanup_errors.append(exc)
        if stdout_drain is not None:
            try:
                stdout_drain.finish()
            except BaseException as exc:  # pragma: no cover - Win32 cleanup boundary
                cleanup_errors.append(exc)
        if stderr_drain is not None:
            try:
                stderr_drain.finish()
            except BaseException as exc:  # pragma: no cover - Win32 cleanup boundary
                cleanup_errors.append(exc)
        if cleanup_errors:
            summary = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            if active_error is not None:
                active_error.add_note(f"prediction launcher cleanup errors: {summary}")
            else:
                raise RuntimeError(f"prediction launcher cleanup failed: {summary}") from cleanup_errors[0]


def _prediction_child_process_argv(
    spec: _PredictionChildLaunchSpec,
) -> tuple[str, ...]:
    import_roots_json = json.dumps(
        [str(path) for path in spec.import_binding.import_roots],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        str(spec.import_binding.python_executable),
        "-P",
        "-S",
        "-B",
        "-u",
        "-X",
        "utf8",
        "-X",
        f"pycache_prefix={spec.pycache_prefix}",
        "-c",
        _PREDICTION_CHILD_BOOTSTRAP,
        import_roots_json,
        str(spec.child_request_path.resolve()),
        spec.expected_child_request_artifact_id,
        str(spec.training_corpus_path.resolve()),
        str(spec.predictor_request_path.resolve()),
        str(spec.output_root.resolve()),
        str(spec.run_ordinal),
        spec.run_nonce,
        str(spec.bge_snapshot_path),
    )


def _prediction_child_sys_argv(spec: _PredictionChildLaunchSpec) -> list[str]:
    process_argv = _prediction_child_process_argv(spec)
    return ["-c", *process_argv[11:]]


class _BoundedPipeDrain:
    def __init__(self, read_fd: int) -> None:
        self._read_fd = read_fd
        self._hasher = hashlib.sha256()
        self._retained = bytearray()
        self._total_bytes = 0
        self._error: BaseException | None = None
        self._finished: _BoundedStreamCapture | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            with os.fdopen(self._read_fd, "rb", closefd=True) as handle:
                while True:
                    chunk = handle.read(65_536)
                    if not chunk:
                        break
                    self._hasher.update(chunk)
                    self._total_bytes += len(chunk)
                    remaining = _MAX_CAPTURED_STREAM_PREFIX_BYTES - len(self._retained)
                    if remaining > 0:
                        self._retained.extend(chunk[:remaining])
        except BaseException as exc:  # pragma: no cover - OS pipe failure boundary
            self._error = exc

    def finish(self) -> _BoundedStreamCapture:
        if self._finished is not None:
            return self._finished
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise RuntimeError("prediction child output drain did not terminate")
        if self._error is not None:
            raise RuntimeError("prediction child output drain failed") from self._error
        retained = bytes(self._retained)
        self._finished = _BoundedStreamCapture(
            raw_sha256=self._hasher.hexdigest(),
            total_bytes=self._total_bytes,
            retained_prefix=retained,
            retained_prefix_sha256=hashlib.sha256(retained).hexdigest(),
            retained_prefix_bytes=len(retained),
            prefix_truncated=self._total_bytes > len(retained),
        )
        return self._finished


def _build_formal_child_environment(
    source: Mapping[str, str],
    *,
    import_binding: QualificationChildImportBinding,
    pycache_prefix: pathlib.Path,
) -> tuple[
    dict[str, str],
    tuple[_EnvironmentValueReceipt, ...],
    str,
]:
    by_casefold: dict[str, tuple[str, str]] = {}
    for key, value in source.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("formal child environment must contain text keys/values")
        folded = key.casefold()
        if folded in by_casefold and by_casefold[folded][0] != key:
            raise ValueError("formal child environment contains case-ambiguous keys")
        by_casefold[folded] = (key, value)
    environment: dict[str, str] = {}
    for canonical_key in _FORMAL_ENVIRONMENT_ALLOWLIST:
        found = by_casefold.get(canonical_key.casefold())
        if found is not None:
            environment[canonical_key] = found[1]
    environment.update(_FORMAL_ENVIRONMENT_FIXED)
    if not isinstance(import_binding, QualificationChildImportBinding):
        raise TypeError("formal child import_binding is invalid")
    canonical_import_roots = tuple(str(path) for path in import_binding.import_roots)
    if any(not pathlib.Path(path).is_absolute() for path in canonical_import_roots):
        raise ValueError("formal child import roots must be absolute")
    environment["PYTHONPATH"] = os.pathsep.join(canonical_import_roots)
    prefix = pathlib.Path(pycache_prefix)
    if not prefix.is_absolute():
        raise ValueError("formal child pycache prefix must be absolute")
    environment["PYTHONPYCACHEPREFIX"] = str(prefix)
    system_root = environment.get("SystemRoot")
    if system_root is None or not system_root:
        raise ValueError("formal child environment requires SystemRoot")
    environment["PATH"] = os.pathsep.join(
        str(path)
        for path in controlled_child_path(
            import_binding,
            system_root=pathlib.Path(system_root),
        )
    )
    for required in ("PATH", "SystemRoot"):
        if required not in environment or not environment[required]:
            raise ValueError(f"formal child environment requires {required}")
    for key, value in environment.items():
        if "\x00" in key or "=" in key or "\x00" in value:
            raise ValueError("formal child environment contains invalid Windows text")
    canonical_environment = dict(sorted(environment.items()))
    projection = tuple(
        _EnvironmentValueReceipt(
            key=key,
            value_sha256=hashlib.sha256(value.encode("utf-8")).hexdigest(),
            value_utf8_bytes=len(value.encode("utf-8")),
        )
        for key, value in canonical_environment.items()
    )
    return (
        canonical_environment,
        projection,
        _environment_contract_id(projection),
    )


def _environment_contract_id(
    projection: tuple[_EnvironmentValueReceipt, ...],
) -> str:
    return _sha256_json(
        {
            "environment_projection": [
                {
                    "key": item.key,
                    "value_sha256": item.value_sha256,
                    "value_utf8_bytes": item.value_utf8_bytes,
                }
                for item in projection
            ],
            "environment_built_from_empty_allowlist": True,
        }
    )


def _parse_opaque_text_row(value: object, field_name: str) -> Mapping[str, str]:
    row = _mapping(value, field_name)
    _exact_keys(row, {"item_id", "text", "text_sha256"}, field_name)
    text = _text(row["text"], f"{field_name}.text")
    text_sha256 = _digest(row["text_sha256"], f"{field_name}.text_sha256")
    if text_sha256 != hashlib.sha256(text.encode("utf-8")).hexdigest():
        raise ValueError(f"{field_name} text_sha256 mismatch")
    return {
        "item_id": _digest(row["item_id"], f"{field_name}.item_id"),
        "text": text,
        "text_sha256": text_sha256,
    }


def _validate_canonical_text_rows(
    rows: tuple[Mapping[str, str], ...],
    expected_count: int,
    field_name: str,
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"{field_name} count drifted")
    item_ids = tuple(row["item_id"] for row in rows)
    text_sha256s = tuple(row["text_sha256"] for row in rows)
    if item_ids != tuple(sorted(item_ids)) or len(set(item_ids)) != len(item_ids):
        raise ValueError(f"{field_name} item ids must be unique and canonical")
    if len(set(text_sha256s)) != len(text_sha256s):
        raise ValueError(f"{field_name} exact texts must be unique")


def _with_artifact_id(core: Mapping[str, object]) -> dict[str, object]:
    if "artifact_id" in core:
        raise ValueError("artifact core must not predefine artifact_id")
    return {**core, "artifact_id": _sha256_json(core)}


def _artifact_id(payload: Mapping[str, object]) -> str:
    core = {key: value for key, value in payload.items() if key != "artifact_id"}
    return _sha256_json(core)


def _sha256_json(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _canonical_artifact_bytes(payload: Mapping[str, object]) -> bytes:
    return canonical_json_bytes(payload) + b"\n"


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_nlink,
    )


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be a JSON object with string keys")
    return value


def _list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    field_name: str,
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{field_name} keys mismatch; missing={missing}, extra={extra}")


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be canonical non-empty text")
    return value


def _digest(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


def _canonical_hex_float(value: object, field_name: str) -> float:
    text = _text(value, field_name)
    try:
        parsed = float.fromhex(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a hexadecimal float") from exc
    normalized = 0.0 if parsed == 0.0 else parsed
    if not math.isfinite(parsed) or normalized.hex() != text:
        raise ValueError(f"{field_name} must be a finite canonical hexadecimal float")
    return parsed


def _canonical_float_hex(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("canonical float value must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("canonical float value must be finite")
    if numeric == 0.0:
        numeric = 0.0
    return numeric.hex()


__all__ = [
    "RELATIONSHIP_READER_PREDICTION_LEDGER_COMMIT_SCHEMA_VERSION",
    "RELATIONSHIP_READER_SCORING_REQUEST_SCHEMA_VERSION",
    "execute_relationship_condition_reader_qualification_prediction_stage",
]
