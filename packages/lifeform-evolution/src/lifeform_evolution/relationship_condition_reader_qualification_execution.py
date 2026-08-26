"""Fresh-process execution boundary for the model-free qualification scorer.

This module deliberately does not import the scorer in the parent process.  A
Windows child imports it lazily only after it has been created suspended,
assigned to a fresh one-process Job Object, and resumed exactly once.  The
published stage manifest is intentionally non-authorizing: source/runtime
integrity and an external execution anchor remain responsibilities of the
outer qualification CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
import pathlib
import stat
import subprocess
import sys
import threading
from typing import Callable, Mapping, Protocol

from volvence_zero.canonical_json import canonical_json_bytes, strict_json_loads

from .relationship_condition_reader_qualification_execution_protocol import (
    RELATIONSHIP_READER_EXECUTION_INTEGRITY_RECEIPT_SCHEMA_VERSION,
    relationship_condition_reader_qualification_integrity_guard,
    validate_relationship_condition_reader_qualification_public_anchor_receipt,
)
from .relationship_condition_reader_qualification_executor import (
    execute_relationship_condition_reader_qualification_prediction_stage,
)
from .relationship_condition_reader_qualification_runtime_binding import (
    QualificationChildImportBinding,
    build_qualification_child_import_binding,
    controlled_child_path,
    expected_child_sys_path,
    validate_child_file_backed_module_origin_attestation,
)


RELATIONSHIP_READER_QUALIFICATION_SCORING_STAGE_MANIFEST_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-scoring-stage-manifest.v1"
)
RELATIONSHIP_READER_QUALIFICATION_AUTHORIZED_EXECUTION_MANIFEST_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-authorized-execution-manifest.v1"
)

_SCORING_REQUEST_SCHEMA_VERSION = "relationship-condition-reader-qualification-scoring-request.v1"
_REPORT_SCHEMA_VERSION = "relationship-condition-reader-qualification-report.v1"
_SCORER_ATTESTATION_SCHEMA_VERSION = "relationship-condition-reader-qualification-scorer-attestation.v3"
_SCORER_MANIFEST_SCHEMA_VERSION = "relationship-condition-reader-qualification-scorer-manifest.v1"
_PREDICTION_EXECUTOR_RESULT_SCHEMA_VERSION = "relationship-condition-reader-qualification-executor-result.v1"
_PREDICTION_LAUNCHER_ATTESTATION_SCHEMA_VERSION = "relationship-condition-reader-qualification-launcher-attestation.v3"
_SCORER_OUTPUT_PATHS = (
    "report.json",
    "scorer_attestation.json",
    "manifest.json",
)
_SCORER_MANIFEST_BOUND_PATHS = (
    "report.json",
    "scorer_attestation.json",
)
_SCORER_EVENT_SEQUENCE = (
    "scoring_request_validated",
    "commit_receipt_validated",
    "prediction_ledger_semantically_revalidated",
    "challenge_labels_opened",
    "group_split_opened",
    "report_fsynced_and_reopened",
)
_INTEGRITY_PHASE_ORDINALS = {
    "post_anchor_pre_execution": 0,
    "pre_prediction_child_1": 1,
    "post_prediction_child_1": 2,
    "pre_prediction_child_2": 3,
    "post_prediction_child_2": 4,
    "pre_scorer": 5,
    "post_scorer": 6,
    "final_validation": 7,
}
_FORBIDDEN_MODEL_MODULE_PREFIXES = ("torch", "sentence_transformers")
_REQUIRED_SCORER_MODULES = frozenset(
    {
        "lifeform_evolution.relationship_condition_reader_qualification_runtime_binding",
        "lifeform_evolution.relationship_condition_reader_qualification_scorer",
        "volvence_zero.canonical_json",
        "volvence_zero.social_cognition",
    }
)
_MAX_SMALL_ARTIFACT_BYTES = 2_000_000
_MAX_CAPTURED_STREAM_PREFIX_BYTES = 65_536

# The child environment starts empty.  Only these inherited values plus the
# fixed values below are copied into it.  In particular, CUDA/model caches and
# credentials are not inherited by the model-free scorer.
_FORMAL_SCORER_ENVIRONMENT_ALLOWLIST = (
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "WINDIR",
)
_FORMAL_SCORER_ENVIRONMENT_FIXED = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "PYTHONUTF8": "1",
}

_CREATE_SUSPENDED = 0x00000004
_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_CREATE_NO_WINDOW = 0x08000000
_FORMAL_SCORER_CREATION_FLAGS = _CREATE_SUSPENDED | _EXTENDED_STARTUPINFO_PRESENT | _CREATE_NO_WINDOW
_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_FORMAL_SCORER_JOB_LIMIT_FLAGS = _JOB_OBJECT_LIMIT_ACTIVE_PROCESS | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE


@dataclass(frozen=True)
class _LoadedArtifact:
    path: pathlib.Path
    payload: Mapping[str, object]
    raw: bytes
    raw_sha256: str
    file_identity: tuple[int, int, int, int, int]


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
class _ScorerLaunchSpec:
    scoring_request_path: pathlib.Path
    expected_scoring_request_artifact_id: str
    output_root: pathlib.Path
    capsule_root: pathlib.Path
    run_nonce: str
    python_executable: pathlib.Path
    import_binding: QualificationChildImportBinding
    pycache_prefix: pathlib.Path
    environment_items: tuple[tuple[str, str], ...]
    environment_projection: tuple[_EnvironmentValueReceipt, ...]
    environment_contract_id: str


@dataclass(frozen=True)
class _ScorerLaunchResult:
    process_id: int
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


class _ScorerLauncher(Protocol):
    def __call__(self, spec: _ScorerLaunchSpec) -> _ScorerLaunchResult: ...


class _IntegrityGuard(Protocol):
    def __call__(
        self,
        *,
        phase: str,
        previous_integrity_receipt_artifact_id: str | None,
    ) -> Mapping[str, object]: ...


def execute_relationship_condition_reader_qualification_scoring_stage(
    *,
    scoring_request_path: pathlib.Path,
    expected_scoring_request_artifact_id: str,
    stage_root: pathlib.Path,
    integrity_guard: _IntegrityGuard,
    previous_integrity_receipt_artifact_id: str,
    expected_source_tree_artifact_id: str,
    expected_bge_snapshot_tree_artifact_id: str,
    expected_runtime_identity_artifact_id: str,
    repository_root: pathlib.Path,
    repository_source_roots: tuple[pathlib.Path, ...],
    frozen_source_entries: Mapping[str, Mapping[str, object]],
    frozen_site_packages_root: pathlib.Path,
    python_executable: pathlib.Path | None = None,
    scorer_timeout_seconds: int = 600,
) -> Mapping[str, object]:
    """Run and verify one fresh model-free scorer process on Windows.

    The returned manifest proves only this local scoring-stage mechanism.  It
    explicitly does not authorize formal evidence or any four-capability
    product claim; an outer CLI must independently validate the frozen source,
    runtime identity, and public execution anchor before composing a result.
    """

    if os.name != "nt":
        raise RuntimeError("formal relationship-reader scoring requires Windows")
    executable = pathlib.Path(python_executable or sys.executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"scorer Python executable is absent: {executable}")
    if (
        isinstance(scorer_timeout_seconds, bool)
        or not isinstance(scorer_timeout_seconds, int)
        or scorer_timeout_seconds < 1
    ):
        raise ValueError("scorer_timeout_seconds must be a positive integer")

    def launcher(spec: _ScorerLaunchSpec) -> _ScorerLaunchResult:
        return _launch_windows_fresh_scorer_subprocess(
            spec,
            timeout_seconds=scorer_timeout_seconds,
        )

    return _execute_scoring_stage_with_launcher(
        scoring_request_path=scoring_request_path,
        expected_scoring_request_artifact_id=(expected_scoring_request_artifact_id),
        stage_root=stage_root,
        python_executable=executable,
        environment_source=os.environ,
        launcher=launcher,
        integrity_guard=integrity_guard,
        previous_integrity_receipt_artifact_id=(previous_integrity_receipt_artifact_id),
        expected_source_tree_artifact_id=expected_source_tree_artifact_id,
        expected_bge_snapshot_tree_artifact_id=(expected_bge_snapshot_tree_artifact_id),
        expected_runtime_identity_artifact_id=(expected_runtime_identity_artifact_id),
        repository_root=repository_root,
        repository_source_roots=repository_source_roots,
        frozen_source_entries=frozen_source_entries,
        frozen_site_packages_root=frozen_site_packages_root,
    )


def execute_authorized_relationship_condition_reader_qualification_execution(
    *,
    execution_protocol_payload: Mapping[str, object],
    execution_protocol_raw: bytes,
    expected_execution_protocol_id: str,
    public_anchor_receipt_payload: Mapping[str, object],
    expected_public_anchor_receipt_artifact_id: str,
    repository_root: pathlib.Path,
    preflight_root: pathlib.Path,
    bge_snapshot_root: pathlib.Path,
    execution_root: pathlib.Path,
    run_nonce: str,
    python_executable: pathlib.Path | None = None,
    prediction_timeout_seconds: int = 7_200,
    scorer_timeout_seconds: int = 600,
) -> Mapping[str, object]:
    """Run the anchored qualification sequence and publish its final manifest.

    This is the thinnest authorized outer runner.  Before the prediction
    commit it validates only the embedded preflight binding shape and extracts
    the already-frozen manifest/publication identities; it never invokes the
    full preflight-tree reobserver that would open sealed evaluator files.
    """

    executable = pathlib.Path(python_executable or sys.executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"qualification Python executable is absent: {executable}")
    for value, field_name in (
        (prediction_timeout_seconds, "prediction_timeout_seconds"),
        (scorer_timeout_seconds, "scorer_timeout_seconds"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field_name} must be a positive integer")

    def integrity_guard_factory() -> _IntegrityGuard:
        return relationship_condition_reader_qualification_integrity_guard(
            execution_protocol=execution_protocol_payload,
            expected_execution_protocol_id=expected_execution_protocol_id,
            repository_root=repository_root,
            bge_snapshot_root=bge_snapshot_root,
        )

    def prediction_stage(**kwargs: object) -> Mapping[str, object]:
        return execute_relationship_condition_reader_qualification_prediction_stage(**kwargs)

    def scoring_stage(**kwargs: object) -> Mapping[str, object]:
        return execute_relationship_condition_reader_qualification_scoring_stage(**kwargs)

    return _execute_authorized_qualification_with_stages(
        execution_protocol_payload=execution_protocol_payload,
        execution_protocol_raw=execution_protocol_raw,
        expected_execution_protocol_id=expected_execution_protocol_id,
        public_anchor_receipt_payload=public_anchor_receipt_payload,
        expected_public_anchor_receipt_artifact_id=(expected_public_anchor_receipt_artifact_id),
        repository_root=repository_root,
        preflight_root=preflight_root,
        bge_snapshot_root=bge_snapshot_root,
        execution_root=execution_root,
        run_nonce=run_nonce,
        python_executable=executable,
        prediction_timeout_seconds=prediction_timeout_seconds,
        scorer_timeout_seconds=scorer_timeout_seconds,
        anchor_validator=(validate_relationship_condition_reader_qualification_public_anchor_receipt),
        integrity_guard_factory=integrity_guard_factory,
        prediction_stage=prediction_stage,
        scoring_stage=scoring_stage,
    )


def _execute_authorized_qualification_with_stages(
    *,
    execution_protocol_payload: Mapping[str, object],
    execution_protocol_raw: bytes,
    expected_execution_protocol_id: str,
    public_anchor_receipt_payload: Mapping[str, object],
    expected_public_anchor_receipt_artifact_id: str,
    repository_root: pathlib.Path,
    preflight_root: pathlib.Path,
    bge_snapshot_root: pathlib.Path,
    execution_root: pathlib.Path,
    run_nonce: str,
    python_executable: pathlib.Path,
    prediction_timeout_seconds: int,
    scorer_timeout_seconds: int,
    anchor_validator: Callable[..., str],
    integrity_guard_factory: Callable[[], _IntegrityGuard],
    prediction_stage: Callable[..., Mapping[str, object]],
    scoring_stage: Callable[..., Mapping[str, object]],
) -> Mapping[str, object]:
    """Injectable outer core used by process-free sequence tests."""

    root = pathlib.Path(execution_root).resolve()
    if root.exists():
        raise FileExistsError(f"qualification execution root exists: {root}")
    protocol_id = _digest(
        expected_execution_protocol_id,
        "expected_execution_protocol_id",
    )
    anchor_id = _digest(
        expected_public_anchor_receipt_artifact_id,
        "expected_public_anchor_receipt_artifact_id",
    )
    nonce = _digest(run_nonce, "run_nonce")
    if not isinstance(execution_protocol_raw, bytes):
        raise TypeError("execution_protocol_raw must be bytes")
    parsed_protocol = _mapping(
        strict_json_loads(execution_protocol_raw, max_bytes=8_000_000),
        "execution protocol raw",
    )
    protocol = _mapping(execution_protocol_payload, "execution protocol payload")
    if parsed_protocol != protocol:
        raise ValueError("execution protocol payload differs from supplied raw bytes")
    if not all(
        callable(value)
        for value in (
            anchor_validator,
            integrity_guard_factory,
            prediction_stage,
            scoring_stage,
        )
    ):
        raise TypeError("authorized execution dependencies must be callable")

    validated_anchor_id = anchor_validator(
        public_anchor_receipt_payload,
        expected_receipt_artifact_id=anchor_id,
        execution_protocol_payload=protocol,
        execution_protocol_raw=execution_protocol_raw,
        expected_execution_protocol_id=protocol_id,
        expected_execution_root=root,
    )
    if validated_anchor_id != anchor_id:
        raise ValueError("public anchor validator returned an unexpected receipt id")
    if root.exists():
        raise FileExistsError("execution root appeared during public-anchor validation")

    frozen = _extract_frozen_execution_identities(protocol)
    if frozen["execution_root"] != str(root):
        raise ValueError("execution protocol proposed root differs from requested root")
    (
        repository_source_roots,
        frozen_source_entries,
        frozen_site_packages_root,
    ) = _extract_child_import_inputs(
        protocol,
        repository_root=pathlib.Path(repository_root).resolve(),
    )
    expected_integrity_ids = _mapping(
        frozen["expected_integrity_artifact_ids"],
        "expected integrity artifact ids",
    )
    guard = integrity_guard_factory()
    if not callable(guard):
        raise TypeError("integrity guard factory returned a non-callable value")
    initial_integrity_receipt = _run_integrity_guard(
        guard,
        phase="post_anchor_pre_execution",
        execution_protocol_id=protocol_id,
        expected_integrity_ids={key: _digest(value, key) for key, value in expected_integrity_ids.items()},
        previous_integrity_receipt_artifact_id=None,
    )
    if root.exists():
        raise FileExistsError("execution root appeared before prediction stage")

    prediction_result = _mapping(
        prediction_stage(
            preflight_root=pathlib.Path(preflight_root).resolve(),
            execution_root=root,
            expected_qualification_protocol_id=frozen["qualification_protocol_id"],
            expected_preflight_manifest_artifact_id=frozen["preflight_manifest_artifact_id"],
            expected_publication_request_artifact_id=frozen["publication_request_artifact_id"],
            execution_protocol_id=protocol_id,
            run_nonce=nonce,
            integrity_guard=guard,
            previous_integrity_receipt_artifact_id=initial_integrity_receipt["artifact_id"],
            expected_source_tree_artifact_id=expected_integrity_ids["source_tree_artifact_id"],
            expected_bge_snapshot_tree_artifact_id=expected_integrity_ids["bge_snapshot_tree_artifact_id"],
            expected_runtime_identity_artifact_id=expected_integrity_ids["runtime_identity_artifact_id"],
            bge_snapshot_path=pathlib.Path(bge_snapshot_root).resolve(),
            python_executable=pathlib.Path(python_executable).resolve(),
            repository_root=pathlib.Path(repository_root).resolve(),
            repository_source_roots=repository_source_roots,
            frozen_source_entries=frozen_source_entries,
            frozen_site_packages_root=frozen_site_packages_root,
            child_timeout_seconds=prediction_timeout_seconds,
        ),
        "prediction stage result",
    )
    launcher_attestation = _validate_prediction_stage_for_outer_runner(
        result=prediction_result,
        execution_root=root,
        protocol_id=protocol_id,
        frozen=frozen,
        initial_integrity_receipt=initial_integrity_receipt,
        expected_integrity_ids=expected_integrity_ids,
        run_nonce=nonce,
    )

    scoring_stage_root = root / "scoring_stage"
    scoring_result = _mapping(
        scoring_stage(
            scoring_request_path=pathlib.Path(
                _text(
                    prediction_result["scoring_request_path"],
                    "prediction scoring_request_path",
                )
            ),
            expected_scoring_request_artifact_id=prediction_result["scoring_request_artifact_id"],
            stage_root=scoring_stage_root,
            integrity_guard=guard,
            previous_integrity_receipt_artifact_id=prediction_result["last_integrity_receipt_artifact_id"],
            expected_source_tree_artifact_id=expected_integrity_ids["source_tree_artifact_id"],
            expected_bge_snapshot_tree_artifact_id=expected_integrity_ids["bge_snapshot_tree_artifact_id"],
            expected_runtime_identity_artifact_id=expected_integrity_ids["runtime_identity_artifact_id"],
            python_executable=pathlib.Path(python_executable).resolve(),
            repository_root=pathlib.Path(repository_root).resolve(),
            repository_source_roots=repository_source_roots,
            frozen_source_entries=frozen_source_entries,
            frozen_site_packages_root=frozen_site_packages_root,
            scorer_timeout_seconds=scorer_timeout_seconds,
        ),
        "scoring stage result",
    )
    scoring_manifest, scorer_report = _validate_scoring_stage_for_outer_runner(
        result=scoring_result,
        stage_root=scoring_stage_root,
        protocol_id=protocol_id,
        expected_scoring_request_artifact_id=_digest(
            prediction_result["scoring_request_artifact_id"],
            "scoring request artifact_id",
        ),
        previous_integrity_receipt_artifact_id=_digest(
            prediction_result["last_integrity_receipt_artifact_id"],
            "prediction last integrity receipt artifact_id",
        ),
    )
    final_integrity_receipt = _run_integrity_guard(
        guard,
        phase="final_validation",
        execution_protocol_id=protocol_id,
        expected_integrity_ids={key: _digest(value, key) for key, value in expected_integrity_ids.items()},
        previous_integrity_receipt_artifact_id=_digest(
            scoring_manifest["last_integrity_receipt_artifact_id"],
            "scoring last integrity receipt artifact_id",
        ),
    )
    all_integrity_receipts = _collect_and_validate_full_integrity_chain(
        initial=initial_integrity_receipt,
        launcher_attestation=launcher_attestation,
        scoring_manifest=scoring_manifest,
        final=final_integrity_receipt,
        execution_protocol_id=protocol_id,
        expected_integrity_ids=expected_integrity_ids,
    )

    final_manifest = _with_artifact_id(
        {
            "schema_version": (RELATIONSHIP_READER_QUALIFICATION_AUTHORIZED_EXECUTION_MANIFEST_SCHEMA_VERSION),
            "execution_protocol_id": protocol_id,
            "execution_protocol_raw_sha256": hashlib.sha256(execution_protocol_raw).hexdigest(),
            "execution_protocol_raw_bytes": len(execution_protocol_raw),
            "public_anchor_receipt_artifact_id": anchor_id,
            "public_anchor_receipt": dict(public_anchor_receipt_payload),
            "qualification_protocol_id": frozen["qualification_protocol_id"],
            "preflight_binding_artifact_id": frozen["preflight_binding_artifact_id"],
            "preflight_manifest_artifact_id": frozen["preflight_manifest_artifact_id"],
            "publication_request_artifact_id": frozen["publication_request_artifact_id"],
            "execution_root": str(root),
            "run_nonce": nonce,
            "expected_integrity_artifact_ids": dict(expected_integrity_ids),
            "integrity_receipts": all_integrity_receipts,
            "integrity_receipt_count": 8,
            "integrity_phase_order": list(_INTEGRITY_PHASE_ORDINALS),
            "last_integrity_receipt_artifact_id": final_integrity_receipt["artifact_id"],
            "prediction_stage_result": dict(prediction_result),
            "prediction_launcher_attestation": dict(launcher_attestation),
            "scoring_stage_manifest": dict(scoring_manifest),
            "scorer_report": dict(scorer_report),
            "exact_source_reader_development_admitted": scorer_report["exact_source_reader_development_admitted"],
            "verdict": scorer_report["verdict"],
            "external_execution_anchor_verified": True,
            "qualification_execution_authorized": True,
            "source_tree_artifact_id_verified": True,
            "bge_snapshot_tree_artifact_id_verified": True,
            "runtime_identity_artifact_id_verified": True,
            "formal_evidence_authorized": False,
            "campaign_execution_admitted": False,
            "readable_product_effect": False,
            "appendable_product_effect": False,
            "learnable_product_effect": False,
            "steerable_product_effect": False,
            "four_able_complete": False,
            "human_product_validation": False,
            "production_active": False,
            "os_security_boundary": False,
            "windows_directory_entry_durability_attested": False,
        }
    )
    _write_artifact_create_only(root / "final_manifest.json", final_manifest)
    return final_manifest


def _extract_frozen_execution_identities(
    protocol: Mapping[str, object],
) -> Mapping[str, object]:
    preflight = _mapping(
        protocol.get("qualification_preflight"),
        "protocol qualification_preflight",
    )
    source_tree = _mapping(
        protocol.get("execution_source_tree"),
        "protocol execution_source_tree",
    )
    bge_tree = _mapping(
        protocol.get("bge_snapshot_tree"),
        "protocol bge_snapshot_tree",
    )
    runtime = _mapping(protocol.get("runtime_identity"), "protocol runtime_identity")
    file_rows = _list(preflight.get("files"), "protocol preflight files")
    by_path: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(file_rows):
        row = _mapping(value, f"protocol preflight file {index}")
        relative_path = _text(row.get("path"), f"protocol preflight path {index}")
        if relative_path in by_path:
            raise ValueError("protocol preflight binding contains duplicate paths")
        by_path[relative_path] = row
    required_paths = ("manifest.json", "public/publication_request.json")
    if any(path not in by_path for path in required_paths):
        raise ValueError("protocol preflight binding omits outer-runner identities")
    expected_integrity_ids = {
        "source_tree_artifact_id": _digest(
            source_tree.get("artifact_id"),
            "protocol source tree artifact_id",
        ),
        "bge_snapshot_tree_artifact_id": _digest(
            bge_tree.get("artifact_id"),
            "protocol BGE snapshot tree artifact_id",
        ),
        "runtime_identity_artifact_id": _digest(
            runtime.get("artifact_id"),
            "protocol runtime identity artifact_id",
        ),
    }
    return {
        "qualification_protocol_id": _digest(
            preflight.get("qualification_protocol_id"),
            "protocol qualification_protocol_id",
        ),
        "preflight_binding_artifact_id": _digest(
            preflight.get("artifact_id"),
            "protocol preflight binding artifact_id",
        ),
        "preflight_manifest_artifact_id": _digest(
            by_path["manifest.json"].get("artifact_id"),
            "protocol preflight manifest artifact_id",
        ),
        "publication_request_artifact_id": _digest(
            by_path["public/publication_request.json"].get("artifact_id"),
            "protocol publication request artifact_id",
        ),
        "expected_integrity_artifact_ids": expected_integrity_ids,
        "execution_root": _text(
            protocol.get("proposed_execution_root"),
            "protocol proposed_execution_root",
        ),
    }


def _extract_child_import_inputs(
    protocol: Mapping[str, object],
    *,
    repository_root: pathlib.Path,
) -> tuple[
    tuple[pathlib.Path, ...],
    Mapping[str, Mapping[str, object]],
    pathlib.Path,
]:
    """Derive the only child import roots from the frozen protocol."""

    root = pathlib.Path(repository_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"qualification repository root is absent: {root}")
    source_tree = _mapping(
        protocol.get("execution_source_tree"),
        "protocol execution_source_tree",
    )
    rows = _list(source_tree.get("entries"), "protocol execution source entries")
    entries: dict[str, Mapping[str, object]] = {}
    source_root_relatives: set[pathlib.PurePosixPath] = set()
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"protocol execution source entry {index}")
        _exact_keys(
            row,
            {"path", "raw_sha256", "raw_bytes"},
            f"protocol execution source entry {index}",
        )
        path_text = _text(row["path"], f"protocol execution source path {index}")
        relative = pathlib.PurePosixPath(path_text)
        if (
            relative.is_absolute()
            or relative.as_posix() != path_text
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("protocol execution source path is not canonical relative POSIX")
        if path_text in entries:
            raise ValueError("protocol execution source entries contain duplicate paths")
        entries[path_text] = {
            "raw_sha256": _digest(
                row["raw_sha256"],
                f"protocol execution source raw_sha256 {index}",
            ),
            "raw_bytes": _nonnegative_integer(
                row["raw_bytes"],
                f"protocol execution source raw_bytes {index}",
            ),
        }
        parts = relative.parts
        if len(parts) >= 4 and parts[0] == "packages" and parts[2] == "src":
            source_root_relatives.add(pathlib.PurePosixPath(*parts[:3]))
    if not entries or not source_root_relatives:
        raise ValueError("protocol execution source tree has no repository source roots")
    source_roots = tuple(
        sorted(
            (root / relative for relative in source_root_relatives),
            key=lambda path: str(path).encode("utf-8"),
        )
    )

    runtime = _mapping(protocol.get("runtime_identity"), "protocol runtime_identity")
    explicit_site_root = runtime.get("site_packages_root")
    if explicit_site_root is not None:
        site_packages_root = pathlib.Path(_text(explicit_site_root, "runtime site_packages_root"))
        if not site_packages_root.is_absolute():
            raise ValueError("runtime site_packages_root must be absolute")
    else:
        distributions = _list(
            runtime.get("distributions"),
            "runtime distributions",
        )
        site_roots = {
            pathlib.Path(
                _text(
                    _mapping(value, f"runtime distribution {index}").get("dist_info_path"),
                    f"runtime distribution {index} dist_info_path",
                )
            ).parent
            for index, value in enumerate(distributions)
        }
        if len(site_roots) != 1:
            raise ValueError("runtime distributions must share one site-packages root")
        site_packages_root = next(iter(site_roots))
    return source_roots, entries, site_packages_root.resolve()


def _validate_prediction_stage_for_outer_runner(
    *,
    result: Mapping[str, object],
    execution_root: pathlib.Path,
    protocol_id: str,
    frozen: Mapping[str, object],
    initial_integrity_receipt: Mapping[str, object],
    expected_integrity_ids: Mapping[str, object],
    run_nonce: str,
) -> Mapping[str, object]:
    _exact_keys(
        result,
        {
            "schema_version",
            "qualification_protocol_id",
            "execution_protocol_id",
            "preflight_manifest_artifact_id",
            "publication_request_artifact_id",
            "launcher_attestation_artifact_id",
            "last_integrity_receipt_artifact_id",
            "training_corpus_artifact_id",
            "child_request_artifact_id",
            "prediction_ledger_artifact_id",
            "commit_receipt_artifact_id",
            "scoring_request_artifact_id",
            "scoring_request_path",
            "fresh_process_count",
            "predictor_processes_exited",
            "predictor_job_objects_empty",
            "deterministic_outputs_byte_exact",
            "parent_opened_challenge_labels",
            "parent_opened_group_split",
            "scorer_launched",
            "os_security_boundary",
            "windows_directory_entry_durability_attested",
            "qualification_scored",
            "external_execution_anchor_verified",
            "qualification_execution_authorized",
            "reader_development_admission",
            "readable_claim_proven",
            "four_able_claim_proven",
        },
        "prediction stage result",
    )
    required = {
        "schema_version": _PREDICTION_EXECUTOR_RESULT_SCHEMA_VERSION,
        "qualification_protocol_id": frozen["qualification_protocol_id"],
        "execution_protocol_id": protocol_id,
        "preflight_manifest_artifact_id": frozen["preflight_manifest_artifact_id"],
        "publication_request_artifact_id": frozen["publication_request_artifact_id"],
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
    if any(result[key] != value for key, value in required.items()):
        raise ValueError("prediction stage lineage or honesty boundary drifted")
    for field_name in (
        "launcher_attestation_artifact_id",
        "last_integrity_receipt_artifact_id",
        "training_corpus_artifact_id",
        "child_request_artifact_id",
        "prediction_ledger_artifact_id",
        "commit_receipt_artifact_id",
        "scoring_request_artifact_id",
    ):
        _digest(result[field_name], f"prediction {field_name}")
    if not execution_root.is_dir() or execution_root.is_symlink():
        raise ValueError("prediction stage did not publish a regular execution root")
    expected_scoring_path = (execution_root / "scoring_request.json").resolve()
    scoring_path = pathlib.Path(_text(result["scoring_request_path"], "prediction scoring_request_path")).resolve()
    if scoring_path != expected_scoring_path:
        raise ValueError("prediction scoring request escaped the execution root")
    scoring_request = _load_canonical_artifact(
        scoring_path,
        expected_schema_version=_SCORING_REQUEST_SCHEMA_VERSION,
        max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
    )
    _validate_scoring_request(scoring_request.payload)
    if (
        scoring_request.payload["artifact_id"] != result["scoring_request_artifact_id"]
        or scoring_request.payload["execution_protocol_id"] != protocol_id
        or scoring_request.payload["qualification_protocol_id"] != frozen["qualification_protocol_id"]
        or scoring_request.payload["run_nonce"] != run_nonce
    ):
        raise ValueError("prediction scoring request lineage drifted")

    launcher = _load_canonical_artifact(
        execution_root / "launcher_attestation.json",
        expected_schema_version=_PREDICTION_LAUNCHER_ATTESTATION_SCHEMA_VERSION,
        max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
    )
    if launcher.payload["artifact_id"] != result["launcher_attestation_artifact_id"]:
        raise ValueError("prediction launcher attestation identity mismatch")
    _validate_prediction_launcher_attestation_for_outer_runner(
        launcher.payload,
        result=result,
        execution_root=execution_root,
        protocol_id=protocol_id,
        qualification_protocol_id=_digest(
            frozen["qualification_protocol_id"],
            "qualification protocol id",
        ),
        run_nonce=run_nonce,
        initial_integrity_receipt=initial_integrity_receipt,
        expected_integrity_ids=expected_integrity_ids,
    )
    return launcher.payload


def _validate_prediction_launcher_attestation_for_outer_runner(
    payload: Mapping[str, object],
    *,
    result: Mapping[str, object],
    execution_root: pathlib.Path,
    protocol_id: str,
    qualification_protocol_id: str,
    run_nonce: str,
    initial_integrity_receipt: Mapping[str, object],
    expected_integrity_ids: Mapping[str, object],
) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "qualification_protocol_id",
            "execution_protocol_id",
            "child_request_artifact_id",
            "run_nonce",
            "runs",
            "run_count",
            "integrity_receipts",
            "integrity_receipt_count",
            "integrity_phase_order",
            "previous_integrity_receipt_artifact_id",
            "last_integrity_receipt_artifact_id",
            "expected_integrity_artifact_ids",
            "processes_created_suspended",
            "job_assigned_before_initial_thread_resume",
            "job_kill_on_close",
            "job_active_process_limit",
            "shell",
            "close_fds",
            "environment_built_from_empty_allowlist",
            "torchinductor_cache_directories_controlled",
            "torchinductor_cache_directories_distinct",
            "torchinductor_cache_directories_materialized",
            "torchinductor_cache_directories_empty",
            "source_capsule_used",
            "repository_import_path_used",
            "bge_snapshot_tree_verified_by_launcher",
            "external_execution_anchor_verified",
            "qualification_execution_authorized",
            "os_security_boundary",
            "windows_directory_entry_durability_attested",
            "artifact_id",
        },
        "prediction launcher attestation",
    )
    expected_phases = (
        "pre_prediction_child_1",
        "post_prediction_child_1",
        "pre_prediction_child_2",
        "post_prediction_child_2",
    )
    required = {
        "schema_version": _PREDICTION_LAUNCHER_ATTESTATION_SCHEMA_VERSION,
        "qualification_protocol_id": qualification_protocol_id,
        "execution_protocol_id": protocol_id,
        "child_request_artifact_id": result["child_request_artifact_id"],
        "run_nonce": run_nonce,
        "run_count": 2,
        "integrity_receipt_count": 4,
        "integrity_phase_order": list(expected_phases),
        "previous_integrity_receipt_artifact_id": initial_integrity_receipt["artifact_id"],
        "last_integrity_receipt_artifact_id": result["last_integrity_receipt_artifact_id"],
        "expected_integrity_artifact_ids": dict(expected_integrity_ids),
        "processes_created_suspended": True,
        "job_assigned_before_initial_thread_resume": True,
        "job_kill_on_close": True,
        "job_active_process_limit": 1,
        "shell": False,
        "close_fds": True,
        "environment_built_from_empty_allowlist": True,
        "torchinductor_cache_directories_controlled": True,
        "torchinductor_cache_directories_distinct": True,
        "torchinductor_cache_directories_materialized": True,
        "torchinductor_cache_directories_empty": True,
        "source_capsule_used": False,
        "repository_import_path_used": True,
        "bge_snapshot_tree_verified_by_launcher": False,
        "external_execution_anchor_verified": False,
        "qualification_execution_authorized": False,
        "os_security_boundary": False,
        "windows_directory_entry_durability_attested": False,
    }
    if any(payload[key] != value for key, value in required.items()):
        raise ValueError("prediction launcher attestation lineage drifted")
    runs = _list(payload["runs"], "prediction launcher runs")
    if len(runs) != 2:
        raise ValueError("prediction launcher must bind two runs")
    process_ids: list[int] = []
    cache_paths: list[pathlib.Path] = []
    for ordinal, value in enumerate(runs, start=1):
        process_id, cache_path = _validate_prediction_launcher_run_for_outer_runner(
            _mapping(value, f"prediction launcher run {ordinal}"),
            ordinal=ordinal,
            execution_root=execution_root,
            parent_run_nonce=run_nonce,
        )
        process_ids.append(process_id)
        cache_paths.append(cache_path)
    if len(set(process_ids)) != 2:
        raise ValueError("prediction launcher runs must bind distinct process ids")
    if len(set(cache_paths)) != 2:
        raise ValueError("prediction launcher runs must bind distinct TorchInductor cache directories")
    receipts = _list(payload["integrity_receipts"], "prediction integrity receipts")
    if len(receipts) != 4:
        raise ValueError("prediction launcher must bind four integrity receipts")
    previous_id = _digest(
        initial_integrity_receipt["artifact_id"],
        "initial integrity receipt artifact_id",
    )
    for phase, value in zip(expected_phases, receipts, strict=True):
        receipt = _validate_integrity_receipt_payload(
            _mapping(value, f"prediction integrity receipt {phase}"),
            phase=phase,
            execution_protocol_id=protocol_id,
            expected_integrity_ids={key: _digest(expected, key) for key, expected in expected_integrity_ids.items()},
            previous_integrity_receipt_artifact_id=previous_id,
        )
        previous_id = _digest(receipt["artifact_id"], "integrity artifact_id")
    if previous_id != result["last_integrity_receipt_artifact_id"]:
        raise ValueError("prediction launcher integrity chain tail mismatch")


def _validate_prediction_launcher_run_for_outer_runner(
    payload: Mapping[str, object],
    *,
    ordinal: int,
    execution_root: pathlib.Path,
    parent_run_nonce: str,
) -> tuple[int, pathlib.Path]:
    _exact_keys(
        payload,
        {
            "run_ordinal",
            "run_nonce",
            "process_id",
            "process_argv",
            "exit_code",
            "process_exited",
            "job_object_empty",
            "environment_contract_id",
            "environment_projection",
            "creation_flags",
            "shell",
            "close_fds",
            "process_created_suspended",
            "job_assigned_before_resume",
            "initial_thread_resume_previous_count",
            "job_limit_flags",
            "job_active_process_limit",
            "stdout_capture",
            "stderr_capture",
            "torchinductor_cache",
            "source_capsule_used",
            "repository_import_path_used",
            "bge_snapshot_path_sha256",
            "bge_snapshot_tree_verified",
        },
        f"prediction launcher run {ordinal}",
    )
    expected_nonce = hashlib.sha256(f"{parent_run_nonce}:prediction-child:{ordinal}".encode("utf-8")).hexdigest()
    required = {
        "run_ordinal": ordinal,
        "run_nonce": expected_nonce,
        "exit_code": 0,
        "process_exited": True,
        "job_object_empty": True,
        "creation_flags": 0x08080004,
        "shell": False,
        "close_fds": True,
        "process_created_suspended": True,
        "job_assigned_before_resume": True,
        "initial_thread_resume_previous_count": 1,
        "job_limit_flags": 0x00002008,
        "job_active_process_limit": 1,
        "source_capsule_used": False,
        "repository_import_path_used": True,
        "bge_snapshot_tree_verified": False,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError(f"prediction launcher run {ordinal} containment contract drifted")
    process_id = _positive_integer(
        payload["process_id"],
        f"prediction launcher run {ordinal} process_id",
    )
    _digest(
        payload["bge_snapshot_path_sha256"],
        f"prediction launcher run {ordinal} bge_snapshot_path_sha256",
    )
    argv = _list(payload["process_argv"], f"prediction launcher run {ordinal} process_argv")
    if not argv or any(not isinstance(value, str) or not value for value in argv):
        raise ValueError(f"prediction launcher run {ordinal} process_argv is invalid")

    environment_rows = _list(
        payload["environment_projection"],
        f"prediction launcher run {ordinal} environment_projection",
    )
    canonical_rows: list[dict[str, object]] = []
    environment_by_key: dict[str, Mapping[str, object]] = {}
    environment_by_casefold: dict[str, str] = {}
    for index, value in enumerate(environment_rows):
        row = _mapping(value, f"prediction launcher run {ordinal} environment row {index}")
        _exact_keys(
            row,
            {"key", "value_sha256", "value_utf8_bytes"},
            f"prediction launcher run {ordinal} environment row {index}",
        )
        key = _text(row["key"], f"prediction launcher run {ordinal} environment key")
        if key in environment_by_key:
            raise ValueError(f"prediction launcher run {ordinal} environment keys must be unique")
        folded_key = key.casefold()
        if folded_key in environment_by_casefold:
            raise ValueError(
                f"prediction launcher run {ordinal} environment keys must be unique under Windows case folding"
            )
        environment_by_casefold[folded_key] = key
        if key != key.upper():
            raise ValueError(
                f"prediction launcher run {ordinal} environment keys must use canonical uppercase spelling"
            )
        value_sha256 = _digest(
            row["value_sha256"],
            f"prediction launcher run {ordinal} environment value_sha256",
        )
        value_utf8_bytes = _nonnegative_integer(
            row["value_utf8_bytes"],
            f"prediction launcher run {ordinal} environment value_utf8_bytes",
        )
        canonical_row = {
            "key": key,
            "value_sha256": value_sha256,
            "value_utf8_bytes": value_utf8_bytes,
        }
        canonical_rows.append(canonical_row)
        environment_by_key[key] = canonical_row
    if [row["key"] for row in canonical_rows] != sorted(environment_by_key):
        raise ValueError(f"prediction launcher run {ordinal} environment rows are not canonical")
    allowed_environment_keys = {
        "APPDATA",
        "CUDA_VISIBLE_DEVICES",
        "HF_HUB_OFFLINE",
        "KMP_DUPLICATE_LIB_OK",
        "KMP_INIT_AT_FORK",
        "LOCALAPPDATA",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "PYTHONPYCACHEPREFIX",
        "PYTHONSAFEPATH",
        "PYTHONUTF8",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TOKENIZERS_PARALLELISM",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRANSFORMERS_OFFLINE",
        "USERNAME",
        "USERPROFILE",
        "WINDIR",
    }
    if not set(environment_by_key) <= allowed_environment_keys:
        raise ValueError(f"prediction launcher run {ordinal} environment contains a non-allowlisted key")
    expected_contract_id = _sha256_json(
        {
            "environment_projection": canonical_rows,
            "environment_built_from_empty_allowlist": True,
        }
    )
    if payload["environment_contract_id"] != expected_contract_id:
        raise ValueError(f"prediction launcher run {ordinal} environment contract id mismatch")

    capsule = pathlib.Path(execution_root).resolve() / "predictor_capsule"
    expected_pycache = capsule / f"pycache-run-{ordinal}"
    expected_torchinductor = capsule / f"torchinductor-cache-run-{ordinal}"
    expected_environment_values = {
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "KMP_DUPLICATE_LIB_OK": "True",
        "KMP_INIT_AT_FORK": "FALSE",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": str(expected_pycache),
        "PYTHONSAFEPATH": "1",
        "PYTHONUTF8": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TORCHINDUCTOR_CACHE_DIR": str(expected_torchinductor),
        "TRANSFORMERS_OFFLINE": "1",
    }
    for key, expected_value in expected_environment_values.items():
        row = environment_by_key.get(key)
        if row is None:
            raise ValueError(f"prediction launcher run {ordinal} omitted environment key {key}")
        encoded = expected_value.encode("utf-8")
        if row["value_sha256"] != hashlib.sha256(encoded).hexdigest() or row["value_utf8_bytes"] != len(encoded):
            raise ValueError(f"prediction launcher run {ordinal} environment value {key} drifted")
    for key in ("PATH", "PYTHONPATH", "SYSTEMROOT", "USERNAME"):
        receipt = environment_by_key.get(key)
        if receipt is None or receipt["value_utf8_bytes"] == 0:
            raise ValueError(f"prediction launcher run {ordinal} requires a non-empty hashed {key}")

    cache = _mapping(
        payload["torchinductor_cache"],
        f"prediction launcher run {ordinal} TorchInductor cache",
    )
    _exact_keys(
        cache,
        {
            "path",
            "absent_before_launch",
            "materialized",
            "is_directory",
            "is_symlink",
            "is_reparse_point",
            "empty",
        },
        f"prediction launcher run {ordinal} TorchInductor cache",
    )
    expected_cache_receipt = {
        "path": str(expected_torchinductor),
        "absent_before_launch": True,
        "materialized": True,
        "is_directory": True,
        "is_symlink": False,
        "is_reparse_point": False,
        "empty": True,
    }
    if cache != expected_cache_receipt:
        raise ValueError(f"prediction launcher run {ordinal} TorchInductor cache receipt drifted")
    if not capsule.is_dir() or capsule.is_symlink():
        raise ValueError(f"prediction launcher run {ordinal} capsule is not a plain directory")
    capsule_stat = capsule.stat(follow_symlinks=False)
    if os.name == "nt" and getattr(capsule_stat, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"prediction launcher run {ordinal} capsule is a reparse point")
    if not expected_torchinductor.is_dir() or expected_torchinductor.is_symlink():
        raise ValueError(f"prediction launcher run {ordinal} TorchInductor cache is not a plain directory")
    cache_stat = expected_torchinductor.stat(follow_symlinks=False)
    if os.name == "nt" and getattr(cache_stat, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f"prediction launcher run {ordinal} TorchInductor cache is a reparse point")
    if next(expected_torchinductor.iterdir(), None) is not None:
        raise ValueError(f"prediction launcher run {ordinal} TorchInductor cache is not empty")
    cache_after = expected_torchinductor.stat(follow_symlinks=False)
    if _file_identity(cache_stat) != _file_identity(cache_after):
        raise ValueError(f"prediction launcher run {ordinal} TorchInductor cache changed during observation")
    if os.path.lexists(expected_pycache):
        raise ValueError(f"prediction launcher run {ordinal} Python pycache prefix was materialized")
    return process_id, expected_torchinductor


def _validate_scoring_stage_for_outer_runner(
    *,
    result: Mapping[str, object],
    stage_root: pathlib.Path,
    protocol_id: str,
    expected_scoring_request_artifact_id: str,
    previous_integrity_receipt_artifact_id: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    loaded_manifest = _load_canonical_artifact(
        stage_root / "scorer_stage_manifest.json",
        expected_schema_version=(RELATIONSHIP_READER_QUALIFICATION_SCORING_STAGE_MANIFEST_SCHEMA_VERSION),
        max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
    )
    if loaded_manifest.payload != result:
        raise ValueError("returned scoring stage differs from its canonical manifest")
    required = {
        "execution_protocol_id": protocol_id,
        "scoring_request_artifact_id": expected_scoring_request_artifact_id,
        "integrity_receipt_count": 2,
        "integrity_phase_order": ["pre_scorer", "post_scorer"],
        "previous_integrity_receipt_artifact_id": (previous_integrity_receipt_artifact_id),
        "external_execution_anchor_verified": False,
        "qualification_execution_authorized": False,
        "formal_evidence_authorized": False,
        "campaign_execution_admitted": False,
        "readable_product_effect": False,
        "appendable_product_effect": False,
        "learnable_product_effect": False,
        "steerable_product_effect": False,
        "four_able_complete": False,
        "os_security_boundary": False,
    }
    if any(result.get(key) != value for key, value in required.items()):
        raise ValueError("scoring stage lineage or authorization ceiling drifted")
    _digest(
        result.get("last_integrity_receipt_artifact_id"),
        "scoring last integrity receipt artifact_id",
    )
    report = _load_canonical_artifact(
        stage_root / "scorer_output" / "report.json",
        expected_schema_version=_REPORT_SCHEMA_VERSION,
        max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
    )
    if (
        report.payload["artifact_id"] != result.get("scorer_report_artifact_id")
        or report.payload["artifact_id"]
        != _mapping(
            result.get("scorer_report_receipt"),
            "scorer report receipt",
        ).get("artifact_id")
        or report.raw_sha256
        != _mapping(
            result.get("scorer_report_receipt"),
            "scorer report receipt",
        ).get("raw_sha256")
        or len(report.raw)
        != _mapping(
            result.get("scorer_report_receipt"),
            "scorer report receipt",
        ).get("raw_bytes")
    ):
        raise ValueError("scoring stage report receipt identity mismatch")
    scoring_request = _load_canonical_artifact(
        pathlib.Path(_text(result.get("scoring_request_path"), "scoring request path")),
        expected_schema_version=_SCORING_REQUEST_SCHEMA_VERSION,
        max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
    )
    if scoring_request.payload["artifact_id"] != expected_scoring_request_artifact_id:
        raise ValueError("scoring stage request identity mismatch")
    _validate_scorer_report(report.payload, request=scoring_request.payload)
    return loaded_manifest.payload, report.payload


def _collect_and_validate_full_integrity_chain(
    *,
    initial: Mapping[str, object],
    launcher_attestation: Mapping[str, object],
    scoring_manifest: Mapping[str, object],
    final: Mapping[str, object],
    execution_protocol_id: str,
    expected_integrity_ids: Mapping[str, object],
) -> list[Mapping[str, object]]:
    raw_receipts = [
        initial,
        *_list(
            launcher_attestation["integrity_receipts"],
            "launcher integrity receipts",
        ),
        *_list(scoring_manifest["integrity_receipts"], "scorer integrity receipts"),
        final,
    ]
    phases = tuple(_INTEGRITY_PHASE_ORDINALS)
    if len(raw_receipts) != len(phases):
        raise ValueError("qualification must bind all eight integrity phases")
    expected_ids = {key: _digest(value, key) for key, value in expected_integrity_ids.items()}
    receipts: list[Mapping[str, object]] = []
    previous_id: str | None = None
    for phase, value in zip(phases, raw_receipts, strict=True):
        receipt = _validate_integrity_receipt_payload(
            _mapping(value, f"integrity receipt {phase}"),
            phase=phase,
            execution_protocol_id=execution_protocol_id,
            expected_integrity_ids=expected_ids,
            previous_integrity_receipt_artifact_id=previous_id,
        )
        receipts.append(receipt)
        previous_id = _digest(receipt["artifact_id"], "integrity artifact_id")
    if len({receipt["artifact_id"] for receipt in receipts}) != len(receipts):
        raise ValueError("qualification integrity receipts must be phase-distinct")
    return receipts


def _execute_scoring_stage_with_launcher(
    *,
    scoring_request_path: pathlib.Path,
    expected_scoring_request_artifact_id: str,
    stage_root: pathlib.Path,
    python_executable: pathlib.Path,
    environment_source: Mapping[str, str],
    launcher: _ScorerLauncher,
    integrity_guard: _IntegrityGuard,
    previous_integrity_receipt_artifact_id: str,
    expected_source_tree_artifact_id: str,
    expected_bge_snapshot_tree_artifact_id: str,
    expected_runtime_identity_artifact_id: str,
    repository_root: pathlib.Path,
    repository_source_roots: tuple[pathlib.Path, ...],
    frozen_source_entries: Mapping[str, Mapping[str, object]],
    frozen_site_packages_root: pathlib.Path,
) -> Mapping[str, object]:
    """Injectable orchestration core; fixture launchers never run the scorer."""

    request_path = pathlib.Path(scoring_request_path)
    if not request_path.is_absolute():
        raise ValueError("scoring_request_path must be absolute")
    expected_request_id = _digest(
        expected_scoring_request_artifact_id,
        "expected_scoring_request_artifact_id",
    )
    request = _load_canonical_artifact(
        request_path,
        expected_schema_version=_SCORING_REQUEST_SCHEMA_VERSION,
        max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
    )
    _validate_scoring_request(request.payload)
    if request.payload["artifact_id"] != expected_request_id:
        raise ValueError("scoring request external artifact identity mismatch")

    executable = pathlib.Path(python_executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"scorer Python executable is absent: {executable}")
    if not callable(launcher):
        raise TypeError("scorer launcher must be callable")
    if not callable(integrity_guard):
        raise TypeError("scorer execution requires an integrity_guard callback")
    initial_integrity_receipt_id = _digest(
        previous_integrity_receipt_artifact_id,
        "previous_integrity_receipt_artifact_id",
    )
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
    import_binding = build_qualification_child_import_binding(
        python_executable=executable,
        repository_root=repository_root,
        repository_source_roots=repository_source_roots,
        frozen_source_entries=frozen_source_entries,
        frozen_site_packages_root=frozen_site_packages_root,
    )
    root = pathlib.Path(stage_root).resolve()
    if request_path.resolve() == root or root in request_path.resolve().parents:
        raise ValueError("scoring request must be outside the scorer stage root")
    if root.exists():
        raise FileExistsError(f"qualification scoring stage root exists: {root}")
    capsule_root = root / "scorer_capsule"
    output_root = root / "scorer_output"
    pycache_prefix = capsule_root / "pycache"
    environment, projection, environment_contract_id = _build_formal_scorer_environment(
        environment_source,
        import_binding=import_binding,
        pycache_prefix=pycache_prefix,
    )
    pre_scorer_integrity_receipt = _run_integrity_guard(
        integrity_guard,
        phase="pre_scorer",
        execution_protocol_id=_digest(
            request.payload["execution_protocol_id"],
            "execution_protocol_id",
        ),
        expected_integrity_ids=expected_integrity_ids,
        previous_integrity_receipt_artifact_id=initial_integrity_receipt_id,
    )

    root.mkdir(parents=True)
    capsule_root.mkdir()

    spec = _ScorerLaunchSpec(
        scoring_request_path=request_path.resolve(),
        expected_scoring_request_artifact_id=expected_request_id,
        output_root=output_root.resolve(),
        capsule_root=capsule_root.resolve(),
        run_nonce=_digest(request.payload["run_nonce"], "run_nonce"),
        python_executable=executable,
        import_binding=import_binding,
        pycache_prefix=pycache_prefix,
        environment_items=tuple(environment.items()),
        environment_projection=projection,
        environment_contract_id=environment_contract_id,
    )
    launch_result = launcher(spec)
    _validate_scorer_launch_result(launch_result, spec=spec)
    if launch_result.exit_code != 0:
        stderr_excerpt = launch_result.stderr_capture.retained_prefix.decode(
            "utf-8",
            errors="replace",
        )[:4_096]
        raise RuntimeError(
            "qualification scorer exited nonzero; "
            f"exit_code={launch_result.exit_code}, stderr_prefix={stderr_excerpt!r}"
        )

    loaded = _load_and_validate_scorer_outputs(
        spec=spec,
        launch_result=launch_result,
        request=request,
    )
    if any(capsule_root.iterdir()):
        raise ValueError("scorer capsule must remain empty")
    _assert_loaded_artifacts_still_stable(loaded.values())
    post_scorer_integrity_receipt = _run_integrity_guard(
        integrity_guard,
        phase="post_scorer",
        execution_protocol_id=_digest(
            request.payload["execution_protocol_id"],
            "execution_protocol_id",
        ),
        expected_integrity_ids=expected_integrity_ids,
        previous_integrity_receipt_artifact_id=_digest(
            pre_scorer_integrity_receipt["artifact_id"],
            "pre_scorer integrity receipt artifact_id",
        ),
    )
    _validate_integrity_receipt_consistency(
        pre_scorer_integrity_receipt,
        post_scorer_integrity_receipt,
    )

    report = loaded["report.json"].payload
    attestation = loaded["scorer_attestation.json"].payload
    scorer_manifest = loaded["manifest.json"].payload
    stage_manifest = _with_artifact_id(
        {
            "schema_version": (RELATIONSHIP_READER_QUALIFICATION_SCORING_STAGE_MANIFEST_SCHEMA_VERSION),
            "qualification_protocol_id": request.payload["qualification_protocol_id"],
            "execution_protocol_id": request.payload["execution_protocol_id"],
            "run_nonce": request.payload["run_nonce"],
            "scoring_request_path": str(request.path.resolve()),
            "scoring_request_artifact_id": request.payload["artifact_id"],
            "scoring_request_raw_sha256": request.raw_sha256,
            "scoring_request_raw_bytes": len(request.raw),
            "scorer_output_root": str(output_root.resolve()),
            "scorer_report_receipt": _artifact_receipt(loaded["report.json"]),
            "scorer_attestation_receipt": _artifact_receipt(loaded["scorer_attestation.json"]),
            "scorer_manifest_receipt": _artifact_receipt(loaded["manifest.json"]),
            "scorer_report_artifact_id": report["artifact_id"],
            "scorer_attestation_artifact_id": attestation["artifact_id"],
            "scorer_manifest_artifact_id": scorer_manifest["artifact_id"],
            "scorer_process_id": launch_result.process_id,
            "scorer_parent_process_id": os.getpid(),
            "fresh_scorer_process_count": 1,
            "scorer_process_exited": True,
            "scorer_exit_code": 0,
            "scorer_job_object_empty": True,
            "process_created_suspended": True,
            "job_assigned_before_resume": True,
            "initial_thread_resume_previous_count": 1,
            "job_limit_flags": _FORMAL_SCORER_JOB_LIMIT_FLAGS,
            "job_active_process_limit": 1,
            "environment_built_from_empty_allowlist": True,
            "environment_contract_id": launch_result.environment_contract_id,
            "environment_projection": _environment_projection_payload(launch_result.environment_projection),
            "stdout_capture": _stream_capture_payload(launch_result.stdout_capture),
            "stderr_capture": _stream_capture_payload(launch_result.stderr_capture),
            "scoring_request_validated_before_scorer_launch": True,
            "scorer_outputs_validated_after_process_exit": True,
            "scorer_event_sequence": list(_SCORER_EVENT_SEQUENCE),
            "integrity_receipts": [
                pre_scorer_integrity_receipt,
                post_scorer_integrity_receipt,
            ],
            "integrity_receipt_count": 2,
            "integrity_phase_order": ["pre_scorer", "post_scorer"],
            "previous_integrity_receipt_artifact_id": (initial_integrity_receipt_id),
            "last_integrity_receipt_artifact_id": (post_scorer_integrity_receipt["artifact_id"]),
            "expected_integrity_artifact_ids": expected_integrity_ids,
            "scorer_model_free": True,
            "model_or_cuda_used": False,
            "torch_imported": False,
            "sentence_transformers_imported": False,
            "exact_source_reader_development_admitted": report["exact_source_reader_development_admitted"],
            "verdict": report["verdict"],
            "source_capsule_used": False,
            "repository_import_path_used": True,
            "source_tree_artifact_id_verified": True,
            "bge_snapshot_tree_artifact_id_verified": True,
            "runtime_identity_artifact_id_verified": True,
            "external_execution_anchor_verified": False,
            "qualification_execution_authorized": False,
            "formal_evidence_authorized": False,
            "campaign_execution_admitted": False,
            "readable_product_effect": False,
            "appendable_product_effect": False,
            "learnable_product_effect": False,
            "steerable_product_effect": False,
            "four_able_complete": False,
            "human_product_validation": False,
            "production_active": False,
            "os_security_boundary": False,
            "windows_directory_entry_durability_attested": False,
        }
    )
    _assert_loaded_artifacts_still_stable(loaded.values())
    _write_artifact_create_only(root / "scorer_stage_manifest.json", stage_manifest)
    return stage_manifest


def _load_and_validate_scorer_outputs(
    *,
    spec: _ScorerLaunchSpec,
    launch_result: _ScorerLaunchResult,
    request: _LoadedArtifact,
) -> dict[str, _LoadedArtifact]:
    output_root = spec.output_root
    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("scorer output root must be a non-symlink directory")
    root_stat = output_root.stat(follow_symlinks=False)
    if os.name == "nt" and getattr(root_stat, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError("scorer output root must not be a reparse point")
    observed_paths = tuple(sorted(path.name for path in output_root.iterdir()))
    if observed_paths != tuple(sorted(_SCORER_OUTPUT_PATHS)):
        raise ValueError("scorer output file set drifted")
    loaded = {
        "report.json": _load_canonical_artifact(
            output_root / "report.json",
            expected_schema_version=_REPORT_SCHEMA_VERSION,
            max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
        ),
        "scorer_attestation.json": _load_canonical_artifact(
            output_root / "scorer_attestation.json",
            expected_schema_version=_SCORER_ATTESTATION_SCHEMA_VERSION,
            max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
        ),
        "manifest.json": _load_canonical_artifact(
            output_root / "manifest.json",
            expected_schema_version=_SCORER_MANIFEST_SCHEMA_VERSION,
            max_bytes=_MAX_SMALL_ARTIFACT_BYTES,
        ),
    }
    _validate_scorer_report(loaded["report.json"].payload, request=request.payload)
    _validate_scorer_attestation(
        loaded["scorer_attestation.json"].payload,
        spec=spec,
        launch_result=launch_result,
        request=request.payload,
    )
    _validate_scorer_manifest(
        loaded["manifest.json"].payload,
        request=request.payload,
        loaded=loaded,
    )
    if tuple(sorted(path.name for path in output_root.iterdir())) != observed_paths:
        raise ValueError("scorer output file set changed during verification")
    return loaded


def _validate_scoring_request(payload: Mapping[str, object]) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "qualification_protocol_id",
            "execution_protocol_id",
            "run_nonce",
            "prediction_ledger_path",
            "prediction_ledger_artifact_id",
            "prediction_ledger_raw_sha256",
            "prediction_ledger_raw_bytes",
            "commit_receipt_path",
            "commit_receipt_artifact_id",
            "challenge_labels_path",
            "challenge_labels_artifact_id",
            "challenge_labels_raw_sha256",
            "challenge_labels_raw_bytes",
            "group_split_path",
            "group_split_artifact_id",
            "group_split_raw_sha256",
            "group_split_raw_bytes",
            "minimum_normalized_margin_hex",
            "artifact_id",
        },
        "scoring request",
    )
    if payload["schema_version"] != _SCORING_REQUEST_SCHEMA_VERSION:
        raise ValueError("scoring request schema drifted")
    for field_name in (
        "qualification_protocol_id",
        "execution_protocol_id",
        "run_nonce",
        "prediction_ledger_artifact_id",
        "prediction_ledger_raw_sha256",
        "commit_receipt_artifact_id",
        "challenge_labels_artifact_id",
        "challenge_labels_raw_sha256",
        "group_split_artifact_id",
        "group_split_raw_sha256",
        "artifact_id",
    ):
        _digest(payload[field_name], field_name)
    if payload["qualification_protocol_id"] == payload["execution_protocol_id"]:
        raise ValueError("execution protocol must differ from qualification protocol")
    for field_name in (
        "prediction_ledger_path",
        "commit_receipt_path",
        "challenge_labels_path",
        "group_split_path",
    ):
        path_text = _text(payload[field_name], field_name)
        if not pathlib.Path(path_text).is_absolute():
            raise ValueError(f"{field_name} must be absolute")
    for field_name in (
        "prediction_ledger_raw_bytes",
        "challenge_labels_raw_bytes",
        "group_split_raw_bytes",
    ):
        _positive_integer(payload[field_name], field_name)
    if (
        _canonical_hex_float(
            payload["minimum_normalized_margin_hex"],
            "minimum_normalized_margin_hex",
        )
        != 0.01
    ):
        raise ValueError("qualification minimum margin must remain exactly 0.01")


def _validate_scorer_report(
    payload: Mapping[str, object],
    *,
    request: Mapping[str, object],
) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "qualification_protocol_id",
            "execution_protocol_id",
            "scoring_request_artifact_id",
            "prediction_ledger_artifact_id",
            "challenge_labels_artifact_id",
            "group_split_artifact_id",
            "row_count",
            "effective_group_count",
            "rows_per_group",
            "correct_row_count",
            "margin_passing_row_count",
            "passing_row_count",
            "passing_group_count",
            "minimum_normalized_margin_hex",
            "exact_source_reader_development_admitted",
            "verdict",
            "statistical_independence_claim",
            "campaign_execution_admitted",
            "readable_product_effect",
            "appendable_product_effect",
            "learnable_product_effect",
            "steerable_product_effect",
            "four_able_complete",
            "formal_evidence_authorized",
            "human_product_validation",
            "production_active",
            "artifact_id",
        },
        "scorer report",
    )
    required_lineage = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "qualification_protocol_id": request["qualification_protocol_id"],
        "execution_protocol_id": request["execution_protocol_id"],
        "scoring_request_artifact_id": request["artifact_id"],
        "prediction_ledger_artifact_id": request["prediction_ledger_artifact_id"],
        "challenge_labels_artifact_id": request["challenge_labels_artifact_id"],
        "group_split_artifact_id": request["group_split_artifact_id"],
        "row_count": 224,
        "effective_group_count": 28,
        "rows_per_group": 8,
        "minimum_normalized_margin_hex": request["minimum_normalized_margin_hex"],
    }
    if any(payload[key] != value for key, value in required_lineage.items()):
        raise ValueError("scorer report lineage or frozen counts drifted")
    correct_count = _bounded_integer(payload["correct_row_count"], 224, "correct")
    margin_count = _bounded_integer(
        payload["margin_passing_row_count"],
        224,
        "margin passing",
    )
    passing_count = _bounded_integer(payload["passing_row_count"], 224, "passing")
    passing_groups = _bounded_integer(
        payload["passing_group_count"],
        28,
        "passing group",
    )
    if passing_count > min(correct_count, margin_count):
        raise ValueError("scorer passing count exceeds component counts")
    admitted = _boolean(
        payload["exact_source_reader_development_admitted"],
        "exact_source_reader_development_admitted",
    )
    expected_admitted = correct_count == 224 and margin_count == 224 and passing_count == 224 and passing_groups == 28
    if admitted is not expected_admitted:
        raise ValueError("scorer report admission disagrees with frozen threshold")
    expected_verdict = (
        "exact_source_reader_development_admitted" if admitted else "exact_source_reader_development_not_admitted"
    )
    if payload["verdict"] != expected_verdict:
        raise ValueError("scorer report verdict disagrees with admission")
    for field_name in (
        "statistical_independence_claim",
        "campaign_execution_admitted",
        "readable_product_effect",
        "appendable_product_effect",
        "learnable_product_effect",
        "steerable_product_effect",
        "four_able_complete",
        "formal_evidence_authorized",
        "human_product_validation",
        "production_active",
    ):
        if payload[field_name] is not False:
            raise ValueError(f"scorer report must keep {field_name}=false")


def _validate_scorer_attestation(
    payload: Mapping[str, object],
    *,
    spec: _ScorerLaunchSpec,
    launch_result: _ScorerLaunchResult,
    request: Mapping[str, object],
) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "qualification_protocol_id",
            "execution_protocol_id",
            "scoring_request_artifact_id",
            "prediction_ledger_commit_artifact_id",
            "process_pid",
            "parent_pid",
            "run_nonce",
            "process_executable",
            "process_argv",
            "process_cwd",
            "process_sys_path",
            "process_runtime_flags",
            "environment_key_names",
            "environment_value_sha256s",
            "unlisted_environment_variables_recorded",
            "loaded_file_backed_module_origins",
            "volvence_zero_namespace_search_locations",
            "event_sequence",
            "challenge_labels_first_open_after_commit_validation",
            "model_or_cuda_used",
            "torch_imported",
            "sentence_transformers_imported",
            "os_security_boundary",
            "windows_directory_entry_durability_attested",
            "artifact_id",
        },
        "scorer attestation",
    )
    required = {
        "schema_version": _SCORER_ATTESTATION_SCHEMA_VERSION,
        "qualification_protocol_id": request["qualification_protocol_id"],
        "execution_protocol_id": request["execution_protocol_id"],
        "scoring_request_artifact_id": request["artifact_id"],
        "prediction_ledger_commit_artifact_id": request["commit_receipt_artifact_id"],
        "process_pid": launch_result.process_id,
        "parent_pid": os.getpid(),
        "run_nonce": request["run_nonce"],
        "process_argv": _expected_scorer_attestation_argv(spec),
        "process_cwd": str(spec.capsule_root.resolve()),
        "event_sequence": list(_SCORER_EVENT_SEQUENCE),
        "challenge_labels_first_open_after_commit_validation": True,
        "model_or_cuda_used": False,
        "torch_imported": False,
        "sentence_transformers_imported": False,
        "os_security_boundary": False,
        "windows_directory_entry_durability_attested": False,
    }
    if any(payload[key] != value for key, value in required.items()):
        raise ValueError("scorer attestation lineage, process, or honesty drifted")
    executable_text = _text(payload["process_executable"], "process_executable")
    if pathlib.Path(executable_text).resolve() != spec.python_executable.resolve():
        raise ValueError("scorer executable differs from launched executable")

    expected_names = [item.key for item in spec.environment_projection]
    if payload["environment_key_names"] != expected_names:
        raise ValueError("scorer environment key set differs from exact allowlist")
    hashes = _mapping(
        payload["environment_value_sha256s"],
        "environment_value_sha256s",
    )
    expected_hashes = {item.key: item.value_sha256 for item in spec.environment_projection}
    if hashes != expected_hashes:
        raise ValueError("scorer environment values differ from launcher projection")
    if payload["unlisted_environment_variables_recorded"] is not True:
        raise ValueError("scorer must attest its complete environment")

    runtime_flags = _mapping(
        payload["process_runtime_flags"],
        "process_runtime_flags",
    )
    expected_runtime_flags: Mapping[str, object] = {
        "dont_write_bytecode": True,
        "no_site": 1,
        "pycache_prefix": str(spec.pycache_prefix),
        "safe_path": True,
        "utf8_mode": 1,
    }
    _exact_keys(
        runtime_flags,
        set(expected_runtime_flags),
        "process_runtime_flags",
    )
    if any(
        type(runtime_flags[key]) is not type(value) or runtime_flags[key] != value
        for key, value in expected_runtime_flags.items()
    ):
        raise ValueError("scorer interpreter isolation flags drifted")

    process_sys_path = _list(payload["process_sys_path"], "process_sys_path")
    expected_sys_path = list(
        expected_child_sys_path(
            spec.import_binding,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        )
    )
    if process_sys_path != expected_sys_path:
        raise ValueError("scorer process_sys_path differs from the controlled import path")

    origins = _list(
        payload["loaded_file_backed_module_origins"],
        "loaded_file_backed_module_origins",
    )
    for index, value in enumerate(origins):
        origin = _mapping(value, f"loaded module origin {index}")
        _exact_keys(
            origin,
            {"module_name", "origin"},
            f"loaded module origin {index}",
        )
        module_name = _text(origin["module_name"], "loaded module_name")
        origin_path = pathlib.Path(_text(origin["origin"], "loaded module origin"))
        if not origin_path.is_absolute():
            raise ValueError("loaded module origin must be absolute")
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in _FORBIDDEN_MODEL_MODULE_PREFIXES
        ):
            raise ValueError("model module was loaded in the scorer process")
    validate_child_file_backed_module_origin_attestation(
        loaded_module_origins=origins,
        volvence_zero_namespace_search_locations=(payload["volvence_zero_namespace_search_locations"]),
        binding=spec.import_binding,
        required_module_names=_REQUIRED_SCORER_MODULES,
    )


def _validate_scorer_manifest(
    payload: Mapping[str, object],
    *,
    request: Mapping[str, object],
    loaded: Mapping[str, _LoadedArtifact],
) -> None:
    _exact_keys(
        payload,
        {
            "schema_version",
            "qualification_protocol_id",
            "execution_protocol_id",
            "scoring_request_artifact_id",
            "files",
            "file_count",
            "model_or_cuda_used",
            "artifact_id",
        },
        "scorer manifest",
    )
    required = {
        "schema_version": _SCORER_MANIFEST_SCHEMA_VERSION,
        "qualification_protocol_id": request["qualification_protocol_id"],
        "execution_protocol_id": request["execution_protocol_id"],
        "scoring_request_artifact_id": request["artifact_id"],
        "file_count": 2,
        "model_or_cuda_used": False,
    }
    if any(payload[key] != value for key, value in required.items()):
        raise ValueError("scorer manifest lineage or model-free boundary drifted")
    files = _list(payload["files"], "scorer manifest files")
    if len(files) != 2:
        raise ValueError("scorer manifest must bind exactly two pre-manifest files")
    receipts: list[Mapping[str, object]] = []
    for index, value in enumerate(files):
        receipt = _mapping(value, f"scorer manifest file {index}")
        _exact_keys(
            receipt,
            {"path", "artifact_id", "raw_sha256", "raw_bytes"},
            f"scorer manifest file {index}",
        )
        receipts.append(receipt)
    if [receipt["path"] for receipt in receipts] != list(_SCORER_MANIFEST_BOUND_PATHS):
        raise ValueError("scorer manifest file order or set drifted")
    for receipt, relative_path in zip(
        receipts,
        _SCORER_MANIFEST_BOUND_PATHS,
        strict=True,
    ):
        _validate_manifest_file_receipt(
            receipt,
            loaded[relative_path],
            f"scorer output {relative_path}",
        )


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
        raise ValueError(f"{field_name} receipt identity mismatch")


def _validate_scorer_launch_result(
    result: _ScorerLaunchResult,
    *,
    spec: _ScorerLaunchSpec,
) -> None:
    if not isinstance(result, _ScorerLaunchResult):
        raise TypeError("scorer launcher returned an unsupported result")
    if _positive_integer(result.process_id, "scorer process_id") == os.getpid():
        raise ValueError("scorer must run in a distinct process")
    if (
        result.exit_code is None
        or isinstance(result.exit_code, bool)
        or not isinstance(
            result.exit_code,
            int,
        )
    ):
        raise ValueError("scorer launcher must report an integer exit code")
    required = {
        "process_exited": True,
        "job_object_empty": True,
        "environment_contract_id": spec.environment_contract_id,
        "environment_projection": spec.environment_projection,
        "creation_flags": _FORMAL_SCORER_CREATION_FLAGS,
        "shell": False,
        "close_fds": True,
        "process_created_suspended": True,
        "job_assigned_before_resume": True,
        "initial_thread_resume_previous_count": 1,
        "job_limit_flags": _FORMAL_SCORER_JOB_LIMIT_FLAGS,
        "job_active_process_limit": 1,
        "source_capsule_used": False,
        "repository_import_path_used": True,
    }
    if any(getattr(result, key) != value for key, value in required.items()):
        raise ValueError("scorer launcher containment or environment contract drifted")
    _validate_stream_capture(result.stdout_capture, "scorer stdout")
    _validate_stream_capture(result.stderr_capture, "scorer stderr")


def _run_integrity_guard(
    guard: _IntegrityGuard,
    *,
    phase: str,
    execution_protocol_id: str,
    expected_integrity_ids: Mapping[str, str],
    previous_integrity_receipt_artifact_id: str | None,
) -> Mapping[str, object]:
    previous_id = (
        None
        if previous_integrity_receipt_artifact_id is None
        else _digest(
            previous_integrity_receipt_artifact_id,
            "previous integrity receipt artifact_id",
        )
    )
    receipt = _mapping(
        guard(
            phase=phase,
            previous_integrity_receipt_artifact_id=previous_id,
        ),
        f"integrity receipt {phase}",
    )
    return _validate_integrity_receipt_payload(
        receipt,
        phase=phase,
        execution_protocol_id=execution_protocol_id,
        expected_integrity_ids=expected_integrity_ids,
        previous_integrity_receipt_artifact_id=previous_id,
    )


def _validate_integrity_receipt_payload(
    receipt: Mapping[str, object],
    *,
    phase: str,
    execution_protocol_id: str,
    expected_integrity_ids: Mapping[str, str],
    previous_integrity_receipt_artifact_id: str | None,
) -> Mapping[str, object]:
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
    expected_ordinal = _INTEGRITY_PHASE_ORDINALS.get(phase)
    if expected_ordinal is None:
        raise ValueError(f"unsupported qualification integrity phase: {phase}")
    if (
        receipt["schema_version"] != RELATIONSHIP_READER_EXECUTION_INTEGRITY_RECEIPT_SCHEMA_VERSION
        or receipt["execution_protocol_id"] != execution_protocol_id
        or receipt["phase"] != phase
        or receipt["phase_ordinal"] != expected_ordinal
        or receipt["previous_integrity_receipt_artifact_id"] != previous_integrity_receipt_artifact_id
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
    _positive_integer(
        receipt["bge_snapshot_entry_count"],
        "BGE snapshot entry count",
    )
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


def _validate_integrity_receipt_consistency(
    pre_scorer: Mapping[str, object],
    post_scorer: Mapping[str, object],
) -> None:
    if (
        pre_scorer["phase"] != "pre_scorer"
        or post_scorer["phase"] != "post_scorer"
        or post_scorer["previous_integrity_receipt_artifact_id"] != pre_scorer["artifact_id"]
    ):
        raise ValueError("scorer integrity receipt chain drifted")
    for field_name in (
        "execution_protocol_id",
        "source_tree_artifact_id",
        "source_tree_entry_count",
        "bge_snapshot_tree_artifact_id",
        "bge_snapshot_entry_count",
        "runtime_identity_artifact_id",
    ):
        if pre_scorer[field_name] != post_scorer[field_name]:
            raise ValueError(f"scorer integrity guard observed {field_name} drift")
    if pre_scorer["artifact_id"] == post_scorer["artifact_id"]:
        raise ValueError("scorer integrity receipts must be phase-distinct")


def _validate_stream_capture(
    capture: _BoundedStreamCapture,
    field_name: str,
) -> None:
    if not isinstance(capture, _BoundedStreamCapture):
        raise TypeError(f"{field_name} capture has unsupported type")
    _digest(capture.raw_sha256, f"{field_name} raw_sha256")
    _digest(capture.retained_prefix_sha256, f"{field_name} retained_prefix_sha256")
    total = _nonnegative_integer(capture.total_bytes, f"{field_name} total_bytes")
    retained = _nonnegative_integer(
        capture.retained_prefix_bytes,
        f"{field_name} retained_prefix_bytes",
    )
    if retained != len(capture.retained_prefix):
        raise ValueError(f"{field_name} retained prefix byte count mismatch")
    if retained > _MAX_CAPTURED_STREAM_PREFIX_BYTES or retained > total:
        raise ValueError(f"{field_name} retained prefix exceeds its bound")
    if hashlib.sha256(capture.retained_prefix).hexdigest() != (capture.retained_prefix_sha256):
        raise ValueError(f"{field_name} retained prefix hash mismatch")
    if capture.prefix_truncated is not (total > retained):
        raise ValueError(f"{field_name} truncation flag mismatch")
    if total == retained and capture.raw_sha256 != capture.retained_prefix_sha256:
        raise ValueError(f"{field_name} full-stream hash mismatch")


def _launch_windows_fresh_scorer_subprocess(
    spec: _ScorerLaunchSpec,
    *,
    timeout_seconds: int,
) -> _ScorerLaunchResult:
    """Create suspended, assign a fresh one-process Job, then resume once."""

    if os.name != "nt":
        raise RuntimeError("formal relationship-reader scoring requires Windows")
    if os.path.lexists(spec.pycache_prefix):
        raise FileExistsError(f"formal scorer pycache prefix must be absent before launch: {spec.pycache_prefix}")
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
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    argv = _scorer_process_argv(spec)
    environment = dict(spec.environment_items)
    stdout_read = -1
    stdout_write = -1
    stderr_read = -1
    stderr_write = -1
    stdin_fd = -1
    process_handle = None
    thread_handle = None
    job = None
    stdout_drain: _BoundedPipeDrain | None = None
    stderr_drain: _BoundedPipeDrain | None = None
    process_wait_completed = False
    process_assigned_to_job = False
    initial_resume_count = -1
    try:
        stdout_read, stdout_write = os.pipe()
        stderr_read, stderr_write = os.pipe()
        stdin_fd = os.open(os.devnull, os.O_RDONLY)
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
        limits.BasicLimitInformation.LimitFlags = _FORMAL_SCORER_JOB_LIMIT_FLAGS
        limits.BasicLimitInformation.ActiveProcessLimit = 1
        if not kernel32.SetInformationJobObject(
            job,
            9,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")

        process_handle, thread_handle, process_id, _thread_id = _winapi.CreateProcess(
            str(spec.python_executable),
            subprocess.list2cmdline(argv),
            None,
            None,
            True,
            _FORMAL_SCORER_CREATION_FLAGS,
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
        stdout_write = -1
        os.close(stderr_write)
        stderr_write = -1
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
            raise RuntimeError("fresh scorer initial suspension count must be exactly one")
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
            raise TimeoutError("qualification scorer exceeded its timeout")
        if wait_result != 0:  # WAIT_OBJECT_0
            raise OSError(f"qualification scorer wait failed with code {wait_result}")
        process_wait_completed = True
        exit_code = int(_winapi.GetExitCodeProcess(process_handle))
        # Close the final process handle before observing the Job transition.
        _winapi.CloseHandle(process_handle)
        process_handle = None
        stdout_capture = stdout_drain.finish()
        stderr_capture = stderr_drain.finish()
        # With ActiveProcessLimit=1, the signaled primary process was the only
        # process that could ever be active in this Job.  Closing the last Job
        # handle under KILL_ON_JOB_CLOSE completes the containment boundary
        # before any scorer artifact is inspected.  Querying ActiveProcesses
        # is not a reliable completion gate because Windows decrements it only
        # after every system-wide reference to the terminated process is gone.
        if not kernel32.CloseHandle(job):
            raise OSError(ctypes.get_last_error(), "CloseHandle(job) failed")
        job = None
        return _ScorerLaunchResult(
            process_id=int(process_id),
            exit_code=exit_code,
            process_exited=True,
            job_object_empty=True,
            environment_contract_id=spec.environment_contract_id,
            environment_projection=spec.environment_projection,
            creation_flags=_FORMAL_SCORER_CREATION_FLAGS,
            shell=False,
            close_fds=True,
            process_created_suspended=True,
            job_assigned_before_resume=True,
            initial_thread_resume_previous_count=initial_resume_count,
            job_limit_flags=_FORMAL_SCORER_JOB_LIMIT_FLAGS,
            job_active_process_limit=1,
            stdout_capture=stdout_capture,
            stderr_capture=stderr_capture,
            source_capsule_used=False,
            repository_import_path_used=True,
        )
    finally:
        active_error = sys.exception()
        cleanup_errors: list[BaseException] = []
        if process_handle is not None and not process_wait_completed:
            if process_assigned_to_job:
                if job and not kernel32.TerminateJobObject(job, 1):
                    cleanup_errors.append(OSError(ctypes.get_last_error(), "TerminateJobObject failed"))
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
                    cleanup_errors.append(RuntimeError("qualification scorer could not be reaped after termination"))
                else:
                    process_wait_completed = True
            except BaseException as exc:  # pragma: no cover - Win32 cleanup boundary
                cleanup_errors.append(exc)
        if job and not kernel32.CloseHandle(job):
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
        for fd in (stdin_fd, stdout_write, stderr_write, stdout_read, stderr_read):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError as exc:  # pragma: no cover - Win32 cleanup boundary
                    cleanup_errors.append(exc)
        for drain in (stdout_drain, stderr_drain):
            if drain is not None:
                try:
                    drain.finish()
                except BaseException as exc:  # pragma: no cover - Win32 cleanup boundary
                    cleanup_errors.append(exc)
        if cleanup_errors:
            summary = "; ".join(f"{type(error).__name__}: {error}" for error in cleanup_errors)
            if active_error is not None:
                active_error.add_note(f"scorer launcher cleanup errors: {summary}")
            else:
                raise RuntimeError(f"scorer launcher cleanup failed: {summary}") from cleanup_errors[0]


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
            raise RuntimeError("scorer output drain did not terminate")
        if self._error is not None:
            raise RuntimeError("scorer output drain failed") from self._error
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


def _build_formal_scorer_environment(
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
            raise ValueError("formal scorer environment requires text keys and values")
        folded = key.casefold()
        if folded in by_casefold and by_casefold[folded][0] != key:
            raise ValueError("formal scorer environment contains case-ambiguous keys")
        by_casefold[folded] = (key, value)
    environment: dict[str, str] = {}
    for canonical_key in _FORMAL_SCORER_ENVIRONMENT_ALLOWLIST:
        found = by_casefold.get(canonical_key.casefold())
        if found is not None:
            environment[canonical_key] = found[1]
    environment.update(_FORMAL_SCORER_ENVIRONMENT_FIXED)
    if not isinstance(import_binding, QualificationChildImportBinding):
        raise TypeError("formal scorer import_binding is invalid")
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in import_binding.import_roots)
    cache_root = pathlib.Path(pycache_prefix)
    if not cache_root.is_absolute():
        raise ValueError("formal scorer pycache prefix must be absolute")
    if os.path.lexists(cache_root):
        raise FileExistsError(f"formal scorer pycache prefix must be absent before launch: {cache_root}")
    environment["PYTHONPYCACHEPREFIX"] = str(cache_root)
    system_root = environment.get("SYSTEMROOT")
    if system_root is None or not system_root:
        raise ValueError("formal scorer environment requires SYSTEMROOT")
    environment["PATH"] = os.pathsep.join(
        str(path)
        for path in controlled_child_path(
            import_binding,
            system_root=pathlib.Path(system_root),
        )
    )
    for required in ("PATH", "SYSTEMROOT"):
        if required not in environment or not environment[required]:
            raise ValueError(f"formal scorer environment requires {required}")
    for key, value in environment.items():
        if "\x00" in key or "=" in key or "\x00" in value:
            raise ValueError("formal scorer environment contains invalid Windows text")
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
            "environment_projection": _environment_projection_payload(projection),
            "environment_built_from_empty_allowlist": True,
        }
    )


def _environment_projection_payload(
    projection: tuple[_EnvironmentValueReceipt, ...],
) -> list[dict[str, object]]:
    return [
        {
            "key": item.key,
            "value_sha256": item.value_sha256,
            "value_utf8_bytes": item.value_utf8_bytes,
        }
        for item in projection
    ]


def _scorer_process_argv(spec: _ScorerLaunchSpec) -> list[str]:
    bootstrap = (
        "import json,pathlib,sys;"
        "roots=json.loads(sys.argv[1]);"
        "expected=roots+[str(pathlib.Path(sys.base_prefix)/f'python{sys.version_info.major}{sys.version_info.minor}.zip'),"
        "str(pathlib.Path(sys.base_prefix)/'DLLs'),str(pathlib.Path(sys.base_prefix)/'Lib'),sys.base_prefix];"
        "(_ for _ in ()).throw(RuntimeError('scorer bootstrap sys.path drifted')) if sys.path!=expected else None;"
        "from lifeform_evolution."
        "relationship_condition_reader_qualification_scorer "
        "import score_relationship_condition_reader_qualification as run;"
        "run(scoring_request_path=pathlib.Path(sys.argv[2]),"
        "output_root=pathlib.Path(sys.argv[3]))"
    )
    import_roots_json = canonical_json_bytes([str(path) for path in spec.import_binding.import_roots]).decode("utf-8")
    return [
        str(spec.python_executable),
        "-P",
        "-S",
        "-B",
        "-u",
        "-X",
        "utf8",
        "-X",
        f"pycache_prefix={spec.pycache_prefix}",
        "-c",
        bootstrap,
        import_roots_json,
        str(spec.scoring_request_path.resolve()),
        str(spec.output_root.resolve()),
    ]


def _expected_scorer_attestation_argv(spec: _ScorerLaunchSpec) -> list[str]:
    return [
        "-c",
        canonical_json_bytes([str(path) for path in spec.import_binding.import_roots]).decode("utf-8"),
        str(spec.scoring_request_path.resolve()),
        str(spec.output_root.resolve()),
    ]


def _stream_capture_payload(capture: _BoundedStreamCapture) -> Mapping[str, object]:
    return {
        "raw_sha256": capture.raw_sha256,
        "total_bytes": capture.total_bytes,
        "retained_prefix_sha256": capture.retained_prefix_sha256,
        "retained_prefix_bytes": capture.retained_prefix_bytes,
        "prefix_truncated": capture.prefix_truncated,
    }


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
        path=source.resolve(),
        payload=payload,
        raw=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        file_identity=_file_identity(after),
    )


def _assert_loaded_artifacts_still_stable(
    artifacts: object,
) -> None:
    for loaded in artifacts:
        if not isinstance(loaded, _LoadedArtifact):
            raise TypeError("loaded artifact stability check received invalid value")
        current = loaded.path.stat(follow_symlinks=False)
        if _file_identity(current) != loaded.file_identity:
            raise ValueError(f"qualification artifact changed after verification: {loaded.path}")


def _write_artifact_create_only(
    path: pathlib.Path,
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    raw = _canonical_artifact_bytes(payload)
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x+b") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
        handle.seek(0)
        if handle.read() != raw:
            raise RuntimeError(f"same-descriptor artifact readback failed: {target}")
    with target.open("rb") as handle:
        if handle.read() != raw:
            raise RuntimeError(f"closed-reopen artifact readback failed: {target}")
    return {
        "artifact_id": payload["artifact_id"],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes": len(raw),
    }


def _artifact_receipt(loaded: _LoadedArtifact) -> Mapping[str, object]:
    return {
        "artifact_id": loaded.payload["artifact_id"],
        "raw_sha256": loaded.raw_sha256,
        "raw_bytes": len(loaded.raw),
    }


def _with_artifact_id(core: Mapping[str, object]) -> dict[str, object]:
    if "artifact_id" in core:
        raise ValueError("artifact core must not predefine artifact_id")
    return {**core, "artifact_id": _sha256_json(core)}


def _artifact_id(payload: Mapping[str, object]) -> str:
    return _sha256_json({key: value for key, value in payload.items() if key != "artifact_id"})


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


def _bounded_integer(value: object, maximum: int, field_name: str) -> int:
    parsed = _nonnegative_integer(value, field_name)
    if parsed > maximum:
        raise ValueError(f"{field_name} exceeds its frozen bound")
    return parsed


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
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


__all__ = [
    "RELATIONSHIP_READER_QUALIFICATION_AUTHORIZED_EXECUTION_MANIFEST_SCHEMA_VERSION",
    "RELATIONSHIP_READER_QUALIFICATION_SCORING_STAGE_MANIFEST_SCHEMA_VERSION",
    "execute_authorized_relationship_condition_reader_qualification_execution",
    "execute_relationship_condition_reader_qualification_scoring_stage",
]
