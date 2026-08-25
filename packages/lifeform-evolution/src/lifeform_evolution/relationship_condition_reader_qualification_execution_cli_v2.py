"""Model-free V2 qualification protocol, anchor, and execution CLI.

Invoke this composition root as::

    python -m lifeform_evolution.relationship_condition_reader_qualification_execution_cli_v2

Imports remain lazy: parser construction, help, protocol inspection, and public
anchor observation do not import the predictor, scorer, Torch, or
sentence-transformers.  ``record-anchor`` performs the frozen unauthenticated
GitHub observation itself and writes only the protocol-pre-registered receipt
path.  A later ``validate-anchor`` or ``execute`` call must independently
supply both the protocol ID and receipt artifact ID.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pathlib
import stat
import sys
from typing import Mapping


_MODULE_PATH = pathlib.Path(os.path.abspath(__file__))
_REPOSITORY_ROOT = _MODULE_PATH.parents[4]
_MAX_PROTOCOL_BYTES = 8_000_000
_MAX_ANCHOR_RECEIPT_BYTES = 2_000_000
_MAX_PREFLIGHT_PUBLICATION_REQUEST_BYTES = 2_000_000
_PREFLIGHT_PUBLICATION_REQUEST_PATH = pathlib.PurePosixPath("public/publication_request.json")
_PREFLIGHT_PUBLICATION_REQUEST_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-publication-request.v1"
)
_FORBIDDEN_MODEL_MODULE_PREFIXES = ("torch", "sentence_transformers")


def _is_reparse(value: os.stat_result) -> bool:
    return bool(os.name == "nt" and getattr(value, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _reject_reparse_components(path: pathlib.Path, label: str) -> None:
    candidate = pathlib.Path(path)
    while True:
        if os.path.lexists(candidate):
            value = os.lstat(candidate)
            if stat.S_ISLNK(value.st_mode) or _is_reparse(value):
                raise ValueError(f"{label} must not traverse a symlink or reparse point: {candidate}")
        parent = candidate.parent
        if parent == candidate:
            return
        candidate = parent


def _canonical_absolute_path(value: pathlib.Path, label: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    normalized = pathlib.Path(os.path.abspath(path))
    if os.path.normcase(str(path)) != os.path.normcase(str(normalized)):
        raise ValueError(f"{label} must be lexically canonical")
    _reject_reparse_components(normalized, label)
    return normalized


def _require_absent_canonical_directory_target(
    value: pathlib.Path,
    label: str,
) -> pathlib.Path:
    target = _canonical_absolute_path(value, label)
    if os.path.lexists(target):
        raise FileExistsError(f"{label} must remain absent while freezing")
    existing_ancestor = target.parent
    while not os.path.lexists(existing_ancestor):
        parent = existing_ancestor.parent
        if parent == existing_ancestor:
            raise FileNotFoundError(f"{label} has no existing canonical ancestor")
        existing_ancestor = parent
    if not existing_ancestor.is_dir():
        raise ValueError(f"{label} nearest existing ancestor must be a directory")
    resolved_ancestor = existing_ancestor.resolve(strict=True)
    if str(resolved_ancestor) != str(existing_ancestor):
        raise ValueError(f"{label} traverses a filesystem alias or case variant")
    return target


def _path_key(path: pathlib.Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _require_protocol_output_outside_frozen_source_roots(
    path: pathlib.Path,
    *,
    repository_runtime_coverage: Mapping[str, object],
) -> None:
    raw_roots = repository_runtime_coverage.get("source_roots")
    if not isinstance(raw_roots, list) or not raw_roots or not all(isinstance(value, str) for value in raw_roots):
        raise TypeError("repository runtime coverage source_roots must be a non-empty list of strings")
    output = _canonical_absolute_path(path, "V2 protocol output path")
    for index, relative in enumerate(raw_roots):
        pure = pathlib.PurePosixPath(relative)
        if pure.is_absolute() or pure.as_posix() != relative or any(part in {"", ".", ".."} for part in pure.parts):
            raise ValueError(f"repository runtime source root {index} is not canonical relative POSIX")
        source_root = pathlib.Path(os.path.abspath(_REPOSITORY_ROOT / pure))
        if output == source_root or source_root in output.parents:
            raise ValueError(
                "V2 protocol output must remain outside every frozen repository source root "
                "to avoid protocol-to-coverage self-reference"
            )


def _install_workspace_source_roots(
    repository_root: pathlib.Path = _REPOSITORY_ROOT,
) -> tuple[pathlib.Path, ...]:
    root = _canonical_absolute_path(repository_root, "repository root")
    _reject_reparse_components(_MODULE_PATH, "V2 qualification execution CLI")
    packages_root = root / "packages"
    _reject_reparse_components(packages_root, "workspace packages root")
    if not packages_root.is_dir():
        raise FileNotFoundError(f"workspace packages root is absent: {packages_root}")

    source_roots: list[pathlib.Path] = []
    with os.scandir(packages_root) as entries:
        for entry in entries:
            package_path = pathlib.Path(entry.path)
            value = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(value.st_mode) or _is_reparse(value):
                raise ValueError(f"workspace packages must not contain symlink or reparse entries: {package_path}")
            if not stat.S_ISDIR(value.st_mode):
                continue
            source_root = package_path / "src"
            if not os.path.lexists(source_root):
                continue
            _reject_reparse_components(source_root, "workspace source root")
            source_value = os.lstat(source_root)
            if not stat.S_ISDIR(source_value.st_mode):
                raise ValueError(f"workspace source root must be a directory: {source_root}")
            source_roots.append(source_root)

    source_roots.sort(key=lambda path: path.relative_to(root).as_posix().encode("utf-8"))
    if not source_roots:
        raise FileNotFoundError("workspace packages/*/src roots are absent")
    root_keys = [_path_key(path) for path in source_roots]
    if len(set(root_keys)) != len(root_keys):
        raise ValueError("workspace source roots collide after path normalization")

    source_key_set = set(root_keys)
    retained = [
        existing
        for existing in sys.path
        if isinstance(existing, str) and _path_key(pathlib.Path(existing or os.curdir)) not in source_key_set
    ]
    sys.path[:] = [str(path) for path in source_roots] + retained
    return tuple(source_roots)


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _sha256_digest(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("value must be a 64-character lowercase SHA-256 digest")
    return value


def _add_protocol_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--execution-protocol-path",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument("--expected-execution-protocol-id", required=True)


def _add_anchor_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--anchor-receipt-path",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument("--expected-anchor-receipt-artifact-id", required=True)
    parser.add_argument(
        "--expected-execution-root",
        type=pathlib.Path,
        required=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze and publicly anchor V2 relationship-condition reader "
            "qualification, or run an independently identified authorization"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser(
        "freeze-protocol",
        help="freeze exact preflight/source/BGE/runtime state without execution",
    )
    freeze.add_argument("--preflight-root", type=pathlib.Path, required=True)
    freeze.add_argument("--expected-qualification-protocol-id", required=True)
    freeze.add_argument("--bge-snapshot-root", type=pathlib.Path, required=True)
    freeze.add_argument(
        "--proposed-execution-root",
        type=pathlib.Path,
        required=True,
    )
    freeze.add_argument("--anchor-receipt-relative-path", required=True)
    freeze.add_argument("--protocol-output-path", type=pathlib.Path, required=True)

    show = commands.add_parser(
        "show-protocol",
        help="validate and summarize an exact frozen V2 protocol",
    )
    _add_protocol_identity_arguments(show)

    record = commands.add_parser(
        "record-anchor",
        help="live-observe an unauthenticated, sole-revision public Gist",
    )
    _add_protocol_identity_arguments(record)
    record.add_argument("--gist-id", required=True)
    record.add_argument(
        "--expected-execution-root",
        type=pathlib.Path,
        required=True,
    )
    record.add_argument(
        "--timeout-seconds",
        type=_positive_integer,
        default=30,
    )

    validate = commands.add_parser(
        "validate-anchor",
        help="validate a receipt using an independently supplied artifact ID",
    )
    _add_protocol_identity_arguments(validate)
    _add_anchor_identity_arguments(validate)

    execute = commands.add_parser(
        "execute",
        help="run an externally identified V2 protocol and anchor receipt",
    )
    _add_protocol_identity_arguments(execute)
    _add_anchor_identity_arguments(execute)
    execute.add_argument("--preflight-root", type=pathlib.Path, required=True)
    execute.add_argument("--bge-snapshot-root", type=pathlib.Path, required=True)
    execute.add_argument(
        "--run-nonce",
        type=_sha256_digest,
        required=True,
        help="externally fixed 64-character lowercase SHA-256 digest",
    )
    execute.add_argument("--python-executable", type=pathlib.Path)
    execute.add_argument(
        "--prediction-timeout-seconds",
        type=_positive_integer,
        default=7_200,
    )
    execute.add_argument(
        "--scorer-timeout-seconds",
        type=_positive_integer,
        default=600,
    )
    return parser


def _load_v1_protocol_module() -> object:
    return importlib.import_module("lifeform_evolution.relationship_condition_reader_qualification_execution_protocol")


def _load_v2_protocol_module() -> object:
    return importlib.import_module(
        "lifeform_evolution.relationship_condition_reader_qualification_execution_protocol_v2"
    )


def _load_v2_execution_module() -> object:
    return importlib.import_module("lifeform_evolution.relationship_condition_reader_qualification_execution_v2")


def _canonical_json_api() -> tuple[object, object]:
    module = importlib.import_module("volvence_zero.canonical_json")
    return module.canonical_json_bytes, module.strict_json_loads


def _assert_model_free() -> None:
    imported = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in _FORBIDDEN_MODEL_MODULE_PREFIXES)
    )
    if imported:
        raise RuntimeError(f"V2 CLI imported forbidden model module: {imported[0]}")


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_nlink


def _read_stable_regular_file(
    path: pathlib.Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    candidate = _canonical_absolute_path(path, label)
    if not os.path.lexists(candidate):
        raise FileNotFoundError(f"{label} is absent: {candidate}")
    before = os.lstat(candidate)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    if before.st_size > max_bytes:
        raise ValueError(f"{label} exceeds its byte bound")
    with candidate.open("rb") as handle:
        during = os.fstat(handle.fileno())
        raw = handle.read(max_bytes + 1)
    after = os.lstat(candidate)
    if not (_file_identity(before) == _file_identity(during) == _file_identity(after)):
        raise ValueError(f"{label} changed identity while being read")
    if len(raw) > max_bytes or len(raw) != before.st_size:
        raise ValueError(f"{label} changed size while being read")
    return raw


def _load_canonical_json_object(
    path: pathlib.Path,
    *,
    label: str,
    max_bytes: int,
) -> tuple[Mapping[str, object], bytes]:
    raw = _read_stable_regular_file(path, label=label, max_bytes=max_bytes)
    canonical_json_bytes, strict_json_loads = _canonical_json_api()
    parsed = strict_json_loads(raw, max_bytes=max_bytes)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain a JSON object")
    if raw != canonical_json_bytes(parsed) + b"\n":
        raise ValueError(f"{label} must be canonical LF-terminated JSON")
    return parsed, raw


def _require_preflight_publication_execution_root(
    *,
    protocol_module: object,
    preflight_root: pathlib.Path,
    preflight_binding: Mapping[str, object],
    proposed_execution_root: pathlib.Path,
) -> None:
    """Reject a protocol proposal that its frozen publication request cannot execute."""

    root = _canonical_absolute_path(preflight_root, "preflight root")
    canonicalize_execution_root = (
        protocol_module.canonical_relationship_condition_reader_qualification_execution_root_v2
    )
    requested_root_text = canonicalize_execution_root(
        proposed_execution_root,
        "proposed execution root",
    )
    _require_absent_canonical_directory_target(
        pathlib.Path(requested_root_text),
        "proposed execution root",
    )
    publication_path = root / _PREFLIGHT_PUBLICATION_REQUEST_PATH
    publication, raw = _load_canonical_json_object(
        publication_path,
        label="preflight publication request",
        max_bytes=_MAX_PREFLIGHT_PUBLICATION_REQUEST_BYTES,
    )
    if publication.get("schema_version") != _PREFLIGHT_PUBLICATION_REQUEST_SCHEMA_VERSION:
        raise ValueError("preflight publication request schema drifted")

    raw_files = preflight_binding.get("files")
    if not isinstance(raw_files, list):
        raise TypeError("preflight binding files must be an array")
    bound_rows = [
        row
        for row in raw_files
        if isinstance(row, Mapping)
        and row.get("path") == _PREFLIGHT_PUBLICATION_REQUEST_PATH.as_posix()
    ]
    if len(bound_rows) != 1:
        raise ValueError("preflight binding must contain exactly one publication request")
    bound = bound_rows[0]
    if (
        bound.get("raw_sha256") != hashlib.sha256(raw).hexdigest()
        or bound.get("raw_bytes") != len(raw)
        or bound.get("artifact_id") != publication.get("artifact_id")
    ):
        raise ValueError("preflight publication request differs from its frozen binding")

    proposed_text = publication.get("proposed_execution_root")
    if not isinstance(proposed_text, str) or not proposed_text:
        raise ValueError("preflight publication request proposed execution root is malformed")
    preflight_execution_root_text = canonicalize_execution_root(
        pathlib.Path(proposed_text),
        "preflight publication request proposed execution root",
    )
    if preflight_execution_root_text != requested_root_text:
        raise ValueError(
            "preflight publication request proposed execution root differs from the "
            "execution protocol proposal"
        )
    if publication.get("proposed_execution_root_exists_at_prepare") is not False:
        raise ValueError("preflight publication request did not attest an absent execution root")


def _write_create_only_canonical_json(
    path: pathlib.Path,
    payload: Mapping[str, object],
) -> bytes:
    target = _canonical_absolute_path(path, "create-only output path")
    parent = target.parent
    _reject_reparse_components(parent, "create-only output parent")
    if not parent.is_dir():
        raise FileNotFoundError(f"create-only output parent is absent: {parent}")
    if os.path.lexists(target):
        raise FileExistsError(f"create-only output already exists: {target}")
    canonical_json_bytes, _strict_json_loads = _canonical_json_api()
    raw = canonical_json_bytes(dict(payload)) + b"\n"
    with target.open("x+b") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
        handle.seek(0)
        if handle.read() != raw:
            raise RuntimeError(f"create-only same-descriptor readback failed: {target}")
    reopened = _read_stable_regular_file(
        target,
        label="create-only output",
        max_bytes=max(len(raw), 1),
    )
    if reopened != raw:
        raise RuntimeError(f"create-only close/reopen readback failed: {target}")
    return raw


def _load_execution_protocol(
    *,
    protocol_module: object,
    path: pathlib.Path,
    expected_protocol_id: str,
) -> tuple[Mapping[str, object], bytes, pathlib.Path]:
    protocol_path = _canonical_absolute_path(path, "V2 execution protocol path")
    protocol, raw = protocol_module.load_relationship_condition_reader_qualification_execution_protocol_v2(
        protocol_path,
        expected_protocol_id=expected_protocol_id,
    )
    return protocol, raw, protocol_path


def _bound_anchor_receipt_path(
    *,
    protocol: Mapping[str, object],
    supplied_path: pathlib.Path | None,
) -> pathlib.Path:
    anchor = protocol.get("external_public_anchor")
    if not isinstance(anchor, Mapping) or not isinstance(anchor.get("receipt_path"), str):
        raise ValueError("V2 protocol external anchor receipt path is malformed")
    expected = _REPOSITORY_ROOT / pathlib.PurePosixPath(str(anchor["receipt_path"]))
    expected = _canonical_absolute_path(expected, "pre-registered anchor receipt path")
    if supplied_path is not None:
        supplied = _canonical_absolute_path(supplied_path, "anchor receipt path")
        if _path_key(supplied) != _path_key(expected):
            raise ValueError("anchor receipt path does not match the protocol-pre-registered path")
    return expected


def _protocol_summary(
    *,
    command: str,
    protocol: Mapping[str, object],
    protocol_raw: bytes,
    protocol_id: str,
    protocol_path: pathlib.Path,
) -> Mapping[str, object]:
    preflight = protocol.get("qualification_preflight")
    source = protocol.get("execution_source_tree")
    bge = protocol.get("bge_snapshot_tree")
    runtime = protocol.get("runtime_identity")
    anchor = protocol.get("external_public_anchor")
    retired = protocol.get("retired_predecessor")
    if not all(isinstance(value, Mapping) for value in (preflight, source, bge, runtime, anchor, retired)):
        raise ValueError("V2 protocol summary inputs are malformed")
    return {
        "schema_version": ("relationship-condition-reader-qualification-execution-cli-summary.v2"),
        "command": command,
        "execution_protocol_id": protocol_id,
        "protocol_path": str(protocol_path),
        "protocol_raw_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "protocol_raw_bytes": len(protocol_raw),
        "qualification_protocol_id": preflight["qualification_protocol_id"],
        "preflight_binding_artifact_id": preflight["artifact_id"],
        "source_tree_artifact_id": source["artifact_id"],
        "source_tree_entry_count": source["entry_count"],
        "bge_snapshot_tree_artifact_id": bge["artifact_id"],
        "bge_snapshot_entry_count": bge["entry_count"],
        "runtime_identity_artifact_id": runtime["artifact_id"],
        "proposed_execution_root": protocol["proposed_execution_root"],
        "anchor_receipt_relative_path": anchor["receipt_path"],
        "retired_predecessor_execution_protocol_id": retired["execution_protocol_id"],
        "retired_predecessor_can_authorize_v2": False,
        "external_public_anchor_validated": False,
        "qualification_execution_authorized": False,
        "model_or_cuda_execution_used": False,
        "windows_directory_entry_durability_attested": False,
    }


def _freeze_protocol(args: argparse.Namespace) -> Mapping[str, object]:
    _assert_model_free()
    v1_protocol = _load_v1_protocol_module()
    v2_protocol = _load_v2_protocol_module()
    preflight_root = _canonical_absolute_path(args.preflight_root, "preflight root")
    bge_root = _canonical_absolute_path(args.bge_snapshot_root, "BGE snapshot root")
    execution_root = _canonical_absolute_path(
        args.proposed_execution_root,
        "proposed execution root",
    )
    output_path = _canonical_absolute_path(
        args.protocol_output_path,
        "V2 protocol output path",
    )
    preflight = v1_protocol.build_relationship_condition_reader_execution_preflight_binding(
        preflight_root=preflight_root,
        expected_qualification_protocol_id=args.expected_qualification_protocol_id,
    )
    _require_preflight_publication_execution_root(
        protocol_module=v2_protocol,
        preflight_root=preflight_root,
        preflight_binding=preflight,
        proposed_execution_root=execution_root,
    )
    source = v1_protocol.build_relationship_condition_reader_execution_source_tree_manifest(
        repository_root=_REPOSITORY_ROOT,
    )
    repository_runtime_coverage = v2_protocol.build_relationship_condition_reader_repository_runtime_coverage(
        repository_root=_REPOSITORY_ROOT,
        execution_source_tree=source,
    )
    _require_protocol_output_outside_frozen_source_roots(
        output_path,
        repository_runtime_coverage=repository_runtime_coverage,
    )
    bge = v1_protocol.build_bge_m3_snapshot_tree_manifest(snapshot_root=bge_root)
    runtime = v1_protocol.build_relationship_condition_reader_qualification_runtime_identity()
    protocol = v2_protocol.build_relationship_condition_reader_qualification_execution_protocol_v2(
        preflight_binding=preflight,
        source_tree_manifest=source,
        repository_runtime_coverage=repository_runtime_coverage,
        bge_snapshot_tree_manifest=bge,
        runtime_identity=runtime,
        proposed_execution_root=execution_root,
        anchor_receipt_relative_path=args.anchor_receipt_relative_path,
    )
    protocol_id = v2_protocol.relationship_condition_reader_qualification_execution_protocol_id_v2(protocol)
    v2_protocol.validate_relationship_condition_reader_qualification_execution_protocol_v2(
        protocol,
        expected_protocol_id=protocol_id,
    )
    raw = _write_create_only_canonical_json(output_path, protocol)
    _assert_model_free()
    return _protocol_summary(
        command="freeze-protocol",
        protocol=protocol,
        protocol_raw=raw,
        protocol_id=protocol_id,
        protocol_path=output_path,
    )


def _show_protocol(args: argparse.Namespace) -> Mapping[str, object]:
    protocol_module = _load_v2_protocol_module()
    protocol, raw, path = _load_execution_protocol(
        protocol_module=protocol_module,
        path=args.execution_protocol_path,
        expected_protocol_id=args.expected_execution_protocol_id,
    )
    return _protocol_summary(
        command="show-protocol",
        protocol=protocol,
        protocol_raw=raw,
        protocol_id=args.expected_execution_protocol_id,
        protocol_path=path,
    )


def _record_anchor(args: argparse.Namespace) -> Mapping[str, object]:
    _assert_model_free()
    protocol_module = _load_v2_protocol_module()
    protocol, protocol_raw, protocol_path = _load_execution_protocol(
        protocol_module=protocol_module,
        path=args.execution_protocol_path,
        expected_protocol_id=args.expected_execution_protocol_id,
    )
    execution_root_text = (
        protocol_module.canonical_relationship_condition_reader_qualification_execution_root_v2(
            args.expected_execution_root,
            "expected execution root",
        )
    )
    execution_root = _require_absent_canonical_directory_target(
        pathlib.Path(execution_root_text),
        "expected execution root",
    )
    receipt_path = _bound_anchor_receipt_path(protocol=protocol, supplied_path=None)
    if os.path.lexists(receipt_path):
        raise FileExistsError(f"V2 anchor receipt already exists: {receipt_path}")
    receipt = protocol_module.observe_relationship_condition_reader_qualification_public_anchor_v2(
        execution_protocol_payload=protocol,
        execution_protocol_raw=protocol_raw,
        expected_execution_protocol_id=args.expected_execution_protocol_id,
        expected_execution_root=execution_root,
        gist_id=args.gist_id,
        timeout_seconds=args.timeout_seconds,
    )
    receipt_id = protocol_module.validate_relationship_condition_reader_qualification_public_anchor_receipt_v2(
        receipt,
        expected_receipt_artifact_id=receipt["artifact_id"],
        execution_protocol_payload=protocol,
        execution_protocol_raw=protocol_raw,
        expected_execution_protocol_id=args.expected_execution_protocol_id,
        expected_execution_root=execution_root,
    )
    receipt_raw = _write_create_only_canonical_json(receipt_path, receipt)
    _assert_model_free()
    return {
        "schema_version": ("relationship-condition-reader-qualification-anchor-record-summary.v2"),
        "command": "record-anchor",
        "execution_protocol_id": args.expected_execution_protocol_id,
        "execution_protocol_path": str(protocol_path),
        "anchor_receipt_path": str(receipt_path),
        "anchor_receipt_artifact_id": receipt_id,
        "anchor_receipt_raw_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "anchor_receipt_raw_bytes": len(receipt_raw),
        "github_reobservation_performed_by_cli": True,
        "fixed_unauthenticated_github_transport_used": True,
        "caller_supplied_github_observation_metadata_used": False,
        "qualification_execution_authorized": False,
        "model_or_cuda_execution_used": False,
        "windows_directory_entry_durability_attested": False,
        "next_required_external_input": (
            "independently copy anchor_receipt_artifact_id into --expected-anchor-receipt-artifact-id"
        ),
    }


def _load_and_validate_anchor(
    args: argparse.Namespace,
) -> tuple[
    Mapping[str, object],
    bytes,
    pathlib.Path,
    Mapping[str, object],
    bytes,
    pathlib.Path,
    pathlib.Path,
]:
    protocol_module = _load_v2_protocol_module()
    protocol, protocol_raw, protocol_path = _load_execution_protocol(
        protocol_module=protocol_module,
        path=args.execution_protocol_path,
        expected_protocol_id=args.expected_execution_protocol_id,
    )
    receipt_path = _bound_anchor_receipt_path(
        protocol=protocol,
        supplied_path=args.anchor_receipt_path,
    )
    receipt, receipt_raw = _load_canonical_json_object(
        receipt_path,
        label="V2 public anchor receipt",
        max_bytes=_MAX_ANCHOR_RECEIPT_BYTES,
    )
    execution_root_text = (
        protocol_module.canonical_relationship_condition_reader_qualification_execution_root_v2(
            args.expected_execution_root,
            "expected execution root",
        )
    )
    execution_root = _require_absent_canonical_directory_target(
        pathlib.Path(execution_root_text),
        "expected execution root",
    )
    protocol_module.validate_relationship_condition_reader_qualification_public_anchor_receipt_v2(
        receipt,
        expected_receipt_artifact_id=args.expected_anchor_receipt_artifact_id,
        execution_protocol_payload=protocol,
        execution_protocol_raw=protocol_raw,
        expected_execution_protocol_id=args.expected_execution_protocol_id,
        expected_execution_root=execution_root,
    )
    return (
        protocol,
        protocol_raw,
        protocol_path,
        receipt,
        receipt_raw,
        receipt_path,
        execution_root,
    )


def _validate_anchor(args: argparse.Namespace) -> Mapping[str, object]:
    (
        _protocol,
        protocol_raw,
        protocol_path,
        _receipt,
        receipt_raw,
        receipt_path,
        execution_root,
    ) = _load_and_validate_anchor(args)
    return {
        "schema_version": ("relationship-condition-reader-qualification-anchor-validation-summary.v2"),
        "command": "validate-anchor",
        "execution_protocol_id": args.expected_execution_protocol_id,
        "execution_protocol_path": str(protocol_path),
        "protocol_raw_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "protocol_raw_bytes": len(protocol_raw),
        "anchor_receipt_artifact_id": args.expected_anchor_receipt_artifact_id,
        "anchor_receipt_path": str(receipt_path),
        "anchor_receipt_raw_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "anchor_receipt_raw_bytes": len(receipt_raw),
        "expected_execution_root": str(execution_root),
        "public_anchor_receipt_contract_validated": True,
        "github_reobservation_performed_by_this_command": False,
        "qualification_execution_performed": False,
        "model_or_cuda_execution_used": False,
    }


def _execute(args: argparse.Namespace) -> Mapping[str, object]:
    (
        protocol,
        protocol_raw,
        _protocol_path,
        receipt,
        _receipt_raw,
        _receipt_path,
        execution_root,
    ) = _load_and_validate_anchor(args)
    preflight_root = _canonical_absolute_path(args.preflight_root, "preflight root")
    bge_root = _canonical_absolute_path(args.bge_snapshot_root, "BGE snapshot root")
    python_executable = None
    if args.python_executable is not None:
        python_executable = _canonical_absolute_path(
            args.python_executable,
            "Python executable",
        )
    execution_module = _load_v2_execution_module()
    return execution_module.execute_authorized_relationship_condition_reader_qualification_execution_v2(
        execution_protocol_payload=protocol,
        execution_protocol_raw=protocol_raw,
        expected_execution_protocol_id=args.expected_execution_protocol_id,
        public_anchor_receipt_payload=receipt,
        expected_public_anchor_receipt_artifact_id=(args.expected_anchor_receipt_artifact_id),
        repository_root=_REPOSITORY_ROOT,
        preflight_root=preflight_root,
        bge_snapshot_root=bge_root,
        execution_root=execution_root,
        run_nonce=args.run_nonce,
        python_executable=python_executable,
        prediction_timeout_seconds=args.prediction_timeout_seconds,
        scorer_timeout_seconds=args.scorer_timeout_seconds,
    )


def _dispatch(args: argparse.Namespace) -> Mapping[str, object]:
    if args.command == "freeze-protocol":
        return _freeze_protocol(args)
    if args.command == "show-protocol":
        return _show_protocol(args)
    if args.command == "record-anchor":
        return _record_anchor(args)
    if args.command == "validate-anchor":
        return _validate_anchor(args)
    if args.command == "execute":
        return _execute(args)
    raise AssertionError(f"unreachable V2 command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    _install_workspace_source_roots()
    args = _build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    payload = _dispatch(args)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
