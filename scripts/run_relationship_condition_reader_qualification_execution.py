#!/usr/bin/env python3
"""Freeze, anchor, inspect, and run reader qualification execution.

The command surface is deliberately a lazy-import composition root.  Merely
parsing arguments or requesting help does not import the predictor, scorer,
Torch, sentence-transformers, or any other workspace module.
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


_SCRIPT_PATH = pathlib.Path(os.path.abspath(__file__))
_REPOSITORY_ROOT = _SCRIPT_PATH.parents[1]
_MAX_PROTOCOL_BYTES = 8_000_000
_MAX_ANCHOR_RECEIPT_BYTES = 2_000_000
_FORBIDDEN_FREEZE_MODULE_PREFIXES = ("torch", "sentence_transformers")


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


def _path_key(path: pathlib.Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _install_workspace_source_roots(
    repository_root: pathlib.Path = _REPOSITORY_ROOT,
) -> tuple[pathlib.Path, ...]:
    root = _canonical_absolute_path(repository_root, "repository root")
    _reject_reparse_components(root, "repository root")
    _reject_reparse_components(_SCRIPT_PATH, "qualification execution CLI")
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

    retained: list[str] = []
    source_key_set = set(root_keys)
    for existing in sys.path:
        if not isinstance(existing, str):
            continue
        if _path_key(pathlib.Path(existing or os.curdir)) not in source_key_set:
            retained.append(existing)
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
            "Freeze and externally anchor the exact relationship-condition "
            "reader qualification execution, or run the already-authorized "
            "protocol"
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
    freeze.add_argument(
        "--protocol-output-path",
        type=pathlib.Path,
        required=True,
    )

    show = commands.add_parser(
        "show-protocol",
        help="validate and summarize an exact frozen protocol",
    )
    _add_protocol_identity_arguments(show)

    record = commands.add_parser(
        "record-anchor",
        help="record an externally observed first-revision public Gist anchor",
    )
    _add_protocol_identity_arguments(record)
    record.add_argument(
        "--observed-protocol-raw-path",
        type=pathlib.Path,
        required=True,
    )
    record.add_argument("--gist-id", required=True)
    record.add_argument("--history-version", required=True)
    record.add_argument("--created-at", required=True)
    record.add_argument("--updated-at", required=True)
    record.add_argument("--api-raw-url", required=True)
    record.add_argument("--revision-raw-url", required=True)
    record.add_argument("--observed-at-utc", required=True)
    record.add_argument(
        "--expected-execution-root",
        type=pathlib.Path,
        required=True,
    )

    validate = commands.add_parser(
        "validate-anchor",
        help="validate an exact anchor using an externally supplied receipt ID",
    )
    _add_protocol_identity_arguments(validate)
    _add_anchor_identity_arguments(validate)

    execute = commands.add_parser(
        "execute",
        help="run the externally identified and publicly anchored protocol",
    )
    _add_protocol_identity_arguments(execute)
    _add_anchor_identity_arguments(execute)
    execute.add_argument("--preflight-root", type=pathlib.Path, required=True)
    execute.add_argument("--bge-snapshot-root", type=pathlib.Path, required=True)
    execute.add_argument("--run-nonce", required=True)
    execute.add_argument("--python-executable", type=pathlib.Path)
    execute.add_argument(
        "--prediction-timeout-seconds",
        type=_positive_integer,
        default=7200,
    )
    execute.add_argument(
        "--scorer-timeout-seconds",
        type=_positive_integer,
        default=600,
    )
    return parser


def _load_protocol_module() -> object:
    return importlib.import_module("lifeform_evolution.relationship_condition_reader_qualification_execution_protocol")


def _load_execution_module() -> object:
    return importlib.import_module("lifeform_evolution.relationship_condition_reader_qualification_execution")


def _canonical_json_api() -> tuple[object, object]:
    module = importlib.import_module("volvence_zero.canonical_json")
    return module.canonical_json_bytes, module.strict_json_loads


def _assert_freeze_process_model_free() -> None:
    imported = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in _FORBIDDEN_FREEZE_MODULE_PREFIXES)
    )
    if imported:
        raise RuntimeError(f"protocol freeze process imported a forbidden model module: {imported[0]}")


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_nlink


def _read_stable_regular_file(
    path: pathlib.Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    candidate = _canonical_absolute_path(path, label)
    _reject_reparse_components(candidate, label)
    if not os.path.lexists(candidate):
        raise FileNotFoundError(f"{label} is absent: {candidate}")
    before = os.lstat(candidate)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if before.st_nlink != 1:
        raise ValueError(f"{label} must have exactly one hard link")
    if before.st_size > max_bytes:
        raise ValueError(f"{label} exceeds its frozen byte bound")
    with candidate.open("rb") as handle:
        during = os.fstat(handle.fileno())
        raw = handle.read(max_bytes + 1)
    after = os.lstat(candidate)
    if not (_file_identity(before) == _file_identity(during) == _file_identity(after)):
        raise ValueError(f"{label} changed identity while being read")
    if len(raw) > max_bytes or len(raw) != before.st_size:
        raise ValueError(f"{label} changed size or exceeds its frozen byte bound")
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


def _protocol_summary(
    *,
    command: str,
    protocol: Mapping[str, object],
    protocol_raw: bytes,
    protocol_id: str,
    protocol_path: pathlib.Path,
) -> dict[str, object]:
    preflight = protocol["qualification_preflight"]
    source = protocol["execution_source_tree"]
    bge = protocol["bge_snapshot_tree"]
    runtime = protocol["runtime_identity"]
    anchor = protocol["external_public_anchor"]
    if not all(isinstance(value, Mapping) for value in (preflight, source, bge, runtime, anchor)):
        raise ValueError("execution protocol summary inputs are malformed")
    return {
        "schema_version": ("relationship-condition-reader-qualification-execution-cli-summary.v1"),
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
        "external_public_anchor_validated": False,
        "qualification_execution_authorized": False,
        "model_or_cuda_execution_used": False,
        "windows_directory_entry_durability_attested": False,
    }


def _load_execution_protocol(
    *,
    protocol_module: object,
    path: pathlib.Path,
    expected_protocol_id: str,
) -> tuple[Mapping[str, object], bytes, pathlib.Path]:
    protocol_path = _canonical_absolute_path(path, "execution protocol path")
    protocol, raw = protocol_module.load_relationship_condition_reader_qualification_execution_protocol(
        protocol_path,
        expected_protocol_id=expected_protocol_id,
    )
    canonical_json_bytes, _strict_json_loads = _canonical_json_api()
    if raw != canonical_json_bytes(dict(protocol)) + b"\n":
        raise ValueError("execution protocol must be canonical LF-terminated JSON")
    return protocol, raw, protocol_path


def _bound_anchor_receipt_path(
    *,
    protocol: Mapping[str, object],
    supplied_path: pathlib.Path,
) -> pathlib.Path:
    anchor = protocol.get("external_public_anchor")
    if not isinstance(anchor, Mapping):
        raise ValueError("execution protocol external_public_anchor is malformed")
    relative = anchor.get("receipt_path")
    if not isinstance(relative, str):
        raise ValueError("execution protocol anchor receipt path is malformed")
    expected = _REPOSITORY_ROOT / pathlib.PurePosixPath(relative)
    supplied = _canonical_absolute_path(supplied_path, "anchor receipt path")
    if _path_key(supplied) != _path_key(expected):
        raise ValueError("anchor receipt path does not match the protocol-pre-registered path")
    return supplied


def _freeze_protocol(args: argparse.Namespace) -> Mapping[str, object]:
    _assert_freeze_process_model_free()
    protocol_module = _load_protocol_module()
    preflight_root = _canonical_absolute_path(args.preflight_root, "preflight root")
    bge_root = _canonical_absolute_path(args.bge_snapshot_root, "BGE snapshot root")
    execution_root = _canonical_absolute_path(
        args.proposed_execution_root,
        "proposed execution root",
    )
    output_path = _canonical_absolute_path(
        args.protocol_output_path,
        "protocol output path",
    )
    preflight = protocol_module.build_relationship_condition_reader_execution_preflight_binding(
        preflight_root=preflight_root,
        expected_qualification_protocol_id=(args.expected_qualification_protocol_id),
    )
    source = protocol_module.build_relationship_condition_reader_execution_source_tree_manifest(
        repository_root=_REPOSITORY_ROOT,
    )
    bge = protocol_module.build_bge_m3_snapshot_tree_manifest(
        snapshot_root=bge_root,
    )
    runtime = protocol_module.build_relationship_condition_reader_qualification_runtime_identity()
    protocol = protocol_module.build_relationship_condition_reader_qualification_execution_protocol(
        preflight_binding=preflight,
        source_tree_manifest=source,
        bge_snapshot_tree_manifest=bge,
        runtime_identity=runtime,
        proposed_execution_root=execution_root,
        anchor_receipt_relative_path=args.anchor_receipt_relative_path,
    )
    protocol_id = protocol_module.relationship_condition_reader_qualification_execution_protocol_id(protocol)
    protocol_module.validate_relationship_condition_reader_qualification_execution_protocol(
        protocol,
        expected_protocol_id=protocol_id,
    )
    protocol_raw = _write_create_only_canonical_json(output_path, protocol)
    _assert_freeze_process_model_free()
    return _protocol_summary(
        command="freeze-protocol",
        protocol=protocol,
        protocol_raw=protocol_raw,
        protocol_id=protocol_id,
        protocol_path=output_path,
    )


def _show_protocol(args: argparse.Namespace) -> Mapping[str, object]:
    protocol_module = _load_protocol_module()
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
    protocol_module = _load_protocol_module()
    protocol, protocol_raw, protocol_path = _load_execution_protocol(
        protocol_module=protocol_module,
        path=args.execution_protocol_path,
        expected_protocol_id=args.expected_execution_protocol_id,
    )
    observed_path = _canonical_absolute_path(
        args.observed_protocol_raw_path,
        "externally observed protocol raw path",
    )
    if _path_key(observed_path) == _path_key(protocol_path):
        raise ValueError("externally observed protocol raw path must be distinct from the local protocol path")
    observed_raw = _read_stable_regular_file(
        observed_path,
        label="externally observed protocol raw",
        max_bytes=_MAX_PROTOCOL_BYTES,
    )
    if observed_raw != protocol_raw:
        raise ValueError("externally observed public protocol raw bytes do not match the local protocol")
    execution_root = _canonical_absolute_path(
        args.expected_execution_root,
        "expected execution root",
    )
    if os.path.lexists(execution_root):
        raise FileExistsError("anchor recording requires the proposed execution root to be absent")
    anchor = protocol.get("external_public_anchor")
    if not isinstance(anchor, Mapping):
        raise ValueError("execution protocol external anchor is malformed")
    gist_owner = anchor.get("gist_owner")
    filename = anchor.get("filename")
    if not isinstance(gist_owner, str) or not isinstance(filename, str):
        raise ValueError("execution protocol anchor owner or filename is malformed")
    receipt = protocol_module.build_relationship_condition_reader_qualification_public_anchor_receipt(
        execution_protocol_payload=protocol,
        execution_protocol_raw=protocol_raw,
        expected_execution_protocol_id=args.expected_execution_protocol_id,
        expected_execution_root=execution_root,
        gist_owner=gist_owner,
        gist_id=args.gist_id,
        gist_url=f"https://gist.github.com/{gist_owner}/{args.gist_id}",
        filename=filename,
        public=True,
        history_version=args.history_version,
        history_revision_count=1,
        first_revision=True,
        created_at=args.created_at,
        updated_at=args.updated_at,
        api_raw_url=args.api_raw_url,
        revision_raw_url=args.revision_raw_url,
        observation_transport=("unauthenticated_github_rest_api_and_raw_http"),
        observed_at_utc=args.observed_at_utc,
        observed_protocol_raw=observed_raw,
    )
    if not isinstance(anchor.get("receipt_path"), str):
        raise ValueError("execution protocol anchor receipt path is malformed")
    receipt_path = _bound_anchor_receipt_path(
        protocol=protocol,
        supplied_path=(_REPOSITORY_ROOT / pathlib.PurePosixPath(str(anchor["receipt_path"]))),
    )
    receipt_id = protocol_module.validate_relationship_condition_reader_qualification_public_anchor_receipt(
        receipt,
        expected_receipt_artifact_id=receipt["artifact_id"],
        execution_protocol_payload=protocol,
        execution_protocol_raw=protocol_raw,
        expected_execution_protocol_id=args.expected_execution_protocol_id,
        expected_execution_root=execution_root,
    )
    receipt_raw = _write_create_only_canonical_json(receipt_path, receipt)
    return {
        "schema_version": ("relationship-condition-reader-qualification-anchor-record-summary.v1"),
        "command": "record-anchor",
        "execution_protocol_id": args.expected_execution_protocol_id,
        "execution_protocol_path": str(protocol_path),
        "anchor_receipt_path": str(receipt_path),
        "anchor_receipt_artifact_id": receipt_id,
        "anchor_receipt_raw_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "anchor_receipt_raw_bytes": len(receipt_raw),
        "caller_supplied_github_observation_contract_validated": True,
        "github_reobservation_performed_by_cli": False,
        "external_public_anchor_independently_verified_by_cli": False,
        "qualification_execution_authorized": False,
        "model_or_cuda_execution_used": False,
        "windows_directory_entry_durability_attested": False,
        "next_required_external_input": (
            "copy anchor_receipt_artifact_id into an independent --expected-anchor-receipt-artifact-id argument"
        ),
    }


def _load_and_validate_anchor(
    args: argparse.Namespace,
) -> tuple[
    object,
    Mapping[str, object],
    bytes,
    pathlib.Path,
    Mapping[str, object],
    bytes,
    pathlib.Path,
    pathlib.Path,
]:
    protocol_module = _load_protocol_module()
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
        label="public anchor receipt",
        max_bytes=_MAX_ANCHOR_RECEIPT_BYTES,
    )
    execution_root = _canonical_absolute_path(
        args.expected_execution_root,
        "expected execution root",
    )
    protocol_module.validate_relationship_condition_reader_qualification_public_anchor_receipt(
        receipt,
        expected_receipt_artifact_id=(args.expected_anchor_receipt_artifact_id),
        execution_protocol_payload=protocol,
        execution_protocol_raw=protocol_raw,
        expected_execution_protocol_id=args.expected_execution_protocol_id,
        expected_execution_root=execution_root,
    )
    return (
        protocol_module,
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
        _protocol_module,
        _protocol,
        protocol_raw,
        protocol_path,
        _receipt,
        receipt_raw,
        receipt_path,
        execution_root,
    ) = _load_and_validate_anchor(args)
    return {
        "schema_version": ("relationship-condition-reader-qualification-anchor-validation-summary.v1"),
        "command": "validate-anchor",
        "execution_protocol_id": args.expected_execution_protocol_id,
        "execution_protocol_path": str(protocol_path),
        "protocol_raw_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "protocol_raw_bytes": len(protocol_raw),
        "anchor_receipt_artifact_id": (args.expected_anchor_receipt_artifact_id),
        "anchor_receipt_path": str(receipt_path),
        "anchor_receipt_raw_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "anchor_receipt_raw_bytes": len(receipt_raw),
        "expected_execution_root": str(execution_root),
        "public_anchor_receipt_contract_validated": True,
        "github_reobservation_performed_by_cli": False,
        "external_public_anchor_independently_verified_by_cli": False,
        "qualification_execution_performed": False,
        "model_or_cuda_execution_used": False,
    }


def _execute(args: argparse.Namespace) -> Mapping[str, object]:
    (
        _protocol_module,
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
    execution_module = _load_execution_module()
    return execution_module.execute_authorized_relationship_condition_reader_qualification_execution(
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
    raise AssertionError(f"unreachable command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    _install_workspace_source_roots()
    args = _build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    payload = _dispatch(args)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
