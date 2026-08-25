"""Complete repository import-root coverage for reader qualification children.

The historical execution source manifest intentionally freezes Python source
only.  This module adds a separate, content-addressed runtime boundary over the
package source roots derived from that manifest.  It does not mutate or widen
the historical manifest schema.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import stat
from typing import Mapping

from lifeform_evolution.relationship_condition_reader_qualification_execution_protocol import (
    RELATIONSHIP_READER_EXECUTION_SOURCE_TREE_SCHEMA_VERSION,
    validate_relationship_condition_reader_execution_source_tree_manifest,
)
from volvence_zero.canonical_json import canonical_json_bytes


RELATIONSHIP_READER_REPOSITORY_RUNTIME_COVERAGE_SCHEMA_VERSION = (
    "relationship-condition-reader-qualification-repository-runtime-coverage.v1"
)

_PATH_CONTRACT = "repo_relative_posix_utf8_ordinal_casefold_unique.v1"
_CONTENT_CONTRACT = "exact_raw_bytes_no_eol_normalization.v1"
_LINK_CONTRACT = "directories_and_files_non_symlink_non_reparse_files_nlink_one.v1"
_EXCLUDED_CACHE_DIRECTORY_NAME = "__pycache__"
_FORBIDDEN_NATIVE_SUFFIXES = (".dll", ".pyd")
_BYTECODE_SUFFIX = ".pyc"
_PYTHON_SOURCE_SUFFIX = ".py"
_HASH_CHUNK_BYTES = 16 * 1024 * 1024


def build_relationship_condition_reader_repository_runtime_coverage(
    *,
    repository_root: pathlib.Path,
    execution_source_tree: Mapping[str, object],
) -> Mapping[str, object]:
    """Observe every regular file that can accompany frozen repository source.

    Every ``packages/*/src`` root is derived solely from the supplied, already
    content-addressed execution source tree.  Python bytecode is excluded only
    below a validated ``__pycache__`` directory.  Direct bytecode and repository
    native modules are refused before any child can be launched.
    """

    root = _canonical_directory(repository_root, "repository root")
    source_tree = _mapping(execution_source_tree, "execution source tree")
    source_tree_artifact_id = validate_relationship_condition_reader_execution_source_tree_manifest(source_tree)
    frozen_entries = _execution_source_entries(source_tree)
    source_root_relatives = _derive_source_root_relatives(frozen_entries)

    observed_paths: list[pathlib.Path] = []
    excluded_cache_directory_count = 0
    excluded_bytecode_file_count = 0
    for relative_root in source_root_relatives:
        source_root = root / pathlib.PurePosixPath(relative_root)
        _assert_directory_without_reparse(
            source_root,
            root=root,
            field_name=f"repository source root {relative_root}",
        )
        for current, dirnames, filenames in os.walk(
            source_root,
            topdown=True,
            onerror=_raise_walk_error,
            followlinks=False,
        ):
            current_path = pathlib.Path(current)
            _assert_directory_without_reparse(
                current_path,
                root=root,
                field_name="repository source directory",
            )
            for dirname in sorted(dirnames, key=lambda value: value.encode("utf-8")):
                candidate = current_path / dirname
                _assert_directory_without_reparse(
                    candidate,
                    root=root,
                    field_name="repository source directory",
                )
                if dirname.casefold() == _EXCLUDED_CACHE_DIRECTORY_NAME.casefold():
                    excluded_cache_directory_count += 1
            dirnames[:] = sorted(dirnames, key=lambda value: value.encode("utf-8"))

            for filename in sorted(filenames, key=lambda value: value.encode("utf-8")):
                path = current_path / filename
                relative = _relative_to_root(path, root=root, field_name="repository runtime file").as_posix()
                suffix = pathlib.PurePosixPath(relative).suffix.casefold()
                _validate_regular_file_before_read(
                    path,
                    root=root,
                    field_name=f"repository runtime file {relative}",
                )
                if suffix in _FORBIDDEN_NATIVE_SUFFIXES:
                    raise ValueError(f"repository runtime coverage refuses native shadow file: {relative}")
                if suffix == _BYTECODE_SUFFIX:
                    parts = pathlib.PurePosixPath(relative).parts
                    if not any(part.casefold() == _EXCLUDED_CACHE_DIRECTORY_NAME.casefold() for part in parts[:-1]):
                        raise ValueError(
                            f"repository runtime coverage refuses bytecode outside __pycache__: {relative}"
                        )
                    excluded_bytecode_file_count += 1
                    continue
                observed_paths.append(path)

    entries = _file_entries(root=root, paths=observed_paths)
    frozen_python_entries = _package_python_entries(frozen_entries, source_root_relatives=source_root_relatives)
    observed_python_entries = [
        row
        for row in entries
        if pathlib.PurePosixPath(_text(row["path"], "repository runtime entry path")).suffix.casefold()
        == _PYTHON_SOURCE_SUFFIX
    ]
    if observed_python_entries != frozen_python_entries:
        raise ValueError("frozen Python source entries do not exactly match complete repository runtime coverage")

    python_join_sha256 = hashlib.sha256(canonical_json_bytes({"entries": frozen_python_entries})).hexdigest()
    core = {
        "schema_version": RELATIONSHIP_READER_REPOSITORY_RUNTIME_COVERAGE_SCHEMA_VERSION,
        "repository_root": str(root),
        "execution_source_tree_schema_version": source_tree["schema_version"],
        "execution_source_tree_artifact_id": source_tree_artifact_id,
        "execution_source_tree_entry_count": source_tree["entry_count"],
        "path_contract": _PATH_CONTRACT,
        "content_contract": _CONTENT_CONTRACT,
        "link_contract": _LINK_CONTRACT,
        "excluded_cache_directory_names": [_EXCLUDED_CACHE_DIRECTORY_NAME],
        "forbidden_native_suffixes": list(_FORBIDDEN_NATIVE_SUFFIXES),
        "source_roots": list(source_root_relatives),
        "source_root_count": len(source_root_relatives),
        "entries": entries,
        "entry_count": len(entries),
        "total_raw_bytes": sum(_nonnegative_integer(row["raw_bytes"], "repository runtime bytes") for row in entries),
        "excluded_cache_directory_count": excluded_cache_directory_count,
        "excluded_bytecode_file_count": excluded_bytecode_file_count,
        "frozen_python_source_entry_count": len(frozen_python_entries),
        "frozen_python_source_join_sha256": python_join_sha256,
    }
    payload = _with_artifact_id(core)
    _validate_coverage_shape(payload)
    return payload


def validate_relationship_condition_reader_repository_runtime_coverage(
    payload: Mapping[str, object],
    *,
    repository_root: pathlib.Path | None = None,
    execution_source_tree: Mapping[str, object] | None = None,
) -> str:
    """Validate shape/content address and optionally reobserve the repository."""

    coverage = _mapping(payload, "repository runtime coverage")
    _validate_coverage_shape(coverage)
    if repository_root is not None and execution_source_tree is None:
        raise ValueError("execution_source_tree is required for repository reobservation")
    if execution_source_tree is not None:
        _validate_execution_source_tree_binding(coverage, execution_source_tree=execution_source_tree)
    if repository_root is not None:
        if execution_source_tree is None:  # pragma: no cover - guarded above
            raise RuntimeError("execution source tree guard failed")
        observed = build_relationship_condition_reader_repository_runtime_coverage(
            repository_root=repository_root,
            execution_source_tree=execution_source_tree,
        )
        if coverage != observed:
            raise ValueError("repository runtime coverage does not match the current exact source roots")
    return _digest(coverage["artifact_id"], "repository runtime coverage artifact_id")


def _validate_execution_source_tree_binding(
    coverage: Mapping[str, object],
    *,
    execution_source_tree: Mapping[str, object],
) -> None:
    source_tree = _mapping(execution_source_tree, "execution source tree")
    source_tree_artifact_id = validate_relationship_condition_reader_execution_source_tree_manifest(source_tree)
    expected_binding = {
        "execution_source_tree_schema_version": source_tree["schema_version"],
        "execution_source_tree_artifact_id": source_tree_artifact_id,
        "execution_source_tree_entry_count": source_tree["entry_count"],
    }
    if any(
        type(coverage[field]) is not type(value) or coverage[field] != value
        for field, value in expected_binding.items()
    ):
        raise ValueError("repository runtime coverage is bound to a different execution source tree")

    frozen_entries = _execution_source_entries(source_tree)
    source_roots = _derive_source_root_relatives(frozen_entries)
    if coverage["source_roots"] != list(source_roots):
        raise ValueError("repository runtime source roots differ from the bound execution source tree")
    frozen_python_entries = _package_python_entries(frozen_entries, source_root_relatives=source_roots)
    coverage_entries = coverage["entries"]
    if not isinstance(coverage_entries, list):  # pragma: no cover - strict shape already checked
        raise RuntimeError("validated repository runtime entries changed type")
    observed_python_entries = [
        dict(_mapping(row, "repository runtime entry"))
        for row in coverage_entries
        if pathlib.PurePosixPath(
            _text(_mapping(row, "repository runtime entry")["path"], "repository runtime path")
        ).suffix.casefold()
        == _PYTHON_SOURCE_SUFFIX
    ]
    if observed_python_entries != frozen_python_entries:
        raise ValueError("repository runtime Python entries differ from the bound execution source tree")


def _execution_source_entries(source_tree: Mapping[str, object]) -> list[dict[str, object]]:
    values = source_tree["entries"]
    if not isinstance(values, list):
        raise TypeError("execution source tree entries must be a list")
    entries: list[dict[str, object]] = []
    for index, value in enumerate(values):
        row = _mapping(value, f"execution source entry {index}")
        if set(row) != {"path", "raw_sha256", "raw_bytes"}:
            raise ValueError(f"execution source entry {index} fields drifted")
        entries.append(
            {
                "path": _relative_posix_path(row["path"], f"execution source path {index}"),
                "raw_sha256": _digest(row["raw_sha256"], f"execution source raw_sha256 {index}"),
                "raw_bytes": _nonnegative_integer(row["raw_bytes"], f"execution source raw_bytes {index}"),
            }
        )
    paths = [_text(row["path"], "execution source path") for row in entries]
    _validate_canonical_path_order(paths, field_name="execution source entries")
    return entries


def _derive_source_root_relatives(entries: list[dict[str, object]]) -> tuple[str, ...]:
    roots: set[str] = set()
    for row in entries:
        path = _text(row["path"], "execution source path")
        parts = pathlib.PurePosixPath(path).parts
        if (
            len(parts) >= 4
            and parts[0] == "packages"
            and parts[2] == "src"
            and pathlib.PurePosixPath(path).suffix == _PYTHON_SOURCE_SUFFIX
        ):
            roots.add(pathlib.PurePosixPath(*parts[:3]).as_posix())
    source_roots = tuple(sorted(roots, key=lambda value: value.encode("utf-8")))
    if not source_roots:
        raise ValueError("execution source tree contains no packages/*/src Python entries")
    if len({value.casefold() for value in source_roots}) != len(source_roots):
        raise ValueError("repository source roots contain a casefold collision")
    return source_roots


def _package_python_entries(
    entries: list[dict[str, object]],
    *,
    source_root_relatives: tuple[str, ...],
) -> list[dict[str, object]]:
    prefixes = tuple(f"{value}/" for value in source_root_relatives)
    return [
        dict(row)
        for row in entries
        if _text(row["path"], "execution source path").startswith(prefixes)
        and pathlib.PurePosixPath(_text(row["path"], "execution source path")).suffix == _PYTHON_SOURCE_SUFFIX
    ]


def _file_entries(*, root: pathlib.Path, paths: list[pathlib.Path]) -> list[dict[str, object]]:
    relative_and_path = [
        (
            _relative_to_root(path, root=root, field_name="repository runtime file").as_posix(),
            path,
        )
        for path in paths
    ]
    relative_and_path.sort(key=lambda pair: pair[0].encode("utf-8"))
    relative_paths = [relative for relative, _path in relative_and_path]
    _validate_canonical_path_order(relative_paths, field_name="repository runtime entries")
    entries: list[dict[str, object]] = []
    for relative, path in relative_and_path:
        raw_sha256, raw_bytes = _hash_stable_regular_file(
            path,
            root=root,
            field_name=f"repository runtime file {relative}",
        )
        entries.append(
            {
                "path": relative,
                "raw_sha256": raw_sha256,
                "raw_bytes": raw_bytes,
            }
        )
    return entries


def _validate_coverage_shape(payload: Mapping[str, object]) -> None:
    expected_keys = {
        "schema_version",
        "repository_root",
        "execution_source_tree_schema_version",
        "execution_source_tree_artifact_id",
        "execution_source_tree_entry_count",
        "path_contract",
        "content_contract",
        "link_contract",
        "excluded_cache_directory_names",
        "forbidden_native_suffixes",
        "source_roots",
        "source_root_count",
        "entries",
        "entry_count",
        "total_raw_bytes",
        "excluded_cache_directory_count",
        "excluded_bytecode_file_count",
        "frozen_python_source_entry_count",
        "frozen_python_source_join_sha256",
        "artifact_id",
    }
    if set(payload) != expected_keys:
        raise ValueError("repository runtime coverage fields drifted")
    required = {
        "schema_version": RELATIONSHIP_READER_REPOSITORY_RUNTIME_COVERAGE_SCHEMA_VERSION,
        "execution_source_tree_schema_version": RELATIONSHIP_READER_EXECUTION_SOURCE_TREE_SCHEMA_VERSION,
        "path_contract": _PATH_CONTRACT,
        "content_contract": _CONTENT_CONTRACT,
        "link_contract": _LINK_CONTRACT,
        "excluded_cache_directory_names": [_EXCLUDED_CACHE_DIRECTORY_NAME],
        "forbidden_native_suffixes": list(_FORBIDDEN_NATIVE_SUFFIXES),
    }
    if any(type(payload[field]) is not type(value) or payload[field] != value for field, value in required.items()):
        raise ValueError("repository runtime coverage contract drifted")

    root_text = _text(payload["repository_root"], "repository runtime repository_root")
    if not pathlib.Path(root_text).is_absolute():
        raise ValueError("repository runtime repository_root must be absolute")
    _digest(payload["execution_source_tree_artifact_id"], "execution source tree artifact_id")
    source_entry_count = _positive_integer(
        payload["execution_source_tree_entry_count"],
        "execution source tree entry_count",
    )

    raw_source_roots = payload["source_roots"]
    if not isinstance(raw_source_roots, list):
        raise TypeError("repository runtime source_roots must be a list")
    source_roots = [
        _source_root_relative(value, f"repository runtime source root {index}")
        for index, value in enumerate(raw_source_roots)
    ]
    _validate_canonical_path_order(source_roots, field_name="repository runtime source roots")
    if _positive_integer(payload["source_root_count"], "repository runtime source_root_count") != len(source_roots):
        raise ValueError("repository runtime source_root_count mismatch")

    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise TypeError("repository runtime entries must be a list")
    entries: list[dict[str, object]] = []
    for index, value in enumerate(raw_entries):
        row = _mapping(value, f"repository runtime entry {index}")
        if set(row) != {"path", "raw_sha256", "raw_bytes"}:
            raise ValueError(f"repository runtime entry {index} fields drifted")
        path = _relative_posix_path(row["path"], f"repository runtime entry path {index}")
        parts = pathlib.PurePosixPath(path).parts
        if (
            any(part.casefold() == _EXCLUDED_CACHE_DIRECTORY_NAME.casefold() for part in parts[:-1])
            and pathlib.PurePosixPath(path).suffix.casefold() == _BYTECODE_SUFFIX
        ):
            raise ValueError("repository runtime entries contain excluded cache bytecode")
        if pathlib.PurePosixPath(path).suffix.casefold() in {*_FORBIDDEN_NATIVE_SUFFIXES, _BYTECODE_SUFFIX}:
            raise ValueError("repository runtime entries contain a forbidden import shadow")
        if not any(path.startswith(f"{root}/") for root in source_roots):
            raise ValueError("repository runtime entry is outside its frozen source roots")
        entries.append(
            {
                "path": path,
                "raw_sha256": _digest(row["raw_sha256"], f"repository runtime raw_sha256 {index}"),
                "raw_bytes": _nonnegative_integer(row["raw_bytes"], f"repository runtime raw_bytes {index}"),
            }
        )
    paths = [_text(row["path"], "repository runtime path") for row in entries]
    _validate_canonical_path_order(paths, field_name="repository runtime entries")
    if _positive_integer(payload["entry_count"], "repository runtime entry_count") != len(entries):
        raise ValueError("repository runtime entry_count mismatch")
    total_raw_bytes = sum(_nonnegative_integer(row["raw_bytes"], "repository runtime bytes") for row in entries)
    if _nonnegative_integer(payload["total_raw_bytes"], "repository runtime total_raw_bytes") != total_raw_bytes:
        raise ValueError("repository runtime total_raw_bytes mismatch")
    _nonnegative_integer(
        payload["excluded_cache_directory_count"],
        "excluded cache directory count",
    )
    _nonnegative_integer(
        payload["excluded_bytecode_file_count"],
        "excluded bytecode file count",
    )

    python_entries = [
        row
        for row in entries
        if pathlib.PurePosixPath(_text(row["path"], "repository runtime path")).suffix.casefold()
        == _PYTHON_SOURCE_SUFFIX
    ]
    python_count = _positive_integer(
        payload["frozen_python_source_entry_count"],
        "frozen Python source entry count",
    )
    if python_count != len(python_entries) or python_count > source_entry_count:
        raise ValueError("frozen Python source entry count mismatch")
    expected_join = hashlib.sha256(canonical_json_bytes({"entries": python_entries})).hexdigest()
    if _digest(payload["frozen_python_source_join_sha256"], "frozen Python source join sha256") != expected_join:
        raise ValueError("frozen Python source join digest mismatch")
    _validate_artifact_id(payload)


def _canonical_directory(value: pathlib.Path, field_name: str) -> pathlib.Path:
    candidate = pathlib.Path(value)
    if not candidate.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    if not os.path.lexists(candidate):
        raise FileNotFoundError(f"{field_name} is absent: {candidate}")
    observed = os.lstat(candidate)
    if stat.S_ISLNK(observed.st_mode) or _is_reparse(observed) or not stat.S_ISDIR(observed.st_mode):
        raise ValueError(f"{field_name} must be a non-symlink, non-reparse directory: {candidate}")
    resolved = candidate.resolve(strict=True)
    if _normalized_path(candidate) != _normalized_path(resolved):
        raise ValueError(f"{field_name} must be canonical: {candidate}")
    return resolved


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


def _validate_regular_file_before_read(
    path: pathlib.Path,
    *,
    root: pathlib.Path,
    field_name: str,
) -> os.stat_result:
    candidate = pathlib.Path(path).absolute()
    _relative_to_root(candidate, root=root, field_name=field_name)
    if not os.path.lexists(candidate):
        raise FileNotFoundError(f"{field_name} is absent: {candidate}")
    before = os.lstat(candidate)
    if stat.S_ISLNK(before.st_mode) or _is_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{field_name} must be a regular non-symlink, non-reparse file")
    if before.st_nlink != 1:
        raise ValueError(f"{field_name} must have exactly one hard link")
    return before


def _hash_stable_regular_file(
    path: pathlib.Path,
    *,
    root: pathlib.Path,
    field_name: str,
) -> tuple[str, int]:
    candidate = pathlib.Path(path).absolute()
    before = _validate_regular_file_before_read(candidate, root=root, field_name=field_name)
    digest = hashlib.sha256()
    raw_bytes = 0
    with candidate.open("rb") as handle:
        during = os.fstat(handle.fileno())
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            raw_bytes += len(chunk)
    after = os.lstat(candidate)
    if _file_identity(before) != _file_identity(during) or _file_identity(during) != _file_identity(after):
        raise RuntimeError(f"{field_name} changed identity while being hashed")
    if raw_bytes != before.st_size:
        raise RuntimeError(f"{field_name} changed byte count while being hashed")
    return digest.hexdigest(), raw_bytes


def _relative_to_root(path: pathlib.Path, *, root: pathlib.Path, field_name: str) -> pathlib.Path:
    candidate = pathlib.Path(path).absolute()
    try:
        return candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field_name} escapes repository root: {candidate}") from error


def _source_root_relative(value: object, field_name: str) -> str:
    path = _relative_posix_path(value, field_name)
    parts = pathlib.PurePosixPath(path).parts
    if len(parts) != 3 or parts[0] != "packages" or parts[2] != "src":
        raise ValueError(f"{field_name} must match packages/*/src")
    return path


def _relative_posix_path(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must be valid UTF-8 text") from error
    path = pathlib.PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field_name} must be canonical repository-relative POSIX text")
    return text


def _validate_canonical_path_order(paths: list[str], *, field_name: str) -> None:
    if not paths:
        raise ValueError(f"{field_name} must not be empty")
    if paths != sorted(paths, key=lambda value: value.encode("utf-8")):
        raise ValueError(f"{field_name} must use UTF-8 ordinal path order")
    if len(paths) != len(set(paths)):
        raise ValueError(f"{field_name} contain duplicate paths")
    if len(paths) != len({value.casefold() for value in paths}):
        raise ValueError(f"{field_name} contain a casefold path collision")


def _raise_walk_error(error: OSError) -> None:
    raise error


def _validate_artifact_id(payload: Mapping[str, object]) -> None:
    supplied = _digest(payload["artifact_id"], "repository runtime coverage artifact_id")
    core = {key: value for key, value in payload.items() if key != "artifact_id"}
    expected = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
    if supplied != expected:
        raise ValueError("repository runtime coverage artifact_id mismatch")


def _with_artifact_id(core: Mapping[str, object]) -> dict[str, object]:
    return {
        **core,
        "artifact_id": hashlib.sha256(canonical_json_bytes(dict(core))).hexdigest(),
    }


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field_name} must be non-empty text")
    return value


def _digest(value: object, field_name: str) -> str:
    text = _text(value, field_name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _nonnegative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{field_name} must be a non-negative integer")
    return value


def _positive_integer(value: object, field_name: str) -> int:
    integer = _nonnegative_integer(value, field_name)
    if integer == 0:
        raise ValueError(f"{field_name} must be positive")
    return integer


def _normalized_path(value: pathlib.Path) -> str:
    return os.path.normcase(os.path.normpath(str(value)))


def _is_reparse(value: os.stat_result) -> bool:
    return bool(os.name == "nt" and getattr(value, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_nlink,
    )


__all__ = [
    "RELATIONSHIP_READER_REPOSITORY_RUNTIME_COVERAGE_SCHEMA_VERSION",
    "build_relationship_condition_reader_repository_runtime_coverage",
    "validate_relationship_condition_reader_repository_runtime_coverage",
]
