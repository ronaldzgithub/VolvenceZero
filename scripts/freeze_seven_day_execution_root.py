#!/usr/bin/env python3
"""Materialize one preregistration-bound, read-only seven-day execution root."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence


EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})
EXCLUDED_PATTERNS = ("**/__pycache__/**", "**/*.pyc", "**/*.pyo")


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _load_mapping(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload


def _snapshot_contract(
    preregistration: Mapping[str, object],
) -> tuple[tuple[str, ...], int, str]:
    raw_snapshot = preregistration.get("execution_source_snapshot")
    if not isinstance(raw_snapshot, Mapping):
        raise ValueError("preregistration lacks execution_source_snapshot")
    raw_roots = raw_snapshot.get("roots")
    file_count = raw_snapshot.get("file_count")
    tree_sha256 = raw_snapshot.get("tree_sha256")
    if not isinstance(raw_roots, list) or not raw_roots:
        raise ValueError("execution_source_snapshot.roots must be a non-empty list")
    if not all(isinstance(item, str) and item for item in raw_roots):
        raise ValueError("execution_source_snapshot.roots contains an invalid value")
    if len(set(raw_roots)) != len(raw_roots):
        raise ValueError("execution_source_snapshot.roots contains duplicates")
    if (
        isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count <= 0
    ):
        raise ValueError("execution_source_snapshot.file_count must be positive")
    if not isinstance(tree_sha256, str) or len(tree_sha256) != 64:
        raise ValueError("execution_source_snapshot.tree_sha256 must be SHA-256")
    try:
        int(tree_sha256, 16)
    except ValueError as exc:
        raise ValueError(
            "execution_source_snapshot.tree_sha256 must be SHA-256"
        ) from exc
    return tuple(raw_roots), file_count, tree_sha256


def _validate_pattern(pattern: str) -> None:
    candidate = Path(pattern)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(
            "execution_source_snapshot roots must stay below the repository"
        )


def _collect_files(root: Path, patterns: Sequence[str]) -> tuple[Path, ...]:
    files: set[Path] = set()
    for pattern in patterns:
        _validate_pattern(pattern)
        for candidate in root.glob(pattern):
            if candidate.is_symlink():
                raise ValueError(f"execution source cannot be a symlink: {candidate}")
            if candidate.is_dir():
                for path in candidate.rglob("*"):
                    if path.is_symlink():
                        raise ValueError(
                            f"execution source cannot contain a symlink: {path}"
                        )
                    if path.is_file():
                        files.add(path)
            elif candidate.is_file():
                files.add(candidate)
    included = tuple(
        sorted(
            (
                path
                for path in files
                if "__pycache__" not in path.parts
                and path.suffix not in EXCLUDED_SUFFIXES
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    if not included:
        raise FileNotFoundError("seven-day execution source snapshot is empty")
    return included


def _tree_sha256(root: Path, files: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def freeze_execution_root(
    *,
    repo_root: Path,
    preregistration_path: Path,
    output_root: Path,
    manifest_schema_version: str = "seven-day-frozen-execution-root.v1",
) -> dict[str, object]:
    source = repo_root.resolve()
    prereg_path = preregistration_path.resolve()
    target = output_root.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {source}")
    if not prereg_path.is_file():
        raise FileNotFoundError(f"preregistration does not exist: {prereg_path}")
    if target.exists():
        raise FileExistsError(f"frozen execution root already exists: {target}")
    if target == source or source in target.parents:
        raise ValueError("frozen execution root must be outside the source repository")

    preregistration = _load_mapping(prereg_path, label="preregistration")
    patterns, expected_count, expected_tree = _snapshot_contract(preregistration)
    source_files = _collect_files(source, patterns)
    actual_tree = _tree_sha256(source, source_files)
    if len(source_files) != expected_count or actual_tree != expected_tree:
        raise ValueError(
            "current source tree differs from the preregistered execution snapshot"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        entries: list[dict[str, str]] = []
        for source_path in source_files:
            relative = source_path.relative_to(source)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination)
            entries.append(
                {
                    "path": relative.as_posix(),
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                }
            )
        copied_files = _collect_files(staging, patterns)
        copied_tree = _tree_sha256(staging, copied_files)
        if len(copied_files) != expected_count or copied_tree != expected_tree:
            raise RuntimeError("copied execution root differs from preregistration")
        manifest: dict[str, object] = {
            "schema_version": manifest_schema_version,
            "preregistration_sha256": hashlib.sha256(
                prereg_path.read_bytes()
            ).hexdigest(),
            "source_tree_sha256": expected_tree,
            "file_count": expected_count,
            "excluded": list(EXCLUDED_PATTERNS),
            "files": entries,
            "read_only": True,
        }
        (staging / "frozen_execution_root_manifest.json").write_bytes(
            _canonical_bytes(manifest)
        )
        staging.rename(target)
        _make_read_only(target)
        return manifest
    except (OSError, RuntimeError, ValueError, shutil.Error):
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = freeze_execution_root(
        repo_root=args.repo_root,
        preregistration_path=args.preregistration,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "schema_version": manifest["schema_version"],
                "output_root": str(args.output_root.resolve()),
                "preregistration_sha256": manifest["preregistration_sha256"],
                "source_tree_sha256": manifest["source_tree_sha256"],
                "file_count": manifest["file_count"],
                "read_only": manifest["read_only"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
