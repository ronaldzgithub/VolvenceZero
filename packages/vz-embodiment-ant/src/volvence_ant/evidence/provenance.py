"""Versioned provenance and integrity helpers for digital-ant artifacts.

The artifact payload is the machine-readable SSOT.  A sidecar manifest binds
it to the exact bytes, repository state, seed schedule, configuration and
optional model/reference inputs used by the run.  Dirty worktrees remain
exportable for internal review, but are explicitly not externally retainable.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

ANT_ARTIFACT_SCHEMA_VERSION = "digital-ant-evidence.v2"
ANT_MANIFEST_SCHEMA_VERSION = "digital-ant-manifest.v2"


class AntArtifactIntegrityError(RuntimeError):
    """An evidence artifact does not match its integrity manifest."""


@dataclass(frozen=True)
class ArtifactFileDigest:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class AntRunProvenance:
    git_sha: str
    git_branch: str
    working_tree_dirty: bool
    python_version: str
    platform: str
    dependency_versions: tuple[tuple[str, str], ...]
    seed_schedule: tuple[int, ...]
    config_digest: str
    model_fingerprint: str | None = None

    @property
    def externally_retainable(self) -> bool:
        return bool(self.git_sha) and not self.working_tree_dirty


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _dependency_versions(names: Sequence[str]) -> tuple[tuple[str, str], ...]:
    resolved: list[tuple[str, str]] = []
    for name in names:
        try:
            resolved.append((name, version(name)))
        except PackageNotFoundError:
            resolved.append((name, "not-installed"))
    return tuple(resolved)


def stable_json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON file after its complete bytes reach disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def collect_ant_provenance(
    *,
    repo_root: Path,
    seeds: Sequence[int],
    config: Mapping[str, Any],
    model_fingerprint: str | None = None,
) -> AntRunProvenance:
    return AntRunProvenance(
        git_sha=_git(repo_root, "rev-parse", "HEAD"),
        git_branch=_git(repo_root, "branch", "--show-current"),
        working_tree_dirty=bool(_git(repo_root, "status", "--porcelain")),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        dependency_versions=_dependency_versions(
            ("vz-contracts", "vz-substrate", "vz-runtime", "numpy", "torch")
        ),
        seed_schedule=tuple(int(seed) for seed in seeds),
        config_digest=stable_json_digest(dict(config)),
        model_fingerprint=model_fingerprint,
    )


def file_digest(path: Path, *, relative_to: Path | None = None) -> ArtifactFileDigest:
    resolved = (
        path
        if path.is_absolute() or relative_to is None
        else relative_to / path
    )
    raw = resolved.read_bytes()
    display_path = (
        resolved.relative_to(relative_to)
        if relative_to is not None
        else resolved
    )
    return ArtifactFileDigest(
        path=str(display_path),
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
    )


def write_ant_artifact_bundle(
    *,
    artifact_path: Path,
    payload: Mapping[str, Any],
    provenance: AntRunProvenance,
    input_paths: Sequence[Path] = (),
    repo_root: Path,
) -> Path:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        "schema_version": ANT_ARTIFACT_SCHEMA_VERSION,
        **dict(payload),
        "provenance": asdict(provenance),
    }
    atomic_write_json(artifact_path, enriched)
    artifact = file_digest(artifact_path, relative_to=repo_root)
    inputs = tuple(file_digest(path, relative_to=repo_root) for path in input_paths)
    manifest_path = artifact_path.with_suffix(".manifest.json")
    manifest = {
        "schema_version": ANT_MANIFEST_SCHEMA_VERSION,
        "artifact": asdict(artifact),
        "inputs": [asdict(item) for item in inputs],
        "provenance": asdict(provenance),
        "externally_retainable": provenance.externally_retainable,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def verify_ant_artifact_manifest(*, manifest_path: Path, repo_root: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ANT_MANIFEST_SCHEMA_VERSION:
        raise AntArtifactIntegrityError(
            f"unsupported manifest schema: {manifest.get('schema_version')!r}"
        )
    entries = (manifest["artifact"], *manifest.get("inputs", ()))
    for entry in entries:
        path = repo_root / entry["path"]
        if not path.is_file():
            raise AntArtifactIntegrityError(f"manifest file missing: {path}")
        actual = file_digest(path, relative_to=repo_root)
        if actual.sha256 != entry["sha256"] or actual.size_bytes != entry["size_bytes"]:
            raise AntArtifactIntegrityError(f"manifest digest mismatch: {path}")


__all__ = [
    "ANT_ARTIFACT_SCHEMA_VERSION",
    "ANT_MANIFEST_SCHEMA_VERSION",
    "AntArtifactIntegrityError",
    "AntRunProvenance",
    "ArtifactFileDigest",
    "atomic_write_json",
    "collect_ant_provenance",
    "file_digest",
    "stable_json_digest",
    "verify_ant_artifact_manifest",
    "write_ant_artifact_bundle",
]
