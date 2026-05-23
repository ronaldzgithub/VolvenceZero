"""Verify Phase 2/3 shadow evidence artifact manifest integrity.

Checks that every file listed in ``phase2_shadow_evidence_manifest.json``
exists and matches its recorded sha256 / byte size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


_SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset(
    {
        "phase2-shadow-evidence-manifest.v1",
        "phase2-shadow-evidence-multiseed-manifest.v1",
    }
)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be a JSON object")
    schema_version = payload["schema_version"]
    if schema_version not in _SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported schema_version {schema_version!r}; "
            f"expected one of {sorted(_SUPPORTED_MANIFEST_SCHEMA_VERSIONS)!r}"
        )
    return payload


def _resolve_artifact_path(raw_path: str, *, manifest_path: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = candidate
    if cwd_candidate.exists():
        return cwd_candidate
    sibling_candidate = manifest_path.parent / candidate.name
    return sibling_candidate


def verify_manifest(path: Path) -> tuple[str, ...]:
    """Return verified artifact paths; raise on any mismatch."""
    manifest = _load_manifest(path)
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("manifest.artifacts must be a non-empty list")

    verified: list[str] = []
    for item in artifacts:
        if not isinstance(item, dict):
            raise ValueError("manifest.artifacts entries must be JSON objects")
        raw_path = item["path"]
        expected_sha = item["sha256"]
        expected_size = item["size_bytes"]
        if not isinstance(raw_path, str):
            raise TypeError("artifact.path must be a string")
        if not isinstance(expected_sha, str):
            raise TypeError("artifact.sha256 must be a string")
        if not isinstance(expected_size, int):
            raise TypeError("artifact.size_bytes must be an integer")
        artifact_path = _resolve_artifact_path(raw_path, manifest_path=path)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"artifact not found: {artifact_path}")
        data = artifact_path.read_bytes()
        actual_sha = hashlib.sha256(data).hexdigest()
        actual_size = len(data)
        if actual_sha != expected_sha:
            raise ValueError(
                f"sha256 mismatch for {artifact_path}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        if actual_size != expected_size:
            raise ValueError(
                f"size mismatch for {artifact_path}: "
                f"expected {expected_size}, got {actual_size}"
            )
        verified.append(str(artifact_path))
    return tuple(verified)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to phase2_shadow_evidence_manifest.json.",
    )
    args = parser.parse_args()
    verified = verify_manifest(args.manifest)
    print(f"verified {len(verified)} artifact(s)")
    for item in verified:
        print(f"  {item}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
