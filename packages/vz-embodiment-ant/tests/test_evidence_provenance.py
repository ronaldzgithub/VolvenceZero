"""Digital-ant artifact provenance and integrity contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from volvence_ant.evidence.provenance import (
    ANT_ARTIFACT_SCHEMA_VERSION,
    AntArtifactIntegrityError,
    AntRunProvenance,
    stable_json_digest,
    verify_ant_artifact_manifest,
    write_ant_artifact_bundle,
)


def _provenance(*, dirty: bool) -> AntRunProvenance:
    return AntRunProvenance(
        git_sha="abc123",
        git_branch="test",
        working_tree_dirty=dirty,
        python_version="3.11",
        platform="test",
        dependency_versions=(("numpy", "1"),),
        seed_schedule=(0, 1),
        config_digest=stable_json_digest({"ticks": 10}),
    )


def test_bundle_records_digest_and_dirty_retain_boundary(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    manifest = write_ant_artifact_bundle(
        artifact_path=artifact,
        payload={"artifact_kind": "test", "value": 1},
        provenance=_provenance(dirty=True),
        repo_root=tmp_path,
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == ANT_ARTIFACT_SCHEMA_VERSION
    assert manifest_payload["externally_retainable"] is False
    verify_ant_artifact_manifest(manifest_path=manifest, repo_root=tmp_path)


def test_manifest_verification_fails_after_tamper(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    manifest = write_ant_artifact_bundle(
        artifact_path=artifact,
        payload={"artifact_kind": "test"},
        provenance=_provenance(dirty=False),
        repo_root=tmp_path,
    )
    artifact.write_text("{}", encoding="utf-8")
    with pytest.raises(AntArtifactIntegrityError, match="digest mismatch"):
        verify_ant_artifact_manifest(manifest_path=manifest, repo_root=tmp_path)
