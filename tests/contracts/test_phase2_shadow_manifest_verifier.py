"""Contract tests for scripts/verify_phase2_shadow_evidence_manifest.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_phase2_shadow_evidence_manifest.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "verify_phase2_shadow_evidence_manifest", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import script from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(tmp_path: pathlib.Path, artifact: pathlib.Path) -> pathlib.Path:
    data = artifact.read_bytes()
    manifest = {
        "schema_version": "phase2-shadow-evidence-manifest.v1",
        "artifact_kind": "phase2_shadow_evidence_manifest",
        "source_schema_version": "phase2-shadow-evidence-smoke.v1",
        "artifacts": (
            {
                "path": str(artifact),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            },
        ),
        "provenance": {
            "git_sha": "test",
            "git_branch": "test",
            "working_tree_dirty": False,
            "python_version": "test",
            "platform": "test",
        },
    }
    path = tmp_path / "phase2_shadow_evidence_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _write_multiseed_manifest(tmp_path: pathlib.Path, artifact: pathlib.Path) -> pathlib.Path:
    data = artifact.read_bytes()
    manifest = {
        "schema_version": "phase2-shadow-evidence-multiseed-manifest.v1",
        "artifact_kind": "phase2_shadow_evidence_multiseed_manifest",
        "source_schema_version": "phase2-shadow-evidence-multiseed.v1",
        "artifacts": (
            {
                "path": str(artifact),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            },
        ),
        "provenance": {
            "git_sha": "test",
            "git_branch": "test",
            "working_tree_dirty": False,
            "python_version": "test",
            "platform": "test",
        },
    }
    path = tmp_path / "phase2_shadow_evidence_multiseed_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def test_manifest_verifier_accepts_matching_artifact(tmp_path: pathlib.Path) -> None:
    module = _load_script_module()
    artifact = tmp_path / "phase2_shadow_evidence_smoke.json"
    artifact.write_text('{"ok": true}', encoding="utf-8")
    manifest = _write_manifest(tmp_path, artifact)

    verified = module.verify_manifest(manifest)

    assert verified == (str(artifact),)


def test_manifest_verifier_accepts_multiseed_manifest(tmp_path: pathlib.Path) -> None:
    module = _load_script_module()
    artifact = tmp_path / "phase2_shadow_evidence_multiseed.json"
    artifact.write_text('{"ok": true}', encoding="utf-8")
    manifest = _write_multiseed_manifest(tmp_path, artifact)

    verified = module.verify_manifest(manifest)

    assert verified == (str(artifact),)


def test_manifest_verifier_rejects_tampered_artifact(tmp_path: pathlib.Path) -> None:
    module = _load_script_module()
    artifact = tmp_path / "phase2_shadow_evidence_smoke.json"
    artifact.write_text('{"ok": true}', encoding="utf-8")
    manifest = _write_manifest(tmp_path, artifact)
    artifact.write_text('{"ok": false}', encoding="utf-8")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        module.verify_manifest(manifest)
