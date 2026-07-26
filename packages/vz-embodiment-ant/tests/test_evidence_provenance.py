"""Digital-ant artifact provenance and integrity contracts."""

from __future__ import annotations

import json
import math
from dataclasses import fields
from pathlib import Path

import pytest

from volvence_ant.evidence.provenance import (
    ANT_ARTIFACT_SCHEMA_VERSION,
    AntArtifactExistsError,
    AntArtifactIntegrityError,
    AntRunProvenance,
    artifact_verdict_summary,
    atomic_write_json,
    collect_ant_provenance,
    ensure_artifact_writable,
    require_ant_artifact_envelope,
    resolve_effective_backend,
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
    assert not tuple(tmp_path.glob("*.tmp"))
    assert not tuple(tmp_path.glob(".*.tmp"))
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


def test_every_artifact_carries_provenance_and_a_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "p1" / "ecology_p1.seed0.run.json"
    manifest = write_ant_artifact_bundle(
        artifact_path=artifact,
        payload={
            "schema_version": "digital-ant-ecology-p1-development.v25",
            "verdict": "BLOCK",
        },
        provenance=_provenance(dirty=True),
        repo_root=tmp_path,
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    # A domain schema in the payload is preserved -- load_p1_prerequisite
    # matches on that exact string -- and the envelope keeps its own key.
    assert payload["schema_version"] == "digital-ant-ecology-p1-development.v25"
    assert payload["evidence_envelope_schema_version"] == ANT_ARTIFACT_SCHEMA_VERSION
    assert payload["provenance"]["git_sha"] == "abc123"
    assert payload["provenance"]["working_tree_dirty"] is True
    assert manifest.is_file()
    assert manifest.name.endswith(".manifest.json")


def test_payload_may_not_shadow_the_envelope_provenance(tmp_path: Path) -> None:
    with pytest.raises(AntArtifactIntegrityError, match="envelope-owned keys"):
        write_ant_artifact_bundle(
            artifact_path=tmp_path / "result.json",
            payload={"provenance": {"git_sha": "forged"}},
            provenance=_provenance(dirty=False),
            repo_root=tmp_path,
        )


def test_the_shared_writer_still_replaces_in_place_by_default(
    tmp_path: Path,
) -> None:
    """Eleven drivers outside the ecology lane write a fixed output path.

    They call this writer with no ``overwrite`` argument and have no force
    flag, so flipping the default would make their very next invocation raise
    on an artifact that already exists on disk. The no-overwrite rule is opt-in
    per lane (see ``test_the_ecology_lane_opts_in_to_the_no_overwrite_rule``),
    not a property of the shared writer.
    """

    artifact = tmp_path / "phase0_homing.json"
    write_ant_artifact_bundle(
        artifact_path=artifact,
        payload={"artifact_kind": "test", "verdict": "BLOCK"},
        provenance=_provenance(dirty=False),
        repo_root=tmp_path,
    )
    manifest = write_ant_artifact_bundle(
        artifact_path=artifact,
        payload={"artifact_kind": "test", "verdict": "PASS"},
        provenance=_provenance(dirty=False),
        repo_root=tmp_path,
    )
    assert json.loads(artifact.read_text(encoding="utf-8"))["verdict"] == "PASS"
    verify_ant_artifact_manifest(manifest_path=manifest, repo_root=tmp_path)


def test_the_ecology_lane_opts_in_to_the_no_overwrite_rule(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "result.json"
    write_ant_artifact_bundle(
        artifact_path=artifact,
        payload={"artifact_kind": "test", "verdict": "BLOCK"},
        provenance=_provenance(dirty=False),
        repo_root=tmp_path,
        overwrite=False,
    )
    with pytest.raises(AntArtifactExistsError, match="verdict='BLOCK'"):
        write_ant_artifact_bundle(
            artifact_path=artifact,
            payload={"artifact_kind": "test", "verdict": "PASS"},
            provenance=_provenance(dirty=False),
            repo_root=tmp_path,
            overwrite=False,
        )
    # The refused write left the BLOCK artifact byte-identical.
    assert json.loads(artifact.read_text(encoding="utf-8"))["verdict"] == "BLOCK"

    manifest = write_ant_artifact_bundle(
        artifact_path=artifact,
        payload={"artifact_kind": "test", "verdict": "PASS"},
        provenance=_provenance(dirty=False),
        repo_root=tmp_path,
        overwrite=True,
    )
    assert json.loads(artifact.read_text(encoding="utf-8"))["verdict"] == "PASS"
    verify_ant_artifact_manifest(manifest_path=manifest, repo_root=tmp_path)


def test_orphan_manifest_also_blocks_a_silent_rewrite(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    manifest = write_ant_artifact_bundle(
        artifact_path=artifact,
        payload={"artifact_kind": "test", "verdict": "BLOCK"},
        provenance=_provenance(dirty=False),
        repo_root=tmp_path,
        overwrite=False,
    )
    artifact.unlink()
    assert manifest.is_file()
    with pytest.raises(AntArtifactExistsError, match="artifact is missing"):
        write_ant_artifact_bundle(
            artifact_path=artifact,
            payload={"artifact_kind": "test", "verdict": "PASS"},
            provenance=_provenance(dirty=False),
            repo_root=tmp_path,
            overwrite=False,
        )


def test_a_protected_artifact_that_is_not_utf8_still_refuses_the_write(
    tmp_path: Path,
) -> None:
    """``read_text`` raises ``UnicodeDecodeError``, which is not an ``OSError``.

    Letting it escape would replace the "artifact is protected" refusal with a
    decoding traceback, and an operator would not learn that the file on the
    path is evidence.
    """

    artifact = tmp_path / "result.json"
    artifact.write_bytes(b"\xff\xfe\x00 not utf-8")
    assert artifact_verdict_summary(artifact) == "unreadable: UnicodeDecodeError"
    with pytest.raises(AntArtifactExistsError, match="unreadable"):
        ensure_artifact_writable(artifact, overwrite=False)
    assert artifact.read_bytes() == b"\xff\xfe\x00 not utf-8"


def test_non_finite_values_cannot_enter_an_artifact_or_a_digest(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Out of range float"):
        stable_json_digest({"harmful_tick_rate": math.nan})
    with pytest.raises(ValueError, match="Out of range float"):
        atomic_write_json(tmp_path / "nan.json", {"escape_latency": math.inf})
    assert not (tmp_path / "nan.json").exists()
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_unserialisable_values_raise_instead_of_being_stringified() -> None:
    class Opaque:
        pass

    with pytest.raises(TypeError):
        stable_json_digest({"config": Opaque()})


def test_atomic_write_json_can_refuse_an_existing_destination(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.json"
    atomic_write_json(path, {"value": 1})
    atomic_write_json(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == 2
    with pytest.raises(AntArtifactExistsError, match="refusing to overwrite"):
        atomic_write_json(path, {"value": 3}, overwrite=False)
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == 2


def test_provenance_records_device_and_separates_seed_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VZ_TENSOR_DEVICE", raising=False)
    provenance = collect_ant_provenance(
        repo_root=Path(__file__).resolve().parents[3],
        seeds=(0, 1),
        config={"ticks": 4},
        training_seeds=(0,),
        layout_seeds=(2_000_003, 2_010_010),
    )
    assert provenance.requested_device == "cpu"
    assert provenance.effective_backend == "pure:cpu"
    # Training seeds and layout seeds stay separately recoverable instead of
    # being collapsed into one schedule digest.
    assert provenance.training_seeds == (0,)
    assert provenance.layout_seeds == (2_000_003, 2_010_010)

    # train_ant_ecology.py communicates --device through the environment; a
    # CUDA/MPS run must not be indistinguishable from a CPU run in the record.
    monkeypatch.setenv("VZ_TENSOR_DEVICE", "cpu")
    assert (
        collect_ant_provenance(
            repo_root=Path(__file__).resolve().parents[3],
            seeds=(0,),
            config={"ticks": 4},
        ).requested_device
        == "cpu"
    )


def test_envelope_version_is_pinned_to_the_provenance_field_set() -> None:
    """The envelope version must move whenever the envelope shape moves.

    The previous wave added a top-level ``evidence_envelope_schema_version``
    key and four provenance fields while leaving the version string at ``v2``,
    so an old and a new artifact were indistinguishable by their declared
    contract. Pinning both here makes the next shape change fail at this
    assertion instead of silently shipping.
    """

    assert ANT_ARTIFACT_SCHEMA_VERSION == "digital-ant-evidence.v3"
    assert tuple(field.name for field in fields(AntRunProvenance)) == (
        "git_sha",
        "git_branch",
        "working_tree_dirty",
        "python_version",
        "platform",
        "dependency_versions",
        "seed_schedule",
        "config_digest",
        "model_fingerprint",
        "requested_device",
        "effective_backend",
        "training_seeds",
        "layout_seeds",
    )


def test_a_pre_v3_envelope_is_rejected_loudly(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    write_ant_artifact_bundle(
        artifact_path=artifact,
        payload={"artifact_kind": "test"},
        provenance=_provenance(dirty=False),
        repo_root=tmp_path,
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    require_ant_artifact_envelope(payload, path=artifact)

    stale = dict(payload)
    stale["evidence_envelope_schema_version"] = "digital-ant-evidence.v2"
    with pytest.raises(AntArtifactIntegrityError, match="unsupported evidence envelope"):
        require_ant_artifact_envelope(stale, path=artifact)

    legacy = {
        key: value
        for key, value in payload.items()
        if key != "evidence_envelope_schema_version"
    }
    with pytest.raises(AntArtifactIntegrityError, match="predates the evidence envelope"):
        require_ant_artifact_envelope(legacy, path=artifact)


def test_an_unavailable_accelerator_is_never_recorded_as_cpu() -> None:
    assert resolve_effective_backend("cpu") == "pure:cpu"
    # No CUDA host in this test environment: the request must fail loudly
    # rather than degrade into a CPU record.
    with pytest.raises(RuntimeError, match="cuda"):
        resolve_effective_backend("cuda")
    with pytest.raises(ValueError, match="non-empty"):
        resolve_effective_backend("  ")
