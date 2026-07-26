"""Ecology archive promotion and integrity tests."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest

from volvence_zero.agent import (
    AgentLearningCheckpoint,
    encode_agent_learning_archive,
)
from volvence_zero.owner_hydration import OwnerPersistenceSnapshot

from volvence_ant.evidence.ecology_checkpoint import (
    _validated_report_verdict,
    load_promoted_ecology_checkpoint,
    write_ecology_checkpoint_bundle,
)
from volvence_ant.evidence.provenance import (
    AntArtifactExistsError,
    AntArtifactIntegrityError,
    AntRunProvenance,
    file_digest,
)
from volvence_ant.experiments.ecology_curriculum import (
    ECOLOGY_CURRICULUM_SCHEMA_VERSION,
    ECOLOGY_REQUIRED_GATE_NAMES,
    EcologyCheckpointCandidate,
    EcologyCheckpointReport,
    EcologyCurriculumConfig,
    EcologyGate,
)


def _checkpoint() -> AgentLearningCheckpoint:
    return AgentLearningCheckpoint(
        checkpoint_id="ecology-test",
        joint_loop_state=("joint",),
        prediction_state=("prediction",),
        credit_state=("credit",),
        regime_state=("regime",),
        dual_track_gate_state=("dual",),
        reflection_state=("reflection",),
        policy_fingerprint="policy",
        temporal_fingerprint="temporal",
        memory_fingerprint="memory",
        fingerprint="checkpoint-fingerprint",
    )


def _archive() -> bytes:
    return encode_agent_learning_archive(
        checkpoint_id="ecology-test:body:0",
        owner_snapshots=(
            OwnerPersistenceSnapshot(
                owner_name="fixture-owner",
                schema_version=1,
                payload={"weight": 0.25},
            ),
        ),
        policy_fingerprint=hashlib.sha256(b"policy").hexdigest(),
        temporal_fingerprint=hashlib.sha256(b"temporal").hexdigest(),
        memory_fingerprint=hashlib.sha256(b"memory").hexdigest(),
    )


def _candidate(verdict: str) -> EcologyCheckpointCandidate:
    config = EcologyCurriculumConfig(
        n_ants=1,
        temporal_latent_dim=4,
        stage_rounds=1,
        stage_episodes=1,
        mastery_min_episodes=1,
        mastery_min_pickups=1,
        mastery_min_heat_events=1,
        validation_rounds=1,
        validation_seeds=(5,),
        heldout_rounds=1,
        heldout_seeds=(7,),
    )
    passed = verdict == "PASS"
    report = EcologyCheckpointReport(
        schema_version=ECOLOGY_CURRICULUM_SCHEMA_VERSION,
        config=config,
        initial_policy_fingerprints=("initial",),
        learned_policy_fingerprints=("policy",),
        no_optimize_policy_fingerprints=("initial",),
        training_schedule=(),
        learned_training=(),
        no_optimize_training=(),
        valence_off_training=(),
        segment_credit_off_training=(),
        learned_mastery=(),
        action_probes=(),
        validation_metrics=(),
        learned_metrics=(),
        cold_metrics=(),
        no_optimize_metrics=(),
        valence_off_metrics=(),
        segment_credit_off_metrics=(),
        gates=tuple(
            EcologyGate(
                name=name,
                passed=passed,
                observed=verdict,
                threshold="test fixture",
            )
            for name in ECOLOGY_REQUIRED_GATE_NAMES
        ),
        verdict=verdict,
        diagnostic_breakpoints=() if passed else ("test",),
        description=verdict,
    )
    return EcologyCheckpointCandidate(
        checkpoints=(_checkpoint(),),
        checkpoint_archives=(_archive(),),
        report=report,
    )


def _provenance() -> AntRunProvenance:
    return AntRunProvenance(
        git_sha="test-sha",
        git_branch="test",
        working_tree_dirty=False,
        python_version="test",
        platform="test",
        dependency_versions=(),
        seed_schedule=(7,),
        config_digest="config",
        model_fingerprint="model",
    )


def _write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    verdict: str,
) -> Path:
    monkeypatch.setattr(
        "volvence_ant.evidence.ecology_checkpoint.collect_ant_provenance",
        lambda **_: _provenance(),
    )
    report_path = tmp_path / "ecology.json"
    write_ecology_checkpoint_bundle(
        candidate=_candidate(verdict),
        archive_path=tmp_path / "ecology.vzac",
        report_path=report_path,
        repo_root=tmp_path,
    )
    return report_path


def test_promoted_ecology_bundle_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_promoted_ecology_checkpoint(
        report_path=_write(tmp_path, monkeypatch, verdict="PASS"),
        repo_root=tmp_path,
    )

    assert loaded.verdict == "PASS"
    assert loaded.checkpoint_archives == (_archive(),)
    assert loaded.config.n_ants == 1


def test_blocked_ecology_bundle_cannot_be_loaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = _write(tmp_path, monkeypatch, verdict="BLOCK")

    with pytest.raises(AntArtifactIntegrityError, match="not promoted"):
        load_promoted_ecology_checkpoint(
            report_path=report_path,
            repo_root=tmp_path,
        )


def test_checkpoint_writer_rejects_verdict_gate_contradiction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate("BLOCK")
    contradictory = replace(
        candidate,
        report=replace(candidate.report, verdict="PASS"),
    )
    monkeypatch.setattr(
        "volvence_ant.evidence.ecology_checkpoint.collect_ant_provenance",
        lambda **_: _provenance(),
    )

    with pytest.raises(AntArtifactIntegrityError, match="contradicts"):
        write_ecology_checkpoint_bundle(
            candidate=contradictory,
            archive_path=tmp_path / "ecology.vzac",
            report_path=tmp_path / "ecology.json",
            repo_root=tmp_path,
        )


def test_checkpoint_loader_rejects_legacy_curriculum_schema() -> None:
    payload = _candidate("PASS").report.to_dict()
    payload["schema_version"] = "digital-ant-ecology-curriculum.v4"

    with pytest.raises(AntArtifactIntegrityError, match="unexpected ecology"):
        _validated_report_verdict(payload)


def test_promotion_bundle_records_training_and_layout_seeds_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _capture(**kwargs: object) -> AntRunProvenance:
        captured.update(kwargs)
        return _provenance()

    monkeypatch.setattr(
        "volvence_ant.evidence.ecology_checkpoint.collect_ant_provenance",
        _capture,
    )
    write_ecology_checkpoint_bundle(
        candidate=_candidate("PASS"),
        archive_path=tmp_path / "ecology.vzac",
        report_path=tmp_path / "ecology.json",
        repo_root=tmp_path,
    )

    # The curriculum seed drives training; validation and held-out layout seeds
    # are a disjoint namespace and must stay separately recoverable instead of
    # collapsing into the one seed_schedule digest.
    assert captured["training_seeds"] == (0,)
    assert captured["layout_seeds"] == (5, 7)


def test_promotion_bundle_refuses_to_replace_an_existing_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = _write(tmp_path, monkeypatch, verdict="BLOCK")
    archive_path = tmp_path / "ecology.vzac"
    original = archive_path.read_bytes()

    with pytest.raises(AntArtifactExistsError, match="verdict"):
        write_ecology_checkpoint_bundle(
            candidate=_candidate("PASS"),
            archive_path=archive_path,
            report_path=report_path,
            repo_root=tmp_path,
        )
    # Both halves of the refused bundle are intact -- the archive is checked
    # before its bytes are written, so no half-replaced bundle can exist.
    assert archive_path.read_bytes() == original
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["promotion_verdict"] == "BLOCK"


def test_promotion_bundle_refuses_an_orphan_archive_without_its_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = _write(tmp_path, monkeypatch, verdict="BLOCK")
    archive_path = tmp_path / "ecology.vzac"
    report_path.unlink()
    report_path.with_suffix(".manifest.json").unlink()

    with pytest.raises(AntArtifactExistsError, match="report absent"):
        write_ecology_checkpoint_bundle(
            candidate=_candidate("PASS"),
            archive_path=archive_path,
            report_path=report_path,
            repo_root=tmp_path,
        )


def test_promotion_bundle_replaces_only_with_explicit_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = _write(tmp_path, monkeypatch, verdict="BLOCK")
    write_ecology_checkpoint_bundle(
        candidate=_candidate("PASS"),
        archive_path=tmp_path / "ecology.vzac",
        report_path=report_path,
        repo_root=tmp_path,
        overwrite=True,
    )

    assert (
        load_promoted_ecology_checkpoint(
            report_path=report_path,
            repo_root=tmp_path,
        ).verdict
        == "PASS"
    )


def test_promotion_loader_rejects_a_pre_v3_evidence_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An artifact whose envelope predates ``v3`` cannot be promoted.

    Its provenance block has no ``requested_device`` / ``effective_backend`` /
    ``training_seeds`` / ``layout_seeds``, so it cannot answer the questions
    plan section 2.1 asks of a formal artifact. The manifest is regenerated
    here so the envelope check is what fires, not the digest check.
    """

    report_path = _write(tmp_path, monkeypatch, verdict="PASS")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["evidence_envelope_schema_version"] = "digital-ant-evidence.v2"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest_path = report_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact"] = asdict(file_digest(report_path, relative_to=tmp_path))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AntArtifactIntegrityError, match="unsupported evidence envelope"):
        load_promoted_ecology_checkpoint(
            report_path=report_path,
            repo_root=tmp_path,
        )
