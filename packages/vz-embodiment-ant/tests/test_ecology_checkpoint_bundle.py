"""Ecology archive promotion and integrity tests."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from volvence_zero.agent import (
    AgentLearningCheckpoint,
    encode_agent_learning_archive,
)
from volvence_zero.owner_hydration import OwnerPersistenceSnapshot

from volvence_ant.evidence.ecology_checkpoint import (
    load_promoted_ecology_checkpoint,
    write_ecology_checkpoint_bundle,
)
from volvence_ant.evidence.provenance import (
    AntArtifactIntegrityError,
    AntRunProvenance,
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
        mastery_min_obstacle_contacts=1,
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
