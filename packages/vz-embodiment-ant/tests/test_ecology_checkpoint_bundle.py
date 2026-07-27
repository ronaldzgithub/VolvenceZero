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
    ECOLOGY_CHECKPOINT_BUNDLE_KIND,
    EcologyP2PromotionEvidence,
    _validated_report_verdict,
    load_promoted_ecology_checkpoint,
    p2_promotion_evidence,
    write_ecology_checkpoint_bundle,
    write_ecology_p2_promotion_bundle,
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
from volvence_ant.experiments.ecology_p2 import (
    ECOLOGY_P2_GATE_NAMES,
    ECOLOGY_P2_SCHEMA_VERSION,
    EcologyP2Config,
    heldout_layout_seeds,
)


#: The 30 frozen held-out layout seeds a formal confirmatory matrix scores on.
_HELDOUT_LAYOUT_SEEDS = heldout_layout_seeds(EcologyP2Config())


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
        dense_local_shaping_off_training=(),
        segment_credit_off_training=(),
        learned_mastery=(),
        action_probes=(),
        validation_metrics=(),
        learned_metrics=(),
        cold_metrics=(),
        no_optimize_metrics=(),
        dense_local_shaping_off_metrics=(),
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


def _p2_report_payload(*, verdict: str = "PASS", n_ants: int = 1) -> dict:
    """Synthetic P2 confirmatory payload.

    Exists only to exercise the promotion admission contract; it is never
    evidence that the real confirmatory matrix passed.
    """

    passed = verdict == "PASS"
    return {
        "schema_version": ECOLOGY_P2_SCHEMA_VERSION,
        "preregistration_digest": "d" * 64,
        "verdict": verdict,
        "training_seeds": [0, 1, 2],
        "arms": ["learned"],
        "device": "cpu",
        "heldout_layout_seeds": list(_HELDOUT_LAYOUT_SEEDS),
        "source_git_sha": "a" * 40,
        "config": {"n_ants": n_ants, "temporal_latent_dim": 4},
        "gates": [
            {
                "name": name,
                "passed": passed or name != ECOLOGY_P2_GATE_NAMES[0],
                "observed": "synthetic",
                "threshold": "synthetic",
            }
            for name in ECOLOGY_P2_GATE_NAMES
        ],
    }


def _p2_evidence(
    *, verdict: str = "PASS", n_ants: int = 1
) -> EcologyP2PromotionEvidence:
    return p2_promotion_evidence(
        _p2_report_payload(verdict=verdict, n_ants=n_ants),
        report_path="synthetic-p2.json",
        report_sha256="b" * 64,
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
    p2_verdict: str | None = "PASS",
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
        p2_promotion=(
            None if p2_verdict is None else _p2_evidence(verdict=p2_verdict)
        ),
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
    assert loaded.p2_promotion is not None
    assert loaded.p2_promotion.verdict == "PASS"


def test_a_curriculum_pass_without_p2_evidence_is_not_loadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plan sections 2.3 and 5.7: P0/P1 checkpoints are never promotable.

    ``EcologyCurriculumConfig`` accepts one ant, one round and one held-out
    seed, so a curriculum ``PASS`` proves nothing about formal capability. It
    used to be fully loadable into the live demo runner while a genuine P2
    ``PASS`` produced no loadable bundle at all.
    """

    report_path = _write(
        tmp_path, monkeypatch, verdict="PASS", p2_verdict=None
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["promotion_verdict"] == "BLOCK"
    assert "P2 confirmatory verdict" in payload["promotion_block_reason"]

    with pytest.raises(
        AntArtifactIntegrityError, match="no P2 promotion verdict"
    ):
        load_promoted_ecology_checkpoint(
            report_path=report_path,
            repo_root=tmp_path,
        )


def test_a_blocked_p2_verdict_cannot_promote_a_passing_curriculum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = _write(
        tmp_path, monkeypatch, verdict="PASS", p2_verdict="BLOCK"
    )

    with pytest.raises(AntArtifactIntegrityError, match="not promoted"):
        load_promoted_ecology_checkpoint(
            report_path=report_path,
            repo_root=tmp_path,
        )


def test_a_tampered_p2_block_cannot_launder_a_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate set is re-checked on load, not summarised into a verdict."""

    report_path = _write(
        tmp_path, monkeypatch, verdict="PASS", p2_verdict="BLOCK"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["promotion_verdict"] = "PASS"
    payload["p2_promotion"]["verdict"] = "PASS"
    # Drop the failing gate from both lists, exactly as a launderer would.
    dropped = payload["p2_promotion"]["failed_gates"][0]
    payload["p2_promotion"]["failed_gates"] = []
    payload["p2_promotion"]["gate_names"] = [
        name
        for name in payload["p2_promotion"]["gate_names"]
        if name != dropped
    ]
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path = report_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact"] = asdict(file_digest(report_path, relative_to=tmp_path))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AntArtifactIntegrityError, match="gate set mismatch"):
        load_promoted_ecology_checkpoint(
            report_path=report_path,
            repo_root=tmp_path,
        )


def test_p2_promotion_evidence_rejects_drifted_or_contradicted_reports() -> None:
    stale = _p2_report_payload()
    stale["schema_version"] = "digital-ant-ecology-p2-confirmatory.v1"
    with pytest.raises(AntArtifactIntegrityError, match="confirmatory schema"):
        p2_promotion_evidence(stale, report_path="x", report_sha256="y")

    thinned = _p2_report_payload()
    thinned["gates"] = thinned["gates"][:-1]
    with pytest.raises(AntArtifactIntegrityError, match="gate set mismatch"):
        p2_promotion_evidence(thinned, report_path="x", report_sha256="y")

    contradictory = _p2_report_payload(verdict="BLOCK")
    contradictory["verdict"] = "PASS"
    with pytest.raises(AntArtifactIntegrityError, match="contradicts"):
        p2_promotion_evidence(contradictory, report_path="x", report_sha256="y")


def test_p2_lane_produces_a_loadable_bundle_without_a_curriculum_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A P2 PASS must be able to emit the loadable bundle.

    The confirmatory lane owns no curriculum report, so promotion ships the
    learned shard's own journalled colony archive plus the P2 verdict.
    """

    monkeypatch.setattr(
        "volvence_ant.evidence.ecology_checkpoint.collect_ant_provenance",
        lambda **_: _provenance(),
    )
    report_path = tmp_path / "p2-promoted.json"
    write_ecology_p2_promotion_bundle(
        checkpoint_archives=(_archive(),),
        p2_promotion=_p2_evidence(),
        archive_path=tmp_path / "p2-promoted.vzac",
        report_path=report_path,
        repo_root=tmp_path,
    )
    loaded = load_promoted_ecology_checkpoint(
        report_path=report_path,
        repo_root=tmp_path,
    )
    assert loaded.verdict == "PASS"
    assert loaded.checkpoint_archives == (_archive(),)
    assert loaded.config.n_ants == 1
    assert loaded.config.temporal_latent_dim == 4
    assert loaded.p2_promotion is not None
    assert loaded.p2_promotion.source_git_sha == "a" * 40

    with pytest.raises(AntArtifactIntegrityError, match="archive count"):
        write_ecology_p2_promotion_bundle(
            checkpoint_archives=(_archive(), _archive()),
            p2_promotion=_p2_evidence(),
            archive_path=tmp_path / "mismatch.vzac",
            report_path=tmp_path / "mismatch.json",
            repo_root=tmp_path,
        )


def test_p2_promotion_bundle_records_layout_seed_and_device_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plan section 2.1: device, training seeds AND layout seeds, every run.

    The promoted bundle used to record ``layout_seeds=()`` and no device, so
    the 30 frozen held-out seeds the verdict was scored on and the confirmatory
    backend were unrecoverable from the artifact -- while the curriculum lane
    right next to it recorded both.
    """

    captured: dict[str, object] = {}

    def _capture(**kwargs: object) -> AntRunProvenance:
        captured.update(kwargs)
        return _provenance()

    monkeypatch.setattr(
        "volvence_ant.evidence.ecology_checkpoint.collect_ant_provenance",
        _capture,
    )
    evidence = _p2_evidence()
    write_ecology_p2_promotion_bundle(
        checkpoint_archives=(_archive(),),
        p2_promotion=evidence,
        archive_path=tmp_path / "p2.vzac",
        report_path=tmp_path / "p2.json",
        repo_root=tmp_path,
    )

    assert captured["training_seeds"] == (0, 1, 2)
    assert captured["layout_seeds"] == _HELDOUT_LAYOUT_SEEDS
    assert len(captured["layout_seeds"]) == 30
    assert captured["device"] == "cpu"


def test_p2_promotion_evidence_requires_its_run_provenance() -> None:
    """A promotable verdict may not go missing its device or layout seeds."""

    seedless = _p2_report_payload()
    seedless["heldout_layout_seeds"] = []
    with pytest.raises(AntArtifactIntegrityError, match="held-out layout seeds"):
        p2_promotion_evidence(seedless, report_path="x", report_sha256="y")

    duplicated = _p2_report_payload()
    duplicated["heldout_layout_seeds"] = [7, 7, 8, 9, 10, 11]
    with pytest.raises(AntArtifactIntegrityError, match="duplicates"):
        p2_promotion_evidence(duplicated, report_path="x", report_sha256="y")

    deviceless = _p2_report_payload()
    deviceless["device"] = "  "
    with pytest.raises(AntArtifactIntegrityError, match="records no device"):
        p2_promotion_evidence(deviceless, report_path="x", report_sha256="y")


def test_p2_promotion_evidence_names_a_missing_field(tmp_path: Path) -> None:
    """A malformed artifact must not surface as a bare ``KeyError``."""

    truncated = _p2_report_payload()
    del truncated["arms"]
    with pytest.raises(
        AntArtifactIntegrityError, match="missing a required field 'arms'"
    ):
        p2_promotion_evidence(
            truncated, report_path="synthetic.json", report_sha256="y"
        )


def test_promoted_bundle_with_a_contradictory_curriculum_report_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``checkpoint_shape`` and the shipped curriculum report must agree.

    The loader stopped re-deriving the shape from the curriculum config (which
    was right -- a consumer must not be a second parser), but it also stopped
    cross-checking the two declarations, so a bundle whose report describes a
    4-ant colony next to a 1-ant shape declaration loaded cleanly and the
    report travelled as evidence for a checkpoint it does not describe.
    """

    report_path = _write(tmp_path, monkeypatch, verdict="PASS")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["checkpoint_shape"]["n_ants"] == 1
    payload["report"]["config"]["n_ants"] = 4
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path = report_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact"] = asdict(file_digest(report_path, relative_to=tmp_path))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        AntArtifactIntegrityError, match="contradicts the curriculum report"
    ):
        load_promoted_ecology_checkpoint(
            report_path=report_path,
            repo_root=tmp_path,
        )


def test_a_pre_v5_bundle_is_refused_and_the_error_names_the_remedy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BEHAVIOUR CHANGE, confirmed deliberate.

    ``v4`` bundles were admitted on their curriculum report alone, which plan
    sections 2.3/5.7 forbid, so every promotion bundle on disk is unloadable
    and ``--ecology-checkpoint-report <old>`` aborts at startup. That is the
    intended contract; what the error owes the operator is the way out.
    """

    assert ECOLOGY_CHECKPOINT_BUNDLE_KIND == "digital-ant-ecology-checkpoint.v5"
    report_path = _write(tmp_path, monkeypatch, verdict="PASS")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["artifact_kind"] = "digital-ant-ecology-checkpoint.v4"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path = report_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact"] = asdict(file_digest(report_path, relative_to=tmp_path))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AntArtifactIntegrityError) as error:
        load_promoted_ecology_checkpoint(
            report_path=report_path,
            repo_root=tmp_path,
        )
    message = str(error.value)
    assert "digital-ant-ecology-checkpoint.v4" in message
    assert "run_ant_ecology_p2.py promote" in message

    # The same remedy travels on the "no P2 verdict" refusal.
    without_p2 = _write(
        tmp_path / "nested",
        monkeypatch,
        verdict="PASS",
        p2_verdict=None,
    )
    with pytest.raises(AntArtifactIntegrityError) as second:
        load_promoted_ecology_checkpoint(
            report_path=without_p2,
            repo_root=tmp_path / "nested",
        )
    assert "run_ant_ecology_p2.py promote" in str(second.value)


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
        p2_promotion=_p2_evidence(),
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
