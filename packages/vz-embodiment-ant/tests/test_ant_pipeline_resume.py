"""Pipeline stage resume requires a matching marker and valid manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_ant_pipeline
from volvence_ant.evidence.provenance import (
    AntArtifactIntegrityError,
    AntRunProvenance,
    stable_json_digest,
    write_ant_artifact_bundle,
)


def _provenance() -> AntRunProvenance:
    return AntRunProvenance(
        git_sha="abc",
        git_branch="test",
        working_tree_dirty=True,
        python_version="3.13",
        platform="test",
        dependency_versions=(),
        seed_schedule=(0,),
        config_digest=stable_json_digest({"ticks": 2}),
    )


def test_pipeline_resume_requires_matching_marker_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tmp_path / "results"
    output = results / "matched_control.json"
    write_ant_artifact_bundle(
        artifact_path=output,
        payload={"verdict": "BLOCK"},
        provenance=_provenance(),
        repo_root=tmp_path,
    )
    monkeypatch.setattr(run_ant_pipeline, "_ROOT", tmp_path)
    monkeypatch.setattr(run_ant_pipeline, "_RUNNER_STATE", results / ".runner")
    fingerprint = run_ant_pipeline._stage_fingerprint(
        profile="formal",
        stage="matched",
        command=["python", "runner.py", "--ticks", "2", "--workers", "4"],
        outputs=(output,),
    )
    assert not run_ant_pipeline._resume_stage(
        stage="matched",
        fingerprint=fingerprint,
        outputs=(output,),
    )

    run_ant_pipeline._commit_stage(
        stage="matched",
        fingerprint=fingerprint,
        outputs=(output,),
    )
    assert run_ant_pipeline._resume_stage(
        stage="matched",
        fingerprint=fingerprint,
        outputs=(output,),
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        run_ant_pipeline._resume_stage(
            stage="matched",
            fingerprint="different",
            outputs=(output,),
        )

    output.write_text("{}", encoding="utf-8")
    with pytest.raises(AntArtifactIntegrityError, match="digest mismatch"):
        run_ant_pipeline._resume_stage(
            stage="matched",
            fingerprint=fingerprint,
            outputs=(output,),
        )


def test_pipeline_fingerprint_ignores_worker_count() -> None:
    outputs = (run_ant_pipeline._RESULTS / "matched_control.json",)
    first = run_ant_pipeline._stage_fingerprint(
        profile="formal",
        stage="matched",
        command=["python", "runner.py", "--ticks", "2", "--workers", "1"],
        outputs=outputs,
    )
    second = run_ant_pipeline._stage_fingerprint(
        profile="formal",
        stage="matched",
        command=["python", "runner.py", "--ticks", "2", "--workers", "5"],
        outputs=outputs,
    )
    assert first == second
