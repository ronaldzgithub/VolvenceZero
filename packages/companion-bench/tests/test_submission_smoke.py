# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""End-to-end submission orchestration smoke (P10 acceptance)."""

from __future__ import annotations

import importlib.resources as res
import json
import pathlib
import textwrap

import pytest

from companion_bench.spec import load_scenarios_dir
from companion_bench.submission import (
    SubmissionAttestation,
    SubmissionManifest,
    dry_run_with_fakes,
    load_manifest,
    write_submission_summary,
)
from companion_bench.sut_client import EchoFakeSUTClient
from companion_bench.user_simulator import DeterministicFakeUtteranceClient


_MANIFEST_FIXTURE = textwrap.dedent(
    """\
    submission_id: smoke-test
    system_name: SmokeSystem
    model_identifier: fake-sut/echo
    base_url: http://localhost
    api_key_env: UNSET
    system_prompt: ""
    generation_config:
      temperature: 0.0
    attestation:
      no_companionbench_derivative_in_training: true
      no_scenario_specific_prompt: true
      no_public_test_set_tuning: true
      cross_user_memory_isolation: true
    leaderboard_category: bespoke
    """
)


def test_load_manifest_round_trip(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text(_MANIFEST_FIXTURE, encoding="utf-8")
    manifest = load_manifest(p)
    assert manifest.submission_id == "smoke-test"
    assert manifest.attestation.all_affirmed()


def test_load_manifest_rejects_unaffirmed_attestation(tmp_path: pathlib.Path) -> None:
    bad = _MANIFEST_FIXTURE.replace(
        "no_companionbench_derivative_in_training: true",
        "no_companionbench_derivative_in_training: false",
    )
    p = tmp_path / "m.yaml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="affirm all four attestation"):
        load_manifest(p)


def test_load_manifest_rejects_unknown_category(tmp_path: pathlib.Path) -> None:
    bad = _MANIFEST_FIXTURE.replace(
        "leaderboard_category: bespoke",
        "leaderboard_category: marketing-hype",
    )
    p = tmp_path / "m.yaml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="leaderboard_category must be"):
        load_manifest(p)


def test_dry_run_smoke_with_fakes_one_family(tmp_path: pathlib.Path) -> None:
    public_dir = pathlib.Path(str(res.files("companion_bench") / "scenarios" / "public"))
    # Restrict to en variants so the F1 count matches the original 4
    # initial reviewer-curated scenarios; debt #55 i18n adds one
    # F1-continuity-zh-001 to the same family but it is excluded here
    # so the smoke fixture stays stable.
    specs = tuple(
        s for s in load_scenarios_dir(public_dir, include_held_out=False)
        if s.family.value == "F1" and s.language == "en"
    )
    assert len(specs) == 4
    manifest = SubmissionManifest(
        submission_id="t",
        system_name="t",
        model_identifier="fake-sut/echo",
        base_url="http://localhost",
        api_key_env="UNSET",
        system_prompt="",
        generation_config={},
        attestation=SubmissionAttestation(True, True, True, True),
        leaderboard_category="bespoke",
    )
    artifact_dir = tmp_path / "artifacts"
    result = dry_run_with_fakes(
        manifest=manifest,
        specs=specs,
        sut_client=EchoFakeSUTClient(),
        user_backend=DeterministicFakeUtteranceClient(),
        paraphrase_seeds=(0,),
        artifact_dir=artifact_dir,
    )
    assert len(result.arc_bundles) == 4
    assert artifact_dir.exists()
    bundle_files = list(artifact_dir.glob("*.bundle.json"))
    assert len(bundle_files) == 4
    summary_path = tmp_path / "summary.json"
    write_submission_summary(result, summary_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["arc_count"] == 4
    assert "aggregate" in payload
    assert "per_axis_scores" in payload


def test_run_submission_resumes_existing_arc_bundles(tmp_path: pathlib.Path) -> None:
    public_dir = pathlib.Path(str(res.files("companion_bench") / "scenarios" / "public"))
    specs = load_scenarios_dir(public_dir, include_held_out=False)[:2]
    manifest = SubmissionManifest(
        submission_id="resume-smoke",
        system_name="ResumeSmoke",
        model_identifier="fake-sut/echo",
        base_url="http://localhost",
        api_key_env="UNSET",
        system_prompt="",
        generation_config={},
        attestation=SubmissionAttestation(True, True, True, True),
        leaderboard_category="bespoke",
    )
    artifact_dir = tmp_path / "artifacts"
    first = dry_run_with_fakes(
        manifest=manifest,
        specs=specs,
        sut_client=EchoFakeSUTClient(),
        user_backend=DeterministicFakeUtteranceClient(),
        paraphrase_seeds=(0,),
        artifact_dir=artifact_dir,
    )
    assert first.aggregate.arc_count == 2
    second = dry_run_with_fakes(
        manifest=manifest,
        specs=specs,
        sut_client=EchoFakeSUTClient(),
        user_backend=DeterministicFakeUtteranceClient(),
        paraphrase_seeds=(0,),
        artifact_dir=artifact_dir,
    )
    assert second.aggregate.arc_count == 2
    assert len(second.arc_bundles) == 0


def test_dry_run_full_public_set(tmp_path: pathlib.Path) -> None:
    """Full 30-scenario × 1-seed run completes inside the smoke budget.

    Public set = 24 en (initial) + 6 zh demo (debt #55 Batch 1).
    """
    public_dir = pathlib.Path(str(res.files("companion_bench") / "scenarios" / "public"))
    specs = load_scenarios_dir(public_dir, include_held_out=False)
    assert len(specs) == 30
    manifest = SubmissionManifest(
        submission_id="t",
        system_name="t",
        model_identifier="fake-sut/echo",
        base_url="http://localhost",
        api_key_env="UNSET",
        system_prompt="",
        generation_config={},
        attestation=SubmissionAttestation(True, True, True, True),
        leaderboard_category="bespoke",
    )
    result = dry_run_with_fakes(
        manifest=manifest,
        specs=specs,
        sut_client=EchoFakeSUTClient(),
        user_backend=DeterministicFakeUtteranceClient(),
        paraphrase_seeds=(0,),
    )
    # All 30 arcs ran and got a final score (24 en + 6 zh demo).
    assert len(result.arc_bundles) == 30
    for b in result.arc_bundles:
        assert 0.0 <= b.final_score.final <= 100.0
