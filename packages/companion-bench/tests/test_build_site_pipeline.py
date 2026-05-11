# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""End-to-end pipeline test: smoke artifacts → build_site.py → site/data/.

Drives ``submission.dry_run_with_fakes`` for a single scenario, persists the
artifacts the way ``run_real_submission.py`` would, then runs the build_site
pipeline and asserts the resulting JSON matches what the static site expects.
"""

from __future__ import annotations

import importlib.resources as res
import json
import pathlib
import sys

import pytest

import companion_bench
from companion_bench.spec import load_scenarios_dir
from companion_bench.submission import (
    SubmissionAttestation,
    SubmissionManifest,
    dry_run_with_fakes,
    write_submission_summary,
)
from companion_bench.sut_client import EchoFakeSUTClient
from companion_bench.user_simulator import DeterministicFakeUtteranceClient


REPO_ROOT = pathlib.Path(companion_bench.__file__).resolve().parents[3].parent


def _public_specs():
    public_dir = pathlib.Path(str(res.files("companion_bench") / "scenarios" / "public"))
    return load_scenarios_dir(public_dir, include_held_out=False)


def _manifest(submission_id: str, system_name: str) -> SubmissionManifest:
    return SubmissionManifest(
        submission_id=submission_id,
        system_name=system_name,
        model_identifier=f"fake/{submission_id}",
        base_url="http://localhost",
        api_key_env="UNSET",
        system_prompt="(none)",
        generation_config={},
        attestation=SubmissionAttestation(True, True, True, True),
        leaderboard_category="bespoke",
    )


def _run_submission_to_artifact(submission_dir: pathlib.Path, submission_id: str, system_name: str) -> None:
    submission_dir.mkdir(parents=True, exist_ok=True)
    specs = _public_specs()
    # Two scenarios is enough to exercise the pipeline (one F1, one F5 for
    # A6-cap coverage).
    specs = tuple(s for s in specs if s.scenario_id in {"F1-continuity-001", "F5-boundary-001"})
    manifest = _manifest(submission_id, system_name)
    result = dry_run_with_fakes(
        manifest=manifest,
        specs=specs,
        sut_client=EchoFakeSUTClient(),
        user_backend=DeterministicFakeUtteranceClient(),
        paraphrase_seeds=(0,),
        artifact_dir=submission_dir,
    )
    write_submission_summary(result, submission_dir / "summary.json")


def _import_build_site():
    """Import the build_site module from scripts/companion_bench/."""
    scripts_dir = REPO_ROOT / "scripts" / "companion_bench"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import build_site  # noqa: WPS433
    return build_site


def test_build_site_end_to_end(tmp_path: pathlib.Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    site_dir = tmp_path / "site"
    (site_dir / "data").mkdir(parents=True, exist_ok=True)

    _run_submission_to_artifact(artifact_dir / "smoke-A", "smoke-a", "Smoke A")
    _run_submission_to_artifact(artifact_dir / "smoke-B", "smoke-b", "Smoke B")

    build_site = _import_build_site()
    rc = build_site.main([
        "--artifact-dir", str(artifact_dir),
        "--site-dir", str(site_dir),
    ])
    assert rc == 0

    # scenarios.json built and shaped as expected.
    scenarios = json.loads((site_dir / "data" / "scenarios.json").read_text(encoding="utf-8"))
    assert scenarios["companion_bench_version"] == "1.0.0"
    assert scenarios["scenario_count"] == 24
    sample = scenarios["scenarios"][0]
    for key in ("scenario_id", "family", "scenario_hash", "user_simulator", "expected_axes"):
        assert key in sample, f"scenarios.json[0] missing key {key!r}"

    # aggregate_results.json has both submissions.
    aggregate = json.loads((site_dir / "data" / "aggregate_results.json").read_text(encoding="utf-8"))
    assert {row["submission_id"] for row in aggregate["systems"]} == {"smoke-a", "smoke-b"}
    for row in aggregate["systems"]:
        assert row["leaderboard_category"] == "bespoke"
        assert row["axis_means"]["A6"] is not None
        # Trueskill / BT populated by the pairwise pass.
        assert row["trueskill_conservative"] is not None
        assert row["bradley_terry_score"] is not None

    # Per-submission detail JSONs exist with arcs / per-turn rubric / ledger.
    for sid in ("smoke-a", "smoke-b"):
        detail = json.loads((site_dir / "data" / "submissions" / f"{sid}.json").read_text(encoding="utf-8"))
        assert detail["submission_id"] == sid
        assert detail["aggregate"]["arc_count"] == 2
        assert {arc["scenario_id"] for arc in detail["arcs"]} == {"F1-continuity-001", "F5-boundary-001"}
        for arc in detail["arcs"]:
            assert "axis_scores" in arc
            assert "per_turn_rubric" in arc
            assert "callback_ledger" in arc
            assert "sessions" in arc
        # family means populated for the families we exercised.
        assert "F1" in detail["family_means"]
        assert "F5" in detail["family_means"]

    # pairwise.json carries arcs (one match per shared scenario × seed) + ELO.
    pairwise = json.loads((site_dir / "data" / "pairwise.json").read_text(encoding="utf-8"))
    assert len(pairwise["arcs"]) == 2     # 2 shared scenarios × 1 paraphrase × 1 system pair
    assert {entry["system"] for entry in pairwise["elo"]["trueskill"]} == {"smoke-a", "smoke-b"}
    assert {entry["system"] for entry in pairwise["elo"]["bradley_terry"]} == {"smoke-a", "smoke-b"}
