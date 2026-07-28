from __future__ import annotations

import json
from pathlib import Path

import pytest

from volvence_zero.state_kv_judge_court import (
    CourtClaimState,
    JudgeCourtState,
    build_judge_court_report,
    load_judge_panel,
)


def _write_panel(
    root: Path,
    *,
    judge: str,
    gate_state: str = "pass",
    prefix_id: str = "prefix-a",
    candidate_ci_low: float = 0.78125,
    control_ci_low: float = 0.3125,
    control_ci_high: float = 0.6875,
) -> Path:
    directory = root / judge.replace("/", "__")
    directory.mkdir(parents=True)
    payload = {
        "schema_version": "state-kv-retention-gate.v1",
        "gate_state": gate_state,
        "claims": [],
        "inputs": [
            {
                "verdict_path": f"{judge}/repair/verdict_identification.json",
                "fingerprint_path": f"{judge}/repair/substrate_fingerprint.json",
                "lane": "p2",
                "p2_pair": "repair-vs-execute",
                "candidate_arm": "state-kv-arm-g-prefix-pure",
                "verdict_state": "retain-strict",
                "c5_grade": "decode-matched",
                "substrate_fingerprint": "Qwen/Qwen2.5-0.5B-Instruct@abc123",
                "judge_model_id": judge,
                "prefix_artifact_id": prefix_id,
                "probe_limit": 0,
                "probe_count": 16,
                "case_count": 32,
                "max_new_tokens": 16,
                "temperature": 0.2,
                "sampling_seed": 1701,
                "stochastic_generation_rollout": True,
                "turn_count": 160,
                "seeded_turn_count": 160,
                "unique_turn_seed_count": 80,
                "candidate": {
                    "correct": 32,
                    "total": 32,
                    "ci_low": candidate_ci_low,
                    "ci_high": 1.0,
                },
                "control": {
                    "correct": 16,
                    "total": 32,
                    "ci_low": control_ci_low,
                    "ci_high": control_ci_high,
                },
            },
            {
                "verdict_path": f"{judge}/boundary/verdict_identification.json",
                "fingerprint_path": f"{judge}/boundary/substrate_fingerprint.json",
                "lane": "p2",
                "p2_pair": "boundary-vs-commit",
                "candidate_arm": "state-kv-arm-g-prefix-pure",
                "verdict_state": "retain-strict",
                "c5_grade": "decode-matched",
                "substrate_fingerprint": "Qwen/Qwen2.5-0.5B-Instruct@abc123",
                "judge_model_id": judge,
                "prefix_artifact_id": prefix_id,
                "probe_limit": 0,
                "probe_count": 16,
                "case_count": 32,
                "max_new_tokens": 16,
                "temperature": 0.2,
                "sampling_seed": 1701,
                "stochastic_generation_rollout": True,
                "turn_count": 160,
                "seeded_turn_count": 160,
                "unique_turn_seed_count": 80,
                "candidate": {
                    "correct": 32,
                    "total": 32,
                    "ci_low": candidate_ci_low,
                    "ci_high": 1.0,
                },
                "control": {
                    "correct": 16,
                    "total": 32,
                    "ci_low": control_ci_low,
                    "ci_high": control_ci_high,
                },
            },
        ],
        "aggregates": [
            {
                "arm": "state-kv-arm-a-pure",
                "correct": 32,
                "total": 64,
                "accuracy": 0.5,
                "ci_low_min": control_ci_low,
                "ci_high_max": control_ci_high,
                "bootstrap_seeds": [20260728],
            },
            {
                "arm": "state-kv-arm-g-prefix-pure",
                "correct": 64,
                "total": 64,
                "accuracy": 1.0,
                "ci_low_min": candidate_ci_low,
                "ci_high_max": 1.0,
                "bootstrap_seeds": [20260728],
            },
        ],
        "required_p2_pairs": ["repair-vs-execute", "boundary-vs-commit"],
        "bootstrap_seeds": [20260728],
        "stochastic_generation_rollout_covered": True,
        "notes": [],
    }
    path = directory / "verdict_retention_gate.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _claim_state(report, name: str) -> CourtClaimState:
    return next(claim.state for claim in report.claims if claim.name == name)


def test_judge_court_passes_two_distinct_retention_panels(tmp_path: Path) -> None:
    first = load_judge_panel(_write_panel(tmp_path, judge="BAAI/bge-m3"))
    second = load_judge_panel(
        _write_panel(tmp_path, judge="sentence-transformers/all-MiniLM-L6-v2")
    )

    report = build_judge_court_report(panels=(first, second), min_judges=2)

    assert report.court_state is JudgeCourtState.PASS
    assert report.judge_model_ids == (
        "BAAI/bge-m3",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    assert _claim_state(report, "claim_multi_judge_coverage") is CourtClaimState.PASS
    assert report.as_json_dict()["schema_version"] == "state-kv-judge-court.v1"


def test_judge_court_requires_distinct_judges(tmp_path: Path) -> None:
    panel = load_judge_panel(_write_panel(tmp_path, judge="BAAI/bge-m3"))

    report = build_judge_court_report(panels=(panel,), min_judges=2)

    assert report.court_state is JudgeCourtState.INSUFFICIENT_DATA
    assert (
        _claim_state(report, "claim_multi_judge_coverage")
        is CourtClaimState.INSUFFICIENT_DATA
    )


def test_judge_court_fails_mixed_artifacts(tmp_path: Path) -> None:
    first = load_judge_panel(
        _write_panel(tmp_path, judge="BAAI/bge-m3", prefix_id="prefix-a")
    )
    second = load_judge_panel(
        _write_panel(tmp_path, judge="other/embedder", prefix_id="prefix-b")
    )

    report = build_judge_court_report(panels=(first, second), min_judges=2)

    assert report.court_state is JudgeCourtState.FAIL
    assert (
        _claim_state(report, "claim_consistent_material")
        is CourtClaimState.FAIL
    )


def test_judge_court_fails_when_a_panel_retention_gate_fails(
    tmp_path: Path,
) -> None:
    first = load_judge_panel(_write_panel(tmp_path, judge="BAAI/bge-m3"))
    second = load_judge_panel(
        _write_panel(tmp_path, judge="other/embedder", gate_state="fail")
    )

    report = build_judge_court_report(panels=(first, second), min_judges=2)

    assert report.court_state is JudgeCourtState.FAIL
    assert _claim_state(report, "claim_panel_retained") is CourtClaimState.FAIL


def test_judge_court_fails_when_second_judge_stays_at_chance(
    tmp_path: Path,
) -> None:
    first = load_judge_panel(_write_panel(tmp_path, judge="BAAI/bge-m3"))
    second = load_judge_panel(
        _write_panel(
            tmp_path,
            judge="sentence-transformers/all-MiniLM-L6-v2",
            candidate_ci_low=0.34375,
        )
    )

    report = build_judge_court_report(panels=(first, second), min_judges=2)

    assert report.court_state is JudgeCourtState.FAIL
    assert (
        _claim_state(report, "claim_court_identification")
        is CourtClaimState.FAIL
    )


def test_judge_panel_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "verdict_retention_gate.json"
    path.write_text(json.dumps({"schema_version": "other"}), encoding="utf-8")

    with pytest.raises(ValueError, match="expected"):
        load_judge_panel(path)
