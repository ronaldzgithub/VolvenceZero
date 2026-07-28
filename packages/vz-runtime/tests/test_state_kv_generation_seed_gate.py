from __future__ import annotations

import json
from pathlib import Path

import pytest

from volvence_zero.state_kv_generation_seed_gate import (
    GenerationSeedGateState,
    SeedGateClaimState,
    build_generation_seed_gate_report,
    load_generation_seed_panel,
)


def _write_report(
    root: Path,
    *,
    seed: int,
    gate_state: str = "pass",
    candidate_ci_low: float = 0.75,
    prefix_id: str = "prefix-a",
) -> Path:
    directory = root / str(seed)
    directory.mkdir(parents=True)
    inputs = []
    for pair in ("repair-vs-execute", "boundary-vs-commit"):
        inputs.append(
            {
                "lane": "p2",
                "p2_pair": pair,
                "candidate_arm": "state-kv-arm-g-prefix-pure",
                "substrate_fingerprint": "Qwen@abc",
                "judge_model_id": "BAAI/bge-m3",
                "prefix_artifact_id": prefix_id,
                "probe_limit": 0,
                "probe_count": 16,
                "case_count": 32,
                "max_new_tokens": 16,
                "temperature": 0.2,
                "sampling_seed": seed,
                "stochastic_generation_rollout": True,
            }
        )
    payload = {
        "schema_version": "state-kv-retention-gate.v1",
        "gate_state": gate_state,
        "inputs": inputs,
        "aggregates": [
            {
                "arm": "state-kv-arm-a-pure",
                "correct": 32,
                "total": 64,
                "accuracy": 0.5,
                "ci_low_min": 0.375,
                "ci_high_max": 0.625,
            },
            {
                "arm": "state-kv-arm-g-prefix-pure",
                "correct": 56,
                "total": 64,
                "accuracy": 0.875,
                "ci_low_min": candidate_ci_low,
                "ci_high_max": 0.953,
            },
        ],
        "required_p2_pairs": ["repair-vs-execute", "boundary-vs-commit"],
        "stochastic_generation_rollout_covered": True,
    }
    path = directory / "verdict_retention_gate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _claim(report, name: str) -> SeedGateClaimState:
    return next(item.state for item in report.claims if item.name == name)


def test_generation_seed_gate_passes_three_distinct_rollouts(
    tmp_path: Path,
) -> None:
    panels = tuple(
        load_generation_seed_panel(_write_report(tmp_path, seed=seed))
        for seed in (1701, 1702, 1703)
    )

    report = build_generation_seed_gate_report(panels=panels)

    assert report.gate_state is GenerationSeedGateState.PASS
    assert report.generation_seeds == (1701, 1702, 1703)
    assert report.aggregate_candidate.correct == 168
    assert report.aggregate_candidate.total == 192
    assert report.aggregate_candidate.ci_low > 0.5
    assert report.aggregate_control.ci_low <= 0.5 <= report.aggregate_control.ci_high


def test_generation_seed_gate_requires_three_distinct_seeds(
    tmp_path: Path,
) -> None:
    panels = tuple(
        load_generation_seed_panel(_write_report(tmp_path, seed=seed))
        for seed in (1701, 1702)
    )

    report = build_generation_seed_gate_report(panels=panels)

    assert report.gate_state is GenerationSeedGateState.INSUFFICIENT_DATA
    assert (
        _claim(report, "claim_generation_seed_coverage")
        is SeedGateClaimState.INSUFFICIENT_DATA
    )


def test_generation_seed_gate_fails_one_weak_seed(tmp_path: Path) -> None:
    panels = (
        load_generation_seed_panel(_write_report(tmp_path, seed=1701)),
        load_generation_seed_panel(_write_report(tmp_path, seed=1702)),
        load_generation_seed_panel(
            _write_report(tmp_path, seed=1703, candidate_ci_low=0.45)
        ),
    )

    report = build_generation_seed_gate_report(panels=panels)

    assert report.gate_state is GenerationSeedGateState.FAIL
    assert (
        _claim(report, "claim_cross_seed_identification")
        is SeedGateClaimState.FAIL
    )


def test_generation_seed_gate_rejects_duplicate_seed_panels(
    tmp_path: Path,
) -> None:
    first_path = _write_report(tmp_path, seed=1701)
    first = load_generation_seed_panel(first_path)
    second = load_generation_seed_panel(first_path)
    third = load_generation_seed_panel(_write_report(tmp_path, seed=1702))

    report = build_generation_seed_gate_report(
        panels=(first, second, third),
        min_generation_seeds=2,
    )

    assert report.gate_state is GenerationSeedGateState.FAIL
    assert (
        _claim(report, "claim_generation_seed_coverage")
        is SeedGateClaimState.FAIL
    )


def test_generation_seed_gate_fails_mixed_artifacts(tmp_path: Path) -> None:
    panels = (
        load_generation_seed_panel(_write_report(tmp_path, seed=1701)),
        load_generation_seed_panel(
            _write_report(tmp_path, seed=1702, prefix_id="prefix-b")
        ),
        load_generation_seed_panel(_write_report(tmp_path, seed=1703)),
    )

    report = build_generation_seed_gate_report(panels=panels)

    assert report.gate_state is GenerationSeedGateState.FAIL
    assert (
        _claim(report, "claim_consistent_material")
        is SeedGateClaimState.FAIL
    )


def test_generation_seed_panel_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"schema_version": "other"}), encoding="utf-8")

    with pytest.raises(ValueError, match="expected"):
        load_generation_seed_panel(path)
