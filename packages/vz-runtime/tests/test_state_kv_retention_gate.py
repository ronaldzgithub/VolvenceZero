from __future__ import annotations

import json
from pathlib import Path

import pytest

from volvence_zero.state_kv_retention_gate import (
    GateClaimState,
    RetentionGateState,
    build_retention_gate_report,
    load_retention_evidence,
)


def _write_case(
    root: Path,
    *,
    pair: str,
    prefix_id: str = "prefix-a",
    substrate: str = "Qwen/Qwen2.5-0.5B-Instruct@abc123",
    judge: str = "BAAI/bge-m3",
    verdict_state: str = "retain-strict",
    candidate_correct: int = 29,
    candidate_total: int = 32,
    control_correct: int = 16,
    control_total: int = 32,
    temperature: float = 0.0,
    sampling_seed: int | None = None,
) -> Path:
    directory = root / pair
    directory.mkdir(parents=True)
    verdict = {
        "schema_version": "state-kv-identification.v1",
        "verdict_state": verdict_state,
        "substrate_kind": "frozen-weights",
        "substrate_fingerprint": substrate,
        "candidate_arm": "state-kv-arm-g-prefix-pure",
        "claims": [],
        "c5_grade": "decode-matched",
        "c5_detail": "decode_fp identical across users on every probe",
        "matching": [
            {
                "arm": "state-kv-arm-a-pure",
                "correct": control_correct,
                "total": control_total,
                "accuracy": control_correct / control_total,
                "ci_low": 0.3125,
                "ci_high": 0.6875,
                "judge_model_id": judge,
            },
            {
                "arm": "state-kv-arm-g-prefix-pure",
                "correct": candidate_correct,
                "total": candidate_total,
                "accuracy": candidate_correct / candidate_total,
                "ci_low": 0.71875,
                "ci_high": 1.0,
                "judge_model_id": judge,
            },
        ],
        "prompt_fp_table": [
            {
                "arm": "state-kv-arm-a-pure",
                "user": f"{pair}-a",
                "probe": "h0",
                "prompt_fp": "prompt-a",
                "prompt_state_sections": 0,
                "decode_fp": "decode-a",
                **(
                    {"sampling_seed": 101}
                    if sampling_seed is not None
                    else {}
                ),
                "conditioning_applied": False,
                "conditioning_delivered": False,
            },
            {
                "arm": "state-kv-arm-a-pure",
                "user": f"{pair}-b",
                "probe": "h0",
                "prompt_fp": "prompt-a",
                "prompt_state_sections": 0,
                "decode_fp": "decode-a",
                **(
                    {"sampling_seed": 101}
                    if sampling_seed is not None
                    else {}
                ),
                "conditioning_applied": False,
                "conditioning_delivered": False,
            },
            {
                "arm": "state-kv-arm-g-prefix-pure",
                "user": f"{pair}-a",
                "probe": "h0",
                "prompt_fp": "prompt-a",
                "prompt_state_sections": 0,
                "decode_fp": "decode-g",
                **(
                    {"sampling_seed": 202}
                    if sampling_seed is not None
                    else {}
                ),
                "conditioning_applied": True,
                "conditioning_delivered": True,
            },
            {
                "arm": "state-kv-arm-g-prefix-pure",
                "user": f"{pair}-b",
                "probe": "h0",
                "prompt_fp": "prompt-a",
                "prompt_state_sections": 0,
                "decode_fp": "decode-g",
                **(
                    {"sampling_seed": 202}
                    if sampling_seed is not None
                    else {}
                ),
                "conditioning_applied": True,
                "conditioning_delivered": True,
            },
        ],
        "judge_model_id": judge,
        "notes": [],
    }
    fingerprint = {
        "schema_version": "state-kv-substrate-fingerprint.v1",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "weights_sha256": "abc123",
        "personal_conditioning_prefix_id": prefix_id,
        "identification_material": {
            "lane": "p2",
            "p2_pair": pair,
            "user_ids": [f"{pair}-a", f"{pair}-b"],
            "probe_ids": ["h0"],
            "case_count": 2,
            "temperature": temperature,
            "sampling_seed": sampling_seed,
        },
    }
    verdict_path = directory / "verdict_identification.json"
    verdict_path.write_text(
        json.dumps(verdict, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "substrate_fingerprint.json").write_text(
        json.dumps(fingerprint, ensure_ascii=False), encoding="utf-8"
    )
    return verdict_path


def _claim_state(report, name: str) -> GateClaimState:
    return next(claim.state for claim in report.claims if claim.name == name)


def test_retention_gate_passes_on_two_heldout_pairs(tmp_path: Path) -> None:
    first = load_retention_evidence(
        _write_case(tmp_path, pair="repair-vs-execute", candidate_correct=29)
    )
    second = load_retention_evidence(
        _write_case(tmp_path, pair="boundary-vs-commit", candidate_correct=27)
    )

    report = build_retention_gate_report(
        evidences=(first, second),
        required_p2_pairs=("repair-vs-execute", "boundary-vs-commit"),
        bootstrap_seeds=(7, 11, 13),
    )

    assert report.gate_state is RetentionGateState.PASS
    assert report.stochastic_generation_rollout_covered is False
    candidate = next(
        item
        for item in report.aggregates
        if item.arm_label == "state-kv-arm-g-prefix-pure"
    )
    assert candidate.correct == 56
    assert candidate.total == 64
    assert candidate.ci_low_min > 0.5
    assert report.inputs[0].temperature == 0.0
    assert report.inputs[0].sampling_seed is None


def test_retention_gate_marks_stochastic_rollout_coverage(tmp_path: Path) -> None:
    first = load_retention_evidence(
        _write_case(
            tmp_path,
            pair="repair-vs-execute",
            candidate_correct=29,
            temperature=0.2,
            sampling_seed=1701,
        )
    )
    second = load_retention_evidence(
        _write_case(
            tmp_path,
            pair="boundary-vs-commit",
            candidate_correct=27,
            temperature=0.2,
            sampling_seed=1701,
        )
    )

    report = build_retention_gate_report(
        evidences=(first, second),
        required_p2_pairs=("repair-vs-execute", "boundary-vs-commit"),
        bootstrap_seeds=(7, 11, 13),
    )

    assert report.gate_state is RetentionGateState.PASS
    assert report.stochastic_generation_rollout_covered is True
    assert all(item.seeded_turn_count == item.turn_count for item in report.inputs)
    assert all(item.unique_turn_seed_count == 2 for item in report.inputs)


def test_retention_gate_reports_missing_pair_as_insufficient_data(
    tmp_path: Path,
) -> None:
    evidence = load_retention_evidence(
        _write_case(tmp_path, pair="repair-vs-execute")
    )

    report = build_retention_gate_report(
        evidences=(evidence,),
        required_p2_pairs=("repair-vs-execute", "boundary-vs-commit"),
    )

    assert report.gate_state is RetentionGateState.INSUFFICIENT_DATA
    assert (
        _claim_state(report, "claim_heldout_pair_coverage")
        is GateClaimState.INSUFFICIENT_DATA
    )


def test_retention_gate_fails_mixed_artifacts(tmp_path: Path) -> None:
    first = load_retention_evidence(
        _write_case(tmp_path, pair="repair-vs-execute", prefix_id="prefix-a")
    )
    second = load_retention_evidence(
        _write_case(tmp_path, pair="boundary-vs-commit", prefix_id="prefix-b")
    )

    report = build_retention_gate_report(
        evidences=(first, second),
        required_p2_pairs=("repair-vs-execute", "boundary-vs-commit"),
    )

    assert report.gate_state is RetentionGateState.FAIL
    assert (
        _claim_state(report, "claim_consistent_artifact")
        is GateClaimState.FAIL
    )


def test_retention_gate_fails_mixed_rollout_seed(tmp_path: Path) -> None:
    first = load_retention_evidence(
        _write_case(
            tmp_path,
            pair="repair-vs-execute",
            temperature=0.2,
            sampling_seed=1701,
        )
    )
    second = load_retention_evidence(
        _write_case(
            tmp_path,
            pair="boundary-vs-commit",
            temperature=0.2,
            sampling_seed=31337,
        )
    )

    report = build_retention_gate_report(
        evidences=(first, second),
        required_p2_pairs=("repair-vs-execute", "boundary-vs-commit"),
    )

    assert report.gate_state is RetentionGateState.FAIL
    assert (
        _claim_state(report, "claim_consistent_artifact")
        is GateClaimState.FAIL
    )


def test_retention_gate_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "verdict_identification.json"
    path.write_text(
        json.dumps({"schema_version": "other"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="expected"):
        load_retention_evidence(path)
