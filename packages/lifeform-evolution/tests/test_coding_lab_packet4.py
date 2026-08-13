"""Packet 4 门评审机制测试：ALLOW / BLOCK / 回滚往返。"""

from __future__ import annotations

import pathlib

from volvence_zero.credit.gate import GateDecision

from lifeform_evolution.coding_lab_packet4 import (
    GENESIS_HASH,
    CodingArtifactPointer,
    build_coding_modification_gate_review,
    read_registry,
    rollback_registry,
    verify_pointer_round_trip,
    write_registry,
)


def _candidate_report(
    *,
    admitted: bool = True,
    gain_ci_lower: float = 0.31,
    zero_code: bool = True,
    executor_changed: bool = False,
) -> dict:
    return {
        "admission": {"admitted": admitted},
        "aggregate": {"gain_vs_noop_ci_lower_min": gain_ci_lower},
        "free_bias_present": False,
        "zero_code_strict_noop": zero_code,
        "substrate_trainable_parameter_count": 0,
        "reader_parameters_changed": False,
        "executor_parameters_changed": executor_changed,
        "production_wiring_changed": False,
        "feedback_to_learning": False,
    }


def test_admitted_candidate_allows() -> None:
    review = build_coding_modification_gate_review(
        candidate_report=_candidate_report(),
        candidate_manifest_sha256="c" * 64,
        candidate_report_sha256="r" * 64,
        incumbent=None,
    )
    assert review.decision is GateDecision.ALLOW
    assert review.blocking_reasons == ()
    assert review.old_value_hash == GENESIS_HASH
    assert review.validation_delta == 0.31


def test_unadmitted_candidate_blocks() -> None:
    review = build_coding_modification_gate_review(
        candidate_report=_candidate_report(admitted=False),
        candidate_manifest_sha256="c" * 64,
        candidate_report_sha256="r" * 64,
        incumbent=None,
    )
    assert review.decision is GateDecision.BLOCK
    assert review.contract_integrity == 0.0


def test_thin_validation_delta_blocks_offline_gate() -> None:
    # OFFLINE 档要求 validation_delta ≥ 0.05（cognition 门常量）。
    review = build_coding_modification_gate_review(
        candidate_report=_candidate_report(gain_ci_lower=0.01),
        candidate_manifest_sha256="c" * 64,
        candidate_report_sha256="r" * 64,
        incumbent=None,
    )
    assert review.decision is GateDecision.BLOCK
    assert any("validation" in reason for reason in review.blocking_reasons)


def test_structural_breach_blocks() -> None:
    review = build_coding_modification_gate_review(
        candidate_report=_candidate_report(executor_changed=True),
        candidate_manifest_sha256="c" * 64,
        candidate_report_sha256="r" * 64,
        incumbent=None,
    )
    assert review.decision is GateDecision.BLOCK


def test_pointer_round_trip_and_rollback(tmp_path: pathlib.Path) -> None:
    registry = tmp_path / "active_artifact.json"
    incumbent = CodingArtifactPointer(
        run_id="run-a", manifest_sha256="a" * 64, report_sha256="b" * 64
    )
    assert verify_pointer_round_trip(incumbent)
    write_registry(registry, incumbent)
    assert read_registry(registry) == incumbent

    candidate = CodingArtifactPointer(
        run_id="run-b", manifest_sha256="c" * 64, report_sha256="d" * 64
    )
    write_registry(registry, candidate)
    assert read_registry(registry) == candidate

    rollback_registry(registry, incumbent)
    assert read_registry(registry) == incumbent

    # Genesis rollback removes the pointer entirely.
    rollback_registry(registry, None)
    assert read_registry(registry) is None
