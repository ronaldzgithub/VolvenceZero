from __future__ import annotations

from volvence_zero.agent.gate9_bounded_selfmod_evidence import (
    GATE9_MEMORY_ARMS,
    GATE9_OPTIMIZER_ARMS,
    GATE9_OPTIMIZER_SCENARIOS,
    GATE9_REQUIRED_FILES,
    GATE9_SEEDS,
    export_gate9_evidence_bundle,
    run_gate9_evidence,
    verify_gate9_evidence_bundle,
)


def test_gate9_runs_both_matched_control_suites_on_owner_surfaces() -> None:
    report = run_gate9_evidence(seed_schedule=(GATE9_SEEDS[0],))

    assert {
        row.arm for row in report.optimizer_results
    } == set(GATE9_OPTIMIZER_ARMS)
    assert {
        row.scenario for row in report.optimizer_results
    } == set(GATE9_OPTIMIZER_SCENARIOS)
    assert {
        row.arm for row in report.memory_results
    } == set(GATE9_MEMORY_ARMS)
    assert all(row.rollback_exact for row in report.optimizer_results)
    assert all(row.rollback_exact for row in report.memory_results)
    assert all(
        row.frozen_substrate_mutation_count == 0
        for row in report.memory_results
    )
    assert all(
        row.pe_lineage_mismatch_count == 0
        for row in report.memory_results
    )
    assert report.verdict in {
        "invalid",
        "not-supported",
        "causal-supported",
    }


def test_gate9_formal_bundle_has_required_files(tmp_path) -> None:
    report = run_gate9_evidence()
    export_gate9_evidence_bundle(report, output_dir=tmp_path)
    verification = verify_gate9_evidence_bundle(tmp_path)

    assert verification["passed"] is True
    assert {path.name for path in tmp_path.iterdir()} == set(
        GATE9_REQUIRED_FILES
    )
