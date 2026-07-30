from __future__ import annotations

import json

import pytest

from volvence_zero.agent.gate78_shared_trace import (
    export_gate78_shared_trace_bundle,
)
from volvence_zero.agent.gate8_wake_sleep_evidence import (
    GATE8_ARMS,
    GATE8_REQUIRED_FILES,
    export_gate8_evidence_bundle,
    run_gate8_evidence,
    verify_gate8_evidence_bundle,
)


@pytest.fixture
def trace_root(tmp_path):
    root = tmp_path / "trace"
    export_gate78_shared_trace_bundle(output_dir=root)
    return root


def test_gate8_development_run_uses_four_owner_controls_without_locked(
    trace_root,
) -> None:
    report = run_gate8_evidence(
        trace_root=trace_root,
        seed_schedule=(701,),
        partition="trace-development-heldout",
        evaluation_limit=2,
        formal_locked_run=False,
    )

    assert report.arm_schedule == GATE8_ARMS
    assert report.formal_locked_run is False
    assert report.partition == "trace-development-heldout"
    assert {row.arm for row in report.arm_results} == set(GATE8_ARMS)
    assert all(row.prompt_token_increment == 0 for row in report.arm_results)
    assert all(
        row.worker_execution_count == row.unique_job_count
        for row in report.arm_results
    )
    assert all(
        row.duplicate_job_execution_count == 0
        for row in report.arm_results
    )
    assert all(
        row.owner_writeback_lineage_coverage == 1.0
        for row in report.arm_results
    )
    assert all(row.rollback_exact for row in report.arm_results)
    full = next(
        row
        for row in report.arm_results
        if row.arm == "sleep-consolidation"
    )
    no_sleep = next(
        row for row in report.arm_results if row.arm == "no-sleep"
    )
    memory_only = next(
        row
        for row in report.arm_results
        if row.arm == "memory-only-sleep"
    )
    policy_only = next(
        row
        for row in report.arm_results
        if row.arm == "policy-only-sleep"
    )
    assert full.memory_entry_count == 2
    assert full.temporal_operation_count > 0
    assert no_sleep.memory_entry_count == 0
    assert no_sleep.temporal_operation_count == 0
    assert memory_only.memory_entry_count == 2
    assert memory_only.temporal_operation_count == 0
    assert policy_only.memory_entry_count == 0
    assert policy_only.temporal_operation_count > 0
    assert full.delayed_payoff > memory_only.delayed_payoff
    assert full.delayed_payoff > policy_only.delayed_payoff


def test_gate8_development_rejects_locked_partition(trace_root) -> None:
    with pytest.raises(ValueError, match="must not consume locked"):
        run_gate8_evidence(
            trace_root=trace_root,
            seed_schedule=(701,),
            partition="trace-locked-confirmation",
            formal_locked_run=False,
            evaluation_limit=1,
        )


def test_gate8_bundle_exports_required_files(
    trace_root,
    tmp_path,
) -> None:
    report = run_gate8_evidence(
        trace_root=trace_root,
        seed_schedule=(701,),
        partition="trace-development-heldout",
        evaluation_limit=1,
        formal_locked_run=False,
    )
    output_dir = tmp_path / "bundle"
    export_gate8_evidence_bundle(report, output_dir=output_dir)
    verification = verify_gate8_evidence_bundle(output_dir)

    assert verification["passed"] is True
    assert {path.name for path in output_dir.iterdir()} == set(
        GATE8_REQUIRED_FILES
    )
    verdict = json.loads(
        (output_dir / "promotion_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    assert verdict["retuning_allowed"] is False
