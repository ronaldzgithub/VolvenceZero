from __future__ import annotations

import json

import pytest

from volvence_zero.agent.gate78_shared_trace import (
    export_gate78_shared_trace_bundle,
)
from volvence_zero.agent.gate7_causal_takeover_evidence import (
    GATE7_ARMS,
    GATE7_REQUIRED_FILES,
    export_gate7_evidence_bundle,
    run_gate7_evidence,
    verify_gate7_evidence_bundle,
)


@pytest.fixture
def trace_root(tmp_path):
    root = tmp_path / "trace"
    export_gate78_shared_trace_bundle(output_dir=root)
    return root


def test_gate7_development_run_uses_all_five_arms_without_locked(
    trace_root,
) -> None:
    report = run_gate7_evidence(
        trace_root=trace_root,
        seed_schedule=(701,),
        partition="trace-development-heldout",
        controller_dim=8,
        ssl_updates=1,
        rl_cycles=1,
        train_limit=2,
        evaluation_limit=2,
        formal_locked_run=False,
    )

    assert report.arm_schedule == GATE7_ARMS
    assert report.formal_locked_run is False
    assert report.partition == "trace-development-heldout"
    assert {row.arm for row in report.arm_results} == set(GATE7_ARMS)
    assert all(row.future_residual_leakage_count == 0 for row in report.arm_results)
    assert all(row.token_space_rl_mutation_count == 0 for row in report.arm_results)
    assert all(row.rollback_exact for row in report.arm_results)
    full = next(row for row in report.arm_results if row.arm == "full")
    no_ssl = next(row for row in report.arm_results if row.arm == "no-ssl")
    no_rl = next(row for row in report.arm_results if row.arm == "no-rl")
    joint_unfrozen = next(
        row for row in report.arm_results if row.arm == "joint-unfrozen"
    )
    assert full.ssl_update_count == 1
    assert full.rl_update_count in {0, 1}
    assert full.structure_fingerprint_change_during_rl == 0
    assert no_ssl.ssl_update_count == 0
    assert no_ssl.rl_update_count == 1
    assert no_rl.ssl_update_count == 1
    assert no_rl.rl_update_count == 0
    assert joint_unfrozen.structure_fingerprint_change_during_rl == 1


def test_gate7_development_rejects_locked_partition(trace_root) -> None:
    with pytest.raises(ValueError, match="must not consume locked"):
        run_gate7_evidence(
            trace_root=trace_root,
            seed_schedule=(701,),
            partition="trace-locked-confirmation",
            formal_locked_run=False,
            train_limit=1,
            evaluation_limit=1,
        )


def test_gate7_bundle_exports_required_files(
    trace_root,
    tmp_path,
) -> None:
    report = run_gate7_evidence(
        trace_root=trace_root,
        seed_schedule=(701,),
        partition="trace-development-heldout",
        ssl_updates=1,
        rl_cycles=1,
        train_limit=1,
        evaluation_limit=1,
        formal_locked_run=False,
    )
    output_dir = tmp_path / "bundle"
    export_gate7_evidence_bundle(report, output_dir=output_dir)
    verification = verify_gate7_evidence_bundle(output_dir)

    assert verification["passed"] is True
    assert {path.name for path in output_dir.iterdir()} == set(
        GATE7_REQUIRED_FILES
    )
    verdict = json.loads(
        (output_dir / "promotion_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    assert verdict["retuning_allowed"] is False
