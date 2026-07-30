from __future__ import annotations

import pytest

from volvence_zero.agent.gate78_shared_trace import (
    export_gate78_shared_trace_bundle,
)
from volvence_zero.agent.gate_v2_retest_common import (
    GATE_V2_REQUIRED_FILES,
)
from volvence_zero.agent.gate1_pe_causal_v2_retest import (
    export_gate1_v2_bundle,
    run_gate1_v2_retest,
    verify_gate1_v2_bundle,
)
from volvence_zero.agent.gate4_active_learning_v2_retest import (
    export_gate4_v2_bundle,
    run_gate4_v2_retest,
    verify_gate4_v2_bundle,
)
from volvence_zero.agent.gate6_meta_init_v2_retest import (
    export_gate6_v2_bundle,
    run_gate6_v2_retest,
    verify_gate6_v2_bundle,
)


@pytest.fixture
def trace_root(tmp_path):
    root = tmp_path / "trace"
    export_gate78_shared_trace_bundle(output_dir=root)
    return root


def test_gate1_v2_keeps_pe_drive_matched_and_rollback_exact(
    trace_root,
) -> None:
    report = run_gate1_v2_retest(
        trace_root=trace_root,
        seed_schedule=(701,),
        evaluation_limit=1,
    )
    assert {row.arm for row in report.results} == {
        "pe-eta-v2",
        "pe-drive-off-v2",
    }
    assert all(row.lineage_coverage == 1.0 for row in report.results)
    assert all(row.rollback_exact for row in report.results)
    full = next(row for row in report.results if row.arm == "pe-eta-v2")
    disabled = next(
        row for row in report.results if row.arm == "pe-drive-off-v2"
    )
    assert full.pe_applied_count == 1
    assert full.temporal_parameter_change_count == 1
    assert disabled.pe_applied_count == 0
    assert disabled.temporal_parameter_change_count == 0


def test_gate4_v2_uses_all_preregistered_controls(trace_root) -> None:
    report = run_gate4_v2_retest(
        trace_root=trace_root,
        seed_schedule=(701,),
    )
    assert len(report.results) == 5
    assert all(
        row.typed_candidate_lineage_coverage == 1.0
        for row in report.results
    )
    assert all(row.isolated_reset_exact for row in report.results)


def test_gate6_v2_uses_owner_init_and_exact_rollback(trace_root) -> None:
    report = run_gate6_v2_retest(
        trace_root=trace_root,
        seed_schedule=(701,),
        evaluation_limit=1,
    )
    assert len(report.results) == 6
    assert all(row.lineage_complete for row in report.results)
    assert all(row.fact_leakage_count == 0 for row in report.results)
    assert all(row.rollback_exact for row in report.results)


@pytest.mark.parametrize(
    ("runner", "kwargs"),
    (
        (run_gate1_v2_retest, {"evaluation_limit": 1}),
        (run_gate4_v2_retest, {}),
        (run_gate6_v2_retest, {"evaluation_limit": 1}),
    ),
)
def test_v2_development_rejects_locked_partition(
    trace_root,
    runner,
    kwargs,
) -> None:
    with pytest.raises(ValueError, match="must not consume locked"):
        runner(
            trace_root=trace_root,
            seed_schedule=(701,),
            partition="trace-locked-confirmation",
            formal_locked_run=False,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("runner", "exporter", "verifier", "kwargs"),
    (
        (
            run_gate1_v2_retest,
            export_gate1_v2_bundle,
            verify_gate1_v2_bundle,
            {"evaluation_limit": 1},
        ),
        (
            run_gate4_v2_retest,
            export_gate4_v2_bundle,
            verify_gate4_v2_bundle,
            {},
        ),
        (
            run_gate6_v2_retest,
            export_gate6_v2_bundle,
            verify_gate6_v2_bundle,
            {"evaluation_limit": 1},
        ),
    ),
)
def test_v2_bundle_has_exact_required_surface(
    trace_root,
    tmp_path,
    runner,
    exporter,
    verifier,
    kwargs,
) -> None:
    report = runner(
        trace_root=trace_root,
        seed_schedule=(701,),
        **kwargs,
    )
    output = tmp_path / "bundle"
    exporter(report, output_dir=output)
    assert verifier(output)["passed"] is True
    assert {path.name for path in output.iterdir()} == set(
        GATE_V2_REQUIRED_FILES
    )
