from __future__ import annotations

from lifeform_domain_emogpt.lab import (
    RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME,
    relationship_transfer_package_dir,
)
from lifeform_evolution.relationship_lab_gate0 import (
    Gate0CalibrationConfig,
    GateCheckStatus,
    run_relationship_gate0_calibration,
)


def test_v2_gate0_machinery_passes_without_claiming_real_baseline() -> None:
    report = run_relationship_gate0_calibration(
        package_root=relationship_transfer_package_dir(
            RELATIONSHIP_TRANSFER_V2_PACKAGE_NAME
        ),
        config=Gate0CalibrationConfig(samples_per_action=64),
        created_at_iso="2026-08-20T00:00:00+00:00",
    )
    assert report.machinery_ready
    assert not report.gate0_passed
    checks = {item.check_id: item for item in report.checks}
    assert checks["mirrored_counterfactual"].status is GateCheckStatus.PASS
    assert dict(checks["mirrored_counterfactual"].metrics) == {
        "mirrored_pairs": 6,
        "surface_families": 6,
        "byte_identical": True,
        "opposite_actions": True,
    }
    assert checks["reactive_action_effect"].status is GateCheckStatus.PASS
    assert checks["environment_determinism"].status is GateCheckStatus.PASS
    assert checks["sut_truth_leakage"].status is GateCheckStatus.PASS
    assert checks["decision_trace_contract"].status is GateCheckStatus.PASS
    assert checks["frozen_baseline_non_saturation"].status is GateCheckStatus.PENDING
