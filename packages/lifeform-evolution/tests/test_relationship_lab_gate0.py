from __future__ import annotations

import json

from lifeform_domain_emogpt.lab import (
    load_relationship_transfer_dataset,
    sha256_json,
)
from lifeform_evolution.relationship_lab_gate0 import (
    FrozenBaselineAttestation,
    Gate0CalibrationConfig,
    GateCheckStatus,
    load_frozen_baseline_attestation,
    run_relationship_gate0_calibration,
    write_relationship_gate0_report,
)


_CREATED_AT = "2026-08-19T01:00:00+00:00"


def _baseline(*, correct: int = 12, total: int = 24) -> FrozenBaselineAttestation:
    dataset = load_relationship_transfer_dataset()
    return FrozenBaselineAttestation(
        arm_id="stateless",
        dataset_fingerprint=dataset.dataset_fingerprint,
        model_id="frozen-test-substrate",
        weights_sha256=sha256_json("weights"),
        prompt_sha256=sha256_json("prompt"),
        generation_config_sha256=sha256_json("generation"),
        seed_schedule_sha256=sha256_json("seeds"),
        decision_ledger_sha256=sha256_json("ledger"),
        evaluated_split="calibration",
        valid_decisions=total,
        correct_decisions=correct,
        evaluated_decisions=total,
        context_tokens_total=800,
        hidden_test_opened=False,
        frozen_at_iso="2026-08-19T00:00:00+00:00",
    )


def test_gate0_machinery_passes_but_baseline_tooth_stays_pending() -> None:
    report = run_relationship_gate0_calibration(
        config=Gate0CalibrationConfig(samples_per_action=64),
        created_at_iso=_CREATED_AT,
    )
    assert report.machinery_ready
    assert not report.gate0_passed
    statuses = {check.check_id: check.status for check in report.checks}
    assert statuses["mirrored_counterfactual"] is GateCheckStatus.PASS
    assert statuses["reactive_action_effect"] is GateCheckStatus.PASS
    assert statuses["environment_determinism"] is GateCheckStatus.PASS
    assert statuses["sut_truth_leakage"] is GateCheckStatus.PASS
    assert statuses["decision_trace_contract"] is GateCheckStatus.PASS
    assert statuses["frozen_baseline_non_saturation"] is GateCheckStatus.PENDING


def test_frozen_non_saturated_baseline_closes_gate0() -> None:
    baseline = _baseline()
    report = run_relationship_gate0_calibration(
        config=Gate0CalibrationConfig(samples_per_action=64),
        baseline=baseline,
        created_at_iso=_CREATED_AT,
    )
    assert report.machinery_ready
    assert report.gate0_passed
    assert report.baseline_attestation_id == baseline.artifact_id
    status = {check.check_id: check.status for check in report.checks}
    assert status["frozen_baseline_non_saturation"] is GateCheckStatus.PASS


def test_valid_stateless_abstention_has_no_artificial_accuracy_floor() -> None:
    report = run_relationship_gate0_calibration(
        config=Gate0CalibrationConfig(samples_per_action=64),
        baseline=_baseline(correct=0, total=24),
        created_at_iso=_CREATED_AT,
    )
    assert report.machinery_ready
    assert report.gate0_passed
    check = next(item for item in report.checks if item.check_id == "frozen_baseline_non_saturation")
    assert dict(check.metrics)["accuracy"] == 0.0


def test_saturated_baseline_fails_gate0_without_changing_machinery_verdict() -> None:
    report = run_relationship_gate0_calibration(
        config=Gate0CalibrationConfig(samples_per_action=64),
        baseline=_baseline(correct=24, total=24),
        created_at_iso=_CREATED_AT,
    )
    assert report.machinery_ready
    assert not report.gate0_passed
    status = {check.check_id: check.status for check in report.checks}
    assert status["frozen_baseline_non_saturation"] is GateCheckStatus.FAIL


def test_baseline_attestation_round_trip_and_tamper_detection(tmp_path) -> None:
    baseline = _baseline()
    path = tmp_path / "baseline.json"
    path.write_text(baseline.to_json(), encoding="utf-8")
    assert load_frozen_baseline_attestation(path) == baseline

    tampered = json.loads(baseline.to_json())
    tampered["correct_decisions"] = 23
    path.write_text(json.dumps(tampered), encoding="utf-8")
    try:
        load_frozen_baseline_attestation(path)
    except ValueError as exc:
        assert "artifact_id does not match" in str(exc)
    else:
        raise AssertionError("tampered baseline attestation must fail loudly")


def test_report_writer_emits_content_addressed_json_and_markdown(tmp_path) -> None:
    report = run_relationship_gate0_calibration(
        config=Gate0CalibrationConfig(samples_per_action=64),
        created_at_iso=_CREATED_AT,
    )
    json_path, markdown_path = write_relationship_gate0_report(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["artifact_id"] == report.artifact_id
    assert payload["verdicts"] == {
        "gate0_passed": False,
        "machinery_ready": True,
    }
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "machinery_ready: **true**" in markdown
    assert "gate0_passed: **false**" in markdown
