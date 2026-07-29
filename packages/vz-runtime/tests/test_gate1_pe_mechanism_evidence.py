from __future__ import annotations

import json

from volvence_zero.agent.gate1_pe_mechanism_evidence import (
    GATE1_PE_MECHANISM_REQUIRED_FILES,
    GATE1_PE_MECHANISM_SCHEMA_VERSION,
    audit_prediction_lineage,
    export_gate1_pe_mechanism_bundle,
)


def _lineage_row(
    prediction_id: str,
    *,
    outcome_id: str = "outcome-1",
) -> dict[str, object]:
    return {
        "prediction_id": prediction_id,
        "environment_event_id": "event-1",
        "environment_outcome_id": outcome_id,
        "observed_at": "2026-07-30T00:00:00+00:00",
    }


def test_gate1_mechanism_bundle_exports_preregistered_contract(
    tmp_path,
) -> None:
    paths = export_gate1_pe_mechanism_bundle(output_dir=tmp_path)

    assert {path.name for path in paths} == set(
        GATE1_PE_MECHANISM_REQUIRED_FILES
    )
    assert all(path.is_file() for path in paths)
    assert (
        tmp_path / "action_selection.jsonl"
    ).read_text(encoding="utf-8") == ""
    manifest = json.loads(
        (tmp_path / "manifest.yaml").read_text(encoding="utf-8")
    )
    verdict = json.loads(
        (tmp_path / "promotion_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    ablation = json.loads(
        (tmp_path / "ablation_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema_version"] == (
        GATE1_PE_MECHANISM_SCHEMA_VERSION
    )
    assert manifest["surface_kinds"] == [
        "numeric",
        "probability",
        "enum",
        "vector",
        "distribution",
    ]
    assert verdict == {
        "schema_version": GATE1_PE_MECHANISM_SCHEMA_VERSION,
        "gate_scope": "Gate 1 PE/LSS mechanism",
        "status": "mechanism-supported",
        "mechanism_passed": True,
        "causal_status": "not-evaluated",
        "thesis_status": "not-evaluated",
        "failed_gates": [],
    }
    assert all(ablation["gates"].values())
    assert ablation["lineage_audit"]["lineage_coverage"] == 1.0
    assert (
        ablation["evaluation_decoupling"]["byte_invariant"] is True
    )


def test_gate1_lineage_audit_rejects_duplicate_and_mismatch() -> None:
    predictions = [
        {"prediction_id": "prediction-1"},
        {"prediction_id": "prediction-1"},
    ]
    outcomes = [_lineage_row("prediction-1")]
    errors = [
        _lineage_row(
            "prediction-1",
            outcome_id="different-outcome",
        )
    ]

    audit = audit_prediction_lineage(
        predictions=predictions,
        outcomes=outcomes,
        prediction_errors=errors,
    )

    assert not audit.passed
    assert audit.duplicate_settlement_count == 1
    assert audit.accepted_mismatch_count == 2
    assert audit.lineage_coverage == 0.0
