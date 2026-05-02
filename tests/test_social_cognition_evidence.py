from __future__ import annotations


def test_social_cognition_evidence_report_passes_tom_owner_gates() -> None:
    from lifeform_evolution import (
        format_social_cognition_evidence_report,
        run_social_cognition_evidence,
        social_cognition_evidence_report_to_dict,
    )

    report = run_social_cognition_evidence()
    payload = social_cognition_evidence_report_to_dict(report)
    text = format_social_cognition_evidence_report(report)

    assert report.passed is True
    assert {gate.gate_id for gate in report.gates} == {
        "T1",
        "T2",
        "T3",
        "R1",
        "R2",
        "G1",
        "GROUP1",
    }
    assert payload["passed"] is True
    assert {gate["gate_id"] for gate in payload["gates"]} == {
        "T1",
        "T2",
        "T3",
        "R1",
        "R2",
        "G1",
        "GROUP1",
    }
    t3 = next(gate for gate in payload["gates"] if gate["gate_id"] == "T3")
    assert t3["metrics"]["belief_records"] == 1.0
    assert t3["metrics"]["preference_records"] == 1.0
    assert t3["metrics"]["assembly_belief_count"] == 1.0
    assert t3["metrics"]["assembly_preference_count"] == 1.0
    r1 = next(gate for gate in payload["gates"] if gate["gate_id"] == "R1")
    assert r1["metrics"]["role_pe_credit"] == -0.71
    r2 = next(gate for gate in payload["gates"] if gate["gate_id"] == "R2")
    assert r2["metrics"]["role_prediction_count"] == 1.0
    assert r2["metrics"]["assembly_role_count"] == 1.0
    g1 = next(gate for gate in payload["gates"] if gate["gate_id"] == "G1")
    assert g1["metrics"]["common_ground_atom_count"] == 2.0
    assert g1["metrics"]["assembly_common_ground_count"] == 2.0
    group1 = next(gate for gate in payload["gates"] if gate["gate_id"] == "GROUP1")
    assert group1["metrics"]["group_count"] == 1.0
    assert group1["metrics"]["assembly_group_count"] == 1.0
    assert group1["metrics"]["assembly_group_joint_commitment_count"] == 1.0
    assert "Social cognition evidence report" in text
    assert "[T1] PASS" in text
    assert "[T3] PASS" in text


def test_social_cognition_evidence_cli_writes_json(tmp_path) -> None:
    import json

    from lifeform_evolution.cli import main

    out = tmp_path / "social-cognition-evidence.json"
    exit_code = main(
        [
            "--scenario",
            "low-mood-disclosure",
            "--min-regime-match-rate",
            "0.0",
            "--social-cognition-evidence-report",
            "--social-cognition-evidence-json",
            str(out),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert {gate["gate_id"] for gate in payload["gates"]} == {
        "T1",
        "T2",
        "T3",
        "R1",
        "R2",
        "G1",
        "GROUP1",
    }
    t3 = next(gate for gate in payload["gates"] if gate["gate_id"] == "T3")
    assert t3["metrics"]["assembly_belief_count"] == 1.0
    assert t3["metrics"]["assembly_preference_count"] == 1.0
    r1 = next(gate for gate in payload["gates"] if gate["gate_id"] == "R1")
    assert r1["metrics"]["role_pe_credit"] == -0.71
    r2 = next(gate for gate in payload["gates"] if gate["gate_id"] == "R2")
    assert r2["metrics"]["assembly_role_count"] == 1.0
    g1 = next(gate for gate in payload["gates"] if gate["gate_id"] == "G1")
    assert g1["metrics"]["assembly_common_ground_count"] == 2.0
    group1 = next(gate for gate in payload["gates"] if gate["gate_id"] == "GROUP1")
    assert group1["metrics"]["assembly_group_joint_commitment_count"] == 1.0
