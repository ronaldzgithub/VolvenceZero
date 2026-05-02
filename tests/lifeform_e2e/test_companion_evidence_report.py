"""Companion evidence report contract tests."""

from __future__ import annotations


def test_companion_evidence_report_passes_all_current_gates() -> None:
    from lifeform_evolution import (
        companion_evidence_report_to_dict,
        format_companion_evidence_report,
        run_companion_evidence,
    )

    report = run_companion_evidence()
    payload = companion_evidence_report_to_dict(report)
    text = format_companion_evidence_report(report)

    assert report.passed is True
    assert {gate.gate_id for gate in report.gates} == {"C1", "C2", "C3", "C4"}
    assert payload["passed"] is True
    assert 0.0 <= payload["composite_score"] <= 1.0
    assert len(payload["gates"]) == 4
    assert len(payload["transcripts"]) == 4
    assert "Companion evidence report" in text
    assert "composite_score:" in text
    assert "widening_transcripts: 4" in text
    assert "[C1] PASS" in text
    assert "[C4] PASS" in text


def test_companion_evidence_cli_writes_json(tmp_path) -> None:
    import json

    from lifeform_evolution.cli import main

    out = tmp_path / "companion-evidence.json"
    exit_code = main(
        [
            "--scenario",
            "low-mood-disclosure",
            "--min-regime-match-rate",
            "0.0",
            "--companion-evidence-report",
            "--companion-evidence-json",
            str(out),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert {gate["gate_id"] for gate in payload["gates"]} == {"C1", "C2", "C3", "C4"}
    assert {item["condition"] for item in payload["transcripts"]} == {
        "paraphrase-low-mood",
        "tone-shift-repair",
        "delayed-return",
        "preference-conflict",
    }
