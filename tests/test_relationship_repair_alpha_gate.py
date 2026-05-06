"""Closed-alpha matched-control gate for relationship repair expression."""

from __future__ import annotations

import json
from pathlib import Path

from lifeform_evolution.relationship_repair_alpha_gate import (
    format_relationship_repair_alpha_report,
    run_relationship_repair_alpha_gate,
)


def test_relationship_repair_alpha_gate_passes_and_writes_report(
    tmp_path: Path,
) -> None:
    out_path = tmp_path / "relationship_repair_alpha_gate.json"

    scope_root = tmp_path / "scope"
    report = run_relationship_repair_alpha_gate(
        out_path=out_path,
        scope_root_dir=scope_root,
    )

    assert report.passed, (
        "relationship repair alpha gate failed: "
        + "; ".join(
            f"{name}={'ok' if ok else 'FAIL'} ({detail})"
            for name, ok, detail in report.gate_items
        )
    )
    assert report.treatment.rupture_kind == "over_directive"
    assert report.matched_control.rupture_kind == "over_directive"
    assert report.treatment.repair_alpha_rationale_present is True
    assert report.treatment.repair_first_intent_present is True
    assert report.treatment.repair_alpha_phrase_present is True
    assert report.treatment.observed_repair_memory_count >= 1
    assert report.treatment.same_user_recall_count >= 1
    assert report.treatment.cross_user_leakage_count == 0
    assert report.matched_control.repair_alpha_rationale_present is False
    assert report.matched_control.repair_first_intent_present is False
    assert report.matched_control.repair_alpha_phrase_present is False
    assert report.matched_control.observed_repair_memory_count == 0

    serialized = json.loads(out_path.read_text(encoding="utf-8"))
    assert serialized["passed"] is True
    assert serialized["treatment"]["repair_alpha_enabled"] is True
    assert serialized["treatment"]["observed_repair_memory_count"] >= 1
    assert serialized["treatment"]["same_user_recall_count"] >= 1
    assert serialized["treatment"]["cross_user_leakage_count"] == 0
    assert serialized["matched_control"]["repair_alpha_enabled"] is False

    formatted = format_relationship_repair_alpha_report(report)
    assert "Relationship repair alpha gate: PASSED" in formatted
    assert "same_user_recall=" in formatted


def test_relationship_repair_alpha_gate_cli_writes_report(tmp_path: Path) -> None:
    from lifeform_evolution.cli import main_repair_alpha_gate

    out_path = tmp_path / "cli-report.json"
    scope_root = tmp_path / "cli-scope"

    exit_code = main_repair_alpha_gate(
        [
            "--out",
            str(out_path),
            "--scope-root",
            str(scope_root),
            "--quiet",
        ]
    )

    assert exit_code == 0
    serialized = json.loads(out_path.read_text(encoding="utf-8"))
    assert serialized["passed"] is True
    assert serialized["treatment"]["same_user_recall_count"] >= 1
