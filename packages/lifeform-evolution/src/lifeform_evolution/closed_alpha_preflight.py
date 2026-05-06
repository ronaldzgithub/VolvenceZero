"""Closed-alpha preflight: aggregate internal evidence gates."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from lifeform_evolution.open_dialogue_v0_gate import run_v0_gate
from lifeform_evolution.relationship_repair_alpha_gate import (
    RepairAlphaGateReport,
    run_relationship_repair_alpha_gate,
)


@dataclasses.dataclass(frozen=True)
class ClosedAlphaPreflightReport:
    passed: bool
    open_dialogue_v0_passed: bool
    relationship_repair_alpha_passed: bool
    artifacts_dir: str
    scope_root_dir: str
    open_dialogue_report_path: str
    relationship_repair_report_path: str
    gate_items: tuple[tuple[str, bool, str], ...]
    relationship_repair_alpha: RepairAlphaGateReport

    def to_json(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "open_dialogue_v0_passed": self.open_dialogue_v0_passed,
            "relationship_repair_alpha_passed": (
                self.relationship_repair_alpha_passed
            ),
            "artifacts_dir": self.artifacts_dir,
            "scope_root_dir": self.scope_root_dir,
            "open_dialogue_report_path": self.open_dialogue_report_path,
            "relationship_repair_report_path": self.relationship_repair_report_path,
            "gate_items": [
                {"name": name, "passed": passed, "detail": detail}
                for name, passed, detail in self.gate_items
            ],
            "relationship_repair_alpha": (
                self.relationship_repair_alpha.to_json()
            ),
        }


def format_closed_alpha_preflight_report(
    report: ClosedAlphaPreflightReport,
) -> str:
    status = "PASSED" if report.passed else "FAILED"
    lines = [f"Closed alpha preflight: {status}"]
    lines.append(
        f"  open_dialogue_v0={report.open_dialogue_v0_passed} "
        f"report={report.open_dialogue_report_path}"
    )
    lines.append(
        "  relationship_repair_alpha="
        f"{report.relationship_repair_alpha_passed} "
        f"report={report.relationship_repair_report_path}"
    )
    lines.append("  gate_items:")
    for name, passed, detail in report.gate_items:
        lines.append(f"    - {name}: {'ok' if passed else 'FAIL'} ({detail})")
    return "\n".join(lines)


def run_closed_alpha_preflight(
    *,
    artifacts_dir: str | Path = "artifacts/closed_alpha_preflight",
    scope_root_dir: str | Path = "artifacts/closed_alpha_preflight_scope",
) -> ClosedAlphaPreflightReport:
    artifacts = Path(artifacts_dir)
    scope_root = Path(scope_root_dir)
    open_dialogue_dir = artifacts / "open_dialogue"
    open_dialogue_scope = scope_root / "open_dialogue"
    repair_dir = artifacts / "relationship_repair_alpha_gate"
    repair_scope = scope_root / "relationship_repair_alpha_gate"
    repair_report_path = repair_dir / "report.json"

    open_dialogue_report = run_v0_gate(
        out_dir=open_dialogue_dir,
        scope_root_dir=open_dialogue_scope,
    )
    repair_report = run_relationship_repair_alpha_gate(
        out_path=repair_report_path,
        scope_root_dir=repair_scope,
    )

    gate_items = (
        (
            "open_dialogue_v0_gate",
            open_dialogue_report.passed,
            open_dialogue_report.description,
        ),
        (
            "relationship_repair_alpha_gate",
            repair_report.passed,
            (
                "repair alpha treatment/control + observed memory + "
                "same-user recall + cross-user isolation."
            ),
        ),
    )
    report = ClosedAlphaPreflightReport(
        passed=all(passed for _, passed, _ in gate_items),
        open_dialogue_v0_passed=open_dialogue_report.passed,
        relationship_repair_alpha_passed=repair_report.passed,
        artifacts_dir=str(artifacts),
        scope_root_dir=str(scope_root),
        open_dialogue_report_path=str(open_dialogue_dir / "v0_gate_report.json"),
        relationship_repair_report_path=str(repair_report_path),
        gate_items=gate_items,
        relationship_repair_alpha=repair_report,
    )
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "closed_alpha_preflight_report.json").write_text(
        json.dumps(report.to_json(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


__all__ = (
    "ClosedAlphaPreflightReport",
    "format_closed_alpha_preflight_report",
    "run_closed_alpha_preflight",
)
