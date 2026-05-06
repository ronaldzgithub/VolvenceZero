"""Aggregate preflight for closed-alpha internal gates."""

from __future__ import annotations

import json
from pathlib import Path

from lifeform_evolution.closed_alpha_preflight import (
    format_closed_alpha_preflight_report,
    run_closed_alpha_preflight,
)


def test_closed_alpha_preflight_passes_and_writes_manifest(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    scope_root = tmp_path / "scope"

    report = run_closed_alpha_preflight(
        artifacts_dir=artifacts_dir,
        scope_root_dir=scope_root,
    )

    assert report.passed, (
        "closed alpha preflight failed: "
        + "; ".join(
            f"{name}={'ok' if ok else 'FAIL'} ({detail})"
            for name, ok, detail in report.gate_items
        )
    )
    assert report.open_dialogue_v0_passed is True
    assert report.relationship_repair_alpha_passed is True
    assert Path(report.open_dialogue_report_path).exists()
    assert Path(report.relationship_repair_report_path).exists()

    manifest_path = artifacts_dir / "closed_alpha_preflight_report.json"
    serialized = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert serialized["passed"] is True
    assert serialized["open_dialogue_v0_passed"] is True
    assert serialized["relationship_repair_alpha_passed"] is True

    formatted = format_closed_alpha_preflight_report(report)
    assert "Closed alpha preflight: PASSED" in formatted


def test_closed_alpha_preflight_cli_writes_manifest(tmp_path: Path) -> None:
    from lifeform_evolution.cli import main_alpha_preflight

    artifacts_dir = tmp_path / "cli-artifacts"
    scope_root = tmp_path / "cli-scope"

    exit_code = main_alpha_preflight(
        [
            "--artifacts-dir",
            str(artifacts_dir),
            "--scope-root",
            str(scope_root),
            "--quiet",
        ]
    )

    assert exit_code == 0
    manifest_path = artifacts_dir / "closed_alpha_preflight_report.json"
    serialized = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert serialized["passed"] is True
