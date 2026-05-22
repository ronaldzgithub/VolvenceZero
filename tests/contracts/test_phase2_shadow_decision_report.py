"""Contract tests for scripts/build_phase2_shadow_decision_report.py."""

from __future__ import annotations

import importlib.util
import json
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_phase2_shadow_decision_report.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "build_phase2_shadow_decision_report", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import script from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(*, runner_kind: str, winrate: float) -> dict:
    return {
        "schema_version": "phase2-shadow-evidence-multiseed.v1",
        "runner_kind": runner_kind,
        "include_phase3_combos": False,
        "head_to_head_results": (
            {
                "profile_a": "candidate",
                "profile_b": "pe-eta",
                "winrate_a_vs_b": winrate,
            },
        ),
        "provenance": {
            "git_sha": "test",
            "git_branch": "test",
            "working_tree_dirty": False,
            "python_version": "test",
            "platform": "test",
        },
    }


def test_synthetic_evidence_remains_shadow() -> None:
    module = _load_script_module()
    report = module.build_decision_report(_payload(runner_kind="synthetic", winrate=0.9))

    assert report["decisions"][0]["decision"] == "REMAIN_SHADOW"
    assert "synthetic evidence" in report["decisions"][0]["reasons"][0]


def test_default_runner_winrate_thresholds() -> None:
    module = _load_script_module()

    assert module.build_decision_report(_payload(runner_kind="default", winrate=0.7))["decisions"][0]["decision"] == "ACTIVE_CANDIDATE"
    assert module.build_decision_report(_payload(runner_kind="default", winrate=0.5))["decisions"][0]["decision"] == "REMAIN_SHADOW"
    assert module.build_decision_report(_payload(runner_kind="default", winrate=0.4))["decisions"][0]["decision"] == "DISABLED"


def test_cli_writes_json_and_markdown(tmp_path: pathlib.Path) -> None:
    module = _load_script_module()
    source = tmp_path / "phase2_shadow_evidence_multiseed.json"
    source.write_text(json.dumps(_payload(runner_kind="default", winrate=0.7)), encoding="utf-8")

    # Exercise helper paths without subprocess.
    report = module.build_decision_report(json.loads(source.read_text(encoding="utf-8")))
    md = module._build_markdown(report, source_path=source)

    assert report["schema_version"] == "phase2-shadow-decision-report.v1"
    assert "`ACTIVE_CANDIDATE`" in md
