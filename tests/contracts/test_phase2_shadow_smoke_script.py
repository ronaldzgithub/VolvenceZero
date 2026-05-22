"""Contract tests for scripts/run_phase2_shadow_evidence_smoke.py."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import pathlib
from types import SimpleNamespace

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_phase2_shadow_evidence_smoke.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_phase2_shadow_evidence_smoke", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import script from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase2_shadow_smoke_script_writes_json_contract(tmp_path, monkeypatch) -> None:
    module = _load_script_module()

    async def fake_smoke(*, cases, runner_factory=None):
        del cases, runner_factory
        return SimpleNamespace(
            baseline_label="pe-eta",
            path_reports=(
                SimpleNamespace(
                    path_label="pe-eta",
                    benchmark_report=SimpleNamespace(
                        metric_means=(
                            ("mean_persona_geometry_drift", 0.1),
                            ("cpd_beta_switch_recommended_count", 0.0),
                            ("mean_least_control_score", 0.2),
                        )
                    ),
                ),
                SimpleNamespace(
                    path_label="cpd-beta-switch",
                    benchmark_report=SimpleNamespace(
                        metric_means=(
                            ("mean_persona_geometry_drift", 0.1),
                            ("cpd_beta_switch_recommended_count", 1.0),
                            ("mean_least_control_score", 0.3),
                        )
                    ),
                ),
            ),
            metric_deltas_from_baseline=(
                ("pe-eta", (("cpd_beta_switch_recommended_count", 0.0),)),
                ("cpd-beta-switch", (("cpd_beta_switch_recommended_count", 1.0),)),
            ),
            description="fake phase2 smoke",
        )

    monkeypatch.setattr(module, "run_phase2_shadow_evidence_smoke", fake_smoke)
    monkeypatch.setattr(
        module,
        "_collect_provenance",
        lambda: {
            "git_sha": "test-sha",
            "git_branch": "test-branch",
            "working_tree_dirty": False,
            "python_version": "3.test",
            "platform": "test-platform",
        },
    )

    output_dir = tmp_path / "phase2"
    exit_code = asyncio.run(
        module.main(output_dir=output_dir, case_limit=1, synthetic_runner=True)
    )

    assert exit_code == 0
    artifact = output_dir / "phase2_shadow_evidence_smoke.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    brief = output_dir / "phase2_shadow_evidence_smoke.md"
    manifest = output_dir / "phase2_shadow_evidence_manifest.json"

    assert payload["schema_version"] == "phase2-shadow-evidence-smoke.v1"
    assert payload["artifact_kind"] == "phase2_shadow_evidence_smoke"
    assert payload["runner_kind"] == "synthetic"
    assert payload["include_phase3_combos"] is False
    assert payload["provenance"] == {
        "git_sha": "test-sha",
        "git_branch": "test-branch",
        "working_tree_dirty": False,
        "python_version": "3.test",
        "platform": "test-platform",
    }
    assert payload["baseline_label"] == "pe-eta"
    assert payload["profile_labels"] == [
        "pe-eta",
        "cpd-beta-switch",
        "counterfactual-credit",
        "tom-owner",
        "persona-geometry-readout",
    ]
    assert payload["case_ids"] == ["repair"]
    assert "cpd_beta_switch_recommended_count" in payload["focus_metrics"]
    assert payload["focus_metric_means"]["cpd-beta-switch"][
        "cpd_beta_switch_recommended_count"
    ] == 1.0
    assert payload["focus_metric_deltas_from_baseline"]["cpd-beta-switch"][
        "cpd_beta_switch_recommended_count"
    ] == 1.0
    assert payload["head_to_head_results"] == [
        {
            "profile_a": "cpd-beta-switch",
            "profile_b": "pe-eta",
            "case_count": 3,
            "winrate_a_vs_b": 0.8333,
            "judge_kind": "deterministic",
            "notes": (
                "Deterministic metric-means comparison over 3 metrics; "
                "lower_is_better=('mean_least_control_effort', 'mean_persona_geometry_drift')."
            ),
        }
    ]
    gate = payload["cross_generation_gate_evidence"]
    assert gate["validation_score"] == 0.8333
    assert gate["head_to_head_aggregate_winrate"] == 0.8333
    assert gate["audit_evidence_id"] is None
    brief_text = brief.read_text(encoding="utf-8")
    assert "# Phase 2/3 Shadow Evidence Smoke" in brief_text
    assert "`cpd-beta-switch` vs `pe-eta`" in brief_text
    assert "Head-to-head aggregate winrate: `0.8333`" in brief_text
    assert "Runner kind: `synthetic`" in brief_text
    assert "Git SHA: `test-sha`" in brief_text
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["schema_version"] == "phase2-shadow-evidence-manifest.v1"
    assert manifest_payload["source_schema_version"] == "phase2-shadow-evidence-smoke.v1"
    records = {item["path"]: item for item in manifest_payload["artifacts"]}
    artifact_record = records[str(artifact)]
    brief_record = records[str(brief)]
    assert artifact_record["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert artifact_record["size_bytes"] == len(artifact.read_bytes())
    assert brief_record["sha256"] == hashlib.sha256(brief.read_bytes()).hexdigest()
    assert brief_record["size_bytes"] == len(brief.read_bytes())
    module._verify_written_manifest(manifest)
    brief.write_text(brief_text + "\nTAMPERED\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        module._verify_written_manifest(manifest)


def test_phase2_shadow_smoke_script_can_include_phase3_combos(tmp_path, monkeypatch) -> None:
    module = _load_script_module()

    async def fake_ablation(*, cases, profile_labels, baseline_label, runner_factory=None):
        del cases, runner_factory
        return SimpleNamespace(
            baseline_label=baseline_label,
            path_reports=tuple(
                SimpleNamespace(
                    path_label=label,
                    benchmark_report=SimpleNamespace(
                        metric_means=(("cpd_beta_switch_recommended_count", 1.0),)
                    ),
                )
                for label in profile_labels
            ),
            metric_deltas_from_baseline=tuple(
                (label, (("cpd_beta_switch_recommended_count", 0.0),))
                for label in profile_labels
            ),
            description="fake phase2+phase3 smoke",
        )

    fake_agent = SimpleNamespace(run_dialogue_pe_eta_ablation_benchmark=fake_ablation)
    monkeypatch.setitem(__import__("sys").modules, "volvence_zero.agent", fake_agent)
    monkeypatch.setattr(
        module,
        "_collect_provenance",
        lambda: {
            "git_sha": "test-sha",
            "git_branch": "test-branch",
            "working_tree_dirty": False,
            "python_version": "3.test",
            "platform": "test-platform",
        },
    )

    output_dir = tmp_path / "phase3"
    exit_code = asyncio.run(
        module.main(
            output_dir=output_dir,
            case_limit=1,
            synthetic_runner=True,
            include_phase3_combos=True,
        )
    )

    assert exit_code == 0
    payload = json.loads(
        (output_dir / "phase2_shadow_evidence_smoke.json").read_text(encoding="utf-8")
    )
    assert "cpd-counterfactual-credit" in payload["profile_labels"]
    assert "tom-persona-geometry" in payload["profile_labels"]
    assert "audit-persona-geometry" in payload["profile_labels"]
    assert payload["include_phase3_combos"] is True
