"""Contract tests for scripts/run_phase2_shadow_evidence_multiseed.py."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_phase2_shadow_evidence_multiseed.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_phase2_shadow_evidence_multiseed", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import script from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase2_multiseed_script_writes_artifacts(tmp_path, monkeypatch) -> None:
    module = _load_script_module()

    async def fake_run_once(*, seed, case_limit, synthetic_runner, include_phase3_combos):
        del case_limit, synthetic_runner, include_phase3_combos
        return {
            "baseline_label": "pe-eta",
            "case_ids": ["repair"],
            "per_path_metric_means": {
                "pe-eta": {"mean_least_control_score": 0.2},
                "counterfactual-credit": {"mean_least_control_score": 0.3 + seed * 0.1},
            },
            "description": "fake",
        }

    monkeypatch.setattr(module, "_run_once", fake_run_once)
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
    output_dir = tmp_path / "multiseed"

    exit_code = asyncio.run(
        module.main(
            output_dir=output_dir,
            seeds=(0, 1),
            case_limit=1,
            synthetic_runner=True,
            include_phase3_combos=False,
        )
    )

    assert exit_code == 0
    payload = json.loads(
        (output_dir / "phase2_shadow_evidence_multiseed.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "phase2-shadow-evidence-multiseed.v1"
    assert payload["seeds"] == [0, 1]
    assert payload["profile_metric_summaries"]["counterfactual-credit"][
        "mean_least_control_score"
    ] == {"mean": 0.35, "std": 0.0707, "stderr": 0.05}
    assert payload["cross_generation_gate_evidence"]["head_to_head_aggregate_winrate"] == 1.0

    manifest = json.loads(
        (output_dir / "phase2_shadow_evidence_multiseed_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "phase2-shadow-evidence-multiseed-manifest.v1"
    assert len(manifest["artifacts"]) == 2
