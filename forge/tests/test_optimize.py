from __future__ import annotations

import json
from pathlib import Path
import shutil

from volvence_forge.config import ForgeConfig, ForgePaths
from volvence_forge.foundation import sha256_text
from volvence_forge.optimize import select_pareto_candidates


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = (
    "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/"
    "runtime_assets/companion_playbook_overlay.json"
)


def _config(tmp_path: Path) -> ForgeConfig:
    forge_root = tmp_path / "forge"
    shutil.copytree(REPO_ROOT / "forge" / "schemas", forge_root / "schemas")
    shutil.copy2(REPO_ROOT / "forge" / "editable_surface.yaml", forge_root / "editable_surface.yaml")
    (forge_root / "ledger.jsonl").write_text("", encoding="utf-8")
    target = tmp_path / TARGET
    target.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / TARGET, target)
    return ForgeConfig.load(
        ForgePaths.discover(repo_root=tmp_path, transcripts_root=tmp_path / "transcripts")
    )


def _write_json(path: Path, value: dict[str, object]) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def _candidate(
    root: Path,
    *,
    proposal_id: str,
    validation_delta: float,
    added_lines: int,
    gate_decision: str = "ALLOW",
    validation_status: str = "PASS",
) -> Path:
    proposal_dir = root / proposal_id
    proposal_dir.mkdir(parents=True)
    patch = (
        f"--- a/{TARGET}\n"
        f"+++ b/{TARGET}\n"
        "@@ -1,1 +1,2 @@\n"
        " {\n"
        + "".join(f"+candidate-line-{index}\n" for index in range(added_lines))
    )
    (proposal_dir / "patch.diff").write_text(patch, encoding="utf-8")
    manifesto = {
        "schema_version": "forge-proposal-manifesto.v1",
        "proposal_id": proposal_id,
        "pattern_id": "fp_0123456789abcdef",
        "target": TARGET,
        "target_preimage_sha256": "a" * 64,
        "evidence": [
            {
                "source_id": "bench_bundle:fixture",
                "source_kind": "bench_bundle",
                "locator": "arc:fixture/session:1/turn:1",
                "excerpt": "continuity score=1",
                "digest": "b" * 64,
            }
        ],
        "root_cause": "owner-bound playbook gap",
        "targeted_fix": "append one reviewed rule",
        "predicted_impact": {
            "metric": "pattern_occurrence_count",
            "direction": "decrease",
            "expected_delta": -1,
            "baseline_value": 2,
            "evaluation_window": "next_mine_run",
        },
        "at_risk_regressions": ["unrelated routing"],
        "preserve_behaviors": ["boundary remains passing"],
        "rollback": {
            "method": "reverse_patch",
            "working_directory": "repository_root",
            "command": f"git apply --reverse {proposal_id}/patch.diff",
        },
        "generator": {"backend": "fixture", "model": "fixture"},
        "created_at": "2026-08-01T00:00:00Z",
    }
    manifesto_text = _write_json(proposal_dir / "manifesto.json", manifesto)
    validation = {
        "schema_version": "forge-validation-report.v2",
        "proposal_id": proposal_id,
        "component": "companion_runtime_playbook_overlay",
        "runtime_gate_evidence": {
            "component": "companion_runtime_playbook_overlay",
            "target": TARGET,
            "frozen_suite": (
                "packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/"
                "runtime_assets/test_suite.yaml"
            ),
            "frozen_suite_sha256": "c" * 64,
            "evaluated_test_ids": ["case-1", "case-2"],
            "baseline_passed_test_ids": ["case-1"],
            "candidate_passed_test_ids": ["case-1", "case-2"],
            "baseline_pass_rate": 0.5,
            "candidate_pass_rate": 0.5 + validation_delta,
            "validation_delta": validation_delta,
            "capacity_cost": 0.1,
            "contract_integrity": True,
            "rollback_resilience": True,
            "judge": {"backend": "fixture", "model": "fixture"},
        },
        "status": validation_status,
        "patch_sha256": sha256_text(patch),
        "manifesto_sha256": sha256_text(manifesto_text),
        "checks": [
            {
                "name": "fixture",
                "status": validation_status,
                "detail": "fixture",
            }
        ],
        "validated_at": "2026-08-01T00:00:01Z",
    }
    validation_text = _write_json(proposal_dir / "validation.json", validation)
    gate = {
        "schema_version": "forge-gate-decision.v1",
        "proposal_id": proposal_id,
        "target": TARGET,
        "decision": gate_decision,
        "reasons": [] if gate_decision == "ALLOW" else ["fixture block"],
        "desired_gate": "offline",
        "inputs": {
            "patch_sha256": sha256_text(patch),
            "manifesto_sha256": sha256_text(manifesto_text),
            "validation_sha256": sha256_text(validation_text),
        },
        "metrics": {
            "baseline_pass_rate": 0.5,
            "candidate_pass_rate": 0.5 + validation_delta,
            "validation_delta": validation_delta,
            "capacity_cost": 0.1,
            "contract_integrity": 1.0,
            "rollback_resilience": 1.0,
        },
        "authority": "volvence_zero.credit.gate.evaluate_gate_reasons",
        "created_at": "2026-08-01T00:00:02Z",
    }
    _write_json(proposal_dir / "gate_decision.json", gate)
    return proposal_dir


def test_pareto_selects_dominant_allowed_candidate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    root = tmp_path / "population" / "proposals"
    root.mkdir(parents=True)
    _candidate(
        root,
        proposal_id="pr_1111111111111111",
        validation_delta=0.2,
        added_lines=1,
    )
    _candidate(
        root,
        proposal_id="pr_2222222222222222",
        validation_delta=0.1,
        added_lines=2,
    )

    result = select_pareto_candidates(config=config, proposals_root=root)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert result.decision == "SELECT"
    assert result.selected_proposal_ids == ("pr_1111111111111111",)
    assert report["component_frontiers"][0]["pareto_front_ids"] == [
        "pr_1111111111111111"
    ]


def test_optimizer_stops_when_gate_blocks_population(tmp_path: Path) -> None:
    config = _config(tmp_path)
    root = tmp_path / "population" / "proposals"
    root.mkdir(parents=True)
    _candidate(
        root,
        proposal_id="pr_3333333333333333",
        validation_delta=0.04,
        added_lines=1,
        gate_decision="BLOCK",
    )

    result = select_pareto_candidates(config=config, proposals_root=root)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert result.decision == "STOP"
    assert result.selected_proposal_ids == ()
    assert report["stop_reasons"]
    assert report["candidates"][0]["blocking_reasons"] == [
        "OFFLINE gate decision is not ALLOW"
    ]
