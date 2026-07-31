from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from volvence_forge.config import ForgeConfig, ForgePaths
from volvence_forge.foundation import EmbeddingBackend, StructuredBackend
from volvence_forge.propose import propose_changes


REPO_ROOT = Path(__file__).resolve().parents[2]


def _config(tmp_path: Path) -> ForgeConfig:
    forge_root = tmp_path / "forge"
    for name in ("schemas", "prompts"):
        shutil.copytree(REPO_ROOT / "forge" / name, forge_root / name)
    shutil.copy2(REPO_ROOT / "forge" / "editable_surface.yaml", forge_root / "editable_surface.yaml")
    (forge_root / "ledger.jsonl").write_text("", encoding="utf-8")
    rules = tmp_path / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "test.mdc").write_text("# Existing rule\n\nKeep the contract check.\n", encoding="utf-8")
    return ForgeConfig.load(ForgePaths.discover(repo_root=tmp_path, transcripts_root=tmp_path / "transcripts"))


class _Backend(StructuredBackend):
    backend_name = "test-replay"
    model_name = "fixture"

    def complete_json(self, *, system, user, schema):
        del system, user, schema
        return {
            "target": ".cursor/rules/test.mdc",
            "operation": "append_section",
            "section_content": "# Bounded retry handoff\n\nPreserve the contract check before retrying.",
            "root_cause": "The retry path loses the verifier context.",
            "targeted_fix": "Require a bounded handoff section before another retry.",
            "prediction": {
                "metric": "pattern_occurrence_count",
                "direction": "decrease",
                "expected_delta": -1,
                "evaluation_window": "next_mine_run",
            },
            "at_risk_regressions": ["rule file grows without a corresponding test"],
            "preserve_behaviors": ["Keep the existing contract check"],
        }


class _Embedder(EmbeddingBackend):
    model_name = "fixture-embedding"

    def encode(self, texts):
        return np.ones((len(texts), 2), dtype=np.float64)


def test_propose_writes_bundle_without_mutating_target(tmp_path: Path) -> None:
    config = _config(tmp_path)
    target = tmp_path / ".cursor" / "rules" / "test.mdc"
    before = target.read_text(encoding="utf-8")
    patterns = tmp_path / "patterns.jsonl"
    patterns.write_text(
        json.dumps(
            {
                "schema_version": "forge-failure-pattern.v1",
                "pattern_id": "fp_0123456789abcdef",
                "title": "bounded retry handoff",
                "verifier_cause": "contract failed",
                "agent_behavior_cause": "retry lost context",
                "exposed_mechanism": "rule guidance gap",
                "occurrence_count": 2,
                "evidence_refs": [
                    {
                        "source_id": "source-1",
                        "source_kind": "transcript",
                        "locator": "run.jsonl#L1",
                        "excerpt": "contract failed",
                        "digest": "a" * 64,
                    }
                ],
                "source_kinds": ["transcript"],
                "centroid_digest": "b" * 64,
            "editable_target": ".cursor/rules/test.mdc",
                "editable_component": "repository_agent_rules",
                "surface_status": "in-surface",
                "surface_similarity": 0.9,
                "preserve_behaviors": ["Keep the existing contract check"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = propose_changes(
        config=config,
        failure_patterns_path=patterns,
        backend=_Backend(),
        embedder=_Embedder(),
        output_dir=tmp_path / "proposal-output",
    )

    assert len(result.proposal_dirs) == 1
    proposal_dir = result.proposal_dirs[0]
    assert (proposal_dir / "patch.diff").is_file()
    assert (proposal_dir / "manifesto.json").is_file()
    assert (proposal_dir / "failure_pattern.json").is_file()
    assert target.read_text(encoding="utf-8") == before
