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


class _RuntimeBackend(StructuredBackend):
    backend_name = "test-replay"
    model_name = "runtime-fixture"

    def complete_json(self, *, system, user, schema):
        del system, user, schema
        return {
            "target": (
                "packages/lifeform-domain-character/src/lifeform_domain_character/"
                "scenario_packages/fixture/scenes.yaml"
            ),
            "operation": "append_yaml_sequence_item",
            "document_path": "/scenes",
            "section_content": (
                "  - scenario_id: runtime_repair_01\n"
                "    family: relationship_repair\n"
                "    semantic_routing:\n"
                "      method: embedding_similarity_plus_schema_bound_structured_output\n"
            ),
            "root_cause": "The reviewed semantic asset lacks a repair scene.",
            "targeted_fix": "Append one owner-bound semantic repair scene.",
            "prediction": {
                "metric": "pattern_occurrence_count",
                "direction": "decrease",
                "expected_delta": -1,
                "evaluation_window": "next_mine_run",
            },
            "at_risk_regressions": ["the new scene overlaps an existing semantic regime"],
            "preserve_behaviors": ["boundary rubric remains passing"],
        }


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
    manifesto = json.loads((proposal_dir / "manifesto.json").read_text(encoding="utf-8"))
    rollback_command = manifesto["rollback"]["command"]
    assert manifesto["rollback"]["working_directory"] == "repository_root"
    assert rollback_command.startswith("git apply --reverse ")
    assert "proposal-output/proposals/" in rollback_command
    assert str(tmp_path) not in rollback_command
    assert target.read_text(encoding="utf-8") == before


def test_propose_appends_one_runtime_yaml_item_without_mutating_target(tmp_path: Path) -> None:
    config = _config(tmp_path)
    scenario_dir = (
        tmp_path
        / "packages"
        / "lifeform-domain-character"
        / "src"
        / "lifeform_domain_character"
        / "scenario_packages"
        / "fixture"
    )
    scenario_dir.mkdir(parents=True)
    target = scenario_dir / "scenes.yaml"
    target.write_text(
        'schema_version: "1.0"\nscenes:\n  - scenario_id: existing_01\n    family: existing\n',
        encoding="utf-8",
    )
    (scenario_dir / "test_suite.yaml").write_text(
        "routing_tests: []\nllm_evaluation:\n  semantic_coherence: []\n",
        encoding="utf-8",
    )
    before = target.read_text(encoding="utf-8")
    patterns = tmp_path / "runtime-patterns.jsonl"
    patterns.write_text(
        json.dumps(
            {
                "schema_version": "forge-failure-pattern.v2",
                "pattern_id": "fp_1123456789abcdef",
                "title": "runtime repair semantic gap",
                "verifier_cause": "bench relationship repair rubric failed",
                "agent_behavior_cause": "the response did not repair the rupture",
                "exposed_mechanism": "reviewed runtime scene coverage is incomplete",
                "occurrence_count": 1,
                "evidence_refs": [
                    {
                        "source_id": "bench_bundle:fixture",
                        "source_kind": "bench_bundle",
                        "locator": "arc:fixture/session:1/turn:2",
                        "excerpt": "repair rubric average=1.0",
                        "digest": "c" * 64,
                    }
                ],
                "source_kinds": ["bench_bundle"],
                "centroid_digest": "d" * 64,
                "editable_target": target.relative_to(tmp_path).as_posix(),
                "editable_component": "character_scenario_semantics",
                "surface_status": "in-surface",
                "surface_similarity": 0.9,
                "preserve_behaviors": ["boundary rubric remains passing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = propose_changes(
        config=config,
        failure_patterns_path=patterns,
        backend=_RuntimeBackend(),
        embedder=_Embedder(),
        output_dir=tmp_path / "runtime-proposal-output",
    )

    patch = (result.proposal_dirs[0] / "patch.diff").read_text(encoding="utf-8")
    assert "+  - scenario_id: runtime_repair_01" in patch
    assert not any(line.startswith("-") and not line.startswith("---") for line in patch.splitlines())
    assert target.read_text(encoding="utf-8") == before
