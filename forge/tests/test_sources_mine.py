from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from volvence_forge.config import ForgeConfig, ForgePaths
from volvence_forge.foundation import PromptStore, SchemaStore
from volvence_forge.mine import mine_bundle
from volvence_forge.sources import SourceParseError, load_source_bundle


REPO_ROOT = Path(__file__).resolve().parents[2]


def _fixture_root(tmp_path: Path) -> tuple[ForgeConfig, Path]:
    (tmp_path / "forge").mkdir()
    shutil.copy2(REPO_ROOT / "forge" / "editable_surface.yaml", tmp_path / "forge" / "editable_surface.yaml")
    (tmp_path / "forge" / "ledger.jsonl").write_text(
        json.dumps({"event": "initialized"}) + "\n", encoding="utf-8"
    )
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    rules = tmp_path / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "test.mdc").write_text("# Existing rule\n", encoding="utf-8")
    (transcripts / "run.jsonl").write_text(
        json.dumps({"type": "turn_ended", "status": "error", "error": "timeout"})
        + "\n"
        + json.dumps(
            {
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "pytest"},
                        {"type": "tool_result", "is_error": True, "content": "failed"},
                    ]
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_root = tmp_path / "artifacts" / "run"
    artifact_root.mkdir(parents=True)
    (artifact_root / "promotion_verdict.json").write_text(
        json.dumps({"gates": {"effect": False}, "promotion_allowed": False}),
        encoding="utf-8",
    )
    (artifact_root / "report.md").write_text("effect gate failed\n", encoding="utf-8")
    plans = tmp_path / ".cursor" / "plans"
    plans.mkdir(parents=True)
    (plans / "run.plan.md").write_text(
        "---\nname: test plan\noverview: test overview\ntodos: []\n---\n# Plan\n",
        encoding="utf-8",
    )
    paths = ForgePaths.discover(repo_root=tmp_path, transcripts_root=transcripts)
    return ForgeConfig.load(paths), REPO_ROOT / "forge"


class _StructuredBackend:
    backend_name = "test-replay"
    model_name = "fixture"

    def complete_json(self, *, system, user, schema):
        del system, user, schema
        return {
            "records": [
                {
                    "verifier_cause": "a verifier reported a failed contract",
                    "agent_behavior_cause": "the agent retried without preserving context",
                    "exposed_mechanism": "rule guidance lacks bounded retry handoff",
                    "confidence": 0.9,
                }
            ]
        }


class _EmbeddingBackend:
    model_name = "fixture-embedding"

    def encode(self, texts):
        return np.tile(np.asarray((1.0, 0.0), dtype=np.float64), (len(texts), 1))


class _DistinctCauseBackend:
    backend_name = "test-replay"
    model_name = "distinct-causes"

    def __init__(self) -> None:
        self._cursor = 0

    def complete_json(self, *, system, user, schema):
        del system, user, schema
        mechanisms = ("runtime recovery mechanism", "causal promotion gate mechanism")
        mechanism = mechanisms[self._cursor]
        self._cursor += 1
        return {
            "records": [
                {
                    "verifier_cause": mechanism,
                    "agent_behavior_cause": mechanism,
                    "exposed_mechanism": mechanism,
                    "confidence": 0.9,
                }
            ]
        }


class _NearButDistinctEmbeddingBackend:
    model_name = "fixture-near-distinct"

    def encode(self, texts):
        values = []
        for value in texts:
            if "causal promotion gate mechanism" in value:
                values.append((0.78, np.sqrt(1.0 - 0.78**2)))
            else:
                values.append((1.0, 0.0))
        return np.asarray(values, dtype=np.float64)


def test_load_source_bundle_is_structured_and_deterministic(tmp_path: Path) -> None:
    config, _ = _fixture_root(tmp_path)
    bundle = load_source_bundle(config.paths)

    assert len(bundle.transcripts) == 1
    assert bundle.transcripts[0].tool_sequence == ("pytest",)
    assert bundle.transcripts[0].error_refs
    assert len(bundle.verdicts) == 1
    assert bundle.verdicts[0].failed_gate_refs
    assert len(bundle.plans) == 1


def test_invalid_transcript_fails_loudly(tmp_path: Path) -> None:
    config, _ = _fixture_root(tmp_path)
    (config.paths.transcripts_root / "bad.jsonl").write_text("not-json\n", encoding="utf-8")
    with pytest.raises(SourceParseError, match="Invalid transcript JSON"):
        load_source_bundle(config.paths)


def test_legacy_heading_only_plan_is_explicitly_supported(tmp_path: Path) -> None:
    config, _ = _fixture_root(tmp_path)
    legacy = config.paths.plans_root / "legacy.plan.md"
    legacy.write_text("# Legacy campaign\n\n> Frozen evidence narrative.\n", encoding="utf-8")
    bundle = load_source_bundle(config.paths)
    assert any(plan.name == "Legacy campaign" for plan in bundle.plans)


def test_mine_bundle_uses_semantic_backend_and_schema(tmp_path: Path) -> None:
    config, forge_root = _fixture_root(tmp_path)
    bundle = load_source_bundle(config.paths, max_transcripts=1, max_verdicts=1, max_plans=1)
    patterns = mine_bundle(
        bundle=bundle,
        config=config,
        structured_backend=_StructuredBackend(),
        embedding_backend=_EmbeddingBackend(),
        schema_store=SchemaStore(forge_root / "schemas"),
        prompt_store=PromptStore(forge_root / "prompts"),
    )

    assert patterns
    assert patterns[0]["schema_version"] == "forge-failure-pattern.v1"
    assert patterns[0]["surface_status"] == "in-surface"
    assert patterns[0]["editable_target"] == ".cursor/rules/test.mdc"
    assert str(patterns[0]["pattern_id"]).startswith("fp_")


def test_mine_does_not_merge_near_but_distinct_failure_causes(tmp_path: Path) -> None:
    config, forge_root = _fixture_root(tmp_path)
    bundle = load_source_bundle(config.paths, max_transcripts=1, max_verdicts=1, max_plans=1)

    patterns = mine_bundle(
        bundle=bundle,
        config=config,
        structured_backend=_DistinctCauseBackend(),
        embedding_backend=_NearButDistinctEmbeddingBackend(),
        schema_store=SchemaStore(forge_root / "schemas"),
        prompt_store=PromptStore(forge_root / "prompts"),
    )

    assert len(patterns) == 2
    assert {tuple(pattern["source_kinds"]) for pattern in patterns} == {
        ("promotion_verdict",),
        ("transcript",),
    }
