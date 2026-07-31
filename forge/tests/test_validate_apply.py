from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from volvence_forge.apply import ApplyError, apply_proposal, reject_proposal
from volvence_forge.config import ForgeConfig, ForgePaths
from volvence_forge.foundation import EmbeddingBackend, StructuredBackend
from volvence_forge.propose import propose_changes
from volvence_forge.validate import CommandOutcome, validate_proposal


REPO_ROOT = Path(__file__).resolve().parents[2]


class _ProposalBackend(StructuredBackend):
    backend_name = "test-replay"
    model_name = "fixture"

    def complete_json(self, *, system, user, schema):
        del system, user, schema
        return {
            "target": ".cursor/rules/test.mdc",
            "operation": "append_section",
            "section_content": "# Bounded recovery\n\nRecord verifier context before one bounded retry.",
            "root_cause": "The failure path loses verifier context.",
            "targeted_fix": "Require a bounded recovery handoff.",
            "prediction": {
                "metric": "pattern_occurrence_count",
                "direction": "decrease",
                "expected_delta": -1,
                "evaluation_window": "next_mine_run",
            },
            "at_risk_regressions": ["successful direct execution becomes unnecessarily verbose"],
            "preserve_behaviors": ["contract check passes"],
        }


class _RelevanceBackend(StructuredBackend):
    backend_name = "external-test-judge"
    model_name = "fixture"

    def complete_json(self, *, system, user, schema):
        del system, user, schema
        return {
            "relevant": True,
            "evidence_alignment": True,
            "preservation_assessment": True,
            "reason": "The append-only rule directly addresses the cited recovery gap.",
        }


class _Embedder(EmbeddingBackend):
    model_name = "fixture-embedding"

    def encode(self, texts):
        return np.tile(np.asarray((1.0, 0.0), dtype=np.float64), (len(texts), 1))


def _pass_command(argv, *, cwd, timeout):
    del argv, cwd, timeout
    return CommandOutcome(returncode=0, stdout="ok", stderr="")


def _config(tmp_path: Path) -> ForgeConfig:
    forge_root = tmp_path / "forge"
    for name in ("schemas", "prompts"):
        shutil.copytree(REPO_ROOT / "forge" / name, forge_root / name)
    shutil.copy2(REPO_ROOT / "forge" / "editable_surface.yaml", forge_root / "editable_surface.yaml")
    (forge_root / "ledger.jsonl").write_text(
        json.dumps({"event": "initialized", "schema_version": "forge-ledger.v1"}) + "\n",
        encoding="utf-8",
    )
    rules = tmp_path / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "test.mdc").write_text("# Existing\n\nKeep contract checks.\n", encoding="utf-8")
    return ForgeConfig.load(ForgePaths.discover(repo_root=tmp_path, transcripts_root=tmp_path / "transcripts"))


def _proposal(config: ForgeConfig, tmp_path: Path) -> Path:
    pattern_path = tmp_path / "patterns.jsonl"
    pattern_path.write_text(
        json.dumps(
            {
                "schema_version": "forge-failure-pattern.v1",
                "pattern_id": "fp_0123456789abcdef",
                "title": "bounded recovery gap",
                "verifier_cause": "contract failed after repeated tool error",
                "agent_behavior_cause": "the retry lost verifier context",
                "exposed_mechanism": "recovery handoff is absent",
                "occurrence_count": 2,
                "evidence_refs": [
                    {
                        "source_id": "transcript:fixture",
                        "source_kind": "transcript",
                        "locator": "line:1",
                        "excerpt": "structured error",
                        "digest": "a" * 64,
                    }
                ],
                "source_kinds": ["transcript"],
                "centroid_digest": "b" * 64,
                "editable_target": ".cursor/rules/test.mdc",
                "editable_component": "repository_agent_rules",
                "surface_status": "in-surface",
                "surface_similarity": 0.9,
                "preserve_behaviors": ["contract check passes"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = propose_changes(
        config=config,
        failure_patterns_path=pattern_path,
        backend=_ProposalBackend(),
        embedder=_Embedder(),
        output_dir=tmp_path / "proposal-output",
    )
    return result.proposal_dirs[0]


def test_validation_passes_complete_bundle_and_does_not_apply(tmp_path: Path) -> None:
    config = _config(tmp_path)
    proposal_dir = _proposal(config, tmp_path)
    target = tmp_path / ".cursor" / "rules" / "test.mdc"
    before = target.read_text(encoding="utf-8")

    result = validate_proposal(
        config=config,
        proposal_dir=proposal_dir,
        relevance_backend=_RelevanceBackend(),
        command_runner=_pass_command,
    )

    assert result.status == "PASS"
    assert target.read_text(encoding="utf-8") == before


def test_validation_without_external_judge_blocks(tmp_path: Path) -> None:
    config = _config(tmp_path)
    proposal_dir = _proposal(config, tmp_path)
    result = validate_proposal(
        config=config,
        proposal_dir=proposal_dir,
        relevance_backend=None,
        command_runner=_pass_command,
    )
    assert result.status == "BLOCK"
    assert any(check["name"] == "targeted-relevance-held-in" for check in result.checks)


def test_apply_requires_named_human_and_rechecks_hashes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    proposal_dir = _proposal(config, tmp_path)
    validation = validate_proposal(
        config=config,
        proposal_dir=proposal_dir,
        relevance_backend=_RelevanceBackend(),
        command_runner=_pass_command,
    )
    with pytest.raises(ApplyError, match="named human reviewer"):
        apply_proposal(
            config=config,
            proposal_dir=proposal_dir,
            validation_report_path=validation.report_path,
            human_approved_by="",
        )
    (proposal_dir / "patch.diff").write_text(
        (proposal_dir / "patch.diff").read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ApplyError, match="Patch changed after validation"):
        apply_proposal(
            config=config,
            proposal_dir=proposal_dir,
            validation_report_path=validation.report_path,
            human_approved_by="reviewer@example",
        )


def test_human_apply_and_reject_are_auditable(tmp_path: Path) -> None:
    config = _config(tmp_path)
    proposal_dir = _proposal(config, tmp_path)
    validation = validate_proposal(
        config=config,
        proposal_dir=proposal_dir,
        relevance_backend=_RelevanceBackend(),
        command_runner=_pass_command,
    )
    applied = apply_proposal(
        config=config,
        proposal_dir=proposal_dir,
        validation_report_path=validation.report_path,
        human_approved_by="reviewer@example",
    )
    assert applied.decision == "applied"
    assert "# Bounded recovery" in (tmp_path / ".cursor" / "rules" / "test.mdc").read_text(encoding="utf-8")
    ledger_events = [json.loads(line) for line in config.paths.ledger_path.read_text(encoding="utf-8").splitlines()]
    assert ledger_events[-1]["prediction"]["baseline_value"] == 2

    second_root = tmp_path / "second"
    second_root.mkdir()
    second_config = _config(second_root)
    rejected_dir = _proposal(second_config, second_root)
    before = (second_root / ".cursor" / "rules" / "test.mdc").read_text(encoding="utf-8")
    rejected = reject_proposal(
        config=second_config,
        proposal_dir=rejected_dir,
        human_approved_by="reviewer@example",
        reason="risk is not covered by the current held-out suite",
    )
    assert rejected.decision == "rejected"
    assert (second_root / ".cursor" / "rules" / "test.mdc").read_text(encoding="utf-8") == before
