from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from volvence_forge.apply import ApplyError, apply_proposal
from volvence_forge.config import ForgeConfig, ForgePaths
from volvence_forge.foundation import EmbeddingBackend, StructuredBackend
from volvence_forge.propose import propose_changes
from volvence_forge.validate import validate_proposal
from volvence_forge.validate import CommandOutcome


REPO_ROOT = Path(__file__).resolve().parents[2]


class _ProposalBackend(StructuredBackend):
    backend_name = "test-replay"
    model_name = "fixture"

    def complete_json(self, *, system, user, schema):
        del system, user, schema
        return {
            "target": ".cursor/rules/test.mdc",
            "operation": "append_section",
            "section_content": "# Validated handoff\n\nKeep the existing check before retry.",
            "root_cause": "The handoff is absent.",
            "targeted_fix": "Document the bounded handoff.",
            "prediction": {
                "metric": "pattern_occurrence_count",
                "direction": "decrease",
                "expected_delta": -1,
                "evaluation_window": "next_mine_run",
            },
            "at_risk_regressions": ["rule verbosity"],
            "preserve_behaviors": ["Keep the existing check"],
        }


class _RelevanceBackend(StructuredBackend):
    backend_name = "test-relevance"
    model_name = "fixture"

    def complete_json(self, *, system, user, schema):
        del system, user, schema
        return {
            "relevant": True,
            "evidence_alignment": True,
            "preservation_assessment": True,
            "reason": "The append-only section addresses the cited pattern.",
        }


class _Embedder(EmbeddingBackend):
    model_name = "fixture-embedding"

    def encode(self, texts):
        return np.ones((len(texts), 2), dtype=np.float64)


def _config(tmp_path: Path) -> ForgeConfig:
    forge_root = tmp_path / "forge"
    for name in ("schemas", "prompts"):
        shutil.copytree(REPO_ROOT / "forge" / name, forge_root / name)
    shutil.copy2(REPO_ROOT / "forge" / "editable_surface.yaml", forge_root / "editable_surface.yaml")
    (forge_root / "ledger.jsonl").write_text("", encoding="utf-8")
    rules = tmp_path / ".cursor" / "rules"
    rules.mkdir(parents=True)
    (rules / "test.mdc").write_text("# Existing\n\nKeep the existing check.\n", encoding="utf-8")
    paths = ForgePaths.discover(repo_root=tmp_path, transcripts_root=tmp_path / "transcripts")
    config = ForgeConfig.load(paths)
    subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.email", "forge@example.invalid"), cwd=tmp_path, check=True)
    subprocess.run(("git", "config", "user.name", "Forge Test"), cwd=tmp_path, check=True)
    subprocess.run(("git", "add", ".cursor/rules/test.mdc"), cwd=tmp_path, check=True)
    subprocess.run(("git", "commit", "-qm", "fixture"), cwd=tmp_path, check=True)
    return config


def _patterns(tmp_path: Path) -> Path:
    path = tmp_path / "patterns.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "forge-failure-pattern.v1",
                "pattern_id": "fp_0123456789abcdef",
                "title": "bounded handoff",
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
                "preserve_behaviors": ["Keep the existing check"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _proposal(tmp_path: Path):
    config = _config(tmp_path)
    result = propose_changes(
        config=config,
        failure_patterns_path=_patterns(tmp_path),
        backend=_ProposalBackend(),
        embedder=_Embedder(),
        output_dir=tmp_path / "proposal-output",
    )
    return config, result.proposal_dirs[0]


def _runner(argv, *, cwd, timeout):
    del argv, cwd, timeout
    return CommandOutcome(returncode=0, stdout="ok", stderr="")


def test_validate_is_fail_closed_without_relevance_backend(tmp_path: Path) -> None:
    config, proposal_dir = _proposal(tmp_path)
    result = validate_proposal(
        config=config,
        proposal_dir=proposal_dir,
        relevance_backend=None,
        command_runner=_runner,
    )
    assert result.status == "BLOCK"
    assert any(check["name"] == "targeted-relevance-held-in" and check["status"] == "BLOCK" for check in result.checks)


def test_validate_then_apply_requires_named_human_and_records_ledger(tmp_path: Path) -> None:
    config, proposal_dir = _proposal(tmp_path)
    validation = validate_proposal(
        config=config,
        proposal_dir=proposal_dir,
        relevance_backend=_RelevanceBackend(),
        command_runner=_runner,
    )
    assert validation.status == "PASS"

    with pytest.raises(ApplyError, match="named human reviewer"):
        apply_proposal(
            config=config,
            proposal_dir=proposal_dir,
            validation_report_path=validation.report_path,
            human_approved_by=" ",
        )
    result = apply_proposal(
        config=config,
        proposal_dir=proposal_dir,
        validation_report_path=validation.report_path,
        human_approved_by="external-reviewer",
    )
    assert result.decision == "applied"
    assert "# Validated handoff" in (tmp_path / ".cursor" / "rules" / "test.mdc").read_text()
    events = [json.loads(line) for line in config.paths.ledger_path.read_text().splitlines() if line]
    assert events[-1]["decision"] == "applied"
    assert events[-1]["reviewer"] == "external-reviewer"
