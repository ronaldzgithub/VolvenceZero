from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from volvence_forge.apply import ApplyError, apply_proposal, reject_proposal
from volvence_forge.config import ForgeConfig, ForgePaths
from volvence_forge.foundation import EmbeddingBackend, StructuredBackend
from volvence_forge.foundation import sha256_text
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


class _RuntimeProposalBackend(StructuredBackend):
    backend_name = "test-replay"
    model_name = "runtime-proposal-fixture"

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
                "  - scenario_id: repair_02\n"
                "    family: relationship_repair\n"
                "    semantic_routing:\n"
                "      method: embedding_similarity_plus_schema_bound_structured_output\n"
            ),
            "root_cause": "The runtime semantic asset lacks a repair regime scene.",
            "targeted_fix": "Append one reviewed repair scene.",
            "prediction": {
                "metric": "pattern_occurrence_count",
                "direction": "decrease",
                "expected_delta": -1,
                "evaluation_window": "next_mine_run",
            },
            "at_risk_regressions": ["repair scene overlaps a boundary regime"],
            "preserve_behaviors": ["boundary negative remains passing"],
        }


class _RuntimeValidationBackend(StructuredBackend):
    backend_name = "external-test-judge"
    model_name = "runtime-suite-fixture"

    def complete_json(self, *, system, user, schema):
        del system, user
        if schema.get("$id") == "forge.relevance-judgment.v1":
            return {
                "relevant": True,
                "evidence_alignment": True,
                "preservation_assessment": True,
                "reason": "The scene is a narrow response to the cited repair failure.",
            }
        return {
            "baseline_passed_test_ids": ["route_01"],
            "candidate_passed_test_ids": ["route_01", "coherence_01"],
            "reason": "The candidate adds semantic repair coverage while retaining the negative case.",
        }


def _pass_command(argv, *, cwd, timeout):
    del argv, cwd, timeout
    return CommandOutcome(returncode=0, stdout="ok", stderr="")


def _config(tmp_path: Path, *, enable_runtime_fixture: bool = False) -> ForgeConfig:
    forge_root = tmp_path / "forge"
    for name in ("schemas", "prompts"):
        shutil.copytree(REPO_ROOT / "forge" / name, forge_root / name)
    shutil.copy2(REPO_ROOT / "forge" / "editable_surface.yaml", forge_root / "editable_surface.yaml")
    if enable_runtime_fixture:
        policy_path = forge_root / "editable_surface.yaml"
        policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        policy["editable"].append(
            {
                "component": "character_scenario_semantics_fixture",
                "paths": [
                    "packages/lifeform-domain-character/src/lifeform_domain_character/"
                    "scenario_packages/*/scenes.yaml",
                    "packages/lifeform-domain-character/src/lifeform_domain_character/"
                    "scenario_packages/*/ssot_fragment.json",
                ],
                "semantic_description": "Test-only runtime semantic asset fixture.",
                "requires_offline_gate": True,
                "validation": {
                    "frozen_suite": (
                        "packages/lifeform-domain-character/src/lifeform_domain_character/"
                        "scenario_packages/*/test_suite.yaml"
                    ),
                    "candidate": [["python", "fixture-validator.py", "{candidate_path}"]],
                    "held_in": [["pytest", "fixture-held-in", "-q"]],
                    "held_out": [["pytest", "fixture-held-out", "-q"]],
                },
            }
        )
        policy["read_only"] = [
            pattern
            for pattern in policy["read_only"]
            if pattern
            not in {
                "packages/**",
                "packages/lifeform-domain-character/src/**/scenario_packages/*/scenes.yaml",
                "packages/lifeform-domain-character/src/**/scenario_packages/*/ssot_fragment.json",
            }
        ]
        policy_path.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")
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


def _runtime_proposal(config: ForgeConfig, tmp_path: Path) -> tuple[Path, Path]:
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
        (
            "routing_tests:\n"
            "  - test_id: route_01\n"
            "llm_evaluation:\n"
            "  semantic_coherence:\n"
            "    - case_id: coherence_01\n"
        ),
        encoding="utf-8",
    )
    pattern_path = tmp_path / "runtime-patterns.jsonl"
    pattern_path.write_text(
        json.dumps(
            {
                "schema_version": "forge-failure-pattern.v2",
                "pattern_id": "fp_1123456789abcdef",
                "title": "runtime repair gap",
                "verifier_cause": "bench repair rubric failed",
                "agent_behavior_cause": "the assistant missed the rupture",
                "exposed_mechanism": "reviewed scene coverage is incomplete",
                "occurrence_count": 1,
                "evidence_refs": [
                    {
                        "source_id": "bench_bundle:fixture",
                        "source_kind": "bench_bundle",
                        "locator": "arc:fixture/session:1/turn:1",
                        "excerpt": "repair score=1",
                        "digest": "c" * 64,
                    }
                ],
                "source_kinds": ["bench_bundle"],
                "centroid_digest": "d" * 64,
                "editable_target": target.relative_to(tmp_path).as_posix(),
                "editable_component": "character_scenario_semantics_fixture",
                "surface_status": "in-surface",
                "surface_similarity": 0.9,
                "preserve_behaviors": ["boundary negative remains passing"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = propose_changes(
        config=config,
        failure_patterns_path=pattern_path,
        backend=_RuntimeProposalBackend(),
        embedder=_Embedder(),
        output_dir=tmp_path / "runtime-proposal-output",
    )
    return result.proposal_dirs[0], target


def _write_gate_decision(
    proposal_dir: Path,
    validation_path: Path,
    *,
    decision: str,
) -> None:
    patch = (proposal_dir / "patch.diff").read_text(encoding="utf-8")
    manifesto_text = (proposal_dir / "manifesto.json").read_text(encoding="utf-8")
    manifesto = json.loads(manifesto_text)
    validation_text = validation_path.read_text(encoding="utf-8")
    payload = {
        "schema_version": "forge-gate-decision.v1",
        "proposal_id": manifesto["proposal_id"],
        "target": manifesto["target"],
        "decision": decision,
        "reasons": [] if decision == "ALLOW" else ["validation_delta 0.000 below required margin 0.050"],
        "desired_gate": "offline",
        "inputs": {
            "patch_sha256": sha256_text(patch),
            "manifesto_sha256": sha256_text(manifesto_text),
            "validation_sha256": sha256_text(validation_text),
        },
        "metrics": {
            "baseline_pass_rate": 0.5,
            "candidate_pass_rate": 1.0,
            "validation_delta": 0.5,
            "capacity_cost": 0.1,
            "contract_integrity": 1.0,
            "rollback_resilience": 1.0,
        },
        "authority": "volvence_zero.credit.gate.evaluate_gate_reasons",
        "created_at": "2026-08-01T00:00:00Z",
    }
    (proposal_dir / "gate_decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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


def test_runtime_validation_emits_delta_and_apply_requires_offline_allow(tmp_path: Path) -> None:
    config = _config(tmp_path, enable_runtime_fixture=True)
    proposal_dir, target = _runtime_proposal(config, tmp_path)
    before = target.read_text(encoding="utf-8")
    validated_candidates: list[str] = []

    def candidate_aware_runner(argv, *, cwd, timeout):
        del cwd, timeout
        if "fixture-validator.py" in argv:
            candidate_path = Path(argv[-1])
            assert candidate_path.is_file()
            assert "scenario_id: repair_02" in candidate_path.read_text(
                encoding="utf-8"
            )
            validated_candidates.append(str(candidate_path))
        return CommandOutcome(returncode=0, stdout="ok", stderr="")

    validation = validate_proposal(
        config=config,
        proposal_dir=proposal_dir,
        relevance_backend=_RuntimeValidationBackend(),
        command_runner=candidate_aware_runner,
    )
    report = json.loads(validation.report_path.read_text(encoding="utf-8"))

    assert validation.status == "PASS"
    assert report["runtime_gate_evidence"]["validation_delta"] == 0.5
    assert report["runtime_gate_evidence"]["rollback_resilience"] is True
    assert len(validated_candidates) == 1
    assert "{candidate_path}" not in validated_candidates[0]
    assert target.read_text(encoding="utf-8") == before
    with pytest.raises(ApplyError, match="requires readable OFFLINE gate decision"):
        apply_proposal(
            config=config,
            proposal_dir=proposal_dir,
            validation_report_path=validation.report_path,
            human_approved_by="reviewer@example",
        )

    _write_gate_decision(proposal_dir, validation.report_path, decision="BLOCK")
    with pytest.raises(ApplyError, match="blocked by OFFLINE gate"):
        apply_proposal(
            config=config,
            proposal_dir=proposal_dir,
            validation_report_path=validation.report_path,
            human_approved_by="reviewer@example",
        )

    _write_gate_decision(proposal_dir, validation.report_path, decision="ALLOW")
    applied = apply_proposal(
        config=config,
        proposal_dir=proposal_dir,
        validation_report_path=validation.report_path,
        human_approved_by="reviewer@example",
    )
    assert applied.decision == "applied"
    assert "scenario_id: repair_02" in target.read_text(encoding="utf-8")
    events = [json.loads(line) for line in config.paths.ledger_path.read_text().splitlines()]
    assert events[-1]["gate_decision_ref"]["decision"] == "ALLOW"
