from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from volvence_forge.cli import main
from volvence_forge.config import ForgeConfig, ForgePaths
from volvence_forge.foundation import canonical_json, sha256_bytes, sha256_text
from volvence_forge.research_control import list_research_inbox, validate_research_request
from volvence_forge.research_opportunity import (
    ResearchOpportunityError,
    scan_research_opportunities,
    validate_research_opportunity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class OpportunityFixture:
    config: ForgeConfig
    task_manifest: Path
    task_project: Path
    executable: Path
    run_root: Path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _ref(path: Path, *, root: Path) -> dict[str, str]:
    return {"locator": path.relative_to(root).as_posix(), "sha256": _sha(path)}


def _config(tmp_path: Path) -> ForgeConfig:
    repo = tmp_path / "repo"
    forge = repo / "forge"
    forge.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "forge" / "editable_surface.yaml", forge)
    shutil.copytree(REPO_ROOT / "forge" / "schemas", forge / "schemas")
    shutil.copy2(REPO_ROOT / "forge" / "research_task_registry.yaml", forge)
    return ForgeConfig.load(ForgePaths.discover(repo_root=repo, transcripts_root=repo / "transcripts"))


def _project_manifest(task_project: Path) -> dict[str, Any]:
    task_yaml = task_project / "task.yaml"
    content = task_yaml.read_bytes()
    digest = hashlib.sha256()
    digest.update(b"task.yaml\0")
    digest.update(content)
    digest.update(b"\0")
    return {
        "schema_version": "task_project_manifest.v1",
        "source": "external_task_project",
        "task_id": "memory_inheritance_project",
        "task_ref": "task:memory_inheritance_project",
        "path": str(task_project),
        "descriptor_path": str(task_yaml),
        "sha256": digest.hexdigest(),
        "files": [
            {
                "path": "task.yaml",
                "sha256": sha256_bytes(content),
                "bytes": len(content),
            }
        ],
    }


def _fixture(tmp_path: Path) -> OpportunityFixture:
    config = _config(tmp_path)
    repo = config.paths.repo_root
    task_project = repo / "research" / "praxist_tasks" / "memory_inheritance_project"
    task_project.mkdir(parents=True)
    (task_project / "task.yaml").write_text(
        "\n".join(
            (
                "schema_version: 1",
                "task_id: memory_inheritance_project",
                "praxist_plugins:",
                "  task_ref: task:memory_inheritance_project",
                "  workflow:",
                "    stage: workflow_stage:research_loop",
                "",
            )
        ),
        encoding="utf-8",
    )
    manifest = _project_manifest(task_project)
    baseline = _write_json(repo / "research/contracts/baseline.json", {"baseline": "v1"})
    protocol = _write_json(repo / "research/contracts/protocol.json", {"protocol": "sealed-v1"})
    task_manifest = _write_json(
        repo / "research/tasks/memory_inheritance/task.json",
        {
            "schema_version": "forge-research-task.v1",
            "task_id": "memory_inheritance",
            "claim_id": "claim:memory-inheritance",
            "owner": "vz-memory",
            "objective": "Improve inheritance through a bounded Praxist research task.",
            "capability_axes": ["appendable", "readable", "learnable"],
            "source_base_revision": "a" * 40,
            "baseline": _ref(baseline, root=repo),
            "praxist": {
                "task_project_id": "memory_inheritance_project",
                "task_project_manifest_sha256": manifest["sha256"],
            },
            "sandbox": {
                "editable_roots": ["research_surface"],
                "protected_roots": ["research/contracts", "research/tasks"],
                "praxist_can_modify_task_contract": False,
                "praxist_can_modify_formal_evaluator": False,
                "praxist_can_read_sealed_holdout": False,
                "praxist_can_change_production_wiring": False,
                "praxist_can_access_production_credentials": False,
            },
            "validation": {
                "development_evaluator_id": "praxist:development-evaluator",
                "formal_validator_id": "volvence:sealed-validator",
                "formal_protocol": _ref(protocol, root=repo),
                "shadow_required_checks": ["formal_quality", "rollback_drill"],
                "active_required_checks": [
                    "formal_quality",
                    "rollback_drill",
                    "shadow_observation",
                ],
            },
            "release": {
                "mode": "runtime_wiring",
                "target": "memory_inheritance_policy",
                "initial_wiring": "disabled",
                "gate_authority": "volvence_zero.credit.gate.evaluate_gate_reasons",
                "rollback_instructions": "Restore the previous content hash.",
            },
            "authority": {
                "praxist_is_research_retention_authority_only": True,
                "evaluation_is_learning_source": False,
                "production_promotion_authorized": False,
            },
        },
    )
    executable = tmp_path / "tools" / "praxist"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    executable.chmod(0o755)
    return OpportunityFixture(
        config=config,
        task_manifest=task_manifest,
        task_project=task_project,
        executable=executable,
        run_root=(tmp_path / "runs").resolve(),
    )


def _pattern(
    *,
    component: str | None = "repository_agent_rules",
    target: str | None = ".cursor/rules/test.mdc",
    surface_status: str = "in-surface",
    occurrence_count: int = 3,
    marker: str = "a",
    exposed_mechanism: str = "bounded retry state is not preserved",
) -> dict[str, Any]:
    evidence_refs = [
        {
            "source_id": f"source-{marker}",
            "source_kind": "transcript",
            "locator": "turn:4",
            "excerpt": "typed failure evidence",
            "digest": marker * 64,
        }
    ]
    base = {
        "verifier_cause": f"verifier contract failed {marker}",
        "agent_behavior_cause": f"agent lost bounded state {marker}",
        "exposed_mechanism": exposed_mechanism,
        "evidence_refs": evidence_refs,
    }
    return {
        "schema_version": "forge-failure-pattern.v3",
        "pattern_id": "fp_" + sha256_text(canonical_json(base))[:16],
        "title": exposed_mechanism,
        **base,
        "occurrence_count": occurrence_count,
        "source_kinds": ["transcript"],
        "centroid_digest": marker * 64,
        "editable_target": target,
        "editable_component": component,
        "surface_status": surface_status,
        "surface_similarity": 0.91 if surface_status == "in-surface" else 0.12,
        "preserve_behaviors": ["preserve passing behavior"],
    }


def _write_patterns(config: ForgeConfig, patterns: list[dict[str, Any]]) -> Path:
    path = config.paths.artifacts_root / "forge_mine_fixture" / "failure_patterns.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(pattern) + "\n" for pattern in patterns),
        encoding="utf-8",
    )
    return path


def _write_registry(
    fixture: OpportunityFixture,
    *,
    mappings: list[dict[str, Any]],
    max_new_requests_per_scan: int = 1,
) -> Path:
    path = fixture.config.paths.forge_root / "test_research_task_registry.yaml"
    payload = {
        "schema_version": "forge-research-task-registry.v1",
        "policy": {"max_new_requests_per_scan": max_new_requests_per_scan},
        "mappings": mappings,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _mapping(
    fixture: OpportunityFixture,
    *,
    component: str = "repository_agent_rules",
    target: str | None = None,
    mapping_id: str = "memory_inheritance",
) -> dict[str, Any]:
    return {
        "mapping_id": mapping_id,
        "binding_revision": "fixture-v1",
        "match": {
            "editable_component": component,
            "editable_target": target,
        },
        "task_manifest": str(fixture.task_manifest),
        "task_project": str(fixture.task_project),
        "praxist_executable": str(fixture.executable),
        "run_root": str(fixture.run_root),
        "launch": {
            "config_file": None,
            "agent_system": "claude_sdk",
            "runtime": "agent_runtime:claude_sdk",
            "codex_native": False,
            "model_provider": "model_provider:fixture",
            "model": "fixture-model",
            "strategy": "auto",
            "cohort": 2,
            "generations": 3,
            "startup_timeout_seconds": 30,
        },
    }


def test_unmapped_typed_pattern_is_preserved_without_request(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    patterns = _write_patterns(fixture.config, [_pattern()])
    registry = _write_registry(fixture, mappings=[])

    result = scan_research_opportunities(
        config=fixture.config,
        failure_patterns_path=patterns,
        registry_path=registry,
    )

    assert result.discovered_count == 1
    assert result.new_request_count == 0
    assert result.statuses[0].state == "NEEDS_TASK_DESIGN"
    assert result.statuses[0].blocker_codes == ("NO_REGISTERED_TASK",)
    assert not (fixture.config.paths.artifacts_root / "research_control").exists()
    opportunity = validate_research_opportunity(
        config=fixture.config,
        opportunity_path=result.statuses[0].opportunity_path,
    )
    assert opportunity["nomination"]["readiness"] == "ROUTABLE"
    assert opportunity["authority"]["research_start_authorized"] is False
    routing = json.loads(result.statuses[0].routing_path.read_text(encoding="utf-8"))
    assert routing["decision"] == "NEEDS_TASK_DESIGN"
    assert routing["request"] is None


def test_exact_mapping_submits_request_but_never_approves_or_starts(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    patterns = _write_patterns(fixture.config, [_pattern()])
    registry = _write_registry(fixture, mappings=[_mapping(fixture)])

    result = scan_research_opportunities(
        config=fixture.config,
        failure_patterns_path=patterns,
        registry_path=registry,
    )

    status = result.statuses[0]
    assert result.new_request_count == 1
    assert status.state == "AWAITING_RESEARCH_APPROVAL"
    assert status.request_path is not None
    request = validate_research_request(
        config=fixture.config,
        request_path=status.request_path,
    )
    assert request["trigger"]["kind"] == "forge_failure_pattern"
    assert request["trigger"]["submitted_by"] == "forge:research-opportunity-scanner.v1"
    assert request["trigger"]["evidence"] == [
        {
            "locator": status.opportunity_path.relative_to(fixture.config.paths.repo_root).as_posix(),
            "sha256": _sha(status.opportunity_path),
        }
    ]
    assert request["authority"]["research_start_authorized"] is False
    assert list(status.request_path.parent.glob("approvals/*.json")) == []
    assert list(status.request_path.parent.glob("events/*.json")) == []
    assert list_research_inbox(config=fixture.config)[0].state == "AWAITING_RESEARCH_APPROVAL"
    route = json.loads(status.routing_path.read_text(encoding="utf-8"))
    assert route["mapping"]["owner"] == "vz-memory"
    assert route["mapping"]["capability_axes"] == [
        "appendable",
        "readable",
        "learnable",
    ]


def test_scan_recovers_request_when_route_write_was_interrupted(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    patterns = _write_patterns(fixture.config, [_pattern()])
    registry = _write_registry(fixture, mappings=[_mapping(fixture)])
    first = scan_research_opportunities(
        config=fixture.config,
        failure_patterns_path=patterns,
        registry_path=registry,
    )
    original = first.statuses[0]
    assert original.request_path is not None
    original.routing_path.unlink()
    request = json.loads(original.request_path.read_text(encoding="utf-8"))
    run_dir = Path(request["launch"]["run_dir"])
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("already launched\n", encoding="utf-8")

    recovered = scan_research_opportunities(
        config=fixture.config,
        failure_patterns_path=patterns,
        registry_path=registry,
    )

    status = recovered.statuses[0]
    assert recovered.new_request_count == 0
    assert status.request_id == original.request_id
    assert status.request_path == original.request_path
    assert status.routing_path.exists()
    assert len(list_research_inbox(config=fixture.config)) == 1


def test_routing_never_uses_causal_prose_as_a_component_match(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    pattern = _pattern(
        component="forge_analysis_prompts",
        exposed_mechanism="repository_agent_rules should be investigated",
    )
    patterns = _write_patterns(fixture.config, [pattern])
    registry = _write_registry(fixture, mappings=[_mapping(fixture)])

    result = scan_research_opportunities(
        config=fixture.config,
        failure_patterns_path=patterns,
        registry_path=registry,
    )

    assert result.new_request_count == 0
    assert result.statuses[0].state == "NEEDS_TASK_DESIGN"
    assert result.statuses[0].blocker_codes == ("NO_REGISTERED_TASK",)


def test_submission_limit_uses_typed_priority_and_advances_idempotently(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    low = _pattern(occurrence_count=2, marker="a")
    high = _pattern(occurrence_count=9, marker="b")
    patterns = _write_patterns(fixture.config, [low, high])
    registry = _write_registry(
        fixture,
        mappings=[_mapping(fixture)],
        max_new_requests_per_scan=1,
    )

    first = scan_research_opportunities(
        config=fixture.config,
        failure_patterns_path=patterns,
        registry_path=registry,
    )

    assert first.new_request_count == 1
    assert [status.priority_score for status in first.statuses] == [9, 2]
    assert [status.state for status in first.statuses] == [
        "AWAITING_RESEARCH_APPROVAL",
        "DEFERRED_BY_SCAN_LIMIT",
    ]

    second = scan_research_opportunities(
        config=fixture.config,
        failure_patterns_path=patterns,
        registry_path=registry,
    )

    assert second.new_request_count == 1
    assert [status.state for status in second.statuses] == [
        "AWAITING_RESEARCH_APPROVAL",
        "AWAITING_RESEARCH_APPROVAL",
    ]
    assert len(list_research_inbox(config=fixture.config)) == 2


def test_registry_overlap_and_pattern_identity_corruption_fail_loudly(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    pattern = _pattern()
    patterns = _write_patterns(fixture.config, [pattern])
    broad = _mapping(fixture)
    exact = _mapping(
        fixture,
        target=".cursor/rules/test.mdc",
        mapping_id="memory_inheritance_exact",
    )
    registry = _write_registry(fixture, mappings=[broad, exact])

    with pytest.raises(ResearchOpportunityError, match="overlap"):
        scan_research_opportunities(
            config=fixture.config,
            failure_patterns_path=patterns,
            registry_path=registry,
        )

    corrupted = dict(pattern)
    corrupted["pattern_id"] = "fp_" + "0" * 16
    corrupted_patterns = _write_patterns(fixture.config, [corrupted])
    empty_registry = _write_registry(fixture, mappings=[])
    with pytest.raises(ResearchOpportunityError, match="pattern_id"):
        scan_research_opportunities(
            config=fixture.config,
            failure_patterns_path=corrupted_patterns,
            registry_path=empty_registry,
        )


def test_research_scan_cli_emits_machine_readable_nomination(tmp_path: Path, capsys) -> None:
    fixture = _fixture(tmp_path)
    patterns = _write_patterns(fixture.config, [_pattern()])
    registry = _write_registry(fixture, mappings=[])

    exit_code = main(
        [
            "--repo-root",
            str(fixture.config.paths.repo_root),
            "research-scan",
            str(patterns),
            "--registry",
            str(registry),
            "--once",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["discovered_count"] == 1
    assert payload["new_request_count"] == 0
    assert payload["opportunities"][0]["state"] == "NEEDS_TASK_DESIGN"
