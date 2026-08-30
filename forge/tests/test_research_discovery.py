from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
import volvence_forge.research_loop as research_loop_module

from volvence_forge.config import ForgeConfig, ForgePaths
from volvence_forge.foundation import (
    ReplayStructuredBackend,
    canonical_json,
    sha256_bytes,
    sha256_text,
)
from volvence_forge.research_control import (
    ResearchControlStatus,
    inspect_research_request,
    review_research_request,
    validate_research_request,
)
from volvence_forge.research_discovery import (
    CodexNativeResearchDiscoveryBackend,
    ReplayResearchDiscoveryBackend,
    ResearchDiscoveryError,
    discover_research_topics,
    review_research_topic,
    seal_research_demand,
    submit_bound_topic_for_a0,
    validate_research_demand_binding,
    validate_research_discovery_run,
    validate_research_topic_proposal,
)
from volvence_forge.research_loop import ResearchLoopError, run_demand_research_loop_once

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DiscoveryFixture:
    config: ForgeConfig
    demand_path: Path
    registry_path: Path
    source_path: Path
    source_sha256: str


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _ref(path: Path, *, root: Path) -> dict[str, str]:
    return {"locator": path.relative_to(root).as_posix(), "sha256": _sha(path)}


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
        "task_id": "readable_research_project",
        "task_ref": "task:readable_research_project",
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


def _authority() -> dict[str, bool]:
    return {
        "discovery_only": True,
        "human_topic_binding_required": True,
        "human_a0_required": True,
        "research_start_authorized": False,
        "formal_validation_performed": False,
        "production_promotion_authorized": False,
        "runtime_wiring_changed": False,
        "evaluation_is_learning_source": False,
    }


def _demand_identity(payload: dict[str, Any]) -> str:
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"demand_id", "created_at"}
    }
    return "research-demand:" + sha256_text(canonical_json(body))


def _request_identity(payload: dict[str, Any]) -> str:
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"request_id", "created_at"}
    }
    return "research-request:" + sha256_text(canonical_json(body))


def _fixture(tmp_path: Path) -> DiscoveryFixture:
    repo = tmp_path / "repo"
    forge = repo / "forge"
    forge.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "forge/editable_surface.yaml", forge)
    shutil.copytree(REPO_ROOT / "forge/schemas", forge / "schemas")
    (forge / "prompts").mkdir()
    for name in (
        "research_discovery.system.md",
        "research_discovery.user.md",
    ):
        shutil.copy2(REPO_ROOT / "forge/prompts" / name, forge / "prompts" / name)
    config = ForgeConfig.load(
        ForgePaths.discover(repo_root=repo, transcripts_root=repo / "transcripts")
    )

    source_path = repo / "research/industry_four_ables/readable.md"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "A bounded state reader should publish immutable named state before a consumer acts.\n",
        encoding="utf-8",
    )
    relationship_failure = _write_json(
        repo / "artifacts/relationship_lab/frozen_failure.json",
        {
            "schema_version": "relationship-lab-failure.v1",
            "gate": "named_state_action_transmission",
            "passed": False,
        },
    )

    task_project = repo / "research/praxist_tasks/readable_research_project"
    task_project.mkdir(parents=True)
    (task_project / "task.yaml").write_text(
        "\n".join(
            (
                "schema_version: 1",
                "task_id: readable_research_project",
                "praxist_plugins:",
                "  task_ref: task:readable_research_project",
                "  workflow:",
                "    stage: workflow_stage:research_loop",
                "",
            )
        ),
        encoding="utf-8",
    )
    manifest = _project_manifest(task_project)
    baseline = _write_json(repo / "research/contracts/readable/baseline.json", {"v": 1})
    protocol = _write_json(repo / "research/contracts/readable/protocol.json", {"v": 1})
    task_manifest = _write_json(
        repo / "research/tasks/readable_research/task.json",
        {
            "schema_version": "forge-research-task.v1",
            "task_id": "readable_research",
            "claim_id": "claim:readable-state-action",
            "owner": "vz-substrate",
            "objective": "Test bounded named-state readout mechanisms.",
            "capability_axes": ["readable", "steerable"],
            "source_base_revision": "a" * 40,
            "baseline": _ref(baseline, root=repo),
            "praxist": {
                "task_project_id": "readable_research_project",
                "task_project_manifest_sha256": manifest["sha256"],
            },
            "sandbox": {
                "editable_roots": ["research/candidate_surfaces/readable"],
                "protected_roots": ["research/contracts", "research/tasks"],
                "praxist_can_modify_task_contract": False,
                "praxist_can_modify_formal_evaluator": False,
                "praxist_can_read_sealed_holdout": False,
                "praxist_can_change_production_wiring": False,
                "praxist_can_access_production_credentials": False,
            },
            "validation": {
                "development_evaluator_id": "praxist:readable-development",
                "formal_validator_id": "volvence:readable-formal",
                "formal_protocol": _ref(protocol, root=repo),
                "shadow_required_checks": ["formal_quality", "rollback_drill"],
                "active_required_checks": [
                    "formal_quality",
                    "rollback_drill",
                    "shadow_observation",
                ],
            },
            "release": {
                "mode": "evidence_only",
                "target": "readable_state_evidence",
                "initial_wiring": "disabled",
                "gate_authority": "volvence_zero.credit.gate.evaluate_gate_reasons",
                "rollback_instructions": "Retain the previous evidence bundle.",
            },
            "authority": {
                "praxist_is_research_retention_authority_only": True,
                "evaluation_is_learning_source": False,
                "production_promotion_authorized": False,
            },
        },
    )
    executable = tmp_path / "tools/praxist"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    executable.chmod(0o755)
    registry_path = forge / "research_task_registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "forge-research-task-registry.v1",
                "policy": {"max_new_requests_per_scan": 1},
                "mappings": [
                    {
                        "mapping_id": "readable_research_v1",
                        "binding_revision": "fixture-v1",
                        "match": {
                            "editable_component": "readable_state_evidence",
                            "editable_target": None,
                        },
                        "task_manifest": str(task_manifest),
                        "task_project": str(task_project),
                        "praxist_executable": str(executable),
                        "run_root": str(tmp_path / "runs"),
                        "launch": {
                            "config_file": None,
                            "agent_system": "codex_sdk",
                            "runtime": "agent_runtime:codex_sdk",
                            "codex_native": True,
                            "model_provider": "model_provider:openai_compatible",
                            "model": "gpt-5.6-luna",
                            "strategy": "mixed",
                            "cohort": 2,
                            "generations": 2,
                            "startup_timeout_seconds": 30,
                        },
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    demand: dict[str, Any] = {
        "schema_version": "forge-volvence-research-demand.v1",
        "claim_id": "claim:readable-state-action",
        "title": "Close the named-state action transmission gap",
        "objective": "Find a bounded mechanism that makes named state causally readable.",
        "owner": "vz-substrate",
        "capability_axes": ["readable"],
        "need": {
            "current_gap": "The frozen relationship evidence does not transmit named state into action.",
            "required_outcome": "A falsifiable readout-to-action mechanism under a frozen substrate.",
            "success_criteria": ["Named state changes the intended bounded action readout."],
            "falsification_criteria": ["Matched controls explain the same action change."],
            "protected_boundaries": [
                "Do not use evaluation as a learning source.",
                "Do not change production wiring.",
            ],
        },
        "evidence": [_ref(relationship_failure, root=repo)],
        "discovery": {
            "source_roots": [
                "research/industry_four_ables",
                "artifacts/relationship_lab/frozen_failure.json",
            ],
            "max_source_files": 8,
            "max_source_bytes": 16384,
            "max_topics": 2,
        },
        "routing": {"requested_mapping_id": "readable_research_v1"},
        "status": "OPEN",
        "authority": _authority(),
        "created_at": "2026-08-30T00:00:00Z",
    }
    demand["demand_id"] = _demand_identity(demand)
    demand_path = _write_json(repo / "research/demands/readable.json", demand)
    return DiscoveryFixture(
        config=config,
        demand_path=demand_path,
        registry_path=registry_path,
        source_path=source_path,
        source_sha256=_sha(source_path),
    )


def _response(fixture: DiscoveryFixture) -> dict[str, Any]:
    return {
        "topics": [
            {
                "title": "Named residual readout transmission ablation",
                "hypothesis": "A frozen named residual can make the action intervention condition explicit.",
                "mechanism": "Publish a bounded immutable readout before a separately gated action.",
                "demand_relevance": "It directly tests the Demand's readout-to-action gap.",
                "research_question": "Does the named readout change action only under the intended condition?",
                "suggested_method": "Run matched absent, swapped, random-gate and intended-condition arms.",
                "success_signals": ["The intended arm clears the frozen transmission threshold."],
                "falsification_signals": ["Random-gate or swapped-state controls match the effect."],
                "source_refs": [
                    {
                        "locator": fixture.source_path.relative_to(
                            fixture.config.paths.repo_root
                        ).as_posix(),
                        "sha256": fixture.source_sha256,
                        "claim": "The implementation note requires immutable named state before action.",
                    }
                ],
                "caveats": ["This is a mechanism hypothesis, not production evidence."],
            }
        ]
    }


def _backend(fixture: DiscoveryFixture) -> ReplayResearchDiscoveryBackend:
    return ReplayResearchDiscoveryBackend(
        ReplayStructuredBackend(responses=[_response(fixture)], model_name="codex-replay")
    )


def test_demand_discovery_is_unbound_content_addressed_and_idempotent(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = discover_research_topics(
        config=fixture.config,
        demand_path=fixture.demand_path,
        backend=_backend(fixture),
    )

    assert result.reused is False
    assert len(result.proposal_paths) == 1
    run = validate_research_discovery_run(
        config=fixture.config,
        run_path=result.run_path,
    )
    proposal = validate_research_topic_proposal(
        config=fixture.config,
        proposal_path=result.proposal_paths[0],
    )
    assert run["execution"] == {
        "backend": "replay",
        "model": "codex-replay",
        "prompt_revision": "demand-topic-discovery.v1",
        "sandbox": "read_only",
        "approval_mode": "deny_all",
        "turn_limit": 1,
    }
    assert proposal["binding_status"] == "UNBOUND"
    assert proposal["authority"]["research_start_authorized"] is False
    assert not (fixture.config.paths.artifacts_root / "research_control").exists()

    reused = discover_research_topics(
        config=fixture.config,
        demand_path=fixture.demand_path,
        backend=ReplayResearchDiscoveryBackend(
            ReplayStructuredBackend(responses=[], model_name="codex-replay")
        ),
    )
    assert reused.reused is True
    assert reused.run_id == result.run_id
    assert reused.proposal_paths == result.proposal_paths


def test_human_authored_demand_draft_seals_create_only(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    draft = fixture.config.paths.repo_root / "research/demand_drafts/readable.json"
    draft.parent.mkdir(parents=True)
    draft.write_bytes(fixture.demand_path.read_bytes())
    fixture.demand_path.unlink()

    sealed = seal_research_demand(
        config=fixture.config,
        draft_path=draft,
    )
    repeated = seal_research_demand(
        config=fixture.config,
        draft_path=draft,
    )

    assert sealed.reused is False
    assert repeated.reused is True
    assert repeated.demand_id == sealed.demand_id
    assert repeated.demand_path == sealed.demand_path
    assert sealed.demand_path.parent == (
        fixture.config.paths.repo_root / "research/demands"
    )


def test_discovery_rejects_source_outside_frozen_corpus(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    response = _response(fixture)
    response["topics"][0]["source_refs"][0]["sha256"] = "f" * 64
    backend = ReplayResearchDiscoveryBackend(
        ReplayStructuredBackend(responses=[response], model_name="codex-replay")
    )

    with pytest.raises(ResearchDiscoveryError, match="outside the frozen corpus"):
        discover_research_topics(
            config=fixture.config,
            demand_path=fixture.demand_path,
            backend=backend,
        )
    assert not list(
        (fixture.config.paths.artifacts_root / "research_discovery").glob(
            "**/run.json"
        )
    )


def test_source_drift_invalidates_topic_before_human_binding(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = discover_research_topics(
        config=fixture.config,
        demand_path=fixture.demand_path,
        backend=_backend(fixture),
    )
    fixture.source_path.write_text(
        "The source changed after discovery.\n",
        encoding="utf-8",
    )

    with pytest.raises(ResearchDiscoveryError, match="digest mismatch"):
        validate_research_topic_proposal(
            config=fixture.config,
            proposal_path=result.proposal_paths[0],
        )
    with pytest.raises(ResearchDiscoveryError, match="digest mismatch"):
        review_research_topic(
            config=fixture.config,
            demand_path=fixture.demand_path,
            proposal_path=result.proposal_paths[0],
            mapping_id="readable_research_v1",
            registry_path=fixture.registry_path,
            reviewed_by="Named Research Owner",
            reason="Stale source bytes must not bind.",
            decision="APPROVE",
        )


def test_named_binding_submits_to_a0_without_starting_praxist(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    discovery = discover_research_topics(
        config=fixture.config,
        demand_path=fixture.demand_path,
        backend=_backend(fixture),
    )
    reviewed = review_research_topic(
        config=fixture.config,
        demand_path=fixture.demand_path,
        proposal_path=discovery.proposal_paths[0],
        mapping_id="readable_research_v1",
        registry_path=fixture.registry_path,
        reviewed_by="Named Research Owner",
        reason="This exact topic tests the frozen Volvence Demand.",
        decision="APPROVE",
    )
    assert reviewed.decision == "APPROVE"
    binding = validate_research_demand_binding(
        config=fixture.config,
        binding_path=reviewed.binding_path,
    )
    assert binding["authority"]["topic_submission_to_a0_authorized"] is True

    submitted = submit_bound_topic_for_a0(
        config=fixture.config,
        binding_path=reviewed.binding_path,
    )
    request = validate_research_request(
        config=fixture.config,
        request_path=submitted.request_path,
    )
    assert request["trigger"]["kind"] == "typed_signal"
    assert request["trigger"]["submitted_by"] == "forge:demand-discovery-loop.v1"
    assert len(request["trigger"]["evidence"]) == 3
    assert request["authority"]["human_research_approval_required"] is True
    assert request["authority"]["research_start_authorized"] is False
    assert not (submitted.request_path.parent / "approvals").exists()
    assert not Path(request["launch"]["run_dir"]).exists()

    repeated = submit_bound_topic_for_a0(
        config=fixture.config,
        binding_path=reviewed.binding_path,
    )
    assert submitted.reused is False
    assert repeated.reused is True
    assert repeated.request_id == submitted.request_id
    assert repeated.request_path == submitted.request_path


def test_bound_submission_reissues_pre_a0_request_after_source_checkout_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source_root = tmp_path / "praxist-source"
    executable = source_root / "bin/praxist"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    executable.chmod(0o755)
    (source_root / "pyproject.toml").write_text(
        "[project]\nname = 'praxist-fixture'\nversion = '0.0.0'\n",
        encoding="utf-8",
    )
    package = source_root / "praxist"
    package.mkdir()
    runtime = package / "runtime.py"
    runtime.write_text("REVISION = 1\n", encoding="utf-8")
    registry = yaml.safe_load(fixture.registry_path.read_text(encoding="utf-8"))
    registry["mappings"][0]["praxist_executable"] = str(executable)
    fixture.registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False),
        encoding="utf-8",
    )

    discovery = discover_research_topics(
        config=fixture.config,
        demand_path=fixture.demand_path,
        backend=_backend(fixture),
    )
    reviewed = review_research_topic(
        config=fixture.config,
        demand_path=fixture.demand_path,
        proposal_path=discovery.proposal_paths[0],
        mapping_id="readable_research_v1",
        registry_path=fixture.registry_path,
        reviewed_by="Named Research Owner",
        reason="This exact topic tests the frozen Volvence Demand.",
        decision="APPROVE",
    )
    old = submit_bound_topic_for_a0(
        config=fixture.config,
        binding_path=reviewed.binding_path,
    )

    runtime.write_text("REVISION = 2\n", encoding="utf-8")
    replacement = submit_bound_topic_for_a0(
        config=fixture.config,
        binding_path=reviewed.binding_path,
    )
    repeated = submit_bound_topic_for_a0(
        config=fixture.config,
        binding_path=reviewed.binding_path,
    )

    old_status = inspect_research_request(
        config=fixture.config,
        request_path=old.request_path,
    )
    assert old.request_id != replacement.request_id
    assert old_status.state == "SUPERSEDED"
    assert old_status.replacement_request_id == replacement.request_id
    assert replacement.reused is False
    assert repeated.reused is True
    assert repeated.request_id == replacement.request_id
    assert not (replacement.request_path.parent / "approvals").exists()


def test_bounded_loop_waits_for_both_human_gates_before_reconcile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    first = run_demand_research_loop_once(
        config=fixture.config,
        backend=_backend(fixture),
    )
    assert first.new_discovery_count == 1
    assert first.new_request_count == 0
    assert first.reconciliations == ()
    assert not (fixture.config.paths.artifacts_root / "research_control").exists()

    proposal_path = first.discoveries[0].run_path.parent / "topics"
    proposals = tuple(sorted(proposal_path.glob("*.json")))
    assert len(proposals) == 1
    reviewed = review_research_topic(
        config=fixture.config,
        demand_path=fixture.demand_path,
        proposal_path=proposals[0],
        mapping_id="readable_research_v1",
        registry_path=fixture.registry_path,
        reviewed_by="Named Research Owner",
        reason="Bind the exact Demand topic for A0 consideration.",
        decision="APPROVE",
    )
    assert reviewed.decision == "APPROVE"

    second = run_demand_research_loop_once(
        config=fixture.config,
        backend=ReplayResearchDiscoveryBackend(
            ReplayStructuredBackend(responses=[], model_name="codex-replay")
        ),
    )
    assert second.new_discovery_count == 0
    assert second.discoveries[0].reused is True
    assert second.new_request_count == 1
    assert second.awaiting_a0_count == 1
    assert second.reconciliations == ()
    request_path = second.submissions[0].request_path
    request = validate_research_request(
        config=fixture.config,
        request_path=request_path,
    )
    assert request["authority"]["research_start_authorized"] is False
    assert not Path(request["launch"]["run_dir"]).exists()

    third = run_demand_research_loop_once(
        config=fixture.config,
        backend=ReplayResearchDiscoveryBackend(
            ReplayStructuredBackend(responses=[], model_name="codex-replay")
        ),
    )
    assert third.new_discovery_count == 0
    assert third.new_request_count == 0
    assert third.submissions[0].reused is True
    assert third.awaiting_a0_count == 1

    approval = review_research_request(
        config=fixture.config,
        request_path=request_path,
        reviewed_by="Named A0 Approver",
        reason="Approve only the exact bounded Praxist research budget.",
        decision="APPROVE",
    )
    runner_sentinel = object()
    calls: list[Path] = []

    def fake_reconcile(*, config, request_path, runner):
        assert config == fixture.config
        assert runner is runner_sentinel
        calls.append(request_path)
        return (
            ResearchControlStatus(
                request_id=second.submissions[0].request_id,
                task_id="readable_research",
                state="RUNNING",
                request_path=request_path,
                approval_path=approval.approval_path,
                latest_event_path=None,
                run_id=Path(request["launch"]["run_dir"]).name,
                run_dir=request["launch"]["run_dir"],
                monitor_command=None,
            ),
        )

    monkeypatch.setattr(
        research_loop_module,
        "reconcile_research_control",
        fake_reconcile,
    )
    approved = run_demand_research_loop_once(
        config=fixture.config,
        backend=ReplayResearchDiscoveryBackend(
            ReplayStructuredBackend(responses=[], model_name="codex-replay")
        ),
        max_new_discoveries=0,
        max_new_requests=0,
        runner=runner_sentinel,
    )
    assert calls == [request_path]
    assert approved.reconciliations[0].state_before == "APPROVED"
    assert approved.reconciliations[0].state_after == "RUNNING"
    assert approved.to_jsonable()["authority"] == {
        "human_topic_binding_required": True,
        "human_a0_required": True,
        "automatic_a0_authorized": False,
        "automatic_candidate_import_authorized": False,
        "production_promotion_authorized": False,
        "runtime_wiring_changed": False,
        "evaluation_is_learning_source": False,
    }


def test_bounded_loop_rejects_discovery_owned_request_with_extra_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    discovered = run_demand_research_loop_once(
        config=fixture.config,
        backend=_backend(fixture),
    )
    proposal_path = discovered.discoveries[0].run_path.parent / "topics"
    proposal = next(proposal_path.glob("*.json"))
    review_research_topic(
        config=fixture.config,
        demand_path=fixture.demand_path,
        proposal_path=proposal,
        mapping_id="readable_research_v1",
        registry_path=fixture.registry_path,
        reviewed_by="Named Research Owner",
        reason="Bind the exact Demand topic for A0 consideration.",
        decision="APPROVE",
    )
    submitted = run_demand_research_loop_once(
        config=fixture.config,
        backend=ReplayResearchDiscoveryBackend(
            ReplayStructuredBackend(responses=[], model_name="codex-replay")
        ),
    )
    request_path = submitted.submissions[0].request_path
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["trigger"]["evidence"].append(
        _ref(fixture.source_path, root=fixture.config.paths.repo_root)
    )
    request["request_id"] = _request_identity(request)
    replacement = (
        request_path.parent.parent
        / str(request["request_id"]).partition(":")[2]
        / "request.json"
    )
    _write_json(replacement, request)
    shutil.rmtree(request_path.parent)
    review_research_request(
        config=fixture.config,
        request_path=replacement,
        reviewed_by="Named A0 Approver",
        reason="Approve the exact altered Request to test loop ownership.",
        decision="APPROVE",
    )

    with pytest.raises(ResearchLoopError, match="exact approved"):
        run_demand_research_loop_once(
            config=fixture.config,
            backend=ReplayResearchDiscoveryBackend(
                ReplayStructuredBackend(responses=[], model_name="codex-replay")
            ),
            max_new_discoveries=0,
            max_new_requests=0,
        )


def test_bounded_loop_excludes_blocked_demand_request_from_reconcile(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    discovered = run_demand_research_loop_once(
        config=fixture.config,
        backend=_backend(fixture),
    )
    proposal = next((discovered.discoveries[0].run_path.parent / "topics").glob("*.json"))
    review_research_topic(
        config=fixture.config,
        demand_path=fixture.demand_path,
        proposal_path=proposal,
        mapping_id="readable_research_v1",
        registry_path=fixture.registry_path,
        reviewed_by="Named Research Owner",
        reason="Bind the exact Demand topic for dependency filtering.",
        decision="APPROVE",
    )
    submitted = run_demand_research_loop_once(
        config=fixture.config,
        backend=ReplayResearchDiscoveryBackend(
            ReplayStructuredBackend(responses=[], model_name="codex-replay")
        ),
    )
    review_research_request(
        config=fixture.config,
        request_path=submitted.submissions[0].request_path,
        reviewed_by="Named A0 Approver",
        reason="Approve the exact Request before testing dependency exclusion.",
        decision="APPROVE",
    )
    demand_id = json.loads(fixture.demand_path.read_text(encoding="utf-8"))[
        "demand_id"
    ]

    blocked = run_demand_research_loop_once(
        config=fixture.config,
        backend=ReplayResearchDiscoveryBackend(
            ReplayStructuredBackend(responses=[], model_name="codex-replay")
        ),
        max_new_discoveries=0,
        max_new_requests=0,
        blocked_demand_ids=frozenset({demand_id}),
    )

    assert blocked.demand_count == 0
    assert blocked.binding_count == 0
    assert blocked.reconciliations == ()


def test_rejected_or_misaligned_topic_cannot_enter_a0(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    discovery = discover_research_topics(
        config=fixture.config,
        demand_path=fixture.demand_path,
        backend=_backend(fixture),
    )
    rejected = review_research_topic(
        config=fixture.config,
        demand_path=fixture.demand_path,
        proposal_path=discovery.proposal_paths[0],
        mapping_id="readable_research_v1",
        registry_path=fixture.registry_path,
        reviewed_by="Named Research Owner",
        reason="The proposed mechanism is not discriminating enough.",
        decision="REJECT",
    )
    with pytest.raises(ResearchDiscoveryError, match="only an APPROVE"):
        submit_bound_topic_for_a0(
            config=fixture.config,
            binding_path=rejected.binding_path,
        )

    demand = json.loads(fixture.demand_path.read_text(encoding="utf-8"))
    demand["claim_id"] = "claim:another-owner"
    demand["created_at"] = "2026-08-30T00:01:00Z"
    demand["demand_id"] = _demand_identity(demand)
    misaligned_path = _write_json(
        fixture.config.paths.repo_root / "research/demands/misaligned.json",
        demand,
    )
    second = discover_research_topics(
        config=fixture.config,
        demand_path=misaligned_path,
        backend=_backend(fixture),
    )
    with pytest.raises(ResearchDiscoveryError, match="claim_id"):
        review_research_topic(
            config=fixture.config,
            demand_path=misaligned_path,
            proposal_path=second.proposal_paths[0],
            mapping_id="readable_research_v1",
            registry_path=fixture.registry_path,
            reviewed_by="Named Research Owner",
            reason="Attempt a wrong-task binding.",
            decision="APPROVE",
        )


@pytest.mark.live_network
def test_codex_native_discovery_uses_saved_login_read_only(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = discover_research_topics(
        config=fixture.config,
        demand_path=fixture.demand_path,
        backend=CodexNativeResearchDiscoveryBackend(model_name="gpt-5.6-luna"),
    )

    run = validate_research_discovery_run(
        config=fixture.config,
        run_path=result.run_path,
    )
    assert run["execution"]["backend"] == "codex_sdk"
    assert run["execution"]["model"] == "gpt-5.6-luna"
    assert run["execution"]["sandbox"] == "read_only"
    assert result.proposal_paths
    for proposal_path in result.proposal_paths:
        validate_research_topic_proposal(
            config=fixture.config,
            proposal_path=proposal_path,
        )
