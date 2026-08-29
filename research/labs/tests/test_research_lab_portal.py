from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from volvence_labs.portal import (
    ArtifactRef,
    LifecycleStage,
    OwnerCommandResult,
    PortalCommandError,
    ResearchLabCollector,
    ResearchLabCommandService,
    ResearchLabItem,
    create_server,
)


FIXED_NOW = datetime(2026, 8, 29, 14, 30, tzinfo=timezone.utc)
TASK_ID = "example_research_task"
EXTERNAL_TASK_ID = "external_0123456789abcdef"


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_artifact_id(
    prefix: str,
    payload: dict[str, object],
    identity_field: str,
) -> str:
    body = {
        key: value
        for key, value in payload.items()
        if key not in {identity_field, "created_at"}
    }
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def _artifact_content_ref(repo: Path, path: Path) -> dict[str, str]:
    return {
        "locator": path.relative_to(repo).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _make_discovery(repo: Path) -> tuple[Path, Path, Path]:
    registry_path = repo / "forge" / "research_task_registry.yaml"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        "schema_version: forge-research-task-registry.v1\nmappings: []\n",
        encoding="utf-8",
    )
    source_path = repo / "research" / "industry_four_ables" / "readable.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "Publish immutable named state before a bounded action.\n",
        encoding="utf-8",
    )
    source_ref = _artifact_content_ref(repo, source_path)
    demand: dict[str, object] = {
        "schema_version": "forge-volvence-research-demand.v1",
        "claim_id": "claim:example",
        "title": "Close the named-state transmission gap",
        "objective": "Find a falsifiable readout-to-action mechanism.",
        "owner": "vz-memory",
        "capability_axes": ["readable"],
        "need": {},
        "evidence": [],
        "discovery": {
            "source_roots": ["research/industry_four_ables"],
            "max_source_files": 8,
            "max_source_bytes": 16384,
            "max_topics": 2,
        },
        "routing": {"requested_mapping_id": "example_research_v1"},
        "status": "OPEN",
        "authority": {},
        "created_at": "2026-08-30T00:00:00Z",
    }
    demand["demand_id"] = _canonical_artifact_id(
        "research-demand",
        demand,
        "demand_id",
    )
    demand_path = repo / "research" / "demands" / "example.json"
    _write_json(demand_path, demand)
    demand_ref = {
        "artifact_id": demand["demand_id"],
        "artifact": _artifact_content_ref(repo, demand_path),
    }
    run_key = "7" * 64
    run_id = f"research-discovery-run:{run_key}"
    run_root = (
        repo
        / "artifacts"
        / "research_discovery"
        / str(demand["demand_id"]).partition(":")[2]
        / "runs"
        / run_key
    )
    proposal: dict[str, object] = {
        "schema_version": "forge-research-topic-proposal.v1",
        "demand": demand_ref,
        "discovery_run_id": run_id,
        "corpus_sha256": "8" * 64,
        "capability_axes": ["readable"],
        "topic": {
            "title": "Named residual transmission ablation",
            "hypothesis": "A named residual changes action only under its intended condition.",
            "mechanism": "Publish a frozen readout before a separately gated action.",
            "demand_relevance": "Directly tests the Demand gap.",
            "research_question": "Does the intended arm beat swapped and random controls?",
            "suggested_method": "Run matched absent, swapped, random, and intended arms.",
            "success_signals": ["The intended arm clears the frozen threshold."],
            "falsification_signals": ["Random controls match the intended effect."],
            "caveats": ["This is a hypothesis, not production evidence."],
        },
        "source_refs": [
            {
                **source_ref,
                "claim": "The source requires immutable named state before action.",
            }
        ],
        "binding_status": "UNBOUND",
        "authority": {},
        "created_at": "2026-08-30T00:01:00Z",
    }
    proposal["proposal_id"] = _canonical_artifact_id(
        "research-topic-proposal",
        proposal,
        "proposal_id",
    )
    proposal_path = (
        run_root
        / "topics"
        / f"{str(proposal['proposal_id']).partition(':')[2]}.json"
    )
    _write_json(proposal_path, proposal)
    run: dict[str, object] = {
        "schema_version": "forge-research-discovery-run.v1",
        "run_id": run_id,
        "run_key": run_key,
        "demand": demand_ref,
        "corpus": {"tree_sha256": "8" * 64},
        "execution": {
            "backend": "codex_sdk",
            "model": "gpt-5.6-luna",
        },
        "proposals": [
            {
                "artifact_id": proposal["proposal_id"],
                "artifact": _artifact_content_ref(repo, proposal_path),
            }
        ],
        "authority": {},
        "created_at": "2026-08-30T00:02:00Z",
    }
    _write_json(run_root / "run.json", run)
    return demand_path, proposal_path, registry_path


def _make_task(repo: Path) -> Path:
    path = repo / "research" / "tasks" / TASK_ID / "task.json"
    _write_json(
        path,
        {
            "schema_version": "forge-research-task.v1",
            "task_id": TASK_ID,
            "claim_id": "claim:example",
            "owner": "vz-memory",
            "objective": "Improve one bounded research mechanism without changing production authority.",
            "capability_axes": ["appendable", "readable"],
            "release": {
                "mode": "runtime_wiring",
                "target": "example_policy",
                "initial_wiring": "disabled",
            },
        },
    )
    (repo / "docs" / "specs").mkdir(parents=True, exist_ok=True)
    (repo / "docs" / "specs" / "00_INDEX.md").write_text("# index\n", encoding="utf-8")
    return path


def _make_request(repo: Path, *, approved: bool, correct_sha: bool = True) -> Path:
    request_id = "research-request:" + "a" * 64
    root = repo / "artifacts" / "research_control" / TASK_ID / ("a" * 64)
    request_path = root / "request.json"
    request_sha = _write_json(
        request_path,
        {
            "schema_version": "forge-research-request.v1",
            "request_id": request_id,
            "task_id": TASK_ID,
            "claim_id": "claim:example",
            "owner": "vz-memory",
            "created_at": "2026-08-29T14:00:00Z",
            "bindings": {
                "task_project": {
                    "root": str(repo / "research" / "praxist_tasks" / TASK_ID),
                }
            },
        },
    )
    if approved:
        _write_json(
            root / "approvals" / "approval.json",
            {
                "schema_version": "forge-research-approval.v1",
                "approval_id": "research-approval:" + "b" * 64,
                "request_id": request_id,
                "request_sha256": request_sha if correct_sha else "0" * 64,
                "decision": "APPROVE",
                "authority": {"research_start_authorized": True},
                "created_at": "2026-08-29T14:01:00Z",
            },
        )
    return request_path


def _make_external_request(
    repo: Path,
    *,
    approved: bool = False,
    completed: bool = False,
    handed_off: bool = False,
) -> tuple[Path, Path]:
    external_root = repo / "external-foundry"
    intent_path = external_root / "artifacts/research_lab/intent.json"
    intent_sha = _write_json(
        intent_path,
        {
            "schema_version": "foundry-research-lab-intent.v1",
            "intent_id": "rli_0123456789abcdef",
        },
    )
    descriptor_id = "external-research-descriptor:" + "d" * 64
    descriptor_path = external_root / "artifacts/research_lab/descriptor.json"
    descriptor_sha = _write_json(
        descriptor_path,
        {
            "schema_version": "forge-external-research-descriptor.v1",
            "descriptor_id": descriptor_id,
            "domain": {
                "domain_id": "foundry",
                "task_id": "opp_research_lab_001",
                "intent_id": "rli_0123456789abcdef",
            },
        },
    )
    task_project = repo / "research/praxist_tasks/foundry_optimizer"
    task_project.mkdir(parents=True, exist_ok=True)
    (task_project / "task.yaml").write_text("task_id: foundry_optimizer\n", encoding="utf-8")
    run_dir = repo / "artifacts/praxist_runs/run_foundry_001"
    if completed:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "result.json", {"best": "candidate-a"})
    request_id = "research-request:" + "e" * 64
    root = repo / "artifacts/research_control" / EXTERNAL_TASK_ID / ("e" * 64)
    request_path = root / "request.json"
    request_sha = _write_json(
        request_path,
        {
            "schema_version": "forge-external-research-request.v1",
            "request_id": request_id,
            "task_id": EXTERNAL_TASK_ID,
            "claim_id": "foundry:rli_0123456789abcdef",
            "owner": "external-domain:foundry",
            "objective": "Explore evaluator-backed variants without changing Foundry governance surfaces.",
            "external_domain": {
                "descriptor_id": descriptor_id,
                "adapter_id": "foundry-research-lab-intent.v1",
                "intent_schema_version": "foundry-research-lab-intent.v1",
                "domain_id": "foundry",
                "task_id": "opp_research_lab_001",
                "intent_id": "rli_0123456789abcdef",
                "ownership": {
                    "research_intent": "foundry",
                    "budget": "foundry",
                    "evidence_classification": "foundry",
                    "result_adoption": "foundry",
                    "human_application": "foundry",
                },
                "result_policy": {
                    "result_locator": "result.json",
                    "evidence_class": "simulation",
                    "adoption_mode": "proposal_only",
                },
            },
            "bindings": {
                "external_descriptor": {
                    "locator": str(descriptor_path),
                    "sha256": descriptor_sha,
                },
                "external_intent": {"locator": str(intent_path), "sha256": intent_sha},
                "external_budget": {"locator": str(intent_path), "sha256": intent_sha},
                "external_evidence": [],
                "task_project": {"root": str(task_project)},
            },
            "launch": {
                "run_dir": str(run_dir),
                "run_id": run_dir.name,
                "profile": {
                    "runtime": "agent_runtime:codex_sdk",
                    "model_provider": "model_provider:openai_compatible",
                    "model": "gpt-5.6-luna",
                },
            },
            "created_at": "2026-08-30T00:00:00Z",
        },
    )
    approval_id = "research-approval:" + "f" * 64
    approval_sha = None
    if approved:
        approval_path = root / "approvals" / ("f" * 64 + ".json")
        approval_sha = _write_json(
            approval_path,
            {
                "schema_version": "forge-research-approval.v1",
                "approval_id": approval_id,
                "request_id": request_id,
                "request_sha256": request_sha,
                "decision": "APPROVE",
                "authority": {"research_start_authorized": True},
                "created_at": "2026-08-30T00:01:00Z",
            },
        )
    if completed:
        _write_json(
            root / "events/000001-event.json",
            {
                "schema_version": "forge-research-control-event.v1",
                "event_id": "research-control-event:" + "1" * 64,
                "sequence": 1,
                "request_id": request_id,
                "state": "RUN_COMPLETED",
                "run": {
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir),
                    "praxist_state": "completed",
                    "generation": 3,
                    "findings_total": 9,
                    "updated_at": "2026-08-30T00:05:00Z",
                },
                "created_at": "2026-08-30T00:05:00Z",
            },
        )
    if handed_off:
        if approval_sha is None:
            raise AssertionError("handoff fixture requires approval")
        _write_json(
            root / "handoffs/handoff.json",
            {
                "schema_version": "forge-external-research-handoff.v1",
                "handoff_id": "external-research-handoff:" + "2" * 64,
                "request": {"request_id": request_id, "sha256": request_sha},
                "result": {
                    "evidence_class": "simulation",
                    "adoption_mode": "proposal_only",
                },
                "authority": {
                    "volvence_promotion_eligible": False,
                    "modification_gate_applicable": False,
                    "runtime_wiring_applicable": False,
                },
                "created_at": "2026-08-30T00:06:00Z",
            },
        )
    return request_path, descriptor_path


def _write_review(repo: Path, request_path: Path, *, decision: str) -> Path:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    suffix = "b" if decision == "APPROVE" else "c"
    path = request_path.parent / "approvals" / f"{suffix * 64}.json"
    _write_json(
        path,
        {
            "schema_version": "forge-research-approval.v1",
            "approval_id": "research-approval:" + suffix * 64,
            "request_id": request["request_id"],
            "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "decision": decision,
            "review": {"reviewed_by": "Test Reviewer", "reason": "Fixture decision"},
            "authority": {"research_start_authorized": decision == "APPROVE"},
            "created_at": "2026-08-29T14:01:00Z",
        },
    )
    return path


class FakeForgeRunner:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, arguments: object) -> OwnerCommandResult:
        if not isinstance(arguments, tuple):
            raise AssertionError("portal must pass one frozen argv tuple")
        self.calls.append(arguments)
        if "research-approve" in arguments:
            command_index = arguments.index("research-approve")
            request_path = Path(arguments[command_index + 1])
            decision = "REJECT" if "--reject" in arguments else "APPROVE"
            _write_review(self.repo, request_path, decision=decision)
            return OwnerCommandResult(0, f"{decision}: fake approval\n", "")
        if "research-reconcile" in arguments:
            return OwnerCommandResult(
                0,
                json.dumps([{"state": "WAITING_FOR_CAPACITY", "run_id": None}]),
                "",
            )
        raise AssertionError(f"unexpected Forge command: {arguments}")


class FakeDiscoveryRunner:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, arguments: object) -> OwnerCommandResult:
        if not isinstance(arguments, tuple):
            raise AssertionError("portal must pass one frozen argv tuple")
        self.calls.append(arguments)
        command_index = arguments.index("research-bind-topic")
        demand_path = Path(arguments[command_index + 1])
        proposal_path = Path(arguments[command_index + 2])
        registry_path = Path(arguments[arguments.index("--registry") + 1])
        mapping_id = arguments[arguments.index("--mapping-id") + 1]
        actor = arguments[arguments.index("--reviewed-by") + 1]
        reason = arguments[arguments.index("--reason") + 1]
        decision = "REJECT" if "--reject" in arguments else "APPROVE"
        demand = json.loads(demand_path.read_text(encoding="utf-8"))
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        binding: dict[str, object] = {
            "schema_version": "forge-research-demand-binding.v1",
            "demand": {
                "artifact_id": demand["demand_id"],
                "artifact": _artifact_content_ref(self.repo, demand_path),
            },
            "proposal": {
                "artifact_id": proposal["proposal_id"],
                "artifact": _artifact_content_ref(self.repo, proposal_path),
            },
            "registry": _artifact_content_ref(self.repo, registry_path),
            "mapping": {"mapping_id": mapping_id},
            "decision": decision,
            "review": {"reviewed_by": actor, "reason": reason},
            "authority": {},
            "created_at": "2026-08-30T00:03:00Z",
        }
        binding["binding_id"] = _canonical_artifact_id(
            "research-demand-binding",
            binding,
            "binding_id",
        )
        proposal_digest = str(proposal["proposal_id"]).partition(":")[2]
        binding_path = (
            proposal_path.parent.parent
            / "bindings"
            / proposal_digest
            / f"{str(binding['binding_id']).partition(':')[2]}.json"
        )
        binding_sha = _write_json(binding_path, binding)
        return OwnerCommandResult(
            0,
            json.dumps(
                {
                    "schema_version": "forge-research-demand-binding-result.v1",
                    "binding_id": binding["binding_id"],
                    "binding": str(binding_path),
                    "binding_sha256": binding_sha,
                    "decision": decision,
                }
            ),
            "",
        )


class FakePromotionRunner:
    def __init__(self, repo: Path, *, authorize_outcome: str = "AUTHORIZED") -> None:
        self.repo = repo
        self.calls: list[tuple[str, ...]] = []
        self.receipt_counter = 0
        self.authorize_outcome = authorize_outcome

    def __call__(self, arguments: object) -> OwnerCommandResult:
        if not isinstance(arguments, tuple):
            raise AssertionError("portal must pass one frozen argv tuple")
        self.calls.append(arguments)
        if "research-import-praxist" in arguments:
            command_index = arguments.index("research-import-praxist")
            handoff = json.loads(Path(arguments[command_index + 2]).read_text(encoding="utf-8"))
            candidate_id = "research-candidate:" + "e" * 64
            _write_json(
                self.repo / "artifacts" / "research_promotion" / TASK_ID / ("e" * 64) / "candidate.json",
                {
                    "schema_version": "forge-research-candidate.v1",
                    "candidate_id": candidate_id,
                    "task_id": TASK_ID,
                    "source": {"run_id": handoff["run_id"]},
                    "created_at": "2026-08-29T15:00:00Z",
                },
            )
            return OwnerCommandResult(0, f"SEALED: {candidate_id}\n", "")
        if "research-authorize" in arguments:
            command_index = arguments.index("research-authorize")
            task_path = Path(arguments[command_index + 1])
            candidate_path = Path(arguments[command_index + 2])
            validation_path = Path(arguments[command_index + 3])
            gate_path = Path(arguments[command_index + 4])
            to_wiring = arguments[arguments.index("--to-wiring") + 1]
            previous_path = (
                Path(arguments[arguments.index("--previous-receipt") + 1])
                if "--previous-receipt" in arguments
                else None
            )
            receipt = self._write_receipt(
                candidate_path=candidate_path,
                action="authorize",
                from_wiring="disabled" if to_wiring == "shadow" else "shadow",
                to_wiring=to_wiring,
                task_sha256=hashlib.sha256(task_path.read_bytes()).hexdigest(),
                validation_sha256=hashlib.sha256(validation_path.read_bytes()).hexdigest(),
                gate_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
                previous_path=previous_path,
                outcome=self.authorize_outcome,
            )
            returncode = 0 if self.authorize_outcome == "AUTHORIZED" else 2
            return OwnerCommandResult(returncode, f"{self.authorize_outcome}: {receipt['receipt_id']}\n", "")
        if "research-rollback" in arguments:
            command_index = arguments.index("research-rollback")
            previous_path = Path(arguments[command_index + 1])
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            to_wiring = arguments[arguments.index("--to-wiring") + 1]
            candidate_path = _candidate_path(self.repo)
            receipt = self._write_receipt(
                candidate_path=candidate_path,
                action="rollback",
                from_wiring=previous["transition"]["resulting_wiring"],
                to_wiring=to_wiring,
                task_sha256=previous["bindings"]["task_manifest_sha256"],
                validation_sha256=previous["bindings"]["validation_sha256"],
                gate_sha256=previous["bindings"]["gate_sha256"],
                previous_path=previous_path,
                outcome="AUTHORIZED",
            )
            return OwnerCommandResult(0, f"AUTHORIZED: {receipt['receipt_id']}\n", "")
        raise AssertionError(f"unexpected Forge command: {arguments}")

    def _write_receipt(
        self,
        *,
        candidate_path: Path,
        action: str,
        from_wiring: str,
        to_wiring: str,
        task_sha256: str,
        validation_sha256: str,
        gate_sha256: str,
        previous_path: Path | None,
        outcome: str,
    ) -> dict[str, object]:
        self.receipt_counter += 1
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        receipt_id = "research-receipt:" + f"{self.receipt_counter:064x}"
        payload: dict[str, object] = {
            "schema_version": "forge-research-promotion-receipt.v1",
            "receipt_id": receipt_id,
            "task_id": TASK_ID,
            "candidate_id": candidate["candidate_id"],
            "outcome": outcome,
            "action": action,
            "transition": {
                "from_wiring": from_wiring,
                "requested_wiring": to_wiring,
                "resulting_wiring": to_wiring if outcome == "AUTHORIZED" else from_wiring,
            },
            "bindings": {
                "task_manifest_sha256": task_sha256,
                "candidate_sha256": candidate_sha,
                "validation_sha256": validation_sha256,
                "gate_sha256": gate_sha256,
                "previous_receipt_sha256": (
                    hashlib.sha256(previous_path.read_bytes()).hexdigest() if previous_path is not None else None
                ),
            },
            "blocking_reasons": [] if outcome == "AUTHORIZED" else ["fixture gate block"],
            "authority": {"target_adapter_apply_required": True},
            "created_at": f"2026-08-29T15:{self.receipt_counter:02}:00Z",
        }
        path = candidate_path.parent / "receipts" / f"{self.receipt_counter:064x}.json"
        _write_json(path, payload)
        return payload


def _candidate_path(repo: Path) -> Path:
    return repo / "artifacts" / "research_promotion" / TASK_ID / ("e" * 64) / "candidate.json"


def _make_candidate(repo: Path) -> Path:
    path = _candidate_path(repo)
    _write_json(
        path,
        {
            "schema_version": "forge-research-candidate.v1",
            "candidate_id": "research-candidate:" + "e" * 64,
            "task_id": TASK_ID,
            "created_at": "2026-08-29T15:00:00Z",
        },
    )
    return path


def _make_formal_gate(repo: Path, candidate_path: Path, *, name: str, minute: int) -> tuple[Path, Path]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    validation_path = repo / "artifacts" / "research_validation" / f"{name}.json"
    validation_sha = _write_json(
        validation_path,
        {
            "schema_version": "forge-research-validation.v1",
            "task_id": TASK_ID,
            "candidate_id": candidate["candidate_id"],
            "candidate_sha256": candidate_sha,
            "status": "PASS",
            "created_at": f"2026-08-29T15:{minute:02}:00Z",
        },
    )
    gate_path = repo / "artifacts" / "research_gate" / f"{name}.json"
    _write_json(
        gate_path,
        {
            "schema_version": "forge-research-gate.v1",
            "task_id": TASK_ID,
            "candidate_id": candidate["candidate_id"],
            "candidate_sha256": candidate_sha,
            "validation_sha256": validation_sha,
            "decision": "ALLOW",
            "created_at": f"2026-08-29T15:{minute + 1:02}:00Z",
        },
    )
    return validation_path, gate_path


def _binding(item: ResearchLabItem, kind: str) -> ArtifactRef:
    return next(ref for ref in item.bindings if ref.kind == kind)


def _running_status(repo: Path, *, run_id: str = "run_example", pid: int = 1234) -> dict[str, object]:
    task_path = repo / "research" / "praxist_tasks" / TASK_ID
    run_dir = task_path / "experiments" / run_id
    _write_json(
        run_dir / "startup_config.json",
        {
            "schema_version": "praxist.startup.v1",
            "canonical_args": {
                "runtime": "agent_runtime:codex_sdk",
                "model_provider": "model_provider:openai_compatible",
                "model": "gpt-5.6-luna",
            },
        },
    )
    return {
        "run_id": run_id,
        "state": "running",
        "source": "registry",
        "pid": pid,
        "task_path": str(task_path),
        "run_dir": str(run_dir),
        "generation": 0,
        "findings_total": 0,
        "peer_health_summary": {"green": 0, "yellow": 4, "red": 0},
        "peers": [{"peer_id": f"peer_{index}"} for index in range(4)],
        "model": "gpt-5.6-luna",
        "model_provider_ref": "model_provider:openai_compatible",
        "started_at": "2026-08-29T14:02:00Z",
        "updated_at": "2026-08-29T14:03:00Z",
    }


def _collector(repo: Path, statuses: list[dict[str, object]] | None = None) -> ResearchLabCollector:
    loader = (lambda: statuses) if statuses is not None else None
    return ResearchLabCollector(
        repo,
        status_loader=loader,
        clock=lambda: FIXED_NOW,
        revision_loader=lambda _root: "f" * 40,
    )


def test_running_snapshot_binds_exact_approval_and_live_praxist_status(tmp_path: Path) -> None:
    _make_task(tmp_path)
    _make_request(tmp_path, approved=True)
    snapshot = _collector(tmp_path, [_running_status(tmp_path)]).collect()

    assert snapshot.schema_version == "volvence-research-lab-snapshot.v2"
    assert snapshot.summary.registered_tasks == 1
    assert snapshot.summary.active_runs == 1
    assert snapshot.summary.awaiting_human == 0
    item = snapshot.items[0]
    assert item.lifecycle.stage is LifecycleStage.RESEARCH_RUNNING
    assert item.authority.a0_research_start_authorized is True
    assert item.authority.runtime_wiring == "disabled"
    assert item.run is not None
    assert item.run.pid == 1234
    assert item.run.peers_total == 4
    assert item.run.runtime == "agent_runtime:codex_sdk"
    assert item.run.model == "gpt-5.6-luna"
    assert item.available_actions == ("view_run",)
    assert {ref.kind for ref in item.bindings} >= {"task", "research request", "research approval"}
    with pytest.raises(FrozenInstanceError):
        snapshot.repo_revision = "mutated"  # type: ignore[misc]


def test_external_simulation_track_stops_at_immutable_foundry_handoff(tmp_path: Path) -> None:
    _make_task(tmp_path)
    _make_external_request(tmp_path, approved=True, completed=True)
    before = _collector(tmp_path, []).collect().get_task(EXTERNAL_TASK_ID)
    assert before is not None
    assert before.research_mode == "external_simulation"
    assert before.lifecycle.stage is LifecycleStage.RESEARCH_COMPLETE
    assert before.available_actions == ("record_external_handoff",)
    assert before.authority.modification_gate_decision == "not_applicable"
    assert before.authority.runtime_wiring == "not_applicable"
    assert before.evidence.development == "simulation_completed"
    assert not any(ref.kind == "candidate" for ref in before.bindings)

    _make_external_request(tmp_path, approved=True, completed=True, handed_off=True)
    after = _collector(tmp_path, []).collect().get_task(EXTERNAL_TASK_ID)
    assert after is not None
    assert after.lifecycle.stage is LifecycleStage.RESEARCH_COMPLETE
    assert after.available_actions == ()
    assert after.evidence.development == "simulation_handoff_sealed"
    handoff = next(ref for ref in after.bindings if ref.kind == "external handoff")
    assert handoff.artifact_id == "external-research-handoff:" + "2" * 64


def test_request_without_approval_is_awaiting_a0(tmp_path: Path) -> None:
    _make_task(tmp_path)
    _make_request(tmp_path, approved=False)
    snapshot = _collector(tmp_path, []).collect()

    item = snapshot.items[0]
    assert item.lifecycle.stage is LifecycleStage.AWAITING_A0
    assert item.authority.a0_research_start_authorized is False
    assert item.available_actions == ("review_a0",)
    assert snapshot.summary.awaiting_human == 1


def test_discovery_snapshot_keeps_topic_unbound_and_source_exact(tmp_path: Path) -> None:
    _make_task(tmp_path)
    demand_path, proposal_path, registry_path = _make_discovery(tmp_path)

    snapshot = _collector(tmp_path, []).collect()

    assert snapshot.discovery.demand_count == 1
    assert snapshot.discovery.open_demand_count == 1
    assert snapshot.discovery.proposal_count == 1
    assert snapshot.discovery.awaiting_binding_count == 1
    assert snapshot.discovery.registry is not None
    assert snapshot.discovery.registry.sha256 == hashlib.sha256(
        registry_path.read_bytes()
    ).hexdigest()
    demand = snapshot.discovery.demands[0]
    proposal = demand.proposals[0]
    assert demand.artifact.locator == demand_path.relative_to(tmp_path).as_posix()
    assert demand.run_backend == "codex_sdk"
    assert demand.run_model == "gpt-5.6-luna"
    assert proposal.artifact.locator == proposal_path.relative_to(tmp_path).as_posix()
    assert proposal.effective_state == "UNBOUND"
    assert proposal.available_actions == ("bind_topic",)
    assert proposal.source_refs[0].claim.startswith("The source requires")
    assert proposal.binding is None
    assert proposal.request is None
    assert not (tmp_path / "artifacts" / "research_control").exists()


def test_topic_binding_uses_exact_revision_and_still_requires_a0(tmp_path: Path) -> None:
    _make_task(tmp_path)
    _, _, _ = _make_discovery(tmp_path)
    collector = _collector(tmp_path, [])
    before = collector.collect()
    demand = before.discovery.demands[0]
    proposal = demand.proposals[0]
    registry = before.discovery.registry
    assert registry is not None
    runner = FakeDiscoveryRunner(tmp_path)
    service = ResearchLabCommandService(collector, runner=runner)

    result = service.bind_topic(
        {
            "snapshot_revision": before.revision,
            "demand_id": demand.demand_id,
            "demand_sha256": demand.artifact.sha256,
            "proposal_id": proposal.proposal_id,
            "proposal_sha256": proposal.artifact.sha256,
            "registry_sha256": registry.sha256,
            "mapping_id": "example_research_v1",
            "actor": "Meng Fu",
            "reason": "Bind this exact proposal to the registered Volvence task.",
            "decision": "approve",
        }
    )

    assert result["action"] == "bind_topic"
    assert result["outcome"] == "bound_for_a0_submission"
    assert "research-bind-topic" in runner.calls[0]
    after = collector.collect().discovery.demands[0].proposals[0]
    assert after.effective_state == "BOUND_FOR_A0"
    assert after.binding_decision == "APPROVE"
    assert after.reviewed_by == "Meng Fu"
    assert after.available_actions == ()
    assert after.request is None
    assert not (tmp_path / "artifacts" / "research_control").exists()


def test_topic_binding_http_route_preserves_separate_a0_gate(tmp_path: Path) -> None:
    _make_task(tmp_path)
    _make_discovery(tmp_path)
    collector = _collector(tmp_path, [])
    before = collector.collect()
    demand = before.discovery.demands[0]
    proposal = demand.proposals[0]
    registry = before.discovery.registry
    assert registry is not None
    runner = FakeDiscoveryRunner(tmp_path)
    service = ResearchLabCommandService(collector, runner=runner)
    csrf_token = "x" * 32
    ui_origin = "http://localhost:3000"
    server = create_server(
        collector,
        port=0,
        command_service=service,
        allowed_origins=(ui_origin,),
        csrf_token=csrf_token,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    body = json.dumps(
        {
            "snapshot_revision": before.revision,
            "demand_id": demand.demand_id,
            "demand_sha256": demand.artifact.sha256,
            "proposal_id": proposal.proposal_id,
            "proposal_sha256": proposal.artifact.sha256,
            "registry_sha256": registry.sha256,
            "mapping_id": "example_research_v1",
            "actor": "Meng Fu",
            "reason": "Bind the exact proposal through the local workbench.",
            "decision": "approve",
        }
    ).encode()
    request = Request(
        f"http://{host}:{port}/api/v1/topics/bind",
        method="POST",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Origin": ui_origin,
            "X-Research-Lab-CSRF": csrf_token,
        },
    )
    try:
        with urlopen(request, timeout=5) as response:
            result = json.loads(response.read())
        assert result["action"] == "bind_topic"
        assert result["outcome"] == "bound_for_a0_submission"
        assert len(runner.calls) == 1
        after = collector.collect().discovery.demands[0].proposals[0]
        assert after.effective_state == "BOUND_FOR_A0"
        assert after.request is None
        assert not (tmp_path / "artifacts" / "research_control").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_rejected_a0_review_is_a_visible_terminal_blocker(tmp_path: Path) -> None:
    _make_task(tmp_path)
    request_path = _make_request(tmp_path, approved=False)
    _write_review(tmp_path, request_path, decision="REJECT")

    snapshot = _collector(tmp_path, []).collect()

    item = snapshot.items[0]
    assert item.lifecycle.stage is LifecycleStage.BLOCKED
    assert item.lifecycle.blocking_reason == "exact A0 review rejected this ResearchRequest"
    assert item.authority.a0_research_start_authorized is False
    assert item.available_actions == ("inspect_blocker",)


def test_malformed_request_is_visible_as_degraded_source(tmp_path: Path) -> None:
    _make_task(tmp_path)
    request = tmp_path / "artifacts" / "research_control" / TASK_ID / ("a" * 64) / "request.json"
    request.parent.mkdir(parents=True, exist_ok=True)
    request.write_text("{broken", encoding="utf-8")

    snapshot = _collector(tmp_path, []).collect()

    assert snapshot.items[0].lifecycle.stage is LifecycleStage.NEEDS_TASK_DESIGN
    assert any(warning.code == "INVALID_JSON_ARTIFACT" for warning in snapshot.warnings)
    control_health = next(value for value in snapshot.source_health if value.source == "control")
    assert control_health.status.value == "degraded"


def test_approval_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    _make_task(tmp_path)
    _make_request(tmp_path, approved=True, correct_sha=False)

    snapshot = _collector(tmp_path, []).collect()

    item = snapshot.items[0]
    assert item.lifecycle.stage is LifecycleStage.AWAITING_A0
    assert item.authority.a0_research_start_authorized is False
    assert any(warning.code == "APPROVAL_BINDING_MISMATCH" for warning in item.warnings)


def test_duplicate_active_runs_block_the_task(tmp_path: Path) -> None:
    _make_task(tmp_path)
    _make_request(tmp_path, approved=True)
    statuses = [
        _running_status(tmp_path, run_id="run_one", pid=111),
        _running_status(tmp_path, run_id="run_two", pid=222),
    ]

    snapshot = _collector(tmp_path, statuses).collect()

    item = snapshot.items[0]
    assert item.lifecycle.stage is LifecycleStage.BLOCKED
    assert item.run is None
    assert any(warning.code == "DUPLICATE_ACTIVE_RUNS" for warning in item.warnings)


def test_shadow_authorization_does_not_masquerade_as_applied_wiring(tmp_path: Path) -> None:
    _make_task(tmp_path)
    promotion = tmp_path / "artifacts" / "research_promotion" / TASK_ID / ("c" * 64)
    candidate_id = "research-candidate:" + "c" * 64
    candidate_sha = _write_json(
        promotion / "candidate.json",
        {
            "schema_version": "forge-research-candidate.v1",
            "candidate_id": candidate_id,
            "task_id": TASK_ID,
            "created_at": "2026-08-29T14:04:00Z",
        },
    )
    validation_sha = _write_json(
        promotion / "validation.json",
        {
            "schema_version": "forge-research-validation.v1",
            "task_id": TASK_ID,
            "candidate_id": candidate_id,
            "candidate_sha256": candidate_sha,
            "status": "PASS",
            "created_at": "2026-08-29T14:05:00Z",
        },
    )
    gate_sha = _write_json(
        promotion / "gate.json",
        {
            "schema_version": "forge-research-gate.v1",
            "task_id": TASK_ID,
            "candidate_id": candidate_id,
            "candidate_sha256": candidate_sha,
            "validation_sha256": validation_sha,
            "decision": "ALLOW",
            "created_at": "2026-08-29T14:06:00Z",
        },
    )
    _write_json(
        promotion / "receipts" / "receipt.json",
        {
            "schema_version": "forge-research-promotion-receipt.v1",
            "receipt_id": "research-receipt:" + "d" * 64,
            "task_id": TASK_ID,
            "candidate_id": candidate_id,
            "outcome": "AUTHORIZED",
            "action": "authorize",
            "transition": {
                "from_wiring": "disabled",
                "requested_wiring": "shadow",
                "resulting_wiring": "shadow",
            },
            "bindings": {
                "candidate_sha256": candidate_sha,
                "validation_sha256": validation_sha,
                "gate_sha256": gate_sha,
            },
            "authority": {"target_adapter_apply_required": True},
            "created_at": "2026-08-29T14:07:00Z",
        },
    )

    snapshot = _collector(tmp_path, []).collect()

    item = snapshot.items[0]
    assert item.lifecycle.stage is LifecycleStage.AWAITING_A1
    assert item.authority.authorized_wiring == "shadow"
    assert item.authority.runtime_wiring == "disabled"
    assert item.authority.target_adapter_apply_required is True
    assert item.evidence.shadow == "authorized_not_applied"


def test_promotion_graph_reads_owner_roots_and_refuses_cross_round_gate(tmp_path: Path) -> None:
    _make_task(tmp_path)
    promotion = tmp_path / "artifacts" / "research_promotion" / TASK_ID / ("c" * 64)
    candidate_id = "research-candidate:" + "c" * 64
    candidate_sha = _write_json(
        promotion / "candidate.json",
        {
            "schema_version": "forge-research-candidate.v1",
            "candidate_id": candidate_id,
            "task_id": TASK_ID,
            "created_at": "2026-08-29T14:04:00Z",
        },
    )
    validation_sha = _write_json(
        tmp_path / "artifacts" / "research_validation" / "formal.json",
        {
            "schema_version": "forge-research-validation.v1",
            "task_id": TASK_ID,
            "candidate_id": candidate_id,
            "candidate_sha256": candidate_sha,
            "status": "PASS",
            "created_at": "2026-08-29T14:05:00Z",
        },
    )
    _write_json(
        tmp_path / "artifacts" / "research_gate" / "stale.json",
        {
            "schema_version": "forge-research-gate.v1",
            "task_id": TASK_ID,
            "candidate_id": candidate_id,
            "candidate_sha256": candidate_sha,
            "validation_sha256": "0" * 64,
            "decision": "ALLOW",
            "created_at": "2026-08-29T14:07:00Z",
        },
    )

    stale = _collector(tmp_path, []).collect().items[0]
    assert stale.lifecycle.stage is LifecycleStage.FORMAL_VALIDATION
    assert stale.authority.formal_validation_status == "pass"
    assert stale.authority.modification_gate_decision == "not_evaluated"
    assert {ref.locator for ref in stale.bindings} >= {
        "artifacts/research_validation/formal.json",
    }
    assert any(warning.code == "GATE_VALIDATION_DIGEST_MISMATCH" for warning in stale.warnings)

    _write_json(
        tmp_path / "artifacts" / "research_gate" / "exact.json",
        {
            "schema_version": "forge-research-gate.v1",
            "task_id": TASK_ID,
            "candidate_id": candidate_id,
            "candidate_sha256": candidate_sha,
            "validation_sha256": validation_sha,
            "decision": "ALLOW",
            "created_at": "2026-08-29T14:08:00Z",
        },
    )

    exact = _collector(tmp_path, []).collect().items[0]
    assert exact.lifecycle.stage is LifecycleStage.AWAITING_A1
    assert exact.authority.modification_gate_decision == "allow"
    assert {ref.locator for ref in exact.bindings} >= {
        "artifacts/research_validation/formal.json",
        "artifacts/research_gate/exact.json",
    }


def test_completed_run_requires_canonical_exact_handoff_before_import(tmp_path: Path) -> None:
    _make_task(tmp_path)
    _make_request(tmp_path, approved=True)
    status = _running_status(tmp_path)
    status["state"] = "completed"

    before = _collector(tmp_path, [status]).collect().items[0]
    assert before.lifecycle.stage is LifecycleStage.RESEARCH_COMPLETE
    assert before.lifecycle.blocking_reason == "committed Praxist handoff is not present"
    assert before.available_actions == ("inspect_handoff",)
    assert before.run is not None and before.run.pid is None

    run_dir = Path(str(status["run_dir"]))
    _write_json(
        run_dir / "volvence_handoff.json",
        {
            "schema_version": "forge-praxist-candidate-handoff.v1",
            "task_id": TASK_ID,
            "run_id": "run_example",
            "created_at": "2026-08-29T14:04:00Z",
        },
    )

    after = _collector(tmp_path, [status]).collect().items[0]
    assert after.lifecycle.stage is LifecycleStage.RESEARCH_COMPLETE
    assert after.lifecycle.blocking_reason is None
    assert after.available_actions == ("import_candidate",)
    assert any(ref.kind == "praxist handoff" for ref in after.bindings)


def test_loopback_server_exposes_get_and_refuses_post(tmp_path: Path) -> None:
    _make_task(tmp_path)
    collector = _collector(tmp_path, [])
    server = create_server(collector, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        with urlopen(f"http://{host}:{port}/api/v1/snapshot", timeout=5) as response:
            payload = json.loads(response.read())
        assert payload["schema_version"] == "volvence-research-lab-snapshot.v2"
        assert payload["items"][0]["task_id"] == TASK_ID

        with urlopen(f"http://{host}:{port}/api/v1/tasks/{TASK_ID}", timeout=5) as response:
            task_payload = json.loads(response.read())
        assert task_payload["item"]["task_id"] == TASK_ID

        request = Request(f"http://{host}:{port}/api/v1/scan", method="POST", data=b"{}")
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=5)
        assert error.value.code == 405
        assert json.loads(error.value.read())["error"] == "read_only"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_server_rejects_non_loopback_bind(tmp_path: Path) -> None:
    _make_task(tmp_path)
    with pytest.raises(ValueError, match="loopback"):
        create_server(_collector(tmp_path, []), host="0.0.0.0")


def test_external_submission_requires_registered_exact_descriptor_and_fixed_argv(
    tmp_path: Path,
) -> None:
    _make_task(tmp_path)
    request_path, descriptor_path = _make_external_request(tmp_path)
    request_path.unlink()
    collector = _collector(tmp_path, [])
    previous = collector.collect()
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor_sha = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    calls: list[tuple[str, ...]] = []

    def runner(arguments: object) -> OwnerCommandResult:
        assert isinstance(arguments, tuple)
        calls.append(arguments)
        created, _ = _make_external_request(tmp_path)
        request = json.loads(created.read_text(encoding="utf-8"))
        return OwnerCommandResult(
            0,
            json.dumps(
                {
                    "descriptor_id": descriptor["descriptor_id"],
                    "domain_id": "foundry",
                    "request_id": request["request_id"],
                    "request_sha256": hashlib.sha256(created.read_bytes()).hexdigest(),
                }
            ),
            "",
        )

    service = ResearchLabCommandService(
        collector,
        runner=runner,
        external_domain_roots={"foundry": tmp_path / "external-foundry"},
    )
    result = service.submit_external(
        {
            "snapshot_revision": previous.revision,
            "domain_id": "foundry",
            "descriptor_locator": "artifacts/research_lab/descriptor.json",
            "descriptor_id": descriptor["descriptor_id"],
            "descriptor_sha256": descriptor_sha,
            "actor": "Meng Fu",
            "reason": "Submit the exact Foundry simulation Intent for A0 review.",
        }
    )

    assert result["action"] == "submit_external"
    assert result["task_id"] == EXTERNAL_TASK_ID
    assert result["outcome"] == "awaiting_a0"
    assert calls == [
        (
            "--repo-root",
            str(tmp_path),
            "research-submit-external",
            str(descriptor_path.resolve()),
            "--requested-by",
            "Meng Fu",
            "--reason",
            "Submit the exact Foundry simulation Intent for A0 review.",
            "--json",
        )
    ]


def test_external_handoff_command_never_enters_volvence_promotion(tmp_path: Path) -> None:
    _make_task(tmp_path)
    request_path, _ = _make_external_request(tmp_path, approved=True, completed=True)
    collector = _collector(tmp_path, [])
    snapshot = collector.collect()
    item = snapshot.get_task(EXTERNAL_TASK_ID)
    assert item is not None
    request_ref = next(ref for ref in item.bindings if ref.kind == "research request")
    calls: list[tuple[str, ...]] = []

    def runner(arguments: object) -> OwnerCommandResult:
        assert isinstance(arguments, tuple)
        calls.append(arguments)
        _make_external_request(tmp_path, approved=True, completed=True, handed_off=True)
        handoff_path = request_path.parent / "handoffs/handoff.json"
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        return OwnerCommandResult(
            0,
            json.dumps(
                {
                    "handoff_id": handoff["handoff_id"],
                    "handoff_sha256": hashlib.sha256(handoff_path.read_bytes()).hexdigest(),
                }
            ),
            "",
        )

    service = ResearchLabCommandService(collector, runner=runner)
    result = service.record_external_handoff(
        {
            "snapshot_revision": snapshot.revision,
            "task_id": EXTERNAL_TASK_ID,
            "artifact_id": request_ref.artifact_id,
            "artifact_sha256": request_ref.sha256,
            "actor": "Meng Fu",
            "reason": "Seal simulation evidence for Foundry-owned review.",
        }
    )

    assert result["action"] == "record_external_handoff"
    assert result["outcome"] == "handed_off_for_external_review"
    assert "research-handoff-external" in calls[0]
    assert not (tmp_path / "artifacts/research_promotion" / EXTERNAL_TASK_ID).exists()


def test_exact_a0_service_delegates_fixed_argv_and_refreshes_snapshot(tmp_path: Path) -> None:
    _make_task(tmp_path)
    _make_request(tmp_path, approved=False)
    collector = _collector(tmp_path, [])
    before = collector.collect()
    request_ref = next(ref for ref in before.items[0].bindings if ref.kind == "research request")
    runner = FakeForgeRunner(tmp_path)
    service = ResearchLabCommandService(collector, runner=runner)

    result = service.review_a0(
        {
            "snapshot_revision": before.revision,
            "task_id": TASK_ID,
            "artifact_id": request_ref.artifact_id,
            "artifact_sha256": request_ref.sha256,
            "actor": "Meng Fu",
            "reason": "Approve the exact frozen bounded task",
            "decision": "approve",
        }
    )

    assert result["outcome"] == "approved"
    assert result["previous_revision"] == before.revision
    assert result["current_revision"] != before.revision
    assert runner.calls == [
        (
            "--repo-root",
            str(tmp_path),
            "research-approve",
            str(tmp_path / request_ref.locator),
            "--approved-by",
            "Meng Fu",
            "--reason",
            "Approve the exact frozen bounded task",
        )
    ]
    after = collector.collect().items[0]
    assert after.lifecycle.stage is LifecycleStage.PREFLIGHT
    assert after.authority.a0_research_start_authorized is True
    assert after.available_actions == ("reconcile",)


def test_command_service_rejects_stale_revision_before_owner_call(tmp_path: Path) -> None:
    _make_task(tmp_path)
    _make_request(tmp_path, approved=False)
    collector = _collector(tmp_path, [])
    request_ref = next(ref for ref in collector.collect().items[0].bindings if ref.kind == "research request")
    runner = FakeForgeRunner(tmp_path)
    service = ResearchLabCommandService(collector, runner=runner)

    with pytest.raises(PortalCommandError) as error:
        service.review_a0(
            {
                "snapshot_revision": "0" * 64,
                "task_id": TASK_ID,
                "artifact_id": request_ref.artifact_id,
                "artifact_sha256": request_ref.sha256,
                "actor": "Meng Fu",
                "reason": "Stale review must not execute",
                "decision": "approve",
            }
        )

    assert error.value.code == "stale_snapshot"
    assert runner.calls == []


def test_command_service_rejects_wrong_request_digest_before_owner_call(tmp_path: Path) -> None:
    _make_task(tmp_path)
    _make_request(tmp_path, approved=False)
    collector = _collector(tmp_path, [])
    snapshot = collector.collect()
    request_ref = next(ref for ref in snapshot.items[0].bindings if ref.kind == "research request")
    runner = FakeForgeRunner(tmp_path)
    service = ResearchLabCommandService(collector, runner=runner)

    with pytest.raises(PortalCommandError) as error:
        service.review_a0(
            {
                "snapshot_revision": snapshot.revision,
                "task_id": TASK_ID,
                "artifact_id": request_ref.artifact_id,
                "artifact_sha256": "0" * 64,
                "actor": "Meng Fu",
                "reason": "Wrong bytes must never receive approval",
                "decision": "approve",
            }
        )

    assert error.value.code == "artifact_digest_mismatch"
    assert runner.calls == []


def test_approved_request_reconcile_uses_one_exact_bounded_pass(tmp_path: Path) -> None:
    _make_task(tmp_path)
    _make_request(tmp_path, approved=True)
    collector = _collector(tmp_path, [])
    snapshot = collector.collect()
    request_ref = next(ref for ref in snapshot.items[0].bindings if ref.kind == "research request")
    runner = FakeForgeRunner(tmp_path)
    service = ResearchLabCommandService(collector, runner=runner)

    result = service.reconcile(
        {
            "snapshot_revision": snapshot.revision,
            "task_id": TASK_ID,
            "artifact_id": request_ref.artifact_id,
            "artifact_sha256": request_ref.sha256,
            "actor": "Meng Fu",
            "reason": "Run one approved control-plane reconciliation",
        }
    )

    assert result["outcome"] == "reconciled"
    assert result["message"] == "Forge reconciliation state: WAITING_FOR_CAPACITY; run_id=-"
    assert runner.calls == [
        (
            "--repo-root",
            str(tmp_path),
            "research-reconcile",
            "--once",
            "--request",
            str(tmp_path / request_ref.locator),
            "--json",
        )
    ]


def test_running_task_cannot_reconcile_or_create_a_duplicate_run(tmp_path: Path) -> None:
    _make_task(tmp_path)
    _make_request(tmp_path, approved=True)
    collector = _collector(tmp_path, [_running_status(tmp_path)])
    snapshot = collector.collect()
    request_ref = next(ref for ref in snapshot.items[0].bindings if ref.kind == "research request")
    runner = FakeForgeRunner(tmp_path)
    service = ResearchLabCommandService(collector, runner=runner)

    with pytest.raises(PortalCommandError) as error:
        service.reconcile(
            {
                "snapshot_revision": snapshot.revision,
                "task_id": TASK_ID,
                "artifact_id": request_ref.artifact_id,
                "artifact_sha256": request_ref.sha256,
                "actor": "Meng Fu",
                "reason": "Inspect duplicate-run protection",
            }
        )

    assert error.value.code == "action_not_available"
    assert runner.calls == []


def test_candidate_import_delegates_exact_completed_run_without_starting_praxist(tmp_path: Path) -> None:
    task_path = _make_task(tmp_path)
    _make_request(tmp_path, approved=True)
    status = _running_status(tmp_path)
    status["state"] = "completed"
    run_dir = Path(str(status["run_dir"]))
    handoff_path = run_dir / "volvence_handoff.json"
    _write_json(
        handoff_path,
        {
            "schema_version": "forge-praxist-candidate-handoff.v1",
            "task_id": TASK_ID,
            "run_id": "run_example",
            "created_at": "2026-08-29T14:04:00Z",
        },
    )
    collector = _collector(tmp_path, [status])
    before = collector.collect()
    item = before.items[0]
    task_ref = _binding(item, "task")
    handoff_ref = _binding(item, "praxist handoff")
    runner = FakePromotionRunner(tmp_path)
    service = ResearchLabCommandService(collector, runner=runner)

    result = service.import_candidate(
        {
            "snapshot_revision": before.revision,
            "task_id": TASK_ID,
            "task_artifact_id": task_ref.artifact_id,
            "task_sha256": task_ref.sha256,
            "handoff_sha256": handoff_ref.sha256,
            "run_id": "run_example",
            "actor": "Meng Fu",
            "reason": "Seal the exact completed research boundary",
        }
    )

    assert result["outcome"] == "sealed"
    assert result["binding"]["kind"] == "candidate"
    assert runner.calls == [
        (
            "--repo-root",
            str(tmp_path),
            "research-import-praxist",
            str(task_path),
            str(handoff_path),
            "--run-dir",
            str(run_dir),
        )
    ]
    after = collector.collect().items[0]
    assert after.lifecycle.stage is LifecycleStage.CANDIDATE_RETAINED
    assert after.available_actions == ("run_formal_validation",)


def test_a1_authorization_and_rollback_delegate_exact_receipt_chain(tmp_path: Path) -> None:
    task_path = _make_task(tmp_path)
    candidate_path = _make_candidate(tmp_path)
    validation_path, gate_path = _make_formal_gate(tmp_path, candidate_path, name="shadow", minute=2)
    collector = _collector(tmp_path, [])
    before = collector.collect()
    item = before.items[0]
    assert item.available_actions == ("authorize_shadow",)
    task_ref = _binding(item, "task")
    candidate_ref = _binding(item, "candidate")
    validation_ref = _binding(item, "validation")
    gate_ref = _binding(item, "gate")
    runner = FakePromotionRunner(tmp_path)
    service = ResearchLabCommandService(collector, runner=runner)

    authorized = service.authorize_shadow(
        {
            "snapshot_revision": before.revision,
            "task_id": TASK_ID,
            "task_artifact_id": task_ref.artifact_id,
            "task_sha256": task_ref.sha256,
            "candidate_artifact_id": candidate_ref.artifact_id,
            "candidate_sha256": candidate_ref.sha256,
            "validation_sha256": validation_ref.sha256,
            "gate_sha256": gate_ref.sha256,
            "previous_receipt_id": None,
            "previous_receipt_sha256": None,
            "actor": "A1 Reviewer",
            "reason": "Authorize exact bounded SHADOW evidence",
        }
    )

    assert authorized["outcome"] == "authorized"
    assert runner.calls[0] == (
        "--repo-root",
        str(tmp_path),
        "research-authorize",
        str(task_path),
        str(candidate_path),
        str(validation_path),
        str(gate_path),
        "--to-wiring",
        "shadow",
        "--authorized-by",
        "A1 Reviewer",
        "--reason",
        "Authorize exact bounded SHADOW evidence",
    )
    shadow = collector.collect()
    shadow_item = shadow.items[0]
    assert shadow_item.authority.authorized_wiring == "shadow"
    assert shadow_item.authority.runtime_wiring == "disabled"
    assert shadow_item.available_actions == ("rollback",)
    receipt_ref = _binding(shadow_item, "receipt")

    rolled_back = service.rollback(
        {
            "snapshot_revision": shadow.revision,
            "task_id": TASK_ID,
            "receipt_id": receipt_ref.artifact_id,
            "receipt_sha256": receipt_ref.sha256,
            "actor": "Rollback Operator",
            "reason": "Exercise the adjacent downgrade boundary",
        }
    )

    assert rolled_back["outcome"] == "authorized"
    assert runner.calls[1] == (
        "--repo-root",
        str(tmp_path),
        "research-rollback",
        str(tmp_path / receipt_ref.locator),
        "--to-wiring",
        "disabled",
        "--authorized-by",
        "Rollback Operator",
        "--reason",
        "Exercise the adjacent downgrade boundary",
    )
    assert collector.collect().items[0].lifecycle.stage is LifecycleStage.ROLLED_BACK


def test_a1_rejects_unreviewed_gate_digest_before_owner_call(tmp_path: Path) -> None:
    _make_task(tmp_path)
    candidate_path = _make_candidate(tmp_path)
    _make_formal_gate(tmp_path, candidate_path, name="shadow", minute=2)
    collector = _collector(tmp_path, [])
    snapshot = collector.collect()
    item = snapshot.items[0]
    runner = FakePromotionRunner(tmp_path)
    service = ResearchLabCommandService(collector, runner=runner)

    with pytest.raises(PortalCommandError) as error:
        service.authorize_shadow(
            {
                "snapshot_revision": snapshot.revision,
                "task_id": TASK_ID,
                "task_artifact_id": _binding(item, "task").artifact_id,
                "task_sha256": _binding(item, "task").sha256,
                "candidate_artifact_id": _binding(item, "candidate").artifact_id,
                "candidate_sha256": _binding(item, "candidate").sha256,
                "validation_sha256": _binding(item, "validation").sha256,
                "gate_sha256": "0" * 64,
                "previous_receipt_id": None,
                "previous_receipt_sha256": None,
                "actor": "A1 Reviewer",
                "reason": "Reject unreviewed Gate bytes",
            }
        )

    assert error.value.code == "artifact_digest_mismatch"
    assert runner.calls == []


def test_a1_preserves_legal_blocked_receipt_as_a_command_result(tmp_path: Path) -> None:
    _make_task(tmp_path)
    candidate_path = _make_candidate(tmp_path)
    _make_formal_gate(tmp_path, candidate_path, name="shadow", minute=2)
    collector = _collector(tmp_path, [])
    snapshot = collector.collect()
    item = snapshot.items[0]
    runner = FakePromotionRunner(tmp_path, authorize_outcome="BLOCKED")
    service = ResearchLabCommandService(collector, runner=runner)

    result = service.authorize_shadow(
        {
            "snapshot_revision": snapshot.revision,
            "task_id": TASK_ID,
            "task_artifact_id": _binding(item, "task").artifact_id,
            "task_sha256": _binding(item, "task").sha256,
            "candidate_artifact_id": _binding(item, "candidate").artifact_id,
            "candidate_sha256": _binding(item, "candidate").sha256,
            "validation_sha256": _binding(item, "validation").sha256,
            "gate_sha256": _binding(item, "gate").sha256,
            "previous_receipt_id": None,
            "previous_receipt_sha256": None,
            "actor": "A1 Reviewer",
            "reason": "Retain the exact negative admission result",
        }
    )

    assert result["outcome"] == "blocked"
    blocked = collector.collect().items[0]
    assert blocked.lifecycle.stage is LifecycleStage.BLOCKED
    assert blocked.lifecycle.blocking_reason == "fixture gate block"


def test_a2_requires_fresh_exact_evidence_and_previous_shadow_receipt(tmp_path: Path) -> None:
    _make_task(tmp_path)
    candidate_path = _make_candidate(tmp_path)
    _make_formal_gate(tmp_path, candidate_path, name="shadow", minute=2)
    collector = _collector(tmp_path, [])
    runner = FakePromotionRunner(tmp_path)
    service = ResearchLabCommandService(collector, runner=runner)
    first = collector.collect()
    first_item = first.items[0]
    service.authorize_shadow(
        {
            "snapshot_revision": first.revision,
            "task_id": TASK_ID,
            "task_artifact_id": _binding(first_item, "task").artifact_id,
            "task_sha256": _binding(first_item, "task").sha256,
            "candidate_artifact_id": _binding(first_item, "candidate").artifact_id,
            "candidate_sha256": _binding(first_item, "candidate").sha256,
            "validation_sha256": _binding(first_item, "validation").sha256,
            "gate_sha256": _binding(first_item, "gate").sha256,
            "previous_receipt_id": None,
            "previous_receipt_sha256": None,
            "actor": "A1 Reviewer",
            "reason": "Authorize SHADOW before fresh active evidence",
        }
    )
    _make_formal_gate(tmp_path, candidate_path, name="active", minute=10)
    active_ready = collector.collect()
    item = active_ready.items[0]
    assert item.lifecycle.stage is LifecycleStage.AWAITING_A2
    assert item.available_actions == ("authorize_active",)
    previous_ref = _binding(item, "receipt")

    result = service.authorize_active(
        {
            "snapshot_revision": active_ready.revision,
            "task_id": TASK_ID,
            "task_artifact_id": _binding(item, "task").artifact_id,
            "task_sha256": _binding(item, "task").sha256,
            "candidate_artifact_id": _binding(item, "candidate").artifact_id,
            "candidate_sha256": _binding(item, "candidate").sha256,
            "validation_sha256": _binding(item, "validation").sha256,
            "gate_sha256": _binding(item, "gate").sha256,
            "previous_receipt_id": previous_ref.artifact_id,
            "previous_receipt_sha256": previous_ref.sha256,
            "actor": "A2 Reviewer",
            "reason": "Authorize ACTIVE from fresh exact canary evidence",
        }
    )

    assert result["outcome"] == "authorized"
    assert "--previous-receipt" in runner.calls[-1]
    assert runner.calls[-1][runner.calls[-1].index("--previous-receipt") + 1] == str(tmp_path / previous_ref.locator)
    final = collector.collect().items[0]
    assert final.authority.authorized_wiring == "active"
    assert final.authority.runtime_wiring == "disabled"
    assert final.available_actions == ("rollback",)


def test_mutation_http_requires_origin_csrf_and_exact_binding(tmp_path: Path) -> None:
    _make_task(tmp_path)
    _make_request(tmp_path, approved=False)
    collector = _collector(tmp_path, [])
    snapshot = collector.collect()
    request_ref = next(ref for ref in snapshot.items[0].bindings if ref.kind == "research request")
    runner = FakeForgeRunner(tmp_path)
    service = ResearchLabCommandService(collector, runner=runner)
    csrf_token = "x" * 32
    ui_origin = "http://localhost:3000"
    server = create_server(
        collector,
        port=0,
        command_service=service,
        allowed_origins=(ui_origin,),
        csrf_token=csrf_token,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    endpoint = f"http://{host}:{port}/api/v1/a0/review"
    body = json.dumps(
        {
            "snapshot_revision": snapshot.revision,
            "task_id": TASK_ID,
            "artifact_id": request_ref.artifact_id,
            "artifact_sha256": request_ref.sha256,
            "actor": "Meng Fu",
            "reason": "Approve through the local exact-bound workbench",
            "decision": "approve",
        }
    ).encode()
    try:
        with urlopen(f"http://{host}:{port}/api/v1/session", timeout=5) as response:
            session = json.loads(response.read())
        assert session["mutations_enabled"] is True
        assert session["csrf_token"] == csrf_token
        assert set(session["supported_actions"]) == {
            "bind_topic",
            "review_a0",
            "reconcile",
            "record_external_handoff",
            "import_candidate",
            "authorize_shadow",
            "authorize_active",
            "rollback",
        }

        missing_origin = Request(
            endpoint,
            method="POST",
            data=body,
            headers={"Content-Type": "application/json", "X-Research-Lab-CSRF": csrf_token},
        )
        with pytest.raises(HTTPError) as forbidden:
            urlopen(missing_origin, timeout=5)
        assert forbidden.value.code == 403
        assert json.loads(forbidden.value.read())["error"] == "origin_forbidden"
        assert runner.calls == []

        wrong_csrf = Request(
            endpoint,
            method="POST",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Origin": ui_origin,
                "X-Research-Lab-CSRF": "y" * 32,
            },
        )
        with pytest.raises(HTTPError) as csrf_forbidden:
            urlopen(wrong_csrf, timeout=5)
        assert csrf_forbidden.value.code == 403
        assert json.loads(csrf_forbidden.value.read())["error"] == "csrf_forbidden"
        assert runner.calls == []

        approved = Request(
            endpoint,
            method="POST",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Origin": ui_origin,
                "X-Research-Lab-CSRF": csrf_token,
            },
        )
        with urlopen(approved, timeout=5) as response:
            result = json.loads(response.read())
        assert result["outcome"] == "approved"
        assert len(runner.calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_promotion_http_route_delegates_exact_a1_command(tmp_path: Path) -> None:
    _make_task(tmp_path)
    candidate_path = _make_candidate(tmp_path)
    _make_formal_gate(tmp_path, candidate_path, name="shadow", minute=2)
    collector = _collector(tmp_path, [])
    snapshot = collector.collect()
    item = snapshot.items[0]
    runner = FakePromotionRunner(tmp_path)
    service = ResearchLabCommandService(collector, runner=runner)
    csrf_token = "x" * 32
    ui_origin = "http://localhost:3000"
    server = create_server(
        collector,
        port=0,
        command_service=service,
        allowed_origins=(ui_origin,),
        csrf_token=csrf_token,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    body = json.dumps(
        {
            "snapshot_revision": snapshot.revision,
            "task_id": TASK_ID,
            "task_artifact_id": _binding(item, "task").artifact_id,
            "task_sha256": _binding(item, "task").sha256,
            "candidate_artifact_id": _binding(item, "candidate").artifact_id,
            "candidate_sha256": _binding(item, "candidate").sha256,
            "validation_sha256": _binding(item, "validation").sha256,
            "gate_sha256": _binding(item, "gate").sha256,
            "previous_receipt_id": None,
            "previous_receipt_sha256": None,
            "actor": "A1 Reviewer",
            "reason": "Exercise the exact local A1 route",
        }
    ).encode()
    request = Request(
        f"http://{host}:{port}/api/v1/a1/authorize-shadow",
        method="POST",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Origin": ui_origin,
            "X-Research-Lab-CSRF": csrf_token,
        },
    )
    try:
        with urlopen(request, timeout=5) as response:
            result = json.loads(response.read())
        assert result["action"] == "authorize_shadow"
        assert result["outcome"] == "authorized"
        assert len(runner.calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
