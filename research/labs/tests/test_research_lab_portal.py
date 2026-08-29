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


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    assert snapshot.schema_version == "volvence-research-lab-snapshot.v1"
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


def test_request_without_approval_is_awaiting_a0(tmp_path: Path) -> None:
    _make_task(tmp_path)
    _make_request(tmp_path, approved=False)
    snapshot = _collector(tmp_path, []).collect()

    item = snapshot.items[0]
    assert item.lifecycle.stage is LifecycleStage.AWAITING_A0
    assert item.authority.a0_research_start_authorized is False
    assert item.available_actions == ("review_a0",)
    assert snapshot.summary.awaiting_human == 1


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
        assert payload["schema_version"] == "volvence-research-lab-snapshot.v1"
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
            "review_a0",
            "reconcile",
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
