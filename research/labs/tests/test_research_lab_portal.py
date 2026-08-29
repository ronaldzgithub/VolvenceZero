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
    LifecycleStage,
    OwnerCommandResult,
    PortalCommandError,
    ResearchLabCollector,
    ResearchLabCommandService,
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
    _write_json(
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
            "bindings": {"candidate_sha256": candidate_sha},
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
