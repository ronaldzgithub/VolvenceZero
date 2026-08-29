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

from volvence_labs.portal import LifecycleStage, ResearchLabCollector, create_server


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
    _write_json(
        promotion / "candidate.json",
        {
            "schema_version": "forge-research-candidate.v1",
            "candidate_id": "research-candidate:" + "c" * 64,
            "task_id": TASK_ID,
            "created_at": "2026-08-29T14:04:00Z",
        },
    )
    _write_json(
        promotion / "validation.json",
        {
            "schema_version": "forge-research-validation.v1",
            "task_id": TASK_ID,
            "status": "PASS",
            "created_at": "2026-08-29T14:05:00Z",
        },
    )
    _write_json(
        promotion / "gate.json",
        {
            "schema_version": "forge-research-gate.v1",
            "task_id": TASK_ID,
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
            "outcome": "AUTHORIZED",
            "action": "authorize",
            "transition": {
                "from_wiring": "disabled",
                "requested_wiring": "shadow",
                "resulting_wiring": "shadow",
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
