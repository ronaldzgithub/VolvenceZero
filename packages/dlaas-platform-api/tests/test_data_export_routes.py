from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dlaas_platform_api import app as app_module
from dlaas_platform_api.app import attach_dlaas_routes
from dlaas_platform_launcher import INSTANCE_MANAGER_APP_KEY
from lifeform_service.session_manager import SessionManager
from volvence_zero.memory import (
    MemoryStratum,
    MemoryWriteRequest,
    Track,
    UserIdentity,
    build_scoped_memory_store,
    scoped_memory_dir,
)


def _memory_export(*, durability: str = "restart_durable") -> dict[str, object]:
    return {
        "encoding": "base64",
        "media_type": "application/vnd.volvence.memory-checkpoint+json",
        "payload_base64": "eyJfX3dpdG5lc3MiOiJwZXJzaXN0ZWQifQ==",
        "receipt": {
            "schema_id": "volvence.memory.scoped-checkpoint-export",
            "schema_version": 1,
            "operation": "export",
            "checkpoint_id": "persist-memory/store",
            "checkpoint_key": "memory/store",
            "checkpoint_version": 3,
            "payload_sha256": "a" * 64,
            "payload_bytes": 28,
            "entry_count": 2,
            "durability": durability,
            "completed_at_ms": 1_800_000_000_000,
            "canonical_roundtrip_matches": True,
        },
    }


class _ExportManager:
    def __init__(self, exported: dict[str, object] | None) -> None:
        self.exported = exported
        self.calls: list[str] = []

    def export_persisted_memory_scope(self, end_user_ref: str):
        self.calls.append(end_user_ref)
        return self.exported


def _app(manager: object) -> web.Application:
    app = web.Application()
    app["session_manager"] = manager
    return attach_dlaas_routes(app)


async def test_data_export_reads_actual_persisted_bytes_through_session_manager(
    tmp_path: Path,
) -> None:
    identity = UserIdentity(user_id="player-1", scope_key="player-1")
    store = build_scoped_memory_store(identity=identity, root_dir=tmp_path)
    store.write(
        MemoryWriteRequest(
            content="The player's action changed the next scene.",
            track=Track.WORLD,
            stratum=MemoryStratum.DURABLE,
            strength=0.95,
        ),
        timestamp_ms=1_800_000_000_000,
    )
    assert store.save_to_backend()
    persisted = (
        scoped_memory_dir(root_dir=tmp_path, user_id="player-1")
        / "memory__store_v1.json"
    ).read_bytes()
    manager = SessionManager.__new__(SessionManager)
    manager._alpha_memory_scope_root_dir = str(tmp_path)  # noqa: SLF001
    manager._tenant_id = ""  # noqa: SLF001
    manager._scope_strategy = ""  # noqa: SLF001
    manager._memory_backend_name = ""  # noqa: SLF001

    client = TestClient(TestServer(_app(manager)))
    await client.start_server()
    try:
        response = await client.post(
            "/dlaas/v1/instances/ai-1/data/export",
            json={"end_user_ref": "player-1", "scopes": ["runtime"]},
        )
        body = await response.json()
        assert response.status == 201
        exported = base64.b64decode(
            body["memory_export"]["payload_base64"],
            validate=True,
        )
        assert exported == persisted
        assert body["export_receipt"]["payload_sha256"] == hashlib.sha256(
            persisted
        ).hexdigest()
        assert body["export_receipt"]["entry_count"] == 1
    finally:
        await client.close()


async def test_data_export_returns_real_inline_owner_payload_and_receipt_only_audit() -> None:
    manager = _ExportManager(_memory_export())
    app = _app(manager)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/dlaas/v1/instances/ai-1/data/export",
            json={
                "contract_id": "contract-1",
                "end_user_ref": "player-1",
                "scopes": ["runtime", "platform"],
                "reason": "account_export",
            },
        )
        body = await response.json()
        assert response.status == 201
        assert body["status"] == "completed"
        assert body["request_status"] == "partial"
        assert body["complete"] is False
        assert body["artifact_ref"] == ""
        assert body["delivery"] == "inline"
        assert body["requested_scopes"] == ["runtime", "platform"]
        assert body["exported_scopes"] == ["runtime"]
        assert body["unexported_scopes"] == ["platform"]
        assert body["memory_export"] == _memory_export()
        assert manager.calls == ["player-1"]

        # Platform governance/audit keeps only the owner receipt, never the
        # portable checkpoint bytes/base64 body.
        audit_payloads = [
            event.to_json()
            for event in app[app_module._AUDIT_EVENTS_KEY].values()
        ]
        assert audit_payloads
        assert "payload_base64" not in json.dumps(audit_payloads)
        assert "export_receipt" in json.dumps(audit_payloads)
    finally:
        await client.close()


async def test_data_export_requires_an_existing_restart_durable_scope() -> None:
    missing = _ExportManager(None)
    app = _app(missing)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/dlaas/v1/instances/ai-1/data/export",
            json={"end_user_ref": "missing", "scopes": ["runtime"]},
        )
        assert response.status == 404
        assert (await response.json())["error"] == "memory_scope_not_found"
    finally:
        await client.close()

    ephemeral = _ExportManager(_memory_export(durability="process_local"))
    app = _app(ephemeral)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/dlaas/v1/instances/ai-1/data/export",
            json={"end_user_ref": "player-1", "scopes": ["runtime"]},
        )
        assert response.status == 409
        assert (await response.json())["error"] == "memory_export_not_restart_durable"
    finally:
        await client.close()


async def test_data_export_validates_subject_and_runtime_scope() -> None:
    manager = _ExportManager(_memory_export())
    client = TestClient(TestServer(_app(manager)))
    await client.start_server()
    try:
        missing_subject = await client.post(
            "/dlaas/v1/instances/ai-1/data/export",
            json={"scopes": ["runtime"]},
        )
        assert missing_subject.status == 400
        platform_only = await client.post(
            "/dlaas/v1/instances/ai-1/data/export",
            json={"end_user_ref": "player-1", "scopes": ["platform"]},
        )
        assert platform_only.status == 422
        assert manager.calls == []
    finally:
        await client.close()


class _RemoteExportLauncher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def forward_data_export(self, *, ai_id: str, end_user_ref: str):
        self.calls.append((ai_id, end_user_ref))
        return 200, {"status": "ok", "memory_export": _memory_export()}


async def test_data_export_uses_target_pod_forwarder_without_local_read() -> None:
    manager = _ExportManager(None)
    launcher = _RemoteExportLauncher()
    app = _app(manager)
    app[INSTANCE_MANAGER_APP_KEY] = launcher
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/dlaas/v1/instances/ai-remote/data/export",
            json={"end_user_ref": "player-1", "scopes": ["runtime"]},
        )
        assert response.status == 201
        assert (await response.json())["memory_export"] == _memory_export()
        assert launcher.calls == [("ai-remote", "player-1")]
        assert manager.calls == []
    finally:
        await client.close()
