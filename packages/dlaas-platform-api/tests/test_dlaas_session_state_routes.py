from __future__ import annotations

from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dlaas_platform_api.app import attach_dlaas_routes
from dlaas_platform_launcher import INSTANCE_MANAGER_APP_KEY
from lifeform_service.app import create_app as create_lifeform_app
from lifeform_service.session_manager import SessionNotFoundError
from lifeform_service.verticals import _try_coding


class _Receipt:
    def to_json(self) -> dict[str, object]:
        return {
            "schema_id": "volvence.memory.checkpoint-persistence-receipt",
            "schema_version": 1,
            "operation": "load",
            "durability": "restart_durable",
        }


class _ReadOnlySessionManager:
    def __init__(self, sessions: dict[str, object]) -> None:
        self.sessions = sessions
        self.reads: list[str] = []
        self.create_calls = 0

    async def get_session(self, session_id: str) -> object:
        self.reads.append(session_id)
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(session_id) from exc

    async def create_session(self, **_kwargs) -> object:
        self.create_calls += 1
        raise AssertionError("GET session state must never create a session")

    def vertical_name_for(self, session_id: str) -> str:
        if session_id not in self.sessions:
            raise SessionNotFoundError(session_id)
        return "character"


def _local_app(manager: _ReadOnlySessionManager) -> web.Application:
    app = web.Application()
    app["session_manager"] = manager
    return attach_dlaas_routes(app)


async def test_session_state_returns_the_kernel_owned_open_scene() -> None:
    manager = _ReadOnlySessionManager(
        {
            "session-1": SimpleNamespace(
                open_scene=SimpleNamespace(scene_id="scene-00023"),
                latest_memory_checkpoint_receipt=_Receipt(),
            )
        }
    )
    client = TestClient(TestServer(_local_app(manager)))
    await client.start_server()
    try:
        response = await client.get(
            "/dlaas/v1/instances/ai-1/sessions/session-1"
        )
        assert response.status == 200
        assert await response.json() == {
            "status": "ok",
            "contract": "dlaas.session-state.v1",
            "ai_id": "ai-1",
            "session_id": "session-1",
            "vertical": "character",
            "exists": True,
            "open_scene_id": "scene-00023",
            "memory_checkpoint_receipt": _Receipt().to_json(),
        }
        assert manager.reads == ["session-1"]
        assert manager.create_calls == 0
    finally:
        await client.close()


async def test_session_state_reads_a_real_lifeform_session_scene(monkeypatch) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    vertical = _try_coding()
    assert vertical is not None
    app = attach_dlaas_routes(create_lifeform_app(vertical=vertical))
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        created = await client.post(
            "/dlaas/v1/instances/ai-coding/sessions",
            json={"session_id": "real-session"},
        )
        assert created.status == 201
        session = await app["session_manager"].get_session("real-session")
        await session.run_turn("Open a kernel-owned scene.")
        assert session.open_scene is not None

        response = await client.get(
            "/dlaas/v1/instances/ai-coding/sessions/real-session"
        )
        body = await response.json()
        assert response.status == 200
        assert body["open_scene_id"] == session.open_scene.scene_id
        assert "memory_checkpoint_receipt" in body
    finally:
        await client.close()


async def test_session_state_reports_no_open_scene_without_mutation() -> None:
    manager = _ReadOnlySessionManager(
        {
            "session-1": SimpleNamespace(
                open_scene=None,
                latest_memory_checkpoint_receipt=None,
            )
        }
    )
    client = TestClient(TestServer(_local_app(manager)))
    await client.start_server()
    try:
        response = await client.get(
            "/dlaas/instances/ai-1/sessions/session-1"
        )
        body = await response.json()
        assert response.status == 200
        assert body["exists"] is True
        assert body["open_scene_id"] is None
        assert body["memory_checkpoint_receipt"] is None
        assert manager.create_calls == 0
    finally:
        await client.close()


async def test_session_state_missing_is_explicit_and_never_creates() -> None:
    manager = _ReadOnlySessionManager({})
    client = TestClient(TestServer(_local_app(manager)))
    await client.start_server()
    try:
        response = await client.get(
            "/dlaas/v1/instances/ai-1/sessions/missing"
        )
        assert response.status == 404
        assert await response.json() == {
            "status": "not_found",
            "error": "session_not_found",
            "ai_id": "ai-1",
            "session_id": "missing",
            "exists": False,
            "open_scene_id": None,
        }
        assert manager.reads == ["missing"]
        assert manager.create_calls == 0
    finally:
        await client.close()


class _RemoteSessionStateLauncher:
    def __init__(self, *, missing: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.missing = missing

    async def forward_interaction(self, *, ai_id, envelope):
        raise AssertionError((ai_id, envelope))

    async def forward_session_state(self, *, ai_id, session_id):
        self.calls.append((ai_id, session_id))
        if self.missing:
            return 404, {
                "status": "not_found",
                "error": "session_not_found",
                "ai_id": ai_id,
                "session_id": session_id,
                "exists": False,
                "open_scene_id": None,
            }
        return 200, {
            "status": "ok",
            "contract": "dlaas.session-state.v1",
            "ai_id": ai_id,
            "session_id": session_id,
            "vertical": "character",
            "exists": True,
            "open_scene_id": "scene-remote",
            "memory_checkpoint_receipt": None,
        }


async def test_session_state_uses_the_multi_pod_sticky_forwarder() -> None:
    manager = _ReadOnlySessionManager({})
    launcher = _RemoteSessionStateLauncher()
    app = _local_app(manager)
    app[INSTANCE_MANAGER_APP_KEY] = launcher
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get(
            "/dlaas/v1/instances/ai-remote/sessions/session-remote"
        )
        assert response.status == 200
        assert (await response.json())["open_scene_id"] == "scene-remote"
        assert launcher.calls == [("ai-remote", "session-remote")]
        assert manager.reads == []
    finally:
        await client.close()


async def test_session_state_forwards_remote_not_found_without_local_fallback() -> None:
    manager = _ReadOnlySessionManager({})
    launcher = _RemoteSessionStateLauncher(missing=True)
    app = _local_app(manager)
    app[INSTANCE_MANAGER_APP_KEY] = launcher
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get(
            "/dlaas/v1/instances/ai-remote/sessions/missing"
        )
        body = await response.json()
        assert response.status == 404
        assert body["error"] == "session_not_found"
        assert body["exists"] is False
        assert launcher.calls == [("ai-remote", "missing")]
        assert manager.reads == []
        assert manager.create_calls == 0
    finally:
        await client.close()
