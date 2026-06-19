from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dlaas_platform_api.app import attach_dlaas_routes


@dataclass
class _FakeTimeNode:
    time_node_id: str = "tn_early"
    restore_status: str = "ready"

    def to_json(self) -> dict[str, Any]:
        return {
            "time_node_id": self.time_node_id,
            "ai_id": "ai_1",
            "scope_key": "family_1:subject_1",
            "source_session_id": "current",
            "as_of_ms": 1711929600000,
            "captured_at_ms": 1711929600123,
            "snapshot_version": "tn.v1",
            "restore_status": self.restore_status,
            "owner_slots": ["memory", "semantic_state"],
            "evidence": {
                "source_count": 2,
                "latest_source_captured_at_ms": 1711929600000,
                "watermark": "sha256:test",
            },
        }


class _FakeSession:
    latest_active_snapshots: dict[str, Any] = {}

    async def run_turn(self, brief: str, **_kwargs: Any):
        class _Response:
            text = f"reply:{brief}"
            rationale_tags: tuple[str, ...] = ()

        class _Result:
            response = _Response()
            active_regime = "task_focus"
            active_abstract_action = None

        return _Result()


class _FakeSessionManager:
    def __init__(self) -> None:
        self._sessions = {"current": _FakeSession()}
        self._readonly: set[str] = set()
        self.fork_calls: list[dict[str, Any]] = []

    async def list_time_nodes(self, **_kwargs: Any):
        return (_FakeTimeNode(),)

    async def get_time_node(self, time_node_id: str):
        return _FakeTimeNode(time_node_id=time_node_id)

    async def fork_session(self, **kwargs: Any):
        self.fork_calls.append(kwargs)
        self._sessions[kwargs["fork_session_id"]] = _FakeSession()
        self._readonly.add(kwargs["fork_session_id"])
        return {
            "status": "ok",
            "ai_id": "ai_1",
            "source_session_id": kwargs["source_session_id"],
            "fork_session_id": kwargs["fork_session_id"],
            "time_node_id": kwargs["time_node_id"],
            "snapshot_version": "tn.v1",
            "mode": kwargs["mode"],
        }

    async def get_session(self, session_id: str):
        from lifeform_service import SessionNotFoundError

        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(session_id) from exc

    async def create_session(self, *, session_id: str, user_id: str | None = None):
        self._sessions[session_id] = _FakeSession()
        return self._sessions[session_id]

    def session_end_user(self, session_id: str) -> str | None:
        return None

    def is_historical_readonly(self, session_id: str) -> bool:
        return session_id in self._readonly


def _build_app(manager: _FakeSessionManager | None = None) -> web.Application:
    app = web.Application()
    app["session_manager"] = manager or _FakeSessionManager()
    return attach_dlaas_routes(app)


async def _request(method: str, path: str, *, json: dict[str, Any] | None = None):
    client = TestClient(TestServer(_build_app()))
    await client.start_server()
    try:
        resp = await client.request(method, path, json=json)
        return resp.status, await resp.json()
    finally:
        await client.close()


async def test_time_nodes_fail_closed_when_flag_off(monkeypatch) -> None:
    monkeypatch.delenv("DLAAS_TEMPORAL_FORK", raising=False)
    status, body = await _request("GET", "/dlaas/v1/instances/ai_1/time-nodes")
    assert status == 503
    assert body["error"] == "temporal_fork_disabled"


async def test_shadow_lists_time_nodes(monkeypatch) -> None:
    monkeypatch.setenv("DLAAS_TEMPORAL_FORK", "shadow")
    status, body = await _request(
        "GET",
        "/dlaas/v1/instances/ai_1/time-nodes?session_id=current&scope_key=family_1:subject_1",
    )
    assert status == 200
    assert body["mode"] == "shadow"
    assert body["items"][0]["time_node_id"] == "tn_early"


async def test_shadow_rejects_fork(monkeypatch) -> None:
    monkeypatch.setenv("DLAAS_TEMPORAL_FORK", "shadow")
    status, body = await _request(
        "POST",
        "/dlaas/v1/instances/ai_1/sessions/fork",
        json={
            "source_session_id": "current",
            "fork_session_id": "fork_1",
            "time_node_id": "tn_early",
            "scope_key": "family_1:subject_1",
            "mode": "historical_readonly",
        },
    )
    assert status == 503
    assert body["error"] == "temporal_fork_disabled"


async def test_active_fork_creates_historical_session(monkeypatch) -> None:
    monkeypatch.setenv("DLAAS_TEMPORAL_FORK", "active")
    status, body = await _request(
        "POST",
        "/dlaas/v1/instances/ai_1/sessions/fork",
        json={
            "source_session_id": "current",
            "fork_session_id": "fork_1",
            "time_node_id": "tn_early",
            "scope_key": "family_1:subject_1",
            "mode": "historical_readonly",
            "metadata": {"moonlight.as_of_ms": 1711929600000},
        },
    )
    assert status == 201
    assert body["fork_session_id"] == "fork_1"
    assert body["mode"] == "historical_readonly"

