from __future__ import annotations

from aiohttp.test_utils import TestClient, TestServer

from dlaas_platform_api.app import attach_dlaas_routes
from dlaas_platform_launcher import INSTANCE_MANAGER_APP_KEY
from lifeform_service.app import create_app as create_lifeform_app
from lifeform_service.verticals import _try_coding


def _coding_payload() -> dict[str, object]:
    return {
        "request_id": "dlaas-coding-request",
        "project_id": "project-1",
        "repository_id": "repo-1",
        "task_id": "task-1",
        "task_kind": "bugfix",
        "task_summary": "Verify uniform DLaaS Brain forwarding",
        "repository_revision": "abc123",
        "target_paths": ["src/state.py"],
    }


async def test_dlaas_uniform_brain_route_dispatches_non_operations_vertical(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    vertical = _try_coding()
    assert vertical is not None
    app = attach_dlaas_routes(create_lifeform_app(vertical=vertical))
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        session = await client.post(
            "/dlaas/v1/instances/ai-coding/sessions",
            json={"session_id": "coding-session"},
        )
        assert session.status == 201
        assert (await session.json())["vertical"] == "coding"

        context = await client.post(
            "/dlaas/v1/instances/ai-coding/sessions/coding-session/brain/context-packs",
            json=_coding_payload(),
        )
        assert context.status == 201
        assert context.headers["X-Volvence-Brain"] == "coding"
        assert (await context.json())["schema_version"] == "coding-context-pack.v1"
    finally:
        await client.close()


class _RemoteBrainLauncher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def forward_interaction(self, *, ai_id, envelope):
        raise AssertionError((ai_id, envelope))

    async def forward_session_create(self, *, ai_id, payload):
        self.calls.append((ai_id, "session"))
        return 201, {
            "status": "created",
            "ai_id": ai_id,
            "session_id": payload["session_id"],
            "created": True,
        }

    async def forward_brain_request(
        self,
        *,
        ai_id,
        session_id,
        operation,
        payload,
    ):
        del payload
        self.calls.append((ai_id, operation))
        return 201, {
            "schema_version": "coding-context-pack.v1",
            "session_id": session_id,
            "routing": "remote",
        }


async def test_dlaas_uniform_brain_route_uses_remote_sticky_forwarder(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VZ_ATTACH_DEFAULT_MCP_BUNDLE", "0")
    vertical = _try_coding()
    assert vertical is not None
    launcher = _RemoteBrainLauncher()
    app = create_lifeform_app(vertical=vertical)
    app[INSTANCE_MANAGER_APP_KEY] = launcher
    attach_dlaas_routes(app)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        session = await client.post(
            "/dlaas/v1/instances/ai-remote/sessions",
            json={"session_id": "coding-remote"},
        )
        assert session.status == 201
        context = await client.post(
            "/dlaas/v1/instances/ai-remote/sessions/coding-remote/brain/context-packs",
            json=_coding_payload(),
        )
        assert context.status == 201
        assert (await context.json())["routing"] == "remote"
        assert launcher.calls == [
            ("ai-remote", "session"),
            ("ai-remote", "context-packs"),
        ]
    finally:
        await client.close()
