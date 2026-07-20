"""HTTP/SSE integration for the digital-ant app."""

from __future__ import annotations

import json
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from volvence_ant.app.server import create_app


async def test_app_api_runs_one_real_fixed_rule_tick_and_exports_replay() -> None:
    client = TestClient(TestServer(create_app()))
    await client.start_server()
    try:
        created = await client.post(
            "/api/v1/runs",
            json={
                "arm": "fixed_rule",
                "autostart": False,
                "max_ticks": 3,
                "tick_interval_ms": 0,
            },
        )
        assert created.status == 201
        run_id = (await created.json())["run_id"]

        disturbance = await client.post(
            f"/api/v1/runs/{run_id}/disturbances",
            json={
                "kind": "relocate_food",
                "x": -3.0,
                "y": 2.0,
            },
        )
        assert disturbance.status == 202
        command = await client.post(
            f"/api/v1/runs/{run_id}/commands",
            json={"kind": "step"},
        )
        assert command.status == 200

        events = await client.get(f"/api/v1/runs/{run_id}/events")
        first_chunk = await events.content.readuntil(b"\n\n")
        assert b"event:" in first_chunk
        events.close()

        replay_response = await client.get(f"/api/v1/runs/{run_id}/replay")
        replay = await replay_response.json()
        assert replay["schema_version"] == "digital-ant-app.v1"
        audit_payloads = [
            event["payload"]
            for event in replay["audit_events"]
            if event["kind"] == "disturbance"
        ]
        assert any(payload["status"] == "queued" for payload in audit_payloads)
    finally:
        await client.close()


async def test_sse_stream_contains_authoritative_frame_and_terminal_status() -> None:
    client = TestClient(TestServer(create_app()))
    await client.start_server()
    try:
        response = await client.post(
            "/api/v1/runs",
            json={
                "arm": "fixed_rule",
                "autostart": True,
                "max_ticks": 1,
                "tick_interval_ms": 0,
            },
        )
        run_id = (await response.json())["run_id"]
        stream = await client.get(f"/api/v1/runs/{run_id}/events")
        body = await stream.text()
        assert "event: frame" in body
        assert "event: status" in body
        frame_lines = [
            line for line in body.splitlines() if line.startswith("data: ")
        ]
        payloads = [json.loads(line.removeprefix("data: ")) for line in frame_lines]
        assert any(payload.get("tick") == 1 for payload in payloads)
    finally:
        await client.close()


async def test_api_rejects_direct_motor_command_shape() -> None:
    client = TestClient(TestServer(create_app()))
    await client.start_server()
    try:
        response = await client.post(
            "/api/v1/runs",
            json={"arm": "fixed_rule", "autostart": False},
        )
        run_id = (await response.json())["run_id"]
        rejected = await client.post(
            f"/api/v1/runs/{run_id}/commands",
            json={"kind": "pause", "turn_command": 0.5},
        )
        assert rejected.status == 400
    finally:
        await client.close()


async def test_server_hosts_vite_production_build_when_present(
    tmp_path: Path,
) -> None:
    (tmp_path / "index.html").write_text(
        '<html><body><div id="root"></div></body></html>',
        encoding="utf-8",
    )
    client = TestClient(TestServer(create_app(web_root=tmp_path)))
    await client.start_server()
    try:
        response = await client.get("/")
        assert response.status == 200
        html = await response.text()
        assert '<div id="root"></div>' in html
    finally:
        await client.close()
