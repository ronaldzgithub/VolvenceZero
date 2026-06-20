"""End-to-end smoke for the multi-angle bake plane.

Exercises submit -> monitor (status + SSE) -> result with the synthetic
runner, plus cancellation and tenant/operator auth boundaries. No GPU
or figure/character compiler required.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dlaas_platform_api.bake import (
    SyntheticBakeAngleRunner,
    attach_bake_routes,
)
from dlaas_platform_registry import (
    ApplicationStore,
    PlatformAuthBundle,
    PlatformAuthConfig,
    REGISTRY_APP_KEY,
    Registry,
    TenantStore,
)

_SECRET = "test-control-plane-secret"
_OP_HEADERS = {"X-Control-Plane-Secret": _SECRET}


def _build_app() -> web.Application:
    registry = Registry(db_path=":memory:")
    app = web.Application()
    app[REGISTRY_APP_KEY] = PlatformAuthBundle(
        tenant_store=TenantStore(registry),
        auth_config=PlatformAuthConfig(control_plane_secret=_SECRET),
        application_store=ApplicationStore(registry),
    )
    # Pin the GPU-free synthetic runner explicitly: the production default
    # is now third_party_llm, which would fail loud here without a provider.
    attach_bake_routes(
        app, registry=registry, runner=SyntheticBakeAngleRunner()
    )
    return app


def _bake_body() -> dict:
    return {
        "source_ref": "work:dream-of-red-chamber",
        "corpus_mode": "curated",
        "angles": [
            {"kind": "author", "slug": "caoxueqin", "display_name": "曹雪芹"},
            {"kind": "interpreter", "slug": "narrator"},
            {"kind": "character", "slug": "jiabaoyu", "display_name": "贾宝玉"},
        ],
        "raw_materials": [
            {"kind": "text", "text": "第一回 甄士隐梦幻识通灵 ...", "angle_slugs": ["narrator"]},
            {"kind": "uri", "ref": "https://example.test/hlm.txt"},
        ],
    }


async def _wait_terminal(client: TestClient, run_id: str, *, timeout: float = 5.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(f"/dlaas/v1/bake/{run_id}", headers=_OP_HEADERS)
        body = await resp.json()
        assert body["status"] == "ok", body
        run = body["run"]
        if run["status"] in ("done", "failed", "partial", "cancelled"):
            return run
        await asyncio.sleep(0.05)
    raise AssertionError("bake run did not reach terminal state in time")


@pytest.mark.asyncio
async def test_bake_submit_monitor_result_synthetic() -> None:
    app = _build_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/dlaas/v1/bake", json=_bake_body(), headers=_OP_HEADERS
        )
        assert resp.status == 202, await resp.text()
        submitted = await resp.json()
        run_id = submitted["run_id"]
        assert len(submitted["run"]["angles"]) == 3

        terminal = await _wait_terminal(client, run_id)
        assert terminal["status"] == "done"
        statuses = {a["angle"]["kind"]: a["status"] for a in terminal["angles"]}
        assert statuses == {
            "author": "done",
            "interpreter": "done",
            "character": "done",
        }

        result = await client.get(
            f"/dlaas/v1/bake/{run_id}/result", headers=_OP_HEADERS
        )
        result_body = await result.json()
        assert result_body["run_status"] == "done"
        templates = result_body["templates"]
        assert len(templates) == 3
        # author/interpreter route to the figure family (bundle id set),
        # character routes to the character family (no figure artifact).
        by_kind = {t["angle_kind"]: t for t in templates}
        assert by_kind["author"]["figure_artifact_id"].startswith("figure-bundle:")
        assert by_kind["character"]["figure_artifact_id"] == ""
        assert by_kind["character"]["bundle_id"].startswith("character-template:")
        assert all(t["template_id"] for t in templates)
        assert all(t["lifecycle_stage"] == "pretrained" for t in templates)


@pytest.mark.asyncio
async def test_bake_sse_stream_emits_progress() -> None:
    app = _build_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/dlaas/v1/bake", json=_bake_body(), headers=_OP_HEADERS
        )
        submitted = await resp.json()
        run_id = submitted["run_id"]
        stream = await client.get(
            f"/dlaas/v1/bake/{run_id}/events", headers=_OP_HEADERS
        )
        assert stream.status == 200
        seen_done = False
        kinds = []
        # Read until the stream-end comment or EOF.
        async for raw in stream.content:
            line = raw.decode("utf-8")
            if line.startswith("event:"):
                kinds.append(line.split(":", 1)[1].strip())
            if line.startswith(": stream-end"):
                seen_done = True
                break
        assert "run" in kinds
        assert "angle" in kinds
        assert "done" in kinds
        assert seen_done


@pytest.mark.asyncio
async def test_bake_rejects_invalid_request() -> None:
    app = _build_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/dlaas/v1/bake",
            json={"source_ref": "x", "angles": []},
            headers=_OP_HEADERS,
        )
        assert resp.status == 400
        body = await resp.json()
        assert body["error"] == "invalid_bake_request"


@pytest.mark.asyncio
async def test_bake_requires_auth() -> None:
    app = _build_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/dlaas/v1/bake", json=_bake_body())
        assert resp.status in (401, 403)
