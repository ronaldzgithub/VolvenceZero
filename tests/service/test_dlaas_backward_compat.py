"""Slice 7.4 — backward compatibility + perf smoke.

These tests guarantee that adding the DLaaS platform tier never
breaks the legacy ``/v1/sessions/...`` API. The promise made to
existing integrators (see ``docs/specs/dlaas-platform.md``
"WiringLevel 迁移路径") is that the old endpoints stay ACTIVE as
the canonical path until Slice 7 is fully signed off.

Two paths are exercised:

1. ``attach_dlaas_routes`` (Slice 1 minimal mode) — single shared
   SessionManager, no registry, no launcher. The DLaaS endpoint
   accepts any ``ai_id`` because it falls back to the underlying
   SessionManager. Old ``/v1/...`` keeps working.
2. ``build_dlaas_app`` (full stack mode) — registry + launcher +
   ops + eval. Old ``/v1/...`` continues to operate against the
   same SessionManager that powers Slice 1's fallback path; new
   DLaaS calls flow through the launcher.

The perf smoke verifies the dispatch overhead is in the
sub-second range on the synthetic substrate; we don't pin a
specific latency floor here because that is environment-sensitive,
but we lock down "10 chat turns finish under 60s" as a smoke
ceiling so a regression that stalls dispatch (e.g. a missing
``await``) gets caught.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest


CONTROL_PLANE_SECRET = "cp_secret_compat"


@pytest.fixture
async def slice1_client(aiohttp_client):
    """Slice 1 mode: ``attach_dlaas_routes`` only, no registry / launcher."""
    from dlaas_platform_api import attach_dlaas_routes
    from lifeform_service.app import create_app
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    app = create_app(vertical=spec, max_sessions=4, idle_eviction_seconds=None)
    attach_dlaas_routes(app, default_ai_id="ai_compat")
    return await aiohttp_client(app)


@pytest.fixture
async def fullstack_client(aiohttp_client, tmp_path: Path):
    """Slice 3+ mode: full stack via ``build_dlaas_app``."""
    from dlaas_platform_api import build_dlaas_app
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    app = build_dlaas_app(
        db_path=str(tmp_path / "compat.sqlite"),
        control_plane_secret=CONTROL_PLANE_SECRET,
        vertical=spec,
        max_sessions=4,
        idle_eviction_seconds=None,
    )
    return await aiohttp_client(app)


# ---------------------------------------------------------------------------
# Slice 1 mode: legacy + new endpoint coexistence
# ---------------------------------------------------------------------------


async def test_slice1_legacy_v1_endpoints_still_work(slice1_client):
    """``GET /v1/info`` must keep returning a healthy info payload."""
    resp = await slice1_client.get("/v1/info")
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert "vertical" in body or "verticals" in body or "name" in body


async def test_slice1_legacy_session_create_still_works(slice1_client):
    """``POST /v1/sessions`` continues to mint legacy sessions."""
    resp = await slice1_client.post("/v1/sessions", json={})
    assert resp.status in (200, 201), await resp.text()
    body = await resp.json()
    assert "session_id" in body
    sid = body["session_id"]
    # And legacy turn dispatch still reaches the kernel.
    resp = await slice1_client.post(
        f"/v1/sessions/{sid}/turns",
        json={"user_input": "你好"},
    )
    assert resp.status == 200, await resp.text()


async def test_slice1_dlaas_endpoint_routes_to_default_session_manager(
    slice1_client,
):
    """In Slice 1 mode every ``ai_id`` is served by the single
    fallback SessionManager — there is no launcher binding."""
    resp = await slice1_client.post(
        "/dlaas/instances/anything_goes/interactions",
        json={
            "contract_id": "ctr_compat",
            "session_id": "sess_slice1_compat",
            "end_user_ref": "user_compat",
            "interaction_type": "chat",
            "human_brief": "你好",
            "lang": "cn",
        },
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["status"] == "ok"
    assert body["ai_id"] == "anything_goes"


# ---------------------------------------------------------------------------
# Full stack mode: launcher gates ai_id, but legacy endpoints stay alive
# ---------------------------------------------------------------------------


async def test_fullstack_legacy_v1_still_alive(fullstack_client):
    resp = await fullstack_client.get("/v1/info")
    assert resp.status == 200, await resp.text()


async def test_fullstack_dlaas_rejects_unadopted_ai_id(fullstack_client):
    """In full-stack mode the launcher REQUIRES adoption — calls
    against an unknown ``ai_id`` get a typed 404, not a fallthrough
    to the shared SessionManager."""
    resp = await fullstack_client.post(
        "/dlaas/instances/never_adopted/interactions",
        json={
            "contract_id": "ctr_x",
            "session_id": "sess_x",
            "end_user_ref": "u",
            "interaction_type": "chat",
            "human_brief": "你好",
        },
    )
    assert resp.status == 404, await resp.text()
    payload = await resp.json()
    assert payload["error"] == "ai_id_not_found"


async def test_fullstack_legacy_v1_session_still_works(fullstack_client):
    """Old direct ``POST /v1/sessions`` keeps working even when the
    full DLaaS stack is mounted — they share the same fallback
    SessionManager but the contract surface is unchanged."""
    resp = await fullstack_client.post("/v1/sessions", json={})
    assert resp.status in (200, 201), await resp.text()
    body = await resp.json()
    sid = body["session_id"]
    resp = await fullstack_client.post(
        f"/v1/sessions/{sid}/turns",
        json={"user_input": "继续"},
    )
    assert resp.status == 200, await resp.text()


# ---------------------------------------------------------------------------
# Perf smoke (loose ceiling, no latency pinning)
# ---------------------------------------------------------------------------


async def test_dispatch_overhead_smoke(slice1_client):
    """Run 10 chat turns sequentially and verify total wallclock time
    stays below a generous ceiling.

    The number is deliberately loose (60s for 10 turns) so the test
    never flakes on slow CI; it only catches "dispatch added an
    accidental sleep / blocking call" regressions.
    """
    body_template = {
        "contract_id": "ctr_perf",
        "session_id": "sess_perf",
        "end_user_ref": "user_perf",
        "interaction_type": "chat",
        "lang": "cn",
    }
    started = time.monotonic()
    for i in range(10):
        resp = await slice1_client.post(
            "/dlaas/instances/ai_compat/interactions",
            json={**body_template, "human_brief": f"turn {i}"},
        )
        assert resp.status == 200
    elapsed = time.monotonic() - started
    assert elapsed < 60.0, (
        f"10 chat turns took {elapsed:.2f}s — dispatch overhead regressed."
    )


async def test_concurrent_dispatch_smoke(slice1_client):
    """Five concurrent chat turns to distinct sessions complete OK.

    The synthetic substrate runtime is sync under the hood (the
    SessionManager docstring warns about it) but aiohttp queues
    coroutines on a single event loop; this smoke makes sure the
    dispatch handler does not deadlock under interleaved
    ``run_turn`` calls.
    """

    async def one_turn(idx: int) -> int:
        resp = await slice1_client.post(
            "/dlaas/instances/ai_compat/interactions",
            json={
                "contract_id": "ctr_conc",
                "session_id": f"sess_conc_{idx}",
                "end_user_ref": f"user_conc_{idx}",
                "interaction_type": "chat",
                "human_brief": f"hello {idx}",
                "lang": "cn",
            },
        )
        return resp.status

    statuses = await asyncio.gather(*(one_turn(i) for i in range(5)))
    assert all(s == 200 for s in statuses), statuses
