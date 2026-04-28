"""End-to-end tests for the lifeform HTTP service.

We use ``aiohttp.test_utils.TestClient`` so the tests bind to an
ephemeral port and exercise the real ASGI plumbing without any mocking
of the kernel or the lifeform layer. The vertical under test is the
companion vertical with its full pre-trained bootstraps.

What's covered:

* Health + info routes return the configured vertical and bootstrap flags.
* ``POST /v1/sessions`` mints a session id; explicit ids are honoured.
* Invalid bodies produce structured 400 errors (JSON, not HTML).
* ``POST /v1/sessions/{id}/turns`` runs a turn and returns the kernel's
  active regime / abstract action / response text.
* ``POST /v1/sessions/{id}/end-scene`` closes the open scene and reports
  whether the slow loop drained.
* ``DELETE /v1/sessions/{id}`` works once and 404s on the second call.
* Multi-tenant isolation: two different session ids produce two
  independent ``LifeformSession``s with independent turn counts.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def event_loop():
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client(aiohttp_client):
    from lifeform_service.app import create_app
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    app = create_app(vertical=spec, max_sessions=8, idle_eviction_seconds=None)
    return await aiohttp_client(app)


# ---------------------------------------------------------------------------
# Static surfaces
# ---------------------------------------------------------------------------


async def test_health_reports_vertical_and_session_count(client):
    resp = await client.get("/v1/health")
    assert resp.status == 200
    body = await resp.json()
    assert body["status"] == "ok"
    assert body["vertical"] == "companion"
    assert body["session_count"] == 0


async def test_info_advertises_pre_trained_bootstraps(client):
    resp = await client.get("/v1/info")
    assert resp.status == 200
    body = await resp.json()
    assert body["vertical"] == "companion"
    # Companion vertical ships both pre-trained bootstraps.
    assert body["has_temporal_bootstrap"] is True
    assert body["has_regime_bootstrap"] is True
    assert body["bootstraps_dir"] is not None
    assert body["scenarios_dir"] is not None


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


async def test_create_session_mints_id_and_201s(client):
    resp = await client.post("/v1/sessions", json={})
    assert resp.status == 201
    body = await resp.json()
    assert body["session_id"].startswith("sess-")
    assert body["vertical"] == "companion"
    # Health now sees one session.
    health = await (await client.get("/v1/health")).json()
    assert health["session_count"] == 1


async def test_create_session_honours_explicit_id(client):
    resp = await client.post(
        "/v1/sessions", json={"session_id": "tenant-42"}
    )
    body = await resp.json()
    assert resp.status == 201
    assert body["session_id"] == "tenant-42"


async def test_create_session_409s_on_duplicate_explicit_id(client):
    await client.post("/v1/sessions", json={"session_id": "dup"})
    resp = await client.post("/v1/sessions", json={"session_id": "dup"})
    assert resp.status == 409
    body = await resp.json()
    assert body["error"] == "session_already_exists"


async def test_close_session_works_then_404s(client):
    create = await (await client.post("/v1/sessions", json={})).json()
    sid = create["session_id"]
    resp1 = await client.delete(f"/v1/sessions/{sid}")
    assert resp1.status == 200
    resp2 = await client.delete(f"/v1/sessions/{sid}")
    assert resp2.status == 404
    body = await resp2.json()
    assert body["error"] == "session_not_found"


# ---------------------------------------------------------------------------
# Turn lifecycle
# ---------------------------------------------------------------------------


async def test_turn_runs_and_returns_kernel_state(client):
    create = await (await client.post("/v1/sessions", json={})).json()
    sid = create["session_id"]

    turn = await (
        await client.post(
            f"/v1/sessions/{sid}/turns",
            json={"user_input": "I have been feeling stuck lately."},
        )
    ).json()
    assert turn["session_id"] == sid
    assert turn["turn_index"] == 1
    assert turn["response_text"].strip() != ""
    assert turn["active_regime"]
    assert turn["scene_id"].startswith("scene-")


async def test_turn_404s_on_unknown_session(client):
    resp = await client.post(
        "/v1/sessions/no-such/turns", json={"user_input": "hi"}
    )
    assert resp.status == 404


async def test_turn_400s_on_missing_user_input(client):
    create = await (await client.post("/v1/sessions", json={})).json()
    sid = create["session_id"]
    resp = await client.post(f"/v1/sessions/{sid}/turns", json={})
    assert resp.status == 400
    body = await resp.json()
    assert body["error"] == "invalid_user_input"


async def test_turn_400s_on_invalid_json(client):
    create = await (await client.post("/v1/sessions", json={})).json()
    sid = create["session_id"]
    resp = await client.post(
        f"/v1/sessions/{sid}/turns",
        data="not-json-at-all",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["error"] == "invalid_json"


# ---------------------------------------------------------------------------
# End-scene
# ---------------------------------------------------------------------------


async def test_end_scene_closes_and_drains(client):
    create = await (await client.post("/v1/sessions", json={})).json()
    sid = create["session_id"]
    await client.post(
        f"/v1/sessions/{sid}/turns", json={"user_input": "Hi just checking in."}
    )
    resp = await client.post(f"/v1/sessions/{sid}/end-scene", json={})
    assert resp.status == 200
    body = await resp.json()
    assert body["closed_scene_id"] is not None
    assert body["slow_loop_drained"] is True


# ---------------------------------------------------------------------------
# State + multi-tenant isolation
# ---------------------------------------------------------------------------


async def test_session_state_reports_turn_progress(client):
    create = await (await client.post("/v1/sessions", json={})).json()
    sid = create["session_id"]
    await client.post(
        f"/v1/sessions/{sid}/turns", json={"user_input": "Hello first turn."}
    )
    state = await (await client.get(f"/v1/sessions/{sid}/state")).json()
    assert state["turn_count"] == 1
    assert state["open_scene_turn_count"] == 1
    assert state["last_active_regime"]


async def test_two_sessions_are_independent_tenants(client):
    a = await (await client.post("/v1/sessions", json={"session_id": "a"})).json()
    b = await (await client.post("/v1/sessions", json={"session_id": "b"})).json()
    assert a["session_id"] != b["session_id"]
    await client.post(
        "/v1/sessions/a/turns", json={"user_input": "I am tenant A."}
    )
    state_a = await (await client.get("/v1/sessions/a/state")).json()
    state_b = await (await client.get("/v1/sessions/b/state")).json()
    assert state_a["turn_count"] == 1
    assert state_b["turn_count"] == 0
