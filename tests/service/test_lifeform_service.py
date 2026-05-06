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


@pytest.fixture
async def alpha_client(aiohttp_client, tmp_path):
    from lifeform_service.alpha import AlphaServiceConfig
    from lifeform_service.app import create_app
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    app = create_app(
        vertical=spec,
        max_sessions=8,
        idle_eviction_seconds=None,
        alpha_config=AlphaServiceConfig(
            enabled=True,
            memory_scope_root_dir=str(tmp_path / "memory"),
            evidence_root_dir=str(tmp_path / "evidence"),
            alpha_users=frozenset({"alice", "bob"}),
        ),
    )
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


async def test_chat_ui_is_served(client):
    resp = await client.get("/chat")
    assert resp.status == 200
    assert resp.content_type == "text/html"
    text = await resp.text()
    assert "Volvence Zero Chat" in text
    assert "/v1/sessions" in text
    assert "dialogue-outcomes" in text


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


async def test_alpha_info_advertises_policy(alpha_client):
    resp = await alpha_client.get("/v1/info")
    assert resp.status == 200
    body = await resp.json()
    assert body["alpha"]["enabled"] is True
    assert "not therapy" in body["alpha"]["disclaimer"]


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


async def test_alpha_create_session_requires_identity(alpha_client):
    resp = await alpha_client.post("/v1/sessions", json={})
    assert resp.status == 400
    body = await resp.json()
    assert body["error"] == "missing_alpha_user"


async def test_alpha_create_session_binds_user(alpha_client):
    resp = await alpha_client.post(
        "/v1/sessions",
        headers={"X-Alpha-User": "alice"},
        json={"session_id": "alice-s1"},
    )
    assert resp.status == 201
    body = await resp.json()
    assert body["user_id"] == "alice"
    assert body["service_version"] == "closed-alpha-v0"


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


async def test_alpha_turn_returns_rationale_tags_and_safety(alpha_client):
    create = await (
        await alpha_client.post(
            "/v1/sessions",
            headers={"X-Alpha-User": "alice"},
            json={"session_id": "alpha-turn"},
        )
    ).json()
    sid = create["session_id"]
    resp = await alpha_client.post(
        f"/v1/sessions/{sid}/turns",
        json={"user_input": "I feel stuck and need a slower frame."},
    )
    assert resp.status == 200
    body = await resp.json()
    assert isinstance(body["response_rationale_tags"], list)
    assert "alpha_disclaimer" in body["safety"]


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


async def test_alpha_typed_feedback_repair_memory_and_delete(alpha_client):
    create = await (
        await alpha_client.post(
            "/v1/sessions",
            headers={"X-Alpha-User": "alice"},
            json={"session_id": "alpha-repair"},
        )
    ).json()
    sid = create["session_id"]
    await alpha_client.post(
        f"/v1/sessions/{sid}/turns",
        json={"user_input": "I am scared and do not want to be optimized."},
    )
    await alpha_client.post(
        f"/v1/sessions/{sid}/dialogue-outcomes",
        json={"kind": "OVER_DIRECTIVE", "confidence": 0.95},
    )
    repair = await (
        await alpha_client.post(
            f"/v1/sessions/{sid}/turns",
            json={"user_input": "That felt too procedural. Can we repair it?"},
        )
    ).json()
    assert "repair_alpha=over_directive" in repair["response_rationale_tags"]
    await alpha_client.post(
        f"/v1/sessions/{sid}/dialogue-outcomes",
        json={"kind": "OVER_DIRECTIVE", "confidence": 0.95},
    )
    await alpha_client.post(
        f"/v1/sessions/{sid}/dialogue-outcomes",
        json={"kind": "FELT_HEARD", "confidence": 0.9},
    )
    await alpha_client.post(
        f"/v1/sessions/{sid}/turns",
        json={"user_input": "That felt heard. Please remember the slower frame."},
    )
    ended = await (
        await alpha_client.post(f"/v1/sessions/{sid}/end-scene", json={})
    ).json()
    assert ended["evidence_artifact_ref"]

    memory = await (
        await alpha_client.get(
            "/v1/users/me/memory/rupture-repair",
            headers={"X-Alpha-User": "alice"},
        )
    ).json()
    assert any("repair_outcome:observed" in e["tags"] for e in memory["entries"])

    bob_memory = await (
        await alpha_client.get(
            "/v1/users/me/memory/rupture-repair",
            headers={"X-Alpha-User": "bob"},
        )
    ).json()
    assert bob_memory["entries"] == []

    summary = await (
        await alpha_client.get(
            "/v1/users/me/relationship-summary",
            headers={"X-Alpha-User": "alice"},
        )
    ).json()
    assert summary["observed_repair_count"] >= 1

    deleted = await (
        await alpha_client.delete(
            "/v1/users/me/memory",
            headers={"X-Alpha-User": "alice"},
        )
    ).json()
    assert deleted["deleted_entry_ids"]
    after = await (
        await alpha_client.get(
            "/v1/users/me/memory/rupture-repair",
            headers={"X-Alpha-User": "alice"},
        )
    ).json()
    assert after["entries"] == []


async def test_alpha_duplicate_feedback_clicks_do_not_break_next_turn(alpha_client):
    create = await (
        await alpha_client.post(
            "/v1/sessions",
            headers={"X-Alpha-User": "alice"},
            json={"session_id": "alpha-duplicate-feedback"},
        )
    ).json()
    sid = create["session_id"]
    await alpha_client.post(
        f"/v1/sessions/{sid}/turns",
        json={"user_input": "我想测试一下重复反馈。"},
    )
    for _ in range(2):
        resp = await alpha_client.post(
            f"/v1/sessions/{sid}/dialogue-outcomes",
            json={"kind": "MISSED", "confidence": 0.9},
        )
        assert resp.status == 201

    turn = await alpha_client.post(
        f"/v1/sessions/{sid}/turns",
        json={"user_input": "继续。"},
    )
    assert turn.status == 200


async def test_alpha_pause_and_admin_report(alpha_client):
    create = await (
        await alpha_client.post(
            "/v1/sessions",
            headers={"X-Alpha-User": "alice"},
            json={"session_id": "alpha-admin"},
        )
    ).json()
    sid = create["session_id"]
    paused = await (
        await alpha_client.post(f"/v1/sessions/{sid}/pause", json={})
    ).json()
    assert paused["paused"] is True
    report = await (
        await alpha_client.get(
            "/v1/admin/weekly-report",
            headers={"X-Alpha-User": "alice"},
        )
    ).json()
    assert report["active_user_count"] >= 1


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


def test_cli_can_require_alpha_preflight(monkeypatch):
    from lifeform_service import cli
    from lifeform_service.verticals import discover_verticals

    class _Report:
        passed = True

    called = {}

    def fake_preflight(*, artifacts_dir, scope_root_dir):
        called["artifacts_dir"] = artifacts_dir
        called["scope_root_dir"] = scope_root_dir
        return _Report()

    def fake_format(report):
        assert report.passed is True
        return "preflight ok"

    def fake_run_app(app, *, host, port, print):  # noqa: A002, ARG001
        called["host"] = host
        called["port"] = port

    import lifeform_evolution.closed_alpha_preflight as preflight

    monkeypatch.setattr(preflight, "run_closed_alpha_preflight", fake_preflight)
    monkeypatch.setattr(preflight, "format_closed_alpha_preflight_report", fake_format)
    monkeypatch.setattr(cli.web, "run_app", fake_run_app)

    assert "companion" in discover_verticals()
    exit_code = cli.main(
        [
            "--alpha-enabled",
            "--memory-scope-root-dir",
            "alpha-memory",
            "--evidence-root-dir",
            "alpha-evidence",
            "--require-alpha-preflight",
            "--idle-eviction-seconds",
            "0",
        ]
    )

    assert exit_code == 0
    assert called["artifacts_dir"] == "alpha-evidence/preflight"
    assert called["scope_root_dir"] == "alpha-evidence/preflight_scope"
