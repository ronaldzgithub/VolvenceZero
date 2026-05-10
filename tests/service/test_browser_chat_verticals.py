"""Browser-chat multi-vertical surface — e2e contract tests.

Covers the new HTTP shape that lets the chat UI pick a vertical
per-session instead of pinning to one at service startup:

* ``GET /v1/verticals`` lists every registered vertical with
  capability flags (``alpha_supported`` / ``templates_supported``).
* ``POST /v1/sessions {"vertical": "..."}`` mints a session bound
  to the specified vertical; the session response echoes the
  bound vertical name.
* Unknown vertical name → 422 ``unknown_vertical``.
* Alpha mode + vertical without ``alpha_factory`` → 422
  ``vertical_not_alpha_capable``.
* ``GET /v1/templates?vertical=foo`` filters by vertical; bad
  ``vertical`` → 422.
* Save-as-template lands in the *session's* vertical subdir even
  when the chat UI's default vertical is different.

DLaaS back-compat is verified separately by
``tests/service/test_dlaas_full_lifecycle.py``; this file
exercises only the chat-browser lane.
"""

from __future__ import annotations

import asyncio
import pathlib

import pytest


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def all_verticals():
    from lifeform_service.verticals import discover_verticals

    return discover_verticals()


@pytest.fixture
async def multi_client(aiohttp_client, all_verticals, tmp_path):
    """Browser-chat-style fixture: registry + templates root."""
    from lifeform_service.app import create_app

    if "companion" not in all_verticals:
        pytest.skip("companion vertical not installed")
    app = create_app(
        verticals=all_verticals,
        default_vertical="companion",
        max_sessions=8,
        idle_eviction_seconds=None,
        templates_root_dir=str(tmp_path / "templates"),
    )
    return await aiohttp_client(app)


@pytest.fixture
async def alpha_multi_client(aiohttp_client, all_verticals, tmp_path):
    from lifeform_service.alpha import AlphaServiceConfig
    from lifeform_service.app import create_app

    if "companion" not in all_verticals:
        pytest.skip("companion vertical not installed")
    app = create_app(
        verticals=all_verticals,
        default_vertical="companion",
        max_sessions=8,
        idle_eviction_seconds=None,
        templates_root_dir=str(tmp_path / "templates"),
        alpha_config=AlphaServiceConfig(
            enabled=True,
            memory_scope_root_dir=str(tmp_path / "memory"),
            evidence_root_dir=str(tmp_path / "evidence"),
            alpha_users=frozenset({"alice"}),
        ),
    )
    return await aiohttp_client(app)


# ---------------------------------------------------------------------------
# GET /v1/verticals
# ---------------------------------------------------------------------------


async def test_list_verticals_lists_discovered_with_capability_flags(
    multi_client, all_verticals
):
    body = await (await multi_client.get("/v1/verticals")).json()
    assert body["default_vertical"] == "companion"
    assert body["alpha_enabled"] is False
    by_name = {item["name"]: item for item in body["verticals"]}
    assert set(by_name.keys()) == set(all_verticals.keys())
    assert by_name["companion"]["is_default"] is True
    # Companion has alpha_factory but no template_adapter.
    assert by_name["companion"]["alpha_supported"] is True
    assert by_name["companion"]["templates_supported"] is False
    # Zhang_wuji is template-aware (registered in our prior wave).
    if "zhang_wuji" in by_name:
        assert by_name["zhang_wuji"]["templates_supported"] is True


async def test_list_verticals_reports_alpha_enabled(alpha_multi_client):
    body = await (await alpha_multi_client.get("/v1/verticals")).json()
    assert body["alpha_enabled"] is True


# ---------------------------------------------------------------------------
# POST /v1/sessions {"vertical": "..."}
# ---------------------------------------------------------------------------


async def test_create_session_uses_default_vertical_when_unspecified(
    multi_client,
):
    resp = await multi_client.post("/v1/sessions", json={})
    assert resp.status == 201
    body = await resp.json()
    assert body["vertical"] == "companion"


async def test_create_session_uses_explicit_vertical(multi_client, all_verticals):
    if "zhang_wuji" not in all_verticals:
        pytest.skip("zhang_wuji vertical not installed")
    resp = await multi_client.post(
        "/v1/sessions", json={"vertical": "zhang_wuji"}
    )
    assert resp.status == 201
    body = await resp.json()
    assert body["vertical"] == "zhang_wuji"


async def test_create_session_422s_on_unknown_vertical(multi_client):
    resp = await multi_client.post(
        "/v1/sessions", json={"vertical": "no-such-vertical"}
    )
    assert resp.status == 422
    body = await resp.json()
    assert body["error"] == "unknown_vertical"


async def test_create_session_422s_on_non_alpha_vertical_in_alpha_mode(
    alpha_multi_client, all_verticals
):
    """Alpha mode + vertical without alpha_factory → 422.

    ``coding`` and ``growth_advisor`` are the verticals in the
    discovered set that don't ship an ``alpha_factory``. We pick
    whichever one exists; if neither is installed we skip.
    """
    candidate = next(
        (
            name
            for name, spec in all_verticals.items()
            if spec.alpha_factory is None
        ),
        None,
    )
    if candidate is None:
        pytest.skip("no non-alpha-capable vertical in this install")
    resp = await alpha_multi_client.post(
        "/v1/sessions",
        headers={"X-Alpha-User": "alice"},
        json={"vertical": candidate},
    )
    assert resp.status == 422, await resp.text()
    body = await resp.json()
    assert body["error"] == "vertical_not_alpha_capable"


# ---------------------------------------------------------------------------
# GET /v1/templates?vertical=...
# ---------------------------------------------------------------------------


async def test_templates_default_to_default_vertical_when_query_unset(
    multi_client,
):
    """Companion has no template adapter so the default-vertical
    response surfaces ``templates_supported=False``."""
    body = await (await multi_client.get("/v1/templates")).json()
    assert body["vertical"] == "companion"
    assert body["templates_supported"] is False
    assert body["templates"] == []


async def test_templates_filter_by_vertical_query(multi_client, all_verticals):
    if "zhang_wuji" not in all_verticals:
        pytest.skip("zhang_wuji vertical not installed")
    body = await (
        await multi_client.get("/v1/templates?vertical=zhang_wuji")
    ).json()
    assert body["vertical"] == "zhang_wuji"
    # Empty templates dir → supported=True but empty list.
    assert body["templates_supported"] is True
    assert body["templates"] == []


async def test_templates_422s_on_unknown_vertical_query(multi_client):
    resp = await multi_client.get("/v1/templates?vertical=no-such")
    assert resp.status == 422
    body = await resp.json()
    assert body["error"] == "unknown_vertical"


# ---------------------------------------------------------------------------
# Save-as-template uses session's vertical
# ---------------------------------------------------------------------------


async def test_save_lands_in_session_vertical_subdir(
    multi_client, all_verticals, tmp_path
):
    """Even though the service default is ``companion``, a session
    created with ``vertical=zhang_wuji`` must save to
    ``<root>/zhang_wuji/...``."""
    if "zhang_wuji" not in all_verticals:
        pytest.skip("zhang_wuji vertical not installed")
    create = await (
        await multi_client.post(
            "/v1/sessions", json={"vertical": "zhang_wuji"}
        )
    ).json()
    sid = create["session_id"]
    assert create["vertical"] == "zhang_wuji"
    # Drive a turn so the session has non-trivial state.
    await multi_client.post(
        f"/v1/sessions/{sid}/turns",
        json={"user_input": "测试输入"},
    )
    resp = await multi_client.post(
        f"/v1/sessions/{sid}/save-as-template",
        json={
            "template_id": "session-bound-save",
            "replay_provenance": "vertical-binding test",
        },
    )
    assert resp.status == 201, await resp.text()
    saved = (await resp.json())["saved"]
    expected_dir = pathlib.Path(saved["file_path"]).parent
    assert expected_dir.name == "zhang_wuji", expected_dir
    # Listing zhang_wuji now contains the new template.
    listing = await (
        await multi_client.get("/v1/templates?vertical=zhang_wuji")
    ).json()
    template_ids = {item["template_id"] for item in listing["templates"]}
    assert "session-bound-save" in template_ids


# ---------------------------------------------------------------------------
# create_app validation
# ---------------------------------------------------------------------------


def test_create_app_rejects_both_vertical_and_verticals(all_verticals):
    from lifeform_service.app import create_app

    if "companion" not in all_verticals:
        pytest.skip("companion vertical not installed")
    with pytest.raises(ValueError, match="not both"):
        create_app(
            vertical=all_verticals["companion"],
            verticals=all_verticals,
            default_vertical="companion",
            idle_eviction_seconds=None,
        )


def test_create_app_requires_default_when_verticals_supplied(all_verticals):
    from lifeform_service.app import create_app

    if "companion" not in all_verticals:
        pytest.skip("companion vertical not installed")
    with pytest.raises(ValueError, match="default_vertical"):
        create_app(
            verticals=all_verticals,
            idle_eviction_seconds=None,
        )


def test_create_app_requires_known_default_in_alpha_mode(all_verticals):
    """Default vertical must be alpha-capable when alpha is on."""
    from lifeform_service.alpha import AlphaServiceConfig
    from lifeform_service.app import create_app

    candidate = next(
        (
            name
            for name, spec in all_verticals.items()
            if spec.alpha_factory is None
        ),
        None,
    )
    if candidate is None:
        pytest.skip("no non-alpha-capable vertical in this install")
    with pytest.raises(ValueError, match="alpha"):
        create_app(
            verticals=all_verticals,
            default_vertical=candidate,
            alpha_config=AlphaServiceConfig(enabled=True, memory_scope_root_dir="/tmp/x"),
            idle_eviction_seconds=None,
        )
