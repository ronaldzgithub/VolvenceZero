"""Browser-chat template surface — e2e contract tests.

Covers the three new HTTP routes (``GET /v1/templates``,
``POST /v1/sessions`` with ``template_id``,
``POST /v1/sessions/{id}/save-as-template``) plus negative-path
behavior for verticals without a template adapter and invalid
``template_id`` shapes.

The character vertical's :class:`CharacterTemplateAdapter` is used as
the concrete adapter under test. We seed a realistic
:class:`LifeformTemplate` on disk via the canonical
``save_lifeform_template`` helper (same path the
``examples/train_zhang_wuji_template.py`` script uses) so the listing
+ load + save round-trip exercises the full schema, not a mock.

DLaaS is intentionally **not** part of these tests — this surface is
the local browser-chat lane only.
"""

from __future__ import annotations

import pathlib

import pytest

from lifeform_domain_character import (
    build_zhang_wuji_lifeform,
    save_lifeform_template,
    vitals_drive_levels_from_session,
)
from lifeform_service.app import create_app
from lifeform_service.verticals import discover_verticals
from volvence_zero.memory import build_default_memory_store


@pytest.fixture
def event_loop():
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


async def _seed_zhang_wuji_template(
    *, root_dir: pathlib.Path, template_id: str
) -> pathlib.Path:
    """Write a real LifeformTemplate JSON for the vertical+id combo.

    The file is placed under ``<root_dir>/zhang_wuji/`` so it matches
    the layout :func:`lifeform_service.app._resolve_templates_root`
    produces for the zhang_wuji vertical. We drive one turn so the
    saved template has a non-trivial vitals/memory snapshot, mirroring
    what ``examples/train_zhang_wuji_template.py`` produces in
    miniature.
    """
    subdir = root_dir / "zhang_wuji"
    subdir.mkdir(parents=True, exist_ok=True)
    memory_store = build_default_memory_store()
    bundle = build_zhang_wuji_lifeform(memory_store=memory_store)
    session = bundle.lifeform.create_session(session_id=f"seed-{template_id}")
    await session.run_turn("测试输入用于落档。")
    save_lifeform_template(
        profile=bundle.profile,
        template_id=template_id,
        output_dir=subdir,
        memory_store=memory_store,
        vitals_drive_levels=vitals_drive_levels_from_session(session),
        replay_provenance="browser-chat-test fixture",
    )
    return subdir / f"{template_id}.json"


# ---------------------------------------------------------------------------
# Vertical without adapter — companion (no template support)
# ---------------------------------------------------------------------------


@pytest.fixture
async def companion_client(aiohttp_client, tmp_path):
    spec = discover_verticals()["companion"]
    app = create_app(
        vertical=spec,
        max_sessions=4,
        idle_eviction_seconds=None,
        templates_root_dir=str(tmp_path / "templates"),
    )
    return await aiohttp_client(app)


async def test_list_templates_returns_unsupported_for_no_adapter_vertical(
    companion_client,
):
    resp = await companion_client.get("/v1/templates")
    assert resp.status == 200
    body = await resp.json()
    assert body["vertical"] == "companion"
    assert body["templates_supported"] is False
    assert body["templates"] == []


async def test_create_session_with_template_id_503s_when_unsupported(
    companion_client,
):
    resp = await companion_client.post(
        "/v1/sessions", json={"template_id": "anything"}
    )
    assert resp.status == 503
    body = await resp.json()
    assert body["error"] == "templates_not_supported"


async def test_save_as_template_503s_when_unsupported(companion_client):
    create = await (
        await companion_client.post("/v1/sessions", json={})
    ).json()
    sid = create["session_id"]
    resp = await companion_client.post(
        f"/v1/sessions/{sid}/save-as-template",
        json={"template_id": "any"},
    )
    assert resp.status == 503
    body = await resp.json()
    assert body["error"] == "templates_not_supported"


# ---------------------------------------------------------------------------
# Vertical with adapter — zhang_wuji (full template lifecycle)
# ---------------------------------------------------------------------------


@pytest.fixture
async def zhang_wuji_client(aiohttp_client, tmp_path):
    """Service fixture wired to the zhang_wuji vertical with a fresh
    on-disk template (``demo``) seeded under the templates root.
    """
    spec = discover_verticals().get("zhang_wuji")
    if spec is None:
        pytest.skip("lifeform-domain-character not installed")
    templates_root = tmp_path / "templates"
    await _seed_zhang_wuji_template(
        root_dir=templates_root, template_id="demo"
    )
    app = create_app(
        vertical=spec,
        max_sessions=4,
        idle_eviction_seconds=None,
        templates_root_dir=str(templates_root),
    )
    return await aiohttp_client(app)


async def test_list_templates_lists_seeded_template(zhang_wuji_client):
    resp = await zhang_wuji_client.get("/v1/templates")
    assert resp.status == 200
    body = await resp.json()
    assert body["vertical"] == "zhang_wuji"
    assert body["templates_supported"] is True
    template_ids = [item["template_id"] for item in body["templates"]]
    assert "demo" in template_ids
    [demo] = [item for item in body["templates"] if item["template_id"] == "demo"]
    assert demo["display_name"] == "张无忌"
    assert demo["replay_provenance"]
    assert demo["integrity_hash"]
    assert demo["file_path"].endswith("demo.json")


async def test_create_session_default_records_template_context(
    zhang_wuji_client,
):
    """A vertical-default session under a template-aware vertical must
    still be save-as-template eligible."""
    resp = await zhang_wuji_client.post("/v1/sessions", json={})
    assert resp.status == 201
    body = await resp.json()
    assert body["session_id"]
    assert body["template_id"] is None


async def test_create_session_from_template_succeeds(zhang_wuji_client):
    resp = await zhang_wuji_client.post(
        "/v1/sessions",
        json={"session_id": "from-tpl", "template_id": "demo"},
    )
    assert resp.status == 201
    body = await resp.json()
    assert body["session_id"] == "from-tpl"
    assert body["template_id"] == "demo"


async def test_create_session_from_unknown_template_404s(zhang_wuji_client):
    resp = await zhang_wuji_client.post(
        "/v1/sessions", json={"template_id": "no-such-template"}
    )
    assert resp.status == 404
    body = await resp.json()
    assert body["error"] == "template_not_found"


async def test_save_as_template_writes_file_and_lists_under_new_id(
    zhang_wuji_client,
):
    create = await (
        await zhang_wuji_client.post("/v1/sessions", json={})
    ).json()
    sid = create["session_id"]
    # Drive at least one turn so the session has a non-trivial state.
    await zhang_wuji_client.post(
        f"/v1/sessions/{sid}/turns",
        json={"user_input": "训练完毕，留下一句话。"},
    )
    resp = await zhang_wuji_client.post(
        f"/v1/sessions/{sid}/save-as-template",
        json={
            "template_id": "saved-from-chat",
            "replay_provenance": "test e2e save-as-template",
        },
    )
    assert resp.status == 201, await resp.text()
    body = await resp.json()
    saved = body["saved"]
    assert saved["template_id"] == "saved-from-chat"
    assert saved["file_path"].endswith("saved-from-chat.json")
    assert pathlib.Path(saved["file_path"]).is_file()
    # Listing now contains both the seeded template and the new one.
    listing = await (await zhang_wuji_client.get("/v1/templates")).json()
    template_ids = {item["template_id"] for item in listing["templates"]}
    assert {"demo", "saved-from-chat"} <= template_ids


async def test_save_as_template_409s_on_duplicate(zhang_wuji_client):
    create = await (
        await zhang_wuji_client.post("/v1/sessions", json={})
    ).json()
    sid = create["session_id"]
    payload = {"template_id": "demo", "overwrite_existing": False}
    resp = await zhang_wuji_client.post(
        f"/v1/sessions/{sid}/save-as-template", json=payload
    )
    assert resp.status == 409
    body = await resp.json()
    assert body["error"] == "template_already_exists"


async def test_save_as_template_rejects_path_traversal(zhang_wuji_client):
    create = await (
        await zhang_wuji_client.post("/v1/sessions", json={})
    ).json()
    sid = create["session_id"]
    resp = await zhang_wuji_client.post(
        f"/v1/sessions/{sid}/save-as-template",
        json={"template_id": "../escape"},
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["error"] == "invalid_save_request"


async def test_save_as_template_requires_template_id(zhang_wuji_client):
    create = await (
        await zhang_wuji_client.post("/v1/sessions", json={})
    ).json()
    sid = create["session_id"]
    resp = await zhang_wuji_client.post(
        f"/v1/sessions/{sid}/save-as-template", json={}
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["error"] == "invalid_template_id"
