"""Route-level smoke for ``POST /dlaas/v1/instances/{ai_id}/export-template``.

Covers the "save a live adopted soul as a new template" capability:

* flag-gated (default OFF → 404),
* operator-authed,
* mints a NEW published template cloning the source template's config
  (``learned_state="template_clone"`` when no self-learned bundle is held
  for the ai_id), and
* opens a persona lifecycle for the new template so it surfaces in the
  soul console.
"""

from __future__ import annotations

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dlaas_platform_api.control_plane import attach_control_plane_routes
from dlaas_platform_registry import (
    ApplicationStore,
    PersonaLifecycleStore,
    PlatformAuthBundle,
    PlatformAuthConfig,
    REGISTRY_APP_KEY,
    Registry,
    TemplateStore,
    TenantStore,
)

_SECRET = "test-control-plane-secret"
_OP_HEADERS = {"X-Control-Plane-Secret": _SECRET}


def _build_app() -> tuple[web.Application, Registry]:
    registry = Registry(db_path=":memory:")
    app = web.Application()
    app[REGISTRY_APP_KEY] = PlatformAuthBundle(
        tenant_store=TenantStore(registry),
        auth_config=PlatformAuthConfig(control_plane_secret=_SECRET),
        application_store=ApplicationStore(registry),
    )
    attach_control_plane_routes(app, registry=registry)
    return app, registry


async def _seed_source_template(registry: Registry) -> str:
    tenant = await TenantStore(registry).create(
        tenant_name="Tenant X",
        contact_email="x@example.com",
        business_type="generic",
    )
    spec = await TemplateStore(registry).create(
        tenant_id=tenant.tenant_id,
        template_name="Sales Coach",
        domain="sales",
        description="source",
        runtime_template_id="cultivation.expert.v0",
        persona_spec={"display_name": "Sales Coach", "app_id": "demo"},
        seed_config={"cultivation_protocol_bundle": {"protocols": ["p1"]}},
    )
    return spec.template_id


def _body(resp_text: str) -> dict:
    return json.loads(resp_text)


async def test_export_disabled_returns_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DLAAS_INSTANCE_EXPORT_TEMPLATE_ENABLED", raising=False)
    app, registry = _build_app()
    source_id = await _seed_source_template(registry)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/dlaas/v1/instances/ai_demo/export-template",
            headers=_OP_HEADERS,
            json={"source_template_id": source_id},
        )
        assert resp.status == 404
        assert _body(await resp.text())["error"] == "instance_export_disabled"


async def test_export_mints_published_template_and_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DLAAS_INSTANCE_EXPORT_TEMPLATE_ENABLED", "1")
    app, registry = _build_app()
    source_id = await _seed_source_template(registry)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/dlaas/v1/instances/ai_demo/export-template",
            headers=_OP_HEADERS,
            json={"source_template_id": source_id, "template_name": "Saved Coach"},
        )
        assert resp.status == 200, await resp.text()
        body = _body(await resp.text())
        new_id = body["template_id"]
        assert new_id and new_id != source_id
        assert body["ai_id"] == "ai_demo"
        assert body["source_template_id"] == source_id
        # No self-learned uptake bundle is held for this ai_id in-process,
        # so we honestly clone the source seed_config.
        assert body["learned_state"] == "template_clone"
        # NB: like the other control-plane template-create handlers, the
        # spread `**spec.to_json()` carries the template's own `status`
        # field ("published") — consumers key off `template_id`, not the
        # envelope status (the portal action uses the HTTP status).
        assert body["status"] == "published"

    # New template is published and carries the cloned learned bundle +
    # the provenance stamp.
    new_spec = await TemplateStore(registry).get(new_id)
    assert new_spec.status.value == "published"
    assert new_spec.seed_config.get("cultivation_protocol_bundle") == {
        "protocols": ["p1"]
    }
    saved = dict(new_spec.persona_spec).get("saved_from_instance") or {}
    assert saved.get("ai_id") == "ai_demo"
    assert saved.get("source_template_id") == source_id

    # A persona lifecycle was opened so the saved soul shows in the console.
    lifecycle = await PersonaLifecycleStore(registry).get_by_template(new_id)
    assert lifecycle.template_id == new_id


async def test_export_requires_source_template_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DLAAS_INSTANCE_EXPORT_TEMPLATE_ENABLED", "1")
    app, _registry = _build_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/dlaas/v1/instances/ai_demo/export-template",
            headers=_OP_HEADERS,
            json={},
        )
        assert resp.status == 400
        assert _body(await resp.text())["error"] == "missing_field"
