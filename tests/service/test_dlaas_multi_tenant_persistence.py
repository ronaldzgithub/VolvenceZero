"""Slice 7.2 — multi-tenant isolation + persistence.

These tests hit the real aiohttp surface end-to-end:

* A control-plane secret bootstraps two tenants.
* Each tenant authenticates with its own ``api_key`` / ``api_secret``
  pair and creates resources.
* Tenant A cannot see tenant B's templates / contracts even when it
  knows the ID strings (registry returns 403 ``tenant_mismatch``).
* The registry persists every resource to a SQLite file; tearing
  down the app and reopening the same DB path returns the same
  rows.
* ``X-DLaaS-Tenant-Key`` / ``X-DLaaS-Tenant-Secret`` aliases work
  alongside the canonical ``X-Tenant-Api-Key`` / ``X-Tenant-Api-Secret``
  headers (DLaaS public compatibility surface).

The vertical here is the ``companion`` vertical that ships with
``lifeform-domain-emogpt``; the kernel is exercised only via the
control-plane CRUD path (no chat turns).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlaas_platform_api import build_dlaas_app


CONTROL_PLANE_SECRET = "cp_secret_test_001"
SERVICE_SECRET = "svc_secret_test_001"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "dlaas.sqlite"


@pytest.fixture
async def dlaas_full_client(aiohttp_client, db_path):
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    app = build_dlaas_app(
        db_path=str(db_path),
        control_plane_secret=CONTROL_PLANE_SECRET,
        service_secret=SERVICE_SECRET,
        vertical=spec,
        max_sessions=4,
        idle_eviction_seconds=None,
    )
    return await aiohttp_client(app)


async def _create_tenant(client, name: str) -> dict:
    resp = await client.post(
        "/dlaas/tenants",
        headers={"X-Control-Plane-Secret": CONTROL_PLANE_SECRET},
        json={
            "tenant_name": name,
            "contact_email": f"ops@{name.lower()}.example",
            "business_type": "education",
            "billing_plan": "pay_as_you_go",
            "quota": {"max_instances": 5},
        },
    )
    assert resp.status == 200, await resp.text()
    payload = await resp.json()
    assert payload["status"] == "ok"
    assert payload["api_key"].startswith("tk_")
    assert payload["api_secret"].startswith("ts_")
    return payload


def _tenant_headers(tenant: dict) -> dict[str, str]:
    return {
        "X-Tenant-Api-Key": tenant["api_key"],
        "X-Tenant-Api-Secret": tenant["api_secret"],
    }


async def test_two_tenants_cannot_see_each_others_templates(dlaas_full_client):
    a = await _create_tenant(dlaas_full_client, "Acme")
    b = await _create_tenant(dlaas_full_client, "Globex")

    # Each tenant declares one shell + one template.
    for tenant, shell_id in ((a, "acme_web"), (b, "globex_web")):
        resp = await dlaas_full_client.post(
            "/dlaas/shells",
            headers=_tenant_headers(tenant),
            json={
                "shell_id": shell_id,
                "shell_kind": "deployment",
                "shell_type": "web_chat",
                "display_name": f"{tenant['tenant_name']} Web",
                "embodiment": {
                    "perception": ["page_context"],
                    "expression": ["text_streaming"],
                    "action": [],
                    "constraints": {},
                },
            },
        )
        assert resp.status == 200, await resp.text()

    resp_a = await dlaas_full_client.post(
        "/dlaas/templates",
        headers=_tenant_headers(a),
        json={
            "template_name": "Acme Template",
            "domain": "education",
            "runtime_template_id": "companion",
        },
    )
    assert resp_a.status == 200, await resp_a.text()
    template_a_id = (await resp_a.json())["template_id"]

    resp_b = await dlaas_full_client.post(
        "/dlaas/templates",
        headers=_tenant_headers(b),
        json={
            "template_name": "Globex Template",
            "domain": "education",
            "runtime_template_id": "companion",
        },
    )
    assert resp_b.status == 200, await resp_b.text()
    template_b_id = (await resp_b.json())["template_id"]

    # Tenant A's listing returns only its own template.
    resp = await dlaas_full_client.get(
        f"/dlaas/tenants/{a['tenant_id']}/templates",
        headers=_tenant_headers(a),
    )
    assert resp.status == 200
    body = await resp.json()
    template_ids = [t["template_id"] for t in body["templates"]]
    assert template_a_id in template_ids
    assert template_b_id not in template_ids

    # Tenant A trying to read tenant B's template by ID is rejected
    # at the assert_tenant_id_matches edge.
    resp = await dlaas_full_client.get(
        f"/dlaas/templates/{template_b_id}",
        headers=_tenant_headers(a),
    )
    assert resp.status == 403, await resp.text()
    payload = await resp.json()
    assert payload["error"] == "tenant_mismatch"

    # Tenant A trying to list tenant B's tenant scope is also rejected.
    resp = await dlaas_full_client.get(
        f"/dlaas/tenants/{b['tenant_id']}/templates",
        headers=_tenant_headers(a),
    )
    assert resp.status == 403, await resp.text()


async def test_invalid_tenant_credentials_return_403_not_404(dlaas_full_client):
    a = await _create_tenant(dlaas_full_client, "Acme2")
    resp = await dlaas_full_client.post(
        "/dlaas/templates",
        headers={
            "X-Tenant-Api-Key": a["api_key"],
            "X-Tenant-Api-Secret": "ts_garbage",
        },
        json={"template_name": "x", "domain": "education"},
    )
    assert resp.status == 403
    payload = await resp.json()
    assert payload["error"] == "invalid_tenant_credentials"


async def test_compatibility_alias_headers_authenticate(dlaas_full_client):
    a = await _create_tenant(dlaas_full_client, "Acme3")
    resp = await dlaas_full_client.post(
        "/dlaas/templates",
        headers={
            "X-DLaaS-Tenant-Key": a["api_key"],
            "X-DLaaS-Tenant-Secret": a["api_secret"],
        },
        json={"template_name": "alias works", "domain": "education"},
    )
    assert resp.status == 200, await resp.text()


async def test_missing_control_plane_secret_rejected(dlaas_full_client):
    resp = await dlaas_full_client.post(
        "/dlaas/tenants",
        json={"tenant_name": "x", "contact_email": "x@y.com"},
    )
    assert resp.status == 401
    payload = await resp.json()
    assert payload["error"] == "missing_control_plane_secret"


async def test_invalid_control_plane_secret_rejected(dlaas_full_client):
    resp = await dlaas_full_client.post(
        "/dlaas/tenants",
        headers={"X-Control-Plane-Secret": "wrong"},
        json={"tenant_name": "x", "contact_email": "x@y.com"},
    )
    assert resp.status == 403
    payload = await resp.json()
    assert payload["error"] == "invalid_control_plane_secret"


async def test_persistence_round_trip_across_app_rebuild(
    aiohttp_client, db_path, tmp_path
):
    """Tenants + shells + templates survive an app rebuild on the same DB."""
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]

    # ---- First app instance: write resources ----
    app_a = build_dlaas_app(
        db_path=str(db_path),
        control_plane_secret=CONTROL_PLANE_SECRET,
        service_secret=SERVICE_SECRET,
        vertical=spec,
        max_sessions=4,
        idle_eviction_seconds=None,
    )
    client_a = await aiohttp_client(app_a)
    a = await _create_tenant(client_a, "PersistMe")
    headers = _tenant_headers(a)
    resp = await client_a.post(
        "/dlaas/shells",
        headers=headers,
        json={
            "shell_id": "persist_web",
            "shell_kind": "deployment",
            "embodiment": {"perception": [], "expression": [], "action": []},
        },
    )
    assert resp.status == 200, await resp.text()
    resp = await client_a.post(
        "/dlaas/templates",
        headers=headers,
        json={
            "template_name": "Persist Template",
            "domain": "education",
            "runtime_template_id": "companion",
        },
    )
    assert resp.status == 200, await resp.text()
    template_id = (await resp.json())["template_id"]

    # Close the underlying registry connection so SQLite flushes WAL.
    from dlaas_platform_registry import REGISTRY_APP_KEY

    bundle = app_a[REGISTRY_APP_KEY]
    bundle.tenant_store._registry.close()
    await client_a.close()

    # ---- Second app instance: same DB path, fresh app ----
    app_b = build_dlaas_app(
        db_path=str(db_path),
        control_plane_secret=CONTROL_PLANE_SECRET,
        service_secret=SERVICE_SECRET,
        vertical=spec,
        max_sessions=4,
        idle_eviction_seconds=None,
    )
    client_b = await aiohttp_client(app_b)

    # Tenant credentials still authenticate.
    resp = await client_b.get(
        f"/dlaas/tenants/{a['tenant_id']}/templates",
        headers=headers,
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    template_ids = [t["template_id"] for t in body["templates"]]
    assert template_id in template_ids

    # Shell record is reachable too.
    resp = await client_b.post(
        "/dlaas/contracts",
        headers=headers,
        json={
            "tenant_id": a["tenant_id"],
            "template_id": template_id,
            "shell_id": "persist_web",
            "engine_tools": {"web_search": True},
        },
    )
    # Contract creation succeeds → confirms shell + template both
    # round-tripped from disk.
    assert resp.status == 200, await resp.text()
