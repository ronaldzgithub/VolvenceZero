"""Contract test for U7 — control-plane template create.

POST /dlaas/control/templates is an operator-side variant of
POST /dlaas/templates: it authorises with the control-plane secret
(X-Control-Plane-Secret) rather than tenant API keys, and takes the
target ``tenant_id`` as an explicit field in the JSON body.

This unlocks deploy-side services (``docker/family-bake-worker``,
``docker/family-transcode-worker``) calling the control plane from
their own service account instead of having to do a tenant bootstrap
and persist the resulting tenant API key.

Covered:

* Happy path: secret + explicit tenant_id mints a template, all
  figure_* fields persist, response includes ``template_id``.
* Missing X-Control-Plane-Secret -> 401.
* Wrong secret -> 403.
* Unknown tenant_id -> 404 with ``tenant_not_found``.
* The minted template_id is usable by ordinary tenant-auth GET on
  the same template (proves the row is persisted, not in some
  side-table).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlaas_platform_api import build_dlaas_app


CONTROL_PLANE_SECRET = "cp_secret_u7_template_create"


async def _build_app(tmp_path: Path):
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    return build_dlaas_app(
        db_path=str(tmp_path / "u7_template_create.sqlite"),
        control_plane_secret=CONTROL_PLANE_SECRET,
        vertical=spec,
        max_sessions=4,
        idle_eviction_seconds=None,
    )


@pytest.fixture
async def client(aiohttp_client, tmp_path: Path):
    return await aiohttp_client(await _build_app(tmp_path))


async def _bootstrap_tenant(client) -> tuple[str, dict[str, str]]:
    cp_headers = {"X-Control-Plane-Secret": CONTROL_PLANE_SECRET}
    resp = await client.post(
        "/dlaas/tenants",
        headers=cp_headers,
        json={
            "tenant_name": "Family Memorial Operator",
            "contact_email": "ops@family.example",
            "business_type": "memorial_operator",
        },
    )
    assert resp.status == 200, await resp.text()
    tenant = await resp.json()
    tenant_headers = {
        "X-Tenant-Api-Key": tenant["api_key"],
        "X-Tenant-Api-Secret": tenant["api_secret"],
    }
    return tenant["tenant_id"], tenant_headers


async def test_control_plane_template_create_happy_path(client) -> None:
    tenant_id, tenant_headers = await _bootstrap_tenant(client)
    cp_headers = {"X-Control-Plane-Secret": CONTROL_PLANE_SECRET}
    payload = {
        "tenant_id": tenant_id,
        "template_name": "memorial-grandpa-via-operator",
        "domain": "family_memorial",
        "runtime_template_id": "einstein-bundle",
        "figure_artifact_id": "figure-bundle:family_grandpa_u7:abc0123456789def",
        "citation_policy": "required",
        "coverage_policy": "strict_refuse",
        "figure_time_window": "family-u7-1920-2010",
    }
    resp = await client.post(
        "/dlaas/control/templates",
        headers=cp_headers,
        json=payload,
    )
    assert resp.status == 200, await resp.text()
    created = await resp.json()
    assert created["tenant_id"] == tenant_id
    template_id = created["template_id"]
    assert template_id
    assert created["figure_artifact_id"] == payload["figure_artifact_id"]
    assert created["citation_policy"] == "required"
    assert created["coverage_policy"] == "strict_refuse"
    assert created["figure_time_window"] == payload["figure_time_window"]

    # The minted row is a first-class template: the tenant can read
    # it back via the normal tenant-auth GET. This proves U7 isn't
    # writing to a parallel store.
    resp = await client.get(
        f"/dlaas/templates/{template_id}", headers=tenant_headers
    )
    assert resp.status == 200, await resp.text()
    fetched = await resp.json()
    assert fetched["template_id"] == template_id
    assert fetched["figure_artifact_id"] == payload["figure_artifact_id"]


async def test_control_plane_template_create_missing_secret_is_401(client) -> None:
    tenant_id, _ = await _bootstrap_tenant(client)
    resp = await client.post(
        "/dlaas/control/templates",
        json={
            "tenant_id": tenant_id,
            "template_name": "missing-secret",
            "runtime_template_id": "einstein-bundle",
        },
    )
    assert resp.status == 401, await resp.text()


async def test_control_plane_template_create_wrong_secret_is_403(client) -> None:
    tenant_id, _ = await _bootstrap_tenant(client)
    resp = await client.post(
        "/dlaas/control/templates",
        headers={"X-Control-Plane-Secret": "definitely-wrong"},
        json={
            "tenant_id": tenant_id,
            "template_name": "wrong-secret",
            "runtime_template_id": "einstein-bundle",
        },
    )
    assert resp.status == 403, await resp.text()


async def test_control_plane_template_create_unknown_tenant_is_404(client) -> None:
    cp_headers = {"X-Control-Plane-Secret": CONTROL_PLANE_SECRET}
    resp = await client.post(
        "/dlaas/control/templates",
        headers=cp_headers,
        json={
            "tenant_id": "tenant_that_does_not_exist",
            "template_name": "no-such-tenant",
            "runtime_template_id": "einstein-bundle",
        },
    )
    assert resp.status == 404, await resp.text()
    body = await resp.json()
    assert body["error"] == "tenant_not_found"


async def test_control_plane_template_create_invalid_policy_is_400(client) -> None:
    tenant_id, _ = await _bootstrap_tenant(client)
    cp_headers = {"X-Control-Plane-Secret": CONTROL_PLANE_SECRET}
    resp = await client.post(
        "/dlaas/control/templates",
        headers=cp_headers,
        json={
            "tenant_id": tenant_id,
            "template_name": "bad-policy",
            "runtime_template_id": "einstein-bundle",
            "citation_policy": "this_is_not_a_policy",
        },
    )
    assert resp.status == 400, await resp.text()
    body = await resp.json()
    assert body["error"] == "invalid_template_policy"
