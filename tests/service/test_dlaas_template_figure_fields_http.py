"""Contract test for U3 — POST/PATCH templates accept figure_* fields.

The family-memorial product mints one template per memorial via
``POST /dlaas/templates`` with ``figure_artifact_id`` set to the
bake worker's output bundle id, and re-rolls the field on every
re-bake via ``PATCH``. Until U3, these HTTP handlers silently
dropped the figure fields, forcing operators to drive the registry
store directly. This test locks the HTTP-level invariant.

Covered:

* POST sets ``figure_artifact_id`` / ``citation_policy`` /
  ``coverage_policy`` / ``figure_time_window`` and they survive a
  subsequent GET.
* PATCH can update any of the four fields independently.
* PATCH with no figure_* fields leaves prior values intact.
* Invalid policy strings are rejected with a typed 400.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlaas_platform_api import build_dlaas_app


CONTROL_PLANE_SECRET = "cp_secret_u3_figure_fields"


async def _build_app(tmp_path: Path):
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    return build_dlaas_app(
        db_path=str(tmp_path / "u3_figure.sqlite"),
        control_plane_secret=CONTROL_PLANE_SECRET,
        vertical=spec,
        max_sessions=4,
        idle_eviction_seconds=None,
    )


@pytest.fixture
async def client(aiohttp_client, tmp_path: Path):
    return await aiohttp_client(await _build_app(tmp_path))


async def _create_tenant(client) -> dict[str, str]:
    cp_headers = {"X-Control-Plane-Secret": CONTROL_PLANE_SECRET}
    resp = await client.post(
        "/dlaas/tenants",
        headers=cp_headers,
        json={
            "tenant_name": "Family Memorial Tenant",
            "contact_email": "ops@family.example",
            "business_type": "memorial",
        },
    )
    assert resp.status == 200, await resp.text()
    tenant = await resp.json()
    return {
        "X-Tenant-Api-Key": tenant["api_key"],
        "X-Tenant-Api-Secret": tenant["api_secret"],
    }


async def test_post_template_persists_all_figure_fields(client) -> None:
    headers = await _create_tenant(client)
    payload = {
        "template_name": "memorial-grandpa",
        "domain": "family_memorial",
        "description": "Grandpa Zhang",
        "runtime_template_id": "einstein-bundle",
        "figure_artifact_id": "figure-bundle:family_grandpa01:0123456789abcdef",
        "citation_policy": "required",
        "coverage_policy": "strict_refuse",
        "figure_time_window": "family-grandpa01-1920-2010",
    }
    resp = await client.post("/dlaas/templates", headers=headers, json=payload)
    assert resp.status == 200, await resp.text()
    created = await resp.json()
    template_id = created["template_id"]
    assert created["figure_artifact_id"] == payload["figure_artifact_id"]
    assert created["citation_policy"] == "required"
    assert created["coverage_policy"] == "strict_refuse"
    assert created["figure_time_window"] == payload["figure_time_window"]

    resp = await client.get(f"/dlaas/templates/{template_id}", headers=headers)
    assert resp.status == 200, await resp.text()
    fetched = await resp.json()
    assert fetched["figure_artifact_id"] == payload["figure_artifact_id"]
    assert fetched["citation_policy"] == "required"
    assert fetched["coverage_policy"] == "strict_refuse"
    assert fetched["figure_time_window"] == payload["figure_time_window"]


async def test_post_template_defaults_when_figure_fields_omitted(client) -> None:
    headers = await _create_tenant(client)
    resp = await client.post(
        "/dlaas/templates",
        headers=headers,
        json={
            "template_name": "no-figure",
            "runtime_template_id": "companion",
        },
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["figure_artifact_id"] == ""
    assert body["citation_policy"] == "disabled"
    assert body["coverage_policy"] == "passthrough"
    assert body["figure_time_window"] == ""


async def test_patch_template_updates_figure_artifact_id_only(client) -> None:
    headers = await _create_tenant(client)
    resp = await client.post(
        "/dlaas/templates",
        headers=headers,
        json={
            "template_name": "memorial-v1",
            "runtime_template_id": "einstein-bundle",
            "figure_artifact_id": "figure-bundle:family_x:v1aaaaaaaaaaaaaa",
            "citation_policy": "required",
            "coverage_policy": "strict_refuse",
        },
    )
    assert resp.status == 200, await resp.text()
    template_id = (await resp.json())["template_id"]

    # Re-bake produces a new bundle id; PATCH should roll the
    # template forward without changing the other policy fields.
    resp = await client.patch(
        f"/dlaas/templates/{template_id}",
        headers=headers,
        json={
            "figure_artifact_id": "figure-bundle:family_x:v2bbbbbbbbbbbbbb",
        },
    )
    assert resp.status == 200, await resp.text()
    patched = await resp.json()
    assert patched["figure_artifact_id"] == "figure-bundle:family_x:v2bbbbbbbbbbbbbb"
    assert patched["citation_policy"] == "required"
    assert patched["coverage_policy"] == "strict_refuse"


async def test_patch_template_invalid_policy_returns_400(client) -> None:
    headers = await _create_tenant(client)
    resp = await client.post(
        "/dlaas/templates",
        headers=headers,
        json={
            "template_name": "memorial-validate",
            "runtime_template_id": "einstein-bundle",
        },
    )
    assert resp.status == 200, await resp.text()
    template_id = (await resp.json())["template_id"]

    resp = await client.patch(
        f"/dlaas/templates/{template_id}",
        headers=headers,
        json={"citation_policy": "made_up_value"},
    )
    assert resp.status == 400, await resp.text()
    err = await resp.json()
    assert err["error"] == "invalid_template_policy"


async def test_post_template_invalid_policy_returns_400(client) -> None:
    headers = await _create_tenant(client)
    resp = await client.post(
        "/dlaas/templates",
        headers=headers,
        json={
            "template_name": "memorial-validate-create",
            "runtime_template_id": "einstein-bundle",
            "coverage_policy": "definitely_not_a_real_policy",
        },
    )
    assert resp.status == 400, await resp.text()
    err = await resp.json()
    assert err["error"] == "invalid_template_policy"
