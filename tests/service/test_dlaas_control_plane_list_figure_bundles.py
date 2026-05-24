"""Contract test for U9 — control-plane list figure bundles.

``GET /dlaas/control/figure-bundles`` is the catalog view that lets
per-tenant apps (digital-employee admin UI in particular) render a
"public persona library" picker without forcing operators to embed
the ``FIGURE_BUNDLE_ROOT`` contents in env / config or to grant the
control-plane secret to every consumer.

Covered:

* The seeded Einstein bundle is in the catalog out of the box (the
  ``FigureBundleStore.default_store`` initialiser seeds it on first
  access, see ``lifeform_service/figure_bundle_store.py``).
* The view is de-duplicated by bundle identity even though
  ``FigureBundleStore.register`` writes the same bundle under both
  its ``bundle_id`` and its ``figure_id``.
* Each row has the fields the picker UI consumes (``figure_id``,
  ``figure_name``, ``figure_lifespan``, ``version_window``,
  ``time_windows``, ``has_lora``, ``has_presence``).
* Control-plane-secret auth works AND tenant auth works (so a
  per-tenant app can call it directly).
* No-auth / wrong-auth is rejected with the standard 401 / 403.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlaas_platform_api import build_dlaas_app


CONTROL_PLANE_SECRET = "cp_secret_u9_list_bundles"


async def _build_app(tmp_path: Path):
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    return build_dlaas_app(
        db_path=str(tmp_path / "u9_list_bundles.sqlite"),
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
            "tenant_name": "U9 Tenant",
            "contact_email": "ops@u9.example",
            "business_type": "digital-employee",
        },
    )
    assert resp.status == 200, await resp.text()
    return await resp.json()


async def test_list_returns_seeded_einstein(client) -> None:
    cp_headers = {"X-Control-Plane-Secret": CONTROL_PLANE_SECRET}
    resp = await client.get(
        "/dlaas/control/figure-bundles", headers=cp_headers
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["status"] == "ok"
    bundles = body["figure_bundles"]
    assert isinstance(bundles, list)
    assert bundles, "expected at least the seeded Einstein bundle"

    einstein = next((b for b in bundles if b["figure_id"] == "einstein"), None)
    assert einstein is not None, bundles
    assert einstein["figure_name"]
    assert isinstance(einstein["figure_lifespan"], list)
    assert len(einstein["figure_lifespan"]) == 2
    assert isinstance(einstein["version_window"], list)
    assert len(einstein["version_window"]) == 2
    assert isinstance(einstein["time_windows"], list)
    assert isinstance(einstein["has_lora"], bool)
    assert isinstance(einstein["has_presence"], bool)
    assert isinstance(einstein["domain_coverage_seed"], list)


async def test_list_is_deduplicated_by_bundle_identity(client) -> None:
    """`FigureBundleStore.register` puts each bundle under two keys
    (``bundle_id`` + ``figure_id``); the catalog view must collapse
    those into a single row keyed by bundle identity, not key."""

    cp_headers = {"X-Control-Plane-Secret": CONTROL_PLANE_SECRET}
    resp = await client.get(
        "/dlaas/control/figure-bundles", headers=cp_headers
    )
    assert resp.status == 200, await resp.text()
    bundles = (await resp.json())["figure_bundles"]
    bundle_ids = [b["bundle_id"] for b in bundles]
    assert len(bundle_ids) == len(set(bundle_ids))
    figure_ids = [b["figure_id"] for b in bundles]
    assert len(figure_ids) == len(set(figure_ids))


async def test_list_accepts_tenant_auth(client) -> None:
    """Per-tenant apps must not be forced to hold the operator secret."""

    tenant = await _create_tenant(client)
    tenant_headers = {
        "X-Tenant-Api-Key": tenant["api_key"],
        "X-Tenant-Api-Secret": tenant["api_secret"],
    }
    resp = await client.get(
        "/dlaas/control/figure-bundles", headers=tenant_headers
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["status"] == "ok"
    assert "figure_bundles" in body


async def test_list_missing_auth_is_401(client) -> None:
    resp = await client.get("/dlaas/control/figure-bundles")
    assert resp.status == 401, await resp.text()


async def test_list_wrong_control_plane_secret_is_403(client) -> None:
    resp = await client.get(
        "/dlaas/control/figure-bundles",
        headers={"X-Control-Plane-Secret": "wrong-secret"},
    )
    assert resp.status == 403, await resp.text()
