"""Contract test for U5 — wake path binds figure bundle from template_id.

The family-memorial bake-worker mints a per-memorial template via
``POST /dlaas/templates`` (carrying ``figure_artifact_id``) and then
materializes the memorial's ``ai_id`` via
``POST /dlaas/v1/instances/<ai>/wake`` with ``template_id`` in the
body. Without U5, the wake handler did not consult the template, so
the bundle was never bound to the ``ai_id``'s SessionManager — every
memorial chat would fall back to the platform-default Einstein bundle
and the L3 / L4 contract would silently apply against the wrong
coverage map.

This test locks the new contract: a wake request that names a
``template_id`` whose template carries a ``figure_artifact_id`` MUST
end with ``instance_manager.get(ai).figure_bundle`` set to the
matching bundle. Mirrors the U2 contract (adopt path) so adopt + wake
produce identical post-conditions on the SessionManager.

We drive the full HTTP surface (so the WakeRequest schema +
control_plane lookup helper are both exercised), and assert against
the launcher's in-memory state — the same launcher every chat call
goes through.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlaas_platform_api import build_dlaas_app
from dlaas_platform_launcher import INSTANCE_MANAGER_APP_KEY


CONTROL_PLANE_SECRET = "cp_secret_u5_wake_bind"


async def _build_app(tmp_path: Path):
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["einstein-bundle"]
    return build_dlaas_app(
        db_path=str(tmp_path / "u5_wake.sqlite"),
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
            "tenant_name": "U5 Wake Tenant",
            "contact_email": "ops@u5.example",
            "business_type": "memorial",
        },
    )
    assert resp.status == 200, await resp.text()
    tenant = await resp.json()
    return {
        "X-Tenant-Api-Key": tenant["api_key"],
        "X-Tenant-Api-Secret": tenant["api_secret"],
    }


async def _create_template_with_figure(client, tenant_headers) -> str:
    """Create a template that names the default Einstein bundle.

    We re-use the seeded Einstein bundle (its ``figure_id`` is the
    string ``einstein``) so the lookup in U5 finds a real bundle in
    the default FigureBundleStore.
    """
    resp = await client.post(
        "/dlaas/templates",
        headers=tenant_headers,
        json={
            "template_name": "u5-memorial-template",
            "runtime_template_id": "einstein-bundle",
            "figure_artifact_id": "einstein",
            "citation_policy": "required",
            "coverage_policy": "strict_refuse",
        },
    )
    assert resp.status == 200, await resp.text()
    return (await resp.json())["template_id"]


async def test_wake_with_template_id_binds_figure_bundle(client) -> None:
    headers = await _create_tenant(client)
    template_id = await _create_template_with_figure(client, headers)

    ai_id = "memorial_u5_alpha"
    resp = await client.post(
        f"/dlaas/v1/instances/{ai_id}/wake",
        json={
            "runtime_template_id": "einstein-bundle",
            "template_id": template_id,
            "reason": "u5-contract-test",
        },
    )
    assert resp.status == 200, await resp.text()

    launcher = client.app[INSTANCE_MANAGER_APP_KEY]
    assert launcher.has(ai_id), "wake should have materialized the ai_id"
    session_manager = launcher.get(ai_id)
    bound = session_manager.figure_bundle
    assert bound is not None, (
        "U5 invariant: wake with template_id MUST bind the figure bundle "
        "to the SessionManager — bake-worker depends on this. "
        f"Got figure_bundle={bound!r}."
    )
    assert getattr(bound, "figure_id", "") == "einstein"


async def test_wake_without_template_id_keeps_legacy_no_bind(client) -> None:
    """Legacy callers (bootstrap-einstein etc.) don't pass template_id.
    For them ``SessionManager.figure_bundle`` stays None — the vertical
    factory still attaches its own bundle through the regular session
    creation path; we only validate that wake itself doesn't touch
    the manager-level binding in the no-template_id case."""

    ai_id = "memorial_u5_beta"
    resp = await client.post(
        f"/dlaas/v1/instances/{ai_id}/wake",
        json={
            "runtime_template_id": "einstein-bundle",
            "reason": "u5-legacy-no-template",
        },
    )
    assert resp.status == 200, await resp.text()

    launcher = client.app[INSTANCE_MANAGER_APP_KEY]
    session_manager = launcher.get(ai_id)
    # No template_id => no manager-level bind. The synthesizer may
    # still attach a bundle via the einstein-bundle factory at session
    # construction time, which is a different code path.
    assert session_manager.figure_bundle is None


async def test_wake_with_unknown_template_id_returns_404(client) -> None:
    ai_id = "memorial_u5_gamma"
    resp = await client.post(
        f"/dlaas/v1/instances/{ai_id}/wake",
        json={
            "runtime_template_id": "einstein-bundle",
            "template_id": "tpl_does_not_exist",
            "reason": "u5-bad-template",
        },
    )
    assert resp.status == 404, await resp.text()
    err = await resp.json()
    assert err["error"] == "template_not_found"


async def test_wake_with_template_having_no_figure_artifact_id_is_noop(
    client,
) -> None:
    """A template with empty ``figure_artifact_id`` should not raise
    but also should not bind anything (caller may simply be using
    template_id for telemetry / future template-policy fields)."""

    headers = await _create_tenant(client)
    resp = await client.post(
        "/dlaas/templates",
        headers=headers,
        json={
            "template_name": "u5-no-figure",
            "runtime_template_id": "einstein-bundle",
        },
    )
    assert resp.status == 200, await resp.text()
    template_id = (await resp.json())["template_id"]

    ai_id = "memorial_u5_delta"
    resp = await client.post(
        f"/dlaas/v1/instances/{ai_id}/wake",
        json={
            "runtime_template_id": "einstein-bundle",
            "template_id": template_id,
            "reason": "u5-empty-figure",
        },
    )
    assert resp.status == 200, await resp.text()
    launcher = client.app[INSTANCE_MANAGER_APP_KEY]
    session_manager = launcher.get(ai_id)
    assert session_manager.figure_bundle is None


async def test_two_ai_ids_wake_with_distinct_templates_bind_distinctly(
    client,
) -> None:
    """Two memorials, two templates, two wakes; each ai_id ends up
    bound to ITS OWN bundle. Same invariant as U2 but exercised
    through the wake path."""

    headers_a = await _create_tenant(client)
    headers_b = await _create_tenant(client)
    tpl_a = await _create_template_with_figure(client, headers_a)
    tpl_b = await _create_template_with_figure(client, headers_b)

    for ai_id, tpl in (
        ("memorial_u5_e1", tpl_a),
        ("memorial_u5_e2", tpl_b),
    ):
        resp = await client.post(
            f"/dlaas/v1/instances/{ai_id}/wake",
            json={
                "runtime_template_id": "einstein-bundle",
                "template_id": tpl,
                "reason": "u5-two-ai",
            },
        )
        assert resp.status == 200, await resp.text()

    launcher = client.app[INSTANCE_MANAGER_APP_KEY]
    m_a = launcher.get("memorial_u5_e1")
    m_b = launcher.get("memorial_u5_e2")
    assert m_a is not m_b
    assert m_a.figure_bundle is not None
    assert m_b.figure_bundle is not None
    # Both templates reference the seeded "einstein" bundle id, so
    # they resolve to the SAME bundle object — which is fine; the
    # invariant is "each ai_id has its own SessionManager which got
    # bind_figure_bundle called on it"; the bundle object can be
    # shared without crosstalk because SessionManager._figure_bundle
    # is a per-manager slot. For DISTINCT bundles see the U2 contract
    # test (test_dlaas_adopt_binds_per_ai_id_bundle.py); here we lock
    # the wake-path-level invariant.
    assert m_a.figure_bundle is m_b.figure_bundle  # same shared bundle
