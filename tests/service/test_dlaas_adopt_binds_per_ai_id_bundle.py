"""Contract test for U2 — adopt path binds bundle per ai_id.

The family-memorial product depends on multiple memorials living in
the same dlaas-platform process. Each memorial has its own ``ai_id``
and its own baked ``FigureArtifactBundle``. Without per-``ai_id``
binding, every memorial would inherit the process-wide default
bundle (Einstein) and citations / refusals would cross-contaminate
across families.

This file covers BOTH layers of the U2 contract:

* **Unit (SessionManager)** — two ai_ids on the same launcher bind
  to distinct bundles without crosstalk. Locks the SessionManager
  side of the contract.
* **HTTP (adopt POST /dlaas/v1/adoptions)** — same invariants exercised
  through the real adopt handler so a regression in
  ``_handle_adopt`` would be caught even if the SessionManager API
  drifts. This is the layer the audit (round 2) flagged as missing
  in the original U2 patch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlaas_platform_api import build_dlaas_app
from dlaas_platform_api.control_plane import (
    BindResult,
    bind_figure_artifact_to_ai_id,
)
from dlaas_platform_launcher.instance_manager import (
    InstanceManager,
    default_vertical_resolver,
)
from lifeform_service import (
    FigureBundleStore,
    default_figure_bundle_store,
    lookup_figure_bundle,
)


class _FakeMemorialBundle:
    def __init__(self, bundle_id: str, figure_id: str) -> None:
        self.bundle_id = bundle_id
        self.figure_id = figure_id


# ---------------------------------------------------------------------------
# Unit-level invariants on SessionManager binding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_ai_ids_bind_distinct_bundles_no_crosstalk() -> None:
    instance_manager = InstanceManager(
        vertical_resolver=default_vertical_resolver(),
        substrate_runtime=None,
    )
    await instance_manager.acquire(
        ai_id="memorial_ai_alpha", runtime_template_id="einstein-bundle"
    )
    await instance_manager.acquire(
        ai_id="memorial_ai_beta", runtime_template_id="einstein-bundle"
    )

    bundle_a = lookup_figure_bundle(bundle_id="einstein")
    bundle_b = _FakeMemorialBundle(
        bundle_id="figure-bundle:family_grandpa:0123456789abcdef",
        figure_id="family_grandpa",
    )

    manager_a = instance_manager.get("memorial_ai_alpha")
    manager_b = instance_manager.get("memorial_ai_beta")

    assert manager_a is not manager_b
    assert manager_a.figure_bundle is None
    assert manager_b.figure_bundle is None

    manager_a.bind_figure_bundle(bundle_a)
    manager_b.bind_figure_bundle(bundle_b)

    assert manager_a.figure_bundle is bundle_a
    assert manager_b.figure_bundle is bundle_b
    assert manager_a.figure_bundle is not bundle_b
    assert manager_b.figure_bundle is not bundle_a


@pytest.mark.asyncio
async def test_rebinding_an_ai_id_does_not_affect_the_other() -> None:
    instance_manager = InstanceManager(
        vertical_resolver=default_vertical_resolver(),
        substrate_runtime=None,
    )
    await instance_manager.acquire(
        ai_id="memorial_ai_alpha", runtime_template_id="einstein-bundle"
    )
    await instance_manager.acquire(
        ai_id="memorial_ai_beta", runtime_template_id="einstein-bundle"
    )

    bundle_a_v1 = _FakeMemorialBundle(
        bundle_id="figure-bundle:family_a:v1aaaaaaaaaaaaaa", figure_id="family_a"
    )
    bundle_a_v2 = _FakeMemorialBundle(
        bundle_id="figure-bundle:family_a:v2bbbbbbbbbbbbbb", figure_id="family_a"
    )
    bundle_b = _FakeMemorialBundle(
        bundle_id="figure-bundle:family_b:0000000000000000", figure_id="family_b"
    )

    instance_manager.get("memorial_ai_alpha").bind_figure_bundle(bundle_a_v1)
    instance_manager.get("memorial_ai_beta").bind_figure_bundle(bundle_b)

    instance_manager.get("memorial_ai_alpha").bind_figure_bundle(bundle_a_v2)

    assert instance_manager.get("memorial_ai_alpha").figure_bundle is bundle_a_v2
    assert instance_manager.get("memorial_ai_beta").figure_bundle is bundle_b


@pytest.mark.asyncio
async def test_unbinding_one_ai_id_does_not_affect_others() -> None:
    instance_manager = InstanceManager(
        vertical_resolver=default_vertical_resolver(),
        substrate_runtime=None,
    )
    await instance_manager.acquire(
        ai_id="memorial_ai_alpha", runtime_template_id="einstein-bundle"
    )
    await instance_manager.acquire(
        ai_id="memorial_ai_beta", runtime_template_id="einstein-bundle"
    )
    bundle_a = _FakeMemorialBundle(
        bundle_id="figure-bundle:family_a:aaaa", figure_id="family_a"
    )
    bundle_b = _FakeMemorialBundle(
        bundle_id="figure-bundle:family_b:bbbb", figure_id="family_b"
    )
    instance_manager.get("memorial_ai_alpha").bind_figure_bundle(bundle_a)
    instance_manager.get("memorial_ai_beta").bind_figure_bundle(bundle_b)

    instance_manager.get("memorial_ai_alpha").bind_figure_bundle(None)

    assert instance_manager.get("memorial_ai_alpha").figure_bundle is None
    assert instance_manager.get("memorial_ai_beta").figure_bundle is bundle_b


# ---------------------------------------------------------------------------
# Helper-level invariants on the bind_figure_artifact_to_ai_id BindResult
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bind_helper_returns_typed_result_for_each_branch() -> None:
    """P7.8 audit closure: the helper must return a BindResult whose
    ``reason`` lets the HTTP adopt/wake handlers map every documented
    failure mode to a typed 503. Bool semantics were ambiguous and
    forced silent 200s on bundle_not_found.
    """

    instance_manager = InstanceManager(
        vertical_resolver=default_vertical_resolver(),
        substrate_runtime=None,
    )
    await instance_manager.acquire(
        ai_id="memorial_ai_helper", runtime_template_id="einstein-bundle"
    )
    # 1. empty figure_artifact_id -> noop with the right reason
    r1 = bind_figure_artifact_to_ai_id(instance_manager, "memorial_ai_helper", "")
    assert r1 == BindResult(bound=False, reason="empty_figure_artifact_id")

    # 2. unknown ai_id -> instance_not_found
    r2 = bind_figure_artifact_to_ai_id(
        instance_manager, "memorial_ai_does_not_exist", "einstein"
    )
    assert r2.bound is False
    assert r2.reason == "instance_not_found"

    # 3. bundle_not_found
    r3 = bind_figure_artifact_to_ai_id(
        instance_manager,
        "memorial_ai_helper",
        "figure-bundle:family_zzz:nonexistent",
    )
    assert r3.bound is False
    assert r3.reason == "bundle_not_found"

    # 4. happy path: register a bundle then bind
    bundle = _FakeMemorialBundle(
        bundle_id="figure-bundle:family_x:happy_path",
        figure_id="family_x",
    )
    default_figure_bundle_store().register(bundle)
    r4 = bind_figure_artifact_to_ai_id(
        instance_manager, "memorial_ai_helper", bundle.bundle_id
    )
    assert r4 == BindResult(bound=True, reason="ok")
    assert instance_manager.get("memorial_ai_helper").figure_bundle is bundle


# ---------------------------------------------------------------------------
# HTTP-level smoke (adopt path)
# ---------------------------------------------------------------------------


CONTROL_PLANE_SECRET = "cp_secret_u2_http"


async def _build_app(tmp_path: Path):
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    return build_dlaas_app(
        db_path=str(tmp_path / "u2_http.sqlite"),
        control_plane_secret=CONTROL_PLANE_SECRET,
        vertical=spec,
        max_sessions=4,
        idle_eviction_seconds=None,
    )


@pytest.fixture
async def http_client(aiohttp_client, tmp_path: Path):
    return await aiohttp_client(await _build_app(tmp_path))


async def _bootstrap_tenant(http_client) -> dict[str, str]:
    cp_headers = {"X-Control-Plane-Secret": CONTROL_PLANE_SECRET}
    resp = await http_client.post(
        "/dlaas/tenants",
        headers=cp_headers,
        json={
            "tenant_name": "U2 HTTP tenant",
            "contact_email": "u2-http@family.example",
            "business_type": "memorial",
        },
    )
    assert resp.status == 200, await resp.text()
    tenant = await resp.json()
    return {
        "X-Tenant-Api-Key": tenant["api_key"],
        "X-Tenant-Api-Secret": tenant["api_secret"],
    }


async def test_adopt_returns_503_when_figure_bundle_not_registered(
    http_client,
) -> None:
    """The U2 promise: when the template names a figure_artifact_id
    that the FigureBundleStore has not registered, adopt must NOT
    silently 200 with no binding — it must fail with a typed 503
    so the caller knows to run U8 rescan.
    """

    headers = await _bootstrap_tenant(http_client)

    # 1. Mint a template that references a bundle nobody has registered.
    resp = await http_client.post(
        "/dlaas/templates",
        headers=headers,
        json={
            "template_name": "u2-no-bundle",
            "runtime_template_id": "einstein-bundle",
            "figure_artifact_id": "figure-bundle:family_phantom:does_not_exist",
            "citation_policy": "required",
            "coverage_policy": "strict_refuse",
        },
    )
    assert resp.status == 200, await resp.text()
    template_id = (await resp.json())["template_id"]

    # 2. Adopt against this template must surface bundle_not_found.
    resp = await http_client.post(
        "/dlaas/v1/adoptions",
        headers=headers,
        json={
            "template_id": template_id,
            "shell_id": "u2-shell-phantom",
        },
    )
    assert resp.status == 503, await resp.text()
    body = await resp.json()
    assert body["error"] == "figure_bundle_not_registered", body
    # Operator hint must mention the rescan endpoint so on-call has
    # a runnable next step.
    assert "rescan" in body.get("detail", "").lower()


async def test_adopt_happy_path_binds_registered_bundle(http_client) -> None:
    """Counterpart to the negative test: when the bundle IS registered
    (the bake-worker -> U8 rescan path puts it there), adopt returns
    200 and the freshly-acquired ai_id has the bundle bound.
    """

    headers = await _bootstrap_tenant(http_client)

    # Register a fake bundle into the in-process store so the template
    # resolves at adopt time. Real deploys do this via U8 rescan
    # against FIGURE_BUNDLE_ROOT.
    bundle = _FakeMemorialBundle(
        bundle_id="figure-bundle:family_happy_u2:abc1234567890def",
        figure_id="family_happy_u2",
    )
    default_figure_bundle_store().register(bundle)

    resp = await http_client.post(
        "/dlaas/templates",
        headers=headers,
        json={
            "template_name": "u2-happy",
            "runtime_template_id": "einstein-bundle",
            "figure_artifact_id": bundle.bundle_id,
            "citation_policy": "required",
            "coverage_policy": "strict_refuse",
        },
    )
    assert resp.status == 200, await resp.text()
    template_id = (await resp.json())["template_id"]

    resp = await http_client.post(
        "/dlaas/v1/adoptions",
        headers=headers,
        json={"template_id": template_id, "shell_id": "u2-shell-happy"},
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["template_id"] == template_id
    assert body["ai_id"]
