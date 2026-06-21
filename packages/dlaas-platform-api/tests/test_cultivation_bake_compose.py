"""Tests for composing bake into the self-learning workflow.

Covers the pure angle resolver and the in-process `submit_bake_run`
(reused by cultivation induct) with the GPU-free synthetic runner.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from aiohttp import web

from dlaas_platform_api.bake import (
    BAKE_BUNDLE_APP_KEY,
    SyntheticBakeAngleRunner,
    attach_bake_routes,
    submit_bake_run,
)
from dlaas_platform_api.cultivation import _resolve_bake_angle
from dlaas_platform_contracts import BakeRequest
from dlaas_platform_registry import (
    ApplicationStore,
    PlatformAuthBundle,
    PlatformAuthConfig,
    REGISTRY_APP_KEY,
    Registry,
    TenantStore,
)


def test_resolve_bake_angle_from_provenance():
    assert (
        _resolve_bake_angle(SimpleNamespace(provenance={"source_kind": "character"}))
        == "character"
    )
    assert (
        _resolve_bake_angle(
            SimpleNamespace(provenance={"source_angle": "interpreter"})
        )
        == "interpreter"
    )
    # author / figure / expert / unknown all map to the figure 'author' angle.
    assert (
        _resolve_bake_angle(SimpleNamespace(provenance={"source_kind": "figure"}))
        == "author"
    )
    assert _resolve_bake_angle(SimpleNamespace(provenance={})) == "author"


def _build_app() -> tuple[web.Application, Registry]:
    registry = Registry(db_path=":memory:")
    app = web.Application()
    app[REGISTRY_APP_KEY] = PlatformAuthBundle(
        tenant_store=TenantStore(registry),
        auth_config=PlatformAuthConfig(control_plane_secret="s"),
        application_store=ApplicationStore(registry),
    )
    attach_bake_routes(app, registry=registry, runner=SyntheticBakeAngleRunner())
    return app, registry


def test_submit_bake_run_creates_run_and_jobs():
    async def run() -> None:
        app, registry = _build_app()
        bundle = app[BAKE_BUNDLE_APP_KEY]
        bake_request = BakeRequest.from_json(
            {
                "source_ref": "cultivation:child-psych",
                "angles": [
                    {
                        "kind": "author",
                        "slug": "child-psych",
                        "display_name": "儿童心理专家",
                    }
                ],
                "raw_materials": [
                    {
                        "kind": "text",
                        "text": "领域: 儿童心理 / 收敛流派: 依恋",
                        "angle_slugs": ["child-psych"],
                    }
                ],
            }
        )
        run_id, run_obj = submit_bake_run(app, bake_request, tenant_id="")
        assert run_id.startswith("bake_")
        assert run_obj.source_ref == "cultivation:child-psych"
        jobs = bundle.store.list_jobs(run_id)
        assert len(jobs) == 1
        assert jobs[0].angle.slug == "child-psych"
        await bundle.executor.stop()
        registry.close()

    asyncio.run(run())
