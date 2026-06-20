"""Smoke tests for the third-party LLM compile plane."""

from __future__ import annotations

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dlaas_platform_api.third_party_llm import (
    ThirdPartyLlmConfig,
    attach_third_party_llm_routes,
)
from dlaas_platform_registry import (
    ApplicationStore,
    PlatformAuthBundle,
    PlatformAuthConfig,
    REGISTRY_APP_KEY,
    Registry,
    TenantStore,
)

_SECRET = "test-control-plane-secret"
_HEADERS = {"X-Control-Plane-Secret": _SECRET}


async def _fake_chat(request: web.Request) -> web.Response:
    body = await request.json()
    schema_name = (
        body.get("response_format", {})
        .get("json_schema", {})
        .get("name", "unknown")
    )
    return web.json_response(
        {
            "id": "chatcmpl_fake",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "slug": "fake",
                                "display_name": "Fake",
                                "description": f"schema={schema_name}",
                                "domain_coverage_seed": ["test"],
                                "knowledge_seeds": [{"title": "T"}],
                                "boundary_priors": {
                                    "out_of_scope_topics": ["outside test"]
                                },
                            }
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        }
    )


def _platform_app(config: ThirdPartyLlmConfig) -> web.Application:
    registry = Registry(db_path=":memory:")
    app = web.Application()
    app[REGISTRY_APP_KEY] = PlatformAuthBundle(
        tenant_store=TenantStore(registry),
        auth_config=PlatformAuthConfig(control_plane_secret=_SECRET),
        application_store=ApplicationStore(registry),
    )
    attach_third_party_llm_routes(app, registry=registry, config=config)
    return app


@pytest.mark.asyncio
async def test_status_and_json_endpoint() -> None:
    provider = web.Application()
    provider.router.add_post("/v1/chat/completions", _fake_chat)
    async with TestClient(TestServer(provider)) as provider_client:
        config = ThirdPartyLlmConfig(
            provider="fake",
            base_url=str(provider_client.make_url("/v1")),
            api_key="k",
            model="fake-model",
            timeout_seconds=5,
        )
        async with TestClient(TestServer(_platform_app(config))) as client:
            status = await client.get(
                "/dlaas/v1/third-party-llm/status", headers=_HEADERS
            )
            assert status.status == 200
            status_body = await status.json()
            assert status_body["configured"] is True
            assert status_body["api_key_present"] is True

            resp = await client.post(
                "/dlaas/v1/third-party-llm/json",
                headers=_HEADERS,
                json={
                    "system_prompt": "Return JSON.",
                    "user_prompt": "Fake profile.",
                    "schema_name": "fake_schema",
                    "schema": {
                        "type": "object",
                        "required": ["slug", "display_name"],
                    },
                },
            )
            assert resp.status == 200, await resp.text()
            body = await resp.json()
            assert body["status"] == "ok"
            assert body["content"]["slug"] == "fake"
            assert body["provider"] == "fake"
            assert body["model"] == "fake-model"


@pytest.mark.asyncio
async def test_third_party_llm_requires_auth() -> None:
    config = ThirdPartyLlmConfig(
        provider="fake",
        base_url="http://127.0.0.1:9/v1",
        api_key="k",
        model="m",
        timeout_seconds=1,
    )
    async with TestClient(TestServer(_platform_app(config))) as client:
        resp = await client.get("/dlaas/v1/third-party-llm/status")
        assert resp.status in (401, 403)


@pytest.mark.asyncio
async def test_third_party_llm_unconfigured_json_is_503() -> None:
    config = ThirdPartyLlmConfig(
        provider="fake",
        base_url="",
        api_key="",
        model="",
        timeout_seconds=1,
    )
    async with TestClient(TestServer(_platform_app(config))) as client:
        resp = await client.post(
            "/dlaas/v1/third-party-llm/json",
            headers=_HEADERS,
            json={"system_prompt": "x", "user_prompt": "y", "schema": {}},
        )
        assert resp.status == 503
        body = await resp.json()
        assert body["error"] == "third_party_llm_unconfigured"
