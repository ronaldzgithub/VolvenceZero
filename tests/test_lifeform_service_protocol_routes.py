"""Packet 9.x engineering wrap: HTTP route smoke tests.

Covers:

* Routes mount when ``protocol_uptake_service`` is supplied.
* GET /v1/protocols / candidates work on empty service.
* PROTOCOL_LLM not configured → extraction routes return 503.
* from-payload route works without LLM (API injection).
* Approve / reject / list pipeline.
"""

from __future__ import annotations

import json

import pytest
from aiohttp import web

from lifeform_protocol_runtime import inject_protocol_from_payload
from lifeform_service.app import create_app
from lifeform_service.protocol_uptake import (
    ProtocolUptakeConfig,
    ProtocolUptakeService,
)
from lifeform_service.verticals import discover_verticals


def _build_app(*, llm_client=None) -> web.Application:
    """Build a minimal app with the uptake service installed.

    Uses a placeholder substrate via the real verticals (since
    create_app needs at least one vertical). When no verticals
    are installed the test is skipped — uptake routes are
    orthogonal to vertical wiring."""

    verticals = discover_verticals()
    if not verticals:
        pytest.skip("no verticals available for create_app")

    default_name = next(iter(verticals))
    uptake = ProtocolUptakeService(
        config=ProtocolUptakeConfig(
            autoload_dir=None,
            autoload_force_approve=False,
            llm_client_factory=(lambda: llm_client),
        ),
    )
    return create_app(
        verticals=verticals,
        default_vertical=default_name,
        protocol_uptake_service=uptake,
    )


@pytest.fixture
async def client(aiohttp_client):
    app = _build_app()
    return await aiohttp_client(app)


async def test_list_protocols_empty(client) -> None:
    resp = await client.get("/v1/protocols")
    assert resp.status == 200
    body = await resp.json()
    assert body == {"protocols": [], "count": 0}


async def test_list_candidates_empty(client) -> None:
    resp = await client.get("/v1/protocols/candidates")
    assert resp.status == 200
    body = await resp.json()
    assert body == {"candidates": [], "count": 0}


async def test_extraction_routes_503_when_llm_unset(client) -> None:
    resp = await client.post(
        "/v1/protocols/from-description",
        data=json.dumps({"description": "x", "protocol_id": "p", "advisor_name": "a"}),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status == 503
    body = await resp.json()
    assert body["error"] == "protocol_llm_not_configured"


async def test_from_payload_route_works_without_llm(aiohttp_client) -> None:
    """API injection doesn't need an LLM; route must succeed."""
    app = _build_app()
    cli = await aiohttp_client(app)
    payload = {
        "request_id": "req-test-1",
        "protocol": {
            "protocol_id": "api:test-bot",
            "advisor_name": "test-bot",
            "description": "API-injected test protocol",
            "boundaries": [
                {
                    "boundary_id": "bd:test:no-promo",
                    "description": "no promo",
                    "trigger_reasons": ["promo language"],
                    "severity": "soft_remind",
                }
            ],
            "strategies": [
                {
                    "rule_id": "rule:test:greet",
                    "problem_pattern": "first contact",
                    "recommended_ordering": ["greet"],
                }
            ],
        },
    }
    resp = await cli.post(
        "/v1/protocols/from-payload",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status == 201, await resp.text()
    body = await resp.json()
    assert body["submitted"] is True
    assert body["protocol_id"] == "api:test-bot"

    # Now visible in candidates list.
    listing = await (await cli.get("/v1/protocols/candidates")).json()
    assert listing["count"] == 1
    assert listing["candidates"][0]["protocol_id"] == "api:test-bot"


async def test_approve_pipeline_through_http(aiohttp_client) -> None:
    """from-payload → approve → /v1/protocols sees it."""
    app = _build_app()
    cli = await aiohttp_client(app)

    payload = {
        "request_id": "req-approve",
        "protocol": {
            "protocol_id": "api:approve-bot",
            "advisor_name": "approve-bot",
            "description": "test",
            "boundaries": [
                {
                    "boundary_id": "bd:x",
                    "description": "y",
                    "trigger_reasons": ["t"],
                    "severity": "soft_remind",
                }
            ],
            "strategies": [
                {
                    "rule_id": "rule:x",
                    "problem_pattern": "p",
                    "recommended_ordering": ["s"],
                }
            ],
        },
    }
    await cli.post(
        "/v1/protocols/from-payload",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )

    resp = await cli.post(
        "/v1/protocols/candidates/api:approve-bot/approve"
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["approved"] is True

    listing = await (await cli.get("/v1/protocols")).json()
    assert listing["count"] == 1
    assert listing["protocols"][0]["protocol_id"] == "api:approve-bot"

    # Pending list now empty.
    pending = await (await cli.get("/v1/protocols/candidates")).json()
    assert pending["count"] == 0


async def test_reject_removes_candidate(aiohttp_client) -> None:
    app = _build_app()
    cli = await aiohttp_client(app)
    await cli.post(
        "/v1/protocols/from-payload",
        data=json.dumps(
            {
                "request_id": "req-reject",
                "protocol": {
                    "protocol_id": "api:reject-bot",
                    "advisor_name": "x",
                    "description": "test",
                    "boundaries": [
                        {
                            "boundary_id": "bd:x",
                            "description": "y",
                            "trigger_reasons": ["t"],
                        }
                    ],
                    "strategies": [
                        {
                            "rule_id": "rule:x",
                            "problem_pattern": "p",
                            "recommended_ordering": ["s"],
                        }
                    ],
                },
            }
        ),
        headers={"Content-Type": "application/json"},
    )
    resp = await cli.post(
        "/v1/protocols/candidates/api:reject-bot/reject",
        data=json.dumps({"reason": "test reject"}),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status == 200
    pending = await (await cli.get("/v1/protocols/candidates")).json()
    assert pending["count"] == 0


async def test_unload_removes_approved_protocol(aiohttp_client) -> None:
    app = _build_app()
    cli = await aiohttp_client(app)
    await cli.post(
        "/v1/protocols/from-payload",
        data=json.dumps(
            {
                "request_id": "req-unload",
                "protocol": {
                    "protocol_id": "api:unload-bot",
                    "advisor_name": "x",
                    "description": "test",
                    "boundaries": [
                        {
                            "boundary_id": "bd:x",
                            "description": "y",
                            "trigger_reasons": ["t"],
                        }
                    ],
                    "strategies": [
                        {
                            "rule_id": "rule:x",
                            "problem_pattern": "p",
                            "recommended_ordering": ["s"],
                        }
                    ],
                },
            }
        ),
        headers={"Content-Type": "application/json"},
    )
    await cli.post("/v1/protocols/candidates/api:unload-bot/approve")
    resp = await cli.delete("/v1/protocols/api:unload-bot")
    assert resp.status == 200
    body = await resp.json()
    assert body["unloaded"] is True
    listing = await (await cli.get("/v1/protocols")).json()
    assert listing["count"] == 0
