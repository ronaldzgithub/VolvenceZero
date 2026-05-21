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
from pathlib import Path

import pytest
from aiohttp import web

from lifeform_protocol_runtime import inject_protocol_from_payload
from lifeform_service.app import create_app
from lifeform_service.protocol_persistence import ProtocolPersistenceStore
from lifeform_service.protocol_uptake import (
    ProtocolUptakeConfig,
    ProtocolUptakeService,
)
from lifeform_service.verticals import discover_verticals


def _build_app(
    *,
    llm_client=None,
    approved_dir: Path | None = None,
) -> web.Application:
    """Build a minimal app with the uptake service installed.

    Uses a placeholder substrate via the real verticals (since
    create_app needs at least one vertical). When no verticals
    are installed the test is skipped — uptake routes are
    orthogonal to vertical wiring.

    Pass ``approved_dir`` to attach a disk-backed
    :class:`ProtocolPersistenceStore` (enables the
    ``/v1/protocols/library/*`` routes)."""

    verticals = discover_verticals()
    if not verticals:
        pytest.skip("no verticals available for create_app")

    default_name = next(iter(verticals))
    persistence = (
        ProtocolPersistenceStore(approved_dir)
        if approved_dir is not None
        else None
    )
    uptake = ProtocolUptakeService(
        config=ProtocolUptakeConfig(
            autoload_dir=None,
            autoload_force_approve=False,
            llm_client_factory=(lambda: llm_client),
        ),
        persistence=persistence,
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


# ---------------------------------------------------------------------------
# Library (disk-backed) HTTP routes
# ---------------------------------------------------------------------------


_LIBRARY_INJECT_PAYLOAD = {
    "request_id": "req-library",
    "protocol": {
        "protocol_id": "api:library-bot",
        "advisor_name": "library-bot",
        "description": "library route fixture",
        "boundaries": [
            {
                "boundary_id": "bd:lib:x",
                "description": "y",
                "trigger_reasons": ["t"],
                "severity": "soft_remind",
            }
        ],
        "strategies": [
            {
                "rule_id": "rule:lib:x",
                "problem_pattern": "p",
                "recommended_ordering": ["s"],
            }
        ],
    },
}


async def _inject_and_approve(cli, *, pid: str) -> None:
    payload = json.loads(json.dumps(_LIBRARY_INJECT_PAYLOAD))
    payload["request_id"] = f"req-{pid}"
    payload["protocol"]["protocol_id"] = pid
    await cli.post(
        "/v1/protocols/from-payload",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    resp = await cli.post(f"/v1/protocols/candidates/{pid}/approve")
    assert resp.status == 200, await resp.text()


async def test_library_routes_503_when_not_configured(aiohttp_client) -> None:
    """Without ``--protocol-approved-dir`` the library routes 503."""
    app = _build_app()
    cli = await aiohttp_client(app)
    for path in (
        "/v1/protocols/library",
    ):
        resp = await cli.get(path)
        assert resp.status == 503
        body = await resp.json()
        assert body["error"] == "protocol_library_not_configured"
    for path in (
        "/v1/protocols/library/anything/load",
        "/v1/protocols/library/anything/unload",
    ):
        resp = await cli.post(path)
        assert resp.status == 503
    resp = await cli.delete("/v1/protocols/library/anything")
    assert resp.status == 503


async def test_library_get_lists_persisted_protocols(
    tmp_path: Path, aiohttp_client
) -> None:
    """Approve via HTTP → GET /library returns one entry, is_active=True."""
    app = _build_app(approved_dir=tmp_path / "lib")
    cli = await aiohttp_client(app)
    await _inject_and_approve(cli, pid="api:library-list")
    resp = await cli.get("/v1/protocols/library")
    assert resp.status == 200
    body = await resp.json()
    assert body["approved_dir"] == str((tmp_path / "lib").resolve())
    assert body["count"] == 1
    entry = body["entries"][0]
    assert entry["protocol_id"] == "api:library-list"
    assert entry["is_active"] is True


async def test_library_load_after_unload(tmp_path: Path, aiohttp_client) -> None:
    """Unload from active set → library still lists with is_active=False;
    POST /load brings it back into the active set."""
    app = _build_app(approved_dir=tmp_path / "lib")
    cli = await aiohttp_client(app)
    pid = "api:library-load"
    await _inject_and_approve(cli, pid=pid)
    unload = await cli.post(f"/v1/protocols/library/{pid}/unload")
    assert unload.status == 200
    assert (await unload.json())["unloaded"] is True
    library = await (await cli.get("/v1/protocols/library")).json()
    assert library["entries"][0]["is_active"] is False
    active = await (await cli.get("/v1/protocols")).json()
    assert active["count"] == 0
    load = await cli.post(f"/v1/protocols/library/{pid}/load")
    assert load.status == 200, await load.text()
    body = await load.json()
    assert body["loaded"] is True
    assert body["protocol_id"] == pid
    active = await (await cli.get("/v1/protocols")).json()
    assert [p["protocol_id"] for p in active["protocols"]] == [pid]


async def test_library_load_missing_returns_404(
    tmp_path: Path, aiohttp_client
) -> None:
    app = _build_app(approved_dir=tmp_path / "lib")
    cli = await aiohttp_client(app)
    resp = await cli.post("/v1/protocols/library/never:approved/load")
    assert resp.status == 404
    body = await resp.json()
    assert body["error"] == "library_entry_not_found"


async def test_library_delete_removes_disk_and_active(
    tmp_path: Path, aiohttp_client
) -> None:
    app = _build_app(approved_dir=tmp_path / "lib")
    cli = await aiohttp_client(app)
    pid = "api:library-delete"
    await _inject_and_approve(cli, pid=pid)
    resp = await cli.delete(f"/v1/protocols/library/{pid}")
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["deleted"] is True
    library = await (await cli.get("/v1/protocols/library")).json()
    assert library["count"] == 0
    active = await (await cli.get("/v1/protocols")).json()
    assert active["count"] == 0


async def test_library_delete_missing_returns_404(
    tmp_path: Path, aiohttp_client
) -> None:
    app = _build_app(approved_dir=tmp_path / "lib")
    cli = await aiohttp_client(app)
    resp = await cli.delete("/v1/protocols/library/never:approved")
    assert resp.status == 404
