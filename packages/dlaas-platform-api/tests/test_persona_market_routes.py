"""Route-level end-to-end for the persona-market template economy.

Drives the actual aiohttp handlers (parsing, auth gating, response
shape) through the full economy flow:

  source tenant lists -> subscriber licenses (+ re-mint instruction) ->
  usage metered -> ledger shows the 70/30 split -> operator settles.

Auth is monkeypatched so the test does not need a live registry: the
tenant identity is taken from an ``X-Tenant-Id`` header and the
control-plane secret check is a no-op.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from aiohttp import web

import dlaas_platform_api.persona_market as pm


@pytest.fixture(autouse=True)
def _stub_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_tenant_auth(request: web.Request):
        return SimpleNamespace(
            tenant_id=request.headers.get("X-Tenant-Id", "tenant-a")
        )

    def fake_control_plane(request: web.Request) -> None:
        return None

    monkeypatch.setattr(pm, "require_tenant_auth", fake_tenant_auth)
    monkeypatch.setattr(pm, "require_control_plane_secret", fake_control_plane)


class _FakeRequest:
    def __init__(
        self,
        app: web.Application,
        *,
        json_body=None,
        match_info=None,
        headers=None,
        query=None,
    ) -> None:
        self.app = app
        self._json = json_body
        self.match_info = match_info or {}
        self.headers = headers or {}
        self.query = query or {}

    @property
    def can_read_body(self) -> bool:
        return self._json is not None

    async def json(self):
        return self._json if self._json is not None else {}


def _app() -> web.Application:
    app = web.Application()
    pm.ensure_persona_market_store(app)
    return app


def _body(resp: web.Response) -> dict:
    return json.loads(resp.text)


async def test_full_economy_flow_through_routes() -> None:
    app = _app()
    provider = {"X-Tenant-Id": "tenant-a"}
    subscriber = {"X-Tenant-Id": "tenant-b"}

    # 1. Provider lists a template (priced).
    publish = await pm._handle_publish_listing(
        _FakeRequest(
            app,
            headers=provider,
            json_body={
                "listing_ref": "pl_sales",
                "display_name": "Sales Coach",
                "source_template_ref": "tpl_123",
                "asset_bundle_hash": "abc123",
                "price_cents": 10_000,
                "currency": "USD",
                "visibility": "platform",
            },
        )
    )
    assert publish.status == 200
    # Activate it (owner self-publish from pending_review -> active).
    await pm._handle_patch_listing(
        _FakeRequest(
            app,
            headers=provider,
            match_info={"listing_ref": "pl_sales"},
            json_body={"status": "active"},
        )
    )

    # 2. Subscriber sees + licenses it; gets a re-mint instruction.
    listed = _body(
        await pm._handle_list_listings(_FakeRequest(app, headers=subscriber))
    )
    assert any(l["listing_ref"] == "pl_sales" for l in listed["listings"])

    sub_resp = await pm._handle_subscribe(
        _FakeRequest(
            app,
            headers=subscriber,
            json_body={"listing_ref": "pl_sales"},
        )
    )
    sub_body = _body(sub_resp)
    assert sub_body["subscription"]["entitlement_status"] == "active"
    assert sub_body["remint"]["source_template_ref"] == "tpl_123"
    assert sub_body["remint"]["asset_bundle_hash"] == "abc123"

    # Self-subscribe is rejected.
    self_sub = await pm._handle_subscribe(
        _FakeRequest(app, headers=provider, json_body={"listing_ref": "pl_sales"})
    )
    assert self_sub.status == 409

    # 3. Meter a usage tick -> ledger entry with the 70/30 split.
    usage = _body(
        await pm._handle_usage_event(
            _FakeRequest(
                app,
                headers=subscriber,
                json_body={
                    "listing_ref": "pl_sales",
                    "kind": "subscription_period",
                    "idempotency_key": "period-1",
                },
            )
        )
    )
    assert usage["ledger_entry"]["gross_cents"] == 10_000
    assert usage["ledger_entry"]["platform_fee_cents"] == 7_000
    assert usage["ledger_entry"]["provider_earning_cents"] == 3_000

    # Idempotent replay does not double-charge.
    replay = _body(
        await pm._handle_usage_event(
            _FakeRequest(
                app,
                headers=subscriber,
                json_body={
                    "listing_ref": "pl_sales",
                    "idempotency_key": "period-1",
                },
            )
        )
    )
    assert replay.get("idempotent_replay") is True

    # 4. Provider ledger view shows its 30% share.
    ledger = _body(
        await pm._handle_get_ledger(
            _FakeRequest(app, headers=provider, query={"role": "provider"})
        )
    )
    assert ledger["totals"]["provider_earning_cents"] == 3_000
    assert ledger["totals"]["platform_fee_cents"] == 7_000

    # 5. Operator settles the pending entries.
    settle = _body(
        await pm._handle_run_settlements(
            _FakeRequest(
                app,
                headers={"X-Control-Plane-Secret": "x"},
                json_body={"provider_tenant_id": "tenant-a"},
            )
        )
    )
    assert settle["settled_count"] == 1
    assert settle["provider_earning_cents"] == 3_000


async def test_suspended_listing_blocks_new_subscription() -> None:
    app = _app()
    await pm._handle_publish_listing(
        _FakeRequest(
            app,
            headers={"X-Tenant-Id": "tenant-a"},
            json_body={
                "listing_ref": "pl_x",
                "display_name": "X",
                "price_cents": 100,
            },
        )
    )
    # Operator suspends it.
    await pm._handle_suspend_listing(
        _FakeRequest(
            app,
            headers={"X-Control-Plane-Secret": "x"},
            match_info={"listing_ref": "pl_x"},
            json_body={},
        )
    )
    resp = await pm._handle_subscribe(
        _FakeRequest(
            app,
            headers={"X-Tenant-Id": "tenant-b"},
            json_body={"listing_ref": "pl_x"},
        )
    )
    assert resp.status == 404  # suspended -> not subscribable
