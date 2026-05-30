"""HTTP routes + store for the DLaaS persona market (template economy).

This is the platform-tier economic SSOT for cross-tenant AI template
listings: a company lists a template it trained, other companies
subscribe + re-mint under their own tenant, usage is metered, and the
revenue is split (default 70% platform / 30% provider) into an immutable
ledger.

See ``docs/specs/persona_market/01_template_economy.md``.

Auth (mirrors control_plane.py):

* Listing publish / browse / subscribe / a tenant's own usage + ledger
  views require tenant credentials (``require_tenant_auth``).
* Listing mutation (patch / delist) requires the authenticated tenant to
  own the listing (``source_tenant_id`` match), else 403.
* Operator actions (suspend a listing, run settlements, post usage on
  behalf of any tenant, read the full ledger) accept the control-plane
  secret.

Store: in-memory, lock-guarded, single-process SSOT (matches the prior
in-memory persona-market store noted in known-debts). The store is
structured so a persistent backend can replace ``_PersonaMarketStore``
without changing the route layer.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from aiohttp import web

from dlaas_platform_contracts import (
    MarketplaceLedgerEntry,
    PersonaListingSpec,
    PersonaListingStatus,
    PersonaListingVisibility,
    PersonaMarketUsageEvent,
    PersonaPriceModel,
    PersonaProvenance,
    PersonaSubscriptionSpec,
    PersonaSubscriptionStatus,
    SettlementStatus,
    compute_revenue_split,
)
from dlaas_platform_registry import (
    require_control_plane_secret,
    require_tenant_auth,
)

PERSONA_MARKET_STORE_KEY = "dlaas_persona_market_store"
"""``app[PERSONA_MARKET_STORE_KEY]`` — the in-memory market store."""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _error(status: int, code: str, detail: str) -> web.Response:
    return web.json_response(
        {"status": "error", "error": code, "detail": detail}, status=status
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class _PersonaMarketStore:
    """Lock-guarded in-memory persona-market state."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._listings: dict[str, PersonaListingSpec] = {}
        self._subscriptions: dict[str, PersonaSubscriptionSpec] = {}
        # (listing_ref, subscriber_tenant_id) -> subscription_ref
        self._sub_index: dict[tuple[str, str], str] = {}
        self._usage: dict[str, PersonaMarketUsageEvent] = {}
        # (listing_ref, idempotency_key) -> usage_id
        self._usage_idem: dict[tuple[str, str], str] = {}
        self._ledger: list[MarketplaceLedgerEntry] = []

    # ---- listings ----

    async def get_listing(self, listing_ref: str) -> PersonaListingSpec | None:
        async with self._lock:
            return self._listings.get(listing_ref)

    async def put_listing(self, spec: PersonaListingSpec) -> None:
        async with self._lock:
            self._listings[spec.listing_ref] = spec

    async def list_listings(
        self, *, only_active: bool = True
    ) -> list[PersonaListingSpec]:
        async with self._lock:
            out = list(self._listings.values())
        if only_active:
            out = [
                l
                for l in out
                if l.status == PersonaListingStatus.ACTIVE
                and l.visibility == PersonaListingVisibility.PLATFORM
            ]
        return sorted(out, key=lambda l: l.created_at_ms, reverse=True)

    # ---- subscriptions ----

    async def get_subscription_for(
        self, listing_ref: str, subscriber_tenant_id: str
    ) -> PersonaSubscriptionSpec | None:
        async with self._lock:
            ref = self._sub_index.get((listing_ref, subscriber_tenant_id))
            return self._subscriptions.get(ref) if ref else None

    async def put_subscription(self, spec: PersonaSubscriptionSpec) -> None:
        async with self._lock:
            self._subscriptions[spec.subscription_ref] = spec
            self._sub_index[(spec.listing_ref, spec.subscriber_tenant_id)] = (
                spec.subscription_ref
            )

    async def list_subscriptions_for_tenant(
        self, subscriber_tenant_id: str
    ) -> list[PersonaSubscriptionSpec]:
        async with self._lock:
            return [
                s
                for s in self._subscriptions.values()
                if s.subscriber_tenant_id == subscriber_tenant_id
            ]

    # ---- usage + ledger ----

    async def usage_for_idem(
        self, listing_ref: str, idempotency_key: str
    ) -> PersonaMarketUsageEvent | None:
        if not idempotency_key:
            return None
        async with self._lock:
            uid = self._usage_idem.get((listing_ref, idempotency_key))
            return self._usage.get(uid) if uid else None

    async def record_usage_and_ledger(
        self,
        usage: PersonaMarketUsageEvent,
        ledger: MarketplaceLedgerEntry,
    ) -> None:
        async with self._lock:
            self._usage[usage.usage_id] = usage
            if usage.idempotency_key:
                self._usage_idem[(usage.listing_ref, usage.idempotency_key)] = (
                    usage.usage_id
                )
            self._ledger.append(ledger)

    async def append_ledger(self, entry: MarketplaceLedgerEntry) -> None:
        async with self._lock:
            self._ledger.append(entry)

    async def query_ledger(
        self,
        *,
        listing_ref: str | None = None,
        subscriber_tenant_id: str | None = None,
        provider_tenant_id: str | None = None,
        settlement_status: SettlementStatus | None = None,
    ) -> list[MarketplaceLedgerEntry]:
        async with self._lock:
            rows = list(self._ledger)
        if listing_ref:
            rows = [r for r in rows if r.listing_ref == listing_ref]
        if subscriber_tenant_id:
            rows = [r for r in rows if r.subscriber_tenant_id == subscriber_tenant_id]
        if provider_tenant_id:
            rows = [r for r in rows if r.source_tenant_id == provider_tenant_id]
        if settlement_status is not None:
            rows = [r for r in rows if r.settlement_status == settlement_status]
        return rows

    async def settle_pending(
        self, *, provider_tenant_id: str | None, listing_ref: str | None
    ) -> list[MarketplaceLedgerEntry]:
        """Mark matching pending entries settled. Returns the settled set."""
        settled: list[MarketplaceLedgerEntry] = []
        async with self._lock:
            for i, r in enumerate(self._ledger):
                if r.settlement_status != SettlementStatus.PENDING:
                    continue
                if provider_tenant_id and r.source_tenant_id != provider_tenant_id:
                    continue
                if listing_ref and r.listing_ref != listing_ref:
                    continue
                updated = MarketplaceLedgerEntry(
                    entry_id=r.entry_id,
                    listing_ref=r.listing_ref,
                    subscription_ref=r.subscription_ref,
                    source_tenant_id=r.source_tenant_id,
                    subscriber_tenant_id=r.subscriber_tenant_id,
                    gross_cents=r.gross_cents,
                    platform_fee_cents=r.platform_fee_cents,
                    provider_earning_cents=r.provider_earning_cents,
                    currency=r.currency,
                    settlement_status=SettlementStatus.SETTLED,
                    usage_id=r.usage_id,
                    reason=r.reason,
                    created_at_ms=r.created_at_ms,
                )
                self._ledger[i] = updated
                settled.append(updated)
        return settled


def ensure_persona_market_store(app: web.Application) -> _PersonaMarketStore:
    """Idempotently attach the in-memory market store to ``app``."""
    store = app.get(PERSONA_MARKET_STORE_KEY)
    if store is None:
        store = _PersonaMarketStore()
        app[PERSONA_MARKET_STORE_KEY] = store
    return store


def _get_store(request: web.Request) -> _PersonaMarketStore:
    store = request.app.get(PERSONA_MARKET_STORE_KEY)
    if store is None:
        store = ensure_persona_market_store(request.app)
    return store


# ---------------------------------------------------------------------------
# Handlers — listings
# ---------------------------------------------------------------------------


async def _handle_publish_listing(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    body = await request.json()
    if not isinstance(body, dict):
        return _error(400, "invalid_body", "expected a JSON object")
    listing_ref = str(body.get("listing_ref", "")).strip()
    if not listing_ref:
        return _error(400, "missing_listing_ref", "listing_ref is required")

    store = _get_store(request)
    existing = await store.get_listing(listing_ref)
    if existing and existing.source_tenant_id != tenant.tenant_id:
        return _error(403, "listing_owner_mismatch", listing_ref)

    now = _now_ms()
    try:
        price_model = PersonaPriceModel(
            str(body.get("price_model", PersonaPriceModel.MONTHLY_PER_COMPANY.value))
        )
        visibility = PersonaListingVisibility(
            str(body.get("visibility", PersonaListingVisibility.PLATFORM.value))
        )
    except ValueError as exc:
        return _error(400, "invalid_enum", str(exc))

    persona_config = body.get("persona_config") or {}
    if not isinstance(persona_config, dict):
        persona_config = {}

    # A re-publish keeps the existing review status unless the platform
    # auto-approves; a fresh listing starts pending_review so the
    # operator console can gate platform visibility.
    status = existing.status if existing else PersonaListingStatus.PENDING_REVIEW
    spec = PersonaListingSpec(
        listing_ref=listing_ref,
        source_tenant_id=tenant.tenant_id,
        source_company_id=str(body.get("source_company_id", "")),
        source_template_ref=str(body.get("source_template_ref", "")),
        display_name=str(body.get("display_name", "")),
        vertical=str(body.get("vertical", "")),
        archetype=str(body.get("archetype", "")),
        summary=str(body.get("summary", "")),
        persona_config=persona_config,
        asset_bundle_hash=str(body.get("asset_bundle_hash", "")),
        price_model=price_model,
        price_cents=int(body.get("price_cents", 0) or 0),
        currency=str(body.get("currency", "USD")),
        platform_fee_bps=int(body.get("platform_fee_bps", 7000)),
        provider_share_bps=int(body.get("provider_share_bps", 3000)),
        license_scope=str(body.get("license_scope", "single_tenant_remint")),
        visibility=visibility,
        status=status,
        published_by=str(body.get("published_by", "")),
        payout_account_ref=str(body.get("payout_account_ref", "")),
        created_at_ms=existing.created_at_ms if existing else now,
        updated_at_ms=now,
    )
    await store.put_listing(spec)
    return web.json_response({"status": "ok", "listing": spec.to_json()})


async def _handle_list_listings(request: web.Request) -> web.Response:
    await require_tenant_auth(request)
    store = _get_store(request)
    only_active = request.query.get("all", "").lower() != "true"
    listings = await store.list_listings(only_active=only_active)
    return web.json_response(
        {"status": "ok", "listings": [l.to_json() for l in listings]}
    )


async def _handle_get_listing(request: web.Request) -> web.Response:
    await require_tenant_auth(request)
    store = _get_store(request)
    listing_ref = request.match_info["listing_ref"]
    listing = await store.get_listing(listing_ref)
    if listing is None:
        return _error(404, "listing_not_found", listing_ref)
    return web.json_response({"status": "ok", "listing": listing.to_json()})


async def _handle_patch_listing(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    store = _get_store(request)
    listing_ref = request.match_info["listing_ref"]
    listing = await store.get_listing(listing_ref)
    if listing is None:
        return _error(404, "listing_not_found", listing_ref)
    if listing.source_tenant_id != tenant.tenant_id:
        return _error(403, "listing_owner_mismatch", listing_ref)
    body = await request.json()
    if not isinstance(body, dict):
        return _error(400, "invalid_body", "expected a JSON object")

    updates: dict[str, Any] = {}
    if "price_cents" in body:
        updates["price_cents"] = int(body.get("price_cents", 0) or 0)
    if "price_model" in body:
        try:
            updates["price_model"] = PersonaPriceModel(str(body["price_model"]))
        except ValueError as exc:
            return _error(400, "invalid_enum", str(exc))
    if "summary" in body:
        updates["summary"] = str(body.get("summary", ""))
    if "payout_account_ref" in body:
        updates["payout_account_ref"] = str(body.get("payout_account_ref", ""))
    # The owner may move draft/pending -> active (self-publish) but not
    # un-suspend a platform-suspended listing.
    if "status" in body:
        try:
            requested = PersonaListingStatus(str(body["status"]))
        except ValueError as exc:
            return _error(400, "invalid_enum", str(exc))
        if listing.status == PersonaListingStatus.SUSPENDED:
            return _error(409, "listing_suspended", "operator must lift suspension")
        if requested not in (
            PersonaListingStatus.DRAFT,
            PersonaListingStatus.PENDING_REVIEW,
            PersonaListingStatus.ACTIVE,
            PersonaListingStatus.DELISTED,
        ):
            return _error(403, "status_not_owner_settable", requested.value)
        updates["status"] = requested

    merged = {**listing.to_json(), **{k: _enum_value(v) for k, v in updates.items()}}
    spec = PersonaListingSpec.from_json(merged)
    spec = _with_updated_at(spec)
    await store.put_listing(spec)
    return web.json_response({"status": "ok", "listing": spec.to_json()})


async def _handle_delist_listing(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    store = _get_store(request)
    listing_ref = request.match_info["listing_ref"]
    listing = await store.get_listing(listing_ref)
    if listing is None:
        return _error(404, "listing_not_found", listing_ref)
    if listing.source_tenant_id != tenant.tenant_id:
        return _error(403, "listing_owner_mismatch", listing_ref)
    spec = _with_status(listing, PersonaListingStatus.DELISTED)
    await store.put_listing(spec)
    return web.json_response({"status": "ok", "listing": spec.to_json()})


async def _handle_suspend_listing(request: web.Request) -> web.Response:
    # Operator-only safety lever.
    require_control_plane_secret(request)
    store = _get_store(request)
    listing_ref = request.match_info["listing_ref"]
    listing = await store.get_listing(listing_ref)
    if listing is None:
        return _error(404, "listing_not_found", listing_ref)
    body = await request.json() if request.can_read_body else {}
    target = PersonaListingStatus.SUSPENDED
    if isinstance(body, dict) and body.get("status") == "active":
        target = PersonaListingStatus.ACTIVE
    spec = _with_status(listing, target)
    await store.put_listing(spec)
    return web.json_response({"status": "ok", "listing": spec.to_json()})


# ---------------------------------------------------------------------------
# Handlers — subscriptions
# ---------------------------------------------------------------------------


async def _handle_subscribe(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    body = await request.json()
    if not isinstance(body, dict):
        return _error(400, "invalid_body", "expected a JSON object")
    listing_ref = str(body.get("listing_ref", "")).strip()
    if not listing_ref:
        return _error(400, "missing_listing_ref", "listing_ref is required")
    store = _get_store(request)
    listing = await store.get_listing(listing_ref)
    if listing is None:
        return _error(404, "listing_not_found", listing_ref)
    if listing.status not in (
        PersonaListingStatus.ACTIVE,
        PersonaListingStatus.PENDING_REVIEW,
    ):
        return _error(409, "listing_not_subscribable", listing.status.value)
    if listing.source_tenant_id == tenant.tenant_id:
        return _error(409, "self_subscribe_forbidden", listing_ref)

    now = _now_ms()
    existing = await store.get_subscription_for(listing_ref, tenant.tenant_id)
    provenance = PersonaProvenance(
        source_tenant_id=listing.source_tenant_id,
        source_listing_ref=listing.listing_ref,
        published_by=listing.published_by,
        consent_scope=listing.license_scope,
        adopted_at_ms=now,
    )
    if existing:
        spec = existing
    else:
        spec = PersonaSubscriptionSpec(
            subscription_ref=f"sub_{uuid.uuid4().hex[:16]}",
            listing_ref=listing_ref,
            subscriber_tenant_id=tenant.tenant_id,
            subscribed_by=str(body.get("subscribed_by", "")),
            entitlement_status=PersonaSubscriptionStatus.ACTIVE,
            reminted_template_ref="",
            provenance=provenance,
            started_at_ms=now,
            current_period_end_ms=0,
            created_at_ms=now,
        )
        await store.put_subscription(spec)
    return web.json_response(
        {
            "status": "ok",
            "subscription": spec.to_json(),
            # Re-mint instruction: the subscriber app recreates the
            # template under its own tenant from these references. No
            # source runtime state is shared.
            "remint": {
                "source_template_ref": listing.source_template_ref,
                "asset_bundle_hash": listing.asset_bundle_hash,
                "persona_config": dict(listing.persona_config),
                "license_scope": listing.license_scope,
            },
        }
    )


async def _handle_list_subscriptions(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    store = _get_store(request)
    subs = await store.list_subscriptions_for_tenant(tenant.tenant_id)
    return web.json_response(
        {"status": "ok", "subscriptions": [s.to_json() for s in subs]}
    )


# ---------------------------------------------------------------------------
# Handlers — usage + ledger + settlements
# ---------------------------------------------------------------------------


async def _handle_usage_event(request: web.Request) -> web.Response:
    # Subscriber tenant OR control-plane (server-side meter) may post.
    is_control_plane = "X-Control-Plane-Secret" in request.headers
    subscriber_tenant_id = ""
    if is_control_plane:
        require_control_plane_secret(request)
    else:
        tenant = await require_tenant_auth(request)
        subscriber_tenant_id = tenant.tenant_id

    body = await request.json()
    if not isinstance(body, dict):
        return _error(400, "invalid_body", "expected a JSON object")
    listing_ref = str(body.get("listing_ref", "")).strip()
    if not listing_ref:
        return _error(400, "missing_listing_ref", "listing_ref is required")
    if is_control_plane:
        subscriber_tenant_id = str(body.get("subscriber_tenant_id", "")).strip()
    if not subscriber_tenant_id:
        return _error(400, "missing_subscriber_tenant_id", "required")

    store = _get_store(request)
    listing = await store.get_listing(listing_ref)
    if listing is None:
        return _error(404, "listing_not_found", listing_ref)
    sub = await store.get_subscription_for(listing_ref, subscriber_tenant_id)
    if sub is None:
        return _error(409, "not_subscribed", listing_ref)

    idem = str(body.get("idempotency_key", "")).strip()
    dup = await store.usage_for_idem(listing_ref, idem)
    if dup is not None:
        return web.json_response(
            {"status": "ok", "usage": dup.to_json(), "idempotent_replay": True}
        )

    quantity = int(body.get("quantity", 1) or 1)
    unit_price_cents = int(
        body.get("unit_price_cents", listing.price_cents) or 0
    )
    gross_cents = int(body.get("gross_cents", unit_price_cents * quantity) or 0)
    now = _now_ms()
    usage = PersonaMarketUsageEvent(
        usage_id=f"use_{uuid.uuid4().hex[:16]}",
        listing_ref=listing_ref,
        subscriber_tenant_id=subscriber_tenant_id,
        kind=str(body.get("kind", "subscription_period")),
        quantity=quantity,
        unit_price_cents=unit_price_cents,
        gross_cents=gross_cents,
        outcome_ref=str(body.get("outcome_ref", "")),
        idempotency_key=idem,
        created_at_ms=now,
    )
    platform_fee_cents, provider_earning_cents = compute_revenue_split(
        gross_cents, listing.platform_fee_bps
    )
    ledger = MarketplaceLedgerEntry(
        entry_id=f"led_{uuid.uuid4().hex[:16]}",
        listing_ref=listing_ref,
        subscription_ref=sub.subscription_ref,
        source_tenant_id=listing.source_tenant_id,
        subscriber_tenant_id=subscriber_tenant_id,
        gross_cents=gross_cents,
        platform_fee_cents=platform_fee_cents,
        provider_earning_cents=provider_earning_cents,
        currency=listing.currency,
        settlement_status=SettlementStatus.PENDING,
        usage_id=usage.usage_id,
        reason=usage.kind,
        created_at_ms=now,
    )
    await store.record_usage_and_ledger(usage, ledger)
    return web.json_response(
        {"status": "ok", "usage": usage.to_json(), "ledger_entry": ledger.to_json()}
    )


async def _handle_get_ledger(request: web.Request) -> web.Response:
    is_control_plane = "X-Control-Plane-Secret" in request.headers
    store = _get_store(request)
    if is_control_plane:
        require_control_plane_secret(request)
        rows = await store.query_ledger(
            listing_ref=request.query.get("listing_ref") or None,
            subscriber_tenant_id=request.query.get("subscriber_tenant_id") or None,
            provider_tenant_id=request.query.get("provider_tenant_id") or None,
        )
    else:
        tenant = await require_tenant_auth(request)
        # A tenant sees only rows where it is the provider or subscriber.
        role = request.query.get("role", "provider")
        if role == "subscriber":
            rows = await store.query_ledger(subscriber_tenant_id=tenant.tenant_id)
        else:
            rows = await store.query_ledger(provider_tenant_id=tenant.tenant_id)
    gross = sum(r.gross_cents for r in rows)
    platform = sum(r.platform_fee_cents for r in rows)
    provider = sum(r.provider_earning_cents for r in rows)
    return web.json_response(
        {
            "status": "ok",
            "entries": [r.to_json() for r in rows],
            "totals": {
                "gross_cents": gross,
                "platform_fee_cents": platform,
                "provider_earning_cents": provider,
            },
        }
    )


async def _handle_run_settlements(request: web.Request) -> web.Response:
    require_control_plane_secret(request)
    body = await request.json() if request.can_read_body else {}
    body = body if isinstance(body, dict) else {}
    store = _get_store(request)
    settled = await store.settle_pending(
        provider_tenant_id=str(body.get("provider_tenant_id", "")) or None,
        listing_ref=str(body.get("listing_ref", "")) or None,
    )
    provider_total = sum(r.provider_earning_cents for r in settled)
    return web.json_response(
        {
            "status": "ok",
            "settled_count": len(settled),
            "provider_earning_cents": provider_total,
            "entries": [r.to_json() for r in settled],
        }
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _enum_value(v: Any) -> Any:
    return v.value if isinstance(v, (PersonaPriceModel, PersonaListingStatus)) else v


def _with_updated_at(spec: PersonaListingSpec) -> PersonaListingSpec:
    data = spec.to_json()
    data["updated_at_ms"] = _now_ms()
    return PersonaListingSpec.from_json(data)


def _with_status(
    spec: PersonaListingSpec, status: PersonaListingStatus
) -> PersonaListingSpec:
    data = spec.to_json()
    data["status"] = status.value
    data["updated_at_ms"] = _now_ms()
    return PersonaListingSpec.from_json(data)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def attach_persona_market_routes(app: web.Application) -> web.Application:
    """Register the persona-market routes on ``app``."""
    ensure_persona_market_store(app)
    R = app.router
    R.add_post("/dlaas/v1/persona-market/listings", _handle_publish_listing)
    R.add_get("/dlaas/v1/persona-market/listings", _handle_list_listings)
    R.add_get(
        "/dlaas/v1/persona-market/listings/{listing_ref}", _handle_get_listing
    )
    R.add_patch(
        "/dlaas/v1/persona-market/listings/{listing_ref}", _handle_patch_listing
    )
    R.add_post(
        "/dlaas/v1/persona-market/listings/{listing_ref}/delist",
        _handle_delist_listing,
    )
    R.add_post(
        "/dlaas/v1/persona-market/listings/{listing_ref}/suspend",
        _handle_suspend_listing,
    )
    R.add_post("/dlaas/v1/persona-market/subscriptions", _handle_subscribe)
    R.add_get("/dlaas/v1/persona-market/subscriptions", _handle_list_subscriptions)
    R.add_post("/dlaas/v1/persona-market/usage-events", _handle_usage_event)
    R.add_get("/dlaas/v1/persona-market/ledger", _handle_get_ledger)
    R.add_post("/dlaas/v1/persona-market/settlements/run", _handle_run_settlements)
    return app


__all__ = [
    "PERSONA_MARKET_STORE_KEY",
    "attach_persona_market_routes",
    "ensure_persona_market_store",
]
