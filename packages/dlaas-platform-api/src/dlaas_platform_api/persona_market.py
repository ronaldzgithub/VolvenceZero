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

Store: ``PersonaMarketStore`` is an in-memory, single-process SSOT
(matches the prior in-memory persona-market store noted in
known-debts). Methods are synchronous — the aiohttp handlers run on one
event-loop thread so dict ops are atomic, and a persistent backend can
later replace this class without changing the route layer.
"""

from __future__ import annotations

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

_NOT_SUBSCRIBABLE_STATUSES = frozenset(
    {
        PersonaListingStatus.DELISTED,
        PersonaListingStatus.REJECTED,
        PersonaListingStatus.SUSPENDED,
    }
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _error(status: int, code: str, detail: str) -> web.Response:
    return web.json_response(
        {"status": "error", "error": code, "detail": detail}, status=status
    )


class PersonaMarketError(Exception):
    """Typed marketplace error carrying a stable ``code`` for HTTP mapping."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail or code


_ERROR_HTTP_STATUS = {
    "not_found": 404,
    "forbidden": 403,
    "self_subscribe": 409,
    "not_subscribable": 409,
    "not_subscribed": 409,
    "suspended": 409,
}


def _market_error_response(err: PersonaMarketError) -> web.Response:
    return _error(_ERROR_HTTP_STATUS.get(err.code, 400), err.code, err.detail)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class PersonaMarketStore:
    """In-memory persona-market SSOT: listings, subscriptions, ledger."""

    def __init__(self) -> None:
        self._listings: dict[str, PersonaListingSpec] = {}
        self._subscriptions: dict[str, PersonaSubscriptionSpec] = {}
        # (listing_ref, subscriber_tenant_id) -> subscription_id
        self._sub_index: dict[tuple[str, str], str] = {}
        self._usage: dict[str, PersonaMarketUsageEvent] = {}
        # (listing_ref, idempotency_key) -> usage_id
        self._usage_idem: dict[tuple[str, str], str] = {}
        self._ledger: list[MarketplaceLedgerEntry] = []

    # ---- listings ----

    def publish(
        self,
        *,
        tenant_id: str,
        display_name: str,
        persona_config: dict[str, Any] | None = None,
        listing_ref: str | None = None,
        source_company_id: str = "",
        source_template_ref: str = "",
        vertical: str = "",
        archetype: str = "",
        summary: str = "",
        asset_bundle_hash: str = "",
        price_model: str = PersonaPriceModel.MONTHLY_PER_COMPANY.value,
        price_cents: int = 0,
        currency: str = "USD",
        platform_fee_bps: int = 7000,
        provider_share_bps: int = 3000,
        license_scope: str = "single_tenant_remint",
        visibility: str = PersonaListingVisibility.PLATFORM.value,
        published_by: str = "",
        payout_account_ref: str = "",
    ) -> PersonaListingSpec:
        ref = (listing_ref or "").strip() or f"pl_{uuid.uuid4().hex[:16]}"
        existing = self._listings.get(ref)
        if existing is not None and existing.source_tenant_id != tenant_id:
            raise PersonaMarketError("forbidden", f"listing {ref} owned by another tenant")
        now = _now_ms()
        spec = PersonaListingSpec(
            listing_ref=ref,
            source_tenant_id=tenant_id,
            source_company_id=source_company_id,
            source_template_ref=source_template_ref,
            display_name=display_name,
            vertical=vertical,
            archetype=archetype,
            summary=summary,
            persona_config=dict(persona_config or {}),
            asset_bundle_hash=asset_bundle_hash,
            price_model=PersonaPriceModel(price_model),
            price_cents=int(price_cents),
            currency=currency,
            platform_fee_bps=int(platform_fee_bps),
            provider_share_bps=int(provider_share_bps),
            license_scope=license_scope,
            visibility=PersonaListingVisibility(visibility),
            status=existing.status if existing else PersonaListingStatus.PENDING_REVIEW,
            published_by=published_by,
            payout_account_ref=payout_account_ref,
            created_at_ms=existing.created_at_ms if existing else now,
            updated_at_ms=now,
        )
        self._listings[ref] = spec
        return spec

    def get(self, listing_ref: str) -> PersonaListingSpec | None:
        return self._listings.get(listing_ref)

    def available_for(self, tenant_id: str) -> list[PersonaListingSpec]:
        """Listings a tenant can browse: platform-visible OR its own,
        excluding delisted / rejected / suspended."""
        out = [
            l
            for l in self._listings.values()
            if l.status not in _NOT_SUBSCRIBABLE_STATUSES
            and (
                l.visibility == PersonaListingVisibility.PLATFORM
                or l.source_tenant_id == tenant_id
            )
        ]
        return sorted(out, key=lambda l: l.created_at_ms, reverse=True)

    def update(
        self, *, tenant_id: str, listing_ref: str, **fields: Any
    ) -> PersonaListingSpec:
        listing = self._listings.get(listing_ref)
        if listing is None:
            raise PersonaMarketError("not_found", listing_ref)
        if listing.source_tenant_id != tenant_id:
            raise PersonaMarketError("forbidden", listing_ref)
        if listing.status == PersonaListingStatus.SUSPENDED:
            raise PersonaMarketError("suspended", "operator must lift suspension")
        data = listing.to_json()
        for key, value in fields.items():
            if value is None:
                continue
            data[key] = value.value if hasattr(value, "value") else value
        data["updated_at_ms"] = _now_ms()
        spec = PersonaListingSpec.from_json(data)
        self._listings[listing_ref] = spec
        return spec

    def delist(self, *, tenant_id: str, listing_ref: str) -> PersonaListingSpec:
        listing = self._listings.get(listing_ref)
        if listing is None:
            raise PersonaMarketError("not_found", listing_ref)
        if listing.source_tenant_id != tenant_id:
            raise PersonaMarketError("forbidden", listing_ref)
        return self.set_status(listing_ref, PersonaListingStatus.DELISTED)

    def set_status(
        self, listing_ref: str, status: PersonaListingStatus
    ) -> PersonaListingSpec:
        listing = self._listings.get(listing_ref)
        if listing is None:
            raise PersonaMarketError("not_found", listing_ref)
        data = listing.to_json()
        data["status"] = status.value
        data["updated_at_ms"] = _now_ms()
        spec = PersonaListingSpec.from_json(data)
        self._listings[listing_ref] = spec
        return spec

    # ---- subscriptions ----

    def subscription_for(
        self, listing_ref: str, subscriber_tenant_id: str
    ) -> PersonaSubscriptionSpec | None:
        sub_id = self._sub_index.get((listing_ref, subscriber_tenant_id))
        return self._subscriptions.get(sub_id) if sub_id else None

    def subscribe(
        self,
        *,
        subscriber_tenant_id: str,
        listing_ref: str,
        subscribed_by: str = "",
    ) -> PersonaSubscriptionSpec:
        listing = self._listings.get(listing_ref)
        if listing is None or listing.status in _NOT_SUBSCRIBABLE_STATUSES:
            raise PersonaMarketError("not_found", listing_ref)
        if listing.source_tenant_id == subscriber_tenant_id:
            raise PersonaMarketError("self_subscribe", listing_ref)
        existing = self.subscription_for(listing_ref, subscriber_tenant_id)
        if existing is not None:
            return existing
        now = _now_ms()
        provenance = PersonaProvenance(
            source_tenant_id=listing.source_tenant_id,
            source_listing_ref=listing.listing_ref,
            published_by=listing.published_by,
            consent_scope="cross_tenant_adopt",
            adopted_at_ms=now,
        )
        spec = PersonaSubscriptionSpec(
            subscription_id=f"ps_{uuid.uuid4().hex[:16]}",
            listing_ref=listing_ref,
            subscriber_tenant_id=subscriber_tenant_id,
            subscribed_by=subscribed_by,
            entitlement_status=PersonaSubscriptionStatus.ACTIVE,
            reminted_template_ref="",
            provenance=provenance,
            started_at_ms=now,
            current_period_end_ms=0,
            created_at_ms=now,
        )
        self._subscriptions[spec.subscription_id] = spec
        self._sub_index[(listing_ref, subscriber_tenant_id)] = spec.subscription_id
        return spec

    def subscriptions_for(
        self, subscriber_tenant_id: str
    ) -> list[PersonaSubscriptionSpec]:
        return [
            s
            for s in self._subscriptions.values()
            if s.subscriber_tenant_id == subscriber_tenant_id
        ]

    # ---- usage + ledger ----

    def usage_for_idem(
        self, listing_ref: str, idempotency_key: str
    ) -> PersonaMarketUsageEvent | None:
        if not idempotency_key:
            return None
        uid = self._usage_idem.get((listing_ref, idempotency_key))
        return self._usage.get(uid) if uid else None

    def record_usage(
        self,
        *,
        listing_ref: str,
        subscriber_tenant_id: str,
        kind: str = "subscription_period",
        quantity: int = 1,
        unit_price_cents: int | None = None,
        gross_cents: int | None = None,
        outcome_ref: str = "",
        idempotency_key: str = "",
    ) -> tuple[PersonaMarketUsageEvent, MarketplaceLedgerEntry]:
        listing = self._listings.get(listing_ref)
        if listing is None:
            raise PersonaMarketError("not_found", listing_ref)
        sub = self.subscription_for(listing_ref, subscriber_tenant_id)
        if sub is None:
            raise PersonaMarketError("not_subscribed", listing_ref)
        unit = listing.price_cents if unit_price_cents is None else int(unit_price_cents)
        gross = unit * int(quantity) if gross_cents is None else int(gross_cents)
        now = _now_ms()
        usage = PersonaMarketUsageEvent(
            usage_id=f"use_{uuid.uuid4().hex[:16]}",
            listing_ref=listing_ref,
            subscriber_tenant_id=subscriber_tenant_id,
            kind=kind,
            quantity=int(quantity),
            unit_price_cents=unit,
            gross_cents=gross,
            outcome_ref=outcome_ref,
            idempotency_key=idempotency_key,
            created_at_ms=now,
        )
        platform_fee_cents, provider_earning_cents = compute_revenue_split(
            gross, listing.platform_fee_bps
        )
        ledger = MarketplaceLedgerEntry(
            entry_id=f"led_{uuid.uuid4().hex[:16]}",
            listing_ref=listing_ref,
            subscription_id=sub.subscription_id,
            source_tenant_id=listing.source_tenant_id,
            subscriber_tenant_id=subscriber_tenant_id,
            gross_cents=gross,
            platform_fee_cents=platform_fee_cents,
            provider_earning_cents=provider_earning_cents,
            currency=listing.currency,
            settlement_status=SettlementStatus.PENDING,
            usage_id=usage.usage_id,
            reason=kind,
            created_at_ms=now,
        )
        self._usage[usage.usage_id] = usage
        if idempotency_key:
            self._usage_idem[(listing_ref, idempotency_key)] = usage.usage_id
        self._ledger.append(ledger)
        return usage, ledger

    def query_ledger(
        self,
        *,
        listing_ref: str | None = None,
        subscriber_tenant_id: str | None = None,
        provider_tenant_id: str | None = None,
        settlement_status: SettlementStatus | None = None,
    ) -> list[MarketplaceLedgerEntry]:
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

    def settle_pending(
        self,
        *,
        provider_tenant_id: str | None = None,
        listing_ref: str | None = None,
    ) -> list[MarketplaceLedgerEntry]:
        settled: list[MarketplaceLedgerEntry] = []
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
                subscription_id=r.subscription_id,
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


def ensure_persona_market_store(app: web.Application) -> PersonaMarketStore:
    """Idempotently attach the in-memory market store to ``app``."""
    store = app.get(PERSONA_MARKET_STORE_KEY)
    if store is None:
        store = PersonaMarketStore()
        app[PERSONA_MARKET_STORE_KEY] = store
    return store


def _get_store(request: web.Request) -> PersonaMarketStore:
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
    if not str(body.get("display_name", "")).strip():
        return _error(400, "missing_display_name", "display_name is required")
    store = _get_store(request)
    persona_config = body.get("persona_config") or {}
    if not isinstance(persona_config, dict):
        persona_config = {}
    try:
        spec = store.publish(
            tenant_id=tenant.tenant_id,
            listing_ref=str(body.get("listing_ref", "")) or None,
            display_name=str(body.get("display_name", "")),
            persona_config=persona_config,
            source_company_id=str(body.get("source_company_id", "")),
            source_template_ref=str(body.get("source_template_ref", "")),
            vertical=str(body.get("vertical", "")),
            archetype=str(body.get("archetype", "")),
            summary=str(body.get("summary", "")),
            asset_bundle_hash=str(body.get("asset_bundle_hash", "")),
            price_model=str(
                body.get("price_model", PersonaPriceModel.MONTHLY_PER_COMPANY.value)
            ),
            price_cents=int(body.get("price_cents", 0) or 0),
            currency=str(body.get("currency", "USD")),
            platform_fee_bps=int(body.get("platform_fee_bps", 7000)),
            provider_share_bps=int(body.get("provider_share_bps", 3000)),
            license_scope=str(body.get("license_scope", "single_tenant_remint")),
            visibility=str(
                body.get("visibility", PersonaListingVisibility.PLATFORM.value)
            ),
            published_by=str(body.get("published_by", "")),
            payout_account_ref=str(body.get("payout_account_ref", "")),
        )
    except ValueError as exc:
        return _error(400, "invalid_enum", str(exc))
    except PersonaMarketError as exc:
        return _market_error_response(exc)
    return web.json_response({"status": "ok", "listing": spec.to_json()})


async def _handle_list_listings(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    store = _get_store(request)
    listings = store.available_for(tenant.tenant_id)
    return web.json_response(
        {"status": "ok", "listings": [l.to_json() for l in listings]}
    )


async def _handle_get_listing(request: web.Request) -> web.Response:
    await require_tenant_auth(request)
    store = _get_store(request)
    listing = store.get(request.match_info["listing_ref"])
    if listing is None:
        return _error(404, "not_found", request.match_info["listing_ref"])
    return web.json_response({"status": "ok", "listing": listing.to_json()})


async def _handle_patch_listing(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    store = _get_store(request)
    body = await request.json()
    if not isinstance(body, dict):
        return _error(400, "invalid_body", "expected a JSON object")
    fields: dict[str, Any] = {}
    for key in ("price_cents", "price_model", "summary", "payout_account_ref"):
        if key in body:
            fields[key] = body[key]
    if "status" in body:
        try:
            requested = PersonaListingStatus(str(body["status"]))
        except ValueError as exc:
            return _error(400, "invalid_enum", str(exc))
        if requested not in (
            PersonaListingStatus.DRAFT,
            PersonaListingStatus.PENDING_REVIEW,
            PersonaListingStatus.ACTIVE,
            PersonaListingStatus.DELISTED,
        ):
            return _error(403, "status_not_owner_settable", requested.value)
        fields["status"] = requested.value
    try:
        spec = store.update(
            tenant_id=tenant.tenant_id,
            listing_ref=request.match_info["listing_ref"],
            **fields,
        )
    except ValueError as exc:
        return _error(400, "invalid_enum", str(exc))
    except PersonaMarketError as exc:
        return _market_error_response(exc)
    return web.json_response({"status": "ok", "listing": spec.to_json()})


async def _handle_delist_listing(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    store = _get_store(request)
    try:
        spec = store.delist(
            tenant_id=tenant.tenant_id,
            listing_ref=request.match_info["listing_ref"],
        )
    except PersonaMarketError as exc:
        return _market_error_response(exc)
    return web.json_response({"status": "ok", "listing": spec.to_json()})


async def _handle_suspend_listing(request: web.Request) -> web.Response:
    require_control_plane_secret(request)
    store = _get_store(request)
    body = await request.json() if request.can_read_body else {}
    target = PersonaListingStatus.SUSPENDED
    if isinstance(body, dict) and body.get("status") == "active":
        target = PersonaListingStatus.ACTIVE
    try:
        spec = store.set_status(request.match_info["listing_ref"], target)
    except PersonaMarketError as exc:
        return _market_error_response(exc)
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
    try:
        spec = store.subscribe(
            subscriber_tenant_id=tenant.tenant_id,
            listing_ref=listing_ref,
            subscribed_by=str(body.get("subscribed_by", "")),
        )
    except PersonaMarketError as exc:
        return _market_error_response(exc)
    listing = store.get(listing_ref)
    remint = {
        "source_template_ref": listing.source_template_ref if listing else "",
        "asset_bundle_hash": listing.asset_bundle_hash if listing else "",
        "persona_config": dict(listing.persona_config) if listing else {},
        "license_scope": listing.license_scope if listing else "",
    }
    return web.json_response(
        {"status": "ok", "subscription": spec.to_json(), "remint": remint}
    )


async def _handle_list_subscriptions(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    store = _get_store(request)
    subs = store.subscriptions_for(tenant.tenant_id)
    return web.json_response(
        {"status": "ok", "subscriptions": [s.to_json() for s in subs]}
    )


# ---------------------------------------------------------------------------
# Handlers — usage + ledger + settlements
# ---------------------------------------------------------------------------


async def _handle_usage_event(request: web.Request) -> web.Response:
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
    idem = str(body.get("idempotency_key", "")).strip()
    dup = store.usage_for_idem(listing_ref, idem)
    if dup is not None:
        return web.json_response(
            {"status": "ok", "usage": dup.to_json(), "idempotent_replay": True}
        )
    try:
        usage, ledger = store.record_usage(
            listing_ref=listing_ref,
            subscriber_tenant_id=subscriber_tenant_id,
            kind=str(body.get("kind", "subscription_period")),
            quantity=int(body.get("quantity", 1) or 1),
            unit_price_cents=(
                int(body["unit_price_cents"])
                if "unit_price_cents" in body
                else None
            ),
            gross_cents=(
                int(body["gross_cents"]) if "gross_cents" in body else None
            ),
            outcome_ref=str(body.get("outcome_ref", "")),
            idempotency_key=idem,
        )
    except PersonaMarketError as exc:
        return _market_error_response(exc)
    return web.json_response(
        {"status": "ok", "usage": usage.to_json(), "ledger_entry": ledger.to_json()}
    )


async def _handle_get_ledger(request: web.Request) -> web.Response:
    is_control_plane = "X-Control-Plane-Secret" in request.headers
    store = _get_store(request)
    if is_control_plane:
        require_control_plane_secret(request)
        rows = store.query_ledger(
            listing_ref=request.query.get("listing_ref") or None,
            subscriber_tenant_id=request.query.get("subscriber_tenant_id") or None,
            provider_tenant_id=request.query.get("provider_tenant_id") or None,
        )
    else:
        tenant = await require_tenant_auth(request)
        role = request.query.get("role", "provider")
        if role == "subscriber":
            rows = store.query_ledger(subscriber_tenant_id=tenant.tenant_id)
        else:
            rows = store.query_ledger(provider_tenant_id=tenant.tenant_id)
    return web.json_response(
        {
            "status": "ok",
            "entries": [r.to_json() for r in rows],
            "totals": {
                "gross_cents": sum(r.gross_cents for r in rows),
                "platform_fee_cents": sum(r.platform_fee_cents for r in rows),
                "provider_earning_cents": sum(
                    r.provider_earning_cents for r in rows
                ),
            },
        }
    )


async def _handle_run_settlements(request: web.Request) -> web.Response:
    require_control_plane_secret(request)
    body = await request.json() if request.can_read_body else {}
    body = body if isinstance(body, dict) else {}
    store = _get_store(request)
    settled = store.settle_pending(
        provider_tenant_id=str(body.get("provider_tenant_id", "")) or None,
        listing_ref=str(body.get("listing_ref", "")) or None,
    )
    return web.json_response(
        {
            "status": "ok",
            "settled_count": len(settled),
            "provider_earning_cents": sum(r.provider_earning_cents for r in settled),
            "entries": [r.to_json() for r in settled],
        }
    )


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
    "PersonaMarketError",
    "PersonaMarketStore",
    "attach_persona_market_routes",
    "ensure_persona_market_store",
]
