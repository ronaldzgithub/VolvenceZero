"""Cross-tenant persona marketplace service (R14 identity exchange).

Authority for the persona marketplace: tenants publish `ready` personas
as listings and adopt each other's via subscriptions that carry explicit
:class:`PersonaProvenance` (source tenant + consent scope). The deploy
layer (digital-employee persona-factory) calls this surface to record
provenance rather than blind-cloning a template across tenants.

Storage: in-memory, keyed on the aiohttp ``web.Application`` — mirrors
the cognition store's pragmatic single-process model. Persistence is a
follow-up (it can reuse the Registry DB the same way Track 4 did for
cognition); the contract + tenancy + consent semantics are what this
packet pins.

Tenancy is enforced at the edge via ``require_tenant_auth``:

* publish / delist act on the *authenticated* tenant only.
* list returns platform-visible listings from any tenant + the caller's
  own company-visible listings.
* subscribe records a cross-tenant adoption with provenance; a tenant
  cannot subscribe to its own listing.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from aiohttp import web

from dlaas_platform_contracts import (
    PersonaListingSpec,
    PersonaListingStatus,
    PersonaListingVisibility,
    PersonaProvenance,
    PersonaSubscriptionSpec,
    PersonaSubscriptionStatus,
)
from dlaas_platform_registry import require_tenant_auth

PERSONA_MARKET_STORE_KEY = "dlaas_persona_market"

# Default consent scope a publish grants for cross-tenant adoption.
_DEFAULT_CONSENT_SCOPE = "cross_tenant_adopt"


class PersonaMarketError(Exception):
    """Raised by the store for caller-correctable conditions."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


class PersonaMarketStore:
    """In-memory authority for persona listings + subscriptions.

    All mutating methods take an explicit ``tenant_id`` so the store is
    unit-testable without the HTTP/auth layer; the route handlers
    resolve the authenticated tenant and forward it here.
    """

    def __init__(self) -> None:
        self._listings: dict[str, PersonaListingSpec] = {}
        self._subscriptions: dict[str, PersonaSubscriptionSpec] = {}

    # -- publish / delist (owner-scoped) ---------------------------------
    def publish(
        self,
        *,
        tenant_id: str,
        display_name: str,
        persona_config: dict[str, Any],
        vertical: str = "",
        archetype: str = "",
        summary: str = "",
        visibility: str = "platform",
        published_by: str = "",
        listing_ref: str | None = None,
    ) -> PersonaListingSpec:
        if not display_name.strip():
            raise PersonaMarketError("invalid", "display_name is required")
        ref = listing_ref or f"pl_{uuid.uuid4().hex[:16]}"
        existing = self._listings.get(ref)
        if existing is not None and existing.source_tenant_id != tenant_id:
            raise PersonaMarketError("forbidden", "not the listing owner")
        listing = PersonaListingSpec(
            listing_ref=ref,
            source_tenant_id=tenant_id,
            display_name=display_name.strip()[:200],
            persona_config=persona_config,
            vertical=vertical,
            archetype=archetype,
            summary=summary[:2000],
            visibility=PersonaListingVisibility(visibility),
            status=PersonaListingStatus.LISTED,
            published_by=published_by,
            created_at_ms=(existing.created_at_ms if existing else _now_ms()),
        )
        self._listings[ref] = listing
        return listing

    def delist(self, *, tenant_id: str, listing_ref: str) -> PersonaListingSpec:
        listing = self._listings.get(listing_ref)
        if listing is None:
            raise PersonaMarketError("not_found", "listing not found")
        if listing.source_tenant_id != tenant_id:
            raise PersonaMarketError("forbidden", "not the listing owner")
        from dataclasses import replace

        delisted = replace(
            listing,
            status=PersonaListingStatus.DELISTED,
            delisted_at_ms=_now_ms(),
        )
        self._listings[listing_ref] = delisted
        return delisted

    # -- read (tenant-scoped visibility) ---------------------------------
    def available_for(self, tenant_id: str) -> list[PersonaListingSpec]:
        out: list[PersonaListingSpec] = []
        for listing in self._listings.values():
            if listing.status is not PersonaListingStatus.LISTED:
                continue
            if listing.visibility is PersonaListingVisibility.PLATFORM:
                out.append(listing)
            elif listing.source_tenant_id == tenant_id:
                out.append(listing)
        out.sort(key=lambda l: l.created_at_ms, reverse=True)
        return out

    def get_listing(self, listing_ref: str) -> PersonaListingSpec | None:
        return self._listings.get(listing_ref)

    # -- subscribe (cross-tenant adoption with provenance) ---------------
    def subscribe(
        self,
        *,
        subscriber_tenant_id: str,
        listing_ref: str,
        subscribed_by: str = "",
        consent_scope: str = _DEFAULT_CONSENT_SCOPE,
    ) -> PersonaSubscriptionSpec:
        listing = self._listings.get(listing_ref)
        if listing is None or listing.status is not PersonaListingStatus.LISTED:
            raise PersonaMarketError("not_found", "listing not available")
        if listing.source_tenant_id == subscriber_tenant_id:
            raise PersonaMarketError(
                "self_subscribe", "cannot subscribe to your own listing"
            )
        # Idempotent per (listing, subscriber).
        for sub in self._subscriptions.values():
            if (
                sub.listing_ref == listing_ref
                and sub.subscriber_tenant_id == subscriber_tenant_id
            ):
                return sub
        now = _now_ms()
        provenance = PersonaProvenance(
            source_tenant_id=listing.source_tenant_id,
            source_listing_ref=listing.listing_ref,
            published_by=listing.published_by,
            consent_scope=consent_scope,
            adopted_at_ms=now,
        )
        sub = PersonaSubscriptionSpec(
            subscription_id=f"ps_{uuid.uuid4().hex[:16]}",
            listing_ref=listing_ref,
            subscriber_tenant_id=subscriber_tenant_id,
            provenance=provenance,
            status=PersonaSubscriptionStatus.SEEDED,
            subscribed_by=subscribed_by,
            created_at_ms=now,
        )
        self._subscriptions[sub.subscription_id] = sub
        return sub

    def subscriptions_for(
        self, subscriber_tenant_id: str
    ) -> list[PersonaSubscriptionSpec]:
        out = [
            s
            for s in self._subscriptions.values()
            if s.subscriber_tenant_id == subscriber_tenant_id
        ]
        out.sort(key=lambda s: s.created_at_ms, reverse=True)
        return out


def ensure_persona_market_store(app: web.Application) -> None:
    if PERSONA_MARKET_STORE_KEY not in app:
        app[PERSONA_MARKET_STORE_KEY] = PersonaMarketStore()


def _store(request: web.Request) -> PersonaMarketStore:
    return request.app[PERSONA_MARKET_STORE_KEY]


def attach_persona_market_routes(app: web.Application) -> None:
    """Wire the ``/dlaas/v1/persona-market/*`` exchange surface."""
    ensure_persona_market_store(app)
    app.router.add_post(
        "/dlaas/v1/persona-market/listings", _handle_publish
    )
    app.router.add_get(
        "/dlaas/v1/persona-market/listings", _handle_list
    )
    app.router.add_post(
        "/dlaas/v1/persona-market/listings/{listing_ref}/delist",
        _handle_delist,
    )
    app.router.add_post(
        "/dlaas/v1/persona-market/subscriptions", _handle_subscribe
    )
    app.router.add_get(
        "/dlaas/v1/persona-market/subscriptions", _handle_subscriptions
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _read_json(request: web.Request) -> dict[str, Any]:
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001 - malformed body
        return {}
    return data if isinstance(data, dict) else {}


def _market_error(err: PersonaMarketError) -> web.Response:
    status = {
        "invalid": 400,
        "not_found": 404,
        "forbidden": 403,
        "self_subscribe": 409,
    }.get(err.code, 400)
    return web.json_response(
        {"status": "error", "error": err.code, "detail": err.detail},
        status=status,
    )


async def _handle_publish(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    data = await _read_json(request)
    try:
        listing = _store(request).publish(
            tenant_id=tenant.tenant_id,
            display_name=str(data.get("display_name", "") or ""),
            persona_config=dict(data.get("persona_config") or {}),
            vertical=str(data.get("vertical", "") or ""),
            archetype=str(data.get("archetype", "") or ""),
            summary=str(data.get("summary", "") or ""),
            visibility=str(data.get("visibility", "platform") or "platform"),
            published_by=str(data.get("published_by", "") or ""),
            listing_ref=(str(data["listing_ref"]) if data.get("listing_ref") else None),
        )
    except PersonaMarketError as exc:
        return _market_error(exc)
    except ValueError as exc:
        return web.json_response(
            {"status": "error", "error": "invalid", "detail": str(exc)}, status=400
        )
    return web.json_response({"status": "ok", "listing": listing.to_json()})


async def _handle_list(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    listings = _store(request).available_for(tenant.tenant_id)
    return web.json_response(
        {
            "status": "ok",
            "count": len(listings),
            "items": [l.to_json() for l in listings],
        }
    )


async def _handle_delist(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    listing_ref = request.match_info["listing_ref"]
    try:
        listing = _store(request).delist(
            tenant_id=tenant.tenant_id, listing_ref=listing_ref
        )
    except PersonaMarketError as exc:
        return _market_error(exc)
    return web.json_response({"status": "ok", "listing": listing.to_json()})


async def _handle_subscribe(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    data = await _read_json(request)
    listing_ref = str(data.get("listing_ref", "") or "")
    if not listing_ref:
        return web.json_response(
            {"status": "error", "error": "invalid", "detail": "listing_ref required"},
            status=400,
        )
    try:
        sub = _store(request).subscribe(
            subscriber_tenant_id=tenant.tenant_id,
            listing_ref=listing_ref,
            subscribed_by=str(data.get("subscribed_by", "") or ""),
            consent_scope=str(
                data.get("consent_scope", _DEFAULT_CONSENT_SCOPE)
                or _DEFAULT_CONSENT_SCOPE
            ),
        )
    except PersonaMarketError as exc:
        return _market_error(exc)
    listing = _store(request).get_listing(listing_ref)
    return web.json_response(
        {
            "status": "ok",
            "subscription": sub.to_json(),
            "listing": listing.to_json() if listing else None,
        }
    )


async def _handle_subscriptions(request: web.Request) -> web.Response:
    tenant = await require_tenant_auth(request)
    subs = _store(request).subscriptions_for(tenant.tenant_id)
    return web.json_response(
        {
            "status": "ok",
            "count": len(subs),
            "items": [s.to_json() for s in subs],
        }
    )


def _now_ms() -> int:
    return int(time.time() * 1000)
