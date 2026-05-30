"""Persona-market (template economy) platform contracts.

These are platform-owned, cross-tenant marketplace records. A company
that trained a high-performing AI promotes it into a reusable template,
lists it on the persona market with a price + license, and other
companies subscribe and re-mint it under their own tenant. Continued
usage is metered and split between the platform and the originating
("provider") company.

Design invariants (see docs/specs/persona_market):

* The marketplace is the economic SSOT: price, license, subscription
  entitlement, usage, and the revenue-split ledger live here, not in any
  single app BFF.
* A subscription never copies the source tenant's runtime state / scoped
  memory. It carries a ``PersonaProvenance`` + a re-mint instruction; the
  subscriber re-mints the template under its own tenant.
* Revenue defaults to a 70% platform / 30% provider split, expressed in
  basis points so it is auditable and overridable per listing.

This module imports no ``lifeform-*`` / kernel runtime; it only
describes marketplace state around hosted lives.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Revenue-split defaults (basis points; 10000 == 100%).
# ---------------------------------------------------------------------------

DEFAULT_PLATFORM_FEE_BPS = 7000
"""Platform keeps 70% of marketplace gross by default."""

DEFAULT_PROVIDER_SHARE_BPS = 3000
"""Originating company keeps 30% of marketplace gross by default."""

BPS_DENOMINATOR = 10000


def compute_revenue_split(
    gross_cents: int, platform_fee_bps: int = DEFAULT_PLATFORM_FEE_BPS
) -> tuple[int, int]:
    """Split ``gross_cents`` into ``(platform_fee_cents, provider_earning_cents)``.

    The platform fee is rounded to the nearest cent and the provider
    earning takes the remainder so the two always sum back to
    ``gross_cents`` exactly (no rounding leak). Negative gross (a refund
    / reversal) flows through with the same proportion so a correction
    entry mirrors the original split.
    """

    if platform_fee_bps < 0 or platform_fee_bps > BPS_DENOMINATOR:
        raise ValueError(
            f"platform_fee_bps must be within [0, {BPS_DENOMINATOR}]; "
            f"got {platform_fee_bps!r}"
        )
    # round-half-away-from-zero on the magnitude so refunds mirror charges.
    sign = -1 if gross_cents < 0 else 1
    magnitude = abs(gross_cents)
    platform_magnitude = (magnitude * platform_fee_bps + BPS_DENOMINATOR // 2) // BPS_DENOMINATOR
    platform_fee_cents = sign * platform_magnitude
    provider_earning_cents = gross_cents - platform_fee_cents
    return platform_fee_cents, provider_earning_cents


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PersonaListingStatus(str, Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELISTED = "delisted"
    REJECTED = "rejected"


class PersonaListingVisibility(str, Enum):
    # Visible only inside the publishing company (private catalogue).
    COMPANY = "company"
    # Visible to every company on the platform (the shared marketplace).
    PLATFORM = "platform"


class PersonaSubscriptionStatus(str, Enum):
    # Subscriber recorded entitlement but has not re-minted yet.
    SEEDED = "seeded"
    # Active paid entitlement.
    ACTIVE = "active"
    # Payment lapsed / listing delisted — bounded grace window.
    GRACE = "grace"
    # Subscriber cancelled.
    CANCELLED = "cancelled"
    # Platform froze the entitlement (e.g. listing suspended).
    SUSPENDED = "suspended"


class PersonaPriceModel(str, Enum):
    FREE = "free"
    MONTHLY_PER_COMPANY = "monthly_per_company"
    MONTHLY_PER_SEAT = "monthly_per_seat"
    PER_OUTCOME = "per_outcome"
    PER_CALL = "per_call"
    HYBRID = "hybrid"


class SettlementStatus(str, Enum):
    PENDING = "pending"
    SETTLED = "settled"
    REVERSED = "reversed"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonaProvenance:
    """Auditable lineage of a cross-tenant adoption."""

    source_tenant_id: str
    source_listing_ref: str
    published_by: str = ""
    consent_scope: str = "cross_tenant_adopt"
    adopted_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "source_tenant_id": self.source_tenant_id,
            "source_listing_ref": self.source_listing_ref,
            "published_by": self.published_by,
            "consent_scope": self.consent_scope,
            "adopted_at_ms": self.adopted_at_ms,
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "PersonaProvenance":
        # Provenance without a source tenant + listing is not auditable
        # lineage — fail loudly rather than silently store a blank record.
        source_tenant_id = str(data.get("source_tenant_id", "")).strip()
        source_listing_ref = str(data.get("source_listing_ref", "")).strip()
        if not source_tenant_id or not source_listing_ref:
            raise ValueError(
                "PersonaProvenance requires source_tenant_id and source_listing_ref"
            )
        return PersonaProvenance(
            source_tenant_id=source_tenant_id,
            source_listing_ref=source_listing_ref,
            published_by=str(data.get("published_by", "")),
            consent_scope=str(data.get("consent_scope", "cross_tenant_adopt")),
            adopted_at_ms=int(data.get("adopted_at_ms", 0) or 0),
        )


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonaListingSpec:
    """A priced, licensable persona/template listing on the market."""

    listing_ref: str
    source_tenant_id: str
    source_company_id: str = ""
    source_template_ref: str = ""
    display_name: str = ""
    vertical: str = ""
    archetype: str = ""
    summary: str = ""
    persona_config: Mapping[str, Any] = field(default_factory=dict)
    asset_bundle_hash: str = ""
    price_model: PersonaPriceModel = PersonaPriceModel.MONTHLY_PER_COMPANY
    price_cents: int = 0
    currency: str = "USD"
    platform_fee_bps: int = DEFAULT_PLATFORM_FEE_BPS
    provider_share_bps: int = DEFAULT_PROVIDER_SHARE_BPS
    license_scope: str = "single_tenant_remint"
    visibility: PersonaListingVisibility = PersonaListingVisibility.PLATFORM
    status: PersonaListingStatus = PersonaListingStatus.PENDING_REVIEW
    published_by: str = ""
    payout_account_ref: str = ""
    created_at_ms: int = 0
    updated_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "listing_ref": self.listing_ref,
            "source_tenant_id": self.source_tenant_id,
            "source_company_id": self.source_company_id,
            "source_template_ref": self.source_template_ref,
            "display_name": self.display_name,
            "vertical": self.vertical,
            "archetype": self.archetype,
            "summary": self.summary,
            "persona_config": dict(self.persona_config),
            "asset_bundle_hash": self.asset_bundle_hash,
            "price_model": self.price_model.value,
            "price_cents": self.price_cents,
            "currency": self.currency,
            "platform_fee_bps": self.platform_fee_bps,
            "provider_share_bps": self.provider_share_bps,
            "license_scope": self.license_scope,
            "visibility": self.visibility.value,
            "status": self.status.value,
            "published_by": self.published_by,
            "payout_account_ref": self.payout_account_ref,
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "PersonaListingSpec":
        persona_config = data.get("persona_config") or {}
        if not isinstance(persona_config, Mapping):
            persona_config = {}
        return PersonaListingSpec(
            listing_ref=str(data.get("listing_ref", "")),
            source_tenant_id=str(data.get("source_tenant_id", "")),
            source_company_id=str(data.get("source_company_id", "")),
            source_template_ref=str(data.get("source_template_ref", "")),
            display_name=str(data.get("display_name", "")),
            vertical=str(data.get("vertical", "")),
            archetype=str(data.get("archetype", "")),
            summary=str(data.get("summary", "")),
            persona_config=dict(persona_config),
            asset_bundle_hash=str(data.get("asset_bundle_hash", "")),
            price_model=PersonaPriceModel(
                str(data.get("price_model", PersonaPriceModel.MONTHLY_PER_COMPANY.value))
            ),
            price_cents=int(data.get("price_cents", 0) or 0),
            currency=str(data.get("currency", "USD")),
            platform_fee_bps=int(
                data.get("platform_fee_bps", DEFAULT_PLATFORM_FEE_BPS)
            ),
            provider_share_bps=int(
                data.get("provider_share_bps", DEFAULT_PROVIDER_SHARE_BPS)
            ),
            license_scope=str(data.get("license_scope", "single_tenant_remint")),
            visibility=PersonaListingVisibility(
                str(data.get("visibility", PersonaListingVisibility.PLATFORM.value))
            ),
            status=PersonaListingStatus(
                str(data.get("status", PersonaListingStatus.PENDING_REVIEW.value))
            ),
            published_by=str(data.get("published_by", "")),
            payout_account_ref=str(data.get("payout_account_ref", "")),
            created_at_ms=int(data.get("created_at_ms", 0) or 0),
            updated_at_ms=int(data.get("updated_at_ms", 0) or 0),
        )


# ---------------------------------------------------------------------------
# Subscription
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonaSubscriptionSpec:
    """A subscriber tenant's entitlement to a listing."""

    subscription_id: str
    listing_ref: str
    subscriber_tenant_id: str
    subscribed_by: str = ""
    entitlement_status: PersonaSubscriptionStatus = PersonaSubscriptionStatus.SEEDED
    reminted_template_ref: str = ""
    provenance: PersonaProvenance | None = None
    started_at_ms: int = 0
    current_period_end_ms: int = 0
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "listing_ref": self.listing_ref,
            "subscriber_tenant_id": self.subscriber_tenant_id,
            "subscribed_by": self.subscribed_by,
            "entitlement_status": self.entitlement_status.value,
            "reminted_template_ref": self.reminted_template_ref,
            "provenance": self.provenance.to_json() if self.provenance else None,
            "started_at_ms": self.started_at_ms,
            "current_period_end_ms": self.current_period_end_ms,
            "created_at_ms": self.created_at_ms,
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "PersonaSubscriptionSpec":
        prov = data.get("provenance")
        provenance = (
            PersonaProvenance.from_json(prov) if isinstance(prov, Mapping) else None
        )
        return PersonaSubscriptionSpec(
            subscription_id=str(data.get("subscription_id", "")),
            listing_ref=str(data.get("listing_ref", "")),
            subscriber_tenant_id=str(data.get("subscriber_tenant_id", "")),
            subscribed_by=str(data.get("subscribed_by", "")),
            entitlement_status=PersonaSubscriptionStatus(
                str(
                    data.get(
                        "entitlement_status", PersonaSubscriptionStatus.SEEDED.value
                    )
                )
            ),
            reminted_template_ref=str(data.get("reminted_template_ref", "")),
            provenance=provenance,
            started_at_ms=int(data.get("started_at_ms", 0) or 0),
            current_period_end_ms=int(data.get("current_period_end_ms", 0) or 0),
            created_at_ms=int(data.get("created_at_ms", 0) or 0),
        )


# ---------------------------------------------------------------------------
# Usage + Ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonaMarketUsageEvent:
    """A metered usage tick attributable to a listing/subscription.

    ``idempotency_key`` makes ingest replay-safe; the store rejects a
    duplicate key for the same listing so a retried POST never
    double-charges.
    """

    usage_id: str
    listing_ref: str
    subscriber_tenant_id: str
    kind: str = "subscription_period"
    quantity: int = 1
    unit_price_cents: int = 0
    gross_cents: int = 0
    outcome_ref: str = ""
    idempotency_key: str = ""
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "usage_id": self.usage_id,
            "listing_ref": self.listing_ref,
            "subscriber_tenant_id": self.subscriber_tenant_id,
            "kind": self.kind,
            "quantity": self.quantity,
            "unit_price_cents": self.unit_price_cents,
            "gross_cents": self.gross_cents,
            "outcome_ref": self.outcome_ref,
            "idempotency_key": self.idempotency_key,
            "created_at_ms": self.created_at_ms,
        }


@dataclass(frozen=True)
class MarketplaceLedgerEntry:
    """An immutable double-attribution ledger row (gross / fee / earning).

    Corrections never mutate an existing entry — a refund appends a new
    entry with negative amounts and ``settlement_status=reversed``.
    """

    entry_id: str
    listing_ref: str
    subscription_id: str
    source_tenant_id: str
    subscriber_tenant_id: str
    gross_cents: int
    platform_fee_cents: int
    provider_earning_cents: int
    currency: str = "USD"
    settlement_status: SettlementStatus = SettlementStatus.PENDING
    usage_id: str = ""
    reason: str = ""
    created_at_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "listing_ref": self.listing_ref,
            "subscription_id": self.subscription_id,
            "source_tenant_id": self.source_tenant_id,
            "subscriber_tenant_id": self.subscriber_tenant_id,
            "gross_cents": self.gross_cents,
            "platform_fee_cents": self.platform_fee_cents,
            "provider_earning_cents": self.provider_earning_cents,
            "currency": self.currency,
            "settlement_status": self.settlement_status.value,
            "usage_id": self.usage_id,
            "reason": self.reason,
            "created_at_ms": self.created_at_ms,
        }


__all__ = [
    "BPS_DENOMINATOR",
    "DEFAULT_PLATFORM_FEE_BPS",
    "DEFAULT_PROVIDER_SHARE_BPS",
    "MarketplaceLedgerEntry",
    "PersonaListingSpec",
    "PersonaListingStatus",
    "PersonaListingVisibility",
    "PersonaMarketUsageEvent",
    "PersonaPriceModel",
    "PersonaProvenance",
    "PersonaSubscriptionSpec",
    "PersonaSubscriptionStatus",
    "SettlementStatus",
    "compute_revenue_split",
]
