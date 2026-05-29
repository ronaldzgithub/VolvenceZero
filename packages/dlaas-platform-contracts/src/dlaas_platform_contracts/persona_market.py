"""Cross-tenant persona marketplace contracts (R14 identity exchange).

These types are the canonical wire shape for the DLaaS persona
marketplace: a tenant publishes a `ready` persona as a
:class:`PersonaListingSpec`, and another tenant adopts it via a
:class:`PersonaSubscriptionSpec` that carries explicit
:class:`PersonaProvenance` (who published it, under what consent scope).

R14 (persistent regime identity) + tenancy: adoption never silently
clones a tenant's DLaaS template across the isolation boundary. The
subscription records *provenance + consent* so the adopting tenant
re-mints the persona under its own tenant with an auditable lineage
back to the source, rather than a blind re-seed. The deploy layer
(digital-employee persona-factory) consumes these contracts; the
platform-api persona-market service is the authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PersonaListingVisibility(str, Enum):
    COMPANY = "company"
    PLATFORM = "platform"


class PersonaListingStatus(str, Enum):
    LISTED = "listed"
    DELISTED = "delisted"


class PersonaSubscriptionStatus(str, Enum):
    SEEDED = "seeded"
    ACTIVE = "active"
    REVOKED = "revoked"


@dataclass(frozen=True)
class PersonaProvenance:
    """Auditable lineage of an adopted persona.

    Travels with every :class:`PersonaSubscriptionSpec` so the adopting
    tenant (and any audit) can answer "where did this persona come from,
    and under what consent did we adopt it?".
    """

    source_tenant_id: str
    source_listing_ref: str
    published_by: str = ""
    # Consent scope under which the source tenant authorised adoption,
    # e.g. "cross_tenant_adopt". Empty = no explicit consent recorded.
    consent_scope: str = ""
    adopted_at_ms: int = 0

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "PersonaProvenance":
        if not isinstance(data, Mapping):
            raise ValueError("PersonaProvenance payload must be a JSON object")
        source_tenant_id = str(data.get("source_tenant_id", "") or "")
        source_listing_ref = str(data.get("source_listing_ref", "") or "")
        if not source_tenant_id or not source_listing_ref:
            raise ValueError(
                "PersonaProvenance requires source_tenant_id + source_listing_ref"
            )
        return cls(
            source_tenant_id=source_tenant_id,
            source_listing_ref=source_listing_ref,
            published_by=str(data.get("published_by", "") or ""),
            consent_scope=str(data.get("consent_scope", "") or ""),
            adopted_at_ms=int(data.get("adopted_at_ms", 0) or 0),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "source_tenant_id": self.source_tenant_id,
            "source_listing_ref": self.source_listing_ref,
            "published_by": self.published_by,
            "consent_scope": self.consent_scope,
            "adopted_at_ms": self.adopted_at_ms,
        }


@dataclass(frozen=True)
class PersonaListingSpec:
    """A persona published to the marketplace by its source tenant.

    ``persona_config`` is the opaque re-mint payload (the NL brief +
    structured spec snapshot the deploy factory uses); the platform
    never interprets it, only stores + serves it tenant-scoped.
    """

    listing_ref: str
    source_tenant_id: str
    display_name: str
    persona_config: Mapping[str, Any] = field(default_factory=dict)
    vertical: str = ""
    archetype: str = ""
    summary: str = ""
    visibility: PersonaListingVisibility = PersonaListingVisibility.PLATFORM
    status: PersonaListingStatus = PersonaListingStatus.LISTED
    published_by: str = ""
    created_at_ms: int = 0
    delisted_at_ms: int = 0

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "PersonaListingSpec":
        if not isinstance(data, Mapping):
            raise ValueError("PersonaListingSpec payload must be a JSON object")
        listing_ref = str(data.get("listing_ref", "") or "")
        source_tenant_id = str(data.get("source_tenant_id", "") or "")
        display_name = str(data.get("display_name", "") or "")
        if not listing_ref or not source_tenant_id or not display_name:
            raise ValueError(
                "PersonaListingSpec requires listing_ref + source_tenant_id "
                "+ display_name"
            )
        return cls(
            listing_ref=listing_ref,
            source_tenant_id=source_tenant_id,
            display_name=display_name,
            persona_config=dict(data.get("persona_config") or {}),
            vertical=str(data.get("vertical", "") or ""),
            archetype=str(data.get("archetype", "") or ""),
            summary=str(data.get("summary", "") or ""),
            visibility=PersonaListingVisibility(
                str(data.get("visibility", "platform") or "platform")
            ),
            status=PersonaListingStatus(
                str(data.get("status", "listed") or "listed")
            ),
            published_by=str(data.get("published_by", "") or ""),
            created_at_ms=int(data.get("created_at_ms", 0) or 0),
            delisted_at_ms=int(data.get("delisted_at_ms", 0) or 0),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "listing_ref": self.listing_ref,
            "source_tenant_id": self.source_tenant_id,
            "display_name": self.display_name,
            "persona_config": dict(self.persona_config),
            "vertical": self.vertical,
            "archetype": self.archetype,
            "summary": self.summary,
            "visibility": self.visibility.value,
            "status": self.status.value,
            "published_by": self.published_by,
            "created_at_ms": self.created_at_ms,
            "delisted_at_ms": self.delisted_at_ms,
        }


@dataclass(frozen=True)
class PersonaSubscriptionSpec:
    """A tenant's adoption of a marketplace listing, with provenance."""

    subscription_id: str
    listing_ref: str
    subscriber_tenant_id: str
    provenance: PersonaProvenance
    status: PersonaSubscriptionStatus = PersonaSubscriptionStatus.SEEDED
    subscribed_by: str = ""
    adopted_ref: str = ""
    created_at_ms: int = 0

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "PersonaSubscriptionSpec":
        if not isinstance(data, Mapping):
            raise ValueError("PersonaSubscriptionSpec payload must be a JSON object")
        subscription_id = str(data.get("subscription_id", "") or "")
        listing_ref = str(data.get("listing_ref", "") or "")
        subscriber_tenant_id = str(data.get("subscriber_tenant_id", "") or "")
        if not subscription_id or not listing_ref or not subscriber_tenant_id:
            raise ValueError(
                "PersonaSubscriptionSpec requires subscription_id + listing_ref "
                "+ subscriber_tenant_id"
            )
        return cls(
            subscription_id=subscription_id,
            listing_ref=listing_ref,
            subscriber_tenant_id=subscriber_tenant_id,
            provenance=PersonaProvenance.from_json(data.get("provenance") or {}),
            status=PersonaSubscriptionStatus(
                str(data.get("status", "seeded") or "seeded")
            ),
            subscribed_by=str(data.get("subscribed_by", "") or ""),
            adopted_ref=str(data.get("adopted_ref", "") or ""),
            created_at_ms=int(data.get("created_at_ms", 0) or 0),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "listing_ref": self.listing_ref,
            "subscriber_tenant_id": self.subscriber_tenant_id,
            "provenance": self.provenance.to_json(),
            "status": self.status.value,
            "subscribed_by": self.subscribed_by,
            "adopted_ref": self.adopted_ref,
            "created_at_ms": self.created_at_ms,
        }


__all__ = [
    "PersonaListingSpec",
    "PersonaListingStatus",
    "PersonaListingVisibility",
    "PersonaProvenance",
    "PersonaSubscriptionSpec",
    "PersonaSubscriptionStatus",
]
