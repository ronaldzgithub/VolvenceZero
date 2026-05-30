"""Tests for the cross-tenant persona marketplace (Track 2 FULL).

* Contract round-trip (`PersonaListingSpec` / `PersonaSubscriptionSpec`
  / `PersonaProvenance`).
* `PersonaMarketStore` publish / list-visibility / subscribe-with-
  provenance / self-subscribe + consent semantics.
"""

from __future__ import annotations

import pytest

from dlaas_platform_contracts import (
    PersonaListingSpec,
    PersonaProvenance,
    PersonaSubscriptionSpec,
)
from dlaas_platform_api.persona_market import (
    PersonaMarketError,
    PersonaMarketStore,
)


def test_listing_round_trip() -> None:
    spec = PersonaListingSpec.from_json(
        {
            "listing_ref": "pl_1",
            "source_tenant_id": "tenant-a",
            "display_name": "Sales coach",
            "persona_config": {"nl_brief": "be a sales coach"},
            "vertical": "sales",
            "visibility": "platform",
        }
    )
    again = PersonaListingSpec.from_json(spec.to_json())
    assert again == spec
    assert again.persona_config["nl_brief"] == "be a sales coach"


def test_subscription_round_trip_carries_provenance() -> None:
    sub = PersonaSubscriptionSpec.from_json(
        {
            "subscription_id": "ps_1",
            "listing_ref": "pl_1",
            "subscriber_tenant_id": "tenant-b",
            "provenance": {
                "source_tenant_id": "tenant-a",
                "source_listing_ref": "pl_1",
                "consent_scope": "cross_tenant_adopt",
                "adopted_at_ms": 123,
            },
        }
    )
    again = PersonaSubscriptionSpec.from_json(sub.to_json())
    assert again == sub
    assert again.provenance.source_tenant_id == "tenant-a"
    assert again.provenance.consent_scope == "cross_tenant_adopt"


def test_provenance_requires_source_fields() -> None:
    with pytest.raises(ValueError):
        PersonaProvenance.from_json({"source_tenant_id": "tenant-a"})


def test_publish_then_list_visibility() -> None:
    store = PersonaMarketStore()
    platform = store.publish(
        tenant_id="tenant-a",
        display_name="Public coach",
        persona_config={"nl_brief": "x"},
        visibility="platform",
    )
    store.publish(
        tenant_id="tenant-a",
        display_name="Private coach",
        persona_config={"nl_brief": "y"},
        visibility="company",
    )
    # tenant-b sees only the platform listing.
    b_view = store.available_for("tenant-b")
    refs_b = {l.listing_ref for l in b_view}
    assert platform.listing_ref in refs_b
    assert len(b_view) == 1
    # tenant-a sees both (its own company-visible + platform).
    assert len(store.available_for("tenant-a")) == 2


def test_subscribe_records_provenance_and_consent() -> None:
    store = PersonaMarketStore()
    listing = store.publish(
        tenant_id="tenant-a",
        display_name="Coach",
        persona_config={"nl_brief": "x"},
        published_by="alice",
    )
    sub = store.subscribe(
        subscriber_tenant_id="tenant-b",
        listing_ref=listing.listing_ref,
        subscribed_by="bob",
    )
    assert sub.subscriber_tenant_id == "tenant-b"
    assert sub.provenance.source_tenant_id == "tenant-a"
    assert sub.provenance.source_listing_ref == listing.listing_ref
    assert sub.provenance.published_by == "alice"
    assert sub.provenance.consent_scope == "cross_tenant_adopt"
    # Idempotent per (listing, subscriber).
    again = store.subscribe(
        subscriber_tenant_id="tenant-b", listing_ref=listing.listing_ref
    )
    assert again.subscription_id == sub.subscription_id


def test_cannot_self_subscribe() -> None:
    store = PersonaMarketStore()
    listing = store.publish(
        tenant_id="tenant-a", display_name="Coach", persona_config={}
    )
    with pytest.raises(PersonaMarketError) as exc:
        store.subscribe(
            subscriber_tenant_id="tenant-a", listing_ref=listing.listing_ref
        )
    assert exc.value.code == "self_subscribe"


def test_subscribe_to_delisted_is_not_found() -> None:
    store = PersonaMarketStore()
    listing = store.publish(
        tenant_id="tenant-a", display_name="Coach", persona_config={}
    )
    store.delist(tenant_id="tenant-a", listing_ref=listing.listing_ref)
    with pytest.raises(PersonaMarketError) as exc:
        store.subscribe(
            subscriber_tenant_id="tenant-b", listing_ref=listing.listing_ref
        )
    assert exc.value.code == "not_found"


def test_delist_requires_owner() -> None:
    store = PersonaMarketStore()
    listing = store.publish(
        tenant_id="tenant-a", display_name="Coach", persona_config={}
    )
    with pytest.raises(PersonaMarketError) as exc:
        store.delist(tenant_id="tenant-b", listing_ref=listing.listing_ref)
    assert exc.value.code == "forbidden"


# ---------------------------------------------------------------------------
# Template economy: usage metering, 70/30 split ledger, settlement.
# ---------------------------------------------------------------------------


def _subscribed_store() -> tuple[PersonaMarketStore, str]:
    store = PersonaMarketStore()
    listing = store.publish(
        tenant_id="tenant-a",
        display_name="Sales coach",
        persona_config={"nl_brief": "x"},
        price_cents=10_000,
        currency="USD",
    )
    store.subscribe(
        subscriber_tenant_id="tenant-b", listing_ref=listing.listing_ref
    )
    return store, listing.listing_ref


def test_usage_event_splits_70_30() -> None:
    store, listing_ref = _subscribed_store()
    usage, ledger = store.record_usage(
        listing_ref=listing_ref,
        subscriber_tenant_id="tenant-b",
        kind="subscription_period",
        quantity=1,
    )
    assert usage.gross_cents == 10_000
    assert ledger.gross_cents == 10_000
    assert ledger.platform_fee_cents == 7_000
    assert ledger.provider_earning_cents == 3_000
    # Split always reconstitutes gross exactly.
    assert ledger.platform_fee_cents + ledger.provider_earning_cents == 10_000
    assert ledger.source_tenant_id == "tenant-a"
    assert ledger.subscriber_tenant_id == "tenant-b"


def test_usage_requires_subscription() -> None:
    store = PersonaMarketStore()
    listing = store.publish(tenant_id="tenant-a", display_name="Coach")
    with pytest.raises(PersonaMarketError) as exc:
        store.record_usage(
            listing_ref=listing.listing_ref, subscriber_tenant_id="tenant-b"
        )
    assert exc.value.code == "not_subscribed"


def test_usage_idempotency_dedup() -> None:
    store, listing_ref = _subscribed_store()
    store.record_usage(
        listing_ref=listing_ref,
        subscriber_tenant_id="tenant-b",
        idempotency_key="period-2026-05",
    )
    dup = store.usage_for_idem(listing_ref, "period-2026-05")
    assert dup is not None
    # The ledger only carried one entry for the idempotent period.
    rows = store.query_ledger(listing_ref=listing_ref)
    assert len(rows) == 1


def test_ledger_views_and_settlement() -> None:
    store, listing_ref = _subscribed_store()
    store.record_usage(
        listing_ref=listing_ref, subscriber_tenant_id="tenant-b", quantity=1
    )
    store.record_usage(
        listing_ref=listing_ref, subscriber_tenant_id="tenant-b", quantity=2
    )
    provider_rows = store.query_ledger(provider_tenant_id="tenant-a")
    subscriber_rows = store.query_ledger(subscriber_tenant_id="tenant-b")
    assert len(provider_rows) == 2
    assert len(subscriber_rows) == 2
    # 10000 + 20000 gross -> provider 3000 + 6000 = 9000.
    assert sum(r.provider_earning_cents for r in provider_rows) == 9_000

    settled = store.settle_pending(provider_tenant_id="tenant-a")
    assert len(settled) == 2
    # Re-running settles nothing (idempotent on already-settled rows).
    assert store.settle_pending(provider_tenant_id="tenant-a") == []


def test_refund_mirrors_split() -> None:
    store, listing_ref = _subscribed_store()
    # A correction posts a negative gross; the split mirrors the charge.
    _usage, ledger = store.record_usage(
        listing_ref=listing_ref,
        subscriber_tenant_id="tenant-b",
        kind="refund",
        gross_cents=-10_000,
    )
    assert ledger.platform_fee_cents == -7_000
    assert ledger.provider_earning_cents == -3_000
