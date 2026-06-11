# Persona Market — Template Economy

Status: spec for the cross-tenant AI template economy. The persona
market is the platform-tier economic SSOT that turns a company-trained
AI into a licensable, metered, revenue-split asset.

## Why

An employee trains a digital-employee twin until it performs well. The
company promotes that performance into a reusable **template asset**,
lists it on the market with a price + license, and other companies
**subscribe** and re-mint it in their own tenant. Continued usage is
metered and split: by default **70% platform / 30% originating company**.

## First-principles invariants

1. **Asset, not memory copy.** A listing carries a sanitised template
   asset bundle (SOPs, persona spec, tool contract, evaluation samples)
   plus an `asset_bundle_hash`. It never carries the source tenant's
   scoped memory, an employee's private habits, or customer PII.
2. **Re-mint, not share.** A subscriber receives a `PersonaProvenance` +
   re-mint instruction and re-creates the template under its own DLaaS
   tenant. No cross-tenant `ai_id` or runtime state is shared.
3. **Marketplace is the economic SSOT.** Price, license, entitlement,
   usage, and the revenue-split ledger live in the platform
   (`dlaas-platform-api` persona-market store), not in any app BFF.
   Apps (digital-employee, dlaas-portal) mirror, never own, this state.
4. **Revenue from continued use.** Listing → subscription is not the
   end. Subscription periods, per-seat, per-outcome, and per-call usage
   are all attributable to a `listing_ref` and produce ledger entries.
5. **Auditable + reversible.** Ledger entries are immutable; a refund or
   dispute appends a negative correction entry with the same split, it
   never edits history. Listings can be delisted/suspended; existing
   subscriptions move to `grace`/`suspended`.

## Contracts (`dlaas_platform_contracts.persona_market`)

- `PersonaListingSpec` — listing identity, source tenant/company,
  `source_template_ref`, `asset_bundle_hash`, `price_model`,
  `price_cents`, `currency`, `platform_fee_bps` (default 7000),
  `provider_share_bps` (default 3000), `license_scope`, `visibility`,
  `status`, `payout_account_ref`.
- `PersonaSubscriptionSpec` — `subscription_ref`, `listing_ref`,
  `subscriber_tenant_id`, `entitlement_status`, `reminted_template_ref`,
  `provenance`, period fields.
- `PersonaMarketUsageEvent` — metered tick: `kind`, `quantity`,
  `unit_price_cents`, `gross_cents`, `outcome_ref`, `idempotency_key`.
- `MarketplaceLedgerEntry` — `gross_cents`, `platform_fee_cents`,
  `provider_earning_cents`, `settlement_status`.
- `compute_revenue_split(gross_cents, platform_fee_bps)` — the single
  rounding rule (platform rounds half-away-from-zero; provider takes the
  remainder so the two always sum back to gross).

## Status machines

Listing: `draft → pending_review → active → {suspended, delisted}` (or
`pending_review → rejected`).

Subscription: `seeded → active → {grace, cancelled, suspended}`.

Ledger settlement: `pending → settled` (or `→ reversed` for corrections).

## API (`/dlaas/v1/persona-market/*`)

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | `/listings` | tenant | publish/refresh a listing (source tenant) |
| GET | `/listings` | tenant | browse active platform listings |
| GET | `/listings/{ref}` | tenant | listing detail |
| GET | `/listings/{ref}/bundle` | tenant (owner or live subscriber) | fetch the full licensed asset bundle |
| PATCH | `/listings/{ref}` | tenant (owner) | update price/status (owner only) |
| POST | `/listings/{ref}/delist` | tenant (owner) | delist |
| POST | `/listings/{ref}/suspend` | control-plane | operator suspend |
| POST | `/subscriptions` | tenant | subscribe (returns provenance + remint instr.) |
| GET | `/subscriptions` | tenant | subscriber's entitlements |
| POST | `/usage-events` | tenant or control-plane | meter usage → ledger entry |
| GET | `/ledger` | tenant or control-plane | ledger (filter by listing/subscriber/provider) |
| POST | `/settlements/run` | control-plane | mark a batch settled |

## Licensed bundle release

A publish MAY carry the full sanitised asset bundle as an opaque
`asset_bundle` JSON object alongside `asset_bundle_hash`. The platform
stores the bundle bound to the hash it was published with, OUTSIDE the
listing spec — browse (`GET /listings`) and detail (`GET
/listings/{ref}`) responses never include it, so the licensed content
is not exposed to unsubscribed tenants.

`GET /listings/{ref}/bundle` is the single release point:

- Caller must be the listing owner, or hold a subscription whose
  entitlement is live (`seeded` / `active` / `grace`). Otherwise
  `403 not_licensed`. A suspended listing blocks non-owner fetches
  (`409 suspended`).
- Listing published without a bundle (hash-only, e.g. an older
  producer) → `404 bundle_not_available`; the consumer decides its own
  fallback and must surface it, never silently.
- The bundle's publish-time hash is re-verified against the listing's
  current `asset_bundle_hash`; a re-publish that changed the hash
  without a fresh bundle yields a loud `409 bundle_hash_mismatch` —
  the platform never releases a stale or drifted licensed asset.
- Response: `{ status, listing_ref, asset_bundle, asset_bundle_hash,
  license_scope }`.

## Revenue split

```
gross_cents = unit_price_cents * quantity
platform_fee_cents, provider_earning_cents = compute_revenue_split(gross_cents, platform_fee_bps)
# default platform_fee_bps = 7000 -> 70% platform / 30% provider
```

## Economy extension

digital-employee is the first producer/consumer, but the market is
vertical-agnostic: any DLaaS vertical can publish a
`PersonaListingSpec`. The market becomes the platform-wide template /
capability asset exchange layer.
