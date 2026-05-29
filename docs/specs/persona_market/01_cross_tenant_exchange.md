# Persona Marketplace — Cross-Tenant Identity Exchange (R14)

Status: implemented (Full Version Track 2). Owner: `dlaas-platform-contracts`
(types) + `dlaas-platform-api` (service surface).

## Why

R14 says a regime identity is persistent and ownable. The persona
marketplace lets one tenant publish a `ready` persona and another adopt
it. The hard constraint: adoption must NOT silently clone a tenant's
DLaaS template across the isolation boundary. Instead a subscription
records **provenance + consent**, and the adopting tenant re-mints the
persona under its own tenant with an auditable lineage back to the
source.

## Contracts (`dlaas_platform_contracts.persona_market`)

- `PersonaListingSpec` — a published persona: `listing_ref`,
  `source_tenant_id`, `display_name`, opaque `persona_config` (the
  re-mint payload — NL brief + structured spec snapshot; the platform
  never interprets it), `vertical` / `archetype` / `summary`,
  `visibility` (`company` | `platform`), `status` (`listed` | `delisted`).
- `PersonaProvenance` — `source_tenant_id`, `source_listing_ref`,
  `published_by`, `consent_scope` (default `cross_tenant_adopt`),
  `adopted_at_ms`.
- `PersonaSubscriptionSpec` — `subscription_id`, `listing_ref`,
  `subscriber_tenant_id`, embedded `provenance`, `status`
  (`seeded` | `active` | `revoked`), `adopted_ref`.

All three are frozen dataclasses with `from_json` / `to_json`.

## Service surface (`dlaas_platform_api.persona_market`)

In-memory `PersonaMarketStore` (persistence is a follow-up; it can reuse
the Registry DB the same way Track 4 did for cognition). Routes, all
behind `require_tenant_auth` (`X-Tenant-Api-Key` + secret):

| Method + path | Effect |
| --- | --- |
| `POST /dlaas/v1/persona-market/listings` | publish/refresh a listing for the authenticated tenant |
| `GET /dlaas/v1/persona-market/listings` | platform-visible listings + caller's own company-visible |
| `POST /dlaas/v1/persona-market/listings/{ref}/delist` | owner delists |
| `POST /dlaas/v1/persona-market/subscriptions` | adopt a listing (records provenance + consent) |
| `GET /dlaas/v1/persona-market/subscriptions` | caller's subscriptions |

### Tenancy + consent rules

- publish / delist act on the authenticated tenant only (delist by a
  non-owner → `403 forbidden`).
- list visibility: `platform` listings are cross-tenant visible;
  `company` listings only to the source tenant.
- subscribe is cross-tenant; a tenant cannot subscribe to its own
  listing (`409 self_subscribe`); subscribing to a delisted listing →
  `404 not_found`. Idempotent per `(listing, subscriber)`.
- every subscription carries `PersonaProvenance` so the adopting tenant
  + audit can trace lineage and the consent scope under which the source
  authorised adoption.

## Deploy consumption

`apps/digital-employee` persona-factory `subscribeListing` calls the
exchange API to record the cross-tenant subscription with provenance,
then re-seeds the factory pipeline under the subscriber tenant (kept as
a fallback when the exchange API is unreachable). See deploy-side
`D-market-1`.
