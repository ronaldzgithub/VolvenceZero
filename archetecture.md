# VolvenceZero Architecture Boundary Charter

> Status: active architecture entry
> Last updated: 2026-05-02

This document is the stable entry point for the code boundary story referenced by
`docs/specs/00_INDEX.md`, `docs/DATA_CONTRACT.md`, `docs/SYSTEM_DESIGN.md`, and
`SPLIT.md`. The filename keeps the historical spelling used by those documents.

## First Principles

VolvenceZero is a layered adaptive organism, not a prompt stack. The architecture
is intentionally organized around a few invariants:

- **Prediction error first**: `prediction_error` is the primitive learning signal;
  `credit` and `evaluation` are downstream readout / aggregation layers.
- **Snapshot-only exchange**: runtime owners publish immutable snapshots; consumers
  read those snapshots and do not call owner internals or rebuild owner summaries.
- **Stable substrate, adaptive controllers**: the base model remains slow or
  frozen; online adaptation happens in bounded controller / memory / credit layers.
- **Latent control over token control**: long-running behavior is learned in
  temporal controller state (`z_t`, `beta_t`) rather than through surface-text
  keyword rules.
- **Rollback by construction**: new owners and fields enter through
  `WiringLevel.DISABLED` / `SHADOW` / `ACTIVE`, with contract tests and evidence
  gates before widening their consumer radius.

## Current Wheel Boundary

The Phase 1 repository shape is one monorepo with multiple wheels. Repository
boundary is not the module boundary; wheel boundary is.

| Wheel | Owns | Boundary rule |
|---|---|---|
| `vz-contracts` | `Snapshot`, `RuntimeModule`, `WiringLevel`, shared guards | Foundation; no upstream `volvence_zero.*` imports |
| `vz-substrate` | frozen LLM substrate, residual capture, bounded adapter-delta entry | No policy ownership |
| `vz-memory` | continuum memory, CMS state, reflection-backed memory readouts | Publishes memory summaries and stable memory readouts |
| `vz-cognition` | PE, credit, dual-track, regime, semantic owners, evaluation, social cognition types | Current home for PE / credit / self-model capabilities |
| `vz-application` | domain knowledge, case memory, playbook, boundary policy, retrieval / assembly owners | Consumes cognition and memory snapshots; does not own kernel state |
| `vz-temporal` | metacontroller, `beta_t` segment closure, internal RL on `z_t` | Owns temporal abstraction and controller state |
| `vz-runtime` | orchestration and stable `Brain` / `BrainSession` facade | The only kernel wheel allowed to compose all business wheels |
| `lifeform-*` | product / lifeform adapters, vitals, expression, vertical packages, services, evolution loops | May depend on kernel facade and contracts; kernel must not import lifeform |
| `lifeform-domain-figure` | real-person digital revival vertical: `HistoricalFigureProfile` + corpus ingest + `FigureArtifactBundle` (retrieval index / coverage map / style prior / steering / persona LoRA) | Parallel to `lifeform-domain-character`; never imports it. Compiles into existing application owners; rare-heavy artifacts (steering / LoRA) gated through `ModificationGate.OFFLINE` |
| `lifeform-domain-growth-advisor` | long-term private-domain growth-advisor companion (LTV path): `GrowthAdvisorProfile` (谌老师 reference) compiles to existing `DomainExperiencePackage` + `VitalsBootstrap`, encodes 7-day onboarding playbook + 4 need-mining funnels + 4 anti-sales boundaries (`bp-no-hard-sell` / `bp-no-overclaim` / `bp-no-flooding` / `bp-no-judgmental`) | Parallel to `lifeform-domain-character` / `lifeform-domain-figure`; never imports either. No new kernel owner; behaviour differences across the 7 days reach the kernel through `applicability_scope` (e.g. `growth_advisor:day1` ... `growth_advisor:day7`) and `regime_tags` carried by typed compiled records, never by user-text grep |
| `dlaas-platform-contracts` | DLaaS typed dataclass: `InteractionEnvelope` / `OutputAct` / `TenantSpec` / `ShellSpec` / `AssetSpec` / `TemplateSpec` / `ContractSpec` / `FocusPersonSpec` / `IdentityLinkSpec` / `HandoffTicketSpec` | Foundation for the platform tier; zero `vz-*` / `lifeform-*` imports |
| `dlaas-platform-registry` | Multi-tenant resource SSOT (tenants / shells / assets / templates / contracts / focus_persons / identity_links); SQLite-backed CRUD + auth | Talks to no kernel; surfaces `tenant_state` / `contract_state` |
| `dlaas-platform-launcher` | `InstanceManager`: `{ai_id → Lifeform}`, shared substrate, awake/sleep/LRU; surfaces `instance_status` | Composes `lifeform-core.Lifeform` facade + `lifeform-service.SessionManager`; never imports kernel internals |
| `dlaas-platform-api` | aiohttp `/dlaas/*` router + three auth-header middleware (`X-Tenant-Api-Key/Secret`, `X-Control-Plane-Secret`, `X-Service-Secret`); typed `InteractionEnvelope` dispatch + `OutputAct` packaging | Pure HTTP boundary; no cognitive state |
| `dlaas-platform-ops` | pause / resume / operator-message / handoff queue / SSE conversations stream; ledger; surfaces `handoff_ticket_state` | Reads `rupture_state` snapshot via `lifeform-service` to drive handoff; never adds kernel owners |
| `dlaas-platform-eval` | audience analysis / exam questions+runs / launch license gate; LLM judge as readout only | Reuses `lifeform-evolution.closed_alpha_preflight` framework; never writes kernel learning state |

Historical capability names such as `vz-pe-credit`, `vz-self-model`, or
`vz-evaluation` are not current wheel names. In this repository they are owned by
`vz-cognition` subpackages (`prediction`, `credit`, `dual_track`, `regime`,
`semantic_state`, `evaluation`). If a future split extracts them, the first step is
to update `docs/DATA_CONTRACT.md`, `tests/contracts/test_import_boundaries.py`, and
each affected `pyproject.toml` in the same change.

## Split Axes

Each boundary exists to protect one invariant, not to mirror a directory layout:

- R2: `vz-substrate` isolates rare-heavy substrate refresh from online control.
- R3/R4: `vz-temporal` owns latent temporal control and abstract actions.
- R5/R6: `vz-memory` owns memory strata and their summaries.
- R-PE/R9/R10/R11/R12/R14: `vz-cognition` owns PE, credit, semantic state,
  evaluation readouts, dual-track and regime identity.
- R8/R15: `vz-contracts` and the contract tests make cross-wheel exchange explicit.
- Product variability stays in `lifeform-*` and vertical packages, not in kernel
  owners.
- DLaaS multi-tenant governance + runtime envelope translation + ops + eval gate
  stay in `dlaas-platform-*` (third tier); they never become kernel owners and
  never add cognitive state. See `docs/specs/dlaas-platform.md`.

## Migration Rules

1. Add or change a runtime slot only through `docs/DATA_CONTRACT.md` and the owning
   spec under `docs/specs/`.
2. Declare every cross-wheel import in both the consumer `pyproject.toml` and
   `tests/contracts/test_import_boundaries.py`.
3. Prefer enriching the publishing owner snapshot over adding runtime-side
   reconstruction logic.
4. Keep benchmark and evaluation gates tied to structured snapshots or artifacts;
   free-text heuristics may exist only as explicitly local diagnostics.
5. Move a wheel boundary only when `SPLIT.md` trigger conditions and evidence gates
   justify the cost.

## Document Map

- `docs/specs/00_INDEX.md` is the first stop for capability-domain routing.
- `docs/DATA_CONTRACT.md` is the slot and snapshot contract registry.
- `docs/SYSTEM_DESIGN.md` explains the current data flow and implementation shape.
- `SPLIT.md` defines repository split timing and mechanics.
- `docs/next_gen_emogpt.md` is the design source for R-IDs and NL / ETA rationale.
- `docs/specs/rupture-and-repair.md` defines the v0 `rupture_state` owner and the `dialogue_external_outcome` snapshot channel (single legal path for external outcomes into the kernel).
