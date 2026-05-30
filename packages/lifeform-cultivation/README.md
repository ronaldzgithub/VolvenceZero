# lifeform-cultivation

Autonomous industry-expert cultivation ("行业专家自动养成"). Grows a default
system expert with **minimal human interaction**: an operator seeds a rough
persona, the engine researches the domain, and the expert converges onto a
**single coherent school of thought** before being inducted as a default
DLaaS expert.

## First-principles placement

A "school / 流派" is NOT a regime label and NOT raw ingested corpus. In
Volvence Zero it is the agent's **Behavior Protocol active-mixture**. So this
wheel does **not** implement a contradiction resolver — it re-homes
cultivation onto the kernel's existing machinery and only orchestrates intake
cadence + reads published readouts (R12). Conflicting theories are reconciled
by the protocol runtime:

- **Identity Core + boundary union** — hard-block anchor that resists drift
  from shallow inputs (the seed compiles into an ACTIVE `BehaviorProtocol`).
- **PE-utility soft-blend / arbitration** — competing theory-protocols earn
  their weight from prediction error; mutually-incompatible ones are dropped.
- **Slow reflection** (`protocol_reflection` / `protocol_revision_queue`,
  ACTIVE for the `cultivation.expert.v0` vertical) — decays / retires failing
  strategies so the mixture converges; the Identity Core stays fixed (R15
  rollback retained).

## Public surface

| Symbol | Role |
|---|---|
| `CultivationSeed` / `CultivationCurriculum` | operator-supplied rough persona + autonomous study schedule |
| `build_identity_core_protocol(seed)` | compile the seed into the ACTIVE Identity Core `BehaviorProtocol` |
| `CultivationEngine` | runs research → uptake-as-protocol → study → reflect; returns `CultivationProgress` |
| `CultivationSink` / `SessionCultivationSink` | kernel-facing edge (session + ingestion surfaces only); `uptake_protocol` routes researched text through the platform `ProtocolUptakeService`; `read_active_mixture` reads the school readout |
| `assess_protocol_coherence` / `ProtocolCoherenceAssessment` | school-convergence readout = active-mixture concentration (primary) |
| `assess_coherence` / `CoherenceAssessment` | legacy regime-concentration readout (degraded fallback only) |

## Boundary

Depends only on `lifeform-core`, `lifeform-ingestion`, and `vz-contracts`
(BehaviorProtocol schema). It never imports a `vz-*` kernel-owner module nor
the `dlaas-platform-*` tier. Cognition stays kernel-owned.

## Where it is wired

- Control-plane API: `POST/GET /dlaas/v1/cultivation*`
  (`dlaas-platform-api`, see `docs/specs/dlaas-api-v1.md` → Expert Cultivation).
- Web research: vz-bundle `search_web` / `fetch_webpage` (no new crawler).
- Operator console: `apps/dlaas-portal` → `/[locale]/cultivation`.
