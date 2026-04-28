# vz-application

> Vertical adapter — application-domain owners

Application-side runtime owners that turn `DomainExperiencePackage` data into the kernel's `domain_knowledge / case_memory / strategy_playbook / boundary_policy / retrieval_policy / response_assembly / experience_consolidation / experience_fast_prior` snapshots.

## What it owns

| Slot | Module |
|---|---|
| `domain_knowledge` | `DomainKnowledgeModule` |
| `case_memory` | `CaseMemoryModule` |
| `strategy_playbook` | `StrategyPlaybookModule` |
| `boundary_policy` | `BoundaryPolicyModule` |
| `retrieval_policy` | `RetrievalPolicyModule` |
| `response_assembly` | `ResponseAssemblyModule` |
| `experience_consolidation` | `ExperienceConsolidationModule` |
| `experience_fast_prior` | `ExperienceFastPriorModule` |

Plus `ApplicationRareHeavyState`, `ApplicationDomainKnowledgeStore`, `ApplicationCaseMemoryStore`, the `DomainExperiencePackage` compiler, retrieval-readout strategy, and knowledge-channels helpers.

## How the cycle was broken

The frozen-dataclass **snapshot types** that `vz-cognition.evaluation` consumes (`BoundaryPolicySnapshot`, `CaseMemorySnapshot`, `DomainKnowledgeSnapshot`, `ExperienceFastPriorSnapshot`, `ResponseAssemblySnapshot`, `StrategyPlaybookSnapshot`, `ApplicationOutcomeAttribution`, `ApplicationSequencePayoff`) live in `volvence_zero.application_types` (vz-cognition wheel), not in this wheel. The application owners in this wheel re-import them and re-export from `application/runtime.py` for backward-compat, but they are **defined** in vz-cognition. This breaks the otherwise-unavoidable cycle:

```
Before:  vz-cognition.evaluation -> vz-application.application.runtime -> vz-cognition.dual_track  (cycle)
After:   vz-cognition.evaluation -> vz-cognition.application_types               (no cycle)
         vz-application.application.runtime -> vz-cognition.application_types    (linear)
         vz-application.application.runtime -> vz-cognition.dual_track           (linear)
```

## Hard limits

- These owners are **siblings** of `vz-memory`, not part of its internals.
- Application owners cannot directly mutate `vz-memory` or `vz-cognition`. They only emit proposals into `ModificationGate` (in `vz-cognition.credit`).
- A new vertical = a new `DomainExperiencePackage` (data-only) + optional new `lifeform-domain-*` wheel. The kernel does not change.
