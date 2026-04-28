# lifeform-domain-emogpt

Vertical for the **relationship-aware companion** archetype — what was historically called EmoGPT.

A vertical is data + light behavior glue that compiles into the kernel's owner snapshots:
- `DomainExperiencePackage` JSON → `vz-application.domain_knowledge` / `case_memory` / `strategy_playbook` / `boundary_policy`
- Regime priors → `vz-cognition.regime` warm-start
- Scenario packs → `vz-cognition.evaluation` benchmark inputs

Adding a new vertical (e.g. coding assistant, customer-service bot) is a new `lifeform-domain-*` package — kernel stays untouched. This is what proves trigger ② of `SPLIT.md` ("second consumer").
