# U10 Paper-Suite Closure And Claim Discipline

> Status: draft
> Last updated: 2026-04-25
> Parent: `docs/implementation/12_eta_paper_grade_uplift_plan.md`
> Primary claim target: stronger ETA/NL claims must be backed by repeated, reproducible, matched-control evidence bundles.

## Problem

U05-U09 create stronger mechanisms and environments. U10 defines when the repo is allowed to make stronger claims about them.

The goal is to prevent over-claiming:

- synthetic proof success is not real residual evidence
- dialogue PE improvement is not the same as ETA Internal RL proof
- one good seed is not paper-grade evidence
- module presence is not mechanism evidence

## Current Anchors

- `volvence_zero/agent/paper_suite.py`
- `volvence_zero/agent/eta_proof_benchmark.py`
- `volvence_zero/agent/dialogue_benchmark.py`
- `volvence_zero/evaluation/backbone.py`
- `docs/specs/evidence_program.md`
- `docs/specs/evaluation.md`
- `scripts/run_eta_paper_suite.sh`
- `.github/workflows/paper-suite-nightly.yml`
- `.github/workflows/paper-suite-release.yml`

## Claim Levels

### Level 1: Engineering Proof

Allowed claim:

> The repo contains executable ETA/NL proof surfaces with contracts, metrics, and tests.

Required:

- smoke benchmark runs
- reports are shaped correctly
- acceptance gates fail closed when required evidence is absent

### Level 2: Mechanism Evidence

Allowed claim:

> The mechanism contributes beyond matched controls in repo-native proof environments.

Required:

- matched controls
- scaffold ablations
- positive pairwise effect against best competing control
- mechanism metrics, not only terminal success

### Level 3: Real Residual Evidence

Allowed claim:

> z-space Internal RL improves sparse-reward behavior through real open-weight residual control.

Required:

- `transformers-open-weight` primary lane
- hook coverage and fallback rate within threshold
- positive replacement effect delta
- sparse-reward held-out success above best control

### Level 4: Paper-Grade Release Claim

Allowed claim:

> The ETA/NL system has paper-grade evidence for the named claim.

Required:

- paper-suite full tier
- repeated seeds
- interval summaries
- pairwise effect confidence interval above threshold
- raw rollout artifacts
- provenance
- claim verdict `retain`
- external-safe review packet when human legibility is part of the claim

## Claim Registry Updates

Add or formalize claim IDs:

- `claim_eta_scaffold_free_temporal_abstraction`
- `claim_eta_internal_rl_sparse_reward_advantage`
- `claim_eta_real_open_weight_residual_control`
- `claim_nl_slow_loop_improves_eta_fast_path`
- `claim_eta_nl_long_horizon_adaptation`
- `claim_external_human_legibility`

Each claim must define:

- required gates
- required environment tiers
- required profiles
- required artifacts
- minimum repeated-run tier
- verdict thresholds
- allowed wording when verdict is `retain`, `weak`, or `fail`

## Paper-Suite Artifacts

Every full-tier run should export:

- manifest
- provenance
- environment split registry
- reward taxonomy summary
- run summaries
- benchmark report
- backend report
- assessment report
- pairwise effects
- claim verdicts
- raw rollout sample or path to raw rollout bundle
- failure diagnosis
- evidence bundle

For human-facing claims:

- blinded packet
- internal unblinding key
- rating template
- rating aggregate

## Required Gates

Core gates:

- `sparse-reward-success`
- `scaffold-ablation-retention`
- `hard-causal-takeover`
- `real-residual-backend-fidelity`
- `abstract-action-reuse`
- `heldout-composition`
- `credit-alignment`
- `nl-slow-shapes-fast`
- `policy-update-evidence`
- `statistical-batch-evidence`
- `artifact-provenance-complete`

Each gate should publish:

- pass/fail
- evidence tuples
- best competing control
- threshold
- observed value
- failure mode

## Repeated-Run Discipline

Minimum tiers:

- `ci-smoke`: fast shape and fail-closed checks
- `paper-suite-small`: repeated seeds for development signal
- `paper-suite-full`: release-grade claim evidence

The full tier should require:

- fixed seed schedule
- fixed environment splits
- fixed matched-control set
- versioned manifest hash
- reported dependency/runtime fingerprint
- pairwise effect intervals

## Release Rules

The release gate should fail closed when:

- any required artifact is missing
- best-control gap is non-positive
- real residual claim lacks real residual backend evidence
- scaffold ablation collapses the mechanism
- takeover gate is absent or failed
- reward shaping leaked into primary sparse profile
- provenance is incomplete

The release gate may produce `weak` only when required artifacts exist but effect intervals or held-out results are not strong enough for `retain`.

## Implementation Tasks

1. Extend shared paper-suite claim registry with U05-U09 claim IDs.
2. Add environment/reward/takeover/backend/NL-slow metrics to evidence bundles.
3. Add pairwise effects against best competing control for all primary claims.
4. Export raw rollout references for release-tier suites.
5. Add fail-closed tests for missing backend evidence, missing takeover evidence, reward leakage, and absent provenance.
6. Update release workflow docs so “paper-grade” language requires U10 retained verdict.

## Exit Criteria

U10 is complete when:

- each paper-grade claim maps to gates, artifacts, and thresholds
- paper-suite full tier exports complete evidence bundles
- claim verdicts fail closed on missing evidence
- synthetic, real residual, dialogue, and human-legibility claims are separated
- release docs cannot use paper-grade wording without a retained verdict

## Rollback Boundary

U10 changes evidence and release discipline. It should not alter runtime learning behavior except by blocking over-broad claim verdicts or artifact promotion when evidence is missing.
