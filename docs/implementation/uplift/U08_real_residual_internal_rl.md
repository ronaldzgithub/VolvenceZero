# U08 Real Residual Internal RL

> Status: draft
> Last updated: 2026-04-25
> Parent: `docs/implementation/12_eta_paper_grade_uplift_plan.md`
> Primary claim target: z-space Internal RL should improve sparse-reward performance through real residual control over a frozen open-weight model.

## Problem

The current ETA proof harness can compare Internal RL profiles and includes open-weight residual entrypoints. U08 turns that into a hard evidence path for real residual-control claims.

The key distinction:

- synthetic/trace backends can support engineering and mechanism proof
- `transformers-open-weight` evidence is required for real residual-control claims

## Current Anchors

- `volvence_zero/substrate/`
- `volvence_zero/internal_rl/environment.py`
- `volvence_zero/internal_rl/sandbox.py`
- `volvence_zero/agent/eta_proof_benchmark.py`
- `volvence_zero/integration/final_wiring.py`
- `docs/implementation/07_real_substrate_calibration_report.md`
- `docs/implementation/11_eta_internal_rl_strong_proof_harness.md`

## Environment Definition

For paper-grade ETA Internal RL, the environment must be defined as:

```text
external task environment
+ frozen autoregressive model
+ residual capture backend
+ z_t -> decoder -> residual controller
+ switch unit / persistence logic
```

Where:

- observation is residual activation or a declared residual-derived summary
- action is `z_t`, not tokens
- decoder output becomes residual intervention `e_t,l <- e_t,l + U_t e_t,l`
- reward is sparse terminal/delayed reward from U05 environments
- the base model is frozen in the live/runtime path

## Backend Evidence Requirements

Every real residual run must report:

- backend label
- model ID
- hook layer tuple
- capture success rate
- hook coverage
- fallback rate
- intervention application count
- replacement effect delta
- residual signal quality
- runtime origin

A real residual-control claim fails closed if:

- fallback rate is above threshold
- hook coverage is absent or too low
- replacement effect delta is not positive against controls
- the benchmark silently substitutes trace/synthetic backend for the primary real lane

## Matched Controls

Minimum profiles:

- `full-internal-rl`
- `full-no-optimize`
- `full-no-replacement`
- `learned-lite-causal`
- `noop-backend`
- `full-no-fast-prior`
- `trace-only`
- `synthetic-open-weight`

The real lane should compare against both mechanism controls and backend controls.

Required comparisons:

- `full-internal-rl` vs best mechanism control
- `transformers-open-weight` vs `noop-backend`
- `transformers-open-weight` vs `trace-only`
- `transformers-open-weight` vs `synthetic-open-weight`

## Multi-Step Persistence

An abstract action must affect more than one environment/generation step in the primary real residual profile.

Report:

- `mean_steps_per_abstract_action`
- `median_steps_per_abstract_action`
- `persistence_window_success_rate`
- `premature_switch_rate`
- `always_switch_rate`
- `never_switch_rate`

The claim should fail if success depends only on one-step z replacement.

## Sparse Reward Constraint

The primary real residual profile must optimize only terminal/delayed reward.

Allowed:

- terminal task success
- delayed credit assignment over abstract-action windows
- diagnostic readouts excluded from optimizer

Not allowed in primary profile:

- dense per-step shaping reward
- semantic-label bonus
- profile-specific route hints

Dense reward may remain as an ablation to explain learning dynamics.

## Implementation Tasks

1. Make `ETAOpenWeightRuntimeConfig` and open-weight backend settings explicit in paper-suite manifests.
2. Add fail-closed validation for fallback rate, hook coverage, and replacement effect delta.
3. Ensure proof rollouts can run with real residual snapshots for U05 sparse-reward environments.
4. Add multi-step persistence metrics to rollout reports.
5. Add backend-control comparisons to paper-suite pairwise effects.
6. Add tests proving synthetic success cannot satisfy real residual-control claims.

## Exit Criteria

U08 is complete when:

- real residual paper-suite runs include `transformers-open-weight` evidence by default for real residual claims
- backend fallback and hook coverage are exported with every relevant claim
- `full-internal-rl` beats the best competing control in sparse-reward held-out runs
- replacement effect delta is positive and reported
- trace/synthetic success remains labeled as mechanism evidence, not real residual evidence

## Rollback Boundary

U08 should remain frozen-substrate by default. If open-weight capture/intervention is unstable, the pipeline should downgrade the claim verdict rather than mutate the base model or silently switch backend.
