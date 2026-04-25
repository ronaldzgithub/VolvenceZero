# U07 Hard Causal Takeover Gate

> Status: draft
> Last updated: 2026-04-25
> Parent: `docs/implementation/12_eta_paper_grade_uplift_plan.md`
> Primary claim target: Internal RL must only start after the causal policy can preserve the structure discovered by non-causal SSL.

## Problem

ETA's training asymmetry is essential:

1. SSL discovery may use non-causal information over the full sequence.
2. Deployment and RL must use only causal prefix information.
3. The causal policy must approximate the discovered structure without future access.
4. The discovered structure must be frozen before RL optimizes in z-space.

The current repo has transition telemetry and phase concepts. U07 makes that transition a blocking gate.

## Current Anchors

- `volvence_zero/joint_loop/pipeline.py`
- `volvence_zero/joint_loop/runtime.py`
- `volvence_zero/internal_rl/sandbox.py`
- `volvence_zero/temporal/ssl.py`
- `volvence_zero/temporal/interface.py`
- `volvence_zero/agent/eta_proof_benchmark.py`

## Ownership Rule

Separate the owner responsibilities:

| Phase | Owner | Allowed mutation |
|---|---|---|
| `ssl-discovery` | SSL trainer / offline pipeline | posterior, switch, decoder, family discovery |
| `transition` | takeover gate | no mutation except evaluation/checkpoint metadata |
| `rl-runtime` | causal z-policy / Internal RL sandbox | causal policy parameters and bounded priors |
| `rollback` | pipeline/session owner | restore checkpoint and mark failed transition |

Internal RL must not mutate SSL structure when `structure_frozen` is false.

## Takeover Metrics

The gate should evaluate:

- `posterior_agreement`: causal policy z distribution agrees with non-causal posterior summary
- `switch_sparsity_retention`: causal rollout preserves switch sparsity within tolerance
- `family_reuse_retention`: causal rollout uses discovered families instead of collapsing to new/no family
- `decoder_effect_retention`: decoded z still produces measurable residual intervention effect
- `heldout_prefix_stability`: causal prefix-only rollout stays stable on held-out prefixes
- `takeover_rollout_readiness`: enough rollout samples exist to estimate the above

Each metric must have:

- raw value
- threshold
- pass/fail
- confidence or sample count
- best failing example or diagnostic reason

## Hard Gate Semantics

If takeover fails:

- do not start RL optimization
- do not apply causal replacement to runtime control
- do not import rare-heavy temporal artifacts
- restore the pre-transition checkpoint
- emit a fail-closed assessment record

If takeover passes:

- mark `structure_frozen = true`
- export takeover metrics into paper-suite artifacts
- allow causal z-policy RL updates within the bounded owner surface

## Interaction With Online Runtime

Live online runtime should treat takeover status conservatively:

- `takeover_ready=false`: allow evidence-only or SSL-only paths, but block RL mutation
- `takeover_ready=true`: allow scheduled RL if other gates pass
- `takeover_unknown`: fail closed for paper-grade claims

This prevents a live or paper-suite path from silently mixing discovery and RL in the same mutable structure.

## Implementation Tasks

1. Promote existing transition telemetry in `SSLRLTrainingPipeline` into a formal `TakeoverGateReport`.
2. Add explicit thresholds to pipeline configuration.
3. Ensure `InternalRLSandbox.optimize()` refuses to run when takeover status is failed or unknown in paper-grade profiles.
4. Add checkpoint restore around the transition step.
5. Export takeover metrics through `ETAProofBenchmarkReport`, assessment gates, and paper-suite artifacts.
6. Add tests for pass, fail, rollback, and no-partial-mutation paths.

## Required Tests

- takeover passes when causal policy retains posterior/switch/family metrics
- takeover fails when switch sparsity collapses
- takeover fails when causal policy ignores discovered families
- takeover failure prevents RL update and parameter change
- rollback restores pre-transition temporal parameter hash
- paper-suite claim fails closed when takeover metrics are absent

## Exit Criteria

U07 is complete when:

- Internal RL cannot start in paper-grade profile without a passing takeover gate
- failed takeover leaves no partial mutation
- takeover metrics are included in evidence bundles
- takeover status participates in ETA claim verdicts

## Rollback Boundary

U07 affects training pipeline and paper-grade benchmark profiles. Existing lightweight CI-smoke paths may keep legacy behavior only if they are explicitly labeled non-paper-grade.
