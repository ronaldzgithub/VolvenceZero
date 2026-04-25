# U09 NL Slow-Loop Support For ETA

> Status: draft
> Last updated: 2026-04-25
> Parent: `docs/implementation/12_eta_paper_grade_uplift_plan.md`
> Primary claim target: Nested Learning support should measurably improve ETA fast-path abstraction, policy initialization, and long-horizon payoff.

## Problem

NL support cannot remain only a parallel memory subsystem or dashboard telemetry. For paper-grade ETA/NL integration, slow and rare-heavy loops must improve the ETA fast path under matched controls.

U09 defines how CMS, background-slow consolidation, rare-heavy artifacts, and long-horizon credit should feed temporal abstraction and Internal RL.

## Current Anchors

- `volvence_zero/memory/cms.py`
- `volvence_zero/memory/store.py`
- `volvence_zero/memory/runtime_evidence.py`
- `volvence_zero/agent/session_post_slow_loop.py`
- `volvence_zero/joint_loop/runtime.py`
- `volvence_zero/joint_loop/pipeline.py`
- `volvence_zero/credit/`
- `volvence_zero/reflection/`
- `docs/specs/continuum-memory.md`
- `docs/specs/multi-timescale-learning.md`

## Required Couplings

### Slow Shapes Fast

CMS slow bands must produce measurable improvements in:

- fast policy initialization
- family reuse on held-out episodes
- switch stability after context reset
- reduced rollout sample count before improvement

Controls:

- nested CMS on vs non-nested CMS
- slow-to-fast reset on vs off
- tower-native consolidation on vs artifact-only memory

### Long-Horizon Credit Reaches Temporal Families

Delayed and cross-session credit must reach:

- action-family payoff
- family competition score
- temporal priors
- causal policy initialization
- rare-heavy artifact ranking

Credit should not remain only an evaluation readout.

### Rare-Heavy Artifacts Need Held-Out Acceptance

Rare-heavy candidates must be evaluated before import.

Required checks:

- pre-import held-out replay
- worst-case delta
- positive-case fraction
- best-control comparison
- rollback checkpoint availability
- substrate mutation doctrine compliance

Artifacts that improve mean score but hurt held-out or worst-case stability should be rejected or held for review.

### Evidence-Driven Self-Modification

Self-modification gates should move from static thresholds toward evidence-driven decisions.

Inputs:

- prediction error trend
- held-out replay result
- family stability
- rollback risk
- substrate pressure
- rare-heavy pressure
- cross-session payoff

Outputs:

- allow
- defer
- reject
- human review

Every decision should include machine-readable reasons.

## Memory And Runtime Metrics

Required metrics:

- `slow_to_fast_init_benefit`
- `nested_context_reset_count`
- `family_reuse_after_reset`
- `heldout_gain_after_consolidation`
- `credit_to_family_write_count`
- `long_horizon_payoff_coverage`
- `rare_heavy_pre_import_pass_fraction`
- `worst_case_delta`
- `rollback_applied`
- `gate_decision_reason_count`

## Implementation Tasks

1. Extend slow-loop reports with direct ETA impact metrics, not only memory lifecycle telemetry.
2. Add cross-session abstract-action payoff ledger or strengthen the existing delayed attribution ledger for family-level use.
3. Route long-horizon credit into temporal family competition through owner APIs.
4. Require rare-heavy held-out replay before import in paper-grade profiles.
5. Add evidence-driven gate decisions with explicit reasons and thresholds.
6. Add non-nested and no-slow-reset controls to dialogue and ETA paper suites.

## Exit Criteria

U09 is complete when:

- nested CMS improves ETA fast-path metrics over non-nested controls
- slow-to-fast reset benefit is visible in held-out family reuse or sample efficiency
- long-horizon credit changes temporal family competition or policy priors through owner APIs
- rare-heavy import requires held-out replay acceptance
- self-modification decisions are evidence-bearing and reversible

## Rollback Boundary

U09 may affect memory, temporal priors, rare-heavy artifacts, and gate decisions. It must preserve owner boundaries:

- `MemoryStore` remains the memory owner
- temporal owner remains responsible for temporal family state
- substrate owner remains frozen by default in live runtime
- rare-heavy imports must be checkpointed and rollback-ready
