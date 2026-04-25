# U05 Sparse-Reward Environment Upgrade

> Status: draft
> Last updated: 2026-04-25
> Parent: `docs/implementation/12_eta_paper_grade_uplift_plan.md`
> Primary claim target: ETA Internal RL should solve hierarchical sparse-reward tasks through latent abstract actions, not dense shaping or semantic scaffolds.

## Problem

The current ETA proof harness has a useful miniature hierarchy, but paper-grade ETA evidence needs harder sparse-reward environments with clear train/eval/held-out splits.

The goal of U05 is to upgrade the task substrate so later stages can prove:

- abstract actions persist across multiple environment steps
- held-out compositions require reuse, not memorized route matching
- delayed sparse reward can be attributed to the correct abstract-action window
- matched controls face exactly the same task and reward structure

## Current Anchors

- `volvence_zero/internal_rl/proof_environment.py`
- `volvence_zero/internal_rl/environment.py`
- `volvence_zero/internal_rl/sandbox.py`
- `volvence_zero/agent/eta_proof_benchmark.py`
- `tests/test_eta_proof_benchmark.py`

## Environment Tiers

### Tier 1: `mini-hierarchy-v2`

Extend the current route graph into a richer hierarchical environment.

Required properties:

- explicit graph roles: `entry`, `hub`, `branch`, `loop`, `return`, `distractor`, `terminal`
- route lengths spanning short, medium, and long horizons
- distractor transitions that look locally plausible but hurt terminal success
- held-out route compositions that reuse known subgoals in unseen order
- route signatures generated from environment execution, not hand-authored proof tuples

This tier is the fast CI/proof harness path.

### Tier 2: `text-gridworld`

Add a symbolic text/grid environment that isolates temporal abstraction from natural-language response quality.

Required properties:

- state is compact and inspectable
- action labels are environment protocol actions, not product semantic labels
- reward is terminal by default
- subgoal boundaries exist but are not exposed to the controller during training
- held-out maps combine known motifs in unseen layouts

This tier should become the primary mechanistic ETA benchmark because it is easier to analyze than dialogue and harder than the current route toy.

### Tier 3: `long-horizon-dialogue-sparse`

Retain dialogue as a product-facing extrapolation tier, but make reward sparse and delayed.

Required properties:

- 20-50 turn episodes
- terminal reward from repair, trust, task completion, or sustained alignment
- no per-turn dense reward in the primary profile
- dialogue PE evidence remains separate from ETA Internal RL evidence

This tier supports product relevance. It should not replace Tier 1/2 mechanistic proof.

## Reward Taxonomy

Every reward source must be tagged as one of:

| Kind | Meaning | Allowed in primary sparse profile |
|---|---|---|
| `terminal` | emitted only at successful/failed episode end | yes |
| `delayed` | emitted after a multi-step outcome window resolves | yes |
| `shaping` | intermediate progress hint | no, ablation only |
| `diagnostic` | metric readout, not used for optimization | yes, if excluded from optimizer |

The benchmark report must publish the reward mix. A run cannot claim sparse-reward success if shaping reward contributes to policy optimization in the primary profile.

## Dataset Splits

Each environment tier should expose:

- `train`: structures seen by SSL/RL
- `eval`: same distribution, unseen episode instances
- `heldout-composition`: known subgoals in unseen order
- `heldout-distractor`: same target but misleading local transitions
- `heldout-long`: longer horizon than training routes

The held-out split must be fixed in the manifest so repeated runs remain comparable.

## Matched Controls

All profiles must run the same environment, reward source, and split.

Minimum control set:

- `full-internal-rl`
- `full-no-optimize`
- `full-no-replacement`
- `learned-lite-causal`
- `noop-backend`
- `full-no-fast-prior`
- `trace-only` or equivalent non-real-backend diagnostic

Controls may disable mechanisms, but they must not receive easier rewards, shorter routes, fewer distractors, or pre-labeled subgoals.

## Metrics

Primary metrics:

- `terminal_success_rate`
- `heldout_strong_success_rate`
- `heldout_family_reuse_rate`
- `delayed_credit_alignment`
- `mean_reward_sparsity`
- `mean_steps_per_abstract_action`
- `best_control_success_gap`

Mechanism metrics:

- `switch_boundary_alignment`
- `family_reuse_on_heldout`
- `replacement_effect_delta`
- `policy_update_rate`
- `parameter_change_norm`
- `rollouts_per_update`

Failure diagnostics:

- `distractor_failure_rate`
- `premature_switch_rate`
- `family_collapse_rate`
- `reward_shaping_leakage`

## Implementation Tasks

1. Extend `MiniHierarchicalEnvironment` or add a parallel v2 environment with richer graph topology.
2. Add a text-gridworld environment module under `volvence_zero/internal_rl/`.
3. Add environment manifests for train/eval/held-out splits in `eta_proof_benchmark.py` or a dedicated case-library module.
4. Update `ETAProofCase` reporting to include reward taxonomy and split metadata.
5. Add tests that verify terminal-only reward profiles do not leak shaping reward into optimization.
6. Add matched-control tests proving every profile receives identical case specs.

## Exit Criteria

U05 is complete when:

- all three tiers are specified, even if Tier 3 remains slower/nightly only
- reward taxonomy is machine-readable in benchmark reports
- held-out composition and distractor splits exist
- matched controls share identical task/reward specs
- sparse-reward metrics are exported to the ETA paper-suite aggregate

## Rollback Boundary

U05 should not change live runtime behavior. It only adds benchmark/task infrastructure and reports. If new environments destabilize CI, gate them behind paper-suite tiers while preserving existing `ci-smoke` coverage.
