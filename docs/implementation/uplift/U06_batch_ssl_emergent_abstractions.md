# U06 Batch SSL Emergent Abstractions

> Status: draft
> Last updated: 2026-04-25
> Parent: `docs/implementation/12_eta_paper_grade_uplift_plan.md`
> Primary claim target: temporal abstractions should emerge from residual trajectories and bottleneck pressure, not from semantic labels or handcrafted family scaffolds.

## Problem

The current metacontroller SSL path is already Eq.3-shaped, but it is still closer to a lightweight proof learner than a paper-grade discovery process.

U06 upgrades SSL from trace-local training into batch residual-trajectory training and makes scaffold removal a first-class gate.

## Current Anchors

- `volvence_zero/temporal/ssl.py`
- `volvence_zero/temporal/noncausal_embedder.py`
- `volvence_zero/temporal/metacontroller_components.py`
- `volvence_zero/temporal/m3_optimizer.py`
- `volvence_zero/temporal/training.py`
- `volvence_zero/agent/eta_proof_benchmark.py`
- `tests/test_eta_proof_benchmark.py`

## Design Target

The SSL stage should learn:

- a non-causal posterior `q_phi(z_t | e_1:T)` during discovery
- a switch gate `beta_t` whose sparsity follows from the bottleneck objective
- a decoder from latent code to residual controller effect
- latent action families from posterior density and reuse, not product semantic names

The runtime/causal policy may later approximate this structure, but it must not define the discovered structure itself.

## Batch Training Contract

Introduce a batch SSL training contract with:

- input: residual trajectory batches with optional next-action or next-state targets
- forbidden input: subgoal labels, product action names, expected family IDs
- objective: prediction loss + `alpha * KL(q_phi(z_t | e_1:T) || N(0, I))`
- optional auxiliary terms: switch persistence, decoder effect consistency, posterior smoothness
- output: frozen discovery snapshot containing posterior summaries, switch statistics, decoder parameters, and latent clusters

The trainer should report every loss term separately. Aggregate `total_loss` is insufficient for emergence claims.

## Latent Family Discovery

Family formation should move toward data-derived clustering.

Candidate sources:

- posterior samples `z_tilde`
- persistent latent windows after switch interpolation
- decoder outputs with similar downstream residual effect
- repeated high-payoff abstract-action windows from held-out rollouts

Disallowed as primary family sources:

- semantic action labels
- fixed seed prototypes
- keyword-derived route names
- profile-specific family assignment shortcuts

Semantic names may be attached as readouts after discovery. They cannot be used to choose or stabilize the family during the primary proof path.

## Switch Emergence Criteria

`beta_t` cannot be considered emergent merely because a threshold produces binary switches.

Required evidence:

- increasing KL/bottleneck pressure changes switch sparsity predictably
- switch boundaries align with latent subgoal transitions better than random and better than no-bottleneck controls
- mean persistence window increases on compositional tasks
- switch entropy does not collapse to always-switch or never-switch
- held-out routes reuse switch/family patterns learned in training

## Scaffold Ablation Matrix

U06 must preserve or extend these profiles:

- `pe-eta`
- `pe-eta-no-semantic-label`
- `pe-eta-no-reflection-cache`
- `pe-eta-pe-readout-only`
- `full-no-fast-prior`
- `full-no-optimize`
- `full-no-replacement`

For ETA emergence claims, the main comparison is not only full vs weak baseline. It is full vs full-with-scaffold-removed.

## Reports And Metrics

Batch SSL report fields should include:

- `batch_count`
- `trajectory_count`
- `prediction_loss_mean`
- `kl_loss_mean`
- `switch_sparsity_mean`
- `switch_entropy`
- `posterior_drift_mean`
- `noncausal_information_content`
- `cluster_stability`
- `family_birth_count`
- `family_merge_count`
- `family_prune_count`
- `scaffold_ablation_retention`

The report should also publish the `alpha` schedule so switch behavior can be interpreted against bottleneck strength.

## Implementation Tasks

1. Add a batch dataset object for residual trajectories.
2. Extend `MetacontrollerSSLTrainer` or add a batch trainer wrapper that accumulates losses over batches.
3. Add data-derived latent family initialization and persistence metrics.
4. Add a no-label discovery path that does not depend on `active_label` semantics.
5. Extend paper-suite summaries with SSL emergence metrics.
6. Add tests for no-label retention, KL sensitivity, and switch entropy non-collapse.

## Exit Criteria

U06 is complete when:

- family bank initialization can be derived from residual trajectory data
- no-semantic-label runs retain measurable family reuse above threshold
- no-reflection-cache and no-fast-prior runs do not fully erase switch/family structure
- `alpha` changes produce interpretable switch-sparsity differences
- batch SSL reports are exported into ETA paper-suite artifacts

## Rollback Boundary

U06 may change temporal training and benchmark discovery paths. It must not force live sessions to depend on experimental batch SSL artifacts until U07 takeover gates pass.
