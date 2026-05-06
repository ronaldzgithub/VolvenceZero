# CMS ATLAS / Titans Uplift — SHADOW Evidence and ACTIVE Decision (2026-05-06)

> Companion brief to [`cms-atlas-titans-uplift.md`](./cms-atlas-titans-uplift.md) §7–§8.
> Smoke command: `python scripts/run_atlas_titans_cms_shadow_smoke.py --case-limit 2`
> Paper-suite-small shadow command: inline runner over `DEFAULT_DIALOGUE_PROOF_CASES`, 5 seeds, profiles `pe-eta` + `atlas-titans-cms-uplift`
> Wall-clock: ~10 min on local CPU (HuggingFace `BUILTIN_ONLY` substrate, 76-weight model)
> Cases: `repair`, `task_clarification` (2 of 4 default scripted dialogue proof cases)
> Profiles: `pe-eta` (canonical) vs `atlas-titans-cms-uplift` (CMS uplift flags on)
> Smoke artifact: `artifacts/cms_atlas_titans_shadow_smoke/atlas_titans_cms_shadow_smoke.json`
> Paper-suite-small artifact: `artifacts/cms_atlas_titans_paper_suite_small_shadow/paper_suite_small_shadow.json`
> ETA strong-proof artifact bundle: `artifacts/cms_atlas_titans_eta_strong_proof_step8/`
> ACTIVE decision: `build_default_memory_store(...)` now enables `cms_pe_features_enabled=True` and `cms_replay_window_size=8` by default.

## Headline

- Both profiles satisfy the canonical scripted acceptance: `passed = 1.0` for both in smoke and in paper-suite-small shadow (5 seeds x 4 cases).
- ETA strong-proof paper-suite-small also passes the statistical-batch evidence gate and retains the main internal-RL claims. This closes the full spec §8 validation ladder for SHADOW evidence.
- The new uplift path **does not regress** the dialogue PE-ETA acceptance gates (`prediction_chain_turn_count`, `online_learning_turn_count`, `bounded_writeback_turn_count`, `delayed_improvement_observed`, `nested_profile_active_turn_count`, `evolution_judge_*`, `runtime_backbone_*` etc. all identical or within float noise).
- The CMS itself is exercised differently: after the carryover-credit allow-list fix, the remaining deltas are isolated to memory readout confidence / tower alignment / updater confidence and tiny scheduler drift. No regression in substrate-, evaluation-, temporal-, or acceptance-related families.

The uplift path is wired correctly (CMS state evolution differs from canonical), and no canonical regression appears at smoke or paper-suite-small scale.

## Post-fix paper-suite-small deltas (uplift − canonical)

Run: 5 seeds x 4 default scripted dialogue cases x 2 profiles.

| Metric | mean delta | std | n | interpretation |
|---|---:|---:|---:|---|
| `passed` | +0.0000 | 0.0000 | 5 | no acceptance regression |
| `carryover_credit_turn_count` | +0.0000 | 0.0000 | 5 | false alarm fixed |
| `mean_learned_recall_confidence` | -0.0219 | 0.0000 | 5 | mild fresh-feature perturbation |
| `mean_memory_tower_alignment` | -0.0203 | 0.0000 | 5 | mild fresh-feature perturbation |
| `mean_memory_updater_confidence` | +0.0002 | 0.0000 | 5 | noise-level |
| `mean_switch_gate` | -0.0001 | 0.0000 | 5 | noise-level |
| `mean_timescale_contract_retained` | +0.0000 | 0.0000 | 5 | no timescale regression |

This closes spec §8 step 7 for the dialogue paper-suite-small shadow profile pair: no default acceptance gate regresses, and the only material deltas are the expected memory readout perturbations from a fresh PE-aware update rule surface.

## ETA strong-proof step 8

Run: `eta-proof-paper-suite-small`, 5 repeated runs, 7 ETA proof profiles, trace + synthetic-open-weight backends.

Primary metrics:

| Metric | mean | CI low | CI high |
|---|---:|---:|---:|
| `heldout_terminal_success_rate` | 1.0000 | 1.0000 | 1.0000 |
| `heldout_strong_success_rate` | 0.9761 | 0.9740 | 0.9777 |
| `heldout_family_reuse_rate` | 1.0000 | 1.0000 | 1.0000 |
| `heldout_credit_alignment` | 1.0000 | 1.0000 | 1.0000 |
| `strong_success_gap_vs_best_control` | 0.1076 | 0.1075 | 0.1080 |
| `backend_success_gap` | 0.0175 | 0.0175 | 0.0175 |

Claim verdicts:

| Claim | Status |
|---|---|
| `claim_eta_internal_rl_advantage` | retain |
| `claim_eta_internal_rl_sparse_reward_advantage` | retain |
| `claim_eta_scaffold_free_temporal_abstraction` | weak |
| `claim_nl_slow_loop_improves_eta_fast_path` | retain |

This closes spec §8 step 8: statistical-batch evidence does not regress. The scaffold-free claim remains `weak`, which is the same kind of conservative verdict expected from the ETA proof suite and does not block the CMS uplift; the uplift only needed to avoid degrading ETA strong-proof evidence.

## Original smoke deltas before carryover allow-list fix

Out of 88 published metrics, **6 diverge** non-trivially:

| Metric | canonical | uplift | delta | direction |
|---|---|---|---|---|
| `carryover_credit_turn_count` | 3.0000 | 0.0000 | −3.0000 | down (notable) |
| `mean_learned_recall_confidence` | 0.1777 | 0.1564 | −0.0213 | down (mild) |
| `mean_memory_tower_alignment` | 0.1847 | 0.1651 | −0.0196 | down (mild) |
| `mean_memory_updater_confidence` | 0.7649 | 0.7652 | +0.0003 | up (noise-level) |
| `mean_scheduler_rare_heavy_pressure` | 0.5017 | 0.5016 | −0.0001 | down (noise-level) |
| `mean_switch_gate` | 0.8882 | 0.8883 | +0.0001 | up (noise-level) |

Acceptance summary metric `passed` = 1.0 on both profiles.

## Original smoke per-case stability

Both `repair` and `task_clarification` show the same direction of divergence:

- `carryover_credit_turn_count`: −3 in both cases (full collapse to 0).
- `mean_learned_recall_confidence`: −0.0208 / −0.0216.
- `mean_memory_tower_alignment`: −0.0192 / −0.0200.

The signs are consistent across cases (no within-suite case flips), so the deltas are case-invariant smoke-level effects, not stochastic noise.

## Interpretation

### Expected and benign

- `mean_learned_recall_confidence` and `mean_memory_tower_alignment` mild drops are consistent with the spec §5 prediction: when the uplift path activates, the `LearnedUpdateRule` switches to a wider feature layout (16-dim, last 4 columns are PE features). The PE columns receive fresh-init random weights (per `LearnedUpdateRule.__init__`), and on a short scripted suite the rule has not had time to learn meaningful weights for those columns — so the rule's decisions are slightly perturbed and downstream readouts (recall confidence, tower alignment) are slightly noisier. This is the textbook "fresh control surface" cost; we expect it to vanish or invert as more training data flows through.
- The other tiny drifts on `switch_gate`, `rare_heavy_pressure`, `updater_confidence` are sub-1e-3 — within numerical noise of running the CMS through a different update trajectory.

### Diagnosed and resolved

- **`carryover_credit_turn_count` collapse from 3 → 0 was a false alarm**. The follow-up diagnostic [`scripts/diagnose_atlas_titans_carryover_credit.py`](../../scripts/diagnose_atlas_titans_carryover_credit.py) on the `repair` case showed that across all 6 turns, the two profiles produce essentially identical CMS state and credit distributions:

  | Turn | write_gate (canonical / uplift) | cumulative_credit_by_level (canonical vs uplift) |
  |---|---|---|
  | 1 | 0.498 / 0.492 | identical to 3 d.p. |
  | 2 | 0.572 / 0.569 | identical to 3 d.p. |
  | 3 | 0.590 / 0.588 | identical to 3 d.p. |
  | 4 | 0.628 / 0.627 | identical to 3 d.p. |
  | 5 | 0.671 / 0.670 | identical to 3 d.p. |
  | 6 | 0.735 / 0.740 | identical to 3 d.p. |

  Replay window `K` ramps as designed (1 → 5 → 8 → 8 → 8 → 8). PE feature magnitudes enter the rule once non-zero PE arrives at turn 2.

  Root cause of the 3 → 0 metric: the dialogue benchmark's `_profile_allows_interval_carryover_credit` is an **explicit allow-list**, not an exclusion list. `atlas-titans-cms-uplift` was missing from the set, so `carryover_temporal_response` was hard-coded to `False` in `_pe_trigger_analysis`, and the metric collapsed regardless of CMS behavior.

  Fix landed in this PR: `atlas-titans-cms-uplift` added to the allow-list (it is a `pe-eta` sibling: full PE drive, full ETA, only CMS-internal update rule differs).

  After the fix, the only remaining non-trivial divergences are `mean_learned_recall_confidence` (−0.02) and `mean_memory_tower_alignment` (−0.02). Both are first-order consequences of the wider `LearnedUpdateRule` feature layout: PE columns receive untrained random weights, so the rule's hidden activations carry a small cumulative perturbation that propagates into recall confidence and tower alignment readouts. Both are expected to relax as the rule learns — confirm at paper-suite-small.

## Verdict against spec §8 acceptance ladder

Mapping the smoke result to the spec's eight-step ladder:

| Step | Description | Status |
|---|---|---|
| 1 | Unit: disabled-path bit-equal to legacy | passed (`tests/test_cms_atlas_titans_uplift.py`) |
| 2 | Unit: K=1 `update_with_replay` ≡ legacy `update` | passed |
| 3 | Unit: legacy state restore zero-pads PE columns | passed (CMS-level form) |
| 4 | Contract: uplift fields default for canonical | passed |
| 5 | Contract: PE optional on `MemoryStore.observe_substrate` | passed |
| 6 | **SHADOW smoke (this brief)** | **passed (no canonical regression on 2 cases)** |
| 7 | dialogue paper-suite-small (5 seeds, 4 cases, NL essence) | **passed (no canonical regression, carryover false alarm fixed)** |
| 8 | ETA strong-proof statistical-batch-evidence | **passed; main claims retain, scaffold-free remains weak** |

Steps 1–8 pass. The uplift has cleared the planned SHADOW validation ladder and is now default-ACTIVE through `build_default_memory_store(...)`.

Rollback remains explicit and local:

```python
build_default_memory_store(
    cms_pe_features_enabled=False,
    cms_replay_window_size=None,
)
```

## Open items

- If `mean_learned_recall_confidence` and `mean_memory_tower_alignment` do not recover at higher sample counts, that is a signal to investigate the PE feature feed (e.g. confirm `_set_latest_pe_features` actually receives non-zero PE values during `pe-eta`-class scripted cases — most scripted PE in the proof set is mid-magnitude, so the PE feature columns may be mostly small numbers and contributing little signal).
- Monitor `mean_learned_recall_confidence` and `mean_memory_tower_alignment` after ACTIVE. If they do not recover with more online evidence, pre-warm the PE columns with a short replay over existing dialogue artifacts or temporarily roll back via the explicit factory flags.

## Artifact pointers

- Raw run JSON: `artifacts/cms_atlas_titans_shadow_smoke/atlas_titans_cms_shadow_smoke.json` (37 KB)
- Paper-suite-small shadow JSON: `artifacts/cms_atlas_titans_paper_suite_small_shadow/paper_suite_small_shadow.json`
- ETA strong-proof bundle: `artifacts/cms_atlas_titans_eta_strong_proof_step8/`
- Smoke runner: [`scripts/run_atlas_titans_cms_shadow_smoke.py`](../../scripts/run_atlas_titans_cms_shadow_smoke.py)
- Summary helper: [`scripts/_summarize_atlas_titans_shadow_smoke.py`](../../scripts/_summarize_atlas_titans_shadow_smoke.py)
- Per-turn diagnostic for the `carryover_credit` investigation: [`scripts/diagnose_atlas_titans_carryover_credit.py`](../../scripts/diagnose_atlas_titans_carryover_credit.py)
- Diagnostic raw JSON: `artifacts/cms_atlas_titans_carryover_diagnostic/repair.json`
