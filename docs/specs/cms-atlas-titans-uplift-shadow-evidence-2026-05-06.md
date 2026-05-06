# CMS ATLAS / Titans Uplift — SHADOW Smoke Evidence (2026-05-06)

> Companion brief to [`cms-atlas-titans-uplift.md`](./cms-atlas-titans-uplift.md) §7–§8.
> Run command: `python scripts/run_atlas_titans_cms_shadow_smoke.py --case-limit 2`
> Wall-clock: ~10 min on local CPU (HuggingFace `BUILTIN_ONLY` substrate, 76-weight model)
> Cases: `repair`, `task_clarification` (2 of 4 default scripted dialogue proof cases)
> Profiles: `pe-eta` (canonical) vs `atlas-titans-cms-uplift` (CMS uplift flags on)
> Raw artifact: `artifacts/cms_atlas_titans_shadow_smoke/atlas_titans_cms_shadow_smoke.json`

## Headline

- Both profiles satisfy the canonical scripted acceptance: `passed = 1.0` for both.
- The new uplift path **does not regress** the dialogue PE-ETA acceptance gates (`prediction_chain_turn_count`, `online_learning_turn_count`, `bounded_writeback_turn_count`, `delayed_improvement_observed`, `nested_profile_active_turn_count`, `evolution_judge_*`, `runtime_backbone_*` etc. all identical or within float noise).
- The CMS itself is exercised differently: 6 metrics diverge by more than 1e-6, isolated in the memory / credit / scheduler family. No regression in substrate-, evaluation-, or temporal-related families.

The uplift path is wired correctly (CMS state evolution differs from canonical), and no canonical regression appears at smoke scale.

## Aggregate metric deltas (uplift − canonical)

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

## Per-case stability

Both `repair` and `task_clarification` show the same direction of divergence:

- `carryover_credit_turn_count`: −3 in both cases (full collapse to 0).
- `mean_learned_recall_confidence`: −0.0208 / −0.0216.
- `mean_memory_tower_alignment`: −0.0192 / −0.0200.

The signs are consistent across cases (no within-suite case flips), so the deltas are case-invariant smoke-level effects, not stochastic noise.

## Interpretation

### Expected and benign

- `mean_learned_recall_confidence` and `mean_memory_tower_alignment` mild drops are consistent with the spec §5 prediction: when the uplift path activates, the `LearnedUpdateRule` switches to a wider feature layout (16-dim, last 4 columns are PE features). The PE columns receive fresh-init random weights (per `LearnedUpdateRule.__init__`), and on a 6-turn case the rule has not had time to learn meaningful weights for those columns — so the rule's decisions are slightly perturbed and downstream readouts (recall confidence, tower alignment) are slightly noisier. This is the textbook "fresh control surface" cost; we expect it to vanish or invert as more training data flows through.
- The other tiny drifts on `switch_gate`, `rare_heavy_pressure`, `updater_confidence` are sub-1e-3 — within numerical noise of running the CMS through a different update trajectory.

### To investigate before promoting to ACTIVE

- **`carryover_credit_turn_count` collapse from 3 → 0** is the only macroscopic divergence. Hypothesis: the uplift's tighter Titans-style write gating (PE-driven) reduces the situations where credit naturally carries over consecutive turns. This is plausibly aligned with the design goal (more selective memory writes), but it is a one-step-removed downstream signal of a CMS change, so it deserves a closer look at scale — it could equally be a side effect of the rule's untrained PE columns producing a different `write_gate` schedule on `online-fast`.

  Required follow-up before ACTIVE: re-run on the full 4-case `paper-suite-small` profile-set with N≥5 seeds (per spec §8 step 7), and confirm `carryover_credit_turn_count` either stabilizes to a non-zero rate or shows a credit-quality compensation (e.g. higher `delayed_improvement_observed` per remaining carryover turn).

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
| 7 | dialogue paper-suite-small (5+ seeds, 4 cases, NL essence) | NOT YET RUN |
| 8 | ETA strong-proof statistical-batch-evidence | NOT YET RUN |

Steps 1–6 pass. The path is clear to run step 7 next; step 8 follows. Until step 7 + 8 close, the uplift remains SHADOW-evidence-only.

## Open items

- Re-confirm the `carryover_credit_turn_count` drop is mechanism-driven not artifact, by checking either:
  - the per-turn `switch_gate` / `write_gate` schedule of `online-fast`, or
  - the credit owner's `CreditSnapshot.cumulative_credit_by_level` distribution on the two profiles.
- After paper-suite-small run, capture the same deltas with seed variance so we can put confidence intervals on the 6 diverged metrics.
- If `mean_learned_recall_confidence` and `mean_memory_tower_alignment` do not recover at higher sample counts, that is a signal to investigate the PE feature feed (e.g. confirm `_set_latest_pe_features` actually receives non-zero PE values during `pe-eta`-class scripted cases — most scripted PE in the proof set is mid-magnitude, so the PE feature columns may be mostly small numbers and contributing little signal).

## Artifact pointers

- Raw run JSON: `artifacts/cms_atlas_titans_shadow_smoke/atlas_titans_cms_shadow_smoke.json` (37 KB)
- Smoke runner: [`scripts/run_atlas_titans_cms_shadow_smoke.py`](../../scripts/run_atlas_titans_cms_shadow_smoke.py)
- Summary helper: [`scripts/_summarize_atlas_titans_shadow_smoke.py`](../../scripts/_summarize_atlas_titans_shadow_smoke.py)
