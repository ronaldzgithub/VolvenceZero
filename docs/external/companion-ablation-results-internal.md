# Companion Bench — same-substrate ablation results (internal)

> Status: **TEMPLATE — awaiting first real run (P1)**
> Last updated: 2026-06-28
> Scope: debt #87 (thesis first-stage retain verdict) + #82 / #84
> Audience: internal review. This is the per-run record; it is NOT an external
> claim until a verdict reaches `first-stage-retained` AND a judge-bias check
> backs it (see [`companion-ablation.md`](../specs/companion-ablation.md)).

This document is the **per-run record** for the same-substrate Companion Bench
ablation. Each fresh experiment appends a new dated section at the top, keeping
older runs as an audit trail. Do not delete the template after the first run —
copy it.

## How to produce a run

```bash
# 0. wiring smoke (no GPU/keys) — proves the flow + verdict
python scripts/companion_bench/run_same_substrate_ablation.py --phase p0-smoke --family F1

# 1. judge variance/calibration evidence first (#48 / #71)
python scripts/companion_bench/run_same_substrate_ablation.py --phase judge-evidence

# 2. boot the five same-substrate endpoints (GPU box)
VZ_SUBSTRATE_MODEL_ID=Qwen/Qwen2.5-7B-Instruct LIFEFORM_LOCAL_API_KEY=... \
  REFH_EXTRACTOR_BASE_URL=https://api.anthropic.com/v1 \
  REFH_EXTRACTOR_MODEL=anthropic/claude-3.7-sonnet \
  REFH_EXTRACTOR_KEY_ENV=ANTHROPIC_API_KEY REFH_EXTRACTOR_FAMILY=anthropic \
  bash scripts/companion_bench/serve_same_substrate_ablation.sh

# 3. directional run (public 24 × 1 seed), then verdict
python scripts/companion_bench/run_same_substrate_ablation.py --phase p1

# 4. retain run (24 public + 96 held-out × 3 seeds), then verdict
python scripts/companion_bench/run_same_substrate_ablation.py --phase p2
```

## Red lines (must hold for any quotable number)

1. `same-substrate VERIFIED` — `assert_same_substrate.py` passed for all 5 tracks.
2. `non-Qwen judges` — per-turn + arc + user-sim are a different family from the Qwen substrate.
3. `cross_user_memory_isolation` + the other three CompanionBench attestations are True for every track.
4. `judge-evidence` recorded — judge variance/calibration is on file (a high-σ judge invalidates small deltas).
5. No held-out scenario text leaked into any prompt.

## Verdict states (from compare_companion_ablation.py)

- `kill-criteria-triggered` → shrink thesis to product-memory/companion; downgrade the thesis doc.
- `wiring-ready` → flow works, tracks incomplete; not a result.
- `weak-positive` → all core comparisons positive but CIs not clear of zero; add seeds/arcs.
- `first-stage-retained` → claims 1-4 all retain; human world-model thesis FIRST STAGE retained.
- `world-model-extension-ready` → NOT decidable here; needs an independent physical-side benchmark.

---

## Run log (newest first)

### TEMPLATE — copy this block for every new run

> **Status**: pending first run
> **Run timestamp**: _e.g. 2026-MM-DDTHHMMSSZ_
> **Tier**: _p1 (public 24 × 1 seed) / p2 (public + held-out × 3 seeds)_
> **Substrate**: _e.g. Qwen/Qwen2.5-7B-Instruct_   **same-substrate guard**: _VERIFIED / NOT VERIFIED_
> **Judges**: per-turn=_Claude_, arc=_GPT-5_, user-sim=_Claude_ (all non-Qwen)
> **Judge-evidence**: _path to judge_robustness / calibration artifacts_
> **Verdict file**: _artifacts/companion-ablation/<date>/verdict_<tier>.json_

#### Per-track final_mean

| Track | final_mean (0-100) | CI95 | arc_count | Notes |
|---|---|---|---|---|
| raw | _xx.xx_ | _[lo, hi]_ | _n_ |  |
| ref-harness | _xx.xx_ | _[lo, hi]_ | _n_ |  |
| camel | _xx.xx_ | _[lo, hi]_ | _n_ |  |
| volvence-cold | _xx.xx_ | _[lo, hi]_ | _n_ |  |
| volvence | _xx.xx_ | _[lo, hi]_ | _n_ |  |

#### Claims

| Claim | status | delta / ci_low | Notes |
|---|---|---|---|
| claim_pipeline_gt_raw (volvence − raw) | _retain/weak/fail_ | _±x.xx / ±x.xx_ |  |
| claim_gt_standard_layers (vs ref-harness & camel) | _retain/weak/fail_ | _…_ |  |
| claim_training_adds_value (volvence − cold) | _retain/weak/fail_ | _±x.xx / ±x.xx_ |  |
| claim_heldout_cohort_stable | _retain/weak/insufficient_ | _rel_ci_halfwidth_ |  |

#### Continuity axis (A3) — the long-session signal

| Track | A3 mean | Notes |
|---|---|---|
| raw | _xx.xx_ |  |
| ref-harness | _xx.xx_ |  |
| camel | _xx.xx_ |  |
| volvence-cold | _xx.xx_ |  |
| volvence | _xx.xx_ |  |

#### State + decision

- [ ] **STATE**: _kill-criteria-triggered / wiring-ready / weak-positive / first-stage-retained_
- [ ] If `first-stage-retained`: ok to describe the human-world-model thesis first stage as retained (NOT world-model enablement).
- [ ] If `kill-criteria-triggered`: shrink thesis per #87; downgrade `research/strategy/human-world-model-thesis-2026-06.md`.
- [ ] Follow-up debts opened (if any): _…_

#### Diagnosis (human analysis)

_If volvence ≤ ref-harness/camel: is the gap real or judge noise (check judge-evidence σ)?
Is the expression layer over-truncating? Is the same-substrate guard actually VERIFIED?
If full ≤ cold: are the trained bootstraps over-fit to other scenarios?_

---

## Open questions resolved by the first run

1. On a real Qwen, does the Volvence pipeline beat the bare substrate at all? (claim 1)
2. Does it beat a standard memory wrapper AND a standard agent framework? (claim 2) — the question every DD asks.
3. Do the trained bootstraps pay off over the cold pipeline? (claim 3)
4. Is the advantage stable on held-out scenarios across seeds? (claim 4)
