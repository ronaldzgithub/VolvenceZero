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

### 2026-06-29 — hosted directional, ALL families, 3 tracks (raw / ref-harness / camel), no-GPU

> **Status**: directional (NOT retain). All ~30 public scenarios (F1–F6), single seed.
> **Substrate**: `qwen-turbo` (DashScope). same-substrate guard: VERIFIED (all 3 tracks `qwen-turbo`; camel + ref-harness upstream both point at it).
> **Judges (cross-family, non-Qwen, OpenRouter)**: user-sim=`meta-llama/llama-3.3-70b-instruct`, per-turn=`deepseek/deepseek-chat` (high-volume → fast), arc=`mistralai/mistral-large` (per-arc). per-turn family ≠ arc family.
> **Tracks**: `raw`, `ref-harness` (4 memory components, deepseek extractor), `camel` (camel-ai 0.2.90 ChatAgent on the same qwen-turbo). `volvence`/`volvence-cold` absent (need GPU).
> **Runner**: `run_hosted_ablation.py --family all --with-camel`. Arc failures (transient API): raw 1, ref-harness 2, camel 2 (isolated; run continued).
> **Judge-evidence**: SHADOW scaffold only (`judge_evidence/`); no real robustness sweep yet (#48/#71). Since the deltas below are within noise, judge choice cannot manufacture a signal that is not present.

#### Per-track results (all families, 1 seed)

| Track | final_mean | CI95 | A1 | A2 | A3 (continuity) | A4 | A5 | A6 | arcs |
|---|---|---|---|---|---|---|---|---|---|
| raw (bare qwen-turbo) | 81.54 | [76.5, 85.9] | 94.1 | 97.1 | 57.8 | 89.0 | 95.6 | 98.6 | 29 |
| ref-harness (standard memory wrapper) | 81.61 | [77.1, 85.9] | 94.7 | 97.0 | 57.1 | 90.1 | 95.4 | 98.9 | 28 |
| camel (standard agent framework) | 78.27 | [74.6, 82.1] | 94.5 | 97.3 | 48.3 | 90.2 | 96.2 | 98.9 | 28 |

#### Headline finding (honest)

On the **full** companion benchmark with a real substrate + real cross-family judges:

- **A standard memory wrapper does NOT beat the bare model** (81.61 vs 81.54; CIs almost identical). The large F1-only continuity win (+24 on A3) **washed out** when averaged across all six families — memory helps the continuity-family scenarios but is neutral / slightly noisy elsewhere, so the net effect is ~0.
- **A standard agent framework (CAMEL) slightly underperforms the bare model** (78.27 vs 81.54; CIs overlap but the point estimate and continuity A3 48.3 vs 57.8 are clearly lower). The transcripts show why: CAMEL produces long, flowery, less-grounded replies that the judges score lower on a companion task.

#### Why this matters for the thesis (strategically GOOD)

The bar "beat raw + a standard memory wrapper + a standard agent framework" is **real and non-trivial** on the full benchmark: naive approaches do not clear it. So if Volvence's controller layer beats this ~81.5 raw baseline (and the wrapper/agent baselines that tie/lose to it), that is a meaningful, non-gamed result — not "we beat a strawman with no memory". It also shows the benchmark is not trivially won by bolting on memory.

#### Caveats (do not over-read)

- Single seed; CIs are wide (~±4–5) so only large effects are detectable. The raw≈ref-harness tie and camel's dip are directional.
- Weak substrate (`qwen-turbo`): a stronger substrate could change whether the wrapper's overhead helps or hurts.
- Aggregate only (per-family breakdown not retained beyond per-axis); the continuity story (memory helps F1 specifically) is visible only because F1 was run separately (see prior entry).
- Single judge config; real #71 robustness sweep still pending.
- `volvence` / `volvence-cold` not tested → the actual thesis question (does the cognitive controller beat these baselines?) is still open and needs a GPU-served substrate.

---

### 2026-06-28 — hosted directional (raw vs ref-harness), no-GPU

> **Status**: directional (NOT retain) — single family, single seed, n=5 arcs.
> **Tier**: hosted directional (no GPU). Substrate served via DashScope API.
> **Substrate**: `qwen-turbo` (DashScope, OpenAI-compatible). same-substrate guard: VERIFIED (raw + ref-harness both `qwen-turbo`).
> **Judges (cross-family, non-Qwen, OpenRouter)**: user-sim=`meta-llama/llama-3.3-70b-instruct`, per-turn=`mistralai/mistral-large`, arc=`deepseek/deepseek-chat`. (OpenAI ToS-blocked + Anthropic/Google unavailable on this OpenRouter account.)
> **Tracks present**: `raw`, `ref-harness`. (`volvence`/`volvence-cold` need a GPU-served substrate; `camel` needs `camel-ai` installed.)
> **Runner**: `scripts/companion_bench/run_hosted_ablation.py --family F1`.
> **Judge-evidence**: NOT yet run (single judge config; no robustness/calibration sweep). Treat deltas as directional only.

#### Per-track final_mean (F1, 5 arcs, seed 0)

| Track | final_mean | A1 | A2 | A3 (continuity) | A4 | A5 | A6 |
|---|---|---|---|---|---|---|---|
| raw (bare qwen-turbo) | 72.94 | 94 | 92.6 | 40 | 86.6 | 91 | 96.2 |
| ref-harness (standard memory wrapper) | 79.12 | 91 | 90 | 64 | 81 | 91 | 89.6 |

Directional read: a correctly-scoped standard memory wrapper beats the bare same model by **+6.2** overall and **+24 on continuity (A3)** — the axis cross-session memory should help. This is the "what does a standard wrapper add on a fixed substrate" baseline the controller layer must then beat.

#### Scope-keying bug found + fixed (why running it for real mattered)

The first F1 attempt showed ref-harness *losing* (64.99, A3=25). Root cause: CompanionBench's `arc_runner` sends `user_id=None`, so the harness fell back to a header surrogate (`User-Agent`+`Authorization`) that is **identical across every arc** → all 5 arcs collapsed into one memory scope → cross-arc memory bleed. Fix: when no `user_id`, key scope on the arc id parsed from the `{arc_id}-s{idx}` session convention (sessions of one arc share memory; arcs stay isolated). Applied to both `companion-ref-harness` and `companion-camel-baseline` `derive_scope_key`, with regression tests. Post-fix: 79.12 (A3=64).

#### State + decision

- **STATE**: `wiring-ready` (only 2 of 5 tracks present; the four #87 claims compare against `volvence`, which is absent → `insufficient_data`). This is NOT a thesis verdict.
- This run establishes: (a) the formal pipeline runs end-to-end on a real hosted model with real cross-family judges, no GPU; (b) a fair, arc-scoped standard memory wrapper beats bare qwen-turbo on F1, especially continuity.
- Next to make it a real verdict: add `volvence` / `volvence-cold` on the SAME qwen (GPU box), add `camel` (`pip install camel-ai`), run all six families + multiple seeds + held-out, and run the judge robustness/calibration sweep before quoting numbers.

#### Caveats (do not over-read)

- Single family (F1 = continuity, which favors memory), single seed, n=5 arcs → no stable CI; not significant.
- Weak substrate (`qwen-turbo`) + single judge config (no #48/#71 robustness yet).
- Only raw vs ref-harness; the thesis question (Volvence vs these baselines) is not yet tested.

---

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
