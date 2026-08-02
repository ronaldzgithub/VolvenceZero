# Stage-1 rate-axis feasibility pilot (NOT the pre-registered verdict)

**Claim scope:** feasibility-only. The authoritative Gate-1 verdict requires the
pre-registered full frozen sweep (`artifacts/eta_stage1_rate_axis_prereg_20260802/`:
200 train / 60 heldout routes x 6 alphas x 3 seeds x 40 updates). This pilot runs
one seed, three alphas, eight updates, and two small route counts to read the
*direction* of the rate axis as the seeded corpus grows past the 7 hardcoded
routes that produced the 2026-08-01 `kill-eta` verdict.

## What the 2026-08-01 kill-eta run left unexplained
- The KL **rate axis barely moved** with alpha (span ~0.20).
- The 3 train routes were **memorised**: train distortion 0.007 vs heldout 4.7.

## Pilot result (seed 0, alphas {0.03, 0.3, 3.0}, 8 updates, frozen arm)

| train routes | steps | spearman(alpha,rate) | rate span | rate @0.03 / 0.3 / 3.0 | train dist @0.03/0.3/3.0 | heldout dist @0.03/0.3/3.0 |
|---|---|---|---|---|---|---|
| 6  | 54  | -0.5 | 0.808 | 1.241 / 0.434 / 1.192 | 1.347 / 1.367 / 1.537 | 1.636 / 1.651 / 1.847 |
| 24 | 211 | -0.5 | 0.640 | 1.047 / 0.407 / 0.993 | 1.640 / 1.699 / 1.830 | 1.637 / 1.704 / 1.850 |

## Reading
1. **Memorisation is resolved.** At 24 routes train and heldout distortion are
   nearly identical (1.640/1.699/1.830 vs 1.637/1.704/1.850). The free-lunch
   memorisation that made the 8-01 rate axis meaningless is gone with a larger,
   compositionally-split corpus.
2. **The rate axis now moves.** Span is 0.64-0.81, 3-4x the ~0.20 8-01 baseline
   and well past the pre-registered `rate_span_min` = 0.30. Distortion also rises
   monotonically with alpha (correct direction), i.e. KL is buying action error.
3. **But rate is non-monotonic in alpha.** It drops sharply at alpha=0.3 then
   rises again at alpha=3.0, so spearman only reaches -0.5, short of the
   pre-registered -0.8. The alpha=3.0 rebound is the likely culprit -- probably
   posterior-variance / optimisation instability at high KL weight with only 8
   updates and 1 seed.

## Directional conclusion (feasibility only)
The data-mechanism hypothesis is **partially supported**: enriching the corpus
removes memorisation and wakes the rate axis (span), but does not yet deliver the
monotone alpha->rate response Gate 1 requires. Two follow-ups are indicated
before/for the authoritative run:
- run the pre-registered full sweep (more updates, 3 seeds, 6 alphas) to see
  whether monotonicity emerges with budget and averaging; and
- if the alpha=3.0 rebound persists, treat it as the posterior-variance
  parameterization issue the Gate-1 pre-registration names as the fallback, fix
  it, and re-run -- do NOT proceed to Stage 2 pretraining until Gate 1 passes.

## Reproduce
```
python scripts/run_eta_rate_axis_pilot.py \
  --output-dir artifacts/eta_stage1_rate_axis_pilot_20260802 \
  --route-counts 6 24 --alphas 0.03 0.3 3.0 --seeds 1 --updates 8
```
Authoritative Gate-1 run:
```
python scripts/run_eta_rate_distortion.py \
  --output-dir artifacts/eta_stage1_gate1_full_<date> \
  --preregistration artifacts/eta_stage1_rate_axis_prereg_20260802/preregistration.json \
  --alphas 0.01 0.03 0.1 0.3 1.0 3.0 --seeds 3 --arms frozen \
  --corpus-seed 20260802 --objective-count 8 --train-routes 200 --heldout-routes 60
```
