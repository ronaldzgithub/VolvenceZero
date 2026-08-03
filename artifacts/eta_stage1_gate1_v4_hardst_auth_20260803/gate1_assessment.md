# ETA Stage 1 / Gate 1 assessment - v4 + smooth + switch-gated + hard-st

- created: 2026-08-03T02:58:33.583835+00:00
- sweep: `artifacts/eta_stage1_gate1_v4_hardst_auth_20260803` (18 cells, frozen arm, 300 updates)
- preregistration: `artifacts/eta_stage1_gate1_v4_hardst_20260803_prereg.json` (sha256 `b0d18f60...`)

## Verdict: **PASS**

| check | value | threshold | pass |
|---|---|---|---|
| spearman(alpha, rate) | -1.000 | <= -0.8 | yes |
| rate span | 1.933 | >= 0.30 | yes |
| span vs 7-route baseline (0.20) | ~10x | material | yes |
| switching (heldout boundary F1 > 0) | 0.240-0.671 at every alpha | > 0 | yes |

## Directional reads

- Rate axis: spearman(alpha, rate) = -1.000 over 6 alphas, span 1.933 (~10x the 0.20 seven-route baseline).
- Never-switch collapse resolved: hard switch frequency 0.12-0.96 across the grid; heldout boundary F1 0.240-0.671, first authoritative sweep with boundary_f1 > 0.
- Boundary contrast emerges under rate pressure: boundary vs continuation gate probability 0.199 vs 0.050 at alpha=3.0.
- Near-vertical gap detected on the frozen arm: 74.4% of the distortion span drops over 19.6% of the rate span (alpha 1.0 -> 0.3), clearing the preregistered drop/rate/noise thresholds.
- Gap caveat: boundary F1 inside the gap region (0.394) is NOT higher than outside (0.537), so the paper's full Gate-3 criterion (gap + F1-in-gap + joint-arm control) remains open and requires the Stage-2 pretrained substrate.
- Attribution: v4 staged revelation supplies mid-route information, switch-gated KL makes keeping free, hard-st closes the continuous-gate smuggling loophole, 300 updates give the steering path time to use z.

## Disposition

Gate 1 passes under the frozen preregistration; Stage 2 (domain
continued-pretraining + linear probe) is now unlocked per the ladder.
The frozen-arm gap is a directional bonus only: the dual-arm validity
control and the F1-in-gap criterion belong to Gate 3 and must not be
claimed from this artifact.
