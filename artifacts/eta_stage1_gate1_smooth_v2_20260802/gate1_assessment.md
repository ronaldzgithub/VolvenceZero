# Gate 1 assessment (smooth posterior + v2 observation, reduced authoritative sweep)

**Verdict: FAIL** — rate-axis gate passes but no switching occurs; do not proceed to Stage 2.

Combined rule: Gate 1 passes iff the rate-axis gate passes **AND** boundary_f1 > 0.

| gate | metric | observed | threshold | pass? |
|---|---|---|---|---|
| rate axis | spearman(α, rate) | **-1.000** | ≤ -0.8 | **YES** |
| rate axis | rate span | 0.691 | ≥ 0.30 | YES |
| rate axis | vs 7-route baseline (0.20) | 0.691 | materially larger | YES |
| switching | max boundary F1 | **0.000** | > 0 | **NO** |
| switching | max hard switch freq | 0.000 | > 0 | NO |
| switching | max mean switch prob | 0.456 | ≥ 0.55 to fire | (under threshold) |

## What changed vs the 2026-08-02 legacy run
- **Rate axis fixed.** spearman moved -0.657 → **-1.000** (perfect monotone), seed variance collapsed. The smooth posterior (`σ = softplus(Wh) + floor`, unbounded mean) removed the clamp-saturation artifact.
- **Leakage removed.** v2 states the plan once at step 0 and then exposes only current location + transitions; the audit showed this forces intra-route ambiguity in 74/88 routes.

## Rate–distortion shape (the real finding)
- Rate spans 0.001 → 0.69 (~600×) for only a **0.045** distortion improvement (1.4997 → 1.4551).
- Steering itself drops distortion from the **2.4239** unsteered baseline to ~1.50; time-varying high-information codes add almost nothing.
- The curve is **near-horizontal**, the opposite of ETA's predicted near-vertical gap.

## Read
With both prior confounds (posterior saturation, observation leakage) removed, the controller still **never switches** and extra information buys almost no accuracy. The learned control collapses to a near-constant steering direction — the degenerate "switch once" baseline. At Qwen2.5-0.5B / this scale the ETA temporal-abstraction signature does not appear.

## Next step
Do **not** proceed to Stage 2 continued pretraining on the strength of temporal abstraction. The rate-axis mechanism is now healthy, so the open question is substrate/scale capacity, not the posterior artifact. Any further ETA claim must first exhibit a near-vertical gap **and** boundary_f1 > 0 on a larger substrate before spending pretraining compute.

Artifacts: `report.md`, `report.json`, `curves.json`, `rate_distortion_curves.png`, `gate1_assessment.json`.
