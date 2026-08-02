# Gate 1 assessment (reduced authoritative sweep)

**Verdict: FAIL** — do not proceed to Stage 2 continued pretraining.

| metric | observed | threshold | pass? |
|---|---|---|---|
| spearman(α, rate) | -0.657 | ≤ -0.8 | NO |
| rate span | 0.585 | ≥ 0.30 | YES |
| vs 7-route baseline (0.20) | 0.585 | materially larger | YES |

The companion rate-distortion runner stamped `incomplete-sweep` because only the
frozen arm ran; that is expected for this Gate-1 protocol and is not itself the
Gate-1 decision. Gate 1 uses the rate-axis response thresholds above.

## Directional reads
- Memorization is resolved (train ≈ heldout distortion).
- Rate axis is awake (span ~3× the old 7-route baseline).
- Non-monotonic rebound at α=1.0 keeps spearman short of -0.8.
- Pre-registered fallback: fix posterior-variance parameterization, then re-run Gate 1.

Artifacts: `report.md`, `curves.json`, `gate1_assessment.json`, `rate_distortion_curves.png`.
