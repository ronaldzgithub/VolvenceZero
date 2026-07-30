# Gate 8 longitudinal fifth campaign

## Outcome

Gate 8 is now `longitudinal-supported` on its frozen claim. The overall
campaign remains `mechanism-supported`: Gate 5 is still not supported,
Gate 2 has no live longitudinal path, several causal gates remain closed, and
production/live promotion is not authorized.

## Evidence

The campaign reused the immutable fresh Qwen source created in the fourth
campaign. Gate 8 had not consumed that source before the preregistration. Three
seeds each supplied 510 settled transitions; the four matched arms therefore
processed 6120 arm-transitions. Every arm/seed crossed 51 consumer sessions
with 50 filesystem persistence and constructor-restart boundaries.

The authoritative result is `longitudinal-supported`:

- cold-start loss reduction versus no-sleep: `+0.567363`
- callback consistency gain versus no-sleep: `+1.000000`
- delayed-payoff gain versus no-sleep: `+0.567363`
- minimum delayed-payoff margin versus single-owner controls: `+0.167363`
- maximum full owner-state drift: `0.467850` under the `0.50` bound
- all paired-seed 95% confidence lower bounds are positive
- prompt increment, duplicate execution, turn/slow-job contamination,
  substrate mutation, persistence mismatch and rollback mismatch are zero

## Claim boundary

The three seed gains are identical, so the paired-seed intervals have zero
between-seed variance. This is strong deterministic replication on distinct
trace lineages, not evidence of distributional variance across independently
sampled model behaviors.

The callback and temporal-alignment metrics are deterministic owner readouts.
They do not provide blinded human relationship-quality ground truth. Debt #51
therefore remains open, as does debt #92. Gate 8's formal artifact is immutable
and may not be rerun on the same source to improve the claim.

A concurrent second run completed after the authoritative artifact had already
been written. It is recorded in the source ledger as
`invalid-duplicate-not-admitted`; it has no effect on the verdict, confidence
intervals, or claim boundary.

## System reconciliation

Causal-supported gates remain `2 / 8 / 11`. Longitudinal-supported gates are
now `8 / 11`; Gate 5 remains longitudinal `not-supported`. The campaign-level
common evidence tier remains `mechanism-supported`, `thesis_retained=false`,
and production/live promotion remains blocked.
