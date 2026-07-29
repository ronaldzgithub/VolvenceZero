# Gate 1–6 evidence campaign reconciliation

- campaign status: `mechanism-supported`
- target gates: `1 / 2 / 4 / 5 / 6`
- mechanism verdicts: `5/5 supported`
- causal verdicts: `1/5 supported`, `4/5 not-supported`
- longitudinal verdicts: `0 supported`
- Gate 2 SHADOW observation: `not-supported`
- thesis retained: `false`
- known debt #92 closed: `false`

## What survived

- Gate 1: PE is an auditable primary signal, numerically grounded to true LSS.
- Gate 2: the v35 conditional action-value selector passed its preregistered
  permutation null on fresh validation and locked confirmation.
- Gate 4: typed feedback requests and OpenLoop actuation are owner-routed,
  auditable, boundary-safe, and rollback-capable.
- Gate 5: multi-frequency CMS cadence and rollback mechanisms are runnable and
  auditable under a frozen substrate.
- Gate 6: nested initialization is owner-controlled, zero-leakage, auditable,
  and checkpoint rollback-capable.

## What did not survive

- Gate 1 did not show a held-out behavioral gain from PE drive.
- Gate 2 v36 did not pass fresh-validation closed-loop SHADOW observation, so
  runtime SHADOW/live promotion remains frozen.
- Gate 4 did not show segment-aware or PE-driven label efficiency.
- Gate 5 did not reach the preregistered CMS Pareto minimum effect.
- Gate 6 did not beat direct copy initialization and did not reveal a
  distinguishable user-related prior.

## Interpretation

The common result is mechanism support, not a system-level causal or
longitudinal proof. The campaign does not emit `thesis-retained`, does not
close known debt #92, and does not authorize runtime SHADOW/live promotion.
All failed locked gates are immutable; no threshold retuning or same-partition
rerun is allowed.
