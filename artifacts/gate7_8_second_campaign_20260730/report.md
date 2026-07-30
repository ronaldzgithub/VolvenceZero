# Gate 7–8 second campaign reconciliation

## Outcome

- Gate 7 formal locked verdict: `invalid`.
- Gate 8 formal locked verdict: `causal-supported`.
- Gate 1/4/6 v2 development retests: all `not-supported`; their locked partition was not consumed.
- Gate 2 v35 open-loop causal verdict remains valid; v36 SHADOW/live injection remains frozen.
- Gate 5 remains deferred until a longitudinal-scale corpus changes its evidence premise.

The campaign-level evidence grade remains `mechanism-supported`. This is not a
thesis-level retain verdict and does not authorize runtime live promotion.

## Gate 7

All five arms ran once on locked data. Source admission, future leakage,
token-space mutation and rollback gates passed, but seeds 709 and 719 changed
the action-family structure during the nominally frozen RL phase. The cause
was an unguarded anti-collapse topology split. The implementation was fixed
and directly verified after the run, but the immutable locked verdict was not
rewritten. A new source/schema is required for confirmation.

## Gate 8

The four matched arms completed on three seeds. Full sleep reduced
next-session cold-start loss by `0.454176`, increased callback consistency by
`1.0`, and increased delayed payoff by `0.454176` versus no-sleep. Its minimum
payoff margin over the two single-owner controls was `0.054176`; maximum owner
drift was `0.337482`. Prompt increment, duplicate job execution, lineage gaps,
turn/slow-job latency contamination and 12-arm rollback mismatches were all
zero.

## Retests

The new v2 corpus removed the old input-identifiability excuses but did not
rescue Gate 1/4/6:

- Gate 1: PE changed temporal parameters in every full-arm episode, while
  next-session controller loss stayed byte-for-byte equal to drive-off.
- Gate 4: segment-aware saved no labels and was not final-accuracy
  non-inferior.
- Gate 6: paired/swapped state was distinguishable, but primary meta-init
  lost to copy-init and produced negative transfer on every episode.

Per preregistration, these development NO-GO results stopped each gate before
locked consumption.
