# ETA rate-distortion observation protocol determinism audit

Zero-compute test of whether the expert action is a deterministic function of a single observation (i.e. whether temporal segmentation is required at all).

- corpus seed 20260802, 64 train / 24 heldout routes, 850 total phases
- action vocabulary (11): hub, red, blue, green, yellow, orange, purple, black, white, north, south

## Determinism by observation view

| view | steps | distinct texts | ambiguous texts | ambiguous steps | determinism |
|---|---|---|---|---|---|
| full | 850 | 850 | 0 | 0 | 1.0000 |
| no_source | 850 | 302 | 63 | 438 | 0.4847 |
| local_only | 850 | 130 | 45 | 688 | 0.1906 |
| local_no_completed | 850 | 10 | 10 | 850 | 0.0000 |

## Intra-route recurrence (would force a switch)

| view | routes | routes with intra-route conflict | conflicting slots |
|---|---|---|---|
| full | 88 | 0 | 0 |
| no_source | 88 | 0 | 0 |
| local_only | 88 | 0 | 0 |
| local_no_completed | 88 | 74 | 74 |

## Verdict

- segmentation redundant under v1: **True**
- source_text leak determinism delta (no_source - full): **-0.5153**
- views that force switching (intra-route conflict > 0): **['local_no_completed']**
- completed-objectives leak blocks switching under no_source: **True**

Dropping source_text alone leaves intra-route conflict at zero, so a constant per-route code still solves v1 minus the fingerprint. Only removing BOTH source_text and completed objectives from recurring steps forces an evolving latent code. Protocol v2 must therefore give the route plan once at step 0 and drop both fields from every later step.
