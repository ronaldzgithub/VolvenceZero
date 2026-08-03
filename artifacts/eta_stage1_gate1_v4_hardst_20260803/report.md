# ETA rate-distortion criterion

- schema: `eta-rate-distortion-evidence.v1`
- model: `Qwen/Qwen2.5-0.5B-Instruct` on `mps` (origin `hf-local`, fallback=False)
- injection layer: 20, control norm cap: 24.33 (probe hidden norm 97.33)
- n_z=16, alpha grid=[0.01, 0.1, 1.0], seeds=[0], updates/run=300
- observation protocol: `partially-observable-staged-plan.v4`
- posterior parameterization: `smooth`
- rate gating: `switch-gated`
- gate mode: `hard-st`
- train steps=551, heldout steps=299

## Verdict: `incomplete-sweep`

Both arms are required for the validity control; ran only ('frozen',).

**Not preregistered — mechanism-only smoke.** This verdict is not authoritative and must not be cited as evidence.

- arms distinguishable: False (max separation 0.0000, threshold 0.0200)

## Gap assessments

### frozen arm

- gap detected: **False**
- distortion span 0.1814, rate span 2.3774, noise scale 0.0000
- max adjacent drop 0.0859 (47.4% of span) over 92.4% of the rate span, between alpha=0.1 and alpha=0.01
- boundary F1 inside gap 0.000 vs outside 0.506

## Aggregate curves (train distortion)

| arm | alpha | rate | distortion | ±std | heldout d | boundary F1 | switch freq |
|---|---|---|---|---|---|---|---|
| frozen | 1 | 0.1305 | 0.9467 | 0.0000 | 1.1235 | 0.417 | 0.388 |
| frozen | 0.1 | 0.3116 | 1.1281 | 0.0000 | 1.2093 | 0.434 | 0.172 |
| frozen | 0.01 | 2.5079 | 1.0422 | 0.0000 | 1.1438 | 0.666 | 1.000 |

Unsteered baseline distortion (train): 2.6026
