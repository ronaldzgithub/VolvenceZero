# ETA rate-distortion criterion

- schema: `eta-rate-distortion-evidence.v1`
- model: `Qwen/Qwen2.5-0.5B-Instruct` on `mps` (origin `hf-local`, fallback=False)
- injection layer: 20, control norm cap: 24.33 (probe hidden norm 97.33)
- n_z=16, alpha grid=[0.01, 0.1, 1.0], seeds=[0], updates/run=300
- observation protocol: `partially-observable-staged-plan.v4`
- posterior parameterization: `smooth`
- rate gating: `switch-gated`
- train steps=551, heldout steps=299

## Verdict: `incomplete-sweep`

Both arms are required for the validity control; ran only ('frozen',).

**Not preregistered — mechanism-only smoke.** This verdict is not authoritative and must not be cited as evidence.

- arms distinguishable: False (max separation 0.0000, threshold 0.0200)

## Gap assessments

### frozen arm

- gap detected: **False**
- distortion span 0.1597, rate span 0.6306, noise scale 0.0000
- max adjacent drop 0.1547 (96.8% of span) over 62.8% of the rate span, between alpha=1 and alpha=0.1
- boundary F1 inside gap 0.000 vs outside 0.000

## Aggregate curves (train distortion)

| arm | alpha | rate | distortion | ±std | heldout d | boundary F1 | switch freq |
|---|---|---|---|---|---|---|---|
| frozen | 1 | 0.0324 | 1.0791 | 0.0000 | 1.2086 | 0.000 | 0.000 |
| frozen | 0.1 | 0.4283 | 0.9245 | 0.0000 | 1.0291 | 0.000 | 0.000 |
| frozen | 0.01 | 0.6630 | 0.9194 | 0.0000 | 1.0237 | 0.000 | 0.000 |

Unsteered baseline distortion (train): 2.6026
