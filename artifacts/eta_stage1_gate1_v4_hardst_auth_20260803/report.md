# ETA rate-distortion criterion

- schema: `eta-rate-distortion-evidence.v1`
- model: `Qwen/Qwen2.5-0.5B-Instruct` on `mps` (origin `hf-local`, fallback=False)
- injection layer: 20, control norm cap: 24.33 (probe hidden norm 97.33)
- n_z=16, alpha grid=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0], seeds=[0, 1, 2], updates/run=300
- observation protocol: `partially-observable-staged-plan.v4`
- posterior parameterization: `smooth`
- rate gating: `switch-gated`
- gate mode: `hard-st`
- train steps=551, heldout steps=299

## Verdict: `incomplete-sweep`

Both arms are required for the validity control; ran only ('frozen',).

This verdict is authoritative under the frozen protocol.

- arms distinguishable: False (max separation 0.0000, threshold 0.1788)

## Gap assessments

### frozen arm

- gap detected: **True**
- distortion span 0.2162, rate span 1.9328, noise scale 0.0894
- max adjacent drop 0.1608 (74.4% of span) over 19.6% of the rate span, between alpha=1 and alpha=0.3
- boundary F1 inside gap 0.394 vs outside 0.537

## Aggregate curves (train distortion)

| arm | alpha | rate | distortion | ±std | heldout d | boundary F1 | switch freq |
|---|---|---|---|---|---|---|---|
| frozen | 3 | 0.0678 | 0.9478 | 0.0455 | 1.1494 | 0.310 | 0.119 |
| frozen | 1 | 0.0696 | 1.0119 | 0.0843 | 1.1658 | 0.283 | 0.172 |
| frozen | 0.3 | 0.4486 | 0.8511 | 0.0700 | 1.0332 | 0.505 | 0.570 |
| frozen | 0.1 | 1.0057 | 0.8716 | 0.2016 | 1.0663 | 0.573 | 0.624 |
| frozen | 0.03 | 1.9787 | 0.7957 | 0.0148 | 1.0096 | 0.665 | 0.959 |
| frozen | 0.01 | 2.0007 | 1.0041 | 0.1204 | 1.1097 | 0.600 | 0.720 |

Unsteered baseline distortion (train): 2.6026
