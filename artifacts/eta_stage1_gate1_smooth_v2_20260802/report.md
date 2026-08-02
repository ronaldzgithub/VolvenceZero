# ETA rate-distortion criterion

- schema: `eta-rate-distortion-evidence.v1`
- model: `Qwen/Qwen2.5-0.5B-Instruct` on `mps` (origin `hf-local`, fallback=False)
- injection layer: 20, control norm cap: 24.33 (probe hidden norm 97.33)
- n_z=16, alpha grid=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0], seeds=[0, 1, 2], updates/run=40
- observation protocol: `partially-observable-no-route-identity.v2`
- posterior parameterization: `smooth`
- train steps=551, heldout steps=299

## Verdict: `incomplete-sweep`

Both arms are required for the validity control; ran only ('frozen',).

This verdict is authoritative under the frozen protocol.

- arms distinguishable: False (max separation 0.0000, threshold 0.0221)

## Gap assessments

### frozen arm

- gap detected: **True**
- distortion span 0.0446, rate span 0.6906, noise scale 0.0110
- max adjacent drop 0.0275 (61.5% of span) over 12.9% of the rate span, between alpha=0.3 and alpha=0.1
- boundary F1 inside gap 0.000 vs outside 0.000

## Aggregate curves (train distortion)

| arm | alpha | rate | distortion | ±std | heldout d | boundary F1 | switch freq |
|---|---|---|---|---|---|---|---|
| frozen | 3 | 0.0011 | 1.4997 | 0.0036 | 1.5008 | 0.000 | 0.000 |
| frozen | 1 | 0.0020 | 1.4989 | 0.0038 | 1.4999 | 0.000 | 0.000 |
| frozen | 0.3 | 0.0138 | 1.4883 | 0.0058 | 1.4992 | 0.000 | 0.000 |
| frozen | 0.1 | 0.1028 | 1.4609 | 0.0173 | 1.4830 | 0.000 | 0.000 |
| frozen | 0.03 | 0.2793 | 1.4607 | 0.0163 | 1.4813 | 0.000 | 0.000 |
| frozen | 0.01 | 0.6917 | 1.4551 | 0.0195 | 1.4781 | 0.000 | 0.000 |

Unsteered baseline distortion (train): 2.4239
