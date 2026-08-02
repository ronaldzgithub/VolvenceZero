# ETA rate-distortion criterion

- schema: `eta-rate-distortion-evidence.v1`
- model: `Qwen/Qwen2.5-0.5B-Instruct` on `mps` (origin `hf-local`, fallback=False)
- injection layer: 20, control norm cap: 24.33 (probe hidden norm 97.33)
- n_z=16, alpha grid=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0], seeds=[0, 1, 2], updates/run=24
- observation protocol: `partially-observable-no-remaining-route.v1`
- train steps=551, heldout steps=299

## Verdict: `incomplete-sweep`

Both arms are required for the validity control; ran only ('frozen',).

This verdict is authoritative under the frozen protocol.

- arms distinguishable: False (max separation 0.0000, threshold 0.0278)

## Gap assessments

### frozen arm

- gap detected: **False**
- distortion span 0.1871, rate span 0.5847, noise scale 0.0139
- max adjacent drop 0.1537 (82.2% of span) over 64.6% of the rate span, between alpha=1 and alpha=0.01
- boundary F1 inside gap 0.000 vs outside 0.000

## Aggregate curves (train distortion)

| arm | alpha | rate | distortion | ±std | heldout d | boundary F1 | switch freq |
|---|---|---|---|---|---|---|---|
| frozen | 3 | 0.2118 | 1.6486 | 0.0139 | 1.6753 | 0.000 | 0.000 |
| frozen | 0.3 | 0.2648 | 1.5822 | 0.0293 | 1.6179 | 0.000 | 0.000 |
| frozen | 0.1 | 0.2878 | 1.4980 | 0.0191 | 1.5167 | 0.000 | 0.000 |
| frozen | 0.03 | 0.3178 | 1.4635 | 0.0060 | 1.4601 | 0.000 | 0.000 |
| frozen | 1 | 0.4188 | 1.6152 | 0.0121 | 1.6441 | 0.000 | 0.000 |
| frozen | 0.01 | 0.7966 | 1.4615 | 0.0031 | 1.4522 | 0.000 | 0.000 |

Unsteered baseline distortion (train): 2.7896
