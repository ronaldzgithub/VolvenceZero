# ETA rate-distortion criterion

- schema: `eta-rate-distortion-evidence.v1`
- model: `Qwen/Qwen2.5-0.5B-Instruct` on `mps` (origin `hf-local`, fallback=False)
- injection layer: 20, control norm cap: 23.65 (probe hidden norm 94.62)
- n_z=16, alpha grid=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0], seeds=[0, 1, 2], updates/run=300
- observation protocol: `partially-observable-staged-plan.v4`
- posterior parameterization: `smooth`
- rate gating: `switch-gated`
- gate mode: `hard-st`
- train steps=551, heldout steps=299

## Verdict: `kill-eta`

The instrument passed the joint-arm validity control but the frozen arm shows no rate-distortion gap across the alpha grid.

This verdict is authoritative under the frozen protocol.

- arms distinguishable: True (max separation 0.1264, threshold 0.0673)

## Gap assessments

### frozen arm

- gap detected: **False**
- distortion span 0.1084, rate span 2.0680, noise scale 0.0415
- max adjacent drop 0.0988 (91.1% of span) over 84.2% of the rate span, between alpha=0.1 and alpha=0.01
- boundary F1 inside gap 0.000 vs outside 0.267

### joint arm

- gap detected: **True**
- distortion span 0.0555, rate span 1.3277, noise scale 0.0258
- max adjacent drop 0.0353 (63.7% of span) over 15.2% of the rate span, between alpha=0.3 and alpha=0.1
- boundary F1 inside gap 0.385 vs outside 0.214

## Aggregate curves (train distortion)

| arm | alpha | rate | distortion | ±std | heldout d | boundary F1 | switch freq |
|---|---|---|---|---|---|---|---|
| frozen | 3 | 0.0068 | 1.1339 | 0.0346 | 1.1861 | 0.203 | 0.063 |
| frozen | 1 | 0.0160 | 1.1341 | 0.0341 | 1.2001 | 0.190 | 0.055 |
| frozen | 0.3 | 0.0813 | 1.1256 | 0.0405 | 1.1903 | 0.348 | 0.228 |
| frozen | 0.03 | 0.2371 | 1.1250 | 0.0401 | 1.2069 | 0.041 | 0.068 |
| frozen | 0.1 | 0.3344 | 1.1245 | 0.0275 | 1.1553 | 0.379 | 0.489 |
| frozen | 0.01 | 2.0748 | 1.0257 | 0.0720 | 1.1282 | 0.440 | 0.663 |
| joint | 3 | 0.0039 | 1.1554 | 0.0333 | 1.2028 | 0.091 | 0.030 |
| joint | 1 | 0.0101 | 1.1704 | 0.0118 | 1.1909 | 0.071 | 0.092 |
| joint | 0.3 | 0.0854 | 1.1502 | 0.0042 | 1.1708 | 0.400 | 0.446 |
| joint | 0.1 | 0.2866 | 1.1149 | 0.0454 | 1.1854 | 0.370 | 0.340 |
| joint | 0.03 | 0.9728 | 1.1530 | 0.0181 | 1.1851 | 0.363 | 0.402 |
| joint | 0.01 | 1.3316 | 1.1521 | 0.0421 | 1.2250 | 0.332 | 0.394 |

Unsteered baseline distortion (train): 2.7989
