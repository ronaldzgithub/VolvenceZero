# ETA rate-distortion criterion

- schema: `eta-rate-distortion-evidence.v1`
- model: `Qwen/Qwen2.5-0.5B-Instruct` on `mps` (origin `hf-local`, fallback=False)
- injection layer: 20, control norm cap: 29.82 (probe hidden norm 119.29)
- n_z=16, alpha grid=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0], seeds=[0, 1, 2], updates/run=40
- observation protocol: `partially-observable-no-remaining-route.v1`
- train steps=18, heldout steps=31

## Verdict: `kill-eta`

The instrument passed the joint-arm validity control but the frozen arm shows no rate-distortion gap across the alpha grid.

- arms distinguishable: True (max separation 0.4059, threshold 0.0348)

## Gap assessments

### frozen arm

- gap detected: **False**
- distortion span 0.3996, rate span 0.1934, noise scale 0.0347
- max adjacent drop 0.3980 (99.6% of span) over 48.5% of the rate span, between alpha=3 and alpha=0.03
- boundary F1 inside gap 0.000 vs outside 0.141

### joint arm

- gap detected: **False**
- distortion span 0.0000, rate span 0.2889, noise scale 0.0000
- max adjacent drop 0.0000 (100.0% of span) over 20.2% of the rate span, between alpha=3 and alpha=0.01
- boundary F1 inside gap 0.000 vs outside 0.035

## Aggregate curves (train distortion)

| arm | alpha | rate | distortion | ±std | heldout d | boundary F1 | switch freq |
|---|---|---|---|---|---|---|---|
| frozen | 0.1 | 0.4447 | 0.0113 | 0.0006 | 4.7576 | 0.212 | 0.333 |
| frozen | 1 | 0.4703 | 0.1940 | 0.0728 | 3.2089 | 0.000 | 0.000 |
| frozen | 0.3 | 0.4714 | 0.0178 | 0.0037 | 4.6022 | 0.000 | 0.000 |
| frozen | 3 | 0.5222 | 0.4060 | 0.1296 | 2.6217 | 0.000 | 0.000 |
| frozen | 0.03 | 0.6160 | 0.0081 | 0.0011 | 4.7864 | 0.424 | 0.667 |
| frozen | 0.01 | 0.6381 | 0.0065 | 0.0006 | 4.7132 | 0.212 | 0.333 |
| joint | 0.03 | 0.2398 | 0.0001 | 0.0000 | 8.3195 | 0.000 | 0.000 |
| joint | 1 | 0.4183 | 0.0001 | 0.0000 | 8.3750 | 0.000 | 0.000 |
| joint | 0.3 | 0.4183 | 0.0001 | 0.0000 | 8.3193 | 0.000 | 0.000 |
| joint | 0.1 | 0.4192 | 0.0001 | 0.0000 | 8.3154 | 0.000 | 0.000 |
| joint | 3 | 0.4703 | 0.0001 | 0.0000 | 8.3956 | 0.000 | 0.000 |
| joint | 0.01 | 0.5287 | 0.0001 | 0.0000 | 8.2976 | 0.211 | 0.267 |

Unsteered baseline distortion (train): 1.7522
