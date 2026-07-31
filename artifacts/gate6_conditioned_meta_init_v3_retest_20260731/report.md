# gate6-conditioned-meta-init-v3-retest

- partition: `trace-development-heldout`
- formal locked run: `False`
- verdict: `not-supported`
- source fingerprint: `163449e09562d300cdc98c15f98b16c93c83030d20a95976d44c25c9887c88cf`

## Mechanism gates

- source-consumer-admission: `True` (1.000000)
- lineage-complete: `True` (1.000000)
- fact-leakage-zero: `True` (0.000000)
- source-mutation-zero: `True` (0.000000)
- slow-parameter-state-unchanged: `True` (0.000000)
- rollback-exact: `True` (0.000000)

## Causal gates

- meta-effect-vs-copy-init: `False` (-0.049077)
- final-error-noninferior-vs-copy-init: `False` (0.075011)
- meta-effect-vs-random-init: `True` (0.102417)
- final-error-noninferior-vs-random-init: `False` (0.060345)
- meta-effect-vs-no-init: `True` (0.137645)
- final-error-noninferior-vs-no-init: `False` (0.054633)
- negative-transfer-zero: `False` (1.000000)
- paired-outperforms-swapped-diagnostic: `True` (0.273349)
