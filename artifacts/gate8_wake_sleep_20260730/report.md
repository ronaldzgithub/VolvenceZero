# Gate 8 wake/sleep evidence

- partition: `trace-locked-confirmation`
- formal locked run: `True`
- verdict: `causal-supported`
- source fingerprint: `163449e09562d300cdc98c15f98b16c93c83030d20a95976d44c25c9887c88cf`

## Mechanism gates

- source-consumer-admission: `True` (1.000000)
- sleep-prompt-token-increment-zero: `True` (0.000000)
- duplicate-job-execution-zero: `True` (0.000000)
- owner-lineage-complete: `True` (1.000000)
- whole-cycle-rollback-exact: `True` (0.000000)
- turn-latency-excludes-slow-job: `True` (0.000000)

## Causal gates

- cold-start-loss-reduction-vs-no-sleep: `True` (0.454176)
- callback-consistency-gain-vs-no-sleep: `True` (1.000000)
- delayed-payoff-gain-vs-no-sleep: `True` (0.454176)
- full-outperforms-single-owner-controls: `True` (0.054176)
- full-owner-state-drift-bounded: `True` (0.337482)
