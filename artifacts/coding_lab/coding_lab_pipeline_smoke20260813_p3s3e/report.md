# Packet 3：S3-E 编程域复刻判词

- smoke: True
- admitted: False
- failed_conditions: ['gain-vs-noop', 'gain-vs-random-gate', 'gate-selectivity']
- rows: train=43 heldout=50
- post_switch_fraction: 0.300

```json
{
  "seed_count": 1,
  "noop_nll_mean": 2.2215678838361055,
  "pe_gated_online_nll_mean": 3.807429759502038,
  "always_on_belief_nll_mean": 14.457284696099268,
  "random_gate_nll_mean": 4.179282573798211,
  "oracle_gate_ceiling_nll_mean": 1.7062293866713663,
  "pe_hard_gate_ceiling_nll_mean": 13.649167016741739,
  "fresh_ceiling_nll_mean": 13.53846476792888,
  "convergence_improvement_nll_mean": 5.584815432857254,
  "gate_selectivity_mean": 0.0380952380952381,
  "gain_vs_noop_ci_lower_min": -3.3259426227537916,
  "gain_vs_always_on_ci_lower_min": 4.960527095523553,
  "gain_vs_random_gate_ci_lower_min": -1.296550441018626
}
```
