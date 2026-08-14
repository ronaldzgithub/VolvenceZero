# Packet 3：S3-E 编程域复刻判词

- smoke: True
- admitted: False
- failed_conditions: ['gain-vs-noop', 'gain-vs-random-gate', 'gate-selectivity']
- rows: train=9 heldout=25
- post_switch_fraction: 0.160

```json
{
  "seed_count": 1,
  "noop_nll_mean": 2.3054181754589083,
  "pe_gated_online_nll_mean": 3.2315450751781465,
  "always_on_belief_nll_mean": 11.341018605229873,
  "random_gate_nll_mean": 3.389690227031424,
  "oracle_gate_ceiling_nll_mean": 2.126715444324036,
  "pe_hard_gate_ceiling_nll_mean": 11.341018605229873,
  "fresh_ceiling_nll_mean": 11.341018605229873,
  "convergence_improvement_nll_mean": 3.2847315907466506,
  "gate_selectivity_mean": 0.14285714285714285,
  "gain_vs_noop_ci_lower_min": -1.0275136062077113,
  "gain_vs_always_on_ci_lower_min": 7.159546981963039,
  "gain_vs_random_gate_ci_lower_min": -0.04506633562722141
}
```
