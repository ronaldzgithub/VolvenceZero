# Packet 3 前置 b：分辨力预检（余量审计）

- overall: FAIL

| verdict | value |
|---|---|
| corpus_sufficient | True |
| expert_margin | False |
| steer_headroom | True |
| overall_pass | False |

```json
{
  "junction_records": 8044,
  "distinct_state_keys": 77,
  "contrastive_junctions": 37,
  "action_distribution": {
    "investigate": 2956,
    "edit": 828,
    "test": 3513,
    "submit": 680,
    "invalid": 67
  },
  "expert_action_distribution": {
    "edit": 9,
    "test": 13,
    "submit": 3,
    "investigate": 12
  },
  "source_trajectories": 838,
  "train_junctions": 22,
  "eval_junctions": 15
}
```

```json
{
  "junctions_scored": 37,
  "median_gap_nats": 0.3018,
  "mean_gap_nats": 0.1748,
  "bootstrap_ci_lower_5pct": -0.5404,
  "positive_fraction": 0.5135
}
```

```json
{
  "probe_junctions": 12,
  "directions_per_junction": 8,
  "control_norm_cap": 61.7532,
  "mean_abs_nll_shift": 1.75307,
  "max_abs_nll_shift": 6.92667,
  "min_shift_required": 0.01
}
```
