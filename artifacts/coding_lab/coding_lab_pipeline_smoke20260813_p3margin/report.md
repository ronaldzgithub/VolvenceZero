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
  "junction_records": 1333,
  "distinct_state_keys": 51,
  "contrastive_junctions": 5,
  "action_distribution": {
    "investigate": 428,
    "edit": 319,
    "test": 270,
    "submit": 311,
    "invalid": 5
  },
  "expert_action_distribution": {
    "edit": 2,
    "investigate": 1,
    "submit": 1,
    "test": 1
  },
  "source_trajectories": 319,
  "train_junctions": 2,
  "eval_junctions": 3
}
```

```json
{
  "junctions_scored": 5,
  "median_gap_nats": -1.626,
  "mean_gap_nats": -1.2307,
  "bootstrap_ci_lower_5pct": -3.0557,
  "positive_fraction": 0.2
}
```

```json
{
  "probe_junctions": 5,
  "directions_per_junction": 4,
  "control_norm_cap": 61.7532,
  "mean_abs_nll_shift": 2.50832,
  "max_abs_nll_shift": 7.01331,
  "min_shift_required": 0.01
}
```
