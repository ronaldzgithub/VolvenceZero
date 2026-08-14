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
  "contrastive_junctions": 22,
  "action_distribution": {
    "investigate": 2956,
    "edit": 828,
    "test": 3513,
    "submit": 680,
    "invalid": 67
  },
  "expert_action_distribution": {
    "edit": 8,
    "test": 8,
    "submit": 1,
    "investigate": 5
  },
  "source_trajectories": 838,
  "mean_expert_pass_rate": 0.5943,
  "mean_non_expert_pass_rate": 0.2702,
  "train_junctions": 14,
  "eval_junctions": 8,
  "label_policy": {
    "expert_source": "conditional-pass-rate-credit",
    "min_action_support": 5,
    "min_pass_rate_margin": 0.1
  },
  "state_key_accounting": {
    "labelled": 22,
    "excluded_no_leverage": 22,
    "excluded_under_supported": 33
  }
}
```

```json
{
  "scoring": "domain-conditional-pmi",
  "junctions_scored": 22,
  "median_gap_nats": -2.1384,
  "mean_gap_nats": -0.5954,
  "bootstrap_ci_lower_5pct": -1.7138,
  "positive_fraction": 0.3636,
  "raw_median_gap_nats": -0.8046,
  "raw_positive_fraction": 0.3636,
  "neutral_action_nll": {
    "investigate": 3.1429,
    "edit": 3.002,
    "test": 0.4388,
    "submit": 1.338
  }
}
```

```json
{
  "probe_junctions": 12,
  "directions_per_junction": 8,
  "control_norm_cap": 96.0917,
  "mean_abs_nll_shift": 1.72992,
  "max_abs_nll_shift": 7.43709,
  "min_shift_required": 0.01
}
```
