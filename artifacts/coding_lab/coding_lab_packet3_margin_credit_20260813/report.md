# Packet 3 前置 b：分辨力预检（余量审计）

- overall: FAIL

| verdict | value |
|---|---|
| corpus_sufficient | False |
| expert_margin | None |
| steer_headroom | None |
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
  "eval_junctions": 8
}
```

> contrastive corpus has 22 junctions < 24; collect more API-hand trajectories (scripted branches share identical move sequences and cannot contrast).
