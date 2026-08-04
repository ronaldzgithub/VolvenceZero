# ETA 条件化学习式 steering screen（P2c · C2）

> 范围：goal-ambiguous junction 仪器上，matched-budget 条件 vs 无条件学习式低秩乘性写入；不安装控制器、不改 production、不回灌 evaluation。

## 结论

- screen admission：`PASS`
- failed conditions：`()`
- seeds：`(0, 1, 2)`，rank=8，updates=80

## heldout NLL（seed 平均）

| arm | expert NLL |
|---|---:|
| noop（目标隐藏基线） | 2.8129 |
| subgoal-revealed 天花板 | 0.2178 |
| conditional（本方案） | 0.0270 |
| unconditional（等预算恒定） | 1.3573 |
| random-condition（错条件） | 7.3772 |

## 判定量（seed 平均 / 最差）

| 门 | 值 | 阈值 |
|---|---:|---:|
| gap closed (noop−cond) | 2.7859 (min 2.7757) | ≥0.3 |
| conditional advantage (uncond−cond) | 1.3303 (min 1.3285) | ≥0.15 |
| condition specificity (rand−cond) | 7.3501 | ≥0.15 |
| gap closed fraction | 1.0735 | ≥0.3 |

## 守门边界

- 距离天花板与恒定算子皆为 matched-budget（同 norm cap）对照；no free bias、zero-code strict no-op、substrate 冻结。
- screen 只决定是否准入独立权威 sweep；不改写任何已封存 verdict，不授权 production。
