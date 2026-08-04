# ETA 条件化学习式 steering screen（P2c · C2）

> 范围：goal-ambiguous junction 仪器上，matched-budget 条件 vs 无条件学习式低秩乘性写入；不安装控制器、不改 production、不回灌 evaluation。

## 结论

- screen admission：`PASS`
- failed conditions：`()`
- seeds：`(0,)`，rank=8，updates=4

## heldout NLL（seed 平均）

| arm | expert NLL |
|---|---:|
| noop（目标隐藏基线） | 2.8129 |
| subgoal-revealed 天花板 | 0.2178 |
| conditional（本方案） | 1.3434 |
| unconditional（等预算恒定） | 2.0199 |
| random-condition（错条件） | 2.2400 |

## 判定量（seed 平均 / 最差）

| 门 | 值 | 阈值 |
|---|---:|---:|
| gap closed (noop−cond) | 1.4695 (min 1.4695) | ≥0.3 |
| conditional advantage (uncond−cond) | 0.6765 (min 0.6765) | ≥0.15 |
| condition specificity (rand−cond) | 0.8965 | ≥0.15 |
| gap closed fraction | 0.5662 | ≥0.3 |

## 守门边界

- 距离天花板与恒定算子皆为 matched-budget（同 norm cap）对照；no free bias、zero-code strict no-op、substrate 冻结。
- screen 只决定是否准入独立权威 sweep；不改写任何已封存 verdict，不授权 production。
