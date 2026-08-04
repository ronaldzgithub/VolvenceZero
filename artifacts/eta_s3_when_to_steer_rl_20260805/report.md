# ETA 学何时扳（S3 · Internal RL）

> 冻结 sensor+executor，唯一在线更新门控策略；只给稀疏终局信用、只观测 PE 代理，从不给每步对错标签。REINFORCE+baseline 学「何时扳」。不安装控制器、不改 production、不回灌 evaluation、不训基底/reader/executor。

## 结论

- S3 admission：`FAIL`
- failed：`('gain-vs-noop', 'gain-vs-always-on', 'gain-vs-random-gate')`
- seeds `(0, 1, 2, 3, 4)`，episodes 1200，post-switch 占比 0.430

## heldout NLL（seed 平均）

| arm | NLL |
|---|---:|
| noop | 2.8129 |
| always_on_belief | 1.7909 |
| random_gate | 2.0263 |
| **pe_gated_online（学到的门控）** | 0.9508 |
| oracle_gate（上界诊断） | 1.0903 |
| pe_hard_gate（硬规则上界诊断） | 1.0903 |
| fresh_ceiling | 0.0271 |

## 判定门（seed 平均；bootstrap CI 下界取最差 seed）

| 门 | 值 | 阈值 |
|---|---:|---:|
| 收敛改善（初始→最终） | 0.7867 | ≥0.2 |
| gain vs noop | 1.8621（CI下界 -0.1330） | ≥0.3, CI>0 |
| gain vs always-on | 0.8402（CI下界 0.0000） | ≥0.2, CI>0 |
| gain vs random-gate | 1.0755（CI下界 0.0000） | ≥0.2, CI>0 |
| 门控选择性 steer(非切换)−steer(切换) | 0.3819 | ≥0.3 |

## 边界

- 存在硬规则上界（belief==fresh 才出手）；本 claim = **RL 从稀疏终局信用学到逼近该上界**，对应 companion 无免费每步标签的真实约束。
- `substrate_trainable=0`、reader/executor 冻结（未变）、no free bias、zero-code no-op；仅策略参数在线更新。PASS 只准入独立权威 sweep，不授权 production。
- 不改写任何封存 verdict（kill-eta / S2 / B screen / C2 / 08 / S3-A）。