# ETA 读→扳闭环（P2c · S3 前置）

> 用 refit 冻结线性 sensor 从上下文残差在线读出 subgoal（非 oracle），再用 C2 执行器扳目标剥离动作；route-level bootstrap CI 守门。不安装控制器、不改 production、不回灌 evaluation。

## 结论

- S3 前置 admission：`PASS`
- failed conditions：`()`
- reader heldout top-1：`1.000`（chance 0.125）
- seeds：`(0, 1, 2, 3, 4)`，rank=8，updates=80，bootstrap=5000

## heldout NLL（seed 平均）

| arm | expert NLL |
|---|---:|
| noop | 2.8129 |
| subgoal-revealed 天花板 | 0.2178 |
| conditional-oracle | 0.0230 |
| **conditional-online（读→扳）** | 0.0230 |
| unconditional（等预算恒定） | 1.3909 |
| random-condition | 7.6203 |

## 判定量（seed 平均；bootstrap CI 下界取最差 seed）

| 门 | 值 | 阈值 |
|---|---:|---:|
| online gap closed (noop−online) | 2.7899 | ≥0.3 |
| online conditional advantage (uncond−online) | 1.3678 | ≥0.15 |
| online−noop CI 下界(min) | 2.3983 | >0 |
| online−uncond CI 下界(min) | 1.2589 | >0 |
| reader heldout acc | 1.000 | ≥0.8 |

## 守门边界

- sensor 是 refit 冻结线性读出（无 LM 训练、steer 时冻结）；executor 为冻结基底上的 rank-8 乘性写入（no free bias、zero-code strict no-op）。
- condition 读自**上下文携带目标**的残差（agent 从记忆/上下文知道 subgoal），应用于**目标剥离**路口动作前向——这才是可部署的读→扳环。
- PASS 只表示可准入 S3 Internal RL 的正式预注册；不授权 production。
