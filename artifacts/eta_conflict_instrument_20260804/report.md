# ETA 冲突映射仪器有效性（P2c · C1，只读）

> 范围：目标剥离路口仪器的结构余量 + 基底不确定性；不训练参数、不加 bias、不改 production。

## 结论

- 仪器有效性：`VALID`
- 观测协议：`goal-ambiguous-junction.v5`
- failed conditions：`()`

## 结构余量（heldout，无模型）

| 指标 | 值 |
|---|---:|
| conflict row fraction | 1.000 |
| constant-operator error | 0.461 |
| oracle (view,subgoal) error | 0.000 |
| view-subgoal residual ambiguity | 0 |
| unique local views | 10 |
| rows / mean out-edges | 165 / 5.87 |

## 基底不确定性（heldout，frozen merged 模型）

| 指标 | 值 |
|---|---:|
| goal-stripped expert NLL (mean/median) | 2.8129 / 2.7871 |
| subgoal-revealed expert NLL (mean) | 0.2178 |
| steerable headroom (stripped − revealed) | 2.5951 |
| fraction base uncertain (NLL>0.1) | 0.806 |

## 守门边界

- 恒定算子错误率证明无条件映射不足；(view,subgoal) 残余歧义为 0 证明 subgoal 是唯一缺失比特。
- 本结果只判仪器是否值得跑 C2 条件化学习式 steering screen；不改写任何已封存 verdict，不授权 production。
