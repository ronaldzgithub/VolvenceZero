# ETA 门控余量审计（S3-A，只读诊断）

> 证明「何时扳」在本仪器上有可测且可观测的门控余量，再花 Internal RL 算力。过期 belief 在切换路口错条件（灾难），noop 平庸，择时可赢。不安装控制器、不改 production、不回灌 evaluation、不训基底。

## 结论

- 门控余量 admission：`PASS`
- failed：`()`
- post-switch 行占比：`0.430`（阈值 ≥0.1）

## 各臂 heldout expert NLL（全体行均值）

| arm | NLL |
|---|---:|
| noop（目标隐藏基线） | 2.8129 |
| always_on_belief（过期条件恒定出手） | 1.7909 |
| **oracle_gate_belief（择时上界）** | 1.0903 |
| pe_gate_belief（可观测一致性门） | 1.0903 |
| fresh_ceiling（08 online 参考） | 0.0271 |

## post-switch 子集（belief 过期处）

| arm | NLL |
|---|---:|
| noop | 2.5324 |
| always_on_belief | 4.1607 |
| fresh_ceiling | 0.0616 |

## 门控余量与可观测性

| 量 | 值 | 阈值 |
|---|---:|---:|
| 余量 = always_on − oracle_gate | 0.7007 | ≥0.3 |
| 增益 = noop − oracle_gate | 1.7226 | ≥0.3 |
| 可观测门捕获 = always_on − pe_gate | 0.7007 | — |
| staleness 可检测性 P(belief≠fresh \| post-switch) | 1.000 | ≥0.5 |
| 误报 P(belief≠fresh \| 非 post-switch) | 0.000 | — |
| reader margin（belief 上下文）均值 | 0.951 | — |

## 含义

- oracle_gate 明显优于 always_on ⇒ **择时有价值**（错条件出手是净损）。
- oracle_gate 明显优于 noop ⇒ 择时仍胜「什么都不做」。
- pe_gate（belief 与 fresh 读一致才出手）已捕获大部分余量 ⇒ **存在可观测信号**，S3-C 的策略有东西可学（RL 从稀疏结局信用学阈值/组合，而非硬编码规则）。
- staleness 可检测性高 ⇒ 门控信号存在；这是 S3 从 PE 代理学「何时扳」的前提。

PASS ⇒ 准入 S3-C（Internal RL 学何时扳）；不改写任何封存 verdict。