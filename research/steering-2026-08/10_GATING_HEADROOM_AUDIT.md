# 10 · 门控余量审计（S3-A，已执行只读）= **PASS（余量存在且可观测）**

- run 脚本：[`scripts/run_eta_gating_headroom_audit.py`](../../scripts/run_eta_gating_headroom_audit.py)（复用 08 的 capture/reader 与 C2 的执行器，只读诊断）
- 产物：`artifacts/eta_s3a_gating_headroom_20260805/`（`report.json` / `report.md` / `artifact_manifest.json`）
- 基底：frozen merged S1 模型 `artifacts/eta_stage2_merged_v2_20260803`，layer 20 / 896，MPS
- 配置：rank 8，operator seed 0，updates 80，lr 0.01，batch 32，307 train / 165 heldout junction rows，K=8

## 一句话结论

**「何时扳」在本仪器上有可测且完全可观测的门控余量，S3-C（Internal RL 学何时扳）可准入。** 我们用一个诚实的非 oracle 失败模型——**过期 belief**（agent 从上一步上下文读子目标，记忆滞后）——制造余量：路线经过 objective 后 active_subgoal 切换，belief 在这些 **post-switch 路口**过期。用过期 belief 恒定出手在切换处**灾难**（post-switch 子集 always-on 4.16 > noop 2.53），而**择时**（该出手才出手）能把整体 NLL 从 always-on 1.79 降到 **1.09**（余量 0.70），且比什么都不做的 noop 2.81 赢 **1.72**。关键：**过期信号完全可观测**（belief 与 fresh 读一致性检测率 1.0、误报 0），所以可观测门 `pe_gate` **完全等于** oracle 门——S3-C 的策略确有信号可学。

## 主判（heldout，全体 165 行）

| arm | expert NLL | 说明 |
|---|---:|---|
| noop（目标隐藏基线） | 2.813 | 什么都不做 |
| always_on_belief（过期条件恒定出手） | 1.791 | 被切换处的错条件灾难拖高 |
| **oracle_gate_belief（择时上界）** | **1.090** | belief 实际正确才出手，否则 noop |
| pe_gate_belief（可观测一致性门） | **1.090** | belief 与 fresh 读一致才出手（无 oracle） |
| fresh_ceiling（08 online 参考） | 0.027 | 每步用 fresh 读的正确条件 |

| 判定门 | 值 | 阈值 | 结果 |
|---|---:|---:|:--:|
| post-switch 行占比 | 0.430（71/165） | ≥0.10 | ✅ |
| 余量 = always_on − oracle_gate | 0.701 | ≥0.30 | ✅ |
| 增益 = noop − oracle_gate | 1.723 | ≥0.30 | ✅ |
| staleness 可检测性 P(belief≠fresh \| post-switch) | 1.000 | ≥0.50 | ✅ |

**S3-A admission = PASS（4 门全过）。** 结构：`free_bias=false`、`zero_code_strict_noop` max\|Δ\|=0.0、`substrate_trainable=0`、read-only（不安装控制器、不改 production、不回灌 evaluation）。

## post-switch 子集（belief 过期处，71 行）——错条件确实有害

| arm | NLL |
|---|---:|
| noop | 2.532 |
| **always_on_belief** | **4.160** |
| fresh_ceiling | 0.062 |

**过期条件出手（4.16）比什么都不做（2.53）还差 1.63。** 这是「错条件是净损、该收手时要收手」的直接因果证据，也解释了 always-on 为何整体被拖高。

## 三个数字的第一性含义

1. **余量 0.70（always-on→oracle-gate）**：择时把「错条件灾难」挡掉，纯来自「在对的时候出手」。等预算、同执行器、唯一差别是门控 ⇒ 「何时扳」有独立因果价值（不是 P2a 的无余量重演）。
2. **pe_gate == oracle_gate（可观测门捕获全部余量）**：过期在本仪器上**完全可检测**（belief vs fresh 读不一致率：post-switch 1.0、非 post-switch 0.0）⇒ 策略有干净信号可依据。
3. **fresh_ceiling 0.027**：与 08 online 一致，锁定「读得到+扳得动」的上界；门控的活是「什么时候用这只手」。

## 对 S3-C 的意义与诚实边界

- 存在一个 **hard 一致性规则**（belief==fresh 才出手）即可达到 oracle 门 1.09。因此 S3-C 的 RL claim **不是**「只有 RL 能达到这个门」，而是——**在只给稀疏结局信用（terminal+delayed）、只观测 PE 代理、从不给每步对错标签的条件下，Internal RL 能否在小样本内学到达到该上界的门控**。这正对应产品里的真实约束：陪伴任务没有免费的每步对错标签，只有稀疏的关系结局。
- 因此 S3-C 主判 = **sample-budget 收敛门 + 优于 always-on/random-gate + 门控集中在 post-switch**；oracle/pe 硬门作为**可达上界诊断**，不作 RL 的信用来源。
- 不改写任何封存 verdict（`kill-eta` / S2 / B screen / C2 / 08 均不变）。

## 下一步

S3-B：按本审计数字修订 [09_S3_INTERNAL_RL_PREREG_SKELETON.md](09_S3_INTERNAL_RL_PREREG_SKELETON.md)（staleness 仪器、PE 代理、阈值实数化、hard-rule 上界作诊断），冻结正式 prereg；随后 S3-C 实现 owner 模块跑 Internal RL。
