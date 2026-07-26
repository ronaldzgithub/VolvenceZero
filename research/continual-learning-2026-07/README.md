# 持续学习（Continual Learning）业界路径专项 · 2026-07

> Status: research note，外部证据，**不是 runtime contract，不进主链**。
> 论文：25 篇，全部严格对 `research/papers/**` 既有 311 篇 PDF 去重后新增。
> PDF: [`../papers/continual-learning-2607/`](../papers/continual-learning-2607/)
> 下载脚本: [`../download_continual_learning_2607.sh`](../download_continual_learning_2607.sh)
> 下载校验: [`download-summary.md`](download-summary.md)（25/25 成功，0 付费墙）

---

## 阅读顺序

1. [`01_LANDSCAPE.md`](01_LANDSCAPE.md) — 七派全景 + 逐篇技术要点 + 两条决定性负面结果
2. [`02_VZ_DELTA.md`](02_VZ_DELTA.md) — 与 VolvenceZero 的逐条对照、7 条值得借鉴项（含落点）、4 条明确划界项、2 个未决争论
3. [`03_PERSONAL_PARAMETRIC.md`](03_PERSONAL_PARAMETRIC.md) — **第二轮专项（22 篇）**：per-user LoRA / PEFT 个人参数化路线，Mindverse Second Me 技术拆解及其七族替代方案（含超网络摊销、Cartridges 上下文蒸馏、知识编辑的证伪）

已有的邻接专项（不重复）：
- [`../owm-continual-learning-2026-06/analysis.md`](../owm-continual-learning-2026-06/analysis.md) — OWM 正交权重调制（正则化派的几何实现）
- [`../neuromorphic-dmp-2026/analysis.md`](../neuromorphic-dmp-2026/analysis.md) — fast/slow 记忆分离
- [`../bolt-2026-07/`](../bolt-2026-07/) — online-fast 后验更新
- Nested Learning / Titans / Miras / Atlas / MesaNet / ACE / EvolveMem 等已在 `../papers/` 与 `../frontier-map-2024-2026.md`、`../frontier-sweep-2026-07-20.md` 覆盖

---

## 一句话总判断

> **2026 年持续学习领域最重要的两项进展都是负面结果，而它们同时证伪了这个领域自己的两条默认工程实践。**

1. **CL-BENCH（2606.05661，UC Berkeley + Snorkel）**：在 6 个真实有状态领域上，**naive in-context learning 打败了所有专用记忆系统**；最好的系统只吃到 25.4% 的可用 headroom。累积的 state 经常帮倒忙——记忆模块引入虚假泛化与陈旧信念。
2. **Spurious Forgetting（2501.13453，ICLR 2025）**：大部分被称作"灾难性遗忘"的现象**根本不是知识丢失**，而是**任务对齐（task alignment）被底层近正交更新掀翻**。仅冻结底部若干层，就把 SEQ 基线从 11% 拉到 44%，而所有正则化 / replay / model-merging / gradient 方法加起来最高只到 22%。

合起来的含义：**"加更多记忆 = 学得更好"是假的，"指标掉了 = 知识没了"也是假的**。整个领域的主流工程实践，有相当一部分建立在对失败模式的错误诊断上。

对我们的直接后果写在 [`02_VZ_DELTA.md`](02_VZ_DELTA.md) 的 B / C 两条：我们目前**记忆写入是无门的**（ModificationGate 没覆盖到 CMS 沉淀 / reflection writeback），而我们的 evaluation cascade **不区分"对齐掉了"和"知识掉了"**。这两条都不需要新算法，只需要把已有的 gate 与证据语义扩到当前没被覆盖的写面。

---

## 七派速查

| 派别 | 核心问题 | 代表 | 对 VZ 的性质 |
|---|---|---|---|
| **S0 评测/立场** | 怎么才算真的学到了 | CL-BENCH、AGENTCL、Modular Memory 立场书 | **证伪工具**，直接可搬 |
| **S1 稀疏定位写入** | 参数往**哪里**写 | Sparse Memory Finetuning (Meta) | **最高价值借鉴**，天然满足 R10/R15 |
| **S2 自编辑 RL** | 让模型自己决定怎么改自己 | SEAL (MIT) | **R10 反例**，借 reward 结构不借写面 |
| **S3 机理诊断** | 遗忘到底是什么 | Spurious Forgetting、LRCP | **诊断口径**，改我们的归因逻辑 |
| **S4 持续预训练** | 大规模 refresh 怎么做 | LR re-warm + re-decay + replay | rare-heavy 的成熟配方，直接用 |
| **S5 模块化/合并** | 多个 adapter 怎么共存 | Merge-before-Forget、CP-MoE、ProCL | 在**重新发明**我们的 CMS 多时间尺度 |
| **S6 测试时训练** | 上下文能否压进权重 | TTT-E2E (Stanford/NVIDIA) | 架构上是 R2 反例，工程细节可借 |
| **S7 Agent 记忆** | 冻结基座 + 外部状态 | RIZZ、Janus、ALMA、SSGM、CLaaS | 最热闹也最脆；Janus/SSGM 值得抄 |

---

## 我们的位置（一句话）

业界 25 篇里，**没有一篇把持续学习当作 owner 架构与治理问题**——它们要么在优化一个算法（S1–S6），要么在优化一个记忆数据结构（S7）。我们的 R1（四时间尺度）+ R2（冻结基底）+ R5/R6（CMS）+ R9（层级信用）+ R10（门控自修改）+ R15（可回滚）是**唯一一套把"什么时候允许写、写多少、怎么回滚"写进契约的**。

同时有两处我们独有、业界完全没有对应物的结构：

- **R7 双轨（world/self）**：25 篇全部只有一个学习目标（任务成功）。SSGM 把 goal/role drift 列为风险，但没有任何一篇把关系轨做成一等公民。
- **R-PE（内禀预测误差为原始信号）**：业界的学习信号源要么是外部 verifier（RIZZ、CLaaS）、要么是下游任务 reward（SEAL）、要么是标签（S1–S5）。没有人从内禀 PE 出发。这既是我们的独特性，也是我们最大的证伪风险敞口——**没有外部锚，就更难自证**。这正是要把 S0 的 gain metric 搬进来的原因。
