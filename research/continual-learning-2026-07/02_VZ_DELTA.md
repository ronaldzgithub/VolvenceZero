# 与 VolvenceZero 的差异、可借鉴项与划界

> 配套阅读：[`01_LANDSCAPE.md`](01_LANDSCAPE.md)（业界七派全景）。
> 本文所有关于 VZ 现状的陈述都在 2026-07-26 的 `main`（`d56bdf2`）上核对过代码与 spec，出处逐条标注。
> Status: research note。**不是 runtime contract，不进主链**；下面的"落点"是建议，不是已批准的变更。

---

## 1. 结构性差异：我们不在业界任何一派里

| 维度 | 业界 25 篇的做法 | VZ | 性质 |
|---|---|---|---|
| 持续学习被当作 | **算法问题**（S1–S6）或**数据结构问题**（S7） | **owner 架构 + 治理问题** | 我们独有 |
| 时间尺度 | 多数只有一个；S5 有快慢两个 | 四个（online-fast / session-medium / background-slow / rare-heavy），R1 | 我们更完整 |
| 基座策略 | 可改（S1–S6）或完全冻结（S7） | live 冻结 + rare-heavy 离线 owner path，R2 | 与 S7 同，但保留了 S1 的门 |
| 学习目标数 | **全部只有 1 个（任务成功）** | **2 个（world/self 双轨），R7** | **25 篇里零对应物** |
| 学习信号源 | 外部 verifier / 下游 reward / 标签 | **内禀 prediction error，R-PE** | **25 篇里零对应物** |
| 写入门控 | 2026 上半年才首次提出（Janus、SSGM） | ModificationGate + PeWriteGate 已在代码里 | 我们领先约一个身位 |
| 回滚 | 几乎无人处理 | R15 写进契约 | 我们独有 |

两处业界完全没有的东西，值得单独说：

**R7 双轨**。25 篇论文的学习目标全是"把任务做对"。SSGM（2603.11768）把 **Goal / Role Drift**（"长期角色扮演中，累积交互偏置导致对齐漂移"）列为一类记忆失败，缓解建议是 "Role Partitioning"——这是整个语料里离双轨最近的一句话，但它是当作**风险**提的，不是当作**学习轨**。我们把关系轨做成一等公民（独立 `z_rel`、独立信用分配、独立评估指标 F2/F3，见 `docs/specs/dual-track-learning.md`），在这批论文里没有任何对应物。

**R-PE**。业界的信号源要么是外部 verifier（RIZZ 的 fuzzy ratio、CLaaS 的环境 reward）、要么是下游任务表现（SEAL）、要么是监督标签（S1–S5）。**没有人从内禀预测误差出发。**

这是我们的独特性，**也是我们最大的证伪风险敞口**：外部 verifier 派有一个天然的、第三方可复核的锚点，我们没有。这直接决定了下面 §2.E 的优先级——**我们比任何人都更需要 gain metric**，因为我们比任何人都更难自证。

---

## 2. 值得借鉴（按优先级，含落点）

### A.【最高】Sparse Memory Finetuning 的槽定位 → rare-heavy 写入定位 + R10/R15 的构造性满足

**证据**：Sparse Memory FT（2510.15103，Meta）在同等新知识获取下，held-out NaturalQuestions F1 只掉 **11%**，而 **LoRA 掉 71%、full FT 掉 89%**。

**为什么这条对我们特别重要，而不只是"又一个好方法"**：

我们 rare-heavy 目前的写入面是 LoRA / PEFT bake（`vz-substrate/rare_heavy_training.py`、`lifeform-domain-figure/lora_bake_peft.py`）。论文明确给出了 LoRA 遗忘更严重的**机理解释**：LoRA 参数少，但低秩更新**稠密地作用于 hidden state**，一次更新仍然全局影响——**参数少 ≠ 影响面小**。这不是调参能解决的，是参数化形式本身的性质。

更关键的是，稀疏槽写入**天然构造性地满足我们两条最吃紧的不变量**：

- **R10（有界自修改）**：被写的槽数 `t` 是一个**显式、可记账、可预算的整数**。这和 OWM 的 `rank(P)` 记账是同一类东西（见 [`../owm-continual-learning-2026-06/analysis.md`](../owm-continual-learning-2026-06/analysis.md) §4.1），但比 OWM 更适合我们——OWM 的正交投影是端到端 substrate 在线更新（R2 反例），稀疏槽写入是离线 rare-heavy 的定点写入，不冲突。
- **R15（可回滚）**：**回滚代价从 O(全模型) 降到 O(t)**。只需存被改的 `t` 个槽的旧值。这是我们目前 rare-heavy 回滚证据**最薄弱的一环**——现在的 LoRA bake 回滚靠整个 adapter artifact 的版本管理，粒度粗、证据弱。

**落点（可直接排期）**：2604.05248 已经给出把 **Qwen-2.5-0.5B** retrofit 成 memory-layer 模型的开源 pipeline，消费级硬件可跑——**和我们 `state_kv` / substrate lane 用的是同一个底座**（见 `README.md` 的 State-KV P1 lane、`scripts/run_state_kv_identification.py`）。

建议做成一条 P1 实验：在 `vz-substrate` 的 bounded adapter-delta 入口**旁边**加一条 memory-slot 写入路径（不是替换），背景语料用我们自己的 pretraining proxy 算 IDF，产出 `artifacts/` 下的 slot-diff 作为回滚证据。判据直接用论文的对照结构：**同等新知识获取下的 held-out 退化**。

**两个免费的坑（论文踩过了）**：

1. **优化器是一等变量**。作者原本全用 AdamW，后来发现"per-parameter 自适应步长、weight decay、momentum 会与稀疏性发生意外交互"；换 **SGD** 后 held-out 遗忘进一步下降——而 full FT / LoRA 换 SGD 没有类似收益。如果我们跑这个实验时沿用现有 AdamW 配置，会**吃不到大部分收益并误判该方法无效**。
2. **排序基于 batch，不假设任务边界**。连续 batch 可以来自完全不同的分布。这对我们的流式部署是好消息，不需要额外的任务切分。

**一个必须自己验证的开放问题**：论文**全部实验都是事实型 QA**（TriviaQA / SimpleQA / NaturalQuestions）。关系、人格、风格这类**分布式属性**能否被稀疏定位，是完全开放的问题——**而这恰好是我们的主战场**。这条不能假设，必须自己测。

### B.【高】Janus 的"记忆更新 = 部署决策" → 补上我们唯一没有门的写面

**先纠正一个容易搞错的判断**：我们的记忆写入**不是无门的**。现状是：

| 写面 | 现有门 | 位置 |
|---|---|---|
| PE 驱动的条目写入 | `PeWriteGate`——bounded-learned 准入阈值，envelope 硬夹（±0.10），`reset()` 精确回滚到固定阈值 | `vz-memory/memory/pe_write_gate.py` |
| CMS 各频段吸收 | `write_gate` 作为**混合速率**（连续吸收率），按时间尺度分别决策 | `vz-memory/memory/cms.py` |
| 反思 → 协议修订 | R10 `ModificationGate`（SHADOW 默认，需 matched-control 证据才 ACTIVE） | `vz-cognition/reflection/engine.py` |

**真正的缺口是另一个东西**：上面三个门全都是**前瞻式**（对单次写入的阈值/速率）或**基于内部有用性的回顾式**。**没有任何一个门问："新的记忆状态在一组保留任务上，是否比旧的记忆状态表现更好？"**

Janus（2606.31121）指出的正是这一点：盲目接受每次记忆更新，会让最终记忆**覆写有用知识、引入过度特化的规则、偏向近期样本**。

**而且这里有一个直接命中我们的细节**。`PeWriteGate` 的结算信号是：**条目被再次检索（touched 且净强度未损）或被提升 = 有用；衰减、删除、从未再访 = 无用**。这是一个**内部自洽性信号，且与近因高度相关**——恰恰就是 Janus 与 CL-BENCH 共同点名的失败模式（CL-BENCH：系统"过度依赖最近的任务实例，低估较早但相关的任务变体"）。**一条被反复检索的记忆，和一条能改善未来任务的记忆，不是同一件事。**

**落点**：在 `vz-memory` 的沉淀/写回路径上加一层 **memory-version 级**的比较门，复用我们已有的 gate 语义（`validation_delta` / `rollback_evidence`，见 `docs/specs/credit-and-self-modification.md`）。Janus 的两个设计让它便宜到可以落地：

1. **Memory Momentum Trigger**——只在候选更新**偏离近期更新轨迹**时才触发对比评估；不偏离就直接接受。**绝大多数轮次零额外开销。**
2. **紧凑混合评估集**代替完整历史重放，三类构成：**coverage**（已见任务分布）+ **boundary**（过去记忆选择曾改变正确性的任务）+ **fresh**（最近任务切片）。

其中 **boundary 集**这个概念对我们特别有用——它就是"记忆是否起作用的分水岭样本"，天然对应我们 evidence 体系里的 kill-condition 样本。

论文效果：6 数据集 × 2 骨干 × 2 updater，平均 **+2.7 ~ +4.6** 分。注意它是 **method-agnostic 的插件**，不改底层 updater 的更新规则——这与我们"不动 owner 内部、只加门"的架构习惯完全兼容。

### C.【高】Spurious Forgetting 的归因二分 → 改掉一条会让我们回滚掉有效 refresh 的逻辑

**这条最便宜，也最容易被忽略。**

`Spurious Forgetting`（2501.13453，ICLR 2025）证明：

```
任务表现 = 任务对齐 (task alignment) + 底层知识 (underlying knowledge)
```

新任务训练的**前 ~150 步**，底层（含输入嵌入）发生近正交更新，**掀翻已建立的任务对齐**，表现从接近 100% 崩到约 10%——**但知识还在**，用少量旧数据 replay 即可重新对齐恢复。仅冻结底部若干层：SEQ 从 **11% → 44%**，而所有正则化 / 生成式 replay / model-merging / gradient 方法最好只到 **22%**。

**对我们的直接后果**：我们的 evaluation cascade（`docs/specs/evaluation-cascade.md`、`vz-cognition/evaluation/cascade.py`）目前**不区分这两种下降**。任何 rare-heavy refresh 之后的指标下降，都会被当作退化处理。

按这篇论文，**这是一个会让我们回滚掉有效 refresh 的假信号**。而且我们特别容易撞上：我们的 rare-heavy 就是"在新数据上继续训练一个已对齐的模型"——正是论文里 spurious forgetting 最典型的触发场景（论文验证的四个真实场景之一就是"安全对齐"，与我们的 boundary policy 同构）。

**落点**：在 `evaluation-cascade.md` 加一条**归因二分**要求——rare-heavy refresh 后的指标下降，在做过下述对照之前**不构成回滚触发条件**：

- 少量旧数据 replay 后是否恢复？→ 恢复 = spurious（对齐问题），不恢复 = 真遗忘
- 底层冻结的对照臂是否不下降？→ 不下降 = spurious

这条**不需要新算法、不需要新 owner**，只是把一个二值判断插进已有的 failure semantics（F1–F4）。

配套证据：LRCP（2601.18699）在 20 个 2026 模型上给出空间定位——**早层 attention 熵扩散 + 中深层 FFN/专家表示塌缩**。它和 Spurious Forgetting 是同一个故事的两半：**遗忘有明确空间定位，不是弥散的**。这也是 §2.A 稀疏定位写入路线成立的理论基础。

### D.【中】CL-BENCH / AGENTCL 的评测方法学 → companion-bench 的两个缺口

核对结果：`docs/specs/companion-bench.md` 中**没有 stateless 基线、没有 gain、没有 headroom 归一化**的概念。两条可直接搬：

**D1. gain metric**（CL-BENCH）

```
ĝ = (r̄_stateful − r̄_stateless) / (r_max − r̄_stateless)
```

我们现在报的是**绝对表现**。这意味着我们**无法区分"基座本来就会"和"我们的架构学到了"**——这是我们对外声称"会成长"时最容易被打穿的地方，而且因为我们走 R-PE 内禀信号路线（没有外部 verifier 锚点），这个缺口比对别人更致命。

CL-BENCH 还有一个细节值得抄：**归一化 reward 用固定外部参照**（他们用 GPT-5.4 ICL stateless）而非各系统自身基线，**使分数与提交无关**——新系统加入不会改变旧系统的分数。这对我们做纵向对比（`run_longitudinal_continuity.sh`）尤其有用。

**D2. compositional stream vs naive stream**（AGENTCL）

AGENTCL 的核心方法学发现是：**随机任务流区分不出记忆设计**，它把差异压平了。必须构造**早期子解/证据/workflow 被有意设计成可在后期复用**的组合式流。

我们仓库根目录的 `three_path_20turn_benchmark.json` / `three_path_50turn_mixed_benchmark.json` 如果是随机拼的，**按这篇论文它就没有区分力**——不是分数不好看，是**分数好看也说明不了问题**。

配套的三个指标可以直接映射：**Plasticity Gain**（早期记忆是否帮到后期）/ **Stability Gain**（第二趟只读，经验是否持久可复用）/ **Generalization Gain**（冻结记忆后在 held-out 上的迁移）。第三个我们已有 held-out 子模块（`companion-bench.md` §7），差的是前两个的两趟结构。

> 这两条属于**证伪工具**，不是性能工具——正好对齐 `docs/specs/evidence_program.md` 的口径。

### E.【中】SSGM 的记忆失败四维分类 → 直接变成 kill-condition 清单

SSGM（2603.11768）的分类表（完整表见 [`01_LANDSCAPE.md`](01_LANDSCAPE.md) §S7-4）里，有四条**结构性地命中我们**，不是"可能相关"：

| SSGM 失败模式 | 为什么必然命中我们 | 对应我们的 R |
|---|---|---|
| **Semantic Drift**（迭代摘要导致细微差别单调流失） | CMS 有多频段迭代压缩，**有损压缩是 semantic drift 的直接驱动因子** | R5/R6 |
| **Goal / Role Drift**（累积交互偏置导致对齐漂移，长期角色扮演场景） | 我们做的就是长期角色扮演 + 持久身份 | **R14** |
| **Temporal Obsolescence**（陈旧记忆与新状态冲突，用户个性化场景） | 我们的 growth-advisor / figure 垂类全是长周期个性化 | R5 |
| **Privacy Leakage**（多租户 agent 的跨会话/跨用户检索） | DLaaS 是多租户 | R8/R12 |

**Goal/Role Drift 这条最值得单拎**：它是整批论文里唯一触及"关系/角色轨也会漂移"的地方，而我们有 R14（regime 持久身份）却**没有 role drift 的检测口径**。SSGM 给的缓解是 Role Partitioning——我们的双轨隔离在结构上已经做了一半，缺的是**漂移度量**。

**落点**：把这张表转成 `docs/specs` 下的 kill-condition 清单。它的价值不在缓解手段（Weibull 衰减、写入防火墙这些都很朴素），在于**它是一份别人踩过的坑的完整枚举**——我们照着自查比自己想更快。

### F.【中】TTT-E2E 的三条工程经验 + 一次概念让步

**先划清**：TTT-E2E（2512.23675）的架构本身是 **R2 反例**——它在测试时改 substrate 权重。**不抄架构。**

但三条经验是免费的先验，如果我们未来在 adapter 上做任何在线快速更新：

1. **只更新 MLP 层**——在内循环更新 attention 层会**导致外循环不稳定**。
2. **只对 1/4 的 block 做 TTT**——更新更多层的收益/成本存在明确拐点。
3. **mini-batch 而非在线单 token 梯度**——单 token 梯度步"很容易偶然导致梯度爆炸"，且无法并行。

另外它开篇的框架转换值得作为外部引用：

> 我们把长上下文语言建模**表述为一个持续学习问题，而不是架构设计问题**。

这正好支持我们把长程上下文做成 **CMS（memory）而不是 context window** 的选择。一个 Stanford/NVIDIA/Berkeley 的联合工作在 3B/164B token 规模上做出这个让步，是我们 R5 的强外部证据。

### G.【低但应固化】持续预训练三件套 → rare-heavy 的默认配方

**LR re-warming + LR re-decaying + 少量旧数据 replay**（2403.08763，TMLR）在 **10B 参数规模**上被证明足以**匹配从头全量重训**，计算量只是零头。

**rare-heavy 尺度不需要新算法。** 如果 `run_learned_active_evidence.sh` / `rare_heavy_training.py` 还没固定这个配方，就固定它——并把它当作任何 rare-heavy 新方法必须先超过的基线。

---

## 3. 明确划界：不该借鉴的四条

### 3.1 SEAL 式自编辑 —— R10 反例

SEAL（2506.10943，MIT）让模型生成 self-edit（自己的训练数据 + 超参）并直接写权重。**违反 R10 的三个方面**：没有容量预算、没有回滚证据、改什么完全由模型生成物决定。

不需要我们自己论证——**论文自己承认**：

- 顺序 self-edit 流下早期任务表现**持续下降**，仍然灾难性遗忘；
- **每次 self-edit 评估需 30–45 秒**（必须微调并评估整个模型来算 reward），"实时连续编辑在生产上不可行"；
- 当前实现要求每个 context 都配显式下游任务，**无法扩展到无标注语料**。

**可以借的**：用"下游表现给自修改提案打分"这个 reward 结构——这和我们 `validation_delta` 的语义同构，也和 ReST-EM（拒绝采样 + SFT）的过滤式思路一致。它还提供了一个有用的负面工程情报：作者试过 **GRPO 和 PPO 都训练不稳定**，因为奖励依赖动作发生时的模型参数 θ，旧数据会过时失配。我们若在 self-modification 上做 RL，这是免费的避坑。

**不能借的**：模型生成自己的训练数据并直接写权重这个写面。与 DGM 同类风险（见 [`../README.md`](../README.md) §三.2）。

### 3.2 CLaaS 式在线参数热重载 —— R2 反例

CLaaS（2606.05559）把 LoRA 更新**热重载进推理服务器**形成实时改进闭环。这是我们明确禁止的写面（R2：live 默认冻结，rare-heavy 仅离线 owner path）。

**可以借的**：experience replay buffer + 异步训练 + 梯度复用的**样本效率结构**——但只能落在 background-slow / rare-heavy，不能进 online-fast。它排除 GRPO 的理由也值得记：GRPO 依赖 group 统计量，需要可重置的离线环境，而"真实世界环境不能被轻易重置，每个场景只能采样一次"——这个约束和我们的真实部署完全一致。

### 3.3 ALMA 式开放代码搜索 —— Two-Gate 反例

ALMA（2602.07755）让 Meta Agent 以**代码为搜索空间**开放式探索记忆设计，"理论上可发现任意记忆设计"。

**"理论上可发现任意设计" = policy-reachable 模型族的 VC 维无界**，直接违反 Two-Gate（2510.04399）的有界条件——见 [`../README.md`](../README.md) §二.3。

**可以借的**："记忆设计本身是可搜索对象"这个观念，以及它的 archive + 反思 + 评估日志回写的搜索循环结构。**但搜索空间必须是我们已声明的 slot / schema 参数空间，不是任意代码。**

### 3.4 "加更多记忆系统"这个默认反射 —— 被两个独立基准证伪

CL-BENCH：**naive ICL 在多数任务上超过专用记忆架构**；最好的系统只拿到 25.4% 的归一化 gain；累积 state 经常帮倒忙（虚假泛化 + 陈旧信念）。
AGENTCL：现有记忆设计在 naive 与 held-out 设置下**频繁引发认知干扰或性能退化**。

**工程含义**：在能证明某个记忆 owner 带来正 gain 之前，**它应该被当作负债而不是资产**。这条应该直接进我们新增 memory owner 的准入条件——先有 gain 证据，再有 owner。

这与我们已有的 SHADOW → ACTIVE 证据门是同一个精神，只是把**判据**从"forward parity / 不劣化"收紧到"**正 gain**"。

---

## 4. 两个未决争论（值得盯，也是我们能吃到的空档）

### 4.1 参数更新 vs 外部记忆 vs 纯 ICL —— 一个没人做过的三路对照

两条结论方向相反：

- **CLaaS**：参数更新的 forward transfer **优于** in-context learning，且遗忘更少；replay 是样本效率的关键。
- **CL-BENCH**：专用记忆系统**不如** naive ICL。

严格说这不矛盾（一个比"参数更新 vs ICL"，一个比"记忆系统 vs ICL"），**但没有任何一篇在同一任务流上把三路一起比过**。

而**我们的四时间尺度架构恰好是唯一能同时跑三路的**：frozen + naive ICL（online-fast 关掉记忆写入）/ 外部记忆（CMS 正常）/ 有界参数写入（rare-heavy 稀疏槽）。用 §2.D 的 gain metric 量。

这是一个**别人做不了、我们能做、且结论对整个领域有价值**的实验。如果要选一件事把这份调研变成产出，我选这个。

### 4.2 稀疏定位写入能否扩到"技能"和"关系"，而不只是"事实"

Sparse Memory FT 的全部实验都是**事实型 QA**。事实是**局部的**——一条事实对应少数记忆槽，这是 TF-IDF 定位成立的前提。

但**关系、人格、风格、边界策略是分布式属性**——它们大概率**不对应稀疏的槽集合**。如果是这样，稀疏定位写入在我们的主战场上会**退化成普通的稀疏微调，失去 11% vs 71% 的优势**。

这个问题**业界没有答案，而它决定了 §2.A 对我们到底是最高价值还是无关**。所以 §2.A 的实验设计必须从一开始就包含一条关系/风格的臂，不能只复现事实型 QA 的结果就宣布成功。

---

## 5. 一页纸行动摘要

| # | 事项 | 类型 | 成本 | 落点 |
|---|---|---|---|---|
| A | 稀疏记忆槽写入路径（Qwen2.5-0.5B retrofit），**必须含关系/风格臂** + SGD 而非 AdamW | 新实验 | 中 | `vz-substrate` bounded adapter-delta 旁路 |
| B | memory-version 级比较门（momentum trigger + coverage/boundary/fresh 评估集） | 补门 | 中 | `vz-memory` 沉淀/写回路径 |
| C | 遗忘归因二分（对齐 vs 知识），未做对照前不得触发回滚 | 改判据 | **低** | `docs/specs/evaluation-cascade.md` |
| D | gain metric（stateful − stateless，headroom 归一化，固定外部参照）+ compositional stream | 改评测 | 中 | `docs/specs/companion-bench.md`、`three_path_*.json` |
| E | SSGM 四维失败表 → kill-condition 清单，重点补 **role drift 度量** | 补清单 | **低** | `docs/specs/` |
| F | TTT 三条工程经验存档（只更 MLP / 1/4 block / mini-batch） | 存档 | **低** | 本文即可 |
| G | 固化 rare-heavy 配方（LR re-warm + re-decay + replay）为强制基线 | 固化 | 低 | `rare_heavy_training.py` |

C、E、F、G 是低成本的判据与清单修正，**不需要新 owner、不需要新算法**，但 C 直接防住一类会让我们回滚掉有效 refresh 的假信号，E 直接补上 R14 的一个检测缺口。

A 和 4.1 的三路对照是**同一条实验线**——A 提供"有界参数写入"那一路，D 提供度量。三者一起做比分开做便宜。
