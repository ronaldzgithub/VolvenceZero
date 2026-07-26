# 个人参数化路线（per-user LoRA / PEFT）全景 · Mindverse Second Me 及其替代方案

> 第二轮专项：22 篇新增 PDF，严格对既有 333 篇去重。
> PDF: [`../papers/personal-parametric-2607/`](../papers/personal-parametric-2607/)
> 下载脚本: [`../download_personal_parametric_2607.sh`](../download_personal_parametric_2607.sh)（22/22 成功）
> Status: research note，外部证据，**不是 runtime contract，不进主链**。

---

## 0. 一句话总判断

> **这个领域在 18 个月里完成了一次方向翻转：从"给每个用户训一个 LoRA"翻转到"用超网络在一次前向里造出 LoRA"。Mindverse Second Me 属于翻转前的那一代。**

最有力的证据不是我的判断，是**作者自己的反水**：OPPU（One-PEFT-Per-User，2402.04401）和 PER-PCS（2406.10471）的第一作者 Zhaoxuan Tan，在 2025-10 发了 **Profile-to-PEFT**（2510.16282），开篇就说 OPPU 范式"计算昂贵、对实时更新不切实际"，并用超网络把 per-user 训练**整个消掉**。

---

## 1. 先把 Second Me 拆开：它到底是什么方案

`AI-native Memory 2.0: Second Me`（**2503.08102**，Mindverse.ai + UCSD Jingbo Shang）。

**三层记忆架构**（沿用其 LPM 1.0）：

| 层 | 定义 | 实质 |
|---|---|---|
| **L0** 原始数据层 | 把记忆定义为全部非结构化数据 | **就是 RAG / RALM** |
| **L1** 自然语言记忆层 | 可用自然语言概括的记忆：简历式 bio、关键语句清单、偏好标签 | 结构化摘要 |
| **L2** AI-native 记忆层 | **不必用自然语言表达的记忆，通过模型参数学习和组织**，每个 LPM 就是一个神经网络 | **个人 LoRA** |

**训练管线**（全自动，用户数据彼此隔离）：

```
原始数据 → 数据挖掘（抽实体/主题）→ 记忆合成（self-location reinforcement +
memory cognition enhancement + 多 agent 模拟生成 context-enhance / context-critic 数据）
→ 五级过滤 → PEFT SFT → 从 SFT 模型生成 DPO 数据 → DPO → 自动评估（LLM as Judge）
```

基座 **Qwen2.5-7B-Instruct**。偏好对约占 SFT 训练数据的 **20%**。DPO 不注入新知识，只细化对用户优先级的理解。

**唯一被单独消融的变量是 CoT 风格**（表 1，满分比例）：

| CoT 策略 | Memory (Self) | Memory (Third-Party) | Context Enhance | Context Critic |
|---|---|---|---|---|
| Strong（DeepSeek-R1 生成，严格格式+长度约束） | **0.91** | **0.71** | 0.75 | **0.85** |
| Multi-step | 0.64 | 0.43 | 0.85 | 0.77 |
| Weak | 0.86 | 0.58 | **0.87** | 0.64 |

注意 **Context Enhance 一列反向**——Weak CoT 最好。论文没有解释这个反转。

**2.0 相对 1.0 的关键改动**：**L2 从"任务执行者"改成"编排器（orchestrator）"**，调用外部专家模型处理复杂需求。

### 1.1 三条必须说清楚的定性

1. **Second Me 不是持续学习方案。** 它是**一次性（或周期性重跑）的离线个性化 post-training pipeline**。要更新用户模型就要重跑整条管线。论文自己把 L2 的未解问题列为四条：**训练效率、服务效率、冷启动、灾难性遗忘**——这四条正是持续学习要解决的问题，它明确地把它们留在了门外。

2. **它的形状 = 我们 `lifeform-domain-figure` 的 persona LoRA。** `FigureArtifactBundle`（retrieval index / coverage map / style prior / steering / persona LoRA）通过 `ModificationGate.OFFLINE` 门控——**这是同一个东西，而我们多一个 gate**。

3. **它的全部 ≈ 我们四个时间尺度里的一个（rare-heavy）。** 他们把"个性化"当作终点，我们把它当作最慢的那一层。这不是褒贬，是坐标定位：拿 Second Me 当竞品比较是错位的，它对应的是我们的一个 artifact 烘焙流程。

---

## 2. 七族替代方案

### F1 One-PEFT-Per-User —— 朴素基线（Second Me 属于这族）

**OPPU**（2402.04401，Notre Dame + Amazon，EMNLP 2024）：每用户一个 LoRA 模块存其行为模式与偏好，**参数化用户知识 + 非参数化（检索 + profile）知识结合**。LaMP 七任务上超过所有 prompt-based 方法。

卖点有两个，都不是性能：

- **Ownership**：模型归用户个人所有 → 定制自由 + 隐私
- **Behavior shift 适应**：能跟上用户行为模式的演化

**结构性缺陷**（后续所有工作的出发点）：存储与计算随用户数**线性增长**；用户数据稀疏时从零微调效果差；**用户之间无法共享知识**，个人模型无法获得社区收益。

### F2 协作拼装 —— PER-PCS

**PER-PCS**（2406.10471，同一组）：选出 sharer，把他们的 PEFT **拆成可复用碎片**并为每个碎片训一个门，放进共享池；目标用户用自己的历史数据**自回归地挑选并拼装**出个性化 PEFT，**无需额外训练**。

效果与 OPPU 相当，资源消耗显著更低；只共享一小部分参数，保住隐私与模型所有权。

### F3 超网络摊销 —— 当前的方向（per-user 训练归零）

**这是本轮最重要的一族。** 共同思想：**不训练 LoRA，生成 LoRA。**

| 工作 | 输入 → 输出 | 关键结果 |
|---|---|---|
| **Profile-to-PEFT**（2510.16282，Notre Dame + Amazon） | 编码后的 user profile → 整套 LoRA 参数 | 部署时**零 per-user 训练**；超过 prompt-based 与 OPPU，部署算力大幅更低；**泛化到未见用户（OOD）**；跨不同活跃度用户与不同 embedding backbone 稳健；支持隐私保护的本地部署 |
| **Text-to-LoRA**（2506.06105，Sakana AI，ICML 2025） | 任务的**自然语言描述** → LoRA（单次前向） | 在 9 个预训练 LoRA 上训练后，即席重建的 LoRA **匹配任务专用 adapter**；能**压缩数百个 LoRA 实例**并零样本泛化到全新任务 |
| **Drag-and-Drop LLMs**（2506.16406，NUS / UT Austin / 圣加仑 / Oxford） | 少量**无标注 prompt** → LoRA 权重更新 | 比全量微调低 **12,000×** 开销；在未见的常识推理/数学/代码/多模态基准上比**最强的训练所得 LoRA 平均高 30%**；从未见过目标数据或标签 |
| **Generative Adapter**（2411.05877，Microsoft Research + UW，ICLR 2025） | 测试时上下文 → 低秩 adapter（**单次前向**） | StreamingQA 32K 上下文，F1 **19.5 → 31.5**（比 SFT 高 63.5%）；MetaICL 26 任务平均 44.9；**MSC 用户个性化上比塞完整对话历史省 4× 计算与内存** |

Generative Adapter 的定位尤其值得注意：它明确指出存在**精度—算力权衡**——微调训练成本高，prompting 推理开销高——而"上下文 → 参数"的单次前向映射同时绕开两者。一个 generator 可适配该基座的所有场景。

### F4 上下文 → 权重 / KV 蒸馏 —— Cartridges 线

**Cartridges**（2506.06266，Stanford Hazy Research，Christopher Ré 组）：离线在每个语料上训一个**小 KV cache**（称作 cartridge），推理时加载。训练成本可在**所有引用该语料的查询上摊销**。

**关键负面发现**：**朴素地用 next-token prediction 在语料上训 cartridge，打不过 ICL。** 必须用 **SELF-STUDY**——生成关于语料的**合成对话**，并用 **context-distillation 目标**训练。

结果：匹配 ICL 质量的同时，**内存少 38.6×，峰值吞吐高 26.4×**；把有效上下文从 128k 扩到 **484k**（MTOB 教材，比 ICL 在前 13 万 token 上高 11.0 chrF）；且**多个 cartridge 可在推理时直接拼接联合查询，无需联合优化或重训**。

**Learned Structure in Cartridges**（2508.17032）后续分析：cartridge 里的 **key 充当可共享的路由器**——这解释了为什么它们可组合。

**Context Distillation as Latent Memory Management**（2605.28889，CUHK + 华为诺亚）：把上下文蒸馏形式化为**潜在记忆管理问题**。每个上下文蒸馏成**独立 LoRA**，构成模块化记忆库；查询时检索候选记忆 → 路由到最合适的 adapter → **Self-Gating 机制决定该潜在记忆是否应被激活**。作者明确指出 Self-Gating 的价值在于**关掉不必要的潜在记忆来提升鲁棒性**。

**Do LMs Need Sleep?**（2605.26099）：类睡眠固化机制——周期性地把近期上下文转成**持久 fast weights**，然后清空 KV cache；模型对累积上下文做离线递归遍历，用学到的局部规则更新 fast weights。

### F5 知识编辑 —— 这条基本被证伪

**WikiBigEdit**（2503.05683）用 Wikidata 快照两两差分构造真实世界知识演化，8 个时间片、5 个 LLM，对比知识编辑 / RAG / 持续微调（LoRA + merging）：

> **RAG 大幅超过专用知识编辑技术**（代价是更高推理成本）。**在同等推理成本下，简单的持续微调在规模上持续优于高级编辑技术。**

配套证据：**WISE** 在前 10K 次更新内表现稳步下降，最终**收敛回更新前的水平**（少于 500 次更新时与 RAG 相当）；**AlphaEdit** 从 1 次编辑到 1000 次编辑，准确率从 **0.96 掉到 0.72**。

`Lifelong Sequential Knowledge Editing without Model Degradation`（2502.01636）与 `The Labyrinth and the Thread`（2605.26670）分别从模型退化与正则化设计角度做了缓解尝试，但都没有推翻上述量级的结论。

### F6 服务基础设施 —— 决定"每用户一个 LoRA"是否可行的那一层

**S-LoRA**（2311.03285，Berkeley / Stanford，Ion Stoica 组）：

- 所有 adapter 存**主存**，只把当前运行查询用到的取进 GPU
- **Unified Paging**：统一内存池同时管理**不同 rank 的动态 adapter 权重**与**不同序列长度的 KV cache**，减少碎片
- 把**可批处理的基座计算**与**各自的 LoRA 计算**分离
- 定制 CUDA kernel 直接操作非连续内存；新的张量并行策略

结果：**单 GPU（或跨 GPU）服务数千 adapter**，相比 HuggingFace PEFT 与 vLLM 吞吐提升至多 **4×**，可服务 adapter 数量高**几个数量级**。

> 这是"one-PEFT-per-user"从幻想变成工程可行的原因。对应我们的 `docs/specs/persona-lora-concurrency.md`。

### F7 紧凑用户表示 —— 不用 adapter

**TAP-PER**（2606.04547，Microsoft）：把用户偏好编码成**可学习的 prefix embedding**，取代 per-user adapter。把用户建模分解为 **user-state** 与 **query-conditioned** 两个分量，并引入**时间信号**捕捉兴趣演化。

LaMP 六任务上超过 prompt-based 与 model-based 基线；**每用户参数比 OPPU 少 130×**，在 1000 用户规模下总参数占用约为 PER-PCS 的**一半**。

**MTA**（2511.20072，港城大）：**Meta-LoRA Bank**（选锚点用户，预训练元人格特质）→ **Adaptive LoRA Fusion**（检索并动态合并最相关的锚点 meta-LoRA 合成用户专属 LoRA，**消除 per-user 存储**）→ **LoRA Stacking**（在合并结果上叠一个超低秩 LoRA 做少样本个性化）。

---

## 3. 横向对照

| 方案 | per-user 训练 | per-user 存储 | 冷启动 | 未见用户 | 灾难性遗忘 | 回滚粒度 |
|---|---|---|---|---|---|---|
| **OPPU / Second Me L2** | 需要（小时级重跑管线） | 一个完整 LoRA | **差**（需足够历史） | 不支持 | **有**（论文列为未解） | adapter 版本 |
| PER-PCS | 只训门 | 一小部分碎片 | 中 | 部分 | 中 | 碎片 |
| **P2P / T2L / DnD** | **零** | **零**（只存 profile） | **好** | **支持** | 不适用（不改基座） | **重新生成** |
| Generative Adapter | **零** | 零 | 好 | 支持 | 不适用 | 重新生成 |
| Cartridges | 需要（离线，跨查询摊销） | 一个小 KV cache | 中 | 不支持 | 不适用 | 版本，**且可组合** |
| **知识编辑** | 需要 | 累积编辑 | — | — | **严重：10K 次内退化回原点** | **几乎不可回滚** |
| TAP-PER | 需要（轻） | prefix（比 OPPU 少 130×） | 中 | 部分 | 低 | prefix |
| MTA | 只训超低秩栈顶 | 无（动态合并） | 好 | 部分 | 低 | 栈顶 LoRA |

---

## 4. 与 VZ 的关系

### 4.1 我们已有的对应物

| VZ | 对应族 |
|---|---|
| `lifeform-domain-figure` 的 persona LoRA + `FigureArtifactBundle`，走 `ModificationGate.OFFLINE` | **F1**（= Second Me L2，但带 gate） |
| `docs/specs/persona-lora-concurrency.md` | **F6** |
| `docs/specs/personal-conditioning.md`、`character-soul-bootstrap.md` | F1/F7 之间 |
| CMS + reflection 沉淀 | 与 **F4** 同一问题域，但我们是在线多频段，他们是离线一次性 |

### 4.2 结构性差异

F1–F7 全族做的是**"把用户历史压成一个静态个性化模型"**——一次性或周期性重跑的**离线拟合**。我们做的是**在线、分时间尺度、由 PE 驱动的持续适应**，persona LoRA 只是 rare-heavy 那一层的 artifact。

**所以 Second Me 不是我们的竞品，是我们一个子模块的对照实现。** 内部若有人拿它当整体方案对标，坐标就错了。

---

## 5. 值得借鉴（3 条）

### H.【高】超网络摊销 → 解决我们 persona LoRA 的冷启动与上线成本

我们的 figure / growth-advisor 垂类，**每上一个新角色都要烘一次 LoRA**。F3 三篇独立证明这一步可以摊销成**一次前向**：P2P 零 per-user 训练且泛化到未见用户；DnD 比全量微调低 12,000× 开销且在未见任务上比训练所得 LoRA 高 30%；T2L 能把数百个 LoRA 压进一个超网络。

如果在我们的 persona 空间成立，这会直接改写 `figure-vertical` 的上线成本模型——**从"每个角色一次烘焙"变成"每个角色一条 profile"**。

**但必须先验证一个开放问题**：F3 三篇**全部在任务适配上验证**（GSM8K、ARC、BoolQ、常识推理），**没有一篇在人格/风格/关系上验证**。这与 [`02_VZ_DELTA.md`](02_VZ_DELTA.md) §2.A 稀疏槽写入面对的是**同一个开放问题**——分布式属性（人格、风格、关系）能否被定位/摊销，业界没有答案。

**两条实验线因此应该合并设计**：稀疏槽写入（能不能定位）与超网络生成（能不能摊销），共用同一套"人格/风格臂"评测。

### I.【高】Cartridges 的两条经验 → 我们的角色语料压缩

**I-1（负面经验，最有价值）**：**朴素 next-token prediction 打不过 ICL，必须用合成对话 + context-distillation 目标。**

这是一个强烈的告警：如果我们把角色语料直接拿去做 LM loss 微调，**大概率吃不到效果**。而且这不是孤证——**Second Me 独立地也走了合成路线**（多 agent 模拟 + 五级过滤 + CoT 风格化）。**两条互不引用的线同时指向"必须合成，不能直喂"。**

我们的 `lifeform-synthetic-data`（离线统一体验语料，确定性 world/FSM truth + expression-only LLM 渲染）在结构上已经是这个答案，但它服务的是评测与训练语料生成，**没有明确用 context-distillation 目标去压缩角色语料**。这是一个可以直接接上的缺口。

**I-2**：**cartridge 可在推理时拼接而无需联合优化**（后续工作解释为 key 充当可共享路由器）。这对我们的多角色 / 多领域组合是直接有用的性质——**组合性是设计出来的，不是免费的**。

### J.【中】Self-Gating → "读出也要门"

`Context Distillation as Latent Memory Management` 在检索 + 路由之后还加了一道 **Self-Gating**，决定潜在记忆**是否应该被激活**，理由是"关掉不必要的潜在记忆能提升鲁棒性"。

把它和第一轮的两条放一起看：

- **Janus**：记忆**写入**要门（更新是部署决策）
- **RIZZ**：记忆写入要 **verifier** 门
- **Self-Gating**：记忆**读出/激活**也要门

**2026 年三条互不引用的线同时得出"记忆需要门"**，且覆盖了写和读两侧。我们目前 `PeWriteGate` 管写入准入，检索侧（`vz-memory/retrieval.py`）**没有对应的激活门**——检索到就注入。这是一个和 [`02_VZ_DELTA.md`](02_VZ_DELTA.md) §2.B 同源、但在另一侧的缺口。

---

## 6. 明确划界（1 条，但很硬）

### 知识编辑（ROME / MEMIT / WISE / AlphaEdit）不要碰

三个独立证据：

1. **WikiBigEdit**：RAG 大幅胜出；**同等推理成本下，简单的持续微调（LoRA + merging）在规模上持续优于高级编辑技术**。
2. **WISE**：前 10K 次更新内稳步退化，**最终收敛回更新前水平**。
3. **AlphaEdit**：1 → 1000 次编辑，0.96 → 0.72。

它既**没有容量记账**（编辑数不构成预算），也**没有回滚粒度**（累积编辑无法逐条撤销），与 **R10 / R15 完全不兼容**。

> 如果内部有人提"用 knowledge editing 做用户事实的持久更新"，上面三条可以直接结束讨论。它在 2026 年已经是一条被数据关掉的路。

---

## 7. 未决问题：per-user 参数化能不能承载"关系"

**F1–F7 的全部评测都是 LaMP 系**（新闻分类、评分预测、标题生成、推文改写、学术引用）——**没有任何一个评关系连续性、信任修复、陪伴质量**。Second Me 的四个任务（Memory Self / Third-party / Context Enhance / Context Critic）最接近，但仍是"信息服务"而非"关系"。

这意味着两件事：

- **好消息**：我们的 F2（交互质量）/ F3（关系连续性）指标在这个领域**没有可比基线，是我们独有的地盘**。
- **坏消息**：**也没有外部锚**。这和 [`02_VZ_DELTA.md`](02_VZ_DELTA.md) §1 指出的 R-PE 风险敞口是同一个问题的两面——我们在一条没人走过的路上，既拿不到别人的验证，也拿不到别人的证伪。

**这直接抬高了 gain metric（§2.D）的优先级**：越是没有外部基线的地方，越需要 stateful 与 stateless 的自对照。

---

## 8. 一页纸补充摘要（接 [`02_VZ_DELTA.md`](02_VZ_DELTA.md) §5）

| # | 事项 | 类型 | 成本 | 落点 |
|---|---|---|---|---|
| H | 超网络生成 persona LoRA（P2P/DnD 路线），**与 A 合并共用人格/风格臂** | 新实验 | 中 | `lifeform-domain-figure` 烘焙流程 |
| I | 角色语料改用 **context-distillation 目标 + 合成对话**，勿直喂 LM loss；验证 cartridge 组合性 | 改方法 | 中 | `lifeform-synthetic-data` × figure 烘焙 |
| J | 检索侧 **Self-Gating**（激活门），补齐"读出也要门" | 补门 | 低 | `vz-memory/retrieval.py` |
| K | 把 F5 知识编辑写入 known-debt 的**排除清单**（附三条证据） | 存档 | **低** | `docs/specs/` |

H 与 [`02_VZ_DELTA.md`](02_VZ_DELTA.md) 的 A 是**同一条实验线的两个分支**（能不能定位 / 能不能摊销），应共用评测臂。J 与 B 是**同一个"记忆需要门"结论的两侧**（写入 / 读出），应一起设计。
