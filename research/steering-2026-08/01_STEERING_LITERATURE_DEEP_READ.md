# 01 · Steering 文献深读（2026-08）

逐篇提炼**核心机制 + 可引用量化结论**，页码引自本地 PDF（`papers/`）。分三簇：A 可靠性负结果、B 方向来源与定位、C 学习式与策略、D 综述与条件化。

---

## A 簇 · 可靠性：为什么"可读却不可扳"是已知现象

### A1 · Understanding (Un)Reliability of Steering Vectors（Braun et al., ICLR 2025 WS · 2505.22637）

**这是解释我们 S2 null 的核心论文。** 在 Llama2-7B-chat、36 个行为多选数据集、layer 13、CAA（diff-of-means）方法上系统测 steering 可靠性。

核心量化结论：

1. **反向 steer 是常态，不是例外**：所有 7 种 prompt 构造平均都产生净正效应，但**逐样本高方差**。全集平均有 **29%–43% 的样本被"反向 steer"**（施加 steering 反而降低目标行为 logit 差）；单数据集范围 **3%–50%**。最不可 steer 的 6 个数据集里，某些 prompt 类型的平均 logit 差**为负**（低至 −1.19），反向样本比例接近 50%–60%（Fig 1、Fig 5）。

2. **可 steer 性由"方向一致性"预测**（关键机制）：把每个样本的激活差 `a(x,y+) − a(x,y−)` 与最终 steering 向量做余弦相似度，其**均值 Spearman ρ=0.76（p=1.03e-7）**预测 steering 效应大小，**ρ=−0.78（p=2.18e-8）**预测反向样本比例（Fig 2、Fig 6）。**方向一致性高 ⇒ 行为是"一条连贯的线性方向" ⇒ 可 steer；方向发散/正交 ⇒ 不可 steer。**

3. **可 steer 性由"可分性"预测**：沿 diff-of-means 线投影正/负激活，用 discriminability index `d′ = |μ+−μ−| / sqrt(½(σ+²+σ−²))`；**d′ 与 steering 成功正相关（ρ=0.71）**。可 steer 的数据集正负分布分离清晰；不可 steer 的重叠、方差大（Fig 3、Fig 7–10）。

4. **同一行为、不同 prompt 得到的向量方向大不同**：pairwise 余弦 **0.07–0.86**——说明"随便挑个 prompt 抽向量"极不稳定。

> **对我们的直接含义**：probe 高准确率（0.9833）**不蕴含**方向一致性高或 d′ 高。probe 可以在一个弯曲/发散的流形上做出高判别，而该流形**不是**一条可加的连贯 steering 方向。Braun 给了一套**在施加干预之前**就能预测 steerability 的只读指标（余弦一致性 + d′）——我们 S2 从未跑过它。这把 S2 的 null 从"神秘失败"变成"可预测的失败"。

### A2 · Steering off Course（ACL 2025 · 2025.acl-long.974）

跨 **36 个模型**系统测经典 steering 方法，结论：静态 steering **跨模型不泛化**，很多在原模型有效的向量换个模型/规模就失效。→ 对我们意味着：即使在 0.5B 上修好，也不能默认迁移到 production 基底；每个基底都要重测可 steer 性。

### A3 · FaithSteer-BENCH（2026 · 2603.18329）

部署级压力测试框架。核心警醒：许多 steering "可控性"其实是 **prompt-conditional alignment（伪可控）**，脱离特定 prompt 分布即失效；且普遍存在**能力税**（通用能力下降）。→ 与我们的 gate-wise 证据文化一致：steering 通过 S2 因果门后，仍需产品域压力测 + 能力税度量，不能一次通过就当解决。

---

## B 簇 · 方向来源与定位：该往哪个方向扳、该在哪里扳

### B1 · Contrastive Activation Addition / CAA（Rimsky & Panickssery et al., ACL 2024 · 2312.06681）

奠基性的 **diff-of-means** 方法：steering 向量 = 多对正/负样本激活差的**均值**，`s_l = mean[a(x,y+) − a(x,y−)]`。相较早期 ActAdd（单对 prompt 差），CAA 用几十到上百对求均值降方差，得到更鲁棒的向量。综述记载：CAA 施加后 MMLU 仅降 **2–4%**（高 specificity）；样本效率上 **~80–100 对/属性**后方差趋于收敛（Tan et al.）。

> **对我们的直接含义**：CAA 的方向来源是**生成式对比的均值差**，理论上（Marks & Tegmark 几何）是"因果相关方向"。我们 S2 用的是**判别式 probe 权重**——综述 §4.3 与 Braun 都指出 probe 权重会过拟合到**非因果**特征。方向来源错配，是 S2 null 的第二个结构性嫌疑。

### B2 · Activation Steering via Generative Causal Mediation / GCM（Sankaranarayanan et al., MIT/Stanford · 2602.16080）

回答"**该在哪里 steer**"。问题设定：如何定位**弥散在长响应多个 token 上**的概念（如"用诗体 vs 散文体说话"）。做法：构造对比的长响应数据集 → 量化每个组件（如注意力头）对概念的**中介强度** → 选最强中介做 steering。在 refusal / sycophancy / style transfer 三个行为、三个模型上，**GCM 明显优于基于相关性的 probe 基线**，尤其在"稀疏注意力头"预算下。论文明确：**单 token / 固定位置的代理不足以捕捉弥散概念**（footnote 1：单个头无法完全定位弥散概念）。

> **对我们的直接含义**：子目标是**跨整条轨迹弥散的多 token 概念**，而我们固定在 layer20、单步/单位置读写。GCM 说这类 proxy 天然不足，且相关性 probe 选位会输给因果中介选位。→ S1/S2' 的定位应改为在轨迹上做因果中介，而非钉死一层。

---

## C 簇 · 学习式干预与策略：最强路线

### C1 · ReFT / LoReFT（Wu et al., Stanford, NeurIPS 2024 · 2404.03592）

在**冻结基底**上**学习**对隐表征的干预（drop-in 替换 PEFT）。LoReFT = 在低秩投影矩阵张成的线性子空间内干预（基于 DAS）。量化：比 LoRA **参数效率高 15×–65×**，在常识推理/指令跟随/NLU 上达到 SOTA；干预可推理时施加/移除，不改冻结基底权重（综述 Table 9 给 Reliability/Specificity/Reversibility 均为 +）。库：pyreft。

> **对我们的直接含义**：这正是我们 **B screen** 的血统——layer20 full-width 896 经可学习 GRU 入 16 维 `z_t`，actuator 为 rank-8 `A·diag(tanh(Cz_t))·Bᵀ·e_t`（[stage3.md](../../.cursor/plans/stage3.md) L99-104）。综述给出的方法排序（见 D1）里**优化式/学习式 = 最强档**，B screen 押的是头号种子。

### C2 · RePS: Reference-free Preference Steering（NeurIPS 2025）

用**偏好优化目标**训练表征干预（无需参考模型），且**随模型规模变强**。→ 给 S3 提供了成熟的训练目标形态：当 S2 因果门通过后，S3 的"学何时/往哪/多狠"可以用偏好式 / advantage 式目标，而非从零发明 reward。

---

## D 簇 · 综述与条件化：坐标系与"何时出手"

### D1 · From Weights to Activations（Ostermann & Gurgurov et al., ACL 2026 · long.1377）

把 steering 正式定位为与 FFT/PEFT/prompting 并列的**第四类模型适应范式**（改激活轨迹，而非权重或输入），并给 8 维功能坐标系（Table 1）：Reliability / Generalization / Specificity / Compute·Data Efficiency / Composability / Usability / Reversibility。

关键排序与事实：
- **三类 steering 的能力画像**：Difference-based（Reliability +, Specificity +, Generalization 0）、**Optimization-based（Reliability +, Generalization +, Specificity + —— 全面最强）**、Dictionary/SAE（Reliability 0，且 AxBench (Wu et al. 2025) 显示 SAE **不敌** DiffMean 与优化式）。综述明确的实践排序：**优化式 ≥ DiffMean > SAE/静态向量**。
- steering 的独特优势区：高 specificity / 高效率 / 高可逆；弱在全局保证与可用性。
- 领域热度：CL+NeurIPS 摘要里 steering 从 2020 年 3 篇涨到 2025 年 **188 篇**（Table 2、Fig 4）。
- **五个开放挑战**（对产品化极关键）：① 纠缠表征导致的**副作用**；② 与 FFT/RLHF 对齐机制的**未知交互**；③ **带保证的模块化控制**（steering 提供局部控制，需外部组件做全局约束/运行时监控）；④ 超越向量加法的**可组合性**；⑤ interpretability-by-design。

> **对我们的直接含义**：综述把 steering 明确框成"**局部、可逆、需嵌入更大验证系统**的模块"，与我们"读出 owner（冻结传感器）+ 有界执行器 + 外层 gate/回滚"的架构完全一致。挑战 ③ 正是我们契约式 gate 文化的价值所在。

### D2 · Conditional Activation Steering / CAST（Lee et al., IBM, ICLR 2025 · 2409.05907）

给 steering 加"**条件**"维度：`h′ ← h + f(sim(h, proj_c·h)) · α · v`，其中 `c` 是**条件向量**（当作开关，度量当前激活是否落在某类上下文），`v` 是**行为向量**。可实现"若输入属于 X 则施加干预"的规则，并支持条件向量的逻辑组合；无需权重优化；IBM 开源 [activation-steering](https://github.com/IBM/activation-steering)。

> **对我们的直接含义**：CAST 是我们 **S3"学何时出手"的规则版同构**——condition vector ≈ 我们的门控 readout，behavior vector ≈ 沿子目标轴的动作。我们的 Internal RL 版是它的**学习式推广**（把手写规则换成 PE 门控 + 段信用 + advantage/偏好优化学出的门控策略）。

---

## 横切总结：文献画出的"steering 成功前提"

把九篇拼起来，steering 有效需同时满足（缺一即落入 Braun 的负结果区）：

1. **目标行为在激活空间是一条连贯线性方向**（A1：方向一致性高、d′ 高）——**施加前可只读预检**。
2. **方向来源用 diff-of-means 或学习式，而非判别式 probe 权重**（B1、D1）。
3. **定位用因果中介，而非相关性 probe 选层；弥散概念不能用单 token 代理**（B2）。
4. **在有余量/模糊处测建设性效应，而非饱和位置**（A1 的可分性 + Marks&Tegmark 在模糊语句上干预）。
5. **优先学习式（ReFT/RePS）> 均值差 > 静态/SAE**（D1 排序）。
6. **条件化门控（CAST）+ 外层验证/回滚**才谈得上部署（D2 + A3 + D1 挑战③）。

下一份文档把这六条映射成对 S2 null 的归因与可直接执行的 S2'/S3 配方。
