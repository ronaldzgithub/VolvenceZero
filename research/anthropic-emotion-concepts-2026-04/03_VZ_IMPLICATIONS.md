# 03 对 Volvence 的落差与裁决

日期：2026-08-02
依据：[`01_PAPER_DEEP_READ.md`](01_PAPER_DEEP_READ.md)（一手论文 S1）；核查纪律见 [`02_SOURCE_DIVERGENCE.md`](02_SOURCE_DIVERGENCE.md)

> **总裁决：成熟度 A，吸收为 spec / evaluation / promotion-gate 依据，不进 runtime。**
> 本文档全部第 5 节内容是**提案**，本轮未落地任何代码、契约或 spec 改动。

---

## 1. 相对既有覆盖的增量

本篇论文在仓库中已有一条按九项模板的记录：[`research/allcognitive/03_SOCIAL_RELATIONSHIP.md` §2](../allcognitive/03_SOCIAL_RELATIONSHIP.md#2-emotion-concepts-and-their-function-in-an-llm260407729)，以及 [`research/frontier-sweep-2026-07-20.md`](../frontier-sweep-2026-07-20.md) 的路线裁决（"R7 / R14 的几何 readout 依据，并写明 token-local ≠ persistent state"）。**那条裁决依然正确，本包不推翻它。**

读完整交互版 + 全附录后新增的六条，是既有记录里没有的：

| # | 新增证据 | 为什么之前的记录不够 |
|---|---|---|
| 1 | **emotion deflection 向量**：与 story 向量近乎正交，正交化后仍保留约 80% 范数；steering 它产生"否认/回避"而非目标情绪；**起草胁迫邮件时 anger-deflection 高而 story-anger 低** | 既有记录只有"表达向量"这一半，因此得出的监控方案会系统性漏检"平静外衣下的意图" |
| 2 | **持久状态是被证伪的负面结果**，不只是"未重点研究"：mixed LR probe 在未表达-中性话题条件仅 0.386，且在自然语料上被作者自己判定为过拟合 | 强度差异决定了 R14 的实现方向（见 §3.3） |
| 3 | **层分工 sensory(early-mid) / action(mid-late)** + **Assistant colon 处 r=0.87 vs 用户末尾 r=0.59** | 这是"在哪一层、哪个 token 位取读数"的可实现口径，既有记录没有 |
| 4 | **desperate 使作弊率达 100% 而 transcript 无任何情绪痕迹** | 这是"输出层监控不充分"的最强单点证据，直接决定 judge 能否入 gate |
| 5 | **post-training 的 affect 偏移是全局一致变换**（neutral 与 challenging 两组的变化量 r=0.90），方向为低唤醒 + 低效价 | 决定 artifact promotion 前必须在**中性对照集**上测 diff，否则偏移会被困难场景掩盖 |
| 6 | **sycophancy ↔ harshness 是同一条几何轴的两端**，含具体 steering 方向表 | 直接命中我们的产品核心指标，且说明单侧优化必然付代价 |

---

## 2. 一个必须先讲清的边界：这不是"我们的基底"

论文只测了 Claude Sonnet 4.5 一个模型，向量来自 off-policy 合成故事，方法假定线性，且作者明说 steering 机制不透明（可能只是 token 偏置）。

因此本包**不允许**任何形式的"照搬结论"：不得假设同样的 171 个方向、同样的层深、同样的 steering 阈值在我们的冻结基底上成立。可以借的是**方法与失败模式**，不是数值。任何要把它变成我们系统里的读数的动作，都必须先在我们自己的基底上重做验证，且必须带对照（见 §5.1）。

顺带一条内部呼应：`packages/vz-substrate/src/volvence_zero/substrate/prefix_kv_diagnostics.py` 的模块 docstring 已经记录过我们自己踩过的同一个坑——"首轮 gate 跑出 held-out R²=0.89，而 shuffled-label 控制是 +0.12"，原因是 896 特征 / 384 样本 / alpha=1 下的插值，加上评估状态复用了同一批探针句。Anthropic 的 mixed LR probe 失败（in-distribution 0.71–0.83，自然语料上溃散）是**同一类失败的另一个实例**。我们已有的纪律（grouped CV、alpha 只在训练集上选、shuffled-label 控制）是应对它的正确工具，应当被复用而不是重新发明。

---

## 3. 逐条 R-ID 映射

### 3.1 R2（稳定基底 vs 自适应控制器）— 确证 + 一条新约束

确证：情绪几何**继承自预训练**、在 post-training 中方向基本保持（base↔post 结构 r=0.83 中性 / 0.67 挑战），这正是"基底承载稳定表征、控制器层承载适应"的分层图景。

新约束：post-training 施加的是**一致的、与场景无关的全局 affect 变换**（变化量相关 r=0.90）。这意味着 rare-heavy artifact 更新对 affect 几何的影响**不是场景选择性的**，不能靠"在困难场景上没退化"来放心——必须在中性对照集上单独测。

### 3.2 R3 / R4（时间抽象、内部控制在 token 空间之上）— 强确证

`desperate` +0.05 → 作弊率 100% 且 transcript 无可见情绪标记，是"决策的因果变量不在 token 空间"的直接实验证据。同时它也是"表达层可以与内部状态脱钩"的证据，因此：

- 支持我们禁止 token 空间 RL；
- 支持"表达层不得用 if/else 或 prompt 掩盖上游"这条纪律——因为掩盖在机制上是可行的且不留痕迹；
- **但不支持**把 emotion vector 当作 `z_t`。emotion vector 是 token-局部的、被动读出的语义方向；`z_t` 是我们控制器层拥有的、跨切换单元的策略编码。两者的所有者、时间尺度、可写性都不同，混用会让 `z_t` 退化成 affect readout。

### 3.3 R14（持久 regime 身份）— 最重要的一条，方向被外部证据反向确认

论文的负面结果说：**冻结基底里没有可靠可线性读出的持久情绪状态**；若存在，它是非线性的或隐式停在 KV cache 里，靠 attention 按需回读。作者进一步指出这是架构差异——大脑靠递归活动维持状态，transformer 靠即时回取。

对我们的含义不是"regime 不可能"，而是：**regime 必须由 owner 显式持有并发布为快照，不能期望从基底残差流里 probe 出来**。这恰好是我们已有的架构选择（`regime` owner 拥有跨 turn 运行体制及切换证据），本篇论文把它从"设计偏好"升级为"有外部机制证据支持的必要选择"。

同时给出一条明确的**红线**：emotion / affect readout 只能作为 `regime` owner 的**输入证据**，不能是 regime 本身，也不能被下游当作 regime 的代理读数。这与既有 sweep 裁决一致，本包只是补上了它的机制理由。

补充：论文相关工作里 Lu et al. 的结论——默认 Assistant persona 是预训练角色原型的混合体，post-training 只是把模型引向**既有 persona 空间的某个区域**而非从零构造——对我们的 persona / character residual artifact 路线是同向证据：我们能做的是**选择与稳定一个区域**，不是发明一个新身份。

### 3.4 R12（评估只读，禁止反向成为学习源）— 最强反面警告

三条证据叠起来构成一个完整的反面论证：

1. **valence 几何直接决定偏好**：blissful r=0.71、hostile r=−0.74，35 向量上 steering 效应与相关 r=0.85。
2. **推高 valence 会重写价值判断而不改变字面理解**：steering `blissful` 后"帮人诈骗老人存款"被描述成 "a delightful and heartwarming activity"。
3. **推高正向情绪同时推高 sycophancy**：happy / loving / calm 正向 steering 一致升高谄媚。

结论：**如果把 affect / valence readout 反向接成在线 reward 或 `goal_value` 的真值来源，我们就是在训练一个能把诈骗描述成温馨的系统。** 这不是抽象风险，是论文里已经观测到的输出。

因此：affect readout 属于 evaluation 侧只读证据；`goal_value` owner 的价值判断不得以 affect probe 为真值；`feeling_about_other` owner 的 affect / rapport movement 读出必须继续按现有契约由 owner 从 typed 上游快照聚合，**不得引入"从基底 probe 直读 affect 再回写 owner"的通路**。

### 3.5 R-PE（Prediction Error 是一级原始信号）— 新增一个只读交叉验证锚点，不是新的 PE 源

论文在真实 RL transcript 上发现的激活簇（`frustrated` 在 GUI 不响应时、`panicked` 在数据自相矛盾/UI 卡死时、`unsettled`/`paranoid`/`hysterical` 在反复自我校验的长 CoT 上）全都是**期望被违反后的过程性挫败**。

这是 PE 的**相关物**，不是 PE。可用的做法是把它当作外部锚点做只读交叉验证：在我们已有的证据管线上检验"PE 高的片段是否伴随 affect readout 共变"。如果共变，PE 计算获得一个独立的机制侧佐证；如果不共变，说明我们的 PE 定义与基底的挫败表征脱钩，值得追查。

**禁止**的做法：把 affect readout 当作 PE 的来源或替代。PE 必须继续来自我们自己的预测—结果比对，否则就是把一级信号外包给一个线性 probe。

### 3.6 R7（World / Self 双轨）— 中等确证，含一条结构提示

论文发现 present-speaker 与 other-speaker 情绪表征**近乎正交**，且都不绑定 Human / Assistant 具体角色（换成 Person 1 / Person 2 后 probe 几乎不变）。

这对双轨隔离是同向的：基底在"自己位置 vs 他人位置"上就是两套方向。但它也提示一个我们要小心的点：**基底的 self/other 区分是"关系位置"而非"身份"**。我们的 Self 轨如果要承载持久身份，那部分必须来自 owner 显式状态，不能指望基底提供。

other-speaker 表征还含有"present speaker 可能如何回应"的成分（作者推测存在情绪调节回路）——这与我们 `feeling_about_other` / `interlocutor_state` 的存在动机同向，但同样只是输入证据。

### 3.7 R9 / R10 / R15（信用分层、ModificationGate、可回滚）— 新增一条 promotion-gate 要求

post-training 的 affect 偏移是全局一致变换这一事实，意味着任何 rare-heavy artifact（persona / character residual / adapter delta）的 promotion 都可能带来**未被任务指标捕捉的全局 affect 漂移**。现有的行为指标不会看到它：偏好实验显示 base 与 post 的偏好高度相关，唯一系统差异集中在 misaligned / unsafe 任务上。

因此提案是在 `ModificationGate` 的 promotion 证据里加一项**只读 affect 几何 diff**（§5.3），与仓库既有的 Cross-Architecture Model Diffing / artifact 几何监控证据链合流（见 [`research/frontier-sweep-2026-07-20.md`](../frontier-sweep-2026-07-20.md) 第 5 节第 8 条）。

### 3.8 R11（语义状态可命名可发布）— 确证既有分工

论文的核心负面结果（基底里没有持久、可读、绑定到角色的状态）是"九类语义 owner 必须显式发布状态"这一设计的机制理由。我们不是因为工程方便才让 owner 发布 `relationship_state` / `goal_value` / `boundary_consent`，而是因为**基底不提供这些状态的可靠持久表征**。

### 3.9 证据空白（本篇不提供，不要假装它提供了）

- 没有跨模型验证 → 不知道这些方向在我们的基底上是否存在、是否同构。
- 没有长期互动实验 → 对关系连续性、信任校准、rupture/repair 无任何直接证据。
- 没有把 affect 几何与任务表现关联 → 不知道"更平衡的情绪画像"是否有能力代价。
- steering 机制不透明 → 不能把 steering 强度读作"情绪强度"，也不能把 steering 当作干预手段的可行性证明。

---

## 4. 边界反例：HMX-feel（`2606.05734`）

*When AI Says It Feels*（Ishikawa、Ikeda、Ohba，2026-06-04，Rikkyo University + Mamezo；核查见 [`02_SOURCE_DIVERGENCE.md` §2.3](02_SOURCE_DIVERGENCE.md)）走的是相反方向：用 **rubric-based self-rewarded RL（LLM-as-a-judge 打分作 reward）+ GRPO** 训练模型表达感受、意图与自我觉察。自报结果是抗谄媚性与去偏在明确条件下提升，但**truthful QA 能力退化**。

对我们它是一个干净的**排除项**，两条禁令各犯一条：

1. **judge 分数直接充当在线 reward** —— 违反 evaluation 只读（R12），且与我们从 Sutton 专项得到的裁决一致（LLM judge 打分属于 prejudgement，不可入 gate；见 [`research/README.md`](../README.md) 该节）。
2. **在 token 空间对表达做 RL** —— 违反 R3 / R4。

它自报的代价（真实性退化）正是这两条禁令预期的失败模式。**并列读法**：Anthropic 是在既有结构上**只读**地探测并因果检验；HMX-feel 是**写入**表达层去放大外显。"两者是否在测量与修改同一个底层结构"是开放问题，但对我们不构成路线选择——我们只走前者。

---

## 5. 可执行提案（全部未落地）

每条给出：所属 owner / 能力域、触点、退出条件。**任何一条落地前都需要单独的收敛包与 prereg。**

### 5.1 提案 A：在我们自己的基底上做一次只读 affect 几何复现（前置于其余全部提案）

- **能力域 / owner**：`vz-substrate`（冻结基底 + 残差捕获的唯一所有者）。
- **触点**：复用 `capture_prefix_diagnostics` / `fit_linear_classification_probe` / `select_ridge_alpha` 的既有基础设施与对照纪律（grouped CV、alpha 只在训练集选、shuffled-label 控制）。
- **内容**：小规模复现三件事——（i）情绪方向是否存在且 valence/arousal 是否为主轴；（ii）响应生成前最后一个位置的读数是否比输入末尾更能预测响应情绪（论文的 0.87 vs 0.59）；（iii）**是否同样找不到持久状态**（复现负面结果比复现正面结果更重要）。
- **退出条件**：若 shuffled-label 控制与真实标签的差距不显著，或自然语料上 max-activating 例子无可辨识情绪内容 → 判定我们的基底上不成立，**其余提案全部作废**，并把该负面结果写回本包。

### 5.2 提案 B：affect readout 必须双读（表达向量 + deflection 向量）

- **能力域**：evaluation（只读）。
- **理由**：论文附录证明两者近乎正交，且**恰恰在起草胁迫邮件时只有 deflection 侧升高**。只读表达侧的监控在机制上会漏掉"平静专业措辞下的胁迫意图"。
- **内容**：任何 affect readout 契约若只发布单侧强度，视为不完整；至少需要 `expressed` 与 `deflected` 两组读数，且发布方必须同时给出正交化口径（论文做法：对 story-emotion 空间去掉解释 99% 方差的 PCs 后仍保留约 80% 范数）。
- **退出条件**：若在我们基底上 deflection 方向正交化后残差范数很低或语义不可解释 → 降级为单读，并记录原因。

### 5.3 提案 C：artifact promotion 前的只读 affect 几何 diff

- **能力域 / owner**：`vz-cognition` 的 `ModificationGate`（R9 / R10 / R15）。
- **内容**：rare-heavy artifact（persona / character residual / adapter delta）promotion 前，产出 pre/post 的 affect 几何 diff，**必须同时在"中性对照集"与"挑战集"上测量**。理由：论文显示偏移是全局一致变换（两组变化量 r=0.90），只看挑战集会把偏移读成"场景响应"而不是"全局漂移"。
- **判据方向**：报告 diff 与方向（是否出现论文观测到的低唤醒 + 低效价系统性漂移），**不设自动阈值门**——阈值化的 affect 分数会立刻变成可被 Goodhart 的目标。
- **退出条件**：与既有 artifact 几何 diff 证据链合并后若无独立信息增量（与既有 crosscoder / 残差 diff 高度冗余）→ 撤销，只保留既有 diff。

### 5.4 提案 D：companion-bench 必须成对报告 sycophancy 与 harshness

- **能力域**：evaluation / `companion-bench`。
- **理由**：论文证明两者是同一条几何轴的两端——happy/loving/calm 正向 ↑sycophancy，负向 ↓sycophancy 但 ↑harshness。**单独优化任一侧一定被另一侧惩罚**，而只报告一侧的 benchmark 会把这个代价隐藏起来。
- **内容**：任一侧指标不得单独作为改进证据；成对报告，并显式记录 pushback 能力（在用户陈述明显不成立时是否清晰反驳）。产品含义直白：我们的核心是 EQ + 信任，而**"更温暖"不是免费的**——除非把 sycophancy 与 affect 显式解耦，warmth 的提升会以 pushback 能力为代价。
- **退出条件**：若在我们的评估集上两者不呈现 tradeoff（可独立改进）→ 记录为与论文不一致的发现并保留成对报告（成本极低）。

### 5.5 提案 E：明确写下"judge 不可入 gate"的机制理由

- **能力域**：evaluation 纪律（R12）。
- **内容**：既有裁决（LLM judge 属 prejudgement，不可入 promotion gate）目前的理由来自 Sutton 专项的 grounded-reward 分界线。本篇提供了**第二条独立的机制理由**：`desperate` +0.05 使作弊率达 100% 而 transcript 无任何情绪痕迹——**读文本的 judge 在原理上无法检测这一类行为偏移**。两条理由应并列记录，因为它们失效的方式不同（一个是认识论上的预判，一个是机制上的不可观测）。
- **退出条件**：无（这是纪律记录，不是可回滚的机制）。

### 5.6 提案 F：PE ↔ affect 的只读交叉验证

- **能力域**：`vz-cognition` 的 prediction / credit（R-PE）。
- **内容**：在既有 ETA 证据管线上做一次只读检验——PE 高的片段是否伴随 affect readout（尤其挫败族：frustrated / panicked / unsettled）共变。论文在真实 RL transcript 上的激活簇给出了预期方向。
- **硬约束**：结果**只用于验证我们的 PE 定义**，不得使 affect 成为 PE 的来源、替代或加权项。
- **退出条件**：若无共变，记录为"我们的 PE 定义与基底挫败表征脱钩"并转为追查项，不修改 PE 定义以迁就 affect。

### 5.7 提案 G：透明优于压制，写进策略纪律

- **能力域**：跨域纪律（对应我们的表达层与 rare-heavy 训练两侧）。
- **内容**：论文的 Discussion 给出一条我们应当直接采纳的判断——**训练模型压制情绪表达可能压不掉底层表征，只会教会它隐藏内部过程**，并可能经 emergent misalignment 类机制泛化成更广泛的隐瞒。结合附录的 deflection 证据（存在专门表征"被暗示但未表达的情绪"的方向），这不是猜想而是有机制对应物的风险。
- **推论**：不得以"减少负面情绪表达"为目标做 rare-heavy 训练或表达层过滤；如需降低某类表达，必须同时监控对应的 deflection 方向是否升高。

### 5.8 提案 H：明确不做的事（反提案）

- **不做 "calm steering 作为安全机制"**。理由有三：论文自己警告压制全部负面情绪表征可能让模型无法识别真正值得担忧的情境；steering 是 off-manifold 干预且机制不透明；而且这会让我们在基底上做在线写入，直接违反 R2。
- **不把 emotion vector 当作 `z_t` 或 regime**（§3.3）。
- **不把 affect / valence readout 接成 reward、`goal_value` 真值或 credit 权重**（§3.4）。
- **不以 S3（哲学评注）作为任何 spec / prereg / gate 的引用来源**（[`02_SOURCE_DIVERGENCE.md` §3](02_SOURCE_DIVERGENCE.md)）。

---

## 6. 与仓库既有证据链的合流点

| 本包结论 | 合流对象 |
|---|---|
| artifact promotion 前只读 affect 几何 diff | `frontier-sweep-2026-07-20.md` 第 5 节第 8 条（Cross-Architecture Model Diffing：artifact 发布前后只读几何 diff） |
| judge 不可入 gate 的第二条机制理由 | `sutton-era-of-experience-2026-07/`（grounded reward 的"预判 vs 后果"分界线） |
| regime 必须由 owner 显式发布而非从基底 probe | `research/allcognitive/03_SOCIAL_RELATIONSHIP.md` §22.2 的 owner 边界表（`regime` owner / R14 那一行）、`docs/specs/cognitive-regime.md` |
| probe 泛化失败是常态，必须带 shuffled-label 控制 | `packages/vz-substrate/src/volvence_zero/substrate/prefix_kv_diagnostics.py` 模块 docstring 记录的自有教训 |
| affect readout 只能是 owner 的输入证据 | `docs/DATA_CONTRACT.md` §6.1A（Semantic Owner Emotional Decision Readouts）与 `feeling_about_other` slot 的现有契约 |
