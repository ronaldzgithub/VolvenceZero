# Xaira Therapeutics — 深度分析

- **分组 / 成熟度**：C 生物基础模型（生成式蛋白 / 抗体设计）｜ 成熟度高（Baker 实验室 RFdiffusion 谱系，多篇 Nature/Science；2024 以 ~10 亿美元启动资金成立，员工为本地两篇论文共同作者）
- **一句话主张**：把 Baker 实验室的生成式蛋白 / 抗体设计栈（**RFdiffusion 扩散生成骨架 + ProteinMPNN 逆折叠填序列 + RoseTTAFold2 自洽性过滤 → 湿实验验证**的漏斗）规模化为完整的 AI 驱动药物发现引擎；本地两篇分别奠基"序列设计头"（ProteinMPNN）与"从头抗体骨架生成"（RFdiffusion-antibodies）。
- **主要创作者 + 血统**：Marc Tessier-Lavigne（创始 CEO）、Hetunandan Kamisetty（联创/CTO）、David Baker（联创，Institute for Protein Design）、Nathaniel Bennett / Joseph Watson（RFdiffusion-抗体一作）、Justas Dauparas（ProteinMPNN 一作）。血统 = RoseTTAFold / RFdiffusion / ProteinMPNN 的物理 + 深度学习蛋白设计栈。
- **为何与 VZ 共振 / 对立**：**flat note 把它读成"R2 基底 + adapter"与"R3：扩散去噪是约束引导的迭代预测生成"——后者是必须先拆穿的确证偏误。** RFdiffusion 与 ProteinMPNN 都是**有监督 / 生成式、单轨、离线训练**的设计工具：扩散迭代发生在**原子坐标空间**而非 VZ 的 latent 时间控制器空间，没有 z_t/β_t、没有时间抽象、没有 World/Self 双轨、没有一级 PE 信号。本分析**以反证为重心**，先证伪"扩散迭代 = R3/R4 时间控制"的误读；剩下真正成立的关系只有三条：**ProteinMPNN = 冻结结构表征上的轻量逆折叠头（R2-相邻）**、**"设计 → 廉价过滤 → 湿实验验证"漏斗（R12 不可外包真值）**、以及**"结构-然后-序列"可组合两段流水线（R8 模块边界）**。

## 1. 核心逻辑（论文级 · PDF-grounded）

### Robust protein sequence design using ProteinMPNN（bioRxiv 2022.06.03.494563 → Science 378:49–56, 2022；Dauparas, Baker 等）

- **问题**：蛋白**序列设计**问题——给定一个主链骨架，找到能折叠成该结构的氨基酸序列。物理法（Rosetta）把它当能量优化问题，需大规模侧链 rotamer 搜索，慢且需专家针对每个挑战手工定制（如表面疏水残基限制、核心/边界歧义）。
- **方法/机制**：从一个消息传递神经网络（MPNN）出发，**3 编码层 + 3 解码层、隐藏维 128、仅约 1.7M 参数**（极轻量）。输入为主链几何特征（N、Cα、C、O 与虚拟 Cb 间距离用 16 个 RBF 编码、相对位置编码），**自回归**地逐残基预测氨基酸。关键设计：(i) 用原子间**距离**而非二面角作为归纳偏置（recovery 41.2%→49.0%）；(ii) 编码器**边更新**；(iii) **乱序自回归解码（order-agnostic）**——随机采样解码顺序，从而能固定部分序列上下文、支持多链 / 对称（同源寡聚体把对称等价位 logits 平均）/ 重复蛋白 / 多态正负设计。训练集为 PDB 装配体（截至 2021-08-02，<3.5Å），30% 同一性聚类得 25,361 簇。
- **关键结果（PDF 内具体数字）**：
  - 原生序列恢复率 **52.4% vs Rosetta 32.9%**，且在从核心到表面所有埋藏度上都更优；耗时 **1.2 秒 vs Rosetta 258.8 秒**（100 残基单 CPU）。恢复率随埋藏度强相关（核心 90–95%、表面 35%）。
  - **训练时加主链高斯噪声（std=0.02Å）**：原生骨架 recovery 略降，但对 AlphaFold 预测的骨架 recovery 提升；噪声更大（0.3Å）的模型生成的序列被 AF 单序列预测正确解码的比例**多 2–3 倍**——噪声让模型聚焦整体拓扑而非精细局部细节，提升真实设计的"可设计性"。
  - **救活失败设计**：AF 幻觉（hallucination）生成的序列大多不可溶（中位可溶产量 9 mg/L）；ProteinMPNN 重新设计后 **96 个里 73 个可溶、中位产量 247 mg/L、50 个达目标寡聚态**，许多高度热稳定（95℃ 仍保持二级结构）。晶体结构与设计模型几近一致（单体 2.35Å/130 残基；四面体纳米颗粒 1.2Å；C5/C6 寡聚体可溶率 88% vs Rosetta 40%）。
  - 序列质量 = **平均 log 概率**与恢复率强相关 → 可作为**廉价排序信号**挑选实验候选；采样温度调多样性。
- **局限（PDF 自陈）**：in silico 的原生序列恢复率**对晶体分辨率敏感、且不必然与正确折叠相关**（单残基替换即可阻断折叠却几乎不改恢复率）——"序列设计方法的终极检验是实验表征，正如翻译质量终究要由人来评判"。纯主链几何输入，当时尚未含核酸 / 小分子上下文。

### Atomically accurate de novo design of single-domain antibodies（RFdiffusion for antibodies；bioRxiv 2024.03.14.585103 → Nature 2025；Bennett, Watson, Baker 等，Xaira 员工为共同作者）

- **问题**：目前**无法**理性地从头设计结合指定表位的新抗体，只能靠动物免疫或文库筛选（耗时、可能错过治疗相关表位）。普通（vanilla）RFdiffusion 设计的 binder 几乎只靠规则二级结构（螺旋/折叠）与表位作用，**无法从头设计抗体**（抗体靠高变 CDR 环介导结合，Extended Data Fig. 1）。
- **方法/机制**：
  - **扩散骨架生成**：RFdiffusion 沿用 AlphaFold2/RF2 的**帧表示**（每残基 Cα 坐标 + N-Cα-C 刚体取向）。训练时按噪声调度在 T 步内把结构腐蚀到与随机分布无异（Cα 加 3D 高斯噪声，取向在 SO3 上做布朗运动）；采样一个 PDB 结构 + 随机时间步 t，加 t 步噪声，网络预测去噪结构 pX0，对真实结构 X0 取 **MSE 损失**。推理时从噪声分布采 XT，**迭代去噪**生成新结构。
  - **抗体专项微调**：在抗体复合物结构上微调（底层 RF2 已在整个 PDB ~200k 结构上训练，而抗体仅 ~8,100 个，故迁移很关键）。**框架（framework）序列+结构**经"template 轨"以**成对距离+二面角矩阵**提供，且以全局帧不变方式给出 → 框架内部结构被固定、而抗体对靶点的**刚体 dock 由 RFdiffusion 设计**；表位通过 one-hot **"hotspot"特征**指定（适配为 CDR 环要接触的靶点残基）。
  - **可组合两段流水线**：RFdiffusion 生成骨架 → **ProteinMPNN 设计 CDR 环序列**（框架序列固定，不改）。
  - **微调 RoseTTAFold2 作过滤器**：AlphaFold2 无法稳健预测抗体-抗原结构，故微调 RF2 来重预测设计、区分真/诱饵复合物（自洽性 self-consistency 过滤），用 pAE 衡量置信度。
- **关键结果（PDF 内具体数字）**：
  - 实验确认对 **4 个疾病相关表位**的 binder（流感 HA、RSV site III、Covid RBD、TcdB；另测 IL-7Rα）。最高亲和力：**流感 HA Kd 78nM**、RSV site III 1.4μM、Covid RBD 5.5μM、TcdB 262nM。筛选规模：酵母展示每靶 9000 个设计，或 E. coli + SPR 每靶 95 个。
  - **冷冻电镜 3.0Å** 解出 VHH_flu_01 结合天然糖基化 HA 三聚体：VHH 主链对 RFdiffusion 设计 **RMSD 1.45Å**、**CDR3 RMSD 0.8Å**，approach angle 与预测高度吻合——原子级精度的从头抗体设计。设计的 CDR 显著区别于天然抗体（泛化超出训练集；TcdB 表位在 PDB 中无任何已知抗体）。
  - TcdB / Covid RBD 的结合经"与已知结构表征的从头 binder 竞争"确认结合到**预期表位**，且对高度相关的 TcsL 毒素无结合（特异性）。
  - **反直觉的关键负面结果**：在他们用的 RF2 设置（提供 100% 界面 hotspot）下，**过滤 vs 未过滤设计的成功率没有显著差异**（仅在更严格的 0%/10% hotspot 设置下有些信号）——廉价过滤器在该设置下未能富集实验成功者（数据集小，需更多数据评估）。
- **局限（PDF 自陈）**：**亲和力偏中等、成功率仍相当低**；骨架步可换更新的架构 / flow-matching 提升可设计性与多样性；**未建模糖基**（N296 糖基导致 VHH_flu_01 亚化学计量结合）；**ProteinMPNN 未做修改**——设计更接近人源 CDR 可降低免疫原性、直接优化 developability 是未来方向；RF2 抗体预测仍需改进以提升实验成功率与 in silico benchmark。

## 2. 与 VZ 的关系（三视角）

> **本 lab 重心在 §2.2 反证**：先拆穿"扩散去噪 = VZ R3/R4 时间控制"的误读与"生成式工具印证 R-PE/双轨"的确证偏误。先反证、后确证。三视角结论可以不一致。

### 2.1 确证（先进性背书）

- **R2 冻结/稳定结构表征 + 轻量适配头（中 · 本 lab 最干净的关系，但带边界）**：**ProteinMPNN 是 roster 中"轻量任务头"最纯的样本**——仅约 **1.7M 参数**的逆折叠头，读取（冻结的）主链几何表征，输出序列；它不重训结构预测基底，只在其表征之上做一个廉价、可复用、无需专家定制的头。这与 Group C（ESMFold 折叠头、Chai pair-bias 头）"大基底 + 下游小头"是同一范式的又一非语言样本，可作 [`../../../docs/specs/multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md) 中 R2 跨模态合法性的旁证。**诚实边界**：RFdiffusion 本身是对结构生成网络的**全量专项微调**（权重会动），并非严格的"bit 级冻结基底 + 物理独立 adapter"——与 Profluent ProGen3 同样的祛魅（详见 §2.2 反证 B）；干净的 R2 样本是 **ProteinMPNN 这个头**，不是整条栈。
- **R12 评估覆盖且只读、真值不外包（中-强 · 跨模态独立印证）**：整条栈的可信度最终锚在**湿实验作为不可外包的 ground truth**——亲和力由 SPR 实测、结合表位由竞争实验确认、设计精度由**冷冻电镜 3.0Å（RMSD 1.45Å）**这一只读存在性检验拍板，而非靠模型置信度自证。ProteinMPNN 论文更明确写道"in silico 恢复率不必然与折叠相关，终极检验是实验表征"。这正是 [`../../../docs/specs/evaluation.md`](../../../docs/specs/evaluation.md)、[`../../../docs/specs/evidence_program.md`](../../../docs/specs/evidence_program.md) 的"评估只读、正交、可对照"，与 [`../../../docs/specs/prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md)"PE 信号不外包"原则同构（真值来自现实测量，非另一个模型的判断）。跨领域（蛋白湿实验 vs 关系评估）独立印证"先把真值锚定硬"。
- **R8 模块边界 / 可组合流水线（中）**：RFdiffusion（生成骨架）→ ProteinMPNN（在固定框架上设计 CDR 序列）→ RF2（重预测过滤）是一条**职责清晰、各阶段拥有自己表征**的两/三段流水线：下游只消费上游发布的产物（骨架坐标、序列），不重建上游内部状态。这对 [`../../../docs/specs/contract-runtime.md`](../../../docs/specs/contract-runtime.md) 的"快照即唯一数据通道"是结构性同构的工程旁证。

### 2.2 反证（红队）

**反证 A（叙事陷阱 · 必须先拆）：扩散去噪是"约束引导的迭代预测生成过程"，印证 VZ 的 R3/R4（时间抽象 / latent 控制）。**
- 这是 flat note"R3：扩散去噪是约束引导的迭代预测生成过程"的确证偏误读法。诚实拆解：RFdiffusion 的迭代去噪发生在**原子坐标 + 取向（SO3）空间**，是一个**离线训练的生成采样器**的 T 步反演；它的"迭代"是去噪步，不是 VZ 在 token 空间之上学习到的时间抽象（z_t）与切换单元（β_t）。它没有时间维上的策略层、没有在线控制、没有"控制器代码"。
- **裁决：survives（但需修正 flat note / 99 的读法）**。扩散迭代 **不构成** R3/R4 的证据。边界：**生成式扩散迭代（坐标空间反演）≠ latent 时间控制（token 空间之上的 z_t/β_t）**；引用本 lab 时不得作为 R3/R4 论据（99 表中 Xaira 的 R3/R4 ✓ 应降级/标注，与 Latent Labs 同类术语撞车）。

**反证 B：一个冻结/微调的有监督生成模型 + 1.7M 参数逆折叠头，离线训练即达原子级 SOTA，证明根本不需要 PE 闭环 / 在线学习 / 多时间尺度——VZ 的 R-PE/R1 是过度设计。**
- ProteinMPNN/RFdiffusion 全程无预测误差驱动的在线适应：一次性离线训练 → 推理时纯前向生成 → 外部过滤/湿实验。表面上"无 PE 也能强"。
- **裁决：survives**。反例不适用于 VZ 目标域。蛋白/抗体设计是**静态、单次、真值离线可得**的优化任务；VZ 目标域是**非平稳、长程、真值持续到达且会漂移**的关系养成。边界（写入 spec，与 Latent/Profluent 一致收窄）：**R-PE/R1 的适用前提是"目标非平稳 + 反馈连续 + 无一次性真值"**；对静态单次生成任务，离线训练 + 外部验证是合法且更简的方案。这反而澄清了 R-PE 主张的边界 → [`../../../docs/specs/prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md)。
- **附带祛魅**：所谓"R2 基底 + adapter"在 RFdiffusion 这里是**全量专项微调**（权重会动），不是真冻结基底；VZ 的真冻结-基底 R2 比这条栈更严，应注明区别（与 Profluent ProGen3 反证 A 同源）。

**反证 C：整条栈是纯单轨（只有"设计/World"轨，无 Self 轨）却极成功，说明 R7 双轨隔离非必需。**
- **裁决：survives（域外）**。RFdiffusion/ProteinMPNN 是**工具**而非有持久身份的关系主体；双轨（World/Self）是为有持久身份的养成式 agent 设计的，无状态生成器本就不需要 Self 轨。边界：R7 适用于"持久身份 + 关系主体"，不适用于无状态工具 → [`../../../docs/specs/dual-track-learning.md`](../../../docs/specs/dual-track-learning.md)。本反例不构成压力。

**反证 D（真实风险 · 本 lab 最有价值的反例）：生成幻觉 + 廉价过滤器不可靠——"看上去自信、自洽的生成物大多是错的，且廉价代理过滤器可能根本不富集真成功者"。**
- 两条 PDF 自陈证据叠加：(i) **生成幻觉**：AF 幻觉序列大多不可溶（9 mg/L），RFdiffusion 抗体**成功率仍相当低、亲和力中等**——生成模型会产出大量"结构上合理却实际失败"的候选；(ii) **廉价过滤器失灵**：RFdiffusion 论文明确报告**在 100% hotspot 设置下 RF2 过滤 vs 未过滤的实验成功率无显著差异**，且 ProteinMPNN 论文指出 **in silico 自洽性/恢复率不必然预测真实折叠**（单残基替换即可阻断折叠却不改恢复率）。
- 这直接命中 VZ 盲点：若 VZ 用"软验证器"（world/self 预测模型）为控制器层提供 PE-based 奖励或评估读出，一旦该验证器与昂贵真值的**校准/富集能力不足**，就会**自信地放过错误候选**且因是 readout 而难被察觉——这与 Latent Labs 反证 D（验证器分布覆盖）是同一风险的两个侧面。
- **裁决：genuine-risk**。VZ 需在 [`../../../docs/specs/evaluation.md`](../../../docs/specs/evaluation.md) / [`../../../docs/specs/prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md) 显式登记"**廉价代理/软验证器对昂贵真值的富集有效性必须被实测，而非假定**"：任何内部过滤器在进入决策前要先证明它确实提升 top-k 的真值命中率（对照 RFdiffusion 的"过滤 vs 未过滤"对照实验设计），否则只是增加自信而非提升正确率。进风险登记 / ROI 台账。

### 2.3 局部算法借鉴（算法级解耦）

| # | 机制（剥离叙事） | 目标 VZ spec | 落地动作 | 预期收益 | 风险 / 前提 |
|---|---|---|---|---|---|
| 1 | **"过量生成 → 廉价 in-silico 自洽性/置信度过滤 → 昂贵 ground-truth 验证"漏斗 + 对照式过滤器有效性检验**：RFdiffusion 每靶生成数千设计，用 RF2 重预测自洽性（pAE）+ ProteinMPNN 平均 log-prob 廉价排序，仅 top-k（95 个甚至全部）进湿实验真值；并**用"过滤 vs 未过滤"对照实测过滤器到底有没有富集成功者** | [`../../../docs/specs/prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md), [`../../../docs/specs/evaluation.md`](../../../docs/specs/evaluation.md), [`../../../docs/specs/affordance.md`](../../../docs/specs/affordance.md) | 候选回应/行动**过量生成** → 用 VZ 自有 world/self 预测模型做**廉价内部验证器**（自洽性 + 置信度）排序 → 仅 top-k 才付"昂贵真值"（真实交互 / 只读评估）；**关键增量**：把验证器分数当 readout 进快照，并定期跑"过滤 vs 随机/未过滤"对照来证明该验证器真的提升真值命中率，否则停用 | 样本高效（数十而非数百万）、从廉价代理到昂贵真值的原则化漏斗；对照检验把反证 D 的风险变成可监控指标 | 验证器可能不富集（反证 D）——必须实测富集有效性、声明校准 regime；验证器是只读 readout，不得静默成为决策第二所有者（R8/R12） |
| 2 | **"结构-然后-序列"可组合两段流水线（生成骨架 → 逆折叠填序列），各阶段拥有自己表征、下游只消费上游产物** | [`../../../docs/specs/contract-runtime.md`](../../../docs/specs/contract-runtime.md), [`../../../docs/specs/semantic-state-owners.md`](../../../docs/specs/semantic-state-owners.md) | 把 VZ 的"先定意图/计划骨架（plan_intent owner）→ 再生成具体表达（表达层）"做成同构的两段管线：上游模块发布不可变骨架快照，下游只读该快照生成内容，**格式变更只改发布方一处**；阶段间零直接调用 | 可组合、可替换单段而不动全栈（如换骨架生成器不影响序列头）；天然契合快照 SSOT 边界 | 类比级（蛋白 ≠ 对话）；阶段契约须落到 DATA_CONTRACT 的 slot 注册表，避免下游重建上游内部状态 |
| 3 | **极轻量逆折叠头 + 训练注噪提升鲁棒性**：1.7M 参数头读冻结结构表征出序列；**训练时加主链高斯噪声**故意牺牲原生恢复率，换取对不完美/预测态输入的鲁棒性与真实可设计性（noise 0.3Å → 自洽设计多 2–3×） | [`../../../docs/specs/multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md), [`../../../docs/specs/credit-and-self-modification.md`](../../../docs/specs/credit-and-self-modification.md) | 把 VZ 控制器/适配头做**轻量**（廉价、可复用、无需逐场景定制），并在训练适配头时**注入对基底状态的扰动**（模拟基底在新 regime 下的不完美表征），优化"对噪声基底态的鲁棒下游表现"而非过拟合干净基底 | 适配头在面对漂移/不完美基底态时更稳健（不对精细细节过拟合）；轻量头契合 R2 有界适配 | 注噪幅度需按 VZ 经历分布重标定；恢复率↔鲁棒性权衡要以下游真实指标（非内部 recovery）为准，否则又落入反证 D |

## 3. 一句话定位

Xaira（RFdiffusion-抗体 + ProteinMPNN）对 VZ 的价值**不在它的扩散叙事**（扩散去噪是坐标空间的离线生成采样，与 VZ 的 latent 时间控制 z_t/β_t 无关，R3/R4 是误读，裁决 survives 并需修正 99/flat note 条目），而在三点：一是 **ProteinMPNN 作为 1.7M 参数逆折叠头**给出 R2"冻结表征 + 轻量适配头"最干净的非语言样本（但整栈是全量微调，需祛魅）；二是 **"设计 → 廉价自洽性过滤 → 冷冻电镜/SPR 不可外包真值"漏斗**强背书 R12 与"PE 不外包"；三是它自陈的**"过滤 vs 未过滤无显著差异"与生成幻觉**暴露了 VZ 软验证器的**真实风险**（genuine-risk：廉价代理对昂贵真值的富集有效性必须实测、不可假定），应进风险登记。最该直接搬的是**"过量生成 → 廉价过滤 + 对照式过滤器有效性检验 → 昂贵真值"漏斗**与**结构-然后-序列的可组合两段流水线**。

## 附：本地论文清单（同目录 PDF）

- `proteinmpnn-robust-protein-sequence-design-biorxiv-2022.06.03.494563.pdf` — ProteinMPNN（2022 bioRxiv → Science 378:49–56）：1.7M 参数 MPNN 逆折叠头，乱序自回归解码，恢复率 52.4% vs Rosetta 32.9%，训练注噪提升鲁棒性，救活失败的 Rosetta/AF 设计。
- `rfdiffusion-de-novo-design-of-antibodies-biorxiv-2024.03.14.585103.pdf` — RFdiffusion for antibodies（2024 bioRxiv → Nature 2025，Xaira 员工为共同作者）：在抗体复合物上微调 RFdiffusion 扩散生成 CDR 骨架 + 框架/hotspot 条件化 + ProteinMPNN 填 CDR 序列 + 微调 RF2 自洽性过滤；4 表位实验确认，流感 HA 冷冻电镜 1.45Å RMSD。
- 谱系参考（付费未存本地）：De novo design of protein structure and function with RFdiffusion（doi:10.1038/s41586-023-06415-8，2023）奠基 vanilla RFdiffusion。
