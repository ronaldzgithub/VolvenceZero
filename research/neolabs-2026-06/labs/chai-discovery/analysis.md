# Chai Discovery — 深度分析

- **分组 / 成熟度**：C 生物基础模型（结构预测 + 生成式设计）｜ 成熟度中-高（bioRxiv + 已释放权重/网页服务，未进正刊）
- **一句话主张**：开放的 AlphaFold3 级全原子多模态结构预测（Chai-1），并延伸为零样本生成式抗体/binder 设计（Chai-2），用稀疏湿实验约束（epitope/contact）提示引导、用 in-silico 置信度排序后再进湿实验。
- **主要创作者 + 血统**：Joshua Meier（联创/CEO，前 ESM 核心作者）、Jack Dent（联创/CTO）、Jacques Boitreaud、Matthew McPartlon、Kevin Wu、Zhuoran Qiao（NeuralPlexer）等。血统直接来自 ESM 蛋白语言模型一脉 + AlphaFold3（Abramson et al.）架构谱系。
- **为何与 VZ 共振 / 对立**：共振点单一但极干净——Chai-1/Chai-2f 在**冻结的大蛋白语言模型 embedding 之上加一个训练好的 pair-bias 结构头**，是全 roster 对 **R2（冻结基底 + 轻量自适应头）最干净的跨模态样本**。但必须诚实：它是**监督式**（结构 ground-truth 损失）、**非 PE 驱动**、**单轨（只有 World，无 Self）**、**无在线适应/无记忆连续谱/无 token 之上的 latent 控制**。因此本分析以"先反证、剥离叙事"为纪律，避免把一个监督折叠器读成 VZ 的全面背书。

## 1. 核心逻辑（论文级 · PDF-grounded）

### Chai-1: Decoding the molecular interactions of life（biorxiv:2024.10.10.615955, 2024-09）
- **问题**：要做药物发现级别的结构预测，需要在蛋白-配体、蛋白多聚体、抗体-抗原、核酸等多模态上同时达到 SoTA，且最好能在没有 MSA（共进化信息）时也可用，并能吸收湿实验约束。
- **方法/机制**：架构大体沿用 AlphaFold3（Abramson et al.），重度依赖 **pair-bias self-attention**。关键新增三点：(1) **蛋白语言模型 embedding 输入轨道**——用一个 **30 亿参数**的蛋白语言模型（引用 ESM 系，[8]）为每个输入序列生成**逐残基 embedding**，作为与 MSA/模板并列的**可选输入特征**；非蛋白链（DNA/RNA/配体）给 mask token，修饰残基回退到母体残基或 "X"。(2) **约束特征**：pocket（token+chain+距离阈值 θ_P∈(6,20)）、contact（token 对 + θ_D∈(6,30Å)）、docking（四 bin 距离 one-hot），训练时按 10% 概率独立纳入并做 chain/token dropout，约束数量按几何分布 p=1/3 采样，避免模型过度依赖约束。(3) 单一模型覆盖全部评测（训练数据截止 2021-01-12，与评测集无重叠）。训练：128×A100、batch 128、30 天。推理：4 recycle、5 trunk×5 diffusion=25 个结构，再用置信度模型（ipTM）排序。
- **关键结果（PDF 内具体数字）**：
  - **蛋白-配体（PoseBusters，ligand RMSD<2Å）**：Chai-1 **77.05%** ≈ AF3 76.34%，RF2AA 42%；给 apo 结构做 docking 提升到 **81.20%**。
  - **蛋白-蛋白（DockQ>0.23 成功率）**：Chai-1(含 MSA) **0.751**，单序列模式 **0.698**，AF2.3 0.677（p=6.24×10⁻¹⁰）。
  - **蛋白-抗体界面**：Chai-1(MSA) **0.529**，单序列 **0.479**，AF2.3 0.380；抗体集上**单序列 ≈ 含 MSA**（变异序列共进化信号本就稀薄）。
  - **蛋白单体 Cα-LDDT**：Chai-1(MSA) **0.915** > AF2.3 0.903；但**单序列 0.852 反而劣于 AF2.3**（单体折叠仍吃 MSA）。
  - **CASP15（69 单体）**：Chai-1 LDDT **0.849** > AF2.3 0.843；在 AF2.3 难例（LDDT<0.75）上 **0.643 vs 0.552**（n=14, p=3.66×10⁻⁴）；ESM3-98B 报告 0.801。
  - **单序列模式整体超过 ESMFold**，是首个无 MSA 即能高精度多聚体折叠、且含 MSA 时超过 AF2.3 的模型。
  - **约束提示**：单个 contact(≤15Å) 把抗体-抗原 acceptable 从 35%→57%；四个 epitope 残基使各档成功率较 blind 翻倍以上——但 high-quality 仍仅 **4-8%**，说明高精度抗体-抗原预测整体仍难。
- **局限（PDF 明确写出）**：(1) 常能预测对单链，却**摆不对链间相对取向**（无 contact 时复合物质量差）；(2) 对**修饰残基极敏感**——去掉/替换修饰会大幅改变预测（模型把同序列有/无修饰当成不同输入）。

### Chai-2: Zero-shot antibody design in a 24-well plate（biorxiv:2025.07.05.663018, 2025-06）
- **问题**：完全从头（de novo）设计可结合的功能性抗体一直难成；既往计算法湿实验命中率罕超 **0.1%**，必须高通量筛千万级设计，抵消了计算设计的意义。
- **方法/机制**：Chai-2 是**单一端到端生成模型**，含两个子模块：**Chai-2d**（设计，all-atom 生成框架，同时设计骨架+侧链原子结构）与 **Chai-2f**（折叠，架构沿用 Chai-1，同样用 PLM embedding+MSA+模板，支持约束输入，用于对设计打分排序）。流程：给定靶标结构 + **仅 1-4 个 epitope 残基**（用 10Å Cα-Cα 截断从已知非抗体结合界面随机采）+ 从最常用治疗性 VHH/VH-VL 框架中选一个 scaffold；模型设计全部 CDR（含长度），保留 scaffold；in-silico 排序后**每靶标只挑 ≤20 个设计**直接进湿实验（BLI 测亲和）。**全程无 per-target 微调**，训练截止 2021-09-30，并剔除与设计靶标 >70% 同一性的结构。**关键诚实点**：未做多轮湿实验迭代——"this was our first attempt ever at generating a binder for nearly every target"（单轮盲测）。
- **关键结果（PDF 内具体数字）**：
  - **de novo 抗体命中率**：摘要 **16%**；正文全体均值 **15.5%**（VHH 20.0% / scFv 13.7%）——较既往 SoTA（<0.1%）**约 2 个数量级 / >100 倍**提升。
  - **靶标覆盖**：52 个**无任何已知抗体**的新靶标中，**26/52（50%）**至少得到 1 个 binder（单轮 ≤20 设计）。
  - **分模态成功率**：binder 级——minibinder **68%（75/111）**、VHH 20%（41/205）、scFv 14%（68/496）；target 级——minibinder **100%（5/5）**、VHH 56%（10/18）、scFv 49%（21/43）。
  - **miniprotein**：每靶标 ≥3 倍于次优方法；**首个计算设计的 TNFα binder**（公认 top-1% 难度）；多个 picomolar 亲和（IL7Rα/PD-L1/PDGFRβ/InsulinR），TNFα 低 nM。
  - **新颖性（非记忆）**：所有设计 CDR 编辑距离 >10、无设计在已知抗体 2Å RMSD 内、绝大多数 >10Å；多数靶标的 binder 含多个结构簇（探索多构象）。
  - **特异性**：23 个 binder 中仅 **1 个（4%）**对 off-target 有背景以上结合；可设计人/cyno 交叉反应抗体。
  - **折叠增益驱动设计**：Chai-2f 把抗体-抗原 DockQ>0.8（近实验精度）的比例**从 Chai-1 的 17% 翻倍到 34%**（无模板时 3 倍差距）；给 epitope 约束后 >40%（含模板）/32%（无模板）。
- **局限（PDF 明确写出）**：(1) 仅测了 scFv/VHH 结合，未测 Fab/全长 mAb，亲和可能改变；(2) 只表征**结合**，热稳定/聚集/免疫原性等成药性仍待测；(3) CDR loop 构象柔性仍是瓶颈，miniprotein 的 α/β scaffold 简单故命中更高；(4) **设计质量强依赖底层折叠精度**——折叠错则设计随之劣化。

## 2. 与 VZ 的关系（三视角）

> **纪律：先反证。** Chai 的 R2 样本"太干净"，最容易触发确证偏误——把一个监督式、单轨、无 PE、无在线适应的折叠/设计器读成"全面印证 VZ"。下面先用 PDF 证据红队，再保留可信的确证。

### 2.1 确证（先进性背书）

- **R2 冻结基底 + 轻量自适应头（强，跨模态独立验证 —— 但"冻结"需诚实标注为推断）**：Chai-1/Chai-2f 把一个 **30 亿参数蛋白语言模型的逐残基 embedding 当作与 MSA/模板并列的输入轨道**，其上训练 pair-bias 结构头。这是非语言模态（蛋白三维结构）上"大基底产出表示 + 轻量任务头消费表示"的干净实例，为 R2 提供**跨模态合法性背书** → [`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md)。**最强证据是单序列模式**：去掉 MSA/模板、仅靠序列+PLM embedding 仍达 SoTA 多聚体折叠（蛋白-蛋白 0.698、抗体 0.479），证明大基底确已把进化/结构先验压进 embedding。**诚实边界**：论文方法节只说"训练于 MSA 与 PLM embedding 的组合""该模型生成逐残基 embedding"，并把 embedding 当**可选输入特征**——**全文未出现"frozen/freeze"字样，也未说联合微调 PLM**。"冻结"是由"作为固定输入特征轨道使用、与 MSA/模板同列"强烈推断而来，**不是论文明述的事实**，引用时须如实标注为架构推断。
- **R3/R4 零样本条件化（弱，且属输入空间提示而非 latent 控制）**：Chai-2 "无 per-target 微调"，靠 **epitope 残基 + scaffold 选择**在**输入空间提示**一个冻结生成模型即得到针对新靶标的设计，类比**控制器层 few-shot/zero-shot 条件化** → [`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md)。**但须剥离叙事**：这是对生成模型的输入条件化（prompting），**不是**学到的时间抽象或 z_t/β_t latent 控制，因此只能作为"条件化可零样本泛化"的弱旁证，不能算 R3/R4 的核心机制背书。

### 2.2 反证（红队）

**反证 A（headline）：Chai-1 = 冻结 PLM + 训练 pair-bias 头，纯监督即达 SoTA —— 那"PE 作为一级信号、双轨、在线适应"是否多余？一个监督式冻结基底+头就够了。**
PDF 事实：Chai 的学习信号是**结构 ground-truth 损失**（监督/扩散），没有内禀预测误差、没有 Self 轨、没有在线适应、没有 token 之上的 latent 策略。一个如此"瘦"的范式横扫多模态，表面上像在说"R2 的架构成立，但 R-PE/R7/R1 都不必要"。
- **裁决：survives（目标域不匹配）**。反例不适用于 VZ 的目标域。结构预测有**稠密的、客观的 ground-truth 标签**（PDB 晶体结构），监督损失天然可得；VZ 的目标域（关系/EQ/长程养成）**没有 ground-truth 标签**——这正是 PE 必须成为一级信号的根本原因（无标签时，预测误差是唯一可得的、内生的学习信号）。Chai 恰恰反向印证：**有标签处用监督，无标签处才需要 R-PE**。
- **边界条件（写入 spec）**：R2 的"冻结基底 + 轻量头"被 Chai 跨模态证实可泛化（可在 [`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md) 引用）；但 R-PE 的成立前提是**目标域缺乏稠密客观标签**——须在 [`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md) 写明"为何关系域不能照搬 Chai 式监督折叠"。

**反证 B（纠正确证偏误 —— 直接修正本 lab 旧 flat note 的 R-PE 主张）：旧笔记称"约束化提示 + in-silico 排序（湿实验前）≈ 预测误差门控的设计循环"。**
PDF 反证：Chai-2 **明确未做多轮湿实验迭代**——"we did not perform successive rounds of wet lab experimentation… first attempt ever"（单轮盲测）；in-silico 排序用的是**学到的置信度头（ipTM）**对 25 个采样**选优**，**没有把环境（湿实验）反馈回灌模型**。所以它根本**不是闭环 PE，而是开环的 in-silico 置信度过滤**。
- **裁决：needs-boundary-condition**。该类比仅在"in-silico 置信度门控的候选过滤"意义上成立，**不构成预测误差反馈回路**；因此**不应把 Chai 列为 R-PE 的证据来源**（这是对旧笔记的纠偏）。真正的 R-PE 证据应来自 active inference / ICM / rBio（见 99 综合），而非 Chai。
- **边界条件**：在 [`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md) 标注：Chai 的"排序后再湿实验"= 只读置信度门控（属 R12 readout 性质），**不是** PE 闭环；引用时不得混淆。

**反证 C：Chai-2 无 per-target 训练、一个冻结模型靠条件化即泛化到全新靶标 —— 是否说明"无需在线适应，冻结模型 + 提示就够"？**
- **裁决：survives + 收窄边界**。Chai 的零样本泛化之所以成立，是因为**蛋白结合/折叠由稳定物理规律支配**（靶标分布是物理平稳的）；VZ 的 per-user 关系/regime 是**非平稳、随对象漂移**的，需要 online-fast 控制器适应（R1）。
- **边界条件**：在 [`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md) 写明——"条件化即零样本"只在**目标分布物理平稳**时充分；VZ 的关系域非平稳，故仍需有界在线适应层。

**反证 D：Chai 是纯 World 单轨（分子现实），无 Self 轨却极成功 —— 双轨隔离（R7）是否过度设计？**
- **裁决：survives（域外）**。Chai 的任务**本就只有 World**（分子是否如此结合是客观事实，无"自我"维度）；VZ 的关系养成内生地需要 World（对用户/世界的预测）与 Self（对自身状态/承诺/边界的预测）双轨。Chai 不构成对 R7 的反例，只是任务维度不含 Self。无需改 spec，但可在 [`dual-track-learning.md`](../../../docs/specs/dual-track-learning.md) 注明"单轨足够的任务 = 不含自我维度的纯客观预测任务"作为边界示例。

### 2.3 局部算法借鉴（算法级解耦）

| # | 机制（剥离叙事） | 目标 VZ spec | 落地动作 | 预期收益 | 风险 / 前提 |
|---|---|---|---|---|---|
| 1 | **冻结大基底 embedding 作输入轨道 + 轻量任务头**：把 30 亿参 PLM 的逐残基 embedding 当固定输入特征，其上只训练 pair-bias 结构头；单序列模式证明基底已承载先验 | [`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md) | 作为 R2 的**跨模态合法性论据**写入 spec：范式 = 冻结/慢更新基底产出 embedding，仅训练有界任务头消费之；不对基底做端到端在线梯度 | 跨模态（非语言）独立验证"冻结基底 + 轻量头"可达 SoTA，给 VZ 的 R2 姿态硬背书 | 它是**监督式**——只借鉴**架构解耦**，不要连训练范式一起搬；且"冻结"在原文是推断而非明述，引用须标注 |
| 2 | **零样本条件化（无 per-target 训练）**：靠稀疏提示（epitope 残基 + scaffold）条件化一个冻结生成模型，泛化到全新靶标 | [`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md), [`affordance.md`](../../../docs/specs/affordance.md) | 作为**控制器层 few-shot/zero-shot 条件化**的工程对照：用稀疏可命名提示条件化冻结基底产出行为，而非为每个新对象重训 | 印证"条件化即可零样本泛化"在平稳域成立，支持控制器层 few-shot 接口设计 | 这是**输入空间 prompting**，非 z_t latent 控制；VZ 的关系域非平稳，零样本不充分，须叠加在线适应（见反证 C） |
| 3 | **训练期约束 dropout + 稀疏可选约束条件化**：pocket/contact/docking 特征按 10% 独立纳入、做 chain/token dropout、数量按几何分布采样，使模型有/无约束都稳健 | [`semantic-state-owners.md`](../../../docs/specs/semantic-state-owners.md), [`contract-runtime.md`](../../../docs/specs/contract-runtime.md) | 当系统接受**可选的可命名提示**（如用户明示的边界/目标）作为条件时，训练/适配期对该提示做 dropout，避免过度依赖；约束以快照可读字段进入 | 提示在场则增益、缺席也不崩，避免对稀疏外部约束的脆性依赖 | 仅是条件化鲁棒性技巧，不触及学习信号本质；约束须作为 owner 发布的快照字段（R8/R11），不得旁路 |
| 4 | **只读校准置信度排序，行动前做 in-silico 过滤**：ipTM 与真实质量良好校准，在 25 个采样里选优，只让高置信候选进入高成本湿实验 | [`evaluation.md`](../../../docs/specs/evaluation.md) | 在提交**高成本关系行动**前，用只读校准 readout 对候选回应排序/门控，仅 commit 高置信者；置信度本身不回灌为学习信号 | 廉价 in-silico 过滤降低高成本错误，契合 R12"评估只读" | **必须只读**（R12），不得变成学习源；且不得与 PE 闭环混淆（见反证 B）——这是过滤，不是预测误差反馈 |

## 3. 一句话定位

Chai Discovery 是 VZ **R2（冻结基底 + 轻量自适应头）最干净的跨模态样本**——单序列模式证明大蛋白语言模型 embedding 已承载结构先验，pair-bias 头只需轻量训练即达 SoTA；但它**监督、单轨、无 PE、无在线适应**，因此只背书 R2 的**架构解耦**，并反向澄清三件事：(1) R-PE 的必要性恰在 Chai 不具备的**无标签关系域**；(2) Chai 的"排序后湿实验"是**开环 in-silico 过滤**而非 PE 闭环（纠正旧笔记的 R-PE 误标）；(3) 零样本条件化只在**物理平稳域**充分，非平稳的关系域仍需在线适应。可借鉴的局部机制是"冻结基底+轻量头解耦""稀疏可选约束的 dropout 条件化"与"行动前只读校准过滤"，但都需剥离其监督训练范式。

## 附：本地论文清单（同目录 PDF）
- `chai-1-decoding-molecular-interactions-of-life-biorxiv-2024.10.10.615955.pdf` — Chai-1（2024）：SoTA 全原子多模态结构预测，冻结 PLM embedding + pair-bias 头 + 约束特征，单序列模式
- `chai-2-zero-shot-antibody-design-24-well-plate-biorxiv-2025.07.05.663018.pdf` — Chai-2（2025）：16% 零样本 de novo 抗体命中、50% 靶标覆盖、68% miniprotein 成功；Chai-2d 生成 + Chai-2f 折叠打分
