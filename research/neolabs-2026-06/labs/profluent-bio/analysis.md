# Profluent Bio — 深度分析

- **分组 / 成熟度**：C 生物基础模型（蛋白语言模型 / 基因编辑器设计）｜ 成熟度高（Nature OpenCRISPR + ProGen 系列，ProGen3 NeurIPS 2025 spotlight）
- **一句话主张**：把蛋白序列当语言、用计算最优扩展律训练大规模生成式蛋白语言模型（PLM），从数十亿自然序列学到功能蛋白的"语法"，再用 **DPO/IRPO 偏好对齐**把基底引向湿实验测得的适应度（fitness），设计自然界没有的酶与基因编辑器。
- **主要创作者 + 血统**：Ali Madani（创始人/CEO，ProGen 作者，Salesforce Research 血统）、Jeffrey Ruffolo、Aadyot Bhatnagar、Stephen Nayfach、Erik Nijkamp（ProGen2，Salesforce）。
- **为何与 VZ 共振 / 对立**：共振于 **R2（冻结大基底 + 下游适配）** 与 **R9/R10（有界自修改：DPO 把新数据吸收进对齐层而非从头重训）**；对立/逼问点在于——本 lab 的"对齐"在实现上是**全权重受控微调（受冻结参考模型 KL 锚定 + NLL 正则约束），并非真正意义上的"冻结基底 + 独立 adapter"**；且 PLM 是纯 SSL/监督、单轨、无一级 PE 信号、每个 assay 训练一个独立专家模型。本分析**以反证为先**，诚实裁决 roster 中"在不破坏基底前提下吸收新数据"的最佳样本到底好在哪、边界在哪。

## 1. 核心逻辑（论文级 · PDF-grounded）

### ProGen2: Exploring the Boundaries of Protein Language Models（arXiv:2206.13517, 2022）
- **问题**：缺乏对"超大规模模型 + 数据如何影响蛋白模型有效性"的系统理解；蛋白工程长期依赖定向进化（随机突变 → 测量 → 迭代），效率低。
- **方法/机制**：一族自回归 transformer decoder（151M / 764M / 2.7B / 6.4B），**next-token 最大似然**为唯一目标；RoPE 位置编码、attn 与 mlp 并行电路。基底在 UniRef90 + BFD30（约 10 亿序列）上预训练；可在特定家族上继续微调（two-layer sandwich、抗体 OAS 等）。
- **关键结果（PDF 内）**：模型越大，held-out 困惑度越低（Test-max90 12.9→9.9，Test-max50 15.0→13.9，151M→6.4B，Table 2），远未过拟合。生成序列中位 TMscore 0.89、与天然蛋白序列同源度低至 11%（采到新折叠）。**关键警示**：作者明确区分 **p₀（训练分布）≠ p∞（进化稳态分布，log p∞ ∝ log fitness）**——准确估计训练分布**不等于**估计适应度。零样本适应度预测（Table 3）上 **PROGEN2-base（764M）反而最好（ρ=0.505），比 10 倍大的 PROGEN2-xlarge（ρ=0.476）和 RITA-XL 都强**——更大容量不必然带来更好的预测性能，数据分布与其偏差起决定作用。
- **局限**：单一 next-token 目标只拟合数据分布、不直接拟合功能/适应度；零样本 fitness 预测随规模非单调；纯序列、无结构/配体/RNA 上下文（图 2 中配体结合区预测置信度低）。

### Design of highly functional genome editors by modeling CRISPR-Cas（OpenCRISPR-1，bioRxiv 2024.04.22.590591 → Nature 2025）
- **问题**：天然 CRISPR-Cas 编辑器移植到人类细胞常有功能权衡（基础活性、PAM 选择性、热稳定性等）；定向进化受崎岖非凸适应度地形限制，结构指导法依赖难得的结构假设。
- **方法/机制**：构建迄今最大 **CRISPR-Cas Atlas**（挖掘 26.2 Tb 基因组/宏基因组，得 1,246,163 个 CRISPR 操纵子）。**分层专家化微调**：通用 PLM（UniRef+BFD 500M）→ 在 Atlas 上微调得 CRISPR-Cas PLM → 再微调得 Cas9 PLM；外加独立的 **gRNA encoder-decoder** 条件生成与 Cas9 配对的 crRNA/tracrRNA（蛋白 + RNA 多模态耦合）。生成 4M 序列（半数自由生成、半数用 N/C 端最多 50 残基 prompt 引导到目标家族）。
- **关键结果（PDF 内）**：跨 CRISPR-Cas 家族生成的蛋白簇为天然的 **4.8×**（Cas9 4.1×、Cas12a 6.7×、Cas13 7.1×，70% ID 阈值）。**OpenCRISPR-1 距 SpCas9 约 400 个突变**，活性与特异性与 SpCas9 相当或更优，并兼容碱基编辑——**首个完全由 ML 设计、成功精准编辑人类基因组的可编程编辑器**。
- **局限**：核心机制是**全权重领域专家化微调**（generalist → specialist），每层专家是一个新模型副本，非冻结基底 + adapter；功能验证仍重度依赖大规模湿实验筛选。

### Scaling unlocks broader generation and deeper functional understanding（ProGen3，bioRxiv 2025.04.15.649055，NeurIPS 2025 spotlight）
- **问题**：PLM 文献缺三件事——(i) 大模型的**最优训练数据分布**研究；(ii) **规模如何影响生成（而非仅 embedding）**的湿实验验证；(iii) **对齐（post-training）随规模的收益**。
- **方法/机制**：ProGen3 = 稀疏 **MoE** 自回归 PLM（8 个专家激活 2 个，每次前向仅激活 27% 参数），8 个尺寸 112M–46B，上下文 8192，支持 N→C / C→N / 中段 infill（GLM）。数据 **Profluent Protein Atlas v1（PPA-1，3.4B 全长蛋白 / 1.1T tokens，全部剔除片段）**。拟合稀疏 PLM 计算最优扩展律 **Nopt(D)=(2.462×10⁻⁷)·D^1.479**，据此训 46B / 1.5T tokens。对齐用 **IRPO（迭代推理偏好优化，DPO 的推广）** 把模型似然对齐到湿实验测得属性（活性、结合、机体适应度、稳定性）。
- **关键结果（PDF 内）**：
  - **数据分布**：4 种平衡方案中 **Inverse Log** 在 OOD（50%/30% ID）验证集最优，**Uniform（≈50% ID 去重）最差**——说明 PLM 能从"相关蛋白出现频率"中学到有用信号，过度去重反而有害。46B 验证 loss 1.345，**低于扩展律预测前沿**（延长 warmup 至 10k 步稳定训练）。
  - **生成**：更大模型生成更多通过质控的有效序列、覆盖更广的 30% ID 家族簇，且大模型覆盖簇是小模型的**超集**；湿实验（split-GFP 表达）证实大模型在更广家族上达到与天然相当的表达率。
  - **核心警示（非可辨识性）**：**规模 >3B 后零样本 fitness 预测反而变差**（ProteinGym，Figure 4a）——"更好的自然分布估计器 = 更差的适应度预测器"（强化 Weinstein 等的假设）。
  - **对齐**：大模型从对齐获益最大。对齐 ProGen3-46B 适应度预测 **ρ=0.673**，超 KERMUT（0.628）、逼近 ConFit（0.679）；稳定性预测 **ρ=0.737**（≈ ProteinDPO 0.72），仅用单突变训练却在多突变上 **ρ=0.820 vs ProteinDPO 0.468**。对齐还提升生成序列在硅/体外稳定性（in vitro 表达：8/32 提升、3/32 下降、21/32 无显著变化）。
  - **对齐的实现与边界（Appendix D，关键）**：IRPO 损失 = 偏好项（**β log[pθ/pref] 之差**，β 控制 pθ 偏离冻结参考 pref 的幅度）+ **α·NLL 正则项**（保留最高奖励完成的似然）；α=0 时退化为纯 DPO。**Table 7（1.4B）**：预训练 ρ=0.471 / ppl 8.150；纯 DPO（α=0）ρ=0.670 但 **ppl 飙到 13.874（灾难性遗忘）**；α=0.05、block 64 时 ρ=0.643 / ppl 9.712。作者明言 **NLL 项是防止验证困惑度退化的关键正则，使对齐模型保留预训练习得的广博知识**；按"困惑度退化不超过预训练 2 点"来选 α。β 固定 0.10，α 随规模取 0.05/0.02/0.01。**每个 ProteinGym assay 训练一个独立模型**。
- **局限**：所谓"对齐"在实现上是**对模型权重 θ 的受控微调**，pref 仅作 KL 锚（正则），并非冻结基底上的独立 adapter；纯 SSL + 监督偏好，无一级 PE、无双轨、无持久身份；按 assay/任务切分出大量专家模型。

## 2. 与 VZ 的关系（三视角）

> **本 lab 重心在 §2.2 反证**：ProGen3 的 DPO/IRPO 是 roster 中被 [`../../99_synthesis_vz_mapping.md`](../../99_synthesis_vz_mapping.md) 称作"在不破坏基底前提下吸收新数据"的最佳样本。先诚实拆穿它到底是不是"冻结基底"，再确证。

### 2.1 确证（先进性背书）

- **R2 冻结基底 + 下游适配（强，跨模态独立验证）**：ProGen2/3 在蛋白序列模态上独立复现"大基底 + 计算最优扩展律 + 冻结预训练表征"范式（稀疏扩展律 Nopt∝D^1.479 与稠密 PLM 的 D^1.370 高度一致）。结合 99 中 EvolutionaryScale/Arc/Isomorphic/Chai 等，"冻结大基底 + 下游头"是**跨蛋白/基因组/细胞/影像的规律**——为 VZ 的 R2 姿态提供非语言的合法性背书 → [`../../../docs/specs/multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md)。
- **R9/R10 有界自修改 + R1/R13 SSL↔RL 交替（强，可量化）**：预训练（SSL 压缩自然分布）→ IRPO 偏好对齐（强化式吸收湿实验信号）是 **SSL→RL 交替**作用于压缩基底的干净样本；更重要的是 IRPO 给出**可量化的有界性旋钮**——β（KL 锚定到冻结参考的偏离幅度）+ α（NLL 保留正则）。这正是"如何把新数据吸收进控制器/对齐层而不破坏基底"的工程化答案 → [`../../../docs/specs/credit-and-self-modification.md`](../../../docs/specs/credit-and-self-modification.md)。
- **R-PE / R12 评估与基底目标解耦（中，反向背书）**：ProGen2 的 **p₀≠p∞** 与 ProGen3 的**非可辨识性**（规模越大、越像自然分布，fitness 预测反而越差）共同证明：**基底的 SSL/next-token 目标 ≠ 下游"价值/适应度"目标**。这从反面支持 VZ 把 PE/评估/credit 作为**独立于基底压缩损失**的一级信号，而非把价值塞进基底训练目标 → [`../../../docs/specs/prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md), [`../../../docs/specs/evaluation.md`](../../../docs/specs/evaluation.md)。

### 2.2 反证（红队）

**反证 A（headline）：ProGen3 的 DPO/IRPO 被宣称为"控制器层适配，非端到端重训基底"——但诚实读 Appendix D：对齐是对模型权重 θ 的全量受控微调，pref（冻结参考）只是 KL 锚/正则，并非"冻结基底 + 独立 adapter"。**
纯 DPO（α=0）虽把 fitness ρ 推到 0.670，却使验证困惑度从 8.150 飙到 13.874——即**基底被改到灾难性遗忘自然蛋白分布**。作者靠 α·NLL 正则项 + "困惑度退化≤2 点"的早停才把基底知识保住。换言之，这不是"基底冻结、只动控制器"，而是"**基底会动，但被 KL 锚 + NLL 正则强行拉回**"的受控微调。
- **裁决：needs-boundary-condition。** DPO/IRPO 验证的是**机制形状**（用冻结参考做 KL 锚 + 保留正则 + 可调有界旋钮来吸收新信号），**不**验证"frozen substrate"本身。
- **边界条件（写入 spec，并修正 99 §3.1 的措辞）**：
  1. 99 中"DPO 是控制器层适配、非端到端重训基底"的表述**不精确**——ProGen3 的 DPO/IRPO 在数学上是对**整模型权重**做梯度更新，靠正则限制漂移。VZ 的 R2（**真正冻结基底 + 物理独立的 adapter-delta**）比 Profluent 的做法**更强**，应在 spec 注明区别，不要把 ProGen3 当作"冻结基底 + adapter"的范例，而当作"**受控漂移 + 可度量遗忘护栏**"的范例。
  2. 若 VZ 借鉴 DPO 形状，必须把它落在**有界 adapter-delta 层**（保持基底 bit 级冻结）+ 用 KL/正则限幅，而非直接梯度更新基底（[`../../../docs/specs/credit-and-self-modification.md`](../../../docs/specs/credit-and-self-modification.md) 的 ModificationGate）。
  3. "灾难性遗忘"在这里被**显式量化为 held-out 困惑度上升**——这正是 VZ 需要的"自修改可回滚/可拒收"的度量信号（R15）。

**反证 B：蛋白 LM 是纯 SSL/监督、单轨、无一级 PE 信号——"PE 一级、双轨、持久身份"是多余的。**
ProGen 全系无 prediction-error 一级对象（PE 只隐含在 next-token loss 里）、无 World/Self 双轨、对齐时**每个 assay 训一个独立模型**。
- **裁决：survives（领域不适用）+ 局部 genuine-risk（若照搬专家化范式）。** 蛋白设计是**无主体、无关系、单任务**的离线优化，本就不需要 PE 一级化 / 双轨 / 持久身份；VZ 目标域（长程关系养成）恰恰相反。但"每 assay 一个专家模型"的范式若被 VZ 照搬（每个用户/场景一个微调副本），会**碎裂单一持久身份**、违反 R7/R14。
- **边界**：VZ 不能从 PLM 继承"按任务切分专家模型"的工程惯性；新数据必须吸收进**同一持久身份下的有界控制器层**，而非裂变出独立模型 → [`../../../docs/specs/dual-track-learning.md`](../../../docs/specs/dual-track-learning.md), [`../../../docs/specs/cognitive-regime.md`](../../../docs/specs/cognitive-regime.md)。

**反证 C：非可辨识性——"更大、更像自然分布的基底，下游 fitness 预测反而更差"，说明 VZ 押注"更强基底自动更好"是错的。**
ProGen2 Table 3（base 反超 xlarge）与 ProGen3 Figure 4a（>3B 后 ProteinGym 下降）一致。
- **裁决：needs-boundary-condition（实为对 VZ 的正向压力）。** 它证伪的不是"需要强基底"，而是"**强基底 = 强下游价值**"这一隐含假设。基底优化的是分布拟合，价值/关系是另一个目标面。
- **边界**：VZ 必须在 spec 写明——基底（substrate）只负责**压缩与表征**，关系/EQ/价值由**独立的评估 + credit + 控制器层**承载，且后者**不可**通过把价值塞进基底 SSL 目标来获得 → [`../../../docs/specs/prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md), [`../../../docs/specs/evaluation.md`](../../../docs/specs/evaluation.md)。

### 2.3 局部算法借鉴（算法级解耦）

| # | 机制（剥离叙事） | 目标 VZ spec | 落地动作 | 预期收益 | 风险 / 前提 |
|---|---|---|---|---|---|
| 1 | **IRPO/DPO 双旋钮有界吸收**：偏好项用冻结参考 pref 做 KL 锚（β 限偏离）+ α·NLL 正则保留基底知识；α→0 即纯 DPO，会灾难性遗忘 | [`../../../docs/specs/credit-and-self-modification.md`](../../../docs/specs/credit-and-self-modification.md) | 把"把湿实验/关系偏好信号吸收进控制器"形式化为 DPO 形状的 adapter-delta 训练：β 锚定到**冻结基底**、α 设遗忘下限；以此作为 ModificationGate 决定**取 adapter-delta（在线/中频）还是 rare-heavy（离线重训）**的判据——偏离/遗忘超阈即升级为 rare-heavy 或拒收 | 给"在不破坏基底前提下吸收新信号"一个**可量化、可调的有界算子**；β/α 直接对应 R2 的有界性与 R9/R10 的门控强度 | 必须落在物理冻结基底上的 adapter，不可直接梯度更新基底；偏好对（preference pair）的构造质量决定一切；纯 DPO（无 α）已证会遗忘，禁止无正则 |
| 2 | **held-out 困惑度作为遗忘护栏 / 回滚闸**："对齐后困惑度退化≤2 点"作为接受微调的硬条件 | [`../../../docs/specs/evaluation.md`](../../../docs/specs/evaluation.md), [`../../../docs/specs/contract-runtime.md`](../../../docs/specs/contract-runtime.md) | 每次 adapter-delta 更新后，在**只读**的基底能力保持集上测退化（VZ 版"困惑度"=基底语言/关系基线表现）；超阈则拒收/回滚该 delta，记录到证据台账 | 把"自修改可回滚"（R15）落成一个**可执行的接受/拒收度量**，而非口号；遗忘可观测、可门控 | 该度量必须**只读**、不可反向变成训练目标（R12）；需选好"基底能力保持集"以真正代表不可退化的核心能力 |
| 3 | **Inverse-Log 频率保留式数据再平衡**：训练分布既不取自然频率（Unmodified）也不去重到均匀（Uniform 最差），而用 n/(1+log n) 兼顾常见与稀有，最优 OOD 泛化 | [`../../../docs/specs/multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md) | 在 rare-heavy 基底刷新 / 预训练阶段，对经历语料按"频率保留但温和再平衡"采样，避免对高频交互过度去重导致 OOD（新场景/新用户）泛化下降 | OOD（新关系情境）泛化提升；保留"出现频率"这一被 Uniform 抹掉的有用信号 | 仅作用于离线基底/预训练阶段（R2 的 rare-heavy），不进运行时；再平衡曲线需按 VZ 经历分布重标定 |

## 3. 一句话定位
Profluent（ProGen2/3 + OpenCRISPR）是 roster 中关于"**如何把新数据有界地吸收进对齐层**"最值得借鉴、也最需要被诚实祛魅的样本：它的 DPO/IRPO **不是**"冻结基底 + adapter"，而是"**KL 锚 + NLL 正则约束下的受控权重漂移**"——VZ 的真冻结-基底 R2 比它更严，但应直接吸收它的**双旋钮有界算子（β/α）**与**困惑度遗忘护栏**作为 ModificationGate 与可回滚证据的实现；同时它的 p₀≠p∞ / 非可辨识性从反面背书 VZ"评估与价值必须独立于基底 SSL 目标"。

## 附：本地论文清单（同目录 PDF）
- `progen2-exploring-boundaries-of-protein-language-models-2206.13517.pdf` — ProGen2（2022，151M–6.4B AR PLM，p₀≠p∞）
- `opencrispr-design-of-genome-editors-modeling-crispr-cas-biorxiv-2024.04.22.590591.pdf` — OpenCRISPR-1（2024→Nature 2025，分层专家化微调 + gRNA 条件生成）
- `progen3-scaling-broader-generation-functional-understanding-biorxiv-2025.04.15.649055.pdf` — ProGen3（2025，46B 稀疏 MoE + IRPO/DPO 对齐湿实验数据）
