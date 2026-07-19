# Inceptive — 深度分析

- **分组 / 成熟度**：C 生物基础模型（RNA "生物软件"）｜ 成熟度中（公司自身几乎不公开发表，证据主要来自创始人血统与联创学术工作）
- **一句话主张**：把 RNA（尤其 mRNA）当作可编程的"生物软件"，用 Transformer/深度学习在高通量化学映射实验数据上学习 RNA 结构/性质，反过来设计治疗性分子。
- **主要创作者 + 血统**：
  - Jakob Uszkoreit（联创/CEO，"Attention Is All You Need" 八位共同作者之一，提出"用 self-attention 取代 RNN"的最初想法并发起评估）。
  - Rhiju Das（联创，Stanford 生化，Eterna 众包平台创始人，Ribonanza 通讯作者）。
- **为何与 VZ 关系特殊（且需克制）**：本 lab 与 VZ 的连接**主要是血统与方法论，而非架构主张**。Transformer 是 VZ 选择**冻结**的基底本体，但它是 Google Brain 2017 的成果、不是 Inceptive 论文，把它读成"对 VZ 的背书"是确证偏误。Ribonanza 是 Das 的学术众包项目，其对 VZ 的价值集中在 **R12（硬基准建设 + 数据质量纪律）**，**不是架构**。**本分析以"诚实降权 + 先反证"为基调：Inceptive 在 33 家中对 VZ 相对边缘。**

## 1. 核心逻辑（论文级 · PDF-grounded）

> 必须严格三分：**① 创始人奠基论文（Transformer，非 Inceptive 作品）｜② 公司方向（RNA 生物软件，几乎无第一方论文）｜③ 联创数据集论文（Ribonanza，Das）**。

### ① 创始人奠基论文：Attention Is All You Need（1706.03762, NIPS 2017）

- **归属说明**：Vaswani/Shazeer/Parmar/**Uszkoreit**/Jones/Gomez/Kaiser/Polosukhin 八人等贡献。这是 **Transformer 本体**，**不是 Inceptive 的工作**；它出现在本目录仅因 Uszkoreit 是 Inceptive 联创。
- **问题**：主流序列转导模型基于 RNN/LSTM/CNN，沿位置串行计算（hₜ 依赖 hₜ₋₁），**无法在样本内并行**，长序列受内存约束；长程依赖的路径长度随距离增长（ConvS2S 线性、ByteNet 对数），难学。
- **方法/机制**：提出**完全基于注意力、彻底抛弃递归与卷积**的编码器-解码器。核心算子：
  - **Scaled Dot-Product Attention**：`softmax(QKᵀ/√dk)V`，用 1/√dk 缩放避免大 dk 下 softmax 进入小梯度区。
  - **Multi-Head Attention**：h=8 头并行，各头 dk=dv=64，投影到不同表示子空间后拼接；联合关注不同位置的不同表示子空间。
  - **三种注意力用法**：encoder 自注意力、decoder 带掩码自注意力（防止看到未来位置、保持自回归）、encoder-decoder 交叉注意力。
  - **逐位置 FFN**（dff=2048）、**残差 + LayerNorm**、**正弦位置编码**（可外推到更长序列，与学习式位置嵌入效果近乎相同）。base：N=6、d_model=512、65M 参数。
- **关键结果（PDF 内）**：自注意力层最大路径长度 **O(1)**（递归为 O(n)），每层复杂度 O(n²·d)。WMT14 EN-DE **28.4 BLEU**（big，超此前含 ensemble 最佳 >2 BLEU，新 SoTA），EN-FR **41.8 BLEU**（单模型 SoTA，训练成本 < 此前 SoTA 的 1/4）；big 模型 8×P100 训练 3.5 天。泛化到英语成分句法分析：semi-supervised **F1 92.7**，WSJ-only 40K 句也 91.3，几乎无任务特定调参。
- **局限**：注意力为 **O(n²)** 复杂度（长序列需受限注意力，论文列为 future work）；对 VZ 而言，这是一篇**奠基架构**论文，意义是"提供被冻结的基底"，**不构成对 VZ 控制器/记忆/PE 设计的任何特定背书**。

### ② 公司方向：RNA 作为"生物软件"（无第一方论文，UNVERIFIED）

- 公司主张：用大规模化学映射实验数据训练 RNA 结构/性质模型，再反向**学习式设计 mRNA 序列**（稳定性、表达、可制造性）。Inceptive 自身几乎不发表，**此方向无可供 PDF-grounded 核验的第一方论文**，仅能由创始人血统（Uszkoreit 的 Transformer + Das 的 RNA 众包）推断。诚实标注：**UNVERIFIED**。

### ③ 联创数据集论文：Ribonanza（bioRxiv 2024.02.24.581671, Das 等）

- **归属说明**：通讯作者 Rhiju Das（Inceptive 联创），主体是 Eterna + Kaggle 双众包学术项目，**非 Inceptive 公司论文**。
- **问题**：RNA 从序列预测结构仍是未解问题，进展被**实验数据稀缺**拖慢；既有努力存在三大病灶——3D 坐标稀缺、**评估缺乏严谨性**、深度学习二级结构模型泛化差。此前**没有盲测**能严格检验"化学映射数据能否训练出预测模型"。
- **方法/机制（核心是"双众包 + 前瞻性盲测"）**：
  - **数据众包（Eterna）**：公民科学家通过 "OpenKnot" 挑战设计/收集复杂结构（含假结）RNA；对 2M 条多样序列做 **DMS + SHAPE(2A3) 化学映射**（标记未配对核苷酸）。
  - **模型众包（Kaggle）**：三个月盲赛招募独立团队建模。**严格分离数据设计者 / 实验者 / 建模者三类角色**。
  - **前瞻性盲测**：私有榜（Private Leaderboard）的绝大多数序列**在比赛开始后才合成测量**，且**刻意用更长序列（207–457 nt）** 与训练集（115–206 nt）形成**长度 OOD**，强制检验长度泛化。评估用 MAE（抗离群）。
  - **基线 RNAdegformer**：Transformer 编码器 + 1D 卷积（捕捉局部 motif）+ 用序列距离矩阵和 EternaFold BPP 矩阵**偏置注意力**；翻转增强（3'↔5'）。
  - **蒸馏 RibonanzaNet**：把多个 Kaggle 顶模整合为**单个、纯序列（不需 BPP）** 自包含模型；并用 top-3 预测的伪标签提升。
- **关键结果（PDF 内）**：
  - 891 名参赛者 / 755 队，**20 队超过 RNAdegformer 基线**；前 50 队**公共榜 vs 私有榜 Spearman rs = 0.82** —— 表明模型在"未来数据"上泛化良好、且 MAE 是合理指标（**前瞻性评估纪律的直接证据**）。
  - in-silico mutate-and-map 显示：训练数据从 1.4K→14K→140K，模型**逐步涌现出三茎假结的内部结构表示**（包括纯序列模型在 140K 时重平衡出三茎假结）。
  - RibonanzaNet（单模型）**超过 Kaggle 第一名**（后者为多模型 blend）。下游微调：RibonanzaNet-SS 在 PDB 测试 **F1 0.875**（次优 HFold 0.856），CASP15 RNA **F1 0.937**；并在 dropout、降解（OpenVaccine/PERSIST-seq）等异构任务上超既有模型。
  - **关键 R2 证据**：**不预训练直接微调，F1 从 0.875 跌到 0.7** —— 证明 Ribonanza 大规模预训练数据对迁移学习不可或缺。
- **局限**：未能前瞻性评估 3D 结构（等 CASP16）；2M 序列约 4 亿核苷酸级测量，仅 **8000 万**达可接受信噪比，远小于 NLP 的万亿级语料；用 RibonanzaNet-SS 改进 3D RMSD 在不同分子上**不一致**。

## 2. 与 VZ 的关系（三视角）

> **本 lab 重心在 §2.2 反证 + 诚实降权**：先承认它对 VZ 架构相对边缘，再界定确证与可借鉴机制。Transformer 是被冻结的基底（非背书），Ribonanza 的价值在评估纪律（非架构）。

### 2.1 确证（先进性背书）

- **R12（强，且是本 lab 真正的信号）**：Ribonanza 的**前瞻性盲测 + 长度 OOD + 比赛后才测量私有榜 + 三角色严格分离**，是"**先把硬的、不可泄漏的评估建好，再宣称进展**"的范本；公共/私有榜 Spearman 0.82 证明评估口径可信。这正是 VZ R12"评估覆盖存在、只读、不可被反向当学习源"以及 evidence program 的跨域独立印证 → [`evaluation.md`](../../../docs/specs/evaluation.md), [`evidence_program.md`](../../../docs/specs/evidence_program.md)。
- **R2（中，跨模态独立验证；但须克制）**：Transformer 是 VZ 选择冻结的基底架构本体；Ribonanza"不预训练 → F1 0.875 跌至 0.7"在**完全非语言模态**上独立证明"预训练基底 + 下游微调头"的范式收益。**诚实边界**：这印证的是 R2 的跨模态合法性，**不是** Inceptive 对 VZ 的特定背书——VZ 用 Transformer 仅作为被冻结的 substrate → [`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md)。
- **R-PE（弱）**：RNA 结构/性质预测误差（化学反应活性 MAE）是序列设计的学习信号。但这是**常规监督损失**，不是 VZ 意义上的"一级原始 PE + epistemic/aleatoric 分离"。**判定弱，不作为背书**。

### 2.2 反证（红队）

**反证 A（headline · 诚实降权）：Transformer 的标题就是"Attention Is All You Need"——若一个统一缩放的注意力架构即可达成 SoTA，VZ 叠加的 latent 控制器（z_t/β_t）、多时间尺度循环、显式记忆连续谱是否是过度工程？**
- **裁决：survives**。Transformer 解决的是**序列转导/任务能力（IQ）**，在其目标域（翻译、句法）确实"够用"；但它**不解决** VZ 的目标域——关系/EQ/长程养成/持久身份。Transformer 在 VZ 中恰恰被定位为**被冻结的基底**（R2），z_t/β_t 控制层与它正交、不是它的竞品。"Attention is all you need" 对应的是表达基底，VZ 的不变量针对的是基底**之上**的控制与适应。
- **边界条件（写入 spec）**：在 R2/R4 注明"Transformer 作为冻结表达基底被采纳；VZ 的控制器层不替代也不重训该基底，二者属不同抽象层"，避免把"基底够强"误读为"控制器多余"。

**反证 B（本 lab 的核心诚实点）：把 Transformer 与 Ribonanza 算作"Inceptive 对 VZ 的证据"是血统错配。**
- **裁决：needs-boundary-condition（针对调研口径，非针对 VZ 不变量）**。Transformer 是 Google 2017 成果，Ribonanza 是 Das 的学术众包项目；Inceptive 公司本身**几乎无第一方可核验论文**。把它们当作"这家 lab 验证了 VZ"会重蹈 99 的确证偏误。
- **边界**：本 lab 的确证一律**降权登记**——R12 来自 Ribonanza（Das 学术），R2 来自 Transformer（行业公共基底），**均不计为 Inceptive 公司层面的独立背书**。Inceptive 在 ROI 台账中应标"低优先 / 边缘"。

**反证 C：Ribonanza 的成功源于"海量监督数据 + 众包多样性"驱动，是否说明 VZ 应优先堆可标注数据规模，而非纠结架构/控制纪律？**
- **裁决：needs-boundary-condition**。RNA 域存在**廉价、可测量的真值信号**（化学反应活性、可大规模 Illumina 测序），所以"scale 标注数据"奏效。VZ 的核心目标（关系质量、EQ、regime）**没有等价的廉价真值**，无法照搬"堆标注数据"。
- **边界**：数据规模教训仅适用于**存在可验证标签**的子任务；关系域必须走"软验证器"路线（见 [`99_synthesis_vz_mapping.md`](../../99_synthesis_vz_mapping.md) §四.1 的 rBio）。把 Ribonanza 的数据范式直接外推到关系养成 = 类别错误。

### 2.3 局部算法借鉴（算法级解耦）

| # | 机制（剥离叙事） | 目标 VZ spec | 落地动作 | 预期收益 | 风险 / 前提 |
|---|---|---|---|---|---|
| 1 | **前瞻性、不可泄漏的"未来数据"盲测 + 长度 OOD 强制泛化**：私有榜序列在评估开始后才采集，且刻意改变分布（更长序列），公共/私有榜相关性（Spearman 0.82）反过来验证评估指标本身 | [`evaluation.md`](../../../docs/specs/evaluation.md), [`evidence_program.md`](../../../docs/specs/evidence_program.md) | 在 evidence program 增设"时间前瞻 hold-out"：关系/养成评估集的一部分**在模型冻结后才采集**，并刻意制造分布偏移（更长会话跨度 / 新 regime），用公共-私有一致性校准评估指标可信度 | 防评估泄漏与过拟合榜单；为"评估指标本身是否可信"提供可量化校验，强化 R12 只读纪律 | 关系域真值采集成本高、周期长，需设计低成本代理；切勿让 hold-out 反向变成训练源（R12） |
| 2 | **双众包 + 三角色严格分离**：数据设计者 / 实验者 / 建模者互不重叠，最大化数据与模型多样性并杜绝评估污染 | [`evidence_program.md`](../../../docs/specs/evidence_program.md) | 在评估流程中制度化"出题/标注/被测系统"角色隔离；引入外部/多来源场景设计，避免自产自评 | 提升评估多样性与抗污染性；降低"自己出题自己满分"的系统性偏差 | 仅作评估治理流程借鉴，不引入 Inceptive/Eterna 的具体任务内容 |
| 3 | **置信度自评分（eF1 / eF1,crossed-pair）旗标低置信预测**：用模型自身成对分数的简单组合估计预测可靠度，显式标出"不自信"区域 | [`evaluation.md`](../../../docs/specs/evaluation.md), [`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md) | 评估读出层增设"自评置信度"通道，对低置信的关系判断打旗标供只读监控（不回写控制器） | 让评估能区分"自信对/不自信"，为 PE readout 提供可解释的不确定性维度 | 必须只读（R12）；置信分不得静默成为第二学习信号（R-PE 不外包） |

> 说明：Transformer 的 multi-head attention / 正弦位置编码等**不列为"借鉴机制"**——它们是 VZ 已采纳的**冻结基底本体**，属 R2 既有事实，而非本次新增的算法级 ROI。

## 3. 一句话定位

Inceptive 对 VZ **相对边缘**：其与 VZ 的两条连接都是血统而非公司主张——**Transformer（Uszkoreit）是 VZ 冻结的基底本体（R2，非背书）**，**Ribonanza（Das）的真正价值在前瞻性硬基准 + 双众包评估纪律（R12）**；公司自身几乎无可核验产出。可吃的高价值机制仅在评估治理层（前瞻 hold-out、角色分离、置信度旗标），架构层无新增 ROI；在 ROI 台账中应标"低优先 / 评估方法论"。

## 附：本地论文清单（同目录 PDF）
- `attention-is-all-you-need-1706.03762.pdf` — Transformer（NIPS 2017，创始人 Uszkoreit 共著；**非 Inceptive 论文**，VZ 冻结基底本体）
- `ribonanza-deep-learning-rna-structure-dual-crowdsourcing-biorxiv-2024.02.24.581671.pdf` — Ribonanza / RibonanzaNet（2024，联创 Das 通讯；双众包 + 前瞻盲测数据集论文）
