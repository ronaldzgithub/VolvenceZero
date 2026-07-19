# Insitro — 深度分析

- **分组 / 成熟度**：C 生物基础模型（因果生物学药物发现）｜ 成熟度中-高（多篇 bioRxiv/medRxiv，平台工程扎实，开源 redun / EmbedGEM 工具）
- **一句话主张**："因果生物学"药物发现——用自监督从高内涵细胞影像 / H&E 组织影像学到**无偏表示**，再连到遗传学（GWAS/PRS）与分子状态（RNA/CNA），用"从形态推断分子/疾病"的可学习映射放大微弱的疾病信号。
- **主要创作者 + 血统**：Daphne Koller（创始人/CEO，概率图模型 PGM 与信用分配先驱）、Theofanis Karaletsos（前 insitro，现 CZI，贝叶斯表示学习）、Srinivasan Sivanandan / Ci Chu / Eilon Sharon（CellPaint-POSH 平台）、Zachary McCaw / Sumit Mukherjee（统计遗传学 / EmbedGEM）。血统出自概率图模型 + 高内涵筛选（HCS）+ 视觉自监督（DINO-ViT）一脉。
- **为何与 VZ 共振 / 对立**：这是一家**应用 ML-for-biology**（监督 / 自监督、任务专一）公司，对 VZ 的**认知架构不变量是弱证据**——必须诚实。**共振于 R2**（冻结影像基础模型 + 下游薄头，histopath 论文最干净）与 **R12**（EmbedGEM 是 roster 里少见的"系统化评估表示效用"的框架，是本家最相关的一块）。**但必须先红队**：flat note 把"从形态预测分子状态 = readout 模型，其误差接地发现"读成 R-PE/R3 证据是**类型错误**——这些误差是**离线监督残差/评估量**，不是运行时一级学习信号；全系**单轨、无关系、无在线适应、无 regime**。本分析以反证为先，把"R2/R12 背书"的边界讲清楚，避免确证陷阱。

## 1. 核心逻辑（论文级 · PDF-grounded）

### A Pooled Cell Painting CRISPR Screening Platform Enables de novo Inference of Gene Function by Self-supervised Deep Learning（bioRxiv:2023.08.13.553051 · CellPaint-POSH + CP-DINO）
- **问题**：pooled CRISPR 筛选便宜可扩展但读出维度低（fitness / 荧光标记），依赖**预定义的表型 marker**，无法做 hypothesis-free 探索；perturb-seq 高维但昂贵；既有 pooled optical screening 都用**假设特定**的定制 assay。能否做"通用形态表型"的无假设反向遗传筛选？
- **方法/机制**：把 Cell Painting 重造为与 in-situ sequencing 兼容（RNA 基的 Mitoprobe 替代 MitoTracker、加 RNAse 抑制剂、把逆转录移到 Cell Painting 之前）。ML 流水线：3 层 FCN 做 base-calling（解卷积细胞回收率 66.6%→78.8%），单细胞 tile + sgRNA 身份。两套特征：经典 **CellStats**（1301 个工程特征）与自监督 **CP-DINO**（DINO-ViT，无标签）。三场实验：124 基因形态 PoC、300 基因 MoA、1640 基因 druggable genome。
- **关键结果（PDF 内）**：CellStats 对 StringDB（边>0.95 为真阳）AUC **0.77**；CP-DINO 300 > ImageNet-dino > CellStats（StringDB 召回）；**CP-DINO 1640 在同等数据量（~1-1.5M cell tile）下捕获更多语义结构**（因 5× 扰动多样性，非记忆化）；跨实验泛化到 held-out 124 基因 PoC；hypothesis-free 重建出 mTORC1 / TGFβ / EGF / 脂肪生成（无脂质染色即聚出 ACLY/ACACA/SCD/FASN）等网络；加入 pS6 biomarker 通道显著改善 mTORC1 **抑制因子**（TSC1/TSC2/DDIT4/TBC1D7）检测。
- **局限（作者自述）**：对成像伪影敏感；某些细胞 ISS 信号低；对神经元/3D 肝细胞等重叠结构**分割困难**；高维表型**比转录组更难解释**（需 CellStats 互补 + perturb-seq 正交验证）。

### EmbedGEM: A framework to evaluate the utility of embeddings for genetic discovery（bioRxiv:2023.11.24.568344）
- **问题**：ML 嵌入越来越多用于遗传发现，但（i）易被协变量混淆、（ii）疾病相关性难以判定。缺一个**系统化评估"嵌入对遗传发现是否有用"**的框架。本家最相关的一篇。
- **方法/机制**：沿**两条解耦轴**比较嵌入（及任意多元性状）：（1）**可遗传性 heritability**——对嵌入做 PCA 得正交分量，逐 PC 做单变量 GWAS，因正交故 per-component Wald 统计独立，合并为 χ² 检验；度量 = 独立（clumped）全基因组显著（p≤5×10⁻⁸）位点数 + 命中处 mean/median χ²（信号强度代理）。（2）**疾病相关性 disease relevance**——对每个正交 PC 算 PRS，比较"含 PRS 的全模型 vs 不含的简化模型"对留出疾病人群的预测增量（AUROC/AUPRC 或 r²/MAE），**用置换 PRS + bootstrap 估 null** 给显著性 p 值。redun 工作流实现。
- **关键结果（PDF 内）**：合成数据正确排序三种原型（高遗传+高相关 / 低+低 / 高遗传+低相关）。UKB NAFLD：LF% 嵌入**遗传性低于** 203 个 NAFLD 性状，但**疾病相关性更高**，且 LF% 嵌入在两轴上都**优于**单变量 LF 预测——强监督嵌入比强单变量 biomarker 统计功效更高。**并非所有嵌入都有用**：ImageNet 预训练 ResNet 嵌入产生**零**独立全基因组信号、无任何疾病相关性（因捕获的是低层图像特征，与人类生物学无关）。核心教训：**只看遗传性（信号强度代理）不足，必须独立评估目标相关性**。
- **局限**：无法判定 GWS 命中里哪些是假阳（除非有金标准变体集）；正交化用 PCA 只是其中一种；监督嵌入的效用强绑定于训练任务。

### Deep Learning on iPSC-derived Motor Neurons Carrying fALS-genetics Reveals Disease-Relevant Phenotypes（bioRxiv:2024.01.04.574270）
- **问题**：ALS 高度异质，传统 anchor 表型（TDP-43 错位）微弱、跨 donor 不可靠、未产出疗法。能否用无偏 ML 表型替代一维 anchor？
- **方法/机制**：把 8 个 fALS 突变（7 基因）编辑进多 donor iPSC，造**同基因对照对**（isogenic pairs），10 天 hNIL 协议分化为运动神经元；高内涵成像（DAPI/TUJ1/TDP-43/STMN2）。自监督 **iDINO**（DINO-ViT，~140h × 8 A100），按通道子集训练（仅核心形态 vs 全通道）。三类任务：① 仅用形态（DAPI+TUJ1）嵌入预测 TDP-43 C/N（Spearman 均值 ~0.57）；② 从形态嵌入 imputation RNA 表达（multitask LASSO，跨 donor 留出，全基因 Pearson 0.5、ALS 基因集 0.84）；③ 训"疾病轴"分类器（mutant vs WT）跨 donor。
- **关键结果（PDF 内）**：**仅核心形态**即可预测 TDP-43 错位（该信号本身极微弱）；RNA imputation 复现 C9ORF72 的 STMN2 下调；疾病轴显著优于 ancher 表型，**达同等检验力少用 80% 细胞**；VCP 突变的 neurite 缺陷被模型发现并经 GSEA（树突棘形态/轴突运输通路下调）跨模态佐证。
- **局限**：效应量微弱、强 donor 混淆、**细胞密度混淆**（必须密度匹配 + 协变量校正，否则分类器学的是活性而非疾病态）；bulk RNA 仅 well 级，无法单细胞 imputation。

### Machine learning enabled prediction of digital biomarkers from whole slide histopathology images（medRxiv:2024.01.06.24300926）
- **问题**：预测性 biomarker（IHC / 测序）需专用 assay、慢、贵、跨中心不一致。能否从普遍采集的 H&E 影像同时预测多种分子因子？
- **方法/机制**：训一个**泛实体瘤 H&E 基础模型**（ViT，DINO 自监督，3M tiles / TCGA，256×256 @1MPP），产出 768 维通用 tile 表征；在**冻结表征**之上训多任务下游头预测 352 个药靶的 CNA / RNA 表达 / **扩增签名**（amplification signature = 差异表达基因加权和，一个**派生量**）。tile 级预测后按患者平均；再在同一表征上训**专门化模型**（单癌种/单 biomarker）。
- **关键结果（PDF 内）**：泛癌 AUROC：CNA **0.734**、RNA **0.853**、SIG **0.897**（连续/派生量比离散稀疏事件更可学）；**多任务 >> 单任务**（transcriptome-wide multi-task Spearman 0.628 vs per-target **0.004**）；**tile 级预测后聚合 > 先平均表征（0.628 vs 0.584）> MIL 注意力（0.404）**；TCGA→cohort A 跨数据集迁移仅小幅掉点（pan-cancer RNA -13%），部分子集反升；尽管只用 bulk 训练，tile 级预测的空间叠加与盲法病理学家标注吻合；Cabozantinib 案例：**未训练任何结局数据**，VEGFR2 扩增签名即预测总生存（HR 0.087）。
- **局限**：biomarker 不直接对应蛋白丰度；临床结局数据极少（仅少量靶能验证）；专门化在签名预测上提升、却以表达预测掉点为代价（需保留泛癌模型）。

## 2. 与 VZ 的关系（三视角）

> **纪律：先反证，后确证。** Insitro 全系是**监督 / 自监督的 ML-for-biology**，对 VZ 认知架构不变量是**弱证据**。最容易被读成"从形态预测分子 = readout 模型 + 误差接地 = R-PE/R3"，但其真实机制是**离线监督 imputation 的残差**与**离线 SSL 表征**；真正站得住的只有 **R2**（限定到"冻结基础模型 + 下游薄头"）与 **R12**（EmbedGEM 的评估方法论，本家最强相关的一块）。R-PE/R3/R4/R7 在这里**缺席或被误读**。

### 2.1 确证（先进性背书）

- **R2（中-强 · 跨模态独立验证）**：histopath 论文是**最干净**的样本——H&E 基础模型（DINO ViT）训完后**冻结**产出 768 维 featurization，所有 352 靶 × 3 模态预测、以及单癌种"专门化模型"都只在**冻结表征上叠下游头**；CP-DINO 同样推理时确定性产出 embedding + 下游分类。这是在**组织影像 / 细胞影像两个非语言模态**上对"冻结大基底 + 下游薄层"的背书 → 可写入 [`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md)：R2 是跨模态规律。**注意边界**：fALS 的 iDINO 是**按 use-case 端到端 SSL 重训**（非冻结通用基底），不算 R2 干净样本。
- **R12（中-强 · 本家最相关，且为"评估方法论"而非"被印证的公理"）**：**EmbedGEM 把"嵌入是否有用"拆成两条解耦、可证伪的轴**（heritability = 内在信号强度/可学习性代理；disease relevance = 对目标的**增量**预测，且用 permutation + bootstrap 估 null），并实证"ImageNet 嵌入遗传性看似有结构却**零疾病相关性**"——这正是 VZ R12"评估覆盖存在、只读、防自欺"的跨域镜像 → [`evaluation.md`](../../../docs/specs/evaluation.md) / [`evidence_program.md`](../../../docs/specs/evidence_program.md)。这是**方法论借用**，不是"又一条公理被生物学印证"。
- **R5/R6（弱-中 · 派生索引的工程类比）**：四篇共同范式是"从原始模态（影像）压缩出可检索/可下游的学习表征"，histopath 更把表征用于检索式的"语义合成队列"——对应 VZ"在不可变记忆/快照之上构建**派生索引**"的类比 → [`continuum-memory.md`](../../../docs/specs/continuum-memory.md)（派生索引层，owner 拥有、消费者只读）。仅类比，非强证据。

### 2.2 反证（红队）

**反证 A（headline · 拆穿"从形态预测分子 = readout 模型，其误差 = R-PE"）：flat note 把 insitro 的"预测模型误差即发现信号"读成 R-PE 证据，但这些误差是 imputation 模型的**离线监督残差 / 留出评估量**，不是 VZ 的运行时一级原始信号（evaluation/credit 是其下游 readout）。**
- **裁决：needs-boundary-condition**。诚实地说：RNA imputation 的 Pearson、disease-axis 的 accuracy、histopath 的 AUROC 都是**部署后被人类离线读取的评估指标**，模型本身是确定性、无误差回路的特征提取/预测器；它既不持续产生 PE、也不被 needs/homeostasis/credit 实时读出去驱动适应。把"预测残差"升格为 R-PE 是**类型错误**。
- **边界（写入 spec）**：[`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md) 应注明"离线监督 imputation 残差 / 留出评估指标 ≠ 运行时一级 PE"；R-PE 证据须来自有**预测—观测—残差—适应**闭环的来源。insitro 在 R-PE 上**沉默**，既不证实也不证伪。

**反证 B（拆穿"嵌入 = z_t latent 控制器"）：insitro 的所有 embedding 是**感知表征 / 派生索引**（ViT patch 平均向量），不是控制器代码 z_t，也无时间抽象 / 无 β_t 切换 / 无在 latent 空间的规划或决策。**
- **裁决：survives（不适用）**。这些模型从不做运行时控制，embedding 是离线产物，喂给**线性 / LASSO / 浅 MLP 下游头**做静态预测。它对 R3/R4（控制发生在 token 之上的 latent 空间、时间抽象一等）**沉默**，不构成反例。
- **边界**：引用 insitro 时只取"冻结基底 + 表征"骨架，**不得**把其感知 embedding 当作 R3/R4 的 latent 控制证据。

**反证 C：insitro 全系是**单轨、冻结、无 Self、无关系、无 regime、无在线适应**的特征提取 + 监督预测系统——"足够大的冻结基底 + 静态下游头"在其目标域就够了，VZ 的双轨（R7）、持久 regime（R14）、在线快层（R1）是过度工程？**
- **裁决：survives**。insitro 目标域是**有客观 oracle 的单步任务**（StringDB 网络召回、TDP-43 C/N 真值、CNA/RNA 测量、生存结局）；VZ 目标域是**多月至多年、无硬奖励的关系养成**。insitro 的成功恰建立在"有可验证标准"之上——这正是关系/EQ 域**缺失**的前提。其极简单轨结构不适用于需 World/Self 隔离与持久身份的养成场景，故对 R7/R14 **沉默**而非反对。

**反证 D（IQ vs EQ · 产品取向对立）：insitro 是纯 IQ 向（药物发现），目标是**任务正确性**；VZ 产品核心是关系/EQ/信任。这是否意味着"learned 表征 + 监督头"路线对 VZ 不适用？**
- **裁决：survives（领域不同，不构成证伪）**。两者目标域正交：insitro 证明"在有验证器的任务域，冻结基底 + 监督头很强"，这对 VZ 的**底层 substrate / 记忆索引层**仍有借鉴；但**不能**把其"任务正确性优先"叙事搬进 VZ 的关系养成主链。**风险登记**：若 VZ 误把 insitro 式"监督预测准确率"当成主评估目标，会把 R12 的"覆盖存在"塌缩成"任务正确性"，违反 R12（评估不得变成学习源）。

**反证 E（Koller PGM 血统 ≠ VZ 信用分配 R10 的直接证据）：flat note 称"Koller 的 PGM 血统对应 VZ 概率/信用机制（R10）"。**
- **裁决：needs-boundary-condition**。这四篇论文里**没有**任何分层信用分配 / 门控自修改的运行时机制；PGM/信用是创始人**学术血统**，不是这批 PDF 的可借鉴算法。把血统当证据是**叙事层映射**，违反 98 的"算法层 vs 叙事层"纪律。
- **边界**：R9/R10 证据须来自有显式 credit assignment + ModificationGate 的机制（见 Thinking Machines trust-region 等），insitro 这批 PDF 不提供。

### 2.3 局部算法借鉴（算法级解耦）

| # | 机制（剥离叙事） | 目标 VZ spec | 落地动作 | 预期收益 | 风险 / 前提 |
|---|---|---|---|---|---|
| 1 | **EmbedGEM 双轴解耦评估 + permutation/bootstrap null**：把"一个表示是否有用"拆成（a）**内在信号强度**（heritability 类比：表示是否承载可被结构化检出的信号）与（b）**对目标的增量相关性**（disease relevance 类比：含该表示的全模型 vs 不含的简化模型，对**留出**目标的预测增量），用置换 + bootstrap 估 null 给 p 值 | [`evidence_program.md`](../../../docs/specs/evidence_program.md)、[`evaluation.md`](../../../docs/specs/evaluation.md)（R12） | 把"评估 VZ 内部表示（z_t / 记忆嵌入 / 派生索引）效用"做成两条解耦轴：① 表示是否有结构（内在），② 表示对**关系目标**是否有**增量预测**（全 vs 简化模型 + null 检验）；只读、不反向当学习源 | 防"表示看起来漂亮但对目标零增量"的自欺（EmbedGEM 实证 ImageNet 嵌入即此类）；给 R12"覆盖存在而非任务"一个可证伪、可计算的方法论骨架 | 须保持**只读**（R12），不得把评估分数反向当训练目标；"增量 vs null"口径需按关系域重标；目标标签（关系质量）本身难获取 |
| 2 | **预测连续派生签名 > 预测原始离散事件 + 多任务共享表征**（histopath：SIG 0.897 > CNA 0.734；transcriptome multi-task 0.628 vs per-target 0.004）：用一个由 owner 定义的**连续派生量**（如加权差异签名）替代稀疏/离散的硬标签，并在冻结表征上**多任务共享**学习 | [`semantic-state-owners.md`](../../../docs/specs/semantic-state-owners.md)、[`continuum-memory.md`](../../../docs/specs/continuum-memory.md) | 当 VZ 需要预测/命名内部状态（如 relationship_state、open_loop）时，优先让 owner 发布**连续派生签名**而非稀疏离散标签；下游用共享表征多任务预测 | 连续量统计功效更高、更可学；多任务跨语义 owner 共享冻结表征，省数据；契合 R11"内部状态可命名可发布" | 派生签名须由**owner 模块**定义并发布（R8/R11），不可外部硬编码加权；签名口径变更走快照契约 |
| 3 | **分布外留出作为泛化硬标准 + 密度/混淆显式校正**（fALS：donor holdout 交叉验证、密度匹配 + 协变量校正，否则分类器学的是活性而非疾病态；histopath：TCGA→cohort A 跨数据集留出）：把"跨个体 / 跨数据集留出"作为 OOD 鲁棒性的硬验证，且**显式识别并扣除已知混淆**再评估 | [`evaluation.md`](../../../docs/specs/evaluation.md)（R12 评估协议） | 在 VZ 评估协议里写入"跨用户 / 跨会话 / 跨场景留出"作为分布外泛化的硬门槛，并要求**显式枚举并校正已知混淆**（如活跃度、会话长度）后再读评估分 | 防"在分布内看着好、换用户就崩"与"分类器学了混淆代理"两类经典失败；提升 R12 评估的可信度 | 混淆清单需由评估 owner 维护；留出切分须无泄漏（同 patient/donor 不跨 train/test，对应"同用户不跨切分"） |

> 备选（次优）：**hypothesis-free 形态网络重建 + biomarker 通道增益**（CellPaint-POSH：无显式 marker 即聚出通路网络，加 pS6 通道才检出抑制因子）→ 可作为"无预定义标签下用 SSL 表征做关系/行为模式涌现聚类"的远距类比，对应 ETA 的"模式从数据涌现、不靠关键词硬规则"（契合 no-keyword 规则），但与 VZ 关系域距离较远，列为低优先观察项。

## 3. 一句话定位
Insitro 是一家**应用 ML-for-biology** 公司，对 VZ 认知架构的真正背书只有两点且都被收窄：**R2**（histopath 的"冻结 H&E 基础模型 + 下游薄头"是干净跨模态样本）与 **R12**（EmbedGEM 是 roster 里最系统的"评估表示效用"方法论，本家最相关）；其余对 R-PE/R3/R4/R7/R9/R10 **沉默或被 flat note 误读**（监督 imputation 残差 ≠ 运行时 PE、感知 embedding ≠ z_t 控制器、PGM 血统 ≠ 信用机制证据）。它教给 VZ 的不是"更多公理被印证",而是 **EmbedGEM 的双轴 + null 评估法、"连续派生签名 + 多任务共享表征优于离散单任务"、"跨个体留出 + 混淆显式校正"** 这三件**可直接搬进评估 / 记忆索引层而不碰 substrate**、且强化 R12"只读、防自欺"的方法论。

## 附：本地论文清单（同目录 PDF）
- `pooled-cell-painting-crispr-de-novo-gene-function-biorxiv-2023.08.13.553051.pdf` — CellPaint-POSH + CP-DINO（hypothesis-free 形态 CRISPR 筛选，CP-DINO 1640 > CellStats/ImageNet-dino）
- `embedgem-evaluating-embeddings-for-genetic-discovery-biorxiv-2023.11.24.568344.pdf` — EmbedGEM（评估嵌入遗传发现效用的双轴框架，本家最相关 R12）
- `deep-learning-ipsc-motor-neurons-fals-phenotypes-biorxiv-2024.01.04.574270.pdf` — iDINO fALS 运动神经元（仅形态预测 TDP-43 错位 / RNA imputation / 疾病轴省 80% 细胞）
- `ml-prediction-digital-biomarkers-from-histopathology-medrxiv-2024.01.06.24300926.pdf` — H&E 泛癌基础模型 + 多任务数字 biomarker（SIG 0.897，multi-task >> per-task，Cabozantinib 零结局训练预测生存）
