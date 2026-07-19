# Recursion — 深度分析

- **分组 / 成熟度**：C 生物基础模型（表型组学虚拟细胞）｜ 成熟度高（NeurIPS 2023 Workshop + CVPR 2024 + NeurIPS 2024，开放权重 OpenPhenom / Phenom-1）
- **一句话主张**：在海量 Cell Painting 显微图像（最大 RPI-93M，9300 万张）上做自监督**掩码自编码（MAE）**，构建表型组学基础模型，把基因/化学扰动连到细胞形态，再用对比学习把**分子结构 ↔ 表型**对齐成联合检索空间，画出"生物地图"服务药物发现。
- **主要创作者 + 血统**：Oren Kraus、Kian Kenyon-Dean、Berton Earnshaw（Phenom MAE 通讯/主力）、Maciej Sypetkowski、Dominique Beaini（Valence Labs，MolGPS / MolPhenix）；Chris Gibson（联创/CEO）、Imran Haque（前 SVP AI）。血统出自高内涵筛选（HCS）+ 计算机视觉自监督一脉，并自建超算 BioHive-1。
- **为何与 VZ 共振 / 对立**：**共振于 R2** —— Phenom MAE 是把"冻结大基底 + 下游探针"范式落在**显微图像这一完全非语言模态**上的干净样本，MolPhenix 更明文**冻结 Phenom1 与 MolGPS、只训一个薄对比头**，是 roster 里少见的"双冻结基底 + 极小可训层"实例。**但必须先红队**：MAE 重建与对比 S2L 都是 **SSL / 跨模态对齐目标，不是运行时 PE**；全系**单轨、无关系、无在线适应**；flat note 把"分子→形态预测"读成"细胞响应世界模型、残差=扰动 PE"是**过度引申**——本分析以反证为先，把"R2 跨模态背书"的边界讲清楚，避免确证陷阱。

## 1. 核心逻辑（论文级 · PDF-grounded）

### Masked Autoencoders are Scalable Learners of Cellular Morphology（arXiv:2309.16064, NeurIPS 2023 GenAI&Bio Workshop）
- **问题**：HCS 显微筛选里如何从细胞表型量化并关联基因/化学扰动？传统靠定制分割+特征提取管线（CellProfiler）或弱监督分类（WSL），前者脆弱、后者在大数据集上**不增反降**。能否用纯自监督在原始显微图像上学到可扩展、可泛化的形态表示？
- **方法/机制**：在四个数据集（RxRx1 12.5 万 / RxRx3 220 万 / 私有 RPI-52M / RPI-93M）上训练 MAE：ViT 编码器（ViT-S/B/L，22/86/304M）+ 25M 解码器，随机掩码 8×8 或 16×16 patch（掩码率 25%/75%），仅对**被掩码 patch** 算 L2 重建损失；另训 MU-Net（U-Net 版 MAE）与 WSL DenseNet-161 基线对照。最大模型 ViT-L/8+ 在 RPI-93M 的 **35 亿 crop** 上训练 >20,000 A100·小时，加了一项防发散/促纹理的损失项。推理时编码器**确定性**产出 embedding，按 well 平均聚合，做 **TVN（典型变异归一化，拟合于阴性对照）** 批次校正，再用余弦相似度衡量扰动间关系。
- **关键结果（PDF 内）**：ViT-L/8+（RPI-93M）相对最佳 WSL 基线在已知生物关系召回上**最高 +28%**；recall 随训练 FLOPs（模型×数据规模）单调上升，而 WSL 从 RxRx1 扩到 RxRx3 反而**下降**；CORUM/hu.MAP/Reactome/StringDB 召回（top/bottom 5%）达 **.62/.44/.27/.48**。
- **局限**：评估锚定"已知关系召回"这一**有客观 oracle** 的任务；TVN 等批次校正是召回成立的强前提；纯离线表示学习，无在线/反思机制。

### Masked Autoencoders for Microscopy are Scalable Learners of Cellular Biology（arXiv:2404.10242, CVPR 2024 · Phenom-1 / OpenPhenom）
- **问题**：把上面的工作做成可发布的显微基础模型，并解决两个工程瓶颈——大 ViT-MAE 训练发散、以及**不同实验室通道数/通道含义不一致**导致模型无法迁移。
- **方法/机制**：（a）**Fourier 域重建损失** \(L_{MAE+}=(1-\alpha)L_{MAE}+\alpha L_{FT}\)（α=0.01），稳定大模型训练并复现"double-descent"，改善高频纹理重建；（b）**通道无关架构 CA-MAE**：把每个通道当作独立模态，生成 C×N token，对所有通道共享同一线性投影与正弦位置编码，训练时每通道独立解码器——使编码器在**测试时可接受任意通道数/顺序**。在 RxRx3 上评估已知关系召回，并迁移到 **JUMP-CP**（不同显微镜/实验室/通道结构的 OOD 数据集）。
- **关键结果（PDF 内）**：最佳 MAE ViT-L/8+（RPI-93M）相对最佳 WSL 在 hu.MAP 上 **+11.5%**；中间 checkpoint（epoch 1/25/46）显示**重建质量与下游召回同步提升**——证明"图像重建是捕获生物信息的合适代理任务"；批次校正消融（CORUM）从无校正 **.124 → TVN .622**；CA-MAE ViT-L/16+ 在 JUMP-CP 扰动检索达 **.95**（联合嵌入 5 个 Cell Painting + 3 个 Brightfield 通道，尽管只在 6 通道上训练）；线性回归预测 955 个 CellProfiler 形态特征，MAE 全面优于 WSL（Intensity 中位 R² **.737 vs .297**，+148%）。
- **局限**：**sibling 检索反而更差**（MAE 表示太"高分辨"，把生物学相关的近邻扰动也区分开了）——作者明说需要**额外的微调/对齐**才能服务具体应用任务；CA-MAE 未扩到最强的 ViT-L/8+（token 太多）。

### How Molecules Impact Cells: Contrastive PhenoMolecular Retrieval（arXiv:2409.08302, NeurIPS 2024 · Valence Labs · MolPhenix）
- **问题**：能否学一个**分子结构 ↔ 表型实验**的联合潜空间，做"给定显微图像，零样本检索出施加的分子"（contrastive phenomolecular retrieval）？难点：配对数据比文本-图像少一个数量级、批次效应、**无活性分子**（对细胞无形态影响→等同对照、误标注）、**浓度**剧烈影响效应。
- **方法/机制**：三条 guideline——（1）**用冻结的单模态预训练编码器**：图像走 Phenom1（MAE ViT-L/8），分子走 MolGPS（1B 参数 MPNN），两者**均冻结**，只训一个中等（38.7M）对比头；用 Phenom1 embedding **跨实验平均**来边缘化批次效应；（2）**S2L 损失**（soft-weighted sigmoid locked loss）：在 SigLIP（sigmoid、抗标签噪声）与 CWCL（用单模态相似度做连续软标签）基础上，用 **Phenom1 空间内的 arctan-L2 距离**（而非余弦——因余弦无法把"无活性-无活性"与"活性-无活性"分开）计算样本间相似度作软标签，并据 p 值欠采样无活性分子；（3）**浓度编码**：隐式（把同分子不同浓度当不同类）+ 显式（one-hot / log / sigmoid 拼接进分子编码器）。
- **关键结果（PDF 内）**：活性分子零样本 top-1% 检索 **77.33%**，相对前 SOTA（CLOOME）**8.1×**；S2L 在多数设置优于 CLIP/SigLIP/CWCL；不计 Phenom1 训练，MolPhenix 比 CLOOME 基线**快 8.4×**；附录显示分子编码器可零样本预测分子活性（AUC ~0.90）并回收 ChEMBL 基因敲除↔分子关联。
- **局限**：作者明列——**无湿实验验证**；假设初始细胞态恒定（单一细胞系 HUVEC-19）；未纳入文本/基因扰动等更多模态。整体是**静态嵌入对齐 + 检索**，无前向预测的闭环。

## 2. 与 VZ 的关系（三视角）

> **纪律：先反证，后确证。** Recursion 最容易被读成"用世界模型 + 预测残差印证 R-PE/R3"，但其真实机制是 **SSL 重建 + 跨模态对比对齐**；只有 R2（且严格限定到"冻结基底 + 薄探针/对齐头"）站得住，R-PE/R7/R12-存在性在这里**缺席或被误读**。

### 2.1 确证（先进性背书）

- **R2（强 · 跨模态独立验证）**：Phenom MAE 推理阶段编码器**确定性产出 embedding**、下游只做 TVN + 余弦/线性探针；MolPhenix 更**明文冻结 Phenom1 与 MolGPS**、仅训 38.7M 对比头——这是在**显微图像 + 分子图**两个非语言模态上对"冻结大基底 + 下游薄层"的干净背书，且 MolPhenix 证明冻结基底能把配对数据需求降一个数量级。可写入 [`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md)：R2 是跨模态规律，且"冻结基底之上叠极小可训层"在数据稀缺时反而更稳。
- **R5/R6（中 · 跨模态派生索引的工程范例）**：MolPhenix 把两个冻结基底的输出**对齐进联合潜空间做检索**，并用 Phenom1 空间内的样本间距离反过来定义训练软标签——这正是 VZ"在不可变记忆/快照之上构建**派生索引**、用嵌入相似度检索"的跨模态实例 → [`continuum-memory.md`](../../../docs/specs/continuum-memory.md)（派生索引层，owner 拥有、消费者只读）。
- **SSL（强）**：MAE 中间 checkpoint 显示"重建质量↑ ⇒ 下游召回↑"，且 recall 随 FLOPs 单调上升——是"压缩式自监督建立冻结基底"的范例 → [`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md) 最低频 SSL 层。

### 2.2 反证（红队）

**反证 A（headline · 拆穿"MAE 重建残差 = R-PE"的误读）：MAE 的 L2 重建损失与 MolPhenix 的对比 S2L 都是离线 SSL / 跨模态对齐目标，看似"预测误差驱动"，实则与 VZ 的 R-PE（运行时一级原始信号、evaluation/credit 是其下游 readout）不是同一回事。**
- **裁决：survives**。重建/对比损失是**离线训练期**的标量优化目标，模型部署后编码器是**确定性、无误差回路**的特征提取器；它既不持续产生、也不被 needs/homeostasis/credit 读出。把 MAE 残差读成 R-PE 是**类型错误**——既不证实也不证伪 R-PE，只是沉默。
- **边界（写入 spec）**：在引用 Recursion 作 R2/SSL 证据时，**不得**把重建/对比损失列为 R-PE 证据；[`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md) 应注明"图像重建/对比对齐损失 ≠ 运行时一级 PE"。

**反证 B（拆穿 flat note 的"分子→形态 = 细胞响应世界模型，残差 = 扰动 PE"）：MolPhenix 不是一个对细胞响应做前向预测的世界模型，而是分子↔表型的对比检索/对齐；它没有"预测—观测—残差"的闭环。**
- **裁决：needs-boundary-condition**。诚实地说：MolPhenix 学的是 \(f_{\theta_m}(m,c)\) 与 \(f_{\theta_x}(x)\) 两个映射进同一空间后的**相似度排序**，不是 \(P(x_{t+1}|x_t,\text{action})\) 式的可滚动世界模型；"世界模型 + PE"框架是 flat note 的**引申**，非论文主张（论文甚至明说无湿实验闭环、假设初始细胞态恒定）。若要把它当 VZ 世界模型证据，需补"它只覆盖单步静态对齐、无时间动力学"的边界。
- **边界（写入 spec）**：VZ 的世界模型/PE 证据应来自有显式预测-观测-残差回路的来源；Recursion 只能作为"**跨模态嵌入对齐**"证据，不可升格为"细胞响应世界模型"。

**反证 C：Recursion 全系是单轨、冻结、无 Self、无关系、无 regime、无在线适应的特征提取+检索系统——"一个足够大的冻结基底 + 静态探针/对齐头"在其目标域就够了，VZ 的双轨（R7）、持久 regime（R14）、在线快层（R1）、存在性评估（R12）是过度工程。**
- **裁决：survives**。Recursion 的目标域是**有客观 oracle 的单步任务**（已知生物关系召回、replicate 一致性 p 值、ChEMBL 配对）；VZ 目标域是**多月至多年、无硬奖励的关系养成**。Recursion 的成功恰恰建立在"有可验证标准"之上——这是关系/EQ 域**缺失**的前提。其极简单轨结构不适用于需要 World/Self 隔离与持久身份的养成场景。
- **边界**：Recursion 证明"在有客观验证器的任务域，冻结基底 + 薄头很强"；它对 R7/R12-存在性/R14 **沉默**，不构成反例。

**反证 D：Phenom 的内部状态是稠密 embedding（ViT patch 平均向量），不可命名、不可发布——与 VZ R8/R11"内部状态可命名可发布快照"对立。**
- **裁决：survives（不适用）**。这些模型从未声称做可治理运行时；embedding 是离线产物。它不挑战 R11，只是不提供 R11 证据。
- **边界**：引用时只取 R2 的"冻结 + 下游薄层"骨架与跨模态对齐机制，**不**把不透明稠密向量当作可接受的运行时状态形态。

### 2.3 局部算法借鉴（算法级解耦）

| # | 机制（剥离叙事） | 目标 VZ spec | 落地动作 | 预期收益 | 风险 / 前提 |
|---|---|---|---|---|---|
| 1 | **双冻结基底 + 薄可训对齐头的跨模态检索**（MolPhenix Guideline 1）：图像/分子各用冻结预训练编码器，仅训 ~38.7M 对比头把两空间对齐成联合检索空间；用一侧 embedding 的跨实验平均边缘化噪声 | [`continuum-memory.md`](../../../docs/specs/continuum-memory.md)（派生索引层）、[`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md)（R2） | 把"在不可变记忆/冻结 substrate 之上叠一个**小可训层**构建跨模态派生索引"作为 R2 合规的检索机制写入；派生索引 owner 拥有、消费者只读 | 给 R2"冻结基底 + 极小可训层在数据稀缺时更稳/更快（数据需求降一个数量级、训练快 8.4×）"提供硬证据；提供合规的派生索引工程范式 | 对齐头属控制器层、基底保持冻结（R2）；**不得**据此对基底做端到端微调 |
| 2 | **用基底自身相似度生成软标签 + null/无信号样本处理**（MolPhenix Guideline 2，S2L）：在冻结基底空间内用 **arctan-L2 距离**（余弦区分不开 null）计算样本间连续相似度做软标签，并据 p 值（replicate 一致性）**欠采样无活性样本** | [`continuum-memory.md`](../../../docs/specs/continuum-memory.md)（记忆显著性/写入门控）、[`evaluation.md`](../../../docs/specs/evaluation.md) | 借鉴"用 substrate 内距离定义软标签 + 显式识别并降权'无信号/对照级'样本"的机制，用于记忆写入显著性判定与派生索引训练（区分"有意义事件"与"无信号噪声"） | 在数据噪声大时提升派生索引/记忆质量；给"哪些 episode 值得沉淀"一个可计算的软判据（非关键词硬规则，契合 no-keyword 规则） | arctan-L2 阈值与 p 值口径需按 VZ 模态重标；"无信号"定义须由 owner 模块给出而非外部硬编码 |
| 3 | **相对参照系归一化 / centering-at-control**（Phenom MAE：TVN 拟合阴性对照、按 well/plate 平均聚合）：把表示**减去/对齐到一个"中性基线"参照**再做相似度，召回从 .124→.622 | [`semantic-state-owners.md`](../../../docs/specs/semantic-state-owners.md)、[`continuum-memory.md`](../../../docs/specs/continuum-memory.md) | 借鉴"把内部表示相对一个显式中性/基线参照做归一化后再比较/检索"的思路：派生索引与语义状态快照在比较前先减去一个由 owner 维护的基线，抵消上下文/批次漂移 | 显著提升跨上下文检索/比较的稳健性（消除"批次效应"类漂移）；让快照比较更可解释（相对基线而非绝对） | 基线参照须由 owner 显式维护并发布（R11），不可外部临时重建；归一化属消费者读取技巧，不改基底 |

> 备选（次优）：**CA-MAE 通道无关 token 化**（每通道当独立模态、共享投影 + 正弦位置编码、测试时接受任意通道数）→ [`environment-interface.md`](../../../docs/specs/environment-interface.md)：可借鉴为"对变长/变构输入模态稳健的 substrate 摄取接口"，但与 VZ 关系域距离较远，列为低优先观察项。

## 3. 一句话定位
Recursion 是 VZ **R2 在显微图像/分子图两个非语言模态上的强背书来源**，且 MolPhenix 的"**双冻结基底 + 薄对齐头做跨模态派生索引**"是 roster 里最契合"R2 合规检索/记忆索引"的工程范式；但其全系本质是**离线 SSL 重建 + 对比对齐**，对 R-PE/R7/R12-存在性/R14 **沉默或被误读**——它教给 VZ 的不是"更多公理被印证"，而是"把跨模态对齐当派生索引、把基底自身距离当软标签、把表示相对中性基线归一化"这三件**可直接搬进记忆/索引层而不碰基底**的局部技巧。

## 附：本地论文清单（同目录 PDF）
- `masked-autoencoders-scalable-learners-of-cellular-morphology-2309.16064.pdf` — Phenom MAE 前身（NeurIPS 2023 Workshop，ViT-L/8，93M 图像）
- `masked-autoencoders-for-microscopy-scalable-cellular-biology-2404.10242.pdf` — CA-MAE / Phenom-1 / OpenPhenom（CVPR 2024，Fourier 损失 + 通道无关架构）
- `molphenix-contrastive-phenomolecular-retrieval-2409.08302.pdf` — MolPhenix（NeurIPS 2024，Valence Labs，分子↔表型对比检索，8.1× SOTA）
