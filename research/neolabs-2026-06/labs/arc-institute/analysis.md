# Arc Institute — 深度分析

- **分组 / 成熟度**：C 生物基础模型（基因组 + 虚拟细胞）｜ 成熟度高（Science / Nature + 全开放权重；Evo / Evo2 / State 三代连续工作）
- **一句话主张**：在**单核苷酸分辨率的原始 DNA 序列**上做大规模自回归预训练，得到一个跨 DNA/RNA/蛋白（中心法则三模态）、跨所有生命域的**通用基因组基底**；并在此范式下把"虚拟细胞"建成**扰动→响应的预测器**（State），实现零样本功能预测与多尺度生物设计。
- **主要创作者 + 血统**：Patrick Hsu（联创）、Brian Hie（Arc/Stanford）、Silvana Konermann（联创）、Eric Nguyen / Garyk Brixi / Michael Poli（主力，Poli 同时是 StripedHyena/Hyena 作者、Liquid AI）、Christopher Ré（Stanford HazyResearch，S4/Hyena 血统）。**架构血统直接连到 Cartesia/Liquid 的深度信号处理（SSM/Hyena）社区**。
- **为何与 VZ 共振 / 对立**：强共振于 **R2**（冻结大基底 + 零样本下游 readout，在完全非语言的基因组/细胞模态上独立验证）；Evo 的 StripedHyena（hyena = data-controlled 卷积，SSM 家族）是 **Cartesia 有界隐状态主张的跨领域交叉验证（R3/R4）**；State 的"扰动→响应"是 VZ **World 轨预测/世界模型**的概念近邻（R7 + R-PE 的借鉴源）。**但本 lab 是检验"把训练损失误读成 R-PE""把 State 误读成干净 R2"两个确证陷阱的关键样本——本分析以反证为先。**

## 1. 核心逻辑（论文级 · PDF-grounded）

### Evo: Sequence modeling and design from molecular to genome scale（biorxiv 2024.02.27.582234）
- **问题**：此前生物 ML 是**模态专用**（蛋白 / 调控 DNA / RNA 各一套），生成只限单分子或短 DNA；而基因调控、CRISPR 免疫、转座等复杂过程依赖跨模态、跨基因的系统级相互作用。需要一个在**单核苷酸分辨率**上统一分子→系统→基因组尺度的 DNA 基底。
- **方法/机制**：**Evo = 7B 参数、StripedHyena 架构、131kb（131k token）上下文、字节级单核苷酸 tokenizer**。StripedHyena 是混合架构——**29 层 data-controlled 卷积（hyena）算子 + 3 层（10%）带 RoPE 的多头注意力**：hyena 层承担主体序列处理（短/长卷积滤波，擅长过滤 DNA 噪声、把核苷酸聚合成 motif），注意力层补"从上下文精确召回"的能力。在 **OpenGenome（300B 核苷酸 token，8 万+ 细菌/古菌基因组 + 数百万原核噬菌体/质粒序列）** 上做**纯 next-token 预测预训练（无显式监督/注释）**。出于生物安全**排除感染真核宿主的病毒**。两阶段：先 8k 上下文，再扩到 131k。
- **关键结果（PDF 内）**：首个 DNA 预训练 scaling laws 分析，训练 300+ 模型横跨 Transformer++ / Mamba / Hyena / StripedHyena——**Transformer++ 在字节分辨率上 perplexity 显著最差**；Hyena/StripedHyena scaling rate 最优且训练稳定，而 **Transformer++ 与 Mamba 出现数值不稳定**。零样本：在 E. coli 蛋白 DMS 适应度上**媲美 SOTA 蛋白语言模型**、在 ncRNA 上**超过专用 RNA 语言模型**、仅凭调控序列预测 promoter-RBS 表达活性；**无监督预测基因必需性（nucleotide resolution）**；生成合成 CRISPR-Cas 复合体与完整转座系统、生成 **>650kb** 编码丰富序列（比此前方法长数个数量级）。
- **局限**：仅原核数据 → 对人类蛋白适应度预测能力受限；长序列生成相干性下降（新颖 CRISPR-Cas 采样频率低、prompt 可控性中等、偶把 Cas9 token 生成成 Cas12）；基因组级生成缺关键标志基因（如完整 tRNA repertoire）。**这些都是 SSL 生成模型的固有约束，非 PE 驱动的在线适应。**

### Evo 2: Genome modeling and design across all domains of life（biorxiv 2025.02.18.638918）
- **问题**：把原核范式扩到**真核基因组**——后者有大量非编码区、可变剪接、多层表观调控，长度/复杂度数量级更高，需要在数据策展、架构、训练/推理基础设施、推理时算力四方面全面升级。
- **方法/机制**：**Evo 2 = 7B 与 40B 两版，9.3T DNA 碱基（OpenGenome2，8.8T 核苷酸，覆盖细菌/古菌/真核/噬菌体），1M token 上下文，单核苷酸分辨率**。新架构 **StripedHyena 2（首个卷积 multi-hybrid）**：三类输入相关卷积算子——**short explicit (SE) / medium regularized (MR) / long implicit (LI) hyena**——加注意力，按条纹排布。40B 规模下相对高度优化的 Transformer 在 16k 上下文 **1.3× 加速、1M 上下文 3× 加速**。两阶段：8192 上下文预训练（数据加权聚焦 genic window）→ midtraining 扩到 1M。仍排除真核宿主病毒。
- **关键结果（PDF 内）**：**无 variant-specific finetuning、无 MSA**，即可对从非编码致病突变到临床 BRCA1 变体的功能影响打分，noncoding 致病性预测达 SOTA；在 Evo2 embedding 上训练的**监督模型**对 BRCA1 未知意义变体分类达 SOTA。**机制可解释性（稀疏自编码器 SAE）揭示模型自主学到 exon–intron 边界、转录因子结合位点、蛋白结构元件、prophage 区域等生物特征**。生成：线粒体/原核/真核基因组级序列，自然度与相干性超前代；**推理时搜索（inference-time search）引导可控生成表观基因组结构**（指定染色质可及区位置/长度，甚至把 Morse 码写进表观设计）——**生物领域首个 inference-time scaling 结果**。全开放（权重/训练码/推理码/数据）。
- **局限**：真核病毒域被刻意排除→该域 perplexity 高（设计性盲区）；长程生成与可控性仍是开放问题；推理时搜索依赖外部 scoring 函数质量。

### State: Predicting cellular responses to perturbation across diverse contexts（biorxiv 2025.06.26.661135）
- **问题**：扰动响应预测的核心难点是**泛化到未观测的细胞情景**；深度模型常**打不过线性基线**。两大噪声源掩盖真信号：基线群体内部的生物异质性 H(D_basal)（scRNA-seq 测量即破坏细胞，无法观测扰动前状态）+ 跨数据集的技术变异 ε。
- **方法/机制**：**State = 多尺度双模块**。**ST（State Transition）**：transformer，对**成组细胞集合（set，最优 256/组）**做双向自注意力，预测扰动后转录组分布，用 **MMD（最大均值差异）损失**对齐预测与观测的扰动细胞分布；训练于 **100M+ 扰动细胞、70 个细胞情景**。**SE（State Embedding）**：dense 双向 transformer 编码器 + 小 MLP 解码器，**双轴损失**（细胞内跨基因 + minibatch 内每基因跨细胞）学习对技术变异稳健、对扰动敏感的细胞嵌入；训练于 **167M 观测性单细胞**。ST 可直接吃原始 HVG 表达，**或**吃 SE 嵌入——后者下，**ST 在 SE 潜空间预测扰动引起的嵌入偏移，再由学到的 MLP 解码回表达空间**。理论上证明 State 在渐近极限下**泛化最优传输（OT）映射**。
- **关键结果（PDF 内）**：扰动效应判别 **+50%**、真差异表达基因识别 **2× 准确率**。分数据集：Tahoe-100M 判别 **+54%**、表达变化 Pearson **+63%**；Parse-PBMC **+29% / +47%**；遗传扰动数据集 AUPRC 比次优高 **184%**。**首个在跨情景泛化上稳定打过简单线性基线的模型**。SE 嵌入支持**零样本**：ST 在 Tahoe 上预训练后，无需在 query 情景训练即可预测该情景扰动（zero-shot）。配套 **Cell-Eval** 生物可解释评估框架。
- **局限 / 关键纪律**：**主力 benchmark（§2.2）用的是 ST+HVG（原始基因表达，完全不经 SE）**——即最强数字并不来自"冻结基底"；SE 专用于低数据/新情景 zero-shot。全流程是**离线 SSL + 监督**，无任何在线 PE 信号。

## 2. 与 VZ 的关系（三视角）

> **本 lab 重心在 §2.2 反证**：生物基础模型最容易触发两个确证陷阱——把训练损失读成 R-PE、把 State 读成干净 R2。先反证、后确证。

### 2.1 确证（先进性背书）

- **R2（强，跨模态独立验证 · 本 lab 最硬背书）**：Evo / Evo2 是"**冻结大基底 + 零样本下游 readout**"在**完全非语言模态**上的物理证据——同一个不动权重的基因组基底，零样本同时做蛋白适应度、ncRNA、调控活性、基因必需性、变体致病性。Evo2 的 BRCA1 分类更是**在冻结 embedding 上加监督头**达 SOTA = 教科书级的"frozen substrate + 任务头"。这证明 R2 不是 NLP 工程技巧而是**跨模态结构规律** → [`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md)。
- **R3/R4（中强，跨领域交叉验证）**：Evo 的 StripedHyena/hyena（data-controlled 卷积，SSM/深度信号处理家族）在 scaling laws 上**优于 Transformer++ 且更稳定**，把"在 token 输出之下演化的紧凑递归/卷积 latent"扩到 1M 上下文——这是 **Cartesia"有界隐状态可扩展"主张的独立跨领域交叉验证**。State 进一步：ST 在 **SE 潜空间**预测扰动偏移、再解码——**控制/预测发生在嵌入空间而非原始 token/表达空间** → [`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md)。
- **R7 + 世界模型（中，概念背书 · 借鉴主源）**：State 把"给定情景 + 干预（扰动）→ 预测响应分布"形式化为一个**软世界模型**，结构上对应 VZ 的 **World 轨**"给定上下文 + 行动 → 预测响应"。这是 §2.3 的主要借鉴来源 → [`dual-track-learning.md`](../../../docs/specs/dual-track-learning.md)、[`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md)。
- **R12（中，方法背书）**：Cell-Eval 是一套**只读、生物可解释、覆盖多维度（表达/DE/效应量）的评估框架**，且与扰动 mean 基线对比以区分"泛化 vs 记忆"——印证"评估应覆盖意义、且不是学习源" → [`evaluation.md`](../../../docs/specs/evaluation.md)。

### 2.2 反证（红队）

**反证 A（headline · 最易踩的确证陷阱）：基因组/细胞模型有清晰的"预测下一 token / 对齐分布"的损失，是否就印证了 R-PE（prediction error 一级信号）？**
诚实地说：**不**。Evo/Evo2 是 next-token SSL，State 是 MMD 分布对齐 + 监督——它们都是**离线训练目标**，在数据集上一次性最小化经验损失，**没有运行时、没有在线 PE 作为一级控制信号、没有 PE 的下游 readout（needs/homeostasis/credit）**。把训练 perplexity 当成 R-PE，就是把"优化目标"误读成"运行时认知信号"。
- **裁决：survives（目标域不同）+ needs-boundary-condition（必须在 spec 写明区分）**。R-PE 的主张是"**运行时**的 PE 是一级原始信号，evaluation/credit 是其 readout"，针对的是 always-on 数字生命的认知循环；基因组模型的训练损失属于 rare-heavy 离线层，二者不是同一对象。
- **边界（写入 spec）**：在 `prediction-error-loop.md` 明确"训练损失（offline SSL 目标）≠ 运行时 PE（一级信号）"——可以借鉴 State 的**预测-观测差**作为 PE 的**形式**，但 PE 必须活在运行时、可被 needs/credit 读取，不能退化为一个训练 loss。

**反证 B：State 被概括为"稳定 ST 基底 + SE"，是否是一个干净的 R2（冻结基底 + 自适应控制器）样本？**
诚实地说：**不干净，且现有 99 综合的措辞需要修正**。PDF 事实：(1) **主力 benchmark 用 ST+HVG，根本不经过 SE**——最强数字里**没有任何"冻结基底"**；(2) SE 才是"在 167M 观测数据上预训练、对 ST 提供输入表征"的那个更像"冻结编码器"的部件，ST 是在其上学扰动迁移的"头/控制器"——即**SE 是基底、ST 是控制器**，与"ST 基底 + SE"的措辞相反；(3) 全流程离线，**根本不涉及"在线端到端梯度是否打到基底"这个 R2 关心的问题**。
- **裁决：needs-boundary-condition**。R2 的"冻结基底"由 **Evo/Evo2 干净地背书**（基因组基底零样本 readout，权重不动），但 **State 是一个弱/含糊的 R2 例子**，不应被当作干净 R2 引用。
- **边界（写入 spec / 修订 99）**：把 State 的 R2 角色降级为"**SE（观测预训练编码器）作为相对稳定的表征基底 + ST/decoder 作为在潜空间学习的扰动头**"这一**配置之一**（仅在低数据 zero-shot 用），而非通用主张；干净 R2 证据以 Evo/Evo2 为准。

**反证 C：Evo 的 hyena/SSM 有界隐状态用 O(1)/紧凑状态压缩了 1M token 全历史——是否说明 VZ 显式分层记忆（R5/R6）多余？**（与 Cartesia 同源反例）
- **裁决：needs-boundary-condition（与 Cartesia 裁决一致），且 Evo 内部提供了反向证据**。关键观察：**Evo / Evo2 都不是纯有界状态，而是 hyena/卷积 + 注意力的 hybrid——注意力层正是为"从上下文精确召回"而加**；Evo2 的 StripedHyena 2 还专门做了 1M needle-in-haystack 召回评估。这说明**纯有界递归压缩不足以支撑精确召回，必须配显式召回通道**——这恰恰**支持** VZ"压缩 stratum 之外仍需可寻址召回"的立场。
- **边界**：有界状态适合做瞬态/情景压缩底层组件；但精确召回需要显式通道（注意力/可寻址记忆），且状态不可命名/不可发布快照 → 不能替代 9 类语义 owner（R11/R8）→ [`continuum-memory.md`](../../../docs/specs/continuum-memory.md)、[`semantic-state-owners.md`](../../../docs/specs/semantic-state-owners.md)。

**反证 D：Evo2 SAE 揭示模型"自主涌现"出 exon/TF/结构等特征，是否说明结构应让其涌现、VZ 显式命名内部状态（R11）是过度设计？**
- **裁决：survives**。涌现发生在**静态生物序列的自监督表征**里，是被动可解释性发现；VZ 的 R11 针对的是**运行时、跨模块、需被消费者读取并据以协作的契约状态**（commitment/boundary_consent 等）——这些必须可命名、可发布快照、fail-loudly，不能靠下游探针事后解读。两者目标域不同，涌现不否定运行时契约需求。

### 2.3 局部算法借鉴（算法级解耦）

| # | 机制（剥离叙事） | 目标 VZ spec | 落地动作 | 预期收益 | 风险 / 前提 |
|---|---|---|---|---|---|
| 1 | **State 的"扰动→响应"软世界模型**：给定基线群体 + 干预标签，在**潜嵌入空间**预测响应的**分布偏移**（set-based 自注意力建模异质性，MMD 对齐分布），再解码 | [`dual-track-learning.md`](../../../docs/specs/dual-track-learning.md)、[`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md) | World 轨建成"给定上下文 + 行动/干预 → 预测对方/环境的**潜空间响应分布**"的预测器；**PE = 预测潜偏移与实际观测潜偏移的散度**（借鉴 MMD 形式做分布级而非点估计）；预测在 z 空间、由 World 轨 owner 发布快照 | 给 World 轨一个有原则的**分布级潜空间世界模型** + 把 PE 接成运行时 readout；分布预测比点预测更稳健于异质性 | PE 必须是**运行时一级信号**而非训练 loss（见反证 A）；World/Self 双轨严格隔离（R7），不得互读快照；潜空间预测不溢出为 token 空间长期策略（R4） |
| 2 | **StripedHyena 的"有界卷积/递归压缩主体 + 稀疏注意力召回"hybrid 配方**（Evo: 90% hyena + 10% attention；Evo2 多算子 + 1M 召回验证） | [`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md)、[`continuum-memory.md`](../../../docs/specs/continuum-memory.md) | 瞬态→情景层采用"**有界递归/卷积压缩器承担主体长程压缩 + 一薄层显式可寻址召回（注意力/检索）**"的双件设计；压缩器输出汇总进记忆 owner 快照，召回层服务于可命名 owner | 跨领域验证的高效长程压缩 + 明确"召回需独立显式通道"的工程证据，避免把召回也压进不可寻址状态 | 有界状态不可命名/寻址 → 必须包成 owner 管辖的底层组件（R8/R11）；召回层不得静默成为记忆第二所有者 |
| 3 | **Evo2 inference-time search 引导生成**：用外部 scoring 模型在**推理时搜索**引导冻结基底的生成（可控表观结构），**不更新基底权重** | [`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md)、[`affordance.md`](../../../docs/specs/affordance.md) | 把"外部 scorer 在推理时引导冻结基底"作为 VZ 的**有界控制模式**：控制决策落在 z_t/引导信号空间，由控制器层施加，**基底始终冻结**；可作 R2"冻结基底 + 自适应控制器"的具体实现样式 | 在不对基底做梯度的前提下获得可控行为（干净 R2）；inference-time scaling 提供"用算力换可控性"的旋钮 | scorer/引导器是控制器，**基底必须保持冻结**（R2，禁在线端到端梯度）；引导决策须可发布为快照 readout，不得变成隐式 owner |

## 3. 一句话定位

Arc Institute（Evo / Evo2 / State）是 VZ **R2 最硬的跨模态背书**（冻结基因组基底 + 零样本 readout，非语言模态独立证明）与 **World 轨潜空间世界模型的概念/算法主源**（State 扰动→响应分布预测）；同时是两个确证陷阱的清醒剂——**训练 SSL 损失 ≠ 运行时 R-PE**（裁决 survives + 写边界）、**State 不是干净的冻结基底样本**（裁决 needs-boundary-condition，需修订 99 措辞）；Evo 的 hyena+attention hybrid 反而为"压缩之外仍需显式召回"提供了支持 VZ 的反向证据。

## 附：本地论文清单（同目录 PDF）
- `evo-sequence-modeling-design-molecular-to-genome-scale-biorxiv-2024.02.27.582234.pdf` — Evo 1（7B StripedHyena，原核基因组，2024）
- `evo2-genome-modeling-design-across-all-domains-of-life-biorxiv-2025.02.18.638918.pdf` — Evo 2（7B/40B StripedHyena 2，9.3T bp 全生命域，1M 上下文，2025）
- `state-predicting-cellular-responses-to-perturbation-biorxiv-2025.06.26.661135.pdf` — State（虚拟细胞，ST + SE 扰动响应预测，2025）
