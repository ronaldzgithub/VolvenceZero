# CZI Virtual Cell（Chan Zuckerberg Initiative）— 深度分析

- **分组 / 成熟度**：C 生物基础模型（虚拟细胞 + 推理 LLM）｜ 成熟度中-高（开放预印 + 开放权重/代码，非营利；rBio 仅 proof-of-concept）
- **一句话主张**：构建"AI 虚拟细胞"——以**冻结的生成式跨物种细胞基础模型**（TranscriptFormer）为世界模型/虚拟仪器做 in-silico 生物学；并提出 **rBio：在没有硬标签可验证的开放域里，用冻结世界模型当"软验证器（soft verifier）"为推理控制器提供 RL 奖励**，以模拟（simulation）而非实验数据训练。
- **主要创作者 + 血统**：Theofanis Karaletsos（AI 负责人，三篇通讯）、Stephen R. Quake（科学负责人）、Ana-Maria Istrate（rBio 一作 / TF 作者）、Charlotte Bunne、Yusuf Roohani（AIVC/State）、James D. Pearce（TF 一作）。血统横跨单细胞基因组学 + 概率生成模型 + 推理 LLM 后训练（GRPO/DeepSeek-R1 一脉）。
- **为何与 VZ 共振 / 对立**：**这是整个 33-lab roster 里与 VZ 核心难题最同构的一家**。VZ 的根本困境是"关系/EQ/regime 质量没有可验证标量奖励"；rBio 恰恰解决了同构问题——"基因扰动结果在实验室无法穷尽验证 → 没有硬标量奖励 → 用冻结世界模型当软验证器提供奖励"。**但本分析以反证为先**：rBio 把奖励来源放在**外部模型**上、且在 **token 空间做 GRPO**，这两点恰好踩在 VZ 的 R-PE-不外包 与 R4-不在-token-空间-做-RL 两条红线上。机制可借鉴，整包不可吞。

## 1. 核心逻辑（论文级 · PDF-grounded）

### rbio-1: training scientific reasoning LLMs with biological world models as soft verifiers（biorxiv 2025.08.18.670981）

- **问题**：推理模型通常在**可形式化验证**的系统里训练（代码能否运行、数学是否成立）。但生物学是开放域，没有可大规模 scale 的精确规则；要验证一个假设（如"敲低 gene A 是否导致 gene B 差异表达"）只能做湿实验——慢、贵、不随算力 scale。能否**不依赖额外实验数据**地训练生物推理模型？
- **方法/机制**：把生物世界模型当作"近似 oracle"提供**软验证**。
  - **基座 + 算法**：base = `Qwen2.5-3B-Instruct`，用 **GRPO**（Group Relative Policy Optimization）做 RL 后训练。对每个 prompt，LLM 生成 N 个 completion，每个由验证器打分得 reward $r_i$；reward 经**组内归一化**成 advantage $\hat A_i = (r_i-\mathrm{mean}(r))/\mathrm{std}(r)$，配 clipped surrogate + 对参考模型的 KL 罚（$\beta$）。
  - **三类验证器（核心创新）**：
    1. **Hard verification（rBio-Hard）**：有实验数据 $D_{EXP}$ 时，二值 reward $r\in\{0,1\}$（预测与实验一致给 1）。
    2. **Soft verification via models（rBio-VCM）**：无实验数据时，用**冻结生物模型** $M_{bio}\in\{$MLP, TranscriptFormer$\}$ 输出**概率/连续分数**当 reward，$r_{soft}=p(q,o_i|M_{bio})\in[0,1]$。例：MLP 对扰动预测的概率；TranscriptFormer 的**点互信息 PMI**（基因共表达强度）。
    3. **Soft verification via prior（rBio-Prior）**：用知识库 GO（Gene Ontology）注释打分——ROUGE（n-gram/LCS 重叠）、keywords 命中率、likelihood（注释在自身推理模型下的对数似然）。模型被要求把所用基因信息放进 `<gene_info>` 标签以便比对。
  - **归一化**：多数软验证器分数不在 $[0,1]$，用带阈值的 min-max 归一化（阈值以下映射到 $[0,0.5]$，以上映射到 $[0.5,1]$；MLP 阈值取 0.5），把"显著/不显著"压成可用 reward。
  - **可组合验证（composable verification）**：多验证器奖励**相加** $r_i=\sum_j r_{i,j}(q,o_i|V_j)$；训练时按各来源数据密度采样 prompt。
  - **训练配置**：100k steps、8×H100、约 10 天；batch_size=4、n_generation=4、lr=5e-6；推理 N=5 代、temperature=0.7。代码/权重开放（czi-ai/rbio）。
- **关键结果（PDF 内具体数字）**：
  - **软验证 ≈ 硬验证**：`rbio-MLP-leave-one-out`（用 OOD 细胞系训练的 MLP 预测当奖励，**完全不碰实验数据**）F1 0.65/0.66 vs `rbio-EXP-leave-one-out` 0.67；MCC 0.60 vs 0.61；且 **Balanced Accuracy/TPR 反超**（TPR 0.77/0.76 vs EXP 0.68 vs SUMMER 0.63）。
  - **跨任务迁移**：`rbio-TF`（只用 TranscriptFormer 共表达 PMI 训练，**prompt 与扰动任务无关**）迁移到扰动预测：MCC 0.21 vs base 0.03、Balanced Acc 0.59 vs 0.52、TNR 0.94 vs 0.55——证明世界模型可把 off-task 生物知识蒸馏进推理控制器。
  - **可组合性**：`rbio-TF+GO+MLP` 全面碾压 `rbio-TF`（F1 0.29→0.68、MCC 0.21→0.64），逼近纯实验数据模型；每加一个来源大体单调提升。
  - **测试期 CoT**：链式思维把 `rbio-TF+GO+MLP` 推到 F1 0.74、Balanced Acc 0.83，**在 PerturbQA 上超过 SOTA 的 SUMMER**，且无需工具/实验数据。
  - 所有推理模型均超过专用扰动模型 GEARS 和 base Qwen2.5-3B。
- **局限（PDF 内 + 红队视角）**：(1) **奖励来源全在外部模型/知识库**——GO 在某些组合里**反而掉点**（"adding GO seems to hurt"，作者自陈待查），即一个失准的验证器会静默污染学习；(2) 训练对象是**整个 3B LLM 的端到端 GRPO**，非"冻结基底 + 有界 adapter"；(3) **token 空间 RL**（GRPO 作用在 token 级策略 + CoT trace）；(4) 软验证引入更多 false positive（TPR 高但 F1 略逊 EXP）；(5) 质性幻觉分析仅 N=5，作者明确告诫勿过度解读。

### TranscriptFormer: a cross-species generative cell atlas（biorxiv 2025.04.25.650731）

- **问题**：单细胞转录组难以跨巨大进化距离整合；需要一个跨物种、可当"虚拟仪器"的生成式细胞基础模型。
- **方法/机制**：**自回归生成模型**，把每个细胞建成"cell sentence"（基因 token + 表达计数，随机排序），联合建模基因与计数；创新点：gene/transcript head 耦合、**expression-aware 多头自注意力**、causal masking、计数似然；用 **ESM-2 蛋白嵌入**表示基因 token，使跨物种同源基因落到共享 species-agnostic 空间。**纯 log-likelihood 预训练，不给 cell type/species 标签**（仅 assay technology）。三档：TF-Metazoa（444M 参数 / 112M 细胞 / 12 物种 / 跨 1.53 亿…实为 15.3 亿年进化）、TF-Exemplar、TF-Sapiens。
- **关键结果**：零样本细胞类型分类 SOTA（Tabula Sapiens 2.0 macro-F1 0.910；OOD 物种平均 F1 0.778，跨 6.85 亿年进化距离仍 >0.65）；零样本疾病态识别（SARS-CoV-2 F1 0.859）；**Contextualized Gene Embeddings 零样本涌现** cell type/tissue/donor 结构（cell type 解释 >95% PC1/PC2 方差）；**作为虚拟仪器 via prompting**：用 PMI 预测 TF-基因调控关系，与 STRING v12.0 交叉验证（如 E2F8 预测 227 靶点、87 已知），并条件化 marker 基因生成细胞类型特异 TF——**生成式模型即可查询的世界模型，而非查找表**。
- **局限**：仍是离线 SSL，无运行时；生成相干性/可控性是开放问题；donor 与 batch 效应难完全分离。

### How to build the virtual cell with AI: priorities and opportunities（arXiv:2409.11654 / Cell 2024）

- **问题/主张**：传统虚拟细胞是规则/方程/agent 模型，无法处理多尺度、海量交互组件、强非线性。提出 **AI Virtual Cell（AIVC）** 愿景。
- **核心概念（对 VZ 高度可迁移）**：
  - **Universal Representations（UR）**：跨模态/物种/尺度（分子→细胞→多细胞）的**模态不变共享表征**，作为可泛化到未观测状态的"参考"。
  - **Virtual Instruments（VI）**：在 UR 上操作的 **Manipulator**（施加扰动：unperturbed UR → perturbed UR）+ **Decoder**（读出表型变化）——即 **in-silico 反事实实验**。
  - **LLM 作为可解释/交流中间层**：把生物模型的结果翻译成人类可读对话（rBio 正是这一层的落地）。
  - **评估要超越窄 benchmark**：聚焦"核心能力"而非单点指标；强调隐私/可信。
- **局限**：愿景论文，无第一方实验结果；具体机制由 TranscriptFormer/State/rBio 等后续工作落地。

## 2. 与 VZ 的关系（三视角）

> **本 lab 重心在 §2.3 局部算法借鉴**：rBio 的"软验证器 RL"是整个 roster 里**唯一**直接对应 VZ"关系质量无硬奖励"这一根本难题的机制。但**先反证**：rBio 的奖励来源与优化空间都踩在 VZ 红线上，必须精确切出可借鉴的内核，剥掉不可吞的外壳。

### 2.1 确证（先进性背书）

- **R-PE（强，但有限定 · 本 lab 最独特贡献）**：rBio 证明——**当没有可验证硬标量时，可以用一个预测模型的输出当作分级学习信号**。软验证器发出的是 $[0,1]$ 概率（而非二值），本质是"预测/置信度"作为奖励，这正是 R-PE"用预测（误差）当一级信号"的同构形态。在生物域被**定量验证**（软验证 ≈ 硬验证：F1 0.66 vs 0.67），为 VZ"用预测当学习信号"提供了**跨领域的可行性证据** → [`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md)。
- **R2（强 · 验证器/基底侧干净）**：所有验证器（MLP、TranscriptFormer、ESM、GO）在 rBio 训练中**全程冻结、仅推理出 reward，从不更新**；TranscriptFormer 本身就是冻结的生成基底、零样本当虚拟仪器。这是"冻结大基底当 oracle、控制器在其上学"在**非语言模态**上的干净样本 → [`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md)。（**注意**：仅"验证器/基底侧"干净；被训练的策略侧不干净，见反证 B。）
- **R7 + 世界模型（中强 · 借鉴主源）**：TranscriptFormer = 生成式**世界模型**；AIVC 的 Virtual Instrument（Manipulator 施扰动 + Decoder 读表型）= **在 UR 上做反事实 in-silico 实验**，结构上对应 VZ World 轨"给定上下文 + 行动 → 预测响应"。可组合验证器 = 多个世界模型互相交叉校验 → [`dual-track-learning.md`](../../../docs/specs/dual-track-learning.md)。
- **R8/R11（中 · 概念背书）**：AIVC 的 **Universal Representation**（模态不变、可发布、作为跨模块"参考"的共享表征）是 VZ"可命名、可发布快照状态"的概念近邻；"LLM 作为可解释中间层"印证 R4——**语言是表达/交流层，不是控制层** → [`semantic-state-owners.md`](../../../docs/specs/semantic-state-owners.md)、[`contract-runtime.md`](../../../docs/specs/contract-runtime.md)。
- **R12（中 · 方法背书 + 反向警示）**：AIVC 明确要求"评估超越窄 benchmark、覆盖核心能力"；而 rBio 的 **"GO 反而掉点"** 是一条天然的只读评估证据——**一个失准的奖励/验证器会被评估抓到**，印证"评估应只读、能暴露学习源问题" → [`evaluation.md`](../../../docs/specs/evaluation.md)、[`evidence_program.md`](../../../docs/specs/evidence_program.md)。

### 2.2 反证（红队）

**反证 A（headline · 直击 R-PE）：rBio 用 LLM/世界模型当验证器，这是否违反 R-PE"PE 不外包给一个其任务表示可能漂移的外部模型"？**

诚实地说：**rBio 本身就是 R-PE 警告的那个风险的现身**，不是它的反例。rBio 的奖励来源**完全在外部模型/知识库**（MLP/TF/GO），推理 LLM 的全部学习信号 = 验证器输出；这里的"PE"不是系统对自身世界的内禀预测，而是**外部模型给的代理标签**。证据就在 PDF 内：**GO 验证器在某些组合里反而掉点**（作者自陈"seems to hurt … need to look into further"）——这正是"任务表示失准的外部源静默污染下游"的 R-PE 预言。
- **裁决：needs-boundary-condition（机制可借，但条件极严）+ 若照搬则 genuine-risk**。
- **边界（写入 spec）**：软验证器**何时可接受 vs 何时变成不可问责的第二 PE 源**——
  - ✅ 可接受：验证器是**系统自身的 world/self 预测基底**（PE 仍内禀，不是第三方任务模型）；**冻结 + 版本化 + 可审计**；**多验证器组合交叉校验**，无单一源独占奖励所有权；信用分配在下游、只读。
  - ❌ 不可接受（= 第二 PE 源）：验证器是**目标各异、不透明、会漂移的外部模型**，静默成为事实上的奖励所有者——一旦其任务表示偏移，所有下游一起偏且无法检测（GO-掉点即此类的轻症）。
  - **对 VZ 的关键改写**：rBio"验证器 = 外部模型"是**不该抄的部分**；"用冻结预测模型在无硬标量处发软概率奖励"才是该抄的内核。VZ 必须把验证器实例化为**自己的双轨 World/Self 预测模型**，让 PE 保持内禀 → R-PE 不破。

**反证 B：世界模型是否冻结（R2 是否干净）？**

**验证器侧干净，策略侧不干净。** 验证器（MLP/TF/ESM/GO）确实全程冻结仅推理——R2-clean。但**被训练的对象是整个 `Qwen2.5-3B-Instruct` 的端到端 GRPO**（100k steps 全模型 RL），**不是**"冻结基底 + 有界 adapter 控制器"。从 VZ 的 R2 视角，rBio 在策略侧根本没做冻结基底/有界控制器的切分。
- **裁决：survives（"冻结世界模型当验证器/oracle"这一主张干净成立）+ needs-boundary-condition（rBio 训练的是全模型，非有界控制器）**。
- **边界**：VZ 借鉴时，**substrate 必须冻结，只训练控制器层（z_t 空间的有界 adapter-delta）**，奖励来自冻结世界模型；不得复制 rBio 的全模型 RL。

**反证 C：rBio 是否在 token 空间做 RL（vs R4）？**

**是，且毫不含糊。** GRPO 作用在 token 级策略 $\pi_\theta(o_{i,t}|q,o_{i<t})$，奖励打在答案/格式上，推理 trace 是 token、测试期还叠 CoT——这是教科书式的 **token 空间 RL**，正是 R4 禁止用于长程控制的范式。
- **裁决：照搬则 genuine-risk（违反 R4）；可借鉴的内核是奖励构造（软验证器），不是 token 空间优化本身**。
- **边界**：把奖励语义（冻结预测模型 → 软概率 PE 奖励 + 组内归一化 advantage + 可组合验证器）搬到**控制器（z_t）空间的 RL**，而非 token 策略。VZ 的"长期策略学习在控制器代码 z_t 空间、token 是表达层"必须守住。

**反证 D：rBio 用"模拟代替实验数据"训练，是否印证 VZ 可以用模拟/自评估闭环替代真实信号？**
- **裁决：needs-boundary-condition**。rBio 的模拟之所以成立，是因为底层世界模型**确实在大规模真实实验数据上预训练过**（TF 训练于 112M 真实细胞、ESM 于真实蛋白）——模拟是"把已编码的真实知识 walk 出来"，不是凭空生成。对 VZ：软验证器只有在其**底层预测基底见过足够真实关系/交互数据**时才可信；否则就是自我循环幻觉。**评估必须先做硬**（R12），用只读 eval 持续检测验证器是否漂移（呼应 GO-掉点）。

### 2.3 局部算法借鉴（算法级解耦）

> **本节是 czi 分析的中心**：soft-verifier-RL 是整个 roster 里对 VZ"关系/EQ/regime 质量无硬标量奖励"最直接可迁移的机制。剥离生物叙事后，把"外部验证器"改造为"VZ 自身双轨预测基底"，把"token 空间 GRPO"改造为"z_t 空间控制器学习"。

| # | 机制（剥离叙事） | 目标 VZ spec | 落地动作 | 预期收益 | 风险 / 前提 |
|---|---|---|---|---|---|
| **1（最高优先）** | **软验证器 RL（rBio-VCM）**：无硬标量时，用**冻结预测世界模型**对控制器生成的候选行为发**软概率奖励** $r=p(\cdot|M)\in[0,1]$（带阈值 min-max 归一化），经**组内归一化** advantage 做策略学习 | [`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md)、[`credit-and-self-modification.md`](../../../docs/specs/credit-and-self-modification.md) | 把 VZ 自己的 **World/Self 双轨预测模型**当软验证器：对一个候选关系行为，World 轨预测对方/环境响应、Self 轨预测自身状态轨迹 → **PE = 预测响应与实际观测响应的分级散度**当 reward；在**控制器 z_t 空间**做有界 RL（GRPO 式组内归一化 advantage），而非外部 scalar | **给"关系质量无法直接打分"这一根本难题一个已被生物域定量验证的原则化解法**；软概率信号比二值更稳健、可分级 | **验证器必须是 VZ 自身内禀预测基底**（否则违反 R-PE 不外包，见反证 A）；**substrate 冻结、只训控制器层**（R2，反证 B）；**优化在 z_t 不在 token**（R4，反证 C）；信用分配下游、只读 |
| **2** | **可组合 + 密度加权验证器**：多个冻结验证器奖励**相加**，多样性单调提升泛化；但单一失准源（GO）会掉点 | [`credit-and-self-modification.md`](../../../docs/specs/credit-and-self-modification.md)、[`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md) | 用**多个内禀预测器组合**当验证器集成（World 快/中/慢预测 + Self 轨），按可靠性加权；**逐源监控贡献**——某源持续拉低 eval 即视为漂移/失准源（GO-掉点的检测器），可回滚该源权重 | **结构性防御单验证器漂移**（直接化解反证 A 的第二-PE-源风险）；no single source 独占奖励所有权 | 逐源归因必须**可审计、可回滚**（R15）；组合不得让任一外部源静默成为隐式 owner；World/Self 不得互读（R7） |
| **3** | **冻结生成式世界模型当"虚拟仪器" via prompting**（TranscriptFormer：自回归生成模型零样本查询 PMI；AIVC：Manipulator 施扰动 + Decoder 读表型 = 反事实 in-silico 实验） | [`dual-track-learning.md`](../../../docs/specs/dual-track-learning.md)、[`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md) | VZ World 轨建成**冻结生成式预测器**，在 z 空间被查询"给定上下文 + 拟施加的行动/干预 → 预测分级（概率）关系响应"；这正是机制 1 的软验证器**底层基底**；AIVC 的"unperturbed UR → perturbed UR → readout"= VZ 反事实 rollout 的范式 | 给软验证器一个**有原则的冻结世界-模型基底**；反事实 in-silico 评估候选行为而不真去试错 | World/Self 严格隔离（R7）；查询在潜空间、不外溢成 token 空间长期策略（R4）；世界模型须在足量真实交互数据上预训练才可信（反证 D） |

补充（中优先，记入路线）：AIVC 的 **Universal Representation**（模态不变共享表征）可作为 R8/R11"可发布快照/共享契约表征"的设计论据；**LLM 作为可解释中间层**支撑 R4"语言是表达层、控制在其下"——VZ 的叙事生成层与 z_t 控制层应严格解耦（与机制 1/3 同一纪律）。

## 3. 一句话定位

CZI Virtual Cell 是 VZ **"无硬奖励下用软验证器学习"这一根本难题的唯一直接镜像**：rBio 在生物域定量证明了"冻结世界模型发软概率奖励 ≈ 硬标签奖励"，TranscriptFormer/AIVC 提供了冻结生成式世界模型与反事实虚拟仪器的范式；但 rBio 把奖励**外包给外部模型**且在 **token 空间做 GRPO**，恰好踩在 R-PE-不外包 与 R4-不在-token-空间-RL 两条红线上——**正确的借鉴是把验证器换成 VZ 自身双轨预测基底、把优化挪到 z_t 控制器空间、用可组合验证器防漂移**，从而把 rBio 的内核安全地搬进 VZ 解决"关系质量无法打分"的问题。

## 附：本地论文清单（同目录 PDF）

- `rbio-1-reasoning-llms-with-biological-world-models-as-verifiers-biorxiv-2025.08.18.670981.pdf` — rBio-1（Qwen2.5-3B + GRPO，软验证器 RL：MLP/TranscriptFormer/GO 当冻结验证器，PerturbQA SOTA，2025）
- `transcriptformer-cross-species-generative-cell-atlas-biorxiv-2025.04.25.650731.pdf` — TranscriptFormer（444M/542M/368M 自回归生成基底，112M 细胞/12 物种/15.3 亿年进化，零样本虚拟仪器 PMI，2025）
- `how-to-build-the-virtual-cell-with-ai-priorities-opportunities-2409.11654.pdf` — AIVC 愿景（Universal Representations + Virtual Instruments + LLM 可解释层 + 超越窄 benchmark 的评估，Cell 2024）
