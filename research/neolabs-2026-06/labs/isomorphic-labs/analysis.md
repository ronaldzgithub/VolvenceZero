# Isomorphic Labs — 深度分析

- **分组 / 成熟度 / 一句话主张**：C 生物基础模型（结构预测 / 药物设计）｜ 高（AlphaFold2 *Nature* 2021、AlphaFold3 *Nature* 2024，本目录 2 篇核心 PDF 均已核验）｜ 用**单一深度学习框架**从序列（+MSA/模板）端到端预测蛋白乃至全生物分子复合物（蛋白/核酸/小分子/离子/修饰残基）的联合三维结构，AF3 将结构模块换为扩散解码器统一化学空间，服务理性药物设计。
- **主要创作者 + 血统**：John Jumper、Demis Hassabis、Richard Evans、Alexander Pritzel、Olaf Ronneberger（AF2 核心）；Josh Abramson、Jonas Adler、Max Jaderberg、Sergei Yakneen（AF3，Isomorphic 署名）。血统出自 **DeepMind → Isomorphic Labs**（AF3 由 Google DeepMind + Isomorphic Labs 联合署名）。**与 VZ 的关系定性（重要）**：AlphaFold 是**监督式端到端训练**的结构预测器，**不是**"冻结基底 + 自适应控制器"、**不是** PE 驱动的在线学习、**没有**关系/双轨——因此它对 R2 / R-PE / R7 是**弱证据甚至反例**。其真实价值在三个**局部机制**：(a) recycling（循环再喂）这种 latent 空间迭代精修计算模式；(b) AF3 在约束下的扩散式生成 + 多样本排序；(c) pLDDT/PAE/PDE 这类**可命名、已校准、可发布的内部不确定性**（R11/R12）。本分析**先反证、不夸大收敛**。

## 1. 核心逻辑（论文级 · PDF-grounded）

### 1.1 AlphaFold2（*Nature* 596:583, 2021 / doi:10.1038/s41586-021-03819-2）

- **问题**：仅凭氨基酸序列预测蛋白三维结构（"蛋白折叠问题"，50 年未解）；既有物理法/进化法在无近源同源结构时远低于实验精度。
- **方法/机制**：两段式端到端网络。**①Evoformer 主干（48 blocks，无共享权重）**：把结构预测看作 3D 空间图推理；同时维护 **MSA 表示 (s,r,c)** 与 **pair 表示 (r,r,c)**，二者每块互相通信——MSA→pair 走 outer-product-mean，pair→MSA 走带 pair-bias 的行注意力；pair 内部用受三角不等式启发的 **triangle multiplicative update + triangle self-attention**（围绕起点/终点节点）。**②结构模块（8 blocks，共享权重）**：把每个残基表示为全局坐标系下独立的旋转+平移（"residue gas"），用 **IPA（不变点注意力）** 在不改 3D 位置的前提下更新单序列表示、再等变地更新刚体帧；损失为 **FAPE（帧对齐点误差，clamped L1）**。**③recycling（循环 3 次）**：把整网输出递归喂回同一模块、反复施加最终损失实现迭代精修。训练：PDB 监督 + **noisy-student 自蒸馏**（用已训网络预测约 35 万条序列、筛高置信再从头训）+ **BERT 式 MSA 掩码**联合训练（非预训练）。**置信输出**：小型 per-residue 头产出 **pLDDT**（局部精度自估），pair 表示线性投影产出 **pTM/PAE**。
- **关键结果（带 PDF 数字）**：CASP14（n=87 domains）骨架中位精度 **0.96 Å r.m.s.d.95**（次优方法 2.8 Å）；全原子 **1.5 Å**（次优 3.5 Å）；碳原子直径约 1.4 Å 作参照。近期 PDB 全链中位 **1.46 Å**。**pLDDT 可靠预测真实 lDDT-Cα（最小二乘 lDDT-Cα=0.997×pLDDT−1.17，Pearson r=0.76）**；pTM 估计 TM-score（r=0.85）。消融：去掉 recycling、IPA、triangle/gating、端到端结构梯度均显著掉点（end-to-end 结构梯度对精度关键）。可解释性：冻结主网、为 48 个 Evoformer block 各训一个结构模块，得到 192 步中间结构轨迹——表明网络从早期即形成结构假设并**持续平滑精修**。
- **局限（作者自述）**：① MSA 深度 < ~30 序列时精度骤降（>100 序列后增益很小）；② 对**异型接触**（结构主要由与其他链相互作用决定的桥接域）弱；③ 预测的是 PDB 中"最可能出现"的**静态**结构。

### 1.2 AlphaFold3（*Nature* 630:493, 2024 / doi:10.1038/s41586-024-07487-w）

- **问题**：能否在**单一深度学习框架**内高精度预测含蛋白/核酸/小分子/离子/修饰残基的**通用生物分子复合物**联合结构（而非各类型专用工具拼装）？
- **方法/机制**：保留"主干演化 pair 表示 → 结构模块生成坐标"的总骨架，但大改各部件。**①MSA 大幅去重**：MSA 模块缩到 4 blocks、仅用廉价 pair-weighted averaging，MSA 表示不再保留。**②Pairformer（48 blocks）替代 Evoformer**：只在 pair 表示 + single 表示上运算，所有信息经 pair 表示流动（三角更新/三角注意力沿用 AF2）。**③扩散模块替代结构模块**：直接在**原始原子坐标**上做扩散去噪，**无旋转帧、无等变性处理**；多尺度噪声让低噪声学局部立体化学、高噪声学全局排布，从而**省去立体化学违例损失与扭转角参数化**，天然容纳任意化学组分；推理时从随机噪声循环去噪生成。这是**生成式**过程（产出答案分布）。**④抗幻觉**：生成模型易在无序区"编造"结构，故用 **AF-Multimer v2.3 预测做交叉蒸馏**（教 AF3 在无序区输出伸展 loop）。**⑤置信模块（4 blocks）**：因扩散每步只训单步、无法直接回归误差，引入 **diffusion mini-rollout** 生成整体结构再训练置信头，产出 **pLDDT / PAE / PDE**；训练中主干→扩散之间有**梯度停止（STOP）**。推理用 **5 seeds × 5 diffusion samples**、按置信 + 手性/碰撞罚分排序选最优。
- **关键结果（带 PDF 数字）**：PoseBusters 蛋白-配体（n=428，pocket-aligned RMSD<2Å）**显著超越** AutoDock Vina（即便 AF3 不用任何结构输入；Fisher P=2.27×10⁻¹³）与 RoseTTAFold All-Atom；蛋白-核酸/RNA 超 RoseTTAFold2NA（但未及人工辅助的 AIchemy_RNA2）；蛋白-蛋白 DockQ>0.23 成功率较 AF-Multimer v2.3 提升（Wilcoxon P=1.8×10⁻¹⁸），抗体-抗原尤其改善（P=6.5×10⁻⁵，需从 1000 seeds 排序）；单体 LDDT 提升 P=1.7×10⁻³⁴。置信与精度良好校准（ipTM↔DockQ、pLDDT↔LDDT_to_polymer）。
- **局限（作者自述）**：① **手性违例 4.4%**、偶发原子**碰撞/重叠**（罚分缓解不能消除）；② 扩散引入**幻觉**（无序区编造有序结构，靠交叉蒸馏 + 排序压制）；③ 仍是**静态**预测——多 seed **不能**近似溶液构象系综；apo/holo 的 E3 泛素连接酶都只预测闭合态；④ 高精度需大量采样 + 排序，算力成本高；对浅 MSA 仍弱。

> 小结：AF2→AF3 是同一主线的演进——AF2 立起"MSA+pair 双表示 + 三角几何归纳偏置 + residue-gas/IPA 结构模块 + FAPE + recycling + pLDDT/pTM 置信"；AF3 去 MSA 依赖、Pairformer 化、把结构模块换成**坐标空间扩散生成**，统一全化学空间并升级置信为 pLDDT/PAE/PDE。对 VZ 最硬的可借鉴物是 **recycling 迭代精修**、**校准置信头（可命名不确定性）**、**约束下扩散 + 多样本排序**——而非整体叙事。

## 2. 与 VZ 的关系（三视角 · 先反证后确证）

### 2.2 反证（红队）— 先行

逐条给裁决（survives / needs-boundary-condition / genuine-risk）+ 边界条件：

1. **反例（对 R2，最强）：AlphaFold 是端到端监督训练，没有"冻结基底"**。AF2/AF3 整网（Evoformer/Pairformer 主干 + 结构/扩散模块）在结构任务上**联合端到端训练**；AF2 消融更显示"去掉端到端结构梯度"会大幅掉点，即端到端梯度是精度来源。AF3 进一步**弱化/去除** MSA 与"预训练表示"的角色。**这直接戳破 flat note 与 99 综合把 Isomorphic 标为 R2 ★（"冻结预训练表示 + 任务特定结构模块"）的说法——那是确证偏误下的误读。**
   → **裁决：needs-boundary-condition（且需修正 99 的过度主张）。** **澄清**：AF 的"主干→结构模块"分段是**网络内部的功能模块化**，不是 R2 的"冻结大基底 + 在其上叠加有界自适应控制器"。AF 没有任何成分被冻结后再挂在线适配头。**边界条件**：(i) VZ 引用 AlphaFold 时**不得**将其当作 R2（冻结基底范式）的背书；R2 的跨模态背书应来自真正"冻结 + 头"的样本（ESMFold=冻结 ESM-2+折叠头、BaseFold=冻结 AF2 仅改输入、Chai=冻结 PLM embedding+pair-bias 头），而非 AlphaFold 本体；(ii) 在 spec/综合中把 Isomorphic 在 R2 列降级为"弱/不适用"，把误标纠正记入反证矩阵。

2. **反例（对 R-PE）：AF 是离线监督/去噪回归，无一级运行时 PE 信号**。AF2 用 FAPE 监督，AF3 用扩散去噪损失，均为离线训练目标，运行时不存在"PE 作为一级原始信号驱动感知-行动-学习"的闭环。
   → **裁决：survives（域外/正交，不构成反证）。** AF 解决"从序列到结构的监督映射"，与"PE 是否为一级原始信号"正交，既不支持也不反对 R-PE。**边界条件**：不得把扩散的"去噪即预测误差最小化"误读为 R-PE 的工程背书；R-PE 主线论据仍来自 [`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md)（active inference / ICM）。**但有一处正向迁移**：pLDDT/PAE 是模型对自身输出**误差的学习自估**，这是 R11/R12（可发布不确定性、只读评估）的好对照，详见 §2.1。

3. **反例（对 R7 双轨）：AF 无关系/自我/双轨结构**。AlphaFold 是单目标结构预测器，不含 World/Self 双轨、不涉关系/EQ/养成。
   → **裁决：survives（域外，无承载）。** 不构成对 R7 的反证或支持。**边界条件**：AlphaFold 不进入 R7 相关论据集。

4. **反例（对 R3/R4）：recycling/扩散是迭代计算，但不是"学到的时间抽象/切换控制"**。AF 在 latent 几何/pair 空间做迭代精修（recycling 3 次）与坐标扩散去噪，确实"不在 token 空间"——表面契合 R3/R4。但 recycling 是**固定步数的迭代精修**、扩散是**固定调度的去噪**，二者都**不是**学到的 z_t/β_t 控制器，没有"何时切换/保持"的元决策（β_t）。
   → **裁决：needs-boundary-condition。** recycling/扩散为"有意义计算发生在结构化 latent 空间并被迭代精修"提供了**跨模态的弱-中背书**，但**不能**被当作 R3/R4 的"学到的时间抽象/控制"证据。**边界条件**：在 [`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md) 区分"latent 空间迭代精修（AF 提供）"与"latent 空间**受控**时间抽象 z_t/β_t（VZ 需自建，来源仍是 MuZero/CfC 等）"；借鉴 recycling 仅作**计算模式**，不据此声称 AF 验证了控制器空间决策。

5. **反例/风险（对 R12，genuine-risk 提示）：生成式 latent 组件会在欠定区幻觉**。AF3 自承扩散在无序/低数据区**编造看似合理的结构**，必须靠交叉蒸馏 + 置信排序 + SASA 罚分压制；且静态预测无法反映构象系综、apo/holo 不分。
   → **裁决：genuine-risk（对 VZ 任何引入生成式 latent 组件的设计）。** 若 VZ 在控制器/记忆层引入扩散式或生成式 latent 生成，**同样会在关系/情景的欠定区生成貌似合理但无据的内部状态**。**边界条件/待办**：(i) 任何生成式 latent 组件必须配**校准的置信/误差自估** + **只读评估门控**（R12 [`evaluation.md`](../../../docs/specs/evaluation.md)），评估须**先做硬**；(ii) 借鉴 AF3 的"多样本 + 排序 + 显式罚分（碰撞/手性=VZ 的边界/一致性约束）"作为抗幻觉范式；此条进风险登记。

### 2.1 确证（先进性背书）

> 先经红队后保留的可信确证（均为**非语言/跨模态**独立样本，但强度普遍为弱-中，且**不含 R2/R-PE/R7**）：

- **R11 + R12（中，跨模态独立验证 · 本 lab 最硬确证）**：pLDDT / PAE / PDE / pTM / ipTM 是模型对**自身输出**逐残基/逐对的**误差自估**，且**经校准、可发布、被下游消费用于排序**（pLDDT↔真实 lDDT-Cα，Pearson r=0.76；ipTM↔DockQ 良好对齐）。这是"**把内部状态/不确定性命名并作为可发布契约产物**"（R11 [`semantic-state-owners.md`](../../../docs/specs/semantic-state-owners.md)）与"**只读评估 readout 与生成过程解耦**"（R12 [`evaluation.md`](../../../docs/specs/evaluation.md)）在结构生物学上的干净工程实例。
- **R3/R4（弱-中，带边界）**：结构计算发生在**结构化 latent 空间**（pair/几何/坐标）并被**迭代精修**（recycling 192 步轨迹平滑收敛；扩散多尺度去噪），是"有意义计算不在 token/表达层、而在 latent 空间逐步逼近"的跨模态对照——**但仅为计算模式，非学到的控制**（见 §2.2 反例 4）。
- **R8/R15（弱，可解释/契约取向）**：置信输出作为稳定、可校准、可被下游排序消费的**契约式产物**，弱背书"内部产物以可发布、可解释形式跨边界传递"的取向。

### 2.3 局部算法借鉴（算法级解耦）

剥离"统一结构预测/药物设计"叙事后的可移植机制（机制 → 目标 spec → 落地动作 → 预期收益 → 风险/前提）：

1. **机制：recycling（循环再喂迭代精修）**——把整模块输出递归喂回同一模块、固定 N 次，每次在前一次结果上精修同一 latent 假设（AF2 消融：去掉 recycling 明显掉点，而额外训练开销很小；192 步轨迹显示假设早现并持续平滑精修）。
   → **目标 spec**：[`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md)（R3/R4，z_t）
   → **落地动作**：让控制器在提交 z_t 前跑一个**有界 recycling 循环**——每一遍以上一遍的 latent 状态为条件再精修，固定上限步数，循环全程留在 latent 空间。
   → **预期收益**：以极小额外算力换取"latent 假设逐步收敛"的稳定性与精度增益；天然落在 latent（合 R4，不溢出 token）。
   → **风险/前提**：recycling 是**固定步数迭代精修**，**不是**学到的切换/控制（β_t），不能据此声称实现了时间抽象；步数须有界并设收敛/早停判据（合 no-swallow-errors 的 fail-loudly）。

2. **机制：校准置信头作为可命名、可发布的不确定性**（pLDDT/PAE/PDE：小头从 pair/single 表示预测模型对自身输出的误差，经校准、被下游排序消费；AF3 用 mini-rollout 解决扩散无法直接回归误差的问题）。
   → **目标 spec**：[`semantic-state-owners.md`](../../../docs/specs/semantic-state-owners.md)（R11）+ [`evaluation.md`](../../../docs/specs/evaluation.md)（R12）+ [`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md)（R-PE readout）
   → **落地动作**：让 VZ 各模块在快照中**自带一个学到的"自信度/预期误差"字段**（小头预测自身可靠性），**只读**发布、被下游门控（ModificationGate）与排序消费；由模块自身打包描述（合 SSOT 铁律）。
   → **预期收益**：把"内部不确定性"升为一级**可命名快照字段**，为 R9/R10 的有界自修改提供"可靠性门控"输入；与生成式组件配套即抗幻觉。
   → **风险/前提**：置信头是**对监督目标的回归**，AF 有 ground-truth 结构可校准；**VZ 关系域缺真值标签**——须用代理目标（接 99 的 rBio"软验证器"思路：用 VZ 自身 world/self 双轨预测做软真值），否则置信不可校准。

3. **机制：约束下扩散生成 + 多样本采样-排序-罚分**（AF3：5 seeds×5 samples，按置信 + 手性/碰撞罚分排序选优；交叉蒸馏抗幻觉）。
   → **目标 spec**：[`affordance.md`](../../../docs/specs/affordance.md) + [`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md)（候选生成与选择在 z_t 空间）
   → **落地动作**：VZ 生成候选回应/计划时，采**采样多候选 → 内部置信 + 显式约束罚分（VZ 的边界/一致性/承诺约束 = AF3 的手性/碰撞）排序 → 选优**，而非单路贪心解码；候选生成与排序均在 latent/控制器层。
   → **预期收益**：用"内部评判者 + 显式约束"做**有界 latent 搜索**，提升稳健性并把硬约束（边界同意、承诺一致）显式编码进排序。
   → **风险/前提**：多样本算力成本高；排序器质量是瓶颈；**生成式组件会幻觉**（AF3 教训），必须先有硬评估 + 校准置信（合 §2.2 反例 5 的 genuine-risk）。

## 3. 一句话定位

Isomorphic/AlphaFold 是一面**纠偏镜**：它是端到端监督训练的结构预测器，**不**支撑 VZ 的 R2（冻结基底）、R-PE（在线 PE）、R7（双轨）——99 综合把它标为 R2 ★ 属确证偏误，应纠正；其真正可借鉴物是三个解耦机制——**recycling 式 latent 迭代精修**（R3/R4 的计算模式，非控制）、**校准置信头作为可命名可发布的不确定性**（R11/R12 本 lab 最硬确证）、**约束下扩散 + 多样本排序**（带"生成式必幻觉、评估须先硬"的 genuine-risk 警示）。

## 附：本地论文清单（同目录 PDF）

| 论文 | 年 | ID | 可获取 | 核验状态 |
|---|---|---|---|---|
| Highly accurate protein structure prediction with AlphaFold (AF2) | 2021 | doi:10.1038/s41586-021-03819-2 | PDF（本目录） | 已核验（abstract+Evoformer/结构模块/IPA/FAPE/recycling+CASP14 数字+消融+局限） |
| Accurate structure prediction of biomolecular interactions with AlphaFold 3 | 2024 | doi:10.1038/s41586-024-07487-w | PDF（本目录） | 已核验（Pairformer+扩散模块+mini-rollout 置信+交叉蒸馏+PoseBusters/DockQ 数字+手性/幻觉/动力学局限） |
