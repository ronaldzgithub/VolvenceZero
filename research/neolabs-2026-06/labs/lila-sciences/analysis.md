# Lila Sciences — 深度分析

- **分组 / 成熟度 / 一句话主张**：B 自主 AI 科学家 / 闭环发现｜**低**（公司第一方可验证论文≈0，"科学超级智能"主张多见于 blog/新闻/融资稿）｜公司主张：构建"科学超级智能"（Lila Iris 模型）+ 自主机器人"AI 科学工厂"，跨生命/化学/材料执行完整科学方法。
- **主要创作者 + 血统**：Geoffrey von Maltzahn（联创/CEO）、Andrew Beam（CTO/AI）、John Gregoire（首席自主科学官，闭环材料 active learning）、**Rafael Gómez-Bombarelli（物理科学 CSO，分子 VAE 一作）**、**Kenneth O. Stanley（开放性 SVP，POET 共同资深作者）**。**与 VZ 的关系是分裂的**：公司叙事（无界开放式 + 全自动科学闭环）在 R9/R10/R15（有界自修改）与 R12（评估先做硬）上**直接挑战** VZ；但两位科学领袖的**入职前奠基算法**（连续 latent 设计空间、最小准则协同进化课程）恰好在 R3/R4 与"课程涌现"上与 VZ 共振。本分析的纪律是：**把可验证的创始人算法与不可验证的公司叙事彻底分开**。

> **核验边界（最重要）**：本目录两篇 PDF **都不是 Lila 的第一方成果**，而是 Gómez-Bombarelli（2016/17，Harvard/Aspuru-Guzik 组）与 Stanley（2019，Uber AI Labs）入职 Lila **之前**的论文。**Lila 公司的"科学超级智能 / Lila Iris / AI 科学工厂"主张在本目录无任何可核验证据 → 一律标记 `UNVERIFIED / 营销叙事`，不得作为 VZ 的设计依据导入。** 下文 §1 的两篇按 PDF 逐字核验；公司层主张只作"叙事"处理。

---

## 1. 核心逻辑（论文级 · PDF-grounded）

### 1.A 创始人奠基论文（可验证 · 与 Lila 路线相关但非 Lila 产出）

**① Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules**（Gómez-Bombarelli et al., arXiv:1610.02415v3, 2017）
- **问题**：分子设计是"在巨大（10^23–10^60 候选）、离散、无结构的化学空间里找最大化目标性质的分子"的优化问题；既有方法（固定库穷举、遗传算法/离散局部搜索）依赖手工突变规则、无法用梯度引导。
- **方法/机制**：训练三耦合网络——**encoder**（SMILES 字符串 → 连续 latent 向量）、**decoder**（latent → SMILES）、**predictor**（MLP 从 latent 预测性质）。普通 AE 的 latent 存在"死区"（解码出非法 SMILES），故改为 **VAE**（对 encoder 加噪 + 正则惩罚），迫使 latent 各处都能解码出有效分子。**联合训练**重建损失 + 性质预测损失，使 latent 按性质值组织成梯度。设计动作 = 在连续 latent 里做操作：解码随机向量 / 扰动已知分子 / **slerp 球面插值** / 用 Gaussian process 代理模型做**梯度/贝叶斯优化**后解码。
- **关键结果（PDF 内数字）**：ZINC(250k)/QM9(108k) 两套，latent 维 196/156；编码-解码忠实度高（训练集解码成功率 ZINC 92.1% / QM9 99.6%，但 QM9 模型对真实小分子泛化差，ZINC-外样本仅 1.4%）；随机 latent 点解码有效率 73.9%/79.3%；性质预测 MAE 与 graph convolution 相当（如 ZINC logP MAE 0.13–0.15）；以 `5×QED − SAS` 为目标，从最差 10% 起优化，GP-on-latent **稳定优于**随机搜索与遗传算法基线。
- **局限**：SMILES 字符级脆弱（解码有效率可低至 <1%，需 RDKit 后处理丢弃非法分子）；高维 latent 欧氏距离≠分子相似度（需 slerp）；会生成图合法但化学不可行的官能团；目标函数必须人手精心设计才能逼近"真正可取"的分子。**这是一个"连续可微 latent 设计空间 + 代理模型优化"的纯算法范式，不含任何自主实验闭环。**

**② Paired Open-Ended Trailblazer (POET)**（Wang, Lehman, Clune, Stanley, arXiv:1901.01753v3, 2019）
- **问题**：机器学习史上"问题由人提、算法只解"；能否让算法**同时生成问题与求解**，自建不断扩张的课程，让阶段性解成为后续更难问题的踏脚石（stepping stones），实现**无界（open-ended）**的复杂度与多样性增长？
- **方法/机制**：维护一组 **(环境, 智能体) 配对**，三任务主循环——(1) **变异生成新环境**（扰动环境编码参数）；(2) 用 **ES（演化策略）** 独立优化每个配对智能体；(3) **transfer / goal-switching**：把某环境的智能体迁移到另一环境，若更优则替换。继承 **MCC（最小准则协同进化）**：新环境必须通过 **minimal criterion**（`50 ≤ E(θ) ≤ 300`，"不太难也不太易"）+ 父代足够进步（reward ≥ 200）+ 优先**新颖性**才能入群；环境群有上限，超限按队列淘汰最老。
- **关键结果（PDF 内数字）**：2-D Bipedal Walker 障碍域（gap/stump/stairs/roughness 五类参数编码）。POET 创造并解决的环境，**用同一 ES 从零直接优化无法解**（五次 ES run 最高分仅 17.9 / 39.6 / 13.6 / 24.0 / 19.2，远低于成功阈值 230，单样本 t 检验 p<0.01）；**显式直达式课程对照**在"很难/极难"级别同样失败，**transfer 被证明是成功关键**。规模：256 CPU 核、~10 天/run、25,200 迭代、活跃环境群 20、ES 群体 512；解决 challenging/very/extremely 环境分别需 638±133 / 1180±343 / 2178±368 迭代。
- **局限**：仅 2-D 玩具域；环境编码是**人手设计**的 5 参数（"open-ended"受限于此编码空间）；计算极贵（256 核 × 10 天）；**无界生长本身没有终止/收敛/资源边界**，统计比较困难（需事后 single-sample t-test）。**这是一个"环境-解协同进化的课程涌现机制"，其无界性是其卖点也是 VZ 视角下的最大风险源。**

### 1.B 公司层主张（UNVERIFIED · 营销叙事 · 不导入）

- "Scientific superintelligence / Lila Iris / 自主机器人 AI 科学工厂、跨生命-化学-材料执行完整科学方法"——**本目录零第一方证据**。无架构论文、无基准、无可对照的实验报告。按 [`99_synthesis_vz_mapping.md`](../../99_synthesis_vz_mapping.md) 五·1 的判断：**只引入闭环机制（来自创始人算法），不引入公司主张，且必须等 eval 做硬之后。**
- 关联教训（非 Lila 但同赛道）：自主合成闭环（Periodic 引用的 **A-Lab 2023**）的结果**后被学界质疑**——"自动化闭环 ≠ 结果可验证"。这是 §2.2 的核心反证素材。

---

## 2. 与 VZ 的关系（三视角 · 先反证后确证）

### 2.2 反证（红队）— 先行

逐条裁决（survives / needs-boundary-condition / genuine-risk）：

1. **反例：公司"科学超级智能 + 全自动科学闭环"叙事——"自动化可先跑起来，评估可后补"。** 自主科学家赛道（含 Lila）默认"AI 提假设 → 机器人做实验 → 实时学习"的闭环本身就是产出；A-Lab 自主合成结果被质疑直接证明：**无可验证、可对照 eval 的自动化闭环，只会更快放大错误。**
   → **裁决：genuine-risk（对"评估可后补"叙事），对 VZ 自身则强力 survives 并加固。** **边界条件**：VZ 的 R12 必须**硬性前置于任何自动化/自修改闭环**——评估覆盖"存在/连续性"而非任务、且**只读**。在 [`evaluation.md`](../../../docs/specs/evaluation.md) / [`evidence_program.md`](../../../docs/specs/evidence_program.md) 中明确写入"A-Lab 教训"作为反例锚点：闭环上线前 eval 必须先做硬。**Lila 公司主张进风险登记，禁止作为证据引用。**

2. **反例：POET 的"open-ended 无界生成"——"无界自改/自生成可持续产出，有界是自缚"。** POET 论证 open-endedness 是逼近更高能力的更优路径（直达式课程失败、无界协同进化成功），与 Sakana AI Scientist / Darwin Gödel Machine 同属"无界自修改"阵营。
   → **裁决：needs-boundary-condition。** **边界条件**：POET 的有效性建立在**有 minimal criterion 闸门**（`50≤E≤300`、父代进步阈值、新颖性筛选、环境群上限队列淘汰）之上——**它恰恰不是真正无界，而是"有界准则 + 多样性"驱动的受控扩张**。这与 VZ 的 R9/R10/R15（有门控、有 owner、可回滚的分层自修改）**一致而非对立**。VZ 应在 [`credit-and-self-modification.md`](../../../docs/specs/credit-and-self-modification.md) 写明：自生成/自修改的"开放性"必须挂在 minimal-criterion 式闸门 + 退出条件 + 评估证据上；**rare-heavy artifact 走 ModificationGate，不得 bypass。** 无闸门的无界（纯营销式"superintelligence"）判 genuine-risk，已在条 1 处理。

3. **反例：自主科学家把 PE/任务/奖励来源外包给 LLM 或外部生成器——"PE 来源可外包"。** 该赛道常用 LLM 当 task/reward 生成器。
   → **裁决：genuine-risk（对外包做法），VZ 不变量 survives。** **边界条件**：R-PE 要求内禀 PE **不外包**——一旦生成器的任务表示偏了，所有下游一起偏且无法检测。VZ 的 PE 必须是 [`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md) 里 VZ 自身可控的一级运行时对象；外部模型至多作 affordance/工具，不得成为 PE 的隐式 owner。

4. **反例：POET 的"目标无关多样性优先"（novelty over objective）可能与 VZ 关系优先目标冲突。** POET 刻意追求新颖而非单一目标。
   → **裁决：survives（目标域不同）。** **边界条件**：VZ 目标域是关系/EQ/长程养成，**不是**开放式能力涌现；POET 的多样性机制只在"课程/环境生成"这一**机制层**借鉴，不把"新颖优先于目标"提升为 VZ 的顶层价值。

### 2.1 确证（先进性背书）

> 注意：以下背书**只来自创始人奠基算法**，不来自 Lila 公司主张。且二者均为 ML 同社区成果，**不构成跨领域独立收敛证据**（弱于生物/神经科学旁证）。

- **R3/R4 latent 控制（中·算法级，非跨领域）**：分子 VAE 直接示范"把离散对象压到**连续可微 latent**，在 latent 里做优化/插值/梯度搜索再解码回表层"——与 VZ"控制发生在 token 空间之上的 z_t 空间、表层只是解码产物"同构。slerp/联合性质预测组织 latent 的做法是工程可借的具体证据。
- **R3（时间抽象 / 课程涌现，中·机制级）**：POET 示范"有意义的能力可从**自生成课程**中涌现，无需预定义全部踏脚石"——为 VZ"时间抽象/课程不必硬编码、可涌现"提供独立机制背书（但仅在玩具域）。
- **R1/R13（弱·概念）**：自主科学家闭环 = "压缩（SSL/世界模型）↔ 强化（实验验证）"交替的工程范例（[`99`](../../99_synthesis_vz_mapping.md) 三·6）。但因公司层 UNVERIFIED，此条仅作概念呼应，不记强背书。
- **R9/R10（弱·反向背书）**：POET 的 minimal criterion 闸门**反向印证** VZ"有界自修改"路线——见 §2.2 条 2，开放性必须挂闸门才有效。

### 2.3 局部算法借鉴（算法级解耦）— 剥离公司叙事，只取创始人算法机制

明确：**不借公司任何东西**；以下 5 元组全部来自两篇创始人论文的可验证机制。

1. **机制**：**连续可微 latent 设计空间 + 代理模型梯度/贝叶斯优化**（分子 VAE：encoder/decoder/predictor 联合训练，latent 按目标量组织成梯度，slerp 插值，GP-on-latent 优化后解码）。
   → **目标 spec**：[`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md)（R3/R4，z_t 控制空间）
   → **落地动作**：把"在 z_t latent 空间做带代理目标的有向搜索/插值、再解码到表达层"作为 VZ 控制器层操作的设计参照；联合训练一个轻量 predictor 让 z_t 空间**按目标量（如关系质量代理信号）组织成可优化梯度**。
   → **预期收益**：为"控制在 latent 空间、表层是解码产物"提供成熟可借的优化机制，避免在 token 空间做长期决策。
   → **风险/前提**：VAE 解码有效率/泛化脆弱（QM9 外样本 1.4%）——必须有"有效性后处理/约束"对应物；代理目标设计不当会优化到化学/语义上"合法但不可取"的解，目标函数（关系质量）须谨慎设计。

2. **机制**：**Minimal Criterion 协同进化课程闸门**（POET：新环境须满足 `不太难不太易` + 父代进步阈值 + 新颖性 + 群上限队列淘汰）。
   → **目标 spec**：[`credit-and-self-modification.md`](../../../docs/specs/credit-and-self-modification.md)（R9/R10/R15 有界自修改）+ 兼 [`environment-interface.md`](../../../docs/specs/environment-interface.md)（场景/课程生成）
   → **落地动作**：把"自生成场景/自修改候选"的接纳条件实现为 minimal-criterion 式闸门——只接纳"对当前控制器既非太易也非太难、且足够新颖、父代已达进步阈值"的候选；超限按队列淘汰；每个新自适应层带退出条件。
   → **预期收益**：让 VZ 的开放性/自生成有**受控扩张**而非无界爆炸；天然契合 ModificationGate 与可回滚迁移。
   → **风险/前提**：阈值（难度区间/新颖性）需校准；POET 仅在 2-D 玩具域验证，移植到关系养成域需重设"难度/进步"度量（且度量本身要走 R-PE，不可关键词匹配）。

3. **机制**：**Transfer / goal-switching 踏脚石迁移**（POET：周期性把一处学到的解迁移到另一环境，更优则替换；被证明是 open-ended 成功关键）。
   → **目标 spec**：[`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md)（R1/R2，控制器层跨情境复用）
   → **落地动作**：在多场景/多 regime 下，允许某 regime 学到的控制器适配项作为"踏脚石"被其他 regime 的适配器尝试性复用，仅当带来评估改进时采纳（只读评估门控）。
   → **预期收益**：跨情境复用阶段性能力，避免每个场景从零优化，提升养成样本效率。
   → **风险/前提**：迁移须经只读 eval 门控（R12），且不得污染 R7 双轨隔离（World/Self 不互相直写）；POET 的迁移在参数空间，VZ 须映射到"控制器适配项"而非基底，遵守 R2 冻结基底。

---

## 3. 一句话定位

Lila Sciences 的**公司叙事（"科学超级智能 + 全自动科学闭环"）是 UNVERIFIED 营销，必须当作 R12 的反面教材而非证据**——A-Lab 教训证明"自动化≠可验证、评估必须先做硬"；其真正价值在两位科学领袖的**入职前算法**：分子 VAE 给 R3/R4 提供"连续 latent 设计空间 + 代理优化"的成熟机制，POET 给 VZ 提供"minimal-criterion 闸门 + 踏脚石迁移"的**受控开放性**模板，恰恰反向印证 VZ"有界、可回滚、评估先行"的克制是对的。

## 附：本地论文清单（同目录 PDF）

| 论文 | 年 | ID | 核验状态 | 归属说明 |
|---|---|---|---|---|
| Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules | 2016/17 | arXiv:1610.02415v3 | 已逐篇核验（PDF） | 创始人 Gómez-Bombarelli **入职前**奠基工作，非 Lila 第一方 |
| Paired Open-Ended Trailblazer (POET) | 2019 | arXiv:1901.01753v3 | 已逐篇核验（PDF） | 创始人 Stanley（Uber AI Labs）**入职前**奠基工作，非 Lila 第一方 |
| Progress and prospects for autonomous materials workflows（Gregoire，闭环材料 active learning） | 2019 | doi:10.1039/C9SC03766G | 未下载（本目录无 PDF） | 创始人入职前相关工作，UNVERIFIED |

> **公司层（Lila Iris / 科学超级智能 / AI 科学工厂）：本目录 0 第一方可验证论文 → 全部 `UNVERIFIED / 营销叙事`，不作为 VZ 设计依据。**
