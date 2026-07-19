# Noetik — 深度分析

- **分组 / 成熟度 / 一句话主张**：C 生物基础模型（空间多组学虚拟细胞）｜**低**（公司第一方可验证论文 = 0；全部信息源仅为 noetik.ai 网站技术报告，无 bioRxiv/arXiv 预印、无同行评审、无开放权重/基准）｜公司主张：自监督、大规模多模态空间组学"世界模型"（**OCTO**），构建"虚拟细胞"做 in-silico 肿瘤实验、寻找免疫治疗靶点。
- **主要创作者 + 血统（为何与 VZ 共振或对立）**：Ron Alfa（联创/CEO，前 Recursion）、Jacob Rinaldi（联创/CSO，前 Recursion）。血统来自 **Recursion**（表型组学/细胞影像基础模型），因此 OCTO 的"自监督掩码-预测空间组学世界模型"叙事**在概念层**与 VZ 的 World 轨预测器（R7 + R-PE）、冻结基底（R2）、潜空间预测（R3/R4）听起来共振。**但本 lab 的纪律必须是：所有"共振"目前都停留在公司网站叙事层，无任何可逐行核验的论文支撑 → 一律 `UNVERIFIED`，不得作为 VZ 任一不变量的证据。** 这家 lab 在本框架里的真正作用，是检验"被一个听起来与 VZ 同构的营销叙事吸引、从而把未经核验的主张当成跨领域独立收敛证据"这一确证陷阱——本分析以反证为先。

> **核验边界（最重要）**：本目录 **0 个 PDF**（已列目录确认）。下文 §1 的所有"问题/方法/结果"均**转述自 noetik.ai 网站技术报告与本仓库 flat note**，**未经任何 PDF 逐行核验，全部标 `UNVERIFIED`**。这与本批次中"创始人入职前有可核验 PDF"的 lab（如 Lila 的 Gómez-Bombarelli/Stanley 论文）**性质不同**：Noetik 在本目录连一篇可核验的第一方或创始人论文都没有。因此 **Noetik 不能为任何 R 不变量提供哪怕弱的背书**；它至多是一个"自监督虚拟细胞世界模型"的**概念回声**，而该概念的可核验证据应到 [`czi-virtual-cell`](../czi-virtual-cell/analysis.md)（rBio / TranscriptFormer）与 [`arc-institute`](../arc-institute/analysis.md)（State）那里取，**不在 Noetik**。

---

## 1. 核心逻辑（论文级 · 全部 UNVERIFIED · 无 PDF，源自 noetik.ai 网站技术报告）

> 以下每条均无 PDF 可核验，数字/设置一律 `UNVERIFIED`，仅作"叙事记录"，不得引用为事实。

### OCTO: A world model for cancer biology — Tech Report 1（2024，noetik.ai，UNVERIFIED）
- **问题（转述，UNVERIFIED）**：肿瘤是空间异质的多模态系统；单一模态（单纯转录或单纯影像）无法刻画细胞在其空间微环境中的状态与互作。需要一个统一吸收多重蛋白、空间转录、DNA、组织形态（H&E）的基底。
- **方法/机制（转述，UNVERIFIED）**：宣称为一个**结构化掩码 transformer**，对齐多模态空间组学（multiplex 蛋白免疫荧光 + 空间转录组 + DNA + H&E 病理图像），用**掩码-预测**做自监督；把"从空间上下文预测细胞状态"框定为一个显式"**世界模型**"。
- **关键结果（UNVERIFIED）**：**无 PDF、无基准数字、无对照、无同行评审可核验。** 网站层主张其学到肿瘤微环境的可迁移表征。**此处不记录任何具体数字，因为没有任何可核验来源。**
- **局限 / 纪律**：作为"证据"它的局限是**根本性的**——不可核验。任何"OCTO 在某任务上达到/超过 X"的说法在本目录都无法被检验，**禁止进入 VZ 的设计依据或确证清单**。

### OCTO-VirtualCell / Celleporter — Tech Report 3（2025，noetik.ai，UNVERIFIED）
- **问题（转述，UNVERIFIED）**：把 OCTO 表征用于"虚拟细胞"——在 in-silico 中预测细胞的空间单细胞表达，并以反事实（扰动）方式预测干预后的细胞状态，用于寻找免疫治疗靶点。
- **方法/机制（转述，UNVERIFIED）**：宣称可预测约 **4000 万细胞**的空间单细胞表达；"虚拟细胞反事实 = 扰动预测"，即给定上下文 + 干预 → 预测响应（概念上与扰动响应世界模型同构）。
- **关键结果（UNVERIFIED）**：**无 PDF、无 held-out 评估协议、无与线性/简单基线的对照、无同行评审。** "4000 万细胞"为网站叙事规模声明，**不作为事实采信**。
- **局限 / 纪律**：与 Tech Report 1 同——不可核验是其作为证据的**致命局限**。尤其"扰动预测/反事实"这类主张，在 [`arc-institute`](../arc-institute/analysis.md) 的 State 里有 PDF 级证据显示**扰动响应预测极易打不过线性基线**；Noetik 无任何对照，故其反事实主张的可信度**应默认按最弱处理**。

---

## 2. 与 VZ 的关系（三视角 · 先反证后确证）

> **本 lab 的重心几乎全在 §2.2。** 因为没有可核验证据，§2.1 不应给出任何实质背书，§2.3 只能是"概念性、且把可核验证据指回 CZI/Arc"的处理。

### 2.2 反证（红队）— 先行

逐条裁决（survives / needs-boundary-condition / genuine-risk）。注意：这里的"反例"主要不是 Noetik 的科学结论挑战 VZ（它无可核验结论可挑战），而是 **Noetik 这类 0-论文 lab 暴露的"确证方法论风险"**。

1. **反例（headline · 最该警惕）：一个听起来与 VZ 同构的营销叙事（"自监督掩码-预测空间组学世界模型 = 预测误差驱动 + 冻结基底 + 潜空间反事实"）是否可以被当作 VZ 不变量的跨领域独立收敛证据？**
   诚实地说：**绝对不能。** OCTO 的叙事确实在词面上同时碰到了 R-PE（掩码-预测≈预测误差）、R2（多模态基底）、R3/R4（潜空间反事实）、R7（世界模型）——**正因为它"全都像"，它才是确证偏误的完美诱饵**。但本目录 0 PDF、0 基准、0 同行评审，"全都像"恰恰说明这是抽象 R 轴 + 抽象营销词的**词面重合**，不是机制级证明。
   → **裁决：genuine-risk（针对"把不可核验网站主张当证据导入"这一行为），VZ 不变量本身不被它撼动也不被它背书（中立）。** **边界条件**：在 [`evidence_program.md`](../../../docs/specs/evidence_program.md) / [`evaluation.md`](../../../docs/specs/evaluation.md) 写入明确口径——**对齐 [`99_synthesis_vz_mapping.md`](../../99_synthesis_vz_mapping.md) 五·1 的 A-Lab 教训**：任何"实验室主张"在没有第一方可核验论文/基准前，**只能进风险登记，禁止作为 R 不变量的确证来源**；尤其禁止因叙事与 VZ 同构而降低核验门槛。Noetik 的全部主张归入此类。

2. **反例：OCTO 的"虚拟细胞反事实 = 扰动预测"是否给 VZ 的 World 轨/PE 提供了一个可借鉴的扰动响应世界模型？**
   诚实地说：**概念上相邻，但 Noetik 提供不了证据**——它没有 held-out 协议、没有与线性基线的对照。而 Arc 的 State（有 PDF）恰恰显示：扰动响应预测**长期打不过线性基线**，"做出一个会预测的模型"与"做出一个真正泛化的模型"之间隔着严格评估。
   → **裁决：needs-boundary-condition（针对"扰动/反事实世界模型可直接借鉴"），且证据须改引 Arc/CZI。** **边界条件**：World 轨的"反事实/扰动预测"设计若要借鉴，**证据锚点必须是 [`arc-institute`](../arc-institute/analysis.md) 的 State（分布级预测 + Cell-Eval 只读评估 + 对线性基线的硬对照）与 [`czi-virtual-cell`](../czi-virtual-cell/analysis.md) 的 rBio（软验证器奖励）**，而**不是** Noetik 的无对照主张。在 [`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md) / [`dual-track-learning.md`](../../../docs/specs/dual-track-learning.md) 中标注：反事实世界模型必须先过"能否稳定打过简单基线"的硬评估。

3. **反例：掩码-预测自监督损失是否印证 R-PE（预测误差是一级运行时信号）？**（与 Arc 反证 A 同型的确证陷阱）
   诚实地说：**不**。即使 OCTO 真如其所述用掩码-预测，那也是**离线 SSL 训练目标**，是 rare-heavy 层一次性最小化的经验损失，**不是 always-on 数字生命运行时的一级 PE，也没有 needs/credit/homeostasis 的下游 readout**。把训练 loss 读成 R-PE 是把"优化目标"误读成"运行时认知信号"。
   → **裁决：survives（目标域不同）+ needs-boundary-condition（须在 spec 写明区分）。** **边界条件**：同 Arc——在 [`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md) 写明"离线 SSL 掩码-预测损失 ≠ 运行时 PE 一级信号"。**且因 Noetik 不可核验，连"它是否真用掩码-预测"都无法确认，此条只作方法论提醒，不记 Noetik 任何功劳。**

4. **反例：Noetik 主打"in-silico 实验 / 自动找靶点"的自主发现闭环，是否说明"自动化闭环可先跑、评估可后补"？**
   诚实地说：这与自主科学家赛道（Lila/Periodic）的风险同源。无可验证 eval 的自动化闭环只会更快放大错误。
   → **裁决：genuine-risk（针对"评估可后补"），VZ 自身 survives 并加固。** **边界条件**：R12 必须硬性前置于任何 in-silico 发现闭环；评估只读、覆盖"存在/连续性"而非单一任务 → [`evaluation.md`](../../../docs/specs/evaluation.md)、[`evidence_program.md`](../../../docs/specs/evidence_program.md)。

### 2.1 确证（先进性背书）

> **结论先行：Noetik 在本目录 0 可核验证据 → 不为任何 R 不变量提供背书（包括弱背书）。** 下表把"叙事上听起来碰到的 R 轴"显式标注为"**UNVERIFIED · 不计背书**"，以防被后续 rollup 误当成证据。

| 叙事触及的 R 轴 | OCTO 网站叙事（UNVERIFIED） | 为何**不计**背书 |
|---|---|---|
| R-PE | 掩码-预测 = 自监督预测误差 | 即便属实也是离线训练 loss，非运行时 PE；且不可核验（见 §2.2 条 3） |
| R2 | 多模态空间组学基底供下游模拟查询 | 无权重/无基准，无法验证"冻结基底 + 下游 readout"成立；干净 R2 证据在 [`arc-institute`](../arc-institute/analysis.md) Evo/Evo2 |
| R3/R4 | 潜空间反事实/虚拟细胞预测 | 无 PDF 证明预测发生在潜空间且泛化；证据应取 Arc State 的 SE 潜空间 |
| R7 + 世界模型 | 显式"世界模型"叙事 | 概念回声，无对照、无评估框架；World 轨借鉴源以 Arc State / CZI rBio 为准 |

**唯一可诚实陈述的"确证"是负向的**：Noetik 的存在**侧面印证了 99 综合反复强调的"虚拟细胞自监督世界模型"这一跨社区趋势的热度**——即多家（CZI、Arc、Noetik）独立朝"自监督世界模型 + 反事实扰动"方向走。但**趋势热度 ≠ 对 VZ 不变量的独立验证**；真正的验证证据必须来自有 PDF 的 CZI/Arc，**Noetik 仅作趋势的一个未核验数据点登记**。

### 2.3 局部算法借鉴（算法级解耦）— 仅概念，且证据指回 CZI/Arc

> **纪律（本节最重要）**：Noetik **没有任何可核验机制可供"剥离叙事后搬运"**。因此本节**不提出任何以 Noetik 不可核验主张为依据的借鉴**。下表只记录"Noetik 叙事所指向的概念方向"，并把**真正的可落地机制与证据明确转交给 CZI/Arc 的 analysis.md**，避免 ROI 台账误收一条无证据的借鉴。

| # | 概念方向（来自 Noetik 叙事，UNVERIFIED） | 目标 VZ spec | 处理（不直接借 Noetik） | 备注 / 前提 |
|---|---|---|---|---|
| 1 | "空间上下文 → 细胞状态"的**反事实/扰动世界模型** | [`dual-track-learning.md`](../../../docs/specs/dual-track-learning.md)、[`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md) | **改引 [`arc-institute`](../arc-institute/analysis.md) State 的"扰动→响应分布"软世界模型**（有 PDF、有 MMD 分布对齐、有对线性基线的硬对照）作为 World 轨借鉴源；Noetik 仅作同方向的未核验旁证 | World/Self 双轨隔离（R7）；PE 须为运行时一级信号非训练 loss；先过"打过简单基线"硬评估 |
| 2 | 多模态自监督基底供下游"模拟查询" | [`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md) | **改引 [`arc-institute`](../arc-institute/analysis.md) Evo/Evo2** 作为"冻结基底 + 零样本 readout"的干净 R2 跨模态背书；Noetik 无权重/无基准，不采信 | 基底冻结、禁在线端到端梯度打基底（R2） |
| 3 | "软验证器/世界模型给无硬奖励的控制器提供学习信号"（Noetik 的 in-silico 发现隐含此结构） | [`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md)、[`credit-and-self-modification.md`](../../../docs/specs/credit-and-self-modification.md) | **改引 [`czi-virtual-cell`](../czi-virtual-cell/analysis.md) rBio 的"软验证器奖励"**（99 五·1 列为本批次对 VZ 最独特贡献）；Noetik 不提供任何可核验实现 | 软验证器须为 VZ 自身双轨模型，不外包 PE 来源（R-PE） |

---

## 3. 一句话定位

Noetik（OCTO / OCTO-VirtualCell）是一个**叙事上与 VZ 高度同构、但证据上完全不可核验（0 PDF / 0 基准 / 0 同行评审）的 0-论文 lab**——正因"全都像 VZ"，它是**确证偏误的完美诱饵**：必须判定为 **不能为任何 R 不变量提供背书**，其全部主张归入 `UNVERIFIED / 风险登记`；它能给 VZ 的唯一价值是**方法论警示**（A-Lab 同型教训：实验室主张在无第一方可核验证据前禁止当证据）。凡"自监督虚拟细胞世界模型 / 反事实扰动预测 / 软验证器"这些概念方向上的**可落地借鉴与可核验证据，一律改引 [`arc-institute`](../arc-institute/analysis.md) 的 State/Evo 与 [`czi-virtual-cell`](../czi-virtual-cell/analysis.md) 的 rBio，而非 Noetik**。

## 附：本地论文清单（同目录 PDF）

**本目录 0 个 PDF（已列目录确认）。** 以下条目全部 `UNVERIFIED`，源自 noetik.ai 网站技术报告，无可下载预印、无同行评审、无开放权重/基准：

| 论文 / 技术报告 | 年 | ID | 核验状态 | 说明 |
|---|---|---|---|---|
| OCTO: A world model for cancer biology — Tech Report 1 | 2024 | UNVERIFIED（noetik.ai 网站） | **未核验 · 无 PDF** | 宣称：multiplex 蛋白 + 空间转录 + DNA + H&E 的结构化掩码 transformer |
| OCTO-VirtualCell / Celleporter — Tech Report 3 | 2025 | UNVERIFIED（noetik.ai 网站） | **未核验 · 无 PDF** | 宣称：预测约 4000 万细胞的空间单细胞表达 + 反事实扰动 |

> **总纪律**：Noetik 全部信息为公司网站叙事 → 一律 `UNVERIFIED`，**禁止作为 VZ 任一 R 不变量的设计依据或确证来源**；可核验证据请到 [`arc-institute`](../arc-institute/analysis.md) / [`czi-virtual-cell`](../czi-virtual-cell/analysis.md) 同主题分析中取用。
