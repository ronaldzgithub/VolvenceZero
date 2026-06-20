# 关系域软验证器 Spec（落差分析 + SHADOW 验证设计）

> Status: design / SHADOW-only proposal（尚未落地任何 owner 改动）
> Last updated: 2026-06-20
> 对应需求: R-PE, R2, R4, R7, R10, R12, R15
> 来源研究: `research/neolabs-2026-06/labs/czi-virtual-cell/analysis.md`（rBio soft-verifier RL）、`research/neolabs-2026-06/99_synthesis_vz_mapping.md` §四.1、`research/probe/11_vz_implications.md`（R-PE / R10 行动清单）

## 要解决的问题

VZ 的产品本质是关系 / EQ / 信任，而**关系质量没有可验证的标量奖励**——你无法对"这一轮陪伴是否真的修复了信任"打出一个 ground-truth 数字。这是整套技术路径里"信心最低、却正对产品命脉"的一个未解点（见 research `README.md` 五、`99_synthesis_vz_mapping.md` 六.1）。

CZI 的 **rBio** 在生物域定量证明了同构难题的一个解法：在没有硬标签时，用一个**冻结的预测世界模型**当"软验证器（soft verifier）"，对候选行为发**软概率奖励** $r=p(\cdot\mid M)\in[0,1]$，做 RL——软验证 ≈ 硬验证（F1 0.66 vs 0.67）。

但 rBio **不能整包搬进 VZ**：它把奖励来源放在**外部模型**上、且在 **token 空间做 GRPO**，这两点恰好踩在 R-PE（PE 不外包）与 R4（不在 token 空间做长程 RL）两条红线上。

**本 spec 的目的不是落地实现，而是：(1) 精确刻画从 rBio 类比到 VZ 关系尺度之间的落差；(2) 给出一个可证伪、可回滚的 SHADOW 验证设计，用来回答"VZ 自身的内禀预测误差能否在关系域当学习信号、而不塌缩成自我确认幻觉"。** 在 SHADOW 证据未通过前，关系域不得引入任何软验证器驱动的 learning writeback。

## 落差分析：rBio（生物域）vs VZ（关系域）

| 维度 | rBio（已被定量验证） | VZ 关系域（待验证） | 落差性质 |
|---|---|---|---|
| **验证器来源** | 外部冻结模型（MLP / TranscriptFormer / GO） | 必须是 VZ **自身双轨 World/Self 预测基底**（否则违反 R-PE 不外包） | 改写：把"外部任务模型"换成"内禀预测器" |
| **优化空间** | token 空间 GRPO（全 3B 模型端到端 RL） | 控制器 **z_t 空间**有界 RL，substrate 冻结（R2/R4） | 改写：奖励语义可借，优化空间不可借 |
| **ground-truth 锚** | **有**：底层世界模型在 112M 真实细胞上预训练；扰动最终可被湿实验证伪 | **薄/无**：关系质量没有可穷尽验证的客观终态；最接近的锚是事后人审（债 #51 external_validated） | **这是最大、最危险的落差** |
| **失准检测** | GO 验证器"反而掉点"被作者发现——因为有硬 EXP 数据对照 | 若无外部锚，软验证器可以**自我确认**：奖励它自己预测的东西，eval 也跟着涨，但关系实际没改善 | 必须人工造一个外部锚来打破自循环 |
| **噪声源** | 实验测量噪声（aleatoric），相对有界 | **用户本身高度随机**（情绪、语境、表达），aleatoric 噪声大——noisy-TV 风险高 | 必须复用 epistemic/aleatoric 分离，只奖 epistemic |

**核心结论**：rBio 的软验证之所以成立，是因为它的世界模型**确实见过大规模真实数据、且终态可证伪**。VZ 关系域的软验证器**缺这个锚**。因此把 rBio 内核搬进 VZ 的**唯一安全方式**，是先用一个**独立的外部人审信号**把"软验证器学习是否真的对齐了真实关系改善"验证出来，再决定是否让它进入学习链路。否则它就是"自我循环幻觉"（见 czi analysis 反证 D）。

## 关键不变量

软验证器机制在 VZ 落地时，必须同时满足以下边界（任一破坏即回退）：

1. **验证器是内禀的（R-PE）**：reward 来自 VZ 自己的 World/Self 预测误差（`PredictionErrorSnapshot.error` 的 `relationship_error` 轴 + dual-track tension），**不是**第三方任务模型的输出。不存在"目标各异、会漂移、不可问责"的外部奖励所有者。
2. **substrate 冻结、只训控制器（R2）**：软验证器奖励只驱动 `z_rel` 控制器层的有界更新；基础模型不做端到端梯度更新。
3. **优化在 z_t 不在 token（R4）**：组内归一化 advantage 作用在控制器代码空间，token 是表达层、不是 RL 目标。
4. **双轨不互读（R7）**：World 轨预测对方/环境响应，Self 轨预测自身状态轨迹；软验证器是两轨**各自**的 PE readout 的组合，不让一轨直接读另一轨内部状态。
5. **评估只读（R12）**：外部人审锚（external_validated）用于**验证软验证器是否可信**，本身**绝不回灌**学习链路——它是裁判，不是奖励源。否则等于把评估反向训练成第二套 reward。
6. **可回滚（R15）**：全程走 `WiringLevel` 三态 + readout-only→acceptance 三阶升级；任何阶段一次 rollback drill 失败即回退并写 known-debt。

## 机制映射（剥离生物叙事 → VZ owner）

> 本节只描述"如果验证通过会怎么接"，不是 SHADOW 阶段要做的事。SHADOW 阶段只产 readout 证据，见下一节。

| rBio 机制 | VZ 落点 owner | 映射 |
|---|---|---|
| 软验证器 $r=p(\cdot\mid M)$ | `PredictionErrorModule` | 候选关系行为的 `relationship_error` 分级散度 → 软概率 reward（不是二值） |
| 组内归一化 advantage | `temporal`（z_t 控制器）+ `credit` | 对一个 turn 的 N 个候选 `z_rel`，组内归一化 $\hat A_i=(r_i-\text{mean})/\text{std}$ |
| 可组合验证器（多源相加） | `credit` 多源聚合 | World 快/中/慢 PE + Self 轨 PE + vitals 慢尺度 PE 加权组合，逐源监控贡献 |
| 冻结世界模型当虚拟仪器 | `dual_track` World 轨 | 给定上下文 + 拟施加行动 → 预测分级关系响应（反事实 rollout） |
| GO-掉点检测（失准源） | `credit` 逐源贡献监控 + `evaluation` 只读 | 某源持续拉低外部锚相关性 → 视为漂移源，回滚其权重 |

复用既有零件（避免造新 owner）：
- **epistemic/aleatoric 分离**：直接用 `PEDecomposition`（Curiosity-Critic，已上线）。reward **只取 `improvement_magnitude`（epistemic）**，对 `aleatoric_magnitude` 不奖励——这是 noisy-TV / 用户随机性的结构性防御。
- **外部人审锚**：用 `relationship-continuity-external-validation.md`（债 #51）的 `external_validated` 三态标注 + 双盲第三方评分作为**唯一的对照真值**。
- **门控**：`ModificationGate` 的 Two-Gate（validation_delta / capacity_cost / rollback_evidence）+ `FramingAwarenessCheck`（OA-3）原样适用。

## SHADOW 验证设计（本 spec 的核心交付）

> 目标：在**不改变任何学习链路**的前提下，回答一个可证伪的科学问题——
> **「VZ 自身的关系域 PE，作为软验证器奖励，是否与独立外部人审的关系改善正相关；还是只与它自己的预测相关（自我确认）？」**

### 中心实验：自我确认证伪（self-confirmation falsification）

这是整份设计的命门，也是 rBio 没有、VZ 必须自己造的环节。

- **观测组（影子奖励）**：对每个 user turn，PE owner 计算软验证器奖励 $r_{\text{soft}}$（仅 readout，不写回）。同时记录候选 `z_rel` 的影子 RL advantage（**计算但不 apply**）。
- **外部锚组**：在一组 held-out 会话上收集 `external_validated` 关系连续性评分（双盲第三方，债 #51 协议）。
- **判决量**：
  - $\rho_{\text{ext}}$ = corr(影子奖励驱动的 session 改善方向, 外部人审评分变化)。
  - $\rho_{\text{self}}$ = corr(影子奖励, 软验证器自身下一轮预测)。
  - **通过条件**：$\rho_{\text{ext}}$ 显著为正，且 $\rho_{\text{ext}}$ 不被 $\rho_{\text{self}}$ 主导（即不是纯自循环）。
  - **证伪条件（关键）**：若 $\rho_{\text{self}}$ 高而 $\rho_{\text{ext}}\le 0$ —— 说明软验证器在自我确认，**整条路径在关系域不成立**，必须停在 SHADOW，并把该负结果写入 known-debt 作为"关系域软验证器不可用"的硬证据。

### 三阶升级协议（对齐既有 readout-only → acceptance-gate 模式）

复用 `prediction-error-loop.md` Wave E3 与 `credit-and-self-modification.md` Wave E3 已建立的协议，新增 gate `VZ_RELATIONAL_SOFT_VERIFIER`：

| 阶段 | WiringLevel | 行为 | 准入条件 | 退出 / 回滚 |
|---|---|---|---|---|
| `readout-only`（默认起点） | SHADOW | 计算 $r_{\text{soft}}$、影子 advantage、$\rho_{\text{ext}}$ / $\rho_{\text{self}}$；**绝不写回** | 无门槛；纯诊断 | — |
| `readout-with-acceptance` | SHADOW→ 局部 ACTIVE | 软验证器贡献进入 `credit` readout，但仍不驱动 `z_rel` apply | 在 ≥ 200 turn 真 trace + ≥ 1 批外部锚上 $\rho_{\text{ext}}\ge 0.2$ 显著为正；epistemic reward 占比 ≥ baseline；无单源漂移 | $\rho_{\text{ext}}$ 退到 < 0 持续 ≥ 1 批外部锚 → 退回 readout-only |
| `acceptance gate`（终态） | ACTIVE | 软验证器 epistemic 奖励驱动 `z_rel` 有界 RL（substrate 冻结） | ≥ 500 turn 真 trace + ≥ 2 批外部锚上 $\rho_{\text{ext}}\ge 0.35$；rollback drill 通过；aleatoric 不塌缩到 0 持续 ≥ 100 turn；Two-Gate / Framing 全绿 | 一次 rollback drill 失败 / $\rho_{\text{ext}}$ 跌破阈值 → 立刻退回上一阶段 + 写 known-debt |

约束（同既有协议）：升级不能跨 wave 同时发生；每次升级必配 rollback drill 测试；默认 SHADOW 保证对既有 PE/credit 测试与运行时 **零影响**。

### 落地点（建议，SHADOW 阶段只需前两项）

- **探针**：`artifacts/eq_uplift/probe_relational_soft_verifier.py` → 输出 `artifacts/eq_uplift/relational_soft_verifier_shadow.json`（含 $\rho_{\text{ext}}$ / $\rho_{\text{self}}$ / per-source 贡献 / epistemic 占比），模仿 `probe_pe_window_long_form.py` 的写法。
- **长 scenario**：复用 `long-form-life-arc.json` 一类 ≥ 38-turn scenario，叠加债 #51 的外部锚收集点。
- **report-only metric**：`evaluation` 新增 `rsv_rho_external` / `rsv_rho_self` / `rsv_epistemic_ratio`，**严格 report-only**，不进任何 acceptance gate（同 `pe_aleatoric_magnitude` 先例）。
- **rollback drill**：`tests/contracts/test_relational_soft_verifier_rollback_drill.py`（升到 readout-with-acceptance 时才需要）。

## 风险地图 / 会证伪整条路径的条件

| 风险 | 表现 | 检测 | 处置 |
|---|---|---|---|
| **自我确认幻觉**（最致命） | $\rho_{\text{self}}$ 高、$\rho_{\text{ext}}\le 0$ | 中心实验判决量 | 停在 SHADOW，写"关系域软验证器不可用"硬债务 |
| **noisy-TV / 用户随机性上瘾** | aleatoric 驱动奖励、对噪声反应 | `PEDecomposition` epistemic 占比 | reward 只取 epistemic；占比塌缩即回退 |
| **单验证器源漂移**（GO-掉点同构） | 某源拉低 $\rho_{\text{ext}}$ | `credit` 逐源贡献监控 | 回滚该源权重；no single source 独占 |
| **外部锚被反向当奖励**（违反 R12） | external_validated 进了学习链路 | 契约测试 | fail-closed：锚只读，不可回灌 |
| **substrate 被端到端更新**（违反 R2/R4） | 梯度落到基底 / token | import & gate 契约测试 | 只允许 z_rel 有界更新 |

> **诚实声明**：本设计能验证的上限是"软验证器在关系域**可用且不自欺**"，**不能**证明它一定能造出高质量关系——后者还依赖 World/Self 预测基底本身是否在足量真实关系交互数据上训练过（czi analysis 反证 D）。若 SHADOW 阶段 $\rho_{\text{ext}}$ 长期上不去，最可能的根因不在本机制，而在**预测基底的关系数据覆盖不足**——那是另一条独立的 blocker，应分开记债。

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|---|---|---|
| 依赖 | Prediction Error 主链 | 软验证器奖励 = `relationship_error` 轴的 epistemic 部分；复用 `PEDecomposition` |
| 依赖 | 双轨学习 | World/Self 各自 PE 组合成可组合验证器；不互读 |
| 依赖 | 信用分配与自修改 | 组内归一化 advantage / 逐源贡献 / Two-Gate / Framing 全在 `credit` + `ModificationGate` |
| 依赖 | 评估体系 | external_validated 外部锚（债 #51）是唯一对照真值，只读 |
| 依赖 | 契约式运行时 | `VZ_RELATIONAL_SOFT_VERIFIER` 走 WiringLevel 三态，可回滚 |
| 协作 | Lifeform Vitals | vitals 慢尺度 PE 可作为可组合验证器的一个慢源 |

## 拟同步修订的既有 spec（验证通过后才动）

- `prediction-error-loop.md`：新增 §"关系域软验证器奖励来源（epistemic-only）"。
- `credit-and-self-modification.md`：新增 §"可组合验证器 + 逐源漂移监控"与对应 gate 升级条件。
- `dual-track-learning.md`：新增 §"World/Self PE 作为软验证器组合源"。
- `evaluation.md` / `relationship-continuity-external-validation.md`：登记外部锚作为软验证器的**只读对照**，明确不可回灌。

## 变更日志

- 2026-06-20: 初始版本。基于 research `czi-virtual-cell/analysis.md` 的 rBio soft-verifier-RL，建立关系域落差分析 + SHADOW 验证设计；中心实验为"自我确认证伪"（$\rho_{\text{ext}}$ vs $\rho_{\text{self}}$）；定义 `VZ_RELATIONAL_SOFT_VERIFIER` 三阶升级协议。本 spec 为 design-only，未改动任何 owner 或运行时行为。
