# AlphaEvolve 与进化算法方向：VZ 借鉴意义深度分析

> **文档角色**：跨轴精读专题（B3 Open-Ended Learning ↔ C1 Self-Improvement ↔ R10/R12/R13）。  
> **状态**：研究借鉴清单 — **不是** spec 改动本身；任何落地按 [`.cursor/rules/cursor-convergence-workflow.mdc`](../../../../.cursor/rules/cursor-convergence-workflow.mdc) convergence packet 推进。  
> **日期**：2026-05  
> **主要锚论文**：Novikov et al., *AlphaEvolve: A coding agent for scientific and algorithmic discovery*, arXiv:2506.13131.

---

## 包名歧义澄清（必读）

**`packages/lifeform-evolution/`** 在本仓库中的语义是 **lifeform 纵向养成 / 证据闭环 / 场景包与 benchmark**（longitudinal evolution of a companion），**不是**遗传编程或 MAP-Elites 意义上的 **evolutionary algorithm（进化算法）**。

- 若要把 AlphaEvolve 风格的 **程序种群 + archive + island** 机制工程化，应落在 **runtime / evaluation / gate / scenario 工具** 等有明确 owner 与 snapshot 契约的位置，**不要**仅因目录名含 "evolution" 就把算法塞进 `lifeform-evolution`，以免破坏 R8（owner SSOT）与模块边界。

---

## §1 AlphaEvolve 机制解剖（按 white paper §2 / §4 / §6）

以下内容与 [`research/papers/dm/alphaevolve-coding-agent-scientific-engineering-discovery-2506.13131.pdf`](../../../papers/dm/alphaevolve-coding-agent-scientific-engineering-discovery-2506.13131.pdf) 的结构对齐，便于追溯。

### 1.1 高层回路

AlphaEvolve 是 **LLM 编排的自治流水线**：用户给出初始程序（用 `# EVOLVE-BLOCK-START` / `END` 标注可演化块）、**评估函数** `evaluate(...) -> dict[str, float]`（约定最大化）、可选配置；系统在 **演化循环** 中用 LLM 生成 **代码修改（diff）**，应用后由 **Evaluators** 打分，高分程序写回 **Program database**，驱动下一轮。

### 1.1b FunSearch → AlphaEvolve（论文 Table 1 对齐）

White paper 将 AlphaEvolve 明确为 **FunSearch**（Romera-Paredes et al., 2023）的实质性扩展。对比维度如下（与原文 Table 1 一致）：

| 维度 | FunSearch | AlphaEvolve |
|------|-----------|-------------|
| 演化粒度 | 单个函数 | **整个代码文件 / 多组件代码库** |
| 代码体量 | 约 10–20 行 | **可达数百行** |
| 语言 | Python | **任意语言** |
| 单次评估预算 | 通常 ≤20 min（1 CPU） | **可达数小时**，加速器上 **并行** |
| LLM 调用量 | 百万级采样 | **数千级** LLM 采样即可（更强样本效率） |
| 模型规模 | 小型 code LLM；更大模型收益有限 | **SOTA 前沿模型**，scaling 明显 |
| 上下文 | 主要为先前解 | **丰富上下文 + 评测反馈** |
| 目标 | 单一标量目标 | **多目标同时优化** |

**对 VZ 的读法**：AlphaEvolve 解决的是 **「在同一 verifiable 目标下如何把搜索做得更像工业系统」** —— 而非重新定义「何为真理」。因此 **DM-4（eval 锚定）** 仍是前提；EVO 机制是 **在门槛之内如何把候选生成做得不退化**。

### 1.2 Program Database（演化档案）

论文 §2.5：**演化数据库**的目标是 **最优地复现已探索过的想法**，以便在后续代次中重新采样。设计挑战是 **exploration–exploitation** 平衡——既要持续改进最优程序，又要 **维持多样性** 以覆盖搜索空间。

实现上：**结合 MAP-Elites 风格思想与 island-based 种群模型**（文中引用 Mouret & Clune MAP-Elites；Tanese / 相关工作用于 island）。这与经典 GA「单一种群易早熟收敛」形成对照：**档案不是单一排行榜**，而是 **结构化留存多样高分程序** 的机制。

**VZ 映射**：ModificationGate 的 **rare-heavy 候选池** 若只用标量 net-benefit 排序，等价于「单峰爬山」；引入 **behavior descriptor × elite** 与 **岛间迁移限速**，对应 **防止 gate 候选在单一优化压力下塌缩**。

### 1.3 Prompt Sampler（§2.2）

演化提示包含：**从数据库采样的多条历史优良程序** + **如何对当前程序提出修改的系统说明**。额外可配置：

- **Explicit context**：问题描述、方程、代码片段、文献（含 PDF）。
- **Stochastic formatting**：模板占位符 + 配置文件中的概率分布，增大提示多样性。
- **Rendered evaluation results**：程序、执行输出、`evaluate` 分数。
- **Meta prompt evolution**：由 LLM 在额外步骤中生成/改进提示指令；**共演化**的提示存放在 **与程序库类似的独立数据库**。

**VZ 映射**：Reflection / expression 层 **集中管理的 prompt 模板**（见 [`.cursor/rules/llm-prompt-centralization.mdc`](../../../../.cursor/rules/llm-prompt-centralization.mdc)）可增加 **SHADOW 内的 meta-slot**，由 background-slow 迭代 — **不得**触及 metacontroller 的 z_t/β_t 决策逻辑。

### 1.4 Creative Generation（§2.3）

- **输出格式**：大规模改动用 `SEARCH`/`REPLACE` diff 块；短程序或可整体重写时允许输出整块代码。
- **模型**：**Gemini 2.0 Flash + Gemini 2.0 Pro** 组合 — Flash **低延迟、提高单位时间候选数**；Pro **偶尔给出高质量突破**。显式 **ensemble 权衡吞吐与质量**。

**VZ 张力**：此处 **mutation operator = LLM** 工作在 **token / 代码文本空间**。直接照搬为「用 LLM 生成 metacontroller 或 owner 内部结构的补丁」会撞击 **R4** —— 借鉴时需把 **变异对象** 限定在 **已有 owner 定义的 proposal schema**（数值门限、小型结构化 delta），而非开放式改写控制器语义。

### 1.5 Evaluation（§2.4）

原则：对每个候选运行用户定义的 \(h\)（即 `evaluate`）。工程增强包括：

- **Evaluation cascade（假设检验）**：多组测试难度递增；仅当前阶足够好才进入下一阶段 — ** cheap filter → expensive truth**。
- **LLM-generated feedback**：当「简洁性」等性质难以写入 \(h\) 时，可用 **单独 LLM 调用** 打分或淘汰；分数并入字典。
- **Parallelized evaluation**：总评估可达 **~100 compute-hours/候选**；评估 ** embarrassingly parallel** 时分发到集群，避免 wall-clock 阻塞代数。
- **Multiple scores**：显式 **多目标**；文章观察到即便只关心单一指标，**多目标演化也常提升该指标** —— 因档案中多样「好」的定义能刺激 LLM 提出结构不同的候选。

**VZ 映射**：[`docs/specs/evaluation.md`](../../../../docs/specs/evaluation.md) 的 **六大家族** 可组织为 **cascade**：契约测试（cheap）→ SHADOW replay → 长时窗 probe → cross-generation head-to-head。LLM judge **仅作只读 readout**（R12），**禁止**作为 reward 反灌 Face 或基底（与 workspace 铁律一致）。

### 1.6 Distributed pipeline（§2.6）

异步流水线（asyncio）：controller、LLM samplers、evaluation nodes **并发**；优化目标是 **整体吞吐**（单位预算内提出并评估更多想法），而非单次延迟。

**VZ 映射**：后台评估队列 / audit agent / MemProbe **并行化** 的工程姿态一致 — **不要在 turn 路径塞满演化**。

### 1.7 Ablation（§4）

在两个任务上做消融：**矩阵乘法张量分解**、**kissing number 界**。对比设置包括：

- **No evolution**：反复把同一初始程序喂给 LLM（无档案迭代）。
- **No context in prompts**：不加显式问题上下文。
- **No meta prompt evolution**：关闭 meta-prompt 共演化。
- **No full-file evolution**：只允许演化 loss（对照 FunSearch 式单函数）。
- **Small base LLM only**：只用小模型。

图 8 显示：**各组件均有显著贡献** —— 说明 AlphaEvolve 不是单一「大模型采样」技巧，而是 **档案 + 上下文 + meta-prompt + 全文件 + 强模型** 的 **系统工程叠乘**。

### 1.8 Limitations（§6）

核心限制：**必须有自动化评估器**。数学与计算科学大量问题可满足；**自然科学**等依赖 **人工实验** 的环节 **不在优化重心**（尽管提到可与 AI Co-Scientist 类自然语言假设流水线 **链接**）。

**对 EQ/关系主线的含义**：养成对话 **没有单一可证明正确答案** —— 不能直接复制 AlphaEvolve 的「\(h\) = 硬真理」。必须用 **多代理指标 + 代际比较 + regret / probe** 构造 **软可验证组合**（见 §4）。

---

## §2 进化算法谱系定位（VZ 四轴）

四轴：**Verifier 类型** | **演化对象（控制器在哪）** | **自指环深度** | **多样性机制**。

简表（每条线在仓库中的索引优先指向已有 probe 清单）：

| 谱系 | 代表 | Verifier | 演化对象 | 自指 | 多样性 | R-冲突 | VZ 关联 |
|------|------|----------|----------|------|--------|--------|---------|
| (a) GP 经典 | Koza / Banzhaf | 任务适配度 | 程序树 | 低 | 交叉/变异 | 弱 | 历史背景 |
| (b) FunSearch | Romera-Paredes 2023 Nature | 数学/约束检查 | 单 Python 函数 | 低 | LLM 变异 + 过滤 | 弱 | **源头**：LLM=变异算子 |
| (c) AlphaEvolve | 2506.13131 | 同上 + 工程指标 | 全文件/多语言 | 中（含改训练栈） | MAP-Elites + island | **R4（token 变异）** / **R2（自训 LLM）** | **工程范本** |
| (d) Promptbreeder | Fernando 2023 | 下游任务分 | prompt 串 | 中 | 自引用变异 | **R4** | 反向证据 |
| (e) ELM | Lehman et al. handbook ch. | 仿真回报 | 代码策略 | 低 | LLM 变异 | token 层 | 对照 |
| (f) MAP-Elites | Mouret-Clune 2015 | 行为描述符格 | 任意基因组 | 低 | **QD archive** | 无 | **直接借机制** |
| (g) POET | Wang 2019 | env 内成功 | agent + env | 高 | MCC niche | env 共演化复杂 | R14 niche 隐喻 |
| (h) PAIRED | Dennis 2020 | regret | env 生成器 | 中 | UED | 无 | **R12 形式工具** |
| (i) AlphaGeometry 2 | 2502.03544 | symbolic proof | 构造步骤 | 低 | 搜索树共享 | 弱 | **非对称 proposer/verifier** |
| (j) Darwin-Gödel | Sakana 2505.22954 | 自 Archive | 自身代码 | **极高** | 开放式 | **R8/R10/R15** | **反向证据** |
| (k) AI Co-Scientist | 2502.18864 | 湿实验/锦标赛 | NL 假设 | 中 | tournament | **R4** | 借 tournament，不借 debate |
| (l) ACE | 2510.04618 | 任务成功 | playbook context | 低 | 迭代压缩 | 弱 | **R8 友好演化** |
| (m) AlphaTensor/AlphaDev | Nature | 正确性+延迟 | 汇编/IR | 低 | RL 搜索 | 弱 | 同 **verifiable** 哲学 |

---

## §3 VZ 借鉴三轨

### 3.1 借鉴轨 — EVO-1 … EVO-6

以下每条按 convergence packet 风格拆成五段（现状盲点 / 可落地动作 / 涉及 spec / 预期收益 / 引用论文）。

#### EVO-1 — ModificationGate 候选档案 = MAP-Elites + island

- **现状盲点**：`credit` / gate 路径若只保留「当前最优」提案，易 **早熟收敛**到单一行为类型（与 AlphaEvolve 论文强调的 exploration–exploitation 张力同构）。
- **可落地动作**：在 **gate owner 内部**（非跨模块直接调用）维护 **档案**：键 = **行为描述符**（例如 artifact 类型 × 影响面 × roll-back 成本 bin × 跨家族 eval 轮廓 digest）；值 = **elite 候选**集合。引入 **island**（例如按 rare-heavy / session-medium 分层）与 **限速迁移**。全程 **snapshot 发布**，消费者只读。
- **涉及 spec**：[`docs/specs/credit-and-self-modification.md`](../../../../docs/specs/credit-and-self-modification.md)。
- **预期收益**：同一 ModificationGate 阈值下，**多样性好提案**进入 prompt 上下文，提升 **突破局部最优** 概率；符合 R8（owner 内状态）。
- **引用论文**：AlphaEvolve §2.5；Mouret & Clune MAP-Elites (arXiv:1504.04909)。

#### EVO-2 — Evaluation cascade + LLM-judge 只读

- **现状盲点**：六大家族若「平行罗列」而无 **cheap→expensive** 序，会浪费算力或在噪声指标上过早淘汰真改进。
- **可落地动作**：定义 **固定 cascade**：契约断言 → SHADOW snapshot diff → 长时窗 MemProbe / VZ-MemProbe 子集 → cross-generation replay。可选 **LLM judge** 仅输出 **自然性/简洁性 readout**，写入 evaluation snapshot，**不**进入 credit 主链作 reward。
- **涉及 spec**：[`docs/specs/evaluation.md`](../../../../docs/specs/evaluation.md)。
- **预期收益**：与 AlphaEvolve §2.4 **evaluation cascade** 对齐；降低 **误杀**与 **reward hacking** 面（配合 OA 组）。
- **引用论文**：AlphaEvolve §2.4。

#### EVO-3 — Scenario package = QD behavior archive

- **现状盲点**：场景生成若为 **静态模板列表**，难以覆盖 **regime × 强度 × rupture 阶段** 的组合空间。
- **可落地动作**：在 **scenario 生成工具** 侧引入 **行为描述符**（例如 `(regime_id, stress_bin, repair_stage)`），对每个格子保留 **elite 场景**；新场景由 **变异 + 约束检查** 生成（语义方法 / 结构化变异 —— 禁止关键词驱动业务逻辑，见 workspace 规则）。
- **涉及 spec**：[`.cursor/rules/scenario-package-generation.mdc`](../../../../.cursor/rules/scenario-package-generation.mdc)；[`docs/specs/cognitive-regime.md`](../../../../docs/specs/cognitive-regime.md)。
- **预期收益**：**POET niche** 思想在 **对话评测床** 上的实例化；强化 R14 **持久 regime 身份** 的压力测试覆盖。
- **引用论文**：POET (arXiv:1901.01753)；MAP-Elites；AlphaEvolve 多目标讨论 §2.4。

#### EVO-4 — Meta-prompt 在 expression 层 SHADOW 演化

- **现状盲点**：反思 / 总结类 prompt 长期手工维护，改进节奏慢。
- **可落地动作**：在 **expression-layer / reflection** 的集中模板中增加 ** evolution slot**；仅在 **SHADOW** wiring 下由 background-slow 写入候选模板；对比 snapshot 与 **同一评估 cascade**；永不将演化结果接进 **z_t/β_t** 或 **Face 梯度**。
- **涉及 spec**：[`.cursor/rules/llm-prompt-centralization.mdc`](../../../../.cursor/rules/llm-prompt-centralization.mdc)；[`docs/specs/expression-layer.md`](../../../../docs/specs/expression-layer.md)；[`docs/specs/multi-timescale-learning.md`](../../../../docs/specs/multi-timescale-learning.md)。
- **预期收益**：吸收 AlphaEvolve **meta prompt evolution** 的 **吞吐优势**，而不触碰 R4。
- **引用论文**：AlphaEvolve §2.2。

#### EVO-5 — Proposer / Verifier 不对称共演化（regret 引导）

- **现状盲点**：仅静态 benchmark 无法随 **当前系统弱点** 自适应施压。
- **可落地动作**：**场景 proposer**（生成挑战会话配置）与 **evaluator 只读族** **不同步 fine-tune**；proposer 档案更新规则使用 **PAIRED 式 regret 信号**（或简化版：当前 agent 在 held-out 场景上的失败模式分布）— **引导**下一批场景覆盖 **高 regret 区域**，而非用 evaluator 分数直接当在线 reward。
- **涉及 spec**：[`docs/specs/multi-timescale-learning.md`](../../../../docs/specs/multi-timescale-learning.md)；[`docs/specs/evaluation.md`](../../../../docs/specs/evaluation.md)。
- **预期收益**：结合 AlphaGeometry 2 **验证器硬、提议器探** 的不对称与 PAIRED **UED** 形式；仍守 R12。
- **引用论文**：PAIRED (arXiv:2012.02096)；AlphaGeometry 2 (arXiv:2502.03544)（思想引用）。

#### EVO-6 — Cross-generation winrate（与 DM-7 合并叙述）

- **现状盲点**：开放对话 **无单一标量 reward**；绝对分数不可比。
- **可落地动作**：将 **代际 head-to-head** 与 **hold-out scenario replay** 写入门控 **硬条件**（与 DM-7 一致）；AlphaEvolve 侧提供 **演化 = test-time compute 扩展** 的类比动机。
- **涉及 spec**：[`docs/specs/evaluation.md`](../../../../docs/specs/evaluation.md)；[`docs/specs/credit-and-self-modification.md`](../../../../docs/specs/credit-and-self-modification.md)。
- **预期收益**：**软可验证** 体系下的 **总体进步度量**；服务 R10。
- **引用论文**：OEL / XLand (arXiv:2107.12808)；AlphaEvolve §6（test-time compute 讨论）。

### 3.2 反向证据轨

| ID | 陈述 | 对策 |
|----|------|------|
| B-1 | AlphaEvolve 的变异在 **token/代码空间** | 变异限定 **owner proposal schema**；metacontroller 仍由内禀 PE / structured state 驱动 |
| B-2 | **加速自研 LLM 训练** 的自指环 | **不借鉴**端到端基底更新；与 **R2** 冲突 |
| B-3 | **Darwin-Gödel Machine** 开放式自改 archive | 保留 **分层 gate + bounded + rollback**；见其 C1 轴讨论 [`research/probe/candidates/C1.md`](../../candidates/C1.md) |
| B-4 | **AI Co-Scientist** token-space debate | 借 **锦标赛排序**思想；不借 **多智能体辩论链** |

### 3.3 不借鉴轨

| ID | 陈述 |
|----|------|
| N-1 | **LLM 直接改任意 owner 源码** — 违反 R8 SSOT 与模块边界 |
| N-2 | **演化作为 substrate-owner refresh 主路径** — refresh 保持显式 pipeline / frozen runner |
| N-3 | **Evaluator readout 反灌为学习 reward** — 违反 R12；PE 仍为一级信号（R-PE） |

---

## §4 EQ / 关系主线：无 hard verifier 时的「演化」可行性

### 4.1 软可验证代理组合（替代单一 \(h\)）

- **Cross-generation winrate**（DM-7 / EVO-6）
- **PAIRED 式 regret** 或简化 regret 代理（场景难度 vs 当前策略）
- **CMA 四类探针**（VZ-MemProbe）
- **lifeform vitals** / PE 分布 readout（DM-1）
- **Persona / regime 漂移监控**（OA-10 / C2 工具链）

**无一单独充分**；合取构成 **ModificationGate 证据包**。

### 4.2 演化压力放置边界

| 可演化（候选） | 禁止演化 |
|----------------|----------|
| scenario archive、reflection prompt 模板（SHADOW）、retention 规则候选、regime 门限 **proposal** | 基底权重、Face 梯度、z_t 直接 token 补丁 |

### 4.3 R13 对应

- **SSL 侧**：MAP-Elites / QD **维持行为覆盖**（多样性压力）
- **RL 侧**：cascade eval **选拔**（绩效压力）
- 与 **NL 交替** 叙事一致

### 4.4 `lifeform-evolution` 再声明

见本文开头 **包名歧义澄清**。

---

## §5 落地建议（spec 对位索引）

| EVO | 主 spec |
|-----|---------|
| EVO-1 | `credit-and-self-modification.md` |
| EVO-2 | `evaluation.md` |
| EVO-3 | `scenario-package-generation.mdc` |
| EVO-4 | `llm-prompt-centralization.mdc` + `expression-layer.md` |
| EVO-5 | `multi-timescale-learning.md` |
| EVO-6 | `evaluation.md` + `credit-and-self-modification.md`（与 DM-7 重叠） |

---

## §6 Cross-reference：与仓库现有材料的关系

### 6.1 DM-4（结论级）vs EVO 组（机制级）

- **[DM-4](../../../../docs/moving%20forward/探索方向.md)**（条目 DM-4）取自 AlphaEvolve 的 **结论**：**ModificationGate 必须锚定在可验证 / 可对照的 evaluation** —— 回答「**能不能开门**」。
- **EVO-1 — EVO-5** 取自同一论文及其谱系的 **机制**：档案、cascade、meta-prompt、QD、共演化 —— 回答「**门内如何持续找到更好候选而不早熟收敛**」。

二者 **互补**；实施时常 **先固化 DM-4 门槛**，再引入 EVO 机制 **降低搜索退化风险**。

### 6.2 已有短摘要保持单一真相来源

以下文档已含 **AlphaEvolve 条目或程序搜索系综述**，本专题 **不重复摘要全文**，仅 **机制扩展**：

| 文档 | 角色 |
|------|------|
| [`research/probe/candidates/B3.md`](../../candidates/B3.md) B3-1 | Open-Ended 轴 Top 10 一句定位 |
| [`research/probe/candidates/C1.md`](../../candidates/C1.md) C1-04 | C1 轴 Self-Improvement 入选 |
| [`research/deepmind-author-paper-assessment-2026-05.md`](../../../deepmind-author-paper-assessment-2026-05.md) §2.7 / §A2 | DeepMind 程序搜索系总览 |
| [`research/probe/11_vz_implications.md`](../../11_vz_implications.md) | 按 R-ID 汇总的行动清单（含 B3-1） |

### 6.3 与 [`docs/moving forward/探索方向.md`](../../../../docs/moving%20forward/探索方向.md) 的关系

文件 **[探索方向](../../../../docs/moving%20forward/探索方向.md)** 增加 **EVO 组** 表格式清单（EVO-1…EVO-6）；详细机制论证以 **本文** 为准。

---

## 附录 A — 外部机制核对摘要（避免误述）

以下条目在写作时已对照公开论文 / white paper 叙述；若与最终实现细节有出入，以 **原文** 为准。

| 主张 | 核对依据 |
|------|----------|
| Program database = MAP-Elites **风格** + **island** | AlphaEvolve §2.5 正文 |
| Evaluation cascade / parallel eval / multi-score / LLM feedback | AlphaEvolve §2.4 条目列表 |
| Meta-prompt 存 **独立数据库** | AlphaEvolve §2.2 bullet「Meta prompt evolution」 |
| Ablation：**五项**缺一退化 | AlphaEvolve §4 + Figure 8 说明 |
| Limitation：**自动化 evaluator 必须**；wet lab 非重心 | AlphaEvolve §6 opening |
| MAP-Elites：**行为空间网格 + 每格保留 elite** | Mouret & Clune 2015 原始表述 |
| PAIRED：**三角色**（生成环境的 adversary、protagonist、antagonist）+ **regret** | Dennis et al. 2020（UED 框架） |
| POET：**Minimal Criterion Coevolution（MCC）** + env–agent **配对** | Wang et al. 2019 |
| Darwin-Gödel：**开放式代码自我改进 + archive** — C1 轴列为 **反向证据** | [`research/probe/candidates/C1.md`](../../candidates/C1.md) C1-03 |

### 附录 B — PAIRED（regret-based UED）一句话机制

PAIRED 将 **课程生成** 建模为 **三人博弈**：**对抗环境设计者** 提出环境实例；**主角** 与 **对手**（同一策略族上的参照）在该环境上分别 rollout；**regret**（对手回报 − 主角回报）刻画「当前环境是否把主角置于结构性劣势」。优化 regret（而非单纯难度）可避免「过难不可解」或「过易无信息」—— 这与 VZ 中 **用场景库施压当前系统弱点**、同时 **保证场景可解 / 可对照** 的工程意图一致；落地时必须 **evaluator 只读**，避免把 regret 直接改成 token-space RL reward。

### 附录 C — MAP-Elites（QD）一句话机制

MAP-Elites 在 **行为描述符空间**（低维、人类可解释）上划分网格；对每个格子 **保留该行为区域内至今最优个体**。从而在 **不牺牲精英质量** 的前提下 **覆盖多种行为模式** —— 这正是 AlphaEvolve §2.5 所说「维持多样性」与「改进最优」之间的桥梁，也是 EVO-1 / EVO-3 可直接借用的 **数据结构隐喻**。

---

## 参考文献（精选）

- Novikov A, Vũ N, Eisenberger M, et al. *AlphaEvolve: A coding agent for scientific and algorithmic discovery*. arXiv:2506.13131, 2025.
- Romera-Paredes B, et al. *Mathematical discoveries from program search with large language models*. Nature, 2023. (FunSearch)
- Mouret J-B, Clune J. *Illuminating search spaces by mapping elites*. arXiv:1504.04909, 2015.
- Wang R, Lehman J, Clune J, Stanley K O. *POET: Paired Open-Ended Trailblazer*. arXiv:1901.01753, 2019.
- Dennis M, Jaques N, et al. *PAIRED: Emergent Complexity via Unsupervised Environment Design*. arXiv:2012.02096, 2020.
- Fernando C, et al. *Promptbreeder: Self-referential self-improvement via prompt evolution*. arXiv:2309.16797, 2023.
- Lehman J, et al. *Evolution Through Large Models*. Handbook chapter / Springer, 2024.
- Stooke A, et al. *Open-Ended Learning Leads to Generally Capable Agents*. arXiv:2107.12808, 2021.

---

**文档结束。**
