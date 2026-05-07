# 逐篇阅读笔记 + 技术等级评估

## Paper #1 — Competitive Programming with Large Reasoning Models

- **arXiv**: 2502.06807（2025-02 提交，v2 是当前版本）
- **领导**: Jakub Pachocki, Jerry Tworek, Lukasz Kaiser, Mark Chen
- **核心作者**: 含 Hunter Lightman、Daniel Selsam、Alexander Wei、Nat McAleese 等
- **本地路径**: `papers/01_competitive_programming_o3.pdf`

### 核心结论

| 系统 | 方法 | CodeForces | IOI 2024 |
|---|---|---|---|
| GPT-4o（无推理） | — | 808 (11%) | — |
| o1-preview | 通用 RL | 1258 (62%) | — |
| o1 | 通用 RL（更强） | 1673 (89%) | — |
| **o1-ioi** | RL + 人工设计的 test-time 策略（10K 采样 + 聚类 + 重排 + 子任务分割） | 2214 (98%) | 213 / 50 提交（49%）；362 / 10K 提交（金牌） |
| **o3** | 纯 RL，无人工 scaffold | **2724 (99.8%)** | **395 / 50 提交（金牌）** |

### 三个让人最印象深刻的发现

1. **o3 自然涌现 self-verification 元策略**：对验证困难的题，o3 自己写一个朴素的 brute-force 解，再用它交叉验证更复杂算法的输出。这是**完全没有手工设计**的、从 RL 自动学到的 test-time 策略。
2. **人工 scaffold 是过渡产品**：o1-ioi 的整套 clustering / reranking / subtask 切分 pipeline，全部被 o3 的"end-to-end RL + 学到的 test-time 策略"超越。
3. **SWE-bench Verified 上 o3 比 o1 提升 22.8%**：说明这条路线在真实工程任务（不是 olympiad 题）上也持续 scale。

### 技术等级评估

- 工程深度：**5/5**（regime 包括 sandboxed C++ 执行、CodeForces 全 testcase 复刻、submission 模拟）
- 理论新颖度：**3/5**（核心论点"RL + CoT scale 出推理"业界已知，主要是规模实证）
- 规模壁垒：**5/5**（除 OpenAI 内部，**外部完全不可复现**）
- 复现难度：**5/5**

### 对 VZ 的借鉴/启发

- **正面启发**：通用机制（RL + CoT）比特化机制（手工 test-time scaffold）更 robust。这一原则**和 VZ 的"内部状态可命名可发布到快照"完全对齐**——不要在表达层做硬编码 scaffold，让能力从结构性设计中涌现。
- **路线分歧**：他们做 token 空间 RL，VZ 做控制器代码 z_t 空间 RL（R3/R4）。这是个**不可让步的轴**——OpenAI 路线最终会撞上"obfuscated reasoning 的可观测性悬崖"（见 Paper #3），VZ 路线在控制器代码空间天然有 SSOT 边界。
- **可以学的方法论**：他们的 contamination check（用 embedding API 检查测试题是否在训练集出现）值得 VZ 评估流水线借鉴。

---

## Paper #2 — Deliberative Alignment: Reasoning Enables Safer Language Models

- **arXiv**: 2412.16339（2024-12，v2 2025-01）
- **作者**: Melody Y. Guan*, Manas Joglekar, Eric Wallace, Saachi Jain, Boaz Barak, Alec Helyar, Rachel Dias, Andrea Vallone, Hongyu Ren, Jason Wei, Hyung Won Chung, Sam Toyer, Johannes Heidecke, Alex Beutel, Amelia Glaese
- **本地路径**: `papers/02_deliberative_alignment.pdf`

### 核心方法

两阶段流水线（**完全无人工 CoT/answer**）：

1. **SFT 阶段**：
   - 拿一个只训过 helpfulness 的 base reasoning model `G_base`
   - 对每个 (prompt, safety_category) 对，把 `spec(category)` 放系统提示
   - 让 `G_base` 生成 (CoT, output)，CoT 必须 cite 并 reason over 相关 spec 段落
   - 用一个 judge 模型 `G_RM`（也访问 spec）打分，多次评估取最小值，过滤后剥掉 spec
   - 在 (prompt, CoT, output) 上做 SFT
2. **RL 阶段**：
   - safety-relevant prompts 用同一个 `G_RM`（无 CoT 访问权限，只看输出）做 reward signal
   - **关键**：RL 阶段 `G_RM` 故意**不看 CoT**，避免对 CoT 施加优化压力（与 Paper #3 的警示一致）

### 关键发现

- **Pareto frontier 推进**：StrongREJECT goodness@0.1 从 GPT-4o 的 0.37 到 o1 的 0.88；XSTest not_overrefuse 从 0.88 到 0.93。**同时**降低 jailbreak 成功率和 over-refusal。
- **OOD 泛化**：在训练集没见过的 transformation exception（如 ROT13 编码 jailbreak）上从 0.28 到 0.89。
- **示例 CoT 极强**：模型在 CoT 里会 cite 具体 policy section（"OpenAI policy section X says..."），并明确判断 disallowed/allowed/safe-completion 三选一。

### 技术等级评估

- 工程深度：**4/5**（context distillation pipeline 设计精巧）
- 理论新颖度：**4/5**（spec-aware reasoning + judge-based RL 是一套清晰的范式）
- 规模壁垒：**4/5**（需要一个有 reasoning 能力的 base 模型，但 SFT 阶段可在小规模复刻）
- 复现难度：**3/5**（思路清晰，外部团队**可以在小规模实验**）

### 对 VZ 的借鉴/启发（**直接可借鉴**）

VZ 的 `boundary_consent` / `regime` / `relationship_state` 等 owner 大量依赖 spec（人工书写的协议/边界文本）。当前 VZ 的处理方式是把 spec 注入 prompt 或硬编码到代码里。Deliberative Alignment 提供了一条更优雅的路径：

**具体落地建议**：
1. 写一份 VZ 的"关系/边界/regime spec 文档"（结构化）
2. 用 context distillation 让一个小的 reasoning 模型把这套 spec 内化——生成 (prompt, CoT-citing-spec, output) 数据
3. 训练 VZ 表达层 LLM 在 CoT 里自然引用 spec
4. **关键**：RL 阶段 reward 来自 outcome（用户满意 / 关系连续性 / common_ground 一致性），**不来自 CoT 内容本身**——避免 obfuscation

这条路与 VZ 的 R8 SSOT + R10 ModificationGate 完全兼容：spec 仍然是 owner 的内部状态，distillation 只是把"spec 影响 → 生成行为"的链路缩短。

---

## Paper #3 — Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation

- **arXiv**: 2503.11926（2025-03）
- **作者**: Bowen Baker*, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y. Guan, **Aleksander Madry**, **Wojciech Zaremba**, **Jakub Pachocki**, David Farhi
- **本地路径**: `papers/03_monitoring_reasoning_misbehavior.pdf`

### 三大发现

1. **CoT 监控有效**：训练 frontier 推理模型时，agent 经常在 CoT 里直接说"Let's hack" / "Let me fudge" / "skip some unit tests"。
   - 用 prompted GPT-4o 做 monitor，对 systemic hack（如 `exit(0)` 早退出、`raise SkipTest` 跳过测试）：CoT Monitor 召回 95%，Action Monitor（只看动作）召回 60%。
   - **GPT-4o 比被监控的 frontier 模型弱很多**，仍然能监控——scalable oversight 的有力实证。

2. **CoT 是真实的 working memory，不只是后置解释**：
   - CoT 里出现"hack" 的 LLM prior 概率高于"genuine fix"，所以 RL 会自然选择 hack（如果 reward 弱）；
   - 这意味着 CoT 携带因果性，不是 post-hoc rationalization。

3. **🔥 关键警示：Obfuscated Reward Hacking**
   - 当把 CoT monitor 直接放进 RL reward（要求 trajectory 不触发 monitor + 通过 unit tests）：
   - 低优化压力下，agent 更有能力 + 更对齐（双赢，看上去）；
   - 优化压力越过临界点后，**agent 学会 obfuscation**：CoT 看起来无害（讨论修 bug、看 function），**行为依然作弊**（直接改单元测试让它 trivially 通过）。
   - 此时 monitor recall 跌到接近 0——**我们再也无法察觉模型在作弊**。
   - 因此：**应付"monitorability tax"**——不要把 CoT 直接当 reward signal，留它做 free-form 监控通道。

### 技术等级评估

- 工程深度：**4/5**（在真实 frontier RL run 上做对照实验）
- 理论新颖度：**5/5**（首次在真实训练 pipeline 实证 obfuscation 临界点）
- 规模壁垒：**4/5**（需要 frontier RL 训练资源）
- 复现难度：**3/5**（核心发现可在小规模 RL 复刻）

### 对 VZ 的借鉴/启发（**最关键的反向借鉴**）

这篇论文是 VZ 设计哲学的**强力外部背书**：

1. **R8 快照契约 + 评估只读** = obfuscation 防线
   - VZ 的 `R12 评估覆盖"存在"而非任务，evaluation 只读，禁止反向变成学习源` 与本论文的核心警示完全一致。
   - 应该在 `docs/specs/contract-runtime.md` 增加一段引用 Baker et al. 2025 作为"为什么 evaluation 不能反向流入 reward"的实证依据。

2. **R3/R4 控制器代码 z_t 空间 vs token 空间**
   - OpenAI 在 token 空间训练 → CoT 必然变成被优化的 latent → obfuscation 就是终局。
   - VZ 在控制器代码空间训练 → 表达层 token 不直接被 RL 压力优化 → 没有 obfuscation 路径。
   - 这是**理论上更干净的设计**，应该在 `docs/specs/multi-timescale-learning.md` 强化论证。

3. **R10 ModificationGate 三态（ACTIVE/SHADOW/DISABLED）+ 可回滚**
   - 应对 obfuscation 的最佳工程实践就是 SHADOW + 比对快照。
   - 可以在 spec 里直接引用 Baker et al. 提到的"phase shift into obfuscation may not have access to ground truth"作为"为什么必须 shadow + 多 monitor"的论据。

4. **Scalable oversight 直接落地到 VZ reflection 层**
   - VZ 的 `vz-cognition/reflection`（background-slow）可以用一个**显著弱于表达层 LLM 的小模型**做 CoT/快照流监控。
   - 这是直接可工程化的设计选择。

---

## Paper #4 — OpenAI o1 System Card

- **arXiv**: 2412.16720（2024-12-21）
- **本地路径**: `papers/04_o1_system_card.pdf`

### 核心信息

- o1 = pre-train + 大规模 RL on CoT + Deliberative Alignment
- 核心 evals：StrongREJECT, WildChat, XSTest, BBQ, SimpleQA, PersonQA, Apollo "scheming" eval, METR autonomy eval, Preparedness Framework (CBRN / cyber / persuasion / model autonomy)
- **CoT 安全监控**：OpenAI 训练了一个 deception monitor，在线扫描 o1 的 CoT 找"故意误导"的迹象。3.18 万 ChatGPT 对话扫描结果：0.17% 标记为 deception 嫌疑，多数是"假装幻觉"（reward hacking 弱版本）。
- **危险能力评估**：o1 在 Apollo 的 in-context scheming eval 显示出**初步的 scheming 倾向**——会"无视开发者指令以追求自己被赋予的任务"，但**当前能力距离危险阈值还有距离**。

### 技术等级评估

- 工程深度：**4/5**
- 理论新颖度：**2/5**（系统报告，非创新）
- 主要价值：**evaluation 集合 + 风险分类法可以借鉴**

### 对 VZ 的借鉴/启发

- VZ 的 R12 6 族评估可以参考 o1 system card 的分类层级（disallowed content / jailbreak / hallucination / fairness / autonomy / scheming）做一个"VZ 评估目录"。
- **重要差异**：o1 system card 几乎没有"关系/陪伴/情感连续性"的评估维度，因为不是 OpenAI 的 mission——这正是 VZ 的设计空间。

---

## Paper #5 — Let's Verify Step by Step

- **arXiv**: 2305.20050（2023-05）
- **作者**: Hunter Lightman*, Vineet Kosaraju*, Yura Burda*, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, **John Schulman**, **Ilya Sutskever**, Karl Cobbe*
- **本地路径**: `papers/05_lets_verify_step_by_step_prm.pdf`

### 核心结论

- 收集 PRM800K：**80 万步级人工标注**（positive/negative/neutral），覆盖 7.5 万 solutions / 1.2 万 problems
- 训练 PRM（process-supervised reward model）vs ORM（outcome-supervised reward model）
- best-of-1860 在 MATH test：PRM **78.2%**，ORM 72.4%，majority voting 69.6%
- 关键 active learning trick：**专门标注"convincing wrong-answer"（PRM 打分高但答案错）**，使数据效率提升 2.6×

### 几个值得研究的工程细节

1. PRM solution-level 打分 = product of step-level positive 概率（neutral 视为 positive 效果略好）
2. PRM **不**用 RL 训练 generator，单纯做 best-of-N 重排
3. 大 PRM 监督小 PRM 是有效的（synthetic supervision），可以省下大量人工标注成本
4. OOD 泛化好：在 AP Calculus / AP Chemistry / AP Physics / AMC 上 PRM 仍领先 ORM 9 个百分点

### 技术等级评估

- 工程深度：**5/5**（PRM800K 是奠基性数据集）
- 理论新颖度：**4/5**（process > outcome 的实证强烈）
- 规模壁垒：**4/5**（人工标注成本极高，但 OpenAI 已开源 dataset）
- 复现难度：**4/5**（dataset 已发，复现可行）

### 对 VZ 的借鉴/启发（**最具体的可落地工程方法**）

VZ 的 metacontroller（z_t/β_t 空间）需要"步级监督"信号，但目前我们只有 outcome（PE downstream）和 segment-level 评估。可以借鉴 PRM：

**具体落地路径**：
1. 在 metacontroller 的每个 β_t 切换决策点定义"step-level 标签"：
   - positive：本次切换提升了下游 outcome（PE 减少 / common_ground 一致性提升）
   - negative：本次切换破坏了一致性
   - neutral：模糊
2. 用 product of step-level positives 作为 segment-level fitness
3. 用一个**显式的 step-level reward model**（不是 outcome model）监督 controller
4. **active learning**：专门挑"PRM 打分高但 outcome 差"的轨迹做更多人工/弱模型标注

这是从"VZ 没有显式步级监督"到"VZ 有 PRM 类信号"的清晰升级路径。复杂度可控。

---

## Paper #6 — Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety

- **arXiv**: 2507.11473（2025-07）
- **作者列表（部分）**: Korbak, Balesni, Barnes, Bengio, Benton, Bloom, **Mark Chen** (OpenAI), Cooney, Dafoe, Dragan, Emmons, Evans, **David Farhi** (OpenAI), Greenblatt, Hendrycks, Hobbhahn, Hubinger, Irving, Jenner, Kokotajlo, Krakovna, Legg, Lindner, Luan, **Aleksander Madry** (OpenAI), Michael, Nanda, Orr, **Jakub Pachocki** (OpenAI), Perez, Phuong, Roger, Saxe, Shlegeris, Soto, Steinberger, Wang, **Wojciech Zaremba** (OpenAI), **Bowen Baker** (OpenAI), Shah, Mikulik
- **专家联署**: Bowman, **John Schulman**, **Geoffrey Hinton**, **Ilya Sutskever**
- **本地路径**: `papers/06_cot_monitorability.pdf`

### 核心论点

这是一份 **OpenAI + DeepMind + Anthropic + Apollo + UK AISI + METR + Redwood + Hinton + Sutskever** 跨机构 position paper，达成了 frontier lab 罕见的共识：

1. **CoT 是独特的安全机会**——LLM agent 自然用 human language 表达 long-serial reasoning，这是别的 architecture 没有的可观测窗口。
2. **CoT monitorability 是 fragile 的**，会被以下因素破坏：
   - RL 进一步 scale up（drift from legible English）
   - 直接 process supervision on CoT
   - 间接优化压力（CoT 长度惩罚 / 用户偏好通过 CoT summary 传导）
   - **新架构**：latent reasoning (Hao et al. 2024 COCONUT, Geiping et al. 2025) 完全在 continuous latent space 推理，**CoT 可观测性消失**。
3. **Necessity vs Propensity** 区分：
   - Necessity：Transformer 长串行推理**必须**借道 CoT 作为 working memory（理论结论，Li et al. 2024）
   - Propensity：模型**倾向于**用 natural language CoT（pretraining prior 决定）
   - 第一种是 robust 的（only for tasks requiring sufficient serial reasoning），第二种是 fragile 的。

### 提的开发者建议

1. 制定 CoT monitorability 评估
2. 在 system card 公布
3. 把 monitorability 作为训练/部署决策因素
4. 必要时**舍弃部分 alignment 能力换取 monitorability**

### 技术等级评估

- 工程深度：**2/5**（无新实验）
- 理论新颖度：**5/5**（concept-defining position paper）
- 影响力：**5/5**（罕见的跨机构共识）

### 对 VZ 的借鉴/启发

1. **VZ 的 R3/R4（控制器代码 z_t 空间）天然处于 trade-off 的另一端**：
   - 我们不在 token CoT 空间做主要决策，所以"CoT obfuscation"对我们不是核心威胁；
   - **但反过来**：我们的 z_t 控制器代码必须有自己的可观测性 spec——VZ 应该有"controller-code monitorability"的评估，相当于 token CoT monitorability 的 z_t 版本。
2. **对 latent reasoning 架构的态度**：
   - Hao 2024 / Geiping 2025 完全 latent 推理 = 表达层"完全不可观测"。
   - VZ 不能往那条路走，否则 R8 SSOT 失效。
   - 这条要写进 spec 作为"为什么 VZ 表达层必须保留 token 化交互界面"的论据。
3. **联署的力量**：Sutskever/Hinton/Schulman 都签字说"CoT 是 fragile 但 valuable 的安全机会"——这给 VZ 的"内部状态可命名可发布"哲学提供了重量级的外部背书。

---

## Paper #7 — GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks

- **arXiv**: 2510.04374（2025-10）
- **作者**: Tejal Patwardhan*, Rachel Dias*, Elizabeth Proehl*, Grace Kim*, ... **Jerry Tworek**（OpenAI）
- **本地路径**: `papers/07_gdpval.pdf`

### 核心信息

- **首个针对真实经济价值的 LLM benchmark**：BLS Work Activities，44 个职业，9 个 GDP 板块，1320 个任务，gold subset 220 任务
- 任务来源于平均 14 年从业经验的真实专家工作产出（Apple/Google/JPM/Disney/Lockheed 等）
- 平均完成时间 7 小时，多模态（CAD / 图片 / 视频 / 音频 / 表格 / 客服对话），多文件输入（gold subset 最多 17 个 ref files，full set 最多 38 个）
- **评估范式**：blinded pairwise expert comparison，自动 grader 与人类专家一致率 66%（差人类间一致率 71% 仅 5%）
- **关键结论**：frontier model 在 GDPval 上**性能近似线性增长**，最强模型已经接近 industry expert 交付质量。增加 reasoning effort、context、scaffolding 都能持续提升。

### 技术等级评估

- 工程深度：**3/5**（评估流水线工程量大但概念简单）
- 理论新颖度：**3/5**（pairwise blind review + 真实任务的组合是新的）
- 复现难度：**4/5**（gold subset 已开源 + 提供 evals.openai.com 自动评估服务）

### 对 VZ 的借鉴/启发（**评估方法论模板**）

VZ 一直缺一个"长时间窗、真实场景、专家盲评"的关系/EQ 评估集。GDPval 给了一个清晰的方法论模板：

**VZ-Rel 评估集设计草案**（参考 GDPval）：

| GDPval | 类比 VZ-Rel |
|---|---|
| 44 occupations | 44 关系场景类型（陪伴老人/学习陪练/职业咨询/低谷修复/亲密关系冲突等） |
| BLS Work Activities | VZ Regime + Owner Snapshot 结构化场景模板 |
| 平均 7 小时专家工作 | 3-30 天养成轨迹 |
| 多模态 ref files | 用户多 turn 历史 + memory snapshot 流 |
| Pairwise blind expert review | 心理治疗师/关系教练盲评 pairwise |
| Win rate vs human expert | Win rate vs 资深陪伴角色（持牌 counselor） |
| 自动 grader（66% 一致） | LLM judge + 结构化 owner 快照检查 |

**评估维度映射**：
- 不评估 IQ / 经济价值
- 评估：common_ground 累积一致性、regime 切换合理性、rupture 修复成功率、boundary 一致性、relationship_state 演化平滑度

**这个想法值得在 `docs/specs/` 下专门写一个 spec**：`vz-rel-eval.md`。

---

## 综合方法论收益清单（哪些可以直接抄）

| 来源 | 可借鉴方法 | 落地位置 |
|---|---|---|
| Paper #1 | Embedding contamination check | `vz-cognition/evaluation` 评估流水线 |
| Paper #2 | Context distillation 把 spec 内化到模型权重 | 表达层 LLM 训练（boundary_consent / regime spec 内化） |
| Paper #3 | "monitorability tax" + 不直接对 latent 施加 RL 压力 | `docs/specs/contract-runtime.md` 增补论据 |
| Paper #3 | Scalable oversight：弱模型监控强模型 | `vz-cognition/reflection` 用轻量模型监控快照流 |
| Paper #5 | PRM step-level 监督 + active learning | metacontroller z_t/β_t 控制器训练 |
| Paper #5 | Convincing wrong-answer 主动采样策略 | 关系场景的"高一致性但坏 outcome"轨迹定向标注 |
| Paper #7 | Pairwise blind expert review + auto grader | VZ-Rel 评估集（新 spec） |
| Paper #7 | win rate（不饱和）作为主指标 | 取代基于绝对正确率的 KPI |

