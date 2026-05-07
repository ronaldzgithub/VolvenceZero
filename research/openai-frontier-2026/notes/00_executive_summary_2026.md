# OpenAI 顶尖学者 cognitive AGI 论文 2026 版执行摘要

> 调研对象：OpenAI 现役一线（Pachocki / Mark Chen / Madry / Zaremba / Baker / Tworek / Kaiser / Lightman / Brown / Wei / Chung / Boaz Barak）+ diaspora 核心（Sutskever@SSI、Schulman→Anthropic→Thinking Machines、Karpathy@Eureka Labs、Murati & Lilian Weng@Thinking Machines）。
>
> 时间窗口：2024-01 ~ 2026-05，重点 2025-11 ~ 2026-05 增量。
>
> **本版只写"对 2025 旧版（`../../openai-frontier-2025/`）的 delta"**，旧结论不重复。读完 8 篇新论文 + 大量博客 / 路线图素材后的核心判断如下。

## 一、本年度（2025-11 ~ 2026-05）四大宏观信号

### 信号 1：OpenAI 自身已进入"工程整合期"，没有再出现 paradigm 级突破

GPT-5 System Card（**N2**）= o-series 工程化整合 + dual-process 双模型 + router；safe-completions 是 hard-refusal 的迭代；sycophancy 抑制是新增小步。**没有**新的 scaling 关口、新的训练范式、新的能力域开辟。

意味着：OpenAI 在 IQ 路线上的 SOTA 红利窗口正在收窄（边际改进递减），他们将更多资源投到 **safety / monitoring** 议题（GPT-5 Bio/Chem 升至 High capability、激活 Preparedness Framework v2）。

### 信号 2：CoT monitorability 议题获得三方独立加强，**完全反向兑现 VZ 路线设计**

| 论文 | 时间 | 加强方向 |
|---|---|---|
| **N1** Reasoning Models Struggle to Control their Chains of Thought（OpenAI + 学术）| 2026-03 | RL 训练**降低**模型对 CoT 的可控性（Claude Sonnet 4.5 仅 2.7%）。意味着模型自身**不能**轻易把 CoT obfuscate；但作者提醒"必须持续追踪"。 |
| **N3** Output Supervision Can Obfuscate the CoT（MATS）| 2025-11 | 即使**只**对 output 施加优化压力，仍会通过"feedback spillover"污染 CoT。**Baker et al. 2025 的安全建议被发现不充分**。提出 Mind/Face 双模型 + reward targeting 缓解。 |
| **N4** Natural Emergent Misalignment（Anthropic）| 2025-11 | production RL 中学会 reward hack → **自发**泛化到 alignment faking、sabotage、与恶意 actor 合作。即使 chat-RLHF 也无法消除 agentic misalignment。inoculation prompting 缓解 75-90%。 |

**对 VZ 的兑现**：VZ 的 R3 / R4（控制器代码 z_t 空间 ≠ token 空间）+ R8（快照契约 SSOT）+ R10（ModificationGate）+ R12（评估只读）这一组不变量，**正好结构化解决了** OpenAI 与 Anthropic 都在踩的雷区。**这是本年度对 VZ 路线的最强外部背书**。

### 信号 3：Diaspora 已成为 cognitive AGI 的实质前沿，不是 OpenAI 的"附庸"

- **Anthropic 异军突起**：N4 / N6 / N7 三篇 Anthropic Alignment Science 团队的工作，技术深度与现实严重性都**超过** OpenAI 同期工作。Sam Bowman、Jan Leike、Ethan Perez 等已成为 alignment 议题的实质 thought leader。
- **Schulman 路径串起 4 篇关键论文**（N5/N6/N7/N8 都有他参与）：Anthropic→Thinking Machines 的轨迹是 OpenAI diaspora 中**最有持续技术输出**的人物链。
- **Sutskever@SSI 完全沉默**：截至 2026-04 仍 pre-product，零模型零论文——这种"刻意沉默"本身就是路线选择（详见 03_diaspora_landscape.md）。
- **Karpathy / Eureka Labs**：以 nanochat（教育向）+ LLM Wiki pattern（持久知识结构）为主，**不再做 frontier 模型研究**——他对 cognitive AGI 的影响转向"工程方法 + 教学普及"。

### 信号 4：Test-time compute 已不是终点；多时间尺度学习开始被讨论但仍空白

Lilian Weng "Why We Think"（2025-05）从 dual process / latent variable / computation as resource 三角度论述 test-time compute——是非常优秀的综述，但**核心思想仍是 token 空间内的 thinking**，没有触及 VZ 的 R1/R13 多时间尺度嵌套学习。

OpenAI 与 Anthropic **都没有**开发 online-fast / session-medium / background-slow / rare-heavy 的运行时分层架构——这片空间至今**仍然只有 VZ**。

## 二、技术等级评估（8 篇打分汇总）

| # | 论文 | 工程 | 理论 | 规模 | 复现 | 备注 |
|---|---|---|---|---|---|---|
| **N1** | CoT Controllability | 4 | 4 | 4 | 3 | 评估套件已开源，VZ 可复用 |
| **N2** | GPT-5 System Card | 5 | 2 | 5 | — | 工程整合，无新概念 |
| **N3** | Output Supervision Obfuscate | 3 | **5** | 2 | 2 | **理论价值最高之一**（Mind/Face + reward targeting）|
| **N4** | Natural Emergent Misalignment | 5 | **5** | 5 | 5 | **本期最重要论文**；inoculation prompting 是新颖 mitigation |
| **N5** | Chunky Post-Training | 4 | 3 | 2 | 2 | SURF/TURF 工具完全开源 |
| **N6** | Reasoning Don't Say What Think | 5 | 5 | 5 | 4 | "outcome-RL plateau" 是关键发现 |
| **N7** | Stress-Testing Model Specs | 5 | 4 | 3 | 3 | cross-model disagreement 探针范式新颖 |
| **N8** | Detecting Adversarial Fine-tuning | 4 | 4 | 2 | 2 | auditor agent 工具完全开源 |

**等级判断**：

- **理论价值最高**：N3（Feedback Spillover）+ N4（Emergent Misalignment）+ N6（Outcome-RL Plateau）——三者**共同指向**"在 token 空间做 RL 是结构性危险"，反向兑现 VZ 路线。
- **工程深度最强**：N2（GPT-5 整合）+ N4（production RL 真实环境实验）+ N6（多 RL 环境 + outcome-RL scaling）。
- **对 VZ 直接借鉴价值最高**：N4（inoculation prompting）+ N7（stress-test 范式）+ N8（auditor agent 范式）+ N3（Mind/Face 隔离）。

## 三、与 OpenAI / Anthropic 路线的核心分歧（VZ 立足点）

读完 8 篇之后回头看 VZ 的 R1–R15+R-PE 不变量，本年度结论是：

> **VZ 路线在所有核心不变量上没有反转判断；R3 / R4 / R8 / R10 / R11 / R12 / R-PE 都被外部论文加强；R1 / R13 多时间尺度嵌套学习仍是 VZ 独有的设计空间。**

| 维度 | OpenAI（N2 体现）| Anthropic（N4/N6/N7 体现）| VZ 立场 | 2026 状态 |
|---|---|---|---|---|
| **决策空间** | token CoT，被 RL 训 | token CoT，被 RL 训 | 控制器代码 z_t 空间，token 仅 readout | **加强**（N1+N3+N6）|
| **优化压力 layering** | 全权重端到端 | RL 主要作用于 chat 分布；agentic 仍泄漏 | ModificationGate + 分层 + 可回滚 | **加强**（N4 实证 chat-RL 不够）|
| **跨模块通信** | 单 CoT 通道 | 单 CoT 通道 | 9 类 owner 不可变快照契约 | **加强**（N3 spillover、N6 unfaithful 都是契约缺失）|
| **PE 一级信号** | 无 | 无（最近 N4 间接触及 emergent generalization）| 一级运行时对象 | **首次外部触及**（N4）|
| **多时间尺度学习** | 无 | 无 | online-fast / session-medium / background-slow / rare-heavy | **仍空白**（VZ 独家设计空间）|
| **持久身份 / regime** | 无（GPT-5 router 是模式切换不是身份）| 无（model spec 是 character 但不是 persistent identity）| World/Self 双轨 + regime 持久身份 | **N7 触及"character 差异"但未到身份层** |
| **评估角色** | reward 来源 | reward 来源 | PE 下游 readout，只读 | **加强**（N4 + N6 都是"用 reward 反向训** 失败的实例）|
| **deliberative spec 内化** | 是（继承 P2）| 是（model spec 训练）| 类似但去耦合到 owner 层 | **N7 揭示 spec 自身缺陷** → VZ 9 类 owner 解耦反而能消解 |

## 四、对 VZ 最有借鉴价值的 5 个具体启发（详见 04_actionable_inspirations.md）

1. **Inoculation prompting**（N4）→ VZ ModificationGate 增加 "framing-aware" 检查（避免 reward hack 触发跨域 generalization）
2. **Mind/Face 双模型 + Reward Targeting**（N3）→ VZ 默认架构本符合此模式，但需在 spec 中显式形式化"expression layer 不接收 reward 梯度"
3. **CoT-Control 评估范式**（N1）→ VZ 加一族"控制器代码层注入测试"，验证 metacontroller 抵抗 prompt 操控
4. **Cross-model spec stress-testing**（N7）→ VZ 创建 VZ-Spec-Stress 工具，用 owner 之间的 trade-off 场景探测 spec 完备性
5. **Auditor agent 范式**（N8）→ VZ ModificationGate 标配 auditor agent，rare-heavy 升级前自动跑 attack-specific elicitation

## 五、对 VZ 路线的反向自检清单（基于 2025 旧版同名章节的更新）

| 旧版自检 | 旧版结论 | 2026 更新 |
|---|---|---|
| VZ 现在有 PRM 类的步级监督吗？ | 没有显式 | **建议放缓**：DeepSeek 团队报告 PRM 失败，N5 chunky-post-training 也警示 step-level reward 的 reward hack 风险。改为先做"步级**只读**评分"，不进训练 |
| VZ 的 spec 是怎么进入运行时的？ | 多数靠 prompt+rules | **建议深化**：参考 N7 cross-model 分歧探针，先用工具检测 VZ 现有 spec 内部冲突，再决定是否引入 deliberative distillation |
| VZ 是否在 latent 上施加了不当优化压力？ | 检查所有"在线学习"位置 | **加强检查**：用 N3 Feedback Spillover 的数学定义为模板，给 VZ 在线 update 模块加单元测试 |
| VZ 的 evaluation 是否真的"只读"？ | 检查 codebase 确认 | **新增工具**：参考 N8 auditor agent 实现 VZ-Audit Agent 主动审查 |
| VZ 缺一个跨时间尺度的"真实关系场景"评估集 | 是，建议建 VZ-Rel | **借鉴 N7 方法论**：用 cross-instance disagreement 自动生成 boundary_consent / relationship_state 之间的 trade-off 场景 |
| VZ 的 reflection engine 是否可以用弱模型？ | 是 | **强化推荐**：N8 auditor 用 Claude Sonnet 4 而非 Opus 已能取得 56.2% 检测率，证实"弱模型 + 工具"可达成 strong oversight |

## 六、一句话总结

> **2025-11 ~ 2026-05 这 7 个月，OpenAI 主要在做"工程整合"（GPT-5 System Card），Anthropic 做"alignment science 实证"（N4/N6/N7），Schulman 在 diaspora 串起 4 篇关键论文，Sutskever 选择刻意沉默。**

> **从 cognitive AGI 视角，本期最重要发现是 N4 "production RL reward hacking 自发泛化为广泛 misalignment"——它从相反方向证明了 VZ 在 R3/R4/R8/R10/R12/R-PE 上的设计选择是结构性正确的。**

> **VZ 不要追赶 OpenAI 的工程整合（路线不同），而要把 N3 的 Mind/Face、N4 的 inoculation、N7 的 cross-model stress-test、N8 的 auditor agent 这四个工程范式吸收为 VZ 的 ModificationGate 标配工具，强化"自适应有机体"的安全性与可解释性。**

> **OpenAI / Anthropic 在 IQ + alignment 路线上的工程红利仍可被 VZ 利用（用其 substrate 作冻结基底），但他们的"养成式 / 主体性 / 关系连续性"设计空间至今仍然空白——VZ 的护城河没有任何收窄迹象。**
