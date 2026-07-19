# humans& — 深度分析

- **分组 / 成熟度**：D 前沿架构（**对 VZ 最直接相关**：以人为中心 / 关系 / EQ + 信任）｜ 成熟度早期（2026-01 出 stealth，$480M 种子；research-first，尚无第一方论文/产品）
- **一句话主张**：把 AI 重新围绕"人与人之间的关系"构建——AI 作为强化组织与社区的"连接组织"，让人互相理解、建立信任、协作；核心创新在 long-horizon / multi-agent RL + memory + 用户理解。
- **主要创作者 + 血统**：Eric Zelikman（STaR/Quiet-STaR，前 xAI）、Noah Goodman（RSA 概率语用 / ToM）、Andi Peng（Pragmatic Feature Preferences；前 Anthropic post-training）、Yuchen He、Georges Harik、Ray Ramadorai。
- **为何与 VZ 共振**：这是名册里与 VZ"关系与主体性（EQ + 信任）优先于 IQ"愿景**最贴合的一家**。本地 4 篇均为创始人奠基作（公司无本名论文）。但仍以同等红队纪律审视——尤其 Quiet-STaR 的 **token 级思考**对 R4（控制在 z_t 而非 token 空间）构成微妙挑战。

## 1. 核心逻辑（论文级 · PDF-grounded）

### STaR: Bootstrapping Reasoning With Reasoning（2203.14465, 2022）
- **问题**：让模型自学推理，但缺乏大规模 rationale 标注。
- **方法/机制**：自举闭环——①few-shot 提示模型生成 rationale + 答案；②**只保留答对的 (rationale, answer)**；③对答错题给出正确答案做"**rationalization**"（反推 rationale）；④在收集到的 rationale 上微调；迭代。
- **关键结果**：在算术/CommonsenseQA 等上显著提升，逼近大得多的模型；用一个**可验证的正确性门**筛选自生成数据。
- **局限**：依赖**可验证的答案**作筛选门；开放域（无正确答案）不直接适用。

### Quiet-STaR: LMs Can Teach Themselves to Think Before Speaking（2403.09629, 2024）
- **问题**：把 STaR 从"有标准答案的 QA"推广到**任意文本**——推理隐含在所有文本的字里行间（含对话中的心智理论）。
- **方法/机制**：在**每个 token** 处生成可学习的内部"思考（rationale）"来解释/改善后续文本预测；三项关键技术：①**tokenwise 并行采样**（控制生成成本）；②**可学习的 start/end thought token**（把"想"显式标界）；③扩展 teacher-forcing；用 **REINFORCE** 奖励那些**能改善未来预测**的思考。
- **关键结果**：在互联网文本上继续预训练后，**零样本**提升 GSM8K **5.9%→10.9%**、CommonsenseQA **36.3%→47.2%**，且对难预测 token 的困惑度改善最大；**无需在这些任务上微调**。
- **局限**：思考是**token 级 rationale**（仍在词元空间生成），计算成本高；奖励来自"是否改善下一段文本预测"。

### Pragmatic Feature Preferences（2405.14769, 2024）
- **问题**：RLHF 只用成对"哪个更好"的二元偏好，信息稀薄、学 reward 慢。
- **方法/机制**：受**语用人际沟通（RSA 谱系）**启发，把偏好查询**丰富到特征级**——不仅问"哪个更好"，还问"**哪些特征**使它更好（为什么）"；给出从特征级偏好学 reward 的方法（区分用户是否显式指明 reward 相关特征）。
- **关键结果**：在视觉/语言的 linear bandit 上**用更少比较即更快收敛到准确 reward**；蘑菇采集行为实验验证现实适用性。
- **局限**：linear bandit 等受控设定；特征需可表达。

### Open Problems and Fundamental Limitations of RLHF（2307.15217, 2023）
- **问题**：系统梳理 RLHF（行为训练）的根本局限。
- **方法/机制**：分类学式综述——人类反馈的偏差/有限性、reward model 的误设与可被钻空（reward hacking）、策略优化的脆弱性、**评估困难**等。
- **关键结果**：明确 RLHF 不是对齐的充分解，需多层防御与更好评估。
- **局限**：综述（无新方法）。

## 2. 与 VZ 的关系（三视角）

### 2.1 确证（先进性背书）
- **产品同温层（最强）**：把"关系 / 信任 / 连接"设为公司级目标函数，是名册里与 VZ 愿景最贴合的外部对照。
- **R16–R20 + R11（强）**：RSA / Pragmatic Feature Preferences 把"用户为何如此偏好"建为递归社会推理 → `user_model` / `relationship_state` owner 与社会认知（ToM）的方法论母体。
- **R-PE → credit（中）**：从语用/特征级信号反推 reward，对应 PE→credit 的 readout（更稠密的偏好信号）。
- **R1/R13（中）**：STaR 的"生成 rationale（扩充）→ 只强化正确轨迹 → 再训练"是 SSL↔RL 交替的简洁范例。

### 2.2 反证（红队）

- **反例 A（headline）｜Quiet-STaR 的 token 级思考挑战 R4**：Quiet-STaR 把"思考"实现为**词元空间的 rationale**（start/end thought token），并用 REINFORCE 优化——这看似是"在 token 空间做内部控制学习"，与 R4"控制在 z_t 而非 token 空间、禁 token 空间 RL"冲突。
  - **裁决：needs-boundary-condition**。**边界**：区分**层次**——Quiet-STaR 证明"内部、可学习、先于表达的思考过程能提升能力"，这正面支持 ETA 的"表达之下有可学习内部过程"；但其**实现媒介**是 token，VZ 主张同一能力应放在 **z_t 控制器空间**（参见 Coconut 的连续 latent 思考）。结论：**借其"可学习内部思考"目标，不借其"token 级 + token 空间 REINFORCE"实现**。在 spec 写明：Quiet-STaR = R3/R4 的"动机性证据"，Coconut/z_t = 其"正确实现层"。
- **反例 B｜STaR/Quiet-STaR 自举可能形成不可问责的自强化回路**：模型在自己生成的 rationale 上训练。
  - **裁决：survives（STaR）/ needs-boundary-condition（Quiet-STaR）**。**边界**：STaR 有**可验证正确性门**筛选，安全；Quiet-STaR 的"奖励=改善下一段预测"是**自指**信号（模型评判自己思考有没有用），在开放关系域可能自我强化偏差——须配独立只读评估，不可作为关系轨唯一信号（R-PE 不外包给自指回路）。
- **反例 C｜Pragmatic Feature Preferences 把"为什么"喂进 reward → 是否让 PE 外包给用户语用模型**：reward 由学到的语用/特征模型派生。
  - **裁决：needs-boundary-condition**。**边界**：特征级偏好是**更稠密的 PE 上游提案**，不是一级 PE 本身；user_model 由对应 owner 持有并发布快照，PE 主链仍以真实交互结果为准。
- **反例 D｜Open Problems of RLHF 本身就是反例库**：RLHF 的 reward hacking / 评估困难。
  - **裁决：survives 且强化 VZ**。**边界**：直接背书 R12（评估先做硬、只读）与 R-PE（reward 易被 hack，故 PE 须内禀、分离 epistemic/aleatoric）；VZ 不应把 RLHF 当对齐充分解。

### 2.3 局部算法借鉴（算法级解耦）

1. **Pragmatic Feature Preferences / RSA：把"用户为何偏好"建为递归社会推理，喂进 user_model** → `semantic-state-owners.md` + `social_cognition/02_theory_of_mind.md` + `prediction-error-loop.md` → 用特征级语用信号更样本高效地学 user_model / relationship_state；**前提**：作为 PE 上游稠密提案，不替代一级 PE；owner 单写。
2. **Quiet-STaR 的"可学习内部思考"目标（实现挪到 z_t）** → `temporal-abstraction.md` → 让控制器在表达之前于 **latent 空间**生成"思考"，REINFORCE/credit 作用在 z_t；**前提**：不在 token 空间做 RL（这是与 Quiet-STaR 原实现的关键分歧），参见 Coconut。
3. **STaR 的"可验证门筛选自生成数据"** → `credit-and-self-modification.md` + `evaluation.md` → 自举学习时**只保留过硬门的轨迹**再沉淀；**前提**：关系域无"正确答案"，门须是软验证器 + 独立只读评估（对照 Open Problems of RLHF 的告诫）。

## 3. 一句话定位
humans& 是名册里与 VZ **愿景最同温**的一家（关系/信任为一级目标），其 RSA/Pragmatic Preferences 直接给 user_model 与社会认知供方法论母体；但 **Quiet-STaR 的 token 级思考恰好划清了 R4 的边界**——VZ 要借其"可学习内部思考"的洞见，却必须把实现从 token 空间挪到 z_t 控制器空间；Open Problems of RLHF 则反向加固了 VZ 的评估-先做硬与 PE-内禀纪律。

## 附：本地论文清单（同目录 PDF）
- `star-bootstrapping-reasoning-with-reasoning (founders, Zelikman+Goodman)-2203.14465.pdf`
- `quiet-star-language-models-can-teach-themselves-to-think-before-speaking (founder, Zelikman)-2403.09629.pdf`
- `pragmatic-feature-preferences-reward-relevant-preferences-from-human-input (founder, Peng)-2405.14769.pdf`
- `open-problems-and-fundamental-limitations-of-rlhf (co-author, Peng)-2307.15217.pdf`
- UNVERIFIED（DOI 付费，未下载）：RSA — Pragmatic Language Interpretation as Probabilistic Inference (Goodman & Frank, 2016)。
