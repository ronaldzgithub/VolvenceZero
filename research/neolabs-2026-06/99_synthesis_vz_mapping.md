# Neo Labs → VolvenceZero 交叉综合与借鉴判断（bio-heavy, 2026-06）

调研日期：2026-06-13 ｜ 口径见 [`00_roster.md`](00_roster.md) ｜ 下载见 [`_download_summary.md`](_download_summary.md)

> **互补阅读**：本文是**确证导向**的横向综合。其**反证（红队）与局部算法借鉴（算法级解耦）两条轴**，以及每家 PDF-grounded 的深度分析，见 [`98_deep_analysis_framework.md`](98_deep_analysis_framework.md) 与 `labs/<lab>/analysis.md`。98 经 PDF 级红队后**纠正了本文 3 处确证偏误**（R2★ 误标 Isomorphic、把训练损失读作 R-PE、ProGen3/State 措辞），详见 98 §4.1——本文主结论方向成立但强度被高估，请结合 98 阅读。

> 本文是 30 个非头部新型实验室（生物/神经科学为主）的横向综合，对齐 [`../../docs/next_gen_emogpt.md`](../../docs/next_gen_emogpt.md) 的 R 不变量与 7 个认知 primitive。与既有的
> [`../arxiv-survey-2026-05.md`](../arxiv-survey-2026-05.md)、
> [`../core-author-paper-assessment-2026-05.md`](../core-author-paper-assessment-2026-05.md)、
> [`../deepmind-author-paper-assessment-2026-05.md`](../deepmind-author-paper-assessment-2026-05.md) 互补：那三份覆盖 NL/ETA 主线与头部实验室，本份覆盖**新型实验室生态**。

---

## 一、总判断

把这 30 家放在一起看，最重要的发现不是某家的单点突破，而是一个**反直觉的结构性事实**：

> **对 VZ 最有参考价值的信号，几乎不来自估值最高的前沿 AI 大厂（SSI / Thinking Machines / World Labs），而来自三类"边缘"实验室：(1) active inference 一脉、(2) 内禀动机/好奇心一脉、(3) 虚拟细胞世界模型一脉。这三类各自独立地收敛到与 VZ 完全相同的三条公理上——预测误差是一级信号、冻结大基底 + 自适应控制器、控制发生在 latent 空间而非 token 空间。**

这个收敛尤其有意义，因为它跨越了**语言、机器人、神经科学、细胞生物学**四个互不通话的社区：

- **生物基础模型**（ESM3 / Evo / Chroma / RFdiffusion / Phenom / TranscriptFormer）在**完全非语言的模态**上、在巨大规模上**独立验证了 R2**——"冻结大基底 + 下游适配/头"是可泛化的物理事实，不是 NLP 的偶然。这是本次 bio-heavy 调研最大的收获：**R2 不是语言模型的工程技巧，是跨模态的结构规律**。
- **active inference**（Friston：VERSES / Stanhope / Cortical Labs）给 R-PE 提供了**最深的理论母体**：自由能最小化 = 预测误差驱动的感知-行动-学习统一闭环；Cortical Labs 更在**真实活体神经元**上验证了它。
- **内禀动机**（Pathak：Skild AI 的 ICM）给 R-PE 提供了**最直接的工程算子**：内禀奖励 = 前向模型预测误差，且区分可减小（epistemic）与不可减小（aleatoric）。
- **CZI 的 rBio** 给出了**最贴近 VZ 架构的镜像**：在冻结世界模型上，用"软验证器"奖励对推理控制器做 RL——这正是 VZ"latent 控制器 + 冻结基底 + 无硬奖励下的学习信号"要解决的问题，只不过它发生在细胞生物学而非关系领域。

下面按 7 primitive 与 R 轴展开，最后给 ROI 借鉴清单与踩雷清单。

---

## 二、按 VZ 7 个认知 primitive 映射

| Primitive（缺它的 failure mode） | 最强证据来源（neo labs） | 对 VZ 的含义 |
|---|---|---|
| 1. **Frozen substrate**（缺→灾难性遗忘） | EvolutionaryScale, Arc, Isomorphic, Chai, Profluent, Recursion, Generate, Xaira, Basecamp | R2 在蛋白/基因组/细胞/影像上被独立验证：冻结大基底 + 下游头，是跨模态规律 |
| 2. **Latent controller**（缺→token 空间不可扩展） | Reflection AI(MuZero), Sakana(World Models/CTM), Physical Intelligence(π0 flow), Liquid AI(CfC), Cartesia(SSM) | R3/R4 的多条工程实现；控制/规划在 latent 动力学空间而非 token |
| 3. **Emergent switching**（缺→只能硬编码 skill） | Sakana(Transformer² 专家向量), Reflection(Algorithm Distillation) | 测试时混合/切换专家 = β_t 切换的工程对照（仍偏显式，不如 ETA 涌现彻底） |
| 4. **Multi-timescale memory**（缺→记忆与推理脱钩） | Cartesia(HiPPO/S4 有界状态), Liquid AI, Numenta(参考帧), World Labs(空间记忆) | R5/R6：SSM 隐藏状态 = 原则化的"全历史有界压缩"，记忆连续谱的具体机制候选 |
| 5. **Epistemic PE**（缺→reward hacking） | **Skild AI(ICM + epistemic/aleatoric)**, VERSES/Stanhope/Cortical Labs(自由能), CZI(rBio 软验证), Future House(实验接地奖励) | R-PE 的理论母体 + 工程算子 + 生物验证三重支撑 |
| 6. **Bounded self-modification**（缺→sleeper agent） | Thinking Machines(PPO/TRPO trust-region, LoRA-without-regret), Sakana(Evolutionary Merge), Physical Intelligence(MAML), Symbolica(可证不变量) | R9/R10/R15：有界 trust-region 更新 + 有界 adapter-delta + 形式约束 |
| 7. **Read-only geometric monitoring**（缺→alignment faking） | Stanhope/VERSES(可询问信念状态), Symbolica(类型/范畴接口) | R12/R11：可询问、可命名的内部状态；本生态在"几何监控"上较弱（头部 mech-interp 更强） |

**结论**：30 家合起来恰好覆盖 7 个 primitive 的前 6 个；唯一相对薄弱的是 primitive 7（只读几何监控）——这块仍需依赖头部实验室的 persona/representation 工具（见 deepmind 评估）。这进一步印证 VZ 的 14 条 R 不变量站在正确的收敛点上。

---

## 三、按 R 轴交叉评估

### 3.1 R2 — 冻结基底 vs 自适应控制器（本次最强信号）

整个 Group C（14 家生物基础模型）就是一台"R2 验证机"：

- **ESMFold = 冻结 ESM-2 + 折叠头**；**State = 稳定 ST 基底 + SE**；**BaseFold = 冻结 AlphaFold2，增益全来自更丰富输入**；**Chai = 冻结 PLM embedding + pair-bias 头**。
- **借鉴**：这些都是"基底冻结、只动下游/条件化"的干净样本。VZ 的 R2 姿态（[`../../docs/specs/...`] 冻结基底 + 控制器层）在这里得到**跨模态的合法性背书**——可以在 spec 里直接引用"蛋白/基因组基础模型同样遵循冻结基底范式"作为设计论据。
- **关键差异**：ProGen3 用 DPO 把基底**对齐**到湿实验数据（控制器层适配，非端到端重训基底）——这是 R2 的正确做法的范例，可写入 [`credit-and-self-modification`] 的"如何在不破坏基底的前提下吸收新数据"。

### 3.2 R3/R4 — 时间抽象与 latent 控制（不在 token 空间做长期决策）

- **Reflection AI 的 MuZero**：学到的 latent 动力学 + MCTS 规划，是"在 latent 空间想象/规划"的经典；**Algorithm Distillation** 把"学习进展"做成 in-context 算子（无权重更新），直接对应 VZ"在冻结基底上 in-context 适配控制"。
- **Sakana World Models**：V-M-C 三件套，在"梦"里训练策略——latent 想象 rollout 的奠基；**CTM** 把神经元时序同步当 latent 计算基底。
- **Physical Intelligence π0**：冻结 VLM + flow-matching 动作头，动作 chunking 在 latent 动作空间。
- **借鉴方向**：这些都**反对在 token 空间做长期 RL**，与 VZ R3/R4 一致。最可直接借鉴的是 **Algorithm Distillation 的"in-context 学习算子"**——VZ 的 metacontroller 可以把"如何在一次会话内改进"蒸馏成 z_t 空间的 in-context 算子，而非更新权重。

### 3.3 R-PE — 预测误差作为一级原始信号（本次理论纵深最深）

三层证据叠加：

1. **理论母体（active inference）**：Friston 的自由能原理把感知/行动/学习统一为预测误差最小化。VERSES `Designing Ecosystems...`(2212.01354) + `Active Inference: A Process Theory` 是 VZ R-PE 的**最深学术后盾**——"为什么 PE 是一级量而非衍生量"在这里有 30 年计算神经科学的答案。
2. **工程算子（ICM）**：Skild AI 的 Pathak `Curiosity-driven Exploration`(1705.05363) = 内禀奖励是前向模型预测误差；`Large-Scale Study`(1808.04355) 在 54 环境验证、常无外部奖励，并暴露 **noisy-TV 问题**（→ 必须区分 epistemic / aleatoric PE）。
3. **生物验证（DishBrain）**：Cortical Labs 在活体神经元上证明自由能最小化真实发生。
4. **无硬奖励的 readout（rBio）**：CZI 用世界模型/GO 当"软验证器"提供奖励信号——**这是 VZ 最该学的一招**（见 §四.2）。

**借鉴方向**：在 [`prediction-error-loop`] 明确写入"epistemic vs aleatoric 分离"（来源 ICM 大规模研究 + active inference），并把"软验证器奖励"列为关系质量这类无硬奖励场景的候选学习信号机制。

### 3.4 R5/R6 — 记忆连续谱

- **Cartesia 的 HiPPO/S4**：用多项式投影对**全部历史做有界压缩**，以固定大小状态滚动更新——这是"记忆连续谱"最有原则的机制候选，可作为 VZ 瞬态/情景层压缩的非 Transformer 候选。
- **Numenta 参考帧 / World Labs 空间记忆**：分布式持久结构化记忆的另两种范式。
- **借鉴**：在 [`continuum-memory`] 增加"SSM 有界状态作为压缩 stratum"的设计候选小节。

### 3.5 R9/R10/R15 — 有界自修改 + 可回滚

- **Thinking Machines**：PPO/TRPO 的 **trust-region** = 有界、单调改进的策略更新（≈ 门控的自修改）；**LoRA Without Regret** = 何时低秩 adapter 足以匹敌全量微调（= 有界 adapter-delta 进入冻结基底的判据）；**Defeating Nondeterminism** = 可复现推理（支撑 R15 可回滚运行时）。
- **Sakana Evolutionary Merge / Transformer²**：有界、可换的适配配方。
- **Symbolica**：可证不变量 + 形式约束 = 有界、可审计、可回滚的自修改方向（思辨性，尚无规模化证据）。
- **借鉴**：LoRA-without-regret 的判据可直接进 [`credit-and-self-modification`] 的"何时走 adapter-delta vs rare-heavy"决策；trust-region 思想支撑 ModificationGate 的"单调改进 + 可回退"。

### 3.6 R1/R13 — 多时间尺度 + SSL↔RL 交替

- **自主科学家闭环**（Future House Robin/Aviary、Lila、Periodic）= "压缩（SSL/世界模型）↔ 强化（实验验证）"交替的工程范例。
- **重要教训（踩雷预警）**：Periodic Labs 引用的 **A-Lab(2023) 自主合成结果后被学界质疑**——"自动化闭环 ≠ 结果可验证"。这强力支持 VZ 的 **R12（评估必须先做硬）**：任何自修改/学习闭环若没有可验证、可对照的 eval，自动化只会更快地放大错误。

---

## 四、对 VZ 的 ROI 借鉴清单（按预期收益排序）

### 4.1 高优先（建议本季度反映到 spec / shadow prototype）

1. **"软验证器奖励"用于无硬奖励的控制器学习**（来源：CZI **rBio**）
   - **现状盲点**：VZ 的核心目标（关系/EQ/regime）没有可验证奖励。rBio 证明：可以用一个**冻结世界模型**当"软验证器"给推理控制器的 RL 提供奖励。
   - **可落地动作**：在 [`prediction-error-loop`] / [`credit-and-self-modification`] 加入"软验证器"小节——用 VZ 自己的 world/self 双轨预测模型作为软验证器，为控制器层 RL 提供 PE-based 奖励，而非外部 scalar reward。
   - **收益**：给"关系质量无法直接打分"这个根本难题一个原则化、已被生物领域验证的解法。**这是本次 bio-heavy 调研对 VZ 最独特的贡献。**

2. **epistemic / aleatoric PE 分离 + 内禀动机**（来源：Skild **ICM** + active inference）
   - **可落地动作**：把 [`prediction-error-loop`] 的 PE 显式拆成可减小/不可减小两路，只用 epistemic 驱动动机（避免 noisy-TV：用户的随机性不应让系统永远"好奇")。
   - **收益**：从根上防 reward hacking 与"对噪声上瘾"。

3. **in-context 学习算子（z_t 空间）替代权重更新**（来源：Reflection **Algorithm Distillation**）
   - **可落地动作**：metacontroller 把"会话内如何改进"蒸馏为 z_t 空间的 in-context 算子（online-fast 层），权重不动；权重级变更留给 rare-heavy + ModificationGate。
   - **收益**：天然契合 R2/R3 + 多时间尺度分层。

4. **有界 adapter-delta 判据**（来源：Thinking Machines **LoRA Without Regret** + Sakana **Transformer²**）
   - **可落地动作**：把"低秩 adapter 何时足够"的判据写入 ModificationGate，作为 online 适配（adapter）vs rare-heavy（重训）的分流条件。

5. **SSM 有界状态作为记忆 stratum**（来源：Cartesia **HiPPO/S4**）
   - **可落地动作**：在 [`continuum-memory`] 评估用 SSM 有界状态压缩瞬态/情景记忆的可行性。

### 4.2 中优先（记入技术路线，暂不动主链）

- **active inference 的"模型选择 = 证据门控的结构变更"** → R15 可回滚的理论框架（VERSES/Stanhope）。
- **生物基础模型的"冻结基底 + 条件化输入"** → 作为 R2 跨模态合法性论据写入 spec（BaseFold：纯靠输入数据提升，不动基底）。
- **MAML 式有界快速适配**（Physical Intelligence）→ 控制器层 few-shot 适配接口。

---

## 五、被高估 / 容易踩雷（作为实现者必须警觉）

1. **"科学超级智能"叙事 vs 证据**（Lila Sciences / Periodic Labs）：两家几乎无第一方可验证论文，主张多在 blog/新闻。**A-Lab 自主合成结果被质疑**是直接教训——**不要引入它们的主张，只引入闭环机制，且必须等 eval 做硬之后**。
2. **开放式无界自修改**（Sakana **AI Scientist** 的 open-ended 路线；及并行进程目录里出现的 Darwin Gödel Machine 一类）：违反 VZ 的有界 + 可回滚（R9/R10/R15）。demo 诱人，scale 后验证成本爆炸。**保守的有界路线更可持续。**
3. **wetware 不可工程化**（Cortical Labs）：价值在"PE 一级信号"的生物验证，**不是**可落地组件。
4. **把 PE 来源外包给另一个模型**（部分自主科学家用 LLM 当 task/reward 生成器）：一旦该模型的任务表示偏了，所有下游一起偏且无法检测——违反 R-PE"内禀 PE 不外包"。
5. **前沿大厂的"产品级交互模型"**（Thinking Machines interaction models / World Labs Marble / Reflection Asimov）：这些是**产品**而非可借鉴的方法；对 VZ 有方法价值的是其**创始人奠基论文**（PPO、NeRF、MuZero），不是产品本身。
6. **SSI**：完全 stealth，无任何可评估信息，**不应作为参考对象**，仅作背景登记。

---

## 六、5 个最值得吃的交叉前沿（2026→2028 窗口）

1. **无硬奖励下的"软验证器"RL**：rBio 在生物上验证，VZ 在关系上需要——**最直接的跨域迁移红利**。
2. **epistemic/aleatoric PE 在 LLM/关系尺度的稳定估计**：active inference 与 ICM 只在 toy/edge 验证过。
3. **latent action basis 跨模态迁移**：π0 的视觉-动作 z_t 能否迁到对话 z_t？
4. **SSM 有界状态 vs 显式 CMS** 谁更适合记忆连续谱：Cartesia 路线 vs VZ 现有 4-stratum。
5. **可回滚（R15）形式化**：active inference 的证据门控模型选择 + Thinking Machines 确定性推理，是 R15 目前最现实的两块拼图。

---

## 七、一句话总结

> **这 30 家"新型实验室"最大的价值，是用四个互不通话的领域（语言、机器人、神经科学、细胞生物学）独立地、同时地证明了 VZ 三条公理的正确性：预测误差是一级信号、冻结大基底 + 自适应控制器、控制在 latent 空间。** 其中 active inference 给了 R-PE 最深的理论根，Skild/ICM 给了它最直接的算子，生物基础模型给了 R2 最广的跨模态背书，而 **CZI rBio 的"软验证器 RL"是本次唯一一个能直接搬进 VZ 来解决"关系质量无硬奖励"这一根本难题的机制**。需要警惕的是自主科学家赛道的"自动化 ≠ 可验证"陷阱（A-Lab 教训），它恰恰反向印证了 VZ "评估先做硬、自修改要有界可回滚"的克制是对的。

---

## 附：30-lab × R 轴速查

| Lab | 分组 | R2 | R3/R4 | R-PE | R5/R6 | R9/R10/R15 | 成熟度 |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Numenta | A | · | ✓ | ✓ | ✓ | · | 中 |
| Liquid AI | A | ✓ | ✓ | · | ✓ | ✓ | 高 |
| VERSES | A | ✓ | ✓ | ★ | ✓ | ✓ | 中 |
| Stanhope | A | ✓ | ✓ | ★ | · | ✓ | 中 |
| Cortical Labs | A | ✓ | ✓ | ★ | ✓ | · | 中 |
| Cartesia | A | ✓ | ✓ | · | ★ | · | 高 |
| Symbolica | A | · | ✓ | · | · | ✓ | 低 |
| Future House | B | ✓ | ✓ | ✓ | · | ✓ | 高 |
| Lila Sciences | B | · | ✓ | ✓ | · | ✓ | 低 |
| Periodic Labs | B | ✓ | · | ✓ | · | ✓ | 低 |
| EvolutionaryScale | C | ★ | ✓ | · | · | · | 高 |
| Arc Institute | C | ★ | ✓ | ✓ | ✓ | · | 高 |
| Isomorphic | C | ★ | ✓ | · | · | · | 高 |
| Chai Discovery | C | ✓ | ✓ | ✓ | · | · | 中高 |
| Profluent | C | ★ | · | · | · | ✓ | 高 |
| Latent Labs | C | ✓ | ✓ | ✓ | · | · | 中 |
| Recursion | C | ★ | ✓ | ✓ | · | · | 高 |
| Insitro | C | ✓ | ✓ | ✓ | · | ✓ | 中高 |
| Generate Bio | C | ✓ | ✓ | · | · | · | 中高 |
| Xaira | C | ✓ | ✓ | · | · | · | 高 |
| Noetik | C | ✓ | ★ | ★ | · | · | 低 |
| CZI Virtual Cell | C | ★ | ✓ | ★ | ✓ | ✓ | 中高 |
| Basecamp | C | ★ | · | · | ✓ | · | 中 |
| Sakana AI | D | ✓ | ★ | · | · | ✓ | 高 |
| World Labs | D | · | ★ | · | ✓ | · | 中高 |
| Physical Intel | D | ✓ | ★ | · | · | ✓ | 高 |
| Skild AI | D | ✓ | ✓ | ★ | · | · | 中 |
| Thinking Machines | D | ✓ | · | · | · | ★ | 中 |
| Reflection AI | D | ✓ | ★ | ✓ | · | · | 中 |

（★ = 该轴的强样本/标杆；✓ = 相关；· = 弱/无）
