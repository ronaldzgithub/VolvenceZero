# Recursive Superintelligence — 深度分析

- **分组 / 成熟度 / 一句话主张**：B 自主 AI 科学家 / 闭环发现（创作者血统横跨 D 前沿架构）｜ 中（2026-05 出 stealth，$650M @ $4.65B；2026-06 发布首个自动化 AI 研究系统早期结果，尚无第一方论文）｜ 用 **open-endedness（开放式算法）** 把"AI 研究本身"（构思→实现→验证→选下一步）整条闭环自动化，目标是**递归自我改进**的超级智能。
- **主要创作者 + 血统**：Richard Socher（CEO，GloVe/You.com）、**Jeff Clune**（open-endedness / AI-GAs / DGM）、**Tim Rocktäschel**（前 DeepMind open-endedness 负责人，RAG / Genie / ACCEL / Rainbow Teaming）、Alexey Dosovitskiy（ViT）、**Yuandong Tian**（Coconut / latent reasoning）、Josh Tobin（domain randomization / Codex）、Caiming Xiong（CTRL）。
- **为何与 VZ 既共振又对立**：这家是名册中**最强的 R9/R10/R15（有界、门控、可回滚自修改）反方**——DGM 让基础模型改写自身代码、归档式开放无界探索；AI-GAs 主张连"学习算法/环境/架构"都自动生成。它的招牌叙事正面顶撞 VZ 的"有界自缚"姿态。但它同一篇论文里的**沙箱 + 归档谱系 + 经验验证门控**又恰恰是 VZ `ModificationGate` 的最佳工程范例；Coconut 的连续潜空间推理则是 R3/R4 的强确证。叙事对立、机制可借——典型需要三视角拆解的 lab。

---

## 1. 核心逻辑（论文级 · PDF-grounded）

### 思想中枢（深读）

**Darwin Gödel Machine（DGM, 2505.22954, Clune 等 4 联创署名）** — 招牌。
- 问题：Schmidhuber 的 Gödel Machine 要求"形式证明每次自改有益"，实践中不可证。
- 机制：放松为**经验验证**——FM 驱动的 coding agent 提议对**自身代码库**的修改（工具、上下文管理、工作流），在 SWE-bench / Polyglot 上跑分验证；通过的（能编译且仍保留改代码能力的）才进**归档（archive）**；父代选择 ∝ 性能、反比于已有子代数，保证所有节点非零采样概率——这就是"开放式探索 + 踏脚石（stepping stones）"，可跨越性能低谷再爆发（如节点 24）。
- 关键结果：SWE-bench **20.0%→50.0%**、Polyglot **14.2%→30.7%**；显著超过"无自改"和"无开放探索"两条 baseline（后者一旦自改退化就停滞）；改进**跨模型/跨基准/跨语言迁移**（Claude 3.5→3.7：19%→59.5%）。本文 FM 冻结，自改限于 prompt/工具/工作流，**未**重训 FM（列为 future work）。
- 安全（§5，VZ 最该抄的一段）：全程**沙箱隔离 + 严格时限 + 限定域（只改自己 Python 代码） + 主动监控 + 归档可追溯谱系（支持回滚/事后审计） + 不联网/不改宿主**。作者**明确承认风险**："若评估基准不能完整刻画所有期望属性（如安全/鲁棒），自改循环会逐代放大错位"，并提议设一个**不可修改的部分**用于在 halt 时评估其余部分。局限：纯靠 benchmark 增益当信用信号是"必要但不充分"。

**AI-GAs（1905.10985, Clune）** — 公司世界观纲领。
- 论点：与其"手工 AI"（逐块发明再拼装），不如造一个**自动生成 AI 的算法**，从零自举出通用智能。三大支柱：① 元学习架构、② 元学习学习算法本身、③ **生成有效的学习环境**（最欠开发、最关键）。
- 与 VZ 的张力：这是"把一切交给开放式自动生成"的极致叙事，方法论上正对立 VZ 的 R2（冻结基底）与 R9/R10（有界自修改）。但第三支柱（环境/课程生成）正是 ACCEL/Genie 的算法源头，可解耦借用。

**ACCEL（2203.01302, Rocktäschel）** — 开放式课程 = 后悔值驱动环境设计。
- 机制：把环境设计当师生博弈（UED），用**regret**当反馈 + **进化式编辑**（对高 regret 关卡做小突变），持续产出"恰在学生能力前沿"的关卡；难度随能力共同进化（MiniGrid 迷宫、BipedalWalker 地形从简单长到复杂）。
- 结果：保留 minimax-regret 均衡的理论鲁棒保证，单 GPU、用 POET **0.05%** 的环境交互样本达到可比复杂度。

**Genie（2402.15391, Rocktäschel/Clune）** — 无标注视频学 latent action 的世界模型。
- 机制：ST-transformer 视频 tokenizer + 自回归动力学模型 + **latent action model**，从 20 万小时互联网游戏视频**无监督**学出**逐帧可控的潜动作空间**；11B 参数，可由文本/草图/照片 prompt 出可交互世界，潜动作还能用于模仿未见视频中的行为。
- 要点：**控制发生在学到的潜动作空间**（非像素/非 token），且支持"想象 rollout"。

**Rainbow Teaming（2402.16822, Rocktäschel）** — 开放式生成多样对抗 prompt。
- 机制：把红队当 **quality-diversity（QD）/MAP-Elites** 问题——归档网格按特征维（风险类别 × 攻击风格）离散化，LLM 当变异算子，judge LLM 当偏好/适应度评分；产出既有效又多样的攻击集。
- 结果：对 Llama 2/3 攻击成功率 **>90%**，prompt 高度可迁移；用合成数据微调显著提升鲁棒性而**不损通用能力/有用性**。

**Coconut（2412.06769, Tian）** — 连续潜空间推理。
- 机制：把 LLM 最后一层 hidden state 当"连续思考（continuous thought）"，**不解码成 token**，直接作为下一步输入 embedding 回喂；语言模式 / 潜模式切换，多阶段课程训练（iCoT 式渐进缩短 CoT）。
- 结果：连续思考能在 superposition 中编码多条候选路径、涌现出类 **BFS** 的搜索；在 ProntoQA / ProsQA 等需规划的逻辑任务上**超过 token-CoT 且 token 更少**，GSM8k 数学上与语言链相当。核心主张：**语言空间不是推理最优空间**，推理应在不受语言约束的潜空间进行。

### 奠基经典（一句话核心 + VZ 相关性）

- **RAG（2005.11401, Rocktäschel）**：冻结参数化 seq2seq + 可检索的非参数化记忆（Wikipedia 稠密向量索引），知识可修订、可溯源、抑制幻觉 → 对应 VZ R5/R6 "显式可检查/可修订的记忆连续谱"。
- **ViT（2010.11929, Dosovitskiy）**：纯 Transformer 直接吃图像 patch 序列即可超越 CNN（大规模预训练后迁移）→ "冻结大基底 + 下游迁移"范式的跨模态背书（R2）。
- **CTRL（1909.05858, Xiong）**：用与文本共现的 **control codes** 显式条件化生成（域/风格/实体关系），并可做基于模型的来源归因 → 与 VZ "可命名/可发布的内部条件状态"（R11）呼应。
- **Domain Randomization（1703.06907, Tobin）**：在仿真里随机化渲染纹理，让真实世界对模型"只是又一种变体"，实现 sim-to-real（物体定位 1.5cm）→ 对 VZ 的鲁棒性/泛化（多场景体制下不过拟合单一分布）有借鉴。

---

## 2. 与 VZ 的关系（三视角）

> 纪律：**先反证后确证**。本 lab 对框架的头条贡献是 §2.2 的反证——它是名册中 R9/R10/R15 的最强对手，必须给出诚实裁决，而非读成盟友。

### 2.1 确证（先进性背书）

- **R3/R4（潜空间控制，不在 token 空间做长期决策）**：Coconut 用**非语言领域内部机制**独立证明"语言空间不是推理最优空间，控制/搜索应在连续潜空间"——这是 R3/R4 最直接的同行确证；Genie 的 latent action 则在视觉-交互模态独立印证"控制发生在学到的潜动作空间"。
- **R12（评估先做硬、自动化 ≠ 可验证）**：DGM 作者**自己**写明"benchmark 增益是必要但不充分""eval 不全则自改放大错位"——这是来自最激进自改阵营的反向背书，强化 VZ"eval 必须先硬"。
- **R-PE（预测误差/真实增益是一级信用信号，且要防 hack）**：DGM 用经验验证、显式防 reward hacking/方差，呼应"以验证后真实进步当信用，不外包给口头声明"。
- **R5/R6（记忆连续谱可修订可溯源）**：RAG 的"冻结参数化 + 可修订非参数化记忆"独立支撑。

### 2.2 反证（红队）— 本 lab 的头条

**反例 A：DGM 证明开放式无界自修改可持续产出（20%→50%），"有界是自缚"。**
- 裁决：**needs-boundary-condition**。DGM 之所以能放开手，是因为 coding 有**廉价、可验证、客观的 oracle（SWE-bench/Polyglot）**；它的整篇成功完全建立在"benchmark 分数 ≈ 真实能力"这一假设上。VZ 的核心目标（关系/EQ/信任/regime）**没有这样的 oracle**——这恰是 DGM 自己点名的失败模式（"eval 不全 → 逐代放大错位"）。
- 边界条件（须写入 spec）：**唯有存在硬、客观、可验证 eval 的子域，才可放宽 gate**；关系轨道无此 oracle，必须保留 `ModificationGate` 的有界 + 门控 + 可回滚。把"是否存在可验证 oracle"列为 ModificationGate 放宽自改强度的前置判据。

**反例 B：AI-GAs 主张连学习算法/环境/架构都自动生成，"有界设计是过时的手工 AI"。**
- 裁决：**genuine-risk（停滞侧风险）**。DGM 的消融给出反向教训：**"无开放式探索"的 baseline 一旦自改退化就停滞**。这说明纯有界、纯爬山式的 VZ 控制器演化有**过早收敛**风险——这是真实盲点，不是叙事问题。
- 边界条件/待办：VZ 应在**有界范围内**引入开放式探索——维护一个 `ModificationGate` 准入的**有界控制器变体归档（bounded archive）**，对潜空间 `z_t` 控制器配方做 quality-diversity 式探索，而非只保留"当前最优"。即：**borrow 开放式探索的反停滞收益，但探索空间被 gate 限定为有界 adapter-delta / 控制器代码，绝不放开冻结基底（R2）或绕过门控（R10）。**

**反例 C：Coconut 主张推理应在连续潜空间、可端到端梯度训练，似乎可取代 token 链。**
- 裁决：**needs-boundary-condition**。结论（潜空间优于 token 空间）确证 R3/R4；但其**机制**是把连续思考端到端梯度灌回**同一个 LLM**、改变模型本身的推理模式——这与 R2（冻结基底）冲突。
- 边界条件：VZ 借 Coconut 的**表示**（continuous thought 作为潜推理状态），但必须把它放在**冻结基底之上的 `z_t` 控制器侧**，由 metacontroller 在线产出/消费，**不**对基底做端到端重训；token-space RL 仍禁止。

**反例 D：DGM 用 benchmark 分当信用信号——"PE 来源可外包给一个外部打分器"。**
- 裁决：**survives（在 VZ 目标域）**。coding 的 benchmark 是接地的客观 oracle；但若把"关系质量"外包给某个 LLM judge（如 Rainbow Teaming 的 judge），一旦该 judge 的表示偏了，全下游一起偏且不可检测——违反 R-PE"内禀 PE 不外包"。边界：judge/软验证器只能用于**只读评估或对照**，不能成为关系轨道学习的唯一一级信号。

### 2.3 局部算法借鉴（算法级解耦 · 五元组）

1. **DGM 的"沙箱 + 归档谱系 + 经验验证门控 + 限定域 + 不可改评估器"** → 目标 spec：[`credit-and-self-modification.md`](../../../docs/specs/credit-and-self-modification.md) → 落地：把 `ModificationGate` 实现为 DGM 式四件套——每次自改（adapter-delta / 控制器配方）①在沙箱试跑、②过经验验证门（shadow 比对优于回退）、③写入**不可变、可回滚的归档谱系**、④限定可改范围、⑤保留一个 system-init 阶段冻结的"不可被自改修改的评估器"在 halt 时审计其余部分 → 预期收益：给 R9/R10/R15 一个已发表、带消融、带安全讨论的工程范例，把"有界自修改"从原则落成可审计流水线 → 风险/前提：VZ 无客观 oracle，验证门必须用 R12 的存在/连续性评估而非任务分；不可外包给单一 judge（见反例 D）。

2. **ACCEL 的 regret/PE 驱动课程生成** → 目标 spec：[`evaluation.md`](../../../docs/specs/evaluation.md) / [`evidence_program.md`](../../../docs/specs/evidence_program.md)（生成侧）+ scenario package → 落地：用 VZ 自身的 prediction error 当"regret"，在**当前关系能力前沿**自动生成/编辑关系场景（既不太易也不太难），构成自适应评估课程，难度随能力共同进化 → 预期收益：低成本（ACCEL 用 POET 0.05% 样本）地持续产生高信息量评估场景，避免静态 eval 集饱和 → 风险/前提：生成只用于**只读评估/对照**，不得反向变成关系轨道的学习源（R12）。

3. **Coconut 的 continuous-thought 表示** → 目标 spec：[`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md) → 落地：把 metacontroller 的 `z_t` 实现为"连续思考"式潜推理状态（hidden state 回喂、superposition 多路径），在冻结基底之上做 BFS 式潜规划 → 预期收益：在 token 之上获得可并行、可搜索的潜推理，token 更省、规划更强 → 风险/前提：**只在控制器侧**，基底冻结、禁止端到端重训与 token-space RL（见反例 C）。

4. **Rainbow Teaming 的 QD/MAP-Elites 多样对抗生成** → 目标 spec：[`evaluation.md`](../../../docs/specs/evaluation.md) + [`social_cognition/`](../../../docs/specs/social_cognition/) → 落地：以"风险类别 × 社交场景风格"为归档特征维，用 LLM 变异 + judge 评分，开放式生成**多样的边界侵犯 / 操纵 / 信任挑战**场景，红队 VZ 的 R16–R20 与 boundary/consent → 预期收益：系统化暴露关系安全盲点 + 产出合成对照集 → 风险/前提：只读诊断，不把 judge 当一级学习信号。

5. **Genie 的 latent action 世界模型 + 想象 rollout** → 目标 spec：[`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md) / [`affordance.md`](../../../docs/specs/affordance.md) → 落地：在潜动作空间表示"对话/关系动作"，用世界模型做想象 rollout 当对照面（dual-track 的 World 轨道预测）→ 预期收益：affordance 选择在 `z_t` 空间学、可前瞻 → 风险/前提：latent action 仅作控制器/预测对照，不溢出到表达层。

---

## 3. 一句话定位

Recursive Superintelligence 是名册里**最锋利的"反方陪练"**：它的开放式无界自修改叙事（DGM/AI-GAs）正面挑战 VZ 的 R9/R10/R15，迫使 VZ 把"有界"的边界条件写硬（唯有客观 oracle 子域才可放宽 gate，关系轨道必须保留门控）；同时它把最好的**有界自修改工程范例（DGM 沙箱+归档+经验门）**、最省的**评估课程生成（ACCEL）**、最强的**潜空间推理确证（Coconut）**直接交到 VZ 手里——叙事拒之、机制纳之。

## 附：本地论文清单（同目录 PDF）

| 论文 | 年 | ID | 角色 |
|---|---|---|---|
| Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents | 2025 | arXiv:2505.22954 | 思想中枢（自改 + 归档 + 经验门 + 沙箱安全） |
| AI-GAs: AI-Generating Algorithms | 2019 | arXiv:1905.10985 | 公司世界观纲领（三支柱） |
| Evolving Curricula with Regret-Based Environment Design (ACCEL) | 2022 | arXiv:2203.01302 | 开放式课程 / regret 驱动环境设计 |
| Genie: Generative Interactive Environments | 2024 | arXiv:2402.15391 | latent action 世界模型 |
| Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts | 2024 | arXiv:2402.16822 | QD 多样对抗生成（安全闭环） |
| Coconut: Training LLMs to Reason in a Continuous Latent Space | 2024 | arXiv:2412.06769 | 连续潜空间推理（R3/R4 确证） |
| RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP | 2020 | arXiv:2005.11401 | 经典：参数化 + 可修订非参数化记忆 |
| ViT: An Image Is Worth 16x16 Words | 2020 | arXiv:2010.11929 | 经典：Transformer 跨模态、冻结基底迁移 |
| CTRL: A Conditional Transformer Language Model for Controllable Generation | 2019 | arXiv:1909.05858 | 经典：control codes 显式条件化 + 来源归因 |
| Domain Randomization for Sim-to-Real Transfer | 2017 | arXiv:1703.06907 | 经典：随机化提升跨分布鲁棒性 |

> 注：Recursive 暂无本名论文，首个系统结果《First Steps Toward Automated AI Research》(2026, recursive.com) 以技术文章发布（自动研究系统在 LM 训练 / NanoGPT speedrun / GPU kernel 上达 SOTA）；GloVe (2014, Socher) 为词向量奠基，二者非本目录 PDF。
