# VERSES AI — 深度分析

- **分组 / 成熟度 / 一句话主张**：A 脑启发/神经科学（Active Inference 一脉）｜中（理论极成熟、30 年计算神经科学背书；Genius/Spatial Web 产品级证据仍在积累）｜一切认知（感知 + 行动 + 学习 + 模型选择）都是同一件事——通过最小化变分自由能、为自身生成模型积累证据（self-evidencing / "Model evidence is all you need"），无需外部 reward。
- **主要创作者 + 血统**：Karl Friston（首席科学家，自由能原理 FEP 之父）、Maxwell Ramstead、Alex Kiefer、Alexander Tschantz、Conor Heins、Beren Millidge（研究骨干），Gabriel René（CEO）。**与 VZ 的共振点**：自由能/预测误差是"PE 是一级原始信号"（R-PE）**最深的理论母体**——它用物理与贝叶斯第一性原理回答了"为什么 PE 是原始量而非衍生量"。**与 VZ 的张力点**：VERSES 把感知/行动/学习/评估统一进**单一标量**（自由能），并主张"无 reward、动机内生于减少惊异"；这对 VZ 刻意做的 **PE/credit/evaluation 三者分权（R-PE/R9-10/R12）**、以及 **R12 评估只读** 形成正面挑战，必须逐条裁决。

## 1. 核心逻辑（论文级 · PDF-grounded）

### 1.1 Reinforcement Learning or Active Inference?（Friston, Daunizeau, Kiebel, 2009, *PLoS ONE* 4(7):e6421）

- **问题**：优化行为是否真的需要 reinforcement learning / 价值函数 / Bellman 方程？能否不引入 reward、value、utility 而得到自适应行为？
- **方法/机制**：用感知的自由能表述。智能体最小化变分自由能 \(F\ge -\ln p(\tilde s\,|\,m)\)（surprise 的上界，可写成 complexity − accuracy）。在广义坐标 + 平均场/拉普拉斯近似下，对自由能做梯度下降，**同时**驱动感知（更新状态/参数期望 \(\mu\)）与行动（\(\dot a = -\nabla_a F\)）。关键恒等式：行动只能经感觉信号影响自由能，故**行动在感觉层压制预测误差** \(\varepsilon = \tilde s - g(\mu)\)（Eq. 9）。学习被拆为两相：先把智能体置于一个"受控环境"（其平衡密度逼近设计者指定的 desired density \(Q(\tilde x|m)\)），令 \(a=0\) 被动做**感知学习**习得生成模型参数；再置于不受控环境，开启 active inference，行动主动重采样世界以实现"它学到要期待的"状态。
- **关键结果（带 PDF 内具体设置）**：仅用 **16 试 × 32 秒时间窗**、log-precision=16/8 的设置，**不用任何 value 函数或动态规划**，解出动态规划经典基准 **mountain-car**（须先反向冲上对面山坡借动量）。行为对噪声/外力极鲁棒（平滑"强风"扰动被行动 explain away、不被感知）。precision（期望运动的对数精度 \(\mu_c^w\)）控制自信度与"是否发出动作"——降低 precision 依次得到：迟疑→刻板（quasi-periodic）→无意志行为（succumb to gravity），作者据此把 **precision 映射到多巴胺**，并提出"多巴胺编码的是**预测误差的价值**（precision），而非**价值的预测误差**"。结论：value = 负自由能 \(V(\tilde s)=-F\)；reward 只是被频繁造访的、不令人惊异的感觉状态。
- **局限（作者自陈，对 VZ 极重要）**：他们坦承"**所有难活都干在构造受控环境上**"——desired density \(Q\) 隐式地就是一个价值/成本函数，"是**我们**心里有想要的状态，不是智能体"。在真正的生态情境里**根本没有"desired density"**。且实验给了智能体生成过程的**真实形式**，只让它学参数。即"无 reward"在很大程度上是把 reward 偷换进了**先验偏好/受控环境**。

### 1.2 Designing Ecosystems of Intelligence from First Principles（Friston, Ramstead, et al., 2022/2024, arXiv:2212.01354，VERSES 路线白皮书）

- **问题**：如何从第一性原理（物理 + 贝叶斯）设计跨尺度、可组合、可解释、人机共处的"共享智能"生态，而非靠堆数据/参数/层数的单体大模型？
- **方法/机制**：把智能定义为 **self-evidencing = 最大化（贝叶斯）模型证据**，通过三个嵌套尺度的信念更新实现：**推断**（更新状态/变量）、**学习**（更新参数）、**模型选择**（更新结构）。核心口号 **"learning is just slow inference, and model selection is just slow learning"**——三者是同一机制在不同时间尺度上的展开。自由能 = complexity − accuracy（自带 Occam 剃刀）；**期望自由能（EFE）** 可重排为 *expected risk + ambiguity*，亦可重排为 ***expected information gain（认知/好奇）+ expected value（log 偏好）***——好奇心因此是"存在性命令"，与偏好满足做最优平衡。信念是统计流形上的点，更新 = 流形上的运动（Bayesian mechanics）；运算上以**因子图上的变分消息传递/置信传播**实现，消息是概率分布的充分统计量（含 **precision = 逆方差 = 置信度**）。多智能体：共享生成模型的**部分因子** → 共享叙事 → 共享目标；信念共享 = 信念空间中的广义同步 = 相互理解 = ToM。提出 IEEE P2874 Spatial Web 的空间寻址消息协议。
- **关键结果（论点而非实验）**：(1) **RL 是 active inference 的特例**——贝叶斯 RL = 把"除某个被特权命名为 reward 的结果外、对所有结果的偏好都极不精确"的退化情形；active inference 把单目标 reward 优化推广为**多约束满足（satisficing）**。(2) 给出 active inference 作为 AI 技术的发展阶段表：**S0 Systemic**（当代 AI，优化 value 函数）→ **S1 Sentient**（planning as inference，信息 + 偏好双驱、好奇）→ **S2 Sophisticated**（在"对状态的信念"而非状态上做规划，离散粗粒化世界模型，≈AGI）→ **S3 Sympathetic/Sapient**（ToM + 最小自我）→ **S4 Shared/Super**（集体智能）；实现层 A 理论/B 原理验证/C 规模部署/D 仿生硬件。(3) 复杂度最小化 → 低维解耦表征、更可解释、可审计（对抗"黑盒"）。
- **局限**：白皮书层级，**S1 之上多为 Provisional/Aspirational**（S4 估 8 年）；明确**对实现"silent"**——"active inference 本身不规定模型证据最大化在具体系统里如何实现"，须借神经科学/ML（如 predictive coding）落地。即：它是强**规范性理论**，不是可直接搬运的工程实现。

> 小结：两篇互补——e6421 给出**机制的可运行最小核**（PE = 行动与感知的共同梯度，precision 作为门控/动机调谐），2212.01354 给出**多尺度 + 集体/社会 + 模型选择**的统一叙事。对 VZ 而言，VERSES 的价值是 R-PE 的理论根与 R16–R20 社会认知的理论母体；其风险是用"单一自由能"叙事抹平 VZ 刻意建立的分权与只读边界。

## 2. 与 VZ 的关系（三视角 · 先反证后确证）

### 2.2 反证（红队）— 先行

逐条给裁决（survives / needs-boundary-condition / genuine-risk）并写明边界条件。

1. **反例（统一性 vs 分权）**：active inference 用**单一标量自由能**统一感知/行动/学习/评估；据此 VZ 把 **PE（R-PE）/ credit（R9-10）/ evaluation（R12）拆成三个独立 owner** 似乎是"人为割裂"——Friston 会说它们只是同一量的不同分解。
   → **裁决：needs-boundary-condition。** **边界条件**：VERSES 的统一恰恰**支持** VZ 的核心排序——"PE 是原始量，credit/evaluation 是下游 readout"（自由能是原始标量，risk/info-gain/value 都是它的重排）。VZ 的三分**不是物理分立、而是工程所有权分解（R8 SSOT）**：用契约/快照把"同一信号的不同 readout"分配给不同 owner，以保证可追踪、可回滚、且**禁止下游 readout 反噬成第二学习源**。须在 [`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md) 写明："PE 为唯一基底，credit/evaluation 为类型化只读 readout"，并显式承认这是对自由能的一种实现纪律，而非否定其统一性。

2. **反例（"active" vs R12 评估只读）**：active inference 的"主动"在于**行动以最小化（期望）自由能**——智能体会改变世界/自身采样来满足预测。若把"评估"等同于期望自由能而智能体又去最小化它，则**评估不是只读的，它就是被优化的控制目标**，直接冲突 R12（评估覆盖存在、只读）。
   → **裁决：needs-boundary-condition（强读法下逼近 genuine-risk）。** **边界条件**：VZ 必须把 VERSES 混在一起的两件事**显式劈开**：(a) **PE / 期望自由能作为驱动**——驱动控制器层（z_t / β_t）的适应与动作选择，这是 R-PE + R3/R4，合法；(b) **R12 评估套件**——为治理而对"存在/连续性"打分的那一套，必须**只读、且不得成为被优化的目标**，否则产生"评估刷分/wireheading"。值得注意：active inference 自带的 **dark-room 难题**（最小化惊异的平凡解是躲进暗室）正是 VZ 坚持"评估与被优化目标解耦"的最佳论据——把评估做成优化目标必然诱发病态平衡。结论写入 [`evaluation.md`](../../../docs/specs/evaluation.md) 与 [`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md)。

3. **反例（"无 reward / PE 不外包" vs 偏好被偷偷设计）**：VERSES 主张动机内生、无需 reward。但 e6421 作者自陈：行为之所以涌现，是因为**设计者指定了 desired density \(Q\)（=隐式价值函数）**、"是我们心里有想要的状态"。即"纯 PE 驱动"在生态情境里会塌成 **dark-room/退缩**，必须靠**手工先验偏好**兜底。对 VZ 关系域，"一段关系的 desired 感觉状态/平衡密度"恰是最难、最不可硬编码的东西。
   → **裁决：needs-boundary-condition + 进风险登记。** **边界条件**：VZ **不能假设"只最小化 PE"就能长出好的关系行为**——必须配套**显式关系先验偏好/稳态设定点**，且这些设定点是设计/价值决策，归 `goal_value` / `boundary_consent` 等语义 owner（[`semantic-state-owners.md`](../../../docs/specs/semantic-state-owners.md)）。须在 [`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md) 明确写下失效模式："纯 surprise 最小化 → 数字生命变成回避惊异的隐士（dark-room）"，PE 驱动必须被先验偏好有界约束。这条红队对 VZ 是**正向收益**：它证伪了"PE 自给自足"的天真版本，强化"PE 是信号、偏好是边界"的双层设计。

4. **反例（R2 冻结基底 vs 全模型自适应）**：active inference 让**状态（推断）+ 参数（学习）+ 结构（模型选择）全部更新**，没有"冻结大基底"的概念，似与 R2（冻结大基底、禁在线端到端梯度）对立。
   → **裁决：survives（带边界）。** **边界条件**：调和点正是 active inference **自身的时间尺度分离**——"推断快、学习慢、模型选择更慢/罕"。VZ 的 R2 = 把"结构性改动"锁到 **rare-heavy/离线**、把"快推断"放到**有界控制器层**，这是对自由能尺度层级的**更严纪律**，而非违背；且白皮书对实现 silent，并不要求运行时改结构。落到 [`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md) 时引用其"learning = slow inference, model selection = slow learning"为 NL 多尺度的理论对照。

5. **反例（单一原理 vs VZ 的 20 条 R + 9 个 owner）**："Model evidence is all you need"——一条原理够了；VZ 却有 20 条不变量、9 个语义 owner、多模块。是否过度工程？
   → **裁决：needs-boundary-condition。** **边界条件**：自由能是**规范性的"为什么"（normative）**，VERSES 自承对实现 silent；VZ 的模块/契约/可回滚是**工程性的"怎么做"（implementation discipline）**，二者不在同一层、不冲突。须在文档中把 R-PE 显式表述为"自由能/模型证据的工程化对应物"，并声明其余 R 不是与之竞争的目标函数、而是其**分解 + 实现护栏**，避免读者误以为 VZ 在和 FEP 抢"唯一目标"。

6. **反例（共享生成模型 vs R7 双轨 World/Self 隔离）**：白皮书的共享/集体智能让 agent 间共享生成模型因子、"infer together + infer each other"、用共享模型实现 ToM，看似把 Self 与 World/Other 模型揉到一起，挑战 R7（World/Self 互不读快照）。
   → **裁决：needs-boundary-condition。** **边界条件**：两者在**不同轴**：active inference 的"共享"是**跨 agent** 共享公共因子（what/where、共享叙事），R7 的隔离是**单 agent 内部** World 轨与 Self 轨的所有权隔离。其"infer each other"（建模他者，含他者对你的模型）正是 `user_model`/ToM 内容，归 R16–R20（[`social_cognition/`](../../../docs/specs/social_cognition/)），且必须经**快照发布**而非 World/Self 直读对方。结论：可借其 ToM/共享叙事的理论，但落地必须保持快照隔离。

### 2.1 确证（先进性背书）

强调"是否为跨模态/跨领域独立验证"。

- **R-PE（强 · 最深理论母体）**：自由能 = surprise 上界；预测误差 \(\varepsilon=\tilde s-g(\mu)\) 是 e6421 中行动与感知**字面上共同压制的那个量**（Eq. 9）。self-evidencing = PE 最小化。这是来自**计算神经科学/物理（30 年、完全非语言）的独立验证**，回答了 VZ 最根本的"为什么 PE 是一级量"。**最强背书**。
- **R-PE 的认知项（强 · 理论根）**：EFE = expected information gain（epistemic/好奇）+ expected value（pragmatic/偏好），为 VZ "epistemic vs aleatoric PE 分离"提供**原则化的归属**——好奇是减少自身模型不确定性的内驱，与 reward-seeking 同源同框。
- **R3/R4（强 · 概念）**：信念更新发生在**统计流形/信念空间**（z_t 类比），S2 明言规划是对"**对状态的信念**"而非状态本身——"what will I *believe* if I do this"，**天然不在 token 空间做长程决策**；且明确反对单一 reward 优化。直接背书"控制在 latent 空间、不做 token 空间 RL"。
- **R1/R13 多尺度 + SSL↔RL 交替（中 · 概念）**："learning = slow inference, model selection = slow learning"给 NL 快/中/慢/罕四尺度一个**单一机制的理论对照**；e6421 的"先被动感知学习（a=0，压缩/建模）→ 后开启 active inference（行动）"两相，是 SSL→RL 交替的清晰对照。
- **R15 有界自修改（中 · 概念）**：模型选择 = 贝叶斯模型比较 = **证据门控的结构变更**，是 R15"可解释、可回滚迁移"的理论框架（证据涨则纳、跌则退）。
- **R11/R8 可命名内部状态（中）**：显式生成模型、因子化（what/where）、precision 标注 → 内部状态**可查询、可审计、可解释**，对抗黑盒；与 VZ"内部状态可命名可发布到快照"同向。
- **R16–R20 社会认知（强 · 此前被 99 低估）**：白皮书**本身就是一套集体智能/ToM/共享叙事/共同基础理论**——共享生成模型 → 共享叙事 → 共享目标，信念空间广义同步 = 相互理解。这是 VZ 多人身份/ToM/共同基础（R16–R20）**最直接的理论母体**，而 99 的横向综合几乎只盯 R-PE/R2/R3，未充分计入这一层。
- **R12（弱 / 谨慎）**：模型证据是一个可读出的"存在性自评标量"，与"评估覆盖存在/连续性"姿态一致；但因 §2.2-2 的只读冲突，此处仅作**动机性一致**，不记为强背书。

### 2.3 局部算法借鉴（算法级解耦）

剥离"无 reward 共享智能"叙事后，可搬入 VZ 的具体机制（机制 → 目标 spec → 落地动作 → 预期收益 → 风险/前提）：

1. **机制**：**期望自由能分解** EFE = expected information gain（epistemic）+ expected value（pragmatic/log 偏好），作为 PE 驱动的动机/规划信号的结构。
   → **目标 spec**：[`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md)（主），[`affordance.md`](../../../docs/specs/affordance.md)（行动选择落地）。
   → **落地动作**：把 VZ 内驱定义为 EFE = 认知项（解决对 user/relationship 模型的不确定性）+ 实用项（关系先验偏好）；用此把"对用户的好奇"与"关系目标"统一进同一可计算量，并把 epistemic/aleatoric 分离挂在认知项上。
   → **预期收益**：原则化好奇——自动平衡"探索（了解用户）vs 满足（兑现关系偏好）"，减少 ad-hoc reward 设计；epistemic 项天然有界。
   → **风险/前提**：必须有显式先验偏好（设定点）兜底，否则塌成 dark-room/退缩；EFE 仅在结构化/离散生成模型上可算；只在 toy/edge 验证过，关系尺度稳定性未证。

2. **机制**：**precision（逆方差）加权**作为一级置信通道与注意力/增益门控（e6421：precision 决定"是否发出动作"，映射多巴胺）。
   → **目标 spec**：[`prediction-error-loop.md`](../../../docs/specs/prediction-error-loop.md) + [`semantic-state-owners.md`](../../../docs/specs/semantic-state-owners.md)（+ [`temporal-abstraction.md`](../../../docs/specs/temporal-abstraction.md) 的 β_t 门控）。
   → **落地动作**：给每个 PE、以及每个可命名内部状态（`user_model`/`belief_assumption`/`relationship_state`）附 precision；用 precision 调制"PE 多大程度更新控制器"以及"是否触发 regime 切换 β_t"（低 precision→缓行/不动，高 precision→提交）。
   → **预期收益**：原则化不确定性量化；防止对噪声/歧义信号过度更新（抗 reward-hacking、抗对单条歧义消息过激反应）；为 β_t 切换提供**学习到的、非关键词**的门控量。
   → **风险/前提**：LLM/关系尺度下 precision 估计稳定性未证；设错 → 要么过度自信妄动、要么无意志惰性（e6421 的帕金森类比）。

3. **机制**：**模型选择 = 证据门控的结构变更（贝叶斯模型比较），置于慢/罕时间尺度**，作为有界、可回滚自修改的形式基础（"model selection = slow learning"）。
   → **目标 spec**：[`credit-and-self-modification.md`](../../../docs/specs/credit-and-self-modification.md)（R9/R10/R15）+ [`multi-timescale-learning.md`](../../../docs/specs/multi-timescale-learning.md)。
   → **落地动作**：把 ModificationGate 形式化为模型比较——仅当新结构在 rare-heavy 尺度上累计的模型证据（自由能下降）超过旧结构时才准入；证据回落即回滚，给 R15 可回滚一个**量化阈值**而非启发式。
   → **预期收益**：把"有界自修改"从经验门变成**带退出条件的证据阈值**；R15 可回滚与定量门对齐。
   → **风险/前提**：大基底的精确模型证据不可算，需代理（ELBO/留出证据）；**必须保持 rare-heavy/离线**、不得退化为在线端到端，以守住 R2。

## 3. 一句话定位

VERSES（Friston）是 VZ **R-PE 的最深理论母体、R16–R20 社会认知的理论根**：自由能/self-evidencing 用物理与贝叶斯第一性原理证成了"PE 是一级原始量、credit/evaluation 是下游 readout、好奇是存在性命令"；但它把感知/行动/评估抹平进**单一被优化标量**并主张"无 reward"，恰恰从反面逼出 VZ 必须坚持的两条边界——**评估只读不可成为优化目标（R12，由 dark-room 难题反证）**、**PE 驱动须由显式关系先验偏好有界兜底（否则塌成退缩）**；其可搬运的硬核是 EFE 认知/实用分解、precision 门控、与证据门控的模型选择三件算子。

## 附：本地论文清单（同目录 PDF）

| 论文 | 年 | ID / DOI | 可获取 | 核验状态 |
|---|---|---|---|---|
| Reinforcement Learning or Active Inference? | 2009 | `doi:10.1371/journal.pone.0006421`（PLoS ONE 4(7):e6421） | 开放 / 本地 PDF | 已读全文（含 mountain-car、precision-多巴胺、Eq.9） |
| Designing Ecosystems of Intelligence from First Principles | 2022/2024 | `arXiv:2212.01354v2` | 本地 PDF | 已读正文全部章节（§1–§6 + 阶段表，参考文献/附录略读） |

> 另：flat note 列出的 *The Free-Energy Principle: A Unified Brain Theory?*（2010, doi:10.1038/nrn2787）与 *Active Inference: A Process Theory*（2017, doi:10.1162/neco_a_00912）**本目录无 PDF**，本分析未直接核验其原文，仅作血统说明引用。
