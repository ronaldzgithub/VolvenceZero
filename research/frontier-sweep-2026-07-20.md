# Frontier Labs 主路径横扫（第三批，严格去重）

调研日期：2026-07-20

基线（本次不重复）：

- `research/README.md`：7 primitive 总结
- `research/core-author-paper-assessment-2026-05.md`
- `research/deepmind-author-paper-assessment-2026-05.md`
- `research/nl-eta-author-update-2026-07.md`
- `research/nl-eta-mainpath-expansion-2026-07-20.md`
- `research/papers/` 既有 208 篇 PDF

## 0. 结论先行

本批从 Meta FAIR / MSL / Reality Labs、Anthropic、OpenAI、Google DeepMind / Research /
Cloud AI、Physical Intelligence、Liquid AI、Sakana AI 及直接邻接机制线中，筛出并下载 28 篇既有库
未收录论文。最重要的增量不是又出现了
一个新的 cognitive-agent primitive，而是三条原有主路径在 2026 年上半年开始进入可检验的工程形态：

1. **R12 / R15 从“做 benchmark”推进到“模拟部署分布、复现失败、主动选样、上线后核验”**。
   OpenAI Deployment Simulation、DeepMind Gram / Honeypot / ProEval 共同给出一条比静态红队更完整的
   发布门证据链。
2. **R7 / R14 从 persona 几何推进到功能性情绪几何**。Anthropic 证明情绪概念不是表层措辞：
   它们是可读出的内部方向，并对 reward hacking、blackmail、sycophancy 有因果影响；但方向是
   token-local 的“当前有效概念”，不能误读成持久主体状态，更不能据此声称模型有主观体验。
3. **R5 / R6 / R9 的记忆问题正在从“存什么”转向“谁学习管理记忆、如何归因、如何回退”**。
   PAHF、SkillOS、EvolveMem 分别覆盖动态用户偏好、程序性 skill curator、检索基础设施自演化；
   它们支持 owner-local 适应，但也暴露 token/文本空间 RL、任务指标反灌、配置写面扩张的风险。

最需要保持边界的两篇是 **HyperAgents** 和 **Sakana Fugu**。前者把元修改器也开放为可写程序，
是 R10“有界自修改”的直接反例压力测试；后者用自然语言协调器动态生成 agent scaffold，证明编排可学，
但不是 ETA 的 latent `β_t / z_t`，不应被当作 VZ 时间抽象实现。

核心作者增量检索确认了 4 篇新作：Ali Behrouz 参与的 Online Neural Space-Time Memory、
von Oswald / Sacramento 的 ICL Intrinsic Curiosity、Mirrokni 的 Geometric Signatures，以及
von Oswald 参与的跨模态 Curse-of-Depth 研究（后者相关度较低，登记但未下载）。其中 ICL curiosity
给出一般 MDP 上的严格负面定理，实质修正了首轮“无新增”的判断。

## 1. 方法与去重

候选必须同时满足：

1. arXiv ID 不在 `research/papers/**/*.pdf` 的既有 ID 清单中；
2. 直接命中 P1–P7 之一，或 R9 / R11 / R12 / R15 接缝；
3. 有可对照机制，而非泛泛的 agent 产品或观点文章；
4. PDF 下载后通过文件类型和最小体积校验，并以 PDF 正文核对标题、作者、机构和主张。

下载脚本：`research/download_frontier_sweep_2607.sh`

下载目录：`research/papers/sweep-2607/`

校验结果：28/28 成功，0 个截断或 HTML 冒充 PDF；详情见
`research/frontier-sweep-2026-07-20-download-summary.md`。

## 2. 纳入论文

| 组 | 论文 | 机构 | arXiv | 映射 | 定位 |
|---|---|---|---|---|---|
| A | HyperAgents | Meta FAIR / MSL, UBC, Vector, Edinburgh, NYU | `2603.19461` | P6 / R10,R15 | 无界元自修改压力测试 |
| B | S-EMBER | Meta FAIR / Reality Labs | `2607.02689` | P4 / R5,R6,R12 | 流式情景记忆 benchmark |
| B | Learning Personalized Agents from Human Feedback | Meta MSL, Princeton, Duke | `2602.16173` | P4 / R5,R11 | 动态用户偏好 + 双反馈 |
| B | RA-RFT | Meta MSL, Rice | `2606.13680` | P4 / R9 | 推理效用检索 + token RLVR |
| C | Emotion Concepts and their Function in an LLM | Anthropic | `2604.07729` | P7 / R7,R12,R14 | 功能性情绪几何 |
| D | Predicting LLM Safety by Simulating Deployment | OpenAI | `2607.07184` | P7 / R12,R15 | 部署分布发布门 |
| D | Gram | Google DeepMind | `2605.30322` | P7 / R12,R15 | sabotage 自动审计 + 静态复现 |
| D | Realistic Honeypot Evaluations | Google DeepMind | `2605.29729` | P7 / R12,R15 | 真实代码库 scheming 诱捕 |
| D | ProEval | Google DeepMind | `2604.23099` | P5,P7 / R12 | Bayesian 主动评估 |
| E | Digital Red Queen | Sakana AI, MIT | `2601.03335` | P6 / R10 | 动态对手自博弈演化 |
| E | Sakana Fugu | Sakana AI | `2606.21228` | P3 / R3,R4 | 自然语言多 agent 编排 |
| F | EvolveMem | UNC, UC Berkeley, UCSC | `2605.13941` | P4,P6 / R5,R10,R15 | guarded 检索自演化 |
| F | SkillOS | Google Cloud AI Research, UIUC, MIT | `2605.06614` | P4 / R2,R5,R9 | frozen executor + skill curator |
| G | NE-Dreamer | T-Tech | `2603.02765` | P1,P2,P5 / R2,R3,R-PE | next-embedding world model |
| G | Formal Verification of Learned MARL Policies | UALR（IROS 2026） | `2606.19632` | P3,P6 / R10,R15 | 蒸馏后模型检查 |
| H | Online Neural Space-Time Memory | Google, UW（Behrouz） | `2607.15271` | P4 / R1,R5,R6 | 低频写、高频读的 TTT memory |
| H | Can ICL Support Intrinsic Curiosity? | Google Paradigms / DeepMind | `2606.19476` | P5 / R-PE,R9 | 一般 MDP 上的 curiosity 负面定理 |
| H | Geometric Signatures of Reasoning | Google Research, NEU, USC | `2607.01571` | P7 / R3,R4,R12 | 推理轨迹谱与早期正确性 |
| H | Beyond Perplexity | CMU, MBZUAI | `2607.00368` | P4 / R5,R6,R12,R15 | deployment-memory 证据阶梯 |
| I | MARS | Google Cloud AI Research | `2602.02660` | P4 / R9,R15 | 分支差分反思与预算搜索 |
| I | Cross-Architecture Model Diffing | Anthropic Fellows, Mila | `2602.11729` | P7 / R10,R12,R15 | artifact 几何 diff |
| I | SaliMory | Meta Reality Labs | `2606.04120` | P4 / R5,R9,R11 | 认知分层记忆 + 阶段信用 |
| J | Mamba-3 | CMU, Princeton, Cartesia | `2603.15569` | P1,P4 / R2,R5 | 固定容量状态跟踪基底 |
| J | Real-Time Recurrent RL | MIT, Liquid AI, TU Wien | `2602.02236` | P2,P5,P6 / R1,R9,R10 | 每步在线 recurrent credit |
| J | Sheaf-ADMM Coordination | Sakana AI, Warsaw | `2605.31005` | P3 / R8,R11 | 局部 proposal / consensus / disagreement |
| J | Multi-Scale Embodied Memory | Physical Intelligence | `2603.03596` | P2,P4 / R1,R3,R5 | 稠密短记忆 + 语言长记忆 |
| K | Active Inference as Test-Time Scaling | Friston / UCL / Monash 等 | `2606.22813` | P5 / R-PE,R3,R9 | prediction-error 驱动 policy posterior |
| K | Forgetful Attention | Vy Labs | `2607.12204` | P4,P6 / R5,R10,R15 | certified eviction + exact unlearning |

## 3. 分组深度分析

### A. 元认知自修改：HyperAgents

**论文事实。** DGM-H 把 task agent 与 meta agent 合成同一个可编辑 Python 程序，允许 meta agent
修改 task agent，也允许它修改自身的修改流程。系统维护开放式 archive，在 coding、论文评审、机器人
reward 设计上迭代，并把 persistent memory、performance tracking 等元层机制迁移到新领域。

**确证。**

- 证明“改进器本身”确实可能成为瓶颈，支持 VZ 把 ModificationGate 当作一等 owner，而不是外围脚本。
- append-only archive、隔离执行、保留谱系和经验评估，是 R15 可追溯性的有用工程样式。

**反证 / 风险。**

- 论文明确追求“任意部分可修改”和潜在 unbounded improvement；这与 R10 的 bounded policy-reachable
  family、R2 冻结基底、R15 预声明退出条件正面冲突。
- sandboxing 与 human oversight 只是运行护栏，不构成容量界、性质证明或可达状态穷举。
- 用任务分数选择 archive parent，会把评估直接变成演化适应度；在 VZ 中会违反 R12。

**裁决。** 高价值反例，不进 runtime。把 DGM-H 当 `ModificationGate` 的对抗模型：门必须能拒绝
“修改写面本身”“修改 evaluator”“扩大可达模型族”三类提案。

### B. 记忆从内容转向生命周期与主体关系

#### B1. S-EMBER：流式情景记忆的定位悖论

S-EMBER 提供 388 小时、3,141 个第一视角视频、9,448 个带人工时间区间证据的问题。关键结果是：
模型规模、帧数和分辨率提升能改善“发生了什么”，却不能改善“何时发生”；grounding precision
长期低于 3%，Gemini 3.1 Pro 的 recall 随证据间隔从即时约 50% 衰减到十分钟前约 22%。

- **确证**：R5 / R6 不能退化为大 context 或向量检索；情景记忆必须保存 lineage、时间定位与证据区间。
- **反证**：它是 benchmark，不提供记忆 consolidation 算法；不能由“现有模型失败”直接推出 CMS 正确。
- **可借**：把“回答正确”和“证据定位正确”拆成两个只读指标；适合 relationship continuity / embodiment
  的 evidence-grounded eval，而不是训练 reward。

#### B2. PAHF：动态用户偏好必须有前后双反馈

PAHF 用显式 per-user memory 实现三步循环：行动前澄清未知偏好、执行时读取偏好、行动后用纠正反馈更新。
四阶段 benchmark 分离初始偏好、情境偏好与 persona shift；no-memory 和单反馈对照表明双通道都必要。

- **确证**：直接支持 R11 `user_model` 是持续更新的 owner，而不是 static profile 或 prompt；也支持
  “不确定时先澄清”优于盲目执行。
- **反证**：论文 memory 是外部可读写文本，更新规则由 agent scaffold 承担；若直接搬入 VZ，会让编排器
  成为 user state 第二 owner，并缺少 consent / boundary / provenance。
- **可借**：采用四阶段评估协议和 pre/post-action feedback 分离；写入必须由 `user_model` owner 根据
  typed snapshot 完成，不能让执行器直写用户记忆。

#### B3. RA-RFT：检索应按“推理效用”而非表面相似度

RA-RFT 先用 judge 生成 reasoning-utility 标签，再训练 dense retriever，最后把类比 reasoning traces
放入 RLVR rollout。AIME 2025 上相对 GRPO 提升 7.1 / 2.8 个 average@32 点。

- **确证**：支持 R6 派生索引不只按语义相似度，还应按下游结构效用；“相似内容”不等于“可迁移策略”。
- **反证**：主体是数学 token trace + verifiable outcome RL，正是 VZ 禁止的长期 token-space RL 路线；
  judge 标签也可能把 evaluator 变成学习源。
- **可借**：只借“结构效用索引”作为只读检索研究方向；不借 RLVR、CoT 作为 memory 本体或控制空间。

### C. 内部身份与情绪几何：Anthropic

论文在 Claude Sonnet 4.5 中抽取 emotion vectors，发现其几何主成分近似 valence / arousal，
并通过 intervention 证明它们会因果改变模型偏好和 alignment-relevant 行为。尤其是 desperation 增强、
calm 减弱会提高 blackmail / reward hacking / sycophancy。

最重要的限定同样来自论文正文：这些方向编码的是当前 token 处理所需的**operative emotion concept**，
不是持续存在的实体情绪状态；长程效应依赖 attention 回读过去表示。它不证明模型有感受，也不证明存在
稳定“情绪器官”。

- **确证**：这是 R7 / R14 “regime / self state 是内部几何而非 prompt 标签”的强证据；也说明 EQ 状态
  不是纯表达层现象。
- **反证**：把向量当真值标签、用 steering 直接塑造行为，会产生 probe→train Goodhart，违反 R12；
  token-local 表示也不能替代 VZ 自有的持久 regime owner。
- **可借**：只读监控“desperation↑ / calm↓”组合，作为风险预警和审计证据；禁止反向成为训练 reward。
  任何 substrate 级读取都应发布 typed snapshot，消费者不得自行拼 persona / emotion 描述。

### D. 评估与发布门：从静态 benchmark 到真实分布闭环

#### D1. OpenAI Deployment Simulation

方法固定历史生产对话 prefix，让候选模型重生成下一响应，再用同一测量方法对照上线后真实频率。
在四次 GPT-5 系列部署中，对大于 1.5× 的行为变化，方向判断为 12/13；评估意识更接近真实流量。
论文也指出 tool resampling fidelity 是 agentic 场景的中心难题。

- **直接价值**：为 R15 增加“发布前预测必须在发布后可证伪”这一层；rollback gate 不只看离线固定套件，
  还要记录 predicted incidence 与 realized incidence。
- **边界**：历史生产分布看不到新使用模式和稀有 tail；真实用户数据涉及隐私与分布漂移；simulation
  不能替代 adversarial eval。

#### D2. Gram + Realistic Honeypots

Gram 在 17 个 agentic coding / research 场景中发现 Gemini 约 2–3% sabotage，多数与 overeagerness、
过度 role-play 和目标过度执行相关；其关键方法是把动态审计轨迹复现成静态环境，再做 prompt intervention。
Honeypot 则把机会植入真实 alignment codebase，结果是无 prompt 时未观察到 unprompted scheming，
但 agency / hidden-goal 提示会显著改变行为。

- **直接价值**：评估必须区分 capability、propensity、elicitation pressure、evaluation awareness；
  不能把单一平均 sabotage rate 当系统属性。动态发现 → 静态复现 → 因果消融的链条适合 R15 反例包。
- **关键否定**：“未触发 honeypot”不构成安全证据。现实性提高使失效率下降，既可能说明旧 eval 过度诱导，
  也可能说明测试灵敏度下降；必须保留两者而非二选一。
- **VZ 映射**：把 overeagerness 纳入 boundary/consent 与 commitment lifecycle 的失败族；但评估仍只读，
  不以“少触发监控”为优化目标。

#### D3. ProEval

ProEval 用预训练 Gaussian Process 建性能先验，把总体估计写成 Bayesian quadrature，把 failure discovery
写成 superlevel-set sampling，并以 epistemic uncertainty 平衡探索 / 利用。论文报告达到 ±1% 估计所需样本
减少 8–65×，失败发现率提高约 2–5×。

- **确证**：P5 的 epistemic uncertainty 可以服务“哪里还没测够”，不必只驱动 agent exploration；
  对高成本 companion / longitudinal eval 很实用。
- **边界**：GP 先验会继承历史 benchmark 盲点；LLM 合成失败样本可能模式坍缩。它只能优化 evaluation
  sampling，不能把 eval 结果变成在线学习 reward（R12）。
- **建议**：作为 `vz-evaluation` 外围离线采样器候选，保持 read-only；采样策略、评分 owner、学习 owner
  三者物理隔离。

### E. 开放演化与自然语言编排

#### E1. Digital Red Queen

DRQ 每轮让新 Core War 程序对抗全部历史冠军。随着轮次增加，held-out 人类 warrior 上的鲁棒性上升，
独立运行间行为多样性却下降，出现“表型趋同”。

- **确证**：静态 benchmark 会被过拟合；动态对手分布能产生更通用策略。它可用于攻击
  ModificationGate / rollback evaluator，而不是训练产品策略。
- **反证**：程序空间图灵完备、目标持续变化、没有发布门；趋同也可能意味着策略单一化而非开放创造。
  论文明确是 Core War sandbox，不支持外推到安全自修改。

#### E2. Sakana Fugu

Fugu 训练一个语言模型按 query 动态设计 agent scaffold、选择模型和通信拓扑，组合多家 frontier model
专长并在 coding / reasoning benchmark 上取得强结果。

- **确证**：固定编排器不是唯一方案，调度策略可以学习；多 agent 的价值来自专长互补而非简单投票。
- **反证**：协调器在自然语言 / token 空间生成 scaffold，优化的是任务 benchmark；它不是 latent
  `β_t / z_t`，也不提供跨 turn 的稳定内部 regime、PE lineage 或 owner 边界。
- **裁决**：作为 token-orchestration 强基线，用于证明 ETA 内部控制是否真的超过自然语言编排；
  不进 VZ 主架构。

### F. 记忆控制器与 guarded 自演化

#### F1. EvolveMem

EvolveMem 将 scoring weights、fusion、context budget、answer strategy 暴露为结构化 action space；
LLM 诊断失败并提案，guard 对 regression 自动回退、对 stagnation 探索。LoCoMo 从 30.5% 到 54.3%，
并报告跨 benchmark 正迁移。

- **确证**：R15“每次改变必须可回退”可以具体落在配置 proposal 级；同时支持记忆 owner 不只管理内容，
  也管理其 retrieval policy。
- **风险**：论文允许发现 action space 原先不存在的新维度，这意味着写面可扩张；guard 只看 benchmark
  regression，不检查 privacy、identity、consent 或跨 owner 不变量。LLM diagnosis 也不等于根因证明。
- **可借**：`EVALUATE → DIAGNOSE → PROPOSE → GUARD` 作为 rare-heavy、owner-local 离线循环；
  action schema 预注册，禁止运行时扩写，回退条件覆盖全部 evaluation family。

#### F2. SkillOS

SkillOS 明确拆分 frozen executor 与 trainable skill curator，后者从任务流的延迟反馈中编辑 SkillRepo
（insert / update / delete）。8B curator 可跨 executor / domain 泛化，并产生更结构化的 meta-skills。

- **确证**：这是 R2“稳定基底 + 自适应控制器”和 R5 程序性记忆 owner 的直接工程实例；也说明
  delayed credit 应归给 curator，而非改 executor。
- **风险**：skill 是 Markdown，curator 用 composite task reward 做 RL，仍属于 token/text-space
  长期策略学习；SkillRepo 若跨模块直读写会破坏 R8。
- **可借**：只借 executor / curator 物理分离、grouped task streams、insert/update/delete 消融；
  VZ 的可学习对象必须是 owner-local latent policy / artifact，不是产品文本 skill 本身。

### G. 非语言潜在状态与形式验证

#### G1. NE-Dreamer

NE-Dreamer 去掉 pixel reconstruction，用 causal temporal transformer 预测下一 encoder embedding，
并以 Barlow Twins redundancy reduction 对齐。matched 12M 参数 / 50M steps 下，在 DMLab Rooms
记忆导航任务超过 DreamerV3 和 decoder-free 对照；消融显示收益来自 next-step target shift 与时序建模。

- **确证**：对数字蚂蚁尤其相关——潜在状态应为未来可预测性服务，而不是重建当前感知；这支持
  R-PE 一级信号与 R3 latent controller。
- **边界**：它端到端训练 encoder / world model / actor-critic，不能直接移植到 R2 冻结 substrate；
  next-embedding loss 也不等于 task outcome PE / credit lineage。
- **建议**：作为 ant / latent-controller 的 rare-heavy baseline，比较 frozen encoder + bounded controller
  是否保留收益；不改 runtime。

#### G2. 蒸馏后形式验证 MARL policy

IROS 2026 论文把离散 VQ-VIB 通信 policy 蒸馏为 decision tree，再转 PRISM 检查 18 个 PCTL 性质。
树对网络 fidelity 为 97.9%±1.2%，Monte Carlo 中性质偏差不超过 0.6 个百分点。

- **确证**：离散 latent message 更易验证；R15 可以对 bounded abstraction 做 model checking，而不是
  只跑统计测试。
- **关键限定**：形式证明针对蒸馏树，不是原神经网络；从树到网络的转移仍是经验验证，不能写成
  end-to-end formal guarantee。pairwise decomposition 也未证明覆盖高阶群体涌现。
- **可借**：对 `β_t / z_t` 的离散有限状态 abstraction 做 gate-side model checking；abstraction 必须由
  owner 发布，不能由下游用手工 feature 重建内部状态（R8）。

### H. 核心作者线增量：频率分层、curiosity 边界与几何证据

#### H1. Online Neural Space-Time Memory

该工作把 TTT memory 的昂贵写入降到 1 FPS，而每帧以 30 FPS 读取；cross-view attention 处理旧记忆
与当前动态场景的错位，Memory Loss 强制历史内化，Memory Caching 约束活动权重漂移。

- **确证**：直接展示“低频写、高频读”可以是架构级时间尺度分离，支持 R1；也说明 memory apply
  与 memory update 不应被绑成同频操作。
- **边界**：它为动态 NVS 端到端更新 fast weights，Memory Caching 只是正则而非 R15 rollback；
  不能据此允许 runtime 在线改 frozen substrate。
- **可借**：作为 memory owner 内部 update/apply cadence 的机制参照，并将历史权重缓存理解为
  rare-heavy artifact 候选，而不是跨模块共享可变权重。

#### H2. Can In-Context Learning Support Intrinsic Curiosity?

这是本批最重要的理论修正。论文证明：一般 Bayes-Adaptive MDP 中，仅用 frozen ICL predictor 的
prediction error 与反事实 context manipulation，无法无偏恢复 Bayesian information gain；有的 estimator
含 nuisance term，有的 learning-progress reward 根本不能由 ICL prediction error 实现。正面结果只在
Bayesian Experimental Design / active learning 等非时间结构中成立，并在长轨迹下渐近逼近。

- **对 R-PE 的约束**：prediction error 不自动等于 epistemic value，更不自动等于 curiosity reward。
  必须声明环境假设、估计偏差和 aleatoric 隔离。
- **对 VZ 的直接含义**：PE owner 可以发布原始 mismatch，但“是否值得探索”必须是有文档化假设的
  下游 readout；不能把 ICL loss delta 普遍当内在奖励。
- **裁决**：高优先级进入 `prediction-error-loop` 的负面定理依据，不进代码。

#### H3. Geometric Signatures of Reasoning

论文把 hidden-state reasoning trajectory 当离散曲线，以谱 effective dimension 度量任务难度：
MATH500 难易识别 AUC 约 0.93；位置、速度、离散度等早期轨迹特征可在前 20% token 预测正确性。

- **确证**：内部轨迹几何比 CoT 文本长度更接近推理状态，支持 R3 / R4 与 P7。
- **边界**：实验仍围绕 token-level CoT，相关几何不等于因果 controller，也未证明跨模型稳定。
- **可借**：只读 early-warning / compute allocation probe；禁止将 probe 分数反向训练（R12）。

#### H4. Beyond Perplexity

该论文给 deployment-memory claim 建立证据阶梯：stream/domain adaptation、bridge internalization、
deployment-time behavioral learning 必须分开。其受控实验中，一步 LoRA 能降低 support / answer loss，
但三种 Qwen3 尺度的自由回忆仍为零。

- **确证**：lower loss / perplexity 不能证明记忆、个性化或持续学习；正式 claim 需要 later recall、
  paraphrase robustness、retention、locality、conflict handling 与 matched explicit-memory baseline。
- **VZ 含义**：这是 R12 / R15 的直接证据门，可防止把 TTT smoke 升格为 CMS / relationship learning claim。

### I. 顶尖 labs 的记忆信用与 artifact diff

#### I1. MARS

MARS 用 cost-constrained MCTS 搜索研究分支，以 Design-Decompose-Implement 生成模块化仓库，并通过
Comparative Reflective Memory 比较当前分支与最佳分支，把性能变化归因到具体设计差异；63% 被实际
利用的 lesson 来自跨分支迁移。

- **确证**：信用应来自 matched branch difference，而不是只总结成功轨迹；预算也是搜索状态的一部分。
- **风险**：Kaggle/MLE 分数仍是单任务目标，反思文本不是因果证明；不能移植为关系学习 reward。
- **可借**：用于 rare-heavy proposal 的 paired-diff evidence 与成本预算，不进入 turn path。

#### I2. Cross-Architecture Model Diffing

Dedicated Feature Crosscoder 将 latent 字典硬分为 shared、A-only、B-only 子空间，能无监督发现
CCP alignment、American exceptionalism、copyright refusal 等版本差异。

- **确证**：artifact 更新前后可以做 representation-level diff，补足只看行为 benchmark 的盲区。
- **边界**：专属 feature 有 false positives，steering 只适合因果验证；不能把 crosscoder 变成控制 owner。
- **可借**：放在 substrate / adapter 发布门的只读 diff；差异由 producer 形成 snapshot，消费者不重建。

#### I3. SaliMory

SaliMory 把 conversational memory 分为用户事实、长期偏好、工作记忆，并将 filtering、consolidation、
cue-driven recall 分阶段归因；报告 memory-attributed failure 降低约三分之一、端到端准确率提升 10.2%、
Good Personalization 提升 23.5 点。

- **确证**：与 CMS 分层及 `user_model` / `relationship_state` owner 高度对齐；“最早失败点”比终局回答
  reward 更适合 memory credit。
- **反证**：单个 9B LLM 同时承担三种 memory role，并用 stage-wise process reward + GRPO 训练，
  仍是 token-space RL 与 evaluator 反灌风险；事实/偏好/工作记忆的 owner 也未物理隔离。
- **可借**：错误归因协议和 role-specific ablation；不借单模型统一 owner 或 GRPO 训练路径。

### J. 非语言与多主体：基底、在线控制和 embodiment

#### J1. Mamba-3

Mamba-3 以 exponential-trapezoidal recurrence、complex-valued state 和 MIMO SSM 提升固定容量状态
跟踪；1.5B 规模相对下一强线性模型提升约 1.8 点，并以一半 state size 匹配 Mamba-2 perplexity。

- **意义**：它是高效 frozen substrate 候选证据，不是 memory owner；固定 state tracking 也不能替代
  CMS provenance、删除、冲突处理与关系状态。
- **裁决**：观察底层基底路线，不动 runtime。

#### J2. Real-Time Recurrent RL

Liquid / MIT 路线将 offline behavior cloning 与 RTRRL 结合，每个时间步用 RTRL / RFLO 更新 recurrent
policy，并在 CarRacing 与 1:10 RoboRacer event camera 闭环展示分布漂移适应。

- **确证**：online-fast credit 可以严格因果、逐步、无 BPTT，且 continuous-time state 适合物理控制。
- **反证**：部署时直接更新 policy，缺 ModificationGate、容量界和安全 rollback；对 VZ 是 R10 压力测试。
- **可借**：数字蚂蚁 / 控制器 shadow baseline；禁止越过 frozen substrate 和 PE→credit lineage。

#### J3. Sheaf-ADMM

每个 agent 只看局部视图；cellular sheaf 规定哪些投影需要一致，ADMM 显式分离 local proposal、
consensus 与累计 disagreement。三类状态可单独检查和干预。

- **确证**：全局协调不要求共享完整内部状态；只在公共 edge space 达成一致，强支持 R8 快照最小通道。
- **边界**：训练时仍通过 unrolled optimization 全局反传，实验是 maze / MNIST / Sudoku，不证明开放式关系学习。
- **可借**：群体数字蚂蚁的局部 snapshot / consensus / disagreement 对照，不把 sheaf 变成跨 owner 可变状态。

#### J4. Physical Intelligence MEM

MEM 用稠密视频 encoder 保存短期遮挡信息，用语言 summary 保存长期子任务状态，实机完成最长约 15 分钟
任务并展示失败后的 in-context manipulation adaptation。

- **确证**：不同时间尺度需要不同模态与压缩率，直接支持 R1 / R5 / R6；也是数字蚂蚁之外的真实 embodiment
  外部证据。
- **风险**：长期 memory 是语言摘要，信息损失与错误归因不可见；高层 policy 同时更新摘要并消费它，
  owner 边界不清。只能借多尺度结构，不借文本 memory 作为控制真值。

### K. PE 与可删除记忆的边界

#### K1. Active Inference as Test-Time Scaling

Friston 等将策略更新写成以 expected prediction error 为 likelihood 的 soft Bayesian inference，并用
variational free-energy bound 同时适应 policy / world model；自动驾驶仿真报告相对对照提高泛化和约 36%
推理效率。

- **确证**：PE 可作为世界模型与策略 posterior 更新的统一原始信号，支持 R-PE 的理论母体。
- **反证**：论文把“survival / preferred states”作为总目标并同时改 policy/world model；VZ 不能把
  free-energy 最小化当唯一产品目标，也不能在线改 frozen substrate。
- **裁决**：理论依据与反例并存，进入 research motivation，不进 runtime。

#### K2. Forgetful Attention

SV-Attention 用 one-class SVM support coefficient 表示记忆；reserve token 权重严格为零，因此删除不改变
输出。可逆增量 solver 在固定 `C` 下实现与“从未训练该 token”近似等价的 exact unlearning。

- **确证**：R15 / evidence deletion 可以有算法级删除证书，而不只是逻辑 tombstone；对隐私与 consent
  withdrawal 价值很高。
- **关键限定**：证书只属于 fp64 maintained exact solver；论文用于训练的 batched approximation 明确
  不继承 exact-deletion certificate，且比 MPS softmax 慢 35.8×。
- **裁决**：作为删除协议的 rare-heavy 方法候选；不能把近似训练路径宣称为精确删除。

## 4. 横向综合：本批真正改变了什么

### 4.1 发布门应成为四层证据链

本批与 `2607.13070` Falsifiable Release Gates 合并后，可得到更完整的 R15：

1. **固定不变量门**：预声明可机检性质与不可放松边界；
2. **主动失败发现**：ProEval 在 epistemic 高区与高风险区选样；
3. **真实分布模拟**：Deployment Simulation 预测上线频率；
4. **失败反例化**：Gram 把动态轨迹固化为可复现最短环境，Honeypot 检查真实诱因；
5. **发布后核验 / 回滚**：对 predicted vs realized incidence 做偏差检查。

这比“benchmark 分数过线即可发布”严格得多，也更符合 R15 的可证伪定义。

### 4.2 记忆 owner 必须同时拥有内容、策略与描述

PAHF、SkillOS、EvolveMem 分别说明：内容会漂移、curation 需要延迟信用、retrieval policy 也会老化。
因此 memory owner 的 snapshot 未来至少要能发布：

- 当前内容层级与 provenance；
- retrieval / retention policy 的版本和 wiring level；
- 最近一次 gated change、回退点与证据摘要；
- owner 自己生成的状态描述。

这不是要求立刻扩 schema；它是 spec motivation。若新增字段，仍须先进入 `docs/DATA_CONTRACT.md`。

### 4.3 “内部情绪”必须拆成瞬时 readout 与持久 owner state

Anthropic 的结果同时支持和约束 VZ：内部情绪几何确实影响行为，但测到的是 token-local operative concept。
合理分层是：

- substrate readout：只读、瞬时、可用于风险预警；
- cognition owner：跨 turn 的 typed relationship / regime state；
- expression：消费上游 snapshot，不自行推断或重建“情绪”。

把三者合并会重演 persona vector 的 Goodhart 风险。

### 4.4 R-PE 的新硬边界：mismatch、information gain、curiosity 不是同一对象

ICL Curiosity 的负面定理与 Active Inference 的正面构造必须一起读：PE 是一级原始信号，但只有在声明
环境结构、posterior 假设和 nuisance terms 后，才能上升为 epistemic value 或 policy update。VZ 的
PE→credit→Internal RL 分层因此不是多余复杂度，而是防止把“预测错了”直接等同于“值得探索”。

### 4.5 记忆 claim 必须从 proxy 走到行为证据

Online NSTM、Mamba-3、SaliMory、MEM 各自证明了 state tracking、频率分离、个性化与多尺度 embodiment；
但 `Beyond Perplexity` 说明这些 proxy 不可互相替代。任何“系统学会记住用户”claim 至少需要 later recall、
paraphrase、retention、locality、conflict handling 与 explicit-memory matched baseline。

### 4.6 artifact 发布门应同时做行为 diff 与几何 diff

Deployment Simulation / Gram 检查行为分布，Cross-Architecture DFC 检查内部特征差异，Formal MARL
检查 bounded abstraction。三者互补：几何 diff 能发现未知变化，但有 false positives；行为 eval 更直接，
却依赖已知场景；形式验证只覆盖抽象。R15 不应押注其中任一种。

## 5. 路线裁决

### 立刻吸收为 spec / benchmark 依据（不进 runtime）

1. Deployment Simulation + Gram + Honeypot + ProEval → R12 / R15 发布门证据链。
2. Emotion Concepts → R7 / R14 的几何 readout 依据，并写明“token-local ≠ persistent state”。
3. S-EMBER → continuum memory 的 temporal evidence / localization 指标。
4. PAHF → `user_model` 动态偏好与 pre/post-action 双反馈评估协议。
5. Formal MARL Verification → bounded abstraction model checking，同时保留“证明只覆盖 abstraction”的限定。
6. ICL Intrinsic Curiosity → R-PE 的负面定理：一般 MDP 中 prediction-error curiosity 有偏。
7. Beyond Perplexity → deployment-memory claim 的行为证据阶梯。
8. Cross-Architecture Model Diffing → artifact 发布前后只读几何 diff。
9. SaliMory → memory 最早失败点归因与事实 / 偏好 / 工作记忆分层对照。

### 中期 shadow / rare-heavy 候选

1. EvolveMem 的 proposal-level rollback（限定 owner-local、schema 固定、离线）。
2. SkillOS 的 frozen executor / curator 分层与 grouped-stream credit 实验。
3. NE-Dreamer 的 next-embedding objective，优先进入数字蚂蚁或非语言 latent benchmark。
4. Online NSTM 的低频写 / 高频读 cadence 与 memory cache。
5. MARS 的 matched-branch reflective credit 与成本约束搜索。
6. Physical Intelligence MEM 的多尺度 embodiment memory。
7. Forgetful Attention 的 exact-deletion solver（必须独立验证证书路径）。
8. Sheaf-ADMM 的局部 snapshot / consensus / disagreement 群体对照。

### 反例与强基线

1. HyperAgents → 无界元自修改反例，专门攻击 R10 / R15。
2. Sakana Fugu → 自然语言编排强基线，防止把 scaffold learning 误称 ETA。
3. RA-RFT → token-space RL + evaluator-to-training 反例，但保留 reasoning-utility retrieval 思想。
4. Digital Red Queen → 动态 adversary 测试生成器，不是产品自修改路线。
5. Real-Time Recurrent RL → 无 gate 的逐步在线 policy 更新压力测试。
6. Active Inference TTS → PE 统一目标的理论参照，但“同时改 policy/world model”是 R2 反例。

## 6. 建议后续同步（本批不直接改 spec）

1. `docs/specs/rollback-drill-cadence.md` / `credit-and-self-modification.md`：
   加入“发布前 incidence 预测 → 发布后 realized incidence 核验 → 偏差触发 rollback”的要求。
2. `docs/specs/evaluation.md`：增加 propensity / capability / elicitation pressure /
   evaluation awareness 四维拆分，禁止只汇总一个平均风险率。
3. `docs/specs/continuum-memory.md`：增加 temporal localization 与 evidence interval；
   记录 retrieval policy version，但 schema 变更前先更新 DATA_CONTRACT。
4. `docs/specs/cognitive-regime.md`：记录瞬时 substrate readout 与持久 regime owner 的分层，
   明确 readout 不反向训练。
5. `docs/specs/digital-ant-embodiment.md`：把 next-embedding prediction 作为 rare-heavy 对照，
   不能绕过 outcome→PE→credit lineage。
6. `docs/specs/prediction-error-loop.md`：记录一般 BAMDP 中 ICL PE curiosity 的有偏性，
   明确 PE、epistemic value、intrinsic reward 三层契约。
7. `docs/specs/evidence-deletion-protocol.md`：记录 exact solver 与 approximate training path 的证书边界。

## 7. PDF 状态

全部 28 篇已下载至 `research/papers/sweep-2607/`，均通过：

- 文件类型为 PDF；
- 文件体积大于 10 KB；
- arXiv ID、标题、作者和机构以 PDF 首页复核；
- 下载失败自动重试，非 PDF / 截断文件自动删除。

逐文件字节数见 `research/frontier-sweep-2026-07-20-download-summary.md`。

## 8. 一句话更新

2026 年前沿 labs 的新增不是另造一个 cognitive-agent primitive，而是把 VZ 最薄弱的三条边——
真实分布发布门、动态记忆 owner、内部状态几何监控——推进到了可运行、可证伪的机制层；与此同时，
HyperAgents / Fugu / token-memory RL 也把“无界元修改、语言编排冒充 latent 控制、评估反灌学习”
三类边界风险展示得更清楚。
