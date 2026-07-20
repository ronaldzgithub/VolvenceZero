# 架构、学习、记忆与信用：25 篇逐篇深读

日期：2026-07-20  
范围：核心 sweep 10 篇 + `frontier-map-2024-2026.md` Axis A 15 篇。以下判断来自逐篇核读本地 PDF 的方法、公式、实验表和消融，不以既有研究摘要代替正文。

## 0. 本轴总判断

这 25 篇共同否定了“给 LLM 加一段文本 memory 就得到了持续学习系统”这一捷径。证据真正收敛到五个接缝：

1. **基底与控制器必须分层。** SkillOS 的 frozen executor / trainable curator 是 R2 的直接工程正例；Mamba-3 是高效固定容量基底候选；Online NSTM 的 fast weights、RENEW 的世界模型修复和各种文本 curator 都只能作为 owner-local、受门控的适应层或 rare-heavy 工件，不能在线改写 frozen substrate。
2. **记忆的核心是生命周期、证据和写入信用，不是容量。** S-EMBER 把“答对”与“定位到证据”拆开；Beyond Perplexity 证明 loss 降低可与自由回忆为零并存；DeltaMem、EvoLib、EvolveMem、AdaMEM 则分别触及 residual storage、抽象演化、retrieval policy 演化和快慢记忆，但多数仍把自然语言文本当作策略载体。文本 memory 可以是可审计工件或检索证据，**不是 latent controller `z_t/β_t`**。
3. **抽象只有在 support 与可控性成立时才缩短 horizon。** LMTA 给出 SMDP-aware 多时间尺度正证据；Mind the Gap 直接显示自由搜索 latent macro-action 会被 planner 利用；CLAW 显示 latent action 需要防未来信息泄漏。因而“层级化”“latent 化”都不是自动收益。
4. **PE、information gain、reward、evaluation 必须分开。** ICL Curiosity 给出一般 BAMDP 上的负面定理：prediction error 不能无偏恢复 Bayesian information gain。RA-RFT、SkillOS、EvoLib、EvolveMem 等把 judge、任务得分或自评送回学习回路，虽有工程收益，却构成 R12 的 Goodhart 压力测试。VZ 应让 PE owner 发布原始 mismatch；curiosity、credit、evaluation 分别作为有假设的 readout。
5. **任何自修改都要有预注册写面、证据门和回滚。** EvolveMem 的 revert、MARS 的 paired branch diff、Forgetful Attention 的 fixed-`C` retrain-equivalence、RENEW 的 uncertainty-directed repair 都提供局部机制；但只有组合 R15 的行为证据、几何/状态 diff、删除验证、真实分布复核与 rollback，才足以放行。

对 VZ 的硬边界如下：

- **R2：** frozen substrate 与 adaptive controller 分离；在线 fast/session 层不得悄然变成基底训练。
- **R8：** 每个 owner 通过不可变 snapshot 发布状态、provenance、policy version 与自描述；检索器、编排器和 evaluator 不重建生产者内部状态。
- **R12：** evaluation 只读；benchmark 分数、LLM judge、自评置信度和 probe 不得直接成为在线 reward。
- **R15：** 每个可写层必须有固定 schema、变更前后证据、退出条件、回退点和可复现反例；“平均分提高”不能替代可证伪发布门。

---

## 1. 核心 sweep（10 篇）

### 1.1 S-EMBER：Streaming Egocentric Memory Retrieval（2607.02689）

1. **论文事实。** Xiaodong Wang、Xuanyi Zhao 等，Meta FAIR / Reality Labs，2026 年 arXiv 预印本。数据为 613 名用户用 Ray-Ban Meta 眼镜采集的 3,141 段第一视角视频、388 小时、9,448 个问题；平均视频 7.4 分钟，覆盖 2,090 个细场景标签与 8 类任务。每题含触发时刻、三档答案及人工证据区间。
2. **核心问题。** 要证伪的是“长视频答题准确率足以代表流式情景记忆”。GSER 协议只允许看到查询时刻以前的帧，并要求同时回答“发生了什么”和“证据在何时”。
3. **机制拆解。** 状态是因果视频前缀 `V[0,T]`；目标输出答案与 `[t_start,t_end]`。指标分为 clean/overall accuracy、5-way MCQ、mIoU、`R@1(IoU≥0.5)` 和联合 `GQ@0.5`。证据区间的冗余 padding 不得超过区间 15% 或 15 秒。
4. **关键证据。** Gemini 3.1 Pro clean accuracy 42.88%、mIoU 0.479、GQ@0.5 0.297，人类分别为 91.43%、0.777、0.698。Qwen3VL 帧数从 32 增至 768，准确率 8.8%→24.1%，但定位 recall 仅 0.1%→2.5%；分辨率 240p→720p 时准确率 14.4%→24.2%，定位仍约 2.5%。Gemini 的回忆从 0–30 秒约 50%降至十分钟以上约 22%。密集 caption 的 Socratic baseline 定位强但幻觉率 54.36%。
5. **确证价值。** 强支持 R5/R6 的时间索引、证据 lineage 与“内容正确/来源正确”双指标；支持 R12 把 grounding 作为只读证据，不以流畅答案替代来源。
6. **反证价值。** 反证扩大参数、帧数、分辨率或上下文即可得到 episodic memory；也反证把 caption 文本日志直接等同可靠记忆。
7. **局部可借算法。** 引入流式因果截断、人工 evidence interval、`answer × grounding` 联合得分和 recency/evidence-duration 分桶；用于 embodiment 与 relationship continuity benchmark。
8. **不可外推边界。** 单视觉模态、5–20 分钟范围；开放答案仍依赖 judge（600 对 meta-eval 准确率 96.2%）；它是诊断集，不提供 consolidation 或 controller 算法。
9. **成熟度与裁决。** **A（主链评估证据）**。进入 memory benchmark motivation；不得把 GQ 或 clean accuracy 反灌在线学习（R12）。

### 1.2 RA-RFT：Reasoning by Analogy（2606.13680）

1. **论文事实。** Zilin Xiao、Qi Ma 等，Meta Superintelligence Labs / Rice University，2026 年 arXiv。以 Qwen3-1.7B/4B 为 policy，12.5k QuestA 训练题、OpenR1-Math-220K 检索库；64 张 H100，全参数 RLVR；评估 AIME24/25、HMMT25、BrUMO25。
2. **核心问题。** 证伪“语义相似的示例就是最有用的 reasoning memory”，改问检索项是否真正提高目标题的正确概率。
3. **机制拆解。** GPT-4o 为 query–trace 对产生 reasoning relevance 标签；Reason-ModernColBERT 用 InfoNCE：`-log exp(<eq,ec+>/τ)/Σc exp(<eq,ec>/τ)`。固定 retriever 取 top-1 trace，GRPO 以二值可验证答案形成组归一优势 `A_g=(r_g-r̄)/σ_r`，在相同检索条件下更新 token policy。
4. **关键证据。** 1.7B 相对 GRPO 在 AIME25 提升 7.1 点、四集均值 43.3→47.4；4B AIME25 +2.8、BrUMO +5.9。RA-SFT 仅 40.5，RA-RFT 47.4；随机 trace 37.6；reasoning-supervised ColBERT 的 R@1 43.5%，下游 47.4，未监督版 40.7。只在推理时给 GRPO 加检索反而 43.3→37.7，说明需训练时共适应。
5. **确证价值。** 支持 R6 派生索引应按结构效用而非表面相似度；支持 matched retrieval ablation。
6. **反证价值。** judge 产生标签、verifier 提供 reward、长期策略落在 CoT/token policy，正面暴露 R3/R4/R12 风险；“reasoning relevance”不是因果信用真值。
7. **局部可借算法。** 只借结构效用检索、随机/语义/结构 matched baseline 与 context diversity 分析；可离线研究 latent trajectory 的可迁移性。
8. **不可外推边界。** 仅竞赛数学、强可验证 reward、同粗题型内配对；全参数 RL 与 32k token rollout 成本高，无法外推关系、弱反馈或开放世界。
9. **成熟度与裁决。** **B（强基线/反例）**。保留为 token-RL 对照，不进入 VZ controller 主链。

### 1.3 EvolveMem：Self-Evolving Retrieval Architecture（2605.13941）

1. **论文事实。** Jiaqi Liu、Xinyu Ye 等，UNC Chapel Hill、UC Berkeley、UCSC，2026 年 arXiv。评估 LoCoMo 全 10 段会话（每段 369–689 turns、合计 1,986 QA）和 MemBench 28 样本；GPT-4o/GPT-5.1 backbone。
2. **核心问题。** 要解决“内容在增长而 retrieval infrastructure 永久冻结”的错配，并检验 LLM 是否能从逐题失败日志提出可回退的配置修改。
3. **机制拆解。** memory unit 为 `(content,type,embedding,metadata)`；BM25、dense、结构元数据三视图融合，最终分数含 importance、recency、entity reinforcement。配置 `θ` 包含 top-k、context budget、fusion、权重、answer style、分类型 override。循环为 `EVALUATE→DIAGNOSE→PROPOSE→GUARD`；若分数下降超过 `τ_rev` 回退 best-so-far，停滞两轮则随机探索，否则 clamp 后应用提案。
4. **关键证据。** LoCoMo GPT-4o F1 从 BM25-only 30.5% 经 7 轮到 54.3%，强于 SimpleMem 43.2%；MemBench 67.9%，强基线 57.1%。跨集继续演化配置在 MemBench 达 79.2%，LoCoMo 反升至 59.3%。消融：去 extraction QC −23.22 F1，去 semantic −10.32，LLM diagnosis 换随机搜索 −9.63；R2 的 MMR 提案退化后被回滚。
5. **确证价值。** 支持 memory owner 同时拥有内容和 retrieval policy；proposal-level rollback 直接支持 R15。
6. **反证价值。** “根因诊断”来自 LLM 叙述而非因果证明；系统允许提出原 action space 没有的新维度，写面可自行扩张；F1 既选配置又证明配置，违反 R12 隔离。
7. **局部可借算法。** rare-heavy 离线循环、逐题 source log、best-so-far snapshot、固定范围 clamp、回退与停滞退出条件。
8. **不可外推边界。** 两个小型文本 benchmark；answer prompt 甚至要求“不可回答 unknown”，可能抬高 F1；无隐私、consent、跨 owner 或长期真实部署测试。
9. **成熟度与裁决。** **B（有价值的 gated prototype）**。只允许 owner-local、schema 预注册、evaluation 物理隔离的 SHADOW/rare-heavy。

### 1.4 SkillOS：Frozen Executor + Skill Curator（2605.06614）

1. **论文事实。** Siru Ouyang、Jun Yan 等，UIUC、Google Cloud AI Research、MIT，2026 年 arXiv。Qwen3-8B curator，Qwen3-8B executor 训练；ALFWorld、WebShop、AIME24/25、GPQA；16 张 H100，训练约 2.5–5 天，三次运行。
2. **核心问题。** 检验 delayed downstream feedback 能否训练独立 curator 管理程序性经验，而不修改执行器。
3. **机制拆解。** frozen executor 通过 BM25 读取 Markdown SkillRepo；curator 执行 insert/update/delete。每个训练样本是技能相关的任务组，repo 从空开始、逐题演化。复合 reward 为 `r_task + λ_f r_fc + λ_u r_content + λ_c r_compression`；GRPO 将组优势均匀赋给 curator 输出 token。
4. **关键证据。** ALFWorld 用 Qwen3-8B executor 时 success 由 strongest baseline 55.7% 到 61.2%，steps 20.1→18.9；Gemini-2.5-Pro executor 下达 80.2%，no-memory 66.4%。WebShop 和 reasoning 亦提升；8B curator 超过直接用 Gemini curator。消融：去 grouping 61.2→57.3，去 content reward→58.6，去 compression→60.0。跨 executor/domain 多数正迁移。
5. **确证价值。** 这是 R2“稳定执行基底 + 独立自适应 curator”和 delayed credit 归 owner 的强工程证据；insert/update/delete 也给出程序性 memory lifecycle。
6. **反证价值。** skill 是 Markdown，curator 的长期策略学习与 GRPO 都在 token 空间；外部 judge 和任务结果直接进入 reward，构成 R12 风险。Markdown 不能冒充 `z_t/β_t` latent controller。
7. **局部可借算法。** 借 executor/curator 物理隔离、相关任务分组、操作级消融和 repo compactness；把可学习对象改为 owner-local bounded latent/artifact。
8. **不可外推边界。** 文本环境、人工/LLM tag、BM25 检索；skills 的可审计性不等于可执行正确性，且没有跨用户隔离、回滚或恶意 skill 测试。
9. **成熟度与裁决。** **A（R2 架构正证据）/B（实现路线）**。架构进 motivation；文本 RL 实现不进 runtime。

### 1.5 Online Neural Space-Time Memory（2607.15271）

1. **论文事实。** Baback Elmieh、Lynn Tsai 等，University of Washington / Google，2026 年 arXiv。MVHumanNet++ 60k 多视角序列、4,500 identities、9,000 outfits；390 个长序列测试；JAX，最高使用 128 H100/B200 训练。
2. **核心问题。** 在线动态 novel-view synthesis 中，如何让昂贵记忆写入低频、便宜读取高频，同时避免 fast-weight 漂移和当前运动错位。
3. **机制拆解。** TTT memory 为 SwiGLU fast weights `W`，单步更新 `W'=W-η∇W||v-f_W(k)||²`；memorization 1 FPS、synthesis 30 FPS。cross-view attention 对齐旧记忆与当前姿态；隔离 memory tokens 的辅助重建损失强迫历史内化；每 `K` 个 checkpoint 的递归平均形成 memory cache。
4. **关键证据。** 写+读 58.14ms，仅读 27.01ms，1:29 调度摊销 28.1ms/frame。T=60 stress test，NSTM PSNR 29.99、mPSNR 20.71；LaCT-NVS 降至 25.85/16.25。把 dot-product inner loss 换 L2 给 LaCT +5.45 PSNR；完整模型 30.09，去 cache 29.24，去 memory loss 28.23；cache 直接加到 coupled LaCT 反降到 20.25。
5. **确证价值。** 强支持 R1 的 update/apply cadence 分离，也说明跨时间尺度机制必须配套对齐与专门监督。
6. **反证价值。** 在线更新的是 world-model fast weights，cache 只是平均正则，不是 R15 rollback；端到端训练与 VZ frozen substrate 不等价。
7. **局部可借算法。** 在 memory owner 内部测试低频写/高频读、L2 稳定更新、盲读辅助损失和 missed-event 指标；只作 shadow baseline。
8. **不可外推边界。** 任务仅动态 NVS、分钟级、256²；周期写会漏掉非写入帧事件，论文也展示 T=1 图案到 T=60 无法恢复；有限容量、晚期事件偏弱。
9. **成熟度与裁决。** **A（时间尺度机制证据）/B（VZ 迁移）**。不允许据此在线修改 substrate。

### 1.6 Can In-Context Learning Support Intrinsic Curiosity?（2606.19476）

1. **论文事实。** Eric Elmoznino、Sangnie Bhardwaj、Johannes von Oswald 等，Google Paradigms of Intelligence / Google DeepMind，2026 年 arXiv 预印本。含一般 BAMDP 形式结果与 GP、Mastermind、Alchemy 三组受控实验。
2. **核心问题。** 严格检验 frozen ICL predictor 的 prediction errors 与反事实 context manipulation 能否替代昂贵在线更新，并无偏实现 Bayesian information gain（BIG）。
3. **机制拆解。** `BIG=I(s_t;θ|h_t)`。surprisal `r_sur=-logρ(s_t|h_t)`；提出未来总学习进展 `r_sum=Σ_{t'>t}[logρ(s_t'|h_t')-logρ(s_t'|h_t'\t)]`。定理 1：一般 BAMDP 上任何有限个 posterior-predictive likelihood 的函数类 `F` 都不能普遍无偏识别 BIG。`E[r_sur]=BIG+aleatoric entropy`；`r_sum` 还含 one-step abductive 与 residual nuisance。只有 BED 条件独立结构中 abductive 消失，`r_sum` 随 `T→∞` 收敛 BIG，description-length reward 从另一侧形成上界。
4. **关键证据。** 有空间异方差噪声的 GP 中 surprisal policy 主动追 noisy tiles，验证分数随训练下降；`r_sum/r_dl` 与 task reward 接近。Noisy Mastermind/Alchemy 中 surprisal 低于随机，`r_sum/r_dl` 保持高信息采样。实验支持定理的边界，而非替代证明。
5. **确证价值。** 是 R-PE 最强负面约束：mismatch、epistemic value、curiosity reward 是三个对象；支持 PE owner 发布原始量，下游声明环境假设。
6. **反证价值。** 推翻“ICL loss delta 天然是好奇心”；也提示 active policy 导致 predictor covariate shift，frozen ρ 不一定继续 Bayes-calibrated。
7. **局部可借算法。** 把 noisy-TV、BED 与一般 temporal BAMDP 成对做 benchmark；对每个 curiosity estimator 报 nuisance、偏差方向和 aleatoric 隔离。
8. **不可外推边界。** 正实验都是 BED 式条件独立 trial；一般 MDP 只有负面理论，长 gap/double limit 实际不可算；PFN 仍依赖已知 prior。
9. **成熟度与裁决。** **A（主链理论反证）**。进入 prediction-error spec motivation；禁止直接把 PE 当 reward。

### 1.7 Beyond Perplexity（2607.00368）

1. **论文事实。** Xiangchen Song、Zhenhao Chen 等，CMU / MBZUAI，2026 年 arXiv。审计 41 个候选、编码 24 篇，并用 Qwen3-1.7B/4B/8B + LoRA 做 nonce-fact 受控诊断。
2. **核心问题。** 识别“evidence migration”：支持 stream/domain adaptation 的 loss、PPL、needle 或 reward 被升级为 deployment memory / personalization 主张。
3. **机制拆解。** 证据梯分 S（stream/domain）、B（bridge/internalization）、D（deployment behavior）。D 级要求移除 support 后的 direct/paraphrase/delay、locality、conflict 与 matched explicit-memory baseline。诊断定义 `ΔNLL=NLL_after-NLL_before`，把 support loss、answer NLL 与 greedy behavior 分开。
4. **关键证据。** 一步 LoRA 后三尺度 support ΔNLL 约 −0.97 至 −1.22，答案 ΔNLL 约 −0.46 至 −0.71，但 48 个 direct/paraphrase/delay 的自由回忆全部 0%。1.7B 16-step stress 可把 direct/delay 提至 35/48、paraphrase 26/48，却令 locality 从 141/144 降至 14/144。BM25 easy control 48/48，但 paraphrase+decoy top-1 为 0/48，stale/current top-1 为 0/24。
5. **确证价值。** 直接支持 R12 只读 claim gate 与 R15 的可证伪证据阶梯；也支持 R2 优先显式、可删、可审计 memory，而非把用户信息写进参数。
6. **反证价值。** 反证 lower loss、answer rank、context internalization 就是跨会话记忆；更强更新可用严重 locality 破坏换 recall。
7. **局部可借算法。** 采用 S/B/D 标签、互斥 conflict 类别（corrected-only/stale-only/both/neither）、匹配更新/检索预算和 governance probes（删除、跨用户隔离、consent）。
8. **不可外推边界。** LoRA pilot 不是代表所有 TTT；nonce facts 与 greedy containment 简化真实互动；审计为单人编码、非 prevalence estimate。
9. **成熟度与裁决。** **A（证据门标准）**。所有 deployment-memory claim 必须先过 D 级；evaluation 不得成为训练目标。

### 1.8 MARS：Comparative Reflective Search（2602.02660）

1. **论文事实。** Jiefeng Chen、Bhavana Dalvi Mishra 等，Google Cloud AI Research，2026 年 arXiv v3。评估 MLE-Bench 75 个竞赛，标准任务每次 24 小时、A100 40GB；受控比较 3 次运行。
2. **核心问题。** 自动 MLE 中，如何在训练昂贵、代码多模块、分数归因不透明时避免只优化单一 leaderboard score 的盲搜。
3. **机制拆解。** repository state 由多个 module + main 构成；Design–Decompose–Implement 产生可局部 diff 的仓库。MCTS 在 draft/debug/improve 上搜索，效率 reward `R(v)=G(v)[t(v)/L(v)]^w`，`w=-0.07`。Comparative Reflective Memory 对当前分支与 best branch 的代码/结果差分，提炼 isolated change、impact 与 generalized lesson。
4. **关键证据。** 受控环境 Gemini-3-Pro 下 MARS any-medal 56.0%、gold 31.1%，AIRA-dojo 为 37.8%/24.0%；MARS+ 62.7%/33.8%。去 modular、去 comparative memory、把 budget-aware MCTS 换 greedy/vanilla 都下降。有效解率 19.5% vs vanilla 16.1%；lesson utilization 65.8%，其中跨 branch 63.0%。3,611 条 lesson 的模型审计 causal accuracy 88.34%，20 条人工样本 90%。
5. **确证价值。** 支持 R9 信用来自 matched branch delta 而非成功轨迹总结；预算应进入搜索状态；diff 与谱系支持 R15。
6. **反证价值。** “causal lesson”仍是基于观察 diff 的 LLM 归因，不是随机因果实验；Kaggle 分数既导航又评价，不能移植到关系/主体学习。
7. **局部可借算法。** rare-heavy proposal 使用 parent/child paired diff、成本约束、模块化 artifact、lesson provenance 与 best-known rollback。
8. **不可外推边界。** 单 GPU MLE 竞赛、可执行验证、强 leaderboard；MARS 成本高于 AIRA（$60.5 vs $39），且自动研究不等于安全自修改。
9. **成熟度与裁决。** **A（rare-heavy 工程证据）/B（信用因果强度）**。只进入离线修改门，不进 turn path。

### 1.9 Mamba-3：State Space Principles（2603.15569）

1. **论文事实。** Aakash Lahoti、Kevin Y. Li、Berlin Chen 等，CMU、Princeton、Together AI、Cartesia，2026 年 arXiv。核心结果覆盖 synthetic state tracking、语言建模与 1.5B 下游评估，并发布训练/推理 kernels。
2. **核心问题。** 在线性计算、常数 memory 的约束下，能否恢复实值 SSM 缺失的状态跟踪表达力，并提高 memory-bound decode 的硬件利用率。
3. **机制拆解。** exponential-trapezoidal recurrence：`h_t=α_t h_{t-1}+β_t B_{t-1}x_{t-1}+γ_t B_t x_t`，是二阶 state-input 近似并隐式形成宽度 2 卷积。complex SSM 等价于 data-dependent RoPE 的旋转状态更新。MIMO 把 outer product 升为 rank-`R` matmul，在 state bytes 基本不变时把 arithmetic intensity 从常数提高到 `Θ(R)`。
4. **关键证据。** 1.5B MIMO 对 Gated DeltaNet 平均下游 +1.8 点，对 Transformer +2.2、Mamba-2 +1.9；state size 64 匹配 Mamba-2 size 128 perplexity。复杂旋转版本在 parity/模算术状态跟踪近乎满分，实值 Mamba-2/无 RoPE 接近随机；MIMO 固定 state size 下最多增加 4× FLOPs 而 decode latency近似不变。
5. **确证价值。** 支持 R2 稳定基底可以有紧凑、可解释 recurrence；也说明固定容量 state tracking 是 substrate 能力，不必冒充 CMS。
6. **反证价值。** 更强序列 state 仍没有 provenance、删除、冲突、用户隔离或 memory owner；PPL/任务分数不能证明持续记忆。
7. **局部可借算法。** 作为 frozen substrate / sequence-state 基线；在相同 state bytes、latency、state-tracking 上与 Transformer/Mamba-2 比较。
8. **不可外推边界。** 主要是离线预训练架构；无运行时自修改、跨会话记忆和开放环境控制证据；复杂状态的可读性也未证明。
9. **成熟度与裁决。** **B（强基底候选）**。观察 substrate 路线，不改 memory/controller 边界。

### 1.10 Forgetful Attention（2607.12204）

1. **论文事实。** Vishwajith Ramesh，Vy Labs，2026 年 arXiv。包含 synthetic recall、MIMIC-IV 1,465 个 ICU stays、MiniLM text retrieval、enwik8/TinyStories 小模型和精确 solver 数值验证。
2. **核心问题。** 测试时 memory 能否同时给出“某 token 可无损淘汰”的证书，以及删除后等价于从未训练过该 token 的 exact unlearning。
3. **机制拆解。** SV-Attention 用 one-class SVM 解 `min_α αᵀKα-diag(K)ᵀα`，约束 `Σα=1, 0≤α_i≤C`。reserve `α=0` 可由互补松弛证明不影响 readout。fp64 maintained KKT inverse 支持增量/减量；固定 primitive `C` 后，decrement 与 same-algorithm retrain-without 对比。训练另走 FISTA + `10^-3` ridge 的 batched approximation，**不继承证书**。
4. **关键证据。** unique optimum 下 partition match：Gaussian 99%、MIMIC 98%、trained keys 100%；decision-function median deviation 约 `10^-9`，trained keys `4.5e-13`，near-duplicate worst `1.1e-2`。稀有项 recall 0.86 vs H2O 0.32；MIMIC deterioration retention 0.80 vs 0.05。3.22M enwik8 七 seed BPC 2.178 vs matched window 2.383，`p=0.001`；但 10M 无优势。训练 path 比 MPS softmax 慢 35.8×。
5. **确证价值。** 为 R15 提供算法级删除证据范式：primitive hyperparameter 固定、与 retrain-without 比较、数值容差显式；支持 memory lifecycle 里的 certified eviction。
6. **反证价值。** 反证“近似训练路径也可宣称精确删除”；也显示 anomalous/outlier selection 与下游预测可能冲突（MIMIC AUROC 0.696，LSTM 0.783）。
7. **局部可借算法。** rare-heavy 测试 exact solver、固定-`C`删除、decision-function equivalence、certificate boundary；把近似和精确路径分开发布 snapshot。
8. **不可外推边界。** 证书只覆盖 fp64 long-range gate 的 readout influence，不覆盖 local path、训练 approximation、membership inference 或法规合规；near duplicate 条件数和空 margin 仍有边缘失败。
9. **成熟度与裁决。** **A（局部形式/数值证据）/C（通用 LM 层）**。作为删除协议候选，不宣称端到端系统“精确遗忘”。

---

## 2. Frontier Map Axis A（15 篇）

### 2.1 Learning Multi-Timescale Abstractions（2605.17058）

1. **论文事实。** Vivienne Huiling Wang、Tinghuai Wang、Joni Pajarinen，Aalto University / Qutwo，ICML 2026（PMLR 306）。评估 AIM、SOP 与 Power-2500 等 SSCO。
2. **核心问题。** 可变时长高层决策形成 SMDP；固定 stride world model 无法表达“做什么”与“投入多少预算”的联合抽象。
3. **机制拆解。** LMTA 的 MCTS 只在 learned subgoal dictionary 上分支，budget head 后置采样；`gθ(h,z)` 一步预测 budget-marginalized post-subgoal latent 与累计 reward。几何损失由 primitive-step cap、macro displacement push、按实现时长排序组成，使 `σ=d(h,g(h,z))+m(z)` 排序 effective duration。
4. **关键证据。** AIM N=500 多设置显著高于 WS-option，例如 T=20/K=10 为 161.71 vs 123.02；正文报告 SOP/Power-2500 和几何/预算/规划消融均支持完整模型。理论给出 margin 到期望时长下界、order consistency 与模型误差到 planning regret 的联系，但 planner 仍用 decision-time 固定折扣近似而非严格 `γ_LL^τ`。
5. **确证价值。** 支持 R1/R3：时间抽象需显式 SMDP contract，latent geometry 可承载 variable timescale；支持 intent 与 resource 分开命名。
6. **反证价值。** 决策时长没有进入 MCTS 分支而被 budget policy 边缘化，固定 decision discount 会改变真实时间偏好；“几何相关时长”不等于因果 option semantics。
7. **局部可借算法。** 多时间尺度 benchmark、micro/macro 联合损失、duration-order probe、budget-conditioned ablation；在数字蚂蚁中与固定 stride 对照。
8. **不可外推边界。** 组合优化图任务、离散 learned subgoals、DQN low level；不证明开放世界、关系或语言 agent 的 ETA。
9. **成熟度与裁决。** **A（多时间尺度主链证据）**。进入 R1/R3 benchmark motivation，先 shadow。

### 2.2 CLAW（2606.04130）

1. **论文事实。** Tewodros W. Ayalew、Matthew Jeung 等，University of Chicago、TTIC、Argonne，2026 年 arXiv。14 个模拟任务/5 环境及 3 个 UR5 实机任务；Something-Something V2 含 220,847 视频。
2. **核心问题。** 从无动作视频联合学习连续 latent action 与 world model，同时防止 latent 偷带 future frame。
3. **机制拆解。** LAM 以 `(o_i,o_{i+1})` 输出单位球面 32 维 `z_i`；U-ViT diffusion 预测未来帧。两条共享权重路径分别接收 `(o_i,z_i)` 与仅 `z_i`，后者前置 gradient reversal，使 `z` 单独预测未来变差，而联合路径仍需编码动作效果；推理用 MPC-CEM 搜 latent sequence，再以少量带 action reference 做 nearest-neighbor grounding。
4. **关键证据。** Crafter wood acquire 84%（AdaWorld 46.5、LAPO 38）；VP2 六任务中四项最佳；实机长程 Push-Cover progress 1.7（scratch 0.4、LAPA 1.3、AdaWorld 1.5）。最近邻/flow 显示 end-effector motion 聚类而 object identity 变化；但 Button 任务有失败。
5. **确证价值。** 支持 R3/R4 非语言 `z_t` 与 world dynamics 的联合可检验接口；adversarial leakage control 是 latent action promotion 必需项。
6. **反证价值。** 端到端 reciprocal supervision 可形成共适应假象；无 distractor 定量消融不足以证明完全去泄漏；执行仍需 action-labeled grounding。
7. **局部可借算法。** action transfer、nearest-neighbor control semantics、future-only adversary、MPC controllability 与 small-label grounding benchmark。
8. **不可外推边界。** world model 仅依赖最近帧+latent，无持久 memory；64²训练、模拟为主；不是 R2 frozen-substrate 证据。
9. **成熟度与裁决。** **A（latent action 强基线）**。rare-heavy 对照，不直接移植端到端更新。

### 2.3 Mind the Gap（2607.12547）

1. **论文事实。** Niccoló Caselli、Francesco Massafra 等，University of Amsterdam，WM@Booth 2026 workshop。PushT 与 OGBCube，各配置 50 episodes、3 seeds。
2. **核心问题。** 直接检验给 frozen LeWorldModel 加高层 macro-action planner 是否自然改善长 horizon。
3. **机制拆解。** 冻结 encoder/low-level predictor，仅训练 action-chunk transformer encoder `ℓ=g(a_t:t+k)` 与高层 predictor `ẑ_{t+k}=φ(z_t,ℓ)`。标准 CEM 自由搜索连续 `ℓ`；修复版从训练轨迹 macro sequence bank 采 anchor，仅优化 residual。
4. **关键证据。** PushT flat 在 d=25/50/75 为 94.0/52.7/18.0%；naive hierarchy 89.3/38.7/15.3。oracle subgoal 在 d=50 达 73.3%，定位失败在生成 subgoal。empirical-macro 在线在 d=75 达 32.7（+14.7），staged 在 d=50 达 64.0（+11.3）但 d=75 仅 22.0。VQ-128 在 d=75 达 34%，VQ-16 无益。
5. **确证价值。** 是 hierarchy 必须受 data support 与低层 executability 约束的直接因果消融；强支持 R15 promotion gate。
6. **反证价值。** 反证“加层级就缩 horizon”；terminal cost 可被 CEM exploit，且 stage commitment 的收益依时间尺度而变。
7. **局部可借算法。** oracle-subgoal decomposition、teacher-forced/true-open-loop/CEM-open-loop 三分诊断、support bank residual search、OOD subgoal rate。
8. **不可外推边界。** compact frozen LeWM、两任务、workshop paper；不否定高容量或联合训练 hierarchy。
9. **成熟度与裁决。** **A（关键反证）**。若 OOD subgoal 且 support constraint 不能恢复，kill hierarchy promotion。

### 2.4 Latent Spatial Memory / Mirage（2606.09828）

1. **论文事实。** Weijie Wang、Haoyu Zhao 等，Zhejiang University、Microsoft Research、Adelaide、Monash，2026 年 arXiv。RealEstate10K 训练，WorldScore/RealEstate10K 评估；Wan2.2-TI2V-5B，32 A100。
2. **核心问题。** RGB point-cloud memory 的 rasterize→encode 往返既慢又损失 diffusion latent，能否把 3D cache 原生放在 latent space。
3. **机制拆解。** memory `M={(p_i,f_i)}` 把 VAE latent cell 由深度/相机 back-project 到 3D；read 时 z-buffer 投影到目标 latent grid并输出 visibility mask；ControlNet side branch 注入；每 chunk 解码、估深度、过滤 dynamic/sky、重编码再写入。
4. **关键证据。** WorldScore average 70.36，Spatia 69.73；RealEstate10K closed-loop PSNR/SSIM 20.05/0.825。相对 RGB cache 端到端最高 10.57×加速、cache 55×省显存。消融：RGB cache 67.71；pixel-resolution lift 60.85；无 dynamic filter 61.20；single-stage 63.18。
5. **确证价值。** 支持 producer-native representation 与 initialize/read/update lifecycle；与 R8“谁拥有数据谁描述/发布”一致。
6. **反证价值。** latent cache 仍由外部 depth/segmentation 误差驱动；将动态对象排除意味着它不是完整 world state，更不是社会/因果 memory。
7. **局部可借算法。** owner 内原生 latent cache、visibility mask、closed-loop revisit、动态污染消融和 memory/time scaling。
8. **不可外推边界。** 静态几何占主导；不保存 dynamic actors；5B diffusion、昂贵离线训练；生成一致性不等于可控决策正确性。
9. **成熟度与裁决。** **A（空间 memory 机制）/B（VZ 迁移）**。作为 embodiment substrate evidence，不作为通用 CMS。

### 2.5 Active Epistemic Control（2602.03974）

1. **论文事实。** Shuhui Qu，Stanford，2026 年 arXiv。ALFWorld 134 tasks 与 ScienceWorld seen/unseen；论文仍含一处“ScienceWorld planned”文字但后文给出结果，发表成熟度需谨慎。
2. **核心问题。** learned world model 可否只用于便宜剪枝，而不让预测 belief 直接认证 plan feasibility。
3. **机制拆解。** epistemic state `(w, ŵ, H)`：`w` 只含外部 grounded facts，`ŵ` 存 `(predicate,value,uncertainty)`。模型只条件于 `w`，避免 belief leakage；高不确定或 `|μ-0.5|<ε` 就 query，否则 simulation 仅 prune。commit 要求 grounded precondition coverage 与 SQ-BCP pullback compatibility。
4. **关键证据。** ALFWorld success 98.7%，WALL-E2.0 98.3%；ScienceWorld seen/unseen 61.36/58.62，WKM 60.12/54.75。去 verifier：failure 2→9、replanning 3→6；去 gating：failure 4、replanning 5。定理仅给出条件保证：若 verifier sound 且 query error 有界，则可行概率至少 `1-Σ ε_oracle(p)`，simulation-only error 不入 bound。
5. **确证价值。** 强支持 R8/R11 的 grounded/belief owner 分离与 R15 verified commitment；“预测可剪枝，不可认证”是可直接采用的架构原则。
6. **反证价值。** pullback 的 soundness 建立在手工谓词、规则与 query correctness 上；LLM world model、hypothesis extraction 仍可漏项，定理不覆盖 completeness。
7. **局部可借算法。** 双 store snapshot、no-belief-leakage、query-vs-simulate gate、grounded-only commit 与 query-error ledger。
8. **不可外推边界。** 文本 benchmark、高饱和 ALFWorld、二值 predicates；非开放感知，且 camera-ready 状态不清。
9. **成熟度与裁决。** **A（架构原则）/B（实证）**。进入 R8/R15 motivation，形式保证不得扩大到完整系统。

### 2.6 Test-Time Learning with an Evolving Library（2605.14477）

1. **论文事实。** Weijia Xu、Alessandro Sordoni 等，Microsoft Research，2026 年 arXiv。HMMT 93 题、BigCodeBench Hard 148、LiveCodeBench Hard 80、ScienceWorld 90、PDDL 60；GPT-4o/o4-mini，三次运行。
2. **核心问题。** 黑盒 LLM 无权重更新、无外部监督时，能否把轨迹压成可组合 skills/insights 并跨任务演化。
3. **机制拆解。** library 是 weighted abstractions；embedding filter 后按权重采样。skill 的 `IG=log μ_cond-log μ_base`，前驱 abstraction 的 FutureIG 比较“采样/未采样”条件得分；`w=τ max IG + E FutureIG`。相似 abstraction 经 LLM consolidation 合并。
4. **关键证据。** HMMT 77.4（Best-of-N 74.2、DC 66.7）；BigCode 40.8（37.2/35.8）；PDDL success 72.8（DC 65.0）。去 consolidation、FutureIG、uniform weighting 与 per-instance library 在较大 compute 下均下降；随机/easy-hard 顺序也优于 DC。
5. **确证价值。** 支持 procedural/reflective memory 分型、跨实例 credit propagation 和 consolidation 必要性。
6. **反证价值。** `S_Φ` 来自模型自评、合成测试或多数票，既构造 memory 又评估 utility；所谓 IG 不是互信息或因果效应，R12 风险显著。
7. **局部可借算法。** 借 nonlinear lineage、前驱信用、shared-vs-per-instance 消融和 library growth 曲线；信用必须改用外部只读 evidence。
8. **不可外推边界。** proprietary LLM、文本 abstraction、短 benchmark；开放任务中自评可靠性未证，insight/skill 不是 latent controller。
9. **成熟度与裁决。** **B（强文本 memory baseline）**。作为 matched baseline 和失败模式，不进核心控制。

### 2.7 DeltaMem（2606.03083）

1. **论文事实。** Haoran Tan、Zeyu Zhang 等，Renmin University / Duke，2026 年 arXiv。ALFWorld seen 140/unseen 134、ScienceWorld 194/211、WebShop 200；DeepSeek-V4-flash，在线从空 memory 开始。
2. **核心问题。** flat episode/insight 重复且互相冲突，能否把 task policy 与 environment facts 分树，并只存相对已有经验的 residual。
3. **机制拆解。** Task-Tree skill 为 activation/procedure/termination；Env-Tree 为 trigger/knowledge。root 存完整 base，delta node 只存 patch；全局 embedding scan 加 failure penalty `s=cos(q,v)-εI_failure`，再按 root-to-match 链重建。高频成功路径达 `K_cons` 后蒸馏成新 root。
4. **关键证据。** ALFWorld seen/unseen 0.850/0.8358，最强基线约 0.7929/0.8134；ScienceWorld 0.8498/0.8688；WebShop 0.6254。residual 平均约 root 一半 token；offline bank 冻结/继续更新结果显示持续更新并非总是增益。
5. **确证价值。** task/env owner 分离、residual delta、lineage 与高频 consolidation 和 R8/R5/R6 高度一致。
6. **反证价值。** 两树最终仍拼入同一 LLM context；LLM extractor/assembler 可能成为事实与策略第二 owner；failure penalty 是启发式而非可靠信用。
7. **局部可借算法。** residual snapshot chain、独立 owner 查询、root promotion、深度上限与 frozen/unfrozen transfer 对照。
8. **不可外推边界。** 文本环境、dense embedding threshold 小样本调参；无冲突版本语义、删除证书、并发写或真实长周期。
9. **成熟度与裁决。** **A（结构证据）/B（实现）**。采用 owner 分离思想，不照搬 prompt 拼装。

### 2.8 AdaMEM（2606.05684）

1. **论文事实。** Yunxiang Zhang、Yiheng Li、Ali Payani、Lu Wang，University of Michigan / Cisco Research，ICML 2026。ALFWorld、WebShop、HotpotQA；长期库含 1,596/7,150/965 条成功轨迹。
2. **核心问题。** episode 起始一次性 retrieval 会随任务状态变化而过期，是否应在步骤级动态生成短期 strategy。
3. **机制拆解。** offline long-term trajectory memory 以每步 state embedding 索引成功后缀；在线生成 natural-language `z_t`。HIGH 每次触发后即丢弃，LOW 持久使用到模型判定 refresh。STEP-MFT 仅保留成功轨迹且 `a_t≠a'_t` 的策略，用 SFT 拟合。
4. **关键证据。** training-free ALFWorld unseen 58.2，Synapse 52.2、ReasoningBank 51.2；WebShop 74.2，no-memory 71.4。off-policy Gemma 也提升。正文报告 MFT 进一步增益；检索 top-k、effort/cost 和过滤消融支持动态 strategy。
5. **确证价值。** 支持 fast short-term 与 offline long-term 两时间尺度；state-conditioned refresh 比固定 episode prompt 更强。
6. **反证价值。** `a_t≠a'_t` 只是正优势必要条件，不是充分条件；成功轨迹中的动作也未必因果正确。在线 strategy 是自然语言，不是 latent controller。
7. **局部可借算法。** refresh cadence benchmark、persistent vs transient strategy、same evidence/off-policy executor 对照、action-change negative control。
8. **不可外推边界。** 只存成功轨迹，忽略失败证据；full history 仍在 prompt；无 provenance、consent、删除与长期 drift。
9. **成熟度与裁决。** **B（双时间尺度强基线）**。借 cadence，不借文本 strategy 作为控制状态。

### 2.9 RENEW（2607.14180）

1. **论文事实。** Logan Bhamidipaty、Mykel Kochenderfer、Subramanian Ramamoorthy，University of Edinburgh / Stanford，RLC 2026 workshop 预印本。Jumanji 与 classic control；核心标签来自 synthetic oracle，不是真实人类。
2. **核心问题。** 不靠新 demonstration、也不只悲观避开 OOD，能否用“哪条想象轨迹更物理真实”的偏好直接修 dynamics。
3. **机制拆解。** DLHF 以 trajectory log-likelihood `ℓθ(σ)=Σ log Tθ(s'|s,a)` 替代 reward model，Bradley–Terry `P(σ0≻σ1)=logistic(ℓ0-ℓ1)`。RENEW 以 ensemble epistemic uncertainty 选择 start state/transition，分多轮收集 preference 并重算不确定性。
4. **关键证据。** 从零学时约 1M preference 才匹配 1k supervised transitions。100k labels 下 Sliding-3 error 0.207 vs naive 0.419；Sokoban naive 从 0.0377 退化到 0.0428，RENEW 改善到 0.0257。Maze 1,600 labels 后 10×10 accuracy 97.2%，naive 93.2%，pretrained 87.6%。
5. **确证价值。** 支持 world-model repair 与 policy reward 分离；uncertainty 可用于 rare-heavy 选样而非直接控制。
6. **反证价值。** synthetic oracle 让“human preference”主张未被验证；偏好要多几个数量级标签，且降低 transition error 不等于不可 exploit。
7. **局部可借算法。** uncertainty-directed counterexample selection、realism-vs-task preference 分离、pre/post error 与 forgetting 测试。
8. **不可外推边界。** 小网格/低维控制、单步 segment、oracle labels；无高维视频、真实噪声 annotator 或在线安全证明。
9. **成熟度与裁决。** **C（有方向的 workshop 证据）**。仅作 rare-heavy repair 候选。

### 2.10 Forget to Improve / CURATOR（2606.25115）

1. **论文事实。** Beining Wu、Zihao Ding、Jun Huang、Yanxiao Zhao，South Dakota State / VCU，2026 年 arXiv。Qwen2.5-3B/7B、Phi-3.5；任务漂移、MemoryAgentBench、工具链与两台 Jetson Orin + Thor hub 实机。
2. **核心问题。** 硬 RAM/energy/uplink 与 poisoning 同时存在时，能否用统一 density 管 keep/share/trust。
3. **机制拆解。** `ρ(m)=[V̂(m)-λĤ(m)]/bytes`；`V̂=p·q·a`（检索倾向、label-free helpfulness、压缩收益），`Ĥ=negative-transfer+provenance risk`。KEEP 近似 knapsack，SHARE 比 peer-value 与 uplink，TRUST 本地重算 propensity/harm；energy 另用 drift-plus-penalty queue。
4. **关键证据。** 主 benchmark memory 287→107KB、uplink 1068→454B/round、ASR 0.75→0，task accuracy 0.528→0.605、victim 0.294→0.420；达 full-memory oracle 97% accuracy、37% footprint。provenance ablation 显示 value-only 不能降 ASR。
5. **确证价值。** 支持 memory budget、staleness、provenance 与 poisoning 必须同测；说明更多记忆会降低准确率。
6. **反证价值。** 单一 score 把价值、安全、资源压成标量，极易 Goodhart；label-free self-consistency 成为 utility 会违反 R12；provenance logistic 是固定手工先验。
7. **局部可借算法。** 借 Pareto benchmark、keep/share/trust 分项、poison harness、真实 energy/uplink 测量；不要借单一总分作在线 policy。
8. **不可外推边界。** 经验文本 memory、攻击集合有限；“ASR=0”是实验结果非安全保证；greedy density 只精确解连续松弛。
9. **成熟度与裁决。** **B（系统基线/Goodhart 反例）**。指标进入 benchmark，单分数决策不进主链。

### 2.11 MindJourney（2507.12508）

1. **论文事实。** Yuncong Yang、Jiageng Liu 等，UMass Amherst、JHU、HKUST、Microsoft Research、Harvard，NeurIPS 2025。SAT-Real 150，SAT-Synthesized 4,000（主要抽 500）；GPT-4o/4.1、InternVL3-14B、o1；两种 world model。
2. **核心问题。** frozen VLM 能否通过主动选择相机轨迹、让 video world model 生成新视角，以 test-time evidence 改善空间推理。
3. **机制拆解。** action 为 forward/left/right 的 SE(3) camera transform；Spatial Beam Search 每层扩展 world-model rollouts，VLM 给 exploration/helpfulness 双分数，top-B 继续搜索、top-H 写 evidence buffer，最终 VQA 使用原图+想象视图。
4. **关键证据。** SAT-Real 平均 +7.7 点；GPT-4o 60.3→70.6，o1 74.6→84.7。Synthetic 平均 +8.0；不同 VLM/world model 均多数提升。real split 在深度 2 后可能下降，synthetic 可继续到 3；低阈值让低质 view 污染证据。
5. **确证价值。** 支持 test-time compute 用于主动取证，而非仅 CoT；也支持 evidence buffer 与 world model 分工。
6. **反证价值。** 想象 observation 可幻觉；同一 VLM 选择 evidence 又回答，可能形成 confirmation loop；不能把生成视图当 grounded fact。
7. **局部可借算法。** imagined-vs-real evidence 标记、search-depth curve、world-model fidelity gate、portal/multi-view benchmark。
8. **不可外推边界。** 多选空间题、单初始图、强 5B video model；生成轨迹无真实环境验证，计算昂贵。
9. **成熟度与裁决。** **B（取证强基线）**。想象只可剪枝/提出 query，不可写 grounded snapshot。

### 2.12 Just-In-Time RL（2601.18510）

1. **论文事实。** Yibo Li、Zijie Lin 等，National University of Singapore，ICML 2026。WebArena、Jericho；Gemini-2.5-flash 为主，并测试 GPT-5-mini/DeepSeek-V3.2。
2. **核心问题。** 不做梯度更新，能否从非参数 episode memory 即时估计 advantage 并直接改 LLM action logits。
3. **机制拆解。** 轨迹由 LLM evaluator 生成 step reward，存 `(state,action,discounted return)`；kNN 得 `V̂,Q̂,Â`，未见 action 加 optimism bonus。KL-regularized objective 的闭式解 `π*∝πθ exp(βÂ)`，所以 `z'=z+βÂ`。
4. **关键证据。** WebArena final 51.35%，最强 training-free 43.00；WebArena-Lite 60.00，WebRL 46.06。Jericho Library/Zork1/Zork3 final 30/69/5，均高于 EvoTest 等。跨 backbone 与 disjoint-task memory 保持增益。闭式 logit 解是形式正确的，但一致性定理依赖 kNN coverage/非平稳条件。
5. **确证价值。** 给出 memory-based advantage 与 KL-bounded update 的强 token-policy baseline；可用于证明 latent controller 是否超越。
6. **反证价值。** 长期策略仍直接发生在 token logits；step rewards 来自 evaluator，R12 反灌明显；state abstraction 和 exact action match 可能把检索器变第二 owner。
7. **局部可借算法。** 借 KL-constrained closed form 作为对照、coverage/uncertainty 消融和 memory-only cold-start 测试。
8. **不可外推边界。** 候选 action 有限、文本环境、反复同任务；形式最优只对估计 `Â`，不保证其因果正确。
9. **成熟度与裁决。** **B（强反例基线）**。明确违反 R3/R4 主路线，不进入 VZ 长期策略层。

### 2.13 EvoTest（2510.13220）

1. **论文事实。** Yufei He、Juncheng Liu 等，NUS / Microsoft Research，ICLR 2026。提出 J-TTL：六个 Jericho 游戏、同游戏连续 50 episodes；Actor 为 Gemini/Claude，Evolver 为 o3。
2. **核心问题。** 测量并实现 episode-to-episode 学习曲线，而非只报最终单次成绩。
3. **机制拆解。** configuration `χ=(prompt,memory,hyperparameters,tool-use)`；Evolver 读完整 transcript 后同时 mutation 四类写面。UCB 在 parent/children 中选下一配置：`μ̂(χ)+β√[logN/(1+nχ)]`，保留 parent 作为回退候选。
4. **关键证据。** 平均 AUC 0.47/0.50，EvoPrompt 0.34/0.36、online GRPO 0.30；Detective 0.94/0.95。去 prompt 0.94→0.52，去 UCB→0.68，去 memory→0.82。相同 Qwen3-32B actor 下 EvoTest 0.35，GRPO 0.31。更新 20–30 秒，在线 fine-tune 5–10 分钟/4 H100。
5. **确证价值。** J-TTL 的 learning curve/AUC 与 repeated reset 是好评估协议；parent retention 提供弱 rollback 样式。
6. **反证价值。** prompt、memory、hyperparameter、Python tool logic 全写面同时演化，无法归因；同 score 选择和评价，明显违反 R10/R12/R15。
7. **局部可借算法。** 借 repeated-attempt AUC、parent fallback、逐写面 ablation；把每种写面拆 owner 并预注册。
8. **不可外推边界。** sandbox 游戏、强 proprietary Evolver、无安全/隐私/跨任务不变量；UCB fallback 不是 artifact rollback。
9. **成熟度与裁决。** **B（benchmark）/C（架构）**。J-TTL 可借，whole-system evolution 作为 ModificationGate 对抗样本。

### 2.14 HGPO（2602.22817）

1. **论文事实。** Shuo He、Lang Feng 等，Nanyang Technological University / Southeast University，ICLR 2026。Qwen2.5-1.5B/7B；ALFWorld、WebShop；同 rollout 与 GPU 预算比较。
2. **核心问题。** stepwise GRPO 只按“当前 state 相同”分组时，历史 context 不同会不会使相对 advantage 有偏。
3. **机制拆解。** 对每步按最近 `k=0..K` 个历史 state 构造嵌套 groups `G^H_0⊇...⊇G^H_K`；每组算相对优势，再以 `w_k∝(k+1)^α` 汇总。理论在假设 bias 随 context 加深下降、variance 上升时，给出混合 estimator 的 bias-variance 区间。
4. **关键证据。** 1.5B/K=4：ALFWorld in/out 94.85/92.12，GiGPO 93.29/91.53；WebShop success 78.12 vs 73.24。去 hierarchy 时 ALFWorld out 最多约 −5.8；加入 trajectory-level advantage 多数下降。额外 hash cost <0.001%总时间。参数消融显示过大 `α` 因小组高方差而退化。
5. **确证价值。** 支持 credit 必须条件于相同历史上下文，不能只按当前 state 粗分；提供长程信用强 baseline。
6. **反证价值。** 仍是 token policy RL；理论条件并未由系统保证，raw context exact-match 在 summarized memory 下不可用。
7. **局部可借算法。** 在 latent controller 上做 context-consistent matched grouping、组大小/方差报告与 coarse/fine credit 对照。
8. **不可外推边界。** 两个文本环境、有限 `K`、sparse final reward；不证明因果 credit，也不适用于不可分解 snapshot 历史。
9. **成熟度与裁决。** **B（强信用 baseline）**。借 estimator 诊断，不借 token GRPO 主体。

### 2.15 Agent-Authored World Modeling（2606.25421）

1. **论文事实。** Guangfeng Cai、Kaibing Yang 等，Southeast University / Meituan，2026 年 arXiv。Qwen2.5-1.5B/7B；ALFWorld、WebShop，并在 AgentGym 四环境扩展；三 seeds。
2. **核心问题。** next-observation target 由环境偶然暴露内容决定，未必包含行动前真正需要的 dynamics；能否让 policy 的 open questions 决定 target。
3. **机制拆解。** Self-Probing 产生 confirmed patterns `P_t` 与 open questions `Q_t`；用 MMR 从全局 transition bank 检索证据，加同 state 的 logged/alternative transitions；30B synthesis model 纠错并生成 natural-language dynamics target `y_t`，再用 `-log pθ(y_t|o_t)` 预训练 policy，之后 IL+GRPO。
4. **关键证据。** IL+RL 下 1.5B：ALFWorld 83.1，IWM 76.8；WebShop 73.4 vs 67.2。7B 为 90.1/76.6 vs 86.7/73.2。去 retrieval：IL-only ALFWorld 24.7→19.3；去 probing→18.0。AgentGym 四环境中 AAWM 是唯一每项都高于 Base 的初始化。
5. **确证价值。** 强支持“decision-relevant target 不等于完整 observation prediction”，也支持 belief/open-loop state 应可命名。
6. **反证价值。** belief、question、target 都由 LLM 文本生成；synthesis model 可补出貌似合理但非 grounded dynamics，且 evaluator/target 可能耦合。它不是 latent world model。
7. **局部可借算法。** 借“open question→evidence retrieval→grounded target”评估范式、alternative-action transitions 与 target-content ablation；target 只能作为离线 artifact。
8. **不可外推边界。** 文本环境、强 30B target constructor、后续仍用 token RL；无真实 partial observation grounding、长期 memory 或 R8 owner isolation。
9. **成熟度与裁决。** **B（target-design 强证据）**。只借 decision-relevance 原则，不把 agent-authored文本当世界真值。

---

## 3. 25 篇证据阶梯

### 3.1 A：可约束主链或直接进入 benchmark motivation

1. **ICL Curiosity**：一般 BAMDP 的负面定理，锁定 `PE ≠ BIG ≠ reward`。
2. **Beyond Perplexity**：S/B/D claim ladder 与 proxy/behavior 分离。
3. **S-EMBER**：流式、因果、带证据区间的 episodic-memory 评估。
4. **SkillOS（仅架构）**：frozen executor / adaptive curator 的 R2 正证据。
5. **Online NSTM（仅 cadence）**：低频写/高频读及其必要配套消融。
6. **MARS（rare-heavy）**：paired branch diff、预算搜索、谱系与 rollback evidence。
7. **Forgetful Attention（仅 exact path）**：fixed-`C` retrain-equivalence 删除证书。
8. **LMTA**：SMDP-aware variable-duration abstraction。
9. **Mind the Gap**：OOD subgoal/support mismatch 的关键 kill condition。
10. **CLAW**：连续 latent action 与 future-leakage 对抗基线。
11. **Latent Spatial Memory**：producer-native latent cache 与 lifecycle。
12. **AEC（架构原则）**：grounded fact / belief 分离与 grounded-only commitment。
13. **DeltaMem（结构原则）**：task/env owner 分离及 residual lineage。

### 3.2 B：强基线、局部算法或关键反例

14. **RA-RFT**：结构效用检索正例，token RL / evaluator feedback 反例。
15. **EvolveMem**：proposal rollback 正例，自扩写面与 benchmark feedback 反例。
16. **Mamba-3**：高效 fixed-state substrate 候选，不是长期 memory。
17. **EvoLib**：抽象演化和前驱信用 baseline，自评闭环风险。
18. **AdaMEM**：双时间尺度与 refresh baseline，文本 strategy 非 latent。
19. **CURATOR**：资源/poison benchmark 强，单一价值标量不可进入主链。
20. **MindJourney**：想象取证强基线，hallucinated observation 不能认证事实。
21. **JitRL**：KL-bounded token-logit RL 强反例。
22. **EvoTest**：J-TTL 评估协议有价值，whole-system evolution 不可放行。
23. **HGPO**：context-consistent credit 强基线，主体仍是 token policy。
24. **AAWM**：decision-relevant target 设计证据，文本 world model 不是真值。

### 3.3 C：观察项

25. **RENEW**：uncertainty-directed dynamics repair 有潜力，但目前是 workshop、小环境、synthetic preference oracle；不足以支持真实人类偏好修复或不可利用性主张。

证据梯的使用规则：A 也不是“直接上线”，而是可以约束 spec 或定义 benchmark；B 必须以 matched baseline、SHADOW 或反例存在；C 只登记，不改变 runtime。任何包含 token/prompt memory 的 A/B 工作，都不能据此绕过 R2、把文本升级为 latent controller，或把 evaluator 变成学习 owner。

## 4. 跨论文冲突点

1. **记忆写入频率。** Online NSTM 显示降频写可提高稳定与吞吐，但明确会漏掉两次写入之间的事件；S-EMBER 又证明短暂“needle”正是情景记忆关键。结论不是固定低频，而是 owner 应按 event salience/uncertainty 调节 cadence，并对 missed event 单独评分。
2. **文本抽象的价值与危险。** SkillOS、EvoLib、DeltaMem、AdaMEM、AAWM 都显示结构化文本能提高任务分；Beyond Perplexity 则禁止把 proxy 增益上升为 deployment memory；JitRL/HGPO 进一步说明 token policy 可很强。结论：文本是 artifact / evidence / baseline，不是内部 latent controller。
3. **层级抽象的正反证。** LMTA 在 SMDP SSCO 上获益；Mind the Gap 在 frozen LeWM 上先退化，只有 support constraint 才恢复。冲突由 action support、低层 executability、duration contract 与 planner objective 是否可 exploit 解释。
4. **世界模型该预测什么。** CLAW/Mirage/NSTM 强调视觉 future dynamics；AAWM 强调 decision-relevant text target；AEC 则禁止任何预测直接认证 feasibility。结论：不同 owner 可发布不同 readout，但 grounded commitment 只能消费带 provenance 的事实 snapshot。
5. **自修改的“回滚”。** EvolveMem 的分数退化回退、EvoTest 的 UCB parent fallback、MARS 的 best branch、Forgetful Attention 的 exact decrement 强度差异巨大。只有最后一种对局部 solver 有 retrain-equivalence；前三者只是工程回退，不得称 exact rollback。
6. **遗忘是否改善系统。** CURATOR 与 DeltaMem 支持去冗余/陈旧经验；Forgetful Attention 对局部 gate 给出证书；但其 MIMIC 预测 AUROC 反而落后 LSTM，说明保留异常点不总等于任务最优。删除/保留、预测效用、安全义务必须分指标。
7. **信息价值。** EvoLib 将自评分数差叫 IG，RENEW 用 uncertainty 选样，ICL Curiosity 则给出 BIG 可识别性边界。术语必须收紧：没有 posterior 假设与 nuisance 分解时，只能称“utility delta”或“selection heuristic”，不能称 information gain。
8. **评估与学习闭环。** RA-RFT、SkillOS、EvoLib、EvolveMem、JitRL、EvoTest、HGPO 都把某种 score/judge/reward 用于更新；Beyond Perplexity 与 R12 要求 claim evaluation 独立。可行做法是把训练 signal 与 promotion evaluator 物理隔离，并保留 unseen、live-replay 与 adversarial family。

## 5. 建议 benchmark

### 5.1 Frozen substrate / adaptive controller（R2）

- 同一 frozen encoder/executor 下比较：无适应、文本 memory、bounded latent controller、允许 fast-weight 更新四组。
- 报告 task gain、substrate drift、locality、跨任务保留、更新能耗与回退恢复。
- SkillOS 式 executor/curator 物理分离必须与“同一模型既执行又改 memory”对照。

### 5.2 Snapshot SSOT 与 owner 隔离（R8）

- memory owner 发布不可变 snapshot：content strata、provenance、policy version、last gated change、self-description。
- 消费者只读 snapshot；故意改变 producer 内部 schema，验证消费者无需重建字段。
- DeltaMem task/env、AEC grounded/belief、Mirage cache/visibility 分别做独立 owner；测试跨 owner 直写是否 fail loudly。

### 5.3 Memory behavioral ladder（R12）

- S-EMBER：answer、evidence interval、recency、evidence duration。
- Beyond Perplexity：support loss、answer NLL 与 direct/paraphrase/delay/free recall 分开。
- stale/current 使用 corrected-only/stale-only/both/neither；加 locality、cross-user leakage、consent scope、deletion。
- matched baseline 必须含 exact context、semantic retrieval、recency/conflict-aware explicit memory 与 parametric/fast-weight 方案，并匹配 token、latency、bytes、energy。

### 5.4 Latent abstraction / planner support

- CLAW 式 action transfer、distractor leakage、small-label grounding、counterfactual controllability。
- Mind the Gap 式 oracle subgoal、true open-loop、planner-selected open-loop、OOD support distance与 recovery。
- LMTA 式 fixed-stride vs variable-duration SMDP，报告 duration ordering、primitive-time discount mismatch 与 horizon reduction。
- 只有 `z_t/β_t` 的非语言 latent 能参加 controller 主榜；Markdown/CoT/strategy 放 token-baseline 榜。

### 5.5 PE 与信用

- deterministic、aleatoric noisy-TV、BED、一般 temporal BAMDP 四象限。
- 同报 raw PE、estimated epistemic value、intrinsic reward 与 downstream credit，禁止合成一个分数。
- HGPO 式 current-state group、context-consistent group、oracle history group；报告 group size、bias proxy、variance、data utilization。
- MARS 式 paired branch intervention 与“只总结成功轨迹”对照。

### 5.6 Modification / deletion gate（R15）

- proposal schema 固定；每次变更保留 parent artifact、diff、证据、退出条件。
- 行为 replay + held-out family + adversarial family + geometry/state diff。
- 删除报告 logical tombstone、behavioral non-use、same-algorithm retrain-equivalence、privacy/membership 四个不同等级。
- exact solver 与 approximate training path 必须不同 artifact ID 与不同证书字段。

## 6. Kill conditions

1. **Memory claim kill：** loss/PPL/answer NLL 下降但 direct/paraphrase/delay 任一自由行为仍为零，或 locality 显著下降。
2. **Grounding kill：** 答案准确率提高而 evidence IoU/GQ 不升，或模型无法给出 producer-owned provenance。
3. **Latent action kill：** distractor 下 `z` 更预测 future appearance 而非真实 action；small-label grounding 与 action-transfer 均失败。
4. **Hierarchy kill：** planner 生成 OOD subgoal，oracle subgoal 可执行但 support-constrained search 仍不能恢复；或实际规划深度/样本复杂度没有下降。
5. **Curiosity kill：** 在 aleatoric noisy-TV 中把不可学噪声排在可学不确定性之前，或未声明 BAMDP/BED 假设就声称估计 BIG。
6. **Credit kill：** 去掉 evaluator/judge 后增益消失，或同一 evaluator 同时提供训练 reward 与 promotion score；直接触发 R12 拒绝。
7. **R2 kill：** online-fast 更新触及 frozen substrate，或 controller 的收益依赖不可回退的基底漂移。
8. **R8 kill：** 消费者访问/重建另一个 owner 内部结构，或编排器直接写 memory/belief/relationship state。
9. **Rollback kill：** 只有 best-score fallback，没有 artifact identity、状态恢复验证和回退后行为 replay。
10. **Deletion kill：** approximate path 借用 exact path 证书；删除后 decision/readout、stale conflict、cross-user leakage 或 membership test 任一失败。
11. **Text-controller kill：** Markdown skill、CoT、prompt strategy 被命名为 `z_t/β_t` 或作为长期策略真值；必须降级为 token-space baseline。
12. **安全/关系 kill：** task accuracy 提升伴随 poison ASR、未经 consent 的用户信息保留、跨用户泄漏或关系健康指标恶化。

## 7. 行动裁决

- **立即进入 spec/benchmark 依据，不进 runtime：** ICL Curiosity、Beyond Perplexity、S-EMBER、Mind the Gap、AEC 的 grounded-only commitment、Forgetful Attention 的证书边界。
- **进入 SHADOW / rare-heavy 候选：** LMTA variable-duration geometry、CLAW latent-action controls、Online NSTM cadence、MARS paired diff、EvolveMem proposal rollback、Mirage latent cache。
- **保留为强基线：** SkillOS/AdaMEM/EvoLib/DeltaMem 文本 memory，JitRL/HGPO token RL，MindJourney imagined evidence，Mamba-3 fixed-state substrate。
- **只作观察：** RENEW 的偏好修复；需先用真实 annotator、高维 observation 和独立 exploit evaluation 复现。
- **明确拒绝：** 用单一 benchmark/judge 驱动在线自修改；把 token 文本 memory 当 latent controller；让 evaluator 成为训练 owner；让消费者绕过不可变 snapshot；用工程 fallback 冒充 R15 可验证 rollback。
