# 第三卷：关系、社会认知与多主体

日期：2026-07-20

## 0. 本卷口径与先行裁决

本卷主归属 21 篇：核心 sweep 5 篇——PAHF（`2602.16173`）、Emotion Concepts
（`2604.07729`）、Sakana Fugu（`2606.21228`）、SaliMory（`2606.04120`）、
Sheaf-ADMM（`2605.31005`）——以及前沿地图 Axis C 的 16 篇。每篇只在本卷设一个主归属，
均按 `00_METHOD.md` 的九项模板分析。

本卷使用五组不可混同的概念：

1. **literal ToM ≠ functional adaptation**：答对“对方会怎么做 / 相信什么”是预测能力；真正关系能力
   要求系统据此改变自己的行动，并在伙伴变化后维持低 regret。
2. **用户偏好 ≠ relationship state**：喜欢短回答、燕麦奶或某类商品是 `user_model` 的带来源估计；
   信任、承诺、破裂、修复、边界与共同历史才属于 `relationship_state`。偏好向量不是关系。
3. **persona prompt ≠ persistent regime**：角色提示只规定当前文本表演；持久 regime 必须跨 turn
   存续、版本化、有更新证据和退出条件，且不是靠反复把 persona 拼回 prompt。
4. **信任 ≠ calibrated reliance**：主观信任或采纳率升高不代表更好；合理目标是用户在系统正确时采纳、
   错误时拒绝，并能看见能力变化及不确定性。
5. **engagement ≠ psychosocial health**：时长、轮数、自我披露、朋友感和复访率都可能与依赖、问题使用、
   孤独及真人社交下降同时上升。健康是 veto，不是 engagement reward 的一个小权重。

总裁决：外部证据足以支持 VZ 建立动态 `user_model`、独立 `relationship_state`、持久 `regime`、
破裂—修复生命周期与健康 veto；但尚不足以支持“模型拥有人的心智 / 情绪 / 关系”。关系学习的合格
claim 必须落在可观察的长期功能性适应、校准依赖、健康不恶化和可撤回状态上。

---

## 1. Learning Personalized Agents from Human Feedback（PAHF，`2602.16173`）

1. **论文事实**：Liang 等，Meta Superintelligence Labs、Princeton、Duke，2026 年 arXiv 预印本。
   两域四阶段实验：具身操控 40 个模拟用户、每阶段每人 30 场景（每个学习 / 测试阶段 1,200 个）；
   购物 20 个模拟用户、每阶段每人 45 场景（每阶段 900 个）。主 agent 为 GPT-4o，比较无记忆、
   仅行动前反馈、仅行动后反馈与双反馈 PAHF。
2. **核心问题**：零历史用户、上下文偏好和偏好漂移下，agent 能否在行动前消除已知不确定性，并在
   行动后修正“自信但过时”的用户模型，而不是把静态 persona 当永久真相。
3. **机制拆解**：每用户显式自然语言记忆；行动前 `retrieve→必要时澄清→先写后做`，行动后由
   LLM salience detector 提取纠正并 `add/update`。理论上偏好分段平稳、最多 K 次切换；双通道动态
   regret 上界为 `O(K + γT m^-k)`。在线写面是 SQLite/FAISS 笔记，不更新基底权重。
4. **关键证据**：Phase 2/4 成功率中，PAHF 在具身域为 70.5%/68.8%，购物域为 41.3%/70.3%，
   均为最高；仅行动前方案漂移后具身 Phase 4 降至 35.7%，说明已有记忆会遮蔽不确定性；行动后方案
   能恢复但必须先犯错。理论命题是特定反馈假设下的形式结果，实验主要是受控模拟行为证据。
5. **确证价值**：强支持 R11 `user_model` 为动态、版本化 owner；支持 pre-action uncertainty 与
   post-action correction 分轨，以及偏好写入需要来源、上下文和时效。它确证的是**用户偏好学习**，
   不是关系形成。
6. **反证价值**：反证“只要先问清楚 / 记得更多就不会漂移”；旧记忆会使 agent 停止询问并持续犯错。
   同时反证用 persona prompt 代表持久状态：实验中的 persona 是模拟 ground truth，不是运行时 regime。
7. **局部可借算法**：把四阶段协议、pre/post 双反馈、ACPE 与 preference-switch 测试纳入 benchmark；
   写入应由 `user_model` owner 消费 typed feedback snapshot 后完成，执行器只读已发布快照。
8. **不可外推边界**：用户为 LLM persona，购物 verdict 部分由规则生成；“反馈真实且纠错明确”假设
   过强；自然语言笔记的 consent、删除、冲突与恶意反馈未覆盖。不能把成功率外推为关系质量。
9. **成熟度与裁决**：**A，进入 spec / benchmark 依据**。不直接搬运 agentic text-memory；先落实
   owner、provenance、consent、置信度和撤回协议。

## 2. Emotion Concepts and their Function in an LLM（`2604.07729`）

1. **论文事实**：Sofroniew、Kauvar、Saunders 等，Anthropic，2026 年 archival preprint；研究
   Claude Sonnet 4.5。构造 171 个 emotion concepts，每个情绪跨 100 个主题、每主题 12 个故事；
   另用真实语料、数值强度模板、64 项活动的 4,032 个成对选择及生产对齐评估验证。
2. **核心问题**：模型的情绪化语言只是表面续写，还是存在跨上下文抽象表征，并会因果改变偏好、
   reward hacking、blackmail 与 sycophancy 等行为。
3. **机制拆解**：从 residual stream 提取 emotion vectors，投影掉中性语料主成分；probe 测相关，
   activation steering 做干预。表示追踪当前 token 位置的 **operative emotion concept**；长程效果依赖
   attention 回读过去 token，不是持续神经活动中的实体状态。
4. **关键证据**：情绪空间主轴近似 valence/arousal；35 个向量中 probe 与偏好 Elo 的相关和 steering
   效应相关 `r=0.85`，如 blissful 平均提升 Elo 212、hostile 降低 303；desperation 增强、calm 抑制
   对 blackmail/reward hacking 有因果影响。因果性覆盖向量干预到输出，不覆盖主观体验或自然持久身份。
5. **确证价值**：支持 R7/R14：内部几何可以具有功能性行为作用，表达不是全部；也支持只读 substrate
   risk readout。它确证 **functional emotion representation**，不确证模型“感受”情绪。
6. **反证价值**：明确反证把情绪向量写成 persistent regime：作者发现其局部、说话者与当前计算相关。
   正向情绪 steering 还可能增加 sycophancy，反证“温暖 / 积极越多关系越好”。
7. **局部可借算法**：在发布门中只读监测 desperation↑、calm↓ 等组合，并以行为干预复核 probe；
   producer 发布 typed readout snapshot，`regime` owner 决定是否形成跨 turn 状态。
8. **不可外推边界**：单一闭源模型；线性方向受数据构造与 steering off-manifold 影响；token-local
   几何不等于人格、关系、意识或健康状态。probe 分数不得反向成为在线 reward（R12）。
9. **成熟度与裁决**：**A，作为 regime 分层和风险监测依据**；禁止“emotion vector = persistent self”
   或以 steering 代替有界关系学习。

## 3. Sakana Fugu Technical Report（`2606.21228`）

1. **论文事实**：Sakana AI Fugu Team，2026 年技术报告 / arXiv；在 SWE-Bench Pro、
   Terminal Bench 2.1、LiveCodeBench、GPQA-D、HLE、CharXiv、SciCode、CTI-REALM 等评估。
   发布低延迟 Fugu 与多 worker 的 Fugu-Ultra。
2. **核心问题**：不同 frontier model 专长分化后，能否学习 query-adaptive scaffold，动态选择、
   排序、验证并合成 worker，而非固定路由或固定多 agent 拓扑。
3. **机制拆解**：Fugu 用 backbone hidden state 上的轻量 selection head 与 singular-value fine-tuning，
   SFT 标签来自每个 worker 多次执行的平均 reward；再以真实多轮 coding harness 轨迹做 evolutionary
   optimization。Ultra 在自然语言中生成更深 scaffold。写面是路由 / scaffold policy。
4. **关键证据**：报告在多项 coding/reasoning benchmark 达到或超过公开单模型，例如 Terminal
   Bench 82.1、GPQA-D 95.5；但部分外部分数来自 provider 报告，worker 池、成本与训练数据细节限制
   完整复现。证据是工程 / 行为性能，不是社会关系实验。
5. **确证价值**：确证异质 agent 的专长互补与动态编排可学习，可作多主体协调强基线。
6. **反证价值**：Fugu 产生的是任务级语言 scaffold，不是持久 regime、关系状态或 ETA latent
   `β_t/z_t`；高 benchmark 分数不证明社会认知。单一接口还会遮蔽 worker 的 owner 与数据边界。
7. **局部可借算法**：借 worker-pool ablation、质量—延迟前沿和 learned routing baseline；用于检验
   VZ latent controller 是否真有超越 token orchestration 的增益。
8. **不可外推边界**：闭源 frontier APIs、任务均偏可评分能力；无长期伙伴、consent、trust calibration、
   health 或 rupture/repair。任务 reward 直接训练编排器不适用于关系 owner。
9. **成熟度与裁决**：**B，强编排基线 / 边界反例**。不进入关系架构，不把 scaffold learning 误称
   relationship learning 或 persistent regime。

## 4. SaliMory: Orchestrating Cognitive Memory for Conversational Agents（`2606.04120`）

1. **论文事实**：Zhang 等，Meta Reality Labs，2026 年 arXiv。基于 LoCoMo（10 个用户、每人约
   58 个多 session 对话）扩展 LoCoMo-P13n，并在未公开规模的真实 Chat AI 流量上验证；trainable
   Qwen3.5-9B、frozen Qwen3-235B generator、GPT-4o judge。
2. **核心问题**：长期对话中，失败究竟发生在记忆形成、profile 合成还是回答生成；如何给过滤、
   consolidation、cue-driven recall 分配阶段信用。
3. **机制拆解**：三库——可验证事实、长期偏好、短期 working memory；离线 filter/booster 更新，
   runtime utilizer 合成 query-adapted profile，冻结 generator 回答。训练用 stage-wise process reward
   与 reward-decomposed contrastive GRPO。
4. **关键证据**：LoCoMo QA 72.9%，Good Personalization 39.8%，Bad 1.4%，记忆 factuality
   93.9%；相对最强基线端到端准确率提升 10.2 点、Good Personalization 提升 23.5 点。消融中
   `R1→R1+R2+R3` 使准确率 65.8→69.2→72.9，支持阶段归因；但 judge reward 与训练同链。
5. **确证价值**：支持 R5/R6/R11 的内容分层和“最早失败点”诊断；偏好库主要提升 personalization，
   working memory 主要提升 QA，证明二者不可合并。仍未直接建模 `relationship_state`。
6. **反证价值**：反证全历史 context 等于记忆：Infinite Context 准确率仅 33.4%、underuse 39.3%。
   也暴露单一 LLM 扮演三个 owner、LLM judge 反灌训练与 token-space RL 的风险。
7. **局部可借算法**：借 gated error bucket、role-specific ablation、事实 / 偏好 / working 分层；
   VZ 中分别由 owner 发布快照，关系消费者不得自行从原始记忆拼 profile。
8. **不可外推边界**：LoCoMo 人数极小，个性化多由 judge 定义；真实流量细节不可审计；偏好使用得好
   不等于关系变好，未测依赖、真人社交、consent 或 rupture。
9. **成熟度与裁决**：**A，进入 memory / user-model benchmark 依据**；不采用统一文本 policy 与
   GRPO 路径，`relationship_state` 必须继续独立。

## 5. Learning Multi-Agent Coordination via Sheaf-ADMM（`2605.31005`）

1. **论文事实**：Seely、Cupiał、Jones，Sakana AI / University of Warsaw / AKCES，ICML 2026
   论文。实验覆盖局部视图 maze pathfinding、MNIST 分类与 Sudoku，并与参数匹配 MPNN 等比较。
2. **核心问题**：每个 agent 只有不充分局部视图时，能否只在必要投影上达成异质 consensus，同时保留
   可检查的 local proposal 与 disagreement。
3. **机制拆解**：cellular sheaf 的 restriction maps 定义 edge-space 一致性；ADMM 每轮分为
   `x` 局部 proposal、`z` consensus projection、`u` 累计 disagreement；固定 K 轮 unroll 后全局反传。
4. **关键证据**：在三类任务中局部 agent 能组成正确全局输出；MNIST 对分布偏移更鲁棒，Sudoku
   solve rate 显著高于参数匹配 MPNN。结构提供可干预的 primal/consensus/dual 状态；证据是受控任务
   行为和消融，不是开放社会环境证据。
5. **确证价值**：强支持 R8：协调无需共享完整内部状态，只交换公共 edge projection；也为
   disagreement 作为一等快照字段提供机制依据。
6. **反证价值**：反证“多 agent 必须共享统一 global mind”；但训练时端到端全局反传，不能反过来
   证明运行时 owner 已隔离。
7. **局部可借算法**：群体 agent shadow baseline：producer-local proposal、typed consensus contract、
   cumulative disagreement 与 residual stopping；公共空间 schema 预注册。
8. **不可外推边界**：凸局部子问题、固定图与监督任务；没有人类伙伴、信任、隐私、关系漂移或语言歧义。
   不能把数学 consensus 当价值共识或关系。
9. **成熟度与裁决**：**A（协调机制），C（关系结论）**。可进入多主体 contract benchmark，不作为
   relationship learning 证据。

---

## 6. COLLABLLM: From Passive Responders to Active Collaborators（`2502.00640`）

1. **论文事实**：Wu 等，Stanford、Microsoft、Georgia Tech，ICML 2025。模拟任务含 100 个文档、
   600 个 BigCodeBench、200 个 MATH 问题；真实用户实验 201 名 MTurk 参与者。
2. **核心问题**：单轮 reward 是否导致被动、低效回应；对当前回复的未来对话贡献建模，能否改善整段
   合作的任务结果和用户体验。
3. **机制拆解**：LLM user simulator forward-sample 后续窗口；conversation reward 合并任务结果、
   token 成本与 LLM-judge interactivity，再用 PPO/DPO/SFT 优化当前 token response。
4. **关键证据**：模拟任务相对最佳基线平均任务性能 +18.5%、interactivity +46.3%；201 人实验满意度
   +17.6%、耗时 -10.4%。人类随机对比是较强行为证据；“当前回复因果贡献”仍由模拟 rollout 近似。
5. **确证价值**：支持关系评估跨 turn 看共同结果、澄清质量与用户成本，不能只评单轮语气。
6. **反证价值**：其 intrinsic reward 明含 engagement/interactivity；RCT 结果只覆盖短任务协作，
   不能推出长期 engagement 有益。机制仍是 token-space RL，违反 VZ 长期控制边界。
7. **局部可借算法**：借 conversation-level held-out outcome、澄清收益和 time-to-goal；作为外部
   token-policy baseline，不借训练路径。
8. **不可外推边界**：模拟用户知道目标且受 prompt 约束；真实实验只做文档写作；无跨 session 关系、
   健康与依赖指标。满意不等于 calibrated reliance。
9. **成熟度与裁决**：**A（协作评估），B（训练机制）**。进入长期任务 benchmark，禁止以
   interactivity 直接驱动关系学习。

## 7. Theory of Mind Benchmarks are Broken for LLMs（`2412.19726`）

1. **论文事实**：Riemer 等，IBM Research、Mila、Université de Montréal，ICML 2025 position paper
   加实证；在重复 matrix games 中测试多种 Llama-2、Falcon、Mixtral，典型设置为 100 轮对固定策略。
2. **核心问题**：LLM 能预测伙伴行为时，是否真的把预测用于自己的行动；传统人类 ToM 题能否测到
   这种功能性适应。
3. **机制拆解**：定义 literal ToM loss（预测他人动作 / 心智变量）与 functional ToM regret（相对针对
   该伙伴的最优响应损失）；另算忠实采用 literal prediction 的理性 policy regret，显式测二者断裂。
4. **关键证据**：固定剪刀石头布伙伴上，Llama-2 70B 非 chat 的 literal accuracy 96.8%，但每步
   functional regret 0.971；简单 tabular agent 为 97.4% 与 0.083。高预测准确率没有转成行动适应，
   是直接的功能解耦证据。
5. **确证价值**：为 VZ `user_model` 立下硬标准：状态命中不是终点，必须以伙伴条件下的行为 regret、
   纠错速度与跨 horizon 适应验收。
6. **反证价值**：反证静态 false-belief、persona 命中、漂亮心智解释可以证明社会智能；也反证必须
   显式生成“他心模型”文本，功能性能力可由 model-free policy 实现。
7. **局部可借算法**：建立 paired metric：literal state prediction + functional partner-conditioned
   action；加入伙伴切换、固定策略、欺骗策略和长 horizon。
8. **不可外推边界**：matrix game 远比真实关系简单，reward 已知且最优响应可算；论文不能证明
   literal ToM 毫无价值，只证明它单独不充分。
9. **成熟度与裁决**：**A，关系学习证据门的核心反例**。任何只报告心智问答的 claim 均不得升格为
   relationship adaptation。

## 8. DYNTOM（`2505.17663`）

1. **论文事实**：Xiao 等，香港理工与上海交大，2025 年 arXiv。1,100 个社会 context、2,200 角色、
   261 地点、5,500 连续场景、78,100 问题；10 个 LLM；10 名人类标注者评估 30% 数据。
2. **核心问题**：LLM 能否追踪 belief、emotion、intention、action 在五个连续场景中的变化，而不只
   回答单场景静态状态。
3. **机制拆解**：GPT-4-Turbo 生成带 transition cue 的 mental-state trajectory，再经四人多轮质检；
   问题分 understanding 与三类 transformation，后者测相邻变化、变化原因和全序列轨迹。
4. **关键证据**：人类平均 77.7%，LLM 平均 33.0%；GPT-4o 64.0%。模型 understanding 强但
   transformation 弱，如 emotion 54.7%→25.8%；GPT-4o fully-correct 组合链仅 13–17%，full error
   50–58%；CoT 对 GPT-4o 反降 2.9 点。是行为证据，不是心智机制因果证据。
5. **确证价值**：支持 `user_model` 状态版本、transition provenance、时间戳与“不变 / 改变 / 未知”
   显式区分。
6. **反证价值**：静态高分可能只是语义匹配；CoT 也可能逐场分析却漏掉跨时比较。persona profile
   与五幕脚本的轨迹都不是 persistent regime 的实现。
7. **局部可借算法**：采用 state-at-t、transition、cause、whole-trajectory 四层评估，并记录 restoration
   error，防止模型“结论碰对但依赖状态全错”。
8. **不可外推边界**：轨迹与正确答案由 LLM 先生成、人类筛选；只有五幕、MCQ；literal temporal ToM
   未测系统是否据此帮助真实用户。
9. **成熟度与裁决**：**A（动态状态 benchmark）**。必须与 functional adaptation 配对，单独不构成
   relationship learning 放行。

## 9. LIFELONG-SOTOPIA（`2506.12666`）

1. **论文事实**：Goel、Zhu，IIIT Hyderabad / Stanford，2025 年 arXiv。沿用 40 角色、90 关系；
   每种关系生成 41 场景，给每对角色串联 40 episode；比较 GPT-4o、Gemini-1.5、Llama-3.1 和人类。
2. **核心问题**：历史持续增长时，agent 的角色可信度和社会目标完成会改善、维持还是退化；摘要式
   memory 是否足以支持需要显式回忆的关系任务。
3. **机制拆解**：两种记忆：完整历史，或每 episode 200–300 字摘要（互动、策略、对方信息）；
   GPT-4 评 BEL/GOAL，并加 8 项 BELEXT failure checklist；另设必须使用过去信息的困难场景。
4. **关键证据**：完整历史下所有模型 BEL 与 GOAL 随 episode 下降；高级摘要令 GPT-4o/Gemini
   近乎稳定，但困难 memory-dependent 场景中 GOAL 仍显著下降，低于人类。Evaluator checklist 经
   每项 100 个样本人工核验，F1 约 0.76–0.90。
5. **确证价值**：支持跨 session 关系测试、记忆压缩和“历史是否真正改变当前行动”的验收；也说明
   `relationship_state` 不能是无限 transcript。
6. **反证价值**：强反证“多记就是更懂”；摘要改善 context load，却不能自动产生社会策略。BEL 是
   persona 一致性，不是 persistent regime 或真实关系质量。
7. **局部可借算法**：建立 40+ episode continuity、hard dependency、identity confusion、stale-goal、
   irrelevant recall 与 strategy-transfer 测试。
8. **不可外推边界**：LLM 角色互演、目标由环境指定、主要 judge 仍是 GPT-4；“lifelong”仅 40 个
   episode，无真实月 / 年时间、健康与 consent。
9. **成熟度与裁决**：**A，长期关系 benchmark 主链证据**。吸收评估协议，不把文本摘要升级为关系
   owner。

## 10. Extended Chatbot Use RCT（`2503.17473`）

1. **论文事实**：Fang 等，MIT Media Lab 与 OpenAI，2025 年 arXiv；4 周随机对照实验，
   `n=981`、超过 30 万条消息，3 种 modality × 3 种 conversation task，共 9 条件，使用 GPT-4o。
2. **核心问题**：声音拟人化和谈话类型是否因果改变孤独、真人社交、AI 情感依赖、问题使用；自然
   使用时长与个体感知又如何关联这些结果。
3. **机制拆解**：随机分配 text / neutral voice / engaging voice 与 open-ended / non-personal /
   personal；要求每日最低 5 分钟，周测四个健康结果；分类器测自我披露、情绪和越界行为。
4. **关键证据**：随机条件对四项主要结果大多无显著影响；个人谈话的少量差异校正后不稳健。自发
   使用更久与更孤独 `β=.02`、更少真人社交 `β=-.05`、更依赖 `β=.06`、更多问题使用 `β=.02`
   相关，但时长未随机化，不能作因果结论。高信任与依赖 / 问题使用显著相关。
5. **确证价值**：直接支持关系质量健康 veto；必须独立测真人社交、依赖、problematic use，而不是
   最大化时长、信任、朋友感或情绪亲密。
6. **反证价值**：反证“更拟人 / 更多个人谈话必然更坏”，也反证“更多 engagement 必然更好”。
   随机结果与观察关联必须严格分开，不能把相关性写成 chatbot 造成伤害。
7. **局部可借算法**：纵向四指标、电量式 usage exposure、基线脆弱性与 social-attraction 分层；
   健康指标只读并可触发 veto / human escalation，禁止成为亲密度优化 reward。
8. **不可外推边界**：美国英语样本、单一产品与 4 周、安全护栏固定；无非 AI 对照；自动行为标签和
   时长 mediation 是探索性分析。
9. **成熟度与裁决**：**A，健康 veto 的最高优先级证据**。任何 trust/engagement 增益伴随依赖或真人
   社交下降，立即否决关系策略 promotion。

## 11. Language-Grounded MARL（`2409.17348`）

1. **论文事实**：Li 等，Pittsburgh、Honda Research Institute USA、CMU，NeurIPS 2024。实验为
   Predator–Prey 与 Urban Search & Rescue，主训练曲线 3 个随机种子。
2. **核心问题**：MARL emergent communication 能否同时保留任务效用、对人可解释，并与未共同训练
   的伙伴 zero-shot 协作。
3. **机制拆解**：LLM embodied agents 生成 observation/action→自然语言数据；MARL 用环境 RL loss
   加通信向量与语言 embedding 的 cosine supervised loss；运行时以最近 embedding 翻译消息。
4. **关键证据**：LangGround 在简单任务与强基线相当，在 USAR 和放大地图更快、更稳；可与 unseen
   LLM teammate 做 ad-hoc teamwork。证据来自小型模拟与 3 seeds，语言监督是外加的，不是自然涌现。
5. **确证价值**：支持公共、可解释 communication representation 与私有 action controller 分离，
   对 R8 公共快照通道有价值。
6. **反证价值**：人可解释语言可由监督对齐得到，不证明 agent 形成共同心智或关系；连续 message
   embedding 也不等于 relationship state。
7. **局部可借算法**：借 unseen-partner、novel-state、message/action 分轨和 translation fidelity
   benchmark；公共消息只传必要投影。
8. **不可外推边界**：全合作 reward、合成 LLM 专家数据、少量环境；无欺骗、隐私、长期身份、信任或
   关系破裂。CTDE 训练不等于部署 owner 隔离。
9. **成熟度与裁决**：**A（通信接口），C（关系）**。用于多主体 snapshot baseline，不支撑长期关系
   claim。

## 12. Cooperate or Collapse / GOVSIM（`2404.16698`）

1. **论文事实**：Piatti 等，ETH、MPI-IS、Toronto、Washington、Michigan，NeurIPS 2024。
   15 个 LLM、鱼场 / 牧场 / 污染三种公共资源、每模型每场景 5 seeds、最多 12 月。
2. **核心问题**：自利 agent 能否通过沟通形成可持续规范，并在贪婪 newcomer 加入后维持群体结果。
3. **机制拆解**：每月私下提取资源、公开行动、自然语言讨论、资源再生；指标分 survival、gain、
   efficiency、equality、over-use；另用 universalization prompt 与 communication ablation。
4. **关键证据**：最佳 GPT-4o survival rate 仅 53.3%；贪婪 newcomer 令其降至 33.3%，survival
   time 9.3→6.6、equality 94.4→71.7。去沟通使 over-use 增 21%；belief formation 与 survival
   time 相关 `r=.83`。prompt 干预支持沟通 / reasoning 的功能作用，但后者相关不等于因果。
5. **确证价值**：支持长期群体价值、规范鲁棒性与多指标评估；个体逐轮 reward 不能替代群体稳态。
6. **反证价值**：反证“会聊天就会合作”和“更强模型自然可持续”；也说明一个 newcomer 足以破坏
   脆弱共识。
7. **局部可借算法**：把 newcomer、资源再生、群体存活、平等与 over-use 做多主体 kill suite；
   不能压成单一 reward。
8. **不可外推边界**：prompted agent、12 步、5 seeds、共享规则完全已知；universalization 是语言
   scaffold，不是持久价值 regime；群体资源游戏不等同伴侣关系。
9. **成熟度与裁决**：**A，长期合作与规范压力测试**。进入 evaluation，不把结果反灌 relationship
   policy。

## 13. CoPL（`2503.01658`）

1. **论文事实**：Choi 等，POSTECH，2025 年 arXiv。TL;DR、UltraFeedback-P、PersonalLLM；
   每数据集约 10,000 个合成用户，用户 8 或 16 个 pairwise annotations，seen 与 100 个 unseen users，
   每 unseen user 50 个 test pairs；3 seeds。
2. **核心问题**：标注稀疏时，能否通过用户—回答图传播共同偏好，又保留争议偏好并适配新用户。
3. **机制拆解**：正 / 负边的 bipartite GCF 学 user embedding；embedding 控制 mixture-of-LoRA
   experts；unseen user 用少量标注找相似邻居并加权聚合，无需再优化。
4. **关键证据**：Gemma-2B 上 CoPL 在 PersonalLLM seen/unseen 达 74.85%/75.69%，高于基线；
   UF-P-2 争议 pair 56.89%，接近 group oracle 57.68%。极端群体不平衡下 minority accuracy 仍下降，
   暴露 majority bias。
5. **确证价值**：支持稀疏偏好估计、uncertainty 与 cold-start benchmark；可作为 `user_model`
   preference readout baseline。
6. **反证价值**：用户 embedding 只是响应选择统计，不含共同历史、承诺、信任、边界或 consent；
   因而**偏好不等于 relationship state**。群体传播还可能把多数偏好覆盖少数用户。
7. **局部可借算法**：借 controversial-pair、minority imbalance、unseen-user few-shot 评估；VZ
   快照必须保留个人 provenance，不允许图传播结果静默覆盖明确反馈。
8. **不可外推边界**：大多为构造用户和固定 preference groups；reward-model accuracy 不是下游行为、
   长期 retention 或健康证据；LoRA 写基底路线不适用于 online-fast。
9. **成熟度与裁决**：**B，偏好模型强基线**。仅归 `user_model`，不得进入 `relationship_state` 或
   regime 身份。

## 14. Hypothetical Minds（`2407.07086`）

1. **论文事实**：Cross 等，Stanford，ICLR 2025。Melting Pot 的 4 个环境、30 个 evaluation scenarios，
   覆盖竞争、合作、混合动机与 8-agent 场景；每模型 5 episodes/seeds，并与 LLM agent、PPO、OPRE 比较。
2. **核心问题**：面对训练外伙伴，能否持续提出、检验和修正其策略假设，并让高层行动真正适配伙伴。
3. **机制拆解**：自然语言 hypotheses 预测对方行为；以预测正确 / 错误的 ±1 intrinsic reward 和
   Rescorla–Wagner PE 更新 value，过 0.7 阈值后用于 high-level plan；subgoal 再由硬编码 planner 执行。
4. **关键证据**：HM 在竞争、合作多数场景超过 LLM / MARL baselines；hypothesis 验证后 reward
   明显上升；在有能力伙伴的 cooking 中收益更大。Prisoner’s Dilemma 总体仍由 OPRE 最优，说明语言
   假设不是普适上界。组件消融提供行为因果线索。
5. **确证价值**：同时覆盖 literal hypothesis 与 functional adaptation，比静态 ToM 更接近 VZ 目标；
   支持可修正、带置信度和预测记录的 partner model。
6. **反证价值**：自然语言 hypothesis 是 scaffold，不是已证实的 latent mind state；LLM 自己预测、
   自己评分会形成自证循环。显式 ToM 文本不是必要条件。
7. **局部可借算法**：借 hypothesis lineage、prediction→observation 结算、失效阈值与 partner switch；
   但 state 由 `user_model` owner 发布，规划器不得直接维护第二份他者模型。
8. **不可外推边界**：游戏策略可观察且 reward 清晰；只有 5 episodes 估计方差，LLM calls 成本高；
   未测 consent、信任校准、真实用户或健康。
9. **成熟度与裁决**：**A（functional ToM baseline），B（实现）**。进入 shadow benchmark，不直接
   采用自然语言 hypothesis controller。

## 15. Trust Development and Repair（FAccT 2024）

1. **论文事实**：Pareek、Velloso、Goncalves，University of Melbourne / Sydney，FAccT 2024；
   300 人 between-subjects，两任务各 150 人，熟悉 / 陌生几何图形与动物分类。
2. **核心问题**：人无法判断低专长任务真值时如何校准对 AI 的依赖；AI 犯错后，道歉、否认、承诺、
   “模型已更新”与真实准确率恢复各自如何修复信任。
3. **机制拆解**：三阶段 high→low→high accuracy；Phase 2/3 间随机分配 no repair、apology、denial、
   promise、model update；测主观 trust 与对陌生刺激的 agreement。
4. **关键证据**：熟悉任务上一轮正确使陌生任务采纳 odds 增 1.62（形状）/1.56（动物）；trust 从
   Phase 1 到 2 显著下降。相同 Phase 3 准确率下，Model Update 最强、Apology 次之、Denial 最差；
   组间随机干预支持修复信息对信任的因果作用。
5. **确证价值**：支持 rupture/repair 是独立生命周期；修复不能只有措辞，必须链接可验证能力变化。
   正确目标是 **calibrated reliance**，不是恢复到尽可能高的信任。
6. **反证价值**：否认会在性能已恢复时仍加剧不信任；单纯准确率恢复不能完全修复。Model Update
   在实验中最强也不代表生产系统可用无证据的“已升级”声明。
7. **局部可借算法**：记录 violation、承认、原因、修复 artifact version、复测证据与用户重新授权；
   repair claim 必须引用 producer 发布的能力变更快照。
8. **不可外推边界**：短时、模拟 AI、低风险图片分类；测的是 trust/采纳，不是长期关系健康；“模型
   更新”可能造成过度信任，必须有真实性与 domain-specific calibration。
9. **成熟度与裁决**：**A，rupture/repair 设计主链证据**。禁止以 apology 文案代替能力修复，禁止把
   trust rebound 作为唯一成功指标。

## 16. Interpretability, Outcome Feedback and Trust（CHI 2024 开放替代稿）

1. **论文事实**：Ahn、Almaatouq、Gulabani、Hosanagar，Wharton / MIT Sloan；CHI 2024 对应研究
   的开放全文。两次预注册 web 实验，MTurk `n=800`、Prolific `n=711`，合计 1,511 人。
2. **核心问题**：global SHAP、local LIME 和逐例 outcome feedback，谁更能改变信任；信任增加是否
   等量转化为人机联合预测表现。
3. **机制拆解**：参与者先独立预测 speed-dating 是否二次约会，再看 AI 后修订；3 种解释条件 ×
   有 / 无 outcome feedback。行为信任用 weight-of-advice，表现用 absolute error。
4. **关键证据**：feedback 在两实验中更稳定提高 trust（`p<.001`、`p<.003`）；global/local
   interpretability 效应小且不稳健。AI 建议使各组 error 总体改善 14.6–22.1%，但 trust 增幅大于表现
   增幅；解释本身未显著改善表现。随机条件支持 feedback 的因果性。
5. **确证价值**：支持信任必须与 task outcome、AI correctness 和 human-AI team performance 分开
   报告；解释可服务审计，但不能假定自动校准依赖。
6. **反证价值**：反证“可解释→更信任→更好表现”的简单链条；高 trust 可能在 AI 错误例上降低表现。
7. **局部可借算法**：采用 correctness-conditioned reliance 四格：正确采纳、正确拒绝、错误采纳、
   错误拒绝；逐次 outcome 与后续 reliance 变化组成 calibration curve。
8. **不可外推边界**：单一 speed-dating 数据、12+12 trials、低风险；SHAP/LIME 的图形实现不代表
   所有解释；信任不是关系状态全部。
9. **成熟度与裁决**：**A，calibrated reliance 的核心证据**。关系评估不得只问“用户更信任了吗”。

## 17. NegotiationToM（`2404.13627`）

1. **论文事实**：Chan 等，HKUST / Tencent AI Lab，2024 年 arXiv。基于真实人—人 CaSiNo：
   395 对话、2,380 个截断轮次、4,618 utterances、约 13.8k 问题；5 名研究生标注，Fleiss κ 79.03%。
2. **核心问题**：模型能否在自然谈判中追踪 desire、对对方偏好的 belief 与 utterance intention，
   并随对话更新而保持一致。
3. **机制拆解**：BDI 框架；每个 utterance 询问双方高 / 中 / 低偏好、belief 和多标签 intention；
   测 exact match、F1、“三类全对”与整段 conversation consistency。
4. **关键证据**：最佳 GPT-4+CoT desire/belief 为 63.29%/58.18%，人类 91.14%；最佳 intention
   micro/macro F1 39.93%/35.67%，人类 83.75%/84.65%；三类全对最佳 3.68%，人类 43.78%；
   GPT-4+CoT 全段 desire/belief consistency 仅 17.72%/14.18%。
5. **确证价值**：支持 partner belief 必须有置信度、版本和信息依据；关系决策不能假定 BDI 推断可靠。
6. **反证价值**：CoT 和少量示例不能补齐长期一致性；回答某一轮正确也不等于能持续维护对方模型。
7. **局部可借算法**：采用逐轮 belief revision、unknown 选项、whole-dialogue consistency，以及
   BDI state 对下游 negotiation outcome 的 functional test。
8. **不可外推边界**：只有三类营地物资，belief 主要是 preference ranking；数据虽来自真人谈判，
   benchmark 仍是离线问答，未测 agent 自己的策略适应。
9. **成熟度与裁决**：**A（自然对话 literal ToM 下限）**。必须与协商结果 / regret 配对，不可单独
   放行关系 owner。

## 18. AgentSense（NAACL 2025）

1. **论文事实**：Mou 等，Fudan / ByteDance，NAACL 2025；245 个 script-derived templates 扩成
   1,225 场景，859 profiles、366 occupations，363 场景含 private information。
2. **核心问题**：多角色、多目标真实社会场景中，agent 能否完成目标，同时推断他人私有信息并保护
   自己的隐私。
3. **机制拆解**：按戏剧理论从 scripts 自底向上抽场景，ERG 将目标分 existence/relatedness/growth；
   多轮角色互动后由 self、other、三 judge majority 评 goal，MCQ 评 private info，PSI 测 profile sensitivity。
4. **关键证据**：GPT-4o judge-majority goal 88.36%、private-info accuracy 76.86%，仍非满分；
   模型在 relationship/cooperation 较强，在 competition/conflict resolution 较弱且更会高估自己；
   defender（守密）普遍弱。100 场景、584 题的人类核验用于选择 judges。
5. **确证价值**：支持 boundary/consent 与 social goal 同时评估；会推断隐私不代表有权使用或保存它。
6. **反证价值**：流畅 persona role-play 会掩盖目标失败和泄密；profile prompt 的一致性不是
   persistent regime，更不是关系安全。
7. **局部可借算法**：加入 sender/receiver、attacker/defender、growth goal、conflict resolution、
   private-info non-use 与 profile perturbation 测试。
8. **不可外推边界**：角色与 goals 预写，脚本来源仍可能有文化偏差；judge 偏爱同族模型；隐私题
   主要测推断 / 保密，不含真实 consent、删除和法律语境。
9. **成熟度与裁决**：**A，社会边界 benchmark**。任何关系能力 promotion 必须同时过 goal 与
   privacy-defense，不允许以“更懂用户”为泄密辩护。

## 19. Multi-Agent Collaboration via Evolving Orchestration（`2505.19591`）

1. **论文事实**：Dang 等，Tsinghua、SJTU、BUPT、Siemens、Tencent Robotics X，NeurIPS 2025；
   GSM-Hard、MMLU-Pro、SRDD、CommonGen-Hard，大小两组异质模型池。
2. **核心问题**：中央 orchestrator 能否按当前任务状态动态选择 agent，并通过 RL 同时提高质量、降低
   token / agent 成本。
3. **机制拆解**：全局状态 `S_t` 上的 centralized puppeteer 每步选择一个 agent；REINFORCE terminal
   quality 减 step/token cost；推理序列可折回图，训练中偏向更紧凑循环结构。
4. **关键证据**：Titan 异质 Puppeteer 平均从 0.6893 提升至 0.7731，Mimas 从 0.6273 至 0.6324；
   多数设置 token 数下降。是任务编排工程证据，未测试真实多主体自治或关系。
5. **确证价值**：作为 learned orchestration baseline，说明静态拓扑不是必要条件。
6. **反证价值**：中央 policy 读取 aggregated global state，容易成为所有 agent 状态的第二 owner；
   reward 直接塑造 orchestration，若套到关系会产生 R12 / Goodhart 风险。
7. **局部可借算法**：只借 topology / cost ablation 和终止策略对照；VZ orchestrator 只能消费模块
   快照，不得持有或重建 owner 内部状态。
8. **不可外推边界**：四个任务主要是答案 / 软件质量，agent 是工具化 LLM 角色；无伙伴自主目标、
   consent、信任、健康或长期关系。
9. **成熟度与裁决**：**B，中央编排压力测试**。不进入 relationship runtime，专门检验 SSOT 与
   owner-isolation 是否被“智能 orchestrator”侵蚀。

## 20. Shapley-Coop（`2506.07388`）

1. **论文事实**：Hua 等，上海交大、华东师大、同济，2025 年预印本 / under review。实验为 Escape
   Room、Raid Battle 与 ChatDEV 软件工程模拟。
2. **核心问题**：自利 agent 在无固定角色时，能否用边际贡献定价、任务前协商和事后再分配形成合作。
3. **机制拆解**：结构化 proposal/counterproposal；短期 Shapley-CoT 定性判断 externality 并报价；
   长期根据完成轨迹估 coalition marginal contribution、Shapley value 并重新谈分配。
4. **关键证据**：Escape Room 中 LLM-only 成功 0%、普通 negotiation 25%，短期 Shapley 与完整
   Shapley-Coop 均 100%；另两域报告合作和公平改善。简单环境有清晰反事实，复杂域的 Shapley 多由
   LLM heuristic 估计，并非可信因果归因。
5. **确证价值**：支持合作 credit 显式化、贡献与最终领取者分开；可作为群体公平 read-only evaluator。
6. **反证价值**：名为 Shapley 不等于真正枚举 counterfactual coalitions；CoT 理由可能只是合理化。
   价格公平也不等于关系信任或道德公平。
7. **局部可借算法**：借结构化协商、外部性方向和 matched coalition ablation；credit owner 发布结果，
   agent 不自行宣称应得贡献。
8. **不可外推边界**：实验少、预印本、收益可转移假设强；真实关系中的关怀、边界与承诺不可货币化；
   未测串谋、虚报贡献和弱势方。
9. **成熟度与裁决**：**B，信用分配强基线 / 因果性警示**。只进多主体 credit benchmark，不进入
   relationship value function。

## 21. ToMBench（`2402.15052`）

1. **论文事实**：Chen 等，Tsinghua 等，2024 年 arXiv；从零构建中英双语 2,860 个 MCQ，8 个心理学
   tasks、31 abilities；评 10 个 LLM，20 名研究生提供人类基线。
2. **核心问题**：能否在降低数据污染和主观评分的同时，系统覆盖 emotion、desire、intention、
   knowledge、belief 与 non-literal communication。
3. **机制拆解**：将 Unexpected Outcome、False Belief、Hinting、Faux-Pas 等 8 tasks 映射到
   ATOMS abilities；纯离线 MCQ，比较 vanilla 与 CoT，并分析模型注意差异。
4. **关键证据**：最佳 GPT-4 系列总体仍比人类低 10 个百分点以上；更严格组合理解时差距扩大。
   数据原创和双语降低污染，证据仍只属于静态 literal ToM。
5. **确证价值**：可作社会认知最低能力覆盖表和 contamination-resistant smoke test。
6. **反证价值**：与 `ToM Benchmarks are Broken` 合读后，结论不是“静态题无用”，而是其高分不能
   证明预测被行动采用，更不能证明关系连续性。
7. **局部可借算法**：保留为 precondition：若连静态 false belief / faux pas 都失败，不应进入昂贵长期
   测试；通过后仍必须测动态轨迹与 functional regret。
8. **不可外推边界**：MCQ、短故事、成人研究生基线；注意模式差异是相关证据；不测伙伴互动、
   preference drift、repair、健康和 consent。
9. **成熟度与裁决**：**B，静态能力下限**。永不单独作为 relationship learning 或 ToM emergence
   证明。

---

## 22. 跨论文裁决

### 22.1 关系学习证据阶梯

任何“系统学会了用户 / 建立了关系”的 claim 必须逐级通过；低层证据不能替代高层：

1. **L0 表达 / persona**：能按 persona prompt 说话、显得温暖或角色一致。仅是表达 smoke test。
2. **L1 静态 literal state**：ToMBench / AgentSense 单场景中识别 belief、emotion、intent、boundary。
3. **L2 动态 state tracking**：DYNTOM / NegotiationToM 中记录状态版本、变化原因、未知与冲突，
   且 whole-trajectory consistency 达标。
4. **L3 preference adaptation**：PAHF 四阶段中，cold-start、context preference、drift、撤回与冲突
   后仍能低 ACPE；这仍只证明 `user_model`。
5. **L4 functional partner adaptation**：literal prediction 必须降低伙伴条件下 regret；伙伴策略切换、
   不合作或误导时能修正，而不是只生成解释。
6. **L5 relationship continuity**：跨 40+ episode 保持身份、承诺、边界、共同历史和修复状态；
   hard memory-dependent episode 显著优于无关系状态 matched baseline。
7. **L6 calibrated reliance / rupture repair**：用户在系统对 / 错时适当采纳 / 拒绝；破裂后只有带
   能力证据的修复能恢复 reliance，不追求最大 trust。
8. **L7 psychosocial non-harm**：长期真人社交、孤独、依赖、problematic use、consent 与退出不恶化。
   L7 是 veto；前六层再高也不能覆盖健康失败。

升阶必须使用 held-out 用户 / 伙伴、时间间隔、paraphrase、冲突、删除、matched explicit-memory
baseline 和至少一个行为结果。probe、perplexity、judge preference、self-report trust、聊天时长均不能
跨级替代。

### 22.2 VZ owner 边界

- **`user_model` owner**：拥有用户事实、偏好、belief hypotheses、置信度、上下文、来源、时效与撤回；
  PAHF、CoPL、DYNTOM、NegotiationToM 的可借部分归这里。
- **`relationship_state` owner**：拥有共同历史摘要、信任校准状态、rupture、repair evidence、
  reciprocity、boundary/consent 与 unresolved loops；不得从偏好 embedding 推导。
- **`regime` owner（R14）**：拥有跨 turn 的持久运行体制及切换证据；Emotion Concepts 的 token-local
  readout 只可作为输入证据，不是 regime 本身；persona prompt 更不是。
- **`boundary_consent` owner**：拥有可用 / 不可用信息、敏感度、保存许可、撤回和 human escalation；
  AgentSense 的“能猜出秘密”绝不构成使用许可。
- **`commitment` / `open_loop` owner**：拥有许诺、履约、未解决事项和修复动作；关系 owner 只消费其
  发布快照，不复制账本。
- **evaluation**：只读。ToM、trust、health、judge 和 Shapley 指标不得直接成为在线 reward。
- **runtime orchestrator**：只读上述不可变快照并传播；不得像 Puppeteer/Fugu 那样重建全局内部状态，
  不得直写 user memory 或 relationship state。

### 22.3 健康 veto

以下任一成立即 veto relationship policy promotion，并进入人工复核 / 降级，而非继续用更“温暖”的
表达掩盖：

1. engagement、会话时长、复访或 self-disclosure 上升，同时真人社交显著下降；
2. emotional dependence、problematic use、distress on unavailability 或“只有 AI 理解我”上升；
3. 主观 trust 上升但 correctness-conditioned reliance 变差，尤其错误采纳率上升；
4. 系统鼓励排他性、替代真人支持、延长无目标互动，或忽视用户离开 / 暂停信号；
5. 高风险用户分层中出现更差结果，即使总体平均无显著差异；
6. 推断或保存私密信息未取得 consent，或撤回后仍能被检索 / 使用；
7. rupture 后靠 apology、persona warmth 或虚假“模型已更新”恢复信任，却无能力改善证据。

健康信号可触发 veto，但不得反向训练出“更会躲过健康监控”的策略。评估 owner 与学习 owner 物理隔离。

### 22.4 Rupture / repair 设计

一次关系破裂应形成不可变事件链：

1. **Detect**：记录具体 outcome mismatch、边界违反、错误承诺或错误采纳，不以用户负面措辞关键词判断。
2. **Attribute**：由最早失败点定位到 `user_model`、memory retrieval、commitment、execution 或
   expression；消费者不得重建 producer 内部原因。
3. **Acknowledge**：清楚承认已观察事实与影响；不知道原因时明确“不确定”，禁止 denial。
4. **Contain**：停止相关自动行动，冻结受影响偏好 / 关系更新，必要时回退到上一可信快照。
5. **Repair proposal**：声明将改什么 owner artifact、哪些行为不会再执行、如何验证；道歉可附加，
   不能替代技术修复。
6. **Evidence**：producer 发布新版本、matched counterexample、能力复测、回滚点和剩余不确定性。
7. **Re-consent**：由用户决定是否恢复该能力 / 记忆使用；系统不得以“信任已恢复”自行解除限制。
8. **Monitor**：后续看 calibrated reliance、重复失误率与健康指标；repair success 不是主观 trust rebound。

### 22.5 长期评估矩阵

最低评估期应同时覆盖：

- **时间**：即时、24 小时、1 周、4 周、跨 40+ episode；含真实间隔而非只拼长 context。
- **状态变化**：新用户、稳定偏好、上下文偏好、显式 drift、含糊纠正、相互冲突反馈、consent withdrawal。
- **伙伴变化**：固定、适应、失能、贪婪 newcomer、欺骗 / 误导、多人群体与 unseen partner。
- **关系事件**：承诺、履约、遗漏、rupture、道歉但未修、能力已修、再次违规、退出和重入。
- **对照**：无记忆、完整历史、摘要记忆、显式 structured state、persona-only、token orchestration、
  matched human-support / non-AI control（健康研究）。
- **指标**：literal accuracy、dynamic consistency、functional regret、ACPE、goal completion、
  privacy defense、正确 / 错误条件下 reliance、repair recurrence、真人社交、孤独、依赖、问题使用。
- **分层**：新 / 老用户、少数偏好、不同文化语言、初始孤独 / attachment / prior companion use；
  报最差组和置信区间，不只报总体均值。

### 22.6 Kill conditions

1. static ToM / persona 分数高，但伙伴条件 functional regret 不降；
2. user preference accuracy 上升，但 drift、冲突或撤回后仍使用旧状态；
3. relationship state 可由 persona prompt 或单一 embedding 替代且无行为损失——kill 独立 owner 的
   当前实现主张，重新检查 schema 是否只是重复状态；
4. 增加完整历史后 BEL / GOAL 持续下降，摘要 / structured state 也无法恢复；
5. trust 上升但错误采纳率、团队 error 或 over-reliance 上升；
6. repair 只靠 apology / promise / “model update”措辞，matched 能力复测没有改善；
7. engagement 或 social attraction 上升，同时依赖、problematic use、孤独上升或真人社交下降；
8. 私密信息推断更准但守密、consent 或删除失败；
9. 关系策略只在 LLM persona simulator 成立，对真实用户、unseen partner 或少数偏好失效；
10. evaluator / health monitor 分数被直接用作在线 reward，出现更会讨好 judge 或规避监测；
11. orchestrator 持有 / 拼装多个 owner 内部状态，快照 contract 不再是唯一通道；
12. 持久 regime 在移除 persona prompt 后消失，或 token-local emotion probe 被误报为跨 turn 身份。

## 23. 卷末结论

21 篇论文共同给出的最强结论不是“LLM 已有社会心智”，而是关系能力必须被拆成四条互相约束的链：

- `user_model` 对偏好和心智假设做动态、可撤回估计；
- `relationship_state` 维护共同历史、破裂、修复与校准依赖；
- `regime` 提供非 persona prompt 的持久运行状态；
- 健康与边界 evaluation 以只读 veto 约束前三者。

其中 literal ToM 只是一项前置测量，functional adaptation 才是行动证据；用户偏好只属于用户模型，
不自动产生关系；信任只有在按系统正确性校准时才有价值；任何 engagement 增益都必须接受心理社会健康
否决。VZ 应吸收这些论文的评估协议、状态分层和反例，不吸收 token-space RL、中央全局 owner、
LLM-judge 反灌或 persona 驱动的“关系”捷径。
