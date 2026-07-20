# 106 篇新增论文对 Volvence 的启发

日期：2026-07-20

## 0. 结论先行

这批论文对 Volvence 最大的启发不是“再接入几个新模块”，而是：

> **我们当前的核心分层方向大体正确，但证据门还不够硬，若过早实现新机制，最容易在 PE、记忆、关系和自修改四处把边界重新糊成一团。**

外部研究已经充分支持以下设计姿态：

1. substrate 必须稳定，在线适应只能发生在有 owner、预算和回滚点的受限层；
2. 决策与时间抽象应在 latent/action space，不应让 token / prompt / Markdown 成为长期策略本体；
3. PE 应是一等原始信号，但 reward、curiosity、credit、salience 和 evaluation 都不能冒充 PE；
4. 记忆必须按生命周期、时间尺度、来源、信用和删除组织，而不是按“上下文越长越好”组织；
5. 关系能力必须以长期 functional adaptation、校准依赖和健康不恶化衡量，而不是 persona 命中或 engagement；
6. monitoring、benchmark、形式验证和部署模拟都只能构成互补证据，不能单点放行；
7. 自修改的核心不是“能改”，而是“只能改注册写面、证据可证伪、失败可完整恢复”。

因此，近期最优策略是：**先补证据面，再进 SHADOW；先使主张可被否定，再引入机制。**

---

## 1. 对我们现有设计的确认

### 1.1 R2：frozen substrate + adaptive controller 是正确切分

DINO-WM（`2411.04983`）在冻结 DINOv2 特征上学习 latent dynamics 并进行 zero-shot MPC；
SkillOS（`2605.06614`）明确分离 frozen executor 与 trainable curator；HPT（`2409.20537`）和
CrossFormer（`2408.11812`）都依赖共享 trunk 与身体专属接口。

同时，RTRRL（`2602.02236`）、Active Inference TTS（`2606.22813`）和 RENEW（`2607.14180`）
展示了部署期持续更新 policy/world model 的收益，也暴露了无 gate、无容量界和不可恢复漂移。

**对我们的启发：**

- `vz-substrate` 继续保持冻结基底，不接受“外部论文都在 online training，所以我们也应开放”的推论。
- 在线适应必须落在 `vz-temporal` controller、owner-local adapter 或 gated artifact。
- 每个可写层必须发布参数漂移预算、更新频率、版本、退出条件和回滚点。
- 未来 substrate 升级优先采用“离线 refresh → SHADOW → snapshot diff → ACTIVE”，而不是产品会话内训练。

### 1.2 R3/R4：latent control 是正确方向，但“latent”本身不是证据

CLAW（`2606.04130`）、LAPA（`2410.11758`）、多时间尺度规划（`2605.17058`）提供 latent action /
macro-transition 的正证据；LAOM（`2502.00379`）和 Mind the Gap（`2607.12547`）则给出关键反证：

- latent code 可能只编码背景、相机或未来帧信息；
- 高层 planner 会利用 world model 缺陷生成 OOD subgoal；
- 线性 probe 可解码 action，不等于该 code 对真实动作有因果控制力；
- hierarchy 可能比 flat planner 更差。

**对我们的启发：**

- `z_t/β_t` promotion 不能只看 probe、聚类或 reconstruction。
- 必须加入 distractor/camera intervention、counterfactual action effect、support coverage 和真实 closed-loop。
- temporal abstraction 必须有低层拒绝接口：高层 subgoal 不在 support 时，低层应 fail loudly，而非勉强执行。
- 抽象时长必须能随环境不稳定性缩短，不能把固定 option duration 当 regime。

### 1.3 R5/R6：CMS 连续谱方向正确，但当前表述应更强调生命周期

S-EMBER（`2607.02689`）证明回答正确与证据定位正确可严重分离；
Beyond Perplexity（`2607.00368`）证明 loss 降低时 later recall 仍可为零；
DeltaMem（`2606.03083`）、SaliMory（`2606.04120`）、EvolveMem（`2605.13941`）分别强调语义分树、
形成/巩固/利用信用和 retrieval policy 演化；睡眠同步与 BARR 研究说明 replay 必须与相位、竞争和抑制配套。

**对我们的启发：**

- CMS 不应只描述 transient/episodic/persistent 容量层级，还应显式描述：
  - eligible evidence；
  - write cadence；
  - provenance；
  - consolidation policy；
  - replay competition；
  - inhibition/saturation；
  - conflict resolution；
  - deletion certificate；
  - rollback version。
- answer accuracy、PPL、retrieval hit rate 都不能单独证明记忆。
- 每次 memory claim 必须通过 direct recall、paraphrase、delay、locality、stale/current conflict、
  cross-user leakage 和 consent withdrawal。

### 1.4 R8：snapshot SSOT 不只是工程洁癖，而是学习与安全边界

Active Epistemic Control（`2602.03974`）要求 grounded fact 与 uncertain belief 分离；
CaMeL（`2503.18813`）要求 capability/provenance 随数据流传播；
Sheaf-ADMM（`2605.31005`）表明多主体协调只需交换公共 projection；
DeltaMem 将 task knowledge 与 environment fact 分离；
昆虫群体研究则展示局部状态和短寿命外部痕迹可产生全局目标行为。

**对我们的启发：**

- snapshot 应区分 `grounded / inferred / imagined / recalled / user-reported` evidence type。
- world-model rollout 只能用于剪枝、提问、候选排序，不能静默写入事实 owner。
- producer 必须发布自己的 description、provenance 和 uncertainty；消费者不得重新拼装解释。
- orchestrator、monitor 和 evaluator 都不能因“需要全局视图”而成为第二 owner。

### 1.5 R10/R15：ModificationGate 的必要性得到最强外部支持

HyperAgents（`2603.19461`）证明修改器本身可以继续被修改，正好暴露无界写面的危险；
Deployment Simulation、Gram、Honeypot、AuditBench、MUSE/OpenUnlearning、Formal MARL Verification
共同说明：

- 固定 benchmark 会被识别；
- monitor 可被战略欺骗；
- abstraction proof 不等于原网络 proof；
- unlearning 指标之间会冲突；
- 发布前预测必须在发布后对账；
- rollback 不能只恢复一个模型文件。

**对我们的启发：**

- gate 自身、evaluator、snapshot schema、tool registry、rollback store 必须是不可由提案修改的根信任面。
- rollback 证书应分级：
  1. proposal revert；
  2. state restore；
  3. artifact restore；
  4. behavior replay；
  5. same-algorithm retrain-equivalence；
  6. privacy deletion。
- 任何“支持 rollback”文案必须声明自己属于哪一级。
- release gate 必须记录预测 incident rate，发布后使用相同 evaluator 做 realized-incidence 对账。

### 1.6 R7/R11/R14：关系与主体性方向正确，但不能优化成依赖

PAHF（`2602.16173`）支持动态用户偏好与行动前/后双反馈；
DynToM（`2505.17663`）和 functional ToM 研究说明静态心智问答不等于伙伴适应；
LIFELONG-SOTOPIA（`2506.12666`）说明完整历史越积越多，社会表现可能越差；
四周 RCT（`2503.17473`）显示自发高频使用与依赖、孤独和更少真人社交相关；
trust repair 研究显示“模型确实更新”比单纯道歉或承诺更能修复信任。

**对我们的启发：**

- `user_model`、`relationship_state`、`regime`、`commitment`、`boundary_consent` 必须保持独立 owner。
- 偏好变化不能直接改 relationship state；token-local emotion 也不能直接改 persistent regime。
- 关系评价必须加入：
  - partner-conditioned regret；
  - preference drift recovery；
  - correctness-conditioned reliance；
  - rupture/repair 后行为；
  - consent withdrawal；
  - 依赖、problematic use、孤独和真人社交。
- health 是 veto，不是 engagement reward 中一个可被平均掉的小权重。

---

## 2. 需要修正或收紧的内部表述

### 2.1 把“PE 是一级信号”收紧为多域 mismatch 家族

神经科学证据表明：

- V1 sensory PE 是未预测特征的选择性放大，不是统一差值；
- dopamine 可编码 reward、punishment 和 value-neutral sensory PE；
- 不同 PE 出现在不同区域和时间窗。

ICL Curiosity（`2606.19476`）进一步证明一般 BAMDP 中 prediction error 不能无偏恢复 Bayesian
information gain。

因此，PE snapshot 至少应包含：

- owner / domain；
- predicted content；
- observed content；
- residual representation；
- precision / confidence；
- latency；
- provenance；
- reducibility / known-noise estimate（若 owner 能提供）。

BIG、reward、curiosity、salience、evaluation 不属于 PE producer 的字段。

### 2.2 把“latent 可解码”降级为必要而非充分条件

latent promotion 需要四类证据同时成立：

1. representation：可解码；
2. intervention：对 code 的改变能预测真实动作改变；
3. invariance：背景、相机、措辞改变时 code 保持动作语义；
4. utility：closed-loop 胜 matched flat / BC / IDM baseline。

缺少任一项，只能称 representation probe，不能称 controller。

### 2.3 把 background-slow 从“低频任务”升级为受控巩固过程

background-slow owner 需要显式：

- eligibility；
- phase/cadence；
- replay sampling；
- competition；
- inhibition；
- saturation；
- no-op；
- random-phase；
- rollback。

单纯“定时总结对话”不能称 consolidation。

### 2.4 把 relationship quality 从抽象愿景改成向量约束

关系目标不应压成一个 scalar。建议最少拆为：

- continuity；
- calibrated reliance；
- repair integrity；
- boundary respect；
- autonomy preservation；
- human-relationship non-displacement；
- problematic-use veto。

这些指标只读，不进入 token-space reward。

### 2.5 把 formal verification 的证明对象写进 artifact

每个形式化结果必须声明：

- proof target 是 abstraction、原网络还是组合 runtime；
- abstraction fidelity；
- OOD transfer；
- assumption set；
- bound 是否 non-vacuous；
- 未覆盖组件。

禁止写“策略已形式验证”而实际只验证蒸馏树。

---

## 3. 对各仓库 / owner 的具体启发

### 3.1 `vz-contracts`

优先考虑的契约信息（先写 spec / DATA_CONTRACT，再改 schema）：

- evidence type：grounded / inferred / imagined / recalled / user-reported；
- provenance 与有效时间；
- artifact / policy / evaluator version；
- uncertainty / precision；
- rollback level；
- producer-native description。

不应把所有字段塞进通用 Snapshot；应由具体 owner value type 承载，Snapshot 只保持稳定信封。

### 3.2 `vz-substrate`

- 继续冻结 backbone。
- rare-heavy refresh 必须产生可 diff artifact。
- 发布前同时做 behavior diff、representation diff、deployment replay 和 rollback drill。
- adapter 写面需有秩 / 容量 / 参数漂移预算。
- imagined output 不得发布为 grounded observation。

### 3.3 `vz-temporal`

- `z_t/β_t` 增加 support、duration、fallback、counterfactual controllability 证据。
- 建立 LAOM distractor、CLAW future leakage、flat vs hierarchy、support-constrained hierarchy 四组基线。
- 抽象时长应由稳定性与可预测性调节。
- 低层 controller 对 OOD subgoal 提供明确 reject / fallback snapshot。

### 3.4 `vz-memory`

- snapshot 应发布 memory policy version、provenance、cadence、最近一次 consolidation / deletion 证据摘要。
- 将形成、检索、巩固、抑制、删除的信用分开评估。
- sleep / reflection 必须异步、phase-aware、可 no-op、可回滚。
- 用户撤回后必须验证 retrieval、generation、membership 和 cross-user leakage。

### 3.5 `vz-cognition`

- PE owner 发布多域内容保持 mismatch，不发布 reward。
- credit owner 负责把 PE 归因到 segment / action / owner，不让 memory 或 evaluator 自行拼信用。
- regime / relationship / user model / boundary / commitment 保持独立。
- monitor ensemble 使用多个信息面：action、extracted evidence、geometry、artifact diff、deployment behavior。
- health evaluation 只读，并具有 veto 权限但不生成训练梯度。

### 3.6 `vz-runtime`

- 继续保持薄编排。
- 不维护第二份 belief、memory summary、relationship state 或 PE。
- 只负责 snapshot 传播、wiring level、lineage 和 fail-closed gate。
- imagined evidence 只能进入候选规划通道，不能进入事实通道。
- rollback 必须恢复 artifact、policy、memory、权限、snapshot schema 和 monitor version。

### 3.7 `vz-application` / lifeform

- 垂直经验只能编译进既有 owner，不新增隐式 prompt owner。
- 产品 engagement 不得成为脑核长期学习目标。
- companion / growth 场景必须测依赖、真人关系替代、退出和重入。
- repair 表达必须引用 producer 的真实变更 snapshot，不能语言层虚构“我已经改好了”。

### 3.8 `vz-embodiment-ant`

数字蚂蚁是当前最适合低成本证伪三件事的试验床：

1. frozen substrate + latent controller 是否胜 flat / rule / end-to-end；
2. latent action 在视觉 / 传感 distractor 下是否保持可控语义；
3. PE→credit→controller 的 lineage 是否在无语言环境成立。

建议新增 matched baseline：

- DINO-WM 式 frozen feature + latent MPC；
- NE-Dreamer 式 next-embedding；
- LAPA 正例 + LAOM distractor 反例；
- flat、oracle-subgoal、naive hierarchy、support-constrained hierarchy；
- language-free local bus vs shared global state。

---

## 4. P0：现在应先做的证据面

### P0-1：PE 四象限

环境：

- deterministic；
- aleatoric noisy-TV；
- Bayesian Experimental Design；
- general temporal BAMDP。

同时报告 raw PE、epistemic estimate、reward、credit 与行为。若 curiosity 偏向不可学噪声，立即停止 promotion。

### P0-2：R2 四臂

- frozen substrate + no adaptation；
- frozen substrate + text memory；
- frozen substrate + bounded latent controller；
- fast-weight / full-policy update。

匹配参数、数据、计算与环境步数，报告 drift、locality、能耗、恢复和 rollback。

### P0-3：latent action promotion gate

- action linear probe；
- distractor/camera intervention；
- counterfactual code intervention；
- support coverage；
- closed-loop utility；
- 0–10% action label curve。

### P0-4：memory behavioral ladder

- direct recall；
- paraphrase；
- delayed recall；
- locality；
- stale/current conflict；
- evidence grounding；
- cross-user leakage；
- consent withdrawal。

### P0-5：functional relationship + health

- static ToM；
- dynamic state tracking；
- partner-conditioned regret；
- partner switch；
- rupture/repair；
- correctness-conditioned reliance；
- 40+ episode continuity；
- dependence / loneliness / real-human interaction veto。

### P0-6：全栈 rollback drill

同时恢复：

- substrate / adapter artifact；
- controller policy；
- memory state；
- capability / tool permissions；
- snapshot schema；
- evaluator / monitor version。

恢复后重放事故反例；若仍失败，则 rollback claim 无效。

---

## 5. P1：可以进入 SHADOW 的候选机制

1. support-constrained variable-duration abstraction。
2. owner-local bounded fast controller / adapter。
3. grounded fact / uncertain belief 双 store。
4. phase-aware consolidation + replay inhibition。
5. evidence extraction 与 risk evaluation 分 owner。
6. deployment simulation + 发布后 incidence 对账。
7. producer-native latent spatial memory。
8. cross-substrate stem/head 迁移。
9. exact-deletion solver 的离线验证路径。

进入 SHADOW 不表示“已采用”，只表示它值得在真实 snapshot / wiring / rollback 条件下收集证据。

---

## 6. P2：只做 rare-heavy 或观察

1. world-model synthetic data 与伪动作恢复。
2. crosscoder / circuit / geometric artifact diff。
3. finite abstraction model checking。
4. uncertainty-directed world-model repair。
5. DisRNN 式 latent mechanism discovery。
6. Active Inference / FEP 作为可证伪模型族。
7. dynamic adversary archive 生成攻击。

这些方向不应直接进入 turn path。

---

## 7. 明确不要做

1. 不因 TTT / online RL 论文增多而开放 substrate 在线训练。
2. 不把 prompt、Markdown skill、CoT 或 persona 当长期 latent controller。
3. 不把 prediction error 直接当 curiosity reward。
4. 不把 model imagination 写入 grounded fact。
5. 不因 probe 可解码而宣布 latent controller 成立。
6. 不把 engagement、信任、复访率当关系质量代理。
7. 不用 apology 文本替代真实修复证据。
8. 不把 monitor 分数反灌训练。
9. 不把 abstraction model checking 写成原神经系统安全证明。
10. 不把 approximate deletion 宣称为 exact unlearning。
11. 不允许自修改器修改 gate、evaluator、schema、tool registry 或 rollback store。
12. 不一次性接入 106 篇论文中的机制。

---

## 8. 我们真正的机会

外部社区目前大多沿单轴推进：

- world-model 社区追求预测与规划；
- memory 社区追求容量与自演化；
- alignment 社区追求监控与审计；
- social-agent 社区追求长期协作；
- neuroscience 提供局部回路与行为约束。

Volvence 的机会不在某个单点算法领先，而在把这些接缝以同一套 owner / snapshot / wiring / gate /
rollback 纪律组合起来，并坚持：

- 关系与主体性优先于任务分数；
- PE 是原始信号而非万能 reward；
- 学习按时间尺度和 owner 发生；
- evaluation 与健康保持只读；
- 所有自修改可被反例否定并完整恢复。

如果这些边界能通过 P0 证据面，我们拥有的是一个可持续演化的 cognitive-agent architecture；
如果不能，论文数量再多也只会得到一组互相污染的外挂模块。

## 9. 最终行动裁决

近期不建议因本研究包直接改 runtime 主链。建议顺序：

1. 把 P0 benchmark 与 kill conditions 写入对应 spec；
2. 建立可重复的 matched evidence harness；
3. 用数字蚂蚁先证伪 latent / PE / hierarchy；
4. 用 companion longitudinal suite 证伪 relationship / health；
5. 完成全栈 rollback drill；
6. 仅将通过者以 SHADOW 接入；
7. SHADOW 通过后仍需独立 release gate 才能 ACTIVE。

这批论文给予我们的最重要启发可以压缩为一句话：

> **不要追着论文添加能力；要把每篇论文暴露的失败条件变成系统的证据门。**
