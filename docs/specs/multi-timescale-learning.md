# 多时间尺度学习框架 Spec

> Status: draft
> Last updated: 2026-04-25
> 对应需求: R1, R2, R13

## 要解决的问题

如何让系统在不同时间尺度上学习，同时保持基底稳定、快速适应不阻塞、慢速沉淀不干扰？

## 关键不变量

- 快速适应不需要重写整个模型
- 慢速沉淀不阻塞实时交互循环
- 强化作用于压缩和结构化的内部基底，而非原始行为
- 不同知识不存在同一个参数块中
- 不同状态不以相同节奏更新
- 对外时间尺度分层保持稳定，但 owner 内部允许更深的 nested tower；公开 contract 负责兼容投影，内部深度不应泄漏成新的跨模块依赖

## 工程挑战

- 设计支持 4 个时间尺度（online-fast / session-medium / background-slow / rare-heavy）的统一学习框架
- 实现"冻结基底 + 有界控制器"的分层架构
- 实现 SSL-RL 交替循环的多尺度版本
- 确保快速适应不阻塞、慢速沉淀不干扰

## 算法候选

来自 `docs/next_gen_emogpt.md`：

| 算法 | 来源 | 用途 |
|------|------|------|
| NSAM 框架 | NL 附录 A.1 | 将每个时间尺度建模为独立的关联记忆层 |
| CMS 多频率 MLP 链 | NL 附录 A.5 | 第 `i` 层每 `c^(i)` 步更新一次，实现多频率更新 |
| Hope（自修改 Titans + CMS）| NL 附录 A.7 | 高频在线适应（Titans）+ 低频持久存储（CMS）|
| M3 优化器 | NL 附录 A.6 | 双时间尺度动量：快动量 + 慢动量 |
| Rate-distortion 分析 | ETA 附录 B.4 | 证明冻结基底的必要性 |
| SSL-RL 交替循环 | ETA 附录 B.6 | SSL 压缩历史 → RL 在压缩基底上强化 |

### 四个时间尺度的 SSL-RL 交替

```
online-fast (每轮/每 wave):
  SSL: metacontroller / memory 的局部压缩与状态更新
  RL:  metacontroller 的切换门和控制器代码实时适应

session-medium (每场景/每会话):
  SSL: CMS 中频层更新，压缩场景级模式
  RL:  抽象动作策略 π 的小幅更新

background-slow (会话间):
  SSL: CMS 低频层更新，压缩跨会话知识
  RL:  控制器先验和策略偏好的反思性更新

rare-heavy (定期离线):
  SSL: offline substrate artifact 训练、持续预训练或蒸馏
  RL:  完整的 Internal RL 训练循环
```

> 2026-05-29: rare-heavy 的**平台级调度器**已落地(此前只有内核内 PE 触发 +
> 会话内联训练,平台 TrainingJob 仅记录不执行)。DLaaS 现有持久 `training_jobs`
> 表 + `TrainingJobExecutor`(队列 + 后台 worker,`VZ_TRAINING_WORKER=1` 启用)推进
> `pending→running→succeeded/failed`,可插拔 runner(synthetic / figure_lora),
> job create 强制 `allow_adapter_training`/`allow_rare_heavy_refresh`,promote 仍走
> ModificationGate 证据。见
> [`training_executor.py`](../../packages/dlaas-platform-api/src/dlaas_platform_api/training_executor.py)
> 与 `dlaas-api-v1.md`「Training jobs — executor」。真 GPU PEFT bake 仍需运维 override。

### 冻结基底 + 自适应控制器分层

| 层 | 更新频率 | 算法基础 |
|----|----------|----------|
| 稳定基底 | live 默认冻结；rare-heavy 仅离线 owner path | 冻结 LLM，ETA rate-distortion 证明 |
| 自适应控制器 | online-fast ~ session-medium | Metacontroller, Internal RL |
| 记忆系统 | 各层不同频率 | CMS 多频率 MLP 链 |
| 反思路径 | background-slow | CMS 低频层, SSL-RL 交替 |

## 接口契约

**消费的输入**：
- `substrate` 快照：当前可实现的 substrate surface（默认 `feature_surface`，有条件时包含 `residual_activations`）
- `evaluation` 快照：学习质量评估信号
- `prediction_error` 快照：PE-scheduled joint loop 与 rare-heavy review 的直接触发信号

**产出的输出**：
- 各时间尺度的参数更新（通过各自所有者模块发布快照）
- 学习循环的状态信息（用于调试和评估），当前主要通过 `JointCycleReport`、`ScheduledJointLoopResult`、`SSLRLTrainingPipeline` / `PipelineResult`、`RareHeavyArtifact` 与 paper-suite artifact bundle 暴露

当前实现口径：

- P08 先以 heuristic temporal policy 提供 online-fast 的最小状态发布
- 第二阶段 joint loop 已补充 dual-track internal rollout，并在 cycle 结束后执行 abstract-action credit enrichment + bounded writeback；主链 rollout 默认优先使用 open-weight residual backend，trace backend 仅作为 fallback
- 当前 joint loop 已支持 policy checkpoint/rollback：当 reward 明显退化、评估出现高等级告警，或 trajectory-level policy objective / KL 偏移超阈值时回滚 internal RL policy
- 当前 `ETANLJointLoop` 是 online-fast 的正式 owner 路径；`SSLRLTrainingPipeline` 作为 offline/batch 两阶段 owner 使用，不再与 live session owner 混淆
- 当前 joint loop cycle report 已显式发布 metacontroller owner state、rollback reasons、residual backend name / fidelity，使 online-fast controller 演化可被会话级闭环检查
- 当前 joint loop 已拆成最小 `ssl_step -> rl_step -> evidence/writeback` 顺序：先执行 metacontroller SSL 更新，再运行 dual-track internal RL，并将 metacontroller 证据写入 evaluation / credit / regime owner
- 当前 joint loop rollback 已扩展为 cycle 级 checkpoint：当一个 cycle 被判定为坏更新时，会同时撤销 SSL 侧和 policy 侧的本轮变更，而不是只回滚 RL policy
- 当前 online-fast SSL 已显式区分 prior / posterior，并用 closed-form KL 约束 metacontroller latent bottle-neck；当前 internal RL env 也已通过 residual intervention backend 更接近 frozen-model-in-env 的控制路径
- 当前 substrate owner 已落地 `TransformersOpenWeightResidualRuntime` 的真实 middle-layer hook capture/intervention，实现了比 trace backend 更接近 frozen-model residual path 的 online-fast 控制接口
- 当前 online-fast 会发布正式 `substrate_self_mod` proposal / telemetry surface：系统仍可基于上一轮 prediction-error carryover 形成 bounded substrate delta proposal，并通过 `ModificationGate.ONLINE` 预览/审计。默认 continual learner 不把 live substrate 更新作为正向学习面；substrate proposal 在默认路径中保留为 review / rare-heavy / experimental evidence，只有显式实验 live-mutation runner 才会经 runtime owner 的 `export/apply/restore_online_fast_state()` 落地 live substrate 更新
- 当前 session-medium rollout 已支持 `baseline / causal / causal-binary` 三条实验路径，可直接比较连续 gate 与二值 gate replacement 的差异
- 当前 `AgentSessionRunner` 默认已直接消费真实 transformers substrate，并把 `reflection` / `temporal` 放入 ACTIVE 主链；response layer 只读取 distilled kernel context，但默认行为已受 active temporal/reflection 状态驱动
- 当前 `AgentSessionRunner` 默认 memory owner 已携带 learned CMS core（默认 nested MLP profile），不再停留在“无 learned core”的空壳路径；context boundary / rare-heavy import 后会触发 owner-controlled nested reset，并通过 memory lifecycle telemetry 发布 `slow_to_fast_init_benefit`
- 当前 online-fast SSL 的参数更新律已从散落的固定系数收敛到 owner-side `learned_update_rule`：metacontroller 会发布 machine-readable updater state，并让 encoder / decoder / switch 的有效更新强度由 prediction loss、KL、posterior drift 与 non-causal 证据共同驱动，而不是只靠手写比例常量
- 当前 `background-slow` 已从 turn-synchronous bounded apply 切到 session-post slow loop：`run_final_wiring_turn()` 默认产出 machine-readable deferred slow-writeback request，`AgentSessionRunner.begin_new_context()` 会在 context boundary 把该 request 连同 session report / trace statistics / PE summary 打包进 slow-loop queue，并在后台执行 owner-side memory / regime / temporal consolidation，不阻塞用户 turn latency。Phase 2 W2.B EQ-owner uplift 后，`session_post_slow_loop` 模块的发布 wiring 默认已从 SHADOW 翻到 ACTIVE：deferred-job queue state 与 recent completion summaries 进入 `active_snapshots`，与 ACTIVE 的 `experience_consolidation`（consolidated public view）并列发布、不重叠
- 当前 background-slow 反思已不再停留在 memory / regime：`ReflectionSnapshot.policy_consolidation.temporal_prior_update` 会通过 bounded bridge 写回 temporal owner 的 controller priors，并已扩展为 group-level selective writeback（如 `encoder` / `decoder` / `track-*` / `action-families` / `beta-threshold`），受 target-specific credit gate 约束
- 当前 background-slow 反思还可基于 delayed attribution 生成 bounded structural temporal proposal（`merge` / `split` / `prune`），并沿同一条 gate / audit / rollback 链路进入 temporal owner
- 当前 memory owner 已开始把三频带 CMS 升级为 machine-readable nested tower：对外仍发布 `online_fast / session_medium / background_slow` 摘要，但 owner 内部 tower readout / meta-init levels 已进入 retrieval 与 slow consolidation 主路径
- 当前 background-slow memory writeback 已切到 tower-native consolidation：session-post slow loop 触发的 reflection apply 不再只做 artifact promotion + lesson count update，而会把 durable / belief / lesson pressure 压到 memory tower 的 online/session/background levels 上
- 当前 session-medium delayed path 已从单条 delayed outcome 扩展为 multi-step attribution ledger：regime owner 会保留最近 resolved attribution，并发布 rolling payoff summary，供 evaluation / credit / reflection 读取，而不需要 consumer 自己拼接长期轨迹
- 当前 `AgentSessionRunner` 已支持 bounded joint-loop schedule：turn 级可区分 `evidence-only`、`ssl-only`、`full-cycle`，由 orchestrator/session owner 调度而不改变 temporal owner
- 当前 `ScheduledJointLoopResult` 已显式发布 owner path 与 schedule telemetry（如 `ssl_interval` / `rl_interval` / due bits），使 session-medium / background-slow cadence 可检查、可测试
- 当前 substrate rare-heavy 已从 v1 的 bounded scalar state 升级到 `adapter-delta-v2` owner path：`SSLRLTrainingPipeline.export_rare_heavy_artifact()` 现在可同时导出 `temporal / memory / substrate` 三类 artifact，其中 substrate checkpoint 不再只携带控制尺度、语义混合权重和 anchor bias，还会携带 owner-side adapter delta payload、compatibility fingerprint、training mode 与 payload parameter count
- 当前 substrate owner 已补充 rare-heavy `export / import / rollback / clone_for_rare_heavy / train_rare_heavy` surface；session 不直写 substrate 内部状态，而是通过 artifact 交给 owner 审阅或在实验/rare-heavy lane 应用。默认 continual learner 的主证据来自 memory / temporal / regime / reflection owner writeback；offline clone 仍是训练 owner，显式 frozen runner 保留 review-only / no-import 语义
- 当前 `SSLRLTrainingPipeline` 已从“trace 顺序绑定”的轻量 pass 收敛为分阶段 batch loop：SSL 阶段按 trace 迭代直到收敛或上限，RL 阶段按 substrate batch 迭代直到收敛或上限，并在结束后导出 substrate rare-heavy checkpoint；若 substrate owner 没有产出 `adapter-delta-v2` checkpoint，pipeline 会 fail closed，而不是静默退回旧 bounded 模式
- 当前 `AgentSessionRunner` 已补充 substrate-aware rare-heavy review 入口：当 PE-scheduled joint loop 发出 `rare_heavy_review_recommended`，session owner 会基于最近 trace window 与最近真实 substrate capture window 克隆 offline temporal/memory/substrate state，运行轻量 `SSLRLTrainingPipeline` 生成 candidate artifact。默认主路径把这些 artifact 视为 review / evidence bundle / rollback-ready upgrade candidate；显式实验或 rare-heavy import lane 才会回写 online substrate owner，显式 frozen runner 仍只保留 review-only 证据链
- 当前 offline pipeline 已把 `TRANSITION` 从占位状态收紧为真实 takeover phase：进入 RL 前会先跑 causal takeover 检查，显式发布 `transition_agreement`、`switch_sparsity_retention`、`family_reuse_retention` 与 `takeover_ready`
- 当前 RL step 也已不再默认单 batch 更新：pipeline 会按 `rl_rollouts_per_step` 组合多个 rollout batch，再执行 grouped dual-track optimize
- 当前 online-fast joint loop 已支持可配置的 rollout accumulation：允许先累积多轮 dual-track rollout，再一次性触发 RL 更新，避免把每一轮都当成独立充分统计样本
- 当前 schedule owner 还把这条 batch policy 公开化了：当 `rl_batch_accumulation_size > 1` 时，scheduled path 会显式区分 `full-cycle-collect`、`full-cycle-batch` 与 `full-cycle-batch-forced`，并通过 schedule telemetry 发布 `rl_batch_target`、`pending_batch_count`、`rl_batch_ready_due` 与 `rl_batch_wait_due`
- 当前 scheduled joint loop 还进一步进入联合调度阶段：batch collect/flush 不再只看 interval 与 wait-limit，还会联合考虑 `prediction-error` 强度、`family_stability`、`rollback_risk`、`transition_pressure`、`substrate_pressure` 与 `rare_heavy_pressure`，并在高风险场景下显式走 `ssl-only-rare-heavy-hold`、`ssl-only-risk-hold` 或 `evidence-only-risk-hold`
- 当前这些 scheduler pressure 也已不再停留在 telemetry：final wiring / evaluation enrichment 会把它们写成 turn-level evaluation records（如 `scheduler_pe_pressure`、`scheduler_family_stability`、`scheduler_rollback_risk`、`scheduler_transition_pressure`、`scheduler_substrate_pressure`、`scheduler_rare_heavy_pressure`、`scheduler_discipline`），因此它们会进入 session report、cross-session growth 和 longitudinal analysis

## 当前 proof surface

当前 repo 在多时间尺度学习上优先证明 3 条工程命题：

1. `PE-schedule coupling`
   - `prediction_error` 已直接驱动 `JointLoopSchedule` 的 `ssl-only[-pe] / full-cycle[-pe] / online-fast substrate due / rare-heavy review`
2. `multi-timescale default path`
   - 默认 `pe-eta` 路径中，可同时观测 `online-fast` controller/memory learning、online-fast substrate proposal evidence、session-post `background-slow` completion、以及 nested CMS lifecycle signals；当前 benchmark 不再把 substrate apply 当作默认正向证据，而是接受 `reflection_promotion_eligible`、`online_fast_substrate_recommended_count`、`rare_heavy_recommended` 与 session-post completion telemetry，并要求 runtime backbone evidence 与 fast-memory/runtime alignment 一起出现
3. `internal-depth-with-contract-stability`
   - memory owner 内部可以增加 nested tower depth，但对外的 owner / snapshot graph 不增加新的跨模块依赖；下游继续只消费 `memory` slot，而不会直接触碰 tower internals
4. `frozen-control evidence retention`
   - 显式 frozen / review-only control 只阻止 live mutation，不阻止证据沉淀；只要 runtime 产出合法 `fast_memory_signal`，session owner 也要把它写入 `MemoryStore`，供 benchmark、evaluation 和 matched-control 比较消费

这些 proof surface 主要由 `volvence_zero/agent/dialogue_benchmark.py` 和 real comprehensive benchmark 提供证据；它们证明的是“默认主路径确实进入了多时间尺度学习，并开始允许 owner 内部 tower 深化，而且 runtime/backbone evidence 已经进入默认 acceptance surface”，不是“论文级最优性或完整因果隔离已经成立”。

## NL 全量真 autograd 迁移（Phase 4–5，CMS + deep optimizer + LSS）

目标线为**全面真 autograd（torch），含 runtime CMS**；纯 Python CMS band 降级为回滚基线，torch 路径经 offline -> SHADOW -> ACTIVE 推进，可回退。autograd 作用于 CMS 控制器/记忆层，不下探 substrate（R2），torch 张量不进公共 snapshot（R8）。

- **Phase 4（CMS band 真 autograd）**：`volvence_zero.memory.torch_cms_band` 提供 `BackendCMSBand`（backend-agnostic 前向，pure/torch 同权重）与 `TorchCMSBand`（真 autograd：同样的 `y = clamp(x + W1·tanh(W2·x))` 前向，用 `torch.autograd` 的 MSE-to-target 代替手写梯度）。`cms_band_shadow_dual_run` 证明 torch 前向同时复现 **pure-backend 前向**与**生产 `CMSBandMLP.forward`**（float64 容差 1e-9）并量延迟，`promotable = within and latency_ok`——torch band 是同一函数，回滚精确。`resolve_cms_backend(active)`：非 ACTIVE 走 pure 回滚基线。
- **Phase 5（deep optimizer 补项 + 真梯度 LSS）**：
  - `m3_optimizer.DeltaMomentumOptimizer` 实现 NL Figure 4 的 delta-momentum（梯度依赖 weight decay：梯度反向 momentum 时强衰减、随梯度幅度轻衰减，使 momentum“需要时停下”）。`compare_momentum_overshoot` 证明 delta-momentum 相比 plain momentum 显著降低 overshoot（更快稳定）。
  - `volvence_zero.prediction.torch_lss` 把 NL 的 LSS（`∂L/∂output`）实现为一等可审计 **offline** artifact：`compute_gradient_lss` 用真 autograd 返回输出层梯度（MSE 下 == `predicted - actual`，即 NL“梯度即被记忆内容”的恒等式）；`compute_lss_through_predictor` 展示 LSS 经 backprop 流入权重（associative memory 构建）；`bridge_runtime_pe_to_lss` 证明 **runtime 语义 PE == −真 LSS**（符号正确、幅度相等），把有界 runtime 代理 grounding 到真梯度信号。runtime online-fast 仍用语义 `PredictionError`，真 LSS 经 rare-heavy 桥接。

### 仍为 backlog 的 NL 构件（still-target，本轮不强行实现）

- **Adam-as-associative-memory**：NL 把 Adam 二阶矩建模为梯度上的 associative memory；当前仅 substrate offline 用标准 PyTorch Adam，未做 NL 推导版本。
- **DGD / 自修改 Titans 递归网络（HOPE 自指权重更新）**：当前 `CMSHopeSelfModificationState` 与 `SubstrateSelfModModule` 是 PE 门控的提议/telemetry，非真正的自修改递归架构。
- **统一 nested autograd tower**：层间知识转移仍靠 cadence gating + meta-init target + M3→CMS 信号注入近似，未做“同一计算图、各层不同学习率的共享 backprop”。

这些 backlog 项有明确 owner（vz-memory / vz-substrate）与退出条件，需独立收敛包 + 评估证据先行后再落地。

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 契约式运行时（5.5）| 多时间尺度模块需要通过快照交换状态 |
| 依赖 | Prediction Error 主链 | 当前 joint loop schedule / rare-heavy review 已直接受 PE 强度驱动 |
| 被依赖 | 时间抽象与内部控制（5.2）| 提供 metacontroller 运行的时间尺度框架 |
| 被依赖 | 连续记忆系统（5.3）| 提供记忆各层的更新频率框架 |
| 被依赖 | 信用分配与自修改（5.6）| 提供门控自修改的时间尺度约束 |
| 协作 | 评估体系（5.7）| 学习质量评估回馈到学习循环 |

## 变更日志

- 2026-06-29: autograd-owner-integration（CMS 主链）。`CMSMemoryCore(cms_backend=WiringLevel)` 与 `build_default_memory_store(cms_torch_backend=...)` 把 torch autograd band 梯度核接入 `_band_mlp_update`：DISABLED 走纯 `CMSBandMLP`（默认/回滚基线）；SHADOW 跑真 autograd 步作证据不写回；ACTIVE 用 `torch_cms_band.torch_band_update_from_params` 做权威 W1/W2 梯度步，并保留 band 的纯 state/momentum 以维持 backflow/mix_from/checkpoint 一致。跨 `SEQUENTIAL/INDEPENDENT/NESTED` × replay on/off 验证；ATLAS/Titans uplift 与 torch backend 正交。`CMSMemoryCore.latest_cms_backend_evidence` 暴露只读证据。
- 2026-06-29: NL 全量真 autograd 迁移 Phase 4–5。新增 `memory/torch_cms_band`（CMS band 真 autograd + backend-agnostic 前向 + SHADOW parity vs 生产 band）、`m3_optimizer.DeltaMomentumOptimizer`（NL Figure 4 delta-momentum + overshoot 证明）、`prediction/torch_lss`（真梯度 LSS offline artifact + runtime PE == −LSS 桥接）。Adam-as-memory / DGD / 自修改 Titans / 统一 nested autograd tower 列为显式 backlog。
- 2026-04-25: 补充当前 joint-loop / pipeline / paper-suite 的具体输出类型，避免把“学习循环状态”误读成无 schema 的自然语言报告
- 2026-04-20: 接口契约补充 `prediction_error` 作为 PE-scheduled joint loop 与 rare-heavy review 的直接触发信号
- 2026-04-09: next_gen_emogpt v2: design thesis recentered on prediction error / LSS as primitive learning signal (NL §3.1); NL appendix compressed to design implications; CMS / M3 / Hope positioned as design patterns, not mandatory implementations; R-PE added as new requirement upstream of R9 credit
- 2026-04-09: U02 Nested CMS meta-learning: `CMSVariant.NESTED` added. In nested mode, background band meta-learns ideal initialization targets for session band (`_nested_session_init_target`), session band meta-learns targets for online band (`_nested_online_init_target`). `_update_nested_meta_targets()` runs each step, tracking convergence of faster bands and adjusting init targets with `meta_lr = background_lr * 0.5`. `reset_context()` re-initializes fast bands from these meta-learned targets (not simple state copy). `CMSCheckpointState` extended with `nested_session_init_target` / `nested_online_init_target`. Verified: init error decreases across repeated context resets (meta-learning converges).
- 2026-04-09: U02 CMS MLP Upgrade: CMSMemoryCore now supports `mode="mlp"` with 2-layer residual MLP per band (`CMSBandMLP`: `y = x + W1 @ tanh(W2 @ x)`). Each band has independent MLP weights, momentum, and gradient-style updates. Anti-forgetting backflow extended to MLP parameter mixing. `CMSVariant` enum (`SEQUENTIAL`/`INDEPENDENT`/`NESTED`) added for band composition modes. `observe_family_signal()` accepts action-family observations into session-medium band. All schema extended with backward-compatible defaults (`mode`, `mlp_param_count`, `variant`, `mlp_params`). Default `mode="vector"` preserves all existing behavior.
- 2026-04-06: P17 Unified SSL→RL Training Pipeline: SSLRLTrainingPipeline orchestrates two-phase training — Phase 1 (SSL) discovers switching structure via Eq.3 with non-causal embedder enrichment, Phase 2 (RL) trains causal policy with binary gate. Pipeline manages convergence-based phase transition, checkpointing, and rollback. PipelineConfig controls n_z, convergence thresholds, and max steps.
- 2026-04-08: 默认 session 主链切换到真实 transformers substrate + ACTIVE temporal/reflection；background-slow reflection 新增 typed controller-prior writeback bridge 并进入默认主链
- 2026-04-22: `background-slow` 默认主路径切换到 session-post slow loop：turn 内 final wiring 改为 deferred request，context boundary enqueue queued consolidation job；dialogue longitudinal / essence benchmark 也开始把 session-post completion 作为默认 background-slow evidence
- 2026-04-06: P14 M3 optimizer integration: dual-timescale momentum (fast_beta=0.9, slow_beta=0.99) replaces direct gradient application in SSL trainer; slow momentum signal feeds into CMS session-medium band; SSLTrainingReport carries m3_slow_momentum_signal; joint loop routes M3 slow signal to CMS after SSL optimization
- 2026-04-06: 补充 joint loop 对 metacontroller owner state / rollback reasons 的当前实现口径
- 2026-04-06: 补充 joint loop 的最小 SSL/RL alternation 与 kernel evidence loop
- 2026-04-06: 补充 prior/posterior KL 与 residual control application helper 的当前实现口径
- 2026-04-06: 补充 residual intervention backend 与 causal-binary rollout path 的当前实现口径
- 2026-03-25: 初始版本，从 SYSTEM_DESIGN.md 和 PRD 提取
