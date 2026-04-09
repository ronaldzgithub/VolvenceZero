# 多时间尺度学习框架 Spec

> Status: draft
> Last updated: 2026-04-08
> 对应需求: R1, R2, R13

## 要解决的问题

如何让系统在不同时间尺度上学习，同时保持基底稳定、快速适应不阻塞、慢速沉淀不干扰？

## 关键不变量

- 快速适应不需要重写整个模型
- 慢速沉淀不阻塞实时交互循环
- 强化作用于压缩和结构化的内部基底，而非原始行为
- 不同知识不存在同一个参数块中
- 不同状态不以相同节奏更新

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
  SSL: 自修改 Titans 的 DGD 更新压缩当前上下文
  RL:  metacontroller 的切换门和控制器代码实时适应

session-medium (每场景/每会话):
  SSL: CMS 中频层更新，压缩场景级模式
  RL:  抽象动作策略 π 的小幅更新

background-slow (会话间):
  SSL: CMS 低频层更新，压缩跨会话知识
  RL:  控制器先验和策略偏好的反思性更新

rare-heavy (定期离线):
  SSL: 基础模型的持续预训练或蒸馏
  RL:  完整的 Internal RL 训练循环
```

### 冻结基底 + 自适应控制器分层

| 层 | 更新频率 | 算法基础 |
|----|----------|----------|
| 稳定基底 | rare-heavy（离线重训练） | 冻结 LLM，ETA rate-distortion 证明 |
| 自适应控制器 | online-fast ~ session-medium | Metacontroller, Internal RL |
| 记忆系统 | 各层不同频率 | CMS 多频率 MLP 链 |
| 反思路径 | background-slow | CMS 低频层, SSL-RL 交替 |

## 接口契约

**消费的输入**：
- `substrate` 快照：当前可实现的 substrate surface（默认 `feature_surface`，有条件时包含 `residual_activations`）
- `evaluation` 快照：学习质量评估信号

**产出的输出**：
- 各时间尺度的参数更新（通过各自所有者模块发布快照）
- 学习循环的状态信息（用于调试和评估）

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
- 当前 session-medium rollout 已支持 `baseline / causal / causal-binary` 三条实验路径，可直接比较连续 gate 与二值 gate replacement 的差异
- 当前 `AgentSessionRunner` 默认已直接消费真实 transformers substrate，并把 `reflection` / `temporal` 放入 ACTIVE 主链；response layer 只读取 distilled kernel context，但默认行为已受 active temporal/reflection 状态驱动
- 当前 background-slow 反思已不再停留在 memory / regime：`ReflectionSnapshot.policy_consolidation.temporal_prior_update` 会通过 bounded bridge 写回 temporal owner 的 controller priors，并已扩展为 group-level selective writeback（如 `encoder` / `decoder` / `track-*` / `action-families` / `beta-threshold`），受 target-specific credit gate 约束
- 当前 background-slow 反思还可基于 delayed attribution 生成 bounded structural temporal proposal（`merge` / `split` / `prune`），并沿同一条 gate / audit / rollback 链路进入 temporal owner
- 当前 session-medium delayed path 已从单条 delayed outcome 扩展为 multi-step attribution ledger：regime owner 会保留最近 resolved attribution，并发布 rolling payoff summary，供 evaluation / credit / reflection 读取，而不需要 consumer 自己拼接长期轨迹
- 当前 `AgentSessionRunner` 已支持 bounded joint-loop schedule：turn 级可区分 `evidence-only`、`ssl-only`、`full-cycle`，由 orchestrator/session owner 调度而不改变 temporal owner
- 当前 `ScheduledJointLoopResult` 已显式发布 owner path 与 schedule telemetry（如 `ssl_interval` / `rl_interval` / due bits），使 session-medium / background-slow cadence 可检查、可测试
- 当前 rare-heavy v0 已补充离线路径：`SSLRLTrainingPipeline.export_rare_heavy_artifact()` 可导出 controller/memory artifact，由 runtime owner 通过显式 import/rollback surface 应用；完整 base-model 级干预仍属于后续增强

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 契约式运行时（5.5）| 多时间尺度模块需要通过快照交换状态 |
| 被依赖 | 时间抽象与内部控制（5.2）| 提供 metacontroller 运行的时间尺度框架 |
| 被依赖 | 连续记忆系统（5.3）| 提供记忆各层的更新频率框架 |
| 被依赖 | 信用分配与自修改（5.6）| 提供门控自修改的时间尺度约束 |
| 协作 | 评估体系（5.7）| 学习质量评估回馈到学习循环 |

## 变更日志

- 2026-04-06: P17 Unified SSL→RL Training Pipeline: SSLRLTrainingPipeline orchestrates two-phase training — Phase 1 (SSL) discovers switching structure via Eq.3 with non-causal embedder enrichment, Phase 2 (RL) trains causal policy with binary gate. Pipeline manages convergence-based phase transition, checkpointing, and rollback. PipelineConfig controls n_z, convergence thresholds, and max steps.
- 2026-04-08: 默认 session 主链切换到真实 transformers substrate + ACTIVE temporal/reflection；background-slow reflection 新增 typed controller-prior writeback bridge 并进入默认主链
- 2026-04-06: P14 M3 optimizer integration: dual-timescale momentum (fast_beta=0.9, slow_beta=0.99) replaces direct gradient application in SSL trainer; slow momentum signal feeds into CMS session-medium band; SSLTrainingReport carries m3_slow_momentum_signal; joint loop routes M3 slow signal to CMS after SSL optimization
- 2026-04-06: 补充 joint loop 对 metacontroller owner state / rollback reasons 的当前实现口径
- 2026-04-06: 补充 joint loop 的最小 SSL/RL alternation 与 kernel evidence loop
- 2026-04-06: 补充 prior/posterior KL 与 residual control application helper 的当前实现口径
- 2026-04-06: 补充 residual intervention backend 与 causal-binary rollout path 的当前实现口径
- 2026-03-25: 初始版本，从 SYSTEM_DESIGN.md 和 PRD 提取
