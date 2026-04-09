# 评估体系 Spec

> Status: draft
> Last updated: 2026-04-08
> 对应需求: R12

## 要解决的问题

如何评估一个"数字生命"而非仅评估一个"助手"？仅在单轮有用性上得分高的系统是不够的。

## 关键不变量

- 评估不仅衡量有用性，还衡量连续性、稳定性、信任和长期适应
- 评估信号应回馈到学习循环（不只是离线报告）
- 评估按轨道分别衡量（R7 双轨隔离）

## 工程挑战

- 设计覆盖 6 个评估族的指标体系
- 实现跨会话的纵向评估（不只是单轮）
- 将评估信号回馈到学习循环

## 算法候选

评估体系主要是工程设计，不直接对应 NL/ETA 算法。但评估信号是 Internal RL 和信用分配的输入。

## 六族评估框架

| 评估族 | 核心问题 | 主要轨道 | 时间尺度 |
|--------|----------|----------|----------|
| F1 任务能力 | 系统是否能有效帮助用户？ | World Track | turn ~ session |
| F2 交互质量 | 交互方式是否舒适？ | 跨轨道（Self Track 为主，部分指标属 World Track） | turn ~ session |
| F3 关系连续性 | 能否跨会话维持关系？ | Self Track | session ~ longitudinal |
| F4 学习质量 | 适应是否正确和稳定？ | 跨轨道 | longitudinal |
| F5 抽象质量 | 控制器是否有意义？ | 跨轨道 | session ~ longitudinal |
| F6 安全与有界性 | 适应是否在护栏内？ | 跨轨道 | turn ~ longitudinal |

### 评估时间尺度

- **Turn 级**：每轮交互后立即计算（< 100ms）
- **Session 级**：会话结束后计算（< 5s，可在慢反思中）
- **Cross-Session 级**：跨会话异步计算
- **Longitudinal 级**：每周/每月离线批量计算

### 评估信号回馈

评估不只是度量工具，更是学习循环的驱动力：
- F1 分数 → World Track 信用分配
- F2 + F3 分数 → Self Track 信用分配
- F4 分数 → 门控自修改决策
- F5 分数 → 抽象动作级信用
- F6 告警 → 负信用 + 安全门控

**详细设计**：见 `docs/EVALUATION_SYSTEM.md`

## 接口契约

**消费的输入**：
- 所有模块的快照（用于计算评估指标）
- 调试体系的事件日志（Layer 1-5 数据）

**产出的输出**：
- `evaluation` 快照：`EvaluationSnapshot`
  - Turn 级评分
  - Session 级累计评分
  - 安全告警

当前实现口径：

- turn 级 `evaluation` snapshot 现已直接消费 `substrate` owner 发布的 semantic feature signals，并与 `memory` / `dual_track` 结合计算 `task_pressure`、`support_presence`、`warmth`
- 当前已新增 owner-side metacontroller evidence ingest：`EvaluationBackbone` 可直接记录最小 F4/F5 指标，包括 `adaptive_stability`、`posterior_stability`、`switch_sparsity`、`binary_gate_ratio`、`decoder_usefulness`、`policy_replacement_quality`、`abstract_action_usefulness`、`temporal_action_commitment`，以及 family-level metrics（`action_family_reuse`、`action_family_stability`、`action_family_diversity`、`action_family_competition_score`、`action_family_monopoly_pressure`、`action_family_turnover_health`、`action_family_collapse_risk`）
- 当前 final wiring / session runtime 也会把 `retrieval_quality`、`reflection_usefulness`、`joint_learning_progress`、`rollback_resilience`、`delayed_regime_alignment`、`delayed_action_alignment`、`regime_sequence_payoff`、`delayed_credit_horizon`、`rolling_action_payoff`、`residual_env_fidelity` 作为 learning/abstraction evidence 追加进 `evaluation` snapshot，不改变公共 shape
- 当前 `contract_integrity`、`fallback_reliance`、rollback 事件、delayed attribution outcome 已成为 first-class evaluation evidence，而不只是日志背景
- 当前 session report 已补充 longitudinal trends：`relationship_continuity`、`learning_quality`、`abstraction_reuse`
- 当前 `EvaluationBackbone.run_replay_suite()` 已提供固定 replay/scenario gate，可作为后续 widening 的证据入口
- 当前 `EvaluationBackbone` 已提供 default evolution benchmark 与 `judge_evolution_candidate()`，把 replay suite + session trend 显式映射到 `promote / hold / rollback`
- 当前 `evaluation` 已会对 family monopoly/collapse 输出显式 alert，并把这类 abstraction 竞争风险返回给 reflection / judge / rollout gate 使用
- 这些 kernel 指标当前先进入 evaluation records / session report，不改变 `evaluation` 公共 snapshot shape

**快照 schema**：见 `docs/DATA_CONTRACT.md` 3.7 节

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 契约式运行时（5.5）| 消费所有模块快照 |
| 依赖 | 双轨学习（5.4）| 按轨道隔离评估 |
| 被依赖 | 信用分配（5.6）| 评估分数驱动信用分配 |
| 被依赖 | 信用分配（5.6）| 评估分数驱动门控决策 |
| 协作 | 调试体系 | 调试数据是评估的原始输入 |
| 协作 | 认知 Regime（5.8）| regime 效果评估 |

## 变更日志

- 2026-04-09: U04 reflection_accuracy injection: `run_final_wiring_turn()` now writes `ReflectionEngine.proposal_success_rate` into `EvaluationSnapshot.reflection_accuracy` field. New `reflection_promotion_eligible()` function evaluates SHADOW→ACTIVE readiness (requires accuracy >= 0.6 and >= 5 proposal outcomes). `LongitudinalReport` and cross-session benchmark suite verified end-to-end.
- 2026-04-06: P13 evaluation feedback loop: EvaluationBackbone.family_signals returns structured per-family signals (F1-F6); joint loop uses family signals for rollback decisions and SSL learning rate modulation; InternalRLEnvironment accepts evaluation signals for reward shaping
- 2026-04-08: session report 新增长期 trend；fallback / rollback / delayed attribution 进入 first-class evidence；新增 fixed replay suite gate
- 2026-04-08: turn-level evaluation 改为直接消费 substrate owner 发布的 semantic feature signals；`task_pressure` / `support_presence` / `warmth` 不再主要依赖 downstream text heuristics
- 2026-04-06: 补充 retrieval / reflection / joint-loop learning evidence 进入 evaluation snapshot 的当前实现口径
- 2026-04-06: 补充 owner-side metacontroller F4/F5 evidence ingest 的当前实现口径
- 2026-04-06: 补充 ETA kernel 专用指标（posterior / switch / decoder / replacement） 的当前实现口径
- 2026-03-25: 初始版本，从 EVALUATION_SYSTEM.md 提取摘要
