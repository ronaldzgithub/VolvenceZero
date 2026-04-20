# 评估体系 Spec

> Status: draft
> Last updated: 2026-04-20
> 对应需求: R12

## 要解决的问题

如何评估一个"数字生命"而非仅评估一个"助手"？仅在单轮有用性上得分高的系统是不够的。

## 关键不变量

- **评估是 prediction error 的可读层**：评估分数是对预测误差的结构化 readout，不是学习信号的源头（R-PE）
- 评估不仅衡量有用性，还衡量连续性、稳定性、信任和长期适应
- 评估信号应回馈到学习循环（不只是离线报告）
- 评估按轨道分别衡量（R7 双轨隔离）

## 工程挑战

- 设计覆盖 6 个评估族的指标体系
- 实现跨会话的纵向评估（不只是单轮）
- 将评估保持为 PE-first 主链的 readout / gate，而不是反向替代 learning primitive
- 设计 scripted dialogue proof harness，验证高 PE 是否真的触发 temporal response 与后段改善
- 设计 perturbation / replay 变体，验证 proof harness 不会被单一话术模板绑死

## 算法候选

评估体系主要是工程设计，不直接对应 NL/ETA 算法。但其定位受 R-PE 约束：

- prediction error 是原始学习信号
- evaluation 是结构化 readout、rollout gate 与 widening 证据层
- 因此 evaluation 可以追加 evidence、report 和 benchmark 工件，但不应抢占 upstream owner 身份

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

当前实现补充：

- `credit` 已直接消费 `prediction_error`；evaluation 对学习循环的主要作用是 readout、gate context 和 widening evidence，而不是独占 credit 源头

**详细设计**：见 `docs/EVALUATION_SYSTEM.md`

## 接口契约

**消费的输入**：
- 所有模块的快照（用于计算评估指标）
- 调试体系的事件日志（Layer 1-5 数据）
- final wiring enrichment 阶段可额外读取 `prediction_error`、joint-loop report、writeback result 等工件，把它们压成 evaluation evidence，而不改变 `evaluation` slot 的公共 shape

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
- 当前 `volvence_zero.agent.dialogue_benchmark` 已新增 fixed scripted dialogue proof harness：它不改变 `evaluation` snapshot schema，而是按 case 聚合 `prediction_error`、`joint_schedule_action`、`abstract_action`、`regime`、`switch_gate` 与 F4/F5/F2 相关 metrics，用于判断 PE 是否真的驱动 temporal abstraction 与后段改善
- 当前 dialogue proof harness 还支持 A/B baseline（`pe-eta` / `eta-no-pe` / `heuristic-baseline`），并基于 case-level summary metrics 输出相对 delta report；其中 `eta-no-pe` 已改为严格 baseline，不再因为普通 interval update 获得 PE-trigger credit
- 当前 dialogue proof harness 还支持更细的定量 response 指标：`recovery_lag_turns` 与 `pressure_localization_score`，用于衡量系统是否更早、更准确地把 temporal response 放在 pressure window 附近
- 当前 dialogue proof harness 还进一步记录 `pressure_response_precision`、`pressure_response_recall`、`over_response_cost`、`stability_after_recovery_score`，用于衡量 response 的精确度、覆盖度、额外成本和恢复后的稳定性
- 当前 dialogue proof harness 已接入固定 perturbation / replay variants（如 `wording_shift`、`pressure_shift_late`），可在改写措辞和压力位置平移后继续比较 `pe-eta`、`eta-no-pe` 与 `heuristic-baseline`
- 当前 dialogue proof harness 还支持 paraphrase families、seed 驱动的 stochastic variants、replay ranking 和 top-k replay selection artifact，可把最有诊断性的 variants 继续输送给后续 artifact acceptance / replay selection 流程
- 这些 kernel 指标当前先进入 evaluation records / session report，不改变 `evaluation` 公共 snapshot shape

### Dialogue Proof Harness 边界

当前 scripted dialogue benchmark 是评估体系的**内部证明工件**，不是新的 runtime slot：

- 它读取 turn-level runtime result 与 evaluation evidence
- 它产出 case/path/comparison/perturbation report
- 它可以作为 widening / rollout 的证据输入
- 它**不**改变 `EvaluationSnapshot` 公共结构，也不成为新的学习 owner

**快照 schema**：见 `docs/DATA_CONTRACT.md` 3.7 节

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 契约式运行时（5.5）| 消费所有模块快照 |
| 依赖 | Prediction Error 主链 | evaluation 是 prediction error 的结构化 readout 与 benchmark 证据层 |
| 依赖 | 双轨学习（5.4）| 按轨道隔离评估 |
| 被依赖 | 信用分配（5.6）| 评估分数驱动信用分配 |
| 被依赖 | 信用分配（5.6）| 评估分数驱动门控决策 |
| 协作 | 调试体系 | 调试数据是评估的原始输入 |
| 协作 | 认知 Regime（5.8）| regime 效果评估 |

## 变更日志

- 2026-04-20: 接口契约补充 final-wiring evaluation enrichment 与 dialogue proof harness 边界；advanced response metrics 与 perturbation benchmark 明确为 evaluation 内部证明工件，不改变 `EvaluationSnapshot` 公共结构
- 2026-04-09: next_gen_emogpt v2: evaluation repositioned as readout of prediction error (R-PE), not the source of learning; acceptance questions now test for explicit prediction error exposure, dual-track error trajectories, and latent control
- 2026-04-20: 新增 dialogue proof harness：`run_dialogue_pe_eta_benchmark()` 基于 scripted multi-turn dialogue cases 聚合 PE trajectory、temporal trajectory 与 delayed outcome evidence，形成 case-level proof verdict，不改变 `evaluation` snapshot 公共结构
- 2026-04-20: 新增 dialogue weak A/B baselines：`run_dialogue_pe_eta_ablation_benchmark()` 比较 `pe-eta`、`eta-no-pe`、`heuristic-baseline` 三条路径的 case-level summary deltas，不改变 `evaluation` snapshot 公共结构
- 2026-04-20: `eta-no-pe` baseline tightened: dialogue benchmark no longer grants PE-trigger credit to plain interval updates on the `eta-no-pe` path, allowing `pe-eta` to separate from the learned-but-non-PE baseline
- 2026-04-20: 新增 dialogue quantitative response metrics: `recovery_lag_turns` and `pressure_localization_score` quantify how early and how precisely the system responds around pressure turns, still without changing `evaluation` snapshot public shape
- 2026-04-20: 新增 advanced dialogue response metrics: `pressure_response_precision`, `pressure_response_recall`, `over_response_cost`, and `stability_after_recovery_score` to quantify response quality beyond simple pass/fail and lag/localization
- 2026-04-20: 新增 dialogue perturbation benchmark: `run_dialogue_pe_eta_perturbation_benchmark()` evaluates fixed replay/perturbation variants across wording shifts and pressure-position shifts, preserving the same public `evaluation` snapshot shape
- 2026-04-20: 新增 systematic replay layer: paraphrase families, stochastic variant generation, replay ranking, and top-k replay selection artifacts now sit above the fixed perturbation layer without changing the public `evaluation` snapshot shape
- 2026-04-09: U04 reflection_accuracy injection: `run_final_wiring_turn()` now writes `ReflectionEngine.proposal_success_rate` into `EvaluationSnapshot.reflection_accuracy` field. New `reflection_promotion_eligible()` function evaluates SHADOW→ACTIVE readiness (requires accuracy >= 0.6 and >= 5 proposal outcomes). `LongitudinalReport` and cross-session benchmark suite verified end-to-end.
- 2026-04-06: P13 evaluation feedback loop: EvaluationBackbone.family_signals returns structured per-family signals (F1-F6); joint loop uses family signals for rollback decisions and SSL learning rate modulation; InternalRLEnvironment accepts evaluation signals for reward shaping
- 2026-04-08: session report 新增长期 trend；fallback / rollback / delayed attribution 进入 first-class evidence；新增 fixed replay suite gate
- 2026-04-08: turn-level evaluation 改为直接消费 substrate owner 发布的 semantic feature signals；`task_pressure` / `support_presence` / `warmth` 不再主要依赖 downstream text heuristics
- 2026-04-06: 补充 retrieval / reflection / joint-loop learning evidence 进入 evaluation snapshot 的当前实现口径
- 2026-04-06: 补充 owner-side metacontroller F4/F5 evidence ingest 的当前实现口径
- 2026-04-06: 补充 ETA kernel 专用指标（posterior / switch / decoder / replacement） 的当前实现口径
- 2026-03-25: 初始版本，从 EVALUATION_SYSTEM.md 提取摘要
