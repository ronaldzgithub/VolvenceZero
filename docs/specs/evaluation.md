# 评估体系 Spec

> Status: draft
> Last updated: 2026-04-25
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
- 当前 `temporal_action_commitment` 已不再主要依赖 public snapshot 的几何代理（code energy + named action bonus）；evaluation enrichment 会优先结合 metacontroller runtime state 的 family/version/competition/switch evidence，使 F5 更接近 owner-side latent control 证据
- 当前 final wiring / session runtime 也会把 `retrieval_quality`、`reflection_usefulness`、`joint_learning_progress`、`rollback_resilience`、`delayed_regime_alignment`、`delayed_action_alignment`、`regime_sequence_payoff`、`delayed_credit_horizon`、`rolling_action_payoff`、`residual_env_fidelity` 作为 learning/abstraction evidence 追加进 `evaluation` snapshot，不改变公共 shape
- 当前 `contract_integrity`、`fallback_reliance`、rollback 事件、delayed attribution outcome 已成为 first-class evaluation evidence，而不只是日志背景
- 当前 session report 已补充 longitudinal trends：`relationship_continuity`、`learning_quality`、`abstraction_reuse`
- 当前 `EvaluationBackbone.run_replay_suite()` 已提供固定 replay/scenario gate，可作为后续 widening 的证据入口
- 当前 `EvaluationBackbone` 已提供 default evolution benchmark 与 `judge_evolution_candidate()`，把 replay suite + session trend 显式映射到 `promote / hold / rollback`
- 当前 `evaluation` 已会对 family monopoly/collapse 输出显式 alert，并把这类 abstraction 竞争风险返回给 reflection / judge / rollout gate 使用
- 当前 `evaluation` 已开始直接读取 memory owner 发布的 runtime-grounded learning evidence：如 `runtime_backbone_signal_quality`、`runtime_backbone_signal_strength`、`fast_memory_signal_norm`、`fast_memory_runtime_alignment`；tower telemetry 仍保留，但默认退到辅助 readout
- 当前 `volvence_zero.agent.dialogue_benchmark` 已新增 fixed scripted dialogue proof harness：它不改变 `evaluation` snapshot schema，而是按 case 聚合 `prediction_error`、`joint_schedule_action`、`abstract_action`、`regime`、`switch_gate` 与 F4/F5/F2 相关 metrics，用于判断 PE 是否真的驱动 temporal abstraction 与后段改善
- 当前 dialogue benchmark 也已开始把 runtime-grounded evidence 聚合进 case/report/gate：`runtime_backbone_evidence_turn_count`、`mean_runtime_backbone_signal_quality`、`mean_fast_memory_runtime_alignment` 已进入 benchmark summary；tower summary 仍保留，但不再单独代表默认主证据
- 当前 `tower-memory-surface` gate 已从单纯可见性检查提升为更接近机制强度 gate：除 tower depth、alignment、consolidation 与 matched-control gap 外，还要求 runtime backbone evidence 与 fast-memory/runtime alignment 一起成立
- 当前 `emergence dashboard` 与 dialogue paper-suite 也已开始直接暴露 runtime summary：canonical/open runtime evidence rate、runtime quality、fast-memory alignment 已进入高层 artifact；tower summary 退到辅证位
- 当前 dialogue proof harness 默认已从弱 A/B baseline 切到更接近论文风格的正交 profile matrix（`pe-eta` / `pe-drive-off` / `eta-off` / `timescale-off`），并继续保留 `pe-eta-no-rare-heavy` 作为 rare-heavy 对照面
- 当前 NL essence 默认 acceptance 已进一步对齐默认 doctrine：默认必需 gate 聚焦 `pe-first / multi-timescale-default / judge-gated-evolution / cross-session-growth`；`slow-shapes-fast` 与 `rare-heavy-net-benefit` 保留为附加强证据面，因为它们代表更强的多时间尺度机制收益，不要求每个默认 benchmark 都必须显著成立
- 当前 stronger proof config 还会把 `pe-eta-no-semantic-label`、`pe-eta-no-reflection-cache`、`pe-eta-pe-readout-only` 纳入同一张 proof matrix，用于把 scaffold removal 与 PE readout-only 变成 first-class comparison，而不是只在单独 debug run 里观察
- 当前 `eta-off` baseline 的口径已进一步收紧为“保留最小 temporal controller capacity，但关闭 ETA-style learned/full temporal path、joint learning 与 PE drive”，不再把“没有 ETA”直接等同于 placeholder/no-temporal-control
- 当前 benchmark 还支持显式 scaffold-ablation controls（如 `pe-eta-no-semantic-label`、`pe-eta-no-reflection-cache`）：它们不改变默认 profile matrix，而是用于单独回答“latent family / PE schedule 是否在去掉一层 heuristic scaffold 后仍能站住”
- 当前 benchmark 也支持 `pe-eta-pe-readout-only`：保留 `prediction_error` slot 与 evaluation-side PE readout，但关闭 PE 对 schedule 与 RL reward 的 primary dominance，用于回答“当前改善主要由 latent mechanism 还是 PE 主导在撑”
- 当前评估层还可对 `pe-eta`、`pe-drive-off`、`pe-eta-pe-readout-only` 生成专门的 PE-dominance comparison report，把 `prediction_chain_turn_count`、`pe_triggered_turn_count`、`pressure_response_precision`、`stability_after_recovery_score` 与 `delayed_improvement_observed` 压成 mechanism-retention / schedule-gap / reward-gap 读数
- 当前评估层还可继续下钻成 case-level diagnosis report：对每个 scripted case 同时比较 `pe-drive-off` 与 `pe-readout-only` 相对 baseline 的掉幅，并给出 `schedule-driven` / `reward-driven` / `latent-fragility-driven` 的 failure mode 解释
- 当前 dialogue proof harness 还支持更细的定量 response 指标：`recovery_lag_turns` 与 `pressure_localization_score`，用于衡量系统是否更早、更准确地把 temporal response 放在 pressure window 附近
- 当前 dialogue proof harness 还进一步记录 `pressure_response_precision`、`pressure_response_recall`、`over_response_cost`、`stability_after_recovery_score`，用于衡量 response 的精确度、覆盖度、额外成本和恢复后的稳定性
- 当前 dialogue proof harness 已接入固定 perturbation / replay variants（如 `wording_shift`、`pressure_shift_late`），可在改写措辞和压力位置平移后继续比较 `pe-eta` 与各个 matched control
- 当前 dialogue proof harness 还支持 paraphrase families、seed 驱动的 stochastic variants、replay ranking 和 top-k replay selection artifact，可把最有诊断性的 variants 继续输送给后续 artifact acceptance / replay selection 流程
- 当前 dialogue proof harness 还支持 rare-heavy artifact acceptance / reject gate + rollback policy：selection artifact 可驱动 rare-heavy 训练与 import，然后用 selected variants 做 pre/post acceptance benchmark，并在 mixed-gain 条件下自动 reject + rollback
- 当前 dialogue benchmark 还新增第一阶段 open-environment extrapolation surface：通过 deterministic、stateful 的用户模拟器生成下一轮 `user_input`，但仍复用 `AgentSessionRunner.run_turn(...)` 与 `dialogue_turn_from_result(...)` 主路径；该层产出独立 `OpenDialogueCaseReport` / `OpenDialogueBenchmarkReport`，刻意不复用 scripted `expected_pressure_turns` 指标
- 当前 open-environment extrapolation 还已进入 comprehensive/staged benchmark 主链：real comprehensive runner 会把 `open_environment` 作为独立 stage 跑完并写入 manifest / final summary，因此 open evidence 不再只是 repo 外围的 standalone widening helper
- 当前 comprehensive benchmark 还会发布一个独立的 `emergence dashboard` artifact：它把 strong-proof panels、open-environment panels、PE-dominance report、case diagnosis、以及 strongest retained path 压成更易读的 summary surface，避免消费者只能自己遍历 case/path report 才看出当前最强证据
- 当前 `emergence dashboard` 还支持稳定导出：评估层会提供 JSON payload builder 与 artifact writer，脚本侧也有独立导出入口，方便把同一份 summary 交给 CI、人工 review、候选比较或 rollout 审核系统，而不是只存在于 Python dataclass 内存里
- 当前 evaluation 层还新增独立的 ETA internal-RL strong-proof 工件：它与 dialogue PE harness 分离，专门验证 hierarchical sparse-reward success、abstract-action reuse、held-out composition、delayed credit alignment 和 policy-update evidence；该工件同样只消费 runtime/evaluation evidence，不新增 runtime slot
- 当前 ETA strong-proof acceptance 也与 NL essence acceptance 显式分离：前者回答“internal RL 是否在抽象动作层形成论文式强证据”，后者回答“系统是否默认呈现 NL/多时间尺度设计本质”
- 当前 ETA strong-proof acceptance 的 gate 也已收紧：每个 gate 都会发布自己的 best competing control 与 raw mechanism metrics，因此默认 acceptance 更偏 fail-closed；当 held-out sparse-reward outcome 与最强 matched control 打平时，`sparse-reward-success` 还要求 full path 通过 fast-prior / replacement-effect 给出额外 mechanism lead，避免把“纯平局”误写成默认胜利
- 当前 ETA proof report 还会显式发布 stronger batch/statistical evidence：`rollout_batch_count`、`mean_rollouts_per_update`、`training_transition_count`、`mean_parameter_change_norm`、`mean_value_loss`、`mean_replacement_effect_delta` 与 `heldout_strong_success_std`
- 当前 evaluation 层也把 `statistical-batch-evidence` 提升为独立 acceptance gate，用来区分“有 mechanism demo”与“有足够统计强度支撑 stronger ETA claim”
- 当前评估层已新增 paper-suite uplift 入口：dialogue 与 ETA proof 均支持 versioned manifest、repeated-run aggregate、`MetricIntervalSummary`、provenance bundle，以及分层 CI（`ci-smoke` / `paper-suite-small` / `paper-suite-full`）
- 当前 ETA proof paper-suite 的 repeated-run aggregate 也已接到这批 stronger metrics：例如 `heldout_strong_success_std`、`mean_rollouts_per_update`、`mean_parameter_change_norm`、`mean_replacement_effect_delta`、`mean_value_loss` 与 `training_transition_count`
- 当前 ETA proof paper-suite aggregate 还会直接输出一层浓缩的 review-style interpretation summary：除了 `interpretation` / `dominant_failure_mode`，还会显式给出 `strongest_competing_control`、`strongest_control_gap` 与一条更接近审稿结论的 `review_summary`
- 当前这条 review-style summary 还不再只看 reference assessment：它会综合 repeated-run 的 strongest-control 排名和 cross-run gap variance，形成更稳定的跨 run verdict
- 当前 ETA sparse-reward / heldout-composition gate 支持 near-tie 机制判定：当 full internal RL 的 held-out strong success 与最强 control 的差距在 tolerance 内时，可用 policy-update / replacement 机制强度作为 tie-break；这避免让未优化 control 仅凭切换形态微分数压过真正发生 Internal RL 更新的路径
- 当前 ETA paper-suite 已新增 open-weight residual evidence 口径：`eta-open-weight-*` manifest 会把 `transformers-open-weight` 作为 primary backend、`trace` 作为 matched fallback control，并把 `real_open_weight_capture_rate`、`real_open_weight_hook_coverage`、`real_open_weight_fallback_rate` 纳入 secondary summaries；真实 residual-control claim 现在 fail-closed 要求 fallback rate 为 `0.0` 且 hook coverage 至少 `0.75`，fallback smoke 只能作为隔离证据
- 当前 ETA claim verdict 已新增 `claim_eta_real_open_weight_residual_control`：它把 real residual capture/control 是否成立从 `claim_eta_internal_rl_advantage` 中拆出来，避免把 synthetic proof harness 的 success 误写成真实 open-weight residual-control success
- 当前 dialogue paper-suite 还会导出固定 `expert_review_packet`，把 canonical transcripts 压成 blinded review items，作为外部/专家锚点评测输入，而不是要求评审者手工从 benchmark logs 里摘录文本
- 当前 paper-suite 证据导出已开始收敛到统一 evidence-program 口径：dialogue / ETA aggregate 可额外发布 pairwise effects、claim verdicts、blind-review artifact 与 unified evidence bundle，具体 claim-to-evidence 映射见 `docs/specs/evidence_program.md`
- 这些 kernel 指标当前先进入 evaluation records / session report，不改变 `evaluation` 公共 snapshot shape

### Dialogue Proof Harness 边界

当前 scripted dialogue benchmark 是评估体系的**内部证明工件**，不是新的 runtime slot：

- 它读取 turn-level runtime result 与 evaluation evidence
- 它产出 case/path/comparison/perturbation report
- 它可以作为 widening / rollout 的证据输入
- 它**不**改变 `EvaluationSnapshot` 公共结构，也不成为新的学习 owner

当前 open-environment dialogue benchmark 同样是评估体系的**内部 widening 证据工件**：

- 它通过 deterministic user simulator 生成更开放的 episode，但 episode 仍只经由 `run_turn(user_input)` 进入主运行时
- 它产出 open-scenario / episode report，重点检查 open episode 中的 PE 暴露、schedule coupling、多时间尺度证据与后段稳定化
- 它**不**改变 `EvaluationSnapshot` 公共结构，也不把 user simulator 提升为新的 runtime owner

同理，ETA internal-RL strong-proof benchmark 也是评估体系的内部证明工件：

- 它读取 `internal_rl` rollout、metacontroller family telemetry 与 delayed credit assignment
- 它产出 case/profile/acceptance report
- 它验证的是 paper-like hierarchical sparse-reward 命题，而不是普通 turn-level PE 响应；当前 acceptance 已把 primary outcomes（held-out success / reuse / credit alignment）与 composite readout（strong success score）显式分离
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

- 2026-04-25: ETA paper-suite 新增 open-weight residual evidence summaries 与 `claim_eta_real_open_weight_residual_control`，把真实 residual-control 证据从 synthetic proof success 中拆分
- 2026-04-22: 补充 case-level PE dominance diagnosis report，用于定位哪一个 case 在去掉 PE 主导后最先塌以及塌在哪一层
- 2026-04-22: 补充 PE-dominance comparison report，用于集中比较 `pe-eta` / `pe-drive-off` / `pe-eta-pe-readout-only` 三条路径的机制保留度
- 2026-04-22: 补充 `pe-eta-pe-readout-only` profile，明确区分“PE 可见”与“PE 主导”两层 benchmark 命题
- 2026-04-22: 补充 scaffold-ablation proof profiles（`pe-eta-no-semantic-label`、`pe-eta-no-reflection-cache`）作为 evaluation 内部 matched controls，用于检验 latent family 与 PE schedule 的去扶手稳健性
- 2026-04-22: 补充 `temporal_action_commitment` 现以 metacontroller family/state evidence 为主；`eta-off` baseline 口径收紧为 matched control 而非 placeholder/no-control
- 2026-04-20: 接口契约补充 final-wiring evaluation enrichment 与 dialogue proof harness 边界；advanced response metrics 与 perturbation benchmark 明确为 evaluation 内部证明工件，不改变 `EvaluationSnapshot` 公共结构
- 2026-04-09: next_gen_emogpt v2: evaluation repositioned as readout of prediction error (R-PE), not the source of learning; acceptance questions now test for explicit prediction error exposure, dual-track error trajectories, and latent control
- 2026-04-20: 新增 dialogue proof harness：`run_dialogue_pe_eta_benchmark()` 基于 scripted multi-turn dialogue cases 聚合 PE trajectory、temporal trajectory 与 delayed outcome evidence，形成 case-level proof verdict，不改变 `evaluation` snapshot 公共结构
- 2026-04-20: 新增 dialogue weak A/B baselines：`run_dialogue_pe_eta_ablation_benchmark()` 比较 `pe-eta`、`eta-no-pe`、`heuristic-baseline` 三条路径的 case-level summary deltas，不改变 `evaluation` snapshot 公共结构
- 2026-04-20: `eta-no-pe` baseline tightened: dialogue benchmark no longer grants PE-trigger credit to plain interval updates on the `eta-no-pe` path, allowing `pe-eta` to separate from the learned-but-non-PE baseline
- 2026-04-20: 新增 dialogue quantitative response metrics: `recovery_lag_turns` and `pressure_localization_score` quantify how early and how precisely the system responds around pressure turns, still without changing `evaluation` snapshot public shape
- 2026-04-20: 新增 advanced dialogue response metrics: `pressure_response_precision`, `pressure_response_recall`, `over_response_cost`, and `stability_after_recovery_score` to quantify response quality beyond simple pass/fail and lag/localization
- 2026-04-20: 新增 dialogue perturbation benchmark: `run_dialogue_pe_eta_perturbation_benchmark()` evaluates fixed replay/perturbation variants across wording shifts and pressure-position shifts, preserving the same public `evaluation` snapshot shape
- 2026-04-20: 新增 systematic replay layer: paraphrase families, stochastic variant generation, replay ranking, and top-k replay selection artifacts now sit above the fixed perturbation layer without changing the public `evaluation` snapshot shape
- 2026-04-20: 新增 rare-heavy acceptance gate: replay-selected variants can now drive artifact training, import, gate evaluation, and rollback when acceptance criteria are not met, still without changing the public `evaluation` snapshot shape
- 2026-04-09: U04 reflection_accuracy injection: `run_final_wiring_turn()` now writes `ReflectionEngine.proposal_success_rate` into `EvaluationSnapshot.reflection_accuracy` field. New `reflection_promotion_eligible()` function evaluates SHADOW→ACTIVE readiness (requires accuracy >= 0.6 and >= 5 proposal outcomes). `LongitudinalReport` and cross-session benchmark suite verified end-to-end.
- 2026-04-06: P13 evaluation feedback loop: EvaluationBackbone.family_signals returns structured per-family signals (F1-F6); joint loop uses family signals for rollback decisions and SSL learning rate modulation; InternalRLEnvironment accepts evaluation signals for reward shaping
- 2026-04-08: session report 新增长期 trend；fallback / rollback / delayed attribution 进入 first-class evidence；新增 fixed replay suite gate
- 2026-04-08: turn-level evaluation 改为直接消费 substrate owner 发布的 semantic feature signals；`task_pressure` / `support_presence` / `warmth` 不再主要依赖 downstream text heuristics
- 2026-04-06: 补充 retrieval / reflection / joint-loop learning evidence 进入 evaluation snapshot 的当前实现口径
- 2026-04-06: 补充 owner-side metacontroller F4/F5 evidence ingest 的当前实现口径
- 2026-04-06: 补充 ETA kernel 专用指标（posterior / switch / decoder / replacement） 的当前实现口径
- 2026-03-25: 初始版本，从 EVALUATION_SYSTEM.md 提取摘要
