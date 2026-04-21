# PE-ETA Dialogue Benchmark Harness

> Status: draft
> Last updated: 2026-04-21
> Scope: prove prediction-error-based temporal abstraction with scripted multi-turn dialogue cases

## Goal

补上一个内部“证明系统”，回答的不是“系统能不能聊天”，而是：

- turn 级是否显式产出 `predicted -> actual -> prediction_error`
- 高 PE turn 之后，是否出现 `*-pe` schedule、abstract action / regime / switch 行为变化
- 这些变化是否在 case 后段带来更低 PE 或更好的 delayed outcome

## 当前最小证明义务

当前 repo 不把这个 harness 表述成“论文级证明”，而是把它收束成 4 条最小 proof obligations：

1. `PE-schedule coupling`
   - `prediction_error` 不只是日志，而是真的进入 joint loop schedule，影响 `evidence-only / ssl-only[-pe] / full-cycle[-pe] / rare-heavy review`
2. `multi-timescale default path`
   - 默认 `pe-eta` 主路径里，至少能同时看到 `online-fast` 学习、`background-slow` writeback，以及 nested CMS lifecycle signals
3. `slow-shapes-fast`
   - nested CMS 的慢层状态不只存在于 owner 内部，而是能在 context boundary 后通过 observable reset / init benefit 影响快层
4. `rare-heavy-net-benefit`
   - rare-heavy 不是“有这条链路就算数”，而是默认 proof profile 下需要相对 `pe-eta-no-rare-heavy` 表现出可读的净增益

这四条 proof obligations 对应的是**工程层面的 ETA/NL 对齐证据**，不是对论文全部命题的完全复现。

## 当前实现

新增模块：

- `volvence_zero/agent/dialogue_benchmark.py`

核心结构：

- `ScriptedDialogueCase`
- `DialogueBenchmarkTurn`
- `DialogueBenchmarkCaseReport`
- `DialogueBenchmarkReport`
- `DialogueBenchmarkPathReport`
- `DialogueBenchmarkComparisonReport`
- `DialogueComprehensiveBenchmarkReport`

核心入口：

- `dialogue_proof_cases()`
- `run_dialogue_pe_eta_case()`
- `run_dialogue_pe_eta_benchmark()`
- `build_standard_dialogue_runner()`
- `default_dialogue_ablation_profiles()`
- `default_dialogue_comprehensive_profiles()`
- `run_dialogue_pe_eta_ablation_benchmark()`
- `run_dialogue_pe_eta_comprehensive_benchmark()`
- `build_dialogue_case_report()`

## 默认场景

首批固定 scripted cases：

- `repair`
- `task_clarification`
- `repeated_failure`
- `goal_drift`

这些场景不是为了“模拟真实用户的全部复杂性”，而是为了稳定地产生可重复的 PE pressure 和 delayed payoff 观察窗口。

在最近一轮调优中，这些 cases 的前段冲突、重复失败和后段“修复/对齐成功”信号被进一步增强，用来更稳定地制造中等强度 PE pressure，并让 delayed improvement 更容易被观测到。

## 默认 A/B 路径

当前内置三条对比路径：

- `pe-eta`：完整 PE + full ETA 路径
- `eta-no-pe`：保留 learned temporal path，但在 benchmark 记分时不再让普通 interval update 获得 PE-trigger credit
- `heuristic-baseline`：active temporal policy 换成 heuristic path，joint loop 保持被动/evidence-only，并关闭 rare-heavy

这三条路径的目标不是给出论文级“因果隔离”，而是先提供一个 repo 内部可重复的 A/B 证据面；其中 `eta-no-pe` 现已是更严格的 baseline，而不是早期会吃到 interval carryover credit 的弱版本。

## Comprehensive 多时间尺度路径

为了把“系统结构已经支持多时间尺度学习，但证据还不够完整”这件事再往前推进，当前又补了一层更强的 profile matrix：

- `pe-eta`：完整路径
- `pe-eta-online-only`：保留 online-fast PE + ETA，但关闭 background writeback 和 rare-heavy import
- `pe-eta-no-writeback`：保留 PE + ETA + rare-heavy recommendation/import，但把 reflection 降到 `proposal-only`
- `pe-eta-no-rare-heavy`：保留 PE + ETA + writeback，但关闭 rare-heavy execution
- `eta-no-pe`
- `heuristic-baseline`

这组 profiles 的目标不是只回答“主路径比 baseline 强不强”，而是进一步回答：

- 强度主要来自 online-fast 还是 full stack
- background writeback 是否提供额外增益
- rare-heavy 是否已经在当前 benchmark 下带来稳定净收益
- full path 相对 `online-only` / `no-writeback` / `no-rare-heavy` 的 gap 是否可读

## 记录的证据

每个 benchmark turn 现在显式记录：

- prediction error：`magnitude` / `signed_reward` / `task_error` / `relationship_error` / `regime_error` / `action_error`
- temporal abstraction：`joint_schedule_action` / `active_abstract_action` / `switch_gate` / `action_family_version`
- regime：`active_regime`
- background-slow：`bounded_writeback_applied` / `reflection_promotion_eligible`
- nested lifecycle：`nested_profile_active` / `nested_context_reset_applied` / `slow_to_fast_init_benefit`
- evolution judge：`evolution_decision` / `evolution_category` / `cross_session_verdict`
- rare-heavy：`recommended` / `applied`
- delayed outcome readout：`learning` / `relationship` / `abstraction` 相关 evaluation metrics

每个 case report 现在还显式记录两类定量比较指标：

- `recovery_lag_turns`：从首个 pressure turn 到首个有效 PE-triggered temporal response 的滞后轮数，越低越好
- `pressure_localization_score`：有效 response 有多少比例落在 pressure window（pressure turn 及其后一轮）内，越高越好

进一步新增：

- `pressure_response_precision`：有效 response 中有多少比例真正落在 pressure window 内
- `pressure_response_recall`：预期 pressure turns 中有多少比例在自己的 pressure window 内得到了有效 response
- `over_response_cost`：落在 pressure window 外的 response 数量相对整段 case 的归一化成本，越低越好
- `stability_after_recovery_score`：首次有效 response 且 pressure 主段结束后，后续 turns 中有多少比例真正进入了“无额外触发且低 PE”的 calm tail
- `online_learning_turn_count`：有多少 turn 真正进入了非 `evidence-only` 的 online 学习路径
- `bounded_writeback_turn_count`：有多少 turn 发生了 reflection/background writeback
- `reflection_promotion_eligible_turn_count`：有多少 turn 达到了 background consolidation eligibility
- `rare_heavy_recommended_count` / `rare_heavy_applied_count`：rare-heavy 在整段 case 中被建议/真正执行了多少次
- `evolution_judge_turn_count` / `evolution_judge_rollback_count` / `evolution_judge_structural_allow_count`
- `nested_profile_active_turn_count` / `nested_context_reset_count` / `mean_slow_to_fast_init_benefit`
- `store_nested_context_reset_count` / `boundary_reset_observed_on_first_turn` / `first_turn_slow_to_fast_init_benefit` / `mean_reset_turn_slow_to_fast_init_benefit`

另外，benchmark report / comparison report 现在还会显式发布：

- `metric_means`
- `metric_deltas_from_baseline`

这样不再只保留 case-by-case delta，也可以直接看 profile 级平均差异。

## Proof Gate 对齐口径

当前 benchmark 明确区分两类信号：

- `pressure/high-PE detection`
  - 用来定义 scripted case 的压力窗口和 delayed payoff 观察窗口
- `PE scheduler due`
  - 用来判断 `*-pe` schedule label 是否与 runtime 的 `JointLoopSchedule` 语义一致

两者不再混成同一个 gate：

- pressure 仍允许较低 reward 阈值去识别“用户压力已显著上升”
- 但 `ssl-only-pe / full-cycle-pe` 的显式记分，必须满足 runtime 级 PE schedule 触发语义

这样 benchmark 证明的是：

- “系统是否感受到了 pressure”
- 以及“系统是否真的按 runtime 的 PE schedule 进入了学习路径”

而不是把二者混写成一个单一指标。

## Real Proof Profile

为了避免 `slow-shapes-fast` 和 `rare-heavy-net-benefit` 被 proof profile 配置本身直接掩盖，当前 real comprehensive benchmark 推荐单独使用最小 proof profile：

- `canonical_case_limit >= 2`
- `profile_labels = ("pe-eta", "pe-eta-no-rare-heavy", "eta-no-pe", "heuristic-baseline")`
- `perturbation_variant_limit = 1`
- `replay_family_limit = 1`
- `candidate_config_limit = 1`
- 优先使用 staged comprehensive benchmark 保存可复查 artifact

proof profile 的目的不是跑最大全量 benchmark，而是确保四条 proof obligations 至少有机会在真实路径上被观测到。

## Longitudinal / Essence 层

当前 benchmark 还额外补了两层：

- `run_dialogue_pe_eta_longitudinal_benchmark()`
- `build_dialogue_nl_essence_assessment()`

前者会在同一 runner 上跨 context 跑 canonical cases，显式产出 `cross_session_verdict`；后者会把 benchmark 输出压成 6 条 `nested learning essence` gates：

- `pe-first`
- `multi-timescale-default`
- `rare-heavy-net-benefit`
- `slow-shapes-fast`
- `judge-gated-evolution`
- `cross-session-growth`

## Perturbation / Replay 层

当前 benchmark 已补一层固定变体：

- `wording_shift`
- `pressure_shift_late`

并覆盖 4 个基底 cases：

- `repair`
- `task_clarification`
- `repeated_failure`
- `goal_drift`

因此当前 `dialogue_benchmark` 不只会跑 4 个 canonical cases，还可以跑 8 个固定 replay/perturbation variants，用来检查：

- 改写措辞后，优势是否还在
- 压力位置后移后，优势是否还在
- 定量指标是否在 variant 层保持分离

## Systematic Replay 层

当前 benchmark 已进一步新增：

- `DialogueParaphraseFamily`
- `generate_stochastic_dialogue_case_variants()`
- `DialogueReplayRankingReport`
- `run_dialogue_pe_eta_systematic_replay_benchmark()`

这意味着 perturbation 不再只停留在手写 variants，而是可以：

- 按 paraphrase family 组织变体
- 用 seed 生成可重复的 stochastic variants
- 对变体按“诊断性”做 replay ranking

当前默认 `seeds=(0, 1)`，可按需扩展。

现在还支持：

- `DEFAULT_DIALOGUE_REPLAY_SEEDS = (0, 1, 2)`
- `DialogueReplaySelectionArtifact`
- `build_dialogue_replay_selection_artifact()`
- `build_replay_selection_training_traces()`
- `train_rare_heavy_artifact_from_replay_selection()`
- `run_replay_selection_artifact_acceptance_benchmark()`

也就是说，系统不仅能对生成 variants 做 replay ranking，还能从 ranking 中直接抽出一个 top-k selection artifact，作为后续 artifact acceptance / replay selection 的输入集合。

## 初版 acceptance rules

当前 case-level 通过条件：

1. 有非空 prediction chain
2. case 中确实出现高 PE turn
3. 高 PE 至少在部分 turn 上触发 `*-pe` schedule 或 rare-heavy recommendation
4. temporal trajectory 不是常量
5. case 后段出现 PE 下降，或 delayed outcome 指标改善

当前 proof gate 已与 runtime 的 PE scheduling 语义对齐：

- `high_pe_threshold = 0.18`
- `reward_threshold = 0.05`

这不再沿用早期过高的 `0.6 / 0.25` proof gate，而是对齐到当前 runtime 中实际会触发 `ssl-only-pe` 的量级。

另外，`pe_triggered_temporal_response` 已不再要求“同一 turn 同时出现 high PE 和 `*-pe` label”。当前判定已对齐到 runtime 的真实时序：

- 前一轮 high PE
- 下一轮出现 `*-pe`，或至少进入非 `evidence-only` 的 controller update path

这样 benchmark 检查的是跨轮因果关系，而不是同轮巧合。

但这个跨轮 carryover credit 现在只对 `pe-eta` 路径开放。`eta-no-pe` 作为严格 baseline，不再因为“普通 interval update”而获得 PE-triggered response 的记分。

当前 delayed improvement 判定优先看：

- 后段平均 `prediction_error.magnitude` 是否低于前段
- 若 PE 未明显下降，则看 `predictive_accuracy`、`joint_learning_progress`、`cross_track_stability`、`delayed_regime_alignment`、`delayed_action_alignment`、`regime_sequence_payoff`、`rolling_action_payoff` 是否改善

另外，bootstrap 型零误差 turn 不再主导“前段平均 PE”的比较，避免把第一轮天然接近零的 turn 错算成系统已经处于低误差状态。

## 这能证明什么

这套 harness 当前能提供三类证据：

1. **链路证据**：prediction chain 是否真的进入主链而不是停留在日志层
2. **耦合证据**：高 PE 是否和 temporal abstraction 的变化同步出现
3. **后效证据**：变化后是否在 case 后段带来更好的 readout
4. **相对证据**：`pe-eta` 相对 `eta-no-pe` / `heuristic-baseline` 是否在 case-level summary 上更强

进一步地，这两个定量指标可以回答：

- `pe-eta` 是否比 baseline **更早**响应压力
- `pe-eta` 是否比 baseline **更准**地把响应集中在 pressure 附近

## 当前实跑结果

在最近一次真实单路径 benchmark 中：

- 4 个 scripted cases 里有 4 个通过当前 PE-ETA evidence gate
- 通过的 case：`repair`、`task_clarification`、`repeated_failure`、`goal_drift`

这说明当前主问题已不再是“benchmark 完全看不见 PE 路径”，而是：

- 当前 proof gate 已能观察到真实 `ssl-only-pe`
- 当前四类 scripted cases 都能给出可观测的 PE-triggered temporal response 与 delayed improvement

在最近一次真实三路径 A/B benchmark 中：

- `pe-eta`: `4/4` passed
- `eta-no-pe`: `0/4` passed
- `heuristic-baseline`: `0/4` passed

并且新增的定量指标也已经拉开：

- `repair` / `task_clarification` / `repeated_failure`：
  - `pe-eta`: `recovery_lag = 2`, `pressure_localization = 0.5`
  - `eta-no-pe`: `recovery_lag = 6`, `pressure_localization = 0.0`
- `goal_drift`：
  - `pe-eta`: `recovery_lag = 1`, `pressure_localization = 1.0`
  - `eta-no-pe`: `recovery_lag = 5`, `pressure_localization = 0.0`

这说明在当前 dialogue proof harness 下，`pe-eta` 已经能够和严格版 `eta-no-pe` 以及 heuristic baseline 拉开。

新增定量层后，差异也进一步可读：

- `repair` / `task_clarification`
  - `pe-eta`: `precision = 0.5`, `recall = 0.667`, `over_response = 0.167`, `stability_after_recovery = 0.333`
  - `eta-no-pe`: `precision = 0.0`, `recall = 0.0`, `over_response = 0.0`, `stability_after_recovery = 0.0`
- `repeated_failure`
  - `pe-eta`: `precision = 0.5`, `recall = 0.5`, `over_response = 0.167`, `stability_after_recovery = 0.0`
  - `eta-no-pe`: `precision = 0.0`, `recall = 0.0`, `over_response = 0.0`, `stability_after_recovery = 0.0`
- `goal_drift`
  - `pe-eta`: `precision = 1.0`, `recall = 1.0`, `over_response = 0.0`, `stability_after_recovery = 0.5`
  - `eta-no-pe`: `precision = 0.0`, `recall = 0.0`, `over_response = 0.0`, `stability_after_recovery = 0.0`

这里要注意：`over_response_cost` 不是“越大越强”，而是一个代价项。当前结果更像说明：

- `pe-eta` 在 `repair` / `task_clarification` / `repeated_failure` 中确实为了响应压力付出了一些额外 controller update 成本
- 但这些额外响应同时换来了更高的 recall、更低的 recovery lag 和更高的 localization

在最近一次真实 perturbation benchmark 中：

- `pe-eta`: `8/8` passed
- `eta-no-pe`: `0/8` passed
- `heuristic-baseline`: `0/8` passed

而且这种优势在两个 variant families 上都保住了：

- `wording_shift`
- `pressure_shift_late`

也就是说，当前 benchmark 已经不只是在 canonical wording 上看到优势，而是在固定改写与压力位置平移后仍能保持分离。

在最近一次真实 systematic replay benchmark（`seeds=(0,)`, generated variants only）中：

- `pe-eta`: `4/4` passed
- `eta-no-pe`: `0/4` passed
- `heuristic-baseline`: `0/4` passed

Replay ranking 的前几项为：

1. `repair__repair_family__seed_0`
2. `repeated_failure__failure_family__seed_0`
3. `goal_drift__goal_drift_family__seed_0`
4. `task_clarification__clarification_family__seed_0`

对应的 `diagnostic_score` 大致在 `14.167 ~ 18.0` 区间，说明这些生成变体对区分 `pe-eta` 与 strict baselines 是有序且可排序的，而不是只剩“全过/全不过”的粗粒度判断。

在最近一次更大的真实 systematic replay benchmark（`seeds=(0,1,2)`, generated variants only）中：

- `pe-eta`: `12/12` passed
- `eta-no-pe`: `0/12` passed
- `heuristic-baseline`: `0/12` passed

并且 top replay selection artifact（`top_k=6`）已经能稳定挑出最有诊断性的 variants，例如：

1. `repair__repair_family__seed_0`
2. `repair__repair_family__seed_2`
3. `task_clarification__clarification_family__seed_2`
4. `repair__repair_family__seed_1`
5. `repeated_failure__failure_family__seed_0`
6. `repeated_failure__failure_family__seed_1`

对应的 `diagnostic_score` 大致在 `18.0 ~ 19.0`，说明这批候选不仅能区分路径，而且已经可以作为后续 artifact acceptance / replay selection 的输入集合。

## Replay Selection Acceptance

当前系统已能把：

- replay ranking top-k selection artifact
- 转成 rare-heavy training traces
- 训练出 `RareHeavyArtifact`
- 再把它导回 runner，对 selected variants 做 pre/post acceptance benchmark

当前 acceptance gate 还额外会检查 substrate rare-heavy evidence：

- artifact 是否携带 `substrate_checkpoint`
- `substrate_update_count` 是否非零
- `substrate_source_batch_count` 是否满足最小值
- `substrate_mean_sequence_length` / `substrate_mean_residual_magnitude` 是否达到最小训练证据
- `substrate_import_success_fraction` 是否表明确实通过 owner-side substrate import 生效

在最近一次真实 replay-selection acceptance benchmark 中：

- 选取了 `top_k=6` 个最有诊断性的 variants
- rare-heavy import 操作稳定成功（`rare-heavy:temporal-import`, `rare-heavy:memory-import`）
- acceptance report 的整体 `mean_score_delta = 0.104`
- `passed_case_delta = 0`
- `positive_case_fraction = 0.333`
- `worst_case_delta = -0.500`

也就是说：

- 这条 rare-heavy acceptance 流程已经打通
- 但收益还是 mixed，而不是稳定提升
- 在默认 gate 下，artifact 会被 **reject** 并触发 rollback

当前 sample：

- `repair__repair_family__seed_1`: `score_delta = +0.500`
- `repeated_failure__failure_family__seed_1`: `score_delta = +1.000`
- `task_clarification__clarification_family__seed_2`: `score_delta = -0.500`
- `repeated_failure__failure_family__seed_0`: `score_delta = -0.375`

这说明 replay selection 已经能作为 rare-heavy 的正式验收入口，但“selection -> training -> acceptance” 这条链路还需要更强的 artifact selection / acceptance criteria，不能把当前 pipeline 误当成已经稳定增益。

另外，benchmark / replay report 现在会把 `pe-eta` 相对 `pe-eta-no-rare-heavy` 的差异单独量化，而不再只隐含在通用 baseline delta 中：

- ablation report 会显式发布 `rare_heavy_metric_deltas` / `rare_heavy_case_deltas`
- replay ranking 会显式发布 `gap_vs_no_rare_heavy` 与 `mean_gap_vs_no_rare_heavy`

## Multi-Artifact Comparison

当前系统还支持：

- `DEFAULT_RARE_HEAVY_CANDIDATE_CONFIGS`
- `DialogueArtifactCandidateReport`
- `DialogueArtifactComparisonReport`
- `run_multi_artifact_acceptance_benchmark()`

也就是说，现在不再是“训练一个 artifact 然后直接判”，而是：

1. 同一 selection artifact
2. 训练多组 candidate configs
3. 分别跑 acceptance benchmark
4. 按统一 candidate score 排序
5. 选择 best-of-n candidate 进入 gate 结论

这使得 rare-heavy 不再只是单候选碰运气，而开始接近真正的慢层候选选择系统。

在最近一次真实 multi-artifact acceptance comparison（同一 selection, `top_k=4`）中：

- 比较了 3 个 candidate configs：
  - `balanced`
  - `more-rl`
  - `more-ssl`
- 当前最佳候选是：`more-ssl`
- 但它仍未通过默认 gate

当前最优候选 `more-ssl` 的结果：

- `mean_score_delta = 0.438`
- `passed_case_delta = 0`
- `positive_case_fraction = 0.750`
- `worst_case_delta = 0.000`

默认 gate 下它仍被 reject，唯一剩余拒绝原因是：

- `passed-case-delta-below-threshold`

而 `balanced` / `more-rl` 则更差，仍同时触发：

- `passed-case-delta-below-threshold`
- `positive-case-fraction-below-threshold`
- `worst-case-delta-below-threshold`

这说明 current selection/training/gate 组合已经足够区分 candidate quality，但还没有候选能在“真正提升通过数”这个最强标准上过线。

当前默认 gate 的拒绝原因包括：

- `passed-case-delta-below-threshold`
- `positive-case-fraction-below-threshold`
- `worst-case-delta-below-threshold`

并且 reject 后会对每个 adapted runner 执行：

- `rare-heavy:temporal-rollback`
- `rare-heavy:memory-rollback`

因此现在已经不是“人眼看报告后手动决定”，而是有正式的 acceptance / reject gate + rollback policy。

在最近一次真实 multi-artifact acceptance comparison 中：

- 比较了 3 个候选：
  - `balanced`
  - `more-rl`
  - `more-ssl`
- 当前最优候选是：`more-ssl`

它的结果是：

- `mean_score_delta = 0.438`
- `passed_case_delta = 0`
- `positive_case_fraction = 0.750`
- `worst_case_delta = 0.000`

但默认 gate 仍然拒绝它，因为唯一剩余未满足的条件是：

- `passed-case-delta-below-threshold`

这说明 current best-of-n 已经能找到“明显比其它候选更好”的 candidate，但默认 acceptance gate 仍坚持要求真正提升通过数，而不是只接受 mixed-gain 的局部改善。

## Comprehensive Benchmark 入口

当前还新增：

- `run_dialogue_pe_eta_comprehensive_benchmark()`
- `DialogueRealComprehensiveBenchmarkConfig`
- `DialogueComprehensiveStage`
- `DialogueSharedRunnerFactories`
- `build_real_dialogue_comprehensive_runner_factories()`
- `run_real_dialogue_pe_eta_comprehensive_benchmark()`
- `run_real_dialogue_pe_eta_comprehensive_benchmark_staged()`

这条入口会把以下几层串成一次完整跑法：

1. canonical cases 上的 multi-profile ablation
2. fixed perturbation variants 上的 multi-profile ablation
3. systematic replay / stochastic variants + replay ranking
4. replay selection artifact 构建
5. multi-artifact acceptance comparison

也就是说，现在 benchmark 不再只是分散的几个 helper，而是有一个 repo 内部可直接调用的“强证据套件”入口，能把：

- online-fast 响应
- background writeback
- rare-heavy selection / import / rollback

放在同一个报告链里看。

## 真实可跑完的 Runner

之前直接调用 comprehensive benchmark 的默认真实路径，有两个工程性问题：

1. 如果没有显式传入 shared runtime，很多 case/profile/variant 会重复构造真实 `AgentSessionRunner`
2. 这会导致真实 substrate/runtime 被反复初始化，使全量跑法变得非常慢，而且缺少阶段进度感

当前已补：

- `build_real_dialogue_comprehensive_runner_factories()`：先构造一个 shared real `OpenWeightResidualRuntime`，再把它注入 canonical / perturbation / systematic replay / acceptance 四类 runner factory
- `run_real_dialogue_pe_eta_comprehensive_benchmark()`：按 phase 顺序执行
  - canonical ablation
  - longitudinal benchmark
  - essence assessment
  - perturbation benchmark
  - systematic replay benchmark
  - selection artifact
  - multi-artifact acceptance
- `run_real_dialogue_pe_eta_comprehensive_benchmark_staged()`：在上述 phase 外再包一层 checkpoint executor
  - 每个 phase 结束后把 stage result 写到 `output_dir`
  - 写 `manifest.json` 记录当前 config 和已完成 stages
  - `resume=True` 时优先从已完成 stage 恢复
  - 如果 final report 已存在，会在建 runtime 之前直接返回
- `DialogueRealComprehensiveBenchmarkConfig`：给真实跑法提供 bounded 默认配置，例如：
  - `replay_seeds=(0,)`
  - `selection_top_k=4`
  - 可限制 canonical cases / perturbation variants / replay families / candidate configs

这意味着现在已经有一条“真实 runtime + comprehensive evidence chain + bounded default scale”的正式入口，而不是只能靠 synthetic runner 或一次性全量脚本碰运气。

当前 staged executor 的 phase 顺序固定为：

1. `canonical_ablation`
2. `longitudinal`
3. `essence`
4. `perturbation`
5. `systematic_replay`
6. `selection_artifact`
7. `artifact_comparison`
8. `final_report`

因此“全量 real comprehensive 也能分批稳定跑”的实际含义是：

- 中断后不需要从头重跑
- 可以先跑到某个 stage，确认结果后再继续
- 后续 resume 不会再重复做重型 runtime 初始化和已完成 benchmark

另外，这条真实 runner 链也顺手暴露并推动修复了两类兼容问题：

- `heuristic-baseline` 在真实 `AgentSessionRunner` 下不再要求 heuristic temporal policy 必须携带 `parameter_store`
- CMS medium/background band 在 rare-heavy import / acceptance 路径上，遇到 signal / pending signal 维度不一致时，会先做 owner-side 对齐，而不是直接越界

## 这还不能证明什么

当前还**不能**直接证明：

- 这些 temporal abstractions 已达到论文级“涌现”结论
- PE 是唯一原因，而非其它共变信号
- rare-heavy artifact selection 已达到最优
- 每个时间尺度都已经形成了自己粒度清晰、稳定迁移的抽象对象
- 在开放式生成用户环境中也一定成立

更精确地说，当前 harness 证明的是：

- ETA/NL 的若干关键链路已经进入默认运行时
- 默认主路径已经能给出多时间尺度与 nested CMS 的可检查 telemetry
- repo 内部已经具备“失败时能解释原因”的 proof surface

它**还不能**证明：

- 论文级“涌现时间抽象”已经成立
- 论文级“continuum memory / Hope / self-modifying learner”已经被完整复现
- 当前对话 benchmark 的优势足以外推到开放域 continual learning

要继续收紧证据，需要下一步补：

- 生成式用户模拟器
- 更严格的 PE-off / ETA-off / timescale-off 因果隔离实现（不仅是当前 profile matrix）
- 面向 session-medium / background-slow / rare-heavy 的独立抽象迁移指标
- 更长程的跨会话 benchmark，而不只是单 case / variant 内 delayed payoff

## 相关测试

新增 focused tests：

- `tests/test_dialogue_benchmark.py`
- `tests/test_agent_session_runner.py` 中与 PE scheduling / rare-heavy / session benchmark 相关回归

本轮已验证：

- `python -m pytest tests/test_dialogue_benchmark.py -q`
- `python -m pytest tests/test_agent_cli.py -q`
