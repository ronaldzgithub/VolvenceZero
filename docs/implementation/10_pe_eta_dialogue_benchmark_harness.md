# PE-ETA Dialogue Benchmark Harness

> Status: draft
> Last updated: 2026-04-20
> Scope: prove prediction-error-based temporal abstraction with scripted multi-turn dialogue cases

## Goal

补上一个内部“证明系统”，回答的不是“系统能不能聊天”，而是：

- turn 级是否显式产出 `predicted -> actual -> prediction_error`
- 高 PE turn 之后，是否出现 `*-pe` schedule、abstract action / regime / switch 行为变化
- 这些变化是否在 case 后段带来更低 PE 或更好的 delayed outcome

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

核心入口：

- `dialogue_proof_cases()`
- `run_dialogue_pe_eta_case()`
- `run_dialogue_pe_eta_benchmark()`
- `build_standard_dialogue_runner()`
- `default_dialogue_ablation_profiles()`
- `run_dialogue_pe_eta_ablation_benchmark()`
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

## 记录的证据

每个 benchmark turn 现在显式记录：

- prediction error：`magnitude` / `signed_reward` / `task_error` / `relationship_error` / `regime_error` / `action_error`
- temporal abstraction：`joint_schedule_action` / `active_abstract_action` / `switch_gate` / `action_family_version`
- regime：`active_regime`
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

## 这还不能证明什么

当前还**不能**直接证明：

- 这些 temporal abstractions 已达到论文级“涌现”结论
- PE 是唯一原因，而非其它共变信号
- rare-heavy artifact selection 已达到最优
- 在开放式生成用户环境中也一定成立

要继续收紧证据，需要下一步补：

- replay ranking / artifact acceptance benchmark
- multi-artifact selection
- 生成式用户模拟器
- 更严格的 PE-off / ETA-off 因果隔离实现（不仅是弱 profile）

## 相关测试

新增 focused tests：

- `tests/test_dialogue_benchmark.py`
- `tests/test_agent_session_runner.py` 中与 PE scheduling / rare-heavy / session benchmark 相关回归

本轮已验证：

- `python -m pytest tests/test_dialogue_benchmark.py -q`
- `python -m pytest tests/test_agent_session_runner.py -q -k "rare_heavy or pe_scheduled or multi_turn_rl_loop_produces_policy_changes or run_substrate_path_benchmark_collects_turn_metrics"`
