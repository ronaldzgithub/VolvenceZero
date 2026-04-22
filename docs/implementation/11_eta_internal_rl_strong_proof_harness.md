# ETA Internal RL Strong-Proof Harness

> Status: draft
> Last updated: 2026-04-21
> Scope: paper-like proof path for ETA internal RL

## Goal

把 ETA 的 internal RL 从“框架近似”推进到 repo 内部可重复、可验收的强证明路径，回答四个问题：

- sparse-reward hierarchical task 上是否真的得到收益
- 抽象动作 family 是否被发现并复用
- held-out 组合重排时是否还能完成任务
- delayed reward 是否能回指到正确的 abstract-action window

## 模块布局

- `volvence_zero/internal_rl/environment.py`
  - 新增 `InternalRLProofEpisode`
  - 新增 `InternalRLProofSubgoal`
  - 新增 `InternalRLDelayedCreditAssignment`
  - proof mode 下用结构化 subgoal / terminal reward 代替默认 dense reward
- `volvence_zero/internal_rl/proof_environment.py`
  - 提供 `MiniHierarchicalEnvironment`
  - 提供 `HierarchicalLocation / HierarchicalTransition / HierarchicalRouteSpec`
  - 提供 `reset(route) / observe(state) / step(state, target_id)` 交互 API
  - case 不再直接手写拼 proof episode，而是由环境对象从 route spec 运行 episode 后生成
- `volvence_zero/internal_rl/sandbox.py`
  - proof mode 下 causal observation 优先使用 `residual_sequence` 摘要
  - rollout 记录 `raw_reward`、proof annotations、delayed credit assignments
  - optimization 对 proof rollout 使用 delayed-return path
- `volvence_zero/agent/eta_proof_benchmark.py`
  - 定义 ETA proof cases / profiles / benchmark reports
  - 提供 backend robustness 比较
  - 提供 ETA strong-proof assessment / acceptance

## 与现有 benchmark 的关系

这条 harness **不是** `dialogue_benchmark` 的替代品。

两者职责分离：

- `dialogue_benchmark`
  - 证明 PE 是否驱动 temporal response 与 delayed improvement
  - 侧重系统默认主链
- `eta_proof_benchmark`
  - 证明 internal RL 是否在抽象动作层解决 hierarchical sparse reward
  - 侧重 ETA 论文式命题

NL essence acceptance 也继续保留；它不负责回答 ETA internal RL 的强证明问题。

## 默认 profile

- `full-internal-rl`
- `full-no-optimize`
- `full-no-replacement`
- `learned-lite-causal`
- `noop-backend`

另外提供 backend robustness 比较：

- `trace`
- `synthetic-open-weight`

## 默认 acceptance gates

- `sparse-reward-success`
- `abstract-action-reuse`
- `heldout-composition`
- `credit-alignment`
- `policy-update-evidence`
- `statistical-batch-evidence`
- `backend-robustness`

这些 gate 与 `DialogueNLEssenceAcceptanceConfig` 分离，避免把 ETA 论文命题和 NL 默认主链验收混在一起。

当前默认 gate 的解释已改成**matched controls + mechanism evidence**：

- `full-no-optimize`
  - 回答“保留 full controller 与 causal-binary replacement，但不做 RL 更新会怎样”
- `full-no-replacement`
  - 回答“保留 full controller 与 RL 更新，但不把 causal z-policy override 真正写进 temporal control 会怎样”
- `learned-lite-causal`
  - 回答“保留 causal replacement 与 RL 更新，但只给弱控制器容量会怎样”
- `noop-backend`
  - 回答“没有真实 residual intervention effect 会怎样”

因此默认 acceptance 不再主要依赖一个混合 negative control，而是要求：

- full internal RL 在 primary held-out 指标上相对 matched controls 维持优势
- delayed credit alignment 和 abstract-action reuse 不能只是 composite score 的副产品
- 训练期间必须有可观测的 policy update evidence，而不是只看最终 success

### 当前进一步收紧

当前 acceptance 还进一步收紧为：

- 每个 gate 都显式记录自己的 **best competing control**
- `sparse-reward-success` / `abstract-action-reuse` / `heldout-composition` / `credit-alignment` 不再只和部分 control 比，而是对照完整 matched-control 池：
  - `full-no-optimize`
  - `full-no-replacement`
  - `learned-lite-causal`
  - `noop-backend`
- report 会把 raw mechanism metrics 和 best-control label 一起发布，避免只靠 composite score 解释胜负
- report 现在还会显式发布 batch/statistical evidence：
  - `rollout_batch_count`
  - `mean_rollouts_per_update`
  - `training_transition_count`
  - `mean_parameter_change_norm`
  - `mean_value_loss`
  - `mean_replacement_effect_delta`
  - `heldout_strong_success_std`

这也意味着默认 acceptance 口径比之前更硬，某些过去可过的 run 现在可能 fail closed；这是为了把“模块存在”继续推进到“机制识别”。

### 当前 stronger ETA obligations

默认 proof harness 现在把以下命题视为 first-class obligations：

- `multi-rollout statistical evidence`
  - 默认训练更新应由 batch rollout 驱动，而不是单条 trajectory
- `true policy-update evidence`
  - 不能只看 `parameters_changed`；还要看 parameter-change norm 和 value/replacement 指标
- `causal replacement intervention evidence`
  - replacement 必须带来可报告的 `replacement_effect_delta`
- `bounded heldout variance`
  - held-out success 不能只靠单次高分，需发布 `heldout_strong_success_std`
- `batch-level credit alignment`
  - delayed credit alignment 必须和 batch rollout 一起解释，而不是只给 case-level pass/fail

## 默认 case library

默认 case library 不再只是 2-step toy episodes，而是通过显式 miniature hierarchical environment API 生成，更接近层级任务环境：

- 一个默认环境 `eta-mini-hierarchy`
- 显式 location graph：`entry`、`hub`、`alpha`、`beta`、`gamma`、`delta`、`epsilon`
- 显式 transition graph：`corridor` / `branch` / `loop` / `return`
- route spec 驱动 3 个 `train`、2 个 `eval`、2 个 `heldout` cases
- heldout route 现在包含显式 loop / branch 组合，不再只是 case factory 里手写的 subgoal tuple
- benchmark 生成 case 时会先 `reset(route)`，再按 route 逐步 `step(...)`，用真实 episode 交互生成 `route_signature` 与 branch depth

每个 case 都显式带：

- `environment_id`
- `route_signature`
- `branch_depth`
- 多个 distractors
- 每个 subgoal 自己的 `completion_threshold` / `min_persistence`
- observation / effect / control 权重

这样默认 proof task 更像“有图结构、有分支、有 loop 与 held-out 重组”的层级 episode，而不是单纯的 signature toy match。 

## 测试策略

- `tests/test_phase2_eta_nl.py`
  - 保持对 `internal_rl` / joint-loop 基础路径的回归覆盖
- `tests/test_eta_proof_benchmark.py`
  - proof rollout delayed credit
  - benchmark profile/report shape
  - backend robustness report
  - acceptance fail-closed / pass-path

## 当前边界

- 这是 repo-native paper proof path，不等于完整复现论文实验环境
- 当前 backend robustness 先覆盖 `trace` 和 `synthetic-open-weight`
- 当前 held-out composition 是结构化 subgoal 重排，不是完整 MuJoCo / gridworld 级环境

但它已经把最关键的命题拆成了可执行、可回归、可 fail-closed 的工程证据面。 

## Paper-Suite Uplift

为了把 ETA strong-proof 从“单次 mechanism report”推进到更接近 paper-grade empirical closure，repo 现在补了 paper-suite 包装层：

- `build_eta_proof_paper_suite_manifest()`
- `run_eta_internal_rl_paper_suite()`
- `export_eta_internal_rl_paper_suite_artifact_bundle()`

### Frozen tiers

当前 ETA proof 也显式区分 3 个 tier：

- `ci-smoke`
  - 单次快速回归
- `paper-suite-small`
  - fixed seeds + repeated runs + aggregate summaries
- `paper-suite-full`
  - 更完整重复与 release-grade artifact pack

### What this layer adds

paper-suite 层不会替代原有：

- `run_eta_internal_rl_proof_benchmark()`
- `run_eta_internal_rl_backend_robustness_benchmark()`
- `build_eta_internal_rl_assessment()`

而是把它们封装为一个冻结、可重复、可导出的实验单元。当前 suite 级 primary metrics 包括：

- `heldout_terminal_success_rate`
- `heldout_strong_success_rate`
- `heldout_family_reuse_rate`
- `heldout_credit_alignment`
- `strong_success_gap_vs_best_control`
- `backend_success_gap`

secondary metrics 包括：

- `assessment_pass_fraction`
- `policy_update_rate`
- `heldout_subgoal_completion_rate`

### Artifact bundle

当前 ETA proof paper-suite bundle 至少导出：

- `paper_suite_manifest.json`
- `paper_suite_provenance.json`
- `paper_suite_run_summaries.json`
- `paper_suite_aggregate.json`
- `reference_benchmark_report.json`
- `reference_backend_report.json`
- `reference_assessment.json`

### CI / script entrypoints

新增脚本入口：

- `scripts/run_eta_paper_suite.sh`

新增 workflow tiers：

- `.github/workflows/paper-suite-nightly.yml`
- `.github/workflows/paper-suite-release.yml`

这样 ETA proof 不再只是 tests 中的 isolated report，而进入和 dialogue proof 相同的 release discipline：

- smoke regression
- nightly repeated-run evidence
- release-grade artifact bundle
