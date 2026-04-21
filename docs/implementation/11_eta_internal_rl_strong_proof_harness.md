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
- `metacontroller-no-rl`
- `noop-backend`
- `learned-lite-baseline`

另外提供 backend robustness 比较：

- `trace`
- `synthetic-open-weight`

## 默认 acceptance gates

- `sparse-reward-success`
- `abstract-action-reuse`
- `heldout-composition`
- `credit-alignment`
- `backend-robustness`

这些 gate 与 `DialogueNLEssenceAcceptanceConfig` 分离，避免把 ETA 论文命题和 NL 默认主链验收混在一起。

当前默认 gate 的解释进一步收紧为两层：

- `metacontroller-no-rl`
  - 作为**结构参照**保留，用来回答“只有 discovered structure、没有 internal RL policy adaptation 时会怎样”
- `learned-lite-baseline` + `noop-backend`
  - 作为默认 acceptance 更接近的**负对照面**
  - 前者回答“弱控制器/弱抽象”会怎样
  - 后者回答“没有真实 residual intervention effect”会怎样

因此默认 `sparse-reward-success` 更偏向比较 full internal RL 相对负对照的强结论，而 `metacontroller-no-rl` 继续保留在 benchmark report 中作为 paper-style 结构参考线。

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
