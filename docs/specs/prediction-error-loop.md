# Prediction Error 主链 Spec

> Status: draft
> Last updated: 2026-04-22
> 对应需求: R-PE

## 要解决的问题

如何把“预测 -> 实际结果 -> prediction error”从辅助日志提升为正式运行时主链，使它成为后续 credit / memory / temporal / regime / reflection 的共同学习原语？

## 关键不变量

- prediction error / LSS 是原始学习信号，不是可选诊断信息
- 系统必须显式发布 prediction chain，而不是只在下游隐式近似
- evaluation 是 prediction error 的 readout / gate 层，不是学习源头
- credit 是 prediction error 的聚合 / 审计层，不是学习源头
- prediction error 必须以 machine-readable 多维结构对外发布，而不是只剩一条文本描述

## 工程挑战

- 定义最小但稳定的 prediction chain 公共契约
- 保持 prediction error 的唯一 owner，避免各 consumer 自己重建 outcome mismatch
- 处理首轮 bootstrap 与跨轮 carryover，不制造同轮自因果闭环
- 让 downstream owner 能直接消费 task / relationship / regime / action 四维误差，而不需要重新解析文本

## 算法候选

来自 `docs/next_gen_emogpt.md`：

- **R-PE**：prediction error 是原始学习信号，evaluation/credit/reward 都是其下游读数或聚合
- **NL / LSS**：local surprise signal 是对预测与现实偏差的局部刻画
- **ETA**：时间抽象控制和 delayed outcome 学习应围绕 latent action 的后果误差展开，而不是只看 token 级局部损失

## 接口契约

**消费的输入**：

- `substrate` 快照：提供 turn-level semantic feature surface
- `evaluation` 快照：提供 family-level当前 readout，辅助构造 next-turn prediction
- `dual_track` 快照：提供 world/self tension 与 track-level state
- `regime` 快照：提供当前 regime 效果与稳定性线索

**产出的输出**：

- `prediction_error` 快照：`PredictionErrorSnapshot`
  - `evaluated_prediction`
  - `actual_outcome`
  - `next_prediction`
  - `error`
  - `turn_index`
  - `bootstrap`
  - `pe_decomposition`（optional, Phase 1.B）：`PEDecomposition` 或 `None`

**当前实现口径**：

- 正式 owner 为 `PredictionErrorModule`
- 公共 `error` 当前固定四个维度：
  - `task_error`
  - `relationship_error`
  - `regime_error`
  - `action_error`
- 聚合读数最小固定为：
  - `magnitude`
  - `signed_reward`
- 当前 owner 内部已收敛为单一 outcome mapper/head：prediction、actual outcome 与 error weighting 都在 `prediction_error` owner 内完成；consumer 不应重建这三段语义
- 当前 `magnitude` / `signed_reward` 不再是简单的四维平权 L1/平均，而是结合 prediction confidence 与 axis expectation strength 的 owner-side calibrated readout
- 当前 `evaluation` 只发布 PE-owner readout（如 `prediction_error_magnitude`、`prediction_error_reward`、`predictive_accuracy`），不再推导第二套 PE 语义
- 当前 proof harness 允许显式区分两层含义：一层是 **PE publication/readout**（slot + evaluation evidence 仍存在），另一层是 **PE primary dominance**（是否直接主导 joint-loop schedule 与 RL reward）。`pe-eta-pe-readout-only` 用于只保留前者
- `bootstrap=True` 表示当前 turn 尚无可结算的上一轮 prediction；下游不应把这类快照当作真实 learning evidence
- live runtime 中，部分 consumer 会把 `prediction_error` 当作“上一轮结算出的 carryover signal”，以维持单轮 DAG 和 owner 边界

### Curiosity-Critic PE 分解（Phase 1.B, running-stats）

来源：Aubret et al., "Curiosity-Critic: Cumulative Prediction Error Improvement"（`arXiv:2604.18701`）。核心命题：把瞬时 PE 替换为 PE 的"可改进部分"，把 epistemic（可学）与 aleatoric（不可学）分离，避免噪声驱动 memory writes / regime switching / metacontroller 行为。

落点：`vz-cognition/prediction/error.py` 内 owner-internal `_PECriticHead` + `_AxisRunningStats`；新 frozen `PEDecomposition` dataclass；`PredictionErrorSnapshot.pe_decomposition: PEDecomposition | None`。

机制（**lightweight, 不训练 critic head**）：

- 每个 (axis, bucket_key) 维护一个 EMA mean / EMA variance；bucket_key = `regime:<regime_id>` / `segment:<segment_id>` / `action:<abstract_action_id>` / `default`。
- 每轮 `compute_error` 后，对每条轴 `|axis_error|` 更新对应 bucket。
- aleatoric_magnitude := `sqrt(EMA_variance)`，clamp 到 `[0, 1]`，代表噪声底；epistemic_magnitude := `max(0, |error| − aleatoric)` 的轴聚合，clamp 到 `[0, 1]`，代表"系统能学下去的部分"。
- per_axis 列出每条轴的 (axis_name, aleatoric, epistemic)。
- decay 默认 `0.9`，由 `PredictionErrorModule(pe_critic_decay=...)` 注入，避免硬编码。

接入点：

- `PredictionErrorSnapshot.pe_decomposition` 在 bootstrap turn 时为 `None`，正常 turn 由 owner 内部 `_PECriticHead.update(...)` 填充；现有 consumer 仍只读 `error.magnitude` / 轴 error / `signed_reward`，向后兼容。
- `evaluation/backbone.py::_prediction_error_scores` 新增两个 metric：`pe_aleatoric_magnitude`、`pe_epistemic_magnitude`，**严格 report-only**，不进入任何 acceptance gate；目的是避免把"分离"反过来训练成第二套 reward。
- `vz-memory/memory/store.py` 在 PE 写入路径里把 `epistemic_magnitude` / `aleatoric_magnitude` 直接写入 owner-internal `MemoryAttributeReadout`（Phase 1.C），让陪伴向"哪条 PE 是可学的"成为可观察 readout。

**Phase 2.B uplift（不在本次实现）**：在 PE owner 内加 learned critic head（小型回归头：substrate feature + action_context → 预测 PE 期望，improvement-PE = actual − critic-prediction），需要 checkpoint / capacity cap / rollback evidence，纳入 `ModificationGate` 准入，已登记于 `docs/known-debts.md`。

**快照 schema**：见 `docs/DATA_CONTRACT.md` 3.9 节

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 契约式运行时 | 通过独立 slot 发布正式 prediction chain |
| 依赖 | 双轨学习 | task / relationship 维度误差需要双轨状态 |
| 依赖 | 认知 Regime | regime stability / action payoff 的一部分来自 regime owner 发布状态 |
| 依赖 | Emergent Action Abstraction | 接收 `temporal_abstraction.closed_segments` 与可观察 `EnvironmentOutcome` 字段作为 action context，不新增第二 PE owner |
| 被依赖 | 信用分配与自修改 | credit 是 prediction error 的聚合与审计层 |
| 被依赖 | 连续记忆系统 | memory owner 用 PE 调整写入、promotion threshold 和 retrieval facets |
| 被依赖 | 时间抽象与内部控制 | temporal owner 用 PE 调节 controller update 与 schedule 选择 |
| 被依赖 | 评估体系 | evaluation 把 PE 作为结构化 readout 和 benchmark 证据输入 |
| 被依赖 | 认知 Regime | regime owner 用 delayed / per-dimension PE 更新 historical effectiveness |
| 被依赖 | 慢反思路径 | reflection 将 PE 作为 tensions、lessons 和 policy consolidation 的正式输入 |

## 变更日志

- 2026-05-06: Phase 1.B 上线 owner-internal Curiosity-Critic running-stats 分解（`PEDecomposition` + `_PECriticHead`）；`PredictionErrorSnapshot.pe_decomposition` 为 optional 字段，bootstrap 时为 `None`；`evaluation` 新增 `pe_aleatoric_magnitude` / `pe_epistemic_magnitude` 两个 report-only metric。Phase 2.B learned critic head 登记为后续 uplift。
- 2026-05-02: 重写对 Emergent Action Abstraction（`docs/specs/emergent-action-abstraction.md`）的依赖口径：PE 消费 temporal segment closure 与可观察 outcome context，不新增 trace owner 或 learning primitive
- 2026-04-22: 补充 `pe-eta-pe-readout-only` proof 口径，明确区分 PE publication/readout 与 PE primary dominance
- 2026-04-22: 当前实现口径补充单一 owner-side mapper/head、confidence-aware calibrated error weighting，以及 evaluation 只发布 PE-owner readout 的边界
- 2026-04-20: 初始版本。将 `prediction_error` 从 credit/evaluation 的上游设计原则提升为独立能力域 spec，固定主链契约 `evaluated_prediction -> actual_outcome -> next_prediction -> error`
