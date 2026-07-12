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

**evaluation→PE/credit 解耦 gate（`VZ_PE_EVALUATION_DECOUPLED`）**：

为了完全兑现 R-PE 第 1 不变量「evaluation 是 readout，不是学习源头」，新增一个显式可回滚 gate，控制 evaluation 是否进入 PE actual outcome 与 evaluation-derived credit：

| WiringLevel | actual outcome 的 `family_signals` | evaluation→credit | 退出条件 |
| --- | --- | --- | --- |
| `SHADOW`（默认，env 未设/falsey） | 来自 `_family_signals(EvaluationSnapshot)`（旧行为，逐字保留） | `derive_learning_evidence_credit_records` + counterfactual `propose_update` 写回照旧 | — |
| `ACTIVE`（`VZ_PE_EVALUATION_DECOUPLED` truthy） | `{}`（中性 0.5），actual outcome 仅由 substrate / dual-track / regime / external-outcome 驱动 | evaluation-derived credit 置空；counterfactual 不传 evaluation（学习写回不触发），仅保留 historical/readout 记录 | SHADOW 对比证据（`tests/test_pe_evaluation_credit_decoupling.py`）显示 ACTIVE 下 actual outcome 对 evaluation 内容不敏感且与 SHADOW 可测量地不同；在此基础上由 operator 决定切 ACTIVE |

- 实现位置：`vz-cognition/.../prediction/error.py::pe_evaluation_decoupled_active` + `_build_outcome_evidence`；`vz-runtime/.../integration/final_wiring.py` 的 credit 派生段复用同一 gate。
- 默认 SHADOW 保证此 packet 对既有 PE/credit 测试与运行时行为零影响（R15 可回滚）。
- 当前 proof harness 允许显式区分两层含义：一层是 **PE publication/readout**（slot + evaluation evidence 仍存在），另一层是 **PE primary dominance**（是否直接主导 joint-loop schedule 与 RL reward）。`pe-eta-pe-readout-only` 用于只保留前者
- `bootstrap=True` 表示当前 turn 尚无可结算的上一轮 prediction；下游不应把这类快照当作真实 learning evidence
- live runtime 中，部分 consumer 会把 `prediction_error` 当作“上一轮结算出的 carryover signal”，以维持单轮 DAG 和 owner 边界

### Curiosity-Critic PE 分解（Phase 1.B running-stats + Phase 2.B learned critic）

来源：Aubret et al., "Curiosity-Critic: Cumulative Prediction Error Improvement"（`arXiv:2604.18701`）。核心命题：把瞬时 PE 替换为 PE 的"可改进部分"，把 epistemic（可学）与 aleatoric（不可学）分离，避免噪声驱动 memory writes / regime switching / metacontroller 行为。

落点：`vz-cognition/prediction/error.py` 内 owner-internal `_PECriticHead` + `_AxisRunningStats` + learned contextual critic；新 frozen `PEDecomposition` dataclass；`PredictionErrorSnapshot.pe_decomposition: PEDecomposition | None`。

机制：

- 每个 (axis, bucket_key) 维护一个 EMA mean / EMA variance；bucket_key = `regime:<regime_id>` / `segment:<segment_id>` / `action:<abstract_action_id>` / `default`。
- 每轮 `compute_error` 后，对每条轴 `|axis_error|` 更新对应 bucket。
- aleatoric_magnitude := `sqrt(EMA_variance)`，clamp 到 `[0, 1]`，代表噪声底。
- Phase 2.B learned critic 读取 `SubstrateSnapshot.feature_surface` digest + `PredictionActionContext`，预测 expected `|axis_error|`；epistemic_magnitude / improvement_magnitude := `max(0, |axis_error| − critic_prediction)` 的轴聚合，clamp 到 `[0, 1]`，代表"系统能继续压低的部分"。
- per_axis 列出每条轴的 (axis_name, aleatoric, epistemic)。
- `PEDecomposition` append-only 新增 `critic_predicted_magnitude`、`improvement_magnitude`、`critic_update_count`、`critic_checkpoint_id`、`critic_gate_decision`，用于审计 learned critic 的 SHADOW 状态。
- decay 默认 `0.9`，由 `PredictionErrorModule(pe_critic_decay=...)` 注入，避免硬编码。

接入点：

- `PredictionErrorSnapshot.pe_decomposition` 在 bootstrap turn 时为 `None`，正常 turn 由 owner 内部 `_PECriticHead.update(...)` 填充；现有 consumer 仍只读 `error.magnitude` / 轴 error / `signed_reward`，向后兼容。
- `evaluation/backbone.py::_prediction_error_scores` 新增两个 metric：`pe_aleatoric_magnitude`、`pe_epistemic_magnitude`，**严格 report-only**，不进入任何 acceptance gate；目的是避免把"分离"反过来训练成第二套 reward。
- `vz-memory/memory/store.py` 在 PE 写入路径里把 `epistemic_magnitude` / `aleatoric_magnitude` 直接写入 owner-internal `MemoryAttributeReadout`（Phase 1.C），让陪伴向"哪条 PE 是可学的"成为可观察 readout。
- learned critic 的 state 由 PE owner 自己 export / restore；checkpoint id 与 capacity/validation readout 只用于审计，禁止让 critic 直接写 evaluation acceptance gate。

#### Wave E3 promotion criteria（debt #7 闭合候选）

learned PE critic head 何时可以从 readout-only 升级为 acceptance gate 的输入：

| 升级阶段 | 准入条件 | 退出 / 回滚条件 |
|---|---|---|
| `readout-only`（当前默认） | 不需要任何门槛；纯诊断 | — |
| `readout-with-acceptance`（建议下一阶段） | 在 ≥ 200 turn 真 trace 上 `improvement_magnitude` mean ≥ running-stats baseline RMSE 改善 ≥ 0.02；`PEDecomposition.critic_gate_decision` ≠ `block` 占比 ≥ 0.95 | improvement_magnitude mean 退到 < 0 持续 ≥ 50 turn → 退回 readout-only |
| `acceptance gate`（终态） | 在 ≥ 500 turn 真 trace 上 RMSE 改善 ≥ 0.05；rollback drill 通过；`epistemic_magnitude` 不出现塌缩到 0 持续 ≥ 100 turn 的退化 | 一次 rollback drill 失败 → 退回 `readout-with-acceptance` |

实施约束：

- 与 counterfactual rewarding-state head 升级（`docs/specs/credit-and-self-modification.md` Wave E3 段）使用相同的 SHADOW → ACTIVE 三态 + `WiringLevel` 协议。
- rollback drill 测试：`tests/contracts/test_learned_baseline_rollback_drill.py`。
- 升级修改了 evaluation acceptance gate 的输入面，但 `PEDecomposition` schema 不变；现有 consumer 仍按 typed 字段读取，向后兼容。

**快照 schema**：见 `docs/DATA_CONTRACT.md` 3.9 节

### PE Distributional Readout（Phase 2 W1.1-1.3 / DM-1）

来源：Botvinick M, Kurth-Nelson Z, Muller T, Dabney W. *Depression as a disorder of distributional coding*. arXiv:2507.16598, 2025.

核心命题：标量 mean PE 会在分布塌缩时丢失最关键的健康信号——价值分布从「健康宽分布」塌缩为「窄峰 + 偏侧」是 depression-like 状态的神经科学标志。把 PE 从单值升级为带分布形状的 readout，让下游可以观察「分布漂移」而非只能观察「均值漂移」。

落点：`vz-cognition/prediction/distribution.py` 内 frozen `DistributionSummary` dataclass；`PredictionErrorModule` 内 owner-internal `_PEDistributionWindow`；`PredictionError.distribution_summary: DistributionSummary | None`。

机制：

- `_PEDistributionWindow` 维护 4 axis × `max_window=64` 的 bounded 滑动窗口，记录每轴 signed PE 样本。
- 每轮 `_advance` 在 alignment overlay 完成后把最终 `error` 推入窗口（bootstrap turn 跳过，避免初始零噪声污染）。
- 窗口未满 `min_window=8` 时返回 `None`（cold-start safety）；满后计算三个 owner-internal 统计：
  - `iqr`：`Q3(|axis|) - Q1(|axis|)`，clamp 到 `[0, 1]`，代表分布宽度（窄分布 = 塌缩信号）。
  - `entropy`：`|axis|` 在 5-bin 等宽 histogram 上的 Shannon entropy（nats），clamp 到 `[0, log(5)]`，代表分布均匀度（低 entropy + 非零 IQR = 单 mode 锁定）。
  - `asymmetry`：`(mean - median) / (iqr + eps)`，signed，clamp 到 `[-1, 1]`，代表分布偏侧方向（+ = 右尾长 / 偶发大正误差；- = 左尾长 / 偶发大负误差）。
- `min_window` / `max_window` / 5-bin 是 owner-internal 常量，下游 consumer 不应依赖。

#### `min_window=8` 的证据来源（Phase 2 W4 / debt #11 close-out, 2026-05-08）

最初设计 `min_window=16` 是「保守的 IQR 估计样本量」假设。Wave 3 联合证据 run（[`artifacts/eq_uplift/distributional_evidence.json`](../../artifacts/eq_uplift/distributional_evidence.json)）显示，在 5-15 turn 的真实 benchmark scenario 下窗口永远填不满，DM-1 在线上无可观察 evidence —— 形成 debt #11。

debt #11 修法 (3) 方法论：先写 38-turn 长 scenario（[`packages/lifeform-domain-emogpt/.../scenarios/long-form-life-arc.json`](../../packages/lifeform-domain-emogpt/src/lifeform_domain_emogpt/scenarios/long-form-life-arc.json)）跑 [`artifacts/eq_uplift/probe_pe_window_long_form.py`](../../artifacts/eq_uplift/probe_pe_window_long_form.py)，输出 [`artifacts/eq_uplift/pe_window_long_form.json`](../../artifacts/eq_uplift/pe_window_long_form.json)。Validation verdict：

- `first_summary_turn=17`（窗口在 16 个非 bootstrap turn 后填满，符合设计）
- `first_drift_turn=21`（vitals warmup 5 个观察后产 drift，符合设计）
- `iqr_8_over_iqr_16` 4 个 axis 全部 `STABLE`（统计 sanity）

→ mechanism 通过；`min_window=8` 把窗口冷启动从 turn 17 降到 turn 9，让 5-15 turn benchmark scenario 也能产出 distributional evidence。统计 sanity 由 [`tests/contracts/test_pe_distribution_summary_contract.py:test_distribution_window_iqr_stable_at_min_window_n8`](../../tests/contracts/test_pe_distribution_summary_contract.py) 守门（n=8 vs n=32 IQR 比值在 `[0.4, 2.5]`）。

未来若证据要求更紧 IQR 估计（如 ModificationGate 想消费分布形状），可重新评估 min_window 的取值；任何变更必须先重跑 `pe_window_long_form` 探针，再调 contract test。

三条不变量：

1. **None-safe 冷启动**：在 PE owner 观察到至少 `min_window` 个非 bootstrap 样本之前，`distribution_summary` 必须为 `None`。Consumer 看到 `None` 时不应合成代位值。
2. **Read-only**：`distribution_summary` 是 readout，**不进入** ModificationGate / credit gate / regime scoring 等控制路径。Wave 1 唯一合法 downstream 是 `lifeform-vitals` 派生的 `distributional_drift_axes`（slow-scale），以及 audit / evaluation 面板。
3. **Owner-internal 常量稳定**：`min_window` / `max_window` / bin 数 owner-internal；公开契约只是「per-axis 三统计 + window_size provenance」。

接入点：

- `PredictionErrorSnapshot.error.distribution_summary` 在 bootstrap turn 与窗口未满时为 `None`，下游若不读该字段则 byte-for-byte 兼容。
- `vitals.py::VitalsModule.observe_pe_distribution(summary)` 是 W1.3 的 lifeform-side 桥接：lifeform session 在每个 user turn 完成后把 PE summary 喂给 vitals owner；vitals 内部维护 frozen baseline 并发布 `distributional_drift_axes`。
- W1 的 evaluation / ModificationGate / credit / regime / memory 主链**完全不读** `distribution_summary`，确保 R-PE「PE 是原始信号」与 R8「snapshot 隔离」均不破。

**快照 schema 扩展**：`PredictionError` append-only 新增 `distribution_summary: DistributionSummary | None`；`DistributionSummary` 字段 `(window_size, iqr, entropy, asymmetry, description)` 全部 frozen，未来扩展只能新增字段（不能改顺序 / 类型 / 单位）。

## 真梯度 LSS（NL）与 runtime 语义 PE 的关系（Phase 5）

NL 把 Local Surprise Signal 定义为 loss 对模型输出的梯度 `∂L/∂output`，并指出“用 backprop 训练一层等价于构建一个把输入映射到其 prediction error 的 associative memory”，该梯度本身就是被记忆的内容。

- **runtime 仍用语义 PE 作为有界代理**：live online-fast 路径继续用 turn 级 `PredictionError`（无需 autograd、每 turn 可跑），不改本 owner 主链与 schema。
- **新增真梯度 LSS 作为 offline 一等 artifact**：`volvence_zero.prediction.torch_lss`（torch，lazy import，不进 facade）用真 autograd 计算 `∂L/∂output`。MSE 下 `LSS == predicted - actual`，正是“梯度即被记忆内容”的恒等式。
- **代理被 grounding，而非主张**：`bridge_runtime_pe_to_lss` 证明 runtime 语义 PE 的 signed error（`actual - predicted`）**恰等于 −真 LSS**（符号正确、幅度相等），所以有界 runtime 信号是真梯度 surprise 的忠实 stand-in。真 LSS artifact 经 rare-heavy 路径桥接，不进公共 snapshot（R8）。

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

- 2026-07-12: CP-12 owner prediction signal contract 落地。新增 vz-contracts
  `volvence_zero.owner_prediction`（`OwnerPredictionKind` 闭集 enum /
  `OwnerPredictionSignal` / `OwnerPredictionSettlement` /
  `settle_owner_prediction`）。五个 first-wave 语义 owner（commitment /
  relationship_state / goal_value / boundary_consent / execution_result）在自身
  快照发布 `owner_prediction_signals`（v1 = persistence-prior 预测自身 compact
  readout，下一轮由 owner 自己 settle）；`PredictionErrorModule` 作为唯一
  mismatch 计算者消费 settled 信号并发布
  `PredictionErrorSnapshot.owner_prediction_settlements`（v1 report-only，不进
  magnitude 公式）。PE dependencies 追加 relationship_state / goal_value /
  boundary_consent / execution_result（对齐 commitment overlay 先例，upstream.get
  容忍禁用 owner）。测试：`tests/contracts/test_owner_prediction_signal.py`。
- 2026-07-12: CP-11 world/self predictive heads SHADOW 落地。PE owner 内部新增
  `_WorldPredictiveHead`（task/regime/action 轴）与 `_SelfPredictiveHead`
  （relationship 轴）：共享 compact evidence 特征（`_featurize_outcome_evidence`
  固定 7 维聚合，不随上游词表漂移），bounded online-SGD 线性头，与手工
  `_PredictionErrorHead` 同轮双跑并按下一轮 realized outcome 计分。
  `PredictionErrorSnapshot.predictive_head_readout`（`PredictiveHeadReadout`）
  发布 learned/baseline 双 MAE 与 improvement，**report-only**：live prediction
  chain 仍由手工 head 产出。ACTIVE 晋升 gate 依计划 CP-11（≥200 turn SHADOW 上
  RMSE/校准改善 ≥0.02，且 kill 条件适用），本轮不改变默认行为。测试：
  `tests/contracts/test_predictive_heads_shadow.py`。
- 2026-06-29: autograd-owner-integration（LSS rare-heavy 接入）。新增 torch-free `prediction/lss_rare_heavy.py`（`LSSRareHeavyCheckpoint` + `build_lss_rare_heavy_checkpoint`，float-only，grounding gate 强制 runtime PE == −真 LSS，fail-closed）。`PredictionErrorModule` 新增 offline surface：`export_rare_heavy_lss` / `import_rare_heavy_lss` / `rare_heavy_lss_calibration` / `export|restore_rare_heavy_lss_state`，import 只改 owner-internal LSS 校准（EMA），**不**触碰 `PredictionErrorSnapshot`（schema 不变）。`RareHeavyArtifact` 追加 optional `lss_checkpoint` 字段并随 `export_rare_heavy_artifact(lss_checkpoint=...)` 携带。
- 2026-06-29: NL/ETA full-autograd 迁移 Phase 5。新增 `prediction/torch_lss`：真梯度 LSS（`∂L/∂output`）作为 offline 可审计 artifact，并证明 runtime 语义 PE == −真 LSS（符号正确、幅度相等）。runtime PE 主链与 schema 不变；真 LSS 经 rare-heavy 桥接，不进公共 snapshot。
- 2026-06-20: 登记关联设计 spec [`relational-soft-verifier.md`](./relational-soft-verifier.md)（design / SHADOW-only）：拟把 `relationship_error` 轴的 epistemic 部分（复用 `PEDecomposition.improvement_magnitude`）作为关系域软验证器奖励来源；未改动本 owner，待 SHADOW 自我确认证伪实验通过后再新增 §"关系域软验证器奖励来源"。
- 2026-05-06: Phase 1.B 上线 owner-internal Curiosity-Critic running-stats 分解（`PEDecomposition` + `_PECriticHead`）；`PredictionErrorSnapshot.pe_decomposition` 为 optional 字段，bootstrap 时为 `None`；`evaluation` 新增 `pe_aleatoric_magnitude` / `pe_epistemic_magnitude` 两个 report-only metric。Phase 2.B learned critic head 登记为后续 uplift。
- 2026-05-02: 重写对 Emergent Action Abstraction（`docs/specs/emergent-action-abstraction.md`）的依赖口径：PE 消费 temporal segment closure 与可观察 outcome context，不新增 trace owner 或 learning primitive
- 2026-05-28: 新增 `VZ_PE_EVALUATION_DECOUPLED` gate（默认 SHADOW，可回滚），ACTIVE 时 evaluation 不再进入 PE actual outcome 与 evaluation-derived credit；契约测试 `tests/test_pe_evaluation_credit_decoupling.py`
- 2026-04-22: 补充 `pe-eta-pe-readout-only` proof 口径，明确区分 PE publication/readout 与 PE primary dominance
- 2026-04-22: 当前实现口径补充单一 owner-side mapper/head、confidence-aware calibrated error weighting，以及 evaluation 只发布 PE-owner readout 的边界
- 2026-04-20: 初始版本。将 `prediction_error` 从 credit/evaluation 的上游设计原则提升为独立能力域 spec，固定主链契约 `evaluated_prediction -> actual_outcome -> next_prediction -> error`
