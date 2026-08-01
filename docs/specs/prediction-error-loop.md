# Prediction Error 主链 Spec

> Status: draft
> Last updated: 2026-08-01
> 对应需求: R-PE

## 要解决的问题

如何把“预测 -> 实际结果 -> prediction error”从辅助日志提升为正式运行时主链，使它成为后续 credit / memory / temporal / regime / reflection 的共同学习原语？

## 关键不变量

- prediction error / LSS 是原始学习信号，不是可选诊断信息
- 系统必须显式发布 prediction chain，而不是只在下游隐式近似
- evaluation 是 prediction error 的 readout / gate 层，不是学习源头
- credit 是 prediction error 的聚合 / 审计层，不是学习源头
- prediction error 必须以 machine-readable 多维结构对外发布，而不是只剩一条文本描述
- 进入 optimizer 的 realized reward 必须能逐条追溯 outcome lineage；"PE 停止驱动学习信号"不等于
  "环境发布的 outcome payoff 消失"（见 §Runtime replay reward eligibility）
- eligibility 门读的字段必须就是 payout 付的字段；两者分离时必须 fail loudly，不得靠别的文件的
  耦合维持相等
- **`reward = 0` 不等于"没有更新"**：ineligible transition 仍进 GAE 与 PPO 批次，
  `gamma*V(s')-V(s)` 仍产生非零 advantage。这是 value bootstrapping，不是 reward 泄漏

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
- Digital Ant ecology 的 environment owner 可在 `EnvironmentOutcome.measurement.action_payoff` 发布有界局部价态：只使用动作前后可感知的 food/heat signal、body-side path-integration home-distance progress 与离散 pickup/delivery/thermal-threshold 事实，不发布坐标、目标方向或动作标签。中性木棍 contact 只进入 status/evidence 可观察事实，不进入 payoff。PE owner 是唯一 actual-outcome 归一化与 mismatch owner；runtime replay 优化 PE owner 发布的 `ActualOutcome.action_payoff`，并把 `error.signed_reward` 作为 prediction residual 单独报告，禁止用 actual-minus-predicted residual 替代 realized utility。embodiment 不直接写 Internal-RL reward。`ecology_local_valence_enabled=False` 仅用于 matched ablation。

**evaluation→PE/credit 解耦 gate（`VZ_PE_EVALUATION_DECOUPLED`）**：

为了完全兑现 R-PE 第 1 不变量「evaluation 是 readout，不是学习源头」，新增一个显式可回滚 gate，控制 evaluation 是否进入 PE actual outcome 与 evaluation-derived credit：

| WiringLevel | actual outcome 的 `family_signals` | evaluation→credit | 退出条件 |
| --- | --- | --- | --- |
| `ACTIVE`（默认，env 未设或 truthy） | `{}`（中性 0.5），actual outcome 仅由 substrate / dual-track / regime / external-outcome 驱动 | evaluation-derived credit 置空；counterfactual 不传 evaluation（学习写回不触发），仅保留 historical/readout 记录 | 默认部署；matched ACTIVE/SHADOW 证据必须保持 ACTIVE 对 evaluation byte-invariant、SHADOW evaluation-sensitive、两臂不同 |
| `SHADOW`（显式 env falsey） | 来自 `_family_signals(EvaluationSnapshot)`（旧行为） | `derive_learning_evidence_credit_records` + counterfactual `propose_update` 写回照旧 | 仅作为 `VZ_PE_EVALUATION_DECOUPLED=SHADOW` 的显式回滚；不得用于新证据主臂 |

- 实现位置：`vz-cognition/.../prediction/error.py::pe_evaluation_decoupled_active` + `_build_outcome_evidence`；`vz-runtime/.../integration/final_wiring.py` 的 credit 派生段复用同一 gate。
- 2026-08-01 起默认 ACTIVE；matched ablation 只证明 evaluation-fed 信号确实改变 PE/credit、因而该通道 load-bearing，不把这种机制差异冒充产品行为改善。`SHADOW` 是单 env flag 回滚点（R15）。
- 当前 proof harness 允许显式区分两层含义：一层是 **PE publication/readout**（slot + evaluation evidence 仍存在），另一层是 **PE primary dominance**（是否直接主导 joint-loop schedule 与 RL reward）。`pe-eta-pe-readout-only` 用于只保留前者
- `bootstrap=True` 表示当前 turn 尚无可结算的上一轮 prediction；下游不应把这类快照当作真实 learning evidence
- live runtime 中，部分 consumer 会把 `prediction_error` 当作“上一轮结算出的 carryover signal”，以维持单轮 DAG 和 owner 边界

### N+1 表示前向预测（离线研究面）

`PredictionErrorModule` 额外拥有一条显式配置的高吞吐 batch surface：turn N 的冻结上下文表示预测 turn N+1 的冻结观测表示，下一轮以同一 target 同时训练和结算。它不替换当前 live 四轴 head，也不新增第二个 mismatch owner。

- 输入契约为 frozen `ForwardRepresentationBatch`：`sample_ids`、context、N+1 target、persistence baseline 与 `history_turns` 必须逐样本对齐、维度一致且全为有限值；`target_lineage` 必须是 substrate owner 发布的 `SubstrateForwardRepresentationLineage`，其维度必须与 target/persistence 一致。
- owner-internal `TorchForwardRepresentationHead` 使用 `input -> tanh(n_z) -> target` 的有界瓶颈；结算发布逐样本 predicted / actual / signed error、MSE、cosine，以及同 target 上的 persistence baseline。
- `ForwardRepresentationBatchSnapshot` 是 PE owner 的离线 settlement artifact，不进入 live `propagate`；其 target 来自 DATA_CONTRACT §3.1 / §6 注册的 offline SHADOW slot `substrate_forward_representation`。调用方只能经 `PredictionErrorModule.process_forward_representation_batch(...)` 训练或评估，禁止直接实例化 head 形成第二 owner。
- checkpoint 为 float-only、带 geometry/schema/parameter fingerprint 与完整 target lineage 校验；head 在第一批绑定 lineage，后续 batch、restore 或 model/readout/sample snapshot 漂移必须 loudly fail。
- target 语义只由 `vz-substrate` 解释：冻结模型最后 token、选定 residual layers、稳定 layer order、L2-normalized readout。PE owner只拥有预测与 mismatch，不遍历 capture、不编码原文。MiniLM/其它外部 sentence encoder target 只允许作为历史 mechanism pilot，不能构造新的 `ForwardRepresentationBatch` 或取得 thesis 资格。
- promotion 条件：真实人类 multi-session heldout 上，N+1 head 必须优于同 substrate target 的 persistence，并通过 temporal-owner 容量阶梯、同一冻结 substrate 的长上下文 matched baseline、完整 runtime attestation 和多 seed 门；此前保持 offline/report-only。PE forward-head `n_z` 只代表 predictor capacity，不是 temporal-controller `n_z`。旧 CP-11 四轴手工 head 的 output/target space 与 substrate 表示不同，不能伪造跨空间数值对照。
- `scripts/run_msc_prediction_test_plan.py` 是该研究线的独立 MPS 命令行控制面。它固定禁止
  CPU fallback，并与经新控制面启动的七日产品实验共享 MPS 互斥锁；控制面落地前手工
  启动的旧进程必须另行确认退出。当前只授权 `preflight` 与
  `mechanism-only-smoke`；`formal` 在 same-substrate context、完整 runtime collector、
  temporal-controller capacity 三门未齐时固定返回退出码 3。CLI 状态不是 evidence，不能由
  “命令存在”推导 thesis 已执行。
- mechanism runner 必须以精确 configuration fingerprint 管理可续跑 journal。语料
  provenance、模型/源码 SHA、device、split 限制、seed、层和超参必须全部进入
  fingerprint；`--resume` 遇任一漂移或已登记文件 SHA 变化都必须 fail loudly。
  语料索引、context/target 数值张量、arm/split 和 seed 结算为不可变单元；
  journal 不得保留 MSC 原文。`run_state.json` 只是可变控制面，不进 evidence
  hash；最终 manifest 封口前 `analysis_allowed=false`，封口后仍保持
  `formal_claim_allowed=false`。中间 checkpoint 禁止用于换 seed、选容量或产生
  effect verdict。`status --output-dir` 只允许暴露进度和 gate，不暴露效应值。
- 长上下文资格由实际语料 token exposure 决定：同一冻结 Qwen tokenizer 对每条 full history
  记录 raw token；若全体低于 32k 声明上限，则 32k 已是零截断 full-history steelman，128k
  不构成不同实验臂。只有样本真实超过 32k 时，才允许另开真实 128k model/config/hardware
  prereg；禁止运行时篡改冻结模型 config 冒充 128k。

### Gate 1 LSS link registry

`prediction/torch_lss.py` 是 PE owner 的 rare-heavy/offline 真梯度审计面，不是
第二个在线 PE owner。Gate 1 mechanism evidence 只允许以下 link：

| surface | loss / parameterization | runtime signed PE |
|---|---|---|
| `numeric` | scalar MSE，直接对 output 求导 | `actual - predicted` |
| `vector` | component-wise MSE，直接对 output 求导 | component-wise `actual - predicted` |
| `probability` | Bernoulli cross-entropy，对 logit 求导 | `target - probability` |
| `enum` | categorical cross-entropy，对 logits 求导 | `one_hot(target) - probabilities` |
| `distribution` | soft-target categorical cross-entropy，对 logits 求导 | `target_distribution - probabilities` |

每个 link 内 runtime signed PE 必须等于 `-dL/d(parameter)`，容差 `1e-9`。
link 之间损失、梯度与 normalization 不同，禁止把不同 link 的 component
直接相加或平均成新的“统一 LSS”。输入必须有限；概率严格位于 `(0,1)`，
categorical/distribution prediction 必须为严格正且归一化的概率单纯形，
distribution target 必须为非负且归一化，enum target 必须在类别范围内。
违反任一约束必须 fail loudly。

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

## PE / Epistemic Value / Intrinsic Reward 三层契约（2026-07-20）

来源：Google Paradigms / DeepMind, "Can In-Context Learning Support Intrinsic Curiosity?"（`arXiv:2606.19476`，负面定理）与 Friston 等 "Active Inference as Test-Time Scaling"（`arXiv:2606.22813`，正面构造）。两篇必须一起读：PE 是一级原始信号，但**只有在声明环境结构、posterior 假设和 nuisance terms 之后**，才能上升为 epistemic value 或 policy update 依据。详见 `research/frontier-sweep-2026-07-20.md` §H2 / §K1 / §4.4。

### 负面定理（BAMDP 有偏性）

在一般 Bayes-Adaptive MDP 中，仅用 frozen ICL predictor 的 prediction error 与反事实 context manipulation，**无法无偏恢复 Bayesian information gain**：

- 部分 estimator 含不可消除的 nuisance term；
- learning-progress 形式的 reward 根本不能由 ICL prediction error 实现；
- 正面结果只在 Bayesian Experimental Design / active learning 等**非时间结构**问题中成立，且是长轨迹渐近逼近。

**对本 owner 的直接约束**：`prediction_error` 的 mismatch（含 `epistemic_magnitude`）不自动等于 epistemic value，更不自动等于 curiosity / intrinsic reward。任何把 "预测错了" 直接当 "值得探索" 的下游用法都是契约违反。

### 三层契约

| 层 | 对象 | Owner | 语义 | 升级条件 |
|---|---|---|---|---|
| L1 raw mismatch | `PredictionError`（四轴 error、`magnitude`、`pe_decomposition`） | `prediction_error` owner | 原始预测失配 + epistemic/aleatoric 分离（Curiosity-Critic 机制，见上节） | 无条件发布（本 owner 主链） |
| L2 epistemic value | "该失配是否可减小 / 是否值得投入学习" 的估计 | 下游 readout（如 memory PE 写门、joint-loop schedule、apprenticeship surprise） | **必须带文档化假设**：环境结构（是否近似 BED/active-learning 情形）、估计偏差来源、aleatoric 隔离方式 | 消费 L1 的 `epistemic_magnitude` 时须在自身 spec 声明假设；不得宣称无偏 information gain |
| L3 intrinsic reward | 进入 Internal RL / credit 的探索激励 | `credit` / `internal_rl`（经 PE→credit lineage） | 有界、可回滚、带 kill 条件的 reward shaping | 每个新 L3 用法必须有 matched-control 证据（explore-on vs explore-off）+ 回滚点；禁止把 L1 mismatch 或裸 ICL loss delta 直接接成 reward |

三条不变量：

1. **层间不塌缩**：L1→L2→L3 每一步都是显式转换，禁止任何 consumer 把 L1 字段直接当 L3 reward 消费（现状核查：`internal_rl` 的 reward 经 credit 聚合，不直读 `magnitude`——保持此边界）。
2. **假设可审计**：L2 消费方的 spec 必须包含"环境假设 + 偏差声明"段；缺失即为契约违反（fail loudly，不静默降级）。
3. **PE→credit→Internal RL 分层不是多余复杂度**：它正是防止"预测错了 = 值得探索"谬误的机制载体；任何"简化直连"的提案须先反驳负面定理的适用性。

Active Inference TTS 作为理论母体保留在 research motivation：PE 可以统一驱动 world model 与 policy posterior 更新，但其"同时在线改 policy/world model + free-energy 单一总目标"的用法是 R2 反例，不进 runtime。

## Runtime replay reward eligibility 与 PE-drive / outcome-payoff 拆分（W3-a, 2026-07-26）

本节冻结 runtime-replay 结算接缝上两条**通用**契约。二者都是 opt-in，默认值逐字节复现历史行为，关闭即精确回滚；内核不引入任何领域语义（不知道 butter / heat / food / ant 是什么），只区分"环境是否为该 transition 发布了 outcome payoff"。

Owner：`volvence_zero.internal_rl.sandbox.InternalRLSandbox.settle_runtime_action`（结算）与
`volvence_zero.joint_loop.runtime.ETANLJointLoop`（声明、转发、发布）。
Runtime 门面：`FinalRolloutConfig.internal_rl_runtime_*`。

### 缺陷 1：measurement-free tick 静默产生 substrate 派生 reward

链路：`final_wiring` 只在 `EnvironmentOutcome.measurement` 存在时把
`environment_action_payoff` 传给 PE owner → `prediction/error.py` 仅在该值非 None 时覆盖
`ActualOutcome.action_payoff`，否则保留 PE owner 自己合成的 action 轴（substrate feature 信号与
evaluation family 信号的混合）→ `settle_runtime_action` 把该值直接当作
`realized_action_payoff` 进入 batch。结算本身只在 `EnvironmentOutcome` 对象为 None 时才 drop，
从不检查 measurement 是否存在。

因此对于**任务路径禁止任何 distance/potential shaping** 的域（数字蚂蚁，见
`docs/specs/digital-ant-embodiment.md` §4），"去掉稠密局部塑形"的那条臂实际上仍在被另一套
稠密的感知派生塑形训练，消融含义反转；按契约不携带 payoff 的动作也会得到严格正 reward。

**契约：`RuntimeReplayRewardEligibility`（typed enum）**

| 取值 | 语义 | 状态 |
|---|---|---|
| `any-settled-outcome` | 每条 lineage 匹配的结算都可获得 realized payoff（可能是 PE owner 合成的 action 轴） | **默认 = 历史行为 = 精确回滚** |
| `environment-measured-only` | 只有 `measurement.action_payoff` 非 None 的 transition 可获得 realized payoff | 严格模式 |

严格模式下的不变量：

1. ineligible transition **仍然结算**：lineage、动力学、milestone/terminal 标记、PE 残差诊断全部保留；
2. `realized_action_payoff = 0`，且 `segment_bonus = 0`——PE 派生的 segment credit 不得把同一份量偷渡回来；
3. transition 被**逐条打标**：`ZTransition.runtime_reward_eligible` +
   `runtime_reward_eligibility_reason ∈ {eligible, ineligible:no-environment-measurement,
   ineligible:no-environment-action-payoff}`，使 "逐条审计 nonzero reward 的 outcome lineage" 可执行；
4. 语言域（companion 等）根本没有 environment measurement，**必须保持默认值**，否则整条 reward 流恒为 0；
5. **门读哪个字段就付哪个字段**（W3-a follow-up，见下）；
6. **ineligible ≠ 无更新**：realized payoff 与 segment bonus 都为 0，但 GAE 仍从 critic
   bootstrap 出非零 advantage，optimizer 仍会更新参数（见下"ineligible transition 上仍然发生的更新"）。

#### 门读哪个字段就付哪个字段（gate/payout 不变量）

`_resolve_reward_eligibility` 用 `EnvironmentOutcome.measurement.action_payoff` 判定资格，而
历史 payout 行读的是 `prediction_error_snapshot.actual_outcome.action_payoff`——**两个不同对象**。
它们相等只因为另外两个文件的耦合：`final_wiring.py` 令
`environment_action_payoff = measurement.action_payoff if measurement is not None else None`，
`prediction/error.py` 仅在该值非 None 时覆盖 PE owner 自己合成的 action 轴。这条耦合不是契约：
自己组装 `PredictionActionContext` 的域、或这两个文件的任何后续改动，都会在
`eligible` 标签不变的前提下静默恢复原缺陷（付出 substrate 派生 reward）。

因此严格模式下 `settle_runtime_action`：

- **消费 gate 自己读的那个字段**（`measurement.action_payoff`），不再依赖上述耦合；
- 同时与 PE owner 发布的 `actual_outcome.action_payoff` **精确比对（`!=`，不带任何容差）**，
  不等即抛 `RuntimeReplayRewardEligibilityError` fail loudly（AGENTS.md §6），因为该偏差意味着
  审计标签已经不描述被支付的 reward；

  **为什么不该有容差**：接通路径上两个量按构造逐位相等，而不只是接近——payout 读
  `_clamp(measurement.action_payoff)`，PE owner 存的是
  `_clamp_signed(measurement.action_payoff)`（`prediction/error.py`，与 sandbox `_clamp` 是同一句
  `max(-1.0, min(1.0, x))`），payout 再 clamp 一次；同界 clamp 幂等，这条路径上没有任何算术能把
  两者分开。outcome lineage（`capture_id` / `environment_outcome_id` 匹配）另行强制，比对不会跨两次
  测量。因此 `1e-9` 的 slack 只可能吞掉**真实**分歧，也就是本守卫存在的理由本身；这与 W3-b-fix
  删除 `_CAUSAL_ACTION_HEAD_ENVELOPE_TOLERANCE` 是同一条论证。证据：
  `test_gate_payout_agreement_has_no_tolerance`（相差 1 ulp、远小于 `1e-9`，现在抛错）、
  `test_clamp_composition_cannot_separate_the_two_axes`、
  `test_wired_path_agrees_bit_for_bit_without_any_slack`（端到端无误报）。

  **作用域：这是一致性校验，不是有限性校验**。`_clamp` 会把非有限值静默压到界上（实测
  `_clamp(nan) = _clamp(inf) = 1.0`、`_clamp(-inf) = -1.0`），两条 lane 压法相同因而合法地一致，
  本 seam 保持沉默。该 laundering 属于共享 `_clamp` 的上游问题（它同时约束 reward/advantage），
  由 `test_exact_agreement_is_not_a_finiteness_check` 记述性锚定，需另开收敛包在 ingress 契约处修；
- 该校验在 **eligible 即执行**，与 `outcome_payoff_reward_enabled` 无关——否则一条坏接线可以
  躲在消融开关后面，等开关打开时才开始付错数。

默认契约 `any-settled-outcome` 完全不受影响：仍然支付 PE owner 的轴，逐字节等于历史行为。
实测（`packages/vz-temporal/tests/test_runtime_replay_reward_eligibility.py`）：同一组发散输入
（measurement `0.2` / PE 轴 `0.9`）下，严格臂现在抛错，默认臂仍然支付 `0.9`、reward `0.96`。

#### ineligible transition 上仍然发生的更新（不是 reward 泄漏，是 value bootstrap）

严格模式下一条 measurement-free transition 的 reward 恒为 `0.0`，发布出的
`RuntimeReplayRewardStream.reward_sum` 也是 `0.0`。**读者不能由此推断"没有发生更新"**：GAE 的
`delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)` 在 `r_t = 0` 时仍然非零，因为两个 surface 是不同的
substrate signature，critic 对它们的估值不同。这是标准的 value bootstrapping，不是 reward 泄漏——
没有任何环境未发布的量进入 reward——但它确实会移动策略参数。

实测（50 条 measurement-free transition，全部 `reward == 0.0`，
`scripts` 之外的一次性复现见变更日志所列测试的同名 fixture）：post-GAE `max |advantage| = 2.54`
（归一化后）、`max |return| = 1.58e-01`、`max |V| = 2.04e-01`，一次 PPO 批次令
`max |Δ track weight| = 1.04e-04`、`max |Δ critic weight| = 1.12e-03`。量级随 surface 变化幅度
浮动（审计侧在蚂蚁 probe 上报过 `8.4e-06` 同一现象），但**方向恒定：reward_sum = 0 不等于
参数不变**。要真正冻结这些 tick 必须把它们排除出 batch，那是另一条契约，本包不提供。

#### 未闭合：segment credit 没有 per-transition eligibility 概念（spec-only）

`segment_bonus` 的来源是 `CreditSnapshot.recent_credits` 中 `source_event == f"segment:{segment_id}"`
的记录，按 **segment id 在滚动窗口内选取**，记录本身不携带"产生它的那一拍是否 eligible"。因此
一条在 measurement-free tick 上产生的 credit record，仍可能在同一 segment 的后续 eligible tick 上
被平均进 `segment_bonus`。严格模式的第 2 条不变量（ineligible tick 自身 `segment_bonus = 0`）成立，
但它不能阻止跨拍的窗口混入。闭合它需要在 credit record 上增加 eligibility 归属并改 credit owner，
属于另一个收敛包。**在此之前，严格模式下 eligible tick 的 `segment_bonus` 不得被解读为
"只由 eligible tick 的证据构成"。**

### 缺陷 2："PE-off" 被静默实现成 "reward-off"

`external_prediction_error_drive=False` 经 `session.py` →
`runtime_replay_prediction_error_enabled` → sandbox，同时把 `segment_bonus` 与
`realized_action_payoff` 都置 0，PE-off matched arm 于是在恒零 reward 流上训练。设计中的 PE-off
（本 spec kill 条件、ant lane 文档）含义是"PE 停止驱动学习信号与 temporal switch 压力，仍是
readout"，而不是"环境 payoff 消失"。

**拆成两条正交、显式命名的契约**：

| 契约 | 字段 | 语义 |
|---|---|---|
| (a) PE 驱动学习信号 | `external_prediction_error_drive` / `runtime_replay_prediction_error_enabled` | PE 派生的 segment credit bonus + PE→`beta_t` switch 压力。这才是消融要关掉的东西 |
| (b) outcome payoff 到达 optimizer | `FinalRolloutConfig.internal_rl_runtime_outcome_payoff_reward: bool \| None` | `None`（默认）= 由 eligibility 契约推导；显式 `True`/`False` 覆盖推导 |

(b) 的推导规则（这是默认可回滚的关键）：

- eligibility = `any-settled-outcome` → (b) 跟随 (a)。此时 realized payoff 可能就是 PE owner 合成的
  轴，跟随 PE drive 在语义上成立，且**与历史行为逐字节一致**；
- eligibility = `environment-measured-only` → (b) 恒为 `True`。此时 realized payoff 按构造是
  环境发布量、不是 PE 派生量，因此与 PE drive 正交，PE-off 臂继续获得环境 payoff。

### 发布：optimizer 实际消费的 reward 流

`ETANLJointLoop.latest_runtime_replay_reward_stream` 发布不可变
`RuntimeReplayRewardStream`，经 `AgentSessionRunner.latest_runtime_replay_reward_stream` 只读透出。
字段：eligibility 契约、(b) 状态、(a) 状态、settled / eligible / ineligible / nonzero-reward /
nonzero-payoff / nonzero-bonus 计数、realized payoff、segment bonus 与 reward 三个求和、
per-reason 计数、最近一次 reason。

这是 readout，不回灌学习路径。它填的空白是：此前没有任何 consumer 能看见 optimizer 的 reward——
环境侧计数器（如 ant 的 `nonzero_ecology_payoffs`）恰好在泄漏的那些 tick 上为 0，而
`nonzero_reward_steps` 是 PE signed residual，两者都不是 optimizer 的 reward。

### Latent code clamp 约定

sandbox 的 `_clamp` 是 `[-1, 1]`，对 reward / advantage 正确，但同一函数也被用于 latent code /
modulated mean / candidate mean / policy mean 的重建，而**在线 owner 把 `z_t` 限制在 `[0, 1]`**
（`temporal/interface.py` 的 `_clamp` 与 `metacontroller_components.clamp_unit`）。当 causal action
head 的残差为负时，replay lane 会重建出冻结 plant 根本无法输出的 mean。新增
`FinalRolloutConfig.internal_rl_runtime_latent_unit_clamp`：`False`（默认）保持历史 signed 边界，
是精确回滚；`True` 只对 latent code / mean 重建改用 `[0, 1]`。**reward / advantage 的
clamp 边界在任何取值下都不变。**

值域本身只有一个 owner：`temporal/interface.py` 的 `LATENT_CODE_BOUNDS`。sandbox 的
`_clamp_unit` 与 `resolve_latent_code_bounds` 都从它派生，**禁止在第二个文件里写死 `[0, 1]`**；
signed 分支保留的 `(-1.0, 1.0)` 是历史回滚基线，不是第二个 latent 值域声明。

**两条 lane 必须收到同一个值。** pure lane 由
`FinalRolloutConfig → session → joint loop → sandbox` 端到端接通；torch lane 同名 kwarg
`torch_causal_ppo_update(..., latent_unit_clamp=)` 由唯一生产调用点
`CausalZPolicy._maybe_run_torch_ppo` 转发。该转发不靠约定：调用前
`assert_runtime_replay_latent_bounds_agree` 对**即将发出的 payload** 按 callee 自身签名做
`bind_partial`，比较**解析出的边界**而非 flag，因此漏传 kwarg、传错值、或任一 lane 的
`resolve_latent_code_bounds` 语义漂移都会抛 `RuntimeReplayLatentBoundContractError`
fail loudly，而不是让同一 batch 训练出两个策略。负对照：去掉转发后，跨 lane 测试在
`latent_unit_clamp` 的 `True` 与 `False` 两个参数化下都失败（torch lane 记录到 `<omitted>`）。

**数字蚂蚁声明该契约**（`ant_runtime_replay_rollout_config`）。它是唯一带 ACTIVE causal action
head 的域，也正是该缺陷的原始来源。32-tick ECOLOGY 会话（`objective=ECOLOGY`、
`sense_schema=ECOLOGY_V2`、seed 7、matched 双臂）实测：

| 量 | OFF（回滚） | ON（声明） |
|---|---|---|
| replay 重建次数 | 244 | 244 |
| 两个边界重建结果不同的次数 | 72（29.5%） | 72 |
| 重建均值最大差 | 4.30e-05 | 4.30e-05 |
| reward 流 settled / eligible / ineligible | 62 / 50 / 12 | 62 / 50 / 12 |

即**冷启动下该声明近似数值惰性**：终态 `max |Δ head bias| = 6.5e-09`、
`max |Δ track weight| = 3.0e-15`、`Δ reward_sum = 2.6e-09`（reward 语义不受影响，符合设计）。
它的作用随 head 学到的残差幅度增长：把同样这 244 次真实蚂蚁重建的 steering 残差设为冻结包络允许的
bias 上限（`CAUSAL_ACTION_HEAD_UPDATE_ENVELOPE.bias_absolute_limit = 0.1`，反对称加在 contrast
pair `(0, 1)` 上）后，**236/244（96.7%）** 次重建发生分歧，重建均值最大差 `0.0805`，且 signed lane
重建出低至 `-0.0804` 的均值——冻结 plant 不可能输出的码，正是审计中 `(action - mean)` 反号、
head 梯度指向错误方向的机制。结论：该契约在冷启动阶段不改变结论，在 head 真正学到转向权威后才
生效，因此可以安全地在正式臂上默认打开。

### 已知缺口（本包未闭合）

- `volvence_zero.internal_rl.__init__` 与 `volvence_zero.joint_loop.__init__` 尚未 re-export
  `RuntimeReplayRewardEligibility` / `RuntimeReplayRewardStream`；当前经模块路径导入。
- ~~torch PPO backend 尚未消费 `latent_unit_clamp`~~：W3-b 给了 torch lane 同名 kwarg，
  W3 follow-up 在唯一生产调用点转发并加了跨 lane 校验（见上）。**残余**：校验发生在调用点，
  不是类型系统；新增第二个 `torch_causal_ppo_update` 调用点必须自行走同一 helper。
- segment credit 的滚动窗口选取没有 per-transition eligibility 归属（见上 spec-only 条目）。
- ineligible transition 仍参与 GAE 与 PPO 批次；`reward_sum = 0` 不代表参数不变（见上）。
- `RuntimeReplayRewardStream` 是进程内累计量，不进 rare-heavy checkpoint，跨 session 不续接。

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

- 2026-08-01: 增加两实验隔离的 MPS CLI 控制面。MSC 线的 status/preflight/smoke/formal
  分级，MPS 算术探针、CPU fallback 禁止、跨实验互斥锁与 formal blocker 退出码均 fail
  closed；32k/128k 改按实际 token exposure 判定，避免在 MSC 全历史未超过 32k 时制造伪臂。

- 2026-08-01: substrate-target owner convergence。新增
  `substrate_forward_representation` offline SHADOW slot；冻结 Qwen target 由
  substrate owner 以 model weights SHA、runtime origin、latest-token selected-layer
  residual geometry 与 sample/value hashes 发布。PE `ForwardRepresentationBatch`
  和 checkpoint 升级为 v2 target-lineage binding；MiniLM target 不再能构造正式
  batch。Companion Bench 容量字段改名 `forward_head_n_z`，并删除所有 ETA
  promotion/kill 语义。

- 2026-08-01: R2 research packet。`VZ_PE_EVALUATION_DECOUPLED` 默认从
  SHADOW 翻为 ACTIVE，`SHADOW` 保留单 env rollback；matched evidence 同时
  要求 ACTIVE evaluation-invariant、SHADOW evaluation-sensitive 与 cross-arm
  difference。新增 PE-owner offline `ForwardRepresentationBatch` N+1 表示预测
  surface、float-only checkpoint 与同 target persistence settlement；不新增 slot，
  不改 live 四轴 schema。

- 2026-07-30: `PredictionErrorModule` 增加 owner-side frozen learning gate。
  `learning_enabled=False` 时仍计算并发布 PE、滚动分布 readout 与下一轮预测上下文，但
  不更新 learned critic、world/self predictive heads 或其持久误差统计。

- 2026-07-27: gate/payout 一致性校验去掉 `1e-9` 容差，改为精确 `!=`
  （收敛包 W3-b-fix-follow-up）。理由：接通路径上两个量按构造逐位相等（同一句
  `max(-1.0, min(1.0, x))` 的幂等 clamp，见上文），outcome lineage 另行强制，容差只可能吞掉
  真实分歧。证据：相差 1 ulp（`math.nextafter(0.2, 1.0)`，远小于 `1e-9`）现在 fail loudly；
  端到端在 `0.0 / ±0.2 / ±1.0 / 0.1+0.2 / 1e-17` 上无误报；默认 `any-settled-outcome` 契约
  逐字节不变（818 行 `float.hex()` 探针在干净 HEAD worktree 与本树上 md5 同为
  `65517dc30205941b079883fd860422d1`）。负对照：放回 `1e-9`，
  `test_gate_payout_agreement_has_no_tolerance` 立刻以 `DID NOT RAISE` 失败。
  **同批披露、不在本包修**：`_clamp` 对非有限值静默 laundering（实测 `_clamp(nan) = 1.0`），
  一个 NaN measurement payoff 会被当作最大奖励支付，且 `_require_signed_unit_interval` 也漏过；
  两条 lane laundering 相同，故本 seam（正确地）不响。由
  `test_exact_agreement_is_not_a_finiteness_check` 锚定现状，修复应在 ingress 契约处另开包。
- 2026-07-27: W3 follow-up（对抗评审闭环，4 处）。(1) **死接缝**：`latent_unit_clamp` 的 torch lane
  从未被唯一生产调用点 `CausalZPolicy._maybe_run_torch_ppo` 转发，spec 要求"两条 lane 必须收到同一个
  值"却无人执行；现转发并新增 `assert_runtime_replay_latent_bounds_agree`——按 callee 签名
  `bind_partial` 检查即将发出的 payload、比较解析出的边界，漏传/传错/语义漂移一律抛
  `RuntimeReplayLatentBoundContractError`。负对照：撤掉转发后跨 lane 测试在两个参数化下都失败。
  (2) **门读一个字段、付另一个字段**：严格模式改为消费 gate 自己读的
  `measurement.action_payoff`，并与 PE owner 的 `actual_outcome.action_payoff` 比对不等即抛
  `RuntimeReplayRewardEligibilityError`（当时带 `1e-9` 容差，2026-07-27 已收紧为精确比对，见上一条）；
  校验在 eligible 即执行，不受
  `outcome_payoff_reward_enabled` 影响。该校验立刻在既有 reward-stream fixture 上抓到一处真实发散
  （measurement 0.2 vs PE 轴 0.42）。默认契约不受影响，逐字节回滚。
  (3) `_clamp_unit` 与新的 `resolve_latent_code_bounds` 从 owner 常量 `LATENT_CODE_BOUNDS` 派生，
  不再硬编码 `[0, 1]`；数字蚂蚁 evidence profile 声明 `internal_rl_runtime_latent_unit_clamp=True`，
  并在 32-tick ECOLOGY 会话上量化（冷启动 72/244 次重建分歧、终态参数差 ≤6.5e-09；把 steering 残差
  提到冻结包络 bias 上限后 236/244 分歧、均值差 0.0805、signed lane 重建出 -0.0804）。
  (4) 记录两条**不修复只披露**的事实：ineligible transition 的 `reward_sum = 0` 不代表参数不变
  （GAE value bootstrap，实测 50 条全零 reward 仍产生 `max |Δw| = 1.04e-04`），以及 segment credit
  按 segment id 滚动窗口选取、没有 per-transition eligibility 归属。
  测试：`packages/vz-temporal/tests/test_runtime_replay_reward_eligibility.py`（25 通过）。
- 2026-07-26: 收敛包 W3-a。新增 §"Runtime replay reward eligibility 与 PE-drive /
  outcome-payoff 拆分"：typed `RuntimeReplayRewardEligibility`（默认
  `any-settled-outcome` = 逐字节回滚，`environment-measured-only` = 严格模式，ineligible
  transition realized payoff 与 segment bonus 同时为 0 并逐条打标）；把
  `external_prediction_error_drive` 拆成 (a) PE 驱动学习信号 / temporal switch 压力 与
  (b) `internal_rl_runtime_outcome_payoff_reward`（`None` 由 eligibility 推导），使 PE-off 臂在
  严格模式下继续获得环境 payoff；新增只读发布面
  `ETANLJointLoop.latest_runtime_replay_reward_stream`（optimizer 实际消费的 reward 流）；新增
  opt-in `internal_rl_runtime_latent_unit_clamp`（latent 重建改用在线 owner 的 `[0, 1]`，
  reward / advantage 边界不变）。数字蚂蚁 evidence profile 声明严格 eligibility。
  测试：`packages/vz-temporal/tests/test_runtime_replay_reward_eligibility.py`。
- 2026-07-20: 新增 §"PE / Epistemic Value / Intrinsic Reward 三层契约"。吸收 ICL Intrinsic Curiosity（`2606.19476`）BAMDP 负面定理与 Active Inference TTS（`2606.22813`）正面构造：L1 raw mismatch / L2 epistemic value / L3 intrinsic reward 三层显式转换，禁止层间塌缩；L2 消费方必须声明环境假设与偏差来源。来源 `research/frontier-sweep-2026-07-20.md` §6 同步项，不改运行时行为。
- 2026-07-14: CP-12 第二波 settlement 覆盖（GAP-05）。
  `PredictionErrorModule._OWNER_PREDICTION_SLOTS` 扩至 9 slot（追加
  plan_intent / open_loop / belief_assumption / user_model），dependencies
  同步追加四个 slot（upstream.get 容忍禁用 owner）。mismatch 计算逻辑不变，
  仍 report-only。同日：CP-14/GAP-04 —— 四轴 PE credit records 的唯一派生点
  收敛为 `CreditModule.process`（`derive_credit_records_from_prediction_error_first`
  含 segment closure）；`run_final_wiring_turn` 不再 post-propagate 重复派生
  context-less 四轴记录（此前每轮双计）。测试：
  `tests/contracts/test_credit_single_derivation_path.py`。
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
- 2026-07-15: CP-11 heads 特征加宽 + gate 窗口度量。(a) `_featurize_outcome_evidence`
  从 7 维聚合扩到 **18 维**：substrate signals 增加 spread/peak 聚合、substrate delta
  增加 max-abs，新增 owner-internal **滞后 realized outcome（4 轴）与滞后 signed
  learned-head error（4 轴）** AR 特征（PE owner 本就消费每轮 `ActualOutcome`，
  lag 状态不出 owner，R8 不变）；checkpoint `feature_dim` 相应变为 18（schema
  版本不变，旧 7 维 checkpoint restore 按既有 fail-loud 路径拒绝）。(b)
  `PredictiveHeadReadout` 追加 report-only 窗口字段：`window_size(=200)` /
  `window_sample_count` / `window_world_improvement` / `window_self_improvement`
  + 残差诊断 `window_axis_learned_maes` / `window_axis_baseline_maes` /
  `window_target_stds` / `window_persistence_maes`（lag-1 persistence 参照）。
  依据：510-turn real-trace soak 显示累计 improvement 在 ~50 turn 后即平台
  （world +0.011 / self +0.014 各窗口平稳），瓶颈是特征而非训练时长；CP-11 gate
  原文 ">= 0.02 improvement over >= 200 turns" 的直接读法是 trailing window，
  `run_learned_shadow_soak.py` 的 `validation_delta` 改为窗口满 200 时取窗口
  improvement（artifact 新增 `validation_delta_basis` 声明度量基准，窗口未满时
  显式标注 cumulative fallback）。live prediction chain 不变，全部 report-only。
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
