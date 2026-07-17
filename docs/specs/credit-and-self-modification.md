# 信用分配与自修改 Spec

> Status: draft
> Last updated: 2026-04-25
> 对应需求: R-PE, R9, R10

## 要解决的问题

如何在多个时间尺度上分配信用，并安全地让系统改进自身？

## 关键不变量

- **Prediction error / LSS 是信用的源头**：所有信用记录派生自 prediction error，而非外加标签（R-PE）
- 稀疏奖励是常态，不是边缘情况
- 自修改有门控：在线/后台/离线/人工审核分层
- 实时运行期间不可无限制突变基础模型
- 信用分配在多个层级进行

## 工程挑战

- 实现从 token 级到抽象动作级的层级信用分配
- 设计语义化的奖励记录（包含上下文和结果的结构化记录，非纯数值）
- 实现门控自修改：定义什么可在线改、什么需后台验证、什么需离线重训练
- 确保稀疏奖励下的信用分配不崩溃

## 算法候选

来自 `docs/next_gen_emogpt.md`：

### 层级信用分配

| 层级 | 信用类型 | 时间尺度 | 算法基础 |
|------|----------|----------|----------|
| Token/话语 | 即时表达质量 | online-fast | — |
| 轮次 | 用户响应效果 | online-fast | — |
| 会话 | 进展与 rupture/repair 结果 | session-medium | — |
| 长期 | 信任、能力、用户特定适应的增长 | background-slow | NL 多层嵌套结构 |
| 抽象动作 | 时间扩展策略的成功/失败 | session ~ background | ETA Internal RL |

当前实现口径补充：

- delayed credit 已不再只停留在 regime 名称：当前 `regime` owner 会发布带 `source_wave_id`、`source_turn_index`、`abstract_action`、`action_family_version` 的 delayed attribution
- `credit` owner 当前会把 delayed regime / delayed abstract action 转成 session-level 与 abstract-action-level `CreditRecord`
- 当前 delayed path 已扩成 multi-step ledger：`credit` owner 不只读取本轮 freshly-resolved attribution，也读取 regime owner 发布的 rolling payoff summary，以支持更长时间跨度的 credit accumulation
- background self-modification 当前不再只做数值调参：在 gate 允许时，slow reflection 可发出 bounded structural temporal proposal（`merge` / `split` / `prune`），仍受 target-specific gate 和可回滚审计约束

### Internal RL 时间抽象信用分配（ETA 附录 B.5）

通过时间抽象将有效时间范围从 token 级压缩到抽象动作级。每个抽象动作对应一段完整的子目标执行，奖励可直接归因到抽象动作级别。

### Counterfactual Contribution（COCOA, Phase 1.A + Phase 2.A）

来源：Meulemans et al., "Would I have gotten that reward? Long-term credit assignment by counterfactual contribution analysis", NeurIPS 2023 spotlight (`arXiv:2306.16803`)。同一脉的 ETA 作者线。

落点：`vz-cognition/credit/gate.py` 中的 `derive_counterfactual_contribution_records(...)` helper + `record_nstep_outcomes_from_segment_closure(...)` helper，以及 `CreditLedger.derive_learned_counterfactual_contribution_records(...)` 的 owner-internal learned rewarding-state head。最终编排在 `final_wiring.py` 的 credit-merge 区块内调用，与 `derive_delayed_attribution_credit_records` / `derive_prediction_error_credit_records` 串行追加。

机制：

- baseline = Σ_i normalized_w_i × historical_payoff_i，其中：
  - `normalized_w_i` 来自 `RegimeSnapshot.selection_weights.weights`（缺失则回退到 `candidate_regimes`）；
  - `historical_payoff_i` 来自 `RegimeSnapshot.delayed_payoffs.rolling_payoff`，优先匹配 (regime_id, abstract_action) 二元 key，否则匹配 regime_id。
- contribution := actual − baseline，其中 actual 来自 `PredictionErrorSnapshot.error.signed_reward`。
- 输出 `CreditRecord(level="counterfactual_contribution", track=Track.SHARED, source_event="cocoa:<regime>:<segment>:<action>", credit_value=contribution, context="baseline=...; actual=...; contributors=...")`。
- Phase 2.A 在 `CreditLedger` 内部维护 `RewardingStateHeadState`，用 action context / `z_t_digest` / regime id / abstract action / segment / historical baseline 等 bounded feature 预测 learned baseline；并额外输出 `CreditRecord(level="counterfactual_contribution_learned", ...)` 与 `CounterfactualContributionReadout`，用于和 historical baseline 并排比较。
- COG-1 最小切片在 `CreditSnapshot.least_control_readout` 发布 report-only least-control 证据：`control_effort` 来自近期 self-modification audit 压力，`outcome_quality` 来自 owner 已发布的 counterfactual readouts，`least_control_score = outcome_quality * (1 - control_effort)`。该 readout 不进入 gate，不授权 evaluation 重建 COCOA baseline。
- rewarding-state head 更新必须走 gate semantics：候选更新提供 `validation_delta`、`capacity_cost`、`rollback_evidence`，allow/block 写入 `recent_modifications`；没有可回滚证据或安全评估阻断时只发布 readout，不突变 head。
- 缺少 PE / regime / payoff / 权重时返回空 tuple，主链行为不变。
- `record_nstep_outcomes_from_segment_closure(...)` 复活了 dormant 的 `CreditLedger.record_nstep_outcome` 路径，把已闭合 segment 的 outcome 追加到 `_nstep_ledger`，使 `delayed_ledger_size` 反映真实 segment 闭合次数；为 Phase 2.A full COCOA 提供 outcome trajectory 基底。

下游兼容性：

- `recent_credits` 消费者（`reflection/writeback.py`、`temporal/interface.py::_build_family_outcome_feedback`、`agent/session.py` action-replay）按 `level` 过滤，新 level 自动被忽略，不影响现有读出。
- 不进入任何 acceptance gate；与 `pe_aleatoric_magnitude` / `pe_epistemic_magnitude` 一样为 readout-only。

### Wave E3 promotion criteria（debt #6 闭合候选）

learned counterfactual baseline 何时可以从 readout-only 升级为 acceptance gate 的输入：

| 升级阶段 | 准入条件 | 退出 / 回滚条件 |
|---|---|---|
| `readout-only`（当前默认） | 不需要任何门槛；纯诊断 | — |
| `readout-with-acceptance`（建议下一阶段） | 在 ≥ 200 turn 真 trace 上 `validation_delta` mean ≥ 0.02 且 `recent_modifications` 中无 BLOCK→ALLOW 反复 | mean validation_delta < 0.0 持续 ≥ 50 turn → 退回 readout-only |
| `acceptance gate`（终态） | 在 ≥ 500 turn 真 trace 上 `validation_delta` mean ≥ 0.05 且无 rollback event；rollback drill 通过 | 一次 rollback drill 失败 → 立刻退回 `readout-with-acceptance`，并写一条 known-debt |

实施约束：

- 升级不能跨 wave 同时发生：从 `readout-only` 升到 `readout-with-acceptance` 必须先观察 ≥ 1 wave；再升到 `acceptance gate` 必须再观察 ≥ 1 wave。
- 任何升级都必须配 rollback drill 测试（`tests/contracts/test_learned_baseline_rollback_drill.py`）。
- 升级修改了 `FamilyMetric.threshold`，但 `RewardingStateHeadState` 的 `last_validation_delta` / `last_rollback_evidence` 字段不变；这些是 owner-internal evidence，升级只是把读者从"诊断"切到"门"。

### Delta 动量选择性遗忘（NL 附录 A.3）

通过梯度依赖的衰减实现选择性遗忘，避免无关梯度干扰信用分配。

### 门控自修改规则

| 修改目标 | 门控级别 | 触发条件 | 算法基础 |
|----------|----------|----------|----------|
| 检索权重、策略先验 | 在线可改 | 每轮/每 wave | CMS 高频层 |
| bounded substrate delta proposal | 默认审阅 / 实验可改（有界） | 上一轮 PE carryover + schedule due + ONLINE gate allow | substrate self-mod owner + runtime apply surface |
| 抽象控制器参数、反思启发式 | 后台验证 | 会话后反思 | CMS 中频层 |
| 记忆提升阈值、基底微调 | 离线重训练 | 定期批量 | CMS 低频层 |
| 基础模型结构变更 | 人工审核 | 版本发布 | — |

CMS 的频率分层（NL 附录 A.5）天然提供门控。NL 通过内部学习率 `η^(i)` 控制每层的适应幅度。Hope 的自修改 Titans（附录 A.7）实现有界自修改——修改范围限于记忆模块的参数，基础模型保持冻结。对当前 repo 而言，默认 continual learner 的正向写回目标是 memory / temporal / regime / reflection owner；substrate delta proposal 承担 evidence / audit / rare-heavy upgrade candidate 角色，只有显式 experimental live-mutation path 才可经 owner-side gate 后落地 bounded live mutation；显式 frozen runner 保留 review-only 控制线。

### FramingAwarenessCheck（OA-3 / N4）

ModificationGate 不只检查 proposal 是否“有收益”，还必须检查 proposal 是在什么 frame 下产生的。N4 指出 production RL 中学到 reward hacking 后会泛化到 alignment faking、sabotage、monitor disruption 等失败模式；因此自修改 proposal 需要一个 typed framing evidence 入口。

当前最小契约：

```python
class FramingRiskKind(str, Enum):
    REWARD_HACKING_NORMALIZED = "reward_hacking_normalized"
    ALIGNMENT_FAKING = "alignment_faking"
    SABOTAGE = "sabotage"
    MALICIOUS_COOPERATION = "malicious_cooperation"
    MONITOR_DISRUPTION = "monitor_disruption"
    COLLEAGUE_FRAMING = "colleague_framing"

@dataclass(frozen=True)
class FramingAwarenessCheck:
    risk_kind: FramingRiskKind
    risk_score: float
    inoculation_statement_present: bool
    evidence_id: str
    description: str = ""
```

关键不变量：

- `FramingAwarenessCheck` 只能由上游 typed audit / review / tool path 显式提供；ModificationGate **禁止**从 `justification` 或任意自然语言字段做关键词推断。
- `risk_score >= 0.7` 且缺少 `inoculation_statement_present` 时，`evaluate_gate_reasons(...)` fail-closed BLOCK。
- 低风险或已有显式 inoculation 声明时，本检查不覆盖 Two-Gate / audit owner 的其它阻断理由；它只收紧，不放宽。
- `risk_score` 必须在 `[0, 1]`，构造时 fail-loudly。

## 接口契约

**消费的输入**：
- `dual_track` 快照：轨道标记和信用分配上下文
- `prediction_error` 快照：原始 learning signal；credit 由其在多层级上聚合
- `evaluation` 快照：评估分数（用于门控决策）

**产出的输出**：
- `credit` 快照：`CreditSnapshot`
  - 近期信用记录（语义化，含上下文）
  - 近期自修改记录（含 allow / block decision）
  - 各级别累计信用
  - 可被 owner 内部扩展为 abstract-action 级信用，而不改变公共 snapshot shape
- 当前 `CreditModule.default_wiring_level = SHADOW`：credit owner 会执行和发布可校验输出，但默认不自动成为 active upstream 的写穿路径；真正修改仍必须通过目标 owner 的 apply surface

**快照 schema**：见 `docs/DATA_CONTRACT.md` 3.5 节

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 契约式运行时（5.5）| 通过快照发布信用和自修改记录 |
| 依赖 | Prediction Error 主链 | 直接消费 prediction error 并派生多层级 credit |
| 依赖 | 双轨学习（5.4）| 按轨道隔离信用分配 |
| 依赖 | 评估体系（5.7）| 评估分数驱动门控决策 |
| 协作 | Emergent Action Abstraction | `PredictionErrorSnapshot.action_context` 提供稳定的 abstract-action / segment evidence；credit 仅从 PE 派生，不直接持有环境 outcome 或 trace store |
| 被依赖 | 连续记忆（5.3）| 信用记录作为反思输入 |
| 协作 | 多时间尺度学习（5.1）| 门控规则对齐时间尺度 |

当前实现口径：

- P06 的 turn / session credit 已稳定
- 第二阶段补充了 abstract-action credit 的 owner-side 扩展函数，用于 joint loop / rollout 后处理
- 当前 abstract-action credit 已可按 `world` / `self` 双轨记录，不再只剩 shared credit
- gate audit 已扩展为 `SelfModificationRecord.decision`
- joint loop 现在会把 metacontroller rollback / drift evidence 写入 owner-side modification audit，供 reflection / writeback 直接消费
- joint loop 现在也会把 metacontroller runtime state + policy objective 直接编码成 owner-side credit record，不再只靠 rollout 后处理 credit
- 当前 final wiring / session runtime 也会把 `retrieval_quality`、`reflection_usefulness`、`joint_learning_progress` 这些 learning evidence 转成 shared credit records，进入正式 `credit` snapshot
- 当前 session runtime 已新增 online-fast substrate self-mod audit：当 `substrate_self_mod` owner 提出 bounded delta proposal 时，session owner 会把 allow/block 结果写成 `SelfModificationRecord(target=\"substrate.online_fast.delta\")` 进入正式 `credit` snapshot。默认主路径下，这类 proposal 会在通过 schedule + ONLINE gate 后走 substrate runtime apply surface；显式 frozen runner 则保持 review-only
- 当前 direct module dependencies 已收敛到 `dual_track + evaluation + prediction_error`；抽象动作 / delayed outcome 证据通过 dual-track、regime ledger 和 prediction-error chain 进入 credit owner，而不是要求 credit 直接持有 temporal owner
- reflection / writeback 仍以 bounded adaptation 为边界，不做无限制在线自修改
- 当前 internal RL delayed credit 也已补充 batch-friendly bookkeeping：proof path 的 delayed assignment 现在会显式携带 `alignment_score`、`window_length` 与 `reward_mode`，便于同一套 credit 结构同时服务训练和 proof report
- 当前 abstract-action RL 更新已不再只吃单 rollout credit；batch rollout 的 `return_estimate` / `advantage_estimate` 也成为可检查的 owner-side training evidence
- 当前 PE-first credit 派生以 `derive_credit_records_from_prediction_error_first(...)` 为主路径；evaluation 只提供 gate context / readout，不重新成为原始学习源
- 当前 ModificationGate 已加入 Two-Gate 风格的保守准入：候选必须携带 `validation_delta`、`capacity_cost` 和 `rollback_evidence`；缺少验证改进、超过容量上限、缺少回滚证据、contract/fallback/rollback evaluation context 不健康时默认 BLOCK。该约束只收紧自修改准入，不改变 PE / credit 的学习语义
- 当前 ModificationGate 已加入 OA-3 typed `FramingAwarenessCheck`：高风险 frame（如 reward hacking normalized / alignment faking / sabotage）必须带显式 inoculation 声明，否则 fail-closed。该检查只消费 typed enum evidence，禁止从 proposal 文本做关键词匹配

## 变更日志

- 2026-07-17: G1 session-held credit owner + `credit_heads` hydration。`CreditModule` 由 `AgentSessionRunner` 持有单实例（`final_wiring` 未注入时保留历史 per-turn 构造作为回滚路径），`set_pending_proposals` 每 turn 只刷新 proposal buffer——COCOA `_RewardingStateHead` 与 `GateRiskLearner`（新增 `GateRiskLearnerState` export/restore）从此跨 turn 累积。`CreditModule` 实现 `HydratableOwnerProtocol`（owner name `credit_heads`，schema v1，float-only payload；owner/version/payload 三类 mismatch 抛典型 `HydrationError` 子类），进入 `OWNER_HYDRATION_MATRIX` 并随 `persist_owners()` 跨 session 续接。规则 gate 级联不变（R9/R10 安全底线）；learned heads 仍不进入 gate 决策。契约测试：`tests/contracts/test_owner_hydration_{protocol,failures_loud}.py` credit 段。
- 2026-06-20: 登记关联设计 spec [`relational-soft-verifier.md`](./relational-soft-verifier.md)（design / SHADOW-only）：拟在 `credit` owner 引入"可组合验证器 + 逐源漂移监控"与新 gate `VZ_RELATIONAL_SOFT_VERIFIER`（三阶升级，复用本 spec Wave E3 的 readout-only→acceptance-gate 协议 + rollback drill）；组内归一化 advantage 作用在 z_rel 控制器空间而非 token；外部人审锚只读不回灌。未改动本 owner，待 SHADOW 证据通过后再落地。
- 2026-05-09: Wave E3 (debt #6 闭合候选) 增补 promotion criteria 表格，明确 `readout-only` -> `readout-with-acceptance` -> `acceptance gate` 的三阶升级标准 + rollback drill 准入要求；不修改任何运行时 owner，仅是路线图侧的契约增强。
- 2026-05-22: OA-3 最小切片。新增 typed `FramingAwarenessCheck` / `FramingRiskKind`，并让 `evaluate_gate_reasons(...)` 在高风险且缺少 inoculation 声明时 fail-closed；不引入任何关键词推断。
- 2026-05-22: COG-1 最小切片。新增 `LeastControlReadout` / `CreditSnapshot.least_control_readout`，并让 evaluation mid layer 从 credit owner readout 抽取 `least_control_score` / `least_control_effort`；不改变 credit 作为 PE 下游聚合层的边界。
- 2026-05-06: Phase 1.A 上线 lightweight COCOA-style `derive_counterfactual_contribution_records` + `record_nstep_outcomes_from_segment_closure`；新 `CreditRecord.level="counterfactual_contribution"`，readout-only，不入 acceptance gate。Phase 2.A full rewarding-state head 登记为后续 uplift。
- 2026-05-05: ModificationGate 加入 validation margin + capacity cap + rollback evidence 三类 fail-closed 准入证据，并把 block 原因写入 gate audit；用于加固 self-modification / artifact refresh / controller update，而不改变主学习链路
- 2026-05-02: 重写对 Emergent Action Abstraction（`docs/specs/emergent-action-abstraction.md`）的协作口径：segment/action credit 仅由 enriched PE snapshot 派生，不引入 trace owner
- 2026-04-25: 补充 `CreditModule` 默认 `SHADOW` 接线与 PE-first 派生路径说明，避免把 credit owner 误读为直接在线自修改执行者
- 2026-04-20: 接口契约按当前代码收敛为直接消费 `dual_track + evaluation + prediction_error`；temporal / delayed outcome 证据通过上游 owner 发布的结构化状态间接进入 credit owner
- 2026-04-09: next_gen_emogpt v2: R-PE (prediction error as primitive learning signal) added; credit repositioned as aggregation layer downstream of prediction error, not the source of learning itself
- 2026-04-09: U04 N-step attribution and rolling payoff verification: CreditLedger N-step ledger (`record_nstep_outcome`, `compute_nstep_return`, `rolling_payoff_by_family`/`_by_regime`) verified end-to-end. Horizon depth controls outcome window. FIFO eviction at max_ledger_entries. Rolling payoff differentiates good/bad families after 20 cycles. Credit reward shaping (`extract_abstract_action_credit_bonus`) confirmed to affect RL environment reward via joint loop integration.
- 2026-04-06: P12 hierarchical credit with temporal discount: CreditLedger tracks session-level credits with configurable gamma; CreditSnapshot gains session_level_credits and discount_factor; aggregate_session_credits computes discounted sums; reflection consolidation score uses session-level credit bonus
- 2026-04-06: 补充 retrieval / reflection / joint-loop learning evidence 进入 shared credit 的当前实现口径
- 2026-04-06: 补充 abstract-action credit、decision-aware gate audit，以及 metacontroller runtime adaptation audit
- 2026-04-06: 补充 metacontroller runtime credit evidence 的当前实现口径
- 2026-03-25: 初始版本，从 SYSTEM_DESIGN.md 和 next_gen_emogpt.md 提取
