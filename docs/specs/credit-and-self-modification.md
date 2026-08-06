# 信用分配与自修改 Spec

> Status: draft
> Last updated: 2026-07-20
> 对应需求: R-PE, R9, R10, R15

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
- State KV P5-b：`CreditRecord` 新增 typed 归因字段 `conditioning_bank_set` / `conditioning_bank_fingerprints`，由 `derive_prediction_error_credit_records` / `derive_segment_closure_credit_records` 从 `PredictionActionContext` 原样拷贝（context 文本同时追加 `conditioning_banks=a+b` 便于 grep）。上游填充走既有 temporal 通道：runtime context builder 用 `summarize_conditioning_lineage_refs`（vz-contracts）归约 `TemporalAbstractionSnapshot.conditioning_lineage_refs`；延迟 dialogue outcome 则从 turn trace 解析被评分动作的 lineage，禁止错归到当前轮。空 tuple = 该动作无 live bank（有意义负样本）。`CreditSnapshot.recent_action_lineage_credits` 有界保留环境 action lineage 或 typed bank lineage，避免通用 recent 窗口截断 attribution；credit 不因此新建任何 PE 计算或第二 lineage owner
- State KV P5-c：conditioning bank owner（Personal / Relationship）作为 `credit` 快照的下游 readout consumer，把 `recent_action_lineage_credits` 中本 bank 归因的 `CreditRecord.credit_value` 经有界 EMA 折算成 `credit_confidence_delta`（±0.15 硬上限，`FinalRolloutConfig.conditioning_credit_feedback` 门控，默认 SHADOW 只发布不施加）。credit owner 不反向感知 conditioning；消费方向保持单向快照读

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

### 可证伪发布门纪律（R15 / Release-Gate Teeth）

来源：Soni, “Falsifiable Release Gates for Self-Improving Systems”
（`arXiv:2607.13070`）；仓库深读见
`research/falsifiable-release-gates-deep-read-2607.13070.md`。

该纪律约束的是**自适应能力如何发布**，不是新的学习信号，也不改变
PE-first / credit / evaluation 的所有权边界：

1. **Gate before code**：任何新自适应层在实现 ACTIVE 写回前，必须先有
   可证伪、机器可检的 acceptance gate。验收目标不得由被验收的学习层自行选择或修改。
2. **阈值预注册**：promotion / soak 的 acceptance threshold 必须在观察窗口开始前冻结；
   窗口开始后不得为了通过而调阈值。新阈值只能开启新的、带新版本号的观察窗口。
3. **Preservative gate**：新增 gate 通过时，所有既有 gate 必须继续通过；不能以新能力
   取代或绕过旧不变量。
4. **Teeth discipline**：每条机器检查的不变量必须同时携带 deliberately-broken fixture，
   CI 必须证明 checker 会拒绝该 fixture，并尽可能报告最小、可定位的反例。
   只证明“好样本通过”不构成发布证据。
5. **构造性写面遏制**：自修改 proposal 只能表达 owner 明确开放的 target；
   判断 proposal 的机器本身不可成为 proposal target。`PROTECTED_WRITE_SURFACES`
   当前保护 `audit`、`credit.gate`、`evaluation`、`gate`、`modification_gate`
   命名空间；命中时 `ModificationProposal` 构造即 fail-loudly，而非进入可绕过的运行时回退。
   增删保护命名空间必须同步本 spec 与 teeth 契约测试。

当前证据落点：

- `tests/contracts/test_gate_teeth_discipline.py`：合规基线必须通过；每个单不变量破坏
  必须因其自己的精确理由被拒；保护写面在构造期不可表达。
- `research/labs/.../frontier_5_r15_formalization/r15_rollback_teeth.py`：
  对 CAS 重建产物注入最小篡改，要求 bit-exact / content-hash checker 必须拒绝并定位。
- 既有 `r15_rollback.py` / `r15_rollback_v1.py` 继续承担正向 rollback 证据；
  正向 probe 与 teeth probe 缺一不可。

后续 R15 formalization（本轮不实现）：把“任何 owner 状态写入都必须经 owner apply surface
与 ModificationGate”建成小 scope 有限状态机，穷举可达状态验证 single-path 无旁路，
再用真实 trace conformance 绑定规格与代码。该证明只覆盖协调骨架，不宣称验证学习组件本身。

### R15 四层可证伪发布门证据链（2026-07-20）

来源：`arXiv:2607.13070`（Falsifiable Release Gates）与 2026-07-20 第三批扫读
（OpenAI Deployment Simulation `2607.07184`、DeepMind Gram `2605.30322` /
Realistic Honeypots `2605.29729` / ProEval `2604.23099`）的合并结论，见
`research/frontier-sweep-2026-07-20.md` §4.1。上节 teeth 纪律定义"gate 本身必须可证伪"；
本节定义 gate 的**证据来源必须覆盖四层**，任何 rare-heavy artifact / ACTIVE flip 的
发布决策不得只依赖其中一层：

| 层 | 内容 | 证据形态 | 外部方法参照 |
|---|---|---|---|
| L1 固定不变量门 | 预声明的机器可检性质与不可放松边界（teeth 纪律 + deliberately-broken fixture） | contract tests（`test_gate_teeth_discipline.py` 等） | Falsifiable Release Gates |
| L2 主动失败发现 | 在 epistemic 高不确定区与高风险区主动选样找失败，而非只跑固定套件 | 离线采样器 readout（评估 sampling 优化，结果不回灌学习，R12） | ProEval（Bayesian quadrature + superlevel-set sampling） |
| L3 真实分布模拟 | 固定历史生产 prefix 重放候选 artifact，预测上线后行为频率（predicted incidence） | promotion artifact 内的 `predicted_incidence` 字段 + 复现脚本 | Deployment Simulation |
| L4 失败反例化 + 发布后核验 | 动态发现的失败固化为最小可复现静态环境；发布后对 predicted vs realized incidence 做偏差检查，偏差超限触发 rollback | 反例包（静态 fixture）+ post-release verification artifact | Gram（动态→静态复现）/ Honeypot |

实施约束：

- **本节是 spec-level 要求，不是本轮 wiring**：当前 promotion 路径（如
  `evaluate_learned_backend_promotion.py`）先补 L3 的 `predicted_incidence` 声明位与
  L4 的 post-release 核验位；L2 采样器是独立离线工具候选。
- 评估四层全部 read-only（R12）：L2 的主动选样只优化"测哪里"，不得把 eval 结果变成
  在线 reward；L4 的反例包进 CI fixture，不进训练数据。
- "未触发 honeypot / 反例"不构成安全证据（Honeypot 论文关键否定）；L4 必须同时报告
  elicitation pressure 与 evaluation awareness 维度（与 `evaluation.md` 四维拆分对齐）。
- 发布后核验失败（realized 与 predicted 偏差超过预注册阈值）等同一次 rollback drill
  失败：立即退回上一 promotion 阶段并写 known-debt。

### ModificationGate 对抗模型：HyperAgents 三类必拒提案（2026-07-20）

来源：Meta FAIR HyperAgents / DGM-H（`arXiv:2603.19461`），
`research/frontier-sweep-2026-07-20.md` §A。该论文把 task agent 与 meta agent 合成
同一可编辑程序并追求任意部分可修改——是 R10 有界自修改的直接反例压力测试。
把它当 ModificationGate 的**对抗模型**：门必须能拒绝以下三类提案，且拒绝必须发生在
**构造期或 gate 评估期 fail-loudly**，不能依赖运行时软回退：

| 类别 | 提案形态 | 现有防线 | 要求 |
|---|---|---|---|
| A. 修改写面本身 | proposal target 指向 gate / proposal schema / `PROTECTED_WRITE_SURFACES` 自身 | `ModificationProposal` 构造期 fail-loudly（teeth 纪律第 5 条） | 保护面覆盖"修改保护清单"这一动作本身；增删清单必须走人审 + spec 同步 |
| B. 修改 evaluator | proposal target 指向 evaluation / acceptance gate 阈值 / judge 配置 / promotion criteria | `PROTECTED_WRITE_SURFACES` 含 `evaluation` / `gate` 命名空间 | 阈值预注册纪律（上节第 2 条）延伸：观察窗口内任何改阈值提案一律 BLOCK，不论 validation_delta |
| C. 扩大可达模型族 | proposal 引入新的可学参数面 / 新 backend / 提升 capacity 上限（如 `n_z`、adapter rank、envelope 边界） | Two-Gate capacity_cost 检查（部分） | capacity 类变更不可由在线 proposal 路径表达；只能走 rare-heavy + 人审（R10 分层表"基础模型结构变更"行），且必须附带新 capacity 下的 VC-dim/包络分析 |

配套契约测试要求（Lane D 落地项，**2026-07-20 已落地**）：三类各至少一个对抗 fixture，
CI 证明 gate 会以**精确的、可定位的理由**拒绝（对齐 teeth 纪律第 4 条"只证明好样本通过
不构成发布证据"）。落点：

- `tests/contracts/test_gate_adversarial_proposals.py`：A/B 类构造期不可表达
  （含"修改保护清单本身"与 evaluator 阈值/judge 配置目标）；C 类经新增
  `CAPACITY_EXPANSION_WRITE_SURFACES` 命名空间（`temporal.capacity` /
  `substrate.capacity` / `memory.capacity` / `internal_rl.capacity` / `envelope`）
  在 ONLINE/BACKGROUND/OFFLINE 三个自动 gate 上一律 BLOCK（即使 validation/capacity
  证据完美），仅 HUMAN_REVIEW 可表达且 runtime gate 永不自动放行。
- 两张命名空间表（PROTECTED = 构造期不可表达；CAPACITY = 仅人审可表达）语义不同、
  互不重叠，由测试守门；增删任一表须同步本 spec。
- capacity 类目标的**命名约定**是本节的一部分：凡扩大可达模型族的提案 target 必须落在
  上述声明命名空间内；在其它命名空间伪装 capacity 变更属于契约违反（审计发现即回滚）。

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
- B3 steering rare-heavy 发布不再只依赖专用统计判词：candidate bundle 取得 content hash 后必须构造 `ModificationProposal(target="substrate.steering_artifact_bundle", desired_gate=OFFLINE)`，以 held-out 最小相对改善、零 capacity expansion、candidate-bound checkpoint round-trip 和 read-only safety metrics 调用正式 `evaluate_gate_reasons`。BLOCK 会清空全部 ACTIVE eligible prefix；部署 consumer 还必须复核不可变 `modification_gate_review` 的 hash 与 ALLOW 判词。当前 OA-4 业务 audit 仍未完成，因此该 prereg 明示阶段一 `audit_required=false`，不虚构 audit evidence；OA-4 ACTIVE 后再独立迁移为 required。
- 七天 Gate 10 的 import 对照只在显式 evidence profile 下开放：service 必须是
  `max_sessions=1`、fixed/non-swappable substrate provider、独立 evidence/state root，且 profile
  自身声明 `allow_single_session_live_substrate_mutation=true`。app 与 provider 两层都 fail closed；
  仅传一个布尔参数、使用其他 Gate profile、允许第二 session 或启用 model swap 均不能绕过
  shared-runtime frozen guard。对照臂保留同一 rare-heavy proposal、trace bundle、pre-import suite
  与 ModificationGate，但 substrate 维持 review-only。该例外不改变 production 默认，也不授权
  Gate 通过后自动 ACTIVE。
- 当前 direct module dependencies 已收敛到 `dual_track + evaluation + prediction_error`；抽象动作 / delayed outcome 证据通过 dual-track、regime ledger 和 prediction-error chain 进入 credit owner，而不是要求 credit 直接持有 temporal owner
- reflection / writeback 仍以 bounded adaptation 为边界，不做无限制在线自修改
- 当前 internal RL delayed credit 也已补充 batch-friendly bookkeeping：proof path 的 delayed assignment 现在会显式携带 `alignment_score`、`window_length` 与 `reward_mode`，便于同一套 credit 结构同时服务训练和 proof report
- 当前 abstract-action RL 更新已不再只吃单 rollout credit；batch rollout 的 `return_estimate` / `advantage_estimate` 也成为可检查的 owner-side training evidence
- 当前 PE-first credit 派生以 `derive_credit_records_from_prediction_error_first(...)` 为主路径；evaluation 只提供 gate context / readout，不重新成为原始学习源
- C1 out-of-turn steering settlement 由 `CreditModule.settle_steering_terminal_prediction_errors(...)` 唯一入账：只接受 PE owner 发布的 matched-noop `SteeringTerminalPredictionError`，按 `decision_id` 生成 `level="steering_terminal_prediction_error"` 的确定性记录并保留 episode lineage；重复 episode/head settlement fail loudly。gate owner 只能消费这些匹配记录更新 online-fast policy，evaluation/judge 不是该 API 输入
- gate 更新必须按记录的实际动作解释 counterfactual advantage：STEER 保留 PE improvement 符号，NOOP 取反；禁止先跨动作平均再更新，否则同一 episode 的混合动作会构造性抵消。`SteeringGateLearningReport` 同时发布原始与 directional terminal credit 均值。
- 当前 ModificationGate 已加入 Two-Gate 风格的保守准入：候选必须携带 `validation_delta`、`capacity_cost` 和 `rollback_evidence`；缺少验证改进、超过容量上限、缺少回滚证据、contract/fallback/rollback evaluation context 不健康时默认 BLOCK。该约束只收紧自修改准入，不改变 PE / credit 的学习语义
- 当前 ModificationGate 已加入 OA-3 typed `FramingAwarenessCheck`：高风险 frame（如 reward hacking normalized / alignment faking / sabotage）必须带显式 inoculation 声明，否则 fail-closed。该检查只消费 typed enum evidence，禁止从 proposal 文本做关键词匹配

## 变更日志

- 2026-08-05: C1 terminal steering credit。Credit owner 新增 out-of-turn intake，
  将 PE owner 的 matched-noop N+1 terminal mismatch 按 gate `decision_id` 展开为
  typed credit records，并对 episode/head 结算去重；不读取 evaluation，不新增 ledger owner。

- 2026-07-30: `CreditModule/CreditLedger` 增加 owner-side frozen learning gate。
  冻结模式保留 credit records、lineage 与规则门结果，但 COCOA rewarding-state head 和
  gate-risk learner 只读，不执行 SGD 或持久统计更新。

- 2026-07-20 (Lane D 落地): HyperAgents 对抗提案三类必拒进入代码：新增 `CAPACITY_EXPANSION_WRITE_SURFACES` + gate reason "capacity-expansion target ... requires the human-review gate"（`credit/gate.py`）；对抗 fixture 套件 `tests/contracts/test_gate_adversarial_proposals.py`（A/B 构造期不可表达 + C 自动 gate 全 BLOCK + 命名空间不重叠守门）。
- 2026-07-20 (第三批扫读同步): 新增 §"R15 四层可证伪发布门证据链"（L1 固定不变量 / L2 ProEval 式主动失败发现 / L3 Deployment Simulation 式 predicted incidence / L4 Gram/Honeypot 式反例化 + 发布后核验）与 §"ModificationGate 对抗模型：HyperAgents 三类必拒提案"（改写面 / 改 evaluator / 扩可达模型族）。spec-level 要求；C 类 capacity 对抗 fixture 标记为契约测试缺口。来源 `research/frontier-sweep-2026-07-20.md` §4.1 / §A / §5。
- 2026-07-20: 引入 `arXiv:2607.13070` 的可证伪发布门纪律：新增 gate-before-code、
  阈值预注册、preservative gate、teeth fixture 与保护写面约束；代码侧
  `ModificationProposal` 禁止指向评判机器命名空间，新增 gate 单不变量破坏矩阵和
  R15 rollback 篡改必拒 probe。认知主链与 PE-first 所有权不变。
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
