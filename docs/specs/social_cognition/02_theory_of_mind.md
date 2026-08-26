# Theory of Mind Spec

> Status: draft
> Last updated: 2026-08-22
> 对应需求: R17, R16, R11, R-PE, R1, R8, R15

## 要解决的问题

当前 `UserModelSnapshot` 把 stable preferences、working style hints、sensitive boundaries、durable goals 都压在同一类 `SemanticRecord` 上。它缺少认知科学中最基础的区分：对方相信什么、打算做什么、正在感受什么、长期偏好什么。这些不是同一类状态，不能共享一个 owner 或一个更新规则。

R17 把 Theory of Mind 拆成四个 learned owners，并要求每个 owner 都有自己的 prediction、PE、timescale 和 reconciliation 规则。

## 关键不变量

- `belief_about_other`、`intent_about_other`、`feeling_about_other`、`preference_about_other` 是不同 owner，不是 `UserModelSnapshot` 的四个字符串字段。
- 所有 ToM state 必须 keyed by `interlocutor_id`，依赖 R16。
- LLM structured output 只能产生 `SemanticProposal`，不能直接成为 ToM state。
- owner 必须使用 memory、confidence、time decay、conflict 和 prediction error reconciler 来维护稳定状态。
- belief / intent / feeling / preference 的 PE 不可混写。
- ToM owners 消费 Environment Event / role / identity 提供的 conversational frame；不得从 renderer 文案或 raw text 下游重建 speaker / audience / subject。
- ToM owner records 仅在 `LLMSemanticProposalRuntime`（或其它 typed proposal source）wired 时产生；默认 `NoOpSemanticProposalRuntime` 下 records 永远为空。这是 Phase 1 W1.C 的 fail-closed 设计（详见 [`packages/vz-runtime/src/volvence_zero/integration/final_wiring.py`](../../../packages/vz-runtime/src/volvence_zero/integration/final_wiring.py) 第 1201-1219 行：当 `semantic_proposal_runtime` 是 `LLMSemanticProposalRuntime` 时，`tom_proposal_runtime` 才会被自动派生）。Benchmark 端的激活状态可以通过 `BenchmarkReport.tom_records_total` 与 family report 中的 `f3.tom_records_total` 观察；0 = EQ owner 链路本轮未激活，需要在 lifeform 入口（如 `build_companion_lifeform_with_real_substrate(use_llm_semantic_runtime=True, ...)` 或 `lifeform-bench --use-llm-semantic-runtime`）显式启用。

## Owner / Timescale / Prediction Error / ETA Consumption

### Owner

四个 owner 分别单写：

- `BeliefAboutOtherModule`
- `IntentAboutOtherModule`
- `FeelingAboutOtherModule`
- `PreferenceAboutOtherModule`

每个 owner 发布 keyed snapshot，`UserModelSnapshot` 在迁移后降级为兼容 aggregate / read model，不再拥有全部 ToM truth。

### Timescale

- `online-fast`: 接收当前 turn semantic proposals，更新 transient affect / intent hypotheses。
- `session-medium`: 在 scene 内根据 follow-through、repair、clarification、explicit feedback 校正 working model。
- `background-slow`: 通过 reflection writeback 将稳定 preference / belief / boundary 提升为 durable records。
- `rare-heavy`: 用于 offline ToM evaluator / calibrator refresh，不在 live runtime 直接更新 substrate。

### Prediction Error

每个 owner 发布不同 prediction：

- belief predicts how new information will be interpreted
- intent predicts likely next action or follow-through
- feeling predicts affective response / rapport movement
- preference predicts durable response style or boundary

outcome mismatch 生成 owner-specific social PE，例如 intent follow-through failure 不应污染 durable preference；affect prediction failure 不应覆盖 belief state。

### ETA Consumption

ToM snapshots 是 controller / regime / planner 的 compact advisories。它们可以影响 `z_t`、`beta_t`、regime priors 和 question budget，但长期策略更新仍在 latent controller space。Renderer 不读取用户文本判断 ToM；它只表达 planner 已选择的 social action.

## 工程挑战

- 旧 `UserModelModule` 既承担 profile summary 又承担 preference-ish state，需要拆职责。
- SemanticProposalAdapter 必须显式 target owner，不能把所有 profile event 都扔到 `user_model`。
- 需要 owner-specific evidence gates：false-belief、intent mismatch、affect misread、preference conflict。
- 迁移期必须避免 `response_assembly` 同时消费 old `user_model` 和 new owners 导致 double counting。

## 算法候选

- LLM structured proposal with typed target owner and confidence。
- Embedding similarity for belief / preference continuity candidates。
- PE-weighted record promotion: repeated low-error predictions increase stability。
- Conflict-aware decay: contradicted beliefs decay faster than durable preferences。
- Session-post reflection for durable ToM consolidation。

## 接口契约

```python
@dataclass(frozen=True)
class OtherMindRecord:
    record_id: str
    interlocutor_id: str
    summary: str
    detail: str
    confidence: float
    status: str
    source_turn: int
    prediction_error_refs: tuple[str, ...]
    evidence: str

@dataclass(frozen=True)
class BeliefAboutOtherSnapshot:
    records: tuple[OtherMindRecord, ...]
    active_predictions: tuple[SocialPrediction, ...]
    control_signal: float
    description: str

@dataclass(frozen=True)
class PreferenceAboutOtherSnapshot:
    records: tuple[OtherMindRecord, ...]
    active_predictions: tuple[SocialPrediction, ...]
    control_signal: float
    description: str
    action_forecasts: tuple[PreferenceActionForecast, ...] = ()
    action_outcome_evidence: tuple[PreferenceActionOutcomeEvidence, ...] = ()
    forecast_settlements: tuple[PreferenceActionForecastSettlement, ...] = ()
    action_outcome_mutation_receipts: tuple[PreferenceActionOutcomeMutationReceipt, ...] = ()
```

`IntentAboutOtherSnapshot`、`FeelingAboutOtherSnapshot` mirror the common frozen record
contract。`PreferenceAboutOtherSnapshot` 额外拥有 P2 的行动前候选动作预测 readout；这不是
新的 owner，也不能由 evaluator 或 renderer 从 records 重新拼装。

### P2a：候选动作预测公共契约

P2a 只冻结公共交换，不新增行为。`PreferenceActionForecast` 必须同时携带：

- `decision_id` / `interlocutor_id` / `issued_turn`；
- 至少两个候选 `SocialActionCandidatePrediction`；
- 每个候选动作在**同一有序 typed outcome vocabulary** 上的归一化概率分布；
- owner 选出的 `recommended_action_id`、置信度、证据与 `source_record_ids`。
- 可选 `RelationshipConditionReadout`；它必须绑定当前 observation SHA-256 与冻结 reader artifact，
  由 owner 校验后随 forecast 发布，consumer 不得从 evidence 文本重建。

`PreferenceAboutOtherSnapshot` 校验所有 source record 都由自己发布、属于同一 interlocutor，
且 `source_turn <= issued_turn`。forecast 本身严格停在行动前，不得包含 observed outcome、
expected action、evaluation、reward、PE 或 credit。关系 vertical 负责冻结具体 action/outcome
枚举；`vz-contracts` 只拥有 domain-neutral 的公共分布形状，避免反向依赖。

兼容与接线边界：字段默认 `()`，所以旧 snapshot 构造和当前产品行为不变。P2b 允许
`PreferenceAboutOtherModule` 在独立 development lane 中填充它，但必须同时注入 frozen
`PreferenceActionForecastRequest` 与 `PreferenceActionForecastRuntime`，且 owner 必须是
`WiringLevel.SHADOW`；缺一项或尝试 ACTIVE 都 fail loudly。collaborator 只返回非 owning
`PreferenceActionForecastProposal`，owner 校验 exact action/outcome surface、ACTIVE
source records 与 interlocutor 后，亲自绑定 forecast id / decision / scope / turn lineage。
P2b 阶段 expression、planner、`social_prediction_error`、credit 与 steering 不消费；不存在
consumer sidecar fallback，owner 未发布就视为没有 forecast。

### P2c：v3-only 多 session owner probe

`relationship_p2_development_v1.json` 只投影已冻结 v3 public evidence；truth 位于独立 evaluator
bundle，runtime view 不含 v4、preferred action 或 hidden dynamics。每个 development episode
依次执行四段 history，每段都由 `PreferenceAboutOtherModule` 写 typed record +
`PreferenceActionOutcomeEvidence`，随后 export `SocialRecordStore` 并在新进程语义下 hydrate；
probe session 只读取恢复后的 owner state，不重放 raw history。

`SocialRecordStore` schema v2 曾增加 `preference_action_outcomes /
preference_action_forecasts / preference_forecast_settlements`，schema v3 增加 P4.2 mutation
receipt/tombstone；当前 schema v4 继续读取 v1/v2/v3，并持久化 forecast 的可选命名条件 readout，
export 一律写 v4。bounded forecast runtime 使用共享 semantic embedding 比较当前 observation 与
owner-published typed past observations，不使用关键词/正则。该 probe 证明 persistence 与 owner
readout 机械闭合，不是 P2 formal 或 Readable 资格结论。

### P2d：命名抽象条件 readout

`PrototypeRelationshipPreferenceForecastRuntime` 通过注入的冻结 text embedder，将当前
observation 与内容寻址的 condition prototypes 做 cosine 比较。它只提交
`RelationshipConditionReadout` proposal：命名 condition、置信度、归一化 margin、全部候选
分数、reader artifact id 与当前 observation hash；`PreferenceAboutOtherModule` 仍是唯一 writer，
并拒绝 source hash 不等于本次 forecast request 的 proposal。artifact 同时绑定 embedding model
id、weights SHA-256、prototype 文本、temperature 与 similarity，禁止无 lineage 的 backend 漂移。

独立的 `FrozenLinearRelationshipConditionReaderArtifact` v2 不替换上述 prototype v1。它以内容
地址绑定 embedding model source / revision / weights、sentence-transformers runtime、width、
严格有序 labels、condition-only corpus artifact/raw digest、group-split artifact、确定性
centroid solver/version，以及逐类 example count / example-id digest、完整 centroid / coefficient /
bias 的 canonical finite float-hex。offline builder 只接受预先标注的 embedding rows 与这些
预声明 pin，对每行先做 unit normalization，再生成无超参数的 unit-normalized class-centroid
linear rule；输入 shape 没有 action、outcome、PE、credit、evaluation 或 judge。inference-only
runtime 直接校验既有 public semantic embedder 的标准 identity 属性，只消费当前 public text，
不在线拟合，并继续发布既有 immutable `RelationshipConditionReadout`（全部候选分数、margin、
confidence、reader artifact id 与 source hash）。artifact owner 同时提供 exact-key
`from_payload` 与 canonical `to_json/from_json`；fresh child 必须经该 loader 重算 artifact id，
并拒绝 duplicate key、非 canonical UTF-8 bytes、额外/缺失字段或参数 shape 漂移，consumer
不得复制解析 schema。

reader 看不到 evaluator、expected action、未来 outcome、PE、credit 或 judge。已见 v3 根因诊断
中，默认字符哈希 seam 为 `4/12`，冻结 BGE-M3 backend 为 `12/12`、六对 mirror 全部双边正确；
这只证明正式 owner 链可以发布/恢复命名 readout，并定位旧失败为 semantic backend，不是 fresh
held-out、P2 formal 或 Readable 通过。v2 当前也只建立 reader mechanism：尚未 qualification、
尚未接入 campaign，不能据此声称 Readable 成立。下一证据仍须由 P1m 在首次打开前冻结生成配方
与 artifact。

### P3：exact settlement 与 PE-derived action credit

`dialogue_external_outcome` 只有同时携带 `session_scope / action_turn_index / forecast_id /
decision_id / action_id` 时才具有 relationship forecast join。preference owner 只结算同 session、
同 decision、同 action 的 pending forecast；每个 forecast 至多一次，未知或冲突 join fail loudly。
`PreferenceActionForecastSettlement` 发布 predicted probability、NLL、evidence confidence、
expected / observed utility 与 signed utility PE，并产生 matching `SocialPredictionError`。
这段结算数学属于 preference owner；owner 以纯函数
`settle_preference_action_forecast(forecast=..., evidence=...)` 发布同一推导边界。下游证据 consumer
必须用该 owner API 从冻结 forecast 与 exact external evidence 重算并要求 settlement 全对象相等，
不得复制概率、utility 或 PE 数学，也不得只核对 join 字段后接受协同伪造的 settlement。
需要验证跨 session durable transition 的 consumer 还必须调用 owner 的
`replay_preference_action_forecast_settlement_persistence(...)`：它从 pre-owner persistence 与 exact
external/owner evidence 重放同一 generic pending 恢复、record/outcome merge window、forecast 消费和
settlement append，返回唯一 canonical post persistence。只把 post persistence 与同拍 public snapshot 对齐
不足以证明合法 transition，因为二者可以一起删除或注入旧历史。
`SocialRecordStore` 另以 `social_record_store_persistence_sha256(...)` 对 canonical current-schema export 的
owner name/schema 与完整 opaque payload 做版本化内容寻址；consumer 只能调用该 owner API，不得解释 payload。
需要固定 pre-state identity 的 executor command/receipt 必须绑定这个 hash，防止 snapshot 与 persistence
协调重封后仍沿用旧 receipt ID。description 是非程序化日志字段，不进入 state identity。

dedicated action credit 还必须精确找到 `social-pe:<settlement_id>`，然后才计算：

```text
credit_value = signed_utility_prediction_error × evidence_confidence
level = relationship_action_prediction_error
track = SELF
```

evaluation、oracle、human anchor 与七日 continuity 均不能成为此 credit 的来源。Brain facade
只把 frozen preference/social-PE snapshots 交给 cognition derivation；lifeform/service 不重建
owner 隐状态。gate 与 temporal SHADOW contract 见
[`relationship-intelligence-closed-alpha.md`](../relationship-intelligence-closed-alpha.md)。

### P4.2：preference-action correction/redaction

用户纠正与删除只由 `PreferenceAboutOtherModule` 解释。输入
`PreferenceActionOutcomeMutation` 使用 target evidence 的 canonical SHA-256 做 optimistic
concurrency；纠正只能替换 typed observation/outcome/reaction，不能改 interlocutor、真实暴露
action 或 source turn。删除同时移除 paired `OtherMindRecord`、typed outcome、pending ToM
prediction 和引用该 record 的 pending action forecast。owner 发布 frozen
`PreferenceActionOutcomeMutationReceipt`，只含 command/before/after hash、失效 forecast id 与
opaque refs；redaction receipt 同时是跨进程 tombstone，禁止旧证据复活。完全相同命令可幂等
重试，hash drift、未知 target、冲突 mutation id 或 action-lineage drift 都 fail loudly。

该命令不是 outcome settlement、PE、credit、reward 或 evaluation。它改变下一次 owner readout
可见的事实面，但不经过 ModificationGate，也不授权任何 product/expression ACTIVE。P4.2 drill
只证明“纠正/删除 → persistence v4 → 新 forecast reader / anti-resurrection”机械闭合，不是
`recovery_after_correction` formal 效果分数。已经结算的 forecast、PE/credit/gate checkpoint 与
lifeform operational evidence 不在本 owner 的删除权限内；当前又没有可逆 exact lineage，故本
slice 明确不做 turn/action 近似回滚。产品级全域撤回需要后续跨 owner 契约。

Implemented Phase 2 scaffold:

- `volvence_zero.social_cognition.OtherMindRecord`: keyed by `interlocutor_id`, typed by `OtherMindRecordKind`, confidence-bounded, status-bearing, and linked to social PE refs.
- `OtherMindRecordKind`: finite enum for `belief`, `intent`, `feeling`, `preference`.
- `OtherMindRecordStatus`: finite lifecycle enum for `active`, `contested`, `retired`.
- `BeliefAboutOtherSnapshot`, `IntentAboutOtherSnapshot`, `FeelingAboutOtherSnapshot`, `PreferenceAboutOtherSnapshot`: distinct frozen snapshot contracts that reject records of the wrong kind.
- `BeliefAboutOtherModule`, `IntentAboutOtherModule`, `FeelingAboutOtherModule`, `PreferenceAboutOtherModule`: empty SHADOW owner scaffolds registered in final wiring. They establish ownership without changing response assembly / planner / renderer behavior.
- Explicit proposal path: ToM owners can consume an explicitly injected `SemanticProposalRuntime` and convert accepted proposals into `OtherMindRecord` with the owner-specific kind. Final wiring does not pass the generic semantic runtime into ToM owners by default, so no raw-text / NoOp / broad runtime accidentally becomes a ToM classifier.
- Evidence probe: `tests/test_social_tom.py` includes an artificial false-belief + preference-conflict probe proving belief records and preference records stay in separate owners and retain distinct `OtherMindRecordKind` values.
- Diagnostic downstream visibility: when ToM owners are explicitly ACTIVE, `response_assembly.semantic_record_counts` includes `belief_about_other` / `intent_about_other` / `feeling_about_other` / `preference_about_other` counts. Planner and renderer still do not consume these snapshots.
- ACTIVE owner predictions are forwarded by the separate
  `SocialPredictionAggregateModule` into `social_prediction`; SHADOW owner
  predictions remain outside the active dependency graph. Therefore a
  SHADOW → ACTIVE matched-control promotion is expected to change
  `social_prediction` when the promoted ToM owner publishes a typed
  `active_prediction`. This is an explicit owner dependency, not
  run-to-run nondeterminism or renderer-side reconstruction.
- Evidence report artifact: `lifeform_evolution.run_social_cognition_evidence()` summarizes T1-T3 gates for ToM owner contract, explicit proposal path, and false-belief / preference separation.
- CLI artifact: `lifeform-bench --social-cognition-evidence-report` prints the report, and `--social-cognition-evidence-json PATH` writes the T1-T3 payload.
- Structured LLM runtime: `LLMToMProposalRuntime` consumes JSON array output and emits typed `SemanticProposal`s targeted at belief / intent / feeling / preference owners. Malformed JSON falls back, provider exceptions propagate, and low-confidence records are dropped before owner mutation.
- Owner hardening: ToM owners drop proposals below the owner confidence floor and ignore wrong-target proposals, preserving SHADOW safety and avoiding broad classifier leakage.
- Evidence gates T4/T5: social cognition evidence now covers structured ToM runtime path and affect/preference separation through `lifeform_evolution.run_social_cognition_evidence()`.
- COG-2 最小收敛切片新增 `ToMInterlocutorRecordCount` 与 `tom_record_counts_by_interlocutor(...)`，从四个 ToM owner 的 public snapshots 聚合 per-interlocutor 记录计数。benchmark / family report 应消费该 typed readout，不得遍历 owner 私有状态或从 raw text 重建"谁有多少 ToM 记录"。
- 2026-07-13 social-learning slice 1: 四个 ToM owner 不再发布空
  `active_predictions`。每个 accepted `OtherMindRecord` 生成 owner-authored
  `SocialPrediction`（`BELIEF_ABOUT_OTHER` / `INTENT_ABOUT_OTHER` /
  `FEELING_ABOUT_OTHER` / `PREFERENCE_ABOUT_OTHER`），`SocialPredictionAggregateModule`
  在 ToM owners 之后运行并只转发这些 typed predictions（不读 records 重建）。
  v1 prediction-only：PE-weighted promote/retire 与 settled ToM PE 是后续切片。

## 与其他能力域的关系

- R16 supplies `interlocutor_id` and audience / subject identity.
- R18 uses ToM state to distinguish addressee vs subject role consequences.
- R19 common ground uses belief owner outputs but does not own beliefs.
- R20 group state may aggregate ToM evidence but cannot rewrite individual ToM owners.
- R-PE / credit receive typed ToM prediction outcomes.
- Environment Interface supplies event provenance and conversational frame for ToM proposals; ToM owners reconcile proposals into learned state.

## 变更日志

- 2026-08-24: `RelationshipConditionReadout` 增加 contracts-owner 公共 strict codec；qualification、
  persistence 与 evidence consumer 复用同一 frozen value shape，禁止各自重建 reader 隐状态或私有 decoder。
- 2026-08-22: P2d named condition reader。新增 frozen `RelationshipConditionReadout` 与绑定
  embedding weights/prototypes 的内容寻址 artifact；非 owning semantic collaborator 提 proposal，
  preference owner 校验 current-observation hash 后发布。`SocialRecordStore` export 升为 v4，继续
  hydrate v1-v3。seen-v3 结果仅作根因诊断，P2 formal/Readable 保持关闭。
- 2026-08-22: P4.2 preference-action mutation owner。新增 optimistic-hash correction/redaction
  command、不可变 receipt 与（现由 `SocialRecordStore` v4 保存的）tombstone；owner 原子更新 record/outcome/
  pending prediction/forecast，跨恢复阻止旧证据复活。独立 P4.2 drill 不读 evaluator、不调用模型，
  不改变 production wiring 或 P4.1 效果口径。
- 2026-08-22: P2c/P3 closure。`SocialRecordStore` v2 持久化 typed action history、pending
  forecasts 与 settlements；v3-only development probe 跨四次恢复后发布独立 forecast。P3
  通过 `dialogue_external_outcome` exact join 结算 probability/NLL/utility PE，action credit
  必须匹配 owner-authored social PE，禁止 evaluation 回灌。P2 formal 仍关闭。
- 2026-08-21: P2b owner-producer slice。`PreferenceAboutOtherModule` 新增成对的
  `PreferenceActionForecastRequest` / `PreferenceActionForecastRuntime` 可选注入，只在
  SHADOW 运行；runtime 仅提 proposal，owner 校验闭合 action/outcome surface 与自身 ACTIVE
  record lineage 后发布正式 forecast。`build_final_runtime_modules` /
  `run_final_wiring_turn` 提供显式注入入口；ACTIVE aggregate 看不到 SHADOW forecast。
- 2026-08-21: P2a contract-only slice。`PreferenceAboutOtherSnapshot` 新增默认空的
  `action_forecasts`，并冻结 `SocialActionOutcomeProbability`、
  `SocialActionCandidatePrediction`、`PreferenceActionForecast` 三个不可变公共 value。
  只允许 `PreferenceAboutOtherModule` 发布；当前无 producer、无新 slot、无 evaluator
  label、无 PE/credit/steering 回灌，产品行为保持不变。
- 2026-07-28: 修正 `feeling_about_other` matched-control 的依赖证据：
  `social_prediction` 是 ACTIVE ToM prediction 的正式 aggregate consumer；
  contract test 同时断言 SHADOW 无转发、ACTIVE 精确转发一条 owner-authored
  `FEELING_ABOUT_OTHER` prediction，避免把预期的因果变化误判为非确定性漂移。
- 2026-07-13: social-learning slice 1。`SocialPredictionKind` 新增四个 ToM
  prediction kind；`Belief/Intent/Feeling/PreferenceAboutOtherModule` 从
  accepted `OtherMindRecord` 发布 `active_predictions`；final wiring 将
  `SocialPredictionAggregateModule` / `SocialPredictionErrorModule` 移到 ToM
  owners 之后，aggregate 只转发 owner-authored ToM predictions 与 memory
  signals。测试：`tests/test_social_tom.py`（owner prediction + aggregate
  forwarding）与 `tests/test_final_wiring.py`。
- 2026-05-22: COG-2 最小切片。新增 per-interlocutor ToM record count helper，作为 wrong-person / witness / private-leakage 场景的稳定 evidence surface；不新增 owner，不改变四个 ToM snapshot shape。
- 2026-05-02: R17 Phase 2 slice 8 landed: `LLMToMProposalRuntime` structured JSON proposal path with low-confidence filtering, owner hardening, final-wiring diagnostics, and T4/T5 evidence gates.
- 2026-05-02: R17 Phase 2 slice 5 landed: ToM owner record counts surface in `response_assembly.semantic_record_counts` as diagnostics only; no planner / renderer consumption.
- 2026-05-02: R17 Phase 2 slice 6 landed: social cognition evidence report artifact for T1-T3 ToM owner separation gates.
- 2026-05-02: R17 Phase 2 slice 7 landed: `lifeform-bench` CLI support for social cognition evidence report stdout and JSON artifact.
- 2026-05-02: R17 Phase 2 slice 4 landed: first ToM evidence probe for false-belief / preference-conflict owner separation plus evidence-program T1-T3 claim wording.
- 2026-05-02: R17 Phase 2 slice 3 landed: explicit ToM proposal runtime path for owner modules, with no default final-wiring consumption and no raw-text classifier behavior.
- 2026-05-02: R17 Phase 2 slice 2 landed: four ToM SHADOW owner scaffolds publish empty snapshots in final wiring, while response assembly / planner remain unchanged.
- 2026-05-02: R17 Phase 2 slice 1 landed in `vz-contracts`: `OtherMindRecord` / kind + status enums / four ToM snapshot contracts with kind validation and empty SHADOW-compatible snapshots.
- 2026-05-02: 补充 Environment Interface 依赖：ToM proposal 进入 owner 前必须带 canonical event / role context。
- 2026-05-02: 初始 draft，冻结 ToM owner decomposition as Social Cognition Learning Layer Phase 2。
