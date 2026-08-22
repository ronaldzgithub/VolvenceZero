# Relationship Intelligence P2–P4 Closed-Alpha Contract

> Status: P2-SHADOW、P3 mechanism 与 P4 product shell 已落地；P2 formal、真人 typing qualification、长 horizon / 多 session 真人 pilot 与 production ACTIVE 均未通过
> Last updated: 2026-08-22
> Owner boundary: `preference_about_other` / `social_prediction_error` / `self_temporal` / `dialogue_external_outcome` 保持各自唯一 owner；`lifeform-service` 只编排产品入口与证据

## 1. 目的与诚实边界

本 slice 把关系智能 MVP 计划中的 P2、P3 与 P4 工程链闭合：系统可以从已持久化的
typed 关系经历，在行动前发布候选动作结果分布；以 PE 派生 credit 学习何时采用该建议；
并通过 closed-alpha HTTP、次日 followup 与既有关系记忆 console 暴露产品入口。

这不等于已经证明 Volvence 的关系智能优势。当前仍然成立的限制是：

- P1m consumer / human-anchor 仪器资格尚未过门，所以 P2 只能称 development SHADOW；
- 没有真实 `RelationshipOutcomeTypingQualification` artifact，真人自由文本 outcome 只能
  以 `unknown` 收集，不能进入 PE；
- relationship action advisory 默认且当前只能 SHADOW，不能改变用户可见表达；
- SHADOW 建议没有真实暴露时，后续用户结果不能被冒充为该建议的 causal outcome；
- production ACTIVE、长 context、多 session 真人效果和四能力总主张均未成立。

## 2. 四能力轴与唯一 owner

| 能力轴 | 本 slice 的机制 | 唯一 owner / 时间尺度 | 当前边界 |
|---|---|---|---|
| Appendable | owner records、typed action outcome、pending forecast 与 settlement 进入 `SocialRecordStore` v2，可跨进程 hydration | `preference_about_other` / session-medium | P2-development 已证明机械恢复，不等于 held-out 效果成立 |
| Readable | 行动前发布 `PreferenceActionForecast`，包含每个候选动作的 typed outcome 分布 | `PreferenceAboutOtherModule` / online-fast readout | formal 关闭；不得称已读懂未觉察抽象关系结构 |
| Learnable | exact outcome join → owner settlement → `SocialPredictionError` → dedicated action credit | `social_prediction_error` 与 credit owner / online-fast→session-medium | evaluation、judge 与 human anchor 都不能进入更新 |
| Steerable | gate 只选 `{noop, steer}`，输出 typed `TemporalActionAdvisoryProposal` 给 `self_temporal` | vertical gate + `self_temporal` / online-fast | 默认 SHADOW；未授权建议绝不改变表达 |

闭环只有在真实动作暴露后才完整：

```text
owner history
→ pre-action forecast
→ bounded gate decision
→ self_temporal advisory
→ actually exposed action
→ later typed external outcome
→ exact settlement / social PE
→ PE-derived credit
→ next gate decision
```

P4 当前只有 `baseline_noop_exposed` 满足“actually exposed”。`steer` 建议仍是
`shadow_counterfactual`，因此即使 typing qualification 通过也不得结算或进入训练候选。

## 3. P2：多经历关系动力学 readout

### 3.1 冻结动作与 outcome surface

Companion vertical 唯一拥有以下 domain vocabulary：

- action：`stay_present_without_probe`、`respect_space_with_return_option`、
  `neutral_noop`；
- forecast outcome：`helped`、`felt_heard`、`missed`、`over_directive`。

公共 shape 位于 `vz-contracts.social_cognition`，不反向依赖 vertical：

- `PreferenceActionOutcomeEvidence`：已经发生、已由 owner 接收的 typed 历史；
- `PreferenceActionForecast`：行动前候选动作分布；
- `PreferenceActionForecastSettlement`：一次 forecast 的 exact post-action 结算。

### 3.2 Owner 发布与恢复

`PreferenceActionForecastRuntime` 只是 collaborator，只能读取 frozen request、同一
interlocutor 的 ACTIVE owner records 与 owner-published typed past outcomes，并返回
`PreferenceActionForecastProposal`。`PreferenceAboutOtherModule` 校验 action/outcome
surface、source record lineage、turn、decision 与 session scope 后才发布正式 forecast。

`SocialRecordStore` persistence schema v2 保存：

- `preference_action_outcomes`；
- `preference_action_forecasts`；
- `preference_forecast_settlements`。

schema v1 仍可 hydrate，新增集合默认为空；export 一律写 v2。P2-development 使用 v3-only
公开轨迹，四段历史跨四次恢复后再做独立 probe，不读取 v4 truth、evaluator、PE、credit、
steering 或 expression。具体 bounded runtime 使用 semantic similarity，不做关键词/正则路由。

## 4. P3：exact PE credit 与择时 gate

### 4.1 Exact settlement

`DialogueExternalOutcomeEvidence` 的 relationship join 必须同时具备：

- `session_scope` 与 `action_turn_index`；
- `forecast_id`、`decision_id`、`action_id`；
- 已冻结四类 outcome 之一。

`PreferenceAboutOtherModule` 只结算仍 pending、session / decision / action 完全一致的 forecast。
每个 forecast 最多结算一次；未知 forecast、冲突 evidence 或 action drift 均 fail loudly。
settlement 发布 predicted probability、NLL、evidence confidence、expected / observed utility 与
有符号 utility PE。utility 只定义在冻结 surface 上：`helped/felt_heard=+1`，
`missed/over_directive=-1`，signed PE 为 `(observed - expected) / 2`。

### 4.2 Credit 与 gate

`derive_preference_action_forecast_credit_records(...)` 必须找到同一 settlement 的
owner-authored `SocialPredictionError`；否则拒绝生成 credit。专用 credit：

```text
level = relationship_action_prediction_error
track = SELF
credit_value = signed_utility_prediction_error × evidence_confidence
```

`RelationshipActionGate`：

- 只读取 frozen forecast 的 confidence、positive mass、相对 noop margin、entropy certainty
  与 typed source support；不读 raw text、truth、evaluation 或 judge；
- `learned` 只消费上述 dedicated PE credit；reader、forecast owner 与 expression executor
  不随 gate update；
- `noop / always / random / oracle` 是显式对照；oracle 必须 `evaluator_only=true`，产品入口拒绝；
- checkpoint 绑定 artifact id/version、参数、update count、已消费 credit 与 pending decision；
  state root 按 subject hash 隔离。

## 5. Temporal advisory 与 causal exposure

`TemporalActionAdvisoryProposal` 只携带 action identity、forecast/decision、policy artifact、
typed rationale 与 evidence refs，不携带 expression 文本。它只允许进入 `self_temporal`：

- `DISABLED`：丢弃 collaborator input；
- `SHADOW`（默认）：snapshot 发布 `SHADOW_RECORDED`，native abstract action 不变；
- `ACTIVE`：只有 `active_authorized=true` 且非 evaluator artifact 才能成为 effective action；
  否则构造期或 owner process fail loudly。

P3/P4 生成的 advisory 固定 `active_authorized=false`。`FinalRolloutConfig` 的单字段
`relationship_action_advisory` 是回滚边界。P4 action audit 必须同时记录 temporal status、
`applied_to_expression`、boundary/consent readout、gate features、policy artifact、rationale 与
evidence hashes。

## 6. P4 产品入口

P4 由 `AlphaServiceConfig.relationship_intelligence_enabled` 显式启用，默认关闭。启用后：

1. `POST /v1/sessions/{session_id}/relationship-turns`：普通自然语言 turn；在 canonical
   turn 前运行 owner forecast 与 gate，canonical response 本身保持原路径，返回去原文的
   typed action audit；
2. `POST /v1/sessions/{session_id}/relationship-outcomes`：只接收 `outcome_text` 与 exact
   forecast / decision / action lineage；客户端不能直接指定 `kind` 绕过 typer；
3. `POST /v1/sessions/{session_id}/relationship-followups/execute-due`：只执行目标 session
   已到期的 followup，继续服从 consent、cooldown、per-session budget 与 tenant gate；
4. 既有 `GET/POST /v1/users/me/relationship-memory...`：展示 owner-authored 理解，支持 keep、
   session-only、delete、rewrite、mark-sensitive 与 no-proactive-mention。

所有入口仍要求 closed-alpha identity 与 session ownership。P4 route 不解析输出文本决定
scene/action/outcome，也不拥有 relationship state。

## 7. 真人 outcome typing 前置门

### 7.1 Runtime result

`LlmStructuredRelationshipOutcomeTyper` 使用 package 内独立 prompt 与 JSON schema，将一条
自由文本映射为：

- `helped | felt_heard | missed | over_directive | unknown`；
- confidence；
- `explicit_report | behavioral_consequence | mixed_or_ambiguous`；
- `needs_human_review`。

实现只调用 structured JSON LLM 并做 exact schema validation；没有 keyword、regex 或
sentiment fallback。额外字段、schema drift、非法 enum 或 `unknown` 未请求 human review
都 fail loudly。

### 7.2 Qualification artifact

没有布尔 PASS 开关。服务只接受 content-hashed
`relationship-outcome-typing-qualification.v1`，并验证：

- `typing_method=llm_structured_output`；
- runtime id 与 `relationship-outcome-typing-result.v1` 精确绑定；
- 至少 3 个独立且唯一的 rater artifacts、隐藏标签；
- 多数一致率 `>= 0.80`；
- typing-anchor agreement 达到预注册阈值；
- `validation_anchor_only=true`、`learning_use_authorized=false`；
- 无关键词/正则路径；支持 `unknown`。

通过 artifact 后，服务启动还必须注入 shared structured-JSON LLM client，且 runtime/schema
与 qualification 精确相同。否则 fail startup。资格未过时，不调用 LLM，outcome 固定
`unknown + needs_human_review=true`，只写隔离 operational evidence。

通过且单样本为 known、不需 review、动作真实暴露后，runtime evidence 使用独立 source
`QUALIFIED_USER_REPORT`，同时携带 qualification id/hash、runtime 与 schema lineage。
它不是用户直接点选 typed label，也不是未资格的 `LLM_PROPOSAL`。

## 8. Evidence、隐私与训练隔离

`RelationshipAlphaArtifactStore` 使用 create-only、canonical JSON 与 SHA-256 校验：

- operational root：action audit 与 outcome receipt；
- training-candidate root：只有每次 outcome 明确授权且 action 确实暴露时才写，必须与
  operational root 物理不同；
- artifact 不保存 raw dialogue、plaintext subject/session identity 或服务器绝对路径；
  对外只返回 content-addressed opaque ref；
- same forecast + same text + same consent 的 retry 返回同一 receipt，不重复 LLM 调用或
  runtime injection；不同文本或不同训练授权视为冲突；
- training candidate 仍标记 `offline_gate_required`，不能直接进入 online update 或共享权重。

## 9. 回滚与迁移退出

- 产品壳：`relationship_intelligence_enabled=false`，三个 P4 route 返回 disabled；普通 turn、
  followup scheduler 与 memory console 原路径不变。
- outcome：移除 qualification path 即回到 collection-only；已有 operational evidence 不删除。
- action：`relationship_action_advisory=DISABLED` 丢弃 advisory；SHADOW 是默认回滚点。
- persistence：v1 `SocialRecordStore` 继续 hydrate；回滚旧代码前必须先确认其不会覆盖 v2
  owner snapshot。不得把 v2 文件手工改成 v1。
- production ACTIVE 退出条件：独立 promotion artifact 授权、实际 action exposure 可证明、
  长 horizon / 多 session safety 与 outcome evidence 通过；当前均未满足。

## 10. 验证与可声称范围

当前测试只证明机制与防火墙：owner/persistence、exact join、PE-only credit、gate controls、
same-turn SHADOW advisory、qualification/tamper、unknown/review、causal exposure、evidence
separation/idempotency、followup session isolation。它们不构成真实用户效果实验。

允许声称：P2–P4 工程链和 collection-only closed-alpha 壳已落地，默认不改变表达。

禁止声称：P2 formal PASS、Readable 已证明、Volvence 优于 full-history prompt/RAG、真人
typing 已 qualified、steer action 已在线学习、production ACTIVE 或超级关系智能已经成立。
