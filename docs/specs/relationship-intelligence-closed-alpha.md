# Relationship Intelligence P2–P4 Closed-Alpha Contract

> Status: P2-SHADOW（含 P2d 命名条件 reader）、P3 mechanism、P4 product shell、P4.1 longitudinal canary、P4.2 preference-action 纠删 owner 与 P4.3 named-reader transmission 工程包已落地；P2 formal、真人 typing qualification、长 horizon / 多 session 真人 pilot 与 production ACTIVE 均未通过
> Last updated: 2026-08-22
> Owner boundary: `preference_about_other` / `social_prediction_error` / `self_temporal` / `dialogue_external_outcome` 保持各自唯一 owner；`lifeform-service` 只编排产品入口与证据

## 1. 目的与诚实边界

本 slice 把关系智能 MVP 计划中的 P2、P3 与 P4 工程链闭合：系统可以从已持久化的
typed 关系经历，在行动前发布候选动作结果分布；以 PE 派生 credit 学习何时采用该建议；
并通过 closed-alpha HTTP、次日 followup 与既有关系记忆 console 暴露产品入口。

这不等于已经证明 Volvence 的关系智能优势。当前仍然成立的限制是：

- P1m 第一次 qualification 已以 `prompt_steelman_baseline_too_weak` 终局失败并关闭场景版本化，
  所以 P2 仍只能称 development SHADOW；human-anchor 也尚未完成；
- 没有真实 `RelationshipOutcomeTypingQualification` artifact，真人自由文本 outcome 只能
  以 `unknown` 收集，不能进入 PE；
- relationship action advisory 默认且当前只能 SHADOW，不能改变用户可见表达；
- SHADOW 建议没有真实暴露时，后续用户结果不能被冒充为该建议的 causal outcome；
- production ACTIVE、长 context、多 session 真人效果和四能力总主张均未成立。

## 2. 四能力轴与唯一 owner

| 能力轴 | 本 slice 的机制 | 唯一 owner / 时间尺度 | 当前边界 |
|---|---|---|---|
| Appendable | owner records、typed action outcome、pending forecast、settlement、命名条件 readout 与纠删 tombstone 进入 `SocialRecordStore` v4，可跨进程 hydration | `preference_about_other` / session-medium | P2-development/P4.2 drill 已证明机械恢复，不等于 held-out 效果成立 |
| Readable | 行动前由 owner 发布 `PreferenceActionForecast`；可附带绑定当前 observation hash 与 reader artifact 的 `RelationshipConditionReadout`，命名抽象条件并给出候选分数 | `PreferenceAboutOtherModule` / online-fast readout | P4.3 在 seen-v3 上证明 readout 可因果传到动作/outcome；因 post-selection 与失灵基线，formal 仍关闭 |
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
- `PreferenceActionOutcomeMutation` / `PreferenceActionOutcomeMutationReceipt`：用户纠正或删除命令及 content-safe 审计/tombstone；
- `RelationshipConditionReadout`：当前 observation 上的命名条件、置信度、margin、候选分数、reader artifact 与 source hash；
- `PreferenceActionForecast`：行动前候选动作分布；
- `PreferenceActionForecastSettlement`：一次 forecast 的 exact post-action 结算。

### 3.2 Owner 发布与恢复

`PreferenceActionForecastRuntime` 只是 collaborator，只能读取 frozen request、同一
interlocutor 的 ACTIVE owner records 与 owner-published typed past outcomes，并返回
`PreferenceActionForecastProposal`。`PreferenceAboutOtherModule` 校验 action/outcome
surface、source record lineage、turn、decision 与 session scope 后才发布正式 forecast。

`SocialRecordStore` persistence schema v4 保存：

- `preference_action_outcomes`；
- `preference_action_forecasts`；
- `preference_forecast_settlements`。
- `preference_action_outcome_mutation_receipts`。

schema v1/v2/v3 仍可 hydrate；旧 forecast 的 condition readout 为空，旧 mutation receipt 默认为空；
export 一律写 v4。P2-development 使用 v3-only
公开轨迹，四段历史跨四次恢复后再做独立 probe，不读取 v4 truth、evaluator、PE、credit、
steering 或 expression。具体 bounded runtime 使用 semantic similarity，不做关键词/正则路由。

### 3.3 P2d 命名条件 reader

`PrototypeRelationshipPreferenceForecastRuntime` 是 `preference_about_other` 的非 owning
collaborator。它用注入的冻结 embedding backend 将当前 observation 与内容寻址的抽象条件
prototype 比较，发布 `RelationshipConditionReadout` proposal；owner 校验 readout 的
`source_observation_sha256` 必须等于本次 request 后，才把它随正式 forecast 发布。artifact
绑定 embedding model id、权重 SHA-256、cosine 规则、prototype 文本 hash 与 temperature；
consumer 不得从 forecast evidence 字符串重建 condition。

该 reader 不接收 evaluator truth、expected action、未来 outcome、PE、credit 或 judge。已打开
v3 上的根因诊断为：旧字符哈希 seam `4/12`，同 owner/persistence/forecast 算法换成冻结
BGE-M3 semantic backend 后 `12/12` 且六对 mirror 全部双边正确；reader artifact id 为
`f8a54447…073d`。这只说明原弱点位于语义 backend，并证明命名 readout 可以沿正式 owner
契约发布和恢复；数据已经见过，不能作为 P2 formal、Readable 或四能力证据。P1m fresh
generated 首次 qualification 随后得到 structured-state `46/48` correct、`24/24` pair flip，
但 prompt/RAG 均为 `24/48`、0 flip，整体 report `9580ddff…fc56` 按冻结门判
`prompt_steelman_baseline_too_weak`。因此这份 fresh structured 结果只能作为命名 readout 的
强方向性证据，不能在失灵基线之上宣称 P2 formal、Volvence advantage 或完整 Readable；
后续不得回改本 P1m 配方追分。

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
- persistence：v1/v2 `SocialRecordStore` 继续 hydrate；回滚旧代码前必须先确认其不会覆盖 v3
  owner snapshot。不得把 v3 文件手工降成旧 schema；redaction tombstone 不能因停用 consumer 而删除。
- production ACTIVE 退出条件：独立 promotion artifact 授权、实际 action exposure 可证明、
  长 horizon / 多 session safety 与 outcome evidence 通过；当前均未满足。

## 10. 验证与可声称范围

当前测试只证明机制与防火墙：owner/persistence、exact join、PE-only credit、gate controls、
same-turn SHADOW advisory、qualification/tamper、unknown/review、causal exposure、evidence
separation/idempotency、followup session isolation。它们不构成真实用户效果实验。

允许声称：P2–P4 工程链和 collection-only closed-alpha 壳已落地，默认不改变表达。

禁止声称：P2 formal PASS、Readable 已证明、Volvence 优于 full-history prompt/RAG、真人
typing 已 qualified、steer action 已在线学习、production ACTIVE 或超级关系智能已经成立。

## 11. P4.1 longitudinal canary（Lab-only）

P4.1 把下一轮测试冻结成一个普通人可以理解的完整故事：**同一个人被排除时希望你别走，
被别人替他做主时又希望你把空间和决定权还给他**。家庭、医疗、朋友、工作、社区与亲密
关系只是不同表面；八次决策交替出现两类关系结构，最后再回到早期表面，检查跨域迁移、
需求反转和隔一段时间后的恢复，而不是检查某句提示词是否碰巧说得好。

冻结协议位于
`lifeform_domain_emogpt/lab_protocols/relationship_p4_longitudinal_canary_v1.json`：

- bundled development 只使用已见的 `relationship_transfer_v3`，两条轨迹各含 4 次增量
  onboarding 与 8 次决策，每个 session 后都 export/hydrate；它们是工程 fixture，不是两名
  独立真人，也不能形成统计证据；
- formal pilot 门槛为至少 20 名独立 subject、每人 8 次决策、末次公开 context 至少 32768
  tokens，并要求相同冻结 substrate / generation config；
- 竞争臂固定为 Qwen steelman full-history、Qwen steelman selective-RAG、Volvence closed-loop
  和 Volvence typed-noop control。每臂拥有自己的反应式轨迹，必须在 environment outcome 前
  冻结 outcome-free `relationship-p4-canary-arm-preaction.v1`（typed action、候选结果分布、
  model/prompt/generation lineage、context tokens、latency 与 response hash）；
- 当前 P1k/P1m 顺序仍禁止产生新 Qwen output，所以 prepare artifact 明确记录
  `model_output_count=0`，两个 Qwen 臂是 `blocked`，不得拿缺席基线做优势结论。

### 11.1 隔离的 Lab ACTIVE

P4.1 不修改 P3/P4 的 production advisory builder；后者仍固定
`active_authorized=false`。Lab 根据 protocol hash、v3 fingerprint、gate artifact/version、
subject/session 上限生成内容寻址的 `relationship-p4-lab-active-authorization.v1`。它只允许
非 oracle、非 evaluator 的 typed action 由 `self_temporal` 以 `APPLIED` 进入
`ReactiveRelationshipEnvironment`，同时硬编码：

- `environment_consumer_only=true`；
- `expression_authorized=false`、`production_authorized=false`；
- `evaluation_feedback_to_learning=false`、`oracle_action_authorized=false`。

因此这个 ACTIVE 只补上离线因果链的“实际动作暴露”，不能进入回复表达或产品 runtime。
产品回滚边界仍是原有 `relationship_action_advisory=SHADOW/DISABLED`，不受本 Lab 包影响。

### 11.2 当前工程 canary 读数

2026-08-22 在默认 bounded owner reader 上运行两条 v3 seen fixture：closed-loop 共得到
`10/16` positive typed outcomes、`3/16` preferred-action matches、`3/14` reversal matches；
typed-noop control 为 `7/16`、`0/16`、`0/14`。closed-loop 的 16 次更新全部来自 exact
settlement → owner-authored social PE → dedicated credit，no-op control 计算 credit 但不把它
交给 gate；32 次动作均由 self-temporal 报告 `APPLIED`，每臂每 subject 有 11 次显式恢复。

这组读数只说明闭环和对照能真实分叉，并同时暴露出默认 reader/gate 的当前弱点：动作匹配与
反转仍很低。由于数据已见、subject 非独立、没有 Qwen 输出、没有自然语言表达暴露，这不是
Volvence advantage，更不是 Readable/Learnable/Steerable 或四能力的正证据。正式效果结论
必须等 P1k/P1m 放行后按冻结的 20 人、32K context、四臂协议另行产生。

### 11.3 P4.2 correction/redaction owner drill

P4.2 已在 `preference_about_other` 唯一 owner 内增加 typed correction/redaction command：调用方
必须提交 target evidence 的 expected SHA-256；owner 拒绝 stale overwrite，且纠正不能改写
interlocutor、真实暴露 action 或 source turn。纠正同步更新 paired record/outcome，并失效引用它
的 pending forecast；删除同时移除 paired record、outcome、pending ToM prediction 和 pending
forecast。结果 receipt 不保存被删原文，`REDACT` receipt 作为 persistence v4 tombstone 跨恢复
阻止旧 evidence id 复活。该命令不产生 PE/credit，也不读取 evaluation/judge。

独立 `--run-mutation-drill` 使用首条 seen v3 subject 做七次显式恢复：先冻结 pending forecast，
纠正一条 onboarding evidence，验证同 turn bounded reader 读取 correction hash；恢复后删除另一条，
验证持久化 payload 不含其 observation/reaction，并主动重放旧证据确认 tombstone fail loudly。
当前 drill 为 `passed=true / model_output_count=0 / evaluator_truth_used=false /
formal_evidence_authorized=false`。它补齐的是 owner 纠删机械契约；`recovery_after_correction`
仍须在未来 formal 独立 subject 上按冻结 secondary measure 计分，不能用本 drill 冒充效果通过。
此外，P4.2 不反向重写已经结算的 forecast、PE/credit、gate checkpoint 或 lifeform operational
evidence：这些属于其他 owner，且当前缺少从 action-outcome record 到它们的可逆 exact lineage。
完整产品撤回需要下一收敛包逐 owner 注册；禁止按 turn/action 猜测 join。

### 11.4 P4.3 named-reader transmission（development-only）

P4.3 固定 P4.1 的同一 seen-v3 两条 4+8 session fixture、同一 owner/hydration、同一
`ALWAYS` gate、同一 Lab ACTIVE 与 reactive environment，只替换 forecast collaborator：旧
bounded reader 对比 P1m 内容寻址 named reader。两臂都计算 settlement/PE/credit，但都不把
credit 交给 gate，因此本包只问一个 Readable 因果问题：命名 readout 是否真的改变 typed action，
并进一步改变 reactive outcome。

本机 frozen BGE 重跑并完成 artifact 终检后，legacy arm 为 `6/16` preferred-action match、
`9/16` positive outcome、`6/14` reversal match；named arm 为 `16/16`、`16/16`、`14/14`，
且 `16/16` forecast 都携带正确 reader lineage。匹配决策中有 `10/16` action change 与
`9/16` outcome change；每个 subject 每臂仍有 11 次显式恢复，gate update 均为 0。报告
artifact 为 `e7bbc914…70bd`，判
`named_reader_transmission_observed_development_only`。

这条证据比“readout 字段存在”更强：它证明 owner 发布的命名状态沿正式动作链产生了可观察
差异。但组件是在看到 P1m 方向性结果后选择，fixture 已见且只有两条，P1m prompt 基线失灵，
所以它不能修复 P1m、授权 P2 formal、证明 Volvence advantage 或完整 Readable。下一包必须
固定 named reader，只切 PE-derived credit 是否回写 learned gate，隔离 Learnable；不得再调整
reader、场景或 prompt。

运行当前源码（避免命中外部 editable checkout）：

```bash
./start_relationship_p4_canary.sh --prepare
./start_relationship_p4_canary.sh --run-mutation-drill
./start_relationship_p4_canary.sh --run-development --format markdown
./start_relationship_p4_canary.sh --run-development --output /tmp/relationship-p4-canary-report.json
python scripts/run_relationship_lab_p4_named_reader.py --validate-existing
```

PowerShell 使用同名 `.ps1`。输出采用 create-only canonical JSON；路径已存在时 fail loudly。
