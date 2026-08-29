# Operations Brain v1 产品契约 Spec

> Status: implemented v1
> Last updated: 2026-08-30
> Owner: `lifeform-domain-operations`
> Service projection: `lifeform-service`
> Upstream authority: AutoCompany

## 1. 产品定义与权威边界

Operations Brain 是 AutoCompany 的**有状态运营认知侧车**。它在周期规划、工作排序、容量再平衡、
依赖消解、事故恢复和运营复盘前发布 memory-first Context Pack，并把 AutoCompany 已核验的工作单结果
追加到长期经历。它不是第二个 COO，不拥有 OKR、预算、审批、工作单 SSOT、dispatch 或 division actuator。

| AutoCompany 独占 | Operations Brain 可做 | Operations Brain 禁止 |
|---|---|---|
| OKR 与优先级、预算/容量、authority policy、审批、工作单 SSOT、division dispatch、治理 ledger、最终状态转换和外部动作 | 消费已确认运营事实；跨周期召回；发布不确定性；给出有界 catalog-action 候选；记录 typed work-order outcome；让合格现场聚合结果经既有 PE 链结算 | 读取或修改 AutoCompany 私有 ledger、创建/批准/派发工作单、调用 division、花钱、部署、升级 evidence class、从文本猜路由、执行或推广自己的建议 |

闭环：

```text
AutoCompany confirmed facts + division scope + action catalog + constraints/window
  -> OperationsContextRequest
  -> 普通 Lifeform turn（结算上一批合格 field operation）
  -> identity-scoped Memory retrieval
  -> OperationsContextPackSnapshot [ACTIVE]
       + OperationsAdviceSnapshot [SHADOW, applied=false]
  -> AutoCompany 自己审批并持久化/派发 typed work order
  -> division 执行并返回 typed outcome
  -> OperationsOutcomeReport（含匹配 work_order_ref 的 evidence）
  -> memory + execution_result
       + 仅 field_operation_result 进入 EnvironmentOutcome
  -> 下一次新 Context Pack turn 由 PE owner 结算 lineage
```

## 2. v1 范围

v1 包含：

- 独立 `lifeform-domain-operations` wheel；
- 严格、版本化、frozen、拒绝未知字段的 request/pack/advice/report/receipt；
- ACTIVE Context Pack 与永久 SHADOW、`applied=false` 的 Advice；
- request/outcome 幂等、same-live-session/latest-pack lineage 和有界进程内 ledger；
- identity-scoped 跨 session outcome recall；
- `lifeform-service` 的 `operations` vertical 与两条 session-local HTTP 投影；
- 仅经 Lifeform facade 写 memory、task event 和合格 environment outcome。

v1 不包含 AutoCompany planner/COO、工作单数据库、division adapter、deployment、预算 gate、审批状态机、
新的 kernel slot、token-space RL、在线 base-model 更新或 production ACTIVE advice。Context Pack 接线与
SHADOW advice 只证明机制成立，不证明运营 uplift。

## 3. 冻结公共合同

实现位于 `lifeform_domain_operations.operations_brain_contracts`。金额使用显式三字母币种与整数 minor
units；文本只承载已确认上下文，不承担 action、evidence、route 或 authority 分类。

| 对象 | schema | 核心语义 |
|---|---|---|
| `OperationsContextRequest` | `operations-context-request.v1` | company/cycle/workstream/decision identity、closed decision point、division/action catalog allowlist、typed facts/constraints/window/uncertainties/evidence |
| `OperationsContextPackSnapshot` | `operations-context-pack.v1` | content-addressed ACTIVE pack、opaque live-session lineage、实际召回经历、source/settlement lineage、同拍 PE readout 与独立 SHADOW advice |
| `OperationsAdviceSnapshot` | `operations-advice.v1` | target division + catalog action、成本上限、审批要求、风险/可逆性、前置事实、预测区间、证伪条件和 source lineage |
| `OperationsOutcomeReport` | `operations-outcome-report.v1` | AutoCompany typed decision、work-order lineage、evidence lane、field aggregate verdict 和多目标运营结果 |
| `OperationsOutcomeReceipt` | `operations-outcome-receipt.v1` | content-addressed live-session/memory/event/environment lineage、work-order ref、学习 route 与下一拍结算状态 |

Pack 与 receipt 的 `content_sha256` 不是语言运行时自己的 JSON 字符串哈希。v1 使用
`operations-canonical-value.v1` domain-separated typed encoding：null/bool 有独立 tag，字符串为
UTF-8 + uint32 长度，数组为有序 length-prefixed sequence，对象按 key 的 UTF-8 bytes 排序，所有数字统一
编码为 big-endian IEEE-754 binary64，且拒绝非有限值与超出 JavaScript safe-integer 范围的整数。
字符串与对象 key 必须是有效 Unicode scalar sequence；孤立 UTF-16 surrogate 必须 fail closed。
`context_pack_id / receipt_id` 必须等于对应前缀加该 digest；consumer 必须重算 digest，不能只比较 id 后缀。

### 3.1 Context Request

必填：

- `request_id / company_id / cycle_id / workstream_id / decision_id`；`workstream_id` 可为空但不可省略；
- `decision_point`：`cycle_planning / work_prioritization / capacity_rebalance /
  dependency_resolution / incident_recovery / operating_review`；
- 非空且唯一的 `division_ids` 与 `action_catalog_ids`；
- 至少一个 `confirmed_fact`、一个 `constraint` 和一个 `evidence_ref`；fact 与 uncertainty 的 evidence
  id 必须存在；fact/constraint 的非空 division scope 必须在 request allowlist；
- `operating_window`：`currency / maximum_external_cost_minor / maximum_human_minutes /
  starts_at_ms / ends_at_ms / maximum_work_orders`；
- 显式 `uncertainties` 概率范围。

Request 没有自由文本 action。Advice 只能引用 allowlist 中的 division 与 catalog action。

### 3.2 Evidence lane 与 outcome matrix

| Evidence class | 合法 `outcome_kind` | PE eligible |
|---|---|---|
| `simulation` | `simulation_result` | 否 |
| `internal_review` | `internal_review_result` | 否 |
| `machine_check` | `machine_check_result` | 否 |
| `field` | `work_order_progress / objective_progress / cost_recorded / incident_recorded / human_load_recorded` | 否；只进 memory/execution result |
| `field` | `field_operation_result` | 是；必须是 AutoCompany 已完成资格判断的多目标聚合结果 |

Evidence role 是 closed enum：`operating_signal / constraint / decision_record / work_order /
internal_review / machine_audit / field_observation / objective_progress / cost / incident / human_load`。
internal review 与 machine check 必须使用各自专用 role；simulation 不能冒充现场、review 或 audit；field
不能冒充内部 lane。每个 field report 必须携带 `work_order` role，且该 evidence 的 `locator` 必须精确等于
report 的 `work_order_ref`。出现 metric、objective、正成本、事故或正 human minutes 时，还必须分别携带
对应的直接 evidence role。

### 3.3 Advice 的有界性

每个 `OperationsAdviceCandidate` 必须包含：

- closed `kind`：`prioritize_work / sequence_dependency / rebalance_capacity /
  recover_incident / pause_work / request_human`；
- request allowlist 中的 `target_division_id / action_catalog_id`；
- `maximum_cost_minor / maximum_human_minutes`，不得超过 operating window；候选数量不得超过
  `maximum_work_orders`；
- `risk_level / reversibility / requires_human_approval`；high、critical 或 irreversible 必须显式要求人审；
- 只引用当前 request 的 prerequisite fact/evidence，或当前 pack 实际召回的 memory entry；
- 至少一个预测区间和证伪条件；预测 horizon 必须位于 operating window 内，risk 不得为 unassessed。

Controller 会重验全部 lineage 和边界。Provider 不获得 mutation 或 actuator handle。Advice 永久
`WiringLevel.SHADOW`、`applied=false`，候选文本不进入 ACTIVE `rendered_context`。

### 3.4 多目标运营结果

`OperationsExecutionOutcome` 保留：

- `objective_result`：`not_observed / advanced / stalled / regressed / mixed`；
- 每项 metric 的 `metric_id / unit / baseline_value / observed_value / evidence_ref_ids`；
- 七类 realized cost：`model / data / human / infrastructure / vendor / incident_response / other`；
- `elapsed_ms / blocker_duration_ms / rework_count / incident_count / human_minutes`；
- `risk_level / reversibility`。

simulation/internal-review/machine-check lane 的这些现场维度必须保持未观察/零值。`verdict` 是
AutoCompany 对已核验 field aggregate 的 typed environment observation，不是 LLM judge、evaluation 分数或
Advice adoption。`field_operation_result` 不能只携带 elapsed time；必须有 objective、metric、cost、blocked
time、rework、incident、human load 或 assessed risk 中至少一项。summary/detail 中的形容词不会被解析。

## 4. Owner、状态与学习路径

### 4.1 Appendable 与 Readable

Controller 先用结构化 request 触发普通 `USER_INPUT` turn，只读该拍公开 `prediction_error` snapshot，
再经 `LifeformSession.retrieve_memory()` 读取 `WORLD` 的 `EPISODIC + DURABLE`。它只解析精确带
`operations-brain` 与当前 company facet 的 canonical `operations-experience-record.v1`；payload company
与 tag 不一致或记录损坏时 fail loudly，不能因同 identity scope 串读另一公司。Context Pack
发布实际使用的 entry/evidence ids、当前不确定性、settled outcome/evidence ids 和 PE readout。

Memory entry、排序、scope、promotion/decay、checkpoint 与恢复仍由 `vz-memory` 唯一拥有。Operations
controller 不读 `runner.memory_store` 或其他 owner 内部对象。

### 4.2 Outcome 与 Learnable lane

所有合法 report 都经 facade 追加 identity-scoped episodic memory，并发布 task event。task id 使用
`work_order_ref`；这里的 `completed` 仅表示 report 已处理，不表示工作成功。只有
`field + field_operation_result` 额外提交 `EnvironmentOutcome`：

```text
favorable -> +1
unfavorable -> -1
mixed / inconclusive -> 0
unit = autocompany_multiobjective_operation_verdict.v1
```

该标量是合格 field aggregate 的 PE observation，不是毛吞吐、成本、judge 或 evaluation reward。完整的
metric/cost/time/incident/human-load/risk 数据保留在 report/memory。下一次**新** Context Pack turn 才由
PE owner 结算；同 session 同时最多一个待结算 field aggregate，且它必须引用最新 Context Pack。

### 4.3 有界产品状态

Controller 只拥有进程内 live-session 幂等/lineage ledger：默认每 session 512 个 request/outcome，最多
1024 个 session；关闭 session 即释放。该 ledger 不是认知记忆或 AutoCompany 工作单 SSOT，不跨进程恢复。

## 5. Session-local HTTP API

创建 `vertical="operations"` session 后调用本地 service 路由：

```text
POST /v1/sessions/{session_id}/operations/context-packs
POST /v1/sessions/{session_id}/operations/outcomes
```

首次 publication 返回 `201`，同 payload replay 返回 `200`。错误：

| HTTP | error | 条件 |
|---|---|---|
| 400 | `invalid_json_body` | body 为空、不是 JSON object 或 JSON 语法损坏 |
| 400 | `invalid_operations_context_request` | schema/enum/ref/scope/window 不合法 |
| 400 | `invalid_operations_outcome` | lane、work-order/metric evidence 或 outcome payload 不合法 |
| 404 | `session_not_found` | session 不存在 |
| 409 | `operations_vertical_required` | session 不是 operations vertical |
| 409 | `historical_session_readonly` | historical session 写请求 |
| 409 | `operations_idempotency_conflict` | 同 id 使用不同 immutable payload |
| 409 | `operations_context_lineage_error` | unknown/cross-session/stale pack 或 decision/currency 不匹配 |
| 409 | `operations_settlement_pending` | 前一 field aggregate 尚未在下一 Context Pack 结算 |

DLaaS 以 `(ai_id, session_id)` 路由并提供显式 session create：

```text
POST /dlaas/v1/instances/{ai_id}/sessions
POST /dlaas/v1/instances/{ai_id}/sessions/{session_id}/operations/context-packs
POST /dlaas/v1/instances/{ai_id}/sessions/{session_id}/operations/outcomes
```

Operations 请求不自动创建缺失 session。Context Pack 与 Receipt 都发布同一个 opaque
`session_lineage_id`；即使两个 `ai_id` 复用同名 `session_id`，它们的 Context Pack identity 也不同，
cross-instance outcome 会 fail closed。multi-pod 模式把 session/context/outcome 原样转发到 `ai_id` 的
sticky owning pod；parent 不创建 kernel session 或 domain controller 副本。

## 6. AutoCompany adapter 接线

1. AutoCompany 从自己的 OKR、治理、预算、division registry 和 action catalog 生成 request；不得发送凭据、
   私有 ledger 或要求 Brain 猜 evidence class/action；
2. adapter 持久化 pack/advice/source lineage。只有 `rendered_context` 可作为 ACTIVE COO/规划上下文；Advice
   仅存储、展示或离线比较；
3. AutoCompany 自己作决定，审批后创建 durable typed work order，再经已有 division intake 派发；
   adapter 必须在该 work order/outcome 关闭前保持其 pack 为 latest，除非新 pack 专用于结算已持久化的
   PE environment outcome；pack allocation 与 work-order dispatch 必须共享 per-session coordination boundary；
4. division 返回 typed outcome；AutoCompany 核验并构造 report，`work_order` evidence locator 必须等于
   durable `work_order_ref`；
5. progress/cost/incident/human-load 可单独提交用于 recall；只有完整、已资格化的
   `field_operation_result` 进入 PE；
6. adapter 持久化 receipt。PE eligible 时先请求下一个新 Context Pack 完成 settlement，再提交下一个
   aggregate；
7. service/controller restart 后旧 Memory 可按 identity 恢复，但旧 pack id 不恢复；adapter 必须请求新 pack。

AutoCompany 适配必须以显式 feature flag / wiring level 上线。关闭 Context Pack 消费即可回滚；SHADOW
Advice 无需执行回滚。任何 Advice ACTIVE 或直接 dispatch 都必须新版本合同、ModificationGate、独立
evidence 与单字段可回滚晋升，不能静默改变 v1。

## 7. 四能力轴审计

| 轴 | v1 成立范围 | 诚实边界 |
|---|---|---|
| Appendable | typed work-order report 追加 scoped episodic memory；持久 backend 下跨 session 恢复 | 不替代 AutoCompany 工作单/治理 ledger |
| Readable | pack 只读 immutable Memory entries 与同拍 PE snapshot，发布完整 source/settlement lineage | 无召回不重建 producer 隐状态；损坏 canonical record fail loudly |
| Learnable | 仅 qualified field aggregate 走 EnvironmentOutcome→PE；其他 lane 只保留 memory/execution | judge/evaluation/advice adoption/build/deploy/完成数不回灌 |
| Steerable | ACTIVE pack 改变下一周期可见经验；Advice 仅 SHADOW 且 catalog/cost/authority 有界 | 不证明 production ACTIVE policy 或直接 division control |

## 8. 回滚、限制与验收

- 回滚：AutoCompany 停止消费 pack；service 可移除 operations routes/vertical；不影响普通 turn 或其他 brain；
- crash：v1 无持久幂等 ledger。外部未收到 receipt 时，AutoCompany 必须凭自己的 adapter ledger 对账并
  请求新 pack，不得复用旧 pack 猜测结果；
- 验收覆盖 strict version/enum/role/pair、work-order locator、metric lineage、成本 shape、Advice 边界、
  request/outcome idempotency、same-session/latest-pack、跨 session recall、下一拍 PE lineage、SHADOW
  isolation、service status/vertical guard、import boundary 和相关 Ruff/pytest；
- production promotion 还要求 AutoCompany durable adapter、typed division intake/outcome、部署接线和
  SHADOW 运行证据；本 wheel 单独通过测试不代表这些外部条件已成立。
