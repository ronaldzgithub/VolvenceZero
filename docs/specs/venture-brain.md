# Venture Brain v1 产品契约 Spec

> Status: implemented v1
> Last updated: 2026-08-30
> Owner: `lifeform-domain-venture`
> Service projection: `lifeform-service`
> Upstream authority: Foundry

## 1. 产品定义与信任边界

Venture Brain 是与 Coding Brain 平级的**有状态商业认知侧车**。Foundry 可以在机会脑暴、
语义比较、实验规划、组合判断辅助、监控归因和跨周期回顾前请求 Context Pack，并在事后把
已经由 Foundry 分类、核验和对账的 typed outcome 提交回来。Venture Brain 负责召回相关经历、
发布当前不确定性、生成可证伪的 SHADOW 候选，并让合格的延迟现场结局经既有 Prediction Error
链结算。

Venture Brain 不是另一个 Foundry，也不是商业 actuator。权威边界固定如下：

| Foundry 独占 | Venture Brain 可做 | Venture Brain 禁止 |
|---|---|---|
| 来源获取与原文核验、evidence class、确定性资格门、portfolio/budget、Accounting、ledger、审批、最终状态转换、全部外部动作 | 消费 Foundry 已确认的结构化事实；跨周期召回；发布不确定性；给出候选机会/比较/实验/停止建议；记录 typed outcome；辅助 PE/credit 学习 | 抓取来源、制造或部署产品、联系客户、花钱、读取或写入 Foundry 私有账本、升级 evidence class、批准实验、执行建议、决定 continue/pause/stop/scale/kill |

闭环是：

```text
Foundry 已确认事实 + evidence refs + 决策点/约束/预算/窗口
  -> VentureContextRequest
  -> 普通 Lifeform turn（结算上一批合格 field outcome）
  -> Memory owner 的 identity-scoped episodic/durable retrieval
  -> VentureContextPackSnapshot [ACTIVE]
       + VentureAdviceSnapshot [SHADOW, applied=false]
  -> Foundry 自己资格审查、审批、执行、Accounting 和状态转换
  -> VentureOutcomeReport（Foundry typed、可验证引用、多目标商业结果）
  -> memory + execution_result
       + 仅 field_experiment_result 进入 EnvironmentOutcome
  -> 下一 Context Pack turn 由 PE owner 结算并发布 lineage
```

## 2. v1 最小可交付范围

v1 包含：

- 一个独立 `lifeform-domain-venture` wheel，复用 `lifeform-core`、Memory、PE、semantic task
  event、content address、identity scope 和 `WiringLevel`；
- 严格、版本化、frozen 的 request/pack/advice/report/receipt 合同；所有 HTTP 输入及嵌套对象
  拒绝未知字段、未知 enum 和非法 evidence-class/outcome-kind pair；
- Context Pack 可 `ACTIVE`，但只由 Foundry typed request 与 owner-published memory/PE readout
  构成；Advice 在 v1 永久 `SHADOW` 且 `applied=false`；
- request/outcome 幂等、同 key 不同 immutable payload 冲突、同 live session lineage；
- 同 identity 跨 session 召回与下一 Context Pack 的 outcome/PE settlement lineage；
- `lifeform-service` 两条薄 HTTP 投影及 venture/historical session guard。

v1 明确不包含：

- Source Adapter、crawler、Product Zero/Foundry builder、deploy、market action 或 payment rail；
- Foundry ledger、Accounting、portfolio allocator、budget gate、approval 或状态机副本；
- 从 summary/detail 等自由文本推断 evidence class、route、verdict、decision 或商业状态；
- evaluation、LLM judge、模型分数、建议被采纳、构建成功、本机部署健康或短期毛收入作为
  赚钱 reward；
- 新 kernel slot、新 owner store、token-space RL、在线 base-model 更新或 production ACTIVE
  steering。

## 3. 冻结公共合同

实现位于 `lifeform_domain_venture.venture_brain_contracts`。金额统一使用显式 ISO 4217
三字母币种和整数 minor units；禁止隐式汇率换算。字符串字段仅是 Foundry 已确认的内容或说明，
不承担路由语义。

| 对象 | schema | 核心语义 |
|---|---|---|
| `VentureContextRequest` | `venture-context-request.v1` | portfolio/cycle/venture/decision identity、closed decision point、confirmed facts、constraints、resource window、uncertainties、typed evidence refs |
| `VentureContextPackSnapshot` | `venture-context-pack.v1` | content-addressed ACTIVE pack、实际渲染的跨周期经历、source entry/evidence ids、当前不确定性、PE settlement lineage、独立 SHADOW advice |
| `VentureAdviceSnapshot` | `venture-advice.v1` | opportunity/comparison/experiment/stop 候选、预测区间、证伪条件、evidence/memory lineage；固定 SHADOW、未应用 |
| `VentureOutcomeReport` | `venture-outcome-report.v1` | Foundry 事后提交的 typed decision、evidence class/kind、verdict、外部引用、多目标结果 |
| `VentureOutcomeReceipt` | `venture-outcome-receipt.v1` | content-addressed 写入回执、memory/event/environment lineage、学习 route、下一拍结算状态、`source_advice_applied=false` |

### 3.1 `VentureContextRequest`

必填字段：

- `request_id / portfolio_id / cycle_id / venture_id / decision_id`；`venture_id` 允许空字符串，
  用于 portfolio-level 决策，但字段本身不能省略；
- `decision_point`：`opportunity_brainstorm / candidate_comparison /
  experiment_planning / portfolio_review / monitor_attribution / stop_review`；
- 至少一个 `confirmed_fact` 和一个 `constraint`；fact 必须引用 request 中存在的 evidence ref；
- `resource_window`：currency、maximum total cost、start/end 和 maximum experiments；
- 当前 `uncertainties`：显式概率上下界及可选 evidence lineage；
- 至少一个 `evidence_ref`：`ref_id / evidence_class / role / locator /
  content_sha256 / observed_at_ms`。

Request 不提供 `route` 或所谓 inferred evidence class。Foundry 必须在 evidence ref 上逐条声明
class/role；Venture Brain 不扫描 fact statement 来改变声明。

### 3.2 Evidence class、role 与 outcome pair

四个 evidence class 与 Foundry 的 evidence model 一一对应：

- `simulation`：fixture、replay、dry run、synthetic revenue；
- `internal_review`：具名人审的内部结果；
- `machine_check`：机器可裁决的 schema/matcher/build/release/runtime 健康事实；
- `field`：带可验证外部引用的客户、交付、付款、成本、退款或现场实验结果。

Closed outcome matrix：

| Evidence class | 合法 `outcome_kind` | PE eligible |
|---|---|---|
| `simulation` | `simulation_result` | 否 |
| `internal_review` | `internal_review_result` | 否 |
| `machine_check` | `machine_check_result` | 否 |
| `field` | `customer_outcome / payment_received / cost_recorded / refund_recorded` | 否；只记忆与 execution result |
| `field` | `field_experiment_result` | 是；必须是 Foundry 已完成聚合和资格判断的多目标实验结果 |

Evidence role 包括 `demand_signal / constraint / decision_record / experiment /
internal_review / machine_audit / field_observation / customer_outcome / payment / cost /
refund`。`internal_review` 只能配 `internal_review` role；`machine_check` 只能配
`machine_audit`；simulation 不能使用 review/audit 或任何 field-only role；field 不能冒充两个内部
lane。所有 outcome evidence refs 的 class 必须与 report class 相同。field report 每个实际出现的
客户结果、正收入、正成本和正退款维度，还必须分别包含 `customer_outcome / payment / cost / refund`
role 的引用；一个笼统 `field_observation` 不能替代这些维度的直接 lineage。

特别地：公开帖子上的 demand/payment 表述仍是 `demand_signal`，不是客户结果或付款；构建成功、
Release Gate 通过和本机部署健康只能落 `machine_check`，不能落 field 或收入。

### 3.3 Typed decision 与 verdict

- `decision` 是 Foundry 已做出的事后记录：`run_experiment / continue / pause / stop /
  scale / kill / no_state_change`。Receipt 记录它，但 Venture Brain 不执行状态转换。
- `verdict` 是 Foundry 对多目标结果的显式判词：`favorable / unfavorable / mixed /
  inconclusive`。只有合格 `field_experiment_result` 的 verdict 被映射为 PE measurement；
  summary/detail 中的形容词不会被解析。

### 3.4 多目标商业结果

每个 report 都必须携带 `VentureCommercialOutcome`，即使所在 lane 不具备真实商业观察；不可观察
维度显式为零或 `not_observed / unassessed`：

- `customer_result`：`not_observed / positive / negative / mixed`；
- `realized_revenue_minor`；
- 七项 realized costs：`acquisition_minor / model_minor / data_minor /
  human_review_minor / delivery_minor / support_minor / risk_reserve_minor`；
- `refund_minor` 与 `realized_net_value_minor`；后者必须严格等于收入减七项成本再减退款；
- `elapsed_ms / risk_level / reversibility`。

`simulation / internal_review / machine_check` 禁止携带客户、财务、elapsed 或 risk 观察；模拟预测应放
Advice 的 estimate range 或模拟器自己的证据 artifact，绝不能填入任何 `realized_*` 字段。Foundry 当前
六项 realized cost 可直接映射同名六项，并在尚未独立核算 delivery 时显式传 `delivery_minor=0`；不得把
估算值填进 realized 字段。

## 4. Owner、状态与学习路径

### 4.1 Context Pack：Appendable + Readable

1. `VentureBrainController` 用结构化 request 触发普通 `USER_INPUT` turn；该 turn 是上一批
   PE-eligible outcome 的正式结算点；
2. 只读该 turn 发布的 `prediction_error` public snapshot；
3. 只经 `LifeformSession.retrieve_memory()` 查询 `Track.WORLD` 的 `EPISODIC + DURABLE`；
4. facets 仅由 portfolio/venture identity、closed decision point 和 fact kind 确定性构造；
   query text 只用于 owner retrieval，不做 evidence/route 分类；
5. 只解析带精确 `venture-brain` tag 的 canonical
   `venture-experience-record.v1`；venture-tagged 记录损坏时 fail loudly；
6. 在 `max_context_chars` 边界内按 Memory owner 顺序渲染，并发布实际使用的
   `source_entry_ids / source_evidence_ref_ids`；无召回时发布明确空状态；
7. snapshot 使用 `WiringLevel.ACTIVE`。它可以由 Foundry adapter 注入其规划上下文，但不包含
   SHADOW Advice 的任何候选文本。

Memory entry、检索排序、scope、promotion/decay、checkpoint 与恢复仍唯一属于 `vz-memory`。
Venture controller 只使用 Lifeform facade，不读取 `runner.memory_store` 或 semantic owner 内部结构。

### 4.2 Outcome：execution、memory 与 PE lane

所有合法 report：

- 经 `LifeformSession.write_memory()` 追加 identity-scoped `WORLD/EPISODIC` canonical JSON，
  并调用 Memory owner 的 persist facade；
- 经 `submit_task_event()` 发布 typed execution result。这里的 `completed` 只表示“Foundry report
  已处理”，不表示产品成功、盈利或建议正确；
- 返回 content-addressed Receipt；同 `outcome_id + payload` 重放同一 Receipt，同 id 不同 payload
  立即冲突。

只有 `field + field_experiment_result` 还会提交一个 `EnvironmentOutcome`。它不把收入作为 reward：

```text
Foundry verdict favorable   -> normalized multiobjective measurement +1
Foundry verdict unfavorable -> normalized multiobjective measurement -1
Foundry verdict mixed       -> 0
Foundry verdict inconclusive-> 0
```

measurement unit 固定为 `foundry_multiobjective_verdict.v1`，并携带 Foundry evidence ref lineage。
实际收入、七项成本、退款、客户结果、时间、风险和可逆性完整保留在 report/memory，不被压成毛收入
标量。PE owner 在**下一次新 Context Pack turn** 结算，pack 发布
`settled_outcome_ids / settled_evidence_ref_ids / pe_magnitude / pe_bootstrap`。

### 4.3 Advice：Steerable 的诚实边界

注入的 `VentureAdviceProvider` 只能返回 frozen `VentureAdviceCandidate`：

- kind 必须是 opportunity/comparison/experiment/stop；
- 至少一个预测区间与至少一个证伪条件；
- evidence ids 必须来自当前 request 或当前 pack 已召回经历；memory entry ids 必须来自当前 pack；
- controller 会再次校验所有 lineage、候选数量和唯一 id；provider 不获得 mutation/actuator handle。

`VentureAdviceSnapshot` 永久 `WiringLevel.SHADOW`、`applied=false`，并与 ACTIVE
`rendered_context` 物理分字段。Receipt 也固定记录 `source_advice_applied=false`。建议被 Foundry
采纳既不是 field evidence，也不是 reward。

### 4.4 有界产品状态

controller 只拥有 live-session 产品幂等/lineage ledger：每 session 默认最多 512 个 request/outcome
key、全进程最多 1024 个 session ledger；关闭 session 即释放。它不是认知记忆，不跨进程恢复，不是
Foundry ledger。长期经验只在 Memory owner 中。

## 5. HTTP API

先创建 `vertical="venture"` session，再调用：

```text
POST /v1/sessions/{session_id}/venture/context-packs
POST /v1/sessions/{session_id}/venture/outcomes
```

成功状态：首次 publication 为 `201`，完全相同的 idempotent replay 为 `200`。失败状态：

| HTTP | error | 条件 |
|---|---|---|
| 400 | `invalid_venture_context_request` | schema version、unknown field、enum、引用或数值不合法 |
| 400 | `invalid_venture_outcome` | schema、class/kind/role、商业算术或 lane payload 不合法 |
| 404 | `session_not_found` | session 不存在 |
| 409 | `venture_vertical_required` | session 不是 venture vertical |
| 409 | `historical_session_readonly` | historical session 写请求 |
| 409 | `venture_idempotency_conflict` | 同 request/outcome id 使用不同 payload |
| 409 | `venture_context_lineage_error` | unknown/cross-session/stale Context Pack、decision/currency 不匹配 |
| 409 | `venture_settlement_pending` | 前一个 field aggregate 尚未由下一 Context Pack turn 结算 |

Service 只解析 JSON、执行 session guard 并投影 controller；不保存业务状态，也不访问 Foundry。

## 6. Foundry adapter 精确接线

Foundry 后续 adapter 应遵循以下顺序：

1. Foundry 自己完成 source/evidence 校验、资格门与授权，再构造完整
   `venture-context-request.v1`。不要发送私有 ledger、凭据、原文秘密或让 Venture Brain猜 class；
2. 保存响应的 `context_pack_id / content_sha256 / source_entry_ids /
   source_evidence_ref_ids / advice.advice_id` 到 Foundry 自己的 adapter lineage；只有
   `rendered_context` 可作为 ACTIVE planning context，`advice` 只展示或离线比较；
3. Foundry 做出决定并经原有审批/预算/外部动作链执行；Venture Brain 不在该路径上；
4. 事后由 Foundry 构造 `venture-outcome-report.v1`：每个 evidence ref 提供 Foundry opaque
   locator 与所引用内容的 lowercase SHA-256；金额来自 Accounting 已核验的 realized 值；
5. 单项 customer/payment/cost/refund 可及时提交用于 recall，但不产生 PE；完成 Foundry
   `business_loop` 聚合与资格判断后，才提交一个 `field_experiment_result` 及 explicit verdict；
6. 保存 Receipt。若它是 PE eligible，必须先请求下一个**新** Context Pack 完成 settlement，再提交
   下一个 field aggregate；核对新 pack 的 settled ids/refs；
7. service 或 controller restart 后，旧 Memory 可按 identity 恢复，但 in-process Context Pack
   lineage 不恢复。Foundry 必须请求新 pack，不能把旧 pack id 用于新 report。

v1 的 deliberate restriction：PE-eligible 延迟结果必须引用该 live session 最新 Context Pack，且同一
session 同时只能有一个待结算 aggregate。这避免 delayed commercial credit 错配；未来若要并行实验，
应新增 versioned multi-pending contract，而不是放宽 v1。

## 7. 四能力轴审计

| 轴 | v1 成立范围 | 诚实边界/回滚 |
|---|---|---|
| Appendable | typed report 追加 scoped episodic memory；配置持久 backend 后跨 session 恢复 | 无 backend 时 Receipt 明示 `memory_persisted=false`；不等于 Foundry ledger |
| Readable | Context Pack 只读 immutable Memory entries 与同拍 PE snapshot，并发布 source/settlement lineage | 无召回不重建隐藏状态；损坏 canonical record fail loudly |
| Learnable | 仅 Foundry-qualified field aggregate 走 EnvironmentOutcome→PE；其他 lane 保留 memory/execution | simulation/review/machine/单项 field/LLM judge/采纳/build/deploy/gross revenue 不作 reward |
| Steerable | Context Pack 可 ACTIVE 地改变下一周期可见经验；Advice 仅 SHADOW | 不证明 production ACTIVE advice；Foundry 可停止消费 pack 立即回滚 |

ACTIVE Context Pack 可能改变 Foundry 后续规划信息，后续真实 typed outcome 再使 PE 可结算，因而闭环
结构成立；商业 uplift、market validation 或“会赚钱”仍需 Foundry 自己的 loop-external evidence，不能
由接线或测试通过推出。

## 8. 退出、回滚与残余风险

- 最小回滚：Foundry adapter 停止消费 Context Pack；Advice 本来就是 SHADOW；
- service 回滚：取消 venture routes 与 vertical discovery，不影响普通 turns、Coding Brain 或其他
  vertical；
- learning 回滚：不提交 typed report 即不产生新 memory/event/outcome；已提交 experience append-only，
  删除必须走 Memory owner 的显式用户数据操作；
- v1 没有持久幂等 ledger；进程 crash 发生在外部收到 Receipt 前时，Foundry 必须以自身 adapter
  ledger 核对后请求新 pack，不得猜测写入是否发生；
- Foundry 当前 realized-cost schema 与本合同新增的 `delivery_minor` 需在 adapter 显式映射；未核算时
  只能传零，不能用 estimated delivery cost；
- 任何 Advice ACTIVE、并行 delayed outcome、多币种聚合或商业 actuator 都必须新 schema、独立 evidence、
  `ModificationGate` 与单字段 `WiringLevel` 晋升，不得静默改变 v1。

## 9. 验收

- contract：version、unknown field、closed enum/pair/role、金额算术、ACTIVE/SHADOW guard；
- controller：request/outcome 幂等与冲突、same-session/latest-pack lineage、bounded ledger；
- recall：同 identity 新 session 召回 canonical outcome；
- learning：下一 pack 发布 field aggregate 的 environment/evidence/PE lineage，其他 lane 无 PE；
- isolation：Advice marker 永不进入 ACTIVE context，domain code 不访问 owner store；
- service：201/200、400/404/409、非 venture 与 historical session boundary；
- regression：相关 contracts、service、wheel boundary 与 repo static checks。
