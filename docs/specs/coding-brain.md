# Coding Brain 产品契约 Spec

> Status: implemented v1
> Last updated: 2026-08-29
> Owner: `lifeform-domain-coding`
> Service projection: `lifeform-service`
> Evidence basis: [`coding-lab.md`](./coding-lab.md) Packet 1 / Packet 2

## 1. 产品定义

Coding Brain 不是 IDE、代码执行沙箱或另一个 coding agent。它是给 coding agent host
调用的**有状态认知侧车**：在一次编码任务开始前返回跨任务可恢复的 Context Pack，在测试、
review 或 merge 产生可验证结果后接收 typed outcome，并让后续 turn 通过既有
Prediction Error → credit → 有界控制链学习。

首个产品闭环固定为：

```text
coding host 的结构化任务事实
  -> CodingContextRequest
  -> LifeformSession.run_turn（结算上一任务 evidence）
  -> Memory owner 检索 episodic / durable experience
  -> CodingContextPackSnapshot [ACTIVE]
  -> host 自己规划、编辑和执行
  -> CodingOutcomeReport（typed test / review / merge evidence）
  -> Memory owner append + execution_result；确定性 oracle 另进 dialogue_external_outcome
  -> 下一次 Context Pack turn 由 PE owner 结算
```

产品价值在于“任务之间积累并调用可验证经验”，不在于重复实现模型、terminal、git、review
或部署平台。Volvence Deploy 只负责身份、租户、持久卷、鉴权、限流和路由；本仓库拥有认知
契约、coding vertical adapter 与 service API。

## 2. v1 范围与非目标

v1 必须具备：

- memory-first Context Pack：只经 `LifeformSession.retrieve_memory()` 读取 Memory owner
  发布的 immutable entries，限定 `EPISODIC + DURABLE`；
- typed outcome：closed enum 覆盖 test/build/CI、code review 与 VCS merge/revert；
- 跨 session 恢复：closed-alpha identity scope 配置持久化后，同一 identity 的新 session
  可召回旧 coding outcome；
- 明确接线：Context Pack `ACTIVE`；controller advice `SHADOW` 且 `applied=false`；
- content-addressed lineage、request/outcome 幂等与冲突 fail loudly；
- 每个确定性测试结局在下一 Context Pack turn 可从 PE owner 的
  `ActualOutcome.external_outcome_refs` 观察到结算证明。

v1 明确不做：

- 不执行 shell、修改文件、创建 PR 或 merge；这些仍属于 host 的 affordance/tool 边界；
- 不把 advisor 候选注入 `rendered_context`，不影响 host 的 ACTIVE 行为；
- 不从任务摘要、日志或 review 文本猜 outcome kind、source、route 或 task kind；
- 不使用 evaluation/judge/review 分数作为 reward；
- 不新增 kernel runtime slot，不让 service 或 coding vertical 直接访问 owner store；
- 不宣称 production ACTIVE steering 或 Coding Lab thesis 已通过。

## 3. 冻结公共对象

实现位于 `lifeform_domain_coding.coding_brain_contracts`，所有输出均为 frozen dataclass，
JSON 解析拒绝未知字段与未知 enum。

| 对象 | schema | 关键字段 | 语义 |
|---|---|---|---|
| `CodingContextRequest` | `coding-context-request.v1` | request/project/repository/task identity、`CodingTaskKind`、summary、revision、target paths、memory/context limit | host 已知的结构化任务事实；不做文本路由 |
| `CodingContextPackSnapshot` | `coding-context-pack.v1` | content-addressed id、ACTIVE wiring、rendered memory、source entry ids、facets、PE settlement refs、SHADOW advice | 当前任务可实际注入 coding agent 的唯一产品输出 |
| `CodingAdviceSnapshot` | `coding-advice.v1` | source turn、candidate regime/action、evidence entry ids、`SHADOW`、`applied=false` | 只读候选，用于比较和未来 promotion evidence |
| `CodingOutcomeReport` | `coding-outcome-report.v1` | outcome/context-pack lineage、closed kind/source、summary/detail、timestamp、evidence ref、changed paths | host 提交的 typed 环境事实 |
| `CodingOutcomeReceipt` | `coding-outcome-receipt.v1` | memory entry、semantic event ids、external evidence id、learning route、pending settlement state | 写入与下一拍结算的不可变回执 |

### 3.1 Closed enums

- `CodingTaskKind`: `feature / bugfix / refactor / review / test / docs / research /
  maintenance / other`。
- `CodingOutcomeKind`: `task_verified / task_regressed / review_approved /
  review_changes_requested / merged / reverted`。
- `CodingOutcomeSource`: `test_suite / build_gate / ci / code_review / vcs`。

合法 pair 是契约，不是建议：

- `task_verified / task_regressed` 只接受 `test_suite / build_gate / ci`；
- `review_approved / review_changes_requested` 只接受 `code_review`；
- `merged / reverted` 只接受 `vcs`。

任意其他组合 400 / `ValueError` fail closed。不得因 detail 中出现 “pass”“review” 等词而
改变 route。

## 4. Owner、写入和学习路径

### 4.1 Context Pack（Appendable + Readable）

1. controller 先用结构化 request 触发一个普通 `USER_INPUT` lifeform turn。该 turn 是上一批
   outcome 的正式结算点，并发布 immutable snapshots；
2. 只读 `prediction_error` 公共 snapshot，提取 magnitude、bootstrap 与
   `actual_outcome.external_outcome_refs`；
3. 经 `LifeformSession.retrieve_memory(RetrievalQuery(...))` 查询 `Track.WORLD`、
   `EPISODIC + DURABLE`。query text 来自 host 显式字段；facets 来自 enum/identity/target path
   的确定性结构映射，不从自然语言分类；
4. 按 Memory owner 返回顺序渲染并在 `max_context_chars` 边界截断；记录所有实际渲染的
   `source_entry_ids`；
5. Context Pack 标记 `WiringLevel.ACTIVE`。无召回时返回明确空状态，而不是 fallback 到历史
   chat 或 service 私有数据库。

Memory 的唯一 owner 仍是 `vz-memory`。coding controller 只提交
`MemoryWriteRequest` / `RetrievalQuery`；entry id、检索分数、scope suppression、touch 与
checkpoint 都由 Memory owner 决定。

### 4.2 Typed outcome（Learnable）

所有合法 outcome 都做两件事：

1. 经 `LifeformSession.write_memory()` 追加 `Track.WORLD / EPISODIC` experience；失败、
   changes-requested、revert 使用更高 strength，但不直接等于 reward；
2. 经 `submit_task_event()` 排入既有 semantic owner，下一 turn 发布 `execution_result`；
   approved/verified/merged 对应 `completed`，regressed/changes-requested/reverted 对应
   `failed`。

只有确定性测试类事实额外执行：

- `task_verified -> DialogueExternalOutcomeKind.TASK_VERIFIED`
- `task_regressed -> DialogueExternalOutcomeKind.TASK_REGRESSED`
- source 固定 `DialogueExternalOutcomeEvidenceSource.ENVIRONMENT`
- `confidence=1.0`，并绑定 Context Pack 的 source turn 与 host evidence ref。

它们在**下一** Context Pack turn 进入既有 `dialogue_external_outcome -> prediction_error ->
credit` 主链。code review 是人类判断，merge/revert 是 VCS 状态；二者不得冒充确定性 task
oracle，因此只走 `execution_result`。evaluation、review 分数与成功率均不写回学习源。

### 4.3 Advisor（Steerable 的诚实边界）

v1 advisor 只投影同一 source turn 已发布的 `active_regime` 与
`active_abstract_action`，携带召回 evidence ids。它固定为：

- `WiringLevel.SHADOW`
- `applied=false`
- 不进入 `rendered_context`
- 不调用 temporal advisory ingress，不改变 residual/control code

因此 v1 只证明 advisor readout 可生成、可对照，不证明产品 ACTIVE steering。未来晋升必须有
独立 evidence、ModificationGate、单字段 wiring 切换与 DISABLED 回滚，不能修改本 v1
回执来“追认”应用。

## 5. Service API

`lifeform-service` 只做 HTTP 投影和 session/vertical guard，不拥有业务状态：

```text
POST /v1/sessions/{session_id}/brain/context-packs
POST /v1/sessions/{session_id}/brain/outcomes

# 兼容别名
POST /v1/sessions/{session_id}/coding/context-packs
POST /v1/sessions/{session_id}/coding/outcomes
```

新调用方使用与 Venture/Operations 一致的 `/brain/*` 路径；service 由 session vertical 选择 Coding
adapter，响应 payload 仍是 Coding owner 的原生 schema。公共 transport 与错误契约见
[vertical-brain-service.md](./vertical-brain-service.md)。

- session 必须由 `vertical="coding"` 创建；其他 vertical 返回 conflict；
- historical read-only session 拒绝两条写路径；
- context request 相同 `request_id + payload` 返回同一 snapshot；同 id 不同 payload返回
  conflict；
- outcome 相同 `outcome_id + payload` 返回同一 receipt；同 id 不同 payload 返回 conflict；
- outcome 必须引用本 service 进程中同 session 已发布的 Context Pack；跨 session 或未知
  pack fail loudly；
- service restart 后认知 memory 由 identity-scoped Memory owner 恢复，但 v1 request ledger
  不恢复；host 必须重新请求 Context Pack 后再提交 outcome。
- controller ledger 默认每 session 最多 512 个 context/outcome key、最多 1024 个 live/dead
  session ledger；显式关闭 session 立即释放。该 ledger 只做产品幂等，不是认知记忆。

## 6. 四能力轴审计

| 轴 | v1 成立范围 | 回滚/边界 |
|---|---|---|
| Appendable | typed outcome 追加到 scoped episodic memory；配置 backend 时跨 session 恢复 | 无 backend 时 receipt 明示 `memory_persisted=false`，不宣称跨进程 |
| Readable | Context Pack 只读 immutable Memory entries 与 PE snapshot，发布 source ids / refs | 无召回返回空 pack，不重建 owner 内部状态 |
| Learnable | 确定性 outcome 经下一拍 PE；其他 typed outcome 经 execution_result，均保留 memory | review/merge 不进 TASK_*；evaluation 永不作 reward |
| Steerable | v1 仅有 SHADOW advisor readout | advisor 未应用；Context Pack ACTIVE 可单独停止消费 |

闭环中 ACTIVE Context Pack 会改变 coding host 下一任务可见信息，从而可能改变行动与后续环境
结局；PE 在后续 outcome 结算时度量预测差异。这里的因果产品增益仍需独立 A/B evidence，不能
由机制接线本身推出。

## 7. 退出与回滚

- 最小回滚：调用方停止消费 Context Pack；advisor 本来就是 SHADOW；
- API 回滚：取消注册两条 coding route，不影响普通 `/turns` 与其他 vertical；
- 数据回滚：v1 不自动删除已追加的用户 memory；删除必须走 Memory owner 的显式用户操作；
- 学习回滚：不提交 typed outcome 即不会建立新 external settlement；已提交 evidence append-only，
  不允许 service 覆盖或改写；
- promotion：任何 advisor ACTIVE 或新 actuator 必须另建版本、证据门与 `WiringLevel`，不得静默
  改变 `coding-advice.v1`。

## 8. 验收

- contract：frozen、strict JSON、closed enum/pair、ACTIVE/SHADOW guard；
- controller：memory-first recall、字符上限、content address、request/outcome 幂等与 conflict；
- learning：`TASK_REGRESSED`/`TASK_VERIFIED` 在下一 Context Pack 的 PE refs 可见；review/merge
  没有 external TASK_* evidence；
- persistence：同 identity 新 session 能召回旧 outcome；
- service：coding vertical guard、historical guard、201/200、400/404/409；
- boundary：service 与 domain 不访问 `runner.memory_store`，Coding Lab observer 迁移到公共 facade。
