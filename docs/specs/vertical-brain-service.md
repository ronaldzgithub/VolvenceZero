# Vertical Brain 统一服务适配契约

> Status: implemented
> Last updated: 2026-08-30
> Transport owner: `lifeform-service`
> Domain owners: `lifeform-domain-coding` / `lifeform-domain-venture` / `lifeform-domain-operations`

## 1. 目的与边界

Coding、Venture 与 Operations 都提供同一种产品闭环：外部 owner 发布 typed Context Request，Brain
返回 immutable Context Pack；外部 owner 完成自己的行动与资格判断后，再提交 typed Outcome Report。
过去三个项目使用不同 URL，DLaaS 还只认识 Operations，导致 transport、错误投影和多 pod 转发重复。

`VerticalBrainAdapter` 只统一两项操作、session guard、发现与错误 shape。它不统一或重解释 domain
payload，不读取自然语言，不持有 Memory、PE、credit、policy、evidence 或 authority。每个 adapter 仍把
payload 交给对应 domain contract 的 `from_json()` 与 controller；因此 owner、严格 schema、lineage、
WiringLevel 和 ModificationGate 均保持原位。

## 2. 公共 HTTP 面

调用方先用现有 `/v1/sessions` 创建并绑定一个 vertical session，之后只需使用：

```text
GET  /v1/brains
POST /v1/sessions/{session_id}/brain/context-packs
POST /v1/sessions/{session_id}/brain/outcomes
```

service 根据 session 冻结的 `vertical_name` 选择 adapter，不接受客户端在每次请求中覆盖 Brain 名称。
成功响应保持 domain owner 的原生 JSON schema，并增加 `X-Volvence-Brain` 响应头。首次写入返回 `201`，
完全相同的幂等 replay 返回 `200`。

`GET /v1/brains` 返回 `vertical-brain-registry.v1`，列出已安装 adapter、request/report schema version 与
公共路径。它是能力发现，不是 ACTIVE 授权；具体 advice 的 `WiringLevel` 仍以 domain snapshot 为准。

公共错误：

| HTTP | error | 条件 |
|---|---|---|
| 400 | `invalid_json_body` | body 缺失、不是 JSON object 或 JSON 损坏 |
| 400 | `invalid_brain_context_request` | 当前 vertical 的 context contract 拒绝 payload |
| 400 | `invalid_brain_outcome` | 当前 vertical 的 outcome contract 拒绝 payload |
| 404 | `session_not_found` | session 不存在 |
| 409 | `vertical_brain_unavailable` | session vertical 没有安装 adapter |
| 409 | `historical_session_readonly` | historical session 写请求 |
| 409 | `brain_idempotency_conflict` | 同一 domain id 重用不同 immutable payload |
| 409 | `brain_context_lineage_error` | outcome 的 Context Pack 或业务 lineage 不匹配 |
| 409 | `brain_settlement_pending` | domain 要求上一结果先由下一 Context turn 结算 |

## 3. 兼容与项目接线

DLaaS 项目使用同一 operation shape，并以 `(ai_id, session_id)` 保持实例隔离：

```text
POST /dlaas/v1/instances/{ai_id}/sessions
POST /dlaas/v1/instances/{ai_id}/sessions/{session_id}/brain/context-packs
POST /dlaas/v1/instances/{ai_id}/sessions/{session_id}/brain/outcomes
```

multi-pod launcher 只暴露 `forward_brain_request(ai_id, session_id, operation, payload)`，payload 对 parent
保持 opaque，并转发到 `ai_id` 的 sticky owning pod。parent 不创建 session、controller 或 product-lineage
副本；缺少该 capability 时返回明确的 `pod_brain_forwarding_unavailable`，不得回落到 parent-local 执行。

原有路径继续注册为兼容别名，错误码也保持 domain-specific，不要求已有调用方一次迁移：

```text
/v1/sessions/{session_id}/coding/{context-packs|outcomes}
/v1/sessions/{session_id}/venture/{context-packs|outcomes}
/v1/sessions/{session_id}/operations/{context-packs|outcomes}
/dlaas/v1/instances/{ai_id}/sessions/{session_id}/operations/{context-packs|outcomes}
```

新项目统一采用以下步骤：

1. 在创建 session 时一次性选择 `coding`、`venture` 或 `operations`；
2. 向公共 `brain/context-packs` 提交该 vertical 的版本化 typed request；
3. 持久化原生 Context Pack、content id 与 domain lineage；
4. 外部项目自行决策、审批和执行，Brain 不获得 actuator；
5. 向公共 `brain/outcomes` 提交该 vertical 的版本化 typed report；
6. 按原生 receipt/settlement contract 请求下一 Context Pack。

统一 URL 不代表统一业务字段。Coding host、Foundry 与 AutoCompany 仍分别拥有 task/review、商业证据与
Accounting、运营状态与 work-order ledger；adapter 禁止跨域映射这些字段。

## 4. 退出与验证

共享路由是 additive migration。回滚时调用方切回原 domain alias，或 service 取消
`register_vertical_brain_routes()`；三个 controller、Memory、PE/credit 和已有数据不变。完成迁移的退出条件
是项目不再拼接 domain 名路径，且统一路由的 201/200、vertical dispatch、historical/session guard 与公共
错误均被 consumer contract test 覆盖。

当前 smoke 使用三个真实 vertical 分别创建 session 并调用同一个 Context Pack URL，验证响应头和
`coding-context-pack.v1`、`venture-context-pack.v1`、`operations-context-pack.v1` 原生 schema；该验证只证明
transport 一致与 owner-preserving dispatch，不证明 Advice ACTIVE、产品 uplift 或四能力 thesis。
