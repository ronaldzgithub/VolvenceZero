# External Research Adapter：Foundry → Research Lab simulation seam

> Status: v1 implemented
> Last updated: 2026-08-30
> Owner: `volvence_forge.research_control`（exact lifecycle）与 `volvence_labs.portal`（local transport/view）
> Domain contract: `foundry-research-lab-intent.v1`

## 1. 决策摘要

Research Lab 允许兄弟仓库把领域研究意图接入现有 Praxist 控制面，但不要求外部领域伪装成
`forge-research-task.v1`，也不复制 Praxist lifecycle：

```text
Foundry ResearchLabIntent
  → Forge external descriptor
  → exact external ResearchRequest
  → A0 named-human review
  → existing doctor / resolve / start / targeted status reconcile
  → immutable simulation handoff
  → Foundry-owned validation / proposal / named-human apply
```

Foundry 继续拥有研究意图、预算、证据分类、结果采纳和人审应用。Research Lab/Forge 只拥有 Request、A0、
Praxist 调和及 handoff。外部结果固定为 `simulation + proposal_only`，永不进入 Volvence Candidate、
`ModificationGate`、`DISABLED/SHADOW/ACTIVE` 或 runtime wiring。

这是 offline development/control plane，不注册 `docs/DATA_CONTRACT.md` runtime slot，也不构成四能力轴效果证据。

## 2. 三层协议

### 2.1 Descriptor：`forge-external-research-descriptor.v1`

Schema：`forge/schemas/external_research.schema.json`。

Descriptor 是薄 transport envelope，只补充 Lab 执行所需字段：

- `adapter.adapter_id = foundry-research-lab-intent.v1`；
- `bindings.intent` 精确绑定 Foundry Intent；
- `bindings.budget` 允许且在 v1 **必须**与 Intent 为同一 locator/hash，`budget_source = intent:/launch`；
- task project root/id/task.yaml hash、launch profile、cohort/generations、证据分类与 adoption policy 全部从
  Intent 读取和复验，不要求 Foundry复制预算文件；
- `control.praxist_executable/run_dir/config_file` 由 Lab operator 冻结；
- `result_policy.result_locator` 是 run root 下的安全相对路径；
- authority 固定禁止外部动作、Foundry checkout/ledger write、direct apply、Product Zero start、Volvence
  promotion、ModificationGate 和 runtime wiring。

`descriptor_id` 是去掉 `descriptor_id/created_at` 后 canonical JSON 的 SHA-256。Intent 同时按 Foundry 的
`rli_<16hex>` identity 规则复验。相对 content locator 以 descriptor 所在目录为基准；Portal ingress 另要求
descriptor locator 位于已注册的 external domain root 下。

### 2.2 Request：`forge-external-research-request.v1`

Schema：`forge/schemas/research_control.schema.json`。

Forge 从 Intent 派生独立的 `external_<16hex>` control task id，并冻结：descriptor、Intent/budget、带原始
`public_source|simulation` 分类的 evidence、完整 task-project snapshot、Praxist executable/source tree、run
identity 和 Intent launch profile。Request authority 明确：A0 只可授权一次 Praxist research lifecycle；结果仍是
simulation，production promotion/ModificationGate/runtime wiring 均不适用。

Approval 与 Event 不另建外部分支：external Request 原样复用
`forge-research-approval.v1` 和 `forge-research-control-event.v1`，以及现有 capacity、doctor、resolve、
`START_INTENT` crash boundary、daemonized start 和 targeted status 实现。

### 2.3 Handoff：`forge-external-research-handoff.v1`

Handoff 只能在 exact A0 APPROVE 且 terminal Event 为 `RUN_COMPLETED` 后创建。它精确绑定 descriptor、Request
file SHA、Approval SHA、terminal Event SHA、run id/dir 和固定 result bytes，并固定：

- `evidence_class = simulation`；
- `adoption_mode = proposal_only`；
- `market_validation_claimed = false`；
- `adoption_status = pending_external_human_review`；
- Volvence promotion、ModificationGate、runtime wiring 全部 false。

同一 Request 只允许一份 immutable handoff；不同内容的第二份 handoff fail closed。Forge 不从该 handoff
创建 Candidate，Foundry 后续如何验证、形成 proposal 或人审 apply 不在本协议权限内。

## 3. 稳定 CLI

```bash
forge research-submit-external /absolute/path/to/descriptor.json \
  --requested-by "Named Operator" \
  --reason "Submit exact Foundry simulation intent" \
  --json

forge research-approve /absolute/path/to/request.json \
  --approved-by "Named Human" \
  --reason "Approve this exact Praxist lifecycle only"

forge research-reconcile --once \
  --request /absolute/path/to/request.json \
  --json

forge research-handoff-external /absolute/path/to/request.json \
  --recorded-by "Named Operator" \
  --reason "Return simulation evidence for Foundry review" \
  --json
```

两个 external 命令都输出稳定 JSON result，包含 artifact id/path/SHA 和 `simulation` authority readout。CLI 不接受
raw extra argv；Praxist 参数只能来自冻结的 Intent/descriptor。

## 4. Loopback API

mutation mode 通过以下入口开放同一 CLI seam：

```text
POST /api/v1/external/requests
POST /api/v1/a0/review
POST /api/v1/reconcile
POST /api/v1/external/handoff
```

External submit payload 精确字段：

```json
{
  "snapshot_revision": "<lab revision>",
  "domain_id": "foundry",
  "descriptor_locator": "artifacts/research_lab/<descriptor>.json",
  "descriptor_id": "external-research-descriptor:<sha256>",
  "descriptor_sha256": "<file sha256>",
  "actor": "Named Operator",
  "reason": "Why this exact request should enter A0"
}
```

Server 必须用 `--external-domain-root foundry=/absolute/path/to/foundry` 注册只读 ingress。Locator 只能是该 root
下的安全相对 regular file。所有 POST 继续要求 exact loopback Origin、CSRF、fresh snapshot revision、named actor、
exact id/hash；server 只构造固定 argv，owner 成功后重新 collect 并验证新 artifact。

`ResearchLabSnapshot.items[].research_mode` 区分 `volvence_promotion` 与 `external_simulation`。外部轨道在完成后
只开放 `record_external_handoff`，handoff sealed 后没有 A1/A2/SHADOW/ACTIVE action。

## 5. 兼容、迁移与回滚

- 现有 `forge-research-task.v1`、`research-submit`、A0、promotion pipeline 和 Portal 卡片保持兼容；
- external Request 只新增一个 request schema branch，共享 Approval/Event/调和实现；
- 根 launcher 检出 sibling Foundry schema 时只注册 read-only root，不扫描、审批或自动提交 Intent；
- Foundry thin client 只需生成 descriptor 并调用 CLI 或 loopback API，不导入 Volvence Python internals；
- 停用时移除 `--external-domain-root` 即关闭新 ingress；既有 Request/Event/handoff 保留审计史；
- 回滚本包不需要 wiring downgrade，因为外部轨道从未获得 Volvence runtime authority。

## 6. Fail-closed 条件

Intent/descriptor identity、task.yaml/full task snapshot、evidence、budget、executable/source、profile、run identity、
result path 或任何 hash 漂移都会拒绝审批、start 或 handoff。未批准、非 terminal、result 越界/symlink、多 handoff、
未注册 domain root、stale revision、错误 CSRF/Origin、匿名 actor、未知字段或 owner response shape 不符均 fail loudly。
