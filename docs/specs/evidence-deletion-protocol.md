# Evidence Deletion Protocol

> Status: scaffold v0.1 (SHADOW)
> Last updated: 2026-05-13
> Owner: cross-cutting-foundation-packet F-B (debt #49)

## 1. 范围

closed-alpha 已实现 `DELETE /v1/users/me/memory`（删 scoped memory），但 `evidence_root_dir/sessions/*.json` 仍是不可删的 audit 文件。本 spec 锁定双层 scope 下"按 end-user / 按 tenant 删除 + 留下不可删的删除证据"路径，对齐 PIPL / GDPR / 中国个人信息保护法的"被遗忘权"。

参考：[`commercialization-assessment.md`](../business/commercialization-assessment.md) §8.1.4 + cross-cutting-foundation-packet F-B 子任务 4。

## 2. 端点表

| 端点 | 触发方 | 删除范围 | 留下证据 |
|---|---|---|---|
| `DELETE /v1/users/me/memory` | end-user (closed-alpha 已实现) | scoped memory | `evidence_deletion_ledger.jsonl` |
| `DELETE /v1/users/me/evidence?since=&until=` | end-user (新) | session evidence files | 同上 |
| `DELETE /v1/tenants/{tid}/users/{uid}/memory` | tenant_admin (新) | end-user scoped memory | 同上 |
| `DELETE /v1/tenants/{tid}/users/{uid}/evidence` | tenant_admin (新) | end-user session evidence | 同上 |
| `DELETE /v1/tenants/{tid}/memory` | tenant_admin (新, 高危) | 全 tenant 的 memory | 同上 + tenant-level 双因素确认 |
| `DELETE /v1/tenants/{tid}/evidence` | tenant_admin (新, 高危) | 全 tenant 的 evidence | 同上 |

## 3. 删除证据 ledger schema

`evidence_deletion_ledger-YYYYMMDD.jsonl`（按日 rotate；append-only；永不删）每行：

```json
{
  "timestamp_iso": "2026-05-13T14:30:00+00:00",
  "scope_key": "brand_a:alice",
  "tenant_id": "brand_a",
  "end_user_id": "alice",
  "deleted_file_count": 12,
  "deleted_file_sha256_set": ["abc123...", "def456...", "..."],
  "actor": "end_user",
  "request_id": "req-2026051314300012345",
  "policy_version": "evidence-deletion-v0"
}
```

**约束**：
- ledger 条目本身**不**记录删除文件的内容（只记 sha256），让审计能追"何时删了多少"，但无法重建被删内容
- ledger 文件按日 rotate，归档保留期由 `EvidenceDeletionPolicy.retention_days`（默认 365 天）控制；归档之后 ledger 本身也可删，但每次归档要写 meta-ledger（"YYYYMMDD ledger 归档于 X，含 N 行"）

## 4. EvidenceDeletionPolicy 字段

| 字段 | 默认 | 含义 |
|---|---|---|
| `retention_days` | 365 | evidence 文件最大留存天数（独立 sweeper 保证；本模块不主动扫） |
| `delete_on_user_request` | True | 必须 True；PIPL / GDPR 不允许关 |
| `retain_deletion_proof` | True | 删除时写不可删 ledger entry |

## 5. 与双层 scope 联动（debt #46）

- end-user 自删：`scope_key = derive_scope_key(tenant, end_user)`，删 own scope 的 evidence
- tenant_admin 删 end-user：必须显式声明 `tenant_id` + `end_user_id`，不允许"误删整个 tenant"
- tenant_admin 删全 tenant：HTTP body 必须带 `confirm_tenant_purge: true` + tenant 二次签名（实施层细节，不在本 spec）

## 6. 不变量（contract test 守门）

| 不变量 | 守门方式 |
|---|---|
| ledger 是 append-only | `tests/contracts/test_evidence_deletion_proof_chain.py`（mock 文件系统验证 ledger 不被删） |
| 删除后 audit 可 enumerate 删除事件 | 同上：删除后 `read_ledger(date)` 必须含本次 record |
| 不能在 ledger 里留下被删内容 | 同上：record schema 不含 `content` / `text` 字段 |
| `delete_on_user_request=False` 必须 fail-loud | `EvidenceDeletionPolicy.__post_init__` 抛 ValueError |
| 双层 scope 删除不串 tenant | `tests/contracts/test_two_layer_scope_isolation.py` |

## 7. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW**（W3） | `EvidenceDeletionPolicy` + `EvidenceDeletionRecord` + `delete_evidence_files_for_scope` 落 `lifeform_service/evidence_deletion.py`；contract test 通过；本 spec v0.1 落档；DELETE endpoint scaffold 在 `protocol_routes` 加占位（fail-loud "scaffolded, not wired"） |
| **ACTIVE**（W6） | 4 个新 DELETE endpoint 真接进 closed-alpha + DLaaS 路由；ledger 真写；闭环测试覆盖 end-user / tenant-admin 两种 actor |

## 8. 风险

| 风险 | 应对 |
|---|---|
| ledger 文件无限增长 | 按日 rotate + meta-ledger 跟踪归档 |
| 删除请求并发与 evidence sweeper 冲突 | 优先级：用户请求 > sweeper；sweeper 见到正在删的文件跳过 |
| tenant_admin 误删全 tenant | HTTP body `confirm_tenant_purge` + 双因素 + 24h 软删除窗口 |

## 变更日志

- 2026-05-13: v0.1 SHADOW scaffold land。
