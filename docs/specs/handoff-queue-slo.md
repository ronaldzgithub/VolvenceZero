# Handoff Queue SLO Spec

> Status: scaffold v0.2 (SHADOW；single-process N=10 stress test 接通)
> Owner: growth-advisor-pilot-packet G-E (debt #70)
> Implementation: [`packages/dlaas-platform-registry/src/dlaas_platform_registry/handoff.py`](../../packages/dlaas-platform-registry/src/dlaas_platform_registry/handoff.py) `HandoffTicketStore`

## 1. 范围

closed-alpha 已有 handoff queue（[`packages/dlaas-platform-registry/src/dlaas_platform_registry/handoff.py`](../../packages/dlaas-platform-registry/src/dlaas_platform_registry/handoff.py) — 注：debt #70 描述位置为 `dlaas-platform-ops` 与现实不符，按现实位置接），但是 demo 级。P2 30 天试点客户上量后必须有明确 SLO。

## 2. SLO

| 指标 | 阈值 | 检测 |
|---|---|---|
| Queue capacity per tenant | 100 pending | implementation 层硬上限 |
| Pickup latency P50 | < 10s | metric collection |
| Pickup latency P99 | < 30s | metric collection |
| Pickup deadline (max) | 5 min | beyond → fallback 触发 |
| SE-on-call uptime | ≥ 99% | ops dashboard |

## 3. Fallback 行为

- 5 min 仍未接手 → 触发 fallback：**STRICT_REFUSE** 模式（拒绝继续对话 + 道歉 + 给客服联系方式），**不**降级回 LLM 自由回答
- 推理：降级到 LLM 自由回答违反"用户主动 escalate 的意图"——他们已经表态需要真人

## 4. State 持久化 + 跨重启恢复

- handoff state 持久化到 `evidence_root_dir/handoff_queue_state.jsonl`（append-only）
- 服务重启后从 state 文件 resume；未完成的 handoff 重新 enqueue
- evidence 删除（debt #49）路径：删 end_user 时其 handoff state 也删（写 deletion ledger）

## 5. Tenant 隔离

每 tenant 独立队列。tenant A 的 handoff 不能挤压 tenant B 的 SE 接手时延（按 tenant 调度，不按全局 FIFO）。

依赖 debt #46 双层 scope（tenant_id × end_user_id）。

## 6. ops dashboard

实时 view：
- 每 tenant 当前 queue 长度
- 各 SE 当前接手数
- alert: queue 长度超 80 / pickup latency P99 超 30s / SE-on-call 离线

## 7. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW v0.1**（W6 准备） | 本 spec 落档 + `tests/perf/test_handoff_queue_concurrent_load.py` 骨架（依赖横切 F-A perf 床）|
| **SHADOW v0.2**（W6+） | 单进程 N=10 async stress test 真跑（不依赖 GPU/真生产负载）：验证 HandoffTicketStore.create / list_for_ai / submit_human_reply 在并发下的行为正确性；spec 加 capacity / timeout / resume 字段细节 |
| **ACTIVE v1.0**（W8） | 真跑 N=50 end_user × 10 tenant 并发 SLO 测试通过（依赖 F-A perf 床）；ops dashboard alert 接通；fallback STRICT_REFUSE 路径真生效；cross-restart resume 真验证 |

## 8. v0.2 capacity / timeout / resume 字段细节

### 8.1 Capacity policy

- per-tenant queue 硬上限 100 pending（HandoffTicketStore 创建时校验）
- 超 80 pending → ops dashboard 黄色 alert（提前预警）
- 超 100 pending → 拒绝创建新 ticket，返回 `HandoffQueueFullError`，触发 `STRICT_REFUSE` fallback（debt #70 §3 已规约）

### 8.2 Timeout fallback

- 5 min 未接手 → 触发 fallback `STRICT_REFUSE` 模式
- fallback 路径必须**显式**走（不是默默降级回 LLM 自由回答）
- ticket status 转 `TIMEOUT_FALLBACK`，operator_id 留空，resolution_notes 自动填 "auto-fallback (5min timeout)"

### 8.3 Cross-restart resume

- HandoffTicketStore 走 SQLite 持久化（已实现），所以重启后 OPEN 状态 ticket 自动可见
- 重启后启动 reconciliation：扫描所有 OPEN 状态 ticket，已超 5 min 的立即转 TIMEOUT_FALLBACK；剩余的回到正常队列等接手
- evidence 删除（debt #49）路径：删 end_user 时其 OPEN handoff state 也删除（写 deletion ledger 含 ticket_id sha256）

## 9. 单进程 N=10 stress test scope（v0.2）

[`tests/perf/test_handoff_queue_concurrent_load.py`](../../tests/perf/test_handoff_queue_concurrent_load.py) 加 `test_handoff_queue_single_process_concurrent_creates_ok`：

- N=10 端用户并发 `await store.create(...)` 在同一 SQLite-backed Registry
- 验证：(a) 10 个 ticket_id 唯一；(b) `list_for_ai(...)` 返 10 条；(c) 每条 status=OPEN；(d) 跨并发 ticket_id 不冲突
- 跑时间预算 < 5s（单进程内存 SQLite，不需要外部资源）
- 仍 `@pytest.mark.perf` 默认 skip；`pytest tests/perf/ -m perf` 显式跑

真生产 N=50 × 10 tenant 测试（依赖 F-A perf 床）等 G-E ACTIVE。

## 8. 风险

| 风险 | 应对 |
|---|---|
| 单 SE-on-call 同时多 tenant 时 burnout | 按 tenant SE quota 控制；超限自动 fallback |
| evidence 删 end_user 后 handoff state 引用悬空 | deletion 路径同步删 handoff state；ledger 留痕 |
| 试点客户突然热度峰值（如品牌方推广） | queue capacity 100 pending 自动 throttle；超限触发限流 + 通知客户 ops |

## 变更日志

- 2026-05-13: v0.1 SHADOW spec。
- 2026-05-14: v0.2 — capacity / timeout / resume 字段细节落档（§8）；single-process N=10 async stress test 真接通（§9，不依赖 F-A perf 床；真 N=50 × 10 tenant 等 G-E ACTIVE）。
