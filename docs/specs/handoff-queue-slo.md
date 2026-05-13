# Handoff Queue SLO Spec

> Status: scaffold v0.1 (SHADOW)
> Owner: growth-advisor-pilot-packet G-E (debt #70)

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
| **SHADOW**（W6 准备） | 本 spec v0.1 + `tests/perf/test_handoff_queue_concurrent_load.py` 骨架（依赖横切 F-A perf 床）|
| **ACTIVE**（W8） | 真跑 N=50 end_user × 10 tenant 并发 SLO 测试通过；ops dashboard alert 接通；fallback STRICT_REFUSE 路径真生效 |

## 8. 风险

| 风险 | 应对 |
|---|---|
| 单 SE-on-call 同时多 tenant 时 burnout | 按 tenant SE quota 控制；超限自动 fallback |
| evidence 删 end_user 后 handoff state 引用悬空 | deletion 路径同步删 handoff state；ledger 留痕 |
| 试点客户突然热度峰值（如品牌方推广） | queue capacity 100 pending 自动 throttle；超限触发限流 + 通知客户 ops |

## 变更日志

- 2026-05-13: v0.1 SHADOW spec。
