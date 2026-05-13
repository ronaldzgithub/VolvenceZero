# Perf Baseline Spec

> Status: scaffold v0.1 (SHADOW)
> Last updated: 2026-05-13
> Owner: cross-cutting-foundation-packet F-A (debt #45)
> 对应 commercialisation-evidence-rollout.md §3 W2 SHADOW + W5 ACTIVE

## 1. 范围

定义 VolvenceZero 三个商业方向（companion / figure / growth-advisor）在生产并发条件下的 SLO baseline。**baseline 数据由 ACTIVE 阶段的 `tests/perf/` 与 `scripts/realistic_load_*.py` 真跑回填**；本 spec 当前只锁定 SLO **目标值** + 测量方法。

## 2. 三方向 SLO 目标（ACTIVE）

| 维度 | 维度说明 | companion | figure | growth-advisor |
|---|---|---|---|---|
| P50 turn latency | 用户体感主观慢的拐点 | < 1.5s | < 2.5s | < 1.5s |
| P99 turn latency | 投诉率拐点 | < 3.0s | < 5.0s | < 3.0s |
| 并发 ai_id 上限 | 单 substrate 节点 | ≥ 50 | ≥ 20 | ≥ 100 |
| GPU mem peak | 单节点峰值 | < 70% capacity | < 80% capacity | < 50% capacity |
| Owner snapshot dispatch P50 | 5 vertical 共载下 | < 50ms | < 50ms | < 50ms |
| L3 引证率（figure 专属） | bundle 上线 SLA | n/a | ≥ 0.95 | n/a |
| L4 拒答率（figure 专属） | OOS 题命中拒答 | n/a | ≥ 0.85 | n/a |
| Boundary 触发率（growth-advisor 专属） | per-boundary | n/a | n/a | ∈ [0.05, 0.50] |
| LoRA swap overhead P50 | hot-swap 开销 | n/a | < 200ms | n/a |
| Handoff queue P99 | 触发到 SE 接手 | n/a | n/a | < 30s |

**baseline 测量条件**：
- substrate：Qwen2.5-32B-Instruct（默认）/ 可切 Llama-3.1-70B 做对照
- GPU：单卡 A100-80G 等价物
- 跑分时长：30 min sustained load
- 并发模型：`asyncio.gather` N 个 LifeformSession 各跑 turn 序列

## 3. 测量方法

### 3.1 单元 perf test（`tests/perf/`）

| 文件 | 覆盖维度 | 默认状态 |
|---|---|---|
| `test_concurrent_lifeform_sessions.py` | P50 / P99 turn latency × 3 vertical | `@pytest.mark.perf` skip-by-default |
| `test_multi_vertical_owner_propagation.py` | Owner snapshot dispatch + PE owner 跨 vertical 隔离 | 同上 |
| `test_persona_lora_hot_swap_concurrency.py` | figure LoRA hot-swap × 10 并发 + frozen base byte-identical | `@pytest.mark.perf @pytest.mark.hf` 双标 |

### 3.2 真负载脚本（`scripts/realistic_load_*.py`）

每个脚本走对应方向的真实服务面（closed-alpha 或 DLaaS），跑 30 min sustained，输出 `artifacts/perf/<vertical>-<date>.json` 带 P50/P99/GPU mem/方向特定指标（L3/L4/boundary trigger 等）。

### 3.3 跨方向 baseline 复现条件

- 每周一夜跑一次（`@pytest.mark.perf` 走 nightly tier）
- 每次 substrate 升级后必跑（参见 [`substrate-upgrade-protocol.md`](substrate-upgrade-protocol.md)）
- 任何 latency 相关重构 PR 必跑

## 4. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW**（W1-W2） | `tests/perf/` 目录骨架 + 3 核心 contract test 通过；3 个 realistic_load 脚本能 `--dry-run` 出 placeholder artifact；本 spec v0.1 落档 |
| **ACTIVE**（W5） | Qwen-1.5B baseline 上 3 个 vertical SLO 全达标；3 个 realistic_load 真跑 30 min 出真 artifact；nightly perf workflow 在 CI 上线 |

## 5. SSOT 约束

1. perf 测试**只读** owner snapshot / metrics，**不写**任何 kernel owner（R8 / R12）
2. GPU 内存采样是 telemetry readout，**不**反向喂 reward / Face / 任何学习信号
3. 所有 SLO 目标值的修订必须 commit 到本 spec，禁止散落在 test 文件常量
4. 跨 vertical PE owner 必须隔离（[`test_multi_vertical_owner_propagation.py`](../../tests/perf/test_multi_vertical_owner_propagation.py) 守门）

## 6. 与 commercialisation 单位经济的对账

- §6.1 假设单 ai_id × 月成本 ≈ 30 元（包含 thinking/followup/tick 内部 turn）— 由本 spec ACTIVE baseline 数据回填验证
- §6.4 P3（C 端陪伴）单位经济假设 ARPU 80 元/月 — 依赖本 spec 的 P99 latency < 3s SLO 才能保证用户感知不明显慢
- §6.3 P2 多席位（10 席位/客户）— 依赖 growth-advisor 并发 ai_id ≥ 100 的 SLO

## 变更日志

- 2026-05-13: v0.1 SHADOW scaffold land，SLO 目标值定义；baseline 数据待 ACTIVE 阶段回填。
