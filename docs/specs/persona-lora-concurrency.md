# PersonaLoRA Pool Concurrency Spec

> Status: scaffold v0.1 (SHADOW)
> Last updated: 2026-05-13
> Owner: figure-evidence-packet G-D (debt #61)

## 1. 范围

`PersonaLoRAPool.activate(figure_id, runtime=runtime)` 在生产并发条件下的状态隔离 + 性能 SLO。N session 同时 activate 不同 figure LoRA，必须保证：

1. 每 session forward 看到的是自己 LoRA 的 logits（无 race）
2. frozen base `state_dict_hash` 不变（R2 守门）
3. swap overhead < 200ms / turn（参考 [`perf-baseline.md`](perf-baseline.md) 表）

## 2. 当前状态

`LoRAAwareResidualRuntime` Protocol（debt #20 closure，Wave D）已实现 single-session activate。但 **多并发** activate 未测：

- 单 process 多 asyncio task 同时 activate？
- forward hook 状态在 task 间是否串？
- 嵌套 activate 抛 RuntimeError 是单线程语义，asyncio 下 race 怎么处理？

### 2.1 2026-05-29 实现进展（DLaaS substrate + LoRA packet）

部分推进，给出**正确性**隔离 evidence（GPU 吞吐基线仍待 #45 perf 床）：

- **Per-tenant 作用域池**：每个 `SessionManager` 持自有
  `PersonaLoRAPool` 并把它传给 synthesizer，激活查 scoped 池而非
  全局 `default_persona_lora_pool`（消除 last-register-wins）。
  `_maybe_activate_persona_lora(..., pool=scoped_pool, enabled=...)`。
- **并发隔离 evidence**：
  [`test_persona_lora_concurrency_evidence.py`](../../packages/lifeform-service/tests/test_persona_lora_concurrency_evidence.py)
  —— N=12 个 asyncio task，同 `figure_id` 不同 checkpoint，交错激活后
  每 task 只看到自己的 checkpoint（无串扰）；adapter_policy=none 时
  全部不激活。
- **vLLM 并发后端**：`VLLMOpenWeightResidualRuntime` 用
  per-request `LoRARequest`，active checkpoint 走 `contextvars`
  按 task 隔离 —— transformers 的 "serial-decode + 单线程嵌套守门"
  假设在 vLLM 后端放松为 per-task。嵌套 activate 仍在同 task 抛
  `RuntimeError`，跨 task 并发不再误判。
- **仍缺**：真 GPU 上 N≥10 并发 logits determinism + swap overhead
  P50/P99 实测（依赖 #45）。本批是单测级隔离证明，非 GPU 吞吐基线。

## 3. SLO

| 指标 | 阈值 | 检测 |
|---|---|---|
| 并发 figure 数 | ≥ 10 | `tests/perf/test_persona_lora_hot_swap_concurrency.py` |
| Swap overhead P50 | ≤ 200ms | 同上 |
| Swap overhead P99 | ≤ 500ms | 同上 |
| Per-session logits determinism | 100% | 同上：每 session 单独跑 + 并发跑，logits L1 距离 < 1e-6 |
| Frozen base state_dict_hash | unchanged 全程 | 同上：activate 前 / 中 / 后 三次抽 SHA-256 |

## 4. 实施策略（待 ACTIVE）

### Option A: per-layer asyncio.Lock（推荐）

每 `delta_vector` 应用到 attention block 时加锁。开销小，但限制并发。

### Option B: per-session forward hook clone

每 session 进入 activate context 时 clone forward hook；退出时 restore。开销大，但完全隔离。

### Option C: 多 process（fallback）

如 Option A/B 都不 work，回退到 launcher 多进程（一个进程 = 一个 figure 实例），代价是 substrate 加载 N 次。

ACTIVE 阶段先试 Option A，stress test 不通过升 Option B；最差 Option C。

## 5. 与横切 F-A perf 床的耦合

本 spec 测试在 `tests/perf/test_persona_lora_hot_swap_concurrency.py`（debt #61），依赖横切 F-A perf 床的 `concurrent_lifeform_factory` + `gpu_mem_tracker` fixture（已 SHADOW land）。

ACTIVE 节奏：F-A ACTIVE (W5) → 本测试 ACTIVE (W7)。

## 6. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW**（W1） | `tests/perf/test_persona_lora_hot_swap_concurrency.py` 骨架 + 本 spec v0.1 |
| **ACTIVE**（W7） | 真跑 N=10 figure 并发 × Qwen substrate × 100 turn → byte-identical logits + frozen base 不变 + swap < 200ms |

## 变更日志

- 2026-05-13: v0.1 SHADOW scaffold。
