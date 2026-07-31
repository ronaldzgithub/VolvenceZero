# Perf Baseline Spec

> Status: SHADOW harness implemented; production baselines not captured
> Last updated: 2026-08-01
> Owner: cross-cutting performance evidence (debt #45)

## 1. 范围与当前结论

Volvence 当前有七个产品 vertical，但本 spec 只锁定已有真负载脚本的三个商业面：`companion` / `figure` / `growth-advisor`。表中数值都是 **SLO 目标**，不是已验证基线。

当前代码现状：

- `tests/perf/` 已有并发 session、多 vertical owner 传播、PersonaLoRA、handoff queue 和 production rollback drill 测试。
- 三个 `scripts/realistic_load_*.py` 只支持 `--dry-run` placeholder artifact；非 dry-run 会明确拒绝，尚未连接真实 HTTP/DLaaS 负载。
- 仓库中没有可作为 30 分钟 sustained production baseline 的当前 artifact，因此不得声称 ACTIVE 达标。

## 2. SLO 目标（未验证）

| 维度 | companion | figure | growth-advisor |
|---|---:|---:|---:|
| P50 turn latency | < 1.5s | < 2.5s | < 1.5s |
| P99 turn latency | < 3.0s | < 5.0s | < 3.0s |
| 单 substrate 节点并发 ai_id | ≥ 50 | ≥ 20 | ≥ 100 |
| GPU memory peak | < 70% | < 80% | < 50% |
| Owner snapshot dispatch P50 | < 50ms | < 50ms | < 50ms |
| Figure LoRA swap P50 / P99 | n/a | ≤ 200ms / ≤ 500ms | n/a |
| Figure L3 引证率 / L4 拒答率 | n/a | ≥ 0.95 / ≥ 0.85 | n/a |
| Boundary 触发率 | n/a | n/a | [0.05, 0.50] |
| Handoff queue P99 | n/a | n/a | < 30s |

基线测量条件必须随 artifact 固定：模型 ID 与 weights fingerprint、后端、硬件、并发数、轨迹数、持续时间、warm-up 策略、代码 revision 与 wiring config。原 v0.1 的 Qwen2.5-32B/A100-80G/30min 只是建议基准环境，不是仓库默认运行时事实。

## 3. 已落地的测量面

| 文件 | 范围 | 证据等级 |
|---|---|---|
| `tests/perf/test_concurrent_lifeform_sessions.py` | 并发 Lifeform session latency scaffold | synthetic/perf harness |
| `tests/perf/test_multi_vertical_owner_propagation.py` | owner dispatch 与 PE 隔离 | contract/perf harness |
| `tests/perf/test_persona_lora_hot_swap_concurrency.py` | PersonaLoRA 并发与 frozen-base 守门 | `perf` + `hf`; 需真 GPU |
| `tests/perf/test_handoff_queue_concurrent_load.py` | handoff queue 并发 | perf harness |
| `tests/perf/test_production_rollback_drill.py` | 生产形态回滚 drill | `perf` + `hf`; 需真环境 |
| `scripts/realistic_load_{companion,figure,growth_advisor}.py` | artifact schema / CLI | dry-run placeholder only |

## 4. ACTIVE 准入

ACTIVE 需要同时满足：

1. 三个真负载脚本接入实际服务面，非 dry-run 运行不再拒绝。
2. 每个 vertical 产出至少一份 30 分钟 sustained artifact，包含上述完整复现指纹。
3. 相关 SLO 全部达标，失败值不得被 placeholder 或 skip 隐藏。
4. nightly/per-release 执行责任有明确 owner，substrate 升级后必须对比 N/N+1。

## 5. SSOT 与回滚

- perf 代码只读 owner snapshot / telemetry，不回写 kernel owner，不把 latency/GPU 指标当作学习 reward。
- 目标值只在本 spec 修订；实测值只来自带 provenance 的 artifact。
- 任一真负载 gate 失败时保持 SHADOW，回滚到上一个 weights fingerprint / artifact 组合，不通过放宽断言推广。

## 变更日志

- 2026-08-01: 对账当前 perf 测试与三个 dry-run 脚本；明确 SLO 是未验证目标，不再把 scaffold 描述成 ACTIVE baseline。
- 2026-05-13: v0.1 SHADOW scaffold 落地，定义首批 SLO 目标。
