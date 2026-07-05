# Companion Bench Same-Substrate Ablation Spec

> Status: draft (tooling landed; first real run pending GPU/keys)
> Last updated: 2026-06-28
> 对应需求: R2（稳定基底 + 自适应控制器）, R7（双轨/关系学习）, R8（快照优先）, R12（评估只读）, R15（可解释 + 可回滚证据）
> 关联 debt: #82 / #84 / #87（人类世界模型 thesis 第一阶段 retain 证据）

## 要解决的问题

把"Volvence 的认知控制器层在长程关系任务上优于标准方案"从叙事变成**可证伪、同基底、可复现**的实验。核心是一条因果切分：**固定 substrate，只变 substrate 之上的层**，任何分差都归因于该层而非基底。这是 thesis 第一阶段（人类世界模型）从 "harness-ready" 升到 "first-stage-retained" 的证据来源。

## 同基底矩阵（5 track，全部跑在同一份冻结 Qwen）

| Track | substrate 之上的层 | 隔离的变量 |
|---|---|---|
| `raw` | 无（裸 Qwen） | 任何层 vs 无层的下界 |
| `ref-harness` | 标准 memory wrapper（[`packages/companion-ref-harness`](../../packages/companion-ref-harness)，summary+embed+user_model+episodic 四件） | "标准记忆封装"基线 |
| `camel` | 标准开源 agent 框架（[`packages/companion-camel-baseline`](../../packages/companion-camel-baseline)，CAMEL ChatAgent + memory） | "标准 agent 框架"基线 |
| `volvence-cold` | Volvence 完整 pipeline，**无**训练 bootstrap | 控制器结构本身的贡献 |
| `volvence` | Volvence 完整 pipeline + 训练 bootstrap | 被验证系统 |

`raw / ref-harness / camel / volvence-cold` 是 matched controls；`volvence` 是被验证系统。
`volvence vs raw` = 任何层是否有用；`volvence vs ref-harness/camel` = 控制器是否优于标准方案；`volvence vs cold` = 训练增量。

## 关键不变量

- **同基底字节级一致**：五条 track 必须命中同一份 Qwen 权重。工程实现：一个 `lifeform-serve` 进程在 :8000 同时暴露 `mode=lifeform`（volvence）与 `mode=raw`（raw track），`ref-harness`/`camel` 的 upstream 都指向 `:8000/v1?mode=raw`，`volvence-cold` 在 :8001 加载同一 `--substrate-model-id`。`assert_same_substrate.py` 在评分前 fail-loud 校验每条 track 的 `substrate_fingerprint.json` 一致。
- **裁判/用户模拟器非 Qwen（#71/#72）**：substrate 是 Qwen，则 per-turn judge / arc judge / user-simulator 必须跨家族（默认 per-turn=Claude、arc=GPT-5、user-sim=Claude）。`run_same_substrate_ablation.py` 在 p1/p2 入口拒绝 Qwen 裁判。
- **baseline wheel 隔离**：`companion-ref-harness` 与 `companion-camel-baseline` 禁止 import `volvence_zero.*` / `lifeform_*` / `companion_bench.*`（AST 契约测试静态守门），否则"赢 baseline"会被自家内核污染。
- **公平同 prompt**：五条 track 共享同一 companion system prompt；只有记忆/agent/认知层不同。
- **evaluation 只读（R12）**：裁判分数是 readout，不回灌学习链路。

> **冻结的 thesis 第一阶段 claim registry SSOT 见 [`human-world-model-ablation.md`](./human-world-model-ablation.md)**——本 spec 是它的实现载体（同基底 5-track 工具链）。registry 在下面四条之上新增 `claim_component_causal_contribution`（PE/ETA/主动学习逐个因果切分），对应的 `PE-off`/`ETA-off`/`active-learning-off`/`LoRA-adapter` 四臂尚未迁到同基底矩阵。

## 四条 retain claim（debt #87）与四态结论

`compare_companion_ablation.py` 读各 track `summary.json`，用 bootstrap CI 算保守非重叠下界 `ci_low = volvence.ci95_lo - control.ci95_hi`，输出：

1. `claim_pipeline_gt_raw` — volvence > raw
2. `claim_gt_standard_layers` — volvence > ref-harness **且** > camel
3. `claim_training_adds_value` — volvence > volvence-cold
4. `claim_heldout_cohort_stable` — volvence 的区间足够紧 + 跑在足够多 arc（held-out + 多 seed）

每条取 `retain`（delta>0 且 ci_low>0）/ `weak`（delta>0 但 CI 触零）/ `fail`（delta<=0）/ `insufficient_data`。整体四态：

- `kill-criteria-triggered` — 1/2/3 任一 `fail`：按 #87 收缩 thesis 为 product-memory/companion 口径，降级 [`human-world-model-thesis-2026-06.md`](../../research/strategy/human-world-model-thesis-2026-06.md)。
- `wiring-ready` — 流程跑通但 track 不全，无法评估。
- `weak-positive` — 1/2/3 全正但未全 retain（或 stability 未 retain）。
- `first-stage-retained` — 1/2/3/4 全 retain：人类世界模型 thesis **第一阶段**可称 retained。
- `world-model-extension-ready` — **本脚本永不自动给出**：需要物理/具身侧独立 benchmark，人类侧成功不能外推。

## 执行阶段（对齐 paper-suite tier）

| Phase | 内容 | 入口 | 真钱 |
|---|---|---|---|
| P0 wiring | deterministic-fake 5-track 全链 + comparator | `run_same_substrate_ablation.py --phase p0-smoke` | 0 |
| judge-evidence | judge robustness + calibration（#48/#71，SHADOW scaffold） | `--phase judge-evidence` | 小 |
| P1 directional | 公开 24 × 1 seed，真 Qwen + 跨家族裁判 | `--phase p1` | 中 |
| P2 retain | 24 公开 + 96 held-out × 3 seed | `--phase p2`（`--include-heldout --require-heldout`） | 批准预算 |

SUT 是自托管 Qwen（GPU，算力近似免费）；真钱主要在裁判 + 用户模拟器。

## 接口契约

**消费的输入**：companion-bench `summary.json`（`aggregate.final_mean` / `final_ci95` / `arc_count` / `axis_means`，`manifest.attestation` 四红线）；per-track `substrate_fingerprint.json`。

**产出的输出**：`verdict_{p1,p2}.json`（四 claim + 四态 state + pairwise effects + recommendations + substrate_note + timestamp）；judge robustness / calibration evidence artifact。

## 组件清单（已落地）

- [`packages/companion-camel-baseline/`](../../packages/companion-camel-baseline) — CAMEL agent-framework baseline wheel（Apache 2.0，隔离）。
- [`packages/companion-ref-harness/`](../../packages/companion-ref-harness) — 标准 memory wrapper，H-A+H-B+H-C 四件全开。
- [`scripts/companion_bench/reference_systems.same_substrate_ablation.yaml`](../../scripts/companion_bench/reference_systems.same_substrate_ablation.yaml) — 5-track roster。
- [`scripts/companion_bench/serve_same_substrate_ablation.sh`](../../scripts/companion_bench/serve_same_substrate_ablation.sh) + [`stop_same_substrate_ablation.sh`](../../scripts/companion_bench/stop_same_substrate_ablation.sh) — 同基底服务编排。
- [`scripts/companion_bench/assert_same_substrate.py`](../../scripts/companion_bench/assert_same_substrate.py) — 同基底 fingerprint 守门。
- [`scripts/companion_bench/compare_companion_ablation.py`](../../scripts/companion_bench/compare_companion_ablation.py) — #87 四 claim verdict。
- [`scripts/companion_bench/run_same_substrate_ablation.py`](../../scripts/companion_bench/run_same_substrate_ablation.py) — P0/judge-evidence/P1/P2 phased driver。
- 结果记录：[`docs/external/companion-ablation-results-internal.md`](../external/companion-ablation-results-internal.md)。

## 与其他能力域的关系

依赖评估体系（companion-bench 6 轴）与证据计划（[`evidence_program.md`](./evidence_program.md) claim registry）；被 thesis / 融资尽调消费。物理侧扩张是独立 benchmark，不在本 spec。

## 变更日志

- 2026-06-28：tooling 全套落地（CAMEL baseline wheel + ref-harness H-B/H-C + roster + serve + substrate guard + comparator + phased driver + P0 wiring smoke 通过）。首个真跑（P1/P2）待 GPU/keys。
