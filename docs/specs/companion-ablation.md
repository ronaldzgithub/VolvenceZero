# Companion Bench Same-Substrate Ablation Spec

> Status: P1 single-substrate-owner tooling landed; first nine-track real run pending weights/bootstrap/keys
> Last updated: 2026-07-14
> 对应需求: R2（稳定基底 + 自适应控制器）, R7（双轨/关系学习）, R8（快照优先）, R12（评估只读）, R15（可解释 + 可回滚证据）
> 关联 debt: #82 / #84 / #87（人类世界模型 thesis 第一阶段 retain 证据）

## 要解决的问题

把"Volvence 的认知控制器层在长程关系任务上优于标准方案"从叙事变成**可证伪、同基底、可复现**的实验。核心是一条因果切分：**固定 substrate，只变 substrate 之上的层**，任何分差都归因于该层而非基底。这是 thesis 第一阶段（人类世界模型）从 "harness-ready" 升到 "first-stage-retained" 的证据来源。

## 同基底矩阵（5 主 track + 4 component arms，全部跑在同一份冻结 Qwen）

| Track | substrate 之上的层 | 隔离的变量 |
|---|---|---|
| `raw` | 无（裸 Qwen） | 任何层 vs 无层的下界 |
| `ref-harness` | 标准 memory wrapper（[`packages/companion-ref-harness`](../../packages/companion-ref-harness)，summary+embed+user_model+episodic 四件） | "标准记忆封装"基线 |
| `camel` | 标准开源 agent 框架（[`packages/companion-camel-baseline`](../../packages/companion-camel-baseline)，CAMEL ChatAgent + memory） | "标准 agent 框架"基线 |
| `volvence-cold` | Volvence 完整 pipeline，**无**训练 bootstrap | 控制器结构本身的贡献 |
| `volvence` | Volvence 完整 pipeline + 训练 bootstrap | 被验证系统 |

`raw / ref-harness / camel / volvence-cold` 是 matched controls；`volvence` 是被验证系统。
`volvence vs raw` = 任何层是否有用；`volvence vs ref-harness/camel` = 控制器是否优于标准方案；`volvence vs cold` = 训练增量。
`pe-off / eta-off / active-learning-off / lora-adapter` 是 component-causal arms，用于第 5 条 `claim_component_causal_contribution`。

## 关键不变量

- **同基底字节级一致 + 单 runtime owner**：九条 track 必须命中同一份 Qwen 权重，且 Volvence/component tracks 必须由一个进程级冻结 substrate owner 承载。工程实现：一个 `lifeform-serve --ablation-bundle` 进程在 :8000 同时暴露 `mode=raw`（raw track）与 `mode=lifeform&vertical=<companion-arm>`（六个 Volvence/component tracks），`ref-harness`/`camel` 的 upstream 都指向 `:8000/v1?mode=raw`。`assert_same_substrate.py` 在评分前 fail-loud 校验每条 track 的 `substrate_fingerprint.json` 一致；`serve_topology.json` 记录 `single-lifeform-ablation-bundle` 与唯一 lifeform owner PID。
- **裁判/用户模拟器非 Qwen（#71/#72）**：substrate 是 Qwen，则 per-turn judge / arc judge / user-simulator 必须跨家族（默认 per-turn=Claude、arc=GPT-5、user-sim=Claude）。`run_same_substrate_ablation.py` 在 p1/p2 入口拒绝 Qwen 裁判。
- **baseline wheel 隔离**：`companion-ref-harness` 与 `companion-camel-baseline` 禁止 import `volvence_zero.*` / `lifeform_*` / `companion_bench.*`（AST 契约测试静态守门），否则"赢 baseline"会被自家内核污染。
- **公平同 prompt**：五条 track 共享同一 companion system prompt；只有记忆/agent/认知层不同。
- **evaluation 只读（R12）**：裁判分数是 readout，不回灌学习链路。

> **冻结的 thesis 第一阶段 claim registry SSOT 见 [`human-world-model-ablation.md`](./human-world-model-ablation.md)**——本 spec 是它的实现载体。同基底服务矩阵已扩到 9 track：5 个主 track 加 `PE-off` / `ETA-off` / `active-learning-off` / `LoRA-adapter` 四个 component arms。真 LoRA artifact 仍 gate on #41 GPU bake。

## 五条 retain claim（debt #87）与四态结论

`compare_companion_ablation.py` 读各 track `summary.json`，用 bootstrap CI 算保守非重叠下界 `ci_low = volvence.ci95_lo - control.ci95_hi`，输出：

1. `claim_pipeline_gt_raw` — volvence > raw
2. `claim_gt_standard_layers` — volvence > ref-harness **且** > camel
3. `claim_training_adds_value` — volvence > volvence-cold
4. `claim_heldout_cohort_stable` — volvence 的区间足够紧 + 跑在足够多 arc（held-out + 多 seed）
5. `claim_component_causal_contribution` — `pe-off` / `eta-off` / `active-learning-off` / `lora-adapter` 均显示组件正向因果贡献

每条取 `retain`（delta>0 且 ci_low>0）/ `weak`（delta>0 但 CI 触零）/ `fail`（delta<=0）/ `insufficient_data`。整体四态：

- `kill-criteria-triggered` — 1/2/3 任一 `fail`：按 #87 收缩 thesis 为 product-memory/companion 口径，降级 [`human-world-model-thesis-2026-06.md`](../../research/strategy/human-world-model-thesis-2026-06.md)。
- `wiring-ready` — 流程跑通但 track 不全，无法评估。
- `weak-positive` — 1/2/3 全正但未全 retain（或 stability 未 retain）。
- `first-stage-retained` — 1/2/3/4 全 retain：人类世界模型 thesis **第一阶段**可称 retained。
- `world-model-extension-ready` — **本脚本永不自动给出**：需要物理/具身侧独立 benchmark，人类侧成功不能外推。

## 执行阶段（对齐 paper-suite tier）

| Phase | 内容 | 入口 | 真钱 |
|---|---|---|---|
| P0 wiring | deterministic-fake 9-track 全链 + comparator | `run_same_substrate_ablation.py --phase p0-smoke` | 0 |
| judge-evidence | judge robustness + calibration（#48/#71，SHADOW scaffold） | `--phase judge-evidence` | 小 |
| P1 directional | 公开 30（24 en + 6 zh）× 1 seed，真 Qwen + 跨家族裁判 | `run_p1_windows.ps1` / `run_p1_apple.sh` / `--phase p1` | 中 |
| P2 retain | 30 公开 + 96 held-out × 3 seed | `--phase p2`（`--include-heldout --require-heldout`） | 批准预算 |

SUT 是自托管 Qwen（GPU，算力近似免费）；真钱主要在裁判 + 用户模拟器。

## 接口契约

**消费的输入**：companion-bench `summary.json`（`aggregate.final_mean` / `final_ci95` / `arc_count` / `axis_means`，`manifest.attestation` 四红线）；per-track `substrate_fingerprint.json`。

**产出的输出**：`verdict_{p1,p2}.json`（四 claim + 四态 state + pairwise effects + recommendations + substrate_note + timestamp）；judge robustness / calibration evidence artifact。

### Windows P1 readiness contract

Windows GPU 是 P1 directional 的一等开发执行面，但不是 retain evidence lane。

1. `volvence` 必须加载完整的 `companion-temporal.snap` +
   `companion-regime.bs` typed bootstrap pair；任一缺失即 fail loudly。
   `volvence-cold` 只显式关闭这两个 bootstrap，semantic proposal、
   affordance、PE/credit、memory 和 expression pipeline 与 `volvence` 相同。
2. `preflight_llm.py` 在付费调用前检查加速器（`--substrate-device`，默认
   `cuda`；Apple silicon 显式传 `mps`，`cpu` 允许但如实记录，设备不可用一律
   fail-loud、无静默降级）、console commands、模型缓存、bootstrap pair、
   端口、跨家族模型与 API 连通性，并写 `companion-p1-run-manifest.v1`。
3. fingerprint 必须包含实际权重文件的 `weights_sha256`；P1
   `assert_same_substrate.py --require-weights-sha256` 不接受只写 model id。
4. `serve_same_substrate_ablation.(sh|ps1)` 对五个 HTTP 服务（single lifeform
   ablation bundle / ref-harness / memory-only / rag / camel）逐端点轮询健康状态；P1 runner
   另做六个 `?vertical=` OpenAI 路由探针。探针超时由
   `--vertical-probe-timeout-s` 控制；Apple/MPS 入口默认 180s 以覆盖
   component arm 冷启动，固定时长 sleep 不构成 readiness。
5. `run_p1_windows.ps1` / `run_p1_apple.sh` 是对应平台的 SSOT 入口；正常与异常退出均清理本轮 PID。
   `-Resume` / `--resume` 只复用已有合法 `summary.json`，DryRun 不启动 GPU/API。
   Apple/MPS 入口默认 `VZ_P1_SUT_MAX_TOKENS=96`（可覆盖；Apple CPU 保持
   `256`），配合 MPS generation input cap 与逐轮 allocator cache 释放，避免
   unified-memory 压力触发 macOS jetsam；平台差异必须随 run manifest / logs
   一并留痕。正式运行前还必须通过磁盘安全门：默认至少保留 `15 GiB`
   （`VZ_P1_MIN_FREE_DISK_GIB` 可提高）供 artifact 与 macOS swap 使用；不足则在
   API preflight 和服务启动前 fail-loud。Windows/CUDA 入口及其 token budget
   不受此约束影响。
6. run manifest 记录 git SHA/clean 状态、权重与 bootstrap hash、learned
   backend wiring、judge model id、serving topology 与 ablation vertical 集合，
   但绝不记录 key 值。
7. `serve_topology.json` 必须声明 `process_count=5` 与
   ports `[8000, 8500, 8501, 8502, 8600]`（GAP-11 memory-only / rag 独立进程）。

P1 结果最多是 directional `weak-positive` 或 directional kill signal。
单 seed/public-only 不能产生 `first-stage-retained`，也不得作为对外 thesis 证据。

## 组件清单（已落地）

- [`packages/companion-camel-baseline/`](../../packages/companion-camel-baseline) — CAMEL agent-framework baseline wheel（Apache 2.0，隔离）。
- [`packages/companion-ref-harness/`](../../packages/companion-ref-harness) — 标准 memory wrapper，H-A+H-B+H-C 四件全开。
- [`scripts/companion_bench/reference_systems.same_substrate_ablation.yaml`](../../scripts/companion_bench/reference_systems.same_substrate_ablation.yaml) — 9-track roster（5 主轨 + 4 component arms）。
- [`scripts/companion_bench/serve_same_substrate_ablation.sh`](../../scripts/companion_bench/serve_same_substrate_ablation.sh) + [`stop_same_substrate_ablation.sh`](../../scripts/companion_bench/stop_same_substrate_ablation.sh) — 单 lifeform substrate owner + ref-harness + camel 的同基底服务编排。
- [`scripts/companion_bench/assert_same_substrate.py`](../../scripts/companion_bench/assert_same_substrate.py) — 同基底 fingerprint 守门。
- [`scripts/companion_bench/compare_companion_ablation.py`](../../scripts/companion_bench/compare_companion_ablation.py) — #87 五 claim verdict。
- [`scripts/companion_bench/run_same_substrate_ablation.py`](../../scripts/companion_bench/run_same_substrate_ablation.py) — P0/judge-evidence/P1/P2 phased driver。
- [`scripts/companion_bench/run_p1_windows.ps1`](../../scripts/companion_bench/run_p1_windows.ps1) — Windows P1 preflight / serve / score / verdict / cleanup 一键入口。
- [`run_companion_bench_p1.sh`](../../run_companion_bench_p1.sh) + [`scripts/companion_bench/run_p1_apple.sh`](../../scripts/companion_bench/run_p1_apple.sh) — Apple-silicon (MPS) P1 一键入口；与 Windows 入口同构（preflight → serve 9-track → score → verdict → teardown），默认打开四个 torch learned backends（`active`）。
- [`scripts/launch_evidence_runs_m2.sh`](../../scripts/launch_evidence_runs_m2.sh) — Apple-silicon evidence 套件：setup / soak / ablation / all；ablation 子命令可作重试型 wrapper，薄 P1 入口优先用 `run_companion_bench_p1.sh`。
- [`scripts/companion_bench/p1_readiness.py`](../../scripts/companion_bench/p1_readiness.py) — 权重 digest、bootstrap/run manifest 与本地 fail-loud gate。
- 结果记录：[`docs/external/companion-ablation-results-internal.md`](../external/companion-ablation-results-internal.md)。

## 与其他能力域的关系

依赖评估体系（companion-bench 6 轴）与证据计划（[`evidence_program.md`](./evidence_program.md) claim registry）；被 thesis / 融资尽调消费。物理侧扩张是独立 benchmark，不在本 spec。

## 变更日志

- 2026-07-14：P1 serving 拓扑从 6 个重复 `lifeform-serve --substrate-mode hf-shared`
  进程收敛为一个 `lifeform-serve --ablation-bundle` 进程。六个
  Volvence/component tracks 通过 `?vertical=` 显式路由共享同一冻结 HF runtime；
  ref-harness/camel 继续独立进程并 upstream 到 `:8000/v1?mode=raw`。P1
  readiness 新增 `serving_topology` / `serve_topology.json` 与六轨 vertical probe，
  防止"同权重但多 GPU 副本"再次污染同基底因果解释。
- 2026-06-28：tooling 全套落地（CAMEL baseline wheel + ref-harness H-B/H-C + roster + serve + substrate guard + comparator + phased driver + P0 wiring smoke 通过）。首个真跑（P1/P2）待 GPU/keys。
- 2026-07-14：component arms serving 迁入同基底矩阵：新增 `companion-pe-drive-off` / `companion-eta-off` / `companion-active-learning-off` / `companion-lora-adapter` verticals；roster、serve launcher、preflight fingerprints、P1/P2 health / summary / fingerprint gate 均扩到 9 track。真 LoRA artifact 仍等待 #41 GPU bake。
- 2026-07-13：Windows P1 readiness 收敛：companion/cold 同完整 runtime
  接线、bootstrap pair hard gate、实际权重 SHA256、逐端点 health、run manifest、
  resume/dry-run 与 PowerShell 一键编排落地；付费五轨 P1 仍未执行。
- 2026-07-14：Apple-silicon 执行面落地。`p1_readiness.require_accelerator`
  显式支持 `cuda / mps / cpu`（fail-loud，无静默降级；`require_cuda` 保留），
  `preflight_llm.py` 增 `--substrate-device`；`serve_same_substrate_ablation.sh`
  与 `.ps1` 对齐（保留 preflight 写入的 weights_sha256 fingerprint 不覆盖、
  ref-harness embedder/extractor 与 camel compaction fail-loud、逐端点健康轮询
  取代固定 sleep）；新增 `launch_evidence_runs_m2.sh` 一键入口。MPS lane 与
  Windows lane 同为 directional，不构成 retain evidence lane。
- 2026-07-14：新增与 Windows `run_companion_bench_p1.ps1` 对称的 Apple 薄入口
  `run_companion_bench_p1.sh` / `run_p1_apple.sh`。默认 `VZ_SUBSTRATE_DEVICE=mps`
  且四个 owner-local torch backends（SSL / temporal runtime / Internal RL / CMS）
  为 `active`；复用既有 `serve_same_substrate_ablation.sh` 九轨拓扑。
