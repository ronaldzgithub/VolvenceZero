# VolvenceZero Cognitive AGI 当前状态

> Status: live status summary
> Last updated: 2026-08-01（#92 后研究路线校正）
> 详细判断、晋升协议与命令见 [`current.md`](./current.md)。
> 本文件只记录当前事实、剩余代码、晋升状态和下一步，不把计划写成已完成。

> **终局证据边界**：#92 已以 `thesis-rejected` CLOSED。mechanism coverage
> 完整；causal-supported=Gate 2/8/11，longitudinal-supported=Gate 8/11，
> full-chain rollback 通过，但整体 thesis 与 production/live ACTIVE 晋升
> 均未获授权。Ecology station1 四门通过，唯一 alignment review 后仍 3/4，
> 因而 station2/P1/P2 均按预注册 kill。权威对账：
> `artifacts/causal_evidence_final_campaign_20260731/`。

## 1. 当前总状态

| 维度 | 当前状态 | 完成度 | 还差什么 |
|---|---|---:|---|
| 架构 / 契约 / owner / 回滚骨架 | P0 continuity + P1 SHADOW learners 均已接线 | 约 90–94% | groups / protocol slow loop ACTIVE 仍待证据门 |
| 第一阶段认知系统代码 | P0 关键代码 + P1 近期缺口（G1–G5）已补 | 约 91–95% | 主要剩 World/Self model 扩容与 evidence-gated ACTIVE |
| 默认 learned 决策主导度 | 仍低（新 learners 均 SHADOW/report-only） | 约 10–20% | 四 torch backend 晋升 + SHADOW learners 转 authoritative |
| learned backend 实现 | 已就位且晋升管线代码完备 | 约 80–88% | ≥500 real-trace、validation delta、控制臂等证据 |
| 晋升证据 | 部分就绪 | 尚未全绿 | promotion report 的 component gates |
| thesis 因果证据 | 终局拒绝（局部支持） | #92 CLOSED | 重新提出整体 thesis 前必须是新的、owner-level 机制与独立预注册证据计划 |
| 开放世界 cognitive AGI | 未开始证明 | 不适用 | 跨域、跨模态、因果结构发现、mesa-objective detection |

## 2. 2026-07-17 两轮代码补齐

### P0 补齐包（六收敛包）

1. `SocialRecordStore` owner hydration（ToM / common-ground / group regime / durability 跨 session）；
2. `RegimeModule` owner hydration（含 selection / feature weights、external-outcome calibration）；
3. PE learned heads（critic + CP-11 predictive heads）与 `DualTrackGateLearner` hydration；
4. `RegimeScoreLearner` SHADOW 双跑 + delayed payoff settlement + checkpoint；
5. `LifeformSession.run_turn(..., environment_frame=...)` 多人产品帧 + `user_model.interlocutor_ids`；
6. companion thinking factory + evaluation mid（SHADOW）/ expensive / cross-generation（DISABLED）注册进 runtime DAG。

### P1 缺口补齐包（G1–G5）

1. **G1**：`CreditModule` session-held 化（learned heads 跨 turn 累积）+ `credit_heads` owner hydration（COCOA head + `GateRiskLearner`）；
2. **G2**：semantic LLM proposal 覆盖 9/9 slot（`plan_intent` / `open_loop` / `execution_result` / `belief_assumption` 加入 generic JSON-schema 路径，per-slot hint 集中管理）；
3. **G3**：`AffordanceScoreLearner` SHADOW 双跑（v1 分数=初始化+回滚点）+ invoker outcome-listener settlement + promotion readout；
4. **G4**：`ConsolidationScoreLearner` SHADOW 双跑（session-held，realized PE settlement，writeback gate 不读）；
5. **G5**：`LifeformSession.group_snapshot` 首个 group 产品 consumer + 三人 frame e2e；thinking advisory SHADOW 路由端到端验证（β_t 与基线字节一致）。

全部 SHADOW / report-only / opt-in；默认行为字节不变；每包单点回滚。

### Spec 同步（同日补齐）

七份能力域 spec 已按包补变更日志：`credit-and-self-modification.md`（G1）、`owner-hydration.md`（P0 五条新 hydrate 条目 + G1）、`semantic-state-owners.md`（G2）、`affordance.md`（G3）、`continuum-memory.md`（G4）、`thinking-loop.md`（G5b）、`social_cognition/05_joint_entity.md`（G5a）；另有 `learned-vs-heuristic-coverage.md` 第三包变更日志。

## 3. 默认 authoritative 路径事实

```text
substrate_mode = synthetic
temporal_latent_dim = 3
temporal_ssl_backend = DISABLED
temporal_runtime_backend = DISABLED
internal_rl_backend = DISABLED
cms_torch_backend = DISABLED
evaluation_mid = SHADOW
evaluation_expensive / evaluation_cross_generation = DISABLED
groups / apprenticeship_protocol_alignment / protocol_reflection / protocol_revision_queue / audit = SHADOW
```

live 决策仍由结构 + 启发式主导；SHADOW learners（regime / affordance / consolidation / gate-risk / dual-track gate / schedule gate）只发布 report-only readout。

## 4. Owner hydration matrix 当前状态

| owner | decision |
|---|---|
| semantic_state / followup_manager / vitals / protocol_registry | hydrate |
| social_record_store / regime / prediction_error_heads / dual_track_gate_learner / credit_heads | hydrate（本两轮新增） |
| memory | external-owner |
| world_temporal / self_temporal | explicit-no-hydrate（checkpoint / rare-heavy owner 管） |

## 5. 剩余代码缺口（真正还要写的）

### P1-2 World / Self predictive model 扩容（主要剩余）

- 更高容量 latent state、compositional prediction、counterfactual rollout；
- World / Self 分轨训练与 checkpoint；
- 不退回 token-space RL。

### 其余深化项

- tension / lesson 提取的 learned 候选（G4 只覆盖 consolidation score）；
- memory retrieval ranking learned 化；
- learned persona / function vectors 与 mesa-objective readout（P2）；
- 跨模态 latent action basis、开放环境因果结构发现（P2，研究前沿）。

## 6. 核心结论：瓶颈是信号真实性、学习器容量与外部真值

> 2026-08-01 校正：#92 已证伪“只差把现有 evidence pipeline 跑完”的
> 判断。下方旧 evidence lane 记录仅保留为历史机制回归说明，不再构成整体
> thesis 或默认 ACTIVE 晋升依据。

旧判断把 owner 内部 readout、合成轨迹与晋升门的完整性误当成了研究命题
本身；即使 `promotion_report.json` 全绿，也不能回答关系状态是否产生了对
未来的真实预测优势。当前三项根缺口是：

1. **信号真实性**：现有 PE 的 predicted / actual 主要是同批上游读数的两套
   owner 内投影，不是对未来观察的前向预测；
2. **学习器容量**：默认 `n_z=3`，且真实目标上的容量曲线尚不存在；
3. **外部真值**：既有主证据来自人写 FSM 与模拟用户，没有独立的真人多
   session held-out 未来作为标签。

研究路线改为：以真实多 session 对话的 N+1 表示为免费标签，先建立同一
冻结表示基底上的 stateless / 全量历史长上下文 / 摘要检索 / Volvence 四臂
对照，再判断 PE 与 ETA 是否 load-bearing。合成数据只保留为 sanity check。

R4 主实验形成独立 verdict 前，冻结新 wheel 与新 spec；允许删除无效主张、
更新既有 owner/spec，以及增加不进入 runtime DAG 的研究 harness。大批量研究
直接调用正式 owner 的 immutable batch contract，绕过逐 turn `propagate`
调度开销，但不得复制 prediction-error owner 或重建第二套 mismatch 语义。

截至 2026-08-01，地基与 mechanism harness 已落地：MSC v0.1 原文已下载到
gitignored 的 `data/external/msc/v0.1/`，其 archive 与 train/validation/heldout
hash、1001/500/501 dyad 划分已由 `DOWNLOAD_PROVENANCE.json` 复核；
`PredictionErrorModule` 新增 offline
N+1 表示 batch head；evaluation→PE gate 默认 ACTIVE 且 matched SHADOW
rollback 通过；Companion Bench 支持 stateless/session/full history 与 token/
latency/truncation 审计；Internal-RL 的 `math.sin` 伪噪声已替换为带 checkpoint
RNG state 的真实随机抽样。

一次真实 MSC CPU mechanism pilot 已重跑并落到
[`artifacts/msc_n_plus_one_mechanism_pilot_20260801/`](../artifacts/msc_n_plus_one_mechanism_pilot_20260801/)；
bundle 对语料 provenance、结果与关键源码逐文件哈希，并硬标记
`thesis_status=not-evaluated`、`formal_experiment_executed=false`。精确指标只留在
artifact，不再写入预注册正文或用作当前状态结论。

**正式 R4 仍未执行。** 现有 pilot 的 `volvence` 是 bounded-state prototype，
`long_context` 只有 256 tokens，且 capacity ladder 测的是 forward head 而不是
ETA temporal controller。

target blocker 已在独立收敛包关闭：`vz-substrate` 现在发布冻结
`SubstrateForwardRepresentationSnapshot`，以 model weights SHA、runtime origin、
layer/width/readout 和 sample/value hashes 冻结 lineage；PE batch/head/checkpoint
强制绑定该 lineage。真实 Qwen CPU smoke 已落到
[`artifacts/msc_n_plus_one_substrate_target_smoke_20260801/`](../artifacts/msc_n_plus_one_substrate_target_smoke_20260801/)，
并保持 `thesis_status=not-evaluated`。剩余 blocker 是同 substrate 的零截断 full-history
长上下文 steelman、完整 Volvence runtime 臂和只改变 temporal controller 的容量
实验；三者完成前不能晋升 `temporal_n_z`、退役 legacy controller 或选择 thesis v3。

两条证据线现有彼此独立的 MPS CLI：七日产品实验用
`scripts/run_seven_day_companion_test_plan.py`，并已按 preregistration schema 覆盖
Gate 1/4/5/6/7/8/9/10/11；Gate 1、Gate 4/5/6/7/9/10 和 Gate 8/11 仍分别调用自己的
正式 runner/auditor，不共享干预或 verdict。未知 schema/gate 和非 MPS preregistration 会
fail loudly；`status` 只有在 exact matrix、evaluation 与 prereg/evaluation-SHA-bound
independent audit 同时有效时才允许分析。formal 退出码 2 作为完整否定性科学结果仍会进入审计。
N+1 研究用
`scripts/run_msc_prediction_test_plan.py`。经两个新入口启动的任务共享 MPS 互斥锁并禁止
CPU fallback；控制面落地前手工启动的旧进程不受该锁保护，必须另行确认结束。
研究 CLI 当前只允许 preflight/mechanism smoke，`formal` 对上述三个 blocker 固定返回
退出码 3。长上下文不再机械要求一个“128k”标签：若冻结 tokenizer 的全体 MSC 历史都
低于 32k，32k 已是零截断 full-history exposure；只有未来样本真正超过 32k 时才另开
真实 128k model/config/hardware prereg。

N+1 mechanism runner 现按语料索引、数值 context/target、arm/split、capacity/seed
和 heldout/seed 自动落盘不可变 checkpoint，并用精确语料/模型/源码/参数
fingerprint 续跑；不保留 MSC 原文。最终 manifest 封口前禁止效应分析，
封口后仍只允许 mechanism-pilot 分析、不允许 formal claim。`status` 可随时并行，
只读 `audit` 仅可针对已封口 bundle 并可与另一条 MPS 执行并行，但任何两个 MPS
阶段不可并行。

2026-08-02，七日替代 36-run 矩阵在 16/36 处按 spec 既有回滚条款停跑，记录为
`artifacts/seven_day_companion_formal_frozen_20260801T122037Z/halt_record.json`，
`halt_class=instrument-discrimination`。停跑理由是七项 readout 中两项被构造性钉死、
`callback_hit_rate` 测的是 owner 自洽性而非跨日回忆、唯一观测 SUT 行为的
`fsm_probe_pass_rate` 全为 null；这不是 effect verdict，也不是“没有提升”。该目录
不允许原样续跑，仪器须先换成冻结 N+1 substrate 表示预测目标并另开预注册。同日
`scripts/run_eta_rate_distortion.py` 补接共享 MPS 互斥锁与 `require_mps()`——此前它
绕过控制面，曾与七日矩阵并行占用 MPS。

### 已降级的旧 evidence lane（机制回归用途）

`run_learned_active_evidence.sh`、`run_companion_bench_p1.sh`、SHADOW learner
settle lane 与跨 session hydration 测试仍用于防止机制退化和验证 rollback；它们
的产物一律标记 `thesis_status=not-evaluated`。World / Self 扩容只能在冻结
substrate/PE 后、唯一变量为 `temporal_n_z ∈ {3,16,64,256}` 的 validation 曲线
显示容量增益后进行；`forward_head_n_z` 曲线永远不授权 ETA 晋升。

## 7. 最简状态陈述

> 第一阶段 wiring 仍约完成 91–95%，但这只说明 owner、SHADOW 与回滚骨架齐备。#92 后的研究瓶颈是取得真实未来标签、真实前向 PE 与容量/长上下文强基线；任何整体 thesis 都必须由新的 held-out 多 seed 主实验重新挣得。
