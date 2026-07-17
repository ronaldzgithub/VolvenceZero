# VolvenceZero Cognitive AGI 当前状态

> Status: live status summary
> Last updated: 2026-07-17（P0 补齐包 + P1 缺口补齐包 G1–G5 之后）
> 详细判断、晋升协议与命令见 [`current.md`](./current.md)。
> 本文件只记录当前事实、剩余代码、晋升状态和下一步，不把计划写成已完成。

## 1. 当前总状态

| 维度 | 当前状态 | 完成度 | 还差什么 |
|---|---|---:|---|
| 架构 / 契约 / owner / 回滚骨架 | P0 continuity + P1 SHADOW learners 均已接线 | 约 90–94% | groups / protocol slow loop ACTIVE 仍待证据门 |
| 第一阶段认知系统代码 | P0 关键代码 + P1 近期缺口（G1–G5）已补 | 约 91–95% | 主要剩 World/Self model 扩容与 evidence-gated ACTIVE |
| 默认 learned 决策主导度 | 仍低（新 learners 均 SHADOW/report-only） | 约 10–20% | 四 torch backend 晋升 + SHADOW learners 转 authoritative |
| learned backend 实现 | 已就位且晋升管线代码完备 | 约 80–88% | ≥500 real-trace、validation delta、控制臂等证据 |
| 晋升证据 | 部分就绪 | 尚未全绿 | promotion report 的 component gates |
| thesis 因果证据 | harness-ready | 约 5% | P1 directional + P2 held-out multi-seed |
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

## 6. 证据线（不变）

```text
读取现有 real-trace missing_gates
→ capacity ladder
→ P1 9-track
→ promotion report（all_eligible=true）
→ temporal runtime ACTIVE canary
→ SSL → Internal RL → CMS torch
→ P2 held-out multi-seed（first-stage-retained）
```

SHADOW learners 的 ACTIVE flip 同样 gate 于各自 promotion readout（≥50 settle + MAE margin）与真 trace 证据。

## 7. 最简状态陈述

> 第一阶段认知系统代码经 P0 + P1 两轮补齐后约完成 91–95%：owner continuity、learned regime/affordance/consolidation SHADOW 候选、9/9 semantic LLM proposal、session-held credit owner、group 产品 consumer 与 thinking advisory SHADOW 链均已在代码中。默认 learned 主导度仍约 10–20%；四个 torch backend 与全部 SHADOW learners 的 ACTIVE 均 gate 于 ≥500 real-trace、validation delta、控制臂、回滚、性能、安全与 P2 held-out multi-seed 证据。当前状态是 wiring-ready 且 promotion-path-complete，不是 first-stage-retained。
