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

## 6. 核心结论：瓶颈已从代码转移到证据

两轮补齐后的关键判断：**写代码已经不是瓶颈**。每个 learned 部件都有实现、SHADOW 双跑、promotion readout 与回滚路径，但没有一个拿到过晋升证据——当前所有 SHADOW learner 的 settle 计数为零或接近零，四个 torch backend 的 component gate 未全绿。"还差多少"的答案已经从"差代码"变成"差证据"。

下一步按杠杆大小排序：

### 第一优先：跑证据线（不需要写代码）

```bash
bash run_learned_active_evidence.sh --resume --substrate-mode hf --substrate-device mps
bash run_companion_bench_p1.sh --resume
```

目标是 `promotion_report.json` 的 `all_eligible=true`，然后按固定顺序逐组件 ACTIVE：

```text
读取现有 real-trace missing_gates
→ capacity ladder
→ P1 9-track
→ promotion report（all_eligible=true）
→ temporal runtime ACTIVE canary
→ SSL → Internal RL → CMS torch
→ P2 held-out multi-seed（first-stage-retained）
```

这是把 learned 主导度从 10–20% 提上去的唯一合法路径。

### 第二优先：让 SHADOW learners 积累 settle（采集 lane 已补，2026-07-17）

RegimeScoreLearner / AffordanceScoreLearner / ConsolidationScoreLearner 各需 ≥50 次 settle + MAE 领先 margin 才达 promotion readout 的 ready。采集面现已闭合：

- soak artifact（`run_learned_shadow_soak.py`）新增 `regime_score_learner` / `reflection_consolidation_learner` / `credit_learned_heads` 三段 readout——regime 与 consolidation learner 在 kernel 主链内每 turn 自然 settle，soak 即积累（8-turn 冒烟已见 settle 5 / 7 次）；
- affordance learner 的 settlement 只来自真实工具调用，kernel-only soak 覆盖不到，已补独立 lifeform 级 lane：`run_affordance_learner_probe.sh` / `.ps1`（真实 registry → module → invoker → outcome listener 全链，机制证据 EXIT(0) 已过；promotion 仍需 ≥50 次真实使用 settle）。

### 第三优先：跨 session continuity 证据（longitudinal lane 已补，2026-07-17）

- 新增 `tests/longitudinal/test_cross_session_learned_state_continuity.py`：同一用户 20 sessions × 2 turns，断言 social record（种子 ToM record 跨全部边界存活）、regime `turn_index` 累计到 40、dual-track gate learner ≥20 次 settle、PE critic / COCOA head 计数跨边界不回退、六个 hydratable owner 每 session 全部持久化；另有跨用户隔离用例（bob 全部计数从零开始）。本机已跑通（2 passed）。
- 双平台入口：`run_longitudinal_continuity.sh` / `.ps1`（含既有 owner-hydration longitudinal 套件）。
- 剩余：在真实部署 scoped backend 上重复该 lane 作为发布证据。

### 第四优先（唯一的代码大项）：World / Self predictive model 扩容

建议等 capacity ladder 结果出来再定容量方向——如果 `n_z=16→64` 无增益，盲目扩 World/Self model 容量是浪费。

## 7. 最简状态陈述

> 第一阶段认知系统代码经 P0 + P1 两轮补齐后约完成 91–95%：owner continuity、learned regime/affordance/consolidation SHADOW 候选、9/9 semantic LLM proposal、session-held credit owner、group 产品 consumer 与 thinking advisory SHADOW 链均已在代码中。默认 learned 主导度仍约 10–20%；四个 torch backend 与全部 SHADOW learners 的 ACTIVE 均 gate 于 ≥500 real-trace、validation delta、控制臂、回滚、性能、安全与 P2 held-out multi-seed 证据。当前状态是 wiring-ready 且 promotion-path-complete，不是 first-stage-retained。**瓶颈已从"写代码"转移到"跑证据"：最优的下一步不是继续写实现，而是把 promotion pipeline 真正跑完一遍。**
