# 探索方向 → 实验 → 主分支选型推进计划

> **目的**：把 [`探索方向.md`](./探索方向.md) 中 43+ 条研究借鉴方向，通过现状核查 + SHADOW 实验 + 数据驱动决策的方式，收敛为可进入主分支的候选。
>
> **修订历史**：
> - v1（2026-05-12）— 初版分层规划：阶段 A 现状核查 / 阶段 B 基础设施 / 阶段 C 并行 SHADOW / 阶段 D 决策。
> - **v2（2026-05-12）— 基于阶段 A 现状核查 brief 修订**：阶段 C 可并行 SHADOW 候选从 6 条收敛到 **4 条**（CMA-2 收编入 EVO-2 cascade；OA-4 改为阶段 B 双门延续 packet）；阶段 B 新增 **packet 0（DATA_CONTRACT §6 同步）** 作为整套规划启动的 prerequisite。详情见 [`experiment-phase-a-brief.md`](./experiment-phase-a-brief.md)。
>
> **阶段状态**：阶段 A ✅ 完成 / 阶段 B ⏳ 待启动 / 阶段 C ⏸ 阻塞中 / 阶段 D ⏸ 阻塞中。

---

我先把"基础设施 + 候选方向"的现状对清楚，再给你一个分层的建议。

## 1. 你这个仓库其实已经为"并行实验"准备好了基础设施

读完 `探索方向.md` 与现有实验骨架（`WiringLevel`、SHADOW evidence 文档、`run_dialogue_pe_eta_ablation_benchmark` profile 对比、`run_atlas_titans_cms_shadow_smoke.py` 这类双 profile 对照脚本），我得到一个关键判断：

**"分支隔离实验"在这个项目里不是首选路径，"profile + SHADOW + 多 seed 对照基准"才是。**

证据：

- 你已经有 `WiringLevel = {DISABLED, SHADOW, ACTIVE}` 三态（`packages/lifeform-thinking/src/lifeform_thinking/scheduler.py` 等）—— 这是单仓内并行跑"旧逻辑 + 新逻辑、同步比对快照"的契约级机制。
- `docs/specs/cms-atlas-titans-uplift-shadow-evidence-2026-05-06.md` 已经示范了完整的 "SHADOW 实验流水"：5 seeds × N cases × 2 profiles → 88 个 metric 的 delta 表 → 由数据决定是否切 ACTIVE → ACTIVE 后保留回滚窗口。
- `.cursor/rules/cursor-convergence-workflow.mdc` 明确："不要一次改整条主链路，不要同时发明新 shape、接所有线"——这恰好否决了"每个方向一条长寿 git branch、最后大合并"的玩法。

所以**主结构应该是：在 main 上做小步 PR，把每个候选方向先落成 SHADOW profile / candidate owner，靠 ablation benchmark 让数据决定哪个进 ACTIVE**。git branch 仅用于：

- 跨模块改契约 shape（高 ripple，必须独立成包，参考 [`cursor-convergence-workflow.mdc`](.cursor/rules/cursor-convergence-workflow.mdc) §高 ripple 共享契约）
- 需要一次性废弃旧组件的硬切换
- 需要长时间持有不能上 main 的训练 artifact

## 2. P0 的 12 条不是同构的——必须先分层再决定并行度

`探索方向.md` 第 1696-1710 行已经给出"资源紧张时最高 ROI 的 12 项"。我把它们按"是否独立可并行实验"重新切一刀：

| 类型 | 条目 | 能否并行 SHADOW 实验？ |
|---|---|---|
| **基础设施（必须先做、单线推进）** | EVO-2（evaluation cascade）、SYS-2（VC 容量双门）、OA-1（spec 引用 N3/N4/N6） | **不可并行**。它们改的是"评估管线 + ModificationGate"，是后面所有 SHADOW 实验的"裁判席"。裁判没就位前，并行实验也没法决胜负。 |
| **契约形式化（单包但与 1 解耦）** | OA-2（Mind/Face 隔离）、OA-3（FramingAwarenessCheck）、DM-4（gate eval 前置）、**OA-4（VZ-Audit Agent，作为 SYS-2 + DM-4 双门的延续 packet）** | 都改 `credit-and-self-modification.md` / `expression-layer.md`，会触碰同一批文件 → **不并行**，按顺序合并。**OA-4 修订位置**：阶段 A brief 核查发现 OA-4 的对照面是 rare-heavy artifact lifecycle 而非 dialogue ablation，因此它从原阶段 C 移到这里。 |
| **可并行 SHADOW 候选（"做不做、谁更好"由数据决定）** | SYS-1（CPD 切换 β_t）、COG-1（反事实信用 reframed）、COG-2（ToM owner reframed）、COG-3（persona/regime geometry） | **完全可以并行**（4 条）。各自落到不同 owner（temporal-abstraction / credit / semantic-state / evaluation readout）。每条挂一个 SHADOW profile，跑同一套场景，比 ablation。 |
| **已实质完成 / 收编**（不需独立 SHADOW profile） | CMA-2（VZ-MemProbe 4 探针已 PASS in `tests/longitudinal/`） | 剩余工作是把 `mp.*` 接入 **EVO-2 cascade** 的 cheap 层跨 generation aggregate，归 EVO-2 packet 的 sub-task。 |

**关键 insight**：你真正想在"experiment 决定主分支方向"上做并行的，是第 3 类。第 1、2 类是给第 3 类铺路的——并行不来。

**阶段 A 修订要点**（详见 [`experiment-phase-a-brief.md`](./experiment-phase-a-brief.md)）：

- **6 条原始候选中 5 条的"现状盲点"段落滞后于代码**：仅 **COG-3**（latent persona-vector readout）盲点完全成立；SYS-1 / COG-1 / COG-2 / CMA-2 / OA-4 都有不同程度的"已实现但 brief 未反映"——COCOA Phase 1.A+2.A 已上线、4 个 ToM slot 默认 ACTIVE、4 个 memprobe 已 PASS、Two-Gate 已强制。这意味着 COG-1 / COG-2 的"剩余工作"比 brief 原本的工作量小一档。
- **CMA-2 与 OA-4 的范畴错位**：两者都不属于 dialogue ablation 的对照面——CMA-2 是 longitudinal pytest 跨 generation aggregate，OA-4 是 rare-heavy artifact promotion gate。原 v1 把它们塞进阶段 C 的"6 个 profile 矩阵"是范畴错配。
- **DATA_CONTRACT §6 滞后于 `final_wiring.py` 实状态**：4 个 ToM slot 在代码中已默认 ACTIVE 但 spec 仍标 "planned migration mirror"。这是阶段 B 启动前必须先解决的 prerequisite。

## 3. multi-subagent（并行子代理）什么时候真正有收益

把"并行子代理"和"并行 SHADOW 实验"分开看，这两件事各有适用场景：

**适合用并行子代理（同一会话里多 Task 并发）的工作**：

1. **第 3 类候选的"现状核查"阶段**（探索方向.md §convergence packet 第 1 步）—— readonly，互不写文件，每条派一个 `explore` subagent 去查"现状盲点段落假设是否成立"。**v2 状态：已完成**，产出 [`experiment-phase-a-brief.md`](./experiment-phase-a-brief.md)。
2. **第 3 类 4 条进入 ACTIVE 后的"代际对比 evidence 收集"** —— 每条已经是独立 profile 了，可以用 shell subagent 并行跑 ablation。
3. **跨方向的文献交叉检查** —— 比如 COG-1 与 SYS-2 都引到了 EWC，让一个 subagent 同时核对两边引用一致性。

**不适合用并行子代理的工作**：

1. **同一 owner / 同一 spec 文件的多条建议改动** —— OA-2 + OA-3 + DM-4 都改 `credit-and-self-modification.md` 和 `expression-layer.md`，并行 write subagent 会冲突。
2. **第 1 类基础设施实施** —— 需要全局思考、不能分包并行。
3. **R8 SSOT 边界判定** —— 多个子代理各自决策时容易"每个 owner 都看起来是别人的责任"，必须主代理统一裁断。

## 4. 我的具体建议（如果你让我把这个落地为一个推进计划）

**阶段 A（本周，并行可用）—— 现状核查矩阵** ✅ **已完成（2026-05-12）**

派 6 个 readonly `explore` subagent 同时跑，每个负责一条 P0 候选，产出固定格式的 brief：
- 现状盲点假设是否成立（grep + spec 阅读）
- 涉及哪些已有 owner / snapshot slot / WiringLevel
- 现有 ablation 框架能否直接挂这条的 SHADOW profile，还是需要新增 profile 维度
- 这条与其他 P0 条目的耦合点（避免后面打架）

这一阶段**纯并行没有冲突风险**。

**产出**：[`experiment-phase-a-brief.md`](./experiment-phase-a-brief.md)（485 行 brief + 6×6 跨候选风险矩阵 + 起跑顺序总结表）。

**阶段 B（接下来 3-5 周，单线串行）—— 裁判席先到位 + 双门治理 + 契约同步**

按顺序、单线推进 5 个收敛包（v2 新增 packet 0 + packet 4）：

0. **DATA_CONTRACT §6 同步**（v2 新增 packet）— 把 4 个 ToM slot、`conversational_role`、`multi_party_identity`、`social_prediction[_error]` 等已在 `final_wiring.py` 默认 ACTIVE 的 slot 在 `docs/DATA_CONTRACT.md` §6 中改为 ACTIVE 状态，消除 R8 spec 与代码偏离。**必须最先做**，否则后续 SHADOW profile 引用 slot 时会陷入"以 spec 为真还是以 wiring 为真"的混乱。工作量：S（纯 spec 同步）。
1. **EVO-2 evaluation cascade**（cheap→expensive + LLM-judge readout 边界）— v2 扩展范围：把 CMA-2 的 `mp.*` probe pass-rate 收编为 cascade cheap 层 metric 之一（不再当作独立候选）。
2. **SYS-2 + DM-4 ModificationGate 双门**（validation margin + capacity cap 形式化；当前 `evaluate_gate_reasons` 已强制 capacity_cost / validation_delta / rollback_evidence，本 packet 是 spec 形式化 + 测试补全）。
3. **OA-1 + OA-2 spec motivation + Mind/Face 形式化**（把 N3/N4/N6 引证写进相关 spec；formalize `expression-layer.md` 的 Mind/Face 边界 + 契约测试）。
4. **OA-4 VZ-Audit Agent**（v2 新增位置）— 作为 SYS-2 + DM-4 双门的延续 packet（"双门 → 三门"）：新建 `audit` owner + `AuditSnapshot` slot；提供 N8 风格 elicited probe 工具集（dataset inspector / benchmark runner / persona drift probe / memprobe runner）；产出 risk score + transcript + tool trace；由 `run_multi_artifact_acceptance_benchmark` 作对照面，不走 dialogue ablation。
5. **OA-3 FramingAwarenessCheck**（N4 inoculation 工程化，75-90% 缓解率目标）。

这是 SSOT 改造 + gate 治理 + audit 落地，并行只会乱。

**阶段 B prerequisite check（启动前必须确认）**：

- ✅ DATA_CONTRACT §6 已同步到 `final_wiring.py` 实状态（packet 0 内含）。
- ⚠ 默认 `SubstrateAdapter` 后端是否真的填 `feature_surface` / `residual_activations`——这是 COG-3 起跑（阶段 C）的硬依赖，若当前是 stub-only，需要补一个最小 substrate-feature 暴露 packet（小工作量，可与 packet 0 并行）。

**阶段 C（约 1-2 个月，profile 并行）—— 真正的 experiment 阶段**

为每条第 3 类 P0 候选起一个独立 SHADOW profile，挂到现有 `run_dialogue_pe_eta_ablation_benchmark` 风格的多 profile 对照基准上：

```
profile = pe-eta（baseline）
       + cpd-beta-switch（SYS-1）
       + counterfactual-credit（COG-1 reframed: least_control 字段 + metric_means 抽 COCOA + commitment lineage）
       + tom-owner（COG-2 reframed: UserModelSnapshot 拆分 + paper-suite 多人场景）
       + persona-geometry-readout（COG-3，read-only）
```

**v2 收缩说明**：

- **移除 `mem-probe` profile**：CMA-2 不属于 dialogue ablation 范畴，已收编入 packet 1 (EVO-2 cascade) 的 cheap 层。
- **移除 `audit-agent` profile**：OA-4 对照面是 artifact-acceptance benchmark，已上移至阶段 B packet 4。

每个 profile 单独 PR，互不阻塞。跑 5 seeds × 全部 paper-suite-small 场景 → 一张大 delta 表 → 由"是否破坏 acceptance gate / 是否在 evaluation cascade 的 cheap 层胜出"两条硬证据决定哪些进 ACTIVE。

**阶段 C 起跑顺序**（详见 [`experiment-phase-a-brief.md`](./experiment-phase-a-brief.md) §附录）：

| 顺位 | 候选 | 起跑前置 | 工作量级别 |
|---|---|---|---|
| 1 | **SYS-1** CPD 涌现 β_t 切换 | 无（单 owner、profile 直挂） | S |
| 2 | **COG-3** Persona/Regime geometry 漂移 readout | 验证 substrate `feature_surface` / `residual_activations` 后端实填 | S-M |
| 3 | **COG-1 reframed** 反事实信用 + least-control | 阶段 B EVO-2 cascade 完成（让 `metric_means` 抽 COCOA readout 路径打通） | M |
| 4 | **COG-2 reframed** ToM owner 内核拆分 | 阶段 B packet 0（DATA_CONTRACT §6 同步） + paper-suite 多人 fixture | M |

**阶段 D（决策点）—— 不是合并 branch，而是"profile → ACTIVE 切换"**

此时再决定：
- 哪些 SHADOW profile 直接切 ACTIVE（保留旧逻辑为 DISABLED 一个 release cycle）
- 哪些数据不达标，留 SHADOW 继续观察或 DISABLED 雪藏
- 哪些方向之间的组合最强（v2 已识别强互补对：**SYS-1 ⊗ COG-1**（PE-first 配对：边界识别 + 边界归因）、**COG-2 ⊗ COG-3**（跨 interlocutor persona 泄漏检测）、**OA-4 ⊗ COG-3**（drift 作为 audit elicited probe）），用组合 profile 跑第二轮。

## 5. 关于"要不要分 branch"的直接回答

- **不需要为每条 P0 方向各开一个长寿 branch**——这违反 §3 第 2 项收敛工作流，而且会让 evaluation cascade（裁判席）出现"每个 branch 各自版本"的灾难。
- **每个 SHADOW profile 各开一个短寿 feature branch 然后合 main**，是标准做法，和"profile 隔离"叠加形成双层保护：代码层 PR review，运行时层 WiringLevel 隔离。
- **唯一需要长寿 branch 的情况**：SYS-5（Latent Action RL）—— 它要训练新的 <1M 参数控制器，可能需要 artifact 落盘和反复 retrain，这种"训练实验"用独立 branch + artifact 仓更合适，跟其它方向的"spec + SHADOW evidence"性质不一样。

---

## 6. 阶段状态与下一步候选动作（v2）

### 阶段状态

| 阶段 | 状态 | 关键产出 / 阻塞 |
|---|---|---|
| 阶段 A — 现状核查矩阵 | ✅ 完成（2026-05-12） | [`experiment-phase-a-brief.md`](./experiment-phase-a-brief.md) |
| 阶段 B — 裁判席 + 双门治理 + 契约同步 | ⏳ 待启动 | 5 个串行 packet（0→1→2→3→4→5）；packet 0 是整套规划的硬 prerequisite |
| 阶段 C — 4 条 SHADOW profile 并行 | ⏸ 阻塞中 | 等阶段 B packet 0 完成（COG-3 还需 substrate hook 验证） |
| 阶段 D — profile → ACTIVE 决策 | ⏸ 阻塞中 | 等阶段 C 至少 1 个 profile 跑出 5 seeds × paper-suite-small 的对照证据 |

### 下一步候选动作

1. **写一份"阶段 B 的裁判席单线推进 packet 提纲"**（packet 0 DATA_CONTRACT 同步 / packet 1 EVO-2 cascade / packet 2 SYS-2+DM-4 双门 / packet 3 OA-1+OA-2 / packet 4 OA-4 audit-agent / packet 5 OA-3 framing check 的具体 convergence packet 拆分）；
2. **基于已有 `run_atlas_titans_cms_shadow_smoke.py` 设计"阶段 C 的 4-profile 对照基准骨架"**（不实现，只给 ablation profile 矩阵 + acceptance gate 列表 + metric_means 抽取扩展点清单）；
3. **直接起跑阶段 B packet 0**（DATA_CONTRACT §6 同步，是阶段 A brief 识别的 prerequisite，工作量 S，纯 spec 同步）；
4. **直接起跑阶段 C 顺位 1 candidate SYS-1**（不需阶段 B 任何 packet 完成；前提是接受"先有候选 profile、再补 cascade 完整裁判席"的工作流，evidence 可后置）。

选项 1/2 是规划层产出（继续在文档层推进）；选项 3/4 是直接动 spec/代码进入实施。视具体约束（是否允许动代码、阶段 B 排队是否优先级最高）选择。