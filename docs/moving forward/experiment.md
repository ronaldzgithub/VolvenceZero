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
| **契约形式化（单包但与 1 解耦）** | OA-2（Mind/Face 隔离）、OA-3（FramingAwarenessCheck）、DM-4（gate eval 前置） | 三条都改 `credit-and-self-modification.md` / `expression-layer.md`，会触碰同一批文件 → **不并行**，按顺序合并。 |
| **可并行 SHADOW 候选（"做不做、谁更好"由数据决定）** | SYS-1（CPD 切换 β_t）、COG-1（反事实信用）、COG-2（ToM owner）、COG-3（persona/regime geometry）、CMA-2（VZ-MemProbe）、OA-4（VZ-Audit Agent） | **完全可以并行**，各自落到不同 owner（temporal-abstraction / credit / semantic-state / continuum-memory / 新 audit owner）。每条挂一个 SHADOW profile，跑同一套场景，比 ablation。 |

**关键 insight**：你真正想在"experiment 决定主分支方向"上做并行的，是第 3 类。第 1、2 类是给第 3 类铺路的——并行不来。

## 3. multi-subagent（并行子代理）什么时候真正有收益

把"并行子代理"和"并行 SHADOW 实验"分开看，这两件事各有适用场景：

**适合用并行子代理（同一会话里多 Task 并发）的工作**：

1. **第 3 类 6 条的"现状核查"阶段**（探索方向.md §convergence packet 第 1 步）—— readonly，互不写文件，每条派一个 `explore` subagent 去查"现状盲点段落假设是否成立"，30 分钟拿到 6 份对齐报告。**强烈推荐**。
2. **第 3 类 6 条进入 ACTIVE 后的"代际对比 evidence 收集"** —— 每条已经是独立 profile 了，可以用 shell subagent 并行跑 ablation。
3. **跨方向的文献交叉检查** —— 比如 COG-1 与 SYS-2 都引到了 EWC，让一个 subagent 同时核对两边引用一致性。

**不适合用并行子代理的工作**：

1. **同一 owner / 同一 spec 文件的多条建议改动** —— OA-2 + OA-3 + DM-4 都改 `credit-and-self-modification.md` 和 `expression-layer.md`，并行 write subagent 会冲突。
2. **第 1 类基础设施实施** —— 需要全局思考、不能分包并行。
3. **R8 SSOT 边界判定** —— 多个子代理各自决策时容易"每个 owner 都看起来是别人的责任"，必须主代理统一裁断。

## 4. 我的具体建议（如果你让我把这个落地为一个推进计划）

**阶段 A（本周，并行可用）—— 现状核查矩阵**

派 6-8 个 readonly `explore` subagent 同时跑，每个负责一条 P0 候选，产出固定格式的 brief：
- 现状盲点假设是否成立（grep + spec 阅读）
- 涉及哪些已有 owner / snapshot slot / WiringLevel
- 现有 ablation 框架能否直接挂这条的 SHADOW profile，还是需要新增 profile 维度
- 这条与其他 P0 条目的耦合点（避免后面打架）

这一阶段**纯并行没有冲突风险**。

**阶段 B（接下来 2-3 周，单线串行）—— 裁判席先到位**

按顺序、单线推进 3 个收敛包：
1. EVO-2 evaluation cascade（cheap→expensive + LLM-judge readout 边界）
2. SYS-2 + DM-4 ModificationGate 双门（validation margin + capacity cap）
3. OA-1 + OA-2 spec motivation + Mind/Face 形式化

这是 SSOT 改造，并行只会乱。

**阶段 C（约 1-2 个月，profile 并行）—— 真正的 experiment 阶段**

为每条第 3 类 P0 候选起一个独立 SHADOW profile，挂到现有 `run_dialogue_pe_eta_ablation_benchmark` 风格的多 profile 对照基准上：

```
profile = pe-eta（baseline）
       + cpd-beta-switch（SYS-1）
       + counterfactual-credit（COG-1）
       + tom-owner（COG-2）
       + persona-geometry-readout（COG-3，read-only）
       + mem-probe（CMA-2，read-only）
       + audit-agent（OA-4）
```

每个 profile 单独 PR，互不阻塞。跑 5 seeds × 全部 paper-suite-small 场景 → 一张大 delta 表 → 由"是否破坏 acceptance gate / 是否在 evaluation cascade 的 cheap 层胜出"两条硬证据决定哪些进 ACTIVE。

**阶段 D（决策点）—— 不是合并 branch，而是"profile → ACTIVE 切换"**

此时再决定：
- 哪些 SHADOW profile 直接切 ACTIVE（保留旧逻辑为 DISABLED 一个 release cycle）
- 哪些数据不达标，留 SHADOW 继续观察或 DISABLED 雪藏
- 哪些方向之间的组合最强（比如 SYS-1 + COG-1 是否互补），用组合 profile 跑第二轮

## 5. 关于"要不要分 branch"的直接回答

- **不需要为每条 P0 方向各开一个长寿 branch**——这违反 §3 第 2 项收敛工作流，而且会让 evaluation cascade（裁判席）出现"每个 branch 各自版本"的灾难。
- **每个 SHADOW profile 各开一个短寿 feature branch 然后合 main**，是标准做法，和"profile 隔离"叠加形成双层保护：代码层 PR review，运行时层 WiringLevel 隔离。
- **唯一需要长寿 branch 的情况**：SYS-5（Latent Action RL）—— 它要训练新的 <1M 参数控制器，可能需要 artifact 落盘和反复 retrain，这种"训练实验"用独立 branch + artifact 仓更合适，跟其它方向的"spec + SHADOW evidence"性质不一样。

---

需要我接下来做哪件事？我可以马上：

1. **派阶段 A 的并行 explore subagent 矩阵**（6-8 条 P0 候选的现状核查），产出对齐 brief 给你做最终选型决定；
2. **写一份"阶段 B 的裁判席单线推进 packet 提纲"**（EVO-2 → SYS-2/DM-4 → OA-1/OA-2 的具体 convergence packet 拆分）；
3. **基于已有 `run_atlas_titans_cms_shadow_smoke.py` 设计"阶段 C 的多 profile 对照基准骨架"**（不实现，只给 ablation profile 矩阵 + acceptance gate 列表）。

Ask 模式下我不会直接动代码，但任何一种产出我都能先以文档形式给你过目。你想从哪个切口起？