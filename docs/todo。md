我把 EmoGPT v4.0 PRD 通读了一遍，再把 VolvenceZero 的现状（`docs/prd.md` 路标 + `packages/*` 实际落地）对照过一次。先放一句话定调，再给你按"能力域对齐 / 缺口 / 第一性原则下的实现方向"展开的 Gap 表。

## 一句话定位

EmoGPT v4.0 的 PRD 主要是一份**产品/服务面 PRD**（DLaaS 控制面 + 学徒训练 + 仪表盘 + Affordance 体系 + AAC 决策框架 + 多路异步思考时钟）。它的"内核学习论"已经被 VolvenceZero 用 NL+ETA / R-PE / R1–R15 重新地基化。所以你不应该照搬 EmoGPT 的模块树，而是**把它的产品需求当 prompt，把 VolvenceZero 的 owner 契约当 ground truth**——任何"借鉴"都要先回答：这是不是某个现有 owner 应该多发布一段快照、加一条 propose API，而不是新开一个模块。

---

## 已经齐了的东西（不要重做）

EmoGPT 的若干"听起来很丰富"的能力，在 VolvenceZero 里其实已经以**更收敛的形式**存在：

| EmoGPT 名字 | VolvenceZero 对应 owner | 备注 |
|---|---|---|
| HomeostasisEvaluator + 持续 tension PE | `lifeform-core/vitals.py` `VitalsModule` + `proactive_pe_threshold` | drive 偏离 band 即慢尺度 PE，已经是正式 R-PE 源 |
| TickEngine（SYSTEM / ENERGY / CONTEXT） | `lifeform-core/tick_engine.py` | 已经分了 SYSTEM/ENERGY/CONTEXT，三档 tick |
| SceneManager + 场景闭合 → SlowThinking | `lifeform-core/scene_manager.py` → `runner.begin_new_context()` → `vz-runtime/agent/session_post_slow_loop.py` | scene 闭合即触发 R6 session-post slow loop |
| FollowupManager / followup_due trigger | `lifeform-core/followup_manager.py` | 主动 followup 由 vitals PE + open-loop / commitment 驱动，不是关键词 |
| EtiquetteWatchdog 礼仪硬拦截 | `lifeform-expression/etiquette_watchdog.py` | 已存在，且严格做成 **UX-only verdict**（不阻 learning），比 EmoGPT 的更干净 |
| ETA 双轨（World/Self） | `vz-cognition/dual_track/core.py` + `world_temporal` / `self_temporal` 双 owner + `temporal_abstraction` 聚合面 | 双 owner 已经比 EmoGPT 的"同 GoalGraph 内 track 字段隔离"更彻底 |
| Recursive Semantic RL（step/tactic/pattern/principle） | `vz-cognition/credit` + reflection writeback 的 layer 概念 | 概念存在；EmoGPT 的四层证据门槛在 VZ 里是更通用的 ModificationGate ladder |
| MemoryOS L0/L2/L1 三层 | `vz-memory/cms.py` CMS nested MLP tower（online_fast / session_medium / background_slow） | VZ 已经走到"learned 多频率层 + R6 慢反思" |
| Person 卡 / CounterpartProfile | `semantic_state` 的 9 个 owner 之一 `user_model` | 已是一等 owner，发布快照而非内嵌在 MemoryOS 里 |
| 多 trigger（user_input / internal_drive / followup_due / tool_result / ...） | `BrainSession.submit_*` 系列（tool_result / profile / task / reviewed_knowledge / semantic events） | trigger 已经是 typed event，不是字符串组别 |
| MethodEngine + interaction_regime | `vz-cognition/regime/identity.py` `RegimeIdentity` + `RegimeSelectionWeights` + `RegimeBootstrap` | regime 是持久身份（R14），不是 prompt 标签 |
| RewardComposer / Credit | `vz-cognition/credit/gate.py` + reflection writeback | PE 在主链生效，credit 是 PE 的聚合 |
| 双 vertical（DLaaS template 概念的轻量版） | `lifeform-domain-emogpt` / `lifeform-domain-coding` + `lifeform-service.verticals` | drive 集合互不重叠，service registry 自动发现，比 EmoGPT 控制面 template 概念更克制但走通了 ② |
| Lazy / Speculative expression | 没有，但 `prompt_planner.py` 的 frozen `PromptPlan` 是同等粒度的内容选择面 | 见下面缺口 |
| 6 族 family report (R12) | `lifeform-evolution/family_report.py` `FamilyId.F1..F6` + `lifeform-bench --family-report` | 已经是 CLI fail-closed gate |

→ 这一栏的东西**不要照 EmoGPT 的命名再开一遍**，否则会把 SSOT 弄碎。

---

## 真正的 Gap（按能力域）

下面才是你要看的部分。每条都给 EmoGPT 的需求 + VZ 的现状 + **在 VZ 第一性原则下应该怎么落**（核心：每个新行为先回答"这条是不是已有 owner 应当发布的快照？是不是 propose API 的扩展？是不是评估 readout？"）。

### Gap 1 — Affordance 体系（Tool / Action / Organ / Shell）

EmoGPT 在 §9 把"AI 能做的事"统一成 `AffordanceDescriptor`（YAML，含 `when_to_use ≥ 50 字` / `when_not_to_use ≥ 50 字` / `parameters JSON Schema` / `safety_model`）+ 单注册表 + 4 种渲染器。

**VZ 现状**：`semantic_state/__init__.py` 有 `semantic_events_from_tool_result`，但工具是从 outside 注入的 *event*，**没有"AI 能调用什么"的注册表**。这意味着 VZ 当前的 lifeform 不能主动调用工具，只能由 host 把 tool result 喂回来。

**第一性下该怎么落**：
- 不要新开 ToolManager。`Affordance` 就是 lifeform 层的契约，由 `lifeform-core` 或新的 `lifeform-affordance` 拥有；它**只发布快照**给 prompt planner / response synthesizer。
- 描述符 schema 落到 `vz-contracts`，因为它要被多 vertical 共享。
- 选择哪个 affordance 是 `temporal_abstraction` 的**抽象动作**之一，不是新的硬编码路由层 —— 这正符合 R3/R4：`z_t` 空间的离散 action 现在多了一类候选。
- 工具调用结果回流走已有的 `submit_tool_result` → `semantic_events_from_tool_result`，无新通道。

### Gap 2 — Apprentice 学徒学习管线

EmoGPT §10 是把**所有"学习"统一成"同一认知 pipeline + forced compliance"**：novel / materials / persona_research / real_person / imagine 五种源走同一管道，差别只在 `compliance_profile` / `interaction_source`。

**VZ 现状**：完全没有学徒入口。预训练只有 `lifeform-super-loop` 通过 scenarios 跑，scenarios 是 JSON 文件而非 runtime 摄入。

**第一性下该怎么落**：
- 不要写 5 个 ContentSource。把"学徒源"当成 **DomainExperiencePackage 的运行时变体**：源文件 → `IngestionEnvelope` → 走完整 turn 路径，但 vitals resistance 设为 0、turn label 为 `apprenticeship`。
- "forced compliance"在 VZ 里就是 `WiringLevel.SHADOW` 或 vitals 的 `recharge_per_regime` override，不是新增门控。
- 学徒证据的 durable 化必须走 R6 session-post slow loop，**不可** 直接写 memory / regime / temporal —— 这条 EmoGPT PRD 自己也在 §10.3 写明（"durable 学习走 runtime 权威证据 + 反思"），但表面上看像是新增了 5 个机制，VZ 实际只需要扩 `IngestionEnvelope` 一种 trigger。
- Apprentice mode flag 是 `LifeformConfig.apprentice_mode: bool`，只影响 vitals override 和 ResponseSynthesizer 的 visibility，不进 kernel。

### Gap 3 — Runtime 内容摄入（Book / Web / TaskResult）

EmoGPT §10.1–10.2 的 `BookContentSource` / `WebContentSource` / `TaskResultSource`。

**VZ 现状**：`TaskResult` 已经有 `submit_tool_result`，但没有"读完 PDF / 抓网页"的 chunked ingestion。

**第一性下该怎么落**：
- 这是 lifeform 层的 **adapter**，`lifeform-ingestion` 新 wheel（依赖 `lifeform-core`，不依赖 kernel）。
- 它把外部 corpus 切片 → 每片调用 `BrainSession.submit_*` 之一 → 走完整 turn 但 vitals 不消耗、scene 是 `ingestion-scene` 类型。
- **关键不变量**：摄入产生的 PE 也是真 PE，会被 background-slow consolidation 吸收；这不是"特殊学习管道"，只是"另一种 trigger"。

### Gap 4 — Active Exploration + Mid-Session Reflector + ProvisionalLesson

EmoGPT §6.3–6.5 的 ThinkingLoopScheduler 三种异步思考：
- **Active Exploration**：Panorama 在 `consultation_required` 之前主动收集证据
- **MidSessionReflector**：会话进行中的 self / world 双 lane 反思
- **ProvisionalLesson**：在线信号产生的 30min TTL 弱先验，不创 GoalGraph 节点

**VZ 现状**：只有 session-post slow loop（R6），没有**会话内异步**思考路径。

**第一性下该怎么落**：
- 这其实是 R1 多时间尺度里 **session-medium**/**online-fast 之间的中间频道**。VZ 当前主链只有 fast（per-turn）和 slow（post-scene）两档，缺中间。
- 新建 `ThinkingScheduler`（lifeform 层，不是 kernel）：管理后台 Task，task 完成后产出 `ThinkingArtifact` 这个**只读** snapshot，让 owner 自己决定是否 apply。
- ProvisionalLesson 不应是新 owner，应该是 `case_memory` 内一种 lifecycle state（`lifecycle_state ∈ {candidate, provisional, validated, retired}`）；scene-end 时 reconcile 是 reflection writeback 的标准入口。
- MidSessionReflector 的 self/world 双 lane = 直接复用现有 `world_temporal` / `self_temporal` 双 owner，**不是新对象**。

### Gap 5 — Speculative / Lazy Expression

EmoGPT §8.3–8.4：decision-bypass 模式下投机表达 + 工具驱动 detail（cache hit ≥ 70% / token drop ≥ 30%）。

**VZ 现状**：`prompt_planner.py` 是 frozen `PromptPlan`，没有 cache 命中验收，没有投机执行。

**第一性下该怎么落**：
- 这是纯**性能优化**，是 lifeform 层 ResponseSynthesizer 的一个可选 wrapper，不进 kernel。
- 投机时的 fingerprint 必须严格基于发布快照（perception / regime / panorama / memory_recall hash），**不可**用关键词比对 user_input。
- "硬结构化 turn 必须 fresh build"这一条要保留 —— 在 VZ 里就是 `regime ∈ {problem_solving, repair_and_deescalation}` 时禁止 speculative adopt。

### Gap 6 — DLaaS 控制面（Tenant / Shell / Template / Contract / Studio / Exam / License）

EmoGPT §11 是一整套**控制面**：多租户、license gate、teaching case 审阅、launch license、ops dashboard。

**VZ 现状**：只有 `lifeform-service`（aiohttp，单租户，session-only）。

**第一性下该怎么落**：
- **这是部署面，不是认知能力，不要污染 kernel**。建议：
  - `lifeform-service` 扩 tenant 隔离（每个 session 带 `tenant_id`，session_manager 按 tenant 配额），不动 brain。
  - License / Exam / TeachingCase 是 **service 层的 DB record**，learning 始终走 R6 slow loop —— EmoGPT 自己也写了"学习 ownership 是运行时认知路径"，VZ 落地比 EmoGPT 简单：teaching case 进来 = 一次 apprenticeship trigger（见 Gap 2），exam = 一次 scripted scenario benchmark（见 `lifeform-evolution.benchmark`），license = `--require-family-pass` 是否连续两次通过的运营元数据。
  - **完全不要做** `runtime_template_id` / `seed_payload` / `activation_status` 这一套 `InterviewCognitiveSphere.to_dict()` —— EmoGPT 的这套 sphere 概念在 VZ 里其实就是 `RegimeBootstrap + MetacontrollerParameterSnapshot + DomainExperiencePackage`，已经是标准 vertical bootstrap 三件套，复用即可。
- 优先级：Tenant 隔离 > Exam/License > TeachingCase > Audience profile > Studio。

### Gap 7 — AAC 决策生命周期（Advocacy → Alignment → Commitment → Followup）  ✅ **已落地 (2026-04-29)**

EmoGPT §5.6 的 `advocacy_state` / `alignment_state` / `followup_policy` / `followup_status` / `followup_outcome_ref`。

**VZ 现状（更新）**：`commitment` owner 增加了 `AdvocacyState` + `AlignmentState` 双轴 lifecycle，通过 `CommitmentLifecycleEntry` 公布在 `CommitmentSnapshot`。Transition 完全从 `SemanticProposalOperation` 推导，REJECT alignment 进 `FollowupManager` 触发 follow-up。详见 `docs/specs/aac-commitment-lifecycle.md`。

**还没做的部分（Gap 7 part 2）**：
- Reflection writeback 的 `outcome_kind` enum（`commitment_progressed / completed / stalled / rejected / followup_no_response`）—— 应该是 `PolicyConsolidation` 上的 outcome 字段，从 lifecycle 聚合推导，**不**新开 owner
- `relationship_state` owner cross-reference commitment lifecycle，让 REJECT 的 alignment_state 直接驱动 trust delta，不靠文本推断

### Gap 8 — Cognitive Depth + TurnParticipationSignal（参与门）

EmoGPT §5.1–5.2 的五档 `REFLEXIVE / SHALLOW / FOCUSED / ALERT / DEEP` + `TurnParticipationSignal(panorama_level, method_level, task_level)`。

**VZ 现状**：`prompt_planner.py` 的 `_REGIME_DEFAULT_SECTIONS` 直接按 regime 选 section，没有"depth / 参与门"独立轴。

**第一性下该怎么落**：
- `cognitive_depth` 是**算力预算轴**，应该是 metacontroller `z_t` 的一个维度（深度也是抽象动作的属性），不是新 owner。
- `TurnParticipationSignal` 在 VZ 里其实是 prompt_planner 的输入特征，可以让 `regime` snapshot 多发一段 `participation_hint: {pretend_silent, brief, structured}`，让 prompt planner 读它，而不是让 prompt planner 自己再算。
- **不要**用关键词检测"这是不是闲聊"——用现有 `regime ∈ {casual_social, acquaintance_building}` + drive 偏离量做 soft gate。

### Gap 9 — InterlocutorState + Resistance + Coaxing（差异化体验）

EmoGPT §13 的"压力驱动有脾气的 AI"——12 维 InterlocutorState + Resistance 区间映射 + Coaxing Loop + RelationshipStage 渐进解锁。

**VZ 现状**：`relationship_state` 已是 9 个语义 owner 之一，但里面的 `trust_level` / `stage` / `attachment_style` 颗粒度非常粗；没有 InterlocutorState 12 维。

**第一性下该怎么落**：
- **不要**做 EmoGPT 那张"resistance 区间表"和"coaxing 类型 → 系数表"——那张表是 VZ 第一性原则明确禁止的硬编码映射。
- 落点应该是：`relationship_state` owner 多发布一组**learned readouts**（不是规则查表），由 metacontroller 在 z_t 空间学到"什么样的 relationship 状态下倾向选哪个 abstract action"。
- `InterlocutorState` 12 维可以作为 **perception readout** 加到 `user_model` owner，但每一维都必须从 LLM 结构化输出 + 嵌入相似度产出，**禁止**关键词。
- "渐进解锁"在 VZ 里就是 `vitals.recharge_per_regime` 在不同 relationship_stage 下的 schedule（已经有 regime-keyed 机制，不是新引擎）。

### Gap 10 — 9 类 RuntimeEventInput 完备性

EmoGPT §7.2 列出至少 12 类 RuntimeEventInput：`decision_made` / `assumption_recorded` / `problem_progress_assessed` / `outcome_observed` / `commitment_created` / `commitment_resolved` / `user_feedback_received` / `instruction_received` / `tool_outcome` / `crystal_evaluation` / `crystal_suppression` / `package_publication` / `bootstrap_consumption`。

**VZ 现状**：`semantic_state` 已经有 8 种 `SemanticProposalOperation`（OBSERVE / CREATE / REVISE / DEFER / ACTIVATE / COMPLETE / CLOSE / BLOCK）+ 4 种外源 event constructor（tool_result / profile / task / reviewed_knowledge）。**结构上已经齐**，缺的只是**特定语义事件类型**（如 `problem_progress_assessed`、`commitment_resolved` 这种 outcome 事件）。

**第一性下该怎么落**：
- 不开新 envelope。给 `commitment` / `plan_intent` / `execution_result` 这几个 owner 各加一个 `outcome_kind` enum 字段就够了。

### Gap 11 — ScenarioPackage 热加载 / install / uninstall

EmoGPT §11.10 让场景包可在线 install / uninstall + ETA 重载 + ActionGroup。

**VZ 现状**：`lifeform-evolution.scenario_pack` 是**离线**加载（`load_scenario_pack` 路径加载 JSON）；vertical 切换需要重启进程。

**第一性下该怎么落**：
- 第一阶段不做热加载（部署时一次性 freeze 是更安全的 SSOT 立场）。如果未来真要做：
  - `DomainExperiencePackage` 已经是 frozen dataclass，热加载等于重新构造 lifeform + drain 现有 session。
  - 不要做 `injecting SSOT fragment + 触发 ETA reload` 这种"运行时改本体"的设计，违反 R8 owner 单写者。

### Gap 12 — Affordance 之上的"Tool 选择质量"评估族

EmoGPT §9.5 + CP-16：`when_to_use` 文字长度 lint、selection lint warning 计数。这是把"prompt 里 tool 描述质量"也纳入评估。

**第一性下该怎么落**：等 Gap 1 落地后，把 lint 报告挂到 R12 F4（learning quality）或 F5（abstraction quality）的 metric 之一，不另开族。

---

## 优先级建议

按"对 NL+ETA 主路径增益最大、对 SSOT 破坏最小"排序：

| 优先级 | 缺口 | 估算工作量 | 影响 |
|---|---|---|---|
| **P0** | Gap 4（中频思考时钟 + ProvisionalLesson 作为 case_memory lifecycle）| 中 | 直接补齐 R1 中间频带，VZ 当前最薄弱 |
| **P0** | Gap 7（commitment owner 加 advocacy/alignment 状态机）| 小 | 补 R11 内部状态显式化，无新 owner |
| **P1** | Gap 1（Affordance 注册表 + lifeform-affordance wheel）| 中 | 让 lifeform 真正能"主动做事"，是 coding vertical 的硬阻塞 |
| **P1** | Gap 2 + Gap 3（统一 IngestionEnvelope + Apprenticeship trigger）| 中 | 解锁多源经验吸收，不需要新学习路径 |
| **P2** | Gap 8（cognitive_depth 进 z_t、participation_hint 进 regime snapshot）| 小 | 收紧 prompt planner 输入面 |
| **P2** | Gap 10（语义 owner 加 outcome_kind 枚举）| 小 | 闭合 commitment lifecycle 证据链 |
| **P3** | Gap 5（speculative / lazy expression）| 中 | 纯性能，等真正有延迟问题再做 |
| **P3** | Gap 9（InterlocutorState 学习式 readout）| 大 | 想清楚再做，最容易滑进硬编码 |
| **P4** | Gap 6（DLaaS 控制面：tenant 隔离 → exam → license）| 大 | 走 service 层增量，非 kernel |
| **P5** | Gap 11 / Gap 12 | 小 | 等其他都稳了再说 |

---

## 三条不能让步的"借鉴红线"

我把这三条单独拎出来，因为 EmoGPT PRD 里有不少东西**长得像功能实际是反模式**，对照 VZ `.cursor/rules` 你必须挡住：

1. **不照抄 Resistance / Coaxing / Surprise 的查表系数**（EmoGPT §13 那一堆 `× 1.5` / `× 0.7`）。这是 `no-keyword-matching-hacks.mdc` 明确禁止的硬编码 → 必须走 metacontroller learned weights。

2. **不照抄 EtiquetteWatchdog 的关键词过滤逻辑**（EmoGPT §4.5 "夜间静默时段 / 频率限制 / 不依赖 LLM 的纯 if/else"）。VZ 的 `etiquette_watchdog.py` 已经把它收敛成 *UX-only verdict* —— 拦的是 lifeform 层的发声时机，**绝不**反向影响 kernel 学习信号。继续保持这个边界。

3. **不照抄 `MetaAnalyzer` 只读不写但分析整个系统的"上帝视角"**（EmoGPT §17 的 18 条 CP）。VZ 评估走 R12 family report + `--require-family-pass` 是更克制的设计：评估是 **PE 的 readout**，不是上帝视角的"系统健康度"。CP-1 到 CP-18 这种 schema 检查可以做，但要降级成 contract test，不要做成"运行时 MetaAnalyzer 模块"。

---

## 你可以验证我这份判断的两个动作

1. 翻 `.cursor/rules/first-principles-not-patches.mdc` 里那张 8 库 owner 表，把上面 Gap 1–12 每个都落到一个 owner 上 —— 如果落不到，说明它要么不属于 VZ（应该留在 service / lifeform 层），要么需要新加一个 owner 但必须先在 `docs/DATA_CONTRACT.md` 的 slot 注册表 propose。

2. 翻 `docs/specs/00_INDEX.md` 的 11 个能力域索引，对每一条 Gap 找它该落在哪个 spec 下；找不到就是该补 spec —— spec 同步是动手前的硬前置。

需要的话我可以下一步帮你把 P0 的两条（Gap 4 + Gap 7）写成 `docs/specs/*.md` 的 spec 草稿，配 owner / contract / propagate 入口设计，再决定要不要切到 Agent 模式落地。