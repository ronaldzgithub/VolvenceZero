# Semantic State Owners Spec

> Status: draft
> Last updated: 2026-04-25
> 对应需求: R1, R2, R5, R7, R8, R11, R12, R15

## 要解决的问题

多轮交互中存在大量不能靠 prompt residue 临时重建的语义状态：计划、承诺、开放问题、用户模型、执行结果、信念假设、关系状态、目标价值与授权边界。它们必须由正式 owner 持有并通过不可变快照发布。

## 关键不变量

- 语义细节不存入 ETA / NL 本体；ETA 只消费 compact control advisory，NL 只决定更新时间尺度与沉淀路径。
- 九个 owner 都是独立 runtime slot，拥有自己的 frozen snapshot。
- 语义理解通过 typed `SemanticProposal` 进入 owner，禁止关键词规则直接驱动状态更新。
- 默认 package / synthetic path 使用 `NoOpSemanticProposalRuntime`，只发布低置信 observation，不伪造深语义。
- 每个 slot 有独立 wiring level / kill switch，迁移可回滚。

## 接口契约

新增 slots：

| Slot | Owner | Snapshot |
|------|-------|----------|
| `plan_intent` | `PlanIntentModule` | `PlanIntentSnapshot` |
| `commitment` | `CommitmentModule` | `CommitmentSnapshot` |
| `open_loop` | `OpenLoopModule` | `OpenLoopSnapshot` |
| `user_model` | `UserModelModule` | `UserModelSnapshot` |
| `execution_result` | `ExecutionResultModule` | `ExecutionResultSnapshot` |
| `belief_assumption` | `BeliefAssumptionModule` | `BeliefAssumptionSnapshot` |
| `relationship_state` | `RelationshipStateModule` | `RelationshipStateSnapshot` |
| `goal_value` | `GoalValueModule` | `GoalValueSnapshot` |
| `boundary_consent` | `BoundaryConsentModule` | `BoundaryConsentSnapshot` |

Proposal flow:

```text
substrate + memory + user_input
→ SemanticProposalRuntime
→ SemanticProposalBatch
→ owner-side merge in SemanticStateStore
→ public semantic snapshots
→ temporal / boundary / response / evaluation consumers
```

External adapter flow:

```text
tool/profile/task/reviewed-knowledge event
→ SemanticEventAdapter
→ AdapterSemanticProposalRuntime
→ SemanticProposalBatch
→ owner-side merge in SemanticStateStore
```

Adapters are structured-field mappers. They may map `status`, `consent_grants`, `consent_denials`, task state, or reviewed evidence fields to typed operations, but they must not inspect arbitrary text with keyword rules to decide behavior.

Character chapter adapter flow（2026-07-13）:

```text
ReviewedChapterExperience / CharacterSemanticEventBundle
→ CharacterChapterSemanticAdapter
→ AdapterSemanticProposalRuntime
→ SemanticProposalBatch
→ SemanticStateStore single-writer merge
```

`CharacterSemanticEvent` 是 reviewed chapter artifact 的 typed proposal source，不是新的 semantic owner。它必须携带 `target_slot`（9 slots 闭集）、`operation`、`summary`、`detail`、`confidence`、`evidence_locator`。adapter 只做结构字段映射，不能读取原文小说、不能用关键词决定 slot 或 operation。角色 vertical 可生成这些 events，但最终状态仍由 `SemanticStateStore` 持有并发布快照。

## 不属于本 spine 的相邻 owner

`decision_workspace`（`vz-cognition/decision_workspace`，默认 `SHADOW`）持有**决策结构**：哪些选项还活着、在哪些维度上比较、排名悬在哪些未知上、引用了哪些证据、结论处于什么状态。它**不是第十个 semantic owner**，理由有三：

- **不进 `semantic_spine_coverage` 分母**。该指标算的是五个 core semantic owner；把它加进去会让全部历史 paper-suite / companion 读数因为与关系状态无关的原因整体平移。
- **不拥有语义事实，只持引用**。用户重视什么归 `goal_value`，什么还没解决归 `open_loop`，什么需要核验归 `belief_assumption`，哪些计划是候选归 `plan_intent`。workspace 只记录"选项 A 在桌上、对应 plan-ref X"。这条靠**结构**保证而非靠自觉：`DecisionOption` / `DecisionUnknown` 根本没有能放文本的字段。同一事实两个写者，在跨会话水合后必然分叉。
- **不决定自己何时存在**。激活值读自 `regime` 的 `participation_hint.panorama_level`（见 [cognitive-regime.md](./cognitive-regime.md) §Panorama 参与门）。该模块完全没有"重要性""风险""话题"的概念，只读一个枚举。SILENT → 不实例化；BRIEF → 只维护选项集与未知项、不发布维度与结论；STRUCTURED → 全量。

测试 `tests/test_decision_workspace.py`：固定一组"最像决策"的 owner 快照，只改门的取值，验证三档行为差异——任何本地"这看起来很重要"的判断都会立刻暴露。

### 安全在排名之上，不在排名之内

`boundary_policy` 报 `RiskBand.CRITICAL` 或 `refer_out_required` 时，workspace 发布 `conclusion_state = withheld-safety` 且 `safety_hold=True`，**无论区间分离得多干净、无论用户怎么给权重**。安全若是六维之一，"用户把自己的安全调到最低"就是一个可达状态——不可接受。

三个实现选择值得记：模块位置排在 `boundary_policy` 之后，因为读到上一轮的安全带意味着刚转危的这一轮仍会发布推荐；不自建检测器，读既有 owner，因为第二个安全判断就是第二个要维护对的东西，而出错的往往正是它；通过 vz-contracts 的 `BoundaryReadout` 协议读取而非 import owner——vz-cognition 在 vz-application 之下，函数内 import 能绕过契约测试（它只查模块级 import），但分层照样破了，只是看不见。`_CRITICAL_RISK_BAND` 字面量与枚举的一致性有专门测试守门：安全检查静默失效是最糟的失效方式。

### 区间估值、期权价值与信息价值

`decision_workspace/valuation.py`。三个量由**同一个模型**算出，不是三套启发式——两套分开调参的启发式总能被调到在第四幕转录上一致。

- **未知项加宽区间，已知折扣平移区间。** 三年锁定期是折扣，整段下移；"股权文件没人看过"不是折扣，它使区间变宽而 `base` 不动——未知不是"估计错了"的证据，只是"估计没有支撑"的证据。把未决问题坍缩成中点正是让决策辅助变危险的伪精确。
- **维度按共单调聚合**（极值直接相加）。独立性假设会收窄总区间，从而让"某选项胜出"更容易成立；相关性未知时应取使宽度最大的假设，保守方向朝着"不下断言"。
- **只有严格分离才有 leader。** 区间重叠时 `leader_ref` 为 `None`，只发 `most_robust_ref`。第四幕台本的"现在收益最高的是先分开三个月"正是这条要挡的——重叠时那个结论不存在。
- **期权价值 = 可逆性 × 该选项买到的信息**。不可逆但能解答一切的选项拿不到分：你用不上学到的东西。期权价值不抬高区间下界（选项自身走坏的分支里，保住的选择权一文不值），且尺度刻意小于维度值——重叠时它决定 `most_robust_ref`，但永远无法把一个区间推离另一个。
- **VOI = 解答该未知后排名是否会变**，量化为"头两名区间重叠减少了多少"。第一版用的是"leader 区间宽度减少量"，那是盲的：加宽了**挑战者**的未知会得 0 分，哪怕它正是决定能否下结论的那一个。VOI 为 0 的未知不该问——这也是**收敛终止条件**，缺了它系统会一直追问，读起来像审讯。

`decision_workspace/rendering.py` 把这些变成 typed `ClaimLicence` 而非 prompt 措辞：prompt 里的"别过度断言"是请求，装配器必须查询的 licence 是约束。区间重叠 → 只许 `robustness`（更稳健、更可逆、保留更多选择）；严格分离 → 才许 `comparative`；无 evidence ref 的数字只许标记为待核验；`safety_hold` 直接撤销一切排名断言。

第四幕作为**验收样本**跑在 `tests/test_decision_valuation.py`：断言的是"算术支持哪些说法"，不是复现台词。样本里"谈好再离"被刻意给了最高的账面数字，否则"可逆选项胜出"这条断言毫无力度。结果：无 leader（重叠）、`most_robust = separate`、下一个该问的是股权归属、所有数字均不可作为事实陈述。

### 全景怎么出现在回复里

在此之前，判断力、结构、算术都在，但用户什么也看不见：workspace 的读者数量为 0，三档 `panorama_level` 跑出的 prompt plan 完全相同——`panorama_level` 在下游只做一件事，SILENT 时丢掉 `CLARIFICATION`。

补齐的链路：

```
decision_workspace 快照
  → plan_panorama_render(workspace, valuation?) → PanoramaRenderPlan | None
  → LifeformSession.panorama_render_plan（provider）
  → PromptPlanner 加 SectionId.DECISION_PANORAMA
  → GroundedResponseSynthesizer._render_decision_panorama
```

**关键分工：能说什么在认知层决定，怎么说在表达层。** claim licence 与算术放在一起，措辞留给表达层。写进 prompt 的"别过度断言"是请求，装配器必须查询的 licence 是约束——而后者失效时会留下痕迹，前者不会。这也让契约测试断言的是"约束的效果"而非"具体措辞"，改写一个句子永远不会悄悄放松系统有资格断言的东西。

三档行为：

| 档 | 输出 |
|---|---|
| SILENT | `plan_panorama_render` 返回 `None`，没有 section。**是真实的缺席，不是空 section**——下游没有东西可以误展开 |
| BRIEF | "看起来有 4 个选项。要不要摊开来看？"——只发现，不铺开。检测到就立刻铺全景，正是对还没决定要不要一起想的人讲话 |
| STRUCTURED | 选项数 / 维度数 + licence 允许的排名说法 + 未决项 + 下一个该问的 + 未核实数字的标记 |

区间重叠时渲染器**没有任何一条分支**能到达"谁最高"的句子——它只会说"范围重叠，没有哪个赢下来，但先分开是最可逆、买到最多信息的那个"。这正是台本那句"现在收益最高的是先分开三个月"的替代，且是从算术里得出的。安全保留时连排名句都不出。

`LifeformSession.panorama_render_plan` **不附带 valuation**：区间需要数字来源（用户给的、或带溯源的研究主张），从结构本身编出数字正是这套设计拒绝的伪精确。没有数字时全景描述决策而拒绝排名。

### 什么算证据：来源 / 时间 / 适用范围

工具返回的一条主张只有在能说清**它从哪来、什么时候为真、适用于什么**时才算证据；缺任何一项，它就只是一句碰巧挨着工具调用的断言。搜索摘要与凭空编造的句子到达 owner 时形状完全一致，文本本身区分不了，所以规则必须咬在结构上。

`EvidenceProvenance`（`semantic_state/contracts.py`）携带 `claim_id` / `source` / `as_of` / `scope` / `confidence`，`ToolResultSemanticEvent.provenance` 是它的元组。`scope` 是适用边界——可比公司的估值不是这家公司的估值，而这个差别一旦进了句子就看不见了。

**执行方式是调低一个诚实的置信度，而不是给未溯源主张开一条特殊路径**。特殊路径是会在下一个调用点被忘记的东西。`BeliefAssumptionModule` 本来就按 `BELIEF_VERIFICATION_CONFIDENCE_THRESHOLD`（0.55）分桶：≥ 阈值进 `beliefs`，< 阈值进 `verification_needs`。adapter 把溯源不全的主张封顶在阈值之下，剩下的由既有分桶完成。该阈值已从字面量提为具名常量并由两处共用——阈值动了而封顶没动的话，无法核实的研究会悄悄开始被当作 belief 发布，而这个失效看起来像是系统变得更自信了。

由此**自动**得到一条闭环，没有任何地方被告知要去追查自己的未溯源主张：

```
未溯源主张 → verification_needs → unknown_dominance 升高
           → panorama 门读到 → VOI 把它排为下一个该问的
```

逐条主张分别入库而非合并成一条记录：一次研究调用常常同时返回可核验的融资日期和纯属猜测的估值，合并后它们共享一个置信度，可核验的那部分会把猜测一起拖过阈值。

审计串写明缺了哪一项（`incomplete_provenance=as_of`）——"置信度低"不是诊断，"没有时间"才是。

工具侧的 `research_public_company`（`lifeform-domain-growth-advisor/research_affordances`）把这条规则前推到边界：`output_schema` 把 `source` / `as_of` / `scope` 列为 required，拿不出溯源的后端在契约层就失败，而不是返回一段听起来很权威的散文。两条边界写进描述符本身而非留给调用方——参数只接受公司标识而**没有**接受个人的参数（"帮我决定要不要离婚"的对话恰恰会诱导去查具体某个私人），且需要 `public_research` 授权、在 `emotional_support` / `repair_and_deescalation` regime 下被禁用。急性痛苦中的人不是在要求你去调查他的伴侣。

invoker 侧按**载荷形状**（`claims` 键）而非工具名提取，所以这不是 affordance spec 禁止的硬编码路由。

## ETA / NL 集成

- `TrackTemporalModule` 直接消费九个 semantic slots，并把它们压成 `semantic_pressure`，作为 control advisory 写入 public temporal description / feedback signal。
- `ResponseAssemblyModule` 消费九个 slots，发布 `semantic_record_counts`、`semantic_control_signal`、`semantic_residue_summary`。
- `BoundaryPolicyModule` 消费 `boundary_consent`，缺失授权或拒绝边界会提升澄清/边界约束。
- `EvaluationBackbone` 记录 semantic readout metrics，并发布 `semantic_spine_coverage` / `cognitive_loop_readiness` 作为窄 cognitive loop 的证据读数；evaluation 只消费 owner 快照，不把 evaluation 变成学习源头。
- session-post request 携带 semantic state descriptions，供 background-slow 层沉淀与审计。
- `AgentSessionRunner` exposes a bounded pending external-event queue. Each turn drains the queue into `AdapterSemanticProposalRuntime`, so external events are consumed exactly once unless resubmitted.
- `BrainSession` exposes package-facing helper methods for tool result, profile/settings, task/calendar, and reviewed-knowledge events. These helpers enqueue structured events only; they do not mutate owner stores.

## 回滚

每个 semantic slot 都由 `FinalRolloutConfig` 暴露 wiring level，并支持 `kill_switches`。禁用某个 slot 时，下游通过 runtime placeholder 退化，不读取 owner 私有状态。

## 验收读数

当 `relationship_state`、`goal_value`、`boundary_consent`、`commitment`、`execution_result` 与 `evaluation` 同时 ACTIVE 时，`FinalAcceptanceReport` 必须能看到：

- `semantic_spine_coverage = 1.0`
- `cognitive_loop_readiness` 已发布
- session / cross-session report 中的 `semantic_spine_readiness` 趋势由 `cognitive_loop_readiness` 派生，用于判断地基是否退化；`semantic_spine_coverage` 只作为完整性验收，不混入趋势
- dialogue benchmark case report、emergence dashboard 与 paper-suite metric values 汇总 `mean_semantic_spine_coverage` / `mean_cognitive_loop_readiness`，作为产品对话回归层和证据产物层的地基证据
- NL essence assessment 发布 `semantic-spine-ready` gate；该 gate 先作为审计证据，不进入默认 required gate 列表
- `claim_companion_stateful_relationship` 的当前轻量 verdict 消费 `semantic-spine-ready` 与 dashboard 读数，作为完整 companion 证据前的状态感知地基门
- paper-suite manifest 将 canonical semantic spine 指标列入 secondary metrics；companion verdict 优先消费 repeated-run summary，reference dashboard 仅作 fallback
- `semantic_state.quality` 提供 proposal-level quality harness，先用于 `boundary_consent` / `goal_value` 的 precision / recall / false-positive / fallback 评估；它只评估 proposal runtime，不写 owner store，并发布 shadow-only `would_block` / `would_allow` / gate reason 读数
- dialogue paper-suite export 可将 proposal quality shadow report 作为 `semantic_proposal_quality_shadow.json` sidecar 与 `EvidenceBundle.reference_artifacts` 条目导出；该 artifact 标记为 non-gating，不改变 owner apply 或 claim verdict

该检查只验证 owner 快照是否形成窄 cognitive loop 证据，不把 readiness 当作学习奖励，也不允许 evaluation 重建 owner 内部状态。

## 变更日志

- 2026-07-27 (P4 最后一公里): 全景真正出现在回复里。`PanoramaRenderPlan` + `plan_panorama_render`（认知层决定能说什么）、`SectionId.DECISION_PANORAMA` + planner 接线、`_render_decision_panorama`（表达层决定怎么说，措辞受 licence 约束）、`LifeformSession.panorama_render_plan` provider。此前三档 prompt plan 完全相同、workspace 读者为 0。测试 `tests/test_panorama_render.py`。
- 2026-07-27 (P4 研究工具): 证据溯源契约（`EvidenceProvenance` + `ToolResultSemanticEvent.provenance`），溯源不全的主张按 `BELIEF_VERIFICATION_CONFIDENCE_THRESHOLD` 封顶从而落入 `verification_needs`；阈值由字面量提为具名常量供 owner 与 adapter 共用。`research_public_company` 描述符落在 growth-advisor vertical，schema 强制 source/as_of/scope、参数不接受个人、需 `public_research` 授权且在情绪/修复 regime 下禁用。invoker 按载荷形状提取 claims（非按工具名路由）。`submit_tool_result` 增加 `provenance` 形参，走既有 `EnvironmentOutcome` 通道，无新数据通道。测试 `tests/test_research_evidence_path.py`。
- 2026-07-27 (P4): `decision_workspace` 增加安全保留（读 `boundary_policy`，经 vz-contracts `BoundaryReadout` 协议，`BoundaryDecisionReadout` 补 `risk_band`）、区间估值 / 期权价值 / VOI（`valuation.py`）与 claim licence（`rendering.py`）。模块位置移到 `boundary_policy` 之后以保证同轮读取。过程中修掉一处 VOI 盲区：宽度收益原本只测 leader 区间，导致加宽挑战者的未知得 0 分；改为测头两名的重叠减少量。测试 `tests/test_decision_valuation.py`（含第四幕验收样本）+ `tests/test_decision_workspace.py` 安全段。
- 2026-07-27: 新增相邻 owner `decision_workspace`（`SHADOW`，见上节）。spine 仍是 9 个 owner，`semantic_spine_coverage` 分母不变。所有权边界靠字段形状强制（记录类型没有可放文本的字段）+ 行为测试（同一批 owner 快照下只改 panorama 门取值）。测试 `tests/test_decision_workspace.py`。
- 2026-07-17: G2 LLM proposal 覆盖 9/9。`_GENERIC_LLM_SLOT_IDS` 从 4 slot 扩到 8（`plan_intent` / `open_loop` / `execution_result` / `belief_assumption` 加入既有 JSON-schema generic 路径；commitment 仍走专用分类器，合计 9/9 全部 semantic owner 具备 typed LLM proposal source）。per-slot 语义说明集中在 `_GENERIC_SLOT_SEMANTIC_HINTS`（llm-prompt-centralization；原四 slot 的 prompt 字节不变）。owner 单写者、`min_proposal_confidence` 过滤、unparseable→NoOp fail-safe 均不变。测试：`tests/test_llm_semantic_runtime.py` 新四 slot 参数化用例 + hint-line 边界用例。
- 2026-07-14: CP-12 第二波 publisher 接线（GAP-05）。`plan_intent`（kind
  `PLAN_INTENT_PROGRESS`, track world）/ `open_loop`（`OPEN_LOOP_CLOSURE`,
  world）/ `belief_assumption`（`BELIEF_ASSUMPTION_STABILITY`, world）/
  `user_model`（`USER_MODEL_PACING`, self）开始在快照发布
  `owner_prediction_signals`，机制与 first wave 完全一致（store-held v2
  learned forecaster + owner 自 settle）。`user_model` 只预测自身 aggregate
  pacing/stability readout，不与 ToM 四 owner 重复拥有对他人的
  belief/intent/feeling/preference。PE settlement 覆盖扩至 9 slot。测试：
  `tests/contracts/test_owner_prediction_signal.py`（ALL_WAVES 参数化）。
- 2026-07-12: CP-12 owner prediction signal contract。`SemanticOwnerModule` 新增
  `owner_prediction_kind` / `owner_prediction_track` 类属性与
  `_owner_prediction_signals(...)` 助手；五个 first-wave owner（commitment /
  relationship_state / goal_value / boundary_consent / execution_result）在快照
  发布 `owner_prediction_signals`：每轮签发一条对自身 compact readout 的
  persistence-prior v1 预测，并对上一轮 pending 预测由 owner 自己 settle
  （observed readout + outcome evidence）。pending 预测与 id 序列由
  `SemanticStateStore` 持有（owner 模块每轮重建，store 是 durable 组件）。
  mismatch 只由 PE owner 计算（见 `prediction-error-loop.md` 同日条目）；
  消费者无需读取 owner 内部字段即可完成 settlement 消费。第二波
  （plan_intent / open_loop / belief_assumption / user_model）kind 已在闭集
  enum 预留，publisher 于 2026-07-14 接线（见上方条目）。
- 2026-07-13: 登记 character chapter adapter flow。逐章主观烘焙的
  `CharacterSemanticEventBundle` 只能作为 typed proposal source，仍由
  `SemanticStateStore` 单写者合并；禁止用原文关键词或角色 vertical 直写 9 个
  semantic owners。
- 2026-05-03: 新增 `semantic_state.quality` proposal quality harness，首批覆盖 `boundary_consent` / `goal_value` scripted LLM cases，用于在 owner 合并前评估 typed proposal 输入质量；shadow gate 只报告 would-block，不阻断 runtime。
- 2026-05-03: dialogue paper-suite export 新增 non-gating `semantic_proposal_quality_shadow.json` sidecar，并把同一 payload 挂入 evidence bundle reference artifacts。
- 2026-05-03: Commitment / OpenLoop / BoundaryConsent / GoalValue / RelationshipState 增加 owner-side lifecycle / continuity readouts；`LLMSemanticProposalRuntime` 最小扩展到 `boundary_consent`、`goal_value` 的 schema-bound typed proposal 路径，非目标 slot 继续 delegate。
- 2026-05-03: `clone_semantic_store` 开始保留 lifecycle / follow-up policy / typed outcome maps，避免跨上下文复制时丢失 owner-side continuity evidence。
- 2026-05-03: paper-suite manifest 将 canonical semantic spine coverage / cognitive loop readiness 纳入 secondary metrics，companion verdict 优先消费 repeated-run summary。
- 2026-05-03: `claim_companion_stateful_relationship` 当前轻量 verdict 接入 `semantic-spine-ready` 与 dashboard 读数；retain 仍需 cross-session gate，避免把单轮读数夸大为完整 companion 证明。
- 2026-05-03: NL essence assessment 新增 `semantic-spine-ready` gate，将 semantic spine 读数提升为 paper-suite 审计门，但暂不加入默认 required gate。
- 2026-05-03: Dialogue benchmark case report、emergence dashboard 与 paper-suite metric values 开始汇总 semantic spine coverage / cognitive loop readiness，让对话回归和证据产物能直接观察认知地基状态。
- 2026-05-03: `cognitive_loop_readiness` 进入 session / cross-session 趋势，`EvolutionJudgement` 在 `semantic_spine_readiness` 明显退化时回滚，避免扩能力掩盖认知地基退化。
- 2026-05-03: `FinalAcceptanceReport` 开始要求 ACTIVE 核心 semantic spine 发布 `semantic_spine_coverage` 与 `cognitive_loop_readiness`，作为继续扩能力前的验收门槛。
- 2026-05-03: `EvaluationBackbone` 新增 `semantic_spine_coverage` 与 `cognitive_loop_readiness`，基于 `relationship_state`、`goal_value`、`boundary_consent`、`commitment`、`execution_result` 五个 owner 的公开快照评估窄 cognitive loop 的地基成熟度。
- 2026-05-03: 四个情绪决策相关 owner 增加 owner-side readout：
  - `relationship_state` 发布 `emotional_load`、`repair_need`、`trust_delta`、`attunement_gap`、`stabilization_need`
  - `goal_value` 发布 `value_conflict`、`decision_readiness`、`active_tradeoff_count`、`reversibility_need`、`goal_shift_pressure`
  - `boundary_consent` 发布 `autonomy_risk`、`consent_clarity`、`professional_scope_pressure`、`overreach_risk`
  - `user_model` 发布 `preferred_support_pacing`、`decision_style`、`overwhelm_pattern_strength`，并开始从 typed profile proposals 沉淀 `durable_goals`
  - `response_assembly.support_before_decision_pressure` 优先消费这些 owner-side readouts，ETA 只消费压缩后的 action-family advisory，不拥有语义事实
- 2026-04-25: 初始版本，建立九个 semantic owner、typed proposal path、ETA/NL/response/evaluation/session-post 集成边界。
- 2026-04-25: 新增 external semantic adapters：tool result、profile/settings、task/calendar 与 reviewed knowledge 事件经 adapter runtime 转为 typed proposals 后进入 semantic owners。
