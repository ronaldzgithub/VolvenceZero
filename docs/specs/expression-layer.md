# Expression Layer Spec

> Status: draft
> Last updated: 2026-05-06
> 对应需求: R4 (内部控制在 token 空间之上), R8 (契约优先 / 快照优先), R11 (内部状态可发布)

## 要解决的问题

`AgentResponse` 是 kernel 跨过 lifeform 边界的最终表达制品。下游消费者
（评估、gate、reflection、operator dashboards）需要可靠地知道**这条回复是被哪些
内部状态塑形的**，而不能依赖 substring 匹配某段 UX 文案——文案可改、可国际化，
substring 匹配是脆弱硬编码（违反 `no-keyword-matching-hacks`）。

本 spec 定义两条契约：

1. **typed `rationale_tags`** —— 每条 `AgentResponse` 都要发布结构化 audit 标签。
2. **reflection lesson / tension 作为 typed enum** —— 文案在 lifeform 层、id 在
   kernel 层，单一来源是 enum。

## 关键不变量

1. `AgentResponse.rationale_tags: tuple[str, ...]` 是表达层暴露给下游的**唯一**
   结构化 audit surface。`rationale: str` 是人类可读摘要，**phrasing 不是契约**。
2. 每个标签是 `"key=value"` 形式或 bare flag（如
   `"reflection_writeback=applied"`）；标签集合可增不可删。
3. 任何 gate / 评估 / reflection 想要"这一回合是否触发了 X"必须读
   `rationale_tags`，禁止在 `rationale` / `text` 上 `in` 操作。
4. Reflection 发布的 lesson / tension id（`ReflectionSnapshot.lessons_extracted`
   / `tensions_identified`）必须是 `ReflectionLessonId` /
   `ReflectionTensionId` enum 中的 value。增加新 id 必须先在 enum 加成员。
5. UX 文案（per-id hint）住在 `lifeform_expression.reflection_hints`，**不在
   kernel**。kernel `ResponseSynthesizer` 不渲染 per-id 的具体提示文本。
6. lifeform-expression 的 `_render_reflection` 是 reflection hint 文案的**唯一
   渲染入口**；多入口意味着 SSOT 破裂。
7. **Mind/Face 隔离**：Face 层（`lifeform-expression` / OpenAI compat renderer /
   外部 LLM judge 输出）只表达和 readout，不拥有学习、gate、credit、PE 或 memory
   状态。Face 层不得 import / 调用 `volvence_zero.credit`,
   `volvence_zero.evaluation`, `volvence_zero.prediction`,
   `volvence_zero.temporal`, `volvence_zero.memory`,
   `volvence_zero.regime`, `volvence_zero.audit` 等 Mind / gate owner 包；只允许
   读取已发布的 response / application contract surface（如
   `volvence_zero.agent.response`, `volvence_zero.application.runtime`），以及窄化的
   expression-facing readout contract（如 `volvence_zero.regime.hints`）。
8. LLM judge / expression naturalness score 永远是 readout；不得写回 reward、
   `ModificationGate`、controller gradient、Face fine-tune 或 owner 内部状态。

## 工程挑战

- 现有 `_attach_plan_rationale` / `_build_rationale` 把 plan tag 拼进
  `rationale` 字符串。Wave 1 part A 改成 dual-write：rationale 仍然包含人类可读
  字符串，rationale_tags 是结构化镜像。
- 渲染分支（`_render_acknowledge` / `_render_regime_frame` / `_render_open_loop`
  / `_render_next_step`）选了哪个 variant 也是下游 gate 需要知道的。这些 variant
  通过 `section_tags` 列表回流到 `rationale_tags`。
- `relationship_repair_alpha_gate` 是第一条强约束 gate；W1 part A 用它做 lockdown
  test。

## 接口契约

### Inputs

| Owner | Snapshot | 标签 (示例) |
|---|---|---|
| `regime` | `RegimeSnapshot` | `regime=<id>`, `regime_switched` |
| `temporal_abstraction` | `TemporalAbstractionSnapshot` | `temporal=<action>`, `switch_gate=<float>` |
| `rupture_state` | `RuptureStateSnapshot` | `repair_alpha=<kind>`, `repair_confidence=<float>` |
| `vitals` (lifeform) | `VitalsSnapshot` | `vitals_pressure=<drives>`, `vitals_total_pe=<float>` |
| `interlocutor` | `InterlocutorState` | `interlocutor_conf=<float>`, `il_*` |
| `reflection` | `ReflectionSnapshot` | `primary_lesson=<id>`, `primary_tension=<id>`, `reflection_writeback=applied` |
| `affordance` (lifeform) | `AffordanceSnapshot` | `affordance=selected:<name>;score:<float>`, `affordance_blocked=<n>` |

### Outputs

`AgentResponse(text, regime_id, abstract_action, rationale, rationale_tags)`

`rationale_tags` 必含至少：
- `regime=<id|none>`
- `switch_gate=<float>`
- `risk=<low|medium|high|critical>`
- `plan=intent:<intent>;sections:<…>;q:<int>`（Grounded / LLM 路径）

### Render-section variant tags (新增)

由 `GroundedResponseSynthesizer._render_section` 在选择 variant 时写入：

| Section | Tag | 示例 variants |
|---|---|---|
| ACKNOWLEDGE_PRESSURE | `acknowledge_section=<variant>` | `repair_alpha`, `repair_regime`, `interlocutor_repair`, `interlocutor_direct_task`, `interlocutor_emotional`, `emotional_support_regime`, `continuum_high`, `default` |
| REGIME_FRAME | `regime_frame_section=<variant>` | `repair_alpha`, `emotional_support`, `guided_exploration_*`, `problem_solving`, ... |
| OPEN_LOOP_HANDOFF | `open_loop_section=<variant>` | `repair_alpha`, `case_or_playbook`, `default` |
| NEXT_STEP | `next_step_section=<variant>` | `repair_alpha`, `support_or_repair`, `problem_solving`, `guided_exploration_*` |
| AFFORDANCE_OFFER | `affordance=selected:<name>;score:<float>` | owner-approved offer；只提出可用能力，**不自动调用**；planner 由 `AffordanceSnapshot.selected` 触发，缺 snapshot 时 no-op |

## 与其他能力域的关系

- `vz-cognition.reflection` 拥有 lesson/tension id（`ReflectionLessonId` /
  `ReflectionTensionId`）。lifeform-expression 不重新定义 id，只翻译。
- `lifeform-evolution.relationship_repair_alpha_gate` 是 typed-tag 契约的第一个
  消费者。任何 gate / 评估扩展必须沿用 typed-tag 路径。
- LLM-backed `LifeformLLMResponseSynthesizer` 通过 `_attach_plan_rationale` 同样
  发布 typed tag。
- `tests/contracts/test_mind_face_isolation.py` 静态守门 Face 层不得反向 import
  Mind / gate owner 包，防止表达层在未来 PR 中悄悄变成第二学习源。

## 变更日志

- **2026-05-06**: 初版（W1 part A + part B）。引入 `rationale_tags` 字段、render
  section variant tags、`ReflectionLessonId` / `ReflectionTensionId` enum、
  `lifeform_expression.reflection_hints` 模块。kernel `ResponseSynthesizer.synthesize`
  移除 inline `lesson_hint_map` / `tension_hint_map`；UX 文案下放 lifeform 层。
- **2026-05-22**: OA-2 最小收敛切片。显式形式化 Mind/Face 隔离，并增加静态
  contract test 防止 Face 层反向 import credit / evaluation / PE / memory / regime /
  audit owner。
