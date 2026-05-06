# Rupture and Repair Spec

> Status: draft
> Last updated: 2026-05-05
> 对应需求: R-PE, R7, R8, R11, R15

## 要解决的问题

Companion AI 不是封闭优化问题。关系不是通过完美响应建立，而是通过
**rupture-repair 循环**建立。系统必须：

1. 观察到一次 rupture（misread / over-directive / pushed-too-fast / cold /
   unsafe / abandoned）；
2. 用**具体**的方式修复（命名发生了什么、给出一个可撤销调整）；
3. 记得这次修复；
4. 不再以相同方式犯同样的错误；
5. 并且**从不**把内部 PE spike 单独当作 rupture 的判决——rupture 的权威信号
   来自**外部**（用户、环境、人工复审）。

本 spec 定义 `rupture_state` owner 和 `dialogue_external_outcome` 的契约，
确保这条循环端到端可追溯、可回滚，且永远不会退化为 LLM 自判或关键词匹配。

## 关键不变量

1. `rupture_state` 是新的 kernel owner（`vz-cognition`）；它读上游 snapshot，
   自身从不被其它 owner 重建。
2. `rupture_kind` 是**证据桶标签**（evidence-bucket label），不是情绪分类。
3. 只有至少一个**非 PE 的 typed evidence source** 触发时，才能写出
   `rupture_kind`；否则 `internal_suspected_only=True` 且 `rupture_kind=None`。
4. 添加一个新的 `RuptureKind` 必须先添加一个新的 typed signal source；不允许
   用 LLM 或关键词在运行时凭空生成新 kind。
5. `dialogue_external_outcome` 是外部 outcome 进入内核的**唯一** snapshot 通道；
   `submit_dialogue_outcome` 不直接写 memory / regime / PE 内部状态。
6. 由 rupture-repair 产生的持久记忆写入只能通过 `ReflectionEngine.apply(...)`
   触发，走 `memory_store.create_checkpoint` / `restore_checkpoint` 路径。
7. LLM proposal 默认禁用；启用时只能作为低置信度 proposal，**不得**成为 judge
   或 reward。

## 词汇表（closed vocabularies）

### `RuptureKind`

| value | 触发路径（典型 evidence source） |
|---|---|
| `misread` | `EXTERNAL_USER.MISSED` 或 `HUMAN_REVIEW.MISSED` |
| `over_directive` | `EXTERNAL_USER.OVER_DIRECTIVE` 或 `HUMAN_REVIEW.OVER_DIRECTIVE` |
| `pushed_too_fast` | `EXTERNAL_USER.COME_BACK` + behavioral 重复收回 |
| `cold` | `EXTERNAL_USER.MISSED` + 关系温度 PE 高但无 decision 进展 |
| `unsafe` | `EXTERNAL_USER.UNSAFE` 或 `ENVIRONMENT.UNSAFE_FLAG` |
| `abandoned` | `EXTERNAL_USER.ABANDONED` 或 behavioral 沉默退出 |

### `DialogueExternalOutcomeKind`

| value | 含义 |
|---|---|
| `HELPED` | user 明确表示当前交换有帮助 |
| `FELT_HEARD` | user 表示 before-advice 的被听见感 |
| `MISSED` | user 说 missed / cold / not heard |
| `OVER_DIRECTIVE` | user 表示被推着走 / optimized |
| `DECISION_CLEARER` | user 表示决定 / 下一步更清晰 |
| `COME_BACK` | user 要求稍后回到此话题 |
| `UNSAFE` | user 或 environment 标记不安全 |
| `ABANDONED` | user 退出 / 不再响应 |

### `DialogueExternalOutcomeEvidence.source`

- `USER_EXPLICIT`：user 通过 `submit_dialogue_outcome(...)` 直接提交；最高权威。
- `HUMAN_REVIEW`：blind review entry。
- `ENVIRONMENT`：运行环境安全 / 工具 outcome（不从文本推断）。
- `LLM_PROPOSAL`：**默认禁用**；启用需 `BrainConfig.allow_llm_outcome_proposals=True`；
  即使启用也只作为低置信度 proposal，写入 `confidence ≤ 0.4`。

## 结构化映射 `ExternalOutcomeKind → RuptureKind`

这是一张**封闭 1:1 结构化查表**，不是启发式：

| `ExternalOutcomeKind` | `RuptureKind` |
|---|---|
| `HELPED` | （none） |
| `FELT_HEARD` | （none） |
| `MISSED` | `misread` |
| `OVER_DIRECTIVE` | `over_directive` |
| `DECISION_CLEARER` | （none） |
| `COME_BACK` | `pushed_too_fast` |
| `UNSAFE` | `unsafe` |
| `ABANDONED` | `abandoned` |

`RuptureKind.cold` 需要两个 source 组合（`MISSED` + relationship 温度相关 PE），
它是唯一不在上表中由单个 external outcome 直接产生的 kind；由 owner 聚合时命中。

## 接口契约

### 发布 slot

| Slot | Owner | Value | Wiring | Frequency | Consumers |
|---|---|---|---|---|---|
| `rupture_state` | `RuptureStateModule`（vz-cognition） | `RuptureStateSnapshot` | SHADOW | per turn | reflection, dialogue_trace (diagnostic) |
| `dialogue_external_outcome` | `DialogueExternalOutcomeModule`（vz-runtime） | `DialogueExternalOutcomeSnapshot` | ACTIVE | per turn | prediction_error, regime, rupture_state, reflection |

### `RuptureStateSnapshot`

```python
@dataclass(frozen=True)
class RuptureStateSnapshot:
    rupture_signal_strength: float              # [0, 1]; 归一化的外部证据源计数
    rupture_kind: RuptureKind | None            # 仅在至少一个非 PE 源触发时非 None
    confidence: float                           # [0, 1]; 贡献源 confidence 均值
    internal_suspected_only: bool               # True iff 只有 INTERNAL_PE 触发
    evidence_sources: tuple[RuptureEvidenceSource, ...]
    contributing_signals: tuple[RuptureContributingSignal, ...]
    description: str
```

### `DialogueExternalOutcomeSnapshot`

```python
@dataclass(frozen=True)
class DialogueExternalOutcomeSnapshot:
    turn_index: int
    entries: tuple[DialogueExternalOutcomeEvidence, ...]
    description: str
```

### v0 聚合规则（transparent, not learned）

- `rupture_signal_strength = bounded_count(active non-PE typed sources) / MAX_EXPECTED`
- `confidence = mean(source.confidence for source in contributing_signals)`
- `internal_suspected_only = (only RuptureEvidenceSource.INTERNAL_PE fired)`
- `rupture_kind` 通过上表 1:1 映射得到；同轮多个映射命中时，按 severity 顺序
  `unsafe > abandoned > over_directive > misread > pushed_too_fast > cold` 取最高。
- 没有手调权重；学习式权重属于 post-v0。

## 与其他能力域的关系

- **依赖上游 slot**：`prediction_error`, `relationship_state`, `response_assembly`,
  `dialogue_trace_snapshot`, `dialogue_external_outcome`。
- **被 ReflectionEngine 消费**：`reflection` 在 session-post slow loop 聚合
  `rupture_state` snapshot 序列 + 收到的 `DialogueExternalOutcomeEvidence` +
  `RegimeSnapshot.delayed_attributions`，产出内部 `RuptureRepairLesson`，由
  `ReflectionEngine.apply` 写入 durable memory。
- **不做的事**：
  - 不直接写 memory、不直接改 regime、不直接改 ETA；
  - 不从自由文本推断；
  - response_assembly 在 v0 **不**根据 `rupture_state` 调整输出
    （只 SHADOW 发布；active repair primitive 属于 post-v0 M7）。

## MemoryEntry / Kind 的实现选择

源文档 `docs/moving forward/real-open-dialogue-learning-loop.md` 中提到的
`MemoryEntryKind` / `DurableMemoryEntry` 在本实现中**故意**落地为：

1. `MemoryEntry` 继续作为唯一的 memory artifact 类型（无新 enum）；
2. rupture-repair 条目通过固定的 tag schema 识别：
   - 必含 tags：`rupture_repair`, `rupture_kind:<kind>`,
     `repair_outcome:<observed|pending>`, `user_scope:<user_id_or_anon>`,
     `source_wave:<wave_id>`；
   - `content` 中带结构化 JSON：`rupture_kind`, `repair_move`,
     `source_turn_index`, `source_wave_id`, `observed_outcome_kind`, `confidence`；
3. 写入路径只有 `ReflectionEngine.apply(...)`；
4. 读取侧通过 tag 过滤 + content 解析（由 reflection / retrieval 负责）。

**理由**：vz-memory 的 SSOT 是 `MemoryStratum + tags + structured content`。
为单一消费者新增跨 package schema 违反 R8（memory owner 之外的东西不该迫使
memory 改 schema）。v0 接受 "typed schema in untyped field" 的债务；如果
rupture-repair 变成承重能力，**post-v0 迁移路径**是在 vz-memory 引入一个
`RuptureRepairHistory` **derived readout**（不是 `MemoryEntryKind` enum），
从现存 tag 反查回填，消费者切过去后再弃用 tag。迁移文档在
`docs/specs/continuum-memory.md`。

## Non-goals

- 任何情绪分类器（本 spec 是 evidence-bucket，不是 sentiment）；
- 任何从 user text 的关键词匹配；
- 在 v0 把 `rupture_state` 提升到 ACTIVE（属于 post-v0 M7）；
- 在 v0 启用 LLM proposal（属于 post-v0 M9）；
- 跨用户自动学习（属于 post-v0 M12，需要人工 review）；
- 把 human review 变成 reward（永远禁止）。

## Closed Alpha: Expression Advisory

Closed alpha 允许在**内部脚本 / demo / tests** 中让 `rupture_state`
影响下一轮表达，但只能通过一次性的 typed advisory：

1. `RuptureStateModule` 仍然是 rupture 的唯一 owner，`rupture_state`
   仍默认 SHADOW；
2. runtime 编排层可在 propagate 后读取 `rupture_state`，派生
   `RepairExpressionAdvisory` 并放入 `ResponseContext`；
3. expression planner 只能消费该 advisory，不能 import
   `RuptureStateSnapshot`、不能读取 SHADOW snapshot、不能从 raw user text
   重新判断 rupture、不能持久化 repair 状态；
4. durable rupture-repair memory 仍只能由 `ReflectionEngine.apply(...)`
   写入。

这个路径是 R15 的可回滚 alpha 开关：默认关闭，只在明确启用的
internal companion demo / tests 中用于验证关系修复表达是否可见。
alpha gate 必须包含 matched control：同一 typed external rupture 在
advisory 表达开关关闭时仍可被 `rupture_state` 观察，但不得产生
`repair_alpha=<kind>` 表达 rationale。
`lifeform_evolution.relationship_repair_alpha_gate` 是当前内部 gate
入口，会产出 treatment/control 结构化 report。treatment 还必须在
repair expression 后通过 typed positive outcome 形成至少一条
`repair_outcome:observed` durable rupture-repair memory；control 不得产生
observed repair memory。treatment 还必须证明同一 user scope 的新 session
可读取该 durable memory，且不同 user scope 读取不到。

## 变更日志

- 2026-05-05: 初稿（M0）。
