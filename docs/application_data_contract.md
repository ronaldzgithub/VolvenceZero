# VolvenceZero Application Data Contract

> Status: draft
> Last updated: 2026-04-22
> Scope: application-layer contracts for knowledge, experience, boundary, and retrieval orchestration
> Source: `docs/application_system_design.md`, `docs/DATA_CONTRACT.md`, `docs/specs/contract-runtime.md`

---

## 1. Purpose

本文档定义应用层新增能力的 **公共数据契约**。它回答三个问题：

1. 新增哪些 slot / owner。
2. 它们各自发布什么 machine-readable state。
3. ETA 在线控制层如何通过统一接口驱动知识与经验检索。

本文档故意**不绑定具体存储技术**。  
Postgres、pgvector、对象存储或其他实现都属于 owner 的内部实现细节，不属于 runtime 公共契约。

---

## 2. Design Invariants

应用层 contract 必须继承现有 runtime 约束：

- 每个 slot 有唯一 owner。
- 公共交换只通过不可变 snapshot。
- 知识、经验、边界、策略先验不能混进 `memory` 一个大字段。
- response 层只消费公共快照，不直连私有数据结构。
- `session_post_slow_loop` 继续作为 `background-slow` 的正式执行面。
- evaluation 只能读取这些 surface 形成 evidence，不能越权成为知识或经验 owner。

### 2.1 How Experience Enters ETA

经验可以进入 ETA，但必须通过**公共 contract**进入，而不是直接并入 `temporal` owner 的私有状态。

应用层要求 experience 通过 4 个正式接入点进入 ETA：

1. **retrieval mix**
   - ETA 通过 `retrieval_policy` 发布 `experience_domains` 与 `experience_weight`
   - 经验因此进入 ETA 的 turn-time 检索混合控制

2. **fast-path priors**
   - `case_memory` 与 `strategy_playbook` 作为 ETA 可读的公共经验先验面
   - 它们影响 ETA 如何排序候选处理方式，但不直接重写 ETA 内部状态

3. **delayed credit**
   - `experience_consolidation` 必须回看 `(abstract_action, regime, retrieval mix, action_family_version)` 的多轮结果
   - 这让经验进入 ETA 的慢层信用闭环，而不只停留在“当前轮命中了什么”

4. **evolution gating**
   - replay / benchmark / evolution judge 应裁决经验产物的 `promote / hold / rollback`
   - 经验不只被 ETA 读取，还应约束 ETA 及其外围 application prior 如何演化

因此，不变量是：

- ETA 可以消费经验
- ETA 不拥有经验本体
- 经验 owner 不得成为 `temporal` 的第二 owner
- experience -> ETA 的所有影响都应通过 public snapshot 或正式 gate 暴露

同样地，knowledge -> ETA 也必须遵守相同边界：

- ETA 可以发布知识检索控制
- ETA 不拥有知识事实本体
- `domain_knowledge` 不得因为 turn-time usefulness 而回收 `temporal` / `memory` owner 身份
- knowledge 对 ETA 的影响必须经 `retrieval_policy`、`domain_knowledge`、`boundary_policy` 这些公共 surface 暴露

---

## 3. Slot Map

| Slot | Owner | Primary timescale | Purpose |
|------|-------|-------------------|---------|
| `retrieval_policy` | `RetrievalPolicyModule` | `online-fast` | ETA/Regime/DualTrack 对检索层的统一控制接口 |
| `domain_knowledge` | `DomainKnowledgeModule` | `online-fast` read, `session-medium ~ background-slow` refresh | 公共专业事实与来源证据 |
| `case_memory` | `CaseMemoryModule` | `session-medium` | 案例经验样本 |
| `strategy_playbook` | `StrategyPlaybookModule` | `background-slow` produce, `online-fast` consume | 可迁移策略先验 |
| `experience_fast_prior` | `ExperienceFastPriorModule` | `background-slow` produce, `online-fast` consume | 把 delayed credit 压缩成 regime/retrieval/action-family 可消费的快路径先验 |
| `boundary_policy` | `BoundaryPolicyModule` | `online-fast` | 风险边界、回答深度、降级/转介 |
| `experience_consolidation` | `ExperienceConsolidationModule` | `background-slow` | 慢反思后学到什么的公共工件 |

---

## 4. Shared Enums And Primitive Types

```python
from dataclasses import dataclass
from enum import Enum


class KnowledgeDepth(str, Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    DEEP = "deep"


class EvidenceStrength(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProfessionalScope(str, Enum):
    GENERAL_SUPPORT = "general-support"
    DOMAIN_INFORMATION = "domain-information"
    DOMAIN_DECISION_SUPPORT = "domain-decision-support"
    PROFESSIONAL_HANDOFF = "professional-handoff"


class KnowledgeSourceType(str, Enum):
    LAW = "law"
    OFFICIAL_GUIDE = "official-guide"
    INTERNAL_GUIDE = "internal-guide"
    REVIEWED_ARTICLE = "reviewed-article"
    PLAYBOOK = "playbook"


class ExperienceOutcomeLabel(str, Enum):
    IMPROVED = "improved"
    STABLE = "stable"
    WORSENED = "worsened"
    UNKNOWN = "unknown"
```

---

## 5. Retrieval Policy Contract

## 5.1 Why This Slot Exists

知识与经验系统不直接“自己判断该搜什么”。  
控制权来自 ETA 在线控制层，再由 `regime`、`dual_track`、`boundary_policy` 共同约束。

因此需要一个单独的 `retrieval_policy` slot，作为：

- 控制层 -> 检索层 的统一接口
- Phase 1 的最小应用编排入口
- 未来知识/经验混合策略的唯一公共 read surface

## 5.2 Snapshot Schema

```python
@dataclass(frozen=True)
class RetrievalPolicySnapshot:
    knowledge_domains: tuple[str, ...]
    experience_domains: tuple[str, ...]
    knowledge_weight: float
    experience_weight: float
    world_weight: float
    self_weight: float
    retrieval_depth: KnowledgeDepth
    citation_required: bool
    jurisdiction_required: bool
    risk_band: RiskBand
    regime_id: str | None
    abstract_action: str | None
    intent_description: str
    description: str
```

## 5.3 Owner Responsibilities

`RetrievalPolicyModule` 负责：

- 读取 `world_temporal`, `self_temporal`, `dual_track`, `regime`
- 输出本轮检索策略
- 不直接执行检索
- 不拥有知识事实或案例经验本身
- 默认应优先消费已发布的控制状态与语义状态（如 `abstract_action`, track weights, regime），而不是靠关键词匹配直接路由 domain
- 作为 experience 进入 ETA 的第一入口：决定是否需要经验检索、需要哪些 `experience_domains`、以及 `experience_weight`
- 作为 **compact retrieval control readout surface**：它应把 ETA / regime / dual-track / memory continuum / delayed fast prior 压成最小控制读出，而不是把应用层知识/经验本体并进自身私有状态

## 5.4 Direct Dependencies

- `world_temporal`
- `self_temporal`
- `dual_track`
- `regime`

`boundary_policy` 对 retrieval 的影响属于第二阶段可选闭环；Phase 1 不要求 `retrieval_policy` 反向依赖 `boundary_policy`，避免主链循环依赖。

---

## 6. Domain Knowledge Contract

## 6.1 Public Shape

```python
@dataclass(frozen=True)
class KnowledgeCitation:
    citation_id: str
    source_type: KnowledgeSourceType
    title: str
    locator: str
    snippet: str
    url: str | None


@dataclass(frozen=True)
class KnowledgeHit:
    hit_id: str
    domain: str
    topic_tags: tuple[str, ...]
    jurisdiction_tags: tuple[str, ...]
    freshness_label: str
    confidence: float
    evidence_strength: EvidenceStrength
    summary: str
    conflict_markers: tuple[str, ...]
    citations: tuple[KnowledgeCitation, ...]
    description: str


@dataclass(frozen=True)
class DomainKnowledgeSnapshot:
    retrieval_policy_id: str
    active_domains: tuple[str, ...]
    hits: tuple[KnowledgeHit, ...]
    citation_required: bool
    jurisdiction_required: bool
    unresolved_conflicts: tuple[str, ...]
    description: str
```

## 6.2 Owner Responsibilities

`DomainKnowledgeModule` 负责：

- 根据 `retrieval_policy` 拉取公共专业事实
- 输出 compact hits，而不是原始文档全文
- 公开来源、地域、冲突、时效标签
- 不输出“应该如何安抚用户”这类经验判断

## 6.3 Direct Dependencies

- `retrieval_policy`
- `memory`
- `dual_track`
- `regime`

其中：

- `memory` 只提供个体上下文和 query facets
- `dual_track` / `regime` 提供这轮检索排序 prior

---

## 7. Case Memory Contract

## 7.1 Public Shape

```python
@dataclass(frozen=True)
class CaseInterventionStep:
    step_id: str
    step_order: int
    regime_id: str | None
    abstract_action: str | None
    action_label: str
    description: str


@dataclass(frozen=True)
class CaseOutcomeSummary:
    outcome_label: ExperienceOutcomeLabel
    delayed_signal_count: int
    escalation_observed: bool
    repair_observed: bool
    confidence: float
    description: str


@dataclass(frozen=True)
class CaseEpisodeHit:
    case_id: str
    domain: str
    problem_pattern: str
    user_state_pattern: str
    risk_markers: tuple[str, ...]
    track_tags: tuple[str, ...]
    regime_tags: tuple[str, ...]
    intervention_steps: tuple[CaseInterventionStep, ...]
    outcome: CaseOutcomeSummary
    relevance_score: float
    continuum_location: ContinuumLocation | None
    description: str


@dataclass(frozen=True)
class CaseMemorySnapshot:
    retrieval_policy_id: str
    hits: tuple[CaseEpisodeHit, ...]
    active_problem_patterns: tuple[str, ...]
    active_risk_markers: tuple[str, ...]
    continuum_profile_id: str | None
    active_band_ids: tuple[str, ...]
    mean_continuum_position: float
    description: str
```

## 7.2 Owner Responsibilities

`CaseMemoryModule` 负责：

- 发布“类似事情过去怎么发生、怎么处理、结果如何”的案例样本
- 保持案例经验与普通 `memory` 分离
- 不直接生成策略先验结论
- 作为 ETA fast-path 可读的经验样本面，而不是直接写入 `temporal`
- 发布的 `continuum_location` 不只是证据字段，还应用于后续 playbook ranking
- 共享 core memory 发布的 continuum frequency 语义，但不回收 `memory` owner 身份
- 其 compact sample surface 可以影响 ETA，但不能因为被 ETA 消费而变成 `temporal` / `regime` 的第二 owner

## 7.3 Direct Dependencies

- `retrieval_policy`
- `memory`
- `dual_track`
- `prediction_error`（Phase 2 attach）

Phase 1 中该 slot 可不存在；Phase 2 才引入。

---

## 8. Strategy Playbook Contract

## 8.1 Public Shape

```python
@dataclass(frozen=True)
class PlaybookRule:
    rule_id: str
    problem_pattern: str
    recommended_regime: str | None
    recommended_ordering: tuple[str, ...]
    recommended_pacing: str
    avoid_patterns: tuple[str, ...]
    knowledge_weight_hint: float
    experience_weight_hint: float
    applicability_scope: tuple[str, ...]
    confidence: float
    continuum_band_id: str | None
    mean_continuum_position: float
    description: str


@dataclass(frozen=True)
class StrategyPlaybookSnapshot:
    matched_problem_patterns: tuple[str, ...]
    matched_rules: tuple[PlaybookRule, ...]
    continuum_profile_id: str | None
    active_band_ids: tuple[str, ...]
    description: str
```

## 8.2 Owner Responsibilities

`StrategyPlaybookModule` 负责：

- 发布从案例经验中沉淀出的可迁移策略先验
- 不直接重写 `temporal` 或 `regime` 内部状态
- 作为 ETA 在线控制层的外部经验参考面
- 作为 experience 进入 ETA 的第二入口：为 ETA 提供 ordering / pacing / avoid-pattern prior
- 在当前口径下，rule ranking 应消费 case hit 的 continuum 位置、band 角色与恢复来源，而不是只按离散 pattern 先后顺序
- playbook 是 compact abstraction owner，不是 session glue 中可随处扩写的辅助字典

## 8.3 Direct Dependencies

- `case_memory`
- `regime`
- `dual_track`

Phase 3 才引入。

---

## 9. Boundary Policy Contract

## 9.1 Public Shape

```python
@dataclass(frozen=True)
class BoundaryDecision:
    decision_id: str
    risk_band: RiskBand
    professional_scope: ProfessionalScope
    answer_depth_limit: str
    citation_required: bool
    clarification_required: bool
    refer_out_required: bool
    blocked_topics: tuple[str, ...]
    required_disclaimers: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class BoundaryPolicySnapshot:
    active_decision: BoundaryDecision
    trigger_reasons: tuple[str, ...]
    description: str
```

## 9.2 Owner Responsibilities

`BoundaryPolicyModule` 负责：

- 把风险、专业边界、回答深度限制变成公共状态
- 为 response assembly 和 evaluation 提供统一可读面
- 不自己生成回答文本
- 不自己做经验沉淀

## 9.3 Direct Dependencies

- `retrieval_policy`
- `domain_knowledge`
- `regime`
- `prediction_error`

Phase 1 即引入。

---

## 9.5 Experience Fast Prior Contract

## 9.5.1 Public Shape

```python
@dataclass(frozen=True)
class ExperienceFastPriorSnapshot:
    regime_biases: tuple[ExperienceFastPriorRegimeBias, ...]
    knowledge_weight_bias: float
    experience_weight_bias: float
    action_biases: tuple[ExperienceFastPriorActionBias, ...]
    family_biases: tuple[ExperienceFastPriorFamilyBias, ...]
    sequence_biases: tuple[ExperienceFastPriorSequenceBias, ...]
    prior_strength: float
    source_attribution_ids: tuple[str, ...]
    source_sequence_ids: tuple[str, ...]
    description: str
```

## 9.5.2 Owner Responsibilities

`ExperienceFastPriorModule` 负责：

- 读取 `experience_consolidation` 的 delayed outcome ledger 与 sequence payoffs
- 把慢层信用压缩成 compact fast prior，而不是直接修改 `regime` / `retrieval_policy` / `temporal` 私有状态
- 作为 experience 进入 ETA 的 delayed-credit-to-fast-path 公共中继面
- 保持 advisory / bias 语义，不回收下游 owner 身份
- 它应继续作为 slow -> fast 的**唯一公共压缩中继面**；若需要更 learned 的 readout，应替换在该压缩或消费 seam 内，而不是额外新增旁路 signal owner

## 9.5.3 Direct Dependencies

- `experience_consolidation`

---

## 10. Experience Consolidation Contract

## 10.1 Public Shape

```python
@dataclass(frozen=True)
class ExperienceDelta:
    delta_id: str
    delta_type: str
    target_slot: str
    summary: str
    confidence: float
    blocked: bool
    continuum_band_id: str | None
    continuum_position: float
    description: str


@dataclass(frozen=True)
class ApplicationOutcomeAttribution:
    attribution_id: str
    source_context_session_id: str
    source_wave_id: str
    regime_id: str | None
    abstract_action: str | None
    action_family_version: int
    retrieval_policy_id: str | None
    knowledge_weight: float
    experience_weight: float
    retrieval_mix_alignment: float
    regime_alignment: float
    abstract_action_alignment: float
    outcome_score: float
    resolved_turn_index: int
    continuum_profile_id: str | None
    dominant_band_id: str | None
    mean_continuum_position: float
    continuum_alignment: float
    description: str


@dataclass(frozen=True)
class ApplicationSequencePayoff:
    sequence_id: str
    regime_sequence: tuple[str, ...]
    action_family_version: int
    sample_count: int
    rolling_payoff: float
    latest_outcome: float
    continuum_profile_id: str | None
    dominant_band_id: str | None
    mean_continuum_position: float
    description: str


@dataclass(frozen=True)
class BoundaryPriorHint:
    hint_id: str
    regime_id: str | None
    trigger_reasons: tuple[str, ...]
    answer_depth_limit_hint: str
    clarification_required: bool
    refer_out_required: bool
    blocked_topics: tuple[str, ...]
    required_disclaimers: tuple[str, ...]
    confidence: float
    description: str


@dataclass(frozen=True)
class CaseMemoryPriorUpdate:
    update_id: str
    target: str
    record: CaseMemoryRecord
    confidence: float
    description: str


@dataclass(frozen=True)
class StrategyPlaybookPriorUpdate:
    update_id: str
    target: str
    rule: PlaybookRule
    confidence: float
    description: str


@dataclass(frozen=True)
class BoundaryPolicyPriorUpdate:
    update_id: str
    target: str
    hint: BoundaryPriorHint
    confidence: float
    description: str


@dataclass(frozen=True)
class ApplicationPriorUpdate:
    source_session_post_job_id: str
    case_memory_updates: tuple[CaseMemoryPriorUpdate, ...]
    strategy_playbook_updates: tuple[StrategyPlaybookPriorUpdate, ...]
    boundary_policy_updates: tuple[BoundaryPolicyPriorUpdate, ...]
    description: str


@dataclass(frozen=True)
class ApplicationPriorWritebackReport:
    proposed_target_count: int
    applied_targets: tuple[str, ...]
    blocked_targets: tuple[str, ...]
    audit_record_count: int
    description: str


@dataclass(frozen=True)
class ExperienceConsolidationSnapshot:
    source_session_post_job_id: str
    promoted_case_count: int
    playbook_delta_count: int
    boundary_delta_count: int
    deltas: tuple[ExperienceDelta, ...]
    delayed_outcome_ledger: tuple[ApplicationOutcomeAttribution, ...]
    sequence_payoffs: tuple[ApplicationSequencePayoff, ...]
    latest_prior_update: ApplicationPriorUpdate | None
    latest_writeback_report: ApplicationPriorWritebackReport | None
    continuum_profile_id: str | None
    active_band_ids: tuple[str, ...]
    description: str
```

## 10.2 Owner Responsibilities

`ExperienceConsolidationModule` 负责：

- 把 session-post slow loop 的学习结果公开出来
- 说明本轮背景慢反思到底学到什么
- 不直接取代 `session_post_slow_loop`
- 发布 typed `ApplicationPriorUpdate` / writeback report，作为 application prior 的正式 slow-path 更新契约
- 不自己成为 `case_memory` / `strategy_playbook` / `boundary_policy` 的第二 owner
- 为 application 层公开 delayed outcome attribution：至少覆盖 `regime`, `abstract_action`, retrieval mix, `action_family_version` 与 sequence payoff
- 作为 experience 进入 ETA 的第三入口：把经验变成 ETA 可读的 delayed credit surface，并在 judge / credit gate 放行时驱动 owner-side prior update

## 10.3 Direct Dependencies

该 slot 不进入 turn-time propagate 主链。  
它由 `session_post_slow_loop` 驱动，属于 `background-slow` owner-side enrichment surface。

apply 语义要求：

- `experience_consolidation` 公开 proposal / applied / blocked / audit-ready 摘要
- 真正的 apply 只能发生在 session owner 驱动的 owner-side writeback helper 中
- `EvolutionJudgement` 与 target-specific credit gate 必须先裁决，再允许 application prior update 进入 owner
- rollback 必须沿 application owner checkpoint / rare-heavy rollback 链回滚，而不是由 `ExperienceConsolidationModule` 直接反写其他 owner

上游来源：

- `reflection`
- `case_memory`
- `prediction_error`
- delayed outcome evidence

---

## 11. Application Rare-Heavy Contract

Phase 4 不直接引入新的 turn-time slot，而是为应用层增加一个 **rare-heavy checkpoint/state contract**，让知识域偏置、case clusters 和 distilled playbook 沿现有 artifact/import/rollback 链进入系统。

```python
@dataclass(frozen=True)
class ApplicationCaseCluster:
    cluster_id: str
    problem_pattern: str
    exemplar_count: int
    mean_relevance: float
    risk_markers: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ApplicationRareHeavyCheckpoint:
    checkpoint_id: str
    domain_template_biases: tuple[tuple[str, float], ...]
    case_clusters: tuple[ApplicationCaseCluster, ...]
    distilled_playbook_rules: tuple[PlaybookRule, ...]
    description: str
```

**不变量**：
- checkpoint 属于 session owner 管辖的 application rare-heavy state
- `retrieval_policy` 只能读取 domain bias 的公共效果，不能直接接管 checkpoint 本体
- `case_memory` 只能读取 cluster fallback / enrichment，不得因此回收 `memory` owner 身份
- `strategy_playbook` 只能消费 distilled rules 作为 prior，不得反向成为 `temporal` 的第二 owner
- rollback 必须沿 rare-heavy rollback 链回滚 application state

---

## 12. Response Assembly Inputs

应用层 response assembly 只消费公共快照，不消费 owner 私有存储。

最小新增输入：

- `retrieval_policy`
- `domain_knowledge`
- `boundary_policy`
- `regime`
- `temporal_abstraction`
- `memory`
- `reflection`

后续扩展输入：

- `case_memory`
- `strategy_playbook`
- `experience_consolidation`（主要承担 delayed credit、writeback report 与 updater readout；不要求进每轮主生成）

其中：

- `case_memory` / `strategy_playbook` 对应 ETA 的 fast-path experience priors
- `experience_fast_prior` 对应 ETA 的 delayed-credit-to-fast-policy bridge
- `experience_consolidation` 对应 ETA 的 slow-path delayed credit / evolution evidence
- `response_assembly` 的 ordering plan 不应只复制 `playbook_ordering`
- 它应显式综合 `boundary_policy` 与 continuum target position，决定首步是 `stabilize`、`clarify_goal` 还是 `structure_options`

### 12.1 ResponseAssemblySnapshot

```python
@dataclass(frozen=True)
class ResponseAssemblySnapshot:
    regime_id: str | None
    regime_name: str
    abstract_action: str | None
    response_mode: str
    answer_depth_limit: str
    citation_mode: str
    clarification_required: bool
    refer_out_required: bool
    ordering_plan: tuple[str, ...]
    knowledge_briefs: tuple[str, ...]
    case_briefs: tuple[str, ...]
    playbook_ordering: tuple[str, ...]
    required_disclaimers: tuple[str, ...]
    required_disclaimer_phrases: tuple[str, ...]
    control_code: tuple[float, ...]
    control_scale: float
    max_questions: int
    prompt_residue_summary: str
    prompt_residue_ratio: float
    continuum_target_position: float
    ordering_driver: str
    description: str
```

### 12.2 Generation Constraints boundary

表达层和 runtime 应优先消费 `ResponseAssemblySnapshot` 中可直接执行的字段：

- `response_mode`
- `answer_depth_limit`
- `citation_mode`
- `max_questions`
- `required_disclaimer_phrases`
- `ordering_plan`
- `continuum_target_position`
- `control_code`
- `control_scale`

只有无法参数化的残余语义才进入 `prompt_residue_summary`。

`ordering_driver` 应说明 ordering 主导来源，例如：

- `continuum-support-first`
- `continuum-clarify-first`
- `continuum-structure-first`
- `continuum-support-clarify`

表达层 generation constraints 在当前口径下还应把 `continuum_target_position` 映射成显式 decoding profile，例如：

- `support-first`
- `clarify-first`
- `structure-first`

也就是说，generation-side control 不再只是连续调 `temperature/max_new_tokens`，  
而是先选中一个 decoding profile 模板，再在模板内部做连续微调。

---

## 13. Evaluation Evidence Contract

evaluation 不新增 owner 身份，只追加证据。

建议新增三类 evidence families：

1. `knowledge_fit`
   - knowledge hit count
   - citation compliance
   - jurisdiction match
   - unresolved conflict exposure

2. `boundary_correctness`
   - clarification triggered
   - referral triggered
   - blocked-topic enforcement
   - answer-depth compliance

3. `response_assembly_control`
   - response depth compliance
   - clarification compliance
   - refer-out compliance
   - ordering-plan alignment
   - prompt residue ratio

4. `experience_transfer`
   - case hit usefulness
   - playbook matched
   - playbook retained after slow consolidation
   - experience delta promotion count
   - delayed fast prior availability
   - delayed regime bias applied
   - delayed retrieval mix bias applied
   - delayed action family bias applied
   - delayed retrieval mix alignment
   - delayed regime alignment
   - delayed abstract-action alignment
   - regime sequence payoff
   - application rare-heavy import retained in fast path

这些 evidence 属于 final wiring enrichment 或 session-post enrichment，  
不改变 `evaluation` slot 的基础公共 shape。

---

## 14. Storage-Abstraction Boundary

本契约刻意不定义：

- 使用什么数据库
- embedding 维度
- 向量索引实现
- 文档切块策略
- offline clustering 技术选型

这些全部属于 owner 内部实现。

runtime 层只关心：

- 发布值的 schema
- 依赖哪些 upstream
- 哪些 timescale 负责更新
- 哪些 surface 是 public，哪些是 owner-only

---

## 15. Phase Mapping

### Phase 1 Required

- `retrieval_policy`
- `domain_knowledge`
- `boundary_policy`

默认 runtime 口径要求上述三者进入 turn-time active chain；其余 application surface 只有在显式 widen phase 时才进入 active application 输出。

### Phase 2 Added

- `case_memory`

### Phase 3 Added

- `strategy_playbook`
- `experience_consolidation`
- `experience_fast_prior`

### Phase 4 Added

- `ApplicationRareHeavyCheckpoint`
- application rare-heavy import / rollback path

---

## 16. Acceptance Checklist

引入这些应用层 contract 时，必须满足：

- 每个 slot 有唯一 owner
- 所有新增 snapshot 都是 frozen dataclass
- 直接依赖与 enrichment 边界明确
- `memory` 不吸收 `domain_knowledge` / `case_memory`
- `session_post_slow_loop` 继续 bounded、observable、fail-closed
- response 层只读公共快照
- evaluation 只做 readout / evidence，不抢 owner 身份
- application rare-heavy refresh 只通过 session owner import / rollback，不创建新的 second owner

