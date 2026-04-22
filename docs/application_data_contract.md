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

---

## 3. Slot Map

| Slot | Owner | Primary timescale | Purpose |
|------|-------|-------------------|---------|
| `retrieval_policy` | `RetrievalPolicyModule` | `online-fast` | ETA/Regime/DualTrack 对检索层的统一控制接口 |
| `domain_knowledge` | `DomainKnowledgeModule` | `online-fast` read, `session-medium ~ background-slow` refresh | 公共专业事实与来源证据 |
| `case_memory` | `CaseMemoryModule` | `session-medium` | 案例经验样本 |
| `strategy_playbook` | `StrategyPlaybookModule` | `background-slow` produce, `online-fast` consume | 可迁移策略先验 |
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
    description: str


@dataclass(frozen=True)
class CaseMemorySnapshot:
    retrieval_policy_id: str
    hits: tuple[CaseEpisodeHit, ...]
    active_problem_patterns: tuple[str, ...]
    active_risk_markers: tuple[str, ...]
    description: str
```

## 7.2 Owner Responsibilities

`CaseMemoryModule` 负责：

- 发布“类似事情过去怎么发生、怎么处理、结果如何”的案例样本
- 保持案例经验与普通 `memory` 分离
- 不直接生成策略先验结论

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
    description: str


@dataclass(frozen=True)
class StrategyPlaybookSnapshot:
    matched_problem_patterns: tuple[str, ...]
    matched_rules: tuple[PlaybookRule, ...]
    description: str
```

## 8.2 Owner Responsibilities

`StrategyPlaybookModule` 负责：

- 发布从案例经验中沉淀出的可迁移策略先验
- 不直接重写 `temporal` 或 `regime` 内部状态
- 作为 ETA 在线控制层的外部经验参考面

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
    description: str


@dataclass(frozen=True)
class ExperienceConsolidationSnapshot:
    source_session_post_job_id: str
    promoted_case_count: int
    playbook_delta_count: int
    boundary_delta_count: int
    deltas: tuple[ExperienceDelta, ...]
    description: str
```

## 10.2 Owner Responsibilities

`ExperienceConsolidationModule` 负责：

- 把 session-post slow loop 的学习结果公开出来
- 说明本轮背景慢反思到底学到什么
- 不直接取代 `session_post_slow_loop`
- 不直接越权 apply 到其他 owner 内部状态

## 10.3 Direct Dependencies

该 slot 不进入 turn-time propagate 主链。  
它由 `session_post_slow_loop` 驱动，属于 `background-slow` report surface。

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

后续扩展输入：

- `case_memory`
- `strategy_playbook`
- `experience_consolidation`（只做 reporting / rationale enrichment，不进每轮主生成也可）

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

3. `experience_transfer`
   - case hit usefulness
   - playbook matched
   - playbook retained after slow consolidation
   - experience delta promotion count
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

### Phase 2 Added

- `case_memory`

### Phase 3 Added

- `strategy_playbook`
- `experience_consolidation`

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

