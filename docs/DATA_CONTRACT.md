# EmoGPT Next-Gen — 数据契约文档

> Status: draft
> Version: 0.4
> Last updated: 2026-04-29
> Source: `docs/next_gen_emogpt.md`（R8, R11）、`docs/SYSTEM_DESIGN.md`、`SPLIT.md`

---

## 1. 契约总则

本文档定义系统中所有模块间交换的数据结构、快照格式和接口契约。

**铁律**（源自 R8）：

1. **快照是模块间唯一数据通道**：模块 A 需要模块 B 的数据 → 读取 B 发布的不可变快照，禁止直接调用 B 的方法
2. **谁拥有数据，谁负责描述**：模块内部状态的总结/描述由模块自身生成并打包到快照中发布，消费者直接使用
3. **快照不可变**：所有快照和 value 必须是不可变对象（frozen dataclass）

**禁止**：
- `copy.deepcopy()` — 用 `dataclasses.replace()` 实现结构共享
- 返回内部可变对象引用
- 原地修改快照
- 消费者重建生产者内部状态

### 1.1 契约表面的 wheel 边界

数据契约同时承担**模块边界**（R8）与**仓库边界**（R15）两条作用：

| 层 | wheel | 契约作用 |
|----|-------|----------|
| 内核 contracts | `vz-contracts` | 所有 vz-* / lifeform-* 共享的 Snapshot / RuntimeModule / Guards 类型；零产品知识 |
| 内核 owner snapshot | `vz-substrate` / `vz-temporal` / `vz-memory` / `vz-cognition` / `vz-application` / `vz-runtime` | §3 列出的运行时 slot |
| 生命体侧契约 | `lifeform-core` | `VitalsBootstrap` / `VitalsSnapshot` / `DriveSpec` / `DriveLevel` / `TurnSummary` 等；不进入内核运行时 slot |
| 垂直经验 | `lifeform-domain-*` | `DomainExperiencePackage` / `VitalsBootstrap`；编译进既有内核 application owner |

当前 wheel 命名以代码为准：`prediction_error`、`credit`、`dual_track`、`regime`、`semantic_state`、`evaluation` 和 social cognition owner 均由 `vz-cognition` 承载。`vz-pe-credit` / `vz-self-model` / `vz-evaluation` 只可作为历史能力域简称，不是当前 package 名。

**关键不变量**：

- `vz-*` 不得 import `lifeform-*`；CI 由 `tests/contracts/test_import_boundaries.py` 强制
- vertical 不引入新的 runtime owner，只通过 `volvence_zero.application.domain_experience` 编译进 `domain_knowledge` / `case_memory` / `strategy_playbook` / `boundary_policy` / `application rare-heavy state`
- vitals layer 的 `VitalsSnapshot` 是 lifeform-side 公共契约，由 `VitalsModule` 唯一拥有；**不**作为内核 runtime slot 出现在 §6 注册表
- `Brain` / `BrainSession` 是内核暴露给 lifeform 层的 stable facade，详见 `docs/specs/core-package-boundary.md`

详见 `SPLIT.md` 与 `archetecture.md`。

### 1.2 Figure cleaning vertical schema (non-runtime)

`packages/lifeform-domain-figure/src/lifeform_domain_figure/cleaning/` 提供的 schema 是 **figure vertical 内部跨子模块共享**的 typed records，**不**进入 §6 runtime slot 注册表（不是 runtime owner，没有 module / `RuntimeModule` 包装，跨 wheel 也不消费）。它们登记在此处仅为 SSOT 完整性。来源：`docs/known-debts.md` debt #28 L1 packet（2026-05-10）；spec：[`docs/specs/figure-corpus-cleaning.md`](specs/figure-corpus-cleaning.md)。

| Type | 字段（关键） | 用途 |
|------|--------------|------|
| `RawDocument` | `text` / `parser_version` / `layout_quality` / `ocr_confidence` / `encoding_detected` / `language_detected` / `license_notice` / `raw_sha256` | 4 个 parser（CPAE PDF / Wikisource / Gutenberg / IA OCR JSON）的输出；中性，不绑定任何 archive 类型 |
| `CleanedDocument` | `text` / `raw_sha256` / `cleaner_pipeline_version` / `cleaning_log: tuple[CleaningOpRecord, ...]` / `parser_version` | 6-op cleaner pipeline 的输出；同 raw 多版本（v1 / v2 / ...）共存 |
| `CleaningOpRecord` | `op: CleaningOp` / `op_version` / `chars_before` / `chars_after`（`<=` chars_before） | 每个 cleaner op 的执行记录；orchestrator 自动记录，不可手填错乱 |
| `CleaningOp` (enum) | `BOILERPLATE_STRIP` / `WHITESPACE_NORMALIZE` / `TYPOGRAPHY_NORMALIZE` / `DEDUPE_INTRA_DOC` / `PII_REDACT` / `PARAGRAPH_NORMALIZE` | 关闭枚举；新增需同步 spec + 子模块 + bump pipeline 版本 |

**契约不变量**（详见 spec）：

- `cleaning/` 子包**禁止** import `Figure*Source` typed records（必须经 `cleaning/bridging.py` 二段式）— `tests/contracts/test_cleaning_pipeline_versions.py` AST 静态守门
- `cleaning/` 子包**禁止** import 任何 HTTP 客户端（cleaning 与 V2 fetcher 解耦）— 同上守门
- raw bytes content-addressable by sha256；cleaner 版本目录永不覆盖

### 1.3 Figure verification vertical schema (non-runtime)

`packages/lifeform-domain-figure/src/lifeform_domain_figure/verification/` 提供的 schema 是 **figure vertical 内部跨子模块共享**的 typed records，**不**进入 §6 runtime slot 注册表（不是 runtime owner）。来源：`docs/known-debts.md` debt #28 L2 first batch（2026-05-10）；spec：[`docs/specs/figure-corpus-verification.md`](specs/figure-corpus-verification.md)。

| Type | 字段（关键） | 用途 |
|------|--------------|------|
| `CheckKind` (enum, 7 values) | `DATE_PLAUSIBILITY` / `LICENSE_PAGE_LEVEL` / `CROSS_SOURCE_BYTE` / `IDENTITY_DISAMBIGUATION` / `AUTHORSHIP_ATTRIBUTION` / `VERSION_RECONCILIATION` / `TRANSLATION_LINEAGE` | 关闭枚举；前 3 已实现，后 4 deferred 到 #26 metadata client |
| `IMPLEMENTED_CHECK_KINDS: frozenset[CheckKind]` | bundle gate 强制全 PASS 的子集 | 新 verifier 实现时**必须**同步加入；contract test 自动 surface 缺失 |
| `Verdict` (enum, 3 values) | `PASS` / `FAIL` / `NEEDS_REVIEW` | 关闭枚举 |
| `VerificationCheck` | `check_kind` / `verdict` / `evidence: tuple[str, ...]` (非空) / `reviewer_id` (`auto:<id>:<int>` 或 `human:<id>`) / `reviewed_at_iso` / `source_byte_sha256` (64-char hex) | 一条 verifier verdict；不可变；anchor key 与 L1 `RawDocument.raw_sha256` / `SourceProvenance.byte_sha256` 同 hash |

**契约不变量**（详见 spec）：

- `verification/` 子包**禁止** import `Figure*Source` typed records — `tests/contracts/test_verification_module_boundaries.py` AST 静态守门
- `verification/` 子包**禁止** import 任何 HTTP 客户端 — 同上守门
- `verification/` 子包**禁止** import 任何 `volvence_zero.{cognition,temporal,memory,substrate,application,runtime}.*` 内核模块（verifier 是 readout / gate，绝不反向写 kernel；R12 evaluation 单向性）— 同上守门
- `VerificationLedger` append-only；override 通过 append `human:` check 实现，`latest_per_kind` 取每 kind 最新一条作为生效 verdict
- bundle gate 阶段性放行 `IMPLEMENTED_CHECK_KINDS`；新 kind 实现时同步更新

### 1.4 Figure crawl vertical schema (non-runtime)

`packages/lifeform-domain-figure/src/lifeform_domain_figure/crawl/` 提供的 schema 是 **figure vertical 内部跨子模块共享**的 typed records，**不**进入 §6 runtime slot 注册表。来源：`docs/known-debts.md` debt #28 L0 + debt #19 closure（2026-05-10）；spec：[`docs/specs/figure-corpus-crawl.md`](specs/figure-corpus-crawl.md)。

| Type | 字段（关键） | 用途 |
|------|--------------|------|
| `CrawlStatus` (enum, 7 values) | `SUCCESS` / `FETCHED_NOT_MODIFIED` / `SKIPPED_ROBOTS` / `SKIPPED_SCOPE` / `SKIPPED_RATE` / `FAILED_HTTP` / `FAILED_PARSER_PRECHECK` | 关闭枚举 |
| `VALID_FETCH_KINDS` (frozenset[str], 5 values) | `generic` / `cpae` / `wikisource` / `gutenberg` / `internet_archive` | 关闭 fetcher 注册集合；新增需同步 dispatcher 与 spec |
| `CrawlRequest` | `url` / `fetch_kind` / `request_id` (= sha256(fetch_kind + "\n" + url)) / `enqueued_at_iso` / `referrer` / `expected_content_type` | 不可变 work item；request_id 是 dedup key |
| `CrawlResult` | `request` / `status` / `fetched_at_iso` / `raw_sha256` (== L1 anchor when SUCCESS) / `content_type_actual` / `byte_len` / `http_status` / `etag` / `last_modified` / `error` | 不可变 outcome；SUCCESS 必填 raw_sha256，其它必空 |
| `ScopePolicy` | `allowed_hosts: frozenset[str]` (非空) / `user_agent` / `allowed_path_prefixes` / `host_roles: dict[str, frozenset[ScopeRole]]` / `max_pages_per_host` / `max_body_bytes` / `incremental` | SSRF allowlist + 全局 budget；`DEFAULT_HOSTS = DEFAULT_CORPUS_HOSTS \| DEFAULT_METADATA_HOSTS`；`host_roles` 携带每 host 的 `ScopeRole` 标签（debt #26 closure） |
| `ScopeRole` (enum, 2 values) | `CORPUS_FETCH` / `METADATA_FETCH` | 关闭枚举；`BaseHTTPClient.get(..., required_role=...)` 强制 host 必须携带匹配角色 |
| `LiveFetchedBytes` (in `corpus/archives`) | `body` / `raw_sha256` / `content_type` / `http_status` | V2 `ArchiveFetcher.fetch().raw_payload` 形态（debt #19 closure），区别于 V1 的 `*Payload` raw_payload |

**契约不变量**（详见 spec）：

- `crawl/` 子包**允许** import `requests` / `urllib.robotparser` / `urllib.parse`（figure vertical 唯一 HTTP 出口；与 L1 / L2 反转）
- `crawl/` 子包**禁止** import `Figure*Source` typed records — `tests/contracts/test_crawler_module_boundaries.py` AST 静态守门
- `crawl/` 子包**禁止** import 任何 `volvence_zero.{cognition,temporal,memory,substrate,application,runtime}.*` 内核模块 — 同上守门
- `crawl/` 子包**禁止** import `lifeform_domain_figure.verification.*`（crawler 不感知 verifier）— 同上守门
- 5 SSRF gate（scheme + host + path-prefix + redirect-1-hop-rescope + body-size-cap）全部在 `BaseHTTPClient.get` 强制
- robots.txt fail-closed（fetch failure → host 拒收）
- `request_id` / `raw_sha256` / `byte_sha256` / `RawDocument.raw_sha256` 是同字节流的同 hash（content-addressable 三段贯通）
l- `BaseHTTPClient.get(..., required_role=ScopeRole)` 强制角色匹配；L0 fetcher 传 CORPUS_FETCH，metadata client 传 METADATA_FETCH

### 1.5 Figure D4 metadata client schema (debt #26 V2 closure)

`packages/lifeform-domain-figure/src/lifeform_domain_figure/metadata/` V2 live clients 的输出 schema 与 V1 offline 桩**完全相同**（typed `*Payload` dataclasses 不变）；新增的是 V2 wiring 层。来源：`docs/known-debts.md` debt #26 closure（2026-05-10）；spec：[`docs/specs/figure-corpus-crawl.md`](specs/figure-corpus-crawl.md) §"Metadata HTTP backbone"。

| Type | 字段（关键） | 用途 |
|------|--------------|------|
| `MetadataResponse` | `body: bytes` / `content_type` / `fetched_at_iso` / `from_cache: bool` | metadata HTTP 响应包装；区分 fresh fetch vs cache hit |
| `MetadataHTTPClient` | wraps `BaseHTTPClient` 强制 `required_role=ScopeRole.METADATA_FETCH` | metadata clients 共用的 HTTP 出口；统一 SSRF / retry / role |
| `MetadataCache` | content-addressable `data/metadata_cache/{provider}/{key_sha256}/` + TTL | 24h 默认 TTL；TTL=0 关闭过期；`fetch_or_get` 一站式 cache-aware 拉取 |
| 4 live client factories (`live_openalex_client` / `live_wikidata_client` / `live_crossref_client` / `live_sep_client`) | 返回 V2 client，与 V1 offline factory 共享 Protocol 形状 | API 端点：`api.openalex.org/works` (cursor 分页) / `Special:EntityData/{qid}.json` / `api.crossref.org/works/{doi}` / `plato.stanford.edu/entries/{slug}/` (HTML via bs4) |

**契约不变量**：

- 所有 metadata client **禁止** import `Figure*Source` typed records（同 verification 子包）— `tests/contracts/test_verification_module_boundaries.py` AST 守门
- 所有 metadata client 走 `MetadataHTTPClient`，role 必为 METADATA_FETCH（cross-role SSRF 在 `BaseHTTPClient.get` 拒收）
- `MetadataCache` 只读 `provider + key` 对，永不写入 derived state；R12 单向性
- V1 offline factory（`offline_*_client()`）行为不变，向后兼容

### 1.6 Bundle metadata digest fingerprint (debt #25 closure)

`FigureArtifactBundle.metadata_digest_fingerprint: str = ""` 字段（默认空，向后兼容）记录 enrichment 用的 `MetadataDigest.fingerprint`。`compute_bundle_integrity_hash(..., metadata_digest_fingerprint="")` 默认空时**不**折入 hash（既有 bundle 字节级稳定）；非空时**折入**作为 R15 byte-level 回滚契约的最后一环。

**契约不变量**：

- `FigureBundleInputs.metadata_digest=None` (默认) → bundle.metadata_digest_fingerprint 为空 → integrity_hash 与 land 之前**字节级一致**（既有 bundle ID 不变）
- `metadata_digest=<digest>` → fingerprint 进入 hash → 不同 digest 产不同 bundle ID（一份 audit chain）
- `attach_steering_to_bundle` / `attach_lora_to_bundle` 重算 hash 时**保留** metadata_digest_fingerprint（否则 LoRA bake 后会丢失 metadata 审计链）
- 契约测试 [`tests/contracts/test_figure_bundle_metadata_fingerprint.py`](../tests/contracts/test_figure_bundle_metadata_fingerprint.py) 同时守 attach 路径

---

## 2. 基础类型

### 2.1 Snapshot（快照基类）

所有模块发布的快照的基类。

```python
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Snapshot:
    slot_name: str          # 快照 slot 标识，全局唯一
    owner: str              # 发布模块的唯一标识
    version: int            # 单调递增的版本号
    timestamp_ms: int       # 发布时间戳（毫秒）
    value: Any              # 具体快照内容（frozen dataclass）
```

**不变量**：
- `slot_name` 在整个系统中唯一，一个 slot 只有一个 owner
- `version` 每次发布递增，消费者可用于检测变更
- `value` 必须是 frozen dataclass 或不可变类型

### 2.2 Track（轨道标记）

双轨学习的轨道标记（R7）。

```python
from enum import Enum

class Track(Enum):
    WORLD = "world"         # 世界/任务轨道
    SELF = "self"           # 自我/关系轨道
    SHARED = "shared"       # 共享（明确需要时）
```

### 2.3 Timescale（时间尺度）

多时间尺度学习的尺度标记（R1）。

```python
class Timescale(Enum):
    ONLINE_FAST = "online-fast"         # 每轮/每 wave
    SESSION_MEDIUM = "session-medium"   # 每场景/每会话
    BACKGROUND_SLOW = "background-slow" # 会话后反思
    RARE_HEAVY = "rare-heavy"           # 离线重训练
```

### 2.4 ModificationGate（自修改门控级别）

门控自修改的级别标记（R10）。

```python
class ModificationGate(Enum):
    ONLINE = "online"               # 在线可改
    BACKGROUND = "background"       # 需后台验证
    OFFLINE = "offline"             # 需离线重训练
    HUMAN_REVIEW = "human-review"   # 需人工审核
```

### 2.5 WiringLevel（接线级别）

运行时统一的模块接线级别，用于支持“局部完备、默认未全连”的实施模式。

```python
class WiringLevel(Enum):
    DISABLED = "disabled"   # 模块不进主执行链，发布 runtime stub
    SHADOW = "shadow"       # 模块执行但输出不写入 active upstream
    ACTIVE = "active"       # 模块输出写入正式 upstream
```

### 2.6 Environment Event（planned，Phase 0 design freeze）

Environment Event 是 `docs/specs/environment-interface.md` 定义的生命体与环境之间的 canonical event 语义。Phase 0 只冻结字段语义，不承诺新增 Python dataclass 或 kernel slot。

**语义字段**：

- `event_id`
- `event_kind`
- `trigger_kind`
- `actor_id`
- `active_speaker_id`
- `addressee_ids`
- `subject_ids`
- `audience_ids`
- `scene_id`
- `timestamp_ms`
- `provenance`
- `consent_context`
- `payload_summary`

**不变量**：

- Environment Event 不是 kernel owner，也不进入 §6 kernel slot 注册表。
- `lifeform-*` / host / service adapter 负责生产 canonical event / outcome；`vz-*` 只能通过 `Brain` / `BrainSession` facade 与公共 snapshot 消费。
- social cognition owners 消费 Environment Event conversational frame 或其 owner snapshot，不从 renderer / prompt / raw text 重建社会事实。
- tool / affordance / expression outcome 必须能关联到 prior prediction 或 prediction context，再进入 `prediction_error` typed evidence。

### 2.6.1 Emergent Action Abstraction（planned，Phase 1 clean）

`docs/specs/emergent-action-abstraction.md` 冻结的是 ETA/NL-clean 的动作反馈抽象：不新增 `action_outcome_trace` slot，不新增 delayed ledger owner，不新增 action/outcome encoder owner。复杂环境反馈进入现有 `prediction_error` / `temporal_abstraction` / `credit` 主链。

**EnvironmentOutcome 最小观察字段**：

- `latency_ms`
- `monetary_cost`
- `reversibility`
- `environment_state_delta_kind`

**Temporal segment closure 字段**：

- `TemporalAbstractionSnapshot.closed_segments`
- `TemporalSegmentClosure.segment_id`
- `TemporalSegmentClosure.open_turn_index`
- `TemporalSegmentClosure.close_turn_index`
- `TemporalSegmentClosure.abstract_action_id`
- `TemporalSegmentClosure.z_t_digest`
- `TemporalSegmentClosure.beta_open_digest`
- `TemporalSegmentClosure.beta_close_digest`

**Prediction action context 字段**：

- `PredictionActionContext.segment_id`
- `PredictionActionContext.abstract_action_id`
- `PredictionActionContext.z_t_digest`
- `PredictionActionContext.regime_id`
- `PredictionActionContext.affordance_name`
- `PredictionActionContext.environment_event_id`
- `PredictionActionContext.environment_outcome_id`

**不变量**：

- PE owner 仍是唯一 mismatch owner。
- `PredictionActionContext` is injected every turn even when a live session reuses `PredictionErrorModule`; this updates lineage only and does not reset the previous prediction chain.
- `environment_outcome_id` is carried as next-turn lineage evidence from `BrainSession.submit_tool_result(...)`; it does not directly alter PE magnitude/reward formulas.
- `CreditModule` consumes `temporal_abstraction` as a declared dependency and may derive `abstract_action_segment` credit from `PredictionErrorSnapshot.action_context` plus `temporal_abstraction.closed_segments`.
- Credit record context may include `segment_id`, `abstract_action_id`, `environment_event_id`, and `environment_outcome_id` as lineage evidence without changing credit math.
- delayed outcome 边界来自 `beta_t` segment closure，不来自 horizon sweep。
- trust / common-ground / commitment / information gain 不进入 `EnvironmentOutcome`，由各自 owner 的 snapshot delta 表达。
- replay 是 existing snapshots 的 out-of-turn export，不是新的 runtime schema。

### 2.7 Session-Post Slow Loop（会话后慢环）

`background-slow` 的默认运行时形态是 **session-post slow loop**：

- turn 主链只生成 deferred consolidation / writeback request
- context / session boundary 把 request 排进 queue
- queue worker 只调用 owner-side apply surface，不直接篡改 owner 内部状态

```python
@dataclass(frozen=True)
class SessionPostWritebackRequest:
    context_session_id: str
    source_wave_id: str
    session_report: EvaluationReport
    reflection_snapshot: ReflectionSnapshot
    credit_snapshot: CreditSnapshot | None
    evolution_judgement: EvolutionJudgement | None
    cross_session_verdict: str
    writeback_source: str | None
    reflection_apply_enabled: bool
    structural_writeback_allowed: bool
    checkpoint_id: str
    description: str

@dataclass(frozen=True)
class SessionPostSlowLoopJob:
    job_id: str
    context_session_id: str
    closed_at_turn: int
    session_report: EvaluationReport
    prior_session_report_count: int
    trace_count: int
    substrate_batch_count: int
    prediction_error_summary: tuple[tuple[str, float], ...]
    writeback_request: SessionPostWritebackRequest
    description: str

@dataclass(frozen=True)
class SessionPostSlowLoopResult:
    job_id: str
    context_session_id: str
    closed_at_turn: int
    writeback_result: WritebackResult | None
    applied: bool
    blocked: bool
    description: str

@dataclass(frozen=True)
class SessionPostSlowLoopResultSummary:
    job_id: str
    context_session_id: str
    closed_at_turn: int
    applied_operation_count: int
    blocked_operation_count: int
    applied: bool
    blocked: bool
    description: str

@dataclass(frozen=True)
class SessionPostSlowLoopSnapshot:
    queue_state: SessionPostSlowLoopQueueState
    recent_results: tuple[SessionPostSlowLoopResultSummary, ...]
    last_completed_job_id: str | None
    last_completed_context_session_id: str | None
    description: str
```

**不变量**：
- queue 不是新的 memory / temporal / regime owner
- request payload 必须是 immutable 的 machine-readable contract
- apply 仍受 `writeback_mode`、credit gate、evolution judgement 约束
- turn latency 不等待 slow loop 完成
- `session_post_slow_loop` 是独立公共 slot；queue state / 最近完成结果必须通过快照发布，而不是要求消费者读取 `AgentSessionRunner` 私有状态

### 2.7.1 Default Continual Learner Surface（默认持续学习面）

`JointCycleReport` 现在携带一个 runtime-native surface，用于把默认 continual learner 的 owner-side 写回状态作为机器可读证据发布，而不是让 benchmark 从零散操作名里重建语义：

```python
@dataclass(frozen=True)
class DefaultContinualLearningSurface:
    surface_id: str
    active: bool
    owner_path: str
    memory_regime_writeback_applied: bool
    temporal_writeback_applied: bool
    regime_evidence_applied: bool
    substrate_live_mutation_applied: bool
    substrate_review_only: bool
    rare_heavy_review_recommended: bool
    applied_operations: tuple[str, ...]
    blocked_operations: tuple[str, ...]
    rollback_applied: bool
    evolution_decision: str
    evolution_category: str
    description: str
```

**不变量**：
- 默认 continual learner 的正向学习面是 memory / temporal / regime / reflection owner writeback
- `substrate_live_mutation_applied` 在默认路径必须保持 `False`；substrate 更新只能出现在 rare-heavy / experimental lane
- `active` 不代表无约束突变，只代表 owner-side bounded writeback 或 regime evidence 已进入默认学习闭环
- 所有 blocked / rollback 信息必须保留为 public evidence，不能被 benchmark 用缺省值吞掉

### 2.8 Application Retrieval / Knowledge / Boundary（应用层检索与边界）

应用层第一阶段新增三类正式 slot，用于把“ETA 在线控制 -> 专业知识证据 -> 边界约束”提升为公共运行时 surface：

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

@dataclass(frozen=True)
class DomainKnowledgeSnapshot:
    retrieval_policy_id: str
    active_domains: tuple[str, ...]
    hits: tuple[KnowledgeHit, ...]
    citation_required: bool
    jurisdiction_required: bool
    unresolved_conflicts: tuple[str, ...]
    description: str

@dataclass(frozen=True)
class BoundaryPolicySnapshot:
    active_decision: BoundaryDecision
    trigger_reasons: tuple[str, ...]
    description: str
```

**不变量**：
- `retrieval_policy` 是控制层到检索层的唯一主接口
- `domain_knowledge` 只发布 compact 外部事实证据，不越权写 `memory`
- `boundary_policy` 只发布回答边界与降级策略，不直接接管 response owner
- 具体存储技术不属于 runtime contract；owner 只对外发布 machine-readable snapshot

### 2.9 Application Case Memory（应用层案例经验）

应用层第二阶段新增 `case_memory` slot，用于把案例经验样本与普通连续记忆显式分离：

```python
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

**不变量**：
- `case_memory` 是 `memory` 的 sibling owner，不是 `memory` 的附带字段
- `case_memory` 只发布 compact case hits，不发布完整案例原文
- response/evaluation 只能消费公共 snapshot，不得直连 case store
- `case_memory` 当前只提供 retrieval mix 和 evidence，不直接生成策略先验

### 2.10 Application Playbook / Experience Consolidation（应用层策略先验与经验沉淀）

应用层第三阶段新增两个正式 surface：`strategy_playbook` 与 `experience_consolidation`。

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

**不变量**：
- `strategy_playbook` 只发布经验先验，不直接重写 `temporal` / `regime` owner 内部状态
- `experience_consolidation` 是 `background-slow` report surface，由 `session_post_slow_loop` 驱动，而不是新的 apply owner
- `experience_deltas` 必须 machine-readable，可审计，不得退回“只写自然语言总结”
- fast path 可消费 `strategy_playbook` 的 ordering prior，但不得把它提升为第二个控制器 owner

### 2.11 Application Rare-Heavy Checkpoint（应用层离线刷新工件）

应用层第四阶段复用现有 rare-heavy artifact/import/rollback 链，新增一个 application rare-heavy checkpoint：

```python
@dataclass(frozen=True)
class ApplicationRareHeavyCheckpoint:
    checkpoint_id: str
    domain_template_biases: tuple[tuple[str, float], ...]
    case_clusters: tuple[ApplicationCaseCluster, ...]
    distilled_playbook_rules: tuple[PlaybookRule, ...]
    description: str
```

**不变量**：
- checkpoint 本体由 session owner 管理，不成为新的 runtime slot
- 其影响只能通过 `retrieval_policy` / `case_memory` / `strategy_playbook` 的公共快照向外显现
- import / rollback 必须与现有 rare-heavy import / rollback 同步执行
- application rare-heavy refresh 不得直接重写 `memory` / `temporal` / `regime` owner 内部状态

### 2.12 RuntimePlaceholderValue（缺失与禁用占位）

用于统一表示缺失 upstream 和禁用模块发布的 stub 快照。

```python
@dataclass(frozen=True)
class RuntimePlaceholderValue:
    reason: str
    expected_slot: str
    produced_by: str
    detail: str
```

### 2.12.1 Owner Hydration Contract（跨 session owner 续接，Packet D — long-horizon-closure）

**所在 wheel**：`vz-contracts` 提供协议；`vz-cognition` / `lifeform-core` 实现；`vz-runtime` 编排

```python
@dataclass(frozen=True)
class OwnerPersistenceSnapshot:
    owner_name: str            # stable owner identifier
    schema_version: int         # owner-internal
    payload: Mapping[str, Any]  # JSON-serialisable
    description: str = ""

class HydratableOwnerProtocol(Protocol):
    def export_persistence_snapshot(self) -> OwnerPersistenceSnapshot: ...
    def hydrate_from_persistence(
        self, snapshot: OwnerPersistenceSnapshot
    ) -> None: ...

# Typed exceptions (fail-loud, never silent fallback):
class HydrationError(Exception): ...
class HydrationVersionMismatchError(HydrationError): ...
class HydrationPayloadInvalidError(HydrationError): ...
class HydrationOwnerMismatchError(HydrationError): ...
```

**Persistence backend key 前缀**：`owner_hydration/<owner_name>` 写入与 `MemoryStore` 同一 `PersistenceBackend`（`memory/store` 是 MemoryStore 的；`owner_hydration/...` 是这层的，互不冲突）。当前已落地三个 hydratable owner：

- `owner_hydration/semantic_state` — 9 个 SemanticStateStore slot
- `owner_hydration/followup_manager` — FollowupManager 的 pending queue / dedup keys / counter
- `owner_hydration/vitals` — VitalsModule 的 drive levels / 提前提示 / IQR baseline

**关键不变量**：

- 每个 hydratable owner 自己实现 export / hydrate；外部 store 不直写 owner 内部
- `BrainConfig.owner_hydration_wiring: WiringLevel = ACTIVE` 默认（long-horizon-closure follow-up）；`SHADOW` 写不读，`DISABLED` 完全关闭
- `OwnerHydrationStore`（`vz-runtime/owner_hydration_store.py`）是编排器，复用 `MemoryStore.persistence_backend`；只在 `BrainConfig.owner_hydration_wiring != DISABLED` 且 backend 存在时构造（anonymous session 自然 no-op）
- `LifeformSession.end_scene` 自动调用 `persist_owners()`，scene 边界写出 hydration payload
- 跨 user 隔离继承自 `MemoryStore` 的 per-user scope key 路径（`build_scoped_memory_store`）
- 所有 hydration 失败抛 typed `HydrationError` 子类，禁止 bare `except`

详见 [docs/specs/owner-hydration.md](specs/owner-hydration.md)。Acceptance：`tests/contracts/test_owner_hydration_protocol.py` + `tests/contracts/test_owner_hydration_failures_loud.py` + `tests/longitudinal/test_cross_session_owner_hydration.py`。

### 2.13 Lifeform-side Vitals Contract（生命体侧 always-on PE 契约）

**所在 wheel**：`lifeform-core`（不进入内核运行时 slot 注册表）

```python
@dataclass(frozen=True)
class DriveSpec:
    name: str
    target: float                                  # 理想 level [0, 1]
    homeostatic_band: tuple[float, float]          # 舒适带
    decay_per_tick: float                          # SYSTEM tick 衰减
    pe_weight: float                               # 慢尺度 PE 中的权重
    initial_level: float = 0.5
    recharge_per_turn: float = 0.0                 # baseline charge on user turns
    recharge_per_regime: dict[str, float] = ...    # regime 触发的额外 charge

@dataclass(frozen=True)
class DriveLevel:
    name: str
    level: float
    deviation: float
    out_of_band: bool
    pe_contribution: float

@dataclass(frozen=True)
class VitalsBootstrap:
    schema_version: int  # = 1
    drives: tuple[DriveSpec, ...]
    proactive_pe_threshold: float
    proactive_followup_priority: float
    proactive_cooldown_ticks: int

@dataclass(frozen=True)
class VitalsSnapshot:
    schema_version: int  # = 1
    tick_index: int
    drive_levels: tuple[DriveLevel, ...]
    total_pe: float
    above_proactive_threshold: bool
    last_proactive_at_tick: int | None
```

**关键不变量**：

- `VitalsModule` 是 drive level 的唯一 owner；消费者只读 `VitalsSnapshot`
- decay 只在 `TickKind.SYSTEM` 发生；`ENERGY` / `CONTEXT` tick 仅推进 `tick_index`
- `recharge_per_regime` 允许负值（如 `direction_certainty` 在 `guided_exploration` regime 下使用 `-0.05`）；level 在 `[0, 1]` 内 clamp
- `VitalsSnapshot` 不进入内核 §3 / §6 注册表；它是 `lifeform-core` 的 owner snapshot，仅通过 `LifeformSession.vitals_snapshot` 暴露给 `FollowupManager` / `PromptPlanner` / benchmark

详见 `docs/specs/lifeform-vitals.md`。

### 2.14 Lifeform-side DomainExperiencePackage（生命体侧 vertical 经验包契约）

**所在 wheel**：`vz-application`（schema 与 compiler）/ `lifeform-domain-*`（数据）

每个 vertical 通过 `DomainExperiencePackage` 编译进既有内核 application owner，**不**新增 runtime owner：

| package 字段 | 编译目标 owner |
|--------------|----------------|
| `knowledge_records` | `domain_knowledge` (`DomainKnowledgeStore`) |
| `case_records` | `case_memory` (`ApplicationCaseMemoryStore`) |
| `playbook_rules` | `strategy_playbook` (`ApplicationRareHeavyState`) |
| `boundary_hints` | `boundary_policy` (`ApplicationRareHeavyState`) |
| 可选 evaluation scenarios | `lifeform-evolution` benchmark 输入（不进入运行时 slot）|

vertical 同时可附带预训练 `MetacontrollerParameterSnapshot`（β_t / z_t）+ `RegimeBootstrap`（regime selection_weights）作为 magic-byte pickle envelope 跟随 vertical wheel 发布；`build_*_lifeform()` 默认加载，`use_*_bootstrap=False` 用于 ablation。

详见 `docs/specs/domain-experience-layer.md`。

### 2.15 Figure Artifact Bundle（真实人物 vertical 不可变 artifact）

**所在 wheel**：`lifeform-domain-figure`（schema + 编译流水线）

`FigureArtifactBundle` 是 [`lifeform-domain-figure`](../packages/lifeform-domain-figure/) vertical 的**不可变快照**，承载一个真实人物的全部 runtime artifact。它由该 vertical 唯一拥有 (R8)，跨 wheel 只读消费；**不**新增 kernel runtime owner。

| bundle 字段 | 服务的保真层 | 运行时消费者 |
|---|---|---|
| `retrieval_index: FigureRetrievalIndex` | L3 引证保真 | `lifeform-expression.GroundedDecoder` |
| `coverage_map: FigureCoverageMap` | L4 不知拒答 | `lifeform-expression.ScopeRefuser` |
| `style_prior: FigureStylePrior` | L1 语气保真 | `lifeform-expression.StylePriorInjector` |
| `steering: FigureSteeringSet \| None` | L2 立场保真 | `vz-substrate.SubstrateDeltaAdapterLayer`（常量 delta） |
| `lora: FigureLoRAArtifact \| None` | L1 + L2 强化 | `vz-substrate.PersonaLoRAPool` |
| `domain_package: DomainExperiencePackage` | 知识 / 案例 / 策略 / 边界 | 既有 application owner（不新增） |
| `vitals_bootstrap: VitalsBootstrap` | drives | `lifeform-core.VitalsModule`（既有 owner） |
| `integrity_hash: str` | bundle 完整性 | DLaaS / rollback drill |
| `version_window: tuple[int, int]` | 时间分层（早 vs 晚期） | DLaaS template `figure_time_window` 字段 |

**Figure bundle 不变量**：

- 整个 bundle 是 frozen dataclass；任何字段修改只能通过 `dataclasses.replace()` 产出新 bundle，不允许原地修改（R8）
- bundle 的运行时消费链路全部通过快照只读，**不**有人 import `lifeform_domain_figure` 内部模块（除 vertical 自己 + DLaaS adopt 加载点）
- `FigureSteeringSet` / `FigureLoRAArtifact` 进入 bundle 前必须过 `ModificationGate.OFFLINE`（R10），强制 `validation_delta ≥ 0.05` + `is_reversible=True` + 非空 `rollback_evidence`
- bundle 默认 `WiringLevel.SHADOW`；ACTIVE 切换需要 evaluation 6 族证据 + DLaaS template 显式 wiring（R15）
- 跨 vertical 隔离：`lifeform-domain-figure` 不 import `lifeform-domain-character`，反之亦然；CI 由 [`tests/contracts/test_import_boundaries.py`](../tests/contracts/test_import_boundaries.py) 强制

详见 [`docs/specs/figure-vertical.md`](specs/figure-vertical.md)。

### 2.16 Growth-Advisor Profile（LTV 私域运营 vertical 不可变 reviewed artifact）

**所在 wheel**：`lifeform-domain-growth-advisor`（schema + 编译流水线）

`GrowthAdvisorProfile` 是 [`lifeform-domain-growth-advisor`](../packages/lifeform-domain-growth-advisor/) vertical 的**不可变 reviewed artifact**，承载长程私域运营成长规划师人设的全部 runtime 种子。它由该 vertical 唯一拥有 (R8)，跨 wheel 只读消费；**不**新增 kernel runtime owner。

| profile 字段 | 编译目标 | 运行时消费者 |
|---|---|---|
| `knowledge_seeds: tuple[GrowthAdvisorKnowledgeSeed, ...]` | `DomainKnowledgeRecord` | `vz-application` `domain_knowledge` owner |
| `signature_cases: tuple[GrowthAdvisorSignatureCase, ...]` | `CaseMemoryRecord` | `vz-application` `case_memory` owner |
| `strategy_priors: tuple[GrowthAdvisorStrategyPrior, ...]` | `PlaybookRule` (with `applicability_scope`) | `vz-application` `strategy_playbook` owner |
| `boundary_priors: tuple[GrowthAdvisorBoundaryPrior, ...]` | `BoundaryPriorHint` | `vz-application` `boundary_policy` owner |
| `drive_priors: tuple[GrowthAdvisorDrivePrior, ...]` | `DriveSpec` (via `VitalsBootstrap`) | `lifeform-core.VitalsModule`（既有 owner） |

**Growth-advisor profile 不变量**：

- profile 是 frozen dataclass；`__post_init__` 强制 `boundary_priors` 非空 — LTV 档案不允许在没有显式 anti-sales / anti-overclaim / anti-flooding / anti-judgmental 边界的情况下存在
- 4 条 anchoring boundary id 必须包含 `bp-no-hard-sell` / `bp-no-overclaim` / `bp-no-flooding` / `bp-no-judgmental`；前两条结构性地阻止该 vertical 退化成普通销售 bot
- 7-day playbook 通过 `CharacterStrategyPrior.applicability_scope=("growth_advisor:dayN", ...)` 实现跨多日的行为差异，**不**通过用户原文关键词匹配
- 4 大需求挖掘 funnel（height / immunity / nutrition / vision-brain）通过 `applicability_scope=("funnel:X", ...)` 编码，下游 owner 通过 scope 匹配做 turn-级路由
- 跨 vertical 隔离：`lifeform-domain-growth-advisor` 不 import `lifeform-domain-character` / `lifeform-domain-figure`，反之亦然；CI 由 [`tests/contracts/test_import_boundaries.py`](../tests/contracts/test_import_boundaries.py) 三对 parallel pair 强制
- `lifeform-service.verticals.discover_verticals()` 通过 `_try_growth_advisor` 软发现，未安装 wheel 时静默跳过，已安装时暴露为 `name="growth_advisor"`

---

## 3. 模块快照契约

### 3.1 稳定基底层 (Substrate)

**Slot**: `substrate`

```python
@dataclass(frozen=True)
class FeatureSignal:
    name: str
    values: tuple[float, ...]
    source: str
    layer_hint: int | None = None

@dataclass(frozen=True)
class ResidualActivation:
    layer_index: int                    # 残差流层索引
    activation: tuple[float, ...]       # 激活向量 e_{t,l}（不可变 tuple）
    step: int                           # 时间步

@dataclass(frozen=True)
class ResidualSequenceStep:
    step: int
    token: str
    feature_surface: tuple[FeatureSignal, ...]
    residual_activations: tuple[ResidualActivation, ...]
    description: str

@dataclass(frozen=True)
class UnavailableField:
    field_name: str
    reason: str
    detail: str

class SurfaceKind(Enum):
    PLACEHOLDER = "placeholder"
    FEATURE_SURFACE = "feature-surface"
    RESIDUAL_STREAM = "residual-stream"

@dataclass(frozen=True)
class SubstrateSnapshot:
    model_id: str                       # 基础模型版本标识
    is_frozen: bool                     # 是否冻结
    surface_kind: SurfaceKind           # 当前暴露的 substrate 表面
    token_logits: tuple[float, ...]     # 当前步 token 概率分布（可为空）
    feature_surface: tuple[FeatureSignal, ...]
    residual_activations: tuple[ResidualActivation, ...]
    residual_sequence: tuple[ResidualSequenceStep, ...]
    unavailable_fields: tuple[UnavailableField, ...]
    description: str
```

**阶段化 contract**：

- 当前稳定 contract：`surface_kind=FEATURE_SURFACE`，发布 `feature_surface` 与可选 `token_logits`
- 保守占位 contract：`surface_kind=PLACEHOLDER`，明确哪些字段 unavailable
- 当前增强 contract：`surface_kind=RESIDUAL_STREAM`，发布当前步 `residual_activations` + 可选 `residual_sequence`
- `residual_sequence` 是 temporal / internal_rl 的正式 sequence-aware 输入；fallback adapter 可发布空序列或单步合成序列
- 当前已补充 hook-ready owner contract：`OpenWeightResidualRuntime.capture(source_text) -> OpenWeightRuntimeCapture`，由 `OpenWeightResidualStreamSubstrateAdapter` 负责把 open-weight runtime 暴露为稳定的 `SubstrateSnapshot`
- substrate runtime capability 现明确区分两类路径：默认 live runtime 允许 `capture()` / `apply_control()` / `generate()`，并可在通过 session/joint-loop 的 schedule + gate 后调用 `apply_online_fast_state()` / `import_rare_heavy_state()`；`train_rare_heavy()` 仍只允许 offline clone 执行。显式 frozen runner 则保留“只读 live runtime + review-only artifact”语义
- 当前已落地 `TransformersOpenWeightResidualRuntime`：可对 Hugging Face open-weight causal LM 的中间层 block 注册真实 forward hook，发布 middle-layer residual capture，并通过 owner-side hook 返回受控干预后的新 capture；owner 同时负责把更大 hidden state 压缩成稳定 summary signals（如 `top_logit_entropy`、`top_logit_margin`、`hook_layer_coverage` / `hook_fire_rate`、`planned_layer_fraction`、`token_step_coverage`、`residual_sequence_present`、`fallback_active`）。其中 `hook_layer_coverage` 表示实际 requested hooks fire rate，`planned_layer_fraction` 表示选层比例，consumer 不应混用二者
- substrate owner 现进一步在 `feature_surface` 发布 turn-level semantic hints：`semantic_task_pull`、`semantic_support_pull`、`semantic_repair_pull`、`semantic_exploration_pull`，以及 `semantic_text_weight` / `semantic_residual_weight`；下游直接消费这些公开 signals，而不在 consumer 侧重建文本语义
- substrate owner 当前还会在 `feature_surface` 发布 substrate rare-heavy telemetry，例如 `substrate_rare_heavy_update_count` 与 `substrate_delta_parameter_count`，用于让 evaluation / acceptance / replay artifact 读取“是否真的存在 substrate-level slow update evidence”，而不是由 consumer 侧猜测
- 当前 runtime owner 已显式支持 `SubstrateFallbackMode`：`allow-builtin` 允许回退到内置 tiny transformers runtime，`deny` 在首选 HF model 不可用时直接 fail closed
- 当前默认 session/runner/CLI 已优先使用 `TransformersOpenWeightResidualRuntime`；若首选 HF model 不可用且 fallback mode 允许，则回退到内置 tiny transformers runtime，而不是 synthetic runtime
- 内置 tiny transformers runtime 现固定 deterministic seed，保证 fallback 模式下的 substrate capture 和 semantic hints 可复现
- 当前 session/runner 已允许通过 `substrate_adapter_factory(user_input, turn_index)` 注入 open-weight adapter；表达层不再直接消费完整 snapshot dict，而只消费 richer distilled response context，避免跨 loop 持有 live snapshot 引用
- 当前 substrate rare-heavy checkpoint 也已升级到 owner-side `adapter-delta-v2` contract：checkpoint 除了已有的 `control_scale`、`semantic_text_weight`、`semantic_residual_weight`、`semantic_anchor_bias`、`update_count` 等 evidence 字段外，还允许发布 `training_mode`、`compatibility_fingerprint`、`adapter_scale`、`adapter_parameter_count`、`adapter_training_loss` 与 `adapter_layers`。这些字段只允许 substrate owner 在 `export / import / restore_rare_heavy_state()` surface 上读写，session / joint loop 只能搬运 artifact，不可重建或直写 payload。默认主路径下，live session 可在通过 pre-import replay / evolution gate 后自动导入；显式 frozen runner 只保留生成或评审这类 artifact 的能力

**消费者**：Metacontroller、记忆系统、双轨学习层、评估体系
**发布频率**：每 turn（当前稳定）；未来可扩展到每 token

### 3.2 时间抽象与内部控制层 (TemporalAbstraction)

**Slot**: `temporal_abstraction`

```python
@dataclass(frozen=True)
class ControllerState:
    code: tuple[float, ...]             # 控制器代码 z_t
    code_dim: int                       # 控制器代码维度 n_z
    switch_gate: float                  # 切换门 β_t ∈ [0, 1]
    is_switching: bool                  # β_t > threshold → True
    steps_since_switch: int             # 自上次切换以来的步数

@dataclass(frozen=True)
class TemporalAbstractionSnapshot:
    controller_state: ControllerState
    active_abstract_action: str         # 当前抽象动作的语义描述
    controller_params_hash: str         # U_t 参数的哈希（用于变更检测）
    description: str                    # 模块自身生成的状态描述
    action_family_version: int = 0      # owner-side discovered family bank 版本
    memory_feedback_signal: tuple[float, ...] = ()  # temporal owner 发布给 memory owner 的上一轮 learned feedback signal
    memory_retrieval_facets: tuple[str, ...] = ()    # temporal owner 自报的 memory retrieval facets
```

**当前实现口径**：

- P08 已固定 `controller_state` 的 machine-readable shape
- 当前实现支持 `placeholder` / `heuristic` / `learned-lite` / `full-learned` 四类可替换策略位点
- `active_abstract_action` 和 `description` 是可读输出，不作为 machine state 的唯一来源
- `full-learned` owner 的 runtime-visible state 当前已发布 prior mean/std、posterior mean/std、posterior sample noise、`z_tilde`、posterior drift、binary switch ratio / sparsity / persistence window、decoder output / applied control、policy replacement score，以及 discovered family summary/version；公共 `TemporalAbstractionSnapshot` 额外允许发布 `memory_feedback_signal` 与 `memory_retrieval_facets`，供 memory owner 在自己的 processing path 中消费，而不是由 orchestrator 或 memory owner 反向解析 temporal 内部结构
- internal RL 当前允许通过 causal policy proposal 覆盖 owner 的 `z_candidate`，但覆盖仍通过 temporal owner 完成最终 `z_t` 更新，保持单一 owner
- substrate owner 当前允许 owner-side residual intervention backend 基于现有 `SubstrateSnapshot` 生成受控 residual effect；backend 名称和 rollout path evidence 仅在 owner/internal report 层发布，不改变公共 snapshot shape
- 当前 residual intervention backend 已补充真正 open-weight 运行时位点：`OpenWeightResidualInterventionBackend(runtime, source_text)` 委托 runtime 自己执行中间层干预，公共 `ResidualControlApplication` shape 保持不变；当前 `TransformersOpenWeightResidualRuntime` 已实现 middle-layer hook capture/intervention，`TraceResidualInterventionBackend` 退回为近似基线而非唯一 backend
- 当前 default runtime 已把 temporal owner 拆成 staged slots：
  - `world_temporal` / `self_temporal`：same-wave early control，主要消费 `substrate` 与 `memory`
  - `world_temporal_consolidation` / `self_temporal_consolidation`：late consolidation，主要消费 `reflection` 与 `prediction_error`
  - `temporal_abstraction`：公共聚合 slot，由 `TemporalAggregateModule` 聚合 world/self temporal 快照后发布
- staged temporal slots 不引入第二 owner：world/self track policy 仍各自拥有自己的内部状态；聚合 slot 只发布 compact public state，不重建 producer internals
- 当前 default self-track temporal owner 若未显式传入，会从 world-track discovered metacontroller snapshot 克隆初始参数，保证默认主链共享同一条 discovered lineage，同时维持独立 store/owner
- 当前默认 final wiring 已把 `temporal_abstraction` 放入 ACTIVE 主链；其缺失在 acceptance report 中视为回归

**消费者**：编排器、双轨学习层、认知 Regime 层、评估体系
**发布频率**：每 turn

### 3.3 连续记忆系统 (Memory)

**Slot**: `memory`

2026-05-06 update: CMS ATLAS / Titans uplift is default-ACTIVE via
`build_default_memory_store(...)` after the SHADOW validation ladder passed.
Rollback remains explicit by constructing the store with
`cms_pe_features_enabled=False, cms_replay_window_size=None`.

```python
@dataclass(frozen=True)
class MemoryEntry:
    entry_id: str                       # 唯一标识
    content: str                        # 记忆内容
    track: Track                        # 所属轨道
    stratum: str                        # 所属层级: transient | episodic | durable | derived
    created_at_ms: int                  # 创建时间
    last_accessed_ms: int               # 最后访问时间
    strength: float                     # 记忆强度 ∈ [0, 1]
    tags: tuple[str, ...]               # 语义标签

@dataclass(frozen=True)
class MemoryWriteRequest:
    content: str
    track: Track
    stratum: MemoryStratum
    tags: tuple[str, ...] = ()
    strength: float = 0.5

@dataclass(frozen=True)
class CMSBandState:
    name: str
    vector: tuple[float, ...]
    last_update_ms: int
    cadence_interval: int
    observations_since_update: int
    pending_signal: tuple[float, ...]
    learning_rate: float = 0.0
    effective_learning_rate: float = 0.0
    momentum: tuple[float, ...] = ()
    anti_forgetting_strength: float = 0.0
    update_gate: float = 0.0
    slow_mix: float = 0.0
    reset_mix: float = 0.0
    confidence: float = 0.0
    update_summary: str = ""
    mode: str = "vector"              # "vector" | "mlp"
    mlp_param_count: int = 0          # 0 for vector mode
    # ATLAS / Titans uplift readouts (additive, frozen).
    # See docs/specs/cms-atlas-titans-uplift.md §6.
    replay_window_size: int = 0                # 0 when uplift disabled, else current K for this band
    pe_feature_summary: tuple[float, ...] = ()  # last PE feature 4-tuple fed to the rule for this band

@dataclass(frozen=True)
class CMSTowerLevelState:
    level_id: str
    role: str
    vector: tuple[float, ...]
    cadence_interval: int
    source_level_ids: tuple[str, ...] = ()
    description: str = ""

@dataclass(frozen=True)
class CMSTowerProfile:
    profile_id: str
    levels: tuple[CMSTowerLevelState, ...]
    readout_vector: tuple[float, ...]
    description: str

@dataclass(frozen=True)
class CMSContinuumBand:
    band_id: str
    role: str
    vector: tuple[float, ...]
    cadence_interval: int
    update_frequency: float
    persistence_bias: float
    retrieval_weight: float
    pending_signal: tuple[float, ...] = ()
    source_band_ids: tuple[str, ...] = ()
    description: str = ""

@dataclass(frozen=True)
class CMSContinuumReconstructionEdge:
    edge_id: str
    source_band_id: str
    target_band_id: str
    transfer_kind: str
    strength: float
    description: str

@dataclass(frozen=True)
class CMSContinuumProfile:
    profile_id: str
    bands: tuple[CMSContinuumBand, ...]
    reconstruction_edges: tuple[CMSContinuumReconstructionEdge, ...]
    readout_band_id: str
    description: str

@dataclass(frozen=True)
class CMSHopeSelfModificationState:
    enabled: bool
    update_count: int
    last_target_id: str
    generated_learning_rate: float
    generated_decay_rate: float
    generated_reset_rate: float
    last_improvement: float
    last_stability: float
    last_reward: float
    guarded: bool
    guard_reason: str = ""
    description: str = ""

@dataclass(frozen=True)
class CMSState:
    online_fast: CMSBandState
    session_medium: CMSBandState
    background_slow: CMSBandState
    total_observations: int
    total_reflections: int
    description: str
    variant: str = "sequential"       # "sequential" | "independent" | "nested"
    tower_profile: CMSTowerProfile | None = None
    tower_depth: int = 0
    continuum_profile: CMSContinuumProfile | None = None
    update_rule_state: LearnedUpdateRuleState | None = None
    hope_self_modification_state: CMSHopeSelfModificationState | None = None
    # ATLAS / Titans uplift readouts (additive, frozen).
    atlas_replay_active: bool = False                   # True when ATLAS past-aware joint-optimization is on
    titans_pe_gate_active: bool = False                 # True when Titans PE-driven write gating is on
    replay_window_sizes: tuple[tuple[str, int], ...] = ()  # (band_id, configured K) pairs

@dataclass(frozen=True)
class CMSCheckpointState:
    online_fast: tuple[float, ...]
    session_medium: tuple[float, ...]
    background_slow: tuple[float, ...]
    last_update_ms: int
    total_observations: int
    total_reflections: int
    session_observations_since_update: int
    background_observations_since_update: int
    session_pending_signal: tuple[float, ...]
    background_pending_signal: tuple[float, ...]
    mode: str = "vector"              # "vector" | "mlp"
    mlp_params: tuple[tuple[tuple[float, ...], ...], ...] = ()
    nested_session_init_target: tuple[float, ...] = ()   # nested 变体: session band 元学习的初始化目标
    nested_online_init_target: tuple[float, ...] = ()    # nested 变体: online band 元学习的初始化目标
    tower_meta_levels: tuple[tuple[str, tuple[float, ...]], ...] = ()
    update_rule_state: LearnedUpdateRuleState | None = None
    hope_self_modification_state: CMSHopeSelfModificationState | None = None
    # ATLAS / Titans uplift readouts (additive, frozen).
    atlas_replay_active: bool = False
    titans_pe_gate_active: bool = False
    replay_window_sizes: tuple[tuple[str, int], ...] = ()

@dataclass(frozen=True)
class MemorySnapshot:
    # artifact / explanation layer 的按层级摘要；不等同于 learned core 全量真相
    transient_summary: str              # 瞬态 artifact 摘要（模块自身生成）
    episodic_summary: str               # 会话 artifact 摘要
    durable_summary: str                # 持久 durable artifact 摘要

    # 本轮检索到的相关 artifact；由 learned-core-guided recall 选出
    retrieved_entries: tuple[MemoryEntry, ...]

    # 统计信息
    total_entries_by_stratum: tuple[tuple[str, int], ...]  # (stratum, count) pairs
    pending_promotions: int             # 待提升的记忆数量
    pending_decays: int                 # 待衰减的记忆数量
    cms_state: CMSState | None          # owner 发布的 machine-readable CMS 多频带状态

    description: str                    # 模块自身生成的整体状态描述
    lifecycle_metrics: tuple[tuple[str, float], ...] = ()  # owner 负责的 lifecycle telemetry，如 reset、slow->fast benefit、learned recall evidence
    cms_band_vectors: tuple[tuple[str, tuple[float, ...]], ...] = ()  # CMS 三带原向量；评估面消费
    suppressed_cross_scope_entries: tuple[MemoryEntry, ...] = ()  # 被 multi-party scope 过滤掉的检索结果（owner 自报，不是消费者推算）
    active_subject_scope: tuple[str, ...] = ()  # 本轮 retrieval 真正使用的 active subject scope
    social_pe_signals: tuple[MemorySocialPESignal, ...] = ()  # owner 自身发布的 typed social PE signals；社交 prediction / error owner 只做契约级 lift，不重建
    attribute_summary: tuple[MemoryAttributeReadout, ...] = ()  # Phase 1.C owner-internal 卡片属性读出（PE intensity + primary axis + regime + substrate digest + epistemic/aleatoric），capped 16 条；不写 checkpoint，重启后由 PE / substrate 自然回填

@dataclass(frozen=True)
class MemoryAttributeReadout:
    entry_id: str
    pe_intensity: float                              # 来自 PredictionError.magnitude
    pe_primary_axis: str                             # 来自 PE 主导轴
    regime_id: str                                   # 来自 PE action_context
    substrate_feature_digest: tuple[float, ...]      # 截短 _substrate_embedding，dim ≤ 8
    epistemic_magnitude: float                       # 来自 PEDecomposition.epistemic_magnitude（缺失则 0）
    aleatoric_magnitude: float                       # 来自 PEDecomposition.aleatoric_magnitude（缺失则 0）
    timestamp_ms: int
```

**owner 规则**：

- 所有记忆写入必须通过 `MemoryWriteRequest` 形式进入 Memory owner API
- 消费者不得直接持有或修改 memory 内部存储结构
- 提升、衰减、部分重建的 pending 状态由 Memory owner 自身发布
- `cms_state` 是 Memory owner 对外发布的唯一 CMS 可读状态；消费者不得自行拼装 band cadence
- `cms_state.continuum_profile` 是连续谱频率 contract 的机器可读入口；消费者需要理解 bands / reconstruction edges / readout band 时读取该字段，不从三带摘要反推
- `cms_state.tower_profile` / `tower_depth` 只发布 nested tower 的 compact readout 与层级身份，不暴露 owner 内部全量参数
- `cms_state.update_rule_state` 与 `hope_self_modification_state` 只作为 owner-side learned update / bounded self-modification 证据，不授权外部消费者写入 CMS 参数
- `lifecycle_metrics` 只发布 owner 自身负责的 nested lifecycle telemetry；消费者不得自行推断 reset、slow-to-fast transfer、或 learned-core-guided recall 是否发生
- `hope_self_modification_state` 是 Memory owner 内部 tiny Hope 机制的只读证据面；它描述 owner 生成的有界 update 系数和 guard 状态，不授权消费者改写 CMS 参数
- 显式 `MemoryEntry` 属于 artifact / explanation layer；主记忆基底由 owner 内部 learned core 承担
- `social_pe_signals` 是 Memory owner 对社交 PE 主链的唯一 typed 入口；下游 `social_prediction` / `social_prediction_error` owner 必须通过 `social_prediction_from_memory_signal` / `social_prediction_error_from_memory_signal` 升级，**禁止**消费者从 `suppressed_cross_scope_entries` / `active_subject_scope` 反向重建 `SocialPrediction` / `SocialPredictionError`，也禁止借用 `MemoryModule` 名字写到下游 owner 的 snapshot 中
- semantic retrieval index 属于 Memory owner 内部 derived index，不通过独立 slot 暴露
- runtime retrieval facets 只消费上游 owner 已发布的 `memory_retrieval_facets`（如 `temporal_abstraction` / `dual_track` / `prediction_error`）；不得由 memory 解析 peer snapshot 内部字段重建 facet，也不得通过同轮直接调用形成第二 owner 或循环依赖

**Rupture-repair 记忆 tag schema**（M0 契约冻结，M3 实施）：

rupture-repair 条目**不**引入新的 `MemoryEntryKind` enum。它们复用
`MemoryEntry`，通过固定的 tag schema + `content` 中的结构化 JSON 识别：

必含 tags：

- `rupture_repair`
- `rupture_kind:<kind>`（`kind` 来自 `RuptureKind` 封闭枚举）
- `repair_outcome:<observed|pending>`
- `user_scope:<user_id_or_anon>`
- `source_wave:<wave_id>`

`MemoryEntry.content` 为 JSON 对象，包含：

```json
{
  "rupture_kind": "misread",
  "repair_move": "<id or placeholder>",
  "source_turn_index": 12,
  "source_wave_id": "<wave_id>",
  "observed_outcome_kind": "HELPED",
  "confidence": 0.8
}
```

写入路径：**只有** `ReflectionEngine.apply(...)` 可以写入带 `rupture_repair`
tag 的条目；`stratum=DURABLE` 仅当至少一个 typed external outcome 已观察到。
持久化范围：`DURABLE` 条目落盘；`DERIVED` 派生索引在加载时由 owner 重建。

**Post-v0 迁移路径**：如果 rupture-repair 变成承重能力，在 vz-memory 引入
`RuptureRepairHistory` derived readout（**不**是 `MemoryEntryKind` enum），
从现存 tag 反查回填，消费者切过去后再弃用 tag。详见
[`docs/specs/rupture-and-repair.md`](./specs/rupture-and-repair.md)。

**消费者**：编排器、时间抽象层、双轨学习层、认知 Regime 层、慢反思路径
**发布频率**：每 turn（瞬态/情景）、每会话（持久）

### 3.4 双轨学习层 (DualTrack)

**Slot**: `dual_track`

```python
@dataclass(frozen=True)
class TrackState:
    track: Track
    active_goals: tuple[str, ...]       # 当前活跃目标
    recent_credits: tuple[tuple[str, float], ...]  # (event_id, credit) pairs
    controller_code: tuple[float, ...]  # 轨道专属控制器代码 z_task 或 z_rel
    tension_level: float                # 张力水平 ∈ [0, 1]
    abstract_action_hint: str | None = None
    action_family_version_hint: int = 0
    controller_source: str = "memory"

@dataclass(frozen=True)
class DualTrackSnapshot:
    world_track: TrackState
    self_track: TrackState
    cross_track_tension: float          # 跨轨道张力（两轨目标冲突程度）
    description: str                    # 模块自身生成的状态描述
    memory_retrieval_facets: tuple[str, ...] = ()  # dual_track owner 自报的 memory retrieval facets
```

**当前实现口径**：

- P03 阶段先以结构化状态 owner 落地，不要求完整 temporal / evaluation / credit 全部接入
- `recent_credits` 当前可由 owner 发布为“最近重要状态信号”，后续再与正式 credit owner 对齐
- `controller_code` 当前允许是从已知状态压缩出的占位向量，而不是最终 learned controller code
- `abstract_action_hint` / `controller_source` 当前用于显式说明 dual-track state 是否已经消费 temporal owner 发布的控制证据
- 默认 final wiring 下，dual-track 当前优先消费上一轮已发布的 `temporal_abstraction` 快照，避免形成同轮循环依赖
- `memory_retrieval_facets` 是 dual-track owner 对 memory retrieval 的唯一 compact hint；memory 不从 `world_track.active_goals` / `self_track.active_goals` / `cross_track_tension` 反向拼装 dual-track facet

**消费者**：编排器、记忆系统、信用分配、评估体系
**发布频率**：每 turn

### 3.5 信用分配与自修改 (CreditAssignment)

**Slot**: `credit`

```python
@dataclass(frozen=True)
class CreditRecord:
    record_id: str
    level: str                          # token | turn | session | long_term | abstract_action
    track: Track
    source_event: str                   # 触发信用分配的事件描述
    credit_value: float                 # 信用值
    context: str                        # 上下文描述（语义化，非纯数值）
    timestamp_ms: int

@dataclass(frozen=True)
class SelfModificationRecord:
    target: str                         # 修改目标描述
    gate: ModificationGate              # 门控级别
    decision: GateDecision              # allow | block
    old_value_hash: str                 # 修改前值的哈希
    new_value_hash: str                 # 修改后值的哈希
    justification: str                  # 修改理由
    timestamp_ms: int
    is_reversible: bool                 # 是否可回滚

@dataclass(frozen=True)
class ModificationProposal:
    target: str
    desired_gate: ModificationGate
    old_value_hash: str
    new_value_hash: str
    justification: str
    is_reversible: bool = True
    validation_delta: float = 0.0        # 候选相对基线的验证改进
    capacity_cost: float = 0.0           # 容量/影响面成本估计，越高越保守
    rollback_evidence: str = ""          # checkpoint / replay / rollback lineage

@dataclass(frozen=True)
class RewardingStateHeadState:
    rule_id: str
    feature_dim: int
    update_count: int
    weights: tuple[float, ...]
    bias: float
    last_prediction: float
    last_target: float
    last_validation_delta: float
    last_capacity_cost: float
    last_rollback_evidence: str
    description: str = ""

@dataclass(frozen=True)
class CounterfactualContributionReadout:
    source_event: str
    historical_baseline: float
    learned_baseline: float
    actual_outcome: float
    learned_contribution: float
    validation_delta: float
    update_count: int
    checkpoint_id: str
    gate_decision: GateDecision
    description: str = ""

@dataclass(frozen=True)
class CreditSnapshot:
    recent_credits: tuple[CreditRecord, ...]
    recent_modifications: tuple[SelfModificationRecord, ...]
    cumulative_credit_by_level: tuple[tuple[str, float], ...]  # (level, sum) pairs
    description: str
    rewarding_state_head: RewardingStateHeadState | None = None
    counterfactual_readouts: tuple[CounterfactualContributionReadout, ...] = ()
```

**`CreditRecord.level` 取值**：`token` / `turn` / `session` / `long_term` / `abstract_action` / `prediction_error` / `evaluation_readout` / `social_prediction_error` / `abstract_action_segment` / `counterfactual_contribution`（Phase 1.A COCOA-lightweight，readout-only，不参与 acceptance gate）/ `counterfactual_contribution_learned`（Phase 2.A learned rewarding-state head readout，SHADOW 默认）。

**当前实现口径**：

- P06 当前落地结构化信用记录、gate audit 与 bounded self-modification proposal；默认 `CreditModule` 以 `SHADOW` 接线运行，真实写入仍必须通过对应 owner 的 apply surface 和 gate，而不是由 credit owner 直接突变外部模块
- `recent_modifications` 当前记录 allow / block decision，作为审计轨迹和后续 reflection 输入
- `cumulative_credit_by_level` 先提供最小聚合，后续再扩展到更细粒度的长期统计
- 第二阶段允许在 owner 内部基于 temporal / rollout 结果扩展出 `abstract_action` 级 credit，而不改变 `CreditSnapshot` shape
- metacontroller credit 当前会消费 posterior drift、binary gate ratio、policy replacement score 等 ETA kernel evidence，并将其压入 `CreditRecord.context`
- `derive_credit_records_from_prediction_error_first(...)` 是当前 PE-first credit 派生路径；evaluation 只提供 readout / gate context，不重新成为原始学习源
- 自修改 gate 当前采用 Two-Gate 风格的保守准入：候选必须提供 validation margin、capacity cap 内的容量成本和 rollback evidence；缺证据默认 block，并把原因写入 `SelfModificationRecord.justification`
- Phase 2.A 的 `RewardingStateHeadState` 归属 `CreditLedger`，只用 owner 内部 context 向量预测 expected outcome；更新必须带 `validation_delta` / `capacity_cost` / `rollback_evidence`，并写入 `recent_modifications` 审计。`counterfactual_readouts` 是对比 readout，不授权下游重建 head 权重。

**消费者**：编排器、记忆系统（反思输入）、评估体系
**发布频率**：每 turn（即时信用）、每会话（会话级信用）

### 3.6 认知 Regime 层 (CognitiveRegime)

**Slot**: `regime`

```python
@dataclass(frozen=True)
class RegimeIdentity:
    regime_id: str                      # 唯一标识
    name: str                           # 语义名称
    embedding: tuple[float, ...]        # 运行时向量表示（非字符串标签）
    entry_conditions: str               # 进入条件描述
    exit_conditions: str                # 退出条件描述
    historical_effectiveness: float     # 历史效果评分 ∈ [0, 1]

@dataclass(frozen=True)
class RegimeSelectionWeights:
    weights: tuple[tuple[str, float], ...]
    learning_rate: float = 0.02

@dataclass(frozen=True)
class DelayedOutcomeAttribution:
    regime_id: str
    outcome_score: float
    source_turn_index: int
    source_wave_id: str
    abstract_action: str | None = None
    action_family_version: int = 0
    resolved_turn_index: int = 0

@dataclass(frozen=True)
class DelayedOutcomePayoff:
    regime_id: str
    abstract_action: str | None
    action_family_version: int
    sample_count: int
    rolling_payoff: float
    latest_outcome: float
    last_source_wave_id: str

@dataclass(frozen=True)
class RegimeSequencePayoff:
    regime_sequence: tuple[str, ...]
    family_version: int
    sample_count: int
    rolling_payoff: float
    latest_outcome: float
    last_source_wave_id: str

@dataclass(frozen=True)
class RegimeSnapshot:
    active_regime: RegimeIdentity
    previous_regime: RegimeIdentity | None
    switch_reason: str                  # 切换原因（如有切换）
    candidate_regimes: tuple[tuple[str, float], ...]  # (regime_id, score) pairs
    turns_in_current_regime: int
    description: str
    delayed_outcomes: tuple[tuple[str, float], ...]   # owner-attributed delayed regime outcomes
    delayed_attributions: tuple[DelayedOutcomeAttribution, ...] = ()
    delayed_attribution_ledger: tuple[DelayedOutcomeAttribution, ...] = ()
    delayed_payoffs: tuple[DelayedOutcomePayoff, ...] = ()
    sequence_payoffs: tuple[RegimeSequencePayoff, ...] = ()
    identity_hints: tuple[str, ...]                   # typed identity proposals for reflection/memory
    effectiveness_trend: tuple[tuple[str, float], ...] = ()
    regime_changed: bool = False
    selection_weights: RegimeSelectionWeights | None = None
```

**当前实现口径**：

- P04 阶段已经提供结构化 regime identity 和 candidate scoring
- 当前选择逻辑基于 `memory`、`dual_track`、`evaluation` 的状态评分基线
- 第二阶段补充 regime owner 的 bounded policy apply：strategy priors 与 historical effectiveness 可 checkpoint / rollback
- 当前 `RegimeModule` 已补充 owner-side delayed attribution queue：上一轮 regime 选择会在后续 turn 的 evaluation 上结算，并通过 `delayed_outcomes` + `delayed_attributions` 发布结果；后者会携带 `source_wave_id`、`source_turn_index`、`abstract_action`、`action_family_version`
- 当前 regime owner 还会发布 `delayed_attribution_ledger` 与 `delayed_payoffs`：前者保留最近若干条 resolved attribution，后者按 `(regime, abstract_action, action_family_version)` 聚合 rolling payoff，供 credit / evaluation / reflection 直接消费
- 当前 `identity_hints` 由 regime owner 从 memory snapshot 中投影为 typed identity proposal，供 reflection/memory owner 决定是否沉淀为 durable identity entries
- 当前 regime owner 还会发布 `sequence_payoffs`、`effectiveness_trend`、`regime_changed` 与 `selection_weights`，使 delayed sequence outcome 与 learned selection bias 可被下游审计；consumer 不应从私有 ledger 重建这些读数
- 该评分基线是过渡实现；后续可由更强的 temporal / learned policy 替换

**消费者**：编排器、时间抽象层、记忆系统、评估体系
**发布频率**：每 turn

### 3.7 评估体系 (Evaluation)

**Slot**: `evaluation`

```python
@dataclass(frozen=True)
class EvaluationScore:
    family: str                         # 评估族: task | interaction | relationship | learning | abstraction | safety
    metric_name: str                    # 具体指标名
    value: float                        # 分值
    confidence: float                   # 置信度 ∈ [0, 1]
    evidence: str                       # 证据描述

@dataclass(frozen=True)
class EvaluationSnapshot:
    turn_scores: tuple[EvaluationScore, ...]        # 本轮评分
    session_scores: tuple[EvaluationScore, ...]     # 会话累计评分
    alerts: tuple[str, ...]                          # 安全/有界性告警
    description: str
    reflection_accuracy: float = 0.0                 # 反思 proposal 预测准确率 (由 final_wiring 从 ReflectionEngine.proposal_success_rate 注入)
    longitudinal_verdict: str = ""                   # 跨 session 纵向评估结论 ("growing" | "stable" | "regressing" | "")
```

**当前实现口径**：

- P05 阶段先提供 turn / session 两级的最小评估通路
- `turn_scores` 必须包含 evidence；`session_scores` 为当前 session 的聚合视图
- 告警先以结构化字符串对外发布，后续可升级为更细粒度 alert schema
- owner-side kernel evaluation 当前已直接记录 `posterior_stability`、`switch_sparsity`、`binary_gate_ratio`、`decoder_usefulness`、`policy_replacement_quality`，以及 family-level abstraction metrics（如 `action_family_reuse`、`action_family_stability`、`action_family_diversity`、`delayed_action_alignment`、`regime_sequence_payoff`、`delayed_credit_horizon`、`rolling_action_payoff`）
- `EvaluationBackbone` 当前还提供 default replay benchmark 与 evolution judge（promote / hold / rollback），但这些 judgement 仍以 report/evidence 形式存在，不改变 `EvaluationSnapshot` 公共 shape
- session / cross-session report 当前补充 `identity_continuity`、`relationship_repair_continuity`、`async_robustness` 三类存在性 readout；它们只进入 report/trend/evidence，不改变 `EvaluationSnapshot` 公共 shape，也不成为学习源头

**消费者**：编排器、信用分配、门控自修改
**发布频率**：每 turn（即时评分）、每会话（会话评分）

### 3.8 慢反思路径 (SlowReflection)

**Slot**: `reflection`

```python
@dataclass(frozen=True)
class MemoryConsolidation:
    new_durable_entries: tuple[MemoryEntry, ...]    # 新产生的持久记忆
    promoted_entries: tuple[str, ...]               # 被提升的记忆 ID
    decayed_entries: tuple[str, ...]                # 被衰减的记忆 ID
    beliefs_updated: tuple[str, ...]                # 更新的信念描述

@dataclass(frozen=True)
class PolicyConsolidation:
    controller_updates: tuple[str, ...]             # 控制器参数更新描述
    strategy_priors_updated: tuple[str, ...]        # 更新的策略先验
    regime_effectiveness_updated: tuple[tuple[str, float], ...]  # (regime_id, new_score) pairs
    temporal_prior_update: TemporalPriorUpdate | None
    controller_guard_blocked: bool
    controller_guard_audit_present: bool

@dataclass(frozen=True)
class TemporalPriorUpdate:
    target: str
    residual_strength: float
    memory_strength: float
    reflection_strength: float
    switch_bias_delta: float
    persistence_delta: float
    learning_rate_delta: float
    description: str

@dataclass(frozen=True)
class ConsolidationScore:
    promotion_score: float
    decay_score: float
    threshold_delta: float
    strategy_gain: float
    regime_effectiveness_gain: float
    confidence: float
    description: str

@dataclass(frozen=True)
class ReflectionSnapshot:
    memory_consolidation: MemoryConsolidation
    policy_consolidation: PolicyConsolidation
    consolidation_score: ConsolidationScore
    interaction_trace_summary: str                  # 交互轨迹摘要
    tensions_identified: tuple[str, ...]            # 识别到的张力
    lessons_extracted: tuple[str, ...]              # 提取的持久教训
    writeback_mode: str                             # disabled | proposal-only | apply
    review_required: bool
    description: str
```

**当前实现口径**：

- P07 默认以 `proposal-only` 运行
- 第二阶段补充 bounded apply path；`ReflectionEngine.apply` 只执行 memory owner 有界写回并保留 checkpoint，regime / temporal 写回由编排层显式调用目标 owner 的正式 API
- `memory_consolidation` 和 `policy_consolidation` 仍先表达提案和审计结果，再由 gate / rollout 决定是否 apply
- `consolidation_score` 是 reflection owner 发布的统一 bounded score 路径；memory / regime / temporal 的写回幅度由该 score 决定，但目标 owner 自己负责最终应用
- `beliefs_updated` 已接入 memory owner 的 audited apply，不再是仅存在于 proposal 中的伪状态
- `review_required=True` 表示需要后续 gate / human / rollout 决策后才能放大范围
- 当前 `policy_consolidation` 已成为 reflection owner 对 regime / temporal owner 的 typed 写回契约；编排层只负责 target-specific gate + audit，再调用目标 owner 的正式 apply API
- 默认主链中的 active reflection / temporal 会把该 bounded prior writeback 纳入 `writeback_result.applied_operations` 与 credit modification audit

**消费者**：记忆系统、信用分配、Metacontroller、认知 Regime 层
**发布频率**：每会话后（异步）

### 3.9 Prediction Error（PredictionError）

**Slot**: `prediction_error`

```python
@dataclass(frozen=True)
class PredictedOutcome:
    source_turn_index: int
    target_turn_index: int
    predicted_task_progress: float
    predicted_relationship_delta: float
    predicted_regime_stability: float
    predicted_action_payoff: float
    confidence: float
    description: str

@dataclass(frozen=True)
class ActualOutcome:
    observed_turn_index: int
    task_progress: float
    relationship_delta: float
    regime_stability: float
    action_payoff: float
    description: str

@dataclass(frozen=True)
class PredictionError:
    task_error: float
    relationship_error: float
    regime_error: float
    action_error: float
    magnitude: float
    signed_reward: float
    description: str

@dataclass(frozen=True)
class PEDecomposition:
    aleatoric_magnitude: float                # variance floor (sqrt of EMA variance), [0, 1]
    epistemic_magnitude: float                # 可学的部分；Phase 2.B 为 improvement-PE 轴聚合, [0, 1]
    per_axis: tuple[tuple[str, float, float], ...]  # (axis_name, aleatoric, epistemic)
    description: str = ""
    critic_predicted_magnitude: float = 0.0
    improvement_magnitude: float = 0.0
    critic_update_count: int = 0
    critic_checkpoint_id: str = ""
    critic_gate_decision: str = "shadow"

@dataclass(frozen=True)
class PECriticHeadState:
    rule_id: str
    feature_dim: int
    update_count: int
    axis_weights: tuple[tuple[str, tuple[float, ...]], ...]
    axis_biases: tuple[tuple[str, float], ...]
    last_prediction: float
    last_target: float
    last_validation_delta: float
    last_capacity_cost: float
    last_rollback_evidence: str
    description: str = ""

@dataclass(frozen=True)
class PredictionErrorSnapshot:
    evaluated_prediction: PredictedOutcome | None
    actual_outcome: ActualOutcome
    next_prediction: PredictedOutcome
    error: PredictionError
    turn_index: int
    bootstrap: bool
    description: str
    memory_retrieval_facets: tuple[str, ...] = ()  # PE owner 自报的 memory retrieval facets
    pe_decomposition: PEDecomposition | None = None  # Phase 1.B Curiosity-Critic readout；bootstrap 时为 None
```

**当前实现口径**：

- `prediction_error` 已是正式 ACTIVE runtime slot，而不是临时日志对象
- 快照最小公开链固定为 `evaluated_prediction -> actual_outcome -> next_prediction -> error`
- `error` 维度固定覆盖 `task` / `relationship` / `regime` / `action`
- `bootstrap=True` 表示当前 turn 还没有可结算的上一轮 prediction，不应被下游当作正式误差信号消费
- live session 中，部分消费者会把 `prediction_error` 视为“上一轮 carryover learning evidence”，以避免同轮自因果闭环
- `memory_retrieval_facets` 由 PredictionError owner 根据自身 error 维度生成；memory 不从 `PredictionError.error` 各分量重新选择 dominant dimension
- Phase 2.B 在 `_PECriticHead` 内维护 learned contextual critic（`SubstrateSnapshot.feature_surface` digest + `PredictionActionContext` -> expected `|axis_error|`）。`aleatoric_magnitude` 仍来自 EMA variance floor；`epistemic_magnitude` / `improvement_magnitude` 代表 `actual |axis_error| - critic_prediction` 的可改进部分。该 readout 仍为 report-only，不进入 evaluation acceptance gate。

**消费者**：记忆系统、时间抽象层、认知 Regime 层、信用分配、慢反思路径；`evaluation` 在 final wiring 中追加 PE evidence，但不把它变成新的模块 owner
**发布频率**：每 turn

---

## 4. 编排器接口

### 4.1 Upstream Dict

编排器传递给每个模块的上游快照字典：

```python
UpstreamDict = dict[str, Snapshot]
```

**键**为 `slot_name`，值为对应模块发布的最新 `Snapshot`。

### 4.2 模块处理接口

```python
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Mapping, TypeVar

ValueT = TypeVar("ValueT")

class RuntimeModule(ABC, Generic[ValueT]):
    """Base module contract for all runtime owners."""

    slot_name: ClassVar[str]
    owner: ClassVar[str]
    value_type: ClassVar[type[Any]]
    dependencies: ClassVar[tuple[str, ...]] = ()
    default_wiring_level: ClassVar[WiringLevel] = WiringLevel.ACTIVE

    def __init__(self, *, wiring_level: WiringLevel | None = None) -> None:
        self._wiring_level = wiring_level or self.default_wiring_level
        self._version = 0

    @property
    def wiring_level(self) -> WiringLevel:
        return self._wiring_level

    def seed_version(self, version: int) -> None:
        """Seed local publication version from previously published snapshots."""

    def publish(self, value: ValueT) -> Snapshot[ValueT]:
        """Increment version and wrap a frozen value in a Snapshot."""

    @abstractmethod
    async def process(self, upstream: Mapping[str, Snapshot[Any]]) -> Snapshot[ValueT]:
        """
        接收上游快照，执行处理，返回自身快照。

        约束:
        - 只从带守卫的 upstream view 读取数据，不持有/import/调用其他模块
        - 返回的 Snapshot 必须是 frozen dataclass
        - 模块内部状态的描述由自身生成并打包到快照中
        """

    async def process_standalone(self, **kwargs: Any) -> Snapshot[ValueT]:
        """
        独立调用模式（预训练/测试场景）。
        不依赖 upstream，直接接收必要参数。
        """
        raise NotImplementedError
```

### 4.3 编排器快照传播

```python
async def propagate(
    modules: list[RuntimeModule[Any]],
    *,
    upstream: UpstreamDict | None = None,
    registry: SlotRegistry | None = None,
    recorder: EventRecorder | None = None,
    shadow_snapshots: MutableMapping[str, Snapshot] | None = None,
    session_id: str = "runtime",
    wave_id: str = "wave-0",
    auto_sort: bool = True,
) -> UpstreamDict:
    """
    按依赖顺序执行模块，收集快照。

    默认语义:
    - `auto_sort=True` 时按模块声明的 `dependencies` 做拓扑排序
    - 依赖图成环时 `topo_sort_modules()` 回退到调用方给定顺序；需要显式检查时调用 `detect_dependency_cycle()`
    - `auto_sort=False` 时保留调用方给定顺序，仍执行同样的 ownership / dependency / schema / immutability guard

    运行时语义:
    - ACTIVE: 执行并将输出写入 active upstream
    - SHADOW: 执行并校验，但输出只写入 shadow_snapshots
    - DISABLED: 不执行模块逻辑，发布 runtime placeholder snapshot 到 active upstream

    守卫:
    - OwnershipGuard: slot owner 唯一、版本递增
    - DependencyGuard: 只能消费声明的 slot
    - SchemaGuard: 发布值必须符合声明 schema
    - ImmutabilityGuard: 发布后消费前校验哈希不变
    """
    result = dict(upstream or {})
    for module in modules:
        ...
    return result
```

### 4.4 缺失 upstream 与 stub 语义

- 缺失依赖 slot 时，运行时统一返回 `Snapshot[..., value=RuntimePlaceholderValue(...)]`
- `missing-upstream` 与 `disabled-module` 是两类不同 reason
- placeholder snapshot 的 `version=0` 仅用于缺失 upstream；禁用模块发布的 stub 使用正式递增版本
- 模块不允许私自发明其他缺失/降级格式

---

## 5. 快照依赖图

```
substrate ───────────────┬────────→ world_temporal / self_temporal ──→ temporal_abstraction
                         ├────────→ memory ─────────────────────────→ dual_track
                         ├────────→ evaluation
                         └────────→ prediction_error

memory ──────────────────┬────────→ dual_track ──────────┬────────→ evaluation
                         ├────────→ regime               ├────────→ credit
                         ├────────→ reflection           └────────→ prediction_error
                         └────────→ retrieval_policy

evaluation ──────────────┬────────→ regime ──────────────┬────────→ prediction_error
                         ├────────→ credit               ├────────→ retrieval_policy
                         └────────→ reflection           └────────→ response_assembly

prediction_error ────────┬────────→ memory / temporal / regime / credit / reflection
                         ├────────→ case_memory / boundary_policy
                         └────────→ substrate_self_mod

session_post_slow_loop ──→ experience_consolidation ─────→ experience_fast_prior
experience_fast_prior ───┬────────→ temporal owners
                         ├────────→ regime
                         └────────→ retrieval_policy

retrieval_policy ────────┬────────→ domain_knowledge ────┬────────→ boundary_policy
                         ├────────→ case_memory ─────────┼────────→ strategy_playbook
                         └────────→ response_assembly    └────────→ response_assembly

reflection ──────────────→ proposals; runtime invokes owner-side writeback: memory / regime / temporal / credit audit
```

**依赖规则**：
- 每个模块只读取上游快照，不反向依赖
- `reflection` 与 `session_post_slow_loop` 都属于 background/session-post 路径；它们只发布公共 report / proposal surface，真正 apply 仍调用目标 owner 的正式 API
- `reflection` 的产物通过编排层调用正式 API 写回 `memory`、`regime`、`temporal`，`ReflectionEngine` 不持有或直接调用 `RegimeModule`，并通过 `credit` 保留审计证据
- `prediction_error` 是显式学习证据层；部分 live runtime 路径把它当作跨 turn carryover signal，而不是同 turn 自举输入

**关于直接消费与间接消费**：上图展示的是**直接快照依赖**。Slot 注册表（第 6 节）中列出的消费者是**声明的直接消费者**——即模块在 `process()` 中从 upstream dict 读取的 slot。模块不通过中间模块间接获取数据，而是直接声明并读取所需的上游快照。

---

## 6. 快照 Slot 注册表

| Slot Name | Owner 模块 | Value 类型 | 默认接线 | 发布频率 | 消费者 |
|-----------|-----------|-----------|----------|----------|--------|
| `substrate` | SubstrateModule | SubstrateSnapshot | SHADOW | 每 turn | temporal_abstraction, memory, dual_track, evaluation, prediction_error |
| `substrate_self_mod` | SubstrateSelfModModule | SubstrateSelfModSnapshot | SHADOW | 每 turn / schedule | session / credit audit / rare-heavy review |
| `world_temporal` | TrackTemporalModule | TemporalAbstractionSnapshot | SHADOW | 每 turn | temporal_abstraction, dual_track |
| `self_temporal` | TrackTemporalModule | TemporalAbstractionSnapshot | SHADOW | 每 turn | temporal_abstraction, dual_track |
| `world_temporal_consolidation` | TrackTemporalConsolidationModule | TemporalConsolidationSnapshot | SHADOW | 每 turn | final wiring / audit only |
| `self_temporal_consolidation` | TrackTemporalConsolidationModule | TemporalConsolidationSnapshot | SHADOW | 每 turn | final wiring / audit only |
| `temporal_abstraction` | TemporalAggregateModule / TemporalModule | TemporalAbstractionSnapshot | SHADOW | 每 turn | memory, dual_track |
| `memory` | MemoryModule | MemorySnapshot | SHADOW | 每 turn ~ 每会话 | dual_track, regime, reflection, temporal_abstraction, evaluation |
| `plan_intent` | PlanIntentModule | PlanIntentSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation, session-post evidence |
| `commitment` | CommitmentModule | CommitmentSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation, session-post evidence |
| `open_loop` | OpenLoopModule | OpenLoopSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation, session-post evidence |
| `user_model` | UserModelModule | UserModelSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation, session-post evidence |
| `execution_result` | ExecutionResultModule | ExecutionResultSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation, prediction-error evidence |
| `belief_assumption` | BeliefAssumptionModule | BeliefAssumptionSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation |
| `relationship_state` | RelationshipStateModule | RelationshipStateSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation |
| `goal_value` | GoalValueModule | GoalValueSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation |
| `boundary_consent` | BoundaryConsentModule | BoundaryConsentSnapshot | ACTIVE | 每 turn | temporal, boundary_policy, response_assembly, evaluation |
| `dual_track` | DualTrackModule | DualTrackSnapshot | SHADOW | 每 turn | memory, evaluation, prediction_error, reflection, credit, regime |
| `evaluation` | EvaluationModule | EvaluationSnapshot | ACTIVE | 每 turn ~ 每会话 | regime, prediction_error, credit, reflection |
| `regime` | RegimeModule | RegimeSnapshot | SHADOW | 每 turn | prediction_error, reflection, retrieval_policy |
| `prediction_error` | PredictionErrorModule | PredictionErrorSnapshot | ACTIVE | 每 turn | memory, temporal_abstraction, regime, credit, reflection；另在 final wiring 中被 evaluation enrichment 读取 |
| `credit` | CreditModule | CreditSnapshot | SHADOW | 每 turn ~ 每会话 | reflection; consumes `prediction_error` + `temporal_abstraction.closed_segments` for PE-derived segment credit |
| `reflection` | ReflectionModule | ReflectionSnapshot | SHADOW / session-post | 每会话后（异步） | temporal_abstraction；另外通过 owner-side writeback 影响 memory / credit / regime |
| `session_post_slow_loop` | SessionPostSlowLoopModule | SessionPostSlowLoopSnapshot | ACTIVE | context / session boundary | reports / experience_consolidation |
| `retrieval_policy` | RetrievalPolicyModule | RetrievalPolicySnapshot | ACTIVE | 每 turn | domain_knowledge, case_memory, boundary_policy, response_assembly |
| `domain_knowledge` | DomainKnowledgeModule | DomainKnowledgeSnapshot | ACTIVE | 每 turn | boundary_policy, response_assembly, evaluation |
| `case_memory` | CaseMemoryModule | CaseMemorySnapshot | ACTIVE | 每 turn | strategy_playbook, response_assembly, evaluation |
| `strategy_playbook` | StrategyPlaybookModule | StrategyPlaybookSnapshot | ACTIVE | 每 turn | response_assembly, experience_consolidation |
| `boundary_policy` | BoundaryPolicyModule | BoundaryPolicySnapshot | ACTIVE | 每 turn | response_assembly |
| `response_assembly` | ResponseAssemblyModule | ResponseAssemblySnapshot | ACTIVE | 每 turn | session / response generation |
| `experience_consolidation` | ExperienceConsolidationModule | ExperienceConsolidationSnapshot | ACTIVE | session-post | experience_fast_prior, reports |
| `experience_fast_prior` | ExperienceFastPriorModule | ExperienceFastPriorSnapshot | SHADOW | 每 turn / session-post carryover | temporal, retrieval_policy, regime |
| `dialogue_external_outcome` | DialogueExternalOutcomeModule | DialogueExternalOutcomeSnapshot | ACTIVE | 每 turn | prediction_error, regime, rupture_state, reflection |
| `rupture_state` | RuptureStateModule | RuptureStateSnapshot | SHADOW | 每 turn | reflection, dialogue_trace (diagnostic) |
| `interlocutor_state` | InterlocutorStateModule | InterlocutorStateSnapshot | SHADOW | 每 turn | prompt_planner, response_synthesizer, lifeform-core (LifeformSession.interlocutor_state) |
| `active_mixture` | ProtocolRegistryModule | ActiveMixtureSnapshot | SHADOW | 每 turn | （packet 1.2+ 接入：boundary_policy / metacontroller / vitals / strategy_playbook 读 IDs+权重，不读内容本体） |

这里的“默认接线”指模块类声明的 `default_wiring_level`。`final_wiring`、session runner 或 staged rollout 可以在构造模块时显式覆盖接线级别；文档中的 owner / snapshot shape 不因此改变。

**`dialogue_external_outcome` 契约语义**：这是**外部 outcome 进入内核的唯一 snapshot 通道**。`submit_dialogue_outcome(...)` 只向该 slot 的 owner 追加 typed evidence；它**不**直接写 memory / regime / PE 内部状态。`PredictionErrorModule` 在自身 `derive_actual_outcome(...)` 内消费该 snapshot，`RegimeModule` 在自身 `process(...)` 内根据该 snapshot 创建 `PendingRegimeOutcome` 行。这样保持 `_pending_outcomes` / `ActualOutcome` 的单写者不变量（R8）。

**`DialogueExternalOutcomeKind` vocabulary（W3-A 扩展）**：闭合 enum，分两组：

- **In-turn group（v0）**：`HELPED` / `FELT_HEARD` / `MISSED` / `OVER_DIRECTIVE` / `DECISION_CLEARER` / `COME_BACK` / `UNSAFE` / `ABANDONED` —— 单 turn 的对话级反馈，由用户显式表态 / 人工 review / 环境观测产生。
- **LTV / 转化漏斗 group（W3-A）**：`LEAD_QUALIFIED` / `RECOMMENDATION_MADE` / `PURCHASE_CONFIRMED` / `REPURCHASE` / `CHURNED` —— 长程业务事件，**只能**由外部 CRM / payments / 人工 review 注入（不允许从 chat text 推断）。每个值在四张下游表里都有显式 mapping：
  - `_EXTERNAL_OUTCOME_AXIS_BIAS`（PE bias）— `(task, relationship, regime, action)` deltas
  - `_EXTERNAL_OUTCOME_REGIME_SCORE`（regime delayed-outcome 评分）
  - `_EXTERNAL_KIND_TO_STRUCTURAL_OUTCOME`（dialogue_trace 结构投影）
  - `_REPAIR_POSITIVE_KINDS`（reflection writeback；`PURCHASE_CONFIRMED` / `REPURCHASE` 是仅有的两个 LTV 算 repair-resolved 的；`LEAD_QUALIFIED` / `RECOMMENDATION_MADE` 是中间漏斗步，单独不算 repair landed）

`CHURNED` **不**进入 `EXTERNAL_OUTCOME_TO_RUPTURE_KIND`：它是长程脱落事件，损害已经由 PE bias + regime score 全额捕获，且当前 turn 已无法 anchor rupture 修复。如果未来需要 typed "long-horizon churn rupture"，必须先增 `RuptureKind` 值。

`FeedbackValence`（[`packages/dlaas-platform-contracts/src/dlaas_platform_contracts/dispatch_vocab.py`](../packages/dlaas-platform-contracts/src/dlaas_platform_contracts/dispatch_vocab.py)）镜像了这两组，让外部 CRM / payments 集成可以走 DLaaS feedback envelope（`interaction_type=feedback`，`feedback.valence="purchase_confirmed"` 等）报送；platform-api 在 edge 处典型化为 typed enum，kernel 永远不见 raw string。

**`rupture_state` 契约语义**：`rupture_kind` 是 evidence-bucket label，不是情绪分类；只有至少一个非 PE 的 typed source 触发时才能写出；`internal_suspected_only=True` 意味着只有 `INTERNAL_PE` 触发。snapshot 还发布 `kind_label`（W3 SSOT），是 `RuptureKind` 的人类可读短语，由 owner 从 `RUPTURE_KIND_LABEL` 一处生成；下游表达层不维护重复字典。详见 [`docs/specs/rupture-and-repair.md`](./specs/rupture-and-repair.md)。

**`interlocutor_state` 契约语义**（W2 ssot-cleanup-p0-p4）：12-axis 连续读出（`engagement_intensity`, `emotional_weight`, `resistance_level`, `trust_signal`, `pace_pressure`, … 共 12 维），由 `InterlocutorStateModule` 从六个上游 snapshot（`regime` / `dual_track` / `evaluation` / `prediction_error` / `memory` / `commitment`）派生。snapshot 自身发布**typed zone bool**（`acknowledge_pressure_zone` / `repair_zone` / `direct_task_zone` / `emotional_render_zone` / `pace_pressure_zone` / `cold_rapport_zone` / `low_directness_zone` 等），消费者读 zone bool 即可，**不得**重新应用数值阈值。阈值常量集中在 `volvence_zero.interlocutor.contracts.InterlocutorThresholds`，是该 owner 的 SSOT。详见 [`docs/specs/interlocutor-state.md`](./specs/interlocutor-state.md)。

**`active_mixture` 契约语义**（Behavior Protocol Runtime packet 1.0+，SHADOW）：当前轮激活的 `BehaviorProtocol` 集合 + 各自的 `activation_weight` + 跨协议 `boundary_union_ids`（packet 1.2 起仅发布 IDs，不发布内容本体；canonical 内容由既有 application owners 持有）。owner 是 `vz-application.protocol_runtime.ProtocolRegistryModule`（packet 1.0 在 `vz-cognition` 立，packet 1.2 迁至 `vz-application` 因为 compile 路径产出 application 层 `BoundaryPriorHint`），`default_wiring_level=SHADOW`；ACTIVE 升级被 `FallbackActivationActiveError` fail-loud 守门（packet 1.0.1）。Packet 1.2 起 `load_protocol(bp)` 自动把 `bp.boundary_contracts` 编译为 `BoundaryPriorHint` 并 upsert 到 `ApplicationRareHeavyState`；hint id 用 namespace `protocol:{protocol_id}:boundary:{boundary_id}` 携带 lineage。

**ProtocolRuntime 不是内容 owner**——这是 R8 SSOT 关键纪律：

- ProtocolRuntime 持有的是**协议元数据**（id / version / source / activation_weight / phase_id）和**reviewed-prior 引用**（`BoundaryContract` / `StrategyPrior` / `TemporalArc`），不是 boundary / strategy / case / domain knowledge 的 canonical store。
- canonical 内容仍由 `boundary_policy` / `strategy_playbook` / `case_memory` / `domain_knowledge` 拥有；`BehaviorProtocol` 通过 packet 1.2+ 引入的 **compile 路径**编译进这些既有 owner（与 `DomainExperiencePackage.compile_to_application_owners()` 同形）。
- packet 1.0 SHADOW 期间 `ActiveMixtureSnapshot.boundary_union` 直接发布完整 `BoundaryContract` tuple，**仅供 SHADOW dual-run diff 测试**；packet 1.2 之前必须把消费者侧改为读 IDs + 权重 + reviewed-hint，让 canonical 内容仍走既有 owner。
- **vitals 边界**：`drive_value` 信号源在 spec 里被 deferred；kernel `ProtocolRegistryModule` **不直接读 lifeform-side 的 `VitalsSnapshot`**（违反第 43 行的"vitals 不进入内核 §6 注册表"）。如果未来确实需要 drive coupling，需要新增一个 kernel-side 的 typed `DriveReadoutSnapshot` adapter 进入 propagate 图（owner 待定，需要 R8 review）。

**SHADOW → ACTIVE 升级 checklist**（spec §packet 1.0 fallback）：当前 ActivationController 是 fallback 实现（`identity_gate=1.0` 硬编码 / equal-weight / lexicographic 仲裁）。promotion 必须：

1. 接 R7 Self trait gate + R14 regime identity 真实交叉检查（packet 1.3+）
2. 接 PE history utility（packet 1.5+）
3. 接 typed context match signals（packet 1.5+）
4. 替换 lexicographic 仲裁为 PE-driven 后验仲裁（packet 1.5+）
5. 至少有一个 ACTIVE consumer 通过 matched-control dual-run 测试（packet 1.2+）

详见 [`docs/specs/protocol-runtime.md`](./specs/protocol-runtime.md)。

**`RegimeIdentity.expression_brief` / `application_brief` 契约语义**（W3+W4 ssot-cleanup-p0-p4）：每个 regime 的 `RegimeIdentity` 携带两个 typed brief：
- `expression_brief: ExpressionBrief` — `lifeform-expression` 渲染层读取，把 `acknowledge_hint` / `frame_hint` / `next_step_hint` / `open_loop_hint` / `continuity_hint` 五个语义占位符当 lookup key，渲染最终 prose。
- `application_brief: ApplicationBrief` — `vz-application` 读取，包含 `task_focus / support_focus / repair_focus / exploration_focus`（连续 0..1 mode 强度）、`domain_affinity`（per-domain 加分表，替代 `_regime_bonus(regime_id, {...})`）、`continuum_target_position`、`decision_kind_hint`、`support_decision_threshold`、`knowledge_weight_nudge`。`vz-application` 模块**不得**直接 branch 在 regime_id 字符串字面量上；contract test `tests/contracts/test_application_no_regime_id_branching.py` 守门。新增 regime 只需在 `volvence_zero.regime.templates.REGIME_TEMPLATES` 加一行 brief，application 自动 pickup。

### 6.1A Semantic Owner Emotional Decision Readouts

以下字段由 semantic owner 自身从 typed proposals / owner records 聚合发布；消费者不得从 `description` 或 response 文本重建这些状态：

| Snapshot | Owner-side readout fields | 主要消费者 |
|----------|---------------------------|------------|
| `UserModelSnapshot` | `preferred_support_pacing`, `decision_style`, `overwhelm_pattern_strength`, `durable_goals` | dual_track, response_assembly, evaluation |
| `CommitmentSnapshot` | `due_followup_count`, `stalled_commitment_count`, `recent_completion_count` | followup, response_assembly, evaluation |
| `OpenLoopSnapshot` | `oldest_open_turn`, `stale_loop_count`, `confirmation_debt_count`, `closure_readiness` | response_assembly, evaluation, session-post evidence |
| `RelationshipStateSnapshot` | `emotional_load`, `repair_need`, `trust_delta`, `attunement_gap`, `stabilization_need`, `recent_repair_count`, `unresolved_tension_count`, `attunement_trend`, `trust_recovery_signal`, `relationship_continuity_score`, **W2-A**: `cumulative_trust_level` (long-horizon integrated trust 0-1), `relationship_age_turns` (current_turn − first_record_turn), `funnel_stage` (typed string label: `unknown` / `prospecting` / `discovery` / `nurturing` / `recommending` / `converting` / `repurchasing`; vocabulary in [`packages/vz-cognition/src/volvence_zero/semantic_state/contracts.py`](../packages/vz-cognition/src/volvence_zero/semantic_state/contracts.py) `FUNNEL_STAGE_*` constants) | dual_track, response_assembly, evaluation, dlaas-platform-ops.OutboundScheduler |
| `GoalValueSnapshot` | `value_conflict`, `decision_readiness`, `active_tradeoff_count`, `reversibility_need`, `goal_shift_pressure`, `active_goal_count`, `deferred_goal_count`, `conflicted_goal_count`, `resolved_goal_refs`, `goal_continuity_score` | dual_track, response_assembly, evaluation |
| `BoundaryConsentSnapshot` | `autonomy_risk`, `consent_clarity`, `professional_scope_pressure`, `overreach_risk`, `active_scope_count`, `denial_count`, `revocation_count`, `external_action_blocked`, `memory_scope_status` | boundary_policy, response_assembly, evaluation |

`response_assembly.support_before_decision_pressure` 必须优先消费上述 owner-side readouts；domain/prototype 路由只能作为辅助证据。ETA / temporal 层消费的是压缩后的 action-family advisory，不拥有这些语义事实。

### 6.X Social Cognition Learning Slots（R16-R20，planned / migration log mirror）

下表是 Social Cognition Learning Layer 的 planned slot 注册表。它们必须按 `docs/implementation/15_social_cognition_layer.md` 的 SHADOW → ACTIVE → retire 协议逐步落地；在 SHADOW 期不得破坏现有 flat `user_model` / `relationship_state` / `interlocutor_state` 消费路径。

> 主契约的稳定 slot surface 以 §6 默认接线表为准。本节只保留 planned / SHADOW 迁移镜像；完整 rollout notes 与 slice changelog 迁到 `docs/CONTRACT_MIGRATION_LOG.md`，后续实现流水不再追加到本文档。

| Slot Name | Owner 模块 | Value 类型 | 依赖 | 默认接线 | Timescale | Social prediction emitted | PE consumer |
|-----------|-----------|-----------|------|----------|-----------|---------------------------|-------------|
| `multi_party_identity` | MultiPartyIdentityModule | MultiPartyIdentitySnapshot | substrate, memory, semantic proposals, scene role envelope | DISABLED → SHADOW | online-fast / session-medium / background-slow | active speaker, subject scope, audience scope, identity continuity | social_prediction_error → prediction_error / credit |
| `interlocutor_models` | MultiPartyIdentityModule + keyed semantic owner views | Mapping[str, UserModelSnapshot] | user_model, multi_party_identity | SHADOW | per turn / scene | state-to-person attribution | social_prediction_error |
| `relationship_states` | MultiPartyIdentityModule + keyed relationship views | Mapping[str, RelationshipStateSnapshot] | relationship_state, multi_party_identity | SHADOW | per turn / scene | dyad continuity / repair attribution | social_prediction_error |
| `interlocutor_states` | MultiPartyIdentityModule + readout builder | Mapping[str, InterlocutorState] | evaluation, memory, commitment, multi_party_identity | SHADOW | per turn | current interlocutor readout attribution | social_prediction_error |
| `belief_about_other` | BeliefAboutOtherModule | BeliefAboutOtherSnapshot | semantic proposals, memory, multi_party_identity, prediction_error | DISABLED → SHADOW | online-fast / session-medium / background-slow | interpretation / belief update outcome | social_prediction_error → prediction_error |
| `intent_about_other` | IntentAboutOtherModule | IntentAboutOtherSnapshot | semantic proposals, execution_result, commitment, multi_party_identity | DISABLED → SHADOW | online-fast / session-medium | follow-through / next-action outcome | social_prediction_error → prediction_error |
| `feeling_about_other` | FeelingAboutOtherModule | FeelingAboutOtherSnapshot | evaluation, relationship_states, multi_party_identity | DISABLED → SHADOW | online-fast / session-medium | affect / rapport movement | social_prediction_error → prediction_error |
| `preference_about_other` | PreferenceAboutOtherModule | PreferenceAboutOtherSnapshot | semantic proposals, memory, multi_party_identity | DISABLED → SHADOW | session-medium / background-slow | durable style / boundary stability | social_prediction_error → prediction_error |
| `conversational_role` | ConversationalRoleModule | ConversationalRoleSnapshot | multi_party_identity, host role envelope, common_ground, ToM summaries | DISABLED → SHADOW | online-fast / session-medium | addressee / subject / witness assignment | social_prediction_error → prediction_error / credit |
| `common_ground` | CommonGroundModule | CommonGroundSnapshot | multi_party_identity, conversational_role, belief_about_other, memory | DISABLED → SHADOW | online-fast / session-medium / background-slow | reference resolution / mutual-knowledge sufficiency | social_prediction_error → prediction_error / credit |
| `groups` | GroupModule | GroupSnapshot | multi_party_identity, conversational_role, common_ground, commitment, open_loop | DISABLED → SHADOW | online-fast / session-medium / background-slow | joint commitment durability / group regime fit | social_prediction_error → prediction_error / credit |
| `social_prediction` | SocialPredictionAggregateModule (lifter) | SocialPredictionSnapshot | multi_party_identity, memory.social_pe_signals（Slice 12+：其它 R16-R20 owner 自报 typed signals 后并入） | DISABLED → SHADOW → ACTIVE-when-multi-party | pre-action per turn | 把上游 owner 自报的 typed PE signals 升级为公共 SocialPrediction（不重建） | social_prediction_error |
| `social_prediction_error` | SocialPredictionErrorModule (lifter) | SocialPredictionErrorSnapshot | social_prediction, multi_party_identity, memory.social_pe_signals（计划：evaluation, execution_result, relationship_states, common_ground, groups 各 owner typed signals） | DISABLED → SHADOW → ACTIVE-when-multi-party | post-action per turn / session | 把上游 owner 自报的 typed PE signals 升级为公共 SocialPredictionError（owner 字段来自 signal.source_owner）+ 外部 probe 注入 | prediction_error / credit |

**Social Cognition migration protocol**：

1. **DISABLED**：types and docs exist; no runtime publication.
2. **SHADOW**：new social cognition slots publish alongside existing flat slots; consumers continue using old slots unless explicitly opted in.
3. **ACTIVE**：selected consumers switch to keyed/social slots; old flat slots become compatibility read models only.
4. **Retire flat path**：after evidence gates pass and rollback window expires, flat single-other assumptions are removed or pinned behind `primary` compatibility adapters.

**Social Cognition slot 不变量**：

- Every row must identify an owner, timescale, social prediction, and PE consumer before implementation.
- LLM output can only produce typed proposals; no LLM classifier owns social state.
- Renderer never reconstructs social state from text. It may only express plan / snapshot outputs.
- Social PE is a typed downstream readout into the existing `prediction_error` / `credit` path; evaluation remains readout / gate, not learning source.

### 6.1 Lifeform-side Slots（不进入 kernel slot 注册表）

下表 slot 由 lifeform 层 wheel 拥有；它们**不**进入 kernel propagate 顺序，也**不**作为 kernel owner 单写者校验目标。它们是 lifeform 与 host / service 之间的契约面，由 `lifeform-*` 包发布，供 `lifeform-expression` / `lifeform-service` / 操作员 dashboard 消费。

| Slot Name | Owner 模块 | Wheel | Value 类型 | 默认接线 | 发布频率 | 消费者 |
|-----------|-----------|-------|-----------|----------|----------|--------|
| `vitals` | VitalsModule | `lifeform-core` | VitalsSnapshot | per-vertical | SYSTEM tick + per-turn | lifeform-expression, followup_manager, prompt_planner |
| `affordance` | AffordanceModule | `lifeform-affordance`（**slice 1 落地，slice 2 执行面进行中**） | AffordanceSnapshot | N/A（slice 1 未接 runtime propagate；host 按需 `build_neutral_snapshot(registry)` 或构造 snapshot） | per-call scaffold | prompt_planner, response_synthesizer, AffordanceInvoker（slice 2） |
| `thinking_loop` | ThinkingScheduler | `lifeform-thinking`（**新建中，Phase 1**） | ThinkingLoopSnapshot | DISABLED（v0）→ SHADOW → ACTIVE | scene 内异步 | family_report metrics, debug dashboard |

**lifeform-side slot 不变量**：

- 不可被任何 `vz-*` wheel 反向 import（CI 由 `tests/contracts/test_import_boundaries.py` 强制）
- 不可作为 kernel owner 间 propagate 的输入；只能被 lifeform 层（含 expression / service）消费
- 副作用如果要进入 kernel，**必须**走已有公共入口（`BrainSession.submit_*` / `LifeformSession.run_turn`），不可旁路

### 6.1B Platform-side Slots（DLaaS 控制平面，不进入 kernel slot 注册表）

下表 slot 由新增的 `dlaas-platform-*` wheel 拥有，承担多租户治理与 ops 状态。它们**不**进入 kernel propagate 顺序，**不**被任何 `vz-*` wheel 读取。详见 [`docs/specs/dlaas-platform.md`](./specs/dlaas-platform.md)。

| Slot Name | Owner 模块 | Wheel | Value 类型 | 默认接线 | 发布频率 | 消费者 |
|-----------|-----------|-------|-----------|----------|----------|--------|
| `tenant_state` | TenantRegistry | `dlaas-platform-registry`（**Phase 1 占位**） | TenantState | DISABLED → SHADOW | CRUD 时 | `dlaas-platform-api` auth 中间件、`dlaas-platform-launcher` quota 检查 |
| `contract_state` | ContractRegistry | `dlaas-platform-registry`（**Phase 1 占位**） | ContractState | DISABLED → SHADOW | CRUD 时 / lifecycle 切换 | `dlaas-platform-launcher`（adopt / awake）、`dlaas-platform-api`（runtime 路由） |
| `instance_status` | InstanceManager | `dlaas-platform-launcher`（**Phase 1 占位**） | InstanceStatus | DISABLED → SHADOW | adopt / awake / sleep / evict | `dlaas-platform-api`、`dlaas-platform-ops` |
| `handoff_ticket_state` | HandoffQueue | `dlaas-platform-ops`（**Phase 1 占位**） | HandoffTicketState | DISABLED → SHADOW | rupture_state 快照触发 / 操作员手动 | `dlaas-platform-api`、admin SSE stream |

**platform-side slot 不变量**：

- **绝对禁止**任何 `vz-*` wheel 直接或间接读这些 slot；CI 由 `tests/contracts/test_import_boundaries.py` 中针对 `dlaas-platform-*` 的反向 import 规则强制
- 平台 owner 把 kernel 视为单向调用对象（`lifeform-core.Lifeform` facade + `lifeform-service` HTTP），从不让 kernel 知道 platform 存在
- handoff_ticket_state 的触发证据来自**读** `vz-cognition.rupture_state.RuptureStateSnapshot`，平台不在 kernel 里加任何 handoff owner
- focus_person 写入路径只走 `BrainSession.submit_profile_event`；platform 只持有 `(ai_id, person_id)` 索引，**不**持有 person belief / preference / role 副本
- identity_link 只是把 `(tenant_id, ai_id, canonical_end_user_ref)` 拼成 `volvence_zero.memory.UserIdentity.scope_key` 字符串，0 改 vz-memory schema

### 6.2 Owner 字段扩展（stable readouts + migration log mirror）

下列字段是在 spec 中冻结、Phase 1+ 逐步实施的 owner 字段扩展。它们**不**新增 slot，只在现有 owner 的 `value` dataclass 上加字段。

> 本节只记录消费者可依赖的稳定 readout。字段实施流水、planned 状态和 slice 说明迁到 `docs/CONTRACT_MIGRATION_LOG.md`。

| 现有 Slot | 新增稳定 readout | 所有者职责 |
|---|---|---|
| `memory` | `cms_band_vectors: tuple[tuple[str, tuple[float, ...]], ...]` | memory owner 发布 CMS band 向量，temporal 不再按属性名读取 `cms_state` 内部结构 |
| `case_memory` | `support_prior: float`、`task_prior: float` | case_memory owner 发布 track prior，runtime 不再遍历 `hit.track_tags` 推导 |
| `strategy_playbook` | `support_prior: float`、`task_prior: float` | strategy_playbook owner 发布 playbook prior，runtime 不再按 regime 字符串集合分类 |

**字段扩展不变量**：

- 所有新增字段必须有默认值，向后兼容现有持久化数据
- 字段添加 PR 必须同步更新本注册表
- 字段必须可以被 reflection writeback 通过 `SemanticProposal` typed path 写入；**禁止** owner 私有 setter 直接赋值

### 6.3 新增 vz-contracts 类型（stable surface + migration log mirror）

下列类型属于跨 wheel 共享的不可变契约，应当落到 `vz-contracts`：

> 跨 wheel 类型的稳定入口如下；历史 slice 说明迁到 `docs/CONTRACT_MIGRATION_LOG.md`。

| Module | 稳定类型面 |
|---|---|
| `volvence_zero.thinking` | thinking task / artifact contracts |
| `volvence_zero.affordance` | affordance descriptor schema |
| `volvence_zero.social_cognition` | social cognition contract snapshots and prediction/error types |
| `volvence_zero.environment` | environment event / outcome contracts |
| `volvence_zero.temporal_types` | `ControllerState` / `TemporalSegmentClosure` / `TemporalAbstractionSnapshot` |

---

## 7. 变更协议

### 7.1 快照格式变更

当模块内部表示变化时：

1. **只改一处**：修改模块自身的快照生成逻辑
2. **版本递增**：`Snapshot.version` 递增
3. **向后兼容**：新增字段使用 Optional，不删除已有字段
4. **破坏性变更**：需要同步更新所有消费者，在 `00_INDEX.md` 中记录

### 7.2 新增模块

1. 在本文档中注册新的 Slot
2. 定义 frozen dataclass 的 value 类型
3. 声明消费者和发布频率
4. 更新快照依赖图

### 7.3 自检清单

改代码前检查：

- [ ] 是否 import/持有了另一个独立模块？→ 改为从 upstream 读快照
- [ ] 是否在外部访问模块内部字段？→ 从快照读
- [ ] 是否在外部重写了模块的总结逻辑？→ 使用模块快照已有描述
- [ ] 快照缺信息？→ 去发布模块内部丰富快照
- [ ] 格式变了要改几处？→ 超过 1 处说明 SSOT 被破坏
- [ ] 新增的适应/学习逻辑是否在正确的所有者模块内？

---

## 8. 参考文档

| 文档 | 用途 |
|------|------|
| `docs/next_gen_emogpt.md` | R8（快照优先、契约优先）、R11（可学习的内部状态表示） |
| `docs/SYSTEM_DESIGN.md` | 系统架构设计：模块职责、数据流、分层原则 |
| `docs/prd.md` | 5.5 契约式运行时、6.1 模块间通信总线、6.4 仓库与 wheel 边界 |
| `archetecture.md` | 8 wheel 切分轴 + 替换映射 + 迁移路线 |
| `SPLIT.md` | 仓库边界 charter：Phase 1 monorepo → Phase 2 触发条件 |
| `docs/specs/lifeform-vitals.md` | always-on drive 层契约（R-PE 慢尺度源） |
| `docs/specs/environment-interface.md` | 生命体与环境之间的 Observe / Perceive / Act / Assimilate 总边界协议 |
| `docs/specs/emergent-action-abstraction.md` | ETA/NL-clean action feedback abstraction：EnvironmentOutcome 最小观察字段、temporal segment closure、PE action context、PE-derived credit、snapshot replay export |
| `docs/specs/domain-experience-layer.md` | 通用 vertical 经验包 schema 与编译边界 |
| `docs/specs/core-package-boundary.md` | core package 边界、stable Brain API、HF optional runtime |
| `docs/CONTRACT_MIGRATION_LOG.md` | planned / SHADOW slot、字段扩展与 shared type 的 rollout notes；避免本文档承载实现流水 |
| `.cursor/rules/ssot-module-boundaries.mdc` | 模块 SSOT + 快照隔离的编码规则 |
