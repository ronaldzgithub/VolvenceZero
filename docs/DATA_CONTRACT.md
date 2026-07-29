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
| **公开表征标准** | `companion-standard` | Relationship Representation Standard（Phase A1，`docs/specs/oss-relationship-representation-standard.md`）：9 类 semantic owner snapshot value 类型 + slot 注册表常量、ToM `OtherMindRecord`、`OwnerPredictionSignal`（表示部分）、`SemanticEmbeddingBackend` seam + stub、`Snapshot` 容器、canonical trajectory schema。零依赖纯 stdlib；这些类型的**唯一 SSOT**，内核经 re-export 消费 |
| 内核 contracts | `vz-contracts` | 所有 vz-* / lifeform-* 共享的 RuntimeModule / Guards / propagate 机制类型；从 `companion-standard` re-export 共享表示类型；零产品知识 |
| 内核 owner snapshot | `vz-substrate` / `vz-temporal` / `vz-memory` / `vz-cognition` / `vz-application` / `vz-runtime` | §3 列出的运行时 slot |
| 生命体侧契约 | `lifeform-core` | `VitalsBootstrap` / `VitalsSnapshot` / `DriveSpec` / `DriveLevel` / `TurnSummary` 等；不进入内核运行时 slot |
| 垂直经验 | `lifeform-domain-*` | `DomainExperiencePackage` / `VitalsBootstrap`；编译进既有内核 application owner |

当前 wheel 命名以代码为准：`prediction_error`、`credit`、`dual_track`、`regime`、`semantic_state`、`evaluation` 和 social cognition owner 均由 `vz-cognition` 承载。`vz-pe-credit` / `vz-self-model` / `vz-evaluation` 只可作为历史能力域简称，不是当前 package 名。

**Phase A1 SSOT 迁移注记（2026-07-18）**：以下共享表示类型的定义处已迁至 `companion-standard`，内核侧原模块改为 re-export（既有 `volvence_zero.*` import 路径不变）：

| 类型组 | 新 SSOT | 内核 re-export 处 |
|--------|---------|-------------------|
| `SemanticRecord` + 9 类 snapshot + lifecycle entries + outcome 枚举 + `SEMANTIC_OWNER_SLOTS` + funnel stage 词表 | `companion_standard.semantic_state` | `volvence_zero.semantic_state.contracts` |
| `OwnerPredictionKind` / `OwnerPredictionSignal`（settle 机制留内核） | `companion_standard.owner_prediction` | `volvence_zero.owner_prediction` |
| `OtherMindRecord` + kind / status 枚举（owner snapshots 留内核） | `companion_standard.social_cognition` | `volvence_zero.social_cognition` |
| `SemanticEmbeddingBackend` + stub（backend 注册机制留内核） | `companion_standard.embedding` | `volvence_zero.semantic_embedding` |
| `Snapshot` 容器（propagate / guards 机制留内核） | `companion_standard.kernel` | `volvence_zero.runtime.kernel` |

**关键不变量**：

- `vz-*` 不得 import `lifeform-*`；CI 由 `tests/contracts/test_import_boundaries.py` 强制
- `companion_standard` 不得 import 任何 `volvence_zero.*` / `lifeform_*`（`tests/contracts/test_companion_standard_no_internal_imports.py`）；内核 wheel 中只有 re-export 站点（`vz-contracts` / `vz-cognition`）可直接 import `companion_standard`（`COMPANION_STANDARD_IMPORTERS`）
- vertical 不引入新的 runtime owner，只通过 `volvence_zero.application.domain_experience` 编译进 `domain_knowledge` / `case_memory` / `strategy_playbook` / `boundary_policy` / `application rare-heavy state`
- vitals layer 的 `VitalsSnapshot` 是 lifeform-side 公共契约，由 `VitalsModule` 唯一拥有；**不**作为内核 runtime slot 出现在 §6 注册表
- `Brain` / `BrainSession` 是内核暴露给 lifeform 层的 stable facade，详见 `docs/specs/core-package-boundary.md`

详见 `SPLIT.md` 与 `archetecture.md`。

### 1.1.1 ETA offline expert-action trajectory（non-runtime）

`vz-substrate` 的 `ExpertActionTarget` 是 ETA Eq.3 离线训练轨迹的 typed
动作目标，不是运行时 snapshot slot，因此不进入 §6 slot 注册表。

| Type | 字段 | owner / consumer |
|------|------|------------------|
| `ExpertActionTarget` | `action_id: str`、`values: tuple[float, ...]`、`source: str`、`description: str` | 由环境 demonstrator / 离线数据生产者发布；`vz-temporal::MetacontrollerSSLTrainer` 只在 SSL 训练时消费 |
| `TraceStep.expert_action_target` | `ExpertActionTarget \| None` | `vz-substrate` trajectory contract 拥有；targeted trace 必须每一步完整提供，禁止 targeted / untargeted step 混用 |

契约不变量：

- train trajectory 可携带动作目标；eval / heldout trajectory 的动作目标不得进入 optimizer。
- 动作目标不包含 `beta_t`、subgoal boundary、reward、outcome 或 credit label；这些仍只用于 evaluation。
- `values` 必须非空且有限；`action_id` / `source` 必须非空。违反时 fail loudly。
- frozen residual snapshot 与 expert action target 必须一一对应；长度不一致不得截断或补默认值。
- 该契约只解决离线 Eq.3 action distortion 的数据面，不建立第二个 action-family、
  `beta_t` 或 runtime temporal state owner。

### 1.1.2 ETA proof delayed-outcome readout（non-runtime）

`vz-temporal::InternalRLProofSubgoal` 同时发布运行判定阈值
`completion_threshold` 和跨 backend 保持不变的
`nominal_completion_threshold`。前者允许 open-weight proof runtime 为可达性做
显式校准；后者是环境 owner 定义的名义任务难度，校准不得覆盖。

当 subgoal 完成时，`InternalRLDelayedCreditAssignment` 一并冻结
`alignment_score / completion_threshold / nominal_completion_threshold`，并由
owner 解释 `completion_margin = alignment_score -
nominal_completion_threshold`。ETA evidence 只能把该事后观测 margin 用作 held-out
PE 数值目标；不得从 `family_id`、`beta_t`、边界标签或 evaluation 分数重建 outcome。
该 readout 仅属于 proof/evaluation artifact，不新增 runtime slot，也不回灌
metacontroller、credit 或 evaluation owner。

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
- `action_schema: EnvironmentActionSchema | None`：可选、reviewed、outcome-free；
  只含 `schema_id / applicability_conditions / action_steps / description`
- `situation_summary: str`：可选、outcome-free 的 pre-action observable context；
  供 application background-slow semantic compression 使用，不是 reward/evaluation

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
- `CreditRecord` 以结构化字段发布 `prediction_id / segment_id / abstract_action_id / environment_event_id / environment_outcome_id / conditioning_bank_set / conditioning_bank_fingerprints`；`CreditSnapshot.recent_action_lineage_credits` 有界保留带环境 action lineage **或** typed bank lineage 的 owner view，不受 generic recent window 截断。`context` 只作人读描述，consumer 禁止解析它重建 lineage；这些字段不改变 credit math。
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
- reviewed lived action 仅从 terminal `SCENE_EVENT` 的 canonical
  `EnvironmentOutcome` 经 `ExperiencedActionEvidence` 和既有 session-post
  ModificationGate/credit gate 编译为 `CaseMemoryPriorUpdate`；tool result 与
  non-terminal outcome 禁止走该入口
- `ExperiencedActionEvidence.action_statement` 与 `outcome_statement` 分离；
  可选 `action_schema` 只能来自 environment producer 发布的正式结构；有 schema 时
  CaseMemory intervention ordering 使用其 action steps，禁止从 action/outcome 文本反推；
  `CaseActionGrounding` 只可渲染 action，禁止把事后结果泄漏回决策轮
- `ExperiencedActionEvidence.action_family_id / action_family_version /
  controller_code_digest` 是 outcome 提交时从 temporal snapshot 捕获的 action-time
  lineage；family ID 是跨 bank revision 保持稳定的 opaque controller identity，
  不是语义动作标签；`action_family_version` 是整个 family bank 的全局 revision，
  不是单个 family 的 incarnation/version。无
  `action_schema` 时 CaseMemory 只能保存 `latent-action-family:* + schema-pending`
  审计记录，`intervention_ordering` 必须为空，因此不得进入 response realization
- CaseMemory 的 action case 路由只比较 problem/user-state/risk 适用条件；
  intervention steps、case description 与 outcome 不参与相似度，避免答案泄漏进选择
- lived-action case 的跨 session 复用必须经过 application owner persistence；
  profile seed 与 slow-loop case 的 source lineage 不得混写
- multi-experience semantic candidate 由 application `ActionAbstractionOwner` 唯一拥有：
  至少两条独立 schema-free evidence 必须共享同一 temporal family ID；不同时间采集的
  evidence 允许携带不同 bank revision，candidate/promotion 取所见最大 revision 作为
  审计锚点，不得把 revision 相等误作 family 身份；
  structured decoder 只读 situation/action，不读 outcome、PE 或 evaluation。
  decoder 产出的结构合法 candidate 还必须经过逻辑独立的第二次 semantic
  generalization audit；reviewer 只读带序号的 situation/action 与 candidate，
  不读 outcome id、结果、reward、PE、credit 或 evaluation，也不得修补 candidate。
  shared structure、episode specificity、conditions reuse、steps reuse 四项必须全部
  通过且 confidence `>=0.80`，否则 owner fail closed。
  每条待聚合证据以 `CaseMemoryRecord.action_abstraction_evidence` 的 frozen typed
  payload 随 CaseMemory checkpoint 持久化；晋升以
  `CaseMemoryRecord.action_abstraction_promotion` 记录 family/version、source closure
  以及 `generalization_audit_passed / confidence / rationale`。旧 checkpoint 缺审计字段
  时可恢复为未证明状态，但不得伪装成已通过新门。
  compact `CaseMemorySnapshot` 不发布上述 owner-only payload；其重建 record 回灌 store
  时必须保留既有 typed evidence/promotion，若同 case id 携带矛盾 typed payload 则
  fail loudly，不得以压缩快照擦除 owner 状态。
  consumer 只能调用 owner 的 `pending_action_abstraction_evidence()`，不得解析
  `description`、`problem_pattern` 或 case id 重建证据。owner 会折叠内容完全一致的
  outcome，遇到同 outcome 的矛盾 payload 则 fail loudly；已有 promotion 的整个
  family ID（不只某个 bank revision）不再发布为 pending，避免跨 session 重复提案。
  `LearnedActionSchemaCandidate` 与 reviewed `EnvironmentActionSchema` provenance 隔离；
  promotion 以 `ApplicationModificationEvidence` 进入 runtime，并由正式
  `ModificationProposal(BACKGROUND) + evaluate_gate_reasons()` 决定。缺 evaluation
  snapshot 时 fail closed；evaluation 只作 gate readout，不成为学习源

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

Character chapter bake 只消费 owner-authored audit readout，不新增 runtime slot：

| audit readout | owner | value type / channel | 不变量 |
|---|---|---|---|
| external semantic event delivery | `vz-cognition` `SemanticStateStore` | `tuple[SemanticEventDelivery, ...]`，经 Brain / Lifeform facade | owner 根据正式 record id 解释 event 是否到达声明 slot；character consumer 禁止遍历 semantic snapshot 内部结构 |
| memory artifact count | `vz-memory` `MemoryStore` | `entry_count() -> int`，经 Brain / Lifeform facade | 计数由 Memory owner 发布；consumer 不重算 stratum 或访问 artifact store |
| world/self `z_t` code | `vz-temporal` temporal owners | `AgentTurnResult.track_z_t_codes` | final wiring 从两条正式 temporal snapshot 各发布一条不可变 code；只读审计，不成为第二 temporal writer |

Character behavior fidelity 不新增 runtime slot/owner。`BehaviorFidelityStimulus` 与
`BehaviorFidelityReference` 在 capture 阶段隔离；`BehaviorFidelityCapture` 绑定 source
state digest、candidate digest 和 sandbox fingerprint；`source_state_digest_verified`
区分 capture 前后实际调用只读 digest reader 的证据与调用方自报 digest。v2 capture 还原样记录
CaseMemory snapshot 已发布的 `action_grounding_source_case_id /
action_grounding_action_labels`，只用于证明最终行为来自哪个 owner record，不在 evaluation
侧重新选择或解释案例；v1 artifact 加载时显式迁移为空 lineage 且
`source_state_digest_verified=False`。
`ReviewedBehaviorFidelityAssessment` 再绑定 stimulus/reference/candidate 三个 digest，
发布五维 reviewed semantic score。
`BehaviorFidelityReport` / `BehaviorFidelityComparisonReport` 只作为 R12 evaluation artifact，
不得提交 outcome、reward、credit、memory writeback、regime payoff 或 Internal-RL transition。
评估 Lifeform 必须从 source owner checkpoint 克隆到一次性 persistence sandbox；禁止把
只读评估 turn 直接连到 baked source directory 后再用调用方传入的相同 digest 假定源状态
未变。

多经历 action-abstraction promotion 的 owner-only checkpoint payload 包含
`schema_id / action_family_id / action_family_version / source_outcome_ids /
applicability_conditions`。旧 checkpoint 可恢复为空 conditions，但 learned promotion
在 turn-time 必须 fail closed。CaseMemory-owned applicability evaluator 只消费当前
Memory 语境、schema id、typed conditions 与 record risk markers；它不拥有行动、不产生
学习信号，且禁止读取 intervention ordering、outcome、PE、credit 或 evaluation。

`BehaviorFidelityMatrix` 是 character vertical 发布的 evaluation-only 不可变工件，不是
runtime snapshot。v1 shape 为：

- suite metadata：`schema_version / suite_id / character_id / target_schema_id /
  source_chapter_ids / reviewed_by / description`；
- frozen thresholds：四类 required counts、正例最小 promotion hits、非正例最大误触发、
  每例最低 fidelity、正例平均 baked-cold delta，以及 source-digest/no-feedback/
  competing-family 三个强制门；
- matrix case：`kind / promotion_expectation / expected_behavior_family`，以及相互 digest
  隔离的 `BehaviorFidelityStimulus / BehaviorFidelityReference` 和 reviewed rationale。

该工件只冻结评估分布和 acceptance，不选择 runtime action、不发布 reward，也不回灌
PE、credit、memory、regime 或 Internal RL。loader 对多余字段、错误类型、类别计数、
stimulus/reference binding 和 promotion semantic role mismatch 均 fail loudly。

`BehaviorFidelityArmReport` 是同一 evaluation-only 链的派生工件，按 frozen matrix 的
`PromotionExpectation` 与 capture 中已发生的 `target_promotion_used` 发布
`TP / FP / FN / TN`，以及 promotion precision、recall、specificity。无预测正例时
precision 必须为 `None`；consumer 不得把 undefined precision 改写为通过，也不得把这些
applicability 指标解释成 behavior-fidelity score 或学习信号。

`BehaviorFidelityCaseObservation / BehaviorFidelityArmReport /
BehaviorFidelityCausalAblationReport` 是同一 evaluation-only 域的只读工件，不新增 slot。
每个 observation 必须绑定 matrix digest、arm、case id/kind/promotion expectation、
CaseMemory 已发布的 grounding lineage、source digest/no-feedback 证明；reviewed fidelity
与 competing-family match 可暂缺，但缺失时对应 gate 必须显式为 `insufficient_data`。
四臂 coverage 必须精确覆盖 `baked / cold / no_rl / shuffled_lineage × 全部 matrix cases`，
重复、缺失、额外 case 或 digest/semantic binding 错配均 fail loudly。

因果报告分别发布 `lineage_causal_supported` 与 `behavior_causal_supported`，禁止用前者
替代后者。初始 v1 baseline 的 no-RL arm 形成 target promotion，冻结了 producer
lineage 缺口；正式链修复后 baked/cold/no-RL/shuffled 正例命中为 `4/0/0/0`，
`no_rl_target_promotion_absent=pass`，整体为 `lineage-causal-diagnostic-pass`。
reviewed fidelity 缺失时 `behavior_causal_supported` 仍为 false/`insufficient_data`。

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
- onboarding-arc playbook 通过 `GrowthAdvisorStrategyPrior.applicability_scope=("funnel:*", ...)` 携带 funnel/regime 漂移；关系阶段（icebreaker / baseline / empathy / pain mining / rapport / targeted advice / summary）由 `BehaviorProtocol.TemporalArc.progression_signals`（PE-driven）路由，**不**按日历天数硬切，**不**通过用户原文关键词匹配（calendar-day routing 已于 2026-05-14 移除）
- 4 大需求挖掘 funnel（height / immunity / nutrition / vision-brain）通过 `applicability_scope=("funnel:X", ...)` 编码，下游 owner 通过 scope 匹配做 turn-级路由
- 跨 vertical 隔离：`lifeform-domain-growth-advisor` 不 import `lifeform-domain-character` / `lifeform-domain-figure`，反之亦然；CI 由 [`tests/contracts/test_import_boundaries.py`](../tests/contracts/test_import_boundaries.py) 三对 parallel pair 强制
- `lifeform-service.verticals.discover_verticals()` 通过 `_try_growth_advisor` 软发现，未安装 wheel 时静默跳过，已安装时暴露为 `name="growth_advisor"`

### 2.17 MCP Bundle Bridge Contract（mcp-tools-bundle-bridge packet）

**所在 wheel**：`lifeform-mcp-bridge`（lifeform-side；不进 kernel slot 注册表）；外部 repo 作为 git submodule 引入主项目（默认绑定 `external/vz-bundle/`）

```python
@dataclass(frozen=True)
class MCPServerSpec:
    name: str                                # safe id; descriptor name prefix
    transport: Literal["stdio", "http"] = "stdio"
    command: tuple[str, ...] = ()            # for stdio
    url: str = ""                            # for http (planning stub)
    env: Mapping[str, str] = ...             # extra subprocess env
    safety_manifest_path: str = ""           # required, path to .vzbridge.yaml
    autostart: bool = True
    restart_policy: Literal["never", "on_crash", "always"] = "on_crash"
    call_timeout_seconds: float = 30.0
    enable_resources: bool = True
    enable_prompts: bool = False

@dataclass(frozen=True)
class SafetyManifestEntry:
    tool_name: str
    when_to_use: str                         # >= 50 chars
    when_not_to_use: str                     # >= 50 chars
    cost_model: AffordanceCost
    safety_model: AffordanceSafety
    affordance_tags: tuple[str, ...] = ()
    excluded: bool = False

# Typed errors:
#   MCPBridgeError (base)
#   MCPServerSpawnError
#   MCPConnectionLostError
#   MCPCallTimeoutError
#   MCPProtocolError
#   MCPMissingSafetyManifestError
#   MCPSafetyManifestSchemaError
```

**MCP wire endpoints consumed by the bridge**: `initialize` / `tools/list` / `tools/call` / `resources/list` / `resources/read` / `prompts/list` / `prompts/get`. Stdio JSON-RPC 2.0 hand-rolled in `lifeform_mcp_bridge.client.StdioMCPClient` (no `mcp` SDK dependency required). HTTP+SSE is reserved for a future packet.

**Bridge translation tables**:

| MCP source | Maps to | Path |
|---|---|---|
| `tools/list` entry + manifest entry | `AffordanceDescriptor` (name = `<server>.<tool>`) | `MCPAffordanceAdapter.populate_registry` |
| `tools/call` payload | `AffordanceBackend` invocation result | bridge-bound async backend |
| `resources/list` + `resources/read` | `IngestionEnvelope` (CORPUS, FORCED) | `MCPResourceAdapter.fetch_envelopes` |
| `prompts/list` + `prompts/get` | reviewed knowledge event | `MCPPromptAdapter.fetch_prompt_events` |
| `<repo>/eval-scenarios/*.json` (no RPC) | `MCPEvalScenario` | `EvalScenarioLoader.load_scenarios` |

**关键不变量**：

- MCP server 不是 owner；`AffordanceRegistry` 是 lifeform-side 单 writer，bridge 只是给它喂 reviewed descriptor
- safety_model / cost_model / when_to_use(>=50) / when_not_to_use(>=50) 必须来自 reviewed `.vzbridge.yaml`，缺则 `MCPMissingSafetyManifestError`
- bridge wheel 禁止反向 import `volvence_zero.{cognition,memory,temporal,substrate,application,runtime}.*`（contract test [`tests/contracts/test_mcp_bridge_import_boundary.py`](../tests/contracts/test_mcp_bridge_import_boundary.py)）
- `LifeformConfig.mcp_server_specs: tuple[MCPServerSpec, ...] = ()` + `LifeformConfig.mcp_bridge_wiring: WiringLevel = WiringLevel.ACTIVE`（默认 ACTIVE，与 owner_hydration 同纪律；空 specs 是 no-op）
- MCP server crash 不能让主进程崩溃；`AffordanceCandidate.blocked_reason="mcp_unavailable:<server>"` + 后续调用 `BACKEND_FAILED`
- MCP-supplied tools 与 in-process tools 共享同一 `AffordanceModule` z_t scoring；descriptor name 经 SHA-256 hash 投影，没有"MCP 优先"硬路由

详见 [`docs/specs/mcp-bridge.md`](specs/mcp-bridge.md)。Acceptance：6 个测试 + 外部 bundle template 自带 CI。

### 2.18 Digital-Employee Domain Packages（B2B 数字员工 vertical org/twin + 行业 overlay 契约）

**所在 wheel**：`lifeform-domain-digital-employee`（schema 与数据）

B2B 数字员工产品的两个 persona 形态由该 vertical 唯一拥有 (R8)，全部编译进既有 application owner，**不**新增 kernel runtime owner：

| 构件 | 编译目标 | 说明 |
|---|---|---|
| `build_digital_employee_org_package()` | 四个既有 application owner | 公司级 OrgAgent：SOP grounding / intake triage / delegation brief / compliance guard regime priors |
| `build_digital_employee_twin_package()` | 四个既有 application owner | 成员级 EmployeeTwin：task execution / drafting / clarification / escalation regime priors |
| `IndustryProfile`（frozen dataclass） | 同上（additive overlay） | 行业 overlay：knowledge / case / playbook / boundary records，经 `build_industry_package(profile, role=…)` 叠加到 role base |
| 内置行业 profiles（`profiles/`） | 数据注册表 | `sales-sdr` / `customer-support` / `content-editor`；新增行业 = 新增一个数据模块，零代码分支 |

**Digital-employee 不变量**：

- org / twin / 行业差异**只**通过数据表达（`applicability_scope` tag 如 `industry:sales-sdr`、regime id、intervention ordering）；禁止 keyword→behaviour 映射（`no-keyword-matching-hacks.mdc`）
- 行业 overlay 严格 additive：base record 全部保留，与 base 的 id 冲突 fail loudly；base boundary gate（不可逆 / external-spend / external-publish 必须 human gate；finance/tax 领域 refuse-and-refer）不可被 overlay 移除
- wheel 不携带租户数据：公司 SOP / 品牌 corpus 经 BFF `observe` envelope 运行时进入；成员习惯存活于 `membership_id`-scoped memory（R14）
- v0 复用 companion calibration basin（vitals + temporal + regime bootstraps）；行为差异由 data-only `DomainExperiencePackage` 承载，待专属 super-loop 产出 org/twin bootstraps 后仅 `builder.py` 的 `_load_*_bootstrap` 调用变更
- `lifeform-service.verticals` 注册 `digital-employee.org.v0` / `digital-employee.twin.v0` 两个 runtime template；wheel 缺失或 `VZ_DIGITAL_EMPLOYEE_FORCE_COMPANION=1`（D18 rollback pin）时回退 companion factory，两个分支均打 `[verticals] … resolution=…` stderr breadcrumb，回退不允许静默

---

### 2.19 Digital-Ant Embodiment（数字蚂蚁非语言 substrate，研究测试床）

**所在 wheel**：`vz-embodiment-ant`（独立 owner，**不**新增 kernel runtime slot）

数字蚂蚁是一个非语言的 2D 感觉运动 embodiment，用于在**完全不涉及 LLM/token** 的情况下独立检验
R2 / R3-R4 / R5-R6 / R-PE / SSOT 是否成立。它通过既有 `SubstrateAdapter` 契约接入内核，
**复用**现有 `substrate` slot（§3.1），不引入新的 kernel owner。

| 构件 | 归属 | 说明 |
|---|---|---|
| `AntSubstrateAdapter` | `vz-embodiment-ant` | 发布标准 `SubstrateSnapshot`：`residual_activations`（layer 0 = 冻结感知向量）+ `residual_sequence` + `feature_surface` |
| `sense_encode` / `motor_decode` | 同上 | 两个**冻结**向量函数（纯 numpy，无可学习参数），对应遗传固定的受体映射与运动 plant |
| `AntNavigator` | 同上 | body 侧环形吸引子朝向 + 路径积分（对应中央复合体，冻结、不学习） |
| `AntSession` | 同上 | 经 `vz-runtime` 的 `AgentSessionRunner` facade 复用内核，每 tick 一个闭环 |
| 信息素快照总线（Phase 1） | 同上 | embodiment 内部 colony 总线，多写者只追加带衰减；`PheromoneField` 自己发布 home/trail mass + normalized entropy；**不**进入 §3/§6 kernel 注册表 |
| 障碍几何与碰撞证据 | 同上 | `AxisAlignedObstacle` 由 environment owner 持有；substrate 只见局部左右触角/contact，`WorldTransitionEvidence` 发布 blocked/applied-step 审计事实；**不**新增 kernel slot |
| ecology-v2 对象快照 | 同上 | frozen `ButterSource / WoodStick / BurningMatch` 由 `AntWorld` 唯一持有；仅发布 `WorldObjectSnapshot`（几何、owner 计算的 effect radius、description）给 App |
| `ant-sense.ecology-v2` | 同上 | 保留 `ant-sense.v1` 14 维前缀并追加 5 个局部热感通道；runtime facade 将完整 19 维声明为 temporal `n_input`，由 encoder 压缩到独立的 latent `n_z`，禁止按 `n_z` 截断；对象坐标、木棍方向和推荐动作不进入 controller |
| ecology local valence | `AntWorld` environment owner | `WorldTransitionEvidence` 发布动作前后 local food/heat signal 与离散 milestone/contact；携食返巢 shaping 使用 body-side path-integration home-distance progress，不使用外部坐标或 home pheromone；`AntSession` 仅压成有界 `EnvironmentMeasurement.action_payoff`，不得发布坐标、目标方向或推荐动作。PE owner 负责 actual-outcome 归一化与 mismatch；runtime replay 只优化 PE owner 发布的 realized action payoff，不把 signed prediction residual 当 realized utility |
| runtime beta segment replay | `vz-temporal` joint-loop owner | `internal_rl_runtime_segment_credit=ACTIVE` 时把 lineage-matched real replay 按真实 switch、milestone/terminal 或有界 horizon 闭合为多步 rollout，复用既有 GAE 与 pending queue；Digital Ant 的 24-turn episode 使用 16-transition horizon，必须为同 episode 的后续 scheduled optimizer 留出消费窗口，禁止让无 milestone 的 open segment 在跨 episode replay-excluded checkpoint 时静默丢失；capture/transition 必须持久化实际 posterior sample scale（历史 encoder `0.5`、显式 exploration `1.0`）；启用 action-head 镜像等变时还必须持久化 owner 编码的 mirror state，pure/torch 不得重建或省略该 lane；open segment 进入 owner checkpoint/rollback，不新增 slot/ledger；`joint_loop.learning` persistence schema 为 v5。该 schema 的 world/self 是两条独立 lane（`world_temporal_snapshot` / `self_temporal_snapshot` 与两条 sandbox `metacontroller_snapshot`），`rollback_rare_heavy_import` 逐条恢复，因此**两轨禁止共享同一个 `MetacontrollerParameterStore`**：共享时 world lane 会被随后恢复的 self lane 覆盖，checkpoint 不再 round-trip。owner 在 `ETANLJointLoop` 构造时对共享 store fail loudly；需要第二轨的调用方必须用 `clone_temporal_policy()` 克隆（保留源实现模式，`LEARNED_LITE` 不提升为 `FULL_LEARNED`）。schema 版本不变，这是恢复路径对称性的修复而非 payload shape 变更 |
| runtime replay reward eligibility | `vz-temporal` Internal-RL settlement owner（`InternalRLSandbox.settle_runtime_action`），经 `ETANLJointLoop` 声明与转发，runtime 门面 `FinalRolloutConfig` | typed `RuntimeReplayRewardEligibility`：`any-settled-outcome`（默认，逐字节回滚——每条 lineage 匹配的结算都拿 PE owner 发布的 `ActualOutcome.action_payoff`，环境未发布 measurement 时该值是 PE owner 合成的 action 轴）/ `environment-measured-only`（严格——只有 `EnvironmentOutcome.measurement.action_payoff` 非 None 才有资格；ineligible transition 仍完整结算并保留 lineage、动力学与 PE 残差诊断，但 `realized_action_payoff` 与 `segment_bonus` **同时**为 0，禁止 PE 派生 segment credit 把同一份量偷渡回来）。内核只读 measurement 的存在性与该字段，**不解释任何领域语义**。逐条审计标签发布在 `ZTransition.runtime_reward_eligible` + `runtime_reward_eligibility_reason ∈ {eligible, ineligible:no-environment-measurement, ineligible:no-environment-action-payoff}`。无 environment measurement 的域（语言 companion）必须保持默认值，否则 reward 流恒零。数字蚂蚁 evidence profile 声明严格模式（其任务路径按 `digital-ant-embodiment.md` §4 禁止 distance/potential shaping）。不新增 slot |
| PE drive / outcome payoff 拆分 | 同上 | `external_prediction_error_drive`（= `runtime_replay_prediction_error_enabled`）**只**表示 (a) PE 驱动学习信号：PE 派生 segment credit bonus + PE→`beta_t` **加性** switch prior，PE 仍是 readout。该 prior 数值 `min(0.18, strength·tanh(max(0, magnitude − floor)))` 对边界决策与 `is_switching` **没有任何决定权**：曾经的"floor 交叉即 boundary request"语义在产生任何 journal 之前被 v30 冻结重放测量整体否定（日常拍 PE p50 0.508 与事件 PE 重叠、自然拾取下一拍 ~0.32，`scripts/measure_ant_pe_boundary_margin.py`），幅度阈值不构成事件检测器；边界改由下行"typed environment milestone boundary"契约拥有（`test_pe_magnitude_is_inert_for_boundaries_and_milestone_owns_them` 钉住）。floor 通用默认 0.5，蚂蚁 profile 不再覆盖（0.45 "标定"已随测量撤下）。(b) "环境发布的 outcome payoff 是否到达 optimizer" 由独立契约 `FinalRolloutConfig.internal_rl_runtime_outcome_payoff_reward: bool \| None` 拥有：`None`（默认）由 eligibility 推导——`any-settled-outcome` 下跟随 (a)（逐字节复现历史 PE-off = reward-off 行为），`environment-measured-only` 下恒 `True`（该 payoff 按构造是环境发布量、非 PE 派生量，PE-off 臂必须继续获得）；显式 `True`/`False` 覆盖推导。`AgentSessionRunner` 对 config 与外部注入 joint-loop 的三项声明不一致 fail loudly |
| typed environment milestone boundary | 声明：environment owner（`EnvironmentMeasurement.discrete_milestone`，`vz-contracts`）；转发：`AgentSessionRunner`（typed 信号 `environment_milestone_boundary`）；解析：`vz-temporal` temporal owner；门控：`FinalRolloutConfig.environment_milestone_temporal_switch`（WiringLevel，默认 DISABLED） | R-PE 结构化边界：哪个事件关段是**类型化 readout**，不从原始幅度反推。环境 owner 在 outcome 的 measurement 上声明离散任务里程碑（蚂蚁：仅 pickup/delivery；稠密 local-valence/进度拍禁止声明；heading-stability 等逐拍 `task_progress` 域不受影响——内核**只读该布尔声明**，不从 `task_progress` 是否存在做推断）。orchestrating session 在下一 turn（该 outcome 结算的 turn）把声明转发为方向无关、单拍有效的信号；joint-loop 在 ACTIVE 下记录 `record_external_boundary_request`，temporal owner 相对当前 learned `beta_threshold` 解析并强制 effective beta ≥ threshold（threshold 校准不能永久屏蔽已确认里程碑）；SHADOW/DISABLED 不切段，DISABLED 为逐字节回滚。请求 turn-scoped：唯一写者是每 turn 信号刷新，`reset_episode_runtime_telemetry` 不清除（SSL 族发现在 full-cycle turn 决策前调用该重置，清除会静默丢弃学习 turn 的里程碑）。PE-off 匹配对照臂保持该通道 ACTIVE（环境事实，非 PE readout）。蚂蚁 profile ACTIVE 并把 wiring 记入 `AntStepRecord.backend_wiring`；不新增 §3/§6 slot |
| optimizer reward stream 发布 | `vz-temporal` joint-loop owner | 不可变 `RuntimeReplayRewardStream`，经 `ETANLJointLoop.latest_runtime_replay_reward_stream` 发布、`AgentSessionRunner.latest_runtime_replay_reward_stream` 只读透出。字段：`eligibility_contract / outcome_payoff_reward / prediction_error_reward_enabled / settled_transition_count / eligible_transition_count / ineligible_transition_count / nonzero_reward_transition_count / nonzero_realized_payoff_transition_count / nonzero_segment_bonus_transition_count / realized_action_payoff_sum / segment_bonus_sum / reward_sum / eligibility_reason_counts / last_eligibility_reason / description`。这是 optimizer **实际消费**的 reward，与环境侧 measurement 计数器（在泄漏 tick 上恰为 0）和 PE signed residual 都不是同一个量。readout-only，不回灌学习；进程内累计，不进 rare-heavy checkpoint、跨 session 不续接；不新增 slot |
| runtime replay latent clamp | `vz-temporal` Internal-RL owner | sandbox 的 signed `[-1, 1]` clamp 对 reward/advantage 正确，但同一函数也用于 latent code / modulated mean / candidate mean / policy mean 重建，而在线 owner 把 `z_t` 限制在 `[0, 1]`；action-head 残差为负时 replay lane 会重建出冻结 plant 无法输出的 mean。`FinalRolloutConfig.internal_rl_runtime_latent_unit_clamp`：`False`（默认）保持历史 signed 边界=精确回滚，`True` 只对 latent/mean 重建改用 `[0, 1]`。**reward / advantage clamp 边界在任何取值下都不变。** torch PPO backend 尚未消费该契约（见 spec 已知缺口）|
| causal z-policy action head | `vz-temporal` metacontroller/Internal-RL owner | 通用 factorized `causal_action_head_state -> bounded z_t residual` 参数面；默认保留历史低秩，部署 profile 可在 owner 学习开始前请求不超过 `n_z` 的 rank，已学习 live mapping 禁止原地改 rank；full-rank 使用 identity input factors 与全零 output/bias，消除第二次随机压缩但不引入动作 prior。state 由同一 Ndim encoder 参数对当前 observation 做零 recurrent preimage 编码，覆盖完整 `n_input` 并发布为 signed `[-1,1]`，live/pure/torch replay 与 open-segment persistence 必须使用同一值，禁止以历史依赖的 serving hidden 替代；前向/反向不得二次执行 `[0,1]` 重心变换；可选 `effective_dims` 由冻结 actuator 的 embodiment profile 在启动时声明，`None` 保持全维兼容，显式值必须非空/唯一/界内，pure/torch gradient 与 live/sandbox residual 对非支持 output row 严格为零，禁止 temporal owner 硬编码业务 motor 语义；常数截距保留 owner 学习率 `0.12` 倍、状态路径 `0.05` 倍、单步 `0.01`、总幅度 `0.1`，batch mean 只进入该截距；factor 使用 owner 基础学习率并只消费 centered state covariance，与 torch path 尺度对齐；output factor 保持零初始化，首个非零 covariance batch 先计算 bounded candidate output、再按其真实列范数回传 input，最后原子提交；ACTIVE runtime replay 的 batch target 按真实 transition 数计，部署 profile 必须避免 singleton batch；`DISABLED/SHADOW/ACTIVE` 可回滚门控，禁止对象字段与 motor command；参数进入 owner snapshot、canonical archive、fingerprint 与事务 rollback，不新增 runtime slot |
| runtime exploration context | `vz-temporal` metacontroller owner | 调用方可提供不透明、非语义的 context；owner 只保留 SHA256 摘要并纳入 coherent option identity，不保留或解释原文。缺省保持历史全局序列精确不变；Digital Ant 以 episode seed + body offset 分散序列，matched arms 的同 episode/body context 必须相同；不新增 slot 或 checkpoint 字段 |
| typed task measurement | `vz-contracts` / PE owner | `EnvironmentOutcome.measurement` 仅含环境可观察事实；runtime 保留 lineage，PE 是唯一 mismatch owner |
| opaque learning archive | `vz-runtime` facade | owner 发布 `OwnerPersistenceSnapshot`；`agent-learning-archive.v2` 以 strict canonical JSON 绑定逐 owner schema/payload sha256、整体 state fingerprint，`agent-learning-checkpoint-collection.v1` 再绑定 sense schema / input dim / latent dim / ant count，外层 ecology bundle 为 `digital-ant-ecology-checkpoint.v4`；禁止 pickle/object hook，跨 episode 显式排除未结算 runtime replay，embodiment 只存取 bytes。通用 session owner hydration 同样保存/恢复 `joint_loop.learning`（schema v5、runtime replay excluded）与 `reflection.consolidation_score`；world/self temporal 仍禁止成为第二 hydration writer，其状态只能作为 joint-loop owner archive 的两条独立 lane 存在 |
| ecology curriculum evidence | `vz-embodiment-ant` offline experiment owner | `digital-ant-ecology-curriculum.v14` 发布 mastery/interleaved 训练日程（butter/burning-match/composite 三阶段，木棍是中性物理几何、无 contact mastery/payoff）、P1 forced-return bootstrap（每个 body 在巢外专属黄油源上从未携食状态起步，经真实 contact 形成 `carrying_food: False→True`，再以左右均衡 `±3π/4` heading 训练拾取触发的动作族切换与归巢；后半程每 3 个 primary layout 交错复习一次，共 5 次；只初始化环境状态并同步 body-side PI，不发布坐标、目标方位或动作标签）、P1 forced-approach bootstrap（butter-near 专用：body 生成在拾取盘外、朝向偏离食物方位，生成半径 `1.45–2.9×拾取半径` 与偏离角 `0.4π–0.8π` 由 layout seed 逐 body 随机——固定生成环可被单一"固定曲率轨道"非定向解收割（首次 v22 实测 base policy 把基线转向从 0.083 放大到 0.15 rad 收割整块），随机 ensemble 下唯一通解是梯度转向；每次抽样仍保证直线路径最近距离 ≥1.38×拾取半径、必然错过拾取盘；只初始化状态并同步 PI，不发布坐标/目标方位/动作标签。动机：near 拾取盘的普通布局对食物梯度转向压力不足，food→turn 转向从 v10 至 v21 从未获得训练压力）、training/validation/held-out split、独立 scenario metrics、paired action probes（food/heat/home 发布方向对齐 truth，obstacle 只作 input-reachability 诊断）与 frozen gates；Digital Ant evidence profile 显式请求 `rank=n_z=16`、冻结 plant 支持的 `effective_dims=(0,1,2)`、opponent-coded actuator subspace `contrast_pairs=((0,1),)` 及 `exclusive_steering=True`（R2 所有权转移：base 确定性均值在 contrast pair 上被互补投影为 common mode，head 是 contrast 轴唯一学习型写入者，base 保留速度 common mode；见 temporal-abstraction spec。动机：v22/v22r 固定+随机 forced-approach 双重受控实验证明信用竞争下无约束 base 总用非定向"放大基线转向"退化解吸走转向信用，head 增益钉死 ≈1e-3 不增长）；temporal owner 必须在 live forward/sandbox/pure/torch 四条路径共用 head 与 base 两个投影，禁止 actuator-null common mode 吸收信用；且 `beta_t` 门必须按 contrast pair 共享（opponent-coded pair 是一根执行器轴，逐维门控会从"候选与旧码之差"凭空造出 contrast，实测零参数 head 因此仍产生 ±0.005 rad 转向并掩盖同量级学习信号）。exclusive steering 下冷启 head 精确为零、确定性策略无转向，因此 P1 pretraining 探针门只验 `input_reachable`（管线可达），转向能力由训练后 `paired_action_sensitivity`/`food_steering_alignment`/`carrying_home_action_alignment`/`post_pickup_uturn_progress` 硬门验收——冻结 U-turn lane 还必须在拾取后的前 2 个 action 内出现 `is_switching`，再满足交付或持续巢距下降；evaluation 只读且不回灌学习。P1 报告 schema 为 `digital-ant-ecology-p1-development.v31`（绑定 curriculum v14 的 typed milestone boundary），可恢复 journal 为 `digital-ant-ecology-p1-progress.v28`；旧 journal 必须 fail loudly，禁止把算法版本混在同一实验；长程 checkpoint 前由 Memory owner 把 explicit artifact 层有界到 8192 entries（CMS learned state 不裁剪，entry/index/pending/attribute 原子一致）；评估只读 owner snapshots，不回灌 reward |
| ecology sense reflection transform | `vz-embodiment-ant` frozen substrate owner | `ant-sense.ecology-v2` 发布完整 19 维 signed involutive permutation：左右 receptor 交换，有方向的 pseudoscalar 取反，标量保持；`FinalRolloutConfig.internal_rl_causal_action_head_input_mirror_permutation/signs` 缺省 `None`（DISABLED/回滚），Digital Ant ecology profile 显式 ACTIVE。`vz-temporal` 只能执行并校验该正式交换，不得 import Ant schema 或重建感觉语义；live/pure/torch 共用 `0.5·(f(s) ± f(mirror(s)))` 群投影，runtime state/capture/transition/open segment 同时发布/持久化 mirror state。`joint_loop.learning` schema 为 v5；当前 P1 report/progress 为 `development.v31/progress.v28`，旧代 progress 必须按其各自 schema fail loudly |
| `ColonyRareHeavyBundle` | `vz-embodiment-ant` | per-individual artifact digest/provenance/gate verdict；不含 temporal state、不新增 slot |
| evidence manifest | `vz-embodiment-ant` | `digital-ant-manifest.v2` sidecar 绑定 artifact/input digest 与运行 provenance |
| realtime app DTO | `vz-embodiment-ant` | `digital-ant-app.v2` 的 frozen config/frame/status/command/disturbance；新增 typed object upsert/move/remove、`AppFrame.objects` 和 checkpoint provenance；**不**进入 §3/§6 slot 注册表 |

**Ecology evidence 当前代际（2026-07-28）**：curriculum owner 的现行 schema 为
`digital-ant-ecology-curriculum.v13`；P1 report/progress 为
`digital-ant-ecology-p1-development.v30` / `digital-ant-ecology-p1-progress.v27`；
P2 confirmatory/shard/progress 为 v7。v11 将 forced-return 冻结为左右均衡的
`±3π/4` large-angle start：它保留 v10“零转向不能领取正 home-progress”的约束，同时覆盖
自然拾取后接近 `π` 的返向压力；精确 `π` 因侧向符号退化而不使用。所有旧 journal/report
必须 fail loudly，禁止跨代恢复或聚合。v12 新增冻结 `post_pickup_uturn_progress`
证据：真实拾取后的左右 `±3π/4` lane 必须交付，或在 16 tick 内实现至少 0.4 净巢距下降且
连续至少 3 步下降；policy 与 temporal-learning fingerprint 必须不变。该硬门补上旧
`carrying_home_action_alignment` 只验单步方向、不验转角幅度和轨迹闭环的覆盖缺口。v13
把 forced-return 从“起步即携食”改为真实 pickup transition，并将 5 局返向复习交错到后半程；
冻结 lane 同时要求拾取后 2 个 action 内发生动作族 switch，堵住训练与验收分布错位。

**关键不变量**：

- `semantic_*_pull` 由 embodiment 发布为感觉/动机预测通道，但不能作为 task outcome 的代理。
  正式闭环只通过公共 `EnvironmentOutcome.measurement` 进入 PE owner；Internal RL 只消费
  PE owner 发布的 typed actual outcome、prediction residual 与 credit，禁止 AntWorld 直送
  optimizer reward、runtime 重算 mismatch 或 evaluation 回灌。realized utility 与 prediction
  residual 必须作为不同 reward components 保留，不能互相替代。
- **import 边界**：`vz-embodiment-ant` 只依赖 `vz-contracts` / `vz-substrate` / `vz-runtime`；
  **禁止**直接 import `volvence_zero.temporal` / `volvence_zero.memory` / `volvence_zero.prediction`
  等内核内部实现。经 `tests/test_import_boundaries.py` 强制。
- **三层生物学对齐**：`sense_encode`/`motor_decode`/`AntNavigator` = frozen substrate（基因组层）；
  rare-heavy 角色重编程 = 中间的基因表达程序层（离线、运行时不可触发）；
  `z_t`/`β_t` + CMS 在线学习 = controller 层（突触可塑性）。
- 信息素总线：每个个体的写入是**独立不可变事件**，读取时聚合；禁止多 writer 互相覆盖同一字段。
- 动态障碍只允许在 round 边界通过 environment owner API 原子替换；agent 不得读取全局几何或未来
  扰动。群体评估直接消费 `PheromoneField` 已发布的 mass/entropy，不在 consumer 重建 bus 内部状态。
- 三物体对象的增删改只走 `AntWorld` owner API；任意方向木棍使用连续 capsule 碰撞，燃烧火柴只通过
  `WorldObservation` 局部 heat channel 和 `WorldTransitionEvidence` 的阈值暴露/脱离事实进入闭环。
  `FORAGING` v1 的 pickup/delivery 契约不变，只有显式 `ECOLOGY + ecology-v2` 才启用新事实。
- `WorldObservation.last_turn_command` 是 efference copy，只发布 commanded turn；隐藏 plant 的
  applied turn/step 只存在于只读 transition evidence，不能进入 substrate/optimizer。信息素 ndarray
  使用不可重新开启写权限的 backing buffer；field 外 deposit/sample 返回空，不得夹到边缘格形成伪通信。
- rare-heavy artifact 只能在离线 pipeline 产生，经现有 ModificationGate 审查并在 session 初始化导入；
  runtime 不训练 artifact，角色标签只允许作为 held-out 行为 readout。
- 正式 evidence artifact 必须带 schema version、git SHA/dirty、依赖/seed/config/model fingerprints 和
  输入 sha256/size；dirty tree 的 manifest 必须声明 `externally_retainable=false`。
- realtime app 只在完整 tick/round 边界排队调用环境 owner 的公开扰动 API；浏览器不得提交
  `turn_command` / `step_command`。视觉帧只投影 `AntStepRecord`、`ColonyRoundRecord`、公开 body/food/
  对象/信息素快照；App verdict 只读正式 artifact，绝不回灌 PE/credit/Internal-RL。只有
  held-out gate 为 PASS 且 archive compatibility/fingerprint 完整匹配的 checkpoint 可标记并加载为
  demo checkpoint；BLOCK candidate 仍可留作诊断 artifact，但 loader 必须拒绝。
- learning archive 恢复必须是事务式：所有外层 schema、owner set/version、part digest 先通过后才能
  触发 owner hydration；任何 owner 恢复或重导出 fingerprint 校验失败时，runtime 必须恢复完整
  preimage。colony 中任一 body 失败时，按逆序回滚截至该 body 的 attempted prefix，禁止部分应用；
  尚未尝试的 suffix 不得调用 restore，以免无故清空 owner 的瞬态窗口。
- collection 中每个 agent archive 的 checkpoint id 必须按位置绑定 `body:{index}` 且不可重复；
  artifact loader 与 colony runner 都在恢复前校验该映射，禁止交换个体状态或静默截断。

详见 `docs/specs/digital-ant-embodiment.md` 与 `research/ant/04_digital_ant_feasibility.md`。

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
    conditioning_lineage: ConditioningLineageRef | None = None

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
    conditioning_lineage: ConditioningLineageRef | None = None
    personal_conditioning_applied: bool = False
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
- State KV B 包把上一轮 ACTIVE、non-cold、未撤销的已审计 conditioning bank 编译为公共 `ConditioningLineageRef`，由 `OpenWeightResidualStreamSubstrateAdapter` 在本轮 capture 前调用 substrate owner 的 `capture_conditioned(...)`，让 residual / Prefix-KV 先作用于真实 prompt-token prefill，再发布 `SubstrateSnapshot`。`personal_conditioning_applied` 是 capture 物理投递的 attestation；`conditioning_lineage` 及逐 step 引用只说明哪版 bank 参与，不允许 consumer 用 lineage 推断 applied。`SHADOW` / `DISABLED`、`text` carrier、cold-start 或撤销状态既不投递也不发布 lineage
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
    conditioning_lineage_refs: tuple[ConditioningLineageRef, ...] = ()
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
- temporal owner 当前会把 `SubstrateSnapshot.conditioning_lineage` 原样发布到 `TemporalAbstractionSnapshot.conditioning_lineage_refs`；world/self aggregate 只做去重合并，不反解 bank 内容、不成为 second semantic owner。物理投递由 `SubstrateSnapshot.personal_conditioning_applied` 证明；`state-kv-temporal-causal.v1` matched ablation 进一步比较 baseline / correct-state / wrong-user / revoked 四臂的 residual、`z_t` 与 `beta_t`，禁止只凭 lineage 声称因果生效
- State KV P5-b：runtime 的 PE action-context builder 用 `vz-contracts` 共享归约 `summarize_conditioning_lineage_refs` 把 `conditioning_lineage_refs` 摘要进 `PredictionActionContext.conditioning_bank_set / conditioning_bank_fingerprints / conditioning_router_version`；credit owner 把前两者原样拷入 `CreditRecord.conditioning_bank_set / conditioning_bank_fingerprints`（typed 归因，供 P5-c bank-confidence readout 消费），并在 `context` 文本追加 `conditioning_banks=a+b`。空 tuple = 该动作无 live bank，是有意义负样本；PE 计算与本快照 shape 均不变
- 延迟 dialogue external outcome 不使用“当前轮 live bank”：`AgentSessionRunner` 通过 turn trace 解析被评分动作的 `ConditioningLineage`，下一轮由编排器注入 PE action context；credit owner 将带 bank lineage 的记录保留在 `recent_action_lineage_credits`，conditioning consumer 只读该窗口。无法解析 action turn 时保持 unattributed，禁止猜测邻近 turn。
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
    # #89 anti-forgetting proxies (report-only; both in [0,1]; owner-derived
    # from per-turn band drift; do NOT enter any acceptance gate).
    new_knowledge_absorption: float = 0.0               # online-fast band movement toward the new signal this turn
    old_knowledge_retention: float = 1.0                # 1 - background-slow band drift this turn

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
class LeastControlReadout:
    control_effort: float
    outcome_quality: float
    least_control_score: float
    evidence_count: int
    description: str = ""

@dataclass(frozen=True)
class CreditSnapshot:
    recent_credits: tuple[CreditRecord, ...]
    recent_modifications: tuple[SelfModificationRecord, ...]
    cumulative_credit_by_level: tuple[tuple[str, float], ...]  # (level, sum) pairs
    description: str
    recent_action_lineage_credits: tuple[CreditRecord, ...] = ()
    rewarding_state_head: RewardingStateHeadState | None = None
    counterfactual_readouts: tuple[CounterfactualContributionReadout, ...] = ()
    least_control_readout: LeastControlReadout | None = None
```

**`CreditRecord.level` 取值**：`token` / `turn` / `session` / `long_term` / `abstract_action` / `prediction_error` / `evaluation_readout` / `social_prediction_error` / `abstract_action_segment` / `counterfactual_contribution`（Phase 1.A COCOA-lightweight，readout-only，不参与 acceptance gate）/ `counterfactual_contribution_learned`（Phase 2.A learned rewarding-state head readout，SHADOW 默认）。

**当前实现口径**：

- P06 当前落地结构化信用记录、gate audit 与 bounded self-modification proposal；默认 `CreditModule` 以 `SHADOW` 接线运行，真实写入仍必须通过对应 owner 的 apply surface 和 gate，而不是由 credit owner 直接突变外部模块
- `recent_modifications` 当前记录 allow / block decision，作为审计轨迹和后续 reflection 输入
- `cumulative_credit_by_level` 先提供最小聚合，后续再扩展到更细粒度的长期统计
- 第二阶段允许在 owner 内部基于 temporal / rollout 结果扩展出 `abstract_action` 级 credit；共享 shape 变更必须按本文件协议登记
- metacontroller credit 当前会消费 posterior drift、binary gate ratio、policy replacement score 等 ETA kernel evidence，并将其压入 `CreditRecord.context`
- `derive_credit_records_from_prediction_error_first(...)` 是当前 PE-first credit 派生路径；evaluation 只提供 readout / gate context，不重新成为原始学习源
- 自修改 gate 当前采用 Two-Gate 风格的保守准入：候选必须提供 validation margin、capacity cap 内的容量成本和 rollback evidence；缺证据默认 block，并把原因写入 `SelfModificationRecord.justification`
- Phase 2.A 的 `RewardingStateHeadState` 归属 `CreditLedger`，只用 owner 内部 context 向量预测 expected outcome；更新必须带 `validation_delta` / `capacity_cost` / `rollback_evidence`，并写入 `recent_modifications` 审计。`counterfactual_readouts` 是对比 readout，不授权下游重建 head 权重。
- COG-1 最小切片新增 `least_control_readout`：由 credit owner 从近期 `counterfactual_readouts` 与 `recent_modifications` 派生 report-only 指标，表示“同等 outcome 需要多少控制努力”。evaluation / mid_layer 只能读取该 readout，不重新计算反事实归因。

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
- **HTTP 投影（deploy 债 `D-collab-pe` 上游半）**：DLaaS interaction `extra` 与 OpenAI-compat `x-lifeform-confidence` header 投影 `next_prediction.confidence`（clamp [0,1]，标 `confidence_origin="kernel_pe"`），PE 轴投影 `error.{magnitude, relationship_error, task_error}`。投影是只读 readout，规则冻结在 `docs/specs/dlaas-api-v1.md` §「Interaction response `extra`」；快照缺席时字段缺席，平台不得伪造。

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
| `open_loop` | OpenLoopModule | OpenLoopSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation, session-post evidence；#90：额外依赖 `apprenticeship_alignment`（消费其 `should_request_feedback`），快照新增 `apprenticeship_verification_requests` |
| `user_model` | UserModelModule | UserModelSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation, session-post evidence |
| `execution_result` | ExecutionResultModule | ExecutionResultSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation, prediction-error evidence |
| `belief_assumption` | BeliefAssumptionModule | BeliefAssumptionSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation |
| `relationship_state` | RelationshipStateModule | RelationshipStateSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation |
| `goal_value` | GoalValueModule | GoalValueSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation |
| `boundary_consent` | BoundaryConsentModule | BoundaryConsentSnapshot | ACTIVE | 每 turn | temporal, boundary_policy, response_assembly, evaluation |
| `personal_conditioning` | PersonalConditioningModule | PersonalConditioningSnapshot | SHADOW | 每 turn | session / open-weight substrate generation；只消费 `user_model`、`relationship_state`、`goal_value`、`boundary_consent` 的 typed owner readout，不读取原始对话或重建语义。State KV P0-b 起 value 增加 `rendered_statement`（owner 用确定性模板把同一 typed readout 渲染成英文状态说明，cold-start 必为空串）；ACTIVE 时投递形态由 `FinalRolloutConfig.personal_conditioning_mode` 决定（`residual`=残差通道/臂 E，`text`=system prompt 状态段/臂 B′，两者互斥），详见 [`docs/specs/personal-conditioning.md`](./specs/personal-conditioning.md) §3.1。第二个开关 `FinalRolloutConfig.prompt_state_delivery`（`text` 默认 / `suppressed`）决定**状态派生段是否进入 system prompt**：`suppressed` 下 `build_system_prompt` 只组装不变表达规则段，得到 prompt 逐字节相同的 `state-kv-arm-a-pure` / `state-kv-arm-e-pure` 载体识别臂；与 `personal_conditioning_mode="text"` 组合为非法配置（构造期 raise）。`suppressed` 为证据专用、禁止部署（移除 boundary/disclaimer prompt 引导），详见 [`docs/specs/state-kv-identification-evidence.md`](./specs/state-kv-identification-evidence.md) §3.2。载体识别证据由 `vz-runtime` 的 `state_kv_identification` 只读 owner 产出：schema `state-kv-identification.v1`，四臂设置从 profile registry 解析（不另建 arm 表），逐 turn 证据只从已发出的 `prompt_fp` / `prompt_state_sections` / `decode_fp` / `personal_conditioning` tag 回读（禁止重算"应该等价"的 prompt），四条判据与 C5 分档写入 `artifacts/state_kv/verdict_identification.json`；`SubstrateEvidenceKind.TRACE_ONLY` 下 verdict 强制封顶 `insufficient_data`，无盲裁判时判据 3/4 记 `insufficient_data` 而非填充占位准确率；matching 准确率是 readout，不回灌学习链路（R12）。State KV P5-c 起 value 新增 `credit_confidence_delta`（有界 credit 反馈 readout，语义与门控见 `relationship_conditioning` 行） |
| `relationship_conditioning` | RelationshipConditioningModule | ConditioningBankReadout | SHADOW | 每 turn | State KV P4-a：第二个 conditioning bank（`bank_type=RELATIONSHIP`），也是首个直接发布通用无 scope `ConditioningBankReadout`（`conditioning-bank-readout.v1`）的 owner——后续新 bank 一律复用该 value 类型，不再每 bank 铸造专用契约（`PersonalConditioningSnapshot` v1 属历史冻结形态，不作模板）。只消费 `relationship_state` + `boundary_consent` 的 typed owner readout，编译 10 维 dyad 长程读出（trust / cumulative trust / continuity / repair pressure / emotional load / stabilization need / trust recovery / tension load / consent compliance / consent clarity）+ 确定性 `rendered_statement`（cold-start 必为空串）。scope / freshness / revocation 由 runtime 在消费点经 `bank_readout_to_bank(readout, slot_name, scope, ...)` 注入（owner 不知会话身份；adapter 以 slot↔bank_type 守门并从 `source_versions` 提升 `consent_version`）。默认 SHADOW（回滚 = `DISABLED`）。P4-b 起接入唯一 consumer：session 装配链在该 slot ACTIVE 且 `personal_conditioning_mode="text"` 时，把 readout 的 `rendered_statement` 作为独立段落并入 prompt 状态段（审计 ref 以 `;` 连接各 bank），并经 `bank_readout_to_bank` 投影后进入 turn 级 `ConditioningLineage`（bank 集按 bank_type 排序，`router_version="static-all.v1"` 确定性全选）；residual / prefix_kv 模式下该 bank 无 latent 通道，不入 prompt 也不入 lineage（lineage 只记录真正影响输出的 bank）。State KV P5-c 起两个 conditioning bank owner 增加可选依赖 `credit`：消费 `CreditSnapshot.recent_action_lineage_credits` 中 P5-b typed bank 归因，经共享有界规则 `conditioning_credit_feedback`（EMA α=0.2、gain=0.3、delta 硬上限 ±0.15、timestamp 水位防滚动窗口重复计数）折算 `credit_confidence_delta` 并随快照发布（value 均新增该字段，默认 0.0、cold-start 必为 0）。`FinalRolloutConfig.conditioning_credit_feedback` 门控：`SHADOW`（默认）只发布不施加、`ACTIVE` 施加到 confidence（clamp [0,1]）、`DISABLED` 停止消费（回滚点）。EMA 状态由 runner session 级持有注入，owner 是唯一写者。详见 [`docs/specs/personal-conditioning.md`](./specs/personal-conditioning.md) bank 家族节 |
| `dual_track` | DualTrackModule | DualTrackSnapshot | SHADOW | 每 turn | memory, evaluation, prediction_error, reflection, credit, regime |
| `apprenticeship_alignment` | ApprenticeshipAlignmentModule | ApprenticeshipAlignmentSnapshot | ACTIVE | 每 turn（学徒/ingestion） | prediction_error（离散事件 PE 源）；belief_assumption / goal_value（经 SemanticProposal 单写）；apprenticeship_protocol_alignment（消费 enriched `guidance_constraints`）；**open_loop（#90：消费 `should_request_feedback` 冒出 verification 开环 actuator）**；#90 起 ACTIVE，仅 apprentice/ingestion turn 生效（普通轮 idle → PE overlay + 请求均 no-op），快照新增 `should_request_feedback` / `feedback_request_reason` / `feedback_request_urgency`；详见 [`docs/specs/apprenticeship-alignment.md`](./specs/apprenticeship-alignment.md) |
| `apprenticeship_protocol_alignment` | ApprenticeshipProtocolAlignmentModule (vz-application) | ApprenticeshipProtocolAlignmentSnapshot | SHADOW | 每 turn（学徒/ingestion） | 把 `apprenticeship_alignment.guidance_constraints` 与编译后 protocol 工件（strategy_playbook / domain_knowledge / boundary_policy）做有限选项集层比对；A1（#90 残余，2026-07-16）：快照新增 `pe_overlay_magnitude` / `pe_overlay_source`（结构裁决派生的 PE-shaped 只读 overlay，application 侧消费，kernel PE 不跨 tier 读）与 `revision_proposals`（protocol-lineage 冲突 → 保守 WEIGHT_DECAY L3 typed 提案），`protocol_revision_queue` 消费 `revision_proposals` 走 R10 gate + 人审队列；详见 [`docs/specs/apprenticeship-alignment-protocol-layer-draft.md`](./specs/apprenticeship-alignment-protocol-layer-draft.md) |
| `evaluation` | EvaluationModule | EvaluationSnapshot | ACTIVE | 每 turn ~ 每会话 | regime, prediction_error, credit, reflection |
| `regime` | RegimeModule | RegimeSnapshot | SHADOW | 每 turn | prediction_error, reflection, retrieval_policy |
| `prediction_error` | PredictionErrorModule | PredictionErrorSnapshot | ACTIVE | 每 turn | memory, temporal_abstraction, regime, credit, reflection；另在 final wiring 中被 evaluation enrichment 读取 |
| `credit` | CreditModule | CreditSnapshot | SHADOW | 每 turn ~ 每会话 | reflection; consumes `prediction_error` + `temporal_abstraction.closed_segments` for PE-derived segment credit |
| `reflection` | ReflectionModule | ReflectionSnapshot | SHADOW / session-post | 每会话后（异步） | temporal_abstraction；另外通过 owner-side writeback 影响 memory / credit / regime |
| `session_post_slow_loop` | SessionPostSlowLoopModule | SessionPostSlowLoopSnapshot | ACTIVE | context / session boundary | reports / experience_consolidation |
| `retrieval_policy` | RetrievalPolicyModule | RetrievalPolicySnapshot | ACTIVE | 每 turn | domain_knowledge, case_memory, boundary_policy, response_assembly |
| `domain_knowledge` | DomainKnowledgeModule | DomainKnowledgeSnapshot | ACTIVE | 每 turn | boundary_policy, response_assembly, evaluation |
| `case_memory` | CaseMemoryModule | CaseMemorySnapshot | ACTIVE | 每 turn | strategy_playbook, response_assembly, evaluation；`action_grounding` 是 CaseMemory owner 对当前 Memory 语境与 reviewed case intervention steps 的语义近邻解释，绑定 active abstract action；terminal `SCENE_EVENT` 可经 `ExperiencedActionEvidence` + gated session-post writeback 形成带 `case:slow-loop:*:experienced-action:*` lineage 的 lived-action case。schema-free evidence 只有携带 outcome-bound `ActionLearningLineage`，且双轨 transition 已被 optimizer 消费、policy update 已应用、Credit owner record IDs 非空时，才能保存为 `CaseActionAbstractionEvidence` 并进入多经历聚合；旧 checkpoint 或 no-RL 缺证明时保留普通 lived-action audit case但 fail closed。多经历晋升使用 `case:slow-loop:action-abstraction:*` lineage，并在排序前通过只读 structured applicability gate（缺 provider/typed conditions/高置信适用判定时 fail closed）；无匹配/非具体行动轮显式为 `None` |
| `strategy_playbook` | StrategyPlaybookModule | StrategyPlaybookSnapshot | ACTIVE | 每 turn | response_assembly, experience_consolidation |
| `boundary_policy` | BoundaryPolicyModule | BoundaryPolicySnapshot | ACTIVE | 每 turn | response_assembly |
| `response_assembly` | ResponseAssemblyModule | ResponseAssemblySnapshot | ACTIVE | 每 turn | session / response generation；`action_realization` 只绑定同拍 `CaseMemorySnapshot.action_grounding` 与 `TemporalAbstractionSnapshot.active_abstract_action`，不重建案例语义；expression 只渲染 owner-published statement。State KV P5-a 起，assembly 携带的 `control_code / control_scale` 是否真正到达 `runtime.generate(control_parameters, control_scale)` 由 `FinalRolloutConfig.generation_dynamic_residual` 门控（默认 `ACTIVE`=字节级现状；`SHADOW` 计算不注入；`DISABLED` 表达层丢弃，substrate kwargs 与 temporal 未产码的 run 一致）。每 turn 无条件发布 `dynamic_residual=<wiring>[:scale]` rationale tag 自证通道状态；该开关与 `personal_conditioning` 解耦，profile `dynamic-residual-off` 为显式消融臂，详见 [`docs/specs/temporal-abstraction.md`](./specs/temporal-abstraction.md) 与 [`docs/specs/state-kv-identification-evidence.md`](./specs/state-kv-identification-evidence.md) 载体清单 C4 |
| `experience_consolidation` | ExperienceConsolidationModule | ExperienceConsolidationSnapshot | ACTIVE | session-post | experience_fast_prior, reports |
| `experience_fast_prior` | ExperienceFastPriorModule | ExperienceFastPriorSnapshot | SHADOW | 每 turn / session-post carryover | temporal, retrieval_policy, regime |
| `dialogue_external_outcome` | DialogueExternalOutcomeModule | DialogueExternalOutcomeSnapshot | ACTIVE | 每 turn | prediction_error, regime, rupture_state, reflection |

State KV P4-c 不新增 runtime slot。temporal owner 对候选
`ConditioningBankSnapshot` 执行 `topk-semantic.v1` 并发布不可变
`ConditioningRouterDecision`；session 装配链只执行选择：先以
`is_injectable` 硬门，再用共享 semantic embedding 对 owner 发布的
`rendered_statement` 评分，乘以 confidence 与 freshness。router 默认
`SHADOW`，实际 lineage 保持 `static-all.v1`，旁路结果写入
`ConditioningLineage.shadow_router_version / shadow_router_scores`；
`ACTIVE` 才把同一 selected set 应用于 prompt、latent carrier 与 lineage，
并将实际 `router_version / router_scores` 发布为 `topk-semantic.v1`；
`DISABLED` 为无评分的立即回滚。bank 增益 verdict 是只读 artifact
`state-kv-bank-gain.v1`，不进入学习链路、不成为第二 evaluation owner。
`freshness=0` 是正式过期语义，必须令 `is_injectable=False`，因此不能进入
router、lineage 或任何 carrier。
| `rupture_state` | RuptureStateModule | RuptureStateSnapshot | SHADOW | 每 turn | reflection, dialogue_trace (diagnostic) |
| `interlocutor_state` | InterlocutorStateModule | InterlocutorStateSnapshot | SHADOW | 每 turn | prompt_planner, response_synthesizer, lifeform-core (LifeformSession.interlocutor_state) |
| `active_mixture` | ProtocolRegistryModule | ActiveMixtureSnapshot | SHADOW | 每 turn | （packet 1.2+ 接入：boundary_policy / metacontroller / vitals / strategy_playbook 读 IDs+权重，不读内容本体） |
| `audit` | AuditModule | AuditSnapshot | SHADOW | rare-heavy / promotion event | credit / gate（A5/T11 接入空骨架；OA-4 业务 packet 落地 N8 audit-agent tool loop 后由 rare-heavy 路径切 ACTIVE，详见 [`docs/specs/audit-owner.md`](./specs/audit-owner.md)） |

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

### 6.X Social Cognition Learning Slots（R16-R20）

下表是 Social Cognition Learning Layer 的 slot 注册表。它们按 `docs/implementation/15_social_cognition_layer.md` 的 SHADOW → ACTIVE → retire 协议逐步落地。**"默认接线"列反映 `FinalRolloutConfig` 当前 default 值（即 `final_wiring.py` 现状）**，必须与 [`tests/contracts/test_data_contract_wiring_sync.py`](../tests/contracts/test_data_contract_wiring_sync.py) 保持一致——任何 spec 与 wiring 偏离都会让 contract test FAIL。

> 主契约的稳定 slot surface 以 §6 默认接线表为准。本节是 social cognition 子领域的额外 slot 注册。完整 rollout notes 与 slice changelog 迁到 `docs/CONTRACT_MIGRATION_LOG.md`，后续实现流水不再追加到本文档。
>
> **Keyed-view 标记**：表格中"默认接线 = SHADOW (keyed view)" 的行（`interlocutor_models` / `relationship_states` / `interlocutor_states`）是 keyed mapping，由 `MultiPartyIdentityModule` 在 owner-internal 层维护，**不**直接对应 `FinalRolloutConfig` 顶层字段；contract test 会跳过此类行。

| Slot Name | Owner 模块 | Value 类型 | 依赖 | 默认接线 | Timescale | Social prediction emitted | PE consumer |
|-----------|-----------|-----------|------|----------|-----------|---------------------------|-------------|
| `multi_party_identity` | MultiPartyIdentityModule | MultiPartyIdentitySnapshot | substrate, memory, semantic proposals, scene role envelope | ACTIVE | online-fast / session-medium / background-slow | active speaker, subject scope, audience scope, identity continuity | social_prediction_error → prediction_error / credit |
| `interlocutor_models` | MultiPartyIdentityModule + keyed semantic owner views | Mapping[str, UserModelSnapshot] | user_model, multi_party_identity | SHADOW (keyed view) | per turn / scene | state-to-person attribution | social_prediction_error |
| `relationship_states` | MultiPartyIdentityModule + keyed relationship views | Mapping[str, RelationshipStateSnapshot] | relationship_state, multi_party_identity | SHADOW (keyed view) | per turn / scene | dyad continuity / repair attribution | social_prediction_error |
| `interlocutor_states` | MultiPartyIdentityModule + readout builder | Mapping[str, InterlocutorState] | evaluation, memory, commitment, multi_party_identity | SHADOW (keyed view) | per turn | current interlocutor readout attribution | social_prediction_error |
| `belief_about_other` | BeliefAboutOtherModule | BeliefAboutOtherSnapshot | semantic proposals, memory, multi_party_identity, prediction_error | ACTIVE | online-fast / session-medium / background-slow | interpretation / belief update outcome | social_prediction_error → prediction_error |
| `intent_about_other` | IntentAboutOtherModule | IntentAboutOtherSnapshot | semantic proposals, execution_result, commitment, multi_party_identity | ACTIVE | online-fast / session-medium | follow-through / next-action outcome | social_prediction_error → prediction_error |
| `feeling_about_other` | FeelingAboutOtherModule | FeelingAboutOtherSnapshot | evaluation, relationship_states, multi_party_identity | ACTIVE | online-fast / session-medium | affect / rapport movement | social_prediction_error → prediction_error |
| `preference_about_other` | PreferenceAboutOtherModule | PreferenceAboutOtherSnapshot | semantic proposals, memory, multi_party_identity | ACTIVE | session-medium / background-slow | durable style / boundary stability | social_prediction_error → prediction_error |
| `conversational_role` | ConversationalRoleModule | ConversationalRoleSnapshot | multi_party_identity, host role envelope, common_ground, ToM summaries | ACTIVE | online-fast / session-medium | addressee / subject / witness assignment | social_prediction_error → prediction_error / credit |
| `common_ground` | CommonGroundModule | CommonGroundSnapshot | multi_party_identity, conversational_role, belief_about_other, memory | ACTIVE | online-fast / session-medium / background-slow | reference resolution / mutual-knowledge sufficiency | social_prediction_error → prediction_error / credit |
| `groups` | GroupModule | GroupSnapshot（G1：+`settled_errors` + learned `group_durability_score`，settlement state 停放在 SocialRecordStore，单写者 GroupModule） | multi_party_identity, conversational_role, common_ground, commitment, open_loop | SHADOW | online-fast / session-medium / background-slow | joint commitment durability / group regime fit（durability PE 结算驱动 learned score → 未来预测 confidence） | social_prediction_error → prediction_error / credit |
| `social_prediction` | SocialPredictionAggregateModule (lifter) | SocialPredictionSnapshot | multi_party_identity, memory.social_pe_signals（Slice 12+：其它 R16-R20 owner 自报 typed signals 后并入） | ACTIVE | pre-action per turn | 把上游 owner 自报的 typed PE signals 升级为公共 SocialPrediction（不重建） | social_prediction_error |
| `social_prediction_error` | SocialPredictionErrorModule (lifter) | SocialPredictionErrorSnapshot | social_prediction, multi_party_identity, memory.social_pe_signals, ToM 4 slots + common_ground + groups 的 owner-settled `settled_errors`（G1 起含 group durability settlement） | ACTIVE | post-action per turn / session | 把上游 owner 自报的 typed PE signals 升级为公共 SocialPredictionError（owner 字段来自 signal.source_owner）+ 外部 probe 注入 | prediction_error / credit |

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
