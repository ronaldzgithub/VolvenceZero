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

### 1.1.3 Relationship Lab decision evidence（non-runtime）

`lifeform-domain-emogpt.lab` 拥有 Relationship Lab 的公开观察、封存环境真值、
反应式 action→outcome 转移与 settled decision sidecar；
`lifeform-evolution.relationship_lab_gate0` 只读这些 frozen records 并拥有 Gate 0
报告；P1 由 `relationship_lab_contexts` 经正式 API 编排既有 `MemoryStore` 与
`companion-ref-harness` owner，再由 `relationship_lab_packet1` 发布只读账本与 Gate 1
报告；P1b 由 `relationship_lab_packet1b` 发布 typed readout 与 lineage-complete report，
P1c 由 `relationship_lab_packet1c` 只读 Gate 0/P1b 正式工件并发布资格分叉。它们是跨
wheel 离线证据契约；P1e 冻结 v2 consumer qualification，P1f 只读 v3 公开文本、sealed
condition anchors 与冻结 embedder 发布 semantic-legibility 报告；P1g 在首条 v3 Qwen
输出前冻结完整 consumer lineage，并只读 fresh Gate 0/P1b 工件发布资格分叉；P1h 再由
domain lab 把已见 v3 固定为 training-only，并在零 v4 Qwen 输出时冻结独立 qualification
package、consumer 搜索预算与 P1g 资格门。它们都
**不**进入 §6 runtime slot 注册表。
spec：[`docs/specs/relationship-lab.md`](specs/relationship-lab.md)。

| Type | owner / producer | dependencies | consumer | wiring_level |
|---|---|---|---|---|
| `RelationshipObservation` / `RelationshipTransferDataset` v1/v2/v3/v4 | type/白名单 builder owner：`lifeform-domain-emogpt.lab.dataset`；record producer：versioned rendered package | 公开 action-outcome history、哈希 user scope、current input、closed action surface；v2/v3/v4 每用户四历史、全局动作胜负平衡、未见 probe surface；v3 指纹额外绑定 public-evidence contract；v4 整包为 12-pair qualification-only | 冻结实验 arms / P1f auditor / P1h split owner | `OFFLINE_SHADOW_EVIDENCE_ONLY`；v1 保持默认，v2/v3/v4 只允许显式 package lineage；产品路径 `DISABLED` |
| `AbstractRelationshipCondition` / `RelationshipPolicyProfile` / history-condition binding v2/v3/v4 | sealed truth owner：`lifeform-domain-emogpt.lab.dataset` + versioned `generator_truth.json` | 两个 abstract condition、两种互补 user policy、history/probe condition binding | loader structural audit、reactive environment、只读 evaluator | `OFFLINE_ENVIRONMENT_TRUTH_ONLY`；condition/policy/binding/preferred action 禁止进入 SUT、memory、PE、credit 或 steering |
| `RelationshipPublicEvidenceContract` | `lifeform-domain-emogpt.lab.dataset` + `relationship_transfer_v3/public_evidence_contract.json` | P1e trigger artifact/verdict、公开 history/probe 文本面、BGE-M3 source/weights、sealed anchor source、60-unit top-1/margin 门、pending human anchor 与 claim boundary | P1f auditor / P1g prereg input | `OFFLINE_ENVIRONMENT_TRUTH_ONLY`；contract 纳入 v3 dataset fingerprint；标签不向 SUT/rater 暴露，evaluation 不回灌 learning/steering |
| `PreActionRelationshipDecision` | type owner：`lifeform-domain-emogpt.lab.contracts`；record producer：各实验 arm | candidate typed outcome distributions、chosen action、source snapshot hashes、model/prompt/weights/seed lineage | reactive environment、sidecar builder | `OFFLINE_SHADOW_EVIDENCE_ONLY` |
| `ReactiveRelationshipOutcome` | `lifeform-domain-emogpt.lab.environment` | sealed latent dynamic、实际 selected action、dataset fingerprint、seed | sidecar builder；未来 typed `dialogue_external_outcome` adapter | `OFFLINE_SHADOW_EVIDENCE_ONLY`；P0 禁止提交 runtime |
| `RelationshipDecisionTrace` | contract/sidecar owner：`lifeform-domain-emogpt.lab.contracts` | canonical public trajectory hash、pre-action decision、environment evidence、observed typed outcome、可选 PE/credit/next-state refs | `lifeform-evolution` read-only verdict / replay | `OFFLINE_SHADOW_EVIDENCE_ONLY` |
| `FrozenBaselineAttestation` | `lifeform-evolution.relationship_lab_baseline` | dataset/model/weights/prompt/generation/seed/decision-ledger hashes、有效/正确/总决策数、token 成本 | `lifeform-evolution.relationship_lab_gate0` | `OFFLINE_READOUT_ONLY`；只准 train/validation/calibration |
| `RelationshipGate0Report` | `lifeform-evolution.relationship_lab_gate0` | package hashes、machinery checks、可选 frozen stateless/raw baseline attestation | 研究/晋升 gate | `OFFLINE_READOUT_ONLY` |
| `RelationshipP1ContextBundle` | 编排/发布：`lifeform-evolution.relationship_lab_contexts`；结构化状态 owner：既有 `vz-memory.MemoryStore`；reference RAG owner：`companion-ref-harness` | rendered public histories、0/8/32 ordinary sessions、user-scope hash、正式 MemoryStore retrieval、ref-harness semantic top-k | `lifeform-evolution.relationship_lab_packet1` | `OFFLINE_SHADOW_EVIDENCE_ONLY`；不注册 runtime slot，不成为第二 memory owner |
| `PersistedRelationshipP1StateDigest` | `lifeform-evolution.relationship_lab_contexts` read-only digest publisher | 每用户 MemoryStore frozen entries 与 ref-harness SQLite/embed records 的 scope/count/content digest | fresh-process recovery probe、P1 report | `OFFLINE_READOUT_ONLY`；digest 不是可恢复状态 owner |
| `RelationshipP1ConsoleControlEvidence` | mutation owner：既有 `MemoryStore` 正式 delete/write/persist API；证据 publisher：`relationship_lab_contexts` | 原记录 hash、replacement hash、delete/reload 与 sibling-scope digest | P1 Gate 1 | `OFFLINE_SHADOW_EVIDENCE_ONLY`；不得直接编辑 backend 文件 |
| `RelationshipP1Decision` / `RelationshipP1Run` | `lifeform-evolution.relationship_lab_packet1` | 同一冻结 model/weights/generation lineage、arm prompt/context hashes、strict action enum、token counts、expected action（仅 evaluator 结算后附着） | P1 report / replay | `OFFLINE_READOUT_ONLY`；逐决策 checkpoint 只作崩溃证据，完整 ledger 为 SSOT |
| `RelationshipP1Report` v2 | `lifeform-evolution.relationship_lab_packet1` | Gate 0 attestation、P1 run、state recovery、scope isolation、token scaling、console、structured-state user-swap effect、steelman qualification | 研究/晋升 gate | `OFFLINE_READOUT_ONLY`；v1 派生重放必须绑定 `source_report_artifact_id` |
| `RelationshipEvidenceReadout` | `lifeform-evolution.relationship_lab_packet1b` | frozen public context + one frozen substrate call；两个 closed action 的 typed `-1/0/+1` evidence score、raw JSON、profile-bound system-prompt/request-template/schema/model/context lineage 与 token 成本 | P1b typed action compiler、只读 evaluator | `OFFLINE_SHADOW_EVIDENCE_ONLY`；v1 action tally 与 v2 condition-aware 由不同 frozen prompt/request hash 区分；expected action / generator truth 在 readout 落盘前不可见 |
| `RelationshipP1bRun` / `RelationshipP1bReport` v4 | `lifeform-evolution.relationship_lab_packet1b` | P1 context bundle、稳定 evaluated-context surface、background/RAG config、seed/P1 config、model/weights/generation、Gate 0 attestation、prompt/request/schema/compiler、content-addressed readout ledger、逐臂 metrics、P1 machinery | P1c/P1e/P1g qualification fork / replay | `OFFLINE_READOUT_ONLY`；strict loader 重算 artifact/派生判词；bundle 内 owner record UUID 只作本次完整性，不作跨重建 identity；不写 runtime slot、memory、PE、credit 或 steering |
| `RelationshipP1cCandidateProtocol` | `lifeform-evolution.relationship_lab_packet1c` + packaged content-addressed JSON | stronger candidate/reference、P1b frozen lineage、Gate 0/P1b seeds、background/RAG/generation config、disk/snapshot guard、hidden-test unopened | P1c runner / qualification assessor | `OFFLINE_READOUT_ONLY`；上游 branch 不冒充 frozen weights，实际权重由 fresh Gate 0 attestation 锁定 |
| `RelationshipP1cReport` | `lifeform-evolution.relationship_lab_packet1c` | candidate protocol、fresh Gate 0 attestation/report、同权重 P1b report、三臂 accuracy/pair-flip 与 machinery validity | formal-prereg / scenario-version / evidence-contract 路由 | `OFFLINE_READOUT_ONLY`；只发布 development next action，不注册 runtime slot，不授权 P2/四能力主张 |
| `RelationshipP1eConsumerProtocol` | `lifeform-evolution.relationship_lab_packet1e` + packaged content-addressed JSON | v2 dataset/context lineage、condition-aware readout、四历史可见性、typed-outcome RAG top-4、Qwen2.5-3B/generation/gates/seeds/materialization guard、hidden-test unopened | P1e runner / local-lineage validator | `OFFLINE_READOUT_ONLY`；protocol 必须在任何 v2 模型输出前冻结，禁止 consumer 漂移、少给历史或从 sealed condition/policy 取捷径 |
| `RelationshipP1eReport` | `lifeform-evolution.relationship_lab_packet1e` | frozen P1e protocol、fresh Gate 0 report/attestation、同权重 P1b report、三臂 accuracy/pair-flip、machinery/readout validity 与三路 next action | formal-prereg / still-saturated / public-evidence-contract 路由 | `OFFLINE_READOUT_ONLY`；strict loader 重算 artifact 与派生判词；不注册 runtime slot，不授权 P2/四能力主张 |
| `RelationshipP1fEvidenceUnit` / `RelationshipP1fReport` | `lifeform-evolution.relationship_lab_packet1f` | v3 dataset/contract、P1e trigger、冻结 BGE-M3 identity/weights、public text hash、sealed anchor hash、cosine score/margin、聚合阈值与 pending human anchor | P1g consumer-protocol freeze 路由 / replay | `OFFLINE_READOUT_ONLY`；60 个 unit 全覆盖，strict loader 重算 aggregate/verdict/artifact；报告不含原文/condition id，不注册 runtime slot，不是 PE/credit/reward 或 Readable 证明 |
| `RelationshipP1gConsumerProtocol` | `lifeform-evolution.relationship_lab_packet1g` + packaged content-addressed JSON | P1f artifact/verdict、v3 dataset/contract/context surface、exact Qwen/BGE weights、generation/gates/seeds、condition-aware prompt/request/schema/compiler、四历史、typed-outcome RAG top-4、zero-prior-output/hidden/P2 guards | P1g runner / preflight / qualification assessor | `OFFLINE_READOUT_ONLY`；protocol id `8e08d488…52c6` 在第一条 v3 Qwen 输出前冻结；任一 source/weight/context/threshold 漂移 fail loudly |
| `RelationshipP1gReport` | `lifeform-evolution.relationship_lab_packet1g` | frozen P1g protocol、P1f source、fresh Gate 0 attestation/report、same-substrate P1b report、三臂 accuracy/pair-flip 与 machinery/readout validity | formal-prereg / saturation / consumer-training-version 路由 | `OFFLINE_READOUT_ONLY`；strict loader 重算 artifact 与派生判词；权威 artifact `9d7f05b5…e3c8` 判 `consumer_still_underqualified`，不注册 runtime slot，不授权 P2/四能力主张 |
| `RelationshipP1ContextReplayManifest` | `lifeform-evolution.relationship_lab_contexts`（只读发布 P1g immutable `contexts.json`） | dataset/template/RAG lineage、144 个 context hash、36 个 scene×depth RAG evidence 顺序 | P1i preflight / context reconstruction | `OFFLINE_READOUT_ONLY`；artifact `aea311e2…9791` 只重放已发布的 P1g model-input surface；必须验证 evidence 集合与逐输入 hash，不成为第二 RAG/semantic owner |
| `RelationshipConsumerSplitContract` | `lifeform-domain-emogpt.lab.consumer_split` + v4 content-addressed JSON | P1g underqualification artifact、v3/v4 dataset fingerprint、training/qualification role、零 v4 Qwen output、三轮 candidate budget、leave-one-surface-family-out selection、P1g qualification gate、隔离/feedback/formal/P2 guards | P1i training-view loader / P1h evaluator-only bundle audit | `OFFLINE_ENVIRONMENT_TRUTH_ONLY`；contract `2ce75cb4…4af8`，v3 只能 training、v4 只能 qualification；不注册 runtime slot，不运行模型 |
| `RelationshipConsumerTrainingView` | `lifeform-domain-emogpt.lab.consumer_split` | frozen P1h contract + v3 training dataset only | P1i external-baseline calibration | `OFFLINE_READOUT_ONLY`；frozen type 不含 v4 observation/truth；training label 不得成为 Volvence PE/credit/reward/steering |
| `RelationshipP1iCalibrationProtocol` | `lifeform-evolution.relationship_lab_packet1i` + packaged content-addressed JSON | P1h contract/training view、P1g report/protocol、v3/v4 identity、Qwen/BGE/generation/context lineage、三个预注册 prompt candidate、LOSO selection method、zero-v4/formal/P2 guards | P1i runner / local-lineage validator | `OFFLINE_READOUT_ONLY`；protocol `080c908d…ef40` 在首条 P1i Qwen 输出前冻结；候选只可读 v3 training view，v4 fingerprint 只作 identity，禁止 materialize qualification data |
| `RelationshipP1iCandidateCheckpoint` / `RelationshipP1iCandidateProgress` | `lifeform-evolution.relationship_lab_packet1i` | protocol/candidate/training-context lineage、连续 indexed readout/decision 前缀、expected record count | P1i crash recovery / candidate finalization | `OFFLINE_READOUT_ONLY`；逐条原子追加且 readout 先于 decision，已有记录不可覆盖；只接受完整 lineage 的连续前缀，满 36 条才发布 candidate artifact |
| `RelationshipP1iCandidateArtifact` / `RelationshipP1iCalibrationReport` | `lifeform-evolution.relationship_lab_packet1i` | 每候选 36 条先发布 readout、附 label 后 decision、三臂 aggregate 与六个 surface-family fold metric、确定性 LOSO selection key | frozen consumer protocol / replay audit | `OFFLINE_READOUT_ONLY`；三轮候选 ledger 全保留，strict loader 重算 metric/selection/artifact；training label 仅用于外部 baseline calibration，不进入 PE/credit/reward/steering |
| `RelationshipP1iFrozenConsumerProtocol` | `lifeform-evolution.relationship_lab_packet1i` + run artifact | P1i report top-ranked candidate、selected prompt/request/schema/compiler、完整 substrate/context/RAG lineage、P1h qualification gate、zero-v4-input/output guard | P1j one-shot v4 qualification | `OFFLINE_READOUT_ONLY`；protocol `938af607…10bf` 已在 0/0 v4 input/output 时冻结；qualification feedback 永久禁止返工 consumer，不授权 P2/formal/四能力主张 |
| `RelationshipP1jQualificationProtocol` | `lifeform-evolution.relationship_lab_packet1j` + one-shot run artifact | frozen P1i consumer、P1h split、v4 context manifest/surface、72 条 record plan、零 Qwen output、原 gate 与 no-feedback guards | P1j executor / strict resume | `OFFLINE_READOUT_ONLY`；protocol `20b96957…2a2` 在首次 v4 Qwen output 前冻结；只能在同一 attempt 追加，不得 fresh rerank、换 consumer 或第二次 prepare |
| `RelationshipP1jCheckpoint` / `RelationshipP1jProgress` | `lifeform-evolution.relationship_lab_packet1j` | protocol/consumer/context/model lineage、72 条连续 planned key、逐条 readout/decision 前缀 | P1j crash recovery / terminal assessor | `OFFLINE_READOUT_ONLY`；readout 必须先于 heldout decision 原子落盘，允许恢复一个 dangling readout；已有记录不可覆盖，qualification truth 不进入 model input |
| `RelationshipP1jQualificationReport` | `lifeform-evolution.relationship_lab_packet1j` | 三臂 strict-valid/accuracy/pair-flip/token、readout/decision ledger hash、冻结 gate 派生 verdict、zero-feedback/zero-revision guards | terminal stop / P1k diagnostic 路由 | `OFFLINE_READOUT_ONLY`；权威 report `e9226ee8…fd78` 判 `consumer_failed_v4_qualification`；只发布一次且不得修改 P1i consumer，不开启 formal hidden test、P2 或四能力主张 |
| `RelationshipP1kProtocol` / `RelationshipP1kExecutionGate` / `RelationshipP1kReport` | `lifeform-evolution.relationship_lab_packet1k` | 绑定 P1j underqualification protocol/report、冻结 P1i/substrate 与 v3 prompt-steelman surface、四格正交 oracle disclosure、48 条最大计划、分段放行/早停、terminal ledger 与 executed/skipped path | 诊断定位 / 下一 owner 选择 | `OFFLINE_READOUT_ONLY`；权威 protocol `204e0904…64bd`、terminal report `ba6c5cf7…7138`；A 格 12/12 strict-valid、accuracy=0.50、pair-flip=0，按冻结门早停并跳过 B/C/D，判 `substrate_cannot_apply_disclosed_policy`；dedicated diagnostic prompt，不冒充 P1i consumer；consumes sealed truth；六对单 seed 只作 directional triage；不得回流 consumer、资格门、PE/credit/reward/steering；不授权 P2 |
| `RelationshipP1lProtocol` / `RelationshipP1lReport`（人工盲标） | `lifeform-evolution.relationship_lab_packet1l` | 60 条去真值 public unit、打乱后的两个 condition summary、sealed expected option、3-rater 多数一致/准确率 | 退休 pending human-anchor 的下一份契约修订 | `OFFLINE_READOUT_ONLY`；rater 看不到标签；评分不改 v3 contract 文件、不进学习路径。包号是 P1l，不是计划 §14.2 / P1m 仪器升级 |
| legacy `RelationshipProductPilotSourceProtocol` / shared frozen `RelationshipProductPilotPublicView` / `RelationshipProductPilotEvaluatorBundle` envelopes | `lifeform-domain-emogpt.lab.relationship_product_pilot_source` | 只解释并物化 `relationship-product-pilot-source.v1`；owner implementation raw 固定为 `15162708…972ae`，默认 loader/path 保持 v1；冻结 8 个 matched world clone ×（4 onboarding + 24 decision），public value 仅含公开 context/current input/correction/action surface，sealed value 独占 subject seed、scene/condition/policy/preferred action/environment seed | 已有 Product Horizon v1/v2 protocol、run 与 GPU/model-free `validate-existing` | `OFFLINE_ENVIRONMENT_TRUTH_ONLY`；legacy owner/raw/schema/materialized identities 不得被新 source revision 改写；历史 campaign pin 不允许 fallback 或自动升级 |
| `IndependentRelationshipProductPilotSourceProtocol`（复用上述 frozen public/evaluator envelope shape） | `lifeform-domain-emogpt.lab.relationship_product_pilot_source_v2` | live registry 只登记 `relationship-product-pilot-source.v3`；v3 逐字节 pin 已公开且不修改的 source-v2 JSON，并将其作为 `immutable_semantic_payload_input`。独立 owner 解释 v2 cohort/seeds/environment/domain/rendering/identity namespace，为每 subject×condition 物化 2 onboarding + 12 decision 的 14 个不复用 public semantic surface；archive-only v2 decoder 只复验历史 materialized IDs | `relationship-condition-reader-qualification-protocol.v2` 显式消费；当前 Product Horizon campaign 均不消费。未来 campaign 必须新增并精确 pin protocol/owner/public/sealed identities | `OFFLINE_ENVIRONMENT_TRUTH_ONLY`；历史 source-v2 raw/owner metadata 不改，且不进入 live registry；source-v3/wrong-owner/raw/base/public/sealed 任一漂移均 fail closed；零模型输出、不新增 runtime slot，不构成产品、formal 或四能力证据结论 |
| `RelationshipProductHorizonSourceProtocol` / `RelationshipProductHorizonPublicView` / `RelationshipProductHorizonEvaluatorBundle` v4 envelopes | `lifeform-domain-emogpt.lab.relationship_product_horizon_source_v4`；settlement 仍唯一属于 `ReactiveRelationshipEnvironment` | 独立物化 112 个 synthetic root ×（4 onboarding + 8 matched collection + 40 evaluation）；onboarding 为旧 base policy，collection/evaluation 共享反转后的 complementary policy，以支持 evaluation 全臂冻结 gate 的一次 apply 隔离。每个 evaluation segment 固定 8 决策且两 condition 各 4。root/tape/render/environment seed 分域内容派生，arm identity 不进入 seed；public trajectory 与删除身份/filler 后的 causal-tape signature 均须 112/112 唯一，5,376 个 environment seed 与全部 session/decision/scene ID 全局唯一。public view 由 source owner 提供 nested exact-key `from_payload()`，可在不加载含 hidden dynamics 的 source protocol、sealed evaluator 或 environment 的情况下恢复 immutable typed snapshot | admitted source-v4 artifact `4cc1ec45…dd54` → 三臂 Product Horizon medium matrix；source-v3 calibration/reader、A1、A2、legacy Horizon 均不自动消费 | `OFFLINE_ENVIRONMENT_TRUTH_ONLY`；source 不拥有 collection forced action 或 arm；既有 environment 可确定性 replay 16,128 个 action-conditioned branch（action-specific randomness、不是 CRN），commitment/byte closure 已由独立 admission 封存。strict public loader 只解释已发布 snapshot，不授予 sealed truth 或 campaign authority；112 是 synthetic roots，不是真人 N；尚未物化 BGE reader input，不授权 campaign/model/CUDA/formal/integrated/四轴/production claim；不新增 §6 runtime slot |
| `RelationshipProductHorizonSourceAdmissionProtocol` / compact action-counterfactual commitment index / source-v4 admission manifest | `lifeform-evolution.relationship_product_horizon_source_admission`（source-v4 development input admission）；source/settlement 数学仍由 source-v4 owner 与 `ReactiveRelationshipEnvironment` 唯一拥有 | protocol `b3988b21…2102` 精确 pin source-v4 raw/canonical/public/sealed identity 与 7-file source execution closure；implementation `b55ce6ad…` 生成单份 create-only 六文件 artifact `4cc1ec45…dd54`：先持久化 admission/source protocol、public plan、sealed evaluator、5,376×3 完整 owner replay 导出的 compact commitment digest index，再做第二次全量 semantic rebuild 与磁盘 byte compare，通过后才最后写 manifest。独立 `validate-existing` 已由外部 protocol/artifact 双 ID 只读全量重建通过 | Product Horizon 112-root development campaign 只允许单向 pin `4cc1ec45…dd54`；source-v3 admission、reader/theta0、A1/A2 与 formal ledger 不继承 | `OFFLINE_ENVIRONMENT_TRUTH_ONLY`；开发档刻意不复制 replay A/B、PID comparator、Gist 或进程安全仪式，`process_independence_proven=false`。唯一新增正判词是 `campaign_input_admitted=true`；reader input/qualification、theta0、forced-action schedule、runtime order、campaign/model/CUDA/formal/integrated/四轴/真人/production authority 均为 false/zero。commitment 仅持久化 owner 公开 preimage builder 所定义完整 preimage 的 SHA-256，distribution/draw/outcome/reaction 仍由 pinned owner 重算，禁止建立第二 truth owner；不新增 §6 runtime slot |
| `RelationshipProductHorizonDevelopmentReaderProtocol` / training projection / public embedding table / frozen linear reader / manifest | `lifeform-evolution.relationship_product_horizon_development_reader`（离线 development input owner）；reader 数学仍唯一属于 `lifeform-domain-emogpt.relationship_condition_reader`，BGE adapter 仍属于 `relationship_lab_product_model_adapters` | 只打开 committed v6 preflight 的 public corpus 与四条 condition-only training labels；group split 只作 ID lineage，不打开 challenge label 文件。另只打开 admitted source-v4 manifest/public plan，不打开 sealed source。一次 pinned BGE-M3 CUDA materialization 覆盖 4 条 training、v6 preflight 冻结的 224 条 label-free public challenge text 与 source-v4 的 5,824 次 reader occurrence/1,881 个 exact unique text，合并为 2,109-row canonical table；四条 persisted embedding 经既有 centroid-linear builder 重导同一 reader。本包不声称这 224 条已与 admitted source-v3 calibration set 完成 exact join，该谱系绑定留给后续 theta0 bundle | source-v3 evaluation 外 theta0 calibration 与 source-v4 112-root development campaign 的 model-free reader composition；consumer 使用前必须外传 protocol/artifact 双 ID 调 `validate-existing`，theta0 consumer 还必须独立 pin source-v3 admission 并证明 224 条 exact join | `OFFLINE_READOUT_ONLY`；这是实际 development input materializer，不是 rehearsal/qualification。`challenge_label_file_read_count=0`、`source_v4_sealed_file_read_count=0`；不跑 held-out scorer，不发 Gist，不授权 campaign/formal/integrated/Readable/四轴/production claim。新增离线 evidence artifact，不注册 §6 runtime slot；回滚为停止消费该 bundle |
| `RelationshipProductHorizonTheta0CalibrationProtocol` / public exact-join artifact / append-only calibration trace / `RelationshipActionGateTheta0Artifact` / manifest | `lifeform-evolution.relationship_product_horizon_theta0_calibration`（evaluation 外 development bootstrap 编排）；reader/forecast、owner persistence、reactive settlement、social PE/credit 与 gate 参数仍分别由既有唯一 owner 发布 | protocol `0e51c343…3678c` 独立 pin source-v3 admission、v6 label-free public corpus、实际 CUDA development reader bundle 与 gate replay seam。校准只打开 source-v3 的内容寻址 public admission envelope，不读取 replay sealed 文件；先机械证明 224 条 source text ↔ preflight challenge ↔ embedding table 的 exact ordered permutation。随后每 root 重置 owner、跨 8 roots 携带同一 zero-init gate；每拍先 fsync public preaction，再由 pinned reactive environment 按 temporal 实际 action 结算，累计 192 个 PE-derived credit update。implementation `7922c8a8…` 已物化并由外部双 ID model-free 复验五文件 bundle `d9776631…6f24`；trace `a3be9393…5a7a` 的内容 ID作为 source-batch lineage，发布 nonzero theta0 `f44eb267…3398`，cold restore 参数相同且 update/processed/pending 全为零 | 新 source-v4 Product Horizon development prereg/campaign 可在外传 protocol/artifact 双 ID并通过 model-free `validate-existing` 后单向消费 theta0；不得反向修改 source-v3/reader/attempt03，也不得供 formal unseen 门复用 | `OFFLINE_SHADOW_EVIDENCE_ONLY`；这不是 reader rehearsal/qualification，不运行模型或 CUDA，不读取 challenge labels/group split，不把 condition/policy/preferred action/seed/distribution/draw/evaluation/judge 输入 reader/gate/credit。192 update/unique credit、零 pending 与 source-v3 内 164 次 steer 只闭合 bootstrap；全零终局不得生成 theta0，当前非零 theta0 仍不证明 source-v4 non-noop opportunity、Learnable、Steerable、任一四轴效果、真人或 production ACTIVE；manifest-last，回滚为停止消费 immutable bundle |
| `RelationshipProductHorizonTheta0V2BootstrapProtocol` / frozen forced schedule / stable forced-batch trace / exact `CreditBatch` / APPLY+WITHHOLD receipts / candidate theta0 v2 / manifest | `lifeform-evolution.relationship_product_horizon_theta0_v2_bootstrap`（adaptive development bootstrap 编排）；schedule entry、executor、reactive settlement、owner persistence、social PE/credit 与 gate transition 仍由既有 owner 发布 | 在 f44e terminal all-noop 诊断后另立新 protocol；显式声明 source-v3 outcome 与 parent theta0 双重使用。schedule 在任何本次 forecast/outcome 前以 public position 固定 192 role，块状公式得到全局 96/96、每 root 12/12、每 position 跨 roots 4/4。单一 cold f44e 对全部 exposure 不变；actual forced nonnoop 定义为 temporal delivered action 非 neutral，training-support nonnoop 则定义为 frozen selected action 非 neutral；两者不得混用。实际 action 经 reactive environment 结算，credit 不在线 apply。唯一顺序敏感 batch 由两个 fresh f44e gate 重算同一 plan，分别 APPLY/WITHHOLD；APPLY terminal 通过 nonzero delta、finite 且非全零参数、未触 cap、cold 0/0/0 与 training-support physical nonnoop 门才发布新 theta0 | 独立冻结的 source-v4 transductive public opportunity scanner；scanner 通过也只可进入 collection-only 动态门，不直接授权 112-root evaluation。旧 f44e、source-v3、reader 与 attempt03 均不回改 | `OFFLINE_SHADOW_EVIDENCE_ONLY`；这是失败诊断后的 adaptive stabilization，不是 unseen/formal/reader qualification/能力效果。稳定 trace 明确排除 nondeterministic temporal timestamp，不声称完整 pulse receipt byte replay或磁盘事务原子性；同一 protocol FAIL 后禁止换 schedule/order/base/threshold/bias/seed、选早期 checkpoint或多 epoch。全零 terminal 不抛成 incomplete root，而封存 no-consumable FAIL；成功只发布候选 theta0，campaign/四轴/真人/production claims 仍全 false；不新增 §6 runtime slot，回滚为停止消费 immutable v2 bundle |
| `RelationshipProductHorizonTransductivePublicOpportunityProtocol` / stable public opportunity trace / paired APPLY-vs-strict witnesses / manifest | `lifeform-evolution.relationship_product_horizon_transductive_public_opportunity`（public-only development gate）；public snapshot 解释仍唯一属于 source-v4 owner，reader/gate/executor 数学仍由既有 owner 发布 | protocol `4471c9ab…4d83f`（raw `9ce246b2…35419`）只 pin admitted source-v4 manifest/public plan、development embedding table/reader artifact 与 theta0 v2 artifact/cold frozen policy。每 root 的四条 public onboarding 写入一次，随后 48 个 decision 均从同一 post-onboarding immutable persistence 调真实 typed `APPLY_CANDIDATE` executor；只有 112 个 index-0 首拍动态可达，另外 784 collection + 4,480 evaluation 共 5,264 个是 reset-state counterfactual probes。PASS 仅按 temporal delivered nonnoop，并为 reachable/evaluation 各补一个相同 prestate/forecast/decision/advisory/policy 下的 `FORCE_STRICT_NOOP` actual-action divergence witness；冻结 v1 projection 将 112 个 reachable-first 的 owner/forecast/decision/action/executor/advisory/temporal-controller/cold-policy lineage 纳入 digest，约束未来 collection worker exact match。implementation `0ffda0a1…1373` 的唯一 materialization 发布 artifact `2dec2e3f…774e4c`：reachable / collection stress / evaluation stress delivered nonnoop 分别 `107/112 / 740/784 / 4,279/4,480`，两个 paired witness PASS，stable trace `bc3418bd…b1181`，projection `5eee5690…84d1`；外传双 ID exact replay 与 bytes/mtime no-write 检查通过 | PASS 只提升 `collection_prefix_protocol_freeze_authorized=true`，允许另行冻结每 root 8-decision collection-only dynamic protocol；`collection_prefix_execution_authorized=false` 保持到新 prereg 自己闭合。FAIL 原样阻断该 candidate。不得直接消费为 40-decision evaluation、campaign effect、formal ledger 或任何能力结论 | `OFFLINE_SHADOW_EVIDENCE_ONLY`；三项输入均已见，固定为 transductive。允许 448 次 public onboarding historical outcome read；source-v4 sealed/challenge labels/group split/environment settlement/PE/credit/gate update/model/CUDA 均为零。稳定输出排除 temporal timestamp/receipt ID，并由外传 protocol/artifact ID重放 byte compare；同 protocol FAIL 禁止换 theta/reader/source/threshold/subset/order/selection 重试。PASS 只证明 typed opportunity 未退化，不是成功率、reader qualification、Learnable、Steerable、产品效果、四轴、真人或 production；不新增 §6 runtime slot，回滚为停止消费 immutable scanner artifact |
| `RelationshipProductHorizonDynamicCollectionPrefixProtocol` / append-only dynamic collection trace / manifest | `lifeform-evolution.relationship_product_horizon_dynamic_collection_prefix`（natural-APPLY development gate）；public/evaluator/action-commitment、reader、frozen gate、executor、owner settlement、social PE 与 credit 仍由既有唯一 owner 发布 | protocol `47cea5fa…0dbb2`（raw `3f3f7ad3…0d97f3`）单向 pin source-v4 admission、development reader、theta0 v2 与 transductive scanner。保留全部 112 roots，每 root onboarding 一次后顺序执行 index 0–7；112 个首拍在打开该 root selected branch 前逐行 exact-match scanner 的 23-field seam，后 784 拍要求 prior settlement owner post-persistence 等于 next preaction input。preaction append+fsync 后才按 temporal delivered action 调 source owner 的 single-action commitment builder并结算 actual branch；frozen settlement派生 owner writeback→social PE→credit但不 apply gate，checkpoint 全程 cold `0/0/0`。implementation `c275bd90…ade4` 的首个唯一 materialization 发布三文件 artifact `f1a5b2f6…aaf4` 与 1,906-row trace `e9cf896b…f295`；external 双 ID `validate-existing` exact replay 和 bytes/size/mtime no-write 均通过。v1 trace/manifest 的误名字段 `development_reader_artifact_id=ded8c0dc…cbb6` 实为 leaf reader；package `ba46775b…f967` 与 table `fd1f8a4a…f409d` 仍在 protocol/scanner 分别 pin，未来 schema 必须分列但不得回写本 artifact | PASS 只允许冻结独立 forced common-collection batch protocol；natural credit 不得冒充 forced exposure 或直接进入共同 batch。forced batch execution、40-decision evaluation 与 campaign 均另需新协议 | `OFFLINE_SHADOW_EVIDENCE_ONLY`；输入全已见，属于 transductive development。终局为 112 roots、448 onboarding、896 pre/post/branch/PE/credit/writeback、784 handoff、112 seam match、各 896 个 unique credit/commitment/settlement/evidence ref；temporal action 为 stay 389 / space 353 / noop 154，742 nonnoop 覆盖 112/112 roots；gate/forced/batch/evaluation/model/CUDA/judge/human 全零。environment facade 在全局首拍 fsync 后加载 sealed evaluator，但不向 SUT/preaction发布；这是 cooperative order firewall，不是 OS 安全隔离。成功也不证明 Appendable/Readable/Learnable/Steerable、产品效果、formal/unseen/真人/production；不新增 §6 runtime slot，回滚为停止消费 immutable artifact |
| `RelationshipProductHorizonForcedCommonBatchProtocol` / root-local forced schedule index / append-only forced collection trace / root terminal owner envelopes / root batch transitions / manifest | `lifeform-evolution.relationship_product_horizon_forced_common_batch`（root-local development collection/batch owner）；schedule/forecast/gate/temporal/source/settlement/PE/credit/persistence 数学仍由既有 typed owner 发布 | protocol `dd0d28a7…aff93`（raw `0a1cec80…51fe`）只消费 dynamic PASS manifest，禁止打开 dynamic natural trace/outcome/credit。public-position formula 事前生成 112 份互异的 8-entry schedule（各 local sequence 0–7、role 4/4；全局 448/448、每 column 56/56），完整 payload 进入父索引 `13c3f8e2…4e9f`并在首 forecast 前 create-only fsync。每 root 只跑一次 forced collection：四 onboarding 后八拍 exact owner handoff，preaction fsync 后才按 temporal actual delivered action exact-join source-v4 commitment并派生 writeback→social PE→credit；collection gate 保持 cold且不 apply。每 root 单独形成 8-credit batch，full/frozen/strict 三个 fresh gate 重算同一 plan并提交 APPLY/WITHHOLD/WITHHOLD，后两者共享 canonical receipt；full 只由 theta0+exact batch+APPLY receipt owner replay。terminal owner snapshot 保存完整 opaque envelope，fresh hydrate/export exact round-trip，三臂未来从相同 bytes 分叉。implementation `5d028d9c…2798` 的首个唯一 materialization 发布 artifact `92880fb7…40d7`：112 roots/112 unique batch、896 writeback/credit、112/112 APPLY 参数非零且 finite、最大绝对终态参数 `3.028261 < 4.0`、零 cap-hit；实际 action 为 stay 218 / space 209 / noop 469，即 427 physical nonnoop 且覆盖 112/112 roots。external 双 ID exact replay 与 bytes/size/mtime no-write 通过 | PASS 后的独立 40-decision Product Horizon development campaign protocol可单向 pin 112 个 batch/transition/owner state；dynamic/source/reader/theta0、A1/A2、legacy campaign 不反向消费 | `OFFLINE_SHADOW_EVIDENCE_ONLY`；禁止 896-credit global batch、跨 root policy、三臂重跑 collection、consumer 解析 owner payload、online credit apply、evaluation/judge 回灌或从任意 checkpoint伪造 full。至少一 root APPLY 参数净变化非零且零 cap-hit才可授权 campaign protocol freeze；全零封存 `arm_degeneracy_invalid_contrast_no_claim`。本协议不跑 reader rehearsal/model/CUDA/40-decision evaluation，不估计 Learnable/Steerable 或任一能力效果；448 个 recommendation role 不得冒充 448 physical nonnoop。formal/unseen/integrated/真人/production 均 false，不新增 §6 runtime slot，回滚为停止消费 create-only artifact |
| `RelationshipProductSourceAdmissionRootManifest` / `RelationshipProductSourceActionCounterfactualCommitments` / `RelationshipProductSourceAdmissionComparison` / `RelationshipProductSourceCampaignAdmissionManifest` | `lifeform-evolution.relationship_product_source_admission`（离线准入 verdict）；source 与 settlement 数学仍由 `lifeform-domain-emogpt.lab.relationship_product_pilot_source_v2` / `ReactiveRelationshipEnvironment` 拥有 | 精确 pin source-v3 raw/canonical/public/sealed identity 与 9-file direct execution closure；官方 cooperative CLI 启动两个 fresh model-free worker，各自物化 8 subjects、32 onboarding、192 decisions，并在 campaign 前按既有三动作表封存 576 条 action-conditioned distribution/draw/outcome/reaction/evidence-ref；第三进程逐文件 byte-exact 比对后机械派生 input-only admission | future 全新 Product Horizon development protocol；legacy v1/v2 campaign、reader qualification、A1/A2 均不消费 | `OFFLINE_ENVIRONMENT_TRUTH_ONLY`；可转移判词只包含 deterministic owner replay 与 byte identity，本地 PID 是 self-reported receipt，`fresh_process_independence_proven=false`，不构成 OS/security independence claim。selected action 进入环境 draw hash，因此是 action-specific potential-outcome randomness、不是 common-random-number design。`campaign_input_admitted=true` 只允许新协议消费输入；campaign/runtime-order/model/CUDA/formal/integrated/四轴/真人/production authority 全为 false。source-v3 已进入 reader 谱系，不得冒充 unseen formal source；不新增 §6 runtime slot |
| `RelationshipActionGateTheta0Artifact` / `RelationshipActionGateFrozenPolicy` / `RelationshipActionGateFrozenDecision` / `RelationshipActionGateForcedExposure` / `RelationshipActionGateCreditBatch` / `RelationshipActionGateBatchPlan` / `RelationshipActionGateBatchReceipt` | `lifeform-domain-emogpt.relationship_action_gate.RelationshipActionGate` | 内容寻址 nonzero theta0（完整 source checkpoint payload hash + evaluation 外冻结 calibration-batch ID）、frozen `PreferenceActionForecast`、recommendation/noop forced-action declaration、同一 cold checkpoint/policy、按序 exact `SELF`-track social-PE credit；pure candidate transition 后只切 apply/withhold disposition。可独立持久化的 theta0/decision/exposure/batch/receipt 由 owner 发布 strict `from_payload()` 并重算内容 ID；plan 从 gate+batch 重算，policy 从完整 components + owner replay 构造，不接受摘要 ID 冒充完整状态 | future Product Horizon development consumer / offline validator；当前无 campaign consumer | `OFFLINE_SHADOW_EVIDENCE_ONLY`；旁路新增类型，不修改 legacy v1 artifact/decision/checkpoint/mode/payload，不新增 §6 runtime slot。withhold 保持 exact pre-state，apply 只保证 owner 进程内单引用替换；post-batch policy/restore 必须从 exact batch + APPLY receipt 重放，snapshot 不锁死 mutable gate。owner 只校验 credit/social-PE machine-ID join；forced action 的单拍实际交付与 PE-credit 结算由下列 collection consumer 闭合，跨 root 共同 batch、reaction/outcome/PE exact-match、非 noop evaluation steer、disk/crash atomicity 与能力效果仍必须由未来 campaign receipt 独立闭合 |
| `RelationshipActionCommonBaselineCredit` / preference-action utility public pure helpers | `vz-cognition.credit`（credit record 与代数变换 owner）；冻结 action distribution、utility 解释、settlement 与 social PE 仍由 `preference_about_other` owner 发布 | 对完整 frozen `PreferenceActionForecast` + exact typed `ENVIRONMENT` outcome（confidence 必须为 `1.0`）调用 preference owner 纯函数重放完整 settlement，并要求 public social PE 与既有 `relationship_action_prediction_error` parent `CreditRecord` 均 exact replay。common baseline 永久为同一 pre-action forecast 的 `neutral_noop` expected utility；新值只按 `parent_action_PE + (expected_delivered − expected_noop)/2 = (observed − expected_noop)/2` 确定性派生。immutable record 持有完整 forecast/evidence/settlement/PE/parent credit，并绑定各自 digest、utility/formula/schema ID、三项 utility、parent/adjustment/final canonical float hex 与 content-addressed record ID；compact `to_payload()` 只作 audit projection，未来持久化 consumer 必须另存完整 parents 并 exact replay，不能用摘要恢复状态或解析 `CreditRecord.context` | future versioned relationship-action gate 可消费该 typed credit，再与独立 assignment owner 的完整 pre-outcome schedule membership exact-join；legacy v1 gate/credit consumer 不变 | `OFFLINE_SHADOW_EVIDENCE_ONLY`；本 record 不含 assignment role、propensity、schedule 或 evaluator/judge 字段，单条 record 不是 causal effect、ATE/CATE、Learnable 证据或 treatment proof。只有未来 source 另证 conditional pre-outcome randomization/positivity/actual-delivery join 后，centred assignment moment 才可获得对应有限 estimand；fixed 4/4 balance 本身不够。当前不接 runtime/gate/campaign，不新增 §6 slot；回滚为不调用新增 derive API |
| `RelationshipProductFrozenPulseAuthorization` / `RelationshipProductExecutorCommand` / `RelationshipProductTemporalDelivery` / `RelationshipProductExecutorReceipt` / `RelationshipProductFrozenPreActionSnapshot` / `RelationshipProductFrozenSettlementSnapshot` | `lifeform-domain-emogpt.lab.relationship_product_pulse`（executor 编排）；forecast、gate decision、temporal、PE/credit 仍由原 owner 发布 | exact immutable `RelationshipActionGateFrozenPolicy` + owner `PreferenceActionForecast` + policy/checkpoint authorization；command/receipt 还绑定 `SocialRecordStore` owner 对完整 opaque pre-persistence 发布的版本化 canonical SHA-256；共同 candidate 冻结后唯一 treatment 输入为 `apply_candidate / force_strict_noop`，actual action 只由内容完整绑定的 temporal delivery projection 中 `delivered_action_id` 发布；frozen settlement 保留 exact `RelationshipProductSettlementInput`，且 pre/post owner persistence 只经 owner hydrate/public typed getters 与 owner snapshot 对齐；pre persistence 在真正 settlement 前再次复验，post persistence 必须等于 preference owner 从 pre + exact evidence 重放的唯一 canonical transition；receipt 不携带未纳入 hash 的完整 temporal snapshot | future 独立 Product Horizon development worker/environment/validator；历史 v1/v2 不消费 | `OFFLINE_SHADOW_EVIDENCE_ONLY`；strict arm 不切 gate mode、不伪造 counterfactual update；receipt 内容寻址绑定完整 owner prestate/candidate/delivered advisory、temporal APPLIED projection、checkpoint/update/pending 零变化与 no evaluator/judge feedback。settlement 只接受 `ENVIRONMENT` external source，并从 exact frozen forecast + 完整 external evidence 调用 preference owner 公共纯函数重算完整 settlement，再 exact-join owner outcome→social PE→credit 与 durable persistence；协同删除/注入历史、沿用旧 receipt ID 或构造后原地漂移 pre payload 均 fail closed。只按 delivered action 结算，evaluation 禁止 apply gate credit。单条 receipt 不证明效果；campaign 必须另证 non-noop opportunity 与 actual-action divergence，否则 `arm_degeneracy_invalid_contrast_no_claim`。未接新 campaign/source，不改 attempt03 或 legacy `relationship-product-pulse.v1` |
| `RelationshipProductForcedCollectionScheduleArtifact` / `RelationshipProductForcedCollectionScheduleEntry` / `RelationshipProductForcedCollectionAuthorization` / `RelationshipProductForcedCollectionCommand` / `RelationshipProductForcedCollectionReceipt` / `RelationshipProductForcedCollectionPreActionSnapshot` / `RelationshipProductForcedCollectionSettlementSnapshot` | `lifeform-domain-emogpt.lab.relationship_product_pulse`（collection executor 编排）；schedule、forecast、gate、temporal、environment、PE/credit 各自保持原 owner | 外部 schedule owner 发布完整 content-addressed typed artifact：全部 entry 入 hash、decision 唯一、sequence 必须从 0 连续，entry 只含 decision/sequence/闭域 role（`owner_recommendation / neutral_noop`）。authorization 持有整份 artifact、按 decision exact lookup 唯一 member 并绑定既有 frozen-policy authorization；不接受调用方只声明一个 digest。具体 action 在 owner forecast 发布后机械派生，再经 cold theta0 的 `record_forced_exposure()` 绑定。command/receipt 内容寻址绑定 cold gate candidate（并记录 `gate_would_noop`）、forced delivered advisory、temporal `APPLIED` projection、完整 opaque owner prestate 的 canonical SHA-256 及 checkpoint/update/pending 零变化；settlement 只按 receipt 的 actual delivered action exact-join environment→owner settlement→social PE→credit，并产下一份 owner persistence | future source-v4 Product Horizon development worker 先在 prereg pin 预期 schedule artifact ID，聚合一次共同 8-decision collection 后，才把同一 `CreditBatch` 分给 APPLY/WITHHOLD 三臂；source-v4、历史 v1/v2 与 evaluation executor 不反向消费 | `OFFLINE_SHADOW_EVIDENCE_ONLY`；command 没有 arm、batch disposition 或调用方可选 concrete action，pulse 不拥有 schedule。collection 强制 `apply_credit_to_gate=false`，不得逐 decision 在线更新；现有 evaluation 的 `apply_candidate / force_strict_noop` 二值 treatment 不扩枚举。API 没有 evaluator/judge 输入字段，但 prereg pin/no-leak、schedule artifact admission、ENVIRONMENT evidence 与 source-v4 commitment 的 exact join 仍由 future campaign dependency validator 证明。定向测试只证明单拍 forced delivery、owner/PE/credit lineage 与共享 batch 可由同一输入产生，不证明八拍只执行一次、跨 root/arm byte equality、Learnable/Steerable 效果、campaign/formal/model/CUDA/production authority；回滚为不调用新增入口，legacy payload/hash 不变 |
| `RelationshipProductOnboardingSnapshot` / `RelationshipProductPreActionSnapshot` / `RelationshipProductSettlementSnapshot` | `lifeform-domain-emogpt.lab.relationship_product_pulse`（编排）；状态与数学仍由 `preference_about_other`、`prediction_error`、credit、`RelationshipActionGate` 各自 owner 拥有 | owner persistence snapshot、named forecast runtime、gate checkpoint/mode/authorization、frozen substrate placeholder；settlement 只接收 actual typed external outcome + owner evidence | Product Horizon fresh-session worker / chain receipt publisher | `OFFLINE_SHADOW_EVIDENCE_ONLY`；onboarding 写正式 owner API；pre-action 早于环境结算；PE→credit 只来自 actual outcome，`credit_withheld` 不应用 gate update，`strict_noop` 不生成 counterfactual update；不改变 production wiring |
| `ProductBaselineInput` / `ProductBaselinePublicLedger` / `ProductBaselineResult` / `ProductBaselineDispatcherResponse` / `PrecomputedPublicEmbeddingTable` v2 | `lifeform-evolution.relationship_lab_product_{baselines,model_adapters,baseline_dispatcher}` | public plan/subject/decision boundary、ordered source session/block、history/current ledger entry；Qwen source+revision/tokenizer/generation、BGE source+revision；chronological whole-block truncation或 deterministic semantic top-k；v2 table 显式冻结 model source/revision 与 exact-text vector map | Product Horizon parent / offline baseline-chain validator | `OFFLINE_READOUT_ONLY`；resident model 只作跨 decision 复用，调用保持 stateless；canonical request/response/table 逐层复算，额外 truth、未知/重复/noncanonical JSON、tokenizer/revision/reserve 漂移均 fail loudly；invalid generation 显式映射 typed noop |
| `RelationshipProductHorizonProtocol` / `RelationshipProductTypedChain` / `RelationshipProductBaselineChain` / `RelationshipProductHorizonReport` / `RelationshipProductHorizonManifest` | `lifeform-evolution.relationship_lab_product_horizon` | 上述 public/sealed source、product pulse、public BGE table、resident Qwen/BGE dispatcher；5 typed targeted arms + 2 strong baselines；fresh-child preaction→parent settlement handshake | GPU/model-free `validate-existing` / development product decision | `OFFLINE_READOUT_ONLY`；8×24 full selection、primary index 12–23、directional floor 0.05、至少 6/8 positive subject、安全非劣 margin 0.02；single-axis/formal/human/residual/user-visible/four-able/production claim 均 false；sealed sidecar 只能在全部 SUT/model action 结束后发布，manifest 最后写入 |
| `RelationshipResidualFitProtocol` / `NamedActionSteeringCorpus` / relationship named-action fit bundle/report | corpus owner：`lifeform-domain-emogpt.lab.relationship_residual_fit_corpus`；数学 owner：`volvence_zero.agent.named_action_steering_artifact_training` | pre-action frozen `PreferenceActionForecast` + `LEARNED/update_count>=1/non-oracle/STEER` gate；subject-disjoint rows；frozen Qwen residual runtime、restricted enum scorer、composite protocol+corpus lineage | future relationship physical residual consumer / GPU-free validator | `OFFLINE_SHADOW_EVIDENCE_ONLY`；target 只由 owner recommended action 导出，禁止 evaluator/judge/outcome/reward/credit record 输入；bounded multiplicative delta、无 free bias、zero-code strict noop、matched sensor-off；v1 只证明 typed enum fit prerequisite，不证明 raw strict JSON、用户可见生成、long-horizon effect、完整 Steerable 或四能力 |
| `RelationshipP4LongContextScientificPrereg` / `RelationshipP4LongContextPreparation` | `lifeform-evolution.relationship_lab_p4_long_context_causal_campaign` | registry 原样保留并禁止执行 zero-output v1/v2 protocol/artifact `5387516a…6d2e` / `899b7b0a…6901`、`666d2e85…3baf` / `795dea07…a8bd`；v3 protocol `9f352778…a282d`（raw `ea8a17a1…359e`）与 preparation `c5a708ae…f42e` 冻结 32/64/192-root historical design，但已因 power contract 欠规定在零 model output 时退休，不再是 future-execution prereg | historical validator / v4a lineage；无 execution consumer | `OFFLINE_READOUT_ONLY`；允许 `show-protocol / prepare / validate-existing` 复验 v1/v2/v3；execution/formal/model-output/subject/donor/twin/power-DGP-materialization=0，不新增 runtime slot，不消费或写入 memory/PE/credit/gate/steering，任何四轴/真人/product/production claim 均为 false |
| `RelationshipP4LongContextPowerFailureCertificate` / `RelationshipP4LongContextPowerAdmissionCertificate` | `lifeform-evolution.relationship_lab_p4_long_context_causal_campaign` | historical power-bound v1 `735b20a1…f3fa`（artifact/certificate `fad6c105…9518 / 682efba8…e70e`）的条件精确算术，以及 historical power-admission v2 `67d294fa…6bc9`（raw `130f7667…e04`、artifact/certificate `9883e107…4ee9 / cd6ceca0…36b3`）对 v3 grid-membership 欠规定的终局；v1 unconditional scientific admission=false，`0.62046904104455107035` 不得用作 v3 无条件 FAIL | v4a lineage / historical validator；无 execution consumer | `OFFLINE_READOUT_ONLY`；power-bound CLI `prepare / validate-existing` 复验 v2，`validate-v1-existing` 只复验 v1；historical status=`power_contract_under_specified_no_development_authorization`，`v3_power_failed_under_frozen_grid=null / v3_power_passed=false`；不是 full DGP/source/empirical 或四轴证据 |
| `RelationshipP4LongContextV4PlanningProtocol` / `RelationshipP4LongContextV4PlanningFreeze` | `lifeform-evolution.relationship_lab_p4_long_context_causal_campaign` | v4a protocol `63e007b7…08753`（raw `d06b0710…00e0a`、helper raw `bf38e7ab…a0ef`）与 create-only artifact/certificate `08245400…e56c1 / b7e95f14…6c764`；plan/screen/schedule/manifest raw `9e17383f…0064f / d8f0f6b4…01ba / df426477…f1d6f / 26b46683…3e3e` 冻结 sentinel-first/grid-membership、skip-witness、planning-mean/missingness、generated-action classification、576-tuple contract、全 126 exact-hex screens、任意-Q counter RNG/MC 整数门及 session-major 六 candidate-cell/640-block schedule | separate source-opportunity preflight → tuple-feasibility index → power search → one-shot confirmation → future v4b projection；当前无 execution consumer | `OFFLINE_READOUT_ONLY`；status=`v4_planning_contract_frozen_full_joint_planner_pending`，`power_contract_determinate=true` 但 source/grid unresolved、N=null、power PASS/FAIL=null；`1088=PASS→1152=FAIL` 且 1856 也仅为 screen candidate；source/subject/donor/twin/baseline/model/CUDA/simulation/outcome=0，source/development/model/qualification/formal authorization=false；不注册 runtime slot，不证明四轴 |
| `RelationshipP4LongContextSourceOpportunityPreflightProtocol` / `RelationshipP4LongContextSourceOpportunityPreflightCertificate` | `lifeform-evolution.relationship_lab_p4_long_context_causal_campaign` | source protocol `47bcf656…6494a`（raw `9d4d3ab5…c40b0`、34883 bytes；zero-output section `912e45f6…ee01`；helper raw `72efc093…46f8`、59810 bytes）与 create-only artifact/certificate/projection `8a36d2de…0a99 / 64d879c4…c95a / b8b7823a…e6f1b`；projection/certificate/manifest raw+bytes 为 `ee33fa32…66b2+72969 / f9089ce0…0736+8036 / 2829b16d…60a6+2036`，冻结 typed truth/public-origin 边界、15-axis LSB blueprint、root/prefix/pair inventory、hidden SHA-balanced fact orientation、Latin stratum rotation、typed utility、final-32K twin transform 与 generic-decision 512-atom marginal；root/pair/orientation/balance/twin/atom digests 为 `baafdcd3…97346 / 10b3d5fe…3a326 / cba5e614…15985 / 220c9718…e22f / 689b9102…bf29 / 82c97e44…00ca` | external publication anchor → separate create-only source structural inventory + receipt → tuple-feasibility index；当前无 execution consumer；package API lazy export、CLI exact source roots、input reparse/hardlink rejection | `OFFLINE_READOUT_ONLY`；status=`source_opportunity_preflight_contract_frozen_zero_output_inventory_materializer_not_run`，external anchor=false、materializer absent、source stage=false、576 unresolved、N=null；所有 count 只计 persisted/published artifact rows，内存 deterministic derivation object 不是 source content/materialization；source row/text/tape/subject/donor/twin pack/atom/model/CUDA/power/outcome=0，materialization/tuple/model/development/qualification/formal authorization=false；Prediction Error 仍由 cognition owner 在未来 realized settlement 后发布，不注册 runtime slot，不证明四轴 |
| `RelationshipP4LongContextExternalAnchorRequestProtocol` / `RelationshipP4LongContextExternalAnchorRequest` | `lifeform-evolution.relationship_lab_p4_long_context_causal_campaign` | A0 request protocol `dedfc7ff…4bee`（raw `38ce85d4…a8c6`、16006 bytes）与本地 create-only request/artifact `7897e328…89d1 / 5496fa80…2900`；request raw `0d5147cd…3057`、12115 bytes，manifest raw `17496a50…4125`、1307 bytes；冻结五个 source-preflight publication subject、四个 canonical upstream roots、public GitHub Gist 第一 revision/唯一文件 exact bytes 与 future receipt/reobservation/A0-admission contract；helper raw pin 后同 buffer 执行，prepare canonical-only、relocated validation 不重绑定，local default-stream-only，四 text EOL 固定 helpers=LF/action+schema=CRLF；private origin 不是 publication | source-opportunity preflight → local A0 request → explicit publication authority → new public Gist first revision → independent no-auth/cookie HTTPS receipt（empty description；same owner/id/revision、zero-parent、exact-one `100644` blob tree/raw bytes join）+ fresh unauthenticated reobservation → A0 admission → materializer implementation → A1 materializer/single-attempt envelope anchor+admission → structural inventory；publisher auth 需用户另行明确授权且 credential 不序列化；当前无 external consumer | `OFFLINE_READOUT_ONLY`；status=`external_publication_anchor_request_frozen_publication_not_observed_no_authority`；request/dispatched/publication/observation/external-anchor/A0-admission=false，A0 receipt 不能授权 materialization，A1 仍强制；ephemeral exact derivation 不是 persisted materialization；materializer/materialization/source/tuple/power/model/CUDA/development/qualification/formal/四轴 authority=false；network/Git/Gist/source/model/CUDA/output count=0，不新增 runtime slot |

契约不变量：

- generator truth、`preferred_action`、future outcome 与 judge/evaluation 不得进入
  `RelationshipObservation.to_sut_payload()`；loader 必须 fail loudly on leakage；
- v2/v3/v4 还必须隐藏 `condition_id / policy_id / probe_condition_id /
  history_condition_bindings`；每位用户四条历史须覆盖两个 condition×两个 action，
  每个非空 action 恰好一正一负，probe family 对该用户未见，mirrored siblings 共享
  current bytes/condition 但 policy 与 preferred action 互补。任一不变量破坏都由 dataset
  owner 拒绝加载，consumer 不得重建或放宽；
- v3 必须存在 schema-strict `public_evidence_contract.json`，并与 rendered/truth 一同进入
  dataset fingerprint。公开 history 审计面固定为 `user_utterance + user_reaction`，probe
  固定为 `current_input`，history joiner 固定为换行；auditor v1、cosine、BGE-M3
  source/weights、12 位 score 精度、tie-fail、60/60 top-1、最小 margin 0.02、平均 margin
  0.07 均不得在看到 v3 Qwen 输出后更改；
- pre-action timestamp 必须严格早于 environment outcome；sidecar canonical hash
  mismatch 必须拒绝加载；
- action outcome 由环境 typed transition 产生，不由 LLM、关键词或 evaluation
  推断；LLM 后续只能渲染已结算 outcome；
- `RelationshipDecisionTrace` 不是 owner snapshot，不得注册 runtime slot，也不得
  被 consumer 当作 memory/PE/credit 的第二写者；
- 后续把真实结局接入内核时只能经现有 `dialogue_external_outcome` owner；P0
  不创建平行 outcome channel；
- P1 的 `MemoryStore` 写入按 user scope 独立 backend + subject id，consumer 只能读
  owner API 返回的 typed relationship-outcome records；不得遍历 backend JSON 或从
  ordinary history 重建第二份结构化状态；
- RAG steelman 必须真实走 `companion-ref-harness` embed/index/top-k/blend API，并记录
  embedding weights/config digest；不得用关键词或 scene id 选择记录。P1 默认
  `top_k=4`，v1 P1b development 固定 `top_k=2`，v2 P1e 固定 typed relationship-outcome
  candidate surface 与 `top_k=4`，v3 P1g 继承同一 top-4 surface 并额外绑定 BGE weights；
  该差异必须进入 `rag_config_sha256`，旧 artifact 不得改写；
- `stateless` mirrored pair 只允许 current-turn-only 的同一次 completion；P1
  structured-state user-swap effect 必须另门检查，不能用不同 context hash 冒充行为差异；
- P1 不写 PE/credit/regime/steering，evaluation 与 generator truth 只在决策完成后附着
  expected action；任何新增学习行为必须进入后续独立收敛包；
- P1b evidence readout 只允许两个固定 score 字段和值域 `{-1,0,+1}`；typed compiler
  只能比较 score，平局映射 `neutral_noop`，禁止读取原始文本、outcome 字符串、用户
  scope、scene id 或 expected action；
- v1 readout v3 只做动作级符号 tally，按构造在 v2 会双零，禁止把它作为 v2 steelman。
  P1e 必须在任何 v2 模型输出前冻结 condition-aware readout，使 current message 参与
  语义情境归纳，并给 prompt/structured-state 全部四条历史、给 RAG `top_k=4`；P1d
  不切换 consumer、不运行模型、不发布 baseline verdict；
- P1b 若让 prompt/RAG steelman 超过 prereg 上限，报告必须发布 `dataset_saturated`，
  后续只能版本化场景难度，禁止选择较弱 prompt、随机丢证据或加采样噪声压回阈值；
- P1c 必须先用 stronger candidate fresh Gate 0，再用同一 model/weights/generation 跑
  frozen P1b；candidate manifest、Gate 0/P1b seed、evaluated context surface、background/
  RAG config、prompt/request/schema/compiler、RAG top-k 与阈值任一不匹配都 fail loudly。
  两次独立 owner 重建允许产生不同 record UUID，禁止据此否定字节相同的 evaluated surface。
  P1c 禁止从 raw logs 重建 owner
  状态，只能加载 content-addressed `FrozenBaselineAttestation` 与 `RelationshipP1bReport`；
- P1c 三路输出固定为 `formal_prereg_freeze_candidate`、
  `version_scenario_dataset_saturated`、`rewrite_public_evidence_contract`；Gate 0 rejection
  与 machinery regression 是前置失败，不得升级为资格或能力结论。第一路也只授权冻结
  formal prereg 候选，secret heldout 仍须在冻结完成后另行生成；
- P1c v2 的 Qwen2.5-3B 权威 development report 判
  `version_scenario_dataset_saturated`：prompt/RAG/structured 三臂 accuracy 与 pair flip
  均为 1.0。该事实只触发 `relationship_transfer_v2` 场景版本化，不授权 P2、formal
  heldout 或四能力主张；
- P1e 的 v2 consumer protocol id 为
  `5221909debd8b0248c83332589c2681270118dc54b7014654db2d627ca2fbd1e`。权威
  Qwen2.5-3B development run 的 fresh Gate 0 PASS；prompt/RAG/structured accuracy 为
  0.625/0.25/0.625、pair flip 均为 0.25、24/24 readout valid，因此只允许发布
  `rewrite_public_evidence_contract`。后续必须先版本化公开 evidence/label contract，
  禁止在已见 v2 split 上轮换 prompt 或提前接 PE learning/steering；
- P1f v3 dataset fingerprint 为
  `35b8c46e6fd5810779aff38ed935d8c4f0741bf7d496d2e3eec85f93fbf2134f`，public-evidence
  contract id 为 `8ba8a6788d35e959c4a6fa42d31f54baa7d5e1ba48f52603e4bec510232d3cbb`。
  权威 BGE-M3 audit 对 48 histories + 12 probes 得到 60/60 top-1、最小/平均 margin
  0.020403792213/0.080645917619，report artifact
  `a231e2096b2c4b5fcf3e8b36fd099d0955ce2e355e793d38f5ed8e87a047ecbd`，只授权 P1g
  在任何 v3 Qwen 输出前冻结 consumer protocol。human anchor 仍 pending；不得据此声称
  human readability、Qwen transfer、Readable、formal 或四能力成立；
- P1g consumer protocol id 为
  `8e08d488382442f364aae102d80c268c8c23927d547f64c1e79cb0a87f0f52c6`，明确记录冻结前
  v3 Qwen 输出数为 0。权威 fresh Gate 0 为 24/24 valid、accuracy 0.50，PASS；24/24
  contextual readout valid，prompt/RAG/structured accuracy 为 0.75/0.50/0.50、pair flip
  为 0.50/0.00/0.50，P1g artifact
  `9d7f05b574bafb21641d22c766fe31c4656c09bf6f5e04493474eee6c694e3c8` 只能判
  `consumer_still_underqualified`。prompt 单臂进入资格带不等于 aggregate qualification；
  禁止事后删除 RAG、改成 best-arm gate、降低阈值，或在同一 v3 split 上继续调 prompt。
  后续只能先冻结独立 consumer-training 与 unseen-qualification split，不得把 evaluator
  答案作为 PE/credit/reward；
- P1h consumer split contract id 为
  `2ce75cb44515b4c727ad065995501d063a8f3727923e8a322b4378b53e394af8`。它绑定已见 v3
  fingerprint `35b8c46e…134f` 为 `consumer_training_only`，以及零 Qwen 输出的 v4
  fingerprint `9bfe6ae0…5796` 为 `unseen_qualification_only`。v4 包含 12 组、24 位用户、
  96 条历史且全部 dynamic split 为 development heldout；v3/v4 scene/user scope/event/surface family/
  完整 public text 必须精确不相交。P1i 只能读取不含 qualification dataset 的 frozen
  training view，在 v3 上最多保留三轮候选并用 leave-one-surface-family-out 选择；完整
  bundle 只供 owner/evaluator 做隔离审计。qualification input/truth/feedback、formal hidden
  与 P2 均关闭；因为 v4 文件存在仓库中，它不冒充外部 secret heldout；
- P1i calibration protocol id 为
  `080c908d7824b25081c501abbb6e76a2405a16508dbc0fb9d119b831447eef40`。三套普通 Qwen
  prompt consumer、同一 request/schema/compiler/substrate/RAG/context、三臂×12 用户的
  覆盖面与 LOSO 排序在首条输出前冻结。每个候选 readout ledger 必须先于 evaluator label
  投影保存；逐 fold 与 aggregate metric、候选原始输出、最终 ranking 均不可删除。P1i 只能
  产出一个供 P1j 使用的 frozen consumer protocol，不能运行 v4，也不能把 v3 label 或后续
  v4 feedback 送入 Volvence learning/control。权威 report `c9382c18…8a79a` 的 108/108
  readout strict-valid，排序为 conditioned-match / latent-partition / counterfactual-contrast；
  选中项 RAG accuracy/pair-flip 为 0.583/0.167，双主臂最坏 fold pair-flip 为 0，故只代表
  training-only consumer freeze，不代表 v4 qualified；
- P1j protocol `20b96957f522e5ead7d02b8475c311fe79de18491a11cf75fd4f24120535d2a2`
  已在 v4 Qwen output=0 时冻结；context manifest `4bb57f01…730b` 包含 24 scene×3 depth×4 arm
  的 288 个 context hash，实际 record plan 为 24 scene×3 evaluated arm×1 seed=72。普通背景
  继续由 frozen consumer 的 v3 template asset 提供；v4 只拥有 qualification observation/truth。
  权威 run 已在同一 checkpoint 前缀完成 72/72，readout 先于 truth decision；三臂
  strict-valid=1.0，prompt/RAG/structured accuracy 分别为 0.542/0.500/0.542，pair flip
  分别为 0.417/0.250/0.417，report `e9226ee8…fd78` 判
  `consumer_failed_v4_qualification`。完成态 strict resume 未调用 Qwen且保持首记录与报告
  hash 不变；consumer revision=0、evaluation→Volvence learning/control=false；
- P1j 关闭后不得再开 v5 用点阈值追资格。下一份资格门是计划中的 P1m：≥24 组镜像对、
  Wilson 95% 单侧下限（accuracy ≥0.50、pair flip >0.35）、冻结生成配方；实施时新
  模块名，不得覆盖 `relationship_lab_packet1l`。P1k oracle 四格诊断与 P1l 人工盲标
  分别是诊断上限和 human-anchor，都不是资格门，不得回流 PE/credit/reward/steering；
- 没有冻结真实 substrate baseline 时，只允许 `machinery_ready=true`，Gate 0 必须
  保持 pending；提供后必须 dataset hash 匹配、所有结构化 decision 有效、样本量
  达标且准确率不超过 prereg non-saturation ceiling。stateless 合理弃权不设准确率
  下限，数据集可解性由 sealed oracle 与 action effect 单独验证。

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
- queue 对同一 session lifetime 内完全一致的 `job_id + payload` 只执行一次，并在 `queue_state.duplicate_job_count` 公开重复计数；同一 `job_id` 的不同 payload 必须 fail loudly
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

- `owner_hydration/semantic_state` — 9 个 SemanticStateStore slot；schema v3
  额外持久化 CP-12 outstanding `OwnerPredictionSignal` 与 per-slot sequence，
  使 commitment/open_loop/boundary_consent 等预测能在新 session 由原 owner
  结算；v3 record 新增可选 `semantic_key` / `canonical_value` 供 `user_model`
  profile facts 使用；v1/v2 兼容读取时新字段为空
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
| temporal post-switch minimum dwell | `vz-temporal` FullLearned temporal owner；配置转发：`AgentSessionRunner`；门控：`FinalRolloutConfig.temporal_post_switch_min_dwell`（WiringLevel，默认 DISABLED）与 `temporal_post_switch_min_dwell_actions`（默认 0） | 通用 option-commitment 契约：switch action 计作第 1 个 action；ACTIVE 在最短 dwell 满足前抑制后续**自然** beta termination，使 `steps_since_switch` 可以从 0 自举并进入既有 continuation signal；新的 typed external boundary 始终可以打断 dwell。SHADOW 只发布 would-suppress owner evidence，DISABLED/0 为逐字节回滚。该门不读取环境、carrying 或 action-family 名称，不改变 family persistence 的评估阈值，也不进入 checkpoint 学习参数；Ant profile 显式 ACTIVE/4，并把 wiring/actions 记入 `AntStepRecord.backend_wiring`。不新增公共 slot |
| optimizer reward stream 发布 | `vz-temporal` joint-loop owner | 不可变 `RuntimeReplayRewardStream`，经 `ETANLJointLoop.latest_runtime_replay_reward_stream` 发布、`AgentSessionRunner.latest_runtime_replay_reward_stream` 只读透出。字段：`eligibility_contract / outcome_payoff_reward / prediction_error_reward_enabled / settled_transition_count / eligible_transition_count / ineligible_transition_count / nonzero_reward_transition_count / nonzero_realized_payoff_transition_count / nonzero_segment_bonus_transition_count / realized_action_payoff_sum / segment_bonus_sum / reward_sum / eligibility_reason_counts / last_eligibility_reason / description`。这是 optimizer **实际消费**的 reward，与环境侧 measurement 计数器（在泄漏 tick 上恰为 0）和 PE signed residual 都不是同一个量。readout-only，不回灌学习；进程内累计，不进 rare-heavy checkpoint、跨 session 不续接；不新增 slot |
| runtime replay latent clamp | `vz-temporal` Internal-RL owner | sandbox 的 signed `[-1, 1]` clamp 对 reward/advantage 正确，但同一函数也用于 latent code / modulated mean / candidate mean / policy mean 重建，而在线 owner 把 `z_t` 限制在 `[0, 1]`；action-head 残差为负时 replay lane 会重建出冻结 plant 无法输出的 mean。`FinalRolloutConfig.internal_rl_runtime_latent_unit_clamp`：`False`（默认）保持历史 signed 边界=精确回滚，`True` 只对 latent/mean 重建改用 `[0, 1]`。**reward / advantage clamp 边界在任何取值下都不变。** torch PPO backend 尚未消费该契约（见 spec 已知缺口）|
| causal z-policy action head | `vz-temporal` metacontroller/Internal-RL owner | 通用 factorized `causal_action_head_state -> bounded z_t residual` 参数面；默认保留历史低秩，部署 profile 可在 owner 学习开始前请求不超过 `n_z` 的 rank，已学习 live mapping 禁止原地改 rank；full-rank 使用 identity input factors 与全零 output/bias，消除第二次随机压缩但不引入动作 prior。state 由同一 Ndim encoder 参数对当前 observation 做零 recurrent preimage 编码，覆盖完整 `n_input` 并发布为 signed `[-1,1]`，live/pure/torch replay 与 open-segment persistence 必须使用同一值，禁止以历史依赖的 serving hidden 替代；前向/反向不得二次执行 `[0,1]` 重心变换；可选 `effective_dims` 由冻结 actuator 的 embodiment profile 在启动时声明，`None` 保持全维兼容，显式值必须非空/唯一/界内，pure/torch gradient 与 live/sandbox residual 对非支持 output row 严格为零，禁止 temporal owner 硬编码业务 motor 语义；常数截距保留 owner 学习率 `0.12` 倍、状态路径 `0.05` 倍、单步 `0.01`、总幅度 `0.1`，batch mean 只进入该截距；factor 使用 owner 基础学习率并只消费 centered state covariance，与 torch path 尺度对齐；output factor 保持零初始化，首个非零 covariance batch 先计算 bounded candidate output、再按其真实列范数回传 input，最后原子提交；ACTIVE runtime replay 的 batch target 按真实 transition 数计，部署 profile 必须避免 singleton batch；`DISABLED/SHADOW/ACTIVE` 可回滚门控，禁止对象字段与 motor command；参数进入 owner snapshot、canonical archive、fingerprint 与事务 rollback，不新增 runtime slot |
| runtime exploration context | `vz-temporal` metacontroller owner | 调用方可提供不透明、非语义的 context；owner 只保留 SHA256 摘要并纳入 coherent option identity，不保留或解释原文。缺省保持历史全局序列精确不变；Digital Ant 以 episode seed + body offset 分散序列，matched arms 的同 episode/body context 必须相同；不新增 slot 或 checkpoint 字段 |
| stochastic z-policy exploration | `vz-temporal` Internal-RL owner | `CausalZPolicy` 使用 owner-local seeded PRNG 采样标准正态噪声，禁止从 observation/step/track 用 `math.sin` 等确定性公式伪造探索。matched arms 必须显式复用 seed；world/self 默认 seed 为 0/1。`CausalPolicyCheckpoint.exploration_rng_state` append-only 保存完整 RNG state，restore 后下一抽样必须精确重放；旧 checkpoint 的 `None` 保持兼容。不新增 runtime slot |
| typed task measurement | `vz-contracts` / PE owner | `EnvironmentOutcome.measurement` 仅含环境可观察事实；runtime 保留 lineage，PE 是唯一 mismatch owner |
| opaque learning archive | `vz-runtime` facade | owner 发布 `OwnerPersistenceSnapshot`；`agent-learning-archive.v2` 以 strict canonical JSON 绑定逐 owner schema/payload sha256、整体 state fingerprint，`agent-learning-checkpoint-collection.v1` 再绑定 sense schema / input dim / latent dim / ant count，外层 ecology bundle 为 `digital-ant-ecology-checkpoint.v4`；禁止 pickle/object hook，跨 episode 显式排除未结算 runtime replay，embodiment 只存取 bytes。通用 session owner hydration 同样保存/恢复 `joint_loop.learning`（schema v5、runtime replay excluded）与 `reflection.consolidation_score`；world/self temporal 仍禁止成为第二 hydration writer，其状态只能作为 joint-loop owner archive 的两条独立 lane 存在 |
| ecology curriculum evidence | `vz-embodiment-ant` offline experiment owner | `digital-ant-ecology-curriculum.v14` 发布 mastery/interleaved 训练日程（butter/burning-match/composite 三阶段，木棍是中性物理几何、无 contact mastery/payoff）、P1 forced-return bootstrap（每个 body 在巢外专属黄油源上从未携食状态起步，经真实 contact 形成 `carrying_food: False→True`，再以左右均衡 `±3π/4` heading 训练拾取触发的动作族切换与归巢；后半程每 3 个 primary layout 交错复习一次，共 5 次；只初始化环境状态并同步 body-side PI，不发布坐标、目标方位或动作标签）、P1 forced-approach bootstrap（butter-near 专用：body 生成在拾取盘外、朝向偏离食物方位，生成半径 `1.45–2.9×拾取半径` 与偏离角 `0.4π–0.8π` 由 layout seed 逐 body 随机——固定生成环可被单一"固定曲率轨道"非定向解收割（首次 v22 实测 base policy 把基线转向从 0.083 放大到 0.15 rad 收割整块），随机 ensemble 下唯一通解是梯度转向；每次抽样仍保证直线路径最近距离 ≥1.38×拾取半径、必然错过拾取盘；只初始化状态并同步 PI，不发布坐标/目标方位/动作标签。动机：near 拾取盘的普通布局对食物梯度转向压力不足，food→turn 转向从 v10 至 v21 从未获得训练压力）、training/validation/held-out split、独立 scenario metrics、paired action probes（food/heat/home 发布方向对齐 truth，obstacle 只作 input-reachability 诊断）与 frozen gates；Digital Ant evidence profile 显式请求 `rank=n_z=16`、冻结 plant 支持的 `effective_dims=(0,1,2)`、opponent-coded actuator subspace `contrast_pairs=((0,1),)` 及 `exclusive_steering=True`（R2 所有权转移：base 确定性均值在 contrast pair 上被互补投影为 common mode，head 是 contrast 轴唯一学习型写入者，base 保留速度 common mode；见 temporal-abstraction spec。动机：v22/v22r 固定+随机 forced-approach 双重受控实验证明信用竞争下无约束 base 总用非定向"放大基线转向"退化解吸走转向信用，head 增益钉死 ≈1e-3 不增长）；temporal owner 必须在 live forward/sandbox/pure/torch 四条路径共用 head 与 base 两个投影，禁止 actuator-null common mode 吸收信用；且 `beta_t` 门必须按 contrast pair 共享（opponent-coded pair 是一根执行器轴，逐维门控会从"候选与旧码之差"凭空造出 contrast，实测零参数 head 因此仍产生 ±0.005 rad 转向并掩盖同量级学习信号）。exclusive steering 下冷启 head 精确为零、确定性策略无转向，因此 P1 pretraining 探针门只验 `input_reachable`（管线可达），转向能力由训练后 `paired_action_sensitivity`/`food_steering_alignment`/`carrying_home_action_alignment`/`post_pickup_uturn_progress` 硬门验收——冻结 U-turn lane 还必须在拾取后的前 2 个 action 内出现 `is_switching`，再满足交付或持续巢距下降；evaluation 只读且不回灌学习。P1 报告 schema 为 `digital-ant-ecology-p1-development.v31`（绑定 curriculum v14 的 typed milestone boundary），可恢复 journal 为 `digital-ant-ecology-p1-progress.v28`；旧 journal 必须 fail loudly，禁止把算法版本混在同一实验；长程 checkpoint 前由 Memory owner 把 explicit artifact 层有界到 8192 entries（CMS learned state 不裁剪，entry/index/pending/attribute 原子一致）；评估只读 owner snapshots，不回灌 reward |
| ecology alignment-formation attribution | `vz-embodiment-ant` offline experiment owner | `alignment-formation-attribution.v1` 只读消费 immutable station1/review verdict、owner-exported checkpoint archive 与 station-report journal；发布逐 body food-alignment margin、固定 seed perturbation、butter-near encounter/pickup 统计及 world causal action-head 参数几何。该 artifact 不恢复训练、不写 journal、不建立第二 learning owner；旧 journal 未发布的 per-tick gradient magnitude / per-update gradient cosine 必须标记 unavailable，禁止 consumer 重建。它只为 L1-B 选择 owner-level 机制方向，不能授权 station2/P1/P2 或 gate promotion |
| ecology alignment-formation protection precheck | `vz-embodiment-ant` offline experiment owner（机制 owner 为 `vz-temporal` causal action-head optimizer） | `alignment-formation-protection-precheck.v1` 只读恢复 L1-A 绑定的 immutable review checkpoint，在同一 probe seed 下比较 formation protection ACTIVE 与精确回滚 `DISABLED/0/1.0` 的完整 paired-probe payload；不得训练或写旧 journal。artifact 发布 source/checkpoint digest、机制声明、forward 等价 digest、逐 body food probe 与授权边界。PASS 只允许创建 L1-C fresh prereg，不授权 station run、station2/P1/P2 或 gate promotion |
| ecology station1-v4 preregistration | `vz-embodiment-ant` same-physics preregistration owner | `digital-ant-ecology-same-physics-baseline-preregistration.v3` 绑定 L1-B no-write precheck、完整 code-tree/source SHA256、同一 shared initial/schedule/physics 以及两臂共同的 formation protection `ACTIVE/160/0.25`；臂间唯一差异仍是 typed milestone wiring。冻结 station1 原阈值并新增直接 food alignment 4/4 gate，明确旧五局 review 不可再用。验证通过只授权隔离源码快照 + 新空 journal 的 ep0–19；station2/P1/P2 继续关闭 |
| ecology station1-v4 verdict | `vz-embodiment-ant` same-physics report owner | `digital-ant-ecology-same-physics-station1.v3` 只消费上述 v3 prereg 绑定的两臂各 20 局 progress，发布 pickup non-inferiority、非零 block、typed structure、8 条 structural/persistence lane 与逐 body food-turn alignment。station1-v4 已冻结发布 `BLOCK`：alignment 3/4、`alignment_review_authorized=false`、`next_episode_authorized=null`。该结果禁止 station2/P1/P2 与 ecology gate admission；不得通过第二次 review、换 seed、降门槛或加训练量改写 |
| ecology sense reflection transform | `vz-embodiment-ant` frozen substrate owner | `ant-sense.ecology-v2` 发布完整 19 维 signed involutive permutation：左右 receptor 交换，有方向的 pseudoscalar 取反，标量保持；`FinalRolloutConfig.internal_rl_causal_action_head_input_mirror_permutation/signs` 缺省 `None`（DISABLED/回滚），Digital Ant ecology profile 显式 ACTIVE。`vz-temporal` 只能执行并校验该正式交换，不得 import Ant schema 或重建感觉语义；live/pure/torch 共用 `0.5·(f(s) ± f(mirror(s)))` 群投影，runtime state/capture/transition/open segment 同时发布/持久化 mirror state。`joint_loop.learning` schema 为 v5；当前 P1 report/progress 为 `development.v31/progress.v28`，旧代 progress 必须按其各自 schema fail loudly |
| `ColonyRareHeavyBundle` | `vz-embodiment-ant` | per-individual artifact digest/provenance/gate verdict；不含 temporal state、不新增 slot |
| evidence manifest | `vz-embodiment-ant` | `digital-ant-manifest.v2` sidecar 绑定 artifact/input digest 与运行 provenance |
| realtime app DTO | `vz-embodiment-ant` | `digital-ant-app.v2` 的 frozen config/frame/status/command/disturbance；新增 typed object upsert/move/remove、`AppFrame.objects` 和 checkpoint provenance；**不**进入 §3/§6 slot 注册表 |
| relationship-conditioned Gate 2 selector state | `vz-temporal` offline selector owner（关系语义 owner 仍为 `vz-cognition` `RelationshipConditioningModule`） | `residual-state+relationship-owner-readout.v1` 在完整 residual state 后追加 owner 发布的有序 `ConditioningBankReadout`，只做 `(2x-1)×confidence` 有界化，禁止解释 label、原文或重建关系语义。只接受 non-cold、positive-confidence 的 `RELATIONSHIP` bank；错 bank/cold/zero confidence fail loudly，禁止静默回落 v35 无条件 8076 维 shape。该变换不新增 slot、不进 live session，只为 fresh Gate 2 longitudinal prereg/capture 提供新机制输入 |

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

**Offline N+1 representation exchange（2026-08-01）**：

`vz-substrate` 额外拥有研究用 `substrate_forward_representation` publisher；它不是
新的模型或第二个 embedding owner，而是对同一冻结 substrate capture 的正式解释。

```python
@dataclass(frozen=True)
class SubstrateForwardRepresentation:
    sample_id: str
    source_sha256: str                 # 只保存原文 SHA，不发布原文
    values: tuple[float, ...]          # L2-normalized residual readout
    values_sha256: str

@dataclass(frozen=True)
class SubstrateForwardRepresentationLineage:
    schema_version: str                # substrate-forward-representation.v1
    snapshot_fingerprint: str
    model_fingerprint: SubstrateFingerprint
    runtime_origin: str
    readout_kind: str                  # latest-token-selected-layer-residual-l2.v1
                                       # 或 latest-token-selected-layer-centered-residual-l2.v2
    layer_indices: tuple[int, ...]
    activation_widths: tuple[int, ...]
    representation_dim: int
    reference_corpus_id: str | None = None          # 仅 v2；v1 必须为 None
    reference_statistics_sha256: str | None = None  # 仅 v2；v1 必须为 None

@dataclass(frozen=True)
class SubstrateForwardRepresentationSnapshot:
    lineage: SubstrateForwardRepresentationLineage
    representations: tuple[SubstrateForwardRepresentation, ...]
    description: str
```

- publisher 只接受 `is_frozen=True`、完整 64-hex weights SHA、显式
  `runtime_origin` 的 runtime；model id 不一致、residual 缺失、跨样本 geometry
  漂移、非有限/零范数或 conditioned capture 全部 fail loudly。
- readout 由 source text 的最后 token、已选 residual layers 依 layer index 排序展平并
  L2 normalize；layer/width/model/readout/sample/value hashes 全部进入 snapshot
  fingerprint。consumer 不得另行遍历 `OpenWeightRuntimeCapture` 重建该向量。
- 原文只存在于 substrate capture 调用期间；公共交换只发布 `source_sha256`。
- `prediction_error` 的 offline `ForwardRepresentationBatch` 必须携带此 lineage，且
  target/persistence 与 `representation_dim` 同空间。head 首批绑定 lineage，后续
  batch 或 checkpoint 漂移立即失败。
- 此 slot 当前为 offline/report-only SHADOW，不进入 live `propagate` DAG；回滚为
  停止发布/消费该 slot，旧外部 sentence-encoder pilot 只能保留
  `thesis_status=not-evaluated`，不得自动恢复 thesis 资格。

**Centered readout v2（2026-08-12）**：

同一 owner 追加 `latest-token-selected-layer-centered-residual-l2.v2` readout。
背景：v1 原始 readout 有 55% 以上能量落在所有样本共享的均值方向上，且整体 L2
归一使单层（layer 20）独占约 73% 能量，压制了臂间可分辨性（A1 formal 的
`swapped-user-state` 主判据 gain 仅 5.9e-05 即此缺陷的直接后果）。v2 变换：
逐层 L2 归一 → 拼接 → 减去冻结参考均值（可选再投影掉冻结主成分）→ 整体归一。

```python
@dataclass(frozen=True)
class SubstrateReadoutReferenceStatistics:
    schema_version: str                # substrate-readout-reference-statistics.v1
    corpus_id: str                     # 冻结参考语料标识（train split，非 heldout）
    layer_indices: tuple[int, ...]
    activation_widths: tuple[int, ...]
    sample_count: int
    mean: tuple[float, ...]            # 在逐层归一空间拟合
    principal_components: tuple[tuple[float, ...], ...]  # 可为空；单位正交
    statistics_sha256: str             # 内容自校验，构造即验证
```

- **owner**：仍为 `SubstrateForwardRepresentationPublisher`（`vz-substrate`），
  不新建 slot、不新建第二 owner；v2 通过构造参数 `reference_statistics` 启用。
- **value_type**：`SubstrateForwardRepresentationSnapshot` 不变；v2 lineage 额外
  携带 `reference_corpus_id` 与 `reference_statistics_sha256`，v1 lineage 两者
  必须为 None，交叉校验 fail loudly。
- **dependencies**：`fit_forward_readout_reference_statistics(...)` 只能在冻结
  train-split 参考语料上拟合（在 evaluation/heldout 数据上拟合即判据面泄漏，
  契约禁止）；statistics 作为 model-bound artifact 随 prereg 冻结发布。
- **whitening 档位（2026-08-12 用 d 值定夺）**：在 583 个 MSC train 样本
  （24 dyad）上，v1 的 same-vs-diff dyad Cohen's d = 0.315；v2 逐层归一 + 减均值
  d = 0.432；再去 top-1 主成分 d = 0.592（+88% vs v1）；去 ≥2 个主成分反而塌到
  0.21–0.27（第二主成分携带判别方差）。冻结推荐 `principal_component_count=1`。
  证据：`artifacts/readout_discrimination_v2_20260812/`（report_pc0/1/2/4 +
  reference_statistics + run.log）。诚实记录：v2 的 1-NN dyad 检索为 0.189，低于
  v1 的 0.254（v1 的检索优势来自 L20 主导的局部 token 相似性）；判据对齐的是
  paired 对比效应（d 类量），故 d 是冻结依据，检索数一并留档。
- **wiring_level**：与 v1 相同，SHADOW（offline research），不进入 live DAG。
- **兼容性**：v1 snapshot fingerprint 逐字节不变（fingerprint payload 仅在 v2
  时追加 `reference_statistics_sha256` 键）；capture 与 statistics 的
  layer/width geometry 不一致时 publish fail loudly。
- **回滚**：停止传入 `reference_statistics` 即回到 v1 行为；既有 v1 artifact
  与 fingerprint 不受影响。

**MSC R4 conditioned runtime context（2026-08-05）**：

`publish_runtime_capture_representation(...)` 仍由 `vz-substrate` 解释实际 generation
capture 的 residual geometry，但与 target publisher 的语义严格分开：target 必须
unconditioned；R4 context 允许包含正式 runtime 的 bounded conditioning。两者共享
同一冻结 model/weights/readout/layers/widths 坐标，consumer 只能比较 owner 发布的
向量和 lineage，不能遍历 capture。

service 的 evidence-only `msc-runtime-collector-v1` 通过外层 DTO 发布：

- `context_representation`（values/value SHA/source SHA）与 `context_lineage`；
- 实际 `temporal_n_z`、`active_speaker_id`、完整 runtime slot-surface SHA；
- input/output/total token、generation 与 end-to-end latency、propagate event count；
- `acceptance_passed=true`、`substrate_fallback_active=false`、
  `raw_text_retained=false`、`evaluation_writeback_allowed=false`。

MSC evidence profile 还必须在 startup attestation 与 collection attestation 中绑定
`semantic_proposal_channel`：A2 runtime collector 固定 `noop`，隔离不属于 PE/ETA
intervention 的生成式 proposal collaborator；C3 steering collector 固定 `llm`，与未来
production ACTIVE 的语义状态来源保持一致。两者都不移除语义 owner 或快照，也都禁止继承
环境变量造成 smoke/formal 漂移。普通 companion 的 proposal channel 仍默认为 `llm`。

B3 activation deployment contract 必须继续绑定 C3 的实际 model dtype 与
`semantic_proposal_channel=llm`；authorization 同时核对 service CLI 参数，canary argv
显式携带 dtype，并拒绝 steering/semantic 环境 override。否则 C3 的 SHADOW 运行面与
ACTIVE service 不是同一 lineage，不得晋升。

该 DTO 是 offline evidence exchange，不注册新的 live slot。只有明确 MSC profile、
typed observation permission、精确 frozen weights SHA、显式 residual layers/width/context
limit 与 `temporal_n_z ∈ {3,16,64,256}` 时才可发布；companion temporal bootstrap 在此
profile 禁用，使 `BrainConfig.temporal_latent_dim` 成为唯一容量 owner。普通 service
默认与产品 wiring 不变。每个 capacity/dyad checkpoint 绑定这些字段并仅保存向量、hash、
sample id 与成本；任何 drift、fallback、截断、非有限/零范数或 raw-text retention 均
fail loudly。

**Offline frozen residual readout exchange（2026-08-04）**：

S1 在上述正式 representation 之上增加唯一 owner
`SubstrateResidualReadoutPublisher`；禁止 consumer 再次遍历 runtime capture 或自行
拟合第二套 residual classifier。

```python
@dataclass(frozen=True)
class FrozenResidualReadoutArtifact:
    model_fingerprint: SubstrateFingerprint
    runtime_origin: str
    source_readout_kind: str
    layer_indices: tuple[int, ...]
    activation_widths: tuple[int, ...]
    representation_dim: int
    class_ids: tuple[str, ...]                 # 外部 typed owner 提供的不透明 id
    ridge_alpha: float
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    class_weights: tuple[tuple[float, ...], ...]
    class_biases: tuple[float, ...]
    class_axes: tuple[tuple[float, ...], ...]  # class-vs-rest，逐轴 L2-normalized
    training_snapshot_fingerprint: str
    training_labels_sha256: str
    training_support: int

@dataclass(frozen=True)
class SubstrateResidualReadout:
    sample_id: str
    source_sha256: str
    class_scores: tuple[tuple[str, float], ...]
    predicted_class_id: str
    score_margin: float

@dataclass(frozen=True)
class SubstrateResidualReadoutLineage:
    schema_version: str                    # substrate-residual-readout.v1
    artifact_id: str                       # frozen-residual-readout.v1:<sha256>
    model_fingerprint: SubstrateFingerprint
    runtime_origin: str
    source_snapshot_fingerprint: str
    source_readout_kind: str
    layer_indices: tuple[int, ...]
    activation_widths: tuple[int, ...]
    representation_dim: int

@dataclass(frozen=True)
class SubstrateResidualReadoutSnapshot:
    lineage: SubstrateResidualReadoutLineage
    readouts: tuple[SubstrateResidualReadout, ...]
    description: str
```

- fit 只消费一个 immutable `SubstrateForwardRepresentationSnapshot` 与按
  `sample_id` 精确对齐的 typed label；label 语义仍归调用方 owner，substrate 只把
  class id 当不透明枚举，不解析文本、不建立业务语义 owner。
- artifact 使用 closed-form standardized ridge one-hot fit；standardization 已折入
  effective class weight / bias。每条 steering 候选轴固定为“本类 weight − 其他类
  weight 均值”后 L2 normalize，轴本身**不含 bias**，不得恢复 Stage-3 已定罪的
  free-bias 通道。
- publisher 必须逐项核验 model weights fingerprint、runtime origin、readout kind、
  layer/width/dimension；任何 lineage 漂移 fail loudly。公共快照只含 source SHA、
  class score / prediction / margin 与 artifact/source lineage，不含原文。
- 当前 slot 仅用于 S1/S2 offline evidence，默认 SHADOW，不进入 live
  `propagate` DAG，不授权安装 artifact、动作选择或学习回灌。S1 evaluation
  readout 只能决定是否准入另立预注册的 S2，不能成为训练信号。
- 回滚为停止发布/消费 `substrate_residual_readout` 并保留既有
  `substrate_forward_representation`；production wiring 与冻结基底均不变。

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
- Windows/CUDA 长 context 证据面由 `vz-substrate` 唯一拥有并解释。显式 opt-in
  `WINDOWS_CUDA_CUDNN_SDPA_CACHED_STRICT_V1` 固定 Windows + CUDA、Qwen 声明的
  32768-token context、`bfloat16`、`attn_implementation=sdpa`、exclusive cuDNN SDPA、
  generation-only cached decode、strict-local、`SubstrateFallbackMode.DENY`、fail-on-truncation 与
  generation-only `first-full-prompt-set-once` capture；加载必须同时预绑定 logical model ID、
  verified revision、权重 SHA 和全部非权重 execution-assets SHA。任一平台、device、dtype、
  模型原生 context、attention、generation cache/backend 或 hook layer 漂移均 fail loudly。该 profile 只在调用方
  显式传入时生效；`execution_profile=None` 保持现有 Windows eager / generation
  `use_cache=False` / pooled capture 保护与既有 fallback 行为不变。
- strict runtime 在加载后发布 owner-internal、frozen、content-addressed
  `TransformersExecutionAttestation`，完整记录 profile/model/revision/weights/execution-assets/runtime/platform/
  torch/transformers/CUDA/cuDNN/device/dtype/attention/generation-cache/context/hook lineage；它不进入
  `SubstrateSnapshot`。每次成功生成的 `GenerationResult` 只绑定同一
  `execution_attestation_id`，并携带 frozen `GenerationContextBudgetAttestation`，避免在
  consumer 侧重建 runtime 状态或把完整启动自证复制到每 turn。
- combined context budget 必须由 substrate owner 在 chat template 已实际 tokenize、所有
  character/personal/relationship Prefix-KV 已合并、`effective_max_new_tokens` 已结算之后，
  且在注册 hook 或首次 model forward 之前检查：`actual_input_tokens +
  actual_prefix_slots + effective_max_new_tokens <= model.config.max_position_embeddings`。
  strict generation 的显式 messages 与 plain `prompt/system_context` API 都必须先归一化为
  messages，再走 `apply_chat_template(tokenize=True)` 与同一 fail-loud 校验；禁止以 plain
  tokenizer 绕过模板后截断或超预算。预算 attestation 发布上述实际计数、原生上限与剩余量。
- cached generation 的 residual hook 只在第一次完整 prompt forward 按 layer set-once；
  后续 single-token decode 仍执行有界 delta/hook，但不得覆盖首个 full-prompt capture。
  strict profile 要求 capture token 数与实际 prompt token 数一致，缺 layer、shape 漂移或
  capture 失败均 re-raise，不允许 warning 后返回 `capture=None`。
- standalone capture 与 differentiable instrumental scorer 不伪称 generation cache/template：两者固定
  raw-tokenizer / `use_cache=False`，但 probe、full forward、prefix build 与 prefix-cache upper replay
  全部复用 strict exclusive cuDNN SDPA context；legacy profile 仍为 `nullcontext`。
- 上述 profile、execution attestation 与 generation budget 都是既有 substrate owner 的
  执行/证据契约：**不新增 runtime slot，不改变 `substrate` value shape，也不改变
  `steering_condition_belief / steering_gate_decision / steering_intervention` 三件套 shape**。
- strict 32767+1 诊断使用以下离线公共交换注册；它不进入 live `propagate` DAG：

  | Exchange | Owner | Value Type | Dependencies | Wiring Level |
  |---|---|---|---|---|
  | `strict_capture_audit_summary` | `audit_strict_capture` (`vz-substrate`) | frozen `StrictCaptureAuditSummary` | public frozen `OpenWeightRuntimeCapture`；预注册 layer/width | `DISABLED`（live）；仅 offline engineering diagnostic |
  | `windows_event_log_source_provisioning_audit` | privileged `provision_volvence_evidence_event_log.ps1` infrastructure owner；qualification/campaign 禁止自动调用 `Provision` | 成功/不合规路径发布 raw `volvence-evidence-event-log-provisioning-audit.v2`：含 source/config v1 contract、machine/config content ID、registry/channel before/after、SDDL/owner、script/module/assembly observations、provisioning stages、refresh 三态与 safety boundary。异常路径另发 best-effort process-failure v1：可能只有 partial/null control-plane observations，不含完整 fixed contract、machine-config content ID 或 basis；consumer 禁止把它当 config snapshot。self hash/content ID/provenance 均不是身份签名，Node protocol pin 只固定 reviewed source | elevated manual Provision；source 缺失时必须显式 `-AllowSourceCreation`，该 intent 不证明首次 bootstrap；不读/写 records、不自动修复 drift。Audit exact conformance=false→exit 2；process/mutation/observation failure→receipt + exit 3。mutation 非事务性，注册后失败或独立 observed absent→present 会要求 refresh；未创建且无法确认时 refresh requirement 为 null。创建 Provision 只接受 provider 列表不变或精确新增 `VolvenceEvidence`；前者仍等待 refresh。fresh Audit before/after 必须同时包含 exact source membership 且完整 provider endpoints 相等。endpoint equality 仍不证明连续稳定；artifact adapter core 只复算 raw self-consistency，production acquisition、failure quarantine、独立 live reobservation 与 refresh chronology 仍须后续 owner 包关闭 | `DISABLED`（live）；管理员显式运维；pure artifact adapter core 已存在但 production adapter/integration 未实现；本地伪造、中途改回、module-qualified self-observation 的高级篡改，以及参数绑定/parse、stdout/进程致命失败均未排除；Provision/failure 输出不授权资格 |
  | `windows_event_log_source_audit_artifact_adapter_snapshot` | `windows_cuda_host_stability_qualification` (`vz-runtime` offline evidence owner)；唯一 adapter publisher；qualification projection consumer 尚未接线 | 深冻结 `windows-event-log-source-audit-artifact-adapter-snapshot.v1`：绑定 protocol ID/raw、single-descriptor stdout raw SHA/byte count、non-authoritative capture-envelope content ID/role/exit/machine/boot claims、reviewed provisioner pin、machine-config basis、全量 recomputed conformance 与 snapshot ID；exit 2 只发布 diagnostic。固定 `projection_emitted=false / real_provisioner_observation=false / eligible=false` 及全部 authorization=false | 完整 Audit v2 + caller `windows-event-log-source-audit-capture-envelope.v1` + 外部 protocol ID/raw pin；严格 compact ordered JSON、0/2 与 failure-v1/3 分流、1 MiB pre-read budget、100ns chronology、source/ACL/owner/channel/provider/registry/safety/refresh 复算。envelope 非权威；stderr/OS outcome/exe/argv 未由 adapter 直接采集；after 的 caller content-ID match 不是 scope proof；即使 acquisition v2 冻结 requested same-buffer binding，adapter 仍不独立 attest realized source/executable/process origin。SDDL owner derivation、module/assembly trust、replay 与 live reobservation 未关闭 | `DISABLED`；非 production artifact self-consistency diagnostic；不得作为 002/008 projection、host qualification、CUDA、formal evidence、ACTIVE 或四能力输入；回滚为停止调用 adapter |
  | `windows_event_log_source_audit_acquisition_outcome` | standalone `windows_event_log_source_audit_acquisition` (`vz-runtime` offline evidence owner)；只允许显式 operator invocation，qualification/campaign 不得自动调用；Failure-v1 quarantine、process supervision 与 stream finalization 由同一 owner 判别，避免失败输出越过 adapter 边界 | protocol v2 `3ed1005f…be5be0` / raw `b01c3ae2…be1c1b`；深冻结 discriminated outcome：`audit_v2_conformant_capture_candidate / audit_v2_nonconformant_capture_candidate / failure_v1_quarantined / unclassified_process_or_output_quarantined`。create-only 000 claim、raw stdout/stderr 与 terminal v2 发布 requested same-buffer source binding、nullable exit/close observation、soft timeout/hard cutoff、per-stream pipe/persistence status、pre/post diagnostics、exact non-authoritative adapter envelope 或 quarantine。真实 candidate 必须 child exit+close、双 pipe end+close、双流完整且无 kill/timeout/error/cutoff；失败臂 `captureEnvelope=null`，禁止 source config snapshot/projection | fixed WinPS5.1 `-EncodedCommand` launcher 从 repository cwd 以 FileShare.Read 同柄 bounded-read reviewed provisioner，核对 raw+strict-UTF8 LF pins，Parser.ParseInput exact buffer 并固定 Audit；handle 只声明贯穿 bound-script execution/exit unwind，不声明持有至 OS exit。reviewed provisioner 强制 fresh-checkout `eol=lf`，critical-source LF hash 保留 BOM。120s timeout+5s drain grace+125s overall async supervision 在 close 永不到达时仍 cutoff，late error guardian 到 child close 只报告、不升级。claim 先 fsync；finalize 后 stream write/fsync/same-fd readback。write/fsync 可观测失败若可 readback 则 quarantine；readback/terminal failure 只能 incomplete root。单文件 fsync 不保证 directory durability；cross-root duplicate 未排除。requested binding 不 attest trusted SystemRoot/realized PowerShell image、IFEO/environment、ancestor reparse/admin/kernel；无 Job Object、native reobserver、WORM anchor 或 qualification admission | `DISABLED`；public gate 在读取 options/path/env 前静态禁用，白名单 synthetic/Windows fixture 只验证 contract/mechanism；`acquisition_to_qualification=DISABLED`，固定 `real/eligible/CUDA/formal/ACTIVE/four_capabilities/tamper_resistance=false`；回滚为停止固定 CLI/owner，既有 complete/incomplete 根均保留为 immutable non-evidence |
  | `windows_cuda_host_stability_qualification_terminal` | `windows_cuda_host_stability_qualification` (`vz-runtime` offline evidence owner)；consumer 禁止接受孤立 terminal、自签 eligibility 或重建 producer 判词 | protocol `32f35e4f…e1d5f` / raw `30a88183…c4132`；synthetic create-only 000–009 receipts + 010 manifest + 011 `windows-cuda-host-stability-qualification-terminal.v2` + two streams；synthetic full-root snapshot 发布 Application/System handoff cursors、firmware human/machine distinction、criteria/observation/eligibility 三态；future consumer 必须同时绑定 artifact ID、terminal ID、terminal raw SHA，自报 eligibility 非权威 | host-block raw SHA `5e02aec7…a34f`、stable operator-declaration ID、numeric LE microcode、same-machine/same-boot、cooldown/tail 与 normalized fault fields。当前 002/008 仍只是 synthetic-only `windows-cuda-host-stability-source-audit-projection.v2` 且 `full_raw_audit_bound=false`；artifact adapter snapshot 未接入 projection，仍无 production probe/direct acquisition/live reobserver/raw Event XML parser；旧 block 无 MachineGuid，same-chassis 只能人工声明；outer cursor adoption 是后续独立 consumer 包 | `DISABLED`；public production publisher/validator 静态禁用，只有 synthetic validator 且 `validatedEligible=false`；current outer v1 exact-schema 拒绝 terminal v2；不得授权 CUDA、production ACTIVE 或四能力主张 |
  | `strict_32k_outer_attempt_lease` | `windows_cuda_strict_32k_host_campaign` (`vz-runtime` offline campaign control) | `002_preregistration.json` canonical raw SHA-256；64-hex，一次性、backend-separated exact scope | outer/child protocol ID、qualification artifact ID、host identity、execution backend、000–003 prefix；004 必须在 child creation 前 fsync 消费 | `DISABLED`（live）；production 禁用；synthetic lease 明示 non-evidence |
  | `strict_32k_host_campaign_receipt_chain` | `windows_cuda_strict_32k_host_campaign` (`vz-runtime` offline campaign control) | create-only 000–012 receipts + two stream logs + child exact-five artifact；complete failure seal 与 `incomplete_consumed` 分离 | lease、same-machine/same-boot prelaunch、Circular Application/System cursor/boundary、Node-recomputed delta、anchor/delta cross-binding、fixed child argv；完整 child local import/source closure、producer PASS-return 前整根重验与 terminal/delayed-fault 区间仍未关闭 | `DISABLED`（live）；source-checkout-only scaffold；protocol `cf62484f…3194`，production 禁用 |

  token-level residual、feature-surface 名称、有限性与 geometry 只能由该 owner 解释；
  `vz-runtime` evidence orchestrator 只读 bounded summary，不遍历 capture、不建立第二 owner。
  owner 的 residual hash 必须 framing 实际 sequence position/step、activation cardinality、
  layer/activation step、width 与值，并单独发布顶层 latest residual 是否与 sequence 末步一致；
  duck-typed nested records、非 tuple 容器或非 exact float 值一律 fail loudly。
  smoke runner 只绑定并回显 outer lease，不得自行签发、替换或消费第二个 lease。child 本地
  no-retry 只覆盖冻结输出根；outer no-retry 只覆盖由 outer/child protocol、qualification artifact、
  host identity 与 backend 构成的 exact scope，新 qualification/protocol/backend 是新 scope，禁止写成
  “全局跨根择优已排除”。standalone artifact 不得把任意 lease 字符串解释成已预注册。

- 四能力轴：本组交换是 offline evidence infrastructure，不写 CMS/State-KV/semantic snapshot，
  不发布 live named readout，不消费 PE→credit 学习信号，也不施加 bounded steering；因此它本身对
  Appendable / Readable / Learnable / Steerable 四轴均记为 `not_proven`。synthetic receipt chain
  强制 non-PASS；未来工程 PASS 也不能升级任一能力轴。
  回滚为停止离线诊断调用并删除 additive exchange；既有 `substrate` slot、strict runtime
  和 steering 三件套逐字节不变。
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
- `lifecycle_metrics` 只发布 owner 自身负责的 nested lifecycle telemetry；
  消费者不得自行推断 reset、slow-to-fast transfer、或
  learned-core-guided recall 是否发生。其中
  `slow_to_fast_init_benefit` 的 value type 为 finite `float`，owner 为
  `vz-memory MemoryStore`，dependencies 为当前 reset mode、owner-internal
  copy-init shadow 与 reset 后最多 5 次 replay loss，wiring level 跟随
  `memory` slot；值定义为 `mean(copy_shadow_loss - active_init_loss)`，
  没有 matched observation 时为 `0.0`。consumer 禁止以 band 向量位移
  重算该值。`CMSContextInitializationEvidence` 可发布
  `context_conditioned / prototype_count / context_match_score`，只证明
  initializer 选择，不授权外部写入 prototype。
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

**`CreditRecord.level` 取值**：`token` / `turn` / `session` / `long_term` / `abstract_action` / `prediction_error` / `evaluation_readout` / `social_prediction_error` / `abstract_action_segment` / `steering_terminal_prediction_error`（C1：PE owner 的 matched-noop N+1 terminal mismatch，按 gate decision lineage 路由）/ `counterfactual_contribution`（Phase 1.A COCOA-lightweight，readout-only，不参与 acceptance gate）/ `counterfactual_contribution_learned`（Phase 2.A learned rewarding-state head readout，SHADOW 默认）。

**当前实现口径**：

- P06 当前落地结构化信用记录、gate audit 与 bounded self-modification proposal；默认 `CreditModule` 以 `SHADOW` 接线运行，真实写入仍必须通过对应 owner 的 apply surface 和 gate，而不是由 credit owner 直接突变外部模块
- `recent_modifications` 当前记录 allow / block decision，作为审计轨迹和后续 reflection 输入
- `cumulative_credit_by_level` 先提供最小聚合，后续再扩展到更细粒度的长期统计
- 第二阶段允许在 owner 内部基于 temporal / rollout 结果扩展出 `abstract_action` 级 credit；共享 shape 变更必须按本文件协议登记
- metacontroller credit 当前会消费 posterior drift、binary gate ratio、policy replacement score 等 ETA kernel evidence，并将其压入 `CreditRecord.context`
- `derive_credit_records_from_prediction_error_first(...)` 是当前 PE-first credit 派生路径；evaluation 只提供 readout / gate context，不重新成为原始学习源
- C1 的 `SteeringTerminalPredictionError` 是 PE owner 发布的 out-of-turn typed settlement，不新增 slot；Credit owner 以 `(episode_id, prediction_head_fingerprint)` 去重并按 `decision_id` 发布 `steering_terminal_prediction_error` records，gate owner 只消费 pending lineage。action/noop 必须同 predictor、同 substrate target lineage、同 target coordinate 且均为 frozen heldout forward；evaluation/judge 不得进入该链
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
class RelationshipUpdateProposal:
    proposal_id: str
    target_owner_slot: str
    operation: str
    human_readable_description: str
    source_evidence: tuple[str, ...]
    confidence: float
    requires_user_confirmation: bool = True
    shadow_only: bool = True

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
    relationship_update_proposals: tuple[RelationshipUpdateProposal, ...] = ()
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
- `relationship_update_proposals` 是 reflection owner 从 consolidation、typed tension / lesson 与 PE failure readout 派生的用户可审阅提案；consumer 禁止读取原始用户文本重建描述或目标 owner
- P1 中所有关系更新提案均为 `shadow_only=True`、`requires_user_confirmation=True`；字段默认空 tuple，旧快照和持久化数据保持兼容

**消费者**：记忆系统、信用分配、Metacontroller、认知 Regime 层；lifeform-service 关系记忆 console 只读消费提案
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
- `PredictionErrorModule` 的 `ForwardRepresentationBatch` / `ForwardRepresentationBatchSnapshot` 是 PE owner 的离线 N+1 表示预测研究面：冻结 encoder 提供 context/target 表示，PE owner 负责预测、同 target loss 与 mismatch 结算。它绕过逐 turn `propagate`，但不是 runtime `Snapshot`、不新增 §6 slot、不改变本节四轴 live schema；consumer 禁止直接构造内部 head 或把 batch artifact 注册成第二 owner。
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

substrate ──────────────→ steering_condition_belief ──┬─→ steering_gate_decision
                                                     └─→ steering_intervention
prediction_error ──────────────────────────────────↗
steering_gate_decision ─────────────────────────────→ steering_intervention

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

steering 族的跨 wheel 新行于 2026-08-05 按「owner / value type /
dependencies / wiring level」四元组先行冻结；主表的「消费者」列仍只表示直接下游：

| Slot | Owner | Value Type | Dependencies | Default Wiring |
|------|-------|------------|--------------|----------------|
| `steering_condition_belief` | `SteeringSensorModule` (`vz-cognition`) | `SteeringConditionBelief` | `substrate` | SHADOW |
| `steering_gate_decision` | `SteeringGateModule` (`vz-temporal`) | `SteeringGateDecision` | `steering_condition_belief`, `prediction_error` | SHADOW |
| `steering_intervention` | `SteeringExecutorModule` (`vz-substrate`) | `SteeringIntervention` | `substrate`, `steering_condition_belief`, `steering_gate_decision` | SHADOW |

三项 value 与冻结 artifact 均由 `vz-contracts` 的
`steering_contracts.py` 唯一定义；完整 shape、模型权重指纹绑定、
SHADOW/ACTIVE 可见性与回滚见
[`docs/specs/steering-runtime.md`](./specs/steering-runtime.md)。

`steering_intervention` 在 SHADOW 证据 profile 中可附带 substrate owner 解释的
`noop_context / action_context / sensor_off_action_context`：它们是固定 layer order、
L2-normalized、只含数值与 SHA-256 的冻结 DTO，不新增 slot，也不把原文交给 consumer。
sensor-off executor 是 `SteeringArtifactBundle` 内的 matched-budget unconditional artifact，
只能作为 `steering_intervention` owner 的证据分支；consumer 禁止重建 residual 或第二次
解释 condition code。C3 的 trace/report 与 B3 promotion evidence 都是 offline artifact，
不进入 live DAG，因此不在 slot 注册表创建第二 owner。

| Slot Name | Owner 模块 | Value 类型 | 默认接线 | 发布频率 | 消费者 |
|-----------|-----------|-----------|----------|----------|--------|
| `substrate` | SubstrateModule | SubstrateSnapshot | SHADOW | 每 turn | temporal_abstraction, memory, dual_track, evaluation, prediction_error |
| `substrate_forward_representation` | SubstrateForwardRepresentationPublisher (`vz-substrate`) | SubstrateForwardRepresentationSnapshot | SHADOW（offline research） | frozen corpus batch | prediction_error offline ForwardRepresentationBatch；不进入 live DAG |
| `substrate_residual_readout` | SubstrateResidualReadoutPublisher (`vz-substrate`) | SubstrateResidualReadoutSnapshot（artifact lineage 指向 FrozenResidualReadoutArtifact） | SHADOW（offline evidence） | frozen corpus batch | ETA S2 causal-steering evidence；依赖 `substrate_forward_representation`，不进入 live DAG、不回灌 |
| `steering_condition_belief` | SteeringSensorModule (`vz-cognition`) | SteeringConditionBelief | SHADOW | 每 turn（仅注入 model-bound artifact 时构造） | `steering_gate_decision`, `steering_intervention` |
| `steering_gate_decision` | SteeringGateModule (`vz-temporal`) | SteeringGateDecision | SHADOW | 每 turn（仅注入 model-bound artifact 时构造） | `steering_intervention`；只消费 belief + PE owner 快照，不消费 evaluation |
| `steering_intervention` | SteeringExecutorModule (`vz-substrate`) | SteeringIntervention | SHADOW | 每 turn（仅注入 model-bound artifact 时构造） | session / transformers response generation；仅 ACTIVE 快照可进入用户可见生成，SHADOW 预览只留在 `shadow_snapshots` |
| `substrate_self_mod` | SubstrateSelfModModule | SubstrateSelfModSnapshot | SHADOW | 每 turn / schedule | session / credit audit / rare-heavy review |
| `world_temporal` | TrackTemporalModule | TemporalAbstractionSnapshot | SHADOW | 每 turn | temporal_abstraction, dual_track |
| `self_temporal` | TrackTemporalModule | TemporalAbstractionSnapshot（P3：可携带 `TemporalActionAdvisoryProposal` + `action_advisory_status`） | SHADOW | 每 turn | temporal_abstraction, dual_track；relationship advisory 默认只记录、不改 native action |
| `world_temporal_consolidation` | TrackTemporalConsolidationModule | TemporalConsolidationSnapshot | SHADOW | 每 turn | final wiring / audit only |
| `self_temporal_consolidation` | TrackTemporalConsolidationModule | TemporalConsolidationSnapshot | SHADOW | 每 turn | final wiring / audit only |
| `temporal_abstraction` | TemporalAggregateModule / TemporalModule | TemporalAbstractionSnapshot | SHADOW | 每 turn | memory, dual_track |
| `memory` | MemoryModule | MemorySnapshot | SHADOW | 每 turn ~ 每会话 | dual_track, regime, reflection, temporal_abstraction, evaluation |
| `plan_intent` | PlanIntentModule | PlanIntentSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation, session-post evidence |
| `commitment` | CommitmentModule | CommitmentSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation, session-post evidence |
| `open_loop` | OpenLoopModule | OpenLoopSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation, session-post evidence；#90：额外依赖 `apprenticeship_alignment`（消费其 `should_request_feedback`），快照新增 `apprenticeship_verification_requests` |
| `user_model` | UserModelModule | UserModelSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation, session-post evidence；显式自述 profile fact 通过 `SemanticProposal.semantic_key/canonical_value` 单写入，owner 按 key 发布最新有效 `profile_facts`、核实请求键与 `profile_context_statement`；纠正/撤回对应 `REVISE/BLOCK`，consumer 禁止解析原始记录重建事实 |
| `execution_result` | ExecutionResultModule | ExecutionResultSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation, prediction-error evidence |
| `belief_assumption` | BeliefAssumptionModule | BeliefAssumptionSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation |
| `relationship_state` | RelationshipStateModule | RelationshipStateSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation |
| `goal_value` | GoalValueModule | GoalValueSnapshot | ACTIVE | 每 turn | temporal, response_assembly, evaluation |
| `boundary_consent` | BoundaryConsentModule | BoundaryConsentSnapshot | ACTIVE | 每 turn | temporal, boundary_policy, response_assembly, evaluation |
| `personal_conditioning` | PersonalConditioningModule | PersonalConditioningSnapshot | SHADOW | 每 turn | session / open-weight substrate generation；只消费 `user_model`、`relationship_state`、`goal_value`、`boundary_consent` 的 typed owner readout，不读取原始对话或重建语义。State KV P0-b 起 value 增加 `rendered_statement`（owner 用确定性模板把同一 typed readout 渲染成英文状态说明，cold-start 必为空串）；ACTIVE 时投递形态由 `FinalRolloutConfig.personal_conditioning_mode` 决定（`residual`=残差通道/臂 E，`text`=system prompt 状态段/臂 B′，两者互斥），详见 [`docs/specs/personal-conditioning.md`](./specs/personal-conditioning.md) §3.1。第二个开关 `FinalRolloutConfig.prompt_state_delivery`（`text` 默认 / `suppressed`）决定**状态派生段是否进入 system prompt**：`suppressed` 下 `build_system_prompt` 只组装不变表达规则段，得到 prompt 逐字节相同的 `state-kv-arm-a-pure` / `state-kv-arm-e-pure` 载体识别臂；与 `personal_conditioning_mode="text"` 组合为非法配置（构造期 raise）。`suppressed` 为证据专用、禁止部署（移除 boundary/disclaimer prompt 引导），详见 [`docs/specs/state-kv-identification-evidence.md`](./specs/state-kv-identification-evidence.md) §3.2。载体识别证据由 `vz-runtime` 的 `state_kv_identification` 只读 owner 产出：schema `state-kv-identification.v1`，四臂设置从 profile registry 解析（不另建 arm 表），逐 turn 证据只从已发出的 `prompt_fp` / `prompt_state_sections` / `decode_fp` / `personal_conditioning` tag 回读（禁止重算"应该等价"的 prompt），四条判据与 C5 分档写入 `artifacts/state_kv/verdict_identification.json`；`SubstrateEvidenceKind.TRACE_ONLY` 下 verdict 强制封顶 `insufficient_data`，无盲裁判时判据 3/4 记 `insufficient_data` 而非填充占位准确率；matching 准确率是 readout，不回灌学习链路（R12）。State KV P5-c 起 value 新增 `credit_confidence_delta`（有界 credit 反馈 readout，语义与门控见 `relationship_conditioning` 行） |
| `relationship_conditioning` | RelationshipConditioningModule | ConditioningBankReadout | SHADOW | 每 turn | 第二个 conditioning bank（`bank_type=RELATIONSHIP`）；只消费 `relationship_state` + `boundary_consent` typed readout，compiler `relationship-conditioning.v2` 发布 14 维 dyad 长程状态和同信息 `rendered_statement`，compiler version 进入 `source_fingerprint`。runtime 以 `bank_readout_to_bank(...)` 注入 scope / freshness / revocation。ACTIVE 投递由独立 `relationship_conditioning_mode` 选择：默认 `text` 合并 owner statement；`residual` 使用版本化 residual projector；`prefix_kv` 使用 Relationship 专属、内容寻址的 Prefix-KV artifact。三路互斥且都进入 turn lineage。默认 SHADOW；回滚为 `text`、`SHADOW` 或 `DISABLED`。P5-c 的 `credit_confidence_delta` 仍由 owner 唯一写入，`conditioning_credit_feedback` 默认 SHADOW、ACTIVE 有界施加、DISABLED 停止消费。详见 [`docs/specs/personal-conditioning.md`](./specs/personal-conditioning.md) bank 家族节 |
| `dual_track` | DualTrackModule | DualTrackSnapshot | SHADOW | 每 turn | memory, evaluation, prediction_error, reflection, credit, regime |
| `apprenticeship_alignment` | ApprenticeshipAlignmentModule | ApprenticeshipAlignmentSnapshot | ACTIVE | 每 turn（学徒/ingestion） | prediction_error（离散事件 PE 源）；belief_assumption / goal_value（经 SemanticProposal 单写）；apprenticeship_protocol_alignment（消费 enriched `guidance_constraints`）；**open_loop（#90：消费 `should_request_feedback` 冒出 verification 开环 actuator）**；#90 起 ACTIVE，仅 apprentice/ingestion turn 生效（普通轮 idle → PE overlay + 请求均 no-op），快照新增 `should_request_feedback` / `feedback_request_reason` / `feedback_request_urgency`；详见 [`docs/specs/apprenticeship-alignment.md`](./specs/apprenticeship-alignment.md) |
| `apprenticeship_protocol_alignment` | ApprenticeshipProtocolAlignmentModule (vz-application) | ApprenticeshipProtocolAlignmentSnapshot | SHADOW | 每 turn（学徒/ingestion） | 把 `apprenticeship_alignment.guidance_constraints` 与编译后 protocol 工件（strategy_playbook / domain_knowledge / boundary_policy）做有限选项集层比对；A1（#90 残余，2026-07-16）：快照新增 `pe_overlay_magnitude` / `pe_overlay_source`（结构裁决派生的 PE-shaped 只读 overlay，application 侧消费，kernel PE 不跨 tier 读）与 `revision_proposals`（protocol-lineage 冲突 → 保守 WEIGHT_DECAY L3 typed 提案），`protocol_revision_queue` 消费 `revision_proposals` 走 R10 gate + 人审队列；详见 [`docs/specs/apprenticeship-alignment-protocol-layer-draft.md`](./specs/apprenticeship-alignment-protocol-layer-draft.md) |
| `evaluation` | EvaluationModule | EvaluationSnapshot | ACTIVE | 每 turn ~ 每会话 | regime, prediction_error, credit, reflection |
| `evaluation_mid` | MidLayerModule | MidLayerSnapshot | DISABLED（模块）/ SHADOW（`FinalRolloutConfig`） | session / evidence batch | 只读聚合 `evaluation`、`credit`、`prediction_error`、`regime`；`evaluation_expensive` |
| `evaluation_expensive` | ExpensiveLayerModule | ExpensiveLayerSnapshot | DISABLED | submitted judge / evidence batch | 只读消费 `evaluation_mid`、`substrate`；`evaluation_cross_generation` |
| `evaluation_cross_generation` | CrossGenerationAggregatorModule | CrossGenerationAggregateSnapshot | DISABLED | generation / promotion window | bounded generation window、promotion report；不进入 PE / credit 学习源 |
| `decision_workspace` | DecisionWorkspaceModule | DecisionWorkspaceSnapshot | SHADOW | 每 turn | 只读消费 `regime`、`plan_intent`、`goal_value`、`open_loop`、`belief_assumption`、`boundary_policy`；当前无 authoritative consumer |
| `regime` | RegimeModule | RegimeSnapshot | SHADOW | 每 turn | prediction_error, reflection, retrieval_policy |
| `prediction_error` | PredictionErrorModule | PredictionErrorSnapshot | ACTIVE | 每 turn | memory, temporal_abstraction, regime, credit, reflection；另在 final wiring 中被 evaluation enrichment 读取 |
| `credit` | CreditModule | CreditSnapshot | SHADOW | 每 turn ~ 每会话 | reflection; consumes `prediction_error` + `temporal_abstraction.closed_segments` for PE-derived segment credit；P3 dedicated relationship action credit 只从 exact social PE settlement 派生，经 Brain facade 提供给 vertical gate |
| `reflection` | ReflectionModule | ReflectionSnapshot（含 `relationship_update_proposals`） | SHADOW / session-post | 每会话后（异步） | temporal_abstraction、lifeform-service 关系记忆 console；另外通过 owner-side writeback 影响 memory / credit / regime |
| `relationship_continuity` | RelationshipContinuityEvaluationModule | RelationshipContinuitySnapshot | SHADOW | metrics 查询 / 每日 pilot 汇总 | Brain facade、lifeform-service continuity metrics、pilot evidence；只读消费 `prediction_error` CP-12 settlements、`open_loop` / `boundary_consent` / `relationship_state` 快照与 `RelationshipContinuityConsoleOutcome`，不进入 PE / credit / ModificationGate；持久化 key `evaluation/relationship_continuity`，回滚为 DISABLED / 隐藏 endpoint |
| `session_post_slow_loop` | SessionPostSlowLoopModule | SessionPostSlowLoopSnapshot | ACTIVE | context / session boundary | reports / experience_consolidation |
| `retrieval_policy` | RetrievalPolicyModule | RetrievalPolicySnapshot | ACTIVE | 每 turn | domain_knowledge, case_memory, boundary_policy, response_assembly |
| `domain_knowledge` | DomainKnowledgeModule | DomainKnowledgeSnapshot | ACTIVE | 每 turn | boundary_policy, response_assembly, evaluation |
| `case_memory` | CaseMemoryModule | CaseMemorySnapshot | ACTIVE | 每 turn | strategy_playbook, response_assembly, evaluation；`action_grounding` 是 CaseMemory owner 对当前 Memory 语境与 reviewed case intervention steps 的语义近邻解释，绑定 active abstract action；terminal `SCENE_EVENT` 可经 `ExperiencedActionEvidence` + gated session-post writeback 形成带 `case:slow-loop:*:experienced-action:*` lineage 的 lived-action case。schema-free evidence 只有携带 outcome-bound `ActionLearningLineage`，且双轨 transition 已被 optimizer 消费、policy update 已应用、Credit owner record IDs 非空时，才能保存为 `CaseActionAbstractionEvidence` 并进入多经历聚合；旧 checkpoint 或 no-RL 缺证明时保留普通 lived-action audit case但 fail closed。多经历晋升使用 `case:slow-loop:action-abstraction:*` lineage，并在排序前通过只读 structured applicability gate（缺 provider/typed conditions/高置信适用判定时 fail closed）；无匹配/非具体行动轮显式为 `None` |
| `strategy_playbook` | StrategyPlaybookModule | StrategyPlaybookSnapshot | ACTIVE | 每 turn | response_assembly, experience_consolidation |
| `boundary_policy` | BoundaryPolicyModule | BoundaryPolicySnapshot | ACTIVE | 每 turn | response_assembly |
| `response_assembly` | ResponseAssemblyModule | ResponseAssemblySnapshot | ACTIVE | 每 turn | session / response generation；`action_realization` 只绑定同拍 `CaseMemorySnapshot.action_grounding` 与 `TemporalAbstractionSnapshot.active_abstract_action`，不重建案例语义；expression 只渲染 owner-published statement。State KV P5-a 起，assembly 携带的 `control_code / control_scale` 是否真正到达 `runtime.generate(control_parameters, control_scale)` 由 `FinalRolloutConfig.generation_dynamic_residual` 门控（默认 `ACTIVE`=字节级现状；`SHADOW` 计算不注入；`DISABLED` 表达层丢弃，substrate kwargs 与 temporal 未产码的 run 一致）。每 turn 无条件发布 `dynamic_residual=<wiring>[:scale]` rationale tag 自证通道状态；该开关与 `personal_conditioning` 解耦，profile `dynamic-residual-off` 为显式消融臂，详见 [`docs/specs/temporal-abstraction.md`](./specs/temporal-abstraction.md) 与 [`docs/specs/state-kv-identification-evidence.md`](./specs/state-kv-identification-evidence.md) 载体清单 C4 |
| `experience_consolidation` | ExperienceConsolidationModule | ExperienceConsolidationSnapshot | ACTIVE | session-post | experience_fast_prior, reports |
| `experience_fast_prior` | ExperienceFastPriorModule | ExperienceFastPriorSnapshot | SHADOW | 每 turn / session-post carryover | temporal, retrieval_policy, regime |
| `dialogue_external_outcome` | DialogueExternalOutcomeModule | DialogueExternalOutcomeSnapshot；P3 exact join 为 `session_scope + action_turn_index + forecast_id + decision_id + action_id`；P4 `QUALIFIED_USER_REPORT` 还必须携带 typing qualification/runtime/schema lineage | ACTIVE | 每 turn | prediction_error, regime, rupture_state, reflection, preference_about_other forecast settlement |
| `protocol_phase` | ProtocolPhaseModule | ProtocolPhaseSnapshot | SHADOW（模块）/ ACTIVE（`FinalRolloutConfig`） | 每 turn | `active_mixture`；依赖 prediction_error、interlocutor_state、regime、rupture_state、boundary_policy |
| `protocol_registry` | ProtocolRegistryIntrospectionModule | ProtocolRegistrySnapshot | SHADOW（模块）/ ACTIVE（`FinalRolloutConfig`） | 每 turn / registry 变更后 | protocol_reflection、CLI / monitoring；无 upstream 依赖 |
| `protocol_revision_log` | ProtocolRevisionLogModule | ProtocolRevisionLogSnapshot | SHADOW（模块）/ ACTIVE（`FinalRolloutConfig`） | 每 turn / registry 变更后 | CLI / monitoring；无 upstream 依赖 |
| `protocol_reflection` | ProtocolReflectionEngine | ProtocolReflectionSnapshot | SHADOW | background-slow scan | protocol_revision_queue；依赖 prediction_error、active_mixture、domain_knowledge、case_memory、protocol_registry |
| `protocol_revision_queue` | ProtocolRevisionQueueModule | ProtocolRevisionQueueSnapshot | SHADOW | proposal / review event | review / registry apply；依赖 protocol_reflection、apprenticeship_protocol_alignment |

State KV P4-c 不新增 runtime slot。temporal owner 对候选
`ConditioningBankSnapshot` 执行 `topk-semantic.v1` 并发布不可变
`ConditioningRouterDecision`；session 装配链只执行选择：先以
`is_injectable` 硬门，再用共享 semantic embedding 对 owner 发布的
`rendered_statement` 评分，乘以 confidence 与 freshness。router 默认
`SHADOW`，实际 lineage 保持 `static-all.v1`，旁路结果写入
`ConditioningLineage.shadow_router_version / shadow_router_scores`；
`ACTIVE` 才把同一 selected set 应用于 prompt、latent carrier 与 lineage，
并将实际 `router_version / router_scores` 发布为 `topk-semantic.v1`；
`DISABLED` 为无评分的立即回滚。单 judge bank 增益 verdict 是只读 artifact
`state-kv-bank-gain.v3`；正式双裁判聚合为 `state-kv-bank-gain.v4`，要求
distinct judge、同一 observation SHA / substrate / router / claim set，并采用
all-judges-pass。两者都不进入学习链路、不成为第二 evaluation owner。
`freshness=0` 是正式过期语义，必须令 `is_injectable=False`，因此不能进入
router、lineage 或任何 carrier。

Personal Prefix-KV 的 deployment binding 不新增 slot：
`FinalRolloutConfig.personal_conditioning_prefix_artifact_id: str | None` 默认
`None`；非空时只允许 `personal_conditioning_mode="prefix_kv"`，runner 必须在
启动期与 substrate owner 发布的 `personal_conditioning_prefix_id` 精确比较。
profile registry 是该字段的正式配置来源，unknown/mismatched/empty binding 均
fail loudly。即时回滚是 binding=None 且 Personal wiring=SHADOW/residual。

Relationship latent carrier 不新增 slot，也不改变
`relationship_conditioning` owner。runtime 先把 ACTIVE
`ConditioningBankReadout` 适配为 scoped `ConditioningBankSnapshot`，再以不可变
`ConditioningBankLatentCarrier`（`conditioning-bank-latent-carrier.v1`）声明
`carrier="residual"|"prefix_kv"`、载体 version 与不超过 `0.12` 的 scale；
`projector_version` 是 v1 冻结字段名，对 Prefix-KV 承载的是 artifact-derived
carrier version。substrate 是唯一解释该载体并构造 hidden delta / attention
prefix 的 consumer。residual 默认只接受
`bank_type=RELATIONSHIP` 和 `relationship-conditioning-residual.v2`；显式加载
`relationship-conditioning-projector.v1` artifact 后，接受
`relationship-contrastive-residual-v1:<artifact_id>`，并要求 artifact 的
model id、hidden width、hook layers 与 `vector_labels` 分别精确匹配 runtime 和
admitted bank。其他 bank / label / 版本 fail loudly。两种 projector 都把
`[0,1]` readout 以 `0.5` 为 neutral 映射到有符号坐标，将结果 L2 归一后乘
`scale × confidence × freshness`；artifact 可额外声明每层 `(0,1]` gain。
Relationship 与 Personal delta 在 hook 前保持独立，不能共享 layer gain。
全 neutral 或退化为零的状态不报告 applied。

Relationship Prefix-KV 使用 `relationship-prefix-kv.v1` wrapper 把通用
`PrefixKVArtifact` 绑定到 `bank_type=RELATIONSHIP`、owner schema version 和精确
14 维 readout labels；wrapper 与嵌套 generator 都是内容寻址的不可变 artifact。
runtime 加载时校验 model id、层数、KV head、head dimension、labels 和
`norm_cap <= 0.12`，生成时再校验 carrier version 与 scale 必须精确等于已加载
artifact。有效状态先按 `confidence × freshness` 向 neutral `0.5` 收缩，再生成
每层有界 K/V slots；前缀顺序固定为 Character、Personal、Relationship。
缺 artifact、bank/labels/version/scale 漂移或不支持 cache injection 的 backend
均 fail loudly，不回落 residual。

`GenerationResult.conditioning_bank_carriers_applied` 是物理
投递 attestation，`ConditioningLineage.state_encoder_version` 发布同一载体
version。`FinalRolloutConfig.relationship_conditioning_mode` 默认 `text`；
显式 `residual` / `prefix_kv` profile 才启用 latent，回滚为 `text` 或将 owner 置
`SHADOW` / `DISABLED`。

### 6.4 Common Adapter / Character Package artifact contract

该链不新增 kernel slot，也不把角色身份伪装成 `personal_conditioning` 或
`relationship_conditioning`。正式交换分为两个 owner：

| artifact | owner | value_type | dependencies | wiring_level |
|---|---|---|---|---|
| `common_adapter_bundle` | `vz-substrate` offline rare-heavy pipeline | immutable `CommonAdapterBundle` (`common-adapter-bundle.v1`) | frozen base weights digest, `SubstrateRareHeavyCheckpoint`, `PrefixKVArtifact`, `ControlBasisArtifact`, cognition OFFLINE gate record | process startup ACTIVE only after allow gate；omission = rollback |
| `character_package_manifest` | `lifeform-domain-character` bake pipeline | immutable `CharacterPackageManifest` (`character-package-manifest.v1`) | `LifeformTemplate`, optional `CharacterPrefixKVPackage`, optional PEFT LoRA, fidelity report, common adapter version/fingerprint, OFFLINE gate record | registry entry `ACTIVE / SHADOW / DISABLED` |
| `character_session_binding` | `lifeform-service` manifest loader | immutable `CharacterSessionBinding` | admitted `CharacterPackageManifest`, verified template path, Prefix/KV registry key, scoped character LoRA pool key | per manifest `ACTIVE / SHADOW`; `DISABLED` manifests publish no binding |

L1 bundle 的 `compatibility_fingerprint` 必须绑定
`base_model_id + base_model_weights_sha256 + common_adapter_version` 以及三个 nested
carrier id/geometry。HF runtime 加载时重新解析 snapshot、按 weight file 相对路径和
bytes 计算 SHA-256，并核对 rare-heavy hidden width/hook layers/runtime fingerprint、
State-KV K/V geometry 与 control basis；任何不一致 fail loudly。L1 在进程内只读，
禁止 live mutation。

L1/L2 build 目录中的文件按以下契约分层；中间文件是不可变 build/evidence artifact，
不是可由 runtime 单独消费的新 slot：

| 阶段 | artifact | 契约地位 |
|---|---|---|
| control basis diagnostic | `control-basis.json`、observations、verdict | substrate geometry/provenance；不授权 promotion |
| L1 train | `rare-heavy-checkpoint.json`、`state-kv-prefix.json`、State-KV manifest、candidate、gate proposal | candidate nested material；禁止 serving、自批或写 ACTIVE |
| L1 evaluate | held-out report/observations、`common-adapter-gate-record.json` | cognition 只读 evidence 与 allow/deny decision；禁止绕过 publish |
| L1 publish | `common-adapter-bundle.json` | 唯一 L1 runtime artifact；内嵌 rare-heavy、State-KV、control basis 与 gate record |
| L2 bake | `character-prefix.json`、`shadow-manifest.json` | 角色 candidate；只允许 SHADOW |
| L2 evaluate | fidelity report/evidence、gate、`evaluated-manifest.json` | 唯一可申请 L2 ACTIVE 的 manifest/evidence set |

L1 训练阶段的 PEFT LoRA 只用于冻结 base 上的优化和 `B@A` 投影；当前正式 pipeline
不发布 `adapter_model.safetensors`，也不把训练态 LoRA 目录登记成
`common_adapter_bundle`。最终 bundle 是匹配 frozen base 的自包含数值载荷，不是
merged full model；任何 future PEFT checkpoint 输出都必须新增明确 owner、schema、
digest、Gate 与 rollback 契约后才能成为正式交换。

L1 `common-adapter-candidate.v2` 的 content id 还必须绑定训练集 SHA-256/数量、LoRA 与 State-KV
超参数、hook layers 和显式 seed；held-out evaluation 只能以只读 base/candidate/
counterfactual arms 产生 readout，再由 cognition `ModificationGate.OFFLINE` 决策。
L1 publish 必须复核 held-out digest、从 observation 重算 summary/cognition decision，
并要求 gate `evaluation_ref` 绑定 evaluation report SHA-256；四者不能独立替换。
启动时 `COMMON_ADAPTER_BUNDLE_PATH` 独立于任何 L2 manifest：只要显式配置就必须
解析并 `require_active()`，再传给 process runtime；没有角色包不能成为忽略 L1、静默
退回 base-only 的理由。省略该变量才是明确的 frozen-base rollback。

L2 manifest 必须绑定相同的 `base_model_id + common_adapter_version +
compatibility_fingerprint`，且所有 ref 都有 locator + SHA-256 + artifact id。ACTIVE
要求 Prefix/KV、held-out/source-immutable/feedback-free fidelity pass、同一 adapter
fingerprint 与可逆 OFFLINE allow gate。Character LoRA 晋升必须覆盖 prefix-only、
LoRA-only、prefix+LoRA 三臂；当前 evidence schema 只有组合臂布尔 attestation，故
含 `lora_ref` 的 manifest 保持 SHADOW-only。`lifeform-service` 只构造只读
`CharacterPrefixKVRegistry` 并按 session 中的 typed `character_id` 选择 entry，禁止
文本匹配或 consumer 重建角色 owner 状态。
service 对外只发布 `CharacterSessionBinding` 的整包选择结果，consumer 不得分别重算
template locator、Prefix/KV key 或 LoRA key。Character LoRA pool 属 loader 实例且与
DLaaS figure/persona pool 隔离；session synthesizer 只消费绑定值，角色 LoRA 优先于
figure/persona LoRA，禁止同一 forward nested activation。
`CharacterPackageGateRecord.proposal_id` 必须绑定移除 fidelity/gate 后的 exact ungated
manifest content id，`fidelity_report_sha256` 必须绑定同一次 held-out report；证据不得
跨角色、跨 carrier set 或跨 common-adapter 指纹复用。

`GenerationResult.character_id / character_prefix_applied / character_prefix_id /
character_prefix_wiring_level / character_prefix_shadow_id` 是每轮物理投递/SHADOW
加载事实，不等于 behavior fidelity verdict。common adapter 升级会使旧 manifest 的
双指纹失效；`full-rebake` 必须重 bake，`fidelity-only` 也必须重新跑 held-out
evidence 与 gate 后才可重签。L2 回滚是切 SHADOW/DISABLED 或恢复旧 manifest；L1
回滚是省略/恢复旧 bundle，不改 base 与 L3 tenant state。

`CharacterResidualAdapterPackage` 已废弃为只读、SHADOW、rollback-only legacy
carrier；禁止新 bake、禁止 ACTIVE、禁止与统一 manifest 同时装配。
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

- **Task-execution group（coding-lab lane）**：`TASK_VERIFIED` / `TASK_REGRESSED` —— 编程域证据 lane 的 episode 终局，**只能**由确定性环境 oracle（测试套件 / 构建门）经 `ENVIRONMENT` typed source 注入（不允许从 chat text 推断，不允许 LLM 提案）。设计要点是**关系轴零污染**：这两个值存在的全部理由是 vocabulary 里没有"任务轴负向、关系轴中性"的值（`MISSED` 的 task delta 为 0 且关系 delta 为 -0.60）。映射：
  - `_EXTERNAL_OUTCOME_AXIS_BIAS`：`TASK_VERIFIED = (+0.50, 0, 0, +0.40)`，`TASK_REGRESSED = (-0.50, 0, 0, -0.40)`——只触 task / action 轴；
  - `_EXTERNAL_OUTCOME_REGIME_SCORE`：`0.85` / `0.15`；
  - `_EXTERNAL_KIND_TO_STRUCTURAL_OUTCOME`：`CLARIFIED` / `CORRECTED`；
  - `EXTERNAL_OUTCOME_TO_RUPTURE_KIND`：**双双不进入**（world-track 证据无 rupture 可 anchor，损害由 PE bias + regime score 承载，与 `CHURNED` 同理）；
  - `_REPAIR_POSITIVE_KINDS`：**双双不进入**（任务结局不是 repair 事件）；
  - `FeedbackValence`（DLaaS 镜像）：**暂不扩展**——coding-lab 经 `BrainSession.submit_dialogue_outcome` 直接注入，不走 DLaaS feedback envelope；未来有平台集成需要时再扩展镜像。

  详见 [`docs/specs/coding-lab.md`](./specs/coding-lab.md)。

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
| `UserModelSnapshot` | `preferred_support_pacing`, `decision_style`, `overwhelm_pattern_strength`, `durable_goals`, `profile_facts`, `requested_profile_fact_keys`, `profile_context_statement` | dual_track, response_assembly, evaluation；精确用户事实只由 user_model owner 解释，response_assembly 只转发 owner statement |
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
| `preference_about_other` | PreferenceAboutOtherModule | PreferenceAboutOtherSnapshot（P2/P3/P4.2：`action_forecasts`、`action_outcome_evidence`、`forecast_settlements`、`action_outcome_mutation_receipts`，均默认空；forecast 可含 `RelationshipConditionReadout`） | semantic proposals, memory, multi_party_identity, `dialogue_external_outcome`；可选 typed forecast/condition collaborator；可选 user-directed correction/redaction command；forecast lane 显式 SHADOW | ACTIVE | session-medium / background-slow；forecast 为 pre-action per decision，pending/settlement/命名 readout/纠删 tombstone 经 `SocialRecordStore` v4 恢复 | durable style / boundary stability；owner-authored named condition + candidate action → typed outcome distribution → exact settlement；纠删只改变 owner state，不产生 reward | social_prediction_error → prediction_error；P3 action credit 只从 matching social PE 派生 |
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

**P2a action forecast enriched value 注册（2026-08-21）**：

| Existing slot | Unique owner | Enriched value | Dependencies | Wiring / consumers |
|---|---|---|---|---|
| `preference_about_other` | `PreferenceAboutOtherModule` | `PreferenceAboutOtherSnapshot.action_forecasts`；元素为 frozen `PreferenceActionForecast`，内含同一 typed outcome vocabulary 上的至少两个 `SocialActionCandidatePrediction` | P2b producer 只接收 typed `PreferenceActionForecastRequest`、同一 owner 的 ACTIVE records 与非 owning forecast proposal runtime；不得读 evaluator label | 字段默认空；P2-development 只能显式 `WiringLevel.SHADOW` 发布，expression / planner / steering / PE / credit 均不得消费 |

P2a 没有新增 slot、owner 或持久化写者。`PreferenceActionForecast` 是 owner 在行动前发布的
派生 readout，不是可由 reflection 直接写入的 durable state；其 `source_record_ids` 必须在同一
`PreferenceAboutOtherSnapshot.records` 中存在、属于同一 interlocutor 且不来自未来 turn。
contract 不含 observed outcome、expected action、reward 或 credit。P2b 已提供可选 producer：
collaborator 只能返回 `PreferenceActionForecastProposal`，正式 forecast id / decision / scope /
turn / record lineage 仍由 `PreferenceAboutOtherModule` 绑定；只要配置不是 SHADOW 就 fail loudly。
默认没有 runtime/request，因此所有现有正式 runtime snapshot 仍保持空 tuple，既有 ACTIVE
records 路径与用户可见行为不变。回滚只需移除两个可选注入；已序列化旧构造保持兼容。

**P2c / P2d / P3 / P4.2 owner persistence、readout、settlement、纠删与 advisory enriched value 注册（2026-08-22）**：

| Existing slot | Unique owner | Enriched value / input | Dependencies | Wiring / rollback |
|---|---|---|---|---|
| `preference_about_other` | `PreferenceAboutOtherModule` | `PreferenceActionOutcomeEvidence`、pending `PreferenceActionForecast`（可含 `RelationshipConditionReadout`）、`PreferenceActionForecastSettlement`、`PreferenceActionOutcomeMutationReceipt`；由 `SocialRecordStore` persistence v4 保存 | owner records；P2d condition collaborator 只读 current observation + owner histories；结算时只读 `dialogue_external_outcome` exact join；纠删只接收带 expected evidence hash 的 typed user command | forecast lane SHADOW；v1-v3 persistence 可读，旧 forecast condition readout/旧 mutation receipt 为空；export 写 v4；撤回 consumer 不删除 tombstone |
| `social_prediction_error` | `SocialPredictionErrorModule` + preference owner settlement readout | `social-pe:<settlement_id>`，prediction/outcome/magnitude 必须与 settlement 一致 | exact owner settlement | 只进入既有 PE→credit 方向，不读取 evaluation |
| `self_temporal` | `TrackTemporalModule(track=SELF)` | `TemporalActionAdvisoryProposal`、`TemporalActionAdvisoryStatus` | owner forecast + vertical gate decision，经 runtime facade 单次 staging | `FinalRolloutConfig.relationship_action_advisory=SHADOW` 默认；DISABLED 丢弃；ACTIVE 要求 artifact 明确授权 |
| `dialogue_external_outcome` | `DialogueExternalOutcomeModule` | `QUALIFIED_USER_REPORT` source 的 qualification id/hash、typing runtime/schema；relationship exact join 五元组 | service typing qualification + 已暴露 action audit | 无 qualification 时不构造该 source；移除 path 回到 collection-only |

P3 exact settlement 要求 `session_scope / action_turn_index / forecast_id / decision_id /
action_id` 全部匹配 owner pending forecast。每条 forecast 至多结算一次；未知 forecast、不同
evidence 的重复结算或 surface drift 均 fail loudly。settlement 的 signed utility PE 使用冻结
四类 outcome 的 utility（`helped/felt_heard=+1`，`missed/over_directive=-1`），credit 必须再
匹配 owner-authored social PE，值为 `signed_utility_prediction_error × evidence_confidence`，
`level=relationship_action_prediction_error`、`track=SELF`。human anchor、evaluation 与 judge
都不进入该公式。

P2d 不新增 slot 或第二 writer。`RelationshipConditionReaderArtifact` 绑定 embedding model id、
weights SHA-256、cosine、prototype 文本与 temperature；collaborator 只能返回
`RelationshipConditionReadout` proposal。`PreferenceAboutOtherModule` 必须验证
`source_observation_sha256` 与当前 request 完全一致后，才把 readout 放入正式 forecast。
readout 不含 expected action、observed outcome、evaluation、PE、credit 或 reward；consumer
不得解析 forecast evidence 字符串来重建 condition。已见 v3 上 `4/12 → 12/12` 与 6/6 mirror
pair 只作 backend 根因诊断，不能注册为 formal Readable evidence。独立 v2
`FrozenLinearRelationshipConditionReaderArtifact` 保留同一 slot/owner/readout contract，并额外
内容寻址绑定 embedding model/revision/weights/runtime/width、严格有序 labels、condition-only
corpus artifact/raw digest、group-split artifact、deterministic centroid solver/version、逐类
example count/id digest 与完整 canonical float-hex 参数。offline builder 只读 labelled embedding
rows 和预声明 pins；online runtime 只读 current public text 与 identity-pinned frozen embedder，
禁止拟合。fresh child 只能调用 artifact owner 的 exact-key `from_payload` 或 canonical
`from_json`，由 owner 重算 artifact id 并拒绝 duplicate key、非 canonical UTF-8 bytes、额外/
缺失字段和参数 shape drift；consumer 不得重建 loader schema。该 v2 尚未 qualification、尚未
接入 campaign，只是 reader mechanism，不构成 Readable 结论。
`FrozenLinearRelationshipPreferenceForecastRuntime` 是不新增 slot/owner 的薄 consumer：它只接受上述
exact frozen reader runtime，以 named readout 的 label/confidence 定义 condition-equivalence similarity，
再按既有 `BoundedRelationshipPreferenceForecastRuntime` 的固定默认 prior/evidence weight 从 owner histories
生成 proposal，并把同一
`RelationshipConditionReadout` 原样交给 `PreferenceAboutOtherModule` 校验/发布。adapter 没有 fit、
training-label/label-truth、evaluator、judge、sealed/future outcome truth 或 gate 输入；其存在不等于 reader 已
qualification，也不授权
source-v4 campaign。

P4.2 correction/redaction 不新增 slot 或第二写者。调用方提交 frozen
`PreferenceActionOutcomeMutation`，必须带 `mutation_id / target_evidence_id /
expected_evidence_sha256 / requested_turn / evidence_refs`；`CORRECT` 另带 replacement typed
outcome，`REDACT` 禁止携带 replacement。owner 使用 optimistic hash 拒绝并发覆盖，保持
interlocutor、已暴露 action 与 source-turn lineage，不允许 console 命令改写真实 action。
纠正会同步更新配对 owner record 并失效引用它的 pending forecast；删除会同时移除 outcome、
owner record、pending ToM prediction 与引用它的 pending forecast。结果只发布 content-safe
`PreferenceActionOutcomeMutationReceipt`：包含 command/before/after hash、失效 forecast id 与
opaque evidence refs，不保留被删 observation/reaction。`REDACT` receipt 是持久 tombstone，
hydration 后仍禁止旧 evidence id 复活；相同 mutation id + command hash 可幂等重试，不同命令
复用 id fail loudly。它是用户纠错/隐私命令，不是 PE、credit、evaluation 或学习信号。
本 slice 只拥有 `preference_about_other` state：已经结算的 forecast、既有 PE/credit/gate
checkpoint 与 lifeform operational evidence 不会被它猜测性反向改写，因为当前没有从 owner
outcome record 到这些 artifact 的可逆 exact lineage。产品级全域撤回必须另开 owner-by-owner
收敛包；不得用 turn/action 近似 join 冒充删除完成。

`TemporalActionAdvisoryProposal` 只携带 typed action/lineage/rationale，不含表达文本。SHADOW
时 `active_abstract_action` 保持 native 值并发布 `SHADOW_RECORDED`；只有
`active_authorized=true` 且非 evaluator artifact 的 ACTIVE advisory 才能发布 `APPLIED`。
P3/P4 vertical artifact 固定未授权，因此当前用户可见表达不变。P4 service 另以
`baseline_noop_exposed | shadow_counterfactual` 记录 causal exposure；后者禁止写 runtime
outcome 或 training candidate，避免把未执行建议的结果错误归因给该 action。

P4 product artifacts 是 lifeform-side create-only evidence，不新增 kernel slot：action audit、
outcome receipt 与 opt-in offline training candidate 使用不同 schema 和物理 root，只保存 typed /
hashed metadata，对外返回 content-addressed opaque ref。真人自由文本的 structured LLM typing
必须由 content-hashed qualification artifact 绑定三名独立 rater、隐藏标签、多数一致率
`>=0.80`、预注册 human-anchor 阈值、validation-only / no-learning、无关键词/正则与 unknown
支持；没有布尔 PASS 开关。完整 contract 见
[`docs/specs/relationship-intelligence-closed-alpha.md`](./specs/relationship-intelligence-closed-alpha.md)。

### 6.1 Lifeform-side Slots（不进入 kernel slot 注册表）

下表 slot 由 lifeform 层 wheel 拥有；它们**不**进入 kernel propagate 顺序，也**不**作为 kernel owner 单写者校验目标。它们是 lifeform 与 host / service 之间的契约面，由 `lifeform-*` 包发布，供 `lifeform-expression` / `lifeform-service` / 操作员 dashboard 消费。

| Slot Name | Owner 模块 | Wheel | Value 类型 | 默认接线 | 发布频率 | 消费者 |
|-----------|-----------|-------|-----------|----------|----------|--------|
| `vitals` | VitalsModule | `lifeform-core` | VitalsSnapshot | per-vertical | SYSTEM tick + per-turn | lifeform-expression, followup_manager, prompt_planner |
| `affordance` | AffordanceModule | `lifeform-affordance`（registry、renderers 与 invoker 已落地） | AffordanceSnapshot | ACTIVE（lifeform host 接线；不进入 kernel propagate） | per-call / per-turn | prompt_planner, response_synthesizer, AffordanceInvoker |
| `thinking_loop` | ThinkingScheduler | `lifeform-thinking`（Phase 1 slice 1/2a/2b 已落地） | ThinkingLoopSnapshot | SHADOW（默认 advisory/report-only）→ ACTIVE（显式 opt-in） | scene 内异步 | temporal advisory ingress、family_report metrics、debug dashboard |
| `relationship_memory_console` | RelationshipMemoryActionLedger | `lifeform-service` | RelationshipMemoryActionRecord | ACTIVE（closed-alpha MVP） | 用户 console action | relationship-memory API、P5 continuity metrics；只记录 action/idempotency，不复制 kernel relationship state，semantic mutation 经 Brain facade 排入 owner event queue |

**lifeform-side slot 不变量**：

- 不可被任何 `vz-*` wheel 反向 import（CI 由 `tests/contracts/test_import_boundaries.py` 强制）
- 不可作为 kernel owner 间 propagate 的输入；只能被 lifeform 层（含 expression / service）消费
- 副作用如果要进入 kernel，**必须**走已有公共入口（`BrainSession.submit_*` / `LifeformSession.run_turn`），不可旁路

### 6.1B Platform-side Slots（DLaaS 控制平面，不进入 kernel slot 注册表）

下表 slot 由新增的 `dlaas-platform-*` wheel 拥有，承担多租户治理与 ops 状态。它们**不**进入 kernel propagate 顺序，**不**被任何 `vz-*` wheel 读取。详见 [`docs/specs/dlaas-platform.md`](./specs/dlaas-platform.md)。

| Slot Name | Owner 模块 | Wheel | Value 类型 | 默认接线 | 发布频率 | 消费者 |
|-----------|-----------|-------|-----------|----------|----------|--------|
| `tenant_state` | TenantRegistry | `dlaas-platform-registry`（SQLite CRUD 已实现） | TenantState | ACTIVE（platform control plane） | CRUD 时 | `dlaas-platform-api` auth 中间件、`dlaas-platform-launcher` quota 检查 |
| `contract_state` | ContractRegistry | `dlaas-platform-registry`（SQLite CRUD 已实现） | ContractState | ACTIVE（platform control plane） | CRUD 时 / lifecycle 切换 | `dlaas-platform-launcher`（adopt / awake）、`dlaas-platform-api`（runtime 路由） |
| `instance_status` | InstanceManager | `dlaas-platform-launcher`（instance map 已实现） | InstanceStatus | ACTIVE（platform control plane） | adopt / awake / sleep / evict | `dlaas-platform-api`、`dlaas-platform-ops` |
| `handoff_ticket_state` | HandoffQueue | `dlaas-platform-ops`（queue / ticket / SSE 已实现） | HandoffTicketState | ACTIVE（platform control plane） | rupture_state 快照触发 / 操作员手动 | `dlaas-platform-api`、admin SSE stream |

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
| `reflection` | `relationship_update_proposals: tuple[RelationshipUpdateProposal, ...]` | reflection owner 从 consolidation、typed tension / lesson 与 PE failure readout 发布可审阅关系更新提案；consumer 不得从原始文本重建；P1 默认 SHADOW、必须用户确认 |
| `memory` | `MemoryStoreCheckpoint.entry_attributes: tuple[MemoryAttributeReadout, ...]` | memory owner 在 checkpoint/rollback 中原子保存 artifact 的 PE/substrate attribute 投影；旧 checkpoint 默认空 tuple，避免 Console 删除/改写回滚后留下或丢失耦合投影 |

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
