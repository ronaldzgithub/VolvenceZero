# Specs 分层知识入口总索引

> 本文件是代码探索的**默认起点**。改代码前先查本索引定位目标能力域，再读对应 spec。

---

## 使用方法

1. **先查本索引** → 定位任务属于哪个能力域
2. **读目标 spec** → 拿职责边界、contracts、关键不变量
3. **只有涉及跨域边界/接口/架构意图变更时** → 再读 `docs/next_gen_emogpt.md` 对应需求和算法基础
4. **只有要改实现细节时** → 再读具体代码文件

> 文档与代码不一致时以代码为准，并同步更新对应 spec/文档。

---

## 能力域索引

### 1. Prediction Error 主链

**对应需求**：R-PE（prediction error 为原始学习信号）

| Spec | 内容 |
|------|------|
| [prediction-error-loop.md](./prediction-error-loop.md) | 显式 prediction chain、prediction_error owner、bootstrap / carryover 语义、下游消费边界 |

**核心不变量**：
- prediction error / LSS 是原始学习信号
- evaluation 是 readout / gate 层，不是学习源头
- credit 是 prediction error 的聚合层，不是学习源头
- prediction chain 必须作为正式 runtime object 发布

---

### 2. 多时间尺度学习框架

**对应需求**：R1（多时间尺度学习）、R2（稳定基底 + 自适应控制器）、R13（压缩-强化交替）

| Spec | 内容 |
|------|------|
| [multi-timescale-learning.md](./multi-timescale-learning.md) | 4 个时间尺度的学习循环设计、冻结基底/自适应控制器分层、SSL-RL 交替循环 |

**核心不变量**：
- 快速适应不需要重写整个模型
- 慢速沉淀不阻塞实时交互循环
- 强化作用于压缩和结构化的内部基底，而非原始行为

---

### 3. 时间抽象与内部控制

**对应需求**：R3（时间抽象）、R4（内部控制在 token 空间之上）

| Spec | 内容 |
|------|------|
| [temporal-abstraction.md](./temporal-abstraction.md) | Metacontroller 架构、切换单元、控制器代码空间、Internal RL |

**核心不变量**：
- 实时行为可通过内部状态转换引导，而非仅通过表面文本损失
- 抽象动作可组合、可训练、无需详尽手动标签
- 冻结基础模型是发现时间抽象的前提

---

### 4. 连续记忆系统

**对应需求**：R5（记忆连续谱）、R6（反思与沉淀）

| Spec | 内容 |
|------|------|
| [continuum-memory.md](./continuum-memory.md) | 多层记忆设计（瞬态/情景/持久/派生）、更新频率、提升/衰减规则、异步慢反思路径 |
| [cms-atlas-titans-uplift.md](./cms-atlas-titans-uplift.md) | Tier 2 改动：CMS 内部接入 ATLAS past-aware joint optimization 与 Titans PE-driven 写入门控；SHADOW/ACTIVE 协议、回滚契约、向后兼容法则 |

**核心不变量**：
- 记忆是连续谱，不是二元短期/长期分割
- 记忆写入通过正式 owner 和 API，不可绕过
- 慢反思产出两类产物：记忆沉淀 + 策略沉淀
- ATLAS / Titans uplift 不创造与 CMS 并列的第二个 memory owner；更新规则不被外置 LLM curator 替代

---

### 5. 双轨学习

**对应需求**：R7（自我/关系学习与任务学习分离）

| Spec | 内容 |
|------|------|
| [dual-track-learning.md](./dual-track-learning.md) | World/Self 轨道设计、按轨道隔离的记忆/信用/控制器/评估 |

**核心不变量**：
- 两轨可共享基础设施，但在记忆写入、信用分配、控制器更新、评估指标上保持语义区分
- 关系连续性不是问题解决的副作用

---

### 6. 契约式运行时

**对应需求**：R8（快照优先、契约优先）、R11（可学习的内部状态表示）、R15（迁移可解释性和可回滚）

| Spec | 内容 |
|------|------|
| [contract-runtime.md](./contract-runtime.md) | 模块间通信总线、快照契约、模块基类与生命周期、编排调度、内部状态发布 |

**核心不变量**：
- 每个运行时区域有唯一主要 owner
- 跨模块交换通过公共快照，消费者不重建生产者内部状态
- 快照不可变
- 内部状态可命名、可发布、可检查

---

### 6A. 语义状态一等 Owner

**对应需求**：R1（多时间尺度学习）、R8（快照优先）、R11（内部状态可发布）、R15（可回滚演进）

| Spec | 内容 |
|------|------|
| [semantic-state-owners.md](./semantic-state-owners.md) | Plan/Intent、Commitment、Open Loop、User Model、Execution Result、Belief/Assumption、Relationship State、Goal/Value、Boundary/Consent 九个语义 owner |

**核心不变量**：
- 语义细节由对应 owner 保存并发布，不存入 ETA / NL 本体
- ETA 消费 compact semantic advisories，不成为语义状态第二 owner
- 语义更新通过 typed proposal path，默认路径不使用关键词规则伪装理解

---

### 6B. Environment Interface

**对应需求**：R-PE、R1、R3/R4、R8、R10、R11、R15、R16-R20

| Spec | 内容 |
|------|------|
| [environment-interface.md](./environment-interface.md) | 生命体与环境之间的 Observe / Perceive / Act / Assimilate 总边界协议；Environment Event 语义；行动 outcome 到 PE 的回流；与 social cognition 的正交依赖 |

**核心不变量**：
- 环境不是内核 owner；`lifeform-*` 负责适配环境，`vz-*` 只消费 canonical event / frozen snapshot
- 感知层只产生 typed proposal / owner input，不拥有最终社会状态或记忆状态
- 行动必须能形成 pre-action prediction，并通过 outcome 回流 `prediction_error`
- Social Cognition 消费 Environment Event 的 conversational frame，不从 raw text / renderer / prompt 重建社会事实

---

### 6C. Emergent Action Abstraction（Phase 1 clean，contract + tests landed）

**对应需求**：R-PE、R1、R3、R4、R8、R9、R10、R11、R15

| Spec | 内容 |
|------|------|
| [emergent-action-abstraction.md](./emergent-action-abstraction.md) | ETA/NL-clean 动作反馈抽象：最小 EnvironmentOutcome 观察字段、`beta_t` segment closure、PE action context、PE-derived segment credit、snapshot replay export |

**核心不变量**：
- 不新增 `action_outcome_trace` runtime slot、ledger owner 或 action/outcome encoder owner
- delayed outcome 边界来自 `temporal_abstraction.closed_segments`（`beta_t` segment closure）
- `EnvironmentOutcome` 只包含外部可观察字段，不承载 trust / common-ground / commitment semantic delta
- PE owner 承载 action context；credit 只从 PE 派生 segment/action records
- replay 是既有 snapshot 序列的 out-of-turn export，不是 runtime schema

---

### 7. 信用分配与自修改

**对应需求**：R-PE（prediction error 为原始学习信号）、R9（层级信用分配）、R10（有门控的分层自修改）

| Spec | 内容 |
|------|------|
| [credit-and-self-modification.md](./credit-and-self-modification.md) | prediction error 派生的层级信用分配、语义化奖励记录、门控自修改规则 |

**核心不变量**：
- prediction error / LSS 是所有信用的源头，信用是其聚合层
- 稀疏奖励是常态，不是边缘情况
- 自修改有门控：在线/后台/离线/人工审核分层
- 实时运行期间不可无限制突变基础模型

---

### 8. 评估体系

**对应需求**：R12（评估覆盖"存在"而非仅任务成功）

| Spec | 内容 |
|------|------|
| [evaluation.md](./evaluation.md) | 6 族评估指标（任务能力、交互质量、关系连续性、学习质量、抽象质量、安全与有界性） |

**核心不变量**：
- 评估不仅衡量有用性，还衡量连续性、稳定性、信任和长期适应
- 评估是 prediction error 的 readout / gate，而不是学习源头

---

### 9. 证据计划

**对应需求**：R12（评估覆盖"存在"而非仅任务成功）、R15（迁移可解释性和可回滚）

| Spec | 内容 |
|------|------|
| [evidence_program.md](./evidence_program.md) | claim-to-evidence 映射、blind review、pairwise effect、evidence bundle |

**核心不变量**：
- 对外主张必须映射到 required gates、artifact 与 verdict
- 盲评外发包不得泄漏 profile 条件
- 证据结论必须可回溯到 manifest / provenance / 原始 artifact

---

### 10. 认知 Regime

**对应需求**：R14（社交与认知 Regime 的持久身份）

| Spec | 内容 |
|------|------|
| [cognitive-regime.md](./cognitive-regime.md) | Regime 运行时表示、记忆化、高层控制选择、延迟结果训练 |

**核心不变量**：
- Regime 不是 prompt 标签，而是可记忆、可选择、可训练的持久身份

---

### 11. Domain Experience Layer

**对应需求**：R5（连续记忆）、R6（反思与沉淀）、R7（双轨学习）、R8（契约优先）、R12（评估）、R15（可回滚演进）

| Spec | 内容 |
|------|------|
| [domain-experience-layer.md](./domain-experience-layer.md) | 可复用垂直经验包 schema、编译到现有 application owners 的契约、加载与评测边界 |

**核心不变量**：
- Domain Experience Package 不是新的运行时 owner
- 垂直经验编译到现有 `domain_knowledge` / `case_memory` / `strategy_playbook` / `boundary_policy` / rare-heavy application state
- 包内容是冷启动 scaffold 和评测锚点，不等同于真实长期经验成熟
- 不通过人口学关键词硬编码行为

---

### 11A. Lifeform Vitals (Always-On Drive Layer)

**对应需求**：R-PE（prediction error 为原始学习信号）、R1（多时间尺度）、R8（契约优先）、R11（内部状态可发布）、R14（regime 持久身份）

| Spec | 内容 |
|------|------|
| [lifeform-vitals.md](./lifeform-vitals.md) | DriveSpec / VitalsBootstrap / VitalsSnapshot；介于 turn-driven assistant 与 always-on organism 的边界层；vertical 通过 drive 集合编码"这只生命体在乎什么" |

**核心不变量**：
- 漂出 homeostatic band 的 drive deviation 即慢尺度 prediction error；in-band 时贡献为 0（homeostasis 静默）
- 衰减只发生在 SYSTEM tick；ENERGY/CONTEXT tick 仅推进 tick_index
- VitalsModule 是 drive level 的唯一 owner；消费者只读 VitalsSnapshot
- 跨 proactive_pe_threshold 触发的 followup 由 owner 内部 cooldown 控制，永不洪泛

---

### 12. Core Package Boundary

**对应需求**：R2（稳定基底 + 自适应控制器）、R8（契约优先）、R11（内部状态可发布）、R15（可回滚演进）

| Spec | 内容 |
|------|------|
| [core-package-boundary.md](./core-package-boundary.md) | Python core package 边界、稳定 Brain API、optional HF/model runtime、服务与产品数据外置 |

**核心不变量**：
- `volvence-zero` core package 不包含模型权重、产品数据、用户记忆或部署 secrets
- `volvence_zero.brain` 是稳定 package-facing API
- Qwen / Hugging Face runtime 必须显式配置并通过 optional extra 进入
- 第一阶段只做 local editable package，不做 PyPI upload / deploy / public service

---

### 13. 中频思考循环（Phase 1 slice 1 + 2a + 2b **已落地**）

**对应需求**：R1（多时间尺度）、R6（反思与沉淀）、R8（快照优先）、R11（内部状态可发布）、R15（可回滚演进）

| Spec | 内容 |
|------|------|
| [thinking-loop.md](./thinking-loop.md) | 中频 ThinkingScheduler、mid-reflection / active exploration / provisional case 三类 read-only worker、fingerprint guard 与 case_memory lifecycle |

**核心不变量**：
- worker 永远只读 owner 快照，artifact apply 由 owner 自己决定
- fingerprint mismatch 一律 STALE，不允许"再 apply 一次试试"
- ProvisionalLesson 不创新 owner，进 `case_memory` 加 lifecycle 字段
- mid-reflection 的 self/world 双 lane 复用既有双 owner

**已落地实现**：
- 不可变契约：`volvence_zero.thinking`（`ThinkingTask / ThinkingArtifact / ThinkingDepth / ThinkingTaskStatus / ThinkingPurpose` + `TERMINAL` / `APPLIABLE` 常量）—— 在 `vz-contracts`，跨 wheel 共享
- `case_memory` lifecycle 字段 + `ApplicationCaseMemoryStore.reconcile_provisional_cases` —— 在 `vz-application`
- scene-end 自动 reconcile wiring —— 通过 `LifeformSession.end_scene` → `BrainSession.reconcile_case_memory_provisional`
- 异步 scheduler + mid-reflection worker + fingerprint guard —— 新 wheel `packages/lifeform-thinking/`

来源：`docs/implementation/13_emogpt_prd_alignment_upgrade.md` Gap 4。

---

### 14. AAC Commitment Lifecycle（v1 typed lifecycle landed）

**对应需求**：R-PE（PE 主链）、R8（快照优先）、R11（内部状态可发布）、R14（regime 持久身份）

| Spec | 内容 |
|------|------|
| [aac-commitment-lifecycle.md](./aac-commitment-lifecycle.md) | 权威落地 spec：`SemanticProposalOperation` → advocacy / alignment lifecycle 真值表；follow-up surfacing 已接入 `lifeform-core` |
| [aac-lifecycle.md](./aac-lifecycle.md) | 设计背景与 PRD Gap 7 映射；若与 v1 落地 spec 冲突，以 `aac-commitment-lifecycle.md` 为准 |

**核心不变量**：
- 不新建 owner；commitment 仍是单写者
- 状态迁移**只**通过 `SemanticProposal` typed path，生命周期由 `SemanticProposalOperation` 派生
- alignment proposal source 可来自 LLM structured output / embedding similarity，但 owner 不从文本关键词判定状态
- alignment AGREE→REJECT 是高 PE 信号源，进入 prediction_error 主链

来源：`docs/implementation/13_emogpt_prd_alignment_upgrade.md` Gap 7。

---

### 15. Affordance 体系（v1 已落地）

**对应需求**：R3（时间抽象）、R4（内部控制）、R8（契约优先）、R10（有界自修改）、R11（内部状态可发布）、R15（可回滚演进）

| Spec | 内容 |
|------|------|
| [affordance.md](./affordance.md) | 4 Kind 描述符（Tool/Action/Organ/Shell）、`AffordanceRegistry`、4 渲染器、metacontroller learned 选择 |

**核心不变量**：
- 描述符 schema 在 `vz-contracts`；注册表与 invoker 在 `lifeform-affordance` wheel
- 选择由 metacontroller 在 z_t 空间学，**禁止**硬编码路由
- safety_model + ModificationGate 共同守门；调用结果通过 `BrainSession.submit_tool_result` 回流
- 跨 vertical affordance 隔离

**已落地实现**：
- `volvence_zero.affordance` —— 描述符 schema、4 Kind 枚举、selection-hint 不变量（vz-contracts）
- `lifeform_affordance.registry / scorer / invoker / snapshot` —— 注册表、metacontroller-aware scorer、五阶段 invoker（lookup → safety gate → rate limit → param schema → backend → kernel result wiring）、snapshot 发布
- `lifeform_affordance.renderers.{catalog_json, compact_list, markdown, openai_tools}` —— 4 个渲染器
- 测试：`tests/test_affordance_{registry,scorer,invoker,renderers_and_snapshot}.py`

来源：`docs/implementation/13_emogpt_prd_alignment_upgrade.md` Gap 1。

---

### 16. Runtime Ingestion + Apprenticeship Trigger（Phase 0 design freeze, 实施待 Phase 2）

**对应需求**：R6（反思与沉淀）、R8（契约优先）、R10（有界自修改）、R15（可回滚演进）

| Spec | 内容 |
|------|------|
| [runtime-ingestion.md](./runtime-ingestion.md) | `lifeform-ingestion` adapter wheel + `trigger_kind` 标签 + apprenticeship vitals override；book / web / task_result 三类 source |

**核心不变量**：
- 所有 ingestion 内容只通过 `LifeformSession.run_turn(..., trigger_kind="ingestion")` 进入 kernel
- ingestion adapter **不**直接戳任何 owner store
- durable 化只走 R6 session-post slow loop，无特殊学习路径
- apprenticeship override 是有界（仅当前 turn）；leak-free 由 unit test 强制

来源：`docs/implementation/13_emogpt_prd_alignment_upgrade.md` Gap 2 + Gap 3。

---

### 17. Character Soul Bootstrap

**对应需求**：R5（连续记忆）、R6（反思与沉淀）、R7（双轨学习）、R8（快照优先）、R11（内部状态可发布）、R14（regime 持久身份）、R15（可回滚演进）

| Spec | 内容 |
|------|------|
| [character-soul-bootstrap.md](./character-soul-bootstrap.md) | 小说人物 → reviewed character profile → DomainExperiencePackage / VitalsBootstrap / IngestionEnvelope 的同仓异包 vertical |

**核心不变量**：
- character bootstrap 是 lifeform vertical，不是新 kernel owner
- 角色画像必须是 reviewed structured artifact，不用关键词匹配从小说文本直接驱动行为
- 原文小说只通过 canonical ingestion path 进入，durable 化仍由 R6 slow loop 处理

---

### 17A. Rupture and Repair Loop（v0 SHADOW, M0 contract landed）

**对应需求**：R-PE、R7、R8、R11、R15

| Spec | 内容 |
|------|------|
| [rupture-and-repair.md](./rupture-and-repair.md) | `rupture_state` owner + `dialogue_external_outcome` slot；closed `RuptureKind` / `DialogueExternalOutcomeKind` vocabularies；external-confirmed-only rupture rule；rupture-repair memory tag schema and v0 → post-v0 migration path |

**核心不变量**：

- `rupture_kind` 是 evidence-bucket label，不是情绪分类
- 只有至少一个非 PE 的 typed source 触发时才能写出 `rupture_kind`
- 添加新 `RuptureKind` 必须先添加新 typed signal source（不用关键词 / LLM 分类）
- `submit_dialogue_outcome` 不直接写 memory/regime/PE 内部状态；只发布 `dialogue_external_outcome` snapshot
- rupture-repair durable 写入仅通过 `ReflectionEngine.apply(...)`
- LLM proposal 默认禁用；启用后也只能低置信度 proposal

---

### 17B. Expression Layer (rationale_tags + reflection-hint SSOT)

**对应需求**：R4（内部控制在 token 之上）、R8（契约优先）、R11（内部状态可发布）

| Spec | 内容 |
|------|------|
| [expression-layer.md](./expression-layer.md) | `AgentResponse.rationale_tags` typed audit surface；`ReflectionLessonId` / `ReflectionTensionId` enum；UX hint 文案归 `lifeform-expression`；render-section variant tag |

**核心不变量**：

- 下游 gate / 评估 / reflection 必须读 typed `rationale_tags`，禁止 substring 匹配 `rationale` / `text`
- reflection lesson / tension id 是 enum，新增 id 必须先加 enum 成员
- per-id UX 文案在 lifeform-expression 单一来源，kernel 不再持有 hint_map

来源：W1 (P0) of `docs/known-debts.md` SSOT cleanup 2026-05-06。

---

### 17C. Interlocutor State (12-axis SHADOW owner)

**对应需求**：R8（契约优先 / 快照优先）、R11（内部状态可发布）、R15（可回滚演进）

| Spec | 内容 |
|------|------|
| [interlocutor-state.md](./interlocutor-state.md) | `InterlocutorStateModule` SHADOW owner；12-axis state；10 个 typed zone bool；`InterlocutorThresholds` 单一阈值源 |

**核心不变量**：

- `InterlocutorStateModule` 是 12-axis readout 的唯一所有者；planner / synthesizer / lifeform-core 都读 snapshot
- 阈值常量住在 `InterlocutorThresholds` 一处；消费者读 zone bool，禁止重新算阈值
- `InterlocutorState.__post_init__` 总是用 `compute_zones()` 重写 zone bool；构造时传错的 bool 会被覆盖
- `readout_confidence < 0.30` 时所有 zone False（冷启动安全）

来源：W2 (P1+B) of `docs/known-debts.md` SSOT cleanup 2026-05-06。同时关闭 known-debt #1（interlocutor duck-typed multi-owner reconstruction）。

---

### 17D. Figure Corpus Cleaning Pipeline (L1)

**对应需求**：R8（契约优先 / 快照优先）、R15（迁移可解释性 + 可回滚）

| Spec | 内容 |
|------|------|
| [figure-corpus-cleaning.md](./figure-corpus-cleaning.md) | `bytes -> RawDocument -> CleanedDocument` 全链；4 个 parser（CPAE PDF / Wikisource HTML / Project Gutenberg / Internet Archive OCR JSON）+ 6 个 cleaner op + content-addressable store + cleaner 版本化 + re-clean CLI + 桥接到既有 `*Payload` |

**核心不变量**：

- `cleaning/` 子包**禁止** import `Figure*Source` typed record（必须经 `cleaning/bridging.py` 二段式）
- `cleaning/` 子包**禁止** import 任何 HTTP 客户端（cleaning 与 fetcher 解耦）
- 每个 `CleaningOpRecord` 满足 `chars_after <= chars_before`（monotonically non-expanding）
- raw bytes content-addressable by sha256；cleaner 版本目录（`v{N}/`）多版本共存，永不覆盖旧版

来源：`docs/known-debts.md` debt #28 L1 packet（2026-05-10）。本 spec 只覆盖 L1；L0 crawler frontier 仍是 follow-up；L2 verification 见 17E。

---

### 17E. Figure Corpus Verification + Audit (L2)

**对应需求**：R8（snapshot / contract first）、R12（evaluation 单向性）、R15（迁移可解释 + 可回滚）

| Spec | 内容 |
|------|------|
| [figure-corpus-verification.md](./figure-corpus-verification.md) | 7 `CheckKind` 关闭枚举（3 first batch impl + 4 deferred 至 #26）+ append-only `VerificationLedger`（`data/verification/{byte_sha256}/checks.jsonl`）+ `build_figure_artifact_bundle` 的 `require_verification_pass` gate + L1 → L2 的 `cleaned_to_source_provenance` 接线（修法 5）+ 抽样 / 人审 / 列表 CLI |

**核心不变量**：

- `verification/` 子包**禁止** import `Figure*Source` typed records / HTTP 客户端 / 任何 `volvence_zero.{cognition,temporal,memory,substrate,application,runtime}.*` 内核模块（contract test AST 三类静态守门）
- `VerificationCheck` 不可变；`reviewer_id` 必须形如 `auto:<verifier_id>:<int>` 或 `human:<reviewer-id>`
- Anchor key = `source_byte_sha256` = `SourceProvenance.byte_sha256` = L1 `RawDocument.raw_sha256`（content-addressable 三段贯通）
- Ledger append-only；override 通过 append 一条 `human:` check 实现；`latest_per_kind` 取每 kind 最新一条作为生效 verdict
- Bundle gate 阶段性放行：只检查 `IMPLEMENTED_CHECK_KINDS` 全 PASS（本 packet = 3 个）；新 kind 实现时必须同步加入 frozenset，contract test 自动 surface 缺失覆盖

来源：`docs/known-debts.md` debt #28 L2 first batch (2026-05-10) + L2 second batch (2026-05-10, debt #26 closure)。**全 7 个 verifier 已实现**；bundle gate 现要求 `IMPLEMENTED_CHECK_KINDS = frozenset(CheckKind)` 全 7 PASS。L0 已落地见 17F；metadata client V2 + bundle metadata fingerprint 折入也于本轮落地。

---

### 17F. Figure Corpus Crawler (L0)

**对应需求**：R8（snapshot / contract first）、R12（evaluation 单向性）、R15（迁移可解释 + 可回滚）

| Spec | 内容 |
|------|------|
| [figure-corpus-crawl.md](./figure-corpus-crawl.md) | L0 编排：5 SSRF gate + robots.txt + per-host token bucket + ETag/Last-Modified incremental + 持久化 frontier + 5 archive-aware fetcher（generic / cpae / wikisource / gutenberg / internet_archive）+ CrawlSink 写入 L1 CleaningStore + CLI；同时关闭 debt #19 via `live_archive_fetcher(...)` 工厂 |

**核心不变量**：

- `crawl/` 子包**允许** import `requests` / `urllib.robotparser` / `urllib.parse`（figure-vertical 唯一 HTTP 出口）
- `crawl/` 子包**禁止** import `Figure*Source` typed records / kernel modules / `lifeform_domain_figure.verification.*`（contract test AST 三类静态守门）
- 5 SSRF gate（scheme / host / path-prefix / redirect-1-hop-rescope / body-size-cap）全部在 `BaseHTTPClient.get` 强制
- robots.txt fail-closed（fetch failure → host 拒收）
- per-host token bucket 默认 0.5 req/s + burst 5
- frontier append-only + dedup by `request_id` (sha256(fetch_kind + url))；可 `resume_from_disk`
- `request_id` / `raw_sha256` / `byte_sha256` / `RawDocument.raw_sha256` 同字节流 = 同 hash（content-addressable 三段贯通）
- `live_archive_fetcher(...)` 返回 `LiveFetchedBytes` raw_payload（V2）；`offline_archive_fetcher()` 行为不变（V1 向后兼容）

来源：`docs/known-debts.md` debt #28 L0 + debt #19 V2 closure（2026-05-10）。`requests` 是新依赖。

---

### 18. Social Cognition Learning Layer

**对应需求**：R16（多人身份学习）、R17（Theory of Mind owner 分解）、R18（会话角色学习）、R19（共同基础学习）、R20（群体实体学习）、R-PE、R1、R3/R4、R7、R8、R11、R14、R15

| Spec | 内容 |
|------|------|
| [social_cognition/01_multi_party_identity.md](./social_cognition/01_multi_party_identity.md) | per-interlocutor keying、MemoryEntry subject/audience scope、wrong-person attribution PE |
| [social_cognition/02_theory_of_mind.md](./social_cognition/02_theory_of_mind.md) | belief / intent / feeling / preference 四类 ToM owner 分解，禁止 LLM classifier 成为 owner |
| [social_cognition/03_conversational_role.md](./social_cognition/03_conversational_role.md) | active speaker / addressee / subject / witness / overhearer per-turn role snapshot |
| [social_cognition/04_common_ground.md](./social_cognition/04_common_ground.md) | dyad / group common-ground owner、bounded recursion、reference-resolution PE |
| [social_cognition/05_joint_entity.md](./social_cognition/05_joint_entity.md) | group as adaptive owner、joint attention、joint commitment、group-level PE |

**核心不变量**：
- 不是多人 CRM schema：每个 social cognition state 必须有 owner、timescale、prediction、PE path、ETA consumption boundary
- LLM structured output 只能是 proposal source，不是 social state owner
- renderer 不从文本重建社会状态，只表达 owner snapshot → planner → renderer 的结果
- social PE 是 `prediction_error` / `credit` 的 typed 下游输入，evaluation 仍然只是 readout / gate

---

### 21. MCP Bundle Bridge (mcp-tools-bundle-bridge packet)

**对应需求**：R3, R4, R5, R8, R10, R11, R15

| Spec | 内容 |
|------|------|
| [mcp-bridge.md](./mcp-bridge.md) | `lifeform-mcp-bridge` wheel：把外部 MCP server 的 tools / resources / prompts 翻译成 `AffordanceDescriptor` / `IngestionEnvelope` / reviewed knowledge event；reviewed `.vzbridge.yaml` safety manifest；6 个 acceptance gate；`LifeformConfig.mcp_bridge_wiring` 三态默认 ACTIVE |

**核心不变量**：

- MCP server 不是 owner；`AffordanceDescriptor` / `DomainKnowledgeRecord` 由主项目内 owner 构造写入
- safety_model / cost_model / when_to_use(>=50) / when_not_to_use(>=50) 必须来自 reviewed `.vzbridge.yaml`，缺则 `MCPMissingSafetyManifestError` fail-loud
- MCP-supplied tools 与 in-process tools 共享 `AffordanceRegistry`；选择仍由 `AffordanceModule` z_t 投影驱动
- MCP server crash 不能让主进程崩溃；`AffordanceCandidate.blocked_reason="mcp_unavailable:<server>"`
- bridge wheel 禁止反向 import `volvence_zero.{cognition,memory,temporal,substrate,application,runtime}.*`
- 外部 repo 作为 git submodule 引入主项目（默认绑定 [`external/vz-bundle/`](../../external/vz-bundle)，对应 sibling 路径 `D:/GitHub/vz-bundle`，可后续推到 GitHub）；主项目 monorepo 体积不变重

来源：mcp-tools-bundle-bridge packet（2026-05-13）。Acceptance：6 个测试（`tests/contracts/test_mcp_*` + `tests/lifeform_e2e/test_mcp_*` + `tests/longitudinal/test_mcp_resource_becomes_durable_knowledge.py`）。

---

### 20. Owner Hydration (Packet D — long-horizon-closure)

**对应需求**：R5（连续记忆）, R6（反思与沉淀）, R8（快照优先 / 单一所有者）, R11（内部状态可发布）, R15（迁移可解释性 + 可回滚）

| Spec | 内容 |
|------|------|
| [owner-hydration.md](./owner-hydration.md) | `HydratableOwnerProtocol` 协议 + `OwnerPersistenceSnapshot` 类型 + `BrainConfig.owner_hydration_wiring` 三态 + 三个首批 hydratable owner（`SemanticStateStore` / `FollowupManager` / `VitalsModule`）跨 session 续接 |

**核心不变量**：

- 每个 hydratable owner 自己实现 `export_persistence_snapshot()` / `hydrate_from_persistence(...)`，外部 store 不直写 owner 内部
- 复用 `MemoryStore.persistence_backend`，新加 key 前缀 `owner_hydration/{owner_name}`
- hydration 失败必须抛 typed `HydrationError` 子类（fail-loudly，不允许 bare except）
- `BrainConfig.owner_hydration_wiring: WiringLevel = ACTIVE` 默认（long-horizon-closure follow-up）；`SHADOW` = export+log 不 import；`DISABLED` = 完全关闭
- 跨 user 隔离继承自 `MemoryStore` 的 per-user scope key 路径
- `LifeformSession.end_scene` 自动调用 `persist_owners()`（anonymous / 无 backend 时 no-op）

来源：long-horizon-closure Packet D（2026-05-12）。Acceptance：`tests/contracts/test_owner_hydration_protocol.py` + `tests/contracts/test_owner_hydration_failures_loud.py` + `tests/longitudinal/test_cross_session_owner_hydration.py`。

---

### 19. DLaaS Platform Layer（治理 / 编排基底，新增第三层 wheel 前缀）

**对应需求**：R2（稳定基底 + 自适应控制器）、R4（控制不在 token 空间）、R8（快照优先 / 单一所有者）、R11（内部状态可发布）、R15（迁移可解释性 + 可回滚）

| Spec | 内容 |
|------|------|
| [dlaas-platform.md](./dlaas-platform.md) | 6 个 `dlaas-platform-*` wheel 切分（contracts / registry / launcher / api / ops / eval）；typed `InteractionEnvelope` 路由表（chat/observe/feedback/teach/task/report/command）；`OutputAct` 包装；platform 不持有任何 cognitive state |
| [dlaas-api-v1.md](./dlaas-api-v1.md) | DLaaS v1 对外 API：OpenAI-compatible facade、native runtime envelope、adoption contract、protocol/training intake、environment/feedback convenience aliases、wake/sleep/status lifecycle |

**核心不变量**：
- `vz-*` 内核 7 个 wheel diff = 0 行（仅 substrate streaming additive 接口可例外，单独 review）
- `dlaas-platform-*` 不允许 import `volvence_zero.{cognition,memory,temporal,substrate,application,runtime}.*` 内部
- `interaction_type` 必须 typed enum dispatch，禁止从 `human_brief` 等自然语言字段关键词推断
- focus_persons / identity_links 不创建第二 owner；写入只走 `submit_profile_event`，scope_key 拼接 `UserIdentity`
- handoff trigger = 平台读 `rupture_state` 快照决定阈值；不在 kernel 加 handoff owner
- exam / audience / license 的 LLM judge 仅 readout，不反向写 reward / Face 梯度

来源：`docs/api/DLAAS_README.md`（EmoGPT DLaaS 公共 API 形状）；落地路线见 `docs/moving forward/dlaas-platform-rollout.md`。

---

## 设计源头与支撑文档

| 文档 | 内容 | 何时读 |
|------|------|--------|
| `docs/next_gen_emogpt.md` | **唯一设计源头**：系统需求 R-PE + R1-R20 + NL/ETA 算法详设（附录 A/B/C） | 理解需求根源和算法基础 |
| `docs/prd.md` | 产品需求文档：愿景、工程分解、必要脚手架、里程碑 | 理解工程规划和交付计划 |
| `archetecture.md` | 8 wheel 切分轴 + 替换映射 + 迁移路线 | 理解仓库与 wheel 切分思路 |
| `SPLIT.md` | 仓库边界 charter：Phase 1 monorepo → Phase 2 触发条件 | 理解仓库分裂时机与机械流程 |
| `docs/SYSTEM_DESIGN.md` | 系统架构设计：总体架构、模块职责、数据流、多时间尺度学习循环、wheel 边界、迁移策略 | 理解系统整体结构和模块关系 |
| `docs/DATA_CONTRACT.md` | 数据契约：快照 schema、模块接口、Slot 注册表、依赖图、wheel 边界、变更协议 | 理解模块间数据交换格式和约束 |
| `docs/CONTRACT_MIGRATION_LOG.md` | 契约迁移流水：planned / SHADOW slots、字段扩展、shared type slice changelog | 查实现阶段和 rollout notes，避免污染稳定契约 |
| `docs/DEBUG_SYSTEM.md` | 调试与可观测性体系：5 层可观测性架构、契约守卫、检查点与回滚、跨 wheel 调试边界 | 理解如何调试和监控系统运行 |
| `docs/EVALUATION_SYSTEM.md` | 评估体系：6 族评估框架、双轨评估隔离、信号回馈、lifeform-bench family report | 理解如何评估系统表现和驱动学习 |
| `docs/package_usage.md` | 本机 package 安装、稳定 Brain API、HF/Qwen 可选 runtime、其他项目接入边界 | 其他项目需要调用 core package 时 |

### 文档依赖图

```
docs/next_gen_emogpt.md  ← 唯一设计源头（R-PE + R1-R20 + NL/ETA 算法）
    │
    ├──→ docs/prd.md  ← 产品需求（愿景、能力域分解、里程碑、wheel 边界）
    │       │
    │       ├──→ archetecture.md  ← 8 wheel 切分轴
    │       │       │
    │       │       └──→ SPLIT.md  ← 仓库边界 charter
    │       │
    │       └──→ docs/specs/00_INDEX.md  ← 分层知识入口（本文件）
    │               │
    │               └──→ docs/specs/*.md  ← 各能力域 Spec
    │
    ├──→ docs/SYSTEM_DESIGN.md  ← 系统架构（模块职责、数据流、wheel 边界）
    │       │
    │       ├──→ docs/DATA_CONTRACT.md  ← 数据契约（快照 schema、接口、wheel 边界）
    │       │
    │       ├──→ docs/DEBUG_SYSTEM.md  ← 调试体系（可观测性、契约守卫、跨 wheel 边界）
    │       │       │
    │       │       └──→ docs/EVALUATION_SYSTEM.md  ← 评估体系（调试数据是评估输入）
    │       │
    │       └──→ docs/EVALUATION_SYSTEM.md  ← 评估体系（6 族框架 + lifeform-bench）
    │
    └──→ .cursor/rules/*.mdc  ← 编码规则（从 R1-R15 推导）
```

**读取顺序建议**：
1. 改代码前：`00_INDEX.md` → 目标 spec → 需要时读 `DATA_CONTRACT.md`
2. 理解架构：`next_gen_emogpt.md` → `SYSTEM_DESIGN.md` → `DATA_CONTRACT.md`
3. 理解评估/调试：`EVALUATION_SYSTEM.md` ↔ `DEBUG_SYSTEM.md`（互相引用）

---

## Spec 文件模板

每个 spec 文件应包含以下结构：

```markdown
# {能力域名} Spec

> Status: draft | stable
> Last updated: YYYY-MM-DD
> 对应需求: R{x}, R{y}, ...

## 要解决的问题
一句话描述此能力域要解决什么问题。

## 关键不变量
此能力域必须始终满足的约束。

## 工程挑战
实现此能力域的核心技术难点。

## 算法候选
来自 docs/next_gen_emogpt.md 的算法基础。

## 接口契约
- 此能力域消费的输入
- 此能力域产出的输出

## 与其他能力域的关系
依赖哪些能力域，被哪些能力域依赖。

## 变更日志
重要变更记录。
```
