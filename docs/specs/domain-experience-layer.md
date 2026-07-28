# Domain Experience Layer Spec

> Status: draft
> Last updated: 2026-07-28
> 对应需求: R5, R6, R7, R8, R12, R15

## 要解决的问题

如何让系统承载可复用的垂直经验包，而不是把领域经验散落在 prompt、硬编码规则或运行时私有状态里。

Domain Experience Layer 的目标是为关系陪伴、工程结对、女性情绪决策支持、职业陪伴、家庭转变支持、学习陪伴、健康导航等场景提供同一套冷启动经验入口：

- 领域知识种子
- 案例经验种子
- 策略 playbook
- 边界与转介提示
- 评测场景与证据锚点
- rollout / rare-heavy import 元数据

## 关键不变量

- Domain Experience Package 不是新的运行时 owner；它编译到现有 application owners 的公共数据结构。
- 运行时仍通过 `domain_knowledge`、`case_memory`、`strategy_playbook`、`boundary_policy`、`experience_consolidation`、`experience_fast_prior` 等既有 slot 发布状态。
- 包加载不得绕过 owner-side store、rare-heavy import、typed prior update、credit gate 或 checkpoint / rollback 纪律。
- 包内容是冷启动 scaffold 和评测锚点，不等同于真实长期经验已经成熟。
- 垂直内容不得通过人口学关键词硬编码行为；场景区分应进入 package 的案例、知识、边界和评测材料。

## 与 Mentor Intake / BehaviorProtocol 的边界

人机协同中，mentor 指导先进入 `MentorIntake` 分类入口，而不是默认写入 Domain Experience。二者的第一性区别是：

- `BehaviorProtocol` 回答"接下来应该按什么任务集 / 姿态 / 边界 / 策略行动"，可以在当前会话下一 turn 通过 `ActiveMixtureSnapshot` 生效。
- `Experience` 回答"过去发生过什么、结果如何、这类模式是否被验证"，进入 knowledge / case / consolidation / fast-prior 等 owner，形成慢层证据。

因此，mentor 说"以后遇到这种用户先澄清边界，再给两个选项"属于 `protocol` 或 `protocol_revision`；mentor 复盘"刚才这个用户因为回复太密集而流失"属于 `experience` / `case`，再由 PE 和信用归因进入慢整合。把后者直接变成硬协议会过拟合；把前者只记成经验会错过下一轮应立即改变行动的控制信号。

## 经验如何进入 ETA：4 个正式接入点

经验**可以**进入 ETA 的控制与学习闭环，但**不应**直接成为 `temporal` / metacontroller 的第二 owner。正式接入点恰好对应当前 4 个公共 surface：

1. **检索混合（retrieval mix）**——turn-time
   - ETA 通过 `retrieval_policy` 发布 `experience_domains` / `experience_weight`
   - 经验因此进入 ETA 的 turn-time 检索混合控制
2. **快路径先验（fast-path priors）**——turn-time
   - `case_memory` 提供 compact case hits（"类似事情过去怎么发生、怎么处理"）
   - `strategy_playbook` 提供 ordering / pacing prior（"这类问题通常先做什么、后做什么更稳"）
   - 这些 prior 影响 ETA 排序，但**不**直接重写 ETA 内部状态
3. **延迟信用（delayed credit）**——session-medium ~ background-slow
   - `experience_consolidation` 回看 `(abstract_action, regime, retrieval mix, action_family_version)` 的多轮结果
   - 通过 `experience_fast_prior` 把慢层 ledger 压缩成 regime / retrieval mix / action-family / regime-sequence bias
   - 这让经验进入 ETA 的 slow-shapes-fast 闭环，而不只停留在"当前轮命中了什么"
4. **演化裁决（evolution gating）**——rare-heavy
   - replay / benchmark / `EvolutionJudgement` + target-specific credit gate 裁决经验产物的 `promote / hold / rollback`
   - 经验不只被 ETA 读取，还**约束** ETA 及其外围 application prior 如何演化

**对应不变量**：

- ETA 可以消费经验，但不拥有经验本体
- `case_memory` / `strategy_playbook` / `experience_consolidation` / `experience_fast_prior` 不得回收 `temporal` / `memory` 的 owner 身份
- 经验 → ETA 的所有影响只能通过 public snapshot 或正式 gate 暴露
- 同样的边界对 knowledge → ETA 也成立：`domain_knowledge` 不因 turn-time usefulness 而吞并 `temporal` / `memory` owner

## Abstract action 到具体行动的落地

`TemporalAbstractionSnapshot.active_abstract_action` 是无业务语义的控制器身份，不能由
expression 根据 family id 建立字符串映射。具体行动沿以下正式交换落地：

1. `CaseMemoryModule` 从 `MemorySnapshot.retrieved_entries` 取得当前 owner-published
   语境，只用统一 semantic-embedding seam 判断是否是具体行动请求，并在已检索的
   reviewed `CaseMemoryRecord` 中选择一个语义近邻。
2. `CaseMemorySnapshot.action_grounding: CaseActionGrounding | None` 发布来源 case、
   与当前 abstract action 绑定的 intervention labels、owner 渲染的 action statement
   及 alignment/confidence。CaseMemory 是该解释的唯一 owner。
3. `ResponseAssemblyModule` 只校验 grounding 的 abstract action 与当前
   `TemporalAbstractionSnapshot` 一致，再发布
   `ResponseAssemblySnapshot.action_realization: ResponseActionRealization | None`。
4. expression 只渲染 `ResponseSpeechPlan` 中的 owner-published action statement；
   禁止重新检索案例、重排 intervention steps 或从自然语言做关键词动作分类。

`action_grounding=None` 是冷启动/非行动轮的正常显式状态。恢复旧行为的回滚方式是让
CaseMemory 不发布 grounding；不需要删除 temporal family、case records 或修改模型权重。

### Lived action 的慢层落地

reviewed live-through 不能只把 canonical action 留在普通 Memory 文本里，也不能依赖
profile 预置一个同答案的 signature case。当前正式路径为：

1. 环境 owner 发布 `SCENE_EVENT + terminal measurement` 的 canonical
   `EnvironmentOutcome`；`summary` 只承载已经发生的 action statement，`detail`
   承载 outcome，二者保持隔离。reviewed scene 可另外携带 outcome-free
   `EnvironmentActionSchema`，其 applicability conditions 与 action steps 不包含
   章节人物名或事后结果。
2. session-post 在 scene boundary 将其转换为 typed
   `ExperiencedActionEvidence`；同时绑定 outcome 提交时 temporal owner 已发布的
   `active_abstract_action / action_family_version / controller code digest`。该 lineage
   必须在 action-time 捕获，不能用 scene boundary 时已经变化的 family 反填；
   tool result、非 terminal scene event 不进入该路径。
3. `ApplicationPriorProposalBuilder` 经既有 PE-derived quality、credit gate 和
   structural writeback gate 编译为 `CaseMemoryPriorUpdate`，由
   `ApplicationCaseMemoryStore` 唯一写入；有 action schema 时 intervention ordering
   只使用 schema steps，原始 action/outcome 仅保留为审计描述。没有 schema 时仍可保存
   `latent-action-family:*` 证据，但 `intervention_ordering=()` 且标记
   `schema-pending`，因此 CaseMemory 不得把单次 episode 文本渲染成可迁移动作。
4. 新 session 必须通过 application owner persistence 重新加载该 lived-action
   case；评估时 CaseMemory 仅用 problem/user-state/risk 适用条件做语义选择，
   action steps 和 outcome 均不参与路由。expression 仍只消费
   `CaseActionGrounding.action_statement`。

因此，动作与结果都确实经过环境、PE、慢反思和 application owner；但“后来发生的结果”
不会被拼回当时的动作回答。停止收集 terminal scene evidence 或回滚 corresponding
CaseMemory checkpoint 即可退出，无需触碰 Memory/Temporal owner。

当前 ch-11 内部行为测试已覆盖一个未出现在 ledger/profile/case store 的陌生人威胁
场景：profile-answer holdout 后 baked 命中 `case:slow-loop:*:experienced-action:*`
并发布通用两步行动，cold 不具备该 schema，且回答不泄漏胡青牛、纪晓芙或 canonical
outcome。该测试证明 reviewed abstraction 的 owner 迁移路径；它不证明 Internal RL
自行发现了这组语义标签，也不替代外部盲评。

schema-holdout 测试进一步删除同一 scene 的 reviewed `EnvironmentActionSchema`：
action-time `discovered_family_*`、family version 与 `z_t` digest 仍进入 gated
application persistence，但新场景必须拒绝召回该非语义 case，且不得复述章节动作。
这证明 latent family lineage 已贯穿 bake；它同时明确了剩余断点：需要 background-slow
语义抽象器聚合同一稳定 family 的多次异质经历，经 `ModificationGate` 发布 typed schema，
之后才能声称 Internal RL 发现了可迁移动作抽象。

### Multi-experience action abstraction

当前 background-slow 收敛只属于 CaseMemory/application owner：

1. schema-free terminal outcome 额外携带 environment adapter 当时可观察的
   `situation_summary`；decoder 输入只含多条经历的
   `outcome_id / situation / executed_action`，明确不含 outcome 文本、reward、PE 数值或
   evaluation。
2. `ActionAbstractionOwner` 至少要求两条 outcome id 唯一、situation 不同且
   `action_family_id / action_family_version` 完全一致的经历。单例、重复 episode、
   family/version 冲突在调用 decoder 前 fail closed。
3. 默认 `NoOpActionAbstractionDecoder` 不产生任何候选；只有显式注入的
   structured background decoder 才能发布 `LearnedActionSchemaCandidate`。
   prompt 与 JSON schema 位于 application wheel 的 `prompts/`、`schemas/`，禁止关键词归纳。
4. owner 校验 candidate family/source closure、最低置信度和整句 episode-copy guard，
   再生成带 `ApplicationModificationEvidence` 的 `CaseMemoryPriorUpdate`。
5. runtime 将该 evidence 映射成 `ModificationProposal(BACKGROUND)`，调用正式
   `evaluate_gate_reasons()`；缺 evaluation snapshot、结构门失败或既有 credit block
   均不得写入。evaluation 在这里只是 promotion gate readout，不参与候选生成。
6. 单个 session 未满足计数时，CaseMemory 把 schema-free lineage 存入
   `CaseMemoryRecord.action_abstraction_evidence`，并随既有 checkpoint 原样恢复。
   下一 session 只通过 store owner 的 `pending_action_abstraction_evidence()` 合并历史
   与当前证据，禁止扫描 case description。内容完全一致的 outcome 只计一次，同 outcome
   的矛盾内容直接失败；晋升 record 的 typed promotion marker 会排除整个
   family/version 的 pending evidence，阻止重复 decoder/promotion。

promoted record 仍由 `ApplicationCaseMemoryStore` 唯一写入和持久化，expression 仍只消费
CaseMemory/ResponseAssembly 发布的 action realization。learned candidate 与 reviewer
发布的 `EnvironmentActionSchema` 是两个来源不同的 typed object，禁止混写 provenance。

第十一回 schema-holdout 目前只有一条独立经历，因此按本契约保持 `schema-pending`；
不能复制同一 scene 凑足计数。当前测试只用两条 synthetic heterogeneous evidence
证明 owner/gate 机制可达，不把它计作张无忌已经自主形成语义抽象。

## 工程挑战

- 用一套通用 schema 表达不同垂直场景，而不把女性陪伴、职业、健康等逻辑写进内核。
- 把 package 内容编译到现有 owner 数据结构，避免形成第二套 memory / experience owner。
- 在加载前验证 source、review、risk、boundary、ID 唯一性和证据强度。
- 让 package 既能作为产品冷启动经验，又能作为评测和人评材料的来源。

## 接口契约

**消费的输入**：

- `DomainExperiencePackage`
  - `DomainExperienceManifest`
  - `DomainKnowledgeRecord`
  - `CaseMemoryRecord`
  - `PlaybookRule`
  - `BoundaryPriorHint`
  - 可选 `ReviewedKnowledgeCandidate`
  - 可选 evaluation scenarios

**产出的输出**：

- `CompiledDomainExperiencePackage`
  - `ApplicationPriorUpdate`
  - `ApplicationRareHeavyCheckpoint`
  - 可直接 upsert 的 domain knowledge / case memory records
  - validation report
- `DomainExperienceApplicationReport`
  - 加载包 ID、写入数量、rare-heavy import 操作和持久化操作摘要

**当前实现口径**：

- `volvence_zero.application.domain_experience` 定义 package schema、validation、compiler 和 apply helpers。
- package compiler 只返回 typed outputs；直接写入发生在显式 apply helper 或 session / final wiring 边缘。
- `AgentSessionRunner` 可接收 `domain_experience_packages`，在构造阶段将 package 内容导入现有 application stores 与 rare-heavy state。
- `run_final_wiring_turn()` 可接收 `domain_experience_packages`，用于测试、评测或无状态调用中的 package 注入。
- package records 继续由 `ApplicationDomainKnowledgeStore` / `ApplicationCaseMemoryStore` 持久化；playbook 与 boundary hints 进入 `ApplicationRareHeavyState`，再由现有 runtime modules 读取。

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 契约式运行时 | package 不新增 runtime owner，必须沿既有 snapshot / owner 边界生效 |
| 依赖 | 连续记忆系统 | 案例经验进入 application case memory，不回收 `memory` 主 owner 身份 |
| 依赖 | 双轨学习 | package 可标注 world/self/shared 轨道，但轨道效果由现有 runtime owners 发布 |
| 依赖 | 评估体系 / 证据计划 | evaluation scenarios 可作为 scripted / blind review / rollout gate 的输入 |
| 协作 | 认知 Regime | playbook 和 boundary hint 可以指定 regime，但不直接控制 regime owner |
| 协作 | MCP Bundle Bridge | 外部 MCP server 的 `resources/list` + `prompts/list` 经 `MCPResourceAdapter` / `MCPPromptAdapter` 转换成 ingestion envelope / reviewed knowledge event，**不**走 `DomainExperiencePackage` 直接 compile path（外部 repo 不需要懂 application owner schema）；durable 化仍由 `vz-application` owner 负责。详见 [`docs/specs/mcp-bridge.md`](mcp-bridge.md)。 |

## 变更日志

- 2026-07-28: 冻结 CaseMemory-owned action-abstraction pending/promotion checkpoint 契约：schema-free evidence 可跨 session 恢复，consumer 只读类型化 owner API；矛盾 outcome fail loudly，已晋升 family/version 自动停止重复提案。
- 2026-07-28: 新增 multi-experience background action abstraction：structured decoder 不读 outcome/evaluation，CaseMemory owner 要求至少两条异质同族经历并校验 source closure；candidate 必须经正式 BACKGROUND ModificationGate 才能写入。ch-11 单例继续 fail closed，不声明自主 schema discovery。
- 2026-07-28: 增加 action-schema holdout 的 fail-closed 收敛：terminal outcome 在提交时绑定 temporal family/version/controller-code digest；无 schema 的 family-linked case 可持久化审计，但 intervention ordering 为空，禁止 expression 或 CaseMemory 把原 episode 复述成抽象策略。
- 2026-07-28: 新增 terminal `SCENE_EVENT` → `ExperiencedActionEvidence` → gated `CaseMemoryPriorUpdate` 的 lived-action 慢层落地；新会话从 application persistence 读取已生活过的动作，tool/non-terminal outcome 不参与，action statement 与未来 outcome 隔离。
- 2026-07-28: 新增 reviewed `EnvironmentActionSchema` 未见场景迁移收敛：action applicability 与 steps 经 terminal outcome/slow-loop 写入 CaseMemory，检索只读适用条件，行动/结果不参与路由；ch-11 held-out baked/cold 内部测试通过，外部盲评与 emergent schema discovery 仍未证明。
- 2026-07-28: 新增 CaseMemory-owned action grounding 与 ResponseAssembly action realization，把无语义 abstract-action id 经 reviewed case intervention steps 落成具体动作计划；语义选择走统一 embedding seam，expression 仅渲染，不成为第二动作 owner。
- 2026-05-12: 与 [`mcp-bridge.md`](mcp-bridge.md) 对齐：MCP-derived knowledge 走 ingestion envelope path 进 `domain_knowledge`，不走 `DomainExperiencePackage` 直接 compile（外部 repo 解耦于 schema 演进）。
- 2026-04-29: 吸收原 `docs/application_*.md`（已删除）中"经验进入 ETA 的 4 个正式接入点"设计原则，补充 retrieval mix / fast-path priors / delayed credit / evolution gating 边界。
- 2026-04-25: 初始版本，新增通用 Domain Experience Package 层，编译到现有 application stores、rare-heavy checkpoint 和 typed prior update，不新增 runtime slot。
