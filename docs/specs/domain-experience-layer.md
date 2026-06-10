# Domain Experience Layer Spec

> Status: draft
> Last updated: 2026-04-29
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

- 2026-05-12: 与 [`mcp-bridge.md`](mcp-bridge.md) 对齐：MCP-derived knowledge 走 ingestion envelope path 进 `domain_knowledge`，不走 `DomainExperiencePackage` 直接 compile（外部 repo 解耦于 schema 演进）。
- 2026-04-29: 吸收原 `docs/application_*.md`（已删除）中"经验进入 ETA 的 4 个正式接入点"设计原则，补充 retrieval mix / fast-path priors / delayed credit / evolution gating 边界。
- 2026-04-25: 初始版本，新增通用 Domain Experience Package 层，编译到现有 application stores、rare-heavy checkpoint 和 typed prior update，不新增 runtime slot。
