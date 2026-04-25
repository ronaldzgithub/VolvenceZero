# Domain Experience Layer Spec

> Status: draft
> Last updated: 2026-04-25
> 对应需求: R5, R6, R7, R8, R12, R15

## 要解决的问题

如何让系统承载可复用的垂直经验包，而不是把领域经验散落在 prompt、硬编码规则或运行时私有状态里。

Domain Experience Layer 的目标是为女性情绪决策支持、职业陪伴、家庭转变支持、学习陪伴、健康导航等场景提供同一套冷启动经验入口：

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

## 变更日志

- 2026-04-25: 初始版本，新增通用 Domain Experience Package 层，编译到现有 application stores、rare-heavy checkpoint 和 typed prior update，不新增 runtime slot。
