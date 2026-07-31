---
name: docs 全面对账更新
overview: 按"代码为准"原则，把 docs 下权威文档（DATA_CONTRACT、specs 全量、顶层架构/系统文档、状态类文档）与当前 39-wheel 实现对账更新，分六个可独立验证的批次推进。
todos:
  - id: batch-a-data-contract
    content: DATA_CONTRACT §6 补登 9 个 slot 并修正 affordance/thinking_loop/平台措辞
    status: pending
  - id: batch-a-index
    content: 00_INDEX 挂载 27 个缺失 spec、修 17B/17C 编号冲突、加 Gate 2 conditioned 入口
    status: pending
  - id: batch-b-evidence
    content: evidence_program.md 登记 Gate 2 条件化纵向 prereg 证据台账
    status: pending
  - id: batch-b-embedding
    content: semantic-embedding-backend + domain-experience-layer 补真实 backend 宽向量 case gate
    status: pending
  - id: batch-b-character
    content: character-prefix-package.md 补启动 WiringLevel 安全默认与 env 契约
    status: pending
  - id: batch-b-console
    content: relationship-memory-console.md 补 HTTP/幂等账本与 P0/P5 状态边界
    status: pending
  - id: batch-c-system-design
    content: 重写 SYSTEM_DESIGN.md（7 vertical、39 wheel、新名称）
    status: pending
  - id: batch-c-package-usage
    content: 重写 package_usage.md 为完整包地图
    status: pending
  - id: batch-c-arch-prd
    content: archetecture.md 包表补齐 + prd.md 里程碑状态更新
    status: pending
  - id: batch-d-status
    content: "重写 current.md（纳入 #92 终局）与 todo.md（改为现状指针）"
    status: pending
  - id: batch-e-guides
    content: 重写 EVALUATION_SYSTEM / DEBUG_SYSTEM / SYSTEM_GUIDE
    status: pending
  - id: batch-f-stale-specs
    content: 刷新 28 个过时 spec（三档：核对更新 / 批量轻校对 / 仅修 header）
    status: pending
  - id: verify-contracts
    content: 批次 A 后运行 pytest tests/contracts，全程 grep 校验引用真实性
    status: pending
isProject: false
---

# docs 全面对账更新计划

## 背景

三路探查核对了文档与代码的偏差（基准：`packages/` 共 39 个 wheel，代码约 56 个正式 slot）。结论：契约文档整体最活但有一周级缺口；顶层系统文档普遍停在 5 月口径；specs 索引缺 27 个文件；近期落地的 4 个领域有明确 spec 缺口。所有更新以当前代码行为为准（AGENTS.md 第 1.3 条），每个文档更新后同步修正文内 "Last updated" 日期。

## 批次 A：契约核心（最高优先）

**[docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md)**
- §6 注册表补登 9 个已接线 slot：`decision_workspace`、`evaluation_mid`、`evaluation_expensive`、`evaluation_cross_generation`、`protocol_phase`、`protocol_registry`、`protocol_revision_log`、`protocol_reflection`、`protocol_revision_queue`（owner/value_type/dependencies/wiring_level 从代码 ClassVar 与 `FinalRolloutConfig` 抄录）
- 修正过时措辞：§6.1 `affordance`「未接 runtime propagate」（实际 ACTIVE）、`thinking_loop`「新建中 Phase 1」（`lifeform-thinking` 已存在）、平台 slot「Phase 1 占位」（`dlaas-platform-registry` 已实现）
- 核对主表 ClassVar 默认与 §6.X rollout 的约 10 处不一致，确属有意双 SSOT 的加注释说明，属漂移的修正

**[docs/specs/00_INDEX.md](docs/specs/00_INDEX.md)**
- 挂载 27 个未入索引的 spec（重点：`profile-registry.md`、`protocol-runtime.md`、`companion-bench.md`、`evaluation-cascade.md`、`persona_market/` 整目录、figure/growth-advisor 协议族）
- 解决 17B/17C 章节编号冲突
- 为 Gate 2 conditioned longitudinal 增加索引条目，指向 `temporal-abstraction.md` + `personal-conditioning.md` + `evidence_program.md` 三处入口

## 批次 B：近期实现的 spec 缺口（高优先）

- **[docs/specs/evidence_program.md](docs/specs/evidence_program.md)**：登记 Gate 2 条件化纵向 prereg（schema、manifest、`scripts/preregister_gate2_longitudinal_conditioned.py` 等入口、seed1301 formal artifact、与 v35 对照臂/kill 条件、冻结后宣称规则）
- **[docs/specs/semantic-embedding-backend.md](docs/specs/semantic-embedding-backend.md)** + **[docs/specs/domain-experience-layer.md](docs/specs/domain-experience-layer.md)**：补真实 backend 双阈值（0.02 vs stub 0.16）、`_REAL_ACTION_EMBEDDING_DIM=64` 宽向量二次判定、action vs reflective alignment margin（对照 `vz-application/.../case_memory.py`）
- **[docs/specs/character-prefix-package.md](docs/specs/character-prefix-package.md)**：补启动安全默认契约（`CHARACTER_PACKAGE_MODE=shadow` / `CHARACTER_PACKAGE_WIRING` env、进程默认 SHADOW、未知 character_id 不注入、权重 SHA 不一致 fail loudly 的失败面）；说明张无忌 residual SHADOW 与「新角色只走 Prefix-KV」的双轨过渡
- **[docs/specs/relationship-memory-console.md](docs/specs/relationship-memory-console.md)**：补 HTTP 路由/请求响应形状、`request_fingerprint` 幂等账本；把「P0/P1 landed vs P5 连续性指标未落地」的状态边界写清

## 批次 C：顶层架构文档重写

- **[docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md)**：vertical 从 5 更新为 7（补 repair30、digital-employee）；补 `vz-embodiment-ant`、`companion-{encoder,trajgen,ref-harness,camel-baseline}`、`lifeform-{cultivation,synthetic-data,protocol-runtime,mcp-bridge}` 等约 11 个缺失 wheel；替换 MemoryOS/Orchestrator 旧称
- **[docs/package_usage.md](docs/package_usage.md)**：重写为完整 39-wheel 包地图（分 vz-* / lifeform-* / dlaas-* / companion-* 四族），修正 `VolvenceZero` 旧路径，校验 Brain API 示例仍可用
- **[archetecture.md](archetecture.md)**：包表补齐约 17 个缺失 wheel，更新 header 日期（原则与 R-ID 映射不动）
- **[docs/prd.md](docs/prd.md)**：里程碑状态从 5 月口径更新到当前（learned backends、#92 thesis-rejected、ecology station1 BLOCK），vertical 列表补齐；愿景部分不动

## 批次 D：状态类文档重写（用户已确认重写）

- **[docs/current.md](docs/current.md)**：纳入 2026-07-31 #92 `thesis-rejected` 终局判词，清掉已由 currentstatus 记录为补齐的「剩余项」（session-held credit、thinking advisory 等），header 日期对齐
- **[docs/todo.md](docs/todo.md)**：重写为简短现状指针（已由 `known-debts.md` + `currentstatus.md` 取代，逐条标注原 gap 的落地情况）
- `currentstatus.md`、`known-debts.md`、`ccprogress.md` 已 fresh，不动

## 批次 E：系统级导览文档重写

- **[docs/EVALUATION_SYSTEM.md](docs/EVALUATION_SYSTEM.md)**：补 `evaluation_mid/expensive/cross_generation` 级联；把 F1–F6 中「目标框架」与「代码已落地 readout」明确分区（对照 `EvaluationBackbone` 实际字段）；纳入 7 月 gate/promotion 与 #92 证据终局
- **[docs/DEBUG_SYSTEM.md](docs/DEBUG_SYSTEM.md)**：按当前落地面重写（`dialogue_trace`、evolution NDJSON、gate artifacts），Layer 4/5 明确标注为未实现愿景
- **[docs/SYSTEM_GUIDE.md](docs/SYSTEM_GUIDE.md)**：保留教学结构，更新包清单与「当前默认路径」章节

## 批次 F：过时 specs 刷新（28 个 < 2026-06-01）

按漂移信号分三档处理：
- **核对后更新**（实现明确演进）：`substrate-upgrade-protocol.md`（common adapter 指纹 + State-KV 蒸馏）、`persona-lora-concurrency.md`（共享 Adapter 路由）、`core-package-boundary.md`（brain.py 演进）、`aac-commitment-lifecycle.md`、`social_cognition/01、03`、`perf-baseline.md`、`lifeform-template.md`（角色案例/审阅叠加）、`interlocutor-state.md`、`rupture-and-repair.md`
- **批量轻校对**（信号弱/族群性）：figure 协议族 8 份、growth-advisor 族 4 份、`persona_market/` 2 份 —— 逐份快速对照实现，漂移则修，未漂移则只更新 header 日期并标注复核日期
- **仅修 header/标注**：`aac-lifecycle.md`（背景稿）、`audit-owner.md`、`handoff-queue-slo.md`、`cms-atlas-titans-uplift-shadow-evidence-2026-05-06.md`（证据冻结件，标注为快照不再更新）、`social_cognition/05` 等文内 Last updated 与 git 不一致的统一修正

## 验证方式

- 每处 slot/契约登记均从代码抄录（ClassVar、`FinalRolloutConfig`、实现文件），不凭文档旧文推断
- 批次 A 完成后运行 `pytest tests/contracts`（若存在 DATA_CONTRACT 与代码一致性测试则必须过）
- 其余批次为纯文档改动，逐批人工 spot-check 引用的文件路径与 slot 名真实存在（grep 校验）
- 不改任何代码；不触碰 papers/business/scenarios/moving forward 等非权威目录（范围已确认）

## 建议执行顺序

A → B → D → C → E → F。A/B 是契约与近期实现对账（最影响后续开发），D 很小，C/E 是大重写，F 长尾可分多轮。