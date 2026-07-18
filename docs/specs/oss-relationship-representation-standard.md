# OSS Option A — Relationship Representation Standard Spec

> Status: draft（发布决策待定，见 §9 Open Decisions）
> Last updated: 2026-07-18
> 对应需求: R8（快照优先 / 契约优先）、R11（内部状态可命名可发布）、R15（可回滚迁移）
> 商业动机: `docs/business/BP/volvence-bp-sales-wedge-v1-cn.md` P5「关系维度的默认结算层」——
> 协议开放、智能闭源；schema 是让别人**接入**这一层的接口，不是复制这一层的配方。
> 配对文档（待写）: `docs/external/relationship-representation-rfc-v0.md`（公开 RFC，方法论 owner；
> 本文件是内部 spec，代码级契约 owner。两者必须保持一致，改一处同步另一处）。

## 要解决的问题

对外发布一个开源的「关系表征标准」：一套零依赖、可序列化、带 conformance 测试的 schema，
定义「一个 AI 与一个人的长期关系状态」应当如何被结构化表示——使任何第三方 agent / 运行时
都能生产或消费这种表示，而 Volvence 商业运行时是该标准的最强实现。

**明确不是什么**：本方案不发布任何模型权重、不发布任何 owner 实现逻辑、不发布任何学习机制。
「开源人类表征大模型」中"模型"的部分由 Option B（`oss-relationship-encoder.md`）承担。

## 发布物（三件套）

| # | 发布物 | 现状 | 动作 |
|---|---|---|---|
| 1 | **标准 wheel**（工作名 `relationship-standard`，见 §9） | 不存在，从现有代码抽取 | 本 spec 主体 |
| 2 | **基准**：`companion-bench` | 已 Apache-2.0，30 公开合成场景 | 文档挂接：声明为标准的官方度量衡 |
| 3 | **参考实现**：`companion-ref-harness` | 已 Apache-2.0 | 文档挂接：升级为标准的 naive 参考消费者 |

2、3 已具备，Option A 的工程主体是 #1。

## 标准 wheel 的内容范围

### MUST include（v0 核心）

| 内容 | 当前位置 | 说明 |
|---|---|---|
| 9 类 semantic owner 的 snapshot value 类型（`UserModelSnapshot` / `RelationshipStateSnapshot` / `CommitmentSnapshot` / `OpenLoopSnapshot` / `BoundaryConsentSnapshot` 等） | `packages/vz-cognition/src/volvence_zero/semantic_state/contracts.py` | 纯 frozen dataclass value 类型；owner 实现**不**随行 |
| Slot 注册表（`SEMANTIC_OWNER_SLOTS` 及各 slot 的 owner / value_type / 语义描述） | 同上 | 标准的"字典页" |
| ToM 记录类型（`OtherMindRecord`、`BELIEF/INTENT/FEELING/PREFERENCE` 枚举） | `packages/vz-contracts/src/volvence_zero/social_cognition.py` | 关于他人的四槽位表示 |
| Canonical JSON 序列化 + 稳定 hash | 仿 `companion_bench.spec.scenario_hash` 模式新写 | 跨语言互操作的锚点 |
| Conformance test kit | 新写 | 第三方自测其产出是否符合标准 |
| `SemanticEmbeddingBackend` 接口（仅 Protocol + fallback stub） | `packages/vz-contracts/src/volvence_zero/semantic_embedding.py` | Option B 的模型以此接缝接入；真实 backend 不随行 |

### SHOULD include（v0 决策项，建议纳入）

| 内容 | 当前位置 | 纳入理由 / 风险 |
|---|---|---|
| 最小快照内核（`Snapshot` 容器 + `propagate`，**不含** guards / learned 机制） | `packages/vz-contracts/src/volvence_zero/runtime/kernel.py` 子集 | 让标准可执行而非纯文档；内核本身是 commodity 机制，不含工艺 |

### MUST NOT include（护城河边界，一票否决）

- `learned_update.py` / `tensor_backend.py` / `owner_hydration.py`（学习与持久化工艺）；
  `owner_prediction.py` 的**机制半边**（`settle_owner_prediction` / `OwnerPredictionSettlement`）
  ——注意其**表示半边**（`OwnerPredictionKind` / `OwnerPredictionSignal`）因被 9 类 snapshot
  内嵌而随行标准（SSOT 强制，实施时确认）
- `vz-cognition` 的任何 owner 实现（`owners.py` / `store.py` / `llm_runtime.py` / `proposal_runtime.py`）
- `vz-temporal` 全部（metacontroller / internal RL / joint loop）
- 任何 `.snap` / `.bs` / PEFT checkpoint（`data/`、`packages/lifeform-domain-emogpt/bootstraps/`、`.local/peft-checkpoints/`）
- `external/companionbench-heldout/`（私有 held-out）
- `docs/business/` 全目录及任何客户 / JV / 财务口径
- 全部 prompt 资产（LLM 集中管理的 prompt 属表达层工艺）

## 关键不变量

1. **单一 SSOT，禁止双份 schema**：标准 wheel 成为 9 类 snapshot value 类型与 ToM 记录类型的
   **唯一定义处**；`vz-contracts` / `vz-cognition` 改为 import + re-export（保持既有 import 路径
   向后兼容）。禁止"内部一份、公开一份靠人肉同步"——那是 SSOT 破坏（`ssot-module-boundaries.mdc`）。
2. **依赖方向**：`relationship-standard`（零依赖，纯 stdlib）← `vz-contracts` ← 其余一切。
   标准 wheel 不 import 任何 `volvence_zero.*` / `lifeform_*`，由新守门测试强制
   （仿 `tests/contracts/test_companion_bench_no_internal_imports.py`）。
3. **快照不可变**：随行类型全部 frozen dataclass；标准文档明示消费者不得原地修改。
4. **发布即冻结 v0**：schema 字段变更走公开 RFC 变更流程（对齐 companion-bench
   "adding an action is an RFC-level change" 先例）；内部演进先于公开发布的字段走
   `docs/CONTRACT_MIGRATION_LOG.md`，公开后走 RFC。
5. **标准与实现分离**：标准 wheel 只回答"关系状态长什么样"，不回答"如何算出来"。
   任何泄漏工艺的 PR 直接拒绝。

## 拆包步骤（两阶段，各自可回滚）

### Phase A1 — 内部重构（不涉及开源，先行合入）

1. 在 `packages/` 新建 `relationship-standard/`（独立 `pyproject.toml`，`dependencies = []`，
   Python 3.11+，暂标 Proprietary——**license 翻转是 A2 的事，A1 纯重构**）。
2. 将 §MUST include 的 value 类型移入；`vz-cognition/semantic_state/contracts.py` 与
   `vz-contracts/social_cognition.py` 改为 re-export（既有 import 路径零破坏）。
3. 四处同改（`first-principles-not-patches.mdc` 强制）：
   - `docs/DATA_CONTRACT.md` — shared type slice 与 slot 注册表标注新 SSOT 位置
   - `archetecture.md` — 库表新增一行
   - `tests/contracts/test_import_boundaries.py` — `ALLOWED_VZ_UPSTREAM` 加入
     `relationship_standard`（vz-contracts / vz-cognition 可 import；反向禁止）
   - 相关 `pyproject.toml` 依赖声明
4. Acceptance：全部既有 contract tests 无回归；新增
   `tests/contracts/test_relationship_standard_no_internal_imports.py`。

**回滚**：A1 是纯移动 + re-export，git revert 即可，无状态迁移。

### Phase A2 — 公开发布

1. license 翻转：`relationship-standard/pyproject.toml` 改 Apache-2.0；每个源文件加
   Apache header；扩展 `tests/contracts/test_apache_license_header_present.py` 覆盖本包。
2. 写公开 RFC `docs/external/relationship-representation-rfc-v0.md`
   （仿 `companion-bench-rfc-v0.md`：动机 / schema 定义 / conformance / 变更流程 / anti-claims）。
3. 公开渠道：PyPI 发 wheel + 公开 mirror repo（只含本包 + RFC + conformance kit + LICENSE，
   **不是** monorepo 公开；mirror 由脚本从 monorepo 单向同步，monorepo 是 SSOT）。
4. 发布包文档：新建 `docs/moving forward/relationship-standard-public-launch-packet.md`
   （仿 companion-bench 先例：命名、渠道、announcement 口径、FAQ）。
5. 三件套挂接：companion-bench README / RFC 增加"标准的官方基准"互引；
   companion-ref-harness 增加 conformance 示例。

**回滚**：公开前任何一步可撤；公开后 schema 不可收回，但 v0 范围刻意最小——
这正是 MUST/SHOULD/MUST-NOT 三档划分的意义。

## 清洗清单（A2 发布前 checklist，逐项打勾）

- [ ] mirror repo 内 `grep -ri` 客户名（华大 / Mobi / 摩比 / 高盖伦 / 1688 / 恒一 / 灵智佳创…）零命中
- [ ] 无任何 `docs/business/`、`docs/scenarios/` 内容混入
- [ ] 无 `.snap` / `.bs` / `.safetensors` / `.ndjson` 数据文件
- [ ] 无内部 R-ID 叙事文档（`next_gen_emogpt.md` 等设计源头不外流；RFC 用对外语言重述）
- [ ] LICENSE 文件在 mirror repo 根部存在（monorepo 根继续无 LICENSE，维持 Proprietary 默认）
- [ ] conformance kit 在干净环境 `pip install` 后可独立运行

## 接口契约

- **消费输入**：无运行时输入；本包是纯类型 + 序列化 + conformance 库。
- **产出输出**：frozen dataclass 类型集、canonical JSON codec、稳定 hash、conformance 断言集、
  `SemanticEmbeddingBackend` Protocol。
- **下游**：`vz-contracts`（re-export）、`vz-cognition`（owner 实现消费类型）、
  `companion-ref-harness`（参考消费者）、Option B encoder（trajectory/label schema 来源）、
  第三方 agent / 运行时。

## 与其他能力域的关系

- **6A 语义状态一等 Owner**：owner 实现不动，只有 value 类型的定义处迁移。
- **semantic-embedding-backend spec**：接缝 Protocol 随标准公开，真实 backend 仍在 `vz-substrate` 私有。
- **Option B**（`oss-relationship-encoder.md`）：硬依赖本标准的 canonical trajectory / label schema。
- **SPLIT.md**：本 spec 落地即回答其"What this charter does NOT decide"中的
  "Public open-source licensing" 未决项（仅对本包；monorepo 其余部分维持 Proprietary）。

## 对外口径纪律（anti-claims）

- 不称"开源大模型"——v0 没有权重，措辞为 **open standard / open schema**。
- 不称"human world model"——该词受 `docs/specs/human-world-model-ablation.md` 的
  evidence gate 约束，`first-stage-retained` 之前不对外使用。
- 与既有客户口径一致："核心内核不开源"（`docs/scenarios/sea/haiyang_first_meeting_qa.md`）
  ——本方案开的是接口层，不与该承诺冲突，对外 FAQ 需主动说明这一点。

## Open Decisions（已全部拍板，见实施记录）

| # | 决策 | 结果 |
|---|---|---|
| 1 | 包名 / 标准名 | **`companion-standard`**（与 Companion Bench 同族，用户确认）；对外名 Relationship Representation Standard |
| 2 | 最小快照内核是否纳入 v0（§SHOULD） | 纳入 `Snapshot` 容器；**`propagate` 不随行**（实施时评估：它与 guards / RuntimeModule / wiring / placeholder 机制不可分割，属 MUST-NOT 工艺面；标准以规范性文字定义传播语义，`volvence_zero.runtime.kernel.propagate` 是其中一个实现） |
| 3 | license 终版 | Apache-2.0，与 companion-bench 一致 |
| 4 | 公开时点 | 与 Companion Bench 合并 announcement（见 launch packet §4） |
| 5 | JSON Schema | v0 已发：`docs/external/relationship-representation-trajectory.schema.json`，由 `companion_standard.jsonschema` 生成，drift 守门在 `test_companion_standard_conformance.py` |

## 实施记录（2026-07-18，Phase A1 + A2 in-repo 完成）

实际落地与本 spec 草稿的差异（代码优先原则，以此为准）：

- **包名**：`packages/companion-standard/`（模块 `companion_standard`），非工作名 `relationship-standard`。
- **模块布局**：`canonical.py`（canonical JSON + stable hash）/ `trajectory.py`（轨迹 schema，
  含 `LabelSource` 枚举——judge 刻意不可表示，R12 类型级强制）/ `semantic_state.py` /
  `owner_prediction.py`（仅表示半边）/ `social_cognition.py`（仅 `OtherMindRecord` + 枚举；
  owner snapshot 因内嵌 runtime 诊断字段留内部）/ `embedding.py`（Protocol + stub；
  backend 注册机制留 `vz-contracts`）/ `kernel.py`（仅 `Snapshot`）/ `conformance.py` / `jsonschema.py`。
- **import 边界实现**：新增 `COMPANION_STANDARD_IMPORTERS`（`tests/contracts/test_import_boundaries.py`）
  ——内核 wheel 中仅 `vz-contracts` / `vz-cognition` 两个 re-export 站点可直接 import
  `companion_standard`，其余 wheel 走 `volvence_zero.*` 稳定面（单一 choke point）。
  反向守门：`tests/contracts/test_companion_standard_no_internal_imports.py`
  （含纯 stdlib 守门）。SSOT 身份守门（re-export 必须是同一对象）：
  `tests/contracts/test_companion_standard_conformance.py`。
- **A1 验收**：`pytest tests/contracts` 3154 passed；4 个失败经 stash 复跑确认为既有失败
  （lscb 品牌守门 / feeling-about-other drift / predictive-heads shadow / dlaas dispatch），与本次无关。
- **A2 in-repo 完成件**：license 翻转 + header + 守门扩展；公开 RFC
  `docs/external/relationship-representation-rfc-v0.md`；JSON Schema 导出；mirror 脚本
  `scripts/companion_standard/publish_public_standard.sh`；launch packet
  `docs/moving forward/companion-standard-public-launch-packet.md`（含清洗 checklist 执行记录，
  全部零命中）。外部动作（PyPI / mirror repo / announcement）留手动，见 launch packet §6。
- **B-M1 同步完成**：`packages/companion-trajgen/`（见 `oss-relationship-encoder.md` 实施记录）。

## 变更日志

- 2026-07-18 v0 草稿。来源：DINQ BP 启发（`docs/business/dinq-bp-inspiration-2026-07.md`）+
  仓库勘查（vz-contracts 零依赖 / companion-bench Apache 先例 / SPLIT.md licensing 未决项）。
- 2026-07-18 Phase A1 + A2 in-repo 实施完成，Open Decisions 全部拍板，新增实施记录一节。
