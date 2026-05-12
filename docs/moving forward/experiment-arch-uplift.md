# 架构改造 Spec — 支撑 P0 完整实验

> **目的**：为 [`experiment.md`](./experiment.md) v2 规划的阶段 B/C/D 提供 cross-cutting 架构基础设施。本 spec 描述 9 项跨 packet 的架构改造（5 项必做 + 4 项建议做），它们不属于任何单一业务 packet，必须作为独立的"架构 packet"先行推进。
>
> **与其它文档的关系**：
> - [`experiment.md`](./experiment.md) — 产品 / 规划层：阶段 A/B/C/D + 每个业务 packet 的目标与起跑顺序
> - [`experiment-phase-a-brief.md`](./experiment-phase-a-brief.md) — 阶段 A 现状核查 brief：每条候选的 owner / slot / 耦合
> - **本文档** — 架构 / 工程层：让阶段 B/C/D 可以**安全、可观测、可回滚**地推进所需的 cross-cutting 基础设施
> - [`探索方向.md`](./探索方向.md) — 研究借鉴清单（来源）
>
> **遵循约束**：所有改造必须遵守 R2（冻结基底）/ R4（内部控制不在 token 长期 RL）/ R8（snapshot 隔离）/ R15（迁移可解释可回滚）四条铁律。本 spec 是这些铁律在"实验基础设施"维度的具体应用。
>
> **状态**：草案 / 待评审 — 不是工程任务清单本身，而是"工程任务必须遵循的架构契约"。

---

## 0. 现状的实验承载力瓶颈

阶段 A 现状核查 + 本 spec 调研发现的"如不改造则无法完整跑 P0 实验"的瓶颈：

| 维度 | 当前实现 | 阶段 B/C 后所需 | gap 大小 |
|---|---|---|---|
| Profile 注册 | `packages/vz-runtime/src/volvence_zero/agent/dialogue/_legacy.py` 中 11 个 `if profile_label == "X"` 硬编码分支 | 4 新候选 + 笛卡尔组合（SYS-1⊗COG-1 等） | **L**（结构性） |
| Evaluation cascade | 单层 `EvaluationSnapshot`（read-only score） | cheap→mid→expensive 三层 + cross-generation aggregator | **L**（greenfield） |
| Audit owner | 不存在 | 独立 `audit` owner + `AuditSnapshot` slot + N8 工具集 | **L**（greenfield） |
| WiringLevel 粒度 | module 级（`TemporalModule.default_wiring_level = SHADOW`） | candidate / capability 级（同 module 内多 capability 独立切换） | **M** |
| Snapshot 扩展机制 | DATA_CONTRACT §6 表格 + `final_wiring.py` 手工同步 | 单一 source of truth；CI 检测偏离 | **M** |
| Scenario 框架 | scripted dialogue case 单 interlocutor fixture | multi-party / wrong-person / witness / private-leakage | **M** |
| Substrate hook | `feature_surface` / `residual_activations` 接口存在但 backend 未必填 | 所有 backend 强制 implement + 一致 schema | **S-M** |
| metric_means 抽取 | 硬编码 key 集 | schema-driven 可扩展 | **S** |
| Shadow evidence harness | 每次手工写脚本（沿用 `run_atlas_titans_cms_shadow_smoke.py`） | 参数化模板 + 自动 evidence 文档生成 | **S** |

---

## 1. 改造概览

| 编号 | 改造项 | 类别 | 工作量 | 影响范围 | ripple 风险 |
|---|---|---|---|---|---|
| **A1** | Profile composition layer | 硬骨架 | M (4-6 PR) | `vz-runtime` | 中 |
| **A2** | Evaluation cascade 基础设施 | 硬骨架 | L (8-12 PR) | `vz-cognition/evaluation` + 多下游 | **高** |
| **A3** | Capability-level WiringLevel | 硬骨架 | M (3-5 PR) | `vz-contracts` + 所有 module | 中 |
| **A4** | DATA_CONTRACT ↔ final_wiring SSOT 同步 | 硬骨架 | S (1-2 PR) | `docs/DATA_CONTRACT.md` + 新 contract test | 低 |
| **A5** | ModificationGate audit-evidence 输入通道 | 硬骨架 | M (3-5 PR) | `vz-cognition/credit` + 新 audit owner | **高** |
| **B1** | Scenario package framework（多人/长程） | 软骨架 | M (4-6 PR) | `tests/scripted_dialogue/` + scenario gen | 低 |
| **B2** | Substrate hook 一致性强制 | 软骨架 | S-M (2-4 PR) | `vz-substrate` 各 backend | 中 |
| **B3** | metric_means schema-driven 抽取 | 软骨架 | S (2-3 PR) | dialogue benchmark + 各 owner 声明侧 | 低 |
| **B4** | Shadow evidence harness 模板化 | 软骨架 | S (1-2 PR) | `scripts/` + spec 模板 | 低 |

**类别说明**：

- **硬骨架（A 组）**：不做就完全无法跑阶段 C 的并行 SHADOW 实验，或会让阶段 B packet 之间相互打架。**必须先于业务 packet 完成**。
- **软骨架（B 组）**：不做也能跑实验，但每条候选 packet 内部会重复发明轮子、重复劳动。**建议先于对应候选 packet 完成**。

---

## 2. 核心硬骨架（A 组，5 项）

### A1. Profile composition layer

#### 现状

`packages/vz-runtime/src/volvence_zero/agent/dialogue/_legacy.py` 的 `build_standard_dialogue_runner` 用 11 个 `if profile_label == "X"` 分支硬编码 profile dispatch。阶段 C 加 4 候选 → 15 分支；组合 profile（SYS-1 ⊗ COG-1）→ 31 分支（含 baseline）；任何 candidate 内部参数微调都要再分裂 profile_label。

#### 改造目标

引入 declarative `ProfileSpec → Capability bundle` composition：

```python
@dataclass(frozen=True)
class ProfileCapability:
    name: str                    # 'cpd-beta-switch', 'least-control-credit', ...
    flag_path: str               # 'temporal.cpd_enabled', 'credit.least_control_enabled'
    default: bool
    applies_to_owner: str        # 'temporal_abstraction' / 'credit' / ...
    requires: tuple[str, ...] = ()   # 依赖的其他 capability
    conflicts_with: tuple[str, ...] = ()  # 互斥 capability

@dataclass(frozen=True)
class ProfileSpec:
    label: str                   # 'cpd-beta-switch-shadow'
    capabilities: tuple[str, ...]
    base_profile: str = "pe-eta"
```

`build_standard_dialogue_runner` 改为：

1. 解析 `ProfileSpec` (lookup by label)
2. 验证 capability 依赖图（DAG + 无冲突）
3. 应用 capability flag 到 `FinalRolloutConfig`
4. 构造 `AgentSessionRunner`

#### 涉及文件 / 新建模块

- 新建 `packages/vz-runtime/src/volvence_zero/agent/profile_registry.py`
- 新建 `docs/specs/profile-registry.md`
- 修改 `_legacy.py`：保留 11 个分支作为 registry 内置 ProfileSpec；`build_standard_dialogue_runner` 调用 registry
- 修改 `default_dialogue_ablation_profiles()` / `default_dialogue_strong_proof_profiles()` 等：从 registry 自动派生

#### 关键不变量 / 红线

1. **现有 11 个 profile 行为必须 byte-equivalent**：迁移期新旧 dispatch 并跑，对照 `metric_means`，delta == 0（或在 float noise 内）才合并
2. **依赖图必须是 DAG**：循环依赖编译期 error
3. **conflicts_with 必须强制**：profile 中同时声明互斥 capability → 编译期 error
4. **legacy if-elif 分支**：保留 ≥1 release cycle 作为 DISABLED 对照，确认 SHADOW evidence 后才删除

#### 安全迁移协议

| 阶段 | 状态 | Done 标志 |
|---|---|---|
| 1 | registry 实现完成，11 个 profile 注册到 registry，新旧 dispatch 并存 | unit test 覆盖 registry 各种边界 |
| 2 | `build_standard_dialogue_runner` 改为 registry-first，legacy if-elif 作为 fallback；CI 跑双 dispatch 对照 | 11 个 profile metric_means delta == 0 |
| 3 | 移除 legacy if-elif | dialogue paper-suite 5 seeds × 4 cases PASS |

#### 工作量

M（4-6 PR / 1.5-2.5 周）

#### 涉及现有 spec

- 新增 [`docs/specs/profile-registry.md`](../specs/profile-registry.md)
- 触及 [`docs/specs/contract-runtime.md`](../specs/contract-runtime.md)（profile 是 WiringLevel 之上的 capability 选择）

---

### A2. Evaluation cascade 基础设施（EVO-2 升级版）

#### 现状

`packages/vz-cognition/src/volvence_zero/evaluation/` 当前是单层结构：`EvaluationModule` 发布单个 `EvaluationSnapshot`，所有 score（acceptance gate / ablation delta / readout）混在一层。`docs/specs/evaluation.md` 的 EVO-2 cascade 在代码中**零基础**（`grep cascade` 在 evaluation 包返回 0 命中）。

#### 改造目标

引入三层 + 一个 aggregator：

```
EvaluationCascade
├── cheap_layer        (每 turn；deterministic；不调 LLM)
│   ├── PE delta / regime stability / posterior_drift
│   ├── mp.* probe pass-rate（CMA-2 收编）
│   └── persona-geometry readout（COG-3 起跑面）
├── mid_layer          (每场景；paper-suite-small × N seeds)
│   ├── acceptance gate set
│   ├── ablation delta vs baseline
│   └── counterfactual contribution readout（COG-1 起跑面）
├── expensive_layer    (每代际 / 每 rare-heavy)
│   ├── head-to-head winrate
│   ├── cross-generation aggregate
│   └── LLM-judge naturalness / coherence（仅 readout，非 gate）
└── CrossGenerationAggregator
    ├── DM-7 / EVO-6 head-to-head 表
    └── ModificationGate evidence 注入接口
```

#### 涉及文件 / 新建模块

新建：

```
packages/vz-cognition/src/volvence_zero/evaluation/
├── cascade.py                     # EvaluationCascade 主 orchestrator
├── cheap_layer.py                 # 现有 EvaluationModule 内 cheap-tier readout
├── mid_layer.py                   # paper-suite-small benchmark hook
├── expensive_layer.py             # LLM-judge readout + winrate aggregator
└── cross_generation_aggregator.py
```

修改：

- `packages/vz-cognition/src/volvence_zero/evaluation/backbone.py` — 现有 EvaluationModule 收编为 cheap_layer 的实现，**EvaluationSnapshot 字段保持完全不变**
- `docs/specs/evaluation.md` — 扩展 cascade 章节
- 新建 `docs/specs/evaluation-cascade.md` — 三层 schema / 跨层依赖图 / failure semantics

#### 关键不变量 / 红线

1. **现有 `EvaluationSnapshot` 必须保持 field-identical**：被 `prediction_error` / `credit` / `regime` / `reflection` 至少 4 个下游 owner 消费，禁止拆分。Cascade 在外侧加新 snapshot，cheap_layer 输出现有 EvaluationSnapshot
2. **LLM-judge 仅 readout，不进 gate**（R12 + OA-2 Mind/Face 隔离 + EVO 特殊条款）
3. **每层 fail-loudly**：cascade 失败不允许静默回退（[`no-swallow-errors-no-hasattr-abuse.mdc`](../../.cursor/rules/no-swallow-errors-no-hasattr-abuse.mdc)）；任一层失败时 ModificationGate **fail-closed**
4. **下游 opt-in 消费**：mid / expensive layer 各自发布**新的独立 snapshot 族**，下游显式声明消费哪些层
5. **cheap_layer 必须可独立运行**：mid / expensive 缺失时 cheap_layer 正常工作，否则会让"每 turn 评估"塌缩

#### 安全迁移协议

| 阶段 | 状态 | Done 标志 |
|---|---|---|
| 1 | `evaluation-cascade.md` spec 完成评审；三层 schema + 依赖图冻结 | spec 内审 PASS |
| 2 | cheap_layer 收编现有 EvaluationModule，对照测试 EvaluationSnapshot 字段不变 | 所有现有 evaluation 下游 PASS |
| 3 | mid_layer 实现 + paper-suite-small benchmark hook；SHADOW 模式 | dialogue paper-suite 5 seeds × 4 cases PASS |
| 4 | expensive_layer + cross_generation_aggregator；SHADOW 模式 | ETA strong-proof claim verdicts 不退化 |
| 5 | ModificationGate evidence 注入接口完成；切 ACTIVE | gate 行为对照原 EvaluationSnapshot consumer，决策一致 |

#### 工作量

L（8-12 PR / 4-6 周）

#### 涉及现有 spec

- 扩展 [`docs/specs/evaluation.md`](../specs/evaluation.md)
- 新增 [`docs/specs/evaluation-cascade.md`](../specs/evaluation-cascade.md)
- 触及 [`docs/specs/credit-and-self-modification.md`](../specs/credit-and-self-modification.md)（ModificationGate evidence 来源）

---

### A3. Capability-level WiringLevel + flag plumbing

#### 现状

`WiringLevel` 当前在 module 级（`TemporalModule.default_wiring_level = SHADOW` / `CreditModule.default_wiring_level = SHADOW`）。阶段 C 要求"同 `TemporalModule` 内 CPD-β-switch capability SHADOW，但其他 capability 保持 ACTIVE"——module 级 wiring 做不到。

#### 改造目标

引入 `CapabilityWiring`：

```python
@dataclass(frozen=True)
class CapabilityWiring:
    capability_name: str
    wiring_level: WiringLevel
    owner: str

# Module 持有：
class TemporalModule:
    capabilities: Mapping[str, CapabilityWiring]
```

`FinalRolloutConfig` 从扁平 flag 升级为按 owner 嵌套的 capability map；module 内部读取 `self.capabilities[name].wiring_level` 决定 shadow path 是否执行；profile registry（A1）的 capability 与本 wiring level 一对一对应。

#### 涉及文件 / 新建模块

- `packages/vz-contracts/src/volvence_zero/contracts/wiring.py` — 扩展 `WiringLevel` enum 无需新枚举；新增 `CapabilityWiring`
- `packages/vz-runtime/src/volvence_zero/integration/final_wiring.py` — `FinalRolloutConfig` capability map 嵌套化
- 各 RuntimeModule 子类（`TemporalModule` / `CreditModule` / `EvaluationModule` 等）— 增加 `capabilities` 字段
- 触及 [`docs/specs/contract-runtime.md`](../specs/contract-runtime.md) §WiringLevel

#### 关键不变量 / 红线

1. **现有 module-level wiring 默认行为完全保留**：capability map 默认为空 → 等价于"整个 module 走 module-level wiring"
2. **正交组合**：`SYS-1 capability` 与 `COG-1 capability` 各自切换，互不影响
3. **编译期一致性**：profile 引用的 capability 必须存在；capability `requires` 依赖图必须无环
4. **fail-loudly**：capability map 中引用不存在的 capability → 启动时 fail，**禁止**静默忽略

#### 安全迁移协议

| 阶段 | 状态 | Done 标志 |
|---|---|---|
| 1 | `CapabilityWiring` 数据类型 + 编译期校验；capability map 默认为空 | 现有所有 module 行为不变 |
| 2 | 各 module 内部读取 capability map 的 shadow path | dialogue paper-suite PASS |
| 3 | A1 profile registry 引用 capability，端到端测试 | profile 切换正确改变 capability wiring |

#### 工作量

M（3-5 PR / 1-2 周）

#### 涉及现有 spec

- 扩展 [`docs/specs/contract-runtime.md`](../specs/contract-runtime.md)

---

### A4. DATA_CONTRACT ↔ final_wiring SSOT 同步机制

#### 现状

阶段 A brief 识别的硬偏离：4 个 ToM slot（`belief_about_other` / `intent_about_other` / `feeling_about_other` / `preference_about_other`）+ `conversational_role` + `multi_party_identity` + `social_prediction[_error]` 在 `packages/vz-runtime/src/volvence_zero/integration/final_wiring.py` 默认 ACTIVE，但 `docs/DATA_CONTRACT.md` §6 表格仍标 "planned migration mirror"。

#### 改造目标（推荐方案 B：轻量 contract test）

不改 spec 表格格式，新增 pytest contract test：

```python
# tests/contracts/test_data_contract_wiring_sync.py
def test_data_contract_slots_match_final_wiring():
    """Each slot row in DATA_CONTRACT.md §6 must match FinalRolloutConfig defaults."""
    declared = parse_data_contract_section_6("docs/DATA_CONTRACT.md")
    actual = FinalRolloutConfig().to_slot_wiring_map()
    diff = {k: (declared[k], actual[k]) for k in declared if declared[k] != actual[k]}
    assert not diff, f"DATA_CONTRACT §6 偏离 final_wiring: {diff}"
```

**先行 prerequisite**（不是改造，是已识别的偏离修复）：把 DATA_CONTRACT §6 的 7 个 social slot 状态改为 ACTIVE。

#### 涉及文件 / 新建模块

- 新建 `tests/contracts/test_data_contract_wiring_sync.py`
- 修改 `docs/DATA_CONTRACT.md` §6 表格（7 个 social slot 状态校正）
- 新建简单的 markdown 表格 parser（或 yaml 嵌入）

#### 关键不变量 / 红线

1. **偏离即 FAIL**：contract test 失败 → CI 阻止合并
2. **故意 planned 状态**：spec 允许"planned but not yet implemented" 状态，contract test 必须支持此标记（如表格中显式 `PLANNED` 字符串），对应代码 default 必须是 `DISABLED`
3. **任何 slot 状态变更**：必须同时改 spec 表格 + final_wiring.py default → 一次提交，否则 CI FAIL

#### 安全迁移协议

| 阶段 | 状态 | Done 标志 |
|---|---|---|
| 1 | 修复现存 7 slot 偏离（spec 表格改为 ACTIVE） | contract test 跑 fixture 验证 parser 正确 |
| 2 | contract test 接入 CI | 全套测试 PASS |

#### 工作量

S（1-2 PR / 3-5 天）

#### 涉及现有 spec

- 修改 [`docs/DATA_CONTRACT.md`](../DATA_CONTRACT.md) §6

---

### A5. ModificationGate 的 audit-evidence 输入通道

#### 现状

`packages/vz-cognition/src/volvence_zero/credit/gate.py:663` 的 `evaluate_gate_reasons` 签名：

```python
def evaluate_gate_reasons(*, proposal: ModificationProposal, evaluation_snapshot: EvaluationSnapshot) -> tuple[str, ...]:
```

只消费 `EvaluationSnapshot + ModificationProposal`。OA-4 新增的 audit 输出（risk score / transcript / tool trace）**无处放**。

#### 改造目标

扩展 gate 接口为可消费三类证据，**fail-closed 默认**：

```python
def evaluate_gate_reasons(
    *,
    proposal: ModificationProposal,
    evaluation_snapshot: EvaluationSnapshot,
    audit_snapshot: AuditSnapshot | None = None,
    audit_required: bool = False,
) -> tuple[str, ...]:
    """三类证据：(1) calibrated evaluation readout / (2) capacity cap / (3) audit transcript。
    任一类硬证据缺失 → 默认 BLOCK。
    """
```

#### 涉及文件 / 新建模块

- 修改 `packages/vz-cognition/src/volvence_zero/credit/gate.py`
- 新建 `packages/vz-cognition/src/volvence_zero/audit/` 目录（owner 骨架，本 packet 仅定义 snapshot schema + publisher 接口；具体 audit-agent 实现归 OA-4 业务 packet）
- 新建 `docs/specs/audit-owner.md`
- 修改 `docs/DATA_CONTRACT.md` §6 注册 `audit` slot
- 触及 [`docs/specs/credit-and-self-modification.md`](../specs/credit-and-self-modification.md) §ModificationGate

#### 关键不变量 / 红线

1. **第一阶段必须用 `audit_snapshot=None` 默认 + `audit_required=False`** — 保持现有所有 gate 调用方行为不变。**禁止**默认 fail-closed，否则所有未接入 audit 的现有路径会被一刀切阻断
2. **第二阶段（OA-4 落地后）**：默认值改为 `audit_required=True` for rare-heavy artifact promotion 路径；dialogue-online 路径保持 `audit_required=False`
3. **SSOT 划分**：`evaluation` = calibrated readout（R12 read-only）；`audit` = staged gate evidence（可持有 transcript / tool trace）；**禁止互相 mutate**
4. **audit publisher 与 gate consumer 分离**：`audit` owner 只 publish AuditSnapshot；`credit/gate` 是 consumer；orchestrator 在 ModificationGate 决策时**消费** audit 输出，**禁止** audit owner 直接 mutate `credit` 状态

#### 安全迁移协议

| 阶段 | 状态 | Done 标志 |
|---|---|---|
| 1 | gate 接口扩展（`audit_snapshot=None` 默认）；`audit` owner snapshot schema + publisher 接口 stub | 现有 gate 调用方 0 行改动；现有 gate 行为完全等于改造前 |
| 2 | OA-4 业务 packet 实现 audit-agent 内容（不在本 A5 范围） | OA-4 packet 自有 Done 标准 |
| 3 | rare-heavy artifact promotion 路径切换 `audit_required=True` | `run_multi_artifact_acceptance_benchmark` PASS |

#### 工作量

M（3-5 PR / 1.5-2 周）— 接口 + audit owner 骨架；audit-agent 内容归 OA-4 业务 packet

#### 涉及现有 spec

- 扩展 [`docs/specs/credit-and-self-modification.md`](../specs/credit-and-self-modification.md)
- 新增 [`docs/specs/audit-owner.md`](../specs/audit-owner.md)
- 修改 [`docs/DATA_CONTRACT.md`](../DATA_CONTRACT.md) §6（新增 `audit` slot 行）

---

## 3. 软骨架（B 组，4 项）

### B1. Scenario package framework（多人 + 长程）

#### 现状

`ScriptedDialogueCase` 当前主要支持单 interlocutor fixture。阶段 C 的 **COG-2** 必须需要 wrong-person / witness / private-leakage 多人场景；阶段 D 组合 profile 需要更长时间窗景。

#### 改造目标

- `ScriptedDialogueCase` 扩展字段支持多 interlocutor / role / `EnvironmentEvent` 帧
- 场景生成支持 `(regime, intensity, rupture)` 三维参数化（EVO-3 QD scenario archive 最小骨架）
- per-scenario fixture 在 `tests/longitudinal/` 与 `tests/scripted_dialogue/` 之间共享 schema

#### 涉及文件 / 新建模块

- 修改 `ScriptedDialogueCase` dataclass，新增字段（用默认值保持向后兼容）
- 新建 `tests/scripted_dialogue/multi_party/` fixture 目录
- 触及 [`docs/specs/social_cognition/03_conversational_role.md`](../specs/social_cognition/03_conversational_role.md)

#### 关键不变量 / 红线

1. **现有单 interlocutor case 必须保持兼容**：扩展字段使用默认值（如 `interlocutors: tuple[str, ...] = ("default",)`)
2. **scenario 生成必须是离线 / pre-commit** — 禁止运行时 LLM-as-curator 生成场景（[`scenario-package-generation.mdc`](../../.cursor/rules/scenario-package-generation.mdc)）
3. **关键词匹配禁令**：场景标签 / 触发条件不得用 keyword matching 决定逻辑（[`no-keyword-matching-hacks.mdc`](../../.cursor/rules/no-keyword-matching-hacks.mdc)）

#### 工作量

M（4-6 PR / 2-3 周）

---

### B2. Substrate hook 一致性强制

#### 现状

`SubstrateSnapshot.feature_surface` / `residual_activations` 接口存在但 backend 未必填。阶段 C 的 COG-3（persona geometry readout）依赖这些字段实填。

#### 改造目标

- `SubstrateAdapter` 抽象基类强制声明 `feature_surface` / `residual_activations` 必须填充
- 现有 backend 逐个补全：builtin / huggingface / synthetic / open-weight
- 新增 `tests/substrate/test_feature_hook_completeness.py`：每个 backend 必须填且 schema 一致

#### 涉及文件 / 新建模块

- 修改 `packages/vz-substrate/src/volvence_zero/substrate/adapter.py`
- 各 backend 实现补全
- 新增测试

#### 关键不变量 / 红线

1. **两阶段强制**：先标记 `recommended but not required`，逐个 backend 补全；全部补全后再升级为 abstract method
2. **schema 一致性**：所有 backend 的 `feature_surface` 必须用同一 dtype / shape 协议（截断 / padding 规则在 spec 中规定）
3. **R2 冻结基底**：substrate hook 暴露 hidden state 仅用于 read-only readout，**禁止**反向写回梯度

#### 工作量

S-M（2-4 PR / 1-1.5 周）

---

### B3. metric_means schema-driven 抽取

#### 现状

阶段 A brief 反复出现的"benchmark `metric_means` 不抽 COCOA readout / 不抽 persona drift / 不抽 mp.*" — 当前 metric_means 是硬编码 key 集，下游消费者依赖固定 key。

#### 改造目标

每个 owner 在 snapshot 发布时声明"哪些 metric 可被 benchmark 抽"：

```python
class RuntimeModule:
    def declare_benchmark_metrics(self) -> tuple[BenchmarkMetricDescriptor, ...]:
        """Owner-declared metrics that should be extracted into benchmark.metric_means."""
```

benchmark `metric_means` 自动 `union(hardcoded_keys, owner_declared_keys)`，候选起跑时只需在 owner 内部添加 readout，不需要改 benchmark 代码。

#### 关键不变量 / 红线

1. **union 而非替换**：现有硬编码 key 全部保留；owner declared 是增量
2. **声明编译期可见**：owner 启动时一次声明，运行时不允许动态变更（避免 metric_means 形状随 turn 变化）

#### 工作量

S（2-3 PR / 1 周）

---

### B4. Shadow evidence harness 模板化

#### 现状

`scripts/run_atlas_titans_cms_shadow_smoke.py` + `docs/specs/cms-atlas-titans-uplift-shadow-evidence-2026-05-06.md` 已示范完整 SHADOW evidence 流水（5 seeds × N cases × 2 profiles → 88 metric delta 表 → ACTIVE 决策），但每次都是手工写脚本。

#### 改造目标

- 新增 `scripts/run_shadow_evidence_template.py`（参数化版本）：接受 `--baseline-profile` / `--candidate-profile` / `--cases` / `--seeds`
- 自动生成 evidence 文档：`docs/specs/<candidate>-shadow-evidence-<date>.md`
- 包含统一的"per-metric delta 表 / acceptance gate / rollback 开关说明"section 模板

#### 工作量

S（1-2 PR / 3-5 天）

---

## 4. 依赖图与推进顺序

```
                  ┌──────────────────────────────────────┐
                  │  A4 DATA_CONTRACT SSOT 同步           │
                  │  （任何后续改动的硬 prerequisite）     │
                  │  S, 1-2 PR                            │
                  └────────────────┬─────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │ A1 Profile      │  │ A3 Capability-  │  │ B2 Substrate    │
   │   registry      │  │   level wiring  │  │   hook 强制      │
   │ M, 4-6 PR       │  │ M, 3-5 PR       │  │ S-M, 2-4 PR     │
   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
            │  (A1 + A3 必须同步设计；接口先决议)        │
            └──────────────┬────────────────────────────┘
                           ▼
   ┌──────────────────────────────────────────────────┐
   │ A2 Evaluation cascade 基础设施 (L, 8-12 PR)       │
   │ + B3 metric_means schema-driven (S, 2-3 PR)       │
   │ ↑ 内含 CMA-2 mp.* 收编                            │
   └──────────────────────┬───────────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────────┐
   │ A5 ModificationGate audit-evidence 接口 (M)        │
   │ + B4 Shadow evidence harness 模板 (S)             │
   └──────────────────────┬───────────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────────┐
   │ 阶段 B packet 2-5（业务 packet）：                │
   │   SYS-2+DM-4 → OA-1+OA-2 → OA-4 → OA-3            │
   │ 这些 packet 现在终于有干净的 substrate 可以接入    │
   └──────────────────────┬───────────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────────┐
   │ B1 Scenario package framework (M)                 │
   │   （COG-2 起跑前必须完成）                         │
   └──────────────────────┬───────────────────────────┘
                          │
                          ▼
                  ┌──────────────────────┐
                  │  阶段 C 4 候选 SHADOW │
                  │  按 brief 顺位起跑    │
                  └──────────────────────┘
```

**关键 ordering 约束**：

1. **A4 最先**：任何其它改造引用 slot 前，必须先消除 spec ↔ wiring 偏离
2. **A1 + A3 同步设计**：一为声明侧，一为执行侧，本质同一抽象的两端。**必须在同一 spec 里一次设计完毕**再拆 PR 实施
3. **A2 在 A1/A3 之后**：cascade 配置由 capability 与 profile 决定；A2 内部三层 schema 冻结后才动代码
4. **A5 接口先决议**：在 OA-4 业务 packet 启动前，A5 接口必须完成；audit owner 骨架与 audit-agent 内容拆开做
5. **B1 在 COG-2 起跑前**：阶段 C 顺位 4（COG-2）之前必须完成 B1

---

## 5. 安全协议（所有架构 packet 通用）

### 5.1 WiringLevel 三态迁移

每项架构改造必须遵循：

| 阶段 | wiring | 行为 |
|---|---|---|
| 1 | DISABLED | 新逻辑不执行；旧逻辑唯一 source of truth |
| 2 | SHADOW | 新逻辑执行，输出对照旧逻辑；旧逻辑仍是正式 upstream |
| 3 | ACTIVE | 新逻辑成为正式 upstream；旧逻辑保留为 fallback ≥1 release cycle |
| 4 | (cleanup) | 移除旧逻辑 | 

**SHADOW 期最短要求**：≥5 seeds × paper-suite-small 全 PASS，且 metric_means delta == 0（或在 float noise 内）。

### 5.2 不回归红线（每个架构 packet Done 检查必含）

```
[ ] make test PASS
[ ] scripts/run_dialogue_paper_suite.sh 5 seeds × paper-suite-small PASS
[ ] scripts/run_eta_paper_suite.sh PASS
[ ] tests/longitudinal/test_vz_memprobe_*.py 4 个 probe PASS
[ ] 现有 11 个 dialogue profile metric_means byte-equivalent（A1 迁移期专项）
[ ] ETA strong-proof claim verdicts 不退化（claim_eta_internal_rl_advantage 保持 retain）
[ ] DATA_CONTRACT contract test PASS（A4 接入后）
[ ] 影响契约 shape 时同步 docs/specs/*.md 与 docs/DATA_CONTRACT.md
```

### 5.3 回滚开关要求

每个架构 packet 必须留：

- 环境变量 / config flag 形式的回滚开关
- PR 描述中明确"回滚指令"段：如何在不 revert PR 的情况下回退到旧路径
- ≥1 release cycle 的旧逻辑保留期

### 5.4 PR 模板补充字段

建议每个架构 packet PR 描述强制包含：

```markdown
## 架构 packet 自检

- [ ] 本 PR 属于 [A1/A2/A3/A4/A5/B1/B2/B3/B4] 中的 _____ 改造
- [ ] 当前阶段：[DISABLED / SHADOW / ACTIVE / cleanup]
- [ ] 现有 11 个 dialogue profile metric_means 对照结果：(粘贴 delta 表)
- [ ] 回滚开关：(env var / config flag 路径)
- [ ] 退出条件：(旧逻辑何时可删除)
- [ ] 影响的 spec 已同步：(列文件)
```

---

## 6. 二阶副作用与新形成的边界（必须 spec 化）

改造完成后**新形成的契约边界**——这些不是回归，但若不 spec 化，后续 packet 会重新踩：

| 改造 | 新形成的边界 | 必须 spec 化的事项 |
|---|---|---|
| A1 | profile registry 成为新 SSOT，若 legacy 残留则双 truth | "迁移完成"判定标准 + 删除 legacy dispatch 的退出条件 |
| A2 | cascade 三层之间的 fail propagation | cheap 失败时 mid/expensive 是否运行；任一层失败时 ModificationGate fail-closed |
| A3 | capability 依赖图可能形成环 | DAG 校验 + 编译期错误（运行时不允许） |
| A5 | audit owner 与 evaluation owner 双信号源 | evaluation = readout / audit = gate evidence 的 SSOT 划分（已写在 A5 红线，需在 spec 中正式记录） |
| B1 | 多人场景的 SemanticRecord per-interlocutor SSOT | `belief_assumption` vs `belief_about_other` 边界（COG-2 也涉及） |

---

## 7. 工作量与时间估计

| 类型 | 项 | PR 估计 | 时间估计（单人推进） |
|---|---|---|---|
| 硬骨架 | A1-A5 | 18-26 PR | 4-6 周 |
| 软骨架 | B1-B4 | 9-15 PR | 2-3 周 |
| 业务 packet | 阶段 B packet 0-5（见 [`experiment.md`](./experiment.md)） | 已在 experiment.md v2 | 3-5 周（架构就绪后） |
| 候选 SHADOW | 阶段 C 4 候选 | 每条 3-5 PR ≈ 12-20 PR | 1-2 月（并行可缩） |
| **合计** | | **约 45-70 PR** | **约 3-5 月** |

**并行潜力**：

- A4 完成后，A1 / A3 / B2 三组可独立并行（不同 owner）
- A2 期间，B3 可并行（同 owner 但不同子模块）
- 阶段 C 4 候选可全并行（不同 owner）

---

## 8. 关键风险与缓解

### 风险 1：A1 / A3 接口不同步设计 → 实施两遍

**显化路径**：A1 已落地但 A3 未完成 → 阶段 C 候选只能用 profile_label 整体切换，无法做 capability 级 ablation。

**缓解**：A1 + A3 在同一份 [`docs/specs/profile-registry.md`](../specs/profile-registry.md) 中一次设计完，再拆 PR 实施。

### 风险 2：A2 cheap_layer 兼容性破坏现有 evaluation 下游

**显化路径**：cascade 把现有 EvaluationModule 拆分到多层 snapshot → `prediction_error` / `credit` / `regime` / `reflection` 全部 rebase。

**缓解**：cheap_layer 输出现有 EvaluationSnapshot 字段完全不变，mid / expensive 在外侧加新 snapshot 族，下游 opt-in 消费。

### 风险 3：A5 接口默认值方向错误

**显化路径**：`evaluate_gate_reasons` 扩展时直接 fail-closed → 所有未接入 audit 的现有调用方被 gate 拒绝。

**缓解**：分两阶段。阶段 1：`audit_snapshot=None` + `audit_required=False` 默认；阶段 2（OA-4 落地后）：rare-heavy artifact 路径切换 `audit_required=True`，dialogue-online 路径保持原值。

### 风险 4：SHADOW 期不足导致过早合并 ACTIVE

**显化路径**：A2 cascade SHADOW 仅跑 1 seed → 切 ACTIVE 后发现 paper-suite-full 某个边界 case 偏移。

**缓解**：本 spec §5.1 强制 SHADOW 期 ≥5 seeds × paper-suite-small 全 PASS；PR 模板强制粘贴 metric_means delta 表。

### 风险 5：B2 一步到位强制 abstract method 让现存 backend broken

**显化路径**：`SubstrateAdapter` 直接把 `feature_surface` 升级为 abstract method → 未实现的 backend 启动即 fail。

**缓解**：两阶段（B2 红线）—— 先 recommended，逐个 backend 补全，全部补全后再升级 abstract。

---

## 9. 与既有规则的关系

本 spec 是以下 cursor 规则在"实验基础设施"维度的具体应用，不引入新原则：

- [`first-principles-not-patches.mdc`](../../.cursor/rules/first-principles-not-patches.mdc) — 库与所有权（A2 / A5 严格遵守 owner 划分；audit 不 mutate credit）
- [`ssot-module-boundaries.mdc`](../../.cursor/rules/ssot-module-boundaries.mdc) — A1 profile 不持有 module；A2 cascade 三层 snapshot 各自 owner；A5 audit publisher / credit consumer 分离
- [`no-swallow-errors-no-hasattr-abuse.mdc`](../../.cursor/rules/no-swallow-errors-no-hasattr-abuse.mdc) — A2 cascade fail-loudly；A3 capability 不存在 fail-loudly
- [`no-keyword-matching-hacks.mdc`](../../.cursor/rules/no-keyword-matching-hacks.mdc) — B1 scenario 标签不得 keyword-driven
- [`cursor-convergence-workflow.mdc`](../../.cursor/rules/cursor-convergence-workflow.mdc) — 每项架构改造单独成 packet；3-8 文件；可回滚；spec-sync gate
- [`scenario-package-generation.mdc`](../../.cursor/rules/scenario-package-generation.mdc) — B1 多人场景遵循生成规则
- [`llm-prompt-centralization.mdc`](../../.cursor/rules/llm-prompt-centralization.mdc) — A2 expensive_layer LLM-judge 用集中 prompt，禁止内联

---

## 10. 退出条件 / 完成判定标准

本架构 spec 的"完成"=阶段 C 可以**安全、并行、可观测、可回滚**地跑 4 候选 SHADOW profile。具体判定：

1. **A1-A5 全部 ACTIVE**（不是 SHADOW），legacy if-elif dispatch / 老 EvaluationModule 接口 / 老 gate 签名全部移除或标记 DISABLED
2. **B1-B4 至少 B2 / B3 / B4 完成**（B1 可滞后到 COG-2 起跑前）
3. **`tests/contracts/test_data_contract_wiring_sync.py` 与所有架构 packet contract test 在 CI 持续 PASS ≥1 release cycle**
4. **现有 11 个 dialogue profile + 4 memprobe + ETA strong-proof 全部 PASS**，且 metric delta vs 改造前 == 0（或在 float noise 内）
5. **`docs/specs/profile-registry.md` / `docs/specs/evaluation-cascade.md` / `docs/specs/audit-owner.md` 三份新 spec 评审通过**

满足上述 5 条 → 本 spec 进入"维护态"，阶段 C 业务 packet 可以正式起跑。
