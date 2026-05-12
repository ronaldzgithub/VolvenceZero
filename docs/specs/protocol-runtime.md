# Behavior Protocol Runtime Spec

> Status: draft (Phase 0 design freeze)
> Last updated: 2026-05-11
> 对应需求: R-PE, R2, R3, R4, R6, R7, R8, R10, R11, R14, R15

## 要解决的问题

当前系统支持"垂直经验包"（`DomainExperiencePackage`，见 §11 Domain Experience Layer）和"角色 / 客户灵魂画像"（`CharacterSoulProfile` / `GrowthAdvisorProfile`，见 §17 Character Soul Bootstrap），但这些都是**离线 reviewed-frozen 静态数据**：人手写或人 review 后编译入 application owners，运行时不可修订、不可热加载、不可与其他同类协议混合激活。

实际场景需要的能力是：

1. **任务一次性吸收（task uptake）**：用户给一份运营指导书 / 角色设定 / 行为准则文档，系统**当场**把它转成可激活的行为配置，不需要批量同类语料、不需要重启、不需要训练
2. **多协议并行激活与调度**：同一会话中可能同时需要"谌老师私域陪伴"主协议 + "通用陪伴" 常驻协议 + "危机识别" 警戒协议；需要软混合而非硬选择
3. **优先级随经验自学**：哪个协议在哪类上下文下更合适，由 PE 历史决定，不靠手写 priority list
4. **冲突分层处理**：当不同协议要求不同行为时，区分边界冲突 / 偏好冲突 / 切换冲突 / 身份冲突，分别用不同机制处理
5. **协议条目可被反思修订**：strategy_priors 不是 frozen prompt，PE 验证不通过的条目权重下降，反思可以提议新条目
6. **身份核心永不替换**：无论激活哪个协议，identity core（"我是谁"）保持稳定（R7 Self 轨道、R14 持久身份）

当前 `lifeform-domain-growth-advisor/profiles/cheng_laoshi.py`（1455 行手写先验）旁路了上述全部 6 条能力——是 cold-start 脚手架，不是终态。本 spec 立 **Behavior Protocol Runtime** 子系统作为这些能力的宿主，并把 TaskUptake 作为它的入口适配器之一。

## 认知科学位置

`BehaviorProtocol` 不是 "idea"（folk-psychology 的瞬时念头 / 命题），而是一个**持续可激活的任务配置**。最贴近的对应（按重要性排序）：

| 认知科学概念 | 对应映射 | 来源 |
|---|---|---|
| **Task Set / Cognitive Set** | 整体最贴近：PFC 维持的 task representation，bias 全脑下游处理 | Monsell 2003；Miller & Cohen 2001 |
| **Schema** | 知识 + 期望 + 默认行为 | Bartlett 1932；Piaget |
| **Script** | 7-day 时间剧本、对话节奏 | Schank & Abelson 1977 |
| **Production Rules (IF-THEN)** | 4 个挖需求漏斗 | ACT-R (Anderson)；SOAR (Newell) |
| **HTN Goal Hierarchy** | "建立 LTV 关系" → "前 7 天建立信任" → 具体动作 | Sacerdoti |
| **Contention Scheduling Schema** | 多协议竞争激活机制 | Norman & Shallice 1986 |
| **Goal-Directed vs Habit System** | 协议初期 model-based deliberate；PE 验证后向 model-free habitual 迁移 | Daw, Niv, Dayan 2005 |
| **Active Inference Policy** | 协议 = policy；激活权重 = 后验；冲突 = expected free energy 比较 | Friston |
| **Frame (Goffman)** | 社会姿态 / 角色 mode | Goffman 1974 |

**一句话定位**：BehaviorProtocol 是 task set + script + production rules + boundary contracts 的捆绑，由"PFC 类机构"（本系统的 metacontroller + Protocol Runtime）持有，用来 bias 控制器层下游的全部处理。它对应 R3 的 β_t 切换单元（"切到这个任务模式"）+ R4 的 z_t 控制器代码（"在这个模式下用这套抽象决策"）。

**与日常用语的对应**：
- 不是 idea / thought
- 不是 plan / 计划
- 是"**身份模式 / 心态 / 姿态**"——例如"我现在是面试官模式 / 医生模式 / 哄孩子模式 / 谌老师模式"
- 每个模式都改变怎么看、怎么说、什么是失礼、什么是成功

## 关键不变量

1. **Identity Core 不在 Active Mixture 里**：identity core 是混合**之上**的恒定层，决定哪些协议根本不被允许激活（与 R7 Self 轨道、R14 持久身份对齐）
2. **边界契约跨协议取并集**：所有 active protocol 的 `boundary_contracts` 取并集，任一条 fire 即 block；边界永远不进 utility 计算（与 `bp-no-hard-sell` / R10 自修改门控对齐）
3. **协议优先级是后验，不是 hard list**：激活权重 = 后验（context match × PE history utility × identity gate），随时间被学（注：`drive_value` 信号源因 vitals 跨层边界限制 deferred，详见 §调度）
4. **β_t 切换由 PE 学，不由外部计数器打 tag**：当前 `applicability_scope=("growth_advisor:day3",)` 这种"按日历天数"的 string tag 是过渡形态，正式实现必须改为"按 PE 信号驱动的关系阶段"
5. **协议 schema 区分 frozen 与 PE-revisable 字段**：identity_core / boundary_contracts frozen；strategy_priors / temporal_arc / drive 初值带 PE 权重衰减入口
6. **协议必须自带 success / failure PE 定义**：协议入库时必须声明"成功长什么样、失败长什么样"的可测信号，否则 PE / reflection 无东西可学
7. **协议不创造新 application owner**：协议的具体内容（knowledge / case / playbook / boundary）仍编译到现有 `domain_knowledge` / `case_memory` / `strategy_playbook` / `boundary_policy` owner；ProtocolRuntime 只发布"当前激活混合"快照，不持有领域内容
8. **TaskUptake 是入口适配器**：PDF / 自然语言任务描述 / API 注入 / 文档目录扫描等都通过 TaskUptake 转成 `BehaviorProtocol`；ProtocolRuntime 不与具体输入格式耦合
9. **协议条目可被反思修订**：经过 PE 验证多次失败的 strategy_prior 权重下降；ReflectionEngine 可提议新条目；提议进入 R10 ModificationGate 决定是否落库
10. **协议生命周期可回滚**（R15）：每次协议加载、修订、退役都进 revision_log；可回到任意历史版本

## 与现有概念的边界（避免成为第二 owner）

本 spec 引入**新的运行时层**，但严格不与既有 owner 重复。下表把 BehaviorProtocol 与系统中已有的相邻概念分清楚：

| 概念 | 时间尺度 | 单元 | 是 owner 吗 | 与 BehaviorProtocol 关系 |
|---|---|---|---|---|
| **Substrate**（冻结 LLM） | rare-heavy | 模型权重 | 是（vz-substrate） | Protocol 不修改基底；只 bias 控制器 |
| **CharacterSoulProfile / GrowthAdvisorProfile** | 离线 reviewed | dataclass bundle | 否（编译入 application owners） | 当前是手写脚手架；未来作为"frozen=True 的 BehaviorProtocol"特例 |
| **DomainExperiencePackage** | 离线 reviewed | 编译产物 | 否（编译入 application owners） | BehaviorProtocol 是 superset：增加 identity / boundary / activation / PE-revision 元信息 |
| **CognitiveRegime**（R14） | 持久 per-interlocutor | 关系/任务身份 | 是（vz-cognition） | Regime 与 Protocol 正交：同一协议在不同 regime 下激活方式不同；不同协议可共享 regime |
| **DriveSpec / VitalsBootstrap** | always-on | 稳态驱动力 | 是（lifeform-vitals） | 协议只**初始化** drive 并提供 recharge 偏置；运行时 drive 由 VitalsModule 拥有 |
| **BoundaryPolicy** | turn-time gate | 硬约束 | 是（vz-cognition） | 协议提供 BoundaryPriorHint；BoundaryPolicy 仍是唯一执行 owner，但读 ProtocolRuntime 快照决定哪些 hint 当前生效 |
| **strategy_playbook / case_memory / domain_knowledge** | turn-time / session-medium | 内容存储 | 是（vz-application） | 协议条目编译入这些 owner；运行时通过 ProtocolRuntime 快照影响**权重**而非内容本体 |
| **ETA metacontroller** | online-fast | z_t / β_t | 是（vz-temporal） | 协议为 metacontroller 提供 prior（哪些 z_t / β_t 在当前协议下更可能）；不替代 metacontroller 决策 |
| **prediction_error owner** | online-fast | PE chain | 是（vz-cognition） | 协议自带 success/failure 信号定义；这些定义被 PE chain 消费，但 PE owner 仍是单写者 |
| **ReflectionEngine** | background-slow | 反思任务 | 是（vz-cognition.reflection） | 反思可提议协议条目修订；提议落库走 R10 ModificationGate |

**核心边界纪律**：

- ProtocolRuntime 只发布 `ActiveMixtureSnapshot`；不持有协议内容本体
- 协议内容（knowledge / case / playbook / boundary 条目）通过编译进入既有 application owners
- 其他 owner（boundary_policy / metacontroller / vitals）通过**读 ProtocolRuntime 快照**决定如何调权自己拥有的状态，而不是被 ProtocolRuntime 直写
- 这一组关系完全等价于 Domain Experience Layer 的"经验 → ETA 4 接入点"机制，只是把"静态垂直经验"扩展为"可激活、可混合、可修订的运行时配置"

## 五个机构（架构总览）

```
                  ┌──────────────────────────┐
                  │     Substrate (frozen)   │
                  └────────────┬─────────────┘
                               │ bias 全栈
              ┌────────────────┴────────────────┐
              │       Active Mixture (帧级)      │
              │  谌老师                0.70     │
              │  通用陪伴              0.20     │
              │  危机识别              0.10     │
              │  ────────────────────────────  │
              │  Frozen Boundary Union (并集)   │
              └────────────────┬────────────────┘
                               │
              ┌────────────────┴────────────────┐
              │     Activation Controller        │
              │  context match × PE utility ×   │
              │  identity gate                   │
              │  (drive coupling deferred,       │
              │   see §调度)                     │
              └────────────────┬────────────────┘
                               │
              ┌────────────────┴────────────────┐
              │       Protocol Registry          │
              │  已加载 / 休眠 / 退役协议元信息   │
              └────────────────┬────────────────┘
                               │
              ┌────────────────┴────────────────┐
              │   Identity Core  (R7 Self / R14) │
              │   恒定层；不进入混合；只做 gate    │
              └──────────────────────────────────┘
```

### 1. Identity Core（身份基座）

- **唯一性**：每个生命体一个 Identity Core；跨协议不变
- **职责**：作为 protocol activation 的 hard gate（identity-incompatible 协议直接 0 权重）
- **来源**：与 R7 dual-track 的 Self 轨道、R14 持久身份对齐；由 `vz-cognition.dual_track` 既有 owner 持有
- **新增**：本 spec 不创造新 owner，只规定 ProtocolRuntime 必须读 Self 轨道快照 + R14 regime identity 来计算 `identity_gate`

### 2. Protocol Registry（协议库）

- **唯一性**：每个 lifeform 一个 Registry
- **职责**：持有所有已加载协议的元信息；维护 lifecycle（loaded / dormant / retired）；维护 revision_log
- **持有什么**：协议**元信息**（id / version / source / activation_conditions / current_weight_history）
- **不持有什么**：协议的领域内容（已编译进 application owners）
- **新增**：是新 owner，住在 `vz-application.protocol_runtime/` 子包。**不**放 `lifeform-*` wheel：`ActiveMixtureSnapshot` 在 propagate 图里被 boundary_policy / metacontroller / vitals 等 kernel 模块读取，owner 必须可被 kernel import；而 `vz-* ↛ lifeform-*` 是 import-boundary 硬纪律（见 `tests/contracts/test_import_boundaries.py`）。`lifeform-protocol-runtime` wheel 推迟到 packet 1.5+ 引入 DocumentUptake 时再立——那时才有 LLM 调用之类的 lifeform-side 职责。

  **Wheel 迁移记录（packet 1.2）**：最初 packet 1.0 把 owner 放在 `vz-cognition.protocol_runtime/`，因为它消费的信号（PE / regime / interlocutor / Self / rupture）都来自 cognition 层。Packet 1.2 引入 boundary 编译路径后，owner 需要 import `vz-application.types.BoundaryPriorHint` + `vz-application.rare_heavy_state.ApplicationRareHeavyState`——按 tier 顺序 `vz-cognition ↛ vz-application`，所以子包整体迁移到 `vz-application.protocol_runtime/`。upper-tier owner 仍可消费 lower-tier cognition 快照（boundary_policy 也一样）。

### 3. Activation Controller（激活控制器）

- **职责**：每帧计算 `protocol_id -> activation_weight`，发布 `ActiveMixtureSnapshot`
- **算法**（详见 §调度章节）：identity_gate × softmax(α·context + β·PE_utility + γ·drive_value)
- **类比**：Norman & Shallice 的 Supervisory Attentional System
- **与 metacontroller 关系**：Activation Controller 决定**协议层**的混合权重；metacontroller 决定**协议内**的 z_t / β_t 选择。两者都属于 R2 自适应控制器，分工不同。

### 4. Active Mixture（激活混合）

- **职责**：当前帧的 weighted active set + Frozen Boundary Union；这是 ProtocolRuntime 对外发布的核心快照
- **类比**：PFC 的当前 task representation
- **R3 对应**：β_t 是切换/混合单元，z_t 是混合后的控制代码
- **下游消费者**：boundary_policy / metacontroller / vitals / strategy_playbook 都按需读这个快照调权

### 5. TaskUptake（任务吸收适配器，多入口）

- **职责**：把外部输入转成 `BehaviorProtocol` 候选，提交 Registry 加载
- **不是 owner**：是无状态适配器；产物提交后职责结束
- **不是单一模块**：每类入口住在最合适的 wheel，按 kernel boundary 切分：

| 入口类型 | 住在哪 | 何时落地 |
|---|---|---|
| **FixtureUptake**（已有 reviewed profile → BehaviorProtocol） | per-vertical：每个 `lifeform-domain-*` wheel 自带 `fixture_uptake.py` | Packet 1.0（cheng_laoshi 优先） |
| **DocumentUptake**（PDF / Markdown LLM-辅助抽取） | 新 `lifeform-protocol-runtime` wheel | Packet 1.1+ |
| **TaskDescriptionUptake**（用户口述任务） | 同上（lifeform-protocol-runtime） | Packet 1.1+ |
| **DirectoryScanUptake**（项目级批量加载） | 同上（lifeform-protocol-runtime） | Packet 1.2+ |
| **APIInjectionUptake**（外部系统提供已结构化的协议） | DLaaS platform layer 或 lifeform-protocol-runtime | 按需 |

- **审查门**：candidate 在加载前必须经 R10 ModificationGate（边界契约的 review level 比 strategy 高）
- **kernel-side `ProtocolRegistryModule.load_protocol(protocol)` 是公共入口**：所有 uptake 适配器最终都通过这一个 API 把 BehaviorProtocol 注入 Registry；`ProtocolRegistryModule` 不感知输入格式

## BehaviorProtocol Schema

落地在 `[packages/vz-contracts/src/volvence_zero/behavior_protocol.py](../../packages/vz-contracts/src/volvence_zero/behavior_protocol.py)`（跨 wheel 共享）。本节是当前实际 schema 的反映（不是初版草案 — 已根据 packet 1.0 → packet 6.0 的所有变更收紧）。

```python
@dataclass(frozen=True)
class BehaviorProtocol:
    # === 标识（required）===
    protocol_id: str                    # e.g. "growth_advisor:cheng-laoshi"
    version: str
    advisor_name: str                   # 人可读的角色 / 人物名
    description: str                    # 1-3 句话定位
    source_kind: ProtocolSourceKind     # FIXTURE / PDF_UPTAKE / MARKDOWN_UPTAKE / TASK_DESCRIPTION / API_INJECTION / DIRECTORY_SCAN
    source_locator: str                 # PDF path / API request id / fixture module path

    # === Frozen 部分（永不被 PE 修订）===
    identity_assertion: IdentityAssertion       # 该协议要求的 Identity Core 兼容性
    boundary_contracts: tuple[BoundaryContract, ...]  # 跨协议取并集；frozen
    activation_conditions: ActivationConditions       # 何种 context 下匹配；context-match 信号源

    # === Soft 部分（PE 可修订权重；ReflectionEngine 可提议替换）===
    strategy_priors: tuple[StrategyPrior, ...]
    temporal_arc: TemporalArc                  # 阶段定义由 PE 信号驱动，不是日历天数
    # NOTE: ``initial_drives`` 字段已移除（packet 1.0.1 后）；drive
    # 从 ``BehaviorProtocolSignalSource.DRIVE_HOMEOSTASIS_*`` 信号
    # 源间接表达，避免 vitals 跨层耦合。

    # === PE 信号定义（默认必填；详见 ``__post_init__`` legacy_fixture 例外）===
    success_signals: tuple[SuccessSignal, ...]
    failure_signals: tuple[FailureSignal, ...]

    # === 可选内容（默认空 tuple）===
    knowledge_seeds: tuple[KnowledgeSeed, ...] = ()         # packet 1.4a
    signature_cases: tuple[SignatureCase, ...] = ()        # packet 1.4b
    parent_protocol_id: str | None = None                  # 继承链锚点（packet 6.5 计划编译合并）
    review_status: ReviewStatus = ReviewStatus.DRAFT
    revision_log: tuple[ProtocolRevision, ...] = ()
    legacy_fixture: bool = False           # 纯 legacy fixture 可豁免 PE 信号必填


@dataclass(frozen=True)
class IdentityAssertion:
    """协议对 Identity Core 的兼容性声明。"""

    requires_self_traits: tuple[str, ...]      # 例如 "warm_peer_register" / "long_horizon"
    forbidden_self_traits: tuple[str, ...]     # 例如 "high_pressure_sales"
    required_regime_compatibility: tuple[str, ...]  # 兼容的 R14 regime 类型


@dataclass(frozen=True)
class BoundaryContract:
    """跨协议取并集；不进 utility 计算；R10 ModificationGate 高 review level。"""

    boundary_id: str
    trigger_reasons: tuple[str, ...]           # 来自 typed signal source（不能是关键词）
    blocked_topics: tuple[str, ...]
    required_disclaimers: tuple[str, ...]
    refer_out_required: bool
    severity: BoundarySeverity                 # SOFT_REMIND | HARD_BLOCK | ESCALATE_HUMAN
    review_level: ReviewLevel                  # 决定哪一级 ModificationGate 才能修改


@dataclass(frozen=True)
class ActivationConditions:
    """协议何时该被激活；信号源必须 typed，不能是用户文本关键词。"""

    context_match_signals: tuple[ContextMatchSignal, ...]  # 来自 R6 retrieval / R14 regime / R17C interlocutor_state
    co_activation_compatible: tuple[str, ...]              # 哪些其他协议可以同时激活
    co_activation_incompatible: tuple[str, ...]            # 哪些协议互斥
    minimum_weight_floor: float                            # 即使匹配低也保持的最低权重（如"危机识别"长驻 0.05）


@dataclass(frozen=True)
class TemporalArc:
    """协议时间剧本：阶段由关系状态触发，不由日历天数触发。"""

    phases: tuple[TemporalPhase, ...]
    progression_signals: tuple[ProgressionSignal, ...]    # 从 R-PE / R17C / R14 计算"现在处于哪个 phase"
    fallback_progression: TemporalArcFallback             # 当没有典型信号时的兜底（不能是日历）


@dataclass(frozen=True)
class TemporalPhase:
    phase_id: str                              # e.g. "icebreaker" / "trust_building" / "value_anchor" / "long_term_companion"
    entry_conditions: tuple[ProgressionSignal, ...]
    exit_conditions: tuple[ProgressionSignal, ...]
    expected_drives_state: tuple[DriveExpectation, ...]   # 该 phase 下 drive 应该的稳态范围


@dataclass(frozen=True)
class StrategyPrior:
    """与现有 PlaybookRule 同形；增加 PE 修订入口。"""

    rule_id: str
    problem_pattern: str
    recommended_ordering: tuple[str, ...]
    recommended_pacing: str
    avoid_patterns: tuple[str, ...]
    applicability_phase: tuple[str, ...]       # 哪些 TemporalPhase 适用（不是 string tag "day3"）

    # === PE-revisable 元信息 ===
    initial_weight: float                      # 入库初值
    pe_decay_rate: float                       # PE 失败时权重下降速率
    pe_reinforce_rate: float                   # PE 成功时权重上升速率
    minimum_weight_floor: float                # 即使持续失败也不会被 0 化（避免协议崩塌）
    revision_history: tuple[StrategyPriorRevision, ...]


@dataclass(frozen=True)
class SuccessSignal:
    """协议自带的 PE readout：成功是什么样的。"""

    signal_id: str
    description: str                           # 人可读
    measurable_via: BehaviorProtocolSignalSource
    expected_value_range: tuple[float, float] = (0.0, 1.0)
    weight_in_pe: float = 1.0


@dataclass(frozen=True)
class FailureSignal:
    """协议自带的 PE readout：失败是什么样的。"""

    signal_id: str
    description: str
    measurable_via: BehaviorProtocolSignalSource
    threshold: float = 0.0
    weight_in_pe: float = 1.0


@dataclass(frozen=True)
class ActiveMixtureSnapshot:
    """ProtocolRuntime 对外发布的核心快照（实际 schema）。"""

    active_protocols: tuple[ActiveProtocolEntry, ...]    # protocol_id + activation_weight + current_phase_id + activation_reasons
    boundary_union_ids: tuple[str, ...]                  # IDs only (Choice A 锁定，packet 1.2)
    identity_gate_traits: tuple[str, ...] = ()           # 当前 Identity Core 的 trait set
    revision_fingerprint: str = ""                       # 内容 hash 用于 fingerprint guard
    description: str = ""                                # 人可读摘要


@dataclass(frozen=True)
class ActiveProtocolEntry:
    protocol_id: str
    activation_weight: float
    current_phase_id: str | None = None                  # packet 5.0 由 ``protocol_phase`` upstream 驱动
    activation_reasons: tuple[ActivationReason, ...] = ()
```

`ActivationReasonKind` 实际成员（实际枚举，而非 spec 草案的 3 类）：

- `IDENTITY_GATE` — 身份门通过/未通过的标记（每个 entry 都带）
- `CONTEXT_MATCH` — α·context_match + β·pe_utility softmax 路径下的主标记；`detail` 字符串里携带 α / β / signals_fired / pe_utility 的具体数值
- `EQUAL_WEIGHT_FALLBACK` — 冷启动 / 全 0 信号时的 equal-weight 路径
- `PE_HISTORY` / `DRIVE_COUPLING` / `MINIMUM_FLOOR` — 占位枚举成员，packet 1.5* 后并未直接在 audit detail 中作为独立 reason kind 使用（PE 通过 `CONTEXT_MATCH` detail 表达；DRIVE 永久 deferred 见 §调度；MINIMUM_FLOOR 已并入 softmax 实现）。保留枚举成员避免 schema 破坏

### 与现有类型的兼容关系

- `BoundaryContract` 是 `BoundaryPriorHint`（vz-application）的 superset：增加了 `severity` / `review_level`
- `StrategyPrior` 是 `PlaybookRule`（vz-application）的 superset：增加了 PE-revisable 元信息
- `BehaviorProtocol.parent_protocol_id` 支持继承（e.g. `growth_advisor:cheng_laoshi` 继承 `growth_advisor:base`），编译时 merge
- `frozen=True` 的协议（`review_status=ACTIVE` 且 `pe_decay_rate=0` 全部条目）等价于今天的 `GrowthAdvisorProfile`——这是向后兼容入口

## 调度：每帧 activation_weight 的算法

### 信号源（kernel-side only）

| 信号 | 含义 | 数据源 | 时间尺度 |
|---|---|---|---|
| **context_match** | 当前对话符合协议 ActivationConditions 的程度 | R17C interlocutor_state、R14 regime、R6 retrieval、R-PE 当前 prediction | turn-time |
| **PE_utility** | 该协议在过去类似上下文里的 PE 表现 | R-PE 历史 + experience_consolidation 的 fast_prior | session-medium |
| **identity_gate** | 0 或 1：与 Identity Core 兼容否 | R7 dual_track Self 快照 + R14 regime identity | 持久 |

**`drive_value` 暂不作为 kernel-side 信号源**（packet 1.0.1 决定）：

`VitalsSnapshot` 是 lifeform-side 契约（见 `docs/DATA_CONTRACT.md` 第 33-34 / 43 / 596 行）—— **明文规定不进入 kernel §6 注册表**。`vz-cognition.ProtocolRegistryModule`（kernel 模块）若直接读 `VitalsSnapshot`，会破坏 `lifeform-* / vz-*` 的 import + runtime 边界。三种解法：

- **(a) 永久 defer**（**当前选择**）：drive coupling 退出 ActivationController；kernel-side 三信号（context_match / PE_utility / identity_gate）已经够丰富。drive 内部状态仍通过 PE 主链回流（`VitalsSnapshot.total_pe` 进 `prediction_error`，再以 PE_utility 形式重新进入协议加权）。
- **(b) future option**：若数据驱动后真需要 drive coupling，新增一个 kernel-side 的 typed `DriveReadoutSnapshot` adapter（owner 待定，需要 R8 review 决定 `vz-cognition` 还是新 wheel；adapter 只读 `VitalsSnapshot.drive_levels` 中的 readout 子集，不写）。Spec 在 (b) 实施前不做承诺。
- (c) 把 ActivationController 拆成 kernel + lifeform 两部分——结构性改动太大，不在本协议范围。

合成公式因此简化为 3 因子：

```
identity_gate_i = identity_compatible(protocol_i, identity_core_snapshot) ∈ {0, 1}

raw_score_i = α · context_match_i
            + β · pe_utility_i

floor_i = max(protocol_i.activation_conditions.minimum_weight_floor, 0)

base_weight_i = identity_gate_i × max(softmax(raw_score)_i, floor_i)

# 互斥协议联合归一化：incompatible 的协议共享一个权重池
weight_i = enforce_co_activation_constraints(base_weight, registry)
```

**关键纪律**：

- `α / β` 是 metacontroller 学的（不写死）；初值由 protocol 的 `pe_decay_rate / pe_reinforce_rate` 暗示
- `identity_gate` 是硬过滤；不通过则全协议 0 权重，无论其他信号多强
- `minimum_weight_floor` 让"危机识别"这种长驻协议永远保持 ≥ 0.05 的警戒激活
- `softmax` 而不是 `argmax`：默认是混合而不是单选；只有当某个协议 dominate（如 0.95+）时才接近 hard switch

**Packet 1.5a + 1.5a' 实现**（context_match — 6 个 kernel-side detector）：

- `context_match_i = Σ signal.weight × signal_is_firing(signal, upstream)`：
  - `INTERLOCUTOR_ZONE_TRANSITION`（1.5a）：interlocutor_state 任一 zone bool 为 True 时 fire
  - `RUPTURE_KIND_FIRED`（1.5a）：rupture_state 解析出非空 `rupture_kind` 时 fire
  - `BOUNDARY_VIOLATION_FIRED`（1.5a）：boundary_policy 决策含非空 `trigger_reasons` 时 fire
  - `USER_DROPOUT_OBSERVED`（1.5a'）：rupture_state.rupture_kind == `ABANDONED` 时 fire（与 `EXTERNAL_OUTCOME_TO_RUPTURE_KIND[ABANDONED]` 同源）
  - `REGIME_TRANSITION_RECENT`（1.5a'）：regime.turns_in_current_regime ≤ 1 时 fire（cold-start 容忍）
  - `RETRIEVAL_HITS_PRESENT`（1.5a'）：retrieval_policy.knowledge_domains 非空时 fire
- `DRIVE_HOMEOSTASIS_HOLD/BREACH` 永久 defer（vitals 跨层边界，packet 1.0.1）；保留枚举成员，detector 返 False，protocols 仍可声明但 kernel-side 不贡献

**Packet 1.5b 实现**（pe_utility / β term）：

- α=1, β=1 硬定（α/β 学习仍留 packet 1.5c）
- `pe_utility_i` 由 `ProtocolRegistryModule` 内部维护：
  - 每 turn 读 `prediction_error.error.signed_reward ∈ [-1, 1]`（kernel ACTIVE 上游）
  - 把上一 turn 已发布的 `active_mixture` 每个 protocol 的 weight 缓存为 `_last_active_weights`
  - 当本 turn 的 PE 到达，按 `Δ_i = signed_reward × last_weight_i` 把奖励 attribute 给上一 turn 的 active 协议
  - EMA 更新：`pe_utility_i ← (1-η) · pe_utility_i + η · Δ_i`，η = 0.25
  - 未参与上一 turn 的协议自然按 `Δ_i = 0` 衰减；`pe_utility_i` 钳位在 `[-1, +1]`
- bootstrap PE（`PredictionErrorSnapshot.bootstrap=True`）跳过 attribution（占位 PE 不应污染 EMA）
- 同 `turn_index` 重复 PE（replay / retry）通过 `_last_pe_turn_index` 去重
- `pe_utility_by_id: Mapping[str, float]` 由 owner 传给 `compute_active_mixture`；测试或外部直调可不传，缺省全 0 → 1.5a 行为

**调度逻辑（1.5c-iii 后整合）**：

- `raw_score_i = α · context_match_i + β · pe_utility_i`，α/β 由 owner 在线维护（不再硬定）
- `has_signal = max(context_match) > 0 OR max|pe_utility| > 0`
- `has_signal` → `softmax(raw_score)` + `minimum_weight_floor` enforcement，`ActivationReason.kind = CONTEXT_MATCH`，`detail` 携带 `α=x.xxx, β=y.yyy` + `signals_fired=[...]` 与 `pe_utility=±0.xxx`
- `not has_signal`（cheng_laoshi 默认 + cold start）→ `equal_weight_with_floor`，`kind = EQUAL_WEIGHT_FALLBACK`，`detail` 仍打印当前 α/β 用于审计

**Packet 1.5c-iii 实现**（α/β 在线学习）：

- α / β 不再硬定，是 `ProtocolRegistryModule` 内部 owner-side state，初值 1.0
- 每 PE turn 在 `_update_pe_history` 之后调用 `_update_alpha_beta`：
  - 计算 `cm_range = max(_last_context_scores.values) - min(...)`，对所有上一 turn eligible 的协议
  - 同样的 `pe_range = max(_last_pe_utilities.values) - min(...)`
  - REINFORCE-style 代理梯度：`α_grad = signed_reward · cm_range`，`β_grad = signed_reward · pe_range`
  - 钳位更新：`α ← clamp(α + η_meta · α_grad, [0.1, 5.0])`，η_meta = 0.05
  - 直觉：当某个信号在上一 turn 把协议区分得清楚（range 大）且结果好（signed_reward > 0），它就是有用的信号 → 提高它的系数
- skip 条件：cold start（cache 空）/ bootstrap PE / 重复 turn_index / 单协议 mixture（`len(_last_context_scores) < 2`，cheng_laoshi 默认场景）/ `cm_range == 0 AND pe_range == 0`（无差异化信号）
- `compute_active_mixture` 接 `alpha=1.0, beta=1.0` 默认参数 + `audit_context_scores: dict | None = None` 输出参数。owner 通过 audit_out 抓取本 turn 的 context_match 用于下 turn 学习；外部 / 测试直调可全部省略，向后兼容
- η_meta = 0.05 比 pe_utility EMA 的 η = 0.25 慢 5 倍：α/β 是全局 hyperparameter（影响每个决策的形状），慢更新避免抖动；pe_utility 是协议级局部状态，可以更敏感

**Packet 1.5c-i 实现**（PE-driven 互斥仲裁）：

- `_resolve_co_activation_incompatibility` 在 softmax 之前先做硬过滤：每条 `co_activation_incompatible` 声明产生一对 A↔B 决策，drop 谁取决于 `pe_utility`：
  1. `pe_utility(A) > pe_utility(B)` → drop B
  2. `pe_utility(A) < pe_utility(B)` → drop A
  3. 相等（含 cold-start 0/0）→ lex 兑现（保留较小 `protocol_id`）
- 删除路径里若 self 被 drop，停止枚举 self 的剩余 incompatible 声明（self 已经不存在了）
- `pe_utility_by_id` 是同一份 owner-side EMA dict，与 softmax 的 β 项共享数据源 — 互斥仲裁与软混合权重一致地反映"协议过往表现"，不会出现"PE 高的协议先被 lex 干掉再说"的悖论

### 不允许的优先级方式

- ❌ 写死 priority list：`["谌老师", "通用陪伴", "危机识别"]`
- ❌ 关键词触发：`if "焦虑" in user_text: activate("crisis_mode")`
- ❌ 外部计数器打 tag：`if session.day == 3: load("day3_protocol")`
- ❌ 表达层 LLM 决定：让 LLM "选一个最合适的角色" — 这是把控制层下放到 token 空间，违反 R4

### 允许的来源

- ✅ Activation Controller 学到的 `α/β/γ` 权重
- ✅ 从 PE 历史压缩的 `experience_fast_prior`
- ✅ R17C interlocutor_state 的 zone bool
- ✅ R14 regime identity embedding
- ✅ R-PE 当前 prediction 输出

## 4 类冲突分层处理

冲突有**性质上不同**的 4 种类型，处理机制必须分层。错配处理机制是常见的架构 bug 来源。

### 类型 A：边界契约冲突（不可妥协）

**场景**：协议 P1（"谌老师陪伴"）`bp-no-hard-sell`；协议 P2（"销售冲业绩"）要求"立刻问下单"。

**处理**：
- 边界契约**永远跨协议生效**，与 activation_weight 无关
- 所有 active protocol（含 weight ≥ floor 的）的 boundary_contracts 取**并集**
- 任一条 fire 即 block
- 不进入 utility 计算，不被混合调和

**认知科学对应**：morality / value commitments，超越 cost-benefit。

**实现位置**：BoundaryPolicy（既有 owner）读 `ActiveMixtureSnapshot.boundary_union` 决定本帧执行哪些边界。

**已有基础**：`bp-no-hard-sell.trigger_reasons=first_week_relationship_age` 已经把"关系年龄"做成 typed signal；扩展到协议层只是把 owner 集合从单 profile 扩展到 active mixture 的 union。

### 类型 B：策略偏好冲突（可调和）

**场景**：协议 P1 说"先问年龄"；协议 P2 说"先共情"。

**处理**：
- 这是**常态**情况，不算"冲突"
- 按 active mixture 权重做 soft blending
- z_t 控制器代码空间合成一个折中行为
- metacontroller 在 z_t 空间内学最佳混合

**认知科学对应**：Drift-Diffusion 累积证据；Bayesian Model Averaging。

**实现位置**：metacontroller（既有 owner）读 `ActiveMixtureSnapshot.active_protocols[*].activation_weight + StrategyPrior.recommended_ordering`，在 z_t 空间合成。

**关键纪律**：合成发生在控制器层（z_t 空间），不在表达层（token 空间）。R4 对应。

### 类型 C：时间剧本冲突（β_t 切换）

**场景**：协议同时触发"icebreaker phase"和"value_anchor phase"。

**处理**：
- 由 β_t 切换单元做选择
- 触发上层切换决策（不是混合）
- 选择本身**是 PE 学的**：历史上当用户处于 X 状态时，β_t = phase_A 比 β_t = phase_B 的 PE 更低 → 学到偏好 phase_A

**认知科学对应**：Set-shifting / cognitive flexibility（Wisconsin Card Sort 类任务）；Norman & Shallice 的 "schema selection at the action level"。

**实现位置**：`vz-temporal.metacontroller` 的 β_t 已是一等学习对象（见 §3 temporal-abstraction.md）；本 spec 要求 β_t 的输入特征中**加入**当前 ActiveMixtureSnapshot 的 phase 候选集合。

**关键纪律**：不要用日历天数硬切（当前 `growth_advisor:day3` 这种）；要用 PE 信号驱动的关系阶段（`TemporalArc.progression_signals`）。

### 类型 D：身份级冲突（升级反思）

**场景**：当前活跃协议要求做的事，与 Identity Core 不兼容（"谌老师协议被混入了催单话术，但我是温柔陪伴师"）。

**处理**：
- 触发 R-PE 高分（identity_violation 是高 weight 的 failure_signal）
- 写入 episodic（R5）等待反思
- 不在当前帧解决
- background-slow 反思整合后，可能产生：
  - 协议条目权重大幅下降
  - 协议被标记为"identity-incompatible"，下次 activation_gate 直接 0
  - 升级到人工 review 触发协议退役

**认知科学对应**：cognitive dissonance（Festinger 1957）；conscience / superego（Freud）。

**实现位置**：
- 当帧：identity_gate 已经是硬过滤（应该不会冲突）；如果冲突 fire 说明 identity_gate 计算有 bug 或协议 IdentityAssertion 声明错误
- 慢层：R6 ReflectionEngine 周期性扫描 identity_violation PE 高分事件，提议协议条目修订或退役

**关键纪律**：身份冲突**绝不**当场用启发式处理（如"立刻切换协议"），必须走慢层反思 + R10 ModificationGate。当场只做安全降级（边界契约的 ESCALATE_HUMAN severity）。

## TaskUptake：协议入口适配器

TaskUptake 不是一个 owner，而是一组**无状态适配器**，把外部输入转成 `BehaviorProtocol` 候选并提交 Registry。

### 入口类型

| 入口 | 输入 | 适配器 | 产出 |
|---|---|---|---|
| **DocumentUptake** | PDF / Markdown / 长文本 | LLM-辅助抽取（chunked，结构化 JSON） | `BehaviorProtocolCandidate` + `RequiresReview` 字段 |
| **TaskDescriptionUptake** | 用户口述任务（"我要你做私域客服，前 7 天不能推销"） | LLM 解析 + 模板填充 | 简化版 `BehaviorProtocolCandidate` |
| **DirectoryScanUptake** | 项目级文档目录 | 批量 DocumentUptake | 多个 candidate |
| **APIInjectionUptake** | 外部系统提供的结构化协议 | schema 校验 | `BehaviorProtocolCandidate`（通常已 reviewed） |
| **FixtureUptake** | 现有手写 profile（向后兼容） | profile → protocol 转换 | `BehaviorProtocol`（review_status=ACTIVE，pe_decay_rate=0） |

### LLM-辅助抽取的纪律

参考既有 `lifeform_domain_character.extraction.profile_llm`：

1. **LLM 不是 owner**：LLM 只产 candidate；reviewer 决定是否落库（R8 / R10）
2. **Per-chunk extraction**：长文档分块抽取，每块产一个局部 candidate，最后 aggregate
3. **`requires_review` 字段必填**：低置信度字段必须 surface，reviewer 必须逐条处理
4. **provenance_chunks**：candidate 必须能反查到原文哪一段；不能"凭空生成"
5. **不接管 owner 决策**：LLM 抽出的 boundary_contracts 在 review 之前不进 active union

### Review 分级（R10 ModificationGate）

| 协议字段 | Review 级别 | 谁能批准 |
|---|---|---|
| `identity_assertion` | L4（最高） | 人工 review + 系统 admin |
| `boundary_contracts` | L3 | 人工 review |
| `success_signals` / `failure_signals` | L3 | 人工 review |
| `strategy_priors` | L2 | 自动 lint + 人工 spot-check |
| `temporal_arc` | L2 | 自动 lint + 人工 spot-check |
| `initial_drives` | L2 | 自动 lint |
| `revision_log` 写入 | L1 | 自动（PE 验证写入） |

### 协议生命周期状态机

```
[Candidate]                                  ← TaskUptake 适配器输出
    │
    │ R10 ModificationGate review
    ▼
[Draft]                                      ← review_status=DRAFT
    │
    │ 加载到 Registry，但不进 active mixture
    ▼
[Shadow]                                     ← review_status=SHADOW
    │     ↓ 影子运行：计算 activation_weight 但不执行行为
    │     ↓ PE 数据收集
    │
    │ Shadow 期 PE 通过 evaluation gate
    ▼
[Active]                                     ← review_status=ACTIVE
    │     ↓ 进入 active mixture，影响行为
    │     ↓ PE 持续验证；ReflectionEngine 可提议修订
    │
    │ 持续失败 / identity violation / 人工撤回
    ▼
[Retired]                                    ← review_status=RETIRED
    │     ↓ 不再激活，但 revision_log 保留供 R15 回滚
```

## 协议 → PE 映射规约

这是元架构里 R-PE 的硬要求：**协议必须自带可测的 success / failure 信号定义**，否则 PE / reflection 无东西可学。

PDF 里大量"她们想要的 / 怕的"是描述性的，不可直接做 PE 信号。TaskUptake 必须**强制翻译**为可测信号。

### 信号源清单（实际 enum 成员）

`BehaviorProtocolSignalSource` 当前 8 个成员（packet 1.0 的 6 个 + 1.5a' 的 2 个）：

| SignalSource | 含义 | 上游 / detector | Detector 状态 |
|---|---|---|---|
| `INTERLOCUTOR_ZONE_TRANSITION` | 12 轴 zone 任一为 True | `interlocutor_state` | ✅ packet 1.5a |
| `RUPTURE_KIND_FIRED` | rupture_state 解析出非空 rupture_kind | `rupture_state` | ✅ packet 1.5a |
| `BOUNDARY_VIOLATION_FIRED` | boundary_policy 决策含非空 trigger_reasons | `boundary_policy` | ✅ packet 1.5a |
| `USER_DROPOUT_OBSERVED` | rupture_state.rupture_kind == ABANDONED | `rupture_state` | ✅ packet 1.5a' |
| `REGIME_TRANSITION_RECENT` | regime.turns_in_current_regime ≤ 1 | `regime` | ✅ packet 1.5a' |
| `RETRIEVAL_HITS_PRESENT` | retrieval_policy.knowledge_domains 非空 | `retrieval_policy` | ✅ packet 1.5a' |
| `DRIVE_HOMEOSTASIS_HOLD` | drive 在稳态带内 | vitals (lifeform-side) | ⏳ 永久 deferred（packet 1.0.1 vitals 跨层决议） |
| `DRIVE_HOMEOSTASIS_BREACH` | drive 出稳态带 | vitals (lifeform-side) | ⏳ 永久 deferred |

Packet 7.0 计划补 5 个：`USER_REPLY_LATENCY` / `USER_REPLY_LENGTH` / `USER_INITIATIVE_QUESTION`（来自 `dialogue_trace`）+ `COMMITMENT_FULFILLED` / `COMMITMENT_BROKEN`（来自 §14 aac-commitment-lifecycle）。

### PDF 信号翻译示例（以东方测评私域运营规划为例）

| PDF 自然语言 | 翻译后的 PE 信号 |
|---|---|
| "前 7 天信任关键期，只输出价值不推销" | success: `phase=icebreaker..value_anchor` 期间无 `BOUNDARY_VIOLATION_FIRED(bp-no-hard-sell)` 且 phase=long_term_companion 时 `USER_INITIATIVE_QUESTION(product_topic) ≥ 1`<br>failure: 期间 `USER_DROPOUT > 5 days` |
| "信数据，不信玄学" | success: `dialogue_trace.user_reference_to_data ≥ 1`<br>failure: rupture_state 触发 `disengagement_kind` |
| "深层诉求：缓解焦虑" | success: `INTERLOCUTOR_ZONE_TRANSITION(anxiety_high → anxiety_normal)` |
| "怕踩坑、怕智商税" | success: `USER_INITIATIVE_QUESTION(label_photo_share) ≥ 1`<br>failure: `dialogue_trace.user_skeptical_phrasing ≥ threshold` |

### 强制要求

- 协议加载前，TaskUptake 必须验证 `len(success_signals) ≥ 1` 且 `len(failure_signals) ≥ 1`
- 每个 signal 的 `measurable_via` 必须指向 typed SignalSource，不能是自由文本描述
- 缺 PE 映射的协议直接拒绝加载，进入 `requires_review` 待补
- 这条纪律保护元架构：**没有 PE 入口的协议不允许进入运行时**

## 反思修订协议条目

### 修订路径

```
[ActiveMixtureSnapshot 持续运行]
    │
    │ PE 信号写入 prediction_error owner
    │
    ▼
[R5 episodic stratum 累积]
    │
    │ 周期性触发（不阻塞 turn）
    │
    ▼
[R6 ReflectionEngine 扫描]
    │ - 哪些 strategy_prior 持续 PE 失败？
    │ - 哪些 success_signal 持续不命中？
    │ - 是否出现协议未覆盖的成功 case？
    │
    ▼
[ReflectionEngine 产出 ProtocolRevisionProposal]
    │
    │ R10 ModificationGate
    │
    ▼
[ProtocolRegistry 应用修订]
    │ - 写入 revision_log
    │ - 更新 strategy_prior.initial_weight 或新增 strategy_prior
    │ - SHADOW 影子运行新版 → ACTIVE
```

### 修订类型（`ProtocolRevisionChangeKind`）

实际枚举（5 类，packet 5.2 后）+ 4 类待补（packet 6.1 计划）：

| ChangeKind | 含义 | 触发规则 / 调用方 | Review 级别 | 状态 |
|---|---|---|---|---|
| `WEIGHT_DECAY` | 把 strategy_prior 的 initial_weight × multiplier (默认 0.5) | `propose_strategy_decay` (PE 平均 < -0.3 持续 ≥ 5 turn) | L3 | ✅ packet 3.2 |
| `DEACTIVATE` | 把 strategy_prior 的 initial_weight 强制为 0 | 严重失败的 strategy（手动 / 严重 PE） | L3 | ✅ packet 3.3 |
| `REPLACE_TEXT` | 替换 strategy_prior 的描述性文本字段 | （仅 schema，未来 LLM 反思可用） | L3 | schema only |
| `ARCHIVE` | 从 knowledge_seeds / signature_cases 移除条目 | `propose_knowledge_archival` / `propose_case_archival` | L1-L2 | ✅ schema + apply；rule 占位（packet 6.2 真实化） |
| `ADD_STRATEGY` | 把新 StrategyPrior append 到协议 | `propose_strategy_addition` (4 turn 成功 + 协议 weight 低) | L3 | ✅ packet 5.2 |
| `WEIGHT_REINFORCE` | 把 strategy_prior 的 initial_weight × multiplier > 1 | `propose_strategy_reinforce`（持续正 PE） | L1（自动） | ⏳ packet 6.1 |
| `BOUNDARY_REFINEMENT` | 修改 BoundaryContract 字段（trigger_reasons / blocked_topics） | 边界误报率高 | L4 | ⏳ packet 6.1 |
| `IDENTITY_CLARIFICATION` | 修改 identity_assertion | identity_violation PE 反复出现 | L4 | ⏳ packet 6.1 |
| `PROTOCOL_RETIREMENT` | 把 protocol.review_status 标 RETIRED | `propose_protocol_retirement`（协议整体 PE 持续负） | L3-L4 | ⏳ packet 6.1（依赖 6.4 RETIRED 状态） |

### 关键纪律

- 反思**不直接改 strategy_priors 内容**；只产 proposal
- proposal 必须带 evidence（哪几个 PE 事件支持）
- 修订写入 revision_log；revision_log 是 append-only（R15 可回滚）
- WeightDecay 不能把权重压到 `minimum_weight_floor` 以下（防止协议崩塌）
- 协议条目变化超过阈值 → 必须 SHADOW 影子运行后才 ACTIVE

## 接口契约

### ProtocolRuntime 发布

- `ActiveMixtureSnapshot`（每 turn 发布）
- `ProtocolRegistrySnapshot`（生命周期事件时发布）
- `ProtocolRevisionLog`（修订事件时发布）

### ProtocolRuntime 消费（kernel-side 上游 only）

| 来源 | 用途 | Owner wheel |
|---|---|---|
| `dual_track` Self snapshot | identity_gate 计算 | vz-cognition |
| `regime` snapshot (R14 `RegimeIdentity`) | identity_gate + context_match | vz-cognition |
| `interlocutor_state` snapshot | context_match 信号 | vz-cognition |
| `prediction_error` snapshot | PE_utility + reflection 输入 | vz-cognition |
| `retrieval_policy` snapshot | context_match 信号 | vz-application |
| `rupture_state` snapshot | identity 冲突预警 | vz-cognition |

**`vitals` 不在此表**：`VitalsSnapshot` 是 lifeform-side 契约，kernel `ProtocolRegistryModule` 不能直接消费。`drive_value` 信号源 deferred；详见 §调度。

### 消费 ActiveMixtureSnapshot 的 owner

| Owner | 读什么 | 怎么用 |
|---|---|---|
| `boundary_policy` | `boundary_union` | 本帧执行哪些边界 |
| `metacontroller` | `active_protocols[*].activation_weight + StrategyPrior.recommended_ordering` | z_t / β_t 选择的 prior |
| `vitals` | `active_protocols[*].current_phase_id` | 各 drive 的 expected_homeostatic_band 调整 |
| `strategy_playbook` | `active_protocols` | 哪些 PlaybookRule 在本帧加权 |
| `case_memory` | `active_protocols` | 哪些 case 在本帧检索时加权 |
| `domain_knowledge` | `active_protocols` | retrieval mix 的 domain_weight |

**关键纪律**：消费者**只读**快照，**不写**回 ProtocolRuntime；ProtocolRuntime 不被任何下游 owner 写入（R8 单写者）。

## ProtocolRuntime 与 application owners 的内容边界（R8 SSOT 收紧）

`BehaviorProtocol` 内部带 `boundary_contracts` / `strategy_priors` / `temporal_arc`，**外观上**像 boundary / strategy / case 内容的容器。这一节明确：**ProtocolRuntime 不是这些内容的 canonical owner**；不收紧会导致 packet 1.2 boundary_policy 接入时形成第二 owner（违反 R8）。

### 三条硬纪律

1. **canonical store 不变**：boundary 内容 canonical store 仍是 `boundary_policy`；strategy 是 `strategy_playbook`；case 是 `case_memory`；knowledge 是 `domain_knowledge`。这些 owner 是 R8 单写者，packet 1.2+ 任何 consumer 必须从这些 owner 读 canonical content。
2. **ActiveMixtureSnapshot 发布的是配置层**：协议 IDs + activation_weight + reviewed-hint metadata（而不是 boundary / strategy 内容本体）。consumer 拿到 ID 后回到对应 owner 读内容；ID 不存在 = 加载失败（fail-loud）。
3. **`BehaviorProtocol` → application owners 的 compile 路径**（packet 1.2+）：与 `DomainExperiencePackage.compile_to_application_owners()` 同形——`ProtocolRegistryModule.load_protocol(bp)` 触发：
   - `bp.boundary_contracts` 编译为 `BoundaryPriorHint`，注入 `boundary_policy` owner（标记 `provenance=protocol:<protocol_id>` 便于 unload 时追溯）
   - `bp.strategy_priors` 编译为 `PlaybookRule`，注入 `strategy_playbook` owner
   - 信息流向：**协议作为来源 → application owner 作为执行者**，单向；application owner 不写回 protocol

### `ActiveMixtureSnapshot.boundary_union_ids` 字段的定位（packet 1.2 选择 A 已锁定）

最初 packet 1.0 SHADOW 的字段名是 `boundary_union: tuple[BoundaryContract, ...]`，直接发布完整 `BoundaryContract` tuple——这是过渡态。Packet 1.2 锁定 **选择 A**：

- 字段重命名为 `boundary_union_ids: tuple[str, ...]`，仅发布**boundary IDs**，不发布 contract 内容本体
- 消费者拿到 ID 集后回到 `boundary_policy` / `ApplicationRareHeavyState.boundary_prior_hints` 读 canonical 内容（compile 路径已把 protocol-driven 的 `BoundaryPriorHint` 推到 application state）
- 任何 ID 在 application state 找不到 = fail-loud（不允许 ProtocolRuntime 自己持有 fallback 内容）

未取选择 B（"保留完整 contracts + docstring 约束"）的原因：依赖人 review，长期会被绕过。选择 A 是物理隔离。

### `BoundaryPriorHint` 不可表达的 protocol 元字段

`BoundaryContract` 比 `BoundaryPriorHint` 多两个字段（`severity` / `review_level`）。这两个是**协议层元数据**：
- `severity` 用于下游表达层渲染（"硬阻断" vs "软提醒" vs "升级人工"）
- `review_level` 用于 `ModificationGate`（R10）门控修订权限

它们**不**进入 `BoundaryPriorHint`（`vz-application` 的 schema 不变）。如果未来下游 boundary 执行需要这两个字段，要么扩展 `BoundaryPriorHint` schema（vz-application 单独 PR），要么在 `ProtocolRegistryModule` 发布一个独立的 `protocol_boundary_metadata` slot。Packet 1.2 不做任何一种。

### Compile 路径状态（packet 1.2 已落地）

| 协议字段 | 编译目标 | 落地 packet | 状态 |
|---|---|---|---|
| `boundary_contracts` | `BoundaryPriorHint` → `ApplicationRareHeavyState.upsert_boundary_prior_hints` | 1.2 | ✅ 已落地 |
| `strategy_priors` | `PlaybookRule` → `ApplicationRareHeavyState.upsert_distilled_playbook_rules` | 1.3b | ✅ 已落地 |
| `knowledge_seeds` | `DomainKnowledgeRecord` → `ApplicationDomainKnowledgeStore.upsert_records` | 1.4a | ✅ 已落地 |
| `signature_cases` | `CaseMemoryRecord` → `ApplicationCaseMemoryStore.upsert_records` | 1.4b | ✅ 已落地 |
| `temporal_arc.phases` | metacontroller `β_t` 切换信号（不进 application owner） | 1.4+ | ⏳ |

包加载入口：`ProtocolRegistryModule(application_rare_heavy_state=...)` 注入 state；`load_protocol(bp)` 内部调用 `compile_protocol_to_application_artifacts(bp)` 后 upsert 到既有 application 接口。**没有第二 owner**：所有 application 内容仍由 `boundary_policy` / `strategy_playbook` / `case_memory` / `domain_knowledge` 拥有；`hint_id` 通过命名空间前缀 `protocol:{protocol_id}:boundary:{boundary_id}` 携带 lineage。

### Unload 路径（packet 1.2 deferred）

`ApplicationRareHeavyState` 当前没有 per-key remove API。已被 apply 过的协议执行 `unload_protocol` 会 `raise NotImplementedError`（fail-loud）。完整 unload 路径需要：
- 扩展 `ApplicationRareHeavyState` 加 `remove_boundary_prior_hints_by_id_prefix(prefix)` 等 API
- 或者从 checkpoint 重建（不带该协议条目）

Packet 1.6+（反思修订）会用到这条路径，届时一并实现。

### 同样的纪律对其他 application owners 也成立

- `strategy_priors` → `strategy_playbook` canonical
- `case_memory_records`（如未来加入 BehaviorProtocol）→ `case_memory` canonical
- `domain_knowledge_seeds`（如未来加入）→ `domain_knowledge` canonical
- `temporal_arc.phases` → 不与现有 owner 重叠（R3 `temporal_abstraction` 是控制器层产物，不是 reviewed prior 容器），所以 phases 可以原样发布

## 与 cheng_laoshi.py / GrowthAdvisorProfile 的迁移关系

`cheng_laoshi.py` 不删除，但**降级**为 fixture：

### 迁移阶段

**阶段 0（design freeze，本 spec）**：
- 写本 spec
- 不动 `cheng_laoshi.py` 任何代码

**阶段 1（SHADOW）**——按 packet 切分推进：
- **Packet 1.0**（最小 SHADOW，已落地）：
  - 在 `vz-contracts` 加 `BehaviorProtocol` schema + `ActiveMixtureSnapshot` + `BehaviorProtocolSignalSource` 闭合枚举
  - 在 `vz-cognition` 加 `protocol_runtime/` 子包（`ProtocolRegistryModule` SHADOW owner + `ActivationController` 极简版 + 内部 registry store）；packet 1.2 因 tier 限制迁移至 `vz-application`
  - 在 `lifeform-domain-growth-advisor` 加 `fixture_uptake.py`：`growth_advisor_profile_to_behavior_protocol(profile)` per-vertical 适配器（不是新 wheel；FixtureUptake 永远是 per-vertical 的 helper）
  - PE 信号从 `drive_priors` 合成：每条 drive → 1 SuccessSignal (`DRIVE_HOMEOSTASIS_HOLD`) + 1 FailureSignal (`DRIVE_HOMEOSTASIS_BREACH`)
  - `vz-runtime/final_wiring.py` 注册 `ProtocolRegistryModule` SHADOW；下游 consumer 一行不改
- **Packet 1.0.1**（consolidation，已落地）：
  - DATA_CONTRACT.md §6 加 `active_mixture` slot 行 + 契约语义段
  - drive_value 信号源 deferred（vitals 跨层边界），formula 改为 3 因子
  - 新增 §ProtocolRuntime 与 application owners 的内容边界 + §SHADOW → ACTIVE 升级 checklist
  - Runtime fail-loud guard（`FallbackActivationActiveError`）锁住 ACTIVE 升级
- **Packet 1.2**（boundary 编译路径 + Choice A，已落地）：
  - `BoundaryContract` 加 3 字段（`regime_id` / `answer_depth_limit_hint` / `clarification_required`）让 protocol → BoundaryPriorHint 转换无损
  - `ActiveMixtureSnapshot` 字段重命名 `boundary_union` → `boundary_union_ids: tuple[str, ...]`（Choice A 锁定）
  - 新增 `vz-application.protocol_runtime.compiler.compile_protocol_to_application_artifacts(protocol)` + `ProtocolApplicationArtifacts`（owner 子包从 vz-cognition 迁到 vz-application，详见决策 0）

  - 决策 0（wheel 迁移）：把 packet 1.0 在 `vz-cognition.protocol_runtime/` 立的 owner 子包整体移到 `vz-application.protocol_runtime/`。原因：`compile_protocol_to_application_artifacts` 必须 import `vz-application.types.BoundaryPriorHint` + `vz-application.rare_heavy_state.ApplicationRareHeavyState`，但 tier 顺序 `vz-cognition ↛ vz-application`。owner 的输入信号仍是 cognitive（PE / regime / interlocutor / Self / rupture），但产物是 application 内容；所以在 application 层托管更合理（boundary_policy 也是 application 层）。public import path `volvence_zero.protocol_runtime` 不变（PEP 420 namespace package），下游 import 不受影响
  - `ProtocolRegistryModule` 接受 `application_rare_heavy_state` 注入；`load_protocol` 自动 compile + upsert；`unload_protocol` 对已应用协议 fail-loud（packet 1.6+ 真正实现）
  - `final_wiring.py` 把 `application_rare_heavy_state` 传给 `ProtocolRegistryModule`
  - Hint id 命名空间：`protocol:{protocol_id}:boundary:{boundary_id}` 携带 lineage
- **Packet 1.2 后续**（已落地）：`tests/contracts/test_protocol_boundary_matched_control.py` 7 个测试。验证 protocol compile 与 vertical compile 产出的 `BoundaryPriorHint` 除 `hint_id` lineage 前缀外 byte-equivalent；这间接证明 `boundary_policy.process()` 在两条路径下行为完全一致（boundary_policy 只读 `regime_id` / `trigger_reasons` filter + 内容字段，不分支于 `hint_id`）。同时验证 vertical + protocol 同时启用时 state merge 不破坏（同 `(regime_id, trigger_reasons)` key 折叠到单条 entry）
- **Packet 1.3a**（identity gate regime 分支真实化，已落地）：
  - `ProtocolRegistryModule.dependencies = ("dual_track", "regime")` —— 接入 R7 dual-track Self 与 R14 regime 上游
  - `_compute_identity_gate(protocol, dual_track_snapshot, regime_snapshot)` 实现真实 regime compatibility 检查：`required_regime_compatibility` 非空 ∧ `active_regime.regime_id` 不在集合 → gate=0（filter out）
  - self_traits 分支 permissive 占位（`DualTrackSnapshot` 没有 trait 词汇；filter 留给 packet 1.3'）
  - `compute_active_mixture` 用 gate=0 硬过滤掉协议（不进入 active_protocols / boundary_union_ids）
  - SHADOW permissive：regime upstream 缺失时不阻塞（ACTIVE 升级仍由 fallback flag + checklist 守门）
  - cheng_laoshi `required_regime_compatibility=()` → 行为完全不变（regression baseline 保住）
- **Packet 1.3'**（self_traits 真实化机器身，已落地）：
  - `vz-cognition.dual_track.TrackState` 加 `traits: tuple[str, ...] = ()` 字段（默认空，向后兼容；`derive_track_state` 不 populate，所有现有路径不变）
  - `_compute_identity_gate` self_traits 分支重写：当 `self_track.traits` 非空时执行真实 subset / forbidden 检查；missing required → gate=0；forbidden present → gate=0；空 traits 时 `self_traits_populator_pending` SHADOW-permissive 通过
  - dual_track snapshot 整体缺失时 `self_traits_dual_track_absent_shadow_pass` 通过
  - 7 个新合成 traits 测试覆盖：required subset 通过 / required missing 过滤 / forbidden absent 通过 / forbidden present 过滤 / required+forbidden 联合 / 空 traits populator pending / e2e via compute_active_mixture
  - **populator 推迟**：production 代码现在不向 `self_track.traits` 写入；当未来 packet 加 populator（从 R6 reflection / persona seeds / semantic owners 派生 traits）时，identity gate 自动激活真实检查，无需协议侧改动
- **Packet 1.3''**（self_traits populator，已落地）：
  - 新 `vz-contracts/identity_seed.py`：`IdentitySeed(traits, description)` frozen dataclass，跨 wheel 共享（vz-cognition + lifeform-domain-* 都用）
  - `DualTrackModule.__init__` 接受 `identity_seed: IdentitySeed | None = None` kwarg；`derive_track_state` 在 SELF 轨道把 `seed.traits` 写入 `TrackState.traits`（WORLD 轨道始终空）
  - `run_final_wiring_turn` + `build_final_runtime_modules` 透传 `identity_seed` kwarg 到 `DualTrackModule`
  - 新 `lifeform-domain-growth-advisor/identity_seed.py`：`build_growth_advisor_identity_seed(profile)` 返回 `IdentitySeed(traits=("warm_peer_register", "long_horizon"))`
  - 12 个新 dual_track 测试 + 2 个新 protocol e2e 测试（1 个 cheng_laoshi 通过 + 1 个 hostile 协议被过滤）
  - **关闭 condition 1 完整闭环**：cheng_laoshi 通过 `run_final_wiring_turn` e2e 真实激活 self_trait 过滤；伪造 `requires_self_traits=("aggressive_sales",)` 协议会被识别并过滤掉
- **Packet 1.3'''**（production wiring，已落地）：
  - `LifeformConfig.identity_seed` 字段 + `with_identity_seed(seed)` 方法
  - `Lifeform.__init__` 把 `self._config.identity_seed` 透传给 `Brain(identity_seed=...)`
  - `Brain.__init__` + `Brain.with_identity_seed` 镜像 `regime_bootstrap` pattern；`_clone_kwargs` 包含 seed
  - `Brain.create_session` runner_kwargs 加 `identity_seed=self._identity_seed` 透传给 `AgentSessionRunner`
  - `AgentSessionRunner.__init__` 接 `identity_seed` kwarg；`run_final_wiring_turn` 调用现场加 `identity_seed=self._identity_seed`
  - `lifeform_domain_growth_advisor.lifeform_builder.build_growth_advisor_lifeform` 默认 `use_identity_seed=True`，自动调 `base_config.with_identity_seed(build_growth_advisor_identity_seed(profile))`，用户调 `build_cheng_laoshi_lifeform()` 时 traits 自动到位
  - 10 个新 production wiring 测试 (`tests/test_lifeform_identity_seed_wiring.py`)：LifeformConfig + Lifeform + Brain + cheng_laoshi 自动接入 + ablation knob (`use_identity_seed=False`) + 协议 IdentityAssertion-vs-seed 一致性
  - **完成 condition 1 100% 闭合**：从 LifeformConfig 一直到 ProtocolRegistryModule.identity_gate 全链路 production wiring，无需用户手动配 `run_final_wiring_turn(identity_seed=...)`
- **Packet 1.3b**（strategy 编译路径，已落地）：
  - `BehaviorProtocol.StrategyPrior` 加 3 字段（`recommended_regime` / `knowledge_weight_hint` / `experience_weight_hint`）让 protocol → PlaybookRule 转换无损
  - `compile_protocol_to_application_artifacts` 输出 `playbook_rules: tuple[PlaybookRule, ...]`；rule id 命名空间 `protocol:{protocol_id}:playbook:{rule_id}`
  - `ProtocolRegistryModule.load_protocol` 一次 upsert boundary + playbook 两份 artifacts；apply 失败回滚 registry
  - 字段映射：`StrategyPrior.applicability_phase` → `PlaybookRule.applicability_scope`（同义 rename）；PE-revision 元数据（initial_weight / pe_decay_rate / 等）留在 protocol 层不进 PlaybookRule（packet 1.5+ 由 ActivationController 消费）
  - cheng_laoshi 16 strategy_priors → 16 PlaybookRule；strategy matched-control test (`tests/contracts/test_protocol_strategy_matched_control.py` 7 测试) 证明 vertical compile vs protocol compile 字段 byte-equivalent（除 rule_id lineage 前缀）
- **Packet 1.4a**（knowledge 编译路径，已落地）：
  - `BehaviorProtocol` 加 `KnowledgeSeed` dataclass + `knowledge_seeds: tuple[KnowledgeSeed, ...]` 字段
  - `compile_protocol_to_application_artifacts` 输出 `domain_knowledge_records: tuple[DomainKnowledgeRecord, ...]`；record id 命名空间 `protocol:{protocol_id}:knowledge:{seed_id}`
  - `ProtocolRegistryModule.__init__` 加 `domain_knowledge_store` kwarg；`load_protocol` 把 records upsert 到 store
  - `final_wiring.py` 把 `domain_knowledge_store` 传给 ProtocolRegistryModule 构造器
  - 字段映射：`evidence_locator` → `locator`（同义 rename）；`url` 由 `protocol.source_locator` 派生（镜像 vertical 行为）；`jurisdiction_tags` 在 fixture uptake 处硬编码 `("private-domain-companion",)` 镜像 vertical
  - cheng_laoshi 16 knowledge_seeds → 16 DomainKnowledgeRecord；matched-control test (`tests/contracts/test_protocol_knowledge_matched_control.py` 7 测试) 证明 vertical compile vs protocol compile 字段 byte-equivalent（除 record_id lineage 前缀）
- **Packet 1.4b**（case 编译路径，已落地）：
  - `BehaviorProtocol` 加 `SignatureCase` dataclass + `signature_cases: tuple[SignatureCase, ...]` 字段（不携带 `lifecycle` / `ttl` / `continuum_*` / `provisional_origin`，让 compile 用 `CaseMemoryRecord` 默认值；protocol 协议层是 reviewed 内容，永远 `CaseLifecycle.VALIDATED`）
  - `compile_protocol_to_application_artifacts` 输出 `case_memory_records: tuple[CaseMemoryRecord, ...]`；case_id 命名空间 `protocol:{protocol_id}:case:{case_id}`
  - `ProtocolRegistryModule.__init__` 加 `case_memory_store` kwarg；`load_protocol` 把 records upsert 到 store
  - `final_wiring.py` 把 `case_memory_store` 传给 ProtocolRegistryModule 构造器
  - 字段映射：所有 review-time 字段 1:1；`delayed_signal_count` / `reconstruction_source` 在 fixture uptake 处硬编码 `1` / `"reviewed-growth-advisor-profile"` 镜像 vertical
  - cheng_laoshi 12 signature_cases → 12 CaseMemoryRecord；matched-control test (`tests/contracts/test_protocol_case_matched_control.py` 7 测试) 证明 vertical compile vs protocol compile 字段 byte-equivalent（除 case_id lineage 前缀）
  - **Checklist 条目 7（compile 路径）现已完全关闭**：boundary + strategy + knowledge + case 全部 4 类 application owners 接入
- **Packet 1.3+**：TemporalArc 真正翻译 / PE feedback 接入 ActivationController / DocumentUptake LLM-辅助抽取
- **关键纪律**：阶段 1 各 packet 期间 ActiveMixtureSnapshot 只进 `shadow_snapshots`，不进 `active_snapshots`；下游 owner baseline 行为不变

**阶段 2（ACTIVE）**：
- boundary_policy / metacontroller / vitals / strategy_playbook / case_memory 切换为读 `ActiveMixtureSnapshot`
- `cheng_laoshi.py` 保留为 regression baseline + 单元测试 fixture
- 新协议从 TaskUptake DocumentUptake 入口（PDF / Markdown）进入

**阶段 3（涌现）**：
- `applicability_phase` 真正由 PE 驱动，不再由外部 phase 计数器打 tag
- 反思修订路径打通，协议条目可被 PE 反复修订
- 协议间 `α/β` 权重由 metacontroller 学

### SHADOW → ACTIVE 升级 checklist

`ProtocolRegistryModule` 默认 `WiringLevel.SHADOW`。把 `FinalRolloutConfig.protocol_runtime` 翻到 `ACTIVE` 的所有路径都必须通过下表全部条件。任何一条未满足而强行 ACTIVE 化都属于契约违规：

| 条件 | 当前状态 | 满足 packet | 守门方式 |
|---|---|---|---|
| 1. `identity_gate` 接 R7 dual-track Self trait gate + R14 regime identity 真实交叉检查 | ✅ **完全闭合（packet 1.3a + 1.3' + 1.3'' + 1.3'''）**：1.3a R14 regime 真实化；1.3' self_traits 机器身；1.3'' populator + e2e 测试；**1.3''' production wiring**——`LifeformConfig.with_identity_seed` → `Lifeform` → `Brain` → `AgentSessionRunner` → `run_final_wiring_turn` → `DualTrackModule` 全自动透传。`build_cheng_laoshi_lifeform()` 默认 `use_identity_seed=True`，用户拿到的生命体自动带 traits，self_trait 过滤无需手动 wiring 即生效 | 1.3a–1.3''' 全部 | runtime fail-loud + e2e tests + production wiring tests |
| 2. `α / β` 由 metacontroller 学（PE history 接入） | ✅ **完全闭合（packet 1.5b + 1.5c-iii）**：β·pe_utility 上线（1.5b — signed_reward EMA），α/β 在线学习（1.5c-iii — REINFORCE-style 代理梯度 `signed_reward × range(signal)`，clamped to [0.1, 5.0]）。owner-side state，每 PE turn 更新；bootstrap / 单协议 / range=0 都 no-op；cheng_laoshi 字节级不变。R8-clean 拆分（独立 `ProtocolPerformanceModule`）留 packet 1.5c-ii——对 ACTIVE 不是 blocker，是后续重构 | 1.5b + 1.5c-iii | `_update_alpha_beta` + `tests/contracts/test_protocol_alpha_beta_learning.py` |
| 3. typed `context_match` 接入（interlocutor zone / regime / retrieval / PE） | ✅ **完全闭合（packet 1.5a + 1.5a'）**：6 个 kernel-side detector 上线 — interlocutor zone (1.5a) / rupture_kind (1.5a) / boundary_violation (1.5a) / **user_dropout** (1.5a' — 读 `rupture_kind == ABANDONED`) / **regime_transition_recent** (1.5a' — 读 `turns_in_current_regime ≤ 1`) / **retrieval_hits_present** (1.5a' — 读 `retrieval_policy.knowledge_domains` 非空)。`DRIVE_HOMEOSTASIS_*` 永久 deferred（vitals 跨层边界，packet 1.0.1 决议） | 1.5a + 1.5a' | `_compute_context_match` + `tests/contracts/test_protocol_context_match.py` |
| 4. 互斥协议仲裁不再用 lexicographic id | ✅ **完全闭合（packet 1.5c-i）**：A↔B drop 决策按 `pe_utility` 排序，pe_utility 高的留下；冷启动 / 平局走 lex 兑现保持确定性 | 1.5c-i | `_resolve_co_activation_incompatibility(pe_utility_by_id=...)` + `tests/contracts/test_protocol_pe_arbitration.py` |
| 5. 至少 1 个下游 consumer 通过 matched-control dual-run 测试 | ✅ packet 1.2 后续：boundary 路径 matched-control（vertical compile vs protocol compile，hint 内容除 lineage 前缀外 byte-equivalent） | 1.2 后续 | `tests/contracts/test_protocol_boundary_matched_control.py` |
| 6. `boundary_union` 字段定位明确（IDs vs 完整 contracts） | ✅ 1.2 锁定选择 A：`boundary_union_ids: tuple[str, ...]`（IDs only） | 1.2 | schema 决议 + contract test |
| 7. `BehaviorProtocol → application owners` compile 路径打通 | ✅ 1.2 boundary + 1.3b strategy + 1.4a knowledge + 1.4b case（**全 4/4 完成**） | 1.2 / 1.3b / 1.4a / 1.4b | implementation + matched-control tests |

**Packet 1.0.1 引入 runtime guard**：`ProtocolRegistryModule.__init__` 检测 `wiring_level=ACTIVE` 时，若 ActivationController 仍在 fallback 模式（条件 1-4 任一未满足），构造时 `raise FallbackActivationActiveError`（`ContractViolationError` 子类）。这把 docstring 注释升级为可执行约束。

**Packet 1.5a' 翻 flag**：所有 ACTIVE-阻塞 checklist 条目（1/2/3/4/5/6/7）已闭合 → `_ACTIVATION_CONTROLLER_FALLBACK_MODE = False`；`is_fallback_mode()` 返 False；`ProtocolRegistryModule(wiring_level=ACTIVE)` 现在合法（不再 raise）；`FinalRolloutConfig(protocol_runtime=ACTIVE)` 也可以构建成功。Guard 机制本身保留：如果未来某个 packet 发现 1.5* 机制有回归，把 flag 翻回 True 即可立刻恢复 fail-loud 守门（`tests/contracts/test_protocol_runtime_active_gate_guard.py::test_guard_fires_when_flag_is_temporarily_reverted` 用 monkeypatch 验证 defence-in-depth）。

### 不允许的迁移路径

- ❌ 不允许"先在 `cheng_laoshi.py` 内部加 PE-revisable 字段"——会污染 fixture 性质
- ❌ 不允许 boundary_policy / metacontroller 在 SHADOW 阶段就改读 ActiveMixtureSnapshot——必须等 dual-run 验证一致后才切
- ❌ 不允许 ProtocolRuntime 持有领域内容副本——只能持有元信息
- ❌ 不允许 fallback ActivationController 进入 ACTIVE wiring（runtime fail-loud 守门）
- ❌ 不允许 kernel `ProtocolRegistryModule` 直接 import 或读 `lifeform_core.vitals.VitalsSnapshot`

## 与其他能力域的关系

### 依赖（ProtocolRuntime 读它们）

- **§3 temporal-abstraction**：metacontroller 是 z_t / β_t 选择的 owner；ProtocolRuntime 给它 prior
- **§7 credit-and-self-modification**：R10 ModificationGate 是协议加载 / 修订的审查门
- **§8 evaluation**：协议 SHADOW → ACTIVE 升级走 evaluation gate
- **§10 cognitive-regime**：regime identity 是 identity_gate 输入之一
- **§11 domain-experience-layer**：协议内容（knowledge / case / playbook）的编译目标
- **§11A lifeform-vitals**：drive 是协议初始化对象，运行时归 vitals 拥有
- **§13 thinking-loop**：反思 worker 产 ProtocolRevisionProposal
- **§14 aac-commitment-lifecycle**：commitment 状态变化作为 PE 信号
- **§16 runtime-ingestion**：DocumentUptake 文档 ingestion 走 §16 路径
- **§17A rupture-and-repair**：rupture 触发是协议失败信号
- **§17C interlocutor-state**：12 轴 zone bool 是 context_match 信号

### 被依赖（它们读 ProtocolRuntime）

- **§6 contract-runtime**：ActiveMixtureSnapshot 注册到 Slot 注册表，与其他快照对齐
- **§17 character-soul-bootstrap**：character profile 通过 FixtureUptake 适配
- **§19 dlaas-platform**：协议是平台级能力；PlatformRegistry 可分租户切片协议

### 不依赖也不被依赖

- substrate 不读不写 ProtocolRuntime
- session 持久化层不持有协议本体（持有 protocol_id 引用）

## Open Questions（已 resolve / deferred）

| # | 问题 | 决议 | 落地 |
|---|---|---|---|
| 1 | 协议继承 merge 语义 | child 覆盖 parent；多类内容（boundary / strategy / knowledge / case）取并集去重 by id | packet 6.5（计划） |
| 2 | 跨 lifeform 共享协议 | **deferred** — 协议本体共享，revision_log per-lifeform；跨 lifeform 部署等 DLaaS platform layer 落地后再处理 | 不阻塞当前需求 |
| 3 | 协议版本兼容 | **deferred** — 新版本起 fresh revision_log；旧版本快照通过 R15 回滚机制保留 | 不阻塞 |
| 4 | identity_gate 量化 | 0/1 硬过滤（已实施 packet 1.3a） | ✅ |
| 5 | TaskUptake 成本 | 离线背景任务（DocumentUptake 通过 lifeform-protocol-runtime wheel 异步执行；不在 turn-time propagate 中调 LLM） | ✅ packet 2.x 设计 |
| 6 | PE 信号 SignalSource 防碎片化 | 闭合枚举 `BehaviorProtocolSignalSource`；新增成员 = 同 PR 加 detector + 至少 1 个 contract test（与 RuptureKind 同模式） | ✅ packet 1.5a' / 7.0 |
| 7 | α/β 初值 | 1.0 / 1.0 cold start，online 由 owner-side REINFORCE-style 代理梯度学习（packet 1.5c-iii） | ✅ packet 1.5c-iii |
| 8 | 协议数量上限 | 硬上限 8（active 协议 review_status in {SHADOW, ACTIVE} 计入；DRAFT / RETIRED 不计入） | ⏳ packet 6.7（计划） |

## 术语对齐速查表

| 本 spec 术语 | 既有系统 / 论文术语 |
|---|---|
| BehaviorProtocol | Task Set + Schema + Script + Production Rules + Boundary Contracts 的捆绑 |
| Active Mixture | PFC current task representation |
| Protocol Registry | Long-term task set library |
| Activation Controller | Norman & Shallice Supervisory Attentional System |
| Identity Core | R7 Self 轨道 + R14 持久身份 |
| Boundary Union | morality / value commitments |
| ActivationReasonKind: IDENTITY_GATE / CONTEXT_MATCH / EQUAL_WEIGHT_FALLBACK | active inference policy posterior 的 kernel-side 组分；PE_HISTORY / DRIVE_COUPLING / MINIMUM_FLOOR 是占位枚举成员（见 §BehaviorProtocol Schema 后注） |
| TemporalPhase | HTN 子目标 / Schank 脚本场景 |
| TaskUptake | Task instruction following / one-shot policy bootstrapping |
| revision_log | R15 可回滚轨迹 |

## 变更日志

- 2026-05-11：初稿（Phase 0 design freeze）
  - 来源：与用户关于"PDF 给 AI → 自动学习 → 越聊越好"的元架构讨论
  - 关键决策：BehaviorProtocol 是新运行时层但**不**是新 application owner；只发布 ActiveMixtureSnapshot；既有 owner 通过读快照调权
  - 关键决策：Identity Core 不进入 Active Mixture，只作 hard gate（与 R7 / R14 对齐）
  - 关键决策：边界契约跨协议取并集，永远不进 utility 计算
  - 关键决策：β_t 切换由 PE 学，禁用日历天数 string tag
  - 关键决策：协议必须自带 PE 信号定义，否则拒绝加载
  - 关键决策：迁移路径采用 R15 三阶段（design freeze → SHADOW → ACTIVE），`cheng_laoshi.py` 降级为 fixture 不删除

- 2026-05-11 (revision，packet 1.0 实施前)：架构修正
  - 来源：实施前研究发现 `vz-* ↛ lifeform-*` import-boundary 与初稿"立 lifeform-protocol-runtime wheel" 矛盾
  - 修正 1：`ProtocolRegistryModule` owner 住 kernel 子包（不是 lifeform wheel；packet 1.0 立在 `vz-cognition.protocol_runtime/`，packet 1.2 迁到 `vz-application.protocol_runtime/`）。`ActiveMixtureSnapshot` 是 propagate 图的 kernel slot，被 boundary_policy / metacontroller / vitals 等 kernel 模块读取，所以 owner 必须可被 kernel import
  - 修正 2：FixtureUptake 是 per-vertical helper（每个 `lifeform-domain-*` 自带 `fixture_uptake.py`），不是统一模块。`lifeform-protocol-runtime` wheel 推迟到 packet 1.1+ 引入 DocumentUptake 时再立——那时才有 LLM 调用之类的 lifeform-side 职责
  - 修正 3：迁移阶段 1 拆成 packet 1.0 / 1.1 / ... 渐进推进；本 spec 阶段 1 章节列出 packet 切分

- 2026-05-11 (packet 1.0.1 consolidation)：4 项收紧（外部 review 触发）
  - 来源：外部 review 指出 packet 1.0 落地后未闭合的 4 个风险
  - 修正 1（vitals 跨层边界）：`drive_value` 信号源从 ActivationController 删除，formula 简化为 3 因子（context_match / PE_utility / identity_gate）。原因：`VitalsSnapshot` 是 lifeform-side 契约（`docs/DATA_CONTRACT.md` 行 33-34/43/596 明文规定不进入 kernel §6 注册表），kernel `ProtocolRegistryModule` 不能直接读。drive 内部状态仍通过 R-PE 主链回流（`VitalsSnapshot.total_pe → prediction_error → PE_utility`）。如果未来确实需要 drive coupling，新增 kernel-side `DriveReadoutSnapshot` adapter（owner 待定，需要 R8 review）
  - 修正 2（content vs config 边界）：新增 §ProtocolRuntime 与 application owners 的内容边界。明确 ProtocolRuntime 不是 boundary / strategy / case / domain knowledge 的 canonical owner；`BehaviorProtocol` 通过 packet 1.2+ 引入的 compile 路径编译进既有 application owners（与 `DomainExperiencePackage` 同形）；`ActiveMixtureSnapshot.boundary_union` 当前发布完整 `BoundaryContract` tuple 是 SHADOW-only 形式，packet 1.2 立项前必须二选一（选择 A IDs only / 选择 B 文档化只读视图）
  - 修正 3（ACTIVE-gate checklist）：新增 §SHADOW → ACTIVE 升级 checklist，列出 7 条门槛条件 + 满足 packet 编号 + 守门方式。把 docstring 警告升级为可执行约束
  - 修正 4（DATA_CONTRACT.md SSOT 缺口）：`active_mixture` slot 已登记到 `docs/DATA_CONTRACT.md` §6 注册表 + 契约语义段（同 PR 落地）

- 2026-05-11 (packet 1.2)：boundary 编译路径 + Choice A 锁定
  - 来源：packet 1.0.1 留下的两个待决（content vs config 边界 / boundary_union 字段定位）+ checklist 条目 6/7
  - 决策 1（Choice A）：`ActiveMixtureSnapshot.boundary_union` 字段重命名为 `boundary_union_ids: tuple[str, ...]`。仅发布 IDs；canonical 内容由 `boundary_policy` / `ApplicationRareHeavyState` 拥有
  - 决策 2（BoundaryContract 扩展）：加 `regime_id` / `answer_depth_limit_hint` / `clarification_required` 三个字段，使 `BoundaryContract` 能 1:1 反向映射到 `BoundaryPriorHint`，protocol → application 转换无损
  - 决策 3（Compile 路径）：新增 `vz-cognition.protocol_runtime.compiler.compile_protocol_to_application_artifacts(protocol)`，输入 `BehaviorProtocol`，输出 `ProtocolApplicationArtifacts(boundary_prior_hints=...)`。Hint id 命名空间 `protocol:{protocol_id}:boundary:{boundary_id}` 携带 lineage（`BoundaryPriorHint` 自身没有 provenance 字段）
  - 决策 4（Owner 注入）：`ProtocolRegistryModule.__init__` 接受 `application_rare_heavy_state` 参数；`load_protocol(bp)` 先 registry 后 compile + upsert；apply 失败回滚 registry。无 state 注入时退化为 packet 1.0 的纯 registry 行为
  - 决策 5（Unload deferred）：已应用 application 工件的协议 `unload_protocol` raises `NotImplementedError`。`ApplicationRareHeavyState` 没有 per-key remove API；packet 1.6+ 反思修订路径会落地此能力
  - 决策 6（FixtureUptake 透传）：`growth_advisor_profile_to_behavior_protocol` 把 `GrowthAdvisorBoundaryPrior.regime_id / answer_depth_limit_hint / clarification_required` 透传到 `BoundaryContract`，让 cheng_laoshi 转换无损
  - 决策 7（final_wiring 注入）：`build_final_runtime_modules` 默认把 `application_rare_heavy_state` 传给 `ProtocolRegistryModule`，使端到端 compile 路径自动启用
  - Checklist 进度：条目 6（`boundary_union` 字段定位）✅ 完成；条目 7（compile 路径）的 boundary 部分 ✅ 完成；strategy / case / knowledge 部分待 packet 1.3+；条目 5（boundary_policy matched-control test）留待 packet 1.2 后续

- 2026-05-11 (packet 1.2 后续)：boundary 路径 matched-control gate
  - 来源：spec checklist 条目 5（"至少 1 个下游 consumer 通过 matched-control dual-run 测试"）packet 1.2 主体留下的尾巴
  - 落地：`tests/contracts/test_protocol_boundary_matched_control.py` 7 个测试。证明 vertical compile path（`build_growth_advisor_package` 经 `apply_domain_experience_packages` 路径产出的 `BoundaryPriorHint`）与 protocol compile path（`compile_protocol_to_application_artifacts` 经 `ProtocolRegistryModule.load_protocol` 路径产出的 `BoundaryPriorHint`）除 `hint_id` lineage 前缀外完全 byte-equivalent
  - 间接证明：`boundary_policy.process()` 在两条 hint 来源下行为完全一致（filter 用 `regime_id` / `trigger_reasons`，不分支于 `hint_id`）
  - 副带验证：vertical 与 protocol 同时启用时 state merge 不破坏（同 `(regime_id, trigger_reasons)` 折叠到单 entry，4 boundary 仍是 4 entries）
  - Checklist 进度：条目 5 ✅ 完成；剩余条目 1-4 / 7（strategy / case / knowledge 部分）待 packet 1.3+

- 2026-05-11 (packet 1.3b)：strategy 编译路径
  - 来源：spec checklist 条目 7（"BehaviorProtocol → application owners compile 路径"）的 strategy 子部分
  - 决策 1（StrategyPrior schema 扩展）：加 `recommended_regime` (默认 None) / `knowledge_weight_hint` (默认 0.45) / `experience_weight_hint` (默认 0.65) 三字段；让 `BehaviorProtocol` 能 1:1 反向映射到 `PlaybookRule`，protocol → application 转换无损（与 packet 1.2 BoundaryContract 扩展同形）
  - 决策 2（compile 路径）：扩展 `compile_protocol_to_application_artifacts` 输出 `playbook_rules: tuple[PlaybookRule, ...]`；新增 `_strategy_prior_to_playbook_rule(protocol_id, prior)` 私有函数；rule id 命名空间 `protocol:{protocol_id}:playbook:{rule_id}` 镜像 boundary 命名规则
  - 决策 3（字段映射策略）：`applicability_phase` → `applicability_scope` 同义 rename；`recommended_regime` / `knowledge_weight_hint` / `experience_weight_hint` 1:1；PE-revision metadata（`initial_weight` / `pe_decay_rate` / `pe_reinforce_rate` / `minimum_weight_floor` / `revision_history`）**dropped**——`StrategyPlaybookModule` 不读，packet 1.5+ ActivationController 直接从 BehaviorProtocol 读；`PlaybookRule.continuum_band_id` / `mean_continuum_position` 用 PlaybookRule 默认值
  - 决策 4（owner load_protocol 一致性）：单次 `load_protocol(bp)` 调用同时 upsert boundary + playbook 两份 artifacts；任一失败 rollback registry；deferred unload semantics 现在覆盖任何 application 工件应用过的协议（boundary 或 playbook 或两者）
  - 决策 5（FixtureUptake 透传）：`growth_advisor_profile_to_behavior_protocol` 把 `GrowthAdvisorStrategyPrior` 的 `recommended_regime` / `knowledge_weight_hint` / `experience_weight_hint` 透传到 `StrategyPrior`，让 cheng_laoshi 转换无损
  - 测试：扩展 `test_protocol_compile.py` (+8 strategy 测试) / `test_protocol_load_to_application_state.py` (+5 strategy 测试) / `test_growth_advisor_fixture_uptake.py` (+1 字段透传测试)；新 `test_protocol_strategy_matched_control.py` 7 测试镜像 boundary matched-control
  - cheng_laoshi 行为完全不变；matched-control 测试和现有 packet 1.0/1.2/1.3a 测试不受影响
  - Checklist 进度：条目 7（compile 路径）✅ boundary + strategy；待 packet 1.4+ 落 case / knowledge

- 2026-05-11 (packet 1.4b)：case 编译路径 — checklist 条目 7 完全关闭
  - 来源：spec checklist 条目 7（"BehaviorProtocol → application owners compile 路径"）的最后一项 (case)
  - 决策 1（schema 扩展）：`vz-contracts.behavior_protocol` 加 `SignatureCase` dataclass + `BehaviorProtocol.signature_cases: tuple[SignatureCase, ...] = ()` 字段。设计选择：**只携带 review-time 字段** + `delayed_signal_count` + `reconstruction_source`；**不携带** `continuum_*` / `lifecycle` / `ttl_seconds` / `expires_at_tick` / `provisional_origin`，因为 protocol 层只发布 reviewed 内容（永远 `CaseLifecycle.VALIDATED`），运行时 lifecycle 状态由 case_memory owner 自己管
  - 决策 2（compile 路径）：扩展 `compile_protocol_to_application_artifacts` 输出 `case_memory_records: tuple[CaseMemoryRecord, ...]`；新增 `_signature_case_to_case_record(protocol_id, case)` 私有函数；case_id 命名空间 `protocol:{protocol_id}:case:{case_id}` 镜像 boundary / playbook / knowledge 命名规则
  - 决策 3（owner 注入）：`ProtocolRegistryModule.__init__` 加 `case_memory_store: ApplicationCaseMemoryStore | None = None` kwarg；`load_protocol(bp)` 在 store 注入时把 records upsert 到 store；deferred unload semantics 现在覆盖 boundary / playbook / knowledge / case 四类
  - 决策 4（FixtureUptake 透传 + 元数据镜像）：`growth_advisor_profile_to_behavior_protocol` 把 `GrowthAdvisorSignatureCase` 14 个字段透传到 `BehaviorProtocol.SignatureCase`，并把 `delayed_signal_count` 和 `reconstruction_source` 硬编码为 vertical 用的值（`1` / `"reviewed-growth-advisor-profile"`）让 cheng_laoshi 转换 byte-equivalent
  - 决策 5（final_wiring 注入）：`build_final_runtime_modules` 默认把 `case_memory_store` 传给 `ProtocolRegistryModule`，使端到端 compile 路径自动启用
  - 测试：扩展 `test_protocol_compile.py`（+8 case 测试）/ `test_protocol_load_to_application_state.py`（+6 case 测试）/ `test_growth_advisor_fixture_uptake.py`（+3 case 字段测试）；新 `test_protocol_case_matched_control.py` 7 测试镜像 boundary / strategy / knowledge matched-control
  - cheng_laoshi 行为完全不变；packet 1.0 - 1.4a 的所有 contract test 零修改
  - **Checklist 条目 7 ✅ 完全关闭**：4 个 application owners（boundary_policy / strategy_playbook / domain_knowledge / case_memory）的 protocol → canonical-store compile 路径全部打通；R8 SSOT 收紧覆盖全部 application 内容

- 2026-05-11 (packet 1.4a)：knowledge 编译路径
  - 来源：spec checklist 条目 7（"BehaviorProtocol → application owners compile 路径"）的 knowledge 子部分
  - 决策 1（schema 扩展）：`vz-contracts.behavior_protocol` 加 `KnowledgeSeed` dataclass（11 必选字段 + jurisdiction_tags / conflict_markers 默认空）+ `BehaviorProtocol.knowledge_seeds: tuple[KnowledgeSeed, ...] = ()` 字段；contracts 序列化与 schema 测试自动覆盖
  - 决策 2（compile 路径）：扩展 `compile_protocol_to_application_artifacts` 输出 `domain_knowledge_records: tuple[DomainKnowledgeRecord, ...]`；新增 `_knowledge_seed_to_domain_record(protocol, seed)` 私有函数；record id 命名空间 `protocol:{protocol_id}:knowledge:{seed_id}` 镜像 boundary / playbook 命名规则
  - 决策 3（字段映射策略）：`evidence_locator` → `locator` 同义 rename（镜像 packet 1.3b 的 `applicability_phase` ↔ `applicability_scope`）；`url` 在 compile 时由 `protocol.source_locator` 派生，因为 vertical 也是用 `profile.source_uri` 一处写所有 records；`jurisdiction_tags` 在 fixture uptake 处硬编码 `("private-domain-companion",)` 让 cheng_laoshi 转换 byte-equivalent
  - 决策 4（owner 注入）：`ProtocolRegistryModule.__init__` 加 `domain_knowledge_store: ApplicationDomainKnowledgeStore | None = None` kwarg；`load_protocol(bp)` 在 store 注入时 upsert records 到 store；任一 apply 失败回滚 registry；deferred unload semantics 现在覆盖 boundary / playbook / knowledge 三类工件
  - 决策 5（FixtureUptake 透传）：`growth_advisor_profile_to_behavior_protocol` 把 `GrowthAdvisorKnowledgeSeed` 11 个字段透传到 `BehaviorProtocol.KnowledgeSeed`，让 cheng_laoshi 转换无损
  - 决策 6（final_wiring 注入）：`build_final_runtime_modules` 默认把 `domain_knowledge_store` 传给 `ProtocolRegistryModule`，使端到端 compile 路径自动启用
  - 测试：扩展 `test_protocol_compile.py`（+7 knowledge 测试）/ `test_protocol_load_to_application_state.py`（+6 knowledge 测试）/ `test_growth_advisor_fixture_uptake.py`（+3 knowledge 字段测试）；新 `test_protocol_knowledge_matched_control.py` 7 测试镜像 boundary / strategy matched-control
  - cheng_laoshi 行为完全不变；packet 1.0/1.2/1.3 的 contract test 零修改
  - Checklist 进度：条目 7（compile 路径）✅ boundary + strategy + knowledge；待 packet 1.4b 落 case

- 2026-05-11 (packet 4.3)：comprehensive soak — PDF + reflection + gate + revision 端到端
  - 测试：`tests/integration/test_pdf_to_experience_full_loop.py` 2 个 e2e 通过；2101 broader contract/integration tests + 2 skipped (live-LLM env-gated)
  - **Phase 4 + Phase 2 + Phase 3 全部完成**：原始用户需求"PDF → AI 学习 → 累积经验"4 个原子需求全部闭环

- 2026-05-11 (packet 3.5)：reflection 真改协议 e2e
  - `tests/integration/test_reflection_protocol_revision_e2e.py` 2 个测试：12 turn 负 PE → reflection 提议 WEIGHT_DECAY → gate 自动批准 → registry apply → loser 协议权重×0.5；L4 boundary 提议永远 queue 不自动应用

- 2026-05-11 (packet 3.4)：ModificationGate evaluate_protocol_revision + RevisionQueue
  - `tests/contracts/test_modification_gate_protocol_revision.py` 12 个测试。L1/L2 自动批；L3 evidence-window>=8 + 非空 pe_signature → 自动批，否则 queue；L4 fail-safe 永远 queue
  - 新 `vz-application/.../protocol_runtime/revision_queue.py`：RevisionQueue（in-memory pending review）+ ApprovalDecision/Outcome enum

- 2026-05-11 (packet 3.3)：ProtocolRegistry.apply_revision + R15 rollback
  - `ProtocolRegistry.apply_revision`：纯函数 `_apply_change` 处理 WEIGHT_DECAY/DEACTIVATE on STRATEGY_PRIOR + ARCHIVE on KNOWLEDGE_SEED/SIGNATURE_CASE
  - `ProtocolRegistryModule.apply_revision`：包装 + 重新 compile 应用到 application owners
  - `checkout_revision`：内容回滚预留 NotImplementedError（packet 3.3 之后落地）
  - 12 个测试通过

- 2026-05-11 (packet 3.2)：reflection rules
  - `vz-cognition/.../reflection/protocol_revision_rules.py` — `propose_strategy_decay` 上线（attribution-weighted PE 平均 < -0.3 持续 ≥ 5 turn → WEIGHT_DECAY 提议）；knowledge/case archival 占位
  - 8 个 rule 单元测试通过

- 2026-05-11 (packet 3.1)：ProtocolReflectionEngine RuntimeModule skeleton
  - 新 `vz-cognition/.../reflection/engine.py` — `ProtocolReflectionEngine(RuntimeModule[ProtocolReflectionSnapshot])`，slot=`protocol_reflection`，dependencies=`("prediction_error", "active_mixture")`，default SHADOW，scan_period=10 turn
  - 内部 deque 维持 PE + active_mixture 历史（窗口 100）；bootstrap PE 跳过；turn_index 去重
  - 集成进 `final_wiring`；11 个测试通过

- 2026-05-11 (packet 3.0)：ProtocolRevisionProposal schema
  - `vz-contracts/behavior_protocol.py` 加 `ProtocolReflectionSnapshot` / `ProtocolRevisionProposal` / `ProposalEvidence` / `ProtocolRevisionTargetField` / `ProtocolRevisionChangeKind` 5 个类型
  - 12 个 schema 测试通过

- 2026-05-11 (packet 2.6)：matched-control DocumentUptake vs FixtureUptake
  - `tests/contracts/test_pdf_extraction_matched_control.py` 4 个测试：boundary count ±1、strategy 类别覆盖（rapport + funnel）、identity_traits overlap、两条路径都 loadable

- 2026-05-11 (packet 2.5)：PDF e2e — 私域运营规划.pdf
  - `docs/fixtures/sample_protocols/private_domain_growth_advisor_guidance.pdf` (9 pages, 6309 chars) 复制进 repo
  - `tests/integration/test_pdf_extraction_with_mock_llm.py` 1 个 mock e2e 测试（CI-friendly，无 LLM 依赖）
  - `tests/integration/test_pdf_to_protocol_e2e.py` 5 个 env-gated 测试（OPENAI_API_KEY / VZ_DOCUMENT_UPTAKE_LIVE_LLM=1 启用）

- 2026-05-11 (packet 2.4)：load_protocol_candidate + ModificationGate review level 分级
  - `vz-application/.../protocol_runtime/owner.py::load_protocol_candidate` — DRAFT + requires_review=True 候选必须先经 review；force=True 紧急覆盖
  - `lifeform-protocol-runtime/.../document_uptake/review.py` — required_review_level 推导 (boundary→L4 / strategy→L3 / knowledge|case→L2 / identity→L1)；approve_candidate / reject_candidate 带 audit
  - 14 个测试通过

- 2026-05-11 (packet 2.3)：LLM-driven JSON-mode 抽取
  - `lifeform-protocol-runtime/.../document_uptake/extraction.py` — `extract_protocol_candidate(chunks, llm_client, ...)` walk → 3 个 prompt × N chunk → merge into BehaviorProtocolCandidate
  - 集中化 prompt registry: `prompts.py` 三个 prompt（identity / boundary / strategy）
  - `MockLlmJsonClient` 用于 unit test
  - 13 个测试通过

- 2026-05-11 (packet 2.2)：PDF / Markdown ingestion + chunking
  - `lifeform-protocol-runtime/.../document_uptake/ingestion.py` — `read_pdf` (pypdf) / `read_markdown` / `chunk_document`（确定性、段落优先）
  - 14 个测试通过

- 2026-05-11 (packet 2.1)：lifeform-protocol-runtime wheel scaffolding
  - 新 wheel `packages/lifeform-protocol-runtime/`：deps=`vz-contracts + vz-application + lifeform-openai-compat + pypdf`
  - import boundary 强制：vz-* ↛ lifeform-protocol-runtime；lifeform-protocol-runtime ↛ dlaas_platform_*；lifeform-protocol-runtime ↛ 其他 lifeform-domain-*
  - 2 个新 contract test 通过

- 2026-05-11 (packet 2.0)：BehaviorProtocolCandidate / ProtocolProvenance schema
  - `vz-contracts/behavior_protocol.py` 加 `BehaviorProtocolCandidate` (frozen, requires_review=True 默认) + `ProtocolProvenance` (source / extracted_at / confidence)
  - `ProtocolSourceKind` 加 `MARKDOWN_UPTAKE` 成员
  - 13 个 schema 测试通过

- 2026-05-11 (packet 4.1)：matched-control 评估 harness
  - `tests/integration/test_protocol_active_vs_legacy_ablation.py` 9 个测试：load vs no-load 对比，验证 4 个 application owner 家族（boundary/playbook/knowledge/case）在 load 路径下都获得 `protocol:cheng-laoshi:*` lineage
  - `tests/integration/test_active_mixture_consumer_audit.py` 3 个测试：审计 active_mixture 至少有一个有效 consumer 通道（直读 deps OR 间接 lineage）

- 2026-05-11 (packet 4.0)：promote `FinalRolloutConfig.protocol_runtime` 默认 ACTIVE
  - 来源：1.5a' 翻 fallback flag 后 ACTIVE wiring 合法；但默认 rollout 还是 SHADOW，意味着所有现有 lifeform 实例（包括 cheng_laoshi）的 `active_mixture` snapshot 仍然只进 `shadow_snapshots`，下游 consumer 看不到。packet 4.0 翻默认让 1.x 系列在生产路径上真正生效
  - 决策 1（直接 flip 默认值）：`FinalRolloutConfig.protocol_runtime: WiringLevel = WiringLevel.SHADOW` → `WiringLevel.ACTIVE`
  - 决策 2（旧 SHADOW 测试反向断言）：`test_protocol_runtime_active_gate_guard.py::test_final_wiring_default_protocol_runtime_is_still_shadow` → `_is_active`；`test_protocol_runtime_owner_uniqueness.py::test_active_mixture_owner_default_wiring_is_shadow` → `_is_active`
  - 决策 3（test_protocol_registry_shadow.py 大改）：原本通篇假设 SHADOW 默认。现在更名为"ACTIVE 行为 + 显式 SHADOW 仍可用"测试集；`_build_module_with_cheng_laoshi()` 显式指定 ACTIVE；shadow path 保留为 explicit-SHADOW 测试
  - 决策 4（matched-control 调整）：`test_loading_protocol_does_not_widen_active_value_drift` 把 `active_mixture` 从 drift 比较中排除（它本身就是 packet under test）；其他活跃 slot 仍受 matched-control 守护
  - 测试：1942 broader contract + final_wiring + protocol_runtime + protocol_load + behavior_protocol + growth_advisor + dual_track + lifeform_identity_seed 全过；无回归
  - cheng_laoshi 行为：`build_cheng_laoshi_lifeform()` 默认走 ACTIVE 路径，`active_mixture` snapshot 进 `active_snapshots`；snapshot **value** 与 packet 1.5a' 完全等价（owner 内部状态 / 计算逻辑都没变）
  - 显式 SHADOW 入口：`ProtocolRegistryModule(wiring_level=WiringLevel.SHADOW)` 仍可用（dual-run / 调试 / 回滚场景）

- 2026-05-11 (packet 1.5a')：context_match 信号源补齐 + ACTIVE flag 翻转（**整个 ACTIVE checklist 完全闭合，protocol_runtime 可升 ACTIVE**）
  - 来源：1.5a 留下了 condition 3 的 partial 状态：3 个 detector（interlocutor / rupture / boundary）已上线但 USER_DROPOUT_OBSERVED 还是占位返 False、retrieval 和 regime-direct 信号源类型甚至还没在 enum 里。1.5c-iii 让 α/β 也真实化后，剩下 condition 3 是唯一还没闭合的 ACTIVE 阻塞。1.5a' 一次补齐：3 个新 detector + flag 翻转 + 测试反向断言
  - 决策 1（schema 扩展 BehaviorProtocolSignalSource）：加 `REGIME_TRANSITION_RECENT` 和 `RETRIEVAL_HITS_PRESENT` 两个新 enum 成员。`USER_DROPOUT_OBSERVED` 不需要新成员（packet 1.0 就在 enum 里，只是 detector 之前是占位）。新增成员遵循"closed-enum + 同 PR 加 detector + 至少一个测试"的原则
  - 决策 2（USER_DROPOUT_OBSERVED 复用 rupture_state）：不引入 dialogue_trace 直读路径。`RuptureStateModule` 已经把 `DialogueExternalOutcomeKind.ABANDONED → RuptureKind.ABANDONED` 的映射做完了（`EXTERNAL_OUTCOME_TO_RUPTURE_KIND`），所以 `_user_dropout_observed` detector 就读 `rupture_state.rupture_kind == ABANDONED`。这避免了 protocol_runtime 重做一遍 dialogue 推断，是 R8 SSOT 的正确做法
  - 决策 3（REGIME_TRANSITION_RECENT 阈值 ≤ 1）：`turns_in_current_regime == 1` 是规范的"刚切换的第一个 turn"；`== 0` 是 cold-start 容忍（fresh session 还没第一次更新）。> 1 不 fire（"已经稳定的 regime"）
  - 决策 4（RETRIEVAL_HITS_PRESENT 读 knowledge_domains 非空）：fire 条件是"retrieval policy 想查任何 domain"，不是"实际有 hits"。理由：从 protocol 视角看，"这是一个需要知识检索的 turn"是与"实际命中了什么"分离的相关性维度；前者属于 retrieval policy 的决策，后者属于 domain knowledge 的命中
  - 决策 5（依赖扩张：retrieval_policy）：`ProtocolRegistryModule.dependencies` 从 7-tuple 扩到 8-tuple（加 `retrieval_policy`）。regime 和 rupture_state 已在 tuple 里（之前的 packet 占好位），不需要重复声明。所有 reader 仍 SHADOW-tolerant
  - 决策 6（向后兼容：detector 函数新参数加 default None）：`_compute_context_match` 和 `_signal_is_firing` 新增 `regime_snapshot` / `retrieval_policy_snapshot` 参数，给 default `None`。这避免了 packet 1.5a 的现有测试 callers 全要改签名
  - 决策 7（**翻 fallback flag**）：`_ACTIVATION_CONTROLLER_FALLBACK_MODE: bool = False`。runtime ACTIVE 入口的 `FallbackActivationActiveError` 不再触发。`is_fallback_mode()` 现在返 False
  - 决策 8（active gate test 反向重写 + defence-in-depth）：`test_protocol_runtime_active_gate_guard.py` 完全重写：核心断言变成"ACTIVE 构造现在成功"+"`is_fallback_mode()` 返 False"+"`FinalRolloutConfig(protocol_runtime=ACTIVE)` 构建成功"。但保留一个用 `monkeypatch` 临时把 flag 翻 True 的测试 — 如果未来某 packet 发现 1.5* 机制回归需要重新 gate，guard 仍能 fail-loud
  - 决策 9（schema test：保 packet-1.0 + 加 1.5a'）：`test_behavior_protocol_schema.py` 把 `_PACKET_1_0_SIGNAL_SOURCES` 保留作 baseline，新增 `_EXPECTED_SIGNAL_SOURCES = baseline | {REGIME_TRANSITION_RECENT, RETRIEVAL_HITS_PRESENT}`。新增 `test_packet_1_0_signal_sources_remain_present` 防止后续 packet 删 packet-1.0 成员（closed-enum 向后兼容契约）
  - 测试：9 个新 detector 测试拓展到 `tests/contracts/test_protocol_context_match.py`（USER_DROPOUT 3 个 / REGIME_TRANSITION 4 个 / RETRIEVAL 3 个）+ active_gate guard 9 个测试反向重写（含 1 个 monkeypatch defence-in-depth）+ `test_protocol_runtime_owner_uniqueness` dependency 断言扩 8-tuple + schema test 加 packet-1.5a' 期望集和 packet-1.0 baseline pin
  - cheng_laoshi 行为：`activation_conditions.context_match_signals = ()` → 6 个 detector 都不读到任何信号 → score 全 0 → EQUAL_WEIGHT_FALLBACK 路径 → 单协议 weight=1.0，字节级不变（之前 packets 的 cheng_laoshi 字节级不变测试无需修改）
  - **Checklist 进度**：条目 3 🟡 → ✅ **完全闭合**；条目 1/2/4/5/6/7 仍 ✅。**整个 ACTIVE checklist 全部 ✅**；`_ACTIVATION_CONTROLLER_FALLBACK_MODE` 翻为 False；`ProtocolRegistryModule` 可升 ACTIVE wiring。R8-clean `ProtocolPerformanceModule` 拆分（packet 1.5c-ii）和 metacontroller-driven α/β（包 1.5c-iv 假想）是后续 refactor / 增强，不再 ACTIVE 阻塞

- 2026-05-11 (packet 1.5c-iii)：α/β 在线学习（checklist 条目 2 完全闭合）
  - 来源：1.5b 让 β·pe_utility 真实，1.5c-i 让硬过滤 PE-driven，但 α/β 系数本身仍硬定 1/1。这意味着系统不能根据"context_match 还是 pe_utility 更预测下一步 PE"动态调整对二者的信任度。1.5c-iii 把 α/β 也变成 owner-side 在线学习状态，使 raw_score 公式真正自适应
  - 决策 1（state 在 ProtocolRegistryModule）：α/β 与 pe_utility 同居一处。R8-clean 拆 `ProtocolPerformanceModule` 留 packet 1.5c-ii — 拆分本身是 refactor 不变行为，可以晚做。注释里显式标了这条路径
  - 决策 2（梯度公式：REINFORCE-style 代理）：`α_grad = signed_reward × range(context_match across last actives)`，β 对称。直觉是："当 context_match 把协议拉开了，且决策结果好，就是 context_match 信号有效，提高 α"。负 PE 反向；range=0（单协议或无差异化）时不更新
  - 决策 3（学习率 η_meta = 0.05）：比 pe_utility EMA 的 η = 0.25 慢 5×。α/β 全局 hyperparameter 影响每个决策，必须慢更新；pe_utility 协议级局部，可以快
  - 决策 4（钳位 [0.1, 5.0]）：防止 α/β 塌缩到 0（信号被永久静音）或跑飞（softmax → argmax）。0.1 floor 让信号始终有最小贡献，5.0 ceiling 给充分动态范围
  - 决策 5（skip 条件齐全）：cold start / bootstrap / 重复 turn_index / 单协议 / 全 0 range。每个 skip 都精确对应一个语义场景；cheng_laoshi 单协议默认场景刚好命中 `len < 2` 早 return，字节级行为保住
  - 决策 6（API 形状：默认 + audit out）：`compute_active_mixture(alpha=1.0, beta=1.0, audit_context_scores=None)`。默认值匹配 1.5b 行为，所有现有 callers / tests 向后兼容；out param 用于 owner 抓取本 turn context_match 给下 turn 学习用，不动 snapshot schema
  - 决策 7（双轨 cache 而非单一）：缓存 `_last_context_scores` 和 `_last_pe_utilities` 分别记录上 turn softmax 看见的 cm 和 pe（不是 α·cm 或 EMA-更新后的 pe）。否则会用错误的 slice 算 range
  - 决策 8（fallback flag 不动）：`_ACTIVATION_CONTROLLER_FALLBACK_MODE` 仍 True；docstring 升级到"α/β 已在线学习；剩下 1.5a' 的 dropout/retrieval 信号源 + 1.5c-ii 的 R8 拆分 是 follow-up 不是 blocker"。明确指出**只要 1.5a' 落地，aggregate flag 即可翻 False**
  - 测试：14 个新 `tests/contracts/test_protocol_alpha_beta_learning.py`：初值 1.0 / cold start 不变 / 缺上游不变 / 单协议永远不变（cheng_laoshi 保护）/ 多协议 + 正 PE + cm 差异 → α↑ / 多协议 + 正 PE + pe 差异 → β↑ / 负 PE 下降 / bootstrap 跳过 / 重复 turn_index 不崩 / 上钳位饱和 / 下钳位饱和 / α/β 影响 softmax 形状 / α/β 出现在 audit detail / cheng_laoshi 默认 e2e 字节级不变
  - cheng_laoshi 行为：永远走 `len(_last_context_scores) < 2` 早 return → α/β 永远 1.0 → softmax(任意值) for 1 protocol = (1.0,) → 字节级等价。pin 在 `test_singleton_mixture_never_updates_alpha_beta` 和 `test_cheng_laoshi_default_e2e_unaffected_by_packet_1_5c_iii`
  - **Checklist 进度**：条目 2 🟡 → ✅ **完全闭合**；条目 4 仍 ✅；条目 3 仍 🟡 partial（剩 1.5a' 的 dropout/retrieval/regime-direct 信号源）；剩余 ACTIVE 阻塞只有条目 3，落 1.5a' 即可解锁 ACTIVE wiring

- 2026-05-11 (packet 1.5c-i)：PE-driven 互斥仲裁（checklist 条目 4 完全闭合）
  - 来源：1.5b 让 β 项真实化、软混合权重跟随经验调整。但**硬过滤**（`_resolve_co_activation_incompatibility`）仍按 lex tiebreak — 一个协议虽然 PE 历史好，但 lex 序输了就被 drop，反而被 PE 历史差但 lex 序赢的协议干掉。这与 1.5b 的 β·pe_utility 软偏向逻辑相矛盾。1.5c-i 把这条数据通路统一：硬过滤和软混合都用 owner-side 同一份 `pe_utility` EMA
  - 决策 1（同 source 共享）：`_resolve_co_activation_incompatibility` 增加 `pe_utility_by_id: Mapping[str, float] | None = None` 参数；`compute_active_mixture` 把已有的 `pe_utility_by_id` 透传过去。一份 EMA 同时驱动两条决策（硬 drop + 软权重），避免 SSOT 二份写
  - 决策 2（tiebreak 阶梯）：先按 `pe_utility` 比，相等再按 lex。这意味着 cold start（owner 还没攒任何 PE 历史）和精确平局（极少见但确定会发生）都走 lex —— 测试 / 外部直调 `compute_active_mixture` 不传 `pe_utility_by_id` 时也走 lex，向后兼容 1.5b 之前所有 callers
  - 决策 3（self-drop 提前 break）：iter 一个 protocol 的 incompatible list 时，若该 protocol 自己被 drop 了，立即 break。这避免"已经死掉的 A 还在替 B/C 安排殡仪"的语义混乱（A 已经不存在，无权再下达 drop 命令）
  - 决策 4（fallback flag 不动）：`_ACTIVATION_CONTROLLER_FALLBACK_MODE` 仍 True；`is_fallback_mode()` docstring 升级到"PE 仲裁真实化"。runtime ACTIVE 入口仍由剩余条目（α/β 学习）守门
  - 决策 5（不动 schema）：`BehaviorProtocol.ActivationConditions.co_activation_incompatible` schema 完全不变，仅 resolver 行为升级。所有现有 fixtures（cheng_laoshi 默认无 incompatible）行为字节级等价
  - 测试：12 个新 `tests/contracts/test_protocol_pe_arbitration.py`：无 incompatible 全 pass / 冷启动 lex / 显式全 0 lex / 高 PE 战胜 lex / 低 PE 输给 lex 优势方 / 仅一方声明也工作 / 双向声明幂等 / 三向冲突 alpha 全胜 / 三向冲突 alpha 自损 stop iter / 平 PE 落 lex / `compute_active_mixture` e2e 切换 / cheng_laoshi e2e 字节级不变（无 incompatible 声明）
  - cheng_laoshi 行为：`co_activation_incompatible = ()` → 永远不进 arbitration 路径 → 完全字节级等价（pin 在 `test_cheng_laoshi_e2e_unaffected_by_packet_1_5c_i`）
  - **Checklist 进度**：条目 4 ⏳ → ✅ **完全闭合**；条目 2 仍 🟡 partial（α/β 学习留 1.5c-iii）；条目 3 仍 🟡 partial

- 2026-05-11 (packet 1.5b)：PE history utility（β 真实化，checklist 条目 2 partial 闭合）
  - 来源：1.5a 让 α 项真实化后，β 项还是 0；这意味着即使协议在 PE 历史上表现差也不会被降权。1.5b 的目的是让协议 weight 跟随经验自适应——这是用户最初提出"有内部机制把指导文档快速整合到行为，并积累经验"的核心机制
  - 决策 1（owner-side 滑动 EMA）：per-protocol `pe_utility` 在 `ProtocolRegistryModule` 内部维护，不抽离独立 owner。理由：load/unload 协议是 EMA 条目的自然生命周期锚点；当前只有自己一个 reader；R8-clean split（拆 `ProtocolPerformanceModule`）留待 packet 1.5c 当 metacontroller 成为第二个 reader 时一起做。注释里显式标了这个 deferred 决策
  - 决策 2（attribution 公式）：`Δ_i = signed_reward × last_weight_i`，`pe_utility_i ← (1-η) · pe_utility_i + η · Δ_i`，η=0.25。PE at turn t 反映 turn t-1 行动的后果，所以归因到 turn t-1 的 active mixture 而不是当前的（这意味着 turn 1 没有归因可做，是冷启动）。η=0.25 经验值，~4 turn 跟随阶跃响应；保守不抖动
  - 决策 3（衰减+更新一遍历）：每 PE turn 先把 dict 里所有协议的 EMA 乘 (1-η)，再加上当 turn 活跃协议的 η·Δ。等价于"所有协议都有 sample，未活跃的 sample=0"。这样未活跃协议自然 forgetting，不会保留 stale 历史
  - 决策 4（钳位 [-1, +1]）：防止小样本场景下持续累计溢出。η 衰减本身已经是软饱和，钳位是双重保险
  - 决策 5（bootstrap 跳过）：`PredictionErrorSnapshot.bootstrap=True` 是 placeholder（系统刚启动还没真实预测），attribute 进 EMA 会污染。跳过
  - 决策 6（turn_index 去重）：`_last_pe_turn_index` 记录上次处理过的 PE turn_index，重复或更早的 turn_index 直接 return。防 replay / retry 双计
  - 决策 7（β=1 硬定）：α=1, β=1 在 raw_score 公式中硬定。纯靠 EMA 累计已经能实现 "好的协议被偏向"。α/β 学习留 packet 1.5c，不是 1.5b 范围
  - 决策 8（fallback 触发条件扩展）：原来"`max(context_match) > 0` → softmax，否则 equal_weight"。1.5b 改为"`max(context_match) > 0 OR max|pe_utility| > 0` → softmax"。cheng_laoshi 默认形态（无 signals + 冷启动）仍走 EQUAL_WEIGHT_FALLBACK，但只要有过几 turn 真实 PE，就会切到 CONTEXT_MATCH 路径
  - 决策 9（依赖扩张）：`ProtocolRegistryModule.dependencies` 加 `prediction_error`，从 6-tuple 扩到 7-tuple。`prediction_error` 是 ACTIVE kernel 上游（已经被 PredictionErrorModule 发布），SHADOW-tolerant 缺失不报错
  - 决策 10（fallback flag 不动）：`_ACTIVATION_CONTROLLER_FALLBACK_MODE` 仍 True；`is_fallback_mode()` docstring 升级到"β 真实化，α/β 学习仍硬定"。runtime ACTIVE 入口仍由 `FallbackActivationActiveError` 守门
  - 测试：13 个新 `tests/contracts/test_protocol_pe_utility.py`：cold start / SHADOW 缺上游 / 单 turn 归因 / 多 turn EMA 累积 / 负 PE 下降 / bootstrap 跳过 / turn_index 去重 / 差异化 softmax 权重 / pe_utility 出现在 audit detail / cheng_laoshi 单协议权重不变 / cheng_laoshi 默认 EQUAL_WEIGHT_FALLBACK 保留 / 未活跃协议衰减 / 钳位 saturate
  - cheng_laoshi 行为：单协议 → softmax(任意值) = (1.0,) → weight=1.0 不变。无 PE 历史 + 无 signals → EQUAL_WEIGHT_FALLBACK 路径。default `build_cheng_laoshi_lifeform()` 拿到的生命体行为字节级等价（pin 在 `test_cheng_laoshi_singleton_weight_unchanged_under_pe_history` 和 `test_cheng_laoshi_default_path_under_packet_1_5b`）
  - **Checklist 进度**：条目 2 ⏳ → 🟡 partial（β 真实化，α/β 学习留 1.5c）；条目 3 仍 🟡 partial（未变）；条目 4 仍 ⏳

- 2026-05-11 (packet 1.5a)：typed context_match 计分框架（checklist 条目 3 partial 闭合）
  - 来源：1.3 系列把 identity_gate 完全真实化后，剩下的 ACTIVE checklist 条目 2/3/4 都围绕 activation formula 的另两个因子。1.5a 是最小可验证的一刀：把 `context_match` 从"空集，contribution = 0"升级到"3 个 kernel-side detector 真实参与 softmax"
  - 决策 1（α=1, β=0 暂定）：把 formula 中 α 临时硬定为 1.0、β 仍 0；本 packet 不引入 PE history utility 也不接 metacontroller 学 α/β（留 packet 1.5b/c）。这样的拆分让 1.5a 自己就能产出可观测行为变化（"信号 fired → 权重升高"），不需要等 1.5b/c 才能验证
  - 决策 2（3 detector 范围）：仅实现 3 个 kernel-readable 信号源：
    - `INTERLOCUTOR_ZONE_TRANSITION` ← `interlocutor_state.state.*_zone` 任一 True → fire
    - `RUPTURE_KIND_FIRED` ← `rupture_state.rupture_kind is not None` → fire
    - `BOUNDARY_VIOLATION_FIRED` ← `boundary_policy.trigger_reasons` 非空 → fire
  - 决策 3（DRIVE 永久 defer，USER_DROPOUT 推迟）：`DRIVE_HOMEOSTASIS_HOLD/BREACH` 永久 deferred（vitals 不在 kernel propagate 图里，packet 1.0.1 决议；通过 PE 主链回流而不是直读 vitals）。`USER_DROPOUT_OBSERVED` 推迟到 1.5a'（需要 dialogue_trace inspection 或 typed proxy slot）。两类信号的 detector 显式返回 False 而非 raise，让现有协议 schema 能继续声明这些信号但暂不影响计分
  - 决策 4（softmax 触发条件）：当所有 eligible 协议 `max(score) == 0` 时回落到 `equal_weight_with_floor`（保 cheng_laoshi 默认形态：`activation_conditions.context_match_signals` 为空 → score 全 0 → 与 packet 1.4b 完全等价）。任一 score > 0 → 切到 numerically-stable softmax + per-protocol floor，`ActivationReason.kind = CONTEXT_MATCH`，`detail` 列出 `signals_fired=[...]` 用于审计
  - 决策 5（依赖扩张）：`ProtocolRegistryModule.dependencies` 从 `("dual_track", "regime")` 扩到 `("dual_track", "regime", "interlocutor_state", "rupture_state", "boundary_policy")`。所有读路径 SHADOW-tolerant：缺 upstream → "信号不 fire" 而不是 fail-loud（ACTIVE 升级仍由 `FallbackActivationActiveError` + checklist 守门）
  - 决策 6（import boundary 更新）：`vz-application` 的 `ALLOWED_VZ_UPSTREAM` 加 `interlocutor`（rupture_state / behavior_protocol 已在）；vz-application 的 pyproject.toml 已声明 vz-cognition 依赖，无需变动
  - 决策 7（fallback flag 不动）：`_ACTIVATION_CONTROLLER_FALLBACK_MODE` 仍 True；`is_fallback_mode()` docstring 升级到"context_match partial 真实化，PE-utility / α-β 学习仍是 0"。runtime ACTIVE 入口（FallbackActivationActiveError）仍会在 ACTIVE 尝试时 fail-loud
  - 测试：15 个新 `tests/contracts/test_protocol_context_match.py`：空信号集→score 0 / 3 detector 各自 fire+不 fire 6 case / DRIVE+USER_DROPOUT defer 2 case / 多信号聚合 / softmax 差异化权重 / cheng_laoshi e2e 行为在所有信号都 fire 的极端 fixture 下仍然 byte-equivalent；`test_protocol_runtime_owner_uniqueness` 的依赖断言扩 5-tuple
  - cheng_laoshi 行为：`activation_conditions.context_match_signals = ()` → score 全 0 → 走 EQUAL_WEIGHT_FALLBACK 路径，`weight = 1.0`（单协议混合）；与 packet 1.4b 完全字节级等价。3 个 1.5a 测试明确 pin 这一保证
  - **Checklist 进度**：条目 3 ⏳ → 🟡 partial（3 detector 上线，retrieval / regime-direct / α-β 学习留 1.5a' / 1.5b）；条目 2/4 仍 ⏳；条目 5/6/7 + 条目 1 仍 ✅

- 2026-05-11 (packet 1.3''')：production wiring of identity seed (1.3 系列收尾)
  - 来源：1.3'' 完成机器身 + populator，但 cheng_laoshi 生命体的用户仍需手动调 `run_final_wiring_turn(identity_seed=...)` 才能让 trait 生效。1.3''' 把这条线接到 `LifeformConfig`，让 `build_cheng_laoshi_lifeform()` 自动带 seed
  - 决策 1（LifeformConfig 字段 + with_identity_seed）：在 `lifeform-core/lifeform.py` 添加 `identity_seed: IdentitySeed | None = None` 字段和 `with_identity_seed(seed)` helper，镜像 `with_vitals` pattern。frozen dataclass + `dataclasses.replace`
  - 决策 2（Lifeform → Brain）：`Lifeform.__init__` 把 `self._config.identity_seed` 透传给 `Brain(identity_seed=...)` 构造器
  - 决策 3（Brain 镜像 RegimeBootstrap）：`Brain.__init__` 加 `identity_seed: IdentitySeed | None` kwarg；`_clone_kwargs` 包含；新增 `with_identity_seed(seed)` clone helper（镜像 `with_regime_bootstrap`）；`Brain.identity_seed` 只读 property
  - 决策 4（Brain → AgentSessionRunner）：`Brain.create_session` runner_kwargs 加 `identity_seed=self._identity_seed`，透传给 `AgentSessionRunner.__init__`
  - 决策 5（AgentSessionRunner → run_final_wiring_turn）：`AgentSessionRunner.__init__` 接 `identity_seed: IdentitySeed | None = None` kwarg，存为 `self._identity_seed`；唯一调用 `run_final_wiring_turn(...)` 的位置加 `identity_seed=self._identity_seed`
  - 决策 6（vertical fixture 自动接入）：`build_growth_advisor_lifeform` 加 `use_identity_seed: bool = True` 参数（镜像 `use_vitals_bootstrap`），默认 True 自动调用 `base_config.with_identity_seed(build_growth_advisor_identity_seed(profile))`。Ablation 路径 `use_identity_seed=False` 保留
  - 测试：10 个新 `tests/test_lifeform_identity_seed_wiring.py` —— LifeformConfig + Lifeform + Brain wiring + cheng_laoshi 默认 + ablation + 协议-vs-seed 一致性
  - cheng_laoshi 用户体验：之前 `build_cheng_laoshi_lifeform()` 拿到的生命体走 `self_traits_populator_pending` SHADOW path；现在自动激活真实过滤，protocol identity gate 的 self_trait 分支 100% 工作
  - **整个 1.3 系列完成**：从 packet 1.3a (regime 真实化) 一路到 1.3''' (production wiring)，spec checklist condition 1 完全闭合

- 2026-05-11 (packet 1.3'')：identity_gate self_traits populator
  - 来源：spec checklist 条目 1 自 packet 1.3' 起 mostly-real 但缺 populator —— 这一轮把 traits 真实写入 dual_track，让 cheng_laoshi e2e 实际激活 self_trait 过滤
  - 决策 1（schema 位置）：`IdentitySeed` 放在 `vz-contracts/identity_seed.py`（不是 `vz-cognition.dual_track`），原因：跨 wheel 共享（vz-cognition 消费 + lifeform-domain-* 生产），住 vz-contracts 避免 lifeform 侧多增 vz-cognition 依赖。镜像 `BehaviorProtocol` 在 vz-contracts 的位置选择
  - 决策 2（DualTrackModule 注入）：`__init__` 接受 `identity_seed: IdentitySeed | None = None` kwarg；`derive_track_state` 在 SELF 轨道用 seed.traits 写入 `TrackState.traits`；WORLD 轨道始终空（identity 描述 lifeform 自身，不描述世界）。默认 None 保住所有现有 dual_track baseline 测试
  - 决策 3（final_wiring 透传）：`run_final_wiring_turn` 和 `build_final_runtime_modules` 加 `identity_seed: IdentitySeed | None = None` kwarg，传给 DualTrackModule 构造器
  - 决策 4（vertical fixture）：`build_growth_advisor_identity_seed(profile)` 返回 `IdentitySeed(traits=("warm_peer_register", "long_horizon"))` 对应 cheng_laoshi `BehaviorProtocol.IdentityAssertion.requires_self_traits` 的同一组字符串。**这不是循环依赖**：seed 是 lifeform-construction 时刻冻结的输入，protocol 运行时只能 assert against it，不能 mutate；seed 是真理来源
  - 决策 5（推迟 LifeformConfig wiring 到 packet 1.3'''）：当前 packet 不动 LifeformConfig / Lifeform.create_session / BrainSession 三个位置；用户暂时通过 `run_final_wiring_turn(identity_seed=seed)` 手动传。production 路径全自动注入留 packet 1.3''' 处理（plumbing-only，~3 文件）
  - 决策 6（import boundary 更新）：`vz-cognition` 和 `vz-runtime` 的 ALLOWED_VZ_UPSTREAM 加 `identity_seed`；`lifeform-domain-growth-advisor` 不需要 vz-cognition 依赖（直接 import vz-contracts.identity_seed）
  - 测试：12 个新 `tests/test_dual_track_identity_seed.py` + 2 个新 e2e `tests/contracts/test_protocol_identity_gate.py`（cheng_laoshi 通过 + hostile 过滤）；所有 dual_track baseline 测试零修改
  - cheng_laoshi 行为：fixture 测试用 `dual_track_snapshot=None` 走 absent 路径不变；production e2e 路径首次激活 self_trait 真实过滤
  - **Checklist 条目 1 ✅ 完全关闭**：R14 regime + R7 self_traits 双分支真实化 + populator 接通 + e2e 验证

- 2026-05-11 (packet 1.3')：identity_gate self_traits 分支真实化机器身
  - 来源：spec checklist 条目 1（"identity_gate 接 R7+R14 真实交叉检查"）的 R7 Self trait 子部分。Packet 1.3a 已经完成 R14 regime 部分；这一轮把 R7 self_trait 的检查机器身就位
  - 决策 1（dual_track schema 扩展）：`vz-cognition.dual_track.TrackState` 加 `traits: tuple[str, ...] = ()` 字段。默认空保住向后兼容（所有现存 `derive_track_state` 调用 / 测试 / propagate 路径不变）。`SubstrateModule` 的 4 个 TrackState 构造点都用 kwargs，不受字段顺序影响
  - 决策 2（_compute_identity_gate 重写）：self_traits 分支从"整体延后"改为"按 traits 实际状态分支"：
    - dual_track snapshot 缺失 → `self_traits_dual_track_absent_shadow_pass` 通过
    - dual_track 存在但 traits 空 → `self_traits_populator_pending` 通过（标记 populator 待来）
    - traits 非空 → 真实检查：required 必须 ⊆ traits（缺则 gate=0），forbidden 不可 ∩ traits（出现则 gate=0）。匹配时按"required_match" / "forbidden_absent" 标记进 audit detail
  - 决策 3（populator 推迟）：本 packet 不实现 trait 派生 populator —— `derive_track_state` 不写 `traits`。这意味着生产代码下 `self_track.traits` 始终空，identity gate 仍走 `self_traits_populator_pending` 路径，cheng_laoshi 行为不变。未来 packet（packet 1.3'' 或类似）实现 populator 时，identity gate 自动激活真实过滤，无需 protocol_runtime 侧改动
  - 决策 4（fallback flag 不翻）：`is_fallback_mode()` 仍为 True。条目 1 的"机器身"完成但"实际过滤效果"待 populator；条目 2/3/4 仍 placeholder。flag 语义是"ACTIVE 不安全"，不是"具体哪一项还在 fallback"
  - 测试：7 个新合成 traits 测试（subset 匹配 / required 缺失 / forbidden 命中 / 空 traits 占位 / e2e via compute_active_mixture），3 个旧测试改名（`self_traits_check_deferred_packet_1_3_prime` → 按场景分别为 `_dual_track_absent_shadow_pass` / `_populator_pending`）。20 测试全过
  - cheng_laoshi 行为完全不变（test fixtures 都用 `dual_track_snapshot=None` 触发 absent 分支；production code 走 populator-pending 分支）
  - Checklist 进度：条目 1 🟢 mostly-real（机器身就位，populator 单独 packet）；剩余条目 2/3/4 待 packet 1.5

### Phase 9 — 架构最后一公里 (packet 9.0 → 9.4)

- 2026-05-12 (packet 9.4)：spec 终扫同步 (Phase 9)
  - 标记 PE→学习闭环 ✅ 完整闭合（packet 9.0）
  - 标记 archival rules ✅ 真实逻辑（packet 9.1 + 9.2）
  - 标记 ProtocolPhaseModule / ProtocolRegistryIntrospectionModule / ProtocolRevisionLogModule ✅ ACTIVE（packet 9.3）
  - 明文 disposition：USER_REPLY_LATENCY / USER_REPLY_LENGTH 是 interlocutor 字段 proxy（无 dialogue_trace 直读），DRIVE_HOMEOSTASIS_* 永久 deferred（vitals 跨层）；这些是设计决议而非未完成项
  - 工程交付层（lifeform-service 接线、RevisionQueue 持久化、CLI Web UI）明文标 OUT-OF-SCOPE for spec

- 2026-05-12 (packet 9.3)：SHADOW → ACTIVE flip
  - `ProtocolPhaseModule` ACTIVE — `ActiveProtocolEntry.current_phase_id` 现在真的来自 PE-driven phase 推进，不再 fallback to first phase
  - `ProtocolRegistryIntrospectionModule` ACTIVE — registry 内容 (含 lineage_ids) 在 active_snapshots 可见
  - `ProtocolRevisionLogModule` ACTIVE — 修订审计在 active_snapshots 可见
  - `ProtocolReflectionEngine` 保持 SHADOW（生成 proposals；production opt-in）
  - `ProtocolRevisionQueueModule` 保持 SHADOW（auto-apply mutation；production opt-in）
  - 50 targeted + 2054 broader tests 过；老 drift gate 加入新 ACTIVE publishers 到 expected drifters

- 2026-05-12 (packet 9.2)：propose_knowledge_archival / propose_case_archival 真实逻辑
  - 之前 6.2 的 scaffold 现在有真实证据：lineage_ids 来自 9.1 的 `ProtocolRegistrySnapshot.entries[].knowledge_lineage_ids` / `case_lineage_ids`
  - 算法：last N turns 内，protocol active 但 lineage_id 从未出现在 hits → emit ARCHIVE proposal (review_level=L2)
  - 对未激活协议 (weight=0) 不产生 false positive
  - 25 tests

- 2026-05-12 (packet 9.1)：per-protocol seed/case lineage 发布
  - `ProtocolRegistryEntry` 加 `knowledge_lineage_ids` / `case_lineage_ids` 字段
  - `ProtocolRegistryIntrospectionModule.process` 通过纯函数 `compile_protocol_to_application_artifacts` 计算 lineage（idempotent，无副作用）
  - 反思引擎可以无 cognition→application import 跨界地拿到 per-protocol lineage 表
  - 7 tests

- 2026-05-12 (packet 9.0)：ProtocolRevisionQueueModule —— 关闭 PE→学习闭环
  - 新模块消费 `protocol_reflection` upstream → 走 `evaluate_protocol_revision`（ModificationGate）→ AUTO_APPROVED 调 `ProtocolRegistryModule.apply_revision`，QUEUED_FOR_HUMAN 入 `RevisionQueue`
  - 之前反思引擎产出的 proposals 是 dead-end snapshot，没人 apply。现在 closed-loop "PE → reflection → 提案 → 自动 apply（安全 ChangeKind）/ 人审 (高风险)" 真实工作
  - 默认 SHADOW；生产侧通过 `FinalRolloutConfig.protocol_revision_queue=ACTIVE` opt-in
  - dedup by proposal_id 防止重复路由；apply 失败有 rationale 落 audit 不抛
  - 10 tests

### Architectural Disposition: proxy detectors

部分 SignalSource 在 spec 中被声明，但底层 typed source 不存在（或永久不会存在）：

| SignalSource | 当前实现 | Disposition |
|---|---|---|
| `USER_REPLY_LATENCY` | 用 `interlocutor_state.engagement_intensity` + `directness` 反向 proxy | 持续 proxy；dialogue_trace 加入 ms-级 latency 字段后再升级 |
| `USER_REPLY_LENGTH` | 用 `interlocutor_state.engagement_intensity` + `self_disclosure_level` 正向 proxy | 持续 proxy；dialogue_trace 加入 token-count 字段后再升级 |
| `DRIVE_HOMEOSTASIS_HOLD/BREACH` | 永久返回 False | **永久 deferred** — vitals 在 lifeform-core，不在 kernel propagate graph（packet 1.0.1 R8 决议）。protocol 可以声明此 source 但 kernel 不会激活。希望接 vitals 的协议应改用 INTERLOCUTOR_ZONE_TRANSITION 等 kernel-side 替代 |

这些不是"待实施"。它们是为了协议 schema 闭合而保留的成员，对应的 detector 行为已设计成 fail-safe（返回 False 而不是 raise）。

### Out-of-scope for spec（工程交付层）

明文记录这些是部署/平台/UI 任务，不在协议运行时 spec 的"是否完成"判定中：

- `lifeform-service`（HTTP/REST 接口）接 `ProtocolRegistryModule` —— 部署层
- `RevisionQueue` 持久化后端（DB / Redis）—— 部署层
- `protocol_revision_review` CLI 升级为 web UI —— UX 层
- 协议跨 lifeform 部署（DLaaS platform layer）—— 跨 wheel 编排层
- live LLM 端到端验证的常态化（OPENAI_API_KEY env-gated tests）—— CI 层

这些不影响协议运行时**架构完整性**——spec 范围内的"思想到位 + 测试覆盖"目标已 100% 闭合。

- 2026-05-11 (packet 8.0)：spec 终扫同步（本 packet）
  - 把所有 packet 5.0 → 7.4 的实现状态写进 spec
  - 修订类型表 / SignalSource 表 / Open Q 表 / schema 字段名全部对齐 code
  - 1981+ 测试覆盖之前已通过；本 packet 仅做文档同步

- 2026-05-11 (packet 7.4)：CLI tool revision-review
  - `volvence_zero/cli/protocol_revision_review.py`：操作员 CLI，list / approve / reject 走 `RevisionQueue`
  - L4 队列的人工审核路径首次有可执行入口
  - 7 tests

- 2026-05-11 (packet 7.0–7.3)：完整 SignalSource + 完整 TaskUptake 矩阵
  - `BehaviorProtocolSignalSource` 加 5 个成员：USER_REPLY_LATENCY / USER_REPLY_LENGTH / USER_INITIATIVE_QUESTION / COMMITMENT_FULFILLED / COMMITMENT_BROKEN —— 全部带 detector 实现
  - `lifeform-protocol-runtime` 加三个 uptake adapter：TaskDescriptionUptake (短文本) / DirectoryScanUptake (扫描目录) / APIInjectionUptake (JSON 注入)
  - 22 tests

- 2026-05-11 (packet 6.0–6.9)：spec / schema 偏差修正 + coherent feature gap 闭合
  - **6.0**：spec & schema 文本反差修正 — BehaviorProtocol 实际 schema、`SuccessSignal/FailureSignal` 默认值、`ActiveMixtureSnapshot` 字段名 `boundary_union_ids` (而非草案的 `boundary_union`)、`ActivationReasonKind` 全员、SignalSource 成员表
  - **6.1**：4 个新 ChangeKind — WEIGHT_REINFORCE / BOUNDARY_REFINEMENT / IDENTITY_CLARIFICATION / PROTOCOL_RETIREMENT；apply 路径 + reflection 规则 (`propose_strategy_reinforce` / `propose_protocol_retirement`)
  - **6.2**：ProtocolReflectionEngine 加 `domain_knowledge` / `case_memory` 依赖，duck-typed 读 hits 入 history；archival rule scaffold 就位（per-protocol seed 跟踪推迟到 ProtocolRegistry 暴露 seed 列表后）
  - **6.3**：`checkout_revision` 完整内容回滚 — 每次 apply 存储 post-apply snapshot，回滚到任意 revision 或初始
  - **6.4**：`ReviewStatus.RETIRED` + `Registry.loaded()` 过滤 RETIRED + `loaded_all()` 全集 audit
  - **6.5**：`parent_protocol_id` 链式继承 — `merge_protocol_chain` walk 链；child 覆盖 parent；by-id union；cycle / missing parent loud-fail
  - **6.6**：`LoadContext` review_level enforcement — boundary HARD_BLOCK ⇒ L4；identity assertion ⇒ L3；soft remind ⇒ L2；其余 L1
  - **6.7**：`Registry.ACTIVE_PROTOCOL_HARD_CAP=8` — Open Q8 锁定；DRAFT / RETIRED 不计入；超限 raise `ProtocolLimitExceededError`
  - **6.8**：introspection sibling owners — `protocol_registry` slot (per-protocol summary) + `protocol_revision_log` slot (cross-protocol audit trail)；都 SHADOW 默认
  - **6.9**：full unload — `remove_*_by_id_prefix` 在 ApplicationRareHeavyState / DomainKnowledgeStore / CaseMemoryStore；`unload_protocol` 完整清理（boundary / playbook / knowledge / case 四类）
  - 65+ tests

- 2026-05-11 (packet 5.0–5.2)：3 处最严重 gap 修复
  - **5.0**：`ProtocolPhaseModule` + `protocol_phase` slot — PE-driven phase advancement（不是 calendar tag），`TemporalArc.progression_signals` 字段，per-protocol streak counting，`ActiveProtocolEntry.current_phase_id` 现在真实
  - **5.1**：response_assembly + strategy_playbook 真实 declare 并消费 active_mixture / protocol_phase；audit 测试升级到至少 2 个 ACTIVE 消费者；boundary_policy 跨层用 compile path（避免 cycle）
  - **5.2**：`ADD_STRATEGY` ChangeKind + `propose_strategy_addition` 反思规则 + recompile to playbook owner — "系统能从经验里长出新规则"的最后一块砖
  - 49+ tests

- 2026-05-11 (packet 1.3a)：identity gate regime 分支真实化
  - 来源：spec checklist 条目 1（"identity_gate 接 R7+R14 真实交叉检查"）的 R14 子部分
  - 决策 1（依赖图扩展）：`ProtocolRegistryModule.dependencies` 从 `()` 扩为 `("dual_track", "regime")`；topo-sort 把 protocol_runtime 排在 dual_track / regime 之后；contract test `test_active_mixture_owner_declares_expected_dependencies` 锁定
  - 决策 2（regime 分支真实化）：`_compute_identity_gate(protocol, *, dual_track_snapshot, regime_snapshot)` 中，`required_regime_compatibility` 非空 ∧ `RegimeSnapshot.active_regime.regime_id` 不在集合 → 返回 `(0.0, reasons)`，协议被硬过滤；匹配 → `(1.0, ["regime_match:..."])`；空 required → permissive pass。`compute_active_mixture` 在合并阶段前 drop 掉 gate=0 的协议
  - 决策 3（self_traits 分支 permissive 占位）：`DualTrackSnapshot` 当前没有 trait 词汇，packet 1.3a 不阻塞；非空 `requires_self_traits` / `forbidden_self_traits` 标记 `self_traits_check_deferred_packet_1_3_prime` 后通过。包 1.3' 在 dual_track 加 trait 字段后真实化
  - 决策 4（SHADOW permissive missing upstream）：regime / dual_track 上游缺失时（SHADOW 模式 / 测试 fixture）gate 默认 permissive，避免阻塞 SHADOW dual-run；ACTIVE 升级仍由 `FallbackActivationActiveError` + checklist 守门，所以"missing upstream pass"不会偷偷溜进 ACTIVE 路径
  - 决策 5（fallback flag 不翻）：`is_fallback_mode()` 仍为 True。条目 1 部分完成 + 条目 2/3/4 仍 placeholder。flag 语义是"ACTIVE 不安全"，不是"具体哪一项还在 fallback"
  - 测试：`tests/contracts/test_protocol_identity_gate.py` 13 个测试覆盖三个分支 × 四种 upstream 组合 + cheng_laoshi 行为回归
  - cheng_laoshi（`required_regime_compatibility=()`）行为完全不变；matched-control 测试和现有 packet 1.0/1.2 测试零修改
  - Checklist 进度：条目 1 🟡 partial（regime ✅，self_traits ⏳ packet 1.3'）；剩余条目 2/3/4 / 7 部分待后续
