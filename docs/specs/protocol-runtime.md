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

不是最终代码；是 spec 级的 schema 草案。落地时进 `vz-contracts`（跨 wheel 共享）。

```python
@dataclass(frozen=True)
class BehaviorProtocol:
    # === 标识 ===
    protocol_id: str                    # e.g. "growth_advisor:cheng_laoshi:v0.1.0"
    version: str
    source_kind: ProtocolSourceKind     # PDF_UPTAKE | TASK_DESCRIPTION | API | FIXTURE | ...
    source_locator: str                 # PDF path / API request id / fixture module path
    parent_protocol_id: str | None      # 继承（e.g. growth_advisor:cheng_laoshi 继承 growth_advisor:base）

    # === Frozen 部分（永不被 PE 修订）===
    identity_assertion: IdentityAssertion       # 该协议要求的 Identity Core 兼容性
    boundary_contracts: tuple[BoundaryContract, ...]  # 跨协议取并集；frozen
    activation_conditions: ActivationConditions       # 何种 context 下匹配；context-match 信号源

    # === Soft 部分（PE 可修订权重；ReflectionEngine 可提议替换）===
    strategy_priors: tuple[StrategyPrior, ...]
    temporal_arc: TemporalArc                  # 阶段定义基于关系状态，不是日历天数
    initial_drives: tuple[DriveSpec, ...]      # 仅用于初始化 vitals；运行时 drive 由 VitalsModule 拥有

    # === PE 信号定义（必填；R-PE 入口）===
    success_signals: tuple[SuccessSignal, ...]
    failure_signals: tuple[FailureSignal, ...]

    # === Lifecycle ===
    revision_log: tuple[ProtocolRevision, ...]
    review_status: ReviewStatus                # DRAFT | SHADOW | ACTIVE | RETIRED


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
    measurable_via: SignalSource               # USER_REPLY_LATENCY | USER_INITIATIVE_QUESTION | DRIVE_HOMEOSTASIS_HOLD | ...
    expected_value_range: tuple[float, float]
    weight_in_pe: float


@dataclass(frozen=True)
class FailureSignal:
    """协议自带的 PE readout：失败是什么样的。"""

    signal_id: str
    description: str
    measurable_via: SignalSource
    threshold: float
    weight_in_pe: float


@dataclass(frozen=True)
class ActiveMixtureSnapshot:
    """ProtocolRuntime 对外发布的核心快照。"""

    snapshot_id: str
    tick_index: int
    active_protocols: tuple[ActiveProtocolEntry, ...]    # protocol_id + activation_weight + 当前 phase
    boundary_union: tuple[BoundaryContract, ...]         # 取并集后的边界
    identity_gate_traits: tuple[str, ...]                # 当前 Identity Core 的 trait set
    revision_fingerprint: str                            # 用于 fingerprint guard


@dataclass(frozen=True)
class ActiveProtocolEntry:
    protocol_id: str
    activation_weight: float
    current_phase_id: str | None
    activation_reasons: tuple[ActivationReason, ...]     # context_match / pe_history / identity_gate 各贡献了多少（drive_coupling deferred）
```

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

**Packet 1.5a partial 实现**：

- α 临时硬定 1.0，β 仍 0（α/β 学习留 packet 1.5b/c）
- `context_match_i = Σ signal.weight × signal_is_firing(signal, upstream)`，3 个 kernel-side detector：
  - `INTERLOCUTOR_ZONE_TRANSITION`：interlocutor_state 任一 zone bool 为 True 时 fire
  - `RUPTURE_KIND_FIRED`：rupture_state 解析出非空 `rupture_kind` 时 fire
  - `BOUNDARY_VIOLATION_FIRED`：boundary_policy 决策含非空 `trigger_reasons` 时 fire
- `USER_DROPOUT_OBSERVED` 占位返 False（待 1.5a' 接 dialogue_trace）
- `DRIVE_HOMEOSTASIS_HOLD/BREACH` 永久 defer（vitals 跨层边界，packet 1.0.1）
- 当所有 eligible 协议 `max(score) == 0`（cheng_laoshi 默认形态：`activation_conditions.context_match_signals` 为空），算法回落到 `equal_weight_with_floor`，`ActivationReason.kind = EQUAL_WEIGHT_FALLBACK`；任一 score > 0 → 切到 `softmax`，`kind = CONTEXT_MATCH`，`detail` 列出 `signals_fired=[...]` 用于审计

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

### 信号源清单（典型）

| SignalSource | 含义 | 既有 owner |
|---|---|---|
| `USER_REPLY_LATENCY` | 回复延迟分布 | 来自 dialogue_trace |
| `USER_REPLY_LENGTH` | 回复长度分布 | 来自 dialogue_trace |
| `USER_INITIATIVE_QUESTION` | 用户主动提问频率 | 来自 dialogue_trace + interlocutor_state |
| `USER_DROPOUT` | 用户停止回复（隔多久） | 来自 dialogue_trace |
| `DRIVE_HOMEOSTASIS_HOLD` | 某 drive 是否保持在稳态带 | 来自 VitalsModule |
| `RUPTURE_KIND_FIRED` | rupture_state 是否触发 | 来自 §17A rupture_state owner |
| `BOUNDARY_VIOLATION_FIRED` | 边界契约是否被触发 | 来自 BoundaryPolicy |
| `INTERLOCUTOR_ZONE_TRANSITION` | 12 轴 zone 跨越 | 来自 §17C interlocutor_state |
| `COMMITMENT_FULFILLED / BROKEN` | AAC commitment 状态 | 来自 §14 aac-commitment-lifecycle |
| `REGIME_TRANSITION` | R14 regime 切换 | 来自 cognitive_regime owner |

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

### 修订类型

| 修订类型 | 触发条件 | Review 级别 |
|---|---|---|
| `WeightDecay` | strategy_prior 在 phase X 持续 PE 失败 N 次 | L1（自动） |
| `WeightReinforce` | strategy_prior 持续 PE 成功 N 次 | L1（自动） |
| `NewStrategyPrior` | 反思发现重复成功模式协议未覆盖 | L2（lint + spot check） |
| `BoundaryRefinement` | 边界 trigger_reasons 误报率高 | L3（人工） |
| `IdentityClarification` | identity_violation PE 反复出现 | L4（人工 + admin） |
| `ProtocolRetirement` | 协议整体 PE 持续负 | L3-L4 |

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
| 2. `α / β` 由 metacontroller 学（PE history 接入） | 全 0，等价于 α=1·context + β=0；context_match 已 partial 真实化（packet 1.5a），PE-utility 与 α/β 学习仍是 0 | 1.5b–c | runtime fail-loud |
| 3. typed `context_match` 接入（interlocutor zone / regime / retrieval / PE） | 🟡 **partial（packet 1.5a）**：3 个 kernel-side detector 上线（`INTERLOCUTOR_ZONE_TRANSITION` / `RUPTURE_KIND_FIRED` / `BOUNDARY_VIOLATION_FIRED`），合成 `score = Σ weight × firing` 并参与 softmax；空信号集落回 equal-weight。`USER_DROPOUT_OBSERVED` 需 dialogue_trace inspection（待 1.5a' 接入）；`DRIVE_HOMEOSTASIS_*` 永久 deferred（vitals 不在 kernel 图里，packet 1.0.1 决议）。完全闭合还需补 retrieval-policy / regime-direct 信号源 + α 学习 | 1.5a → 1.5a' → 1.5b | runtime fail-loud |
| 4. 互斥协议仲裁不再用 lexicographic id | 当前用 lexicographic（占位） | 1.5c | runtime fail-loud |
| 5. 至少 1 个下游 consumer 通过 matched-control dual-run 测试 | ✅ packet 1.2 后续：boundary 路径 matched-control（vertical compile vs protocol compile，hint 内容除 lineage 前缀外 byte-equivalent） | 1.2 后续 | `tests/contracts/test_protocol_boundary_matched_control.py` |
| 6. `boundary_union` 字段定位明确（IDs vs 完整 contracts） | ✅ 1.2 锁定选择 A：`boundary_union_ids: tuple[str, ...]`（IDs only） | 1.2 | schema 决议 + contract test |
| 7. `BehaviorProtocol → application owners` compile 路径打通 | ✅ 1.2 boundary + 1.3b strategy + 1.4a knowledge + 1.4b case（**全 4/4 完成**） | 1.2 / 1.3b / 1.4a / 1.4b | implementation + matched-control tests |

**Packet 1.0.1 引入 runtime guard**：`ProtocolRegistryModule.__init__` 检测 `wiring_level=ACTIVE` 时，若 ActivationController 仍在 fallback 模式（条件 1-4 任一未满足），构造时 `raise ContractViolationError`。这把 docstring 注释升级为可执行约束。

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

## Open Questions

以下问题在阶段 1 实现前需要回答：

1. **协议继承的 merge 语义**：`parent_protocol_id` 链式继承时，strategy_priors 和 boundary_contracts 的 merge 规则（覆盖 / 并集 / 加权混合）需要明确
2. **跨 lifeform 共享协议**：同一 BehaviorProtocol 是否可被多个 lifeform 复用？revision_log 是 per-lifeform 还是 per-protocol？建议：协议本体共享，revision_log per-lifeform
3. **协议版本兼容**：`v0.1.0` → `v0.2.0` 升级时，是否要重放历史 revision_log？建议：不重放，新版本起 fresh revision_log
4. **identity_gate 量化**：identity 兼容是 0/1 硬过滤，还是连续值？建议先 0/1，未来可演化为连续
5. **TaskUptake 的成本**：LLM-辅助抽取一份 PDF 可能要几万 token；是 turn-time 操作还是离线操作？建议离线（背景任务）
6. **PE 信号的语义对齐**：不同协议声明的 success_signal 可能引用不同 SignalSource；如何避免 SignalSource 注册表碎片化？建议：SignalSource 是封闭枚举，新增 source 必须先扩枚举（与 §17A RuptureKind 同理）
7. **协议冲突的优先级权重**：`α/β` 的初值如何确定？建议：从首个 successful protocol 的 PE 历史 bootstrap，否则用 (0.5, 0.5) 默认
8. **协议数量的上限**：active mixture 同时激活多少协议合理？建议：硬上限 8（避免 softmax 摊薄到无意义），floor 协议优先

## 术语对齐速查表

| 本 spec 术语 | 既有系统 / 论文术语 |
|---|---|
| BehaviorProtocol | Task Set + Schema + Script + Production Rules + Boundary Contracts 的捆绑 |
| Active Mixture | PFC current task representation |
| Protocol Registry | Long-term task set library |
| Activation Controller | Norman & Shallice Supervisory Attentional System |
| Identity Core | R7 Self 轨道 + R14 持久身份 |
| Boundary Union | morality / value commitments |
| context_match / PE_utility / identity_gate | active inference policy posterior 的三个 kernel-side 组分（drive_value deferred，见 §调度） |
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
