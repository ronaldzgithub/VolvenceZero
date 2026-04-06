# EmoGPT Next-Gen — 数据契约文档

> Status: draft
> Version: 0.1
> Last updated: 2026-03-25
> Source: `docs/next_gen_emogpt.md`（R8, R11）、`docs/SYSTEM_DESIGN.md`

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

### 2.6 RuntimePlaceholderValue（缺失与禁用占位）

用于统一表示缺失 upstream 和禁用模块发布的 stub 快照。

```python
@dataclass(frozen=True)
class RuntimePlaceholderValue:
    reason: str
    expected_slot: str
    produced_by: str
    detail: str
```

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
    unavailable_fields: tuple[UnavailableField, ...]
    description: str
```

**阶段化 contract**：

- 当前稳定 contract：`surface_kind=FEATURE_SURFACE`，发布 `feature_surface` 与可选 `token_logits`
- 保守占位 contract：`surface_kind=PLACEHOLDER`，明确哪些字段 unavailable
- 未来增强 contract：`surface_kind=RESIDUAL_STREAM`，发布真实 `residual_activations`

**消费者**：Metacontroller、记忆系统、双轨学习层
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
```

**当前实现口径**：

- P08 已固定 `controller_state` 的 machine-readable shape
- 当前实现支持 `placeholder` / `heuristic` 两种可替换策略位点
- `active_abstract_action` 和 `description` 是可读输出，不作为 machine state 的唯一来源

**消费者**：编排器、双轨学习层、认知 Regime 层、评估体系
**发布频率**：每 turn

### 3.3 连续记忆系统 (Memory)

**Slot**: `memory`

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
    stratum: str
    tags: tuple[str, ...] = ()
    strength: float = 0.5

@dataclass(frozen=True)
class MemorySnapshot:
    # 按层级组织的记忆摘要
    transient_summary: str              # 瞬态工作状态摘要（模块自身生成）
    episodic_summary: str               # 会话情景状态摘要
    durable_summary: str                # 持久语义记忆摘要

    # 本轮检索到的相关记忆
    retrieved_entries: tuple[MemoryEntry, ...]

    # 统计信息
    total_entries_by_stratum: tuple[tuple[str, int], ...]  # (stratum, count) pairs
    pending_promotions: int             # 待提升的记忆数量
    pending_decays: int                 # 待衰减的记忆数量

    description: str                    # 模块自身生成的整体状态描述
```

**owner 规则**：

- 所有记忆写入必须通过 `MemoryWriteRequest` 形式进入 Memory owner API
- 消费者不得直接持有或修改 memory 内部存储结构
- 提升、衰减、部分重建的 pending 状态由 Memory owner 自身发布

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

@dataclass(frozen=True)
class DualTrackSnapshot:
    world_track: TrackState
    self_track: TrackState
    cross_track_tension: float          # 跨轨道张力（两轨目标冲突程度）
    description: str                    # 模块自身生成的状态描述
```

**当前实现口径**：

- P03 阶段先以结构化状态 owner 落地，不要求完整 temporal / evaluation / credit 全部接入
- `recent_credits` 当前可由 owner 发布为“最近重要状态信号”，后续再与正式 credit owner 对齐
- `controller_code` 当前允许是从已知状态压缩出的占位向量，而不是最终 learned controller code

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
    old_value_hash: str                 # 修改前值的哈希
    new_value_hash: str                 # 修改后值的哈希
    justification: str                  # 修改理由
    timestamp_ms: int
    is_reversible: bool                 # 是否可回滚

@dataclass(frozen=True)
class CreditSnapshot:
    recent_credits: tuple[CreditRecord, ...]
    recent_modifications: tuple[SelfModificationRecord, ...]
    cumulative_credit_by_level: tuple[tuple[str, float], ...]  # (level, sum) pairs
    description: str
```

**当前实现口径**：

- P06 先落地结构化信用记录和 gate audit，不执行真正的在线自修改
- `recent_modifications` 当前记录 allow / block 结果，作为审计轨迹和后续 reflection 输入
- `cumulative_credit_by_level` 先提供最小聚合，后续再扩展到更细粒度的长期统计

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
class RegimeSnapshot:
    active_regime: RegimeIdentity
    previous_regime: RegimeIdentity | None
    switch_reason: str                  # 切换原因（如有切换）
    candidate_regimes: tuple[tuple[str, float], ...]  # (regime_id, score) pairs
    turns_in_current_regime: int
    description: str
```

**当前实现口径**：

- P04 阶段已经提供结构化 regime identity 和 candidate scoring
- 当前选择逻辑基于 `memory`、`dual_track`、`evaluation` 的状态评分基线
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
```

**当前实现口径**：

- P05 阶段先提供 turn / session 两级的最小评估通路
- `turn_scores` 必须包含 evidence；`session_scores` 为当前 session 的聚合视图
- 告警先以结构化字符串对外发布，后续可升级为更细粒度 alert schema

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

@dataclass(frozen=True)
class ReflectionSnapshot:
    memory_consolidation: MemoryConsolidation
    policy_consolidation: PolicyConsolidation
    interaction_trace_summary: str                  # 交互轨迹摘要
    tensions_identified: tuple[str, ...]            # 识别到的张力
    lessons_extracted: tuple[str, ...]              # 提取的持久教训
    writeback_mode: str                             # disabled | proposal-only | apply
    review_required: bool
    description: str
```

**当前实现口径**：

- P07 默认以 `proposal-only` 运行，不直接修改 memory / credit / temporal owner 状态
- `memory_consolidation` 和 `policy_consolidation` 先表达提案和审计结果
- `review_required=True` 表示需要后续 gate / human / rollout 决策后才能放大范围

**消费者**：记忆系统、信用分配、Metacontroller、认知 Regime 层
**发布频率**：每会话后（异步）

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

class Module(ABC):
    @property
    @abstractmethod
    def slot_name(self) -> str:
        """快照 slot 标识，全局唯一"""

    @property
    @abstractmethod
    def owner(self) -> str:
        """模块唯一所有者标识"""

    @property
    @abstractmethod
    def value_type(self) -> type[Any]:
        """该 slot 对外发布的 value 类型"""

    @property
    def dependencies(self) -> tuple[str, ...]:
        """声明的直接上游依赖 slot"""

    @property
    def wiring_level(self) -> WiringLevel:
        """运行时接线级别"""

    @abstractmethod
    async def process(self, upstream: UpstreamDict) -> Snapshot:
        """
        接收上游快照，执行处理，返回自身快照。

        约束:
        - 只从带守卫的 upstream view 读取数据，不持有/import/调用其他模块
        - 返回的 Snapshot 必须是 frozen dataclass
        - 模块内部状态的描述由自身生成并打包到快照中
        """

    async def process_standalone(self, **kwargs) -> Snapshot:
        """
        独立调用模式（预训练/测试场景）。
        不依赖 upstream，直接接收必要参数。
        """
        raise NotImplementedError
```

### 4.3 编排器快照传播

```python
async def propagate(
    modules: list[Module],
    *,
    upstream: UpstreamDict | None = None,
    registry: SlotRegistry | None = None,
    recorder: EventRecorder | None = None,
    shadow_snapshots: MutableMapping[str, Snapshot] | None = None,
    session_id: str = "runtime",
    wave_id: str = "wave-0",
) -> UpstreamDict:
    """
    按顺序执行模块，收集快照。

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
                    ┌──────────────────────────────────────┐
                    │                                      │
substrate ──────────┼──→ temporal_abstraction ──┬──→ dual_track
                    │                           │        │
                    ├──→ memory ────────────────┤        │
                    │        │                  │        │
                    │        └──────────────────┼──→ regime
                    │                           │        │
                    └───────────────────────────┼──→ credit
                                                │        │
                                                └──→ evaluation
                                                         │
                                         (async) ──→ reflection
                                                         │
                                                    ┌────┴────┐
                                                    ▼         ▼
                                                 memory    credit
                                              (write-back) (update)
```

**依赖规则**：
- 每个模块只读取上游快照，不反向依赖
- `reflection` 是唯一的异步模块，会话后运行
- `reflection` 的产物通过正式 API 写回 `memory` 和 `credit`

**关于直接消费与间接消费**：上图展示的是**直接快照依赖**。Slot 注册表（第 6 节）中列出的消费者是**声明的直接消费者**——即模块在 `process()` 中从 upstream dict 读取的 slot。模块不通过中间模块间接获取数据，而是直接声明并读取所需的上游快照。

---

## 6. 快照 Slot 注册表

| Slot Name | Owner 模块 | Value 类型 | 发布频率 | 消费者 |
|-----------|-----------|-----------|----------|--------|
| `substrate` | SubstrateModule | SubstrateSnapshot | 每 token/turn | temporal_abstraction, memory, dual_track |
| `temporal_abstraction` | MetacontrollerModule | TemporalAbstractionSnapshot | 每 turn | orchestrator, dual_track, regime, evaluation |
| `memory` | MemoryModule | MemorySnapshot | 每 turn ~ 每会话 | orchestrator, temporal_abstraction, dual_track, regime, reflection |
| `dual_track` | DualTrackModule | DualTrackSnapshot | 每 turn | orchestrator, memory, credit, evaluation |
| `credit` | CreditModule | CreditSnapshot | 每 turn ~ 每会话 | orchestrator, memory, evaluation |
| `regime` | RegimeModule | RegimeSnapshot | 每 turn | orchestrator, temporal_abstraction, memory, evaluation |
| `evaluation` | EvaluationModule | EvaluationSnapshot | 每 turn ~ 每会话 | orchestrator, credit, gate |
| `reflection` | ReflectionModule | ReflectionSnapshot | 每会话后（异步） | memory, credit, temporal_abstraction, regime |

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
| `docs/prd.md` | 5.5 契约式运行时、6.1 模块间通信总线 |
| `.cursor/rules/ssot-module-boundaries.mdc` | 模块 SSOT + 快照隔离的编码规则 |
