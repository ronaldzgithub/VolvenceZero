# Profile Registry + Capability Wiring Spec

> Status: draft
> Last updated: 2026-05-13
> 对应需求: R8, R11, R15（架构 packet A1 + A3 同步设计）
> 对应改造路线图: [`docs/moving forward/experiment-arch-uplift.md`](../moving%20forward/experiment-arch-uplift.md) §2 A1 + A3
> 对应 plan：架构改造 plan T1 spec-first 前置

---

## 要解决的问题

阶段 C 实验承载力需要：

- **声明侧**：用 declarative `ProfileSpec → Capability bundle` 替代 [`packages/vz-runtime/src/volvence_zero/agent/dialogue/_legacy.py`](../../packages/vz-runtime/src/volvence_zero/agent/dialogue/_legacy.py) 中 `build_standard_dialogue_runner` 11 个 `if profile_label == "X"` 硬编码分支。
- **执行侧**：把 `WiringLevel` 从 module 级粒度（`TemporalModule.default_wiring_level = SHADOW`）下沉到 capability 级粒度，使同一 module 内不同 capability 可独立切换 SHADOW / ACTIVE / DISABLED。

A1（声明侧）与 A3（执行侧）本质是同一抽象的两端：profile 声明 capability 组合，capability 决定 module 内部走哪条 shadow path。**必须在同一份 spec 里一次设计完毕**，否则会陷入"profile 整体切换但 module 不响应"的双 truth 状态。

---

## 关键不变量

1. **profile = capability bundle**：profile_label 不再硬编码行为，只是一组 capability 名称的不可变集合。
2. **capability 是正交单位**：每个 capability 修改一个明确的 `applies_to_owner`；两个 capability 修改同一 owner 时必须显式声明 `conflicts_with` 或合并为一个。
3. **依赖必须是 DAG**：`requires` 形成的依赖图无环；启动时编译期校验，违反即 fail-loudly。
4. **现有 11 个 profile 行为 byte-equivalent**：迁移完成时，11 个内置 ProfileSpec 在新 dispatch 下产出的 dialogue benchmark `metric_means` 必须与 legacy if-elif 路径 delta == 0（或在 float noise 内）。
5. **capability map 默认为空 = 等价于现有 module-level wiring**：A3 的扩展不改变现有不带 capability 切换的 module 行为。
6. **fail-loudly**：profile 引用不存在 capability、capability 引用不存在 owner、依赖图有环 → 启动时 raise；禁止静默忽略（[`no-swallow-errors-no-hasattr-abuse.mdc`](../../.cursor/rules/no-swallow-errors-no-hasattr-abuse.mdc)）。
7. **registry 是 SSOT**：迁移完成后，legacy 11 个 if-elif 分支必须移除（DISABLED ≥1 release cycle 后清理）。

---

## 工程挑战

- 设计 ProfileSpec 与 capability 的最小完备数据类型集
- 把现有 11 个 profile 完整拆解为 capability bundle（不增不减）
- 把 module-level `default_wiring_level` 扩展为 capability-level `CapabilityWiring`，同时保持现有 module 行为不变（capability map 默认为空时）
- 编译期校验 DAG + 不存在 capability / owner 引用
- 与 `FinalRolloutConfig` 现有扁平 flag 的兼容迁移（capability map 嵌套化但默认空）

---

## 算法候选

不涉及。Profile registry / capability wiring 是工程架构层基础设施，不对应 NL/ETA 算法。

---

## 接口契约

### A1.1 ProfileCapability

声明"一个 capability 是什么"：

```python
@dataclass(frozen=True)
class ProfileCapability:
    name: str                                # 全局唯一，kebab-case，例如 "cms-atlas-titans-uplift"
    applies_to_owner: str                    # 受影响的 owner slot 名（与 RuntimeModule.slot_name 对齐）
    flag_overrides: Mapping[str, Any] = field(default_factory=dict)
                                             # owner module 启动参数覆盖
                                             # key 为 module 构造参数名或 module 暴露的 flag 路径
    wiring_overrides: Mapping[str, "WiringLevel"] = field(default_factory=dict)
                                             # 同 module 内 sub-capability 的 wiring level 覆盖
                                             # key 为 capability-name-within-owner
    requires: tuple[str, ...] = ()           # 依赖的其他 capability name
    conflicts_with: tuple[str, ...] = ()     # 互斥的 capability name
    description: str = ""                    # 给 spec / PR review 看的说明
```

**字段语义**：

- `name`：profile 引用 capability 时用的 key；全局唯一；kebab-case。
- `applies_to_owner`：必须对应 [`docs/DATA_CONTRACT.md`](../DATA_CONTRACT.md) §6 注册的 slot 名（如 `temporal_abstraction` / `credit` / `evaluation` / `memory` 等）。一个 capability 只能 apply 到一个 owner（否则违反 R8 SSOT）。
- `flag_overrides`：传递给 owner module 构造器或 `FinalRolloutConfig` 的 flag override。**禁止**让 capability 直接 mutate 其他 owner 的状态。
- `wiring_overrides`：sub-capability 级 WiringLevel 覆盖（详见 A3 capability wiring）。
- `requires`：依赖图边；必须能构成 DAG。
- `conflicts_with`：互斥关系；同一 profile 中不允许同时引用互斥 capability。

### A1.2 ProfileSpec

声明"一个 profile 由哪些 capability 组成"：

```python
@dataclass(frozen=True)
class ProfileSpec:
    label: str                               # profile_label，与 build_standard_dialogue_runner 入参对齐
    capabilities: tuple[str, ...]            # capability name 列表，顺序仅用于日志
    base_profile: str = "pe-eta"             # 继承的 base profile；"pe-eta" 是隐式 baseline
    aliases: tuple[str, ...] = ()            # 兼容性别名，例如 "eta-no-pe" 是 "pe-drive-off" 的别名
    description: str = ""                    # 给 spec / PR review 看的说明
```

**字段语义**：

- `label`：等价于现有 `profile_label`；CI / benchmark / SHADOW evidence 文档继续引用此 label。
- `capabilities`：capability bundle；空 tuple = 与 base_profile 完全相同。
- `base_profile`：继承的 base；通常是 `"pe-eta"`，表示 "baseline 之上加这些 capability"。
- `aliases`：迁移期保留的别名，例如 `pe-drive-off` 与 `eta-no-pe` 是同一 ProfileSpec 的两个 label。
- 不持有 `flag_overrides`：所有行为差异都通过 capability 表达；profile 只是 capability 组合。

### A1.3 ProfileRegistry

全局 registry，编译期校验：

```python
class ProfileRegistry:
    """Global registry for ProfileSpec + ProfileCapability declarations."""

    def __init__(self) -> None:
        self._capabilities: dict[str, ProfileCapability] = {}
        self._profiles: dict[str, ProfileSpec] = {}
        self._aliases: dict[str, str] = {}  # alias -> canonical label

    def register_capability(self, capability: ProfileCapability) -> None: ...
    def register_profile(self, profile: ProfileSpec) -> None: ...
    def resolve_profile(self, label: str) -> ResolvedProfile: ...
    def list_profiles(self) -> tuple[str, ...]: ...
    def list_capabilities(self) -> tuple[str, ...]: ...
    def validate(self) -> None: ...  # 编译期一次性校验所有 invariant
```

校验由 `validate()` 完成（在 module import 时调用）：

1. 所有 `ProfileSpec.capabilities` 引用的 capability name 都已注册。
2. 所有 `ProfileCapability.applies_to_owner` 对应的 owner 在 `FinalRolloutConfig` 字段中存在。
3. `requires` 图无环（DAG，深度优先 cycle detect）。
4. 同一 profile 的 capability 集合不存在 `conflicts_with` 关系。
5. `aliases` 不能与任何 canonical `label` 冲突。

任一项违反 → raise `ProfileRegistryViolationError`。

### A1.4 ResolvedProfile

resolve 后的运行时表示：

```python
@dataclass(frozen=True)
class ResolvedProfile:
    label: str
    capabilities: tuple[ProfileCapability, ...]    # 按 requires 拓扑排序
    merged_flag_overrides: Mapping[str, Any]       # 全部 capability 的 flag_overrides union
    merged_wiring_overrides: Mapping[str, "CapabilityWiring"]  # 详见 A3.2

    def apply_to_config(self, base: "FinalRolloutConfig") -> "FinalRolloutConfig":
        """Return new FinalRolloutConfig with flag/wiring overrides applied."""
```

`apply_to_config` 是纯函数，返回新的 frozen `FinalRolloutConfig`；禁止原地修改。

---

### A3.1 CapabilityWiring

声明"某 owner 内某 sub-capability 的 wiring level"：

```python
@dataclass(frozen=True)
class CapabilityWiring:
    capability_name: str          # 与 ProfileCapability.name 一致
    owner: str                    # 与 applies_to_owner 一致
    wiring_level: WiringLevel     # SHADOW / ACTIVE / DISABLED
    description: str = ""
```

### A3.2 RuntimeModule.capabilities

扩展 [`packages/vz-contracts/src/volvence_zero/runtime/kernel.py:409`](../../packages/vz-contracts/src/volvence_zero/runtime/kernel.py) `RuntimeModule` 基类：

```python
class RuntimeModule(ABC, Generic[ValueT]):
    slot_name: ClassVar[str]
    owner: ClassVar[str]
    value_type: ClassVar[type[Any]]
    dependencies: ClassVar[tuple[str, ...]] = ()
    default_wiring_level: ClassVar[WiringLevel] = WiringLevel.ACTIVE

    # A3 新增字段：sub-capability 级 wiring map
    capabilities: ClassVar[Mapping[str, WiringLevel]] = MappingProxyType({})

    def __init__(
        self,
        *,
        wiring_level: WiringLevel | None = None,
        capability_overrides: Mapping[str, WiringLevel] | None = None,
    ) -> None:
        self._wiring_level = wiring_level or self.default_wiring_level
        # capability_overrides 来自 ResolvedProfile.merged_wiring_overrides
        # 覆盖 ClassVar 的 default
        self._capability_wiring = {**self.capabilities, **(capability_overrides or {})}
        self._version = 0

    def capability_wiring(self, capability_name: str) -> WiringLevel:
        """Return wiring level for a sub-capability declared on this module.

        Falls back to module-level wiring_level if capability not declared.
        """
        return self._capability_wiring.get(capability_name, self._wiring_level)

    def capability_active(self, capability_name: str) -> bool:
        return self.capability_wiring(capability_name) == WiringLevel.ACTIVE

    def capability_shadow(self, capability_name: str) -> bool:
        return self.capability_wiring(capability_name) == WiringLevel.SHADOW
```

**关键不变量（A3.2）**：

- `capabilities` ClassVar 默认为空 MappingProxyType ⇒ **现有所有 module 行为完全不变**（默认空时 `capability_wiring(x)` 永远返回 module-level wiring_level）。
- module 内部判断 sub-capability 是否走 shadow path 时使用 `self.capability_active(name)` / `self.capability_shadow(name)`，**禁止**用 `hasattr` 或 dict default 隐藏不存在的 capability。
- 未在 module `capabilities` ClassVar 中声明但被 ResolvedProfile 引用的 capability → 启动时 fail-loudly（在 ProfileRegistry.validate 阶段捕获）。

### A3.3 FinalRolloutConfig 扩展

[`packages/vz-runtime/src/volvence_zero/integration/final_wiring.py:171`](../../packages/vz-runtime/src/volvence_zero/integration/final_wiring.py) `FinalRolloutConfig` 保留现有扁平 owner-level wiring 字段（向后兼容），并新增 nested capability override map：

```python
@dataclass(frozen=True)
class FinalRolloutConfig:
    # 现有 owner-level wiring（保持不变）
    substrate: WiringLevel = WiringLevel.ACTIVE
    memory: WiringLevel = WiringLevel.ACTIVE
    # ...（其余 30+ 字段保持不变）

    # A3 新增：nested capability wiring map
    # outer key = owner slot name
    # inner key = capability name within that owner
    capability_wirings: Mapping[str, Mapping[str, WiringLevel]] = field(
        default_factory=lambda: MappingProxyType({})
    )
```

**默认行为不变性**：`capability_wirings` 默认为空 dict ⇒ 现有所有 profile / runtime 行为与改造前完全一致。

---

## 11 个现有 Profile 的 Capability 拆解（迁移基准）

下表是 A1 实施（T5）的迁移基准。**11 个 profile 在新 schema 下产出的 dialogue benchmark metric_means 必须与改造前 byte-equivalent**。

### Profile 1：`pe-eta`（baseline）

```python
ProfileSpec(label="pe-eta", capabilities=(), description="canonical baseline")
```

- capability bundle 为空 = 直接使用 `FinalRolloutConfig` 默认 + `AgentSessionRunner` 默认 + `_benchmark_joint_schedule()` + `allow_live_substrate_mutation=True`。

### Profile 2：`atlas-titans-cms-uplift`

```python
ProfileSpec(
    label="atlas-titans-cms-uplift",
    capabilities=("cms-atlas-titans-uplift",),
    description="CMS uplift SHADOW-evidence profile",
)

ProfileCapability(
    name="cms-atlas-titans-uplift",
    applies_to_owner="memory",
    flag_overrides={
        "cms_pe_features_enabled": True,
        "cms_replay_window_size": 8,
    },
    description="See docs/specs/cms-atlas-titans-uplift.md",
)
```

注：现有 `atlas-titans-cms-uplift` 还显式构造 `FullLearnedTemporalPolicy()` 后传给 `memory_store`；这是因为 `MemoryStore` 需要 `latent_dim`，与 capability 无关，属于 base profile 的默认 wiring 一部分（参考 [`_legacy.py:7878-7894`](../../packages/vz-runtime/src/volvence_zero/agent/dialogue/_legacy.py)）。在 capability 的 `flag_overrides` 中只暴露 CMS 相关 flag。

### Profile 3：`pe-eta-online-only`

```python
ProfileSpec(
    label="pe-eta-online-only",
    capabilities=("reflection-proposal-only", "rare-heavy-off"),
)

ProfileCapability(
    name="reflection-proposal-only",
    applies_to_owner="reflection",
    flag_overrides={"reflection_mode": "WritebackMode.PROPOSAL_ONLY"},
)

ProfileCapability(
    name="rare-heavy-off",
    applies_to_owner="substrate_self_mod",
    flag_overrides={"rare_heavy_enabled": False},
)
```

### Profile 4：`pe-eta-no-writeback`

```python
ProfileSpec(
    label="pe-eta-no-writeback",
    capabilities=("reflection-proposal-only",),
)
```

### Profile 5：`pe-eta-no-rare-heavy`

```python
ProfileSpec(
    label="pe-eta-no-rare-heavy",
    capabilities=("rare-heavy-off",),
)
```

### Profile 6：`pe-eta-no-semantic-label`

```python
ProfileSpec(
    label="pe-eta-no-semantic-label",
    capabilities=("no-semantic-label-temporal-policy",),
)

ProfileCapability(
    name="no-semantic-label-temporal-policy",
    applies_to_owner="temporal_abstraction",
    flag_overrides={
        "world_temporal_policy_class": "_NoSemanticLabelTemporalPolicy",
        "self_temporal_policy_class": "_NoSemanticLabelTemporalPolicy",
    },
)
```

### Profile 7：`pe-eta-no-reflection-cache`

```python
ProfileSpec(
    label="pe-eta-no-reflection-cache",
    capabilities=("no-reflection-cache-temporal-policy",),
)

ProfileCapability(
    name="no-reflection-cache-temporal-policy",
    applies_to_owner="temporal_abstraction",
    flag_overrides={
        "world_temporal_policy_class": "_NoReflectionCacheTemporalPolicy",
        "self_temporal_policy_class": "_NoReflectionCacheTemporalPolicy",
    },
    conflicts_with=("no-semantic-label-temporal-policy",),  # 互斥：两者都覆盖 temporal_policy_class
)
```

### Profile 8：`pe-eta-pe-readout-only`

```python
ProfileSpec(
    label="pe-eta-pe-readout-only",
    capabilities=("pe-readout-only",),
)

ProfileCapability(
    name="pe-readout-only",
    applies_to_owner="prediction_error",
    flag_overrides={
        "joint_schedule_ssl_interval": 1,
        "joint_schedule_rl_interval": 2,
        "external_prediction_error_drive": False,
        "prediction_error_readout_only": True,
        "primary_prediction_error_dominance_enabled": False,
    },
)
```

### Profile 9：`pe-drive-off`（别名 `eta-no-pe`）

```python
ProfileSpec(
    label="pe-drive-off",
    capabilities=("pe-drive-off",),
    aliases=("eta-no-pe",),
)

ProfileCapability(
    name="pe-drive-off",
    applies_to_owner="prediction_error",
    flag_overrides={
        "joint_schedule_ssl_interval": 1,
        "joint_schedule_rl_interval": 2,
        "external_prediction_error_drive": False,
        "allow_live_substrate_mutation": False,
    },
    conflicts_with=("pe-readout-only",),
)
```

### Profile 10：`timescale-off`

```python
ProfileSpec(
    label="timescale-off",
    capabilities=("timescale-off",),
)

ProfileCapability(
    name="timescale-off",
    applies_to_owner="memory",
    flag_overrides={
        "memory_store_nested_profile": False,
        "joint_schedule_ssl_interval": 1,
        "joint_schedule_rl_interval": 2,
        "allow_live_substrate_mutation": False,
    },
)
```

### Profile 11：`eta-off`（别名 `heuristic-baseline` ※ 行为不完全相同，见说明）

```python
ProfileSpec(
    label="eta-off",
    capabilities=("eta-off",),
)

ProfileSpec(
    label="heuristic-baseline",
    capabilities=("heuristic-baseline",),
)

ProfileCapability(
    name="eta-off",
    applies_to_owner="temporal_abstraction",
    flag_overrides={
        "temporal_policy_class": "LearnedLiteTemporalPolicy",
        "passive_joint_loop": True,
        "reflection_wiring": "WiringLevel.DISABLED",
        # joint_schedule: ssl=0, rl=0, PE thresholds=999.0
        "joint_schedule_ssl_interval": 0,
        "joint_schedule_rl_interval": 0,
        "joint_schedule_pe_thresholds_disabled": True,
        "rare_heavy_enabled": False,
        "external_prediction_error_drive": False,
        "allow_live_substrate_mutation": False,
    },
)

ProfileCapability(
    name="heuristic-baseline",
    applies_to_owner="temporal_abstraction",
    flag_overrides={
        "temporal_policy_class": "HeuristicTemporalPolicy",
        # 其余 flag 与 eta-off 相同
        ...
    },
    conflicts_with=("eta-off",),
)
```

**重要说明**：当前 `_legacy.py:7969-8001` 把 `eta-off` 与 `heuristic-baseline` 合并到同一分支但只通过 `temporal_policy` 不同区分。在新 schema 下应为两个独立 ProfileSpec（避免别名歧义；两者 `temporal_policy_class` 不同，不是同义别名）。

---

## ResolvedProfile 应用到 module 的流程

```mermaid
flowchart LR
    L["profile_label (str)"]
    R[ProfileRegistry]
    RP[ResolvedProfile]
    C[FinalRolloutConfig]
    M[RuntimeModule instances]

    L --> R
    R -->|resolve| RP
    RP -->|merge with base| C
    C -->|capability_overrides| M
```

1. `build_standard_dialogue_runner(profile_label="X")` 入口
2. `ProfileRegistry.resolve_profile("X")` 返回 `ResolvedProfile`
3. `ResolvedProfile.apply_to_config(FinalRolloutConfig())` 返回新 config，包含 capability flag 合并 + capability_wirings 嵌套 map
4. `AgentSessionRunner.__init__` 接收 config，构造各 module 时把 `config.capability_wirings.get(module.slot_name, {})` 传入 `capability_overrides=`
5. module 内部用 `self.capability_active("foo")` 决定 sub-capability shadow path

---

## 错误处理与 fail-loudly

以下情况必须在启动期 raise（不允许运行时静默回退）：

- profile_label 不存在：`raise ProfileRegistryViolationError(f"unknown profile: {label}")`
- profile 引用了未注册 capability：`raise ProfileRegistryViolationError(...)`
- capability 引用了未注册 owner（不在 `FinalRolloutConfig` 字段集）：`raise ProfileRegistryViolationError(...)`
- 依赖图有环：`raise ProfileRegistryViolationError(f"cycle in capability requires: ...")`
- 同一 profile 含互斥 capability：`raise ProfileRegistryViolationError(f"conflicts_with violated: ...")`
- module `capabilities` ClassVar 不包含 ResolvedProfile 引用的 capability：`raise ProfileRegistryViolationError(...)`
- alias 与 canonical label 冲突：`raise ProfileRegistryViolationError(...)`

捕获异常时不允许使用 bare `except` 或 `except Exception` 静默吞掉（[`no-swallow-errors-no-hasattr-abuse.mdc`](../../.cursor/rules/no-swallow-errors-no-hasattr-abuse.mdc)）。

---

## 迁移协议（A1 + A3 实施时遵循）

### 阶段 1：注册 + 双跑（registry SHADOW）

- 新建 `packages/vz-runtime/src/volvence_zero/agent/profile_registry.py`
- 注册 11 个内置 ProfileSpec 与所有相关 ProfileCapability
- 注册 `RuntimeModule.capabilities` ClassVar 默认为空 dict（不影响现有 module）
- `build_standard_dialogue_runner` **保留**现有 11 个 if-elif 分支为正式 upstream
- CI 增加双跑 contract test：对每个内置 profile_label，分别用 registry-first 和 legacy if-elif 构造 `AgentSessionRunner`，对比 `AgentSessionRunner` 内部 config（不跑 benchmark；仅结构层面对照）

**Done 标志（阶段 1）**：双跑 contract test 全 PASS；现有 dialogue paper-suite 行为完全不变（registry 不进 dispatch）。

### 阶段 2：registry-first dispatch（registry ACTIVE，legacy 仍可用）

- `build_standard_dialogue_runner` 改为：先查 registry；registry 命中则使用 registry 路径；未命中则 fallback 到 legacy if-elif（用于迁移期容错）
- 新增 SHADOW evidence：dialogue paper-suite 5 seeds × 4 cases × 11 profile，对比 registry-first 与 legacy 路径的 `metric_means`
- delta == 0（或在 float noise 内）才允许进入阶段 3

**Done 标志（阶段 2）**：11 profile metric_means byte-equivalent；SHADOW evidence 文档落在 `docs/specs/profile-registry-shadow-evidence-<date>.md`（沿用 `cms-atlas-titans-uplift-shadow-evidence-*.md` 模板）。

### 阶段 3：legacy DISABLED + 阶段 4 清理

- registry 成为 SSOT；legacy 11 个 if-elif 分支保留 ≥1 release cycle 作为 fallback，但启动时 emit warning（"legacy dispatch used, capture this in PR"）
- ≥1 release cycle 稳定期 + 现有 dialogue paper-suite + ETA strong-proof 全 PASS 后，进入 T16 cleanup：移除 legacy if-elif 分支。

---

## 与既有 spec / 规则的关系

- 扩展 [`docs/specs/contract-runtime.md`](contract-runtime.md) §WiringLevel + §模块基类：本 spec 新增 `RuntimeModule.capabilities` ClassVar 与 `capability_wiring()` 方法是 RuntimeModule 基类的最小扩展。
- 兼容 [`docs/DATA_CONTRACT.md`](../DATA_CONTRACT.md) §6：profile / capability **不直接**注册 slot；slot 注册仍是 DATA_CONTRACT §6 的职责。
- 与 [`docs/moving forward/experiment-arch-uplift.md`](../moving%20forward/experiment-arch-uplift.md) §2.A1 + §2.A3 + §8 风险 1 一致：本 spec 实现"A1 + A3 同 spec 设计"以缓解"实施两遍"风险。
- 不引入新规则；遵守现有 [`.cursor/rules/ssot-module-boundaries.mdc`](../../.cursor/rules/ssot-module-boundaries.mdc) / [`first-principles-not-patches.mdc`](../../.cursor/rules/first-principles-not-patches.mdc) / [`no-swallow-errors-no-hasattr-abuse.mdc`](../../.cursor/rules/no-swallow-errors-no-hasattr-abuse.mdc)。

---

## Done 检查（spec 评审）

- [ ] ProfileCapability / ProfileSpec / ProfileRegistry / ResolvedProfile / CapabilityWiring 数据类型 schema 完整
- [ ] 11 个现有 profile（含 `eta-no-pe` 别名 + `heuristic-baseline`）全部能在新 schema 下表达，capability bundle 拆解可还原 legacy 行为
- [ ] 迁移协议三阶段（注册 + 双跑 → registry-first + SHADOW evidence → DISABLED + cleanup）逻辑闭环
- [ ] 与现有 [`docs/specs/contract-runtime.md`](contract-runtime.md) / [`docs/DATA_CONTRACT.md`](../DATA_CONTRACT.md) 不产生冲突
- [ ] fail-loudly 错误清单覆盖所有可能违反路径
