# Affordance 体系 Spec

> Status: draft
> Last updated: 2026-04-29
> 对应需求: R3, R4, R8, R10, R11, R15
> 来源: `docs/implementation/13_emogpt_prd_alignment_upgrade.md` Gap 1

## 要解决的问题

VolvenceZero 当前能消费"工具调用结果"事件（`semantic_events_from_tool_result`），但没有"AI 能调用什么"的注册表。这意味着：

- lifeform 不能主动选择并执行外部能力，只能由 host 把结果**喂回来**
- coding vertical 想真正做"工程结对"必须读文件、grep、跑测试 —— **缺工具调用通道**是它当前的硬阻塞
- 不同 vertical（companion / coding / 未来的医疗 / 教育）需要的能力差异巨大，但**底层契约**应当一致

EmoGPT v4 PRD §9 用 `AffordanceDescriptor` + 单注册表 + 4 渲染器解决这件事。本 spec 把它落到 VolvenceZero 第一性下：

- Affordance 选择**不是**新硬编码路由层，而是 `temporal_abstraction` 抽象动作的一类候选（R3 / R4 直接落地）
- 描述符 schema 在 `vz-contracts`（多 vertical 共享）
- 注册表与渲染器在新 wheel `lifeform-affordance`（lifeform 层，不进 kernel）
- 工具结果回流走**已有**的 `BrainSession.submit_tool_result` 通道，无新数据通道
- 在 Environment Interface 口径下，affordance 是 `Act` 面的一类可学习、有界环境作用能力；它必须声明 prediction / safety / cost / outcome return channel。

## 关键不变量

- `AffordanceDescriptor` 是 frozen dataclass，必须满足：`when_to_use ≥ 50 char` / `when_not_to_use ≥ 50 char` / `parameters_schema` 是合法 JSON Schema dict
- `AffordanceRegistry` 是 lifeform 进程内单写者，启动时 atomic 写、运行时只读
- Affordance **选择**由 metacontroller 在 z_t 空间学；**禁止**在 prompt_planner / response_synthesizer / 任何 owner 内出现 `if name == "read_file"` 这类硬编码路由
- 每个 affordance 必须声明 `safety_model.requires_user_confirmation` / `cost_model`；ModificationGate 在执行前据此放行或拒绝
- `lifeform-affordance` wheel **不可**被任何 `vz-*` wheel import；CI 由 `tests/contracts/test_import_boundaries.py` 强制
- Affordance 调用产生的副作用结果**只能**通过 `BrainSession.submit_tool_result` 回流；**不可**直接戳 owner store
- 不同 vertical 的 affordance 集合**互不污染**：`AffordanceRegistry` 启动时按 vertical id 隔离，跨 vertical 调用 fail loudly
- affordance outcome 必须能关联到 prior prediction 或 prediction context，作为 `prediction_error` / credit 的 typed evidence；invoker 不得成为 memory / temporal / social state owner。

## 工程挑战

- 让 metacontroller 真正学到 affordance 选择，而不是退化成 "always pick the first available"
- 让 affordance 描述符的 `when_to_use` / `when_not_to_use` 文字真正进入 LLM prompt（让 LLM 能"读得到"该不该用），但**不**让 LLM 直接做最终选择决策
- 让 safety_model 与 vertical-level boundary policy 协调（同一个 affordance 在不同 boundary 下可能被禁用）
- 让 coding vertical 第一批 affordance（read_file / grep / run_test / list_dir）足够通用，又不超出 sandbox 边界

## 算法候选

来自 `docs/next_gen_emogpt.md`：

| 算法 | 用途 |
|---|---|
| ETA `β_t` 切换单元 | 何时进入"使用 affordance"动作 |
| ETA Internal RL z_t | 选哪个 affordance + 参数（连续 z_t → 离散候选选择，与 regime selection_weights 同路径） |
| NL 多频率 MLP 链 | affordance 使用历史的多频带学习（高频：本 turn 选择；低频：vertical-level affordance 偏好分布） |

## 接口契约

### 新增 `vz-contracts` 类型

```python
class AffordanceKind(str, Enum):
    TOOL = "tool"           # 函数调用 / API
    ACTION = "action"       # 内部 action（plan / commit / clarify）
    ORGAN = "organ"         # 组合能力（多步内部流程）
    SHELL = "shell"         # 部署面 capability（text_streaming / voice / image）

@dataclass(frozen=True)
class AffordanceCost:
    latency_class: Literal["instant", "fast", "slow", "very_slow"]
    monetary_class: Literal["free", "low", "medium", "high"]
    rate_limit_per_minute: int | None = None

@dataclass(frozen=True)
class AffordanceSafety:
    requires_user_confirmation: bool
    irreversible: bool
    requires_consent_grant: tuple[str, ...] = ()   # consent grant 名称
    blocked_in_regimes: tuple[str, ...] = ()
    audit_required: bool = False

@dataclass(frozen=True)
class AffordanceDescriptor:
    name: str                          # 全局唯一
    kind: AffordanceKind
    version: str
    display_name: str
    description: str
    when_to_use: str                   # ≥ 50 char, post-init enforced
    when_not_to_use: str               # ≥ 50 char, post-init enforced
    parameters_schema: Mapping[str, Any]   # JSON Schema, frozen
    output_schema: Mapping[str, Any]
    cost_model: AffordanceCost
    safety_model: AffordanceSafety
    preconditions: tuple[str, ...]     # e.g. ("scene.is_open", "user_consent.tool_use")
    affordance_tags: tuple[str, ...]
    examples: tuple[str, ...]
    source_path: str                   # 来源 vertical 路径
    excluded_from_runtime_selection: bool = False

    def __post_init__(self) -> None:
        if len(self.when_to_use) < 50:
            raise ValueError(
                f"AffordanceDescriptor.when_to_use must be ≥ 50 chars, "
                f"got {len(self.when_to_use)} for {self.name!r}"
            )
        if len(self.when_not_to_use) < 50:
            raise ValueError(
                f"AffordanceDescriptor.when_not_to_use must be ≥ 50 chars, "
                f"got {len(self.when_not_to_use)} for {self.name!r}"
            )
        if not isinstance(self.parameters_schema, Mapping) or "type" not in self.parameters_schema:
            raise ValueError(f"AffordanceDescriptor.parameters_schema invalid for {self.name!r}")
```

### 新增 `lifeform-affordance` wheel

```
packages/lifeform-affordance/
├── pyproject.toml              # depends on lifeform-core + vz-contracts
└── src/lifeform_affordance/
    ├── __init__.py
    ├── registry.py             # AffordanceRegistry
    ├── snapshot.py             # AffordanceModule + AffordanceSnapshot
    ├── invoker.py              # AffordanceInvoker（调用 + 结果回流）
    └── renderers/
        ├── __init__.py
        ├── markdown.py
        ├── openai_tools.py
        ├── catalog_json.py
        └── compact_list.py
```

### `AffordanceSnapshot`（lifeform-side slot）

```python
@dataclass(frozen=True)
class AffordanceCandidate:
    descriptor_name: str
    score: float                  # 来自 metacontroller，[0, 1]
    rationale: str
    expected_cost: AffordanceCost
    blocked_reason: str = ""      # 非空说明被 boundary / safety 阻断

@dataclass(frozen=True)
class AffordanceSnapshot:
    available: tuple[AffordanceDescriptor, ...]   # 当前 vertical 注册的全集
    candidates_for_turn: tuple[AffordanceCandidate, ...]
    selected: AffordanceCandidate | None          # metacontroller 最终选定（可能为 None）
    description: str
```

`AffordanceModule.process()` 路径：

1. 读 `regime` snapshot + `temporal_abstraction` snapshot
2. 调 metacontroller 暴露的 `score_affordance_candidates(state) -> Mapping[name, float]`
3. 应用 `safety_model.blocked_in_regimes` / `boundary_consent` 过滤
4. 发布 `AffordanceSnapshot`

**不**包含执行动作；执行由 `AffordanceInvoker` 在显式 host 调用时进行。

### `AffordanceInvoker`

```python
class AffordanceInvoker:
    async def invoke(
        self,
        *,
        descriptor: AffordanceDescriptor,
        parameters: Mapping[str, Any],
        session: BrainSession,
        idempotency_key: str,
    ) -> AffordanceInvocationResult:
        ...
```

行为：

1. 校验 `parameters` 符合 `parameters_schema`
2. 校验 `safety_model.requires_user_confirmation` —— 需要时返回 `pending_confirmation` 状态而非执行
3. 调实际 backend（HTTP / 函数 / shell），带超时与 rate limit
4. 把 `(success/failure, summary, detail)` 通过 `session.submit_tool_result(...)` 回流到 kernel
5. 返回 `AffordanceInvocationResult(status, kernel_event_ids)`

`AffordanceInvoker` 是 lifeform 层的 invoker，**不是** kernel 行为。它封装的是"产品如何让 lifeform 真去调工具"，调用结果走的仍是已有的 semantic event 通道。

### Vertical 注册

每个 vertical 在自己的 `lifeform-domain-*` 包内带 `affordances/*.yaml`：

```yaml
# lifeform-domain-coding/affordances/read_file.yaml
name: read_file
kind: tool
version: "0.1.0"
display_name: "Read file"
description: "Read a UTF-8 text file from the workspace and return its content."
when_to_use: |
  Use when you need exact textual contents of a code file before you can
  reason about it. Prefer this over guessing what's inside the file.
when_not_to_use: |
  Don't use to enumerate large directories — use list_dir for that.
  Don't use on binary files; the call will fail and waste a turn.
parameters_schema:
  type: object
  properties:
    path: {type: string}
    max_bytes: {type: integer, default: 65536}
  required: [path]
output_schema:
  type: object
  properties:
    content: {type: string}
    truncated: {type: boolean}
cost_model:
  latency_class: fast
  monetary_class: free
safety_model:
  requires_user_confirmation: false
  irreversible: false
preconditions: ["scene.is_open"]
affordance_tags: ["read", "code"]
examples:
  - "read_file(path='packages/lifeform-core/src/lifeform_core/vitals.py')"
```

`build_coding_lifeform()` / `build_companion_lifeform()` 在构造时把自己 vertical 的 affordances 一次性注册进 `AffordanceRegistry`。

### Metacontroller 集成

`vz-temporal/temporal/interface.py:MetacontrollerRuntimeState` 增加：

```python
@dataclass(frozen=True)
class AffordanceSelectionState:
    candidate_scores: tuple[tuple[str, float], ...]   # (name, score)
    selected_name: str | None
    selection_entropy: float                           # 多样性监控
```

发布到 `temporal_abstraction` snapshot。`AffordanceModule` 直接读这个 sub-state 而不重算评分——R8 owner 单写者。

Metacontroller 学习路径（与 regime selection_weights 同路径）：

- SSL pass：把 affordance 选择历史作为额外行为序列压缩
- RL pass：把 affordance 调用 outcome（来自 tool_result event）的 reward 作为信用源

## ETA / NL 集成

- **R3 / R4**：affordance 选择是 z_t 抽象动作的一类候选，与 regime 选择共享同一 metacontroller。这正是 R3 "时间抽象作为一等能力" 的扩展。
- **R8**：`AffordanceRegistry` 是单写者；descriptor 不可变；执行结果通过 owner snapshot 通道回流
- **R10**：safety_model + ModificationGate 共同守门；`requires_user_confirmation` 强制走人类 gate
- **R11**：affordance 候选评分 + 选择结果都在 snapshot 里 nameable / publishable

## 当前 proof surface

引入后必须能证明的命题：

1. `descriptor-quality-enforced`
   - `when_to_use ≥ 50` / `when_not_to_use ≥ 50` 由 dataclass `__post_init__` 强制（fail loudly）
   - acceptance：`tests/contracts/test_affordance_descriptor.py`
2. `selection-is-learned-not-hardcoded`
   - grep 仓库找不到 `if name == "read_file"` 等硬编码路由
   - 所有选择路径都流经 `AffordanceSelectionState`
   - acceptance：`tests/contracts/test_no_keyword_routing.py`
3. `affordance-improves-coding-f1`
   - coding vertical 在 `bug-no-repro` / `concrete-debug` scenario 下 ACTIVE affordance 选择，相对 ablation `affordance_wiring=DISABLED` 在 family report F1 有可测增益
   - acceptance：`lifeform-bench --vertical coding --with-affordance` 对比
4. `cross-vertical-isolation`
   - companion vertical 不能 invoke coding vertical 的 affordance（fail loudly）
   - acceptance：`tests/lifeform_e2e/test_affordance_cross_vertical.py`
5. `tool-result-replays-canonical-channel`
   - affordance invocation 结果只通过 `BrainSession.submit_tool_result` 进入 kernel，contract test grep 调用图
   - acceptance：`tests/contracts/test_affordance_canonical_return.py`

## 接口契约（公开数据流向）

**消费的输入**：

- `regime` 快照
- `temporal_abstraction` 快照（含 `AffordanceSelectionState`）
- `boundary_consent` 快照（受 safety_model 影响）
- `vitals` 快照（间接：成本控制）

**产出的输出**：

- `affordance` lifeform-side slot（不进 kernel slot 注册表）
- 通过 `BrainSession.submit_tool_result` 间接产生 `execution_result` / `tool_outcome` 事件

`affordance` snapshot 是 lifeform 层公开契约；prompt planner 与 response synthesizer 可以读它来决定是否在 prompt 中渲染 affordance 列表。

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|---|---|---|
| 依赖 | Environment Interface | affordance 是 `Act` 面的有界外部作用能力，调用结果必须通过 outcome 回流 PE |
| 依赖 | 时间抽象与内部控制 | affordance 选择由 metacontroller 学 |
| 依赖 | 契约式运行时 | descriptor + snapshot 走 R8 |
| 依赖 | 信用分配与自修改 | safety_model 走 ModificationGate |
| 协作 | 认知 Regime | 不同 regime 下不同 affordance 候选集 |
| 协作 | Domain Experience Layer | vertical 自带 affordances，与 DomainExperiencePackage 同包发布 |
| 协作 | 评估体系 | F1 / F4 / F5 / F6 都受影响 |

## 回滚

`AffordanceModule.WiringLevel`：

- `DISABLED`（v0 默认）：注册表为空，prompt 不渲染 affordance
- `SHADOW`：注册表加载，metacontroller 评分发布到 snapshot，但 invoker 不实际调用
- `ACTIVE`：invoker 启用

每个 vertical 独立 wiring level（`coding_affordance_wiring` / `companion_affordance_wiring`），互不影响。

每个具体 affordance 还可独立 disable（`AffordanceDescriptor.excluded_from_runtime_selection=True`），运维侧可单独熄灭某个工具。

回滚路径：

- 系统级故障：`AffordanceModule.WiringLevel = DISABLED`
- 单工具故障：descriptor 改 `excluded_from_runtime_selection=True`，下次启动生效
- safety 升级：`AffordanceSafety.requires_user_confirmation = True`，所有调用强制走 gate

## 变更日志

- 2026-04-29：初始版本，对应 `docs/implementation/13_emogpt_prd_alignment_upgrade.md` Gap 1 设计冻结。
