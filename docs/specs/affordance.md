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

Packet C (long-horizon-closure) 落地：

`vz-temporal/temporal/interface.py:MetacontrollerRuntimeState` 增加：

```python
@dataclass(frozen=True)
class AffordanceSelectionState:
    candidate_scores: tuple[tuple[str, float], ...]   # (name, score)
    selected_name: str | None
    selection_entropy: float                           # 多样性监控
```

`MetacontrollerRuntimeState.affordance_selection: AffordanceSelectionState | None = None` 是新字段，**默认 None**。当 metacontroller owner 自己选择发布该字段时，`AffordanceModule` 优先采用（R8 单 owner）。

`lifeform-affordance/module.py:AffordanceModule` 实现：

- `slot_name = "affordance"`，`dependencies = ("temporal_abstraction",)`，**默认 `WiringLevel.ACTIVE`**（long-horizon-closure follow-up）
- `process()` 的优先级：
  1. 若 metacontroller 已发布 `affordance_selection`，直接采用（future path）
  2. 否则用 `score_affordance_candidates(z_t=temporal_snapshot.controller_state.code, descriptor_names=...)` 在本地做投影
  3. 若 z_t 为空（cold start），所有 candidate 给 0.5 中性分，selected = None
- `score_affordance_candidates` 是 pure function：用 SHA-256 把每个描述符名映射到投影向量，再与 z_t 做 dot product + sigmoid。**绝不**在内部 branch on descriptor name。

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
6. `affordance-selection-no-rules` (Phase 1 clean)
   - affordance selection continues through metacontroller state / `z_t`; no routing by `descriptor.name` or outcome text.
   - acceptance: grep finds no `if name == ...` routing logic; selection state remains published by the temporal owner.
7. `affordance-outcome-observable-only` (Phase 1 clean)
   - `AffordanceInvoker` only fills `latency_ms`, `monetary_cost`, `reversibility`, and `environment_state_delta_kind` before calling `BrainSession.submit_tool_result`.
   - acceptance: environment + invoker tests show outcome has no trust / common-ground / commitment semantic delta.
8. `affordance-snapshot-replay-compatible` (Phase 1 clean)
   - affordance invocation can be audited through the existing snapshot replay artifact; no separate trace runtime schema is required.
   - acceptance: snapshot replay artifact includes `EnvironmentOutcome` observable fields and `prediction_error.action_context`.
9. `affordance-outcome-carries-prediction-id` (Packet A — long-horizon-closure)
   - When the caller threads `plan_ref` through `AffordanceInvoker.invoke(... plan_ref="p-XYZ")`, the next-turn `prediction_error.action_context.prediction_id` MUST equal that `plan_ref`. When `plan_ref=None` (default), the field MUST stay empty (back-compat path).
   - This is the lineage that lets long-horizon credit attribute outcomes to specific prior predictions, instead of leaving every invoker call with `prediction_id=None`.
   - acceptance: `tests/lifeform_e2e/test_affordance_pe_lineage.py`
10. `affordance-scoring-from-metacontroller` (Packet C — long-horizon-closure)
    - `AffordanceModule` exists, satisfies `RuntimeModule[AffordanceSnapshot]`, default wiring ACTIVE; `AffordanceSelectionState` dataclass exists in `vz-temporal`; `MetacontrollerRuntimeState.affordance_selection: AffordanceSelectionState | None = None` field exists (None by default).
    - `score_affordance_candidates` is the pure-function projection from `z_t` to per-descriptor scores via deterministic SHA-256 hashing. No branch on descriptor name strings.
    - Different `z_t` produces different selection (otherwise the projection isn't z_t-driven). Same `z_t` produces deterministic identical scores.
    - acceptance: `tests/contracts/test_no_keyword_routing.py` + `tests/contracts/test_affordance_module_contract.py` + `tests/lifeform_e2e/test_affordance_metacontroller_scoring.py`

## PE Prediction Lineage (Packet A — long-horizon-closure)

`AffordanceInvoker.invoke(... plan_ref: str | None = None)` threads a caller-supplied `plan_ref` through `_finalize` to `BrainSession.submit_tool_result(plan_ref=...)`. Concretely:

```text
AffordanceInvoker.invoke(plan_ref="p-XYZ")
  -> AffordanceInvoker._finalize(plan_ref="p-XYZ")
  -> BrainSession.submit_tool_result(plan_ref="p-XYZ")
     -> EnvironmentOutcome(prediction_id="p-XYZ")
     -> AgentSessionRunner.remember_environment_outcome(outcome_id)
     -> AgentSessionRunner.remember_environment_prediction_id("p-XYZ")
  -- next turn --
  -> AgentSessionRunner._consume_pending_environment_prediction_id() -> "p-XYZ"
  -> run_final_wiring_turn(environment_prediction_id="p-XYZ")
  -> _prediction_action_context_from_upstream(environment_prediction_id="p-XYZ")
  -> PredictionActionContext(prediction_id="p-XYZ", environment_outcome_id=...)
  -> PredictionErrorModule.process(...) publishes the action context on the snapshot
```

Invariants:

- `plan_ref=None` (default) preserves the legacy "no lineage" behavior verbatim. Existing tests must not regress.
- The lineage applies on the `_finalize` branches that actually call `submit_tool_result` — `SUCCEEDED` and `BACKEND_FAILED`. Pre-flight gate denials (BOUNDARY / RATE_LIMITED / PARAMETER_INVALID / BACKEND_MISSING / EXCLUDED) do not reach the kernel and therefore do not produce lineage.
- `prediction_id` and `environment_outcome_id` are independent fields on `PredictionActionContext`. The former is caller-bound (which prior prediction was this call attached to?), the latter is auto-derived from the call's own outcome id.
- The kernel does NOT validate that `plan_ref` corresponds to an actually-registered prior prediction id — that is downstream credit's job. The lineage layer only carries the string.

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
| ?? | Emergent Action Abstraction | ??????? `EnvironmentOutcome` ??????selection ?? `z_t` ??replay ? existing snapshots ?? |
| 依赖 | 契约式运行时 | descriptor + snapshot 走 R8 |
| 依赖 | 信用分配与自修改 | safety_model 走 ModificationGate |
| 协作 | 认知 Regime | 不同 regime 下不同 affordance 候选集 |
| 协作 | Domain Experience Layer | vertical 自带 affordances，与 DomainExperiencePackage 同包发布 |
| 协作 | 评估体系 | F1 / F4 / F5 / F6 都受影响 |

## MCP-backed Affordances（mcp-tools-bundle-bridge packet）

External MCP servers（[`docs/specs/mcp-bridge.md`](mcp-bridge.md)）可以贡献第二级 affordance 来源：

- 每个 MCP server 的 `tools/list` → `AffordanceDescriptor`，descriptor name 加 server name 前缀（`<server>.<tool>`）
- `safety_model` / `cost_model` / `when_to_use` / `when_not_to_use` **必须**来自外部 repo 的 reviewed `.vzbridge.yaml`；缺则 `MCPMissingSafetyManifestError`
- MCP-supplied tools 与 in-process tools 共享同一 `AffordanceRegistry` 与 `AffordanceInvoker`；选择仍由 `AffordanceModule` z_t 投影驱动（无"MCP 优先"硬路由）
- MCP server crash → 该 server 的 candidate 在 snapshot 中带 `blocked_reason="mcp_unavailable:<server>"`；主进程不崩
- bridge 实现在 `lifeform-mcp-bridge` wheel；wheel 边界禁止反向 import `vz-*` 内核

## 回滚

`AffordanceModule.WiringLevel`（long-horizon-closure follow-up：默认 `ACTIVE`）：

- `ACTIVE`（**当前默认**）：注册表加载，metacontroller 评分发布到 snapshot，invoker 启用
- `SHADOW`：评分发布但 invoker 不实际调用（benchmark ablation 用）
- `DISABLED`：注册表为空，prompt 不渲染 affordance（rollback 用）

每个 vertical 独立 wiring level（`coding_affordance_wiring` / `companion_affordance_wiring`），互不影响。

每个具体 affordance 还可独立 disable（`AffordanceDescriptor.excluded_from_runtime_selection=True`），运维侧可单独熄灭某个工具。

回滚路径：

- 系统级故障：`AffordanceModule.WiringLevel = DISABLED`
- 单工具故障：descriptor 改 `excluded_from_runtime_selection=True`，下次启动生效
- safety 升级：`AffordanceSafety.requires_user_confirmation = True`，所有调用强制走 gate

## Production Automatic Tool Loop

> Status: ACTIVE implementation target (2026-05-22)

Automatic tool calling is a **lifeform/platform orchestration concern**, not a
kernel `propagate()` concern. The loop runs outside `vz-*` owner propagation and
uses the existing canonical result bus:

```text
LifeformSession.run_turn(...)
  -> ToolLoopOrchestrator reads typed affordance/tool intent
  -> AffordanceInvoker.invoke(..., session=BrainSession, plan_ref=...)
  -> BrainSession.submit_tool_result(...)
  -> next LifeformSession.run_turn(...) consumes outcome via PE/action context
```

### Loop Invariants

- No `vz-*` wheel imports `lifeform-affordance` or calls an invoker.
- The orchestrator is bounded by `max_tool_steps`, `max_wall_ms`, and
  `allow_async_tasks`.
- Tool selection is descriptor/snapshot driven. User text keyword routing is
  forbidden.
- Every executed tool call has a stable `event_id`, `action_id`, and optional
  `plan_ref` so replay can connect invocation -> `EnvironmentOutcome` ->
  `PredictionActionContext` -> credit.
- Pre-flight denials remain orchestration events unless the backend actually
  ran. They do not masquerade as tool results.
- Long-running tools return a task handle and later submit completion through
  the same `submit_tool_result` channel.

### New Lifeform-Side Types

`lifeform_affordance.tool_loop` owns:

- `ToolLoopPolicy`: step, wall-clock, async, and server/client loop settings.
- `ToolCallIntent`: typed request for one affordance invocation.
- `ToolLoopStep`: immutable audit record for one step.
- `ToolLoopResult`: final text/tool-call/task-handoff result plus readout data.
- `ToolLoopOrchestrator`: bounded loop executor around `LifeformSession`.

These types are outside `vz-contracts` because they orchestrate product
execution. Kernel owners only see the existing environment outcome and semantic
event contracts.

### Invoker Hardening

`AffordanceInvoker.invoke(...)` is additive-extended with:

- `idempotency_key`: repeat calls return the same `AffordanceInvocationResult`
  without re-running side effects.
- `timeout_seconds`: caller override; default derives from
  `AffordanceCost.latency_class`.
- `session_budget`: global/session budget gate in addition to per-tool rate
  limits.
- `PENDING_CONFIRMATION`: explicit status for tools requiring human confirmation.
- Async task statuses: `queued`, `running`, `succeeded`, `failed`, `cancelled`
  via `AffordanceTaskHandle`.

Fast tools still produce immediate invocation results. Slow tools may return a
handle and later call `submit_deferred_result(...)` to publish the canonical
kernel outcome.

## 变更日志

- 2026-07-12: per-turn publisher closure — `LifeformConfig.affordance_wiring`
  controls a lifeform-side `AffordanceModule`. When an affordance registry
  exists, `Lifeform.create_session()` constructs the module and
  `LifeformSession.run_turn()` publishes its immutable snapshot from the
  completed kernel turn's public snapshots. ACTIVE enters
  `latest_active_snapshots`, SHADOW enters observability-only shadow snapshots,
  DISABLED skips construction. The kernel DAG remains unchanged and does not
  import `lifeform-affordance`.
- 2026-05-22: Production automatic tool loop contract — added lifeform-side
  `ToolLoopOrchestrator`, bounded execution policy, OpenAI/DLaaS bridge
  expectations, invoker idempotency/timeout/session-budget/confirmation status,
  and async task handle lifecycle. Tool results remain canonical through
  `BrainSession.submit_tool_result`; kernel propagation is unchanged.
- 2026-05-12: long-horizon-closure follow-up — `AffordanceModule.default_wiring_level` 与 `FinalRolloutConfig.affordance` 默认从 SHADOW 翻 ACTIVE，合并入正常主路径而非 opt-in。Benchmark 仍可显式 SHADOW；DISABLED 留作 rollback。
- 2026-05-12: Packet C (long-horizon-closure) — landed `AffordanceModule` (lifeform-affordance/module.py), `AffordanceSelectionState` dataclass + `MetacontrollerRuntimeState.affordance_selection` reserved field in vz-temporal, pure-function `score_affordance_candidates` z_t→name SHA-256 projection. New acceptance gates `affordance-scoring-from-metacontroller` (functional) and the existing `selection-is-learned-not-hardcoded` is now backed by static contract test `tests/contracts/test_no_keyword_routing.py` plus contract / e2e tests. `FinalRolloutConfig.affordance: WiringLevel` field added (additive, no kernel propagate change yet — lifeform layer reads the flag).
- 2026-05-12: Packet A (long-horizon-closure) — added `plan_ref` parameter to `AffordanceInvoker.invoke` threaded all the way to `PredictionActionContext.prediction_id` next turn; new acceptance gate `affordance-outcome-carries-prediction-id` and `tests/lifeform_e2e/test_affordance_pe_lineage.py`. Back-compat path (`plan_ref=None`) unchanged.
- 2026-05-02: Rewrote Phase 1 clean gates as `affordance-selection-no-rules` / `affordance-outcome-observable-only` / `affordance-snapshot-replay-compatible`, aligned with the ETA/NL-first `emergent-action-abstraction` spec.
- 2026-04-29：初始版本，对应 `docs/implementation/13_emogpt_prd_alignment_upgrade.md` Gap 1 设计冻结。
