# 中频思考循环 Spec

> Status: draft
> Last updated: 2026-04-29
> 对应需求: R1, R6, R8, R11, R15
> 来源: `docs/implementation/13_emogpt_prd_alignment_upgrade.md` Gap 4

## 要解决的问题

VolvenceZero 当前主链只有两档时间尺度生效：

- **online-fast**：per-turn 同步路径（perception → temporal → memory → reflection writeback proposals → response）
- **background-slow**：scene 闭合后的 R6 session-post slow loop（durable promotion / decay / belief writeback / temporal-prior apply）

R1 要求的 4 档时间尺度里，**`session-medium`（scene 内、跨 turn）** 和 **`online-fast` 内部的 read-only side-thinking** 都没有专门的 owner 与生命周期。EmoGPT v4 PRD §6 把这一段拆成 ThinkingLoopScheduler 的三种异步思考（Active Exploration、MidSession Reflector、ProvisionalLesson）；本 spec 把这套需求映射到 VolvenceZero 第一性设计：

- **不**新建第二个 owner 替代 `world_temporal` / `self_temporal` / `memory` / `case_memory`
- **不**让后台思考路径产生 owner mutation 副作用
- 只新增**生命周期管理 + read-only worker + 通过既有 reflection writeback 的 typed proposal 路径**

## 关键不变量

- `ThinkingScheduler` 是 lifeform 层 wheel，**不**进 kernel 主链
- 后台 `ThinkingTask` worker 只读快照、产 `ThinkingArtifact`，**永远**不调任何 owner 的 mutation API
- 每个 task 起手时记录 `snapshot_fingerprint`，apply 前比对；不匹配一律 `STALE`，不允许"再 apply 一次试试"
- `case_memory` 的 `provisional` lifecycle 只是 lifecycle metadata；retrieval / response 仍只读 `validated` lifecycle 的记录（默认）
- ProvisionalLesson 不创新 owner、不绕过 `ApplicationCaseMemoryStore`、TTL 到期或 scene 闭合时统一 reconcile
- mid-reflection 的 self / world 双 lane 复用既有 `world_temporal` / `self_temporal` 双 owner，不新增 owner
- worker 调度受 `ThinkingScheduler.WiringLevel` 守护（DISABLED / SHADOW / ACTIVE）；ACTIVE 之前必须有 SHADOW 阶段产证据
- 任何 worker 故障**必须** fail-loudly 写入 task `status=FAILED` + `error_class`，**不可**静默吞错

## 工程挑战

- 在不破坏 R8 owner 单写者的前提下，让 worker 看见 owner 状态、产出建议、由 owner 自己决定是否 apply
- 让 mid-reflection 不阻塞 user turn —— P95 延迟漂移 ≤ 8% 是硬门槛
- 让 self / world 双 lane 错峰、互不饿死
- 让 ProvisionalLesson 既能被 retrieval 看见（弱先验影响打分），又不污染 validated 检索结果（不替代）

## 算法候选

来自 `docs/next_gen_emogpt.md`：

| 算法 | 用途 |
|---|---|
| NL CMS 中频层 | provisional → validated promotion 是 CMS 中频带的"显式生命周期版本" |
| ETA SSL-RL 交替 | mid-reflection 是 SSL pass 之一（压缩当前 scene 内行为历史） |
| ETA `β_t` 切换单元 | 何时排队 active exploration 由 metacontroller 的切换信号驱动 |

## 接口契约

### 新增 `vz-contracts` 类型

```python
class ThinkingDepth(str, Enum):
    FAST = "fast"           # wave 内同步（不归本 scheduler 管，在此仅作枚举完整性）
    MID = "mid"             # session-medium：scene 内异步
    SLOW = "slow"           # session-post：已有 R6 路径

class ThinkingTaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    STALE = "stale"         # fingerprint 不匹配
    CANCELLED = "cancelled"
    FAILED = "failed"

class ThinkingPurpose(str, Enum):
    WORLD_LANE_REFLECT = "world_lane_reflect"
    SELF_LANE_REFLECT = "self_lane_reflect"
    EXPLORATION = "exploration"
    PROVISIONAL_RECONCILE = "provisional_reconcile"

@dataclass(frozen=True)
class ThinkingTask:
    task_id: str
    depth: ThinkingDepth
    purpose: ThinkingPurpose
    requested_at_turn_index: int
    snapshot_fingerprint: str           # SHA256(json(关键 owner snapshots))
    consumer_owner: str                  # 哪个 owner apply artifact
    deadline_at_turn_index: int | None   # 过期前没完成则 STALE

@dataclass(frozen=True)
class ThinkingArtifact:
    task_id: str
    status: ThinkingTaskStatus
    payload: Any                         # frozen dataclass; consumer 决定 schema
    produced_at_turn_index: int
    consumer_owner: str
    error_class: str = ""
    error_detail: str = ""
```

### 新增 `lifeform-thinking` wheel

```
packages/lifeform-thinking/
├── pyproject.toml
└── src/lifeform_thinking/
    ├── __init__.py
    ├── scheduler.py        # ThinkingScheduler
    ├── fingerprint.py      # owner snapshot → fingerprint 计算
    └── workers/
        ├── __init__.py
        ├── mid_reflection.py    # WORLD_LANE_REFLECT / SELF_LANE_REFLECT
        ├── exploration.py       # EXPLORATION
        └── provisional.py       # PROVISIONAL_RECONCILE
```

依赖关系：`lifeform-thinking` → `lifeform-core` + `vz-contracts`；**不可**反向 import 任何 `vz-cognition` / `vz-memory` / `vz-temporal` 的 owner store。

### 新增 `case_memory` lifecycle 字段

`vz-application/runtime.py:CaseMemoryRecord`：

```python
class CaseLifecycle(str, Enum):
    CANDIDATE = "candidate"          # 反思生成的弱先验候选
    PROVISIONAL = "provisional"      # 已被注入 retrieval 但仍带 TTL
    VALIDATED = "validated"          # 通过 promotion threshold，永久
    RETIRED = "retired"              # 被 supersede 或失效

# 增加字段（向后兼容默认 VALIDATED 不影响现有数据）
lifecycle: CaseLifecycle = CaseLifecycle.VALIDATED
ttl_seconds: int | None = None
expires_at_tick: int | None = None
provisional_origin: str = ""        # task_id / reflection_id 来源
```

### Reflection writeback 路径扩展

`vz-cognition/reflection/writeback.py`：

```python
@dataclass(frozen=True)
class ProvisionalCaseProposal:
    case_id: str
    record: CaseMemoryRecord
    ttl_seconds: int                  # 默认 1800（30 min）
    origin_task_id: str

@dataclass(frozen=True)
class ProvisionalCaseReconcileResult:
    promoted: tuple[str, ...]
    retired: tuple[str, ...]
    expired: tuple[str, ...]

# 新方法
def reconcile_provisional_cases(
    *,
    now_tick: int,
    case_memory_snapshot: CaseMemorySnapshot,
    eviction_policy: ProvisionalEvictionPolicy,
) -> ProvisionalCaseReconcileResult:
    ...
```

调用时机：

- 每个 turn 结束时**只**做 expire 检查（廉价）
- scene-end / session-post slow loop 时做完整 promote / retire 决策

### Owner-side apply 路径

mid-reflection worker 的 artifact payload 是 `MidReflectionPayload`：

```python
@dataclass(frozen=True)
class MidReflectionPayload:
    track: Literal["world", "self"]
    reward_signal: float                  # [-1, 1]
    reward_evidence: tuple[str, ...]
    suggested_controller_pressure: float  # [-1, 1]
    suggested_provisional_cases: tuple[ProvisionalCaseProposal, ...]
```

consumer：

- `world_temporal` / `self_temporal` owner 在自己 process 路径内消费 `suggested_controller_pressure`，作为 advisory 进入既有 controller_code 更新（不替代信号源）
- `case_memory` owner 收下 `suggested_provisional_cases`，写为 `lifecycle=PROVISIONAL`

active exploration worker 的 artifact 是 `ExplorationPayload`：

```python
@dataclass(frozen=True)
class ExplorationPayload:
    consultation_axes_resolved: tuple[str, ...]
    consultation_axes_needs_user: tuple[str, ...]
    evidence_summary: str
    confidence: float
```

consumer：`boundary_consent` owner（更新 `consultation_need`），并通过快照让下一轮 `prompt_planner` 看见。

### Fingerprint 计算

`lifeform-thinking/fingerprint.py:compute_fingerprint(snapshots: Mapping[str, Snapshot]) -> str`：

- 输入 owner snapshot 的 `(slot_name, version, value-hash)` 三元组排序后 SHA256
- 必须显式列出 fingerprint 关心的 slot 集合，作为 dataclass 字段；**禁止** wildcard "all snapshots"
- 每个 worker 类型独立 fingerprint scope：mid-reflection 关心 `regime / world_temporal / self_temporal / memory`；exploration 关心 `boundary_consent / panorama-equivalent`（VZ 这边是 `regime + temporal_abstraction`）

## ETA / NL 集成

- **R1 中频带闭合**：mid-reflection 是 session-medium 频带的显式 SSL pass。它产 `controller_pressure` 信号，由 owner 自己融合进 controller_code 更新；这等价于 ETA paper-suite 里的 SSL-RL 交替循环 SSL 阶段，但作用范围限制在 scene 内、不触发 substrate 修改。
- **R6 慢反思扩展**：原有 session-post slow loop 不动；新加的中频 reflection 是它的"小弟"，写到同一个 `case_memory` lifecycle 池但用更严格的 TTL 与门槛。
- **R-PE 信号源**：mid-reflection 自己产生 PE 是错的——PE 仍由现有 owner 产生。mid-reflection 只是把 turn 之间的 PE history 压缩成 `controller_pressure` advisory。
- **R8 owner 单写者**：worker 永远 read-only。任何 mutation 都通过 owner 自己消费 artifact 实现。

## 当前 proof surface

引入后必须能证明的命题：

1. `mid-reflection-improves-fast-track`
   - 多轮 scenario 下 mid-reflection ON 相对 OFF 在 family report F4 有可测正向 delta
   - acceptance：`tests/lifeform_e2e/test_mid_reflection_benefit.py`
2. `provisional-promotion-is-evidence-driven`
   - provisional 提到 validated 必须经过 `min_total_records >= 2 + min_same_polarity >= 2 + min_mean_abs_reward >= 0.15`，与 EmoGPT pattern threshold 对齐
   - acceptance：unit test 验证 reconcile 决策表
3. `stale-artifact-never-applies`
   - artifact apply 前 fingerprint mismatch 一律 `STALE`，contract test 强制
   - acceptance：`tests/contracts/test_thinking_lifecycle.py`
4. `worker-cannot-mutate-owners`
   - lifeform-thinking 包内 grep 只能见 snapshot 读取，不见任何 owner store 的 import
   - acceptance：`tests/contracts/test_import_boundaries.py` + grep lint
5. `mid-reflection-does-not-block-turn`
   - benchmark `--with-mid-reflection` 后 P95 turn 延迟漂移 ≤ 8%
   - acceptance：`lifeform-bench` 报告

## 接口契约（公开数据流向）

**消费的输入**：

- `world_temporal` 快照
- `self_temporal` 快照
- `memory` 快照
- `regime` 快照
- `case_memory` 快照
- `boundary_consent` 快照
- `prediction_error` 快照（不直接驱动 worker，但是 reflection writeback 的输入）

**产出的输出**：

- `thinking_loop` lifeform-side slot（Mapping[task_id → ThinkingTask 状态]，仅 telemetry）
- 通过 reflection writeback proposal 间接影响：`case_memory`（lifecycle 变化）、`world_temporal` / `self_temporal`（controller_pressure advisory）、`boundary_consent`（consultation_need 更新）

`thinking_loop` snapshot 只用于可观测性 / debug / family report metric 输入；**不**进入 kernel 主链 propagate 顺序。

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|---|---|---|
| 依赖 | 契约式运行时 | 通过 `ThinkingTask` / `ThinkingArtifact` 不可变契约通信 |
| 依赖 | 连续记忆系统 | provisional case 进入 `case_memory` lifecycle |
| 依赖 | 双轨学习 | mid-reflection self/world 双 lane 复用现有双 owner |
| 协作 | 认知 Regime | exploration worker 读 regime；regime 切换可触发 worker 排队 |
| 协作 | 信用分配 | reflection writeback 的 outcome enum 可来自 mid-reflection artifact |
| 协作 | 评估体系 | F4 / F5 family metric 监控 mid-reflection 效果 |

## 回滚

`ThinkingScheduler.WiringLevel`：

- `DISABLED`（v0 默认）：scheduler 不存在或不启动 worker
- `SHADOW`：worker 跑但 artifact 不 apply，仅 telemetry
- `ACTIVE`：consumer apply

每个 worker 类型独立 wiring level（`mid_reflection_wiring` / `exploration_wiring` / `provisional_wiring`），可单独回滚。

`case_memory.lifecycle` 字段不可移除（向后兼容），但行为可降级：把所有 provisional 当作 validated 处理（v0 默认行为不变）。

## 生产 wiring（slice 2c）

`ThinkingScheduler` / `mid_reflection_worker` 从 slice 2b 起已经可以独立构造，但默认 `LifeformSession` 路径没有调度任何 mid-reflection task。slice 2c 补齐这一段：

### `ThinkingAdapter`（`lifeform-thinking.adapter`）

```python
class ThinkingAdapter:
    async def on_turn_begin(*, snapshots, turn_index) -> None: ...
    async def on_turn_end(*, snapshots, turn_index) -> None: ...
    async def drain() -> tuple[ThinkingArtifact, ...]: ...
```

- `on_turn_end`：每轮 kernel 结束后，按 `WORLD_LANE_REFLECT` / `SELF_LANE_REFLECT` 两个 purpose 各提交一个 task，upstream scope 是 `MID_REFLECTION_SCOPE = ("dual_track", "regime")`；scheduler 内部对 `wiring_level=DISABLED` 的 submit 立即 CANCELLED。
- `on_turn_begin`：下一轮 kernel 开始前，把上一轮 pending 批次逐个 `collect()`，current snapshots 为会话持有的 "上一轮 active_snapshots"。fingerprint 命中 → `COMPLETED` 落到 `latest_artifacts_by_consumer`；scope 内任一 slot 漂移 → scheduler 自动翻 `STALE`，adapter 不 publish。
- `drain`：scene close 时把所有 in-flight task 跑完，不让 worker 活过 scene。

### `LifeformSession` 一侧

- 新增构造参数 `thinking_adapter: Any = None`。类型是 `Any` 而不是具体类，**`lifeform-core` 不 import `lifeform-thinking`** —— duck-type 通过结构化协议 `ThinkingAdapterProtocol` 描述契约。
- `run_turn` 的 hook 顺序：**`on_turn_begin` → kernel → on_turn_end`**。这保证 fingerprint guard 永远以"本轮 kernel 未跑前的最新 snapshots"为锚点。
- `end_scene` 在 case memory reconcile 之后调用 `drain`。
- 观察面：`thinking_adapter` / `thinking_adapter_snapshot` / `latest_thinking_artifacts_by_consumer`。
- 失败隔离：3 个 hook 任一 raise 都被 try/except 捕获后只记日志，保证 kernel 主链不受 adapter bug 连累。

### `Lifeform` 一侧

- 新增 `thinking_adapter_factory: Callable[[], Any] | None = None` 参数。`create_session` 每次调用一次工厂，产出一个 per-session adapter，session 之间互不共享调度器状态。
- 便捷方法 `Lifeform.with_thinking_adapter_factory(factory)` 产生一个挂了 adapter 的 clone。

### 默认装配 (`build_default_thinking_adapter`)

工厂等价于：

```python
ThinkingAdapter(
    wiring_level=ThinkingWiringLevel.SHADOW,
    max_concurrent_tasks=2,
    worker=mid_reflection_worker,
    scope=MID_REFLECTION_SCOPE,
)
```

默认 SHADOW 是保守选择：adapter 跑但 downstream consumer（prompt planner、family report）目前只读 `latest_thinking_artifacts_by_consumer` 做 observability。切到 ACTIVE 意味着 prompt planner / evaluator 承诺消费；这一步留给后续 slice。

### 生产上该怎么开

1. 从 `SHADOW` 开始：`session.thinking_adapter_snapshot` 监控 `total_completed / total_stale / total_failed` 曲线；`STALE` 占比飙高说明 fingerprint scope 颗粒度错了，需调整。
2. 两轮 multi-round benchmark 对比 ON vs OFF，F4（learning quality）至少不降。
3. 确认 P95 turn 延迟漂移 ≤ 8%（hard gate）——workers 跑在 asyncio.Semaphore 之后的并发轨道，不应该阻塞。
4. 明确的下游 consumer 设计落地后，再切 `ACTIVE`。

## 变更日志

- 2026-04-29：初始版本，对应 `docs/implementation/13_emogpt_prd_alignment_upgrade.md` Gap 4 设计冻结。
- 2026-04-29：slice 2c 产生 wiring —— 新增 `ThinkingAdapter` + `LifeformSession` hook pattern + `Lifeform.with_thinking_adapter_factory` 便捷方法；默认 SHADOW 可观察、DISABLED 可秒切；duck-typed `ThinkingAdapterProtocol` 保证 `lifeform-core` 不反向依赖 `lifeform-thinking`。

## 变更日志补充

- 2026-07-14: CP-21 生产 consumer 闭合（GAP-06）。此前 worker 产出
  `MidReflectionPayload` 而 temporal owner 只接受 `ControllerPressureAdvisory`，
  且无人调用 `observe_thinking_artifact` —— 链路断开。本轮闭合：
  (a) 发布侧转换：`lifeform_thinking.workers.mid_reflection.controller_pressure_advisory_from_mid_reflection`
  （payload owner 同时拥有两种表示，消费者不重建语义）；
  (b) `ThinkingAdapter.latest_advisory_artifacts_by_consumer` 暴露契约就绪的
  advisory artifact（原 `latest_artifacts_by_consumer` 保留富 payload 观测面）；
  (c) `BrainSession.submit_thinking_artifact(artifact, apply_enabled=False)` 成为
  lifeform → kernel 的唯一路由入口，按 `consumer_owner` 分发到 world/self
  temporal owner；
  (d) `LifeformSession` 在 `on_turn_begin` 收集后自动以 SHADOW 语义转发，
  routing 结果暴露于 `latest_thinking_advisory_routes`；temporal owner 新增
  `latest_thinking_advisory_readout` 只读观测面。
  同时把三个 adapter hook 的失败处理修正为本 spec 冻结的“失败隔离”语义
  （log + 继续，不再 re-raise；kernel 侧契约违规仍 fail loudly）。测试：
  `tests/lifeform_e2e/test_thinking_temporal_consumer.py`。
- 2026-07-13: CP-21 temporal consumer slice. `vz-contracts.thinking` 新增
  `ControllerPressureAdvisory(track, pressure_delta, confidence, evidence)`，作为
  mid-reflection worker 给 temporal owner 的 compact payload。`FullLearnedTemporalPolicy.observe_thinking_artifact(...)`
  现在执行 owner / status / track / fingerprint gate：SHADOW (`apply_enabled=False`)
  只记录 evidence，不影响 `beta_t`；ACTIVE (`apply_enabled=True`) 才把 bounded
  pressure delta 纳入 switch pressure。该入口只消费不可变 `ThinkingArtifact`，
  不让 thinking worker 持有 temporal store。测试：
  `tests/contracts/test_thinking_envelopes.py`、`tests/test_temporal_interface.py`。
