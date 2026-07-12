# Owner Hydration Spec

> Status: draft
> Last updated: 2026-05-12
> 对应需求: R5（连续记忆）, R6（反思与沉淀）, R8（快照优先 / 单一所有者）, R11（内部状态可发布）, R15（迁移可解释性 + 可回滚）
> 来源: long-horizon-closure Packet D。

## 要解决的问题

VolvenceZero 当前所有 owner（`SemanticStateStore`、`FollowupManager`、`VitalsModule`、`MemoryStore`、`RegimeModule` 等）都设计成 **进程内 single writer**。`MemoryStore` 是唯一具备跨进程持久化（`PersistenceBackend` + `FileSystemPersistenceBackend`）的 owner；其余 owner 重启进程就被擦回 bootstrap 默认值。

后果：

- 跨 session 的关系状态（`relationship_state`、`commitment`、`open_loop`、`user_model`）**不会**自动续接。
- `FollowupManager` 队列中的"还没追问完"的 commitment / open loop 在新 session 启动时丢失。
- `VitalsModule` 的 drive levels 在新 session 启动时被重置回 `initial_level`，"昨晚没和我说话所以现在我应该主动来问候你"这件事永远做不到。
- rupture-repair 写入了 DURABLE memory，但**没有路径**把这些 entries 反向重建为新 session 的 `relationship_state` 快照。

Owner Hydration 协议解决"会话间 owner 状态续接"这一类**可回滚的** persistence path。它**不是**第二个 owner，**不是**新的 truth source；它是把已经存在的 single writer 的状态在进程边界做 export / hydrate。

## 关键不变量

1. **R8 单 owner 不破坏**：每个支持 hydration 的 owner 自己实现 `export_persistence_snapshot()` 和 `hydrate_from_persistence(payload)`。外部 store 不直写 owner 内部状态。
2. **R5/R6 主链不绕开**：hydration **只读取已经持久化的事实**（`MemoryStore.DURABLE` 已存在的 entries / 已经 commit 的 owner export），不创建新 truth。
3. **fail-loudly**：hydration 失败必须抛具体异常（`HydrationVersionMismatchError` / `HydrationPayloadInvalidError` 等），不允许 bare `except: pass` 静默回退。`SHADOW` wiring 下也只是 log + 跳过这一个 owner，不掩盖错误。
4. **idempotent + bounded**：每个 owner 的 hydrate 在重复调用时收敛到相同状态；payload 大小有界（不允许无限累积）。
5. **可回滚**：`BrainConfig.owner_hydration_wiring: WiringLevel = ACTIVE` 默认（long-horizon-closure follow-up：production-grade 默认就是续接，避免静默退化为"每次重启重置"）。`SHADOW` = export+log 但不 import；`DISABLED` = 完全关闭（rollback 旋钮）。任何阶段都可秒切，对 anonymous / 无 backend 的 session 是 no-op。
6. **存储分层**：复用既有 `MemoryStore.persistence_backend`（`FileSystemPersistenceBackend` 等），新加 key 前缀 `owner_hydration/{owner_name}`。**不**新建 storage 抽象。
7. **单进程内 idempotent 写**：owner_hydration_writer 在 turn / scene 边界写出，保证不在同一 turn 内的 propagate 中段写入。

## 协议契约

### `vz-contracts` 中的新类型

```python
# packages/vz-contracts/src/volvence_zero/owner_hydration.py

@dataclass(frozen=True)
class OwnerPersistenceSnapshot:
    """Versioned, owner-published state usable for cross-session hydration.

    ``schema_version`` is OWNER-internal; bumping it requires a
    migration path or fail-loud rejection on hydrate.
    """
    owner_name: str            # stable owner identifier (e.g. "semantic_state")
    schema_version: int         # owner-internal
    payload: Mapping[str, Any]  # JSON-serialisable
    description: str = ""


class HydratableOwnerProtocol(Protocol):
    """Optional owner ability: dump + restore for cross-session continuity.

    Owners that do NOT implement this protocol simply do not
    participate in owner hydration; the kernel does not require it.
    """

    def export_persistence_snapshot(self) -> OwnerPersistenceSnapshot: ...

    def hydrate_from_persistence(
        self, snapshot: OwnerPersistenceSnapshot
    ) -> None:
        """Replace internal state with the snapshot's payload.

        MUST raise (not swallow) ``HydrationPayloadInvalidError`` /
        ``HydrationVersionMismatchError`` if payload is unusable.
        """


class HydrationError(Exception):
    """Base class for hydration failures."""

class HydrationVersionMismatchError(HydrationError):
    """Snapshot schema_version is unknown to the owner."""

class HydrationPayloadInvalidError(HydrationError):
    """Snapshot payload is structurally invalid (missing required keys, etc)."""

class HydrationOwnerMismatchError(HydrationError):
    """``snapshot.owner_name`` does not match the receiving owner."""
```

### 三个首批支持的 Owner

`OwnerHydrationStore.OWNER_HYDRATION_MATRIX` 是 owner-by-owner 决策的机器 SSOT：

| Owner | Decision | Storage key | Notes |
|---|---|---|---|
| `semantic_state` | `hydrate` | `owner_hydration/semantic_state` | 9 个 slot 的 records / completed_refs / revision_counts / record_lifecycle / record_followup_policy / record_outcome |
| `followup_manager` | `hydrate` | `owner_hydration/followup_manager` | `_pending` queue / `_seen_keys` / `_counter` |
| `vitals` | `hydrate` | `owner_hydration/vitals` | `_levels` / proactive cooldown / IQR baseline state |
| `protocol_registry` | `hydrate` | `owner_hydration/protocol_registry` | 当 application owner 暴露 `HydratableOwnerProtocol` 时持久化 protocol registry |
| `memory` | `external-owner` | `memory/store` | `MemoryStore` 已有 `save_to_backend` / `load_from_backend`，不复制到 owner hydration |
| `regime` | `explicit-no-hydrate` | `none` | 暂无 hydratable protocol；避免外部 store 成为第二 regime owner |
| `world_temporal` | `explicit-no-hydrate` | `none` | controller continuity 由 checkpoint / rare-heavy snapshot owner 负责 |
| `self_temporal` | `explicit-no-hydrate` | `none` | controller continuity 由 checkpoint / rare-heavy snapshot owner 负责 |

每个 owner 的 `schema_version` 从 `1` 起，owner-internal。

### Storage 路径

复用 `MemoryStore.persistence_backend`，每个 owner 以 stable key 写入：

```text
owner_hydration/semantic_state
owner_hydration/followup_manager
owner_hydration/vitals
```

Key 前缀使其在 listing 时可被识别为 hydration 类型（与 `memory/store` 区分）。

### LifeformTemplate v2 bridge（character live-through）

`lifeform-domain-character` 的 `LifeformTemplate` schema v2 可以携带
`owner_hydration_snapshots: tuple[OwnerPersistenceSnapshot, ...]`。这是模板
artifact 对 owner-published snapshot 的搬运，不是第二 owner：

- save path 只能接收 owner 自己 `export_persistence_snapshot()` 产出的
  `OwnerPersistenceSnapshot`，或 caller 已持有的同类型 snapshot。
- load path 不直写 `SemanticStateStore` 私有字段；在 full restore 路径，
  它把 snapshot 写入与 `MemoryStore.persistence_backend` 相同语义的
  backend，再由现有 `OwnerHydrationStore.hydrate_owner_if_present(...)`
  恢复 owner。
- alpha / per-user scoped path 使用 `BrainConfig.owner_hydration_seed_snapshots`
  seed-once：只有 `owner_hydration/<owner_name>` 不存在时才写入模板 seed；
  已经存在的 per-user owner 状态永远优先，避免模板覆盖用户关系演化。
- v1 模板无该字段时保持旧行为；v2 payload 版本不匹配仍由各 owner
  `hydrate_from_persistence` fail loudly。

这条桥主要用于逐章主观烘焙：character vertical 可以保存
`semantic_state` 的 9 owner spine，让 `give_birth` 后的 browser chat 不退回
只有 profile / memory 的浅层状态。

### Brain / Session 集成

```python
# vz-runtime/src/volvence_zero/brain.py

@dataclass(frozen=True)
class BrainConfig:
    ...
    # Packet D (long-horizon-closure): cross-session owner hydration.
    # ACTIVE by default — operators that want the legacy
    # "every session is fresh" behavior must explicitly opt out.
    owner_hydration_wiring: WiringLevel = WiringLevel.ACTIVE
```

`Brain.create_session(...)` 流程（伪代码）：

```python
def create_session(self, *, session_id, ...):
    memory_store = self._build_or_inject_memory_store(...)
    hydration_store: OwnerHydrationStore | None = None
    if self._config.owner_hydration_wiring != WiringLevel.DISABLED:
        backend = getattr(memory_store, "persistence_backend", None)
        if backend is not None:
            hydration_store = OwnerHydrationStore(
                backend=backend,
                wiring_level=self._config.owner_hydration_wiring,
            )
    runner = AgentSessionRunner(
        ...,
        memory_store=memory_store,
        hydration_store=hydration_store,  # NEW
    )
    return BrainSession(runner=runner)
```

`AgentSessionRunner.__init__` 在拿到 `hydration_store` 后，会立刻 hydrate `_semantic_state_store`（如果 backend 中有 payload）。

`LifeformSession` / `Lifeform` 接受 `hydration_store` 参数并 hydrate `_followup_manager` / `_vitals` (when present).

### 写出时机

写出由 explicit hook 触发，**不在 turn 中段**：

- `BrainSession.persist_owners()`：write semantic_state；用户在 scene close 或自动 hook 中调用
- `LifeformSession.end_scene(...)`：scene 边界自动调用 `BrainSession.persist_owners()` 并紧跟着 export `_followup_manager` / `_vitals` 自身。
- 重要：`SHADOW` wiring 下，写出仍然发生（用于观测 round-trip），但 read path 不应用回 owner。`ACTIVE` 下读+写都生效。

## 当前 proof surface

- `tests/contracts/test_owner_hydration_protocol.py`：`SemanticStateStore` / `FollowupManager` / `VitalsModule` 实例都满足 round-trip `hydrate(export()) == export()`。
- `tests/contracts/test_owner_hydration_failures_loud.py`：损坏的 payload 抛 typed exception；不静默吞。
- `tests/test_semantic_state_hydration.py` / `tests/test_vitals_hydration.py` / `tests/test_followup_manager_hydration.py`：单 owner round-trip 测试。
- `tests/longitudinal/test_cross_session_owner_hydration.py`（Packet E）：同 user_id + 同 memory_scope_root_dir → 销毁 BrainSession → 新 BrainSession → 上次的 commitment / rupture / vitals 仍然可见。

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|---|---|---|
| 复用 | 连续记忆系统（5.3） | 共用 `PersistenceBackend`；hydration key 与 memory key 不冲突 |
| 协作 | 语义状态一等 Owner（6A） | `SemanticStateStore` 的 9 个 slot 是首批 hydratable owner |
| 协作 | Lifeform Vitals（11A） | `VitalsModule` 是 lifeform-side 首批 hydratable owner |
| 协作 | AAC Commitment Lifecycle（14） | commitment lifecycle 跨 session 续接由 `SemanticStateStore` 的 hydration 提供 |
| 协作 | Rupture and Repair Loop（17A） | rupture-repair durable memory 已经在 `MemoryStore`，与 owner hydration 互补 |

## 回滚路径

| 触发 | 操作 |
|---|---|
| 系统级故障 | `BrainConfig.owner_hydration_wiring = DISABLED` |
| 单 owner schema 升级失败 | 升 `schema_version`；旧 payload 触发 `HydrationVersionMismatchError`；用户 / 操作者决定是否重建 |
| Backend IO 故障 | hydration_store 在 `__init__` 失败时 fail-loudly；`SHADOW` 不掩盖 |
| 数据污染 | 删除 backend 内的 `owner_hydration/{owner_name}` key 文件；下次启动等于第一次启动 |

## Non-goals

- **不**做 owner state 的"实时" replication 到远端
- **不**为 owner hydration 做新的 storage 抽象（复用 `PersistenceBackend`）
- **不**触碰 `MemoryStore` 的 hydration（它已经有 `save_to_backend` / `load_from_backend`）
- **不**新增第二 memory owner

## 变更日志

- 2026-07-13: long-horizon closure matrix。`OwnerHydrationStore` 新增
  `OwnerHydrationMatrixEntry` / `OWNER_HYDRATION_MATRIX`，把 semantic_state /
  followup_manager / vitals / protocol_registry 的 hydrate 决策、memory 的
  external-owner 决策，以及 regime / world_temporal / self_temporal 的
  explicit-no-hydrate 决策固化为可测试 readout。测试：
  `tests/contracts/test_owner_hydration_protocol.py`。
- 2026-05-12: 初稿 (Packet D — long-horizon-closure)，定义 `HydratableOwnerProtocol`、3 个首批 owner、`BrainConfig.owner_hydration_wiring`、SHADOW/ACTIVE/DISABLED 三态、复用 `MemoryStore.persistence_backend`。
- 2026-05-12: long-horizon-closure follow-up：默认 wiring 从 DISABLED 翻 ACTIVE；`LifeformSession.end_scene` 自动 `persist_owners()`，让 ACTIVE 默认对外有现实意义。Anonymous / 无 backend 的 session 仍是 no-op。
- 2026-07-13: 登记 `LifeformTemplate` v2 bridge。模板可搬运 owner 自己发布的 `OwnerPersistenceSnapshot`，加载时仍经 `OwnerHydrationStore` 恢复，不新增第二 owner。
- 2026-07-13: v2 bridge alpha path 补充 seed-once 语义：`owner_hydration_seed_snapshots` 只在 scoped backend 缺 key 时写入，既有 per-user owner state 优先。
