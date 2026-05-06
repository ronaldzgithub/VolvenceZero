# Interlocutor State Spec

> Status: stable (W2 of ssot-cleanup-p0-p4)
> Last updated: 2026-05-06
> 对应需求: R8 (契约优先 / 快照优先), R11 (内部状态可发布 / 可命名), R15 (可回滚演进)

## 要解决的问题

EmoGPT v4.0 §13 描述了一个"压力驱动有脾气的 AI"——12 维 InterlocutorState
+ Resistance / Coaxing 区间映射 + RelationshipStage 渐进解锁。直接照抄那
张查表是关键词硬编码（违反 `no-keyword-matching-hacks`）。

VolvenceZero 落到第一性原理上：

* 12 个维度是**连续特征**，从既有 owner 快照派生（不读用户文本）；
* "区间"不是查表，而是**命名 zone bool**，由 owner 在快照内一次性发布；
* RelationshipStage 渐进解锁继续走 `vitals.recharge_per_regime`，不在这里重新发明。

W2 之前，12 维读出在三个消费者各自重建：`prompt_planner` / `response_synthesizer`
/ `LifeformSession.interlocutor_state` 都从同一组上游 snapshot 各自跑一遍 duck-typed
builder + 各自定义阈值。这是典型的 R8 违反：消费者重建了生产者的内部状态。

W2 把读出收敛成一个 SHADOW owner：`InterlocutorStateModule`，并把阈值常量收敛到一个
`InterlocutorThresholds` 类。

## 关键不变量

1. **唯一生产者**：`InterlocutorStateModule` 是 12-axis readout 的唯一所有者。
   消费者只读 `latest_active_snapshots["interlocutor_state"]`（或对应 SHADOW
   snapshot），不重建。
2. **唯一阈值源**：`InterlocutorThresholds` 是 zone 分类的 SSOT。任何想新增 zone
   或调阈值的工作只动这个类一处。
3. **zone bool 与 axis 自洽**：`InterlocutorState.__post_init__` 始终用
   `compute_zones()` 重写 zone bool；构造时传错的 zone bool 会被覆盖。
4. **冷启动安全**：`readout_confidence < 0.30` 时 `compute_zones` 返回全部 False；
   downstream consumer 不需要再做 confidence gate（但仍可读 `readout_confidence`
   表达 audit 强度）。
5. **不读文本**：所有 12 维的 pull/push 公式只读上游 owner 的结构化字段，
   绝不解析 `user_input`。

## 工程挑战

- `InterlocutorState` 是 `dataclass(frozen=True)`，所以 `__post_init__` 计算 zone
  必须用 `object.__setattr__`。
- 既有测试通过 `replace(neutral_state, emotional_weight=0.7)` 构造极端态。`replace`
  会触发 `__post_init__` —— zone bool 自动重算，测试无需改。
- `lifeform-core/lifeform.py` 在 SHADOW slot 缺失时仍要工作（legacy tests 不注册
  owner）。回退路径调用同一个 `readout_interlocutor_state`，所以行为与 owner 一致。

## 接口契约

### Owner

```python
class InterlocutorStateModule(RuntimeModule[InterlocutorStateSnapshot]):
    slot_name = "interlocutor_state"
    dependencies = (
        "regime", "dual_track", "evaluation",
        "prediction_error", "memory", "commitment",
    )
    default_wiring_level = WiringLevel.SHADOW
```

### Snapshot

```python
@dataclass(frozen=True)
class InterlocutorStateSnapshot:
    state: InterlocutorState  # 12 axes + 10 zone bools + readout_confidence + rationale
    description: str
```

`description` 是 owner 自己的 short audit string；消费者用它做日志和
dashboards，不重新拼。

### 12 axes (consumed by ETA / planner / synthesizer)

`engagement_intensity` / `self_disclosure_level` / `task_focus_level` /
`emotional_weight` / `cognitive_engagement` / `resistance_level` /
`openness_to_guidance` / `directness` / `trust_signal` (signed) /
`stability` / `rapport_warmth` / `pace_pressure`

### Zone bools (the SSOT-cleanup output)

| Zone | 触发条件 | 主要消费者 |
|------|----------|------------|
| `acknowledge_pressure_zone` | `emotional_high OR resistance_high OR trust_negative` | planner (添加 ACKNOWLEDGE_PRESSURE) |
| `emotional_high_zone` | `emotional_weight >= 0.55` | planner (rationale tag) |
| `resistance_high_zone` | `resistance_level >= 0.50` | planner (rationale tag) |
| `trust_negative_zone` | `trust_signal <= -0.10` | planner (rationale tag) |
| `repair_zone` | `resistance >= 0.30 OR trust_signal <= 0.05` | synthesizer (render repair variant) |
| `direct_task_zone` | `task_focus >= 0.685 AND directness >= 0.58 AND emotional <= 0.58` | synthesizer (render direct variant) |
| `emotional_render_zone` | `emotional >= 0.56 AND self_disclosure >= 0.65` | synthesizer (render emotional variant) |
| `pace_pressure_zone` | `pace_pressure >= 0.65` | planner (drop META + cap Qs) |
| `low_directness_zone` | `directness <= 0.40` | planner (cap Qs) |
| `cold_rapport_zone` | `rapport_warmth <= 0.40 AND engagement >= 0.30` | planner (add CONTINUITY_NOTE) |

低于 `min_confidence` (=0.30) 时所有 zone 均为 False。

## 与其他能力域的关系

- **生产**：通过 `dependencies` 声明的六个 owner 提供 input。owner 内部不调用任何
  其他模块的方法——只读 snapshot 字段。
- **消费**：
  - `lifeform-expression.prompt_planner` — `_apply_interlocutor_state` 读 zone bool；
    section add/drop / question budget cap 不再重新算阈值。
  - `lifeform-expression.response_synthesizer` — `_state_indicates_*` 读 zone bool；
    渲染 variant 选择不再重新算阈值。
  - `lifeform-core.LifeformSession.interlocutor_state` — 优先读 SHADOW snapshot；
    legacy fallback 通过 `readout_interlocutor_state` 重建（行为等价）。

## 不在范围

- 不接 ETA/NL 的 z_t 选择面：W2 是观察侧 SSOT，不动控制器。
- 不开 ACTIVE：v0 默认 SHADOW；ACTIVE 由后续 wave 在通过 matched-control gate
  后单独 promote。
- LLM 解析 user text 给 axis 提供 proposal 的 path 不在 W2，按 R8 是新工作。

## 变更日志

- **2026-05-06**：初版（W2）。模块从单文件拆成 `contracts.py` / `readout.py`
  / `owner.py`；新增 `InterlocutorThresholds` 单一阈值源、10 个 zone bool；
  `InterlocutorStateModule` 注册到 `final_wiring` SHADOW；`prompt_planner` /
  `response_synthesizer` / `lifeform-core` 改为读 zone bool。
