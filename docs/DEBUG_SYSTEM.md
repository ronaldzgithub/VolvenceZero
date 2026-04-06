# EmoGPT Next-Gen — 调试与可观测性体系

> Status: draft
> Version: 0.1
> Last updated: 2026-03-25
> 对应需求: R8（快照优先、契约优先）、R11（可学习的内部状态表示）、R15（迁移可解释性和可回滚）

---

## 1. 设计目标

EmoGPT 不是一个静态模型——它是一个持续适应的有机体，拥有多时间尺度学习循环、连续记忆谱、时间抽象控制器和门控自修改。这意味着传统的"输入→输出"调试方式完全不够用。

**调试体系必须回答的核心问题**：

| 层面 | 问题 |
|------|------|
| **即时行为** | 这轮回复为什么是这样的？哪个模块的快照导致了这个决策？ |
| **控制流** | 当前抽象动作是什么？为什么切换/不切换？β_t 的值和趋势？ |
| **记忆** | 检索了哪些记忆？为什么检索了这些？遗漏了什么？ |
| **学习动态** | 信用分配是否合理？自修改是否越界？适应是否在漂移？ |
| **跨会话** | 持久记忆是否正确沉淀？regime 效果是否正确更新？ |
| **契约完整性** | 快照是否不可变？所有权是否被侵犯？消费者是否在重建生产者状态？ |

---

## 2. 分层可观测性架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layer 5: 纵向分析面板                          │
│  跨会话趋势 · 学习曲线 · 适应轨迹 · 漂移检测                       │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 4: 会话级仪表盘                          │
│  会话回放 · regime 时间线 · 信用分配热图 · 反思产物审查              │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 3: Turn 级检查器                         │
│  快照链检查 · 控制器状态追踪 · 记忆检索审计 · 双轨状态对比           │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 2: 契约守卫                              │
│  不可变性验证 · 所有权检查 · 依赖图合规 · 快照 schema 校验          │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 1: 结构化事件日志                        │
│  每个模块的快照发布/消费记录 · 时间戳 · 因果链                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer 1: 结构化事件日志

### 3.1 事件类型

系统中所有可观测行为统一为结构化事件流。

```python
@dataclass(frozen=True)
class DebugEvent:
    event_id: str               # 全局唯一 ID
    timestamp_ms: int           # 毫秒时间戳
    event_type: str             # 事件类型（见下表）
    module_owner: str           # 产生事件的模块
    wave_id: str                # 所属 wave/turn 标识
    session_id: str             # 所属会话标识
    payload: dict               # 事件特定数据（JSON-serializable）
    parent_event_id: str | None # 因果父事件（构建因果链）
```

| 事件类型 | 触发时机 | payload 关键字段 |
|----------|----------|-----------------|
| `snapshot.published` | 模块发布快照 | `slot_name`, `version`, `value_hash`, `description` |
| `snapshot.consumed` | 模块读取上游快照 | `consumer`, `slot_name`, `version` |
| `controller.switch` | Metacontroller 切换抽象动作 | `beta_t`, `old_z_hash`, `new_z_hash`, `reason` |
| `controller.hold` | Metacontroller 保持当前动作 | `beta_t`, `steps_held`, `z_hash` |
| `memory.write` | 记忆写入 | `entry_id`, `stratum`, `track`, `content_preview` |
| `memory.retrieve` | 记忆检索 | `query_hash`, `results_count`, `top_entry_ids` |
| `memory.promote` | 记忆提升 | `entry_id`, `from_stratum`, `to_stratum` |
| `memory.decay` | 记忆衰减 | `entry_id`, `old_strength`, `new_strength` |
| `credit.assign` | 信用分配 | `level`, `track`, `credit_value`, `source_event` |
| `selfmod.execute` | 自修改执行 | `target`, `gate`, `old_hash`, `new_hash`, `justification` |
| `selfmod.blocked` | 自修改被门控拦截 | `target`, `gate`, `reason` |
| `regime.switch` | Regime 切换 | `from_regime`, `to_regime`, `reason` |
| `regime.hold` | Regime 保持 | `regime_id`, `turns_held` |
| `reflection.start` | 慢反思启动 | `session_id`, `trace_length` |
| `reflection.complete` | 慢反思完成 | `memory_consolidated`, `policy_consolidated`, `lessons_count` |
| `evaluation.score` | 评估评分 | `family`, `metric`, `value`, `confidence` |
| `evaluation.alert` | 安全/有界性告警 | `alert_type`, `severity`, `details` |
| `contract.violation` | 契约违反（必须 fail loudly） | `violation_type`, `module`, `details` |

### 3.2 因果链

每个事件通过 `parent_event_id` 链接到触发它的上游事件，形成因果 DAG。

```
snapshot.published(substrate, v=42)
  └→ snapshot.consumed(temporal_abstraction, substrate, v=42)
       └→ controller.switch(beta=0.93, reason="tension_spike")
            └→ snapshot.published(temporal_abstraction, v=17)
                 ├→ snapshot.consumed(regime, temporal_abstraction, v=17)
                 │    └→ regime.switch(from="casual", to="emotional_support")
                 └→ snapshot.consumed(dual_track, temporal_abstraction, v=17)
                      └→ credit.assign(level="abstract_action", track="self")
```

### 3.3 日志存储与查询

| 维度 | 策略 |
|------|------|
| 存储格式 | 结构化 JSON Lines（每行一个事件） |
| 保留策略 | 热数据（最近 7 天）全量保留；冷数据按会话压缩归档 |
| 索引维度 | `session_id`, `wave_id`, `module_owner`, `event_type`, `timestamp_ms` |
| 查询接口 | 按会话/wave/模块/事件类型/时间范围过滤 |

---

## 4. Layer 2: 契约守卫

契约守卫是运行时断言层，确保系统的架构不变量不被破坏。违反必须 **fail loudly**（R8 + 编码规则 `no-swallow-errors`）。

### 4.1 不可变性守卫

```python
class ImmutabilityGuard:
    """
    验证快照在发布后未被修改。

    策略：发布时计算 value 的结构哈希，消费时重新计算并比对。
    违反时：抛出 ContractViolationError，记录 contract.violation 事件。
    """
```

**检查点**：
- 快照发布时：计算 `value_hash`，记录到事件日志
- 快照消费时：重新计算哈希，与发布时比对
- 检测到不一致：立即抛出异常 + 记录 `contract.violation` 事件

### 4.2 所有权守卫

```python
class OwnershipGuard:
    """
    验证每个 slot 只有一个 owner 在写入。

    策略：维护 slot → owner 注册表，发布时验证 owner 身份。
    违反时：抛出 OwnershipViolationError。
    """
```

**检查点**：
- 模块注册时：验证 `slot_name` 未被其他 owner 占用
- 快照发布时：验证发布者是注册的 owner
- 检测到第二 owner：立即抛出异常

### 4.3 依赖图守卫

```python
class DependencyGuard:
    """
    验证模块只消费声明的上游快照，不越界访问。

    策略：每个模块声明其消费的 slot 列表，消费时验证。
    违反时：抛出 DependencyViolationError。
    """
```

**检查点**：
- 模块从 upstream dict 读取快照时：验证 slot_name 在声明的依赖列表中
- 检测到未声明的依赖：立即抛出异常

### 4.4 Schema 守卫

```python
class SchemaGuard:
    """
    验证快照 value 符合声明的 frozen dataclass schema。

    策略：发布时验证 value 类型和字段完整性。
    违反时：抛出 SchemaViolationError。
    """
```

**检查点**：
- 快照发布时：验证 `value` 是声明类型的实例
- 验证所有必需字段存在且类型正确
- 验证 `frozen=True`（不可变性）

### 4.5 门控守卫

```python
class GateGuard:
    """
    验证自修改操作不越过声明的门控级别。

    策略：每个自修改目标有声明的 ModificationGate，执行前验证。
    违反时：阻止修改 + 记录 selfmod.blocked 事件。
    """
```

**检查点**：
- 自修改执行前：验证当前上下文（在线/后台/离线）匹配目标的门控级别
- 门控不匹配：阻止修改，记录 `selfmod.blocked` 事件
- 所有自修改必须记录 `selfmod.execute` 事件，包含 `is_reversible` 标记

---

## 5. Layer 3: Turn 级检查器

### 5.1 快照链检查器

每个 turn 结束后，检查器验证完整的快照传播链：

```
输入: 本 turn 所有 snapshot.published 和 snapshot.consumed 事件
输出: SnapshotChainReport

检查项:
├── 所有模块是否都发布了快照？（缺失 = 模块可能挂起）
├── 版本号是否单调递增？（回退 = 状态回滚异常）
├── 消费的版本是否是最新发布的版本？（滞后 = 传播延迟）
├── 是否有快照被发布但无人消费？（孤儿 = 可能的配置错误）
└── 因果链是否完整？（断链 = 事件丢失）
```

### 5.2 控制器状态追踪器

追踪 Metacontroller 的决策过程：

```
ControllerTrace (per turn):
├── beta_t 值和趋势（最近 N 轮）
├── 当前控制器代码 z_t 的语义描述
├── 自上次切换以来的步数
├── 切换/保持的决策理由
├── 残差流控制器 U_t 的范数变化
└── 与 regime 选择的一致性检查
```

**异常检测**：
- `β_t` 长期卡在中间值（0.3-0.7）→ 切换单元未收敛到准二值行为
- `steps_since_switch` 异常大 → 控制器可能卡死在某个抽象动作
- `steps_since_switch` 异常小 → 控制器可能在频繁无意义切换
- `z_t` 范数突变 → 控制器代码空间可能不稳定

### 5.3 记忆检索审计器

审计每轮记忆检索的质量：

```
MemoryRetrievalAudit (per turn):
├── 检索查询的语义摘要
├── 返回的记忆条目（ID、内容摘要、相关性分数）
├── 按轨道分布：World vs Self
├── 按层级分布：transient vs episodic vs durable
├── 检索耗时
└── 人工标注接口：标记遗漏/噪声（用于离线改进）
```

**异常检测**：
- 检索结果全部来自同一层级 → 可能的层级偏差
- 检索结果全部来自同一轨道 → 可能的轨道偏差
- 检索结果为空但记忆库非空 → 检索失效
- 检索耗时突增 → 索引性能问题

### 5.4 双轨状态对比器

对比 World Track 和 Self Track 的状态：

```
DualTrackComparison (per turn):
├── 两轨活跃目标对比
├── 两轨张力水平对比
├── 跨轨道张力值和趋势
├── 信用分配的轨道分布
└── 控制器代码 z_task vs z_rel 的余弦相似度
```

**异常检测**：
- 跨轨道张力持续高位 → 两轨目标严重冲突，需要人工审查
- 信用分配严重偏向一轨 → 另一轨可能被忽视
- `z_task` 和 `z_rel` 高度相似 → 双轨可能退化为单轨

---

## 6. Layer 4: 会话级仪表盘

### 6.1 会话回放

完整的会话交互回放，每轮附带所有模块的快照状态：

```
SessionReplay:
├── 用户输入序列
├── 系统响应序列
├── 每轮的完整快照链（所有 slot 的快照）
├── 时间线标注：
│   ├── regime 切换点
│   ├── 控制器切换点（β_t > threshold）
│   ├── 记忆写入/提升/衰减事件
│   ├── 信用分配事件
│   └── 自修改事件
└── 可按模块/事件类型过滤
```

### 6.2 Regime 时间线

```
RegimeTimeline (per session):
┌────────┬──────────────┬────────────┬──────────────┬─────────┐
│ casual │ emotional    │ guided     │ emotional    │ casual  │
│ social │ support      │ exploration│ support      │ social  │
├────────┼──────────────┼────────────┼──────────────┼─────────┤
│ T1-T3  │ T4-T8        │ T9-T14     │ T15-T18      │ T19-T22 │
│        │ ↑用户焦虑     │ ↑探索触发   │ ↑rupture检测  │ ↑修复完成│
└────────┴──────────────┴────────────┴──────────────┴─────────┘
```

标注信息：
- 每次切换的触发原因
- 每个 regime 段的持续轮数
- 每个 regime 段的评估分数
- 与控制器切换的对齐度

### 6.3 信用分配热图

```
CreditHeatmap (per session):

              T1   T2   T3   T4   T5   T6   ...
token         0.3  0.5  0.4  0.2  0.6  0.3
turn          0.4  0.3  0.5  0.7  0.4  0.5
session       ─────────── 0.6 ───────────── (会话结束时计算)
abstract_act  ── 0.3 ──┤── 0.7 ──────┤── 0.4 ──
long_term     ──────────────── (反思后计算) ────────────

World Track:  0.2  0.4  0.3  0.1  0.5  0.2
Self Track:   0.5  0.2  0.4  0.8  0.3  0.5
```

### 6.4 反思产物审查

慢反思完成后的产物检查：

```
ReflectionAudit:
├── 记忆沉淀:
│   ├── 新增持久记忆条目（内容、轨道、标签）
│   ├── 提升的记忆（从哪层到哪层、理由）
│   ├── 衰减的记忆（理由、衰减幅度）
│   └── 更新的信念（变更前后对比）
├── 策略沉淀:
│   ├── 控制器参数更新（变更幅度、方向）
│   ├── 策略先验更新（变更前后对比）
│   └── Regime 效果评分更新
├── 质量检查:
│   ├── 教训是否泛化（非特定会话的表面摘要）
│   ├── 记忆沉淀与策略沉淀是否一致
│   └── 是否有遗漏的重要张力
└── 人工标注接口：标记错误沉淀/遗漏
```

当前实现口径：

- P07 默认输出 proposal-first 的 reflection snapshot
- writeback mode 和 review_required 作为审查入口
- 正式写回默认关闭，先审计再放大范围

---

## 7. Layer 5: 纵向分析面板

### 7.1 学习曲线追踪

跨会话追踪系统的学习进展：

```
LearningCurves:
├── 评估族分数趋势（6 族 × 多会话）
├── 记忆系统增长曲线（各层级条目数、提升/衰减率）
├── 控制器稳定性（z_t 方差趋势、切换频率趋势）
├── Regime 效果趋势（各 regime 的历史评分）
└── 信用分配分布趋势（各级别、各轨道的分配比例）
```

### 7.2 适应轨迹

追踪系统对特定用户的适应过程：

```
AdaptationTrajectory:
├── 用户模型演化（持久记忆中用户相关条目的变化）
├── 关系模型演化（信任水平、依附风格估计的变化）
├── 策略偏好演化（控制器先验的变化方向）
├── Regime 使用模式变化
└── 个性化稳定性（适应是否收敛而非振荡）
```

### 7.3 漂移检测

检测系统是否在适应过程中发生有害漂移：

| 漂移类型 | 检测方法 | 告警条件 |
|----------|----------|----------|
| 基底漂移 | 基底层输出分布的 KL 散度 | 冻结模型不应有显著变化 |
| 控制器漂移 | z_t 分布的滑动窗口统计 | 方差持续增大或均值单调偏移 |
| 记忆漂移 | 持久记忆的语义聚类变化 | 核心信念被静默替换 |
| Regime 漂移 | Regime 使用频率分布变化 | 某 regime 使用率异常归零 |
| 信用漂移 | 信用分配的轨道/级别分布变化 | 某轨道信用持续归零 |
| 安全漂移 | 安全评估分数趋势 | 安全分数持续下降 |

**漂移响应**：
- 轻度漂移（统计显著但幅度小）→ 记录告警，继续监控
- 中度漂移（幅度超过阈值）→ 触发人工审查，暂停相关自修改
- 重度漂移（核心不变量被破坏）→ 回滚到最近的安全检查点

### 7.4 自修改审计轨迹

完整的自修改历史，支持回滚：

```
SelfModificationAuditTrail:
├── 时间线：所有 selfmod.execute 和 selfmod.blocked 事件
├── 每次修改的:
│   ├── 修改目标和门控级别
│   ├── 修改前后值的哈希（可回滚）
│   ├── 修改理由
│   ├── 修改后的评估分数变化
│   └── 是否已回滚
├── 统计:
│   ├── 各门控级别的修改频率
│   ├── 修改成功率（改善 vs 退化）
│   └── 回滚率
└── 回滚接口：按修改 ID 回滚到修改前状态
```

---

## 8. 调试工作流

### 8.1 "这轮回复为什么是这样的？" 工作流

```
1. 定位 turn
   └→ 按 session_id + wave_id 查询事件日志

2. 查看快照链
   └→ 读取本 turn 所有 snapshot.published 事件
   └→ 检查每个模块的 description 字段（模块自身生成的状态描述）

3. 追踪控制器决策
   └→ 读取 controller.switch / controller.hold 事件
   └→ 查看 beta_t 值、z_t 语义描述、切换理由

4. 审查记忆检索
   └→ 读取 memory.retrieve 事件
   └→ 检查检索结果与回复的相关性

5. 检查 regime 状态
   └→ 读取 regime 快照
   └→ 确认 regime 选择与当前情境的匹配度

6. 回溯因果链
   └→ 从回复生成事件出发，沿 parent_event_id 回溯
   └→ 找到根因事件
```

### 8.2 "学习是否在正确方向？" 工作流

```
1. 拉取纵向数据
   └→ 最近 N 个会话的评估分数趋势

2. 检查漂移指标
   └→ 各类漂移检测结果

3. 审查自修改历史
   └→ 最近的自修改事件及其效果

4. 审查反思产物
   └→ 最近 N 次反思的记忆沉淀和策略沉淀质量

5. 对比适应轨迹
   └→ 用户模型和关系模型的演化方向是否合理
```

### 8.3 "契约是否被破坏？" 工作流

```
1. 检查契约违反事件
   └→ 查询所有 contract.violation 事件

2. 运行契约守卫全量检查
   └→ ImmutabilityGuard: 抽样验证快照哈希
   └→ OwnershipGuard: 验证 slot 注册表一致性
   └→ DependencyGuard: 验证依赖声明与实际消费一致
   └→ SchemaGuard: 验证所有快照的 schema 合规

3. 检查快照链完整性
   └→ 是否有版本回退
   └→ 是否有孤儿快照
   └→ 是否有断裂的因果链
```

---

## 9. 快照 Diff 工具

支持对比任意两个快照版本的差异：

```python
@dataclass(frozen=True)
class SnapshotDiff:
    slot_name: str
    version_a: int
    version_b: int
    added_fields: tuple[str, ...]       # 新增字段
    removed_fields: tuple[str, ...]     # 删除字段
    changed_fields: tuple[tuple[str, str, str], ...]  # (field, old_value_repr, new_value_repr)
    description_diff: str               # description 字段的文本 diff
```

**使用场景**：
- 对比同一模块相邻两轮的快照 → 理解单步变化
- 对比会话首尾的快照 → 理解会话级变化
- 对比跨会话的快照 → 理解长期适应

---

## 10. 检查点与回滚

### 10.1 检查点策略

| 检查点类型 | 触发时机 | 保存内容 | 保留策略 |
|-----------|----------|----------|----------|
| Turn 检查点 | 每轮结束 | 所有模块的快照 | 会话内保留，会话后压缩 |
| Session 检查点 | 会话结束 | 所有模块快照 + 记忆系统完整状态 | 保留最近 N 个会话 |
| Reflection 检查点 | 反思前 | 反思输入的完整状态 | 与 Session 检查点同生命周期 |
| SelfMod 检查点 | 自修改前 | 被修改目标的当前值 | 永久保留（支持回滚） |
| Safety 检查点 | 安全评估通过时 | 全系统状态 | 永久保留（安全基线） |

### 10.2 回滚机制

```
回滚粒度:
├── Turn 级回滚: 回退到本轮开始前的状态
├── Session 级回滚: 回退到会话开始前的状态
├── SelfMod 回滚: 撤销特定的自修改操作
└── Safety 回滚: 回退到最近的安全检查点
```

**回滚约束**：
- 回滚必须记录为 `selfmod.execute` 事件（type=rollback）
- 回滚后必须重新运行契约守卫全量检查
- 回滚不可回滚（防止无限递归）

---

## 11. 开发时调试工具

### 11.1 模块独立测试

每个模块支持 `process_standalone` 模式，可脱离编排器独立运行：

```python
snapshot = await module.process_standalone(
    test_input=...,
    expected_output_schema=...,
)
```

**用途**：
- 单模块单元测试
- 预训练场景的独立验证
- 快照格式变更后的兼容性测试

### 11.2 快照注入

在编排器中注入自定义快照，覆盖特定模块的输出：

```python
override = {
    "memory": custom_memory_snapshot,
    "regime": custom_regime_snapshot,
}
result = await propagate(modules, upstream, overrides=override)
```

**用途**：
- 复现特定场景（注入特定记忆状态）
- 测试模块对异常输入的反应
- A/B 测试不同的上游状态

### 11.3 慢动作模式

逐步执行编排流程，每步暂停等待确认：

```
[Step 1/7] substrate → published snapshot v=42
  → 检查快照内容? [y/n/skip-all]

[Step 2/7] temporal_abstraction consuming substrate v=42
  → beta_t = 0.87, switching to new controller
  → 检查控制器状态? [y/n/skip-all]

[Step 3/7] temporal_abstraction → published snapshot v=17
  ...
```

### 11.4 契约违反断点

在契约守卫检测到违反时自动暂停，进入交互式调试：

```
⚠ ContractViolation: ImmutabilityGuard
  slot: memory, version: 23
  expected_hash: a3f2...
  actual_hash: b7c1...
  
  → 查看快照发布事件? [y/n]
  → 查看快照消费事件? [y/n]
  → 查看因果链? [y/n]
```

---

## 12. 性能观测

### 12.1 模块耗时追踪

```
ModuleLatencyReport (per turn):
├── substrate:              12ms
├── temporal_abstraction:   45ms  ⚠ (> 30ms threshold)
├── memory:                 23ms
├── dual_track:              8ms
├── regime:                  5ms
├── credit:                 11ms
├── evaluation:             15ms
└── total propagation:     119ms
```

### 12.2 记忆系统容量

```
MemoryCapacityReport:
├── transient:  142 entries (limit: 500)   28%
├── episodic:   1,203 entries (limit: 5000) 24%
├── durable:    8,421 entries (limit: 50000) 17%
├── derived:    356 entries (rebuilt on demand)
└── pending: 23 promotions, 45 decays
```

### 12.3 快照大小追踪

```
SnapshotSizeReport (per turn):
├── substrate:              2.3 KB
├── temporal_abstraction:   0.8 KB
├── memory:                 4.1 KB  ⚠ (> 3KB threshold)
├── dual_track:             1.2 KB
├── regime:                 0.6 KB
├── credit:                 1.8 KB
├── evaluation:             1.1 KB
└── total:                 11.9 KB
```

---

## 13. 与评估体系的集成

调试体系为评估体系提供原始数据：

| 调试层 | 提供给评估体系的数据 |
|--------|---------------------|
| Layer 1 事件日志 | 评估指标计算的原始事件流 |
| Layer 2 契约守卫 | 安全与有界性评估的输入 |
| Layer 3 Turn 检查器 | 交互质量和抽象质量评估的输入 |
| Layer 4 会话仪表盘 | 关系连续性和学习质量评估的输入 |
| Layer 5 纵向分析 | 长期适应和漂移检测的输入 |

详见 `docs/EVALUATION_SYSTEM.md`。

---

## 14. 参考文档

| 文档 | 用途 |
|------|------|
| `docs/next_gen_emogpt.md` | R8（快照优先）、R11（内部状态可发布）、R15（可回滚） |
| `docs/SYSTEM_DESIGN.md` | 系统架构：模块职责、数据流 |
| `docs/DATA_CONTRACT.md` | 快照 schema、Slot 注册表 |
| `docs/EVALUATION_SYSTEM.md` | 评估体系：调试数据如何回馈评估 |
| `.cursor/rules/no-swallow-errors-no-hasattr-abuse.mdc` | 契约违反必须 fail loudly |
