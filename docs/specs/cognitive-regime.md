# 认知 Regime Spec

> Status: draft
> Last updated: 2026-04-06
> 对应需求: R14

## 要解决的问题

如何让系统维护持久的交互模式身份，而非将其视为临时 prompt 标签？

## 关键不变量

- Regime 不是 prompt 标签，而是可记忆、可选择、可训练的持久身份
- Regime 在运行时状态中表示（非字符串标签）
- Regime 可从记忆中召回历史效果
- Regime 可被高层控制选择（由抽象控制层选择，而非硬编码规则）
- Regime 可通过延迟结果训练（通过信用分配回路）

## 工程挑战

- 设计 regime 的运行时表示（向量嵌入，不只是字符串标签）
- 实现 regime 的记忆化（可召回历史 regime 及其效果）
- 实现 regime 的高层控制选择（由抽象控制层选择，而非硬编码规则）
- 实现 regime 的延迟结果训练（通过信用分配回路）
- 场景检测必须使用语义级方法，不使用关键词匹配

## 算法候选

Regime 的选择和训练与 ETA 的 metacontroller 紧密相关：
- 控制器代码 `z_t` 的聚类可以对应不同 regime
- 切换单元 `β_t` 的切换时刻可以与 regime 切换对齐
- Internal RL 可以训练 regime 选择策略

### Regime 类型

- casual social contact（日常社交）
- acquaintance building（关系建立）
- emotional support（情感支持）
- guided exploration（引导探索）
- problem solving（问题解决）
- repair and de-escalation（修复与降级）

### Regime 运行时表示

```python
@dataclass(frozen=True)
class RegimeIdentity:
    regime_id: str
    name: str
    embedding: tuple[float, ...]        # 向量表示，非字符串标签
    entry_conditions: str
    exit_conditions: str
    historical_effectiveness: float     # 历史效果评分
```

## 接口契约

**消费的输入**：
- `temporal_abstraction` 快照：控制器状态（regime 选择与控制器切换对齐）
- `memory` 快照：regime 历史效果记忆
- `dual_track` 快照：两轨状态（regime 选择需考虑两轨）
- `evaluation` 快照：regime 效果评估分数

**产出的输出**：
- `regime` 快照：`RegimeSnapshot`
  - 当前活跃 regime
  - 前一个 regime（如有切换）
  - 切换原因
  - 候选 regime 及评分
  - 当前 regime 持续轮数

**当前实现口径**：

- P04 阶段先保证结构化 identity、稳定 candidate scoring 和可审计切换原因
- 当前选择逻辑基于 `memory`、`dual_track`、`evaluation` 的状态评分基线
- 当前 `RegimeModule` 已新增 owner-side `metacontroller` evidence ingest path：joint loop 可直接用 controller active label / guard rollback evidence 更新 strategy priors，而不改变 `regime` snapshot 契约
- 当前 `RegimeModule` 已新增 owner-side delayed attribution queue：上一轮 regime 选择会在后续 turn 的 evaluation 上结算，并通过 `delayed_outcomes` 发布
- 当前 `RegimeSnapshot.identity_hints` 作为 typed identity proposal 暴露给 reflection/memory owner；durable identity 写入仍由 reflection/memory owner 决定，Regime owner 不直接越权写 memory
- 后续可由 temporal / learned selector 替换，但不改变 `regime` snapshot 契约

**快照 schema**：见 `docs/DATA_CONTRACT.md` 3.6 节

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 契约式运行时（5.5）| 通过快照发布 regime 状态 |
| 依赖 | 时间抽象（5.2）| 控制器切换与 regime 切换对齐 |
| 依赖 | 连续记忆（5.3）| 从记忆中召回 regime 历史效果 |
| 依赖 | 双轨学习（5.4）| regime 选择需考虑两轨状态 |
| 被依赖 | 评估体系（5.7）| F5 中的 regime 对齐度评估 |
| 协作 | 信用分配（5.6）| regime 效果通过信用分配回路训练 |

## 变更日志

- 2026-04-09: next_gen_emogpt v2: regimes positioned as prediction spaces within the dual-track framework; regime selection weight updates driven by prediction error from delayed outcomes; repo default term: `abstract action` (paper synonym: `subgoal`)
- 2026-04-09: U03 Regime A/B verification: RegimeSelectionWeights confirmed to diverge from uniform (1.0) after delayed outcomes accumulate via process_standalone loop. Learned weights stay within [0.3, 2.0] range. effectiveness_trend published in RegimeSnapshot and verified non-empty after 4 turns.
- 2026-04-06: 补充 owner-side metacontroller evidence ingest 的当前实现口径
- 2026-03-25: 初始版本，从 SYSTEM_DESIGN.md 和 next_gen_emogpt.md 提取
