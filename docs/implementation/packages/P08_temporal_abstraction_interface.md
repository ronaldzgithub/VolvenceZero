# P08 Temporal Abstraction Interface

> Status: draft
> Last updated: 2026-04-06
> Primary owner: TemporalModule
> Primary slot: `temporal_abstraction`
> Primary consumer: `dual_track`

## 1. 包目标

先落地时间抽象的正式接口、状态 contract 和可替换实现位点，不把当前系统绑死在完整 ETA 训练闭环上。

## 2. 覆盖能力域与 spec

- Temporal abstraction
- Internal control
- 对应文档：
  - `docs/specs/temporal-abstraction.md`
  - `docs/specs/multi-timescale-learning.md`
  - `docs/DATA_CONTRACT.md`
  - `docs/EVALUATION_SYSTEM.md`

## 3. 前置条件

- `P00` 完成。
- `P01` 完成。

## 4. 范围内交付

- `ControllerState` 和 `TemporalAbstractionSnapshot` 的正式接口。
- switch state、hold state、controller code state 的最小结构。
- standalone training / inference interface。
- placeholder / heuristic / learned-lite / full learned 的实现插槽。

## 5. 范围外内容

- 完整 residual-stream 干预。
- 完整 Internal RL 训练基础设施。

## 6. 数据契约变更

- 固定 controller state 的 shape。
- 将可读语义描述和机器可消费状态拆开。
- 对未来 richer controller params 保留兼容扩展字段。

## 7. 实施步骤

1. 根据 `P01` 定义当前可实现的输入 surface。
2. 固定 `z_t`、`beta_t`、`steps_since_switch` 等基础状态。
3. 定义 temporal module 的 standalone 模式。
4. 设计 placeholder 与 learned implementation 的统一接口。
5. 将 F5 抽象质量评估需要的观测点显式暴露。

## 8. 接线策略

### 未接线完成态

- `temporal_abstraction` 在 `disabled` 或 `shadow` 模式发布稳定快照。
- 下游可读取 controller state，但不依赖其驱动正式主链决策。

### 最终接线点

- `P09` 决定 temporal snapshot 何时进入正式 orchestrated chain。

## 9. 验收标准

- temporal state 可发布、可调试、可替换。
- 下游 consumer 不直接依赖 temporal 内部私有实现。
- 接口允许从 placeholder 平滑升级到 learned controller。

## 10. 退出条件与回滚

### 退出条件

- `temporal_abstraction` shape 稳定。
- dual-track、regime、credit 能以 contract 方式消费它。

### 回滚触发

- temporal interface 依赖不可实现的 substrate 细节。
- 多个下游开始读取 temporal 私有字段。

### 回滚动作

- 回退到更小的 temporal snapshot。
- 暂停 active 接线，只保留 shadow 模式和 standalone 验证。

## 11. 需要同步更新的文档

- `docs/specs/temporal-abstraction.md`
- `docs/specs/multi-timescale-learning.md`
- `docs/DATA_CONTRACT.md`
- `docs/EVALUATION_SYSTEM.md`
