# P03 Dual Track Core

> Status: draft
> Last updated: 2026-04-06
> Primary owner: DualTrackModule
> Primary slot: `dual_track`
> Primary consumer: `credit`

## 1. 包目标

把 world/self 双轨做成正式运行时状态 owner，先形成明确的状态表示和冲突表示，再让其它包按轨道消费。

## 2. 覆盖能力域与 spec

- Dual-track learning
- 对应文档：
  - `docs/specs/dual-track-learning.md`
  - `docs/DATA_CONTRACT.md`
  - `docs/EVALUATION_SYSTEM.md`

## 3. 前置条件

- `P00` 完成。
- `P02` 最好已冻结基本 memory contract，但不是硬阻塞。

## 4. 范围内交付

- `world` / `self` / `shared` 的正式状态表示。
- 活跃目标、张力、轨道专属控制状态的快照 contract。
- 跨轨道张力表示与冲突状态位。
- 下游 credit / evaluation / regime 可消费的显式字段。

## 5. 范围外内容

- 复杂的双轨 RL 更新。
- 高级冲突裁决策略学习。

## 6. 数据契约变更

- 固定 `TrackState` 与 `DualTrackSnapshot`。
- 如需新增冲突仲裁建议字段，应由 dual-track owner 发布。

## 7. 实施步骤

1. 固定双轨状态最小字段集。
2. 设计跨轨道张力与冲突语义。
3. 明确与 memory、evaluation、credit 的接口边界。
4. 为后续 regime 和 temporal 预留轨道对齐接口。
5. 避免把双轨状态退化为纯文本描述。

## 8. 接线策略

### 未接线完成态

- `dual_track` 在 shadow 模式产生结构化状态。
- `credit` 和 `evaluation` 可以消费这些状态但不驱动正式主链。

### 最终接线点

- `P09` 将 `dual_track` 接为正式上游，供 `credit`、`evaluation`、`regime` 使用。

## 9. 验收标准

- 两轨在状态、信用、评估输入上保持分离。
- 跨轨道张力可观测、可追踪。
- 没有把 self/relationship 退化为 task 的附属字段。

## 10. 退出条件与回滚

### 退出条件

- 双轨 contract 稳定，consumer 可在不访问内部结构的情况下工作。

### 回滚触发

- 下游 consumer 无法区分两轨。
- 双轨状态必须依赖尚未稳定的 temporal internals 才能工作。

### 回滚动作

- 回退到更小的双轨状态集。
- 暂停 advanced consumer 接线，只保留 shadow 观察。

## 11. 需要同步更新的文档

- `docs/specs/dual-track-learning.md`
- `docs/DATA_CONTRACT.md`
- `docs/EVALUATION_SYSTEM.md`
