# P05 Evaluation Backbone

> Status: draft
> Last updated: 2026-04-06
> Primary owner: EvaluationModule
> Primary slot: `evaluation`
> Primary consumer: `credit`

## 1. 包目标

把评估体系前置为正式基础设施，先落地最小但可工作的评估骨架、证据链和告警机制，避免后续出现“先自修改后评估”。

## 2. 覆盖能力域与 spec

- Evaluation
- Safety and boundedness 前置
- 对应文档：
  - `docs/EVALUATION_SYSTEM.md`
  - `docs/DATA_CONTRACT.md`
  - `docs/DEBUG_SYSTEM.md`

## 3. 前置条件

- `P00` 完成。

## 4. 范围内交付

- `EvaluationRecord` / `EvaluationReport` / `EvaluationSnapshot` 的最低骨架。
- turn / session 两级的最小评估通路。
- 证据字符串、告警级别、信号来源字段。
- 与 credit、gate、reflection 的反馈接口。

## 5. 范围外内容

- 完整 longitudinal 纵向评估。
- 全量自动化 judge 或复杂 benchmark。

## 6. 数据契约变更

- 固定最小 `EvaluationSnapshot`。
- 明确哪些指标为必填、哪些可延后补充。

## 7. 实施步骤

1. 选定第一阶段必须可计算的评估指标。
2. 定义 turn/session 两级最小评估记录。
3. 固定告警严重级别和回馈出口。
4. 将调试事件与评估记录建立最小映射。
5. 为 later cross-session / longitudinal 评估预留扩展位点。

## 8. 接线策略

### 未接线完成态

- `evaluation` 已在主链 active 发布最小快照。
- 下游 `credit` 还未 fully active 也能消费该信号。

### 最终接线点

- `P09` 将 `evaluation` 的输出纳入全量 gate 和 rollout 决策。

## 9. 验收标准

- 至少存在可运行的 turn / session 评估。
- 每个评估分数都包含 evidence 或 signal source。
- 告警可以被 gate 或 rollout 逻辑消费。

## 10. 退出条件与回滚

### 退出条件

- 评估基础 schema 稳定。
- 下游模块可以依赖最小 evaluation 信号。

### 回滚触发

- 评估字段频繁变化导致 consumer 不稳定。
- 评估无法提供基本 gate 输入。

### 回滚动作

- 回退到更小的指标集和 snapshot shape。
- 先保留日志与告警，不扩大自动 gate 行为。

## 11. 需要同步更新的文档

- `docs/EVALUATION_SYSTEM.md`
- `docs/DATA_CONTRACT.md`
- `docs/DEBUG_SYSTEM.md`
