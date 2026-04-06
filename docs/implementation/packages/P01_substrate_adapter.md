# P01 Substrate Adapter

> Status: draft
> Last updated: 2026-04-06
> Primary owner: SubstrateModule
> Primary slot: `substrate`
> Primary consumer: `temporal_abstraction`, `memory`

## 1. 包目标

把“稳定基底”从研究假设收敛为可实现 contract，明确当前系统实际能暴露什么 substrate state，不能暴露什么。

## 2. 覆盖能力域与 spec

- Stable substrate
- Contract Runtime
- Multi-timescale learning 的基底边界
- 对应文档：
  - `docs/SYSTEM_DESIGN.md`
  - `docs/DATA_CONTRACT.md`
  - `docs/specs/multi-timescale-learning.md`

## 3. 前置条件

- `P00` 完成。

## 4. 范围内交付

- 明确当前 substrate 实现模式：open-weight hook、代理 latent surface、或 placeholder。
- 为 `substrate` slot 定义阶段化 contract。
- 将“研究终态 contract”和“当前实现 contract”区分记录。
- 固定 `model_id`、freeze 边界和可观测字段。

## 5. 范围外内容

- 完整 metacontroller 训练。
- 完整 residual-stream 级干预。

## 6. 数据契约变更

- 可能需要把 `SubstrateSnapshot` 分为“当前稳定字段”和“未来扩展字段”。
- 必须明确哪些字段是 optional / unavailable，而不是虚假承诺。

## 7. 实施步骤

1. 选定当前 substrate 接入假设。
2. 为不可实现字段设计降阶表示，例如摘要化 latent surface 或 feature surface。
3. 固定 `substrate` 快照的可用性等级和更新频率。
4. 为 future advanced substrate 预留兼容扩展位点。
5. 记录所有 consumer 在当前阶段允许依赖的字段集合。

## 8. 接线策略

### 未接线完成态

- `substrate` 可 standalone 发布稳定 shape 的快照。
- `memory` 和 `temporal_abstraction` 可在 shadow 模式消费它。

### 最终接线点

- `P09` 将以本包冻结的 `substrate` contract 串联正式主链。

## 9. 验收标准

- 文档明确回答当前能否读写残差流。
- 所有下游模块不会依赖不可实现的 substrate 字段。
- `substrate` schema 在后续包实施期间保持稳定或兼容扩展。

## 10. 退出条件与回滚

### 退出条件

- substrate 接入模式已明确。
- 下游 consumer 不再假设“必然存在完整残差流 hook”。

### 回滚触发

- 后续包出现对不可实现中间层信号的硬依赖。
- 当前 contract 无法覆盖最小 consumer 需求。

### 回滚动作

- 回退到更保守的 `substrate` shape。
- 强制下游 consumer 回到 shadow / placeholder 模式。

## 11. 需要同步更新的文档

- `docs/DATA_CONTRACT.md`
- `docs/SYSTEM_DESIGN.md`
- `docs/specs/multi-timescale-learning.md`
