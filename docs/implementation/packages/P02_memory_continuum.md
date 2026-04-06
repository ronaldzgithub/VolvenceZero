# P02 Memory Continuum

> Status: draft
> Last updated: 2026-04-06
> Primary owner: MemoryModule
> Primary slot: `memory`
> Primary consumer: `dual_track`

## 1. 包目标

先把记忆系统做成正式 owner，包括 strata、summary、read/write API 和检索 contract，再逐步扩大到全系统消费。

## 2. 覆盖能力域与 spec

- Continuum memory
- Slow reflection 前置
- 对应文档：
  - `docs/specs/continuum-memory.md`
  - `docs/DATA_CONTRACT.md`
  - `docs/EVALUATION_SYSTEM.md`

## 3. 前置条件

- `P00` 完成。

## 4. 范围内交付

- transient / episodic / durable / derived 四层的 owner 语义。
- 正式记忆写入 API 与检索 API。
- 各层摘要与整体 description 的发布逻辑。
- 按轨道标记的记忆条目与检索结果。
- 提升、衰减、部分重建的状态位与队列表示。

## 5. 范围外内容

- 最终的策略沉淀写回。
- 高级检索优化或学习排序。

## 6. 数据契约变更

- 固定 `MemoryEntry`、`MemorySnapshot` 和 summary 字段。
- 如需要，增加 pending work queue / proposal 字段，但由 memory owner 自身发布。

## 7. 实施步骤

1. 定义四层记忆的 owner 边界和写入来源。
2. 设计 write API，不允许外部模块绕过 owner 写库。
3. 设计 retrieval contract，区分本轮检索结果与持久状态摘要。
4. 把提升、衰减和部分重建建模为显式状态，而不是隐式副作用。
5. 预留 `reflection` 写回入口，但在本包内只定义接口，不启用正式 writeback。

## 8. 接线策略

### 未接线完成态

- Memory 可 standalone 读写与发布摘要。
- `dual_track` 和 `regime` 可在 shadow 模式消费 `memory`。

### 最终接线点

- `P09` 把 `memory` 接到正式 turn 链与 `reflection` 异步写回链。

## 9. 验收标准

- 所有记忆写入都经过 Memory owner。
- `memory` 快照同时提供机器可消费状态和 owner 自描述。
- 检索、提升、衰减都有结构化事件或状态可观测。

## 10. 退出条件与回滚

### 退出条件

- strata、write API、retrieval contract 稳定。
- 其他模块不再持有记忆内部表示的第二所有权。

### 回滚触发

- 消费者需要读取 memory 内部私有结构。
- write API 无法支撑最小业务路径。

### 回滚动作

- 回退到上一版 memory snapshot shape。
- 保留 owner API，不允许临时外部直写作为补丁。

## 11. 需要同步更新的文档

- `docs/specs/continuum-memory.md`
- `docs/DATA_CONTRACT.md`
- `docs/EVALUATION_SYSTEM.md`
