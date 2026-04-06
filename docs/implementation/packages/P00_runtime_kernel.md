# P00 Runtime Kernel

> Status: draft
> Last updated: 2026-04-06
> Primary owner: Runtime / Orchestrator
> Primary slot: runtime infra
> Primary consumer: all modules

## 1. 包目标

冻结 next-gen runtime 的最小骨架，使后续所有模块都能在不全量接线的条件下独立落地。

## 2. 覆盖能力域与 spec

- Contract Runtime
- R8 / R11 / R15
- 对应文档：
  - `docs/specs/contract-runtime.md`
  - `docs/DATA_CONTRACT.md`
  - `docs/DEBUG_SYSTEM.md`

## 3. 前置条件

- 无。

## 4. 范围内交付

- `Snapshot` / `Module` / `UpstreamDict` 的正式定义。
- slot 注册表与 owner 注册机制。
- propagate 语义与模块执行顺序约束。
- 缺失 upstream、stub snapshot、`disabled` / `shadow` / `active` 的行为定义。
- Layer 1 / Layer 2 的最小运行时观测与守卫接口。

## 5. 范围外内容

- 具体 memory、dual-track、regime、temporal 的业务逻辑。
- 任何针对 foundation model 的控制逻辑。

## 6. 数据契约变更

- 固定 `Snapshot` 基类字段。
- 明确 slot 注册、版本递增和 owner 校验规则。
- 明确模块声明依赖与 DependencyGuard 的关系。

## 7. 实施步骤

1. 明确 `Module.process()` 与 orchestrator 的关系，消除“不可直接调用 process”与“propagate 调用 process”之间的歧义。
2. 定义 slot 注册流程、owner 唯一性检查、schema 校验入口。
3. 定义 `disabled`、`shadow`、`active` 的统一语义。
4. 定义缺失 upstream 的标准表示，避免各模块各自发明 fallback。
5. 固定基础事件日志结构，至少覆盖 `snapshot.published`、`snapshot.consumed`、`contract.violation`。

## 8. 接线策略

### 未接线完成态

- Runtime 能驱动单模块或多模块传播。
- 未接线模块可通过 stub snapshot 参与 propagation。
- 契约守卫与事件日志可工作。

### 最终接线点

- 由 `P09` 使用本包固定的 runtime 规则串联所有正式 slot。

## 9. 验收标准

- 所有模块共用同一套 `Snapshot` / `Module` 抽象。
- slot 不能出现第二 owner。
- 未声明依赖的上游消费会 fail loudly。
- `disabled` / `shadow` / `active` 行为在文档和实现上唯一。

## 10. 退出条件与回滚

### 退出条件

- 注册表、propagate 语义、缺省 upstream 语义稳定。
- 后续包不再需要各自补 runtime 规则。

### 回滚触发

- 发现多个包需要自定义 propagation 或 fallback。
- 守卫规则与后续包 contract 冲突。

### 回滚动作

- 回退到上一版 runtime contract。
- 暂停后续包接线，只保留 standalone 测试路径。

## 11. 需要同步更新的文档

- `docs/specs/contract-runtime.md`
- `docs/DATA_CONTRACT.md`
- `docs/DEBUG_SYSTEM.md`
