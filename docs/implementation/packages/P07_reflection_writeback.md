# P07 Reflection Writeback

> Status: draft
> Last updated: 2026-04-06
> Primary owner: ReflectionModule
> Primary slot: `reflection`
> Primary consumer: `memory`

## 1. 包目标

实现异步慢反思和 writeback 合约，让系统具备会话后沉淀能力，同时保持可禁用、可审计、可回滚。

## 2. 覆盖能力域与 spec

- Slow reflection
- Continuum memory writeback
- Credit-informed consolidation
- 对应文档：
  - `docs/specs/continuum-memory.md`
  - `docs/specs/credit-and-self-modification.md`
  - `docs/DATA_CONTRACT.md`
  - `docs/DEBUG_SYSTEM.md`

## 3. 前置条件

- `P02` 完成。
- `P05` 完成。
- `P06` 至少完成 shadow 级 gate。

## 4. 范围内交付

- `ReflectionSnapshot` 与两类产物：memory consolidation / policy consolidation。
- 会话后异步启动、执行、完成的状态 contract。
- 写回提案、写回执行、写回审计轨迹。
- writeback kill switch 和 review point。

## 5. 范围外内容

- 大规模离线训练。
- 完整自动策略更新。

## 6. 数据契约变更

- 固定 `ReflectionSnapshot` 的最小 shape。
- 明确 `memory`、`credit`、`temporal_abstraction` 接受的 writeback 边界。

## 7. 实施步骤

1. 定义 reflection 输入：trace、evaluation、credit、tensions。
2. 定义两类产物及其目标 owner。
3. 设计 writeback proposal 和 execute 的状态机。
4. 明确哪些写回可自动执行，哪些只允许审计或人工确认。
5. 为 session 级与 cross-session 级反思预留兼容接口。

## 8. 接线策略

### 未接线完成态

- Reflection 在 `disabled` 或 `shadow` 模式异步生成产物。
- 写回默认关闭或只写 proposal，不直接修改正式状态。

### 最终接线点

- `P09` 统一开启 reflection 的正式异步挂载与受控 writeback。

## 9. 验收标准

- reflection 不阻塞 turn 主链。
- 反思产物区分记忆沉淀与策略沉淀。
- 所有 writeback 都可审计、可禁用、可回滚。

## 10. 退出条件与回滚

### 退出条件

- 异步执行模型稳定。
- writeback 边界清晰，owner 不混乱。

### 回滚触发

- reflection 直接写坏 memory 或 credit 状态。
- 异步执行影响主链稳定性。

### 回滚动作

- 关闭正式 writeback，只保留 proposal。
- 退回到 session report 模式，暂停状态修改。

## 11. 需要同步更新的文档

- `docs/specs/continuum-memory.md`
- `docs/specs/credit-and-self-modification.md`
- `docs/DATA_CONTRACT.md`
- `docs/DEBUG_SYSTEM.md`
