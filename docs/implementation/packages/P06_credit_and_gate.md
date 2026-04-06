# P06 Credit And Gate

> Status: draft
> Last updated: 2026-04-06
> Primary owner: CreditModule
> Primary slot: `credit`
> Primary consumer: `reflection`

## 1. 包目标

在已有 dual-track 和 evaluation 骨架上，落地层级信用分配与门控自修改，使 adaptive loop 先有守门能力，再扩大策略更新范围。

## 2. 覆盖能力域与 spec

- Credit assignment
- Self-modification gate
- 对应文档：
  - `docs/specs/credit-and-self-modification.md`
  - `docs/DATA_CONTRACT.md`
  - `docs/EVALUATION_SYSTEM.md`
  - `docs/DEBUG_SYSTEM.md`

## 3. 前置条件

- `P03` 完成。
- `P05` 完成。

## 4. 范围内交付

- token / turn / session / long-term / abstract_action 层级的信用记录结构。
- ModificationGate 的执行策略与阻断条件。
- 自修改记录与 rollback metadata。
- 与 evaluation 告警、dual-track 状态、temporal 状态的对接接口。

## 5. 范围外内容

- 完整 RL 训练。
- 自动在线更新所有 controller 参数。

## 6. 数据契约变更

- 固定 `CreditRecord`、`SelfModificationRecord`、`CreditSnapshot`。
- 明确自修改记录必须包含理由、哈希、可回滚性。

## 7. 实施步骤

1. 固定信用分配层级及其最小语义。
2. 定义 gate 规则与运行时上下文映射。
3. 把 evaluation score 和 alert 映射到 gate 决策。
4. 为 reflection 提供结构化信用输入。
5. 为 future temporal controller credit 保留抽象动作级入口。

## 8. 接线策略

### 未接线完成态

- `credit` 可在 shadow 模式分配信用并记录 gate 决策。
- 所有自修改先只允许记录 proposal 或 blocked event。

### 最终接线点

- `P09` 将 `credit` 接成正式上游，驱动 reflection 和 rollout gate。

## 9. 验收标准

- 层级信用有结构化记录，不是单一数值。
- gate 能阻止越权自修改。
- 每次自修改或拦截都可追溯、可回滚。

## 10. 退出条件与回滚

### 退出条件

- gate 规则稳定。
- 下游 reflection 和 rollout 可以依赖 `credit` 快照。

### 回滚触发

- gate 无法阻止越权更新。
- 信用分配退化为不可解释的单一黑箱分值。

### 回滚动作

- 收紧到只记录不执行的模式。
- 暂停自修改，只保留评估与日志。

## 11. 需要同步更新的文档

- `docs/specs/credit-and-self-modification.md`
- `docs/DATA_CONTRACT.md`
- `docs/EVALUATION_SYSTEM.md`
- `docs/DEBUG_SYSTEM.md`
