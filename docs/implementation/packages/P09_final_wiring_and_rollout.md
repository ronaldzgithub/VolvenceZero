# P09 Final Wiring And Rollout

> Status: draft
> Last updated: 2026-04-06
> Primary owner: Integration / Rollout owner
> Primary slot: full chain integration
> Primary consumer: whole system

## 1. 包目标

作为最后一个收敛包，统一把各 owner 已完成的 contract 接到正式运行链中，并提供 rollout、回滚、观测和验收闭环。

## 2. 覆盖能力域与 spec

- Runtime integration
- Rollout
- Observability
- Acceptance
- 对应文档：
  - `docs/implementation/00_master_plan.md`
  - `docs/implementation/01_package_registry.md`
  - `docs/DEBUG_SYSTEM.md`
  - `docs/EVALUATION_SYSTEM.md`

## 3. 前置条件

- `P02` 到 `P08` 均已至少达到未接线完成态。
- `P05` 与 `P06` 已具备 gate 和 alert 能力。

## 4. 范围内交付

- 统一的主执行链 wiring。
- `disabled` / `shadow` / `active` 的 rollout 配置。
- 包级开关、kill switch、shadow compare 与 rollback playbook。
- 统一验收清单与最终态证据包。

## 5. 范围外内容

- 新 owner、新 slot 或新的主 contract 发明。
- 额外的大型算法升级。

## 6. 数据契约变更

- 原则上不新增主 schema，只消费前面包已经冻结的 contract。
- 如果 integration 发现 contract 缺口，应回退到对应 owner 包修复，而不是在接线包里补丁化新增。

## 7. 实施步骤

1. 逐包核对 owner、slot、consumer、接线级别。
2. 为每个包定义 rollout 阶段：disabled → shadow → active。
3. 先启用不会修改正式状态的 shadow 路径。
4. 在评估和调试证据通过后，再把 consumer 切到 active。
5. 清理或隔离所有临时 stub、legacy path 和重建逻辑。

## 8. 接线策略

### 未接线完成态

- 不适用；本包本身就是正式接线包。

### 正式接线顺序

1. `memory`
2. `dual_track`
3. `regime`
4. `evaluation`
5. `credit`
6. `reflection`
7. `temporal_abstraction`

说明：

- `evaluation` 必须先于 `credit` active。
- `reflection` 必须先经过 proposal-only 或 shadow。
- `temporal_abstraction` 最后接入，避免在其他 owner 未稳定前放大复杂度。

## 9. 验收标准

- 所有核心 slot 已正式注册并接入主链。
- 所有跨模块消费走快照 contract，无直接模块调用。
- 所有 adaptive 路径都有 gate、事件日志、告警和回滚点。
- 没有遗留的第二 owner 或隐式共享状态。

## 10. 退出条件与回滚

### 退出条件

- 全链在 active 模式下满足最小最终态验收。
- 关键 shadow / active 对比无明显退化。

### 回滚触发

- 任一核心 invariant 被破坏。
- gate 失效、自修改越权、主链不稳定或 contract violation 升高。

### 回滚动作

- 按包级开关依次回退：`temporal` → `reflection` → `credit` → `regime` → `dual_track`。
- 必要时回退到最近安全检查点。
- 保留 `evaluation` 和事件日志，确保故障可诊断。

## 11. 需要同步更新的文档

- `docs/implementation/00_master_plan.md`
- `docs/implementation/01_package_registry.md`
- `docs/DEBUG_SYSTEM.md`
- `docs/EVALUATION_SYSTEM.md`
