# P04 Regime Identity

> Status: draft
> Last updated: 2026-04-06
> Primary owner: RegimeModule
> Primary slot: `regime`
> Primary consumer: `evaluation`

## 1. 包目标

把 regime 从 prompt 标签升级为正式 runtime identity，具备可记忆、可选择、可评估和可训练的边界。

## 2. 覆盖能力域与 spec

- Cognitive regime
- 对应文档：
  - `docs/specs/cognitive-regime.md`
  - `docs/DATA_CONTRACT.md`
  - `docs/EVALUATION_SYSTEM.md`

## 3. 前置条件

- `P00` 完成。
- `P02` 与 `P03` 的基础 contract 已稳定。

## 4. 范围内交付

- `RegimeIdentity` 与 `RegimeSnapshot` 的正式 owner 语义。
- active / previous / candidate regime 的状态发布。
- 历史效果读取接口与 memory 对接方式。
- 与 dual-track 状态的输入对齐。

## 5. 范围外内容

- 完整 learned regime policy。
- 高级 regime clustering 或自动发现。

## 6. 数据契约变更

- 固定 regime identity 的结构化表示。
- 如需保留人类可读理由，必须与机器可消费状态分开。

## 7. 实施步骤

1. 固定 regime 运行时 identity 结构，确保不退化为单纯字符串标签。
2. 定义 candidate scoring 和 active selection 的 contract。
3. 定义和 memory 的历史效果读取边界。
4. 定义和 evaluation 的效果反馈接口。
5. 保持语义方法优先，不使用关键词规则驱动 regime 选择。

## 8. 接线策略

### 未接线完成态

- `regime` 在 shadow 模式计算候选与 active regime。
- 输出仅用于日志和评估，不直接控制主链行为。

### 最终接线点

- `P09` 负责把 regime 选择接入正式 orchestrated flow。

## 9. 验收标准

- regime 具有结构化 identity，而非纯文本标签。
- 与 memory、dual-track、evaluation 的接口清晰。
- 切换与保持均可观测、可审计。

## 10. 退出条件与回滚

### 退出条件

- regime contract 稳定。
- 评估能够读取 regime 段和效果。

### 回滚触发

- regime 选择依赖硬编码关键词或 prompt hack。
- candidate/active shape 频繁变化影响 consumer。

### 回滚动作

- 回退到较小的 regime identity schema。
- 将 regime 限制在 shadow 模式，暂停正式接线。

## 11. 需要同步更新的文档

- `docs/specs/cognitive-regime.md`
- `docs/DATA_CONTRACT.md`
- `docs/EVALUATION_SYSTEM.md`
