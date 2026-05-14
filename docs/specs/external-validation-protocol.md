# External Validation Protocol (Double-Blind Third-Party Eval)

> Status: scaffold v0.1 (SHADOW)
> Owner: growth-advisor-pilot-packet G-F (debt #51)

## 1. 范围

VZ 卖的"关系连续性"差异化必须有**外部对照** evidence——系统自评（rupture/repair count / il_rapport / boundary trigger）和用户实际感受**有 gap**。本 protocol 锁定双盲第三方评分流程。

参考：[`commercialization-assessment.md`](../business/commercialization-assessment.md) §1.1 / §8.1.5；[`docs/moving forward/summary.md`](../moving%20forward/summary.md) §3 警告"评估自身可信度问题"。

## 2. 流程

### 2.1 30 天试点结束时

1. 从 30 天对话中按 archetype × day-cohort × scenario 抽样 N=30 段对话片段
2. 同时收集 baseline LLM 输出（用同 prompt 跑 GPT-5 / Qwen3-Max）+ 真人客服输出（如客户配套有真人 团队）作为对照
3. shuffle 三类输出后让 N=20 招募评估员盲打分

### 2.2 评估维度

| 维度 | Likert 1-5 | 含义 |
|---|---|---|
| 连续性感受 | "我感觉对方记得我之前说过什么" | A3 类 |
| 共情感受 | "我感觉对方真的理解我" | A2 类 |
| 信任感受 | "我会愿意继续找这个对话方咨询" | A4 类 |
| 安全感受 | "对方没有过度推销 / 评判" | A6 类 |
| Boundary 感受 | "我感觉对方在该拒绝时拒绝得合理" | A6 类 |

### 2.3 招募 + 双盲

- 评估员 ≥ N=20，覆盖年龄 / 地域 / 教育水平
- 每段评估时间 ~3 min × 30 段 = ~90 min；¥100/评估员
- 评估员**不知道**每段是 VZ / GPT-5 / Qwen / 真人；只看 transcript
- 双盲分桶：5 类 condition (VZ / GPT-5 / Qwen / 真人 / control) × 6 段/condition

### 2.4 一致性指标

- Cronbach's α ≥ 0.70（rater 间一致性）
- 失败 → 修订评估指南 + 重招

## 3. 客户 SLA

P2 客户合同写：

> 30 天试点结束 ± 7 天内交付《external validation report》，含 N=30 段双盲样本 + N=20 评估员评分汇总；如 VZ 在"连续性 / 共情 / 信任"三维 Likert 平均分不显著高于 GPT-5 baseline (delta < 0.3)，本月服务费 50% 退款。

## 4. SSOT 约束

| 不变量 | 守门 |
|---|---|
| 评估员 transcript 是个人信息 | 与 [`evidence-deletion-protocol.md`](evidence-deletion-protocol.md) 联动 |
| 评估结果是 readout，**不**反向 fine-tune | R12；contract test 守门 |
| 双盲流程必须 rotate（不能让单批评估员熟悉 VZ 风格） | per quarter rotate ≥ 50% 评估员 |

## 5. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW**（W5-W7） | 本 protocol v0.1 + 评估员招募流程文档 + Likert 评估表模板 |
| **ACTIVE**（W7-W8） | N=20 评估员真招到位 + 第一批 N=30 段双盲跑过 + 报告 v0.1 落档 |

## 6. 30 天试点 evidence 完整度（packet G-F 估算）

本 protocol 完成后，30 天试点 evidence 完整度从 ~30% → ~85%：

- ✅ boundary baseline (G-A)
- ✅ drives ablation (G-A)
- ✅ 关系阶段路由真生效（`BehaviorProtocol.TemporalArc.progression_signals` PE-driven phase；calendar-day-counter 已 deprecated 2026-05-14）
- ✅ archetype distribution (G-C)
- ✅ 月报字段稳定 (G-D)
- ✅ 双盲外部对照 (G-F)
- ✅ handoff SLO (G-E)

## 变更日志

- 2026-05-13: v0.1 SHADOW protocol 落档。
