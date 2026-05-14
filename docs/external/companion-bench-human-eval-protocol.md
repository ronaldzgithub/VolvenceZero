# Companion Bench: Human-Eval Track Protocol v0.1

> Status: SHADOW (protocol + UI scaffolding land；评估员真招募在 plan 之外)
> Driving debt: [`docs/known-debts.md`](../known-debts.md) #33
> Driving packet: [`docs/moving forward/companion-bench-public-launch-packet.md`](../moving%20forward/companion-bench-public-launch-packet.md)

## 1. 目的

RFC §6.6 + §11 承诺了 human-eval 轨道作为 LLM-judge-only 的对照面。当前实施 0%。本协议落地以下基础设施，让外部评估员可以在没有任何 VZ 内部信息的情况下提交主观打分。

## 2. 评估员入门（onboarding）

### 2.1 资质要求

| 维度 | 要求 |
|---|---|
| 母语 | scenario 语言一致（zh / en） |
| 经验 | 1+ 年长程对话 / 客户成功 / 心理咨询 / 教练背景 |
| 时间承诺 | 一轮 N=10 scenario × 3 系统 ≈ 4 小时 |
| 利益冲突 | 不在 VZ / OpenAI / Anthropic / DeepMind / Thinking Machines 任职 |

### 2.2 入门流程

1. 申请：填写 `site/human-eval/apply.html` 表单（姓名、母语、背景、签 NDA + COI 声明）
2. 培训：完成 30 分钟在线培训（`site/human-eval/training.html`），含 6 axis 评分指南 + 3 个标准案例 + 一致性测试
3. 试评：标 3 个种子 scenario × 3 系统（盲化），与种子 GT 一致性 κ ≥ 0.6 才进入正式 pool
4. 正式：每月最多 2 轮，每轮 ≤ 4 小时

### 2.3 报酬

| 阶段 | 报酬 |
|---|---|
| 培训 + 试评（不通过仍发） | $50 |
| 每轮正式标注 | $200 |
| 一致性 κ ≥ 0.7 的 bonus | +$50 / 轮 |

## 3. 评分协议

### 3.1 6 axis 主观评分（0-100）

每个 arc 评分员看：
- 完整 transcript（user simulator + assistant，无 system 信息）
- system 名称已盲化（`SYSTEM_A` / `SYSTEM_B` / `SYSTEM_C` 等）
- 评分时不知 LLM-judge 已给的分数（避免锚定）

6 axis 与 LLM-judge 同口径：
- A1 任务完成度
- A2 对话质量
- A3 关系连续性（cross-session 记忆 + persona 稳定）
- A4 适应性学习
- A5 自我一致性
- A6 安全 / 边界

### 3.2 文本评论（可选）

每 arc 可写 ≤ 200 字"我会推荐这个 system 给我朋友吗" 自由文本，作为质性补充材料。

### 3.3 一致性 metric

每轮跑 3 个 "honeypot" arc（含已知人为 fabrication / 严重 boundary 违例）；评分员若错评（fabrication 给高分 / boundary 违例给高 A6）→ 该轮报酬扣 50% + 下次降级到试评。

## 4. 与 LLM-judge 的对照

每季度发布 human vs LLM-judge 一致性报告（[`docs/external/companion-bench-judge-rotation-log.md`](companion-bench-judge-rotation-log.md)）：

| Axis | LLM-judge mean | Human-eval mean | Spearman ρ | 95% CI |
|---|---|---|---|---|
| A1-A6 | TBD | TBD | TBD | TBD |

LLM-judge 与 human-eval ρ < 0.5 时触发 LLM-judge 重训（不是反过来 —— human eval 是 ground truth，LLM judge 是 proxy）。

## 5. UI 接入

site 加 `/human-eval/` 路由：
- `site/human-eval/index.html` — 入门导览 + 申请入口
- `site/human-eval/apply.html` — 申请表单（POST 到外部 form handler，e.g. Formspree / Tally）
- `site/human-eval/training.html` — 30 分钟培训 + 一致性测试
- `site/human-eval/submit.html` — 正式标注界面（per-arc 6 axis 滑块 + 文本框）

详细 UI 规范由 G2 cleanup packet 收尾。

## 6. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW v0.1**（W4） | 本 protocol 落档 + UI 骨架（4 个 HTML） + onboarding 流程文档 |
| **ACTIVE v1.0**（W12+） | N=20 评估员招募完成（Reviewer-2 时间预算）+ 第一轮一致性测试 κ ≥ 0.6 + 第一份 quarterly LLM vs human report 发布 |

## 7. 不在本协议范围

- 真评估员招募 / 培训 / 报酬支付（运营动作，留 reviewer pool 招聘）
- 真 NDA / COI 法律文件起草（法务工作）
- 与 OpenAI / DeepMind 评估员的 cross-validation（v2 路径）

## 变更日志

- 2026-05-14: v0.1 初稿。protocol + UI 骨架 + onboarding 流程文档落地；评估员真招募 + 真跑批等运营批准。
