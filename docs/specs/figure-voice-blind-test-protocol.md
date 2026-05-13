# Figure Voice Blind Test Protocol

> Status: scaffold v0.1 (SHADOW)
> Last updated: 2026-05-13
> Owner: figure-evidence-packet G-C (debt #60)

## 1. 范围

L1 StylePriorInjector + L2 PersonaLoRA 的"听起来像 X"是否被人类听众感知到。3 condition × N=20 评估员双盲面板。

## 2. 三 condition

| condition | 说明 |
|---|---|
| **raw** | 不加 bundle，纯 substrate 回答 |
| **bundle** | L1 StylePrior + L3 GroundedDecoder + L4 ScopeRefuser，**不**加 LoRA |
| **bundle+lora** | bundle + L2 PersonaLoRA |

## 3. 流程

### 3.1 生成

`scripts/figure_voice_blind_test.py` 从 `data/figure_grounding_gt/<figure_id>/assertions.jsonl` 抽 N=30 question；每个 question × 3 condition = 90 段 Q→A，shuffle 后输出 CSV。

### 3.2 评估员

- N=20 评估员；不需要领域专家（普通听众即可，重点是"主观印象"）
- 每位评估员看到 90 段 shuffle 后的 Q→A，对每段评 Likert 1-5："这听起来像 Einstein 的程度"
- 不告诉评估员每段是哪个 condition

### 3.3 时间预算

- 每段 ~30 sec 阅读 + ~5 sec 打分 = 35 sec
- 90 段 × 35 sec = ~52 min × ¥100 = ¥1,750/评估员
- N=20 × ¥1,750 = ¥35k 预算（已在 figure-evidence-packet §3 P1 cost 估算内）

注：packet §3 简化为 ~17 min × N=20，是只跑 30 段不跑 90 段的版本；本 protocol 全 90 段是稳健版。

### 3.4 收集 + 分析

- 评估员通过 Google Form / 自建问卷站收集
- 计算每 condition 的 mean Likert + 95% CI
- 计算 Cronbach's α（rater 间 consistency）
- 计算 raw vs bundle delta + bundle vs bundle+lora delta

## 4. ACTIVE 通过 SLA

| 指标 | 阈值 |
|---|---|
| Cronbach's α | ≥ 0.7 |
| raw vs bundle delta | ≥ 0.5（L1+L3+L4 真有可感知效果）|
| bundle vs bundle+lora delta | ≥ 0.3（L2 LoRA 进一步可感知）|
| N=20 评估员退出率 | ≤ 25%（5 人退出可接受）|

如果"raw vs bundle delta < 0.5" → L1 StylePrior 不可独立卖（只能 L1+L2 套餐卖），影响 P1 minimum-viable 价格区间从 30 万 → 80 万。

## 5. 与 bundle / R12 的关系

`FigureArtifactBundle.voice_blind_test_report` 字段不进 integrity_hash。盲测结果是 readout，**不**反向 fine-tune StylePrior / LoRA（违反 R12）。

## 6. 与 #40 + #41 的耦合

debt #40（synthetic LoRA delta 经 LayerNorm 被吃掉）+ #41（真 Qwen PEFT 未跑）→ **bundle vs bundle+lora delta 在 SHADOW 阶段 = 0**（synthetic LoRA 不真改 forward）。本盲测真有意义需 #41 ACTIVE。

## 7. 退出标准

| 阶段 | 标准 |
|---|---|
| **SHADOW**（W1-W2） | `scripts/figure_voice_blind_test.py` --dry-run + 本 spec v0.1 |
| **ACTIVE**（Phase B；与 P1 第二款 figure 上线同步） | N=20 评估员真跑过；Cronbach's α ≥ 0.7；3 condition delta 数据落 bundle |

## 变更日志

- 2026-05-13: v0.1 SHADOW scaffold。
