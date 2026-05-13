# Companion Bench: LLM Judge Robustness Report v0

> Status: SHADOW placeholder (real numbers populated after #48 sweep runs)
> Last updated: 2026-05-13
> Driver: [`scripts/companion_bench/judge_robustness_sweep.py`](../../scripts/companion_bench/judge_robustness_sweep.py)
> Driving debt: [`docs/known-debts.md`](../known-debts.md) #48
> Driving packet: [`docs/moving forward/companion-bench-public-launch-packet.md`](../moving%20forward/companion-bench-public-launch-packet.md) §2.1

## 1. 目的

回答 RFC 公开评论期 + 任何严肃 reviewer 第一问：**"你们的 leaderboard 是否被 LLM judge 的家族偏好（self-preference）污染了？"**

## 2. 方法论

### 2.1 多家族 judge sweep

固定 reference SUT 池（VZ + 4 家 SOTA），让 N=5 个 LLM judge 家族 (GPT-5 / Claude Opus 4.7 / DeepSeek V4 / Qwen3 / Gemini 2.5) 分别给出 per-axis × per-arc 评分。

### 2.2 评估指标

| 指标 | 含义 | 通过阈值 |
|---|---|---|
| Per-axis variance σ | 同一 (SUT, arc) 在 5 judge 下分数标准差 | < 8.0 |
| Spearman ρ matrix | 跨 judge 的 SUT 排名相关性 | 平均 ρ ≥ 0.75 |
| Worst-case ranking change | 任意两 judge 给同 SUT 排名差 | ≤ 1 名 |
| Self-preference Δ | 同家族 judge 给同家族 SUT 的分数 vs 跨家族平均 | < +5 分 |

### 2.3 跑分配置

- Reference SUT 池：5 个固定（VZ companion + 4 SOTA closed/open）
- Scenario subset: 24 公开 scenario × 1 seed
- 每 (SUT, judge) 跑一次（共 5×5 = 25 跑分组合）
- 估算 token cost: 见 [`companion-bench-cost-model-v0.md`](companion-bench-cost-model-v0.md)

## 3. 结果（待真跑回填）

### 3.1 Per-axis variance σ

```
  axis  |  σ across 5 judge families
  ------+----------------------------
  A1    |  TBD
  A2    |  TBD
  A3    |  TBD
  A4    |  TBD
  A5    |  TBD
  A6    |  TBD
```

### 3.2 Spearman matrix

```
              gpt5   claude47  deepseek4  qwen3   gemini25
  gpt5         1.00    TBD       TBD       TBD     TBD
  claude47     TBD     1.00      TBD       TBD     TBD
  deepseek4    TBD     TBD       1.00      TBD     TBD
  qwen3        TBD     TBD       TBD       1.00    TBD
  gemini25     TBD     TBD       TBD       TBD     1.00
```

### 3.3 Worst-case ranking change

TBD（每对 SUT 在 5 judge 下最大排名差）

### 3.4 Self-preference Δ

TBD（同家族 vs 跨家族平均的偏差）

## 4. 结论 & 行动

待 #48 sweep ACTIVE。预设结果分支：

| 结果 | 行动 |
|---|---|
| 全 4 阈值通过 | leaderboard 用 5-judge ensemble 平均；公开报告标 "judge-robust" |
| Per-axis σ > 8 但其他通过 | per-axis 单独标注 σ；leaderboard 仍发 |
| Spearman ρ < 0.75 | 取消 v1.0 公开化；扩大 judge 池 + 重新 sweep |
| Self-preference Δ > +5 | 强制 cross-family judge ensemble (排除 same-family judge×SUT) |

## 5. 复现性

```bash
python scripts/companion_bench/judge_robustness_sweep.py \
    --judge-families gpt5,claude47,deepseek4,qwen3,gemini25 \
    --reference-sut vz,gpt5,claude47,deepseek4,qwen3 \
    --seed 42
```

每季度跑一次，artifact 落 `artifacts/companion_bench/judge_robustness_sweep-<date>.json`。

## 6. 引用

- [`docs/specs/companion-bench.md`](../specs/companion-bench.md) §5 Judge contract
- [`docs/external/companion-bench-rfc-v0.md`](companion-bench-rfc-v0.md) §6.4 Aggregation
- MT-Bench / Chatbot Arena self-preference bias 文献

## 变更日志

- 2026-05-13: v0 SHADOW placeholder 落档。
