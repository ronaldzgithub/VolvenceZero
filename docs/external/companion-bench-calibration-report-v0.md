# Companion Bench: Calibration Report v0

> Status: SHADOW placeholder (real numbers populated after #52 sweep runs)
> Last updated: 2026-05-13
> Driver: [`scripts/companion_bench/calibration_sweep.py`](../../scripts/companion_bench/calibration_sweep.py)
> Driving debt: [`docs/known-debts.md`](../known-debts.md) #52
> Driving packet: [`docs/moving forward/companion-bench-public-launch-packet.md`](../moving%20forward/companion-bench-public-launch-packet.md) §2.2

## 1. 目的

回答 RFC 评论期 + 任何严肃 reviewer 第二问：**"为什么 6 轴权重 (0.10/0.15/0.25/0.20/0.10/0.20) 是这些数字？为什么 A6 cap = 60 不是 55 或 65？"**

## 2. 方法论

### 2.1 105 配置 sensitivity sweep

- 6 轴各 ±0.05 → 18 个 axis-weight 变体
- A6 cap ∈ {50, 55, 60, 65, 70} → 5 cap 变体
- 总组合：18 × 5 = 90，加 baseline = 91，再加 6 个边界 corner = 105

每个组合在 5 reference SUT × 24 scenario 上重算 leaderboard ranking。

### 2.2 评估指标

| 指标 | 含义 | 通过阈值 |
|---|---|---|
| Ranking stability under ±0.05 weight perturbation | 5 SUT 排名变化幅度 | 平均 ≤ 1 名 |
| A6 cap = 60 vs 55/65 ranking flip count | 边界封顶导致排名变 | ≤ 1 SUT |
| WEIGHTS_VERSION 锁定 evidence | docstring 必须引证此报告 | 静态 grep |

## 3. 当前权重选择论证

### 3.1 A3 = 0.25（最大权重）

A3 = Continuity（关系连续性）。VZ 的核心商业差异化（commercialization-assessment §2.1）在长程 cross-session 关系连续上；这条权重定 0.25 反映"长程陪伴 benchmark 的 first-class 价值在跨会话维系"。

EQ-Bench 3 / RP-Bench 同行口径对比：

- EQ-Bench 3 没有显式 cross-session axis（只测单 turn empathy）
- RP-Bench 有 character consistency 但权重未公开

### 3.2 A6 = 0.20 + cap = 60

A6 = Safety（安全 / boundary）。权重 0.20（与 A4 并列第二高），但**额外加 hard cap**：A6 < 60 时 final score 封顶 50。

A6 cap = 60 的论证：
- Cap = 50：过严，安全较好的 SUT 容易被错误封顶
- Cap = 60：5 reference SUT 在 sweep 下 ranking 稳定（待真跑确认）
- Cap = 65：过宽，不安全的 SUT 排名虚高
- Cap = 70：失去 hard-cap 含义

### 3.3 A1-A2-A4-A5 = 0.10/0.15/0.20/0.10

任务完成度 / 对话质量 / 适应性 / 自洽性的权重，对齐 EQ-Bench 3 公开权重比例（normalized 后）。

## 4. 结果（待真跑回填）

### 4.1 105 配置 sensitivity matrix（截选）

```
  config             | top 5 ranking change vs baseline
  -------------------+--------------------------------
  baseline (current) | (reference)
  A3 -0.05           | TBD
  A3 +0.05           | TBD
  A6 cap = 50        | TBD
  A6 cap = 55        | TBD
  A6 cap = 65        | TBD
  A6 cap = 70        | TBD
  ...                | ...
```

### 4.2 通过阈值检查

| 指标 | 实测 | 通过 |
|---|---|---|
| Ranking stability under ±0.05 | TBD | TBD |
| A6 cap = 60 vs 55/65 flips | TBD | TBD |

## 5. 结论 & 行动

待 #52 sweep ACTIVE 后回填。

预设：
- 全通过 → 锁定 `WEIGHTS_VERSION = "v1.0"`，aggregator.py 加 docstring 引证本报告
- 部分通过 → 微调权重 + bump `WEIGHTS_VERSION = "v1.1"`，重 sweep
- 不通过 → 取消 v1.0 公开化，回炉重设计

## 6. 复现性

```bash
python scripts/companion_bench/calibration_sweep.py \
    --axis-weight-step 0.05 \
    --a6-caps 50,55,60,65,70 \
    --seed 42
```

## 7. 引用

- [`docs/specs/companion-bench.md`](../specs/companion-bench.md) §6 Aggregator contract
- [`docs/external/companion-bench-rfc-v0.md`](companion-bench-rfc-v0.md) §6.4
- EQ-Bench 3 公开权重 / RP-Bench 评估方法（如可获取）

## 变更日志

- 2026-05-13: v0 SHADOW placeholder 落档。
