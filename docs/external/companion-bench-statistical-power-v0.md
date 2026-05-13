# Companion Bench: Statistical Power Analysis v0

> Status: SHADOW placeholder
> Driver: [`scripts/companion_bench/statistical_power_analysis.py`](../../scripts/companion_bench/statistical_power_analysis.py)
> Driving debt: [`docs/known-debts.md`](../known-debts.md) #54
> Driving packet: [`docs/moving forward/companion-bench-public-launch-packet.md`](../moving%20forward/companion-bench-public-launch-packet.md) §2.4

## 1. 目的

回答 reviewer 第四问：**"24 公开 + 96 held-out = 120 scenario 够不够大让排名差异有统计学意义？"**

## 2. 方法论

Bootstrap CI 驱动的 power 曲线：在 n ∈ {24, 48, 96, 120, 200} 多档下，估算两 SUT ELO 差距多少时才能以 95% 置信度判定 distinguishable。

## 3. 结果（待真跑回填）

### 3.1 Power curve

```
  n_scenarios | distinguishable threshold ELO
  ------------+------------------------------
  24          | TBD
  48          | TBD
  96          | TBD
  120         | TBD
  200         | TBD
```

### 3.2 Per-SUT ELO ± 95% CI

```
  SUT      | mean ELO | 95% CI
  ---------+----------+--------
  vz       | TBD      | TBD
  gpt5     | TBD      | TBD
  ...
```

### 3.3 Indistinguishable pairs

TBD（在当前 n=120 下哪些 SUT 对 95% CI 重叠）

## 4. v1.0 leaderboard 展示策略

待真跑后决定：

| 情况 | leaderboard 展示 |
|---|---|
| 大多 SUT pairs distinguishable | 标准 ELO 排行 + ELO ± CI 显示 |
| > 50% pairs indistinguishable | "distinguishable bands"（top tier / mid tier / bottom tier）+ 不展示 ELO |
| 完全不可分 | 不发 v1.0 leaderboard，触发 v1.x scenario expansion 到 200+ |

## 5. v1.x roadmap

如果 n=24 不够，扩到 200+ 的路径：
- 每季度新加 24 scenario（held-out paraphrase rotation 已有 #35 cadence）
- 新 scenario family（除现有 6 family 外，加生命阶段 / 跨文化等）
- 与 EQ-Bench 3 / RP-Bench 联合贡献 scenario

## 复现性

```bash
python scripts/companion_bench/statistical_power_analysis.py \
    --n-scenarios 24,48,96,120,200 \
    --bootstrap-resamples 1000 --seed 42
```

## 变更日志

- 2026-05-13: v0 SHADOW placeholder。
