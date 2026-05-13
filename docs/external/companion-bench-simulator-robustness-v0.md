# Companion Bench: User Simulator Robustness Report v0

> Status: SHADOW placeholder
> Driver: [`scripts/companion_bench/simulator_robustness_sweep.py`](../../scripts/companion_bench/simulator_robustness_sweep.py)
> Driving debt: [`docs/known-debts.md`](../known-debts.md) #53
> Driving packet: [`docs/moving forward/companion-bench-public-launch-packet.md`](../moving%20forward/companion-bench-public-launch-packet.md) §2.3

## 1. 目的

回答 reviewer 第三问：**"user simulator 用什么 LLM？换 simulator 排名会不会大幅变化？"**

simulator 是 Companion Bench 跑分链路上**未审视的偏置点**——如果 simulator 用 GPT-5，跑出的 SUT 反应可能更适合 GPT-5 风格（"GPT 模拟用户认为 GPT 回应最贴心"），导致排名失真。

## 2. 方法论

### 2.1 多家族 simulator sweep

固定 reference SUT 池（5 SUT），让 N=4 个 LLM simulator 家族（GPT-5 / Claude Opus 4.7 / Qwen3 / DeepSeek V4）分别驱动同一组 24 公开 scenario，记录每个 SUT 在不同 simulator 下的 6 轴分数。

### 2.2 评估指标

| 指标 | 通过阈值 |
|---|---|
| Per-SUT × per-axis variance σ across 4 simulator families | < 7.0 |
| Per-SUT ranking stability across 4 simulators | 排名变化 ≤ 1 名 |
| Bias direction（同家族 simulator → 同家族 SUT 加分？） | < +4 分 |

## 3. 结果（待真跑回填）

### 3.1 Per-SUT × per-axis variance σ

```
  SUT      |  A1  |  A2  |  A3  |  A4  |  A5  |  A6
  ---------+------+------+------+------+------+------
  vz       | TBD  | TBD  | TBD  | TBD  | TBD  | TBD
  gpt5     | TBD  | TBD  | TBD  | TBD  | TBD  | TBD
  ...
```

### 3.2 Bias direction

TBD

## 4. Quarterly rotation policy

待真跑后落 spec §4.x：每季度公开榜单跑分时，simulator LLM 必须从公开 4+ 家族池**随机抽**（rotation log 公开），杜绝长期单一 simulator 偏置积累。

## 5. 复现性

```bash
python scripts/companion_bench/simulator_robustness_sweep.py \
    --simulator-families gpt5,claude47,qwen3,deepseek4 \
    --seed 42
```

## 变更日志

- 2026-05-13: v0 SHADOW placeholder。
