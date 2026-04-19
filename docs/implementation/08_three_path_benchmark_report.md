# 三路径 Benchmark 报告

> Status: draft
> Last updated: 2026-04-19
> Scope: 20-turn + 50-turn benchmark over `builtin-fallback`, `hf-local-distilgpt2`, `hf-local-qwen-0.5b`

## 路径

- `builtin-fallback`
- `hf-local-distilgpt2`
- `hf-local-qwen-0.5b`

## 场景

- `task`
- `support`
- `mixed`

每个场景 20 turn。

## 总体结论（20-turn）

### 1. 三条路径都稳定可运行

- 三条路径在三个场景下的 `acceptance_rate` 都为 `1.0`
- 三条路径都能持续产出非空 residual sequence、evaluation turn scores、family/regime 轨迹

### 2. 真实路径在 substrate 质量上显著优于 builtin

所有场景下：

- `learning:substrate_signal_quality`
  - `builtin-fallback` 约 `0.1646 ~ 0.1647`
  - `hf-local-distilgpt2` 约 `0.6726 ~ 0.6840`
  - `hf-local-qwen-0.5b` 约 `0.5954 ~ 0.6024`

这说明真实路径已经明显优于 builtin 的 substrate 信号质量。

### 3. `hf-local-qwen-0.5b` 在关系稳定性上最强

关键指标 `relationship:cross_track_stability`：

| 场景 | builtin | distilgpt2 | qwen-0.5b |
|------|---------|------------|-----------|
| task | `0.8444` | `0.7556` | `0.9041` |
| support | `0.8305` | `0.7355` | `0.9138` |
| mixed | `0.8387` | `0.7257` | `0.9146` |

Qwen 0.5B 在三类场景上都高于 builtin 和 distilgpt2。

### 4. `hf-local-distilgpt2` 在 task integration / decoder usefulness 上更强

例如 `task` 场景：

- `task:info_integration`
  - builtin: `0.7207`
  - distilgpt2: `0.7792`
  - qwen-0.5b: `0.7524`

- `learning:decoder_usefulness`
  - builtin: `0.1716`
  - distilgpt2: `0.8017`
  - qwen-0.5b: `0.2045`

说明 distilgpt2 仍然是一个很强的“任务导向 / decoder-friendly” substrate 基线。

### 5. adaptive stability 仍是两条真实路径的主要弱项

`learning:adaptive_stability`：

| 场景 | builtin | distilgpt2 | qwen-0.5b |
|------|---------|------------|-----------|
| task | `0.9599` | `0.4546` | `0.6977` |
| support | `0.9229` | `0.3997` | `0.5872` |
| mixed | `0.9353` | `0.4075` | `0.7235` |

这里 Qwen 明显优于 distilgpt2，但仍低于 builtin。

## 初步默认基底判断

基于 20-turn 结果的**初步**判断：

### `builtin-fallback`

- 仍然是最稳的开发/CI 路径
- 但 substrate 信号质量远弱于真实路径
- 不适合作为长期默认真实基底

### `hf-local-distilgpt2`

优点：

- decoder usefulness 强
- task integration 强
- residual sequence 稳定

缺点：

- relationship stability 明显偏弱
- adaptive stability 偏弱

### `hf-local-qwen-0.5b`

优点：

- relationship stability 最强
- adaptive stability 明显优于 distilgpt2
- substrate signal quality 明显高于 builtin

缺点：

- decoder usefulness 弱于 distilgpt2
- mixed / task 场景的 task integration 略低于 distilgpt2

## 当前倾向（20-turn）

如果只看 20-turn 结果，**`hf-local-qwen-0.5b` 更像当前更平衡的真实本地基底候选**。

## 50-turn mixed benchmark

### 总体稳定性

三条路径在 50-turn mixed 场景下都保持：

- `acceptance_rate = 1.0`
- `full_cycle_count = 50`
- 持续非空 residual sequence

### 50-turn 核心结果

| 路径 | mean_seq_len | mean_policy_objective | substrate_signal_quality | cross_track_stability | adaptive_stability | task:info_integration |
|------|--------------|-----------------------|--------------------------|----------------------|-------------------|-----------------------|
| `builtin-fallback` | `10.6` | `~0.0` | `0.1646` | `0.8720` | `0.8811` | `0.7200` |
| `hf-local-distilgpt2` | `14.5` | `0.6645` | `0.6867` | `0.8341` | `0.3675` | `0.7844` |
| `hf-local-qwen-0.5b` | `16.22` | `0.5432` | `0.5977` | `0.9325` | `0.4366` | `0.7570` |

### 50-turn 解读

#### `builtin-fallback`

优点：

- 最稳的 `adaptive_stability`
- 依旧是开发/CI 的最安全基线

缺点：

- `substrate_signal_quality` 最低
- `fallback_reliance = 1.0`
- 不是值得继续作为长期真实基底的路径

#### `hf-local-distilgpt2`

优点：

- `task:info_integration` 最强
- `decoder_usefulness` 最强
- `mean_policy_objective` 最高

缺点：

- `cross_track_stability` 低于 builtin 和 qwen
- `adaptive_stability` 三条路径中最弱

结论：

- 更像“任务驱动、decoder 友好”的真实 substrate\n- 适合继续做 task-oriented 实验，但不适合作为当前最平衡默认基底

#### `hf-local-qwen-0.5b`

优点：

- `mean_seq_len` 最长
- `cross_track_stability` 最强
- `substrate_signal_quality` 显著高于 builtin
- `adaptive_stability` 明显优于 distilgpt2
- family 相关指标整体更平衡

缺点：

- `decoder_usefulness` 低于 distilgpt2
- `task:info_integration` 虽然高于 builtin，但低于 distilgpt2

结论：

- 当前更像“最平衡的真实本地基底”\n- 适合作为继续推进真实 substrate 学习闭环的默认候选

## 最终 verdict

### `builtin-fallback`

- verdict: `fallback-only`

理由：

- 稳定，但 substrate 信号质量最低
- 继续保留为开发/CI 基线，而不是默认真实基底

### `hf-local-distilgpt2`

- verdict: `acceptable`

理由：

- 已通过 strict-local 单轮与多轮验收
- 在任务导向和 decoder usefulness 上很强
- 但 relationship / adaptive stability 不够平衡

### `hf-local-qwen-0.5b`

- verdict: `preferred`

理由：

- 已通过 strict-local 单轮验收
- 20-turn 与 50-turn 都显示其在真实路径上更平衡
- relationship stability 明显最好
- substrate signal quality 显著高于 builtin
- 是当前最适合继续做真实 substrate 学习闭环默认候选的模型

## 默认基底决策

当前建议：

1. **默认真实本地 substrate 候选**：`Qwen/Qwen2.5-0.5B-Instruct`
2. **稳定任务导向基线**：`distilgpt2`
3. **开发 / CI fallback 基线**：`builtin-fallback`

这不是要求立刻把所有默认代码路径都切到 Qwen，而是说明：

- 如果目标是继续推进真实 substrate 的长期学习价值验证，优先使用 `Qwen-0.5B`
- 如果目标是最小成本任务性验证，保留 `distilgpt2`

## 当前文件

- 原始 JSON：`three_path_20turn_benchmark.json`
- 原始 JSON：`three_path_50turn_mixed_benchmark.json`
- 校准报告：`docs/implementation/07_real_substrate_calibration_report.md`
