# 真实 Substrate 校准报告

> Status: draft
> Last updated: 2026-04-19
> Scope: hook-layer calibration for `distilgpt2` and `Qwen/Qwen2.5-0.5B-Instruct`

## 目标

为当前两条真实本地路径固定推荐的 hook 层配置，避免后续 benchmark 被劣质 layer 组合污染。

## 校准输入

- source text:
  `Help me carefully plan a difficult project while staying supportive and concrete.`

## 结果

### `distilgpt2`

- 候选组合：
  - `(0, 1)`
  - `(1, 2, 3)`
  - `(3, 4, 5)`
  - `(2, 3, 4)`
- 推荐层：
  - `(3, 4, 5)`

#### 关键观察

- 推荐组合的 `signal_quality` 最高：`0.5289`
- 中后层组合整体优于早层组合
- `semantic_separation` 与 `hook_layer_coverage` 的综合表现最好

### `Qwen/Qwen2.5-0.5B-Instruct`

- 候选组合：
  - `(0, 1, 2)`
  - `(8, 9, 10)`
  - `(11, 12, 13)`
  - `(20, 21, 22)`
- 推荐层：
  - `(20, 21, 22)`

#### 关键观察

- 推荐组合的 `signal_quality` 最高：`0.3964`
- Qwen 在晚层的语义分离度略优于中层
- 说明当前更强模型在更靠后的抽象层上更适合做 substrate capture

## 结论

当前推荐默认 hook 配置：

| 模型 | 推荐层 |
|------|--------|
| `distilgpt2` | `(3, 4, 5)` |
| `Qwen/Qwen2.5-0.5B-Instruct` | `(20, 21, 22)` |

## 影响

- 后续三路径 benchmark 将以这两组推荐层作为真实路径默认层
- 若未来引入更强模型，需要重新执行同样的 calibration 流程

## 相关代码入口

- `volvence_zero.substrate.run_hook_layer_calibration`
- `volvence_zero.substrate.build_transformers_runtime_with_fallback`
