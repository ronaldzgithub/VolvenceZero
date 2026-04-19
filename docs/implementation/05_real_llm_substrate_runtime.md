# 真实冻结 LLM substrate 运行说明

> Status: draft
> Last updated: 2026-04-09
> Scope: local open-weight substrate runtime

## 目标

本文档固定真实冻结 LLM substrate 的运行方式，目标是：

- 优先使用本地开源权重
- 保持冻结 substrate，不做在线微调
- 在 `AgentSessionRunner` / `run_final_wiring_turn` 主链路上稳定运行
- 明确 fail-closed 与 fallback 语义

## 运行模式

### `strict-local`

- 只能使用本地真实模型
- 等价于：
  - `runtime_mode="strict-local"`
  - `local_files_only=True`
  - `fallback_mode=DENY`
- 本地模型缺失时直接报错

适用于：

- 验证“是否真的接入本地冻结 LLM substrate”
- 发布前的 fail-closed 验收

### `prefer-local`

- 优先加载本地真实模型
- 本地模型不存在时允许回退到 builtin runtime
- 等价于：
  - `runtime_mode="prefer-local"`
  - `local_files_only=True`
  - `fallback_mode=ALLOW_BUILTIN`

适用于：

- 日常开发
- 本地模型偶尔不可用但仍希望主链路可运行

### `builtin-only`

- 直接使用 builtin fallback runtime
- 不尝试加载真实 HF 本地模型

适用于：

- CI
- 快速 smoke test
- 不需要验证真实 substrate 时

## 推荐的第一目标模型

建议优先使用小型、transformers 兼容、hook 路径稳定的模型：

- `distilgpt2`

推荐原因：

- 本地加载成本低
- 中间层 hook 结构简单
- 足够完成“一轮真实模型主链跑通”的验收

## 当前模型兼容性矩阵

| 模型 | strict-local | prefer-local | 当前状态 | 备注 |
|------|--------------|--------------|----------|------|
| `distilgpt2` | 可用 | 可用 | 已通过 1-turn 与 5-turn 验收 | 当前推荐默认模型 |
| `Qwen/Qwen2.5-3B-Instruct` | 未完成 | 未完成 | tokenizer 本地离线兼容仍待收口 | 已加入 tokenizer `use_fast=False` 本地回退尝试 |

### Qwen 当前缺口

- 主要问题不在主链路，而在 tokenizer 的本地离线兼容路径
- 当前 runtime 已加入“本地 fast tokenizer 失败时回退到 slow tokenizer”的策略
- 下一步需要在你的机器上确认 Qwen 本地权重目录、tokenizer 文件和 strict-local 行为是否一致
- 当前探针结果仍显示 `ConnectTimeout`，说明 **本地离线依赖尚未完全闭合**

### 兼容性探针

可以使用：

- `volvence_zero.substrate.probe_local_model_compatibility`

它会检查：

- tokenizer 是否本地可用
- model weights 是否本地可用
- strict-local runtime 是否真的能完成一次 capture

如果 `strict_local_runtime_available` 为 `False`，则说明还不能把该模型纳入正式 strict-local 验收矩阵。

## 推荐运行方式

### 最严格验收

```python
runner = AgentSessionRunner(
    substrate_model_id="distilgpt2",
    substrate_runtime_mode="strict-local",
)
```

预期行为：

- 本地已下载模型：成功运行
- 本地未下载模型：立即失败，不允许 builtin fallback

### 日常开发

```python
runner = AgentSessionRunner(
    substrate_model_id="distilgpt2",
    substrate_runtime_mode="prefer-local",
)
```

预期行为：

- 本地已下载模型：走真实 HF 本地模型
- 本地未下载模型：回退到 builtin runtime，并显式标记 fallback

### 显式本地目录路径

如果某个模型（例如 Qwen）在 repo id + cache 形式下本地离线不稳定，可以直接指定本地目录：

```python
runner = AgentSessionRunner(
    substrate_model_id="Qwen/Qwen2.5-3B-Instruct",
    substrate_model_source="/absolute/path/to/local/model",
    substrate_runtime_mode="strict-local",
)
```

语义上：

- `substrate_model_id` 用于逻辑标识和结果展示
- `substrate_model_source` 用于真正的 `from_pretrained(...)` 加载源

这能避免 “repo id 解析 + cache 猜测” 带来的不确定性。

## 如何判断当前是否真的跑在真实模型上

看 `AgentTurnResult` 和 `SubstrateSnapshot`：

### `AgentTurnResult`

- `substrate_model_id`
- `substrate_runtime_origin`
- `substrate_fallback_active`
- `substrate_capture_source`
- `substrate_residual_sequence_length`

期望真实本地模型路径时：

- `substrate_runtime_origin == "hf-local"`
- `substrate_fallback_active is False`
- `substrate_capture_source == "real"`
- `substrate_residual_sequence_length > 0`

### `SubstrateSnapshot.description`

当前会包含：

- `runtime_origin=...`
- `capture_source=...`
- `fallback_active=...`
- `residual_sequence_len=...`

示例：

```text
Transformers open-weight capture model=distilgpt2 ... origin=hf-local ...
runtime_origin=hf-local capture_source=real fallback_active=0 residual_sequence_len=12.
```

## 最低验收条件

一次 `run_turn()` 或 `run_final_wiring_turn()` 结束后，至少满足：

- `SubstrateSnapshot.surface_kind == RESIDUAL_STREAM`
- `residual_sequence` 非空
- `AgentTurnResult.substrate_residual_sequence_length > 0`
- `AgentTurnResult.substrate_runtime_origin` 明确
- `evaluation` / `temporal_abstraction` / `reflection` / `credit` 都正常产出

## 回退策略

- 开发时优先使用 `prefer-local`
- 验收时使用 `strict-local`
- CI 使用 `builtin-only`

如果真实模型 hooks 不稳定：

1. 先保持 `prefer-local`
2. 保留 builtin fallback 可运行性
3. 不回退 substrate contract，只回退真实 runtime 模式

## 正式验收协议

### 验收 1：strict-local 单轮

目标：证明系统不是 fallback，而是真正跑在本地冻结模型上。

通过标准：

- `substrate_runtime_origin == "hf-local"`
- `substrate_fallback_active is False`
- `substrate_capture_source == "real"`
- `substrate_residual_sequence_length > 0`
- `acceptance_passed is True`

### 验收 2：strict-local 多轮稳定性

目标：证明真实 substrate 不是一次性可用，而是多轮稳定。

通过标准：

- 连续 5 turn 都满足单轮标准
- `mean_seq_len > 0`
- `all_acceptance == True`
- 至少 1 次 `full-cycle`

### 验收 3：`hf-local` vs `builtin` 对比

目标：证明真实 substrate 至少具备稳定、可比较、可分析的学习信号。

通过标准：

- 两条路径都能跑通 benchmark
- 输出 JSON 对比报告
- 至少包含：
  - acceptance rate
  - residual sequence 长度
  - turn score 数量
  - family / regime 轨迹
  - joint loop 行为

### 推荐验收命令

1. `strict-local` 单轮：

```python
runner = AgentSessionRunner(
    substrate_model_id="distilgpt2",
    substrate_runtime_mode="strict-local",
    substrate_device="cpu",
)
```

2. `strict-local` 多轮 benchmark：

使用代码内的 `run_substrate_path_benchmark()` 对固定输入序列运行。

3. hook 层校准：

使用 `run_hook_layer_calibration()` 对不同 `layer_indices` 组合生成校准报告。

## 相关文件

- `volvence_zero/substrate/residual_backend.py`
- `volvence_zero/substrate/adapter.py`
- `volvence_zero/agent/session.py`
- `tests/test_substrate_adapter.py`
- `tests/test_agent_session_runner.py`
