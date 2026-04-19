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

## 相关文件

- `volvence_zero/substrate/residual_backend.py`
- `volvence_zero/substrate/adapter.py`
- `volvence_zero/agent/session.py`
- `tests/test_substrate_adapter.py`
- `tests/test_agent_session_runner.py`
