# 真实 Substrate 验收协议

> Status: draft
> Last updated: 2026-04-19
> Scope: local frozen LLM substrate validation

## 目标

把“真实 substrate 已接入”收束为一套可重复、可比较、可归档的标准验收流程。

## 三类验收

### 1. Strict-local 单轮验收

目的：

- 验证系统确实跑在本地冻结模型上
- 排除 builtin / synthetic fallback

必须满足：

- `substrate_runtime_origin == "hf-local"`
- `substrate_fallback_active is False`
- `substrate_capture_source == "real"`
- `substrate_residual_sequence_length > 0`
- `SubstrateSnapshot.surface_kind == RESIDUAL_STREAM`
- `acceptance_passed is True`

## 2. Strict-local 多轮稳定性验收

目的：

- 验证真实 substrate 可连续运行，而不是一次性成功

建议窗口：

- 5-turn smoke
- 20-turn benchmark
- 50-turn benchmark

最低通过条件：

- 多轮全部 `acceptance_passed`
- `residual_sequence` 连续非空
- 至少 1 次 `full-cycle`
- `evaluation.turn_scores` 每轮非空

## 3. `hf-local` vs `builtin` 对比验收

目的：

- 验证真实 substrate 是否已经具备可比较的学习价值

统一对比维度：

- acceptance rate
- residual sequence 长度
- turn score 数量
- family/regime 轨迹
- joint loop 触发频率
- policy objective
- 关键指标差值

## 代码入口

### Hook 层校准

- `volvence_zero.substrate.run_hook_layer_calibration`

输出：

- 每组 `layer_indices` 的 `signal_quality`
- `semantic_separation`
- `hook_layer_coverage`
- 推荐 hook 层组合

### 多路径 benchmark

- `volvence_zero.agent.run_substrate_path_benchmark`

输出：

- `SubstrateBenchmarkReport`
- 每 turn 的 substrate 元信息
- acceptance / sequence length / score count / full-cycle 统计

## 推荐执行顺序

1. `strict-local` 单轮
2. `strict-local` 5-turn
3. hook 层校准
4. `hf-local` vs `builtin`
5. 20-turn / 50-turn benchmark

## 当前基线模型

- `distilgpt2`：已通过 strict-local 单轮和 5-turn 稳定性验收
- `Qwen/Qwen2.5-0.5B-Instruct`：已通过 strict-local 单轮验收，并在 20/50-turn benchmark 中成为 `preferred`
- `Qwen/Qwen2.5-3B-Instruct`：本地 tokenizer 离线兼容仍待收口

## 结果归档建议

每次正式验收应记录：

- 模型名
- runtime mode
- 是否 fallback
- hook layers
- benchmark 输入集
- JSON 结果
- 通过 / 不通过结论

## 相关文件

- `volvence_zero/substrate/residual_backend.py`
- `volvence_zero/substrate/adapter.py`
- `volvence_zero/agent/session.py`
- `tests/test_substrate_adapter.py`
- `tests/test_agent_session_runner.py`
- `docs/implementation/05_real_llm_substrate_runtime.md`
