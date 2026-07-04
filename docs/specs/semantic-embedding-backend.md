# Semantic Embedding Backend Spec

> Status: v0.1
> Last updated: 2026-07-04
> 对应需求: R2（冻结基底 + 有界控制器）、R4（表达层不替代内部控制）、R8（快照 SSOT / 单一所有者）、R15（可回滚迁移）

## 要解决的问题

多处语义相似度决策（`dual_track` 的 WORLD/SELF track 分配、`evaluation` 的 task-pressure/support-presence prototype 打分、`application` 检索）此前建立在一个确定性字符-hash **stub**（`stub_semantic_embedding`）之上——它保证 determinism 与 SSOT 一致，但不携带真实语义。本 spec 定义一个**可注入的真实文本编码 backend 接缝**，让这些消费者在有真实 substrate 时用**已加载的 LM** 编码任意文本，无 substrate 时回退 stub。落地 [known-debts #91](../known-debts.md) 修法 1。

## 关键不变量

- **单一 SSOT 入口**：所有需要"可升级"语义嵌入的消费者调用 `volvence_zero.semantic_embedding.semantic_embedding(text, *, dim)`；不新增 fork（`test_semantic_embedding_ssot.py` AST 守门保留 #3 白名单）。
- **默认 fallback stub**：无 backend 注入时 `semantic_embedding == stub_semantic_embedding`（byte-identical）。这保证所有无 substrate / synthetic-substrate 的单测行为不变。
- **同空间比较**：任一 prototype-比较决策中，prototype 与被比较内容必须由**同一个 active backend** 编码（prototype 改为 lazy / 按需编码，绝不在 import 期冻结）。
- **backend 仅真实 runtime**：真实 backend 只对 `TransformersOpenWeightResidualRuntime`（真前向，含 builtin GPT-2 CPU）注入；`SyntheticOpenWeightResidualRuntime` 及其它保持 stub。
- **不吞错误**：seam 不吞 backend 异常（`no-swallow-errors`）；backend 自身负责空/短文本稳健性（内部 delegate stub）。
- **冻结基底**：backend 只读 LM 的 `capture()` 特征面，不改基底权重（R2）。
- **可回滚**：`BrainConfig.semantic_embedding_backend_wiring = DISABLED` 或 env `VZ_SEMANTIC_EMBEDDING_BACKEND=stub` 一键回退 stub。

## 接口契约

- **Seam（`vz-contracts`，`volvence_zero.semantic_embedding`）**：
  - `SemanticEmbeddingBackend` Protocol：`embed(text: str, *, dim: int) -> tuple[float, ...]`（返回 L2-归一化向量）。
  - `set_semantic_embedding_backend(backend | None)` / `get_semantic_embedding_backend()` / `reset_semantic_embedding_backend()`（进程级，wiring 时设置）。
  - `semantic_embedding(text, *, dim=8)`：backend 存在→`backend.embed`，否则→`stub_semantic_embedding`。
  - `semantic_cosine(a, b)`：与 backend 无关（两侧均 L2-归一化，点积）。
- **真实 backend（`vz-substrate`，`SubstrateTextEncoderBackend`）**：持有 `OpenWeightResidualRuntime`，`embed` = `runtime.capture(source_text=text)` → 取 `feature_surface` 稳定序展平 → L2 归一化 → 截/补到 `dim`（口径与 `memory.retrieval._substrate_embedding` 一致）；LRU 缓存 `(text, dim)`（`VZ_SEMANTIC_EMBED_CACHE`）；空文本 delegate stub。
- **注入点（`vz-runtime`，`Brain._install_semantic_embedding_backend`）**：session 解析 runtime 后，按 wiring/env + isinstance 决定 set/reset。

## 与其他能力域的关系

| 关系 | Spec | 说明 |
|------|------|------|
| 落地 | [known-debts #91](../known-debts.md) | 本 spec = #91 修法 1 |
| 盘点 | [learned-vs-heuristic-coverage.md](./learned-vs-heuristic-coverage.md) | dual_track/evaluation stub 行随本 spec 更新 |
| 复用口径 | [continuum-memory.md](./continuum-memory.md) | `_substrate_embedding` 投影口径来源 |
| 消费者 | [dual-track-learning](./00_INDEX.md) / [评估体系](./00_INDEX.md) | track 分配 / prototype 打分 |

## 当前落地范围（v0.1）

- **已迁到 seam**：`dual_track/core.py`（WORLD/SELF track）、`evaluation/semantic_readouts.py`（+ `backbone.py` 调用点）、`application/storage.py`（domain-knowledge 检索）。
- **暂未迁移（#91 follow-up）**：`application/scoring_helpers.py` + `application/runtime_helpers.py`——`scoring_helpers.semantic_embedding` 被 `runtime_helpers` 与 **stub-空间字面量 prototype** 配对使用，直接切 seam 会造成跨空间 cosine 错误；需与该处字面量 prototype 一并 lazy 化后迁移。
- **DLaaS 多 substrate 进程共存**：进程级全局 backend 仅适用于单 substrate；多实例部署应保持 `DISABLED`（stub），作为 #91 follow-up。

## 变更日志

- 2026-07-04：v0.1 初版。seam（vz-contracts）+ SubstrateTextEncoderBackend（vz-substrate）+ Brain 注入 + dual_track/evaluation/storage 迁移 + 契约测试。落地 known-debts #91 修法 1。
