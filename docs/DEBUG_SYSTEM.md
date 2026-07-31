# Volvence 调试与可观测性

> Status: current implemented surfaces + explicit target gaps
> Last updated: 2026-08-01

## 1. 原则

调试首先回答“哪个 owner 在何时根据哪些 snapshot 发布了什么”，而不是先读最终文案。
任何 diagnostic 都不能成为第二 owner，也不能用 `getattr(..., default)`、silent fallback
或文本关键词掩盖契约错误。

## 2. 当前真实可观测面

| 层 | 已落地 surface | 可信边界 |
|---|---|---|
| L1 Runtime snapshots | `AgentTurnResult.active_snapshots/shadow_snapshots`、slot/version/owner/value、rationale tags | authoritative 与 SHADOW 必须分开；placeholder 不是正常 value |
| L2 Contract guards | ownership、dependency、schema、immutability、import-boundary tests | contract violation fail loudly；不做业务解释 |
| L3 Dialogue trace | typed `DialogueActionTrace`、prediction/action/outcome lineage、unresolved ids、external outcome evidence | session-local readout；不从回复文本重建 outcome |
| L3 Evolution trace | `lifeform-evolution.TraceCollector` 的 `trace.v1` NDJSON、snapshot IO、benchmark/family reports | 一行一 turn；需要额外隐私/PII 管理 |
| L3 Evidence artifacts | manifest、raw JSONL/NDJSON、ablation、verdict、freeze/rollback evidence | verdict 以 bundle owner 为准；聚合器不改判词 |
| L3 Persistence | Memory checkpoint、owner hydration snapshot、rare-heavy artifact fingerprint | rollback 必须 round-trip；scope 必须绑定 identity |

旧文档中的完整 Session Dashboard（Layer 4）与 Longitudinal Panel（Layer 5）仍是
目标形态，不是仓库内一个已部署的统一产品。当前有 CLI/report/artifact 组件，但没有
统一 web dashboard；不得把设计 wireframe 写成已实现功能。

## 3. 一次 turn 怎么查

1. 记录 `session_id / wave_id / turn_index` 与输入的 typed trigger。
2. 查看 active 与 shadow slot 集合，确认目标 owner 的 wiring。
3. 从 `substrate` capture、上一轮 prediction、当前 actual outcome 开始追 PE lineage。
4. 检查 `memory` retrieval、`temporal_abstraction` 的 `beta_t/z_t/closed_segments`。
5. 查看 semantic/regime/application snapshots，而不是从 response text 猜决策。
6. 查看 `response_assembly` 与 generation attestation，确认 latent/text/character carrier。
7. 若有延迟 outcome，沿 `dialogue_trace` 或 environment lineage 查到 consuming turn。
8. 最后才看 evaluation/credit/reflection readout。

## 4. 常见故障定位

### Slot 不见了

- 先查 `FinalRolloutConfig` 与 module class default；
- ACTIVE surface 没有但 shadow 有：这是 SHADOW，不是数据丢失；
- value 是 `RuntimePlaceholderValue`：区分 `disabled-module` 与
  `missing-upstream`；
- 缺依赖应由 DependencyGuard 或直接字典读取 fail loudly，不手工补 stub。

### 行为没变化

- `character_prefix_applied`、conditioning lineage、dynamic residual rationale tag
  分别证明不同物理通道；SHADOW 只证明加载/计算，不证明注入；
- learned report 更新不代表 consumer 读取；检查 gate 与 modulation strength；
- evaluation score 变化不是 PE/reward 变化；检查真正的 typed outcome。

### 跨 session 没延续

- 确认 typed identity、scope permissions、`memory_scope_root_dir` 与 persistence backend；
- 检查 owner 是否实现 hydration protocol；world/self temporal 通常由 joint-loop
  checkpoint owner 管，不应被第二 hydration writer 覆盖；
- hydration schema/fingerprint 不匹配必须报错，不能回到空状态假装成功。

### 证据跑分异常

- 先校验 prereg/source/substrate/control-basis/seed digest；
- 再核对 matched arms 是否只差一个机制；
- 不允许在看结果后换 seed、降阈值或重算“更好看”的 metric；
- source drift 后只能用冻结隔离快照复核，不能静默封包。

## 5. Dialogue trace

`vz-contracts.dialogue_trace` 定义 action、prediction、outcome、resolution 与 session
snapshot。External outcome 只经 `dialogue_external_outcome` slot 进入 kernel；HTTP/UI
handler 不直接写 PE、regime 或 memory。

Relationship Memory Console 的 correction record 会携带
`dialogue_outcome_evidence_id/kind`；exact retry 由 action ledger fingerprint 去重。调试
时要区分 `status=queued` 的 semantic event 与已持久化 Memory owner operation。

## 6. Evolution NDJSON 与 report

`TraceCollector(output_path=...)` 写 `trace.v1` line-delimited JSON，一行一 turn，供
dataset adapter、SSL demo、multi-round loop 与外部工具消费。它是离线 artifact，
不是 runtime slot；采集到的 text 可能含用户内容，生产使用前必须有 retention、scope、
deletion 与脱敏策略。

Family report、semantic proposal ablation、companion evidence、learned-shadow evidence
等各有独立 schema。不要把不同 schema 的字段拼成无 provenance 的“万能日志”。

## 7. Gate artifact 最小检查

一个可审计证据目录至少核对：

- `manifest` / `freeze_manifest` 与 required files；
- source/code tree/substrate/artifact SHA-256；
- raw outcomes、prediction errors、segments、action selection；
- ablation/matched control；
- machine `promotion_verdict`；
- rollback evidence 与 source unchanged；
- claim boundary。

缺其中任一项时只能降级 evidence tier，不能补写 guessed value。

## 8. Checkpoint 与 rollback

- Memory owner checkpoint 必须原子覆盖 entry、semantic index、attribute index；
- owner hydration 使用 owner-authored export/hydrate payload；
- rare-heavy artifact 用 content id、compatibility fingerprint 与 gate record；
- rollback failure 不应吞掉原异常，应同时暴露上下文；
- UI/route disable 只回滚产品入口，不会自动撤销已经明确写入的 owner state。

## 9. 当前 Layer 4/5 缺口

尚未形成统一产品的目标包括：

- session replay web UI、regime timeline、credit heatmap、reflection review；
- multi-user longitudinal curves、drift panel、artifact promotion timeline；
- Relationship Memory Console P5 七日 continuity panel；
- 跨进程统一 trace index 与 production rollback drill dashboard。

这些目标必须从现有 snapshot/artifact 读取，不能建立旁路 owner 或 evaluation→reward
回流。

## 10. 常用验证

```bash
pytest -q tests/contracts/test_import_boundaries.py
pytest -q tests/contracts/test_data_contract_wiring_sync.py
pytest -q tests/test_core_package_boundary.py
```

变更 owner/schema 时运行直接相关 contract tests；真实模型/GPU/外部 API、长轨迹与
多 seed 只在对应机制或 promotion gate 改变时运行。
