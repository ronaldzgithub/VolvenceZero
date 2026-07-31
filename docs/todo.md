# Volvence Todo 状态指针

> Last updated: 2026-08-01
> 本文件不再维护第二份 backlog。开放项唯一台账是 [known-debts.md](./known-debts.md)，当前实现与证据状态见 [currentstatus.md](./currentstatus.md) 和 [current.md](./current.md)。

## 原 todo 的 12 个 gap 对账

| 原 gap | 2026-08-01 状态 | 后续入口 |
|---|---|---|
| 1. Affordance / Tool / Action | registry、snapshot、renderers、invoker 与 outcome settlement 已落地；host 默认 ACTIVE，learned scorer 仍 SHADOW/evidence-gated | [affordance spec](./specs/affordance.md) |
| 2. Apprenticeship | ingestion/apprenticeship owner、typed feedback request 与 PE overlay 已落地；protocol alignment/revision proposal 默认 SHADOW | [apprenticeship specs](./specs/apprenticeship-alignment.md) |
| 3. Runtime ingestion | `lifeform-ingestion` 与 canonical ingestion turn/session-post durable path 已落地 | [runtime-ingestion](./specs/runtime-ingestion.md) |
| 4. Active exploration / mid-session reflection | `lifeform-thinking` task/artifact/scheduler 与 temporal advisory SHADOW 链已落地；authoritative promotion 仍需证据 | [thinking-loop](./specs/thinking-loop.md) |
| 5. Speculative / lazy expression | `lifeform-expression`、typed speech plan/rationale tags 已落地；没有把 speculative generation 建成第二策略 owner | [expression-layer](./specs/expression-layer.md) |
| 6. DLaaS control plane | contracts/registry/launcher/api/ops/eval 六 wheel 已落地；剩余生产化与评估债逐项登记 | [known debts](./known-debts.md) §12–17、45–50 |
| 7. AAC lifecycle | typed advocacy→alignment→commitment→followup lifecycle 已落地 | [aac commitment](./specs/aac-commitment-lifecycle.md) |
| 8. Cognitive depth / participation | Regime participation readout 已在主链；`decision_workspace` 当前 SHADOW，无 authoritative consumer | [cognitive-regime](./specs/cognitive-regime.md) |
| 9. Interlocutor state | 12-axis owner、typed zones、snapshot consumers 与 rollout wiring 已落地 | [interlocutor-state](./specs/interlocutor-state.md) |
| 10. Runtime event inputs | canonical typed environment/dialogue/semantic/ingestion inputs已分 owner 落地；新增事件必须按 DATA_CONTRACT 注册，不再维护“9 类一次性清单” | [environment-interface](./specs/environment-interface.md) |
| 11. Scenario package hot lifecycle | vertical/registry/protocol package 装载面已存在；通用 install/uninstall 与产品运营面仍按具体债项推进 | [known debts](./known-debts.md) |
| 12. Tool selection quality evaluation | affordance outcome readout 与 F1–F6 框架已有；真实 tool-choice ground truth、纵向/成本证据仍未形成统一 production gate | [evaluation](./specs/evaluation.md) |

## 当前真正的优先级

1. 不重开 #92；任何新整体 thesis 使用新提案、独立预注册和新总 EXIT。
2. 先处理 [known-debts.md](./known-debts.md) 中仍开放且有明确 owner/退出条件的项。
3. learned component 只按 SHADOW evidence → 单组件 canary → ACTIVE 推进。
4. Relationship Memory Console 的 P3 UI、P5 continuity aggregator、P6 自动 apply
   与其他产品债分包实现，不在本文复制计划。

旧 todo 中已经落地的能力禁止以同名重新实现平行 owner；若实现与文档冲突，以代码为准，
同步修正对应 spec 和 `DATA_CONTRACT.md`。
