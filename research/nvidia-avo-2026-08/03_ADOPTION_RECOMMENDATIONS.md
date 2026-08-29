# 对 Volvence 的借鉴与实施建议

## 1. 决策摘要

### 现在吸收

1. 在 task-local research sandbox 内，把 coding agent 升级为完整 variation operator。
2. 把 `lineage / knowledge bundle / evaluator / tool interface / resource budget` 作为可重放输入。
3. 把失败尝试和 supervisor intervention 变成 typed、create-only research evidence。
4. 对 agentic loop、memory、supervisor 做 matched ablation，不只展示最终 best candidate。
5. 同时计量 environment actions、tool calls、model tokens、wall-clock、compute 和每次有效 commit 成本。

### 保留现状

1. A0 named-human approval。
2. Praxist research retention、formal validator、`ModificationGate`、target deployment 四权分离。
3. sealed heldout 对 research agent 不可见。
4. content-addressed handoff 与相邻 `DISABLED → SHADOW → ACTIVE`。
5. runtime 学习只走 PE→credit；research score 不回灌。

### 当前不要做

1. 不修改 generic control plane 来追求七天自动运行；先在一个 task 内做 pilot。
2. 不让 supervisor 自行改模型、预算、evaluator、protected roots 或 approval。
3. 不把 current best 单谱系替换掉 Frontier/Pareto/QD。
4. 不在 relationship runtime 上直接复制 correctness/performance 单标量 commit gate。
5. 不因 AVO public ARC 满分宣称 Volvence 四轴、unseen generality 或持续学习已成立。

## 2. 最小收敛包：AVO-style `coding_memory_inheritance` pilot

当前仓库已经把 [`coding_memory_inheritance`](../praxist_tasks/coding_memory_inheritance/README.md)
指定为 Research Control Plane 的首个 pilot。它比新建一个 CUDA 任务更适合做最小验证：

- 只有一个可编辑 artifact：`policy.json`；
- public development evaluator 可执行、确定、CPU-only；
- task/evaluator/assets/audit rules 都在保护面；
- 已有 Frontier lanes、QD、negative-evidence ratio 与资源预算；
- 结果本来就只具有 development retention 语义。

### 2.1 唯一 owner 与写面

- Research task owner：现有 `coding_memory_inheritance` task project。
- 变异写面：候选目录中的一个 schema-bound `policy.json`。
- AVO-style operator：Praxist task-local agent；不进入 `volvence_forge` generic owner，也不进入任何
  `vz-*` wheel。
- Supervisor：初期只作为 task-local advisory role/receipt；不新增 runtime slot。
- 正式上架：继续由现有 Research Promotion Pipeline 处理，pilot 不实现 target apply。

这个边界符合仓库“一次只收敛一个 owner、一个 artifact、一个主要 consumer”的要求。

### 2.2 输入合同

每次 variation run 固定接收：

| 输入 | 内容 | 约束 |
|---|---|---|
| Task | objective、editable/protected roots、public evaluator、budget | content digest 固定 |
| Lineage | baseline、eligible parents、negative attempts、metrics | 只读、content-addressed |
| Knowledge | protocol intent、resource plan、research directions、policy schema | allowlist + digest |
| Evaluator | public `context_replay` development evaluator | agent 可调用、不可修改 |
| Tools | read/edit candidate、run evaluator、inspect structured result | 无 arbitrary protected write |
| Budget | max internal actions、eval count、tokens、wall-clock、parallelism | 到点 STOP，不扩权 |

第一轮不要改 `forge-research-task.v1` schema。上述信息已经大部分存在于 task project；先把 task-local
manifest/receipt 做成实验附件。只有 pilot 显示跨任务复用价值后，再考虑把 `knowledge_bundle`、
`variation_protocol`、`supervisor_policy` 和完整 cost budget 提升为 generic ResearchTask 字段。

### 2.3 一个 variation step

```text
load exact task + lineage + budget
  → inspect parent policies and structured results
  → state one falsifiable mechanism hypothesis
  → create one candidate policy
  → run schema / protocol-integrity checks
  → run preliminary or complete evaluator as budget allows
  → inspect structured failures
  → repair or abandon
  → emit candidate + attempt receipts + hypothesis outcome
```

与 AVO 不同，未成功尝试不能只留在 conversation history。每个 attempt 至少发布：

- parent/candidate content refs；
- hypothesis id 与 mechanism family；
- exact evaluator receipt；
- failure category / metric vector；
- abandoned / repaired / retained decision；
- token/tool/eval/wall-clock cost；
- protected-surface and heldout-access attestations。

不要求保存完整私有 reasoning；保存可验证行动、假设摘要和结果即可。

### 2.4 Supervisor 设计

Supervisor 只观察结构化 research events，禁止从自由文本关键词做控制。初期触发可用：

- 连续 `N` 个 evaluated attempts 没有新的 eligible Pareto point；
- 相同 failure category / candidate content family 重复超过预注册阈值；
- diversity cell occupancy 或 parent lineage 集中度越界；
- evaluator infrastructure failure rate 越界；
- 剩余预算不足以完成一次 close-grade eval。

输出是 `research-supervisor-advice.v0` 式实验 artifact，而不是 authority：

- observed trigger facts；
- recommended mechanism families / parents to inspect；
- forbidden repeats；
- remaining budget；
- `can_modify_task=false`、`can_modify_evaluator=false`、`can_authorize_candidate=false`、
  `can_change_wiring=false`。

Agent 可以解释并选择是否执行建议；下一轮记录是否采用与结果。这样可以测 supervisor 的真实边际效应。

## 3. 必须做的 matched ablation

AVO 当前最大证据缺口是没有隔离 operator、memory 和 supervisor。Volvence pilot 应预先冻结至少三臂；
预算不足时先做 A/B，再决定是否进入 C：

| Arm | Candidate generation | Lineage/memory | Supervisor |
|---|---|---|---|
| A 固定工作流 | 一次 plan→write→evaluate | 当前 parent + result | off |
| B Agentic variation | 自主 inspect/edit/eval/repair | 成功 lineage + typed negatives | off |
| C Agentic + supervisor | 同 B | 同 B | advisory on |

如果资源允许，再增加 B0（只有成功 lineage、无 typed negative attempts），单独量出负经验记忆的价值。

所有 arms 必须共享：

- exact model/reasoning setting；
- task revision、baseline、public evaluator 与 candidate schema；
- total token/tool/evaluator/wall-clock budget；
- seed/cohort schedule；
- protected roots 和 no-heldout policy；
- final loop-external evaluation protocol。

不能用“B 跑得更久”证明 agentic operator 更好。

## 4. 指标

### 4.1 Primary

`externally_validated_improvement_per_budget`：在固定研究预算下，最终由 loop-external validator 接受的
改进，而不是 development best score。

pilot 尚未接 formal validator 时，不得伪造该指标；第一阶段只报告 development proxy，并保持
`formal_validation_performed=false`。

### 4.2 Secondary

| 面板 | 指标 |
|---|---|
| Search quality | eligible Pareto points、best/worst-chain margin、coverage/retention |
| Efficiency | tokens、tool calls、eval calls、wall-clock、cost per eligible candidate |
| Exploration | unique mechanism families、diversity cells、parent-lineage HHI |
| Failure learning | repeated failure rate、同类错误再犯间隔、negative evidence reuse |
| Long-horizon | restart recovery、state loss、duplicate work after resume、stall duration |
| Integrity | protected-write attempts、heldout access、hash drift、schema violations |
| Attribution | A/B/C effect with uncertainty；不只给最终 best |

ARC 的 environment action metric可以作为一项，但不能代替内部成本。AVO public ARC 中 reasoning 与只读
inspection 免费，正说明系统评估需要双账本。

## 5. Kill conditions

任一条触发即停止 AVO-style 扩展，保留负结果：

1. 在 matched budget 下，B 对 A 没有稳定 development 增益，或只有更高成本下才更好。
2. C 没减少 stall/repeated failure，反而降低 diversity 或增加 evaluator 调用浪费。
3. agent 需要修改 task/evaluator/protected assets 才能取得提升。
4. 发生 sealed heldout 暴露、formal protocol 污染或 production credential/wiring 请求。
5. attempt receipts 无法从 artifacts 重建，conversation loss 后不能继续。
6. development improvement 在 loop-external validation 消失。
7. best-only commit 导致 Frontier/QD coverage 下降。
8. task-specific prompt/harness 修改大到无法复用同一 operator interface；跨域主张失败。

科学 FAIL 不是 infrastructure error，也不能通过放宽断言、删掉负证据或换 evaluator 就地重跑。

## 6. 进入 generic contract 的条件

只有 task-local pilot 同时满足以下条件，才值得改通用 ResearchTask / control artifacts：

- B 在相同预算下优于 A；
- supervisor 有可归因边际收益或明确被判无效；
- restart 后从 typed lineage 恢复，不依赖完整 conversation；
- 负尝试保存有用且隐私/成本可控；
- loop-external validator 保留 development 改进；
- 没有新增 authority 或 bypass promotion chain；
- 第二个不同 task 能复用同一 `P/K/f/tools/budget` 接口。

届时可以考虑一个独立收敛包，给 `forge-research-task.v2` 增加：

- `knowledge_bundle_refs`；
- `variation_operator` 及 exact model/runtime/tool profile；
- `attempt_evidence_policy`；
- `supervisor_policy`（advisory-only）；
- 全成本预算与停止条件。

不要在 v1 schema 上无证据地提前增加这些字段。

## 7. 关系与 runtime 主线的使用边界

AVO-style search 未来可以提出：

- bounded memory composition policy；
- scenario/evidence harness；
- owner-local rare-heavy artifact；
- evaluator implementation 的候选，但 evaluator 自身必须由另一 task/review path 验证。

它不能直接优化：

- `goal_value`、`relationship_state` 或 `boundary_consent` 真值；
- runtime PE predictor 或 credit source；
- `z_t` policy 的 token-space patch；
- residual steering free bias；
- production prompt/owner code 的任意修改。

在没有 hard verifier 的关系域，commit 条件必须是多面板 evidence + protected invariants + external validation，
不能复制 kernel 的“correctness + throughput 不劣”二元门。

## 8. 回滚与退出

本建议当前只形成研究文档，没有实现或运行时改动。

未来 task-local pilot 的退出方式应是：

- 停止该 task 的 A0-approved run；
- 保留 immutable attempts、findings、negative evidence 和 generation boundary；
- 不产生 handoff 或只产生 `evidence_only` handoff；
- 不修改 generic control plane；
- runtime 始终 `DISABLED`，所以没有 ACTIVE rollback。

若以后进入 generic schema，仍必须先双写旧/新 artifacts、对比判词，再切 consumer；失败时回到旧 Task
revision，不删除历史 receipts。
