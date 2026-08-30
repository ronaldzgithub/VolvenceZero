# Research Portfolio Orchestration：研究组合登记、依赖与并发编排

> Status: v1 portfolio contract、bounded loop、A0-bound concurrency 与 exact pause/resume/cancel control landed
> Last updated: 2026-08-30
> Owner: `volvence_forge`（development-plane scheduling artifacts only）
> Upstream: [`demand-driven-research-loop.md`](./demand-driven-research-loop.md)
> Lifecycle delegate: [`research-control-plane.md`](./research-control-plane.md)
> Promotion boundary: [`research-promotion-pipeline.md`](./research-promotion-pipeline.md)

## 1. 决策摘要

Forge 用一个 content-addressed `forge-research-portfolio.v1` 登记一组长期研究目标，并显式冻结：

- 每个 study 对应的 exact `ResearchDemand`；
- 科学依赖 DAG、优先级与 concurrency lane；
- 全局和 lane 级 active-run 上限；
- runnable mapping 或 `NEEDS_TASK_DESIGN` 状态；
- 断点策略固定为 Praxist `completed_generation`；
- 下游依赖必须等待具名人类对 exact completed Request/evidence 作 `PROCEED`，不能把 run completion、
  Frontier 排名或 evaluator 分数直接解释为科学前提成立。

Portfolio 只决定哪些 Demand lineage 有资格进入既有 bounded Research Loop。它不复制 Demand discovery、
A0、doctor/resolve/start/status 或 Praxist registry；这些动作仍分别委托给既有
`research_discovery`、`research_loop` 和 `research_control` owner。

```text
Portfolio + exact Demands
  → DAG / priority / lane eligibility
  → existing Demand Research Loop
  → human TopicBinding
  → human A0
  → existing Praxist control lifecycle
  → RUN_COMPLETED
  → named-human StudyOutcome(PROCEED | REVISE | STOP)
  → downstream dependency eligibility
```

## 2. 为什么需要独立 Portfolio owner

`ResearchDemand` 只说明一个 Volvence owner 需要什么；`ResearchRequest` 只冻结一次 exact Praxist 启动；
Praxist 只拥有一个 run 内的 generations/Frontier/Gems。三者都不应成为跨课题 DAG、优先级和资源 lane 的
第二 owner。

因此 Portfolio 是离线组合级 owner，但它没有以下权限：

- 不批准 TopicBinding 或 A0；
- 不修改 task/evaluator/baseline；
- 不将 `RUN_COMPLETED` 自动写成 `PASS`；
- 不调用 formal validator、ModificationGate 或 target deployment adapter；
- 不改变 `DISABLED / SHADOW / ACTIVE`；
- 不把 research/evaluation score 写入 PE 或 credit。

本工件不是 runtime snapshot，不注册 `docs/DATA_CONTRACT.md` slot。

## 3. Artifact contracts

正式 schema：`forge/schemas/research_portfolio.schema.json`。

### 3.1 `forge-research-portfolio.v1`

Portfolio identity 由除 `portfolio_id / created_at` 外的 canonical JSON 计算。每个 study 必须包含：

- 唯一 `study_id`、title、objective、claim/owner/four-axis 声明；
- exact Demand id + locator + SHA-256；
- `depends_on` DAG、非负 priority、已注册 concurrency lane；
- `RUNNABLE_MAPPING` 时同时冻结 mapping/task id，并与 task registry、Demand owner/axes 复核；
- `NEEDS_TASK_DESIGN` 时 mapping/task 必须为空，不能假装已有可运行 harness；
- `required_completion_decision=PROCEED` 和全部非授权 authority bits。

调度策略 v1 固定 `dependency_then_priority`；priority 数字越小越先。未知/out-of-band active run 的策略固定
`BLOCK`。同一 runnable mapping/task 不得被两个 study 重复拥有。

### 3.2 `forge-research-study-outcome.v1`

Outcome 精确绑定 Portfolio、study、`RUN_COMPLETED` Request 与至少一个 evidence artifact，decision 只能是：

- `PROCEED`：只满足下游 dependency scheduling；
- `REVISE`：当前 study 需要新 Task/Request 或新 evidence；
- `STOP`：停止该依赖分支。

Outcome 要求 named reviewer 和 reason，create-only；它明确不授权 formal validation、candidate import、
SHADOW、ACTIVE 或 wiring。一个 study 当前只允许一个 outcome；要重跑必须产生新的 Portfolio version 或未来
显式 attempt contract，不能覆盖旧决定。

## 4. 状态投影与 bounded loop

`research-portfolio-status` 从 immutable artifacts 投影以下窄状态：

- `NEEDS_TASK_DESIGN`
- `WAITING_FOR_DEPENDENCIES`
- `REGISTERED`
- Research Control 的 A0/start/run states
- `COMPLETED_ACCEPTED`
- `REVISION_REQUIRED`
- `STOPPED_BY_OUTCOME`

`research-portfolio-loop --once` 只把满足以下条件的 Demand ids 传给既有
`run_demand_research_loop_once(...)`：

1. study 是 `RUNNABLE_MAPPING`；
2. 所有 predecessor 已有 exact `PROCEED` outcome，或该 study 已处于需继续 targeted reconcile 的运行态；
3. 本 pass 的 discovery/request/reconcile 数量仍在显式上限内。

Research Loop 新增内部 exact Demand allowlist seam；默认 CLI 行为不变。allowlist 中出现未在 validated Demand
root 的 identity 必须 fail closed。Portfolio 不扫描或提交其他 Demand，也不允许下游课题在前置 scientific
decision 前被 A0 自动启动。

## 5. CLI

```text
forge research-portfolio-seal <portfolio-draft.json> [--output <path>] [--json]
forge research-portfolio-validate <portfolio.json> [--json]
forge research-portfolio-status <portfolio.json> [--json]
forge research-portfolio-loop <portfolio.json> --once \
  --backend codex_sdk --model gpt-5.6-luna [bounded limits...] [--json]
forge research-portfolio-review <portfolio.json> \
  --study-id <id> --request <request.json> --evidence <artifact> \
  --reviewed-by <human> --reason <reason> \
  --decision proceed|revise|stop [--json]

forge research-approve <request.json> \
  --portfolio <portfolio.json> --study-id <study> \
  --approved-by <human> --reason <reason>
```

`research-portfolio-seal` 接受不含 `portfolio_id` 的 human-authored draft，计算 canonical identity，完整复核
Demand refs、DAG 与 registry mapping，并 create-only 写入 `research/portfolios/<digest>.json`。同 identity 重放
幂等复用；不同内容不能覆盖既有工件。validate/status/loop 只接受已经具备 exact identity 的 sealed Portfolio。

外部 scheduler 只周期调用 `--once`。Portfolio、Demand、Binding、Approval、Event、Outcome 和 Praxist registry
共同构成恢复依据，不建立常驻 wrapper、latest mutable JSON 或根目录 queue script。

## 6. 并发与恢复边界

Portfolio 的 global/lane 并发意图只有在 named human 使用 `research-approve --portfolio --study-id` 时才进入
exact A0 `execution_policy`。Research Control 会重验 Request 与 Portfolio study 的 exact Demand、mapping、
ResearchTask、task project、Praxist executable、launch profile 和 run root，然后按以下规则调和：

- 同一 exact Portfolio 且已有合法 `START_CONFIRMED` lineage 的 live run 才是“已知 run”；
- 已知 run 分别计入 `max_active_runs_global` 与 study lane 的 `max_active_runs`；
- 达到任一上限保持 `WAITING_FOR_CAPACITY`；
- 未知、out-of-band、legacy A0 或其他 Portfolio 的 active run 按 `BLOCK` 处理；
- 未携带 Portfolio policy 的 legacy A0 继续 host-wide 单 active-run。

resolve 前后各检查一次配额。当前 Praxist 没有跨 client reservation，因此最后一次 check 与 start 之间仍有
极窄竞争窗；发现超额或未知 run 时 fail closed，不抢占、不自动 stop。

Portfolio 已冻结的 `resume_policy=completed_generation` 由 Research Control 的 exact
`forge-research-control-directive.v1` 与 append-only event chain 执行。Portfolio 不直接调用
`praxist stop/resume`，不使用 `--force`，也不裁剪 run artifacts；中断恢复只能从 Praxist 认可的 committed
generation boundary 继续。

## 7. 四能力轴与诚实边界

- **Appendable**：Portfolio/Outcome 与既有 event chain 可恢复；这不是 CMS memory。
- **Readable**：只读 exact Demand/Request/status/outcome，不从自然语言日志推断状态。
- **Learnable**：科学结果只经人审 Outcome 控制调度；evaluation 不是 PE/credit 来源。
- **Steerable**：本层只调度离线研究，不触碰 runtime steering 或 wiring。

所以当前可以声称“研究组合登记、依赖调度、A0-bound 受控并发和审计型断点控制机制成立”；在真实
interruption drill 完成前不声称恢复证据充分，run-count 配额也不等于硬件资源隔离，更不声称任何四能力
产品效果成立。

## 8. 回滚与退出

停止外部 scheduler 即停止自动 pass；已有 detached run 不受影响。删除 Portfolio CLI/module/schema/spec 可退出
组合层，原 Demand/Binding/A0/Research Control 仍可单任务使用。Portfolio 从未改变 runtime wiring，因此没有
自己的 ACTIVE rollback；生产回滚继续走 target owner 的相邻 authorization receipt。

## 9. 当前登记的 4ables program

当前 create-only Portfolio 为
`research/portfolios/baaf616c923bc77b3eb38a0fb68ce7a3d8b48bb3c6f9129cd592d67fcbde1f6b.json`，identity 是
`research-portfolio:baaf616c923bc77b3eb38a0fb68ce7a3d8b48bb3c6f9129cd592d67fcbde1f6b`。它登记五项 study：

1. `readout_cross_view_causal_validity`：P0，已有 exact Demand、ResearchTask、registry mapping 和可运行
   Praxist task；
2. `substrate_control_authority`：等待 P0 `PROCEED`，再设计 task；
3. `relationship_memory_write_eligibility`：等待 P0 `PROCEED`，再在独立 memory lane 设计 task；
4. `per_instance_layer_dose_headroom`：等待 substrate authority `PROCEED`；
5. `steering_side_effect_matrix`：等待 per-instance headroom `PROCEED`。

P0 的一次 Codex-native、read-only bounded discovery 已生成三个 `UNBOUND` TopicProposal，分别覆盖 factorized
cross-view validity、matched-random causal target-logit effect、以及 weakest-view/per-instance replication。
Discovery 本身没有选择 TopicProposal、创建 Binding/Request 或启动 Praxist；必须由 named human 精确选择一个
Proposal 绑定 `readout_cross_view_causal_validity_v1`，之后再单独审批生成的 A0 Request。
