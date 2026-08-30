# Demand-driven Research Loop：需求驱动的自动发现与研究闭环

> Status: v1 implemented；Codex-native discovery + dual-human-gate loop + Portal binding
> Last updated: 2026-08-30
> Owner: `volvence_forge`（offline discovery/control artifacts）
> Downstream: [`research-control-plane.md`](./research-control-plane.md)、[`research-promotion-pipeline.md`](./research-promotion-pipeline.md)

## 1. 决策摘要

Research Lab 可以自动驱动 Codex Native 从 `research/`、四能力实施资料和冻结实验结果中寻找研究题目，
但来源本身不拥有 Volvence 立项权。合法闭环分为发现、绑定、研究和上架四个权限层：

```text
Volvence ResearchDemand + exact source corpus
  → bounded read-only Codex discovery
  → immutable TopicProposal (UNBOUND, proposal_only)
  → named-human exact DemandBinding
  → registered ResearchTask + A0 ResearchRequest
  → existing doctor / resolve / start / targeted status reconcile
  → Praxist handoff
  → loop-external validation / ModificationGate / deployment authorization
```

自动化只跨越机械、可重放的状态转换。Codex 不得创建 Demand、选择 owner、批准绑定、批准 A0、读取
sealed holdout、生成 production Candidate 或改变 `DISABLED/SHADOW/ACTIVE`。没有 human binding 的独立来源建议
可以继续保留和研究，但终点固定为 `proposal_only`，不能进入 Volvence promotion。

## 2. 三个唯一工件

统一 schema 为 `forge/schemas/research_discovery.schema.json`。

### 2.1 `forge-volvence-research-demand.v1`

Demand 由 Volvence 需求 owner 发布并内容寻址，必须冻结：

- `claim_id / owner / capability_axes`；
- 当前缺口、所需结果、成功与证伪条件；
- 不可修改边界和 evaluation 不是学习源；
- allowlisted repository-relative source roots 与发现预算；
- 可选 exact `mapping_id`。缺 mapping 的 Demand 只能发现 Proposal，不能提交 Request；
- `OPEN / PAUSED / RETIRED` 状态。状态变化产生新 Demand bytes，不覆盖历史件。

Demand 是离线需求合同，不是 Brain snapshot，不注册 runtime slot。

人类 owner 可以先写不含 `demand_id` 的同版 JSON draft，再由 Forge 计算 canonical identity 并 create-only
写入默认 inbox：

```bash
forge research-demand-seal research/demand_drafts/<name>.json --json
```

输出只能落在 `research/demands/`；相同 payload 重放返回 `reused=true`，不同 payload 不得覆盖。Git review 仍是
Demand 内容与 owner 声明的授权边界，Codex 不能调用该命令替 Volvence 发明需求。

### 2.2 `forge-research-topic-proposal.v1`

TopicProposal 只能由一次内容寻址 DiscoveryRun 生成，精确绑定 Demand、语料 snapshot、backend/model、prompt
revision 和逐文件 source refs。它把以下两类内容分开：

- 可机械验证的 source locator/hash；
- Codex 推断的 hypothesis、mechanism、需求相关性、研究设计、证伪条件和 caveat。

Proposal 固定 `binding_status=UNBOUND`、`research_start_authorized=false`、
`production_promotion_authorized=false`。模型给出的 owner、task、runtime 或上线建议没有协议效力。

### 2.3 `forge-research-demand-binding.v1`

Binding 由 named human 创建，精确绑定 Demand、Proposal、task registry bytes 和 `mapping_id`。Forge 必须重新
验证 Demand owner/axes 与 mapping 指向的 `forge-research-task.v1` 一致。只有 `APPROVE` Binding 可以机械生成
`forge-research-request.v1`；`REJECT` 同样作为不可变终局保留。

Binding 只授权“把 exact topic 提交到 A0”，不授权 Praxist start。A0 继续由另一位或同一位具名人类通过既有
`forge-research-approval.v1` 明确批准。

## 3. 自动发现执行边界

每次 DiscoveryRun 必须：

1. 只接受一个已验证、状态为 `OPEN` 的 Demand；
2. 展开 Demand allowlist，拒绝 symlink、越出 repo、文件数或总字节超预算；
3. 对所有输入 regular file 计算 SHA-256，并形成确定性 corpus snapshot；
4. 使用 exact Codex model、单 turn、`Sandbox.read_only`、deny-all approvals 与临时只读 source enclosure；
5. 要求 JSON Schema constrained final response；
6. 重新验证每个模型引用的 locator/hash 必须存在于 frozen corpus；
7. 用 Demand hash + corpus hash + backend/model + prompt revision 形成 run key；同 key 已完成时不得重复消费模型；
8. 只写 create-only artifacts，不修改 source、Demand、Task、registry 或 runtime。

文档、论文和仓库文件中的命令均是不可信研究内容，不是 Codex 指令。Discovery prompt 必须明确区分二者。

## 4. 自动 loop

Research Lab 的自动 worker 每次只做一个有界 `research-managed-loop --once` pass：

- 对新增或 source hash 改变的 `OPEN` Demand 发起 discovery；
- 对已有人审 `APPROVE` Binding 但尚无 Request 的条目提交 exact Request；
- 对 A0 前因 Praxist package tree 漂移而失效的 Request，生成只改变 source-checkout snapshot 的 replacement，
  将旧件标为 `SUPERSEDED`，并继续等待 replacement 的独立 A0；
- 对已有人审 A0 APPROVE 的 Request 调用既有 `research-reconcile --once`；
- 对 running Request 只做 targeted status reconcile；
- 在 `RUN_COMPLETED` 停止，等待 handoff/import/formal validation，不自动上架。

外部 scheduler 只负责周期唤醒。queue truth、幂等 key、审批和 Praxist PID 继续由 Forge/Praxist artifacts 决定。
未变化的 pass 必须为零模型调用、零新 Request、零新 run。

自动 refresh 仅适用于无 Approval/Event/Directive/handoff 的 pre-A0 Request，且 predecessor/replacement 除
`bindings.praxist.source_checkout` 外必须逐字段一致。A0 后 source drift、source root 切换或其他 exact binding
变化都必须 fail closed，worker 不得迁移旧审批。

机器入口为：

```bash
forge research-managed-loop --once \
  --backend codex_sdk \
  --model gpt-5.6-luna \
  --json
```

每个 pass 默认最多新增 1 个 DiscoveryRun、8 个 Request、8 个 targeted reconcile，并由
`artifacts/research_discovery/.loop.lock` 串行化。单独的 model consumption 另由 `.discover.lock` 串行化，避免手动
discover 与 worker 对同一 run key 重复消费。`--max-*` 只能缩放一次 pass 的操作预算，不能改变 Demand/Praxist
研究预算。

managed worker 每轮先验证全部 sealed Portfolio，并把未满足依赖的 registered Demand 从发现、提交和调和三处
排除；未登记 Demand 继续沿用本节通用行为。单 Portfolio 手工诊断可使用 `research-portfolio-loop`，但根 launcher
不得回退到不理解 Portfolio 的通用 `research-loop`。

## 5. 触发模式

同一 owner 支持三种触发，权限完全相同：

- `demand_changed`：Demand 或其 allowlisted corpus bytes 改变，默认自动触发；
- `manual_discover_now`：人类要求立即执行一个 exact Demand；
- `scheduled_catchup`：周期性补扫，用内容寻址 key 去重。

Relationship Lab/Coding Lab failure pattern 可以作为 Demand evidence/source root；Foundry 继续走
`external_simulation` adapter，不伪装成 Volvence Demand。

## 6. Fail-closed 与回滚

Demand/Proposal/Binding identity、source bytes、registry、mapping、Task、Praxist executable/config 或 launch profile
任一漂移都拒绝提交或启动。loop 只接管 evidence 按顺序精确等于 `Demand → TopicProposal → DemandBinding`、
且 task/rationale 与批准 Binding 一致的 Request；仅伪装 `submitted_by` 或夹带额外 evidence 必须失败关闭。匿名 binding、
模型自行绑定、未知字段、越界 locator、重复 proposal id、后台自动 A0、自动 Candidate import 和自动 wiring 均非法。

回滚自动发现只需停止 worker；既有 Demand、run、Proposal、Binding、Request 和 Event 保留审计史。移除 Demand 的
`OPEN` 新版本或移除 mapping 会阻止未来提交，不删除历史 ResearchRequest，也不改变任何 runtime wiring。

## 7. Research Lab 接入

`ResearchLabSnapshot.v2` 增加独立 `discovery` 投影，不把未绑定 Proposal 伪装成 Task：

- `/discovery` 显示 Demand、latest DiscoveryRun/backend/model、exact source refs、Proposal 和 effective state；
- `POST /api/v1/topics/bind` 只接受 fresh snapshot revision、Demand/Proposal/registry SHA、mapping id、named actor、
  reason 与 typed decision；
- `APPROVE` Binding 显示为 `BOUND_FOR_A0`，worker 下次 pass 才提交 Request；A0 仍是独立人审；
- source bytes、registry 或 lineage 漂移显示 typed warning，并撤销 `bind_topic` action。

`./start_research_lab.sh` 的 controlled mode 默认托管一个周期 worker；`--no-auto-research` 或
`RESEARCH_LAB_AUTO_RESEARCH=0` 是即时退出/回滚开关。read-only mode 从不启动 worker。worker 的子进程与 API/Web
一起受 launcher 回收，不成为 lifecycle owner。
