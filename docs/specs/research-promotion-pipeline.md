# 统一研究候选上架流水线 Spec

> Status: v1 design freeze + Forge 离线桥接收敛包已落地；未创建具体 Praxist task、未启动 run、未改变 production wiring
> Last updated: 2026-08-29
> 对应需求: R8, R10, R12, R15；服务 Appendable / Readable / Learnable / Steerable 四轴的证据治理，但不单独证明任一能力轴

## 1. 要解决的问题

Volvence 已经分别拥有研究脚本、claim-to-evidence、`ModificationGate`、
`WiringLevel.DISABLED/SHADOW/ACTIVE` 和组件专用 canary，但缺少一条所有研究任务都能复用的、
机器可检的候选交接主线。结果是：

- 研究系统的 Frontier、实验 PASS、正式 held-out 与生产激活容易被口头混为一种“晋升”；
- 每条研究线重复发明 candidate、lineage、gate binding 和 rollback receipt；
- 自治研究代理可能同时看到开发 evaluator、正式判据和部署写面，形成选择污染或权限越界；
- 某个实验结果即使有效，也缺少从 exact artifact 到 exact runtime target 的统一审计链。

本 spec 建立统一的 **Research → Validation → Admission → Deployment** 流水线。Praxist 可以接管
单个任务的开发研究循环，但不能接管 Volvence 的证据解释权、`ModificationGate` 或生产接线权。

## 2. 不替代 SHADOW / ACTIVE

本流水线是 `SHADOW/ACTIVE` 的上游包络，不是替代品：

```text
frozen baseline + ResearchTask
                │
                ▼
      Praxist development sandbox
  variants / results / findings / frontier
                │ committed handoff
                ▼
       content-addressed Candidate
                │
                ▼
  loop-external formal validation + heldout
                │
                ▼
       cognition ModificationGate.OFFLINE
                │
                ▼
     DISABLED → SHADOW → bounded canary → ACTIVE
```

三种“晋升”必须使用不同语义：

| 层 | 决策 | 权威 owner | 明确不是 |
|---|---|---|---|
| Research retention | 是否进入 Praxist frontier/incubator、是否可作后代 parent | Praxist generic control plane + task-owned lane policy | production release |
| Admission | exact candidate 是否具备外部准入资格 | `vz-cognition` `ModificationGate` | runtime wiring |
| Deployment | exact admitted candidate 是否可影响用户可见行为 | target owner + deployment/canary control plane | research ranking |

因此 `frontier != SHADOW`、`ModificationGate.ALLOW != ACTIVE`。SHADOW 仍负责真实运行时分布、
snapshot lineage、后端兼容、延迟、staleness、并发与 consumer 接线的无行为影响验证。

## 3. 唯一 owner 与软件边界

| 对象 | 唯一 owner | 权限 |
|---|---|---|
| 开发环 task/run/peer/generation/frontier/Gems | Praxist | 研究调度与 retention；不能授权 Volvence 上线 |
| 任务目标、baseline、开发 evaluator、正式 protocol、required checks | Volvence task owner | 冻结领域真相；Praxist 只读 |
| 规范化 Task/Handoff/Candidate/Receipt | `volvence_forge` | 离线校验、内容寻址、授权链；不写 runtime |
| 正式验证 verdict | task 声明的 loop-external validator | 读取 sealed heldout；不得由 Praxist peer 修改 |
| 自修改准入 | `vz-cognition` `ModificationGate` | 唯一 ALLOW/BLOCK owner |
| 组件 apply / WiringLevel / canary | exact target owner 与既有 deployment adapter | 唯一运行时写面 |

`volvence_forge` 继续位于 `packages/` 同级，不 import `volvence_zero.*` 或 `lifeform_*`。本流水线
不新增 Brain `RuntimeModule`、snapshot 或 `docs/DATA_CONTRACT.md` slot；跨边界只交换离线、不可变、
内容寻址的 JSON artifact。业务 wheel 禁止 import `volvence_forge`。

## 4. 两条正交状态轴

研究成熟度与部署接线不得压成一个枚举。

### 4.1 Research maturity

| 状态 | 含义 |
|---|---|
| `candidate_sealed` | 已从一个 committed research boundary 导入并冻结 exact bytes；仅是候选 |
| `externally_validated` | loop-external validator 用冻结 protocol/heldout 完成复验 |
| `rejected` | 合法负结果；保留 candidate、evidence 与 reasons |
| `retired` | 不再投入资源；历史 lineage 不删除 |

Praxist 的 `preliminary/mature/diagnostic`、`parent_eligible` 与 lane 只作为 source provenance。
只有 `mature` 且非 late-after-boundary 的 handoff 才能请求 runtime promotion；它仍不能跳过外部复验。

### 4.2 Deployment state

运行时继续只使用既有：

```text
DISABLED → SHADOW → ACTIVE
```

- candidate 初始恒为 `DISABLED`；
- generic pipeline 只发布 authorization receipt，绝不修改 `FinalRolloutConfig`、环境变量或产品资产；
- target adapter 每次只应用一个字段，并另行生成实际 canary/deployment receipt；
- 回滚使用相反的单步转换，保留全部历史 receipt。

### 4.3 Release mode

所有研究任务都使用共同的 Task/Handoff/Candidate/Validation 契约，但终点可以不同：

| `release_mode` | 终点 |
|---|---|
| `evidence_only` | 外部验证后封存，不产生 apply/wiring authorization |
| `repository_apply` | 继续走既有 `forge validate/apply` + named human review，不使用 WiringLevel |
| `runtime_wiring` | 才允许请求本 spec 的 SHADOW/ACTIVE authorization receipt |

## 5. 五类正式工件

统一 JSON Schema：`forge/schemas/research_promotion.schema.json`。所有 path/hash/schema 违反均
fail loudly；科学负结果则形成 `BLOCKED` receipt，不伪装成基础设施错误。

### 5.1 `forge-research-task.v1`

Task 是 Volvence 版本化、Praxist 只读的研究与发布合同，至少冻结：

- `task_id / claim_id / owner / capability_axes / objective`；
- source base revision 与 baseline content ref；
- Praxist task project id 和 manifest digest；
- sandbox `editable_roots / protected_roots`；
- development evaluator id、不同的 loop-external formal validator id 与 formal protocol ref；
- SHADOW / ACTIVE 各自 required check 名单；
- release mode、exact target、`ModificationGate` authority、rollback instructions；
- 五个恒假权限位：Praxist 不得修改 Task、正式 evaluator、production wiring，不得读取 sealed
  heldout 或 production credentials。

`active_required_checks` 必须是 `shadow_required_checks` 的超集。开发 evaluator 与正式 validator
必须是不同 identity；换 evaluator、protocol、threshold、baseline 或 source revision等于新 Task revision，
不得就地继承旧 authorization。

### 5.2 `forge-praxist-candidate-handoff.v1`

Handoff 是 task-local exporter 写出的稳定桥接面，绑定：

- `run.json`；
- `task_project_manifest.json`；
- `gen_N/generation_boundary.json`；
- 一个 recognized result summary；
- variant/result artifact 到 Volvence target path 的显式文件映射及逐文件 SHA-256；
- generation、variant、parent lineage、retention lane/maturity/parent eligibility；
- `late_after_generation_boundary`。

importer 只接受 run root 内 regular files，拒绝绝对 locator、`..`、symlink、hash 漂移和 path
越界。它必须验证 Praxist `run_id/task_id/task_project manifest/generation boundary` 的交叉绑定。
Leaderboard、PI memo、prompt、report 等 derived/audit view 不能替代这些 canonical inputs。

Handoff 固定声明：

```text
research_retention_only = true
production_promotion_authorized = false
requested_wiring = disabled
formal_validation_performed = false
```

### 5.3 `forge-research-candidate.v1`

Candidate 是 Forge 导入后发布的 content-addressed bundle：

- `candidate_id = research-candidate:<canonical payload sha256>`；
- 冻结 Task/Handoff、本次 Praxist canonical refs 与全部候选文件 hash；
- 保留 source retention/maturity/parent lineage；
- `research_maturity=candidate_sealed`；
- `gate_decision=not_evaluated`；
- `wiring_level=disabled`；
- `production_promotion_authorized=false`。

candidate 不复制或解释模型内部状态；每次授权前重新读取并复算所有 source bytes，防止 import 后
替换文件。candidate 本身不是 owner snapshot，也不得进入 PE/credit 学习链。

### 5.4 `forge-research-validation.v1` 与 `forge-research-gate.v1`

Validation 由 Task 指定的 loop-external validator 发布，必须：

- exact 绑定 candidate id/raw SHA；
- validator id 与 formal protocol path/hash 等于 Task 冻结值；
- 明示 `sealed_holdout.used=true`、`visible_to_praxist=false`；
- 逐项发布 named check、boolean result 与内容寻址 evidence refs；
- `PASS` 当且仅当全部 checks 为 true；
- 固定 `evaluation_is_learning_source=false` 与 `production_promotion_authorized=false`。

Validation artifact、本轮 evidence refs、Gate artifact 与 gate review 都必须物理位于 Praxist run root
之外；仅把 `validator_id` 或 gate authority 字符串写成正确值不能证明 loop-external 身份。

Gate artifact 由 target-specific、loop-external adjudicator 调用正式
`volvence_zero.credit.gate.evaluate_gate_reasons` 后规范化发布。Forge core 只核对其 authority、
ALLOW/BLOCK、candidate/validation/target hashes，不重新实现或推断 gate。`BLOCK` 必须带 reasons。

### 5.5 `forge-research-promotion-receipt.v1`

Receipt 是 authorization，不是 execution proof。它冻结：

- requested/resulting wiring；
- release mode 与 exact target；
- `AUTHORIZED/BLOCKED` 与全部 blocking reasons；
- Task/Candidate/Validation/Gate/previous receipt hashes；
- 本阶段实际满足的 required checks；
- named human authorizer 与理由；
- rollback target/instructions；
- `runtime_mutated=false`、`production_default_changed=false`、
  `target_adapter_apply_required=true`。

receipt id 排除 timestamp 后由 canonical payload 计算。旧 receipt 不覆盖；下一步通过
`previous_receipt_sha256` 形成 append-only hash chain。

## 6. 状态转换与门

### 6.1 Import

```text
committed Praxist handoff
  → validate task/run/boundary/result/files
  → candidate_sealed + gate:not_evaluated + wiring:disabled
```

Import 不要求 candidate 在 Frontier，也不把 preliminary/negative evidence 丢弃；但任何 handoff 都
只能得到 `DISABLED` candidate。

### 6.2 SHADOW authorization

只允许 `DISABLED → SHADOW`，并同时要求：

1. candidate source 重新复算无漂移；
2. Praxist source maturity 为 `mature` 且不晚于 generation cutoff；
3. loop-external Validation 是 exact-bound `PASS`；
4. Task 的全部 `shadow_required_checks` 为 true；
5. exact-bound `ModificationGate.OFFLINE` 为 `ALLOW`；
6. named human authorization 非空。

任一科学门失败写 `BLOCKED` receipt，resulting wiring 保持 `DISABLED`。
首次授权不需要 previous receipt；从一次已授权 rollback 后重新上架时，必须输入紧邻的
`AUTHORIZED` DISABLED receipt，并使用新的 Validation/Gate evidence，保持 hash chain 连续。

### 6.3 ACTIVE authorization

只允许 `SHADOW → ACTIVE`，除 SHADOW 全部要求外还必须：

- 输入紧邻的、同 candidate 的 `AUTHORIZED` SHADOW receipt；
- 新 Validation/Gate exact 绑定本次 active evidence；
- Task 全部 `active_required_checks` 为 true，其中必须包含 shadow observation、rollback/canary
  等 task-owned 名称；
- target adapter 后续仍按自身 promotion order、source fingerprint 与 canary 契约执行。

`DISABLED → ACTIVE` 永远是非法转换，不产生 receipt。

### 6.4 Rollback

rollback 不依赖新的收益证据或 ALLOW：

```text
ACTIVE → SHADOW
SHADOW → DISABLED
```

它要求同 candidate 的 previous authorization receipt、named operator 与原因，并发布新的
`AUTHORIZED` rollback receipt。generic receipt 仍不亲自改 wiring；target adapter 应优先执行回滚，
再发布实际部署回执。不得因当前 evidence 缺失而阻止降低权限。

## 7. Praxist takeover 沙箱

“takeover”只表示 Praxist 接管 Task 的 development research loop：假设、变体、开发 evaluator、
Finding、Frontier/Gems 和下一代议程。它不能接管以下对象：

- Volvence Task contract、formal protocol 与 required checks；
- sealed heldout 与独立 validator；
- gate/evaluator/permission/promotion pipeline 源码；
- production credentials、runtime state 与 deployment config；
- target owner 的 apply surface。

推荐三层物理隔离：

1. **Research sandbox**：pinned base revision 的 disposable worktree/container，只挂载开发 split 与
   allowlisted editable roots；
2. **Validation enclosure**：只读 candidate + sealed heldout，禁写 candidate，Praxist peer 不可见；
3. **Runtime SHADOW/canary**：exact admitted artifact 进入真实 owner 接线，但 SHADOW 不影响 active
   snapshots 或用户可见输出。

同一 owner 的多个研究 Task 默认不得并发进入部署阶段。跨 owner 的 artifact 先各自冻结，再用新的
composition Task 验证；禁止一个 Praxist task 同时改多个 owner 并用总分掩盖归因。

## 8. 四能力轴与学习信号边界

本流水线对四轴只做治理：

- **Appendable**：保存研究 lineage 与 receipt chain，不等于 CMS 跨 session 能力成立；
- **Readable**：只消费 owner/task 发布的结构化 artifact，不重建内部状态；
- **Learnable**：Praxist evaluator 只调度研究，永不写入 Volvence PE/credit；runtime 学习仍只有
  `PE → credit`；
- **Steerable**：只为已有 bounded target 发行 wiring authorization，不创造 free-bias、无界 action
  或 token-space RL。

任何 research/evaluation score、LLM judge、Frontier rank 或 PI/Chair 建议直接进入 PE/credit 都是
契约违反。

## 9. v1 promotion artifact 实现面

Forge CLI 提供四个纯离线入口：

```text
forge research-validate-task <task.json>
forge research-import-praxist <task.json> <handoff.json> --run-dir <run>
forge research-authorize <task.json> <candidate.json> <validation.json> <gate.json> --to-wiring shadow|active
forge research-rollback <previous-receipt.json> --to-wiring disabled|shadow
```

以上四个 promotion 命令只生成/校验 artifact 与 authorization receipt：

- 不初始化具体 Praxist task；
- 不启动 Praxist run；Praxist A0 启动由独立的
  [`research-control-plane.md`](./research-control-plane.md) 在 exact human approval 后负责；
- 不运行 formal evaluator；
- 不调用 `ModificationGate`；
- 不 apply patch、加载 artifact 或翻转 runtime wiring；
- 不修改 `forge/ledger.jsonl`；receipt chain 自身是本包的 append-only audit surface，后续可在不改变
  artifact identity 的前提下增加 ledger projection。

rollback 命令只读取 previous receipt，不重新读取可能已经归档、损坏或离线的 Task、candidate、
Praxist run 与收益证据；降低权限不能被研究输入的当前可用性阻断。

组件专用 gate adapter 和 deployment consumer 继续位于循环外。把第一个
`coding_memory_inheritance` task 接入后，才能验证完整 task-specific evaluator 与 target apply seam。

## 10. Failure semantics 与安全检查

- JSON/schema、path containment、symlink、hash、identity、run/boundary 绑定错误：raise/fail loudly；
- 合法 Validation `BLOCK`、Gate `BLOCK`、required check false、source maturity 不足：写
  `BLOCKED` receipt，保留负证据；
- 缺 previous SHADOW receipt、跨级 ACTIVE、跨 candidate receipt、release mode 非
  `runtime_wiring`：非法请求，拒绝生成 artifact；
- authorization 输出只能位于 `artifacts/`；run output 不写进 Praxist source checkout；
- checker 必须同时有 deliberately-broken fixture：篡改 source、越界 target、Frontier 冒充 gate、
  direct ACTIVE、跨 candidate previous receipt 至少各一项；
- 不允许宽泛 fallback、从 prose/关键词推断 status，或缺字段时猜默认 ALLOW。

## 11. 迁移、退出与回滚

本包不改变现有 Forge proposal、rare-heavy、steering B3 或任何 production default；它提供共同的
上游 envelope。既有 component-specific 路径可以逐个 opt-in：先双写原工件和通用 receipt，比对 hash/
判词，再让 target adapter 要求通用 receipt。未 opt-in consumer 行为不变。

退出本通用层：停止生成/消费 `forge-research-*.v1`，删除 Forge CLI/module/schema 和本 spec/index
引用；既有 component gate、SHADOW/ACTIVE 与 rollback 路径仍可独立工作。已生成 artifact 保留审计史。

回滚某个已部署 candidate：不删除 Praxist run 或负证据，按 receipt chain 执行
`ACTIVE→SHADOW→DISABLED`，由 target owner 恢复前一 content hash，并保留实际 canary/deployment receipt。

## 12. 已知限制与下一包

- v1 importer 依赖 task-local exporter 提供规范化 Handoff；不会猜测任意 task 的 variant 语义；
- companion Research Control Plane 已提供 Request/Approval/Event registry 和 host-wide 单 active-run
  调和，但它尚未接 automatic opportunity detector，也不会从 run completion 自动生成 Handoff；
- generic authorization 不证明 target adapter 已执行，实际部署仍需 owner-specific receipt；
- v1 不替代组件特有统计、消融、安全、延迟和人类 anchor；required check 只统一命名与绑定；
- 尚未建立跨 Task deployment portfolio/dependency registry；Research Control inbox 只仲裁 host 上的
  Praxist 启动，不拥有同 owner 的候选组合或部署并发；
- v1 receipt 形成可验证 hash chain，但尚无全局 latest-pointer/单写者 registry；target adapter 必须核对
  自己的当前 deployment receipt，不能把一条旧分支自动当成现态；
- 尚未接入第一个真实 Praxist run，因此本包只证明机制与拒绝面，不构成任何研究或产品效果证据。

下一收敛包应只接一个 pilot：`coding_memory_inheritance`。它负责 task project、开发 evaluator、
sealed heldout exporter、loop-external validator 和 memory-owner target adapter；不得同时接 steering/CMS
第二个 owner。

## 13. 参考

- [`rsi-forge.md`](./rsi-forge.md)
- [`research-control-plane.md`](./research-control-plane.md)
- [`evidence_program.md`](./evidence_program.md)
- [`credit-and-self-modification.md`](./credit-and-self-modification.md)
- [`evaluation-cascade.md`](./evaluation-cascade.md)
- [`steering-runtime.md`](./steering-runtime.md)
- [`../moving forward/主线提升方案_2026-08.md`](../moving%20forward/主线提升方案_2026-08.md)
- Praxist `docs/concepts/architecture.md`、`docs/guides/task-projects.md`、
  `docs/guides/research-loop-variant-generation-flow.md`
