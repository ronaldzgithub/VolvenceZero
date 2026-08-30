# Research Opportunity Discovery：类型化研究机会提名与 A0 提交

> Status: v1 mechanism landed；首个 exact pilot 已生成 A0 Request 并等待人审；无有效 Praxist run
> Last updated: 2026-08-29
> Owner: `volvence_forge`（development-plane opportunity artifacts only）
> Downstream contract: [`research-control-plane.md`](./research-control-plane.md)

## 1. 决策摘要

自动 Praxist 的合法入口不是“让一个根目录脚本阅读任意文本并自行开题”，而是把已有 Forge
failure mining 与 A0 Research Control Plane 接成两段明确的权限链：

```text
structured evidence
  → forge mine（语义分析与聚类）
  → forge-failure-pattern.v3
  → forge research-scan --once（本 spec）
  → immutable ResearchOpportunity
  → exact task-registry routing
  → immutable routing receipt
  → ResearchRequest / NEEDS_TASK_DESIGN
  → named-human A0 review
  → research-reconcile --once
  → Praxist doctor / resolve / detached start
```

本包只实现 `failure-pattern → opportunity → Request`。它不批准 Request、不调用 Praxist、不创建或
修复 task project、不读取 sealed holdout、不执行 formal validation、不调用 `ModificationGate`，也不改变
`DISABLED / SHADOW / ACTIVE`。

## 2. 唯一 owner 与数据边界

`volvence_forge` 是以下工件的唯一 writer：

- `forge-research-opportunity.v1`：从一个 exact typed failure record 形成的不可变提名；
- `forge-research-opportunity-routing.v1`：一次 exact registry 版本下的路由结果；
- `forge-research-task-registry.v1` 的加载与机械校验逻辑。

上游 failure cause、evidence ref、component 和 target 仍由 Forge mining owner 发布；scanner 只复制冻结
record，不重新解释其 prose。Task 的 objective、owner、四能力轴、evaluator、sandbox 与 release policy
仍由 `forge-research-task.v1` owner 发布。Praxist 继续唯一拥有 task resolution、run、peer、generation、
Frontier/Incubator/Gems 与进程生命周期。

这些都是离线 development artifacts，不是 Brain snapshot，不注册 `docs/DATA_CONTRACT.md` runtime slot，
不进入 PE/credit，也不把 occurrence count 或 research status 当作 reward。

## 3. 唯一允许的发现输入

v1 只接受逐行通过 `failure_pattern.schema.json` 验证且 schema version 精确为
`forge-failure-pattern.v3` 的 JSONL。scanner 还会重算：

- `pattern_id` 对 causal fields + evidence refs 的 canonical identity；
- 每条 record 的 canonical SHA-256；
- 输入中 `pattern_id` 唯一性。

空 JSONL 是合法的“本轮无机会”。非法 JSON、旧 schema、伪造 identity 或重复 id 都 fail loudly。

`forge mine` 可以使用结构化 LLM 与 embedding 形成 failure pattern；但进入本 spec 后不再调用模型。
特别是：

- `title / verifier_cause / agent_behavior_cause / exposed_mechanism / excerpt` 只作证据；
- task routing 绝不读取这些 prose；
- 不允许关键词、正则、字符串包含或 embedding 相似度选择 task、owner、预算或 model。

## 4. Task registry

默认注册表为 `forge/research_task_registry.yaml`，schema 为
`forge-research-task-registry.v1`。它是人审治理面并被 Forge editable policy 明确标为只读。当前只登记
`coding_memory_inheritance_v1` 这一条 exact component/target pilot；mapping 只能把 typed opportunity
提交为未授权 Request，不能自行批准或创建真实研究预算。

每个 mapping 必须冻结：

- 唯一 `mapping_id` 与人工维护的 `binding_revision`；
- exact `editable_component`，以及可选 exact `editable_target`；
- Volvence ResearchTask、Praxist task project、Praxist executable、run root；
- config file、agent system、runtime、Codex-native 开关、provider、exact model、strategy、cohort、
  generations 与 startup timeout。

相对路径相对于 registry 文件目录解释；提交前全部解析为 exact path/content refs。component-wide mapping
与同 component 的 target-specific mapping 不得并存，以免隐藏优先级。重复 id、重复 match 或任何重叠
都 fail loudly。

`praxist_executable` 兼容显式路径和保留值 `auto`。`auto` 只负责当前主机的启动前发现，固定顺序为：

1. `FORGE_PRAXIST_EXECUTABLE`；
2. 兼容既有 launcher 的 `RESEARCH_LAB_PRAXIST`；
3. 仓库同级 `PRAXIST/.venv/bin/praxist`；
4. 当前 `PATH` 中的 `praxist`；
5. `~/.venvs/praxist/bin/praxist`。

显式环境覆盖存在但无效时必须 fail loudly，不得静默改用其他安装。自动发现结果必须是 non-symlink、
regular、executable file。`run_root` 允许 `~` 展开，因此 registry 可以使用
`~/.local/share/praxist/volvence_runs` 同时服务 macOS 与 WSL。发现只发生在 registry→binding 边界；
归一化 binding、Request 和 A0 仍冻结当前主机的 absolute executable locator、文件 SHA-256、absolute
run directory 与完整 profile。跨主机已有 Request 不可移植或原地改写；另一主机必须重新扫描并形成自己的
exact-bound Request/A0。

由于 Task、task project、registry 都进入 raw-byte identity，本 pilot 对这些 exact-bound text roots 在
`.gitattributes` 中固定 `eol=lf`。这不是放宽 hash，而是确保相同 Git revision 在 macOS 与承载于 Windows
工作树的 WSL 中物化为相同 bytes；未列入固定面的数据仍按各自 owner contract 处理。

`binding_revision` 不是 hash 的替代品。scanner 同时绑定 Task bytes、Task 声明的 task-project manifest
digest、Praxist executable bytes、config bytes 与完整 profile；当 source checkout、外部依赖、dataset/
simulator identity 等不在这些 bytes 中的运行前提变化时，operator 必须显式提升 `binding_revision`。

## 5. Immutable artifact contracts

正式 schema：`forge/schemas/research_opportunity.schema.json`。

### 5.1 `forge-research-opportunity.v1`

Opportunity 保存：

- 输入 JSONL 的 content ref、line number、record SHA-256 与完整 typed record；
- 独立于容器路径的 `opportunity_key`，用于相同 record 的幂等去重；
- exact component/target 与 `ROUTABLE | NEEDS_TASK_DESIGN`；
- 只由 `min(occurrence_count, 1000)` 产生的 scheduling priority；
- 全部 authority false：只提名，不授权研究、验证、上线或 runtime wiring。

只有 `surface_status=in-surface` 且 component/target 均存在时才是 `ROUTABLE`。out-of-surface 或缺字段
保留为 `NEEDS_TASK_DESIGN`，不会被丢弃或硬塞给“最接近”的任务。

存储路径为：

```text
artifacts/research_opportunities/
  <opportunity-key-digest>/<opportunity-id-digest>/opportunity.json
```

### 5.2 `forge-research-opportunity-routing.v1`

Routing receipt 精确绑定 opportunity bytes、registry bytes 和匹配后的 binding SHA-256，并发布三种
decision：

| Decision | 含义 | Request |
|---|---|---|
| `NEEDS_TASK_DESIGN` | source 不可路由，或无 exact registered task | 无 |
| `DEFERRED_BY_SCAN_LIMIT` | task 已精确匹配，但本轮新 Request 配额耗尽 | 无 |
| `SUBMITTED_FOR_A0` | 已形成 exact-bound ResearchRequest | 有，仍未批准 |

匹配成功的 receipt 同时发布 Task 的 `task_id / owner / capability_axes`，但不成为这些字段的第二
owner；它们来自已验证的 ResearchTask snapshot。

## 6. Scan 与幂等算法

每次 `research-scan --once` 获取独立 scan lock，并按以下顺序执行：

1. 验证 registry 与全部 typed failure records；
2. 对相同 canonical record 复用既有 Opportunity；
3. 按 `priority score DESC, opportunity_id ASC` 排序；
4. 只按 exact component/target 查 registry；
5. 无 mapping 写 `NEEDS_TASK_DESIGN` receipt；
6. 有 mapping 时验证 Task、task project、executable 和 profile，派生 deterministic run id；
7. 优先恢复已有 submitted receipt 或“Request 已写、receipt 未写”的 crash boundary；
8. 在 `max_new_requests_per_scan` 内调用既有 `submit_research_request()`；超额写 deferred receipt。

Request 的 trigger kind 固定为 `forge_failure_pattern`，submitter 固定为 scanner protocol identity，evidence
只绑定 Opportunity artifact。它继续由 `research-control-plane.md` 保证 exact task/project/executable/run/
profile binding 与 A0。

若 Request 已存在甚至 run dir 已由后续生命周期占用，scanner 会从 routing receipt 或
`artifacts/research_control` 中恢复 exact Request，不会因重复扫描再次 start 或另建 Request。Request
被 REJECT/BLOCK/完成后也不会因相同 opportunity + binding 自动重开；新研究轮必须有新的 typed
opportunity 或显式提升 binding。

## 7. CLI 与自动执行

单次命令：

```bash
forge research-scan '<failure_patterns.jsonl>' \
  --registry '<research_task_registry.yaml>' \
  --once \
  --json
```

省略 `--registry` 时使用 `forge/research_task_registry.yaml`。命令返回 0 代表扫描本身契约成立；
`NEEDS_TASK_DESIGN` 和 `DEFERRED_BY_SCAN_LIMIT` 是合法业务状态，不伪装成进程错误。

无需根目录常驻脚本。launchd/systemd/CI/Codex automation 等外部 scheduler 可以按有界步骤调用：

```text
mine one immutable evidence window
  → research-scan --once
  → notify human from research-inbox
  → human research-approve
  → research-reconcile --once
```

外部 scheduler 只负责唤醒和传递 exact artifact path，不拥有 queue truth、审批或 Praxist PID。

## 8. Human gate 与后续上架

`SUBMITTED_FOR_A0` 只说明 Request 已进入 inbox。没有 named-human `research-approve`：

- `research-reconcile` 不调用 doctor/resolve/start；
- 不消耗 Praxist research budget；
- 不产生 candidate；
- 不获得 SHADOW/ACTIVE authority。

研究完成后仍必须走 committed handoff → loop-external formal validation → cognition
`ModificationGate.OFFLINE` → A1 SHADOW → canary → A2 ACTIVE。Opportunity priority、Praxist ranking、
Frontier、Gems 或 completion status 均不能替代其中任何一步。

## 9. 四能力轴与诚实主张

- **Appendable**：Opportunity 与 routing receipt create-only、内容寻址、可跨进程恢复；它们不是 CMS。
- **Readable**：只消费 typed record/Task/Request；不从 prose 或日志重建 owner state。
- **Learnable**：本包不学习；occurrence count 只作队列排序，不成为 PE、credit 或 reward。
- **Steerable**：本包不改变 runtime；所有 wiring authority 恒假。

因此本包只能声称“研究机会可类型化提名并安全进入 A0 队列”，不能声称自动科研效果、四能力闭环、
formal PASS 或 production 上线。

## 10. 回滚、限制与下一包

停止 scheduler 即停止新扫描；既有 Opportunity、receipt 与 Request 保留审计史。移除 mapping 会阻止
未来同 binding 的提交，但不会删除或撤销既有审批/run；运行中的 Praxist 仍需显式
`praxist stop <run_id>`。本包没有 runtime wiring，因此没有 ACTIVE rollback。

v1 限制：

- 只消费 Forge failure-pattern，不接 prediction-error/benchmark/protocol-gap 的其他 typed adapter；
- registry 只登记一个 `coding_memory_inheritance` pilot，不提供通用自动 task design；
- 没有自动 task design/repair；缺任务只发布 `NEEDS_TASK_DESIGN`；
- 只做每轮新 Request 上限，不做 GPU/resource portfolio、租约或跨 host 调度；
- 首个 Request
  `research-request:8f44be1d4cdeab1b9a3c34ea4f3f84b292521fac390dc3d79fb6a6dd88ae6be9`
  已处于 `AWAITING_RESEARCH_APPROVAL`；named-human A0 前没有启动权；
- 一条绕过 Forge 的 direct-CLI run 已被停止并排除，说明 v1 能判定 out-of-band identity 无效，但不提供
  OS-level Praxist execution denial。

下一收敛包应只在用户审阅并批准 exact A0 后运行该 `coding_memory_inheritance` pilot，验证
Opportunity → A0 → real Praxist → committed handoff；不得同时扩第二个 owner，也不得跳过 A1/A2。
