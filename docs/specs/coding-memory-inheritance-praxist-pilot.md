# Coding Memory Inheritance Praxist Pilot

> Status: runnable task + portable host registry landed；历史 macOS A0 Request 保留，WSL 尚未提交本机 Request；无有效 research run、formal validation 或 runtime wiring
> Last updated: 2026-08-30
> Owner: `vz-memory`（研究对象语义）；Forge（A0 生命周期与上架 artifact）
> 对应能力轴: Appendable、Readable；不声称 Learnable / Steerable

## 1. 目标与边界

这是统一研究流水线的第一个真实任务绑定。它把 Coding Lab Packet 2 暴露出的一个窄问题交给
Praxist 开发沙箱研究：如何用结构化、可审计的 `policy.json` 在保留 owner 发布的 recalled/failed
evidence 时，把 coding-memory context 控制在冻结比例 `0.10` 以内。

本 pilot 只研究 context composition policy。它不允许 Praxist 修改 `MemoryStore`、Brain、Task、
evaluator、冻结 trajectory、formal validator、`ModificationGate` 或 runtime wiring。Praxist Frontier、
Incubator、Gems 和 PI/Chair 只拥有 development research-retention 语义，不是上线权。

该任务只涉及离线 task/artifact，不新增 runtime snapshot 或公共 slot，因此本包不修改
`docs/DATA_CONTRACT.md`。未来 target adapter 真正接入 `memory_inheritance_policy` 时，必须先注册唯一
owner、value type、dependencies 和 `WiringLevel`。

## 2. 已冻结问题证据

### 2.1 Formal provenance

冻结 Packet 2 v3 报告：

- 24 chains × 10 episodes × 3 arms；
- `context_token_ratio = 0.10592031659161838 > 0.10`；
- memory 与 quality gates 保持通过；
- report SHA-256:
  `9ad76d2f48b32ab8a40bd5288b0dfd52814f6b79970275641a960ce51ace0e42`。

这条记录证明存在 scaling failure，同时要求后续候选保留原质量结论；它不是 Praxist evaluator 的
可变 score。

### 2.2 Public development replay baseline

Task evaluator `coding-memory-context-replay.v2` 在公开 chains 0–7 上重放 legacy policy，不调用 coding
hand，也不调用模型 provider：

| 指标 | 完整 baseline |
|---|---:|
| evaluation units / post-first contexts | `8 / 72` |
| context token ratio | `0.1233350275668608` |
| scaling margin | `-0.02333502756686079` |
| worst-chain scaling margin | `-0.09437921925447607` |
| recalled / failed selection coverage | `1.0 / 1.0` |
| selected-line retention | `0.972027972027972 / 0.9846153846153847` |
| strict-budget pass rate | `0.0` |

Canonical summary SHA-256:
`44d3ff70c9e931694220c702e7a8eb4ab6b2d56c4283628e0bbc00d6910f58b7`。

旧策略同时暴露两个机制问题：generic sections 使平均比例超过上限；`legacy_tail_marker` 在截断后追加
marker，既越过 character budget，又截断已选择的 owner evidence。

## 3. Task project

Task root：`research/praxist_tasks/coding_memory_inheritance/`。

冻结内容包括：

- `task.yaml`、三层 prompt、10 个 task-local role 和 Multi-PI topology；
- structured policy schema 与只含 `policy.json` 的 variant surface；
- frozen public replay manifest、baseline、resource plan、protocol intent 与 research directions；
- preliminary / complete 两阶段 evaluator；
- audit rule、default/effective-config digest、replication、lane、alias 和 closing-policy regression。

当前 Request 绑定 task-project manifest：

- schema: `task_project_manifest.v1`；
- file count: `50`；
- manifest SHA-256:
  `093957ad09311dba58f223d454088a3373bb5fafb9dc761ba89d34b71d4e6128`；
- snapshot SHA-256:
  `0fbee87ed9ffdf80639f13a991425efd91fa8c5e9c482d3a39e42f9328ae95d0`。

Request 创建后上述任一 included byte 漂移都会在审批或 reconcile 时 fail closed，必须生成新 Request。

## 4. Policy 与 evaluator 契约

Praxist 只能改变声明式字段：context character cap、最大 recalled entries、section set/order、recall
order、whole-line truncation、exact-line deduplication 和 generic-section budget fraction。candidate 不能包含
可执行代码。

完整 development parent 至少满足：

1. 8/8 chain evidence，`effort_ratio = coverage_ratio = 1.0`；
2. recalled selection coverage `>= 0.75`；
3. failed selection coverage `>= 0.95`；
4. 每条已选择 evidence 必须作为完整行保留；
5. 每个 rendered context 都满足声明的 character budget；
6. `protocol_integrity_passed=true`，且无 smoke、partial、late、suspect protocol/leakage marker。

`confirmed` 还要求平均 context ratio `<= 0.10`；`incubator` 允许 scaling 尚未通过，但必须是完整、
协议干净且证据保留成立的 Pareto/new-high policy。preliminary、protocol-failed、suspect 或 late result
永远不能成为 parent。

人工编写的回归 fixtures 已证明 evaluator 与 Praxist Frontier 的判别面可达：

- confirmed fixture: ratio `0.0825977644`、selection `0.7552447552 / 1.0`、retention `1 / 1`；
- incubator fixture: ratio `0.1208499362`、同样保留 owner evidence，但 scaling 未过；
- 实际 Frontier selector 将二者分别放入 `confirmed / incubator`。

这些 fixtures 是 task sensitivity / lane regression，不是 Praxist 自主发现的收益，不得报告成 agent
research effect。

## 5. 资源与关闭策略

- CPU-only；host 为 10 logical CPUs / 24 GiB memory；最大 experiment concurrency `2`；
- preliminary canary: `59.01 s`；
- complete legacy baseline: `928.65 s`，最大 RSS `539,869,184 bytes`；
- 两个 complete fixtures 并发时各约 `778–779 s`、约 `0.5 GiB` RSS；
- planning estimate: `18 min`，safety factor `1.5`，close-grade reserve `27 min`；
- fixed/adaptive close horizon `70 min`，预留 `30 min` drain 后仍有 `40 min`，因此 launch guard 成立；
- cohort `4`、generations `3`、mature quorum fraction `0.25`，正常关闭至少需要一个 mature peer；
- 达到 quorum 前只进入 top-up assessment；安全上限与 cohort-drained insufficient-mature 保持独立
  liveness 出口。

## 6. Exact Codex-native profile

Registry mapping `coding_memory_inheritance_v1` 冻结 portable host policy：

- `praxist_executable: auto`：依次解析显式 host override、同级 PRAXIST checkout、PATH、user venv；
- macOS 当前预期解析为同级 `PRAXIST/.venv/bin/praxist`；
- WSL 当前实测解析为 `/home/ronald/.venvs/praxist/bin/praxist`；
- agent/runtime: `codex_sdk / agent_runtime:codex_sdk`；
- provider/model: `model_provider:openai_compatible / gpt-5.6-luna`；
- auth: current host saved ChatGPT login；
- strategy/cohort/generations: `mixed / 4 / 3`；
- requested run root: `~/.local/share/praxist/volvence_runs`，提交前展开为当前 host absolute path。

Codex-native doctor/resolve/start 清除 API key、base URL、binary 与 model 环境覆盖，防止宿主变量静默
改变冻结 profile。WSL Codex-native doctor 已通过；Praxist CLI 未加入 `PATH` 和 registry root 尚未预创建只是
warning，user-venv fallback 可直接处理。Task、task project 与 registry 的 exact-bound text 已固定
`eol=lf`，避免 Windows checkout 的 CRLF 让同一 Git revision 在 WSL 中产生不同 SHA-256。

## 7. Opportunity 与 A0 Request

Typed failure record `fp_c9f9e2529416e440` 只按 exact
`coding_memory_inheritance_policy + research/candidate_surfaces/coding_memory_inheritance/policy.json` 路由。
因果 prose、标题和 excerpt 不参与 task 选择。

portable-host-v1 之前的 macOS 不可变 artifact 保留为历史 host binding：

- Opportunity:
  `research-opportunity:6c829987868be6872a8da3fb7035111743ceb8439d008e980eda9fbaa09f7d13`；
- Request:
  `research-request:8f44be1d4cdeab1b9a3c34ea4f3f84b292521fac390dc3d79fb6a6dd88ae6be9`；
- requested run:
  `run_6c829987868be6872a8d_8335931713ec`；
- historical state: `AWAITING_RESEARCH_APPROVAL`；
- approval/event: none；
- `research_start_authorized=false`、`production_promotion_authorized=false`。

该 Request 只可在其冻结的 macOS path/bytes 上继续 A0，不得搬到 WSL。portable-host-v1 下每台机器的下一
合法动作是重新扫描形成该 host 自己的 exact Request，再由 named human A0 approve/reject。A0 仍只授权一次
Praxist development lifecycle；不授权 formal validation、candidate admission、SHADOW 或 ACTIVE。

## 8. Out-of-band launch incident

2026-08-29 22:06，本机另一路径绕过 Forge，直接创建：

`run_2026-08-29_14-06-13-675653_coding_memory_inheritance`

它位于 task-local `experiments/`，不等于 Request 冻结的 run id/dir；当时没有 Approval，Request 仍为
`AWAITING_RESEARCH_APPROVAL`。该 run 在 generation 0 初始化阶段被停止，约运行 102 秒，发布 `0`
finding，最终状态 `failed/stopped`。目录和 Praxist registry 记录保留供审计。

该 incident 永久排除在本 pilot 的 research、Frontier、Handoff 和 promotion evidence 之外。它说明：

- Forge 对所有**经 Forge 发起**的 start 强制 A0、hash recheck 和 exact run identity；
- 拥有主机 shell 权限的操作者仍可绕过 Forge 直接调用 Praxist，v1 不提供 OS-level execution denial；
- out-of-band run 必须按 identity 判无效，不能因为使用相同 task/model 就冒充 approved run。

## 9. 上架链与 rollback

合法路径保持：

```text
A0 exact Request approval
  → doctor / resolve / detached start / targeted status
  → committed Praxist generation + task-local Handoff
  → A1 loop-external sealed coding-hand validation
  → A2 ModificationGate review
  → named-human DISABLED→SHADOW
  → shadow observation + canary
  → named-human SHADOW→ACTIVE
```

Praxist run failure、weak/negative result 或空 Frontier 不触发部署，只保留研究证据。A0 前 rollback 是
reject Request 或不审批；run 中断走 Praxist stop/resume，不能改用另一个 run identity；部署回滚只允许
`ACTIVE→SHADOW→DISABLED` 相邻降级并恢复前一个 admitted artifact hash。

## 10. 已知阻塞与下一包

- sealed coding-hand validator 与 candidate-aware target adapter 尚未实现；
- public replay 与历史 formal corpus 有重叠，不能充当 sealed holdout；
- regression fixture 虽证明 task sensitivity，但尚无 approved autonomous Praxist finding；
- host-level direct CLI bypass 只能检测和排除，尚无 OS-level admission enforcement；
- formal validation、ModificationGate、SHADOW 与 ACTIVE 均未授权。

下一收敛包只能在用户明确审阅并批准所选 host 的 fresh exact Request 后，启动该 Request 冻结的 run，
观察 generation 0 到合法 boundary，并生成 committed Handoff。
不得复用已停止的 out-of-band run，也不得同时登记第二个 owner。

## 11. 参考

- [`research-opportunity-discovery.md`](./research-opportunity-discovery.md)
- [`research-control-plane.md`](./research-control-plane.md)
- [`research-promotion-pipeline.md`](./research-promotion-pipeline.md)
- [`coding-lab.md`](./coding-lab.md)
- [`appendable-readable-learnable-steerable.md`](../appendable-readable-learnable-steerable.md)
