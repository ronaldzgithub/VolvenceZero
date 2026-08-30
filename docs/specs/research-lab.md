# Research Lab：Forge → Praxist → SHADOW → ACTIVE 统一控制台

> Status: v1.7；snapshot v2 + demand discovery inbox + exact topic binding + managed automatic worker implemented
> Last updated: 2026-08-30
> Owner: `volvence_labs.portal`（read-only aggregation and command delegation only）
> Upstream contracts: [`research-opportunity-discovery.md`](./research-opportunity-discovery.md)、[`research-control-plane.md`](./research-control-plane.md)、[`research-promotion-pipeline.md`](./research-promotion-pipeline.md)

## 1. 决策摘要

Research Lab 是所有研究任务共用的**操作面**，把现有的多段权限链显示在同一个工作台：

```text
typed signal
  → Forge Opportunity / Task design
  → exact ResearchRequest
  → A0 human approval
  → Praxist preflight / detached research
  → retained candidate / handoff
  → loop-external formal validation
  → ModificationGate review
  → A1 SHADOW authorization / observation
  → A2 ACTIVE authorization / canary
  → rollback or retained ACTIVE
```

需求驱动轨道在 Task 之前增加一个不具执行权限的 discovery inbox：

```text
Volvence ResearchDemand → read-only Codex DiscoveryRun → UNBOUND TopicProposal
  → named-human DemandBinding → exact ResearchRequest → A0 → Praxist
```

这条前置轨道不改变后续 Forge/Praxist/promotion owner；完整协议见
[`demand-driven-research-loop.md`](./demand-driven-research-loop.md)。

Lab 不成为新的研究、验证、gate 或 wiring owner。它只做两件事：

1. 从各 owner 发布的不可变 artifact 和 Praxist targeted status 生成一个 frozen、可刷新、只读的
   `ResearchLabSnapshot`；
2. 把人类在 UI 中提交的 typed command 委托给现有 owner seam，并展示 owner 返回的新 artifact。

因此 Lab 不能用按钮直接改数据库、写 candidate、启动任意命令或切换 production wiring。UI 上的一个
“批准”动作仍必须产生与 CLI 完全相同的 exact-bound Approval/receipt。

外部领域走一条不与 Volvence promotion 混合的轨道：Foundry `foundry-research-lab-intent.v1` 经 external
descriptor 形成 exact Request，复用 A0/Research Lab runner 调和，完成后终止于
`forge-foundry-research-handoff.v1` immutable simulation handoff。该轨道没有
Candidate、ModificationGate、SHADOW、ACTIVE 或 runtime wiring。完整协议见
[`external-research-adapter.md`](./external-research-adapter.md)。

## 2. 唯一 owner 与边界

| 数据/动作 | 唯一 owner | Lab 权限 |
|---|---|---|
| FailurePattern / Opportunity / routing | `volvence_forge.research_opportunity` | 只读；可委托一次 bounded scan |
| ResearchDemand / DiscoveryRun / TopicProposal / DemandBinding | `volvence_forge.research_discovery` | 只读；可委托 exact named-human topic binding |
| ResearchRequest / Approval / Event | `volvence_forge.research_control` | 只读；可提交 exact A0 review 与 reconcile |
| Foundry Intent / budget / evidence class / adoption | Foundry | 只读 exact binding；Lab 不重分类、不采纳、不 apply；Foundry consumer 只可导入 completed handoff |
| external descriptor / simulation handoff | `volvence_forge.research_control` | 可委托 submit/handoff；不转入 Volvence promotion |
| task resolution、run、generation、frontier | Research Lab/Forge 选择的 runner（当前 Praxist） | 只读 targeted status；启动只经 approved Forge reconcile，Foundry 不可直连 |
| candidate import / formal validation / gate receipt | Forge promotion pipeline + named validator/Gate | 只读；可委托 exact import/authorize |
| runtime wiring | target owner | 只读 receipt；Lab 不直接 mutate |
| Labs probe CAS / experimental promotions | `volvence_labs` | 保留为实验域，不冒充 production authority |
| `ResearchLabSnapshot` | `volvence_labs.portal` | 唯一发布者；web client 只读 |

本模块位于 offline development plane，不是 Brain runtime module，不进入 PE/credit，也不注册
`docs/DATA_CONTRACT.md` runtime slot。若未来把 Lab 状态接入 Brain，必须另开独立 contract package。

## 3. Canonical snapshot

`ResearchLabSnapshot.v2` 是 frozen dataclass/JSON view，至少包含：

- `schema_version / generated_at / repo_revision / source_health`；
- `summary`：各 lifecycle stage 数量、active run 数、blocked 数、待人审数；
- `discovery`：registry ref、Demand、latest run/backend/model、TopicProposal、source claims、Binding/Request effective state；
- `items[]`：稳定 `item_id/task_id/research_mode/claim_id/owner/capability_axes`；
- `lifecycle`：当前 stage、允许的下一 stage、blocking reason、last transition；
- `bindings`：Task、Request、Approval、run、candidate、validation、gate、receipt 的 content refs；
- `run`：Praxist `run_id/state/pid/generation/findings/run_dir`，无匹配时显式 null；
- `evidence`：development、formal、shadow、canary 分层，禁止混成一个 score；
- `authority`：A0/A1/A2 与 production promotion 均逐字段显示，不从 stage 名推断；
- `available_actions[]`：由 owner contract 和 exact artifact 是否齐全机械派生；
- `warnings[]`：hash drift、source drift、stale status、缺 validator/adapter 等 fail-closed 状态。

collector 不遍历 producer 私有结构重建状态。只接受正式 JSON/schema、Praxist JSON 输出和 Labs 自有 CAS；
任何 malformed artifact 单独形成 typed warning，不能静默跳过后让 UI 显示“健康”。Promotion 视图不能把各目录
“按时间最新”的 Candidate、Validation、Gate、Receipt 直接拼接：Candidate 来自
`artifacts/research_promotion/`，Validation/Gate 同时读取各自 owner 的
`artifacts/research_validation/`、`artifacts/research_gate/`，下游 artifact 必须用 candidate id/raw SHA 和
validation raw SHA 精确绑定后才进入同一视图。Receipt 作为上一轮 authorization boundary 单独显示，以允许新一轮
formal/gate evidence 在其后形成 A2 输入。

Web consumer 必须逐层校验 snapshot v2 的 discovery、summary、source health、item、`research_mode`、lifecycle、authority、
evidence、binding、run 与 warning shape；不得只检查顶层版本后用缺省值把旧进程或残缺 payload 伪装成兼容状态。
常驻 API 若仍运行旧代码，client 必须显示 incompatible snapshot 并要求重启控制台，而不是猜测 promotion track。

## 4. Lifecycle 与正交状态

主 lifecycle stage：

```text
NEEDS_TASK_DESIGN
AWAITING_A0
PREFLIGHT
RESEARCH_RUNNING
RESEARCH_COMPLETE
CANDIDATE_RETAINED
FORMAL_VALIDATION
AWAITING_A1
SHADOW
AWAITING_A2
ACTIVE
ROLLED_BACK
BLOCKED
```

stage 只是导航视图，不替代 authority。以下状态始终正交显示：

- Praxist maturity/frontier lane；
- formal validation PASS/FAIL；
- ModificationGate ALLOW/DENY；
- `WiringLevel.DISABLED / SHADOW / ACTIVE`；
- human approval A0/A1/A2；
- run process health。

Research completion 不自动进入 formal validation；formal PASS 不自动进入 SHADOW；SHADOW observation 不自动
进入 ACTIVE。每条边都必须有 exact artifact 与 named human review。

`research_mode=external_simulation` 时，`RESEARCH_COMPLETE` 是 Lab 轨道终点：缺 handoff 只开放
`record_external_handoff`，handoff sealed 后不开放 Candidate/A1/A2。其 authority readout 对 formal、Gate 与
wiring 显式显示 `external_domain_owned/not_applicable`，不能借用 Volvence stage 文案暗示上线权限。
公共 handoff 必须通过 request/approval/run-completion/result 四节点 hash chain 校验，且具名 A0、
`RUN_COMPLETED` 与 import-only consumer permissions 缺一即不进入 completed view。Lab 保留历史 v1 只读兼容，
但不得把 legacy artifact 当作 Foundry M5 新合同重新发布。

Praxist status 的 `completed` 是 process lifecycle readout，不等于 committed handoff。Lab 只在 completed run 的固定
`<run_dir>/volvence_handoff.json` 读取 `forge-praxist-candidate-handoff.v1`，并核对 `task_id/run_id`；缺失、schema
错误或交叉绑定不一致时只开放 inspection，不开放 Candidate import。Forge importer 仍负责完整的 run/boundary/result/
file hash 二次校验。

## 5. Product surface

### 5.1 首屏：Pipeline Board

首屏必须直接提供工作面，而不是 marketing hero：

- 顶部 stage rail：从 Forge 到 ACTIVE 的阶段计数和阻塞热区；
- 主表：任务、owner、当前阶段、运行健康、证据等级、下一 gate、更新时间；
- 右侧 inspector：exact ids/hash、证据、日志摘要、可执行动作与风险说明；
- system strip：Forge scanner、Praxist registry、formal validator、target adapter 的可用性；
- 明确区分“开发效果”“正式证据”“生产 authority”。

### 5.2 路由

- `/`：全局 Pipeline Board；
- `/discovery`：Demand 与 TopicProposal inbox；具名人类可作 exact bind/reject，不能在此批准 A0；
- `/tasks/:taskId`：一个研究任务的完整 lineage 与 gate history；
- `/runs`：Praxist + Labs runs，按 owner 分栏；
- `/approvals`：待 A0/A1/A2 的 exact review inbox；
- `/evidence`：development/formal/shadow/canary evidence 分层对比；
- `/system`：本机 doctor、registry、validator/adapter readiness。

空状态、source drift、invalid artifact、run stale、审批冲突和成功 transition 都必须有独立状态，不得以空表
或 toast 代替。

上述路由均已作为同一 `ResearchLabSnapshot` 的只读投影实现：侧栏和窄屏导航使用真实 URL，任务表行进入
`/tasks/:taskId`，全局搜索只过滤当前 frozen snapshot，不创建平行索引或新 owner。`/approvals` 仅显示当前
`available_actions` 或 lifecycle 明确要求 A0/A1/A2 人审的任务；`/runs` 只显示带 owner-published run binding 的任务；
`/evidence` 保持 development/formal/shadow/canary 四列；`/system` 原样显示各 source health 与 typed warning。
external simulation 在 rail 和 authority inspector 中把 Formal/Gate/SHADOW/ACTIVE 标为 not applicable，不能显示成
尚待解锁的 Volvence promotion gate。

## 6. Command API

本地服务默认只绑定 `127.0.0.1`。GET 可无 token；所有 mutation 必须同时满足：

- same-origin + CSRF token；
- named reviewer/operator；
- exact artifact id + file SHA-256；
- 当前 snapshot revision 未漂移；
- command 在 `available_actions` 中；
- argv 来自固定 command builder，禁止 raw shell / extra args；
- owner 返回的 artifact 重新加载、校验后才发布新 snapshot。

v1 API：

```text
GET  /api/v1/snapshot
GET  /api/v1/tasks/{task_id}
POST /api/v1/topics/bind
POST /api/v1/a0/review
POST /api/v1/reconcile
POST /api/v1/external/requests
POST /api/v1/external/handoff
POST /api/v1/candidates/import
POST /api/v1/a1/authorize-shadow
POST /api/v1/a2/authorize-active
POST /api/v1/rollback
```

`reconcile` 只能处理已经 exact-approved 的 Request。A1/A2 endpoint 只委托 Forge promotion CLI；真正的 target
adapter 仍消费 receipt 并执行自身协议。

当前实现矩阵：

| Endpoint | 状态 | Owner seam |
|---|---|---|
| `GET snapshot/task/session` | 已实现；含跨 owner exact promotion graph 与 completed handoff | portal collector/session |
| `POST topics/bind` | 已实现；fresh Demand/Proposal/registry SHA + mapping + named human | `forge research-bind-topic --json` |
| `POST a0/review` | 已实现，mutation mode 默认关闭 | `forge research-approve` |
| `POST reconcile` | 已实现，仅当 fresh snapshot 发布 `reconcile` 动作 | `forge research-reconcile --once --request ...` |
| `POST external/requests` | 已实现；registered domain root + descriptor id/file SHA + fresh revision | `forge research-submit-external --json` |
| `POST external/handoff` | 已实现；只接受 completed external Request，固定 simulation/proposal_only | `forge research-handoff-external --json` |
| `POST candidates/import` | 已实现；只接受 completed run 的 exact Task/Handoff/run id | `forge research-import-praxist` |
| `POST a1/authorize-shadow` | 已实现；首次或 fresh rollback boundary 的 exact Candidate/Validation/Gate | `forge research-authorize --to-wiring shadow` |
| `POST a2/authorize-active` | 已实现；要求 fresh Validation/Gate 与 exact previous SHADOW receipt | `forge research-authorize --to-wiring active` |
| `POST rollback` | 已实现；服务从 current receipt 机械派生唯一相邻降级目标 | `forge research-rollback` |
| `POST scan` | 后续收敛包；UI 只能展示 scanner readiness | `forge research-scan --once` 尚未接入 portal |

`GET /api/v1/session` 只向同源本地 UI 发布当前进程 CSRF token 和已启用动作。mutation 服务不接受 locator、raw
argv 或 extra args；客户端只提交 snapshot revision、对应 subject 的 exact id/hash、named actor、reason 与 typed
decision；topic binding 还必须提交 registry SHA 和 exact mapping id。服务从 fresh snapshot 反查正式 locator、重算文件
SHA，并在 action 仍可用时构造固定 argv。Forge 自身仍
二次验证 Request identity、全部 binding bytes、全局 capacity 与 reconcile lock，因此 portal 的预检不替代 owner gate。

唯一 locator 例外是 external submit 的 `descriptor_locator`：它必须是 server 启动时显式注册的
`--external-domain-root DOMAIN_ID=/absolute/root` 下的安全相对 regular file，并同时提交 descriptor id/file SHA。
客户端仍不能提交任意绝对 path、Praxist argv 或 launch override。

Promotion endpoint 同样不接受 path：import 绑定 Task SHA、Handoff SHA 与 run id；A1/A2 绑定 Task/Candidate id+SHA、
Validation/Gate SHA 以及 nullable/exact previous receipt；rollback 只绑定 current Receipt id+SHA，降级目标由 receipt 的
`shadow→disabled` 或 `active→shadow` 唯一决定。owner 成功后，Lab 必须重新 collect 并读取新 Candidate/Receipt，核对
hash chain 后才返回 `current_revision`。`research-authorize` 的合法 `BLOCKED`（CLI exit 2）只有在新负 receipt 精确出现
时才作为业务结果接受；无新 receipt 的 non-zero 仍是 owner command failure。

## 7. Local-first deployment

Lab 控制本机仓库、Praxist registry 和进程，因此 functional control plane 默认本地运行。Web build 可静态部署
为只读 demo，但 hosted UI 不获得本机 mutation token，也不能直接连接 production credentials。未来远程控制必须
新增 authenticated relay、host identity 和 scheduler lease，不能把 localhost API 暴露公网。

本地 API 默认也是 read-only；只有显式传入 `--enable-mutations` 才创建 Forge command service。mutation mode
仍只绑定 loopback，并要求显式 loopback UI Origin、进程级 CSRF、16 KiB body 上限和 exact artifact binding。

仓库根入口 `./start_research_lab.sh` 是本机进程编排器，不是研究 lifecycle owner。运行该脚本本身视为显式选择
controlled local mode；它启动 API、Web 与一个周期调用 `forge research-loop --once` 的 bounded worker，并在退出时回收
三个子进程。worker 可以发现新/变化 Demand、提交已有 APPROVE Binding、调和已有 A0 APPROVE Request；它禁止生成
Binding/A0 Approval、直接调用任意 `praxist start`、自动 import Candidate 或修改 wiring。`--no-auto-research` 或
`RESEARCH_LAB_AUTO_RESEARCH=0` 关闭 worker；`--read-only` 同时关闭 worker 和全部 POST delegation。端口已被占用时
launcher 必须拒绝启动，不能复用或覆盖未知进程。

launcher 的 Praxist host 发现顺序与 Forge registry `auto` 保持一致：共享
`FORGE_PRAXIST_EXECUTABLE`、兼容的 lab-only `RESEARCH_LAB_PRAXIST`、同级 PRAXIST checkout、`PATH`、
`~/.venvs/praxist/bin/praxist`。这允许 macOS source checkout 与 WSL user venv 共用同一仓库配置；实际传给
Lab/Forge 的仍是已验证 executable path，不新增跨 OS shell bridge。

launcher 可只读检测 sibling Foundry checkout，并注册 `foundry=<root>` ingress；这不扫描、不提交、不审批 Intent，
也不写 Foundry。未注册 root 时 external POST fail closed，现有 Volvence UI/CLI 不受影响。

## 8. 收敛包与里程碑

1. **Foundation**：冻结 Forge/Praxist pilot 与 control-plane contracts；不含 UI。
2. **Read-only Lab**：Sites web shell + `ResearchLabSnapshot` collector + local GET API；不含 mutation。
3. **A0 operations**：exact review/reconcile；只能到 Praxist research lifecycle。**已实现。**
4. **Promotion operations**：candidate import 与 A1/A2/rollback backend/Web consumer 已实现；每个 dialog 展示
   本次命令的全部 exact id/hash，合法负 receipt 单独显示 BLOCKED。缺 validator/adapter 时仍只显示 blocker。
5. **Multi-view operations**：真实任务选择、审批 inbox、run registry、evidence matrix、system health 与 task lineage
   路由；全部复用同一个 snapshot 与 command workbench。**已实现。**
6. **Demand discovery operations**：snapshot v2、`/discovery`、exact bind/reject 与 managed bounded worker。**已实现。**
7. **Remote/read-only mirror**：可选；不扩本地控制权限。

根目录 launcher 已实现；第 4 包现已接入 Forge mutation seam 与本地 Web workbench，但不会生成 formal/gate evidence，
也不会替 target adapter apply wiring。运行中的 Praxist Task 仍只显示 `view_run`，不会暴露第二次 start/reconcile。

每包独立提交、测试和回滚。共享 snapshot shape 先冻结，writer/collector 与 web consumer 分开提交。

## 9. 验证与诚实边界

- collector fixture 覆盖每个 stage、hash drift、malformed artifact、stale run 和缺 adapter；
- API 测试验证 localhost、CSRF、revision、exact hash 和 command allowlist；
- web 测试覆盖 pipeline/inspector/empty/error/success 与键盘操作；
- A0 drill 使用 fake runner；真实 Praxist 只在 exact approved Request 上单独验收；
- A1/A2 drill 默认 fixture receipt，不能用 Labs 内部 `PromotionManager` 冒充 production authorization。

Lab 完成只能声称“研究与上架生命周期可统一观察和按既有 gate 操作”，不能声称任务效果、formal PASS、
SHADOW 成功或 production ACTIVE 已成立。
