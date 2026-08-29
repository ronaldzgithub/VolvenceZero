# Research Lab：Forge → Praxist → SHADOW → ACTIVE 统一控制台

> Status: v1.1 architecture freeze；read-only aggregation + opt-in exact A0 operations implemented
> Last updated: 2026-08-29
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

Lab 不成为新的研究、验证、gate 或 wiring owner。它只做两件事：

1. 从各 owner 发布的不可变 artifact 和 Praxist targeted status 生成一个 frozen、可刷新、只读的
   `ResearchLabSnapshot`；
2. 把人类在 UI 中提交的 typed command 委托给现有 owner seam，并展示 owner 返回的新 artifact。

因此 Lab 不能用按钮直接改数据库、写 candidate、启动任意命令或切换 production wiring。UI 上的一个
“批准”动作仍必须产生与 CLI 完全相同的 exact-bound Approval/receipt。

## 2. 唯一 owner 与边界

| 数据/动作 | 唯一 owner | Lab 权限 |
|---|---|---|
| FailurePattern / Opportunity / routing | `volvence_forge.research_opportunity` | 只读；可委托一次 bounded scan |
| ResearchRequest / Approval / Event | `volvence_forge.research_control` | 只读；可提交 exact A0 review 与 reconcile |
| task resolution、run、generation、frontier | Praxist | 只读 targeted status；启动只经 approved Forge reconcile |
| candidate import / formal validation / gate receipt | Forge promotion pipeline + named validator/Gate | 只读；可委托 exact import/authorize |
| runtime wiring | target owner | 只读 receipt；Lab 不直接 mutate |
| Labs probe CAS / experimental promotions | `volvence_labs` | 保留为实验域，不冒充 production authority |
| `ResearchLabSnapshot` | `volvence_labs.portal` | 唯一发布者；web client 只读 |

本模块位于 offline development plane，不是 Brain runtime module，不进入 PE/credit，也不注册
`docs/DATA_CONTRACT.md` runtime slot。若未来把 Lab 状态接入 Brain，必须另开独立 contract package。

## 3. Canonical snapshot

`ResearchLabSnapshot` 是 frozen dataclass/JSON view，至少包含：

- `schema_version / generated_at / repo_revision / source_health`；
- `summary`：各 lifecycle stage 数量、active run 数、blocked 数、待人审数；
- `items[]`：稳定 `item_id/task_id/claim_id/owner/capability_axes`；
- `lifecycle`：当前 stage、允许的下一 stage、blocking reason、last transition；
- `bindings`：Task、Request、Approval、run、candidate、validation、gate、receipt 的 content refs；
- `run`：Praxist `run_id/state/pid/generation/findings/run_dir`，无匹配时显式 null；
- `evidence`：development、formal、shadow、canary 分层，禁止混成一个 score；
- `authority`：A0/A1/A2 与 production promotion 均逐字段显示，不从 stage 名推断；
- `available_actions[]`：由 owner contract 和 exact artifact 是否齐全机械派生；
- `warnings[]`：hash drift、source drift、stale status、缺 validator/adapter 等 fail-closed 状态。

collector 不遍历 producer 私有结构重建状态。只接受正式 JSON/schema、Praxist JSON 输出和 Labs 自有 CAS；
任何 malformed artifact 单独形成 typed warning，不能静默跳过后让 UI 显示“健康”。

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
- `/tasks/:taskId`：一个研究任务的完整 lineage 与 gate history；
- `/runs`：Praxist + Labs runs，按 owner 分栏；
- `/approvals`：待 A0/A1/A2 的 exact review inbox；
- `/evidence`：development/formal/shadow/canary evidence 分层对比；
- `/system`：本机 doctor、registry、validator/adapter readiness。

空状态、source drift、invalid artifact、run stale、审批冲突和成功 transition 都必须有独立状态，不得以空表
或 toast 代替。

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
POST /api/v1/scan
POST /api/v1/a0/review
POST /api/v1/reconcile
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
| `GET snapshot/task/session` | 已实现 | portal collector/session |
| `POST a0/review` | 已实现，mutation mode 默认关闭 | `forge research-approve` |
| `POST reconcile` | 已实现，仅当 fresh snapshot 发布 `reconcile` 动作 | `forge research-reconcile --once --request ...` |
| scan/import/A1/A2/rollback | 后续收敛包；UI 只能展示 blocker | 对应 Forge owner 尚未接入 portal |

`GET /api/v1/session` 只向同源本地 UI 发布当前进程 CSRF token 和已启用动作。mutation 服务不接受 locator、raw
argv 或 extra args；客户端只提交 snapshot revision、Task id、artifact id/hash、named actor、reason 与 typed
decision。服务从 fresh snapshot 反查正式 locator、重算文件 SHA，并在 action 仍可用时构造固定 argv。Forge 自身仍
二次验证 Request identity、全部 binding bytes、全局 capacity 与 reconcile lock，因此 portal 的预检不替代 owner gate。

## 7. Local-first deployment

Lab 控制本机仓库、Praxist registry 和进程，因此 functional control plane 默认本地运行。Web build 可静态部署
为只读 demo，但 hosted UI 不获得本机 mutation token，也不能直接连接 production credentials。未来远程控制必须
新增 authenticated relay、host identity 和 scheduler lease，不能把 localhost API 暴露公网。

本地 API 默认也是 read-only；只有显式传入 `--enable-mutations` 才创建 Forge command service。mutation mode
仍只绑定 loopback，并要求显式 loopback UI Origin、进程级 CSRF、16 KiB body 上限和 exact artifact binding。

仓库根入口 `./start_research_lab.sh` 是本机进程编排器，不是研究 lifecycle owner。运行该脚本本身视为显式选择
controlled local mode；它只启动 API 与 Web、检查端口和依赖、在退出时回收两个子进程。它禁止调用
`praxist start`、自动生成 Approval、自动 reconcile 或直接修改 wiring。`--read-only` 会关闭全部 POST delegation；端口已被
占用时 launcher 必须拒绝启动，不能复用或覆盖未知进程。

## 8. 收敛包与里程碑

1. **Foundation**：冻结 Forge/Praxist pilot 与 control-plane contracts；不含 UI。
2. **Read-only Lab**：Sites web shell + `ResearchLabSnapshot` collector + local GET API；不含 mutation。
3. **A0 operations**：exact review/reconcile；只能到 Praxist research lifecycle。**已实现。**
4. **Promotion operations**：candidate/formal/A1/A2/rollback UI；缺 validator/adapter 时只显示 blocker。
5. **Remote/read-only mirror**：可选；不扩本地控制权限。

根目录 launcher 已实现，但不改变第 4 包仍未接入 owner mutation seam 的事实。

每包独立提交、测试和回滚。共享 snapshot shape 先冻结，writer/collector 与 web consumer 分开提交。

## 9. 验证与诚实边界

- collector fixture 覆盖每个 stage、hash drift、malformed artifact、stale run 和缺 adapter；
- API 测试验证 localhost、CSRF、revision、exact hash 和 command allowlist；
- web 测试覆盖 pipeline/inspector/empty/error/success 与键盘操作；
- A0 drill 使用 fake runner；真实 Praxist 只在 exact approved Request 上单独验收；
- A1/A2 drill 默认 fixture receipt，不能用 Labs 内部 `PromotionManager` 冒充 production authorization。

Lab 完成只能声称“研究与上架生命周期可统一观察和按既有 gate 操作”，不能声称任务效果、formal PASS、
SHADOW 成功或 production ACTIVE 已成立。
