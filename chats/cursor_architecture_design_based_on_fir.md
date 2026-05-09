# Architecture design based on first principles
_Exported on 2026/5/10 at GMT+8 0:21:43 from Cursor (3.3.27)_

---

**User**

@/Users/mengfu/Documents/GitHub/EmoGPT/docs/api/DLAAS_README.md 看看现在的架构如果要实现我这个怎么做 ？ 还差多少， 一定要从第一性原理出发去设计， 并且ssot。 

---

**Cursor**

我从第一性原理出发，按 R8（快照 SSOT + 单一所有者）+ R2（冻结基底）+ R4（控制不在 token 空间）三条铁律，把 DLaaS 这套 API 与现在 VZ 的代码做了一遍逐项对位。结论先放在最前面：

> **VZ 已经把 DLaaS 的"运行时认知"那一半做完了；缺的全部在"控制平面 + 多租户资源治理 + 多渠道 typed envelope"这一半，而这一半在 R8 下天然不属于 `vz-*` 内核，应该作为一组新的 `dlaas-platform-*` wheel 落在 `lifeform-service` 之上。**

下面是分层论证。

---

## 1. 第一性原理：DLaaS 的两层本质

读 `/Users/mengfu/Documents/GitHub/EmoGPT/docs/api/DLAAS_README.md`，DLaaS 在自我描述里就把自己劈成了两个平面（README 第 19–26 行）：

| 平面 | 关心什么 | 时间尺度 | VZ 类比 |
|---|---|---|---|
| **Control plane** | 谁是租户、什么形态的 shell、哪些 asset、Template/Version 生命周期、Contract、Adoption、Exam、Launch License | 低频 / 可审计 / 强一致 | **不存在**——是治理层 |
| **Runtime plane** | 一通对话发生了什么：chat / observe / feedback / teach / task / report / command + 流式输出 + handoff | 每 turn / 每 session / 每 scene | **VZ 的 lifeform-service + 内核** |

把这两个平面套到 R2 / R8 上看，会发现：

- **Control plane 的所有概念都是"治理资源"**：tenant / shell / template / contract / asset / focus_person / identity_link / handoff_ticket / exam / license。它们的特征是：**强 schema、版本化、由人审核、改动需要可回滚证据**。R8 要求每个运行时区域有唯一主所有者；治理资源天然不属于 `vz-cognition`（不是认知状态）、也不属于 `lifeform-*`（不是产品适配）——它们是更高一层的"**编排基底**"。
- **Runtime plane 的所有概念都已经在 VZ 内核里有归属**：chat / observe / feedback / teach / task / report 全是已有 `BrainSession.submit_*` 系列方法或 `run_turn(trigger_kind=…)` 的 typed 包装；OutputAct 是 `AgentResponse + rationale_tags + AffordanceRegistry` 的运输层皮肤。

**结论**：不要把 DLaaS 的 control plane 概念塞进任何 `vz-*` wheel——那会立刻违反 R8（让 `vz-cognition` 里冒出一个 "tenant_owner"）和 R2（治理元数据通过 fine-tune 反向写到基底）。正确做法是新增一组 wheel，让它们**只读 kernel snapshot、只调用 lifeform facade**。

---

## 2. VZ ↔ DLaaS 现状覆盖矩阵

我把 README 里的每一组 endpoint 与 VZ 现有代码对照一遍。"H = 已有"，"P = 部分有"，"M = 缺"。

### 2.1 控制平面（README §11–§12）

| DLaaS 资源 | 状态 | VZ 现有锚点 |
|---|---|---|
| `Tenant`（api_key/secret/quota/billing） | **M** | 完全没有 |
| `Shell`（kind/type/embodiment.perception/expression/action/constraints/channel） | **M** | `lifeform-affordance` 4 Kind（Tool/Action/Organ/Shell）的描述符 schema 已就位（`docs/specs/affordance.md`），但**没有"按 tenant 注册一个 deployment Shell"的运行时注册表**；现在 shell 是隐含在 vertical 里的 |
| `Asset`（uri/type/mime/lang/source_meta） | **P** | `lifeform-ingestion.IngestionEnvelope` 是 in-process 一次性容器，没有**租户级 asset registry / asset_id / 复用** |
| `Template` + `template_version` + `runtime_template_id` | **P** | `lifeform-domain-emogpt` / `lifeform-domain-coding` 是硬编码 vertical；DLaaS 的 `runtime_template_id`（如 `dongfang_growth_advisor__job_seed_v1`）天然映射到 VZ 的 vertical 名，但缺少：(a) 版本发布生命周期 draft/published/deprecated；(b) tenant 级 persona_spec / seed_config 覆盖；(c) Template ↔ Asset 链接表 |
| `activate / readiness` | **P** | `build_companion_lifeform()` 已经把 `DomainExperiencePackage + VitalsBootstrap + temporal/regime bootstrap` 装配出来，相当于"activate"；但**没有"readiness gate"读出 world_nodes/self_nodes/l2_cards 来做 publish 准入** |
| `Studio session`（训练工作区） | **P** | `lifeform-evolution.closed_alpha_preflight` + `lifeform-thinking.evolution_loop` 有"非生产训练通道"的雏形，但没有"studio shell vs deployment shell"两态分离 |
| `Contract`（lifecycle: created/active/paused/...） | **M** | 完全没有 |
| `Adopt → ai_id`（产出 instance_endpoint / instance_token） | **M** | 现在每个 `lifeform-service` 进程**只能托管一个 vertical**（`--vertical` 选定），没有"一个进程多 ai_id"的概念；session 层 ≠ instance 层 |
| `engine_tools / tool_policy_snapshot` | **P** | `lifeform-affordance.registry / safety_model` 已经能做"调用前 safety gate"，但没有 contract→工具白名单的运行时注入 |
| `focus_persons`（per-instance 持久 person 注册） | **P** | spec `docs/specs/social_cognition/01_multi_party_identity.md` 已明确"per-interlocutor keying + MemoryEntry subject/audience scope"；`submit_profile_event` 是写入路径——但**没有"person 是 ai_id 一级公民"的注册表与 attention_priority 快照** |
| `identity_links`（canonical_end_user_ref ↔ channel_ref） | **P** | `volvence_zero.memory.UserIdentity(scope_key)` 已有 user-scoped memory，但没有"一个 canonical user 跨 wechat/web 多 channel"的映射表 |
| `Audience analysis / Exam questions / Exam runs / Launch license` | **M** | `lifeform-evolution` 有 scenario / closed-alpha preflight 框架，可以借作 exam runner 底座，但**没有 license gate 与 publish 联动** |
| `Handoff queue / tickets / human_reply` | **P** | `rupture_state` owner + `dialogue_external_outcome` slot 已就位（`docs/specs/rupture-and-repair.md`），是**最自然的 handoff 触发信号源**；缺外部 ticket CRUD + operator workflow |

### 2.2 运行时平面（README §9–§10、§13）

| DLaaS 端点 | 状态 | VZ 现有锚点 |
|---|---|---|
| `POST /dlaas/instances/{ai_id}/interactions` | **P** | 现有 `POST /v1/sessions/{sid}/turns` 只支持 `chat`，envelope 里**没有** `interaction_type / mode / structured_context / output_contract / target_person_ids / feedback` 字段 |
| `interaction_type=chat` | **H** | `LifeformSession.run_turn(trigger_kind=USER_INPUT)` |
| `interaction_type=observe` | **P** | `lifeform-ingestion.IngestionEnvelope` (FORCED) + `BrainSession.submit_semantic_events / submit_tool_result / submit_profile_event / submit_task_event / submit_reviewed_knowledge_event` 都已存在，**只是没有公开 HTTP 出口** |
| `interaction_type=feedback` | **H** | `POST /v1/sessions/{sid}/dialogue-outcomes` + `submit_dialogue_outcome(kind=…)` typed enum 完全可用（`DialogueExternalOutcomeKind`） |
| `interaction_type=teach / task` | **P** | 内核已有 `TurnTriggerKind.APPRENTICE` + vitals apprentice override（`packages/lifeform-core/.../lifeform.py` 850–930 行），**只缺把 trigger_kind 暴露到 HTTP envelope** |
| `interaction_type=report` | **M** | 没有"按周/按 person 生成结构化报告"的 owner——但这本来就该是 reflection 输出的 readout，不是新 owner |
| `interaction_type=command`（如 `refresh_person_context`） | **M** | 没有；命令应该是路由到 `submit_profile_event` / `end_scene` / `pause` 等已有动作的薄包装 |
| `OutputAct{act_type/capability/payload/degraded}` | **P** | `volvence_zero.agent.response.AgentResponse` + `rationale_tags` + `lifeform-affordance.renderers.{catalog_json,compact_list,markdown,openai_tools}` 已经有结构化输出能力，但 wire format 不是 DLaaS 这套 |
| SSE 流（`event: ack/act/chunk/done`） | **M** | aiohttp 服务现在用同步 JSON 响应；substrate `.generate(...)` 是 sync block；流式需要 substrate 加 streaming + service 加 SSE writer |
| `operator_takeover` placeholder（pause 时不调内核） | **M** | 现有 `_handle_pause_session` 是 no-op stub；DLaaS 需要"paused 时返回 placeholder 而不是 hit kernel" |
| Admin SSE `/dlaas/admin/ops/conversations/stream` | **M** | 完全没有，但只是平台层广播 |

---

## 3. SSOT 兼容的边界设计

下面是我建议的 wheel 切分。**核心约束：所有新增 wheel 只读 kernel snapshot，不持有任何 cognitive owner，不反向 import `vz-*` 的子模块**（已被 `tests/contracts/test_import_boundaries.py` 强制）。

```
┌──────────────────────────────────────────────────────────────────────────┐
│  dlaas-platform-*  (NEW; 控制平面 + 多租户编排，不是认知 owner)         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  dlaas-platform-contracts                                                  │
│    Tenant / Shell / Asset / Template / TemplateVersion / Contract /        │
│    FocusPerson / IdentityLink / HandoffTicket / ExamQuestion / ExamRun /   │
│    LaunchLicense  ——  全部 frozen dataclass + JSON schema                 │
│    （沿用 R8：immutable + dataclasses.replace；禁止 deepcopy）             │
│                                                                            │
│  dlaas-platform-registry                                                   │
│    SQLite/Postgres backed CRUD + auth (X-Tenant-Api-Key /                  │
│    X-Control-Plane-Secret / X-Service-Secret)                              │
│    + quota / rate-limit middleware                                         │
│    （这是 control plane 的唯一 owner — SSOT）                              │
│                                                                            │
│  dlaas-platform-launcher                                                   │
│    contract_id → Lifeform 实例（一个进程多 ai_id 共享 substrate）          │
│    awake/sleep/LRU eviction at ai_id 粒度                                  │
│    instance_endpoint / instance_token 颁发                                 │
│                                                                            │
│  dlaas-platform-api                                                        │
│    FastAPI / aiohttp router for /dlaas/*：                                 │
│      - control plane 直接读写 registry                                     │
│      - runtime plane 把 typed InteractionEnvelope 翻译成 lifeform 调用     │
│      - SSE writer（act / chunk / done）                                    │
│                                                                            │
│  dlaas-platform-ops                                                        │
│    pause / resume / operator-message / handoff queue / SSE ops stream      │
│    handoff trigger = 读 vz-cognition rupture_state 快照                    │
│                                                                            │
│  dlaas-platform-eval                                                       │
│    Audience profile / Exam runner / Launch license gate                    │
│    复用 lifeform-evolution.closed_alpha_preflight 作为骨架                 │
│    LLM judge **仅 readout**（遵守 EVO-2 / R12）                             │
│                                                                            │
└────────────────────┬─────────────────────────────────────────────────────┘
                     │  ① factory(template, contract, focus_persons)
                     │  ② 转发 typed InteractionEnvelope → 6 个已有 lifeform 入口
                     │  ③ 读 kernel snapshot 做 ops 决策
                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  lifeform-service  (扩展：typed InteractionEnvelope 路由)                 │
│                                                                            │
│   InteractionEnvelope = {                                                  │
│     contract_id, session_id, end_user_ref, target_person_ids,              │
│     interaction_type ∈ {chat,observe,feedback,teach,task,report,command},  │
│     mode ∈ {live, apprentice},                                             │
│     human_brief, structured_context, output_contract, feedback, lang       │
│   }                                                                        │
│                                                                            │
│   路由表（这是新增的薄编排，没有新 owner）：                               │
│     chat       → run_turn(USER_INPUT)                                      │
│     observe    → IngestionPipeline.run() OR submit_semantic_events /        │
│                  submit_profile_event / submit_task_event /                │
│                  submit_reviewed_knowledge_event（按 observation_type 分） │
│     feedback   → submit_dialogue_outcome(kind=…)                           │
│     teach/task → run_turn(trigger_kind=APPRENTICE)                         │
│     report     → run_turn + drain_session_post_slow_loop + reflection      │
│                  snapshot 投影                                             │
│     command    → 路由到已有动作（end_scene / pause / submit_profile_event/ │
│                  refresh persons / awake）                                 │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  vz-runtime / vz-cognition / vz-substrate / vz-memory / vz-temporal      │
│  / vz-application / vz-contracts                                          │
│  ——————— 不动。继续保持 R8 单 owner + snapshot SSOT ——————        │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.1 关键不变量（必须写进新 spec）

1. **Control-plane SSOT**：DLaaS 所有资源（tenant/shell/template/contract/...）的唯一所有者是 `dlaas-platform-registry`，运行时其它模块只读它发布的快照。
2. **kernel 隔离**：`dlaas-platform-*` 不允许 import `volvence_zero.cognition.*` / `volvence_zero.memory.*` / `volvence_zero.substrate.*` 内部；只能通过 `vz-contracts` 的 `Snapshot`、`lifeform-core.Lifeform` facade、`lifeform-service.app` HTTP 入口与内核交互。
3. **Template ↔ Vertical 映射不重写 vertical**：DLaaS `runtime_template_id` 只能解析到一个**已注册的 VZ vertical builder**（`lifeform_service.verticals.discover_verticals()`）+ tenant 级 `DomainExperiencePackage` overlay；**禁止**让 platform 持有 `domain_knowledge` / `case_memory` 的第二副本。
4. **focus_persons → social cognition**：`focus_persons` 的写入只通过 `BrainSession.submit_profile_event`（已存在），platform 自己只持有"哪些 person 属于这个 ai_id"的索引，person 的 belief / preference / role 状态仍由 `vz-cognition.social_cognition.*` 单独 owner。
5. **handoff/operator-takeover 路径**：handoff 触发由平台**读** `rupture_state` 快照决定（spec `docs/specs/rupture-and-repair.md` 已有 `RuptureKind` typed vocabulary）；handoff ticket 由平台维护；当 `paused=true` 时 `interactions` 短路返回 placeholder，**不进入 kernel**——这与 DLaaS README §10 行为一致，且天然兼容 R8（kernel 不会感知 ops 状态）。
6. **OutputAct 是运输层皮肤**：`OutputAct{act_type, capability, payload, degraded, original_capability}` 由 `dlaas-platform-api` 在 lifeform 返回的 `AgentResponse + rationale_tags + AffordanceRegistry` 之上生成；**禁止**让内核去理解 DLaaS wire format。
7. **eval / license gate ≠ 学习源**：Exam Runs + Launch License + Audience analysis 全程是 `R12 readout`——LLM judge 只产出"自然性 / 简洁性"等 readout（EVO-2），**禁止**反向写回 reward / Face 梯度（呼应 OA-1 / OA-3 / R-PE）。
8. **每个新 wheel 在 DATA_CONTRACT.md 注册自己的 slot**（`tenant_state` / `contract_state` / `instance_status` / `handoff_ticket_state`），并标注 `WiringLevel = SHADOW → ACTIVE`；不绕过 `archetecture.md` migration rule。

### 3.2 新增 / 复用映射表（实际工程量）

| DLaaS 概念 | 落到哪个 wheel | 工程动作 |
|---|---|---|
| Tenant + auth | `dlaas-platform-registry` + `dlaas-platform-api` 中间件 | **新建**：CRUD + key/secret/quota |
| Shell（embodiment 描述符） | `dlaas-platform-contracts` + `lifeform-affordance.AffordanceRegistry` | shell.embodiment.{perception,expression,action} 直接映射到现有 **4 Kind affordance descriptor** —— 复用率最高的一处 |
| Asset | `dlaas-platform-registry` + `lifeform-ingestion.IngestionEnvelope` | 把 asset_id 作为 envelope provenance；activate 时按 asset_id 拉 URI 喂 IngestionPipeline |
| Template + Version | `dlaas-platform-registry` | 元数据；`runtime_template_id` 必须命中已注册 vertical |
| Activate | `dlaas-platform-api` 调用 lifeform 的 ingestion + drain slow loop | 复用现有 `LifeformSession.end_scene(drain_slow_loop=True)` 来强制 R6 沉淀；readout 现有 memory snapshot 算 world/self/l2 nodes |
| Readiness gate | `dlaas-platform-eval` | 读 memory / regime / temporal snapshot 阈值 |
| Contract | `dlaas-platform-registry` | 资源记录 |
| Adopt → ai_id | `dlaas-platform-launcher` | 进程内多实例编排：`{ai_id → Lifeform}` 字典 + 共享 `OpenWeightResidualRuntime`（**和 lifeform-service 现有 `--substrate-mode hf-shared` 是同一原理，只是把 session 粒度上升到 ai_id 粒度**） |
| Tool policy | `dlaas-platform-launcher` 注入 `AffordanceRegistry` 白名单 | 复用 `lifeform-affordance.scorer + safety_model` |
| Focus persons | `dlaas-platform-registry` 维护 ai_id↔person_id 索引；写入 → `submit_profile_event` | 复用 social_cognition 多人身份 spec |
| Identity links | `dlaas-platform-registry` | canonical_end_user_ref → `UserIdentity.scope_key` |
| `interactions` typed envelope | `lifeform-service.app` 扩 router + DTO | **核心改动**：把现有 `_handle_turn` 升级为 `_handle_interaction`，按 `interaction_type` switch |
| `observe` 多形态 | `lifeform-service` + `lifeform-ingestion` | 把 `observation_type` 路由到对应 `submit_*_event` 方法 |
| `report` | `lifeform-service` + `lifeform-thinking` (drain 反思) | 强制走 `end_scene(drain_slow_loop=True)` 然后从 reflection snapshot 投影 |
| `command` | `lifeform-service` | 显式动作白名单（refresh_person / end_scene / pause），**不**用关键词匹配 |
| OutputAct + SSE | `dlaas-platform-api` 包装层 + `vz-substrate` 加 streaming | substrate streaming 是可选优化，先用整段 `act` 事件即可 |
| Pause / Resume / Operator-message | `dlaas-platform-ops` 维护"per-session pause 状态" | paused 时短路；operator-message + `inject_into_runtime=true` → `interaction_type=teach, mode=apprentice` |
| Handoff queue | `dlaas-platform-ops` 读 `rupture_state` 快照 | 已有 `RuptureKind` + `dialogue_external_outcome` 是触发信号；ticket 是平台 CRUD |
| Audience / Exam / License | `dlaas-platform-eval` | 复用 `lifeform-evolution.closed_alpha_preflight` 框架 |
| Admin SSE `/dlaas/admin/ops/conversations/stream` | `dlaas-platform-ops` | 平台层广播 turn / pause 事件，订阅源是平台自己写的 ledger |

---

## 4. 还差多少（粗估）

按"能复用 vs 必须新建"切量级：

| 模块 | 量级 | 备注 |
|---|---|---|
| `dlaas-platform-contracts` (dataclass + schema) | **S，1–2 PR** | 纯 typed schema |
| `dlaas-platform-registry` (CRUD + auth + SQLite) | **M，4–6 PR** | 多租户 / quota / 三种 auth 中间件 |
| `dlaas-platform-launcher` (一进程多 ai_id) | **M-L，6–10 PR** | 把现有 SessionManager 上一层抽出 InstanceManager |
| `lifeform-service` envelope 升级（chat/observe/feedback/teach/task/report/command 路由） | **M，4–6 PR** | 全是已有内核入口的薄包装 |
| OutputAct + SSE 流式 | **M，3–5 PR** | substrate streaming 是可选 |
| `dlaas-platform-ops` (pause/handoff/SSE ops) | **M，4–6 PR** | 读 `rupture_state` + 平台 CRUD |
| `dlaas-platform-eval` (audience/exam/license) | **M-L，6–8 PR** | 复用 closed_alpha_preflight |
| Template/Activate/Readiness 生命周期 + bootstrap 注册 | **M，4–5 PR** | runtime_template_id 注册表 + readiness 读 snapshot |
| Identity links / focus_persons 注册 | **S-M，2–3 PR** | 大部分能力在 social_cognition spec 里已就位 |

合计 **约 35–55 PR / 6 个 wheel / 3–4 个月**，前提是不破 R2 / R4 / R8，且按 `WiringLevel = SHADOW → ACTIVE` 三态迁移落地。

---

## 5. 落地路线（3 波，与 `探索方向.md` 推进顺序自洽）

> 每条改动按 `.cursor/rules/cursor-convergence-workflow.mdc` convergence packet 走：现状核查 → spec 起草 → SHADOW → 评估证据 → ACTIVE。

### 第 1 波（最小可演示 DLaaS-shape，2–3 周）
**目标**：在不动内核的前提下，把现有 `lifeform-service` 升级到能"长得像 DLaaS"。

1. 新增 `dlaas-platform-contracts` wheel，落 typed dataclass：`Tenant / Shell / Template / TemplateVersion / Contract / FocusPerson / IdentityLink / InteractionEnvelope / OutputAct`。
2. `lifeform-service` 加 `POST /dlaas/instances/{ai_id}/interactions` 端点（**一个进程一个 ai_id 也行**），接受 typed envelope，按 `interaction_type` switch 到现有 `run_turn / submit_*_event / submit_dialogue_outcome / end_scene`。**这个改动 0 新 owner，纯路由**。
3. 在 `docs/specs/` 下新增 `dlaas-platform.md` spec，写明上文§3.1 的 8 条不变量。
4. 把现有 `--alpha-users-file` 那套升级为最小 tenant 模型（in-memory 也行）。

### 第 2 波（多租户 + 多实例，4–8 周）
**目标**：一个 Python 进程托管多个 `ai_id`，shared substrate，control-plane 持久化。

1. 新增 `dlaas-platform-registry`（SQLite + 三种 auth），把 Tenant/Shell/Asset/Template/Contract CRUD 端点接上。
2. 新增 `dlaas-platform-launcher`：把 `SessionManager` 抽出 `InstanceManager`（管 `{ai_id → Lifeform}`），`SessionManager` 降为每个 instance 内部组件。`Adopt` → 创建 contract → 实例化 Lifeform → 注册 focus_persons → 颁发 instance_token。
3. **Activate / Readiness** 走 `lifeform-ingestion.IngestionPipeline + end_scene(drain_slow_loop=True)`，readiness 读 memory / regime snapshot。
4. **focus_persons / identity_links** 落地，写入路径只走现有 `submit_profile_event` + `UserIdentity.scope_key`。

### 第 3 波（Ops / Streaming / Eval gate，6–10 周）
**目标**：handoff、license gate、SSE 流。

1. `dlaas-platform-ops`：pause/resume/operator-message/handoff_queue + admin SSE。**关键不变量**：paused 时短路不进 kernel；handoff 触发由读 `rupture_state` 快照得出（不在 kernel 加 owner）。
2. SSE 流：`vz-substrate` 加可选 streaming `generate_async`；`lifeform-service` 加 SSE writer，把 `AgentResponse` 拆成 `event: act / chunk / done`。
3. `dlaas-platform-eval`：复用 `lifeform-evolution.closed_alpha_preflight` 做 ExamRun runner + LaunchLicense gate；**严格遵守 EVO-2 + OA-1**：LLM judge 只 readout，license 阻止 publish 但不反向写 kernel。

---

## 6. 反模式清单（写在 spec 第 0 节）

这几条触线必须直接拒绝 PR：

1. ❌ 在 `vz-cognition` 里加 `tenant_owner` / `contract_owner` / `handoff_owner` —— 是治理状态，不是认知状态。
2. ❌ 让 platform layer 直接 import `volvence_zero.memory.MemoryStore` 然后写 person profile —— 必须走 `submit_profile_event`。
3. ❌ DLaaS `interaction_type` 用关键词匹配解析 `human_brief` —— 违反 `.cursor/rules/no-keyword-matching-hacks.mdc`；只允许按 typed enum dispatch。
4. ❌ Exam runner 把 LLM judge 分数反向喂回 reward / fine-tune —— 违反 R12 / OA-1 / EVO-2。
5. ❌ Adopt 时把 tenant 自定义 `persona_spec` 直接拼成 prompt 注入 expression layer —— 违反 R4 / OA-2 Mind/Face 隔离；persona 必须经 `DomainExperiencePackage` 编译进 `domain_knowledge` 等已有 owner。
6. ❌ `awake/{ai_id}` 端点对未配额 / 未付费 tenant 直接拉起 substrate —— quota check 必须在 launcher 入口。
7. ❌ 一个 `ai_id` 跨进程共享 `OpenWeightResidualRuntime` 时允许 `supports_live_substrate_mutation=True` —— `_enforce_frozen_for_sharing` 已在 `lifeform-service.app` 第 160–175 行设了护栏，必须继承到 launcher 层。

---

## 7. TL;DR

- DLaaS 的 **runtime plane**（chat/observe/feedback/teach/task/report/command + OutputAct + handoff 信号）VZ 的内核**已经全部具备 owner**，只缺一层 typed envelope 翻译器。
- DLaaS 的 **control plane**（tenant/shell/template/contract/asset/exam/license）VZ **完全没有**，但按 R8 它**也不该有**——它属于一组新的 `dlaas-platform-*` wheel，建在 `lifeform-service` 之上，只读 kernel snapshot、只调 `Lifeform` facade。
- 把这套切干净，DLaaS = `lifeform-service` 的多租户化 + 6 个新平台 wheel + 1 处 envelope 升级，**整个 `vz-*` 内核可以一行不动**。这恰好是 R2 / R4 / R8 / R-PE 想要的结果：治理与认知分轴，互不污染。

如果你想推进这一步，我建议第一个 convergence packet 就从"§5 第 1 波第 2 步：`POST /dlaas/instances/{ai_id}/interactions` typed envelope 路由"开始——这是收益最高、风险最低、能立即跑起来的最小切片，也是后面所有平台 wheel 的接入点。

---

**User**

完全不影响我vz的逻辑对不对。 

---

**Cursor**

是的，**就 R8 + R2 + R4 三条铁律保护的"内核认知逻辑"而言，影响面 = 0**。但我想把这句话拆得更精确，免得后面 PR 进来时才发现哪里其实碰到了——

## 严格意义上的"vz 内核"零改动清单

下面这些 wheel **可以一行代码不动**就支撑整个 DLaaS：

| Wheel | 是否需要动 | 说明 |
|---|---|---|
| `vz-contracts` | ❌ 不动 | DLaaS 的 typed dataclass 都落在新 wheel `dlaas-platform-contracts` |
| `vz-cognition`（PE / credit / dual_track / regime / semantic_state / rupture_state / social / evaluation / reflection） | ❌ 不动 | 所有 cognitive owner 保持单一所有者 |
| `vz-memory` | ❌ 不动 | `UserIdentity.scope_key` 是字符串，平台层直接拼 `f"{tenant_id}/{ai_id}/{user_id}"` 即可，不动 schema |
| `vz-temporal` | ❌ 不动 | metacontroller / β_t / z_t 完全屏蔽于平台 |
| `vz-application` | ❌ 不动 | DomainExperiencePackage 由 vertical wheel 编译，不在平台层重写 |
| `vz-substrate` | **⚠️ 可选** | Phase 1 不动；只有想做"真流式 SSE chunk"时才加一个 additive 的 `generate_async`，**仍然是新增方法不是改逻辑** |
| `vz-runtime` | ❌ 不动 | Brain facade 不变 |

## 严格意义上的"lifeform 既有 wheel"改动清单

| Wheel | 是否需要动 | 说明 |
|---|---|---|
| `lifeform-core` | ❌ 不动 | `LifeformSession` 已经暴露了 chat/observe/feedback/teach/task/report/command 全部需要的入口（`run_turn` / `submit_*_event` / `submit_dialogue_outcome` / `end_scene` / `advance_tick`） |
| `lifeform-ingestion` | ❌ 不动 | `IngestionEnvelope` schema 直接复用 |
| `lifeform-affordance` | ❌ 不动 | shell.embodiment → 现有 4 Kind 描述符；平台层 per-`ai_id` 实例化一个 `AffordanceRegistry` 就够（registry 已支持多实例化） |
| `lifeform-domain-emogpt` / `-coding` / `-character` | ❌ 不动 | 它们就是 DLaaS 里的 `runtime_template_id` 候选 |
| `lifeform-thinking` / `lifeform-evolution` | ❌ 不动 | reflection / closed-alpha preflight 直接被平台层调用 |
| `lifeform-service` | **✓ 必须动** | 加新端点 `POST /dlaas/instances/{ai_id}/interactions`（typed envelope dispatch），但**不删现有 `/v1/sessions/...`**，老端点保留作向后兼容 |

`lifeform-service` 这一步**不是改逻辑**——它是加一条新路由，把 typed envelope 翻译成现有 `LifeformSession` 方法调用。本质上是 §3 那张图里的"路由表"那六行翻译。

## 三个老实要 flag 的"边角"

如果将来 DLaaS 做深，下面这三处可能会蹭到 vz-* 边——但**都不是逻辑变更，是契约扩面**，且都可以通过快照丰富而非新 owner 解决：

1. **`report` interaction_type 想要"按周/按 person 的结构化报告"**
   - 朴素做法：drain 反思后从现有 reflection snapshot 投影。**0 改动。**
   - 进阶做法：希望反思 snapshot 暴露一个 `report_view` 字段——这是 `vz-cognition.reflection` snapshot 的 **enrichment**（按 §SSOT 规则"消费者要新信息 → 在发布模块内部丰富快照"），不是新 owner，不是新逻辑。

2. **`focus_persons.attention_priority` 字段想从内核读出**
   - 该信号本来就该归 `vz-cognition.social_cognition`（spec `01_multi_party_identity.md`）。如果 social cognition snapshot 里还没有 `attention_priority` 字段，这是同一个 owner 的 snapshot enrichment，不破 SSOT。

3. **多 tenant 共享一个 substrate runtime 时的 quota / rate-limit**
   - `lifeform-service.app._enforce_frozen_for_sharing` 已经在 R2 边界做了护栏；多 tenant 的 quota check 在 `dlaas-platform-launcher` 入口做就行，**不进 substrate**。

## 一句话回答

**vz 内核（vz-cognition / vz-memory / vz-temporal / vz-substrate / vz-application / vz-runtime / vz-contracts）和所有 lifeform-* 适配 wheel 都不动**；唯一必须动的是 `lifeform-service` 加一条 typed envelope 路由（加法而非改法），其它全部是新增 6 个 `dlaas-platform-*` wheel。这正是 R8 + R2 + R4 想要的结果——**治理层与认知层在 wheel 边界上彻底分轴**，DLaaS 演进不会反向污染 PE / 控制器 / 记忆 / 基底任何一处。

---

**User**

这么大的一个事情， 我应该怎么实施？ 

---

**Cursor**

这种量级的事情有一条铁律：**永远保持系统可演示、可回滚、可解释**。下面是按 VZ 现有工作流（convergence packet + WiringLevel 三态 + spec 同步）实际可执行的施工方案。

让我先把相关的工作流规则读一下，确保推进方式与你现有的节奏一致。

下面是按你已有的 convergence-packet + WiringLevel 三态 + SPLIT.md 三层切分（kernel / lifeform / 现在新加的 platform）的纪律来组织的施工方案。**核心心法只有三条**，剩下都是节奏。

---

## 心法（行动前必须内化）

1. **永远保持系统当前可演示**。任何时候 `git checkout main` 都能 `lifeform-serve` 跑起来；DLaaS 化的端点永远是**新加的**，老 `/v1/sessions/...` 端点只在最后阶段才 alias 弃用。
2. **垂直切片优先于横向铺面**。第一个 PR 的目标不是"把 6 个新 wheel 都搭出来"，而是**只跑通 1 个 tenant × 1 个 contract × 1 个 ai_id × 1 个 chat turn 的端到端 DLaaS 形状**——这个最小切片同时触碰所有 wheel 的最薄一层，证明边界设计成立。
3. **Spec 先于 wheel，wheel 先于代码**。按 `.cursor/rules/first-principles-not-patches.mdc` 的 spec sync gate：**先写 `docs/specs/dlaas-platform.md` + 新增条目到 `docs/specs/00_INDEX.md` + `docs/DATA_CONTRACT.md`**，再开 wheel 骨架，再填代码。倒过来一定返工。

---

## 第 0 步：方向 packet（**不写代码，1–2 天**）

只产出文档。这个 packet 的目的是把架构对齐落锤，避免后面 5 个 packet 跑歪。

**产出物**：
1. `docs/specs/dlaas-platform.md` — 落上一轮对话里 §3.1 的 8 条不变量 + wheel 切分图 + 三层"治理 / 适配 / 认知"分轴说明。
2. `docs/specs/00_INDEX.md` 增加 "§19. DLaaS Platform Layer" 入口。
3. `docs/DATA_CONTRACT.md` 增加新 slot 占位：`tenant_state` / `contract_state` / `instance_status` / `handoff_ticket_state`，全部标 `WiringLevel = DISABLED`，标注 owner = `dlaas-platform-registry`。
4. 更新 `archetecture.md` 的 wheel 表，加入 6 个 `dlaas-platform-*` 占位（`Owns` / `Boundary rule` 两列填好；`pyproject.toml` 还不需要落地）。
5. 更新 `SPLIT.md`：把 platform 视为**第三层 wheel 前缀**，明确 `dlaas-platform-* 不能 import lifeform-* 的 vertical 适配 internals，只能通过 lifeform-service facade + vz-contracts 的 snapshot`——这条要进 `tests/contracts/test_import_boundaries.py`。
6. 在 `docs/moving forward/` 下新建 `dlaas-platform-rollout.md`，把下面"切片路线"落成你自己的施工日历。

**Done 检查**：
- [ ] 读这几个文档的人能在 30 分钟内回答"为什么 platform 不在 vz-cognition 里"。
- [ ] 读 spec 的人能立刻指出违反不变量的 PR。
- [ ] 你自己能解释清楚 platform 的每个 owner 对应的退出条件 / 回滚方案。

---

## 切片 1：MVP 端到端骨架（**1 个 packet，3–8 文件，~1 周**）

> **唯一目标**：让一条 `POST /dlaas/instances/{ai_id}/interactions` with `interaction_type=chat` 的请求穿过整个新架构，证明边界设计成立。**所有控制面都用硬编码兜底**。

**包含**：
- 新建 `packages/dlaas-platform-contracts/` 只放一个 frozen dataclass `InteractionEnvelope`（chat 字段子集）+ `OutputAct`。
- 新建 `packages/dlaas-platform-api/` 一个 aiohttp 路由 `POST /dlaas/instances/{ai_id}/interactions`，硬编码 `ai_id` → 现有 `lifeform-service` 的 SessionManager。
- `packages/lifeform-service/src/lifeform_service/app.py` 新增 `_handle_dlaas_interaction` handler（**老 `_handle_turn` 不动**）。
- 新增契约测试 `tests/contracts/test_dlaas_envelope_routing.py`：发 chat envelope，校验内核 `run_turn` 被调用、返回 `OutputAct`。
- 新增 `tests/contracts/test_import_boundaries.py` 规则：`dlaas-platform-*` 禁止 import `volvence_zero.{cognition,memory,temporal,substrate}.*` 内部。

**不包含**（明确推迟）：
- 任何 tenant / shell / template / contract CRUD
- observe / feedback / teach / task / report / command（先只做 chat）
- launcher 多实例（单进程单 ai_id 即可）
- registry 持久化（in-memory 字典）
- SSE 流（一次性 JSON 返回）
- ops / eval / handoff

**Done 检查**（不达标不开下一个 packet）：
- [ ] `curl POST /dlaas/instances/ai_demo/interactions` 能返回 `OutputAct`
- [ ] `import_boundaries` 测试 ACTIVE 强制
- [ ] `vz-*` 一行代码未变（`git diff main packages/vz-*` 为空）
- [ ] spec 与代码一致（已写 `dlaas-platform-rollout.md` 标注切片 1 完成）

**为什么这个切片最关键**：它**最小成本验证了上一轮我说的"vz 完全不动"的承诺**。如果在切片 1 就发现要动 `vz-*` 才能跑通，意味着整个边界设计有漏洞，必须立刻回头改 spec。

---

## 切片 2：typed envelope 全 6 类（**2–3 packet / wheel-internal，~2 周**）

切片 1 证明骨架成立后，把 `interaction_type` 的另外 6 类填齐。**全部是路由翻译，0 新 owner**。

每个 packet 一个 interaction_type（顺序按已有内核入口的成熟度，从最稳定的开始）：

| Packet | interaction_type | 翻译目标 | 理由 |
|---|---|---|---|
| 2.1 | `feedback` | `submit_dialogue_outcome(kind=…)` | 完全一对一，最简单 |
| 2.2 | `observe` | `IngestionPipeline` + `submit_*_event` 按 `observation_type` switch | 5 种 observation_type 各占一条 dispatch |
| 2.3 | `teach` / `task` | `run_turn(trigger_kind=APPRENTICE)` | 已有 `is_apprenticeship_trigger`，加 HTTP 出口 |
| 2.4 | `report` + `command` | `end_scene(drain_slow_loop=True)` + 反思 snapshot 投影 / 显式动作白名单 | 最后做，因为依赖 SHADOW 反思快照成熟度 |

**每个 packet 的 Done 检查**：
- [ ] 有契约测试 `tests/contracts/test_dlaas_<type>_dispatch.py` 证明 envelope → kernel 调用一一对应
- [ ] `vz-*` 仍未动
- [ ] DTO 的 typed enum 写在 `dlaas-platform-contracts`，禁止 platform-api 用字符串字面量

---

## 切片 3：control plane 持久化（**3–5 packet，~3–4 周**）

> 此时已经有了能用的"DLaaS 形状但只有 1 个硬编码 ai_id"的服务。开始把控制面落地。

**Packet 3.1：dlaas-platform-registry + Tenant CRUD + auth 中间件**
- SQLite + 一张 `tenants` 表
- 三种 auth header 中间件：`X-Tenant-Api-Key/Secret` / `X-Control-Plane-Secret` / `X-Service-Secret`
- 端点：`POST/GET /dlaas/tenants`
- 现有 `--alpha-users-file` 路径映射为最小 tenant 模型（兼容老配置）

**Packet 3.2：Shell + Asset 资源**
- shell.embodiment 直接复用 `lifeform-affordance` 的 4 Kind 描述符 schema（**重要复用点**）
- asset.uri → `lifeform-ingestion.IngestionEnvelope.provenance.source_uri`
- 端点：`POST/GET /dlaas/shells`、`POST/GET /dlaas/assets`、`POST /dlaas/templates/{id}/assets`

**Packet 3.3：Template + Version + 注册表**
- `runtime_template_id` → 命中已注册 vertical（复用 `lifeform_service.verticals.discover_verticals()`）
- template_version 不可变快照
- `PATCH /dlaas/templates/{id}` 走"造一个新 version + 切 status"

**Packet 3.4：Activate + Readiness**
- `POST /dlaas/templates/{id}/activate` 走 `lifeform-ingestion` 喂 asset → `LifeformSession` 跑一遍 + `end_scene(drain_slow_loop=True)`
- `GET /dlaas/templates/{id}/readiness` 读 memory / regime snapshot 算 world_nodes / self_nodes / l2_cards

**Packet 3.5：Contract + Adopt（单进程多 ai_id 的 launcher v0）**
- `POST /dlaas/contracts` + `POST /dlaas/adopt`
- 把 `lifeform-service.SessionManager` 抽出 `dlaas-platform-launcher.InstanceManager`（管 `{ai_id → Lifeform}`）
- substrate runtime 共享逻辑直接搬现有 `_enforce_frozen_for_sharing` 护栏到 instance 层

**每个 packet 各自 Done 检查**：spec 同步 / 契约测试 / `vz-*` 不动 / 老 `/v1/sessions/...` 仍可用。

---

## 切片 4：focus_persons + identity_links（**1–2 packet，~1–2 周**）

- focus_persons 写入路径走 `submit_profile_event`（**已存在**），platform 只持有 ai_id↔person_id 索引
- identity_links 直接拼 `UserIdentity.scope_key`，0 改 `vz-memory`
- 这一切片之后，多人对话基本可用

---

## 切片 5：Ops（pause / handoff / SSE，**3–5 packet，~3–4 周**）

- **Packet 5.1**：`dlaas-platform-ops` 维护 `{(ai_id, session_id) → paused_state}`；paused 时 `interactions` 短路返回 `operator_takeover` placeholder（**完全在 platform 层，不进内核**）
- **Packet 5.2**：handoff queue + ticket CRUD；handoff trigger 来自**读** `vz-cognition.rupture_state` 快照，按 `RuptureKind` 决定升级阈值
- **Packet 5.3**：admin SSE `/dlaas/admin/ops/conversations/stream`，事件源是 platform 自己的 ledger
- **Packet 5.4（可选）**：真流式 SSE。需要 `vz-substrate` 加 additive `generate_async`——这是 6 个 wheel 里**唯一可能动到 vz-* 的地方**，单独 packet，仔细 review

---

## 切片 6：Eval gate（audience / exam / license，**2–4 packet，~3 周**）

- 复用 `lifeform-evolution.closed_alpha_preflight` 框架
- LLM judge 严格按 EVO-2 / R12 当 readout，**任何反向写回 reward 的 PR 一律拒绝**
- License gate 阻止 `template.status` 切到 `published`，但不写 kernel

---

## 每个 convergence packet 的标准模板

复用 `.cursor/rules/cursor-convergence-workflow.mdc` 的 Done 检查 + 你已有的 PR 习惯：

```markdown
# DLaaS Packet N.M — <主题>

## Owner 边界
- 这个 packet 落哪个 wheel 的哪个 owner
- 哪些消费者读它的 snapshot

## 现状核查（do this first）
- 当前 lifeform-service / vz-* / dlaas-platform-* 已有的相关代码与 spec
- 假设是否成立；如果已经满足，本 packet 取消

## Spec 改动
- 列出要改的 docs/specs/*.md / DATA_CONTRACT.md / archetecture.md
- 涉及哪条不变量

## WiringLevel 迁移路径
- DISABLED → SHADOW → ACTIVE 三态各自的契约测试

## 评估证据
- 哪条 contract test / e2e test 提供"切 ACTIVE 之前必须通过"的证据
- 与 docs/specs/evaluation.md 哪个 family 关联

## 退出条件 / 回滚
- 如何 disable（一个 env var / 一行配置）
- legacy 路径保留多久

## 影响半径
- vz-* diff = 0 行（必须验证）
- lifeform-* diff（如有）行数与文件
- 新 wheel 文件清单（≤ 8 个）
```

---

## 三个高频踩坑（提前装护栏）

1. **CRUD 资源膨胀**。一不留神就开始给 Template 加 20 个字段。每加一个字段先问"内核哪个 owner 已经持有了这个语义？"——大多数答案是"已有"，那就 reference 不重复存。
2. **platform 偷偷读 kernel 内部**。最容易出现在 readiness gate / handoff trigger 里——直接 `from volvence_zero.memory import xxx` 然后摸 `MemoryStore` 内部。`tests/contracts/test_import_boundaries.py` 必须**第 0 步就加 platform 反向规则**，否则等代码写完再加要拆很多。
3. **interaction_type 用关键词解析 `human_brief`**。例如想从 `human_brief="请生成本周报告"` 推断成 `report`——这是 `.cursor/rules/no-keyword-matching-hacks.mdc` 死禁止的。**`interaction_type` 必须客户端显式传**，platform 只做 typed enum dispatch。

---

## 节奏建议（一个人 vs 小团队）

**一个人**（你大概率是这种）：
- 切片 1 做完是关键里程碑——之前的所有架构判断在这里被验证。**预留 2 周 buffer**，因为很可能要回头改 spec 1–2 次。
- 切片 2/3/4 之后**一定要至少做一次 demo**（哪怕只对自己），否则会失去节奏感
- 切片 5/6 单独说服自己"现在真的需要"再做。可以在切片 4 之后停半年，先用现有形态收集真实使用反馈。

**小团队（2–3 人）**：
- 切片 3/4 的 packet 之间依赖弱，可以并行
- 切片 5 的 ops 工作量大、与认知层耦合最深，需要熟悉 `rupture_state` 的人做
- 切片 6 单独抽一个人对接 evaluation 体系

---

## 第一步具体动作

如果你要明天就开始：

1. **今晚**：读完上面这份方案，决定接受/调整哪些设计点。
2. **明天**：开第 0 步的 spec packet——开 plan mode 让我陪你一起把 `docs/specs/dlaas-platform.md` 起草出来；不写代码。
3. **下周**：开切片 1 的 MVP packet，目标"一条 chat envelope 跑通 + vz-* 0 改动"。
4. **下下周**：站在切片 1 上 review，决定按切片 2.x 推进的顺序。

要不要我现在切回 plan mode，陪你把第 0 步的 spec packet（dlaas-platform.md 大纲 + 8 条不变量条款 + DATA_CONTRACT 新增 slot 列表 + import boundary 测试规则）先列出来？这一步完成后你就可以开始切片 1 的实际代码了。

---

**User**

非常好，按照上面的这个方案的5个切片， 开始作计划开始实施。 刚开始不要太重的测试， 作完整个的之后我们再作测试就行。 

---

**Cursor**

在写计划之前，有一个关键的范围问题需要你确认（其它可以我替你做合理默认假设）。

我上一轮给的是**第 0 步 + 切片 1-6**共 7 个台阶；你说的"5 个切片"我想先对齐到底是哪 5 个，因为这直接决定计划的边界、时长和最后的"测试集中收口"在哪里发生。

> **本轮实施的范围对应原方案中的哪一段？**
> - 切片 1–5：MVP 骨架 + 6 类 envelope + 控制面持久化 + focus_persons/identity_links + Ops（pause/handoff/SSE）。不含 Eval gate（exam/license）。— 严格对应你说的“5 个切片”
> - 切片 1–6：上面这 5 个 + Eval gate（audience/exam/license）。一次推完整套 DLaaS ✔
> - 切片 1–4：MVP 骨架 + 6 类 envelope + 控制面持久化 + focus_persons/identity_links。先稳基础，Ops 与 Eval 都推迟
> - 切片 1–3：MVP 骨架 + 6 类 envelope + 控制面持久化。最小可演示 DLaaS 形状即停

> **本轮做完后第一个想看到的 demo 是什么？这决定了切片内部 packet 的优先级排序**
> - 单租户、单 ai_id、纯 chat：通过 DLaaS 形状端点跑通一条对话即可（最小验证）
> - 多租户、多 ai_id、chat + observe + feedback 三类 envelope 跑通：能用 DLaaS 真实形状收集第一批生产反馈
> - 完整生命周期：register tenant → 上传 asset → 创建 template → activate → adopt → 多渠道 interactions + 必要时 operator 介入 ✔

好。范围明确了：**全套 6 切片 + 完整生命周期 demo**。下面我把整体计划落到 `CreatePlan`，每个切片下列出可执行的 packet、要碰的文件、以及和 vz 内核的边界检查。测试按你说的"集中在最后做"，但每个 packet 仍保留两样最便宜的护栏：`import_boundaries` 架构守卫 + 一个 smoke check（仅证明不崩）。

---

**User**

DLaaS on VZ Roadmap

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

我开始按计划逐切片落地。第一步先把 todo 标记为 in_progress，并读取需要扩展的几个核心文档以确保 spec 同步与现有约定一致。

Good, I now have the structure. Let me also check the existing wheel pyproject and the `lifeform-service` structure to understand patterns I need to follow.

Now I have a good picture of the conventions. Let me start Packet 1.1: spec + DATA_CONTRACT slot + archetecture wheel rows + SPLIT three-tier note + rollout doc.

Now add the 4 slot placeholders to DATA_CONTRACT.md.

Now update `archetecture.md` to add the platform wheel rows.

Now update SPLIT.md to add the three-tier note.

Now create the rollout doc.

Packet 1.1 complete. Now Packet 1.2: create `dlaas-platform-contracts` and `dlaas-platform-api` skeletons. Let me also check the existing `lifeform-affordance` for reference.

Now create `dlaas-platform-contracts` wheel.

Now create `dlaas-platform-api` skeleton.

Now I need to look at how `_handle_turn` works to model the new dispatch correctly. Let me also look at `_handle_dialogue_outcome` to understand response patterns and what helpers are needed.

Now I'll create the api `app.py` with the dispatch handler. Slice 1 only handles `chat` per the plan; other types return 501 with explanatory message until later slices fill them in.

Now register the two new wheels in the workspace pyproject.

Packet 1.2 done. Now Packet 1.3: import_boundaries reverse rule (this is critical to seal the architecture). The existing test scans `vz-*` and `lifeform-*` files. I need to add a third pass that scans `dlaas-platform-*` files and forbids importing internal vz subpackages and internal lifeform domain modules.

Now add the new platform-tier tests at the end of the file.

Now let me also add the smoke check that the user said to keep minimal. I'll add a single dedicated smoke test for the chat dispatch.

Good, this gives me a clear pattern for the smoke test. Let me write it.

Let me run the import_boundaries test to make sure everything passes before moving forward.

All 721 boundary tests pass. Let me install the new wheels into the workspace so the smoke test can find them, and verify it.

Now install the new wheels with `--no-deps`.

Excellent. **`lifeform-service` was not touched at all** — even better than the plan promised (the dispatch is grafted from the platform side). The `vz-*` diff is also exactly zero. Let me confirm:

**Slice 1 Done check passes**: `git diff main packages/vz-* packages/lifeform-*` outputs nothing. The architectural promise is verifiably true. Let me try the smoke test once with packages installed.

**4/4 chat smoke tests green**. Slice 1 complete. Mark done and start Slice 2.

Now Slice 2: typed envelope, 4 packets. Let me start by understanding the existing kernel entry points I need to translate to.

Now I have all I need. Let me also check for `IngestionPipeline`.

Now let me also check `lifeform-core.types.TurnTriggerKind` and `is_apprenticeship_trigger`:

Now I have everything to write all 4 packets of Slice 2. Let me design a clean dispatcher module, refactor the api `_handle_interaction` to use it, and add typed observation_type / command_name dispatch.

---

**User**

继续

