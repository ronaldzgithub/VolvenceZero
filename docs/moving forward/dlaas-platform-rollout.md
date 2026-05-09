# DLaaS Platform Rollout

> Status: Slice 1 → 7 完成；6 wheel × 全套契约 / 多租户 / 完整生命周期 / 兼容 + smoke perf 测试 908 项绿色
> Last updated: 2026-05-10
> 来源 spec: [`docs/specs/dlaas-platform.md`](../specs/dlaas-platform.md)
> 目标形状: [`docs/api/DLAAS_README.md`（EmoGPT 仓库）](../api/) — VZ 自身没有这份文档，参考路径以 EmoGPT 仓库为准

## 总目标

在 VZ 现有内核之上，新增 6 个 `dlaas-platform-*` wheel + 1 处 `lifeform-service` 路由扩展，落地完整 DLaaS 控制平面 + 多渠道运行时 envelope，目标做完后能跑通：

```
register tenant
  → upload asset
    → create template (含 runtime_template_id, persona_spec, seed_config)
      → activate (跑 ingestion + drain slow loop)
        → readiness check (读 memory/regime/temporal snapshot)
          → publish template
            → create contract (绑定 tenant × template × shell × tool_policy)
              → adopt (创建 ai_id + 注册 focus_persons)
                → 多渠道 interactions (chat/observe/feedback/teach/task/report/command)
                  → 必要时 operator pause / handoff
```

**vz-* 内核全程零改动**（仅切片 5.4 可选的 substrate streaming additive 接口可例外）。

## 切片节奏

每个切片下 packet 控制在 3-8 文件，按 [`.cursor/rules/cursor-convergence-workflow.mdc`](../../.cursor/rules/cursor-convergence-workflow.mdc) 推进。spec 同步采取**最小内联**策略：每个 packet 改动涉及的 spec 段落在同 PR 内更新。

### 切片 1：MVP 端到端骨架（in progress）

- **Packet 1.1** spec + wheel 占位 + DATA_CONTRACT slot + import_boundaries 规则【本 PR】
- **Packet 1.2** `dlaas-platform-contracts` + `dlaas-platform-api` 骨架，硬编码 ai_id
- **Packet 1.3** `lifeform-service` 加 `_handle_dlaas_interaction`（不删 `_handle_turn`），仅 chat envelope 跑通

**Done 检查**：`git diff main packages/vz-*` 输出为空 + 新端点能跑 chat。

### 切片 2：typed envelope 全 6 类

- **Packet 2.1** feedback → `submit_dialogue_outcome(kind=…)`
- **Packet 2.2** observe → 按 `observation_type` switch 到 `IngestionPipeline.run` / `submit_*_event`
- **Packet 2.3** teach + task → `run_turn(trigger_kind=APPRENTICE)`
- **Packet 2.4** report + command → `end_scene(drain_slow_loop=True)` + 显式动作白名单

### 切片 3：control plane 持久化

- **Packet 3.1** `dlaas-platform-registry` + Tenant CRUD + 三种 auth 中间件
- **Packet 3.2** Shell + Asset 资源（shell.embodiment 复用 affordance 4 Kind 描述符）
- **Packet 3.3** Template + Version + runtime_template_id 注册表
- **Packet 3.4** Activate + Readiness（触发 ingestion + readout memory/regime snapshot）
- **Packet 3.5** Contract + Adopt + `dlaas-platform-launcher` v0（`InstanceManager`）

### 切片 4：focus_persons + identity_links

- **Packet 4.1** focus_persons CRUD（写入只走 `submit_profile_event`）
- **Packet 4.2** identity_links（拼接 `UserIdentity.scope_key`，0 改 vz-memory）

### 切片 5：Ops（pause / handoff / SSE）

- **Packet 5.1** pause / resume / operator-message
- **Packet 5.2** handoff queue + tickets（trigger 读 `rupture_state` 快照）
- **Packet 5.3** admin SSE conversations stream
- **Packet 5.4** 真流式 SSE（可选，唯一可能动 vz-substrate 的点，单独 review）

### 切片 6：Eval gate（audience / exam / license）

- **Packet 6.1** audience analysis
- **Packet 6.2** exam questions + runs
- **Packet 6.3** launch license gate

### 切片 7：测试集中收口

按用户要求"先做完整套，再集中测试"。前 6 切片只保留 `import_boundaries` + per-packet smoke。

- **Packet 7.1** contract 测试套（envelope routing / kernel isolation / paused short-circuit / no keyword dispatch）
- **Packet 7.2** 多租户隔离 + 持久化 e2e
- **Packet 7.3** 完整生命周期 e2e（demo 目标）
- **Packet 7.4** 向后兼容 + 性能 smoke

**重要纪律**：切片 7 之前所有平台 wheel 默认 SHADOW；老 `/v1/sessions/...` 是 ACTIVE 主路径；不得切 platform endpoint 到 ACTIVE。

## 决策登记

| 决策 | 选择 | 备注 |
|---|---|---|
| Wheel 前缀 | `dlaas-platform-*` | 与 EmoGPT DLaaS API 命名对齐；如未来想中性化为 `vz-host-*` 现在改 0 成本 |
| 持久化选型 | SQLite（MVP）→ Postgres（生产） | 先把 schema / 抽象做对，迁移时只改 driver |
| `OutputAct` shape | 直接照搬 DLaaS README §"Runtime Output Acts" | `act_type` / `capability` / `payload` / `degraded` / `original_capability` |
| shell.embodiment ↔ affordance | 直接复用 4 Kind 描述符 | 这是仅有的可能动到 `lifeform-affordance` 的地方，待 packet 3.2 时进一步确认 |

## 进度追踪

- [x] Packet 1.1 spec + wheel 占位 + DATA_CONTRACT slot + SPLIT 三层 + archetecture wheel 表
- [x] Packet 1.2 `dlaas-platform-contracts` + `dlaas-platform-api` 骨架
- [x] Packet 1.3 `lifeform-service` `_handle_dlaas_interaction` + import_boundaries 反向规则 + chat smoke
- [x] Packet 2.1 feedback envelope → `submit_dialogue_outcome(kind=…)`（typed `FeedbackValence` enum）
- [x] Packet 2.2 observe envelope → `IngestionPipeline` / `submit_*_event`（typed `ObservationType` enum，6 个 sub-handler）
- [x] Packet 2.3 teach + task envelope → `run_turn(trigger_kind=APPRENTICE)`
- [x] Packet 2.4 report + command envelope → `end_scene(drain_slow_loop=True)` + typed `CommandName` allowlist
- [x] Packet 3.1 `dlaas-platform-registry` + Tenant CRUD + 3 auth middlewares（X-Tenant + X-Control-Plane + X-Service）
- [x] Packet 3.2 Shell + Asset CRUD（shell.embodiment 直接复用 4 Kind 描述符 schema；asset.uri → IngestionEnvelope.provenance.source_uri）
- [x] Packet 3.3 Template + Version + runtime_template_id registry（PATCH 自动 snapshot 新版本，PUBLISHED 切换强制 readiness）
- [x] Packet 3.4 Activate + Readiness（envelope_from_text 注入 persona/seed → IngestionPipeline → end_scene drain；readiness 读 memory snapshot 计 world/self/l2 节点）
- [x] Packet 3.5 Contract + Adopt + `dlaas-platform-launcher` v0（`InstanceManager` `{ai_id → SessionManager}` shared substrate；`/dlaas/adopt` 一站式创建 contract / 注册 focus_persons / 颁发 instance_endpoint）
- [x] Packet 4.1 focus_persons CRUD（写入路径 `submit_profile_event`，cognitive owner 仍是 vz-cognition.social_cognition）
- [x] Packet 4.2 identity_links 单条 + 批量 + 列表（canonical_end_user_ref → `UserIdentity.scope_key` 拼接，0 改 vz-memory）
- [x] Packet 5.1 pause / resume / operator-message（per-session pause 状态机；inject_into_runtime=true 走 apprentice 注入）
- [x] Packet 5.2 handoff queue + tickets（trigger 读 `rupture_state` snapshot via `evaluate_session`，按 RuptureKind 阈值升级；tenant + admin 端点）
- [x] Packet 5.3 admin SSE `/dlaas/admin/ops/conversations/stream`（LedgerBroker 广播 turn / pause / resume / operator_message / handoff_open|resolved）
- [~] Packet 5.4 真流式 SSE（cancelled — Slice 5 不动 vz-substrate；保留 JSON 一次性返回 + admin SSE。需要时单独 packet 走 substrate streaming）
- [x] Packet 6.1 audience analysis（profile 持久化 + 占位 readout，待 LLM judge plugin 接入）
- [x] Packet 6.2 exam questions + runs（typed RubricEntry；complete 用 caller-supplied ai_responses，execute 走 launcher 跑 apprentice turn；`DefaultRubricGrader` fail-closed）
- [x] Packet 6.3 launch license gate（`/license/evaluate` 仅检查 passing exam_run；signoff 把 run.passed 写入 license。Adoption 已强制 PUBLISHED + ACTIVATED）
- [x] Packet 7.1 envelope dispatch 契约套：`tests/contracts/test_dlaas_dispatch_contracts.py`（22 项；mock session、no kernel——证明 typed enum dispatch 的所有 7 类 + 各类 typed payload 验证）
- [x] Packet 7.2 多租户隔离 + 持久化 e2e：`tests/service/test_dlaas_multi_tenant_persistence.py`（6 项；两租户互不可见 / 401-403 typed errors / 兼容 alias 头 / SQLite 跨进程持久）
- [x] Packet 7.3 完整生命周期 e2e：`tests/service/test_dlaas_full_lifecycle.py`（3 项；register → asset → template → activate → publish → exam → license → adopt → identity_links → 4 类 interactions → pause/operator-message/resume → handoff CRUD + 失败 gate）
- [x] Packet 7.4 向后兼容 + perf smoke：`tests/service/test_dlaas_backward_compat.py`（8 项；Slice 1 模式与全栈模式都保留 `/v1/sessions/...`；全栈拒绝未 adopted 的 ai_id；并发 5 路 dispatch 不死锁；10 turn ≤ 60s 上限）

## 测试覆盖

```
tests/contracts/test_import_boundaries.py             865 / 865  ✓ AST 守卫
tests/contracts/test_dlaas_dispatch_contracts.py       22 / 22   ✓ Slice 7.1
tests/service/test_dlaas_chat_smoke.py                  4 / 4    ✓ Slice 1 baseline (Slice 2 升级)
tests/service/test_dlaas_multi_tenant_persistence.py    6 / 6    ✓ Slice 7.2
tests/service/test_dlaas_full_lifecycle.py              3 / 3    ✓ Slice 7.3 (full happy path + 2 失败 gate)
tests/service/test_dlaas_backward_compat.py             8 / 8    ✓ Slice 7.4
                                                       ──────
                                                      908 / 908
```

vz-* 内核 7 wheel diff 仍然是 **0 行**（承诺不破，包括 Slice 5.4 cancel 保护了 vz-substrate）。`tests/contracts/test_import_boundaries.py` 持续守 6 类边界（kernel ↛ lifeform / kernel ↛ platform / lifeform ↛ platform / platform ↛ kernel internals / platform ↛ lifeform domain internals / 评估 backbone 类型门面）。

## Slice 1 → 6 wheel 总览（共 6 个新 wheel）

| Wheel | 职责 | Slice |
|---|---|---|
| `dlaas-platform-contracts` | typed dataclass：envelope / dispatch_vocab / resources / eval | 1 → 6 持续扩 |
| `dlaas-platform-api` | aiohttp router + 7-类 dispatch + 控制面整合 | 1 起 + Slice 2 dispatch + Slice 3+ full stack |
| `dlaas-platform-registry` | SQLite-backed CRUD + 3 auth；tenants / shells / assets / templates / contracts / focus_persons / identity_links / handoff_tickets / audience_profiles / exam_questions / exam_runs / launch_licenses | 3 / 4 / 5 / 6 |
| `dlaas-platform-launcher` | `{ai_id → SessionManager}` + 共享 substrate；vertical 解析 | 3.5 |
| `dlaas-platform-ops` | pause / resume / operator-message / handoff trigger / SSE ledger | 5 |
| `dlaas-platform-eval` | audience / exam runs / launch license + DefaultRubricGrader | 6 |

## DLaaS 平台后续可选演进

已转移到 [`docs/known-debts.md`](../known-debts.md) #12-#17 作为正式架构债，按现有节奏（路径/问题/风险/触发条件/推荐修法/优先级）维护：

- **#12** Slice 5.4 真流式 SSE（substrate streaming additive 接口；优先级低）
- **#13** Eval gate 用 fail-closed `DefaultRubricGrader`，未接真实 LLM judge（优先级中）
- **#14** Audience analysis 是占位 readout，未真正分析 corpus（优先级低）
- **#15** Activate 用 persona/seed 文本，未真正抓 asset.uri（优先级中）
- **#16** Contract.tool_policy_snapshot 未推到 AffordanceRegistry 运行时白名单（优先级中-高，生产化阻塞项）
- **#17** 单进程多 ai_id 部署上限：跨进程 / 跨 GPU 共享 substrate 缺失（优先级低，SaaS 化阻塞项）
