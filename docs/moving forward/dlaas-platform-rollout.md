# DLaaS Platform Rollout

> Status: in-progress (Slice 1)
> Last updated: 2026-05-09
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
- [ ] Packet 1.2 `dlaas-platform-contracts` + `dlaas-platform-api` 骨架
- [ ] Packet 1.3 `lifeform-service` `_handle_dlaas_interaction` + import_boundaries 反向规则
- [ ] Packet 2.x typed envelope 全 6 类
- [ ] Packet 3.x control plane 持久化
- [ ] Packet 4.x focus_persons / identity_links
- [ ] Packet 5.x Ops
- [ ] Packet 6.x Eval gate
- [ ] Packet 7.x 测试集中收口
