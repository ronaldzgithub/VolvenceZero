# Seven-Day Simulated Companion Evidence Spec

> Status: tooling + final preregistration + product-path smoke complete; formal 36-run matrix running
> Last updated: 2026-08-01
> 对应需求: R5、R6、R7、R8、R12、R15
> 关联债务: #93（只提供 simulated-longitudinal 辅助证据，不关闭 real-user EXIT）

## 目的与主张边界

本能力把 `companion-bench` 的 typed FSM 模拟用户、`lifeform-service` 的公开 HTTP
session/end-scene 接口、owner persist/restart/hydrate 生命周期和
`relationship_continuity` 七项只读指标接成七个虚拟日的自动证据回路。它回答的是：
在**模拟用户 + 真实产品生命周期**下，仅改变 per-user state loading 或 sleep drain
是否导致对话级连续性 readout 改善。

自动结果的唯一允许措辞是 `simulated-user-real-lifecycle-only`。它不等于真实用户产品
价值，不授权 production promotion，也不改变任何运行时 `WiringLevel`。L4 盲评中聊天者
可以是模拟用户，但评分者仍必须是真实、非项目成员；由此最多形成
`human-rated-simulated-user-transcripts-only`，不能满足 #93 的 real-user product-value
EXIT。

## Owner 与契约

- `lifeform-evolution.SevenDayCompanionOrchestrator` 只调用 service HTTP port 和
  lifecycle port；不 import 或直调 Brain 内部 owner。它发布
  `seven-day-companion-run.v1` 与每日 `seven-day-companion-day.v1`。
- `companion-bench.FrozenSevenDayUserScript` 是每个 `(scenario, seed)` 的冻结用户输入
  owner。typed FSM 先生成不可变的实质句，本地 Qwen 只从封闭候选中选择语气开场；
  它不得改写事实、preference、boundary 或 callback。35 个 turns 通过 SHA 冻结，各消融
  臂逐字节重放；臂间 assistant response 不得回头改变用户输入。
- `vz-runtime.seven_day_companion_evidence` 是 out-of-turn evaluation owner，只读 run
  artifact，发布 `seven-day-companion-ablation.v1`。结果禁止进入 PE、credit、reward、
  ModificationGate 或 owner hydration。
- 复用既有 `relationship_continuity` slot 和 owner hydration 契约，不新增 runtime slot；
  因此 `docs/DATA_CONTRACT.md` 无新注册项。

## 七日日程与生命周期

每个场景固定 7 sessions、每天 5 exchanges、相邻 session 虚拟时间间隔
`86_400_000 ms`。每天顺序为：create session → cold-start readout → 5 turns →
console `keep/delete` probe → end-of-day readout → end-scene → pilot capture → close/persist。
readout 必须位于 end-scene 之前，因为 end-scene 会清空当前上游快照。Day 1–6 后必须产生新的
service instance identity，health check 成功且 persistence scope 不变；Day N+1 在同一
user scope 建 session，由 owner 正式 hydration 路径读取状态。

正式 process host 以 argv（`shell=false`）启动 service，每次重启先终止自己拥有的旧
进程、轮询 `/v1/health`、再把 HTTP client 绑定到新的 generation/PID identity。state
controller 只允许操作显式 evidence root 的子目录，不删除数据：当天 active scope 原子
rename 为 immutable `day-N` archive，再按臂把来源 archive copy 到下一实例的 active
scope；archive 与 loaded copy 都发布 SHA-256。`evaluation__relationship_continuity_v1.json`
是测量 checkpoint，不属于被操纵的产品 owner state：state digest 明确排除它，所有臂在
staging 后恢复当臂自己的 measurement bytes，并单独发布 SHA-256。

service 的 `observed_at_ms` 覆盖只在显式 `allow_evidence_time_override` 模式开放；产品
默认拒绝客户端伪造时间。

## 场景包

scenario package `seven_day_companion_v1` 包含 3 persona（researcher / nurse /
designer）× 2 arc（progressive warmth / rupture-repair）。每个 path 都被 arc 引用，phase
order 为 0..6；routing 使用 embedding similarity + schema-bound structured output，
禁止 substring、regex 或 keyword dictionary。每天事件由 typed FSM action 与
`callback / emotion / boundary` event tags 显式发布，不从自然语言反推。

## 消融矩阵与 readout

State 四臂：

1. `correct-user-state`：按序加载同一用户 owner snapshots；
2. `stateless`：新一天不加载先前 owner state；
3. `swapped-user-state`：只加载 matched donor 用户的 owner state；
4. `shuffled-history`：加载同一用户但按预注册乱序构成的 owner state。

`shuffled-history` 在 Day 1–6 结束后为下一天选择的冻结 source-day 序列是
`[1, 1, 2, 1, 4, 3]`；来源只能是已完成的同用户 correct-state archive，禁止未来泄漏。
stateless 必须证明没有 staged digest；其余臂必须证明 source day 不晚于当前日且 loaded
digest 与 source archive 一致。缺少该 attestation 的 run 不能进入分析或 capture。

Sleep 两臂：`sleep-consolidation` 在 end-scene drain slow loop，`no-sleep` 不 drain。
各臂必须共享 scenario、seed、35 个 user turns、虚拟日历、SUT model/adapter fingerprint
和模型版本；唯一操纵变量是 state loading policy 或 slow-loop drain。

每天通过公开 relationship-memory console API 对排序第一、第二的 pending memory proposal
分别执行 `keep` 与 `delete(content_inaccurate)`；两项行为在所有臂完全相同，提供 console、
correction、wrong-attribution 与 usefulness 的真实分母，不做指标插补，也不直调 owner。

每日记录七项 owner readout、cold-start/end-of-day 两个 phase、typed callback
opportunity 与可选 `fsm_probe_pass_rate`。主判据不做缺失值插补：单个 phase 七项任一为
`null` 时该 phase composite 为 `null`；coverage gate 要求 Day-7 composite、callback pair
和 Day 2–7 至少一个完整 cold-start composite 对每个预注册 case 成对齐全。Day-1 只有一个
trust point，允许其 trust delta 为 `null`。LLM judge 和 FSM semantic scorer只能是次级 readout。

## 冻结预注册与当前状态

权威 preregistration：
`artifacts/seven_day_companion_simulated_prereg_20260731T222910Z.json`，SHA-256
`9ae32c6cf4c7484502f21ce090532ff5c9f31c793364d40e75e24501fcb8792c`。正式矩阵冻结为
6 scenarios × seed `1501` × 6 arms = 36 runs、252 sessions、1260 exchanges；formal
禁止 deterministic fake。模拟器为冻结的 `Qwen/Qwen2.5-1.5B-Instruct`（Qwen family），
SUT 为冻结的 `HuggingFaceTB/SmolLM2-360M-Instruct`（SmolLM family），设备固定 MPS。

截至 2026-08-01，最终 6-scenario/210-turn 模拟器预检通过；一条非 claim smoke 完成
35 个真实 SUT 回合、14 个 console actions、6 次不同进程重启，run SHA-256 为
`e5df2bb0bcafbec971b0ae1cb0dc97127731f6050b1d4ece684ce0f40a214a45`。正式 36-run
矩阵正在 `artifacts/seven_day_companion_formal_20260731T222910Z/` 执行，因此状态仍是
`running / no causal result yet`，没有“通过”“失败”或“没有提升”的 effect verdict。
`193423Z` 在任何正式 outcome 前被本版取代，supersession artifact 保留方法学变更。

## Gate 8/11 v1 capture 兼容性

冻结的 `gate811-human-anchor-prereg.v1` 约束 transcript shape、matched variables、事件
覆盖和真人 rater，没有限定聊天者必须是真人，因此 simulated capture 无需修改 v1。
每个七日 run 通过两个 typed-event-complete 的三日窗口产生候选；6 scenarios × 3
capture seeds × 2 windows = 每 contrast 36 candidates，冻结盲化工具选择 24 pairs。

capture 仍须携带 synthetic consent-scope SHA、PII scan artifact SHA、明确 deidentifier
和三类 typed event attestation。pilot 输出恒为
`human_anchor_claim_allowed=false`、`human_ratings_pending=true`；未取得真人评分时只是可发给
rater 的材料，不是 human evidence。

## 失败、退出与回滚

- correct-state 不优于 stateless：把连续性主张收缩到 typed owner metric 行为；不换
  seed、不降阈值。
- sleep 不优于 no-sleep：禁止次日巩固产品主张。
- metric 缺失、模型族重叠、user-turn/calendar/fingerprint 不匹配：分析前 abort，先修
  instrumentation，不能产生效应判词。
- 回滚：停止七日证据 runner、隐藏/撤下对应 evaluation artifact；不删除 owner 已有
  产品状态，不修改 reward/credit/learning，也不切换 production wiring。

## 验证

- `packages/lifeform-evolution/tests/test_seven_day_companion.py`
- `packages/companion-bench/tests/test_seven_day_driver.py`
- `packages/companion-bench/tests/test_seven_day_scenario_package.py`
- `packages/vz-runtime/tests/test_seven_day_companion_evidence.py`
- `packages/vz-runtime/tests/test_gate811_simulated_capture.py`
- `packages/vz-runtime/tests/test_seven_day_companion_preregistration.py`
