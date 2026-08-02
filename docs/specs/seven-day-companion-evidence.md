# Seven-Day Simulated Companion Evidence Spec

> Status: v1 base-only smoke complete / formal halted at 16/36 for instrument-discrimination；v2 base+adapter+character contract and tooling complete, no ACTIVE artifact smoke yet
> Last updated: 2026-08-02
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
`artifacts/seven_day_companion_simulated_prereg_frozen_20260801T122037Z.json`，SHA-256
`9674ec62f363a362f09ef692ec82b176ee43da0ec47ac906879a1fcbddbed1fd`。正式矩阵冻结为
6 scenarios × seed `1501` × 6 arms = 36 runs、252 sessions、1260 exchanges；formal
禁止 deterministic fake。模拟器为冻结的 `Qwen/Qwen2.5-1.5B-Instruct`（Qwen family），
SUT 为冻结的 `HuggingFaceTB/SmolLM2-360M-Instruct`（SmolLM family），设备固定 MPS。

prereg 还冻结 `packages/*/src`、package/root pyproject 与七日 runner 的 1,245 文件执行源
树，SHA-256 为 `5aab91ad394f0d9c5e6b09519a8010566e8f8bc2324777ee6dc21130417f7b8d`；正式进程只从
只读副本导入。此前 `20260731T222910Z` 只覆盖 11 个选定文件，运行到 32 个完整样本后
被主工作区并发源码变更污染，并在 Day 6→7 hydration 出现
`exploration_rng_state` schema drift；它在读取任何 effect 前失效，完整/部分原始记录均只
保留作 interruption audit，不进入分析。

截至 2026-08-01，替代 6-scenario/210-turn 模拟器预检通过；一条非 claim smoke 完成
35 个真实 SUT 回合、14 个 console actions、6 次不同进程重启，run SHA-256 为
`8ddc857cdad8c7951cae31b7b3e1ed05c95b55cd650e5be2062bddcda4a6985a`。
supersession 与 source-drift invalidation artifact 均单独保留方法学变更。

2026-08-02，替代正式 36-run 矩阵在 16/36 处按本 spec `失败、退出与回滚` 的
「停止七日证据 runner」与「metric 缺失 → 分析前 abort，先修 instrumentation」两条既有
条款停跑，记录为
`artifacts/seven_day_companion_formal_frozen_20260801T122037Z/halt_record.json`。
停跑理由是 `instrument-discrimination`：`user_correction_rate` 与
`remembered_item_usefulness` 由逐字节相同的每日 console 探针产生因而被构造性钉在 `0.5`；
`callback_hit_rate` 比较的是 owner 对自身 compact readout 的 forecast 与 realized readout，
测的是 owner 自洽性而非 SUT 是否用上跨日事实；唯一观测 SUT 行为的
`fsm_probe_pass_rate` 全为 null。因此该目录状态是
`halted / instrument-discrimination`，既不是 effect verdict，也不是
`no improvement`；停跑前未读取该矩阵任何 measurement，未换 seed、未降阈值、未改预注册。

16 个完整 run envelope 与 pilot day 产物全部保留并以 rollup SHA 封存。该目录
**不允许原样 `--resume`**：仪器须先按收窄计划 S3 换成冻结 N+1 substrate 表示预测目标
（arm-independent、非循环），并另开预注册；七项 owner readout 届时降为 secondary。

v1 已完成 run 全部冻结为
`HuggingFaceTB/SmolLM2-360M-Instruct + adapter:none`，每个 run 都有 6 次进程重启。这些
run 不得在完成后补写 common adapter 或角色包字段，也不得与 v2 合并分析。

## v2：base + Common Adapter + per-session Character Package

`seven-day-companion-simulated.v2` 是独立的新预注册协议，不改变 v1。v2 只接受：

- 一个通过 `ModificationGate.OFFLINE`、`require_active()` 成功的
  `CommonAdapterBundle`；
- 至少一个所有内容引用均通过校验、`require_active()` 成功的
  `CharacterPackageManifest`；
- 一个显式 `selected_character_id`、支持角色 package template 的 vertical，以及
  `wiring_level=active`；
- 与 common bundle 完全相同的 base model ID 和权重 SHA-256；冻结模拟器使用不同模型族。

预注册冻结 bundle/manifest 的仓库内相对 locator、文件 SHA-256、内容 ID、common adapter
version/compatibility fingerprint、角色 Prefix/KV ID、vertical 与选角。正式 runner 只从这份
`runtime_stack` 构造 `lifeform-serve` argv，不接受运行时另传 adapter 或角色覆盖值。服务启动后
写不可变的 `character-runtime-stack-attestation.v1`；七天 run 还必须同时证明：

1. 每天创建 session 的响应都绑定预注册的 `character_id`；
2. 每个真实 SUT turn 的 typed `response_rationale_tags` 都包含精确的
   `character_id=...`、`character_prefix=active` 与
   `character_prefix_kv=<frozen-prefix-id>`；
3. 七个服务实例发布相同 L1 bundle ID/version/fingerprint 和 L2 manifest binding；
4. 六个 matched arms 使用相同的完整模型栈指纹。

任一字段缺失、SHADOW/disabled、文件 digest 漂移、Prefix/KV 未真实应用、进程重启后装载栈
变化，均在 effect analysis 前 fail closed。v2 smoke 仍是
`simulated-user-real-lifecycle-only`，不授权 production promotion。

v2 必须使用新的输出根和新 preregistration；不能对当前 v1 目录执行 `--resume`。当前仓库尚无
通过正式 ALLOW gate 的 common bundle/角色 manifest，因此状态是“实现和测试接线完成，真实
ACTIVE 七天 smoke 待训练产物”，不能表述为已跑通。

## Gate 8/11 v1 capture 兼容性

冻结的 `gate811-human-anchor-prereg.v1` 约束 transcript shape、matched variables、事件
覆盖和真人 rater，没有限定聊天者必须是真人，因此 simulated capture 无需修改 v1。
每个七日 run 通过两个 typed-event-complete 的三日窗口产生候选；6 scenarios × 3
capture seeds × 2 windows = 每 contrast 36 candidates，冻结盲化工具选择 24 pairs。

capture 仍须携带 synthetic consent-scope SHA、PII scan artifact SHA、明确 deidentifier
和三类 typed event attestation。pilot 输出恒为
`human_anchor_claim_allowed=false`、`human_ratings_pending=true`；未取得真人评分时只是可发给
rater 的材料，不是 human evidence。

capture 的执行源码冻结在
`artifacts/seven_day_companion_capture_source_prereg_frozen_20260801T133600Z.json`，SHA-256
`5f13ed395f66eac59021c6a8515742b5b2400b995cdc32aba8a7786e66b73ee2`；1,247 文件执行树
SHA-256 为 `ad58546d9f508e1c1810f99289dedc413eae22b33f09338f4170d1c8b2adb2bf`。此前
`20260801T131953Z` 冻结在执行前静态复核时发现导出 manifest 缺少 runner 封口所需的
`pair_count`，没有运行 capture、没有读取评分或效应即被 supersession artifact 废止。
替代 runner 必须先完成 18-script MPS preflight，再执行 18 cases × 4 arms = 72 runs；正式
七日矩阵占用 MPS 时不得并发运行。

## 独立完成审计

正式 bundle 完成后，必须从只读的 execution root 调用
`scripts/audit_seven_day_companion_formal.py`，重新校验 preregistration、冻结用户脚本、
36 个 run、service/session identity、重启链、state archive/loaded copy digest、measurement
checkpoint、pilot transcript、每日 readout、console actions 和 promotion verdict。审计器
重新运行同一 `evaluate_seven_day_ablation`，磁盘结果与重算结果不完全相等即 fail closed；
同时拒绝 service log 中的 HTTP 4xx/5xx，并将 `production_promotion_authorized=false`、
`evaluation_writeback_allowed=false` 写入 `seven-day-companion-independent-audit.v1`
报告。审计报告只证明物证完整，不改变 simulated-user-only claim scope，也不替代真人评分。

capture bundle 完成后还必须调用 `scripts/audit_gate811_simulated_capture.py`，独立复算 72 个
source runs、144 条 capture records、48 个盲评 pairs、内部 key、空白评分 CSV 和全部 SHA
绑定。该审计恒保留 `human_ratings_pending=true`；打包成功不等于人评通过。

## MPS 命令行测试控制面

`scripts/run_seven_day_companion_test_plan.py` 是七日产品证据的统一人类命令行入口，提供
`status / preflight / smoke / formal / audit / all` 六阶段。它以 preregistration schema 为
dispatch SSOT，不拥有实验变量、readout 或 verdict：

| preregistration schema | 自动路由的 Gate | campaign runner / auditor |
|---|---|---|
| `seven-day-companion-simulated.v1` / character-stack `v2` | 8/11 | `run_seven_day_companion_formal.py` / `audit_seven_day_companion_formal.py` |
| `gate1-seven-day-companion-prereg.v1` | 1 | `run_seven_day_gate1_formal.py` / `audit_seven_day_gate1_formal.py` |
| `companion-gate-suite-seven-day-prereg.v1` | 4/5/6/7/9/10，由冻结 `gate_id` 唯一选择 | `run_seven_day_gate_suite_formal.py` / `audit_seven_day_gate_suite_formal.py` |

未知 schema、未知 `gate_id` 或非 MPS hardware-specific preregistration 一律 fail loudly。共同
control-plane helper 与本入口纳入所有后续七日 preregistration 的执行源码快照；已经冻结的旧
preregistration 不做 retrofit，仍只能从它自己的只读 execution root 完成。

`scripts/freeze_seven_day_execution_root.py` 是新的 preregistration-bound 执行根物化入口。
它只复制 `execution_source_snapshot.roots` 命中的文件，在复制前后分别复算冻结的 file count
与 tree SHA-256，拒绝 symlink、源码漂移、仓库内部目标和已存在目标。成功后写入
`frozen_execution_root_manifest.json`，其中包含 preregistration canonical SHA、源码树 SHA、
file count 和逐文件 SHA；整个目标树随后改为只读。该冻结器本身也纳入所有后续七日
preregistration 的 code manifest 与 execution source roots，避免“负责冻结的脚本不在冻结树”
形成新的未绑定执行面。Gate 8/11 的正式 auditor 与 simulated-capture auditor 同样必须进入
后续 continuity preregistration 的两份清单；否则 runner 可执行但独立复算链不完整。

所有涉及模型推理的阶段必须通过真实 MPS tensor 算术探针，固定
`PYTORCH_ENABLE_MPS_FALLBACK=0`，并持有共享的
`artifacts/.companion-evidence-mps.lock`。因此经两个新控制面启动的七日计划与 MSC N+1
计划不能并发占用 MPS，
也不能在 MPS 不可用时悄悄改用 CPU。`status` 与 `audit` 不占 MPS；`all` 的顺序固定为
preflight → formal → audit。完整 formal 的退出码 `0` 表示预注册判据支持、`2` 表示预注册
判据未支持；二者都是必须继续独立审计的科学终态。源码漂移、artifact 不完整、service 故障等
integrity failure 才会停止后续阶段；审计失败覆盖 formal 结果并返回失败。
控制面落地前已经手工启动的进程不持有该锁，必须在原终端单独确认已退出。
`preflight` 不需要 `--output-dir`，也不得创建 formal 输出根；`smoke / formal / audit / all`
必须显式绑定输出根。

formal runner 在运行中自动落盘冻结 user script、每个已完成臂的 run
envelope、每日 measurement checkpoint、archive/loaded-copy digest 和 service log。
`--resume` 只可在同一 preregistration 和冻结 execution root 上跳过已完成 run；
任一源码、脚本或状态证明漂移都走既有 fail-closed 路径。运行期 `status` 按冻结的
scenario × seed × arm 精确重建预期文件名，再校验每个 run envelope 的 schema 与身份；额外、
损坏或错配的 run 都单独报告。只有 exact matrix、campaign evaluation artifact 和同时绑定
同一 preregistration SHA 与当前 evaluation SHA 的有效 independent audit 存在时，
`analysis_allowed=true`。此前的中间
文件不得用于效应判读、变更预注册或触发 promotion。

若输出根包含合法 `seven-day-companion-halt-record.v1`，`status` 必须复核其
preregistration SHA、已保存/预期 run 数、禁止 effect/promotion 与禁止原样续跑声明，并输出
`run_state=halted`、`analysis_allowed=false`。halt record 损坏或与磁盘 run 状态不一致时必须
fail loudly；`formal` 与 `all` 在取得 MPS 锁或启动任何模型进程之前硬拒绝该目录。修复仪器后
只能创建新的 preregistration、冻结执行根和输出根，不能把正式停机误当成普通进程中断。

## Gate 1 七天 matched 扩展

Gate 1 使用独立的新预注册和输出根，不得 retrofit 当前 Gate 8/11 的 36-run 冻结矩阵。
两臂固定为 `gate1-pe-temporal-on-v1` 与 `gate1-pe-temporal-off-v1`，都发布同一正式
`PredictionErrorSnapshot`、使用同一 frozen SUT、temperature=0、temporal/regime bootstrap、
correct-user-state、slow-loop drain、场景、seed 与用户脚本。on/off 只改变预注册的 PE drive
bundle：joint-loop external PE drive、primary PE dominance，以及 temporal consolidation 的
直接 PE 学习；bounded runtime-code modulation gate 在两臂都为 `ACTIVE`，off 臂因没有 PE
drive/late write 而保持 no-op。生产默认仍为 `DISABLED`。

`lifeform-serve --companion-evidence-profile ...` 只有同时启用 closed alpha、独立 memory/evidence
root、virtual-calendar override、`hf-shared` 和 `local-files-only` 才能启动；否则 fail loudly。
每个 service evidence root 必须产生 immutable
`companion_evidence_runtime_profile.json`。CUDA 运行还必须记录 GPU name、compute capability、
显存、Torch/CUDA/cuDNN 版本；因此 4090 是新的 hardware-specific prereg，不可把 MPS artifact
搬过去续跑。

HTTP turn evidence append-only 发布 PE magnitude/bootstrap 与 world/self temporal PE-applied flag。
`Gate1SevenDayHarness` 要求每个 case 两臂各 35 个 PE readout、on 臂所有 non-bootstrap turn 的
双轨 PE write load-bearing、off 臂为零写入；效应主判据是 Day 1–2 到 Day 6–7 的 PE adaptation
相对 off 增益，产品次判据为 Day-7 continuity composite，二者都要求 mean ≥ `0.02` 且 paired
95% CI 下界大于零，并要求 boundary/wrong-user safety 零退化。即使全部通过，自动实验仍固定
`production_promotion_authorized=false`；它最多把 Gate 1 提升为 simulated product-ecology
causal support，生产晋升仍需独立真人/部署证据。

入口为 `scripts/preregister_seven_day_gate1.py`、
`scripts/run_seven_day_gate1_formal.py` 与独立重算入口
`scripts/audit_seven_day_gate1_formal.py`；正式默认至少 6 scenarios × 3 seeds × 2 arms = 36 runs、
252 sessions、1260 exchanges。`--resume` 只跳过完全存在且随后通过 profile/run 审计的 arm，
缺字段、attestation SHA 漂移或 matrix extra/missing 一律在效果分析前中止。

## Gate 4/5/6/7/9/10 专用配对包（2026-08-02）

这些 Gate 不能从原 Gate 8/11 state/sleep 矩阵事后推断。仓库现提供独立的
`companion-gate-suite-seven-day-prereg.v1`，每个 Gate 必须单独开新的
hardware/model-specific preregistration 和输出根。共同执行形状为 6 scenarios × 3 seeds；
Gate 7 是 3 臂、54 runs / 378 sessions / 1,890 exchanges，其余均为 2 臂、36 runs /
252 sessions / 1,260 exchanges。所有臂逐字节重放同一冻结用户脚本，使用 correct-user-state、
每日 slow-loop drain、6 次真实进程 restart 和同一 frozen SUT；每个 run 都必须携带 immutable
runtime profile attestation 与 typed HTTP telemetry。

专用 arm 与唯一 owner-level 干预如下：

| Gate | treatment | matched control | load-bearing owner / 主 readout |
|---|---|---|---|
| 4 | `gate4-active-selector-v1` | `gate4-random-feedback-v1` | apprenticeship owner；typed boundary/callback opportunity 上的 feedback request utility |
| 5 | `gate5-multifrequency-cms-v1` | `gate5-single-timescale-v1` | memory owner；nested 2/4 cadence 对 independent 1/1，PE gate 与 ATLAS replay 两臂相同；Day 6–7 absorption/retention |
| 6 | `gate6-conditioned-meta-init-v1` | `gate6-copy-init-v1` | memory owner；真实 day-boundary reset 使用 conditioned prototype 或 copy-init；Day 2–7 首 turn PE |
| 7 | `gate7-ssl-rl-full-v1` | `gate7-no-ssl-v1`、`gate7-no-rl-v1` | temporal owner；两控制臂运行同一候选 optimizer/readout，分别在 owner checkpoint 恢复 SSL 写入或 RL policy/critic 写入；late-vs-early Internal-RL reward |
| 9 | `gate9-m3-slow-on-v1` | `gate9-m3-slow-off-v1` | temporal SSL owner；`slow_gain=1.0/0.0`，M3 slow signal 与 optimizer state 经 HTTP attest；early-vs-late SSL loss |
| 10 | `gate10-rare-heavy-import-v1` | `gate10-rare-heavy-review-v1` | rare-heavy + ModificationGate；pre-import suite 后 bounded import 对 frozen review-only；early-vs-late PE |

Gate 10 是唯一允许 mutable shared substrate 的证据 profile，并且只允许
`max_sessions=1`、fixed/non-swappable provider、独立 evidence root。任一条件缺失时 service
沿用共享 frozen-runtime guard 并 fail loudly。该例外不改变产品默认：无 evidence profile 时
substrate live mutation 仍禁止，review-only 仍是回滚线。

正式判读先过 mechanism load-bearing，再比较 treatment 对每个 control 的 paired primary、
Day-7 continuity 与 safety。primary 和 continuity 都要求 mean ≥ `0.02` 且 paired 95% CI
下界大于零，boundary/wrong-user 不得退化；缺 arm、缺 telemetry、profile SHA 漂移、重启不完整、
HTTP 4xx/5xx 或源码 manifest 漂移均在效应判读前中止。自动结果恒为
`simulated-seven-day-product-ecology-only` 且
`production_promotion_authorized=false`：包已实现不等于 Gate 已通过，更不等于 production ACTIVE。

统一入口：

```bash
.venv/bin/python scripts/preregister_seven_day_gate_suite.py --gate 4 ...
.venv/bin/python scripts/run_seven_day_gate_suite_formal.py --gate 4 --preflight-only ...
.venv/bin/python scripts/run_seven_day_gate_suite_formal.py --gate 4 --smoke-one-pair ...
.venv/bin/python scripts/run_seven_day_gate_suite_formal.py --gate 4 --execute ...
.venv/bin/python scripts/audit_seven_day_gate_suite_formal.py --gate 4 ...
```

`--gate` 可取 `4/5/6/7/9/10`，`--resume` 只跳过同一 frozen prereg/source root 下已经完整
落盘的 run。MPS 与 4090/CUDA artifact 不能混跑或续跑；换硬件必须生成新 prereg。独立审计从
只读 execution root 重算物理 artifact 数、profile attestation、HTTP error、全部 paired effect
和 verdict，不信任磁盘上的 evaluation JSON。

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
- `packages/vz-runtime/tests/test_companion_test_plan_cli.py`
- `packages/vz-runtime/tests/test_companion_gate1_evidence_wiring.py`
- `packages/vz-runtime/tests/test_gate1_seven_day_evidence.py`
- `packages/vz-runtime/tests/test_companion_gate_suite_evidence.py`
- `packages/vz-runtime/tests/test_companion_gate_suite_preregistration.py`
- `packages/vz-temporal/tests/test_gate_suite_evidence_controls.py`
- `packages/vz-memory/tests/test_context_conditioned_meta_init.py`
