# Digital-Ant Embodiment Spec

> 能力域：非语言 embodiment 测试床 | 库：`vz-embodiment-ant`（独立 owner）
> 关联：`research/ant/04_digital_ant_feasibility.md`（可行性）、`docs/DATA_CONTRACT.md` §2.19、
> `docs/next_gen_emogpt.md`（R2 / R3-R4 / R5-R6 / R-PE / R8 / SSOT）

## 1. 目的与边界

数字蚂蚁把 VolvenceZero 分层认知内核（`vz-temporal` / `vz-memory` / `vz-cognition`）**原样复用**，
接到一个完全不涉及语言的 2D 感觉运动 substrate 上。唯一目的：在没有 LLM、没有 token 的情况下，
独立检验 R2（冻结基底 + 有界控制器）、R3-R4（潜控制器空间的时间抽象）、R5-R6（连续记忆谱）、
R-PE（预测误差一级信号）、SSOT（快照隔离）是否**独立于语言**而成立。

它**不是**产品线，也**不是**昆虫神经科学新发现（AntBot / Ardin 2016 已验证路径积分与路线记忆）。
它的价值是「架构证伪能力」+「直观可演示性」，投入规模长期保持在研究旁支量级。

## 2. 边界（SSOT / R8）

- `vz-embodiment-ant` 是独立 owner，只依赖 `vz-contracts`（快照契约）、`vz-substrate`
  （`SubstrateAdapter` ABC + 快照类型）、`vz-runtime`（`AgentSessionRunner` / `Brain` facade）。
- **禁止**直接 import `volvence_zero.temporal` / `volvence_zero.memory` / `volvence_zero.prediction`
  / `volvence_zero.internal_rl` / `volvence_zero.joint_loop` / `volvence_zero.credit` /
  `volvence_zero.dual_track` / `volvence_zero.regime` 等内核内部实现。由
  `packages/vz-embodiment-ant/tests/test_import_boundaries.py` 强制。
- 与内核的唯一耦合是 `SubstrateSnapshot`（输入）+ `AgentTurnResult.active_snapshots["temporal_abstraction"]`
  的 `controller_state.code`（输出 `z_t`）。这两个都是 vz-contracts / vz-runtime facade 表面。

## 3. 两个冻结向量函数（substrate 的字面含义）

| 函数 | 输入 | 输出 | 性质 |
|---|---|---|---|
| `sense_encode` | `WorldObservation` + `NavigatorState` | 固定维感知向量（`SENSE_CHANNELS`） | 纯 numpy，无可学习参数 |
| `motor_decode` | `z_t`（`ControllerState.code`, len n_z） | `(turn_command, step_command)` | 纯 numpy，无可学习参数 |

`z_t` 契约（egocentric 抽象动作）：ndim controller 的 code 被界在 `[0,1]`，因此 `z[0],z[1]`
采用非负 opponent residual steering 编码：`forward=1+z0+z1`、`left=z1-z0`；
固定 forward baseline 让 near-zero code 只产生 near-zero turn，避免在巢边形成 ±45° 小圆；
相等时直行、`z1>z0` 左转、`z0>z1` 右转；`z[2]` → 期望速度（squash）。历史直接
`atan2(z1,z0)` 会让非负 controller 结构性无法右转，已移除。
controller 自由学习「感知特征 → egocentric 动作」的映射；`motor_decode` 只做有界转换。**策略在可学习的内核，plant 冻结在此。**

`AntNavigator`（body 侧，冻结）维护环形吸引子朝向估计 ĥ 与路径积分回巢向量（对应中央复合体）。
正式证据必须让 world 真值运动噪声与 navigator estimate 独立，并只由真实 `world.act` 产生轨迹；
历史上以 noisy estimate 推进真值或通过 `set_body_pose`/`sync_to` 构造的 lane 仅是 legacy smoke，
不能用于 AntBot/Ardin claim。navigator **从不读世界真值位置**——路径积分误差因此是真实测量。

**天空罗盘通道（绝对航向传感器，对应偏振光罗盘）**：`AntNavigator` 融合一个带噪声的绝对航向
观测 `compass = true_heading + N(0, σ_compass)`，用互补滤波修正积分航向
`ĥ ← ĥ + k·wrap(compass − ĥ)`（`compass_gain=k`，`compass_gain=0` 退回纯 dead reckoning）。
这是**航向参考，不是位置读出**，与 AntBot 的天空罗盘 + 光流测距配置对齐——纯 efference-copy 积分的
航向误差按 √N 随机游走增长，物理上到不了 AntBot 量级；有 AntBot 级罗盘（σ≈0.4°）才能达标。
罗盘是所有导航共用的 substrate 传感器（`AntSession` / `FixedRuleAnt` 默认开启，同一 frozen substrate
在 matched-control 各臂间一致），不是只为 homing 实验选择性开启的调参。此通道引入的是一个**受控、
只读的真值航向耦合**，其他真值位置隔离不变。

## 4. Outcome → PE 接缝（正式证据）

`semantic_*_pull` 只表示 substrate 发布的感觉/动机预测通道，不能冒充环境任务结果。正式行为学习链路为：
`AntWorld` 发布 typed、不可变且只含可观察事实的 `EnvironmentOutcome.measurement`；runtime facade 保留
event/prediction/action lineage，并在下一 turn 只交给 PE owner；PE 形成 signed mismatch，credit 再将其
归因给 β segment，Internal RL 只读 PE/credit。

觅食有且只有两个真实、离散、可观察的里程碑：**pickup**（`carrying_food` False→True）与 **delivery**
（终局回巢投递）。`AntSession._environment_outcome` 只在这两个事件上发布 measurement——pickup 给部分
`task_progress=0.5`（非终局），delivery 给 `task_progress=1.0`（终局）。**明确禁止**发布任何"到食物距离
越近奖励越高"的连续势能塑形（potential-based / distance shaping）：连续的 `closer=better` 信号等于把
FSM 手写的梯度跟随答案直接喂给控制器，而"如何朝食物走"恰是控制器**必须自己学**的技能。历史上曾短暂
引入过基于 `food_center` / `eval_home_distance` 的连续势能塑形，已移除（既有泄露方向策略的风险，且经点火
探针验证：即便 credit 变密，运行时 z_t 仍零变化，塑形无助于点火——断点在内核"学习→运行时控制输出"应用
路径，不在此接缝的信号密度）。

禁止 `AntWorld` 直接传 reward 给 temporal/Internal RL，禁止 evaluation 反灌 reward，禁止 runtime
另建 mismatch slot。历史上只依赖 drive PE 的结果可以保留为机制 smoke，但不是 learned foraging 证据。

## 5. 三层生物学 ↔ VZ 对齐

| 生物学层 | VZ 对应 | 在数字蚂蚁中 |
|---|---|---|
| 基因组（不变） | frozen substrate | `sense_encode` / `motor_decode` / `AntNavigator` |
| 基因表达程序（窗口期，离线） | rare-heavy artifact refresh | Phase 2 角色重编程离线循环，产出个体倾向性初始化参数，运行时不可触发 |
| 突触可塑性（在线） | online-fast controller | `z_t` / `β_t` + CMS 在线学习（内核承担） |

### 5.1 隐藏电机扰动校准（experimental / BLOCK）

`MotorDistortionProfile` 是环境 owner 的不可变 actuator transfer：
`applied_turn = clamp(gain·commanded_turn + bias)`，可在 `switch_tick` 一次性切到另一组
gain/bias；空 profile 是严格 identity，单 profile 可广播、多 profile 必须与 body 数一致。公开
`WorldTransitionEvidence` 只增加外部可观察的 `commanded_turn / applied_turn`，不泄露 profile 参数。

`AntObjectiveKind.HEADING_STABILITY` 把初始天空罗盘航向作为 typed task target，每拍只发布归一化
heading deviation 与前后误差改善形成的 `EnvironmentMeasurement`；它不告诉 controller 补偿方向，
仍经 outcome→PE→credit→Internal-RL。matched learned/no-opt 共享 distortion、seed、SSL/PE/rollout、
reflection writeback 与 reward→code bridge，只隔离 policy optimization 持久化。

预冻结 gate：`no_opt_late_error - learned_late_error >= 0.02` 且
`learned_recovery - no_opt_recovery >= 0.01`。2026-07-20 首个短预算
（18 ticks、tick 9 bias `+0.18→-0.18`）结果为 late advantage `0.0003`，两臂 recovery 均为负，
故诚实 verdict = **BLOCK**。不得把正号噪声写成恢复成功；在 gate 通过前不做宣传型 dashboard。

## 6. 冻结 claims 与 kill conditions

| 正式 claim | required arms / 最低证据 | kill condition |
|---|---|---|
| 单体 learned foraging | learned、no-optimize、PE-off、ETA-off、FixedRule、E2E-RL、random；≥5 seeds；held-out maps | learned 无 held-out 增益，或 reward 旁路 PE/credit |
| PE / ETA 因果贡献 | 同感知/预算的真实 PE-off 与 ETA-off；strict ETA 基于 ant traces | 以 random 代理消融，或策略参数未按预期改变 |
| 群体 bus 增益 | kernel-driven 独立 `AntSession` 的 bus-on/off + FixedRule bus-on/off；≥5 seeds | 共享 controller state，或 FixedRule 冒充 VZ 群体 |
| rare-heavy 角色分化 | per-individual `RareHeavyArtifact`、neutral/no-RH/shuffled/rollback；held-out 行为聚类 | 预置角色标签/手工 bias，或全员退化为单一簇 |
| 真实双 substrate | ant + 本地 HF，多 turn，fallback=DENY，hook fire rate≥0.75 | synthetic runtime、fallback>0 或声称共享 policy 权重 |
| 生物学参照 | AntBot/Ardin 权威数据资产、来源/图号/单位/sha256，含误差说明；homing 对标 AntBot 聚合 0.67%（非单条 0.47% cherry-pick），navigator 须含天空罗盘通道 | 合成衰减曲线冒充论文数据；用纯 dead-reckoning 对标带罗盘的 AntBot；靠调小噪声而非显式建模罗盘传感器过阈值 |
| 安全 veto | 完整 `AntSession`，learned/PE-off/chaotic checkpoint，固定延迟覆盖 | 只测 actuator 单元或任一 alarm 未 veto |

正式统计默认 ≥5 seeds、bootstrap CI、pairwise effects、训练/held-out 分离。门槛预先冻结；结果允许
`BLOCK`，不得为获得 `ACTIVE` 修改阈值。Phase 0/1/2 名称是工作流标签，不代表已经通过相应 claim。

## 7. 证明与演示

- **Matched-control（Workstream E）**：正式矩阵为全学习 / no-optimize / PE-off / ETA-off /
  FixedRule / end-to-end RL，random 仅作 floor。所有 arm 共享地图、seed、episode budget 与初始 checkpoint。
  `no-optimize` 必须是真实 Internal-RL 消融：与 learned 共享 PE/SSL schedule、rollout、reflection
  writeback 与 `internal_rl_runtime_modulation_strength`，只把
  `joint_apply_policy_optimization=False`，由 joint loop 在 SSL 后/RL 前 checkpoint、optimizer 后
  restore policy+critic；禁止再用 `joint_apply_writeback=False` 冒充 no-optimize（该开关只控制
  reflection/memory/regime consolidation，历史实现因此让两个 arm 都实际跑了 PPO）。
- **ant-active-evidence lane（Workstream F）**：复用 `evaluate_learned_active_candidate` gate 形态；
  替换 HF 绑定为 `:ant:` real-trace 定义与蚂蚁对照臂；产出
  `digital-ant-evidence-bundle.v2.json` + manifest。旧
  `learned-ant-promotion-evidence.v1` 只作历史输入，不再参与正式 verdict。
  gate 阈值本身 substrate-agnostic（`real_trace_turns>=500`、`validation_delta>=0.02`、
  `strict_eta`/`pe_off`/`eta_off`/`rollback`/`latency`/`safety`）。
- **可视化（Workstream G，`volvence_ant.viz`）**：
  - **G1 正式**：真实本地 HF runtime，fallback=DENY；synthetic G1 标记 `legacy/demo`。
  - **G2 正式**：trained `AntSession` vs FixedRule vs ScriptedBeeline vs random；现有
    FixedRule-vs-beeline 图标记 `legacy/demo`。
  - **G3 正式**：仅读带 provenance 的 AntBot/Ardin reference assets 与 multiseed artifact；
    合成 Ardin 曲线标记 `legacy/demo`。
  - **G4 正式**：alarm 通过完整 `AntSession` 闭环验证；直接调用 actuator 只作 unit smoke。
  - 统一脚本 `scripts/run_ant_demos.py` → `research/ant/results/g{2,3,4}_*.json` +
    `research/ant/figures/*.png`（matplotlib 为可选 `viz` extra；缺失时仅跳过图，仍产 JSON）。
  - **觅食剧场（`volvence_ant.viz.colony_theater`）**：面向直观演示的并排 colony 动画，
    左臂启发式 `FixedRuleAnt` 硬编码 FSM，右臂数字生命 kernel `AntSession`，共享同一
    `ColonyWorld` 信息素总线，中途食物搬迁。渲染为自包含 HTML+Canvas（零依赖，非
    matplotlib）。它只消费已发布的不可变事实——body 几何来自 `AntWorld` 公开 getter、
    信息素来自 bus 快照、行为标签来自各 controller 自己 record 上的 `mode`/`abstract_action`——
    不重建任何内核 owner 私有状态，不成为第二 owner，也不作为学习源。脚本
    `scripts/run_ant_theater.py` → `research/ant/figures/digital_ant_theater.html`；此
    lane 仅供演示，不产出正式 verdict。**诚实边界**：玩具尺度下 kernel 觅食投递数不敌手写
    FSM（与正式 matched-control 一致：`learned` 交付≈0、`fixed_rule`>0），此剧场展示行为而非
    宣称 kernel 觅食效率胜出。
  - **回巢剧场（`volvence_ant.viz.homing_theater`）**：展示被 AntBot 标度验证的诚实强项——
    路径积分导航（对齐 Phase 0 `homing_precision_experiment`，`passes_antbot_scale`）。并排两臂
    共享同一套外出随机游走，唯一差别是 `AntNavigator.compass_gain` 消融：完整路径积分（含天空
    罗盘）归一化回家误差 ≤ AntBot 0.67% 参照、几乎全员回巢；无罗盘死走朝向估计随步数漂移、
    回家信念偏离、迷路。每只蚂蚁画出「它以为家在哪」的信念箭头（把 navigator 的 egocentric
    home 投影到真朝向，纯几何、无位置读取）。可选第三面板复用**真内核**
    `route_learning_experiment`：固定路线反复走，可下降新奇度（认知型 PE）随曝光下降，记忆关闭
    对照不下降——这是记忆/PE 主链而非硬编码。脚本 `scripts/run_ant_homing_theater.py` →
    `research/ant/figures/digital_ant_homing_theater.html`；导航臂纯冻结 substrate（快），
    路线面板走内核（慢，`--no-route` 可跳过）。仅供演示，不改变正式 verdict。

### 7.1 公平训练与 checkpoint

- kernel arms 必须在训练地图上从同一个 owner-exported 初始 checkpoint 分叉，再把各臂训练后的
  checkpoint 导入 held-out 地图评估；禁止在 held-out 地图冷启动后称为 validation。
- formal matched-control 允许以 **seed 为唯一并行单元**使用 `spawn` 多进程；同一 seed 内各 arm
  仍顺序消费同一个初始 checkpoint。父进程必须按冻结 seed schedule 重新排序后聚合，worker 完成顺序
  不得改变 artifact、bootstrap CI 或 verdict。并行度是执行参数，不属于实验语义配置。
- checkpoint 由 `AgentSessionRunner.export_learning_checkpoint` 聚合各 owner 自己发布的
  temporal/Internal-RL、memory、PE heads、credit heads、regime、dual-track gate 与 reflection
  immutable state；embodiment 将其视为 opaque value，不遍历或重建 owner 私有状态。
- rollback drill 必须执行 `export → mutate → restore → fingerprint equality`，同 seed 重跑相同轨迹
  只能作为 determinism smoke，不能替代 rollback。
- ACTIVE evidence 必须记录实际 backend wiring。每一候选组件在隔离实验配置中真实
  `ACTIVE`，后继组件保持 `DISABLED`；证据脚本只给 candidate verdict，不改变生产默认配置。

### 7.2 一键演示与 replay

`scripts/run_ant_pipeline.py` 是统一 DAG 入口：

- `--profile demo --dashboard`：短预算实时 localhost Dashboard，并导出 replay HTML、GIF/MP4、
  JSON 与 manifest；允许 BLOCK。
- `--profile formal`：≥5 seeds、train/held-out、完整消融和 ≥500 real turns。
- formal 默认最多 5 个 matched-control seed worker；`--workers 1` 提供串行等价基线。
- `--resume` 只接受配置指纹一致、manifest 完整性验证通过的 stage。matched-control 每完成一个
  seed，就在 `.partials/matched_control/<fingerprint>/` 原子提交完整 report；partial 不保存 owner
  checkpoint 或私有可变状态。中断后只补跑缺失 seed，再由同一个纯聚合函数生成正式 artifact。
- stage verdict 为 `BLOCK` 不妨碍 resume：恢复只表示该计算完整可信，不代表 claim 通过。
- 可视化只消费不可变 `AntStepRecord` replay；位置、`z_t`、`β_t`、PE、credit、writeback 与
  backend wiring 均来自正式 turn 结果，不成为新的 runtime owner 或学习源。

### 7.3 NE-Dreamer next-embedding rare-heavy 对照（2026-07-20）

来源：NE-Dreamer（`arXiv:2603.02765`），见 `research/frontier-sweep-2026-07-20.md` §G1。
next-embedding prediction（预测下一 encoder embedding 而非重建当前感知）在 matched 参数量下
超过 DreamerV3，对"潜在状态应为未来可预测性服务"是直接支持——与本 spec 的 R-PE / R3
路线同向。据此登记一条 **rare-heavy 对照臂**（不改 runtime）：

- **定位**：`next-embedding` objective 只能作为 rare-heavy baseline arm 进入 matched-control
  矩阵，与既有 learned / PE-off / ETA-off / FixedRule / E2E-RL 臂同地图、同 seed、同预算对比；
  目的是回答"frozen encoder + bounded controller 是否保留 NE-Dreamer 式收益"。
- **硬边界**：NE-Dreamer 原文端到端联合训练 encoder / world model / actor-critic，直接照搬
  违反 R2 冻结基底；对照臂必须保持 `sense_encode` / `motor_decode` 冻结，next-embedding loss
  只允许训练 bounded controller 侧组件。
- **不得绕过 lineage**：next-embedding loss 不等于 task outcome PE；该臂的任何"学习发生了"
  claim 仍必须经 outcome → PE → credit lineage 结算，不得以 embedding-loss 下降直接冒充
  foraging 增益（对齐 §6 kill condition "reward 旁路 PE/credit"）。

## 8. 回滚

整包是新增独立 owner，回滚 = 移除包 + 撤销 DATA_CONTRACT §2.19 / 本 spec / workspace 注册。
不触及任何内核 owner，故对主线零风险。

## 9. Artifact provenance

正式脚本使用 `digital-ant-evidence.v2` payload 与 `digital-ant-manifest.v2` sidecar。manifest 至少记录
git SHA/branch/dirty、Python/依赖版本、seed schedule、config digest、model fingerprint，以及所有输入和
输出文件的 sha256/size。校验失败必须 fail loudly。dirty tree 可产生内部证据，但
`externally_retainable=false`，不得对外声称 retain。

artifact 与 manifest 均通过临时文件 + `os.replace` 原子提交，且 manifest 最后落盘，是正式 bundle 的
完成标志。pipeline stage 还需原子 marker 绑定语义命令指纹与全部 output manifest；仅存在 JSON 文件
不足以跳过计算。旧 runner 启动的进程不会追溯生成 partial/marker，新恢复协议只对升级后启动的 run 生效。
