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
| `sense_encode` | `WorldObservation` + `NavigatorState` + 显式 schema | `ant-sense.v1` 固定 14 维；`ant-sense.ecology-v2` 在尾部追加 5 个局部热感通道 | 纯 numpy，无可学习参数；v1 默认不变 |
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

`ant-sense.ecology-v2` 只追加 `heat_left / heat_right / heat_diff / heat_center /
heat_harmful`。热值来自触角和身体位置的局部采样；对象坐标、火柴方位、逃离方向和木棍几何均不进入
substrate。`ant-sense.v1` 继续绑定原 14 维 `SENSE_CHANNELS_V1`，历史 evidence 不得静默升级。
`AntSession` 必须通过 runtime facade 把当前 sense schema 的完整宽度声明为 temporal `n_input`；
ecology-v2 因此以 19 维进入 ndim encoder，再压缩到独立的 latent `n_z`。不得把 `n_input` 绑定到
`n_z` 或截断携食、障碍、热感等尾部通道；archive compatibility 同时绑定 sense schema、input dim
与 latent dim。

## 4. Outcome → PE 接缝（正式证据）

`semantic_*_pull` 只表示 substrate 发布的感觉/动机预测通道，不能冒充环境任务结果。正式行为学习链路为：
`AntWorld` 发布 typed、不可变且只含可观察事实的 `EnvironmentOutcome.measurement`；runtime facade 保留
event/prediction/action lineage，并在下一 turn 只交给 PE owner；PE 形成 signed mismatch，credit 再将其
归因给 β segment，Internal RL 只读 PE/credit。

`AntObjectiveKind.FORAGING` 有且只有两个真实、离散、可观察的里程碑：**pickup**
（`carrying_food` False→True）与 **delivery**
（终局回巢投递）。`AntSession._environment_outcome` 只在这两个事件上发布 measurement——pickup 给部分
`task_progress=0.5`（非终局），delivery 给 `task_progress=1.0`（终局）。**明确禁止**发布任何"到食物距离
越近奖励越高"的连续势能塑形（potential-based / distance shaping）：连续的 `closer=better` 信号等于把
FSM 手写的梯度跟随答案直接喂给控制器，而"如何朝食物走"恰是控制器**必须自己学**的技能。历史上曾短暂
引入过基于 `food_center` / `eval_home_distance` 的连续势能塑形，已移除（既有泄露方向策略的风险，且经点火
探针验证：即便 credit 变密，运行时 z_t 仍零变化，塑形无助于点火——断点在内核"学习→运行时控制输出"应用
路径，不在此接缝的信号密度）。

`AntObjectiveKind.ECOLOGY` 在同一 lineage 上增加三个稀疏环境事实：木棍真实碰撞、超过火柴有害阈值、
以及从有害热区脱离；它们分别产生负 payoff / 恢复 payoff。该 objective 仍只消费动作后的物理事实，
不消费对象距离、热源方位、推荐转向或 evaluation readout；黄油 pickup/delivery 继续使用上述里程碑。

禁止 `AntWorld` 直接传 reward 给 temporal/Internal RL，禁止 evaluation 反灌 reward，禁止 runtime
另建 mismatch slot。历史上只依赖 drive PE 的结果可以保留为机制 smoke，但不是 learned foraging 证据。

这里的“只有两个 task milestone”不等于系统其余 PE 必须为零：局部 food/pheromone/heading 感知仍会
产生 substrate prediction mismatch，Internal-RL 可把它作为内在 PE 信号。它不读取食物坐标或全局距离，
不是 task shaping；但因此 learned-vs-PE-off 只能证明“含内在 PE 的完整架构贡献”，不能写成
“pickup/delivery 奖励单独导致学习”。若未来需要 milestone-only RL 因果结论，必须新增 typed reward
eligibility 并逐条审计 nonzero reward 的 outcome lineage，不能在 evaluation 侧靠文本/数值猜测。

数字蚂蚁实验显式设置 `internal_rl_runtime_replay=ACTIVE`，把每拍真实 substrate、实际 `z_t`、下一拍
substrate effect 与匹配的 outcome→PE→credit lineage 结算成 Internal-RL batch。该开关只选择 transition
source，不选择 pure/torch optimizer；生产默认仍为 `DISABLED`。ACTIVE 样本不足时必须报告
`waiting-for-runtime-replay`，禁止回退 synthetic。`commanded_turn/applied_turn` 仅保留在 embodiment
审计记录中，不进入 optimizer 作为补偿方向提示。

为打破“零 pickup → 零任务 PE → 零强化”的稀疏探索死锁，ant evidence profile 可显式设置
`internal_rl_runtime_exploration_strength`。该 gate 默认 `0.0`（生产字节等价回滚）；稀疏觅食/
群体 matched arms 统一为 `1.0`，已有 dense PE 的 heading-stability calibration 保持 `0.0`。
temporal owner 把原 sample noise 与可复现的 low-discrepancy sample 混合，并发布以
`0.4 * strength` 为下界的 effective posterior std，防止首个 milestone 前方差塌缩；
探索只能改 sample residual，不得在 continuation/coast 阶段压平 learned posterior mean，
因此 food/heat/home 等 state-conditioned mean 在训练动作中始终可达。它不读取食物位置、不编码补偿方向；实际 std / sample noise / `z_tilde` 进入
runtime state，故 runtime replay 仍可重建 likelihood。同一实验内 learned/no-optimize/PE-off
共享同一探索。Ant runtime 以 session seed（colony 中已包含 body offset）作为不透明 exploration
context；固定 schedule 的每个 episode 使用不同 seed，因此 episode/body 的 option 序列分散，
而 matched arms 在相同 episode/body 上仍严格相同。temporal owner 只保留 context 摘要，不读取
session label 或环境语义。未提供 context 的通用 runtime 保持历史探索序列不变。
它不能单独构成 learning claim，只负责让真实稀疏里程碑有被采到的机会。

ecology 的 frozen evaluation 必须同时设置 `joint_learning_enabled=False` 与
`apply_policy_optimization=False`。前者是 joint-loop owner 的硬边界：即使训练 checkpoint
恢复出 pending replay/batch，调度器也只能发布 `frozen-evidence-only`，不得执行 SSL、Internal
RL、writeback 或 rare-heavy 路径；同一个 flag 还必须关闭 temporal owner 的 fast fit、
action-family outcome/topology/cache 与 learned match-head 写入，但不阻止恢复 checkpoint 或
更新推理 telemetry。后者不能替代此前者。每个 held-out layout 都要比较前后
`temporal_learning_fingerprint`，任一 body 漂移即阻断能力晋级。

诊断 evidence 还必须区分 owner checkpoint 的 `policy_fingerprint`、`temporal_fingerprint` 与
`memory_fingerprint`，并发布训练/held-out pickup、delivery、首次接触、最小食物距离、turn/
`z[0]-z[1]` 分布、switch rate、非零 reward 与 runtime replay lineage/settled coverage。消费者
只读取这些 owner 导出的值，不遍历 joint-loop 私有状态。`policy_fingerprint` 组合
Internal-RL owner 在 world/self checkpoint 发布的 `policy_optimization_fingerprint`；后者只覆盖
optimizer-owned update step 与 critic 参数，不把共享的 temporal SSL / reflection prior
track-weight 变化误报成 RL 持久化。actor/结构的完整变化仍由 `temporal_fingerprint` 负责。

## 5. 三层生物学 ↔ VZ 对齐

| 生物学层 | VZ 对应 | 在数字蚂蚁中 |
|---|---|---|
| 基因组（不变） | frozen substrate | `sense_encode` / `motor_decode` / `AntNavigator` |
| 基因表达程序（窗口期，离线） | rare-heavy artifact refresh | Phase 2 角色重编程离线循环，产出个体倾向性初始化参数，运行时不可触发 |
| 突触可塑性（在线） | online-fast controller | `z_t` / `β_t` + CMS 在线学习（内核承担） |

### 5.1 隐藏电机扰动校准（experimental / frozen gate rerun required）

`MotorDistortionProfile` 是环境 owner 的不可变 actuator transfer：
`applied_turn = clamp(gain·commanded_turn + bias)`，可在 `switch_tick` 一次性切到另一组
gain/bias；空 profile 是严格 identity，单 profile 可广播、多 profile 必须与 body 数一致。公开
`WorldTransitionEvidence` 只增加外部可观察的 `commanded_turn / applied_turn`，不泄露 profile 参数。

`AntObjectiveKind.HEADING_STABILITY` 把初始天空罗盘航向作为 typed task target，每拍只发布归一化
heading deviation 与前后误差改善形成的 `EnvironmentMeasurement`；它不告诉 controller 补偿方向，
仍经 outcome→PE→credit→Internal-RL。matched learned/no-opt 共享 distortion、seed、SSL/PE、
ACTIVE runtime replay、reflection writeback 与 reward→code bridge，只隔离 policy optimization
持久化。

冻结 gate：`no_opt_late_error - learned_late_error >= 0.02` 且
`learned_recovery - no_opt_recovery >= 0.01`。接入真实 runtime replay 后，2026-07-20 曾按预声明预算
（60 ticks、tick 30 bias `+0.18→-0.18`、seeds `0..4`、`n_z=16`）得到 mean late advantage
`0.0628`（bootstrap 95% CI `[0.0082, 0.1255]`）、mean recovery advantage `0.0735`
（95% CI `[0.0073, 0.1520]`），旧聚合 gate verdict = **PASS**。逐 seed 仍只有 3/5
通过单次 effect-size 门（seed 1/2 为 BLOCK），所以结论限于“聚合上 learned 优于 matched
no-optimize”，不能写成每个个体都稳定恢复。证据：
`research/ant/results/motor_calibration.v1.json` + manifest；当前 dirty-tree provenance
使其 `externally_retainable=false`。动态群体 v1 审查又发现旧 `last_turn_command` 实际发布的是
hidden applied turn，违反“plant transfer 只作审计”的冻结边界；现已改为只发布 commanded-turn
efference copy，并移除 heading outcome detail 中的 signed actuator delta。因此上述 PASS 只能作为
历史诊断，当前正式状态回退为 **RERUN REQUIRED / BLOCK**；必须在修正后按同一 5-seed 预算重跑，
且 clean-tree artifact 通过，才可恢复该 claim。

### 5.2 动态群体适应基准（experimental / v1 contract frozen）

`Dynamic Stigmergy Regime-Shift Benchmark v1` 分开检验三件事：信息素是否产生群体协作、
kernel controller 是否发生在线适应、环境规律变化后 learned 是否优于强 FSM。三种结论不得互相
替代：FixedRule bus-on 胜 bus-off 只能证明 embodiment 的 stigmergy；learned 在隐藏电机偏置上
恢复只能证明个体控制适应；只有 held-out regime shift 的 matched aggregate gate 通过，才允许声称
Digital colony 比强 FSM 更能适应变化。

环境 owner 新增不可变 `AxisAlignedObstacle`。障碍几何只由 `AntWorld` / `ColonyWorld` 持有，
个体只能通过左右触角占用值与上一动作 contact bit 感知；碰撞采用连续线段/AABB entry 计算，薄墙
不能被单步穿越。`WorldTransitionEvidence` 发布 commanded/applied step、blocked flag 与 obstacle id
供只读审计；障碍参数和未来变化不进入 substrate。运行中激活障碍若覆盖现有 body，不瞬移 body：
已在障碍内部的 body 只允许离开，离开后不得重新进入。`set_obstacles` 只在 round 边界使用。
Pheromone owner 同时在不可变 `PheromoneField` 中发布 home/trail mass 与 normalized entropy；
实验消费者不得遍历可变 bus 内部格网重建这些摘要。

v1 每个 seed 从同一 owner-exported 初始 checkpoint 分叉，各 arm 使用同一对应地图、body 数、
evaluation ant-action 预算、扰动时刻与 frozen substrate；所有 kernel learning arms 另共享相同
training ant-action 预算，Frozen FSM/random 没有可训练参数，不把空跑冒充训练。正式 arms：

- `learned_bus`、`learned_no_bus`
- `no_optimize_bus`（保留 PE/SSL/replay/writeback，只关闭 policy optimization 持久化）
- `pe_off_bus`
- `fixed_rule_bus`、`fixed_rule_no_bus`
- `random_no_bus` floor

训练与评估不得重放同一随机世界：training food 固定在东侧，evaluation food 旋转到北侧，
training/evaluation world seed 分别由公开的不同派生函数产生；controller seed 在各 arm 间仍按 body
对齐。正式 artifact 必须同时记录两个派生 seed。learned/no-optimize 在 shift 前的 throughput CI
必须对齐。runtime replay 存在一 turn settlement 延迟，因此 shift 后第一个 turn 是显式 latency
boundary：它结算最后一个 pre-shift outcome 并执行第一个 post-shift action；此后导出的
`adaptation_start` policy checkpoint 尚未消费任何 post-shift outcome。learned 的
`adaptation_start→final` policy fingerprint 必须变化而 no-optimize 保持不变；否则 post-shift
差异不能归因为在线适应。最后一个 post-shift action 的 outcome 尚未结算，不进入该 fingerprint
因果窗口。

当前 `no_optimize` 从 training 起即关闭 policy optimization，因此它严格支持的是
**learning-lifecycle contribution**，不是“在同一个 shift 世界状态处分叉”的单点因果结论。
pre-shift performance alignment + `adaptation_start→final` fingerprint 只负责排除最明显混淆；若要声称
“post-shift update 本身导致恢复”，必须另加 environment/bus/navigator/runtime 全状态
checkpoint 后的 frozen-at-shift arm，未落地前不得扩大口径。

每个 regime shift 独立成 episode，禁止把多个变化混在主效应判断中。v1 包含：
`obstacle_block`（新障碍阻断原路线）、`food_relocation`、`motor_bias`；动态 pheromone decay 与
sensor drift 留给 v2 单独收敛包。信息素的“未携带铺 home / 携带铺 trail”仍是公共 frozen physiology，
所以 v1 只声称路线协作涌现，不声称通信语义、角色或语言本身是学出来的。

只读评估至少发布：pre/post-shift 每千 ant-action 投递、delivery curve/AUC、rolling window
恢复到 pre-shift 全阶段吞吐 80% 的 round 数、碰撞数、route stretch、trail mass/normalized entropy curve、bus pairwise effect、
learned-vs-no-optimize/PE-off/FSM pairwise effect、runtime replay lineage 与 policy fingerprint。
evaluation 不回灌 reward。

正式统计冻结为 ≥10 paired seeds、paired bootstrap 95% CI。2026-07-20 只使用 FixedRule 做
horizon 校准（没有查看 learned 结果）：`n_ants=8` 时 50 pre-shift rounds 已有 9/10 seeds
接触并完成投递。因此正式最低预算冻结为 8 ants、每蚁 200 training rounds、50 pre-shift rounds、
100 post-shift rounds、20-round recovery window；低于任一预算只能是 smoke，必须 BLOCK。
promotion gate：

1. **运行预算**：满足上述 colony size / training / pre / post / recovery window 与 seed 数；
2. **静态资格**：learned pre-shift throughput ≥ fixed-rule 的 80%，且至少 80% seeds 发生 pickup；
3. **群体因果**：`learned_bus - learned_no_bus` 的 post-shift throughput effect ≥ 0.02/千
   ant-action，CI 下界 > 0；
4. **学习生命周期贡献**：learned 相对 no-optimize 和 PE-off 的 post-shift throughput effect 均
   ≥ 0.02/千 ant-action，CI 下界 > 0，且 learned/no-optimize 的 pre-shift throughput CI 对齐；
5. **适应优势**：learned 相对 fixed-rule 的 recovery time 至少缩短 20%、相对考虑
   nest/pickup contact radius 的 optimistic oracle delivery shortfall 至少降低 15%、
   post-shift throughput 至少提高 10%，三个 paired CI 均排除 0；
6. **完整性**：runtime replay settled/lineage coverage ≥ 0.99，arms 的初始 checkpoint
   fingerprint 数量等于 ant 数且逐 body 对齐；每个 kernel arm 满足
   `0 ≤ lineage_matches ≤ settled ≤ captured`、`settled/captured ≥ 0.99`、
   `lineage_matches/settled ≥ 0.99`，且 full episode 与 post-shift slice 都逐 ant 检查、
   drop reasons 为空、存在真实 transition、wiring=ACTIVE。双轨回放的绝对预算也必须精确匹配：
   full evaluation 每 ant 的 `captured=2×(pre+post−1)`、
   `settled=transitions=lineage_matches=2×(pre+post−2)`；以第一个 post-shift latency-boundary
   record 为基线的 post slice 四项均为 `2×(post−1)`。learned 的
   `adaptation_start→final` policy fingerprint 分叉、no-optimize 不分叉；障碍激活区域或食物迁移后的 pickup
   radius 不得与任何 arm 的 body 重叠。

任一静态资格失败，整体直接 **BLOCK**，不得用后续偶然效果宣称“优于 FSM”。v1 的 dashboard/theater
只消费 artifact/replay；在正式 gate 通过前必须显示 BLOCK。resume fingerprint 还必须折入 benchmark
完整 Python 依赖闭包的内容摘要，使 dirty-tree 期间的代码变化不能静默复用旧 seed partial。
三个扰动必须各出现一次；单场景 smoke 即使自身门槛通过，suite verdict 仍必须 BLOCK。

### 5.3 三物体 ecology-v2（experimental / promotion-gated）

- `AntWorld` 是黄油、木棍和燃烧火柴的唯一 owner。`ButterSource`、`WoodStick`、
  `BurningMatch` 及其 `WorldObjectSnapshot` 均为 frozen value；增删、平移和替换只在完整
  tick/round 边界原子应用。
- **三物体价态语义（冻结）**：黄油是 appetitive（趋近/拾取/交付产生正学习信号）；燃烧火柴是
  aversive（有害热暴露产生负学习信号，脱离/降温为正）；木棍是 **neutral 物理约束**——仍不可
  穿越、仍进入 obstacle 感知通道，但接触**不产生任何 payoff/valence**，绕行不是学习目标，
  contact 计数只作诊断。三种预期行为都必须由 learned `z_t` policy 产生，环境只提供价态与
  可观察事实。
- 木棍是任意方向 capsule，碰撞用连续 segment/capsule entry 求交，禁止高速穿透；火柴发布指数衰减
  热场和 owner 计算的有害半径，控制器只看局部热样本。环境 owner 自己生成渲染 description，
  App 不遍历或重建对象内部参数。
- `WorldTransitionEvidence` 由环境 owner 发布动作前后的局部 food、home-pheromone 与 heat signal，
  以及 pickup/delivery/contact/threshold crossing。`AntSession` 只把这些可感知变化压成有界
  `EnvironmentMeasurement.action_payoff`：未携食时 food 改善、携食时 home signal 改善、降温/
  逃逸/pickup/delivery 为正，升温/有害暴露为负。木棍 contact 是可观察事实（进入 status/
  evidence），但**不进入 payoff**；纯 contact 且无其他价态事件时不构造 measurement。该先天
  价态不含坐标、目标方位或推荐转向；路径、绕行、回巢和逃逸动作仍必须由 learned `z_t` policy
  产生。显式 `ecology_local_valence_enabled=False` 是 matched ablation，不得在正式 learned arm
  静默关闭。
- Digital Ant 正式 runtime profile 将 `internal_rl_runtime_segment_credit=ACTIVE`：joint-loop owner
  把 lineage-matched one-step replay 按真实 `beta_t` switch 边界聚成 segment，并在 milestone/
  terminal 或 16-step 上限处强制闭合；PPO/critic 对闭合 segment 的多步 transition 运行同一 GAE。
  ecology 单局是 24 turns，但首拍只有 capture、没有 preceding settlement，因此最多只有 23 个已
  结算 transition。上限必须为后续同局 scheduled step 留出“闭合→optimizer”窗口；使用 24 会让
  无 pickup/switch 的 open segment 在跨 episode `include_runtime_replay=False` checkpoint 前从未
  进入 optimizer，形成“零 milestone→零更新→零 milestone”的确定性死锁。
  World/Self 两轨 metacontroller 独立切换，segment 边界取**任一轨** `beta_t` switch（与 milestone/
  terminal 闭合的 OR 语义一致）；分叉切换只会让 segment 更短，两轨打包保持逐拍成对对齐。
  open segment、closed segment 和最长长度进入 owner checkpoint/rollback，但不新增 ledger 或
  runtime slot；DISABLED 精确回到历史 one-step replay。
- ecology evidence profile 还将通用 `internal_rl_causal_action_head=ACTIVE`：低秩 head 只把
  posterior hidden state 映射为 bounded `z_t` residual，以补足逐维 track gain 不能表达的
  state-conditioned 左右响应。head 不含 butter/stick/match 字段、不直接生成 turn/step；参数由
  temporal/Internal-RL owner checkpoint、canonical archive、fingerprint 和事务 rollback 管理。
  Ndim GRU hidden 以 signed `[-1,1]` 坐标进入 head，禁止再按 `[0,1]` 重心变换。通用默认仍为
  `DISABLED`，`SHADOW` 是不改变 live code 的候选评估路径。常数 bias 的总幅度限制为 `0.1`，
  学习尺度为 state path 的 `0.05`；batch mean 只更新该 bias，低秩 factor 只消费 centered
  gradient，避免单个重复探索序列把近场偶然收益固化为显式或隐式的跨状态固定转向。
- `ecology_curriculum` 对黄油、火柴、组合三阶段分别执行 near → medium → far mastery：
  每阶段达到预声明 pickup/delivery/heat-entry/escape 样本量且满足最少 episode 后才提前晋级，
  未达到则在最大预算处显式 BLOCK；已经掌握的阶段按固定频率交错回放。木棍不再是独立训练
  阶段——它作为中性物理几何出现在组合布局中，不设 contact mastery。跨 episode 仅携带
  owner-exported checkpoint。learned、no-optimize、local-valence-off 与 segment-credit-off 从同一
  初始 checkpoint 分叉，并重放 learned 冻结下来的完全相同训练布局/seed 日程。长训练启动前必须
  先通过 food/heat 成对探针（输入可达且转向可区分）；obstacle 成对探针只要求输入可达
  （中性几何进入感知即可，不要求驱动转向）；探针失败时拒绝投入后续预算。
- training、validation、正式 held-out 使用三个不重叠的 seed/布局命名空间。held-out 拆成
  butter-only、butter-with-neutral-stick、burning-match route avoidance、burning-match forced
  escape 和 composite 五类：butter-with-neutral-stick 只验证有中性物理阻挡时觅食仍成功（不比
  contact 率、不与 no-optimize 比"回避能力"）；主动避热不要求先进入热区；强制逃逸从有害区内的
  受控起点单独测 escape/harmful ticks。评估将 joint learning 完全切到 evidence-only，policy 与
  temporal-learning fingerprint 必须全程不变（PE 驱动的 turn-local mixture 不计入
  temporal-learning fingerprint）。
- 正式 gates 还冻结完整 19 维 channel activation（必需激活集合为 food/heat 家族 +
  `obstacle_left/right`，不强制 `obstacle_contact`——不得为过门强迫碰撞）、最近对象距离、首次
  事件 tick、局部 ecology payoff 数、switch/persistence、闭合 segment 长度、food/heat 成对左右
  action sensitivity（obstacle 只作 input-reachability 诊断）、动作平滑度、archive roundtrip/事务
  rollback 以及 runtime replay settled/lineage ≥ 0.99。综合 outcome score 与 composite/
  matched-ablation 比较不含 obstacle-contact 惩罚项。外层 bundle/report schema 为
  `digital-ant-ecology-checkpoint.v4` / `digital-ant-ecology-curriculum.v3`（v3/v2 是木棍仍作
  回避目标的历史语义，其 artifact 只作诊断，loader 必须拒绝）。
  settlement coverage 的分母是 `captured - pending_capture_count`；episode 尾部尚无下一状态的
  capture 被明确发布为 pending，不得误报为 drop，也不得借此忽略真实 drop reason。
  任一失败即 `BLOCK`；不得加载为 demo checkpoint，也不得用 `FixedRuleAnt` 或 Canvas 脚本伪装通过。

2026-07-20 小预算 smoke（`n_ants=1, n_z=4, stage_rounds=8, heldout_rounds=16,
seed=19`）结论为 **BLOCK**：learned policy fingerprint 已变化、no-optimize fingerprint 保持稳定，
四类 held-out 的 eligible settlement/lineage 均为 `1.0/1.0` 且 drop=0；但 learned/cold/no-optimize
均为 0 pickup、0 delivery，木棍接触、火柴有害暴露/脱离也全为 0，组合门同样无行为证据。因此本阶段
断点是对象 encounter/任务行为覆盖，而不是 replay lineage、policy 持久化或 checkpoint 恢复断链。
artifact：`research/ant/results/ecology_checkpoint_smoke.v2.json` + manifest；该 BLOCK archive 仅供
诊断，App loader 必须拒绝。

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
  writeback、`internal_rl_runtime_replay=ACTIVE` 与 `internal_rl_runtime_modulation_strength`，只把
  `joint_apply_policy_optimization=False`，由 joint loop 在 SSL 后/RL 前 checkpoint、optimizer 后
  restore policy+critic；禁止再用 `joint_apply_writeback=False` 冒充 no-optimize（该开关只控制
  reflection/memory/regime consolidation，历史实现因此让两个 arm 都实际跑了 PPO）。
  `eta_off` 使用 frozen learned-lite 且 `ssl_interval=rl_interval=0`；learned-lite 由 temporal owner
  把 legacy readout 投影到配置 `n_z`，因此它与其他 kernel arm 一样保留
  `internal_rl_runtime_replay=ACTIVE` 的严格维度与 lineage 审计，不靠关闭 replay 隐藏契约错误。
  2026-07-20 正式预算（seeds `0..4`、train `200`、held-out `60`、`n_z=16`）结果仍为
  **BLOCK**：learned 仅 seed 1 在 training pickup/delivery，held-out pickup/delivery 均为
  `0/5`，`validation_delta=0.0 < 0.02`；
  FixedRule 为 `5/5` delivery。每个 kernel arm 的每 seed replay 均有 `514` settled/
  lineage-matched transitions，learned policy optimization fingerprint 分叉而 no-optimize 保持不变，
  所以当前断点是探索接触与 held-out 泛化，不是 replay 或 policy writeback 断链。证据：
  `research/ant/results/matched_control.json` + manifest；dirty-tree artifact
  `externally_retainable=false`，只能作本轮本地冻结结论。
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
- 持久化使用 `agent-learning-archive.v2` 单体 archive +
  `agent-learning-checkpoint-collection.v1` 多 ant envelope：每个 owner 先发布自己的
  `OwnerPersistenceSnapshot`，runtime facade 只做 strict canonical UTF-8 JSON、逐 part sha256、
  整体 state fingerprint 与有序 compatibility（sense schema / input dim / latent dim / ant count）绑定。
  格式禁止 pickle、动态类型名、object hook、未知字段、缺失字段、重复 JSON key 与非有限数。
  恢复时先完整校验 envelope/owner set/version，再经 owner hydration API 应用；任一晚期 owner 失败时
  单 ant 和 colony 都恢复 preimage 并复核 fingerprint，rollback 失败则同时抛出原错误与回滚错误。
  colony 仅按逆序回滚截至失败 body 的 attempted prefix，尚未尝试的 suffix 不得执行 restore 或清空
  owner 的瞬态窗口。
  colony archive 的 checkpoint id 必须与有序 `body:{index}` 映射一致且全局唯一，交换、重复或数量
  不符均在任何 owner 变更前拒绝。
  sha256 只提供完整性而非来源认证；HTTP 仍不提供 archive 上传入口，若未来开放外部导入必须在 JSON
  decode 前增加签名验证。
  旧 `agent-learning-checkpoint.v1` pickle 不提供迁移解码器，也不能自动提升为 v2；必须从 owner
  checkpoint 重新导出或重新训练，避免为了兼容而再次执行不可信对象反序列化。
- training→held-out transfer 使用 owner 的 `include_runtime_replay=False` 导出模式：保留已学习参数，
  但不迁移未结算 capture、staged rollout 或 episode-local replay 计数，避免把 training action 与
  held-out outcome 错配；同一 episode 内的 shift/adaptation/final audit checkpoint 仍包含 replay。
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

### 7.2.1 实时实验场（`digital-ant-app.v2`）

`volvence_ant.app` 是 `vz-embodiment-ant` 内的外部实验控制面，前端位于同 wheel 的 `web/`
React/Vite 工程。它不新增 kernel slot：

- Python app runner 串行调用真实 `AntSession.step()` / `KernelColonyRunner.step_round()`；只有完整
  tick/round 完成后才发布 immutable frame，禁止预烘焙 replay 冒充 live。
- 下行是 SSE frame/status/disturbance event，上行是 POST config/command/disturbance。pause/resume/
  single-step/speed 只控制编排节奏；schema 明确不含 `turn_command` / `step_command` 写入口。
- 食物搬迁、alarm、电机 transfer 及 typed `upsert/move/remove_world_object` 只在环境 owner 的
  tick/round 边界应用。操作者可配置隐藏 gain/bias，但这些参数不进入 substrate/frame；agent 仍只
  看到真实物理后果。
- 黄油和火柴点击放置，木棍拖拽定义方向/长度；选择后可平移或删除。浏览器只发送 typed 环境扰动，
  `AppFrame.objects` 直接携带 owner 发布的不可变 `WorldObjectSnapshot`，不允许提交电机动作。
- Canvas 只投影 `AntStepRecord`、`ColonyRoundRecord`、公开 body/food getter、对象/信息素快照；
  慢客户端可以丢旧视觉帧，命令、扰动审计和 replay 不丢。
- PASS/BLOCK 只读正式 evidence artifact，永不作为学习输入。没有通过冻结门槛时默认明确显示
  `BLOCK`，即使真实闭环正在运行；formal evidence verdict 与 ecology checkpoint 的
  loaded/fingerprint/promotion verdict 分栏显示，动画流畅不构成科学 PASS。

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
