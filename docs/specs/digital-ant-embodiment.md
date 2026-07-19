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

`z_t` 契约（egocentric 抽象动作）：`z[0],z[1]` → egocentric 期望朝向单位向量；`z[2]` → 期望速度（squash）。
controller 自由学习「感知特征 → egocentric 动作」的映射；`motor_decode` 只做有界转换。**策略在可学习的内核，plant 冻结在此。**

`AntNavigator`（body 侧，冻结）维护环形吸引子朝向估计 ĥ 与路径积分回巢向量（对应中央复合体）。
正式证据必须让 world 真值运动噪声与 navigator estimate 独立，并只由真实 `world.act` 产生轨迹；
历史上以 noisy estimate 推进真值或通过 `set_body_pose`/`sync_to` 构造的 lane 仅是 legacy smoke，
不能用于 AntBot/Ardin claim。

## 4. Outcome → PE 接缝（正式证据）

`semantic_*_pull` 只表示 substrate 发布的感觉/动机预测通道，不能冒充环境任务结果。正式行为学习链路为：
`AntWorld` 发布 typed、不可变且只含可观察事实的 `EnvironmentOutcome.measurement`；runtime facade 保留
event/prediction/action lineage，并在下一 turn 只交给 PE owner；PE 形成 signed mismatch，credit 再将其
归因给 β segment，Internal RL 只读 PE/credit。pickup 是 lineage，delivery 是默认稀疏终局事实。

禁止 `AntWorld` 直接传 reward 给 temporal/Internal RL，禁止 evaluation 反灌 reward，禁止 runtime
另建 mismatch slot。历史上只依赖 drive PE 的结果可以保留为机制 smoke，但不是 learned foraging 证据。

## 5. 三层生物学 ↔ VZ 对齐

| 生物学层 | VZ 对应 | 在数字蚂蚁中 |
|---|---|---|
| 基因组（不变） | frozen substrate | `sense_encode` / `motor_decode` / `AntNavigator` |
| 基因表达程序（窗口期，离线） | rare-heavy artifact refresh | Phase 2 角色重编程离线循环，产出个体倾向性初始化参数，运行时不可触发 |
| 突触可塑性（在线） | online-fast controller | `z_t` / `β_t` + CMS 在线学习（内核承担） |

## 6. 冻结 claims 与 kill conditions

| 正式 claim | required arms / 最低证据 | kill condition |
|---|---|---|
| 单体 learned foraging | learned、no-optimize、PE-off、ETA-off、FixedRule、E2E-RL、random；≥5 seeds；held-out maps | learned 无 held-out 增益，或 reward 旁路 PE/credit |
| PE / ETA 因果贡献 | 同感知/预算的真实 PE-off 与 ETA-off；strict ETA 基于 ant traces | 以 random 代理消融，或策略参数未按预期改变 |
| 群体 bus 增益 | kernel-driven 独立 `AntSession` 的 bus-on/off + FixedRule bus-on/off；≥5 seeds | 共享 controller state，或 FixedRule 冒充 VZ 群体 |
| rare-heavy 角色分化 | per-individual `RareHeavyArtifact`、neutral/no-RH/shuffled/rollback；held-out 行为聚类 | 预置角色标签/手工 bias，或全员退化为单一簇 |
| 真实双 substrate | ant + 本地 HF，多 turn，fallback=DENY，hook fire rate≥0.75 | synthetic runtime、fallback>0 或声称共享 policy 权重 |
| 生物学参照 | AntBot/Ardin 权威数据资产、来源/图号/单位/sha256，含误差说明 | 合成衰减曲线冒充论文数据 |
| 安全 veto | 完整 `AntSession`，learned/PE-off/chaotic checkpoint，固定延迟覆盖 | 只测 actuator 单元或任一 alarm 未 veto |

正式统计默认 ≥5 seeds、bootstrap CI、pairwise effects、训练/held-out 分离。门槛预先冻结；结果允许
`BLOCK`，不得为获得 `ACTIVE` 修改阈值。Phase 0/1/2 名称是工作流标签，不代表已经通过相应 claim。

## 7. 证明与演示

- **Matched-control（Workstream E）**：正式矩阵为全学习 / no-optimize / PE-off / ETA-off /
  FixedRule / end-to-end RL，random 仅作 floor。所有 arm 共享地图、seed、episode budget 与初始 checkpoint。
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
