# Ecology P0/P1 实施状态

> 更新时间：2026-07-23。P0 已完成；P1 runner、诊断矩阵与 home-action probe 已实现；P2 因 P1 尚未 PASS 而未启动。

## P0 正式结果

正式配置为 4 ants、latent 16、每阶段 3 episodes × 12 rounds、冻结评估 24 rounds。

- verdict：`PASS`
- action chain：19/19 checkpoint snapshots 通过；食物/热源保持左右转向敏感性。
- temporal：31 次真实 switch；25 次 `beta-switch` closure；30 次 `environment-milestone` closure。
- frozen evaluation：butter 与 forced-escape 两场景的 policy/temporal-learning owner 均保持不变。
- replay：settlement=1.0、lineage=1.0、drop=0。
- action-chain guard：18 个会破坏群体敏感性的候选训练更新被拒绝并回滚；失败证据保留在 segment telemetry 中。

正式报告：[`results/ecology_recovery/p0/ecology_mechanism_audit.v1.json`](results/ecology_recovery/p0/ecology_mechanism_audit.v1.json)。

## P1 已实施内容

- curriculum schema 从 v3 升至 v6，P1 development report 升至 v5；旧 schema 被 checkpoint loader 拒绝。
- 每条训练与评估记录包含 `body_id + episode_id + layout_seed` lineage。
- 分开记录 encounter、同体 pickup→delivery、harmful ticks、heat entry/escape latency、switch 和非-timeout closure。
- 固定课程包含 near bootstrap、forced-return bootstrap、butter medium/far、forced escape、heat-route foraging、neutral stick、composite；near/forced-return 不进入 mastery。forced-return 从巢外携食状态起步并同步 body-side path integration，左右 home-bearing 均衡，但不向 controller 发布坐标、目标方向或动作标签。
- development matrix 包含 learned、no-optimize、cold、dense-local-shaping-off、segment-credit-off，所有训练臂共享 initial checkpoint 和 schedule。
- 能力门槛固定为：layout success ≥60%，每个成功 layout 的成功 body ≥60%；heat route/composite harmful tick rate ≤5%。
- P0 action sensitivity、temporal、freeze、replay gates 被带入 P1，不能以能力分数覆盖工程/机制回归。
- oracle-steering、FixedRule、random 三个不写 checkpoint 的诊断基线使用相同 held-out layouts。
- carrying/home 因果探针验证 carrying 与 path-integration home state 是否进入动作，并单独检查转向方向。
- forced escape 除成功率外，必须高于 random floor；成功率相同时要求 median escape latency 严格更低。

## P1 preflight 结果

小预算 preflight（1 ant、1 layout/tier、1 training round、3 evaluation rounds）返回 `BLOCK`：

- butter medium/far、forced escape、heat-route、neutral stick、composite 均为 0 个成功 layout；
- learned/no-optimize/cold 综合闭环成功数均为 0，因此 paired capability effect 未通过；
- P0 action sensitivity、真实 switch/非-timeout closure、frozen evaluation、replay lineage 均通过。

这次 preflight 只验证 runner、schema 和 gate 的诚实性，不代表正式 P1 统计结论。当前 v6/v5 schema 报告为 [`results/ecology_recovery/p1/ecology_p1.preflight.v4.json`](results/ecology_recovery/p1/ecology_p1.preflight.v4.json)：P0 sensitivity、temporal closure、frozen evaluation、replay lineage 通过；六类能力、paired effect、small-budget diagnostic solvability 与 carrying-home alignment 未通过。旧 `preflight.v3` 仅保留为历史证据。

默认 4-ant × 5-layout 诊断矩阵结果：oracle 六类能力均为 5/5；FixedRule 的 butter medium/far 为 5/5，但 heat-route、neutral-stick、composite 为 0/5；random 除 forced escape 外均为 0/5。环境可解性通过，报告见 [`results/ecology_recovery/p1/ecology_p1.diagnostics.seed0.json`](results/ecology_recovery/p1/ecology_p1.diagnostics.seed0.json)。

## 当前差距与串行决策

当前瓶颈已收敛到 carrying 后的返巢 credit：4-ant cold butter 定向评估在 10 个 medium/far layouts 中出现 6 次 pickup、0 次 delivery；carrying 状态能改变 latent/action，但 carrying lane 没有朝 home 方向转动。

进一步审计发现 P0 的逐 episode action-chain guard 对 learned/no-optimize 都是 9/9 全量回滚，导致旧 P1 实际无法累积训练。P1 已改为允许训练过程暂时退化，只在最终 checkpoint 执行冻结 action-chain gate。与此同时，原 `local_home_delta` 使用外部 pheromone 强度，可能奖励错误方向；现已改为方向无关的 path-integration home-distance progress。

第二个因果缺陷位于 runtime replay reward：旧实现优化 `actual - predicted` 的 signed prediction residual；一旦正收益可预测，正确动作的训练信号会趋零。现改为优化 PE owner 发布的 `ActualOutcome.action_payoff`，prediction residual 仅作为独立诊断 component，segment credit 仍独立保留。针对性 runtime replay 回归 19/19 通过。

普通 near 课程仍缺少充分 carrying 样本，因此新增 minimal-criterion forced-return bootstrap。初始 ±90° 版本 5 个 episode 全部负收益、0 delivery，已拒绝；冻结版本使用 near 1.5 距离与左右 ±30° home-bearing。2 ants × 2 episode 的最小验证得到 3 次 delivery，两个 owner checkpoint 均更新，held HOME probe 从 0/2 提升为 1/2。该结果证明训练样本链开始有效，但尚未满足“所有 body 对齐”的正式 gate，因此当前结论保持 `BLOCK`。

因此当前状态为：

1. P0 `PASS`；
2. P1 implementation/preflight 完成，但正式 4-ant × 5-layout × 两次重复尚未完成，且 preflight 为 `BLOCK`；
3. 按预注册规则不得进入 P2，也不得生成 promotion artifact。

正式 seed 0 的 learned arm 已在 bounded work-item 边界安全完成 `50/50`，progress state 同时记录 `training_complete=true`。near bootstrap 中，5 个 composite layout 全部出现 delivery，合计 26 pickups、23 deliveries；5 个 forced-return layout 全部完成返巢，delivery 分布为 2、4、2、5、6，合计 19。第二组 5 个 forced-escape layout 均记录 4/4 body escapes，合计 29 pickups、13 deliveries。主要失败集中在距离迁移：butter medium 5/5、butter far 5/5、wood-stick far 5/5 均为 0 pickup、0 delivery；两组共 10 个 composite-far 也全部为 0 pickup、0 delivery，合计 11 heat entries、3 heat escapes。这些是训练覆盖信号，不是 held-out capability 结果，不改变 `BLOCK`。

实测 checkpoint collection 在 learned 5/50 时增长到 21 MB，owner 尺寸审计确认 95% 以上来自 `joint_loop.learning.memory_checkpoint`：每 body 约 5,115 条 explicit artifacts，且 entries/semantic-index 双份持久化。为避免 50-episode 长跑撞上单-agent 32 MB / collection 128 MB 上限，Memory owner 新增确定性 `enforce_artifact_capacity(8192)`：优先淘汰 transient/episodic、弱、旧条目，并同步清理 semantic index、pending queues 与 attribute readout；CMS learned state 完整保留。容量在每个 ecology training episode 的 checkpoint 边界执行并写入 progress compatibility/episode summary；旧 seed0 archive 已安全迁移。容量从第 9 个 episode 起持续触发，到 `learned 50/50` 时双槽 archive 仍均稳定在约 31 MB，证明长跑期间未继续无界增长。

journal 位于 ignored `.partials/ecology_p1/seed0`，最终 report 尚未生成。当前相关回归集 53/53 通过。

当前 CPU 路径的 4-ant × 10-layout × 40-round 冻结评估约耗时 6 分钟。完整 P1 还包含训练、replay settlement、owner fingerprint 和五个 arms。resumable runner 已实现：每个 arm 使用双槽 `.vzac` journal（当前 + 前一 checkpoint），archive fsync/rename 后才原子推进 state，既能回退又把磁盘上限约束为两个 colony archive/arm；held-out evaluation 每个 layout 后写 journal。恢复时强校验 config、schedule digest、archive SHA256 与 latent/ant-count compatibility；已完成 arm/layout 不重跑，部分 arm 只执行缺失 suffix；训练 checkpoint 改变时旧 evaluation 自动失效并重跑。首次执行→倒退到 penultimate episode→恢复→配置冲突拒绝的端到端测试已通过。

当前机器的 PyTorch 构建支持 MPS，但运行环境报告 MPS 不可用。正式运行的 transient journal 必须放在已忽略的 `research/ant/results/.partials/`，完成报告仍写入 versioned `research/ant/results/ecology_recovery/p1/`。

运行命令：

```bash
python scripts/audit_ant_ecology_mechanisms.py
python scripts/run_ant_ecology_p1.py --diagnostics-only
python scripts/run_ant_ecology_p1.py --seed 0 \
  --progress-dir research/ant/results/.partials/ecology_p1/seed0
python scripts/run_ant_ecology_p1.py --seed 1 \
  --progress-dir research/ant/results/.partials/ecology_p1/seed1 \
  --report research/ant/results/ecology_recovery/p1/ecology_p1.seed1.json
```
