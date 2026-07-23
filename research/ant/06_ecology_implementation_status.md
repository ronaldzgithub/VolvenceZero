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

- curriculum/report schema 从 v3 升至 v5；旧 schema 被 checkpoint loader 拒绝。
- 每条训练与评估记录包含 `body_id + episode_id + layout_seed` lineage。
- 分开记录 encounter、同体 pickup→delivery、harmful ticks、heat entry/escape latency、switch 和非-timeout closure。
- 固定课程包含 near bootstrap、butter medium/far、forced escape、heat-route foraging、neutral stick、composite；near 不进入 mastery。默认每类 bootstrap 使用 5 个固定 layout，避免单一 near 偶遇不足以形成 credit 样本。
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

这次 preflight 只验证 runner、schema 和 gate 的诚实性，不代表正式 P1 统计结论。最新报告：[`results/ecology_recovery/p1/ecology_p1.preflight.v3.json`](results/ecology_recovery/p1/ecology_p1.preflight.v3.json)。

默认 4-ant × 5-layout 诊断矩阵结果：oracle 六类能力均为 5/5；FixedRule 的 butter medium/far 为 5/5，但 heat-route、neutral-stick、composite 为 0/5；random 除 forced escape 外均为 0/5。环境可解性通过，报告见 [`results/ecology_recovery/p1/ecology_p1.diagnostics.seed0.json`](results/ecology_recovery/p1/ecology_p1.diagnostics.seed0.json)。

## 当前差距与串行决策

当前瓶颈已收敛到 carrying 后的返巢 credit：4-ant cold butter 定向评估在 10 个 medium/far layouts 中出现 6 次 pickup、0 次 delivery；carrying 状态能改变 latent/action，但 carrying lane 没有朝 home 方向转动。

进一步审计发现 P0 的逐 episode action-chain guard 对 learned/no-optimize 都是 9/9 全量回滚，导致旧 P1 实际无法累积训练。P1 已改为允许训练过程暂时退化，只在最终 checkpoint 执行冻结 action-chain gate。与此同时，原 `local_home_delta` 使用外部 pheromone 强度，可能奖励错误方向；现已改为方向无关的 path-integration home-distance progress。最终 home alignment gate 仍未通过，因此当前结论保持 `BLOCK`。

因此当前状态为：

1. P0 `PASS`；
2. P1 implementation/preflight 完成，但正式 4-ant × 5-layout × 两次重复尚未完成，且 preflight 为 `BLOCK`；
3. 按预注册规则不得进入 P2，也不得生成 promotion artifact。

当前 CPU 路径的 4-ant × 10-layout × 40-round 冻结评估约耗时 6 分钟。完整 P1 还包含训练、replay settlement、owner fingerprint 和五个 arms，必须按 seed/arm 做可恢复 shard 后再执行。当前机器的 PyTorch 构建支持 MPS，但运行环境报告 MPS 不可用。

运行命令：

```bash
python scripts/audit_ant_ecology_mechanisms.py
python scripts/run_ant_ecology_p1.py --diagnostics-only
python scripts/run_ant_ecology_p1.py --seed 0
python scripts/run_ant_ecology_p1.py --seed 1 \
  --report research/ant/results/ecology_recovery/p1/ecology_p1.seed1.json
```
