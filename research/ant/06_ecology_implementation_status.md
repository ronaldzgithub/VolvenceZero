# Ecology P0/P1 实施状态

> 更新时间：2026-07-30。P0 机制审计仍为 BLOCK；P1 runner、诊断矩阵与 home-action probe 已实现；P2 因 P1 尚未 PASS 而未启动。

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

正式 seed 0 的 no-optimize arm 也已安全完成 `50/50`，`training_complete=true` 且最终 archive SHA256 校验通过。其 near 训练覆盖并不弱：butter-near 合计 28 pickups、19 deliveries，composite-near 合计 37 pickups、35 deliveries；forced-return 五布局 delivery 为 2、3、3、6、6，合计 20，略高于 learned 的 19。距离迁移同样失败：butter-medium 5/5 为 0/0，butter-far 仅一个布局出现 1 pickup 且全组 0 delivery，wood-stick-far 和两组 composite-far 均为 0 pickup、0 delivery。当前训练轨迹没有给出 learned 优于 no-optimize 的证据，甚至部分 near 指标由 no-optimize 更高；是否存在学习增益必须由后续冻结 paired evaluation 判定。

正式 seed 0 的 dense-local-shaping-off arm 已安全完成 `50/50`，`training_complete=true` 且最终 archive SHA256 校验通过。其 butter-near 与 no-optimize 相同，合计 28 pickups、19 deliveries；composite-near 更高，合计 44 pickups、42 deliveries。forced-return 五布局均有 delivery，合计 15，低于 learned 的 19 与 no-optimize 的 20。距离迁移仍失败：butter-medium 和 butter-far 各仅一个布局出现 1 pickup，均无 delivery；wood-stick-far 和两组 composite-far 全部为 0 pickup、0 delivery。关闭 dense local shaping 没有造成 near 行为崩溃，也没有修复远距闭环；训练轨迹暂不支持 local shaping 是现有 near 成功或远距失败的决定性因素。

正式 seed 0 的 segment-credit-off arm 已安全完成 `50/50`，`training_complete=true` 且最终 archive SHA256 校验通过。其 butter-near 合计 27 pickups、19 deliveries，composite-near 合计 36 pickups、33 deliveries；forced-return 五布局 delivery 为 2、4、2、7、6，合计 21，高于 learned 的 19。butter-medium 5/5 为 0/0；butter-far 有三个布局各出现 1 pickup，但全组 0 delivery；wood-stick-far 和两组 composite-far 全部为 0 pickup、0 delivery。关闭 segment credit 没有造成 near 行为崩溃，也未修复远距闭环，训练轨迹暂不支持 segment credit 是现有 near 成功的必要条件。

五个 arms 的训练已全部完成，seed 0 冻结评估已完成 learned 的 `30/30` layouts。能力结果只有 forced-escape 通过（5/5 layouts、20/20 bodies、逃逸延迟 3–5 ticks）；butter-medium、butter-far、heat-route-foraging、neutral-stick、composite 均为 0/5 layouts、0 successful bodies。policy fingerprint 30/30 稳定，replay lineage/settlement coverage 最低均为 1.0，replay drop 总数为 0；但 temporal-learning fingerprint 只有 24/30 稳定，全部 5 个 forced-escape layouts 与 1 个 neutral-stick layout 发生漂移。因此 forced-escape 的行为成功同时违反 frozen-temporal gate，不能作为晋级证据，P1 继续 `BLOCK`。

2026-07-24 的根因收敛包按 P0→P1→P2 串行约束先处理三项 P1 blocker：

1. **冻结写入泄漏**：`joint_learning_enabled=False` 原先只存于 session，未进入 joint-loop；恢复出的 pending batch 仍可触发学习。此外，普通 family outcome feedback 还会绕过调度器更新 action-family 与 learned family-match。现由 session 同时关闭 joint-loop 与 temporal policy/store owner 写入；旧 learned checkpoint 的 forced-escape seed `2020017` 重跑完整 4 ants × 40 rounds，4/4 仍逃逸且 policy/temporal-learning fingerprint 均稳定，replay drop=0。
2. **探索抹平 learned mean**：原 burst/coast 探索在 21/24 coast steps 把 posterior mean 压成 centroid，切断 state-conditioned steering。现探索只拥有 sample residual，保留 learned mean。2 ants × 16 rounds 的同 seed 短对照显示仅此修复未改变 near/medium/far 的 pickup 分布（far 仍为 0），因此它是必要契约修复，但不能单独宣称解决距离能力。
3. **runtime actor 方向与幅度**：one-step runtime fragment 原先把每个末拍当 terminal，未对非终局 next-substrate 做 critic bootstrap，小幅正 local payoff 可能被初始正 value 反转为负 advantage。现非终局使用 next published signature 做 TD bootstrap；低秩 causal action head 对 advantage 做有 `0.05` floor 的 RMS scaling 并 clamp `[-1,1]`，track/value 仍保留物理 payoff 原尺度。回归覆盖正负更新方向、可观察 bias 幅度、rollback 与 checkpoint round-trip。

首次提交后的两个独立短训练 seed 中，medium 均为 2 pickups；far 分别为 1 与 0 pickup。对第二个 seed 从 medium checkpoint 精确重放 far episode发现：两只蚂蚁最大离巢 3.43/4.14，而目标中心距巢 3.50，最近食物边界仍为 0.60，证明距离预算足够、失败来自角度覆盖。原探索 option 为 24 steps，16/24-round episode 只在开头重定向一次；收敛为 8-step option（2-step burst + 6-step coast）后，同 checkpoint/layout/seed 的最近边界距离变成 0，并出现 1 pickup，最大离巢半径仍为 3.70/3.73。该改善不读取坐标或目标方向。

最终算法变更使旧 `development.v5/progress.v1` 以及中间诊断 `development.v6/progress.v2` journal 均不得继续；新实验升级为 `development.v7/progress.v3`，必须使用全新目录与报告。当前仍保持 `BLOCK`，只有 v3 matched arms 与 frozen held-out gates 完成后才能重新判定 P1。

全新 v3 seed 0 journal 已从空目录正式复跑至 learned `10/50`。butter-near 五布局合计 26 pickups、18 deliveries，其中首布局为 0/0，后四布局均同时出现 pickup 与 delivery；burning-match-near 五布局合计 31 pickups、17 deliveries，五个布局均有 pickup、四个布局有 delivery。第 10 个 episode 记录 6 pickups、4 deliveries、4 heat escapes，容量 owner 同步淘汰 4,141 条旧 artifact；双槽 archive 分别约 30 MB 与 31 MB，未发生恢复、容量或 journal 回滚错误。这是修复后三个根因的早期正向训练证据，但只覆盖 learned arm 的 near 阶段，不能替代剩余 40 个训练 episode、matched arms 与 frozen held-out gates，P1 判定仍为 `BLOCK`。续跑 journal 位于 ignored `.partials/ecology_p1_v3/seed0`。

v3 随后推进到 learned `25/50`：composite-near 五布局合计 20 pickups、17 deliveries，forced-return 五布局合计 12 pickups、23 deliveries；但 butter-medium 五布局全部 0/0。冻结轨迹审计排除了距离和信号稀疏：20/20 body 均超过目标半径，每拍都有 local payoff，food 左右差最高约 0.54，但每 body 仅覆盖 2–5 个角度扇区且 `food_diff -> turn` 相关方向跨 body 混乱。进一步定位出三项 owner 契约错误：Ndim GRU hidden 原生为 `[-1,1]`，action head 却按 `[0,1]` 再做 `2*h-1`；24-turn episode 只有 23 个 settled transition，24-step segment 在无 milestone/switch 时来不及被下一拍 optimizer 消费，跨 episode replay-excluded checkpoint 将其丢弃，实测 fixed/no-optimize 均保持 `head_step=0`；8-step exploration proposal 又把连续变化的 posterior mean 写入 identity，并在后 6 拍改成 common-mode residual，所以并非 coherent option。现改为直接消费 signed hidden、Digital Ant segment 上限 16、sample residual 只按 `segment + dimension` 定义并整段保持。相同 v3 checkpoint/layout/seed 的首个 4-body medium 冻结重放由 0/0 变为 2 pickups/1 delivery，完整五布局合计 11 pickups/3 deliveries，角度覆盖从 2–5 提升到 5–9 个扇区，证明 coverage blocker 已修但回巢闭环仍未达到 P1 门槛；把 option 再缩成 4-step 的反例对照在一个布局把 delivery 从 1 提到 2，却在困难布局把 pickup 从 1 降为 0，故已回退，禁止把随机轨迹重排冒充根因修复。2-body 五布局 paired smoke 中 learned/no-optimize 均为 2 pickups/2 deliveries，尚不能声称 optimizer advantage。算法与诊断 truth 变化使 v3 journal 失效；新报告/journal 升级为 `development.v8/progress.v4`，P1 继续 `BLOCK`。

全新 v4 seed 0 journal 随后从空目录推进到 learned `25/50`。前三组 near 的五布局合计分别为 butter `74 pickups / 67 deliveries`、burning-match `89 / 73`、composite `92 / 83`，说明近场闭环显著恢复；forced-return 为 `16 / 23`。但 butter-medium 只有第四个布局出现 `4 / 4`，其余四个仍为 `0 / 0`，尚未达到 3/5 门槛，因此在第 25 局主动停止盲跑。成功 seed 的冻结重放也不是四只 body 普遍成功，而是 body 0/3 各完成两次循环；成功与失败 seed 的最大离巢半径都只有约 `1.24`、绝对转向均值约 `0.48–0.50`，主要差别是固定小圆是否偶然扫过食物。

v4 checkpoint 的 action probe 揭示第四个根因：四只 body 的 food/heat 左右 probe 都产生同方向正转向，只有 home 仍方向对齐；World/Self action-head 第二维 bias 已普遍达到 `+0.42…+0.50`，而第一维只约 `-0.03…-0.09`。每个 episode/body 原先都复用只由 `segment + dimension` 定义的同一探索序列，偶然收益会反复强化同一动作方向；同时低秩 head 的 bias 直接接收完整梯度，output/input factor 分别再乘 basis、旧 output factor 与缩放，导致常数截距比状态路径快多个数量级。保留 v4 的 input/output factor 而把 bias 清零后，四只 body 的 heat probe 立即全部恢复左右方向对齐、home 继续通过，证明 state-conditioned 映射已经存在但被截距淹没；food 分离仍弱，不能把 ablation 当成最终修复。

当前收敛包由 temporal owner 为 coherent option 增加可选的 opaque context digest：Digital Ant 用 episode seed 与 body offset 分散序列，同一 schedule 的 matched arms 仍共享 context；不提供 context 的通用 runtime 保持历史序列。action-head bias 同时收紧为状态路径 `0.05` 倍学习尺度、单步 `0.01`、总幅度 `0.1`，并移除 input factor 额外的 `0.25` 衰减。首轮 2-body 25 局门槛中 bias 已降到 `-0.002…0.017`，近场三阶段和 5/5 forced-return 均保留闭环，但 butter-medium 五布局只有 2 pickups、0 deliveries；body 0 只通过 heat probe，body 1 只通过 food probe。进一步确认未中心化的 factor gradient 仍可借 hidden 公共均值重建隐式截距，现将 batch mean 只分配给受限 bias、output/input factor 只消费 centered state covariance。v10 同 seed 门槛把 butter/composite near 分别从 `18/15、10/7` 提升到 `19/17、12/9`，但 medium 仍为 `2/0`；45 次更新后 output-factor L1 仅 `0.059`，probe 的左右 head residual 差分约 `0.0002`，不足以覆盖冻结基底 `0.02–0.10` 的同向分量。

后续 v11 诊断曾尝试按 `sqrt(n_z)` 放大 factor 学习尺度：25 局 medium 仍为 `2/0`，composite-near 反从 v10 的 `12/9` 回落到 `9/5`；续跑至 50 局后 factor L1 已增长到 `0.29–0.46`，body 1 却丢失第 25 局曾通过的 food probe。再追加 5 个 butter/composite/forced-return/burning-match near rehearsal 虽产生 `21 pickups / 19 deliveries`，仍未恢复 body 1 probe；把 learned bias 归零也没有恢复方向分离。这组反例排除了单纯学习幅度、样本量、末尾 rehearsal 和显式 bias 是剩余充分根因，v11 放大已回退，禁止合入正式线。当前正式算法停在 `development.v10/progress.v6`，其目标是消除重复探索与显式/隐式截距主导；是否提高 4-body 通过比例仍需正式门槛验证，P1 继续 `BLOCK`。

正式 v10 的 4-body 门槛随后推进到 25/50：butter/burning/composite near 分别为 `41/32`、`31/17`、`43/38`，forced-return 为 `14 pickups / 26 deliveries`，但 butter-medium 仍只有 `2 pickups / 0 deliveries`。四个 checkpoint body 的最终 food probe 全部方向失败，只有 body 0 通过 heat、body 0/2 通过 home；从该 checkpoint 追加五个 butter-near 聚焦 episode 虽产生 `38/30`，四个 food probe 仍全部失败。由此进一步排除 body 数、一般样本量和末尾 food rehearsal 是充分根因。审计确认 live action head 使用跨 turn 的 recurrent `posterior_hidden_state`，replay 也据此更新；相同当前感知会因此前序列落在不同 state 坐标，导致训练信号无法沉淀为可复用左右映射。现由 temporal owner 新增 `causal_action_head_state`：同一 Ndim encoder 对当前 observation 做零历史编码，live forward、pure/torch replay、pending capture 与 open segment 共同持久化该 signed state；serving hidden 继续服务 metacontroller/value，不再作为 action-head 输入。`joint_loop.learning` schema 升为 v3，正式 ecology 报告/journal 升为 `development.v12/progress.v7`，v10 journal 只保留为只读根因证据，P1 在 v12 新跑通过前继续 `BLOCK`。

全新 v12 4-body journal 已推进到 25/50。butter/burning/composite near 分别为 `41/32`、`31/17`、`42/38`，forced-return 为 `13 pickups / 26 deliveries`；butter-medium 仍为 `2 pickups / 0 deliveries`，与 v10 的 blocker 完全相同，因此未继续到 50。四个 food probe 仍全部同向；左右 action-head residual L1 差虽从 v10 约 `0.0002` 提高到 `0.00036–0.00091`，仍不足以覆盖 `0.08–0.13` 的基础同向转向。进一步对正式 checkpoint 审计发现，food 左右 `causal_action_head_state` L1 已达 `0.852`，但 rank-4 basis 只保留 `0.0685`；heat 则从 state `1.695` 保留到 basis `0.3505`。同时所有 body 的 input-factor L1 在 29–48 次更新后仍几乎等于同一初值（World `27.49748`、Self `29.72445`），证明输入投影实际冻结。根因是低秩 head 的 output factor 全零初始化，而 input gradient 必须乘 output factor，形成双线性零鞍点；output 后续只长到 L1 `0.033–0.068`，反馈仍不足。现改为小幅确定性零均值 output 初始化，并按 output-column norm 归一化 pure input-factor feedback；正式 schema 升为 `development.v13/progress.v8`，v12 journal 转为只读反例，P1 继续 `BLOCK`。

v13 从空目录复跑到 12/50 后主动停止：butter/burning near 为 `40/30`、`32/17`，composite 首两局为 `19/19`、`2/1`；后者相对 v12 同布局的 `5/5` 明显回退。checkpoint 中 output-factor L1 已被随机初始化直接推高到 `2.68–2.84`，food head residual 左右差约 `0.021–0.022`，四只 body 的 food probe 却仍全部同向，说明随机低秩 prior 只放大了错误映射，不能保留。对下一局做不写 journal 的 optimizer microbatch 审计后发现更直接的根因：大量 switch/milestone 闭合段只有 1 条 transition；centered factor gradient 在 singleton batch 中 `state_rms=signal_rms=covariance=input_delta=output_delta=0`，只有 bias 能更新。只有 `n=2/15/16` 的批次出现非零 covariance 与 factor delta。因此 v13 随机 output 初始化回退为零，保留 output-column norm 作为第二个非零 covariance batch 之后的 input feedback；ACTIVE runtime replay 改按 transition 数而非 segment 数判断 batch，通用默认仍为 1，Digital Ant evidence profile 设为 4。正式 schema 升为 `development.v14/progress.v9`，v13 journal 只读保留，P1 继续 `BLOCK`，必须由全新 v14 门槛验证。

v14 从空目录复跑到 12/50 后再次主动停止：butter/burning near 为 `41/32`、`30/16`，但 composite 首两局仍为 `18/18`、`2/1`，复现 v13 的关键回退。checkpoint 证明 batch target 已生效，4 个 body 的 World/Self head 只执行 3–16 次 multi-transition 更新；随机 output prior 已不存在，output-factor L1 仅 `0.006–0.051`，但 food probe 仍 4/4 同向失败。相对同 run cold checkpoint 的元素级审计进一步显示，World input-factor 仅移动 `0.00003–0.00064 L1`，说明旧 input feedback 的 `0.05` column-norm floor 仍在微小 output 阶段造成梯度饥饿。现改为 owner 内同批 block-coordinate 更新：先从 centered covariance 计算 bounded candidate output，再按 candidate 的真实列范数回传 input，最后原子提交；零 output live prior 和 singleton no-op 均保持。正式 schema 升为 `development.v15/progress.v10`，v14 journal 只读保留，P1 继续 `BLOCK`。

v15 从空目录复跑到 12/50，十二局行为与 v14 完全一致，composite 第二布局仍为 `2/1`；strength 从 `0.35` 临时提高到 `1.0` 的只读 probe 仍 4/4 food 失败，排除单纯 forward gain。block-coordinate 修复确实把 World input-factor 位移提高到 `0.012–0.055 L1`，但 output 与 residual 几乎不变。随后从安全 checkpoint 追加一局只读 optimizer 审计：所有 batch 均为 `n=4/16/17`，state RMS `0.07–0.13`、signal RMS `0.58–1.01`、covariance L1 `3.6–7.7`，证明信用方向统计已存在；每批参数变化却只有 `0.00007–0.00020`。根因是 pure factor optimizer 在按 batch size 平均后仍额外乘 `0.12`，而 torch autograd path 使用 owner 基础学习率。现只对 output/input factor 移除该衰减，bias 继续保留 `0.12 × 0.05`、单步和总幅度上限；正式 schema 升为 `development.v16/progress.v11`，v15 journal 只读保留，P1 继续 `BLOCK`。

v16 从空目录推进到 25/50 后按门槛停止。butter/burning/composite near 分别为 `41/32`、`30/16`、`43/37`，forced-return 为 `12 pickups / 23 deliveries`，butter-medium 仍为 `2 / 0`；说明 factor timescale 修复没有破坏近场与返巢，但没有解除 L1 blocker。四个 body 的 food probe 仍全部同向失败，heat 与 home 各有 3/4 通过。参数审计确认优化器已真正工作：相对 cold，World input-factor L1 位移达到 `0.10–0.49`，Self 达 `0.57–0.93`；因此剩余问题不是“参数不更新”。成对状态审计进一步定位第二次信息瓶颈：food 左右原始触角 L1 为 `0.408`，current-observation action-head state L1 为 `0.80–0.87`，但固定 rank-4 basis 只保留 `0.064–0.070`；heat 从 state `1.59–1.74` 保留到 basis `0.331–0.364`。food 最终 residual 左右差只有 `0.0009–0.0073`，不足以覆盖基础同向转向；这与 medium `2/0` 一致。

现由 temporal owner 增加可选的 action-head 最大 rank。通用默认 `None` 保持历史低秩，Digital Ant evidence profile 显式请求 `rank=n_z=16`；full-rank input 用 identity、output/bias 保持全零，因此保留 encoder state 轴而不注入随机动作 prior。owner 只允许在学习开始前选择 rank，已有 live mapping 拒绝原地改 shape。正式报告/journal 升为 `development.v17/progress.v12`，v16 checkpoint 仅作只读根因证据；必须从空目录运行 v17，并先检查 12 局 probe/近场回归，再决定是否推进到 25/50。P1 当前仍为 `BLOCK`。

v17 从空目录推进到 25/50。butter/burning/composite near 分别为 `41/32`、`30/16`、`41/36`，forced-return 为 `12/24`，butter-medium 仍为 `2/0`；近场与返巢保持，但 L1 blocker 未解除。full-rank 确实把四体 food basis 左右差从 v16 的 `0.064–0.070` 提高到 `0.201–0.234`，约 3 倍，证明表征修复生效；然而 food probe 仍 0/4，最终 residual 差仅 `0.00034–0.00357`。进一步按 body 审计发现，episode 12→25 期间 body 0/1 的 head update step 永久停在 `5/3`，body 2/3 则增长到 `28/24`。

同一 v17 checkpoint 的只读 far 重放定位出新根因：四体都闭合一个 16-transition segment、每拍都有 payoff、都进入 `full-cycle-batch`，但 body 0/1 的 batch 被 `metacontroller-drift` 回滚；所谓 drift 的 temporal shift 为 `0.929/0.774`，persistence shift 仅 `0.00008/0.00014`，track shift仅 `0.045/0.066`。Internal-RL 的 `align_temporal_from_tracks()` 每批把 legacy `switch_bias` 写成 `1-persistence`，而正式 `n_z=16` Ndim switch 不消费该字段，rollback gate 因而用无效兼容字段撤销真实 actor 更新；dual-track aggregate 还错误地把逐拍 `latent_mean` 发布为 `track_parameters`，存在把状态变化误判为参数漂移的同类风险。现令 `n_z>3` 不再执行 legacy alignment，aggregate 改发 owner track weights；legacy 三维行为不变。同 checkpoint/同布局修复后重放，四体 step 均 `+1`，rollback reason 全空。正式 schema 升为 `development.v18/progress.v13`，v17 journal 只读保留，P1 继续 `BLOCK`，必须全新复跑。

v18 从空目录推进到 25/50。12 局时四体 action-head update step 已由 v17 的 `5/3/16/12` 恢复为 `13/13/16/12`，25 局达到 `27/26/28/24`，证明假 drift 回滚根因已解除；近距复合新增三局为 `21/17`，butter 复习保持连续回流。但五个 butter-medium 布局仍仅 `2 pickups / 0 deliveries`，food 左右 probe 继续同向。25 局 state/basis 已保留方向差（food basis `0.215–0.234`，heat `0.424–0.471`），参数总 output-factor L1 达 `0.827–1.010`，真正进入冻结 plant 的两条 steering output row 却仅 `0.085–0.221`，其余大部分信用被 `motor_decode` 从不读取的 `z[3:16]` 吸收。第三个根因因此收敛为 actuator-support 缺失，而不是继续增加训练量。

现由通用 temporal owner 增加可选 `effective_dims` 契约，默认 `None` 保持全维历史行为；Digital Ant profile 依据冻结 `motor_decode` 显式声明 `(0,1,2)`。pure/torch action-head gradient 及 live/sandbox residual 都对非 actuator output row 严格置零，其他 controller code、track/value 学习和 plant 均不改变；恢复 `None` 即回滚。正式 schema 升为 `development.v19/progress.v14`，v18 journal 只读保留。必须从空目录先跑 v19 的 12/25 门槛，P1 继续 `BLOCK`，未通过前不得续跑到 50。

v19 从空目录推进到 25/50。butter/burning near 仍为 `41/32`、`30/16`，composite 前五局与 v18 完全同轨，butter-medium 仍为 `2/0`。四体 update step 为 `27/26/28/24`，所有 `z[3:16]` output row L1 严格为 `0.0`，说明 actuator-support 契约生效且未破坏近场；但有效 steering L1 与 v18 几乎相同，food residual 左右差反而只有 `0.00059–0.00108`，故“无效维度稀释”不是 L1 blocker 的充分根因，禁止据此续跑到 50。

随后对真实 capture 的 likelihood 做代数审计发现更直接的断链：无显式 exploration 的 encoder `z_tilde` 使用 `0.5 × posterior_std × noise`，而 Digital Ant 稀疏探索分支使用 `1.0 × posterior_std × noise`；runtime replay 却对两者固定重建为 `0.5`。实际 capture 的第 0/2 维噪声位移精确等于 replay 预测的 2 倍，score gradient 因错误方差放大约 4 倍并频繁撞上 `±4` clamp。现由 joint-loop owner 在 capture/transition 持久化实际 `posterior_sample_scale`，pure/torch likelihood 共同消费；默认/历史为 `0.5`，显式 exploration 为 `1.0`。`joint_loop.learning` schema 升为 v4，ecology schema 升为 `development.v20/progress.v15`；v19 journal 只读保留，v20 必须从空目录复跑。

v20 从空目录推进到 25/50。butter/burning/composite near 为 `41/32`、`30/16`、`44/39`，比 v19 的 composite 增加 `3/3`，且五个 interleaved butter 继续产生近场回流；但 butter-medium 仍为 `2 pickups / 0 deliveries`，所以未盲目续跑到 50。四体 action-head update step 为 `27/28/29/24`，无效 output row 严格为零。真实 far batch 审计显示 posterior scale 修复后所有 score clamp 均为 `0`，food signal 与 steering score covariance 在八个 World/Self batch 全为正，food basis 与实际 steering 参数更新 cosine 也全部为正，排除 sensor→GAE/score→basis 的符号断链。

同 checkpoint 的 ACTIVE/DISABLED/strength 对照进一步证明 head 在四体上都改善左右相对排序，但幅度不足以覆盖冻结 controller 的同向偏置；参数放大 `4/8/12×` 可让相对 turn delta 依次跨零，却仍不能满足“左物向左、右物向右”的绝对 probe。把已学 mapping 推到 `100–1000×` 饱和后，四体只有一体形成绝对对向，其余分别落到双左、停转或双右。根因是此前 `effective_dims=(0,1,2)` 只删除了 `z[3:16]`，没有删除 actuator-null 的 `z[0]+z[1]` common mode；`motor_decode` 只消费 `z[1]-z[0]`，无效 common mode 仍吸收信用并固化 body-specific intercept。

现由 temporal owner 新增通用、可回滚的 `contrast_pairs` 投影：pair 必须 disjoint、界内且属于 effective support；pure replay score gradient、torch forward gradient、live/SHADOW residual 全部执行同一正交差分投影。Digital Ant profile 声明 `((0,1),)`，速度 `z[2]` 不变，并把该稀疏 profile 的 head strength 从 `0.35` 提升到 `1.0`，消除训练梯度与 serving residual 两端累计的 `0.35²` 时间尺度衰减；通用默认仍为 `None/0.35`。正式 schema 升为 `development.v21/progress.v16`，v20 journal 只读保留，必须从空目录先跑 12/25 probe 与 near/medium 门槛，P1 继续 `BLOCK`。

评估诚实性收敛包(A)：P1 报告新增 `food_steering_alignment` gate,`development` schema 升为 v22(progress 后随 forced-approach 收敛包升为 v17,见下文;v21 journal 不可复用)。此前 `paired_action_sensitivity` 对 food/heat 只要求 `action_sensitive`,不要求 `target_aligned`;而 near tier 的 pickup 可由巢周小半径巡游偶然扫过近处食物产生,**不需要学到的食物梯度转向**,这个假阳性长期掩盖了 medium/far 真正要求的同一能力(逐版本 food probe 0/4 却仍靠 near pickup 显得"管线在工作")。新 gate 要求 near 距离 food 成对探针在 ≥60% body 上绝对方向对齐(左食→左转、右食→右转),只读已发布 probe truth、不回灌学习。按历史审计该 gate 当前应为 BLOCK,作用是把根因断点从 10 分钟整跑前移到秒级 probe,并明确归因为"食物转向从未学到"而非被 near 掩盖。

传递函数天花板测量(B)：`scripts/measure_ant_food_steering_gain.py` 用与 probe 相同的 `AntSession.step` 只读单步,测 cold 与 v21 learned(25/50)在 near/medium 的 `food_diff→turn`。结论决定性:(1) 食物居中(food_diff≈0)时存在**大幅同向基线转向**——cold near `+0.083`、medium `+0.050`,learned near `+0.11…+0.145`、medium `+0.048…+0.070`,且四体一致朝同一方向,即基底 metacontroller code 自带定向环绕,不是 action head 产生;(2) 食物左右信号带来的转向 authority 只有 `~±0.001…0.003` rad,比基线小近两个数量级,`turnL` 与 `turnR` 几乎相同(有时反向),near food_diff 高达 `0.204` 却几乎不改转向;(3) **learning 未增加 food 转向 authority**——cold `min_authority≈+0.001`,learned `≈-0.001…-0.003`,25 个 episode 后食物转向增益仍≈0 甚至略负。由此:medium 的 blocker 不是"还差一个 action-head 衰减因子"(head authority≈0 且不随训练增长),而是 (i) 基底 code 的同向基线主导 + (ii) 食物方向在有界头处的增益被钉死在≈1e-4。结合 near 靠环绕巧合拾取、medium/far 从不真正到达食物,food 转向在任何 range 都缺训练压力(near 无需转向即得奖、far 从不得奖),指向课程设计与表征/基线,而非在线 plumbing。报告:`research/ant/results/.partials/food_steering_gain.v21.json`。

觅食转向压力收敛包(forced_approach)：基于 (B) 的结论,在 P1 固定 schedule 的 forced_return 块后新增 5 个 `forced_approach` butter-near episode(50→55 work items,progress schema 升为 `digital-ant-ecology-p1-progress.v17`,v16 journal fail loudly)。落地时核实了三件推翻原计划的事实:(1) 起始朝向本来就是随机的(`_spawn_body` 用 `uniform(0,2π)`),"随机朝向"不是缺的杠杆;(2) 食物接近 shaping 已存在(`0.45·tanh(food_delta)`),但它只奖励"更近"这个结果,不区分转向与漂移;(3) near 拾取盘(蝶油半径 1.1,圆心距巢 0.95–1.35)与巢(半径 1.0)**几何重叠**,无方向游走即可拾取——这是 near 假阳性的物理根源,也解释了为何 food→turn authority 从无训练压力。`motor_decode` 代数上 `turn≈atan2(8·z1,1)`,反对称 z1=0.05 即给 0.38 rad,执行器天花板不存在;缺的只是让"转向=唯一得奖路径"的布局。forced_approach 与 forced_return 同契约:body 生成在拾取盘外(2×蝶油半径),朝向偏离食物方位 0.6π,左右修正按 body 交替平衡,只初始化状态并同步 PI,不发布坐标/方位/动作标签,不触碰共享 vz-temporal。基线中性化(C)按决策**暂缓**:若 forced_approach 使 head authority 增长并盖过 +0.08 基线,基线不再是绑定约束;若仍不增长,再单独评估动共享 metacontroller 冷 init 的风险。

forced_approach 首批实测(v22 journal 25/55,含完整 forced_approach 块 ep20–24)：**压力假设被证伪,并暴露更深根因**。块内战绩 0/0、4/4、3/3、0/0、3/2——布局确实堵死了"乱走白捡"(ep20/23 为 0/0);但 `measure_ant_food_steering_gain` 复测(`food_steering_gain.v22-25.json`)显示 learned `min_authority` 仍≈0(near `-0.0025…-0.0006`,medium `-0.0032…-0.0018`,不优于 cold `+0.001`),方向对齐 0/4;而**基线同向转向被训练放大**:cold `+0.083`→learned `+0.10…+0.149`(near)。解释:spawn 距离固定 2.2,基线转向放大到 ~0.13 时环绕半径 `0.4/0.13≈3.1`,固定曲率轨道即可扫进拾取盘,优化器用"统一转大圈"这一**非定向退化解**收割了 forced_approach 奖励(几何不匹配的 seed 即 0/0)。由此得出更锋利的根因:contrast_pairs 把 common mode 从 head 投影掉之后,"放大固定转向"的信用只能流入**无约束的 base policy** track weights——base 用退化解吸走全部转向信用,被约束成只能做反对称定向转向的 head 在信用竞争中被饿死在 ≈1e-3。推论:(C) 若只做一次性 init 对称化无效,信用会把基线重新学回来;候选方向是 (a) 随机化 forced_approach 几何(spawn 半径/角度随机,使任何固定曲率解失效,课程侧、可回滚)与/或 (b) 结构性让 head 成为唯一 steering 所有者(冻结/中性化 base policy 的 steering 轴,R2 稳定基底+有界控制器,动共享 vz-temporal,风险大)。已按决策先做 (a):spawn 半径 `1.45–2.9×拾取半径`、偏离角 `0.4π–0.8π` 由 layout seed 逐 body 随机,直线错过保证不变(最近距离 ≥1.38×拾取半径),progress schema 升 `v18`(v17 journal 语义已变,fail loudly);若随机几何下 25 局后 `min_authority` 仍≈0,即为 (b) 的决定性证据。

随机几何复测(v22r journal 25/55,`food_steering_gain.v22r-25.json`)：**仍为否定,(b) 的决定性证据成立**。随机块战绩 0/0、1/0、3/3、0/0、3/2(7 拾取,低于固定几何的 10,漏洞收窄);但 learned `min_authority` 仍≈0(near `-0.0025…-0.0002`,medium `-0.0032…-0.0014`),对齐 0/4,基线转向仍被放大到 ~0.147。两组受控实验(固定几何、随机几何)一致表明:在信用竞争下,无约束 base policy 总是用"放大同向转向、扩大扫掠面积"的非定向退化解吸走转向信用,被 contrast 投影约束的 head 拿不到差分信用,增益钉死在 ≈1e-3。结论:继续加课程压力无效,需要结构性方案——让 head 成为**唯一** steering 所有者(候选实现:扩展 evidence profile 的 `contrast_pairs` 语义,声明该 profile 时把 base code 在反对称 actuator 子空间上的分量投影掉,steering 只能由 state-conditioned head 供给;未声明 profile 的域不受影响,可回滚)。已知风险:heat 逃逸与 carrying-home 转向当前由 base policy 承载,steering 所有权转移后需经 head 重新习得,现有 probe/gate 体系可立即检测这两项能力是否存活。

exclusive steering 收敛包已落地(temporal owner + ant profile,development v23 / progress v19 / curriculum v7)：新增 `FinalRolloutConfig.internal_rl_causal_action_head_exclusive_steering`(默认 False,字节不变回滚路径),ant profile 打开。temporal owner 在四条路径共用 base 侧互补投影(每个 contrast pair 的确定性均值→common mode):live ndim forward(从调制后确定性均值算 delta 加回含噪候选码)、sandbox `CausalZPolicy._policy_mean`、pure `runtime_replay_policy_distribution`、torch PPO 三条 in-graph lane;pure `_trajectory_gradient` 在 pair 维敏感度乘 0.5。关键设计不变量:**投影只作用确定性均值,探索噪声保留反对称分量**——否则 `(a-μ)` 在 contrast 维恒为 0,head 的 PPO 梯度死锁、冷启也无法提出转向(这可能也是 v10–v22 head 学不动的共因之一:base 均值抢占了噪声发现的转向信用)。要求非空 contrast_pairs + head ACTIVE,否则 fail loudly。配套:冷启探针门放宽为只验 input_reachable(exclusive 下冷启 head 精确为零,转向可区分性 0 是设计使然),训练后 `paired_action_sensitivity`/`food_steering_alignment` 硬门不变。验证:vz-temporal 35 通过(含 6 个新 exclusive 契约测试:互补分解、setter/CausalZPolicy 校验、live forward 只动 pair 维、replay 分布投影数值)、vz-embodiment-ant 125 通过(1 个 matched_control fingerprint 失败为 main 既有)、tests/contracts 3596 通过(4 失败均 stash 验证为 HEAD 既有)、temporal_interface+agent_session_runner 91 通过。硬判据:v23 journal 25 局后 `measure_ant_food_steering_gain` 的 learned `min_authority` 必须显著 >1e-3 且方向对齐,同时观察 burning/carrying-home gate 是否经 head 重新习得。

β 门泄漏修复(测量完整性收敛包)：v23 跑到 55/55 后为回答"是否值得再投入算力"做了一组只读审计,连续推翻三个假设并找到真因。(1) **截距假设被否**:head 的 `|Δbias|`≈0.003(折合 turn 0.012–0.016),且四体中三体符号与实测基线相反,常数截距不是基线来源;状态相关增益 `|out0-out1|`=0.033–0.093,大 10–30 倍。(2) **表征瓶颈被否**:food 左右互换移动 33% 的 head 状态范数,到 basis 仍是 33–35%,信号毫无衰减地到达 head——脚本判词建议的"离线表征刷新"方向可排除。(3) **截断假设被否**:z 值在 0.015–0.024,远离 `[0,1]` 边界。(4) **真因是 β 门**:`latent_code[i]=β_i·候选_i+(1-β_i)·旧码_i` 中 β 逐维,即使候选已被 exclusive steering 投影为对称、旧码也对称,`(β_i-β_j)·(候选-旧码)` 仍凭空造出 contrast。判据实验:cold checkpoint 的 head 参数**精确为零**、exploration 关闭,却仍产生 ±0.005 rad 转向振荡,且四个 session seed 波形几乎相同(排除随机噪声),与当时学到的 food 响应完全同量级。修法是原理性的:opponent-coded pair 是**一根**执行器轴,必须整体切换,故 exclusive steering 下 `effective_gate` 按 pair 取共享均值。修复后零参数 head 输出精确 `0.000000`。**同一 v23 checkpoint 重测**:near 绝对方向对齐 0/4 → **2/4**,medium 0/4 → 1/4,残余基线 0.0243 → 0.0080,cold 全部精确为 0(head 为零本就不该有响应)。结论:此前所有"food probe 0/4"的历史读数都被这条泄漏污染,能力被系统性低估;`paired_action_sensitivity` 的冷启断言与 `test_paired_ecology_channels_reach_code_and_motor_output` 同步改为"冷启只验 input reachability + code 可达,`action_sensitive` 应为 False",转向能力改由训练后硬门验收。回归测试:零参数 head + exclusive steering 连走 4 拍,`code[0]==code[1]` 精确成立(已验证该测试在关闭修复时失败)。验证:vz-temporal 36 通过、vz-embodiment-ant 125 通过(仅 matched_control fingerprint 一项为 main 既有失败);`tests/contracts` 当前有 77 个收集错误与 30 个失败,经逐条核对全部来自同工作区其他在飞改动(sibling checkout 缺 `stable_value_hash` 导出、`final_wiring` 被他处加入 `personal_conditioning` import),无一涉及 causal head/temporal/ant。

v23 首批实测(28/55,`food_steering_gain.v23-28.json`)：**硬判据首次转正,信用竞争根因假设被证实**。(1) 退化解被结构性消灭:冷启基线转向从 0.083 降到 0.0028 rad,训练后仅 0.0175(v22r 被放大到 ~0.147)——base 无法再写 contrast 轴;(2) **learning 首次增加转向权威**:near `min_authority` cold +0.0003 → learned +0.0015…+0.0031(约 10 倍),medium 0.0000 → +0.0003…+0.0010,v10–v22r 该数从未增长过;(3) 方向对齐 1/4(28 局时仍在增长期);(4) 行为侧:**medium 三连局送达 1/1、3/1、2/1——medium delivery 自 v10 以来首次离开 0**。burning-near(ep5–9:8/5、6/2、9/6、4/0、6/4)与 composite-near(ep10–14:含 12/10、10/9)在所有权转移后仍健康,说明 heat 逃逸/避让经 head+探索噪声路径存活,未出现担心的能力崩塌。forced_approach 块(ep20–24)战绩 0/0、2/1、3/2、0/0、3/2,与 v22r 相近但这次伴随 authority 实际增长。待办:等 55 局跑完后复测 authority/对齐并查 `paired_action_sensitivity`/`carrying_home_action_alignment`/`food_steering_alignment` 硬门;另注意到本次 `--max-new-work-items 25` 未在 25 局处暂停(继续跑完整日程,对本验证无害),原因待查,与本收敛包改动无关。

v24 全程干净训练(55/55,`development.v24 / progress.v20`,`food_steering_gain.v24-55.json`)：v23 是"训练期带 β 门泄漏、事后才修"的混合体,v24 是修复后从空目录跑完的第一条完整轨迹,故与 v23 只可参考对比。**行为侧**:butter-near `45/33`、burning-near 首轮 `35/18` 第二轮 `43/28`(所有权转移后 heat 逃逸与近场闭环不但存活且继续变强)、composite-near `39/33`、forced_return `15 拾取 / 29 送达`、随机几何 forced_approach `9/5`(块内两局 0/0,退化解仍被堵死);**butter-medium `11 拾取 / 2 送达`**——v10–v22r 该块**稳定停在 `2/0`**,拾取翻五倍以上且送达非零。远距仍全线归零:butter-far `4/0`,两组 composite-far 与 wood-stick-far 合计仅 1 次拾取、0 送达。**测量侧**(cold 全 tier 精确为 0,符合零参数 head 设计):learned near `min_authority` 升到 `+0.0013…+0.0050`(v23 修复后重测为 `+0.0010…+0.0043`),medium `+0.0006…+0.0022`;**8/8 body×tier 组合的 `authority_left` 与 `authority_right` 同时为正**,即食物左右差分响应的符号在所有个体上都正确。但绝对方向对齐 near `0/4`、medium `1/4`(v23 修复后重测为 2/4、1/4;n=4 单 seed,该差异在噪声范围内,不构成回归结论)。**新根因收敛**:残余基线 near `0.0105`、medium `0.0117`,仍约为 food 差分量的两倍,且逐 body 有符号(`+0.0102 / -0.0041 / +0.0052 / -0.0105`)。注意 exclusive steering 之后 contrast 轴由 head 独占,该残余**不再是 base 的退化解**(v22r 的 `0.147` 已降两个数量级),而是 head 自身的非食物条件响应(home bearing / heat / PI 状态经同一根执行器轴表达)。因此 `aligned` 这条绝对判据要求"食物驱动压过其余全部转向驱动",当前 authority/baseline 比值 near `0.48`、medium `0.19`,尚未过线。下一步方向应是提高 food 通道相对其他通道的增益或让多通道转向在时间上分离,而不是继续加训练量——v23→v24 authority 增 ~16% 而 baseline 同步增 ~30%,单纯延长训练不改变比值。

通道增益审计与镜像对称根因(只读,`steering_channel_gain.v24-55.json`)：为回答"food 通道是否被其他转向驱动挤占",新增 `scripts/measure_ant_steering_channel_gain.py`,复用已发布的成对探针,按通道分解 contrast 轴上的摆动量与直流量。**第一个假设(food 增益偏低)被证伪**:food 的 `turn_gain` 是四个通道最高的(0.00851,heat 0.00429、home 0.00309、obstacle 0.00058),它不缺权重。真正压住绝对判据的是一个**逐体常数直流**:同一 body 在四种完全不同的刺激下 `head_off` 几乎不变(body0 约 −0.005、body2 约 −0.012、body3 约 +0.002),并以固定比例传导(`head_off × 0.45 = code_off`,`code_off × −3.8 = turn_off`),说明 exclusive steering 之后该轴确实只有 head 一个写入者,直流全部来自 head 残差。**第二个假设(截距项 bias 是直流来源)也被证伪**,且证伪方式提供了正解:在同一 v24 checkpoint 上 profile-gated 地把 `bias` 在 contrast pair 上投影成共模后重测,body0 的 `head_off` 从 −0.0049 降到 −0.0003、四通道全部对齐,但 body1/2/3 反而恶化(body2 −0.0119 → −0.0152)——这三体的 bias 对比原本在**抵消**一个更大的直流,拿掉补偿反而暴露它。该实验改动已回退,只作为反例保留。**收敛到的根因是镜像对称性**:food/heat/obstacle 探针的左右两个 lane 互为精确镜像,故 `head_off` 恰是 head 对输入镜像**对称**分量的响应,`head_swing/2` 是对**反对称**分量的响应。测得对称/反对称中位比为 food `3.92`、heat `2.67`、home `2.65`、obstacle `7.67`,即 head 转向输出的 73–89% 落在镜像对称分量上。一个在世界镜像后不翻号的转向按构造不可能指向任何方向,所以这部分输出是**可证明的非定向解**,与 v22r 的 base "转大圈"同源,只是换到 head 的状态通路里。逐体交叉验证成立:food 的对称/反对称比 body0 `8.2`、body1 `0.85`、body2 `5.5`、body3 `2.3`,而唯一绝对对齐通过的正是比值小于 1 的 body1。由此得出的修法方向是对 contrast 轴施加**镜像等变约束**(`z_contrast(s) := 0.5·(f(s) − f(mirror(s)))`),它按构造删掉全部对称分量并原样保留反对称分量;因四体 food 差分符号已 4/4 正确、heat 3/4、home 4/4,预期绝对对齐直接转正而不损失任何已有能力。该修法需要 embodiment 发布感觉镜像置换(左右触角互换 + 侧向有符号量取负)并由 temporal owner 消费,跨库契约新增,应作为独立收敛包推进,不与本次改动混合。

镜像等变收敛包已落地(`joint_loop.learning.v5 / development.v25 / progress.v21`)：冻结 `ant-sense.ecology-v2` owner 发布完整 19 维 signed involutive permutation(food/heat/obstacle 左右交换，food/heat gradient、home ego sine、pheromone gradient、last turn 等有向量取反)，temporal owner 用同一 zero-history encoder/head 计算原状态与完整镜像状态；steering pair 只保留 `0.5·(f(s)-f(mirror(s)))`，速度轴保留 `0.5·(f(s)+f(mirror(s)))`。pure replay 追加镜像 state/输出镜像 gradient lane，torch 在同一参数图内双 lane 前向；runtime state、capture、settled transition、open segment 与 archive 共同持久化 mirror state，默认 `None` 是完整回滚。owner/训练/持久化回归当前为 108 项通过(temporal+profile+world 56、runtime replay 39、ecology heavy 13)。**v24 参数只读反事实是混合而非单向结论**：在独立 food transfer seed `700003`，不重训即令 near `0/4→4/4`、medium `1/4→4/4`，authority 保持而最大 baseline 约 `0.011→0.0011`；但当时使用训练 seed `0` 的附加 channel probe 仍为 food `0/4`、heat `0/4`、home `4/4`。复核后发现上一段“左右 lane 是完整镜像”的前提不成立：旧 probe 只镜像被测刺激，home/PI/历史转向状态保持原样，因此其 `head_off` 同时含有**合法的其他通道定向驱动**，不能全部解释为镜像对称非定向分量，旧 `mirror_symmetric_ratio` 只能作为 stimulus-pair offset 诊断，不能作为群对称证明。镜像等变仍是正确的物理不变量，但是否让训练学会在觅食阶段压过 home/PI 驱动，必须由全新 v25 门槛决定；已从空目录启动 `research/ant/results/.partials/ecology_p1_v25/seed0`，先看 25 局 food/heat/home probe 与 near/medium 行为，再决定是否续跑 55，P1 继续 `BLOCK`。

v25 干净训练首批结果(35/55)：**镜像等变已打穿 food 转向的 L1 blocker，但完整 P1 尚未结束。**25 局时正式 P1 probe seed `700003` 已得到 food near `4/4`、medium `3/4`；续跑 butter-medium 五局后升至 near/medium 均 `4/4`，35 局复测仍保持 `4/4 + 4/4`，medium `min_authority=+0.0012…+0.0046`、cold 近似精确为零。30 局同一正式 probe 分布上的通道结果为 food/heat/home/obstacle 全部 `4/4`，证明 food 提升没有牺牲逃逸、回巢或障碍感知；诊断脚本也已修正为使用与 P1 gate 相同的 `config.seed+700003`，训练 seed 只用于 journal compatibility。行为侧：butter-medium 五局全部发生拾取，累计 `10 pickup / 3 delivery`；butter-far 五局中三局到达食物，累计 `4/0`，说明“远距完全找不到食物”已从硬 blocker 转为稳定性问题，但拾取后的长程回巢闭环仍未解决。forced-return 五局为 `16/28`，回巢通道本身在 bootstrap 分布上健康；far 的零送达更可能属于长程闭环/记忆泛化，而非 food 左右映射。下一决策点应在完成剩余 burning/composite 与 frozen evaluation 后判断；在正式 report 产出前 P1 仍为 `BLOCK`，禁止用 35/55 的中间证据宣称整体 PASS。

v26 从空目录重跑暴露并修复正式预算断链：curriculum v9 已由冻结几何/plant 推导 near/medium/far milestone 的充分预算为 `28/33/49`，但 P1 formal gate、`EcologyP1Config` 与 CLI 仍各自硬编码 `24`，导致默认“正式”运行在第一个训练回合前被 `_require_samplable_milestones` 拒绝，永远不可能产出报告。现由 P1 直接消费 curriculum owner 的 `ecology_training_min_stage_rounds()`，formal 下限、dataclass 默认和 CLI 默认统一为 `49`；小预算 diagnostic 仍可执行但不能 PASS。干净 seed-0 journal 在该预算下跑到 learned `30/55`：butter-near `3/0`（新几何已移除巢边免费拾取）、burning-near `14/4`、composite-near `1/1`、forced-return `0 pickup / 3 delivery`、forced-approach `7/0`、butter-medium `14/1`。正式 food probe 在 25 与 30 回合均为 near/medium `4/4 + 4/4`；30 回合 authority 为 near `+0.0024…+0.0040`、medium `+0.0010…+0.0016`，cold 近似零。由此 food→turn L1 blocker 已解决且继续增强，当前首要差距转移到 pickup 后的长程 home/PI 闭环（medium 转化 `1/14`）；P1 仍为 `BLOCK`，下一包不得继续放大 food 通道。

v26 carrying-home 根因审计与 curriculum v10 修复：正式四通道 probe 的 home 方向为 `4/4`，但摆动中位数仅 `0.00467 rad/tick`；直接的 carrying-vs-food 冲突 probe 又显示四体在 opposing food gradient 下仍全部向巢，home/food 中位比 `2.05`，故“food 压过 home”被证伪。49-tick deterministic forced-return 回放给出决定性轨迹：四体从巢距约 `1.33` 在第 6–7 tick 收敛到 `0.66–0.70`，未进入半径 `0.5` 的 delivery disc，随后直行到 `8.13–8.47`；最终 PI direction error 仅 `0.0005–0.001 rad`，PI 漂移亦被证伪。根因是旧 forced-return 半径 `1.5`、heading offset `±π/6`：零转向直线最近巢距 `1.5·sin(π/6)=0.75`，按构造永不交付，却在掠过前持续获得正 dense home-progress，优化器可收割“接近但不闭环”的退化解。curriculum v10 改为左右均衡的 tangent start `±π/2`，零转向第一步必然远离巢，只有正确 home steering 才能得奖；P1 升 `development.v27/progress.v24`，P2 confirmatory/shard/progress 同步升 v4，旧 journal fail loudly。

v27 大角度归巢根因审计与 curriculum v11 修复：干净 seed-0 journal 跑到 `30/55`。前 25 局中 butter/burning/composite near 分别为 `3/0`、`14/4`、`1/1`，新 tangent forced-return 五局 delivery 为 `0/0/1/2/1`，3/5 layout 产生真实交付，证明 v10 已堵住旧稠密奖励漏洞且任务可学；forced-approach 为 `6/0`。随后 butter-medium 五局为 `8 pickup / 0 delivery`，弱于 v26 的 `14/1`。30 局只读探针并未发现感知或方向回退：food near/medium、heat/home/obstacle 均为 `4/4`，food authority near `+0.0026…+0.0043`、medium `+0.0011…+0.0018`，carrying-vs-food 仍为 4/4 向巢且 home/food 中位比 `1.97`。决定性 frozen medium trace 中 body 0/3 于 tick 9/10 拾取，拾取后最小巢距仍约 `2.00`，随后发散到 `9.51/9.76`；最终 PI direction error 只有 `0.0008/0.0014 rad`，但拾取后平均绝对转向仅 `0.0166/0.0060 rad/tick`，远低于 plant 的 `0.785 rad/tick` 上限。由此排除 food、PI、通道竞争和执行器天花板，根因收敛为：v10 forced-return 只训练 `±π/2` 修正，未给自然拾取后接近 `π` 的掉头提供足够幅度压力。curriculum v11 改为 `±3π/4`：零转向仍从第一步远离巢，同时保留左右符号；精确 `π` 因侧向信用不可辨而禁用。P1 升 `development.v28/progress.v25`，P2 同步升 v5；v27 journal 只保留为只读根因证据，P1 继续 `BLOCK`。

v28 归巢评估诚实性收敛包（curriculum v12 / P1 `development.v29`、`progress.v26` / P2 v6）：旧 `carrying_home_action_alignment` 只测冻结 checkpoint 的单步动作方向，不能区分“转向朝巢但幅度接近零”和“能完成拾取后大角度掉头”。新增 `post_pickup_uturn_progress` 冻结硬门：每个 learned body 都在真实黄油拾取后分别承受左右 `±3π/4` 初始返向，16 tick 内关闭 policy optimization 与 joint learning并校验两个 owner fingerprint 不变；每条 lane 必须实际交付，或巢距净下降至少 `0.4` 且连续至少 3 步下降，两侧都通过才计入 ≥60% body 门槛。该门只作 evaluation readout，不回灌学习。旧 v11/v28/v25/v5 artifact 无此轨迹证据，全部拒绝恢复或聚合；下一轮必须从空 journal 重跑，P1 在新 report 产出前继续 `BLOCK`。

v29 seed-0 在 curriculum v12 下推进至 learned `50/55` 后按 bounded pause 停止并只读审计：累计 `71 pickup / 43 delivery`，但冻结 post-pickup U-turn 仍为 `0/4 body`，8 条左右 lane 净巢距全部约 `-4.4`、连续下降步数为 0。根因不是 action head 未更新或执行器无权限，而是训练/验收分布断裂：forced-return 在 T0 直接写入 `carrying_food=True`，完全跳过真实 pickup 后 `False→True` observation transition；真实拾取 trace 中下一拍 `is_switching=False`，持久化旧动作族，候选 carrying action 虽已改变却未被采用。另有后半程无 return rehearsal，早期映射容易被覆盖。curriculum v13 因此改为每 body 专属黄油源上的真实 pickup-return 轨迹，保持左右 `±3π/4` 且不注入动作标签；55 局总预算不变，以 5 次交错返向复习替换末尾重复 composite-far block。冻结 gate 新增“拾取后 2 个 action 内必须发生 `is_switching`”并继续要求交付或持续巢距下降。schema 升为 P1 `development.v30/progress.v27`、P2 v7，旧 journal 仅作只读根因证据；下一轮必须清空 journal 从 seed-0 重跑。

v30 MPS seed-0 完成 `55/55`（累计 `126 pickup / 47 delivery`），但冻结 post-pickup U-turn 仍为 `0/4 body`：8 条 lane 都完成真实拾取，却没有拾取后 switch，净巢距约 `-4.35…-4.56`、连续下降步数为 0；单步 carrying-home alignment 同样为 `0/4`。逐拍 owner trace 显示拾取后的候选 code 已变化（body 0 mean-L1 `0.0246`），却因 `beta≈0.514 < learned threshold 0.586` 被丢弃；拾取 PE magnitude 为 `0.4789`，而 ant profile floor 为 `0.5`，故 PE→temporal bridge 精确输出 0。即使越过 floor，旧固定 pressure 上限 `+0.18` 仍可能低于漂移后的 threshold。根因属于 temporal boundary/segment-credit 契约：真实 pickup 没有关闭 outbound segment，carrying action 与后续 return credit 无法形成独立可学习段。修复将超过 profile floor 的 typed PE 定义为方向无关 boundary request，由 temporal owner 相对当前 `beta_threshold` 完成切段；ant floor 校准为 `0.45`，低于 floor 的扰动保持负控不切段。对 v30 最终 checkpoint 的只读反事实中，8/8 lane 均恢复为拾取后第 2 个 action switch，但旧权重仍全部远离巢（净进度 `-4.18…-4.43`，0 连续下降），证明边界修复有效、同时证明旧权重从未学成 carrying-home 映射，不能直接晋级或续训。curriculum 升 v14，P1 升 `development.v31/progress.v28`，P2 升 v8；v30 journal 仅保留为只读反事实证据，v31 必须从空 journal 重训。

**v31 训练前 PE 边界分离裕度实测（BLOCK v31 现行 floor 标定）**：新增只读脚本 `scripts/measure_ant_pe_boundary_margin.py`，用 v30 MPS 55/55 checkpoint（progress.v27，须以 v27 代码 worktree 的 `PYTHONPATH` 装载）冻结重放 forced-return 5 局（ep15–19）与 butter-medium 5 局（ep25–29），共 1892 个非拾取、非 bootstrap 的日常拍。结果推翻了 floor=0.45 的标定前提：**日常拍 PE p50=0.508、p90=0.618、p99=0.701、max=0.984，68.4% 超过 0.45**（55.4% 甚至超过通用默认 0.5）——回归测试里当作“普通亚 floor 噪声”的 0.44 实际上在日常分布的中位数之下。拾取事件按结算时序分两种口径都不成立：(a) 拾取拍**下一拍**（真正结算拾取 outcome 的 PE）在 forced-return 上为 0.673–0.681，与日常 p99=0.701 顶部重叠；而**自然 medium 拾取**（tick 4–6）只有 **0.319–0.327，低于 floor**，即该机制在它最终要服务的自然拾取上根本不触发；(b) 拾取拍**当拍**读数 0.471–0.477（这正是历史标定值 0.4789 的来源——它其实是拾取前一拍 approach outcome 的结算，属日常量）与日常中位数 0.51 无从区分。分离裕度 `min(拾取 PE) − 日常 p99 = −0.38`。还发现分布随行为相位系统性移动：携食/归巢相位 p50≈0.52，outbound 游走相位 p50≈0.40——floor=0.45 恰好落在携食相位分布内部，**正是本次改动要保护的 carrying 段会被逐拍切碎**。结论：任何幅度阈值都无法分离“拾取事件”与“日常预测误差”，“标定”修法（挪 floor + 滞回）被数据整体排除；唯一成立的方向是**结构化**——由 owner 发布类型化边界事件（如 carrying 状态跃变本身），PE 幅度不再充当事件检测器（R-PE：哪个事件关段应是类型化 readout，而非从原始幅度反推）。这是跨 owner 契约变更 + schema 升版，待决策后单独成包；在此之前 v31 训练不应启动。完整分布与逐事件明细见 `research/ant/results/.partials/pe_boundary_margin.v30.json`。

**结构化收敛包已落地（v31 语义就地重定义为 typed milestone boundary）**：按上段判词实施，v31/progress.v28/curriculum.v14 在产生任何 journal 之前把边界语义从"PE 幅度跨 floor"替换为**类型化环境里程碑**，无需再升版。链路（复用既有类型化管道，不新建平行通道）：(1) `vz-contracts` 的 `EnvironmentMeasurement` 新增 owner 声明字段 `discrete_milestone: bool = False`；(2) ant session（环境 owner）只在 pickup/delivery outcome 上声明 True——稠密 local-valence 拍与 heading-stability 的逐拍 `task_progress` 不声明，内核**只读该布尔**、不从 `task_progress` 存在性反推（消费者不得重建 producer 语义）；(3) `AgentSessionRunner.run_turn` 在信号组装时读取仍在缓冲的上一拍 outcome（比 PE 结算路径提前一拍：拾取后第 1 个决策即可切段，优于冻结 gate 要求的"前 2 个 action 内"），转发 typed 信号 `environment_milestone_boundary`；(4) joint-loop 新增 `environment_milestone_temporal_switch` wiring（内核默认 DISABLED=逐字节回滚，ant profile ACTIVE），ACTIVE 下调用 `record_external_boundary_request`；(5) temporal owner 的强制切段逻辑改为消费 `external_boundary_requested()`，相对当前 learned `beta_threshold` 解析（保留 v31 原有的 threshold-relative 机制，只换请求来源）。PE 通道整体降级为**纯加性 prior**：`prediction_error_boundary_requested()` 已删除，`ANT_PREDICTION_ERROR_BOUNDARY_FLOOR=0.45` 撤下（回退内核默认 0.5），PE-off 匹配对照杆只关加性 prior，里程碑通道两臂一致（环境事实非 PE readout）。实施中发现并修复一个会复现 v30 失败模式的时序坑：SSL expert-action 族发现在 full-cycle turn 的**决策前**调用 `reset_episode_runtime_telemetry()`，若该重置清除边界请求，恰好在学习 turn 上把已确认里程碑静默丢弃——现该重置显式不清除请求（turn-scoped 由每拍信号刷新独占写入），session 级测试的第 3 turn 恰为 full-cycle 钉住此时序。新契约测试：`test_external_boundary_request_crosses_learned_beta_threshold`（最大加性 pressure 0.18 压不过 threshold 0.95、typed 请求强制 effective beta ≥ threshold、请求单拍可清）、`test_pe_magnitude_is_inert_for_boundaries_and_milestone_owns_them`（strength×幅度扫描决策全同；milestone 仅 ACTIVE 切段）、`test_ant_milestone_boundary_replaces_pe_magnitude_event_detector`（0.44/0.4789/0.701/1.0 全不请求边界；milestone+0.32 请求）、`test_ecology_outcome_declares_discrete_milestone_only_on_pickup`（owner 声明面）、session 级两条（提交→下一 turn 强制切段；DISABLED 回滚）。v31 重训解除阻塞。

**v31 站1硬早停（2026-07-30，CPU float64，正式 medium 结论仍为 BLOCK）**：seed-0 从空 journal 按冻结 schedule 完成 ep0–19 后，由 `--max-new-work-items=20` 在站边界正常暂停；站1 checkpoint SHA256 为 `a5a944bb28b7de8ac301875a76bb610a1e8acb42e0a8acbdb2afa4b3716f089c`。行为块为 butter-near `6 pickup / 1 delivery`、burning-near `18/3`、composite-near `2/0`、forced-return `23/5`（pause callback 在 episode 19 原子保存后、stdout replay 行前抛出，故日志最后一局缺行；最后一局 `6/2` 由 `learned.json` 原子 journal 给出）。按预注册计划只能与 v24 同位 `45/33、35/18、39/33、forced-return delivery 29` 比，四段均低于 80%，因此站1 **FAIL / EARLY STOP**，禁止启动 forced_approach/medium 站2，不能用增加训练量补救。

结构链与能力链的归因分开：(1) 新增 `scripts/measure_ant_post_pickup_family_persistence.py`，复用正式 frozen ±135° U-turn lane 并从 journal checkpoint 发布逐 action exact family、首个 switched family、直到下一次 β switch 的连续存活 action 数和右截断标志；正式 U-turn gate 同步要求 family persistence ≥3 action。(2) 20-episode checkpoint 的 8/8 lane 都真实 pickup，在第 1 个 post-pickup action 切换，随后 15 action 内无第二次 β switch（全部右截断），policy/temporal-learning fingerprint 全稳定；D4 与 typed milestone 结构门通过，排除“切了又回跳/边界请求丢失”。(3) 同 checkpoint 的 food transfer probe 在 near/medium 都是绝对对齐 `0/4`，learned `min_authority` 近似 0（body0 甚至反号）；能力断点位于 outbound food→turn 映射尚未形成，不能进入 medium 后再把失败归到 carrying-home D1。

本早停还暴露一条**基线可比性测量债**：v24 之后 curriculum v9 已改变 near 几何，本文既有 v26/v27 同几何首 15 局基线分别约为 `3/0、14/4、1/1`；v31 的 `6/1、18/3、2/0` 相对该口径没有 v24 对表所暗示的数量级崩塌。因此本次证据只支持“违反预注册 v24 80% 门，必须早停”，**不支持**“typed milestone 导致 near 回归”。门槛在看数前已冻结，不能事后改用 v26/v27 继续站2；后续若重开 v31，必须先单独预注册物理课程一致的 baseline packet，再从空 journal 运行。

验收期间并行会话先后启动两个旧版 writer（`--max-new-work-items=10` 与 `8`），并把计划文件的预注册 v24 80% 回归门改写成 v30 观察项后继续站2；发现时均以普通 SIGTERM 停止。它们已把同一双槽 journal 推进到 ep23，已提交 ep20–22 forced_approach 为 `0/0、1/0、1/1`，未提交局丢弃。双槽已覆盖原 20-episode archive，故当前 journal **不得冒充站1 checkpoint**；上述 D6 artifact 仍以 `a5a944...` 绑定原站1 checkpoint，food probe也记录 `completed_training_episodes=20`。22-episode channel probe 只能作受污染补充读数（food/home 均 `0/4`），不得进入站1硬判词或 medium 结论。计划 todo 已恢复为 EARLY STOP，后续站标记为“按预注册早停不执行”。

债务处置：D1 因站1早停未进入 medium，保持 OPEN；D2 的终局 authority/baseline 比未测，保持 OPEN，22-episode 污染读数不替代正式包；D3 far 未运行，保持 OPEN 且继续单独立项；D4 CLOSED（8/8、latency=1、persistence≥15）；D5 保持 v31 后的 matched PE-on/off P2，且两臂 milestone 必须 ACTIVE；D6 CLOSED；D7 原“按 v24 episode 位次同位”被实测证明不足以保证物理可比，升级为 baseline packet 债；D8 本次全程 CPU float64，MPS 校准仍单独立项；D9 既有失败项不混修；D10 detach/原子 journal 工作正常，并已追加 progress-dir 非阻塞进程级 `flock`：第二个 CLI writer 在消耗预算前 fail loudly、报告持锁 PID，进程退出由内核自动释放，不留 stale sentinel。medium 是否闭环的终局结论是 **未验证 / BLOCK**，不是 FAIL：站2在预注册早停下没有获得合法执行权限。

站1后按工具箱重跑完整 P0 mechanism audit 时又发现 provenance 前置债：默认 training episode 1 的 seed `101` 与 literal frozen held-out repro seed `101` 重叠，且旧 driver 在完成全部训练后才调用 schedule owner，白耗预算才 fail loudly。修复把 P0 training 移到唯一 owner 的 `config.seed+1_000_003` 高位命名空间，并把同一 `ecology_mechanism_audit_seed_schedule` preflight 前移到训练前；validation `config.seed+43`、held-out `101/307` 保持冻结。修后正式 CPU/pure audit bundle 已产出于 `research/ant/results/ecology_recovery/p0/20260729T164048Z-seed0-e7ba7360/`，provenance 明确 dirty、`externally_retainable=false`，verdict 仍为诚实 **BLOCK**：action final sensitivity 仅 `2/4` body，body0/1 的 food/heat turn delta 约 `1e-16`；body0/1 action-head update step 仍为 0；torch SSL 因 trace 少于 2 step 没有执行；pure/runtime/torch parity 超过 `1e-3`；frozen evaluation 在 tick 0 就观察到 memory/credit/dual-track/prediction/reflection/regime owner fingerprint 变化。positive/negative temporal control、environment-milestone closure、segment-credit parity 则通过。该 bundle 支持“时间边界机制生效但 action/optimizer/freeze 主链未闭环”的归因，不授权站2。

**action-head 更新可达性收敛包（2026-07-30）**：逐 body 首局时序证明 Digital Ant evidence profile 的真实 batch target 为 4，而非通用默认 1；最短 P0 episode 只有 12 turns、capture/bootstrap 后 10 个可用 settlement，旧 16-step segment 上限使 body0 完全不闭段，body1/3 只留下 2-transition 短段，随后被跨 episode `include_runtime_replay=False` 导出按契约清空。通用 temporal 调度同时修正 target=1 的 ready ACTIVE replay：已达到 target 的 settled batch 本身现在会在下一 scheduled step 触发 full cycle，不再等待无关 PE/RL cadence。Digital Ant profile 的 bounded horizon 收敛为 7（仍大于 4-transition batch，并保留一拍 flush 窗口）；4/5/6-step 受控对照未改善四体最终 sensitivity，4-step 还命中既有困难布局 pickup 回退反证，均未采用。

同配置最终 P0 bundle 为 `research/ant/results/ecology_recovery/p0/20260730T043650Z-seed0-e7ba7360/`，仍因独立债务诚实 **BLOCK**，但最早 optimizer 断点已关闭：四体 action-head update step 从 `(0,0,3,3)` 变为 `(3,2,5,5)`，`action_head_update_applied` 与 `action_chain_no_rollback` 均转 PASS，final sensitivity 从 `2/4` 提升为 `3/4`。残余 action 断点是 body0 food/heat delta 仅 `4.81e-7 / 1.47e-5`，且三 seed posterior-noise repeat 下所有 body 的非零符号都未稳定，故 `action_chain_final_sensitivity` 与 `action_chain_sign_consistency` 保持 BLOCK。专属 head 步长 4×/8× 的只读试验虽放大 body1–3 residual，却没有修复 body0 food covariance 或跨 seed 符号，已拒绝；P1 已采用的 forced-approach 起点在 12-turn P0 预算内同样未改善 body0，未写入 P0 schedule。backend coverage/parity 与 frozen-evaluation 仍保持独立收敛包，不在本包混修；该结果不授权重开 v31 或启动站2。

后续 action-chain 收敛包先关闭了 probe 测量漂移：旧 probe 只钉 world pose，navigator 仍保留 seed-dependent 随机出生航向，导致 sign repeat 实际比较不同的 `home_ego_*` 输入；现每个 probe tick 同时 `sync_to` 正式 pose/home vector，跨 seed sense 逐值相同。该修复证明旧符号翻转包含测量伪差，但没有掩盖真实幅度债。随后把 Ant profile 的 transition batch 从 4 收敛为 2——仍是 centered covariance 的最小非退化 batch，target 4 则把 body0/1 的有效更新推迟到冻结 P0 预算外；horizon 6 相对 7 无增益，保持 7。mirror-equivariant head 把 paired contrast 等分到两侧，sign gate 因此先要求原封不动的 paired `turn_delta >= 1e-4`，再以半阈值分类两侧方向，避免把同一个冻结 sensitivity floor 重复要求两次。最终正式 bundle `research/ant/results/ecology_recovery/p0/20260730T054631Z-seed0-e7ba7360/` manifest 已校验：update step 为 `(6,3,8,8)`，`action_chain_final_sensitivity` 达 `4/4`，三 repeat 的 8 个 food/heat 左右符号组合全部稳定，action-chain 五门全 PASS。总 verdict 仍为诚实 **BLOCK**，剩余断点仅 `backend_lane_coverage / backend_parity / frozen_evaluation`；它们继续作为独立收敛包，本结果仍不授权重开 v31 或启动站2。

backend 收敛包随后关闭了两个真实性缺口。Ndim runtime 的短 CMS context 改为与 pure backend 相同的循环投影，不再用零填充制造伪差；session 保留最近两个真实 substrate snapshot，使 torch SSL 在 6-step exercise 中得到非 singleton trace。backend parity 改成“两阶段”：旧 session 只用于证明声明的 backend 确实执行和写回，随后 pure/runtime/torch 都从同一 checkpoint 恢复到 fresh world/session 再做同态前向，禁止把 exercise-local 参数或 recurrent state 混进 parity。正式中间 bundle `research/ant/results/ecology_recovery/p0/20260730T064050Z-seed0-e7ba7360/` 中三 lane coverage 全部通过：runtime 最大 code/turn delta 为 `1.11e-16 / 1.33e-16`；torch SSL `trained_steps=1`、`2192` 个参数变化并写回，Internal-RL `335` 个参数变化并写回，fresh-checkpoint parity 全部精确为 0。此时总 verdict 仅剩 `frozen_evaluation`。

最后的 frozen-evaluation 收敛包把 `joint_learning_enabled=False` 从 joint-loop/temporal 硬边界传播到 memory、PredictionError、credit、regime，并关闭 session-held dual-track gate 与 reflection consolidation settlement。只读推理仍保留：memory retrieval 不触碰访问/recall 学习统计，PE 仍发布 next-turn context，credit/replay 仍结算 lineage，regime 仍可切换 active runtime identity；持久学习状态不写。Regime owner 同时发布排除 active identity、turn index 与持续轮数的学习指纹，修掉“合法推理态被误算为学习漂移”的契约缺口。最终默认预算 CPU P0 bundle 为 `research/ant/results/ecology_recovery/p0/20260730T072554Z-seed0-e7ba7360/`，artifact manifest 已独立校验，十四道 gate **全部 PASS**：两组 literal frozen repro（butter-only seed 307、heat-forced-escape seed 101）各运行 24 rounds，八个 gated owner 的 `unstable_owner_names=()`，policy/temporal-learning 全程稳定，replay settlement/lineage 均为 `1.0`、drop=`0`；backend coverage/parity 与 action/temporal/segment gates 同时保持 PASS。manifest 如实标记当前 dirty worktree，故 `externally_retainable=false`。这关闭的是站1后新增的 P0 工程债，不改变预注册站1早停：v31 medium 仍是**未验证 / BLOCK**，D1/D2/D3/D5 等能力债仍需新的、先注册物理同口径 baseline packet 与全新 journal 才能重开，现有证据不授权补跑站2。

**同物理 baseline 预注册包（2026-07-30）**：新增
`digital-ant-ecology-same-physics-baseline-preregistration.v1` 生成器与严格校验器。两臂从同一
初始 checkpoint 分叉、绑定同一 curriculum v14 / P1 v31 / progress v28、seed-0、CPU float64 和
55-episode schedule；自动逐字段审计 rollout config，唯一允许差异是
`environment_milestone_temporal_switch=ACTIVE / DISABLED`。历史 v24、v30 与旧 v31 station1
全部标记为 `EXCLUDED`，不再参与判定。阈值在新结果产生前冻结：站1 active 总 pickup ≥ control
的 80%（delivery 只作稀疏观察）并通过既有 switch/persistence 门；站2 medium pickup ≥80% 且
delivery 严格超过 control，同时 carrying alignment/U-turn 转正；任一失败即 BLOCK，不授权后继
station。packet 绑定九个相关 owner/consumer/runner 源文件 SHA256，运行前若代码漂移必须拒绝而不是悄悄
换实验。D7 从“待定义 baseline”推进为“预注册契约已实现，待正式 matched journal 执行”。

最终可执行 prereg bundle 为隔离运行快照签发并回收到主仓库的
`research/ant/results/ecology_recovery/same_physics_baseline/ecology_same_physics_prereg.seed0.20260730T095738Z.json`
及其 manifest。`20260730T093928Z` 产生在可恢复 runner 纳入 source binding 前；`094415Z` 虽含
runner，却漏绑实际修复的 `session_observation.py` consumer，在 control 提交 2 局后经复核主动
SIGTERM 停止，其 journal 保留为作废审计；`095220Z` 已补齐完整 code-tree binding，但共享 worktree
在前飞与 detached 启动之间又发生 Python 源码漂移，后台入口按契约在耗预算前拒绝，journal 停在
control 1 局。三者均禁止执行。为消除并发 writer 对源码的影响，最终 `095738Z` 在
`/private/tmp/volvence-ecology-baseline.t0n2fu` 隔离快照内签发并执行：快照固定 `packages/`、
`scripts/` 与配置，Git objects 只读引用主仓库，不复制历史。该包同时显式绑定关键文件与整个
`packages/**/*.py + scripts/**/*.py + **/pyproject.toml + uv.lock` 聚合哈希，运行入口会在消耗预算
前重算并拒绝任何源码漂移。最终包校验值：完整 schedule
`57f0e58def9c562efc29f43feb20e574ef9279129cc691d3ada8e0b7de5d9e45`、station1 prefix
`12b27ab238dfb06980d8b6001c17bf1222155e5920cb4b373a02b8c6d292298f`、两臂 matched fields
`49c9db71a4dfa6b5361c7cf06f72f3a7759c534eb51e1ee1cb8495aa40a899cf`；dirty worktree 使
`externally_retainable=false`。正式新 journal 位于隔离快照的
`research/ant/results/.partials/ecology_same_physics/seed0-20260730T095738Z`。前飞 control ep0 已
原子提交且 checkpoint/report 均为 1；余下 station1 由 detached PID 58625
继续执行。期间还关闭了一个真实 preflight blocker：Relationship conditioning consumer 未检查
声明 wiring，错误地把 SHADOW readout 当 text 交付，与 personal residual 形成双 delivery；
现只有 `relationship_conditioning=ACTIVE` 才能进入 carrier，37 条 baseline/runtime/multi-bank
相关测试通过。

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

## 2026-07-31：同物理 station1 判词、persistence 归因与 dwell 收敛包

隔离快照 `095738Z` 的正式 matched station1 已完成并回收到
`research/ant/results/ecology_recovery/same_physics_baseline/`。control 为
`54 pickup / 17 delivery`，milestone-active candidate 为 `46 / 6`；pickup 比
`0.85185` 通过 80% 非劣，control signal 与逐 block 非零门也通过，但 candidate 的 frozen
post-pickup family survival 为 `[1,1,1,1,1,1,15,15]`，因此
`typed_milestone_structure=FAIL`，终局 **BLOCK**，没有授权 episode 20。

包A 的只读归因 artifact 为
`persistence_churn_attribution.seed0.20260730T095738Z.json`。逐体轨迹显示 body0/1/2
在两条 matched arm 上都持续 beta 超阈，`steps_since_switch` 每拍重置为 0；body3 首拍后
降到阈值下并累积到 14。fast prior 全零，PE pressure 为零或小于 `0.001155`；milestone
DISABLED control 复现同一 6/8 churn，故 typed request 不是根因。八条 lane 在 pickup 前后
选择的 family label 全部相同，turn delta 仅 `2.1e-5…2.6e-3`，carrying 专属 family 尚未形成。
冻结 checkpoint 上把 segment horizon 7→16 的完整 probe payload 逐字节相同，证明 horizon
不进入 serving forward；它若有影响，只能经训练态改变 learned beta。历史 `a5a944…`
checkpoint 的 immutable probe artifact 仍证明旧态 8/8 survival=15，但 archive 已被已记录的
并行 writer 覆盖，无法在当前代码上交叉 hydrate，此限制已写入 attribution。

包B 选择 temporal-owner 通用 min-dwell，不降低 persistence 门、不加入 ant/carrying/action
字符串规则。`FinalRolloutConfig.temporal_post_switch_min_dwell` 默认 DISABLED，
`temporal_post_switch_min_dwell_actions` 默认 0；Ant profile 显式 ACTIVE/4。switch action
计第 1 拍，窗口内仅抑制自然 beta termination，新的 typed milestone 可立即打断；SHADOW
只记录 would-suppress，DISABLED/0 回滚。原 station1 learned checkpoint 的隔离冻结预检把
survival 提升为 `[4,4,4,4,4,4,15,15]`，8/8 首拍切换、学习指纹稳定，probe ACCEPT。
验证：temporal owner 105/105、same-physics/profile 13/13、ecology 定向 11/11 通过；
`tests/contracts` 为 3954 passed / 2 skipped，另有 4 个既有失败和 5 个 sandbox 禁止绑定
localhost 的错误，与本包无关；整文件 Ruff 仍命中主仓并行重构的既有 unused/line-length，
本包独立文件 Ruff 与三个核心文件 `py_compile` 通过。

包C 使用 schema v2 预注册：有效 bundle 为
`ecology_same_physics_prereg.seed0.20260731T052300Z.json`，完整源码树 binding 已在隔离快照与
主仓库双重验证。两臂仍只允许
`environment_milestone_temporal_switch=ACTIVE/DISABLED` 一个差异；ACTIVE/4 dwell 在两臂
相同。station1 另冻结 food alignment 形成时点：四个 causal gate 通过且 4/4 对齐才直接授权
episode 20；否则只授权 5 局 butter-near review 和一次 frozen re-probe，仍不足 4/4 即 BLOCK。
若触发 review，五局严格按 packet 已绑定的 station schedule rows 0–4 重放，即 seed
`10000/10101/10202/10303/10404` 与原 rollout config，不新增 seed 选择自由度。
在读取新 station1 结果前，review 执行契约也已单独签发为
`ecology_same_physics_alignment_review_prereg.seed0.20260731T053814Z.json`：它绑定旧
station1 prereg SHA、当前完整源码树、五局 schedule digest、单次 probe 与
GO/BLOCK 授权；review progress 还必须绑定实际 station1 candidate checkpoint SHA。
隔离快照为 `/private/tmp/volvence-ecology-review.8I2FFV`，packet 与 manifest 已在该
快照内复验。只有 station1 明确发布 `alignment_review_authorized=true` 时才注入 checkpoint
并执行；直接 GO 或 causal BLOCK 都不会消费这五局预算。
`051841Z` 因隔离快照漏复制外层 `pyproject.toml` 被 code-tree validator 拒绝，保留为
`INVALID_CODE_TREE` 审计件，禁止执行。正式新 journal 位于
`/private/tmp/volvence-ecology-dwell4.ul0hkm/research/ant/results/.partials/ecology_same_physics/seed0-20260731T052300Z`；
control ep0 已原子提交，余下 station1 由 detached PID 21059 执行。
