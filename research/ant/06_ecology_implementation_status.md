# Ecology P0/P1 实施状态

> 更新时间：2026-07-25。P0 已完成；P1 runner、诊断矩阵与 home-action probe 已实现；P2 因 P1 尚未 PASS 而未启动。

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
