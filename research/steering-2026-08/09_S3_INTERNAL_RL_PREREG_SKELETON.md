# 09 · S3 本体：Internal RL 学"何时/多大力扳"预注册骨架

状态：**设计冻结（S3-A 已背书）**。门控余量审计 [10_GATING_HEADROOM_AUDIT.md](10_GATING_HEADROOM_AUDIT.md) = PASS 已实测证明本仪器有可测且完全可观测的门控余量（余量 0.70 vs always-on、增益 1.72 vs noop、staleness 可检测性 1.0）。本文件冻结设计与判定门；正式执行的 source SHA / seeds / episode 预算固定在配套 prereg JSON。

S3 本体是 S3 前置（[08_READ_STEER_S3_PREREQ.md](08_READ_STEER_S3_PREREQ.md) = PASS）之后的**新 claim**，不重解释任何已封存 verdict（`kill-eta` / S2-static FAIL / B screen FAIL 均不变）。前置已把三层里的两层钉死——**读得到**（冻结线性 reader heldout 1.0）+ **扳得动**（rank-8 执行器关掉 2.79 NLL、条件性优 1.37）。S3 只做第三层：**用 PE/结局信用在线学"何时、往哪、多大力"出手**——方向盘和读盘都验过了，S3 才是"学开车"。

## claim_scope

`conditional-steering-internal-rl-when-to-steer`（screen/证据道；只决定是否准入独立权威 sweep 与是否把门控策略从 SHADOW 提升，**不改 production WiringLevel、不回灌 evaluation、不训练 substrate**）。

## source_lineage（绑定既有封存证据 + 复用边界）

- S3 前置 PASS：`artifacts/eta_s3prereq_readloop_20260805/`（reader heldout 1.0、online==oracle 0.023、优 uncond 1.37、bootstrap CI 下界全 >0）
- S3-A 门控余量审计 PASS：`artifacts/eta_s3a_gating_headroom_20260805/`（余量 0.70、增益 1.72、post-switch 占比 0.43、staleness 可检测性 1.0；post-switch 子集 always-on 4.16 > noop 2.53 = 错条件净损）
- C2 执行器：`artifacts/eta_conditional_steering_screen_20260804/` + `eta_conditional_steering_screen.py`（rank-8、no free bias、zero-code strict no-op）
- C1 仪器：目标剥离路口 VALID（`06_*`；(view,subgoal)→action 残余歧义 0 ⇒ 结局信用**稀而准**的结构前提）
- **复用边界（修正）**：`vz-temporal` 的 `CausalZPolicy` / `InternalRLSandbox` 深度耦合 `MetacontrollerParameterStore` / `Track` / z 空间，是 ETA 元控制器策略，**不可复用**为通用门控 RL。S3 **只复用信用契约**——`sparse_proof_reward_taxonomy`、`InternalRLDelayedCreditAssignment`（`volvence_zero.internal_rl.environment` 的 dataclass 语义）；**策略本体在 S3 owner 模块内自写最小 REINFORCE+baseline**（与 C2 自带 Adam loop 同模式，几十行，动作空间小、horizon 短、无 critic/PPO）。`vz-runtime` 是唯一可 import 业务 wheel 的层，此 import 方向合法。

## 系统语义映射（三时间尺度 / R3·R4·R-PE）

| 层 | 组件 | 时间尺度 | 状态 |
|---|---|---|---|
| 识别（sensor） | 冻结线性 condition reader（08） | rare-heavy 离线拟合、运行时冻结 | 已证 heldout 1.0 |
| 干预（executor） | rank-8 乘性写入（C2/08） | 离线训练、运行时冻结、可回滚 | 已证扳得动 |
| **策略（policy）** | π(a_t \| readout, PE) 的门控 | **online-fast**（本 claim 唯一在线更新处） | **S3 待证** |

策略是**唯一**在线更新的有界控制器；substrate、reader、executor 权重在 S3 期间全部冻结。信用来自结局/PE，不来自 token 空间、不来自 evaluation 读数。

## 直面四个硬问题（用户先前提出，逐条给出诚实答复）

1. **K 轴好定义、subgoal 映射难**：已由 08 的 refit 冻结线性 reader 解决（heldout 1.0）；S3 直接消费该 readout（subgoal index + margin），**不再假设 oracle**。
2. **PE/结局怎么归因**：用既有 `InternalRLDelayedCreditAssignment` + `sparse_proof_reward_taxonomy`——只发**结局级**信用（`proof_terminal_success/failure`、`proof_subgoal_complete` delayed、`proof_distractor_penalty`），`proof_subgoal_progress` 等 shaping 全部 `optimizer_visible=false`（仅诊断）。信用按 `credit_horizon` 回传到路口的门控决策。
3. **用什么 RL 算法**：S3 owner 模块内**自写最小 REINFORCE + running baseline**（advantage = 结局回报 − 滑动基线），动作 = **离散门控**（下方动作空间）。**不复用** `CausalZPolicy`（耦合 ETA z 空间，见 source_lineage）。短 horizon + 小动作空间 ⇒ 无需 PPO/critic；**禁止 token 空间 RL、禁止端到端更新基底**。
4. **样本空间太大 / 稀疏快速主动学习是否成立**：在代理迷宫里 K=8 离散 subgoal + 小离散门控空间，样本高效——但这**不假设**，而是设成**判定门**（sample-budget 收敛门，见下）。若代理上都学不动，则 claim FAIL，直接暴露"稀疏快速学习"不成立，避免在 companion 上盲目烧标注。

## 门控问题的具体化（staleness 仪器，来自 S3-A）

余量必须来自"错条件是净损"。S3-A 用的诚实非 oracle 失败模型即 S3 的问题设定：路线经过 objective 后 active_subgoal 切换；agent 持有的 **belief** 来自上一步上下文的 reader 读出（记忆滞后），在 **post-switch 路口过期**。此时：

- 用过期 belief 出手 = 错条件 = 灾难（S3-A post-switch 子集 4.16 > noop 2.53）；
- 收手（noop）= 平庸但安全（2.53）；
- 该出手时（belief 正确）出手 = 近天花板（fresh_ceiling 0.06）。

**存在 hard 上界规则**（belief==fresh 才出手，S3-A pe_gate == oracle_gate = 1.09）。因此 S3 的 claim **不是**"只有 RL 能达到该门"，而是**在只给稀疏结局信用、只观测 PE 代理、从不给每步对错标签下，RL 能否小样本内学到逼近该上界的门控**——对应 companion 无免费每步标签的真实约束。

## configuration（拟冻结）

- **传感器（冻结、只读）**：08 的 refit 冻结线性 reader（`fit_condition_reader` 产物）；每路口产出 fresh 读与滞后 belief 读（各含 subgoal_index + margin）。不随 S3 更新、不回灌。
- **执行器（冻结、有界）**：C2/08 同款 rank-8 乘性写入 `U·(tanh(Z[k])⊙Vᵀh)`，`free_bias=false`、`zero_code_strict_noop=true`；layer 20 / 896；权重冻结，S3 只由门控**选择/缩放**其输出（条件 k = belief 读出）。
- **策略观测（PE 代理，无 oracle）**：`(belief_margin, fresh_margin, belief≠fresh 一致性位, base 动作熵)`——真 PE（预测 vs 结局）只进信用，不进策略观测。
- **策略（在线，唯一可更新）**：π(a_t | 上述 PE 代理) 的门控，动作 **`{noop, steer(s∈levels)}`**（steer 用 belief 条件驱动冻结执行器，s = 有界档位，受同一 norm cap）。参数远小于执行器；自写 REINFORCE+baseline。
- **信用/环境**：复用 `sparse_proof_reward_taxonomy` 语义（terminal + delayed optimizer-visible；shaping/diagnostic 全 `optimizer_visible=false`）；路口结局由 C1 冲突结构 + expert 动作客观判定（steer 后动作是否匹配 expert）。
- **判定臂（matched budget，同施力预算 + 同 episode 预算）**：
  - `pe-gated-online`（本方案：策略按 PE 代理决定何时出手）
  - `always-on-belief`（每步以固定档位用 belief 出手）
  - `random-gate`（同频率随机出手）
  - `noop`
  - `oracle-gate` / `pe-hard-gate`（真值边界 / belief==fresh 硬规则，作**可达上界诊断**，不进主判、不作信用）
- corpus：复用 C1 冲突映射版；train/heldout 路由不相交，带 SHA provenance。

## thresholds / decision_rules（拟定）

0. **可达上界诊断（参照，非门）**：oracle-gate / pe-hard-gate ≈ S3-A 的 1.09 NLL；`pe-gated-online` 目标是从稀疏结局信用**逼近**该上界（noop 2.81、always-on-belief 1.79、fresh 天花板 0.03）。
1. **sample-budget 收敛门（先决，回答"稀疏快速学习"）**：在 ≤ `max_online_episodes`（拟 ≤ 若干百 episode）内，`pe-gated-online` 的 heldout 结局收益相对起点显著上升且 bootstrap 95% CI 不跨 0；否则判 `slow-or-nonconvergent`，如实记录"稀疏快速主动学习在本代理上未成立"。
2. **学到"何时扳"（因果主判）**：heldout 上 `pe-gated-online` 负 NLL/结局正确率 > `noop`，route-level bootstrap 95% CI 不跨 0。
3. **优于恒定出手**：`pe-gated-online` > `always-on-belief`（同预算下"择时"净收益，CI 不跨 0）——这是"学开车"有价值的关键门。
4. **优于随机门**：`pe-gated-online` > `random-gate`（CI 不跨 0），排除"施加频率/预算"混淆。
5. **门控可解释**：策略在**冲突路口**的施加率显著高于**非冲突步**（施加集中在"该出手"处），量化并报告。
6. **信用稀而准**：optimizer-visible reward 仅 terminal+delayed，shaping 全 `optimizer_visible=false`；报告 sparse-only 占比。
7. **结构完整性**：reader/executor/substrate 全程冻结（`substrate_trainable=0`、reader/executor `params_changed=false`），仅策略参数更新；no free bias、zero-code strict no-op；不读 active_subgoal 真值进策略损失（真值仅用于 oracle-gate 诊断与结局判定）。

全过 → 准入独立权威 sweep（再决定是否把门控从 SHADOW 提升）；任一未过 → 封存 FAIL，不改写任何既有 verdict。

## prohibited_after_execution（拟定）

- 读结果后改阈值 / 主判臂 / 动作空间 / seeds / episode 预算；
- 训练或微调 reader / executor / substrate（S3 只更新门控策略）；
- 用 active_subgoal 真值进策略损失（只可作 oracle-gate 诊断与结局判定）；
- 加 additive/常数 bias 或让 zero-code 非 no-op；
- 在 token 空间做策略学习 / 端到端更新基底；
- 把 evaluation 读数回灌学习或改 production WiringLevel；
- 重贴已封存 `kill-eta` / S2 / B screen / C2 / S3-前置 verdict。

## frozen prereg

正式判定门已冻结为 `artifacts/eta_s3_internal_rl_prereg_20260805.json`（SHA256 `62454418…`）：claim_scope、问题设定（staleness 仪器 + PE 代理观测）、动作空间、arms、seeds `[0..4]`、episode 预算、bootstrap 5000/95%、全部 decision rules 与 prohibited 项。**pre-execution 修订**（已记入 prereg `pre_execution_revisions`）：1-seed 探针显示 300 route-episodes / 64 routes（~4.7 次/route）供给不足（收敛向 always-on、selectivity≈0），故改用 **minibatch REINFORCE（batch 8）** 并把预算提到 **1200**（~19 次/route，仍有界诚实）；阈值/arms/seeds 不变。S3-C owner 模块 threshold 默认值须与 prereg 一致（run 脚本 `_assert_prereg_consistency` 强校验）；S3-D run 脚本在 `artifact_manifest.json` 记录各 source SHA + 模型 weights SHA + prereg SHA。

**S3-E 稳健化增补 prereg**：`artifacts/eta_s3e_internal_rl_restart_prereg_20260805.json`（SHA256 `e46b5890…`），在**看到 S3-D 结果后、跑 S3-E 前**冻结。仅新增一条训练侧稳健化机制——**multi-restart（每 seed 4 重启）+ 训练侧 argmax-gate NLL 选最优**（选择只读训练行，从不看 heldout 判据；塌缩的 always-steer 策略被 post-switch 7.0 惩罚拖累、训练 NLL 严格更差，故 best-train 选择可证拒绝塌缩重启，等价诚实验证集模型选择）。**判据 / arm / seeds / episode 预算 / bootstrap 全部继承 S3-D 不变**，S3-D 的 literal FAIL 记录原样保留。结果见 [11](11_S3_INTERNAL_RL_RESULT.md)：admission **PASS**（5/5 seed），worst-seed gain-vs-always-on CI 下界 +0.497。

## 退出 / 回滚

全为 evidence lane。策略以 SHADOW 训练与评估，不安装、不写回、默认 wiring 不变；回滚即删除 S3 artifact 与新接线代码路径。按收敛包纪律：单 owner（策略/环境接线）、单契约（门控动作 + 结局信用语义），与 reader/executor 分包，任一时刻同一快照链只有一个写入者。

## 到 companion 的迁移路径（明确边界）

代理迷宫里结局信用**稀而准**是"免费"的——C1 保证 (view,subgoal)→action 残余歧义 0，"到达/进 distractor"是客观结局。**这正是 companion 与代理的分界**：在真实陪伴任务里，"这一步该不该扳向关系轨"的结局信用**没有免费客观标签**，需要情感专家标注 / 长程关系结果作为稀而准的 terminal 信用来源。S3 代理证据的作用是：先证明**给定稀而准的结局信用，门控策略能在小样本内学会择时**；若成立，才有理由在 companion 上投入昂贵的专家标注去提供同类信用——顺序不能反。
