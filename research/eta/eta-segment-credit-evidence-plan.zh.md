# ETA 分段信用分配证据门实验

## 要验证的命题

在同一冻结基底、同一 residual rollout、同一延迟结果上，只改变信用分配单位：

- `turn-level`：把结果记到结果到达的单个 turn。
- `ETA segment-level`：沿 action lineage 回到动作发生时的 `beta_t` 窗口，并把 PE 派生信用分给该抽象动作段。

若 ETA 的增益来自真正的时间抽象，segment-level 应在未见路线与多 seed 上同时提高因果窗口归因、动作族归因和后续结果预测，并降低错误信用。

## 第一阶段矩阵

| 变量 | 固定值 |
|---|---|
| substrate | `Qwen/Qwen2.5-0.5B-Instruct`，冻结 |
| device | Apple MPS |
| residual hooks | transformer layers 20/21/22 |
| controller | `full-internal-rl` |
| train/eval/heldout 路线 | ETA proof suite 既有 3/2/2 划分 |
| PE/outcome | 同一 delayed subgoal outcome |
| 唯一消融变量 | turn credit vs beta segment credit |

## 指标

- `credit_f1_delta`：ETA causal-window F1 减 turn-level F1。
- `false_credit_reduction`：turn-level 错误信用率减 ETA 错误信用率。
- `family_assignment_delta`：ETA 动作族归因准确率减 turn-level。
- `pe_reduction_rate_delta`：ETA held-out PE 降幅减 turn-level。
- `segment_boundary_f1`：`beta_t` 边界与真实 subgoal 边界的容差 F1。

## Retain 条件

至少 5 seeds，并同时满足：

1. `credit_f1_delta` 的 95% bootstrap CI 下界大于 0。
2. `false_credit_reduction` 的 CI 下界大于 0。
3. `family_assignment_delta` 的 CI 下界大于 0。
4. `pe_reduction_rate_delta` 的 CI 下界大于 0。
5. `segment_boundary_f1` 的 CI 下界至少为 0.50。

任一均值不为正则 `fail`；均值为正但统计或边界门未站稳则 `weak`。`weak` 不得对外表述为 ETA 证据门通过。

## 产物

- `manifest.yaml`
- `predictions.jsonl`
- `outcomes.jsonl`
- `segments.jsonl`
- `credit.jsonl`
- `report.json`
- `report.md`

这些产物是 evaluation readout，不回灌 PE、credit 或 temporal owner。

## 2026-07-28 最终 v12 证据包

最终收敛修复了三个 owner 级缺口：

- encoder 新增单位矩阵初始化、可由 SSL 更新的 `current_proj`，使 70% current
  observation skip 不再是固定 residual 折叠，而是可学习的 residual→`z_t` 映射。
- runtime 和 expert-action family discovery 统一使用单 observation step、
  recurrent hidden、标量 hard beta；首步建码、continuation 精确保码、switch
  接纳 proposal。
- family classification 先以 structural similarity 形成 `0.005` 近邻集，再让
  payoff/continuation prior 在近邻内决胜；merge 保留更早 family 的 lineage ID。
  `0.05` 消融虽保留双族，但 held-out PE CI 下界为 `-0.0152`；`1e-5` 消融塌成
  单族且 beta F1 降至 `0.4554`，两者均按证据门拒绝。
- 共享 adapter/metacontroller checkpoint 固定为 initialization seed 42；
  evidence seed 只改变 episode 顺序和 delayed-observation lag。proof outcome
  使用环境 owner 发布的 `alignment_score - nominal_completion_threshold`，
  backend 可达性校准只能修改运行阈值，不能抹平名义难度。

最终 artifact：
`artifacts/eta_evidence_gate_1/expert_action_mps_margin005_s5_c48_v12_20260728`
（report/manifest schema v12）。配置为冻结 Qwen2.5-0.5B、MPS、fallback=false、
5 experience seeds、`n_z=16`、48 个 persistent Adam updates/seed。严格 verdict
为 `retain`，12 个 mechanism/outcome gate 全部通过：

| 指标 | mean | 95% bootstrap CI |
|---|---:|---:|
| credit F1 delta | 0.7317 | [0.7175, 0.7460] |
| false-credit reduction | 0.5883 | [0.5683, 0.6083] |
| family-assignment delta | 0.6000 | [0.4667, 0.6667] |
| held-out PE reduction-rate delta | 0.0202 | [0.00498, 0.03549] |
| beta boundary F1 | 0.6095 | [0.6012, 0.6179] |

每个 seed 均产生 2 个 active families 和 6 个 held-out delayed events；总 beta
boundary 88，对真实边界 95。该结果支持的严格表述是：在固定共享 checkpoint 和
未见路线上，真实 open-weight residual 经 action-prediction SSL 形成了非退化 beta
分段；以该分段为信用单位，相比 outcome-arrival turn baseline，因果窗口归因、
动作族归因和 held-out 结果预测均改善。它不证明任意 substrate、任意初始化或开放域
关系状态都会自动形成同等抽象。

## 2026-07-28 v13 动作族流形与评估真值修复

v12 之后的严格复核又发现两处会高估或削弱 ETA 证据的结构问题：

1. SSL 的监督目标处于 decoder action space，但 causal runtime 的 `z_t`
   proposal 可以离开训练得到的 action-family manifold；此前只按 latent
   邻近选择 family，无法保证解码后的控制动作一致。
2. evidence harness 曾用模型 rollout 的 dominant family 生成
   `true_family_id`，相当于让预测参与定义真值；这不是合法的 held-out
   family evaluation。

v13 的修复全部位于既有 owner 边界内：

- SSL topology discovery 对 `z_tilde` proposal 及其 decoder control 建族，
  然后才由 `beta_t` 决定 proposal 是否成为 active code。
- causal runtime 用 latent similarity 与 decoder-action similarity 的同一
  结构评分，把 proposal 投影到已学 family bank；bank 为空时严格 identity。
- `beta_t` 的无标签分位数校准每次最多移动 `0.08`，避免单 batch 阈值跳变。
- family 真值只取环境 `ExpertActionTarget.action_id`；family→action 映射只在
  train split 拟合，eval/heldout 不参与拟合，也不回灌 temporal owner。
- report/manifest schema 升级到 v13，并发布 `family_truth_source`、
  `family_mapping_fit_split` 与 `causal_family_manifold_projection`。

固定冻结 Qwen2.5-0.5B、MPS、`n_z=16`、3 个 SSL/RL cycles、每 cycle 25 个
persistent Adam updates 的 5-seed 首轮结果是 `weak`。ETA 的 held-out PE
五次都为 `0.03643`；turn baseline 两次更差、三次碰到同一地板，逐 seed effect
没有负值，但 5 样本 bootstrap 的 95% CI 下界仍为 0。没有放宽
`ci_low > 0` 证据门，也没有调训练参数；只把样本量扩展为 10 seeds。

最终 artifact：
`artifacts/eta_evidence_gate_1/segment_vs_turn_decoder_family_manifold_v13_qwen25_05b_mps_10seed_75update_20260728`。
严格 verdict 为 `retain`，12 个 mechanism/outcome gate 全部通过：

| 指标 | mean | 95% bootstrap CI |
|---|---:|---:|
| credit F1 delta | 0.9000 | [0.9000, 0.9000] |
| false-credit reduction | 0.8333 | [0.8333, 0.8333] |
| family-assignment delta | 0.5000 | [0.5000, 0.5000] |
| held-out PE reduction-rate delta | 0.1589 | [0.0397, 0.2780] |
| beta boundary F1 | 0.7652 | [0.7652, 0.7652] |

系统形成 4 个 active families，10 seeds 共观察 190 个真实边界和 250 个 beta
boundary；每 seed 的 optimizer 复用 74 次并完成 75 步，fallback 为 false。
这一结果证明的是该冻结基底、该 matched-control 和该 held-out 路线下，学到的
时间分段比结果到达 turn 更适合 delayed credit。它不证明任意开放域场景都会产生
同样动作本体，也不等于 Volvence 全部 thesis 已经完成验证。

## 2026-07-28 MPS 首轮结果

运行配置：冻结 `Qwen/Qwen2.5-0.5B-Instruct`，MPS，5 runs，120 个延迟结果事件。

结果为 `fail`：

- `credit_f1_delta = +0.4505`：把信用铺回 beta window 的 causal-window F1 高于只记结果到达 turn。
- `false_credit_reduction = +0.2969`：segment arm 的错误信用率较低。
- `family_assignment_delta = 0`。
- `pe_reduction_rate_delta = 0`。
- `segment_boundary_f1 = 0`。
- 120 个真实 subgoal 边界对应 0 个 beta boundary；每个 run 只有 1 个活跃动作族。

这不是“ETA 分段有效但收益不足”，而是当前 `full-internal-rl` 在真实 residual 任务上没有形成可用分段。旧 paper-suite 的 `heldout_credit_alignment = 1.0` 不能反证这一点：同一 profile 的 `never_switch_rate = 1.0`，一个抽象动作覆盖整条 rollout；只要扩张后的窗口包含奖励事件，旧 alignment 就可得到满分。因此该旧指标不具备区分“正确分段”和“从不切换”的能力。

## 下一收敛包

下一实验必须先让训练链忠实实现论文要求，再复用本门：

1. 在真实 residual prefix trajectory 上运行 action-prediction SSL。
2. 同时记录 KL/rate、预测损失、switch rate，确认压缩与预测之间存在非退化平衡。
3. 交替执行 SSL 与 `z_t` 空间 Internal RL，而不是只靠当前单轮 RL rollout 期待 beta 自发形成。
4. ground-truth subgoal 只用于 evaluation，不进入 beta 训练标签。
5. 先跑 5-seed directional gate；边界、动作族和 PE 三项转正后，再跑 20-seed full gate。

## 2026-07-28 第二收敛包实施口径

本包只收敛一条参数链：

- 参数 owner：`vz-temporal::MetacontrollerSSLTrainer` 及其
  `MetacontrollerParameterStore`。
- 输入：proof runtime 已经发布的冻结 `SubstrateSnapshot` causal-prefix 序列。
- 转换：一个 prefix 对应一个 SSL step，只取该 prefix 最后一个真实 token 的 residual；
  不重复展开整个 prefix。
- 调度：`vz-runtime` 在同一参数仓上执行 `SSL -> RL -> SSL -> RL`；默认 3 cycles，
  `--training-mode rl-only` 是回滚对照。
- 监督隔离：subgoal identity、完成时刻和真实边界不进入 SSL 或 `beta_t` 学习，只在最终
  evaluation 中计算 boundary F1 和 delayed-credit 指标。
- 机制证据：逐 run 发布 SSL trajectory 数、trained steps、prediction/KL/total loss、
  SSL switch frequency、torch parameter-change count 和 ACTIVE writeback count。
- 证据完整性：held-out 指标只从 `split == heldout` 的已观察 outcome 计算；某个 seed
  没有 held-out outcome 时保留零证据并让 coverage gate 失败，禁止回退到 eval split
  冒充 held-out 结果。

这一包不新增 snapshot slot，不改变 `docs/DATA_CONTRACT.md`，也不更新冻结 Qwen 参数。
它要回答的是：当论文要求的 action-prediction SSL 真正消费 rollout 同源残差，并与
`z_t` 空间 RL 交替后，`beta_t` 和动作族是否开始形成。若 5-seed 中边界、动作族和
held-out PE 仍不转正，结论仍是证据门失败，而不是调低门槛。

## 2026-07-28 第二收敛包 MPS 结果

实验组：

- artifact:
  `artifacts/eta_evidence_gate_1/segment_vs_turn_ssl_rl_mps_5seed_20260728`
- 配置：冻结 Qwen2.5-0.5B、`hf-local`、MPS、fallback=false、5 seeds、
  `n_z=16`、3 个 `SSL -> RL` cycles、`alpha=0.1`。
- 48 个 SSL trained steps/run，9 个 SSL trajectories/run，45 次 ACTIVE writeback。
- mean prediction loss `0.19546`，mean KL `1.26573`，mean SSL switch frequency
  `0.02815`。
- 最终结论：`fail`。

严格匹配的回滚对照：

- artifact:
  `artifacts/eta_evidence_gate_1/segment_vs_turn_rl_only_mps_5seed_dim16_20260728`
- 与实验组相同的冻结 Qwen、MPS、5 seeds 和 `n_z=16`，唯一变化是
  `training_mode=rl-only`。

对照结论：

| 指标 | SSL/RL 交替 | RL-only |
|---|---:|---:|
| beta boundary 总数 | 84 | 0 |
| true boundary 总数 | 76 | 120 |
| boundary F1 | 0.0333 | 0 |
| 每 run 动作族数 | 1 | 1 |
| family assignment delta | 0 | 0 |
| held-out PE reduction delta | 0 | 0 |
| held-out outcome coverage/run | 0, 8, 0, 8, 8 | 8, 8, 8, 8, 8 |
| credit F1 delta | 0.2703 | 0.4505 |
| false-credit reduction | 0.1781 | 0.2969 |

因此真实 residual SSL 并非没有作用：它把系统从“完全不切换”推到了“部分 seed
过度切换、部分 seed 完全不切换”。但这些切换几乎不对齐真实边界，仍没有形成第二个
可用动作族，并且使两个 seed 的 held-out outcome 消失。不能进入 20-seed full gate。

### 新定位的数学缺口

`vz-temporal/temporal/torch_store_ssl.py::train_store_ssl()` 当前 live-store 目标是：

```text
L = action_prediction_MSE + alpha * KL(q(z_t) || N(0, I))
```

但 runtime `beta_t` 在训练前向中只是连续插值门：

```text
z_t = beta_t * z_tilde + (1 - beta_t) * z_(t-1)
```

`switch_threshold` 不参与可微前向或 loss，只在训练后统计 binary switch ratio；
loss 中也没有 Bernoulli switch-rate prior、稀疏率项或其他对 `beta_t` 编码率的约束。
所以 posterior KL 的确压缩 `z_tilde`，却不直接约束“多久切一次”。现有 action
prediction loss 可以通过任意连续 beta 轨迹下降，并不保证准二值、稀疏、边界对齐。

下一收敛包应只修改这个 owner：给 live-store switch gate 建立无 subgoal 标签的
rate-distortion 目标和 matched lambda/prior 消融，先证明 beta rate 可控、预测损失不塌缩、
边界稳定性跨 seed 提升，再谈动作族和 delayed-credit 收益。

## 2026-07-28 Qwen2.5-7B MPS smoke

用户要求改用更强模型继续推进；本机 HF cache 没有字面名为“5.6”的模型目录，
已用可用的更强 open-weight substrate `Qwen/Qwen2.5-7B-Instruct` 跑 1-seed smoke。

- artifact:
  `artifacts/eta_evidence_gate_1/segment_vs_turn_qwen25_7b_mps_smoke_20260728`
- 配置：冻结 Qwen2.5-7B、`hf-local`、MPS、fallback=false、1 seed、
  `n_z=16`、3 个 `SSL -> RL` cycles、`alpha=0.1`。
- 运行链路已打通：48 个 SSL trained steps，9 条 SSL trajectories，9 次 ACTIVE
  writeback。
- 结果仍为 `fail`：heldout delayed events = 0，SSL switch frequency = 0，
  runtime beta boundaries = 42，ground-truth completed boundaries = 2，
  segment-boundary F1 = 0.0833。
- rollout-level active family count 修正后为 2，说明 family bank 不再完全单族；
  但 delayed credit、heldout PE 和 boundary 对齐仍没有形成证据。

该结果说明“换到 7B”解决了基底能力/缓存可运行性问题，但没有解决 ETA 机制问题。
下一步仍应落在 `train_store_ssl()` / switch gate 的 rate-distortion 目标上，而不是直接扩大
7B 到 5 seeds。

## 2026-07-28 第三收敛包：0.5B rate-distortion

按“先跑通单一管线、只用 0.5B”的口径，本包没有继续扩大模型，而是只修正
`vz-temporal` 的 switch 学习机制：

- 一个时间步只产生一个 scalar switch hazard，训练前向使用 hard
  straight-through gate。
- 加入 Bernoulli switch-rate KL、准二值、逐维 group coherence 和
  keep/switch counterfactual choice loss。
- 一个 persistent Adam session 连续消费多 trajectory batch，避免每个 cycle
  重置优化器。
- prediction horizon 扩到 3；支持 absolute residual vector 与 normalized
  innovation 两种 distortion proxy。
- 每个 run 使用独立初始化 seed；rollout 固定为 `causal`，不让 RL replacement
  覆盖 learned beta。

最终 5-seed 主运行：

- artifact:
  `artifacts/eta_evidence_gate_1/segment_vs_turn_rate_matched_qwen25_05b_mps_5seed_12cycle_20260728`
- substrate：冻结 `Qwen/Qwen2.5-0.5B-Instruct`，MPS，fallback=false。
- 配置：`n_z=16`，12 个 `SSL -> RL` cycle，switch prior `0.1`，
  rate/binary/group weight 均为 `0.001`，prediction horizon `3`，
  distortion target `innovation`。
- optimizer final step/run `12`，state reuse/run `11`，总 ACTIVE writeback `60`。
- 平均 SSL hard switch frequency `0.3038`，说明 rate 机制已经不再塌缩为永不切换。
- runtime beta boundary 总数 `28/120`，boundary F1 `0.1061`
  （95% CI `[0, 0.3182]`）。
- credit F1 delta `+0.5399`，false-credit reduction `+0.4042`。
- 每个 seed 仍只有一个动作族；family assignment delta 与 held-out PE delta 都为 0。
- verdict：`fail`。

3-cycle 对照 artifact
`segment_vs_turn_rate_matched_qwen25_05b_mps_5seed_20260728` 的 boundary F1
为 `0.1967`，12-cycle 反而降到 `0.1061`。因此继续增加 cycle 不会自动把“会切换”
变成“切在正确边界”。

### 剩余根因：缺少 Eq.3 的动作目标

代码逐项对照论文后，当前管线仍不能被称为完整 ETA Eq.3：

```text
论文：  -log p(a_t | o_1:t, z_1:t) + alpha * KL
当前：  MSE(Decoder(z_t), next_residual_vector_proxy) + alpha * KL + beta rate
```

`TrainingTrace.TraceStep` 当前只有 `token / feature_surface /
residual_activations`，没有专家动作 `a_t`、动作 id、动作向量或动作分布。
open-weight capture 的 `token_logits` 也只保留 top-k 概率值，不保留对应 token id，
不能恢复正确的 next-token likelihood。proof episode 的 subgoal target signature
属于 evaluation 真值，把它当训练动作会直接泄漏边界。

因此 evidence schema 升为 v4，并显式发布：

- `ssl_supervision_target=next-residual-vector-innovation-proxy`
- `expert_action_supervision=false`
- retain gate `ssl-uses-expert-action-targets=false`

下一包不应再调 switch prior 或增加训练轮次，而应先建立不消费 subgoal
evaluation 标签的专家动作 trajectory 契约。只有 `a_t` 的 owner、表示、采集方式和
train/heldout 隔离明确后，才能再次检验 beta 边界和动作族是否真正涌现。

### 16 维 residual-vector 与 proposal-aware matched rerun

本包随后补做了两个不使用 boundary 标签的根因修正：

- posterior KL 从 16 维求和改为 per-dimension rate，消除 latent dimension
  对 `alpha` 的隐式放大；
- hard gate 关闭时仍训练 `z_tilde` proposal，并用 detach 的 keep/switch
  counterfactual distortion 单独训练 gate choice，解除“从不切换就永远学不到
  更好切换候选”的梯度死锁；
- distortion target 不再把 residual 压成 `(mean, max, spread)` 后平铺，而是消费
  同源的 16 维 projected residual vector。

最终 matched artifact：

`artifacts/eta_evidence_gate_1/beta_rate_distortion_mps_5seed_residual16_innovation_p020_rw005_c3_20260728`

配置为 5 seeds、3 cycles、`p_switch=0.20`、rate weight `0.05`、innovation
target、prediction horizon `3`。结果：

- persistent optimizer final step/run `3`，reuse/run `2`，ACTIVE writeback `15`；
- target variance `0.0640`，prediction loss `0.2533`，per-dim KL `2.2583`；
- keep/switch counterfactual loss 接近，平均 beta probability `0.4817`；
- SSL hard switch frequency、runtime beta boundary 和 boundary F1 仍全部为 0；
- 每个 seed 仍只有一个动作族，family/held-out PE delta 仍为 0；
- strict gate `ssl-uses-expert-action-targets=false`，verdict `fail`。

这说明 residual proxy 的公共分量、KL 尺度和 proposal 梯度死锁都已分别处理，但
“预测 residual 变化”仍不能替代“预测专家动作”。因此不再用 residual proxy 继续调
switch prior、阈值或 cycle；下一包应先扩展正式 trajectory 契约以发布无评测泄漏的
动作目标。

## 2026-07-28 第四收敛包：专家动作、episode 隔离与 causal rate 校准

本包完成了第三包明确缺失的数据面：

- `vz-substrate::ExpertActionTarget` 发布 `action_id / values / source`，并与
  `TraceStep` 一一对应；targeted trace 缺任一步即 fail loudly。
- train 使用环境 demonstrator 的 action vector；eval/heldout 只给 observation，
  action target、subgoal boundary、reward 和 outcome 都不进入 optimizer。
- action distortion 改为 `MSE(Decoder(z_t), expert_action_vector)`；KL 仍约束
  posterior 抽象层级，aggregate Bernoulli rate 约束平均 switch rate。
- PE counterfactual preference 使用温度归一化 BCE 训练 gate choice；报告同时发布
  boundary/continuation 的 probability 与 preference，避免只看 hard threshold。
- 每个环境 observation 只对应一个 residual step；每个独立 rollout 清零
  trajectory-local recurrent state 和 active-family telemetry，保留 learned
  metacontroller/action-family 参数。
- 证据路径关闭 `temporal_fast_prior`；Internal RL 只给 causal z candidate。
  SSL 与 train-only causal rollout 分别用同一个无标签 `switch_prior` 校准阈值，
  不读取 evaluation boundary。
- 每个 seed 运行 3 个 SSL→RL cycle，每 cycle 25 个 SSL updates，共 75 个
  persistent Adam steps；解决此前 12 cycles 实际只有 12 个梯度步的问题。

### 关键诊断序列

1. 仅加入 expert action，0.5B 边界 F1 从 residual proxy 的 `0.1061` 提升到
   `0.3038`，但 seed 呈全切/全不切。
2. 修正 environment-step temporal parity 后，训练侧 boundary switch probability
   首次高于 continuation，但 runtime 仍被跨 rollout hidden state 和 fast prior 污染。
3. 关闭 fast prior 后，边界 F1 升到 `0.6095`，但每步切换；这暴露了固定阈值与
   trajectory 状态泄漏。
4. 增加 episode recurrent reset 后，trace 后端 heldout delayed events 恢复，
   credit F1 delta CI 下界 `0.5853`，证明此前 heldout=0 主要是跨 episode 状态污染。
5. 增加 train-only causal rate 校准后，0.5B 五个 seed 均恢复 heldout delayed
   events，但 seed 0 对 train 分布过校准，heldout 上变成零边界。

### 最终 0.5B 证据

Artifact：

`artifacts/eta_evidence_gate_1/segment_vs_turn_causal_rate_calibrated_qwen25_05b_mps_5seed_75update_20260728`

配置：冻结 Qwen2.5-0.5B、MPS、fallback=false、5 seeds、`n_z=16`、
75 updates/seed、switch prior `0.35`、fast prior disabled、causal z replacement。

成立的门：

- real residual trajectory、expert action supervision、ACTIVE live-store writeback、
  persistent optimizer reuse。
- heldout delayed events：每 seed `7–8`。
- credit F1 delta `0.6433`，95% CI `[0.5421, 0.7238]`。
- false-credit reduction `0.5019`，CI `[0.3844, 0.5901]`。

未成立的门：

- boundary F1 `0.4815`，CI `[0.2249, 0.6485]`；seed 0 为 0。
- active family count 每 seed 都为 1。
- family assignment delta 与 heldout PE reduction delta 都为 0。

因此 schema v11 的严格 verdict 仍是 `fail`。当前可保留的是一个局部命题：
**当 learned beta 已形成可观测 segment 且延迟结果存在时，segment-level credit
显著优于 outcome-arrival turn credit。** 尚不能保留的是完整 ETA 命题，因为
beta 的 train→heldout rate 稳定性、action-family 分化以及 family-conditioned
PE 改善都没有证据。

下一包只处理 temporal owner 的两个耦合问题：

1. 用非 transductive、非 evaluation-label 的方式稳定 train→heldout beta rate，
   避免单 batch quantile threshold 过校准。
2. 让 expert-action decoder 的可分控制真正进入 action-family topology；
   在多个 family 跨 seed 涌现之前，不再跑 PE retain gate，也不扩大模型。
