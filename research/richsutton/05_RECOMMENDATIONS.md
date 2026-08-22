# 对 Volvence 的借鉴与实施建议

## 1. 决策摘要

### 现在就吸收

1. 把 plasticity 与 forgetting 分开测量。
2. 把网络内部可塑性变成 owner 发布的一等 frozen readout。
3. 对所有长期在线小网络要求“新知识吸收不随寿命下降”的长流证据。
4. 把 generate-and-test 解释为有界候选生命周期，而不是一次初始化或无限扩容。
5. 抽象同时接受 prediction 和 planning usefulness 检验。
6. 合成数据只负责机制验证、候选和预演；真实 outcome 负责结算 live learning。
7. 在固定 per-step compute / memory 下比较方法。

### 先实验、后决定

1. CMS hidden-unit CBP。
2. per-feature / per-parameter step-size optimization。
3. K=1 与 bounded replay 的取舍。
4. OaK 风格 feature→subtask→option→model candidate lifecycle。
5. 对 frozen substrate 的 rare-heavy representation adaptation。

### 当前不要做

1. 在线端到端更新整个基础 LLM。
2. 因访谈一句话停掉所有合成数据和模拟器。
3. 在 semantic owner 上随机删除/重置状态。
4. 把 evaluation、judge、engagement 或 continuity score 变成 reward。
5. 把现有 band `step_scale` 命名为 IDBD，或把 `z_t / beta_t` 命名为 OaK。
6. 直接接受 no-replay、20W 或“灾难性遗忘完全可治”为技术门槛。

## 2. 推荐的总体推进顺序

```text
P0 可塑性仪表
    ↓ 先确认问题是否存在、出现在哪个 band
P1 隔离复现 Nature / CBP
    ↓ 机制在相同预算下可重现
P2 CMS-only CBP SHADOW ──┐
P3 step-size SHADOW ─────┤ matched factorial
                         ↓
P4 OaK-inspired abstraction utility
                         ↓
P5 更高密度的真实 consequence settlement
                         ↓
P6 resource / product formal gate
```

P0–P3 只解决 `vz-memory` 的一个 owner 和一个底层问题，不与 temporal abstraction 重构放进同一收敛包。P4 是独立的 `vz-temporal` 研究包。P5 主要丰富现有 environment / PE lineage，不发明第二 reward owner。

## 3. 统一评价面板

后续所有持续学习实验都应至少报告五类指标，而不是一个综合分：

| 面板 | 问题 | 建议指标 | 是否可进入在线学习 |
|---|---|---|---|
| Plasticity | 活得越久是否越难学新模式 | 新 regime loss AUC、达到阈值所需 updates、相对 reset oracle gap、按生命期的 slope | 只有来自真实 PE/credit 的细粒度信号可学；汇总指标只评估 |
| Stability | 学新内容是否破坏旧内容 | reappearance performance、retention AUC、稀有高信用事实回访 | 可通过 owner 的合法 replay/credit；不可由评测脚本回灌 |
| Transfer | 旧知识是否帮助未来 | 相对 cold start 的 sample/step saving、held-out family transfer | 只作 formal evidence，除非已注册为 PE 下游 credit |
| Coherence / safety | 主体、关系、承诺和边界是否一致 | invariant violations、consent/boundary violations、semantic conflict | 硬约束/验证锚，不优化成单标量 reward |
| Resource | 能否在 Big World 约束下运行 | time/update、memory/agent、extra forward、settled-outcome latency、可选 joule | 作为 promotion constraint，不混入语义 reward |

### 一个推荐的可塑性归一方式

对第 `t` 个新 regime，令：

- `L_cont(t, k)`：长期 learner 在进入新 regime 后第 `k` 次更新的 loss；
- `L_reset(t, k)`：从相同初始化分布重训的 oracle；
- `L_frozen(t)`：不更新的 baseline。

报告：

```text
normalized_absorption(t)
  = AUC[L_frozen - L_cont] / max(AUC[L_frozen - L_reset], ε)
```

同时报告原始曲线，防止归一分母掩盖任务变难。若 `normalized_absorption` 随生命期持续下降，而当期数据和算力相同，即出现直接的 plasticity-loss 信号。

这只是离线 evidence metric，不得进入运行时 reward。

## 4. P0：CMS Plasticity Observatory

### 目标

先回答“Volvence 的三个 CMS band 是否正在越学越僵”，不改变任何学习行为。

### 唯一 owner 与契约

- owner：`vz-memory.CMSMemoryCore`。
- 推荐优先扩充现有 `CMSState` / `CMSBandState` 的 additive frozen readout，避免创建平行 slot。
- 如果必须新增跨模块公共 slot，实施前先在 `docs/DATA_CONTRACT.md` 注册 `owner / value_type / dependencies / wiring_level`。
- consumer 只能读汇总 snapshot，不得访问 `_w1`、`_w2`、momentum 或激活 buffer。

### 每个 band 的最小读数

#### 参数健康

- `mean_abs_weight`、`p95_abs_weight`、Frobenius norm；
- `update_norm / max(weight_norm, ε)`；
- gradient / momentum norm；
- residual branch contribution 相对 identity path 的比率。

#### 激活健康

- preactivation mean/std；
- `|tanh(h)| > 0.95` 和 `> 0.99` 的饱和比例；
- bounded window 内 activation variance；
- near-constant hidden-unit 比例；
- hidden contribution utility 的 p10/p50/p90。

#### 表征健康

- 对 owner 内 bounded activation sample 计算 stable rank；
- effective rank（奇异值熵）作为补充；
- hidden pairwise correlation 或 redundancy 的低成本 proxy。

#### 学习健康

- effective learning rate / band `step_scale`；
- write gate、momentum gate、reset mix 的分布和饱和比例；
- latest old/new error decomposition；
- 自生命周期起的 plasticity trend，而不只 64-observation 均值。

### 工程约束

- 观测窗口有硬上限；SVD 可按低频 cadence 做，不阻塞 online-fast。
- snapshot 只包含标量/tuple，不泄露 mutable tensor。
- `DISABLED` 必须不分配 activation history，并保持输出字节级一致。
- `SHADOW` 只测量；P0 不应有 `ACTIVE` 行为含义。
- checkpoint 包含必要的 window accumulator，使跨 session 指标连续；不需要恢复原始样本。

### 验证

- 构造权重增长、tanh 饱和、rank collapse、constant-unit 四个 deterministic fixtures，证明仪表能检出。
- healthy random feature fixture 不应误报。
- snapshot frozen / round-trip / no mutable reference tests。
- 运行开销和 memory 上限测试。

### 完成条件

P0 的完成不是“指标好看”，而是：

1. 能在已知故障 fixture 上检出；
2. 能在相同 checkpoint 恢复一致趋势；
3. production 默认行为完全不变；
4. 真实/回放长流得到 band-level baseline。

## 5. P1：隔离复现 Nature，并适配 tanh CMS

### 目标

在触碰 runtime 前，确认：

1. 本地长期流能复现 ordinary BP 的 plasticity loss；
2. CBP / L2 / shrink-and-perturb 的效果与 Nature 方向一致；
3. tanh + residual CMS 结构是否需要不同 utility。

### 位置和权限

- 放在 `research/richsutton/experiments/` 或独立 research harness。
- 不导入 lifeform 产品数据，不写 production memory。
- 使用正式 `CMSBandMLP` 独立调用方式，避免自建看似相同的简化 learner。
- synthetic stream 在这里合法，因为目的就是控制机制变量；报告必须标记为 mechanism evidence，不能升级为产品收益。

### 推荐任务

1. **Slow drift regression**：目标子空间持续缓慢旋转，测 tracking。
2. **Abrupt regimes with return**：新旧 regime 循环出现，分开测 absorption 与 retention。
3. **Sparse rare regime**：低频但高重要性模式，检验 current-utility 是否误删。
4. **Structured transfer**：新 regime 与旧 regime 共享潜变量，测 forward transfer。
5. **Captured settled trace**：只使用已经 canonicalized、去敏且有 lineage 的真实 trace，作为最后一层 external validity；不得用 evaluation label 当 target。

### Matched arms

- 当前 `CMSBandMLP` + 当前 updater；
- 固定 SGD/momentum baseline；
- L2；
- shrink-and-perturb；
- random mature-unit replacement；
- contribution-utility CBP；
- saturation-aware / long-window utility CBP 候选。

所有 arms 共享：参数数目、每步更新预算、replay budget、输入顺序、seed family 和 checkpoint cadence。CBP 的 utility/age 额外内存必须计入资源面板。

### 预注册建议

- 先用 pilot 确认任务能产生可塑性下降，只用于定标，不参与 final seed。
- final 至少 3 seeds；足够长到 baseline 指标出现稳定斜率，优先数千 regime 或数万更新，而不是预先假定 510 turns 足够。
- 预先冻结 primary plasticity metric、retention non-inferiority、resource cap 和 kill rule。
- 同时保留 no-replay / bounded replay 两组，避免 replay 掩盖问题。
- 如果 baseline 根本没有 plasticity loss，就不能用“CBP 不伤害”作为采用理由。

### 退出条件

只有以下均成立才进入 P2：

- 至少一个任务上稳定复现 plasticity loss；
- CBP 类方法在 held-out seeds 上改善 primary plasticity metric；
- retention、coherence proxy 和 resource 没有越过预注册 kill line；
- utility 选择优于或至少解释性地不同于 random replacement；
- output-preserving reset 的瞬时误差在数值容限内。

## 6. P2：CMS-only Continual Backprop SHADOW

### 收敛边界

只改一个 owner、一个 band、一个 feature-renewal 机制。不要同时改 temporal controller、foundation substrate 或 semantic owner。

第一目标建议选 `online-fast`：它对新分布最敏感、持久语义风险最低、更新频率最高。不要先从 `background-slow` 开始，因为低近期 utility 可能删除罕见但长期重要的关系模式。

### CMS 结构中的精确替换

当前结构：

```text
hidden = tanh(W2 @ x)
output = x + W1 @ hidden
```

对 hidden unit `j`：

1. 重新采样 `W2[j, :]`；
2. 把 `W1[:, j]` 清零；
3. 清零 `W2_momentum[j, :]` 和 `W1_momentum[:, j]`；
4. 清零该单元 utility 和 age；
5. RNG state、replacement accumulator、age、utility 进入 owner checkpoint；
6. 不修改 `_state`、CMS cards、semantic snapshots 或其他 band。

由于 outgoing column 为零，替换瞬间 residual 分支对输出的贡献为零；identity path 仍在。这一不变量应有逐元素测试。

### Utility 设计实验，不要先锁死

至少比较：

- Nature contribution utility；
- random mature-unit baseline；
- contribution + tanh saturation penalty；
- short + long horizon utility；
- PE/credit-protected utility：曾在高信用结果中关键的单元有有限保护期。

不能简单把“激活小”当“无用”：稀疏 feature 可能只在重要关系事件出现。长期 utility 也不能无限保护，否则网络重新僵化。

### SHADOW 双跑

每次 eligible update 从相同 pre-state 分叉：

- control：现有 updater；
- candidate：现有 updater + CBP。

两臂消费相同 target、PE lineage、replay window 和 compute budget。SHADOW candidate 不写回 active CMS，只发布聚合 evidence。不要让 consumer 读取 candidate output。

### Public readout

可在现有 CMS evidence snapshot 中追加：

- wiring level、replacement rate、maturity threshold；
- eligible / replacement count；
- utility quantiles、age quantiles；
- output-preservation max delta；
- saturation/rank/weight trend；
- control-vs-candidate absorption/retention/resource；
- checkpoint/rollback attestation。

### 晋升门

从 SHADOW 到有限 ACTIVE canary，至少要求：

1. 真实 settled trace 与 synthetic mechanism suite 方向一致；
2. plasticity primary 明显改善，不只内部 rank 变好；
3. retention / rare-regime / semantic coherence 不劣；
4. 多 seed，预注册，不在 locked result 上调参重跑；
5. output preservation、finite values、latency、memory 全过；
6. full checkpoint 恢复和单字段 rollback 已演练；
7. canary 只启用一个 band，ModificationGate 不被 bypass。

### Kill 条件

- 旧高信用状态回访显著恶化；
- utility 反复替换同一单元且无新贡献；
- step scale / rank / saturation 更差；
- reset 引入非有限值或 checkpoint 不确定性；
- 额外开销超过预注册预算；
- 指标改善只出现在 synthetic，而真实 outcome 不支持。

## 7. P3：Step-size Optimization SHADOW

### 目标

在同一 CMS band 内，让稳定 feature 慢学、快变 feature 快学；验证其是否比现有 band-level `step_scale` 更好，而不是替换整个 learned update 架构。

### 推荐递进

1. **per-feature group step-size**：`W2[j,:] + W1[:,j]` 共用一个 log step；
2. **matrix block step-size**：W1/W2 不同；
3. **per-parameter step-size**：只有前两者有清晰收益时再承担额外状态。

直接从 per-parameter 开始会近似把 optimizer state 再翻倍，并增加数值/审计复杂度。

### 算法约束

- 在 log domain 更新，显式 `min/max` cap；
- meta-update 必须有归一化或 Autostep 类稳定机制，以应对论文展示的尺度敏感性；
- CBP 替换单元时同步重置其 step-size/meta-trace；
- `DISABLED` 精确保留现有更新；
- `SHADOW` 不写 active weights；
- 不允许 optimizer 自己把 evaluation / judge 作为 lifetime objective。

### 信号来源

合理来源是 owner 内由真实 target 前后 PE/credit 得出的 update improvement 和 stability trace。现有 updater 的内部 heuristic 可作为 control，但新的 meta-objective 必须记录：

- 对应 prediction / outcome / credit lineage；
- 哪个参数组收到多大 meta-gradient；
- 是否在 outcome 未结算时保持 no-op；
- delayed credit 如何对应当时 checkpoint。

### Matched factorial

建议最终做 `CBP × step-size` 2×2：

| | 固定/现有步长 | optimized step-size |
|---|---|---|
| 无 CBP | baseline | step-size only |
| 有 CBP | CBP only | combined |

这样能判断两者是互补、冗余还是相互不稳。不能只比较 combined 对最弱 baseline。

### 观测与 kill

发布 step-size p01/p10/p50/p90/p99、cap-hit fraction、near-zero fraction、meta-gradient norm、不同 regime 的响应时间。

以下任一成立即 kill：

- 大量 step-size 长期钉死在上下界；
- meta-trace 非有限或对 target scale 极端敏感；
- plasticity 提升来自牺牲 retention；
- 参数级方法不优于更便宜的 group 级方法；
- 与 CBP 联合后 replacement unit 不能重新成熟。

## 8. P4：OaK-inspired Abstraction Utility（独立研究包）

### 为什么不能直接“实现 OaK”

- OaK 没有公开论文/代码；
- Volvence 当前 ETA operationalization 有正式 `kill-eta`；
- feature、subtask、option、model 跨多个 owner，贸然合并会违反 SSOT；
- 关系域的价值不能压成单 scalar reward。

所以 P4 只验证一个最小命题：**一个候选时间抽象若同时提升未来预测和固定预算规划，是否能经 PE/credit/gate 稳定晋升？**

### 唯一 owner

- candidate lifecycle owner：`vz-temporal`。
- `plan_intent`、`goal_value`、`relationship_state` 等仍由各自 semantic owner 发布，只作为 frozen input，不变成 temporal 内部字段的第二份复制。
- 若增加公共候选/utility snapshot，必须先注册 DATA_CONTRACT；runtime 只经 snapshot 传播。

### 最小候选形态

候选不必一开始就是完整 option keyboard。先限定为：

- 一个从正式 residual / temporal state 产生的低维 predictive feature；
- 一个 typed termination / boundary model；
- 一个可由现有 action space 调用的 abstract action proposal；
- 一个预测该 proposal 在若干步后 outcome / state 的 bounded model。

候选不得由关键词、正则或 prompt label 决定。LLM reflection 可以 background-slow 提出可读描述，但没有预测/规划证据时不能晋升。

### 双重 utility

#### A. Self-verification / prediction utility

- 相对 matched baseline 降低 held-out、已结算 outcome 的 PE；
- 在时间边界、跨 session 恢复后仍校准；
- 不依赖未来泄漏或 evaluator artifact。

#### B. Planning utility

- 在固定 planning compute 下，候选是否改善 action ranking；
- counterfactual 的预测最终由真实 outcome 结算；
- 只改善模型自评但不改善结果的候选不晋升。

#### C. Constraints

- consent / boundary 零违规；
- World / Self 语义不混轨；
- 延迟、内存和候选池有上限；
- 与已有 active abstraction 冗余时要支付容量成本。

### 生命周期

```text
PROPOSED
  → SHADOW_OBSERVED
  → MATURED（覆盖足够多 settled outcomes）
  → GATE_ELIGIBLE
  → ACTIVE_CANARY
  → ACTIVE
  → RETIRING
  → RETIRED / ROLLED_BACK
```

每次状态转移都带 prediction IDs、outcome IDs、credit IDs、dependency IDs 和 wiring。下游 option/model 依赖上游 feature；上游退休时必须显式级联或阻止，不能留下悬空 consumer。

### Gate

- prediction improvement 与 planning improvement 必须分别达标；
- 至少一个 held-out environment family；
- typed outcome coverage 足够，missing outcome 不当失败；
- 对 noop/shuffle/permuted candidate 有可分性；
- canary 可单字段回滚；
- `kill-eta` 只可由新预注册证据程序更新，不能被 P4 名称绕过。

## 9. P5：Grounded Consequence Lane

### 目标

增加“动作 → 可结算结果”的密度和质量。没有这一层，CBP、IDBD 和 OaK 只会更高效地学习代理偏差。

### 优先扩充的 outcome

| 类别 | 例子 | 结算注意 |
|---|---|---|
| 工具/执行结果 | 文件是否生成、API 是否成功、测试是否通过 | 以外部返回为事实，不以模型解释为事实 |
| 用户明确纠正 | 偏好、身份事实、边界、理解错误 | 必须有同意/作用域；撤回可追踪 |
| 承诺结果 | 约定动作是否在时限内完成 | 延迟结算；不能无结果即判失败 |
| 计划里程碑 | 用户/环境确认阶段完成 | typed milestone，不用关键词猜测 |
| 关系边界 | 明确同意、拒绝、撤回 | hard constraint；不压成 engagement reward |
| 预测校准 | 行动前预测后出现可观察结果 | prediction ID 与 outcome ID 一一 lineage |

### 禁止的伪后果

- 回复更长或对话更久；
- 用户使用积极词；
- evaluator 认为更有共情；
- LLM 对自己答案给高分；
- synthetic persona 按脚本继续对话；
- 没有后续反馈就默认成功或失败。

### 缺失与延迟

- outcome 可为 `pending / settled / expired-unobserved`，后两者不能混同；
- delayed credit 使用当时的 prediction/action checkpoint；
- 同一结果可对多个预测产生不同类型 PE，但 credit owner 保持唯一；
- relationship / boundary 解释由相应 semantic owner 发布，不由 runtime 直接猜。

### Promotion effect

P5 不直接“提高 reward”。它提高 PE 的 grounding ratio、settlement coverage 和 lineage quality。只有这些量稳定后，P2–P4 的真实 trace 结果才有资格用于晋升。

## 10. P6：Big-World 资源门与长期 formal

### 固定预算比较

每个 arm 除模型质量外，还应报告：

- 每 environment step / settled outcome 的训练 FLOP 或近似操作数；
- p50/p95 update latency；
- owner state + replay + optimizer + utility 的 bytes；
- extra forward 数；
- checkpoint 大小与恢复时间；
- 可获得时的能源估计，但不把 20W 当近期门。

### 长期流要求

正式 gate 要覆盖：

- abrupt + gradual drift；
- old regime return；
- rare/high-credit event；
- cross-session hydration；
- candidate replacement 后的恢复；
- 多 seed 和一个 held-out family；
- 不同生命期阶段的 plasticity slope。

### 三层判词

1. **CODE**：契约、no-op、rollback、determinism、资源上限通过；
2. **MECHANISM**：可塑性/保持/迁移在 controlled stream 上达到预注册标准；
3. **PRODUCT**：真实 consequence 下的结果改善且无边界/关系伤害。

CODE 通过不能自动升级 MECHANISM，MECHANISM 通过也不能自动升级 PRODUCT。

## 11. 四能力轴自检

| 包 | Appendable | Readable | Learnable | Steerable / 下一拍闭环 |
|---|---|---|---|---|
| P0 仪表 | window/checkpoint 随 owner 恢复 | frozen plasticity readout | 不学习，只测量 | strict no-op，不改变行为 |
| P1 复现 | isolated trace state | 全部指标可审计 | controlled target；不回灌产品 | 无 runtime actuator |
| P2 CBP | utility/age/RNG 随 CMS 恢复 | replacement/readout 由 CMS 发布 | 只消费 owner 合法 target/PE/credit | 先 SHADOW；ACTIVE 时改变下一拍 CMS state |
| P3 step-size | meta-trace/step state 可恢复 | step quantile/cap state 发布 | lifetime objective 追溯 PE/credit | 有界更新，单字段关闭 |
| P4 abstraction | candidate lifecycle 跨 session | dependency/utility frozen snapshot | prediction + planning 经真实 outcome/credit | gate 后才影响 action；可 canary/rollback |
| P5 consequence | outcome/pending ledger 可追加 | typed settlement snapshot | PE 的唯一上游事实 | 结算改变后续 credit 与候选状态 |
| P6 formal | 长期 trace 固定 | evidence artifact 可审计 | 评估与学习隔离 | promotion/rollback 明确 |

## 12. 收敛包建议

真正编码时，依仓库约束拆成以下小包，不要一次替换全链：

### 包 A：Plasticity snapshot

- 只新增 frozen schema、owner 采样和 contract tests；
- 不加 CBP；
- spec 同步；
- 默认 DISABLED / observe-only。

### 包 B：Research harness

- 只建立长期 benchmark 和 evidence artifact；
- 不改 runtime wiring；
- 冻结任务、metric、seed 和 resource budget。

### 包 C：CBP owner implementation

- 只实现 CMS internal utility/age/reset/checkpoint；
- SHADOW dual-run；
- 一个 band；
- 不晋升 ACTIVE。

### 包 D：CBP gate/canary

- 在 C 的 locked evidence 通过后单独接 promotion readout；
- canary 与单字段 rollback；
- 不同时加入 IDBD。

### 包 E：Step-size optimizer

- 从 per-feature group 开始；
- 与现有 updater matched；
- 独立 SHADOW gate。

### 包 F：Abstraction candidate contract

- 先冻结 owner、snapshot、dependency 与 lifecycle；
- 再接一个 producer 和一个主要 consumer；
- planning utility 与 candidate retirement 后置。

每包尽量控制在 3–8 个关键文件；共享 schema 或 `docs/DATA_CONTRACT.md` 变更单独后置，保持单一写入者和可回滚路径。

## 13. 建议的研究问题清单

### 可塑性

- CMS 的 plasticity loss 出现在 update 次数、wall turns 还是 regime 数的哪个尺度？
- tanh saturation、weight growth、rank collapse 哪个最先出现？
- 三个 band 是否需要不同 replacement rate / maturity？
- residual identity path 会掩盖 hidden branch 已僵化吗？

### 稳定性

- contribution utility 是否误删稀有关系 feature？
- credit-protection 多长才不至于永久冻结？
- bounded replay 与 CBP 是互补还是相互冗余？
- semantic snapshots 能否在内部 feature basis 改变后保持解释一致？

### 步长

- per-feature group 是否已捕获大部分 per-weight 收益？
- meta-step 能否跨目标尺度自动归一？
- CBP 新单元的 step-size 应重置、继承邻域还是 meta-initialize？
- 现有 updater 哪些 gate 与 IDBD 重复，哪些提供额外安全？

### 抽象与规划

- predictive feature 是否真的被 planner 使用，还是只让 readout 更好？
- planning usefulness 如何在不使用 evaluator reward 的情况下结算？
- feature 退休时下游依赖如何级联且不破坏 owner 边界？
- 真实关系事件太稀疏时，maturity 如何定义？

### 资源

- 在相同每拍预算下，K=1 大网络是否优于 replay + 小网络？
- utility/optimizer state 的内存是否值得？
- background-slow SVD / planning 是否影响 foreground latency？
- 每个真实 settled outcome 的总计算成本是多少？

## 14. 最终路线建议

Sutton 的工作不要求 Volvence 放弃已有架构；它要求我们把一个此前容易被忽略的问题升格：**持续写入并不自动意味着持续可学。**

因此最有价值、风险最低的路线是：

1. 先在 `vz-memory` 建立可塑性观测；
2. 用隔离、长时、固定预算实验确认问题与候选解；
3. 只对 CMS hidden features 引入 CBP / step-size，保持 semantic state 和基础 LLM 不动；
4. 用 Volvence 已有 PE、credit、snapshot、WiringLevel 和 ModificationGate 补上 Nature / Oak 未处理的稳定、安全和回滚；
5. 等底层可塑性成立后，再重新挑战自主时间抽象和 planning utility；
6. 始终让真实 consequence 负责最终结算，让 synthetic 负责更安全地提出和淘汰假设。

这条路线既吸收了 Oak 的核心野心，也保留了 Volvence 最有价值的架构资产：有界、契约驱动、可审计、关系安全的持续适应。
