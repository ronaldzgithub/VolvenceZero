# 数字蚂蚁 Ecology 恢复与正式验证计划（P0 → P1 → P2）

> 状态：执行计划，尚未开始实施  
> 基线：`ecology_probe_neutral.v2`，结论 `BLOCK`  
> 原则：P0、P1、P2 严格串行；前一阶段未达到退出门槛，不得消耗下一阶段预算，也不得放宽已冻结门槛。

## 1. 目标与当前基线

本计划的最终目标不是让训练日志出现更多 pickup，而是建立一条可审计的因果证据链：

```text
局部 ecology 感知
  → learned latent code
  → 方向正确的 motor action
  → temporal switch / segment credit
  → medium/far 单能力闭环
  → 中性障碍与热源下的组合泛化
  → matched controls 上的稳定增益
```

当前基线已经证明 checkpoint、policy writeback、no-optimize 隔离、archive roundtrip 和 replay
lineage 基本可用，但存在三处硬断点：

1. 训练后 heat 左右输入仍改变 latent code，却不再改变 turn；
2. held-out 的 `switch_count` 全为 0，segment 主要在 24-step 上限被动闭合；
3. near/forced-start 事件可以掩盖 medium/far 失败，导致训练 mastery 高估能力。

本计划依次解决机制正确性、课程正确性和正式统计证据，不能跨级。

## 2. 全局执行规则

### 2.1 冻结规则

- 每次运行必须记录 git SHA、dirty 状态、配置 digest、依赖版本、设备、训练 seed、布局 seed 和模型 fingerprint。
- 每个产物使用新文件名，不覆盖已有 `BLOCK` artifact。
- P0/P1 的阈值在首次读取新结果前写入 schema/test；观察结果后只能修实现，不能为通过而改阈值。
- training、validation、held-out seed 命名空间保持不相交。
- held-out 只运行一次正式判定；调试和调参只看 training/validation。
- 任何异常退出都保留 partial log，并标记 `incomplete`，不得与完整结果合并。

### 2.2 阶段状态

每一阶段只能处于以下状态之一：

- `NOT_STARTED`
- `RUNNING`
- `BLOCKED`
- `PASS`

只有 `PASS` 可以解锁下一阶段。`BLOCKED` 必须附带失败 gate、最小复现和下一步修复假设。

### 2.3 统一产物

每个阶段至少输出：

- canonical JSON report；
- manifest（输入、hash、provenance、配置）；
- 人类可读 Markdown summary；
- 原始逐步或逐 episode trace；
- 测试命令与结果；
- 如果生成 checkpoint，则附 `.vzac`，但 P0/P1 的 checkpoint 一律不可加载为 demo/promotion。

建议目录：

```text
research/ant/results/ecology_recovery/
  p0/<run-id>/
  p1/<run-id>/
  p2/<run-id>/
```

## 3. P0：机制诊断与修复

### 3.1 P0 目标

在投入新课程训练前，逐拍证明三件事：

1. food/heat 的左右差异能够稳定地从 sensor 传到 latent，再传到 turn；
2. temporal controller 在可控状态变化时能够产生真实 beta switch；
3. `learning_enabled=False` 时所有学习 owner 严格冻结。

P0 不评估 ecology 综合任务成绩，也不追求 promotion。

### 3.2 P0-A：Sensor → latent → action 链路审计

#### 实现任务

- 扩展 ecology paired probe，使其可以对 colony 中每个 body 的 checkpoint 独立运行。
- 在以下 checkpoint 时点自动运行 food/obstacle/heat 左右探针：
  - shared initial；
  - 每个 primary episode 后；
  - 每个 interleaved episode 后；
  - 每个训练阶段结束后；
  - 最终 checkpoint。
- 每个探针记录：
  - 原始 19 维 sense；
  - encoder/posterior hidden state 摘要；
  - `z_candidate`、track modulation、causal action-head residual、最终 `z_t`；
  - turn/step command；
  - causal action-head 参数 fingerprint、范数和相对前一 checkpoint 的 change norm；
  - motor readout 对左右 latent delta 的投影；
  - pure/runtime/torch 路径的一致性。
- 增加一个只读的局部敏感度诊断：对 paired latent delta 做有限差分，判断其是否落在 motor readout 的近零空间。

#### 最小实验矩阵

| 维度 | 取值 |
|---|---|
| arm | learned、no-optimize |
| stage | initial、butter、burning-match、composite |
| probe | food-left/right、heat-left/right、obstacle-left/right |
| body | 每只蚂蚁分别报告，另给 colony 汇总 |
| noise | heading=0、step=0、exploration=0 |

Obstacle 是中性几何，只要求 input/code reachable，不要求 turn。

#### 冻结验收门

- food 与 heat：左右 sense 不同，`code_l1_delta > 1e-8`。
- food 与 heat：左右 turn 必须不同，`turn_delta >= 1e-4`；同时方向符号在同一 checkpoint 的重复运行中一致。
- 训练后的 turn sensitivity 不得低于 shared-initial 的 25%，除非绝对值仍高于预先声明的任务有效阈值。
- learned 的 action-head 参数必须有有限、非 NaN 的更新；no-optimize 必须保持 policy fingerprint 不变。
- pure/runtime/torch 的 final code、action distribution 与 turn 在约定容差内一致。
- 任一 body 失败都记录；colony 通过要求至少 80% body 通过，且不得有系统性左右同向。

#### 失败分支

- code 有差异、turn 无差异：检查 causal action head、motor readout 投影、动作裁剪和混合器。
- code 无差异：检查 19 维输入、encoder、checkpoint hydration 和 runtime backend parity。
- 初始通过、某阶段后失败：在首个失败 episode 内做二分 replay，定位导致塌缩的 optimizer update。
- learned/no-optimize 同时变化：优先修 checkpoint/optimization 隔离，不继续训练。

### 3.3 P0-B：Temporal switch 与 segment closure 审计

#### 实现任务

- 构造不含动作标签的确定性 transition protocol：
  - 稳态巡航；
  - food approaching；
  - pickup/carrying；
  - home approaching/delivery；
  - safe→harmful heat；
  - harmful→cooling/escape。
- 逐拍记录 world/self 两轨：
  - beta continuous、beta threshold、binary switch；
  - external switch pressure；
  - `steps_since_switch`；
  - segment open/close tick；
  - closure cause：world switch、self switch、milestone、terminal 或 max-step timeout；
  - SSL 前后 switch 参数和 histogram。
- 将 `AntStepRecord` 和 ecology report 扩展为可区分 segment closure cause，避免只看到长度 24。

#### 对照

- steady-state negative control：输入保持稳定，不应持续抖动切换。
- state-transition positive control：上述关键状态边界必须至少触发 switch 或有明确的 milestone/terminal closure。
- segment-credit on/off：只允许 credit 聚合方式不同，不允许 live policy forward 方程漂移。

#### 冻结验收门

- positive-control trace 至少出现一次真实 beta switch，且能定位到状态变化附近。
- negative-control 不得出现高频 alternating switch；switch rate 上限在测试中预声明。
- 至少一个 segment 由 beta switch 闭合，至少一个由 milestone/terminal 闭合。
- 正常 trace 不能全部依赖 24-step timeout。
- segment-credit on/off 的 sense、pre-credit action 和 rollout lineage 对齐。

#### 失败分支

- beta 永远低于阈值：审计 switch logit 分布、bias、threshold 与 external pressure，不先调 reward。
- beta 有越阈但 `is_switching` 为 false：修 runtime/backend parity。
- switch 正常但 segment 不闭合：修 segment assembler/OR 语义。
- 只有 milestone closure：暂不主张 temporal abstraction，继续修 beta 学习。

### 3.4 P0-C：Frozen evaluation owner-by-owner 审计

#### 实现任务

- 在 evaluation 每一步前后采集 owner-scoped fingerprints：
  - temporal policy parameters；
  - causal action head；
  - critic/optimizer state；
  - SSL/learning state；
  - memory/reflection writeback state；
  - runtime-only controller/replay state。
- 明确区分“允许变化的 episode/runtime state”和“禁止变化的 learned state”。
- 对当前最小复现优先覆盖：
  - `butter_only / seed=307`；
  - `heat_forced_escape / seed=101`。
- 增加 first-difference 报告：第一个变化 owner、字段、tick、前后 digest。

#### 冻结验收门

- `learning_enabled=False` 时，所有 learned-owner fingerprint 逐拍不变。
- policy、temporal-learning、critic/optimizer、action-head fingerprint 全部稳定。
- runtime controller 和 replay counters 可以变化，但不得被错误计入 learning fingerprint。
- settlement/lineage ≥0.99，drop=0。

### 3.5 P0 测试与产物

建议新增：

- `packages/vz-embodiment-ant/tests/test_ecology_action_chain_audit.py`
- `packages/vz-embodiment-ant/tests/test_ecology_temporal_switch_audit.py`
- `packages/vz-embodiment-ant/tests/test_ecology_frozen_evaluation.py`
- `packages/vz-embodiment-ant/src/volvence_ant/experiments/ecology_mechanism_audit.py`
- `scripts/audit_ant_ecology_mechanisms.py`

P0 报告建议 schema：`digital-ant-ecology-mechanism-audit.v1`。

### 3.6 P0 完成定义

只有 P0-A、P0-B、P0-C 全部 PASS，且对应回归测试通过，P0 才算完成。P0 完成后冻结：

- action sensitivity 计算方法；
- switch/segment closure 定义；
- learning/runtime fingerprint 边界；
- P1 使用的 runtime profile。

## 4. P1：课程与能力门槛重构

### 4.1 P1 目标

消除“near 偶遇”和“aggregate event count”造成的 mastery 假阳性，让每个 mastery 都代表可重复的能力闭环。

P1 仍是 development evidence，不产生正式 promotion artifact。

### 4.2 数据模型改造

- pickup/delivery/heat 事件必须带 `body_id` 和 episode lineage，支持 per-ant 统计。
- mastery 从阶段累计计数改为 episode/layout success rate。
- 单独记录：
  - encounter：是否进入对象有效感知范围；
  - acquisition：是否 pickup；
  - completion：是否 delivery；
  - safety：harmful ticks、escape latency；
  - navigation：路径效率和 timeout；
  - temporal：switch 与非-timeout segment closure。
- `local-valence-off` 重命名为 `dense-local-shaping-off`，明确稀疏 milestone reward 仍存在。
- 课程语义改变后提升 curriculum/report schema 版本；旧 loader 必须拒绝混读。

### 4.3 新课程顺序

#### Stage 0：Near bootstrap（不计最终 mastery）

- 目的仅是保证产生足够 pickup/delivery/heat 样本供 optimizer 学习。
- 增加与 forced-escape 对称的 forced-return bootstrap：每个 body 在巢外专属黄油源上从未携食
  状态起步，经真实 contact 完成 pickup 后才进入返向；同步 body-side path integration，左右
  `±3π/4` home-bearing 均衡；controller 仍只消费常规 sense，不提供目标方向或 action label。
- 初始 forced-return block 后，在后半程每 3 个 primary layout 交错一次返向复习（共 5 局），
  避免后续 heat/composite/neutral-context 学习覆盖 pickup-triggered return mapping。
- forced-return near 的专属 source 半径大于 plant 单步上限，保证第一 act 真实 pickup；
  `±3π/4` 下零转向从下一步起远离巢，只有拾取后及时切换并主动修正的策略才能 delivery。
- near 结果进入训练报告，但不得使任何正式能力 gate 通过。

#### Stage 1：Butter medium/far

- medium 和 far 分开判定。
- episode success 定义为同一 body 完成 pickup→delivery 闭环。
- 需要多个布局 seed，不能靠同一地图重复累计。

#### Stage 2：Heat forced escape

- 只判断逃逸率和逃逸时延。
- forced escape 成功不能用于 route-avoidance mastery。

#### Stage 3：Heat route avoidance + foraging

- 必须完成 pickup→delivery，同时 harmful tick rate 受控。
- “没有进入热区但也没有完成任务”记失败，不记成功避热。

#### Stage 4：Butter + neutral stick

- 木棍无 payoff。
- 只判断有中性几何时是否仍能完成 pickup→delivery。
- contact 数只作诊断，不进入 score。

#### Stage 5：Composite

- 同时满足完整 foraging、热暴露约束和 temporal behavior gate。
- 只有 Stage 1–4 都通过才解锁。

### 4.4 P1 mastery 门槛

以下门槛在首轮 P1 结果前冻结：

- 每个 medium/far tier 至少 5 个独立 layout seed。
- 至少 60% layout 成功。
- 成功 layout 中至少 60% body 完成该能力；4 ants 时即至少 3 ants。
- Butter：同一 body 必须完成 pickup→delivery。
- Forced escape：每个成功布局至少 60% body 离开热区，且报告 median/p90 escape latency。
- Route avoidance：至少 60% layout 完成 foraging，harmful tick rate ≤5%。
- Neutral stick：至少 60% layout 完成 foraging；contact 不计分。
- Composite：至少 60% layout 完成 foraging，且 harmful ticks 不高于 matched no-optimize。
- 所有阶段的 food/heat action probe 必须继续通过 P0 冻结门槛。
- held-out 必须出现真实 beta switch，且不能全部由 timeout 关闭 segment。

### 4.5 P1 实验矩阵

开发规模：

| 项目 | 配置 |
|---|---|
| ants | 4 |
| latent | 16 |
| training layouts | 每 tier 5 seeds |
| validation layouts | 每场景 3 seeds |
| diagnostic held-out | 每场景 5 seeds |
| arms | learned、no-optimize、cold、dense-local-shaping-off、segment-credit-off |

所有训练臂从同一 initial checkpoint 分叉，并重放同一冻结 schedule。训练 schedule 由预声明布局生成，不能由 learned 的偶然成功动态缩短其他 arm 的可比预算；若保留 early-stop，必须同时冻结“实际执行 schedule”和“最大预算 schedule”，并明确 estimand。

### 4.6 P1 对照与诊断基线

- `oracle-steering diagnostic`：只验证地图、碰撞、pickup/delivery 和 payoff 管线，不进入学习结论。
- random：校准 encounter floor。
- FixedRule：校准任务可解性和布局难度，不得写入 learned checkpoint。
- no-optimize：隔离真实 policy optimization。
- dense-local-shaping-off：评估连续 shaping 的贡献。
- segment-credit-off：评估 segment credit 的贡献。

如果 FixedRule/oracle 在某布局也无法稳定完成，先修环境或布局，不允许把失败归因于 learned policy。

### 4.7 P1 完成定义

P1 PASS 需要同时满足：

- 所有 Stage 1–5 mastery gate 通过；
- learned 在 butter、heat route、neutral stick、composite 上均不弱于 no-optimize；
- learned 的预声明综合能力分数对 cold/no-optimize 的 paired effect 为正；
- action sensitivity、temporal switch、frozen evaluation 三类 P0 gate 无回归；
- checkpoint roundtrip 与 replay lineage 继续通过；
- P1 重跑一次能够得到同方向结果，避免单次训练偶然性。

如果单能力通过而 composite 失败，停在 P1，优先检查灾难性遗忘与能力组合；不得进入 P2。

## 5. P2：正式证据与 promotion

### 5.1 P2 目标

在冻结实现、课程、指标和阈值后，运行足以支持正式 claim 的 matched-control 证据矩阵。P2 不再调参。

### 5.2 正式配置

| 项目 | 配置 |
|---|---|
| ants | 8 |
| latent | 16 |
| stage rounds | 80 |
| max stage episodes | 4 或 P1 冻结后的等价预算 |
| validation rounds | 80 |
| held-out rounds | 120 |
| held-out seeds | 至少 5，预先冻结 |
| independent training seeds | 至少 3；目标 5 |
| device | 固定一种正式设备；另跑 CPU parity smoke |

正式训练 seed 与 held-out seed 在开始前写入 config，运行后不得替换失败 seed。

### 5.3 正式 arms

核心矩阵：

- learned；
- no-optimize；
- PE-off；
- ETA-off；
- dense-local-shaping-off；
- segment-credit-off；
- FixedRule；
- end-to-end RL；
- random。

每个可学习 arm 共享相同初始 checkpoint、训练布局、episode budget 和 evaluation layout。不能用 random 代替 PE/ETA 消融。

### 5.4 P2 分批执行

#### P2-A：正式 preflight

- 1 个 training seed，完整 8-ant 配置；
- 运行全部测试和 P0 probes；
- 检查磁盘、运行时间、artifact 体积、determinism 和设备 parity；
- preflight 不进入最终统计。

#### P2-B：核心 confirmatory matrix

- learned、no-optimize、PE-off、ETA-off、FixedRule、E2E-RL、random；
- 至少 3 个独立 training seed × 5 held-out seed；
- 任何代码或门槛变化都会使整批失效并重新开始。

#### P2-C：机制消融补全

- dense-local-shaping-off；
- segment-credit-off；
- 必要时加入 P1 预注册的 action-head-off，但不能事后按结果选择。

### 5.5 Primary endpoints

按优先级预注册：

1. Butter medium/far pickup→delivery success rate；
2. Heat-route foraging success 与 harmful tick rate；
3. Neutral-stick context 下的 foraging success；
4. Composite foraging success 与热暴露；
5. learned 相对 no-optimize/cold 的 paired effect；
6. PE-off、ETA-off 对 learned 增益的削弱；
7. temporal switch 与非-timeout segment closure；
8. replay/fingerprint/archive 工程 gate。

Secondary endpoints 包括路径效率、首次 pickup tick、escape latency、per-ant 方差、动作平滑度和 action-probe sensitivity。Secondary endpoint 不能挽救 primary endpoint 的失败。

### 5.6 统计计划

- 地图/seed 使用 paired comparison。
- 同时报告原始计数、比例、effect size 和 bootstrap 95% CI。
- training seed 是独立重复层级；不能把同一 checkpoint 的多个 ant-tick 当独立样本。
- per-ant 指标使用层级汇总，避免伪重复。
- 多个 primary comparison 使用预声明的层级检验或 multiplicity correction。
- 缺失/中断 run 不做有利方向插补；报告原因并按预注册规则重跑完整 shard。

### 5.7 P2 promotion gate

只有以下条件全部满足，checkpoint 才能标记为 `PASS` 并允许 demo loader 加载：

- learned 在所有核心 ecology 场景达到能力门槛；
- learned 对 cold/no-optimize 的 primary paired CI 下界 >0；
- PE-off 与 ETA-off 显示预声明的因果退化；
- learned 不弱于 FixedRule 的安全门槛，并在预声明学习指标上显示自身优势；
- action sensitivity、temporal dynamics、frozen evaluation 全部通过；
- replay settlement/lineage ≥0.99、drop=0；
- archive roundtrip、corruption rollback、schema compatibility 全部通过；
- 至少 5 个 held-out seed，且不存在事后删 seed；
- provenance 完整、工作树干净、artifact hash 可复核。

任一项失败即 `BLOCK`。BLOCK artifact 保留用于诊断，但 loader 必须拒绝。

## 6. 工作包、依赖与建议提交顺序

| 工作包 | 内容 | 依赖 | 完成标志 |
|---|---|---|---|
| P0.1 | action-chain instrumentation | 无 | episode-by-episode collapse 可定位 |
| P0.2 | temporal switch/closure audit | 无 | positive/negative control 通过 |
| P0.3 | frozen owner audit | 无 | learned owner 逐拍稳定 |
| P0.4 | P0 回归套件与报告 | P0.1–P0.3 | P0 report PASS |
| P1.1 | per-ant event lineage | P0 PASS | 可计算 per-ant mastery |
| P1.2 | tier-specific curriculum | P1.1 | near 不再掩盖 medium/far |
| P1.3 | P1 controls 与 gate | P1.2 | development matrix 完整 |
| P1.4 | P1 两次重复运行 | P1.3 | 同方向结果，P1 PASS |
| P2.1 | 预注册 config/schema | P1 PASS | config digest 冻结 |
| P2.2 | 正式 preflight | P2.1 | 8-ant 全链路通过 |
| P2.3 | confirmatory shards | P2.2 | 全部 shard 完成 |
| P2.4 | 统计、bundle、loader gate | P2.3 | PASS 或诚实 BLOCK |

建议每个工作包独立提交，避免将 instrumentation、行为修复、课程改造和正式结果混入同一个 commit。

## 7. 测试层级

每个阶段按以下顺序验证：

1. 单元测试：probe、fingerprint、event lineage、mastery 计算；
2. 契约测试：owner 边界、replay lineage、archive rollback；
3. 小预算 deterministic test：1 ant、固定 seed；
4. P0 mechanism audit：4 ants、短 trace；
5. P1 development run；
6. P2 preflight；
7. P2 formal matrix。

任何低层测试失败都停止更高层运行。

## 8. 计算与运行管理

- P0 以 CPU deterministic path 为主，另做一次 MPS/CUDA parity smoke。
- P1 使用 4 ants，先测得每 1,000 ant-step 的时间与 artifact 增长率。
- P2 的实际 wall-clock、磁盘和 shard 数量在 P1 完成后依据实测数据冻结，不提前凭感觉估算。
- P2 按 `(training_seed, arm)` 分 shard；每个 shard 独立 manifest，最终聚合器只接受 config digest 一致的完整 shard。
- 每个长 run 持续写 append-only trainlog 和阶段 checkpoint，崩溃后从 owner archive 恢复；恢复 run 必须保留 lineage，不能静默重启并合并计数。

## 9. 决策表

| 观察 | 判断 | 下一步 |
|---|---|---|
| latent 有左右差异，turn 为 0 | action head/readout 断链 | 留在 P0-A |
| beta 始终不越阈 | temporal switch 未工作 | 留在 P0-B |
| evaluation fingerprint 变化 | 冻结/指纹边界错误 | 留在 P0-C |
| near 成功，medium/far 失败 | encounter 或泛化不足 | 留在 P1 单能力 |
| 单能力都通过，composite 失败 | 组合/遗忘问题 | 留在 P1 composite |
| learned 与 no-optimize 无差异 | optimizer 无因果收益 | BLOCK，不进 P2 |
| P1 通过，P2 CI 跨 0 | 证据不足或不稳定 | P2 BLOCK；不改阈值 |
| 全部 primary 与工程 gate 通过 | 正式能力成立 | 生成可加载 promotion bundle |

## 10. 最终交付物

完成本计划后应得到：

- 一套可重复运行的 ecology mechanism audit；
- 一套不会把 near 偶遇误判为 mastery 的课程；
- per-ant、per-layout、per-training-seed 的分层数据；
- 完整 matched-control 正式报告；
- 一个诚实的 `PASS` 或 `BLOCK` checkpoint bundle；
- 对“价态是否形成方向行为、temporal abstraction 是否贡献、learned 是否优于对照”的可审计结论。
