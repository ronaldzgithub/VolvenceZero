# 时间抽象与内部控制 Spec

> Status: draft
> Last updated: 2026-08-03
> 对应需求: R3, R4

## 要解决的问题

如何让系统在 token 之上的抽象层级做决策和学习，实现子目标级控制？

## 关键不变量

- 实时行为可通过内部状态转换引导，而非仅通过表面文本损失
- 抽象动作可组合、可训练、无需详尽手动标签
- 冻结基础模型是发现时间抽象的前提（ETA rate-distortion 证明）。
  **2026-08-01 本仓首次复现读数为 `kill-eta`，但该次运行不具备正式资格**
  （`artifacts/eta_rate_distortion_20260801`，见变更日志）：仪器通过联合臂有效性
  对照，冻结臂扫遍 alpha 网格无论文预言的近垂直 gap。判据阈值与 verdict 映射在
  运行前已固定于源码，因此不是事后调参；但该次运行无预注册 artifact、无源码冻结
  （`working_tree_dirty=true`）、与七日矩阵并行占用同一 MPS、无逐 cell checkpoint，
  也未 attest 算子级 CPU fallback 已禁用。因此其状态为 mechanism-grade
  `kill-eta (not preregistered)`：**足以阻止把该前提当作已确立的依据引用，
  不足以据此摘除 ETA 主张或退役 `_step_impl_legacy`**；两者需等预注册重跑。
  **2026-08-03 更新**：预注册重跑后 **Gate 1 已 PASS**——修好尺子后（smooth
  posterior + v4 分段揭示协议 + switch-gated KL + hard-st 离散门 + 300 updates），
  冻结臂 rate 轴 spearman −1.000 / span 1.933，且 never-switch 崩塌解除
  （heldout boundary F1 全 alpha 0.240–0.671）。仪器有效性已确立。
  **2026-08-03 再更新**：Stage 2 两代仪器——v1 FAIL（0.131/0.166）经仪器审计
  **定罪仪器**（计划载体为哈希指纹，heldout 信息天花板 0.1805 < 及格线
  0.25，构造性不可过）；v2 可读仪器（v4 staged-plan 语料 + 轨迹前缀 probe，
  ceiling 1.0 验证）实测**残差流大幅承载 active subgoal**（裸 Qwen 0.901 /
  续训 0.944，`2×chance` 与因果对照 PASS），仅 `随前缀上升` 判据在显式揭示
  制度下 regime 错配（保持衰减 0.979→0.879）→ 按字面 v2 FAIL 封存。
  **2026-08-03 三审**：v3（用户授权，retention 判据 + 全新 seed 20260804 挡
  forking paths）修正后两条件双 PASS（0.967 = 7.7×；late 0.918 / 衰减
  0.077），但因果对照**反向失效——裸 Qwen 基底 0.977 已在天花板反超续训臂
  0.967**，字面仍 FAIL 封存。三轮合并判读：实质命题"0.5B 残差可线性承载
  active subgoal"跨两 seed 四臂复现（0.901/0.944/0.977/0.967）；被证伪的
  是"续训必要性"对照前提。**2026-08-03 裁定**：用户程序级裁定 Gate-2
  看门目的已实质达成，Stage 3 解锁（三 FAIL 封存不改判）。
  **2026-08-04 终审**：36/36 cells 完成，正式 verdict = **`kill-eta`**；
  双臂可分且 frozen rate 轴有效，但 frozen 无近垂直 gap，joint 反而检出
  gap。该判词否定当前 16 维折叠入口 + additive steering / free bias 的
  operationalization，不外推为理论普遍证伪。Stage 4 不启动，production
  WiringLevel 不变；P1 等价性诊断只归因、不重判。见
  [`eta-llm-transfer-evidence.md`](./eta-llm-transfer-evidence.md)
- 内部控制空间维度低于原始 token 动作空间

## 工程挑战

- 实现 metacontroller：从残差流中发现时间抽象动作
- 实现切换单元：稀疏切换、组合泛化
- 实现 Internal RL：在控制器代码空间（而非 token 空间）执行强化学习
- 将抽象动作与产品级行为（regime、策略）对齐

## 算法候选

来自 `docs/next_gen_emogpt.md`：

### Metacontroller 架构（ETA 附录 B.3）

```
残差流 e_{1:T}
    │
    ├──→ 内部序列嵌入器 → s(e_{1:T})  [全局嵌入，训练时非因果]
    │
    ├──→ 编码器 (GRU) → μ_t, Σ_t → z̃_t ~ N(μ_t, Σ_t)
    │         │
    │         └──→ 切换单元 → β_t ∈ [0,1]
    │                  │
    │                  └──→ z_t = β_t ⊙ z̃_t + (1-β_t) ⊙ z_{t-1}  (Eq.2)
    │
    └──→ 解码器 (FFN) → U_t = Decoder(z_t)
              │
              └──→ 残差流控制: e_{t,l} ← e_{t,l} + U_t · e_{t,l}  (Eq.1)
```

### 切换单元

- `β_t ≈ 0`：保持当前控制器（继续执行当前抽象动作）
- `β_t ≈ 1`：切换到新控制器（开始新的抽象动作）
- 训练后自发学会准二值、稀疏切换行为，切换时刻对齐子目标边界

### Internal RL（ETA 附录 B.5）

| 概念 | 定义 |
|------|------|
| 观测 | 残差流激活 `e_{t,l}` |
| 动作 | 控制器代码 `z_t`（低维，`n_z < n_e`） |
| 环境 | 原始环境 + 冻结自回归模型 + 解码器 + 切换单元 |
| 策略 | 因果抽象动作策略 `π(z_t | e_{1:t})` |

**优势**：动作空间降维、时间尺度压缩、信用分配简化、探索效率提升。

运行时对 terminal environment outcome 的跨层 lineage 由 orchestrator 在 outcome
提交时从 `TemporalAbstractionSnapshot` 捕获：
`active_abstract_action / action_family_version / controller_state.code digest`。
这只是保存 temporal owner 已发布的不可变证据，不把 family 语义解释权迁给 runtime。
family ID 本身是 opaque latent identity；没有 gated semantic decoder 产出的 typed
action schema 时，application/expression 禁止将它或同 episode 的 action 文本解释成
可复用行为规则。

### 自监督训练目标（ETA Eq.3）

```
L(φ) = Σ_{(o,a)~D*} Σ_t [
    -ln p_{θ,φ}(a_t | o_{1:t}, z_{1:t})     // 动作预测损失
    + α · D_KL(N(μ_t, Σ_t) || N(0, I))      // 先验匹配正则化
]
```

### CMS 增强 Metacontroller（NL×ETA 附录 C.2）

用 CMS 替换 GRU 编码器，获得多时间尺度记忆：
- 高频层：每步更新，跟踪当前子目标执行进度
- 中频层：每 c_2 步更新，记忆近期子目标序列模式
- 低频层：每 c_3 步更新，保存跨 episode 的策略偏好

## 接口契约

**消费的输入**：
- `substrate` 快照：当前可实现的 substrate surface；当前阶段优先消费 `feature_surface`，只有在 hook 可用时才消费 `residual_activations`
- `memory` 快照：相关记忆上下文
- `reflection` 快照：策略沉淀（控制器参数更新）
- `prediction_error` 快照：上一轮 outcome mismatch 的 carryover learning signal，用于 owner-side controller 调节与 schedule 触发
- `credit` 快照：抽象动作级与 session-level delayed credit 的聚合信号；当前主要由 consolidation 路径消费，用于 owner-side action-family payoff / competition 更新
- `experience_fast_prior` 快照：由 application slow loop 压缩出的 delayed-credit fast bias，用于 owner-side action-family continuation / switch pressure 调节
- `active_mixture` 快照（**protocol-temporal-prior bridge，orchestrator-mediated 一轮 carryover**）：BehaviorProtocol 激活混合的 activation-weight 分布，被压缩为一个有界的 `beta_t` switch-pressure prior（混合越 dominant → 越偏 continuation；越 ambiguous → 越偏 switch）。**不是同轮依赖**：`active_mixture`（owner 在 vz-application）在 propagate DAG 中处于 temporal 下游（`active_mixture → retrieval_policy → world_temporal`），声明为同轮 dependency 会成环，因此由 runtime 把上一轮的值经 `observe_active_mixture_carryover` out-of-band 喂入 temporal owner。三态门控 `FinalRolloutConfig.protocol_temporal_prior`（DISABLED 默认=字节等价基线 / SHADOW=只记录证据不入 `beta_t` / ACTIVE=入 `beta_t`）。协议只作 prior，不替代 metacontroller 决策（R4）；single-protocol 混合产生零差分（fixture 字节等价）

**产出的输出**：
- `temporal_abstraction` 快照：`TemporalAbstractionSnapshot`
- `active_abstract_action` / discovered family 保持无业务语义的 controller identity。具体行动由 application 的 `CaseMemorySnapshot.action_grounding` 解释，再由 `ResponseAssemblySnapshot.action_realization` 与同拍 action id 绑定；temporal owner 不保存 case 文本，expression 也不得建立 family-id → 行为字符串表。
  - 控制器状态（`z_t`, `β_t`, `steps_since_switch`）
  - 当前抽象动作的语义描述
  - 控制器参数哈希

**当前实现口径**：

- P08 先固定接口和状态 contract，不承诺完整 ETA 训练闭环
- 当前实现已支持 `placeholder` / `heuristic` / `learned-lite` / `full-learned` 四类 temporal policy；`full-learned` 是默认 runtime owner，`learned-lite` 保留为 fallback / rollback baseline
- `learned-lite` 当前仍是最小可训练控制器，不等同于 full ETA metacontroller 或因果 `π(z_t | e_{1:t})`
- 第二阶段 runtime 已补充一个独立的参数化 causal z-policy sandbox，支持 dual-track rollout、checkpoint/rollback 和 trajectory-level clipped surrogate objective；当前其 online owner 由 `ETANLJointLoop` 承担，offline/batch owner 由 `SSLRLTrainingPipeline` 承担
- `learned-lite temporal` 与 causal z-policy 当前共享同一控制器参数 store，但 owner 侧已引入显式 `learning_phase` / `structure_frozen` 边界：SSL 阶段允许 discovered action family 更新，RL/runtime 阶段默认冻结结构层，仅在 owner API 内做受限策略更新
- 当前 ndim metacontroller 已收敛到**单一 owner 参数面**：SSL trainer、runtime policy、internal RL、rare-heavy snapshot/export/import 共享同一个 `MetacontrollerParameterStore` 可见的 encoder/switch/decoder 权重，不再允许 ndim 路径在 trainer/runtime 内各自持有私有网络参数
- PE→runtime code 的消费面现有显式、可回滚 owner gate：
  `MetacontrollerParameterStore.runtime_prediction_error_modulated_code()`
  把 PE 学到的 `residual/memory/reflection` temporal weights 编译成
  identity-centered `[0.5, 1.5]` code gain；PE=0 或
  `prediction_error_runtime_modulation_enabled=False` 时严格 no-op。Gate 1
  v3 证明该通路能改变 next-session code，但 policy-loss effect
  `=-0.000881360`，方向为负，因此生产默认 flag 为 `False`，不得把“已接线”
  写成 PE 学习增益；重新启用必须有 fresh 信号设计与 causal evidence。该 gate
  现在由 `FinalRolloutConfig.prediction_error_runtime_modulation` 显式传入，并同时覆盖
  shipped companion bootstrap 使用的 legacy 3 维 forward 与 ndim forward；默认仍为
  `DISABLED`。`BrainConfig.external_prediction_error_drive=False` 还必须关闭 late
  `TrackTemporalConsolidationModule` 的 PE 参数写入与 PE family-outcome 输入，但保留
  `PredictionErrorSnapshot` publication/readout，防止 PE-off matched arm 只关 joint-loop
  日志、却从 consolidation 旁路继续学习。
- ndim encoder 的 `n_input` 与 latent `n_z` 是两个独立契约：`n_input` 必须等于 substrate 发布的完整残差宽度，GRU 再把它压缩为低维 `n_z`。runtime facade 负责在构造时声明并校验 `n_input`；禁止按 `n_z` 截断后部感知通道。encoder checkpoint 与 compact parameter fingerprint 均绑定 `n_input`。
- Transformers residual publisher 的 `activation_width` 是显式、正整数配置：默认 `8` 保持现有生产成本与回滚行为，hidden width 不超过该值时逐坐标原样发布，超过时做确定性 chunk mean。Gate 2 full-width evidence 显式使用 Qwen hidden width `896`，并在 artifact 中同时登记 manifest 与 runtime provenance；两者不一致必须 fail loudly。
- 当前 `TemporalModule` 默认以 `full-learned` 作为 runtime owner policy，并可通过 owner API 导出 machine-readable metacontroller runtime state；这条导出链不改变 `temporal_abstraction` 公共 snapshot schema
- 当前 temporal owner 已支持显式消费 `experience_fast_prior`：它不会直接接收 session-post 对私有参数的写穿，而是把 public fast prior 转成 owner-side continuation bias 与 switch-pressure delta，再在 `step()` 内部作用于 action family continuation 和 switch unit
- 当前这条 owner-side fast-prior consumption 已进一步前推到 action-family discovery / competition：public fast prior 不直接指定 family，而是作为 owner 内部的 `reuse/create/split/competition` 偏置，影响当前 active family 的 continuation、competition score、monopoly / stagnation 调节
- 当前 temporal owner 对 `experience_fast_prior` 的消费已从“轻量 telemetry bias”收紧到更直接的 continuation / exploration pressure：`action_bias`、`family_bias`、`sequence_bias` 与 `prior_strength` 会共同塑造 `switch_pressure_delta`，并放大 owner-side family continuation / competition 信号，使 application delayed credit 能更直接进入 ETA fast path，而不只停留在检索 readout 或 evaluation evidence
- 当前 ETA proof benchmark 也已接入 temporal fast prior：训练 rollouts 会把 delayed credit / family reuse / sequence completion 压缩成一个小的 temporal fast prior，再回灌到后续 proof rollouts；benchmark 会显式发布 `temporal_fast_prior_strength` 与 `temporal_fast_prior_switch_delta`
- Gate 2 residual matched-control 证据通过
  `InternalRLEnvironment.residual_control_mode` 在 decoder 与 substrate
  backend 之间执行 `identity / zero / shuffled / reversed` 变换。默认值为
  `identity`，不改变生产 runtime；evidence transition 同时保留变换前
  control 与实际注入值，zero/shuffle/reverse 不允许只改报告、不改真实
  forward。Qwen2.5-0.5B CPU 单 seed 已证明真实 hook 注入机制与 zero no-op。
  evidence lane 用真实 residual trajectory 做 SSL bootstrap，随后冻结
  encoder/switch/decoder，只允许 causal `z_t` policy/action head 消费
  `NLL(0)-NLL(U)` 派生 PE；latent unit clamp 必须同时作用于采样和冷启动，
  legacy 3 维与 ndim 路径首次 abstract action 都必须真实切换。训练使用
  22 个无语义 action label 的 counterfactual 候选，raw NLL/PE 进入 artifact；
  每 prefix 只把实际最低 NLL 的候选作为 typed `direct_action_target`，action
  head 按 `target_z - current_mean` 做有界更新，普通 PPO/runtime replay
  路径不变。action-head observation width 与 3 维 actuator 已拆成独立契约：
  默认仍可等宽，Gate 2 实测使用 12→3。v19 train oracle 为 `+0.032274`，
  最佳固定候选为 `+0.005697`；direct optimizer 的 v20 8-update 候选使 train
  成为最佳臂并在 development-heldout 得到 `+0.005979`，但 eval 仍为
  `-0.016155`。v22 证明 eval oracle 为 `+0.021150` 而固定最佳为 zero，
  即需要状态条件选择。48 维 signed feature hash 与 24 维 hybrid state 均在
  未见分区退化，故 Gate 默认回滚 12 维。v25 在算法不变时把独立无标签训练
  分布从 8 条 route / 45 个 prefix 扩到 16 条 route / 96 个 prefix，
  `768` 条 direct optimizer transition 消费 `16896` 次候选评分；eval 反而为
  `-0.017924`，development-heldout 为 `-0.001057`。同一分区的 oracle 分别
	  为 `+0.022156` / `+0.035008`。v26 的 PCA selector、v27 的 train-CV
	  PCA/ridge ladder、v28 的无 hash published-coordinate kernel selector 均
	  未通过注入门；v28 同时暴露 publisher 默认把每层 `896` 维 hidden 压成
	  `8` 维。v29 显式发布完整 `896` 维残差，三层 mean/latest/trend 形成
	  `8076` 维 prefix state，但 kernel selector 仍只有 train route-CV
	  `-0.002488`、eval `-0.019454`、development-heldout `-0.001422`，
	  top-3 近随机。故当前只证明执行器可达与机制生效，尚未证明 controller
	  因果增益；selector live injection 保持 disabled。
- v29 后不再把问题表述为“换一个 learned residual encoder 即可”。v30 将
  ex-post 单条 continuation 标签替换为决策时可定义的 prefix-level expected
  value：同一 prefix 使用固定种子的 target continuation cohort 求 22 个动作的
  平均 `NLL(0)-NLL(U)`，另用完全独立的 audit cohort 只做模型选择与只读验证。
  target/audit 的采样 key、cohort 和 index 进入 SHA256 seed；fresh validation
  在运行前冻结，selector 拟合后不得更新。`OpenWeightResidualRuntime` 同时提供
  `score_continuations(...)`：默认实现保持逐条兼容，transformers backend 对同一
  prefix/control 做一次右填充 batch forward；返回顺序和逐条 NLL 必须等价，
  空 cohort 或 token prefix 错位 fail loudly。
- v30 的 Qwen2.5-0.5B、full-width 896、CPU、单 seed 2 target + 2 audit 校准
  生成 `576` 条固定种子 continuation。selector 在 train target / fresh
  validation target / eval target 上的 selected delta 分别为
  `-0.001990 / -0.003317 / -0.005052`；独立 audit 分别为
  `+0.001393 / -0.001125 / -0.000569`。fresh validation audit 为负，故
  `selector_ready_for_shadow_injection=false`，live injection 保持关闭。
  各 split 的 action oracle 仍为正，说明动作可达而 prefix→action 映射没有
  泛化。该 2+2 轨迹是校准/反证，不替代 manifest 默认 4+4、跨 seed 或 locked
  confirmation。下一包必须把训练目标迁到真实 downstream outcome / environment
  PE，不能继续在模型自身采样分布的 NLL 上调 probe 或阈值。
- v31 将 counterfactual selector target 迁到正式结果链：
  `candidate z_t -> residual forward -> EnvironmentOutcome.measurement ->
  PredictionError -> pe:action credit`。primary target 是运行前冻结的 proof
  subgoal signature；环境 owner 只比较干预后实际 residual snapshot 与该 target，
  计算 payoff 时禁止读取 requested control。zero-control 的实际结果冻结为事前
  prediction，各候选实际结果由 PE owner 结算，再由 credit owner 发布 signed
  action credit；该 credit 才能进入 `direct_action_target`。真实 residual prefix
  calibration 在此模式下必须关闭，避免把验证轨迹写进 primary target。下一
  prefix 的未干预 residual trajectory 是独立 audit surface，只参与 train-route
  CV 模型选择与只读验证，不参与 selector 拟合。旧
  `sampled-prefix-expected-value` 保留为显式回滚模式；v31 默认
  `environment-outcome-pe-credit`，self-NLL 不再是动作价值 owner。
- v31 每个候选必须发布不可变 `ETACounterfactualOutcomeRecord`，包含
  observation/segment ID、候选 z、实际 control、primary/audit target、实际
  residual signature、downstream effect、PE magnitude 与 action credit。
  evidence bundle 将其独立写入 `counterfactual_outcomes.jsonl`；仅有汇总均值
  而缺候选级 lineage 时，Gate 2 结果链证据无效。selector live injection 仍
  维持 disabled，直到 fresh validation audit credit 为正并通过既有注入门。
- v36 selector artifact 使用 `residual-action-selector.v1` JSON 契约冻结。
  temporal owner 同时支持 linear PCA-ridge 与实际 Gate 2 使用的 linear-kernel
  ridge artifact；payload 必须携带 model kind、全部数值参数、shape 与
  `model_fingerprint`。反序列化时 owner 重算与拟合阶段完全相同的 fingerprint，
  并验证 input/action/latent/training-row dimensions；指纹、shape、非有限值或
  ridge 约束漂移必须 fail loudly。Gate 2 每个 seed 在 train-only fit 后立即
  round-trip 冻结，evaluation/SHADOW arm 只消费恢复后的 artifact；bundle 的
  `selector_artifact.json` 绑定 run/seed、fit split 与 learned control-basis
  fingerprint。该 artifact 不进入 live session，不新增第二 temporal owner。
- Gate 2 v38 的 development-only
  `residual-state+committed-control-summary.v1` 由 temporal selector 单点
  解释：在既有 full residual mean/latest/trend + summary 后追加最近 k=2
  committed controls 的 10 维有界 readout（aggregate/latest/trend 各 3 维，
  active fraction 1 维）。空历史严格为零；非法 window、control shape 或非有限
  值 fail loudly。该 readout 不新增公共 snapshot/slot，只进入 train-only
  selector artifact；默认关闭时保持 v37 feature 字节不变。
- 五杠杆 L3 的新机制使用
  `residual-state+relationship-owner-readout.v1`。temporal owner 在完整 residual state 后
  追加 `RelationshipConditioningModule` 发布的**完整、有序、不透明** readout，只执行
  `(2x-1) × confidence` 的有界变换；不解释 label、不重建 relationship state，
  不读原文。输入必须是 non-cold、positive-confidence 的 `RELATIONSHIP` owner
  readout，错 bank、cold start 或零置信度立即 fail loudly，禁止伪装为新机制却静默
  回落 8076 维 v35 无条件 selector。该 feature 只供隔离 Gate 2 证据 selector，
  不进 live session、不新增 slot，也不改变 Relationship bank 的语义 owner。
- L3 formal 已在 `eta-gate2-longitudinal-conditioned.v1` 下跑满 fresh seed
  1301 的 510 条 matched outcomes。selector 相对 action permutation / zero /
  matched wrong-condition 的 mean gain 为 `0.003287669 / 0.004308079 /
  0.000055161`，wrong-condition session positive rate=`0.156863`；四个效应门
  全部失败，因此触发 single-seed stop-loss，禁止后续 seeds 1313/1327。
  该载体契约与 fail-loud 机制保留，但不进 live selector，Gate 2 继续只保留
  v35 的受限 open-loop `causal-supported`。
- 当前 proof profile 已包含 matched ablation `full-no-fast-prior`：它保留 full internal RL + causal replacement，但关闭 temporal fast prior ingestion，用于衡量 fast prior 对 held-out family reuse、credit alignment 与 strong success 的增益
- 当前 runtime 已新增 `full-learned` metacontroller owner：内部采用 sequence encoder + learned switch unit + residual decoder 的最小可执行实现，优先消费 `substrate.residual_sequence`
- 当前 `AgentSessionRunner` 默认已切到 hook-shaped residual substrate adapter；默认 session turn 会优先发布 `SurfaceKind.RESIDUAL_STREAM` 而不再停留在纯 trace-sim feature adapter
- `AgentSessionRunner` 每拍重建 runtime module 时，orchestrator 必须把上一拍 owner 发布的
  `world_temporal` / `self_temporal` 不可变快照恢复给对应
  `TrackTemporalModule`。只恢复 snapshot version 而丢失 value 会切断 public
  segment continuity，使 `closed_segments` 永远为空；consumer 禁止根据 action
  文本、family ID 或私有 joint-loop buffer 重建 segment。`steps_since_switch=0`
  表示该拍刚开启 segment，因此下一次 switch 的正式闭合区间从上一拍开始，
  owner 计算 `open_turn_index` 时必须包含该开启拍。
- `learned-lite` 仍保留为 fallback / rollback baseline；`full-learned` 是当前默认 temporal owner
- 当前 online owner 的 rollback 已提升到 cycle 级：坏周期会恢复到 SSL 之前的 checkpoint，保证 temporal owner 不留下半轮 SSL/RL 混合脏状态
- 当前 rare-heavy v0 允许 temporal owner 导出/导入 parameter snapshot：offline pipeline 负责产出 artifact，runtime owner 负责 apply / rollback，不引入第二个 temporal state owner
- metacontroller runtime state 已扩展为可发布 prior mean/std、posterior mean/std、posterior sample noise、`z_tilde`、posterior hidden state、posterior drift、decoder output / applied control、latest switch gate，以及 binary switch ratio / sparsity / persistence window 等 owner-visible ETA 证据；当前还会显式发布 `active_label` 对应的 discovered family、`learning_phase`、`structure_frozen`、family summary/version
- discovered action family 当前已不再从固定 seed prototype 起步，而是从空 bank 开始，并在 temporal owner 内执行 bounded `reuse/create/split/merge/prune`
- `TemporalAbstractionSnapshot` 当前新增 `action_family_version`，作为 `temporal -> dual_track -> regime/evaluation` 的最小版本桥，不把完整 family bank 暴露给 consumer；该字段是整个 discovered-family bank 的全局 revision，不是 active family 的 incarnation/version，稳定身份只能使用 owner 发布的 opaque family ID
- 当前 `temporal` owner 已补充 family competition memory：owner 内部会持续维护 `reuse_streak`、`stagnation_pressure`、`monopoly_pressure`、`competition_score`，并用这些竞争状态影响反塌缩的 topology maintenance
- 当前 public runtime state 只发布 compact family competition summary（如 active-family competition score、monopoly pressure、turnover health、family version/count），不发布 raw internal competition ledger；这条 bridge 为下一阶段 delayed credit ledger 预留了显式版本锚点
- 当前 `full-learned` 已把 `z_t` owner 更新规则收敛到显式 posterior + learned switch 路径：`z_t = beta_t * z_candidate + (1 - beta_t) * z_{t-1}`，其中 `z_candidate` 默认来自 posterior `z_tilde`，也可由 internal RL causal policy override
- **类型化边界请求（environment milestone → beta_t）**：`environment_milestone_temporal_switch=ACTIVE` 时，orchestrating session 把上一拍已提交、本拍待结算 outcome 上的 owner 声明（`EnvironmentMeasurement.discrete_milestone=True`）转发为方向无关、单拍有效的 typed 信号 `environment_milestone_boundary`；joint-loop 将其记录为 `record_external_boundary_request`，temporal owner 仍是唯一 `beta_t` 决策者，并相对自己当前学到的 `beta_threshold` 解析该请求。请求成立时 effective beta 至少达到当前 threshold，因此 SSL/reflection 对 threshold 的有界校准不能把已确认的里程碑边界永久屏蔽；无里程碑、SHADOW 或 DISABLED 均不得强制切段。请求是 turn-scoped：唯一写者是每 turn 的信号刷新，`reset_episode_runtime_telemetry` 明确**不**清除它——SSL 的 expert-action 族发现会在 full-cycle turn 决策前调用该重置，若在此清除会把学习 turn 上的已确认里程碑静默丢弃（session 级测试用第 3 turn 恰为 full-cycle 钉住此时序）。
- **post-switch minimum dwell（option commitment）**：`temporal_post_switch_min_dwell=ACTIVE` 时，FullLearned temporal owner 在一次 beta switch 后至少保留当前 option `temporal_post_switch_min_dwell_actions` 个 action（switch action 计作第 1 个）；窗口内只抑制自然 beta termination，新的 typed external boundary 仍可立即打断。这样 `steps_since_switch` 不会因连续超阈而永远自重置为 0，既有 active-family continuation signal 可以自举。SHADOW 只记录 `would_suppress / applied / remaining` owner evidence，DISABLED/0 是逐字节回滚。该机制只读取 beta、owner runtime state 与 typed boundary，不读取环境对象、文本、action-family 名称或 carrying 语义；Digital Ant profile 显式 ACTIVE/4，冻结 station1 checkpoint 预检把 8-lane survival 从 `[1,1,1,1,1,1,15,15]` 提升为 `[4,4,4,4,4,4,15,15]`，policy/temporal-learning fingerprint 保持稳定。正式因果结论仍须由全新 prereg/journal 给出。
  **PE 幅度对边界完全惰性（已由 v30 实测与测试双重证实）**：`prediction_error_temporal_switch` 只拥有**加性 prior**：`min(0.18, strength·tanh(max(0, magnitude − floor)))` 叠加进 switch pressure，对边界决策与 `is_switching` 没有任何决定权。曾经的"floor 交叉即 boundary request"语义在产生任何 journal 之前被 v30 冻结重放测量整体否定（日常拍 PE p50 0.508 与事件 PE 重叠、自然拾取下一拍 ~0.32 低于任何合理 floor，`scripts/measure_ant_pe_boundary_margin.py`；判词见 `research/ant/06_ecology_implementation_status.md`）——幅度阈值不构成事件检测器，这正是 R-PE"哪个事件关段应是类型化 readout"的运行时代价证据。证据：`packages/vz-temporal/tests/test_temporal_contracts.py::test_pe_magnitude_is_inert_for_boundaries_and_milestone_owns_them`（扫描 strength 与跨 floor 幅度：决策序列全同、无边界请求且 pressure 单调；typed milestone 信号在 ACTIVE 下首拍强制切段，SHADOW/DISABLED 不切）与 `test_external_boundary_request_crosses_learned_beta_threshold`（最大加性 pressure 0.18 也压不过 threshold 0.95；typed 请求则强制 effective beta ≥ threshold）。
- **reward→code 桥（runtime track modulation）**：此前 Internal-RL 只写 `track_weights`（+ `align_temporal_from_tracks` 投进 legacy 3-d `temporal_weights`/`switch_bias`），而 ndim `_step_impl_ndim` 产生 `code` 时不消费这些，导致奖励驱动学习与运行时 `z_t` **结构性脱钩**（只有 SSL 训练的 ndim encoder 能移动 `z_t`）。现新增 `MetacontrollerParameterStore.runtime_track_modulated_code`：以学习到的每-track 混合对 `z_candidate` 做**逐维、以 1.0 为中心、界于 [0.5,1.5]** 的增益调制（`gain_i = 1 + strength·(mean_i·n_z − 1)`）。由 `FinalRolloutConfig.internal_rl_runtime_modulation_strength`（默认 `0.0`）经 `FullLearnedTemporalPolicy.set_runtime_track_modulation` 注入。sandbox 的 `CausalZPolicy` 与 torch PPO 使用同一公式：先生成未调制 causal candidate，再以正在优化的 track 覆盖 store 中对应 track 后调用同一 aggregate gain；`step_with_causal_override` 接收的已是最终调制 candidate，因此 live forward 不再二次调制。这样 rollout reward、PPO surrogate 与真实 ndim 前向对 `track_weights` 的因果语义一致。`strength=0` 是**精确字节级 no-op（即时回滚基线）**，并保留历史 sandbox / torch PPO 方程；均匀混合在任意 strength 下亦为 no-op；`strength>0` 让 RL 学到的偏离-均匀方向真正进入 `code`。与 R2 不冲突（作用于控制器层，基底仍冻结），与 R8 不冲突（只读 owner 内部 `track_weights`，公共 snapshot schema 不变）。证据：`tests/test_temporal_contracts.py`（strength=0 时 mutate `track_weights` → `code` 不变；strength>0 时 skewed 混合 → `code` 变；sandbox mean 与 live helper 严格相等；causal override 不重复调制；torch PPO writeback 后真实 ndim `code` 改变；负 strength fail loudly）
- **生成期动态残差独立开关（State KV P5-a）**：`FinalRolloutConfig.generation_dynamic_residual` 默认 `ACTIVE`（字节级现状 / 回滚点），门控对话生成时 `control_code → runtime.generate(control_parameters, control_scale)` 这条 `z_t`→HF residual 通道。`SHADOW` 计算本应注入的 scale 并写入 rationale tag（`dynamic_residual=shadow:would_be:<scale>`）但不注入；`DISABLED` 在表达层丢弃控制参数，到达 substrate 的 kwargs 与"temporal 未产码"的 run 完全一致。每 turn 无条件发布 `dynamic_residual=<wiring>` attestation tag。该开关与 `personal_conditioning` 完全解耦（消融卫生：此前关闭 State KV 不会关闭这条第二潜通道，见 docs/specs/state-kv-identification-evidence.md 载体清单 C4）；它只作用于对话生成的表达路径，不影响 temporal owner 的 `z_t` 学习、capture 或 Internal RL replay。profile `dynamic-residual-off`（capability 同名）是显式消融臂，不进默认矩阵。
- **真实 runtime transition replay**：`FinalRolloutConfig.internal_rl_runtime_replay` 是独立于 pure/torch optimizer backend 的 transition-source gate，默认 `DISABLED`。Internal-RL owner 在动作拍保存真实 `SubstrateSnapshot`、world/self runtime state、实际 `z_t`、posterior、`beta_t`、行为 likelihood 与 PE `prediction_id`；下一拍只用匹配的既有 `EnvironmentOutcome`、`PredictionErrorSnapshot`、PE 派生 `CreditSnapshot` 和真实 next-substrate delta 结算 `runtime-replay` transition。`SHADOW` 只收集 lineage/coverage，不进入训练 staging；`ACTIVE` 只消费已结算真实 replay，样本不足显式报告 `waiting-for-runtime-replay`，禁止静默回退 synthetic。`FinalRolloutConfig.internal_rl_batch_accumulation_size` 默认 1；synthetic 按 rollout 数判断 batch，ACTIVE runtime replay 则按已结算 transition 数判断，使多个 singleton segment 不会各自进入 centered-gradient optimizer，一个已含足够 transition 的长段仍可立即训练。**已结算且达到 target 的 ACTIVE replay batch 本身就是 owner-side full-cycle 触发条件**：target=1 时也必须在下一次 scheduled step flush，不得等待无关的 PE spike 或周期性 RL tick；否则低 PE lane 会在跨 episode `include_runtime_replay=False` 导出时丢弃尚未优化的闭段，导致 action-head 永远保持 `update_step=0`。pure 与 torch PPO 均按保存的 ndim posterior/switch/modulation 上下文重建 old/new likelihood；capture/transition 还必须持久化实际 `posterior_sample_scale`：历史 encoder `z_tilde` 为 `0.5 × posterior_std`，显式 runtime exploration 为 `1.0 × posterior_std`，禁止用固定 `0.5` 重建 exploration action 的方差。旧 synthetic 公式保持兼容且同批禁止混源。pending capture 与已结算 staging 属于 joint-loop owner 的有界 checkpoint 状态，不新增 runtime slot；action-head ACTIVE/SHADOW 时 runtime state 额外发布正式 `causal_action_head_state`，并随 capture/transition 持久化。cycle rollback/no-optimize 只回滚 policy/critic，不重复消费环境证据。`create_learning_checkpoint(include_runtime_replay=False)` 是 owner 定义的跨 episode transfer 模式：保留 learned policy/temporal/memory state，但导出空 pending rollout 且不携带 replay report/capture，防止上个 episode 的 action 与新环境 outcome 错配；同 episode rollback/audit 仍使用默认 `True`。`joint_loop.learning` persistence schema 因此升级为 v4。
- **`beta_t` segment 长程信用**：`FinalRolloutConfig.internal_rl_runtime_segment_credit` 默认 `DISABLED`，通用 runtime 因而保持 one-step replay；Digital Ant profile 显式设为 `ACTIVE`。ACTIVE 时 joint-loop owner 只在内部维护 world/self 对齐的 open segment：遇到真实 switch 时先闭合旧段并从当前动作开启新段，milestone/terminal 或 `internal_rl_runtime_segment_max_steps` 也强制闭合。只有闭合段进入既有 pending rollout 队列，连续 `step_index` 让既有 GAE 可把末端价态传播给接近、绕行、逃逸和回巢的前序动作。open segment、closed count 与 longest length 随 joint-loop owner checkpoint/canonical archive/rollback；不新增第二 ledger 或公共 slot。`joint_loop.learning` persistence schema 因此升级为 v2。
- **状态条件 causal z-policy action head**：逐维 `track_weights` 无法表达左右感知变化对应的不同转向时，`MetacontrollerParameterStore` owner 可启用通用低秩 head：输入 `causal_action_head_state`，输出 bounded `z_t` mean residual；它不读取业务对象类型、不生成 motor command，只在 latent controller 空间参与 runtime replay PPO。该 state 由同一 Ndim encoder 参数对**当前 observation + 零 recurrent preimage + 无跨 turn CMS context**编码得到，覆盖完整 `n_input` 后落到 `n_z`，正式坐标范围为 signed `[-1,1]`；live forward 与 replay transition 必须消费并持久化完全相同的 owner-published state，禁止用随历史序列漂移的 serving `posterior_hidden_state` 训练、再用另一种特征执行。`internal_rl_causal_action_head` 默认 `DISABLED`，此时不执行第二次 encoder 且保持历史 forward；非 Full-Learned（如 ETA-off Learned-Lite）仍必须按其 `n_z` 发布等长零 state/residual 与 `disabled` wiring，不能用空 tuple 破坏 ACTIVE replay 的快照 shape。`SHADOW` 训练并发布 candidate state/residual 但不改变 live code，`ACTIVE` 才应用；`strength=0` 与 `DISABLED` 均为退出路径。head 前向与反向不得再次按 `[0,1]` 做 `2*h-1` 重心变换。runtime fragment 的末拍仅在真实 `runtime_terminal=True` 时使用零 continuation；非终局边界必须从已发布 next-substrate signature 做 TD bootstrap，禁止把每个真实 tick 错当终局而反转小幅正 payoff 的 advantage。低秩 action head 对 runtime advantage 做 RMS 归一化并保留 `0.05` 尺度下界、最终 clamp 到 `[-1,1]`；track/value 更新仍保留物理 payoff 原尺度。head 的常数截距继续使用 owner 学习率的 `0.12` 倍并再乘状态路径 `0.05`，单次不超过 `0.01` 且总幅度限制在 `[-0.1, 0.1]`；output/input factor 使用 owner 基础学习率，与 torch autograd path 的尺度一致，不再额外乘 `0.12` 或 `0.25`。每批 action gradient 的均值只进入该受限截距，output/input factor 只能消费去均值后的状态协方差信号，禁止利用 state 公共均值绕过 bias 上限。output factor 保持零初始化，避免未经经验的随机 state→action prior 直接进入 ACTIVE 前向；首个非零 covariance batch 在 owner 内执行有界 block-coordinate step：先由该批 covariance 得到 candidate output，再按 candidate 的真实列范数回传 input，最后原子提交两因子。部署 profile 仍必须提供多 transition batch，不能把 singleton 的严格零 covariance 误判为 factor 学习。state/residual 随 runtime capture 和 open segment 持久化，参数随 owner parameter snapshot、canonical archive、fingerprint 和事务 rollback；`joint_loop.learning` persistence schema 为 v3，不新增公共 slot。
- **action-head 形成期一致性保护**：`internal_rl_causal_action_head_formation_protection` 默认 `DISABLED`，且只有 `DISABLED / max_update_steps=0 / conflict_scale=1.0` 是历史精确回滚。非 DISABLED 必须搭配 ACTIVE action head、正整数窗口和 `(0,1)` 衰减系数，否则 fail loudly。窗口按当前 track 的 action-head `update_step` 判定；对每条 transition 计算 `contribution_i = normalized_advantage_i × projected_score_gradient_i`，若 `contribution_i · Σ_j contribution_j < 0`，则它是与本批净证据相反的少数冲突贡献。SHADOW 只在 `OptimizationReport` 发布 `window_active / would_attenuate_count / applied_count=0`；ACTIVE 才按 `conflict_scale` 缩放该 transition 的 action-head advantage，镜像 augmentation 两 lane 必须共享同一 scale。pure 与 torch 路径消费同一 per-transition scale；torch 必须只替换 head leaves 的 policy gradient，track、critic、value、entropy 仍使用原 PPO objective。批次净贡献精确为零时不设任意 tie-break；窗口结束自动 no-op。该机制不读取业务对象、body identity、food/carrying/phase、自然语言或 action-family 名称，不新增 checkpoint 状态或 runtime slot。
- **action-head rank 配置**：上述低秩是通用默认而非不可变 shape。`FinalRolloutConfig.internal_rl_causal_action_head_rank=None` 保持历史 rank；显式正整数在 owner 学习前选择 `min(requested, n_z)`。full-rank 初始化使用 identity input factors、全零 output/bias，保留 encoder 已发布的每个 state 轴，同时保持初始 live residual 严格为零。已有非零 output/bias 或 `update_step>0` 的 mapping 禁止原地改 rank，必须从新 checkpoint/schema 启动；将配置恢复为 `None` 即回到历史低秩初始化。Digital Ant `n_z=16` evidence profile 使用 full-rank，是因为 v16 实测 food 左右 state L1 已为 `0.80–0.87`，固定 rank-4 basis 只剩 `0.064–0.070`，而 heat basis 为 `0.331–0.364`，第二次压缩选择性丢失 food distinction。
- **action-head actuator support**：`FinalRolloutConfig.internal_rl_causal_action_head_effective_dims=None` 保持通用全 `z_t` 维历史行为；显式唯一、非空、界内坐标元组由 embodiment 的冻结 actuator 契约提供。pure optimizer 只在这些 output rows 构造 score-function gradient，torch path 以同一 mask 阻断非 actuator row 的 action-head gradient，live/sandbox residual 也把非支持坐标严格置零。该 mask 只约束状态条件 action head，不改变其余 metacontroller code、track/value 学习或冻结 plant；配置无效必须 fail loudly，恢复 `None` 是即时回滚。Digital Ant 的 `motor_decode` 只消费 `z[0:3]`，故 evidence profile 声明 `(0,1,2)`，避免 13 个行为无效维度用偶然 reward covariance 稀释 steering/speed 信用。
- **action-head actuator subspace**：`FinalRolloutConfig.internal_rl_causal_action_head_contrast_pairs=None` 保持历史独立坐标行为；显式 pair 必须是 disjoint、界内且完全包含于 `effective_dims`。temporal owner 对每个 pair 施加正交投影 `(x_i-x_j)/2, (x_j-x_i)/2`，pure replay 的 score gradient、torch autograd forward 与 live/SHADOW residual 使用同一变换，从而删除冻结执行器不消费的 common mode，而不把目标方向或业务语义写入控制器。Digital Ant 的 steering 只消费 `z[1]-z[0]`，故 profile 声明 `((0,1),)`；速度 `z[2]` 保持独立。恢复 `None` 即时回滚。
- **exclusive steering（contrast 轴所有权转移，R2）**：`FinalRolloutConfig.internal_rl_causal_action_head_exclusive_steering` 默认 `False`（精确回滚路径，字节不变）。`True` 时 temporal owner 对每个 contrast pair 施加 base 侧互补投影 `(x_i+x_j)/2`——**确定性 base policy mean**（posterior mean + track modulation）失去反对称分量，state-conditioned head 成为 contrast 轴上唯一的学习型写入者，base 保留 common mode（速度）。投影必须在四条路径上一致：live ndim forward（从调制后的确定性均值算修正 delta，加回含噪候选码）、sandbox `CausalZPolicy._policy_mean`（mean/std 分离，直接投影 mean）、pure `runtime_replay_policy_distribution`（modulated_mean 在 head residual 加法前投影）、torch PPO in-graph 三条 lane（autograd 因此把 contrast 轴信用全部路由给 head 参数）；pure `_trajectory_gradient` 的 runtime-replay 敏感度在 pair 维乘 0.5（pair 均值使本维权重影响减半）。**关键不变量：投影只作用于确定性均值，不得吞掉探索噪声的反对称分量**——否则 `(action-mean)` 在 contrast 维恒为 0，head 的 PPO 梯度永远为零，冷启策略也无法提出转向。要求非空 `contrast_pairs` 且 head wiring `ACTIVE`，否则 fail loudly（SHADOW/DISABLED head 下无人能转向冻结 plant）。causal latent override 跳过 live 侧均值投影（sandbox 构造 override 时已投影）。
  **β 门必须按 pair 共享**：opponent-coded pair 是**一根**执行器轴，必须整体切换。逐维门控会凭空造出 contrast——即使候选码已投影、上一拍码对称，`β_i·候选_i + (1-β_i)·旧码_i` 仍留下 `(β_i-β_j)·(候选-旧码)`。因此 exclusive steering 下 `effective_gate` 在每个 pair 上取共享均值（binary override 路径本就是标量门，天然满足）；override 路径同样适用，因为该不变量属于执行器轴语义而非候选来源。实测证据：修复前用**参数精确为零**的 head 仍能测到 ±0.005 rad 转向，与当时学到的 food 响应同量级，直接掩盖能力读数；修复后零参数 head 输出精确为 `0.000000`，同一 v23 checkpoint 的 near 绝对方向对齐从 0/4 变为 2/4、残余基线从 0.024 降到 0.008。动机：v22/v22r 两组受控实验（固定/随机 forced-approach 几何）证明信用竞争下无约束 base 总用"放大同向基线转向"的非定向退化解吸走转向信用（0.083→~0.147 rad），head 增益钉死 ≈1e-3 不增长。
- **action-head 镜像等变约束**：`internal_rl_causal_action_head_input_mirror_permutation/signs=None` 保持历史行为并作为即时回滚路径；embodiment 可在冻结 sense schema 上发布一个完整、带符号、二次作用为恒等的输入镜像，temporal owner 只校验/执行该代数变换，不解释 food、heat、home 或左右语义。启用时必须同时启用 ACTIVE head、非空 `contrast_pairs` 与 exclusive steering。owner 用相同 Ndim encoder、零 recurrent preimage 和相同 head 参数分别计算 `f(s)`、`f(mirror(s))`；每个 opponent-coded pair 只保留 `0.5·(f(s)-f(mirror(s)))`，未成 pair 的 actuator 维保留 `0.5·(f(s)+f(mirror(s)))`。由此，世界镜像时转向 pair 必须交换/翻号，速度等标量轴保持不变；head bias 与任意 state-path 镜像对称分量在 contrast 轴上按构造精确抵消。live forward、pure replay batch augmentation 和 torch in-graph forward 必须共享该投影；pure replay 对每个 transition 追加镜像 state + 输出镜像后的 gradient lane，torch 对两个 state 共用同一参数图。runtime state、capture、settled transition、open segment 与 canonical archive 同时持久化 `causal_action_head_mirror_state`，禁止 serving 现算镜像但 replay 仍训练单 lane。该 shape 变化把 `joint_loop.learning` persistence schema 升为 v5；v4 及更早带 replay 的 checkpoint 必须拒绝恢复。
- **action-head 更新包络的唯一 owner**：冻结的 bias/factor 纪律由 `temporal/interface.py` 的 `CausalActionHeadUpdateEnvelope` / `CAUSAL_ACTION_HEAD_UPDATE_ENVELOPE` 单点持有（`factor_absolute_limit=1.5`、`input_factor_step_limit=0.02`、`output_factor_step_limit=0.05`、`bias_absolute_limit=0.1`、`bias_step_limit=0.01`、`bias_learning_rate_ratio=0.12`、`bias_state_path_scale=0.05`）。pure owner 路径 `update_causal_action_head` 与 torch autograd lane 必须消费同一组常数，**禁止在第二个文件里复制这些数字**。torch lane 此前把 `head_input/head_output/head_bias` 当作普通 requires_grad 叶子塞进同一个 `torch.optim.Adam`，写回时逐字节还原 detached 张量，因而完全绕过上述纪律：实测同一 batch 单次更新在生产默认 `lr=0.02` 下把 bias 移动 `0.029`（冻结单步上界 `0.01` 的 2.9 倍），`lr=0.5` 下移动 `1.32`（绝对上界 `0.1` 的 13 倍）——这正是无约束截距安装跨状态固定转向、v10–v22 调试链一直追的失效模式。
  **包络是护栏，不是步长（W3-b 修正）**：只把 Adam 步长 clamp 回包络仍然是错的。Adam 把每个元素的步长归一化到约 `lr`，因此“先 Adam 再 clamp”是一个 bang-bang 最大步长控制器——更新只携带梯度的**符号**。实测：同一 batch 单次更新在 `lr=0.02` 与 `lr=0.5` 下给出**完全相同**的 `bias_step=0.010000 / out_step=0.050000 / in_step=0.020000`（三个单步上界精确顶满），把梯度缩小 1000 倍数值仍不变；生产默认 `lr=0.02` 连续 15 次更新在第 10 次就把 bias 走到绝对上界 `0.1` 并钉死。**钉死的 bias 就是跨状态固定转向截距本身**（`docs/specs/digital-ant-embodiment.md` 冻结这些数字正是为此），即包络要阻止的失效模式被包络自己制造了出来。
  因此步长尺度也收敛为唯一 owner：`causal_action_head_update_scales(learning_rate=, batch_size=)` 返回 `factor_learning_rate = learning_rate / batch_size`、`bias_learning_rate = factor_learning_rate * bias_learning_rate_ratio`、`bias_signal_learning_rate = bias_learning_rate * bias_state_path_scale`。pure 路径消费前两项（保留历史乘法顺序，逐字节不变）；torch lane 的 head 三组参数**不再是 Adam 参数**（`w/log_std/cw/cb` 仍走 Adam，Adam 是逐元素的，移出 head 不改变其余参数的算术），改为在 `loss.backward()` 后按同一尺度做比例梯度步，再调用 `project_causal_action_head_update(baseline=本次调用入口参数, candidate=当前叶子)` 投影。head 叶子不在 optimizer 内，故每个 epoch 前显式清零其 `.grad`；`parameters_changed / parameter_change_rate` 的统计范围仍覆盖 head 叶子，语义不变。单次调用写回时 `update_step` 只 +1，因此整次调用就是一次 owner 更新；逐 step 投影同时保证后续 PPO epoch 的前向也落在包络内。写回前再投影一次（覆盖 `ppo_epochs==0`）并显式 `enforce_envelope=True`，任何逃逸 fail loudly 而不是静默落盘。
  **单步位移上界是有条件的，绝对上界才是无条件的**（勿把二者混为一谈）：`project_causal_action_head_update` 计算 `clamp_absolute(clamp_step(candidate))`，因此 (1) `|result| <= absolute_limit` 无条件成立；(2) baseline 已在包络内时 `result` 落在 `[baseline - step_limit, baseline + step_limit]`——该区间按 binary64 求值，`0.05 + 0.01` 的可表示值是 `0.060000000000000005`，所以精确算术下的 `|result - baseline| <= step_limit` 会以 1 ulp 之差为假，不变量属于区间而不属于差的绝对值；(3) baseline 已在包络**外**且距离超过一个 step 时，step 区间与绝对区间不相交，绝对 clamp 获胜，单次调用位移为 `|baseline| - absolute_limit`，**可以远大于** `step_limit`（实测：permissive 路径装入的 `bias=0.35` 在 `bias_step_limit=0.01` 下一次即被拉回 `0.1`，位移 25 倍上界）。方向恒为朝冻结区收敛（`|result| = absolute_limit < |baseline|`，冻结包络下符号不翻转），pure owner 历来如此，属按设计行为，故记述向行为对齐而非反过来改行为。证据：`tests/test_temporal_contracts.py::test_envelope_projection_step_bound_is_conditional_on_the_baseline`。
  实测（`packages/vz-temporal/tests/test_temporal_contracts.py::_torch_head_step`，`n_z=16`、batch 6、`ppo_epochs=1`、生产默认 `lr=0.02`）：普通 batch 下 `bias_step=1.2305e-5`（上界 `0.01` 的 1/813）、`out_step=5.6579e-4`（1/88）、`in_step=8.3640e-5`（1/239）；把动作与重建均值的距离放大一倍（gain 20→40），三项步长同步放大 `1.991x / 1.997x / 2.003x`；`lr` 从 `0.02` 提到 `0.5`（25 倍），三项步长精确放大 `25.000000x`（修复前是 `1.000000x`）；梯度足够大时（action gain 1000）三项仍精确顶在 `0.01 / 0.05 / 0.02`，再放大 5 倍不再增加。连续 15 次更新后 `max|bias|` 只走到 `0.000302`（轨迹 `0.000014, 0.000075, ... 0.000278, 0.000302`），而非在第 10 次钉死于 `0.1`。**已知残余差异（未收敛）**：pure 路径还额外保证“batch mean 只更新 bias、factor 只消费 centered covariance、block-coordinate 先 output 后 input”，torch lane 的 autograd 梯度不做去均值分解；投影只约束幅度，不约束该分解。torch lane 在 `output_factors` 全零时首个 epoch 也拿不到 input 侧梯度（pure 路径用 block-coordinate 步专门处理该冷启），本包未收敛。
- **action-head 幅度校验（`causal_action_head_envelope_enforced`，默认 `False`）**：`restore_causal_action_head_parameters` 历史上只校验 rank/width 五项 shape，任何 archive（包括无约束 torch lane 产出的、或被手工编辑/损坏的）都能装入违反冻结包络的参数。现新增 `validate_causal_action_head_magnitudes`，越界时 fail loudly 并在消息里点名被违反的具体上界（`factor_absolute_limit` / `bias_absolute_limit`）。`MetacontrollerParameterStore(causal_action_head_envelope_enforced=...)` 是显式契约字段，默认 `False` **精确保留历史 restore 行为**（只校验 shape），`True` 时 `restore_causal_action_head_parameters` 与 `restore_parameter_snapshot` 的 archive/checkpoint 路径同时执行幅度校验；恢复为 `False` 即精确回滚。owner 自己的受纪律写入路径（pure 更新、torch 投影后写回）与该开关无关，始终按 `True` 校验，因此越权 archive 无法借它们落地。
  **校验不带任何容差**：冻结上界上不加 epsilon。owner clamp 的产物天然合法——`max(-0.1, min(0.1, x))` 精确返回 `0.1`，而 `abs(0.1) > 0.1` 为 `False`；`1e-9` 的 slack 只会悄悄把冻结上界放宽（`math.nextafter(0.1, inf)` 与 `0.1` 之差远小于 `1e-9`，会被吞掉）。
  **非有限值一律 fail loudly，且不受该开关支配**：(1) `_envelope_bounded_value` 此前只检查 candidate；非有限 **baseline** 会让 `max(nan-step, min(nan+step, candidate))` 得到 `nan`，再被绝对 clamp 静默压成 `+absolute_limit`——一个损坏基线因此被洗成“合法的、钉在绝对上界上的截距”（实测 `nan`/`inf` baseline 都得到 `0.1`）。现 baseline 与 candidate 同样校验。(2) `restore_causal_action_head_parameters` 补齐 `restore_parameter_snapshot` 早就有的有限性校验，NaN/Inf 无法直接写进 live head。
  **开关的可达路径（opt-in，已贯通到生产配置）**：`MetacontrollerParameterStore.set_causal_action_head_envelope_enforced(bool)` 是构造后的显式切换点；`FinalRolloutConfig.internal_rl_causal_action_head_envelope_enforced: bool = False`（`packages/vz-runtime/src/volvence_zero/integration/final_wiring.py`，与 `internal_rl_causal_action_head_rank` 等字段同处）是 domain 唯一的声明入口，经四个 `set_causal_action_head(...)` 调用点落到 store：`final_wiring.py` 的 WORLD / SELF 两处、`packages/vz-runtime/src/volvence_zero/agent/session.py` 的 WORLD / SELF 两处。**两轨都必须收到**，否则一条 lane 仍可被装入越界 archive。通用默认 `False` 精确保留历史 restore 行为（`tests/test_runtime_transition_replay.py` 刻意从 permissive 路径装入 bias `0.35 / 0.4`），因此该契约必须保持 opt-in。
  **初始化越界阻塞项已解除（2026-07-27，修构造器而非收窄校验）**：此前 `_initial_causal_action_head_parameters` 在 `rank < n_z` 时回落到 `_random_mat(rank, n_z)`，即 scale `1/sqrt(rank)` 的**无界高斯**，与 `factor_absolute_limit=1.5` 毫无关联。种子固定，违例是确定性的；实测（`n_z=16`，即数字蚂蚁正式配置）：`world rank=4 max|input_factor|=1.0749` 合规，`self=1.9823`、`shared=1.6192` **越界**（`n_z=4` 时 `self=1.5976` 越界）。而 `internal_rl_causal_action_head_rank` 只配置传给 `set_causal_action_head` 的那**一个** track，`restore_parameter_snapshot` 却校验全部三个 track，于是未被配置的 SELF/SHARED 头保留随机低秩初值——任何域一旦声明 `True`，第一次 checkpoint 恢复就会在 owner 刚刚自己创建的、`update_step=0` 且 output/bias 全零（live steering 权限精确为零）的头上抛错（实测蚂蚁包 19 个测试失败）。
  **为什么修构造器而不是给 `update_step == 0` 开豁免**：`factor_absolute_limit` 被 owner 写入路径**无条件**施加——pure `update_causal_action_head` 的内联 clamp 与 `project_causal_action_head_update` 的 `clamp_absolute`（该函数不变量 1）都不看 `update_step`，因此 owner 写入路径的像**恒**落在包络内，第一步也不例外。若给初始先验放宽上界，validator 接受的集合就严格大于 owner 自身的像；而 `update_step` 只是 restore 快照里的一个普通字段，越权/损坏 archive 只需声明 `update_step=0` 即可装入任意 input factor——豁免在任何写法下都是宽度等于自身的旁路，正是"跨状态固定转向截距"这条立论要挡的东西。故 gate 保持 `update_step` 盲，由 `test_envelope_has_no_update_step_dependent_exemption` 双向锚定。
  **修法与差分证据**：新增 `_bounded_initial_input_factors`，对随机抽样施加**单个全局正标量**重标定（`limit / peak`，仅在越界时生效），方向结构（相对幅度、符号、行列几何、秩）原样保留，不是逐元素 clip。`_random_mat` 本身**不动**（encoder/decoder 权重 seed 100/101/103/104 仍在消费它，逐字节不变）。全网格差分（`n_z <= 64` × 全部 `rank` × 三轨，共 6240 个配置，`struct.pack` 后按 sha256 比对）：**5573 个逐字节不变，667 个改变，且改变的 667 个恰好等于原先越界的 667 个**——没有任何一个原先合法的配置被移动，数字蚂蚁自己的 WORLD 轨（1.0749）在不变的那一侧。附带效果是把 basis 推离 tanh 饱和区（`n_z=64, rank=1` 的最坏 pre-activation `7.552 -> 2.857`，`tanh` 从 `1.00000` 退回 `0.99342`，即该 basis 坐标从零梯度死区里回来）。证据：`tests/test_temporal_contracts.py::test_pristine_head_initializer_respects_the_envelope_it_is_validated_against`（全网格逐个过 validator）、`::test_bounded_initial_input_factors_rescales_without_reshaping`（合规抽样逐字节原样返回 + 越界抽样保持共同比例）。数字蚂蚁 profile 因此在同一批翻成 `True`。
  **声明的优先级：`None` 表示“本次不声明”**。`set_causal_action_head(..., envelope_enforced: bool | None = None)` 只在显式传 `True`/`False` 时写入 store。此前它把 `False` 默认值无条件写入，于是一个用 `MetacontrollerParameterStore(causal_action_head_envelope_enforced=True)` 构造出来的 store，会被任何一个根本没提到该开关的后续 head 配置调用（改 rank、改 strength、配第二轨）静默关掉强制校验——AGENTS.md §6 明令禁止的静默降级。`FinalRolloutConfig` 走的是显式路径，因此 rollout 配置仍是 domain 声明的权威边界。证据：`tests/test_temporal_contracts.py::test_head_configuration_without_a_declaration_leaves_the_store_alone`、`::test_rollout_config_carries_the_envelope_declaration_to_the_store`、`::test_digital_ant_profile_enforces_archive_envelope`。
- **latent code 值域的唯一 owner**：`temporal/interface.py` 的 `LATENT_CODE_BOUNDS = (0.0, 1.0)` 是 live ndim forward 对 `code` 的冻结值域（`_clamp` 由它派生）。torch runtime-replay lane 此前硬编码三处 `[-1,1]`，而同一函数的两条 synthetic lane 用 `[0,1]`；在饱和 contrast 轴上重建出的均值会离开冻结 plant 的值域，使 `(action - mean)` 相对真实动作反号、head 梯度指向错误方向。现该 lane 经 `resolve_latent_code_bounds(latent_unit_clamp=...)` 选择边界，与 pure lane 的 `latent_unit_clamp` 契约同名同义：`False` 保持历史 signed 边界（精确回滚基线），`True` 在 owner 的单位值域上重建。reward/advantage 的 `[-1,1]` clamp 是另一套约定，两条 lane 都不变。**pure 与 torch 两条 runtime-replay lane 必须收到同一个 `latent_unit_clamp`**，否则同一 batch 会重建出不同均值；这正是 `assert_runtime_replay_latent_bounds_agree` 所执行的范围。
  **该不变量精确限定在 runtime-replay 重建上，不覆盖 synthetic 分支**（此前这句话没有限定作用域，读起来像是覆盖全函数）：torch 的两条 synthetic lane 历来就在 owner 的 `LATENT_CODE_BOUNDS` 上做 clamp，与 `latent_unit_clamp` 无关；pure 的 synthetic lane 则按 `latent_unit_clamp` 选择 `_clamp`（signed，历史默认）或 `_clamp_unit`（= `LATENT_CODE_BOUNDS`）。因此：声明 `True`（数字蚂蚁 evidence profile）时四条 lane 全部落在 `LATENT_CODE_BOUNDS` 上；保持历史默认 `False` 时两条 synthetic lane 的边界不一致，**这一不对称本身就是精确回滚基线**，不是新缺陷。把 torch synthetic 也接到 `resolve_latent_code_bounds` 是错误修法：默认下它会把该 lane 从冻结 plant 的值域上**移走**（改成 signed），实测使 818 行字节探针中 88 行改变，直接破坏通用默认的字节等价。本包只把该 lane 里写死的 `0.0, 1.0` 字面量换成 owner 常量 `LATENT_CODE_BOUNDS`（算术逐字节不变），以满足“值域只有一个 owner、禁止在第二个文件里写死边界”的约束。证据：`tests/test_temporal_contracts.py::test_torch_synthetic_lane_reconstructs_on_the_owner_latent_range`（差分锚定：两组令未 clamp 均值都落到地板以下但数值不同的 track weights 必须给出同一 surrogate；signed 地板下则不会）。
- **ndim drift 语义**：`n_z>3` 的 serving path 只消费 Ndim encoder/switch/decoder 与正式 `track_weights` modulation；Internal-RL 对 track 的更新不得再同步写入仅供 legacy 3-D controller 使用的 `temporal_weights/switch_bias`。否则 `switch_bias=1-persistence` 的无效兼容字段跳变会被 rollback gate 当成真实 metacontroller drift，回滚同批 action-head 更新。dual-track aggregate 的 `track_parameters` 必须发布两个 owner 的正式 track weights（shared 取两 owner 均值），禁止用逐拍 `latent_mean` 冒充参数；state variation 与 parameter drift 必须隔离。legacy `n_z<=3` 的 alignment 行为保持不变。
- **有界 posterior 探索**：`FinalRolloutConfig.internal_rl_runtime_exploration_strength` 默认 `0.0`（精确回滚基线），`(0,1]` 时由 `FullLearnedTemporalPolicy` 把原 sample noise 与可复现 low-discrepancy sample 混合，并设置有界 latent entropy floor（`0.4 * strength`），防止获得首个稀疏 milestone 前 posterior 方差塌缩。采样按 8-step coherent option 分段：sample residual 只由 owner-local `segment + latent dimension` 决定并在整段保持；posterior mean/std 仍逐拍消费当前状态，因此 option 连续性不得抹平 state-conditioned policy。调用方可以附加不透明、非语义的 exploration context；temporal owner 只保存其 SHA256 摘要并把摘要纳入 option identity，禁止保留原文或解释业务含义。未提供 context 时必须保持历史全局序列精确不变；matched arms 必须共享 context，而独立 episode/body 应使用不同 context，避免同一噪声序列被重复奖励成全局固定转向。禁止把连续变化的 posterior mean 写入 option identity，也禁止在 coast 阶段把逐维 residual 换成 common-mode residual；前者令同一 option 每拍跳变，后者会撤销 opponent-coded latent steering。该 horizon 保证 16/24-step 的有界稀疏实验至少覆盖 2/3 个方向 option，而不是把整个 episode 退化成一条射线。该机制是通用 temporal option 探索，不读取任务真值或动作语义。有效 posterior mean/std、最终 sample noise 与 `z_tilde` 均由 owner runtime state 发布，runtime replay likelihood 可精确重建。实验对照各臂必须共享该值，探索本身不算学习收益；已有 dense PE 的任务应保持 `0.0`。
- **真实 no-optimize 对照**：`apply_writeback` 只负责 reflection/memory/regime consolidation，不能冒充 Internal-RL 消融。此前 `no_optimize` 仅设 `joint_apply_writeback=False`，但 `run_cycle()` 已在该门之前直接执行 sandbox optimizer，因此 learned/no-optimize 都会更新 PPO，因果对照无效。现新增独立 `apply_policy_optimization` gate：两臂运行相同 SSL、substrate rollout、PE、optimizer 与报告；gate=False 时在 SSL 后/RL 前建立 owner checkpoint，并在 optimizer 后恢复 policy+critic，使 RL 更新不持久化，同时保留 SSL 更新与完整候选证据。默认 True 不改变生产行为。数字蚂蚁正式 no-optimize 令此 gate=False、reflection writeback 与 learned 保持一致，仅隔离 Internal-RL policy optimization。`CausalPolicyCheckpoint.policy_optimization_fingerprint` 由 Internal-RL owner 仅基于 update step 与 critic 参数生成；共享 reflection prior 可以合法改变 temporal track weights，但不会因此把 no-optimize 误判为 RL 写回。
- **七天 no-SSL 与 M3 slow matched control（2026-08-02）**：`apply_ssl_optimization=False`
  仍对同一 frozen trace 执行 SSL optimizer 并发布 loss/M3 candidate readout，但随后恢复 cycle
  起点的 world/self owner checkpoint，再进入相同 RL rollout；因此它隔离的是 SSL 持久写入，
  不是通过跳过 schedule 制造不同样本。默认 `True` 不改变产品行为。`temporal_ssl_m3_slow_gain`
  由 `FinalRolloutConfig` 注入同一个 `MetacontrollerSSLTrainer` 的 encoder/decoder M3 optimizer，
  合法范围 `[0,1]`，默认 `0.0`；七天 Gate 9 用 `1.0/0.0` matched profiles，并经
  `SSLTrainingReport.encoder_optimizer_state.slow_gain` 与 `m3_slow_momentum_signal` 证明变量真实
  进入 owner。两项开关都是 evidence process-start 配置，不能由 evaluation、prompt 或 turn
  文本动态修改。
- **ETA-off 维度契约**：`LearnedLiteTemporalPolicy` 的 code 维度必须等于其 `MetacontrollerParameterStore.n_z`；legacy 三维 readout 在 `n_z>3` 时由 temporal owner 确定性投影到配置维度。正式 ETA-off 因此与其他 arm 共用 ACTIVE runtime replay 的严格维度检查，不靠关闭 replay 隐藏 schema 破裂。
- 当前 decoder 已升级为 bounded FFN-like control generator；环境侧显式区分 `decoder_output`、`applied_control`、`downstream_effect`
- 当前 SSL trainer 已改成更接近 Eq.3 的结构：prefix posterior inference + Gaussian-like prior regularization + action prediction + closed-form KL，并发布 posterior drift
- **steered-action Eq.3 模式（2026-08-01）**：`StoreSSLTrainingSession` 接受可选 `action_scorer`（`vz-substrate::TransformersSteeredActionScorer`）。启用时 distortion 是穿过被控制冻结模型的专家动作 NLL（受限动作词表 softmax，与 `n_z` 解耦），`z_tilde` 走 seeded `torch.Generator` 真实重参数化，beta gate 用连续 `effective_gate`，损失只有 `distortion + alpha * KL` 两项；传入非零 switch 正则权重或 `write_back=True` 均 fail loudly。同时 `torch_store_ssl` 与 `ssl.py` 的 `switch_rate/binary/group/gate_choice` 默认权重已从非零改为 `0.0`——旧非零值是对 never-switch 塌缩的症状压制，且 2026-08-01 rate-distortion 判据显示该塌缩本身就是诊断读数。需要旧行为的对照臂必须显式传入非零权重。joint 有效性对照经 `build_steered_action_scorer(joint_training=True)` 解冻注入层之上的块（含 final norm），跑完必须 `restore_and_freeze()`；substrate 参数以独立 `substrate_learning_rate` 参与 optimizer 参数组。运行时构造新增 `model_dtype` 显式覆盖（`float16/float32/bfloat16/None`），rate-distortion 判据强制 fp32。
- 当前两阶段 ETA owner 约束已从 telemetry 收紧为**运行时守卫**：`MetacontrollerSSLTrainer.optimize()` 只能在 `ssl*` discovery phase 下运行；causal override / internal RL rollout / optimize 只能在 `runtime` / `rl*` 等 structure-frozen takeover phase 下运行。session 的 `joint_learning_enabled=False` 必须同时进入 joint-loop 与 temporal policy/store owner：joint-loop 返回 `frozen-evidence-only`，即使恢复的 checkpoint 带有 pending batch、PE pressure 或 continuation due，也不得触发 SSL、Internal RL、writeback 或 rare-heavy 推荐；temporal owner 禁止 `fit_from_signals`、action-family history/topology/cache 与 learned family-match 权重写入，只允许更新不持久化的推理 telemetry。该 gate 是 runner mode，不进入 checkpoint；冻结 held-out 以 `learning_parameter_fingerprint` 前后严格相等为证据。
- causal takeover 后的 `structure_frozen=True` 同时冻结 action-family topology：runtime 仍可更新 reuse / competition 等观测 telemetry，但 anti-collapse split/create/merge/prune 只能在显式允许 topology maintenance 的非冻结 SSL phase 执行。RL 期间 family identity、centroid 与 discovery confidence 的结构指纹必须保持不变；若发生拓扑漂移，整轮证据 invalid 并由 whole-cycle checkpoint 回滚。
- 当前 env 已新增 owner-side residual intervention backend，用 `e_{t,l} ← e_{t,l} + U_t · e_{t,l}` 形式的近似 hook 生成 `downstream_effect`；session / joint-loop 主链默认优先走 open-weight residual runtime，trace backend 退为 fallback
- 当前 internal RL sandbox 已支持 `baseline / causal / causal-binary` 三条 rollout 路径；`causal-binary` 会在 replacement 路径上对 `beta_t` 做 Heaviside-like 二值化，更接近 ETA B.5
- 当前 `TemporalModule` 已直接消费 `prediction_error` slot；高 PE 不再只经 evaluation 旁路感知，而是直接进入 owner-side update / scheduling surface
- 当前默认主链已拆成 staged temporal surfaces：`TrackTemporalModule` 先以 `substrate + memory` 产出 same-wave early control；`TrackTemporalConsolidationModule` 再以 `reflection + prediction_error` 做 owner-side late consolidation；公共 `temporal_abstraction` 由 `TemporalAggregateModule` 聚合 `world_temporal` / `self_temporal` 后发布，避免靠共享可变状态偷渡 same-wave 顺序
- 当前 `TrackTemporalConsolidationModule` 已开始直接消费 `credit` 快照中的 abstract-action / session-level evidence，把 delayed credit 写回 action-family 的 `outcome_driven_score`、`long_term_payoff`、`delayed_credit_sum`；这条路径不引入新的 owner，而是在 temporal owner 内完成 family competition 的结果驱动更新
- 当前 live dual-track path 已进一步收敛：track policy 会缓存 consolidation 阶段观察到的 `reflection` 证据，并在后续 early-control `step()` 中作为 owner-side context 参与切换/编码；这让 public dual-track path 不再系统性丢失 reflection
- 当前 heuristic / learned-lite fallback 的公开 action label 已降级为更中性的 latent-family 风格标签，避免 benchmark 仅靠手工语义名就显得“像 ETA”
- 当前 proof harness 允许把这两层 scaffold 分别关掉：`pe-eta-no-semantic-label` 保留 latent/full path 但剥离语义标签扶手；`pe-eta-no-reflection-cache` 保留 latent/full path 但禁用 cached reflection bridge，用于检查 family 与 PE schedule 的稳健性
- 当前 stronger proof matrix 已把 `pe-eta-no-semantic-label`、`pe-eta-no-reflection-cache` 与 `pe-eta-pe-readout-only` 提升为正式 proof-oriented profiles：它们不替代轻量默认 ablation，但用于更直接地区分“latent mechanism 仍成立”与“只是 scaffold 或 PE readout 在撑”
- 当前 `AgentSessionRunner`、`ETANLJointLoop` 和 final wiring 的默认 self-track controller 不再从独立随机 policy 起步；若用户未显式传入 self-track policy，则默认从 world-track discovered metacontroller snapshot 克隆，保证默认 runtime 的双轨 temporal owner 共享同一条 discovered lineage，同时仍保留两个独立 owner/store
- **双轨 store 分离是持久化不变量，不只是初始化偏好**：`ETANLJointLoop` 的每个 learning checkpoint 都把 world/self 作为两条独立 lane 发布（`world_temporal_snapshot` / `self_temporal_snapshot` 加两条 sandbox `metacontroller_snapshot`），而 `rollback_rare_heavy_import` 逐条恢复。因此两轨**禁止**共享同一个 `MetacontrollerParameterStore`：共享时先恢复的 world lane 会被随后恢复的 self lane 原地覆盖，checkpoint 不再 round-trip，`AgentSessionRunner.restore_learning_checkpoint` 的 re-export fingerprint 守卫会（正确地）报 mismatch。owner 在构造时对共享 store 直接 fail loudly，不做静默克隆修复——静默克隆会丢掉调用方已施加在那个唯一 policy 对象上的 per-track runtime 配置。需要第二轨的调用方必须用 `volvence_zero.temporal.clone_temporal_policy()` 克隆；它保留源实现模式（`LEARNED_LITE` 不被提升为 `FULL_LEARNED`），使 matched-control 的 ETA-off 臂两轨都保持 ETA-off。此前 `AgentSessionRunner` 对非 `FULL_LEARNED` 的单 policy 入参回退为两轨别名同一对象，正是这条不对称的来源
- 当前 `internal_rl` 已新增 paper-like proof mode：`InternalRLProofEpisode / InternalRLProofSubgoal / InternalRLDelayedCreditAssignment` 允许用结构化分层 sparse-reward episode 驱动 rollout，而不改变 live session 默认 dense reward 语义
- 当前 `InternalRLSandbox` 在 proof mode 下已支持 sequence-aware causal observation（优先读 `residual_sequence` 摘要）与 delayed-return optimization；这条路径用于验证 internal RL 是否真的在抽象动作层解决延迟奖励，而不是只在 turn 级 dense shaping 下工作
- 当前 ETA proof harness 已新增 real open-weight evidence lane：`transformers-open-weight` backend 通过 `OpenWeightResidualRuntime.capture/apply_control` 生成真实 residual `SubstrateSnapshot` 序列；time step 暂定为 source prefix steps，快照内显式发布 `eta_real_runtime_step_index`、`eta_real_runtime_capture_present`、`eta_real_runtime_fallback_active`、`eta_real_runtime_intervention_protocol_valid` 与 runtime hook evidence。`trace` / `synthetic-open-weight` 仍保留为 matched fallback/control，不再承担真实 residual-control claim 的唯一证据面；real claim 现在要求 fallback rate 为 `0.0`、actual hook fire rate 至少 `0.75`、prefix capture 与 intervention source 对齐。`planned_layer_fraction` 只是选层比例诊断，不再被误当作 hook 健康度
- 当前 real open-weight proof path 会用 frozen real residual prefix captures 校准 proof subgoal signatures，并把 observation alignment / intervention effect 作为 diagnostic reward components 发布；这些 diagnostic components 不进入 sparse optimizer-visible reward
- 当前 `run_eta_open_weight_residual_benchmark()` 与 `build_eta_open_weight_paper_suite_manifest()` 提供从 proof harness 向 open-weight runtime 过渡的研究原型入口：它不改变 live session 默认 dense reward 语义，也不打开 live substrate mutation；默认仍遵守 frozen substrate doctrine
- 当前 ETA strong-proof benchmark 默认不再只比较混合 baseline，而是使用 matched controls（`full-no-optimize` / `full-no-replacement` / `learned-lite-causal` / `noop-backend`）分别隔离 RL 更新、latent replacement、controller capacity 与 backend intervention effect
- 当前 `InternalRLSandbox.optimize()` 已补充 parameter-change evidence：proof benchmark 会记录 training-time `parameters_changed` / `training_parameter_change_rate`，避免把“最终 success 更高”误写成“internal RL 确实发生了 policy adaptation”
- 当前 ETA strong-proof benchmark 已从 dialogue PE harness 中显式分离：新的 proof harness 关注 hierarchical sparse-reward、abstract-action family reuse、held-out composition 与 delayed credit alignment，而不把这些结论混写进普通 `temporal_abstraction` runtime slot
- 当前 causal z-policy 已不再只发布 proxy score：runtime rollout 现在会显式记录 `policy_mean` / `policy_std` / `policy_noise` / `log_prob` / `value_estimate`
- 当前 `CausalZPolicy` 已从单条 rollout 更新扩到 batch rollout 更新；PPO-like surrogate、KL 与 clip 现在围绕显式 stochastic z-policy 分布计算，而不是只围绕 synthetic score。探索噪声由 owner-local seeded PRNG 逐动作采样，不再用 observation/step 的 `math.sin` 确定性投影伪造；seed 使 matched arms 可复现，world/self 默认分别为 0/1，完整 RNG state 随 `CausalPolicyCheckpoint` export/restore，保证 rollback 后的下一抽样精确重放。
- 当前 internal RL 已新增最小 critic 路径：每个 transition 会记录 `return_estimate` / `advantage_estimate`，proof/dense 两条 reward path 共用同一套 return bookkeeping
- 当前 observation side 也已从极简 surface 压缩升级到 richer prefix signature：默认同时吸收 averaged / peaked / trended / persistence-style 证据，再投影到 `n_z`
- 后续可平滑替换为 learned-lite 或 full learned policy，而不改变 snapshot schema
- SYS-1 最小切片新增 `CPDSwitchReadout`，由 `TrackTemporalConsolidationModule` 在消费 `prediction_error` 时发布到 `TemporalConsolidationSnapshot.cpd_switch_readout`。该 readout 只根据 typed PE 数值计算 `pe_spike_score` / `reward_shift_score` / `switch_recommended`，不从文本或关键词推断，不新增第二个 `beta_t` owner，也不直接改变 live `beta_t`。后续 `cpd-beta-switch` SHADOW profile 才能把它接成 switch-pressure evidence。

### NL/ETA 全量真 autograd 迁移（Phase 0–3，full-autograd target line）

目标线已从“纯 Python 有界近似 + 论文语义补齐”提升为**全面真 autograd（torch），含 runtime metacontroller**。纯 Python tuple 数学降级为**回滚基线**，torch 路径通过 `WiringLevel`（`DISABLED -> SHADOW -> ACTIVE`）逐 owner 推进，每步有 parity 证据，可随时回退。与 R2 不冲突：autograd 作用于 metacontroller 控制器层，基础 LLM substrate 仍冻结；与 R8 不冲突：torch 张量只活在 owner 内部，发布前转回 float tuple，公共 `temporal_abstraction` snapshot schema 不变。

- **Phase 0（基石）**：`volvence_zero.tensor_backend`（位于 vz-contracts，零上游，vz-temporal/vz-memory 共享）提供 `TensorBackend` 抽象 —— `PurePythonBackend`（回滚基线 / DISABLED）与 `TorchBackend`（真 autograd / ACTIVE）。`tensor_backend_parity` 提供 GRU/FFN 前向 parity harness；float64 下 pure↔torch 逐字段一致（容差 1e-9）。`configure_determinism()` 固定 seed + deterministic 算法，保证 SHADOW 双跑可比对。torch 缺失时 `resolve_backend` 显式回落 pure（named reason，不静默吞错）。
- **Phase 1（offline SSL 真 autograd）**：`volvence_zero.temporal.torch_metacontroller` 用真 backprop 训练 GRU encoder + switch + decoder。**Eq.3 对齐**：默认 KL 切到 `D_KL(q ‖ N(0,I))`（`KLTarget.STANDARD_NORMAL`），保留 learned-prior 为 CMS-enhanced 变体（appendix C.2）；action loss 用 MSE。**STE 切换**：前向二值、反向直通，替换固定 0.55 阈值，切换稀疏由 `alpha · D_KL` 涌现驱动。`compare_kl_targets` 是 matched ablation，验证 standard-normal KL 随 alpha 增大而压缩（变分瓶颈签名），且 alpha 对切换有因果影响、两种 KL target 不塌缩为同一切换行为。参数快照经 `TorchMetacontrollerArtifact`（仅 float）走 rare-heavy 路径导出/导入。
- **Phase 2（offline Internal RL 真 autograd）**：`volvence_zero.internal_rl.torch_internal_rl` 把因果策略 `pi(z_t | e_{1:t})` 与 critic 实现为真 torch 模块，PPO（GAE + clipped surrogate + entropy + value regression）全程 autograd（替换 `math.sin` 伪随机与解析步）。环境是分层 sparse/delayed-reward proof episode（reward 只在 terminal 交付）。matched control（`no-optimize`）不更新、不提升；full 提升 terminal return 并击败 control。
- **Phase 3（runtime metacontroller SHADOW -> ACTIVE）**：`volvence_zero.temporal.backend_metacontroller` 用 backend-agnostic 前向（同权重可在 pure/torch 上跑），`shadow_dual_run` 逐字段比对 z_t/beta/control 并量延迟；`promotable = within_tolerance and latency_ok`。`resolve_runtime_backend(WiringLevel)` 路由：DISABLED/SHADOW 走 pure（torch 并行只在 shadow 比对，不上 live 路径），ACTIVE 走 torch（显式 pure fallback）。pure↔torch parity 在 float64 下 ≤1e-7（n_z=16 ≤1e-6）——torch 路径是**同一函数**而非 look-alike，故回滚精确。
- **backend evidence 的 coverage / parity 隔离**：session owner 为在线 SSL 保留最近两个不可变 substrate snapshot，构造最短两步 causal trace；单步 substrate 不再让 ACTIVE SSL 永久以 `trained_steps=0` 早退。ACTIVE torch SSL 的持久 optimizer session 只绑定当前 checkpoint preimage；正式 checkpoint hydrate 或 joint-cycle rollback 后必须失效并在下一批重建，不能把 owner 授权的 restore 误报为外部参数篡改。accelerated ndim runtime 对短 CMS context 必须复用 pure `_project_to_ndim` 的 tile 语义，禁止 zero-pad 形成第二套 forward。证据 probe 分两阶段：exercise session 只证明声明 backend 真实执行/write-back；随后从同一正式 checkpoint 新建 session 测 pure/runtime/torch forward parity，禁止拿三条已经学成不同参数的 lane 冒充同参数数值比较。
- `learned-lite` / 旧 `full-learned`（纯 Python heuristic）仍保留为 fallback / rollback baseline；torch 路径默认 DISABLED，需显式 WiringLevel 提升。

**快照 schema**：见 `docs/DATA_CONTRACT.md` 3.2 节

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 契约式运行时（5.5）| 通过快照发布控制器状态 |
| 依赖 | Prediction Error 主链 | 直接消费上一轮 outcome mismatch，驱动 owner-side temporal 调节 |
| 依赖 | 多时间尺度学习（5.1）| 在 online-fast 时间尺度运行 |
| 被依赖 | 双轨学习（5.4）| 提供 z_task / z_rel 控制器代码 |
| 被依赖 | 认知 Regime（5.8）| 控制器切换与 regime 切换对齐 |
| 被依赖 | 信用分配（5.6）| 提供抽象动作级信用分配的基础 |
| 协作 | 评估体系（5.7）| F5 抽象质量评估 |

## 变更日志

- 2026-08-02: 为 rate-distortion 判据补齐执行纪律，判据逻辑本身不变。
  (a) `scripts/preregister_eta_rate_distortion.py` 冻结 alpha 网格、seed
  schedule、优化预算、gap 阈值、arm-separation 规则、封闭 verdict 集与 5 个源码
  SHA；runner 的 `--preregistration` 在 sweep 变量、gap 阈值或源码任一漂移时
  fail closed，缺预注册时 artifact 与 `report.md` 标记
  `claim_scope=mechanism-only-smoke` / `verdict_authoritative=false`。
  (b) 裁决从 sweep 中抽出为纯函数 `adjudicate_rate_distortion(curves, gaps)`，
  审计者可脱离模型与加速器从 `curves.json` / `gap_assessments.json` 复算 verdict。
  (c) runner 接入共享 `artifacts/.companion-evidence-mps.lock` 与 `require_mps()`；
  此前它绕过控制面，`PYTORCH_ENABLE_MPS_FALLBACK` 从未被检查。
  (d) 每个 `(arm, alpha, seed)` cell 落不可变 checkpoint，`--resume` 按 cell 续跑；
  此前全部输出只在结束时一次性写盘，中断即全损。
  (e) `runtime_origin` / `fallback_active` 的 `getattr(..., default)` 改为直接属性
  访问——默认值会在属性改名时把 `fallback_active: false` 静默写进证据。
  (f) 补 31 项 rate-distortion 测试与 20 项 `TransformersSteeredActionScorer`
  测试（此前两者合计 1237 行零测试），覆盖梯度只到控制 delta 不到冻结基底、
  norm cap、joint 臂 pristine 恢复、gap 检测三条拒绝路径与全部 verdict 分支。
- 2026-08-03: **ETA-on-LLM Gate 1 = PASS**。四层根因逐一修复后重跑权威扫
  （`artifacts/eta_stage1_gate1_v4_hardst_auth_20260803/`，预注册
  sha256 `b0d18f60…`，18 cells）：(a) `smooth` posterior 稳定 rate 轴数值；
  (b) 观测协议 `partially-observable-staged-plan.v4` 把计划信息分段揭示
  （step-0 只给第一目标，各 arrival 揭示下一个），使切换获得真实 distortion
  收益；(c) `rate_gating="switch-gated"` 让 KL 只在切换时计费；(d)
  `StoreSSLTrainingSession.steered_gate_mode="hard-st"` 用离散 straight-through
  门堵住连续门每步微幅走私新信息的漏洞。读数：spearman −1.000、span 1.933、
  hard switch 0.12–0.96、heldout boundary F1 全 alpha 0.240–0.671（首个
  switching 存活的权威扫）。frozen 臂另检出方向性近垂直 gap，但缺 joint 臂
  且 gap 区内 F1 未高于区外，属 Gate 3 范畴，不从此 artifact 主张。Stage 2
  解锁；整体 `kill-eta` 仍待 Gate 3。
- 2026-08-03: **ETA-on-LLM Gate 2 = FAIL**（`gate-2-fail-kill-llm-transfer`）。
  执行 Stage 2 全链：语料 `artifacts/eta_stage2_corpus_20260803/`（120 文档 /
  content sha256 `a89b7015…`）→ `continued_pretrain_and_merge` 落 merged 冻结
  基底 `artifacts/eta_stage2_merged_20260803/`（LoRA r16α32，2000 步，
  initial_loss 2.610 → final_loss 0.119，权重指纹 `08472c6d…`）→ 双臂线性
  分类 probe `artifacts/eta_stage2_probe_20260803/`。参数与定稿预注册
  `artifacts/eta_stage2_gate2_prereg_20260803/`（sha256 `a2561f3b…`）逐字节
  一致。8 类（chance 0.125 / majority 0.166）heldout 最后一层读出：续训臂
  acc 0.131、裸 Qwen 0.166（= majority，probe 塌到多数类）；三条件
  `2×chance≥0.25` 否、`续训>基线` 否（0.131<0.166）、`随前缀上升` 是。稳健性：
  全 24 层最优（base 0.214 / pretrained 0.202，非合规读出）仍双否。判读：
  Qwen2.5-0.5B 残差流领域续训后无线性可解码 active-subgoal 层级，next-token
  近记忆化甚至略微恶化最后一层读出。按预注册 `decision_rules`：claim
  `claim_llm_residual_carries_subgoal_hierarchy` 在 0.5B 被驳，整条 LLM 迁移
  路线 kill、Stage 3 不跑；ETA 主张**未**永久摘除（保留 Gate 3 / 独立处置包），
  规模敏感性须另立新预注册。
- 2026-08-03: **Stage-2 仪器审计 + v2 重审**，收窄上一条的判读。审计：v1
  语料/probe 的计划载体是 `_context_sentence` 哈希指纹（与 Gate-1 定罪的
  协议 v2 缺陷同类），heldout 非指纹信息天花板离线复算 0.1805 < 及格线
  0.25——v1 FAIL 定罪仪器而非基底命题。仪器 v2（owner
  `eta_rate_distortion_evidence.py`：`eta_stage2_documents` /
  `eta_stage2_probe_rows` 复用 v4 staged-plan 渲染、累积轨迹前缀 probe、
  train-split 选层、fail-loud 截断；`_rate_distortion_observation_texts`
  扩展返回 per-step active subgoal，v4 文本逐字节不变）以新预注册
  `artifacts/eta_stage2_gate2_prereg_v2_20260803/`（ceiling 1.0 验证）重跑：
  裸 Qwen heldout `0.901`、续训 `0.944`（后段层 0.99–1.00）——**残差流大幅
  承载 active subgoal**；`2×chance` 与因果对照 PASS，仅 `随前缀上升` 在显式
  揭示制度下 regime 错配（0.979→0.879 保持衰减）→ 按字面 v2 FAIL 封存，
  待 v3 预注册决策。`kill-eta` 持续有效，Stage 3 锁定。
- 2026-08-03: **Stage-2 v3 重审（用户授权）**。`assess_gate2` 新增
  `retention.v3` 第二条件（late ≥ 2×chance 且 early−late ≤ 0.15，CLI
  `--gate-conditions`，三分支单测）；v3 预注册
  `artifacts/eta_stage2_gate2_prereg_v3_20260803/`（sha256 `2f3b3bf4…`）用
  **全新 corpus seed 20260804** 挡 forking paths（判据在任何新 seed 读数
  存在前冻结，v1/v2 封存件不再判读），新 seed 天花板 1.0。全链重跑
  （语料 `d78281b5…` → merged `0e387aba…`，1.034→0.023 → probe
  `artifacts/eta_stage2_probe_v3_20260803/`）：`2×chance` PASS（0.967 =
  7.7×，选层 12）、`retention.v3` PASS（0.995/0.918，衰减 0.077）、
  `续训>基线` FAIL——裸 Qwen 基底（选层 21）`0.977` 反超续训臂 `0.967`，
  **基底无需领域续训已在天花板携带子目标**（v4 可读协议下计划在文本中，
  基底自然线性编码）。字面 verdict = FAIL 封存；三轮 FAIL 分别定罪仪器 /
  判据 / 对照设计，实质命题跨两 seed 四臂证实。不注册 v4 重判；阶梯处置
  升级为程序级用户决策，Stage 3 保持锁定。
- 2026-08-03: **Stage 3 解锁并启动**（用户程序级裁定：Gate-2 看门前提已
  实质确立，三个字面 FAIL 封存不改判，解锁的是阶梯推进权）。冻结 Stage-3
  v2 预注册 `artifacts/eta_stage3_prereg_v2_20260803/`（Gate-1 权威扫
  同款参数：v4 + smooth + switch-gated + hard-st + 300 updates，corpus
  seed 20260802；基底 Stage-2 v2 merged `063077b7…`；frozen + joint
  双臂 36 cells）；权威扫 `artifacts/eta_stage3_rate_distortion_20260803/`
  启动。Gate 3 为 ETA 机制终审：retain-eta-on-llm 或 kill 升级永久摘除。
- 2026-08-04: **Stage 3 完成并封存 `kill-eta`**。36/36 cells；双臂可分
  （0.1264 > 0.0673），frozen rate 轴 Spearman −0.9429 / span 2.0680，
  但 frozen 无近垂直 gap，现有 boundary F1 也未在候选区内抬升；joint
  反而检出 gap。当前 ETA-on-LLM operationalization 维持 legacy / SHADOW，
  不改变 memory、PE、relationship continuity 或 production WiringLevel。
  P1 登记 exact-entry / bias / z 因果性 / subgoal oracle 四组只读诊断；忠实
  rewrite 只可作为新 claim、新预注册的条件分支。
- 2026-08-02: 启动 **ETA 迁移 LLM 四级阶梯**，把论文成立的前提逐级搬到冻结 LLM 上；
  `kill-eta` 判定在 Stage 3 通过前保持有效。**证据 SSOT**：
  [`eta-llm-transfer-evidence.md`](./eta-llm-transfer-evidence.md)（已挂
  `evidence_program` claim registry / `00_INDEX` §9）。本包只建机器 + 预注册 +
  直接相关 smoke；缩减 Gate 1 权威扫见该 SSOT 当前状态（FAIL，修方差后重跑）。
  **前提事实核对（这是阶梯存在的根据）**：ETA 原论文的证明**不在 LLM 残差流上**，
  而在作者**从零预训练的小模型**残差流上——网格世界用因果 Transformer、ant 用 SSM
  Hawk，数据 `D` 是领域内观测-动作轨迹（`research/eta/emergent-temporal-abstractions-2512.20605.zh.md`
  L51/L61/L227/L265），且先冻结基底再训元控制器（L93/L243）；论文自证前先用线性
  probe 确认残差流携带子目标信念（L73、附录 B）。LLM 在论文里只作动机类比（L19）与
  未来展望（L215/L217「期待研究……能否迁移到 LLM」），**从未被验证**。我们 8-01 的
  `kill-eta` 直接抓通用 Qwen（从未在本领域轨迹上预训练、残差流是否携带子目标从未
  probe）故对 ETA 不构成公平证伪；四级阶梯的 Stage 2（继续预训练 + 分类 probe 前置）
  正是把论文 L51/L73 的前提补到 LLM 上，Gate 全过前不重判。
  **Stage 1（数据机制）**：环境 owner `vz-temporal::internal_rl/proof_environment.py`
  新增 seeded 程序化生成器（`generate_hierarchical_environment` + hub relay 保证任意
  子目标序可达 + `stitch_waypoints` BFS 拼接 + `generate_hierarchical_routes`
  按 ordering 哈希分区做 train/heldout 组合不相交）；`eta_proof_benchmark` 新增
  `generate_eta_proof_corpus`/`ETAProofCorpus`，rate-distortion harness 增
  `corpus=` 注入与 `RateAxisResponse`（frozen 臂 spearman(alpha,rate) + rate span，
  可离线复算）。Gate 1 阈值预注册于
  `artifacts/eta_stage1_rate_axis_prereg_20260802/`（frozen 臂 spearman ≤ −0.8 且
  rate_span ≥ 0.30）。**可行性 pilot（非正式）**
  `artifacts/eta_stage1_rate_axis_pilot_20260802/`：6→24 路线、1 seed、3 alpha、
  8 updates 下，8-01 kill 的两处异常里**记忆化已消除**（24 路线 train≈heldout
  distortion），**rate span 升到 0.64–0.81（远超 ~0.20 基线）**，但 rate 对 alpha
  非单调（α=3.0 回弹），spearman 仅 −0.5，未达 −0.8——支持数据机制假设但需正式
  200 路线扫描或修 posterior 方差参数化后才能过 Gate 1。**Stage 2（补前提）**：
  substrate owner 新增 `continued_pretrain_and_merge`（rare-heavy PEFT LoRA →
  merge_and_unload → 冻结落盘 + 权重指纹，离线 substrate refresh，不动原始 Qwen）
  与线性分类 probe `fit_linear_classification_probe`（对齐论文附录 B，从各层
  final-position hidden 解码 active subgoal）；`eta_route_probe_rows` +
  `render_eta_route_documents` 提供 SSOT 观测/动作表面；Gate 2（补课后 heldout
  probe ≥ 2× 随机、随前缀上升、优于原始 Qwen）定稿预注册于
  `artifacts/eta_stage2_gate2_prereg_20260803/`（sha256 `a2561f3b…`，取代 08-02
  草稿并用真实 SHA 锁 6 个 owner 源），不过则整体 kill。**Stage 3**：
  `run_eta_rate_distortion.py` 增 `--corpus-seed/--train-routes/--model-source`，
  可在补课基底上按 `eta-rate-distortion-evidence.v1` 已冻结规则重跑双臂判据；
  预注册于 `artifacts/eta_stage3_prereg_20260802/`（6 alpha × 3 seed × 2 臂）。
  **Stage 4（contingent）**：仅设计骨架
  `research/eta/eta-stage4-dialogue-transfer-prereg-skeleton.md`（对话无子目标真值，
  boundary F1 不可作门，退回 gap + heldout 泛化），不执行。所有新代码为 evidence
  lane，不改任何 production WiringLevel，补课基底为独立 artifact，可回滚。
- 2026-08-01: ETA Eq.3 rate-distortion 判据第一次可执行，读数为
  `kill-eta`，但**该次运行不具备正式资格**（无预注册、脏工作树、与七日矩阵并行
  占用 MPS、无 checkpoint）；状态为 mechanism-grade，摘除 ETA 主张须等预注册
  重跑（`artifacts/eta_rate_distortion_20260801`）。本包先修掉与论文
  Eq.3 的四处结构性偏离：(a) `vz-substrate` 新增
  `TransformersSteeredActionScorer` 可微受控前向（hook 层注入控制 delta、
  上半部保留在 autograd 图内、基础参数 `requires_grad=False`、可微 norm cap），
  distortion 第一次是穿过被控制冻结模型的动作 NLL；(b) `torch_store_ssl`
  的 `z_tilde` 改为 seeded `torch.Generator` 真实重参数化采样；(c) distortion
  目标由动作词表受限 softmax 决定、与 `n_z` 解耦；(d)
  `switch_rate/binary/group/gate_choice` 权重默认归零（`torch_store_ssl` 与
  `ssl.py` 双处），steered 模式传非零 switch 权重 fail loudly，损失恢复为
  论文两项。观测协议改为 partially-observable（不再泄漏 remaining route）。
  运行时新增 `model_dtype` 显式加载覆盖（joint 臂 fp16 主权重在 Adam 下
  溢出，两臂统一 fp32 以免 dtype 混淆对比）。正式扫描为两臂 x 6 alpha x
  3 seed x 40 updates：联合臂曲线完全平坦（distortion 恒 `0.0001`），两臂
  分离 `0.4059` >> 阈值，仪器有效；冻结臂 tradeoff 存在但最大坠落段占
  48.5% rate 跨度（阈值 25%）、无近垂直 gap，gap 判定段内 boundary F1
  `0.000` 低于段外 `0.141`，rate 轴对 alpha 响应弱且非单调。按运行前已固定于
  源码的判据规则，verdict = kill-eta。该读数足以阻止把"冻结基底 + 元控制器可
  涌现子目标对齐时间抽象"当作已确立前提引用，但因缺预注册与源码冻结，不足以
  据此摘除 ETA 主张。ETA 主张摘除与 `vz-temporal` 退回 legacy 路径须先有
  预注册重跑，且属于后续独立收敛包；`_step_impl_legacy` 与全部 switch 正则代码
  保留为回滚基线。详见
  `research/eta/eta-segment-credit-evidence-plan.zh.md` 2026-08-01 节。
- 2026-07-29: ETA Gate 2 v32 对 v31 的 `3168` 条 Qwen 候选记录增加
  target-oracle 到独立 audit 的 permutation-null transfer gate，并把 selector
  injection 门收紧为 train/eval/heldout/validation 四个冻结分区 audit 均为正。
  reachable-solution evidence 通过，但 eval audit selected credit 为
  `-0.000131504`，故 selector 注入保持关闭；confirmation 尚未锁定，Gate 2
  仍为 `mechanism-supported`。重判只读复用原始 measurement/PE/credit，并以
  SHA256 固定源 artifact。
- 2026-07-29: ETA Gate 2 v31 将 counterfactual action-value target 从模型自身
  continuation NLL 迁到真实 residual forward 的 EnvironmentOutcome，经 PE owner
  与 credit owner 形成 signed action credit。primary 使用未校准的预注册 proof
  target，next-prefix residual 仅作独立 audit；新增候选级
  `counterfactual_outcomes.jsonl`，旧 v30 target 保留为显式回滚。
- 2026-07-28: ETA Gate 2 v30 将单 realized continuation target 升级为
  prefix-level target/audit 双 cohort expected value，并冻结 fresh validation
  只读检验。transformers substrate 新增同 prefix/control 的 batch continuation
  scoring，保持逐条数值和顺序契约。full-width 896、CPU、单 seed 2+2 校准的
  validation audit selected delta 为负，selector 注入继续关闭；下一目标迁到
  真实 environment outcome / PE。
- 2026-07-28: ETA segment-credit 第六收敛包修复动作族运行时脱轨与评估真值
  泄漏。SSL topology discovery 现在以 `z_tilde` proposal 及其 decoder control
  为对象，`beta_t` 只决定该 proposal 是否在当前时刻成为 active code；因果
  runtime proposal 则由 temporal owner 按同一结构评分投影回已学 family bank。
  评分同时比较 latent code 与 decoder action，避免仅在 latent 邻近、但解码后
  动作不一致的错误归族。bank 为空时保持历史 identity 行为，RL/runtime 阶段仍
  冻结 topology。`beta_t` 的无标签分位数校准改为单次最大 `0.08` 的有界更新，
  防止一个 batch 把 runtime switch threshold 推到退化区域。

  evidence readout 不再用模型 rollout 的 dominant family 反过来充当
  `true_family_id`。v13 以环境 `ExpertActionTarget.action_id` 为唯一真值，
  discovered-family 到 expert-action 的对齐映射只在 train split 拟合，eval/heldout
  只读该映射；该 readout 不回灌 temporal owner。固定冻结
  `Qwen/Qwen2.5-0.5B-Instruct`、MPS、`n_z=16`、每 seed 75 个 persistent
  Adam updates 的 5-seed 预注册配置先得到 `weak`：全部逐 seed PE effect
  非负，但 3 个 seed 的 turn baseline 已碰到同一 PE 地板，bootstrap CI 下界
  为 0。保持全部模型、训练和 gate 参数不变扩展到 10 seeds 后，v13 artifact
  `artifacts/eta_evidence_gate_1/segment_vs_turn_decoder_family_manifold_v13_qwen25_05b_mps_10seed_75update_20260728`
  达到严格 `retain`：credit F1 delta `0.9000 [0.9000, 0.9000]`，
  false-credit reduction `0.8333 [0.8333, 0.8333]`，
  family-assignment delta `0.5000 [0.5000, 0.5000]`，
  held-out PE reduction-rate delta `0.1589 [0.0397, 0.2780]`，
  beta boundary F1 `0.7652 [0.7652, 0.7652]`。该结果只支持本 matched-control
  管线中的 segment-credit 命题，不外推为完整 Volvence thesis 已获验证。
- 2026-07-28: ETA segment-credit 第五收敛包达到严格 `retain`。根因修正在
  `vz-temporal` encoder owner：`NdimEncoderParameters.current_proj` 以单位矩阵
  初始化，保持未训练行为兼容，并由 ACTIVE Torch SSL 学习 residual observation
  到 `z_t` 的当前时刻投影；pure、backend runtime、Torch forward 和 parameter
  snapshot/writeback 使用同一参数。persistent Torch optimizer 以 parameter store
  为作用域；world/self 各自保留独立 Adam state，同一 store 跨 batch 复用，禁止
  跨 owner 共享 momentum。运行时采用标量 hard beta：新 episode 第一步
  建立 code，continuation 精确保留上一 segment code，switch 才接纳 proposal。
  expert-action family discovery 按 runtime 的单 observation step、recurrent hidden
  和 hard gate 语义重放，避免 prefix 训练分布与 serving 分布不一致。
  family classification 先用 structural similarity 建立 `0.005` 近邻候选集，
  再允许 learned payoff/continuation prior 在候选集内决胜；因此结果先验可以
  区分同构动作，但不能把 observation 吸到语义正交的高收益 family。merge 固定
  保留拓扑中更早 family 的 lineage ID，support/stability 只参与 centroid 聚合，
  不再造成抽象身份漂移。

  证据设计同时冻结两项语义：(1) 共享 adapter/metacontroller checkpoint 固定
  `initialization_seed=42`，5 个 evidence seed 只改变 episode 顺序与 delayed
  observation lag，不为每个用户随机重建模型；(2) proof subgoal 同时保留运行时
  `completion_threshold` 与不可被 backend calibration 覆盖的
  `nominal_completion_threshold`，held-out PE 预测真实
  `alignment_score - nominal_completion_threshold`。冻结
  `Qwen/Qwen2.5-0.5B-Instruct`、MPS、5 experience seeds、每 seed 48 个 persistent
  Adam updates 的 v12 artifact：
  `artifacts/eta_evidence_gate_1/expert_action_mps_margin005_s5_c48_v12_20260728`。
  verdict=`retain`；credit F1 delta mean `0.7317`、95% CI
  `[0.7175, 0.7460]`，false-credit reduction `0.5883`
  `[0.5683, 0.6083]`，family-assignment delta `0.6000`
  `[0.4667, 0.6667]`，held-out PE reduction-rate delta `0.0202`
  `[0.00498, 0.03549]`，beta boundary F1 `0.6095`
  `[0.6012, 0.6179]`。每个 seed 均有 2 个 active families、6 个 held-out events；
  88 个 beta boundary 对 95 个真实边界。trace backend 保持单族，作为“无
  action-identifiable residual 不得宣称动作族涌现”的负面对照。
- 2026-07-28: ETA segment-credit 第四收敛包完成专家动作监督、轨迹隔离与
  0.5B causal 分布证据。`vz-substrate` 新增 offline non-runtime
  `ExpertActionTarget`，`TraceStep` 可一一携带 demonstrator action vector；
  targeted trajectory 必须全步完整，eval/heldout target 不进入 optimizer，
  subgoal boundary/reward/outcome 继续只作 evaluation。`torch_store_ssl` 的
  distortion 因而从 next-residual proxy 升为 expert-action vector MSE，并显式
  报告监督类型、action target variance、boundary/continuation switch
  probability 与 PE-derived preference。switch-rate KL 修正为
  `KL(Bern(mean_t p_t) || Bern(prior))`，使 KL 约束抽象率而不强迫每步概率相同；
  threshold 由同一无标签 aggregate rate 校准，不读取边界标签。每个环境 observation
  对应一个 residual step，SSL/runtime temporal axis 保持一致；每个独立 rollout
  开始时清零 trajectory-local GRU hidden/code/active-family telemetry，但保留
  metacontroller 参数、action-family bank 和长期用户状态。证据 harness 禁用
  `temporal_fast_prior`，防止外部 prior 代替 learned beta 决策，并在 SSL 后使用
  train-only causal rollout 的连续 beta 概率做运行分布 rate 校准。

  冻结 `Qwen/Qwen2.5-0.5B-Instruct`、MPS、5 seeds、每 seed 75 个 persistent
  Adam updates 的最终 artifact 为
  `artifacts/eta_evidence_gate_1/segment_vs_turn_causal_rate_calibrated_qwen25_05b_mps_5seed_75update_20260728`
  （evidence/manifest schema v11）。所有 seed 均观察到 7–8 个 heldout delayed
  events；segment credit F1 相对 turn credit 的增益均值 `0.6433`、95% CI
  `[0.5421, 0.7238]`，false-credit reduction 均值 `0.5019`、CI
  `[0.3844, 0.5901]`，两门成立。边界仍未稳定：总 beta boundary `63` 对真实
  `95`，boundary F1 均值 `0.4815`、CI `[0.2249, 0.6485]`；seed 0 的
  train-only causal rate calibration 把 threshold 推到 `0.9402`，heldout 上
  产生 0 个边界，说明跨分布 threshold calibration 仍脆弱。五个 seed 都只有
  一个 active family，family-assignment 与 heldout PE delta 均为 0。严格 verdict
  继续 `fail`；可陈述的结论仅是“专家动作监督下，segment credit 在已观察延迟结果上
  显著优于 turn credit”，不得升级为“ETA 全链已成立”。下一唯一收敛包应归属
  temporal action-family topology / calibration owner：先解决 train→heldout rate
  稳定性和单族坍缩，再重跑 family/PE 门；禁止调低 gate 或使用 evaluation boundary
  校准阈值。
- 2026-07-28: ETA segment-credit 第三收敛包完成 live-store switch
  rate-distortion 机制：`beta_t` 使用标量 hazard 和 hard straight-through gate，
  loss 新增 Bernoulli switch-rate KL、准二值约束、逐维 group coherence、
  keep/switch counterfactual choice，并以 persistent Adam 对多 trajectory batch
  连续优化；默认预测跨度为 3。证据 harness 的每个 seed 现在使用独立的
  metacontroller 初始化，同时固定 `replacement_mode=causal`，使 learned switch
  继续拥有 beta 边界，Internal RL 只提供 z candidate。冻结
  `Qwen/Qwen2.5-0.5B-Instruct` 的 5-seed、12-cycle MPS 运行证明机制本身可动：
  optimizer state 每个 seed 复用 11 次，60 次 ACTIVE writeback，平均 hard switch
  frequency `0.3038`；但总共只有 28/120 个运行时 beta boundary，仅一个 seed
  得到非零 boundary F1，整体 boundary F1 `0.1061`，所有 seed 仍只有一个动作族，
  family 与 held-out PE delta 均为 0，严格证据门继续 `fail`。
  审计确认剩余缺口不是继续调 beta 权重：公共 `TrainingTrace` 只发布 token、
  feature surface 与 residual activation，没有 ETA Eq.3 所需的专家动作 `a_t`；
  当前 distortion 实际监督“下一时刻 projected residual vector（absolute/innovation
  proxy）”。proof subgoal signature 仍只允许用于 evaluation，禁止回灌训练。
  因此 evidence schema 升为 v4，显式发布 `ssl_supervision_target` /
  `expert_action_supervision=false`，并新增 `ssl-uses-expert-action-targets`
  retain gate。下一收敛包必须先建立不泄漏评测边界的专家动作 trajectory
  契约与数据，再判断动作边界是否涌现；当前结果不得表述为 Eq.3 已完成。
  后续 matched rerun 又修正了两个 owner 内缺口：KL 改成 per-dimension rate，
  hard gate 关闭时仍用 switch counterfactual 训练 proposal，并把三标量 residual
  summary target 升为 16 维 projected residual vector。冻结 0.5B、5-seed、
  3-cycle MPS 结果仍为 0 个 runtime beta boundary、一个动作族和 0 family/PE
  delta；平均 beta probability `0.4817`。这进一步确认 residual proxy 不能替代
  Eq.3 的专家动作监督，下一包边界保持为 action trajectory contract，而非继续调
  switch prior 或 threshold。
- 2026-07-28: ETA segment-credit 第二收敛包把 open-weight proof runtime 已发布的
  `SubstrateSnapshot` 前缀序列正式接入 `vz-temporal::MetacontrollerSSLTrainer`。
  owner 新增 `build_training_trace_from_substrate_snapshots()` /
  `optimize_residual_trajectory()`：每个 causal prefix 只抽取其最后一个 token 的
  `feature_surface + residual_activations`，避免把完整前缀反复展开成重复训练样本；
  非 frozen、非 `RESIDUAL_STREAM`、无 residual 或不足两个快照均 fail loudly。
  该入口只消费既有不可变 substrate 快照，不新增 runtime slot，也不改变
  `docs/DATA_CONTRACT.md`。`vz-runtime` 的证据 harness 只负责在同一参数仓上调度
  `SSL -> z_t Internal RL` 周期，并记录 prediction loss、KL、switch frequency、
  autograd parameter change 和 ACTIVE writeback；唯一参数写入 owner 仍是
  `MetacontrollerSSLTrainer` / `MetacontrollerParameterStore`。结构化 subgoal
  继续仅用于 boundary/credit evaluation，禁止作为 `beta_t` 标签。迁移回滚开关为
  `training_mode=rl-only`，退出条件是 real residual intake、非零 SSL 训练步、
  ACTIVE 写回、多个动作族、非零 beta 边界与 held-out PE 证据同时可检查。
  5-seed MPS 结果证明 real residual intake 与 ACTIVE 写回成立，但整体证据门仍
  `fail`：交替训练产生 84 个 beta boundary，却只有 `0.0333` boundary F1、一个
  活跃动作族，family/held-out PE delta 均为 0，并在两个 seed 上丢失 held-out
  outcome。匹配的 `rl-only`/`n_z=16` 对照仍为 0 个 beta boundary。代码审计进一步
  确认 live-store torch SSL 的可微目标当前只有 action MSE + posterior KL；
  `switch_threshold` 只用于统计，loss 没有 beta switch-rate prior。因此下一包的唯一
  owner 是 `torch_store_ssl` 的无标签 rate-distortion/switch-rate 目标，而不是
  evaluation 侧阈值补丁。
- 2026-07-27: 给初始先验加界，解除包络强制的最后一个阻塞项，数字蚂蚁翻成 `True`
  （收敛包 envelope-pristine-init）。缺陷：`validate_causal_action_head_magnitudes`
  的立论是"拒绝 owner 更新路径永远产生不出来的映射"，但 owner 的**构造函数**恰好能产生
  一个——`_initial_causal_action_head_parameters` 在 `rank < n_z` 时回落到无界高斯
  `_random_mat(rank, n_z)`，与 `factor_absolute_limit=1.5` 毫无关联。实测这不是角落
  情形：全可达网格（`n_z <= 64` × 全部 `rank` × 三轨，6048 个走随机回落的配置）里
  **11.0%（667 个）越界**，最坏 `n_z=64, rank=1` 达 `3.965`；数字蚂蚁自己的 `n_z=16`
  上 `self=1.9823`、`shared=1.6192` 越界。
  **选定方向：修构造器，不给 `update_step == 0` 开豁免。** 理由是可判定的而非风格偏好：
  `factor_absolute_limit` 被 owner 写入路径**无条件**施加（pure `update_causal_action_head`
  的内联 clamp 与 `project_causal_action_head_update` 的 `clamp_absolute` 都不读
  `update_step`），因此 owner 写入路径的像恒落在包络内，第一步也不例外——构造器是唯一
  的例外，也就是唯一该修的一侧。反向若给初始先验放宽上界，validator 接受的集合就严格大于
  owner 自身的像，而 `update_step` 只是 restore 快照里的普通字段，越权/损坏 archive 声明
  `update_step=0` 即可装入任意 input factor：豁免在任何写法下都是宽度等于自身的旁路，
  正是"跨状态固定转向截距"这条立论要挡的东西。
  **修法**：新增 `_bounded_initial_input_factors`，仅在越界时按**单个全局正标量**
  `limit / peak` 重标定，方向结构（相对幅度、符号、行列几何、秩）原样保留；已在包络内的
  抽样**原样返回，不做任何浮点运算**。`_random_mat` 本身不动——encoder/decoder 权重
  （seed 100/101/103/104）仍在消费它，逐字节不变。末尾保留一次 `clamp_absolute` 兜住
  `peak * (limit/peak)` 可能高出上界 1 ulp 的情形（实测当前包络与种子下全网格一次未触发，
  保留是因为该性质依赖具体 limit 与 seed，不是可继承的不变量）。
  **差分证据（逐字节，非断言）**：全网格 6240 个配置的 input factor 按 `struct.pack("<d")`
  取 sha256 前后比对——**5573 个逐字节不变，667 个改变，且改变的 667 个恰好等于原先越界的
  667 个**；"改变了但原本合法"的计数为 0，"修完仍越界"的计数为 0。数字蚂蚁的 WORLD 轨
  （1.0749）在不变的那一侧。附带效果：basis 被推离 tanh 饱和区（`n_z=64, rank=1` 最坏
  pre-activation `7.552 -> 2.857`，`tanh 1.00000 -> 0.99342`，该坐标从零梯度死区回来）。
  **`ANT_CAUSAL_ACTION_HEAD_ENVELOPE_ENFORCED` 由 `False` 翻成 `True`**；通用默认
  `internal_rl_causal_action_head_envelope_enforced` 保持 `False`（opt-in 契约不变，
  `tests/test_runtime_transition_replay.py` 仍从 permissive 路径装入 bias `0.35/0.4`）。
  强制函数 `test_pristine_head_initializer_violates_...` 如期失败，已替换为
  `test_pristine_head_initializer_respects_the_envelope_it_is_validated_against`
  （全网格逐个过 validator）；另新增 `test_bounded_initial_input_factors_rescales_without_reshaping`
  与 `test_envelope_has_no_update_step_dependent_exemption`（双向锚定 gate 对 `update_step` 盲）。
  验证：`packages/vz-embodiment-ant/tests` 240 passed / 1 failed（原 19 个包络失败清零；
  剩下的 `test_forbidden_list_covers_every_kernel_wheel_module` 缺
  `volvence_zero.decision_workspace`，属并行 wave 既存失败）；`packages/vz-temporal/tests`
  109 passed；`tests/test_runtime_transition_replay.py` + `tests/test_temporal_interface.py` +
  `packages/vz-runtime/tests` 187 passed；`tests/contracts` 失败/错误集合 = 既存基线
  （4 failed + 5 errors，名单逐条一致）；ruff 对改动文件的发现集合与改动前完全相同
  （interface.py 20 条既存，无新增；另两个文件 0 条）。
  负对照（三项，每项都实测会失败）：还原成无界 `_random_mat` →
  `..._respects_the_envelope...` 失败；把重标定换成**逐元素 clip**（同样落在包络内，但破坏
  方向结构）→ `..._rescales_without_reshaping` 失败，即该测试锚定的是重标定语义而不只是
  "在界内"；给 validator 加 `update_step == 0` 豁免 →
  `..._has_no_update_step_dependent_exemption` 失败。
- 2026-07-27: 打通包络开关的生产可达路径，并把三处过度声明的不变量改写成真的
  （收敛包 W3-b-fix-follow-up，对抗评审残留项闭环）。五项：
  (1) **包络强制此前在生产中不可达**：`grep -rn envelope_enforced packages scripts tests`
  只在 vz-temporal 内部命中，因此 owner 写入路径虽始终强制，ARCHIVE/CHECKPOINT 安装路径
  在任何真实部署里仍接受越界 head——正是"任何 archive 都能装入固定转向截距"这一缺陷本身。
  新增 `FinalRolloutConfig.internal_rl_causal_action_head_envelope_enforced: bool = False`
  并在四个 `set_causal_action_head(...)` 调用点（`final_wiring.py` 与 `agent/session.py`
  各 WORLD/SELF 两处）传下。**打开它的那一步被一个更深的缺陷挡住，本包不修但已量化**：
  owner 自己的 `_initial_causal_action_head_parameters` 在 `rank < n_z` 时用无界高斯初始化
  input factor，实测 `n_z=16` 下 `self=1.9823`、`shared=1.6192` 越过
  `factor_absolute_limit=1.5`，而 `restore_parameter_snapshot` 校验全部三轨、
  `internal_rl_causal_action_head_rank` 只配置一轨，因此任何域声明 `True` 都会在第一次
  checkpoint 恢复时炸在 owner 刚创建的头上（实测蚂蚁包 19 个测试失败，全部是
  `update_step=0`、output/bias 全零、live steering 权限精确为零的头）。收窄校验会削弱 gate，
  给初始化加界会改动通用默认算术，故数字蚂蚁 profile 显式声明 `False` 并原地记录该阻塞，
  由 `test_pristine_head_initializer_violates_the_envelope_it_is_validated_against` 作强制
  函数（阻塞一修该测试即失败，迫使同批翻成 `True`）。
  〔后续：该阻塞已由 2026-07-27「给初始先验加界」收敛包解除，强制函数如期失败并被替换成
  `test_pristine_head_initializer_respects_the_envelope_it_is_validated_against`，蚂蚁 profile
  已翻成 `True`。见本日志首条。〕
  (2) **后一次调用会静默清掉前一次声明**：`set_causal_action_head(..., envelope_enforced)`
  把 `False` 默认值无条件写进 store，于是
  `MetacontrollerParameterStore(causal_action_head_envelope_enforced=True)` 会被任何一次
  不提该开关的 head 配置调用关掉强制校验。改为 `bool | None = None`，`None` = 不声明。
  (3) **给精确不变量留了 slack**：删除 `sandbox.py` 的
  `_REALIZED_PAYOFF_AGREEMENT_TOLERANCE = 1e-9`，改用 `!=`。两个量在接通路径上按构造逐位
  相等（`prediction/error.py::_clamp_signed` 与 sandbox `_clamp` 是同一组界的幂等 clamp），
  outcome lineage 另行强制，容差只可能吞掉真实分歧。见 `prediction-error-loop.md`。
  (4) **synthetic torch 分支不在跨 lane 契约内**：把该分支写死的三处 `0.0, 1.0` 换成 owner
  常量 `LATENT_CODE_BOUNDS`（算术逐字节不变），并把跨 lane 那句话的作用域精确限定到
  runtime-replay 重建（见上文条目）。接到 `resolve_latent_code_bounds` 是错误修法，实测会
  改变 818 行字节探针中的 88 行。
  (5) **两处记述失真**：`project_causal_action_head_update` 的单步位移上界改写为有条件成立
  （baseline 在包络外时绝对 clamp 获胜，实测 `0.35 -> 0.1` 一次位移 25 倍上界；且区间按
  binary64 求值，精确算术下会以 1 ulp 之差为假）；`tests/contracts 3668 passed` 改记为
  失败/错误集合等于既存基线（该 pass 计数当日复测已漂到 `3682`）。
  验证：`packages/vz-temporal/tests` 107 passed（本包新增 9 个测试，6 个 torch-free）；
  `packages/vz-embodiment-ant/tests` 240 passed / 1 failed（`test_forbidden_list_covers_every_kernel_wheel_module`，
  缺 `volvence_zero.decision_workspace`，属并行 wave 的既存失败）；
  `tests/test_runtime_transition_replay.py` + `tests/test_temporal_interface.py` +
  `packages/vz-runtime/tests` 217 passed；`tests/contracts` 失败/错误集合 = 既存基线
  （4 failed + 5 errors）；ruff 对 8 个改动文件的发现集合与干净 HEAD 完全相同
  （117 条既存，无新增；`S110/S112/E722` 为 0）。
  顺带为上一条的 pass-count 论点提供了当场证据：同一会话内 `tests/contracts` 的 passed
  先后测到 `3682` 与 `3672`（并行 wave 正在增删测试），而失败/错误集合三次完全一致。
  pass 计数不可核验，失败/错误集合可核验。
  **通用默认逐字节不变（已证明，非断言）**：818 行 `float.hex()` 探针覆盖 pure
  `update_causal_action_head`（`n_z∈{3,4,16}` × 梯度尺度 `1e-6/1/1e6` × 120 次更新）、
  `project_causal_action_head_update`（5 baseline × 4 candidate，含包络内/边界/外）、
  permissive restore + snapshot round-trip、`FullLearnedTemporalPolicy.step()`
  （head wiring `None/DISABLED/SHADOW/ACTIVE` 各 6 拍）、torch PPO synthetic lane
  （`n_z × lr × action gain × modulation × track weight × head` 共 128 组）、默认
  eligibility 结算、默认 `build_final_runtime_modules`；干净 HEAD worktree 与本树输出
  `diff` 完全相同，md5 均为 `65517dc30205941b079883fd860422d1`。
  负对照（每项都验证过会失败）：还原 `envelope_enforced=False` 默认写入 →
  `..._leaves_the_store_alone` 以 `assert False is True` 失败；删掉两处 `final_wiring`
  转发 → `..._carries_the_envelope_declaration_to_the_store` 失败；把 torch synthetic 接到
  `latent_code_bounds` → `..._reconstructs_on_the_owner_latent_range` 失败且探针 md5 变为
  `1a8e3ded5a0a1bd19ee5c8695d715c85`（88 行不同）；放回 `1e-9` 容差 →
  `..._has_no_tolerance` 以 `DID NOT RAISE` 失败；去掉绝对 clamp →
  `..._is_conditional_on_the_baseline` 以 `0.35 != 0.1` 失败。
  复验上一包的两项结论：torch head step 仍是**比例**而非 bang-bang（`lr=0.02` 普通 batch
  `bias=1.230494e-05 / out=5.657864e-04 / in=8.364014e-05`；gain `20→40` →
  `1.9910x / 1.9968x / 2.0034x`；`lr` ×25 → 精确 `25.000000x`；gain 1000 与 5000 均精确顶在
  `0.01 / 0.05 / 0.02` 不再增长），与原记录一致。
  **未收敛（本包不改，已单列，两项）**：(a) 上述 pristine head 初始化越界，导致包络强制在任何
  域都还打不开——这是本包唯一没有完全闭合的评审项（配置跳已落地并有测试，声明那一步被挡）。
  〔已由 2026-07-27「给初始先验加界」收敛包关闭，见本日志首条。〕
  (b) `_clamp` 对非有限值做静默 laundering——实测
  `_clamp(nan) = _clamp(inf) = 1.0`、`_clamp(-inf) = -1.0`（`nan < 1.0` 与 `nan > -1.0`
  均为 False，两个界各自保留自身操作数）。一个 NaN 的 `EnvironmentMeasurement.action_payoff`
  因此被当作**最大奖励** `+1.0` 支付，且 `_require_signed_unit_interval` 同样漏过。两条 lane
  laundering 方式相同，故 gate/payout 一致性检查（正确地）保持沉默——这不是该 seam 的职责。
  现状由 `test_exact_agreement_is_not_a_finiteness_check` 记述性锚定，修复需另开收敛包
  （`_clamp` 同时约束 reward/advantage，字节兼容面很宽；应在 ingress 契约处校验）。
- 2026-07-27: 修复 W3-b 自身引入的 bang-bang 更新，并补齐三处 fail-loudly 缺口
  （收敛包 W3-b-fix，对抗评审跟进）。缺陷：W3-b 用“Adam 步 + clamp 回包络”实现 torch
  head 纪律，而 Adam 把每元素步长归一化到约 `lr`，结果是一个只携带梯度符号的最大步长
  控制器——实测 `lr=0.02` 与 `lr=0.5` 给出**完全相同**的 `bias_step=0.010000 /
  out_step=0.050000 / in_step=0.020000`（三个上界同时精确顶满），梯度缩小 1000 倍数值
  仍不变；生产默认 `lr=0.02` 连续 15 次更新在第 10 次把 bias 走到绝对上界 `0.1` 并钉死，
  即包络自己制造了它要阻止的跨状态固定转向截距。对照 pure owner 在同量级信号上的
  `bias_step` 比上界低两到三个数量级。修法：新增唯一 owner
  `causal_action_head_update_scales`，pure 与 torch 两条 lane 共享 `learning_rate /
  batch_size`、`bias_learning_rate_ratio`、`bias_state_path_scale`；torch lane 把 head
  三组参数移出 Adam，改为按该尺度做比例梯度步，投影仅作为上界。证据（`n_z=16`、batch 6、
  `ppo_epochs=1`、`lr=0.02`）：普通 batch `bias_step=1.2305e-5`（上界的 1/813）、
  `out_step=5.6579e-4`（1/88）、`in_step=8.3640e-5`（1/239）；梯度翻倍 → 步长
  `1.991x/1.997x/2.003x`；`lr` 提 25 倍 → 步长精确 `25.000000x`（修复前 `1.000000x`）；
  梯度足够大时三项仍精确顶在 `0.01/0.05/0.02` 且不再增长。另修三处：移除
  `validate_causal_action_head_magnitudes` 的 `1e-9` 容差（冻结上界不加 slack，clamp 产物
  本就精确合法）；`_envelope_bounded_value` 对非有限 **baseline** 也 fail loudly（此前
  `nan`/`inf` baseline 被静默洗成 `+absolute_limit=0.1`，实测复现）；
  `restore_causal_action_head_parameters` 补齐有限性校验。包络开关经
  `MetacontrollerParameterStore.set_causal_action_head_envelope_enforced` 与
  `FullLearnedTemporalPolicy.set_causal_action_head(envelope_enforced=)` 暴露，仍缺
  `FinalRolloutConfig` 一跳（见上文条目，属 `vz-runtime`）。
  验证：`packages/vz-temporal/tests` 83 passed（本文件 58，W3-b 后为 52）；
  `tests/test_runtime_transition_replay.py` + `tests/test_temporal_interface.py` 全绿；
  `tests/contracts` 的**失败/错误集合**等于既存基线（4 failed + 5 errors：`dlaas_dispatch`、
  `feeling_about_other_active_matched_control`、`no_lscb_strings`、`predictive_heads_shadow`、
  `openai_compat_streaming_sse`），均与本包无关。
  （原文此处记为 `3668 passed`。pass 计数不是不变量——并行 wave 每天都在加测试，2026-07-27
  复测同一基线已是 `3682 passed`。可核验的断言是失败/错误集合相等，之后一律记这个。）
  **通用默认逐字节不变（已证明，非断言）**：在 HEAD 的干净 worktree 与本树上运行同一个
  只使用 Wave-3 之前公开 API 的探针（pure `update_causal_action_head` 在 `n_z∈{3,4,16}`
  各 120 次更新 + 梯度尺度 `1e-6/1/1e6`、permissive restore 与 snapshot round-trip、
  `FullLearnedTemporalPolicy.step()` 在 head wiring `None/DISABLED/SHADOW/ACTIVE` 各 6 步、
  torch PPO 在 `causal_action_head_enabled=False` 下 `n_z×lr×action gain×modulation` 共
  16 组），全部以 `float.hex()` 逐位比对：651 行输出 md5 相同
  (`32e218856a52a029355de5fdaa9baf1b`)。
  负对照：把 head 三组参数放回 Adam，`test_torch_head_step_is_proportional_not_bang_bang`
  立刻以 `bias=0.01`（精确等于上界）失败，`..._stays_inside_the_absolute_ceiling_over_time`
  以 15 次更新后 `bias=0.0389` 失败；把 `1e-9` 容差放回，
  `..._has_no_slack_at_the_frozen_bound` 立刻 `DID NOT RAISE` 失败。
- 2026-07-26: torch causal-PPO lane 收敛到冻结 action-head 纪律，并在 restore 侧补齐幅度校验
  （收敛包 W3-b）。三处缺陷：(1) torch lane 把 head 三组参数与 track weights/log_std/critic
  一起塞进同一个 Adam 并逐字节写回，完全绕过 pure owner 的 bias/factor 包络——实测生产默认
  `lr=0.02` 下单次更新把 bias 移动 `0.029`（单步上界 `0.01` 的 2.9 倍），`lr=0.5` 下移动
  `1.32`（绝对上界 `0.1` 的 13 倍）；(2) `restore_causal_action_head_parameters` 只校验 shape，
  任何 archive 都能装入越界参数；(3) runtime-replay lane 硬编码三处 `[-1,1]`，与同函数两条
  synthetic lane 的 `[0,1]` 及 live forward 不一致。修法：把包络常数收敛为
  `CAUSAL_ACTION_HEAD_UPDATE_ENVELOPE` 单一 owner，pure 路径改为消费它（数值逐字节不变），
  torch lane 在每个 optimizer step 后调用 owner 的
  `project_causal_action_head_update`，写回时 `enforce_envelope=True`；latent code 值域收敛为
  `LATENT_CODE_BOUNDS`，torch replay lane 经 `resolve_latent_code_bounds(latent_unit_clamp=)`
  与 pure lane 共享同名契约。三条新行为全部 opt-in 且默认精确回滚：head 未声明时
  `head_parameters is None`，`causal_action_head_envelope_enforced` 与 `latent_unit_clamp`
  默认 `False`。证据：`packages/vz-temporal/tests/test_temporal_contracts.py` 52 测试通过
  （新增 10 个，其中 7 个不依赖 torch）；负对照关掉投影后 torch 包络测试立即以
  `1.3229 > 0.01` 失败，证明该上界是紧的而非空断言。
  **未收敛项**：pure lane 的 PPO surrogate 消费 capture 时固化的 action-head residual
  （`sandbox.py`），head 参数移动时其 `approx_kl` / clip fraction 不会变，而 torch lane 在图内
  重算 residual，两个 backend 因此对同一 batch 报告结构性不同的 KL；收敛它需要改
  `sandbox.py`，本包不拥有该文件。torch lane 也仍未复制 pure 路径的“batch mean 只进 bias、
  factor 只消费 centered covariance”分解，投影只约束幅度。
- 2026-07-24: 冻结调度、探索均值与 runtime actor 方向修复。`joint_learning_enabled=False`
  现在进入 joint-loop owner 的硬边界，带 pending batch 的恢复态也只能发布
  `frozen-evidence-only`；posterior 探索仅扰动 sample residual，不再在 coast 阶段抹平 learned
  mean；runtime fragment 非终局末拍从 next-substrate signature 做 TD bootstrap，causal action
  head 使用带 `0.05` floor 的有界 RMS advantage scaling。证据覆盖冻结 fingerprint、探索 mean
  保留、正负更新方向、可观察更新幅度与 checkpoint round-trip。
- 2026-07-24: 稀疏探索 option horizon 从 24-step（3 burst + 21 coast）收敛为 8-step
  （2 burst + 6 coast）。同一 medium checkpoint、同一 far layout/seed 的精确重放把最近食物
  边界距离从 `0.60` 降为 `0.0` 并由 `0` 变为 `1 pickup`，最大离巢半径保持同量级，证明改善
  来自无目标真值的角度覆盖而非速度/距离泄漏。
- 2026-07-24: 修复 signed hidden 与伪 option continuation。Ndim GRU hidden 的 `[-1,1]`
  坐标不再被 action head 二次变换为 `2*h-1`；8-step sample residual 改为只按
  `segment + dimension` 定义并整段保持。相同 v3 checkpoint/layout/seed 的 4-body medium
  冻结重放由 `0 pickup / 0 delivery` 变为 `2 / 1`，角度覆盖从每 body `2–3` 个扇区提升到
  `6–9` 个，且不增加最大离巢预算、不读取目标方向。
- 2026-07-25: 修复跨 episode 重复探索与 action-head 截距主导。temporal owner 接收可选的
  不透明 exploration context 摘要；Digital Ant 用 episode seed 与 body offset 形成 context，
  matched arms 保持相同而不同 episode/body 分散。action head 同时收紧 bias 的学习尺度与总幅度，
  并恢复 input factor 的状态路径尺度。后续 2-body 25 局门槛确认显式 bias 已降到
  `-0.002…0.017`，但不同 body 只分别学会 food 或 heat；因此进一步把 batch mean 只分配给
  受限 bias，低秩 factor 仅学习 centered state covariance，防止在 factor 内重建隐式截距。
- 2026-07-25: 修复 action head 的训练/执行状态漂移。正式 head state 改为同一 Ndim encoder
  对当前 observation 的零历史编码，live forward、pure/torch replay 与 open-segment checkpoint
  共用该 signed state；serving recurrent hidden 继续服务 metacontroller/value 路径，不再充当
  action-head 输入。`joint_loop.learning` persistence schema 升为 v3。
- 2026-07-25: runtime microbatch 审计确认，按 segment 数即时优化会把大量单步段送入
  centered-gradient optimizer，令 state/signal covariance 严格为零。ACTIVE runtime replay
  改按 transition 数累计，Digital Ant target=4；随机 output-factor 初始化因放大未经经验的
  rank-4 prior 而回退为零，首个非零 covariance batch 建 output path，后续 batch 通过保留的
  output-column norm 归一化反馈训练 input factor。
- 2026-07-25: v14 checkpoint 证明 multi-transition batch 已生效，但 World input-factor 在
  12 局后元素级 L1 只移动 `0.00003–0.00064`；旧 `0.05` column-norm floor 仍把微小 output
  path 的 feedback 压低。改为同批 bounded block-coordinate update：candidate output 先建立，
  input 再按真实列范数归一化回传，二者最后原子提交；零 covariance 仍保持严格 no-op。
- 2026-07-25: v15 使 World input-factor 位移提高到 `0.012–0.055`，但 12 局行为与 v14
  完全相同。追加真实局审计显示每批已有 `n=4/16/17`、covariance L1 `3.6–7.7`，参数变化却
  只有 `0.00007–0.00020`；根因是 pure factor path 仍额外乘 `0.12`，而 torch path 使用 owner
  基础学习率。现仅对 output/input factor 移除该衰减，bias 保留原衰减与上限。
- 2026-07-26: exclusive steering 所有权转移。v21 contrast_pairs 把 head 约束到反对称子空间后，
  v22/v22r 两组受控 forced-approach 实验（固定/随机几何）实测:base policy 在信用竞争中总以
  "放大同向基线转向"这一非定向退化解吸走全部转向信用（基线 0.083→~0.147 rad），head 的
  food→turn 增益钉死 ≈1e-3 且 25 局不增长。新增默认 `False` 的
  `internal_rl_causal_action_head_exclusive_steering`：base 确定性均值在每个 contrast pair 上
  被互补投影为 common mode，head 成为 contrast 轴唯一学习型写入者；live/sandbox/pure/torch
  四路径共用投影，探索噪声保留反对称分量以维持 head 的 PPO 梯度。Digital Ant profile 开启；
  通用默认与历史行为字节不变。同批修复 β 门泄漏：逐维 β 会在已投影候选与对称旧码之间
  凭空造出 contrast，实测零参数 head 仍产生 ±0.005 rad 转向并掩盖同量级的学习信号；
  exclusive steering 下改为按 contrast pair 共享 β。证据：`tests/test_temporal_contracts.py`
  第 5 节（含零参数 head 的 contrast 恒零回归测试）。
- 2026-07-25: v20 的真实 batch 审计确认 posterior sample scale 修复后 score clamp 为
  `0`，food signal 与 steering score covariance、food basis 与参数更新方向均为正；但冻结
  checkpoint 的动作头增益扫描只学到左右相对排序，极限放大时四体只有一体形成绝对对向转向。
  根因是 Digital Ant actuator 只消费 `z[1]-z[0]`，而旧 `(0,1,2)` support 仍允许
  `z[0]+z[1]` common mode 吸收信用并形成 body-specific 固定偏置。现新增通用
  `contrast_pairs` 投影，并把该稀疏 evidence profile 的 head strength 从 `0.35` 提升为
  `1.0`，消除训练与执行两端累计的 `0.35²` 时间尺度衰减；通用默认保持不变。
- 2026-07-20: 有界 posterior 探索。新增默认 `0.0` 的
  `internal_rl_runtime_exploration_strength`；ndim temporal owner 生成带 entropy floor 的可复现
  burst/coast low-discrepancy option，并把有效 posterior mean/std、实际 noise / `z_tilde` 发布给
  runtime replay。用于无 shaping 的稀疏 milestone 探索；不编码环境方向、同实验 matched arms
  同值、dense PE 任务保持 0、可即时回滚。
- 2026-07-20: 真实 runtime transition replay。新增默认关闭、与 optimizer backend 分离的 `internal_rl_runtime_replay` 三态 gate；Internal-RL owner 以 prediction lineage 捕获真实 ndim 动作并在下一拍用 PE-first credit 与 next-substrate effect 结算。ACTIVE 无样本时显式等待且不回退 synthetic，SHADOW 仅记录 coverage；pure/torch likelihood 与 live modulation 对齐，checkpoint 默认覆盖 pending capture/staged rollout，并提供不迁移 episode-local replay 的显式 transfer 导出模式。公共 `EnvironmentOutcome` / PE / credit / temporal snapshot schema 均不变。证据：`tests/test_runtime_transition_replay.py`。
- 2026-07-14: SHADOW 证据完整性补全（GAP-09 / CP-05/06/07）。
  (a) CP-05：`train_store_ssl` 的 `StoreSSLReport` 新增
  `candidate_encoder/switch/decoder_parameters` —— SHADOW 下 store 不动但
  候选 checkpoint 可导出；`MetacontrollerSSLTrainer` 保留 owner-local
  `latest_torch_ssl_candidate` 与 `latest_ssl_forward_parity`（SHADOW SSL
  跑完后对未动的 live params 跑 pure/torch forward parity，不再只有 loss
  标量）。(b) CP-06：`runtime_ndim_shadow_compare` 新增行为级对比维度：
  `switch_decision_pure/torch/match`（beta≥threshold 的 segment-closure
  决策）与 `nearest_family_pure/torch` + `family_selection_match`（applied
  latent 最近 action family），call site 传入 store 的 beta_threshold 与
  action_families。(c) CP-07：`InternalRLEnvironment` 发布
  `latest_reward_composition`（含 `pe_derived_abs_fraction`，PE/segment-
  credit 派生分量占 |reward| 的比例，闭集组件名单由 reward owner 定义）；
  `run_internal_rl_proof`（optimize vs no-optimize matched control）经
  `collect_internal_rl_no_optimize_proof` 接入 learned-shadow soak artifact。
  证据面经 `learned_shadow_evidence.collect_learned_shadow_evidence` 统一
  导出。测试：`packages/vz-runtime/tests/test_learned_shadow_evidence.py`、
  `packages/vz-temporal/tests/test_temporal_contracts.py`。
- 2026-07-12: ndim runtime forward 的 `SHADOW` 语义补齐为真实 per-step
  pure/torch 双跑。`FullLearnedTemporalPolicy` 保持 pure path 为唯一 live
  writer，并发布 owner-local `latest_runtime_shadow_report`，比较同一轮的
  previous hidden/code、CMS context、memory/reflection、family continuation
  与 external switch pressure；报告只用于 parity/latency evidence，不进入
  `temporal_abstraction` 快照，也不改变 action-family 状态。回滚到
  `DISABLED` 会清空报告并停止双跑。
- 2026-07-12: `BrainConfig.temporal_latent_dim` /
  `AgentSessionRunner(temporal_latent_dim=...)` 成为 controller capacity 的
  显式入口。默认 3 保留 legacy rollback；evidence profile 可使用
  16/64/256 解锁 ndim controller，而不修改生产默认或公共快照 shape。
  小于 3 的配置 fail loudly；bootstrap snapshot 仍由 temporal owner 自身
  决定维度。
- 2026-07-12: autograd operator wiring 拆成 owner-local 三态环境变量：
  `VZ_TEMPORAL_SSL_BACKEND` / `VZ_TEMPORAL_RUNTIME_BACKEND` /
  `VZ_INTERNAL_RL_BACKEND` / `VZ_CMS_TORCH_BACKEND`，每项只接受
  `disabled|shadow|active`，非法值 fail loudly。单项值优先于旧的
  `VZ_TORCH_BACKENDS=active` 全开快捷方式，使四个 owner 能按
  DISABLED→SHADOW→ACTIVE 独立收敛和回滚；默认值仍全部 DISABLED，
  本变更不构成 ACTIVE 晋升证据。

- 2026-07-20: reward→code 桥（runtime track modulation）。修复诊断出的结构性断链——Internal-RL 写 `track_weights`（+ `align_temporal_from_tracks` → legacy `temporal_weights`/`switch_bias`），但 ndim runtime forward 产生 `code` 只用 `ndim_encoder/switch/decoder_parameters` + `beta_threshold`，`track_weights` 仅进辅助 `track_codes`，故奖励驱动学习对 `z_t` 零传导（探针：`internal_rl_backend` ACTIVE/DISABLED 与 `joint_apply_writeback` on/off 下 `z_t` 逐字节相同；只有 SSL 移动 `z_t`）。实现：`MetacontrollerParameterStore.runtime_track_modulated_code`（逐维、中心 1.0、界 [0.5,1.5] 的增益）；`FullLearnedTemporalPolicy._runtime_track_modulation` + `set_runtime_track_modulation`，在 `_step_impl_ndim` 于 `z_candidate` 上 gated 调制；`FinalRolloutConfig.internal_rl_runtime_modulation_strength`（默认 `0.0`）经 `AgentSessionRunner` 注入 world/self policy。随后完成 sandbox 对齐：`CausalZPolicy` 的行为分布、pure surrogate/gradient 与 torch PPO 都使用同一 aggregate track gain；causal override 被定义为已调制 final candidate，live forward 不二次调制。`strength=0` 精确保留历史 runtime/sandbox/torch 方程；`strength>0` 时 PPO writeback 会改变真实 ndim `code`。蚂蚁探针确认 `strength=0.5` 时 `z_t` 8/8 拍与基线不同。默认 0.0 不构成 ACTIVE 晋升证据。证据：`tests/test_temporal_contracts.py` 15 测试通过（含 sandbox/live 等式、double-modulation guard、PPO→live-code 因果测试）。
- 2026-07-04: protocol-temporal-prior bridge。把 BehaviorProtocol `active_mixture` 接成 metacontroller 的 `beta_t` switch-pressure prior，闭合此前 `protocol-runtime.md` 消费者表声明但 temporal 侧未接线的落差（metacontroller 从未消费 `active_mixture`）。因 `active_mixture` 在 propagate DAG 中处于 temporal 下游（经 `retrieval_policy`），采用 orchestrator-mediated 上一轮 carryover（对齐 `experience_fast_prior` 先例、`experience_consolidation` out-of-band 注入模式），避免 `temporal→active_mixture→retrieval_policy→temporal` 环。实现：`MetacontrollerParameterStore.record_protocol_prior_signals` / `protocol_prior_switch_pressure_delta`（owner-side 私有标量，不改 frozen telemetry schema）；`TemporalPolicy.observe_active_mixture` + `set_protocol_prior_enabled`（base no-op，`FullLearnedTemporalPolicy` 实现：dominance→continuation / ambiguity→switch，bounded ±0.18，SHARED track ×0.85）；`TrackTemporalModule` / `TemporalModule` 新增 `observe_active_mixture_carryover`。runtime：`FinalRolloutConfig.protocol_temporal_prior`（默认 DISABLED）经 `build_final_runtime_modules` set enable flag、经 `run_final_wiring_turn` 从 `upstream_snapshots` 取上一轮 `active_mixture` 喂入。证据：`tests/contracts/test_protocol_temporal_prior.py`（12 测试，含 3 个固化 dual-run 证据：方向因果正确 baseline-vs-active、有界、可跨过 β_t 二值切换阈值翻转 `is_switching`）；默认 DISABLED 下 300 temporal/wiring/protocol 测试零回归。dual-run 观测：ambiguous 混合 prior=+0.12 推高 β_t（更爱切换，turn0 由 0.498→0.618 跨过 0.55 阈值），dominant 混合 prior=−0.072 压低 β_t（更爱延续），方向 8/8 正确。
- 2026-06-29: autograd-owner-integration 上线形式 —— torch backend 现可经 `FinalRolloutConfig` 运行时配置:`temporal_ssl_backend` / `temporal_runtime_backend` / `internal_rl_backend` / `cms_torch_backend`(默认全 `DISABLED` = 行为不变,回滚即重置为 DISABLED)。config 经 `AgentSessionRunner` + `build_final_runtime_modules` 线进 `FullLearnedTemporalPolicy.set_runtime_backend`,经 `ETANLJointLoop(ssl_backend=, internal_rl_backend=)` 线进 SSL trainer / `InternalRLSandbox`,经 `build_default_memory_store(cms_torch_backend=)` 线进 `CMSMemoryCore`。证据:`tests/test_autograd_backend_deploy_wiring.py`。
- 2026-06-29: autograd-owner-integration —— 把 torch 路径从 sidecar 接入 owner 主链。`MetacontrollerSSLTrainer(ssl_backend=WiringLevel)` 经 `temporal/torch_store_ssl.py` 直接对 store 的 `Ndim*Parameters` 做真 autograd（ACTIVE 写回同一 store；SHADOW 仅证据），`SSLTrainingReport` 追加 `torch_*` 证据字段。`FullLearnedTemporalPolicy(runtime_backend=WiringLevel)` 经 `temporal/backend_ndim_runtime.py` 在 `_step_impl_ndim` 路由 encode/switch/decode（ACTIVE 走 torch backend；`runtime_ndim_shadow_compare` 给 pure↔torch parity+latency gate）。`InternalRLSandbox(rl_backend=WiringLevel)` / `CausalZPolicy` 经 `internal_rl/torch_causal_ppo.py` 对真实 `ZTransition` batch 做 PPO autograd（ACTIVE 写回 track weights+critic），`OptimizationReport` 追加 `torch_*` 字段，checkpoint/rollback 不变。修正：ndim 切换 `gate_input = delta + z_tilde` 实为 tuple **拼接**（2·n_z 维），backend/SSL torch 路径已对齐。新增 `temporal/torch_metacontroller.run_strict_eta_evidence` 在受控分层 suite 上严格证明 alpha 单调驱动 switch sparsity + held-out family reuse。默认仍 DISABLED（纯 Python 基线）。
- 2026-06-29: NL/ETA 全量真 autograd 迁移 Phase 0–3 落地。新增 `volvence_zero.tensor_backend`（pure/torch backend 抽象 + parity + 确定性，vz-contracts）、`torch_metacontroller`（Eq.3 对齐 `D_KL(q‖N(0,I))` + STE 切换 + 真 backprop SSL + KL-target matched ablation + artifact roundtrip）、`internal_rl/torch_internal_rl`（真 PPO autograd z-policy + critic + GAE + matched control）、`backend_metacontroller`（runtime backend-agnostic 前向 + SHADOW 双跑 parity + 延迟 gate + WiringLevel 路由）。纯 Python 路径降级为回滚基线，torch 默认 DISABLED。
- 2026-04-25: ETA proof harness 新增 `transformers-open-weight` real residual evidence lane、prefix-step real snapshot contract、open-weight paper-suite manifest 与 runtime gating 口径；真实 residual-control claim 不再只依赖 synthetic proof harness
- 2026-05-22: SYS-1 最小切片。新增 temporal owner 内部的 read-only `CPDSwitchReadout`，把 PE spike + reward shift 转成 beta switch evidence；不新增 runtime slot，不直接改 live switch gate。
- 2026-04-26: real residual evidence 口径细化：actual hook fire rate 与 planned layer fraction 分离，proof rollout 改为 prefix-aligned intervention，并新增 frozen residual signature calibration
- 2026-04-22: 补充 scaffold-ablation matched controls（`pe-eta-no-semantic-label`、`pe-eta-no-reflection-cache`）的当前 proof 口径，用于测试去掉 heuristic scaffold 后的 latent family / PE schedule 稳健性
- 2026-04-22: 当前实现口径补充 dual-track cached-reflection bridge 与 latent-family style fallback labels，进一步把 live path 从 heuristic semantics 收紧到 family/state evidence
- 2026-04-20: 接口契约补充 `prediction_error` 直接输入；当前实现口径明确 `TemporalModule` 已直接消费 `prediction_error` slot，而不再只经 evaluation 旁路感知高 PE
- 2026-04-09: next_gen_emogpt v2 terminology alignment: paper term `subgoal` mapped to repo term `abstract action` as default; `z_t` = controller code, `beta_t` = switch gate, `U_t` = decoder output / residual controller; two-stage (SSL then Internal RL) made non-optional constraint; non-causal → causal transition explicitly documented as design invariant
- 2026-04-09: U03 Emergence vs Heuristic A/B verification: (1) Switch gate: alpha=0.1 vs alpha=0.0 produces different loss profiles, confirming variational bottleneck affects switch behavior. (2) Family competition: payoff-weighted ranking prefers high long_term_payoff families over similarity-only selection when centroids are equidistant. (3) NonCausalSequenceEmbedder.enrich_posterior confirmed to reduce posterior variance (enriched_var <= causal_var) and produce positive kl_tightening. Bidirectional ordering sensitivity verified.
- 2026-04-06: 补充 learned-lite temporal policy 的当前实现口径，并记录 runtime-visible metacontroller owner state
- 2026-04-06: 补充 full-learned metacontroller owner、sequence-aware substrate 输入与 runtime-visible training state
- 2026-04-06: 补充 explicit posterior、learned switch stats、bounded decoder control、Eq.3-style SSL 与 causal replacement rollout 的当前实现口径
- 2026-04-06: P16 Non-Causal Sequence Embedder: bidirectional GRU-based s(e_{1:T}) encoder for training-time posterior inference. Creates information asymmetry: training posterior q(z_t|e_{1:T}) uses full sequence via NonCausalSequenceEmbedder.enrich_posterior, while runtime policy π(z_t|e_{1:t}) only sees causal prefix. SSL trainer now reports noncausal_kl_tightening and noncausal_information_content.
- 2026-04-06: P15 N-dim Tensor Core: introduced configurable n_z latent dimension (default 16 for new policies, backward-compatible at n_z=3). NdimSequenceEncoder uses real GRU cell; NdimSwitchUnit produces element-wise β_t via learned FFN gate; NdimResidualDecoder uses 2-layer FFN with tanh. MetacontrollerParameterStore, CausalZPolicy, SSL trainer all support arbitrary n_z. tensor_ops.py provides pure-Python linear algebra: mat_vec, GRU cell, FFN, sigmoid, tanh.
- 2026-04-06: P10 CMS-enhanced encoder: SequenceEncoder now accepts cms_online_fast/session_medium/background_slow bands; prior mean/std shaped by CMS slow bands; bidirectional encoder↔CMS feedback via encoder_output_for_cms and CMSMemoryCore.observe_encoder_feedback; final_wiring feeds encoder output back to CMS
- 2026-04-06: 补充 Gaussian-like prior/posterior、closed-form KL 与 residual-control application helper 的当前实现口径
- 2026-04-06: 补充 residual intervention backend 契约与 causal-binary rollout path 的当前实现口径
- 2026-03-25: 初始版本，从 SYSTEM_DESIGN.md 和 next_gen_emogpt.md 提取

## 变更日志补充

- 2026-07-29: learned backend 终态链 gate 补全。新增
  `LearnedPromotionChainVerdict` / `evaluate_learned_active_chain(...)`，
  要求 terminal evidence 对 runtime / SSL / Internal-RL / CMS 四组件各有且
  仅有一行，并把 `terminal_candidate_ready`（允许在隔离环境直接验证四项终态
  wiring）与 `production_terminal_ready`（四组件已按固定顺序全部 ACTIVE）
  分开。证据采集可并行、终态候选可一次测试；生产晋级状态必须是
  runtime→SSL→Internal-RL→CMS 的有序前缀，报告只为首个未 ACTIVE 组件生成
  `recommended_env`。缺行、重复行和非前缀 ACTIVE 声明均 fail loudly；默认
  wiring 不变，回滚仍为把单项 backend 降回 SHADOW / DISABLED。
- 2026-07-13: CP-15 / CP-23 ACTIVE candidate gate evaluator. 新增
  `volvence_zero.agent.learned_active_gate`：`LearnedActiveEvidence` /
  `LearnedActiveGateVerdict` / `evaluate_learned_active_candidate(...)`，将
  runtime→SSL→Internal-RL→CMS 的逐字段晋升顺序、500 turn 真 trace、validation
  delta >= 0.02、PE-off / ETA-off 对照方向、rollback drill、latency/safety、
  Internal-RL reward leakage、CMS retention/absorption gate 固化为 typed 判定。
  该 evaluator 不翻默认 wiring；缺证据时返回 blocked。测试：
  `packages/vz-runtime/tests/test_learned_active_gate.py`。
