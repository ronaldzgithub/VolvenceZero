# 时间抽象与内部控制 Spec

> Status: draft
> Last updated: 2026-04-25
> 对应需求: R3, R4

## 要解决的问题

如何让系统在 token 之上的抽象层级做决策和学习，实现子目标级控制？

## 关键不变量

- 实时行为可通过内部状态转换引导，而非仅通过表面文本损失
- 抽象动作可组合、可训练、无需详尽手动标签
- 冻结基础模型是发现时间抽象的前提（ETA rate-distortion 证明）
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

**产出的输出**：
- `temporal_abstraction` 快照：`TemporalAbstractionSnapshot`
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
- 当前 `TemporalModule` 默认以 `full-learned` 作为 runtime owner policy，并可通过 owner API 导出 machine-readable metacontroller runtime state；这条导出链不改变 `temporal_abstraction` 公共 snapshot schema
- 当前 temporal owner 已支持显式消费 `experience_fast_prior`：它不会直接接收 session-post 对私有参数的写穿，而是把 public fast prior 转成 owner-side continuation bias 与 switch-pressure delta，再在 `step()` 内部作用于 action family continuation 和 switch unit
- 当前这条 owner-side fast-prior consumption 已进一步前推到 action-family discovery / competition：public fast prior 不直接指定 family，而是作为 owner 内部的 `reuse/create/split/competition` 偏置，影响当前 active family 的 continuation、competition score、monopoly / stagnation 调节
- 当前 temporal owner 对 `experience_fast_prior` 的消费已从“轻量 telemetry bias”收紧到更直接的 continuation / exploration pressure：`action_bias`、`family_bias`、`sequence_bias` 与 `prior_strength` 会共同塑造 `switch_pressure_delta`，并放大 owner-side family continuation / competition 信号，使 application delayed credit 能更直接进入 ETA fast path，而不只停留在检索 readout 或 evaluation evidence
- 当前 ETA proof benchmark 也已接入 temporal fast prior：训练 rollouts 会把 delayed credit / family reuse / sequence completion 压缩成一个小的 temporal fast prior，再回灌到后续 proof rollouts；benchmark 会显式发布 `temporal_fast_prior_strength` 与 `temporal_fast_prior_switch_delta`
- 当前 proof profile 已包含 matched ablation `full-no-fast-prior`：它保留 full internal RL + causal replacement，但关闭 temporal fast prior ingestion，用于衡量 fast prior 对 held-out family reuse、credit alignment 与 strong success 的增益
- 当前 runtime 已新增 `full-learned` metacontroller owner：内部采用 sequence encoder + learned switch unit + residual decoder 的最小可执行实现，优先消费 `substrate.residual_sequence`
- 当前 `AgentSessionRunner` 默认已切到 hook-shaped residual substrate adapter；默认 session turn 会优先发布 `SurfaceKind.RESIDUAL_STREAM` 而不再停留在纯 trace-sim feature adapter
- `learned-lite` 仍保留为 fallback / rollback baseline；`full-learned` 是当前默认 temporal owner
- 当前 online owner 的 rollback 已提升到 cycle 级：坏周期会恢复到 SSL 之前的 checkpoint，保证 temporal owner 不留下半轮 SSL/RL 混合脏状态
- 当前 rare-heavy v0 允许 temporal owner 导出/导入 parameter snapshot：offline pipeline 负责产出 artifact，runtime owner 负责 apply / rollback，不引入第二个 temporal state owner
- metacontroller runtime state 已扩展为可发布 prior mean/std、posterior mean/std、posterior sample noise、`z_tilde`、posterior hidden state、posterior drift、decoder output / applied control、latest switch gate，以及 binary switch ratio / sparsity / persistence window 等 owner-visible ETA 证据；当前还会显式发布 `active_label` 对应的 discovered family、`learning_phase`、`structure_frozen`、family summary/version
- discovered action family 当前已不再从固定 seed prototype 起步，而是从空 bank 开始，并在 temporal owner 内执行 bounded `reuse/create/split/merge/prune`
- `TemporalAbstractionSnapshot` 当前新增 `action_family_version`，作为 `temporal -> dual_track -> regime/evaluation` 的最小版本桥，不把完整 family bank 暴露给 consumer
- 当前 `temporal` owner 已补充 family competition memory：owner 内部会持续维护 `reuse_streak`、`stagnation_pressure`、`monopoly_pressure`、`competition_score`，并用这些竞争状态影响反塌缩的 topology maintenance
- 当前 public runtime state 只发布 compact family competition summary（如 active-family competition score、monopoly pressure、turnover health、family version/count），不发布 raw internal competition ledger；这条 bridge 为下一阶段 delayed credit ledger 预留了显式版本锚点
- 当前 `full-learned` 已把 `z_t` owner 更新规则收敛到显式 posterior + learned switch 路径：`z_t = beta_t * z_candidate + (1 - beta_t) * z_{t-1}`，其中 `z_candidate` 默认来自 posterior `z_tilde`，也可由 internal RL causal policy override
- 当前 decoder 已升级为 bounded FFN-like control generator；环境侧显式区分 `decoder_output`、`applied_control`、`downstream_effect`
- 当前 SSL trainer 已改成更接近 Eq.3 的结构：prefix posterior inference + Gaussian-like prior regularization + action prediction + closed-form KL，并发布 posterior drift
- 当前两阶段 ETA owner 约束已从 telemetry 收紧为**运行时守卫**：`MetacontrollerSSLTrainer.optimize()` 只能在 `ssl*` discovery phase 下运行；causal override / internal RL rollout / optimize 只能在 `runtime` / `rl*` 等 structure-frozen takeover phase 下运行
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
- 当前 `internal_rl` 已新增 paper-like proof mode：`InternalRLProofEpisode / InternalRLProofSubgoal / InternalRLDelayedCreditAssignment` 允许用结构化分层 sparse-reward episode 驱动 rollout，而不改变 live session 默认 dense reward 语义
- 当前 `InternalRLSandbox` 在 proof mode 下已支持 sequence-aware causal observation（优先读 `residual_sequence` 摘要）与 delayed-return optimization；这条路径用于验证 internal RL 是否真的在抽象动作层解决延迟奖励，而不是只在 turn 级 dense shaping 下工作
- 当前 ETA proof harness 已新增 real open-weight evidence lane：`transformers-open-weight` backend 通过 `OpenWeightResidualRuntime.capture/apply_control` 生成真实 residual `SubstrateSnapshot` 序列；time step 暂定为 source prefix steps，快照内显式发布 `eta_real_runtime_step_index`、`eta_real_runtime_capture_present`、`eta_real_runtime_fallback_active`、`eta_real_runtime_intervention_protocol_valid` 与 runtime hook evidence。`trace` / `synthetic-open-weight` 仍保留为 matched fallback/control，不再承担真实 residual-control claim 的唯一证据面；real claim 现在要求 fallback rate 为 `0.0`、actual hook fire rate 至少 `0.75`、prefix capture 与 intervention source 对齐。`planned_layer_fraction` 只是选层比例诊断，不再被误当作 hook 健康度
- 当前 real open-weight proof path 会用 frozen real residual prefix captures 校准 proof subgoal signatures，并把 observation alignment / intervention effect 作为 diagnostic reward components 发布；这些 diagnostic components 不进入 sparse optimizer-visible reward
- 当前 `run_eta_open_weight_residual_benchmark()` 与 `build_eta_open_weight_paper_suite_manifest()` 提供从 proof harness 向 open-weight runtime 过渡的研究原型入口：它不改变 live session 默认 dense reward 语义，也不打开 live substrate mutation；默认仍遵守 frozen substrate doctrine
- 当前 ETA strong-proof benchmark 默认不再只比较混合 baseline，而是使用 matched controls（`full-no-optimize` / `full-no-replacement` / `learned-lite-causal` / `noop-backend`）分别隔离 RL 更新、latent replacement、controller capacity 与 backend intervention effect
- 当前 `InternalRLSandbox.optimize()` 已补充 parameter-change evidence：proof benchmark 会记录 training-time `parameters_changed` / `training_parameter_change_rate`，避免把“最终 success 更高”误写成“internal RL 确实发生了 policy adaptation”
- 当前 ETA strong-proof benchmark 已从 dialogue PE harness 中显式分离：新的 proof harness 关注 hierarchical sparse-reward、abstract-action family reuse、held-out composition 与 delayed credit alignment，而不把这些结论混写进普通 `temporal_abstraction` runtime slot
- 当前 causal z-policy 已不再只发布 proxy score：runtime rollout 现在会显式记录 `policy_mean` / `policy_std` / `policy_noise` / `log_prob` / `value_estimate`
- 当前 `CausalZPolicy` 已从单条 rollout 更新扩到 batch rollout 更新；PPO-like surrogate、KL 与 clip 现在围绕显式 stochastic z-policy 分布计算，而不是只围绕 synthetic score
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
