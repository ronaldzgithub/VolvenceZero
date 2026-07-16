# Learned vs Heuristic Coverage Spec

> Status: v0.1
> Last updated: 2026-07-16
> 对应需求: R2（稳定基底 + 自适应控制器）、R3/R4（时间抽象 + 内部控制）、R-PE（prediction error 为原始学习信号）、R8（快照 SSOT / 单一所有者）、R9（层级信用）、R15（可解释 + 可回滚迁移）

## 要解决的问题

认知闭环（PE → credit → `z_t`/`β_t` → action family → reflection → memory）的**骨架、契约、快照隔离、回滚机制是真实且完整的**，但闭环里真正承担"学习"的部件占比**从未被量化**。本 spec 是该占比的 SSOT：逐 wheel、逐 owner 把每个关键决策点标注为四类之一，给出以 `file:line` 为证据的 learned 覆盖基线，并把每个"应该 learned 但当前硬编码"的决策点显式链到既有 known-debt。

这直接回答融资尽调命门问题——"内核里到底有多少是学出来的、多少是写死的、哪些本该学却硬编码"——并作为 [known-debt #86](../known-debts.md) 修法 1 的落地产物，同时是 [#88](../known-debts.md)–[#91](../known-debts.md) 四条学习卡点的上位盘点表。

**本 spec 不改任何运行时行为**：它是一份诚实度 + 可演化性文档，不是新 wiring。

## 关键不变量

- **证据即 SSOT**：每条分类以 `file:line` 为准；改对应代码时，先更新此处证据行，再改分类。文档过时以代码为准（对齐 first-principles 规则"代码优先，随后修 spec"）。
- **frozen-substrate 消费不得成为第二 owner**（R8）：读冻结基底 feature_surface 做输入是允许的；把基底状态在下游重建则违规。
- **`should-be-learned-but-hand-crafted` 项必须挂钩**：每一条要么链到一条 known-debt，要么链到一个升级 gate；**禁止**用 `if keyword in text` / 硬编码 regime_id / 固定映射表**默默替代**系统应当学习的东西（对齐 `no-keyword-matching-hacks.mdc`）。
- **升级走 `WiringLevel` 三态**（R15）：任何"从 hand-crafted 提升到 bounded-learned"的改动，必须 `SHADOW`（并跑）→ 比对快照 → `ACTIVE`，且可回滚。
- **占比口径不得单列**：learned 占比必须同时给"按决策点计数"和"按运行时适应影响力"两个 lens（见下），单列任一个都会误导。

## 标注 taxonomy

| 类别 | 定义 | 本内核典型实现 |
|------|------|----------------|
| `hand-crafted` | 固定公式 / 阈值 / 权重 / if-elif 分支 / EMA·running-stats；人写死、运行时不学 | facet `+5` tie-breaker、regime 打分线性系数、reflection 阈值、β_t `0.55` 切换阈值 |
| `bounded-learned` | 有界在线更新的真实 learned 部件（线性 head / 小 MLP·GRU / z_t RL），带 validation gate 且可回滚 | `_PELearnedCritic`、`_RewardingStateHead`(COCOA)、CMS `LearnedUpdateRule` band 更新、PEFT/LoRA delta |
| `frozen-substrate` | 只读冻结基底表征 / feature_surface 作为输入 | substrate residual hook 抽 hidden、`feature_signal_value` 输入、metacontroller `z_t` digest |
| `should-be-learned-but-hand-crafted` | 设计上应从数据涌现，但当前是硬编码规则 / stub / 静态映射 | `score_regimes` 系数、`apply_metacontroller_evidence` 硬编码 regime_id、stub embedding、external-outcome 静态 bias 表、apprenticeship stub-token Jaccard |

## 口径（两个 lens）

- **by-count（决策点计数）**：把每个关键决策点当 1 票统计。反映"表面上有多少地方是学的"。
- **by-influence（运行时适应影响力）**：按"该决策点对系统在线适应结果的实际权重"加权。反映"真正驱动学习的到底是谁"。

两者差异极大：例如 `vz-memory` 决策点计数里 bounded-learned 只占约 1/3，但按在线适应影响力，CMS 的 `LearnedUpdateRule` band 更新几乎主导了记忆侧的全部在线学习。

---

## 分 wheel 决策点注册表

> 说明：所有行号基于 2026-07-04 仓库状态，由逐文件核查得出。分类反映**默认 wiring**（`FinalRolloutConfig` 默认值）下实际执行的 Python 路径，而非"若全部 backend ACTIVE 时"的理论态。

### 5.1 `vz-substrate`（冻结基底 + 有界 adapter-delta）

| 决策点 | 分类 | 证据 |
|--------|------|------|
| `SubstrateSnapshot.is_frozen` 默认 True（基底行为级冻结） | `frozen-substrate` | `substrate/adapter.py:100`（Placeholder）· `:146`（FeatureSurface）· `:421`（OpenWeightResidualStream） |
| 默认 session adapter = OpenWeightResidualStream | `frozen-substrate` | `substrate/session_observation.py:102-105` |
| residual 真实 hidden-state 抽取（forward hook，no-grad capture） | `frozen-substrate` | `substrate/residual_backend.py:1198-1234`（`_capture_with_hooks`）· `:125`（`is_frozen=True`） |
| `semantic_feature_surface_from_text`（blake2b hashing + 固定 `_SEMANTIC_PROTOTYPES` cosine） | `should-be-learned-but-hand-crafted` | `substrate/adapter.py:201-344` |
| `SyntheticOpenWeightResidualRuntime`（trace 模拟 + hashed embedding，占位 residual） | `hand-crafted` | `substrate/residual_synthetic.py:86-193` |
| `SurfaceKind.PLACEHOLDER` 空 snapshot | `hand-crafted` | `substrate/adapter.py:95-131` |
| 有界 adapter-delta（`_clamp_delta_vector` limit≈0.18/0.12，online-fast / rare-heavy） | `bounded-learned` | `substrate/residual_helpers.py:227` · `residual_backend.py:1650-1675` |
| PEFT / persona LoRA delta（有 checkpoint 时，LRU cache，R2 冻结 base） | `bounded-learned` | `substrate/persona_lora_pool.py:75-87,229+` · `peft_adapter_cache.py:1-19,120-148` · LoRA hook 不改 base state_dict：`residual_backend.py:180-188` |
| rare-heavy adapter 训练（2026-07-16 S1：可注入 `RareHeavyAdapterTrainingBackend` 接缝——`PeftLoraRareHeavyBackend` 真 PEFT LoRA 训练（冻结 base、只训 LoRA 矩阵、causal-LM loss、`B@A` 投影为 hidden 宽 per-layer delta），经 `BrainConfig.rare_heavy_training_backend="peft-lora"` 安装、`clone_for_rare_heavy` 传播，`session_training_phase` 的 replay 比对 + ModificationGate 编排不变；未注入 = 内置 adapter-delta autograd loop（transformers）/ 启发式统计调权（synthetic）为文档化回滚 fallback；backend 失败 fail loudly 不静默回退） | `bounded-learned`（注入时；默认 `builtin`） | `substrate/rare_heavy_training.py` · `residual_backend.py` / `residual_synthetic.py`（`set_rare_heavy_training_backend`）· `brain.py`（`_install_rare_heavy_training_backend`）· `tests/contracts/test_rare_heavy_training_backend.py` |

### 5.2 `vz-temporal`（时间抽象 / metacontroller）

> **默认路径关键事实**：`final_wiring.py:1357` 用 `FullLearnedTemporalPolicy()`，但 `MetacontrollerParameterStore` 默认 `n_z=3`（`interface.py:406`）→ 走 `_step_impl_legacy`，**ndim GRU/FFN 根本不实例化**；且三个 torch backend 默认全 `DISABLED`（`final_wiring.py:222-224`）。因此"FullLearned"在默认配置下实际是**可更新参数 store + 纯 Python 启发式算子**，不是 ETA 论文级 torch autograd。

| 决策点 | 分类 | 证据 |
|--------|------|------|
| 默认策略类（名义 FullLearned，实际 legacy n_z=3） | `hand-crafted`（默认执行态） | `final_wiring.py:1357` · store `n_z=3`：`interface.py:406` |
| runtime forward 后端解析（仅 `ACTIVE` 时返回 torch backend，否则 legacy） | `hand-crafted`（默认 DISABLED） | `interface.py:2023-2027`（`_resolve_backend_ndim_mc`） · `final_wiring.py:223` |
| β_t 切换（legacy `SwitchUnit`：加权 delta+std+memory/reflection；阈值自 2026-07-16 起由 store `beta_threshold` 注入，0.55 仅为初始值，`temporal-prior` 组在线可更新） | `hand-crafted`（gate 公式）+ 阈值参数化 | `metacontroller_components.py`（`compute_decision(beta_threshold=…)`）· `interface.py`（legacy/ndim step 均传 `store.beta_threshold`） |
| z_t 编码（legacy `SequenceEncoder`：3 维 recurrence + CMS 0.5/0.3/0.2 混合） | `hand-crafted` | `metacontroller_components.py:192-248` |
| z_t 解码（legacy `ResidualDecoder`：固定矩阵 + 0.65/0.35 blend） | `hand-crafted` | `metacontroller_components.py:432-441` |
| action family 匹配（`_family_match_score`：2026-07-16 起为 learned 匹配 head——`FamilyMatchWeights` 由 store 拥有、settled family outcome 驱动有界 SGD，历史固定系数降级为初始化，权重被 `FAMILY_MATCH_WEIGHT_ENVELOPE` 回滚包络约束） | `bounded-learned`（初始=历史系数，回滚=reset 到 `DEFAULT_FAMILY_MATCH_WEIGHTS`） | `metacontroller_components.py`（`FamilyMatchWeights` / `family_match_features` / `update_family_match_weights`）· `interface.py`（`observe_family_outcome_feedback` settlement） |
| `Ndim{SequenceEncoder,SwitchUnit,ResidualDecoder}` GRU/FFN（`DEFAULT_N_Z=16`） | `bounded-learned`（**默认未实例化**，仅 `n_z>3`；2026-07-16 起可经 `BrainConfig.temporal_profile="learned-ndim"` 一键选择，默认仍 legacy-3dim 待证据门） | `metacontroller_components.py:1238` · `interface.py:1987-1995` · `brain.py`（`TEMPORAL_PROFILE_LATENT_DIMS`） |
| `TorchMetacontroller`（GRU + STE switch + 真 `torch.autograd` SSL Eq.3） | `bounded-learned`（**默认 DISABLED**，非 facade 导出） | `torch_metacontroller.py:126-133` · `ssl.py:428-431` |
| `TorchCausalZPolicy` + PPO/GAE（Internal RL on z_t） | `bounded-learned`（**默认 DISABLED**） | `internal_rl/torch_internal_rl.py:75-144` · `sandbox.py:257,979-992` |
| Internal RL 主路径（默认）= 线性 policy head + `math.sin` 伪噪声 + analytic PPO-shaped 更新 | `hand-crafted` | `internal_rl/sandbox.py:521-541,624+` |
| joint loop SSL→RL（PE 门控：仅 `pe_magnitude>=0.6` 触发全 cycle；默认 M3 启发式 SSL + 伪 PPO） | `hand-crafted`（含小 meta-rule `LearnedUpdateRule`）；2026-07-16 起叠加 **report-only SHADOW learned gate**（`ScheduleGateLearner`：有界 logistic head，以 SSL prediction-loss 改善 settle，遥测发布 `learned_gate_*`，live 决策仍为规则级联） | `joint_loop/scheduling.py:143-146` · `runtime.py`（`_settle_schedule_gate`）· `gate_learner.py` |

**卡点链接**：本 wheel 的 bounded-learned 部件几乎全部默认 `DISABLED`/未实例化 → [#88](../known-debts.md)（ETA torch 后端默认 DISABLED）。

### 5.3 `vz-memory`（连续记忆）

| 决策点 | 分类 | 证据 |
|--------|------|------|
| CMS 三 band 更新门控（write_gate / step_scale / momentum / slow_mix / reset_mix） | `bounded-learned` | `memory/cms.py:460-486`（`LearnedUpdateRule.decide` → `_decide_band_update`）· 规则体 `vz-contracts/learned_update.py:286-366` |
| Band 基础超参（online_lr=0.65 / session_lr=0.3 / background_lr=0.1 / momentum=0.9 / cadence 2·4 / anti_forgetting=0.1） | `hand-crafted` | `memory/cms.py:82-88,111-115` |
| 梯度式向量/MLP 更新公式（error→momentum→apply，`bias_delta*0.02`，手工 backprop） | `hand-crafted`（门控 learned，公式固定） | `memory/cms.py:1840-1859` · `cms_band_mlp.py:121-159` |
| `pe_features_enabled`（Titans PE 4 维进 LearnedUpdateRule） | **运行时 `bounded-learned`**（factory 默认 ON）；裸 `CMSMemoryCore` 构造默认 `False` | **运行时 `build_default_memory_store` 传 True**：`memory/store.py:76,112`；裸构造默认 False：`memory/cms.py:92,116` |
| ATLAS replay（K/gamma 手工默认；replay buffer 加权 joint step） | `hand-crafted` 机制 + `bounded-learned` 门控 | `memory/cms.py:50-58,332-358` |
| torch band backend（W1/W2 SGD；2026-07-16 M2 起晋升路径代码完备：SHADOW dual-run 每次 band 更新同步 settle pure-vs-torch update-outcome 比较（同 pre-weights 同 target 的 one-step MSE）、`cms_backend_promotion_readout()` 输出退出条件（≥50 settle + parity ≥0.99 + torch 不劣于 pure）与 kill 条件（torch MSE 劣化 ≥0.05）、ACTIVE 写回带 `rollback_last_torch_writeback` 单发回滚、抗遗忘 64 窗聚合钩子；readout 打包进 learned-shadow evidence artifact；ACTIVE flip 仍 gate 于 ≥500 turn 真 trace） | `SHADOW/ACTIVE`=`bounded-learned`；默认 `DISABLED`=手工 MLP | `memory/cms.py`（`CMSBackendPromotionReadout` / `_band_mlp_update` SHADOW 段）· `torch_cms_band.py` · `tests/test_m2_cms_torch_closure.py` |
| `_score_entry` 检索排序（lexical 3.0/2.0×0.8 + facet **+5** tie-breaker） | `hand-crafted` | `memory/store.py:867-908` |
| `_build_learned_recall` 混合权重（learned_weight=3.6+… / artifact 1.15+…，固定公式） | `hand-crafted`（消费 learned CMS 态，权重写死） | `memory/store.py:1234-1290` |
| 瞬态/情景/持久分层写入 + promotion/decay（threshold 默认 0.7） | `hand-crafted` | `memory/store.py:126-132` · `artifacts.py:38-46,107-139` |
| `apply_prediction_error_signal`（magnitude≥0.15 写 episodic/durable；±0.15 调 promotion_threshold） | `should-be-learned-but-hand-crafted` | `memory/store.py:639-700` |
| `_owner_signal` 嵌入（substrate feature_surface 55% + semantic hash） | `frozen-substrate` + `hand-crafted` 混合 | `memory/store.py:968-1020` |

**卡点链接**：**纠偏（2026-07-04）**——Titans PE-gate + ATLAS replay 经 `build_default_memory_store` **运行时已默认 CPU-ACTIVE**（非"默认关"；"默认关"只对裸 `CMSMemoryCore(...)` 构造成立）。真正默认关的仅 `cms_torch_backend`（GPU-gated）。[#89](../known-debts.md) 修法 **Stage 0 已 land**（2026-07-04）：report-only 抗遗忘双指标（`new_knowledge_absorption` / `old_knowledge_retention`，CMSState + lifecycle_metrics）+ A/B matched-control 证据（uplift 背景带漂移 ≤ rollback）+ rollback drill，见 [cms-atlas-titans-uplift.md](./cms-atlas-titans-uplift.md)。**残余（Stage 1，GPU）**：`cms_torch_backend` SHADOW→ACTIVE + ≥500 turn 真 trace 增益曲线。

### 5.4 `vz-cognition`（9 owner）

#### PredictionErrorModule — `prediction_error`

| 决策点 | 分类 | 证据 |
|--------|------|------|
| 四轴 prediction/actual 加权映射 + magnitude 计算 | `hand-crafted` | `prediction/error.py:227-257,344-436` |
| AAC alignment 过渡 severity（2026-07-16 起为 learned 校准：静态表=初始化+回滚点，`_AlignmentSeverityCalibrator` 以下一轮 realized relationship delta settle，drift 受 ±0.15 包络约束） | `bounded-learned`（初始=静态表） | `prediction/error.py`（`_AlignmentSeverityCalibrator` · `_compute_alignment_transition_contribution` · `_advance` settle） |
| external outcome → 四轴 bias 映射（2026-07-16 起为 learned 校准：`ExternalOutcomeBiasCalibrator` 由 PE owner 持有，以同轮内部 pre-bias outcome 在线校准 per-kind bias，drift 受 ±0.15 包络约束，静态表=初始化+回滚点） | `bounded-learned`（初始=静态表） | `prediction/error.py`（`ExternalOutcomeBiasCalibrator` · `_apply_external_outcome_bias`） |
| aleatoric 分量（EMA variance floor） | `hand-crafted` | `prediction/error.py:1322-1351,1810-1813` |
| **epistemic 分量 + `_PELearnedCritic`（18-dim 线性回归 + validation gate）** | `bounded-learned` | `prediction/error.py:1412-1565`（gate：`validation_delta<0`→block） |
| critic 特征向量（substrate digest + z_t digest + regime/action hash） | `frozen-substrate` | `prediction/error.py:1391-1408` |
| PE 分布窗口 readout（IQR/entropy，禁止驱动下游控制） | `hand-crafted` | `prediction/error.py:572-575` · `distribution.py:14-19` |

#### CreditModule — `credit`

| 决策点 | 分类 | 证据 |
|--------|------|------|
| PE-first credit 主路径（dominant track + reward） | `hand-crafted` | `credit/gate.py:473-539,702-710` |
| COCOA historical baseline（regime payoff 加权） | `hand-crafted` | `credit/gate.py:1127-1259,1287-1294` |
| **COCOA learned baseline（`_RewardingStateHead` 15-dim 线性 head + SGD，经 gate）** | `bounded-learned` | `credit/gate.py:275-411,1517-1597` |
| ModificationGate 阻断理由（alert / margin / capacity / framing risk；R9/R10 安全底线，**设计上保留规则级联**） | `hand-crafted`（架构约束） | `credit/gate.py`（`_evaluation_and_structural_gate_reasons`） |
| ModificationGate learned risk 旁路（2026-07-16 C3：`GateRiskLearner` 有界 logistic head，以每次规则裁决为 settle 目标，`CreditSnapshot.gate_risk_readout` report-only 发布；无任何进入决策的路径） | `bounded-learned`（SHADOW 旁路，规则不让位） | `credit/gate.py`（`GateRiskLearner` / `gate_risk_features` / `observe_gate_decision`）· `tests/test_c3_gate_closure.py` |
| evaluation readout 缩放进 credit（`score.value*0.25`） | `hand-crafted` | `credit/gate.py:500-518` |
| least-control readout（固定 0.40/0.60，report-only） | `hand-crafted` | `credit/gate.py:1688-1716` |

#### RegimeModule — `regime`

| 决策点 | 分类 | 证据 |
|--------|------|------|
| `score_regimes`（6 regime 固定系数线性公式；注释 "requires real learning over collected traces"） | `should-be-learned-but-hand-crafted` | `regime/scoring.py:201-404` |
| 最终 regime 选择（最高分 wins）+ hold/switch | `hand-crafted` | `regime/identity.py:247-248,788-806` |
| 在线 historical_effectiveness / selection_weights 更新（`*0.7+*0.3`，lr=0.02） | `hand-crafted` | `regime/identity.py:439-465,146` |
| `apply_metacontroller_evidence`（2026-07-16 起查 `templates.py` 的 `metacontroller_evidence_affinity` 表，identity.py 零硬编码 regime_id，AST 契约守门） | `hand-crafted`（表驱动；learned head 待 #44 SYS-1） | `regime/templates.py`（`metacontroller_evidence_deltas`）· `regime/identity.py`（`_apply_evidence_signal`）· `tests/contracts/test_regime_no_hardcoded_evidence_mapping.py` |
| `apply_policy_consolidation`（2026-07-16 起查 `consolidation_affinity` 表） | `hand-crafted`（表驱动） | `regime/templates.py`（`consolidation_gain_multipliers`）· `regime/identity.py` |
| external outcome → regime score（2026-07-16 起为 learned 校准：静态表=初始化+回滚点，`_external_outcome_scores` 以内部 n-step blended 轨迹在线 settle，drift 受 `_EXTERNAL_OUTCOME_SCORE_ENVELOPE` 约束） | `bounded-learned`（初始=静态表） | `regime/identity.py`（`_ingest_external_outcome_attributions` 校准段） |
| regime 模板（固定 3-float embedding + regime_id 字符串） | `hand-crafted` | `regime/templates.py:23-172` |

#### DualTrackModule — `dual_track`

| 决策点 | 分类 | 证据 |
|--------|------|------|
| memory entry → WORLD/SELF track 语义分配（#91 起走 `semantic_embedding` 接缝：有 substrate 时 `frozen-substrate`-grounded，无则 stub fallback） | `frozen-substrate`（有 substrate）/ `hand-crafted`（stub fallback） | `dual_track/core.py:9-12,89-100`；接缝 `semantic_embedding.py`；backend `substrate/text_encoder.py` |
| shared memory 投影准入阈值（`affinity>0.12`） | `hand-crafted` | `dual_track/core.py:110` |
| controller code 多源融合权重（memory 0.25 + temporal 0.20 + semantic 0.20 + owner 0.35） | `hand-crafted` | `dual_track/core.py:256-269` |
| cross-track tension（goal overlap penalty 0.15 + divergence 系数） | `hand-crafted` | `dual_track/core.py:303-314` |
| abstract action hint / substrate goal 注入 | `frozen-substrate` | `dual_track/core.py:235-247` |
| SELF track traits / identity gate（traits populator deferred，直接透传） | `hand-crafted` | `dual_track/core.py:32-43,282-283` |
| `learned_gate_shadow`（world/self gate；session 注入 `DualTrackGateLearner` 时为 bounded online-SGD，按 PE realized outcome 打分；未注入时回退固定公式。2026-07-16 C3 起晋升路径代码完备：SHADOW dual-run 同步 settle 固定公式候选、`promotion_readout()` 输出退出条件（≥50 settle + MAE 领先 ≥0.02）与 kill 条件（落后 ≥0.10 → `reset()`）、`export_state`/`restore_state` checkpoint 回滚契约；ACTIVE flip 仍 gate 于外部锚点证据） | `bounded-learned`（`AgentSessionRunner` 默认注入；report-only shadow 字段） | `dual_track/gate_learner.py`（`promotion_readout` / `DualTrackGateLearnerState`）· `dual_track/core.py`（`_derive_gate_shadow`）· `agent/session.py`（session-held + settlement）· `tests/test_c3_gate_closure.py` |

**卡点链接**：[#91](../known-debts.md) 修法 1 **已 land**（2026-07-04）——`dual_track` track 分配、`evaluation` prototype 打分、`application/storage` 检索已迁到可注入 `semantic_embedding` 接缝（有真实 transformers substrate 时用 LM 编码，无则 stub fallback），见 [semantic-embedding-backend.md](./semantic-embedding-backend.md)。**follow-up 已清（2026-07-16 M1）**：`scoring_helpers`/`runtime_helpers` prototype 为逐调用 lazy 嵌入（同 backend 同空间）；apprenticeship / protocol-alignment 的手写 Jaccard 迁到 `semantic_topic_similarity` hybrid 接缝；多 substrate 进程隔离（owner-scoped 安装 + conflict 降级 stub + status 可观测）代码完备。

#### Semantic State（9 语义 owner，共用 `SemanticOwnerModule`）

slots：`plan_intent` · `commitment` · `open_loop` · `user_model` · `execution_result` · `belief_assumption` · `relationship_state` · `goal_value` · `boundary_consent`（`semantic_state/owners.py:231-779`）

| 决策点 | 分类 | 证据 |
|--------|------|------|
| proposal 来源默认 NoOp fallback（OBSERVE @ confidence 0.20） | `hand-crafted` | `semantic_state/owners.py:154` · `proposal_runtime.py:60-97` |
| LLM structured proposal 路径（可选 wired，表达层） | `frozen-substrate` | `semantic_state/llm_runtime.py:363+`（unparseable→fall through NoOp） |
| owner 级 confidence 过滤（如 commitment 0.40） | `hand-crafted` | `semantic_state/owners.py:141,294-310` |
| lifecycle 状态转移（rule-based AAC） | `hand-crafted` | `semantic_state/lifecycle.py` |
| relationship funnel stage 阶梯（固定 trust/turn 阈值 0.85/25 …） | `hand-crafted` | `semantic_state/owners.py:107-118` |
| event adapter → proposal 映射（status→operation 字典 + 固定 control_signal） | `hand-crafted` | `semantic_state/proposal_runtime.py:137-391` |
| owner prediction v2（五个 first-wave owner 的 `predicted_vector`：store 内 per-slot `_OwnerForecastLearner`，settlement 时按 observed 更新；cold-start 与 persistence prior 一致） | `bounded-learned` | `semantic_state/store.py`（`forecast_owner_vector` / `settle_owner_forecast`）· `semantic_state/owners.py`（`_owner_prediction_signals`） |

#### ReflectionEngine / ReflectionModule — `reflection`

| 决策点 | 分类 | 证据 |
|--------|------|------|
| consolidation score（0.35/0.25/0.40 固定系数） | `hand-crafted` | `reflection/writeback.py:1159-1214` |
| memory consolidation（score 阈值选 promote/decay） | `hand-crafted` | `reflection/writeback.py:722-808` |
| policy consolidation（PE 差分 + 0.05/0.08 阈值 → update id） | `hand-crafted` | `reflection/writeback.py:810-905` |
| tension 检测（`cross_track_tension>0.4/0.2` …） | `hand-crafted` | `reflection/writeback.py:1216-1244` |
| lessons 提取（条件 → 枚举 id） | `hand-crafted` | `reflection/writeback.py:1246-1273+` |
| writeback gate（默认 `PROPOSAL_ONLY`；读 credit `GateDecision.BLOCK`） | `hand-crafted` | `reflection/writeback.py:487,621-635` |

#### Social Cognition（多 owner 子包）

| owner / slot | 决策点 | 分类 | 证据 |
|--------------|--------|------|------|
| `belief/intent/feeling/preference_about_other` | proposal + confidence 过滤 | LLM wired 时 `frozen-substrate` / NoOp 时 `hand-crafted`（fail-closed） | `social/tom.py:57,77-103,279+` |
| `belief/intent/feeling/preference_about_other` | 跨轮 record store + prediction settlement + PE-weighted promote/retire（ACTIVE→CONTESTED→RETIRED；settlement 走 semantic embedding cosine，非关键词） | `bounded-learned`（settlement 驱动置信度/状态更新；`AgentSessionRunner` 默认注入 store） | `social/record_store.py`（`settle_pending_predictions` / `apply_outcome_to_record`）· `social/tom.py`（`_settle_and_merge`） |
| `common_ground` | atom 合并 + 同一 settlement 路径（repair/clarification evidence settle 引用预测） | `hand-crafted` scaffold + `bounded-learned` settlement | `social/common_ground.py:110-176`（合并）· `social/common_ground.py`（`_settle_and_merge`） |
| `groups` | frame-derived identity + regime rehydrate（透传部分） | `hand-crafted` scaffold | `social/group.py` |
| `groups` | G1 durability PE settlement（pending window + observed-state 结算 + learned per-group durability score → 未来预测 confidence，settled_errors 经 lifter 转发） | `bounded-learned`（同 ToM/common-ground settlement 机制） | `social/group.py`（`_settle_group_predictions`）· `social/record_store.py`（`apply_group_settlement` / `pending_group_predictions`）· `tests/test_social_group_settlement.py` |
| `social_prediction` / `social_prediction_error` | 信号 lift（SSOT 转发，不重建） | `hand-crafted` | `social/identity.py:136-201,205+` |
| `interlocutor_state` | readout 聚合 | `hand-crafted` | `interlocutor/owner.py:36-39` |

#### ApprenticeshipAlignmentModule — `apprenticeship_alignment`

| 决策点 | 分类 | 证据 |
|--------|------|------|
| guidance constraint 提取（默认 holistic，整句一条） | `hand-crafted` | `apprenticeship/core.py:127-156,537` |
| cognition coverage / surprise（seam `semantic_embedding` cosine `_nearest`，有 substrate 时 substrate-grounded，无则 stub fallback） | `frozen-substrate`（有 substrate）/ `hand-crafted`（stub fallback） | `apprenticeship/core.py`（`_constraint_embedding` / `_CognitionRecord`） |
| version space status 判定（contradictions→INCONSISTENT / surprise≥shrink→SHRINKING） | `hand-crafted` | `apprenticeship/core.py:319-386` |
| contradiction 检测（2026-07-16 M1 起 topic 相似度走 `semantic_topic_similarity` hybrid 接缝：有 backend 用 embedding cosine，无则 stub-token Jaccard byte-identical；recurrence 阈值不变） | `hand-crafted`（相似度底层随 backend 升级） | `apprenticeship/core.py`（`_detect_contradictions`）· seam `semantic_embedding.py` |
| reconciler 阈值 SSOT（agreement=0.40 / mismatch=0.45 / contradiction_topic=0.60） | `hand-crafted` | `apprenticeship/core.py:298-316` |
| PE overlay severity 合成（contradiction×0.8 + mismatch×0.2） | `hand-crafted` | `prediction/error.py:847-888` |

**卡点链接**：[#90](../known-debts.md) 修法 **部分 land**（2026-07-04）——owner 默认 flip 到 **ACTIVE**（仅 apprentice turn 生效），新增 owner 自持的 `should_request_feedback` 契约字段（由 reliability/surprise/version-space 派生）+ `open_loop` actuator（冒出 verification 开环），稀疏反馈请求现已成主链行为，见 [apprenticeship-alignment.md](./apprenticeship-alignment.md) §7.1。**残余（follow-up）**：约束提取 + 覆盖度仍是 deterministic holistic + `stub_semantic_tokens` Jaccard（应升级 LLM structured extractor + 真实 embedding）；`labels_saved` 随机采样对照基线（归 [#87](../known-debts.md) ablation）；`apprenticeship_protocol_alignment` 仍 SHADOW。**A1（2026-07-16）**：protocol 层残余代码侧闭合——快照新增 PE-shaped overlay readout（`pe_overlay_magnitude`，application 侧 report-only）+ protocol-lineage conflict → 保守 typed 修订提案（WEIGHT_DECAY / L3 / 1-turn window 恒人审），`ProtocolRevisionQueueModule` 统一路由；ACTIVE flip 仍 gate 于证据。

#### EvaluationModule — `evaluation`

| 决策点 | 分类 | 证据 |
|--------|------|------|
| 模块定位（readout-only，"NOT the primary learning signal source"，R-PE/R12） | `hand-crafted`（架构约束，**无 learned head**） | `evaluation/backbone.py:85-94` |
| turn scores 构建（feature_signal_value + 固定加权） | `hand-crafted` + `frozen-substrate` 输入 | `evaluation/backbone.py:1320-1410` |
| structured alerts（metric 阈值 → severity） | `hand-crafted` | `evaluation/backbone.py:122,2831+` |
| family signals 提取（6 family 均值，供 PE 读） | `hand-crafted`（readout 变换） | `evaluation/backbone.py:143-159` |
| metacontroller learned_update_rule readout（读 temporal，report-only） | `frozen-substrate` | `evaluation/backbone.py:682-688` |
| mid_layer（2026-07-16 C4 实体化：credit COG-1 + PE magnitude/decomposition + regime persistence/margin 只读再发射，deps 扩为 evaluation/credit/prediction_error/regime，默认 DISABLED） | `hand-crafted`（readout 聚合，R12 只读） | `evaluation/mid_layer.py` · `tests/test_c4_evaluation_cascade_realized.py` |
| expensive_layer（2026-07-16 C4 实体化：mid score 再发射 + substrate persona-geometry readout + 可注入 `LlmJudgeBackend` 接缝，集中 prompt，零 LLM 默认，gate-ineligible 不变量保持） | `hand-crafted`（readout；LLM-judge 仅 readout） | `evaluation/expensive_layer.py` · `evaluation/prompts.py` |
| cross_generation_aggregator（2026-07-16 C4 实体化：有界 5 代窗口聚合 head-to-head → `ModificationGateEvidence`） | `hand-crafted`（evidence 聚合） | `evaluation/cross_generation_aggregator.py`（`build_cross_generation_window_snapshot`） |

---

## 6. Learned 占比基线

> 数字为逐决策点核查后的粗估区间，非精确统计。两个 lens 并列（见"口径"）。

| Wheel | by-count learned（bounded-learned） | by-influence learned | frozen-substrate 输入 | should-be-learned 桶 | 主导态 |
|-------|------|------|------|------|------|
| `vz-substrate` | ~10–20% | ~10–20%（有 LoRA checkpoint 时更高） | ~40–50% | ~15–20% | 冻结基底 readout 为主 |
| `vz-temporal`（默认 n_z=3 + 三 backend DISABLED） | ~10–15%（名义） / **~0% 实际生效** | **~5%** | 0% | ~10% | 纯 Python 启发式 metacontroller |
| `vz-memory` | ~27–33% | **~45–55%**（CMS 更新门控主导在线适应） | ~7% | ~7–13% | 手工超参 + learned 更新门控 |
| `vz-cognition`（9 owner） | ~8–10%（仅 PE critic + COCOA head） | ~10–15% | ~8–12% | ~28–32% | hand-crafted 规则 + 两条线性 head |

### 内核级 headline（诚实结论）

**在默认 wiring 下，内核里真正"学出来"的东西**（2026-07-16 更新，从六处扩到十四处）：
1. `_PELearnedCritic`（PE epistemic，18-dim 线性回归）—— [#7](../known-debts.md)
2. `_RewardingStateHead`（COCOA credit baseline，15-dim 线性 head）—— [#6](../known-debts.md)
3. CMS `LearnedUpdateRule` band 更新门控（+ 可选 Titans PE-gate）—— [#89](../known-debts.md)
4. `DualTrackGateLearner`（world/self gate shadow，session-held online-SGD，PE realized outcome 打分；report-only，C3 起带 promotion readout + checkpoint 回滚）—— W1.A
5. 语义 owner forecast v2（9 个 semantic owner slot per-维 learned forecast，settlement 时更新）—— W1.B
6. ToM / common-ground prediction settlement + PE-weighted promote/retire（跨轮 record store）—— W1.C（CP-16/17 核）
7. `FamilyMatchWeights` action-family 匹配 head（settled outcome 驱动有界 SGD，回滚包络）—— T2
8. β_t 阈值 learnable（store `beta_threshold`，`temporal-prior` 组在线更新）—— T1
9. PE 轴 external-outcome bias 校准（`ExternalOutcomeBiasCalibrator`）+ AAC alignment severity 校准（`_AlignmentSeverityCalibrator`），静态表=初始化+回滚点 —— C2
10. regime external-outcome score 在线 settle（`_external_outcome_scores`，包络约束）—— C1
11. `GateRiskLearner`（ModificationGate learned risk 旁路，report-only，规则不让位）+ `ScheduleGateLearner`（joint loop SHADOW 门控）—— C3 / T3
12. 群体 durability score（G1：GROUP_COMMITMENT_DURABILITY settlement 驱动 per-group learned confidence）—— CP-18
13. CMS torch band SHADOW dual-run settle + promotion readout（默认 DISABLED，晋升路径代码完备）—— M2
14. rare-heavy `PeftLoraRareHeavyBackend` 真 PEFT LoRA 训练接缝（注入时；默认 builtin fallback）—— S1

其中 4–6 为 2026-07-13 intent-alignment remediation，7–14 为 2026-07-16 认知 AGI 代码完整度收敛包（T1–T3 / C1–C4 / M1–M2 / S1 / A1–A2 / I1 / G1 / D1）落地。11/13 为 report-only shadow / SHADOW dual-run（不进 live 决策）；其余直接参与运行时适应。

**其余多数决策仍是 hand-crafted 规则或 frozen-substrate readout**；ETA 论文级的 torch metacontroller / Internal RL / ndim GRU 已实现且 2026-07-16 起**一键可选**（`temporal_profile="learned-ndim"` / `TorchCausalZPolicy` first-class / `rare_heavy_training_backend="peft-lora"`），但**默认仍 legacy/DISABLED 待证据门**。`should-be-learned-but-hand-crafted` 桶经 2026-07-16 收敛包大幅缩减（见第 7 节：#79/#80 PE 侧/#81/#88 family 匹配/#89 代码侧/#90 protocol 残余/#91 全部划勾），**代码侧剩余实质项**为：`score_regimes` 固定系数（gate 于 [#44](../known-debts.md) SYS-1 真 trace）、`apply_prediction_error_signal` 固定写入阈值、temporal torch 三后端 ACTIVE flip（gate 于 ≥500 turn 证据）、`semantic_feature_surface_from_text` 固定 prototype。

对外口径应为："架构齐、契约稳、回滚全；learned 肌肉已就位且晋升路径（SHADOW dual-run + promotion readout + kill 条件 + checkpoint 回滚）代码完备；默认 wiring 保守，ACTIVE flip 与 capacity→gain 的 scale 曲线 gate 于真 trace / GPU 证据"（对齐 [#86](../known-debts.md)）。

---

## 7. `should-be-learned-but-hand-crafted` → debt 映射

| 决策点 | 证据 | 关联 debt | 升级退出条件 |
|--------|------|-----------|--------------|
| temporal torch 后端（SSL / runtime forward / Internal RL）默认 DISABLED | `final_wiring.py:222-224` | [#88](../known-debts.md) | 三后端 SHADOW→ACTIVE + strict-ETA evidence + ≥500 turn `validation_delta≥0.02` |
| ~~action family cosine 匹配~~ **已 learned 化（2026-07-16 T2）**：`FamilyMatchWeights` learned 匹配 head，历史系数=初始化+回滚点 | `metacontroller_components.py`（`FamilyMatchWeights`） | [#88](../known-debts.md) / [#86](../known-debts.md) | ✅ 代码侧完成；残余随 ndim/torch metacontroller ACTIVE 让位 learned family 发现 |
| CMS 抗遗忘（PE-gate + ATLAS replay 运行时**已 CPU-ACTIVE**；torch band 默认 DISABLED） | `store.py:76-77,112-113`(factory) · `cms.py`(proxy) · `final_wiring.py:229`(torch DISABLED) | [#89](../known-debts.md) | ✅ Stage 0 land（report-only 抗遗忘双指标 + A/B 证据 + rollback drill）；✅ 2026-07-16 M2：Stage 1 **代码侧**闭合（SHADOW dual-run settle + promotion readout + ACTIVE 回滚 drill + 抗遗忘窗口钩子，`tests/test_m2_cms_torch_closure.py`）；残余仅 ≥500 turn 真 trace 增益曲线（GPU 证据） |
| `apply_prediction_error_signal` 写入 magnitude≥0.15 阈值 | `store.py:639-700` | [#89](../known-debts.md) | learned 写入门控替换固定阈值 |
| apprenticeship owner + 稀疏反馈请求（#90：owner ACTIVE + `should_request_feedback` 字段 + open_loop actuator + LLM structured extractor + random-sampling component arm） | `apprenticeship/core.py` · `semantic_state/owners.py`(OpenLoopModule) · `final_wiring.py`(ACTIVE + reposition) · `lifeform_service/verticals.py`(extractor + `companion-active-learning-off`) | [#90](../known-debts.md) / [#87](../known-debts.md) | ✅ ACTIVE + feedback actuator + production LLM extractor + `active-learning-off` random baseline serving land；残余 `apprenticeship_protocol_alignment` ACTIVE |
| dual_track / evaluation / storage / application helpers / apprenticeship coverage 的语义嵌入（#91 修法 1 + follow-up land：走可注入 seam，有 substrate 时真实、无则 stub） | `dual_track/core.py` · `evaluation/semantic_readouts.py` · `application/storage.py` · `application/scoring_helpers.py` · `apprenticeship/core.py` · seam `semantic_embedding.py` | [#91](../known-debts.md) | ✅ seam + substrate LM backend land；application helpers 与 apprenticeship coverage 已迁到同接缝；2026-07-16 M1：多-substrate process isolation（owner + conflict 降级）+ `semantic_topic_similarity` hybrid 接缝 land，代码侧清账，残余 gate 于真实 substrate 增益 evidence |
| `score_regimes` 固定系数线性公式 | `regime/scoring.py:201-404` | [#80](../known-debts.md) / [#86](../known-debts.md) | learned regime scoring over traces（需 [#44](../known-debts.md) SYS-1） |
| ~~`apply_metacontroller_evidence` / `apply_policy_consolidation` 硬编码 regime_id~~ **已表驱动化（2026-07-16 C1）**：affinity 表入 `templates.py`，AST 契约守门 | `regime/templates.py` · `tests/contracts/test_regime_no_hardcoded_evidence_mapping.py` | [#79](../known-debts.md) | ✅ 表驱动 land；learned bias-to-prior head 待 [#44](../known-debts.md) SYS-1 |
| ~~external-outcome → PE 轴 / regime 静态 bias 表~~ **已 learned 校准化（2026-07-16 C1/C2）**：regime 侧 `_external_outcome_scores` 在线 settle、PE 侧 `ExternalOutcomeBiasCalibrator`，静态表=初始化+回滚点 | `regime/identity.py` · `prediction/error.py` | [#80](../known-debts.md) | ✅ 代码侧完成；ACTIVE 增益证据待真 trace |
| runtime hint summary 文本硬编码 | `regime/contracts.py`(DomainHintCatalog) · `runtime_helpers.py`(薄 wrapper) | ~~#81~~（已关闭，A2 2026-07-16） | ✅ typed `DomainHintCatalog`（含 `language` i18n 接缝）替代 if/elif；契约测试 `test_domain_hint_catalog.py` |

---

## 8. 升级 / 回滚建议（按 `WiringLevel` 三态）

**无 GPU 现在可推进**（把占比从 hand-crafted 移向可观测的 learned 对照）：
- temporal `temporal_ssl_backend` / `temporal_runtime_backend` 跑 `SHADOW`，比对 learned vs 启发式 β_t/z_t readout（[#88](../known-debts.md) Stage 0）。
- ~~CMS `pe_features_enabled=True` 跑 `SHADOW` 比对~~ **已 land（2026-07-04）**：PE-gate/replay 运行时本已 ACTIVE；本轮补 report-only 抗遗忘双指标 + A/B matched-control 证据 + rollback drill（[#89](../known-debts.md) Stage 0）。
- ~~apprenticeship 用 fake-provider 跑稀疏反馈 E2E（[#90](../known-debts.md) Stage 0）~~ **已 land**；2026-07-14 补生产 LLM extractor、真实 embedding coverage 与 `active-learning-off` random baseline serving arm。
- ~~`stub_semantic_embedding` 升级为可注入 backend，默认 fallback stub（[#91](../known-debts.md) 修法 1）~~ **已 land（2026-07-04）**：seam + substrate LM backend + Brain 注入，见 [semantic-embedding-backend.md](./semantic-embedding-backend.md)。

**需 GPU / 真 trace 才能 ACTIVE**（`capacity→gain` 曲线，对齐 [#86](../known-debts.md) 修法 2）：
- 三 torch backend + `cms_torch_backend` → `ACTIVE`，`n_z ∈ {3,16,64,256}` 容量阶梯 × ≥500 turn 真 trace 的增益曲线。
- 任一 flip 都必须 `SHADOW`→比对快照→`ACTIVE`，保留 checkpoint 回滚（[#6](../known-debts.md)/[#7](../known-debts.md) 已确立的 promotion criteria 先例）。

**本 spec 本身不 flip 任何 wiring**；上述为路线，不是本次改动。

---

## 与其他能力域的关系

| 关系 | Spec / 文档 | 说明 |
|------|-------------|------|
| 上位诚实债 | [known-debts.md #86](../known-debts.md) | 本 spec = #86 修法 1 的落地产物 |
| 卡点盘点 | [known-debts.md #88–#91](../known-debts.md) | 本 spec 是四条学习卡点的上位覆盖表 |
| 依赖 | [prediction-error-loop.md](./prediction-error-loop.md) | PE learned critic 分类来源 |
| 依赖 | [credit-and-self-modification.md](./credit-and-self-modification.md) | COCOA head + gate 分类来源 |
| 依赖 | [continuum-memory.md](./continuum-memory.md) / cms uplift | CMS band 更新分类来源 |
| 协作 | [relational-soft-verifier.md](./relational-soft-verifier.md) | 关系域 learned 信号（[#85](../known-debts.md)），should-be-learned 的产品命脉子集 |

## 变更日志

- 2026-07-16：认知 AGI 代码完整度提升（收敛包 T1–T3 / C1–C4 / M1，纯代码侧，ACTIVE 均 gate 于证据）。T1：β_t 阈值 learnable 化（store `beta_threshold`）+ ndim GRU 经 `temporal_profile="learned-ndim"` 可实例化；T2：action family 匹配换 `FamilyMatchWeights` learned head（固定系数=初始化+回滚包络）；T3：TorchCausalZPolicy first-class + joint loop `ScheduleGateLearner` SHADOW 门控；C1：regime_id 映射表驱动化（AST 守门）+ external-outcome regime 表 learned settle；C2：PE 轴 bias 表与 alignment severity 表换有界 learned calibration；C3：`DualTrackGateLearner` dual-run/promotion/checkpoint 闭环 + `GateRiskLearner` SHADOW 旁路；C4：evaluation mid/expensive/cross-gen 空壳实体化（R12 只读，LLM-judge gate-ineligible）；M1：语义嵌入多 substrate owner 隔离 + `semantic_topic_similarity` hybrid 接缝（apprenticeship / protocol-alignment Jaccard 迁移）；M2：CMS torch band SHADOW dual-run settle（pure-vs-torch update-outcome MSE）+ `cms_backend_promotion_readout()` 退出/kill 条件 + ACTIVE 写回单发回滚 drill + 抗遗忘 64 窗聚合，readout 进 learned-shadow artifact；S1：rare-heavy 真训练接缝 land——`RareHeavyAdapterTrainingBackend` protocol + `PeftLoraRareHeavyBackend`（真 PEFT LoRA，冻结 base）入 `vz-substrate`，两 runtime `set_rare_heavy_training_backend` 注入 + `clone_for_rare_heavy` 传播，`BrainConfig.rare_heavy_training_backend` 配置面，启发式/内置 loop 降级为文档化 fallback（`tests/contracts/test_rare_heavy_training_backend.py`）；A1：apprenticeship protocol 层残余闭合——PE-shaped overlay readout + protocol-lineage conflict → typed 修订提案经 R10 gate 人审队列（`tests/contracts/test_apprenticeship_protocol_revision_path.py`）；A2：#81 关闭——runtime hint summary / topic_tags 迁移为 `vz-cognition.regime.contracts` typed `DomainHintCatalog`（`ApplicationBrief` 旁，`language` i18n 接缝，文本字节不变），`runtime_helpers` literal-domain 分支归零（`tests/contracts/test_domain_hint_catalog.py`）；I1：ingestion 空壳闭合——slice 2b `web.py` adapter（robots 门 / bounded fetch / readability + 显式 stdlib 兜底，`[web]` extra）+ TeachingCase service 闭合（typed 契约 → envelope → `IngestionPipeline` 唯一 kernel 入口 + `ingestion-` session 隔离 + retire 回滚标记），见 [runtime-ingestion.md](./runtime-ingestion.md) 变更日志；G1：R20 group owner 补 group-level PE settlement 学习闭环——GROUP_COMMITMENT_DURABILITY 预测停放 `SocialRecordStore` 下轮结算（typed observed-state summary 语义相似度），结算驱动有界 learned per-group durability score（先验 0.5，CONFIRMED +0.10 / DISCONFIRMED −0.20）成为未来预测 confidence，`GroupSnapshot.settled_errors` + `group_durability_score` 发布并由 `SocialPredictionErrorModule` 转发（`tests/test_social_group_settlement.py`），wiring 仍 SHADOW 待 CP-18 证据门；D1：DLaaS eval 换真实现闭合——#13 `LLMRubricGrader` 补契约测试（fake transport：加权 / clamp / fail-loud / env seam），#14 audience analysis 从表单 passthrough 升真 corpus 分析（`LLMAudienceAnalyzer` + `load_asset_corpus` typed 内容解析 + route 404/422/502 typed 错误 + `evidence_stats.analyzer` 诚实标注，caller 字段仍优先，R12 只写 audience_profiles 表；`tests/contracts/test_dlaas_d1_llm_eval.py` / `tests/service/test_dlaas_audience_analysis.py`），生产 ACTIVE gate 于真实 LLM env。各包契约测试见 `tests/test_c3_gate_closure.py` / `tests/test_c4_evaluation_cascade_realized.py` / `tests/test_m2_cms_torch_closure.py` / `tests/contracts/test_semantic_embedding_ssot.py` 等。
- 2026-07-14：提升计划 A1/A2 land。#87 同基底消融从 5-track serving 扩到 9-track serving：新增 `companion-pe-drive-off` / `companion-eta-off` / `companion-active-learning-off` / `companion-lora-adapter` verticals、roster 条目、serve launcher、P1 preflight fingerprint 与 readiness gate。#90/#91 follow-up land：`LLMGuidanceConstraintExtractor` 走集中 prompt +真实 HF runtime 注入；apprenticeship coverage 与 application scoring/runtime helpers 迁到 `semantic_embedding` 接缝；`active-learning-off` 使用 reproducible random feedback sampling baseline。无新增 slot schema；`DATA_CONTRACT.md` 无需变更。
- 2026-07-13：W1 intent-alignment remediation land。三处 should-be-learned 移入 bounded-learned：(A) dual-track `learned_gate_shadow` 从固定公式换成 session-held `DualTrackGateLearner`（bounded online-SGD，PE realized outcome 打分，report-only）；(B) 五个 first-wave 语义 owner 的 `predicted_vector` 从 persistence-prior 升级为 store 内 per-slot learned forecast（后续扩到 9 个 semantic owner slot；settlement 更新，cold-start 与 prior 一致）；(C) ToM/common-ground 获得跨轮 `SocialRecordStore` + prediction settlement（semantic embedding cosine）+ PE-weighted promote/retire（CP-16/17 核）。同批 CP-11 heads 补 checkpoint export/restore + self-reward kill-criteria。synthetic soak 工具就绪：`scripts/run_learned_shadow_soak.py` + GPU-server launcher `scripts/run_learned_shadow_soak.sh`（artifact schema `learned-shadow-soak.v1`，内嵌 `learned_active_gate` 诚实 verdicts——synthetic lane `real_trace_turns=0`，预期 BLOCKED on real-trace gates）。10-turn sanity 已在本机验证通过（checkpoint round-trip + kill-criteria + gate verdicts 全链）；**500-turn 全量 run 待在 GPU 服务器执行**（本机 Windows CPU 实测吞吐不足，run 已中止不留半成品 artifact）。
- 2026-07-04：[#89](../known-debts.md) 前提纠偏 + Stage 0 land（CMS 抗遗忘）。**纠正**：CMS 的 PE-gate + ATLAS replay 经 `build_default_memory_store` 运行时已默认 CPU-ACTIVE（bounded-learned），此前本 spec 记「默认关」不准（只对裸 `CMSMemoryCore(...)` 构造成立）；真正默认关的仅 `cms_torch_backend`（GPU-gated）。本轮 land：CMS owner report-only 抗遗忘双指标 `new_knowledge_absorption` / `old_knowledge_retention`（CMSState + lifecycle_metrics，由真实 per-band drift 派生，不改学习行为）+ A/B matched-control 证据（uplift 背景带漂移 ≤ rollback）+ rollback drill（`tests/test_cms_anti_forgetting_evidence.py`）。残余 Stage 1（torch band ACTIVE + ≥500 turn 真 trace）GPU follow-up。
- 2026-07-04：[#90](../known-debts.md) 修法部分 land（主动学习）。`apprenticeship_alignment` 默认 flip 到 ACTIVE（仅 apprentice turn 生效，普通轮 idle → no-op），新增 owner 自持 `should_request_feedback` / `feedback_request_reason` / `feedback_request_urgency` 字段 + `open_loop` actuator（冒出 verification 开环，`build_final_runtime_modules` 重排保证 open_loop 在 apprenticeship 之后）。稀疏反馈请求成主链行为；残余（LLM extractor / labels_saved 对照基线 / protocol ACTIVE）。见 [apprenticeship-alignment.md](./apprenticeship-alignment.md) §7.1。
- 2026-07-04：[#91](../known-debts.md) 修法 1 land。语义嵌入 stub 升级为可注入 `SemanticEmbeddingBackend` 接缝（vz-contracts，默认 fallback stub）+ 复用已加载 LM 的 `SubstrateTextEncoderBackend`（vz-substrate）+ Brain 仅对真实 transformers runtime 注入；`dual_track` / `evaluation` / `application/storage` 迁到接缝。默认 wiring 下无 substrate / synthetic 路径**占比不变**（仍 stub）；有真实 substrate 时上述 stub 决策点变为 substrate-grounded。残余（apprenticeship token 归 [#90](../known-debts.md)；scoring_helpers/runtime_helpers 字面量 prototype follow-up）。见 [semantic-embedding-backend.md](./semantic-embedding-backend.md)。
- 2026-07-04：protocol-temporal-prior bridge 落地（见 `temporal-abstraction.md`）。此前本 spec 隐含的落差之一——BehaviorProtocol `active_mixture` 声明被 metacontroller 消费但 temporal 侧从未接线——已闭合：`active_mixture` 现经 orchestrator-mediated 上一轮 carryover 压缩成 `beta_t` switch-pressure prior。**默认 `FinalRolloutConfig.protocol_temporal_prior=DISABLED`**，故本 spec 的 by-count / by-influence learned 占比在默认 wiring 下**不变**；该 prior 仍是 hand-crafted 的 dominance/ambiguity→switch 映射（应随 ndim/torch metacontroller ACTIVE 让位学得的 family selection），归入 `should-be-learned-but-hand-crafted` 桶、挂钩 [#88](../known-debts.md)。升级路径：`protocol_temporal_prior` SHADOW→ACTIVE 需 dual-run 快照比对 + 回滚。
- 2026-07-04：v0.1 初版。逐文件核查 `vz-substrate` / `vz-temporal` / `vz-memory` / `vz-cognition`（9 owner）关键决策点，四类标注 + `file:line` 证据 + 两 lens learned 占比基线 + should-be-learned→debt 映射。落地 [known-debts #86](../known-debts.md) 修法 1。
