# 阶段 A 现状核查 Brief — 6 条可并行 SHADOW 候选

> 父级会话规划见 [`experiment.md`](./experiment.md)，本文档对应阶段 A 第 1 项产出（"现状核查矩阵 — 6-8 条 P0 候选的并行只读 audit"）。
>
> 范围：**SYS-1 / COG-1 / COG-2 / COG-3 / CMA-2 / OA-4** —— 父级清单中第 3 类「可并行 SHADOW 候选」全部 6 条。
> 不在范围：基础设施类（EVO-2 / SYS-2 / OA-1）与契约形式化类（OA-2 / OA-3 / DM-4）。
>
> 本文档是 **readonly 现状核查**，不规划实现，不动 spec / 代码 / 配置。

---

## TL;DR 对照表

| 候选 ID | 盲点假设 | 主要 owner | 新增 snapshot slot | dialogue ablation 可直挂？ | 主要上游依赖 | 主要耦合 / 风险 |
|---|---|---|---|---|---|---|
| **SYS-1** CPD β_t 切换 | **PARTIALLY-REFUTED**（CPD 缺，但 β 是学到的不是硬规则） | `temporal_abstraction` | 0 必需（CPD 信号嵌入现有 `TemporalAbstractionSnapshot`） | **是**（profile 分支模式同 atlas-titans-cms-uplift） | 无强阻塞；可与 EVO-2 cascade 解耦先跑 | ⚠ 不能让第二模块发"并行 β 边界"，必须留在 temporal owner 内 |
| **COG-1** 反事实信用 + least-control | **PARTIALLY-REFUTED**（COCOA Phase 1.A+2.A 已上线，least-control / commitment 粒度仍缺） | `credit` | 0 必需（least-control 字段建议加到 `CreditSnapshot`） | **半是**（profile 分支可加；but benchmark `metric_means` 当前不抽 COCOA readout） | benchmark 抽数能力 | ⚠ 与 `evaluation` owner 边界：避免 evaluation 重新成为"另一个 truth" |
| **COG-2** ToM owner 学习 | **PARTIALLY-REFUTED**（4 个 ToM slot + role + multi-party 已 ACTIVE，`UserModelSnapshot` 仍是单桶；fixtures 缺多人场景） | `user_model` / `belief_about_other` 等 | 0 必需（已有 4 ToM slot + role + multi-party）；可能拆 `UserModelSnapshot` | **半是**（profile 容易；but wrong-person / witness 必须在场景层而非 profile 层加） | 多主体 fixture / scripted scenario | ⚠ `belief_assumption`（自己的 belief） vs `belief_about_other`（ToM）双 owner 边界要明确 |
| **COG-3** Persona/Regime geometry 漂移监控 | **CONFIRMED**（latent persona-vector 类 readout 完全缺；现有只有 `posterior_drift`） | `evaluation`（read-only readout publisher）；上游 `substrate` / `temporal` | 0 必需（`evaluation` enrichment）；如要独立则新增 `persona_geometry` slot | **是**（profile 与 baseline wiring 完全相同，只加 readout） | 需要 substrate 后端真的填 `feature_surface` / `residual_activations` | ⚠ 与 `regime` 自身的 identity stability、temporal `posterior_drift` 三者要分清，避免重复计 |
| **CMA-2** VZ-MemProbe（4 探针） | **PARTIALLY-REFUTED**（4 探针都已 PASS in `tests/longitudinal/test_vz_memprobe_*.py`，spec 已收敛） | `memory`（探针只读 `MemoryStore.retrieve`）；`evaluation` 是 readout 家族 | 0（`mp.*` 是 pytest readout，不入 snapshot） | **类型错误 — 不应该是 dialogue profile**；建议保留为独立 longitudinal pytest 通道 | EVO-2 cascade 决定 `mp.*` 是否进入跨 generation aggregate | ⚠ 必须只走 `MemoryStore.retrieve` public API，禁止 `_artifact_store` 等内部访问 |
| **OA-4** VZ-Audit Agent 作 Gate 标配 | **PARTIALLY-REFUTED**（two-gate validation_delta + capacity_cost + rollback_evidence 已强制，N8 风格 audit-agent / risk score / 8-attack 验收完全缺） | 新建 `audit` owner（独立 snapshot publisher） | **新增 `audit`/`vz_audit` slot** + risk score + transcript + tool trace 字段 | **类型错误 — dialogue ablation 不是它的对照面**；应挂 `run_multi_artifact_acceptance_benchmark` | SYS-2 / DM-4 形式化先到位；rare-heavy artifact 路径有可观察对象 | ⚠ 不能让 audit 直接 mutate `credit`/`evaluation`；必须以新 owner snapshot 形式被 gate 消费 |

---

## 选型建议

### 可以最先起跑 SHADOW profile（盲点成立 + ablation 可直挂 + 无强阻塞）

- **SYS-1**：单一 owner（`temporal_abstraction`），单一 profile 分支，CPD 可作为 `_apply_prediction_error_signal` 路径上的并行 readout，对外 SHADOW；不需要等 EVO-2/SYS-2/DM-4 任一基础设施。**最优先候选**。
- **COG-3**（read-only readout-only）：profile 层与 baseline 完全相同，只新增 substrate-backed metric，可在 `evaluation` snapshot 中以现有 EvaluationScore 形式发布。需要确认 `SubstrateAdapter` 后端是否真的填 `feature_surface` / `residual_activations`（spec 已留 hook，但取决于具体后端实现）。

### 应该等阶段 B 基础设施先到位

- **OA-4**：N8 风格 audit-agent 只在 rare-heavy artifact lifecycle 上有意义，不是 dialogue PE-ETA ablation 的对照面。强烈建议等 **SYS-2 双门**与 **DM-4 gate-eval 形式化** 落地后再启动；起跑面是 `run_multi_artifact_acceptance_benchmark` 类的 artifact-promotion 测试，而不是 dialogue benchmark。
- **CMA-2**：4 个探针 spec + 测试 + PASS 状态已完成（`tests/longitudinal/test_vz_memprobe_*.py`）。**真正缺的是把 `mp.*` 接进 EVO-2 cascade 跨 generation 汇总**，而不是再造一个 SHADOW profile。建议归入 EVO-2 packet 的 cascade family 子项推进。

### 建议合并 packet 推进（强互补对）

- **SYS-1 ⊗ COG-1**：两者都是 PE-first 改动，共享 `prediction_error` + `temporal_abstraction.closed_segments` 语义。SYS-1 让 segment closure 来自真实 PE spike，COG-1 在 segment closure 上做 counterfactual contribution 归因——SYS-1 把"边界识别更准"，COG-1 把"边界归因更细"，闭环。
- **COG-3 ⊗ OA-4**：COG-3 read-only 几何漂移可作为 OA-4 audit-agent 的工具集中一项 elicited probe；两者都关心 substrate-owner refresh 后的稳定性 readout，组合形成"几何监控 + 行为审计"双视角。
- **COG-2 ⊗ CMA-2**：CMA-2 的 4 个 probe 当前是单 interlocutor，COG-2 上线 ToM 后，需要扩展 per-interlocutor probe 变体（subject / addressee / witness 维度）。这是后续工作，不阻塞当前阶段。

### 需要重新评估必要性（盲点已被部分否定）

- **COG-1** 与 **COG-2** 的"现状盲点"段落都明显**滞后于代码**：COCOA Phase 1.A/2.A 已经在 `final_wiring.py` 与 `credit/gate.py` 落地；4 个 ToM slot + conversational_role + multi_party_identity + social_prediction 都已经在 `FinalRolloutConfig` 中默认 ACTIVE。两条候选的"原始动作清单"如果照抄会重复劳动。建议剩余工作收敛为：
  - **COG-1 剩余**：(a) `least_control` 字段补到 `CreditSnapshot`；(b) `metric_means` extractor 抽 COCOA `counterfactual_contribution_learned` readout 进 dialogue benchmark；(c) commitment_event_id-style lineage 进入 `CreditRecord.context`。
  - **COG-2 剩余**：(a) `UserModelSnapshot` 拆分（belief / desire / intention / affect 不再共桶）；(b) paper-suite-small 增加 wrong-person / witness / private-leakage 三类场景；(c) `belief_assumption` ↔ `belief_about_other` SSOT 边界 spec 化。
- **CMA-2** 实质性的 spec + 探针实现已经完成；剩余工作是 evaluation cascade 集成而非候选本身。

---

## 逐条详细 brief

### SYS-1 — CPD 涌现 β_t 切换

#### Q1. 现状盲点假设：**PARTIALLY-REFUTED**

CPD（change point detection）/ "PE-spike-as-switch-trigger" 在仓库中**确实不存在**——`packages/vz-temporal/` 没有任何 changepoint 相关字符串，`docs/specs/emergent-action-abstraction.md` 也只字未提 CPD：

```13:24:docs/specs/emergent-action-abstraction.md
- `prediction_error` 是唯一的预测-现实 mismatch owner。
- `temporal_abstraction` 是唯一的 `z_t / beta_t` 时间抽象 owner。
...
- 不新增 `DelayedOutcomeLedger` owner；delayed outcome 的边界来自 `beta_t` segment closure。
```

但盲点段落把 β_t 描述为"启发式或强 supervised Interest Function"是不准确的——当前 β 是**learned switch unit**（posterior + 学到的 switch 权重），不是硬规则：

```103:121:docs/specs/temporal-abstraction.md
- 当前 ndim metacontroller 已收敛到**单一 owner 参数面**：SSL trainer、runtime policy、internal RL、rare-heavy snapshot/export/import 共享同一个 `MetacontrollerParameterStore` 可见的 encoder/switch/decoder 权重
...
- 当前 `full-learned` 已把 `z_t` owner 更新规则收敛到显式 posterior + learned switch 路径：`z_t = beta_t * z_candidate + (1 - beta_t) * z_{t-1}`
```

PE 已经是 `TemporalModule` 的 dependency，但作为**continuous SSL fitting 信号**进入策略，不作为 spike-based termination：

```2690:2705:packages/vz-temporal/src/volvence_zero/temporal/interface.py
def _apply_prediction_error_signal(
    self,
    prediction_error_snapshot: PredictionErrorSnapshot | None,
) -> None:
    if prediction_error_snapshot is None or prediction_error_snapshot.bootstrap:
        return
    pe = prediction_error_snapshot.error
    signed_reward = pe.signed_reward
    target_residual = _clamp(0.45 + abs(pe.task_error) * 0.30 + max(signed_reward, 0.0) * 0.10)
```

PE 标量本身（`magnitude` / 各轴 error）已经存在，CPD 工程化所需的输入信号齐全。

**净结论**：盲点的"缺 CPD / PE 没有作为 switch 触发器"成立；"切换是启发式"不准确——是**学到的 + 阈值 + 先验**驱动。

#### Q2. Owners / snapshot slots / WiringLevel

- **Owners 触碰**：主 `temporal_abstraction`（publisher）；上游读 `prediction_error`（已是依赖）；下游 `closed_segments` 的消费者包括 `credit`（DATA_CONTRACT §3.5/§3.6）和 `evaluation`（`final_wiring.py` 聚合 temporal evidence）。可选：`experience_fast_prior` 如果 CPD 也吃 delayed-credit 信号。
- **Snapshot slots**：现有 `TemporalAbstractionSnapshot.closed_segments / controller_state` 已经能承载新增的 CPD statistic / spike flag / rationale；**不应**为 β 边界开第二个 owner（spec 明确禁止并行 trace owner）。
- **WiringLevel SHADOW 支持**：`TemporalModule.default_wiring_level = WiringLevel.SHADOW`；`FinalRollupConfig.temporal = ACTIVE` 默认；staged wiring pattern 见 `packages/vz-runtime/src/volvence_zero/integration/final_wiring.py:1511-1522`。**已支持**。

#### Q3. ablation 框架适配

**直接可挂**——profile 分支模式与 `atlas-titans-cms-uplift` 完全同构：

```7871:7894:packages/vz-runtime/src/volvence_zero/agent/dialogue/_legacy.py
if profile_label == "pe-eta":
    return AgentSessionRunner(
        session_id=_base_session_id(profile_label),
        ...
    )
if profile_label == "atlas-titans-cms-uplift":
    # SHADOW-evidence-only profile: same wiring as ``pe-eta`` but with
    # the CMS ATLAS / Titans uplift flags turned on.
    ...
```

**最小扩展**：(a) 在 `build_standard_dialogue_runner` 添加 `cpd-beta-switch-shadow` 分支；(b) 如需要保留 carryover 语义，加入 `_profile_allows_interval_carryover_credit`；(c) 仅当注册到 `default_dialogue_ablation_profiles()` 才需更新 `tests/test_dialogue_benchmark.py:638` 的 frozen tuple——**否则只通过显式 `profile_labels=` 传入即可，零测试改动**。

#### Q4. 耦合图

- **被阻塞**：无强阻塞。**DATA_CONTRACT / spec sync** 仅在 CPD readout 公开为 contract 字段时才需要。
- **会阻塞**：`closed_segments` 语义变更会影响 OA-* delayed-credit / segment-credit 消费者；`DM-4` 契约形式化如果包含此 slot 也要 rebase。
- **互补**：与 **COG-1**（PE-first credit）天然配对——SYS-1 改善边界识别，COG-1 改善边界内归因。
- **SSOT clash**：低（CPD 留在 temporal owner 内）；**高**（如果第二个模块发并行 β 边界，违反 ETA spec 单 β owner 不变量）。

---

### COG-1 — 反事实信用 + least-control

#### Q1. 现状盲点假设：**PARTIALLY-REFUTED**

盲点段落"R10 焦点是 ModificationGate 是否能开门 / R12 焦点是评估是否可验证 / 缺更细的反事实归因"**显著滞后于代码**——COCOA Phase 1.A + 2.A 已经写入 spec 并在 runtime 落地：

```51:72:docs/specs/credit-and-self-modification.md
### Counterfactual Contribution（COCOA, Phase 1.A + Phase 2.A）

来源：Meulemans et al., "Would I have gotten that reward? Long-term credit assignment by counterfactual contribution analysis", NeurIPS 2023 spotlight (`arXiv:2306.16803`)。同一脉的 ETA 作者线。

落点：`vz-cognition/credit/gate.py` 中的 `derive_counterfactual_contribution_records(...)` helper + `record_nstep_outcomes_from_segment_closure(...)` helper，以及 `CreditLedger.derive_learned_counterfactual_contribution_records(...)` 的 owner-internal learned rewarding-state head。
```

且 `final_wiring.py` 真的 wire 进了 credit-merge 序列：

```2025:2048:packages/vz-runtime/src/volvence_zero/integration/final_wiring.py
counterfactual_credits = credit_module.ledger.derive_learned_counterfactual_contribution_records(
    regime_snapshot=regime_snapshot_value_for_credit,
    temporal_snapshot=temporal_snapshot_value_for_credit,
    prediction_error_snapshot=prediction_snapshot_value,
    evaluation_snapshot=enriched_evaluation,
    timestamp_ms=active_snapshots["evaluation"].timestamp_ms + 5,
)
extra_credits = extra_credits + counterfactual_credits
record_nstep_outcomes_from_segment_closure(
    ledger=credit_module.ledger,
    ...
)
```

**真正缺的**：(a) `least_control` 系列指标在 `dual-track-learning.md` / `evaluation.md` / `CreditSnapshot` 全文搜不到；(b) 当前的 "counterfactual baseline" 是 historical-mixture（actual − baseline），**不是** "if we had not taken this commitment update" 的生成式反事实；(c) commitment_event_id-style lineage 在 COCOA helper signature 中不是一等字段。

#### Q2. Owners / snapshot slots / WiringLevel

- **Owners 触碰**：`credit`（publisher）；上游 `prediction_error` / `regime` / `temporal_abstraction` / `evaluation`；`dual_track` 仅作为 credit context，不构成第二归因 truth。
- **Snapshot slots**：现有 `CreditSnapshot.counterfactual_readouts`、`CreditRecord.level` 已包含 `counterfactual_contribution` / `counterfactual_contribution_learned`、`RewardingStateHeadState`（DATA_CONTRACT §3.5）。**新增需要**：`least_control` / control-effort 字段（无现成位置）；可选 `commitment_event_id` lineage 进入 `CreditRecord.context`。
- **WiringLevel SHADOW**：`CreditModule.default_wiring_level = WiringLevel.SHADOW`；counterfactual levels 已 readout-only / non-gating。**已支持**。

#### Q3. ablation 框架适配

profile 分支可直接加，与 SYS-1 同模式。**关键限制**：现有 `build_pe_counterfactual_closure_report` 是**多轨道平行 trajectories**（profile A/B），不是单 trajectory 内的反事实分支：

```3715:3730:packages/vz-runtime/src/volvence_zero/agent/dialogue/_legacy.py
def build_pe_counterfactual_closure_report(
    comparison_report: DialogueBenchmarkComparisonReport | DialogueComprehensiveBenchmarkReport,
    *,
    baseline_label: str = "pe-eta",
    pe_drive_off_label: str = "pe-drive-off",
    pe_readout_only_label: str = "pe-eta-pe-readout-only",
    eta_off_label: str = "eta-off",
) -> PECounterfactualClosureReport:
```

**最小扩展**：除 profile 分支以外，需要让 benchmark 的 `metric_means` 抽 COCOA readout（今天 closure report 不抽），否则 SHADOW evidence 不会落到对照表中。

#### Q4. 耦合图

- **被阻塞**：head-to-head / verifiable evaluation（EVO-2 / DM-7 / EVO-6 路线）如果想让 readout 进 ModificationGate；OA-2 Mind/Face 隔离要保证 metric 不在 Face 上汇总。
- **会阻塞**：不强阻塞他人。
- **互补**：**SYS-1**（更准的 PE / β）改善 COCOA baseline 输入；OA-4 audit 可消费 credit 升级证据。
- **SSOT clash**：**低**（credit 派生自已发布的 PE / regime / temporal snapshot）；**高**风险出现在 evaluation 或第二 owner 重新实现"contribution"叙事——必须严格让 `CreditRecord` 是唯一权威。

---

### COG-2 — Social Cognition / ToM owner

#### Q1. 现状盲点假设：**PARTIALLY-REFUTED**

`FinalRolloutConfig` 默认已经把 `multi_party_identity`、`conversational_role`、四个 ToM slot（`belief_about_other` / `intent_about_other` / `feeling_about_other` / `preference_about_other`）、`social_prediction` / `social_prediction_error` 提到 ACTIVE，与 spec 描述一致。`docs/DATA_CONTRACT.md` §6 仍然把 social slot 列在"planned migration mirror"，**滞后于代码**。

仍成立的部分：`UserModelSnapshot` 仍是单桶聚合，缺 belief / desire / intention / affect 的内在分解：

```9:16:docs/specs/social_cognition/02_theory_of_mind.md
当前 `UserModelSnapshot` 把 stable preferences、working style hints、sensitive boundaries、durable goals 都压在同一类 `SemanticRecord` 上。它缺少认知科学中最基础的区分：对方相信什么、打算做什么、正在感受什么、长期偏好什么。这些不是同一类状态，不能共享一个 owner 或一个更新规则。
```

```426:436:packages/vz-cognition/src/volvence_zero/semantic_state/contracts.py
@dataclass(frozen=True)
class UserModelSnapshot:
    stable_preferences: tuple[SemanticRecord, ...]
    working_style_hints: tuple[SemanticRecord, ...]
    sensitive_boundaries: tuple[SemanticRecord, ...]
    durable_goals: tuple[SemanticRecord, ...]
    stability_score: float
    control_signal: float
    description: str
    preferred_support_pacing: str = "unknown"
    decision_style: str = "unknown"
    overwhelm_pattern_strength: float = 0.0
```

`InterlocutorState` 是单一 derived readout（12 轴 + zone），**不是** per-agent ToM 或 witness 建模：

```76:81:docs/specs/interlocutor-state.md
`engagement_intensity` / `self_disclosure_level` / `task_focus_level` /
`emotional_weight` / `cognitive_engagement` / `resistance_level` /
`openness_to_guidance` / `directness` / `trust_signal` (signed) /
`stability` / `rapport_warmth` / `pace_pressure`
```

**净结论**：ToM 内核已上线；`UserModelSnapshot` 拆分 + paper-suite 多人场景仍是工作量。

#### Q2. Owners / snapshot slots / WiringLevel

- **Owners 触碰**：9 槽语义脊椎（`user_model` / `belief_assumption` / `relationship_state` / …）；社会层（`multi_party_identity` / `conversational_role` / 4 ToM slot / `common_ground` / `social_prediction[_error]` / SHADOW 中的 `groups`）；`interlocutor_state`；keyed 聚合 `interlocutor_models` / `relationship_states` / `interlocutor_states`（DATA_CONTRACT §6.X）。
- **Snapshot slots**：内核已存在；**新增**主要在 `UserModelSnapshot` 内部解构（belief/desire/intention/affect 子分类）和 keyed per-person view 落地。
- **WiringLevel SHADOW**：四 ToM slot 已经 ACTIVE；profile-shadow 的真正用法是"反向"——把已 ACTIVE 的 slot 在对照 profile 里设为 SHADOW，做 ablation。

#### Q3. ablation 框架适配

profile 分支可加（与 SYS-1 同），**但最大障碍不在 profile 层**：当前 paper-suite scripted fixtures **没有** wrong-person / witness / private-leakage 多人结构。该问题必须在 **scenario 层**通过新 `ScriptedDialogueCase` + `EnvironmentEvent` 帧引入，而不是 profile 维度。`docs/specs/social_cognition/03_conversational_role.md` 的 engineering challenges 已暗示这一点。

#### Q4. 耦合图

- **被阻塞**：EVO-2 cascade（用于看 social PE / credit delta）；OA-2 Mind/Face（防止 ToM evidence 进 Face prompt）。
- **会阻塞**：CMA-2 后续需要 per-interlocutor probe 变体——但当前 4 个 probe 不要求立即扩展。
- **互补**：COG-3（persona drift 可检测跨 interlocutor 人格泄漏）。
- **SSOT clash**：**`belief_assumption`（自己的 belief） vs `belief_about_other`（ToM）** 两 owner 边界必须明确；`user_model` 总桶与 `preference_about_other` / `feeling_about_other` 不能 double-count。`docs/specs/social_cognition/02_theory_of_mind.md` 已警告 `response_assembly` 路由要避免 double-counting。

---

### COG-3 — Persona / Regime Geometry 漂移监控

#### Q1. 现状盲点假设：**CONFIRMED**

latent persona-vector / refusal direction / value-trait 几何 readout **完全不存在**。`cognitive-regime.md` 工程挑战只列嵌入表示、记忆化、控制选择，没有"几何漂移监控"：

```19:25:docs/specs/cognitive-regime.md
## 工程挑战
- 设计 regime 的运行时表示（向量嵌入，不只是字符串标签）
- 实现 regime 的记忆化（可召回历史 regime 及其效果）
- 实现 regime 的高层控制选择（由抽象控制层选择，而非硬编码规则）
- 实现 regime 的延迟结果训练（通过信用分配回路）
- 场景检测必须使用语义级方法，不使用关键词匹配
```

`evaluation.md` 六族 + `mp.*` 都不覆盖 latent persona 几何。仅有的相邻信号是 `metacontroller_state.posterior_drift`（abstraction family），来自 ETA 自身的 posterior 不稳定度，不是 substrate persona-vector：

```639:716:packages/vz-cognition/src/volvence_zero/evaluation/backbone.py
posterior_stability = _clamp(1.0 - metacontroller_state.posterior_drift)
...
EvaluationScore(
    family="abstraction",
    metric_name="posterior_stability",
    value=posterior_stability,
    confidence=0.57,
    ...
),
```

`character-soul-bootstrap.md` 把 soul material 拒之于 `PersonaModule` 之外，但也没定义 latent drift readout。

#### Q2. Owners / snapshot slots / WiringLevel

- **Owners 触碰**：`evaluation`（read-only readout publisher，符合 R12）；上游 `substrate`（提供 hidden state surface）；`temporal_abstraction`（已被复用 `posterior_drift`）；`regime`（identity stability 已发布）。
- **Snapshot slots**：建议**优先**复用 `evaluation` snapshot enrichment（添加新 `EvaluationScore` 而非新 slot）；如果 readout 需独立 lifecycle，再考虑新建 `persona_geometry` slot（成本更高，需要 DATA_CONTRACT 注册）。
- **WiringLevel SHADOW**：`evaluation` 默认 ACTIVE，但只发布 read-only readout 并不构成行为切换——SHADOW vs ACTIVE 的差别仅在指标是否进入 dashboard；**已支持**。
- **substrate hooks**：`SubstrateSnapshot.feature_surface` / `residual_activations` 已有 hook（受 truncated/feature contract 约束），但**取决于具体 backend 是否真的填**。

#### Q3. ablation 框架适配

**直接可挂**——profile 与 baseline 完全相同 wiring，仅在 `metric_means` 中增加 persona drift 指标，对照表的 delta ≈ 0（按设计）。这正是 read-only readout 的天然形态。

**最小扩展**：(a) `build_standard_dialogue_runner` 加新 profile 分支，wiring 与 `pe-eta` 完全相同；(b) 在 dialogue benchmark `metric_means` 抽取层加几个新 key；(c) **不需要**新 acceptance gate（read-only readout 不进 gate）。

#### Q4. 耦合图

- **被阻塞**：substrate backend 真的填 `feature_surface` / `residual_activations`（如果当前是 stub 则需先到位）；EVO-2 cascade 仅在持久化 / 跨 generation aggregate 时需要。
- **会阻塞**：不强阻塞——ModificationGate 链接是消费者 opt-in。
- **互补**：**OA-4**（audit-agent 可把 persona drift 作为 elicited probe 中一项工具）；**CMA-2**（与 `mp.*` 形成 drift family 联合 dashboard）。
- **SSOT clash**：必须区分 **substrate-level persona subspace**（COG-3 关心）/ **ETA controller posterior drift**（已存在）/ **regime owner 自身的 identity stability**——三者都叫"stability"但来源不同，metric 名要带 owner 前缀以避免重复计。

---

### CMA-2 — VZ-MemProbe 评估套件

#### Q1. 现状盲点假设：**PARTIALLY-REFUTED**

4 个 probe 已经全部 PASS in `tests/longitudinal/test_vz_memprobe_*.py`，spec 已收敛。`evaluation.md` Long-Horizon Memory Probes 章节明确 read-only / R12 约束 / NoOpSemanticProposalRuntime 也能跑：

```52:82:docs/specs/evaluation.md
### Long-Horizon Memory Probes（CMA-2 / Phase 2 W2.1）

来源：Logan J. *Continuum Memory Architectures for Long-Horizon LLM Agents*. arXiv:2601.09913, 2026.

R5 / R6 的「记忆连续谱 + 慢反思沉淀」需要长时间窗的**行为级证据**...

VZ-MemProbe 把这 4 个 probe 落到 `tests/longitudinal/test_vz_memprobe_*.py`...

- **不依赖 LLM runtime**...
- **Read-only**...
- **跨 vertical 中性**...
2. **直接 `memory_store.write(...)` 模拟 consolidation**...
   Probe **必须**用 public `memory_store.retrieve(...)` API 做断言，**禁止**直接窥探 `_artifact_store` 等 owner-internal 字段。
```

`continuum-memory.md` 列出 4 个 probe 全部 PASS：

```149:154:docs/specs/continuum-memory.md
| Probe | 文件 | 紧扣的不变量 | 当前状态 |
|---|---|---|---|
| Update | `tests/longitudinal/test_vz_memprobe_update.py` | 后续 belief override 在 retrieval rank 中胜过早期同主题 belief | PASS |
| Temporal | `tests/longitudinal/test_vz_memprobe_temporal.py` | 给定 anchor event，其时间邻域能被召回 | PASS |
| Assoc | `tests/longitudinal/test_vz_memprobe_assoc.py` | 跨 owner 的 3 跳 chain 查询能完整覆盖 | PASS |
| Context | `tests/longitudinal/test_vz_memprobe_context.py` | 同关键词在不同 regime 下，`RetrievalQuery.facets` 能 disambiguate top-1 | PASS |
```

probe 的 retrieval 接口已存在：

```291:305:packages/vz-memory/src/volvence_zero/memory/store.py
def retrieve(
    self,
    query: RetrievalQuery,
    *,
    timestamp_ms: int,
    active_subject_ids: tuple[str, ...] | None = None,
) -> RetrievalResult:
```

**净结论**：候选实质性已落地。剩余的"`mp.*` 进入跨 generation aggregate / dialogue benchmark cascade"是 EVO-2 范畴。

#### Q2. Owners / snapshot slots / WiringLevel

- **Owners 触碰（read-only）**：`memory`（探针只调 `MemoryStore.retrieve`）；`evaluation` 是 readout family 概念，不是 writer。
- **Snapshot slots**：**0 新增**——`mp.*` 是 pytest readout，不需要进 snapshot。
- **WiringLevel SHADOW**：`memory` 默认 SHADOW（DATA_CONTRACT §6 ~1599 行）；探针只读 retrieve，不要求 wiring 切换。

#### Q3. ablation 框架适配

**类型错误**——把 CMA-2 当作 `mem-probe-shadow` dialogue profile 是范畴错配。`run_dialogue_pe_eta_ablation_benchmark` 比较 per-turn dialogue metrics across `profile_label`；memprobe 是 deterministic longitudinal pytest against shared store。

**最小可接入方式**：保留 longitudinal pytest 作为独立 CI 阶段，按 EVO-2 cascade 的 cheap 层接入（probe pass-rate 作为 evaluation 跨 generation aggregate 中的 metric 之一），不要硬塞进 dialogue profile 矩阵。

#### Q4. 耦合图

- **被阻塞**：松耦合——可独立运行；只在 EVO-2 强制要求统一 artifact bundle 时才需要 cascade 集成。
- **会阻塞**：**COG-2** 后续可能需要 subject-keyed probe 变体（per-interlocutor recall）。
- **互补**：**COG-1**（probe pass-rate 作为 outcome）；**COG-3**（drift family 联合 readout）。
- **SSOT clash**：**低**——只要严格遵循 spec（用 `MemoryStore.retrieve` public API + 必要的 owner snapshot），禁止 `_artifact_store`/`hasattr` 内部探查。

---

### OA-4 — VZ-Audit Agent 作为 ModificationGate 标配

#### Q1. 现状盲点假设：**PARTIALLY-REFUTED**

盲点段落"只有开发者手动看一眼日志"高估了缺口——**Two-Gate（validation_delta + capacity_cost + rollback_evidence）已经在 `evaluate_gate_reasons` 强制**：

```663:706:packages/vz-cognition/src/volvence_zero/credit/gate.py
def evaluate_gate_reasons(
    *,
    proposal: ModificationProposal,
    evaluation_snapshot: EvaluationSnapshot,
) -> tuple[str, ...]:
    """Return fail-closed blocking reasons for a self-modification proposal."""
    critical_alert = any(alert.severity == "CRITICAL" for alert in evaluation_snapshot.structured_alerts)
    high_alert = any(alert.severity == "HIGH" for alert in evaluation_snapshot.structured_alerts)
    ...
    if proposal.validation_delta < margin:
        reasons.append(
            f"validation_delta {proposal.validation_delta:.3f} below required margin {margin:.3f}"
        )
    capacity_cap = _capacity_cap_for_gate(proposal.desired_gate)
    if proposal.capacity_cost > capacity_cap:
        reasons.append(
            f"capacity_cost {proposal.capacity_cost:.3f} exceeds cap {capacity_cap:.3f}"
        )
    if not proposal.rollback_evidence:
        reasons.append("missing rollback evidence")
    ...
    return tuple(reasons)
```

且 spec 描述了完整的 gate 输入与 audit 记录路径（`SelfModificationRecord`）：

```145:150:docs/specs/credit-and-self-modification.md
- 当前 ModificationGate 已加入 Two-Gate 风格的保守准入：候选必须携带 `validation_delta`、`capacity_cost` 和 `rollback_evidence`；缺少验证改进、超过容量上限、缺少回滚证据、contract/fallback/rollback evaluation context 不健康时默认 BLOCK。
```

**真正缺**：N8 风格 audit-agent 完全不存在——没有 `VZAuditAgent`、没有 tool loop（`inspect_dataset` / `run_benchmark` / `elicited_benchmark` / `execute_python`）、没有 0-10 risk score + transcript、没有 8 类已知 attack 验收。

#### Q2. Owners / snapshot slots / WiringLevel

- **Owners 触碰**：当前 `credit`（gate decisions + modification audit trail）、`evaluation`（gate context）、`substrate_self_mod`、`vz-runtime` orchestration——**无独立 audit owner**。
- **Snapshot slots**：DATA_CONTRACT §6 没有 `audit` / `vz_audit`。OA-4 必须**新增**：(a) frozen `AuditSnapshot` 类型；(b) 字段：`risk_score: float`、`transcript: tuple[...]`、`tool_traces: tuple[...]`、`detected_attack_classes`、`threshold_decision`。
- **ModificationGate 耦合**：`evaluate_gate` 已经是纯 snapshot 消费者（`EvaluationSnapshot` + proposal）；audit owner 应作为额外的 snapshot publisher，由 `vz-runtime` orchestration 在 gate 决策时**消费**——而非 audit 直接 mutate `credit` 状态。
- **WiringLevel SHADOW**：成熟模式（`credit` 默认 SHADOW；多个 cognition owner SHADOW）。新 audit module 应可在 SHADOW（仅 publish）和 ACTIVE（接入 abort）之间切换。

```1591:1615:docs/DATA_CONTRACT.md
| `substrate_self_mod` | SubstrateSelfModModule | SubstrateSelfModSnapshot | SHADOW | 每 turn / schedule | session / credit audit / rare-heavy review |
| `evaluation` | EvaluationModule | EvaluationSnapshot | ACTIVE | 每 turn ~ 每会话 | regime, prediction_error, credit, reflection |
| `credit` | CreditModule | CreditSnapshot | SHADOW | 每 turn ~ 每会话 | reflection; consumes `prediction_error` + `temporal_abstraction.closed_segments` |
```

#### Q3. ablation 框架适配

**类型错误**——`run_dialogue_pe_eta_ablation_benchmark` 比较的是 per-turn dialogue metrics 跨 `profile_label`，每个 profile 在 `build_standard_dialogue_runner` 需要显式分支。OA-4 的对照面是**post–rare-heavy / post–substrate refresh lifecycle**。

正确的对照面是 `run_multi_artifact_acceptance_benchmark` / replay-selection acceptance 类的 artifact 提升测试。把"每 N 个 dialogue turn 跑一次 audit"硬塞进 dialogue ablation 会混淆 online dialogue benchmarking 和 artifact promotion gate 两种本质不同的测试。

**最小扩展（如果坚持 dialogue 路径）**：除 profile 分支外，需要新增 audit module + audit snapshot 注册——这远超"profile 调整"，应作为完整 packet 而非 SHADOW profile 起跑。

#### Q4. 耦合图

- **被阻塞**：**SYS-2** 双门已经在 `evaluate_gate_reasons` 强制（capacity cap）——OA-4 是第三道证据通道；ordering 应是确定性 cap gate 优先，再 audit readout/orchestrator abort。**DM-4**（gate-eval 形式化）影响 audit 输出如何 bind 到 gate。**EVO-2**（cascade）可路由 audit 结果进 promotion 叙事。
- **会阻塞**：任何 P0 工作如果想在没 audit 的情况下推 ACTIVE rare-heavy / substrate bundle。
- **互补**：**OA-3**（framing checks）vs **OA-4**（post-update probes）覆盖不同 attack surface；**COG-3**（drift 作为 audit 一项工具）；**CMA-2**（memprobe 作为 audit 一项工具）——audit-agent 是天然的多工具组合点。
- **SSOT clash**：**`evaluation`** 保持 R12 PE readout（gate 数学吃 score / alert）；新 **`audit`** snapshot 持有对抗探针结果与 transcript，避免把 evaluation 变成同时承担 readout 和 gate 决策的 jack-of-all-trades。边界：**evaluation = calibrated readout；audit = staged gate evidence**。

---

## 跨候选风险与机会矩阵

下表用 6×6 上三角描述两两关系：✓ = 强互补、~ = 弱互补、⚠ = 潜在 SSOT 冲突或 ordering 风险、— = 几乎无耦合。

| | SYS-1 | COG-1 | COG-2 | COG-3 | CMA-2 | OA-4 |
|---|---|---|---|---|---|---|
| **SYS-1** |  | ✓（PE-first 配对：边界识别 + 边界归因） | ~（segment closure 提供 social PE windowing） | ~（CPD 边界可作 drift 时间锚） | —（probe 与 β 无 owner 重叠） | ~（segment 闭合 = audit window 边界候选） |
| **COG-1** |  |  | ~（counterfactual credit 可吃 social PE delta） | ~（drift 帮助解释 credit 漂移 vs 真实学习） | ~（probe pass-rate 可作 outcome variable） | ✓（credit lineage 是 audit 重要证据源） |
| **COG-2** |  |  |  | ✓（跨 interlocutor persona 泄漏检测） | ✓（per-interlocutor probe 变体） | ~（社会 attack 也是 audit 面） |
| **COG-3** |  |  |  |  | ~（drift family 联合 dashboard） | ✓（drift 是 audit-agent elicited probe 一项） |
| **CMA-2** |  |  |  |  |  | ✓（memprobe 是 audit-agent run_benchmark 工具一项） |
| **OA-4** |  |  |  |  |  |  |

**关键 SSOT 边界提醒**：

1. **`stability` / `drift` 命名空间冲突**：COG-3 的 substrate persona drift、temporal `posterior_drift`、regime identity stability 三者来源不同，metric 名必须带 owner 前缀。
2. **`belief_assumption` vs `belief_about_other`**（COG-2）：自己的 belief vs 关于别人的 belief 是两个 owner，禁止合并。
3. **`evaluation` vs `audit`**（OA-4）：evaluation 是 PE readout（R12 read-only）；audit 是 staged gate evidence（可持有 transcript / tool trace）；两者绝不能互相 mutate。
4. **β_t 边界唯一性**（SYS-1）：CPD 必须留在 `temporal_abstraction` owner 内，禁止第二模块发并行 β 边界。
5. **counterfactual contribution 唯一性**（COG-1）：`CreditRecord` 是唯一权威，evaluation 不能重新发"对比叙事"。

---

## 阶段 B/C 衔接的建议

核查过程中发现的影响父级阶段 B 计划的事项：

1. **DATA_CONTRACT.md §6 滞后于 `final_wiring.py`**：4 个 ToM slot、`conversational_role`、`multi_party_identity`、`social_prediction[_error]` 在代码中已经默认 ACTIVE，但文档仍标记为 "planned migration mirror"。这构成 **R8 spec 与代码偏离**。建议**阶段 B 第 1 个 packet 先把 DATA_CONTRACT 同步到 final_wiring.py 真实状态**，否则后续 SHADOW profile 引用 slot 时会陷入"以 spec 为真还是以 wiring 为真"的混乱。
2. **EVO-2 evaluation cascade 范围扩大**：CMA-2 实际进度建议把"`mp.*` 进入跨 generation aggregate / dialogue benchmark cascade"明确写入 EVO-2 packet 的 sub-task，而不是当作独立 CMA-2 SHADOW profile。这能消除一条候选并让 EVO-2 packet 直接交付 1 个 family 的实证。
3. **OA-4 packet 拆出 SYS-2 orbit**：OA-4 的 audit-agent / risk score / 8-attack 验收是 medium-effort packet，但属于 ModificationGate 强化路径而非 dialogue ablation 路径。建议在阶段 B 第 2 个 packet（SYS-2 + DM-4 双门）之后接续 OA-4，作为"双门 → 三门"的延续，而不是阶段 C 的 SHADOW profile。
4. **COG-3 substrate hook 验证**：COG-3 起跑前需要确认默认 `SubstrateAdapter` 后端是否真的填 `feature_surface` / `residual_activations`。如果当前 backend 是 stub-only，COG-3 起跑前需要补一个最小 substrate-feature 暴露 packet（小工作量），属于阶段 B 范围。
5. **paper-suite-small 多人场景缺口**：COG-2 真正起跑需要 wrong-person / witness / private-leakage 类 ScriptedDialogueCase。这不是 profile 维度而是 fixture 维度的扩展，建议作为 COG-2 packet 的内含项，不影响其他候选。

---

## 附录：阶段 A 起跑顺序总结

| 顺位 | 候选 | 起跑前置 | 工作量级别 |
|---|---|---|---|
| 1 | **SYS-1** | 无 | S（单 owner、profile 直挂） |
| 2 | **COG-3** | 验证 substrate `feature_surface` 后端实填 | S-M（profile 直挂，需要 substrate hook） |
| 3 | **COG-1（reframed）** | 阶段 B EVO-2 cascade 把 metric_means 抽 COCOA 路径打通 | M（least-control 字段 + lineage） |
| 4 | **COG-2（reframed）** | 阶段 B 完成 DATA_CONTRACT §6 同步 + paper-suite 多人 fixture | M |
| 5 | **OA-4** | 阶段 B 完成 SYS-2 + DM-4 双门 packet | M（单独 packet，不走 dialogue ablation） |
| -  | **CMA-2** | 已实质完成，剩余收编入 EVO-2 cascade | XS |

**阶段 A 完成的判定标准**：本 brief 已交付；阶段 B 启动前先解决"DATA_CONTRACT §6 同步"+"substrate feature hook 实填验证"两件 prerequisite。
