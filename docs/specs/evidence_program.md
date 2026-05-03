# 证据计划 Spec

> Status: draft
> Last updated: 2026-04-25
> 对应需求: R12, R15

## 要解决的问题

如何把内部 benchmark / proof harness 升级为可复现、可审阅、可回放的证据生产线，使系统的对外主张能被明确映射到 gate、artifact、盲评和统计结论，而不是靠单次 run 或主观叙述。

## 关键不变量

- 对外主张必须先冻结成可证伪 claim，再绑定 required gates、artifact 和 verdict 规则
- claim verdict 必须回溯到 manifest、seed、git sha、依赖版本和原始 artifact
- dialogue / ETA paper-suite 共享统一的 evidence bundle 口径，不各自发明一套 summary schema
- 盲评外发包不得泄漏 profile label 或内部 case 标识；profile 映射只存在 internal key
- 人评不是替代自动评估，而是额外证据面；自动指标、人评与 claim verdict 必须能并列审阅
- open-environment widening evidence 必须区分 `open_core`、`open_families`、`open_heldout`，不能把单一固定场景误写成开放泛化

## 工程挑战

- 设计 claim registry，把抽象宣传口径压成具体 gate
- 统一 dialogue / ETA aggregate 报告的 pairwise effect、claim verdict 和 evidence bundle 导出
- 让 blind review packet 真正可外发，同时保留内部 unblinding key
- 为人评建立最小协议和可机读 aggregate，而不是只导出一组 transcript
- 让 repeated-run summary 不只给 interval，还能给 matched-control effect 与 retained / weak / fail verdict

## 算法候选

证据计划属于评估与 rollout 审计层，受 R12 / R15 约束：

- evaluation 仍是 PE-first 主链的 readout / gate / widening evidence，不替代 learning primitive
- claim verdict 基于 matched-control comparisons、longitudinal evidence、blind review 与 provenance
- open-environment 作为 widening surface，只能在 held-out 覆盖与统计口径满足时支撑更强 claim

## 接口契约

**消费的输入**：
- dialogue comprehensive / paper-suite aggregate
- ETA proof paper-suite aggregate
- NL essence / ETA acceptance gates
- blind review packet、human rating entries、human rating aggregate
- manifest / provenance / repeated-run summaries / pairwise metric effects

**产出的输出**：
- claim registry / claim verdicts
- external-safe blind review packet
- internal unblinding key
- human rating template / aggregate
- unified evidence bundle
- `snapshot_replay_bundle.json`（planned，Phase 1 随 `docs/specs/emergent-action-abstraction.md` 落地）：导出 existing snapshots（`EnvironmentEvent` / `EnvironmentOutcome` / `temporal_abstraction.closed_segments` / `prediction_error` / `credit`）用于再现与证据审阅，不引入 trace runtime schema

当前实现口径：

- `volvence_zero.agent.paper_suite` 提供共享 `ClaimVerdict` 与 `EvidenceBundle`
- dialogue / ETA paper-suite aggregate 会额外发布 pairwise effects 与 claim verdicts
- dialogue paper-suite export 会同时导出 blinded packet、internal key、rating template、rating aggregate 与 unified evidence bundle
- dialogue emergence dashboard / paper-suite metric values 发布 `canonical_mean_semantic_spine_coverage`、`canonical_mean_cognitive_loop_readiness` 以及 open-environment 对应读数；这些是 semantic owner 快照的证据读数，不作为学习源头
- dialogue NL essence assessment 发布 `semantic-spine-ready` gate，用于审计核心 semantic owner spine 是否具备完整 coverage 与非零 readiness；该 gate 目前不进入默认 required gate 列表
- dialogue paper-suite manifest 将 `canonical_mean_semantic_spine_coverage` 与 `canonical_mean_cognitive_loop_readiness` 列为 secondary metrics；companion stateful relationship verdict 优先消费 repeated-run summary，reference dashboard 只作为 fallback
- ETA paper-suite export 会导出统一 evidence bundle，复用相同的 claim verdict / pairwise effect 口径
- ETA proof suite 当前还区分 `eta-internal-rl-proof` 与 `eta-open-weight-residual-proof` 两类 manifest；真实 residual-control claim 必须绑定 `transformers-open-weight` capture / actual hook fire rate / fallback rate / prefix-aligned intervention 证据，不能由 trace 或 synthetic backend 单独支撑。当前 claim gate 要求 fallback rate 为 `0.0`、actual hook fire rate 至少 `0.75`、residual sequence 非空、intervention protocol valid；显式 fallback smoke run 必须保持 fail/quarantine 语义。`planned_layer_fraction` 只说明选了多少层，不作为 hook 健康硬门槛
- NL slow-loop 支持 ETA fast path 的 claim 需要读取 memory / credit / family payoff / long-horizon coverage 等 runtime evidence，不能只用“有 slow loop job 完成”作为结论

## 与其他能力域的关系

| 关系 | 能力域 | 说明 |
|------|--------|------|
| 依赖 | 评估体系 | claim verdict 消费 evaluation / benchmark evidence |
| 依赖 | 契约式运行时 | provenance 与 artifact 必须回溯到真实 runtime 产物 |
| 依赖 | 多时间尺度学习 / 时间抽象 | claim registry 需要把这些设计命题绑定到可观测 gate |
| 协作 | 调试体系 | blind review / paper-suite 工件是 widening 与审计面 |
| 协作 | Emergent Action Abstraction | `snapshot_replay_bundle.json` 是该能力域 Phase 1 落地的 artifact，进入 unified evidence bundle |
| 被依赖 | rollout / 外部汇报 | evidence bundle 是对外结论、候选比较与审稿材料的统一入口 |

## 初始 Claim Registry

- `claim_pe_multi_timescale_default`
  - 命题：`PE-first + multi-timescale` 是默认路径上的机制事实
  - 需要：`pe-first`、`multi-timescale-default`、`judge-gated-evolution`、`cross-session-growth`
- `claim_temporal_advantage_over_controls`
  - 命题：时间抽象与在线适应在 matched controls 前有稳定优势
  - 需要：dialogue / ETA pairwise effects 为正，且 gap 不只是单次最好结果
- `claim_beyond_scripted_canonical`
  - 命题：优势不只存在于 canonical scripted cases
  - 需要：perturbation / systematic replay / open-environment / held-out families 共同给证据
- `claim_external_human_legibility`
  - 命题：优势能被外部人类评审感知
  - 需要：blinded packet、多评审员评分、inter-rater agreement 与自动指标相关性
- `claim_eta_real_open_weight_residual_control`
  - 命题：ETA residual-control evidence 来自真实 open-weight runtime，而非 synthetic proof harness
  - 需要：`transformers-open-weight` primary backend、低 fallback rate、actual hook fire rate、prefix-aligned before/after control、matched trace fallback control、open-weight paper-suite manifest；`ci-smoke` 只能证明连线与可诊断失败，`retain` 需要 repeated full-suite aggregate
- `claim_nl_slow_loop_improves_eta_fast_path`
  - 命题：NL slow loop 对 ETA fast path 的初始化、family reuse 或 held-out payoff 有可测增益
  - 需要：slow-loop writeback evidence、credit-to-family write count、long-horizon payoff coverage、matched no-fast-prior / non-nested control
- `claim_companion_stateful_relationship`
  - 命题：companion 不是静态 support prompt，而是能感知当前对象状态、在 session 内调整、并在显式用户范围 memory 下跨 session 保留偏好
  - 需要：C1 state sensitivity、C2 within-session adaptation、C3 explicit cross-session retention、C4 default memory isolation、C5 default social scope isolation
  - 当前轻量 verdict 先绑定 `semantic-spine-ready`、`canonical_mean_semantic_spine_coverage`、`canonical_mean_cognitive_loop_readiness` 与 `cross-session-growth`，作为完整 companion 证据前的地基门
- repeated-run verdict 优先使用 paper-suite secondary metric summary 的 sample count / mean，避免只看单次 reference dashboard

## Dialogue Paper-Suite Evidence Map

本节冻结 dialogue paper-suite 当前 claim 的可操作证据口径。这里的 claim verdict 是 rollout / 外部汇报层的证据读数，不改变 runtime slot，也不成为学习源头。

### `claim_temporal_advantage_over_controls`

**命题**：`pe-eta` 的 canonical scripted 表现优于 matched controls，且优势不是只有结果分数，没有机制证据。

**retain 条件**：
- `canonical_pass_rate` 对 `pe-drive-off` 的 pairwise effect `ci_low > 0`
- `canonical_pass_rate` 对 `eta-off` 的 pairwise effect `ci_low > 0`
- `canonical_runtime_backbone_evidence_rate > 0`
- `canonical_mean_runtime_backbone_signal_quality > 0`

**weak 条件**：
- 对 `pe-drive-off` 与 `eta-off` 的 mean delta 为正，但 runtime backbone consistency 不完整，或 CI 尚未站稳

**主要 artifact**：
- `paper_suite_aggregate.json`
- `evidence_bundle.json`

**轻量测试节点**：
- `tests/test_dialogue_benchmark.py::test_dialogue_temporal_advantage_claim_requires_runtime_backbone_consistency`

### `claim_beyond_scripted_canonical`

**命题**：优势不只存在于 canonical scripted cases，还能扩展到 perturbation / held-out open environment，并经过一个不读 runtime telemetry 的 user policy 检查。

**retain 条件**：
- `open_pass_rate` 对 `pe-drive-off` 的 pairwise effect `ci_low > 0`
- open scenarios 包含 `open_heldout`
- 至少一个 open case 的 `OpenDialogueEpisodeState.user_policy_kind == "transcript-only"`
- hidden perturbation family label 不得出现在 runtime transcript / user input / assistant response 中（只作为 evidence-layer 分组）
- 至少一个 expected repair open case 的 `repair_observable` 为 true
- 至少一个 open case 的 `runtime_adaptation_evidence_observed` 为 true
- `perturbation_pass_rate_pe_eta > 0`

**weak 条件**：
- open 或 perturbation mean delta 为正，但 held-out 或 transcript-only policy evidence 不完整

**主要 artifact**：
- `paper_suite_aggregate.json`
- `reference_emergence_dashboard.json`
- `evidence_bundle.json`

**轻量测试节点**：
- `tests/test_dialogue_benchmark.py::test_transcript_only_user_simulator_ignores_runtime_telemetry`
- `tests/test_dialogue_benchmark.py::test_build_open_dialogue_case_report_uses_open_acceptance_surface`
- `tests/test_dialogue_benchmark.py::test_claim_beyond_scripted_requires_open_repair_and_no_hidden_label_leak`

### `claim_companion_stateful_relationship`

**命题**：companion 不是静态 support prompt，而是至少具备可审计的 semantic owner spine，可在后续 C1-C5 完整证据前证明“状态感知地基”存在。

**retain 条件（当前轻量口径）**：
- `semantic-spine-ready` gate 通过
- `canonical_mean_semantic_spine_coverage >= 1.0`
- `canonical_mean_cognitive_loop_readiness > 0`
- `cross-session-growth` gate 通过

**weak 条件**：
- `semantic-spine-ready` gate 通过
- `canonical_mean_semantic_spine_coverage >= 1.0`
- `canonical_mean_cognitive_loop_readiness > 0`

**主要 artifact**：
- `paper_suite_aggregate.json`
- `reference_emergence_dashboard.json`
- `evidence_bundle.json`

**轻量测试节点**：
- `tests/test_dialogue_benchmark.py::test_run_dialogue_paper_suite_repeated_benchmark_emits_interval_summaries`

### `claim_external_human_legibility`

**命题**：优势能被外部评审者在 blinded transcripts 上感知，而不只存在于内部 telemetry。

**retain 条件**：
- `human_ratings_aggregate.rater_count >= 3`
- `inter_rater_agreement >= 0.6`
- 满足以下任一项：
  - 任一维度 `correlation_with_automatic > 0.1`
  - 所有 pairwise preferences 都有 `pair_count > 0`、`win_rate >= 0.5`、`mean_score_delta > 0`

**weak 条件**：
- `rater_count >= 2`
- `inter_rater_agreement >= 0.4`

**主要 artifact**：
- `expert_review_packet_blinded.json`
- `expert_review_key_internal.json`（内部保留，不外发）
- `human_rating_template.csv`
- `human_ratings_aggregate.json`
- `evidence_bundle.json`

**轻量测试节点**：
- `tests/test_dialogue_benchmark.py::test_dialogue_human_rating_csv_aggregate_exports_external_claim`

### `claim_rare_heavy_net_benefit`

**命题**：rare-heavy / slow artifact 路径带来可测净收益，而不是把收益藏在总通过率中。

**retain 条件**：
- 存在 `pe-eta` vs `pe-eta-no-rare-heavy` matched control
- `canonical_pass_rate` 对 `pe-eta-no-rare-heavy` 的 pairwise effect `sample_count > 0`
- 该 effect 的 `ci_low > 0`

**weak 条件**：
- 存在 matched control 且 `mean_delta > 0`，但 CI 尚未站稳

**fail 条件**：
- 缺少 `pe-eta-no-rare-heavy` control，或 no-rare-heavy 对照没有正向 gap

**主要 artifact**：
- `paper_suite_aggregate.json`
- `evidence_bundle.json`

**轻量测试节点**：
- `tests/test_dialogue_benchmark.py::test_dialogue_rare_heavy_claim_requires_no_rare_heavy_control`

### Artifact 使用边界

- `expert_review_packet_blinded.json` 可以外发；`expert_review_key_internal.json` 只用于内部 unblinding 与 aggregate，不得进入 blind-review 包。
- `paper_suite_aggregate.json` 是 claim verdict 的主入口；它必须由 manifest / provenance / pairwise effects / optional human ratings 重新计算，而不是手工改写。
- `evidence_bundle.json` 是跨系统消费入口，包含 manifest、provenance、run summaries、aggregate metrics、pairwise effects、blind review packet 与 claim verdicts。
- 轻量测试只验证 claim 规则和 artifact shape；完整 empirical 结论仍必须来自 `paper-suite-small` / `paper-suite-full` repeated-run aggregate。

## Companion Evidence Map

本节冻结 companionship claim 的最小自动证据口径。它回答“陪伴能力是否只是固定 prompt”的问题，但不把固定 scripted benchmark pass 误写成人类级关系成熟。

### `claim_companion_stateful_relationship`

**命题**：companion 能根据当前对话对象状态调整表达，在同一 session 中形成连续状态，并在显式同一用户 / 同一生命体范围的共享 memory 下跨 session 保留偏好；默认 session 仍保持隔离，避免多租户串记忆。

**retain 条件**：
- C1 `state_sensitivity`: 同一 companion 面对 task vs emotional context 时，`interlocutor_state` 至少在 task focus / directness / rapport warmth 上分化，且 readout confidence 达标
- C2 `within_session_adaptation`: `low-mood-disclosure` 多轮内出现至少两个 expression intents，PE 有变化，且最终 interlocutor readout confidence 达标
- C3 `explicit_cross_session_retention`: 注入共享 `MemoryStore` 后，session B 能检索到 session A 写入的偏好
- C4 `default_memory_isolation`: 未显式注入共享 store 时，两个 session 的 `MemoryStore` 不共享
- C5 `default_social_scope_isolation`: 默认 companion turn 的 R16 scope 固定为 `primary/self`，`multi_party_identity` / `social_prediction` / `social_prediction_error` 以 SHADOW readout 发布且 social PE 默认为空
- AAC1 `alignment_pe_repair_visibility`: commitment alignment 从 AGREE→REJECT 的转变能进入 relationship PE，并产生 `DEFER_ONLY` repair follow-up policy
- RGM1 `regime_delayed_attribution_visibility`: dialogue-like repair/support regime evidence 能产生 delayed attribution、delayed credit records 与 evaluation readout metrics，且不要求硬编码 regime 切换
- RFL1 `reflection_writeback_stability`: reflection 能消费 dialogue slow-loop evidence，并通过 checkpoint / rollback 的 bounded apply path 写入 memory / regime evidence，不直接绕过 owner
- v2 `composite_score` 记录 C1-C5 gate score + widening transcript diversity；v2 transcript diversity 是 widening diagnostic，不单独作为 retain 硬门槛

**weak 条件**：
- C1/C2 通过，但 C3 只能通过人工脚本或临时状态证明，尚无显式 shared-memory gate

**fail 条件**：
- C1 不分化，或 C2 没有 session 内状态变化，或 C3 需要默认跨 session 串记忆才能通过，或 C5 显示默认 social scope 不是 `primary/self`

**主要 artifact**：
- `companion_evidence_report.json`
- `lifeform-bench --companion-evidence-report` stdout
- `companion_evidence_report.json.transcripts[]`：paraphrase / tone shift / delayed return / preference conflict 微场景 transcript，用于后续 blind review / human rating

**轻量测试节点**：
- `tests/lifeform_e2e/test_companion_learning_evidence.py`
- `tests/lifeform_e2e/test_companion_evidence_report.py`

**运行入口**：
- `lifeform-bench --companion-evidence-report`
- 可选 JSON：`lifeform-bench --companion-evidence-report --companion-evidence-json companion_evidence_report.json`

## Social Cognition Evidence Map

本节冻结 R16-R20 的最小自动证据口径。早期 slice 的目标不是证明人类级社会理解，而是证明社会认知状态没有退化为 renderer 文案分支或单一 `user_model` bucket。

### `claim_tom_owner_separation`

**命题**：Theory of Mind 状态至少在契约与显式 proposal path 上区分 belief / intent / feeling / preference；belief conflict 不会写入 preference owner，preference conflict 也不会伪装成 belief。

**retain 条件**：
- R16A `active_identity_memory_scope`: explicit `EnvironmentEvent.frame` publishes ACTIVE `multi_party_identity`, and memory writes inherit the same subject / audience scope without renderer inference
- R16B `active_social_pe_memory_visibility`: ACTIVE `social_prediction` / `social_prediction_error` publish MEMORY_VISIBILITY prediction/error and negative credit when active scope suppresses cross-subject memory
- T1 `tom_owner_contract`: `OtherMindRecordKind` 有限枚举覆盖 belief / intent / feeling / preference，四类 snapshot 会拒绝 wrong-kind record
- T2 `explicit_tom_proposal_path`: 显式 proposal 可以填充目标 ToM owner，final wiring 默认不把通用 semantic runtime 当 ToM classifier
- T3 `false_belief_preference_separation_probe`: 人工 false-belief + preference-conflict probe 中，belief 与 preference 分别落入各自 owner，record kind 不混写
- T4 `structured_tom_runtime_path`: `LLMToMProposalRuntime` 的结构化 JSON 输出能填充目标 ToM owner， malformed / low-confidence 输出不会伪造 durable record
- T5 `affect_preference_separation_probe`: 同一输入中的 transient feeling 与 durable preference 分别进入 `feeling_about_other` / `preference_about_other`，不混写
- R1 `wrong_addressee_role_pe_credit`: 人工 wrong-addressee role PE 可以转成 shared credit，证明 role mistake 能进入 PE/credit 链路而不是 renderer 规则
- R2 `role_prediction_diagnostic_visibility`: EnvironmentEvent role frame 产生 `ROLE_ASSIGNMENT` prediction，且 `response_assembly.semantic_record_counts` 可诊断性显示 role prediction count
- R18A `active_role_frame_diagnostics`: default final wiring 中 ACTIVE `conversational_role` 消费 EnvironmentEvent role frame，并发布 diagnostics，不泄漏到 renderer 文案
- G1 `common_ground_diagnostic_visibility`: 显式 dyad/group common-ground atoms 可以进入 `common_ground` owner，并在 `response_assembly.semantic_record_counts` 诊断性显示 atom count
- G2 `structured_common_ground_runtime_path`: 结构化 common-ground runtime 可以把 dyad/group JSON proposal 写入 `common_ground` owner，并显示为 diagnostic count
- G3 `reference_repair_common_ground_probe`: repair / clarification evidence 可以进入 dyad common-ground atom，而不是由 renderer 文案推断
- GROUP1 `group_diagnostic_visibility`: 显式 group identity / joint commitment 可以进入 `groups` owner，并在 `response_assembly.semantic_record_counts` 诊断性显示 group 与 joint commitment count

**fail 条件**：
- ToM proposal 进入 `user_model.stable_preferences` 才能通过；或四类 ToM 状态共享同一 untyped owner；或 renderer / raw text 分支直接决定 belief/preference 行为。

**轻量测试节点**：
- `tests/contracts/test_social_cognition_contracts.py`
- `tests/test_social_tom.py`
- `tests/test_social_cognition_evidence.py`

**主要 artifact**：
- `social_cognition_evidence_report`（Python API: `lifeform_evolution.run_social_cognition_evidence()`）
- `social_cognition_evidence_report_to_dict(report)`：T1-T3 gate payload，可被后续 CLI / bundle 引用
- CLI stdout: `lifeform-bench --social-cognition-evidence-report`
- JSON artifact: `lifeform-bench --social-cognition-evidence-report --social-cognition-evidence-json social_cognition_evidence_report.json`

## 变更日志

- 2026-05-02: Social Cognition evidence report 增加 R16A active identity memory scope gate，覆盖 EnvironmentEvent frame → ACTIVE multi_party_identity → memory subject/audience scope 链路
- 2026-05-02: Social Cognition evidence report 增加 R16B active social PE memory visibility gate，覆盖 ACTIVE social_prediction/social_prediction_error → memory visibility PE → negative credit 链路
- 2026-05-02: 强化 `claim_beyond_scripted_canonical`，新增 hidden perturbation label non-leak、repair observable 与 runtime adaptation evidence 条件，仍复用现有 dialogue benchmark / paper-suite / evidence bundle
- 2026-05-01: 新增 Dialogue Paper-Suite Evidence Map，冻结 temporal advantage、beyond scripted、external human legibility、rare-heavy net benefit 四类 dialogue claim 的 retain / weak / fail 条件、artifact 边界与轻量测试入口
- 2026-05-02: 新增 Companion Evidence Map，冻结 C1-C4 companion stateful-relationship claim、运行入口与轻量测试节点；v2 增加 widening transcript artifact 与 composite score（diagnostic，不替代 C1-C4 retain gate）
- 2026-05-02: 增加 C5 default social scope isolation gate，覆盖 R16 `primary/self` 默认 scope 与空 social PE 链路，避免 companion v1 在多人化迁移中隐式串人
- 2026-05-02: 增加 AAC1 alignment PE repair visibility gate，覆盖 AAC commitment alignment reject → relationship PE → defer-only repair follow-up 链路
- 2026-05-02: 增加 RGM1 regime delayed attribution visibility gate，覆盖 dialogue repair/support signal → RegimeSnapshot delayed attribution → credit/evaluation readout 链路
- 2026-05-02: 增加 RFL1 reflection writeback stability gate，覆盖 dialogue slow-loop evidence → bounded reflection apply → checkpoint/rollback 链路
- 2026-05-02: 新增 Social Cognition Evidence Map，冻结 R17 ToM owner separation 的 T1-T3 轻量证据门槛
- 2026-05-02: 增加 Social Cognition evidence report artifact，T1-T3 由 `lifeform_evolution.run_social_cognition_evidence()` 汇总输出；CLI 支持 `--social-cognition-evidence-report` 与 `--social-cognition-evidence-json`
- 2026-05-02: Social Cognition evidence report 增加 R1 wrong-addressee role PE credit gate，覆盖 R18 role mistake → credit 链路
- 2026-05-02: Social Cognition evidence report 增加 T4/T5 structured ToM gates，覆盖 LLMToMProposalRuntime 结构化输出和 affect/preference 分离
- 2026-05-02: Social Cognition evidence report 增加 R2 role prediction diagnostic visibility gate，覆盖 R18 role prediction → response_assembly diagnostic count 链路
- 2026-05-02: Social Cognition evidence report 增加 R18A active role frame diagnostics gate，覆盖 ACTIVE conversational_role → diagnostics 且不进入 renderer 文案
- 2026-05-02: Social Cognition evidence report 增加 G1 common-ground diagnostic visibility gate，覆盖 R19 explicit atoms → response_assembly diagnostic count 链路
- 2026-05-02: Social Cognition evidence report 增加 G2/G3 structured common-ground gates，覆盖 R19 structured runtime → owner atom 与 reference repair → dyad atom 链路
- 2026-05-02: Social Cognition evidence report 增加 GROUP1 group diagnostic visibility gate，覆盖 R20 explicit group state → response_assembly diagnostic count 链路
- 2026-05-02: 重写 `docs/specs/emergent-action-abstraction.md` 的 replay artifact 为 `snapshot_replay_bundle.json`，由 existing snapshots 导出，不依赖 trace owner
- 2026-04-25: 补充 ETA open-weight residual-control 与 NL slow-loop-support claim 的 evidence 边界，明确 synthetic / trace backend 不能单独支撑真实 residual-control claim
- 2026-04-26: 细化 real open-weight gate：把 planned layer fraction 与 actual hook fire rate 分离，新增 prefix-aligned intervention 与 smoke/full evidence tier 边界
- 2026-04-25: 初始版本，建立 claim-to-evidence / blind-review / pairwise-effect / evidence-bundle 的统一口径
