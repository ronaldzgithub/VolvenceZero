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
- `snapshot_replay_bundle.json`（Phase 1 runtime export shape started）：导出 existing snapshots 的 action replay section（`PredictionActionContext` / `temporal_abstraction.closed_segments` / `prediction_error` / `credit` summaries）用于再现与证据审阅，不引入 trace runtime schema；完整 paper-suite bundle 挂载仍可后续扩展

当前实现口径：

- `volvence_zero.agent.paper_suite` 提供共享 `ClaimVerdict` 与 `EvidenceBundle`
- dialogue / ETA paper-suite aggregate 会额外发布 pairwise effects 与 claim verdicts
- dialogue paper-suite export 会同时导出 blinded packet、internal key、rating template、rating aggregate 与 unified evidence bundle
- dialogue emergence dashboard / paper-suite metric values 发布 `canonical_mean_semantic_spine_coverage`、`canonical_mean_cognitive_loop_readiness` 以及 open-environment 对应读数；这些是 semantic owner 快照的证据读数，不作为学习源头
- dialogue NL essence assessment 发布 `semantic-spine-ready` gate，用于审计核心 semantic owner spine 是否具备完整 coverage 与非零 readiness；该 gate 目前不进入默认 required gate 列表
- dialogue paper-suite manifest 将 `canonical_mean_semantic_spine_coverage` 与 `canonical_mean_cognitive_loop_readiness` 列为 secondary metrics；companion stateful relationship verdict 优先消费 repeated-run summary，reference dashboard 只作为 fallback
- dialogue paper-suite export 可额外导出 `semantic_proposal_quality_shadow.json`，并在 `EvidenceBundle.reference_artifacts` 中登记同一 payload；该 payload 是 non-gating shadow diagnostic，不参与 retain/fail verdict
- ETA paper-suite export 会导出统一 evidence bundle，复用相同的 claim verdict / pairwise effect 口径
- ETA proof suite 当前还区分 `eta-internal-rl-proof` 与 `eta-open-weight-residual-proof` 两类 manifest；真实 residual-control claim 必须绑定 `transformers-open-weight` capture / actual hook fire rate / fallback rate / prefix-aligned intervention 证据，不能由 trace 或 synthetic backend 单独支撑。当前 claim gate 要求 fallback rate 为 `0.0`、actual hook fire rate 至少 `0.75`、residual sequence 非空、intervention protocol valid；显式 fallback smoke run 必须保持 fail/quarantine 语义。`planned_layer_fraction` 只说明选了多少层，不作为 hook 健康硬门槛
- NL slow-loop 支持 ETA fast path 的 claim 需要读取 memory / credit / family payoff / long-horizon coverage 等 runtime evidence，不能只用“有 slow loop job 完成”作为结论
- Phase 2/3 SHADOW candidate smoke 现在有独立 artifact schema：`phase2_shadow_evidence_smoke.json`，`schema_version="phase2-shadow-evidence-smoke.v1"`。该 artifact 由 `scripts/run_phase2_shadow_evidence_smoke.py` 生成，覆盖 SYS-1 / COG-1 / COG-2 / COG-3 单项 profile 与可选 Phase 3 组合 profile；它是 SHADOW review artifact，不是 retain/fail claim verdict 的替代。
- Phase 2/3 multi-seed evidence 现在有独立 artifact schema：`phase2_shadow_evidence_multiseed.json`，`schema_version="phase2-shadow-evidence-multiseed.v1"`；阶段 D decision report schema 为 `phase2_shadow_decision_report.json`，`schema_version="phase2-shadow-decision-report.v1"`。二者仍是 SHADOW/decision-support artifact，不直接替代完整 paper-suite claim verdict。

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

### Same-substrate Companion Bench ablation claims（debt #87；详见 [`companion-ablation.md`](./companion-ablation.md)）

> **冻结的 thesis 第一阶段 claim registry SSOT 见 [`human-world-model-ablation.md`](./human-world-model-ablation.md)**（5 条 retain claim + 8 臂 matched-control matrix + 证据门槛 + 4 态分级 + kill 条件）。下面是同基底工具链已实现的 4 条；registry 相对本节新增 `claim_component_causal_contribution`（PE/ETA/主动学习逐个因果切分，四臂待迁同基底）。

这组 claim 把"人类世界模型 thesis 第一阶段"压成可证伪、同基底的 matched-control。所有 track 跑同一份冻结 Qwen，由 `compare_companion_ablation.py` 产出 verdict（retain 需 delta>0 且 bootstrap CI 非重叠下界 ci_low>0）。

- `claim_pipeline_gt_raw`
  - 命题：volvence 完整 pipeline 在 Companion Bench 上优于裸 Qwen
  - 需要：`raw` 与 `volvence` track 同基底；final_mean delta>0 且 ci_low>0
- `claim_gt_standard_layers`
  - 命题：volvence 优于标准 memory wrapper（ref-harness）**且**优于标准开源 agent 框架（camel）——回应"给 GPT/Claude 套个 wrapper 你们还赢吗"
  - 需要：`ref-harness` 与 `camel` track 同基底；两条 pairwise effect 都 retain
- `claim_training_adds_value`
  - 命题：训练 bootstrap 有增量（volvence > volvence-cold）
  - 需要：`volvence-cold` 与 `volvence` 同基底；delta>0 且 ci_low>0
- `claim_heldout_cohort_stable`
  - 命题：优势在 held-out 场景跨 seed 稳定
  - 需要：held-out + 多 seed 跑（arc_count 足够）+ 相对 CI 半宽足够紧
- 红线（缺一不可，否则 verdict 不可外引）：same-substrate guard VERIFIED；裁判/用户模拟器非 Qwen（#71/#72）；4 条 CompanionBench attestation 全 True；judge robustness/calibration 证据在档；held-out 文本无泄漏
- 四态结论：`kill-criteria-triggered` / `wiring-ready` / `weak-positive` / `first-stage-retained`；`world-model-extension-ready` 需物理侧独立 benchmark，本链不产出

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

### `phase2_shadow_evidence_smoke.json`

**用途**：Phase 2/3 SHADOW profile 的最小 review artifact。用于检查 profile wiring、candidate readout、metric extraction、deterministic head-to-head 与 cross-generation gate evidence shape 是否完整。

**生成命令**：

```bash
python scripts/run_phase2_shadow_evidence_smoke.py --synthetic-runner --case-limit 1
```

真实 runner evidence 去掉 `--synthetic-runner`；组合 profile 加 `--include-phase3-combos`。

**稳定字段**：

- `schema_version == "phase2-shadow-evidence-smoke.v1"`
- `artifact_kind == "phase2_shadow_evidence_smoke"`
- `runner_kind`: `synthetic` / `default`
- `include_phase3_combos: bool`
- `provenance`
  - `git_sha`
  - `git_branch`
  - `working_tree_dirty`
  - `python_version`
  - `platform`
- `profile_labels`
- `focus_metric_means`
- `focus_metric_deltas_from_baseline`
- `head_to_head_results`
- `cross_generation_gate_evidence`

**Sidecar manifest**：

- 文件名：`phase2_shadow_evidence_manifest.json`
- `schema_version == "phase2-shadow-evidence-manifest.v1"`
- `artifact_kind == "phase2_shadow_evidence_manifest"`
- `source_schema_version == "phase2-shadow-evidence-smoke.v1"`
- `artifacts[]`
  - `path`
  - `sha256`
  - `size_bytes`
- `provenance`（与主 JSON payload 一致）

**Manifest 校验命令**：

```bash
python scripts/verify_phase2_shadow_evidence_manifest.py artifacts/phase2_shadow_evidence_smoke/phase2_shadow_evidence_manifest.json
```

该命令 fail-loudly 校验所有登记 artifact 的 `sha256` / `size_bytes`。reviewer 在引用 JSON / Markdown evidence 前应先跑此命令。

**边界**：

- Synthetic artifact 只验证 schema / wiring / metric surface；不能作为 retain 证据。
- Default runner artifact 可以作为 SHADOW review evidence，但升 ACTIVE 仍需 multi-seed paper-suite-small 或等价 evidence。
- Markdown sibling `phase2_shadow_evidence_smoke.md` 是人类 review 面；JSON 是机器可读 SSOT。
- Manifest 是 artifact 完整性 sidecar；reviewer 应优先用 manifest 校验 JSON / Markdown 是否被改写。

### `phase2_shadow_evidence_multiseed.json` 与 decision report

**用途**：聚合 Phase 2/3 SHADOW profiles 的多次 run，给阶段 D 决策提供稳定输入。

**生成命令**：

```bash
python scripts/run_phase2_shadow_evidence_multiseed.py --case-limit 4 --seeds 0 1 2 3 4 --output-dir artifacts/phase2-shadow-real-multiseed
python scripts/build_phase2_shadow_decision_report.py artifacts/phase2-shadow-real-multiseed/phase2_shadow_evidence_multiseed.json --output-dir artifacts/phase2-shadow-real-multiseed
```

**边界**：

- `phase2_shadow_evidence_multiseed.json` 汇总 mean / std / stderr 与 deterministic head-to-head；它不做 ACTIVE 决策。
- `phase2_shadow_decision_report.json` 给出 `ACTIVE_CANDIDATE` / `REMAIN_SHADOW` / `DISABLED` 建议；真实合并仍需人工 review + rollback plan。
- Synthetic runner 的 decision 永远只能是 `REMAIN_SHADOW`。

### Blind Review External Dispatch（recruitment-agnostic）

本节描述 `claim_external_human_legibility` 的实际外发流程。**该流程刻意不绑定具体招募/分发平台**（Google Form / Prolific / 内部团队都可以承载），只规定文件清单、rater 任务说明与回收 schema，让运营层灵活替换。

**步骤**

1. 生成 packet：跑 `bash scripts/run_dialogue_paper_suite.sh artifacts/dialogue_paper_suite paper-suite-small`，产物里包含
   - `expert_review_packet_blinded.json` —— 外发安全
   - `expert_review_key_internal.json` —— 仅内部
   - `human_rating_template.csv` —— 1 个空模板，含 header `rater_id,item_id,sample_id,blinded_label,dimension_id,score`
2. 招募 rater（≥3 人，独立填写）：把 `expert_review_packet_blinded.json` 与 rater 操作说明分发给 rater；操作说明里至少包括 packet 里的 `review_dimensions[*].prompt`、量表 1–5、不允许查看 packet 之外的内部 telemetry。**任何分发渠道都可以**（云盘 / 邮件 / 表单 / 平台），只要 rater 最终交回**一个 CSV per rater**（保留 header）。
3. 回收 + 合并：把所有 rater 的 CSV 放到一个目录，例如 `artifacts/dialogue_paper_suite/ratings/`。每个 rater 一个 CSV，rater_id 不冲突。然后用 `volvence_zero.agent.load_dialogue_human_rating_entries_csv_dir(csv_dir)` 一次性加载（rater_id 冲突会抛错防止重复计票，紧急情况可用 `forbid_rater_id_collision=False`）。
4. Aggregate + 重导：把合并后的 entries 与 `expert_review_key_internal.json` + 原始 `reference_run_report` 喂给 `aggregate_dialogue_human_ratings(packet, entries, internal_key, reference_report)`，得到 `human_ratings_aggregate`；再调用 `export_dialogue_paper_suite_artifact_bundle(report, output_dir, human_ratings_aggregate=aggregate)` 重导 bundle，让 `claim_external_human_legibility` verdict 反映真实评分。
5. 验收：`paper_suite_aggregate.json` 中
   - `claim_external_human_legibility.status == "retain"` 需要 `rater_count >= 3` AND `inter_rater_agreement >= 0.6` AND（任一维度 `correlation_with_automatic > 0.1` 或所有 pairwise preferences `win_rate >= 0.5` 且 `mean_score_delta > 0`）
   - `weak` 仅需 `rater_count >= 2` 且 `inter_rater_agreement >= 0.4`

**安全护栏**

- `tests/test_dialogue_benchmark.py::test_blind_packet_transcripts_have_no_profile_label_leak` —— 验证 packet 里的 transcript 不含 `pe-eta` / `pe-drive-off` / `eta-off` 这类 profile-label 字符串（case_id 是英文常用词如 `repair` / `goal_drift` 时不能误报，因为它们必然出现在话题中）
- `tests/test_dialogue_benchmark.py::test_dialogue_paper_suite_artifact_bundle_exports_expert_review_packet` —— 验证 `source_profile_label` 不出现在 packet JSON 全文中
- `human_ratings_aggregate.json` 里包含真实 profile_label，**只能内部消费**，不要随 packet 一起外发

**当前 inter-rater agreement 算法**

实现见 `_dialogue_inter_rater_agreement`：对每个 `(item_id, sample_id, dimension_id)` 单元，计算所有 rater 两两绝对差的均值，映射到 [0, 1]（除以 4，假定 1–5 量表），最后对所有单元再取均值。这与 Krippendorff's alpha / Cohen's kappa 不直接可比。Krippendorff alpha v2 是 backlog 项；外发数据对外引用时请明确指出口径。

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
- AAC1 / RGM1 / RFL1 在 dialogue paper-suite 层是**外部注入**：上游 `lifeform_evolution.companion_evidence.run_companion_evidence()` 生产 `CompanionEvidenceGate` 列表后，由编排层（如 lifeform-bench）通过 `export_dialogue_paper_suite_artifact_bundle(..., companion_structural_gates=((gate_id, passed), ...))` 注入。dialogue 层不导入 `lifeform_evolution`，只读不可变 `(gate_id, passed)` 元组（R8 / SSOT 合规）。当未注入时，verdict 的 retain 检查回退到 4 项地基（`semantic-spine-ready` + 两个 spine 指标 + `cross-session-growth`），与历史行为一致；注入后 retain 还要求所有提供的 structural gate 通过
- v2 `composite_score` 记录 C1-C5 gate score + widening transcript diversity；v2 transcript diversity 是 widening diagnostic，不单独作为 retain 硬门槛

**weak 条件**：
- C1/C2 通过，但 C3 只能通过人工脚本或临时状态证明，尚无显式 shared-memory gate

**fail 条件**：
- C1 不分化，或 C2 没有 session 内状态变化，或 C3 需要默认跨 session 串记忆才能通过，或 C5 显示默认 social scope 不是 `primary/self`

**主要 artifact**：
- `companion_evidence_report.json`
- `lifeform-bench --companion-evidence-report` stdout
- `companion_evidence_report.json.transcripts[]`：paraphrase / tone shift / delayed return / preference conflict 微场景 transcript，用于后续 blind review / human rating
- `dialogue_option_discovery_report.json`（Phase A trajectory diagnostic, non-gating）：从 `DialogueBenchmarkTurn` 与 optional snapshot replay `action_replay` 读取 `switch_gate`、`active_abstract_action`、`prediction_error`、`closed_segments` / `z_t_digest`，报告 `termination_event_count`、`option_duration_mean`、`abstract_action_diversity`、`pe_spike_near_termination_rate`、`option_reuse_across_cases`。该 artifact 验证 ETA 时间抽象是否在 dialogue trajectory 上有可审计形状，但当前不改变 retain 条件。
- `pe_counterfactual_closure_report.json`（Phase A trajectory diagnostic, non-gating）：复用现有 `pe-eta` / `pe-drive-off` / `pe-eta-pe-readout-only` / `eta-off` profile comparison，报告 `pe_to_credit_drop`、`pe_to_behavior_drop`、`readout_only_gap`、`eta_dependency_gap`、`closure_strength`。该 artifact 验证 PE drive 与 readout-only / ETA-off 对照之间的因果闭环迹象，但当前不替代后续 delayed / shuffled / wrong-sign PE harness。
- `longitudinal_dialogue_report.json`（PhaseB longitudinal trajectory diagnostic, non-gating）：按 persona/session 聚合多 session 证据，报告 `retention_rate`、`isolation_pass_rate`、`adaptation_trend`、`drift_risk_score`、`trajectory_strength`、`cross_session_verdict`。首版支持从 `DialogueLongitudinalBenchmarkReport` 派生，也支持显式 `LongitudinalDialogueSessionEvidence`（shared-memory / isolation evidence 由上游 owner/report 产生后注入）。该 artifact 验证同一 virtual user 多 session 轨迹的连续性地基，但当前不改变 retain 条件。
- `nl_ablation_matrix_report.json`（PhaseC NL diagnostic, non-gating）：聚合 `full-nl` / `no-ssl` / `no-rl` / `no-reflection` / `no-rare-heavy` / `no-fast-prior` / `timescale-off` 的 structured metrics，报告 `cross_session_growth_score`、`heldout_payoff_score`、`memory_churn_risk`、`behavior_drift_risk`、`slow_to_fast_transfer_gain`、`full_nl_advantage`。首版支持从 dialogue comparison report 或 explicit variant metrics 生成；没有真实 profile 的 variant 必须显式输入，不能伪造。
- `memory_stratum_flow_report.json`（PhaseC memory diagnostic, non-gating）：读取 `MemorySnapshot` 或 normalized dict evidence，报告 stratum progression、promotion/decay pressure、derived index activity、lifecycle signal strength 与 `memory_flow_strength`。它只读 memory owner 发布的 snapshot/readout，不推断 raw text。
- `regime_lockin_report.json`（PhaseC regime diagnostic, non-gating）：读取 `RegimeSnapshot` 或 normalized dict evidence，报告 `lockin_strength`、`switch_rate`、`hysteresis_proxy`、`delayed_attribution_strength`、`sequence_payoff_strength`、`regime_identity_stability`。当前没有 runtime-level hysteresis owner，`hysteresis_proxy` 只由候选 regime 波动但 active regime 保持的结构化证据派生。

**Phase A trajectory evidence 边界**：

- 两个 Phase A artifact 均通过 `export_dialogue_paper_suite_artifact_bundle(..., include_phase_a_trajectory_reports=True)` 或显式传入 report 后进入 `EvidenceBundle.reference_artifacts`。
- 它们是 readout / evidence artifact，不写 owner、不成为 learning source、不新增 runtime slot。
- `dialogue_option_discovery_report` 首版允许 `evidence_quality="turn-telemetry-only"`，因为完整 per-turn replay accumulation 尚未进入 runtime；若提供 `snapshot_replay_artifact`，则升级为 `snapshot-replay+turn-telemetry`。
- `pe_counterfactual_closure_report` 首版只覆盖已有 profile counterfactual；delayed / shuffled / wrong-sign PE 仍是后续 harness，不在本阶段伪造。
- `claim_companion_stateful_relationship` 的 retain 条件暂不消费这两个 artifact；它们为后续从机制 gate 升级到 trajectory gate 提供证据输入。

**PhaseB longitudinal trajectory evidence 边界**：

- `longitudinal_dialogue_report` 通过 `export_dialogue_paper_suite_artifact_bundle(..., include_phase_b_longitudinal_report=True)` 或显式传入 report 后进入 `EvidenceBundle.reference_artifacts`。
- 它是 readout / evidence artifact，不写 owner、不新增 runtime slot、不成为 learning source。
- 首版 v1 personas 是 `direct-but-overwhelmed`、`slow-trust-repair`、`boundary-sensitive`、`preference-conflict`、`delayed-return`，每个 persona 先 3 sessions，作为 synthetic trajectory surface；不能把 v1 结果夸大成人类级 companion retain。
- Shared-memory retention 与 default isolation 是双轨指标：`explicit_retention_observed` / `retrieved_preference_count` 证明显式共享路径存在，`default_isolation_preserved` 证明默认隔离仍保持。
- `claim_companion_stateful_relationship` 的 retain 条件暂不消费该 artifact；未来升级 retain 需要至少同时满足 multi-session trend、memory retention、default isolation、preference-conflict repair 与 human review anchor。

**PhaseC NL / memory / regime longitudinal evidence 边界**：

- `nl_ablation_matrix_report`、`memory_stratum_flow_report`、`regime_lockin_report` 可显式传入 `export_dialogue_paper_suite_artifact_bundle(...)`，或在后续完整 paper-suite runner 中按需生成，最终进入 `EvidenceBundle.reference_artifacts`。
- 三个 artifact 都是 non-gating diagnostics，不写 owner、不新增 runtime slot、不成为 learning source。
- `nl_ablation_matrix_report` 的首版 explicit variant metrics 是 proof harness 入口；若缺少 no-SSL / no-RL / no-reflection 等真实 profile，不得把 full-NL claim 写成 retain，只能作为待补对照。
- `memory_stratum_flow_report` 的 stratum flow 只基于 `MemorySnapshot.total_entries_by_stratum`、`pending_promotions`、`pending_decays`、`cms_band_vectors`、`lifecycle_metrics` 等 owner-owned readouts。
- `regime_lockin_report` 的 lock-in / hysteresis 是从 `RegimeSnapshot.turns_in_current_regime`、`regime_changed`、`candidate_regimes`、`delayed_attributions`、`sequence_payoffs` 派生的 readout，不改变 regime selection policy。
- 后续把 `claim_companion_stateful_relationship` 升级到 trajectory retain 时，至少需要同时具备：PhaseA ETA/PE trajectory evidence、PhaseB longitudinal dialogue evidence、PhaseC NL positive ablation gap、memory stratum flow、regime lock-in，以及 external human review anchor。

**轻量测试节点**：
- `tests/lifeform_e2e/test_companion_learning_evidence.py`
- `tests/lifeform_e2e/test_companion_evidence_report.py`
- `tests/test_dialogue_benchmark.py::test_build_dialogue_option_discovery_report_from_turns_and_replay`
- `tests/test_dialogue_benchmark.py::test_build_pe_counterfactual_closure_report_from_existing_profiles`
- `tests/test_dialogue_benchmark.py::test_dialogue_paper_suite_export_writes_phase_a_trajectory_reports`
- `tests/test_dialogue_benchmark.py::test_build_longitudinal_dialogue_report_from_session_evidence`
- `tests/test_dialogue_benchmark.py::test_dialogue_paper_suite_export_writes_phase_b_longitudinal_report`
- `tests/test_dialogue_benchmark.py::test_build_nl_ablation_matrix_report_from_explicit_variant_metrics`
- `tests/test_dialogue_benchmark.py::test_build_memory_stratum_flow_report_from_dict_snapshots`
- `tests/test_dialogue_benchmark.py::test_build_regime_lockin_report_from_dict_snapshots`
- `tests/test_dialogue_benchmark.py::test_dialogue_paper_suite_export_writes_phase_c_reports`
- `tests/test_eta_nl_clean_action_abstraction.py::test_dialogue_option_discovery_accepts_snapshot_replay_context`

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

## EQ Evidence-Chain Closure Bundle (Wave E1-E5)

Wave E1-E5 把 `default NoOp` 路径下不出 evidence 的链路打通；所有面都通过单一 manifest 暴露，下游 cite 按 `evidence_bundle.json` gate 读，不再各家拼装。

**单命令入口**：`scripts/run_eq_evidence_bundle.sh [ROUNDS] [BUNDLE_DIR]`

**Bundle 装配器**：`python -m lifeform_evolution.evidence_bundle assemble --bundle-dir <dir> --output <path>`

### Gates

| Gate id | 关闭条件 | 关联 debt |
|---|---|---|
| `debt_10b_item3` | 至少一个 long-form scenario 的 `tom_records_total_last > 0` 且 `common_ground_dyad_atoms_total_last > 0`（必须用真 LLM proposal runtime） | #10B item 3 |
| `debt_10c_il_rapport_snr` | 跨 scenario 的 `cross_scenario_summary.il_rapport_trend_snr_mean ≥ 1.5` | #10C |
| `debt_11_long_form_coverage` | 至少一次 bundle 报告 `pe_window_filled_scenario_ratio ≥ 0.5` | #11 follow-up |
| `wave_e4_multi_party_keying` | 3-party scenario artifact 存在；F3 facets 由 `tests/contracts/test_multi_party_shadow_evidence.py` 静态守门 | — |
| `debt_6_rewarding_state_head_promotion` | rollback drill test 存在且通过；promotion 仍需真 trace evidence | #6 |
| `debt_7_pe_critic_head_promotion` | rollback drill test 存在且通过；promotion 仍需真 trace evidence | #7 |

### Required artifacts

bundle dir 中必须有：

- `<scenario_id>_longitudinal.json` × 4（4 个 long-form scenario）
- `long-form-three-party-arc_longitudinal.json`（multi-party SHADOW probe）
- 装配器自动产 `evidence_bundle.json`

每个 artifact 的 SHA-256 + 大小记录在 `evidence_bundle.json.artifact_provenance` 内，外部审阅可通过 sha256 验真。

### 不变量

- 所有 evidence gate 必须 typed JSON readout，不允许从文本输出推断（`no-keyword-matching-hacks` rule）
- bundle 装配器是纯只读脚本，不 mutate 任何 owner / runtime；它只读 artifact 并写 manifest
- 每个 gate 的 `passed` 字段必须能从 typed metric 读出，不依赖人工判读

## 变更日志

- 2026-05-09: Wave E5 (Evidence-Chain Closure milestone) 落地 EQ Evidence-Chain Closure Bundle 段：单命令入口 `scripts/run_eq_evidence_bundle.sh`、装配器 `python -m lifeform_evolution.evidence_bundle assemble`、6 条 typed gate verdict、artifact provenance（sha256 + size）。E1-E4 落地的所有 owner / family / longitudinal 字段已通过 bundle 暴露，外部 cite 可通过单一 `evidence_bundle.json` 路径读取所有 verdict。
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
