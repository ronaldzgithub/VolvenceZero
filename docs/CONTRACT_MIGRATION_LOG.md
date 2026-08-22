# Contract Migration Log

> Status: migration / implementation log
> Last updated: 2026-07-13

## W2 intent-alignment remediation (2026-07-13): CP-04 consent gate + CP-14 delayed attribution e2e

Action-loop residuals; all changes append-only / default-compatible:

- CP-04: `AffordanceModule.dependencies` += `boundary_consent`. When
  `BoundaryConsentSnapshot.external_action_blocked` is True, TOOL / SHELL
  candidates get typed `blocked_reason="boundary_consent:external_action_blocked"`
  and cannot be selected; snapshot description carries
  `consent_external_blocked`. Missing consent snapshot = no-op (no silent
  degradation).
- CP-04: `PromptPlanner.plan` gained optional `affordance_snapshot`; a selected
  affordance adds the new `SectionId.AFFORDANCE_OFFER` section (owner-approved
  offer, never auto-invokes) plus typed tags
  `affordance=selected:<name>;score:<float>` and `affordance_blocked=<n>`.
  `GroundedResponseSynthesizer` gained optional `affordance_snapshot_provider`
  (+ `with_affordance_provider`); `LifeformSession` exposes
  `affordance_snapshot` and `Lifeform.create_session` wires the provider.
  `lifeform-expression` now depends on `lifeform-affordance`.
- CP-14: `volvence_zero.credit` re-exports `derive_segment_closure_credit_records`
  (was gate.py-only). No behavior change; import-surface fix found by the new
  e2e.

Verified by `packages/lifeform-core/tests/test_affordance_consent_gate.py`,
`packages/lifeform-expression/tests/test_prompt_planner_affordance.py`,
`tests/lifeform_e2e/test_delayed_attribution_e2e.py` (multi-turn delayed
attribution + segment-closure match/mismatch table).

## W1 intent-alignment remediation (2026-07-13): learned gates + social settlement

Learning-authenticity uplift; all changes append-only / default-compatible:

- `DualTrackModule.__init__` += optional `gate_learner`
  (`volvence_zero.dual_track.gate_learner.DualTrackGateLearner`, session-held
  bounded online-SGD). When wired, `DualTrackSnapshot.learned_gate_shadow`
  comes from the learner (scored next turn against the PE owner's realized
  task/relationship outcome); when None, the fixed-prior
  `derive_learned_gate_shadow` formula remains as fallback. Report-only.
- Semantic owner prediction v2: `SemanticStateStore` gained per-slot
  `_OwnerForecastLearner` (`forecast_owner_vector` / `settle_owner_forecast`
  / `owner_forecast_stats`; not part of hydration payload v1).
  `OwnerPredictionSignal.predicted_vector` for the five first-wave owners is
  now the learned forecast (cold-start byte-identical to the v1 persistence
  prior); descriptions read "v2-learned". PE settlement surface unchanged.
- W1.C (CP-16/17 core): new session-held
  `volvence_zero.social.record_store.SocialRecordStore` (ToM record windows,
  common-ground atom windows, pending predictions). `settle_pending_predictions`
  + `apply_outcome_to_record` implement embedding-similarity settlement and the
  ACTIVE -> CONTESTED -> RETIRED promote/retire table. Append-only snapshot
  field `settled_errors: tuple[SocialPredictionError, ...] = ()` on the four
  ToM snapshots and `CommonGroundSnapshot`; `SocialPredictionErrorModule`
  forwards owner-settled errors (dependencies += the four ToM slots +
  `common_ground`). ToM/common-ground modules gained optional `record_store`.
- `AgentSessionRunner` holds `DualTrackGateLearner` + `SocialRecordStore`;
  `build_final_runtime_modules` / `run_final_wiring_turn` gained
  `dual_track_gate_learner` / `social_record_store` (default None = prior
  behavior).
- W1.D (CP-11 gate completeness): `PredictionErrorModule` gained
  `export_predictive_head_checkpoint` / `restore_predictive_head_checkpoint`
  (typed `PredictiveHeadCheckpoint`, schema `predictive-head-checkpoint.v1`,
  session-medium; restore fails loudly via `PredictiveHeadCheckpointError`)
  and `predictive_head_kill_criteria` (typed `PredictiveHeadKillCriteria`,
  self-reward autocorrelation check over a 64-sample window; report-only).
  `AgentSessionRunner` exposes `prediction_module` / `semantic_state_store`
  readonly properties; `volvence_zero.social` re-exports `TOM_SLOTS`. New
  `scripts/run_learned_shadow_soak.py` (default 500 synthetic turns)
  accumulates all learned-owner readouts + honest `learned_active_gate`
  verdicts into one artifact (`learned-shadow-soak.v1`).

Verified by `tests/test_dual_track_gate_learner.py`,
`packages/vz-runtime/tests/test_dual_track_gate_learner_session.py`,
`tests/contracts/test_owner_prediction_signal.py` (v2 forecaster tests),
`tests/test_social_tom_settlement.py`,
`tests/contracts/test_predictive_heads_shadow.py` (checkpoint + kill-criteria).

## CP-12 / CP-11 (2026-07-12): owner prediction signals + PE shadow heads

Shared contract slice (new vz-contracts module `volvence_zero.owner_prediction`):
`OwnerPredictionKind` (closed enum, 9 kinds; 5 first-wave wired) /
`OwnerPredictionSignal` / `OwnerPredictionSettlement` / `settle_owner_prediction`.
Append-only snapshot fields, all defaulted (byte-compatible for existing
constructors):

- `CommitmentSnapshot` / `RelationshipStateSnapshot` / `GoalValueSnapshot` /
  `BoundaryConsentSnapshot` / `ExecutionResultSnapshot` +=
  `owner_prediction_signals: tuple[OwnerPredictionSignal, ...] = ()`.
- `PredictionErrorSnapshot` += `owner_prediction_settlements` (only the PE
  owner constructs settlements) and `predictive_head_readout`
  (`PredictiveHeadReadout`, CP-11 SHADOW dual-run MAE vs baseline,
  report-only).
- `PredictionErrorModule.dependencies` += `relationship_state`, `goal_value`,
  `boundary_consent`, `execution_result` (read via `upstream.get`, commitment
  overlay precedent; no cycle — semantic owners depend only on
  substrate/memory).
- `SemanticStateStore` gained owner-local `pending_owner_prediction` /
  `record_owner_prediction` / `next_owner_prediction_sequence` (not part of
  hydration payload v1; predictions settle next in-session turn).

Verified by `tests/contracts/test_owner_prediction_signal.py` and
`tests/contracts/test_predictive_heads_shadow.py`; import boundary table
gained `owner_prediction` for vz-cognition.

## autograd-owner-integration deploy form (2026-06-29): runtime-configurable

The four torch autograd backends are now reachable through the canonical runtime
config and thread to the owners, so they are production-deployable rather than
class-level-only. New `FinalRolloutConfig` fields (all default `DISABLED` =
unchanged behavior; rollback = reset to `DISABLED`):

- `temporal_ssl_backend` -> `ETANLJointLoop` -> `MetacontrollerSSLTrainer(ssl_backend=...)`
- `temporal_runtime_backend` -> `AgentSessionRunner` / `build_final_runtime_modules`
  -> `FullLearnedTemporalPolicy.set_runtime_backend(...)` (world + self tracks)
- `internal_rl_backend` -> `ETANLJointLoop` -> `InternalRLSandbox(rl_backend=...)`
  -> `CausalZPolicy`
- `cms_torch_backend` -> `build_default_memory_store(cms_torch_backend=...)`
  -> `CMSMemoryCore(cms_backend=...)`

`ETANLJointLoop.__init__` gained `ssl_backend` / `internal_rl_backend` kwargs
(default DISABLED). No public snapshot schema changed; defaults keep the pure
path as the live writer. Verified by `tests/test_autograd_backend_deploy_wiring.py`
(config threads to every owner; defaults DISABLED; rollback trivial) with the
default-path regression (`test_final_wiring`, real-runtime suites) unchanged.

## autograd-owner-integration (2026-06-29): torch paths wired into owners

Bridges the torch/autograd modules from sidecar proofs into the real owner
paths, all gated by `WiringLevel` (DISABLED default / SHADOW / ACTIVE) with the
pure-Python path retained as the rollback baseline. No public snapshot schema
changed; append-only evidence fields only:

- `SSLTrainingReport` += `torch_backend`, `torch_prediction_loss`, `torch_kl_loss`,
  `torch_switch_sparsity`, `torch_parameters_changed`, `torch_grad_norm`,
  `torch_wrote_back` (all defaulted; DISABLED leaves them at defaults).
- `OptimizationReport` (internal_rl) += `torch_backend`, `torch_parameters_changed`,
  `torch_policy_loss`, `torch_value_loss`, `torch_approx_kl`, `torch_wrote_back`.
- `RareHeavyArtifact` += optional `lss_checkpoint` (float-only
  `LSSRareHeavyCheckpoint`); `export_rare_heavy_artifact(lss_checkpoint=...)`.
- New owner-internal (non-snapshot) calibration on `PredictionErrorModule`:
  `export/import_rare_heavy_lss`, `rare_heavy_lss_calibration`,
  `export/restore_rare_heavy_lss_state`.

Constructor knobs (default DISABLED, reversible): `MetacontrollerSSLTrainer(
ssl_backend=...)`, `FullLearnedTemporalPolicy(runtime_backend=...)` +
`set_runtime_backend`, `InternalRLSandbox(rl_backend=...)` /
`CausalZPolicy(rl_backend=...)` + `set_rl_backend`, `CMSMemoryCore(cms_backend=...)`
/ `build_default_memory_store(cms_torch_backend=...)`.

`prediction.lss_rare_heavy` was added to the prediction facade (torch-free).
`prediction` is already an allowed upstream tier for vz-temporal, so the
rare-heavy pipeline carries `lss_checkpoint` via `object` typing without a new
module-level import. Import-boundary table unchanged from the prior
full-autograd migration entry.

Bugfix recorded here for traceability: the ndim switch `gate_input = delta +
z_tilde` is tuple CONCATENATION (2*n_z dims, matching the n_z*2-column gate W1),
not an elementwise add. The Phase B/SSL torch forwards were corrected to
concatenate + use the full W1, restoring pure<->torch parity.

## NL/ETA full-autograd migration (2026-06-29): no public schema drift

Phases 0–5 of the NL/ETA full-autograd uplift add real torch autograd paths
(metacontroller SSL with Eq.3 `N(0,I)` KL + STE switch, PPO on `z_t`, runtime
metacontroller + CMS band, delta-momentum, gradient LSS) **without changing any
public snapshot schema**. Mechanism:

- The torch numeric backend (`volvence_zero.tensor_backend`, in `vz-contracts`)
  keeps torch tensors owner-internal; everything published into a snapshot is
  converted back to float tuples at the boundary (`to_floats`).
- New artifacts (`TorchMetacontrollerArtifact`, `CMSBandWeights`,
  `LSSArtifact`, …) carry floats/ints only and travel via the existing
  rare-heavy artifact path; they are not registered as runtime slots.
- The pure-Python path remains the rollback baseline; torch advances per-owner
  through `WiringLevel` `DISABLED -> SHADOW -> ACTIVE` with parity + latency
  gates. `temporal_abstraction`, `memory`, and `prediction_error` slot schemas
  are unchanged.

Import-boundary note: `volvence_zero.tensor_backend` /
`tensor_backend_parity` were added to the allowed upstream set for
`vz-memory`, `vz-temporal`, `vz-cognition`, and `vz-runtime` in
`tests/contracts/test_import_boundaries.py` (they live in the zero-upstream
`vz-contracts`, so no pyproject dependency direction changed). `torch` is an
optional `[torch]` extra on `vz-contracts` / `vz-memory` / `vz-temporal`.

## Slice C.2 (2026-05-03): Semantic spine readiness evidence chain

Builds the first narrow cognitive-loop evidence chain over the five
core semantic owners:

- `relationship_state`
- `goal_value`
- `boundary_consent`
- `commitment`
- `execution_result`

Landed shape:

- `EvaluationBackbone` publishes `semantic_spine_coverage` and
  `cognitive_loop_readiness` from public semantic owner snapshots only.
  Evaluation does not reconstruct owner internals and remains a readout
  / gate layer, not a learning source.
- `FinalAcceptanceReport` requires these readouts when the five core
  semantic owners and `evaluation` are ACTIVE.
- Session and cross-session evaluation now track
  `semantic_spine_readiness`, derived only from
  `cognitive_loop_readiness`; `semantic_spine_coverage` remains a
  completeness check and is not mixed into trend calculations.
- `EvolutionJudgement` rolls back on clear `semantic_spine_readiness`
  regression, preventing capability widening from masking degradation
  in the semantic-state foundation.
- Dialogue benchmark case reports, open dialogue reports, emergence
  dashboard payloads, and dialogue paper-suite metric values surface:
  - `mean_semantic_spine_coverage`
  - `mean_cognitive_loop_readiness`
- NL essence assessment adds `semantic-spine-ready` as an audit gate.
  It is intentionally not part of the default required gate list yet.
- `claim_companion_stateful_relationship` consumes
  `semantic-spine-ready` plus dashboard / repeated-run paper-suite
  summaries. `retain` still requires the cross-session gate; semantic
  spine alone can at most support the current lightweight foundation
  evidence.
- Dialogue paper-suite manifest includes
  `canonical_mean_semantic_spine_coverage` and
  `canonical_mean_cognitive_loop_readiness` as secondary metrics, so
  companion-stateful verdicts prefer repeated-run summaries over a
  single reference dashboard.

Rollback:

- Disable individual semantic owners via `FinalRolloutConfig.kill_switches`;
  downstream modules receive placeholders and must not read owner
  internals.
- Remove `semantic-spine-ready` from any stricter future
  `required_gate_ids` config before rolling back code.
- Revert the readout additions without changing snapshot schemas: the
  semantic owner snapshots themselves are unchanged by this slice.
- Retrieval policy has an explicit temporal-disabled fallback so
  `temporal` kill-switch rollbacks do not force consumers to reconstruct
  temporal state.
- Follow-up hardening: `clone_semantic_store` now preserves lifecycle,
  follow-up policy, and typed outcome maps so cloned semantic stores keep
  owner-side continuity evidence instead of copying only raw records.
- Owner-depth follow-up: commitment, open_loop, boundary_consent,
  goal_value, and relationship_state now publish additional owner-side
  lifecycle / continuity readouts. `LLMSemanticProposalRuntime` remains
  typed-proposal-only and now supports schema-bound proposals for
  `boundary_consent` and `goal_value` while non-target slots still
  delegate to the base runtime.
- Proposal-quality follow-up: `volvence_zero.semantic_state.quality`
  adds a proposal-level harness that evaluates precision, recall,
  false positives, missing operations, and fallback count before owner
  store mutation. Initial tests cover `boundary_consent` and
  `goal_value` scripted LLM cases, including explicit false-positive
  and fallback-count checks. The harness also publishes shadow-only
  `would_block` / `would_allow` counts and gate reasons
  (`false-positive`, `missing-expected-operation`,
  `confidence-below-floor`, `runtime-fallback`) without blocking
  runtime or owner-store writes.
- Environment-outcome follow-up: reused `PredictionErrorModule`
  instances now receive the current turn's `PredictionActionContext`
  without resetting previous prediction state. Tool outcomes recorded
  through `BrainSession.submit_tool_result(...)` are carried as
  next-turn `PredictionActionContext.environment_outcome_id` lineage.
- Evidence export follow-up: dialogue paper-suite export can write a
  non-gating `semantic_proposal_quality_shadow.json` sidecar and also
  include that payload in `EvidenceBundle.reference_artifacts`.
- Action-credit follow-up: `CreditModule` now declares
  `temporal_abstraction` as an upstream dependency and appends
  `derive_segment_closure_credit_records(...)` to the PE-first credit
  path when `PredictionActionContext.segment_id` matches a closed
  temporal segment. PE-derived credit contexts now carry
  segment/action/environment event/outcome lineage without changing the
  numeric credit formula.
- Snapshot replay follow-up: `AgentSessionRunner.export_snapshot_replay_artifact()`
  now includes an `action_replay` section derived from existing
  `prediction_error`, `temporal_abstraction`, and `credit` snapshots.
  `dialogue_trace` remains a parallel debug artifact, not a runtime
  schema dependency.

Focused validation used for this slice:

- `python -m pytest tests/test_evaluation_backbone.py tests/test_semantic_state_owners.py tests/test_final_wiring.py`
- `python -m pytest tests/test_dialogue_benchmark.py::test_nl_essence_assessment_surfaces_semantic_spine_ready_gate tests/test_dialogue_benchmark.py::test_build_dialogue_emergence_dashboard_compresses_strong_proof_and_open_env_evidence tests/test_dialogue_benchmark.py::test_build_dialogue_emergence_dashboard_payload_exposes_summary_keys tests/test_evaluation_backbone.py tests/test_semantic_state_owners.py tests/test_final_wiring.py`
- `python -m pytest tests/test_dialogue_benchmark.py::test_build_dialogue_paper_suite_manifest_and_config_freeze_expected_scope tests/test_dialogue_benchmark.py::test_run_dialogue_paper_suite_repeated_benchmark_emits_interval_summaries tests/test_dialogue_benchmark.py::test_nl_essence_assessment_surfaces_semantic_spine_ready_gate tests/test_evaluation_backbone.py tests/test_semantic_state_owners.py tests/test_final_wiring.py`
- `python -m pytest tests/test_semantic_proposal_quality.py tests/test_llm_semantic_runtime.py`
- `python -m pytest tests/test_final_wiring.py::test_reused_prediction_module_receives_current_action_context tests/test_tool_outcome_evidence.py::test_brain_submit_tool_result_links_next_turn_prediction_context tests/test_dialogue_benchmark.py::test_dialogue_paper_suite_exports_proposal_quality_shadow_artifact`
- `python -m pytest tests/test_credit_gate.py tests/test_eta_nl_clean_action_abstraction.py tests/test_tool_outcome_evidence.py`

Long dialogue replay note: full `tests/test_dialogue_benchmark.py`
enters systematic replay paths and may exceed a short interactive run.
The semantic-spine evidence path was validated through focused shards
covering evaluation, final wiring, dialogue case reports, emergence
dashboard, NL essence, and paper-suite repeated summaries.

## Slice C.1 (2026-05-03): 情绪决策支持 owner-side readout

Extends the existing semantic owner snapshot surface so emotional
decision support is produced by owners before it is consumed by
ETA / response assembly:

- `UserModelSnapshot` adds `preferred_support_pacing`,
  `decision_style`, `overwhelm_pattern_strength`; `durable_goals`
  now receives typed profile goal proposals instead of staying empty.
- `RelationshipStateSnapshot` adds `emotional_load`, `repair_need`,
  `trust_delta`, `attunement_gap`, `stabilization_need`.
- `GoalValueSnapshot` adds `value_conflict`, `decision_readiness`,
  `active_tradeoff_count`, `reversibility_need`,
  `goal_shift_pressure`.
- `BoundaryConsentSnapshot` adds `autonomy_risk`, `consent_clarity`,
  `professional_scope_pressure`, `overreach_risk`.
- `SemanticRecord` now retains proposal `control_signal`, allowing
  owner-side confidence/control aggregation without downstream text
  reconstruction.
- `ResponseAssemblySnapshot.support_before_decision_pressure` now
  prioritizes these owner readouts; domain/prototype evidence remains
  auxiliary. `ResponseAssemblyReadout` in `vz-contracts` includes the
  pressure and `eta_action_family` fields consumed by evaluation.

Compatibility: all new snapshot fields have defaults, preserving
synthetic fixtures and older tests that construct snapshots directly.

## Slice C (2026-05-03): 解 vz-cognition ↔ vz-application 真循环依赖

Closes the architectural debt where `vz-cognition.evaluation.backbone`
imported 8 application-tier dataclass types via
`volvence_zero.application_types`, a cycle-break shim that physically
hosted application schema inside the kernel wheel and forced
`vz-cognition` to permanently own product-tier knowledge.

Slice C replaces the shim with a structural `Protocol` surface in
`vz-contracts`:

- New module `volvence_zero.application_readouts` (vz-contracts) holds
  14 minimal `Protocol` types declaring only the attributes the
  evaluation layer reads: `BoundaryReadout`, `BoundaryDecisionReadout`,
  `CaseMemoryReadout`, `CaseEpisodeHitReadout`,
  `CaseOutcomeSummaryReadout`, `DomainKnowledgeReadout`,
  `StrategyPlaybookReadout`, `PlaybookRuleReadout`,
  `ResponseAssemblyReadout`, `ExperienceFastPriorReadout`,
  `ExperienceFastPriorRegimeBiasReadout`,
  `ExperienceFastPriorFamilyBiasReadout`,
  `ApplicationOutcomeAttributionReadout`,
  `ApplicationSequencePayoffReadout`.
- `vz-cognition.evaluation.backbone` now imports those Protocols and
  uses them as the parameter type annotations on
  `record_learning_evidence`, `record_application_delayed_evidence`,
  and `_learning_evidence_scores`. No method body changed: structural
  Protocol matching means existing concrete dataclass instances
  satisfy the Protocols by attribute presence.
- The dataclass definitions previously hosted in
  `vz-cognition/src/volvence_zero/application_types.py` have been
  moved back to their natural home in
  `vz-application/src/volvence_zero/application/runtime.py`. The shim
  module is deleted.
- `tests/contracts/test_import_boundaries.py` `ALLOWED_VZ_UPSTREAM`:
  - vz-cognition gains `application_readouts`; comment rewritten to
    reflect the Protocol surface design.
  - vz-application / vz-temporal / vz-runtime drop `application_types`
    (the shim no longer exists).

External imports of the form
`from volvence_zero.application.runtime import BoundaryPolicySnapshot`
(used by lifeform-expression, vz-runtime agent code, vz-temporal
joint loop, and several test files) keep working unchanged because
those dataclasses are now defined directly in `application.runtime`
instead of being re-exported from a shim.

Tests: 518 contracts / social / credit / memory / final-wiring /
application-storage / prediction-error / dialogue-outcome tests pass
with 0 regression (1 deselected pre-existing kill-switch failure
unrelated to this change).

## Slice D (2026-05-02): vz-cognition social_*.py 收成 social/ 子包

Pure refactor; no behavior change. Replaces 7 flat top-level files in
`vz-cognition/src/volvence_zero/` with one capability-domain subpackage:

- `social_identity.py` → `social/identity.py`
- `social_role.py` → `social/role.py`
- `social_group.py` → `social/group.py`
- `social_tom.py` + `social_tom_runtime.py` → `social/tom.py`
- `social_common_ground.py` + `social_common_ground_runtime.py` →
  `social/common_ground.py`

The `_runtime.py` suffix is dropped: each LLM proposal runtime is a
collaborator of its owner module and lives in the same file. The new
`volvence_zero.social.__init__` re-exports every previously top-level
public class so external consumers use a single stable import path:
`from volvence_zero.social import CommonGroundModule, ...`.

Cross-wheel changes:

- `tests/contracts/test_import_boundaries.py` `ALLOWED_VZ_UPSTREAM`
  collapses 7 legacy `social_*` tokens into a single `social` token
  for vz-application / vz-temporal / vz-runtime tiers.
- `vz-runtime/.../integration/final_wiring.py` consolidates the 5
  per-domain `from volvence_zero.social_X import` statements into one
  alphabetised `from volvence_zero.social import (...)` block.
- `lifeform-evolution/.../social_cognition_evidence.py` rewritten to
  the new path.
- All 24 affected import lines across 8 files were rewritten by a
  one-shot migration script; residual reference scan returned 0.

Tests: 505 social / memory / contracts / final-wiring tests pass with
0 regression (1 deselected pre-existing failure unrelated to social).

## Slice 12 (2026-05-02): MemoryModule SSOT for social PE signals

Closes the SSOT violation where `SocialPredictionAggregateModule` and
`SocialPredictionErrorModule` reconstructed `MEMORY_VISIBILITY`
predictions / PE from raw `MemorySnapshot.suppressed_cross_scope_entries`
and stamped `owner="MemoryModule"` on records they wrote themselves
(R8 / `ssot-module-boundaries.mdc` violation).

Landed shape:

- New typed contract `MemorySocialPESignal` in
  `volvence_zero.social_cognition` (vz-contracts), plus pure helpers
  `build_memory_visibility_signals`,
  `social_prediction_from_memory_signal`, and
  `social_prediction_error_from_memory_signal`.
- `MemorySnapshot` extended with `social_pe_signals: tuple[MemorySocialPESignal, ...]`;
  `MemoryModule` is the only writer.
- `SocialPredictionAggregateModule` and `SocialPredictionErrorModule`
  are now lifter / pass-through owners; they read
  `MemorySnapshot.social_pe_signals` and forward through the
  contract helpers, never reconstruct from raw memory fields, and
  never borrow another owner's name on their own snapshots.
- `prediction_id` and `signal_id` keep the previous public format
  (`memory_visibility:{scope}:v{seq}` /
  `memory_visibility_pe:{scope}:v{seq}`); `seq` is the publishing
  module's `_version + 1`.
- `MemorySnapshot` doc + owner rules updated in
  `docs/DATA_CONTRACT.md`; `social_prediction` /
  `social_prediction_error` rows reflect the lifter contract.

Tests: `tests/test_social_memory_visibility_loop.py` (5),
`tests/test_final_wiring.py` social-prediction empty-scaffold case,
and the social cognition / credit / contracts subset (481 tests)
all pass with no regression.

This file holds rollout notes, planned slot waves, and landed slice summaries that
should not inflate the stable contract surface in `docs/DATA_CONTRACT.md`.

## Social Cognition Planned Slots

Social Cognition Learning Layer slots follow the protocol in
`docs/implementation/15_social_cognition_layer.md`:

1. `DISABLED`: types and docs exist; no runtime publication.
2. `SHADOW`: new slots publish alongside existing flat slots; consumers keep old
   slots unless explicitly opted in.
3. `ACTIVE`: selected consumers switch to keyed / social slots; old flat slots
   become compatibility read models.
4. Retire flat path after evidence gates pass and rollback window expires.

Planned / staged slots:

- `multi_party_identity`
- `interlocutor_models`
- `relationship_states`
- `interlocutor_states`
- `belief_about_other`
- `intent_about_other`
- `feeling_about_other`
- `preference_about_other`
- `conversational_role`
- `common_ground`
- `groups`
- `social_prediction`
- `social_prediction_error`

Every row must identify an owner, timescale, social prediction, and PE consumer
before implementation. LLM output can only produce typed proposals; no LLM
classifier owns social state.

## Owner Field Extensions

Landed and planned field extensions that do not create new kernel slots:

- `commitment`: AAC lifecycle fields (`advocacy_state`, `alignment_state`,
  `followup_policy`, `last_outcome`, evidence and turn anchor). Landed
  2026-04-29; canonical spec is `docs/specs/aac-commitment-lifecycle.md`.
- `case_memory`: provisional lifecycle fields from `docs/specs/thinking-loop.md`.
  Landed 2026-04-29.
- `regime`: participation and cognitive-depth hints from PRD Gap 8 scaffold.
  Landed 2026-04-29; learned metacontroller readout remains a later slice.
- `user_model`: `interlocutor_readout` and confidence / extraction metadata.
  Planned.
- `plan_intent`: lifecycle outcome entries and aggregate counts. Landed
  2026-04-29.
- `execution_result`: lifecycle outcome entries and aggregate counts. Landed
  2026-04-29.

## Shared Contract Types

Shared immutable contract types added to `vz-contracts`:

- `volvence_zero.thinking`: thinking task / artifact contracts. Landed
  2026-04-29.
- `volvence_zero.affordance`: affordance descriptor schema and selection-hint
  invariant. Landed 2026-04-29.
- `volvence_zero.social_cognition`: multi-party identity, ToM, conversational
  role, common-ground, group, social prediction, and social prediction error
  contracts. Landed through 2026-05-02 SHADOW / evidence slices.
- `volvence_zero.environment`: `EnvironmentEvent` / `EnvironmentOutcome`
  contracts for lifeform-host interaction.
- `volvence_zero.temporal_types`: public temporal snapshot types
  (`ControllerState`, `TemporalSegmentClosure`, `TemporalAbstractionSnapshot`).
  Landed 2026-05-02 to prevent consumers from importing `vz-temporal` owner code
  just to validate snapshot shape.

## Lifeform-Side Contract Notes

Lifeform-side slots do not enter kernel propagation and must not be imported by
`vz-*` wheels:

- `vitals`: owned by `lifeform-core`.
- `affordance`: schema in `vz-contracts`, registry / invoker in
  `lifeform-affordance`.
- `thinking_loop`: async scheduler in `lifeform-thinking`.

Side effects enter the kernel only through public `BrainSession.submit_*` /
`LifeformSession.run_turn` paths.

## Slice Relationship P2–P4 (2026-08-22): forecast settlement、action advisory 与 qualified outcome

本 slice enrich 既有 owner value，不新增 kernel slot：

- `PreferenceAboutOtherSnapshot` 增加默认空的 action history / forecast / settlement；
  `SocialRecordStore` persistence v1 可读、v2 写出；
- `DialogueExternalOutcomeEvidence` 增加 all-or-none relationship exact join，并增加独立
  `QUALIFIED_USER_REPORT` source；该 source 必须携带 typing qualification/runtime/schema；
- `TemporalAbstractionSnapshot` 增加可选 `TemporalActionAdvisoryProposal` 与 status；默认
  relationship advisory 为 SHADOW，P3/P4 artifact 未授权 ACTIVE；
- P4 lifeform-side action audit、outcome receipt、typing qualification 与 training candidate
  不进入 propagate slot 注册表；operational/training roots 分离。

回滚：移除 P4 qualification path 回到 collection-only；把
`relationship_action_advisory` 降为 DISABLED；v1 owner persistence 继续兼容。退出 SHADOW
必须另有真实 action exposure 与 promotion evidence，当前未满足。权威 spec：
`docs/specs/relationship-intelligence-closed-alpha.md`。

## Slice Relationship P1m（2026-08-22）：生成式仪器、fresh reader 资格与止损终局

本 slice 不新增 kernel slot、runtime writer 或 product wiring；它冻结并只读评估 Relationship Lab
instrument：

- `lifeform-domain-emogpt` 以内容寻址 recipe、surface seed inventory 与 renderer transport
  生成 24 组镜像对；v1–v4 transport/preflight 失败保留，v5 deterministic typed surface
  realizer 通过逐字段 expected-output hash 后生成 48 个 scene；
- `lifeform-evolution.relationship_lab_packet1m_qualification` 在 output=0 时冻结 96 条 Qwen
  prompt/RAG 读出与 48 条 owner/reader 读出、模型/reader lineage、A/B 轮换和 Wilson 单侧门；
- 最终 prompt/RAG 都是 24/48 correct、0/24 pair flip；structured named reader 为 46/48、
  24/24 flip。report `9580ddff…fc56` 判 `prompt_steelman_baseline_too_weak`，场景版本化关闭；
- terminal loader 复算 report artifact id、derived metrics、protocol/plan lineage、四本账 SHA/行数
  与 batch execution manifest；evaluation 不回写 PE/credit/gate。

回滚：停止消费 P1m artifacts；产品和 P2–P4 wiring 不变。旧 artifact 不删除、不改判词。fresh
structured 结果只允许作为后续独立 runtime 因果包的候选组件证据，不得写成 P2 formal、
Volvence advantage、Readable 或四能力已通过。权威 spec：`docs/specs/relationship-lab.md`。

## Slice Relationship P4.3（2026-08-22）：named readout 到动作与 outcome 的隔离传导

本 slice 不新增 kernel slot、runtime writer 或产品 ACTIVE；它把 P1m 候选 reader 接入已存在的
P4.1 Lab owner loop，并只发布 development report：

- `run_relationship_p4_subject_mechanism(...)` 公开 arm-neutral runner；原 P4.1 wrapper 继续复用
  同一实现，避免第二条 owner/settlement 链；
- legacy 与 P1m named 两臂都固定 `RelationshipActionGateMode.ALWAYS`，都派生 PE/credit 但不向
  gate 回写，因此唯一改变的 collaborator 是 frozen condition reader；
- seen-v3 两条 subject 上 legacy/named 分别得到 6/16 与 16/16 preferred-action match、9/16 与
  16/16 positive outcome；matched action/outcome change 为 10/16 与 9/16；
- report `e7bbc914…70bd` 绑定 P1m report/protocol/reader、P4 protocol/public-plan hash；create-only
  JSON/Markdown 已用同一 frozen BGE 重跑并验证派生指标、artifact id 与字节内容；
- `component_selected_after_p1m_observation=true / seen_fixture_only=true /
  formal_evidence_authorized=false` 是不可关闭的 claim firewall。

回滚：停止运行/消费 `relationship_lab_p4_named_reader` 即可；P2d reader、P3 gate、P4 product 与
P4.1 protocol 均未改 wiring。该包只证明 development readout transmission，不能修复 P1m、授权
P2 formal、证明 Volvence advantage、完整 Readable 或四能力。下一包固定 reader，只隔离
PE-derived credit 是否更新 learned gate。权威 spec：
`docs/specs/relationship-intelligence-closed-alpha.md`。

## Slice Relationship P4.2（2026-08-22）：preference-action 纠删 receipt/tombstone

本 slice 继续 enrich `preference_about_other`，不新增 slot 或 writer：

- `PreferenceAboutOtherSnapshot` 增加默认空的
  `action_outcome_mutation_receipts`；旧构造保持兼容；
- 新 frozen `PreferenceActionOutcomeMutation` 以 expected evidence SHA-256 做 optimistic
  concurrency；新 `PreferenceActionOutcomeMutationReceipt` 只保存 hash、opaque refs 与失效
  forecast ids；
- `SocialRecordStore` export 升为 v3，继续 hydrate v1/v2；旧 snapshot 的 receipt 集合为空；
- `REDACT` receipt 是持久 tombstone。纠正/删除由 `PreferenceAboutOtherModule` 原子更新
  record、outcome、pending ToM prediction 与 pending forecast，旧 evidence id 不得跨恢复复活；
- mutation 是 user-directed state correction/privacy command，不进入 PE、credit、evaluation、
  ModificationGate 或 product ACTIVE。

回滚：可停止 mutation consumer/drill；不得删除已落盘 redaction tombstone，也不得把 v3 手工
降成 v2 后重新导入旧证据。P4.2 engineering drill 的 PASS 仅代表跨恢复机械闭合，不代表
formal `recovery_after_correction` 效果通过。权威 spec：
`docs/specs/social_cognition/02_theory_of_mind.md` 与
`docs/specs/relationship-intelligence-closed-alpha.md`。

## Slice Relationship P2d（2026-08-22）：命名条件 readout 与 persistence v4

本 slice 继续 enrich `preference_about_other`，不新增 slot 或 writer：

- 新 frozen `RelationshipConditionReadout` 保存命名 condition、置信度、归一化 margin、全部候选
  分数、reader artifact id 与 current-observation SHA-256；
- `RelationshipConditionReaderArtifact` 内容寻址地绑定 embedding model id、weights SHA-256、
  cosine、prototype 文本与 temperature；embedding backend 由 composition root 注入；
- 非 owning collaborator 只提 proposal；`PreferenceAboutOtherModule` 校验 source hash 与当前
  request 后才随正式 `PreferenceActionForecast` 发布，consumer 不解析 evidence 重建状态；
- `SocialRecordStore` export 升为 v4，继续 hydrate v1/v2/v3；旧 forecast 的 condition readout
  显式为空，P4.2 tombstone/receipt 原样保留；
- reader 不接 evaluator、expected action、未来 outcome、PE、credit、reward 或 judge。

回滚：移除 P2d collaborator 即恢复 condition-readout 为空的 SHADOW forecast；v4 payload 继续
由 owner 读取，禁止降版覆盖或丢弃 redaction tombstone。已见 v3 的 `4/12 → 12/12` 与 6/6
mirror-pair 结果只定位旧字符哈希 backend，不是 P2 formal、Readable 或四能力证据；P1m fresh
qualification 仍是退出条件。权威 spec：`docs/specs/social_cognition/02_theory_of_mind.md` 与
`docs/specs/relationship-intelligence-closed-alpha.md`。
