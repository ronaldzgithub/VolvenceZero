# Contract Migration Log

> Status: migration / implementation log
> Last updated: 2026-08-22

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
P2 formal、证明 Volvence advantage、完整 Readable 或四能力。原定后继 P4.4 已固定 reader，只
隔离 PE-derived credit 是否应用于 learned gate；结果记录如下。权威 spec：
`docs/specs/relationship-intelligence-closed-alpha.md`。

## Slice Relationship P4.4（2026-08-22）：exact PE-credit 到 learned gate 的隔离传导

本 slice 不新增 kernel slot、runtime writer、prompt、场景或产品 ACTIVE；它复用 P4.1/P4.3 的
既有 owner loop 与 frozen named reader，并只发布 development report：

- fixture 固定为两个已见、post-selected 的合成 subject，每个 8 decisions，合计 16 decisions；
  owner/hydration、Lab ACTIVE、reactive environment、动作 surface、cold learned gate 初始化与
  logical store reconstruction 节奏在 matched arms 间一致；该历史计数字段不代表 OS child process；
- 唯一 toggle 是 exact owner settlement→`SocialPredictionError`→dedicated credit 是否应用于
  learned gate；evaluation、judge、human anchor 与 generator truth 均不进入 update；
- no-credit 臂为 0/16 steer、0/16 preferred-action match、0/16 credit apply 与 parameter change，
  positive outcome 7/16、reversal match 0/14；PE-credit 臂为 7/16 steer/action match、16/16
  credit apply 与 parameter change，positive outcome 13/16、reversal match 7/14；
- matched arms 产生 14/16 probability change、7/16 action change、6/16 outcome change；16 次
  exact credit apply 全部对应 checkpoint parameter change；
- report `5c955fb1…810c` 判
  `pe_credit_learning_transmission_observed_development_only`，并固定
  `component_selected_after_p1m_observation=true / seen_fixture_only=true /
  formal_evidence_authorized=false`。

回滚：停止运行/消费 `relationship_lab_p4_pe_learning`；create-only report 与 checkpoint lineage
按原 hash 保留，P2d reader、P3/P4 product gate、expression 与 production SHADOW wiring 均不变。
该包只能说明 seen fixture 上 PE-credit 应用改变 learned checkpoint 与后续 typed action/outcome，
不能证明 independent subject / 32K context、真实 residual actuation、用户可见 steer、formal
Learnable、production ACTIVE 或四能力。下一单一因果包应固定 P4.4 gate 隔离真实 residual
actuation；Windows/CUDA 迁移前的 cross-process Appendable preflight 已由 P4.5 完成且只形成平台
development evidence，不能反向升级本包判词。
权威 spec：`docs/specs/relationship-intelligence-closed-alpha.md`。

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

## Slice Relationship P4.5（2026-08-22）：Windows real-child owner hydration preflight

本 slice 不新增 kernel slot、runtime owner 或产品 wiring；唯一数据 owner 仍是
`social_record_store`，preflight 只发布离线 create-only evidence：

- `append_relationship_p4_onboarding_session(...)` 把 P4 onboarding 的正式 owner 写入路径集中为
  一个公共 Lab helper；既有 P4.1 runner 与 P4.5 worker 复用同一实现，不建立第二条 writer；
- `relationship_lab_p4_cross_process_appendable` 冻结 correct / empty / same-stage swapped 三种 prior
  state intervention。每个 4+8 pulse 都由新的 `sys.executable` child 执行；parent request exact-key
  契约禁止携带 history、records、owner snapshot/payload、forecast score 或 evaluator truth；
- child 只通过 `FileSystemPersistenceBackend`→`OwnerHydrationStore(ACTIVE)` hydrate
  `SocialRecordStore`，先 load 继承 backend version，再 append/probe、export/save；request、receipt、
  raw checkpoint、owner payload 与 immutable boundary 都有独立 SHA lineage；
- Windows artifact 共 72 次真实 child invocation；correct/swapped 每 subject 的版本为 `1..12`，
  empty 每拍保持 fresh v1。correct/empty forecast presence 改变 16/16，correct/swapped 推荐动作
  改变 14/16；report `675815b9…2052` 判
  `cross_process_owner_hydration_forecast_effect_observed_development_only`；
- firewall 固定 `seen_fixture_only=true / independent_subject_count=0 /
  formal_evidence_authorized=false`，evaluator/environment、PE/credit/learning、gate、model/Qwen、
  residual、expression 与 production ACTIVE 均未进入本包。当前 filesystem backend 未冻结
  mid-write crash/power-loss 原子性，因此不得声称 crash recovery。

回滚：停止运行/消费 `run_relationship_lab_p4_cross_process_appendable.py` 即可；产品、P2d reader、
P3/P4 gate、SHADOW wiring 与 owner schema 均不迁移。已发布的 create-only state/request/receipt/report
按原 hash 保留，不得删除后用新 PID/nonce 覆盖。下一包固定 P4.4 gate 与本包 hydration，单独隔离
Windows/CUDA residual actuation。权威 spec：`docs/specs/relationship-lab.md` 与
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

## Slice P4.6 preflight（2026-08-22）：Windows/CUDA strict 32767+1 诊断契约

本 slice 只冻结尚未执行的 substrate 工程诊断，不新增 live slot、策略 owner 或产品 wiring：

- `windows-cuda-strict-32k-smoke.v1` 固定 Qwen2.5-1.5B revision/weights/assets、Windows/CUDA
  exclusive cuDNN SDPA cached profile、layer 20 / width 1536、唯一一次 32767 input + 1 generated
  token 调用及十个 `utf8_lf_canonical_v1` 源码指纹；runner 后续固定三项 namespace root 与
  implementation import origin；后续审计确认十项 pin 不是完整 local import closure，协议因此更新为
  `4934a344…e2c1a`、raw SHA `ec7b7bce…5eb4`，并固定
  `transitive_local_source_closure_pinned=false`，闭包关闭前不得称 exact-source physical evidence；
- token-level residual 与 feature-surface 解释新增 additive substrate owner
  `audit_strict_capture(...)`，只发布 frozen `StrictCaptureAuditSummary`；runtime evidence
  orchestrator 禁止遍历原 capture。framed hash 绑定实际 step/layer/cardinality/width/value，
  顶层 latest residual 与 sequence 末步不一致即失败。该交换在 live DAG 为 `DISABLED`，仅供离线诊断；
- runner 在构造模型前写入并 fsync 绑定 outer lease/protocol/source 的 launch receipt；完整根恰有
  launch/attestation/report/manifest/completion 五件，运行时异常留下且不删除不可重用的残根；
- 本地 no-retry 只覆盖一个冻结输出根。outer host-campaign scaffold 已在下一 slice 落地但
  production 明确禁用；standalone 根固定 `external_append_only_anchor_present=false`，不得证明
  物理执行；
- `validate-existing` 不加载 substrate/torch/CUDA，重算所有 receipt/hash/lineage；完整但失败的诊断
  返回进程码 2，不能被 CI/campaign 当成 PASS。

静态验证仅运行 Ruff、协议 JSON 解析、十个源码指纹复算与 `git diff --check`；因当前 Windows
宿主仍有 WHEA19 internal parity / CPU access-violation 阻断，未运行 Python、pytest、CUDA 或模型，
也没有生成 PASS artifact。回滚为停止调用新 CLI 并删除尚未被外层 campaign 消费的 additive
诊断代码；既有 strict runtime、`substrate` slot、P4.6 source pins 与 steering 三件套不变。

## Slice P4.6 outer scaffold（2026-08-22）：Windows/CUDA 一次性 host campaign

本 slice 新增的是离线 evidence control owner，不新增 live slot、策略、PE/credit、产品 wiring 或
production ACTIVE：

- `windows-cuda-strict-32k-host-campaign.v1` protocol 为 `cf62484f…3194`，raw SHA 为
  `5f174024…8f43`；三项 source pin 覆盖 Node owner、Windows PowerShell 5.1 collector 与固定 CLI。
  public API/CLI 不允许注入 collector/executor/validator 或替代 protocol；实现限定 repository
  source checkout，installed wheel 不构成独立可执行发行物；
- deterministic scope 绑定 outer/child protocol、qualification artifact、host identity 与 backend。
  000 scope claim 在 FindAnchors/Baseline 之前 create-only 落盘；002 raw SHA 是 lease；004 在 child
  creation 前 fsync 消费。完整链为 000–012 + 两条 stream + child exact-five；launch 后中断永久
  `incomplete_consumed`，完整失败 seal 后 `retry_permitted=false`；
- prelaunch 在 004 前复核 same-machine/same-boot、baseline boundary、Circular channel config 与
  RecordID 预算。PowerShell 按冻结 end cursor 用 XPath 读取区间；Node 独立重算 truncated、连续
  RecordID、boundary/end XML hash、config stability 与 fault classification，并把 prereg/launch
  anchor 的 RecordID/provider/event/payload/XML 与 delta 交叉绑定；
- test-only backend 使用独立 scope，固定
  `real_execution_observation_authorized=false`，强制
  `synthetic_test_backend_not_evidence`，永远 non-PASS；public validator 拒绝 synthetic chain。
  Node 回归 13/13 覆盖 synthetic firewall、duplicate scope、WHEA、合法 exit-2、float/int 类型漂移、
  prelaunch boot drift、collector failure、launch 后中断、no-retry 与 tamper；
- production 仍固定 `production_preregistration_enabled=false`。host qualification 的 protocol、
  synthetic publisher/full-root validator 已在后续 slice 落地，但 production probe/raw-Audit adapter/
  full validator 及 terminal→outer baseline cursor continuity 尚未实现；同时
  PowerShell/Python/Node runtime identity、realized child environment、完整 local import/source
  closure、process-tree containment、producer PASS-return 前整根重验、terminal/delayed-fault 区间覆盖
  与受限 Event Log source provisioning 仍是独立启用 blocker。Windows Application Event Log 只是
  本地第二通道，管理员和有写权限的普通本地主体都可能伪造，非 external/WORM authority。

本 slice 只运行 Node syntax/test、PowerShell AST parse、JSON/source-pin/lineage 复算；没有执行
Python、pytest、CUDA、模型或真实 Event Log 写入，也没有创建 production lease/PASS。四能力轴均为
`not_proven`。`.mjs` 回归不会被 pytest 自动发现；本 scaffold 的显式静态门为
`node --test packages/vz-runtime/tests/test_windows_cuda_strict_32k_host_campaign.mjs`，并由独立
Windows path-scoped workflow `.github/workflows/windows-cuda-host-campaign-contract.yml` 在 push/PR
运行。普通 pytest 通过仍不得解释成该 Node 门已通过。回滚为停止调用 outer CLI；由于 production
已禁用，无 live wiring 需要切换。已形成的 synthetic create-only chain 只用于测试，不得迁移为
正式证据。

## Slice P4.6 host qualification publisher scaffold（2026-08-22）

本 slice 只新增 `vz-runtime` offline qualification owner 的 synthetic publisher 与 synthetic
full-root integrity validator，不新增 live slot、真实 probe、Event Log consumer、CUDA execution 或
outer wiring：

- 冻结 `windows-cuda-host-stability-qualification-protocol.v1`：protocol ID
  `6d8a551775aa52f52ffa18c6e69ec6399e0f48f13abc122c9559851d5cc92a3a`、raw SHA-256
  `fc5e786274e963162a3ace72ab61d84271b9463f715df2c665e2cdbbb33d9e0b`；Node owner LF source pin
  `878bc348eb119df8601d643af326b6ec0e142be3a8ec11fe84dc7a45488a47d7`，provisioner pin
  `be0c02f136761f83412f31cdbf1f3249ad7ed15de1aff28e27fe1a8597888406`；
- synthetic root 恰有 000–009 receipts、010 manifest、011 terminal 与两条 stream。validator 要求
  exact 文件/目录、regular single-link/path containment、strict canonical JSON、raw receipt chain、
  manifest inventory/artifact ID、terminal ID/raw identity，并从 owner receipts 重算 report 与 terminal，
  不信任 terminal 自报 eligibility；
- microcode 从四个 little-endian bytes 解码后按整数比较 `>=303 (0x12F)`；Event Log projection 校验
  Application/System cursor join、record-count/boundary hash、event timestamp、每 window/channel 4096
  预算、300 秒 cooldown、120 秒 terminal tail 与 normalized provider/event-ID fault rules；没有保存或
  独立解析 raw Event XML；
- public `preregisterHostQualification / runHostQualification / validateHostQualification` 在读取任何
  option/path/Proxy getter 前静态 throw。只有 `__testing.createSyntheticQualificationArtifact` 与
  `validateSyntheticQualificationArtifact` 可用；完整 synthetic 根仍固定
  `synthetic_test_backend_not_evidence / criteria_passed=false / real_host_observation=false /
  validated_eligible=false`，terminal exact schema 禁止 `passed` 与
  `real_cuda_evidence_authorized`；
- 002/008 明确改名为 qualification-owned
  `windows-cuda-host-stability-source-audit-projection.v2`。它不是 PowerShell raw Audit v2，固定
  `full_raw_audit_bound=false / raw_audit_content_id_basis_revalidated=false`。provisioner hardening 把 Audit
  不合规变为完整 receipt 后 exit 2、process failure 变为结构化 failure receipt 后 exit 3；缺失 source 的
  Provision 必须显式携带 `-AllowSourceCreation`，且既有 drift 不自动修复。mutation 非事务性，source 注册后
  的 value/ACL/flush failure 会保守发布 refresh required；未创建路径发布
  `requires_cold_or_service_refresh=null`，绝不冒充已完成 refresh。Audit 比较含 provider membership 的完整
  channel endpoints；创建 source 的 Provision 除稳定投影外，还只接受 provider 列表不变或精确新增
  `VolvenceEvidence`，任意其他 provider membership 变化都不合规；列表不变表示等待 refresh，fresh Audit
  before/after 都必须确认 exact source membership 才能 exit 0。二者都明确
  `continuous_stability_proven=false`。module-qualified cmdlet 与 module/assembly hash 仍是非权威自观测。
  因此 exit code、config content ID、endpoint equality 或 refresh 字段均不能单独授权 qualification；
  production raw-Audit adapter、完整 raw/basis 复验与 refresh chronology 尚未实现；
- current outer protocol `cf62484f…3194` 未改动，继续 exact-schema 接受 terminal v1 并拒绝 v2。
  future consumer 必须同时绑定 qualification artifact ID、terminal ID、terminal raw SHA，并验证
  `(handoff cursor, outer baseline cursor]` 双 channel bridge；本包没有关闭该空窗。

实际验证：`node --check` 覆盖 owner/test；
`node --test packages/vz-runtime/tests/test_windows_cuda_host_stability_qualification.mjs` 为 15/15，覆盖
public zero-read gate、synthetic non-evidence、tamper、missing/extra/empty-directory、source-verification
禁用、4097-record overflow、Audit nonconformance exit 2、operator schema、window identity、little-endian
microcode、WHEA、source drift、duplicate key 与 noncanonical number；
`node --test packages/vz-runtime/tests/test_provision_volvence_evidence_event_log.mjs` 为 12/12，覆盖 v2 exit、
显式 source-creation intent、partial failure、refresh 三态、endpoint equality、MultiString、cmdlet provenance
与 source pin；protocol/source pins 已独立复算。provisioner 只做 PowerShell 7 与 Windows PowerShell 5.1
AST 解析，未执行 `Provision`/`Audit` 或查询 Event Log records。
因现有 host-block，未运行 Python、pytest、CUDA 或模型；未生成真实 qualification、PASS、production
lease，也未解除 BIOS/microcode block。Appendable / Readable / Learnable / Steerable 均为
`not_proven`。回滚为停止调用 synthetic helper/validator 并移除 workflow 中该 Node 静态门；production
和 live wiring 本来就是 `DISABLED`，已生成的 synthetic 根不得转作正式证据。

## Slice P4.6 raw Event Log Audit artifact adapter core（2026-08-22）

本 slice 仍由 `windows_cuda_host_stability_qualification` 这一 `vz-runtime` offline evidence owner
发布，只新增 raw artifact→non-authorizing adapter snapshot，不修改 PowerShell infrastructure owner、
002/008 synthetic projection v2、terminal/root shape、outer consumer 或 live wiring：

- 当前 `windows-cuda-host-stability-qualification-protocol.v1` 更新为 protocol ID
  `32f35e4f7027e9519522e099efb696fb352a48faf3ba69be861929304fae1d5f`、raw SHA-256
  `30a881838b41fa5b7e6de5aba6bc94131245796126be5b49c4ebab539f8c4132`；Node owner LF pin
  `7efff6c353d147f994a1e431903bb1ccb8772e89b7a99753fede5fd3434172e7`，provisioner pin 继续为
  `be0c02f136761f83412f31cdbf1f3249ad7ed15de1aff28e27fe1a8597888406`。上一 slice 的
  `6d8a5517…92a3a / fc5e7862…d9e0b` 只保留为历史身份，不再是当前 bundled protocol；
- `adaptProvisionerAuditV2Artifact` 从显式 artifact root 内的 regular single-link 文件以同一 descriptor
  读取 raw stdout；在 read 前用 `fstat` 同时核对 caller byte count 与 1 MiB 上限，再复算 raw SHA。
  `windows-event-log-source-audit-capture-envelope.v1` 绑定 pre/post role、0|2 exit、empty-stderr claim、
  machine/boot claim 与 100ns chronology，但 envelope 固定 `capture_authoritative=false`；adapter 不把它
  改写为 OS outcome、可信进程或 boot observation；
- raw stdout 与 machine-config basis 都要求 fatal UTF-8、无 BOM、唯一末尾 LF/compact ordered JSON、
  duplicate-key/float/unsafe-number 拒绝；basis 做 canonical base64 round-trip、exact ordered core
  reconstruction 与 SHA/content-ID cross-bind。Audit v2/exit 0|2 和 failure v1/exit 3 严格分流；Provision、
  source-creation/mutation、failure v1、capture/receipt exit 交叉或任何 schema/claim 漂移均 fail loudly；
- adapter 从完整 observations 重算 source values/ACL/owner、Application channel base state、provider
  membership/transition、full/stable endpoint、Application/source registry equality、overall/result/exit、
  provisioning、refresh 与 safety boundary。内部一致的 exit 2 只产生 diagnostic snapshot；
- 唯一新交换是深冻结
  `windows-event-log-source-audit-artifact-adapter-snapshot.v1`。它发布 protocol/raw/source/config lineage、
  caller envelope content ID、raw identity、recomputed conformance 与 canonical `snapshot_id`，同时固定
  `projection_emitted=false / real_provisioner_observation=false / eligible=false`，以及 CUDA、formal、
  production ACTIVE、四能力与 tamper-resistance 授权全部为 false。after-role 只称
  `caller_expected_content_id_matched`，不声称 scope proof；现有 synthetic predicate 完全未改；
- production direct PowerShell acquisition、failure-v1 quarantine、independent registry/channel reobserver、
  release/WORM trust anchor、新 production projection schema/predicate、pre/post replay exclusion 与 outer
  bridge 仍是后续独立包。真实 WinPS 5.1 `ConvertTo-Json` bytes 与 Node `JSON.stringify` 的逐字节兼容性
  也必须在 BIOS 修复后用 fresh Audit fixture 验证；本 slice 只证明静态 artifact self-consistency。

实际验证：qualification Node regression 为 40/40，覆盖既有 synthetic firewall 以及 raw SHA/size/exit、
failure-v1/Provision 分流、invalid UTF-8/BOM/duplicate/float/trailing JSON、compact/order、basis 重签、
provider order/duplicate、100ns 逆序、source lineage/value、registry/safety/refresh、caller config-ID 与
external protocol pin；provisioner 静态合同仍为 12/12，outer Node regression 为 13/13，三组组合为
65/65。owner/test 均通过 `node --check`，protocol/source identities 独立复算；PowerShell 脚本未改，仍只做
PowerShell 7 与 Windows PowerShell 5.1 AST parse，未执行 `Provision`/`Audit` 或 Event Log 查询。

因 host-block 仍未解除，本 slice 没有运行 Python、pytest、CUDA、模型或真实 Event Log，也没有生产
acquisition、qualification/PASS/lease。Appendable / Readable / Learnable / Steerable 继续全部
`not_proven`。回滚为停止调用 `adaptProvisionerAuditV2Artifact`；production 和 live wiring 原本就是
`DISABLED`，任何 adapter snapshot 都不得迁移成 qualification projection 或正式证据。
