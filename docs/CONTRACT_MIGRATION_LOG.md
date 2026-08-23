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

## Slice P4.6 standalone Audit acquisition supervision/source-binding v2（2026-08-23）

本 slice 只修改 `windows_event_log_source_audit_acquisition` 这一 `vz-runtime` offline evidence owner，
不修改 qualification protocol/root、002/008 projection、raw Audit adapter、outer consumer 或 live wiring：

- acquisition protocol 升为 v2：ID `3ed1005fc06c443ba7af4e60b254d6afeb2f84168177322a26cea0856fbe5be0`，
  raw SHA-256 `b01c3ae23c066b8d53d0ecc337e275b8177d3be6ab4f4b787ef51a48a0be1c1b`；owner LF pin
  `b5dfaa8245a8193cf506ce6cf111c95e89b46ceb214ac47eeeff7cd90b22f965`，`.gitattributes` LF pin
  `feec1dfa62d773d1ac7e6e00696d19670634b3ef93048a564b2f4ea42fd133d8`，provisioner pin
  `be0c02f136761f83412f31cdbf1f3249ad7ed15de1aff28e27fe1a8597888406`，CLI pin
  `9cacd8e178d6528b7b7bfa319ba18037a9e32e19b2aa3921434016299740491d`。旧 v1
  `1d962501…57f700 / 7f4d19c0…26e98d / e59b386c…29f5` 仅保留历史身份；
- process observation/terminal 升为 v2。exit/close observation 分别发布
  `child_exit_event / child_close_fallback / not_observed` 与 `child_close_event / not_observed`，未观察时间为
  null；kill request accepted 与 termination 分离。120 秒 soft timeout、5 秒 post-kill drain grace 和
  125 秒 overall referenced timer 由单一 `finalizeOnce` 竞争封口；kill false/throw/true、parent exit 但
  descendant 持 pipe、late close 都不能无限等待或在 finalize 后升级 candidate；
- EventEmitter callback 不再写 descriptor，只做 1 MiB/256 KiB bounded buffer。stdout/stderr 各自发布
  pipe end/close、forced detach、observed/persisted byte count、capture/persistence 首错及 complete 状态。
  lifecycle finalize 后才写、fsync 并从同一 `wx+` descriptor readback/hash；write/fsync failure 若仍可
  readback 会 quarantine，readback/terminal write failure只能留下 `incomplete_consumed`，不得伪造 terminal。
  真实 candidate 同时要求两条 pipe 的 end 与 close；hard cutoff 后保留 error guardian 到 child close，late error
  只发布 hash 化诊断而不修改 frozen observation。async supervision deadline 不覆盖不可中断的同步 persistence；
- requested source execution 改为 protocol-derived UTF-16LE `-EncodedCommand`。launcher UTF-8/UTF-16LE
  hashes 为 `2997b0ec…cf842 / d48be714…8f290`；它用 FileShare.Read 同柄 bounded-read provisioner，分别
  验证 raw/LF pin、BOM/strict UTF-8、Parser.ParseInput exact buffer 与固定 `-Mode Audit`；handle 贯穿绑定脚本
  执行与 exit unwind，但不声明持有到 OS process exit event。
  pre-endpoint 还必须 exact join protocol raw/LF pin。该机制不允许 caller 覆盖 source/payload/mode，但只
  冻结 requested binding；realized PowerShell image、IFEO/DLL/module/environment、ancestor reparse、管理员/
  内核对手仍未 attested；reviewed provisioner 增加更具体的 `.gitattributes eol=lf`，并把 `.gitattributes`
  纳入 critical source pin，避免 Windows fresh checkout 将 raw bytes 物化成 CRLF；critical-source LF decoder
  保留 UTF-8 BOM，使添加 BOM 必然造成 pin drift；
- workflow 的 push/PR paths 补入固定 CLI，并新增 CLI `node --check`。qualification protocol
  `32f35e4f…e1d5f / 30a88183…c4132` 保持不变，production gate 与
  `acquisition_to_qualification=DISABLED` 未翻转。

实际验证：acquisition Node regression 为 59/59。它除原 synthetic/validator/adapter handoff 外，还覆盖
kill false/throw/true never-close、exit-without-close、close-without-end、cutoff 后 late child/pipe error、
pipe error、write/fsync/readback failure；真实 Windows
P/Invoke fixture 让 parent exit 后由 descendant 继续持 stdout/stderr，在约 8.3 秒 hard cutoff 后返回，并按
fixture PID 终止且确认清理。Windows PowerShell 5.1 fixture 证明 Parser(fileName) 设置
`$PSCommandPath/$PSScriptRoot`、exit 2、执行期间 rename=`EBUSY` 与退出后 release；遗留
`LASTEXITCODE=0` 的普通 return、raw 不同但 LF 相同、raw 相同但 LF pin 不同、BOM 与非法 UTF-8 均 exit 3。
owner/test/CLI 通过
`node --check`；与 qualification、provisioner、outer 的四组组合门为 124/124。本 slice 未执行
Provision/Audit、未查询或写入 live Event Log，也未运行 Python、pytest、CUDA 或模型。

因此本 slice 只关闭 acquisition 启用前的 process hang、callback I/O escape 与 requested source-content
reopen 三项局部 P1，不构成真实 acquisition、host qualification 或四能力证据。CUDA、formal、production
ACTIVE、Appendable / Readable / Learnable / Steerable 继续全部 `not_proven`。回滚为停止固定 CLI/owner；
production/live wiring 本来就是 `DISABLED`，任何 complete 或 incomplete root 都保留为 immutable
non-evidence，不迁移到 qualification。

## Slice Relationship P4.7（2026-08-23）：独立长 context 四轴因果实验 zero-output design freeze

本 slice 只新增 `lifeform-evolution.relationship_lab_p4_long_context_causal_campaign` 这一 offline
scientific-prereg owner，不实现 subject source、模型/CUDA runner、execution envelope、formal assessor 或
live/product wiring：

- protocol `5387516a803940a738e13bb47acc8a40b837c3f033797e09dbfaa23c6cda6d2e` 在零新
  subject materialization、baseline/model output、CUDA formal run 与 formal outcome 时冻结；owner 只提供
  `show-protocol / prepare / validate-existing`。create-only preparation/manifest 逐文件绑定 byte count/SHA、
  canonical manifest core 与 artifact id，并固定 execution/formal/model-output/subject-materialization=0。实际
  preparation artifact `899b7b0adc395186e108dc0a90c28c0d25ce67cd5445f61636cc2775d09b6901`
  已发布，GPU-free `validate-existing` 重建同一 id；
- 独立分析单位从已见的两个 synthetic fixture 升为 disjoint `subject_root` inventory：development /
  qualification / sealed formal 预分配 32/64/192，formal 少于 160 个完整 paired root 只能
  `inconclusive_underpowered`。每个 root 固定 4 onboarding + 8 arm-independent matched learning exposure +
  8 frozen-policy evaluation session；每 session 要求 fresh OS process 与隔离磁盘 hydration；
- 九臂共用 `volvence_closed_loop` 参照，分别隔离 empty/swapped hydration、同 reader label permutation、
  exact PE-credit withheld、strict residual noop、matched sensor-off、full-history steelman 与 selective-RAG
  steelman。逐臂只允许一个预注册 JSON pointer 外生变化，删除该路径后的配置 hash 必须完全相同；
- 修正 P4.4 Learnable 混杂：PE-credit/no-credit primary pair 在 learning phase 使用逐 session exact-matched
  action exposure、actual outcome、hidden step 与 common random tape，只有 credit 是否送入 gate 不同；
  actual-exposure receipt 前禁止 settlement，SHADOW suggestion 不得结算。gate 冻结后才在未来 session
  观察 policy-selected action/outcome；
- 修正旧 P4.1/P4.6 context 不可行性：32768 native window 无法同时容纳 32768 public-history token、
  system/request overhead 与生成。P4.7 因而要求 actual public-history ≥32768、至少 96 个不同 public turn /
  16 个 typed settlement、排除 filler/padding/重复/static instruction，strict native window ≥65536、generation
  headroom ≥1024、full-history no truncation。现有 P4.6-fit 只保留 development lineage，不能授权 P4.7；
- 五个 confirmatory claim 以 subject root 为单位，minimum mean delta=0.15、100000 次 frozen-seed
  subject-cluster bootstrap、CI lower>0 与 Holm FWER 0.05；五项任一失败即四轴总判 false。Qwen full-history
  与 RAG 必须先各自进入冻结 informative band，否则停止且不打开 formal；
- 用户允许 CUDA 只发布 `development_cuda_diagnostics_allowed=true`，同时固定其不是 formal 授权。
  historical host block、acquisition v2 非授权、manual/env/ignore-microcode/force 禁止均进入 protocol；未来
  execution envelope 只能补 exact subject/model/artifact/host/reobserver/cursor/Job Object/outcome-typing/power
  lineage，禁止更改科学设计。

实际验证：新增定向 pytest 7/7，覆盖 protocol ID/cohort/horizon/32K→64K feasibility、development/formal
权限隔离、Learnable shared exposure、create-only publication/GPU-free replay、protocol drift/duplicate key/BOM、
同时重签后的 output claim tamper、extra file 与 lifeform-evolution 不直引 cognition internals；CLI/owner
通过 Python compile，三个改动 Python 路径的定向 Ruff 为 PASS；与 import-boundary 组合回归为
2920/2920。未运行模型、CUDA、
subject generation 或 formal analysis，也未新增任何四轴证据。回滚为停止该 CLI/owner并不让后续 envelope
消费 `5387516a…6d2e`；已发布 preparation 保留原 hash，新设计必须另发 protocol id。

## Slice Relationship P4.7-v2（2026-08-23）：执行前红队收口与可唯一重算因果合同

本 slice 没有运行任何 subject、模型、CUDA formal campaign 或 outcome。三路独立只读红队对 v1
`5387516a…6d2e` 给出无 P0、但有必须在首条 development output 前修正的 P1：qualification retry 未冻结；
`0.15` practical delta 未进入 PASS；Holm family、root estimand、缺失处理不能唯一重算；shared exposure 把
exogenous tape 与 action-dependent reaction 混在一起；Readable/Learnable/Steerable/Appendable 的 realized
single-variable receipt 不够强；32K 可被无因果作用长文满足；sensor-off 无预注册判据；validator 把 canonical
object key 顺序当语义，并允许 protocol+ID 重签后打开 free bias/outcome replacement、删除 firewall 或替换
stopping rules。v1 preparation `899b7b0a…6901` 仍按原 bytes 保留，registry 标为
`preserved_zero_output_design_only`，禁止重新 prepare、执行或迁移任何未来 output。

该 slice 当时的权威 v2 protocol 为 `666d2e8546cd4b4cf55ece06354310e10b4dc07298241b94ef9593e4b5f63baf`；
create-only preparation artifact 为
`795dea07eabda98c964ca50ee84694fd93bb6ee27fdd0370db7e6cb3ef01a8bd`：

- v1/v2 immutable registry 固定 schema、canonical id、bundled raw SHA、preparation/manifest schema、status、
  supersession 与已发布 artifact id；在该 slice 中默认 `prepare/show` 只指 v2。`validate-existing` 先比较 artifact 内
  preparation/manifest 的 protocol id，再自动分派 bundled protocol，因此 v1 仍可无参数复验。object key
  reordering 保持 canonical acceptance；artifact 本身继续要求 canonical bytes；v2 每个顶层 section 另有
  independent canonical anchor，改 protocol 与全局 ID 仍不能绕过 nested scientific/safety literals；
- 九臂把误名的 `volvence_typed_noop_control` 改为 `learnable_credit_withheld`。8 个唯一 Holm member 分别是
  Appendable 两项、Readable 一项、Learnable 一项、Steerable strict-noop 与 mandatory non-pure sensor-off
  两项、closed-loop 对两个 steelman 两项；Learnable 可被 integrated 合取引用但不重复检验；
- 每项 PASS 必须同时满足 lineage、observed paired-root delta `>=0.15`、100000 次 joint root bootstrap 下界
  `>0` 与 Holm FWER 0.05；192 个预分配 root 全部进入 contrast-specific worst-case ITT，少于 160 个全局
  complete root 判 underpowered。普通缺失保守插补，truncation/leakage/intervention/evaluation-feedback receipt
  违反直接 `invalid_attempt_no_claim`；
- qualification 明确 one-shot、campaign/session retry=0、replacement=0。baseline accuracy/pair-flip 以 64 个
  subject root cluster 计算，不把 512 decisions 当独立 Bernoulli；首条 development output 前需完整 decision-rule
  prior power artifact，qualification 后只允许 arm-blinded zero-centered covariance 的一次只读 power recheck，
  失败只停止，禁止改 N/阈值/family/source；
- exposure 拆为跨臂 exact 的 frozen exogenous tape 与 frozen transition function 产生、可随 action 分叉的
  endogenous reaction。Readable 只切 sealed identity/derangement label map；Learnable 先收集八次 exact
  action/outcome/PE/credit，再从同一 cold checkpoint 做一次只差 apply bit 的 atomic batch；Steerable 的同
  residual 只约束首个同拍 manipulation diagnostic，纵向主效果允许合法 treatment descendants；实际 action
  必须由 generated token→frozen parser 单链产生；
- 每个 root 至少有 8 个反事实会改变唯一最优 action 的 typed long-range fact；最终一个 fact 距 request 至少
  32768 actual public tokens。每臂每 evaluation decision 都绑定 tokenizer/chat template、ordered/omitted chunks、
  attention mask、rendered input ids、native window/headroom/truncation 与 fact occurrence；RoPE/runtime context
  extension、filler 与 fact restatement 均不能过门。

当前只执行了 v2 owner/CLI Python compile、22 个 P4.7 定向测试，以及 v1/v2 `show-protocol`、v1 artifact
auto-dispatch validate 和 v2 create-only preparation；均通过。未运行 model/CUDA/subject/power simulation、
qualification、formal 或全仓 pytest，因此没有新增 Appendable / Readable / Learnable / Steerable 证据。
回滚只是不让未来 execution envelope 消费 v2；v1/v2 protocol 与两个已发布 preparation 必须继续按原 hash
保留。下一包必须先实现并冻结 power/source/tape/context/artifact inventory，而不是直接生成 development output。

## Slice Relationship P4.7-v3（2026-08-23）：独立单位、typed endpoint 与完整 power DGP 收口

v2 `666d2e85…3baf` 仍是在零 subject/model/CUDA-formal/outcome output 时冻结的历史设计；后续只读红队发现，
它虽然关闭了 8-contrast/ITT/realized-lineage 主骨架，仍会让 swapped-prior donor 从 analysis roots 取样而引入
跨 root 依赖，也未把 utility domain、逐 decision missingness、baseline candidate/qualification bootstrap、
long-range proxy 排除、全九臂 generated action、host-time 顺序与 joint power DGP 冻结到唯一可重算。因此 v2
preparation `795dea07…a8bd` 原样保留为 `preserved_zero_output_design_only`，禁止重新 prepare、执行或迁移 output。

该历史 slice 当时的 design authority v3 protocol 为
`9f352778e128a9573790762222a05225740bdaeb732800dec0eec124116a282d`，bundled raw SHA-256 为
`ea8a17a14a68802d3b60586bf520c9137e6920be4112c951ec8c69f5e6ea359e`；create-only preparation 发布于
`artifacts/relationship_lab/p4_independent_long_context_causal_campaign_design_prereg_v3_20260823/`，artifact id
`c5a708ae5e68261fddbade165b45579e66e4bbe7db1be1f4a83056561a17f42e`。该 artifact 仍只含 zero-output
preparation/manifest，固定 `execution_enabled=false / formal_run_authorized=false` 与全部 model/subject/donor/
twin/power-DGP count=0，不是 power/source preflight、execution envelope 或四轴证据：

- development/qualification/formal 仍有 `32/64/192` 个 analysis roots；每个 split 另建同规模、与全部 analysis
  roots 及其他 split 不相交的 donor bank，一名 analysis root 唯一消费一名 donor，禁止 donor reuse/replacement，
  donor 不进入 estimand/bootstrap。donor failure 只令对应 target swapped-arm decisions 缺失并走预注册 ITT；
- frozen reactive environment oracle 为每个 evaluation decision 预先提交覆盖全部 closed-world actions 的
  `{-1,0,1}` utility vector，model input 前 commit、transition 前隐藏；actual action 只能机械 lookup，禁止事后
  rescale/rescore。192-root ITT 先对 8 个 decision slot 逐项做 contrast-specific worst-case imputation，再求 root
  mean；pre-evaluation/evaluation crash 的缺失范围与 both-missing `-2` 均明确，不能整 root/arm 简化；
- 三个 development candidate 都是 full-history/RAG paired configuration，在同一 32 roots/tape/opportunity 上运行，
  用冻结五级 lexicographic key 唯一选择，禁止人工 tie-break 或朝 informative band 挑选。qualification 固定 root
  accuracy/pair-flip 定义、100000 次 seed-20260825 SHA-256 index plan、joint root resampling、inverted-CDF 0.05
  order statistic 4999 与 exact rational arithmetic；
- fact value 在 split×distance bin 内平衡且独立于 surface/tape/arm/outcome。每名 analysis root 配一份不执行模型、
  不进 arm/estimand 的 counterfactual source twin；除 fact origin 与 utility-oracle descendants 外 bytes/receipts
  必须相同，dependency graph、semantic proxy scan 与 independent reobserver 任一失败即
  `invalid_attempt_no_claim`，防止近程 proxy 冒充 32K 因果读取；
- `global_generated_evaluation_action_contract` 覆盖九臂全部 evaluation decisions：只接受 user-visible generated
  bytes 经 frozen parser 恰解析一个 action，禁止 repair/reprompt/second parse 与 hidden direct-action channel。
  唯一例外是 matched-learning forced action，且明确排除出 Steerable/evaluation endpoint；
- `subject_root × session_index` block 内九臂各执行一次，顺序由 committed SHA-256 Fisher–Yates/rejection RNG
  决定；各 ordinal position 跨 split 最多差 1、root blocks 交错、同 block 同 boot，记录 wall/monotonic/boot，
  禁止 arm/output/outcome/failure 改序；
- confirmatory bootstrap 冻结 192-root SHA-256 rejection/modulo index plan、order statistic、exact rational/Holm
  比较。prior joint power DGP 必须覆盖全 9×8 discrete utility、8-contrast covariance、within-root dependence、
  source opportunity mix、技术缺失与 worst-case ITT，并遍历预注册 variance/ICC/cross-contrast dependence/
  missing-rate/pattern 的全部可行 Cartesian 场景；每个 contrast、轴合取与 integrated joint power 都须 `>=0.80`，
  不能用 zero-missing 或任意低方差单景授权。

v1/v2/v3 registry 继续按各自 protocol/raw/schema/status/artifact id 自动复验历史 artifact；默认
`show-protocol / prepare` 指向 v3。当前仍未生成 subject source、donor bank、counterfactual twin 或 power DGP，
未运行 baseline/model/CUDA/multi-session/qualification/formal；下一包必须先实现并冻结 power/source/tape/context/
artifact inventory 的 zero-output preflight，全部通过前 model output 明确 NO-GO。回滚只是不让未来 execution
envelope 消费 v3；三代 protocol 与三份 preparation 都按原 hash 保留，不能用后续版本覆盖。

## Slice Relationship P4.7-power-admission-v2（2026-08-23）：否决 v1 无条件适用性并冻结欠规定终局

historical power-bound v1 protocol
`735b20a137b03176cf889c0cbe116e29f973c18d4cef4bf38cd42df288dff3fa`（raw SHA-256
`1bb8d21ce3a0dca332324d2e35e3bc2c63ec77fc9bd3917b35d018ebd85559f6`）、artifact
`fad6c105b7c64a6b4ab89bf6e933ecdf4c8f1b1170679d918c2dd77c27809518` 与 certificate
`682efba886b002db849a83ff086963921a173391a4d5e3c050b3d472d17ee70e` 原样保留。其 N=192 最大方差
witness 的精确尾概率 `0.62046904104455107035 < 0.80` 算术正确，但科学终审否决了对 v3 frozen grid 的
无条件适用性：v1 使用的 sentinel-before-Cartesian-filter precedence 是后发语义；witness 派生的 temporal/cross
correlation 均为 `+1`，不属于 v3 显式 ICC/cross-dependence labels；它也没有建立尚未物化的 source-structural
membership。因此 `scientific_admission=false` 只表示“未证明被纳入，也未证明不可行”，该数字只能作为
conditional diagnostic，不得再发布为 v3 decisive numeric FAIL。

该历史 slice 当时的 power-admission authority v2 protocol 为
`67d294faf9209c9d05334f4c0e87371676c9821b7c12e603f3e289f33f566bc9`（raw SHA-256
`130f766787ec0b02bd5857344e58b371d996aa51fabe421cbfbde05347fd0e04`）。create-only 输出位于
`artifacts/relationship_lab/p4_independent_long_context_power_admission_v2_under_specified_20260823/`，artifact id
`9883e10784a06260a220a6fdbf72141b1300c21e97faee6e84a401c40a144ee9`，certificate id
`cd6ceca086a1d8a311c75bdacd70c976e05b90dff2cde55b3ad41c00d29936b3`。v2 固定两个都保留 v3 literal
lists、却对 `+1` witness membership 给出相反答案的解释，且 v3 自身没有 precedence/survival rule 能机械选一；
因此当前唯一终局是 `power_contract_under_specified_no_development_authorization`：

- `power_contract_determinate=false / v3_power_failed_under_frozen_grid=null / v3_power_passed=false`；其中
  `passed=false` 表示 prior power admission 未满足，不是完整 power estimate 或无条件 numeric FAIL；
- v3 在任何 subject/source/donor/twin/model/CUDA-formal/simulation/full-joint-DGP output 前退休；相应计数全为 0，
  `execution_enabled / development_authorized / qualification_authorized / formal_authorized` 全为 false；
- 该 certificate 不是完整 joint DGP/source preflight/empirical result，也不增加 Appendable / Readable / Learnable /
  Steerable、integrated、human、product 或 production ACTIVE 证据；
- v4 在任何 source/model output 前必须冻结明确 pre-filter sentinel 或完整 grid-membership rule、每个 skipped tuple
  的机械 infeasibility witness、planning mean 相对 missingness 的定义顺序、malformed generated action 与
  integrity failure 分类、六个 development candidate-cell 的 counterbalance/state isolation，并在完整 joint power
  planner 后冻结新 sample size。不得在 v3 原地改 N、追认 sentinel 或把旧数字迁移成 PASS/FAIL。

验证记录：当前 CLI `prepare`、`validate-existing` 与 `validate-v1-existing` 均已成功，相关代码 Ruff、format 与
compile 检查均通过；power-admission 专用测试为 `17 passed, 1 skipped`，合并 campaign、power-admission 与 import
boundary 的直接相关回归为 `2967 passed, 1 skipped`。唯一 skip 是 Windows 当前权限不允许创建普通文件 symlink；
目录 junction/reparse 防护测试已实际执行。`validate-v1-existing` 是只读历史复验能力，不改变 v1/v2 artifact。
回滚只需停止 power-bound CLI 与任何未来 consumer；v1/v2 protocol、
两份 create-only artifact/certificate 及 v3 preparation 均按原 hash 保留，runtime/live/product wiring 本来就是关闭的。

## Slice Relationship P4.7-v4a（2026-08-23）：确定 planning 语义并冻结 zero-output abstract schedule

本 slice 继续由唯一 owner `lifeform-evolution.relationship_lab_p4_long_context_causal_campaign` 发布 frozen public
`RelationshipP4LongContextV4PlanningProtocol` / `RelationshipP4LongContextV4PlanningFreeze`，不新增 runtime slot、
snapshot writer、live/product wiring、model/session runner 或 execution consumer。v1/v2/v3 design、historical
power-bound v1 与 power-admission v2 全部按原 hash 保留；v4a 只 supersede 当前 planning authority，不重写历史判词。

v4a protocol id 为
`63e007b7d43bb152e5891162d6567c4edd4396af99cf1c5525c28d0be4c08753`，raw SHA-256 为
`d06b07101624b3996bd712c98d3c633b7b00af7a878912817b5149a199c00e0a`，derivation helper raw SHA-256 为
`bf38e7ab89c56bdae8844f533cac077443d157a793c698adbb11a9591e32a0ef`。create-only 输出位于
`artifacts/relationship_lab/p4_independent_long_context_v4a_zero_output_planning_20260823/`，artifact id
`082454002260db90b7236a1104311a5d92cc3959171bb3190e7a30f8387e56c1`，certificate id
`b7e95f149afe77b283bf135f7cb5d76eb4f4edee4594c8649a778acb4186c764`；plan/screen/schedule/manifest raw
SHA-256 分别为 `9e17383f416eea555799d7e603996a34d526c7c20e9e65e53af25196a700064f`、
`d8f0f6b4fa1927138007bac77b687f3507b09ca0f000c6549b584ba2d33b01ba`、
`df426477209d0e99c74cf62938fcf3700554c6242f9439c2e51ebdd20edf1d6f`、
`26b46683260dc01f632ff9c1874839760f4b075c53eb5cd0298c7fc025633e3e`。

v4a 已机械关闭 v3/v2 暴露的 planning 语义缺口，但没有运行 full planner：

- mandatory global sentinel 明确早于 source-conditioned Cartesian enumeration/filter，且不可被 source/grid label
  过滤；source grid 固定 5 axes、`576` 个 candidate tuple，每个 tuple 最终只能带完整 9-arm×8-decision
  constructive witness 被 admitted，或带 exact constraint/proof payload 被 skipped；timeout、nonconvergence、
  resource exhaustion 与“未搜到”都不是 infeasibility proof；
- planning mean 固定为 substantive malformed 映射之后、technical missingness/ITT 之前的 complete-data mean。
  authenticated zero/multiple/out-of-domain generated action 作为 substantive invalid action 以 `-1` 留在 ITT；
  authenticated technical failure 走 worst-case ITT；receipt/bytes/parser/parent lineage 破坏是
  `invalid_attempt_no_claim`，不得降格成 `-1` 或 missing；
- development inventory 是两个 baseline family × 三 candidate index 的六个 state-isolated cell；CMS、State-KV、
  gate checkpoint、RNG、generation cache、log 与 selective-RAG index 均 cell-local。内容寻址 abstract schedule 有
  `640` 个 block，每 block 六 cell 各一次，以冻结 Williams-balanced order 平衡 ordinal 与 carryover；block 按
  session-major、再按 root ordinal 枚举，所有 root 完成 session `s` 后任何 root 才可开始 `s+1`；任何一个 cell
  integrity failure 都使 selection attempt 无效；
- future planner 冻结 `192..8192`、步长 `64` 的 126 个 candidate N，并以 paired-root Hoeffding upper bound +
  eight-contrast Bonferroni 替代 v3 nested bootstrap/Holm loop。screen 对全部 126 个 N 分别发布 reduced exact
  fraction，numerator/denominator 必须是无前缀、无前导零的 lowercase hex，禁止 monotonic shortcut；实际存在
  `1088=PASS → 1152=FAIL`，因此 first PASS 不能成为后续 N 的下界。N=`1856` 也仅是 practical-boundary mean
  首个满足 Hoeffding positive-mean gate 的候选，不是 selected N，不能发布 full power PASS；
- future categorical RNG 冻结为 `sha256_multiblock_counter_exact_rational_categorical_v1`，覆盖任意 exact rational
  denominator `Q`：`Q=1` 不 hash；`Q>1` 以 `b=bit_length(Q-1)`、`h=ceil(b/256)` 拼接 `h` 个 SHA-256 raw
  digest、取最低 `b` bits 并对 `u>=Q` rejection，candidate N 不进入 counter，CPU/CUDA 必须发布 digest equivalence；
- search 只按整数门 `50*X_search >= 41*8192` 提议最小 N；one-shot confirmation 对 feasibility index 冻结的
  `M=1+A` 个场景各跑 100000 replicate，并仅按 exact integer rule
  `100*M*sum(k=X..100000, C(100000,k)*4^k) <= 5^100000` 判 PASS（等号通过），所有 M 个 scenario 都必须通过。

当前 terminal status 是 `v4_planning_contract_frozen_full_joint_planner_pending`：
`power_contract_determinate=true / source_grid_resolved=false / full_joint_grid_completed=false /
feasible_tuple_count=null / skipped_tuple_count=null / grid_digest=null / sample_size_selected=false /
selected_formal_root_count=null / v4_power_passed=null / v4_power_failed=null /
v4_prior_power_admission_satisfied=false`。source structural/full-joint-DGP artifact、power search/confirmation replicate、
subject/donor/twin materialization、baseline/model output、CUDA planner/formal run 与 empirical outcome count 均为 0；
`source_materialization_authorized / model_output_authorized / development_authorized / qualification_authorized /
formal_authorized=false`。因此本 slice 不是 source preflight、power PASS/FAIL、empirical result 或 Appendable /
Readable / Learnable / Steerable / integrated / human / product / production ACTIVE 证据。

后续 artifact 顺序固定为 `v4a_zero_output_planning_freeze → source_opportunity_preflight →
tuple_feasibility_index → power_search_artifact → one_shot_independent_power_confirmation_artifact →
v4b_scientific_projection`。feasibility index 必须早于 power result；search/confirmation 不得改 tuple membership 或
generator；confirmation 失败不得在同一 protocol 下回到 search。v4b 只能投影 prior artifact ids、selected N 与
由 N 机械派生的字段，其他科学字段必须 pointer-match v4a。

验证记录：v4a CLI `validate-existing` 已通过；Ruff check 与 Python compile 检查均通过。v4a 专用测试为
`21 passed, 1 skipped`；合并 scientific campaign、power-admission、v4a planning 与 import-boundary 的直接相关
回归为 `2990 passed, 2 skipped`。两个 skip 都是 Windows 当前权限不允许创建普通文件 symlink；hardlink、目录
junction/reparse root 与 dangling output alias 防护均已实际执行通过。未运行全仓测试：本 slice 没有修改 runtime
wiring、共享 snapshot/schema 或全局初始化，直接相关合同回归已覆盖影响面；也未运行 source/model/CUDA/power
replicate，因为 v4a 的零输出 firewall 明确禁止这些动作。回滚只需停止
`run_relationship_lab_p4_long_context_v4_planning.py` 与后续 consumer；v4a protocol、plan、screen table、schedule、
manifest 与历史 v1/v2/v3 artifacts 全部按原 hash 保留，不得删除后以新 N 或新 schedule 覆盖。

## Slice Relationship P4.7-source-opportunity-preflight-v1（2026-08-23）：冻结零输出 source 合同，不启动 materializer

本 slice 继续由唯一 owner `lifeform-evolution.relationship_lab_p4_long_context_causal_campaign` 发布 frozen public
`RelationshipP4LongContextSourceOpportunityPreflightProtocol` /
`RelationshipP4LongContextSourceOpportunityPreflightCertificate`。它消费并严格复验 v3 preparation、power-admission
v2 与 v4a planning artifact，只新增离线 source-contract projection/certificate/manifest；不新增 runtime slot、snapshot
writer、source/model/session runner、live/product wiring 或 execution consumer。

protocol id 为 `47bcf6561be1ace0698cc0f96e2e7e35701f46d15baac9eb87ad1d662576494a`，raw SHA-256
`9d4d3ab5cb683d8ff5827e5047e5b176800fe5c4e86ad6a07217b7a2040c40b0`、`34883` bytes；zero-output section
SHA-256 `912e45f629b7fe4c0807144c694bf6d631111118925144e8e13e477b11dfee01`。纯推导 helper raw
SHA-256 `72efc093b815c2ca07872f6cb6a78f53a4d4d5ada5975222b36cf90c640746f8`、`59810` bytes。create-only
artifact 位于 `artifacts/relationship_lab/p4_independent_long_context_source_opportunity_preflight_v1_20260823/`，
artifact id `8a36d2de9077bb5550db8018338eded27b6ce30d77eea17739ffe35b73e00a99`，certificate id
`64d879c4f41ca873f8e40f0344234771343f6efee229b668914b61d31c96c95a`，contract projection id
`b8b7823a6fd2c7ad706c4ffa143438b730da667c26a925f0be87df14212e6f1b`。精确文件集与 default-stream
raw identity 为：

- `source_opportunity_contract_projection.json`：`72969` bytes，
  `ee33fa32a3829cbaaa1c92022016c184197b4cd97e08818af19b98024b4866b2`；
- `source_opportunity_preflight_certificate.json`：`8036` bytes，
  `f9089ce08e6868d402a753ccd3247024a0170a96e8cad9f411f659e913300736`；
- `manifest.json`：`2036` bytes，
  `2829b16d674ae9efe971eaa668610f80a20799e45ac47e208ec4e7a3261760a6`。

projection 内关键有限 inventory digest 为：root
`baafdcd3e54ccfd03a771ea00b59eb542ed1230c8543e0149344a54bbe897346`、analysis/donor pair
`10b3d5fe1de1cad799b6a99af26027d0be2ad1d536389cd2e0fd8d0fb563a326`、fact orientation
`cba5e6141a908b9aa4e726d3f8de147dd1b410b9638e60c8d63c6fc6ffe15985`、formal candidate position balance
`220c97186ee8cafc51476eea222a1f007b2097e01883bb94e89192435d3ee22f`、counterfactual twin
`689b91020b058f3726638ae3af7b44f33891375e32c9ae1b9a6609e57bf4bf29`、generic-decision atom
`82c97e44c2a20a87736d588fb63d83af13f8d473776fe530eabb0b04149d00ca`。这些 digest 只绑定 finite
contract inventory，不是 source row、realized root independence、model response 或 empirical evidence。

zero-output counters 的口径只覆盖 persisted/published artifact rows。strict replay 在内存中机械构造的 frozen
root/surface/orientation/twin/atom derivation objects 只是确定性合同复算，不是 source content，也不构成持久化
materialization；三文件仍没有逐 root source row。此前预检导入/输入边界的漏洞也已关闭：公共 package API 使用
lazy export，CLI 只把仓库内 exact package source roots 加入 import path，输入 lineage 在读取前拒绝 reparse 与
hardlink。该 closure 不扩大为管理员级 tamper resistance、WORM 时间锚、NTFS alternate stream 完整性或全机历史证明。

红队要求的五个科学 P0 已按合同层关闭：

1. **可解性与 truth firewall**：typed truth、utility optimum、source stratum 和 oracle metadata 对未来 SUT 隐藏；
   decisive fact semantics 则必须且只在注册 origin 公开一次。禁止把“隐藏 evaluator truth”误写成“模型永远看不到
   任务所需 decisive fact”；
2. **反 parity shortcut**：不再对每 root 固定 `[0,1]`。seed `20260831` 与固定 ASCII domain 的 SHA-256
   只按 split/pair/32-root-block/within-block 排名，digest collision 再按 within ordinal；rank 0–15 / 16–31 分别
   orientation 0/1。每 block/pair 精确 16/16，全部 126 个 formal N prefix 在每 pair 与两个 decision position 都
   fact0/fact1 平衡；candidate N、arm/cell、model/output/outcome、power、host、CUDA 均不进 counter；
3. **surface code 不是空编号**：capacity `2^15` 的 affine code 按 LSB 顺序解码 15 个 exact binary causal axes，
   每个 bit 写入同名 typed blueprint value，并冻结 root/session/opportunity/fact-origin/target-decision/
   registered-utility-oracle node mapping。global slots `0..16575` injective，analysis/donor 各 8288，formal
   `192..8192 step 64` 均是严格 prefix，donor 同 ordinal 一对一；
4. **twin 是唯一 intervention**：8288 个 analysis root 各有一个不进 N、不会执行的 twin；它只翻转 pair 3、
   decision 7、32K opportunity 的 decisive fact `0↔1`，重算 registered utility-oracle descendant，其他 exogenous
   node 不变。Prediction Error 仍由 cognition owner 在未来 realized settlement 后计算，source graph 不预造 PE；
5. **generic marginal 与 temporal/tuple witness 隔离**：512 atoms 只枚举一个 generic decision 的独立 reference
   `11/20` 与八 comparator `9/20` correctness，输出 9-arm scalar utility / 8 contrasts。`mean=1/5`、
   `variance=99/50`、distinct covariance `99/100`、correlation `1/2` 与 PSD certificate 是 planning marginal；
   不派生 temporal ICC、root-mean `Dbar` covariance 或 576-tuple membership。v4a `Dbar` off-diagonal `1/2` 仅是
   future target constraint，后续 tuple 必须另交 9-arm×8-decision exact witness 或允许的 exact infeasibility proof。

另一个高优先级混淆也已关闭：semantic stratum 用
`(analysis_root_ordinal + reversal_pair_index) mod 4` 对四个 distance bin 做 Latin rotation，每连续四-root block
各出现一次；distance 不再直接标识 stratum。它是独立 P1 收口，不替代上述第 1 个 P0。

terminal status 为 `source_opportunity_preflight_contract_frozen_zero_output_inventory_materializer_not_run`：
`zero_output_preflight_contract_frozen=true` 只表示 repo-level contract/projection 可复验；
`external_publication_anchor_present=false / future_structural_inventory_materialization_authorized=false /
source_structural_inventory_materialized=false / source_opportunity_stage_completed=false / source_preflight_completed=false /
tuple_feasibility_authorized=false / model_output_authorized=false / CUDA_planner_authorized=false / development_authorized=false /
qualification_authorized=false / formal_authorized=false`。source row/text/tape、subject/donor/twin pack、planning atom、
tuple membership、power search/confirmation、baseline/model output、CUDA run 与 empirical outcome count 全为 0，576 tuple
仍 unresolved、selected N 仍 null。因此本 slice 没有形成 Appendable、Readable、Learnable、Steerable 或 integrated
evidence，也不证明真人、产品价值或 production ACTIVE。

实际验证记录：lazy package API / exact CLI source-root smoke、strict protocol load、纯推导 smoke、create-only
`prepare`、独立 `validate-existing` replay、reparse/hardlink rejection、Ruff check 与 Python compile 均已完成；
replay 还严格绑定/复验 v4a、v3 与 power-admission v2 lineage。没有运行 source、model、
CUDA、tuple-feasibility 或 power replicate，也不在本记录预写 pytest 数量。下一步必须先把本 protocol/artifact 绑定
到 external publication anchor，随后才能实现并独立验证**另一个** single-attempt、create-only structural inventory
materializer；materializer 成功与 receipt 齐全后才可进入 tuple feasibility。回滚只需停止
`run_relationship_lab_p4_long_context_source_opportunity_preflight.py` 与后续 consumer；本 protocol、projection、
certificate、manifest 及所有上游 artifact 继续按原字节保留，不得删除后更换 seed、surface registry、twin transform
或 planning generator。

## Slice Relationship P4.7-A0-external-publication-request-v1（2026-08-23）：本地冻结 public Gist request，不发布、不授权

本 slice 继续由唯一 owner `lifeform-evolution.relationship_lab_p4_long_context_causal_campaign` 发布 frozen public
`RelationshipP4LongContextExternalAnchorRequestProtocol` / `RelationshipP4LongContextExternalAnchorRequest`。它只读并
严格复验 source-opportunity preflight、v4a planning、v3 preparation 与 power-admission v2 lineage，在本地生成一份
create-only publication request 和 manifest；不修改上游 source contract，不新增 runtime slot、snapshot writer、
materializer、source/model/session runner、live/product wiring 或 execution consumer。

request protocol 经红队收紧后重新冻结，id 为
`dedfc7ff42f1be0030cdfbe64fd6b1d6dc868adf9db6a9f1150883a9a96a4bee`，raw SHA-256
`38ce85d479c4359c252de8e5293ca1c15d886c5e1757610435aad136feeca8c6`、`16006` bytes。create-only 输出位于
`artifacts/relationship_lab/p4_independent_long_context_external_anchor_request_v1_20260823/`：

- `external_publication_anchor_request.json`：request id
  `7897e3285299eac33385f69fb560a7d68e9f3316fdaf200f27cfa9bbfda489d1`，raw SHA-256
  `0d5147cdf11db9fcaaa793bd9bf9bf8bfb6d07511f2307f5e77cdc0dfd263057`，`12115` bytes；
- `manifest.json`：raw SHA-256
  `17496a50035d5b1dd3849455940ef2bcf4c6c4089dc2093bb194acb14a514125`，`1307` bytes；
- artifact id：`5496fa80bba07c6b2234e0e2ca9293111d7ed6edf0a676ee4f561a7893c22900`。

红队后本地 closure 不再允许“校验一份、执行另一份”或用 relocation 改写 publication identity。derivation helper
先校验 raw pin，再从同一已校验 byte buffer compile/exec；四个 raw-pinned text checkout contract 明确为 v4a/source
helpers `eol=lf`、relationship action owner module/action schema `eol=crlf`。`prepare-request` 必须使用 canonical request
root、五个 canonical publication subject path，以及 source-preflight/v4a-planning/v3-preparation/v2-admission 四个
canonical upstream roots；`validate-request` 只可复验 byte-identical 本地 relocation replica，不能把 relocated path
重绑定成 canonical subject 或 prepare root。所有输入/输出在任何读取前都要求 local default stream；UNC、Windows
device namespace、ADS/非默认 NTFS stream 与既有 symlink/reparse/hardlink/LFS/歧义 relative path 均 fail loudly。

A0 request 固定五个 exact publication subject：source-preflight protocol、纯推导 helper、contract projection、
certificate 与 manifest。未来 publication target 必须是一个新的 public GitHub Gist 第一 revision，且唯一文件的
default-stream bytes 必须精确等于上述 request payload；Gist description 必须为空，所有 observed URL 必须为 HTTPS、
不含 userinfo 且不使用 nondefault port，mutable latest URL 不作 authority。当前 access-controlled/
private origin 明确不接受为 public publication anchor，本地 repo/hash 自洽也不证明匿名读者已从 GitHub 观察到这些
bytes。request 内故意不预填未来 Gist id、revision OID、HTML/raw permalink、receipt 或 admission artifact id，避免
用未知 remote identity 反向影响 source generation。

这个 slice 只完成 A0 request freeze，terminal status 为
`external_publication_anchor_request_frozen_publication_not_observed_no_authority`。精确 firewall 是：
`external_request_dispatched=false / publication_performed=false / external_publication_observed=false /
external_publication_anchor_present=false / external_anchor_admitted=false`；network request、Git commit/push、external
publication 与 receipt count 全为 0。materializer implementation、structural inventory materialization、source
execution、tuple feasibility、power search、model output、CUDA planner、development、qualification、formal 及
Appendable/Readable/Learnable/Steerable/integrated authority 全部为 false；没有 source row/text/tape、subject/donor/
twin pack、planning atom、power replicate、model output、CUDA run 或 empirical outcome。

未来顺序不可压缩：A0 request freeze → 经明确外部权限创建 public Gist first revision → 独立观察者 receipt 与 fresh
unauthenticated online reobservation → A0 admission → materializer implementation → A1 materializer + single-attempt
envelope 的独立 anchor/admission → 唯一 structural inventory materialization attempt。即使 A0 receipt 完整有效，也只
证明某次匿名 GitHub observation 与 frozen request bytes 匹配；它不是 WORM/trusted time，不证明 producer identity，
不能排除 pre-anchor seed grinding，也**不能授权 materialization**。A1 仍是实现 materializer 之后、任何 materialization
之前的强制门，private origin、本地 artifact 或 A0 receipt 均不能替代。

future observer 请求必须不带 `Authorization` header 或 `Cookie`。publisher 创建 Gist 需要另一项用户明确授权；
credential、header、cookie 与 token 永不序列化进 request/receipt。receipt 只有在同一 Gist owner/id、first revision
OID、zero-parent commit、exact-one-entry tree、required filename、mode `100644`、object type `blob`、tree blob OID、
revision-pinned raw final URL、API/HTML identity 及 observed request exact SHA-256/bytes 全部 join 为同一 object graph 时
才可进入 A0 admission；孤立 API 字段、latest URL、单个 raw hash 或匿名 GET 成功都不够。

zero-output count 只统计 persisted/published artifact rows；strict replay 为 exact digest 复算而临时构造的 derivation
objects 不属于 source content 或 persistent materialization，也不改变 materializer/materialization authority=false。

验证记录：本地 CLI `validate-request` 已通过，复验了 request/manifest 的 canonical bytes、自内容寻址 id、上游
lineage 与 ordinary-file closure。A0 定向测试为 `28 passed in 34.29s`；受 verified same-buffer helper loading 直接影响的
v4a planning + source-opportunity preflight 回归为 `32 passed, 1 skipped in 144.33s`，其中 skip 是既有平台条件分支。
owner/CLI/test 的 Ruff format/check 与 `py_compile` 均通过。没有执行 network request、Git commit/push 或 Gist
publication，也没有运行 source materializer、model、CUDA、tuple feasibility 或 power replicate。未运行全仓 pytest，
因为本收敛包只改离线 A0 request owner/contract/CLI 及 helper 加载边界，直接影响面已由上述定向与两组上游回归覆盖。
回滚只需停止
`run_relationship_lab_p4_long_context_external_publication_anchor.py` 与任何后续 consumer，并保留 protocol、request、
manifest 及上游 artifacts 的原始字节。当前没有外部 publication、runtime/live state 或 execution authority 可撤回。
