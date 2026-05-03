# Contract Migration Log

> Status: migration / implementation log
> Last updated: 2026-05-03

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
