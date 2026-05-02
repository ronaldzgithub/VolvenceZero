# Contract Migration Log

> Status: migration / implementation log
> Last updated: 2026-05-02

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
