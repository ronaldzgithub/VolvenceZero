"""Packet D (long-horizon-closure) — HydratableOwnerProtocol contract.

Three owners must satisfy the round-trip invariant:
``hydrate(export()) == export()`` after non-trivial state mutation.
This is the structural guarantee that owner hydration produces no
information loss across a process boundary.

If any of these fail, the corresponding owner's serialiser /
deserialiser disagree on what state matters — a fix is to either
add the missing field to ``export_persistence_snapshot`` or to make
``hydrate_from_persistence`` honour it.

Schema-version bumps and payload corruption / owner mismatch are
covered by ``test_owner_hydration_failures_loud.py``; this test
focuses on the happy round-trip.
"""

from __future__ import annotations

import copy
import json

import pytest

from volvence_zero.owner_hydration import OwnerPersistenceSnapshot


def _preference_action_forecast_codec_fixture(*, with_condition: bool = True):
    from volvence_zero.social_cognition import (
        PreferenceActionForecast,
        RelationshipConditionReadout,
        SocialActionCandidatePrediction,
        SocialActionOutcomeProbability,
    )

    condition = (
        RelationshipConditionReadout(
            condition_label="belonging_erasure",
            confidence=0.8,
            normalized_margin=0.25,
            candidate_scores=(
                ("belonging_erasure", 0.75),
                ("agency_displacement", 0.25),
            ),
            reader_artifact_id="a" * 64,
            source_observation_sha256="b" * 64,
        )
        if with_condition
        else None
    )
    return PreferenceActionForecast(
        forecast_id="forecast-1",
        decision_id="decision-1",
        interlocutor_id="primary",
        candidate_predictions=(
            SocialActionCandidatePrediction(
                action_id="stay_present_without_probe",
                outcomes=(
                    SocialActionOutcomeProbability("helped", 0.7),
                    SocialActionOutcomeProbability("missed", 0.3),
                ),
            ),
            SocialActionCandidatePrediction(
                action_id="respect_space_with_return_option",
                outcomes=(
                    SocialActionOutcomeProbability("helped", 0.4),
                    SocialActionOutcomeProbability("missed", 0.6),
                ),
            ),
        ),
        recommended_action_id="stay_present_without_probe",
        confidence=0.8,
        source_record_ids=("preference-record-1",),
        issued_turn=7,
        evidence=("owner:preference-about-other", "reader:named-condition"),
        session_scope="subject-1",
        condition_readout=condition,
    )


def test_preference_action_forecast_codec_is_lossless_and_freezes_shape() -> None:
    from volvence_zero.social_cognition import (
        preference_action_forecast_from_payload,
        preference_action_forecast_to_payload,
    )

    forecast = _preference_action_forecast_codec_fixture()
    payload = preference_action_forecast_to_payload(forecast)

    assert set(payload) == {
        "forecast_id",
        "decision_id",
        "interlocutor_id",
        "candidate_predictions",
        "recommended_action_id",
        "confidence",
        "source_record_ids",
        "issued_turn",
        "evidence",
        "session_scope",
        "condition_readout",
    }
    assert payload["candidate_predictions"][0] == {
        "action_id": "stay_present_without_probe",
        "outcomes": [
            {"outcome_id": "helped", "probability": 0.7},
            {"outcome_id": "missed", "probability": 0.3},
        ],
    }
    assert payload["condition_readout"] == {
        "condition_label": "belonging_erasure",
        "confidence": 0.8,
        "normalized_margin": 0.25,
        "candidate_scores": [
            {"label": "belonging_erasure", "score": 0.75},
            {"label": "agency_displacement", "score": 0.25},
        ],
        "reader_artifact_id": "a" * 64,
        "source_observation_sha256": "b" * 64,
    }

    json_payload = json.loads(json.dumps(payload, sort_keys=True))
    restored = preference_action_forecast_from_payload(json_payload)
    assert restored == forecast
    assert preference_action_forecast_to_payload(restored) == payload

    legacy_probe = _preference_action_forecast_codec_fixture(
        with_condition=False
    )
    assert preference_action_forecast_from_payload(
        preference_action_forecast_to_payload(legacy_probe)
    ) == legacy_probe


def test_preference_action_forecast_codec_rejects_noncanonical_payloads() -> None:
    from volvence_zero.social_cognition import (
        preference_action_forecast_from_payload,
        preference_action_forecast_to_payload,
    )

    payload = preference_action_forecast_to_payload(
        _preference_action_forecast_codec_fixture()
    )

    with pytest.raises(ValueError, match="fields do not match schema"):
        preference_action_forecast_from_payload({**payload, "unexpected": True})

    tuple_array = {
        **payload,
        "candidate_predictions": tuple(payload["candidate_predictions"]),
    }
    with pytest.raises(TypeError, match="must be an array"):
        preference_action_forecast_from_payload(tuple_array)

    missing_condition_field = copy.deepcopy(payload)
    del missing_condition_field["condition_readout"]["reader_artifact_id"]
    with pytest.raises(ValueError, match="fields do not match schema"):
        preference_action_forecast_from_payload(missing_condition_field)

    boolean_probability = copy.deepcopy(payload)
    boolean_probability["candidate_predictions"][0]["outcomes"][0][
        "probability"
    ] = True
    with pytest.raises(TypeError, match="must be numeric"):
        preference_action_forecast_from_payload(boolean_probability)

    for non_finite in (float("nan"), float("inf"), -float("inf"), 10**10000):
        non_finite_probability = copy.deepcopy(payload)
        non_finite_probability["candidate_predictions"][0]["outcomes"][0][
            "probability"
        ] = non_finite
        with pytest.raises(ValueError, match="must be finite"):
            preference_action_forecast_from_payload(non_finite_probability)


def test_owner_hydration_matrix_freezes_owner_by_owner_decisions() -> None:
    from volvence_zero.owner_hydration_store import OWNER_HYDRATION_MATRIX

    matrix = {entry.owner_name: entry for entry in OWNER_HYDRATION_MATRIX}

    assert matrix["semantic_state"].decision == "hydrate"
    assert matrix["followup_manager"].decision == "hydrate"
    assert matrix["vitals"].decision == "hydrate"
    assert matrix["protocol_registry"].decision == "hydrate"
    assert matrix["social_record_store"].decision == "hydrate"
    assert matrix["prediction_error_heads"].decision == "hydrate"
    assert matrix["dual_track_gate_learner"].decision == "hydrate"
    assert matrix["credit_heads"].decision == "hydrate"
    assert matrix["memory"].decision == "external-owner"
    assert matrix["regime"].decision == "hydrate"
    assert matrix["joint_loop.learning"].decision == "hydrate"
    assert matrix["reflection.consolidation_score"].decision == "hydrate"
    assert matrix["world_temporal"].decision == "explicit-no-hydrate"
    assert matrix["self_temporal"].decision == "explicit-no-hydrate"
    assert len(matrix) == len(OWNER_HYDRATION_MATRIX)
    for entry in OWNER_HYDRATION_MATRIX:
        if entry.decision == "hydrate":
            assert entry.storage_key == f"owner_hydration/{entry.owner_name}"
        else:
            assert entry.reason


def test_semantic_state_store_round_trip() -> None:
    """SemanticStateStore: 9 slots, lifecycle / followup / outcome maps,
    after applying a meaningful set of proposals across multiple slots.
    """
    from volvence_zero.semantic_state.contracts import (
        SemanticProposal,
        SemanticProposalOperation,
    )
    from volvence_zero.semantic_state.store import SemanticStateStore

    source = SemanticStateStore()
    # Commitment: surface, then user pushes back -> BLOCK -> alignment=REJECT
    source.apply(
        slot="commitment",
        proposals=(
            SemanticProposal(
                proposal_id="c-1",
                target_slot="commitment",
                operation=SemanticProposalOperation.ACTIVATE,
                summary="commit to weekly check-in",
                detail="user proposed",
                evidence="user said yes once",
                confidence=0.8,
                control_signal=0.5,
            ),
            SemanticProposal(
                proposal_id="c-1",
                target_slot="commitment",
                operation=SemanticProposalOperation.BLOCK,
                summary="user retracted",
                detail="user later said no",
                evidence="user retracted",
                confidence=0.9,
                control_signal=0.0,
            ),
        ),
        turn_index=3,
    )
    # Open loop: just one OBSERVE
    source.apply(
        slot="open_loop",
        proposals=(
            SemanticProposal(
                proposal_id="ol-1",
                target_slot="open_loop",
                operation=SemanticProposalOperation.OBSERVE,
                summary="user mentioned travel plans",
                detail="follow up next week",
                evidence="user said upcoming trip",
                confidence=0.65,
                control_signal=0.0,
            ),
        ),
        turn_index=5,
    )
    source.apply(
        slot="user_model",
        proposals=(
            SemanticProposal(
                proposal_id="profile-age-1",
                target_slot="user_model",
                operation=SemanticProposalOperation.CREATE,
                summary="self-reported age",
                detail="The user explicitly reported their age.",
                evidence="I am 17.",
                confidence=0.98,
                semantic_key="age",
                canonical_value="17",
            ),
        ),
        turn_index=6,
    )

    exported = source.export_persistence_snapshot()
    assert exported.owner_name == "semantic_state"
    assert exported.schema_version == 3

    target = SemanticStateStore()
    target.hydrate_from_persistence(exported)
    re_exported = target.export_persistence_snapshot()

    assert exported.payload == re_exported.payload, "SemanticStateStore round-trip lost or mutated state."
    # Spot-check that meaningful read accessors return identical results.
    assert target.records_for("commitment") == source.records_for("commitment")
    assert target.lifecycle_for("commitment") == source.lifecycle_for("commitment")
    assert target.records_for("open_loop") == source.records_for("open_loop")
    assert target.records_for("user_model") == source.records_for("user_model")


def test_social_record_store_round_trip_drops_pending_predictions() -> None:
    from volvence_zero.social import SocialRecordStore
    from volvence_zero.social_cognition import (
        CommonGroundAtom,
        OtherMindRecord,
        OtherMindRecordKind,
        OtherMindRecordStatus,
        PreferenceActionOutcomeEvidence,
        PreferenceActionForecast,
        RelationshipConditionReadout,
        SocialActionCandidatePrediction,
        SocialActionOutcomeProbability,
        SocialPrediction,
        SocialPredictionKind,
        SocialPredictionOutcome,
        SocialScopeKind,
        preference_action_forecast_to_payload,
    )
    from volvence_zero.social.record_store import PendingSocialPrediction

    source = SocialRecordStore()
    source.set_tom_records(
        "belief_about_other",
        (
            OtherMindRecord(
                record_id="belief-1",
                interlocutor_id="bob",
                kind=OtherMindRecordKind.BELIEF,
                summary="Bob thinks the plan is risky",
                detail="Bob expressed concern during planning.",
                confidence=0.72,
                status=OtherMindRecordStatus.ACTIVE,
                source_turn=3,
                prediction_error_refs=("spe-1",),
                evidence="turn-3",
            ),
        ),
    )
    source.set_tom_records(
        "preference_about_other",
        (
            OtherMindRecord(
                record_id="preference-1",
                interlocutor_id="bob",
                kind=OtherMindRecordKind.PREFERENCE,
                summary="Bob prefers a low-pressure return option",
                detail="A prior typed outcome supported this preference.",
                confidence=0.76,
                status=OtherMindRecordStatus.ACTIVE,
                source_turn=2,
                prediction_error_refs=(),
                evidence="turn-2",
            ),
        ),
    )
    source.set_preference_action_outcomes(
        (
            PreferenceActionOutcomeEvidence(
                evidence_id="preference-1",
                interlocutor_id="bob",
                observation_summary="Bob was under relationship pressure.",
                action_id="respect_space",
                observed_outcome_id="helped",
                reaction_summary="Bob said having a return option helped.",
                source_turn=2,
                evidence_refs=("dialogue:turn:2",),
            ),
        )
    )
    forecast = PreferenceActionForecast(
        forecast_id="preference-forecast-1",
        decision_id="decision-1",
        interlocutor_id="bob",
        candidate_predictions=(
            SocialActionCandidatePrediction(
                action_id="stay_present",
                outcomes=(
                    SocialActionOutcomeProbability("helped", 0.6),
                    SocialActionOutcomeProbability("missed", 0.4),
                ),
            ),
            SocialActionCandidatePrediction(
                action_id="respect_space",
                outcomes=(
                    SocialActionOutcomeProbability("helped", 0.7),
                    SocialActionOutcomeProbability("missed", 0.3),
                ),
            ),
        ),
        recommended_action_id="respect_space",
        confidence=0.7,
        source_record_ids=("preference-1",),
        issued_turn=2,
        evidence=("owner:preference-about-other",),
        session_scope="session:bob",
        condition_readout=RelationshipConditionReadout(
            condition_label="belonging_erasure",
            confidence=0.8,
            normalized_margin=0.3,
            candidate_scores=(
                ("belonging_erasure", 0.8),
                ("agency_displacement", 0.2),
            ),
            reader_artifact_id="a" * 64,
            source_observation_sha256="b" * 64,
        ),
    )
    source.set_preference_action_forecasts((forecast,))
    source.set_pending_tom_predictions(
        "belief_about_other",
        (
            PendingSocialPrediction(
                prediction=SocialPrediction(
                    prediction_id="pending-belief-1",
                    kind=SocialPredictionKind.BELIEF_ABOUT_OTHER,
                    scope_kind=SocialScopeKind.INTERLOCUTOR,
                    scope_id="bob",
                    subject_ids=("bob",),
                    audience_ids=("self",),
                    predicted_outcome="Bob will still think the plan is risky",
                    confidence=0.7,
                    evidence=("turn-3",),
                ),
                source_record_id="belief-1",
                issued_turn=3,
            ),
        ),
    )
    source.set_common_ground_atoms(
        dyad_atoms=(
            CommonGroundAtom(
                atom_id="cg-1",
                scope_id="self+bob",
                scope_kind=SocialScopeKind.DYAD,
                summary="We agreed to revisit the plan tomorrow",
                recursion_depth=0,
                confidence=0.8,
                accepted_by_ids=("self", "bob"),
                evidence=("turn-4",),
            ),
        ),
        group_atoms=(),
    )
    source.record_group_regime("frame-group:alice+bob+cara", "problem_solving")
    source.apply_group_settlement(
        "frame-group:alice+bob+cara",
        outcome=SocialPredictionOutcome.CONFIRMED,
    )

    exported = source.export_persistence_snapshot()
    assert exported.owner_name == "social_record_store"
    assert exported.schema_version == 4
    assert exported.payload["preference_action_forecasts"] == [
        preference_action_forecast_to_payload(forecast)
    ]

    target = SocialRecordStore()
    target.hydrate_from_persistence(exported)
    re_exported = target.export_persistence_snapshot()

    assert exported.payload == re_exported.payload
    assert target.tom_records("belief_about_other") == source.tom_records("belief_about_other")
    assert target.preference_action_outcomes == source.preference_action_outcomes
    assert target.preference_action_forecasts == (forecast,)
    assert target.common_ground_dyad_atoms == source.common_ground_dyad_atoms
    assert target.group_regime_for("frame-group:alice+bob+cara") == "problem_solving"
    assert target.group_durability_for("frame-group:alice+bob+cara") > 0.5
    assert target.pending_tom_predictions("belief_about_other") == ()


def test_social_record_store_explicitly_adapts_legacy_forecast_payload() -> None:
    from volvence_zero.social import SocialRecordStore

    legacy_forecast = {
        "forecast_id": "legacy-forecast-1",
        "decision_id": "legacy-decision-1",
        "interlocutor_id": "bob",
        "candidate_predictions": (
            {
                "action_id": "stay_present",
                "outcomes": (
                    {"outcome_id": "helped", "probability": "0.7"},
                    {"outcome_id": "missed", "probability": "0.3"},
                ),
            },
            {
                "action_id": "respect_space",
                "outcomes": (
                    {"outcome_id": "helped", "probability": "0.4"},
                    {"outcome_id": "missed", "probability": "0.6"},
                ),
            },
        ),
        "recommended_action_id": "stay_present",
        "confidence": "0.7",
        "source_record_ids": ("legacy-record-1",),
        "issued_turn": "3",
        "evidence": ("legacy-owner-evidence",),
    }
    store = SocialRecordStore()
    store.hydrate_from_persistence(
        OwnerPersistenceSnapshot(
            owner_name="social_record_store",
            schema_version=1,
            payload={"preference_action_forecasts": [legacy_forecast]},
        )
    )

    restored = store.preference_action_forecasts[0]
    assert restored.forecast_id == "legacy-forecast-1"
    assert restored.session_scope == ""
    assert restored.condition_readout is None
    assert restored.candidate_predictions[0].outcomes[0].probability == 0.7


def test_social_record_store_schema_v4_delegates_to_strict_forecast_codec() -> None:
    from volvence_zero.owner_hydration import HydrationPayloadInvalidError
    from volvence_zero.social import SocialRecordStore
    from volvence_zero.social_cognition import (
        preference_action_forecast_to_payload,
    )

    forecast_payload = preference_action_forecast_to_payload(
        _preference_action_forecast_codec_fixture()
    )
    forecast_payload["unexpected"] = True

    with pytest.raises(
        HydrationPayloadInvalidError,
        match="fields do not match schema",
    ):
        SocialRecordStore().hydrate_from_persistence(
            OwnerPersistenceSnapshot(
                owner_name="social_record_store",
                schema_version=4,
                payload={
                    "preference_action_forecasts": [forecast_payload],
                    "preference_action_outcome_mutation_receipts": [],
                },
            )
        )


def test_regime_module_round_trip() -> None:
    from volvence_zero.regime import RegimeCheckpoint, RegimeModule

    source = RegimeModule()
    source.restore_checkpoint(
        RegimeCheckpoint(
            checkpoint_id="regime-test",
            historical_effectiveness=(("problem_solving", 0.77),),
            strategy_priors=(("problem_solving", 0.11),),
            active_regime_id="problem_solving",
            previous_regime_id="guided_exploration",
            turns_in_current_regime=4,
            turn_index=9,
            regime_sequence=("guided_exploration", "problem_solving"),
            attribution_horizons=(2, 4),
            selection_weights=(("problem_solving", 1.05),),
            feature_weights=(("problem_solving", (("task_pressure", 0.2),)),),
            external_outcome_scores=(("helped", 0.88),),
            learned_score_weights=(("problem_solving", (0.1, 0.0, -0.1, 0.02)),),
            learned_score_update_count=3,
            learned_score_abs_error_sum=0.4,
            learned_score_baseline_abs_error_sum=0.7,
            learned_score_settled_count=3,
            learned_score_last_target_regime_id="problem_solving",
        )
    )

    exported = source.export_persistence_snapshot()
    assert exported.owner_name == "regime"
    assert exported.schema_version == 1

    target = RegimeModule()
    target.hydrate_from_persistence(exported)
    re_exported = target.export_persistence_snapshot()

    assert exported.payload == re_exported.payload


def test_prediction_error_heads_round_trip() -> None:
    from volvence_zero.prediction.error import PredictionErrorModule

    source = PredictionErrorModule()
    exported = source.export_persistence_snapshot()
    assert exported.owner_name == "prediction_error_heads"
    assert exported.schema_version == 1

    target = PredictionErrorModule()
    target.hydrate_from_persistence(exported)
    re_exported = target.export_persistence_snapshot()

    assert exported.payload == re_exported.payload


def test_credit_module_learned_heads_round_trip() -> None:
    from volvence_zero.credit.gate import (
        CreditModule,
        GateRiskLearnerState,
        RewardingStateHeadState,
    )

    source = CreditModule()
    source.ledger.restore_rewarding_state_head(
        RewardingStateHeadState(
            rule_id="credit.rewarding_state_head.v1",
            feature_dim=15,
            update_count=6,
            weights=tuple(0.01 * i for i in range(15)),
            bias=0.2,
            last_prediction=0.55,
            last_target=0.6,
            last_validation_delta=0.03,
            last_capacity_cost=0.02,
            last_rollback_evidence="credit-rewarding-state:1:6",
        )
    )
    source.ledger.restore_gate_risk_learner(
        GateRiskLearnerState(
            weights=tuple(0.05 * i for i in range(len(source.ledger.export_gate_risk_learner().weights))),
            update_count=9,
            abs_error_sum=2.1,
            agreement_count=7,
        )
    )

    exported = source.export_persistence_snapshot()
    assert exported.owner_name == "credit_heads"
    assert exported.schema_version == 1

    target = CreditModule()
    target.hydrate_from_persistence(exported)
    re_exported = target.export_persistence_snapshot()

    assert exported.payload == re_exported.payload


def test_dual_track_gate_learner_round_trip() -> None:
    from volvence_zero.dual_track.gate_learner import (
        DualTrackGateLearner,
        DualTrackGateLearnerState,
    )

    source = DualTrackGateLearner()
    source.restore_state(
        DualTrackGateLearnerState(
            weights=(0.1, -0.1, 0.2, 0.0, 0.05, 0.55),
            update_count=7,
            abs_error_sum=1.2,
            heuristic_abs_error_sum=1.6,
            settled_comparison_count=6,
        )
    )
    exported = source.export_persistence_snapshot()
    assert exported.owner_name == "dual_track_gate_learner"

    target = DualTrackGateLearner()
    target.hydrate_from_persistence(exported)
    re_exported = target.export_persistence_snapshot()

    assert exported.payload == re_exported.payload


def test_reflection_consolidation_learner_round_trip() -> None:
    from volvence_zero.reflection.consolidation_learner import (
        ConsolidationScoreLearner,
        ConsolidationScoreLearnerState,
    )

    source = ConsolidationScoreLearner()
    source.restore_state(
        ConsolidationScoreLearnerState(
            weights=(0.1, -0.1, 0.2, 0.0, 0.05, 0.15, -0.2),
            settled_count=7,
            abs_error_sum=1.2,
            baseline_abs_error_sum=1.6,
        )
    )
    exported = source.export_persistence_snapshot()
    assert exported.owner_name == "reflection.consolidation_score"

    target = ConsolidationScoreLearner()
    target.hydrate_from_persistence(exported)
    re_exported = target.export_persistence_snapshot()

    assert exported.payload == re_exported.payload


def test_owner_hydration_seed_once_does_not_overwrite_existing_payload() -> None:
    from volvence_zero.brain import _seed_owner_hydration_snapshots_once
    from volvence_zero.memory import InMemoryPersistenceBackend

    backend = InMemoryPersistenceBackend()
    existing = OwnerPersistenceSnapshot(
        owner_name="semantic_state",
        schema_version=1,
        payload={"records": {"relationship_state": [{"summary": "existing"}]}},
        description="existing",
    )
    seed = OwnerPersistenceSnapshot(
        owner_name="semantic_state",
        schema_version=1,
        payload={"records": {"relationship_state": [{"summary": "template"}]}},
        description="template",
    )

    _seed_owner_hydration_snapshots_once(
        backend=backend,
        snapshots=(existing,),
    )
    _seed_owner_hydration_snapshots_once(
        backend=backend,
        snapshots=(seed,),
    )

    loaded = backend.load_checkpoint(key="owner_hydration/semantic_state")
    assert loaded is not None
    assert b"existing" in loaded[0]
    assert b"template" not in loaded[0]


def test_followup_manager_round_trip() -> None:
    from lifeform_core.followup_manager import FollowupManager

    source = FollowupManager(default_due_delay_ticks=60, max_pending=8)
    source.ingest_open_loops(
        unresolved_loops=("loop-a", "loop-b", "loop-c"),
        current_tick=10,
    )
    source.ingest_at_risk_commitments(
        at_risk_refs=("commit-1",),
        current_tick=15,
    )

    exported = source.export_persistence_snapshot()
    assert exported.owner_name == "followup_manager"
    assert exported.schema_version == 1

    target = FollowupManager(default_due_delay_ticks=60, max_pending=8)
    target.hydrate_from_persistence(exported)
    re_exported = target.export_persistence_snapshot()

    assert exported.payload == re_exported.payload, "FollowupManager round-trip lost or mutated state."
    assert target.pending == source.pending


def test_vitals_module_round_trip() -> None:
    from lifeform_core.vitals import (
        DriveSpec,
        VitalsBootstrap,
        VitalsModule,
    )
    from lifeform_core.types import TickEvent, TickKind

    bootstrap = VitalsBootstrap(
        schema_version=1,
        drives=(
            DriveSpec(
                name="bond_warmth",
                target=0.7,
                homeostatic_band=(0.5, 0.85),
                decay_per_tick=0.005,
                pe_weight=1.0,
                initial_level=0.6,
                recharge_per_turn=0.02,
            ),
            DriveSpec(
                name="user_engagement",
                target=0.6,
                homeostatic_band=(0.3, 0.85),
                decay_per_tick=0.02,
                pe_weight=0.5,
                initial_level=0.5,
                recharge_per_turn=0.3,
            ),
        ),
        proactive_pe_threshold=1.0,
        proactive_followup_priority=0.5,
        proactive_cooldown_ticks=60,
    )
    source = VitalsModule(bootstrap)
    # Drive the levels somewhere non-trivial via several SYSTEM ticks.
    for i in range(10):
        source.on_tick(
            TickEvent(
                tick_index=i + 1,
                kind=TickKind.SYSTEM,
                elapsed_seconds=1.0,
            )
        )
    source.on_turn(regime="emotional_support", user_input_present=True)
    # Force a proactive crossing to populate _last_proactive_at.
    source.consider_proactive_followup(current_tick=20)

    exported = source.export_persistence_snapshot()
    assert exported.owner_name == "vitals"
    assert exported.schema_version == 1

    target = VitalsModule(bootstrap)
    target.hydrate_from_persistence(exported)
    re_exported = target.export_persistence_snapshot()

    assert exported.payload == re_exported.payload, "VitalsModule round-trip lost or mutated state."
    # Snapshot equality on observable fields.
    s = source.current_snapshot()
    t = target.current_snapshot()
    assert s.drive_levels == t.drive_levels
    assert s.tick_index == t.tick_index
    assert s.last_proactive_at_tick == t.last_proactive_at_tick


def test_round_trip_is_idempotent_on_double_hydrate() -> None:
    """Applying the same snapshot twice must produce the same store
    state (no accumulating effects).
    """
    from volvence_zero.semantic_state.contracts import (
        SemanticProposal,
        SemanticProposalOperation,
    )
    from volvence_zero.semantic_state.store import SemanticStateStore

    source = SemanticStateStore()
    source.apply(
        slot="open_loop",
        proposals=(
            SemanticProposal(
                proposal_id="ol-x",
                target_slot="open_loop",
                operation=SemanticProposalOperation.OBSERVE,
                summary="x",
                detail="",
                evidence="",
                confidence=0.5,
                control_signal=0.0,
            ),
        ),
        turn_index=1,
    )
    snap = source.export_persistence_snapshot()

    target = SemanticStateStore()
    target.hydrate_from_persistence(snap)
    after_first = target.export_persistence_snapshot()
    target.hydrate_from_persistence(snap)
    after_second = target.export_persistence_snapshot()

    assert after_first.payload == after_second.payload, "Re-hydrating with the same snapshot must be idempotent."


def test_owner_persistence_snapshot_validates_required_fields() -> None:
    """Frozen dataclass validation: empty owner_name, version 0, or
    non-Mapping payload must raise.
    """
    import pytest

    with pytest.raises(ValueError, match="owner_name"):
        OwnerPersistenceSnapshot(owner_name="", schema_version=1, payload={})
    with pytest.raises(ValueError, match="schema_version"):
        OwnerPersistenceSnapshot(owner_name="x", schema_version=0, payload={})
    with pytest.raises(TypeError, match="Mapping"):
        OwnerPersistenceSnapshot(
            owner_name="x",
            schema_version=1,
            payload="not a mapping",  # type: ignore[arg-type]
        )
