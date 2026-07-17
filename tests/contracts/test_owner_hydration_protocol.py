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

from volvence_zero.owner_hydration import OwnerPersistenceSnapshot


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

    exported = source.export_persistence_snapshot()
    assert exported.owner_name == "semantic_state"
    assert exported.schema_version == 1

    target = SemanticStateStore()
    target.hydrate_from_persistence(exported)
    re_exported = target.export_persistence_snapshot()

    assert exported.payload == re_exported.payload, (
        "SemanticStateStore round-trip lost or mutated state."
    )
    # Spot-check that meaningful read accessors return identical results.
    assert target.records_for("commitment") == source.records_for("commitment")
    assert target.lifecycle_for("commitment") == source.lifecycle_for("commitment")
    assert target.records_for("open_loop") == source.records_for("open_loop")


def test_social_record_store_round_trip_drops_pending_predictions() -> None:
    from volvence_zero.social import SocialRecordStore
    from volvence_zero.social_cognition import (
        CommonGroundAtom,
        OtherMindRecord,
        OtherMindRecordKind,
        OtherMindRecordStatus,
        SocialPrediction,
        SocialPredictionKind,
        SocialPredictionOutcome,
        SocialScopeKind,
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
    assert exported.schema_version == 1

    target = SocialRecordStore()
    target.hydrate_from_persistence(exported)
    re_exported = target.export_persistence_snapshot()

    assert exported.payload == re_exported.payload
    assert target.tom_records("belief_about_other") == source.tom_records("belief_about_other")
    assert target.common_ground_dyad_atoms == source.common_ground_dyad_atoms
    assert (
        target.group_regime_for("frame-group:alice+bob+cara")
        == "problem_solving"
    )
    assert target.group_durability_for("frame-group:alice+bob+cara") > 0.5
    assert target.pending_tom_predictions("belief_about_other") == ()


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

    assert exported.payload == re_exported.payload, (
        "FollowupManager round-trip lost or mutated state."
    )
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

    assert exported.payload == re_exported.payload, (
        "VitalsModule round-trip lost or mutated state."
    )
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

    assert after_first.payload == after_second.payload, (
        "Re-hydrating with the same snapshot must be idempotent."
    )


def test_owner_persistence_snapshot_validates_required_fields() -> None:
    """Frozen dataclass validation: empty owner_name, version 0, or
    non-Mapping payload must raise.
    """
    import pytest

    with pytest.raises(ValueError, match="owner_name"):
        OwnerPersistenceSnapshot(
            owner_name="", schema_version=1, payload={}
        )
    with pytest.raises(ValueError, match="schema_version"):
        OwnerPersistenceSnapshot(
            owner_name="x", schema_version=0, payload={}
        )
    with pytest.raises(TypeError, match="Mapping"):
        OwnerPersistenceSnapshot(
            owner_name="x", schema_version=1, payload="not a mapping"  # type: ignore[arg-type]
        )
