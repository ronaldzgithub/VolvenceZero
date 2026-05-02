"""R16 Slice 11: self-emitted memory-visibility prediction → PE → credit loop.

These tests prove the first end-to-end social cognition learning loop
without any test-side injection of ``SocialPrediction`` /
``SocialPredictionError``. The loop runs entirely from a multi-party
``EnvironmentEvent`` plus pre-existing memory entries:

1. ``MultiPartyIdentityModule`` reads the EnvironmentEvent frame and
   publishes a non-default identity scope (e.g. ``alice``).
2. ``MemoryModule`` filters retrieval by active subject scope and surfaces
   ``suppressed_cross_scope_entries`` whenever stored memory belongs to a
   different subject (e.g. ``bob``).
3. ``SocialPredictionAggregateModule`` emits a
   ``MEMORY_VISIBILITY`` :class:`SocialPrediction` (no injection).
4. ``SocialPredictionErrorModule`` derives a matching
   ``DISCONFIRMED`` :class:`SocialPredictionError` referencing the
   prediction id (no injection).
5. ``derive_social_prediction_error_credit_records`` writes a
   negative ``CreditRecord`` to the credit ledger via the existing
   final-wiring path.

These properties together close the SHADOW → live ACTIVE-content gap
identified in the R16 reflection.
"""

from __future__ import annotations

import asyncio

from volvence_zero.environment import (
    EnvironmentActorRef,
    EnvironmentEvent,
    EnvironmentEventKind,
    EnvironmentFrame,
)
from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.runtime import WiringLevel
from volvence_zero.memory import (
    MemoryStore,
    MemoryStratum,
    MemoryWriteRequest,
    Track,
)
from volvence_zero.social_cognition import (
    PRIMARY_INTERLOCUTOR_ID,
    SELF_INTERLOCUTOR_ID,
    MultiPartyIdentitySnapshot,
    SocialPredictionErrorSnapshot,
    SocialPredictionKind,
    SocialPredictionOutcome,
    SocialPredictionSnapshot,
    SocialScopeKind,
)
from volvence_zero.substrate import (
    FeatureSignal,
    FeatureSurfaceSubstrateAdapter,
)


_BOB_MEMORY_CONTENT = (
    "bob prefers careful planning before making any product decision"
)
_BOB_MEMORY_QUERY_HINT = "careful planning product decision"


def _alice_environment_event() -> EnvironmentEvent:
    return EnvironmentEvent(
        event_id="env-r16-slice11-alice",
        event_kind=EnvironmentEventKind.USER_INPUT,
        trigger_kind="user_input",
        frame=EnvironmentFrame(
            actor=EnvironmentActorRef(actor_id="alice"),
            active_speaker_id="alice",
            addressee_ids=(SELF_INTERLOCUTOR_ID,),
            subject_ids=("alice",),
            audience_ids=(SELF_INTERLOCUTOR_ID, "alice"),
        ),
        scene_id="scene-r16-slice11",
        timestamp_ms=10,
        provenance="r16-slice11-test",
        payload_summary="alice asks for careful planning advice in product context.",
    )


def _seed_bob_memory() -> MemoryStore:
    store = MemoryStore()
    store.write(
        MemoryWriteRequest(
            content=_BOB_MEMORY_CONTENT,
            track=Track.SHARED,
            stratum=MemoryStratum.DURABLE,
            tags=("preference", "planning"),
            strength=0.85,
            subject_ids=("bob",),
            audience_ids=(SELF_INTERLOCUTOR_ID, "bob"),
        ),
        timestamp_ms=1,
    )
    return store


def _multi_party_active_config() -> FinalRolloutConfig:
    """R16 Slice 11 widening: promote identity + social prediction loop to
    ACTIVE so the audience-scope filter and self-emitted PE flow through
    to active consumers (memory, credit). Other social cognition slots
    (ToM, role, common ground, group) remain SHADOW for this slice.
    """
    return FinalRolloutConfig(
        multi_party_identity=WiringLevel.ACTIVE,
        social_prediction=WiringLevel.ACTIVE,
        social_prediction_error=WiringLevel.ACTIVE,
    )


def _run_alice_turn(*, store: MemoryStore):
    return asyncio.run(
        run_final_wiring_turn(
            config=_multi_party_active_config(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="r16-slice11-substrate",
                feature_surface=(
                    FeatureSignal(
                        name="alice_planning_context",
                        values=(0.7,),
                        source="adapter",
                    ),
                ),
            ),
            user_input=_BOB_MEMORY_QUERY_HINT,
            memory_store=store,
            environment_event=_alice_environment_event(),
            session_id="r16-slice11-session",
            wave_id="r16-slice11-wave",
        )
    )


def test_memory_module_suppresses_cross_scope_entries_in_alice_context():
    """The active scope from ``EnvironmentEvent`` filters retrieval and
    surfaces the suppressed entry without any planner / renderer change.
    """
    store = _seed_bob_memory()
    result = _run_alice_turn(store=store)

    memory_snapshot = result.active_snapshots["memory"]
    memory_value = memory_snapshot.value
    assert memory_value.active_subject_scope == ("alice",)
    suppressed_contents = tuple(
        entry.content for entry in memory_value.suppressed_cross_scope_entries
    )
    retrieved_contents = tuple(
        entry.content for entry in memory_value.retrieved_entries
    )
    assert _BOB_MEMORY_CONTENT in suppressed_contents
    assert _BOB_MEMORY_CONTENT not in retrieved_contents


def test_social_prediction_self_emits_memory_visibility_prediction():
    """``SocialPredictionAggregateModule`` self-emits a
    ``MEMORY_VISIBILITY`` :class:`SocialPrediction` for the active scope
    when the multi-party identity is non-default.  No injection used.
    """
    store = _seed_bob_memory()
    result = _run_alice_turn(store=store)

    prediction_snapshot = result.active_snapshots["social_prediction"]
    prediction_value = prediction_snapshot.value
    assert isinstance(prediction_value, SocialPredictionSnapshot)
    assert len(prediction_value.predictions) == 1

    prediction = prediction_value.predictions[0]
    assert prediction.kind is SocialPredictionKind.MEMORY_VISIBILITY
    assert prediction.scope_kind is SocialScopeKind.INTERLOCUTOR
    assert prediction.scope_id == "alice"
    assert prediction.subject_ids == ("alice",)
    assert prediction.predicted_outcome == "memory_subjects_match_active_subjects"
    assert prediction.prediction_id.startswith("memory_visibility:alice:v")


def test_social_prediction_error_self_derives_misattribution_disconfirmation():
    """``SocialPredictionErrorModule`` self-derives a ``DISCONFIRMED``
    :class:`SocialPredictionError` for the suppressed cross-scope entry.
    The error references the same ``prediction_id`` emitted by the
    aggregator and is *not* sourced from ``pending_errors``.
    """
    store = _seed_bob_memory()
    result = _run_alice_turn(store=store)

    error_snapshot = result.active_snapshots["social_prediction_error"]
    error_value = error_snapshot.value
    assert isinstance(error_value, SocialPredictionErrorSnapshot)
    assert len(error_value.errors) == 1

    error = error_value.errors[0]
    assert error.kind is SocialPredictionKind.MEMORY_VISIBILITY
    assert error.outcome is SocialPredictionOutcome.DISCONFIRMED
    assert error.scope_kind is SocialScopeKind.INTERLOCUTOR
    assert error.scope_id == "alice"
    assert error.owner == "MemoryModule"
    assert error.magnitude > 0.0

    prediction_value = result.active_snapshots["social_prediction"].value
    prediction_id = prediction_value.predictions[0].prediction_id
    assert error.prediction_id == prediction_id
    assert any("subject=bob" in evidence for evidence in error.evidence)
    assert "memory_visibility_pe=1" in error_value.description


def test_self_derived_memory_visibility_pe_records_negative_credit():
    """The self-derived social PE flows into the credit ledger via the
    existing :func:`derive_social_prediction_error_credit_records` path.
    No manual ``social_prediction_errors`` injection.
    """
    store = _seed_bob_memory()
    result = _run_alice_turn(store=store)

    credit_snapshot = result.active_snapshots["credit"].value
    matching_credits = tuple(
        record
        for record in credit_snapshot.recent_credits
        if record.source_event == "social_pe:memory_visibility"
    )
    assert matching_credits, "expected a memory_visibility social PE credit record"
    for record in matching_credits:
        assert record.level == "social_prediction_error"
        assert record.credit_value < 0.0
        assert "scope=interlocutor:alice" in record.context
        assert "outcome=disconfirmed" in record.context


def test_default_identity_does_not_self_emit_memory_visibility_prediction():
    """Sanity gate: when no EnvironmentEvent is provided, the multi-party
    identity falls back to ``primary`` and the loop emits no predictions
    nor PE.  This protects existing single-party companion behaviour.
    """
    store = _seed_bob_memory()
    result = asyncio.run(
        run_final_wiring_turn(
            config=FinalRolloutConfig(),
            substrate_adapter=FeatureSurfaceSubstrateAdapter(
                model_id="r16-slice11-default",
                feature_surface=(
                    FeatureSignal(
                        name="default_planning_context",
                        values=(0.5,),
                        source="adapter",
                    ),
                ),
            ),
            user_input=_BOB_MEMORY_QUERY_HINT,
            memory_store=store,
            session_id="r16-slice11-default-session",
            wave_id="r16-slice11-default-wave",
        )
    )

    identity_snapshot = result.active_snapshots["multi_party_identity"]
    assert isinstance(identity_snapshot.value, MultiPartyIdentitySnapshot)
    assert identity_snapshot.value.subject_ids == (PRIMARY_INTERLOCUTOR_ID,)

    prediction_value = result.shadow_snapshots["social_prediction"].value
    error_value = result.shadow_snapshots["social_prediction_error"].value
    assert prediction_value.predictions == ()
    assert error_value.errors == ()
