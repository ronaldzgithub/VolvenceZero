"""Unit tests for ``volvence_zero.interlocutor`` (Gap 9 slice 1).

Covers:

* ``InterlocutorState`` invariants: 11 axes in [0,1], signed
  ``trust_signal`` in [-1,1], cold default matches spec.
* ``InterlocutorReadoutContext.evidence_score()`` scales with
  how many ``has_*`` flags are set.
* ``readout_interlocutor_state`` per-axis response to feature
  changes \u2014 the 12 axes each move in the expected direction
  under a representative profile.
* Cold-start: ``evidence_score=0`` produces a state that sits
  tight around the neutral baseline with low
  ``readout_confidence``.
* Duck-typed ``build_interlocutor_readout_context_from_snapshots``
  handles None inputs (all flags off) and real kernel
  ``RegimeSnapshot`` / ``DualTrackSnapshot`` / ``EvaluationSnapshot``
  / ``PredictionErrorSnapshot`` / ``MemorySnapshot`` /
  ``CommitmentSnapshot`` shapes.
* Commitment alignment trend: recent reject \u2192 resistance up;
  recent agree \u2192 openness + trust up.
"""

from __future__ import annotations

import pytest

from volvence_zero.interlocutor import (
    InterlocutorReadoutContext,
    InterlocutorState,
    build_interlocutor_readout_context_from_snapshots,
    readout_interlocutor_state,
)


# ---------------------------------------------------------------------------
# InterlocutorState invariants
# ---------------------------------------------------------------------------


def test_default_state_is_neutral_with_low_confidence() -> None:
    state = InterlocutorState()
    assert state.engagement_intensity == 0.5
    assert state.self_disclosure_level == 0.5
    assert state.task_focus_level == 0.5
    assert state.trust_signal == 0.0
    assert state.readout_confidence == 0.1
    assert state.rationale == ""


def test_state_rejects_out_of_range_unsigned_axis() -> None:
    with pytest.raises(ValueError, match="engagement_intensity"):
        InterlocutorState(engagement_intensity=1.5)


def test_state_rejects_out_of_range_trust_signal() -> None:
    with pytest.raises(ValueError, match="trust_signal"):
        InterlocutorState(trust_signal=1.5)


def test_state_accepts_boundary_values() -> None:
    # Edge values must be allowed.
    state = InterlocutorState(
        engagement_intensity=0.0,
        resistance_level=1.0,
        trust_signal=-1.0,
        readout_confidence=0.0,
    )
    assert state.engagement_intensity == 0.0
    assert state.trust_signal == -1.0


# ---------------------------------------------------------------------------
# Evidence score
# ---------------------------------------------------------------------------


def test_evidence_score_cold_start() -> None:
    ctx = InterlocutorReadoutContext()
    assert ctx.evidence_score() == pytest.approx(0.10, abs=1e-4)


def test_evidence_score_full_signal() -> None:
    ctx = InterlocutorReadoutContext(
        has_dual_track=True,
        has_evaluation=True,
        has_prediction_error=True,
        has_memory=True,
        has_commitment=True,
    )
    # 0.10 + 0.30 + 0.20 + 0.15 + 0.10 + 0.10 = 0.95
    assert ctx.evidence_score() == pytest.approx(0.95, abs=1e-4)


# ---------------------------------------------------------------------------
# Per-axis response
# ---------------------------------------------------------------------------


def test_engagement_intensity_rises_with_cross_track_tension() -> None:
    calm = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True, cross_track_tension=0.0
    )
    tense = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True, cross_track_tension=0.9
    )
    assert readout_interlocutor_state(tense).engagement_intensity > (
        readout_interlocutor_state(calm).engagement_intensity
    )


def test_self_disclosure_rises_with_self_presence_and_warmth() -> None:
    absent = InterlocutorReadoutContext(
        has_dual_track=True, has_memory=True, has_evaluation=True,
        self_presence=0.0, warmth=0.2,
    )
    present = InterlocutorReadoutContext(
        has_dual_track=True, has_memory=True, has_evaluation=True,
        self_presence=1.0, warmth=0.9, self_drive=0.6,
    )
    assert readout_interlocutor_state(present).self_disclosure_level > (
        readout_interlocutor_state(absent).self_disclosure_level
    )


def test_task_focus_rises_with_task_bias_and_world_drive() -> None:
    chat = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True,
        task_bias=0.0, world_drive=0.0, task_pressure=0.2,
    )
    work = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True,
        task_bias=1.0, world_drive=0.8, task_pressure=0.8, world_presence=0.7,
    )
    assert readout_interlocutor_state(work).task_focus_level > (
        readout_interlocutor_state(chat).task_focus_level
    )


def test_emotional_weight_rises_under_self_tension_and_low_warmth() -> None:
    baseline = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True,
        self_tension=0.1, warmth=0.7, cross_track_tension=0.1,
    )
    distressed = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True, has_prediction_error=True,
        self_tension=0.9, warmth=0.2, cross_track_tension=0.6,
        repair_bias=0.5, pe_magnitude=0.7,
    )
    assert readout_interlocutor_state(distressed).emotional_weight > (
        readout_interlocutor_state(baseline).emotional_weight
    )


def test_cognitive_engagement_rises_with_exploration_and_shared_drive() -> None:
    dormant = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True,
        exploration_bias=0.0, shared_drive=0.0,
    )
    active = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True,
        exploration_bias=0.9, shared_drive=0.8, info_integration=0.8,
        switch_pressure=0.6,
    )
    assert readout_interlocutor_state(active).cognitive_engagement > (
        readout_interlocutor_state(dormant).cognitive_engagement
    )


def test_resistance_rises_with_pe_and_reject_trend() -> None:
    calm = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True, has_prediction_error=True,
        has_commitment=True,
        pe_magnitude=0.1, cross_track_tension=0.1,
        commitment_alignment_trend=0.5,
        cross_track_stability=0.8, warmth=0.7,
    )
    pushback = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True, has_prediction_error=True,
        has_commitment=True,
        pe_magnitude=0.8, cross_track_tension=0.7,
        commitment_alignment_trend=-0.8,
        cross_track_stability=0.2, warmth=0.2,
    )
    assert (
        readout_interlocutor_state(pushback).resistance_level
        > readout_interlocutor_state(calm).resistance_level
    )


def test_openness_rises_with_warmth_and_agree_trend() -> None:
    closed = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True, has_commitment=True,
        warmth=0.2, cross_track_stability=0.3,
        commitment_alignment_trend=-0.5,
    )
    receptive = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True, has_commitment=True,
        has_prediction_error=True,
        warmth=0.9, cross_track_stability=0.9,
        commitment_alignment_trend=0.9, support_presence=0.7,
        pe_signed_reward=0.5,
    )
    assert readout_interlocutor_state(receptive).openness_to_guidance > (
        readout_interlocutor_state(closed).openness_to_guidance
    )


def test_directness_rises_with_task_bias_falls_with_repair_bias() -> None:
    indirect = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True,
        task_bias=0.0, world_drive=0.1,
        repair_bias=0.8, warmth=0.8, self_tension=0.7,
    )
    direct = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True,
        task_bias=0.9, world_drive=0.8, task_pressure=0.7,
        repair_bias=0.0, warmth=0.3, self_tension=0.1,
    )
    assert (
        readout_interlocutor_state(direct).directness
        > readout_interlocutor_state(indirect).directness
    )


def test_trust_signal_positive_for_rising_reward() -> None:
    ctx = InterlocutorReadoutContext(
        has_dual_track=True, has_prediction_error=True, has_commitment=True,
        pe_signed_reward=0.8,
        commitment_alignment_trend=0.8,
        warmth=0.7,
    )
    assert readout_interlocutor_state(ctx).trust_signal > 0.3


def test_trust_signal_negative_for_ruptured_context() -> None:
    ctx = InterlocutorReadoutContext(
        has_dual_track=True, has_prediction_error=True, has_commitment=True,
        pe_signed_reward=-0.8,
        commitment_alignment_trend=-0.9,
        cross_track_tension=0.8,
        pe_relationship_error=-0.7,
    )
    assert readout_interlocutor_state(ctx).trust_signal < -0.2


def test_stability_rises_with_cross_track_stability_and_info_integration() -> None:
    chaotic = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True, has_prediction_error=True,
        cross_track_stability=0.1, switch_pressure=0.9,
        cross_track_tension=0.8, pe_magnitude=0.7,
    )
    steady = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True, has_prediction_error=True,
        cross_track_stability=0.9, info_integration=0.8,
        switch_pressure=0.0, cross_track_tension=0.1, pe_magnitude=0.05,
    )
    assert readout_interlocutor_state(steady).stability > (
        readout_interlocutor_state(chaotic).stability
    )


def test_rapport_warmth_rises_with_warmth_and_support() -> None:
    cold = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True,
        warmth=0.1, support_presence=0.1, cross_track_tension=0.6,
    )
    warm = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True, has_prediction_error=True,
        warmth=0.9, support_presence=0.8, cross_track_stability=0.8,
        pe_signed_reward=0.4, cross_track_tension=0.1,
    )
    assert readout_interlocutor_state(warm).rapport_warmth > (
        readout_interlocutor_state(cold).rapport_warmth
    )


def test_pace_pressure_rises_with_task_and_switch_pressure() -> None:
    slow = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True,
        task_pressure=0.1, switch_pressure=0.0, world_drive=0.1,
        stabilize_bias=0.8, warmth=0.8,
    )
    urgent = InterlocutorReadoutContext(
        has_dual_track=True, has_evaluation=True,
        task_pressure=0.9, switch_pressure=0.8, world_drive=0.8,
        task_bias=0.9, stabilize_bias=0.0, warmth=0.3,
    )
    assert readout_interlocutor_state(urgent).pace_pressure > (
        readout_interlocutor_state(slow).pace_pressure
    )


# ---------------------------------------------------------------------------
# Cold-start & confidence
# ---------------------------------------------------------------------------


def test_cold_readout_stays_near_neutral() -> None:
    cold = readout_interlocutor_state(InterlocutorReadoutContext())
    # With evidence_score=0.10, movement is at most ~0.10 in any axis.
    for axis_name in (
        "engagement_intensity",
        "self_disclosure_level",
        "task_focus_level",
        "emotional_weight",
        "cognitive_engagement",
        "resistance_level",
        "openness_to_guidance",
        "directness",
        "stability",
        "rapport_warmth",
        "pace_pressure",
    ):
        value = getattr(cold, axis_name)
        assert abs(value - 0.5) <= 0.12, (
            f"{axis_name} drifted too far from neutral at cold start: {value}"
        )
    assert abs(cold.trust_signal) < 0.05


def test_readout_confidence_scales_with_evidence() -> None:
    cold = readout_interlocutor_state(InterlocutorReadoutContext())
    warm = readout_interlocutor_state(
        InterlocutorReadoutContext(
            has_dual_track=True, has_evaluation=True,
            has_prediction_error=True, has_memory=True, has_commitment=True,
        )
    )
    assert warm.readout_confidence > cold.readout_confidence
    assert warm.readout_confidence >= 0.80


def test_rationale_tags_with_readout_version_and_regime_id() -> None:
    ctx = InterlocutorReadoutContext(
        active_regime_id="problem_solving",
        has_dual_track=True, has_evaluation=True,
        task_bias=0.9, world_drive=0.8,
    )
    state = readout_interlocutor_state(ctx)
    assert state.rationale.startswith("readout.v1.interlocutor:problem_solving:")


# ---------------------------------------------------------------------------
# Duck-typed builder
# ---------------------------------------------------------------------------


def test_builder_none_inputs_return_low_evidence() -> None:
    ctx = build_interlocutor_readout_context_from_snapshots()
    assert ctx.evidence_score() == pytest.approx(0.10, abs=1e-4)
    assert ctx.active_regime_id == ""


def test_builder_extracts_from_real_kernel_snapshots() -> None:
    from volvence_zero.dual_track.core import DualTrackSnapshot, TrackState
    from volvence_zero.evaluation.backbone import EvaluationScore, EvaluationSnapshot
    from volvence_zero.memory.store import MemoryEntry, MemorySnapshot, Track
    from volvence_zero.prediction.error import (
        ActualOutcome,
        PredictedOutcome,
        PredictionError,
        PredictionErrorSnapshot,
    )
    from volvence_zero.regime import RegimeIdentity, RegimeSnapshot

    regime = RegimeSnapshot(
        active_regime=RegimeIdentity(
            regime_id="problem_solving",
            name="problem solving",
            embedding=(0.8, 0.2, 0.3),
            entry_conditions="",
            exit_conditions="",
            historical_effectiveness=0.6,
        ),
        previous_regime=None,
        switch_reason="",
        candidate_regimes=(("problem_solving", 0.9),),
        turns_in_current_regime=3,
        description="",
    )
    dual_track = DualTrackSnapshot(
        world_track=TrackState(
            track=Track.WORLD, active_goals=(), recent_credits=(),
            controller_code=(0.80, 0.40, 0.20),
            tension_level=0.40,
            abstract_action_hint="task_controller",
        ),
        self_track=TrackState(
            track=Track.SELF, active_goals=(), recent_credits=(),
            controller_code=(0.30, 0.20, 0.15),
            tension_level=0.20,
            abstract_action_hint="task_controller",
        ),
        cross_track_tension=0.20,
        description="",
    )
    evaluation = EvaluationSnapshot(
        turn_scores=(
            EvaluationScore(family="f", metric_name="warmth", value=0.45,
                            confidence=1.0, evidence=""),
            EvaluationScore(family="f", metric_name="support_presence", value=0.40,
                            confidence=1.0, evidence=""),
            EvaluationScore(family="f", metric_name="task_pressure", value=0.75,
                            confidence=1.0, evidence=""),
            EvaluationScore(family="f", metric_name="cross_track_stability",
                            value=0.80, confidence=1.0, evidence=""),
            EvaluationScore(family="f", metric_name="info_integration", value=0.70,
                            confidence=1.0, evidence=""),
        ),
        session_scores=(), alerts=(), description="",
    )
    pe = PredictionErrorSnapshot(
        evaluated_prediction=None,
        actual_outcome=ActualOutcome(
            observed_turn_index=1, task_progress=0.6,
            relationship_delta=0.2, regime_stability=0.7,
            action_payoff=0.5, description="",
        ),
        next_prediction=PredictedOutcome(
            source_turn_index=1, target_turn_index=2,
            predicted_task_progress=0.6,
            predicted_relationship_delta=0.2,
            predicted_regime_stability=0.7,
            predicted_action_payoff=0.5, confidence=0.7, description="",
        ),
        error=PredictionError(
            task_error=0.0, relationship_error=0.0,
            regime_error=0.0, action_error=0.0,
            magnitude=0.25, signed_reward=0.15, description="",
        ),
        turn_index=1, bootstrap=False, description="",
    )
    memory = MemorySnapshot(
        transient_summary="", episodic_summary="", durable_summary="",
        retrieved_entries=(
            MemoryEntry(entry_id="w1", content="x", track=Track.WORLD,
                        stratum="episodic", created_at_ms=0, last_accessed_ms=0,
                        strength=0.5, tags=()),
            MemoryEntry(entry_id="w2", content="y", track=Track.WORLD,
                        stratum="episodic", created_at_ms=0, last_accessed_ms=0,
                        strength=0.5, tags=()),
            MemoryEntry(entry_id="s1", content="z", track=Track.SELF,
                        stratum="episodic", created_at_ms=0, last_accessed_ms=0,
                        strength=0.5, tags=()),
        ),
        total_entries_by_stratum=(), pending_promotions=0, pending_decays=0,
        cms_state=None, description="",
    )
    ctx = build_interlocutor_readout_context_from_snapshots(
        regime_snapshot=regime,
        dual_track_snapshot=dual_track,
        evaluation_snapshot=evaluation,
        prediction_error_snapshot=pe,
        memory_snapshot=memory,
        commitment_snapshot=None,  # test partial
    )
    assert ctx.active_regime_id == "problem_solving"
    assert ctx.turns_in_current_regime == 3
    assert ctx.has_dual_track is True
    assert ctx.has_commitment is False
    assert ctx.cross_track_tension == pytest.approx(0.20)
    assert ctx.world_drive == pytest.approx(0.80)
    assert ctx.task_bias == pytest.approx(1.0)  # both tracks task_controller
    assert ctx.warmth == pytest.approx(0.45)
    assert ctx.pe_magnitude == pytest.approx(0.25)
    assert ctx.world_presence == pytest.approx(2.0 / 3.0)
    assert ctx.self_presence == pytest.approx(1.0 / 3.0)
    # commitment_snapshot=None -> alignment trend stays at 0.
    assert ctx.commitment_alignment_trend == 0.0
    # Five out of six has_* flags set -> evidence_score = 0.85.
    assert ctx.evidence_score() == pytest.approx(0.85, abs=1e-4)


def test_builder_commitment_alignment_trend_from_mock_entries() -> None:
    """Feed a stub commitment snapshot with 2 reject + 1 agree
    -> expected trend = (-1 - 1 + 1) / 3 = -0.333.
    """
    from dataclasses import dataclass
    from enum import Enum

    class _Align(str, Enum):
        AGREE = "agree"
        REJECT = "reject"

    @dataclass(frozen=True)
    class _Entry:
        last_alignment: _Align

    @dataclass(frozen=True)
    class _Value:
        lifecycle_entries: tuple

    value = _Value(
        lifecycle_entries=(
            _Entry(_Align.REJECT),
            _Entry(_Align.REJECT),
            _Entry(_Align.AGREE),
        )
    )
    ctx = build_interlocutor_readout_context_from_snapshots(commitment_snapshot=value)
    assert ctx.has_commitment is True
    assert ctx.commitment_alignment_trend == pytest.approx(-1.0 / 3.0, abs=1e-4)


def test_builder_accepts_snapshot_wrapper_with_value_field() -> None:
    """Real kernel hands ``Snapshot[X]`` objects whose payload
    lives at ``.value``; the builder must unwrap them.
    """
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Wrapper:
        value: object

    @dataclass(frozen=True)
    class _Regime:
        active_regime: object
        turns_in_current_regime: int

    @dataclass(frozen=True)
    class _Identity:
        regime_id: str

    payload = _Regime(
        active_regime=_Identity(regime_id="emotional_support"),
        turns_in_current_regime=2,
    )
    ctx = build_interlocutor_readout_context_from_snapshots(
        regime_snapshot=_Wrapper(value=payload),
    )
    assert ctx.active_regime_id == "emotional_support"
    assert ctx.turns_in_current_regime == 2
