"""Packet 7.0: detector tests for the 5 new SignalSource members.

USER_REPLY_LATENCY / USER_REPLY_LENGTH / USER_INITIATIVE_QUESTION /
COMMITMENT_FULFILLED / COMMITMENT_BROKEN.
"""

from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.behavior_protocol import (
    BehaviorProtocolSignalSource,
    ContextMatchSignal,
)
from volvence_zero.interlocutor.contracts import (
    InterlocutorState,
    InterlocutorStateSnapshot,
    with_zones,
)
from volvence_zero.protocol_runtime.activation import _signal_is_firing


@dataclass(frozen=True)
class _StubCommitmentState:
    honored_commitment_refs: tuple[str, ...] = ()
    at_risk_commitments: tuple = ()


def _interlocutor(**overrides) -> InterlocutorStateSnapshot:
    base = InterlocutorState(
        engagement_intensity=0.50,
        self_disclosure_level=0.50,
        task_focus_level=0.50,
        emotional_weight=0.50,
        cognitive_engagement=0.50,
        resistance_level=0.50,
        openness_to_guidance=0.50,
        directness=0.50,
        trust_signal=0.0,
        stability=0.50,
        rapport_warmth=0.50,
        pace_pressure=0.50,
        readout_confidence=0.85,
        rationale="test",
    )
    if overrides:
        from dataclasses import replace as _replace
        base = _replace(base, **overrides)
    return InterlocutorStateSnapshot(state=with_zones(base), description="")


def _signal(source: BehaviorProtocolSignalSource) -> ContextMatchSignal:
    return ContextMatchSignal(
        signal_id=f"test-{source.value}",
        measurable_via=source,
        weight=1.0,
    )


def test_user_reply_latency_fires_for_low_engagement_low_directness() -> None:
    snap = _interlocutor(engagement_intensity=0.2, directness=0.3)
    assert _signal_is_firing(
        _signal(BehaviorProtocolSignalSource.USER_REPLY_LATENCY),
        interlocutor_snapshot=snap,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
    ) is True


def test_user_reply_latency_does_not_fire_for_high_engagement() -> None:
    snap = _interlocutor(engagement_intensity=0.9, directness=0.2)
    assert _signal_is_firing(
        _signal(BehaviorProtocolSignalSource.USER_REPLY_LATENCY),
        interlocutor_snapshot=snap,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
    ) is False


def test_user_reply_length_fires_for_high_engagement_high_disclosure() -> None:
    snap = _interlocutor(engagement_intensity=0.85, self_disclosure_level=0.7)
    assert _signal_is_firing(
        _signal(BehaviorProtocolSignalSource.USER_REPLY_LENGTH),
        interlocutor_snapshot=snap,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
    ) is True


def test_user_initiative_question_fires_when_leaning_in() -> None:
    snap = _interlocutor(
        engagement_intensity=0.85,
        directness=0.75,
        resistance_level=0.2,
    )
    assert _signal_is_firing(
        _signal(BehaviorProtocolSignalSource.USER_INITIATIVE_QUESTION),
        interlocutor_snapshot=snap,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
    ) is True


def test_user_initiative_question_does_not_fire_with_resistance() -> None:
    snap = _interlocutor(
        engagement_intensity=0.85,
        directness=0.75,
        resistance_level=0.6,
    )
    assert _signal_is_firing(
        _signal(BehaviorProtocolSignalSource.USER_INITIATIVE_QUESTION),
        interlocutor_snapshot=snap,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
    ) is False


def test_commitment_fulfilled_fires_when_honored_refs_present() -> None:
    state = _StubCommitmentState(honored_commitment_refs=("c1", "c2"))
    assert _signal_is_firing(
        _signal(BehaviorProtocolSignalSource.COMMITMENT_FULFILLED),
        interlocutor_snapshot=None,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
        commitment_state=state,
    ) is True


def test_commitment_broken_fires_when_at_risk_present() -> None:
    state = _StubCommitmentState(at_risk_commitments=("c1",))
    assert _signal_is_firing(
        _signal(BehaviorProtocolSignalSource.COMMITMENT_BROKEN),
        interlocutor_snapshot=None,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
        commitment_state=state,
    ) is True


def test_commitment_signals_no_op_when_state_missing() -> None:
    assert _signal_is_firing(
        _signal(BehaviorProtocolSignalSource.COMMITMENT_FULFILLED),
        interlocutor_snapshot=None,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
    ) is False
    assert _signal_is_firing(
        _signal(BehaviorProtocolSignalSource.COMMITMENT_BROKEN),
        interlocutor_snapshot=None,
        rupture_snapshot=None,
        boundary_policy_snapshot=None,
    ) is False
