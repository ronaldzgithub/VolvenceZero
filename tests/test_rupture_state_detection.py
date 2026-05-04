"""Unit tests for rupture_state per-source detection functions."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
    DialogueExternalOutcomeSnapshot,
)
from volvence_zero.rupture_state import (
    RuptureEvidenceSource,
    RuptureKind,
)
from volvence_zero.rupture_state.detection import (
    behavioral_signal,
    external_user_signal,
    llm_proposal_signal,
    pe_spike_signal,
    self_check_signal,
)


@dataclass
class _PEError:
    task_error: float = 0.0
    relationship_error: float = 0.0
    regime_error: float = 0.0
    action_error: float = 0.0
    magnitude: float = 0.0


@dataclass
class _PESnapshot:
    error: _PEError
    bootstrap: bool = False


@dataclass
class _RelationshipStub:
    repair_pressure: float = 0.0
    unresolved_tension_count: int = 0
    stabilization_need: float = 0.0


def test_pe_spike_returns_empty_on_bootstrap() -> None:
    snap = _PESnapshot(error=_PEError(magnitude=0.9), bootstrap=True)
    assert pe_spike_signal(snap) == ()


def test_pe_spike_returns_empty_on_none_error() -> None:
    class _NoError:
        bootstrap = False

    assert pe_spike_signal(_NoError()) == ()


def test_pe_spike_returns_empty_when_below_thresholds() -> None:
    snap = _PESnapshot(
        error=_PEError(magnitude=0.2, relationship_error=0.1),
    )
    assert pe_spike_signal(snap) == ()


def test_pe_spike_returns_signal_when_relationship_error_is_high() -> None:
    snap = _PESnapshot(
        error=_PEError(magnitude=0.2, relationship_error=0.6),
    )
    signals = pe_spike_signal(snap)
    assert len(signals) == 1
    assert signals[0].source is RuptureEvidenceSource.INTERNAL_PE
    assert signals[0].kind_hint is None  # PE never sets a kind hint
    assert signals[0].signal_strength == pytest.approx(0.6)
    assert 0.0 <= signals[0].confidence <= 0.5


def test_behavioral_signal_empty_when_below_thresholds() -> None:
    snap = _RelationshipStub(repair_pressure=0.2, unresolved_tension_count=0)
    assert behavioral_signal(snap) == ()


def test_behavioral_signal_fires_on_repair_pressure() -> None:
    snap = _RelationshipStub(repair_pressure=0.7)
    signals = behavioral_signal(snap)
    assert len(signals) == 1
    assert signals[0].source is RuptureEvidenceSource.BEHAVIORAL_TRACE
    assert signals[0].kind_hint is None  # behavioral alone does not name a kind
    assert signals[0].signal_strength == pytest.approx(0.7)


def test_self_check_signal_is_stubbed_in_v0() -> None:
    assert self_check_signal(object()) == ()
    assert self_check_signal(None) == ()


def _missed_evidence() -> DialogueExternalOutcomeEvidence:
    return DialogueExternalOutcomeEvidence(
        evidence_id="user:explicit:missed:turn-3",
        turn_index=3,
        kind=DialogueExternalOutcomeKind.MISSED,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.9,
        evidence_ref="user:explicit",
    )


def _helped_evidence() -> DialogueExternalOutcomeEvidence:
    return DialogueExternalOutcomeEvidence(
        evidence_id="user:explicit:helped:turn-3",
        turn_index=3,
        kind=DialogueExternalOutcomeKind.HELPED,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.9,
        evidence_ref="user:explicit",
    )


def test_external_user_signal_maps_missed_to_misread() -> None:
    snapshot = DialogueExternalOutcomeSnapshot(
        turn_index=3,
        entries=(_missed_evidence(),),
        description="test",
    )
    signals = external_user_signal(snapshot)
    assert len(signals) == 1
    assert signals[0].source is RuptureEvidenceSource.EXTERNAL_USER
    assert signals[0].kind_hint is RuptureKind.MISREAD
    assert signals[0].signal_strength == pytest.approx(0.9)


def test_external_user_signal_skips_positive_outcomes() -> None:
    snapshot = DialogueExternalOutcomeSnapshot(
        turn_index=3,
        entries=(_helped_evidence(),),
        description="positive-only",
    )
    assert external_user_signal(snapshot) == ()


def test_external_user_signal_never_consumes_llm_proposal_source() -> None:
    llm_ev = DialogueExternalOutcomeEvidence(
        evidence_id="ev-llm-1",
        turn_index=3,
        kind=DialogueExternalOutcomeKind.MISSED,
        source=DialogueExternalOutcomeEvidenceSource.LLM_PROPOSAL,
        confidence=0.5,
        evidence_ref="llm",
    )
    snapshot = DialogueExternalOutcomeSnapshot(
        turn_index=3,
        entries=(llm_ev,),
        description="llm-only",
    )
    # external_user_signal must leave LLM entries to llm_proposal_signal.
    assert external_user_signal(snapshot) == ()


def test_llm_proposal_signal_is_off_by_default() -> None:
    llm_ev = DialogueExternalOutcomeEvidence(
        evidence_id="ev-llm-2",
        turn_index=3,
        kind=DialogueExternalOutcomeKind.MISSED,
        source=DialogueExternalOutcomeEvidenceSource.LLM_PROPOSAL,
        confidence=0.9,
        evidence_ref="llm",
    )
    snapshot = DialogueExternalOutcomeSnapshot(
        turn_index=3,
        entries=(llm_ev,),
        description="llm-only",
    )
    assert llm_proposal_signal(snapshot) == ()


def test_llm_proposal_signal_clamps_confidence_when_enabled() -> None:
    llm_ev = DialogueExternalOutcomeEvidence(
        evidence_id="ev-llm-3",
        turn_index=3,
        kind=DialogueExternalOutcomeKind.MISSED,
        source=DialogueExternalOutcomeEvidenceSource.LLM_PROPOSAL,
        confidence=0.95,
        evidence_ref="llm",
    )
    snapshot = DialogueExternalOutcomeSnapshot(
        turn_index=3,
        entries=(llm_ev,),
        description="llm-only",
    )
    signals = llm_proposal_signal(snapshot, allow_llm_proposals=True)
    assert len(signals) == 1
    assert signals[0].source is RuptureEvidenceSource.LLM_PROPOSAL
    # Clamped at 0.4.
    assert signals[0].confidence == pytest.approx(0.4)
    assert signals[0].signal_strength == pytest.approx(0.4)
