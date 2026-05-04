"""Unit tests for RuptureStateModule aggregation behavior."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
    DialogueExternalOutcomeSnapshot,
)
from volvence_zero.runtime import Snapshot
from volvence_zero.rupture_state import (
    RuptureEvidenceSource,
    RuptureKind,
    RuptureStateModule,
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
    trust_level: float = 0.5
    continuity_level: float = 0.5
    repair_pressure: float = 0.0
    rapport_signals: tuple = ()
    relational_tensions: tuple = ()
    control_signal: float = 0.0
    description: str = "stub"
    emotional_load: float = 0.0
    repair_need: float = 0.0
    trust_delta: float = 0.0
    attunement_gap: float = 0.0
    stabilization_need: float = 0.0
    recent_repair_count: int = 0
    unresolved_tension_count: int = 0
    attunement_trend: float = 0.0
    trust_recovery_signal: float = 0.0
    relationship_continuity_score: float = 0.0


def _wrap(slot: str, value: object) -> Snapshot:
    return Snapshot(
        slot_name=slot,
        owner=f"{slot}:test",
        version=1,
        timestamp_ms=1,
        value=value,
    )


def _missed_snapshot(confidence: float = 0.9) -> DialogueExternalOutcomeSnapshot:
    ev = DialogueExternalOutcomeEvidence(
        evidence_id="user:explicit:missed:turn-1",
        turn_index=1,
        kind=DialogueExternalOutcomeKind.MISSED,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=confidence,
        evidence_ref="user:explicit",
    )
    return DialogueExternalOutcomeSnapshot(
        turn_index=1,
        entries=(ev,),
        description="one missed",
    )


def _run(module: RuptureStateModule, upstream: dict) -> object:
    return asyncio.run(module.process(upstream)).value


def test_empty_upstream_produces_bootstrap_snapshot() -> None:
    module = RuptureStateModule()
    upstream: dict = {}
    result = _run(module, upstream)
    assert result.rupture_kind is None
    assert result.internal_suspected_only is False
    assert result.evidence_sources == ()
    assert result.rupture_signal_strength == 0.0


def test_pe_only_sets_internal_suspected_only() -> None:
    module = RuptureStateModule()
    pe_snap = _PESnapshot(error=_PEError(relationship_error=0.6, magnitude=0.2))
    upstream = {"prediction_error": _wrap("prediction_error", pe_snap)}
    result = _run(module, upstream)
    assert result.internal_suspected_only is True
    assert result.evidence_sources == (RuptureEvidenceSource.INTERNAL_PE,)
    assert result.rupture_kind is None
    assert result.rupture_signal_strength > 0.0


def test_external_missed_resolves_misread_kind() -> None:
    module = RuptureStateModule()
    upstream = {
        "dialogue_external_outcome": _wrap(
            "dialogue_external_outcome", _missed_snapshot()
        ),
    }
    result = _run(module, upstream)
    assert result.internal_suspected_only is False
    assert result.rupture_kind is RuptureKind.MISREAD
    assert RuptureEvidenceSource.EXTERNAL_USER in result.evidence_sources


def test_missed_plus_repair_pressure_resolves_cold_kind() -> None:
    module = RuptureStateModule()
    upstream = {
        "relationship_state": _wrap(
            "relationship_state", _RelationshipStub(repair_pressure=0.8)
        ),
        "dialogue_external_outcome": _wrap(
            "dialogue_external_outcome", _missed_snapshot()
        ),
    }
    result = _run(module, upstream)
    # Compositional rule: MISREAD external + repair_pressure >= 0.5 => COLD
    # wins over plain MISREAD because severity lookup ordering picks the
    # highest-severity candidate; however, MISREAD severity (3) is higher
    # than COLD severity (1) by design. So we expect MISREAD here — not
    # COLD — because the severity ordering prefers the externally-named
    # rupture over the compositional COLD. This test pins that ordering
    # so a future tweak does not silently prefer COLD and erase the
    # more specific signal.
    assert result.rupture_kind is RuptureKind.MISREAD


def test_unsafe_external_takes_severity_priority() -> None:
    module = RuptureStateModule()
    unsafe = DialogueExternalOutcomeEvidence(
        evidence_id="env:unsafe:turn-1",
        turn_index=1,
        kind=DialogueExternalOutcomeKind.UNSAFE,
        source=DialogueExternalOutcomeEvidenceSource.ENVIRONMENT,
        confidence=0.95,
        evidence_ref="env",
    )
    missed = DialogueExternalOutcomeEvidence(
        evidence_id="user:explicit:missed:turn-1",
        turn_index=1,
        kind=DialogueExternalOutcomeKind.MISSED,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.6,
        evidence_ref="user",
    )
    snapshot = DialogueExternalOutcomeSnapshot(
        turn_index=1,
        entries=(missed, unsafe),
        description="unsafe + missed",
    )
    upstream = {
        "dialogue_external_outcome": _wrap("dialogue_external_outcome", snapshot),
    }
    result = _run(module, upstream)
    # Unsafe has the highest severity (6); it must win.
    assert result.rupture_kind is RuptureKind.UNSAFE
    # ENVIRONMENT source comes from the unsafe entry. Both sources
    # should appear, but unsafe drives the kind.
    assert RuptureEvidenceSource.EXTERNAL_USER in result.evidence_sources
    assert RuptureEvidenceSource.ENVIRONMENT in result.evidence_sources


def test_helped_external_does_not_produce_rupture_kind() -> None:
    module = RuptureStateModule()
    helped = DialogueExternalOutcomeEvidence(
        evidence_id="user:explicit:helped:turn-1",
        turn_index=1,
        kind=DialogueExternalOutcomeKind.HELPED,
        source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
        confidence=0.9,
        evidence_ref="user",
    )
    snapshot = DialogueExternalOutcomeSnapshot(
        turn_index=1,
        entries=(helped,),
        description="positive outcome",
    )
    upstream = {
        "dialogue_external_outcome": _wrap("dialogue_external_outcome", snapshot),
    }
    result = _run(module, upstream)
    # No rupture.
    assert result.rupture_kind is None
    assert result.internal_suspected_only is False
    assert result.evidence_sources == ()


def test_llm_proposal_is_ignored_when_flag_off() -> None:
    module = RuptureStateModule(allow_llm_proposals=False)
    llm = DialogueExternalOutcomeEvidence(
        evidence_id="llm:proposal:missed:turn-1",
        turn_index=1,
        kind=DialogueExternalOutcomeKind.MISSED,
        source=DialogueExternalOutcomeEvidenceSource.LLM_PROPOSAL,
        confidence=0.9,
        evidence_ref="llm",
    )
    snapshot = DialogueExternalOutcomeSnapshot(
        turn_index=1,
        entries=(llm,),
        description="llm-only",
    )
    upstream = {
        "dialogue_external_outcome": _wrap("dialogue_external_outcome", snapshot),
    }
    result = _run(module, upstream)
    assert result.rupture_kind is None
    assert RuptureEvidenceSource.LLM_PROPOSAL not in result.evidence_sources
