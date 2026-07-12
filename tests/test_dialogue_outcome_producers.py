from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.agent.dialogue_outcome_producers import (
    PE_CONTINUED_CONFIDENCE,
    PE_CONTINUED_MAGNITUDE_THRESHOLD,
    commitment_outcome_evidence_from_commitment,
    pe_continued_evidence_from_prediction_error,
    scene_closed_evidence,
)
from volvence_zero.dialogue_trace import (
    DialogueOutcomeEvidenceSource,
    DialogueOutcomeKind,
)


@dataclass(frozen=True)
class _PEErrorStub:
    magnitude: float


@dataclass(frozen=True)
class _PESnapshotStub:
    bootstrap: bool
    error: _PEErrorStub | None


@dataclass(frozen=True)
class _CommitmentLifecycleStub:
    record_id: str
    last_outcome: object | None
    last_outcome_at_turn: int


class _OutcomeKindStub:
    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value


@dataclass(frozen=True)
class _CommitmentSnapshotStub:
    lifecycle_entries: tuple[_CommitmentLifecycleStub, ...]


def test_pe_continued_evidence_emits_low_confidence_when_pe_is_small() -> None:
    snapshot = _PESnapshotStub(bootstrap=False, error=_PEErrorStub(magnitude=0.1))
    evidence = pe_continued_evidence_from_prediction_error(
        prediction_error_snapshot=snapshot,
        wave_id="wave-1",
    )

    assert len(evidence) == 1
    record = evidence[0]
    assert record.outcome_kind is DialogueOutcomeKind.CONTINUED
    assert record.confidence == PE_CONTINUED_CONFIDENCE
    assert record.source is DialogueOutcomeEvidenceSource.EVALUATION


def test_pe_continued_evidence_silent_on_bootstrap_or_high_pe() -> None:
    bootstrap_snapshot = _PESnapshotStub(bootstrap=True, error=_PEErrorStub(magnitude=0.0))
    high_pe_snapshot = _PESnapshotStub(
        bootstrap=False,
        error=_PEErrorStub(magnitude=PE_CONTINUED_MAGNITUDE_THRESHOLD + 0.1),
    )

    assert pe_continued_evidence_from_prediction_error(
        prediction_error_snapshot=bootstrap_snapshot,
        wave_id="wave-1",
    ) == ()
    assert pe_continued_evidence_from_prediction_error(
        prediction_error_snapshot=high_pe_snapshot,
        wave_id="wave-1",
    ) == ()


def test_commitment_evidence_maps_typed_outcomes_to_dialogue_kinds() -> None:
    rejected = _OutcomeKindStub(name="REJECTED", value="commitment_rejected")
    completed = _OutcomeKindStub(name="COMPLETED", value="commitment_completed")
    stalled = _OutcomeKindStub(name="STALLED", value="commitment_stalled")
    snapshot = _CommitmentSnapshotStub(
        lifecycle_entries=(
            _CommitmentLifecycleStub(
                record_id="r-1",
                last_outcome=rejected,
                last_outcome_at_turn=2,
            ),
            _CommitmentLifecycleStub(
                record_id="r-2",
                last_outcome=completed,
                last_outcome_at_turn=2,
            ),
            _CommitmentLifecycleStub(
                record_id="r-3",
                last_outcome=stalled,
                last_outcome_at_turn=1,
            ),
        )
    )

    evidence = commitment_outcome_evidence_from_commitment(
        commitment_snapshot=snapshot,
        wave_id="wave-2",
        current_turn_index=2,
    )

    outcomes = {record.outcome_kind for record in evidence}
    assert outcomes == {DialogueOutcomeKind.REJECTED, DialogueOutcomeKind.CLARIFIED}


def test_commitment_evidence_silent_when_no_lifecycle_or_snapshot() -> None:
    assert (
        commitment_outcome_evidence_from_commitment(
            commitment_snapshot=None,
            wave_id="wave-1",
            current_turn_index=1,
        )
        == ()
    )
    snapshot = _CommitmentSnapshotStub(lifecycle_entries=())
    assert (
        commitment_outcome_evidence_from_commitment(
            commitment_snapshot=snapshot,
            wave_id="wave-1",
            current_turn_index=1,
        )
        == ()
    )


def test_scene_closed_evidence_is_high_confidence_structural() -> None:
    evidence = scene_closed_evidence(
        scene_id="scene-1",
        reason="scene-end",
        prediction_id="pe:prediction_error:turn-1:next",
    )

    assert evidence.outcome_kind is DialogueOutcomeKind.SCENE_CLOSED
    assert evidence.source is DialogueOutcomeEvidenceSource.SCENE_EVENT
    assert evidence.source_owner == "SceneManager"
    assert "scene:scene-1" in evidence.evidence_refs
    assert "reason:scene-end" in evidence.evidence_refs
    assert "scene_prediction:pe:prediction_error:turn-1:next" in evidence.evidence_refs
