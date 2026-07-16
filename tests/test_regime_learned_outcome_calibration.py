"""C1 (#80): learned external-outcome calibration on the regime owner."""

from __future__ import annotations

from volvence_zero.regime.identity import (
    _EXTERNAL_OUTCOME_REGIME_SCORE,
    _EXTERNAL_OUTCOME_SCORE_ENVELOPE,
    RegimeModule,
)
from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeEvidenceSource,
    DialogueExternalOutcomeKind,
    DialogueExternalOutcomeSnapshot,
)
from volvence_zero.runtime import WiringLevel


def _snapshot(kind: DialogueExternalOutcomeKind, *, turn_index: int = 1) -> DialogueExternalOutcomeSnapshot:
    return DialogueExternalOutcomeSnapshot(
        turn_index=turn_index,
        entries=(
            DialogueExternalOutcomeEvidence(
                evidence_id=f"user:explicit:{kind.value}:turn-{turn_index}",
                turn_index=turn_index,
                kind=kind,
                source=DialogueExternalOutcomeEvidenceSource.USER_EXPLICIT,
                confidence=0.9,
                evidence_ref="user:explicit",
            ),
        ),
        description=f"one {kind.value}",
    )


def test_external_outcome_scores_initialise_from_static_table() -> None:
    module = RegimeModule(wiring_level=WiringLevel.ACTIVE)
    assert module._external_outcome_scores == _EXTERNAL_OUTCOME_REGIME_SCORE


def test_external_outcome_calibration_moves_toward_internal_trajectory() -> None:
    module = RegimeModule(wiring_level=WiringLevel.ACTIVE)
    # Internal trajectory strongly negative while the HELPED table value
    # is 0.85 -> the learned calibration should drift downward (bounded).
    module._turn_evaluation_scores = [0.1, 0.1, 0.1]
    module._turn_index = 3
    initial = module._external_outcome_scores[DialogueExternalOutcomeKind.HELPED]
    module._ingest_external_outcome_attributions(
        external_outcome_snapshot=_snapshot(DialogueExternalOutcomeKind.HELPED),
        current_regime_id="emotional_support",
        abstract_action=None,
        action_family_version=0,
    )
    updated = module._external_outcome_scores[DialogueExternalOutcomeKind.HELPED]
    assert updated < initial


def test_external_outcome_calibration_stays_inside_envelope() -> None:
    module = RegimeModule(wiring_level=WiringLevel.ACTIVE)
    module._turn_evaluation_scores = [0.0] * 8
    module._turn_index = 8
    initial = _EXTERNAL_OUTCOME_REGIME_SCORE[DialogueExternalOutcomeKind.HELPED]
    for turn in range(1, 200):
        module._ingest_external_outcome_attributions(
            external_outcome_snapshot=_snapshot(
                DialogueExternalOutcomeKind.HELPED, turn_index=min(turn, 8)
            ),
            current_regime_id="emotional_support",
            abstract_action=None,
            action_family_version=0,
        )
    calibrated = module._external_outcome_scores[DialogueExternalOutcomeKind.HELPED]
    assert calibrated >= initial - _EXTERNAL_OUTCOME_SCORE_ENVELOPE - 1e-9
    assert calibrated <= initial + _EXTERNAL_OUTCOME_SCORE_ENVELOPE + 1e-9


def test_metacontroller_evidence_table_matches_historical_mapping() -> None:
    from volvence_zero.regime.templates import (
        consolidation_gain_multipliers,
        metacontroller_evidence_deltas,
    )

    assert dict(metacontroller_evidence_deltas("self_axis")) == {
        "emotional_support": 0.04,
        "repair_and_deescalation": 0.04,
    }
    assert dict(metacontroller_evidence_deltas("world_axis")) == {
        "guided_exploration": 0.04,
        "problem_solving": 0.04,
    }
    assert dict(metacontroller_evidence_deltas("shared_axis")) == {
        "acquaintance_building": 0.04,
        "guided_exploration": 0.04,
    }
    assert dict(metacontroller_evidence_deltas("stabilize_axis")) == {"casual_social": 0.02}
    assert dict(metacontroller_evidence_deltas("sparse_switch")) == {"guided_exploration": 0.03}
    assert dict(metacontroller_evidence_deltas("posterior_guard")) == {
        "repair_and_deescalation": 0.03
    }
    assert dict(metacontroller_evidence_deltas("replacement")) == {"problem_solving": 0.03}
    assert dict(metacontroller_evidence_deltas("rollback_guard")) == {
        "repair_and_deescalation": 0.05
    }
    assert dict(consolidation_gain_multipliers("increase_self_track_priority")) == {
        "emotional_support": 1.35,
        "repair_and_deescalation": 0.95,
        "acquaintance_building": 0.7,
    }
    assert dict(consolidation_gain_multipliers("increase_world_track_priority")) == {
        "problem_solving": 1.35,
        "guided_exploration": 0.8,
    }
