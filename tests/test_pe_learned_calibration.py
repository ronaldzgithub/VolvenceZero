"""C2 (#80/#86): bounded learned calibrations inside the PE owner.

The static external-outcome axis-bias table and the AAC alignment
severity table become initialisations of owner-internal calibrators;
drift is bounded to a fixed envelope so a reset to the table is always
a nearby rollback point.
"""

from __future__ import annotations

from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind
from volvence_zero.prediction.error import (
    _ALIGNMENT_TRANSITION_SEVERITY,
    _EXTERNAL_OUTCOME_AXIS_BIAS,
    ActualOutcome,
    ExternalOutcomeBiasCalibrator,
    PredictionErrorModule,
    _AlignmentSeverityCalibrator,
)


def _outcome(
    *,
    task: float = 0.5,
    relationship: float = 0.0,
    regime: float = 0.5,
    action: float = 0.5,
) -> ActualOutcome:
    return ActualOutcome(
        observed_turn_index=1,
        task_progress=task,
        relationship_delta=relationship,
        regime_stability=regime,
        action_payoff=action,
        description="test outcome",
    )


def test_bias_calibrator_initialises_from_static_table() -> None:
    calibrator = ExternalOutcomeBiasCalibrator()
    for kind, values in _EXTERNAL_OUTCOME_AXIS_BIAS.items():
        assert calibrator.bias_for(kind) == values
    assert calibrator.bias_for(DialogueExternalOutcomeKind.HELPED) == (
        0.0,
        0.50,
        0.0,
        0.30,
    )


def test_bias_calibrator_moves_toward_internal_signal_and_stays_bounded() -> None:
    calibrator = ExternalOutcomeBiasCalibrator()
    kind = DialogueExternalOutcomeKind.MISSED
    initial = _EXTERNAL_OUTCOME_AXIS_BIAS[kind]
    # Internal evidence says the relationship actually moved positive,
    # contradicting the MISSED table's -0.60 relationship bias.
    for _ in range(500):
        calibrator.calibrate(
            kind,
            confidence=1.0,
            internal_outcome=_outcome(relationship=0.9),
        )
    calibrated = calibrator.bias_for(kind)
    assert calibrated is not None
    assert calibrated[1] > initial[1]
    for index in range(4):
        assert calibrated[index] >= initial[index] - 0.15 - 1e-9
        assert calibrated[index] <= initial[index] + 0.15 + 1e-9


def test_severity_calibrator_initialises_from_table_and_settles() -> None:
    calibrator = _AlignmentSeverityCalibrator()
    key = ("agree", "reject")
    assert calibrator.severity_for(key) == _ALIGNMENT_TRANSITION_SEVERITY[key]
    # A "regression" transition repeatedly followed by non-negative
    # relationship movement should get its severity pulled down.
    for _ in range(200):
        calibrator.record_applied(key, is_recovery=False)
        calibrator.settle(realized_relationship_delta=0.8)
    lowered = calibrator.severity_for(key)
    assert lowered < _ALIGNMENT_TRANSITION_SEVERITY[key]
    assert lowered >= _ALIGNMENT_TRANSITION_SEVERITY[key] - 0.15 - 1e-9


def test_severity_calibrator_settle_without_pending_is_noop() -> None:
    calibrator = _AlignmentSeverityCalibrator()
    assert calibrator.settle(realized_relationship_delta=0.5) == 0


def test_pe_module_owns_both_calibrators() -> None:
    module = PredictionErrorModule()
    assert isinstance(module._external_bias_calibrator, ExternalOutcomeBiasCalibrator)
    assert isinstance(
        module._alignment_severity_calibrator, _AlignmentSeverityCalibrator
    )
