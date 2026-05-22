"""OA-3 FramingAwarenessCheck contract tests."""

from __future__ import annotations

import dataclasses

import pytest

from volvence_zero.credit import (
    FramingAwarenessCheck,
    FramingRiskKind,
    ModificationGate,
    ModificationProposal,
    evaluate_gate_reasons,
)
from volvence_zero.evaluation import EvaluationSnapshot


def _clean_evaluation() -> EvaluationSnapshot:
    return EvaluationSnapshot(
        turn_scores=(),
        session_scores=(),
        alerts=(),
        description="clean",
        structured_alerts=(),
    )


def _clean_proposal() -> ModificationProposal:
    return ModificationProposal(
        target="controller.delta",
        desired_gate=ModificationGate.ONLINE,
        old_value_hash="old",
        new_value_hash="new",
        justification="typed framing evidence is supplied separately",
        is_reversible=True,
        validation_delta=0.05,
        capacity_cost=0.10,
        rollback_evidence="rollback checkpoint exists",
    )


def test_framing_check_rejects_out_of_range_risk_score() -> None:
    with pytest.raises(ValueError, match="risk_score"):
        FramingAwarenessCheck(
            risk_kind=FramingRiskKind.ALIGNMENT_FAKING,
            risk_score=1.5,
            inoculation_statement_present=False,
            evidence_id="frame:e1",
        )


def test_high_risk_framing_without_inoculation_blocks() -> None:
    proposal = dataclasses.replace(
        _clean_proposal(),
        framing_check=FramingAwarenessCheck(
            risk_kind=FramingRiskKind.REWARD_HACKING_NORMALIZED,
            risk_score=0.9,
            inoculation_statement_present=False,
            evidence_id="frame:e2",
        ),
    )

    reasons = evaluate_gate_reasons(
        proposal=proposal,
        evaluation_snapshot=_clean_evaluation(),
    )

    assert any("framing risk reward_hacking_normalized" in reason for reason in reasons)


def test_high_risk_framing_with_inoculation_does_not_add_framing_block() -> None:
    proposal = dataclasses.replace(
        _clean_proposal(),
        framing_check=FramingAwarenessCheck(
            risk_kind=FramingRiskKind.ALIGNMENT_FAKING,
            risk_score=0.9,
            inoculation_statement_present=True,
            evidence_id="frame:e3",
        ),
    )

    reasons = evaluate_gate_reasons(
        proposal=proposal,
        evaluation_snapshot=_clean_evaluation(),
    )

    assert not any("framing risk" in reason for reason in reasons)


def test_low_risk_framing_does_not_add_framing_block() -> None:
    proposal = dataclasses.replace(
        _clean_proposal(),
        framing_check=FramingAwarenessCheck(
            risk_kind=FramingRiskKind.MONITOR_DISRUPTION,
            risk_score=0.2,
            inoculation_statement_present=False,
            evidence_id="frame:e4",
        ),
    )

    reasons = evaluate_gate_reasons(
        proposal=proposal,
        evaluation_snapshot=_clean_evaluation(),
    )

    assert not any("framing risk" in reason for reason in reasons)


def test_framing_check_schema_is_stable() -> None:
    assert tuple(f.name for f in dataclasses.fields(FramingAwarenessCheck)) == (
        "risk_kind",
        "risk_score",
        "inoculation_statement_present",
        "evidence_id",
        "description",
    )
    assert {kind.value for kind in FramingRiskKind} == {
        "reward_hacking_normalized",
        "alignment_faking",
        "sabotage",
        "malicious_cooperation",
        "monitor_disruption",
        "colleague_framing",
    }
