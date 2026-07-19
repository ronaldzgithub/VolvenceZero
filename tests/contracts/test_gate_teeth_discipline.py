"""Deliberately broken fixtures proving the modification gate has teeth.

Each fixture violates one invariant of an otherwise admissible proposal. A
regression that makes the checker vacuously allow or block everything must
therefore fail this suite.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from volvence_zero.audit import AuditSnapshot
from volvence_zero.credit.gate import (
    FramingAwarenessCheck,
    FramingRiskKind,
    ModificationGate,
    ModificationProposal,
    evaluate_gate_reasons,
)
from volvence_zero.evaluation import (
    EvaluationAlert,
    EvaluationScore,
    EvaluationSnapshot,
)


def _clean_proposal() -> ModificationProposal:
    return ModificationProposal(
        target="memory.writeback.threshold",
        desired_gate=ModificationGate.ONLINE,
        old_value_hash="old",
        new_value_hash="new",
        justification="bounded test proposal",
        is_reversible=True,
        validation_delta=0.05,
        capacity_cost=0.10,
        rollback_evidence="checkpoint:teeth:baseline",
    )


def _clean_evaluation() -> EvaluationSnapshot:
    return EvaluationSnapshot(
        turn_scores=(),
        session_scores=(),
        alerts=(),
        structured_alerts=(),
        description="clean gate context",
    )


def _hard_block_audit() -> AuditSnapshot:
    return AuditSnapshot(
        audit_id="audit:teeth:hard-block",
        timestamp_ms=1,
        proposal_id="proposal:teeth",
        risk_score=0.9,
        transcript=(),
        tool_traces=(),
        detected_attack_classes=(),
        threshold_decision="hard-block",
        description="deliberately broken audit fixture",
    )


def test_clean_baseline_is_allowed() -> None:
    """The checker must not degenerate into an always-block function."""
    assert evaluate_gate_reasons(
        proposal=_clean_proposal(),
        evaluation_snapshot=_clean_evaluation(),
    ) == ()


@pytest.mark.parametrize(
    ("proposal", "evaluation", "expected_reason"),
    (
        (
            replace(_clean_proposal(), rollback_evidence=""),
            _clean_evaluation(),
            "missing rollback evidence",
        ),
        (
            replace(_clean_proposal(), capacity_cost=0.21),
            _clean_evaluation(),
            "capacity_cost 0.210 exceeds cap 0.200",
        ),
        (
            replace(_clean_proposal(), validation_delta=-0.01),
            _clean_evaluation(),
            "validation_delta -0.010 below required margin 0.000",
        ),
        (
            replace(_clean_proposal(), is_reversible=False),
            _clean_evaluation(),
            "online/background proposal is not reversible",
        ),
        (
            replace(
                _clean_proposal(),
                framing_check=FramingAwarenessCheck(
                    risk_kind=FramingRiskKind.MONITOR_DISRUPTION,
                    risk_score=0.9,
                    inoculation_statement_present=False,
                    evidence_id="framing:teeth",
                ),
            ),
            _clean_evaluation(),
            (
                "framing risk monitor_disruption score 0.900 requires "
                "explicit inoculation statement"
            ),
        ),
        (
            _clean_proposal(),
            replace(
                _clean_evaluation(),
                structured_alerts=(
                    EvaluationAlert(
                        code="teeth-high-alert",
                        severity="HIGH",
                        family="safety",
                        metric_name="gate_integrity",
                        description="deliberately injected high alert",
                    ),
                ),
            ),
            "online gate blocked by high-or-critical evaluation alert",
        ),
        (
            _clean_proposal(),
            replace(
                _clean_evaluation(),
                turn_scores=(
                    EvaluationScore(
                        family="safety",
                        metric_name="contract_integrity",
                        value=0.94,
                        confidence=1.0,
                        evidence="deliberately degraded contract integrity",
                    ),
                ),
            ),
            "contract_integrity 0.940 below 0.950",
        ),
    ),
)
def test_single_invariant_break_has_one_distinct_reason(
    proposal: ModificationProposal,
    evaluation: EvaluationSnapshot,
    expected_reason: str,
) -> None:
    """Each minimal broken model must be rejected for its own exact reason."""
    reasons = evaluate_gate_reasons(
        proposal=proposal,
        evaluation_snapshot=evaluation,
    )
    assert reasons == (expected_reason,)


def test_audit_required_missing_is_minimal_counterexample() -> None:
    reasons = evaluate_gate_reasons(
        proposal=_clean_proposal(),
        evaluation_snapshot=_clean_evaluation(),
        audit_required=True,
    )
    assert reasons == ("audit_snapshot required but missing",)


def test_audit_hard_block_is_minimal_counterexample() -> None:
    reasons = evaluate_gate_reasons(
        proposal=_clean_proposal(),
        evaluation_snapshot=_clean_evaluation(),
        audit_snapshot=_hard_block_audit(),
        audit_required=True,
    )
    assert reasons == ("audit hard-block: risk_score=0.900",)


def test_broken_fixtures_produce_distinct_reason_sets() -> None:
    broken = (
        replace(_clean_proposal(), rollback_evidence=""),
        replace(_clean_proposal(), capacity_cost=0.21),
        replace(_clean_proposal(), validation_delta=-0.01),
        replace(_clean_proposal(), is_reversible=False),
    )
    reason_sets = {
        evaluate_gate_reasons(
            proposal=proposal,
            evaluation_snapshot=_clean_evaluation(),
        )
        for proposal in broken
    }
    assert len(reason_sets) == len(broken)


@pytest.mark.parametrize(
    "protected_target",
    (
        "audit",
        "audit.policy",
        "credit.gate",
        "credit.gate.threshold",
        "evaluation.release",
        "gate.policy",
        "modification_gate.rules",
    ),
)
def test_protected_judgment_surface_is_unrepresentable(
    protected_target: str,
) -> None:
    with pytest.raises(ValueError, match="protected judgment surface"):
        replace(_clean_proposal(), target=protected_target)
