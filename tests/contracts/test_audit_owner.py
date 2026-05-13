"""Audit owner + ModificationGate audit-evidence channel contract test (A5/T11).

Implements [`docs/specs/audit-owner.md`](../../docs/specs/audit-owner.md)
Done 检查 阶段 1:

- ``AuditSnapshot`` frozen + threshold_decision enum + risk_score bounds
  fail-loudly
- ``AuditModule`` declares correct slot / owner / dependencies / wiring
- ``evaluate_gate_reasons`` ``audit_snapshot=None, audit_required=False``
  default keeps existing 4 callers byte-equivalent
- ``audit_required=True`` + missing audit_snapshot → BLOCK
- audit hard-block / soft-warn / detected attack → BLOCK
- FinalRolloutConfig.audit defaults to SHADOW
- audit slot registered in DATA_CONTRACT §6
"""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

from volvence_zero.audit import AuditModule, AuditSnapshot
from volvence_zero.audit.types import (
    AUDIT_THRESHOLD_DECISION_VALUES,
    AuditDetectedAttackClass,
)
from volvence_zero.credit.gate import (
    ModificationGate,
    ModificationProposal,
    evaluate_gate_reasons,
)
from volvence_zero.evaluation.types import EvaluationSnapshot
from volvence_zero.integration.final_wiring import FinalRolloutConfig
from volvence_zero.runtime.kernel import WiringLevel


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


def test_audit_snapshot_fields_frozen() -> None:
    expected = (
        "audit_id",
        "timestamp_ms",
        "proposal_id",
        "risk_score",
        "transcript",
        "tool_traces",
        "detected_attack_classes",
        "threshold_decision",
        "description",
    )
    actual = tuple(f.name for f in dataclasses.fields(AuditSnapshot))
    assert actual == expected


def test_threshold_decision_enum_is_closed() -> None:
    """spec §A5.1: closed enum {pass, soft-warn, hard-block}."""
    assert AUDIT_THRESHOLD_DECISION_VALUES == frozenset(
        {"pass", "soft-warn", "hard-block"}
    )


def test_audit_snapshot_rejects_unknown_threshold_decision() -> None:
    """fail-loudly: unknown decision must raise at construction."""
    with pytest.raises(ValueError, match="threshold_decision"):
        AuditSnapshot(
            audit_id="x",
            timestamp_ms=0,
            proposal_id=None,
            risk_score=0.0,
            transcript=(),
            tool_traces=(),
            detected_attack_classes=(),
            threshold_decision="maybe",  # invalid
            description="",
        )


def test_audit_snapshot_rejects_out_of_range_risk_score() -> None:
    """fail-loudly: risk_score must be within [0, 1]."""
    with pytest.raises(ValueError, match="risk_score"):
        AuditSnapshot(
            audit_id="x",
            timestamp_ms=0,
            proposal_id=None,
            risk_score=2.0,  # out of bound
            transcript=(),
            tool_traces=(),
            detected_attack_classes=(),
            threshold_decision="pass",
            description="",
        )


# ---------------------------------------------------------------------------
# AuditModule skeleton
# ---------------------------------------------------------------------------


def test_audit_module_skeleton() -> None:
    assert AuditModule.slot_name == "audit"
    assert AuditModule.owner == "AuditModule"
    assert AuditModule.dependencies == ("evaluation", "credit")
    assert AuditModule.default_wiring_level is WiringLevel.SHADOW


def test_audit_module_can_be_constructed() -> None:
    """Pre-A5 callers should not need any change to import / construct."""
    m = AuditModule()
    assert m.wiring_level is WiringLevel.SHADOW


# ---------------------------------------------------------------------------
# FinalRolloutConfig.audit default
# ---------------------------------------------------------------------------


def test_final_rollout_config_audit_default_shadow() -> None:
    cfg = FinalRolloutConfig()
    assert cfg.audit is WiringLevel.SHADOW


def test_final_rollout_config_level_for_audit_returns_shadow() -> None:
    cfg = FinalRolloutConfig()
    assert cfg.level_for("audit", WiringLevel.DISABLED) is WiringLevel.SHADOW


# ---------------------------------------------------------------------------
# evaluate_gate_reasons backward compatibility
# ---------------------------------------------------------------------------


def _make_clean_proposal() -> ModificationProposal:
    """Construct a proposal that passes every existing two-gate check."""
    return ModificationProposal(
        target="test.target",
        desired_gate=ModificationGate.ONLINE,
        old_value_hash="old",
        new_value_hash="new",
        justification="test",
        is_reversible=True,
        validation_delta=0.05,
        capacity_cost=0.10,
        rollback_evidence="rollback note",
    )


def _make_clean_evaluation() -> EvaluationSnapshot:
    return EvaluationSnapshot(
        turn_scores=(),
        session_scores=(),
        alerts=(),
        description="clean",
        structured_alerts=(),
    )


def test_evaluate_gate_reasons_default_kwargs_byte_equivalent() -> None:
    """A5 spec §A5.3: default ``audit_snapshot=None, audit_required=False``
    keeps existing 4 callers' behaviour byte-equivalent to pre-A5."""
    reasons = evaluate_gate_reasons(
        proposal=_make_clean_proposal(),
        evaluation_snapshot=_make_clean_evaluation(),
    )
    assert reasons == ()


def test_evaluate_gate_reasons_audit_required_missing_snapshot_blocks() -> None:
    """audit_required=True + missing snapshot → BLOCK."""
    reasons = evaluate_gate_reasons(
        proposal=_make_clean_proposal(),
        evaluation_snapshot=_make_clean_evaluation(),
        audit_snapshot=None,
        audit_required=True,
    )
    assert any("audit_snapshot required" in r for r in reasons)


def test_evaluate_gate_reasons_audit_hard_block() -> None:
    audit = AuditSnapshot(
        audit_id="x",
        timestamp_ms=0,
        proposal_id="p1",
        risk_score=0.8,
        transcript=(),
        tool_traces=(),
        detected_attack_classes=(),
        threshold_decision="hard-block",
        description="",
    )
    reasons = evaluate_gate_reasons(
        proposal=_make_clean_proposal(),
        evaluation_snapshot=_make_clean_evaluation(),
        audit_snapshot=audit,
        audit_required=True,
    )
    assert any("audit hard-block" in r for r in reasons)


def test_evaluate_gate_reasons_audit_soft_warn_only_blocks_online() -> None:
    audit = AuditSnapshot(
        audit_id="x",
        timestamp_ms=0,
        proposal_id="p1",
        risk_score=0.4,
        transcript=(),
        tool_traces=(),
        detected_attack_classes=(),
        threshold_decision="soft-warn",
        description="",
    )
    proposal_online = _make_clean_proposal()
    reasons_online = evaluate_gate_reasons(
        proposal=proposal_online,
        evaluation_snapshot=_make_clean_evaluation(),
        audit_snapshot=audit,
        audit_required=True,
    )
    assert any("audit soft-warn" in r for r in reasons_online)

    # background gate is allowed past soft-warn
    proposal_background = dataclasses.replace(
        proposal_online,
        desired_gate=ModificationGate.BACKGROUND,
        validation_delta=0.10,  # meet BACKGROUND margin
    )
    reasons_bg = evaluate_gate_reasons(
        proposal=proposal_background,
        evaluation_snapshot=_make_clean_evaluation(),
        audit_snapshot=audit,
        audit_required=True,
    )
    assert not any("audit soft-warn" in r for r in reasons_bg)


def test_evaluate_gate_reasons_audit_detected_attack_blocks() -> None:
    audit = AuditSnapshot(
        audit_id="x",
        timestamp_ms=0,
        proposal_id="p1",
        risk_score=0.5,
        transcript=(),
        tool_traces=(),
        detected_attack_classes=(
            AuditDetectedAttackClass(
                attack_class="framing_manipulation",
                detected=True,
                confidence=0.9,
                evidence_summary="...",
            ),
        ),
        threshold_decision="pass",
        description="",
    )
    reasons = evaluate_gate_reasons(
        proposal=_make_clean_proposal(),
        evaluation_snapshot=_make_clean_evaluation(),
        audit_snapshot=audit,
        audit_required=True,
    )
    assert any("audit detected attack" in r for r in reasons)


def test_evaluate_gate_reasons_low_confidence_attack_ignored() -> None:
    """A5 §A5.3: only confidence >= 0.7 detected attacks block."""
    audit = AuditSnapshot(
        audit_id="x",
        timestamp_ms=0,
        proposal_id="p1",
        risk_score=0.5,
        transcript=(),
        tool_traces=(),
        detected_attack_classes=(
            AuditDetectedAttackClass(
                attack_class="framing_manipulation",
                detected=True,
                confidence=0.3,  # below threshold
                evidence_summary="...",
            ),
        ),
        threshold_decision="pass",
        description="",
    )
    reasons = evaluate_gate_reasons(
        proposal=_make_clean_proposal(),
        evaluation_snapshot=_make_clean_evaluation(),
        audit_snapshot=audit,
        audit_required=True,
    )
    assert not any("audit detected attack" in r for r in reasons)


def test_evaluate_gate_reasons_audit_ignored_when_not_required() -> None:
    """audit_required=False: even if audit_snapshot is hard-block, no audit
    reason added (current behaviour preserved for existing 4 callers)."""
    audit = AuditSnapshot(
        audit_id="x",
        timestamp_ms=0,
        proposal_id="p1",
        risk_score=0.99,
        transcript=(),
        tool_traces=(),
        detected_attack_classes=(),
        threshold_decision="hard-block",
        description="",
    )
    reasons = evaluate_gate_reasons(
        proposal=_make_clean_proposal(),
        evaluation_snapshot=_make_clean_evaluation(),
        audit_snapshot=audit,
        audit_required=False,
    )
    assert not any("audit" in r for r in reasons)


# ---------------------------------------------------------------------------
# DATA_CONTRACT §6 audit slot registration
# ---------------------------------------------------------------------------


def test_data_contract_registers_audit_slot() -> None:
    """A5 §A5.4 DATA_CONTRACT 注册: §6 主表必须包含 audit slot 行."""
    path = pathlib.Path(__file__).resolve().parents[2] / "docs" / "DATA_CONTRACT.md"
    text = path.read_text(encoding="utf-8")
    # Match the row anywhere in the file
    assert "| `audit` |" in text, (
        "DATA_CONTRACT.md §6 must register `audit` slot row "
        "(see docs/specs/audit-owner.md §DATA_CONTRACT 注册)"
    )
