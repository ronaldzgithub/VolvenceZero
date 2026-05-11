"""Packet 3.0: ProtocolRevisionProposal + ProposalEvidence schema tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from volvence_zero.behavior_protocol import (
    ProposalEvidence,
    ProtocolRevisionChangeKind,
    ProtocolRevisionProposal,
    ProtocolRevisionTargetField,
    ReviewLevel,
)


def _evidence() -> ProposalEvidence:
    return ProposalEvidence(
        observation_window_turns=10,
        pe_signature="signed_reward_decay@5_turns_below_-0.3",
        summary="strategy weight_decay proposed after 5 negative-PE turns",
    )


# ---------------------------------------------------------------------------
# ProposalEvidence
# ---------------------------------------------------------------------------


def test_evidence_constructs() -> None:
    e = _evidence()
    assert e.observation_window_turns == 10


def test_evidence_rejects_zero_window() -> None:
    with pytest.raises(ValueError, match="observation_window_turns"):
        ProposalEvidence(
            observation_window_turns=0,
            pe_signature="x",
            summary="y",
        )


def test_evidence_rejects_empty_summary() -> None:
    with pytest.raises(ValueError, match="summary"):
        ProposalEvidence(
            observation_window_turns=1,
            pe_signature="x",
            summary=" ",
        )


def test_evidence_is_frozen() -> None:
    e = _evidence()
    with pytest.raises(FrozenInstanceError):
        e.observation_window_turns = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ProtocolRevisionProposal
# ---------------------------------------------------------------------------


def test_proposal_constructs_minimal() -> None:
    p = ProtocolRevisionProposal(
        proposal_id="prop:1",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id="rapport-empathy",
        change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
        evidence=_evidence(),
    )
    assert p.proposal_id == "prop:1"
    assert p.required_review_level is ReviewLevel.L3
    assert p.proposed_payload is None


def test_proposal_carries_payload_dict() -> None:
    p = ProtocolRevisionProposal(
        proposal_id="prop:1",
        target_protocol_id="growth_advisor:cheng-laoshi",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id="rapport-empathy",
        change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
        evidence=_evidence(),
        proposed_payload={"weight_multiplier": 0.5},
    )
    assert p.proposed_payload == {"weight_multiplier": 0.5}


def test_proposal_rejects_empty_proposal_id() -> None:
    with pytest.raises(ValueError, match="proposal_id"):
        ProtocolRevisionProposal(
            proposal_id="",
            target_protocol_id="x",
            target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
            target_entry_id="y",
            change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
            evidence=_evidence(),
        )


def test_proposal_rejects_empty_target_protocol_id() -> None:
    with pytest.raises(ValueError, match="target_protocol_id"):
        ProtocolRevisionProposal(
            proposal_id="prop:1",
            target_protocol_id=" ",
            target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
            target_entry_id="y",
            change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
            evidence=_evidence(),
        )


def test_proposal_rejects_empty_target_entry_id() -> None:
    with pytest.raises(ValueError, match="target_entry_id"):
        ProtocolRevisionProposal(
            proposal_id="prop:1",
            target_protocol_id="x",
            target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
            target_entry_id=" ",
            change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
            evidence=_evidence(),
        )


def test_proposal_is_frozen() -> None:
    p = ProtocolRevisionProposal(
        proposal_id="prop:1",
        target_protocol_id="x",
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id="y",
        change_kind=ProtocolRevisionChangeKind.DEACTIVATE,
        evidence=_evidence(),
    )
    with pytest.raises(FrozenInstanceError):
        p.proposal_id = "new"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Enum closed-vocabulary
# ---------------------------------------------------------------------------


def test_target_field_enum_includes_expected_members() -> None:
    members = {m.name for m in ProtocolRevisionTargetField}
    assert members == {
        "STRATEGY_PRIOR",
        "KNOWLEDGE_SEED",
        "SIGNATURE_CASE",
        "BOUNDARY_CONTRACT",
        "IDENTITY_ASSERTION",
    }, members


def test_change_kind_enum_includes_expected_members() -> None:
    members = {m.name for m in ProtocolRevisionChangeKind}
    assert members == {
        "WEIGHT_DECAY",
        "DEACTIVATE",
        "REPLACE_TEXT",
        "ARCHIVE",
    }, members
