"""Packet 2.3: LLM-driven document extraction tests.

Use ``MockLlmJsonClient`` (deterministic, no network) to feed
canned per-family responses through
``extract_protocol_candidate`` and verify the merge / schema /
candidate-shape behaviour.
"""

from __future__ import annotations

import pytest

from lifeform_protocol_runtime.document_uptake.extraction import (
    MockLlmJsonClient,
    extract_protocol_candidate,
)
from lifeform_protocol_runtime.document_uptake.ingestion import (
    DocumentChunk,
    chunk_document,
)
from volvence_zero.behavior_protocol import (
    BehaviorProtocolCandidate,
    BoundarySeverity,
    ProtocolSourceKind,
    ReviewStatus,
)


# ---------------------------------------------------------------------------
# Canned LLM responses for cheng_laoshi-shape extraction
# ---------------------------------------------------------------------------


_IDENTITY_RESPONSE = {
    "advisor_name": "谌老师",
    "description": "Private-domain growth advisor for parents.",
    "identity_traits": ["warm_peer_register", "long_horizon"],
    "regime_compatibility": ["emotional_support", "guided_exploration"],
}

_BOUNDARY_RESPONSE = {
    "boundaries": [
        {
            "boundary_id": "no-hard-sell",
            "description": "Never push purchases in the first 7 days.",
            "trigger_reasons": ["boundary_violation_fired"],
            "blocked_topics": ["sales", "discount", "limited offer"],
            "refer_out_required": False,
            "severity": "soft_remind",
        },
        {
            "boundary_id": "no-overclaim",
            "description": "Never claim medical effects without disclaimer.",
            "trigger_reasons": ["boundary_violation_fired"],
            "blocked_topics": ["cure", "guaranteed result"],
            "refer_out_required": True,
            "severity": "hard_block",
        },
    ]
}

_STRATEGY_RESPONSE = {
    "strategies": [
        {
            "rule_id": "rapport-empathy",
            "problem_pattern": "User shares emotional struggle",
            "recommended_ordering": [
                "acknowledge_pressure",
                "render_emotional_resonance",
                "soft_question",
            ],
            "recommended_pacing": "slow",
            "avoid_patterns": ["solution_pitch", "comparative_claim"],
            "applicability_phase": ["day1", "day2", "day3"],
        },
        {
            "rule_id": "funnel-nutrition",
            "problem_pattern": "User asks about child nutrition",
            "recommended_ordering": [
                "ask_age_band",
                "share_typed_evidence",
                "schedule_follow_up",
            ],
            "recommended_pacing": "moderate",
            "avoid_patterns": ["medical_opinion", "brand_recommendation"],
            "applicability_phase": ["day4", "day5", "day6"],
        },
    ],
    "knowledge_seeds": [
        {
            "seed_id": "growth-window-3-12",
            "topic": "growth_window",
            "summary": "Children aged 3-12 have variable growth rates...",
            "jurisdiction_tags": ["cn"],
        }
    ],
    "cases": [
        {
            "case_id": "case-anxious-mom",
            "title": "Anxious-mom-day-3-pivot",
            "transcript_summary": "Mom expressed feeling overwhelmed; advisor pivoted from KB to empathy.",
            "lesson": "Empathy first when emotional load is high.",
        }
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunks(text: str = "Sample doc content.") -> tuple[DocumentChunk, ...]:
    """Build deterministic 1-chunk fixture for extraction tests."""
    return chunk_document(text, source_locator="/tmp/sample.pdf", max_tokens=2048)


def _full_mock_client() -> MockLlmJsonClient:
    return MockLlmJsonClient(
        identity=_IDENTITY_RESPONSE,
        boundary=_BOUNDARY_RESPONSE,
        strategy=_STRATEGY_RESPONSE,
    )


# ---------------------------------------------------------------------------
# Schema-shape outputs
# ---------------------------------------------------------------------------


def test_extract_returns_candidate_with_review_required() -> None:
    chunks = _make_chunks()
    candidate = extract_protocol_candidate(
        chunks,
        llm_client=_full_mock_client(),
        source_locator="/tmp/sample.pdf",
    )
    assert isinstance(candidate, BehaviorProtocolCandidate)
    assert candidate.requires_review is True
    assert candidate.protocol.review_status is ReviewStatus.DRAFT


def test_extract_inner_protocol_carries_source_metadata() -> None:
    chunks = _make_chunks()
    candidate = extract_protocol_candidate(
        chunks,
        llm_client=_full_mock_client(),
        source_locator="/tmp/sample.pdf",
        source_kind=ProtocolSourceKind.PDF_UPTAKE,
    )
    assert candidate.protocol.source_kind is ProtocolSourceKind.PDF_UPTAKE
    assert candidate.protocol.source_locator == "/tmp/sample.pdf"
    # Provenance must mirror the protocol's source fields.
    assert candidate.provenance.source_kind is ProtocolSourceKind.PDF_UPTAKE
    assert candidate.provenance.source_locator == "/tmp/sample.pdf"


def test_extract_populates_identity_traits() -> None:
    candidate = extract_protocol_candidate(
        _make_chunks(),
        llm_client=_full_mock_client(),
        source_locator="/tmp/sample.pdf",
    )
    traits = candidate.protocol.identity_assertion.requires_self_traits
    assert "warm_peer_register" in traits
    assert "long_horizon" in traits


def test_extract_populates_boundary_contracts() -> None:
    candidate = extract_protocol_candidate(
        _make_chunks(),
        llm_client=_full_mock_client(),
        source_locator="/tmp/sample.pdf",
    )
    boundary_ids = {b.boundary_id for b in candidate.protocol.boundary_contracts}
    assert boundary_ids == {"no-hard-sell", "no-overclaim"}
    severities = {
        b.boundary_id: b.severity
        for b in candidate.protocol.boundary_contracts
    }
    assert severities["no-hard-sell"] is BoundarySeverity.SOFT_REMIND
    assert severities["no-overclaim"] is BoundarySeverity.HARD_BLOCK


def test_extract_populates_strategy_priors() -> None:
    candidate = extract_protocol_candidate(
        _make_chunks(),
        llm_client=_full_mock_client(),
        source_locator="/tmp/sample.pdf",
    )
    rule_ids = {s.rule_id for s in candidate.protocol.strategy_priors}
    assert rule_ids == {"rapport-empathy", "funnel-nutrition"}


def test_extract_populates_knowledge_and_cases() -> None:
    candidate = extract_protocol_candidate(
        _make_chunks(),
        llm_client=_full_mock_client(),
        source_locator="/tmp/sample.pdf",
    )
    seed_ids = {s.seed_id for s in candidate.protocol.knowledge_seeds}
    case_ids = {c.case_id for c in candidate.protocol.signature_cases}
    assert "growth-window-3-12" in seed_ids
    assert "case-anxious-mom" in case_ids


def test_extract_review_evidence_records_counts() -> None:
    candidate = extract_protocol_candidate(
        _make_chunks(),
        llm_client=_full_mock_client(),
        source_locator="/tmp/sample.pdf",
    )
    blob = " ".join(candidate.review_evidence)
    assert "boundaries=2" in blob
    assert "strategies=2" in blob
    assert "knowledge_seeds=1" in blob
    assert "cases=1" in blob


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------


def test_extract_rejects_empty_chunks() -> None:
    with pytest.raises(ValueError, match="non-empty chunks"):
        extract_protocol_candidate(
            (),
            llm_client=_full_mock_client(),
            source_locator="/tmp/sample.pdf",
        )


def test_extract_rejects_zero_actionable_extraction() -> None:
    """If the LLM returns no boundaries AND no strategies, raise."""
    blank_client = MockLlmJsonClient(
        identity={"advisor_name": "x", "description": "y"},
        boundary={"boundaries": []},
        strategy={"strategies": [], "knowledge_seeds": [], "cases": []},
    )
    with pytest.raises(ValueError, match="zero boundaries"):
        extract_protocol_candidate(
            _make_chunks(),
            llm_client=blank_client,
            source_locator="/tmp/sample.pdf",
        )


def test_extract_dedupes_same_id_across_chunks() -> None:
    """Multi-chunk extraction merges by id; same id from chunk 2 is ignored."""
    chunks = chunk_document(
        "\n\n".join([f"chunk {i} body content here." for i in range(3)]),
        source_locator="/tmp/sample.pdf",
        max_tokens=8,
    )
    # Same response for every chunk → after merge we should still
    # see exactly the canned boundary / strategy IDs (deduped).
    candidate = extract_protocol_candidate(
        chunks,
        llm_client=_full_mock_client(),
        source_locator="/tmp/sample.pdf",
    )
    boundary_ids = [b.boundary_id for b in candidate.protocol.boundary_contracts]
    rule_ids = [s.rule_id for s in candidate.protocol.strategy_priors]
    # No duplicates.
    assert len(boundary_ids) == len(set(boundary_ids))
    assert len(rule_ids) == len(set(rule_ids))


def test_extract_warns_on_boundary_missing_trigger_reasons() -> None:
    """A boundary item without trigger_reasons → skipped + warning logged."""
    bad_client = MockLlmJsonClient(
        identity=_IDENTITY_RESPONSE,
        boundary={
            "boundaries": [
                {
                    "boundary_id": "missing-trigger",
                    "description": "no triggers here",
                    "trigger_reasons": [],
                    "severity": "hard_block",
                }
            ]
        },
        strategy=_STRATEGY_RESPONSE,
    )
    candidate = extract_protocol_candidate(
        _make_chunks(),
        llm_client=bad_client,
        source_locator="/tmp/sample.pdf",
    )
    boundary_ids = {b.boundary_id for b in candidate.protocol.boundary_contracts}
    assert "missing-trigger" not in boundary_ids
    # Warning surfaced in review_evidence
    assert any("missing-trigger" in w for w in candidate.review_evidence)


def test_extract_synthesises_pe_signals() -> None:
    """Every extracted protocol must declare success+failure signals."""
    candidate = extract_protocol_candidate(
        _make_chunks(),
        llm_client=_full_mock_client(),
        source_locator="/tmp/sample.pdf",
    )
    assert len(candidate.protocol.success_signals) >= 1
    assert len(candidate.protocol.failure_signals) >= 1


def test_extract_confidence_higher_when_no_warnings() -> None:
    candidate_clean = extract_protocol_candidate(
        _make_chunks(),
        llm_client=_full_mock_client(),
        source_locator="/tmp/sample.pdf",
    )
    bad_client = MockLlmJsonClient(
        identity=_IDENTITY_RESPONSE,
        boundary={
            "boundaries": [
                {
                    "boundary_id": "missing-trigger",
                    "description": "no triggers",
                    "trigger_reasons": [],
                    "severity": "hard_block",
                },
                _BOUNDARY_RESPONSE["boundaries"][0],
            ]
        },
        strategy=_STRATEGY_RESPONSE,
    )
    candidate_warn = extract_protocol_candidate(
        _make_chunks(),
        llm_client=bad_client,
        source_locator="/tmp/sample.pdf",
    )
    assert candidate_clean.provenance.confidence > candidate_warn.provenance.confidence
