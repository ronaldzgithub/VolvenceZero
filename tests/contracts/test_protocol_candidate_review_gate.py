"""Packet 2.4: candidate review gate contract tests.

Asserts the review-level enforcement behaviour around
``BehaviorProtocolCandidate``:

* ``required_review_level`` derives the right level from
  candidate content shape (boundary → L4, strategy → L3,
  knowledge/case-only → L2, identity-only → L1).
* ``approve_candidate`` rejects under-authorised reviewers and
  produces a fresh ``BehaviorProtocol`` with revised
  ``review_status``.
* ``reject_candidate`` produces an audit record without
  silently dropping anything.
* ``ProtocolRegistryModule.load_protocol_candidate`` blocks
  DRAFT candidates with ``requires_review=True`` unless
  ``force=True``.
* Approved candidates flow through the normal compile path
  (lineage prefix appears in application owners).
"""

from __future__ import annotations

import pytest

from lifeform_protocol_runtime.document_uptake.extraction import (
    MockLlmJsonClient,
    extract_protocol_candidate,
)
from lifeform_protocol_runtime.document_uptake.ingestion import chunk_document
from lifeform_protocol_runtime.document_uptake.review import (
    CandidateRejection,
    approve_candidate,
    reject_candidate,
    required_review_level,
)
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState
from volvence_zero.behavior_protocol import (
    BehaviorProtocolCandidate,
    ProtocolProvenance,
    ProtocolSourceKind,
    ReviewLevel,
    ReviewStatus,
)
from volvence_zero.protocol_runtime import ProtocolRegistryModule


# ---------------------------------------------------------------------------
# Test fixtures (reuse 2.3 mock LLM)
# ---------------------------------------------------------------------------


_IDENTITY = {
    "advisor_name": "test-advisor",
    "description": "test description",
    "identity_traits": ["t1"],
    "regime_compatibility": [],
}

_BOUNDARY_FULL = {
    "boundaries": [
        {
            "boundary_id": "bp-test",
            "description": "test boundary",
            "trigger_reasons": ["boundary_violation_fired"],
            "blocked_topics": ["topic"],
            "refer_out_required": False,
            "severity": "soft_remind",
        }
    ]
}

_STRATEGY_FULL = {
    "strategies": [
        {
            "rule_id": "strategy-test",
            "problem_pattern": "test",
            "recommended_ordering": ["step1", "step2"],
            "recommended_pacing": "moderate",
            "avoid_patterns": [],
            "applicability_phase": [],
        }
    ],
    "knowledge_seeds": [
        {
            "seed_id": "seed-test",
            "domain": "test",
            "title": "test",
            "summary": "test",
            "snippet": "test",
            "evidence_locator": "test",
            "confidence": 0.8,
        }
    ],
    "cases": [],
}


def _make_candidate(
    *,
    boundary: dict | None = None,
    strategy: dict | None = None,
) -> BehaviorProtocolCandidate:
    """Run a deterministic mock extraction to get a candidate."""
    chunks = chunk_document(
        "Test document body.",
        source_locator="/tmp/test.pdf",
        max_tokens=2048,
    )
    client = MockLlmJsonClient(
        identity=_IDENTITY,
        boundary=boundary if boundary is not None else _BOUNDARY_FULL,
        strategy=strategy if strategy is not None else _STRATEGY_FULL,
    )
    return extract_protocol_candidate(
        chunks,
        llm_client=client,
        source_locator="/tmp/test.pdf",
    )


# ---------------------------------------------------------------------------
# required_review_level
# ---------------------------------------------------------------------------


def test_review_level_l4_when_candidate_has_boundary() -> None:
    candidate = _make_candidate()
    assert required_review_level(candidate) is ReviewLevel.L4


def test_review_level_l3_when_strategy_only() -> None:
    candidate = _make_candidate(boundary={"boundaries": []})
    # No boundary, has strategy → L3
    assert required_review_level(candidate) is ReviewLevel.L3


def test_review_level_l2_when_knowledge_or_case_only() -> None:
    """Strategy-less, boundary-less, knowledge-only → L2."""
    candidate = _make_candidate(
        boundary={"boundaries": []},
        strategy={
            "strategies": [],
            "knowledge_seeds": _STRATEGY_FULL["knowledge_seeds"],
            "cases": [],
        },
    )
    assert required_review_level(candidate) is ReviewLevel.L2


# ---------------------------------------------------------------------------
# approve_candidate
# ---------------------------------------------------------------------------


def test_approve_candidate_returns_fresh_protocol_in_shadow_status() -> None:
    candidate = _make_candidate()
    approved, approval = approve_candidate(
        candidate,
        reviewer_id="ops-admin",
        evidence=("manual review pass",),
        minimum_level=ReviewLevel.L4,
    )

    assert approved.review_status is ReviewStatus.SHADOW
    assert approved.protocol_id == candidate.protocol.protocol_id
    assert len(approved.revision_log) == 1
    assert approved.revision_log[0].revised_by == "ops-admin"
    assert approval.reviewer_id == "ops-admin"
    assert approval.review_level_required is ReviewLevel.L4


def test_approve_candidate_rejects_under_authorised_reviewer() -> None:
    candidate = _make_candidate()
    with pytest.raises(PermissionError, match="l4"):
        approve_candidate(
            candidate,
            reviewer_id="ops-junior",
            minimum_level=ReviewLevel.L2,  # less than required L4
        )


def test_approve_candidate_with_no_minimum_level_passes() -> None:
    """Omit minimum_level → trust caller; required level is just an audit."""
    candidate = _make_candidate()
    approved, approval = approve_candidate(
        candidate,
        reviewer_id="ops-admin",
    )
    assert approved.review_status is ReviewStatus.SHADOW
    assert approval.review_level_required is ReviewLevel.L4


def test_approve_candidate_target_active_status() -> None:
    """Caller can request ACTIVE review_status (e.g. after second sign-off)."""
    candidate = _make_candidate()
    approved, _ = approve_candidate(
        candidate,
        reviewer_id="ops-admin",
        target_status=ReviewStatus.ACTIVE,
    )
    assert approved.review_status is ReviewStatus.ACTIVE


def test_approve_candidate_rejects_empty_reviewer() -> None:
    candidate = _make_candidate()
    with pytest.raises(ValueError, match="reviewer_id"):
        approve_candidate(candidate, reviewer_id=" ")


# ---------------------------------------------------------------------------
# reject_candidate
# ---------------------------------------------------------------------------


def test_reject_candidate_produces_audit_record() -> None:
    candidate = _make_candidate()
    rejection = reject_candidate(
        candidate,
        reviewer_id="ops-admin",
        reason="boundary description too vague",
    )
    assert isinstance(rejection, CandidateRejection)
    assert rejection.reviewer_id == "ops-admin"
    assert rejection.reason == "boundary description too vague"
    assert rejection.candidate_id == candidate.protocol.protocol_id


def test_reject_candidate_rejects_empty_reason() -> None:
    candidate = _make_candidate()
    with pytest.raises(ValueError, match="reason"):
        reject_candidate(candidate, reviewer_id="x", reason="")


# ---------------------------------------------------------------------------
# ProtocolRegistryModule.load_protocol_candidate
# ---------------------------------------------------------------------------


def test_load_candidate_blocks_draft_with_requires_review() -> None:
    """An LLM-extracted candidate (DRAFT + requires_review=True)
    must NOT load without going through review first."""
    candidate = _make_candidate()
    assert candidate.requires_review is True
    assert candidate.protocol.review_status is ReviewStatus.DRAFT

    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)

    with pytest.raises(PermissionError, match="review"):
        module.load_protocol_candidate(candidate)

    # State should be untouched.
    assert state.boundary_prior_hints == ()


def test_load_candidate_force_override_bypasses_review() -> None:
    """``force=True`` bypasses review for emergency paths."""
    candidate = _make_candidate()
    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)

    module.load_protocol_candidate(candidate, force=True)
    assert any(
        h.hint_id.startswith("protocol:")
        for h in state.boundary_prior_hints
    )


def test_load_candidate_after_approval_succeeds() -> None:
    """The intended path: extract → approve → load."""
    candidate = _make_candidate()
    approved, _ = approve_candidate(
        candidate,
        reviewer_id="ops-admin",
        evidence=("manual review pass",),
    )
    # Wrap the approved protocol in a fresh candidate-style envelope
    # for the load_protocol_candidate API.
    approved_candidate = BehaviorProtocolCandidate(
        protocol=approved,
        provenance=candidate.provenance,
        requires_review=False,  # already reviewed
    )
    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    module.load_protocol_candidate(approved_candidate)

    assert any(
        h.hint_id.startswith("protocol:")
        for h in state.boundary_prior_hints
    )


def test_load_candidate_pre_reviewed_no_review_path() -> None:
    """``requires_review=False`` (e.g. trusted API injection) loads directly."""
    inner = _make_candidate().protocol
    pre_reviewed = BehaviorProtocolCandidate(
        protocol=inner,
        provenance=ProtocolProvenance(
            source_kind=inner.source_kind,
            source_locator=inner.source_locator,
            extracted_at_iso="2026-05-11T19:00:00+08:00",
            extractor_id="api-injection",
            confidence=1.0,
        ),
        requires_review=False,
    )
    state = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=state)
    module.load_protocol_candidate(pre_reviewed)
    assert any(
        h.hint_id.startswith("protocol:")
        for h in state.boundary_prior_hints
    )
