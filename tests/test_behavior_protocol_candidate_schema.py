"""Packet 2.0: BehaviorProtocolCandidate + ProtocolProvenance schema invariants.

Asserts the schema-level invariants for the DocumentUptake input
contract:

* Default ``requires_review=True`` (LLM-extracted candidates
  cannot bypass review).
* Frozen dataclass (R15 audit trail; reviewers produce new
  protocols, not in-place mutations).
* ``provenance.confidence`` clamped to ``[0, 1]``.
* ``provenance.source_locator`` non-empty.
* ``provenance.extractor_id`` non-empty.
* Provenance.source_kind matches inner protocol.source_kind
  (no silent provenance drift).
* ``ProtocolSourceKind`` includes the new ``MARKDOWN_UPTAKE`` value.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from volvence_zero.behavior_protocol import (
    ActivationConditions,
    BehaviorProtocol,
    BehaviorProtocolCandidate,
    BoundaryContract,
    BoundarySeverity,
    DriveExpectation,
    FailureSignal,
    IdentityAssertion,
    ProtocolProvenance,
    ProtocolSourceKind,
    BehaviorProtocolSignalSource,
    ReviewStatus,
    StrategyPrior,
    SuccessSignal,
    TemporalArc,
)


# ---------------------------------------------------------------------------
# Helpers — minimal valid BehaviorProtocol fixture
# ---------------------------------------------------------------------------


def _minimal_boundary() -> BoundaryContract:
    return BoundaryContract(
        boundary_id="bp-test",
        description="test boundary",
        trigger_reasons=("test_trigger",),
        severity=BoundarySeverity.SOFT_REMIND,
    )


def _minimal_strategy() -> StrategyPrior:
    return StrategyPrior(
        rule_id="strategy-test",
        problem_pattern="test pattern",
        recommended_ordering=("test_step",),
        recommended_pacing="slow",
    )


def _minimal_drive() -> DriveExpectation:
    return DriveExpectation(
        drive_name="connection",
        expected_band=(0.3, 0.7),
    )


def _minimal_protocol(
    *,
    source_kind: ProtocolSourceKind = ProtocolSourceKind.PDF_UPTAKE,
    source_locator: str = "/tmp/test.pdf",
) -> BehaviorProtocol:
    return BehaviorProtocol(
        protocol_id="test:candidate-protocol",
        version="1.0.0",
        advisor_name="test advisor",
        description="minimal candidate protocol",
        source_kind=source_kind,
        source_locator=source_locator,
        identity_assertion=IdentityAssertion(),
        boundary_contracts=(_minimal_boundary(),),
        activation_conditions=ActivationConditions(),
        strategy_priors=(_minimal_strategy(),),
        temporal_arc=TemporalArc(),
        success_signals=(
            SuccessSignal(
                signal_id="success-test",
                description="test success",
                measurable_via=BehaviorProtocolSignalSource.RUPTURE_KIND_FIRED,
            ),
        ),
        failure_signals=(
            FailureSignal(
                signal_id="failure-test",
                description="test failure",
                measurable_via=BehaviorProtocolSignalSource.RUPTURE_KIND_FIRED,
            ),
        ),
        review_status=ReviewStatus.DRAFT,
    )


def _minimal_provenance(
    *,
    source_kind: ProtocolSourceKind = ProtocolSourceKind.PDF_UPTAKE,
    source_locator: str = "/tmp/test.pdf",
) -> ProtocolProvenance:
    return ProtocolProvenance(
        source_kind=source_kind,
        source_locator=source_locator,
        extracted_at_iso="2026-05-11T19:00:00+08:00",
        extractor_id="test-extractor",
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# ProtocolSourceKind enum closed-vocabulary check
# ---------------------------------------------------------------------------


def test_protocol_source_kind_includes_markdown_uptake() -> None:
    """Packet 2.0 added MARKDOWN_UPTAKE for symmetry with PDF_UPTAKE."""
    members = {m.name for m in ProtocolSourceKind}
    assert "MARKDOWN_UPTAKE" in members, members


def test_protocol_source_kind_baseline_members_preserved() -> None:
    """Existing source kinds must NOT regress (closed-enum back-compat)."""
    members = {m.name for m in ProtocolSourceKind}
    for required in (
        "FIXTURE",
        "PDF_UPTAKE",
        "TASK_DESCRIPTION",
        "API_INJECTION",
        "DIRECTORY_SCAN",
    ):
        assert required in members, (required, members)


# ---------------------------------------------------------------------------
# ProtocolProvenance schema
# ---------------------------------------------------------------------------


def test_provenance_constructs_with_minimal_fields() -> None:
    p = _minimal_provenance()
    assert p.source_kind is ProtocolSourceKind.PDF_UPTAKE
    assert p.source_locator == "/tmp/test.pdf"
    assert p.confidence == 0.9


def test_provenance_rejects_empty_source_locator() -> None:
    with pytest.raises(ValueError, match="source_locator"):
        ProtocolProvenance(
            source_kind=ProtocolSourceKind.PDF_UPTAKE,
            source_locator=" ",
            extracted_at_iso="2026-05-11T19:00:00+08:00",
            extractor_id="test",
            confidence=0.5,
        )


def test_provenance_rejects_empty_extractor_id() -> None:
    with pytest.raises(ValueError, match="extractor_id"):
        ProtocolProvenance(
            source_kind=ProtocolSourceKind.PDF_UPTAKE,
            source_locator="/tmp/x",
            extracted_at_iso="2026-05-11T19:00:00+08:00",
            extractor_id="",
            confidence=0.5,
        )


def test_provenance_clamps_confidence_to_unit_interval() -> None:
    with pytest.raises(ValueError, match="confidence"):
        ProtocolProvenance(
            source_kind=ProtocolSourceKind.PDF_UPTAKE,
            source_locator="/tmp/x",
            extracted_at_iso="2026-05-11T19:00:00+08:00",
            extractor_id="test",
            confidence=1.5,
        )
    with pytest.raises(ValueError, match="confidence"):
        ProtocolProvenance(
            source_kind=ProtocolSourceKind.PDF_UPTAKE,
            source_locator="/tmp/x",
            extracted_at_iso="2026-05-11T19:00:00+08:00",
            extractor_id="test",
            confidence=-0.1,
        )


def test_provenance_is_frozen() -> None:
    p = _minimal_provenance()
    with pytest.raises(FrozenInstanceError):
        p.confidence = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BehaviorProtocolCandidate schema
# ---------------------------------------------------------------------------


def test_candidate_default_requires_review() -> None:
    cand = BehaviorProtocolCandidate(
        protocol=_minimal_protocol(),
        provenance=_minimal_provenance(),
    )
    assert cand.requires_review is True
    assert cand.review_evidence == ()


def test_candidate_is_frozen() -> None:
    cand = BehaviorProtocolCandidate(
        protocol=_minimal_protocol(),
        provenance=_minimal_provenance(),
    )
    with pytest.raises(FrozenInstanceError):
        cand.requires_review = False  # type: ignore[misc]


def test_candidate_rejects_provenance_source_kind_mismatch() -> None:
    with pytest.raises(ValueError, match="source_kind"):
        BehaviorProtocolCandidate(
            protocol=_minimal_protocol(
                source_kind=ProtocolSourceKind.PDF_UPTAKE,
                source_locator="/tmp/x.pdf",
            ),
            provenance=_minimal_provenance(
                source_kind=ProtocolSourceKind.MARKDOWN_UPTAKE,
                source_locator="/tmp/x.pdf",
            ),
        )


def test_candidate_rejects_provenance_locator_mismatch() -> None:
    with pytest.raises(ValueError, match="source_locator"):
        BehaviorProtocolCandidate(
            protocol=_minimal_protocol(
                source_kind=ProtocolSourceKind.PDF_UPTAKE,
                source_locator="/tmp/a.pdf",
            ),
            provenance=_minimal_provenance(
                source_kind=ProtocolSourceKind.PDF_UPTAKE,
                source_locator="/tmp/b.pdf",
            ),
        )


def test_candidate_accepts_explicit_review_evidence() -> None:
    cand = BehaviorProtocolCandidate(
        protocol=_minimal_protocol(),
        provenance=_minimal_provenance(),
        requires_review=True,
        review_evidence=(
            "extractor agreed across 3 chunks",
            "boundary count = 4 (matches expected ≥ 3)",
        ),
    )
    assert len(cand.review_evidence) == 2


def test_candidate_pre_reviewed_path_allows_requires_review_false() -> None:
    """API_INJECTION from a trusted upstream may opt out of review.

    Schema does NOT enforce this at construction (it's a policy
    layer concern handled in packet 2.4's load path); this test
    pins that the schema accepts the pre-reviewed shape so the
    upstream can construct it.
    """

    cand = BehaviorProtocolCandidate(
        protocol=_minimal_protocol(
            source_kind=ProtocolSourceKind.API_INJECTION,
            source_locator="api://trusted/v1/protocol",
        ),
        provenance=_minimal_provenance(
            source_kind=ProtocolSourceKind.API_INJECTION,
            source_locator="api://trusted/v1/protocol",
        ),
        requires_review=False,
    )
    assert cand.requires_review is False
