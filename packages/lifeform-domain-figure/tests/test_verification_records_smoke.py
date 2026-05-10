"""Smoke tests for L2 verification record schema (debt #28)."""

from __future__ import annotations

import pytest

from lifeform_domain_figure.verification.records import (
    CheckKind,
    IMPLEMENTED_CHECK_KINDS,
    Verdict,
    VerificationCheck,
)


_DUMMY_SHA = "0" * 64


def test_check_kind_has_seven_values() -> None:
    assert len(list(CheckKind)) == 7
    assert {kind.value for kind in CheckKind} == {
        "date_plausibility",
        "license_page_level",
        "cross_source_byte",
        "identity_disambiguation",
        "authorship_attribution",
        "version_reconciliation",
        "translation_lineage",
    }


def test_implemented_check_kinds_is_full_set() -> None:
    """As of debt #28 L2 second batch (2026-05-10), all 7 kinds are implemented."""

    assert IMPLEMENTED_CHECK_KINDS == frozenset(CheckKind)
    assert len(IMPLEMENTED_CHECK_KINDS) == 7


def test_verdict_has_three_values() -> None:
    assert {v.value for v in Verdict} == {"pass", "fail", "needs_review"}


def test_verification_check_round_trip() -> None:
    check = VerificationCheck(
        check_kind=CheckKind.DATE_PLAUSIBILITY,
        verdict=Verdict.PASS,
        evidence=("year=1905", "lifespan=[1879,1955]"),
        reviewer_id="auto:date_plausibility:1",
        reviewed_at_iso="2026-05-10T12:00:00+00:00",
        source_byte_sha256=_DUMMY_SHA,
    )
    assert check.check_kind is CheckKind.DATE_PLAUSIBILITY
    assert check.verdict is Verdict.PASS
    assert check.source_byte_sha256 == _DUMMY_SHA


def test_verification_check_rejects_empty_evidence() -> None:
    with pytest.raises(ValueError, match="evidence must be a non-empty tuple"):
        VerificationCheck(
            check_kind=CheckKind.DATE_PLAUSIBILITY,
            verdict=Verdict.PASS,
            evidence=(),
            reviewer_id="auto:date_plausibility:1",
            reviewed_at_iso="2026-05-10T12:00:00+00:00",
            source_byte_sha256=_DUMMY_SHA,
        )


def test_verification_check_rejects_blank_evidence_item() -> None:
    with pytest.raises(ValueError, match="must be a non-empty"):
        VerificationCheck(
            check_kind=CheckKind.DATE_PLAUSIBILITY,
            verdict=Verdict.PASS,
            evidence=("good", "  "),
            reviewer_id="auto:date_plausibility:1",
            reviewed_at_iso="2026-05-10T12:00:00+00:00",
            source_byte_sha256=_DUMMY_SHA,
        )


def test_verification_check_rejects_bad_reviewer_id_prefix() -> None:
    with pytest.raises(ValueError, match="prefix must be 'auto' or 'human'"):
        VerificationCheck(
            check_kind=CheckKind.DATE_PLAUSIBILITY,
            verdict=Verdict.PASS,
            evidence=("ok",),
            reviewer_id="bot:something:1",
            reviewed_at_iso="2026-05-10T12:00:00+00:00",
            source_byte_sha256=_DUMMY_SHA,
        )


def test_verification_check_rejects_bad_reviewer_id_no_colon() -> None:
    with pytest.raises(ValueError, match="must be of the form"):
        VerificationCheck(
            check_kind=CheckKind.DATE_PLAUSIBILITY,
            verdict=Verdict.PASS,
            evidence=("ok",),
            reviewer_id="autoonly",
            reviewed_at_iso="2026-05-10T12:00:00+00:00",
            source_byte_sha256=_DUMMY_SHA,
        )


def test_verification_check_rejects_short_sha() -> None:
    with pytest.raises(ValueError, match="64-char hex sha256"):
        VerificationCheck(
            check_kind=CheckKind.DATE_PLAUSIBILITY,
            verdict=Verdict.PASS,
            evidence=("ok",),
            reviewer_id="auto:date_plausibility:1",
            reviewed_at_iso="2026-05-10T12:00:00+00:00",
            source_byte_sha256="abc",
        )


def test_human_reviewer_id_accepted() -> None:
    check = VerificationCheck(
        check_kind=CheckKind.DATE_PLAUSIBILITY,
        verdict=Verdict.FAIL,
        evidence=("manual override",),
        reviewer_id="human:reviewer-x",
        reviewed_at_iso="2026-05-10T12:00:00+00:00",
        source_byte_sha256=_DUMMY_SHA,
    )
    assert check.reviewer_id.startswith("human:")
