"""Smoke tests for the L2 DATE_PLAUSIBILITY verifier."""

from __future__ import annotations

from lifeform_domain_figure.corpus.provenance import (
    CaptureMethod,
    LegalClearance,
    SourceProvenance,
)
from lifeform_domain_figure.verification.records import CheckKind, Verdict
from lifeform_domain_figure.verification.verifiers.date_plausibility import (
    REVIEWER_ID,
    verify_date_plausibility,
)


_SHA = "1" * 64


def _make_provenance() -> SourceProvenance:
    return SourceProvenance(
        source_id="test-paper-1",
        figure_id="einstein",
        source_url="https://example/test/1",
        license_label="public domain",
        legal_clearance=LegalClearance.PUBLIC_DOMAIN_GLOBAL,
        capture_method=CaptureMethod.TRANSCRIBED,
        captured_by="reviewer-x",
        captured_at_iso="2026-05-10T12:00:00+00:00",
        byte_sha256=_SHA,
        provenance_note="test fixture",
    )


def test_in_range_year_passes() -> None:
    prov = _make_provenance()
    check = verify_date_plausibility(
        prov,
        document_year=1905,
        figure_lifespan=(1879, 1955),
        now_iso="2026-05-10T12:00:00+00:00",
    )
    assert check.check_kind is CheckKind.DATE_PLAUSIBILITY
    assert check.verdict is Verdict.PASS
    assert check.reviewer_id == REVIEWER_ID
    assert check.source_byte_sha256 == _SHA


def test_birth_year_inclusive() -> None:
    check = verify_date_plausibility(
        _make_provenance(),
        document_year=1879,
        figure_lifespan=(1879, 1955),
        now_iso="2026-05-10T12:00:00+00:00",
    )
    assert check.verdict is Verdict.PASS


def test_death_year_inclusive() -> None:
    check = verify_date_plausibility(
        _make_provenance(),
        document_year=1955,
        figure_lifespan=(1879, 1955),
        now_iso="2026-05-10T12:00:00+00:00",
    )
    assert check.verdict is Verdict.PASS


def test_below_lifespan_fails() -> None:
    check = verify_date_plausibility(
        _make_provenance(),
        document_year=1850,
        figure_lifespan=(1879, 1955),
        now_iso="2026-05-10T12:00:00+00:00",
    )
    assert check.verdict is Verdict.FAIL
    assert any("out of range" in piece for piece in check.evidence)


def test_above_lifespan_fails() -> None:
    check = verify_date_plausibility(
        _make_provenance(),
        document_year=2000,
        figure_lifespan=(1879, 1955),
        now_iso="2026-05-10T12:00:00+00:00",
    )
    assert check.verdict is Verdict.FAIL


def test_inverted_lifespan_yields_needs_review() -> None:
    check = verify_date_plausibility(
        _make_provenance(),
        document_year=1900,
        figure_lifespan=(1955, 1879),
        now_iso="2026-05-10T12:00:00+00:00",
    )
    assert check.verdict is Verdict.NEEDS_REVIEW
    assert any("ill-formed" in piece for piece in check.evidence)
