"""Smoke tests for the L2 LICENSE_PAGE_LEVEL verifier."""

from __future__ import annotations

from lifeform_domain_figure.cleaning.bridging import L1_LICENSE_SENTINEL
from lifeform_domain_figure.corpus.provenance import (
    CaptureMethod,
    LegalClearance,
    SourceProvenance,
)
from lifeform_domain_figure.verification.records import CheckKind, Verdict
from lifeform_domain_figure.verification.verifiers.license_page_level import (
    REVIEWER_ID,
    verify_license_page_level,
)


_SHA = "2" * 64


def _make_provenance(
    *,
    license_label: str,
    legal_clearance: LegalClearance,
) -> SourceProvenance:
    return SourceProvenance(
        source_id="test-license-1",
        figure_id="einstein",
        source_url="https://example/test/1",
        license_label=license_label,
        legal_clearance=legal_clearance,
        capture_method=CaptureMethod.TRANSCRIBED,
        captured_by="reviewer-x",
        captured_at_iso="2026-05-10T12:00:00+00:00",
        byte_sha256=_SHA,
        provenance_note="test fixture",
    )


def test_pd_label_with_pd_global_passes() -> None:
    prov = _make_provenance(
        license_label="{{PD-old-100}} - public domain",
        legal_clearance=LegalClearance.PUBLIC_DOMAIN_GLOBAL,
    )
    check = verify_license_page_level(prov, now_iso="2026-05-10T12:00:00+00:00")
    assert check.check_kind is CheckKind.LICENSE_PAGE_LEVEL
    assert check.verdict is Verdict.PASS
    assert check.reviewer_id == REVIEWER_ID


def test_all_rights_reserved_with_pd_fails() -> None:
    prov = _make_provenance(
        license_label="Copyright 2010, All rights reserved",
        legal_clearance=LegalClearance.PUBLIC_DOMAIN_GLOBAL,
    )
    check = verify_license_page_level(prov, now_iso="2026-05-10T12:00:00+00:00")
    assert check.verdict is Verdict.FAIL
    assert any("hard-conflict" in piece for piece in check.evidence)


def test_l1_sentinel_yields_needs_review() -> None:
    prov = _make_provenance(
        license_label=L1_LICENSE_SENTINEL,
        legal_clearance=LegalClearance.PUBLIC_DOMAIN_GLOBAL,
    )
    check = verify_license_page_level(prov, now_iso="2026-05-10T12:00:00+00:00")
    assert check.verdict is Verdict.NEEDS_REVIEW


def test_cc_by_with_licensed_open_passes() -> None:
    prov = _make_provenance(
        license_label="Released under CC-BY 4.0",
        legal_clearance=LegalClearance.LICENSED_OPEN,
    )
    check = verify_license_page_level(prov, now_iso="2026-05-10T12:00:00+00:00")
    assert check.verdict is Verdict.PASS


def test_tenant_declared_always_passes() -> None:
    prov = _make_provenance(
        license_label="Internal corpus, tenant attestation on file",
        legal_clearance=LegalClearance.TENANT_DECLARED,
    )
    check = verify_license_page_level(prov, now_iso="2026-05-10T12:00:00+00:00")
    assert check.verdict is Verdict.PASS
    assert any("tenant attestation" in piece for piece in check.evidence)


def test_unrelated_label_yields_needs_review() -> None:
    prov = _make_provenance(
        license_label="some label that matches no list",
        legal_clearance=LegalClearance.PUBLIC_DOMAIN_REGIONAL,
    )
    check = verify_license_page_level(prov, now_iso="2026-05-10T12:00:00+00:00")
    assert check.verdict is Verdict.NEEDS_REVIEW
