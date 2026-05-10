"""Smoke tests for the L1 -> L2 license bridging (debt #28 修法 5)."""

from __future__ import annotations

import pytest

from lifeform_domain_figure.cleaning.bridging import (
    L1_LICENSE_SENTINEL,
    cleaned_to_source_provenance,
)
from lifeform_domain_figure.cleaning.raw_document import (
    CleanedDocument,
    CleaningOp,
    CleaningOpRecord,
    RawDocument,
)
from lifeform_domain_figure.corpus.provenance import (
    CaptureMethod,
    LegalClearance,
)


_SHA = "9" * 64
_OTHER_SHA = "a" * 64


def _make_raw(license_notice: str = "{{PD-old-100}}") -> RawDocument:
    return RawDocument(
        text="some cleaned-precursor text",
        parser_version="test:1",
        layout_quality=1.0,
        ocr_confidence=1.0,
        encoding_detected="utf-8",
        language_detected="en",
        license_notice=license_notice,
        raw_sha256=_SHA,
    )


def _make_cleaned(*, sha: str = _SHA) -> CleanedDocument:
    return CleanedDocument(
        text="some cleaned text",
        raw_sha256=sha,
        cleaner_pipeline_version=1,
        cleaning_log=(
            CleaningOpRecord(
                op=CleaningOp.WHITESPACE_NORMALIZE,
                op_version="1",
                chars_before=20,
                chars_after=17,
            ),
        ),
        parser_version="test:1",
    )


def _common_kwargs() -> dict:
    return {
        "source_id": "test-prov-1",
        "figure_id": "einstein",
        "source_url": "https://example/a",
        "legal_clearance": LegalClearance.PUBLIC_DOMAIN_GLOBAL,
        "capture_method": CaptureMethod.TRANSCRIBED,
        "captured_by": "reviewer-x",
        "captured_at_iso": "2026-05-10T12:00:00+00:00",
        "provenance_note": "test fixture",
    }


def test_byte_sha_propagates_from_raw() -> None:
    prov = cleaned_to_source_provenance(
        _make_cleaned(), _make_raw(), **_common_kwargs()
    )
    assert prov.byte_sha256 == _SHA


def test_license_label_uses_raw_license_notice_by_default() -> None:
    prov = cleaned_to_source_provenance(
        _make_cleaned(), _make_raw("{{PD-old-100}}"), **_common_kwargs()
    )
    assert prov.license_label == "{{PD-old-100}}"


def test_license_label_override_wins() -> None:
    prov = cleaned_to_source_provenance(
        _make_cleaned(),
        _make_raw("{{PD-old-100}}"),
        license_label_override="curator-supplied PD-100 attestation",
        **_common_kwargs(),
    )
    assert prov.license_label == "curator-supplied PD-100 attestation"


def test_empty_license_notice_yields_sentinel() -> None:
    prov = cleaned_to_source_provenance(
        _make_cleaned(),
        _make_raw(license_notice=""),
        **_common_kwargs(),
    )
    assert prov.license_label == L1_LICENSE_SENTINEL


def test_mismatched_sha_refused() -> None:
    cleaned = _make_cleaned(sha=_OTHER_SHA)
    raw = _make_raw()
    with pytest.raises(ValueError, match="does not match"):
        cleaned_to_source_provenance(cleaned, raw, **_common_kwargs())
