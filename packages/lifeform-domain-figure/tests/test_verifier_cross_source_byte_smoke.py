"""Smoke tests for the L2 CROSS_SOURCE_BYTE verifier."""

from __future__ import annotations

import pytest

from lifeform_domain_figure.corpus.provenance import (
    CaptureMethod,
    LegalClearance,
    SourceProvenance,
)
from lifeform_domain_figure.verification.records import CheckKind, Verdict
from lifeform_domain_figure.verification.verifiers.cross_source_byte import (
    REVIEWER_ID,
    verify_cross_source_byte,
)


def _make(*, source_id: str, sha: str, source_url: str = "https://example/x") -> SourceProvenance:
    return SourceProvenance(
        source_id=source_id,
        figure_id="einstein",
        source_url=source_url,
        license_label="public domain",
        legal_clearance=LegalClearance.PUBLIC_DOMAIN_GLOBAL,
        capture_method=CaptureMethod.TRANSCRIBED,
        captured_by="reviewer-x",
        captured_at_iso="2026-05-10T12:00:00+00:00",
        byte_sha256=sha,
        provenance_note="test fixture",
    )


def test_singleton_group_passes() -> None:
    only = _make(source_id="s1", sha="3" * 64)
    checks = verify_cross_source_byte(
        (only,),
        document_group_key="einstein-besso-1909",
        now_iso="2026-05-10T12:00:00+00:00",
    )
    assert len(checks) == 1
    assert checks[0].check_kind is CheckKind.CROSS_SOURCE_BYTE
    assert checks[0].verdict is Verdict.PASS
    assert checks[0].reviewer_id == REVIEWER_ID


def test_consistent_group_all_pass() -> None:
    sha = "4" * 64
    a = _make(source_id="s1", sha=sha)
    b = _make(source_id="s2", sha=sha, source_url="https://example/y")
    checks = verify_cross_source_byte(
        (a, b),
        document_group_key="einstein-besso-1909",
        now_iso="2026-05-10T12:00:00+00:00",
    )
    assert len(checks) == 2
    assert all(c.verdict is Verdict.PASS for c in checks)


def test_disagreeing_group_all_needs_review() -> None:
    a = _make(source_id="s1", sha="5" * 64)
    b = _make(source_id="s2", sha="6" * 64, source_url="https://example/y")
    c = _make(source_id="s3", sha="7" * 64, source_url="https://example/z")
    checks = verify_cross_source_byte(
        (a, b, c),
        document_group_key="einstein-besso-1909",
        now_iso="2026-05-10T12:00:00+00:00",
    )
    assert len(checks) == 3
    assert all(c.verdict is Verdict.NEEDS_REVIEW for c in checks)
    for check in checks:
        joined = "\n".join(check.evidence)
        assert "distinct_byte_sha256_count=3" in joined


def test_empty_group_raises() -> None:
    with pytest.raises(ValueError, match="non-empty tuple"):
        verify_cross_source_byte((), document_group_key="x")


def test_blank_group_key_raises() -> None:
    only = _make(source_id="s1", sha="8" * 64)
    with pytest.raises(ValueError, match="document_group_key must be a non-empty"):
        verify_cross_source_byte((only,), document_group_key="   ")
