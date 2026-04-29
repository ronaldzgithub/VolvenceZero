"""Invariants of IngestionEnvelope / IngestionChunk / IngestionProvenance.

Covers the construction-time guarantees (post_init validations).
Behavioural tests for pipeline + source adapters live in separate
files.
"""

from __future__ import annotations

import pytest

from lifeform_ingestion import (
    IngestionChunk,
    IngestionComplianceProfile,
    IngestionEnvelope,
    IngestionProvenance,
    IngestionSourceKind,
)


def _provenance() -> IngestionProvenance:
    return IngestionProvenance(
        uploader="test",
        upload_ts_ms=1000,
        source_uri="file:///tmp/seed.txt",
        integrity_hash="deadbeef",
    )


def _good_chunk(chunk_id: str = "c-1", text: str = "content") -> IngestionChunk:
    return IngestionChunk(
        chunk_id=chunk_id,
        text=text,
        locator="offset=0-7",
        confidence=1.0,
    )


# ---------------------------------------------------------------------------
# Provenance invariants
# ---------------------------------------------------------------------------


def test_provenance_rejects_empty_uploader() -> None:
    with pytest.raises(ValueError, match="uploader"):
        IngestionProvenance(
            uploader="",
            upload_ts_ms=1000,
            source_uri="x",
            integrity_hash="",
        )


def test_provenance_rejects_empty_source_uri() -> None:
    with pytest.raises(ValueError, match="source_uri"):
        IngestionProvenance(
            uploader="u",
            upload_ts_ms=1000,
            source_uri="",
            integrity_hash="",
        )


def test_provenance_rejects_negative_timestamp() -> None:
    with pytest.raises(ValueError, match="upload_ts_ms"):
        IngestionProvenance(
            uploader="u",
            upload_ts_ms=-1,
            source_uri="x",
            integrity_hash="",
        )


# ---------------------------------------------------------------------------
# Chunk invariants
# ---------------------------------------------------------------------------


def test_chunk_rejects_empty_chunk_id() -> None:
    with pytest.raises(ValueError, match="chunk_id"):
        IngestionChunk(chunk_id="", text="ok", locator="offset=0")


def test_chunk_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValueError, match="confidence"):
        IngestionChunk(
            chunk_id="c-1", text="ok", locator="offset=0", confidence=1.5
        )
    with pytest.raises(ValueError, match="confidence"):
        IngestionChunk(
            chunk_id="c-1", text="ok", locator="offset=0", confidence=-0.01
        )


def test_chunk_rejects_empty_text_without_parse_error() -> None:
    """A silent empty chunk is a bug: either there's real content or
    the adapter is recording a failure.
    """
    with pytest.raises(ValueError, match="parse_error"):
        IngestionChunk(chunk_id="c-1", text="", locator="offset=0")


def test_chunk_accepts_empty_text_with_parse_error() -> None:
    chunk = IngestionChunk(
        chunk_id="c-1",
        text="",
        locator="page=3",
        parse_error="PyPDF2 failed: corrupt xref",
    )
    assert chunk.has_parse_error
    assert chunk.parse_error


def test_chunk_has_parse_error_reflects_trimmed_text() -> None:
    no_err = IngestionChunk(chunk_id="c-1", text="x", locator="l")
    with_err = IngestionChunk(
        chunk_id="c-2", text="", locator="l", parse_error="boom"
    )
    assert no_err.has_parse_error is False
    assert with_err.has_parse_error is True


# ---------------------------------------------------------------------------
# Envelope invariants
# ---------------------------------------------------------------------------


def test_envelope_rejects_empty_envelope_id() -> None:
    with pytest.raises(ValueError, match="envelope_id"):
        IngestionEnvelope(
            envelope_id="",
            source_kind=IngestionSourceKind.CORPUS,
            chunks=(_good_chunk(),),
            provenance=_provenance(),
        )


def test_envelope_rejects_empty_chunks() -> None:
    with pytest.raises(ValueError, match="chunks"):
        IngestionEnvelope(
            envelope_id="env-1",
            source_kind=IngestionSourceKind.CORPUS,
            chunks=(),
            provenance=_provenance(),
        )


def test_envelope_rejects_duplicate_chunk_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        IngestionEnvelope(
            envelope_id="env-1",
            source_kind=IngestionSourceKind.CORPUS,
            chunks=(_good_chunk("dup"), _good_chunk("dup")),
            provenance=_provenance(),
        )


def test_envelope_requires_partial_failures_matches_parse_errors() -> None:
    """``partial_failures`` must list exactly the chunk_ids that have
    non-empty parse_error.
    """
    failed = IngestionChunk(
        chunk_id="c-bad", text="", locator="page=3", parse_error="boom"
    )
    good = _good_chunk("c-ok")
    # Declaring no failures when there IS one is a contract violation.
    with pytest.raises(ValueError, match="partial_failures"):
        IngestionEnvelope(
            envelope_id="env-1",
            source_kind=IngestionSourceKind.BOOK,
            chunks=(good, failed),
            provenance=_provenance(),
            partial_failures=(),
        )
    # Declaring a failure that isn't there is also a violation.
    with pytest.raises(ValueError, match="partial_failures"):
        IngestionEnvelope(
            envelope_id="env-1",
            source_kind=IngestionSourceKind.BOOK,
            chunks=(good,),
            provenance=_provenance(),
            partial_failures=("c-ok",),
        )


def test_envelope_accepts_matched_partial_failures() -> None:
    failed = IngestionChunk(
        chunk_id="c-bad", text="", locator="page=3", parse_error="boom"
    )
    good = _good_chunk("c-ok")
    envelope = IngestionEnvelope(
        envelope_id="env-1",
        source_kind=IngestionSourceKind.BOOK,
        chunks=(good, failed),
        provenance=_provenance(),
        partial_failures=("c-bad",),
    )
    assert envelope.total_chunks == 2
    assert len(envelope.successful_chunks) == 1
    assert len(envelope.failed_chunks) == 1
    assert envelope.successful_chunks[0].chunk_id == "c-ok"


def test_envelope_default_compliance_profile_is_forced() -> None:
    envelope = IngestionEnvelope(
        envelope_id="env-default",
        source_kind=IngestionSourceKind.CORPUS,
        chunks=(_good_chunk(),),
        provenance=_provenance(),
    )
    assert envelope.compliance_profile is IngestionComplianceProfile.FORCED
