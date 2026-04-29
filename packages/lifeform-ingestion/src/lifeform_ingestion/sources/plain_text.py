"""Plain-text source adapter (Gap 3 slice 1).

Turns an in-memory string or a UTF-8 text file into an
``IngestionEnvelope``. Covers the BOOK sub-case of "a plain-text
file" and the generic CORPUS case (inline strings uploaded via a
service endpoint).

Chunking strategy:

* Split on double-newline paragraph boundaries first; fall back to
  hard cuts at ``max_chunk_chars`` when a paragraph is longer than
  the limit. This keeps natural sentences together when possible
  without letting any chunk blow up the context budget.
* Default limit is 2048 chars \u2014 enough for several paragraphs, far
  below anything that would strain a turn.
* Each chunk gets a locator ``"offset=<start>-<end>"`` pointing back
  to the original text so an operator can grep their source.
"""

from __future__ import annotations

import hashlib
import time

from lifeform_ingestion.envelope import (
    IngestionChunk,
    IngestionComplianceProfile,
    IngestionEnvelope,
    IngestionProvenance,
    IngestionSourceKind,
)


DEFAULT_MAX_CHUNK_CHARS = 2048


def chunk_plain_text(
    text: str,
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> tuple[tuple[str, int, int], ...]:
    """Split a text string into ``(chunk_text, start_offset, end_offset)`` tuples.

    Public helper so tests can validate the chunking logic
    independently of envelope construction. Guarantees:

    * Concatenating chunks in order reproduces the original text
      (up to trailing whitespace on each chunk).
    * Each returned chunk has ``len(chunk_text) <= max_chunk_chars``.
    * Empty / whitespace-only input returns an empty tuple \u2014 the
      caller should fail loudly on that rather than emit an empty
      envelope (see ``envelope_from_text``).
    """
    if max_chunk_chars <= 0:
        raise ValueError(
            f"chunk_plain_text max_chunk_chars must be > 0, got {max_chunk_chars!r}"
        )
    stripped_is_empty = not text.strip()
    if stripped_is_empty:
        return ()
    chunks: list[tuple[str, int, int]] = []
    paragraphs = text.split("\n\n")
    pos = 0
    for paragraph in paragraphs:
        # Track the absolute offset of this paragraph in the original
        # string so locators reflect reality even when later paragraphs
        # are shorter than earlier ones.
        para_start = pos
        para_end = para_start + len(paragraph)
        pos = para_end + 2  # account for the split separator "\n\n"
        if not paragraph.strip():
            continue
        if len(paragraph) <= max_chunk_chars:
            chunks.append((paragraph, para_start, para_end))
            continue
        # Hard-cut oversized paragraphs.
        cursor = 0
        while cursor < len(paragraph):
            end = min(cursor + max_chunk_chars, len(paragraph))
            segment = paragraph[cursor:end]
            if segment.strip():
                chunks.append((segment, para_start + cursor, para_start + end))
            cursor = end
    return tuple(chunks)


def envelope_from_text(
    text: str,
    *,
    source_uri: str,
    uploader: str = "system",
    upload_ts_ms: int | None = None,
    envelope_id: str | None = None,
    source_kind: IngestionSourceKind = IngestionSourceKind.CORPUS,
    compliance_profile: IngestionComplianceProfile = IngestionComplianceProfile.FORCED,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> IngestionEnvelope:
    """Build an ``IngestionEnvelope`` from an in-memory text string.

    Fails loudly on empty input \u2014 an ingestion call with no content
    is almost certainly a bug (why invoke the pipeline with nothing?)
    and we do not want to paper over it with an empty envelope.
    Callers that legitimately have a maybe-empty source should check
    before calling.
    """
    if not text.strip():
        raise ValueError(
            f"envelope_from_text: source {source_uri!r} is empty or whitespace-only; "
            f"refusing to build an empty envelope."
        )
    pieces = chunk_plain_text(text, max_chunk_chars=max_chunk_chars)
    if not pieces:
        # Belt-and-braces: should be unreachable because we checked above.
        raise ValueError(
            f"envelope_from_text: chunker returned no chunks for {source_uri!r}"
        )
    integrity_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if envelope_id is None:
        envelope_id = f"ingestion:{integrity_hash[:12]}"
    if upload_ts_ms is None:
        upload_ts_ms = int(time.time() * 1000.0)
    chunks = tuple(
        IngestionChunk(
            chunk_id=f"{envelope_id}:chunk:{index:04d}",
            text=segment,
            locator=f"offset={start}-{end}",
            confidence=1.0,
        )
        for index, (segment, start, end) in enumerate(pieces)
    )
    provenance = IngestionProvenance(
        uploader=uploader,
        upload_ts_ms=upload_ts_ms,
        source_uri=source_uri,
        integrity_hash=integrity_hash,
    )
    return IngestionEnvelope(
        envelope_id=envelope_id,
        source_kind=source_kind,
        chunks=chunks,
        provenance=provenance,
        compliance_profile=compliance_profile,
        partial_failures=(),
    )


__all__ = [
    "DEFAULT_MAX_CHUNK_CHARS",
    "chunk_plain_text",
    "envelope_from_text",
]
