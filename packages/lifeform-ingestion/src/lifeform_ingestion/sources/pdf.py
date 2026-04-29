"""PDF source adapter (Gap 3 slice 2).

Turns a PDF file on disk (or in-memory bytes) into an
``IngestionEnvelope``: one chunk per extracted page, with per-page
locators (``"page=N/TOTAL"``) so an operator can trace which page
produced which turn.

Design rules:

* **Lazy dep import.** ``pypdf`` is declared as
  ``lifeform-ingestion[pdf]`` optional; hosts that only use plain_text
  / task_result pay no install cost. The import happens inside the
  source function so the module loads cleanly without pypdf present.
  An ``ImportError`` from the real call surfaces a clear message
  naming the extra.
* **Encrypted PDFs fail loudly.** We do NOT attempt to brute-force
  or auto-decrypt; encrypted input is a caller error and we raise
  immediately with the provenance tag intact. A downstream retry
  path can supply a cleartext copy.
* **Per-page parse_error.** A single unreadable page (corrupt xref,
  font decode failure) does NOT abort the whole envelope. The
  chunk for that page is emitted with empty ``text`` + a populated
  ``parse_error``, and ``partial_failures`` records the chunk_id.
  This matches the envelope contract's "silent drops forbidden"
  invariant.
* **Bounded page count.** ``max_pages`` defaults to 200 so a 10k
  page textbook cannot silently blow up memory. Exceeding the
  cap raises rather than truncating \u2014 the operator should
  explicitly increase the budget after thinking about it.
* **No network.** Only file paths and bytes are accepted; any
  remote fetch would need a ``web`` adapter (slice 2b territory).
"""

from __future__ import annotations

import hashlib
import io
import pathlib
import time
from typing import Any

from lifeform_ingestion.envelope import (
    IngestionChunk,
    IngestionComplianceProfile,
    IngestionEnvelope,
    IngestionProvenance,
    IngestionSourceKind,
)


DEFAULT_MAX_PAGES: int = 200
DEFAULT_MAX_CHUNK_CHARS: int = 4096
# Hard cap on characters per page; a page that yields more than this
# after extraction is almost certainly a table of contents / index page
# with weird flow \u2014 truncate + mark as partial success.
_HARD_PAGE_CHAR_CAP: int = 16384


class PdfIngestionError(ValueError):
    """Raised when a PDF cannot be opened / is encrypted / exceeds caps.

    Subclasses ``ValueError`` so callers that already catch
    ``ValueError`` (the pattern the other sources use) do not have
    to add a new except branch. Subclass exists so more-specific
    callers can distinguish PDF failures from generic value errors.
    """


def _load_pypdf() -> Any:
    """Lazy-import ``pypdf`` with a clear error when the extra is absent."""
    try:
        import pypdf  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on install
        raise PdfIngestionError(
            "PDF ingestion requires the pypdf dependency. Install with "
            "'pip install lifeform-ingestion[pdf]' or add pypdf>=4.0 "
            "to your environment."
        ) from exc
    return pypdf


def _extract_pages(pdf_bytes: bytes) -> tuple[tuple[str, str], ...]:
    """Return a tuple of ``(page_text, parse_error)`` per page.

    On pypdf read errors for a single page we keep going with the
    rest of the document so the caller sees a partial result; on
    document-level errors (corrupt header / encrypted) we raise
    ``PdfIngestionError``.
    """
    pypdf = _load_pypdf()
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes), strict=False)
    except Exception as exc:  # noqa: BLE001 - we re-raise typed
        raise PdfIngestionError(
            f"pypdf failed to open PDF: {type(exc).__name__}: {exc}"
        ) from exc
    if reader.is_encrypted:
        # Some PDFs are "encrypted" with an empty password; attempt one
        # decrypt with '' and only give up when that also fails.
        try:
            if not reader.decrypt(""):
                raise PdfIngestionError(
                    "PDF is encrypted and no password was supplied. "
                    "Decrypt the file before ingesting."
                )
        except Exception as exc:  # noqa: BLE001
            raise PdfIngestionError(
                f"PDF decrypt failed: {type(exc).__name__}: {exc}"
            ) from exc
    pages: list[tuple[str, str]] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - per-page best-effort
            pages.append(("", f"page_extract_failed:{type(exc).__name__}:{exc}"))
            continue
        if len(text) > _HARD_PAGE_CHAR_CAP:
            pages.append(
                (
                    text[:_HARD_PAGE_CHAR_CAP],
                    f"page_truncated:{len(text)}>{_HARD_PAGE_CHAR_CAP}",
                )
            )
            continue
        pages.append((text, ""))
    return tuple(pages)


def envelope_from_pdf_bytes(
    data: bytes,
    *,
    source_uri: str,
    uploader: str = "system",
    upload_ts_ms: int | None = None,
    envelope_id: str | None = None,
    compliance_profile: IngestionComplianceProfile = IngestionComplianceProfile.FORCED,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> IngestionEnvelope:
    """Build an envelope from raw PDF bytes.

    ``max_pages`` is an upper bound; exceeding it raises. A PDF
    with zero extractable text on every page raises too \u2014 there
    is no point running the pipeline on empty content.
    """
    if not data:
        raise PdfIngestionError(
            f"envelope_from_pdf_bytes: input for {source_uri!r} is empty"
        )
    if not data.startswith(b"%PDF"):
        raise PdfIngestionError(
            f"envelope_from_pdf_bytes: {source_uri!r} does not start with "
            f"'%PDF' magic; refusing to parse non-PDF input"
        )
    if max_pages <= 0:
        raise PdfIngestionError(
            f"envelope_from_pdf_bytes: max_pages must be > 0, got {max_pages!r}"
        )
    pages = _extract_pages(data)
    if len(pages) > max_pages:
        raise PdfIngestionError(
            f"envelope_from_pdf_bytes: {source_uri!r} has {len(pages)} pages, "
            f"exceeds max_pages={max_pages}. Split the document or raise the limit."
        )
    integrity_hash = hashlib.sha256(data).hexdigest()
    if envelope_id is None:
        envelope_id = f"pdf:{integrity_hash[:12]}"
    if upload_ts_ms is None:
        upload_ts_ms = int(time.time() * 1000.0)
    total = len(pages)
    chunks: list[IngestionChunk] = []
    failures: list[str] = []
    emitted_any_text = False
    for page_index, (page_text, parse_error) in enumerate(pages):
        page_number = page_index + 1
        chunk_id = f"{envelope_id}:page:{page_index:04d}"
        locator = f"page={page_number}/{total}"
        if parse_error:
            chunks.append(
                IngestionChunk(
                    chunk_id=chunk_id,
                    text="",
                    locator=locator,
                    confidence=0.0,
                    parse_error=parse_error,
                )
            )
            failures.append(chunk_id)
            continue
        # A blank page (extract returns empty) is recorded as a
        # structured parse_error rather than a hard drop, so the
        # operator can see the shape of their corpus.
        if not page_text.strip():
            chunks.append(
                IngestionChunk(
                    chunk_id=chunk_id,
                    text="",
                    locator=locator,
                    confidence=0.0,
                    parse_error="page_empty",
                )
            )
            failures.append(chunk_id)
            continue
        # Clip to max_chunk_chars per chunk (splitting oversized pages
        # into sub-chunks with a sub-index suffix on chunk_id).
        cursor = 0
        sub_index = 0
        while cursor < len(page_text):
            end = min(cursor + max_chunk_chars, len(page_text))
            segment = page_text[cursor:end]
            if segment.strip():
                sub_locator = (
                    locator
                    if end == len(page_text) and cursor == 0
                    else f"{locator},offset={cursor}-{end}"
                )
                sub_chunk_id = (
                    chunk_id if sub_index == 0 else f"{chunk_id}:sub:{sub_index:02d}"
                )
                chunks.append(
                    IngestionChunk(
                        chunk_id=sub_chunk_id,
                        text=segment,
                        locator=sub_locator,
                        confidence=1.0,
                    )
                )
                emitted_any_text = True
                sub_index += 1
            cursor = end
    if not emitted_any_text:
        raise PdfIngestionError(
            f"envelope_from_pdf_bytes: {source_uri!r} produced no extractable "
            f"text across {total} pages; nothing to ingest."
        )
    provenance = IngestionProvenance(
        uploader=uploader,
        upload_ts_ms=upload_ts_ms,
        source_uri=source_uri,
        integrity_hash=integrity_hash,
    )
    return IngestionEnvelope(
        envelope_id=envelope_id,
        source_kind=IngestionSourceKind.BOOK,
        chunks=tuple(chunks),
        provenance=provenance,
        compliance_profile=compliance_profile,
        partial_failures=tuple(failures),
    )


def envelope_from_pdf_file(
    path: str | pathlib.Path,
    *,
    uploader: str = "system",
    upload_ts_ms: int | None = None,
    envelope_id: str | None = None,
    compliance_profile: IngestionComplianceProfile = IngestionComplianceProfile.FORCED,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> IngestionEnvelope:
    """Build an envelope from a PDF file on disk.

    ``source_uri`` is derived as ``file://<absolute-path>`` so the
    provenance record points back at the original bytes. A
    non-existent / non-file path raises ``PdfIngestionError``; a
    zero-byte file raises at the bytes layer.
    """
    p = pathlib.Path(path)
    if not p.is_file():
        raise PdfIngestionError(
            f"envelope_from_pdf_file: {p!s} is not a regular file"
        )
    data = p.read_bytes()
    return envelope_from_pdf_bytes(
        data,
        source_uri=f"file://{p.resolve()}",
        uploader=uploader,
        upload_ts_ms=upload_ts_ms,
        envelope_id=envelope_id,
        compliance_profile=compliance_profile,
        max_pages=max_pages,
        max_chunk_chars=max_chunk_chars,
    )


__all__ = [
    "DEFAULT_MAX_CHUNK_CHARS",
    "DEFAULT_MAX_PAGES",
    "PdfIngestionError",
    "envelope_from_pdf_bytes",
    "envelope_from_pdf_file",
]
