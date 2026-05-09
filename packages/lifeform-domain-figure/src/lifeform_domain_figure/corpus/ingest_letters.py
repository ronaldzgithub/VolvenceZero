"""Letter-source adapter.

Letters are correspondence: each one has a sender, a recipient, and a
date. The locator format encodes all three so a retrieval result can
be cited as ``letter:einstein-to-bohr:1935-04-12:para=2`` — exactly
the citation form a reviewer would expect to see in the L3 evidence
pointer.

Letter threading (reply chains) is captured via ``in_reply_to`` on
the source dataclass; downstream reflection / case extraction can
walk the chain by id without parsing free text.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from lifeform_ingestion.envelope import (
    IngestionChunk,
    IngestionComplianceProfile,
    IngestionEnvelope,
    IngestionProvenance,
    IngestionSourceKind,
)
from lifeform_ingestion.sources.plain_text import (
    DEFAULT_MAX_CHUNK_CHARS,
    chunk_plain_text,
)


@dataclass(frozen=True)
class FigureLetterSource:
    """One reviewed primary-source letter.

    ``letter_id`` is the canonical citation key. ``sender_id`` /
    ``recipient_id`` are normalised participant ids (e.g., ``"einstein"``,
    ``"bohr"``) so the retrieval index can filter on them without
    parsing free-text salutations. ``date_iso`` is the letter's date as
    ISO-8601 (``YYYY-MM-DD``); empty allowed only when the original
    document is undated.
    """

    letter_id: str
    sender_id: str
    recipient_id: str
    date_iso: str
    language: str
    body: str
    in_reply_to: str = ""
    figure_id: str = ""

    def __post_init__(self) -> None:
        if not self.letter_id.strip():
            raise ValueError("FigureLetterSource.letter_id must be non-empty")
        if not self.sender_id.strip():
            raise ValueError("FigureLetterSource.sender_id must be non-empty")
        if not self.recipient_id.strip():
            raise ValueError("FigureLetterSource.recipient_id must be non-empty")
        if not self.body.strip():
            raise ValueError(
                f"FigureLetterSource.body for {self.letter_id!r} is empty; "
                f"refusing to ingest a body-less letter."
            )
        if not self.language.strip():
            raise ValueError(
                f"FigureLetterSource.language for {self.letter_id!r} is "
                f"empty; downstream retrieval needs an explicit language."
            )


def ingest_letters(
    sources: tuple[FigureLetterSource, ...],
    *,
    uploader: str,
    upload_ts_ms: int | None = None,
    envelope_id: str | None = None,
    compliance_profile: IngestionComplianceProfile = IngestionComplianceProfile.FORCED,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> IngestionEnvelope:
    """Build an :class:`IngestionEnvelope` from one or more letters."""

    if not sources:
        raise ValueError(
            "ingest_letters: sources tuple must be non-empty; refusing to "
            "build an envelope with no letters."
        )
    if upload_ts_ms is None:
        upload_ts_ms = int(time.time() * 1000.0)
    body_concat = "\n\n".join(source.body for source in sources)
    integrity_hash = hashlib.sha256(body_concat.encode("utf-8")).hexdigest()
    if envelope_id is None:
        envelope_id = f"figure-letters:{integrity_hash[:12]}"
    chunks: list[IngestionChunk] = []
    chunk_index = 0
    for source in sources:
        pieces = chunk_plain_text(source.body, max_chunk_chars=max_chunk_chars)
        if not pieces:
            raise ValueError(
                f"ingest_letters: chunker returned no chunks for letter "
                f"{source.letter_id!r}"
            )
        date_label = source.date_iso or "undated"
        for para_index, (segment, start, end) in enumerate(pieces):
            chunks.append(
                IngestionChunk(
                    chunk_id=f"{envelope_id}:chunk:{chunk_index:04d}",
                    text=segment,
                    locator=(
                        f"letter:{source.sender_id}-to-{source.recipient_id}:"
                        f"date={date_label}:lang={source.language}:"
                        f"para={para_index}:offset={start}-{end}"
                    ),
                    confidence=1.0,
                )
            )
            chunk_index += 1
    provenance = IngestionProvenance(
        uploader=uploader,
        upload_ts_ms=upload_ts_ms,
        source_uri=f"figure-letters:{sources[0].letter_id}+{len(sources)}",
        integrity_hash=integrity_hash,
    )
    return IngestionEnvelope(
        envelope_id=envelope_id,
        source_kind=IngestionSourceKind.CORPUS,
        chunks=tuple(chunks),
        provenance=provenance,
        compliance_profile=compliance_profile,
        partial_failures=(),
    )


__all__ = [
    "FigureLetterSource",
    "ingest_letters",
]
