"""Lecture-source adapter.

Lectures and public addresses are semi-formal: typically a single
unified body with audience and venue metadata. The locator format
encodes the venue and date so a retrieval evidence pointer can read
``lecture:lindau-1955:para=4`` and immediately tell the reviewer
where the claim came from.
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
class FigureLectureSource:
    """One reviewed primary-source lecture.

    ``lecture_id`` is the canonical citation key (e.g.,
    ``"einstein-nobel-1922"``). ``venue_id`` is normalised
    (``"princeton-1933"``); ``audience`` is a free-text label for
    descriptive purposes only (it is not consumed by any decision
    logic).
    """

    lecture_id: str
    venue_id: str
    date_iso: str
    audience: str
    language: str
    body: str
    figure_id: str = ""

    def __post_init__(self) -> None:
        if not self.lecture_id.strip():
            raise ValueError("FigureLectureSource.lecture_id must be non-empty")
        if not self.venue_id.strip():
            raise ValueError("FigureLectureSource.venue_id must be non-empty")
        if not self.body.strip():
            raise ValueError(
                f"FigureLectureSource.body for {self.lecture_id!r} is empty; "
                f"refusing to ingest a body-less lecture."
            )
        if not self.language.strip():
            raise ValueError(
                f"FigureLectureSource.language for {self.lecture_id!r} is "
                f"empty; downstream retrieval needs an explicit language."
            )


def ingest_lectures(
    sources: tuple[FigureLectureSource, ...],
    *,
    uploader: str,
    upload_ts_ms: int | None = None,
    envelope_id: str | None = None,
    compliance_profile: IngestionComplianceProfile = IngestionComplianceProfile.FORCED,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> IngestionEnvelope:
    """Build an :class:`IngestionEnvelope` from one or more lectures."""

    if not sources:
        raise ValueError(
            "ingest_lectures: sources tuple must be non-empty; refusing to "
            "build an envelope with no lectures."
        )
    if upload_ts_ms is None:
        upload_ts_ms = int(time.time() * 1000.0)
    body_concat = "\n\n".join(source.body for source in sources)
    integrity_hash = hashlib.sha256(body_concat.encode("utf-8")).hexdigest()
    if envelope_id is None:
        envelope_id = f"figure-lectures:{integrity_hash[:12]}"
    chunks: list[IngestionChunk] = []
    chunk_index = 0
    for source in sources:
        pieces = chunk_plain_text(source.body, max_chunk_chars=max_chunk_chars)
        if not pieces:
            raise ValueError(
                f"ingest_lectures: chunker returned no chunks for lecture "
                f"{source.lecture_id!r}"
            )
        date_label = source.date_iso or "undated"
        for para_index, (segment, start, end) in enumerate(pieces):
            chunks.append(
                IngestionChunk(
                    chunk_id=f"{envelope_id}:chunk:{chunk_index:04d}",
                    text=segment,
                    locator=(
                        f"lecture:{source.lecture_id}:venue={source.venue_id}:"
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
        source_uri=f"figure-lectures:{sources[0].lecture_id}+{len(sources)}",
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
    "FigureLectureSource",
    "ingest_lectures",
]
