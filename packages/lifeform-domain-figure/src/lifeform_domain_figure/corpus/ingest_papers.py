"""Paper-source adapter.

Takes a reviewed :class:`FigurePaperSource` (pure data: title, year,
language, body) and emits an :class:`IngestionEnvelope` whose every
chunk carries a citation-quality locator of the form::

    paper:{paper_id}:para={index}:offset={start}-{end}

The locator never embeds free-text keywords; it is purely structural,
so :class:`lifeform_domain_figure.retrieval_index.FigureRetrievalIndex`
can reproduce the citation deterministically when surfacing evidence
to ``GroundedDecoder``.
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
class FigurePaperSource:
    """One reviewed primary-source paper as input to ingestion.

    ``paper_id`` is the canonical citation key (e.g., ``"einstein-1905-001"``)
    and is the load-bearing field for retrieval locators downstream.
    ``language`` is an ISO-639 code and is preserved on each chunk
    locator so the retrieval index can prefer the same-language match
    when the runtime asks in a particular language.
    """

    paper_id: str
    title: str
    year: int
    language: str
    body: str
    publication_locator: str = ""
    figure_id: str = ""

    def __post_init__(self) -> None:
        if not self.paper_id.strip():
            raise ValueError("FigurePaperSource.paper_id must be non-empty")
        if not self.title.strip():
            raise ValueError("FigurePaperSource.title must be non-empty")
        if not self.body.strip():
            raise ValueError(
                f"FigurePaperSource.body for {self.paper_id!r} is empty; "
                f"refusing to ingest a body-less paper."
            )
        if not self.language.strip():
            raise ValueError(
                f"FigurePaperSource.language for {self.paper_id!r} is "
                f"empty; downstream retrieval needs an explicit language."
            )


def ingest_papers(
    sources: tuple[FigurePaperSource, ...],
    *,
    uploader: str,
    upload_ts_ms: int | None = None,
    envelope_id: str | None = None,
    compliance_profile: IngestionComplianceProfile = IngestionComplianceProfile.FORCED,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> IngestionEnvelope:
    """Build a single :class:`IngestionEnvelope` from one or more papers.

    All papers are concatenated into one envelope so the canonical
    ingestion pipeline receives them as one operator-supplied batch.
    Each chunk's locator carries the paper id and language so
    downstream consumers do not have to re-derive provenance.
    """

    if not sources:
        raise ValueError(
            "ingest_papers: sources tuple must be non-empty; refusing to "
            "build an envelope with no papers."
        )
    if upload_ts_ms is None:
        upload_ts_ms = int(time.time() * 1000.0)
    body_concat = "\n\n".join(source.body for source in sources)
    integrity_hash = hashlib.sha256(body_concat.encode("utf-8")).hexdigest()
    if envelope_id is None:
        envelope_id = f"figure-papers:{integrity_hash[:12]}"
    chunks: list[IngestionChunk] = []
    chunk_index = 0
    for source in sources:
        pieces = chunk_plain_text(source.body, max_chunk_chars=max_chunk_chars)
        if not pieces:
            raise ValueError(
                f"ingest_papers: chunker returned no chunks for paper "
                f"{source.paper_id!r}"
            )
        for para_index, (segment, start, end) in enumerate(pieces):
            chunks.append(
                IngestionChunk(
                    chunk_id=f"{envelope_id}:chunk:{chunk_index:04d}",
                    text=segment,
                    locator=(
                        f"paper:{source.paper_id}:lang={source.language}:"
                        f"para={para_index}:offset={start}-{end}"
                    ),
                    confidence=1.0,
                )
            )
            chunk_index += 1
    provenance = IngestionProvenance(
        uploader=uploader,
        upload_ts_ms=upload_ts_ms,
        source_uri=f"figure-papers:{sources[0].paper_id}+{len(sources)}",
        integrity_hash=integrity_hash,
    )
    return IngestionEnvelope(
        envelope_id=envelope_id,
        source_kind=IngestionSourceKind.BOOK,
        chunks=tuple(chunks),
        provenance=provenance,
        compliance_profile=compliance_profile,
        partial_failures=(),
    )


__all__ = [
    "FigurePaperSource",
    "ingest_papers",
]
