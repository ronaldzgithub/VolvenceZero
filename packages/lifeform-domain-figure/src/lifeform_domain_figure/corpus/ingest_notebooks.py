"""Notebook-source adapter.

Notebooks are working drafts and manuscripts: lower confidence than
published papers, frequently mid-thought, sometimes with marginalia.
The adapter therefore sets a softer default chunk confidence
(``0.85``) so the retrieval index can downweight notebook-derived
evidence relative to a published paper when both are available.

Locator format::

    notebook:{notebook_id}:vol={volume}:page={page}:para={index}:offset={start}-{end}
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


_DEFAULT_NOTEBOOK_CONFIDENCE = 0.85


@dataclass(frozen=True)
class FigureNotebookSource:
    """One reviewed primary-source notebook entry.

    ``notebook_id`` is the canonical citation key (e.g.,
    ``"einstein-zurich-notebook-1912"``). ``volume`` / ``page`` come
    from the manuscript pagination so a citation locator can point a
    reviewer at the exact page in the archive image set.
    """

    notebook_id: str
    volume: str
    page: int
    language: str
    body: str
    figure_id: str = ""
    confidence: float = _DEFAULT_NOTEBOOK_CONFIDENCE

    def __post_init__(self) -> None:
        if not self.notebook_id.strip():
            raise ValueError(
                "FigureNotebookSource.notebook_id must be non-empty"
            )
        if not self.volume.strip():
            raise ValueError(
                f"FigureNotebookSource.volume for {self.notebook_id!r} "
                f"must be non-empty"
            )
        if self.page < 0:
            raise ValueError(
                f"FigureNotebookSource.page must be >= 0, got {self.page!r}"
            )
        if not self.body.strip():
            raise ValueError(
                f"FigureNotebookSource.body for {self.notebook_id!r} is "
                f"empty; refusing to ingest a body-less entry."
            )
        if not self.language.strip():
            raise ValueError(
                f"FigureNotebookSource.language for {self.notebook_id!r} "
                f"is empty; downstream retrieval needs an explicit language."
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"FigureNotebookSource.confidence must be in [0,1], "
                f"got {self.confidence!r}"
            )


def ingest_notebooks(
    sources: tuple[FigureNotebookSource, ...],
    *,
    uploader: str,
    upload_ts_ms: int | None = None,
    envelope_id: str | None = None,
    compliance_profile: IngestionComplianceProfile = IngestionComplianceProfile.FORCED,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> IngestionEnvelope:
    """Build an :class:`IngestionEnvelope` from one or more notebook entries."""

    if not sources:
        raise ValueError(
            "ingest_notebooks: sources tuple must be non-empty; refusing "
            "to build an envelope with no entries."
        )
    if upload_ts_ms is None:
        upload_ts_ms = int(time.time() * 1000.0)
    body_concat = "\n\n".join(source.body for source in sources)
    integrity_hash = hashlib.sha256(body_concat.encode("utf-8")).hexdigest()
    if envelope_id is None:
        envelope_id = f"figure-notebooks:{integrity_hash[:12]}"
    chunks: list[IngestionChunk] = []
    chunk_index = 0
    for source in sources:
        pieces = chunk_plain_text(source.body, max_chunk_chars=max_chunk_chars)
        if not pieces:
            raise ValueError(
                f"ingest_notebooks: chunker returned no chunks for entry "
                f"{source.notebook_id!r}"
            )
        for para_index, (segment, start, end) in enumerate(pieces):
            chunks.append(
                IngestionChunk(
                    chunk_id=f"{envelope_id}:chunk:{chunk_index:04d}",
                    text=segment,
                    locator=(
                        f"notebook:{source.notebook_id}:vol={source.volume}:"
                        f"page={source.page}:lang={source.language}:"
                        f"para={para_index}:offset={start}-{end}"
                    ),
                    confidence=source.confidence,
                )
            )
            chunk_index += 1
    provenance = IngestionProvenance(
        uploader=uploader,
        upload_ts_ms=upload_ts_ms,
        source_uri=f"figure-notebooks:{sources[0].notebook_id}+{len(sources)}",
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
    "FigureNotebookSource",
    "ingest_notebooks",
]
