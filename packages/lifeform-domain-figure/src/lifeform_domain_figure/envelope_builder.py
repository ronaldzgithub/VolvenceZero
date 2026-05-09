"""Top-level ingestion envelope builder for the figure vertical.

Composes the four source-specific adapters in
``lifeform_domain_figure.corpus`` into a single
:class:`IngestionEnvelope` per source kind. Higher-level callers
(P2.1 retrieval index, P4.2 DLaaS adopt path) consume the resulting
envelopes through the canonical ``LifeformSession.run_turn(...,
trigger_kind=INGESTION)`` path; they never read the envelope's
internals here.

The builder dispatches on a typed :class:`FigureCorpusSourceBundle`
rather than free-form keyword strings so adding a new source kind
is a schema change rather than a string-table change
(``no-keyword-matching-hacks.mdc``).
"""

from __future__ import annotations

from dataclasses import dataclass

from lifeform_ingestion.envelope import (
    IngestionComplianceProfile,
    IngestionEnvelope,
)

from lifeform_domain_figure.corpus.ingest_letters import (
    FigureLetterSource,
    ingest_letters,
)
from lifeform_domain_figure.corpus.ingest_lectures import (
    FigureLectureSource,
    ingest_lectures,
)
from lifeform_domain_figure.corpus.ingest_notebooks import (
    FigureNotebookSource,
    ingest_notebooks,
)
from lifeform_domain_figure.corpus.ingest_papers import (
    FigurePaperSource,
    ingest_papers,
)


@dataclass(frozen=True)
class FigureCorpusSourceBundle:
    """Typed bundle of all reviewed primary-source inputs for one figure.

    Each field is an ordered tuple of source records; empty tuples
    are allowed so a caller can ingest only the source kinds they
    actually have. The combined envelope returned by
    :func:`build_figure_ingestion_envelope` requires at least one
    non-empty source kind.
    """

    figure_id: str
    papers: tuple[FigurePaperSource, ...] = ()
    letters: tuple[FigureLetterSource, ...] = ()
    lectures: tuple[FigureLectureSource, ...] = ()
    notebooks: tuple[FigureNotebookSource, ...] = ()

    def __post_init__(self) -> None:
        if not self.figure_id.strip():
            raise ValueError(
                "FigureCorpusSourceBundle.figure_id must be non-empty"
            )
        total = (
            len(self.papers)
            + len(self.letters)
            + len(self.lectures)
            + len(self.notebooks)
        )
        if total == 0:
            raise ValueError(
                f"FigureCorpusSourceBundle for {self.figure_id!r} contains "
                f"no source records; refusing to build an empty bundle."
            )


@dataclass(frozen=True)
class FigureIngestionEnvelopeSet:
    """Per-source-kind set of envelopes returned by the builder.

    Each field is ``None`` when that source kind was empty in the
    input bundle. Callers iterate over :attr:`envelopes` to feed the
    canonical ingestion pipeline once per non-empty source kind.
    """

    figure_id: str
    papers: IngestionEnvelope | None = None
    letters: IngestionEnvelope | None = None
    lectures: IngestionEnvelope | None = None
    notebooks: IngestionEnvelope | None = None

    @property
    def envelopes(self) -> tuple[IngestionEnvelope, ...]:
        return tuple(
            envelope
            for envelope in (self.papers, self.letters, self.lectures, self.notebooks)
            if envelope is not None
        )


def build_figure_ingestion_envelope(
    bundle: FigureCorpusSourceBundle,
    *,
    uploader: str,
    upload_ts_ms: int | None = None,
    compliance_profile: IngestionComplianceProfile = IngestionComplianceProfile.FORCED,
) -> FigureIngestionEnvelopeSet:
    """Compile a :class:`FigureCorpusSourceBundle` into per-kind envelopes.

    One envelope per non-empty source kind. The figure id is woven
    into each envelope id so audit trails can join them back to the
    bundle.
    """

    figure_id = bundle.figure_id
    papers_env = (
        ingest_papers(
            bundle.papers,
            uploader=uploader,
            upload_ts_ms=upload_ts_ms,
            envelope_id=f"figure:{figure_id}:papers",
            compliance_profile=compliance_profile,
        )
        if bundle.papers
        else None
    )
    letters_env = (
        ingest_letters(
            bundle.letters,
            uploader=uploader,
            upload_ts_ms=upload_ts_ms,
            envelope_id=f"figure:{figure_id}:letters",
            compliance_profile=compliance_profile,
        )
        if bundle.letters
        else None
    )
    lectures_env = (
        ingest_lectures(
            bundle.lectures,
            uploader=uploader,
            upload_ts_ms=upload_ts_ms,
            envelope_id=f"figure:{figure_id}:lectures",
            compliance_profile=compliance_profile,
        )
        if bundle.lectures
        else None
    )
    notebooks_env = (
        ingest_notebooks(
            bundle.notebooks,
            uploader=uploader,
            upload_ts_ms=upload_ts_ms,
            envelope_id=f"figure:{figure_id}:notebooks",
            compliance_profile=compliance_profile,
        )
        if bundle.notebooks
        else None
    )
    return FigureIngestionEnvelopeSet(
        figure_id=figure_id,
        papers=papers_env,
        letters=letters_env,
        lectures=lectures_env,
        notebooks=notebooks_env,
    )


__all__ = [
    "FigureCorpusSourceBundle",
    "FigureIngestionEnvelopeSet",
    "build_figure_ingestion_envelope",
]
