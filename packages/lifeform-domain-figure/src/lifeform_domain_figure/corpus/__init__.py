"""Primary-source corpus ingestion adapters for the figure vertical.

Four source-kind adapters that all produce :class:`IngestionEnvelope`
instances with citation-quality locators on every chunk:

* :func:`ingest_papers` — published scientific / philosophical papers.
* :func:`ingest_letters` — correspondence with sender / recipient /
  date threading metadata baked into the locator.
* :func:`ingest_lectures` — public lectures and addresses.
* :func:`ingest_notebooks` — drafts, manuscripts, working notes.

Each adapter shares the common ``chunk_plain_text`` chunker from
``lifeform-ingestion`` so the F2 retrieval index can rely on a single
chunking pass; what differs is the **locator format** — the L3
grounding contract demands a canonical citation per chunk so the
runtime ``GroundedDecoder`` can produce reviewable evidence pointers
back to the underlying primary source.

D2 helpers (``provenance`` / ``dedupe`` / ``citation``) layer on top
of the four adapters:

* :mod:`lifeform_domain_figure.corpus.provenance` — typed
  :class:`SourceProvenance` record + fingerprint.
* :mod:`lifeform_domain_figure.corpus.dedupe` — cross-envelope
  byte-level deduplication report.
* :mod:`lifeform_domain_figure.corpus.citation` — strict typed parser
  for the four locator formats above.

All adapters fail loudly on empty / whitespace-only inputs (mirroring
``lifeform-ingestion`` discipline). None of them branch on text
content (``no-keyword-matching-hacks.mdc``).
"""

from __future__ import annotations

from lifeform_domain_figure.corpus.citation import (
    LocatorKind,
    LocatorOffset,
    ParsedLocator,
    parse_locator,
)
from lifeform_domain_figure.corpus.dedupe import (
    DedupReport,
    DuplicateGroup,
    compute_dedup_report,
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
from lifeform_domain_figure.corpus.provenance import (
    CaptureMethod,
    LegalClearance,
    SourceProvenance,
    fingerprint_provenance,
)


__all__ = [
    # F1.2 source adapters
    "FigureLectureSource",
    "FigureLetterSource",
    "FigureNotebookSource",
    "FigurePaperSource",
    "ingest_lectures",
    "ingest_letters",
    "ingest_notebooks",
    "ingest_papers",
    # D2 provenance
    "CaptureMethod",
    "LegalClearance",
    "SourceProvenance",
    "fingerprint_provenance",
    # D2 dedupe
    "DedupReport",
    "DuplicateGroup",
    "compute_dedup_report",
    # D2 citation parser
    "LocatorKind",
    "LocatorOffset",
    "ParsedLocator",
    "parse_locator",
]
