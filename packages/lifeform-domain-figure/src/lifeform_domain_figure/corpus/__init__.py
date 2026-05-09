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

All adapters fail loudly on empty / whitespace-only inputs (mirroring
``lifeform-ingestion`` discipline). None of them branch on text
content (``no-keyword-matching-hacks.mdc``).
"""

from __future__ import annotations

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


__all__ = [
    "FigureLectureSource",
    "FigureLetterSource",
    "FigureNotebookSource",
    "FigurePaperSource",
    "ingest_lectures",
    "ingest_letters",
    "ingest_notebooks",
    "ingest_papers",
]
