"""Archive adapters: typed translation from archive payloads to source records.

This subpackage provides four pre-baked archive payload normalisers:

* :mod:`lifeform_domain_figure.corpus.archives.cpae` — Princeton
  Collected Papers of Albert Einstein.
* :mod:`lifeform_domain_figure.corpus.archives.wikisource` — Wikisource
  (multi-language).
* :mod:`lifeform_domain_figure.corpus.archives.gutenberg` — Project
  Gutenberg.
* :mod:`lifeform_domain_figure.corpus.archives.internet_archive` —
  Internet Archive (archive.org).

V1 scope (this packet): each archive module defines a typed
``<Archive>Payload`` dataclass (everything a curator manually
downloads or scrapes from the archive page) plus a set of
``<archive>_to_<source_kind>_source`` helpers that translate the
payload into a typed
:class:`lifeform_domain_figure.FigurePaperSource` /
:class:`FigureLetterSource` / :class:`FigureLectureSource` /
:class:`FigureNotebookSource` record.

V1 explicitly **does not** issue HTTP requests:

* The figure vertical inherits the
  ``lifeform-ingestion`` "web sources are slice 2b territory"
  discipline (SSRF / content-type sniffing must land in its own
  reviewable packet).
* Curators feed pre-downloaded payloads. A future packet (V2) adds
  one or more :class:`ArchiveFetcher` implementations that turn an
  archive URL into the same typed payload, behind the same Protocol
  surface.

The :class:`ArchiveFetcher` Protocol exists today only as the typed
contract. Calling :func:`offline_archive_fetcher` returns a stub
that always raises — V1 callers are expected to construct payloads
in code or load them from disk; live HTTP fetchers are V2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lifeform_domain_figure.corpus.archives.chinese_text_project import (
    CTPPayload,
    ctp_to_paper_source,
)
from lifeform_domain_figure.corpus.archives.cpae import (
    CPAEDocumentKind,
    CPAEPayload,
    cpae_to_letter_source,
    cpae_to_paper_source,
)
from lifeform_domain_figure.corpus.archives.gutenberg import (
    GutenbergPayload,
    gutenberg_to_paper_source,
)
from lifeform_domain_figure.corpus.archives.internet_archive import (
    InternetArchivePayload,
    internet_archive_to_lecture_source,
    internet_archive_to_paper_source,
)
from lifeform_domain_figure.corpus.archives.wikisource import (
    WikisourcePayload,
    wikisource_to_lecture_source,
    wikisource_to_paper_source,
)


@dataclass(frozen=True)
class ArchiveFetchResult:
    """Typed result returned by an :class:`ArchiveFetcher`.

    ``raw_payload`` is the canonical typed payload for the archive
    being fetched (one of the four ``*Payload`` classes above);
    ``source_url`` is the URL the curator (or future fetcher) hit.

    The result is intentionally *not* a kernel ``IngestionEnvelope``:
    a fetcher's job is to return the typed payload; conversion into
    typed source records is the caller's choice (paper vs letter
    etc.) and lives in the ``*_to_*_source`` helpers.
    """

    source_url: str
    raw_payload: object


class ArchiveFetcher(Protocol):
    """Forward-declared Protocol for a live archive HTTP fetcher.

    No implementation lives in V1 (the offline stub below always
    raises). A future packet supplies an :class:`HTTPArchiveFetcher`
    that respects an SSRF allowlist + content-type sniffing per
    ``lifeform-ingestion`` slice 2b.
    """

    def fetch(self, url: str) -> ArchiveFetchResult: ...


class _OfflineArchiveFetcher:
    """V1 stub fetcher: every call raises ``NotImplementedError``.

    Returned by :func:`offline_archive_fetcher` so callers that
    accidentally try to use the V1 default fail loudly with a
    pointer to the right migration path, rather than silently
    blocking on a network call that V1 refuses to make.
    """

    def fetch(self, url: str) -> ArchiveFetchResult:
        raise NotImplementedError(
            "V1 of the figure vertical has no live archive fetcher. "
            "Construct an *Payload directly from a pre-downloaded "
            "artifact, or wait for the V2 HTTPArchiveFetcher packet. "
            f"Refused fetch for url={url!r}."
        )


def offline_archive_fetcher() -> ArchiveFetcher:
    """Return the V1 offline stub fetcher (always raises on ``.fetch``)."""
    return _OfflineArchiveFetcher()


__all__ = [
    "ArchiveFetcher",
    "ArchiveFetchResult",
    # CPAE
    "CPAEDocumentKind",
    "CPAEPayload",
    "cpae_to_letter_source",
    "cpae_to_paper_source",
    # Wikisource
    "WikisourcePayload",
    "wikisource_to_lecture_source",
    "wikisource_to_paper_source",
    # Project Gutenberg
    "GutenbergPayload",
    "gutenberg_to_paper_source",
    # Internet Archive
    "InternetArchivePayload",
    "internet_archive_to_lecture_source",
    "internet_archive_to_paper_source",
    # Chinese Text Project (D7 PoC)
    "CTPPayload",
    "ctp_to_paper_source",
    # V1 default fetcher
    "offline_archive_fetcher",
]
