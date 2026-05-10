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

V1 scope: each archive module defines a typed ``<Archive>Payload``
dataclass (everything a curator manually downloads or scrapes from
the archive page) plus a set of ``<archive>_to_<source_kind>_source``
helpers that translate the payload into a typed
:class:`lifeform_domain_figure.FigurePaperSource` /
:class:`FigureLetterSource` / :class:`FigureLectureSource` /
:class:`FigureNotebookSource` record.

V1 also exposed an :class:`ArchiveFetcher` Protocol whose default
implementation (:func:`offline_archive_fetcher`) raised on every
``.fetch(...)`` call: at the time, no live HTTP fetcher existed, and
curators fed pre-downloaded payloads.

V2 (debt #19 closure / debt #28 L0 packet): a real HTTP-backed
fetcher is now available via :func:`live_archive_fetcher`. The
factory returns an :class:`ArchiveFetcher` whose ``fetch`` returns
an :class:`ArchiveFetchResult` carrying a :class:`LiveFetchedBytes`
``raw_payload`` (bytes + sha + content-type) instead of a typed
``*Payload``. The downstream caller pipes the bytes through the L1
cleaning pipeline (``parse_by_content_type`` ->
``clean_raw_document`` -> ``cleaned_to_*_payload``) to construct
the typed payload, then through the existing translator to a
:class:`Figure*Source`.

Why the V2 ``raw_payload`` is bytes-not-typed: only the curator
knows the typed metadata (volume / document_number / page_title /
etc.) and only the L1 parser knows how to extract clean text from
raw bytes. The V2 fetcher cannot fill those fields itself; it
delivers bytes anchored to a content-addressable ``raw_sha256`` and
hands off to L1 + curator.

:func:`offline_archive_fetcher` remains unchanged for backward
compatibility (existing tests / curator manual mode).
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


@dataclass(frozen=True)
class LiveFetchedBytes:
    """V2 ``raw_payload`` shape returned by :func:`live_archive_fetcher`.

    Carries the raw bytes (ready for the L1
    :func:`parse_by_content_type` dispatcher), the content-addressable
    ``raw_sha256`` anchor key (== L1 ``RawDocument.raw_sha256`` ==
    ``SourceProvenance.byte_sha256``), the normalised ``content_type``
    label the L1 parser expects, and the original ``http_status``.

    The downstream pipeline:

    1. ``parse_by_content_type(bytes, source_url, content_type)`` ->
       :class:`RawDocument`
    2. ``clean_raw_document(raw)`` -> :class:`CleanedDocument`
    3. ``cleaned_to_<archive>_payload(cleaned, ...)`` (curator metadata
       supplied) -> typed ``*Payload``
    4. ``<archive>_to_<source_kind>_source(payload, ...)`` -> typed
       ``Figure*Source``
    """

    body: bytes
    raw_sha256: str
    content_type: str
    http_status: int

    def __post_init__(self) -> None:
        if not isinstance(self.body, (bytes, bytearray)) or not self.body:
            raise ValueError("LiveFetchedBytes.body must be non-empty bytes")
        if not isinstance(self.raw_sha256, str) or len(self.raw_sha256) != 64:
            raise ValueError(
                f"LiveFetchedBytes.raw_sha256 must be a 64-char hex sha256; "
                f"got {self.raw_sha256!r}"
            )
        if not isinstance(self.content_type, str) or not self.content_type.strip():
            raise ValueError("LiveFetchedBytes.content_type must be non-empty")


class _LiveArchiveFetcher:
    """V2 :class:`ArchiveFetcher` impl backed by the L0 crawler stack.

    Single-URL mode: applies SSRF scope check + dispatches to the
    L0 archive-specific fetcher + writes bytes to the supplied
    :class:`CleaningStore` (when one is given). Does NOT consult
    robots.txt or rate-limit; callers needing those behaviours
    should drive a full :class:`CrawlScheduler` instead. This mode
    exists for one-off "fetch this exact URL right now" curator
    flows.
    """

    def __init__(
        self,
        *,
        fetch_kind: str,
        scope,
        http_client,
        cleaning_store=None,
    ) -> None:
        from lifeform_domain_figure.crawl.fetchers import build_default_fetchers
        from lifeform_domain_figure.crawl.records import VALID_FETCH_KINDS

        if fetch_kind not in VALID_FETCH_KINDS:
            raise ValueError(
                f"_LiveArchiveFetcher.fetch_kind must be one of "
                f"{sorted(VALID_FETCH_KINDS)!r}; got {fetch_kind!r}"
            )
        self._fetch_kind = fetch_kind
        self._scope = scope
        self._http_client = http_client
        self._cleaning_store = cleaning_store
        self._fetchers = build_default_fetchers()

    def fetch(self, url: str) -> ArchiveFetchResult:
        from lifeform_domain_figure.crawl.fetchers import dispatch_for
        from lifeform_domain_figure.crawl.http_client import (
            HTTPResponse,
            NOT_MODIFIED,
            ScopeRejection,
        )
        from lifeform_domain_figure.crawl.records import (
            CrawlRequest,
            request_id_for,
        )
        from datetime import datetime, timezone
        import hashlib

        if not self._scope.is_in_scope(url):
            reason = self._scope.reason_out_of_scope(url) or "out of scope"
            raise ScopeRejection(f"live_archive_fetcher refused url={url!r}: {reason}")
        request = CrawlRequest(
            url=url,
            fetch_kind=self._fetch_kind,
            request_id=request_id_for(self._fetch_kind, url),
            enqueued_at_iso=datetime.now(timezone.utc).isoformat(),
        )
        fetcher = dispatch_for(request, self._fetchers)
        response = fetcher.fetch(request, self._http_client)
        if response is NOT_MODIFIED:
            raise RuntimeError(
                "live_archive_fetcher: server returned 304 Not Modified but "
                "no etag/last_modified was supplied; this should be unreachable."
            )
        assert isinstance(response, HTTPResponse)
        content_type = fetcher.derive_content_type(request, response)
        raw_sha = hashlib.sha256(response.body).hexdigest()
        if self._cleaning_store is not None:
            stored_sha = self._cleaning_store.put_raw(
                response.body, source_url=url, content_type=content_type
            )
            if stored_sha != raw_sha:
                raise RuntimeError(
                    "live_archive_fetcher: cleaning_store sha mismatch — "
                    f"local={raw_sha!r} store={stored_sha!r}"
                )
        return ArchiveFetchResult(
            source_url=url,
            raw_payload=LiveFetchedBytes(
                body=response.body,
                raw_sha256=raw_sha,
                content_type=content_type,
                http_status=response.http_status,
            ),
        )


def live_archive_fetcher(
    fetch_kind: str,
    *,
    scope=None,
    http_client=None,
    cleaning_store=None,
    user_agent: str | None = None,
) -> ArchiveFetcher:
    """Return a V2 :class:`ArchiveFetcher` backed by the L0 crawler stack.

    ``fetch_kind`` must be one of the registered L0 crawl kinds
    (``"generic"`` / ``"cpae"`` / ``"wikisource"`` / ``"gutenberg"``
    / ``"internet_archive"``).

    When ``scope`` is omitted, builds a default policy with the L0
    baked-in archive host allowlist. When ``http_client`` is omitted,
    builds a fresh :class:`BaseHTTPClient` against the supplied (or
    default) scope. When ``cleaning_store`` is supplied, fetched
    bytes are also written to it (so the L0 / L1 anchor chain is
    established immediately).

    The returned fetcher does NOT consult robots.txt or apply a
    rate limiter; for full crawl behaviour use
    :class:`CrawlScheduler` instead.
    """

    from lifeform_domain_figure.crawl.http_client import BaseHTTPClient
    from lifeform_domain_figure.crawl.scope_policy import (
        DEFAULT_USER_AGENT,
        default_scope_policy,
    )

    effective_scope = scope or default_scope_policy(
        user_agent or DEFAULT_USER_AGENT
    )
    effective_client = http_client or BaseHTTPClient(scope=effective_scope)
    return _LiveArchiveFetcher(
        fetch_kind=fetch_kind,
        scope=effective_scope,
        http_client=effective_client,
        cleaning_store=cleaning_store,
    )


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
    # V2 live fetcher (debt #19 closure)
    "LiveFetchedBytes",
    "live_archive_fetcher",
]
