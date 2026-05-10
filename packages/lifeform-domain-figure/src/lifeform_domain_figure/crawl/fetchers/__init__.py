"""Fetcher protocol and per-archive registry for the L0 crawler.

A :class:`FetcherProtocol` does three things:

* ``supports(url)`` — declare whether this fetcher knows the URL's
  archive (used by :func:`dispatch_for` for fetch_kind=``"generic"``
  fallback inference).
* ``fetch(request, client, *, etag, last_modified)`` — issue the HTTP
  GET via :class:`BaseHTTPClient`, returning either an
  :class:`HTTPResponse` or :data:`NOT_MODIFIED`.
* ``derive_content_type(request, response)`` — normalise the
  response content-type to the L1 parser's expected label
  (e.g., ``CPAE_PDF_CONTENT_TYPE``).

The registry maps each :class:`CrawlRequest.fetch_kind` string to one
fetcher class. Adding a new archive requires (a) a new fetcher
module, (b) extending :data:`VALID_FETCH_KINDS` in :mod:`records`,
and (c) registering here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from lifeform_domain_figure.crawl.fetchers.cpae import CPAEFetcher
from lifeform_domain_figure.crawl.fetchers.generic import GenericHTTPFetcher
from lifeform_domain_figure.crawl.fetchers.gutenberg import GutenbergFetcher
from lifeform_domain_figure.crawl.fetchers.internet_archive import (
    InternetArchiveFetcher,
)
from lifeform_domain_figure.crawl.fetchers.wikisource import WikisourceFetcher
from lifeform_domain_figure.crawl.http_client import (
    BaseHTTPClient,
    HTTPResponse,
    NOT_MODIFIED,
)
from lifeform_domain_figure.crawl.records import CrawlRequest


class FetcherProtocol(Protocol):
    fetch_kind: str

    def supports(self, url: str) -> bool: ...

    def fetch(
        self,
        request: CrawlRequest,
        client: BaseHTTPClient,
        *,
        etag: str = "",
        last_modified: str = "",
    ) -> HTTPResponse | type(NOT_MODIFIED): ...

    def derive_content_type(
        self, request: CrawlRequest, response: HTTPResponse
    ) -> str: ...


_FETCHER_FACTORIES: dict[str, Callable[[], FetcherProtocol]] = {
    "generic": GenericHTTPFetcher,
    "cpae": CPAEFetcher,
    "wikisource": WikisourceFetcher,
    "gutenberg": GutenbergFetcher,
    "internet_archive": InternetArchiveFetcher,
}


def build_default_fetchers() -> dict[str, FetcherProtocol]:
    """Construct one instance of each registered fetcher.

    The scheduler keeps a per-fetch_kind instance for the duration of
    a run; fetchers may carry small per-instance state (e.g.,
    Wikisource's last-used-path flag), but never durable state.
    """

    return {kind: factory() for kind, factory in _FETCHER_FACTORIES.items()}


def dispatch_for(
    request: CrawlRequest, fetchers: dict[str, FetcherProtocol]
) -> FetcherProtocol:
    """Return the fetcher for ``request.fetch_kind``.

    Raises :class:`KeyError` if no fetcher is registered for the
    declared kind.
    """

    fetcher = fetchers.get(request.fetch_kind)
    if fetcher is None:
        raise KeyError(
            f"dispatch_for: no fetcher registered for fetch_kind="
            f"{request.fetch_kind!r}"
        )
    return fetcher


__all__ = [
    "CPAEFetcher",
    "FetcherProtocol",
    "GenericHTTPFetcher",
    "GutenbergFetcher",
    "InternetArchiveFetcher",
    "WikisourceFetcher",
    "build_default_fetchers",
    "dispatch_for",
]
