"""Wikisource fetcher.

URL pattern::

    https://{lang}.wikisource.org/wiki/{Title}

Strategy:

* If the request URL already includes a ``?action=raw`` query, fetch
  it as wikitext (``text/x-wiki``).
* Otherwise rewrite the URL to add ``?action=raw`` and try first; on
  failure fall back to the rendered HTML at the original URL
  (``text/html; profile=wikisource``).

The L1 :func:`parse_wikisource_html` parser accepts both
``WIKISOURCE_HTML_CONTENT_TYPE`` and
``WIKISOURCE_WIKITEXT_CONTENT_TYPE`` so either path lands in the
right cleaner branch.
"""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse

from lifeform_domain_figure.cleaning.parsers.wikisource_html import (
    WIKISOURCE_HTML_CONTENT_TYPE,
    WIKISOURCE_WIKITEXT_CONTENT_TYPE,
)
from lifeform_domain_figure.crawl.http_client import (
    BaseHTTPClient,
    FetchError,
    HTTPResponse,
    NOT_MODIFIED,
)
from lifeform_domain_figure.crawl.records import CrawlRequest
from lifeform_domain_figure.crawl.scope_policy import ScopeRole


WIKISOURCE_HOST_SUFFIX = ".wikisource.org"


def _with_action_raw(url: str) -> str:
    parsed = urlparse(url)
    query = parsed.query
    if "action=raw" in query:
        return url
    new_query = "action=raw" if not query else f"{query}&action=raw"
    return urlunparse(parsed._replace(query=new_query))


class WikisourceFetcher:
    """Fetch a Wikisource page as wikitext (preferred) or HTML."""

    fetch_kind = "wikisource"

    def __init__(self) -> None:
        self._last_used_wikitext_path = False

    def supports(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return host.endswith(WIKISOURCE_HOST_SUFFIX)

    def fetch(
        self,
        request: CrawlRequest,
        client: BaseHTTPClient,
        *,
        etag: str = "",
        last_modified: str = "",
    ) -> HTTPResponse | type(NOT_MODIFIED):
        wikitext_url = _with_action_raw(request.url)
        try:
            response = client.get(
                wikitext_url,
                etag=etag,
                last_modified=last_modified,
                accept="text/x-wiki, text/plain",
                required_role=ScopeRole.CORPUS_FETCH,
            )
        except FetchError:
            self._last_used_wikitext_path = False
            return client.get(
                request.url,
                etag=etag,
                last_modified=last_modified,
                accept="text/html",
                required_role=ScopeRole.CORPUS_FETCH,
            )
        self._last_used_wikitext_path = True
        return response

    def derive_content_type(self, request: CrawlRequest, response: HTTPResponse) -> str:
        if self._last_used_wikitext_path:
            return WIKISOURCE_WIKITEXT_CONTENT_TYPE
        return WIKISOURCE_HTML_CONTENT_TYPE


__all__ = ["WIKISOURCE_HOST_SUFFIX", "WikisourceFetcher"]
