"""Project Gutenberg fetcher.

URL pattern (typical seed)::

    https://www.gutenberg.org/ebooks/{ID}
    https://www.gutenberg.org/files/{ID}/{ID}-0.txt
    https://www.gutenberg.org/files/{ID}/{ID}-h/{ID}-h.htm

Strategy:

* If the request URL already points at a ``.txt`` artefact, fetch
  it directly as ``text/plain``.
* If the URL points at the ``ebooks/{ID}`` landing page, rewrite to
  ``files/{ID}/{ID}-0.txt`` and prefer plain text.
* If the .txt fetch fails, fall back to the original URL and treat
  the response as HTML.

The L1 :func:`parse_gutenberg` parser accepts both
``GUTENBERG_TEXT_CONTENT_TYPE`` and ``GUTENBERG_HTML_CONTENT_TYPE``.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse

from lifeform_domain_figure.cleaning.parsers.gutenberg import (
    GUTENBERG_HTML_CONTENT_TYPE,
    GUTENBERG_TEXT_CONTENT_TYPE,
)
from lifeform_domain_figure.crawl.http_client import (
    BaseHTTPClient,
    FetchError,
    HTTPResponse,
    NOT_MODIFIED,
)
from lifeform_domain_figure.crawl.records import CrawlRequest
from lifeform_domain_figure.crawl.scope_policy import ScopeRole


GUTENBERG_HOSTS = ("www.gutenberg.org", "gutenberg.org")
_EBOOK_LANDING_RE = re.compile(r"^/ebooks/(\d+)/?$")


def _maybe_rewrite_to_text(url: str) -> tuple[str, bool]:
    """Return (preferred_url, is_text_attempt)."""

    parsed = urlparse(url)
    if parsed.path.endswith(".txt"):
        return url, True
    match = _EBOOK_LANDING_RE.match(parsed.path)
    if match is not None:
        ebook_id = match.group(1)
        new_path = f"/files/{ebook_id}/{ebook_id}-0.txt"
        return urlunparse(parsed._replace(path=new_path)), True
    return url, False


class GutenbergFetcher:
    """Fetch a Gutenberg book preferring .txt over .html."""

    fetch_kind = "gutenberg"

    def __init__(self) -> None:
        self._last_used_text_path = False

    def supports(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return host in GUTENBERG_HOSTS

    def fetch(
        self,
        request: CrawlRequest,
        client: BaseHTTPClient,
        *,
        etag: str = "",
        last_modified: str = "",
    ) -> HTTPResponse | type(NOT_MODIFIED):
        text_url, is_text = _maybe_rewrite_to_text(request.url)
        if is_text:
            try:
                response = client.get(
                    text_url,
                    etag=etag,
                    last_modified=last_modified,
                    accept="text/plain",
                    required_role=ScopeRole.CORPUS_FETCH,
                )
            except FetchError:
                self._last_used_text_path = False
                return client.get(
                    request.url,
                    etag=etag,
                    last_modified=last_modified,
                    accept="text/html",
                    required_role=ScopeRole.CORPUS_FETCH,
                )
            self._last_used_text_path = True
            return response
        self._last_used_text_path = False
        return client.get(
            request.url,
            etag=etag,
            last_modified=last_modified,
            accept="text/html",
            required_role=ScopeRole.CORPUS_FETCH,
        )

    def derive_content_type(self, request: CrawlRequest, response: HTTPResponse) -> str:
        if self._last_used_text_path:
            return GUTENBERG_TEXT_CONTENT_TYPE
        return GUTENBERG_HTML_CONTENT_TYPE


__all__ = ["GUTENBERG_HOSTS", "GutenbergFetcher"]
