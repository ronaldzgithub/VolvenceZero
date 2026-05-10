"""CPAE (Princeton Collected Papers of Albert Einstein) fetcher.

URL pattern (typical)::

    https://einsteinpapers.press.princeton.edu/vol{N}-doc/{ID}/pdf

The fetcher recognises the ``einsteinpapers.press.princeton.edu``
host. It always declares the cleaned content type as
``application/pdf`` because the L1 :func:`parse_cpae_pdf` parser
expects PDF bytes; the response ``Content-Type`` header may carry
charset suffixes or generic ``application/octet-stream`` and we
normalise here.
"""

from __future__ import annotations

from urllib.parse import urlparse

from lifeform_domain_figure.cleaning.parsers.cpae_pdf import CPAE_PDF_CONTENT_TYPE
from lifeform_domain_figure.crawl.http_client import (
    BaseHTTPClient,
    HTTPResponse,
    NOT_MODIFIED,
)
from lifeform_domain_figure.crawl.records import CrawlRequest
from lifeform_domain_figure.crawl.scope_policy import ScopeRole


CPAE_HOST = "einsteinpapers.press.princeton.edu"


class CPAEFetcher:
    """Fetch a CPAE document PDF."""

    fetch_kind = "cpae"

    def supports(self, url: str) -> bool:
        return (urlparse(url).hostname or "").lower() == CPAE_HOST

    def fetch(
        self,
        request: CrawlRequest,
        client: BaseHTTPClient,
        *,
        etag: str = "",
        last_modified: str = "",
    ) -> HTTPResponse | type(NOT_MODIFIED):
        return client.get(
            request.url,
            etag=etag,
            last_modified=last_modified,
            accept="application/pdf",
            required_role=ScopeRole.CORPUS_FETCH,
        )

    def derive_content_type(self, request: CrawlRequest, response: HTTPResponse) -> str:
        return CPAE_PDF_CONTENT_TYPE


__all__ = ["CPAE_HOST", "CPAEFetcher"]
