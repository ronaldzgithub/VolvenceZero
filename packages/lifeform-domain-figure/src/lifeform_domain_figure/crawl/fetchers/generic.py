"""Generic HTTP fetcher.

Used for any URL that does not match an archive-specific fetcher.
The content type is taken verbatim from the response header (or the
caller-supplied ``CrawlRequest.expected_content_type`` when the
response is missing or generic). No URL pattern matching.
"""

from __future__ import annotations

from lifeform_domain_figure.crawl.http_client import (
    BaseHTTPClient,
    HTTPResponse,
    NOT_MODIFIED,
)
from lifeform_domain_figure.crawl.records import CrawlRequest
from lifeform_domain_figure.crawl.scope_policy import ScopeRole


class GenericHTTPFetcher:
    """Pass-through fetcher with no archive-specific logic."""

    fetch_kind = "generic"

    def supports(self, url: str) -> bool:
        return True

    def fetch(
        self,
        request: CrawlRequest,
        client: BaseHTTPClient,
        *,
        etag: str = "",
        last_modified: str = "",
    ) -> HTTPResponse | type(NOT_MODIFIED):
        response = client.get(
            request.url,
            etag=etag,
            last_modified=last_modified,
            required_role=ScopeRole.CORPUS_FETCH,
        )
        return response

    def derive_content_type(self, request: CrawlRequest, response: HTTPResponse) -> str:
        if response.content_type:
            return response.content_type
        if request.expected_content_type:
            return request.expected_content_type
        return "application/octet-stream"


__all__ = ["GenericHTTPFetcher"]
