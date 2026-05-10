"""Internet Archive fetcher.

URL pattern (typical seed)::

    https://archive.org/details/{identifier}

Strategy:

1. GET the metadata API (``https://archive.org/metadata/{identifier}``)
   which returns a JSON listing of every file in the item.
2. Find an OCR JSON file (preferred filenames: ``*_djvu.json`` or
   anything ending in ``_chocr.html.json``) within the file list.
3. GET that file directly and return its bytes; the L1 parser expects
   ``application/json; profile=archive-org-ocr``.

The fetcher refuses to proceed when the metadata response shape is
unexpected; rather than guess at file names it raises
:class:`FetchError` so the scheduler records FAILED_PARSER_PRECHECK.

Subdomain redirects: archive.org commonly redirects file fetches to
``ia801.us.archive.org`` / ``ia902.us.archive.org`` (those subdomains
are in the default scope allowlist; the BaseHTTPClient honours one
hop).
"""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from lifeform_domain_figure.cleaning.parsers.archive_org_ocr import (
    ARCHIVE_ORG_OCR_CONTENT_TYPE,
)
from lifeform_domain_figure.crawl.http_client import (
    BaseHTTPClient,
    FetchError,
    HTTPResponse,
    NOT_MODIFIED,
)
from lifeform_domain_figure.crawl.records import CrawlRequest
from lifeform_domain_figure.crawl.scope_policy import ScopeRole


INTERNET_ARCHIVE_HOSTS = (
    "archive.org",
    "ia801.us.archive.org",
    "ia902.us.archive.org",
)
_DETAILS_RE = re.compile(r"^/details/([^/]+)/?$")
_OCR_JSON_NAME_RE = re.compile(
    r"(?:_djvu\.json|_chocr\.html\.json|_ocr\.json)$",
    re.IGNORECASE,
)


def _identifier_from_details_url(url: str) -> str:
    parsed = urlparse(url)
    match = _DETAILS_RE.match(parsed.path)
    if match is None:
        raise FetchError(
            f"InternetArchiveFetcher: url {url!r} is not an /details/{{id}} URL"
        )
    return match.group(1)


class InternetArchiveFetcher:
    """Fetch the OCR JSON for an Internet Archive item via the metadata API."""

    fetch_kind = "internet_archive"

    def supports(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return host in INTERNET_ARCHIVE_HOSTS

    def fetch(
        self,
        request: CrawlRequest,
        client: BaseHTTPClient,
        *,
        etag: str = "",
        last_modified: str = "",
    ) -> HTTPResponse | type(NOT_MODIFIED):
        identifier = _identifier_from_details_url(request.url)
        metadata_url = f"https://archive.org/metadata/{identifier}"
        metadata_response = client.get(
            metadata_url,
            accept="application/json",
            required_role=ScopeRole.CORPUS_FETCH,
        )
        if metadata_response is NOT_MODIFIED:
            raise FetchError(
                "InternetArchiveFetcher: metadata API unexpectedly returned "
                f"304 for identifier={identifier!r}"
            )
        assert isinstance(metadata_response, HTTPResponse)
        try:
            metadata_payload = json.loads(metadata_response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FetchError(
                f"InternetArchiveFetcher: metadata for identifier={identifier!r} "
                f"is not valid JSON ({exc})"
            ) from exc
        files = metadata_payload.get("files")
        if not isinstance(files, list) or not files:
            raise FetchError(
                f"InternetArchiveFetcher: metadata.files missing/empty for "
                f"identifier={identifier!r}"
            )
        ocr_filename: str | None = None
        for entry in files:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if isinstance(name, str) and _OCR_JSON_NAME_RE.search(name):
                ocr_filename = name
                break
        if ocr_filename is None:
            raise FetchError(
                f"InternetArchiveFetcher: no OCR JSON file (*_djvu.json / "
                f"*_chocr.html.json / *_ocr.json) found for identifier="
                f"{identifier!r}"
            )
        ocr_url = f"https://archive.org/download/{identifier}/{ocr_filename}"
        return client.get(
            ocr_url,
            etag=etag,
            last_modified=last_modified,
            accept="application/json",
            required_role=ScopeRole.CORPUS_FETCH,
        )

    def derive_content_type(self, request: CrawlRequest, response: HTTPResponse) -> str:
        return ARCHIVE_ORG_OCR_CONTENT_TYPE


__all__ = ["INTERNET_ARCHIVE_HOSTS", "InternetArchiveFetcher"]
