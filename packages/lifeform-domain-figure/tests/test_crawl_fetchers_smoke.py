"""Smoke tests for the 5 L0 archive-aware fetchers."""

from __future__ import annotations

import json

from lifeform_domain_figure.cleaning.parsers.archive_org_ocr import (
    ARCHIVE_ORG_OCR_CONTENT_TYPE,
)
from lifeform_domain_figure.cleaning.parsers.cpae_pdf import CPAE_PDF_CONTENT_TYPE
from lifeform_domain_figure.cleaning.parsers.gutenberg import (
    GUTENBERG_HTML_CONTENT_TYPE,
    GUTENBERG_TEXT_CONTENT_TYPE,
)
from lifeform_domain_figure.cleaning.parsers.wikisource_html import (
    WIKISOURCE_HTML_CONTENT_TYPE,
    WIKISOURCE_WIKITEXT_CONTENT_TYPE,
)
from lifeform_domain_figure.crawl.fetchers import (
    CPAEFetcher,
    GenericHTTPFetcher,
    GutenbergFetcher,
    InternetArchiveFetcher,
    WikisourceFetcher,
    build_default_fetchers,
    dispatch_for,
)
from lifeform_domain_figure.crawl.http_client import BaseHTTPClient, HTTPResponse
from lifeform_domain_figure.crawl.records import CrawlRequest
from lifeform_domain_figure.crawl.scope_policy import default_scope_policy
from crawl_mocks import FakeSession, make_response


_TS = "2026-05-10T12:00:00+00:00"


def _client(handler):
    scope = default_scope_policy("test-agent/1")
    return BaseHTTPClient(scope=scope, session=FakeSession(handler))


def test_generic_passthrough_content_type() -> None:
    fetcher = GenericHTTPFetcher()
    request = CrawlRequest.build(
        url="https://archive.org/x",
        fetch_kind="generic",
        enqueued_at_iso=_TS,
    )
    client = _client(
        lambda url, headers: make_response(
            status_code=200, body=b"data", content_type="application/json", url=url
        )
    )
    response = fetcher.fetch(request, client)
    assert isinstance(response, HTTPResponse)
    assert fetcher.derive_content_type(request, response) == "application/json"


def test_generic_uses_expected_when_response_blank() -> None:
    fetcher = GenericHTTPFetcher()
    request = CrawlRequest.build(
        url="https://archive.org/x",
        fetch_kind="generic",
        enqueued_at_iso=_TS,
        expected_content_type="text/csv",
    )
    response = HTTPResponse(
        url_final="https://archive.org/x",
        http_status=200,
        content_type="",
        body=b"",
        etag="",
        last_modified="",
    )
    assert fetcher.derive_content_type(request, response) == "text/csv"


def test_cpae_supports_only_princeton_host() -> None:
    fetcher = CPAEFetcher()
    assert fetcher.supports("https://einsteinpapers.press.princeton.edu/vol2-doc/24/pdf")
    assert not fetcher.supports("https://en.wikisource.org/wiki/Foo")


def test_cpae_forces_pdf_content_type() -> None:
    fetcher = CPAEFetcher()
    request = CrawlRequest.build(
        url="https://einsteinpapers.press.princeton.edu/vol2-doc/24/pdf",
        fetch_kind="cpae",
        enqueued_at_iso=_TS,
    )
    client = _client(
        lambda url, headers: make_response(
            status_code=200,
            body=b"%PDF-1.4\n",
            content_type="application/octet-stream",
            url=url,
        )
    )
    response = fetcher.fetch(request, client)
    assert isinstance(response, HTTPResponse)
    assert fetcher.derive_content_type(request, response) == CPAE_PDF_CONTENT_TYPE


def test_wikisource_action_raw_first() -> None:
    fetcher = WikisourceFetcher()
    request = CrawlRequest.build(
        url="https://en.wikisource.org/wiki/Annus_Mirabilis_Letter",
        fetch_kind="wikisource",
        enqueued_at_iso=_TS,
    )
    seen_urls: list[str] = []

    def handler(url, headers):
        seen_urls.append(url)
        return make_response(
            status_code=200, body=b"== heading ==", content_type="text/x-wiki", url=url
        )

    client = _client(handler)
    response = fetcher.fetch(request, client)
    assert isinstance(response, HTTPResponse)
    assert "action=raw" in seen_urls[0]
    assert fetcher.derive_content_type(request, response) == WIKISOURCE_WIKITEXT_CONTENT_TYPE


def test_wikisource_falls_back_to_html() -> None:
    fetcher = WikisourceFetcher()
    request = CrawlRequest.build(
        url="https://en.wikisource.org/wiki/Annus_Mirabilis_Letter",
        fetch_kind="wikisource",
        enqueued_at_iso=_TS,
    )

    def handler(url, headers):
        if "action=raw" in url:
            return make_response(status_code=404, url=url)
        return make_response(
            status_code=200,
            body=b"<html></html>",
            content_type="text/html",
            url=url,
        )

    client = _client(handler)
    response = fetcher.fetch(request, client)
    assert isinstance(response, HTTPResponse)
    assert fetcher.derive_content_type(request, response) == WIKISOURCE_HTML_CONTENT_TYPE


def test_wikisource_supports_subdomain() -> None:
    fetcher = WikisourceFetcher()
    assert fetcher.supports("https://de.wikisource.org/wiki/Foo")
    assert not fetcher.supports("https://example.org/wiki/Foo")


def test_gutenberg_landing_rewritten_to_text() -> None:
    fetcher = GutenbergFetcher()
    request = CrawlRequest.build(
        url="https://www.gutenberg.org/ebooks/12345",
        fetch_kind="gutenberg",
        enqueued_at_iso=_TS,
    )
    seen: list[str] = []

    def handler(url, headers):
        seen.append(url)
        return make_response(status_code=200, body=b"plain text", content_type="text/plain", url=url)

    client = _client(handler)
    response = fetcher.fetch(request, client)
    assert isinstance(response, HTTPResponse)
    assert seen[0].endswith("/files/12345/12345-0.txt")
    assert fetcher.derive_content_type(request, response) == GUTENBERG_TEXT_CONTENT_TYPE


def test_gutenberg_falls_back_to_html_on_text_404() -> None:
    fetcher = GutenbergFetcher()
    request = CrawlRequest.build(
        url="https://www.gutenberg.org/ebooks/12345",
        fetch_kind="gutenberg",
        enqueued_at_iso=_TS,
    )

    def handler(url, headers):
        if url.endswith(".txt"):
            return make_response(status_code=404, url=url)
        return make_response(status_code=200, body=b"<html></html>", content_type="text/html", url=url)

    client = _client(handler)
    response = fetcher.fetch(request, client)
    assert isinstance(response, HTTPResponse)
    assert fetcher.derive_content_type(request, response) == GUTENBERG_HTML_CONTENT_TYPE


def test_gutenberg_passes_explicit_text_url() -> None:
    fetcher = GutenbergFetcher()
    request = CrawlRequest.build(
        url="https://www.gutenberg.org/files/12345/12345-0.txt",
        fetch_kind="gutenberg",
        enqueued_at_iso=_TS,
    )
    seen: list[str] = []

    def handler(url, headers):
        seen.append(url)
        return make_response(status_code=200, body=b"plain", content_type="text/plain", url=url)

    client = _client(handler)
    fetcher.fetch(request, client)
    assert seen == ["https://www.gutenberg.org/files/12345/12345-0.txt"]


def test_internet_archive_metadata_then_ocr_chain() -> None:
    fetcher = InternetArchiveFetcher()
    request = CrawlRequest.build(
        url="https://archive.org/details/sample-id",
        fetch_kind="internet_archive",
        enqueued_at_iso=_TS,
    )
    seen: list[str] = []

    def handler(url, headers):
        seen.append(url)
        if url.endswith("/metadata/sample-id"):
            payload = {
                "files": [
                    {"name": "sample-id_djvu.json"},
                    {"name": "cover.jpg"},
                ]
            }
            return make_response(
                status_code=200,
                body=json.dumps(payload).encode("utf-8"),
                content_type="application/json",
                url=url,
            )
        return make_response(
            status_code=200,
            body=b'{"ocr":[{"page":1,"text":"x","confidence":0.9}],"metadata":{}}',
            content_type="application/json",
            url=url,
        )

    client = _client(handler)
    response = fetcher.fetch(request, client)
    assert isinstance(response, HTTPResponse)
    assert any(u.endswith("/metadata/sample-id") for u in seen)
    assert any(u.endswith("/download/sample-id/sample-id_djvu.json") for u in seen)
    assert fetcher.derive_content_type(request, response) == ARCHIVE_ORG_OCR_CONTENT_TYPE


def test_internet_archive_no_ocr_file_raises() -> None:
    from lifeform_domain_figure.crawl.http_client import FetchError

    fetcher = InternetArchiveFetcher()
    request = CrawlRequest.build(
        url="https://archive.org/details/sample-id",
        fetch_kind="internet_archive",
        enqueued_at_iso=_TS,
    )

    def handler(url, headers):
        return make_response(
            status_code=200,
            body=json.dumps({"files": [{"name": "cover.jpg"}]}).encode("utf-8"),
            content_type="application/json",
            url=url,
        )

    import pytest

    client = _client(handler)
    with pytest.raises(FetchError, match="no OCR JSON file"):
        fetcher.fetch(request, client)


def test_dispatch_for_picks_correct_fetcher() -> None:
    fetchers = build_default_fetchers()
    request = CrawlRequest.build(
        url="https://en.wikisource.org/wiki/Foo",
        fetch_kind="wikisource",
        enqueued_at_iso=_TS,
    )
    assert isinstance(dispatch_for(request, fetchers), WikisourceFetcher)
