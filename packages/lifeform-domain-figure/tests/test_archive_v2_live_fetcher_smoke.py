"""Smoke tests for the V2 live_archive_fetcher (debt #19 closure)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lifeform_domain_figure.cleaning.parsers.archive_org_ocr import (
    ARCHIVE_ORG_OCR_CONTENT_TYPE,
)
from lifeform_domain_figure.cleaning.parsers.cpae_pdf import CPAE_PDF_CONTENT_TYPE
from lifeform_domain_figure.cleaning.parsers.gutenberg import GUTENBERG_TEXT_CONTENT_TYPE
from lifeform_domain_figure.cleaning.parsers.wikisource_html import (
    WIKISOURCE_WIKITEXT_CONTENT_TYPE,
)
from lifeform_domain_figure.cleaning.store import CleaningStore
from lifeform_domain_figure.corpus.archives import (
    LiveFetchedBytes,
    live_archive_fetcher,
    offline_archive_fetcher,
)
from lifeform_domain_figure.crawl.http_client import BaseHTTPClient
from lifeform_domain_figure.crawl.scope_policy import default_scope_policy
from crawl_mocks import FakeSession, make_response


def _client(handler) -> BaseHTTPClient:
    scope = default_scope_policy("test-agent/1")
    return BaseHTTPClient(scope=scope, session=FakeSession(handler))


def test_offline_fetcher_still_raises() -> None:
    fetcher = offline_archive_fetcher()
    import pytest

    with pytest.raises(NotImplementedError):
        fetcher.fetch("https://en.wikisource.org/wiki/Foo")


def test_live_cpae_returns_pdf_payload(tmp_path: Path) -> None:
    body = b"%PDF-1.4\n%fake\n"

    def handler(url, headers):
        return make_response(
            status_code=200,
            body=body,
            content_type="application/octet-stream",
            url=url,
        )

    cleaning_store = CleaningStore(tmp_path / "cleaning")
    fetcher = live_archive_fetcher(
        "cpae",
        scope=default_scope_policy("test-agent/1"),
        http_client=_client(handler),
        cleaning_store=cleaning_store,
    )
    result = fetcher.fetch("https://einsteinpapers.press.princeton.edu/vol2-doc/24/pdf")
    assert isinstance(result.raw_payload, LiveFetchedBytes)
    assert result.raw_payload.content_type == CPAE_PDF_CONTENT_TYPE
    expected_sha = hashlib.sha256(body).hexdigest()
    assert result.raw_payload.raw_sha256 == expected_sha
    fetched_bytes, _sidecar = cleaning_store.get_raw(expected_sha)
    assert fetched_bytes == body


def test_live_wikisource_returns_wikitext_payload(tmp_path: Path) -> None:
    body = b"== heading ==\n\nbody text\n"

    def handler(url, headers):
        return make_response(
            status_code=200,
            body=body,
            content_type="text/x-wiki",
            url=url,
        )

    fetcher = live_archive_fetcher(
        "wikisource",
        http_client=_client(handler),
    )
    result = fetcher.fetch("https://en.wikisource.org/wiki/Foo")
    assert isinstance(result.raw_payload, LiveFetchedBytes)
    assert result.raw_payload.content_type == WIKISOURCE_WIKITEXT_CONTENT_TYPE
    assert result.raw_payload.body == body


def test_live_gutenberg_returns_plain_payload(tmp_path: Path) -> None:
    body = b"plain ebook body"

    def handler(url, headers):
        return make_response(
            status_code=200,
            body=body,
            content_type="text/plain",
            url=url,
        )

    fetcher = live_archive_fetcher(
        "gutenberg",
        http_client=_client(handler),
    )
    result = fetcher.fetch("https://www.gutenberg.org/files/12345/12345-0.txt")
    assert result.raw_payload.content_type == GUTENBERG_TEXT_CONTENT_TYPE


def test_live_internet_archive_returns_ocr_payload(tmp_path: Path) -> None:
    metadata_payload = json.dumps({"files": [{"name": "sample-id_djvu.json"}]}).encode("utf-8")
    ocr_payload = json.dumps(
        {"ocr": [{"page": 1, "text": "x", "confidence": 0.9}], "metadata": {}}
    ).encode("utf-8")

    def handler(url, headers):
        if url.endswith("/metadata/sample-id"):
            return make_response(
                status_code=200,
                body=metadata_payload,
                content_type="application/json",
                url=url,
            )
        return make_response(
            status_code=200,
            body=ocr_payload,
            content_type="application/json",
            url=url,
        )

    fetcher = live_archive_fetcher(
        "internet_archive",
        http_client=_client(handler),
    )
    result = fetcher.fetch("https://archive.org/details/sample-id")
    assert result.raw_payload.content_type == ARCHIVE_ORG_OCR_CONTENT_TYPE


def test_live_fetcher_rejects_out_of_scope() -> None:
    from lifeform_domain_figure.crawl.http_client import ScopeRejection
    import pytest

    fetcher = live_archive_fetcher(
        "generic",
        http_client=_client(lambda u, h: make_response()),
    )
    with pytest.raises(ScopeRejection):
        fetcher.fetch("https://evil.example.com/x")
