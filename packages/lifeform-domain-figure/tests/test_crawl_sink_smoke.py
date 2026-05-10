"""Smoke tests for L0 CrawlSink wiring to L1 CleaningStore."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from lifeform_domain_figure.cleaning.store import CleaningStore
from lifeform_domain_figure.crawl.frontier import CrawlFrontier
from lifeform_domain_figure.crawl.http_client import HTTPResponse
from lifeform_domain_figure.crawl.records import (
    CrawlRequest,
    CrawlResult,
    CrawlStatus,
)
from lifeform_domain_figure.crawl.sink import CrawlSink


_TS = "2026-05-10T12:00:00+00:00"


def _request() -> CrawlRequest:
    return CrawlRequest.build(
        url="https://en.wikisource.org/wiki/Foo",
        fetch_kind="wikisource",
        enqueued_at_iso=_TS,
    )


def _response(body: bytes) -> HTTPResponse:
    return HTTPResponse(
        url_final="https://en.wikisource.org/wiki/Foo",
        http_status=200,
        content_type="text/html",
        body=body,
        etag='"v1"',
        last_modified="",
    )


def test_consume_success_writes_to_l1_store(tmp_path: Path) -> None:
    cleaning_store = CleaningStore(tmp_path / "cleaning")
    frontier = CrawlFrontier(root=tmp_path / "crawl-root", run_id="r1")
    sink = CrawlSink(cleaning_store=cleaning_store, frontier=frontier)
    body = b"<html>body</html>"
    request = _request()
    result = sink.consume_success(request, _response(body), "text/html; profile=wikisource")
    expected_sha = hashlib.sha256(body).hexdigest()
    assert result.status is CrawlStatus.SUCCESS
    assert result.raw_sha256 == expected_sha
    assert result.byte_len == len(body)
    fetched_bytes, sidecar = cleaning_store.get_raw(expected_sha)
    assert fetched_bytes == body
    assert sidecar.content_type == "text/html; profile=wikisource"


def test_consume_success_records_result(tmp_path: Path) -> None:
    cleaning_store = CleaningStore(tmp_path / "cleaning")
    frontier = CrawlFrontier(root=tmp_path / "crawl-root", run_id="r1")
    sink = CrawlSink(cleaning_store=cleaning_store, frontier=frontier)
    sink.consume_success(_request(), _response(b"data"), "text/plain")
    results = list(frontier.iter_results())
    assert len(results) == 1
    assert results[0].status is CrawlStatus.SUCCESS


def test_record_terminal_refuses_success(tmp_path: Path) -> None:
    cleaning_store = CleaningStore(tmp_path / "cleaning")
    frontier = CrawlFrontier(root=tmp_path / "crawl-root", run_id="r1")
    sink = CrawlSink(cleaning_store=cleaning_store, frontier=frontier)
    request = _request()
    bad = CrawlResult(
        request=request,
        status=CrawlStatus.SUCCESS,
        fetched_at_iso=_TS,
        raw_sha256="0" * 64,
        content_type_actual="text/plain",
        byte_len=1,
    )
    with pytest.raises(ValueError, match="must NOT be"):
        sink.record_terminal_result(bad)


def test_consume_success_blank_content_type_rejected(tmp_path: Path) -> None:
    cleaning_store = CleaningStore(tmp_path / "cleaning")
    frontier = CrawlFrontier(root=tmp_path / "crawl-root", run_id="r1")
    sink = CrawlSink(cleaning_store=cleaning_store, frontier=frontier)
    with pytest.raises(ValueError, match="content_type"):
        sink.consume_success(_request(), _response(b"x"), "")
