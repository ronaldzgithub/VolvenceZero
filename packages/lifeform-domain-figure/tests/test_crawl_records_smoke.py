"""Smoke tests for L0 crawl record schema (debt #28)."""

from __future__ import annotations

import pytest

from lifeform_domain_figure.crawl.records import (
    CrawlRequest,
    CrawlResult,
    CrawlStatus,
    VALID_FETCH_KINDS,
    request_id_for,
)


_SHA = "a" * 64
_TS = "2026-05-10T12:00:00+00:00"


def test_valid_fetch_kinds_set() -> None:
    assert VALID_FETCH_KINDS == frozenset(
        {"generic", "cpae", "wikisource", "gutenberg", "internet_archive"}
    )


def test_request_build_helper_derives_request_id() -> None:
    request = CrawlRequest.build(
        url="https://example.org/x",
        fetch_kind="generic",
        enqueued_at_iso=_TS,
    )
    assert request.request_id == request_id_for("generic", "https://example.org/x")
    assert len(request.request_id) == 64


def test_request_id_must_match_url_and_kind() -> None:
    with pytest.raises(ValueError, match="request_id mismatch"):
        CrawlRequest(
            url="https://example.org/x",
            fetch_kind="generic",
            request_id="a" * 64,
            enqueued_at_iso=_TS,
        )


def test_request_rejects_unknown_fetch_kind() -> None:
    with pytest.raises(ValueError, match="fetch_kind must be one of"):
        CrawlRequest.build(
            url="https://example.org/x",
            fetch_kind="bogus",
            enqueued_at_iso=_TS,
        )


def test_result_success_requires_raw_sha_and_content_type() -> None:
    request = CrawlRequest.build(
        url="https://example.org/x",
        fetch_kind="generic",
        enqueued_at_iso=_TS,
    )
    with pytest.raises(ValueError, match="raw_sha256"):
        CrawlResult(
            request=request,
            status=CrawlStatus.SUCCESS,
            fetched_at_iso=_TS,
            raw_sha256="",
            content_type_actual="text/plain",
            byte_len=10,
        )


def test_result_terminal_must_omit_raw_sha() -> None:
    request = CrawlRequest.build(
        url="https://example.org/x",
        fetch_kind="generic",
        enqueued_at_iso=_TS,
    )
    with pytest.raises(ValueError, match="raw_sha256 must be empty"):
        CrawlResult(
            request=request,
            status=CrawlStatus.SKIPPED_ROBOTS,
            fetched_at_iso=_TS,
            raw_sha256=_SHA,
        )


def test_result_success_round_trip() -> None:
    request = CrawlRequest.build(
        url="https://example.org/x",
        fetch_kind="generic",
        enqueued_at_iso=_TS,
    )
    result = CrawlResult(
        request=request,
        status=CrawlStatus.SUCCESS,
        fetched_at_iso=_TS,
        raw_sha256=_SHA,
        content_type_actual="text/plain",
        byte_len=10,
        http_status=200,
        etag='"abc"',
        last_modified="Thu, 01 Jan 1970 00:00:00 GMT",
    )
    assert result.status is CrawlStatus.SUCCESS
    assert result.byte_len == 10
