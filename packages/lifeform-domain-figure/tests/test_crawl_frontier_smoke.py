"""Smoke tests for the L0 CrawlFrontier persistence layer."""

from __future__ import annotations

from pathlib import Path

from lifeform_domain_figure.crawl.frontier import CrawlFrontier
from lifeform_domain_figure.crawl.records import (
    CrawlRequest,
    CrawlResult,
    CrawlStatus,
)


_TS = "2026-05-10T12:00:00+00:00"


def _request(url: str, kind: str = "generic") -> CrawlRequest:
    return CrawlRequest.build(
        url=url, fetch_kind=kind, enqueued_at_iso=_TS
    )


def test_enqueue_and_pop_fifo(tmp_path: Path) -> None:
    frontier = CrawlFrontier(root=tmp_path, run_id="run-1")
    a = _request("https://example.org/a")
    b = _request("https://example.org/b")
    assert frontier.enqueue(a)
    assert frontier.enqueue(b)
    assert frontier.pending_count() == 2
    popped = frontier.next()
    assert popped is not None and popped.url == "https://example.org/a"
    popped = frontier.next()
    assert popped is not None and popped.url == "https://example.org/b"
    assert frontier.next() is None


def test_enqueue_dedup_against_pending(tmp_path: Path) -> None:
    frontier = CrawlFrontier(root=tmp_path, run_id="run-1")
    request = _request("https://example.org/x")
    assert frontier.enqueue(request)
    assert not frontier.enqueue(request)
    assert frontier.pending_count() == 1


def test_enqueue_dedup_against_visited(tmp_path: Path) -> None:
    frontier = CrawlFrontier(root=tmp_path, run_id="run-1")
    request = _request("https://example.org/x")
    frontier.enqueue(request)
    frontier.next()
    assert not frontier.enqueue(request)
    assert frontier.visited_count() == 1


def test_resume_from_disk_rebuilds_pending_minus_visited(tmp_path: Path) -> None:
    frontier = CrawlFrontier(root=tmp_path, run_id="run-1")
    a = _request("https://example.org/a")
    b = _request("https://example.org/b")
    c = _request("https://example.org/c")
    frontier.enqueue(a)
    frontier.enqueue(b)
    frontier.enqueue(c)
    frontier.next()
    resumed = CrawlFrontier.resume_from_disk(root=tmp_path, run_id="run-1")
    assert resumed.pending_count() == 2
    assert resumed.visited_count() == 1
    next_after = resumed.next()
    assert next_after is not None and next_after.url == "https://example.org/b"


def test_record_result_appends(tmp_path: Path) -> None:
    frontier = CrawlFrontier(root=tmp_path, run_id="run-1")
    request = _request("https://example.org/x")
    frontier.enqueue(request)
    frontier.next()
    result = CrawlResult(
        request=request,
        status=CrawlStatus.SKIPPED_ROBOTS,
        fetched_at_iso=_TS,
        error="robots.txt disallow",
    )
    frontier.record_result(result)
    fetched = list(frontier.iter_results())
    assert len(fetched) == 1
    assert fetched[0].status is CrawlStatus.SKIPPED_ROBOTS
    assert fetched[0].error == "robots.txt disallow"
