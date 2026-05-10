"""Smoke tests for end-to-end L0 CrawlScheduler."""

from __future__ import annotations

from pathlib import Path

from lifeform_domain_figure.cleaning.store import CleaningStore
from lifeform_domain_figure.crawl.fetchers import build_default_fetchers
from lifeform_domain_figure.crawl.frontier import CrawlFrontier
from lifeform_domain_figure.crawl.http_client import BaseHTTPClient
from lifeform_domain_figure.crawl.rate_limiter import TokenBucketRateLimiter
from lifeform_domain_figure.crawl.records import CrawlRequest, CrawlStatus
from lifeform_domain_figure.crawl.robots import RobotsRegistry
from lifeform_domain_figure.crawl.scheduler import CrawlScheduler
from lifeform_domain_figure.crawl.scope_policy import ScopePolicy, ScopeRole
from lifeform_domain_figure.crawl.sink import CrawlSink
from crawl_mocks import FakeSession, make_response


_TS = "2026-05-10T12:00:00+00:00"


def _scope() -> ScopePolicy:
    return ScopePolicy(
        allowed_hosts=frozenset({"example.org"}),
        user_agent="test-agent/1",
        host_roles={"example.org": frozenset({ScopeRole.CORPUS_FETCH})},
    )


def _enqueue(frontier: CrawlFrontier, url: str, kind: str = "generic") -> None:
    request = CrawlRequest.build(url=url, fetch_kind=kind, enqueued_at_iso=_TS)
    frontier.enqueue(request)


def _build_scheduler(
    handler,
    *,
    tmp_path: Path,
    rate_per_second: float = 100.0,
    burst: int = 100,
) -> tuple[CrawlScheduler, CrawlFrontier]:
    scope = _scope()
    client = BaseHTTPClient(scope=scope, session=FakeSession(handler))
    robots = RobotsRegistry(http_client=client)
    rate_limiter = TokenBucketRateLimiter(rate_per_second=rate_per_second, burst=burst)
    frontier = CrawlFrontier(root=tmp_path / "crawl", run_id="r1")
    cleaning_store = CleaningStore(tmp_path / "cleaning")
    sink = CrawlSink(cleaning_store=cleaning_store, frontier=frontier)
    scheduler = CrawlScheduler(
        scope=scope,
        http_client=client,
        robots=robots,
        rate_limiter=rate_limiter,
        frontier=frontier,
        sink=sink,
        fetchers=build_default_fetchers(),
        sleep_fn=lambda _s: None,
    )
    return scheduler, frontier


def test_success_path(tmp_path: Path) -> None:
    def handler(url, headers):
        if url.endswith("/robots.txt"):
            return make_response(
                status_code=200,
                body=b"User-agent: *\nAllow: /\n",
                content_type="text/plain",
                url=url,
            )
        return make_response(
            status_code=200,
            body=b"hello",
            content_type="text/plain",
            url=url,
        )

    scheduler, frontier = _build_scheduler(handler, tmp_path=tmp_path)
    _enqueue(frontier, "https://example.org/x")
    results = scheduler.run()
    assert len(results) == 1
    assert results[0].status is CrawlStatus.SUCCESS
    assert results[0].byte_len == 5


def test_skipped_robots(tmp_path: Path) -> None:
    def handler(url, headers):
        if url.endswith("/robots.txt"):
            return make_response(
                status_code=200,
                body=b"User-agent: *\nDisallow: /\n",
                content_type="text/plain",
                url=url,
            )
        raise AssertionError("scheduler should not fetch document after disallow")

    scheduler, frontier = _build_scheduler(handler, tmp_path=tmp_path)
    _enqueue(frontier, "https://example.org/x")
    results = scheduler.run()
    assert len(results) == 1
    assert results[0].status is CrawlStatus.SKIPPED_ROBOTS


def test_skipped_scope(tmp_path: Path) -> None:
    def handler(url, headers):
        raise AssertionError("scheduler should not fetch out-of-scope URL")

    scheduler, frontier = _build_scheduler(handler, tmp_path=tmp_path)
    _enqueue(frontier, "https://evil.example.com/x")
    results = scheduler.run()
    assert len(results) == 1
    assert results[0].status is CrawlStatus.SKIPPED_SCOPE
    assert "not in scope" in results[0].error


def test_skipped_rate(tmp_path: Path) -> None:
    def handler(url, headers):
        if url.endswith("/robots.txt"):
            return make_response(
                status_code=200,
                body=b"User-agent: *\nAllow: /\n",
                content_type="text/plain",
                url=url,
            )
        return make_response(
            status_code=200, body=b"x", content_type="text/plain", url=url
        )

    scheduler, frontier = _build_scheduler(
        handler, tmp_path=tmp_path, rate_per_second=0.001, burst=1
    )
    _enqueue(frontier, "https://example.org/a")
    _enqueue(frontier, "https://example.org/b")
    results = scheduler.run()
    statuses = [r.status for r in results]
    assert CrawlStatus.SUCCESS in statuses
    assert CrawlStatus.SKIPPED_RATE in statuses


def test_failed_http(tmp_path: Path) -> None:
    def handler(url, headers):
        if url.endswith("/robots.txt"):
            return make_response(
                status_code=200,
                body=b"User-agent: *\nAllow: /\n",
                content_type="text/plain",
                url=url,
            )
        return make_response(status_code=500, url=url)

    scope = _scope()
    client = BaseHTTPClient(scope=scope, session=FakeSession(handler), retries=0)
    robots = RobotsRegistry(http_client=client)
    rate_limiter = TokenBucketRateLimiter(rate_per_second=100.0, burst=100)
    frontier = CrawlFrontier(root=tmp_path / "crawl", run_id="r1")
    cleaning_store = CleaningStore(tmp_path / "cleaning")
    sink = CrawlSink(cleaning_store=cleaning_store, frontier=frontier)
    scheduler = CrawlScheduler(
        scope=scope,
        http_client=client,
        robots=robots,
        rate_limiter=rate_limiter,
        frontier=frontier,
        sink=sink,
        sleep_fn=lambda _s: None,
    )
    _enqueue(frontier, "https://example.org/x")
    results = scheduler.run()
    assert len(results) == 1
    assert results[0].status is CrawlStatus.FAILED_HTTP


def test_max_pages_per_host_caps_run(tmp_path: Path) -> None:
    def handler(url, headers):
        if url.endswith("/robots.txt"):
            return make_response(
                status_code=200,
                body=b"User-agent: *\nAllow: /\n",
                content_type="text/plain",
                url=url,
            )
        return make_response(
            status_code=200, body=b"x", content_type="text/plain", url=url
        )

    scope = ScopePolicy(
        allowed_hosts=frozenset({"example.org"}),
        user_agent="test-agent/1",
        host_roles={"example.org": frozenset({ScopeRole.CORPUS_FETCH})},
        max_pages_per_host=2,
    )
    client = BaseHTTPClient(scope=scope, session=FakeSession(handler))
    robots = RobotsRegistry(http_client=client)
    rate_limiter = TokenBucketRateLimiter(rate_per_second=100.0, burst=100)
    frontier = CrawlFrontier(root=tmp_path / "crawl", run_id="r1")
    cleaning_store = CleaningStore(tmp_path / "cleaning")
    sink = CrawlSink(cleaning_store=cleaning_store, frontier=frontier)
    scheduler = CrawlScheduler(
        scope=scope,
        http_client=client,
        robots=robots,
        rate_limiter=rate_limiter,
        frontier=frontier,
        sink=sink,
        sleep_fn=lambda _s: None,
    )
    for n in range(4):
        _enqueue(frontier, f"https://example.org/p{n}")
    results = scheduler.run()
    success_count = sum(1 for r in results if r.status is CrawlStatus.SUCCESS)
    skipped_scope_count = sum(
        1 for r in results if r.status is CrawlStatus.SKIPPED_SCOPE
    )
    assert success_count == 2
    assert skipped_scope_count == 2
