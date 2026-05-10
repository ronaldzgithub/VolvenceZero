"""Cross-cutting contract: crawler MUST respect robots.txt.

Two scenarios sufficient to fence the discipline:

1. A host serving ``Disallow: /`` -> every URL on that host must
   come back as :class:`CrawlStatus.SKIPPED_ROBOTS`.
2. A host serving ``Allow: /`` -> URLs are fetched normally.

The test mocks HTTP at the :class:`requests.Session` level so no
real network traffic occurs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIGURE_TESTS_DIR = (
    _REPO_ROOT / "packages" / "lifeform-domain-figure" / "tests"
)
sys.path.insert(0, str(_FIGURE_TESTS_DIR))

from crawl_mocks import FakeSession, make_response  # noqa: E402

from lifeform_domain_figure.cleaning.store import CleaningStore  # noqa: E402
from lifeform_domain_figure.crawl.fetchers import build_default_fetchers  # noqa: E402
from lifeform_domain_figure.crawl.frontier import CrawlFrontier  # noqa: E402
from lifeform_domain_figure.crawl.http_client import BaseHTTPClient  # noqa: E402
from lifeform_domain_figure.crawl.rate_limiter import TokenBucketRateLimiter  # noqa: E402
from lifeform_domain_figure.crawl.records import CrawlRequest, CrawlStatus  # noqa: E402
from lifeform_domain_figure.crawl.robots import RobotsRegistry  # noqa: E402
from lifeform_domain_figure.crawl.scheduler import CrawlScheduler  # noqa: E402
from lifeform_domain_figure.crawl.scope_policy import ScopePolicy, ScopeRole  # noqa: E402
from lifeform_domain_figure.crawl.sink import CrawlSink  # noqa: E402


_TS = "2026-05-10T12:00:00+00:00"


def _build(tmp_path, robots_body: bytes):
    scope = ScopePolicy(
        allowed_hosts=frozenset({"contract-host.example"}),
        user_agent="contract-test/1",
        host_roles={"contract-host.example": frozenset({ScopeRole.CORPUS_FETCH})},
    )

    def handler(url, headers):
        if url.endswith("/robots.txt"):
            return make_response(
                status_code=200, body=robots_body, content_type="text/plain", url=url
            )
        return make_response(
            status_code=200, body=b"document", content_type="text/plain", url=url
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
        fetchers=build_default_fetchers(),
        sleep_fn=lambda _s: None,
    )
    request = CrawlRequest.build(
        url="https://contract-host.example/x",
        fetch_kind="generic",
        enqueued_at_iso=_TS,
    )
    frontier.enqueue(request)
    return scheduler


def test_disallow_all_skips_url(tmp_path: Path) -> None:
    scheduler = _build(tmp_path, b"User-agent: *\nDisallow: /\n")
    results = scheduler.run()
    assert len(results) == 1
    assert results[0].status is CrawlStatus.SKIPPED_ROBOTS
    assert "robots.txt disallow" in results[0].error


def test_allow_all_fetches(tmp_path: Path) -> None:
    scheduler = _build(tmp_path, b"User-agent: *\nAllow: /\n")
    results = scheduler.run()
    assert len(results) == 1
    assert results[0].status is CrawlStatus.SUCCESS
