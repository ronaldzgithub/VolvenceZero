"""Cross-cutting contract: crawler SUCCESS path writes to L1 store.

Establishes the content-addressable anchor chain end-to-end:

* Crawler scheduler runs against a mock HTTP fixture.
* On SUCCESS, the L1 :class:`CleaningStore` MUST contain a raw entry
  whose sha matches :class:`CrawlResult.raw_sha256` and whose bytes
  match the response body.

This is the test that fails if anyone refactors :class:`CrawlSink`
to skip the L1 ``put_raw`` call (regressing the L0 -> L1 link
debt #28 explicitly requires).
"""

from __future__ import annotations

import hashlib
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


def test_crawler_success_writes_anchor_into_l1_store(tmp_path: Path) -> None:
    document_body = b"<html>contract test body</html>"
    expected_sha = hashlib.sha256(document_body).hexdigest()

    scope = ScopePolicy(
        allowed_hosts=frozenset({"contract-host.example"}),
        user_agent="contract-test/1",
        host_roles={"contract-host.example": frozenset({ScopeRole.CORPUS_FETCH})},
    )

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
            body=document_body,
            content_type="text/html",
            url=url,
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
    results = scheduler.run()
    assert len(results) == 1
    assert results[0].status is CrawlStatus.SUCCESS
    assert results[0].raw_sha256 == expected_sha
    assert expected_sha in tuple(cleaning_store.list_raw())
    fetched_bytes, sidecar = cleaning_store.get_raw(expected_sha)
    assert fetched_bytes == document_body
    assert sidecar.source_url == "https://contract-host.example/x"
