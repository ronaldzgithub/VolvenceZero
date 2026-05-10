"""End-to-end crawler orchestrator.

For each request popped from the frontier:

1. Scope check — :meth:`ScopePolicy.is_in_scope`. Failure ->
   SKIPPED_SCOPE.
2. Robots check — :meth:`RobotsRegistry.is_allowed`. Failure ->
   SKIPPED_ROBOTS.
3. Per-host page budget — host visit count
   < ``ScopePolicy.max_pages_per_host``. Over budget -> SKIPPED_SCOPE
   with reason ``"max_pages_per_host"``.
4. Rate limit acquire — if the bucket is dry, sleep
   ``sleep_hint_seconds`` and try once more; if still dry ->
   SKIPPED_RATE.
5. Dispatch to fetcher — :func:`dispatch_for`.
6. Fetcher.fetch — handles SSRF gates inside :class:`BaseHTTPClient`;
   any :class:`ScopeRejection` is mapped to SKIPPED_SCOPE; any
   :class:`FetchError` / :class:`BodyTooLarge` to FAILED_HTTP.
7. NOT_MODIFIED -> FETCHED_NOT_MODIFIED.
8. Otherwise normalise content_type via
   :meth:`Fetcher.derive_content_type` and hand to :class:`CrawlSink`.

The scheduler is single-threaded; it processes the FIFO frontier
until empty (``run()`` returns) or until the run-level page cap is
hit (``max_pages`` argument).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from urllib.parse import urlparse

from lifeform_domain_figure.crawl.fetchers import (
    FetcherProtocol,
    build_default_fetchers,
    dispatch_for,
)
from lifeform_domain_figure.crawl.frontier import CrawlFrontier
from lifeform_domain_figure.crawl.http_client import (
    BaseHTTPClient,
    BodyTooLarge,
    FetchError,
    HTTPResponse,
    NOT_MODIFIED,
    ScopeRejection,
)
from lifeform_domain_figure.crawl.rate_limiter import TokenBucketRateLimiter
from lifeform_domain_figure.crawl.records import (
    CrawlRequest,
    CrawlResult,
    CrawlStatus,
)
from lifeform_domain_figure.crawl.robots import RobotsRegistry
from lifeform_domain_figure.crawl.scope_policy import ScopePolicy
from lifeform_domain_figure.crawl.sink import CrawlSink


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _terminal_result(
    request: CrawlRequest,
    status: CrawlStatus,
    *,
    error: str = "",
    http_status: int = 0,
) -> CrawlResult:
    return CrawlResult(
        request=request,
        status=status,
        fetched_at_iso=_now_iso(),
        error=error,
        http_status=http_status,
    )


class CrawlScheduler:
    """Orchestrate scope -> robots -> rate -> fetch -> sink end-to-end."""

    def __init__(
        self,
        *,
        scope: ScopePolicy,
        http_client: BaseHTTPClient,
        robots: RobotsRegistry,
        rate_limiter: TokenBucketRateLimiter,
        frontier: CrawlFrontier,
        sink: CrawlSink,
        fetchers: dict[str, FetcherProtocol] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._scope = scope
        self._http_client = http_client
        self._robots = robots
        self._rate_limiter = rate_limiter
        self._frontier = frontier
        self._sink = sink
        self._fetchers = fetchers if fetchers is not None else build_default_fetchers()
        import time

        self._sleep = sleep_fn or time.sleep
        self._host_counts: Counter[str] = Counter()

    def _host_for(self, url: str) -> str:
        return (urlparse(url).hostname or "").lower()

    def _process_one(self, request: CrawlRequest) -> CrawlResult:
        url = request.url
        host = self._host_for(url)
        if not self._scope.is_in_scope(url):
            reason = self._scope.reason_out_of_scope(url) or "out of scope"
            result = _terminal_result(request, CrawlStatus.SKIPPED_SCOPE, error=reason)
            self._sink.record_terminal_result(result)
            return result
        allowed, reason = self._robots.is_allowed(url)
        if not allowed:
            result = _terminal_result(request, CrawlStatus.SKIPPED_ROBOTS, error=reason)
            self._sink.record_terminal_result(result)
            return result
        if self._host_counts[host] >= self._scope.max_pages_per_host:
            result = _terminal_result(
                request,
                CrawlStatus.SKIPPED_SCOPE,
                error=(
                    f"max_pages_per_host reached for host={host!r} "
                    f"(cap={self._scope.max_pages_per_host})"
                ),
            )
            self._sink.record_terminal_result(result)
            return result
        ok, sleep_hint = self._rate_limiter.acquire(host)
        if not ok:
            self._sleep(sleep_hint)
            ok, sleep_hint = self._rate_limiter.acquire(host)
            if not ok:
                result = _terminal_result(
                    request,
                    CrawlStatus.SKIPPED_RATE,
                    error=(
                        f"rate limiter still empty after sleep_hint={sleep_hint:.3f}s "
                        f"for host={host!r}"
                    ),
                )
                self._sink.record_terminal_result(result)
                return result
        try:
            fetcher = dispatch_for(request, self._fetchers)
        except KeyError as exc:
            result = _terminal_result(
                request,
                CrawlStatus.FAILED_PARSER_PRECHECK,
                error=str(exc),
            )
            self._sink.record_terminal_result(result)
            return result
        try:
            response = fetcher.fetch(request, self._http_client)
        except ScopeRejection as exc:
            result = _terminal_result(
                request, CrawlStatus.SKIPPED_SCOPE, error=str(exc)
            )
            self._sink.record_terminal_result(result)
            return result
        except BodyTooLarge as exc:
            result = _terminal_result(
                request, CrawlStatus.FAILED_HTTP, error=str(exc)
            )
            self._sink.record_terminal_result(result)
            return result
        except FetchError as exc:
            result = _terminal_result(
                request, CrawlStatus.FAILED_HTTP, error=str(exc)
            )
            self._sink.record_terminal_result(result)
            return result
        if response is NOT_MODIFIED:
            result = _terminal_result(
                request,
                CrawlStatus.FETCHED_NOT_MODIFIED,
                http_status=304,
            )
            self._sink.record_terminal_result(result)
            self._host_counts[host] += 1
            return result
        assert isinstance(response, HTTPResponse)
        try:
            content_type = fetcher.derive_content_type(request, response)
        except Exception as exc:
            result = _terminal_result(
                request,
                CrawlStatus.FAILED_PARSER_PRECHECK,
                error=f"derive_content_type failed: {exc}",
                http_status=response.http_status,
            )
            self._sink.record_terminal_result(result)
            return result
        result = self._sink.consume_success(request, response, content_type)
        self._host_counts[host] += 1
        return result

    def run(self, *, max_pages: int | None = None) -> tuple[CrawlResult, ...]:
        """Process the frontier until empty or until ``max_pages`` reached.

        Returns the tuple of CrawlResults produced this run (in
        chronological order). Returns ``()`` when the frontier was
        already empty.
        """

        results: list[CrawlResult] = []
        processed = 0
        while True:
            if max_pages is not None and processed >= max_pages:
                break
            request = self._frontier.next()
            if request is None:
                break
            result = self._process_one(request)
            results.append(result)
            processed += 1
        return tuple(results)


__all__ = ["CrawlScheduler"]
