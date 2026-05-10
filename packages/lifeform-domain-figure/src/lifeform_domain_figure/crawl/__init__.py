"""Figure-vertical L0 corpus crawler.

This subpackage implements layer L0 of `docs/known-debts.md` debt #28
(see `docs/specs/figure-corpus-crawl.md` for the spec). It is the
ONLY figure-vertical layer permitted to issue HTTP requests; the L1
cleaning and L2 verification subpackages have AST-enforced
no-HTTP-imports invariants.

End-to-end pipeline::

    seed URL + fetch_kind
        -> CrawlFrontier.enqueue
        -> CrawlScheduler.run() loop:
            scope -> robots -> rate-limit -> dispatch -> fetch -> sink
        -> L1 CleaningStore.put_raw  (raw_sha256 anchor established)
        -> downstream: figure_clean -> figure_verify -> bundle gate

Closes debt #19 V2 archive fetcher: the four archive fetchers
(``CPAEFetcher`` / ``WikisourceFetcher`` / ``GutenbergFetcher`` /
``InternetArchiveFetcher``) are real implementations that pick the
right archive-specific URL pattern + content-type. They are also
exposed to the legacy
:func:`lifeform_domain_figure.corpus.archives.live_archive_fetcher`
factory which returns an ``ArchiveFetcher`` Protocol implementation
backed by the crawler stack (preserving the V1
``offline_archive_fetcher`` for tests / curator manual mode).

Public surface (see module docstrings for details):

* Schema: :class:`CrawlStatus`, :class:`CrawlRequest`,
  :class:`CrawlResult`, :data:`VALID_FETCH_KINDS`.
* Scope: :class:`ScopePolicy`, :func:`default_scope_policy`,
  :data:`DEFAULT_HOSTS`.
* HTTP: :class:`BaseHTTPClient`, :class:`HTTPResponse`,
  :data:`NOT_MODIFIED`, :class:`ScopeRejection`, :class:`FetchError`,
  :class:`BodyTooLarge`.
* Robots: :class:`RobotsRegistry`.
* Rate limit: :class:`TokenBucketRateLimiter`.
* Frontier: :class:`CrawlFrontier`.
* Sink: :class:`CrawlSink`.
* Scheduler: :class:`CrawlScheduler`.
* Fetchers: :func:`build_default_fetchers`, :func:`dispatch_for`,
  five concrete fetchers.
"""

from __future__ import annotations

from lifeform_domain_figure.crawl.fetchers import (
    CPAEFetcher,
    FetcherProtocol,
    GenericHTTPFetcher,
    GutenbergFetcher,
    InternetArchiveFetcher,
    WikisourceFetcher,
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
    VALID_FETCH_KINDS,
    request_id_for,
)
from lifeform_domain_figure.crawl.robots import RobotsRegistry
from lifeform_domain_figure.crawl.scheduler import CrawlScheduler
from lifeform_domain_figure.crawl.scope_policy import (
    DEFAULT_CORPUS_HOSTS,
    DEFAULT_HOSTS,
    DEFAULT_METADATA_HOSTS,
    DEFAULT_USER_AGENT,
    ScopePolicy,
    ScopeRole,
    default_combined_scope_policy,
    default_metadata_scope_policy,
    default_scope_policy,
)
from lifeform_domain_figure.crawl.sink import CrawlSink


__all__ = [
    "BaseHTTPClient",
    "BodyTooLarge",
    "CPAEFetcher",
    "CrawlFrontier",
    "CrawlRequest",
    "CrawlResult",
    "CrawlScheduler",
    "CrawlSink",
    "CrawlStatus",
    "DEFAULT_CORPUS_HOSTS",
    "DEFAULT_HOSTS",
    "DEFAULT_METADATA_HOSTS",
    "DEFAULT_USER_AGENT",
    "FetchError",
    "FetcherProtocol",
    "GenericHTTPFetcher",
    "GutenbergFetcher",
    "HTTPResponse",
    "InternetArchiveFetcher",
    "NOT_MODIFIED",
    "RobotsRegistry",
    "ScopePolicy",
    "ScopeRejection",
    "ScopeRole",
    "TokenBucketRateLimiter",
    "VALID_FETCH_KINDS",
    "WikisourceFetcher",
    "build_default_fetchers",
    "default_combined_scope_policy",
    "default_metadata_scope_policy",
    "default_scope_policy",
    "dispatch_for",
    "request_id_for",
]
