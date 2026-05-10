"""Crawl sink — bridges fetched bytes into L1 CleaningStore.

The sink is intentionally narrow: given a successful HTTP fetch, it
hands the bytes to :class:`CleaningStore.put_raw` (which is
content-addressable and idempotent) and constructs a
:class:`CrawlResult` carrying the resulting ``raw_sha256``. The
sink does NOT trigger L1 parsing or cleaning; that is a separate
curator step run via ``scripts/figure_clean.py`` (separation of
concerns + L0 stays HTTP-focused).

Skipped / failed CrawlResult records are produced by the scheduler
directly; the sink only handles the SUCCESS path.
"""

from __future__ import annotations

from datetime import datetime, timezone

from lifeform_domain_figure.cleaning.store import CleaningStore
from lifeform_domain_figure.crawl.frontier import CrawlFrontier
from lifeform_domain_figure.crawl.http_client import HTTPResponse
from lifeform_domain_figure.crawl.records import (
    CrawlRequest,
    CrawlResult,
    CrawlStatus,
)


class CrawlSink:
    """Write fetched bytes into L1 CleaningStore + record SUCCESS result."""

    def __init__(
        self,
        *,
        cleaning_store: CleaningStore,
        frontier: CrawlFrontier,
    ) -> None:
        if not isinstance(cleaning_store, CleaningStore):
            raise TypeError(
                f"CrawlSink.cleaning_store must be a CleaningStore; got "
                f"{type(cleaning_store).__name__}"
            )
        if not isinstance(frontier, CrawlFrontier):
            raise TypeError(
                f"CrawlSink.frontier must be a CrawlFrontier; got "
                f"{type(frontier).__name__}"
            )
        self._cleaning_store = cleaning_store
        self._frontier = frontier

    def consume_success(
        self,
        request: CrawlRequest,
        response: HTTPResponse,
        content_type: str,
    ) -> CrawlResult:
        """Persist ``response`` to L1 store and append a SUCCESS result."""

        if not content_type.strip():
            raise ValueError(
                "CrawlSink.consume_success: content_type must be non-empty"
            )
        raw_sha256 = self._cleaning_store.put_raw(
            response.body,
            source_url=request.url,
            content_type=content_type,
        )
        result = CrawlResult(
            request=request,
            status=CrawlStatus.SUCCESS,
            fetched_at_iso=datetime.now(timezone.utc).isoformat(),
            raw_sha256=raw_sha256,
            content_type_actual=content_type,
            byte_len=len(response.body),
            http_status=response.http_status,
            etag=response.etag,
            last_modified=response.last_modified,
        )
        self._frontier.record_result(result)
        return result

    def record_terminal_result(self, result: CrawlResult) -> None:
        """Append a non-SUCCESS terminal result to the frontier results log."""

        if result.status is CrawlStatus.SUCCESS:
            raise ValueError(
                "CrawlSink.record_terminal_result: result.status must NOT be "
                "SUCCESS (use consume_success for SUCCESS path)"
            )
        self._frontier.record_result(result)


__all__ = ["CrawlSink"]
