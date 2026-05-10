"""Crawl request / result schema for the L0 corpus crawler.

The crawler is figure-vertical-internal corpus acquisition. It sits
**before** the L1 cleaning pipeline (its output bytes feed
:class:`CleaningStore.put_raw`) and is the only figure-vertical layer
allowed to issue HTTP requests.

Two records:

* :class:`CrawlRequest` — the curator-supplied work item. Carries the
  URL, fetch_kind (which archive-aware fetcher should handle it),
  request_id (sha256 of fetch_kind + url for dedup), and an optional
  expected content-type the L1 parser will receive after the bytes
  land.
* :class:`CrawlResult` — what the scheduler emits per request. Carries
  a closed-vocabulary :class:`CrawlStatus` plus, for SUCCESS, the
  ``raw_sha256`` link that ties the crawl record back to the L1
  ``CleaningStore`` entry.

The ``raw_sha256`` field on :class:`CrawlResult` and the
``byte_sha256`` on :class:`SourceProvenance` and the
``raw_sha256`` on L1 :class:`RawDocument` are the **same hash for
the same byte stream** — the L0 / L1 / L2 chain is content-addressable
end-to-end through this single anchor key.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum


class CrawlStatus(str, Enum):
    """Closed vocabulary of crawl outcomes."""

    SUCCESS = "success"
    FETCHED_NOT_MODIFIED = "not_modified"
    SKIPPED_ROBOTS = "skipped_robots"
    SKIPPED_SCOPE = "skipped_scope"
    SKIPPED_RATE = "skipped_rate"
    FAILED_HTTP = "failed_http"
    FAILED_PARSER_PRECHECK = "failed_parser_precheck"


VALID_FETCH_KINDS: frozenset[str] = frozenset(
    {"generic", "cpae", "wikisource", "gutenberg", "internet_archive"}
)
"""The closed set of fetch_kind labels the L0 dispatcher accepts."""


def request_id_for(fetch_kind: str, url: str) -> str:
    """Deterministic dedup key for a (fetch_kind, url) pair.

    Sha256 keeps the key length stable regardless of URL length and
    avoids any quirks with URL canonicalisation; two requests with
    different fetch_kind but same URL are intentionally distinct
    items (the dispatcher may pick a different fetcher).
    """

    payload = f"{fetch_kind}\n{url}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CrawlRequest:
    """A curator-supplied unit of work for the crawler."""

    url: str
    fetch_kind: str
    request_id: str
    enqueued_at_iso: str
    referrer: str = ""
    expected_content_type: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError("CrawlRequest.url must be a non-empty string")
        if self.fetch_kind not in VALID_FETCH_KINDS:
            raise ValueError(
                f"CrawlRequest.fetch_kind must be one of {sorted(VALID_FETCH_KINDS)!r}; "
                f"got {self.fetch_kind!r}"
            )
        if (
            not isinstance(self.request_id, str)
            or len(self.request_id) != 64
        ):
            raise ValueError(
                f"CrawlRequest.request_id must be a 64-char hex sha256; "
                f"got {self.request_id!r}"
            )
        expected = request_id_for(self.fetch_kind, self.url)
        if self.request_id != expected:
            raise ValueError(
                f"CrawlRequest.request_id mismatch: expected {expected!r} for "
                f"fetch_kind={self.fetch_kind!r} url={self.url!r}; "
                f"got {self.request_id!r}"
            )
        if not isinstance(self.enqueued_at_iso, str) or not self.enqueued_at_iso.strip():
            raise ValueError(
                "CrawlRequest.enqueued_at_iso must be a non-empty ISO-8601 string"
            )

    @classmethod
    def build(
        cls,
        *,
        url: str,
        fetch_kind: str,
        enqueued_at_iso: str,
        referrer: str = "",
        expected_content_type: str = "",
    ) -> "CrawlRequest":
        """Convenience constructor that derives ``request_id`` automatically."""

        return cls(
            url=url,
            fetch_kind=fetch_kind,
            request_id=request_id_for(fetch_kind, url),
            enqueued_at_iso=enqueued_at_iso,
            referrer=referrer,
            expected_content_type=expected_content_type,
        )


@dataclass(frozen=True)
class CrawlResult:
    """The scheduler's outcome for one :class:`CrawlRequest`.

    For ``SUCCESS``, ``raw_sha256`` is the L1 ``CleaningStore`` anchor
    key (== ``SourceProvenance.byte_sha256`` == ``RawDocument.raw_sha256``);
    for any other status it is the empty string. ``content_type_actual``
    captures what the response body actually carried (post any archive-
    specific adjustment by the fetcher); for skipped / failed requests it
    is the empty string.
    """

    request: CrawlRequest
    status: CrawlStatus
    fetched_at_iso: str
    raw_sha256: str = ""
    content_type_actual: str = ""
    byte_len: int = 0
    http_status: int = 0
    etag: str = ""
    last_modified: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.fetched_at_iso, str) or not self.fetched_at_iso.strip():
            raise ValueError(
                "CrawlResult.fetched_at_iso must be a non-empty ISO-8601 string"
            )
        if self.status is CrawlStatus.SUCCESS:
            if not self.raw_sha256 or len(self.raw_sha256) != 64:
                raise ValueError(
                    "CrawlResult.raw_sha256 must be a 64-char hex sha256 when "
                    f"status=SUCCESS (request_id={self.request.request_id!r})"
                )
            if not self.content_type_actual.strip():
                raise ValueError(
                    "CrawlResult.content_type_actual must be non-empty when status=SUCCESS"
                )
            if self.byte_len <= 0:
                raise ValueError(
                    "CrawlResult.byte_len must be > 0 when status=SUCCESS"
                )
        else:
            if self.raw_sha256:
                raise ValueError(
                    f"CrawlResult.raw_sha256 must be empty when status="
                    f"{self.status.value!r}; got {self.raw_sha256!r}"
                )
        if self.byte_len < 0:
            raise ValueError("CrawlResult.byte_len must be >= 0")


__all__ = [
    "CrawlRequest",
    "CrawlResult",
    "CrawlStatus",
    "VALID_FETCH_KINDS",
    "request_id_for",
]
