"""Persistent FIFO frontier with dedup and resumability.

Layout under the configured ``root`` directory::

    root/
      crawl/
        {run_id}/
          queue.jsonl    # pending requests, one CrawlRequest per line
          visited.jsonl  # already-dequeued request_ids, for dedup
          results.jsonl  # CrawlResult records appended by sink

The frontier is intentionally simple: in-memory FIFO + on-disk
JSONL append. Restart-safe via :meth:`resume_from_disk` which
re-reads queue.jsonl and visited.jsonl to rebuild in-memory state.

Dedup discipline:

* A request whose ``request_id`` is already in ``visited`` is rejected
  by :meth:`enqueue` (not re-enqueued).
* A request whose ``request_id`` already appears as a *pending* item
  in queue.jsonl is also rejected (no double-queue).

Result append discipline:

* :meth:`record_result` appends to results.jsonl. Multiple results for
  the same ``request_id`` are allowed (e.g., a re-run that produces a
  fresh fetch); callers reading the log should reduce to the latest.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path

from lifeform_domain_figure.crawl.records import (
    CrawlRequest,
    CrawlResult,
    CrawlStatus,
)


class CrawlFrontier:
    """Filesystem-backed FIFO frontier."""

    def __init__(self, *, root: Path, run_id: str) -> None:
        if not isinstance(root, Path):
            raise TypeError(
                f"CrawlFrontier.root must be a Path; got {type(root).__name__}"
            )
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("CrawlFrontier.run_id must be a non-empty string")
        self._root = root
        self._run_id = run_id
        self._run_dir = root / "crawl" / run_id
        self._queue_path = self._run_dir / "queue.jsonl"
        self._visited_path = self._run_dir / "visited.jsonl"
        self._results_path = self._run_dir / "results.jsonl"
        self._pending: OrderedDict[str, CrawlRequest] = OrderedDict()
        self._visited: set[str] = set()
        self._run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def results_path(self) -> Path:
        return self._results_path

    def _serialise_request(self, request: CrawlRequest) -> str:
        return json.dumps(
            {
                "url": request.url,
                "fetch_kind": request.fetch_kind,
                "request_id": request.request_id,
                "enqueued_at_iso": request.enqueued_at_iso,
                "referrer": request.referrer,
                "expected_content_type": request.expected_content_type,
            },
            ensure_ascii=False,
        )

    def _deserialise_request(self, line: str) -> CrawlRequest:
        payload = json.loads(line)
        return CrawlRequest(
            url=str(payload["url"]),
            fetch_kind=str(payload["fetch_kind"]),
            request_id=str(payload["request_id"]),
            enqueued_at_iso=str(payload["enqueued_at_iso"]),
            referrer=str(payload.get("referrer", "")),
            expected_content_type=str(payload.get("expected_content_type", "")),
        )

    def _serialise_result(self, result: CrawlResult) -> str:
        return json.dumps(
            {
                "request": {
                    "url": result.request.url,
                    "fetch_kind": result.request.fetch_kind,
                    "request_id": result.request.request_id,
                    "enqueued_at_iso": result.request.enqueued_at_iso,
                    "referrer": result.request.referrer,
                    "expected_content_type": result.request.expected_content_type,
                },
                "status": result.status.value,
                "fetched_at_iso": result.fetched_at_iso,
                "raw_sha256": result.raw_sha256,
                "content_type_actual": result.content_type_actual,
                "byte_len": result.byte_len,
                "http_status": result.http_status,
                "etag": result.etag,
                "last_modified": result.last_modified,
                "error": result.error,
            },
            ensure_ascii=False,
        )

    def _deserialise_result(self, line: str) -> CrawlResult:
        payload = json.loads(line)
        request_payload = payload["request"]
        request = CrawlRequest(
            url=str(request_payload["url"]),
            fetch_kind=str(request_payload["fetch_kind"]),
            request_id=str(request_payload["request_id"]),
            enqueued_at_iso=str(request_payload["enqueued_at_iso"]),
            referrer=str(request_payload.get("referrer", "")),
            expected_content_type=str(request_payload.get("expected_content_type", "")),
        )
        return CrawlResult(
            request=request,
            status=CrawlStatus(payload["status"]),
            fetched_at_iso=str(payload["fetched_at_iso"]),
            raw_sha256=str(payload.get("raw_sha256", "")),
            content_type_actual=str(payload.get("content_type_actual", "")),
            byte_len=int(payload.get("byte_len", 0)),
            http_status=int(payload.get("http_status", 0)),
            etag=str(payload.get("etag", "")),
            last_modified=str(payload.get("last_modified", "")),
            error=str(payload.get("error", "")),
        )

    def enqueue(self, request: CrawlRequest) -> bool:
        """Add ``request`` to the queue. Return False if duplicate."""

        rid = request.request_id
        if rid in self._visited or rid in self._pending:
            return False
        self._pending[rid] = request
        with self._queue_path.open("a", encoding="utf-8") as fh:
            fh.write(self._serialise_request(request) + "\n")
        return True

    def next(self) -> CrawlRequest | None:
        """Pop the oldest pending request and mark it visited.

        Returns ``None`` when the queue is empty.
        """

        if not self._pending:
            return None
        rid, request = self._pending.popitem(last=False)
        self._visited.add(rid)
        with self._visited_path.open("a", encoding="utf-8") as fh:
            fh.write(rid + "\n")
        return request

    def record_result(self, result: CrawlResult) -> None:
        with self._results_path.open("a", encoding="utf-8") as fh:
            fh.write(self._serialise_result(result) + "\n")

    def pending_count(self) -> int:
        return len(self._pending)

    def visited_count(self) -> int:
        return len(self._visited)

    def iter_results(self) -> Iterator[CrawlResult]:
        if not self._results_path.exists():
            return
        with self._results_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                yield self._deserialise_result(stripped)

    @classmethod
    def resume_from_disk(cls, *, root: Path, run_id: str) -> "CrawlFrontier":
        """Re-construct a frontier from on-disk state.

        Reads ``visited.jsonl`` then ``queue.jsonl``; pending = items
        in queue.jsonl whose ``request_id`` is not in visited.
        """

        frontier = cls(root=root, run_id=run_id)
        if frontier._visited_path.exists():
            with frontier._visited_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    rid = line.strip()
                    if rid:
                        frontier._visited.add(rid)
        if frontier._queue_path.exists():
            with frontier._queue_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    request = frontier._deserialise_request(stripped)
                    if request.request_id in frontier._visited:
                        continue
                    if request.request_id in frontier._pending:
                        continue
                    frontier._pending[request.request_id] = request
        return frontier


__all__ = ["CrawlFrontier"]
