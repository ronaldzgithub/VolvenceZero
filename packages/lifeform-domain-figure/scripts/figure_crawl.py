"""Figure-vertical L0 corpus crawler CLI (debt #28 + debt #19).

Five subcommands:

* ``enqueue`` — single-URL enqueue.
* ``enqueue-batch`` — bulk enqueue from a JSONL of
  ``{"url": ..., "fetch_kind": ..., "expected_content_type": ..., "referrer": ...}``.
* ``run`` — drive the scheduler until the frontier empties or a
  page cap is reached. Writes fetched bytes into the L1
  ``CleaningStore`` rooted at ``--cleaning-root`` so the L0/L1/L2
  anchor chain is established immediately.
* ``status`` — print pending / visited counts.
* ``list-results`` — print every CrawlResult (optionally filtered
  by status).

This script is curator-facing; runtime systems never call it. It
strictly does not write to L2 ``VerificationLedger`` or to any
kernel owner; the L1 cleaning + L2 verification subsequent steps
are run with their own CLIs (``figure_clean.py`` / ``figure_verify.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from lifeform_domain_figure.cleaning.store import CleaningStore
from lifeform_domain_figure.crawl import (
    BaseHTTPClient,
    CrawlFrontier,
    CrawlRequest,
    CrawlScheduler,
    CrawlSink,
    CrawlStatus,
    DEFAULT_USER_AGENT,
    RobotsRegistry,
    ScopePolicy,
    TokenBucketRateLimiter,
    VALID_FETCH_KINDS,
    default_scope_policy,
    request_id_for,
)


def _build_scope(args: argparse.Namespace) -> ScopePolicy:
    return default_scope_policy(args.user_agent or DEFAULT_USER_AGENT)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cmd_enqueue(args: argparse.Namespace) -> int:
    if args.fetch_kind not in VALID_FETCH_KINDS:
        print(
            f"enqueue: --fetch-kind must be one of {sorted(VALID_FETCH_KINDS)!r}; "
            f"got {args.fetch_kind!r}",
            file=sys.stderr,
        )
        return 2
    frontier = CrawlFrontier.resume_from_disk(root=Path(args.root), run_id=args.run_id)
    request = CrawlRequest(
        url=args.url,
        fetch_kind=args.fetch_kind,
        request_id=request_id_for(args.fetch_kind, args.url),
        enqueued_at_iso=_now_iso(),
        referrer=args.referrer,
        expected_content_type=args.expected_content_type,
    )
    enqueued = frontier.enqueue(request)
    if enqueued:
        print(
            f"enqueue: queued request_id={request.request_id[:12]}... "
            f"url={request.url!r} fetch_kind={request.fetch_kind!r}"
        )
    else:
        print(
            f"enqueue: duplicate (already pending or visited) request_id="
            f"{request.request_id[:12]}... url={request.url!r}"
        )
    return 0


def _cmd_enqueue_batch(args: argparse.Namespace) -> int:
    requests_path = Path(args.requests_file)
    if not requests_path.exists():
        print(f"enqueue-batch: file not found: {requests_path}", file=sys.stderr)
        return 2
    frontier = CrawlFrontier.resume_from_disk(root=Path(args.root), run_id=args.run_id)
    queued = 0
    skipped = 0
    with requests_path.open("r", encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            try:
                fetch_kind = str(payload["fetch_kind"])
                url = str(payload["url"])
            except KeyError as exc:
                print(
                    f"enqueue-batch: line {line_no}: missing required field {exc}",
                    file=sys.stderr,
                )
                return 2
            request = CrawlRequest(
                url=url,
                fetch_kind=fetch_kind,
                request_id=request_id_for(fetch_kind, url),
                enqueued_at_iso=_now_iso(),
                referrer=str(payload.get("referrer", "")),
                expected_content_type=str(payload.get("expected_content_type", "")),
            )
            if frontier.enqueue(request):
                queued += 1
            else:
                skipped += 1
    print(f"enqueue-batch: queued {queued} new request(s); skipped {skipped} duplicate(s)")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    scope = _build_scope(args)
    http_client = BaseHTTPClient(scope=scope)
    robots = RobotsRegistry(http_client=http_client)
    rate_limiter = TokenBucketRateLimiter(rate_per_second=args.rate_rps, burst=args.burst)
    frontier = CrawlFrontier.resume_from_disk(root=Path(args.root), run_id=args.run_id)
    cleaning_store = CleaningStore(Path(args.cleaning_root))
    sink = CrawlSink(cleaning_store=cleaning_store, frontier=frontier)
    scheduler = CrawlScheduler(
        scope=scope,
        http_client=http_client,
        robots=robots,
        rate_limiter=rate_limiter,
        frontier=frontier,
        sink=sink,
    )
    results = scheduler.run(max_pages=args.max_pages)
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status.value] = counts.get(r.status.value, 0) + 1
    print(f"run: processed {len(results)} request(s); status_counts={counts}")
    for r in results:
        if r.status is CrawlStatus.SUCCESS:
            print(
                f"  SUCCESS request_id={r.request.request_id[:12]}... "
                f"raw_sha256={r.raw_sha256[:12]}... "
                f"content_type={r.content_type_actual!r} byte_len={r.byte_len}"
            )
        else:
            print(
                f"  {r.status.value.upper()} request_id={r.request.request_id[:12]}... "
                f"url={r.request.url!r} error={r.error!r}"
            )
    http_client.close()
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    frontier = CrawlFrontier.resume_from_disk(root=Path(args.root), run_id=args.run_id)
    print(
        f"status: run_id={args.run_id!r} pending={frontier.pending_count()} "
        f"visited={frontier.visited_count()} run_dir={frontier.run_dir}"
    )
    return 0


def _cmd_list_results(args: argparse.Namespace) -> int:
    frontier = CrawlFrontier.resume_from_disk(root=Path(args.root), run_id=args.run_id)
    filter_status = (
        CrawlStatus(args.status_filter) if args.status_filter else None
    )
    count = 0
    for r in frontier.iter_results():
        if filter_status is not None and r.status is not filter_status:
            continue
        count += 1
        print(
            f"{r.fetched_at_iso}\t{r.status.value}\t{r.request.fetch_kind}\t"
            f"{r.request.url}\t{r.raw_sha256 or '-'}\t{r.error or '-'}"
        )
    print(f"list-results: printed {count} record(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Figure-vertical L0 corpus crawler CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    enqueue = sub.add_parser("enqueue", help="Enqueue a single URL")
    enqueue.add_argument("--root", required=True, help="Crawl frontier root")
    enqueue.add_argument("--run-id", required=True, help="Crawl run identifier")
    enqueue.add_argument("--url", required=True)
    enqueue.add_argument("--fetch-kind", required=True)
    enqueue.add_argument("--referrer", default="")
    enqueue.add_argument("--expected-content-type", default="")
    enqueue.set_defaults(func=_cmd_enqueue)

    batch = sub.add_parser("enqueue-batch", help="Enqueue a batch from JSONL")
    batch.add_argument("--root", required=True)
    batch.add_argument("--run-id", required=True)
    batch.add_argument("--requests-file", required=True)
    batch.set_defaults(func=_cmd_enqueue_batch)

    run_cmd = sub.add_parser("run", help="Drive the scheduler until empty / cap")
    run_cmd.add_argument("--root", required=True, help="Crawl frontier root")
    run_cmd.add_argument("--run-id", required=True)
    run_cmd.add_argument(
        "--cleaning-root",
        required=True,
        help="L1 CleaningStore root (fetched bytes are written here)",
    )
    run_cmd.add_argument(
        "--user-agent",
        default="",
        help=f"HTTP User-Agent (default {DEFAULT_USER_AGENT!r})",
    )
    run_cmd.add_argument(
        "--rate-rps",
        type=float,
        default=0.5,
        help="Per-host requests per second (default 0.5)",
    )
    run_cmd.add_argument(
        "--burst",
        type=int,
        default=5,
        help="Token bucket burst capacity (default 5)",
    )
    run_cmd.add_argument("--max-pages", type=int, default=None)
    run_cmd.set_defaults(func=_cmd_run)

    status = sub.add_parser("status", help="Print pending / visited counts")
    status.add_argument("--root", required=True)
    status.add_argument("--run-id", required=True)
    status.set_defaults(func=_cmd_status)

    list_cmd = sub.add_parser("list-results", help="Print CrawlResults from results.jsonl")
    list_cmd.add_argument("--root", required=True)
    list_cmd.add_argument("--run-id", required=True)
    list_cmd.add_argument(
        "--status-filter",
        default=None,
        choices=[s.value for s in CrawlStatus],
        help="Filter by status",
    )
    list_cmd.set_defaults(func=_cmd_list_results)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
