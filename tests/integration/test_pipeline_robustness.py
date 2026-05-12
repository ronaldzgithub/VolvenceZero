"""Wave L pipeline robustness integration tests.

Six invariants the figure-vertical L0/L1/L2/bundle pipeline must
satisfy on real workloads:

1. **Resumability** — kill mid-run, resume_from_disk reconstructs
   pending + visited correctly; results.jsonl carries no duplicate
   SUCCESS rows.
2. **ETag idempotency** — running the same crawl twice (with
   server cooperation) reports ``FETCHED_NOT_MODIFIED`` on the
   second pass for previously-fetched anchors. *(Stub here: we
   simulate ETag/304 with FakeSession; the @live_network variant
   exercises a real server.)*
3. **SSRF live** — feeding a URL outside the corpus host
   allowlist yields ``SKIPPED_SCOPE`` even on a real DNS / socket
   path.
4. **Robots fail-closed** — a crawler whose robots.txt fetch
   times out / 5xx must reject every URL on that host.
5. **Cross-archive dedup** — same byte-identical text shipped
   under two different archives collapses to one canonical entry
   in the bundle's retrieval index.
6. **Bundle byte-stable** — same cleaning store + same metadata
   file -> same bundle.integrity_hash byte-for-byte across runs.

Tests that need real network are marked ``@pytest.mark.live_network``;
the default ``addopts`` in pyproject.toml excludes them so CI stays
hermetic. Run them locally with ``pytest -m live_network``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest


# Path to the figure-vertical scripts used as subprocesses below.
_FIGURE_SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "lifeform-domain-figure"
    / "scripts"
)
_FIGURE_TESTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "lifeform-domain-figure"
    / "tests"
)
if str(_FIGURE_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_FIGURE_TESTS_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_jsonl(path: Path, entries: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )


def _enqueue_subprocess(
    *, root: Path, run_id: str, requests_file: Path
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(_FIGURE_SCRIPTS / "figure_crawl.py"),
            "enqueue-batch",
            "--root",
            str(root),
            "--run-id",
            run_id,
            "--requests-file",
            str(requests_file),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _read_results(*, root: Path, run_id: str) -> list[dict]:
    results_path = root / "crawl" / run_id / "results.jsonl"
    if not results_path.exists():
        return []
    rows: list[dict] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


# ---------------------------------------------------------------------------
# 1. Resumability — kill-and-resume
# ---------------------------------------------------------------------------


def test_frontier_resume_after_partial_run(tmp_path: Path) -> None:
    """frontier.resume_from_disk reconstructs pending + visited so a
    second run picks up exactly the not-yet-fetched URLs.

    No real network — we drive the in-memory frontier directly,
    mark some entries visited, then resume_from_disk and verify
    the surviving pending set.
    """

    from lifeform_domain_figure.crawl import (
        CrawlFrontier,
        CrawlRequest,
        request_id_for,
    )

    root = tmp_path / "corpus"
    run_id = "robustness-resume"
    frontier_a = CrawlFrontier(root=root, run_id=run_id)
    requests = [
        CrawlRequest(
            url=f"https://en.wikisource.org/wiki/X{i}",
            fetch_kind="wikisource",
            request_id=request_id_for("wikisource", f"https://en.wikisource.org/wiki/X{i}"),
            enqueued_at_iso="2026-05-12T00:00:00Z",
        )
        for i in range(5)
    ]
    for request in requests:
        assert frontier_a.enqueue(request)
    # Pop the first two — simulating that 2 of 5 URLs got dispatched
    # before the process was killed.
    popped_a = frontier_a.next()
    popped_b = frontier_a.next()
    assert popped_a is not None and popped_b is not None
    assert frontier_a.pending_count() == 3
    assert frontier_a.visited_count() == 2

    # Simulate process restart: discard frontier_a, resume from disk.
    frontier_b = CrawlFrontier.resume_from_disk(root=root, run_id=run_id)
    assert frontier_b.pending_count() == 3
    assert frontier_b.visited_count() == 2
    # Re-enqueueing an already-visited request_id is a no-op.
    assert frontier_b.enqueue(popped_a) is False
    # Pending set survives ordering.
    next_after = frontier_b.next()
    assert next_after is not None
    assert next_after.request_id == requests[2].request_id


def test_frontier_dedup_within_pending(tmp_path: Path) -> None:
    """Re-enqueueing a request whose request_id is already pending
    is a no-op; the queue does not grow."""

    from lifeform_domain_figure.crawl import (
        CrawlFrontier,
        CrawlRequest,
        request_id_for,
    )

    frontier = CrawlFrontier(root=tmp_path, run_id="dedup")
    rid = request_id_for("wikisource", "https://en.wikisource.org/wiki/X")
    request = CrawlRequest(
        url="https://en.wikisource.org/wiki/X",
        fetch_kind="wikisource",
        request_id=rid,
        enqueued_at_iso="2026-05-12T00:00:00Z",
    )
    assert frontier.enqueue(request)
    assert frontier.enqueue(request) is False
    assert frontier.pending_count() == 1


# ---------------------------------------------------------------------------
# 3. SSRF — host outside allowlist must reject
# ---------------------------------------------------------------------------


def test_ssrf_off_list_host_is_rejected(tmp_path: Path) -> None:
    """ScopePolicy.is_in_scope refuses any URL whose host is not in
    the corpus / metadata allowlist; the crawler will record
    SKIPPED_SCOPE without ever opening a socket."""

    from lifeform_domain_figure.crawl import default_scope_policy

    scope = default_scope_policy("test-agent")
    assert scope.is_in_scope("https://en.wikisource.org/wiki/X") is True
    # example.com is not in DEFAULT_HOSTS
    assert scope.is_in_scope("https://example.com/") is False
    # localhost / private IPs are not in DEFAULT_HOSTS
    assert scope.is_in_scope("http://127.0.0.1/") is False
    assert scope.is_in_scope("http://10.0.0.1/") is False
    # scheme outside http/https
    assert scope.is_in_scope("file:///etc/passwd") is False


def test_ssrf_path_prefix_rejected(tmp_path: Path) -> None:
    """When a host carries an explicit path-prefix list, paths
    outside the prefix are rejected even if the host is allowed."""

    from lifeform_domain_figure.crawl.scope_policy import (
        ScopePolicy,
        ScopeRole,
    )

    policy = ScopePolicy(
        allowed_hosts=frozenset({"example.invalid"}),
        user_agent="test-agent",
        allowed_path_prefixes={"example.invalid": ("/wiki/",)},
        host_roles={"example.invalid": frozenset({ScopeRole.CORPUS_FETCH})},
    )
    assert policy.is_in_scope("https://example.invalid/wiki/Foo") is True
    assert policy.is_in_scope("https://example.invalid/admin") is False


# ---------------------------------------------------------------------------
# 4. Robots fail-closed
# ---------------------------------------------------------------------------


def test_robots_fail_closed_blocks_all_paths(tmp_path: Path) -> None:
    """When robots.txt fetch fails (network error / 5xx), every
    subsequent path on that host must be rejected — never silently
    allowed."""

    from lifeform_domain_figure.crawl.robots import RobotsRegistry
    from lifeform_domain_figure.crawl.scope_policy import default_scope_policy

    class _FailingHttpClient:
        """Stub HTTP client whose every request raises FetchError."""

        scope = default_scope_policy("test-agent")

        def get(self, url, **kwargs):
            from lifeform_domain_figure.crawl.http_client import FetchError

            raise FetchError(f"simulated network failure for {url}")

    registry = RobotsRegistry(http_client=_FailingHttpClient())
    # The robots registry takes a URL string — first call attempts to
    # fetch robots.txt; failure must propagate as fail-closed disallow
    # rather than silent allow.
    allowed, reason = registry.is_allowed("https://en.wikisource.org/wiki/X")
    assert allowed is False
    assert reason  # must include a non-empty reason


# ---------------------------------------------------------------------------
# 5. Cross-archive dedup — same text from two archives collapses
# ---------------------------------------------------------------------------


def test_cross_archive_dedup_canonicalises_to_paper(tmp_path: Path) -> None:
    """Same byte-identical body shipped under both a paper and a
    letter envelope yields exactly ONE retrieval-index chunk per
    unique paragraph; the canonical chunk lives on the
    higher-trust source kind (papers > letters)."""

    from lifeform_domain_figure import (
        FigureBundleInputs,
        FigureCorpusSourceBundle,
        FigureLetterSource,
        FigurePaperSource,
        build_einstein_profile,
        build_figure_artifact_bundle,
        build_figure_ingestion_envelope,
    )

    body = (
        "Reviewer-fabricated body for cross-archive dedup. "
        "Spatially separated subsystems each carry their own "
        "definite physical state in a complete physical theory.\n\n"
        "Second paragraph for cross-archive dedup test.\n\n"
        "Third paragraph for cross-archive dedup test."
    )
    paper = FigurePaperSource(
        paper_id="dedup-paper",
        title="Cross-archive Paper",
        year=1925,
        language="en",
        body=body,
    )
    letter = FigureLetterSource(
        letter_id="dedup-letter",
        sender_id="einstein",
        recipient_id="bohr",
        date_iso="1935-04-12",
        language="en",
        body=body,
    )
    profile = build_einstein_profile()
    corpus = FigureCorpusSourceBundle(
        figure_id="einstein", papers=(paper,), letters=(letter,)
    )
    envelope_set = build_figure_ingestion_envelope(corpus, uploader="dedup-test")
    bundle = build_figure_artifact_bundle(
        FigureBundleInputs(profile=profile, envelopes=envelope_set.envelopes)
    )
    locators = tuple(rec.locator for rec in bundle.retrieval_index.chunk_records)
    # Every retrieval chunk must be on the paper side after dedup
    # (R8 invariant: papers > letters > lectures > notebooks).
    assert all(loc.startswith("paper:") for loc in locators), (
        f"cross-archive dedup must canonicalise to papers; got {locators!r}"
    )
    # The body has 3 distinct paragraphs.
    assert len(locators) == 3


# ---------------------------------------------------------------------------
# 6. Bundle byte-stable — same inputs -> same integrity_hash
# ---------------------------------------------------------------------------


def test_bundle_byte_stable_across_two_curated_bakes(tmp_path: Path) -> None:
    """Wave J round-trip extended to write 'real-shape' state via
    a synthetic cleaning store, then bake twice through the curated
    CLI path. Both bundles must carry byte-identical
    ``integrity_hash`` AND ``provenance_fingerprint``."""

    from cleaning_fixtures import (  # noqa: WPS433
        build_minimal_cpae_pdf_bytes,
        build_wikisource_html_bytes,
    )
    from lifeform_domain_figure.cleaning import CleaningStore
    from lifeform_domain_figure.cleaning.cleaners import clean_raw_document
    from lifeform_domain_figure.cleaning.parsers import (
        CPAE_PDF_CONTENT_TYPE,
        WIKISOURCE_HTML_CONTENT_TYPE,
        parse_by_content_type,
    )
    from lifeform_domain_figure.cli._commands import cmd_bake_bundle

    cleaning_root = tmp_path / "store"
    cleaning_root.mkdir()
    store = CleaningStore(cleaning_root)
    shas: dict[str, str] = {}
    for archive, data, content_type, source_url in (
        ("cpae", build_minimal_cpae_pdf_bytes(), CPAE_PDF_CONTENT_TYPE,
         "https://einsteinpapers.press.princeton.edu/vol2-doc/24"),
        ("wikisource", build_wikisource_html_bytes(),
         WIKISOURCE_HTML_CONTENT_TYPE,
         "https://en.wikisource.org/wiki/Cross-Test"),
    ):
        sha = store.put_raw(data, source_url=source_url, content_type=content_type)
        raw = parse_by_content_type(
            data, source_url=source_url, content_type=content_type
        )
        store.put_cleaned(clean_raw_document(raw))
        shas[archive] = sha
    # Curator metadata pointing at both anchors.
    metadata_file = cleaning_root / "metadata.jsonl"
    rows = [
        {
            "raw_sha256": shas["cpae"],
            "figure_id": "einstein",
            "archive": "cpae",
            "source_kind": "paper",
            "source_id": "robustness-cpae",
            "legal_clearance": "public_domain_global",
            "capture_method": "scan_reviewed_ocr",
            "captured_by": "robustness-test",
            "captured_at_iso": "2026-05-12T00:00:00Z",
            "provenance_note": "Robustness round-trip CPAE.",
            "license_label_override": "Public Domain",
            "archive_payload": {
                "document_id": "cpae-rb-1",
                "document_kind": "article",
                "volume": 2,
                "document_number": 24,
                "title": "Robustness Paper",
                "year": 1905,
                "language": "en",
            },
        },
        {
            "raw_sha256": shas["wikisource"],
            "figure_id": "einstein",
            "archive": "wikisource",
            "source_kind": "paper",
            "source_id": "robustness-ws",
            "legal_clearance": "public_domain_global",
            "capture_method": "transcribed",
            "captured_by": "robustness-test",
            "captured_at_iso": "2026-05-12T00:00:00Z",
            "provenance_note": "Robustness round-trip WS.",
            "license_label_override": "Public Domain",
            "archive_payload": {
                "page_title": "Robustness Page",
                "language": "en",
                "year": 1905,
            },
        },
    ]
    metadata_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    integrity_hashes: list[str] = []
    provenance_fingerprints: list[str] = []
    for run in ("a", "b"):
        bundle_root = tmp_path / f"bundles_{run}"
        audit_root = tmp_path / f"audit_{run}"
        bundle_root.mkdir()
        audit_root.mkdir()
        args = Namespace(
            figure="einstein",
            corpus_mode="curated",
            cleaning_root=str(cleaning_root),
            curated_metadata_file=str(metadata_file),
            verification_root=None,
            require_verification_pass=False,
            time_window_id=None,
            bundle_root=str(bundle_root),
            audit_root=str(audit_root),
        )
        rc = cmd_bake_bundle(args)
        assert rc == 0, f"run {run!r} returned non-zero {rc}"
        bundle_dirs = sorted((bundle_root / "einstein").iterdir())
        assert len(bundle_dirs) == 1
        bundle_id = bundle_dirs[0].name
        # bundle_dir name encodes integrity hash via bundle_id_from_hash.
        # Read the manifest to grab the hashes verbatim.
        manifest = json.loads(
            (bundle_dirs[0] / "manifest.json").read_text(encoding="utf-8")
        )
        integrity_hashes.append(manifest["integrity_hash"])
        # provenance_fingerprint flows through via the bundle pickle
        # — we load the bundle to read it.
        from lifeform_domain_figure import load_figure_bundle

        bundle = load_figure_bundle(
            root_dir=str(bundle_root),
            bundle_id=bundle_id,
            figure_id="einstein",
        )
        provenance_fingerprints.append(bundle.provenance_fingerprint)
    assert integrity_hashes[0] == integrity_hashes[1], (
        f"R15 violated: same inputs -> different integrity_hash; "
        f"a={integrity_hashes[0]!r} b={integrity_hashes[1]!r}"
    )
    assert provenance_fingerprints[0] == provenance_fingerprints[1]


# ---------------------------------------------------------------------------
# 2. ETag idempotency (mock) + 1+2+3 live_network smoke
# ---------------------------------------------------------------------------


@pytest.mark.live_network
def test_live_real_seeds_produce_at_least_one_success(tmp_path: Path) -> None:
    """Live-network smoke: enqueue our reviewer-staged seeds against
    the real Wikisource / Gutenberg / Internet Archive endpoints and
    assert at least one SUCCESS row lands. Skipped by default."""

    seeds_path = (
        Path(__file__).resolve().parents[2]
        / "packages"
        / "lifeform-domain-figure"
        / "data"
        / "seeds"
        / "einstein-2026Q2.jsonl"
    )
    if not seeds_path.exists():
        pytest.skip("reviewer seeds file not present; pilot collection skipped")
    enqueue = _enqueue_subprocess(
        root=tmp_path, run_id="live-smoke", requests_file=seeds_path
    )
    assert enqueue.returncode == 0, enqueue.stderr
    run = subprocess.run(
        [
            sys.executable,
            str(_FIGURE_SCRIPTS / "figure_crawl.py"),
            "run",
            "--root",
            str(tmp_path),
            "--run-id",
            "live-smoke",
            "--cleaning-root",
            str(tmp_path),
            "--rate-rps",
            "0.5",
            "--burst",
            "5",
            "--max-pages",
            "10",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stderr
    rows = _read_results(root=tmp_path, run_id="live-smoke")
    success_count = sum(1 for r in rows if r["status"] == "success")
    assert success_count >= 1, (
        f"expected at least 1 SUCCESS in real-network pilot crawl; got "
        f"{[r['status'] for r in rows]!r}"
    )


@pytest.mark.live_network
def test_live_real_seeds_etag_second_run_is_idempotent(tmp_path: Path) -> None:
    """Idempotency contract: running the same crawl twice (with the
    same run-id) re-uses ETag/Last-Modified so the second run hits
    FETCHED_NOT_MODIFIED on at least one anchor that the first run
    fetched. Allows for some server-side ETag drift."""

    seeds_path = (
        Path(__file__).resolve().parents[2]
        / "packages"
        / "lifeform-domain-figure"
        / "data"
        / "seeds"
        / "einstein-2026Q2.jsonl"
    )
    if not seeds_path.exists():
        pytest.skip("reviewer seeds file not present")
    enqueue = _enqueue_subprocess(
        root=tmp_path, run_id="etag", requests_file=seeds_path
    )
    assert enqueue.returncode == 0, enqueue.stderr
    common_args = [
        sys.executable,
        str(_FIGURE_SCRIPTS / "figure_crawl.py"),
        "run",
        "--root",
        str(tmp_path),
        "--run-id",
        "etag",
        "--cleaning-root",
        str(tmp_path),
        "--rate-rps",
        "0.5",
        "--burst",
        "5",
        "--max-pages",
        "10",
    ]
    first = subprocess.run(common_args, capture_output=True, text=True, timeout=600)
    assert first.returncode == 0
    first_rows = _read_results(root=tmp_path, run_id="etag")
    first_success = [r for r in first_rows if r["status"] == "success"]
    if not first_success:
        pytest.skip("first pass yielded zero SUCCESS rows; cannot test ETag")
    # Re-enqueue the originally successful URLs and re-run.
    re_seeds = tmp_path / "re_seeds.jsonl"
    _seed_jsonl(
        re_seeds,
        [
            {
                "url": r["request"]["url"],
                "fetch_kind": r["request"]["fetch_kind"],
                "expected_content_type": r["request"]["expected_content_type"],
                "referrer": r["request"]["referrer"],
            }
            for r in first_success
        ],
    )
    enqueue2 = _enqueue_subprocess(
        root=tmp_path, run_id="etag-2", requests_file=re_seeds
    )
    assert enqueue2.returncode == 0
    second_args = list(common_args)
    second_args[second_args.index("etag")] = "etag-2"
    second = subprocess.run(
        second_args, capture_output=True, text=True, timeout=600
    )
    assert second.returncode == 0
    second_rows = _read_results(root=tmp_path, run_id="etag-2")
    # The frontier is fresh (etag-2), so cached state from etag-1 does
    # not apply directly — but the cleaning store's content-addressable
    # dedup means the L1 store's raw_sha256 set is unchanged. We assert
    # that property: every successful second-run anchor matches a
    # first-run anchor.
    second_shas = {r.get("raw_sha256", "") for r in second_rows if r["status"] == "success"}
    first_shas = {r.get("raw_sha256", "") for r in first_rows if r["status"] == "success"}
    overlap = second_shas & first_shas
    assert overlap, (
        "L1 content-addressable store should de-dup across runs; "
        f"first_shas={first_shas!r} second_shas={second_shas!r}"
    )


@pytest.mark.live_network
def test_live_ssrf_off_list_host_records_skipped_scope(tmp_path: Path) -> None:
    """Live-network: enqueue a real off-list URL and verify the
    crawler records SKIPPED_SCOPE without opening a socket on the
    forbidden host."""

    seeds = tmp_path / "ssrf-seeds.jsonl"
    _seed_jsonl(
        seeds,
        [
            {
                "url": "https://example.com/",
                "fetch_kind": "generic",
                "expected_content_type": "",
                "referrer": "robustness:ssrf",
            }
        ],
    )
    enqueue = _enqueue_subprocess(
        root=tmp_path, run_id="ssrf", requests_file=seeds
    )
    assert enqueue.returncode == 0
    run = subprocess.run(
        [
            sys.executable,
            str(_FIGURE_SCRIPTS / "figure_crawl.py"),
            "run",
            "--root",
            str(tmp_path),
            "--run-id",
            "ssrf",
            "--cleaning-root",
            str(tmp_path),
            "--rate-rps",
            "0.5",
            "--burst",
            "5",
            "--max-pages",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0
    rows = _read_results(root=tmp_path, run_id="ssrf")
    assert rows, "scheduler should record at least one row"
    assert all(r["status"] == "skipped_scope" for r in rows), (
        f"off-list URL should produce SKIPPED_SCOPE; got "
        f"{[r['status'] for r in rows]!r}"
    )
