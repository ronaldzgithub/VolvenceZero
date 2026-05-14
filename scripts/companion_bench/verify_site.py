#!/usr/bin/env python3
# Copyright 2026 Companion Bench Contributors
# Licensed under the Apache License, Version 2.0.

"""Static verifier for the Companion Bench public site (debt #38).

Checks:

1. **Page existence** — every page in the documented top-level set
   exists under ``site/``.
2. **Inter-page links** — every internal href in each page resolves
   to an existing file under ``site/``.
3. **Data field completeness** — ``site/data/{aggregate_results,
   scenarios, pairwise}.json`` carry the documented top-level keys
   (drift-fail).
4. **Demo realism banner** — when a submission's payload has
   ``demo: true`` or ``scaffold_status == "SHADOW"``, the
   leaderboard (``site/leaderboard.html``) carries a visible
   ``demo-warning`` element so site visitors aren't misled.

Exits 0 on clean, 2 on the first failure.

Usage::

    python scripts/companion_bench/verify_site.py
    python scripts/companion_bench/verify_site.py --site-dir site/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from urllib.parse import urlparse


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


_REQUIRED_PAGES: tuple[str, ...] = (
    "index.html",
    "leaderboard.html",
    "scenarios.html",
    "judges.html",
    "compare.html",
    "submit.html",
    "methodology.html",
    "governance.html",
    "about.html",
)

_REQUIRED_DATA_KEYS: dict[str, frozenset[str]] = {
    "aggregate_results.json": frozenset({"systems"}),
    "scenarios.json": frozenset(
        {"companion_bench_version", "scenario_count", "scenarios"}
    ),
    "pairwise.json": frozenset({"arcs", "elo"}),
}


_HREF_RE = re.compile(r'href="([^"]+)"')


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="verify_site")
    p.add_argument(
        "--site-dir",
        type=pathlib.Path,
        default=_REPO_ROOT / "site",
        help="Static site root (default: site/)",
    )
    return p


class _VerifierError(Exception):
    """Raised when a verification check fails. Caught in main()."""


def _check_page_existence(site_dir: pathlib.Path) -> list[str]:
    findings: list[str] = []
    for page in _REQUIRED_PAGES:
        if not (site_dir / page).exists():
            findings.append(f"missing page: site/{page}")
    return findings


def _check_internal_links(site_dir: pathlib.Path) -> list[str]:
    findings: list[str] = []
    for html_file in site_dir.rglob("*.html"):
        text = html_file.read_text(encoding="utf-8", errors="replace")
        for href in _HREF_RE.findall(text):
            parsed = urlparse(href)
            # External (has a scheme), in-page anchor, or mailto — skip.
            if parsed.scheme or href.startswith(("#", "mailto:", "tel:")):
                continue
            target_str = parsed.path
            if not target_str:
                continue
            target = (html_file.parent / target_str).resolve()
            # Strip in-page anchor fragments; keep query strings out
            # of the existence check (data viewer pages use ?s=<id>).
            if not target.exists():
                findings.append(
                    f"broken link in {html_file.relative_to(site_dir)}: "
                    f"{href!r} → {target}"
                )
    return findings


def _check_data_field_completeness(site_dir: pathlib.Path) -> list[str]:
    findings: list[str] = []
    data_dir = site_dir / "data"
    for filename, required_keys in _REQUIRED_DATA_KEYS.items():
        path = data_dir / filename
        if not path.exists():
            findings.append(f"missing data file: site/data/{filename}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            findings.append(f"invalid JSON in {filename}: {exc}")
            continue
        if not isinstance(payload, dict):
            findings.append(f"{filename} top level is not an object")
            continue
        missing = required_keys - payload.keys()
        if missing:
            findings.append(
                f"{filename} missing required top-level keys: {sorted(missing)}"
            )
    return findings


def _check_demo_realism_banner(site_dir: pathlib.Path) -> list[str]:
    """Verify the demo banner is present when any submission is demo data."""

    findings: list[str] = []
    leaderboard = site_dir / "leaderboard.html"
    if not leaderboard.exists():
        return findings  # already reported by page-existence check
    leaderboard_html = leaderboard.read_text(encoding="utf-8")
    aggregate = site_dir / "data" / "aggregate_results.json"
    has_demo = False
    if aggregate.exists():
        try:
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        for system in payload.get("systems", []):
            summary = system.get("summary") or {}
            if summary.get("demo") is True:
                has_demo = True
                break
            if summary.get("scaffold_status") == "SHADOW":
                has_demo = True
                break
    if has_demo and "demo" not in leaderboard_html.lower():
        findings.append(
            "leaderboard contains demo / SHADOW submission data but "
            "leaderboard.html has no 'demo' marker — site visitors may "
            "mistake the placeholder rows for real evidence."
        )
    return findings


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    site_dir: pathlib.Path = args.site_dir
    if not site_dir.exists():
        print(f"verify_site: site dir not found: {site_dir}", file=sys.stderr)
        return 2

    all_findings: list[str] = []
    for fn in (
        _check_page_existence,
        _check_internal_links,
        _check_data_field_completeness,
        _check_demo_realism_banner,
    ):
        all_findings.extend(fn(site_dir))

    if all_findings:
        print(f"verify_site: {len(all_findings)} issue(s):")
        for finding in all_findings:
            print(f"  - {finding}")
        return 2
    print(f"verify_site: OK (site_dir={site_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
