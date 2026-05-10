"""Figure cleaning pipeline CLI (debt #28 L1).

Three subcommands:

* ``clean`` — read one file, store as raw, parse with the dispatcher,
  run the current cleaner pipeline, persist the cleaned text.
* ``re-clean-all`` — iterate every raw entry under the store and run
  the requested cleaner pipeline against each, writing a new
  ``v{N}/`` directory next to existing versions and printing a
  per-raw character-delta summary.
* ``list-versions`` — print every raw sha256 with its persisted
  cleaner pipeline versions.

This script is intended for curators and reviewers; it is not a
runtime entry point. Imports are kept lazy where reasonable so
``--help`` is fast.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lifeform_domain_figure.cleaning import (
    CURRENT_CLEANER_PIPELINE_VERSION,
    CleaningStore,
    clean_raw_document,
    parse_by_content_type,
)


def _cmd_clean(args: argparse.Namespace) -> int:
    store = CleaningStore(Path(args.root))
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"clean: input file not found: {input_path}", file=sys.stderr)
        return 2
    data = input_path.read_bytes()
    raw_sha256 = store.put_raw(
        data, source_url=args.source_url, content_type=args.content_type
    )
    raw = parse_by_content_type(
        data, source_url=args.source_url, content_type=args.content_type
    )
    cleaned = clean_raw_document(raw, pipeline_version=args.pipeline_version)
    cleaned_dir = store.put_cleaned(cleaned)
    print(
        f"clean: raw_sha256={raw_sha256} parser_version={raw.parser_version} "
        f"cleaner_pipeline_version={cleaned.cleaner_pipeline_version} "
        f"chars_in={len(raw.text)} chars_out={len(cleaned.text)} "
        f"out_dir={cleaned_dir}"
    )
    return 0


def _cmd_re_clean_all(args: argparse.Namespace) -> int:
    store = CleaningStore(Path(args.root))
    pipeline_version = args.pipeline_version
    total = 0
    for raw_sha256 in store.list_raw():
        total += 1
        data, sidecar = store.get_raw(raw_sha256)
        existing_versions = store.list_cleaned_versions(raw_sha256)
        previous_cleaned = None
        prior_versions = tuple(v for v in existing_versions if v != pipeline_version)
        if prior_versions:
            highest_existing = max(prior_versions)
            previous_cleaned = store.get_cleaned(raw_sha256, highest_existing)
        try:
            raw = parse_by_content_type(
                data,
                source_url=sidecar.source_url,
                content_type=sidecar.content_type,
            )
        except (ValueError, OSError) as exc:
            print(
                f"re-clean-all: raw_sha256={raw_sha256} parse FAILED ({exc})",
                file=sys.stderr,
            )
            continue
        cleaned = clean_raw_document(raw, pipeline_version=pipeline_version)
        store.put_cleaned(cleaned)
        diff_chars = (
            len(cleaned.text) - len(previous_cleaned.text)
            if previous_cleaned is not None
            else None
        )
        diff_label = (
            f"chars_delta_vs_v{previous_cleaned.cleaner_pipeline_version}={diff_chars:+d}"
            if previous_cleaned is not None
            else "no_prior_version_to_diff"
        )
        print(
            f"re-clean-all: raw_sha256={raw_sha256} -> v{pipeline_version} "
            f"chars_out={len(cleaned.text)} {diff_label}"
        )
    print(f"re-clean-all: processed {total} raw entries")
    return 0


def _cmd_list_versions(args: argparse.Namespace) -> int:
    store = CleaningStore(Path(args.root))
    found_any = False
    for raw_sha256 in store.list_raw():
        found_any = True
        versions = store.list_cleaned_versions(raw_sha256)
        version_label = (
            ",".join(f"v{v}" for v in versions) if versions else "(no cleaned versions)"
        )
        print(f"{raw_sha256}\t{version_label}")
    if not found_any:
        print("(no raw entries)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Figure-vertical L1 cleaning pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean = subparsers.add_parser("clean", help="Clean one input file")
    clean.add_argument("--root", required=True, help="Cleaning store root path")
    clean.add_argument("--input", required=True, help="Input file path")
    clean.add_argument(
        "--content-type",
        required=True,
        help=(
            "Source content_type (e.g., 'application/pdf', "
            "'text/html; profile=wikisource', 'text/plain; profile=gutenberg', "
            "'text/html; profile=gutenberg', 'application/json; profile=archive-org-ocr')"
        ),
    )
    clean.add_argument("--source-url", required=True, help="Original source URL")
    clean.add_argument(
        "--pipeline-version",
        type=int,
        default=CURRENT_CLEANER_PIPELINE_VERSION,
        help=f"Cleaner pipeline version (default {CURRENT_CLEANER_PIPELINE_VERSION})",
    )
    clean.set_defaults(func=_cmd_clean)

    re_clean = subparsers.add_parser(
        "re-clean-all",
        help="Re-clean every stored raw entry against a pipeline version",
    )
    re_clean.add_argument("--root", required=True, help="Cleaning store root path")
    re_clean.add_argument(
        "--pipeline-version",
        type=int,
        default=CURRENT_CLEANER_PIPELINE_VERSION,
        help=f"Cleaner pipeline version (default {CURRENT_CLEANER_PIPELINE_VERSION})",
    )
    re_clean.set_defaults(func=_cmd_re_clean_all)

    list_cmd = subparsers.add_parser(
        "list-versions",
        help="List raw sha256 keys and their persisted cleaner versions",
    )
    list_cmd.add_argument("--root", required=True, help="Cleaning store root path")
    list_cmd.set_defaults(func=_cmd_list_versions)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
