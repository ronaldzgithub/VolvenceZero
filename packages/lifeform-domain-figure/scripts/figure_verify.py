"""Figure-vertical L2 verification CLI (debt #28).

Three subcommands:

* ``run-batch`` — read a JSONL of curator-supplied provenance
  records, run every implemented auto verifier against each, and
  append results to the :class:`VerificationLedger`. Each line of
  the provenance file carries the :class:`SourceProvenance` fields
  plus three verifier-specific extras: ``document_year``,
  ``figure_lifespan`` (``[birth, death]``), and ``document_group_key``
  (used by the cross-source-byte verifier to cluster sources).
* ``review`` — two modes:
  * ``review --sample N [--seed K]`` — print N random anchors and
    their currently-effective verdicts (read-only inspection).
  * ``review --anchor <sha> --check-kind <kind> --verdict
    <pass|fail|needs_review> --reviewer <id> --evidence <text>`` —
    append a single ``human:<reviewer>`` check to the ledger,
    overriding any prior auto verdict for that anchor + kind.
* ``list`` — print every anchor and its latest-per-kind matrix.

This script is curator-facing; runtime systems never call it. It
strictly does not mutate or inspect the cleaning store; the
verification ledger lives next to (not inside) the L1 store.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from lifeform_domain_figure.corpus.provenance import (
    CaptureMethod,
    LegalClearance,
    SourceProvenance,
)
from lifeform_domain_figure.verification import (
    CheckKind,
    Verdict,
    VerificationCheck,
    VerificationLedger,
    verify_cross_source_byte,
    verify_date_plausibility,
    verify_license_page_level,
)


def _load_provenance_jsonl(
    path: Path,
) -> tuple[tuple[SourceProvenance, dict], ...]:
    """Read a JSONL of provenance records + verifier-specific extras.

    Returns a tuple of ``(SourceProvenance, extras_dict)`` pairs in
    file order. ``extras_dict`` carries ``document_year`` /
    ``figure_lifespan`` / ``document_group_key`` for the verifier
    callers; missing fields raise ``ValueError`` (no silent default).
    """

    if not path.exists():
        raise FileNotFoundError(f"provenance file not found: {path}")
    pairs: list[tuple[SourceProvenance, dict]] = []
    with path.open("r", encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            try:
                prov = SourceProvenance(
                    source_id=str(payload["source_id"]),
                    figure_id=str(payload["figure_id"]),
                    source_url=str(payload["source_url"]),
                    license_label=str(payload["license_label"]),
                    legal_clearance=LegalClearance(payload["legal_clearance"]),
                    capture_method=CaptureMethod(payload["capture_method"]),
                    captured_by=str(payload["captured_by"]),
                    captured_at_iso=str(payload["captured_at_iso"]),
                    byte_sha256=str(payload["byte_sha256"]),
                    provenance_note=str(payload["provenance_note"]),
                    jurisdiction_hint=str(payload.get("jurisdiction_hint", "")),
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"provenance file {path}:{line_no} failed to parse: {exc}"
                ) from exc
            extras = {
                "document_year": payload.get("document_year"),
                "figure_lifespan": payload.get("figure_lifespan"),
                "document_group_key": payload.get("document_group_key"),
            }
            pairs.append((prov, extras))
    return tuple(pairs)


def _cmd_run_batch(args: argparse.Namespace) -> int:
    ledger = VerificationLedger(Path(args.root))
    pairs = _load_provenance_jsonl(Path(args.provenance_file))
    if not pairs:
        print("run-batch: provenance file has no entries", file=sys.stderr)
        return 2
    now = datetime.now(timezone.utc).isoformat()
    appended_count = 0
    for prov, extras in pairs:
        document_year = extras["document_year"]
        figure_lifespan = extras["figure_lifespan"]
        if document_year is None or figure_lifespan is None:
            print(
                f"run-batch: source_id={prov.source_id} skipping "
                f"date_plausibility (document_year / figure_lifespan missing)",
                file=sys.stderr,
            )
        else:
            check = verify_date_plausibility(
                prov,
                document_year=int(document_year),
                figure_lifespan=(int(figure_lifespan[0]), int(figure_lifespan[1])),
                now_iso=now,
            )
            ledger.append(check)
            appended_count += 1
        license_check = verify_license_page_level(prov, now_iso=now)
        ledger.append(license_check)
        appended_count += 1
    groups: dict[str, list[SourceProvenance]] = defaultdict(list)
    for prov, extras in pairs:
        key = extras["document_group_key"]
        if not key:
            continue
        groups[str(key)].append(prov)
    for group_key, members in groups.items():
        checks = verify_cross_source_byte(
            tuple(members), document_group_key=group_key, now_iso=now
        )
        for check in checks:
            ledger.append(check)
            appended_count += 1
    print(
        f"run-batch: processed {len(pairs)} provenance(s); "
        f"appended {appended_count} check(s) across "
        f"{sum(1 for _ in ledger.list_anchors())} anchor(s)"
    )
    return 0


def _format_anchor_row(
    anchor_sha: str, ledger: VerificationLedger
) -> str:
    latest = ledger.latest_per_kind(anchor_sha)
    cells = []
    for kind in sorted(CheckKind, key=lambda k: k.value):
        check = latest.get(kind)
        if check is None:
            cells.append(f"{kind.value}=(missing)")
        else:
            cells.append(f"{kind.value}={check.verdict.value}")
    return f"{anchor_sha}\t" + " | ".join(cells)


def _cmd_review(args: argparse.Namespace) -> int:
    ledger = VerificationLedger(Path(args.root))
    if args.anchor is not None:
        if not (
            args.check_kind and args.verdict and args.reviewer and args.evidence
        ):
            print(
                "review: --anchor mode requires --check-kind --verdict "
                "--reviewer --evidence",
                file=sys.stderr,
            )
            return 2
        check = VerificationCheck(
            check_kind=CheckKind(args.check_kind),
            verdict=Verdict(args.verdict),
            evidence=tuple(args.evidence),
            reviewer_id=f"human:{args.reviewer}",
            reviewed_at_iso=datetime.now(timezone.utc).isoformat(),
            source_byte_sha256=args.anchor,
        )
        path = ledger.append(check)
        print(
            f"review: appended human override for anchor={args.anchor[:12]}... "
            f"check_kind={check.check_kind.value} verdict={check.verdict.value} "
            f"-> {path}"
        )
        return 0
    anchors = list(ledger.list_anchors())
    if not anchors:
        print("review: ledger is empty (no anchors)")
        return 0
    rng = random.Random(args.seed)
    sample_size = min(args.sample, len(anchors))
    sampled = rng.sample(anchors, sample_size)
    for anchor in sampled:
        print(_format_anchor_row(anchor, ledger))
    print(
        f"review: sampled {sample_size} of {len(anchors)} anchor(s)"
    )
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    ledger = VerificationLedger(Path(args.root))
    found_any = False
    for anchor in ledger.list_anchors():
        found_any = True
        print(_format_anchor_row(anchor, ledger))
    if not found_any:
        print("(no anchors)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Figure-vertical L2 verification CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_batch = subparsers.add_parser(
        "run-batch", help="Run all auto verifiers across a provenance JSONL"
    )
    run_batch.add_argument("--root", required=True, help="Verification ledger root path")
    run_batch.add_argument(
        "--provenance-file",
        required=True,
        help="JSONL of SourceProvenance + extras (document_year, figure_lifespan, document_group_key)",
    )
    run_batch.set_defaults(func=_cmd_run_batch)

    review = subparsers.add_parser(
        "review",
        help="Sample anchors for inspection, or write a single-anchor override",
    )
    review.add_argument("--root", required=True, help="Verification ledger root path")
    review.add_argument("--sample", type=int, default=10, help="Sample size (read-only mode)")
    review.add_argument("--seed", type=int, default=0, help="Random seed for sample mode")
    review.add_argument("--anchor", default=None, help="Single-anchor override mode: target sha")
    review.add_argument(
        "--check-kind",
        default=None,
        help="Override mode: CheckKind value (e.g., date_plausibility)",
    )
    review.add_argument(
        "--verdict",
        default=None,
        choices=[v.value for v in Verdict],
        help="Override mode: verdict",
    )
    review.add_argument("--reviewer", default=None, help="Override mode: human reviewer id")
    review.add_argument(
        "--evidence",
        nargs="+",
        default=None,
        help="Override mode: one or more evidence bullets",
    )
    review.set_defaults(func=_cmd_review)

    list_cmd = subparsers.add_parser(
        "list", help="List every anchor with its latest-per-kind verdict matrix"
    )
    list_cmd.add_argument("--root", required=True, help="Verification ledger root path")
    list_cmd.set_defaults(func=_cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
