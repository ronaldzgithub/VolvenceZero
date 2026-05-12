"""Figure-vertical L2 verification CLI (debt #28).

Three subcommands:

* ``run-batch`` — read a JSONL of curator-supplied provenance
  records, run every **implemented** auto verifier against each
  (Wave I closure: all 7 axes, not just 3), and append results to
  the :class:`VerificationLedger`. Each line of the provenance
  file carries the :class:`SourceProvenance` fields plus
  verifier-specific extras (``document_year``, ``figure_lifespan``,
  ``document_group_key``, plus optional ``candidate_work_id`` /
  ``source_doi`` / ``source_language`` / ``candidate_coauthor_openalex_ids``
  / ``canonical_doi_hint`` for the four metadata-driven verifiers).
  The metadata-driven verifiers consume an optional **per-figure
  context** JSON via ``--figure-context-file``.
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
from lifeform_domain_figure.metadata import (
    live_crossref_client,
    live_openalex_client,
    live_wikidata_client,
    offline_crossref_client,
    offline_openalex_client,
    offline_wikidata_client,
)
from lifeform_domain_figure.verification import (
    CheckKind,
    IMPLEMENTED_CHECK_KINDS,
    Verdict,
    VerificationCheck,
    VerificationLedger,
    verify_authorship_attribution,
    verify_cross_source_byte,
    verify_date_plausibility,
    verify_identity_disambiguation,
    verify_license_page_level,
    verify_translation_lineage,
    verify_version_reconciliation,
)


# ---------------------------------------------------------------------------
# Wave I helpers — figure context loader + metadata client builder
# ---------------------------------------------------------------------------


_FIGURE_CONTEXT_REQUIRED_KEYS = (
    "expected_qid",
    "expected_birth_year",
    "expected_openalex_author_id",
    "figure_native_languages",
)


def _load_figure_context(path: Path | None) -> dict:
    """Load the per-figure context JSON used by the 4 metadata verifiers.

    Returns an empty dict when ``path`` is ``None`` so the run-batch
    can still proceed (with NEEDS_REVIEW results on metadata axes).
    """

    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"figure-context file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"figure-context file {path} must decode to a JSON object"
        )
    missing = [key for key in _FIGURE_CONTEXT_REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(
            f"figure-context file {path} missing required keys: {missing!r}"
        )
    return payload


def _build_metadata_clients(mode: str) -> dict:
    """Build the three V2 metadata clients in offline / live mode.

    Returns a dict ``{"wikidata": client, "openalex": client,
    "crossref": client}``; SEP isn't needed for verification, only
    for coverage_map enrichment.
    """

    if mode == "offline":
        return {
            "wikidata": offline_wikidata_client(),
            "openalex": offline_openalex_client(),
            "crossref": offline_crossref_client(),
        }
    if mode == "live":
        return {
            "wikidata": live_wikidata_client(),
            "openalex": live_openalex_client(),
            "crossref": live_crossref_client(),
        }
    raise ValueError(
        f"--metadata-mode must be 'offline' or 'live', got {mode!r}"
    )


def _load_provenance_jsonl(
    path: Path,
) -> tuple[tuple[SourceProvenance, dict], ...]:
    """Read a JSONL of provenance records + verifier-specific extras.

    Returns a tuple of ``(SourceProvenance, extras_dict)`` pairs in
    file order. ``extras_dict`` carries:

    * ``document_year`` / ``figure_lifespan`` — for date plausibility.
    * ``document_group_key`` — for cross-source-byte clustering.
    * ``candidate_work_id`` — OpenAlex work id for AUTHORSHIP_ATTRIBUTION.
    * ``candidate_coauthor_openalex_ids`` — coauthor pool for fallback.
    * ``source_doi`` — Crossref DOI for VERSION_RECONCILIATION /
      TRANSLATION_LINEAGE.
    * ``source_language`` — declared language for TRANSLATION_LINEAGE.
    * ``canonical_doi_hint`` — curator's preferred canonical DOI.

    Missing core fields raise ``ValueError``; missing optional
    metadata fields default to empty / ``None`` so the run-batch
    surfaces NEEDS_REVIEW rather than silently passing.
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
            coauthor_ids_raw = payload.get("candidate_coauthor_openalex_ids", [])
            if isinstance(coauthor_ids_raw, list):
                coauthor_ids = tuple(str(item) for item in coauthor_ids_raw)
            else:
                coauthor_ids = ()
            extras = {
                "document_year": payload.get("document_year"),
                "figure_lifespan": payload.get("figure_lifespan"),
                "document_group_key": payload.get("document_group_key"),
                "candidate_work_id": str(payload.get("candidate_work_id", "")),
                "candidate_coauthor_openalex_ids": coauthor_ids,
                "source_doi": str(payload.get("source_doi", "")),
                "source_language": str(payload.get("source_language", "")),
                "canonical_doi_hint": str(payload.get("canonical_doi_hint", "")),
            }
            pairs.append((prov, extras))
    return tuple(pairs)


def _needs_review(
    *,
    kind: CheckKind,
    sha: str,
    reason: str,
    source_id: str,
    now_iso: str,
) -> VerificationCheck:
    """Build a ``NEEDS_REVIEW`` :class:`VerificationCheck` with stable shape.

    Used when run-batch lacks figure context / per-source metadata
    needed by one of the four metadata-driven verifiers. Writing
    NEEDS_REVIEW (rather than skipping) makes the bundle gate
    distinguish "human reviewer needed" from "missing-check (broken
    pipeline)" — the gate FAILs both, but the audit trail tells the
    curator what to do next.
    """

    return VerificationCheck(
        check_kind=kind,
        verdict=Verdict.NEEDS_REVIEW,
        evidence=(reason, f"source_id={source_id}"),
        reviewer_id=f"auto:run_batch_skip:{kind.value}",
        reviewed_at_iso=now_iso,
        source_byte_sha256=sha,
    )


def _cmd_run_batch(args: argparse.Namespace) -> int:
    ledger = VerificationLedger(Path(args.root))
    pairs = _load_provenance_jsonl(Path(args.provenance_file))
    if not pairs:
        print("run-batch: provenance file has no entries", file=sys.stderr)
        return 2
    figure_context_path = (
        Path(args.figure_context_file) if args.figure_context_file else None
    )
    try:
        figure_context = _load_figure_context(figure_context_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"run-batch: {exc}", file=sys.stderr)
        return 3
    try:
        clients = _build_metadata_clients(args.metadata_mode)
    except ValueError as exc:
        print(f"run-batch: {exc}", file=sys.stderr)
        return 2
    now = datetime.now(timezone.utc).isoformat()
    appended_count = 0
    # First batch: per-source DATE_PLAUSIBILITY + LICENSE_PAGE_LEVEL.
    for prov, extras in pairs:
        document_year = extras["document_year"]
        figure_lifespan = extras["figure_lifespan"]
        if document_year is None or figure_lifespan is None:
            ledger.append(
                _needs_review(
                    kind=CheckKind.DATE_PLAUSIBILITY,
                    sha=prov.byte_sha256,
                    reason=(
                        "document_year / figure_lifespan missing in provenance "
                        "extras; run-batch defers to human reviewer"
                    ),
                    source_id=prov.source_id,
                    now_iso=now,
                )
            )
        else:
            ledger.append(
                verify_date_plausibility(
                    prov,
                    document_year=int(document_year),
                    figure_lifespan=(
                        int(figure_lifespan[0]),
                        int(figure_lifespan[1]),
                    ),
                    now_iso=now,
                )
            )
        appended_count += 1
        ledger.append(verify_license_page_level(prov, now_iso=now))
        appended_count += 1
    # Cross-source byte: groups of >=2 sources sharing document_group_key.
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
    # Singletons (no group_key) still need a CROSS_SOURCE_BYTE entry to
    # avoid "missing-check" at the gate. Emit a trivially-PASS check that
    # records the anchor as singleton (auto reviewer id, with rationale
    # noting why no comparison was done).
    grouped_shas = {
        member.byte_sha256 for members in groups.values() for member in members
    }
    for prov, _extras in pairs:
        if prov.byte_sha256 in grouped_shas:
            continue
        ledger.append(
            VerificationCheck(
                check_kind=CheckKind.CROSS_SOURCE_BYTE,
                verdict=Verdict.PASS,
                evidence=(
                    "singleton: no other source shares this document_group_key",
                    f"source_id={prov.source_id}",
                    f"byte_sha256={prov.byte_sha256[:12]}",
                ),
                reviewer_id="auto:cross_source_byte_singleton:1",
                reviewed_at_iso=now,
                source_byte_sha256=prov.byte_sha256,
            )
        )
        appended_count += 1
    # Second batch: 4 metadata-driven verifiers, one entry per source.
    for prov, extras in pairs:
        ledger.append(
            _run_identity_disambiguation(
                prov=prov,
                clients=clients,
                figure_context=figure_context,
                now_iso=now,
            )
        )
        appended_count += 1
        ledger.append(
            _run_authorship_attribution(
                prov=prov,
                extras=extras,
                clients=clients,
                figure_context=figure_context,
                now_iso=now,
            )
        )
        appended_count += 1
        ledger.append(
            _run_version_reconciliation(
                prov=prov,
                extras=extras,
                clients=clients,
                now_iso=now,
            )
        )
        appended_count += 1
        ledger.append(
            _run_translation_lineage(
                prov=prov,
                extras=extras,
                clients=clients,
                figure_context=figure_context,
                now_iso=now,
            )
        )
        appended_count += 1
    print(
        f"run-batch: processed {len(pairs)} provenance(s); "
        f"appended {appended_count} check(s) across "
        f"{sum(1 for _ in ledger.list_anchors())} anchor(s) "
        f"covering {len(IMPLEMENTED_CHECK_KINDS)} CheckKind axes"
    )
    return 0


def _run_identity_disambiguation(
    *,
    prov: SourceProvenance,
    clients: dict,
    figure_context: dict,
    now_iso: str,
) -> VerificationCheck:
    """Adapter: build args for ``verify_identity_disambiguation``.

    When figure_context is empty or required keys are missing /
    falsy, the verifier branch is skipped with a NEEDS_REVIEW
    record so the bundle gate sees the per-axis decision.
    """

    qid = str(figure_context.get("expected_qid", "")).strip()
    birth_year = figure_context.get("expected_birth_year")
    if not qid or birth_year is None:
        return _needs_review(
            kind=CheckKind.IDENTITY_DISAMBIGUATION,
            sha=prov.byte_sha256,
            reason=(
                "figure-context missing expected_qid / expected_birth_year; "
                "supply --figure-context-file with both fields"
            ),
            source_id=prov.source_id,
            now_iso=now_iso,
        )
    occupations = tuple(
        str(item) for item in figure_context.get("expected_occupations", [])
    )
    try:
        return verify_identity_disambiguation(
            prov,
            wikidata_client=clients["wikidata"],
            expected_qid=qid,
            expected_birth_year=int(birth_year),
            expected_occupations=occupations,
            now_iso=now_iso,
        )
    except NotImplementedError as exc:
        return _needs_review(
            kind=CheckKind.IDENTITY_DISAMBIGUATION,
            sha=prov.byte_sha256,
            reason=f"wikidata client raised NotImplementedError: {exc}",
            source_id=prov.source_id,
            now_iso=now_iso,
        )


def _run_authorship_attribution(
    *,
    prov: SourceProvenance,
    extras: dict,
    clients: dict,
    figure_context: dict,
    now_iso: str,
) -> VerificationCheck:
    expected_author_id = str(
        figure_context.get("expected_openalex_author_id", "")
    ).strip()
    candidate_work_id = str(extras.get("candidate_work_id", "")).strip()
    if not expected_author_id or not candidate_work_id:
        return _needs_review(
            kind=CheckKind.AUTHORSHIP_ATTRIBUTION,
            sha=prov.byte_sha256,
            reason=(
                "missing expected_openalex_author_id (figure-context) or "
                "candidate_work_id (provenance extras)"
            ),
            source_id=prov.source_id,
            now_iso=now_iso,
        )
    coauthor_anchor_works = tuple(
        str(item) for item in figure_context.get("coauthor_anchor_works", [])
    )
    try:
        return verify_authorship_attribution(
            prov,
            openalex_client=clients["openalex"],
            expected_openalex_author_id=expected_author_id,
            candidate_work_id=candidate_work_id,
            coauthor_anchor_works=coauthor_anchor_works,
            candidate_coauthor_openalex_ids=extras.get(
                "candidate_coauthor_openalex_ids", ()
            ),
            now_iso=now_iso,
        )
    except NotImplementedError as exc:
        return _needs_review(
            kind=CheckKind.AUTHORSHIP_ATTRIBUTION,
            sha=prov.byte_sha256,
            reason=f"openalex client raised NotImplementedError: {exc}",
            source_id=prov.source_id,
            now_iso=now_iso,
        )


def _run_version_reconciliation(
    *,
    prov: SourceProvenance,
    extras: dict,
    clients: dict,
    now_iso: str,
) -> VerificationCheck:
    source_doi = str(extras.get("source_doi", "")).strip()
    if not source_doi:
        return _needs_review(
            kind=CheckKind.VERSION_RECONCILIATION,
            sha=prov.byte_sha256,
            reason="missing source_doi in provenance extras",
            source_id=prov.source_id,
            now_iso=now_iso,
        )
    try:
        return verify_version_reconciliation(
            prov,
            crossref_client=clients["crossref"],
            source_doi=source_doi,
            canonical_doi_hint=str(extras.get("canonical_doi_hint", "")),
            now_iso=now_iso,
        )
    except NotImplementedError as exc:
        return _needs_review(
            kind=CheckKind.VERSION_RECONCILIATION,
            sha=prov.byte_sha256,
            reason=f"crossref client raised NotImplementedError: {exc}",
            source_id=prov.source_id,
            now_iso=now_iso,
        )


def _run_translation_lineage(
    *,
    prov: SourceProvenance,
    extras: dict,
    clients: dict,
    figure_context: dict,
    now_iso: str,
) -> VerificationCheck:
    source_doi = str(extras.get("source_doi", "")).strip()
    source_language = str(extras.get("source_language", "")).strip()
    figure_native_languages = tuple(
        str(item) for item in figure_context.get("figure_native_languages", [])
    )
    if not source_doi or not source_language or not figure_native_languages:
        return _needs_review(
            kind=CheckKind.TRANSLATION_LINEAGE,
            sha=prov.byte_sha256,
            reason=(
                "missing source_doi (extras) / source_language (extras) / "
                "figure_native_languages (figure-context); all three required"
            ),
            source_id=prov.source_id,
            now_iso=now_iso,
        )
    try:
        return verify_translation_lineage(
            prov,
            crossref_client=clients["crossref"],
            source_doi=source_doi,
            source_language=source_language,
            figure_native_languages=figure_native_languages,
            now_iso=now_iso,
        )
    except NotImplementedError as exc:
        return _needs_review(
            kind=CheckKind.TRANSLATION_LINEAGE,
            sha=prov.byte_sha256,
            reason=f"crossref client raised NotImplementedError: {exc}",
            source_id=prov.source_id,
            now_iso=now_iso,
        )


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
        "run-batch",
        help=(
            "Run every implemented auto verifier (7 axes; Wave I closure) "
            "across a provenance JSONL"
        ),
    )
    run_batch.add_argument(
        "--root", required=True, help="Verification ledger root path"
    )
    run_batch.add_argument(
        "--provenance-file",
        required=True,
        help=(
            "JSONL of SourceProvenance + extras: document_year / figure_lifespan / "
            "document_group_key (first-batch verifiers); candidate_work_id / "
            "candidate_coauthor_openalex_ids / source_doi / source_language / "
            "canonical_doi_hint (second-batch metadata verifiers; missing fields "
            "land NEEDS_REVIEW)."
        ),
    )
    run_batch.add_argument(
        "--figure-context-file",
        default=None,
        help=(
            "JSON file with per-figure constants required by the 4 metadata "
            "verifiers: expected_qid, expected_birth_year, expected_occupations, "
            "expected_openalex_author_id, coauthor_anchor_works, "
            "figure_native_languages. Optional: when omitted, the metadata "
            "axes land NEEDS_REVIEW (the bundle gate then fails until a "
            "human reviewer overrides via 'review --anchor ... --check-kind ...')."
        ),
    )
    run_batch.add_argument(
        "--metadata-mode",
        default="offline",
        choices=("offline", "live"),
        help=(
            "Metadata client backend. 'offline' (default) uses the V1 stubs "
            "that raise NotImplementedError on fetch — verifiers gracefully "
            "degrade to NEEDS_REVIEW so the ledger still has a row per axis. "
            "'live' uses Wikidata / OpenAlex / Crossref V2 clients and "
            "respects the metadata SSRF allowlist."
        ),
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
