"""Command-line entry point for the bounded Forge workflow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from .apply import apply_proposal, reject_proposal
from .config import ForgeConfig, ForgeConfigError, ForgePaths
from .foundation import (
    BackendError,
    ForgeError,
    OpenAICompatibleBackend,
    ReplayStructuredBackend,
    SentenceTransformerEmbeddingBackend,
    StructuredBackend,
)
from .mine import mine_failures
from .propose import propose_changes
from .sources import latest_applied_timestamp, load_source_bundle, parse_evidence_timestamp
from .validate import validate_proposal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge", description="Bounded RSI harness forge for Volvence")
    parser.add_argument("--repo-root", type=Path, help="Volvence repository root (auto-discovered by default)")
    parser.add_argument("--transcripts-root", type=Path, help="Cursor transcript root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mine = subparsers.add_parser("mine", help="Parse evidence and mine semantic failure patterns")
    _add_embedding_arguments(mine)
    _add_backend_arguments(mine, default="openai", allow_none=False)
    mine.add_argument("--output", type=Path)
    mine.add_argument("--verdict-root", type=Path)
    mine.add_argument("--bench-root", type=Path)
    mine.add_argument("--max-transcripts", type=int)
    mine.add_argument("--max-verdicts", type=int)
    mine.add_argument("--max-plans", type=int)
    mine.add_argument("--max-bench-bundles", type=int)
    evidence_window = mine.add_mutually_exclusive_group()
    evidence_window.add_argument("--evidence-since")
    evidence_window.add_argument("--evidence-since-ledger", action="store_true")

    propose = subparsers.add_parser("propose", help="Generate bounded proposal bundles")
    propose.add_argument("failure_patterns", type=Path)
    _add_embedding_arguments(propose)
    _add_backend_arguments(propose, default="openai", allow_none=False)
    propose.add_argument("--output", type=Path)
    propose.add_argument("--max-proposals", type=int, default=3)

    validate = subparsers.add_parser("validate", help="Validate one proposal without applying it")
    validate.add_argument("proposal_dir", type=Path)
    _add_backend_arguments(validate, default="openai", allow_none=True)
    validate.add_argument("--report", type=Path)

    apply = subparsers.add_parser("apply", help="Apply or reject a human-reviewed proposal")
    apply.add_argument("proposal_dir", type=Path)
    apply.add_argument("--validation-report", type=Path)
    apply.add_argument("--human-approved-by", required=True)
    apply.add_argument("--reject", action="store_true")
    apply.add_argument("--reason")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        paths = ForgePaths.discover(repo_root=args.repo_root, transcripts_root=args.transcripts_root)
        config = ForgeConfig.load(paths)
        if args.command == "mine":
            backend = _required_backend(args)
            embedder = _embedder(args)
            evidence_since = None
            if args.evidence_since is not None:
                evidence_since = parse_evidence_timestamp(args.evidence_since)
            elif args.evidence_since_ledger:
                evidence_since = latest_applied_timestamp(paths.ledger_path)
            sources = load_source_bundle(
                paths,
                max_transcripts=args.max_transcripts,
                max_verdicts=args.max_verdicts,
                max_plans=args.max_plans,
                verdict_root=args.verdict_root,
                bench_root=args.bench_root,
                max_bench_bundles=args.max_bench_bundles,
                since=evidence_since,
            )
            result = mine_failures(
                config=config,
                sources=sources,
                embedder=embedder,
                backend=backend,
                output_dir=args.output,
            )
            print(
                f"mined {result.pattern_count} patterns ({result.in_surface_count} in-surface): "
                f"{result.output_dir}"
            )
            return 0
        if args.command == "propose":
            backend = _required_backend(args)
            result = propose_changes(
                config=config,
                failure_patterns_path=args.failure_patterns,
                backend=backend,
                embedder=_embedder(args),
                output_dir=args.output,
                max_proposals=args.max_proposals,
            )
            print(
                f"generated {len(result.proposal_dirs)} bundles; "
                f"skipped {result.skipped_duplicates} duplicates: {result.output_dir}"
            )
            return 0 if result.proposal_dirs else 2
        if args.command == "validate":
            result = validate_proposal(
                config=config,
                proposal_dir=args.proposal_dir,
                relevance_backend=_backend(args),
                report_path=args.report,
            )
            print(f"{result.status}: {result.report_path}")
            return 0 if result.status == "PASS" else 2
        if args.command == "apply":
            if args.reject:
                result = reject_proposal(
                    config=config,
                    proposal_dir=args.proposal_dir,
                    human_approved_by=args.human_approved_by,
                    reason=args.reason or "",
                )
            else:
                if args.validation_report is None:
                    parser.error("apply requires --validation-report unless --reject is used")
                result = apply_proposal(
                    config=config,
                    proposal_dir=args.proposal_dir,
                    validation_report_path=args.validation_report,
                    human_approved_by=args.human_approved_by,
                )
            print(f"{result.decision}: {result.proposal_id} → {result.target}; ledger={result.ledger_path}")
            return 0
        parser.error(f"Unknown command: {args.command}")
    except (ForgeError, ForgeConfigError, BackendError) as exc:
        print(f"forge: {exc}", file=sys.stderr)
        return 2


def _add_backend_arguments(parser: argparse.ArgumentParser, *, default: str, allow_none: bool) -> None:
    choices = ("openai", "replay", "none") if allow_none else ("openai", "replay")
    parser.add_argument("--backend", choices=choices, default=default)
    parser.add_argument("--replay-responses", type=Path)


def _add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("FORGE_EMBEDDING_MODEL", "moka-ai/m3e-base"),
    )
    parser.add_argument("--embedding-device", default=os.environ.get("FORGE_EMBEDDING_DEVICE", "cpu"))
    parser.add_argument("--allow-embedding-download", action="store_true")


def _backend(args: argparse.Namespace) -> StructuredBackend | None:
    if args.backend == "none":
        return None
    if args.backend == "replay":
        if args.replay_responses is None:
            raise BackendError("--backend replay requires --replay-responses")
        return ReplayStructuredBackend.from_path(args.replay_responses)
    return OpenAICompatibleBackend.from_env()


def _required_backend(args: argparse.Namespace) -> StructuredBackend:
    backend = _backend(args)
    if backend is None:
        raise BackendError(f"{args.command} requires a structured backend")
    return backend


def _embedder(args: argparse.Namespace) -> SentenceTransformerEmbeddingBackend:
    return SentenceTransformerEmbeddingBackend(
        args.embedding_model,
        device=args.embedding_device,
        local_files_only=not args.allow_embedding_download,
    )


if __name__ == "__main__":
    raise SystemExit(main())
