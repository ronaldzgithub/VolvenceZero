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
from .optimize import select_pareto_candidates
from .propose import propose_changes
from .rare_heavy import (
    RareHeavyEvaluationSpec,
    RareHeavyTrainingSpec,
    create_rare_heavy_request,
)
from .sources import latest_applied_timestamp, load_source_bundle, parse_evidence_timestamp
from .task_benchmark import run_task_benchmark
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
    mine.add_argument(
        "--live-outcome-root",
        type=Path,
        help=(
            "Explicit lifeform-service live_dialogue_outcomes directory. "
            "It is never auto-discovered because these are privacy-scoped artifacts."
        ),
    )
    mine.add_argument("--max-transcripts", type=int)
    mine.add_argument("--max-verdicts", type=int)
    mine.add_argument("--max-plans", type=int)
    mine.add_argument("--max-bench-bundles", type=int)
    mine.add_argument("--max-live-outcomes", type=int)
    evidence_window = mine.add_mutually_exclusive_group()
    evidence_window.add_argument("--evidence-since")
    evidence_window.add_argument("--evidence-since-ledger", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run the loop-external task-level diagnostic benchmark",
    )
    benchmark.add_argument("target", help="Repository-relative editable harness asset")
    _add_backend_arguments(benchmark, default="openai", allow_none=False)
    benchmark.add_argument("--candidate-asset", type=Path)
    benchmark.add_argument("--suite", type=Path)
    benchmark.add_argument("--output", type=Path)

    propose = subparsers.add_parser("propose", help="Generate bounded proposal bundles")
    propose.add_argument("failure_patterns", type=Path)
    _add_embedding_arguments(propose)
    _add_backend_arguments(propose, default="openai", allow_none=False)
    propose.add_argument("--output", type=Path)
    propose.add_argument("--max-proposals", type=int, default=3)
    propose.add_argument("--candidates-per-pattern", type=int, default=1)

    select = subparsers.add_parser(
        "select",
        help="Select a bounded Pareto front or emit an explicit STOP decision",
    )
    select.add_argument("proposals_root", type=Path)
    select.add_argument("--output", type=Path)
    select.add_argument("--selection-limit-per-component", type=int, default=1)

    rare_heavy = subparsers.add_parser(
        "plan-rare-heavy",
        help="Freeze a DISABLED, content-addressed Common Adapter build request",
    )
    rare_heavy.add_argument("--model-id", required=True)
    rare_heavy.add_argument("--model-weights-sha256", required=True)
    rare_heavy.add_argument("--common-adapter-version", required=True)
    rare_heavy.add_argument("--traces", type=Path, required=True)
    rare_heavy.add_argument("--control-basis", type=Path, required=True)
    rare_heavy.add_argument("--held-out", type=Path, required=True)
    rare_heavy.add_argument("--output", type=Path)
    rare_heavy.add_argument(
        "--runtime-origin",
        choices=("hf-local", "hf-pretrained"),
        default="hf-pretrained",
    )
    rare_heavy.add_argument(
        "--target-modules",
        nargs="+",
        default=("q_proj", "v_proj", "o_proj"),
    )
    rare_heavy.add_argument(
        "--hook-layers",
        required=True,
        help="Comma-separated, explicit layer indices",
    )
    rare_heavy.add_argument("--lora-rank", type=int, default=8)
    rare_heavy.add_argument("--lora-alpha", type=int, default=16)
    rare_heavy.add_argument("--lora-dropout", type=float, default=0.0)
    rare_heavy.add_argument("--learning-rate", type=float, default=5e-4)
    rare_heavy.add_argument("--max-steps", type=int, default=200)
    rare_heavy.add_argument("--seed", type=int, default=20260801)
    rare_heavy.add_argument("--control-scale", type=float, default=0.12)
    rare_heavy.add_argument("--state-kv-states", type=int, default=16)
    rare_heavy.add_argument("--state-kv-epochs", type=int, default=4)
    rare_heavy.add_argument("--state-kv-slots", type=int, default=4)
    rare_heavy.add_argument("--state-kv-rank", type=int, default=4)
    rare_heavy.add_argument("--state-kv-norm-cap", type=float, default=0.2)
    rare_heavy.add_argument("--state-kv-learning-rate", type=float, default=0.05)
    rare_heavy.add_argument("--state-kv-seed", type=int, default=20260726)
    rare_heavy.add_argument("--min-case-count", type=int, default=8)
    rare_heavy.add_argument("--min-mean-relative-improvement", type=float, default=0.01)
    rare_heavy.add_argument("--max-regression-rate", type=float, default=0.25)
    rare_heavy.add_argument("--max-preservation-nll-regression", type=float, default=0.05)
    rare_heavy.add_argument("--min-counterfactual-accuracy", type=float, default=0.60)
    rare_heavy.add_argument(
        "--description",
        default="Shared common adapter: PEFT rare-heavy then State-KV distillation.",
    )

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
                live_outcome_root=args.live_outcome_root,
                max_live_outcomes=args.max_live_outcomes,
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
        if args.command == "benchmark":
            result = run_task_benchmark(
                config=config,
                target=args.target,
                backend=_required_backend(args),
                candidate_asset_path=args.candidate_asset,
                suite_path=args.suite,
                report_path=args.output,
            )
            candidate_summary = (
                "none"
                if result.candidate_pass_rate is None
                else f"{result.candidate_pass_rate:.3f}"
            )
            print(
                f"{result.status}: baseline={result.baseline_pass_rate:.3f}; "
                f"candidate={candidate_summary}; report={result.report_path}"
            )
            return 0 if result.status == "PASS" else 2
        if args.command == "propose":
            backend = _required_backend(args)
            result = propose_changes(
                config=config,
                failure_patterns_path=args.failure_patterns,
                backend=backend,
                embedder=_embedder(args),
                output_dir=args.output,
                max_proposals=args.max_proposals,
                candidates_per_pattern=args.candidates_per_pattern,
            )
            print(
                f"generated {len(result.proposal_dirs)} bundles; "
                f"skipped {result.skipped_duplicates} duplicates: {result.output_dir}"
            )
            return 0 if result.proposal_dirs else 2
        if args.command == "select":
            result = select_pareto_candidates(
                config=config,
                proposals_root=args.proposals_root,
                output_path=args.output,
                selection_limit_per_component=args.selection_limit_per_component,
            )
            print(
                f"{result.decision}: selected={list(result.selected_proposal_ids)}; "
                f"report={result.report_path}"
            )
            return 0 if result.decision == "SELECT" else 2
        if args.command == "plan-rare-heavy":
            try:
                hook_layers = tuple(
                    int(value.strip())
                    for value in args.hook_layers.split(",")
                    if value.strip()
                )
            except ValueError as exc:
                raise ForgeError(
                    "--hook-layers must contain comma-separated integers"
                ) from exc
            result = create_rare_heavy_request(
                config=config,
                model_id=args.model_id,
                model_weights_sha256=args.model_weights_sha256,
                traces_path=args.traces,
                control_basis_path=args.control_basis,
                held_out_path=args.held_out,
                training=RareHeavyTrainingSpec(
                    common_adapter_version=args.common_adapter_version,
                    runtime_origin=args.runtime_origin,
                    description=args.description,
                    seed=args.seed,
                    target_modules=tuple(args.target_modules),
                    hook_layers=hook_layers,
                    control_scale=args.control_scale,
                    lora_rank=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    learning_rate=args.learning_rate,
                    max_steps=args.max_steps,
                    state_kv_seed=args.state_kv_seed,
                    state_kv_states=args.state_kv_states,
                    state_kv_epochs=args.state_kv_epochs,
                    state_kv_slots=args.state_kv_slots,
                    state_kv_rank=args.state_kv_rank,
                    state_kv_norm_cap=args.state_kv_norm_cap,
                    state_kv_learning_rate=args.state_kv_learning_rate,
                ),
                evaluation=RareHeavyEvaluationSpec(
                    min_case_count=args.min_case_count,
                    min_mean_relative_improvement=(
                        args.min_mean_relative_improvement
                    ),
                    max_regression_rate=args.max_regression_rate,
                    max_preservation_nll_regression=(
                        args.max_preservation_nll_regression
                    ),
                    min_counterfactual_accuracy=(
                        args.min_counterfactual_accuracy
                    ),
                ),
                output_path=args.output,
            )
            print(f"planned {result.request_id}: {result.request_path}")
            return 0
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
