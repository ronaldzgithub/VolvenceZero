"""Command-line entry point for the bounded Forge workflow."""

from __future__ import annotations

import argparse
import json
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
    sha256_bytes,
)
from .mine import mine_failures
from .optimize import select_pareto_candidates
from .propose import propose_changes
from .rare_heavy import (
    RareHeavyEvaluationSpec,
    RareHeavyTrainingSpec,
    create_rare_heavy_request,
)
from .research_promotion import (
    authorize_research_candidate,
    import_praxist_candidate,
    rollback_research_candidate,
    validate_research_task,
)
from .research_control import (
    ResearchControlStatus,
    issue_research_control_directive,
    list_research_inbox,
    record_external_research_handoff,
    reconcile_research_control,
    review_research_request,
    submit_external_research_request,
    submit_research_request,
    validate_external_research_descriptor,
)
from .research_discovery import (
    CodexNativeResearchDiscoveryBackend,
    ReplayResearchDiscoveryBackend,
    discover_research_topics,
    review_research_topic,
    seal_research_demand,
    submit_bound_topic_for_a0,
    validate_research_demand,
)
from .research_loop import run_demand_research_loop_once
from .research_opportunity import (
    ResearchOpportunityStatus,
    scan_research_opportunities,
)
from .research_portfolio import (
    inspect_research_portfolio,
    review_research_study_outcome,
    run_research_portfolio_once,
    seal_research_portfolio,
    validate_research_portfolio,
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

    research_submit = subparsers.add_parser(
        "research-submit",
        help="Seal an approval-gated request for one exact Praxist research run",
    )
    research_submit.add_argument("task_manifest", type=Path)
    research_submit.add_argument("--task-project", type=Path, required=True)
    research_submit.add_argument("--praxist-executable", type=Path, required=True)
    research_submit.add_argument("--run-dir", type=Path, required=True)
    research_submit.add_argument("--requested-by", required=True)
    research_submit.add_argument("--reason", required=True)
    research_submit.add_argument(
        "--trigger-kind",
        choices=("human", "forge_failure_pattern", "typed_signal"),
        default="human",
    )
    research_submit.add_argument("--evidence", type=Path, action="append", default=[])
    research_submit.add_argument("--config-file", type=Path)
    research_submit.add_argument(
        "--agent-system",
        choices=("claude_sdk", "codex_sdk"),
        required=True,
    )
    research_submit.add_argument("--runtime", required=True)
    research_submit.add_argument("--codex-native", action="store_true")
    research_submit.add_argument("--model-provider", required=True)
    research_submit.add_argument("--model", required=True)
    research_submit.add_argument(
        "--strategy",
        choices=("auto", "mixed", "explore", "exploit"),
        default="auto",
    )
    research_submit.add_argument("--cohort", type=int)
    research_submit.add_argument("--generations", type=int)
    research_submit.add_argument("--startup-timeout", type=int, default=30)

    external_submit = subparsers.add_parser(
        "research-submit-external",
        help="Bind one external-domain descriptor to the shared A0 Praxist lifecycle",
    )
    external_submit.add_argument("descriptor", type=Path)
    external_submit.add_argument("--requested-by", required=True)
    external_submit.add_argument("--reason", required=True)
    external_submit.add_argument("--json", action="store_true")

    external_handoff = subparsers.add_parser(
        "research-handoff-external",
        help="Seal a completed external simulation result for domain-owned review",
    )
    external_handoff.add_argument("request", type=Path)
    external_handoff.add_argument("--recorded-by", required=True)
    external_handoff.add_argument("--reason", required=True)
    external_handoff.add_argument("--json", action="store_true")

    research_scan = subparsers.add_parser(
        "research-scan",
        help="Nominate typed failure patterns and submit exactly registered tasks for A0 review",
    )
    research_scan.add_argument("failure_patterns", type=Path)
    research_scan.add_argument("--registry", type=Path)
    research_scan.add_argument(
        "--once",
        action="store_true",
        required=True,
        help="Acknowledge that this command performs one bounded scan and exits",
    )
    research_scan.add_argument("--json", action="store_true")

    research_demand_validate = subparsers.add_parser(
        "research-demand-validate",
        help="Validate one complete, content-addressed Volvence ResearchDemand",
    )
    research_demand_validate.add_argument("demand", type=Path)
    research_demand_validate.add_argument("--json", action="store_true")

    research_demand_seal = subparsers.add_parser(
        "research-demand-seal",
        help="Seal one human-authored Demand draft into the automatic discovery inbox",
    )
    research_demand_seal.add_argument("draft", type=Path)
    research_demand_seal.add_argument("--output", type=Path)
    research_demand_seal.add_argument("--json", action="store_true")

    research_discover = subparsers.add_parser(
        "research-discover",
        help="Run one bounded read-only Codex topic discovery for an exact Demand",
    )
    research_discover.add_argument("demand", type=Path)
    research_discover.add_argument(
        "--backend",
        choices=("codex_sdk", "replay"),
        default="codex_sdk",
    )
    research_discover.add_argument(
        "--model",
        default=os.environ.get("FORGE_DISCOVERY_MODEL"),
        help="Exact Codex model; required for --backend codex_sdk",
    )
    research_discover.add_argument(
        "--codex-bin",
        type=Path,
        default=(
            Path(os.environ["FORGE_CODEX_BIN"])
            if os.environ.get("FORGE_CODEX_BIN")
            else None
        ),
    )
    research_discover.add_argument("--replay-responses", type=Path)
    research_discover.add_argument("--json", action="store_true")

    research_bind = subparsers.add_parser(
        "research-bind-topic",
        help="Create one named-human exact TopicProposal-to-Task binding",
    )
    research_bind.add_argument("demand", type=Path)
    research_bind.add_argument("proposal", type=Path)
    research_bind.add_argument("--mapping-id", required=True)
    research_bind.add_argument("--registry", type=Path)
    research_bind.add_argument("--reviewed-by", required=True)
    research_bind.add_argument("--reason", required=True)
    research_bind.add_argument("--reject", action="store_true")
    research_bind.add_argument("--json", action="store_true")

    research_submit_binding = subparsers.add_parser(
        "research-submit-binding",
        help="Submit one approved DemandBinding to the existing human A0 gate",
    )
    research_submit_binding.add_argument("binding", type=Path)
    research_submit_binding.add_argument("--json", action="store_true")

    research_loop = subparsers.add_parser(
        "research-loop",
        help="Run one bounded demand discovery, A0 submission, and approved reconcile pass",
    )
    research_loop.add_argument(
        "--once",
        action="store_true",
        required=True,
        help="Acknowledge that this command performs one bounded pass and exits",
    )
    research_loop.add_argument("--demand-root", type=Path)
    research_loop.add_argument(
        "--backend",
        choices=("codex_sdk", "replay"),
        default="codex_sdk",
    )
    research_loop.add_argument(
        "--model",
        default=os.environ.get("FORGE_DISCOVERY_MODEL"),
        help="Exact Codex model; required for --backend codex_sdk",
    )
    research_loop.add_argument(
        "--codex-bin",
        type=Path,
        default=(
            Path(os.environ["FORGE_CODEX_BIN"])
            if os.environ.get("FORGE_CODEX_BIN")
            else None
        ),
    )
    research_loop.add_argument("--replay-responses", type=Path)
    research_loop.add_argument("--max-demands", type=int, default=128)
    research_loop.add_argument("--max-new-discoveries", type=int, default=1)
    research_loop.add_argument("--max-new-requests", type=int, default=8)
    research_loop.add_argument("--max-reconciles", type=int, default=8)
    research_loop.add_argument("--json", action="store_true")

    portfolio_seal = subparsers.add_parser(
        "research-portfolio-seal",
        help="Seal one human-authored Portfolio draft into the exact registry",
    )
    portfolio_seal.add_argument("draft", type=Path)
    portfolio_seal.add_argument("--output", type=Path)
    portfolio_seal.add_argument("--json", action="store_true")

    portfolio_validate = subparsers.add_parser(
        "research-portfolio-validate",
        help="Validate one content-addressed research portfolio and its exact bindings",
    )
    portfolio_validate.add_argument("portfolio", type=Path)
    portfolio_validate.add_argument("--json", action="store_true")

    portfolio_status = subparsers.add_parser(
        "research-portfolio-status",
        help="Project dependency, A0, Praxist, and outcome state for one portfolio",
    )
    portfolio_status.add_argument("portfolio", type=Path)
    portfolio_status.add_argument("--json", action="store_true")

    portfolio_loop = subparsers.add_parser(
        "research-portfolio-loop",
        help="Run one bounded dependency-eligible pass through the existing Research Loop",
    )
    portfolio_loop.add_argument("portfolio", type=Path)
    portfolio_loop.add_argument(
        "--once",
        action="store_true",
        required=True,
        help="Acknowledge that this command performs one bounded pass and exits",
    )
    portfolio_loop.add_argument(
        "--backend",
        choices=("codex_sdk", "replay"),
        default="codex_sdk",
    )
    portfolio_loop.add_argument(
        "--model",
        default=os.environ.get("FORGE_DISCOVERY_MODEL"),
        help="Exact Codex model; required for --backend codex_sdk",
    )
    portfolio_loop.add_argument(
        "--codex-bin",
        type=Path,
        default=(
            Path(os.environ["FORGE_CODEX_BIN"])
            if os.environ.get("FORGE_CODEX_BIN")
            else None
        ),
    )
    portfolio_loop.add_argument("--replay-responses", type=Path)
    portfolio_loop.add_argument("--max-new-discoveries", type=int, default=1)
    portfolio_loop.add_argument("--max-new-requests", type=int, default=8)
    portfolio_loop.add_argument("--max-reconciles", type=int, default=8)
    portfolio_loop.add_argument("--json", action="store_true")

    portfolio_review = subparsers.add_parser(
        "research-portfolio-review",
        help="Seal a named-human dependency decision for one completed exact Request",
    )
    portfolio_review.add_argument("portfolio", type=Path)
    portfolio_review.add_argument("--study-id", required=True)
    portfolio_review.add_argument("--request", type=Path, required=True)
    portfolio_review.add_argument("--evidence", type=Path, action="append", required=True)
    portfolio_review.add_argument("--reviewed-by", required=True)
    portfolio_review.add_argument("--reason", required=True)
    portfolio_review.add_argument(
        "--decision",
        choices=("proceed", "revise", "stop"),
        required=True,
    )
    portfolio_review.add_argument("--json", action="store_true")

    research_inbox = subparsers.add_parser(
        "research-inbox",
        help="List immutable ResearchRequest lifecycle projections",
    )
    research_inbox.add_argument("--json", action="store_true")

    research_approve = subparsers.add_parser(
        "research-approve",
        help="Approve or reject the exact A0 Praxist research-start scope",
    )
    research_approve.add_argument("request", type=Path)
    research_approve.add_argument("--approved-by", required=True)
    research_approve.add_argument("--reason", required=True)
    research_approve.add_argument("--reject", action="store_true")
    research_approve.add_argument(
        "--portfolio",
        type=Path,
        help="Exact Portfolio whose global/lane capacity is included in this A0",
    )
    research_approve.add_argument(
        "--study-id",
        help="Exact Portfolio study receiving the A0 execution policy",
    )

    research_control = subparsers.add_parser(
        "research-control",
        help="Issue one revision-bound PAUSE, RESUME, or CANCEL directive and reconcile it",
    )
    research_control.add_argument("request", type=Path)
    research_control.add_argument(
        "--action",
        choices=("pause", "resume", "cancel"),
        required=True,
    )
    research_control.add_argument("--expected-event-sha256", required=True)
    research_control.add_argument("--requested-by", required=True)
    research_control.add_argument("--reason", required=True)
    research_control.add_argument("--grace", type=int, default=300)
    research_control.add_argument("--json", action="store_true")

    research_reconcile = subparsers.add_parser(
        "research-reconcile",
        help="Run one bounded Research Control Plane reconciliation pass",
    )
    research_reconcile.add_argument(
        "--once",
        action="store_true",
        required=True,
        help="Acknowledge that this command performs one pass and then exits",
    )
    research_reconcile.add_argument("--request", type=Path)
    research_reconcile.add_argument("--json", action="store_true")

    research_validate = subparsers.add_parser(
        "research-validate-task",
        help="Validate a Volvence-owned research and release contract",
    )
    research_validate.add_argument("task_manifest", type=Path)

    research_import = subparsers.add_parser(
        "research-import-praxist",
        help="Seal a content-addressed DISABLED candidate from a Praxist handoff",
    )
    research_import.add_argument("task_manifest", type=Path)
    research_import.add_argument("handoff", type=Path)
    research_import.add_argument("--run-dir", type=Path, required=True)
    research_import.add_argument("--output", type=Path)

    research_authorize = subparsers.add_parser(
        "research-authorize",
        help="Issue an offline SHADOW/ACTIVE authorization receipt without changing runtime",
    )
    research_authorize.add_argument("task_manifest", type=Path)
    research_authorize.add_argument("candidate", type=Path)
    research_authorize.add_argument("validation", type=Path)
    research_authorize.add_argument("gate", type=Path)
    research_authorize.add_argument("--to-wiring", choices=("shadow", "active"), required=True)
    research_authorize.add_argument("--previous-receipt", type=Path)
    research_authorize.add_argument("--authorized-by", required=True)
    research_authorize.add_argument("--reason", required=True)
    research_authorize.add_argument("--output", type=Path)

    research_rollback = subparsers.add_parser(
        "research-rollback",
        help="Issue an adjacent downgrade receipt from the last authorization receipt",
    )
    research_rollback.add_argument("previous_receipt", type=Path)
    research_rollback.add_argument("--to-wiring", choices=("disabled", "shadow"), required=True)
    research_rollback.add_argument("--authorized-by", required=True)
    research_rollback.add_argument("--reason", required=True)
    research_rollback.add_argument("--output", type=Path)
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
        if args.command == "research-submit":
            result = submit_research_request(
                config=config,
                task_manifest_path=args.task_manifest,
                task_project_path=args.task_project,
                praxist_executable=args.praxist_executable,
                run_dir=args.run_dir,
                requested_by=args.requested_by,
                reason=args.reason,
                trigger_kind=args.trigger_kind,
                evidence_paths=tuple(args.evidence),
                config_file=args.config_file,
                agent_system=args.agent_system,
                runtime=args.runtime,
                codex_native=args.codex_native,
                model_provider=args.model_provider,
                model=args.model,
                strategy=args.strategy,
                cohort=args.cohort,
                generations=args.generations,
                startup_timeout_seconds=args.startup_timeout,
            )
            print(
                f"AWAITING_RESEARCH_APPROVAL: {result.request_id}; "
                f"request={result.request_path}"
            )
            return 0
        if args.command == "research-submit-external":
            descriptor = validate_external_research_descriptor(
                config=config,
                descriptor_path=args.descriptor,
            )
            result = submit_external_research_request(
                config=config,
                descriptor_path=args.descriptor,
                requested_by=args.requested_by,
                reason=args.reason,
            )
            payload = {
                "schema_version": "forge-external-research-submit-result.v1",
                "state": "AWAITING_RESEARCH_APPROVAL",
                "descriptor_id": descriptor["descriptor_id"],
                "domain_id": descriptor["domain"]["domain_id"],
                "intent_id": descriptor["domain"]["intent_id"],
                "request_id": result.request_id,
                "request": str(result.request_path),
                "request_sha256": sha256_bytes(result.request_path.read_bytes()),
                "evidence_class": "simulation",
                "praxist_started": False,
                "volvence_promotion_eligible": False,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    f"AWAITING_RESEARCH_APPROVAL: {result.request_id}; "
                    f"request={result.request_path}; evidence=simulation"
                )
            return 0
        if args.command == "research-handoff-external":
            result = record_external_research_handoff(
                config=config,
                request_path=args.request,
                recorded_by=args.recorded_by,
                reason=args.reason,
            )
            handoff = json.loads(result.handoff_path.read_text(encoding="utf-8"))
            payload = {
                "schema_version": "forge-foundry-research-handoff-result.v1",
                "state": "HANDED_OFF_FOR_EXTERNAL_REVIEW",
                "contract_version": handoff["contract"]["contract_version"],
                "contract_schema_sha256": handoff["contract"]["schema"]["sha256"],
                "handoff_id": result.handoff_id,
                "handoff": str(result.handoff_path),
                "handoff_sha256": sha256_bytes(result.handoff_path.read_bytes()),
                "result": str(result.result_path),
                "result_sha256": sha256_bytes(result.result_path.read_bytes()),
                "hash_chain": handoff["hash_chain"],
                "consumer_permissions": handoff["consumer_permissions"],
                "evidence_class": "simulation",
                "adoption_mode": "proposal_only",
                "volvence_promotion_eligible": False,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    f"HANDED_OFF_FOR_EXTERNAL_REVIEW: {result.handoff_id}; "
                    f"handoff={result.handoff_path}; evidence=simulation"
                )
            return 0
        if args.command == "research-scan":
            result = scan_research_opportunities(
                config=config,
                failure_patterns_path=args.failure_patterns,
                registry_path=args.registry,
            )
            _print_research_scan_result(result, as_json=args.json)
            return 0
        if args.command == "research-demand-validate":
            demand = validate_research_demand(
                config=config,
                demand_path=args.demand,
            )
            payload = {
                "schema_version": "forge-research-demand-validation-result.v1",
                "state": "VALID",
                "demand_id": demand["demand_id"],
                "status": demand["status"],
                "owner": demand["owner"],
                "capability_axes": demand["capability_axes"],
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    f"VALID: demand={demand['demand_id']}; "
                    f"status={demand['status']}; owner={demand['owner']}"
                )
            return 0
        if args.command == "research-demand-seal":
            result = seal_research_demand(
                config=config,
                draft_path=args.draft,
                output_path=args.output,
            )
            payload = {
                "schema_version": "forge-research-demand-seal-result.v1",
                "demand_id": result.demand_id,
                "demand": str(result.demand_path),
                "demand_sha256": sha256_bytes(result.demand_path.read_bytes()),
                "reused": result.reused,
                "discovery_authorized": True,
                "topic_binding_authorized": False,
                "research_start_authorized": False,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    f"SEALED: demand={result.demand_id}; "
                    f"artifact={result.demand_path}; reused={result.reused}"
                )
            return 0
        if args.command == "research-discover":
            discovery_backend = _research_discovery_backend(args, parser=parser)
            result = discover_research_topics(
                config=config,
                demand_path=args.demand,
                backend=discovery_backend,
            )
            payload = {
                "schema_version": "forge-research-discovery-result.v1",
                "run_id": result.run_id,
                "run": str(result.run_path),
                "proposals": [str(path) for path in result.proposal_paths],
                "proposal_count": len(result.proposal_paths),
                "reused": result.reused,
                "research_request_created": False,
                "praxist_started": False,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    f"DISCOVERED: run={result.run_id}; "
                    f"topics={len(result.proposal_paths)}; reused={result.reused}"
                )
            return 0
        if args.command == "research-bind-topic":
            result = review_research_topic(
                config=config,
                demand_path=args.demand,
                proposal_path=args.proposal,
                mapping_id=args.mapping_id,
                registry_path=args.registry,
                reviewed_by=args.reviewed_by,
                reason=args.reason,
                decision="REJECT" if args.reject else "APPROVE",
            )
            payload = {
                "schema_version": "forge-research-demand-binding-result.v1",
                "binding_id": result.binding_id,
                "binding": str(result.binding_path),
                "binding_sha256": sha256_bytes(result.binding_path.read_bytes()),
                "decision": result.decision,
                "research_request_created": False,
                "praxist_started": False,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    f"{result.decision}: binding={result.binding_id}; "
                    f"artifact={result.binding_path}"
                )
            return 0
        if args.command == "research-submit-binding":
            result = submit_bound_topic_for_a0(
                config=config,
                binding_path=args.binding,
            )
            payload = {
                "schema_version": "forge-research-binding-submit-result.v1",
                "state": "AWAITING_RESEARCH_APPROVAL",
                "request_id": result.request_id,
                "request": str(result.request_path),
                "request_sha256": sha256_bytes(result.request_path.read_bytes()),
                "reused": result.reused,
                "praxist_started": False,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    f"AWAITING_RESEARCH_APPROVAL: {result.request_id}; "
                    f"request={result.request_path}"
                )
            return 0
        if args.command == "research-loop":
            result = run_demand_research_loop_once(
                config=config,
                backend=_research_discovery_backend(args, parser=parser),
                demand_root=args.demand_root,
                max_demands=args.max_demands,
                max_new_discoveries=args.max_new_discoveries,
                max_new_requests=args.max_new_requests,
                max_reconciles=args.max_reconciles,
            )
            payload = result.to_jsonable()
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                summary = payload["summary"]
                print(
                    "BOUNDED_ONCE: "
                    f"demands={summary['demand_count']}; "
                    f"new_discoveries={summary['new_discovery_count']}; "
                    f"new_requests={summary['new_request_count']}; "
                    f"awaiting_a0={summary['awaiting_a0_count']}; "
                    f"reconciled={summary['reconciled_count']}; "
                    f"blocked={summary['blocked_count']}"
                )
            return 2 if result.blocked_count else 0
        if args.command == "research-portfolio-seal":
            result = seal_research_portfolio(
                config=config,
                draft_path=args.draft,
                output_path=args.output,
            )
            payload = {
                "schema_version": "forge-research-portfolio-seal-result.v1",
                "portfolio_id": result.portfolio_id,
                "portfolio": str(result.portfolio_path),
                "portfolio_sha256": sha256_bytes(
                    result.portfolio_path.read_bytes()
                ),
                "reused": result.reused,
                "automatic_human_gates_authorized": False,
                "research_start_authorized": False,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    f"SEALED: portfolio={result.portfolio_id}; "
                    f"artifact={result.portfolio_path}; reused={result.reused}"
                )
            return 0
        if args.command == "research-portfolio-validate":
            portfolio = validate_research_portfolio(
                config=config,
                portfolio_path=args.portfolio,
            )
            payload = {
                "schema_version": "forge-research-portfolio-validation-result.v1",
                "state": "VALID",
                "portfolio_id": portfolio["portfolio_id"],
                "study_count": len(portfolio["studies"]),
                "max_active_runs_global": portfolio["scheduling"][
                    "max_active_runs_global"
                ],
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    f"VALID: portfolio={portfolio['portfolio_id']}; "
                    f"studies={len(portfolio['studies'])}"
                )
            return 0
        if args.command == "research-portfolio-status":
            status = inspect_research_portfolio(
                config=config,
                portfolio_path=args.portfolio,
            )
            payload = status.to_jsonable()
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                print(
                    f"PORTFOLIO: {status.portfolio_id}; studies={len(status.studies)}"
                )
                for study in status.studies:
                    print(
                        f"{study.state}: study={study.study_id}; "
                        f"request={study.request_id or '-'}; run={study.run_id or '-'}"
                    )
            return 0
        if args.command == "research-portfolio-loop":
            result = run_research_portfolio_once(
                config=config,
                portfolio_path=args.portfolio,
                backend=_research_discovery_backend(args, parser=parser),
                max_new_discoveries=args.max_new_discoveries,
                max_new_requests=args.max_new_requests,
                max_reconciles=args.max_reconciles,
            )
            payload = result.to_jsonable()
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                summary = payload["status"]["summary"]
                print(
                    "PORTFOLIO_BOUNDED_ONCE: "
                    f"eligible={len(result.eligible_study_ids)}; "
                    f"running={summary['running_count']}; "
                    f"awaiting_a0={summary['awaiting_a0_count']}; "
                    f"needs_task_design={summary['needs_task_design_count']}"
                )
            return 2 if result.loop.blocked_count else 0
        if args.command == "research-portfolio-review":
            result = review_research_study_outcome(
                config=config,
                portfolio_path=args.portfolio,
                study_id=args.study_id,
                request_path=args.request,
                evidence_paths=args.evidence,
                reviewed_by=args.reviewed_by,
                reason=args.reason,
                decision=args.decision,
            )
            payload = {
                "schema_version": "forge-research-study-outcome-result.v1",
                "outcome_id": result.outcome_id,
                "outcome": str(result.outcome_path),
                "decision": result.decision,
                "dependency_scheduling_only": True,
                "production_promotion_authorized": False,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    f"{result.decision}: outcome={result.outcome_id}; "
                    f"artifact={result.outcome_path}"
                )
            return 0
        if args.command == "research-inbox":
            statuses = list_research_inbox(config=config)
            _print_research_statuses(statuses, as_json=args.json)
            return 0
        if args.command == "research-approve":
            decision = "REJECT" if args.reject else "APPROVE"
            result = review_research_request(
                config=config,
                request_path=args.request,
                reviewed_by=args.approved_by,
                reason=args.reason,
                decision=decision,
                portfolio_path=args.portfolio,
                portfolio_study_id=args.study_id,
            )
            print(f"{result.decision}: {result.approval_id}; approval={result.approval_path}")
            return 0
        if args.command == "research-control":
            directive = issue_research_control_directive(
                config=config,
                request_path=args.request,
                action=args.action,
                expected_event_sha256=args.expected_event_sha256,
                requested_by=args.requested_by,
                reason=args.reason,
                grace_seconds=args.grace,
            )
            status = reconcile_research_control(
                config=config,
                request_path=args.request,
            )[0]
            payload = {
                "schema_version": "forge-research-control-result.v1",
                "directive_id": directive.directive_id,
                "directive_path": str(directive.directive_path),
                "action": directive.action,
                "status": _research_status_payload(status),
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(
                    f"{directive.action}: directive={directive.directive_id}; "
                    f"state={status.state}; run_id={status.run_id or '-'}"
                )
            return 2 if status.state in {"BLOCKED", "RUN_FAILED"} else 0
        if args.command == "research-reconcile":
            statuses = reconcile_research_control(
                config=config,
                request_path=args.request,
            )
            _print_research_statuses(statuses, as_json=args.json)
            return 2 if any(status.state in {"BLOCKED", "RUN_FAILED"} for status in statuses) else 0
        if args.command == "research-validate-task":
            task = validate_research_task(config=config, task_path=args.task_manifest)
            print(
                f"VALID: task={task['task_id']}; release={task['release']['mode']}; "
                f"initial_wiring={task['release']['initial_wiring']}"
            )
            return 0
        if args.command == "research-import-praxist":
            result = import_praxist_candidate(
                config=config,
                task_path=args.task_manifest,
                handoff_path=args.handoff,
                run_dir=args.run_dir,
                output_path=args.output,
            )
            print(f"SEALED: {result.candidate_id}; candidate={result.candidate_path}; wiring=disabled")
            return 0
        if args.command == "research-authorize":
            result = authorize_research_candidate(
                config=config,
                task_path=args.task_manifest,
                candidate_path=args.candidate,
                validation_path=args.validation,
                gate_path=args.gate,
                to_wiring=args.to_wiring,
                previous_receipt_path=args.previous_receipt,
                authorized_by=args.authorized_by,
                reason=args.reason,
                output_path=args.output,
            )
            print(
                f"{result.outcome}: {result.receipt_id}; "
                f"resulting_wiring={result.resulting_wiring}; receipt={result.receipt_path}"
            )
            return 0 if result.outcome == "AUTHORIZED" else 2
        if args.command == "research-rollback":
            result = rollback_research_candidate(
                config=config,
                previous_receipt_path=args.previous_receipt,
                to_wiring=args.to_wiring,
                authorized_by=args.authorized_by,
                reason=args.reason,
                output_path=args.output,
            )
            print(
                f"{result.outcome}: {result.receipt_id}; "
                f"resulting_wiring={result.resulting_wiring}; receipt={result.receipt_path}"
            )
            return 0
        parser.error(f"Unknown command: {args.command}")
    except (ForgeError, ForgeConfigError, BackendError) as exc:
        print(f"forge: {exc}", file=sys.stderr)
        return 2


def _add_backend_arguments(parser: argparse.ArgumentParser, *, default: str, allow_none: bool) -> None:
    choices = ("openai", "replay", "none") if allow_none else ("openai", "replay")
    parser.add_argument("--backend", choices=choices, default=default)
    parser.add_argument("--replay-responses", type=Path)


def _print_research_statuses(
    statuses: Sequence[ResearchControlStatus],
    *,
    as_json: bool,
) -> None:
    payloads = [_research_status_payload(status) for status in statuses]
    if as_json:
        print(json.dumps(payloads, indent=2, sort_keys=True))
        return
    if not payloads:
        print("No ResearchRequest artifacts found.")
        return
    for payload in payloads:
        print(
            f"{payload['state']}: task={payload['task_id']}; "
            f"request={payload['request_id']}; run_id={payload['run_id'] or '-'}; "
            f"request_path={payload['request_path']}"
        )


def _print_research_scan_result(result, *, as_json: bool) -> None:
    statuses = [_research_opportunity_status_payload(status) for status in result.statuses]
    payload = {
        "failure_patterns_path": str(result.failure_patterns_path),
        "registry_path": str(result.registry_path),
        "discovered_count": result.discovered_count,
        "new_request_count": result.new_request_count,
        "opportunities": statuses,
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(
        f"discovered={result.discovered_count}; new_requests={result.new_request_count}; "
        f"registry={result.registry_path}"
    )
    for status in statuses:
        print(
            f"{status['state']}: pattern={status['pattern_id']}; "
            f"priority={status['priority_score']}; mapping={status['mapping_id'] or '-'}; "
            f"opportunity={status['opportunity_path']}"
        )


def _research_opportunity_status_payload(
    status: ResearchOpportunityStatus,
) -> dict[str, object]:
    return {
        "opportunity_id": status.opportunity_id,
        "pattern_id": status.pattern_id,
        "priority_score": status.priority_score,
        "state": status.state,
        "blocker_codes": list(status.blocker_codes),
        "opportunity_path": str(status.opportunity_path),
        "routing_path": str(status.routing_path),
        "mapping_id": status.mapping_id,
        "request_id": status.request_id,
        "request_path": str(status.request_path) if status.request_path else None,
    }


def _research_status_payload(status: ResearchControlStatus) -> dict[str, str | None]:
    return {
        "request_id": status.request_id,
        "task_id": status.task_id,
        "state": status.state,
        "request_path": str(status.request_path),
        "approval_path": str(status.approval_path) if status.approval_path else None,
        "latest_event_path": str(status.latest_event_path) if status.latest_event_path else None,
        "latest_event_sha256": status.latest_event_sha256,
        "run_id": status.run_id,
        "run_dir": status.run_dir,
        "monitor_command": status.monitor_command,
    }


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


def _research_discovery_backend(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
) -> ReplayResearchDiscoveryBackend | CodexNativeResearchDiscoveryBackend:
    if args.backend == "replay":
        if args.replay_responses is None:
            parser.error(f"{args.command} --backend replay requires --replay-responses")
        return ReplayResearchDiscoveryBackend.from_path(args.replay_responses)
    if not args.model:
        parser.error(
            f"{args.command} --backend codex_sdk requires --model or FORGE_DISCOVERY_MODEL"
        )
    return CodexNativeResearchDiscoveryBackend(
        model_name=args.model,
        codex_bin=args.codex_bin,
    )


if __name__ == "__main__":
    raise SystemExit(main())
