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
    list_research_inbox,
    record_external_research_handoff,
    reconcile_research_control,
    review_research_request,
    submit_external_research_request,
    submit_research_request,
    validate_external_research_descriptor,
)
from .research_opportunity import (
    ResearchOpportunityStatus,
    scan_research_opportunities,
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
            payload = {
                "schema_version": "forge-external-research-handoff-result.v1",
                "state": "HANDED_OFF_FOR_EXTERNAL_REVIEW",
                "handoff_id": result.handoff_id,
                "handoff": str(result.handoff_path),
                "handoff_sha256": sha256_bytes(result.handoff_path.read_bytes()),
                "result": str(result.result_path),
                "result_sha256": sha256_bytes(result.result_path.read_bytes()),
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
            )
            print(f"{result.decision}: {result.approval_id}; approval={result.approval_path}")
            return 0
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


if __name__ == "__main__":
    raise SystemExit(main())
