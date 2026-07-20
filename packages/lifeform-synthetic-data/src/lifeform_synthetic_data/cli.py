"""Command-line entry point for corpus generation and validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from .conformance import assert_v1_conformance
from .audit import audit_run_streaming, write_audit_bundle
from .contracts import CorpusSplit, ExperienceTrajectory, GenerationTier, ScenarioBlueprint
from .cursor_renderer import (
    DEFAULT_ASSET_BUNDLE,
    CursorAuthoredJsonClient,
    bundled_asset_root,
    validate_cursor_asset_novelty,
    validate_cursor_assets,
)
from .llm import (
    JsonCompletionClient,
    OpenAICompatibleConfig,
    OpenAICompatibleJsonClient,
    RateCard,
)
from .pipeline import CorpusGenerationPipeline, GenerationRunConfig
from .projections import (
    load_master_run,
    project_master_run,
)
from .scenario import load_unified_v1_blueprints, validate_unified_v1_package
from .schema import export_json_schema

_STAGES: dict[
    str,
    tuple[GenerationTier, tuple[tuple[CorpusSplit, int], ...]],
] = {
    "golden96": (
        GenerationTier.STRUCTURAL,
        (
            (CorpusSplit.TRAIN, 1),
            (CorpusSplit.VAL, 1),
            (CorpusSplit.TEST, 1),
        ),
    ),
    "pilot96": (
        GenerationTier.RENDERED,
        (
            (CorpusSplit.TRAIN, 1),
            (CorpusSplit.VAL, 1),
            (CorpusSplit.TEST, 1),
        ),
    ),
    "scale768": (
        GenerationTier.RENDERED,
        (
            (CorpusSplit.TRAIN, 8),
            (CorpusSplit.VAL, 8),
            (CorpusSplit.TEST, 8),
        ),
    ),
    "master10240": (
        GenerationTier.RENDERED,
        (
            (CorpusSplit.TRAIN, 128),
            (CorpusSplit.VAL, 64),
            (CorpusSplit.TEST, 64),
        ),
    ),
    "master50000": (
        GenerationTier.RENDERED,
        (
            (CorpusSplit.TRAIN, 625),
            (CorpusSplit.VAL, 313),
            (CorpusSplit.TEST, 312),
        ),
    ),
    "live1024": (
        GenerationTier.LIVE_THROUGH,
        (
            (CorpusSplit.TRAIN, 12),
            (CorpusSplit.VAL, 8),
            (CorpusSplit.TEST, 8),
        ),
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lifeform-synthetic-data",
        description="Unified synthetic experience corpus v1",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "validate-scenarios",
        help="validate the checked-in 96-scenario package",
    )
    render_assets_parser = subparsers.add_parser(
        "validate-render-assets",
        help="validate all Cursor-authored dialogue variants",
    )
    render_assets_parser.add_argument(
        "--cursor-asset-bundle",
        default=DEFAULT_ASSET_BUNDLE,
    )
    schema_parser = subparsers.add_parser(
        "export-schema",
        help="export the frozen JSON Schema",
    )
    schema_parser.add_argument("--output", type=Path, required=True)
    subparsers.add_parser(
        "conformance",
        help="run the strict v1 round-trip conformance fixture",
    )
    project_parser = subparsers.add_parser(
        "project",
        help="derive all task views from a completed master run",
    )
    project_parser.add_argument("--run-root", type=Path, required=True)
    project_parser.add_argument("--output-root", type=Path)
    project_parser.add_argument(
        "--human-review-sample-rate",
        type=float,
        default=0.05,
    )
    audit_parser = subparsers.add_parser(
        "audit",
        help="run hard quality gates and write the delivery audit bundle",
    )
    audit_parser.add_argument("--run-root", type=Path, required=True)
    audit_parser.add_argument("--expected-count", type=int)
    audit_parser.add_argument("--output-root", type=Path)

    for command in ("estimate", "generate"):
        generation = subparsers.add_parser(
            command,
            help=f"{command} a staged corpus run",
        )
        _add_generation_arguments(generation)
    return parser


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", choices=sorted(_STAGES), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/synthetic/unified_v1"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--created-at")
    parser.add_argument("--git-sha")
    parser.add_argument("--base-seed", type=int, default=17072026)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-cost-usd", type=float, default=0.0)
    parser.add_argument(
        "--renderer",
        choices=("openai", "cursor"),
        default="openai",
    )
    parser.add_argument("--cursor-asset-bundle")
    parser.add_argument("--endpoint-config", type=Path)
    parser.add_argument("--source-run-root", type=Path)
    parser.add_argument("--export-parquet", action="store_true")
    parser.add_argument("--resume", action="store_true")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-scenarios":
        report = validate_unified_v1_package()
        _print_json(
            {
                "package_name": report.package_name,
                "package_hash": report.package_hash,
                "scene_count": report.scene_count,
                "family_count": report.family_count,
                "split_counts": dict(report.split_counts),
                "routing_test_count": report.routing_test_count,
                "negative_routing_test_count": report.negative_routing_test_count,
                "semantic_coherence_count": report.semantic_coherence_count,
            }
        )
        return 0
    if args.command == "validate-render-assets":
        asset_root = bundled_asset_root(args.cursor_asset_bundle)
        report = validate_cursor_assets(
            load_unified_v1_blueprints(),
            root=asset_root,
        )
        novelty = (
            validate_cursor_asset_novelty(candidate_root=asset_root)
            if args.cursor_asset_bundle != DEFAULT_ASSET_BUNDLE
            else None
        )
        _print_json(
            {
                "asset_bundle": args.cursor_asset_bundle,
                "asset_hash": report.asset_hash,
                "family_count": report.family_count,
                "scenario_count": report.scenario_count,
                "turn_count": report.turn_count,
                "variant_count": report.variant_count,
                "normalized_overlap_with_unified_v1": (
                    novelty.normalized_overlap_count
                    if novelty is not None
                    else 0
                ),
                "passed": True,
            }
        )
        return 0
    if args.command == "export-schema":
        export_json_schema(args.output)
        _print_json({"output": str(args.output.resolve())})
        return 0
    if args.command == "conformance":
        _print_json({"conformance_hash": assert_v1_conformance(), "passed": True})
        return 0
    if args.command == "project":
        manifests = project_master_run(
            args.run_root,
            output_root=args.output_root,
            human_review_sample_rate=args.human_review_sample_rate,
        )
        projection_root = args.output_root or args.run_root / "projections"
        relationship_manifest = projection_root / "relationship_encoder_dataset" / "split-manifest.json"
        _print_json(
            {
                "projection_root": str(projection_root.resolve()),
                "views": {manifest.view.value: manifest.record_count for manifest in manifests},
                "relationship_split_manifest": str(relationship_manifest.resolve()),
            }
        )
        return 0
    if args.command == "audit":
        report, model_counts = audit_run_streaming(
            args.run_root,
            expected_count=args.expected_count,
        )
        output_root = args.output_root or args.run_root / "audit"
        manifest_payload = _load_json_object(args.run_root / "run-manifest.json")
        cost_usd, prompt_tokens, completion_tokens = _journal_usage(args.run_root)
        write_audit_bundle(
            report,
            (),
            output_dir=output_root,
            actual_cost_usd=cost_usd,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_distribution=model_counts,
        )
        _print_json(
            {
                "audit_root": str(output_root.resolve()),
                "passed": report.passed,
                "hard_failure_count": report.hard_failure_count,
                "trajectory_count": report.trajectory_count,
                "manifest_trajectory_count": manifest_payload.get("trajectory_count"),
            }
        )
        return 0 if report.passed else 4
    if args.command in {"estimate", "generate"}:
        return _run_generation_command(args)
    raise AssertionError(f"unhandled command {args.command!r}")


def _run_generation_command(args: argparse.Namespace) -> int:
    tier, split_replicates = _STAGES[args.stage]
    created_at = args.created_at or datetime.now(UTC).isoformat()
    git_sha = args.git_sha or _resolve_git_sha()
    base_seed = args.base_seed
    run_config_path = args.output_root / args.run_id / "run-config.json"
    if args.resume:
        if not run_config_path.is_file():
            raise FileNotFoundError(f"--resume requested but run config is missing: {run_config_path}")
        existing = _load_json_object(run_config_path)
        created_at = _required_string(existing, "created_at")
        git_sha = _required_string(existing, "git_sha")
        base_seed_value = existing.get("base_seed")
        if type(base_seed_value) is not int:
            raise ValueError("resume run config base_seed must be an integer")
        base_seed = base_seed_value
    clients: tuple[JsonCompletionClient, ...] = ()
    rate_card: RateCard | None = None
    source_trajectories: tuple[ExperienceTrajectory, ...] = ()
    structural_enricher: Callable[[ExperienceTrajectory], ExperienceTrajectory] | None = None
    cursor_asset_bundle_id: str | None = None
    cursor_asset_hash: str | None = None
    if tier is GenerationTier.RENDERED:
        if args.source_run_root is not None:
            raise ValueError("--source-run-root is only valid for live-through")
        if args.renderer == "cursor":
            if args.endpoint_config is not None:
                raise ValueError("Cursor-authored rendering does not accept --endpoint-config")
            asset_bundle = args.cursor_asset_bundle or DEFAULT_ASSET_BUNDLE
            asset_root = bundled_asset_root(asset_bundle)
            asset_report = validate_cursor_assets(
                load_unified_v1_blueprints(),
                root=asset_root,
            )
            if asset_bundle != DEFAULT_ASSET_BUNDLE:
                validate_cursor_asset_novelty(candidate_root=asset_root)
                cursor_asset_bundle_id = asset_bundle
                cursor_asset_hash = asset_report.asset_hash
            cursor_client = CursorAuthoredJsonClient(
                asset_root=asset_root,
                asset_bundle=asset_bundle,
            )
            clients = (cursor_client,)
            structural_enricher = cursor_client.enrich_truth
            rate_card = RateCard(0.0, 0.0)
        else:
            if args.cursor_asset_bundle is not None:
                raise ValueError("--cursor-asset-bundle requires --renderer cursor")
            if args.endpoint_config is None:
                raise ValueError("OpenAI-compatible rendered stages require --endpoint-config")
            clients, rate_card = _load_endpoint_clients(
                args.endpoint_config,
                max_output_tokens=args.max_output_tokens,
            )
            if args.max_cost_usd <= 0:
                raise ValueError("OpenAI-compatible rendered stages require a positive --max-cost-usd hard limit")
    else:
        if args.cursor_asset_bundle is not None:
            raise ValueError("--cursor-asset-bundle is only valid for rendered Cursor stages")
        if args.endpoint_config is not None:
            raise ValueError("--endpoint-config is only valid for rendered stages")
        if tier is GenerationTier.LIVE_THROUGH:
            if args.source_run_root is None:
                raise ValueError("live-through requires --source-run-root pointing to a completed rendered master run")
        elif args.source_run_root is not None:
            raise ValueError("--source-run-root is only valid for live-through")

    package_report = validate_unified_v1_package()
    blueprints = load_unified_v1_blueprints()
    if tier is GenerationTier.LIVE_THROUGH:
        source_trajectories = _select_live_sample(
            load_master_run(args.source_run_root),
            blueprints=blueprints,
            split_replicates=split_replicates,
        )
    config = GenerationRunConfig(
        output_root=args.output_root,
        run_id=args.run_id,
        created_at=created_at,
        git_sha=git_sha,
        generation_tier=tier,
        base_seed=base_seed,
        split_replicates=split_replicates,
        shard_size=args.shard_size,
        concurrency=args.concurrency,
        max_cost_usd=args.max_cost_usd,
        max_output_tokens=args.max_output_tokens,
        export_parquet=args.export_parquet,
        cursor_asset_bundle=cursor_asset_bundle_id,
        cursor_asset_hash=cursor_asset_hash,
    )
    pipeline = CorpusGenerationPipeline(
        config=config,
        blueprints=blueprints,
        clients=clients,
        rate_card=rate_card,
        scenario_package_hash=package_report.package_hash,
        source_trajectories=source_trajectories,
        structural_enricher=structural_enricher,
    )
    estimate = pipeline.estimate_cost()
    _print_json(
        {
            "stage": args.stage,
            "tier": tier.value,
            "renderer": (args.renderer if tier is GenerationTier.RENDERED else "not_applicable"),
            "cursor_asset_bundle": (
                (args.cursor_asset_bundle or DEFAULT_ASSET_BUNDLE)
                if tier is GenerationTier.RENDERED
                and args.renderer == "cursor"
                else None
            ),
            "planned_count": len(pipeline.plan_jobs()),
            "pending_calls": estimate.pending_calls,
            "conservative_upper_bound_usd": round(
                estimate.conservative_upper_bound_usd,
                6,
            ),
            "max_cost_usd": estimate.max_cost_usd,
            "within_budget": estimate.within_budget,
        }
    )
    if args.command == "estimate":
        return 0 if estimate.within_budget else 2
    result = pipeline.run()
    _print_json(
        {
            "run_root": str(result.run_root.resolve()),
            "planned_count": result.planned_count,
            "completed_count": result.completed_count,
            "resumed_count": result.resumed_count,
            "quarantined_count": result.quarantined_count,
            "shard_count": len(result.shard_paths),
            "manifest_path": str(result.manifest_path.resolve()),
            "actual_cost_usd": round(result.actual_cost_usd, 6),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        }
    )
    return 0 if result.quarantined_count == 0 else 3


def _select_live_sample(
    trajectories: tuple[ExperienceTrajectory, ...],
    *,
    blueprints: tuple[ScenarioBlueprint, ...],
    split_replicates: tuple[tuple[CorpusSplit, int], ...],
) -> tuple[ExperienceTrajectory, ...]:
    by_scenario: dict[str, list[ExperienceTrajectory]] = {blueprint.scenario_id: [] for blueprint in blueprints}
    for trajectory in trajectories:
        if trajectory.generation_tier is not GenerationTier.RENDERED:
            raise ValueError("live-through source run must contain only rendered trajectories")
        if trajectory.scenario_ref not in by_scenario:
            raise ValueError(f"live-through source references unknown scenario: {trajectory.scenario_ref}")
        by_scenario[trajectory.scenario_ref].append(trajectory)

    replicates = dict(split_replicates)
    selected: list[ExperienceTrajectory] = []
    for blueprint in sorted(blueprints, key=lambda item: item.scenario_id):
        required = replicates.get(blueprint.split, 0)
        candidates = by_scenario[blueprint.scenario_id]
        if len(candidates) < required:
            raise ValueError(
                f"scenario {blueprint.scenario_id!r} has {len(candidates)} rendered sources, requires {required}"
            )
        selected.extend(
            sorted(
                candidates,
                key=lambda item: (
                    hashlib.sha256(item.trajectory_id.encode("utf-8")).digest(),
                    item.trajectory_id,
                ),
            )[:required]
        )
    return tuple(selected)


def _load_endpoint_clients(
    path: Path,
    *,
    max_output_tokens: int,
) -> tuple[tuple[OpenAICompatibleJsonClient, ...], RateCard]:
    payload = _load_json_object(path)
    if set(payload) != {"rate_card", "endpoints"}:
        raise ValueError("endpoint config must contain exactly rate_card and endpoints")
    raw_rate = payload["rate_card"]
    if not isinstance(raw_rate, dict):
        raise TypeError("endpoint config rate_card must be an object")
    if set(raw_rate) != {
        "input_usd_per_million",
        "output_usd_per_million",
    }:
        raise ValueError("endpoint config rate_card fields are invalid")
    input_rate = raw_rate["input_usd_per_million"]
    output_rate = raw_rate["output_usd_per_million"]
    if type(input_rate) not in {int, float} or type(output_rate) not in {
        int,
        float,
    }:
        raise TypeError("endpoint rate-card values must be numbers")
    rate_card = RateCard(float(input_rate), float(output_rate))
    raw_endpoints = payload["endpoints"]
    if not isinstance(raw_endpoints, list) or not raw_endpoints:
        raise ValueError("endpoint config endpoints must be a non-empty array")
    clients: list[OpenAICompatibleJsonClient] = []
    for index, raw_endpoint in enumerate(raw_endpoints):
        if not isinstance(raw_endpoint, dict):
            raise TypeError(f"endpoints[{index}] must be an object")
        allowed = {
            "base_url",
            "api_key_env",
            "model_id",
            "timeout_seconds",
            "max_attempts",
            "initial_backoff_seconds",
        }
        unknown = set(raw_endpoint) - allowed
        required = {"base_url", "api_key_env", "model_id"}
        missing = required - set(raw_endpoint)
        if unknown or missing:
            raise ValueError(f"endpoints[{index}] missing={sorted(missing)} unknown={sorted(unknown)}")
        api_key_env = _required_string(raw_endpoint, "api_key_env")
        api_key = os.environ.get(api_key_env)
        if api_key is None or not api_key.strip():
            raise ValueError(f"endpoint credential environment variable is missing: {api_key_env}")
        clients.append(
            OpenAICompatibleJsonClient(
                OpenAICompatibleConfig(
                    base_url=_required_string(raw_endpoint, "base_url"),
                    api_key=api_key,
                    model_id=_required_string(raw_endpoint, "model_id"),
                    rate_card=rate_card,
                    timeout_seconds=float(raw_endpoint.get("timeout_seconds", 90.0)),
                    max_output_tokens=max_output_tokens,
                    max_attempts=int(raw_endpoint.get("max_attempts", 4)),
                    initial_backoff_seconds=float(raw_endpoint.get("initial_backoff_seconds", 1.0)),
                )
            )
        )
    return tuple(clients), rate_card


def _resolve_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    sha = result.stdout.strip()
    if not sha:
        raise RuntimeError("git rev-parse returned an empty SHA")
    return sha


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON file: {path}") from error
    if not isinstance(decoded, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return decoded


def _journal_usage(run_root: Path) -> tuple[float, int, int]:
    journal_path = run_root / "journal.jsonl"
    if not journal_path.is_file():
        raise FileNotFoundError(journal_path)
    cost_usd = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    with journal_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid journal JSON at line {line_number}") from error
            if not isinstance(event, dict) or event.get("event") != "completed":
                continue
            raw_cost = event.get("cost_usd")
            raw_prompt = event.get("prompt_tokens")
            raw_completion = event.get("completion_tokens")
            if type(raw_cost) not in {int, float} or type(raw_prompt) is not int or type(raw_completion) is not int:
                raise ValueError(f"invalid completed usage at journal line {line_number}")
            cost_usd += float(raw_cost)
            prompt_tokens += raw_prompt
            completion_tokens += raw_completion
    return cost_usd, prompt_tokens, completion_tokens


def _required_string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _print_json(value: dict[str, object]) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
