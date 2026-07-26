"""Run the P2 formal confirmatory ecology matrix.

Three sub-commands mirror the three batches of the plan:

* ``preflight``  -- P2-A: one training seed, full stack, timing/size/determinism.
* ``shard``      -- P2-B/P2-C: one ``(training_seed, arm)`` cell, resumable.
* ``aggregate``  -- P2.4: fold complete shards into the promotion verdict.

Every sub-command demands a P1 report whose verdict is ``PASS``; without it the
run exits non-zero before spending any budget.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from volvence_ant.evidence.provenance import collect_ant_provenance
from volvence_ant.experiments.ecology_p2 import (
    ECOLOGY_P2_ABLATION_ARM_NAMES,
    ECOLOGY_P2_ARM_NAMES,
    ECOLOGY_P2_CORE_ARM_NAMES,
    EcologyP2Config,
    EcologyP2PrerequisiteError,
    EcologyP2ProgressPaused,
    aggregate_ecology_p2_shards,
    preregistration_digest,
    run_ecology_p2_preflight,
    run_ecology_p2_shard,
    shard_report_from_dict,
)


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_P1_REPORT = Path(
    "research/ant/results/ecology_recovery/p1/ecology_p1.seed0.json"
)
_DEFAULT_OUTPUT_DIR = Path("research/ant/results/ecology_recovery/p2")


def _config(args: argparse.Namespace) -> EcologyP2Config:
    return EcologyP2Config(
        n_ants=args.n_ants,
        temporal_latent_dim=args.temporal_latent_dim,
        training_rounds=args.training_rounds,
        validation_rounds=args.validation_rounds,
        heldout_rounds=args.heldout_rounds,
        layouts_per_tier=args.layouts_per_tier,
        training_seeds=tuple(sorted(args.training_seeds)),
        device=args.device,
    )


def _resolve(path: Path) -> Path:
    resolved = path if path.is_absolute() else _ROOT / path
    resolved.relative_to(_ROOT)
    return resolved


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    output = _resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return output


async def _run_preflight(args: argparse.Namespace) -> int:
    config = _config(args)
    report = await run_ecology_p2_preflight(
        config,
        training_seed=args.training_seed,
        p1_report_path=args.p1_report,
        repo_root=_ROOT,
        progress_dir=(
            _resolve(args.progress_dir) if args.progress_dir else None
        ),
        arms=tuple(args.arms) if args.arms else ECOLOGY_P2_CORE_ARM_NAMES,
    )
    output = _write_json(
        args.report or _DEFAULT_OUTPUT_DIR / "ecology_p2.preflight.json",
        report.to_dict(),
    )
    print(report.description)
    print(f"report: {output.relative_to(_ROOT)}")
    return 0 if report.passed else 1


async def _run_shard(args: argparse.Namespace) -> int:
    config = _config(args)
    try:
        report = await run_ecology_p2_shard(
            config,
            training_seed=args.training_seed,
            arm=args.arm,
            p1_report_path=args.p1_report,
            repo_root=_ROOT,
            progress_dir=(
                _resolve(args.progress_dir) if args.progress_dir else None
            ),
            max_new_work_items=args.max_new_work_items,
        )
    except EcologyP2ProgressPaused as paused:
        print(str(paused))
        return 0
    output = _write_json(
        args.report
        or _DEFAULT_OUTPUT_DIR
        / "shards"
        / f"ecology_p2.seed{args.training_seed}.{args.arm}.json",
        report.to_dict(),
    )
    print(report.description)
    print(f"shard: {output.relative_to(_ROOT)}")
    return 0


def _run_aggregate(args: argparse.Namespace) -> int:
    config = _config(args)
    shard_dir = _resolve(args.shard_dir)
    paths = sorted(shard_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"no P2 shard reports under {shard_dir}")
    shards = tuple(
        shard_report_from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        for path in paths
    )
    provenance = collect_ant_provenance(
        repo_root=_ROOT,
        seeds=config.training_seeds,
        config=asdict(config),
        model_fingerprint=preregistration_digest(config),
    )
    report = aggregate_ecology_p2_shards(
        shards,
        worktree_clean=not provenance.working_tree_dirty,
        config=config,
    )
    payload = report.to_dict()
    payload["provenance"] = asdict(provenance)
    output = _write_json(
        args.report or _DEFAULT_OUTPUT_DIR / "ecology_p2.confirmatory.json",
        payload,
    )
    print(report.description)
    for endpoint in report.primary_endpoints:
        status = "PASS" if endpoint.passed else "BLOCK"
        print(f"  [{status}] {endpoint.name}")
    print(f"report: {output.relative_to(_ROOT)}")
    return 0 if report.verdict == "PASS" else 1


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-ants", type=int, default=8)
    parser.add_argument("--temporal-latent-dim", type=int, default=16)
    parser.add_argument("--training-rounds", type=int, default=80)
    parser.add_argument("--validation-rounds", type=int, default=80)
    parser.add_argument("--heldout-rounds", type=int, default=120)
    parser.add_argument("--layouts-per-tier", type=int, default=5)
    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--report", type=Path, default=None)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the ant ecology P2 confirmatory matrix"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="P2-A full-stack rehearsal")
    _add_common(preflight)
    preflight.add_argument("--training-seed", type=int, default=None)
    preflight.add_argument("--p1-report", type=Path, default=_DEFAULT_P1_REPORT)
    preflight.add_argument("--progress-dir", type=Path, default=None)
    preflight.add_argument(
        "--arms",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Arms to rehearse; defaults to the P2-B core matrix "
            f"({', '.join(ECOLOGY_P2_CORE_ARM_NAMES)})"
        ),
    )

    shard = sub.add_parser("shard", help="one (training_seed, arm) shard")
    _add_common(shard)
    shard.add_argument("--training-seed", type=int, required=True)
    shard.add_argument(
        "--arm",
        type=str,
        required=True,
        choices=ECOLOGY_P2_ARM_NAMES,
        help=(
            "core matrix: "
            f"{', '.join(ECOLOGY_P2_CORE_ARM_NAMES)}; ablations: "
            f"{', '.join(ECOLOGY_P2_ABLATION_ARM_NAMES)}"
        ),
    )
    shard.add_argument("--p1-report", type=Path, default=_DEFAULT_P1_REPORT)
    shard.add_argument("--progress-dir", type=Path, default=None)
    shard.add_argument(
        "--max-new-work-items",
        type=int,
        default=None,
        help=(
            "Stop cleanly after this many newly committed training episodes "
            "or held-out layouts; requires --progress-dir."
        ),
    )

    aggregate = sub.add_parser("aggregate", help="fold shards into a verdict")
    _add_common(aggregate)
    aggregate.add_argument(
        "--shard-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR / "shards",
    )

    args = parser.parse_args()
    try:
        if args.command == "preflight":
            return asyncio.run(_run_preflight(args))
        if args.command == "shard":
            return asyncio.run(_run_shard(args))
        if args.command == "aggregate":
            return _run_aggregate(args)
    except EcologyP2PrerequisiteError as error:
        print(f"P2 blocked by the serial constraint: {error}")
        return 2
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
