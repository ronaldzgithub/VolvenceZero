"""Run one frozen-schedule P1 ecology development repetition."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from volvence_ant.experiments.ecology_p1 import (
    EcologyP1Config,
    run_ecology_p1,
    run_ecology_p1_diagnostics,
)


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REPORT = Path(
    "research/ant/results/ecology_recovery/p1/ecology_p1.seed0.json"
)
_DEFAULT_DIAGNOSTIC_REPORT = Path(
    "research/ant/results/ecology_recovery/p1/"
    "ecology_p1.diagnostics.seed0.json"
)


async def _run(args: argparse.Namespace) -> int:
    config = EcologyP1Config(
        n_ants=args.n_ants,
        temporal_latent_dim=args.temporal_latent_dim,
        training_rounds=args.training_rounds,
        evaluation_rounds=args.evaluation_rounds,
        layouts_per_tier=args.layouts_per_tier,
        seed=args.seed,
    )
    report = (
        run_ecology_p1_diagnostics(config)
        if args.diagnostics_only
        else await run_ecology_p1(config)
    )
    requested_report = (
        _DEFAULT_DIAGNOSTIC_REPORT
        if args.diagnostics_only and args.report == _DEFAULT_REPORT
        else args.report
    )
    output = (
        requested_report
        if requested_report.is_absolute()
        else _ROOT / requested_report
    )
    output.relative_to(_ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"diagnostics passed={report.passed}"
        if args.diagnostics_only
        else report.description
    )
    print(f"report: {output.relative_to(_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ant ecology P1 matrix")
    parser.add_argument("--n-ants", type=int, default=4)
    parser.add_argument("--temporal-latent-dim", type=int, default=16)
    parser.add_argument("--training-rounds", type=int, default=24)
    parser.add_argument("--evaluation-rounds", type=int, default=40)
    parser.add_argument("--layouts-per-tier", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=_DEFAULT_REPORT,
    )
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
