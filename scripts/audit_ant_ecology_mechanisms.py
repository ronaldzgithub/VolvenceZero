"""Run the P0 digital-ant ecology mechanism audit."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from volvence_ant.experiments.ecology_mechanism_audit import (
    EcologyMechanismAuditConfig,
    run_ecology_mechanism_audit,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_REPORT = Path(
    "research/ant/results/ecology_recovery/p0/ecology_mechanism_audit.v2.json"
)


def _repo_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else _REPO_ROOT / path
    resolved.relative_to(_REPO_ROOT)
    return resolved


async def _run(args: argparse.Namespace) -> int:
    report = await run_ecology_mechanism_audit(
        EcologyMechanismAuditConfig(
            n_ants=args.n_ants,
            temporal_latent_dim=args.temporal_latent_dim,
            episode_rounds=args.episode_rounds,
            episodes_per_stage=args.episodes_per_stage,
            evaluation_rounds=args.evaluation_rounds,
            seed=args.seed,
        )
    )
    output = _repo_path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(report.description)
    print(f"report: {output.relative_to(_REPO_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit digital-ant ecology action, temporal, and freeze chains"
        )
    )
    parser.add_argument("--n-ants", type=int, default=4)
    parser.add_argument("--temporal-latent-dim", type=int, default=16)
    parser.add_argument("--episode-rounds", type=int, default=12)
    parser.add_argument("--episodes-per-stage", type=int, default=3)
    parser.add_argument("--evaluation-rounds", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT)
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
