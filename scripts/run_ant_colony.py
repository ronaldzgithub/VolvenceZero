"""Phase 1 colony foraging -> research/ant/results/phase1_colony.json.

    python scripts/run_ant_colony.py [--n-ants 20] [--rounds 700] [--seed 0]

Compares a colony that shares the pheromone snapshot bus against an identical
colony with no bus. When scouts find food early, a trail corridor self-organises
and recruitment lifts delivery; the run reports the delivery curves so the lift
is visible over time.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from pathlib import Path

from volvence_ant.evidence import collect_ant_provenance, write_ant_artifact_bundle
from volvence_ant.experiments import (
    colony_foraging_experiment,
    kernel_colony_foraging_experiment,
)
from volvence_ant.runtime import AntSessionConfig

_RESULTS_DIR = Path("research/ant/results")
_REPO_ROOT = Path(__file__).resolve().parents[1]


async def main(*, n_ants: int, rounds: int, seeds: tuple[int, ...]) -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    fixed_baselines = []
    for seed in seeds:
        reports.append(
            await kernel_colony_foraging_experiment(
                n_ants=n_ants,
                rounds=rounds,
                seed=seed,
                session_config=AntSessionConfig(
                    temporal_latent_dim=16,
                    session_id=f"formal-kernel-colony:{seed}",
                    seed=seed,
                ),
            )
        )
        fixed_baselines.append(
            colony_foraging_experiment(
                n_ants=n_ants,
                rounds=rounds,
                seed=seed,
            )
        )
    payload = {
        "artifact_kind": "digital-ant-kernel-colony",
        "experiment": "phase1_colony_foraging",
        "n_ants": n_ants,
        "rounds": rounds,
        "seeds": seeds,
        "kernel_reports": [asdict(report) for report in reports],
        "learned_bus_effects": [report.learned_bus_effect for report in reports],
        "fixed_rule_baselines": [asdict(result) for result in fixed_baselines],
        "verdict": (
            "PASS"
            if all(report.learned_bus_effect > 0 for report in reports)
            else "BLOCK"
        ),
    }
    manifest = write_ant_artifact_bundle(
        artifact_path=_RESULTS_DIR / "phase1_colony.json",
        payload=payload,
        provenance=collect_ant_provenance(
            repo_root=_REPO_ROOT,
            seeds=seeds,
            config={"n_ants": n_ants, "rounds": rounds},
        ),
        repo_root=_REPO_ROOT,
    )
    print(f"[phase1] manifest={manifest}; verdict={payload['verdict']}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-ants", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            main(
                n_ants=args.n_ants,
                rounds=args.rounds,
                seeds=tuple(int(value) for value in args.seeds.split(",")),
            )
        )
    )
