"""Phase 0 digital-ant benchmarks -> research/ant/results/*.json.

Runs the two Phase 0 experiments with published biological ground truth:

    python scripts/run_ant_phase0.py

- path-integration homing precision (AntBot ~0.5% of journey length), and
- route familiarity / reducible novelty decay (Ardin 2016, tens of exposures).

The homing lane is fast (frozen navigator only). The route lane uses the real
kernel and is the slow part; tune --exposures / --route-length for budget.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from pathlib import Path

from volvence_ant.evidence import collect_ant_provenance, write_ant_artifact_bundle
from volvence_ant.experiments import (
    homing_precision_experiment,
    route_learning_experiment,
)

_RESULTS_DIR = Path("research/ant/results")
_REFERENCE_DIR = Path("research/ant/reference_data")
_REPO_ROOT = Path(__file__).resolve().parents[1]


async def main(*, exposures: int, route_length: int, n_trials: int, seed: int) -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    homing = homing_precision_experiment(n_trials=n_trials, seed=seed)
    homing_payload = {
        "experiment": "phase0_homing_precision",
        "antbot_reference_ratio": homing.antbot_reference_ratio,
        "passes_antbot_scale": homing.passes_antbot_scale,
        "description": homing.description,
        "curve": [asdict(point) for point in homing.curve],
    }
    homing_manifest = write_ant_artifact_bundle(
        artifact_path=_RESULTS_DIR / "phase0_homing.json",
        payload={"artifact_kind": "phase0-physical-homing", **homing_payload},
        provenance=collect_ant_provenance(
            repo_root=_REPO_ROOT,
            seeds=(seed,),
            config={"n_trials": n_trials},
        ),
        input_paths=(
            _REFERENCE_DIR / "antbot_homing_2019.csv",
            _REFERENCE_DIR / "REFERENCE_METADATA.json",
        ),
        repo_root=_REPO_ROOT,
    )
    print(f"[phase0] {homing.description}")

    route = await route_learning_experiment(
        exposures=exposures, route_length=route_length, seed=seed
    )
    route_payload = {
        "experiment": "phase0_route_learning",
        "description": route.description,
        "exposures": route.exposures,
        "route_length": route.route_length,
        "familiarity_improved": route.familiarity_improved,
        "first_exposure_novelty": route.first_exposure_novelty,
        "last_exposure_novelty": route.last_exposure_novelty,
        "novelty_by_exposure": list(route.novelty_by_exposure),
        "pe_by_exposure": list(route.pe_by_exposure),
        "novel_route_novelty": route.novel_route_novelty,
        "shuffled_route_novelty": route.shuffled_route_novelty,
        "memory_off_last_novelty": route.memory_off_last_novelty,
        "pe_off_last_novelty": route.pe_off_last_novelty,
    }
    route_manifest = write_ant_artifact_bundle(
        artifact_path=_RESULTS_DIR / "phase0_route_learning.json",
        payload={"artifact_kind": "phase0-physically-walked-route", **route_payload},
        provenance=collect_ant_provenance(
            repo_root=_REPO_ROOT,
            seeds=(seed,),
            config={"exposures": exposures, "route_length": route_length},
        ),
        input_paths=(
            _REFERENCE_DIR / "ardin_route_memory_2016.csv",
            _REFERENCE_DIR / "REFERENCE_METADATA.json",
        ),
        repo_root=_REPO_ROOT,
    )
    print(f"[phase0] {route.description}")
    print(f"[phase0] manifests: {homing_manifest}, {route_manifest}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposures", type=int, default=10)
    parser.add_argument("--route-length", type=int, default=5)
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            main(
                exposures=args.exposures,
                route_length=args.route_length,
                n_trials=args.n_trials,
                seed=args.seed,
            )
        )
    )
