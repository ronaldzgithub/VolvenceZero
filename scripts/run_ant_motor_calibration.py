"""Run multi-seed hidden-motor calibration with frozen gates and provenance."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from pathlib import Path

import numpy as np

from run_ant_matched_control import _learned_config, _schedule_gated_arms
from volvence_ant.evidence import (
    collect_ant_provenance,
    write_ant_artifact_bundle,
)
from volvence_ant.experiments import run_motor_calibration_experiment


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULT = _REPO_ROOT / "research/ant/results/motor_calibration.v1.json"


def _bootstrap_ci(
    values: tuple[float, ...], *, seed: int
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap values must not be empty")
    if len(values) == 1:
        return (values[0], values[0])
    rng = np.random.default_rng(seed)
    samples = np.asarray(values, dtype=float)
    means = np.asarray(
        [
            rng.choice(samples, size=len(samples), replace=True).mean()
            for _ in range(4000)
        ]
    )
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


async def main(
    *,
    seeds: tuple[int, ...],
    ticks: int,
    switch_tick: int,
    n_z: int,
) -> int:
    reports = []
    for seed in seeds:
        report = await run_motor_calibration_experiment(
            learned_config=_learned_config(
                seed,
                n_z,
                enable_sparse_exploration=False,
            ),
            no_optimize_config=_schedule_gated_arms(
                seed=seed,
                n_z=n_z,
                enable_sparse_exploration=False,
            )["no_optimize"],
            ticks=ticks,
            switch_tick=switch_tick,
            seed=seed,
        )
        reports.append(report)
        print(
            f"[motor-calibration] seed={seed} "
            f"late={report.learned_late_error_advantage:.6f} "
            f"recovery={report.learned_recovery_advantage:.6f} "
            f"verdict={'PASS' if report.learned_recovers_better else 'BLOCK'}"
        )

    late_values = tuple(
        report.learned_late_error_advantage for report in reports
    )
    recovery_values = tuple(
        report.learned_recovery_advantage for report in reports
    )
    late_ci = _bootstrap_ci(late_values, seed=2718)
    recovery_ci = _bootstrap_ci(recovery_values, seed=3141)
    min_late = reports[0].min_late_error_advantage
    min_recovery = reports[0].min_recovery_advantage
    mean_late = float(sum(late_values) / len(late_values))
    mean_recovery = float(sum(recovery_values) / len(recovery_values))
    passed = bool(
        len(seeds) >= 5
        and mean_late >= min_late
        and mean_recovery >= min_recovery
        and late_ci[0] > 0.0
        and recovery_ci[0] > 0.0
    )
    config = {
        "seeds": seeds,
        "ticks": ticks,
        "switch_tick": switch_tick,
        "n_z": n_z,
        "min_late_error_advantage": min_late,
        "min_recovery_advantage": min_recovery,
    }
    payload = {
        "schema_version": "digital-ant-motor-calibration.v1",
        "overall_verdict": "PASS" if passed else "BLOCK",
        "reports": tuple(asdict(report) for report in reports),
        "aggregate": {
            "seed_count": len(seeds),
            "mean_late_error_advantage": mean_late,
            "late_error_advantage_ci95": late_ci,
            "mean_recovery_advantage": mean_recovery,
            "recovery_advantage_ci95": recovery_ci,
        },
        "config": config,
    }
    provenance = collect_ant_provenance(
        repo_root=_REPO_ROOT,
        seeds=seeds,
        config=config,
    )
    manifest = write_ant_artifact_bundle(
        artifact_path=_RESULT,
        payload=payload,
        provenance=provenance,
        repo_root=_REPO_ROOT,
    )
    print(
        f"[motor-calibration] overall={payload['overall_verdict']} "
        f"artifact={_RESULT} manifest={manifest}"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--ticks", type=int, default=60)
    parser.add_argument("--switch-tick", type=int, default=30)
    parser.add_argument("--n-z", type=int, default=16)
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            main(
                seeds=tuple(int(value) for value in args.seeds.split(",")),
                ticks=args.ticks,
                switch_tick=args.switch_tick,
                n_z=args.n_z,
            )
        )
    )
