#!/usr/bin/env python3
"""Coding-lab Packet 1 verdict run: SHADOW observer over logged trajectories.

Consumes a Packet 0 calibration run directory and produces the Packet 1
verdict artifacts:

* ``forecast_skill`` — pre-oracle owner forecasts (``next_prediction``)
  separate passed from failed episodes better than a within-chain
  permutation null (one-sided);
* ``pe_discrimination`` — settled PE (signed reward / task progress)
  separates passed from failed episodes (instrument has scale);
* ``cross_process_recovery`` — a fresh Brain over the same scoped
  memory root hydrates the persisted entries.

SHADOW proof: the observer replays logs post-hoc; production wiring is
untouched (``production_wiring_changed = false`` in the report).

Usage:

    .venv/bin/python scripts/run_coding_lab_observer.py \
        --calibration-run-dir artifacts/coding_lab/coding_lab_calibration_scripted_20260812
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import sys
import time
from random import Random
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _pkg in ("lifeform-evolution", "lifeform-domain-coding"):
    sys.path.insert(0, str(_REPO_ROOT / "packages" / _pkg / "src"))

from lifeform_evolution.coding_lab_observer import (  # noqa: E402
    ChainObservationResult,
    observe_calibration_chain,
    recovered_memory_entry_count,
)


def _separation(values_passed: list[float], values_failed: list[float]) -> float:
    if not values_passed or not values_failed:
        return 0.0
    return statistics.fmean(values_passed) - statistics.fmean(values_failed)


def _within_chain_permutation_p(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    permutations: int,
    seed: int,
) -> tuple[float, float]:
    """One-sided within-chain permutation test.

    Statistic: mean(value | passed) - mean(value | failed). Labels are
    shuffled WITHIN each chain so chain-level pass-rate structure is
    preserved (the chain is the cluster unit).
    """

    observed = _separation(
        [row[value_key] for row in rows if row["passed"]],
        [row[value_key] for row in rows if not row["passed"]],
    )
    rng = Random(seed)
    chains: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        chains.setdefault(row["chain_id"], []).append(row)
    at_least_as_large = 0
    for _ in range(permutations):
        permuted_passed: list[float] = []
        permuted_failed: list[float] = []
        for chain_rows in chains.values():
            labels = [row["passed"] for row in chain_rows]
            rng.shuffle(labels)
            for row, label in zip(chain_rows, labels, strict=True):
                (permuted_passed if label else permuted_failed).append(row[value_key])
        if _separation(permuted_passed, permuted_failed) >= observed:
            at_least_as_large += 1
    p_value = (at_least_as_large + 1) / (permutations + 1)
    return observed, p_value


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-run-dir", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=str(_REPO_ROOT / "artifacts" / "coding_lab"))
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)

    calibration_dir = pathlib.Path(args.calibration_run_dir)
    chains_dir = calibration_dir / "chains"
    if not chains_dir.is_dir():
        raise SystemExit(f"no chains directory under {calibration_dir!s}")
    run_id = args.run_id or f"coding_lab_observer_{int(time.time())}"
    run_dir = pathlib.Path(args.output_root) / run_id
    if run_dir.exists():
        raise SystemExit(f"observer run dir already exists: {run_dir!s}")
    run_dir.mkdir(parents=True)

    started = time.time()
    rows: list[dict[str, Any]] = []
    recovery_checks: list[dict[str, Any]] = []
    for chain_dir in sorted(chains_dir.iterdir()):
        trajectories = chain_dir / "trajectories"
        if not trajectories.is_dir():
            continue
        chain_id = chain_dir.name
        brain_state_root = run_dir / "brain_state" / chain_id
        result: ChainObservationResult = asyncio.run(
            observe_calibration_chain(
                chain_id=chain_id,
                trajectories_dir=trajectories,
                brain_state_root=brain_state_root,
            )
        )
        recovered = recovered_memory_entry_count(
            chain_id=chain_id, brain_state_root=brain_state_root
        )
        recovery_checks.append(
            {
                "chain_id": chain_id,
                "persisted": result.persisted,
                "entries_before_restart": result.memory_entry_count_before_restart,
                "entries_recovered": recovered,
                "recovered_non_empty": recovered > 0,
            }
        )
        for observation in result.observations:
            rows.append(
                {
                    "chain_id": observation.chain_id,
                    "episode_index": observation.episode_index,
                    "task_id": observation.task_id,
                    "category": observation.category,
                    "passed": observation.passed,
                    "bet_present": observation.bet_at_task_presented.predicted_task_progress,
                    "bet_pre_oracle": observation.bet_pre_oracle.predicted_task_progress,
                    "bet_pre_oracle_confidence": observation.bet_pre_oracle.confidence,
                    "settled_task_progress": observation.settled_task_progress,
                    "settled_signed_reward": observation.settled_signed_reward,
                    "settled_task_error": observation.settled_task_error,
                    "external_outcome_refs": len(observation.external_outcome_refs),
                    "turns_used": observation.turns_used,
                }
            )

    has_both_labels = any(row["passed"] for row in rows) and any(
        not row["passed"] for row in rows
    )
    if not has_both_labels:
        raise SystemExit(
            "observer verdict requires both passed and failed episodes in the "
            "calibration run; re-run Packet 0 with mixed outcomes first"
        )

    forecast_stat, forecast_p = _within_chain_permutation_p(
        rows, value_key="bet_pre_oracle", permutations=args.permutations, seed=101
    )
    forecast_present_stat, forecast_present_p = _within_chain_permutation_p(
        rows, value_key="bet_present", permutations=args.permutations, seed=102
    )
    reward_stat, reward_p = _within_chain_permutation_p(
        rows, value_key="settled_signed_reward", permutations=args.permutations, seed=103
    )
    progress_stat, progress_p = _within_chain_permutation_p(
        rows, value_key="settled_task_progress", permutations=args.permutations, seed=104
    )

    verdicts = {
        "forecast_skill": forecast_p < args.alpha,
        "pe_discrimination": reward_p < args.alpha and progress_p < args.alpha,
        "cross_process_recovery": bool(recovery_checks)
        and all(check["persisted"] and check["recovered_non_empty"] for check in recovery_checks),
        "external_outcome_channel": all(row["external_outcome_refs"] >= 1 for row in rows),
    }
    report = {
        "packet": "coding-lab-packet-1",
        "run_id": run_id,
        "calibration_run_dir": str(calibration_dir),
        "started_unix": int(started),
        "production_wiring_changed": False,
        "shadow_scope": (
            "post-hoc replay observer; forecasts recorded before outcome submission; "
            "verdict scope bound to the calibration run's hand "
            "(scripted => machinery instrumentation, api => frozen-hand evidence)"
        ),
        "episodes": rows,
        "statistics": {
            "forecast_skill": {"statistic": forecast_stat, "p_value": forecast_p},
            "forecast_skill_at_present": {
                "statistic": forecast_present_stat,
                "p_value": forecast_present_p,
            },
            "pe_signed_reward_separation": {"statistic": reward_stat, "p_value": reward_p},
            "pe_task_progress_separation": {"statistic": progress_stat, "p_value": progress_p},
            "permutations": args.permutations,
            "alpha": args.alpha,
            "null": "within-chain label permutation (chain = cluster unit), one-sided",
        },
        "recovery_checks": recovery_checks,
        "verdicts": verdicts,
        "wall_seconds": time.time() - started,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# coding-lab Packet 1 observer verdict",
        "",
        f"- run_id: `{run_id}` over `{calibration_dir.name}`",
        f"- episodes: {len(rows)} (passed {sum(1 for r in rows if r['passed'])})",
        f"- forecast_skill: **{verdicts['forecast_skill']}** "
        f"(stat={forecast_stat:.4f}, p={forecast_p:.4f})",
        f"- pe_discrimination: **{verdicts['pe_discrimination']}** "
        f"(signed_reward stat={reward_stat:.4f} p={reward_p:.4f}; "
        f"task_progress stat={progress_stat:.4f} p={progress_p:.4f})",
        f"- cross_process_recovery: **{verdicts['cross_process_recovery']}**",
        f"- external_outcome_channel: **{verdicts['external_outcome_channel']}**",
        "- production_wiring_changed: false (SHADOW replay observer)",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "verdicts": verdicts}, ensure_ascii=False))
    print(f"report: {run_dir / 'report.json'}")
    return 0 if all(verdicts.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
