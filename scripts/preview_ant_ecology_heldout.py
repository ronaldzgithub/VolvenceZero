"""Read-only held-out evaluation preview for ONE journalled P1 arm.

The formal P1 report refuses to evaluate until all five arms have finished
their 55-episode schedule, because its gates are matched-control comparisons.
That is the right bar for a verdict, but it is the wrong instrument for the
single open question after the ``learned`` arm finishes: does the far tier
still deliver zero on held-out layouts?

Far training episodes run 24 rounds while the frozen evaluation runs the P1
formal held-out budget (``ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS``), and a far
round trip needs ``2*d - 3.4`` units of path at 0.4 per round -- 6.5 rounds at
the near edge of the far band and 13.5 at the far edge, before any search cost.
A far zero in the training log is therefore weak evidence on its own.
This script runs the exact evaluation specs, seeds, data split and round count
the formal report uses, for one arm, so the far question can be answered in
minutes instead of after the remaining three arms have trained.

It is a preview, NOT a verdict: no gate is evaluated, no matched control is
run, and nothing is written to the journal. Frozen checkpoints are restored
with learning, optimization and sparse exploration all disabled.

Usage:

    python scripts/preview_ant_ecology_heldout.py \
        --progress-dir research/ant/results/.partials/ecology_p1_v25/seed0 \
        --json-out research/ant/results/.partials/heldout_preview.v25-55.json

Restrict the work with ``--capability butter_far --capability butter_medium``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from volvence_ant.experiments.ecology_curriculum import (
    EcologyDataSplit,
    _evaluate_arm,
)
from volvence_ant.experiments.ecology_p1 import (
    ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS,
    EcologyP1Config,
    _curriculum_config,
    _evaluation_specs,
    _fixed_schedule,
    _hydrate_progress_checkpoints,
    _load_arm_progress,
    _read_progress_archive,
    _schedule_digest,
)
from volvence_ant.runtime import AntLearningCheckpoint


def _load_arm(
    *,
    progress_dir: Path,
    arm: str,
    config: EcologyP1Config,
) -> tuple[tuple[AntLearningCheckpoint, ...], int, bool]:
    schedule_sha256 = _schedule_digest(_fixed_schedule(config))
    state = _load_arm_progress(
        progress_dir=progress_dir,
        arm=arm,
        config=config,
        schedule_sha256=schedule_sha256,
    )
    if state is None:
        raise FileNotFoundError(f"no journalled arm {arm!r} under {progress_dir}")
    archives = _read_progress_archive(
        progress_dir=progress_dir,
        state=state,
        config=config,
    )
    checkpoints = _hydrate_progress_checkpoints(
        config=config,
        curriculum=_curriculum_config(config),
        archives=archives,
        arm=arm,
    )
    return (
        checkpoints,
        int(state["completed_training_episodes"]),
        bool(state.get("training_complete")),
    )


async def _run(args: argparse.Namespace) -> int:
    config = EcologyP1Config(
        n_ants=args.n_ants,
        temporal_latent_dim=args.temporal_latent_dim,
        evaluation_rounds=args.evaluation_rounds,
        layouts_per_tier=args.layouts_per_tier,
        seed=args.seed,
    )
    curriculum = _curriculum_config(config)
    checkpoints, completed, complete = _load_arm(
        progress_dir=args.progress_dir.resolve(),
        arm=args.arm,
        config=config,
    )
    schedule_length = len(_fixed_schedule(config))
    print(
        f"arm={args.arm} episodes={completed}/{schedule_length} "
        f"training_complete={complete} rounds={config.evaluation_rounds} "
        f"split=heldout"
    )
    if not complete:
        print(
            "NOTE: this checkpoint is mid-schedule, so the preview reads a "
            "partially trained policy."
        )
    specs = _evaluation_specs()
    wanted = set(args.capability or ())
    rows: list[dict[str, object]] = []
    for capability_index, (capability, scenario, tier) in enumerate(specs):
        if wanted and capability not in wanted:
            continue
        print(f"\n--- {capability} ({scenario.value}, {tier.value}) ---")
        for index in range(config.layouts_per_tier):
            seed = (
                config.seed
                + 2_000_003
                + capability_index * 10_007
                + index * 103
            )
            metrics = await _evaluate_arm(
                config=curriculum,
                checkpoints=checkpoints,
                arm=args.arm,
                data_split=EcologyDataSplit.HELDOUT,
                scenario=scenario,
                seed=seed,
                tier=tier,
            )
            row = {
                "capability": capability,
                "scenario": scenario.value,
                "tier": tier.value,
                "seed": seed,
                "pickups": int(metrics.pickups),
                "deliveries": int(metrics.deliveries),
                "harmful_heat_ticks": int(metrics.harmful_heat_ticks),
                "heat_entries": int(metrics.heat_entries),
                "heat_escapes": int(metrics.heat_escapes),
                "obstacle_contacts": int(metrics.obstacle_contacts),
                "first_pickup_tick": metrics.first_pickup_tick,
                "minimum_food_distance": metrics.minimum_food_distance,
            }
            rows.append(row)
            reach = (
                f"{metrics.minimum_food_distance:.2f}"
                if metrics.minimum_food_distance is not None
                else "n/a"
            )
            print(
                f"  seed={seed} pickups={row['pickups']:>3} "
                f"deliveries={row['deliveries']:>3} "
                f"first_pickup_tick={str(row['first_pickup_tick']):>4} "
                f"min_food_dist={reach:>6} "
                f"harmful_heat={row['harmful_heat_ticks']:>3}"
            )

    print("\n=== 能力汇总 ===")
    summary: dict[str, dict[str, float]] = {}
    for capability in sorted({str(row["capability"]) for row in rows}):
        capability_rows = [row for row in rows if row["capability"] == capability]
        reached = [
            row
            for row in capability_rows
            if int(row["pickups"]) > 0
        ]
        entry = {
            "layouts": float(len(capability_rows)),
            "layouts_with_pickup": float(len(reached)),
            "layouts_with_delivery": float(
                sum(1 for row in capability_rows if int(row["deliveries"]) > 0)
            ),
            "pickups": float(sum(int(row["pickups"]) for row in capability_rows)),
            "deliveries": float(
                sum(int(row["deliveries"]) for row in capability_rows)
            ),
            "harmful_heat_ticks": float(
                sum(int(row["harmful_heat_ticks"]) for row in capability_rows)
            ),
        }
        summary[capability] = entry
        print(
            f"  {capability:<20} pickups={int(entry['pickups']):>3} "
            f"deliveries={int(entry['deliveries']):>3} "
            f"layouts_with_pickup="
            f"{int(entry['layouts_with_pickup'])}/{int(entry['layouts'])} "
            f"layouts_with_delivery="
            f"{int(entry['layouts_with_delivery'])}/{int(entry['layouts'])} "
            f"harmful_heat={int(entry['harmful_heat_ticks'])}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "arm": args.arm,
                    "completed_training_episodes": completed,
                    "training_complete": complete,
                    "evaluation_rounds": config.evaluation_rounds,
                    "data_split": EcologyDataSplit.HELDOUT.value,
                    "rows": rows,
                    "summary": summary,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"\nreport: {args.json_out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only held-out evaluation preview for one P1 arm"
    )
    parser.add_argument("--n-ants", type=int, default=4)
    parser.add_argument("--temporal-latent-dim", type=int, default=16)
    # Must track the frozen held-out budget, not a retired literal: the point
    # of this preview is to run "the exact evaluation specs, seeds, data split
    # and round count the formal report uses". A 40-round default silently
    # previewed a third of the formal budget and would answer the far-tier
    # question on a rollout too short to complete a far round trip.
    parser.add_argument(
        "--evaluation-rounds",
        type=int,
        default=ECOLOGY_P1_FORMAL_MIN_HELDOUT_ROUNDS,
    )
    parser.add_argument("--layouts-per-tier", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-dir", type=Path, required=True)
    parser.add_argument("--arm", type=str, default="learned")
    parser.add_argument(
        "--capability",
        action="append",
        default=None,
        help="Restrict to named capabilities (repeatable).",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
