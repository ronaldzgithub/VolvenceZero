"""Read-only PE boundary separation-margin measurement for one P1 journal.

The v31 PE-boundary change turns any prediction error strictly above the
profile floor (ant: 0.45) into a forced temporal segment boundary. The floor
was calibrated against a single observed pickup PE (0.4789), leaving a margin
of only 0.029 -- and the repository holds no record of the NON-pickup PE
distribution. If routine ticks routinely land in 0.45-0.48, every tick closes
a segment, segments collapse to length 1, and the segment credit this change
exists to create is destroyed before the first optimizer batch.

This script answers that question before the training spend: it freeze-replays
episodes from a finished P1 journal (frozen restore, ``optimize=False``,
``learning_enabled=False``, ``enable_sparse_exploration=False``, matching
``_evaluate_arm``), collects ``AntStepRecord.pe_magnitude`` on every tick, and
splits ticks into pickup / delivery / bootstrap / routine classes. Pickup
ticks are marked by the per-body ``carrying_food`` False -> True transition.

PE settlement timing: the record published at tick ``t`` carries the PE that
scored the outcome submitted at ``t - 1``, so the PE that "sees" a pickup
lands on the NEXT record of the same body. Both readings are reported
(``pe_at_event_tick`` and ``pe_next_tick``) so the timing convention is
auditable against the historically observed pickup PE instead of assumed.

Schema note: a P1 journal only rehydrates under code whose
``ECOLOGY_P1_PROGRESS_SCHEMA_VERSION`` matches the journal (the version is
baked into the archive compatibility fingerprint on purpose). To measure a
v30 journal (progress.v27) while the working tree is already at v28, run this
script with ``PYTHONPATH`` pointing at the package sources of a git worktree
checked out at the commit that wrote the journal. The script only uses P1
owner helpers that exist on both sides of that boundary.

Usage (v30 journal written by the /tmp worktree at e42e4d0):

    W=/private/tmp/volvence-ecology-v30-worktree
    PYTHONPATH="$W/packages/vz-embodiment-ant/src:$W/packages/vz-temporal/src:\
$W/packages/vz-contracts/src:$W/packages/vz-runtime/src:\
$W/packages/vz-cognition/src:$W/packages/vz-memory/src:\
$W/packages/vz-substrate/src:$W/packages/vz-application/src" \
    python scripts/measure_ant_pe_boundary_margin.py \
        --progress-dir "$W/research/ant/results/.partials/ecology_p1_v30_mps/seed0" \
        --json-out research/ant/results/.partials/pe_boundary_margin.v30.json

It is a measurement, NOT a verdict on the arm: nothing is written back to the
journal, no gate is evaluated, and no learning state is mutated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from volvence_ant.experiments.ecology_curriculum import (
    EcologyDataSplit,
    _flatten_records,
    _session_config,
    _synchronize_curriculum_navigators,
    _world,
)
from volvence_ant.experiments.ecology_p1 import (
    EcologyP1Config,
    _curriculum_config,
    _fixed_schedule,
    _hydrate_progress_checkpoints,
    _load_arm_progress,
    _read_progress_archive,
    _schedule_digest,
)
from volvence_ant.runtime import KernelColonyRunner

# Default replay set: the forced-return block (episodes 15-19) guarantees real
# pickup transitions, and the butter-medium block (episodes 25-29) is the tier
# the boundary change is meant to unlock. Both come from the fixed training
# schedule, so seeds and geometry are exactly what training saw.
_DEFAULT_EPISODES = (15, 16, 17, 18, 19, 25, 26, 27, 28, 29)

# Candidate floors reported side by side: the generic default (0.5), the ant
# profile value under test (0.45), and one step below for sensitivity.
_CANDIDATE_FLOORS = (0.40, 0.45, 0.50)


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": float(array.size),
        "p50": float(np.percentile(array, 50.0)),
        "p90": float(np.percentile(array, 90.0)),
        "p99": float(np.percentile(array, 99.0)),
        "max": float(array.max()),
    }


def _floor_exceedances(values: list[float]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for floor in _CANDIDATE_FLOORS:
        exceeding = sum(1 for value in values if value > floor)
        result[f"{floor:.2f}"] = {
            "count": float(exceeding),
            "ratio": (float(exceeding) / len(values)) if values else 0.0,
        }
    return result


async def _replay_episode(
    *,
    curriculum: Any,
    checkpoints: tuple[Any, ...],
    plan: Any,
    arm: str,
    rounds: int,
) -> tuple[Any, ...]:
    runner = KernelColonyRunner(
        _world(
            config=curriculum,
            stage=plan.stage,
            seed=plan.seed,
            data_split=EcologyDataSplit.TRAIN,
            tier=plan.tier,
            forced_escape=plan.forced_escape,
            forced_return=plan.forced_return,
            forced_approach=plan.forced_approach,
        ),
        base_config=_session_config(
            config=curriculum,
            seed=plan.seed,
            session_id=(
                f"pe-margin:{arm}:{plan.stage.value}:{plan.tier.value}:"
                f"episode:{plan.episode_index}"
            ),
            optimize=False,
            learning_enabled=False,
            sparse_exploration_enabled=False,
        ),
    )
    if plan.forced_return or plan.forced_approach:
        _synchronize_curriculum_navigators(runner)
    runner.restore_learning_checkpoints(checkpoints)
    await runner.run(rounds)
    return _flatten_records(runner)


def _classify(
    records: tuple[Any, ...],
    *,
    episode_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float], int]:
    """Split one episode's records into event rows and routine PE values.

    Returns ``(pickup_events, delivery_events, routine_values, bootstrap_count)``.
    Routine excludes bootstrap ticks and the event tick AND its successor for
    every pickup/delivery, so whichever settlement timing convention is right,
    the event PE cannot leak into the routine distribution.
    """

    by_body: dict[int, list[Any]] = defaultdict(list)
    for record in records:
        by_body[record.body_id].append(record)

    pickup_events: list[dict[str, Any]] = []
    delivery_events: list[dict[str, Any]] = []
    routine_values: list[float] = []
    bootstrap_count = 0
    for body_id, body_records in sorted(by_body.items()):
        body_records.sort(key=lambda item: item.tick)
        event_indexes: set[int] = set()
        previous_carrying = False
        for index, record in enumerate(body_records):
            picked = record.carrying_food and not previous_carrying
            dropped = previous_carrying and not record.carrying_food
            previous_carrying = record.carrying_food
            if not (picked or dropped):
                continue
            event_indexes.add(index)
            event_indexes.add(index + 1)
            next_pe = (
                float(body_records[index + 1].pe_magnitude)
                if index + 1 < len(body_records)
                else None
            )
            row = {
                "episode_index": episode_index,
                "body_id": body_id,
                "tick": int(record.tick),
                "pe_at_event_tick": float(record.pe_magnitude),
                "pe_next_tick": next_pe,
            }
            # delivered=True on the drop record distinguishes a nest delivery
            # from any other carrying release; keep the raw split visible.
            if picked:
                pickup_events.append(row)
            else:
                delivery_events.append(
                    {**row, "delivered": bool(record.delivered)}
                )
        for index, record in enumerate(body_records):
            if record.pe_bootstrap:
                bootstrap_count += 1
                continue
            if index in event_indexes:
                continue
            routine_values.append(float(record.pe_magnitude))
    return pickup_events, delivery_events, routine_values, bootstrap_count


def _verdict(
    *,
    pickup_events: list[dict[str, Any]],
    routine_values: list[float],
    floor: float,
) -> dict[str, Any]:
    routine_stats = _percentiles(routine_values)
    pickup_next = [
        row["pe_next_tick"]
        for row in pickup_events
        if row["pe_next_tick"] is not None
    ]
    pickup_at = [row["pe_at_event_tick"] for row in pickup_events]
    verdict: dict[str, Any] = {
        "floor": floor,
        "routine": routine_stats,
        "routine_exceedances": _floor_exceedances(routine_values),
        "pickup_pe_next_tick": _percentiles(pickup_next),
        "pickup_pe_at_event_tick": _percentiles(pickup_at),
    }
    if routine_values and pickup_next:
        margin = min(pickup_next) - float(
            np.percentile(np.asarray(routine_values), 99.0)
        )
        verdict["separation_margin_min_pickup_minus_routine_p99"] = margin
    over = sum(1 for value in routine_values if value > floor)
    verdict["routine_ticks_forcing_boundary"] = over
    verdict["over_segmentation_risk"] = bool(
        routine_values
        and float(np.percentile(np.asarray(routine_values), 99.0)) >= floor
    )
    return verdict


async def _run(args: argparse.Namespace) -> int:
    config = EcologyP1Config(
        n_ants=args.n_ants,
        temporal_latent_dim=args.temporal_latent_dim,
        evaluation_rounds=args.evaluation_rounds,
        layouts_per_tier=args.layouts_per_tier,
        seed=args.seed,
    )
    curriculum = _curriculum_config(config)
    schedule = _fixed_schedule(config)
    schedule_sha256 = _schedule_digest(schedule)
    progress_dir = args.progress_dir.resolve()
    state = _load_arm_progress(
        progress_dir=progress_dir,
        arm=args.arm,
        config=config,
        schedule_sha256=schedule_sha256,
    )
    if state is None:
        raise FileNotFoundError(
            f"no journalled arm {args.arm!r} under {progress_dir}"
        )
    archives = _read_progress_archive(
        progress_dir=progress_dir,
        state=state,
        config=config,
    )
    checkpoints = _hydrate_progress_checkpoints(
        config=config,
        curriculum=curriculum,
        archives=archives,
        arm=args.arm,
    )
    completed = int(state["completed_training_episodes"])
    rounds = args.rounds if args.rounds is not None else curriculum.stage_rounds
    print(
        f"arm={args.arm} journal={state['schema_version']} "
        f"episodes={completed}/{len(schedule)} rounds_per_episode={rounds}"
    )

    episode_indexes = tuple(args.episode or _DEFAULT_EPISODES)
    all_pickups: list[dict[str, Any]] = []
    all_deliveries: list[dict[str, Any]] = []
    all_routine: list[float] = []
    bootstrap_total = 0
    episode_rows: list[dict[str, Any]] = []
    for episode_index in episode_indexes:
        plan = schedule[episode_index]
        records = await _replay_episode(
            curriculum=curriculum,
            checkpoints=checkpoints,
            plan=plan,
            arm=args.arm,
            rounds=rounds,
        )
        pickups, deliveries, routine, bootstraps = _classify(
            records,
            episode_index=episode_index,
        )
        all_pickups.extend(pickups)
        all_deliveries.extend(deliveries)
        all_routine.extend(routine)
        bootstrap_total += bootstraps
        stats = _percentiles(routine)
        episode_rows.append(
            {
                "episode_index": episode_index,
                "stage": plan.stage.value,
                "tier": plan.tier.value,
                "seed": plan.seed,
                "forced_return": plan.forced_return,
                "forced_approach": plan.forced_approach,
                "pickup_events": len(pickups),
                "release_events": len(deliveries),
                "routine": stats,
                "routine_exceedances": _floor_exceedances(routine),
            }
        )
        print(
            f"  ep{episode_index:>2} {plan.stage.value}/{plan.tier.value} "
            f"forced_return={plan.forced_return} pickups={len(pickups)} "
            f"releases={len(deliveries)} routine_ticks={int(stats['count'])} "
            f"routine_p99="
            + (f"{stats['p99']:.4f}" if "p99" in stats else "n/a")
        )

    verdict = _verdict(
        pickup_events=all_pickups,
        routine_values=all_routine,
        floor=args.floor,
    )
    print("\n=== 判词 ===")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    payload = {
        "arm": args.arm,
        "journal_schema_version": state["schema_version"],
        "progress_dir": str(progress_dir),
        "completed_training_episodes": completed,
        "rounds_per_episode": rounds,
        "episodes": episode_rows,
        "pickup_events": all_pickups,
        "release_events": all_deliveries,
        "bootstrap_ticks_excluded": bootstrap_total,
        "verdict": verdict,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nreport: {args.json_out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only PE boundary separation-margin measurement for one "
            "P1 journal"
        )
    )
    parser.add_argument("--n-ants", type=int, default=4)
    parser.add_argument("--temporal-latent-dim", type=int, default=16)
    parser.add_argument("--evaluation-rounds", type=int, default=120)
    parser.add_argument("--layouts-per-tier", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-dir", type=Path, required=True)
    parser.add_argument("--arm", type=str, default="learned")
    parser.add_argument(
        "--episode",
        action="append",
        type=int,
        default=None,
        help=(
            "Training-schedule episode index to replay (repeatable); "
            f"default {_DEFAULT_EPISODES}"
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Rounds per episode (default: the curriculum training budget)",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.45,
        help="Boundary floor under test (ant profile default 0.45)",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
