"""Read-only post-pickup switch-rate / family-persistence probe (plan D6).

The v31 typed milestone boundary promises exactly two runtime facts that no
episode scoreboard can show:

1. **Switch rate**: after a real pickup (per-body ``carrying_food``
   False -> True), the temporal owner must close the running segment and
   switch at the next decision -- within <= 2 records of the event.
2. **Persistence**: the family adopted by that forced switch must survive a
   few actions instead of being immediately reverted by the natural beta
   path (debt D4: a switch that lasts one tick creates no carrying segment
   for credit to flow into).

This script freeze-replays selected training-schedule episodes from a P1
journal (frozen restore, ``optimize=False``, ``learning_enabled=False``,
matching ``_evaluate_arm``) and reads both facts straight from
``AntStepRecord`` telemetry (``is_switching``, ``abstract_action``). The
milestone -> boundary path is runtime wiring, not a learning update, so it is
fully active during frozen replay.

Reported per pickup/release event:

- ``switch_delay``: records until the first ``is_switching`` tick strictly
  after the event record (None = no switch within the horizon = a miss).
- ``persistence``: records the post-switch family survives before the next
  switch (horizon-capped).
- ``reverted_to_pre_family``: whether the next switch lands back on the
  pre-event family (the "switched and snapped back" failure mode).

Plus the routine per-tick switch rate as context: a ~100% post-pickup rate is
only evidence if the baseline rate is far below it.

Usage (v31 journal, current working tree):

    python scripts/measure_ant_milestone_switch_probe.py \
        --progress-dir research/ant/results/.partials/ecology_p1_v31/seed0 \
        --json-out research/ant/results/.partials/milestone_switch_probe.v31.json

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

# Forced-return block (15-19) guarantees real pickups from the training
# schedule; the butter-medium block (25-29) is the tier the boundary exists to
# unlock. Same default as the PE margin probe so the two reports line up.
_DEFAULT_EPISODES = (15, 16, 17, 18, 19, 25, 26, 27, 28, 29)

# How many records after an event we search for the forced switch, and how
# many we follow the new family before declaring persistence horizon-capped.
_SWITCH_HORIZON = 4
_PERSISTENCE_HORIZON = 8


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
                f"switch-probe:{arm}:{plan.stage.value}:{plan.tier.value}:"
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


def _event_rows(
    records: tuple[Any, ...],
    *,
    episode_index: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Extract per-event switch/persistence rows from one episode.

    Returns ``(event_rows, switching_tick_count, total_tick_count)``; the two
    counts form the routine switch-rate denominator over ALL records so the
    baseline includes event neighbourhoods (that is the honest baseline: it
    can only make the post-event rate look less special, never more).
    """

    by_body: dict[int, list[Any]] = defaultdict(list)
    for record in records:
        by_body[record.body_id].append(record)

    rows: list[dict[str, Any]] = []
    switching_ticks = 0
    total_ticks = 0
    for body_id, body_records in sorted(by_body.items()):
        body_records.sort(key=lambda item: item.tick)
        total_ticks += len(body_records)
        switching_ticks += sum(
            1 for record in body_records if record.is_switching
        )
        previous_carrying = False
        for index, record in enumerate(body_records):
            picked = record.carrying_food and not previous_carrying
            dropped = previous_carrying and not record.carrying_food
            previous_carrying = record.carrying_food
            if not (picked or dropped):
                continue
            pre_family = record.abstract_action
            switch_delay: int | None = None
            for delta in range(1, _SWITCH_HORIZON + 1):
                probe = index + delta
                if probe >= len(body_records):
                    break
                if body_records[probe].is_switching:
                    switch_delay = delta
                    break
            persistence: int | None = None
            post_family: str | None = None
            reverted: bool | None = None
            if switch_delay is not None:
                switch_index = index + switch_delay
                post_family = body_records[switch_index].abstract_action
                survived = 0
                for probe in range(
                    switch_index + 1,
                    min(
                        switch_index + 1 + _PERSISTENCE_HORIZON,
                        len(body_records),
                    ),
                ):
                    if body_records[probe].is_switching:
                        reverted = (
                            body_records[probe].abstract_action == pre_family
                        )
                        break
                    survived += 1
                persistence = survived
            rows.append(
                {
                    "episode_index": episode_index,
                    "body_id": body_id,
                    "tick": int(record.tick),
                    "event": "pickup" if picked else "release",
                    "delivered": bool(record.delivered) if dropped else None,
                    "pre_family": pre_family,
                    "post_family": post_family,
                    "switch_delay": switch_delay,
                    "persistence": persistence,
                    "reverted_to_pre_family": reverted,
                }
            )
    return rows, switching_ticks, total_ticks


def _summarize(
    rows: list[dict[str, Any]],
    *,
    event: str,
) -> dict[str, Any]:
    events = [row for row in rows if row["event"] == event]
    delays = [
        row["switch_delay"] for row in events if row["switch_delay"] is not None
    ]
    persists = [
        row["persistence"] for row in events if row["persistence"] is not None
    ]
    reverted = [
        row
        for row in events
        if row["reverted_to_pre_family"] is True
    ]
    return {
        "events": len(events),
        "switched_within_horizon": len(delays),
        "switch_rate": (len(delays) / len(events)) if events else None,
        "switched_within_2": sum(1 for delay in delays if delay <= 2),
        "switch_rate_within_2": (
            sum(1 for delay in delays if delay <= 2) / len(events)
            if events
            else None
        ),
        "delay_histogram": {
            str(delay): delays.count(delay) for delay in sorted(set(delays))
        },
        "persistence_min": min(persists) if persists else None,
        "persistence_median": (
            sorted(persists)[len(persists) // 2] if persists else None
        ),
        "persistence_ge_3": (
            sum(1 for value in persists if value >= 3) / len(persists)
            if persists
            else None
        ),
        "reverted_to_pre_family": len(reverted),
    }


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
    all_rows: list[dict[str, Any]] = []
    switching_total = 0
    tick_total = 0
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
        rows, switching, ticks = _event_rows(
            records,
            episode_index=episode_index,
        )
        all_rows.extend(rows)
        switching_total += switching
        tick_total += ticks
        pickups = sum(1 for row in rows if row["event"] == "pickup")
        episode_rows.append(
            {
                "episode_index": episode_index,
                "stage": plan.stage.value,
                "tier": plan.tier.value,
                "forced_return": plan.forced_return,
                "forced_approach": plan.forced_approach,
                "pickup_events": pickups,
                "release_events": len(rows) - pickups,
                "switching_ticks": switching,
                "total_ticks": ticks,
            }
        )
        print(
            f"  ep{episode_index:>2} {plan.stage.value}/{plan.tier.value} "
            f"pickups={pickups} releases={len(rows) - pickups} "
            f"baseline_switch_rate={switching / ticks:.3f}"
        )

    summary = {
        "pickup": _summarize(all_rows, event="pickup"),
        "release": _summarize(all_rows, event="release"),
        "baseline_switch_rate_all_ticks": (
            switching_total / tick_total if tick_total else None
        ),
        "switch_horizon": _SWITCH_HORIZON,
        "persistence_horizon": _PERSISTENCE_HORIZON,
    }
    print("\n=== 判词 ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    payload = {
        "arm": args.arm,
        "journal_schema_version": state["schema_version"],
        "progress_dir": str(progress_dir),
        "completed_training_episodes": completed,
        "rounds_per_episode": rounds,
        "episodes": episode_rows,
        "events": all_rows,
        "summary": summary,
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
            "Read-only post-pickup switch-rate / family-persistence probe "
            "for one P1 journal"
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
    parser.add_argument("--json-out", type=Path, default=None)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
