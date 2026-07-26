"""Read-only carrying-vs-food steering-conflict probe.

v25 mirror equivariance finally gave the causal action head real food-steering
authority (``turn_gain`` 0.025, the largest of the four channels, versus 0.006
for the carrying-home bearing). Under exclusive steering every drive shares the
single opponent-coded actuator axis ``z[0]/z[1]``, so a four-fold food advantage
is not free: right after a pickup the body is standing INSIDE the odour field
with a saturated food gradient, and ``food_left/food_right`` are published raw
regardless of ``carrying_food``. If the learned policy does not suppress food
steering while carrying, the food drive out-votes the home drive and the body
orbits the source instead of returning -- exactly the far-tier signature of
"pickups happen, deliveries stay at zero".

The published ``home`` probe cannot see this: its two lanes differ only in the
carrying bit and there is no food source in the scene, so it measures the home
drive in isolation. This script puts the two drives in direct opposition.

Geometry (mirrors the published home probe so the navigator/PI state is the
same truth the gates read): the body sits at ``(2, 0)`` heading north with the
nest at the origin, so home lies to its LEFT (+turn). A butter source is placed
one pickup radius away on the named side. Three lanes per body:

- ``carry_food_right``  carrying, food to the RIGHT  -> drives CONFLICT
- ``carry_food_left``   carrying, food to the LEFT   -> drives AGREE
- ``free_food_right``   not carrying, food to the RIGHT -> food should win

Decomposition (turn is +left):

    home_drive = 0.5 * (turn[carry_food_right] + turn[carry_food_left])
    food_drive = 0.5 * (turn[carry_food_left] - turn[carry_food_right])

``home_drive`` is the part invariant to which side the food is on; ``food_drive``
is the part that flips with it. The decisive numbers are

- ``conflict_turn`` -- the turn actually commanded in the conflict lane. It must
  be positive (toward home) for a carrying body to ever reach the nest.
- ``home_over_food`` -- ``home_drive / food_drive``. Below 1.0 the food gradient
  owns the axis while carrying.
- ``food_suppression`` -- ``food_drive`` while carrying divided by ``food_drive``
  while free. A forager that has learned the carrying gate drives this well
  below 1.0; a value near 1.0 means ``carrying_food`` does not modulate food
  steering at all.

Nothing here feeds learning: frozen checkpoints are restored, one turn is
stepped, and the published motor command is read.

Usage:

    python scripts/measure_ant_carrying_home_conflict.py \
        --progress-dir research/ant/results/.partials/ecology_p1_v25/seed0 \
        --json-out research/ant/results/.partials/carrying_conflict.v25-55.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
from pathlib import Path

from volvence_ant.env import AntWorld, AntWorldConfig, ButterSource
from volvence_ant.evidence.runtime_profile import (
    ant_runtime_replay_rollout_config,
)
from volvence_ant.experiments.ecology_curriculum import (
    EcologyDataSplit,
    EcologyStage,
    EcologyTrainingTier,
    _session_config,
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
from volvence_ant.runtime import (
    AntLearningCheckpoint,
    AntObjectiveKind,
    AntSenseSchema,
    AntSession,
    AntSessionConfig,
    KernelColonyRunner,
)

# Same antenna geometry the ecology curriculum and its probes use; the default
# AntWorldConfig values (30 deg / 0.6) belong to the v1 evidence lane.
_ANTENNA_OFFSET_DEG = 45.0
_ANTENNA_REACH = 0.9
# Home-probe pose: body at (2, 0) heading north, nest at the origin.
_BODY_X = 2.0
_BODY_Y = 0.0
_BODY_HEADING = math.pi / 2.0
# Lateral food offset. The body sits just inside the odour field at one pickup
# radius, which is where it finds itself the tick after a real pickup.
_FOOD_OFFSET = 1.2
_LANES: tuple[tuple[str, bool, float], ...] = (
    # (lane name, carrying, food x-offset; +x is to the body's right)
    ("carry_food_right", True, _FOOD_OFFSET),
    ("carry_food_left", True, -_FOOD_OFFSET),
    ("free_food_right", False, _FOOD_OFFSET),
)
# Under exclusive steering a zero-parameter head commands exactly 0 rad, so the
# cold arm lands on denormal noise. Ratios below this floor are reported as
# degenerate instead of as a meaningful competition read-out.
_DRIVE_FLOOR = 1e-9


def _cold_initial(config: EcologyP1Config) -> tuple[AntLearningCheckpoint, ...]:
    curriculum = _curriculum_config(config)
    bootstrap = KernelColonyRunner(
        _world(
            config=curriculum,
            stage=EcologyStage.COMPOSITE,
            seed=config.seed,
            data_split=EcologyDataSplit.TRAIN,
            tier=EcologyTrainingTier.NEAR,
        ),
        base_config=_session_config(
            config=curriculum,
            seed=config.seed,
            session_id="carry-conflict:cold-initial",
            optimize=True,
        ),
    )
    return bootstrap.export_learning_checkpoints(
        checkpoint_prefix="carry-conflict:cold-initial",
        include_runtime_replay=False,
    )


def _load_arm(
    *,
    progress_dir: Path,
    arm: str,
    config: EcologyP1Config,
) -> tuple[tuple[AntLearningCheckpoint, ...], int]:
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
    return checkpoints, int(state["completed_training_episodes"])


async def _turn_for_lane(
    *,
    checkpoint: AntLearningCheckpoint,
    temporal_latent_dim: int,
    seed: int,
    carrying: bool,
    food_offset: float,
    session_id: str,
) -> dict[str, float]:
    world = AntWorld(
        config=AntWorldConfig(
            seed=seed,
            step_size=0.4,
            antenna_offset_deg=_ANTENNA_OFFSET_DEG,
            antenna_reach=_ANTENNA_REACH,
        ),
        world_objects=(
            ButterSource(
                object_id="conflict-butter",
                x=_BODY_X + food_offset,
                y=_BODY_Y,
                strength=2.2,
                decay=2.4,
                radius=1.1,
            ),
        ),
    )
    session = AntSession(
        world,
        config=AntSessionConfig(
            temporal_latent_dim=temporal_latent_dim,
            session_id=session_id,
            seed=seed,
            heading_noise=0.0,
            step_noise=0.0,
            rollout_config=ant_runtime_replay_rollout_config(
                enable_sparse_exploration=False,
                sense_schema=AntSenseSchema.ECOLOGY_V2,
            ),
            objective=AntObjectiveKind.ECOLOGY,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
        ),
    )
    session.restore_learning_checkpoint(checkpoint)
    # Walk the path integrator out along +x and turn north, so the home vector
    # the controller reads is earned rather than teleported in. This is the
    # published home-probe preamble.
    for _ in range(5):
        session.navigator.update(
            turn_command=0.0,
            step_command=0.4,
            true_heading=0.0,
        )
    session.navigator.update(
        turn_command=math.pi / 2.0,
        step_command=0.0,
        true_heading=math.pi / 2.0,
    )
    world.set_body_pose(
        x=_BODY_X,
        y=_BODY_Y,
        heading=_BODY_HEADING,
        carrying_food=carrying,
    )
    observation = world.observe()
    record = await session.step()
    return {
        "turn": float(record.command.turn_command),
        "food_left": float(observation.food_left),
        "food_right": float(observation.food_right),
        "food_diff": float(observation.food_left - observation.food_right),
    }


async def _measure_body(
    *,
    arm: str,
    body_id: int,
    checkpoint: AntLearningCheckpoint,
    temporal_latent_dim: int,
    seed: int,
) -> dict[str, object]:
    lanes: dict[str, dict[str, float]] = {}
    for name, carrying, offset in _LANES:
        lanes[name] = await _turn_for_lane(
            checkpoint=checkpoint,
            temporal_latent_dim=temporal_latent_dim,
            seed=seed,
            carrying=carrying,
            food_offset=offset,
            session_id=f"carry-conflict:{arm}:{body_id}:{name}",
        )
    conflict = lanes["carry_food_right"]["turn"]
    agree = lanes["carry_food_left"]["turn"]
    free = lanes["free_food_right"]["turn"]
    home_drive = 0.5 * (conflict + agree)
    food_drive = 0.5 * (agree - conflict)
    # The free lane shares the conflict lane's geometry, so its food drive is
    # measured against the same home pull: turn = home_drive - food_drive.
    free_food_drive = home_drive - free
    degenerate = (
        abs(home_drive) < _DRIVE_FLOOR
        and abs(food_drive) < _DRIVE_FLOOR
        and abs(free_food_drive) < _DRIVE_FLOOR
    )
    return {
        "arm": arm,
        "body_id": body_id,
        "lanes": lanes,
        "conflict_turn": conflict,
        "agree_turn": agree,
        "free_turn": free,
        "home_drive": home_drive,
        "food_drive": food_drive,
        "free_food_drive": free_food_drive,
        # A head that commands no steering at all cannot be scored on which
        # drive wins; keep it out of the ratios rather than reporting noise.
        "degenerate": degenerate,
        # Positive means the body still turns toward the nest while the food
        # gradient pulls the other way -- the precondition for any delivery.
        "homes_under_conflict": (not degenerate) and conflict > 0.0,
        "home_over_food": (
            abs(home_drive) / abs(food_drive)
            if not degenerate and abs(food_drive) > 0.0
            else float("inf")
        ),
        # Well below 1.0 means the carrying bit gates food steering down.
        "food_suppression": (
            abs(food_drive) / abs(free_food_drive)
            if not degenerate and abs(free_food_drive) > 0.0
            else float("inf")
        ),
    }


async def _measure_arm(
    *,
    arm: str,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    config: EcologyP1Config,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for body_id, checkpoint in enumerate(checkpoints):
        rows.append(
            await _measure_body(
                arm=arm,
                body_id=body_id,
                checkpoint=checkpoint,
                temporal_latent_dim=config.temporal_latent_dim,
                # Keep the probe on the frozen distribution the P1 gates read;
                # config.seed identifies the resumable journal only.
                seed=config.seed + 700_003,
            )
        )
    return rows


def _print_rows(arm: str, rows: list[dict[str, object]], completed: int) -> None:
    print(f"\n=== arm={arm} episodes={completed} ===")
    print(
        f"{'body':>4} {'conflict':>10} {'agree':>10} {'free':>10} "
        f"{'home_drv':>10} {'food_drv':>10} {'home/food':>10} "
        f"{'suppress':>9} {'homes':>6}"
    )
    for row in rows:
        degenerate = bool(row["degenerate"])
        ratio = float(row["home_over_food"])
        suppression = float(row["food_suppression"])
        ratio_text = "zero" if degenerate else f"{ratio:.3f}"
        suppression_text = "zero" if degenerate else f"{suppression:.3f}"
        print(
            f"{row['body_id']:>4} {float(row['conflict_turn']):>+10.5f} "
            f"{float(row['agree_turn']):>+10.5f} {float(row['free_turn']):>+10.5f} "
            f"{float(row['home_drive']):>+10.5f} {float(row['food_drive']):>+10.5f} "
            f"{ratio_text:>10} {suppression_text:>9} "
            f"{'Y' if row['homes_under_conflict'] else 'n':>6}"
        )


def _summarise(rows: list[dict[str, object]]) -> dict[str, float]:
    finite_ratio = [
        float(row["home_over_food"])
        for row in rows
        if float(row["home_over_food"]) != float("inf")
    ]
    finite_suppression = [
        float(row["food_suppression"])
        for row in rows
        if float(row["food_suppression"]) != float("inf")
    ]
    return {
        "bodies": float(len(rows)),
        "degenerate_bodies": float(sum(1 for row in rows if row["degenerate"])),
        "homes_under_conflict": float(
            sum(1 for row in rows if row["homes_under_conflict"])
        ),
        "median_conflict_turn": statistics.median(
            float(row["conflict_turn"]) for row in rows
        ),
        "median_home_drive": statistics.median(
            float(row["home_drive"]) for row in rows
        ),
        "median_food_drive": statistics.median(
            float(row["food_drive"]) for row in rows
        ),
        "median_home_over_food": (
            statistics.median(finite_ratio) if finite_ratio else float("inf")
        ),
        "median_food_suppression": (
            statistics.median(finite_suppression)
            if finite_suppression
            else float("inf")
        ),
    }


async def _run(args: argparse.Namespace) -> int:
    config = EcologyP1Config(
        n_ants=args.n_ants,
        temporal_latent_dim=args.temporal_latent_dim,
        seed=args.seed,
    )
    payload: list[dict[str, object]] = []
    summary: dict[str, dict[str, float]] = {}
    completed_by_arm: dict[str, int] = {}

    cold_rows = await _measure_arm(
        arm="cold",
        checkpoints=_cold_initial(config),
        config=config,
    )
    completed_by_arm["cold"] = 0
    _print_rows("cold", cold_rows, 0)
    summary["cold"] = _summarise(cold_rows)
    payload.extend(cold_rows)

    if args.progress_dir is not None:
        checkpoints, completed = _load_arm(
            progress_dir=args.progress_dir.resolve(),
            arm=args.arm,
            config=config,
        )
        learned_rows = await _measure_arm(
            arm=args.arm,
            checkpoints=checkpoints,
            config=config,
        )
        completed_by_arm[args.arm] = completed
        _print_rows(args.arm, learned_rows, completed)
        summary[args.arm] = _summarise(learned_rows)
        payload.extend(learned_rows)

    print("\n=== 汇总(四体中位数) ===")
    for arm, entry in summary.items():
        all_degenerate = entry["degenerate_bodies"] == entry["bodies"]
        ratio = entry["median_home_over_food"]
        suppression = entry["median_food_suppression"]
        ratio_text = "zero-head" if all_degenerate else f"{ratio:.3f}"
        suppression_text = "zero-head" if all_degenerate else f"{suppression:.3f}"
        print(
            f"  {arm:<8} homes_under_conflict="
            f"{int(entry['homes_under_conflict'])}/{int(entry['bodies'])} "
            f"conflict_turn={entry['median_conflict_turn']:+.5f} "
            f"home_drive={entry['median_home_drive']:+.5f} "
            f"food_drive={entry['median_food_drive']:+.5f} "
            f"home/food={ratio_text} suppression={suppression_text}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "completed_training_episodes": completed_by_arm,
                    "geometry": {
                        "body": [_BODY_X, _BODY_Y],
                        "heading": _BODY_HEADING,
                        "food_offset": _FOOD_OFFSET,
                        "antenna_offset_deg": _ANTENNA_OFFSET_DEG,
                        "antenna_reach": _ANTENNA_REACH,
                    },
                    "rows": payload,
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
        description="Read-only carrying-vs-food steering-conflict probe"
    )
    parser.add_argument("--n-ants", type=int, default=4)
    parser.add_argument("--temporal-latent-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-dir", type=Path, default=None)
    parser.add_argument("--arm", type=str, default="learned")
    parser.add_argument("--json-out", type=Path, default=None)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
