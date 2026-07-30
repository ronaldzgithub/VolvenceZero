"""Dev-only HTML theater for a journalled digital-ant checkpoint.

This is intentionally not a promotion/demo admission path.  It restores a P1
progress checkpoint through owner archive APIs, disables learning/writeback for
the replay, and writes a self-contained HTML visualization for local preview.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from volvence_ant.env.ant_world import AntWorldConfig
from volvence_ant.env.colony import ColonyWorld
from volvence_ant.env.pheromone_field import PheromoneBus
from volvence_ant.env.world_objects import ButterSource
from volvence_ant.evidence.runtime_profile import (
    ant_runtime_replay_rollout_config,
)
from volvence_ant.experiments.ecology_p1 import (
    EcologyP1Config,
    _fixed_schedule,
    _load_arm_progress,
    _read_progress_archive,
    _schedule_digest,
)
from volvence_ant.runtime import (
    AntObjectiveKind,
    AntSenseSchema,
    AntSessionConfig,
    KernelColonyRunner,
)
from volvence_ant.viz.colony_theater import (
    ColonyTheaterReport,
    TheaterAntFrame,
    TheaterArmReplay,
    TheaterRoundFrame,
    _run_heuristic_arm,
    write_colony_theater_html,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PROGRESS_DIR = Path("research/ant/results/.partials/ecology_p1_v31/seed0")
_DEFAULT_OUT = Path("research/ant/figures/digital_ant_checkpoint_theater.html")
_FOOD_START = (6.0, 0.0)
_FOOD_MOVED = (-4.0, 4.0)
_FIELD_SPAN = 24.0
_CELL_SIZE = 1.0


def _resolve_repo_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else _REPO_ROOT / path
    resolved.relative_to(_REPO_ROOT)
    return resolved


def _load_config_from_progress(progress_dir: Path, *, arm: str) -> EcologyP1Config:
    state_path = progress_dir / f"{arm}.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError(f"P1 progress for {arm} has no config object")
    return EcologyP1Config(
        n_ants=int(raw_config["n_ants"]),
        temporal_latent_dim=int(raw_config["temporal_latent_dim"]),
        training_rounds=int(raw_config["training_rounds"]),
        evaluation_rounds=int(raw_config["evaluation_rounds"]),
        layouts_per_tier=int(raw_config["layouts_per_tier"]),
        seed=int(raw_config["seed"]),
        layout_success_ratio=float(raw_config["layout_success_ratio"]),
        body_success_ratio=float(raw_config["body_success_ratio"]),
        harmful_tick_rate_max=float(raw_config["harmful_tick_rate_max"]),
    )


def _checkpoint_archives(
    *,
    progress_dir: Path,
    arm: str,
    config: EcologyP1Config,
) -> tuple[tuple[bytes, ...], dict[str, object]]:
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
    return archives, state


def _checkpoint_world(*, seed: int, n_ants: int) -> ColonyWorld:
    return ColonyWorld(
        config=AntWorldConfig(seed=seed, antenna_offset_deg=30.0, antenna_reach=0.9),
        world_objects=(
            ButterSource(
                object_id="butter-preview",
                x=_FOOD_START[0],
                y=_FOOD_START[1],
                strength=1.6,
                decay=5.0,
                radius=1.6,
            ),
        ),
        n_bodies=n_ants,
        bus=PheromoneBus(
            width=_FIELD_SPAN,
            height=_FIELD_SPAN,
            cell_size=_CELL_SIZE,
            decay=0.02,
            deposit_amount=2.0,
        ),
    )


def _assemble_checkpoint_frame(
    world: ColonyWorld,
    modes: list[str],
) -> TheaterRoundFrame:
    ants = tuple(
        TheaterAntFrame(
            x=round(body.x, 4),
            y=round(body.y, 4),
            heading=round(body.heading, 4),
            carrying=body.carrying_food,
            mode=modes[body_id],
        )
        for body_id, body in (
            (index, world.body(index)) for index in range(world.n_bodies)
        )
    )
    food = tuple(
        (round(item.x, 4), round(item.y, 4))
        for item in world.world_objects()
        if isinstance(item, ButterSource) and item.remaining > 0.0
    )
    trail = tuple(
        tuple(round(float(value), 4) for value in row)
        for row in world.pheromone.trail
    )
    return TheaterRoundFrame(
        tick=world.tick,
        delivered=world.food_delivered,
        ants=ants,
        food=food,
        trail=trail,
    )


async def _run_checkpoint_arm(
    *,
    config: EcologyP1Config,
    archives: tuple[bytes, ...],
    rounds: int,
    relocate_at: int,
    seed: int,
    label: str,
) -> TheaterArmReplay:
    world = _checkpoint_world(seed=seed, n_ants=config.n_ants)
    session_config = AntSessionConfig(
        temporal_latent_dim=config.temporal_latent_dim,
        session_id=f"checkpoint-theater:{seed}",
        seed=seed,
        rollout_config=ant_runtime_replay_rollout_config(
            enable_sparse_exploration=False,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
        ),
        joint_apply_writeback=False,
        joint_apply_policy_optimization=False,
        joint_learning_enabled=False,
        objective=AntObjectiveKind.ECOLOGY,
        sense_schema=AntSenseSchema.ECOLOGY_V2,
    )
    runner = KernelColonyRunner(world, base_config=session_config)
    runner.restore_learning_checkpoint_archives(archives)
    frames: list[TheaterRoundFrame] = []
    for round_index in range(rounds):
        if round_index == relocate_at:
            world.move_world_object(
                "butter-preview",
                delta_x=_FOOD_MOVED[0] - _FOOD_START[0],
                delta_y=_FOOD_MOVED[1] - _FOOD_START[1],
            )
        record = await runner.step_round()
        modes = [step.abstract_action for step in record.ant_steps]
        frames.append(_assemble_checkpoint_frame(world, modes))
    return TheaterArmReplay(
        label=label,
        kind="digital-life",
        frames=tuple(frames),
    )


async def _run(args: argparse.Namespace) -> int:
    progress_dir = _resolve_repo_path(args.progress_dir)
    out_path = _resolve_repo_path(args.out)
    config = _load_config_from_progress(progress_dir, arm=args.arm)
    archives, state = _checkpoint_archives(
        progress_dir=progress_dir,
        arm=args.arm,
        config=config,
    )
    relocate_at = args.relocate_at if args.relocate_at is not None else args.rounds // 2
    heuristic = _run_heuristic_arm(
        n_ants=config.n_ants,
        rounds=args.rounds,
        relocate_at=relocate_at,
        seed=args.seed,
    )
    checkpoint = await _run_checkpoint_arm(
        config=config,
        archives=archives,
        rounds=args.rounds,
        relocate_at=relocate_at,
        seed=args.seed,
        label=(
            f"P1 journal checkpoint ({args.arm}, "
            f"ep {state['completed_training_episodes']})"
        ),
    )
    report = ColonyTheaterReport(
        arms=(heuristic, checkpoint),
        nest=(0.0, 0.0),
        n_ants=config.n_ants,
        rounds=args.rounds,
        relocate_at=relocate_at,
        field_span=_FIELD_SPAN,
        cell_size=_CELL_SIZE,
        html_path=None,
    )
    written = write_colony_theater_html(report=report, out_path=out_path)
    summary = {
        "html": str(written.relative_to(_REPO_ROOT)),
        "arm": args.arm,
        "archives_loaded": len(archives),
        "completed_training_episodes": state["completed_training_episodes"],
        "training_complete": state.get("training_complete"),
        "rounds": args.rounds,
        "relocate_at": relocate_at,
        "checkpoint_sha256": state.get("checkpoint_sha256"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-dir", type=Path, default=_DEFAULT_PROGRESS_DIR)
    parser.add_argument("--arm", type=str, default="learned")
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--relocate-at", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
