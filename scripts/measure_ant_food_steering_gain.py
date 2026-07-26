"""Read-only food-steering transfer-function probe.

Measures, for cold and (optionally) a journalled learned checkpoint, the
``food_diff -> turn_command`` response at near and medium range using the SAME
``AntSession.step`` path the live app and the ecology probes use. Nothing here
feeds learning: it restores frozen checkpoints, steps one turn, and reads the
published motor command.

Purpose (diagnostic (B) in the medium-blocker analysis): decide whether the
bounded action head can produce enough correctly-signed steering authority to
overcome the cold/baseline same-direction turn, at near AND at medium range. A
near-only alignment with medium failure points the fix at the frozen sensory
representation / rare-heavy substrate refresh rather than at more online
controller plumbing.

Usage:

    python scripts/measure_ant_food_steering_gain.py \
        --progress-dir research/ant/results/.partials/ecology_p1/seed0

Omit ``--progress-dir`` to measure the cold shared-initial arm only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
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
    _hydrate_progress_checkpoints,
    _load_arm_progress,
    _read_progress_archive,
    _schedule_digest,
    _fixed_schedule,
)
from volvence_ant.runtime import (
    AntLearningCheckpoint,
    AntObjectiveKind,
    AntSenseSchema,
    AntSession,
    AntSessionConfig,
    KernelColonyRunner,
)

_ROOT = Path(__file__).resolve().parents[1]

# Near mirrors the standing food probe geometry (0.6, +-0.35); medium mirrors
# the curriculum medium band centre (2.1-2.9).
_NEAR_DISTANCE = math.hypot(0.6, 0.35)
_MEDIUM_DISTANCE = 2.5
_LATERAL_RAD = math.atan2(0.35, 0.6)
_TURN_THRESHOLD = 1e-4


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
            session_id="food-gain:cold-initial",
            optimize=True,
        ),
    )
    return bootstrap.export_learning_checkpoints(
        checkpoint_prefix="food-gain:cold-initial",
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
        raise FileNotFoundError(
            f"no journalled arm {arm!r} under {progress_dir}"
        )
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


async def _turn_for_food(
    *,
    checkpoint: AntLearningCheckpoint,
    temporal_latent_dim: int,
    seed: int,
    x: float,
    y: float,
    session_id: str,
) -> tuple[float, tuple[float, ...], float, float, float]:
    world = AntWorld(
        config=AntWorldConfig(seed=seed, step_size=0.4),
        world_objects=(ButterSource(object_id="probe-butter", x=x, y=y),),
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
            ),
            objective=AntObjectiveKind.ECOLOGY,
            sense_schema=AntSenseSchema.ECOLOGY_V2,
        ),
    )
    session.restore_learning_checkpoint(checkpoint)
    world.set_body_pose(x=0.0, y=0.0, heading=0.0)
    observation = world.observe()
    record = await session.step()
    return (
        float(record.command.turn_command),
        tuple(float(v) for v in record.code),
        float(observation.food_left),
        float(observation.food_right),
        float(world.config.max_turn_rate),
    )


async def _measure_body(
    *,
    arm: str,
    body_id: int,
    checkpoint: AntLearningCheckpoint,
    temporal_latent_dim: int,
    seed: int,
    distance: float,
) -> dict[str, object]:
    sides = {
        "center": (distance, 0.0),
        "left": (
            distance * math.cos(_LATERAL_RAD),
            distance * math.sin(_LATERAL_RAD),
        ),
        "right": (
            distance * math.cos(_LATERAL_RAD),
            -distance * math.sin(_LATERAL_RAD),
        ),
    }
    turns: dict[str, float] = {}
    codes: dict[str, tuple[float, ...]] = {}
    food_diffs: dict[str, float] = {}
    max_turn = 0.0
    for label, (x, y) in sides.items():
        turn, code, food_left, food_right, max_turn = await _turn_for_food(
            checkpoint=checkpoint,
            temporal_latent_dim=temporal_latent_dim,
            seed=seed,
            x=x,
            y=y,
            session_id=f"food-gain:{arm}:{body_id}:{distance:.3f}:{label}",
        )
        turns[label] = turn
        codes[label] = code
        food_diffs[label] = food_left - food_right
    baseline = turns["center"]
    turn_left = turns["left"]
    turn_right = turns["right"]
    # Positive turn = left (motor_decode atan2(z1-z0, 1+z0+z1)).
    aligned_left = turn_left > _TURN_THRESHOLD
    aligned_right = turn_right < -_TURN_THRESHOLD
    return {
        "body_id": body_id,
        "distance": distance,
        "baseline_turn": baseline,
        "turn_left": turn_left,
        "turn_right": turn_right,
        # Authority = how far the food signal moves the turn against baseline,
        # per side; the weaker side is what a full approach depends on.
        "authority_left": turn_left - baseline,
        "authority_right": baseline - turn_right,
        "min_authority": min(turn_left - baseline, baseline - turn_right),
        "food_diff_left": food_diffs["left"],
        "food_diff_right": food_diffs["right"],
        "aligned": bool(aligned_left and aligned_right),
        "aligned_left": bool(aligned_left),
        "aligned_right": bool(aligned_right),
        "max_turn_rate": max_turn,
        "code_left": list(codes["left"]),
        "code_right": list(codes["right"]),
        "code_center": list(codes["center"]),
    }


async def _measure_arm(
    *,
    arm: str,
    checkpoints: tuple[AntLearningCheckpoint, ...],
    temporal_latent_dim: int,
    seed: int,
) -> dict[str, object]:
    tiers = {"near": _NEAR_DISTANCE, "medium": _MEDIUM_DISTANCE}
    per_tier: dict[str, list[dict[str, object]]] = {}
    for tier_name, distance in tiers.items():
        rows = [
            await _measure_body(
                arm=arm,
                body_id=body_id,
                checkpoint=checkpoint,
                temporal_latent_dim=temporal_latent_dim,
                seed=seed,
                distance=distance,
            )
            for body_id, checkpoint in enumerate(checkpoints)
        ]
        per_tier[tier_name] = rows
    return {"arm": arm, "tiers": per_tier}


def _summarize(arm_report: dict[str, object]) -> None:
    arm = arm_report["arm"]
    tiers = arm_report["tiers"]
    assert isinstance(tiers, dict)
    print(f"\n=== arm: {arm} ===")
    for tier_name, rows in tiers.items():
        assert isinstance(rows, list)
        aligned = sum(bool(row["aligned"]) for row in rows)
        total = len(rows)
        baseline_absmax = max(abs(float(row["baseline_turn"])) for row in rows)
        min_authorities = [float(row["min_authority"]) for row in rows]
        food_diff_absmax = max(
            abs(float(row["food_diff_left"])) for row in rows
        )
        print(
            f"  [{tier_name}] aligned_bodies={aligned}/{total}  "
            f"baseline_bias(|max|)={baseline_absmax:.4f}  "
            f"min_authority(range)="
            f"[{min(min_authorities):.4f},{max(min_authorities):.4f}]  "
            f"food_diff(|max|)={food_diff_absmax:.4f}"
        )
        for row in rows:
            print(
                f"    body{row['body_id']}: "
                f"turnL={float(row['turn_left']):+.4f} "
                f"turnR={float(row['turn_right']):+.4f} "
                f"base={float(row['baseline_turn']):+.4f} "
                f"authL={float(row['authority_left']):+.4f} "
                f"authR={float(row['authority_right']):+.4f} "
                f"aligned={'Y' if row['aligned'] else 'n'}"
            )


def _verdict(reports: list[dict[str, object]]) -> str:
    lines: list[str] = []
    by_arm = {report["arm"]: report for report in reports}
    for arm, report in by_arm.items():
        tiers = report["tiers"]
        assert isinstance(tiers, dict)
        near_rows = tiers["near"]
        medium_rows = tiers["medium"]
        assert isinstance(near_rows, list) and isinstance(medium_rows, list)
        near_a = sum(bool(r["aligned"]) for r in near_rows)
        near_n = len(near_rows)
        med_a = sum(bool(r["aligned"]) for r in medium_rows)
        med_n = len(medium_rows)
        if near_a == 0:
            lines.append(
                f"- {arm}: 近场 food 转向 {near_a}/{near_n} 对齐 —— "
                "连近场都没学到食物梯度转向;near 的 pickup 是巡游巧合,"
                "不是能力。这与逐版本 food probe 0/4 一致。"
            )
        elif med_a < math.ceil(med_n * 0.6):
            lines.append(
                f"- {arm}: 近场 {near_a}/{near_n} 对齐但 medium {med_a}/{med_n} —— "
                "能力在近场存在,却在 range 上不足;指向冻结感觉表征/基线偏置,"
                "而非再拆控制器衰减。应走 offline 表征刷新(C 的基底侧)。"
            )
        else:
            lines.append(
                f"- {arm}: 近场 {near_a}/{near_n}、medium {med_a}/{med_n} 均达标 —— "
                "food 转向已具备,medium 失败(若仍失败)另有原因(奖励稀疏/闭环)。"
            )
    if "cold" in by_arm and "learned" in by_arm:
        for tier in ("near", "medium"):
            cold_rows = by_arm["cold"]["tiers"][tier]  # type: ignore[index]
            learned_rows = by_arm["learned"]["tiers"][tier]  # type: ignore[index]
            assert isinstance(cold_rows, list) and isinstance(learned_rows, list)
            cold_auth = max(float(r["min_authority"]) for r in cold_rows)
            learned_auth = max(float(r["min_authority"]) for r in learned_rows)
            lines.append(
                f"- [{tier}] learned vs cold 的 min_authority(|max|): "
                f"cold={cold_auth:+.4f}, learned={learned_auth:+.4f} —— "
                + (
                    "learning 提升了转向权威。"
                    if learned_auth > cold_auth + _TURN_THRESHOLD
                    else "learning 几乎没有增加转向权威(有界头天花板可能已到)。"
                )
            )
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    config = EcologyP1Config(
        n_ants=args.n_ants,
        temporal_latent_dim=args.temporal_latent_dim,
        seed=args.seed,
    )
    reports: list[dict[str, object]] = []
    cold = _cold_initial(config)
    reports.append(
        await _measure_arm(
            arm="cold",
            checkpoints=cold,
            temporal_latent_dim=config.temporal_latent_dim,
            seed=config.seed + 700_003,
        )
    )
    completed = None
    if args.progress_dir is not None:
        progress_dir = (
            args.progress_dir
            if args.progress_dir.is_absolute()
            else _ROOT / args.progress_dir
        ).resolve()
        learned, completed = _load_arm(
            progress_dir=progress_dir,
            arm=args.arm,
            config=config,
        )
        reports.append(
            await _measure_arm(
                arm=args.arm,
                checkpoints=learned,
                temporal_latent_dim=config.temporal_latent_dim,
                seed=config.seed + 700_003,
            )
        )
    for report in reports:
        _summarize(report)
    print("\n=== 天花板判据 ===")
    if completed is not None:
        print(f"(learned arm 已训练 {completed} 个 work item)")
    print(_verdict(reports))
    if args.json_out is not None:
        out = (
            args.json_out
            if args.json_out.is_absolute()
            else _ROOT / args.json_out
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(f"\nreport: {out.relative_to(_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only food_diff -> turn transfer-function probe"
    )
    parser.add_argument("--n-ants", type=int, default=4)
    parser.add_argument("--temporal-latent-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--progress-dir",
        type=Path,
        default=None,
        help="Journal dir to load a learned arm from (optional).",
    )
    parser.add_argument("--arm", type=str, default="learned")
    parser.add_argument("--json-out", type=Path, default=None)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
