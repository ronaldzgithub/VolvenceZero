"""Read-only per-channel steering-gain audit for the causal action head.

Under exclusive steering the opponent-coded pair ``z[0]/z[1]`` is ONE actuator
axis owned solely by the causal action head, so every steering drive -- food
gradient, heat avoidance, carrying-home bearing -- has to be expressed on that
single axis. The v24 transfer-function measurement showed the food differential
is real and correctly signed on 8/8 body-tier combinations, yet roughly half the
size of the residual turn present when food is centred. This script asks the
follow-up question directly: how much contrast does each channel buy per unit of
sensory contrast, and does food sit below its peers?

It reuses the published paired probes (``run_ecology_checkpoint_action_probes``)
so the numbers come from the same truth the curriculum gates read. Nothing here
feeds learning: frozen checkpoints are restored, one turn is stepped, and the
published code / residual / motor command are read.

Usage:

    python scripts/measure_ant_steering_channel_gain.py \
        --progress-dir research/ant/results/.partials/ecology_p1_v24/seed0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from volvence_ant.evidence.runtime_profile import (
    ANT_CAUSAL_ACTION_HEAD_CONTRAST_PAIRS,
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
from volvence_ant.experiments.ecology_probe import (
    EcologyActionProbe,
    run_ecology_checkpoint_action_probes,
)
from volvence_ant.runtime import AntLearningCheckpoint


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


def _contrast(values: tuple[float, ...], pair: tuple[int, int]) -> float:
    """Antisymmetric component of one opponent-coded actuator axis."""

    return float(values[pair[0]]) - float(values[pair[1]])


def _sensor_contrast(probe: EcologyActionProbe) -> float:
    return sum(
        abs(left - right)
        for left, right in zip(
            probe.left_sensor_pair,
            probe.right_sensor_pair,
            strict=True,
        )
    )


def _channel_row(
    *,
    probe: EcologyActionProbe,
    pair: tuple[int, int],
) -> dict[str, object]:
    sensor_contrast = _sensor_contrast(probe)
    code_swing = _contrast(probe.left_code, pair) - _contrast(
        probe.right_code, pair
    )
    if probe.left_action_head_residual and probe.right_action_head_residual:
        head_left = _contrast(probe.left_action_head_residual, pair)
        head_right = _contrast(probe.right_action_head_residual, pair)
        head_swing = head_left - head_right
        # Head contribution that survives when the paired stimulus cancels.
        # Under exclusive steering the head is the only writer of this axis,
        # so this term should account for the whole residual turn offset; if
        # it does not, some other producer is still reaching the axis.
        head_offset = 0.5 * (head_left + head_right)
    else:
        head_swing = 0.0
        head_offset = 0.0
    # Offset shared by both lanes: what the axis holds when the paired
    # stimulus cancels. This is the term the absolute-alignment gate has to
    # be beaten by.
    code_offset = 0.5 * (
        _contrast(probe.left_code, pair) + _contrast(probe.right_code, pair)
    )
    turn_swing = float(probe.left_turn) - float(probe.right_turn)
    turn_offset = 0.5 * (float(probe.left_turn) + float(probe.right_turn))
    return {
        "kind": probe.kind.value,
        "sensor_contrast": sensor_contrast,
        "code_swing": code_swing,
        "head_swing": head_swing,
        "head_offset": head_offset,
        "code_offset": code_offset,
        "turn_swing": turn_swing,
        "turn_offset": turn_offset,
        "code_gain": (
            code_swing / sensor_contrast if sensor_contrast > 0.0 else 0.0
        ),
        "turn_gain": (
            turn_swing / sensor_contrast if sensor_contrast > 0.0 else 0.0
        ),
        "swing_over_offset": (
            abs(turn_swing) / abs(turn_offset)
            if abs(turn_offset) > 0.0
            else float("inf")
        ),
        # These probes mirror ONE stimulus while holding home/PI/history
        # fixed. Their shared offset therefore includes legitimate steering
        # from all other channels and is not the symmetric component under a
        # full-world reflection. Keep the ratio as a channel-competition
        # diagnostic only; the temporal owner's formal mirror contract uses
        # the complete signed sense permutation.
        "stimulus_pair_offset_ratio": (
            abs(head_offset) / abs(0.5 * head_swing)
            if abs(head_swing) > 0.0
            else float("inf")
        ),
        "target_aligned": bool(probe.target_aligned),
        "input_reachable": bool(probe.input_reachable),
        "update_step": int(probe.left_action_head_update_step),
    }


async def _run(args: argparse.Namespace) -> int:
    config = EcologyP1Config(
        n_ants=args.n_ants,
        temporal_latent_dim=args.temporal_latent_dim,
        seed=args.seed,
    )
    pairs = ANT_CAUSAL_ACTION_HEAD_CONTRAST_PAIRS
    if not pairs:
        raise ValueError(
            "steering-channel audit requires a profile with contrast pairs"
        )
    pair = pairs[0]
    checkpoints, completed = _load_arm(
        progress_dir=args.progress_dir.resolve(),
        arm=args.arm,
        config=config,
    )
    reports = await run_ecology_checkpoint_action_probes(
        temporal_latent_dim=config.temporal_latent_dim,
        # Keep this diagnostic on the exact frozen probe distribution consumed
        # by the P1 gates. ``config.seed`` identifies the resumable training
        # journal and must not be repurposed as the probe-layout seed.
        seed=config.seed + 700_003,
        checkpoints=checkpoints,
    )
    payload: list[dict[str, object]] = []
    print(f"\n=== arm={args.arm} episodes={completed} pair=z{pair} ===")
    header = (
        f"{'body':>4} {'channel':<9} {'sensorΔ':>9} {'codeΔ':>10} "
        f"{'headΔ':>10} {'head_off':>10} {'code_off':>10} {'turnΔ':>9} "
        f"{'turn_off':>9} {'Δ/off':>7} {'aligned':>8}"
    )
    print(header)
    for report in reports:
        for probe in report.probes:
            row = _channel_row(probe=probe, pair=pair)
            row["body_id"] = report.body_id
            payload.append(row)
            ratio = row["swing_over_offset"]
            ratio_text = "inf" if ratio == float("inf") else f"{ratio:.2f}"
            print(
                f"{report.body_id:>4} {row['kind']:<9} "
                f"{row['sensor_contrast']:>9.4f} {row['code_swing']:>+10.5f} "
                f"{row['head_swing']:>+10.5f} {row['head_offset']:>+10.5f} "
                f"{row['code_offset']:>+10.5f} {row['turn_swing']:>+9.5f} "
                f"{row['turn_offset']:>+9.5f} "
                f"{ratio_text:>7} {'Y' if row['target_aligned'] else 'n':>8}"
            )
    print("\n=== 通道汇总(四体中位数) ===")
    summary: dict[str, dict[str, float]] = {}
    for kind in sorted({str(row["kind"]) for row in payload}):
        rows = [row for row in payload if row["kind"] == kind]
        entry = {
            "sensor_contrast": statistics.median(
                float(row["sensor_contrast"]) for row in rows
            ),
            "abs_turn_swing": statistics.median(
                abs(float(row["turn_swing"])) for row in rows
            ),
            "abs_turn_offset": statistics.median(
                abs(float(row["turn_offset"])) for row in rows
            ),
            "abs_turn_gain": statistics.median(
                abs(float(row["turn_gain"])) for row in rows
            ),
            "abs_head_offset": statistics.median(
                abs(float(row["head_offset"])) for row in rows
            ),
            "abs_code_offset": statistics.median(
                abs(float(row["code_offset"])) for row in rows
            ),
            "stimulus_pair_offset_ratio": statistics.median(
                float(row["stimulus_pair_offset_ratio"]) for row in rows
            ),
            "abs_code_gain": statistics.median(
                abs(float(row["code_gain"])) for row in rows
            ),
            "aligned_bodies": float(
                sum(1 for row in rows if row["target_aligned"])
            ),
        }
        summary[kind] = entry
        print(
            f"  {kind:<9} sensorΔ={entry['sensor_contrast']:.4f} "
            f"|turnΔ|={entry['abs_turn_swing']:.5f} "
            f"|turn_off|={entry['abs_turn_offset']:.5f} "
            f"|head_off|={entry['abs_head_offset']:.5f} "
            f"|code_off|={entry['abs_code_offset']:.5f} "
            f"turn_gain={entry['abs_turn_gain']:.5f} "
            f"pair_offset/swing={entry['stimulus_pair_offset_ratio']:.2f} "
            f"aligned={int(entry['aligned_bodies'])}/{len(rows)}"
        )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "arm": args.arm,
                    "completed_training_episodes": completed,
                    "contrast_pair": list(pair),
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
        description="Read-only per-channel steering-gain audit"
    )
    parser.add_argument("--n-ants", type=int, default=4)
    parser.add_argument("--temporal-latent-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-dir", type=Path, required=True)
    parser.add_argument("--arm", type=str, default="learned")
    parser.add_argument("--json-out", type=Path, default=None)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
