"""Read-only post-pickup temporal-switch and family-persistence probe.

The script restores the journalled ecology checkpoint and reuses the formal
frozen ±135-degree U-turn probe. It never enables joint learning or policy
optimization. Each real pickup yields one lane with:

* latency to the first post-pickup temporal switch;
* the exact owner-published action family selected by that switch;
* consecutive actions for which that family remains selected.

Usage:

    python scripts/measure_ant_post_pickup_family_persistence.py \
        --progress-dir research/ant/results/.partials/ecology_p1_v31/seed0 \
        --json-out research/ant/results/.partials/ecology_p1_v31/\
post_pickup_family_persistence.station1.json
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path

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
    ECOLOGY_POST_PICKUP_MIN_FAMILY_PERSISTENCE_ACTIONS,
    ECOLOGY_POST_PICKUP_UTURN_MAX_SWITCH_LATENCY,
    EcologyCheckpointPostPickupUTurnProbe,
    run_ecology_checkpoint_post_pickup_uturn_probes,
)
from volvence_ant.runtime import AntLearningCheckpoint


_ROOT = Path(__file__).resolve().parents[1]
def _load_arm(
    *,
    progress_dir: Path,
    arm: str,
    config: EcologyP1Config,
) -> tuple[tuple[AntLearningCheckpoint, ...], int, str]:
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
    checkpoints = _hydrate_progress_checkpoints(
        config=config,
        curriculum=_curriculum_config(config),
        archives=_read_progress_archive(
            progress_dir=progress_dir,
            state=state,
            config=config,
        ),
        arm=arm,
    )
    return (
        checkpoints,
        int(state["completed_training_episodes"]),
        str(state["checkpoint_sha256"]),
    )


def _summary(
    probes: tuple[EcologyCheckpointPostPickupUTurnProbe, ...],
) -> dict[str, object]:
    lanes = [lane for probe in probes for lane in probe.lanes]
    pickup_lanes = [lane for lane in lanes if lane.picked_up]
    prompt_switches = [
        lane
        for lane in pickup_lanes
        if lane.first_post_pickup_switch_step is not None
        and lane.first_post_pickup_switch_step
        <= ECOLOGY_POST_PICKUP_UTURN_MAX_SWITCH_LATENCY
    ]
    persistent = [
        lane
        for lane in prompt_switches
        if lane.post_switch_family_survival_actions
        >= ECOLOGY_POST_PICKUP_MIN_FAMILY_PERSISTENCE_ACTIONS
    ]
    latencies = [
        int(lane.first_post_pickup_switch_step)
        for lane in pickup_lanes
        if lane.first_post_pickup_switch_step is not None
    ]
    survivals = [
        lane.post_switch_family_survival_actions
        for lane in prompt_switches
    ]
    switch_rate = (
        len(prompt_switches) / len(pickup_lanes)
        if pickup_lanes
        else 0.0
    )
    persistence_rate = (
        len(persistent) / len(prompt_switches)
        if prompt_switches
        else 0.0
    )
    fingerprints_stable = all(
        lane.policy_fingerprint_stable
        and lane.temporal_learning_fingerprint_stable
        for lane in lanes
    )
    accepted = bool(
        pickup_lanes
        and len(pickup_lanes) == len(lanes)
        and switch_rate == 1.0
        and persistence_rate == 1.0
        and fingerprints_stable
    )
    return {
        "lane_count": len(lanes),
        "pickup_lane_count": len(pickup_lanes),
        "prompt_switch_count": len(prompt_switches),
        "prompt_switch_rate": switch_rate,
        "switch_latency_actions": latencies,
        "persistent_family_count": len(persistent),
        "family_persistence_rate": persistence_rate,
        "family_survival_actions": survivals,
        "minimum_family_survival_actions": (
            min(survivals) if survivals else None
        ),
        "right_censored_family_count": sum(
            lane.post_switch_family_observation_censored
            for lane in prompt_switches
        ),
        "fingerprints_stable": fingerprints_stable,
        "acceptance_thresholds": {
            "maximum_switch_latency_actions": (
                ECOLOGY_POST_PICKUP_UTURN_MAX_SWITCH_LATENCY
            ),
            "minimum_family_persistence_actions": (
                ECOLOGY_POST_PICKUP_MIN_FAMILY_PERSISTENCE_ACTIONS
            ),
            "required_switch_rate": 1.0,
            "required_persistence_rate": 1.0,
        },
        "accepted": accepted,
    }


async def _run(args: argparse.Namespace) -> int:
    config = EcologyP1Config(
        n_ants=args.n_ants,
        temporal_latent_dim=args.temporal_latent_dim,
        seed=args.seed,
    )
    progress_dir = args.progress_dir.resolve()
    checkpoints, completed, checkpoint_sha256 = _load_arm(
        progress_dir=progress_dir,
        arm=args.arm,
        config=config,
    )
    probes = await run_ecology_checkpoint_post_pickup_uturn_probes(
        temporal_latent_dim=config.temporal_latent_dim,
        seed=config.seed + args.probe_seed_offset,
        checkpoints=checkpoints,
    )
    summary = _summary(probes)
    payload = {
        "schema_version": "ant-post-pickup-family-persistence.v1",
        "arm": args.arm,
        "completed_training_episodes": completed,
        "checkpoint_sha256": checkpoint_sha256,
        "probe_seed": config.seed + args.probe_seed_offset,
        "summary": summary,
        "probes": [asdict(probe) for probe in probes],
    }
    print(
        f"arm={args.arm} episodes={completed} "
        f"switch_rate={float(summary['prompt_switch_rate']):.1%} "
        "min_persistence="
        f"{summary['minimum_family_survival_actions']} "
        f"accepted={summary['accepted']}"
    )
    for probe in probes:
        for lane in probe.lanes:
            print(
                f"  body={probe.body_id} side={lane.side} "
                f"switch={lane.first_post_pickup_switch_step} "
                f"family={lane.first_post_pickup_switch_family} "
                "survival="
                f"{lane.post_switch_family_survival_actions} "
                "censored="
                f"{lane.post_switch_family_observation_censored}"
            )
    if args.json_out is not None:
        output = (
            args.json_out
            if args.json_out.is_absolute()
            else _ROOT / args.json_out
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"report: {output.relative_to(_ROOT)}")
    return 0 if bool(summary["accepted"]) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure post-pickup temporal-switch latency and exact action-"
            "family persistence from a frozen ecology journal checkpoint"
        )
    )
    parser.add_argument("--n-ants", type=int, default=4)
    parser.add_argument("--temporal-latent-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-dir", type=Path, required=True)
    parser.add_argument("--arm", type=str, default="learned")
    parser.add_argument("--probe-seed-offset", type=int, default=700_003)
    parser.add_argument("--json-out", type=Path, default=None)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
