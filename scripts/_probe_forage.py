"""Throwaway probe: does the learned ant get a real foraging gradient now?

Trains one learned AntSession on the training map, evaluates on the held-out
map, and reports pickups/deliveries plus whether the temporal owner parameters
actually moved (i.e. the reward seam now carries a signal).
"""

import asyncio

from volvence_zero.joint_loop import JointLoopSchedule

from volvence_ant.env.ant_world import AntWorld, AntWorldConfig, FoodSource
from volvence_ant.runtime.ant_session import AntSession, AntSessionConfig


def _train_world(seed: int) -> AntWorld:
    return AntWorld(
        config=AntWorldConfig(seed=seed),
        food_sources=(FoodSource(x=6.0, y=0.0, strength=1.0, decay=5.0),),
    )


def _heldout_world(seed: int) -> AntWorld:
    return AntWorld(
        config=AntWorldConfig(seed=seed),
        food_sources=(FoodSource(x=0.0, y=6.0, strength=1.0, decay=5.0),),
    )


def _cfg(seed: int, n_z: int) -> AntSessionConfig:
    return AntSessionConfig(
        temporal_latent_dim=n_z,
        seed=seed,
        external_prediction_error_drive=True,
        joint_schedule=JointLoopSchedule(ssl_interval=1, rl_interval=3),
        joint_apply_writeback=True,
    )


async def main() -> None:
    seed, n_z, train_ticks, eval_ticks = 0, 16, 120, 80
    cfg = _cfg(seed, n_z)

    boot = AntSession(_train_world(seed), config=cfg)
    initial = boot.export_learning_checkpoint(checkpoint_id="init")

    train = AntSession(_train_world(seed), config=cfg)
    train.restore_learning_checkpoint(initial)
    await train.run(train_ticks)
    trained = train.export_learning_checkpoint(checkpoint_id="trained")
    print(
        f"[train] pickups={train.world.food_pickups} delivered={train.world.food_delivered} "
        f"tparams_changed={initial.temporal_fingerprint != trained.temporal_fingerprint} "
        f"mem_changed={initial.memory_fingerprint != trained.memory_fingerprint}"
    )

    world = _heldout_world(seed)
    ev = AntSession(world, config=cfg)
    ev.restore_learning_checkpoint(trained)
    recs = await ev.run(eval_ticks)
    maxfood = max((world.food_intensity(r.x, r.y) for r in recs), default=0.0)
    maxdist = max((abs(complex(r.x, r.y)) for r in recs), default=0.0)
    print(
        f"[eval ] pickups={world.food_pickups} delivered={world.food_delivered} "
        f"max_food={maxfood:.3f} max_dist={maxdist:.2f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
