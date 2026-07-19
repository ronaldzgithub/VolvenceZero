"""Phase 1 tests: pheromone snapshot bus + stigmergic colony foraging."""

from __future__ import annotations

import pytest

from volvence_ant.env.colony import ColonyWorld
from volvence_ant.env.pheromone_field import PheromoneBus
from volvence_ant.experiments import (
    colony_foraging_experiment,
    kernel_colony_foraging_experiment,
)
from volvence_ant.runtime import AntSessionConfig, KernelColonyRunner


def test_published_field_is_immutable() -> None:
    bus = PheromoneBus()
    snapshot = bus.snapshot
    with pytest.raises(ValueError):
        snapshot.trail[0, 0] = 1.0  # read-only published grid
    with pytest.raises(ValueError):
        snapshot.home[0, 0] = 1.0


def test_deposits_are_additive_not_overwriting() -> None:
    # two independent writers at the same location must ADD, not overwrite.
    # (sampling applies a consistent bilinear fraction, so compare the ratio.)
    one = PheromoneBus(decay=0.0, deposit_amount=1.0)
    one.deposit(x=0.0, y=0.0, trail_amount=1.0)
    one.advance()
    single = one.snapshot.sample(0.0, 0.0)[1]

    two = PheromoneBus(decay=0.0, deposit_amount=1.0)
    two.deposit(x=0.0, y=0.0, trail_amount=1.0)
    two.deposit(x=0.0, y=0.0, trail_amount=1.0)
    two.advance()
    doubled = two.snapshot.sample(0.0, 0.0)[1]

    assert single > 0.0
    assert doubled == pytest.approx(2.0 * single, rel=1e-6)


def test_decay_reduces_mass() -> None:
    bus = PheromoneBus(decay=0.5, deposit_amount=1.0)
    bus.deposit(x=0.0, y=0.0, trail_amount=2.0)
    bus.advance()
    mass_after_deposit = bus.total_mass()[1]
    bus.advance()  # decay, no new deposits
    assert bus.total_mass()[1] == pytest.approx(mass_after_deposit * 0.5, rel=1e-6)


def test_snapshot_advances_tick() -> None:
    bus = PheromoneBus()
    t0 = bus.snapshot.tick
    bus.advance()
    assert bus.snapshot.tick == t0 + 1


def test_colony_reads_bus_but_plain_world_has_no_pheromone() -> None:
    plain = ColonyWorld(n_bodies=2, bus=PheromoneBus())
    # before any deposit the field is zero everywhere
    assert plain.pheromone.sample(0.0, 0.0) == (0.0, 0.0)


def test_bus_helps_collective_foraging() -> None:
    result = colony_foraging_experiment(n_ants=20, rounds=700, seed=0)
    assert result.bus_helps  # with_bus delivered >= without_bus
    # seed 0 establishes a trail corridor -> recruitment fires + a real lift
    assert result.trail_follow_events > 0
    assert result.with_bus_delivered > result.without_bus_delivered


async def test_kernel_colony_uses_isolated_sessions_and_bus_ablation() -> None:
    world = ColonyWorld(n_bodies=2, bus=PheromoneBus(decay=0.0))
    runner = KernelColonyRunner(
        world,
        base_config=AntSessionConfig(temporal_latent_dim=4, seed=0),
    )
    assert runner.sessions[0].runner is not runner.sessions[1].runner
    assert runner.sessions[0].navigator is not runner.sessions[1].navigator
    await runner.step_round()
    assert world.bus.snapshot.tick == 1

    report = await kernel_colony_foraging_experiment(
        n_ants=2,
        rounds=2,
        seed=1,
        session_config=AntSessionConfig(temporal_latent_dim=4, seed=1),
    )
    assert {(arm.arm, arm.bus_enabled) for arm in report.arms} == {
        ("learned", True),
        ("learned", False),
    }
