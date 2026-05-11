"""Schema + populator for ``IdentitySeed`` and ``DualTrackModule`` (packet 1.3'').

Pins the contract that:

* ``IdentitySeed`` (in ``vz-contracts``) is a frozen dataclass with
  trait-uniqueness validation.
* ``DualTrackModule.__init__`` accepts an optional
  ``identity_seed`` kwarg.
* When a seed is wired, ``DualTrackSnapshot.self_track.traits``
  carries those traits; ``world_track.traits`` stays empty
  (identity describes the lifeform, not the world).
* When no seed is wired (default), ``self_track.traits`` stays
  empty — backwards compatible.

Production wiring through ``LifeformConfig.with_identity_seed`` is
out of scope for packet 1.3''; that's the next packet (1.3''').
For now, ``run_final_wiring_turn(identity_seed=...)`` is the
direct path tests use.
"""

from __future__ import annotations

import asyncio

import pytest

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    build_growth_advisor_identity_seed,
)
from volvence_zero.dual_track import DualTrackModule
from volvence_zero.identity_seed import IdentitySeed
from volvence_zero.integration import FinalRolloutConfig, run_final_wiring_turn
from volvence_zero.runtime import WiringLevel
from volvence_zero.substrate.adapter import PlaceholderSubstrateAdapter


# ---------------------------------------------------------------------------
# IdentitySeed schema
# ---------------------------------------------------------------------------


def test_identity_seed_default_constructs() -> None:
    seed = IdentitySeed()
    assert seed.traits == ()
    assert seed.description == ""


def test_identity_seed_with_traits_constructs() -> None:
    seed = IdentitySeed(
        traits=("warm_peer_register", "long_horizon"),
        description="test",
    )
    assert seed.traits == ("warm_peer_register", "long_horizon")
    assert seed.description == "test"


def test_identity_seed_rejects_duplicate_traits() -> None:
    with pytest.raises(ValueError, match="traits must be unique"):
        IdentitySeed(traits=("warm", "warm"))


def test_identity_seed_rejects_empty_string_trait() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        IdentitySeed(traits=("warm", ""))


def test_identity_seed_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    seed = IdentitySeed(traits=("warm",))
    with pytest.raises(FrozenInstanceError):
        seed.traits = ("cold",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DualTrackModule constructor accepts identity_seed
# ---------------------------------------------------------------------------


def test_dual_track_module_accepts_no_identity_seed() -> None:
    module = DualTrackModule()
    assert module.identity_seed is None


def test_dual_track_module_accepts_explicit_none() -> None:
    module = DualTrackModule(identity_seed=None)
    assert module.identity_seed is None


def test_dual_track_module_stores_identity_seed() -> None:
    seed = IdentitySeed(traits=("warm_peer_register",))
    module = DualTrackModule(identity_seed=seed)
    assert module.identity_seed is seed


# ---------------------------------------------------------------------------
# E2E: run_final_wiring_turn threads identity_seed through to DualTrackSnapshot
# ---------------------------------------------------------------------------


def _run_turn(*, identity_seed: IdentitySeed | None) -> dict:
    """Run one ``run_final_wiring_turn`` and return its snapshots."""

    config = FinalRolloutConfig(dual_track=WiringLevel.ACTIVE)
    result = asyncio.run(
        run_final_wiring_turn(
            config=config,
            substrate_adapter=PlaceholderSubstrateAdapter(
                model_id="dual-track-identity-seed-test",
            ),
            session_id="dual-track-identity-seed-session",
            wave_id="dual-track-identity-seed-wave",
            identity_seed=identity_seed,
        )
    )
    return {
        "active": result.active_snapshots,
        "shadow": result.shadow_snapshots,
    }


def test_seed_populates_self_track_traits_in_snapshot() -> None:
    seed = IdentitySeed(traits=("warm_peer_register", "long_horizon"))
    snapshots = _run_turn(identity_seed=seed)
    dual_track_snapshot = snapshots["active"]["dual_track"]
    self_track = dual_track_snapshot.value.self_track
    assert self_track.traits == ("warm_peer_register", "long_horizon")


def test_seed_does_not_populate_world_track_traits() -> None:
    """Identity describes the lifeform's Self, not the World it
    interacts with. World track stays trait-empty regardless of
    the seed.
    """

    seed = IdentitySeed(traits=("warm_peer_register",))
    snapshots = _run_turn(identity_seed=seed)
    dual_track_snapshot = snapshots["active"]["dual_track"]
    assert dual_track_snapshot.value.world_track.traits == ()


def test_no_seed_leaves_self_track_traits_empty() -> None:
    snapshots = _run_turn(identity_seed=None)
    dual_track_snapshot = snapshots["active"]["dual_track"]
    self_track = dual_track_snapshot.value.self_track
    assert self_track.traits == ()


def test_growth_advisor_identity_seed_populates_cheng_laoshi_traits() -> None:
    """End-to-end: vertical fixture builder produces a seed that
    threads through ``run_final_wiring_turn`` and lands on
    ``DualTrackSnapshot.self_track.traits``.
    """

    seed = build_growth_advisor_identity_seed(build_cheng_laoshi_profile())
    snapshots = _run_turn(identity_seed=seed)
    dual_track_snapshot = snapshots["active"]["dual_track"]
    self_track = dual_track_snapshot.value.self_track
    assert "warm_peer_register" in self_track.traits
    assert "long_horizon" in self_track.traits
