"""Lifeform vitals layer \u2014 the slow-scale R-PE source between turns.

These tests pin the contract that makes the lifeform an "always-on
organism" rather than a turn-driven assistant:

* Drives decay on SYSTEM ticks, not on ENERGY/CONTEXT ticks.
* Drives recharge from per-turn baselines and regime-specific bonuses.
* Out-of-band drives produce a non-zero PE contribution; in-band drives
  produce 0 (homeostasis is silent).
* Crossing the proactive-PE threshold surfaces a ``vitals``-sourced
  ``FollowupItem`` via the FollowupManager \u2014 the lifeform's "I am alive
  between turns" signal \u2014 with a cooldown so it never floods.
* The full companion lifeform built via ``build_companion_lifeform`` ships
  the bootstrap end-to-end (bootstrap construction works, idle ticks
  produce real drive deviation).

The tests use the lightweight ``LifeformConfig`` (no domain pack, no HF
substrate) where possible to keep them fast and isolated; the end-to-end
companion build is covered in one final integration test.
"""

from __future__ import annotations

import pytest

from lifeform_core import (
    DriveSpec,
    Lifeform,
    LifeformConfig,
    TickEngineConfig,
    TickKind,
    VitalsBootstrap,
    VitalsModule,
)
from lifeform_core.types import TickEvent


# ---------------------------------------------------------------------------
# Pure-VitalsModule unit tests \u2014 no Lifeform / Brain / kernel involved
# ---------------------------------------------------------------------------


def _make_bootstrap() -> VitalsBootstrap:
    return VitalsBootstrap(
        drives=(
            DriveSpec(
                name="bond",
                target=0.7,
                homeostatic_band=(0.5, 0.85),
                decay_per_tick=0.05,
                pe_weight=1.0,
                initial_level=0.7,
                recharge_per_turn=0.0,
                recharge_per_regime={"emotional_support": 0.4},
            ),
            DriveSpec(
                name="engagement",
                target=0.7,
                homeostatic_band=(0.4, 0.9),
                decay_per_tick=0.10,
                pe_weight=1.0,
                initial_level=0.6,
                recharge_per_turn=0.3,
            ),
        ),
        proactive_pe_threshold=0.4,
        proactive_cooldown_ticks=10,
    )


def _system_tick(index: int) -> TickEvent:
    return TickEvent(tick_index=index, kind=TickKind.SYSTEM, elapsed_seconds=float(index))


def _energy_tick(index: int) -> TickEvent:
    return TickEvent(tick_index=index, kind=TickKind.ENERGY, elapsed_seconds=float(index))


def test_vitals_module_initialises_levels_at_drive_initial_levels():
    module = VitalsModule(_make_bootstrap())
    snap = module.current_snapshot()
    by_name = {d.name: d for d in snap.drive_levels}
    assert by_name["bond"].level == pytest.approx(0.7)
    assert by_name["engagement"].level == pytest.approx(0.6)
    # Both drives start in-band so PE contribution is zero.
    assert snap.total_pe == pytest.approx(0.0)
    assert snap.above_proactive_threshold is False


def test_system_ticks_decay_drives_but_energy_ticks_do_not():
    module = VitalsModule(_make_bootstrap())
    module.on_tick(_system_tick(1))
    module.on_tick(_system_tick(2))
    snap = module.current_snapshot()
    by_name = {d.name: d for d in snap.drive_levels}
    # bond: 0.7 - 2*0.05 = 0.6 (still in band [0.5, 0.85])
    assert by_name["bond"].level == pytest.approx(0.6)
    # engagement: 0.6 - 2*0.10 = 0.4 (right at lower bound; out-of-band is strict <)
    assert by_name["engagement"].level == pytest.approx(0.4)

    # ENERGY tick must not move levels.
    module.on_tick(_energy_tick(3))
    snap = module.current_snapshot()
    by_name = {d.name: d for d in snap.drive_levels}
    assert by_name["bond"].level == pytest.approx(0.6)
    assert by_name["engagement"].level == pytest.approx(0.4)
    # tick_index still advances even on non-system ticks.
    assert snap.tick_index == 3


def test_idle_ticks_eventually_push_drives_out_of_band_and_produce_pe():
    module = VitalsModule(_make_bootstrap())
    for i in range(1, 11):  # 10 ticks
        module.on_tick(_system_tick(i))
    snap = module.current_snapshot()
    out_names = {d.name for d in snap.drive_levels if d.out_of_band}
    # engagement decays 0.1/tick from 0.6 -> 0.0 by tick 6 (out-of-band by tick 3).
    # bond decays 0.05/tick from 0.7 -> 0.2 by tick 10 (out-of-band by tick 5).
    assert "engagement" in out_names
    assert "bond" in out_names
    assert snap.total_pe > 0.0


def test_recharge_per_turn_lifts_drives_back_into_band():
    module = VitalsModule(_make_bootstrap())
    for i in range(1, 6):
        module.on_tick(_system_tick(i))
    pre = module.current_snapshot()
    # 5 ticks: engagement at 0.1, bond at 0.45 -> both out-of-band.
    assert pre.above_proactive_threshold or pre.total_pe > 0.0

    # Recharge with a guided_exploration regime (no bond bonus, just baseline).
    module.on_turn(regime="guided_exploration", user_input_present=True)
    after_turn = module.current_snapshot()
    by_name = {d.name: d for d in after_turn.drive_levels}
    # engagement: 0.1 + 0.3 = 0.4 (back at lower bound but still out-of-band by strict <)
    assert by_name["engagement"].level == pytest.approx(0.4)


def test_regime_specific_recharge_targets_only_matching_drives():
    module = VitalsModule(_make_bootstrap())
    # Push bond all the way down.
    for i in range(1, 11):
        module.on_tick(_system_tick(i))
    # An emotional_support turn carries a +0.4 bond bonus.
    module.on_turn(regime="emotional_support", user_input_present=True)
    snap = module.current_snapshot()
    by_name = {d.name: d for d in snap.drive_levels}
    # bond: 0.2 + 0.4 = 0.6 (back in-band).
    assert by_name["bond"].level == pytest.approx(0.6)
    assert by_name["bond"].out_of_band is False
    # engagement: 0.0 + 0.3 = 0.3 (not bond-recharged; only the baseline).
    assert by_name["engagement"].level == pytest.approx(0.3)


def test_proactive_followup_predicate_respects_threshold_and_cooldown():
    module = VitalsModule(_make_bootstrap())
    # Tick 1 \u2192 still within band on both \u2192 should NOT fire.
    module.on_tick(_system_tick(1))
    assert module.consider_proactive_followup(current_tick=1) is False

    # Run enough ticks to definitely cross the threshold.
    for i in range(2, 12):
        module.on_tick(_system_tick(i))
    assert module.current_snapshot().above_proactive_threshold is True

    # First call fires; second within cooldown does not.
    assert module.consider_proactive_followup(current_tick=11) is True
    assert module.consider_proactive_followup(current_tick=12) is False
    assert module.consider_proactive_followup(current_tick=20) is False
    # After cooldown elapses (>= last + 10) it fires again.
    assert module.consider_proactive_followup(current_tick=21) is True


def test_user_input_absent_skips_per_turn_baseline_but_keeps_regime_bonus():
    module = VitalsModule(_make_bootstrap())
    for i in range(1, 11):
        module.on_tick(_system_tick(i))
    module.on_turn(regime="emotional_support", user_input_present=False)
    snap = module.current_snapshot()
    by_name = {d.name: d for d in snap.drive_levels}
    # engagement: 0.0 + 0.0 (no baseline since no user input) = 0.0
    assert by_name["engagement"].level == pytest.approx(0.0)
    # bond: 0.2 + 0.4 (regime bonus survives) = 0.6
    assert by_name["bond"].level == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Full LifeformSession integration \u2014 ticks + followups + snapshots
# ---------------------------------------------------------------------------


@pytest.fixture
def lifeform_with_vitals() -> Lifeform:
    config = LifeformConfig(
        tick=TickEngineConfig(system_tick_seconds=0.001, energy_every_n_system_ticks=999),
        idle_close_after_system_ticks=None,  # disable scene auto-close so vitals are isolated
        vitals_bootstrap=_make_bootstrap(),
    )
    return Lifeform(config)


async def test_advance_tick_propagates_to_vitals_and_surfaces_proactive_followup(
    lifeform_with_vitals: Lifeform,
):
    session = lifeform_with_vitals.create_session(session_id="vitals-test")
    assert session.vitals_snapshot is not None
    assert session.vitals_snapshot.total_pe == pytest.approx(0.0)

    # Sit idle long enough to cross the proactive threshold.
    await session.advance_tick(15)
    snap = session.vitals_snapshot
    assert snap is not None
    assert snap.tick_index == 15
    assert snap.above_proactive_threshold is True

    pending = session.followup_manager.pending
    vitals_items = [item for item in pending if item.source == "vitals"]
    assert len(vitals_items) == 1
    item = vitals_items[0]
    assert item.priority == pytest.approx(0.55)  # default priority from bootstrap
    assert "drives_out_of_band" in item.metadata
    # The followup is due immediately so it shows up in due_now.
    due = session.due_followups()
    assert any(i.followup_id == item.followup_id for i in due)


async def test_proactive_followup_does_not_fire_within_cooldown(
    lifeform_with_vitals: Lifeform,
):
    session = lifeform_with_vitals.create_session(session_id="cooldown-test")
    await session.advance_tick(15)
    first_count = len(
        [i for i in session.followup_manager.pending if i.source == "vitals"]
    )
    # Continue ticking inside the cooldown \u2014 no new vitals followup.
    await session.advance_tick(5)
    second_count = len(
        [i for i in session.followup_manager.pending if i.source == "vitals"]
    )
    assert second_count == first_count == 1


def test_lifeform_without_vitals_bootstrap_exposes_no_vitals_state():
    config = LifeformConfig(vitals_bootstrap=None)
    life = Lifeform(config)
    session = life.create_session(session_id="no-vitals")
    assert session.vitals_module is None
    assert session.vitals_snapshot is None


# ---------------------------------------------------------------------------
# Companion vertical end-to-end smoke test
# ---------------------------------------------------------------------------


def test_companion_vitals_bootstrap_factory_produces_three_drives():
    from lifeform_domain_emogpt import build_companion_vitals_bootstrap

    bootstrap = build_companion_vitals_bootstrap()
    drive_names = {d.name for d in bootstrap.drives}
    assert drive_names == {"bond_warmth", "user_engagement", "conversation_continuity"}
    # Each drive's homeostatic band is non-degenerate.
    for drive in bootstrap.drives:
        low, high = drive.homeostatic_band
        assert low < high, f"degenerate band for {drive.name}: {drive.homeostatic_band}"


async def test_build_companion_lifeform_default_includes_vitals():
    from lifeform_domain_emogpt import build_companion_lifeform

    life = build_companion_lifeform()
    session = life.create_session(session_id="companion-vitals")
    snap = session.vitals_snapshot
    assert snap is not None
    drive_names = {d.name for d in snap.drive_levels}
    assert "bond_warmth" in drive_names
    assert "user_engagement" in drive_names

    # Sit idle past the companion's proactive threshold; user_engagement
    # decays the fastest so it crosses out-of-band first.
    await session.advance_tick(30)
    after = session.vitals_snapshot
    assert after.tick_index == 30
    out_of_band = {d.name for d in after.drive_levels if d.out_of_band}
    assert "user_engagement" in out_of_band, (
        f"user_engagement should be out-of-band after 30 idle ticks, got "
        f"levels={[(d.name, d.level) for d in after.drive_levels]}"
    )


async def test_build_companion_lifeform_can_disable_vitals_for_ablation():
    from lifeform_domain_emogpt import build_companion_lifeform

    life = build_companion_lifeform(use_vitals_bootstrap=False)
    session = life.create_session(session_id="companion-no-vitals")
    assert session.vitals_module is None
    assert session.vitals_snapshot is None
