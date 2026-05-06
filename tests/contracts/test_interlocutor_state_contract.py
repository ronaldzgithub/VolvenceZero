"""Contract tests for the ``interlocutor_state`` owner (W2 SSOT cleanup).

These tests enforce the invariants in
``docs/specs/interlocutor-state.md``:

* Owner declares the canonical six dependencies and SHADOW default.
* ``InterlocutorState.__post_init__`` always recomputes zone bools
  from the axis values via :func:`compute_zones`; explicit zone
  values passed by callers are overwritten.
* ``compute_zones`` returns all-False below ``min_confidence``.
* Every documented zone is computed.
* Threshold constants are stable.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from volvence_zero.interlocutor import (
    InterlocutorReadoutContext,
    InterlocutorState,
    InterlocutorStateModule,
    InterlocutorStateSnapshot,
    InterlocutorThresholds,
    compute_zones,
    readout_interlocutor_state,
    with_zones,
)
from volvence_zero.runtime import WiringLevel


# ---------------------------------------------------------------------------
# Owner shape
# ---------------------------------------------------------------------------


def test_owner_dependencies_are_canonical() -> None:
    """Wave 2 freeze: the six upstream snapshots are the contract.
    Adding / removing a dependency requires a slot-registry change
    in ``docs/DATA_CONTRACT.md``.
    """

    assert InterlocutorStateModule.slot_name == "interlocutor_state"
    assert InterlocutorStateModule.dependencies == (
        "regime",
        "dual_track",
        "evaluation",
        "prediction_error",
        "memory",
        "commitment",
    )
    assert InterlocutorStateModule.default_wiring_level is WiringLevel.SHADOW
    assert InterlocutorStateModule.value_type is InterlocutorStateSnapshot


def test_owner_emits_snapshot_with_state() -> None:
    module = InterlocutorStateModule()

    async def _run() -> None:
        snap = await module.process({})
        assert isinstance(snap.value, InterlocutorStateSnapshot)
        assert isinstance(snap.value.state, InterlocutorState)
        # No upstream -> low confidence neutral readout.
        assert snap.value.state.readout_confidence < InterlocutorThresholds.min_confidence

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Zone classification SSOT
# ---------------------------------------------------------------------------


def test_zones_are_all_false_below_min_confidence() -> None:
    """Cold-start: a low-confidence readout must not modulate
    anything downstream.
    """

    cold = InterlocutorState(
        readout_confidence=0.10,
        emotional_weight=0.95,
        resistance_level=0.95,
        trust_signal=-0.95,
        pace_pressure=0.95,
        directness=0.05,
        rapport_warmth=0.05,
    )
    for zone_name in (
        "acknowledge_pressure_zone",
        "emotional_high_zone",
        "resistance_high_zone",
        "trust_negative_zone",
        "repair_zone",
        "direct_task_zone",
        "emotional_render_zone",
        "pace_pressure_zone",
        "low_directness_zone",
        "cold_rapport_zone",
    ):
        assert getattr(cold, zone_name) is False, (
            f"zone {zone_name} should be False below min_confidence"
        )


def test_zones_are_recomputed_from_axes_on_construction() -> None:
    """Even if a caller passes explicit zone bools the snapshot must
    overwrite them with the axis-derived values. Otherwise a
    consumer could drift the classification by hand.
    """

    state = InterlocutorState(
        readout_confidence=0.80,
        emotional_weight=0.70,
        resistance_level=0.10,
        trust_signal=0.10,
        pace_pressure=0.20,
        directness=0.70,
        rapport_warmth=0.70,
        engagement_intensity=0.50,
        # Try to lie: claim no zones fire.
        emotional_high_zone=False,
        acknowledge_pressure_zone=False,
        emotional_render_zone=False,
    )
    assert state.emotional_high_zone is True
    assert state.acknowledge_pressure_zone is True
    # emotional_render_zone needs self_disclosure_level >= 0.65 too;
    # the default 0.5 keeps it False here, but emotional_high_zone
    # is enough to assert the recompute fired.


@pytest.mark.parametrize(
    "axes,expected_zones",
    [
        (
            {"emotional_weight": 0.80},
            {"emotional_high_zone", "acknowledge_pressure_zone"},
        ),
        (
            {"resistance_level": 0.80},
            {
                "resistance_high_zone",
                "acknowledge_pressure_zone",
                "repair_zone",
            },
        ),
        (
            {"trust_signal": -0.50},
            {
                "trust_negative_zone",
                "acknowledge_pressure_zone",
            },
        ),
        (
            {"pace_pressure": 0.80},
            {"pace_pressure_zone"},
        ),
        (
            {"directness": 0.10},
            {"low_directness_zone"},
        ),
        (
            {
                "rapport_warmth": 0.20,
                "engagement_intensity": 0.55,
            },
            {"cold_rapport_zone"},
        ),
        (
            {
                "task_focus_level": 0.80,
                "directness": 0.70,
                "emotional_weight": 0.20,
            },
            {"direct_task_zone"},
        ),
        (
            {
                "emotional_weight": 0.70,
                "self_disclosure_level": 0.80,
            },
            {
                "emotional_high_zone",
                "acknowledge_pressure_zone",
                "emotional_render_zone",
            },
        ),
    ],
)
def test_per_axis_zone_classifications(
    axes: dict, expected_zones: set
) -> None:
    base = InterlocutorState(readout_confidence=0.80, **axes)
    fired = {
        zone_name
        for zone_name in (
            "acknowledge_pressure_zone",
            "emotional_high_zone",
            "resistance_high_zone",
            "trust_negative_zone",
            "repair_zone",
            "direct_task_zone",
            "emotional_render_zone",
            "pace_pressure_zone",
            "low_directness_zone",
            "cold_rapport_zone",
        )
        if getattr(base, zone_name)
    }
    missing = expected_zones - fired
    assert not missing, (
        f"expected zones {expected_zones} for axes {axes}, but "
        f"fired={fired} (missing={missing})"
    )


# ---------------------------------------------------------------------------
# Readout determinism
# ---------------------------------------------------------------------------


def test_readout_is_deterministic() -> None:
    """Same context -> same state, byte-for-byte. No global mutable
    state is allowed in the readout path."""

    ctx = InterlocutorReadoutContext(
        active_regime_id="problem_solving",
        has_dual_track=True,
        has_evaluation=True,
        cross_track_tension=0.4,
        warmth=0.6,
        info_integration=0.5,
    )
    a = readout_interlocutor_state(ctx)
    b = readout_interlocutor_state(ctx)
    assert dataclasses.asdict(a) == dataclasses.asdict(b)


def test_readout_zone_bools_match_compute_zones() -> None:
    """Whatever the readout function emits, its zone bools must agree
    with calling ``compute_zones`` on the resulting state. This keeps
    the readout from diverging from the SSOT classifier.
    """

    ctx = InterlocutorReadoutContext(
        active_regime_id="emotional_support",
        has_dual_track=True,
        has_evaluation=True,
        has_commitment=True,
        cross_track_tension=0.55,
        self_tension=0.55,
        warmth=0.30,
        repair_bias=0.40,
        commitment_alignment_trend=-0.30,
    )
    state = readout_interlocutor_state(ctx)
    expected = compute_zones(state)
    for zone_name, expected_value in expected.items():
        assert getattr(state, zone_name) == expected_value, zone_name


def test_with_zones_is_idempotent() -> None:
    state = InterlocutorState(
        readout_confidence=0.7,
        emotional_weight=0.7,
        resistance_level=0.6,
    )
    once = with_zones(state)
    twice = with_zones(once)
    assert dataclasses.asdict(once) == dataclasses.asdict(twice)


# ---------------------------------------------------------------------------
# Threshold stability
# ---------------------------------------------------------------------------


def test_thresholds_are_stable() -> None:
    """Changing a threshold is a contract change; this test pins them
    so accidental edits surface in PR review."""

    th = InterlocutorThresholds
    assert th.min_confidence == 0.30
    assert th.emotional_high == 0.55
    assert th.resistance_high == 0.50
    assert th.trust_negative == -0.10
    assert th.repair_resistance == 0.30
    assert th.repair_trust == 0.05
    assert th.direct_task_focus == 0.685
    assert th.direct_directness == 0.58
    assert th.direct_emotional_max == 0.58
    assert th.emotional_renderer == 0.56
    assert th.emotional_self_disclosure == 0.65
    assert th.pace_high == 0.65
    assert th.directness_low == 0.40
    assert th.rapport_low == 0.40
    assert th.engagement_floor == 0.30
