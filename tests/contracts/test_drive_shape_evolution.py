"""Wave T9 contract tests — compute_drive_shape_evolution.

Pure function under test. We synthesize ``ReplayReport`` instances
to drive the three rules:

* Rule 1: target shift when mean drift > threshold.
* Rule 2: band widening when std exceeds band_std_floor_factor *
  band_width.
* Rule 3: regime recharge boost when a regime appears in >=
  regime_minimum_observations scenes AND current recharge is below
  regime_recharge_floor.

Plus negative tests:

* No deltas when drive levels stay near target with low std.
* character_id mismatch raises.
* Empty replay report yields empty deltas + appropriate note.
"""

from __future__ import annotations

import pytest

from lifeform_domain_character import (
    DriveShapeEvolution,
    DriveSpecDelta,
    ReplayReport,
    SceneReplayRecord,
    build_zhang_wuji_profile,
    compute_drive_shape_evolution,
)


def _scene(
    *,
    scene_id: str,
    phase: str = "child",
    regime: str = "guided_exploration",
    drives: tuple[tuple[str, float], ...],
    pe_magnitude: float = 0.1,
) -> SceneReplayRecord:
    return SceneReplayRecord(
        scene_id=scene_id,
        phase_label=phase,
        predicted_action_snippet="...",
        canonical_action="...",
        outcome_kind="helped",
        pe_magnitude=pe_magnitude,
        active_regime=regime,
        drive_level_after=drives,
    )


def _report(scenes: tuple[SceneReplayRecord, ...]) -> ReplayReport:
    return ReplayReport(
        arc_id="test-arc",
        character_id="zhang-wuji",
        scenes_processed=len(scenes),
        per_scene=scenes,
        total_pe_signal=sum(s.pe_magnitude for s in scenes),
        drive_drift=(),
        regime_sequence_payoff_growth=0,
        final_vitals=(),
    )


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_evolution_raises_on_character_id_mismatch() -> None:
    profile = build_zhang_wuji_profile()
    bad_report = ReplayReport(
        arc_id="x",
        character_id="other-character",
        scenes_processed=0,
        per_scene=(),
        total_pe_signal=0.0,
        drive_drift=(),
        regime_sequence_payoff_growth=0,
        final_vitals=(),
    )
    with pytest.raises(ValueError, match="character_id"):
        compute_drive_shape_evolution(bad_report, profile)


def test_evolution_with_empty_per_scene_records_yields_no_deltas() -> None:
    profile = build_zhang_wuji_profile()
    report = _report(scenes=())
    evolution = compute_drive_shape_evolution(report, profile)
    assert isinstance(evolution, DriveShapeEvolution)
    assert evolution.deltas == ()
    assert "no per-scene records" in " ".join(evolution.notes)


def test_evolution_no_drift_yields_no_deltas() -> None:
    """All drives held within target_drift_threshold AND below
    band_std_floor_factor: no deltas should be proposed.
    """
    profile = build_zhang_wuji_profile()
    target_levels = {drive.name: drive.target for drive in profile.drive_priors}
    scenes = tuple(
        _scene(
            scene_id=f"calm-{i}",
            drives=tuple(
                # Tiny variation so std is low.
                (name, target + ((i % 2) - 0.5) * 0.005)
                for name, target in target_levels.items()
            ),
        )
        for i in range(5)
    )
    report = _report(scenes=scenes)
    evolution = compute_drive_shape_evolution(report, profile)
    target_deltas = [d for d in evolution.deltas if d.field_name == "target"]
    band_deltas = [d for d in evolution.deltas if d.field_name == "homeostatic_band"]
    assert not target_deltas, (
        f"unexpected target deltas: {target_deltas!r}"
    )
    assert not band_deltas, (
        f"unexpected band deltas: {band_deltas!r}"
    )


# ---------------------------------------------------------------------------
# Rule 1: target shift
# ---------------------------------------------------------------------------


def test_target_shift_proposed_on_mean_drift() -> None:
    profile = build_zhang_wuji_profile()
    drive_name = profile.drive_priors[0].name
    spec_target = profile.drive_priors[0].target
    drifted_level = _clip(spec_target + 0.30)  # well above threshold
    scenes = tuple(
        _scene(
            scene_id=f"drift-{i}",
            drives=((drive_name, drifted_level),),
        )
        for i in range(5)
    )
    report = _report(scenes=scenes)
    evolution = compute_drive_shape_evolution(report, profile)
    target_deltas = [
        d
        for d in evolution.deltas
        if d.field_name == "target" and d.drive_name == drive_name
    ]
    assert len(target_deltas) == 1
    delta = target_deltas[0]
    assert delta.old_value == spec_target
    # New target should be near the drifted observed mean.
    assert abs(float(delta.new_value) - drifted_level) < 0.05


# ---------------------------------------------------------------------------
# Rule 2: band widening
# ---------------------------------------------------------------------------


def test_band_widening_proposed_on_high_volatility() -> None:
    """Pick a narrow-band drive (compassion_active band width ~0.30)
    and feed it levels with std > 0.6 * 0.30 = 0.18. We use values
    [0.0, 1.0, 0.0, 1.0, 0.5] (std ~0.45) which clearly exceed the
    threshold and force a band widening proposal.
    """
    profile = build_zhang_wuji_profile()
    # compassion_active is the first drive with band width 0.30.
    drive_name = "compassion_active"
    drive = next(d for d in profile.drive_priors if d.name == drive_name)
    band_low, band_high = drive.homeostatic_band
    # Extreme alternating values force a high std relative to band width.
    levels = [0.0, 1.0, 0.0, 1.0, 0.5]
    scenes = tuple(
        _scene(
            scene_id=f"vol-{i}",
            drives=((drive_name, levels[i]),),
        )
        for i in range(5)
    )
    report = _report(scenes=scenes)
    evolution = compute_drive_shape_evolution(report, profile)
    band_deltas = [
        d
        for d in evolution.deltas
        if d.field_name == "homeostatic_band" and d.drive_name == drive_name
    ]
    assert len(band_deltas) == 1, (
        f"expected band widen proposal for {drive_name!r}; "
        f"deltas={[(d.drive_name, d.field_name) for d in evolution.deltas]}"
    )
    delta = band_deltas[0]
    new_band = delta.new_value
    assert isinstance(new_band, tuple) and len(new_band) == 2
    new_low, new_high = new_band
    assert float(new_low) <= band_low
    assert float(new_high) >= band_high


# ---------------------------------------------------------------------------
# Rule 3: regime recharge boost
# ---------------------------------------------------------------------------


def test_regime_recharge_boost_proposed_when_under_floor() -> None:
    profile = build_zhang_wuji_profile()
    # Use a drive with no recharge_per_regime entry for some regime.
    drive_name = profile.drive_priors[2].name  # loyalty_to_kin
    drive = profile.drive_priors[2]
    # Pick a regime not already in the drive's recharge_per_regime.
    existing_regimes = {r for r, _ in drive.recharge_per_regime}
    novel_regime = next(
        regime
        for regime in (
            "guided_exploration",
            "casual_social",
            "first_encounter",
            "self_reflection",
        )
        if regime not in existing_regimes
    )
    scenes = tuple(
        _scene(
            scene_id=f"reg-{i}",
            regime=novel_regime,
            drives=((drive_name, drive.target),),
        )
        for i in range(3)
    )
    report = _report(scenes=scenes)
    evolution = compute_drive_shape_evolution(report, profile)
    regime_deltas = [
        d
        for d in evolution.deltas
        if d.field_name == "recharge_per_regime" and d.drive_name == drive_name
    ]
    assert len(regime_deltas) == 1
    delta = regime_deltas[0]
    new_map = dict(delta.new_value)
    assert novel_regime in new_map
    assert new_map[novel_regime] > 0.0


def test_regime_recharge_not_proposed_below_minimum_observations() -> None:
    """A regime that appears in only 1 scene must not trigger a
    recharge proposal (default minimum is 2).
    """
    profile = build_zhang_wuji_profile()
    drive_name = profile.drive_priors[3].name
    drive = profile.drive_priors[3]
    existing = {r for r, _ in drive.recharge_per_regime}
    novel_regime = next(
        regime
        for regime in (
            "guided_exploration",
            "casual_social",
            "first_encounter",
        )
        if regime not in existing
    )
    scenes = (
        _scene(
            scene_id="single",
            regime=novel_regime,
            drives=((drive_name, drive.target),),
        ),
    ) + tuple(
        _scene(
            scene_id=f"other-{i}",
            regime="emotional_support",
            drives=((drive_name, drive.target),),
        )
        for i in range(2)
    )
    report = _report(scenes=scenes)
    evolution = compute_drive_shape_evolution(report, profile)
    proposed_regimes_for_drive = [
        regime
        for d in evolution.deltas
        if d.drive_name == drive_name and d.field_name == "recharge_per_regime"
        for regime, _ in d.new_value
    ]
    assert novel_regime not in proposed_regimes_for_drive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))
