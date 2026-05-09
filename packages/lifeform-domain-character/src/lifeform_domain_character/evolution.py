"""Pure function: compute drive-shape evolution from a replay report.

Wave T9 of the pipeline. Aggregates a :class:`ReplayReport` into
typed :class:`DriveSpecDelta` proposals describing how each
``DriveSpec`` should shift if the character's lived experience were
"baked in". The proposals are NOT auto-applied — Wave T10 routes
them through :class:`ModificationGate` (rare-heavy).

Aggregation rules (deliberately conservative; reviewer-overridable):

1. **Target shift** — if the drive's mean level across the replay
   sits more than ``target_drift_threshold`` (default 0.10) away from
   the spec target, propose a target shift toward the observed mean.
2. **Band widening** — if the drive's observed std exceeds
   ``band_std_floor_factor * (band_high - band_low)`` (default 0.6 of
   current band width), propose widening the band by the std excess.
3. **Regime recharge boost** — if a particular regime appears in
   ``replay_report`` (via per-scene ``active_regime``) more often
   than ``regime_minimum_observations`` (default 2 scenes), and the
   drive ``recharge_per_regime`` for that regime is below
   ``regime_recharge_floor`` (default 0.05), propose adding /
   raising the regime recharge.

Why pure functions:

The function takes the typed :class:`ReplayReport` and the base
:class:`CharacterSoulProfile` and returns a typed
:class:`DriveShapeEvolution`. No kernel side effects, no I/O, no
ModificationGate involvement — this is the proposer; Wave T10 is
the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from lifeform_domain_character.profile import (
    CharacterDrivePrior,
    CharacterSoulProfile,
)
from lifeform_domain_character.replay import ReplayReport, SceneReplayRecord


@dataclass(frozen=True)
class DriveSpecDelta:
    """One proposed change to a single drive's spec.

    Each delta names a single field on
    :class:`CharacterDrivePrior` and carries the proposed new value
    plus the evidence pointer the proposal was derived from. A
    delta is purely advisory until applied through the gate
    (Wave T10).
    """

    drive_name: str
    field_name: str  # one of "target", "homeostatic_band", "recharge_per_regime"
    old_value: object
    new_value: object
    evidence_summary: str
    evidence_scene_ids: tuple[str, ...]


@dataclass(frozen=True)
class DriveShapeEvolution:
    """Bundle of proposed drive-spec deltas + audit summary."""

    arc_id: str
    character_id: str
    deltas: tuple[DriveSpecDelta, ...]
    drives_observed: tuple[str, ...]
    drives_unchanged: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)


def compute_drive_shape_evolution(
    replay_report: ReplayReport,
    base_profile: CharacterSoulProfile,
    *,
    target_drift_threshold: float = 0.10,
    band_std_floor_factor: float = 0.6,
    regime_minimum_observations: int = 2,
    regime_recharge_floor: float = 0.05,
    regime_recharge_step: float = 0.05,
    band_widen_step: float = 0.05,
) -> DriveShapeEvolution:
    """Aggregate a replay report into proposed DriveSpec deltas.

    Args:
        replay_report: Output of :class:`ExperientialReplayDriver`.
        base_profile: Profile the lifeform was built from. We diff
            against this profile's :class:`CharacterDrivePrior` set.
        target_drift_threshold: Mean drift threshold above which a
            target shift is proposed.
        band_std_floor_factor: Band-widening trigger expressed as a
            fraction of current band width.
        regime_minimum_observations: Minimum number of scenes the
            regime must have been active in to trigger a recharge
            proposal.
        regime_recharge_floor: Drives whose existing
            ``recharge_per_regime[regime]`` is at or above this
            value won't be proposed a further raise.
        regime_recharge_step: Magnitude of proposed recharge raise.
        band_widen_step: Magnitude of proposed band widen.
    """
    if replay_report.character_id != base_profile.profile_id:
        raise ValueError(
            "compute_drive_shape_evolution: replay_report.character_id="
            f"{replay_report.character_id!r} does not match "
            f"base_profile.profile_id={base_profile.profile_id!r}"
        )
    base_drives = {drive.name: drive for drive in base_profile.drive_priors}
    deltas: list[DriveSpecDelta] = []
    notes: list[str] = []
    drives_observed: set[str] = set()
    # Per-scene observations.
    per_scene = replay_report.per_scene
    if not per_scene:
        return DriveShapeEvolution(
            arc_id=replay_report.arc_id,
            character_id=replay_report.character_id,
            deltas=(),
            drives_observed=(),
            drives_unchanged=tuple(sorted(base_drives)),
            notes=("no per-scene records to analyse",),
        )
    # Collect drive level series across scenes for each drive name.
    per_drive_levels: dict[str, list[float]] = {}
    per_regime_per_drive: dict[str, list[str]] = {}
    for scene in per_scene:
        regime = scene.active_regime
        for name, level in scene.drive_level_after:
            per_drive_levels.setdefault(name, []).append(float(level))
            if regime:
                per_regime_per_drive.setdefault(name, []).append(regime)
        drives_observed.add(scene.scene_id)
    # Compute per-drive deltas.
    drives_unchanged: list[str] = []
    for drive_name, drive in base_drives.items():
        levels = per_drive_levels.get(drive_name, [])
        if not levels:
            drives_unchanged.append(drive_name)
            continue
        mean = sum(levels) / len(levels)
        std = (
            sum((level - mean) ** 2 for level in levels) / len(levels)
        ) ** 0.5
        had_delta = False
        # Rule 1: target shift.
        if abs(mean - drive.target) >= target_drift_threshold:
            new_target = _clip_unit(mean)
            deltas.append(
                DriveSpecDelta(
                    drive_name=drive_name,
                    field_name="target",
                    old_value=drive.target,
                    new_value=new_target,
                    evidence_summary=(
                        f"mean drive level {mean:.3f} drifts "
                        f"{abs(mean - drive.target):.3f} away from spec "
                        f"target {drive.target:.3f} across "
                        f"{len(levels)} observations"
                    ),
                    evidence_scene_ids=tuple(
                        scene.scene_id
                        for scene in per_scene
                        if drive_name in {n for n, _ in scene.drive_level_after}
                    ),
                )
            )
            had_delta = True
        # Rule 2: band widening.
        band_low, band_high = drive.homeostatic_band
        band_width = max(band_high - band_low, 1e-6)
        if std >= band_std_floor_factor * band_width:
            new_low = _clip_unit(band_low - band_widen_step)
            new_high = _clip_unit(band_high + band_widen_step)
            deltas.append(
                DriveSpecDelta(
                    drive_name=drive_name,
                    field_name="homeostatic_band",
                    old_value=(band_low, band_high),
                    new_value=(new_low, new_high),
                    evidence_summary=(
                        f"drive volatility std={std:.3f} >= "
                        f"{band_std_floor_factor:.2f} * band_width "
                        f"({band_width:.3f}); proposing band widen by "
                        f"{band_widen_step:.2f} on each side"
                    ),
                    evidence_scene_ids=tuple(
                        scene.scene_id
                        for scene in per_scene
                        if drive_name in {n for n, _ in scene.drive_level_after}
                    ),
                )
            )
            had_delta = True
        # Rule 3: regime recharge boost.
        regimes = per_regime_per_drive.get(drive_name, [])
        regime_counts: dict[str, int] = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        existing_recharge = dict(drive.recharge_per_regime)
        for regime, count in regime_counts.items():
            if count < regime_minimum_observations:
                continue
            current = existing_recharge.get(regime, 0.0)
            if current >= regime_recharge_floor:
                continue
            new_recharge_map = dict(existing_recharge)
            new_recharge_map[regime] = current + regime_recharge_step
            deltas.append(
                DriveSpecDelta(
                    drive_name=drive_name,
                    field_name="recharge_per_regime",
                    old_value=tuple(sorted(existing_recharge.items())),
                    new_value=tuple(sorted(new_recharge_map.items())),
                    evidence_summary=(
                        f"regime {regime!r} appeared in {count} scenes "
                        f"but drive recharge was {current:.3f}; "
                        f"proposing raise to "
                        f"{current + regime_recharge_step:.3f}"
                    ),
                    evidence_scene_ids=tuple(
                        scene.scene_id
                        for scene in per_scene
                        if scene.active_regime == regime
                    ),
                )
            )
            had_delta = True
        if not had_delta:
            drives_unchanged.append(drive_name)
    return DriveShapeEvolution(
        arc_id=replay_report.arc_id,
        character_id=replay_report.character_id,
        deltas=tuple(deltas),
        drives_observed=tuple(sorted(per_drive_levels)),
        drives_unchanged=tuple(sorted(drives_unchanged)),
        notes=tuple(notes),
    )


def _clip_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "DriveShapeEvolution",
    "DriveSpecDelta",
    "compute_drive_shape_evolution",
]
