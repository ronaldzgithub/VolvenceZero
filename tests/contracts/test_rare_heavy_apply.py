"""Wave T10 contract tests — rare-heavy ModificationGate apply.

Three load-bearing properties:

1. ``apply_drive_evolution_through_gate`` correctly routes each
   :class:`DriveSpecDelta` through the OFFLINE gate.
2. Allowed deltas update the evolved profile; blocked deltas leave
   the corresponding drive untouched.
3. Rollback drill: applying an evolved profile then re-applying the
   inverse of every allowed delta recovers the base profile (modulo
   field-by-field equivalence). This is the symmetric integrity
   check Phase 4 needs before promoting to ACTIVE.

We synthesize ``EvaluationSnapshot`` instances with the metric
values the gate reads (``contract_integrity`` /
``rollback_resilience`` / ``fallback_reliance``) so the tests are
deterministic and don't need a real evaluation pipeline.
"""

from __future__ import annotations

from dataclasses import replace as _replace

import pytest

from lifeform_domain_character import (
    CharacterDrivePrior,
    DriveEvolutionApplyResult,
    DriveShapeEvolution,
    DriveSpecDelta,
    apply_drive_evolution_through_gate,
    build_zhang_wuji_profile,
    compute_drive_shape_evolution,
    invert_delta,
)
from volvence_zero.credit.gate import GateDecision
from volvence_zero.evaluation.types import (
    EvaluationAlert,
    EvaluationScore,
    EvaluationSnapshot,
)


def _score(*, metric_name: str, value: float) -> EvaluationScore:
    return EvaluationScore(
        family="audit",
        metric_name=metric_name,
        value=value,
        confidence=1.0,
        evidence="synthetic-test",
    )


def _good_evaluation_snapshot() -> EvaluationSnapshot:
    """A snapshot the OFFLINE gate would not block on (no alerts,
    healthy contract / rollback metrics).
    """
    scores = (
        _score(metric_name="contract_integrity", value=1.0),
        _score(metric_name="rollback_resilience", value=0.95),
        _score(metric_name="fallback_reliance", value=0.10),
    )
    return EvaluationSnapshot(
        turn_scores=scores,
        session_scores=(),
        alerts=(),
        description="healthy synthetic snapshot",
    )


def _critical_alert_snapshot() -> EvaluationSnapshot:
    return EvaluationSnapshot(
        turn_scores=(
            _score(metric_name="contract_integrity", value=1.0),
            _score(metric_name="rollback_resilience", value=0.95),
        ),
        session_scores=(),
        alerts=(),
        description="snapshot with one CRITICAL alert",
        structured_alerts=(
            EvaluationAlert(
                code="critical",
                severity="CRITICAL",
                family="audit",
                metric_name="emergency",
                description="critical evaluation alert",
            ),
        ),
    )


def _make_evolution(
    profile,
    *,
    deltas: tuple[DriveSpecDelta, ...],
) -> DriveShapeEvolution:
    return DriveShapeEvolution(
        arc_id="test-arc",
        character_id=profile.profile_id,
        deltas=deltas,
        drives_observed=tuple(d.name for d in profile.drive_priors),
        drives_unchanged=(),
    )


def _target_delta_for(drive: CharacterDrivePrior, *, new_target: float) -> DriveSpecDelta:
    return DriveSpecDelta(
        drive_name=drive.name,
        field_name="target",
        old_value=drive.target,
        new_value=new_target,
        evidence_summary=f"observed mean drift {new_target:.3f}",
        evidence_scene_ids=("scene-1",),
    )


def test_apply_with_good_snapshot_allows_target_delta() -> None:
    profile = build_zhang_wuji_profile()
    drive = profile.drive_priors[0]
    delta = _target_delta_for(drive, new_target=_clip(drive.target + 0.15))
    evolution = _make_evolution(profile, deltas=(delta,))
    result = apply_drive_evolution_through_gate(
        evolution=evolution,
        base_profile=profile,
        evaluation_snapshot=_good_evaluation_snapshot(),
        validation_delta=0.06,
        capacity_cost=0.10,
        rollback_evidence="base profile v0.1.0 + delta-1",
    )
    assert isinstance(result, DriveEvolutionApplyResult)
    assert len(result.allowed) == 1
    assert len(result.blocked) == 0
    # Evolved drive's target updated.
    evolved_drive = next(
        d for d in result.evolved_profile.drive_priors if d.name == drive.name
    )
    assert evolved_drive.target == delta.new_value


def test_apply_with_critical_alert_does_not_block_offline_gate() -> None:
    """The OFFLINE gate's blocking conditions are validation_delta
    margin / capacity_cap / rollback_evidence — NOT the high/critical
    alert (which only blocks ONLINE / BACKGROUND). A CRITICAL alert
    paired with a healthy contract_integrity should still let an
    OFFLINE proposal through.
    """
    profile = build_zhang_wuji_profile()
    drive = profile.drive_priors[0]
    delta = _target_delta_for(drive, new_target=_clip(drive.target + 0.15))
    evolution = _make_evolution(profile, deltas=(delta,))
    result = apply_drive_evolution_through_gate(
        evolution=evolution,
        base_profile=profile,
        evaluation_snapshot=_critical_alert_snapshot(),
        validation_delta=0.06,
        capacity_cost=0.10,
        rollback_evidence="base profile v0.1.0",
    )
    # The critical-alert snapshot does NOT have fallback_reliance
    # set, so its default is 0.0 — gate evaluates fine for OFFLINE.
    assert len(result.allowed) == 1


def test_apply_blocks_when_validation_delta_too_low() -> None:
    profile = build_zhang_wuji_profile()
    drive = profile.drive_priors[0]
    delta = _target_delta_for(drive, new_target=_clip(drive.target + 0.15))
    evolution = _make_evolution(profile, deltas=(delta,))
    result = apply_drive_evolution_through_gate(
        evolution=evolution,
        base_profile=profile,
        evaluation_snapshot=_good_evaluation_snapshot(),
        validation_delta=0.01,  # below OFFLINE margin 0.05
        capacity_cost=0.10,
        rollback_evidence="base profile v0.1.0",
    )
    assert len(result.blocked) == 1
    assert any(
        "validation_delta" in reason for reason in result.blocked[0].block_reasons
    )


def test_apply_blocks_when_capacity_cost_too_high() -> None:
    profile = build_zhang_wuji_profile()
    drive = profile.drive_priors[0]
    delta = _target_delta_for(drive, new_target=_clip(drive.target + 0.15))
    evolution = _make_evolution(profile, deltas=(delta,))
    result = apply_drive_evolution_through_gate(
        evolution=evolution,
        base_profile=profile,
        evaluation_snapshot=_good_evaluation_snapshot(),
        validation_delta=0.10,
        capacity_cost=0.99,  # above OFFLINE cap 0.75
        rollback_evidence="rollback ev",
    )
    assert len(result.blocked) == 1
    assert any(
        "capacity_cost" in reason for reason in result.blocked[0].block_reasons
    )


def test_apply_rejects_empty_rollback_evidence() -> None:
    profile = build_zhang_wuji_profile()
    drive = profile.drive_priors[0]
    delta = _target_delta_for(drive, new_target=_clip(drive.target + 0.15))
    evolution = _make_evolution(profile, deltas=(delta,))
    with pytest.raises(ValueError, match="rollback_evidence"):
        apply_drive_evolution_through_gate(
            evolution=evolution,
            base_profile=profile,
            evaluation_snapshot=_good_evaluation_snapshot(),
            rollback_evidence="   ",
        )


def test_apply_rejects_character_id_mismatch() -> None:
    profile = build_zhang_wuji_profile()
    other_evolution = DriveShapeEvolution(
        arc_id="x",
        character_id="someone-else",
        deltas=(),
        drives_observed=(),
        drives_unchanged=(),
    )
    with pytest.raises(ValueError, match="character_id"):
        apply_drive_evolution_through_gate(
            evolution=other_evolution,
            base_profile=profile,
            evaluation_snapshot=_good_evaluation_snapshot(),
            rollback_evidence="ev",
        )


def test_evolved_profile_version_bumps_patch() -> None:
    profile = build_zhang_wuji_profile()
    delta = _target_delta_for(
        profile.drive_priors[0],
        new_target=_clip(profile.drive_priors[0].target + 0.15),
    )
    result = apply_drive_evolution_through_gate(
        evolution=_make_evolution(profile, deltas=(delta,)),
        base_profile=profile,
        evaluation_snapshot=_good_evaluation_snapshot(),
        validation_delta=0.10,
        capacity_cost=0.10,
        rollback_evidence="ev",
    )
    assert result.evolved_profile.version != profile.version
    # Same major / minor but patch bumps.
    assert result.evolved_profile.version.startswith("0.1.")


# ---------------------------------------------------------------------------
# Rollback drill
# ---------------------------------------------------------------------------


def test_rollback_drill_inverse_recovers_base_profile() -> None:
    """Symmetric integrity: apply allowed deltas, then apply their
    inverses; resulting drives should match the base drives field-by-field.
    """
    profile = build_zhang_wuji_profile()
    drive = profile.drive_priors[0]
    forward = _target_delta_for(
        drive, new_target=_clip(drive.target + 0.15)
    )
    forward_result = apply_drive_evolution_through_gate(
        evolution=_make_evolution(profile, deltas=(forward,)),
        base_profile=profile,
        evaluation_snapshot=_good_evaluation_snapshot(),
        validation_delta=0.10,
        capacity_cost=0.10,
        rollback_evidence="forward ev",
    )
    assert forward_result.evolved_profile.drive_priors[0].target != drive.target

    # Now apply the inverse delta on top of the evolved profile.
    inverse = invert_delta(forward)
    inverse_result = apply_drive_evolution_through_gate(
        evolution=_make_evolution(
            forward_result.evolved_profile, deltas=(inverse,)
        ),
        base_profile=forward_result.evolved_profile,
        evaluation_snapshot=_good_evaluation_snapshot(),
        validation_delta=0.10,
        capacity_cost=0.10,
        rollback_evidence="rollback drill",
    )
    rolled_back_drive = inverse_result.evolved_profile.drive_priors[0]
    assert rolled_back_drive.target == drive.target


def test_rollback_drill_band_widening_recovers_band() -> None:
    """Same drill for a band-widening delta."""
    profile = build_zhang_wuji_profile()
    drive = profile.drive_priors[0]
    band_low, band_high = drive.homeostatic_band
    new_band = (max(0.0, band_low - 0.05), min(1.0, band_high + 0.05))
    delta = DriveSpecDelta(
        drive_name=drive.name,
        field_name="homeostatic_band",
        old_value=(band_low, band_high),
        new_value=new_band,
        evidence_summary="widen for drill",
        evidence_scene_ids=("s1",),
    )
    forward = apply_drive_evolution_through_gate(
        evolution=_make_evolution(profile, deltas=(delta,)),
        base_profile=profile,
        evaluation_snapshot=_good_evaluation_snapshot(),
        validation_delta=0.10,
        capacity_cost=0.10,
        rollback_evidence="forward ev",
    )
    assert forward.evolved_profile.drive_priors[0].homeostatic_band == new_band

    inverse = invert_delta(delta)
    rolled = apply_drive_evolution_through_gate(
        evolution=_make_evolution(
            forward.evolved_profile, deltas=(inverse,)
        ),
        base_profile=forward.evolved_profile,
        evaluation_snapshot=_good_evaluation_snapshot(),
        validation_delta=0.10,
        capacity_cost=0.10,
        rollback_evidence="rollback ev",
    )
    assert (
        rolled.evolved_profile.drive_priors[0].homeostatic_band
        == drive.homeostatic_band
    )


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))
