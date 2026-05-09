"""Rare-heavy ModificationGate apply for drive shape evolution.

Wave T10 of the pipeline. Takes a :class:`DriveShapeEvolution`
proposal (Wave T9) and routes each :class:`DriveSpecDelta` through
the existing :func:`evaluate_gate` machinery in
``vz-cognition.credit.gate``. Allowed deltas are applied to the
base profile producing an ``evolved_profile``; blocked deltas are
returned in an audit list for downstream review.

Why OFFLINE gate:

Drive-spec changes mean the *kind of being* the lifeform is —
homeostatic targets, drives that pull, regimes that recharge them.
That's the rare-heavy time scale (R10). Routing through OFFLINE
gate gives:

* validation_delta margin ≥ 0.05 (caller must pass a non-trivial
  improvement signal),
* capacity_cost cap ≤ 0.75,
* mandatory rollback_evidence,
* mandatory ``is_reversible=True``.

Rollback drill (test surface):

A rollback drill in ``test_rare_heavy_apply.py`` confirms that
applying an evolved_profile then re-applying the *inverse* delta
recovers the original profile (modulo float tolerance). This is
the symmetric integrity check Phase 4 needs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace as _replace

from volvence_zero.credit.gate import (
    GateDecision,
    ModificationGate,
    ModificationProposal,
    evaluate_gate,
    evaluate_gate_reasons,
)
from volvence_zero.evaluation.types import EvaluationSnapshot

from lifeform_domain_character.evolution import (
    DriveShapeEvolution,
    DriveSpecDelta,
)
from lifeform_domain_character.profile import (
    CharacterDrivePrior,
    CharacterSoulProfile,
)


@dataclass(frozen=True)
class GatedDriveSpecDelta:
    """A :class:`DriveSpecDelta` after the gate has decided."""

    delta: DriveSpecDelta
    proposal: ModificationProposal
    decision: GateDecision
    block_reasons: tuple[str, ...]


@dataclass(frozen=True)
class DriveEvolutionApplyResult:
    """Return value of :func:`apply_drive_evolution_through_gate`."""

    evolved_profile: CharacterSoulProfile
    base_profile: CharacterSoulProfile
    allowed: tuple[GatedDriveSpecDelta, ...]
    blocked: tuple[GatedDriveSpecDelta, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)


def apply_drive_evolution_through_gate(
    *,
    evolution: DriveShapeEvolution,
    base_profile: CharacterSoulProfile,
    evaluation_snapshot: EvaluationSnapshot,
    validation_delta: float = 0.05,
    capacity_cost: float = 0.20,
    rollback_evidence: str = "",
) -> DriveEvolutionApplyResult:
    """Run each drive-spec delta through the OFFLINE gate.

    Args:
        evolution: Output of
            :func:`compute_drive_shape_evolution`.
        base_profile: Profile to apply allowed deltas onto.
        evaluation_snapshot: Caller-supplied snapshot. The gate
            checks for HIGH/CRITICAL alerts and reads
            ``contract_integrity`` / ``rollback_resilience`` /
            ``fallback_reliance`` metric values.
        validation_delta: validation_delta to attach to each
            proposal (defaults to ``0.05`` — exactly at the OFFLINE
            margin; raise this when the caller has stronger evidence).
        capacity_cost: capacity_cost to attach (default ``0.20``;
            well under the OFFLINE cap of ``0.75``).
        rollback_evidence: Required short string identifying how
            this proposal can be rolled back (e.g. base profile
            version + delta id). Must be non-empty for the gate.
    """
    if evolution.character_id != base_profile.profile_id:
        raise ValueError(
            "apply_drive_evolution_through_gate: evolution.character_id="
            f"{evolution.character_id!r} does not match "
            f"base_profile.profile_id={base_profile.profile_id!r}"
        )
    if not rollback_evidence.strip():
        raise ValueError(
            "apply_drive_evolution_through_gate: rollback_evidence "
            "must be non-empty (the OFFLINE gate requires it)"
        )
    allowed: list[GatedDriveSpecDelta] = []
    blocked: list[GatedDriveSpecDelta] = []
    base_drives = {drive.name: drive for drive in base_profile.drive_priors}
    evolved_drives: dict[str, CharacterDrivePrior] = dict(base_drives)
    for delta in evolution.deltas:
        proposal = _proposal_for_delta(
            delta=delta,
            base_drives=base_drives,
            validation_delta=validation_delta,
            capacity_cost=capacity_cost,
            rollback_evidence=rollback_evidence,
        )
        decision = evaluate_gate(
            proposal=proposal,
            evaluation_snapshot=evaluation_snapshot,
        )
        reasons: tuple[str, ...] = ()
        if decision is GateDecision.BLOCK:
            reasons = evaluate_gate_reasons(
                proposal=proposal,
                evaluation_snapshot=evaluation_snapshot,
            )
            blocked.append(
                GatedDriveSpecDelta(
                    delta=delta,
                    proposal=proposal,
                    decision=decision,
                    block_reasons=reasons,
                )
            )
            continue
        allowed.append(
            GatedDriveSpecDelta(
                delta=delta,
                proposal=proposal,
                decision=decision,
                block_reasons=(),
            )
        )
        # Apply the delta to the running drive set.
        evolved_drives[delta.drive_name] = _apply_delta_to_drive(
            drive=evolved_drives[delta.drive_name], delta=delta
        )
    evolved_profile = _replace(
        base_profile,
        version=_increment_version(base_profile.version),
        drive_priors=tuple(
            evolved_drives[drive.name] for drive in base_profile.drive_priors
        ),
        reviewed_by=(
            base_profile.reviewed_by + " + drive_evolution(wave T10)"
        ),
    )
    return DriveEvolutionApplyResult(
        evolved_profile=evolved_profile,
        base_profile=base_profile,
        allowed=tuple(allowed),
        blocked=tuple(blocked),
        notes=evolution.notes,
    )


def invert_delta(delta: DriveSpecDelta) -> DriveSpecDelta:
    """Return the inverse :class:`DriveSpecDelta` (swap old / new).

    Used by the rollback drill in tests.
    """
    return DriveSpecDelta(
        drive_name=delta.drive_name,
        field_name=delta.field_name,
        old_value=delta.new_value,
        new_value=delta.old_value,
        evidence_summary=f"INVERSE of: {delta.evidence_summary}",
        evidence_scene_ids=delta.evidence_scene_ids,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _proposal_for_delta(
    *,
    delta: DriveSpecDelta,
    base_drives: dict[str, CharacterDrivePrior],
    validation_delta: float,
    capacity_cost: float,
    rollback_evidence: str,
) -> ModificationProposal:
    drive = base_drives[delta.drive_name]
    old_state_repr = repr((drive.name, delta.field_name, delta.old_value))
    new_state_repr = repr((drive.name, delta.field_name, delta.new_value))
    return ModificationProposal(
        target=f"character.drive_priors[{delta.drive_name}].{delta.field_name}",
        desired_gate=ModificationGate.OFFLINE,
        old_value_hash=hashlib.sha256(old_state_repr.encode("utf-8")).hexdigest(),
        new_value_hash=hashlib.sha256(new_state_repr.encode("utf-8")).hexdigest(),
        justification=delta.evidence_summary,
        is_reversible=True,
        validation_delta=validation_delta,
        capacity_cost=capacity_cost,
        rollback_evidence=rollback_evidence,
    )


def _apply_delta_to_drive(
    *,
    drive: CharacterDrivePrior,
    delta: DriveSpecDelta,
) -> CharacterDrivePrior:
    if delta.field_name == "target":
        return _replace(drive, target=float(delta.new_value))
    if delta.field_name == "homeostatic_band":
        new_band = tuple(float(b) for b in delta.new_value)
        if len(new_band) != 2:
            raise ValueError(
                f"_apply_delta_to_drive: homeostatic_band must have 2 "
                f"elements, got {new_band!r}"
            )
        return _replace(drive, homeostatic_band=new_band)
    if delta.field_name == "recharge_per_regime":
        new_recharge = tuple(
            (str(name), float(value)) for name, value in delta.new_value
        )
        return _replace(drive, recharge_per_regime=new_recharge)
    raise ValueError(
        f"_apply_delta_to_drive: unknown field {delta.field_name!r}"
    )


def _increment_version(version: str) -> str:
    """Bump the patch version: '0.1.0' -> '0.1.1'.

    Falls back to appending '+evolved' when the version string is
    not in semver-ish form.
    """
    parts = version.split(".")
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        major, minor, patch = (int(p) for p in parts)
        return f"{major}.{minor}.{patch + 1}"
    return f"{version}+evolved"


__all__ = [
    "DriveEvolutionApplyResult",
    "GatedDriveSpecDelta",
    "apply_drive_evolution_through_gate",
    "invert_delta",
]
