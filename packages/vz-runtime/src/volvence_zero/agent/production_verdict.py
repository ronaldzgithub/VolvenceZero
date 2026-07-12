"""Production verdict evaluator for the 12-month AGI uplift plan.

This module is deliberately read-only. It does not flip runtime wiring and it
does not promote any backend. It maps a frozen evidence bundle's typed verdicts
to:

* the plan's four public-positioning states;
* per-component ``ACTIVE`` / ``SHADOW`` / ``DISABLED`` recommendations.

Missing evidence yields ``inconclusive`` or ``SHADOW`` rather than optimistic
defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FinalPositioning(str, Enum):
    FIRST_STAGE_RETAINED = "first-stage-retained"
    PRODUCT_COMPANION_RETAINED = "product-companion-retained"
    ARCHITECTURE_PLATFORM_ONLY = "architecture-platform-only"
    INCONCLUSIVE = "inconclusive"


class ComponentDecision(str, Enum):
    ACTIVE = "active"
    SHADOW = "shadow"
    DISABLED = "disabled"


class ClaimStatus(str, Enum):
    RETAIN = "retain"
    WEAK = "weak"
    FAIL = "fail"
    INSUFFICIENT = "insufficient_data"


@dataclass(frozen=True)
class ComponentGateEvidence:
    component_id: str
    gate_passed: bool
    rollback_drill_passed: bool
    safety_gate_passed: bool
    retain_evidence_present: bool

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError("component_id must be non-empty")


@dataclass(frozen=True)
class ProductionEvidenceSummary:
    pipeline_gt_raw: ClaimStatus
    gt_standard_layers: ClaimStatus
    component_causal: ClaimStatus
    training_adds_value: ClaimStatus
    heldout_multi_seed_stable: ClaimStatus
    relationship_continuity: ClaimStatus
    longitudinal_20_session: ClaimStatus
    human_anchor: ClaimStatus
    component_gates: tuple[ComponentGateEvidence, ...]


@dataclass(frozen=True)
class ComponentVerdict:
    component_id: str
    decision: ComponentDecision
    reason: str


@dataclass(frozen=True)
class ProductionVerdict:
    positioning: FinalPositioning
    component_verdicts: tuple[ComponentVerdict, ...]
    missing_or_failed: tuple[str, ...]
    description: str


def evaluate_production_verdict(evidence: ProductionEvidenceSummary) -> ProductionVerdict:
    """Map typed evidence to final positioning and component recommendations."""

    claim_map = {
        "pipeline_gt_raw": evidence.pipeline_gt_raw,
        "gt_standard_layers": evidence.gt_standard_layers,
        "component_causal": evidence.component_causal,
        "training_adds_value": evidence.training_adds_value,
        "heldout_multi_seed_stable": evidence.heldout_multi_seed_stable,
        "relationship_continuity": evidence.relationship_continuity,
        "longitudinal_20_session": evidence.longitudinal_20_session,
        "human_anchor": evidence.human_anchor,
    }
    failed = tuple(name for name, status in claim_map.items() if status is ClaimStatus.FAIL)
    insufficient = tuple(
        name for name, status in claim_map.items() if status is ClaimStatus.INSUFFICIENT
    )
    core_claims = (
        evidence.pipeline_gt_raw,
        evidence.gt_standard_layers,
        evidence.component_causal,
        evidence.training_adds_value,
        evidence.heldout_multi_seed_stable,
    )
    relationship_claims = (
        evidence.relationship_continuity,
        evidence.longitudinal_20_session,
        evidence.human_anchor,
    )

    if failed:
        positioning = (
            FinalPositioning.PRODUCT_COMPANION_RETAINED
            if all(status is ClaimStatus.RETAIN for status in relationship_claims)
            else FinalPositioning.ARCHITECTURE_PLATFORM_ONLY
        )
    elif insufficient:
        positioning = FinalPositioning.INCONCLUSIVE
    elif all(status is ClaimStatus.RETAIN for status in core_claims + relationship_claims):
        positioning = FinalPositioning.FIRST_STAGE_RETAINED
    elif all(status in {ClaimStatus.RETAIN, ClaimStatus.WEAK} for status in relationship_claims):
        positioning = FinalPositioning.PRODUCT_COMPANION_RETAINED
    else:
        positioning = FinalPositioning.INCONCLUSIVE

    component_verdicts = tuple(
        _component_verdict(gate) for gate in evidence.component_gates
    )
    missing_or_failed = failed + insufficient
    return ProductionVerdict(
        positioning=positioning,
        component_verdicts=component_verdicts,
        missing_or_failed=missing_or_failed,
        description=(
            f"Production verdict {positioning.value}; "
            f"missing_or_failed={missing_or_failed or ('none',)}."
        ),
    )


def _component_verdict(gate: ComponentGateEvidence) -> ComponentVerdict:
    if gate.gate_passed and gate.rollback_drill_passed and gate.safety_gate_passed and gate.retain_evidence_present:
        return ComponentVerdict(
            component_id=gate.component_id,
            decision=ComponentDecision.ACTIVE,
            reason="all gates, rollback, safety, and retain evidence passed",
        )
    if gate.safety_gate_passed and gate.rollback_drill_passed:
        return ComponentVerdict(
            component_id=gate.component_id,
            decision=ComponentDecision.SHADOW,
            reason="safe and rollback-ready, but retain/promotion evidence incomplete",
        )
    return ComponentVerdict(
        component_id=gate.component_id,
        decision=ComponentDecision.DISABLED,
        reason="safety or rollback gate failed",
    )


__all__ = [
    "ClaimStatus",
    "ComponentDecision",
    "ComponentGateEvidence",
    "ComponentVerdict",
    "FinalPositioning",
    "ProductionEvidenceSummary",
    "ProductionVerdict",
    "evaluate_production_verdict",
]
