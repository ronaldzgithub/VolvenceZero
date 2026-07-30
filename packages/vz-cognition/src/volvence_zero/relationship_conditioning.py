"""Relationship conditioning bank owner (State KV P4-a).

Compiles the dyad-level "what is our relationship right now" readout --
trust and its long-horizon accumulation, repair pressure, emotional load,
and the consent envelope -- from the two typed semantic owners that own
that evidence (``relationship_state``, ``boundary_consent``).

This is the second State KV bank after Personal, and the first to publish
the generic scope-free :class:`ConditioningBankReadout` instead of a
bespoke per-bank contract. The runtime projects it into the scoped
``ConditioningBankSnapshot`` at the point of use (see
``conditioning_bank_adapters.bank_readout_to_bank``); this owner never
learns tenant/user/session identity.

Personal already consumes ``relationship_state``; that does not make this
module a second owner of relationship semantics -- both banks are readout
consumers of the same published snapshots, each compiling its own bounded
coordinates. The Relationship bank's distinct contribution is the
long-horizon half (cumulative trust, recovery, stabilization, tension
load) that the per-turn Personal vector does not carry.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from volvence_zero.conditioning_bank_contracts import (
    CONDITIONING_BANK_READOUT_SCHEMA_VERSION,
    ConditioningBankReadout,
    expected_bank_type,
)
from volvence_zero.conditioning_credit_feedback import (
    BankCreditFeedbackState,
    consume_bank_credit_feedback,
)
from volvence_zero.credit import CreditSnapshot
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel, stable_value_hash
from volvence_zero.semantic_state import (
    BoundaryConsentSnapshot,
    RelationshipStateSnapshot,
)

RELATIONSHIP_CONDITIONING_SLOT = "relationship_conditioning"
RELATIONSHIP_CONDITIONING_COMPILER_VERSION = "relationship-conditioning.v2"

RELATIONSHIP_CONDITIONING_READOUT_LABELS: tuple[str, ...] = (
    "rel_trust",
    "rel_cumulative_trust",
    "rel_continuity",
    "rel_repair_pressure",
    "rel_emotional_load",
    "rel_stabilization_need",
    "rel_trust_recovery",
    "rel_tension_load",
    "rel_attunement_trend",
    "rel_relationship_continuity",
    "rel_repair_progress",
    "rel_relationship_depth",
    "rel_consent_compliance",
    "rel_consent_clarity",
)

# Tension count above which the tension-load coordinate saturates. Four
# simultaneously unresolved tensions is already a fully strained dyad for
# rendering/injection purposes; the raw count stays available upstream.
_TENSION_SATURATION_COUNT = 4

_GROUP_ORDER: tuple[str, ...] = (
    "Trust",
    "Strain",
    "Trajectory",
    "Consent",
)

_LABEL_PHRASES: Mapping[str, tuple[str, str]] = {
    "rel_trust": ("Trust", "current trust"),
    "rel_cumulative_trust": ("Trust", "accumulated trust"),
    "rel_continuity": ("Trust", "continuity"),
    "rel_trust_recovery": ("Trust", "trust recovery"),
    "rel_repair_pressure": ("Strain", "repair pressure"),
    "rel_emotional_load": ("Strain", "emotional load"),
    "rel_stabilization_need": ("Strain", "stabilization need"),
    "rel_tension_load": ("Strain", "unresolved-tension load"),
    "rel_attunement_trend": ("Trajectory", "attunement direction"),
    "rel_relationship_continuity": (
        "Trajectory",
        "relationship continuity",
    ),
    "rel_repair_progress": ("Trajectory", "repair progress"),
    "rel_relationship_depth": ("Trajectory", "relationship depth"),
    "rel_consent_compliance": ("Consent", "compliance"),
    "rel_consent_clarity": ("Consent", "consent clarity"),
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _qualitative_level(value: float) -> str:
    if value < 0.34:
        return "low"
    if value < 0.67:
        return "moderate"
    return "high"


def render_relationship_conditioning_statement(
    *,
    readout: Sequence[float],
    readout_labels: Sequence[str],
    confidence: float,
) -> str:
    """Render the typed readout as a natural-language dyad-state statement.

    Same posture as the personal renderer: pure template over the labelled
    coordinates plus confidence, never semantic records or raw text, so the
    text and latent delivery paths carry identical information. Empty for a
    zero-confidence readout -- with no evidence there is nothing to state.
    """

    if tuple(readout_labels) != RELATIONSHIP_CONDITIONING_READOUT_LABELS:
        raise ValueError(
            "relationship conditioning rendering requires the frozen "
            "relationship bank label contract."
        )
    if len(readout) != len(readout_labels):
        raise ValueError(
            "relationship conditioning rendering requires one coordinate "
            "per label."
        )
    if confidence == 0.0:
        return ""

    grouped: dict[str, list[str]] = {group: [] for group in _GROUP_ORDER}
    for label, value in zip(readout_labels, readout, strict=True):
        group, phrase = _LABEL_PHRASES[label]
        grouped[group].append(
            f"{phrase} {_qualitative_level(value)} ({value:.2f})"
        )

    lines = [
        "Current dyad relationship estimate "
        f"(typed readout only, confidence {confidence:.2f}):"
    ]
    for group in _GROUP_ORDER:
        lines.append(f"- {group}: " + "; ".join(grouped[group]) + ".")
    return "\n".join(lines)


class RelationshipConditioningModule(RuntimeModule[ConditioningBankReadout]):
    """Compile the dyad relationship state into a bounded bank readout."""

    slot_name = RELATIONSHIP_CONDITIONING_SLOT
    owner = "RelationshipConditioningModule"
    value_type = ConditioningBankReadout
    dependencies = (
        "relationship_state",
        "boundary_consent",
        # State KV P5-c: bank-attributed credit readout. Optional -- the
        # credit slot may be SHADOW/absent, in which case the feedback
        # state simply persists unchanged.
        "credit",
    )
    default_wiring_level = WiringLevel.SHADOW

    # Slot <-> bank cross-check at class definition time: a copy-paste error
    # between banks must fail at import, not at the first publication.
    _BANK_TYPE = expected_bank_type(RELATIONSHIP_CONDITIONING_SLOT)

    def __init__(
        self,
        *,
        wiring_level: WiringLevel | None = None,
        credit_feedback_state: BankCreditFeedbackState | None = None,
        credit_feedback_level: WiringLevel = WiringLevel.SHADOW,
    ) -> None:
        super().__init__(wiring_level=wiring_level)
        # Session-lived bounded state injected by the runner so the EMA
        # survives per-turn module reconstruction; standalone/test callers
        # that pass nothing get a private per-instance state.
        self._credit_feedback_state = credit_feedback_state or BankCreditFeedbackState()
        self._credit_feedback_level = credit_feedback_level

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[ConditioningBankReadout]:
        relationship = upstream["relationship_state"]
        boundary = upstream["boundary_consent"]
        if not isinstance(relationship.value, RelationshipStateSnapshot):
            raise TypeError(
                "relationship_conditioning requires RelationshipStateSnapshot."
            )
        if not isinstance(boundary.value, BoundaryConsentSnapshot):
            raise TypeError(
                "relationship_conditioning requires BoundaryConsentSnapshot."
            )

        coverage_flags = (
            bool(
                relationship.value.rapport_signals
                or relationship.value.relational_tensions
                or relationship.value.relationship_age_turns
            ),
            bool(
                boundary.value.granted_consents
                or boundary.value.missing_consents
                or boundary.value.denied_boundaries
            ),
        )
        coverage = sum(float(flag) for flag in coverage_flags) / len(coverage_flags)
        is_cold_start = coverage == 0.0
        repair_evidence_count = (
            relationship.value.recent_repair_count
            + relationship.value.unresolved_tension_count
        )
        repair_progress = (
            _clamp(
                relationship.value.recent_repair_count
                / repair_evidence_count
            )
            if repair_evidence_count
            else 0.5
        )
        readout = (
            _clamp(relationship.value.trust_level),
            _clamp(relationship.value.cumulative_trust_level),
            _clamp(relationship.value.continuity_level),
            _clamp(relationship.value.repair_pressure),
            _clamp(relationship.value.emotional_load),
            _clamp(relationship.value.stabilization_need),
            _clamp(relationship.value.trust_recovery_signal),
            _clamp(
                relationship.value.unresolved_tension_count
                / _TENSION_SATURATION_COUNT
            ),
            _clamp(relationship.value.attunement_trend),
            _clamp(relationship.value.relationship_continuity_score),
            repair_progress,
            _clamp(relationship.value.relationship_age_turns / 20.0),
            _clamp(boundary.value.compliance_score),
            _clamp(boundary.value.consent_clarity),
        )
        if is_cold_start:
            readout = tuple(
                0.0 for _ in RELATIONSHIP_CONDITIONING_READOUT_LABELS
            )
        source_versions = (
            ("relationship_state", relationship.version),
            ("boundary_consent", boundary.version),
        )
        source_fingerprint = stable_value_hash(
            (
                RELATIONSHIP_CONDITIONING_COMPILER_VERSION,
                source_versions,
                stable_value_hash(relationship.value),
                stable_value_hash(boundary.value),
            )
        )
        base_confidence = 0.0 if is_cold_start else _clamp(
            coverage
            * (
                0.35
                + 0.25 * relationship.value.continuity_level
                + 0.20 * relationship.value.cumulative_trust_level
                + 0.20 * boundary.value.consent_clarity
            )
        )
        # State KV P5-c: same bounded credit feedback as the Personal owner
        # (shared update rule in ``conditioning_credit_feedback``). SHADOW
        # publishes the delta report-only; ACTIVE applies it; DISABLED stops
        # consumption (rollback point). Cold start is always exempt.
        credit_confidence_delta = 0.0
        if (
            self._credit_feedback_level is not WiringLevel.DISABLED
            and not is_cold_start
        ):
            credit_snapshot = upstream.get("credit")
            credit_value = (
                credit_snapshot.value
                if credit_snapshot is not None
                and isinstance(credit_snapshot.value, CreditSnapshot)
                else None
            )
            credit_confidence_delta = consume_bank_credit_feedback(
                self._credit_feedback_state,
                bank_name=self._BANK_TYPE.value,
                credit_snapshot=credit_value,
            )
        confidence = (
            _clamp(base_confidence + credit_confidence_delta)
            if self._credit_feedback_level is WiringLevel.ACTIVE
            else base_confidence
        )
        rendered_statement = render_relationship_conditioning_statement(
            readout=readout,
            readout_labels=RELATIONSHIP_CONDITIONING_READOUT_LABELS,
            confidence=confidence,
        )
        return self.publish(
            ConditioningBankReadout(
                schema_version=CONDITIONING_BANK_READOUT_SCHEMA_VERSION,
                bank_type=self._BANK_TYPE,
                readout=readout,
                readout_labels=RELATIONSHIP_CONDITIONING_READOUT_LABELS,
                source_versions=source_versions,
                source_fingerprint=source_fingerprint,
                confidence=confidence,
                provenance=(
                    "owner:RelationshipConditioningModule/"
                    f"{RELATIONSHIP_CONDITIONING_COMPILER_VERSION}/"
                    f"{CONDITIONING_BANK_READOUT_SCHEMA_VERSION}"
                ),
                is_cold_start=is_cold_start,
                description=(
                    "Relationship conditioning compiled from relationship_state "
                    f"and boundary_consent; coverage={coverage:.2f} "
                    f"confidence={confidence:.2f} cold_start={is_cold_start} "
                    f"credit_delta={credit_confidence_delta:+.3f}"
                    f"[{self._credit_feedback_level.value}] "
                    f"compiler={RELATIONSHIP_CONDITIONING_COMPILER_VERSION}."
                ),
                rendered_statement=rendered_statement,
                credit_confidence_delta=credit_confidence_delta,
            )
        )

    async def process_standalone(
        self, **kwargs: Any
    ) -> Snapshot[ConditioningBankReadout]:
        raise NotImplementedError(
            "RelationshipConditioningModule requires typed semantic owner "
            "snapshots."
        )


__all__ = [
    "RELATIONSHIP_CONDITIONING_READOUT_LABELS",
    "RELATIONSHIP_CONDITIONING_SLOT",
    "RelationshipConditioningModule",
    "render_relationship_conditioning_statement",
]
