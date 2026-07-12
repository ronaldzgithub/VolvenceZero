"""Owner/evaluation evidence producers for dialogue outcome resolution.

Each producer is a pure function over a single owner snapshot. Producers do
not read raw user text and do not own prediction-error or commitment
state. They emit conservative, low-confidence ``DialogueOutcomeEvidence``
that ``DialogueTraceStore`` can map onto richer ``DialogueOutcomeKind``
values without adding semantic logic of its own.
"""

from __future__ import annotations

from volvence_zero.dialogue_trace import (
    DialogueExternalOutcomeEvidence,
    DialogueExternalOutcomeKind,
    DialogueOutcomeEvidence,
    DialogueOutcomeEvidenceSource,
    DialogueOutcomeKind,
)
from volvence_zero.environment import EnvironmentOutcome


PE_CONTINUED_MAGNITUDE_THRESHOLD = 0.4
PE_CONTINUED_CONFIDENCE = 0.3
COMMITMENT_OUTCOME_CONFIDENCE = 0.6
SCENE_CLOSED_CONFIDENCE = 0.9
TOOL_OUTCOME_CONFIDENCE = 0.7


_TOOL_STATUS_TO_OUTCOME_KIND: dict[str, DialogueOutcomeKind] = {
    "succeeded": DialogueOutcomeKind.CONTINUED,
    "success": DialogueOutcomeKind.CONTINUED,
    "ok": DialogueOutcomeKind.CONTINUED,
    "completed": DialogueOutcomeKind.CONTINUED,
    "failed": DialogueOutcomeKind.REJECTED,
    "failure": DialogueOutcomeKind.REJECTED,
    "error": DialogueOutcomeKind.REJECTED,
    "timeout": DialogueOutcomeKind.REJECTED,
    "blocked": DialogueOutcomeKind.REJECTED,
    "pending_confirmation": DialogueOutcomeKind.DEFERRED,
    "pending": DialogueOutcomeKind.DEFERRED,
    "deferred": DialogueOutcomeKind.DEFERRED,
}


def pe_continued_evidence_from_prediction_error(
    *,
    prediction_error_snapshot: object | None,
    wave_id: str,
    magnitude_threshold: float = PE_CONTINUED_MAGNITUDE_THRESHOLD,
) -> tuple[DialogueOutcomeEvidence, ...]:
    """Emit a low-confidence ``CONTINUED`` evidence when PE is small.

    Reads only the published ``prediction_error`` owner snapshot. When
    bootstrap is true the producer stays silent because there is no
    previous turn to attribute continuance to.
    """

    if prediction_error_snapshot is None:
        return ()
    bootstrap = getattr(prediction_error_snapshot, "bootstrap", True)
    if bootstrap:
        return ()
    error = getattr(prediction_error_snapshot, "error", None)
    if error is None:
        return ()
    magnitude = float(getattr(error, "magnitude", 0.0))
    if magnitude > magnitude_threshold:
        return ()
    evidence_id = f"prediction_error:{wave_id}:continued"
    return (
        DialogueOutcomeEvidence(
            evidence_id=evidence_id,
            source=DialogueOutcomeEvidenceSource.EVALUATION,
            source_owner="PredictionErrorModule",
            outcome_kind=DialogueOutcomeKind.CONTINUED,
            confidence=PE_CONTINUED_CONFIDENCE,
            evidence_refs=(f"prediction_error_magnitude:{magnitude:.4f}",),
            description=(
                "Low PE magnitude on a non-bootstrap turn; conservative "
                "structural evidence that the prior turn was continued."
            ),
        ),
    )


def commitment_outcome_evidence_from_commitment(
    *,
    commitment_snapshot: object | None,
    wave_id: str,
    current_turn_index: int,
) -> tuple[DialogueOutcomeEvidence, ...]:
    """Emit typed evidence from commitment lifecycle outcome transitions.

    Reads only the published ``commitment`` owner snapshot. Each lifecycle
    entry whose ``last_outcome_at_turn`` matches the current turn becomes
    one piece of evidence. Mapping is structural and avoids any text
    inference.
    """

    if commitment_snapshot is None:
        return ()
    lifecycle_entries = getattr(commitment_snapshot, "lifecycle_entries", ()) or ()
    evidence: list[DialogueOutcomeEvidence] = []
    for entry in lifecycle_entries:
        last_outcome = getattr(entry, "last_outcome", None)
        last_outcome_at_turn = int(getattr(entry, "last_outcome_at_turn", -1))
        record_id = getattr(entry, "record_id", "")
        if last_outcome is None or last_outcome_at_turn != current_turn_index:
            continue
        outcome_kind = _commitment_outcome_to_dialogue_outcome(last_outcome)
        if outcome_kind is None:
            continue
        evidence.append(
            DialogueOutcomeEvidence(
                evidence_id=f"commitment:{record_id}:{wave_id}:{last_outcome.value}",
                source=DialogueOutcomeEvidenceSource.OWNER_SNAPSHOT,
                source_owner="CommitmentModule",
                outcome_kind=outcome_kind,
                confidence=COMMITMENT_OUTCOME_CONFIDENCE,
                evidence_refs=(
                    f"commitment_record:{record_id}",
                    f"commitment_outcome:{last_outcome.value}",
                ),
                description=(
                    "Commitment lifecycle published a typed outcome on this "
                    "turn; trace mapping is structural, no text inference."
                ),
            )
        )
    return tuple(evidence)


def scene_closed_evidence(
    *,
    scene_id: str,
    reason: str,
    prediction_id: str | None = None,
) -> DialogueOutcomeEvidence:
    """Build typed evidence for a scene boundary close."""

    evidence_refs = [f"scene:{scene_id}", f"reason:{reason}"]
    if prediction_id:
        evidence_refs.append(f"scene_prediction:{prediction_id}")
    return DialogueOutcomeEvidence(
        evidence_id=f"scene_closed:{scene_id}:{reason}",
        source=DialogueOutcomeEvidenceSource.SCENE_EVENT,
        source_owner="SceneManager",
        outcome_kind=DialogueOutcomeKind.SCENE_CLOSED,
        confidence=SCENE_CLOSED_CONFIDENCE,
        evidence_refs=tuple(evidence_refs),
        description="Scene boundary closed; outcome is structural, not inferred.",
    )


def tool_outcome_evidence_from_environment_outcome(
    *,
    environment_outcome: EnvironmentOutcome,
    tool_name: str,
) -> tuple[DialogueOutcomeEvidence, ...]:
    """Map a tool ``EnvironmentOutcome.status`` to typed dialogue evidence.

    Reads only structural fields (``status``, ``action_id``, ``confidence``,
    ``prediction_id``). Free-form ``summary`` / ``detail`` text is not
    inspected. Unknown statuses produce no evidence.
    """

    status_key = environment_outcome.status.strip().lower()
    outcome_kind = _TOOL_STATUS_TO_OUTCOME_KIND.get(status_key)
    if outcome_kind is None:
        return ()
    refs: list[str] = [
        f"environment_outcome:{environment_outcome.outcome_id}",
        f"tool:{tool_name}",
        f"tool_status:{status_key}",
        f"action:{environment_outcome.action_id}",
    ]
    if environment_outcome.prediction_id is not None:
        refs.append(f"tool_prediction:{environment_outcome.prediction_id}")
    confidence = max(
        0.0,
        min(1.0, float(environment_outcome.confidence) * TOOL_OUTCOME_CONFIDENCE),
    )
    return (
        DialogueOutcomeEvidence(
            evidence_id=f"tool_outcome:{environment_outcome.outcome_id}",
            source=DialogueOutcomeEvidenceSource.OWNER_SNAPSHOT,
            source_owner=f"AffordanceInvoker:{tool_name}",
            outcome_kind=outcome_kind,
            confidence=confidence,
            evidence_refs=tuple(dict.fromkeys(refs)),
            description=(
                "Tool result EnvironmentOutcome.status mapped structurally; "
                "free-form summary/detail not inspected."
            ),
        ),
    )


_EXTERNAL_KIND_TO_STRUCTURAL_OUTCOME: dict[
    DialogueExternalOutcomeKind, DialogueOutcomeKind
] = {
    # Positive external outcomes map to CLARIFIED (resolved the turn).
    DialogueExternalOutcomeKind.HELPED: DialogueOutcomeKind.CLARIFIED,
    DialogueExternalOutcomeKind.FELT_HEARD: DialogueOutcomeKind.CLARIFIED,
    DialogueExternalOutcomeKind.DECISION_CLEARER: DialogueOutcomeKind.CLARIFIED,
    # COME_BACK = user wants to return later; structurally DEFERRED.
    DialogueExternalOutcomeKind.COME_BACK: DialogueOutcomeKind.DEFERRED,
    # Negative external outcomes map to CORRECTED (user explicitly pushed
    # back). UNSAFE and ABANDONED use REJECTED (stronger rejection).
    DialogueExternalOutcomeKind.MISSED: DialogueOutcomeKind.CORRECTED,
    DialogueExternalOutcomeKind.OVER_DIRECTIVE: DialogueOutcomeKind.CORRECTED,
    DialogueExternalOutcomeKind.UNSAFE: DialogueOutcomeKind.REJECTED,
    DialogueExternalOutcomeKind.ABANDONED: DialogueOutcomeKind.REJECTED,
    # W3-A LTV outcomes. Conversion-funnel events are projected into the
    # structural-replay vocabulary so dialogue-trace consumers (replay /
    # evidence reports) see them as ordinary outcomes; the authoritative
    # business signal stays on the dialogue_external_outcome snapshot.
    DialogueExternalOutcomeKind.LEAD_QUALIFIED: DialogueOutcomeKind.CLARIFIED,
    DialogueExternalOutcomeKind.RECOMMENDATION_MADE: DialogueOutcomeKind.CLARIFIED,
    DialogueExternalOutcomeKind.PURCHASE_CONFIRMED: DialogueOutcomeKind.CLARIFIED,
    DialogueExternalOutcomeKind.REPURCHASE: DialogueOutcomeKind.CLARIFIED,
    # CHURNED is structurally a long-horizon REJECTED — the relationship
    # ended without recovery despite repair attempts.
    DialogueExternalOutcomeKind.CHURNED: DialogueOutcomeKind.REJECTED,
}


def structural_outcome_evidence_from_external(
    evidence: DialogueExternalOutcomeEvidence,
) -> DialogueOutcomeEvidence | None:
    """Produce a structural ``DialogueOutcomeEvidence`` from an external entry.

    The structural evidence is attached to the dialogue trace for replay /
    reporting purposes. It is a lossy projection of the external kind into
    the conservative trace vocabulary; the authoritative signal remains on
    the ``dialogue_external_outcome`` snapshot.
    """

    outcome_kind = _EXTERNAL_KIND_TO_STRUCTURAL_OUTCOME.get(evidence.kind)
    if outcome_kind is None:
        return None
    return DialogueOutcomeEvidence(
        evidence_id=f"external-bridge:{evidence.evidence_id}",
        source=DialogueOutcomeEvidenceSource.SCENE_EVENT,
        source_owner="DialogueExternalOutcomeModule",
        outcome_kind=outcome_kind,
        confidence=float(evidence.confidence),
        evidence_refs=(
            f"external_kind:{evidence.kind.value}",
            f"external_source:{evidence.source.value}",
            f"external_ref:{evidence.evidence_ref}",
        ),
        description=(
            "Structural projection of DialogueExternalOutcomeEvidence for dialogue "
            "trace replay; authoritative signal is on the dialogue_external_outcome "
            "snapshot slot."
        ),
    )


def _commitment_outcome_to_dialogue_outcome(last_outcome: object) -> DialogueOutcomeKind | None:
    name = getattr(last_outcome, "name", None)
    if name is None:
        return None
    if name == "REJECTED":
        return DialogueOutcomeKind.REJECTED
    if name == "STALLED":
        return DialogueOutcomeKind.DEFERRED
    if name == "COMPLETED":
        return DialogueOutcomeKind.CLARIFIED
    if name == "PROGRESSED":
        return DialogueOutcomeKind.CONTINUED
    return None


__all__ = [
    "PE_CONTINUED_CONFIDENCE",
    "PE_CONTINUED_MAGNITUDE_THRESHOLD",
    "COMMITMENT_OUTCOME_CONFIDENCE",
    "SCENE_CLOSED_CONFIDENCE",
    "TOOL_OUTCOME_CONFIDENCE",
    "commitment_outcome_evidence_from_commitment",
    "pe_continued_evidence_from_prediction_error",
    "scene_closed_evidence",
    "structural_outcome_evidence_from_external",
    "tool_outcome_evidence_from_environment_outcome",
]
