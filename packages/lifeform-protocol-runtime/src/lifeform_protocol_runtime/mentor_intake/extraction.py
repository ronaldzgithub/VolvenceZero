"""Typed artifact extraction for non-protocol mentor-intake kinds.

The classifier (:mod:`lifeform_protocol_runtime.mentor_intake.classification`)
decides *which* owner should receive a mentor's guidance. This module turns
the guidance into the typed artifact that owner consumes, WITHOUT becoming a
second owner of any runtime state:

* ``knowledge`` -> :class:`ReviewedKnowledgeDraft` for the kernel-owned
  ``domain_knowledge`` owner (via ``submit_reviewed_knowledge_event``).
* ``case`` -> :class:`SignatureCase` that compiles into the kernel-owned
  ``case_memory`` owner via the protocol compile path.
* ``protocol_revision`` -> :class:`ProtocolRevisionProposal`, validated
  against the narrow surface ``ProtocolRegistry.apply_revision`` actually
  supports, then routed through the R10 ModificationGate by the caller.
* ``experience`` -> a typed :class:`DialogueExternalOutcomeKind`. This one is
  NOT free-text inferred: the closed vocabulary requires a typed evidence
  source, so the mentor must pick the outcome explicitly (human-review
  source). We only validate the choice here.
"""

from __future__ import annotations

from dataclasses import dataclass

from volvence_zero.behavior_protocol import (
    ProposalEvidence,
    ProtocolRevisionChangeKind,
    ProtocolRevisionProposal,
    ProtocolRevisionTargetField,
    ReviewLevel,
    SignatureCase,
)
from volvence_zero.dialogue_trace import DialogueExternalOutcomeKind

from lifeform_protocol_runtime.document_uptake.extraction import LlmJsonClient
from lifeform_protocol_runtime.mentor_intake.prompts import (
    MENTOR_CASE_EXTRACTOR_SYSTEM_PROMPT,
    MENTOR_CASE_EXTRACTOR_USER_TEMPLATE,
    MENTOR_KNOWLEDGE_EXTRACTOR_SYSTEM_PROMPT,
    MENTOR_KNOWLEDGE_EXTRACTOR_USER_TEMPLATE,
)


# ---------------------------------------------------------------------------
# knowledge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewedKnowledgeDraft:
    """Fields for ``BrainSession.submit_reviewed_knowledge_event``.

    A structured projection of mentor guidance; the kernel-owned
    ``domain_knowledge`` owner remains the canonical store.
    """

    knowledge_id: str
    summary: str
    detail: str
    confidence: float
    source_label: str = "mentor-intake"
    relevance_hint: str = ""
    domain: str = ""

    def __post_init__(self) -> None:
        if not self.knowledge_id.strip():
            raise ValueError("ReviewedKnowledgeDraft.knowledge_id must be non-empty")
        if not self.summary.strip():
            raise ValueError("ReviewedKnowledgeDraft.summary must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"ReviewedKnowledgeDraft.confidence must be in [0, 1], "
                f"got {self.confidence!r}"
            )


def extract_reviewed_knowledge_from_guidance(
    guidance: str,
    *,
    llm_client: LlmJsonClient,
    knowledge_id: str,
    mentor_id: str = "anonymous-mentor",
    source_label: str = "mentor-intake",
) -> ReviewedKnowledgeDraft:
    """Convert free-text mentor knowledge guidance into a reviewed record."""

    if not guidance or not guidance.strip():
        raise ValueError(
            "extract_reviewed_knowledge_from_guidance: empty guidance"
        )
    payload = llm_client.complete_json(
        system_prompt=MENTOR_KNOWLEDGE_EXTRACTOR_SYSTEM_PROMPT,
        user_prompt=MENTOR_KNOWLEDGE_EXTRACTOR_USER_TEMPLATE.format(
            mentor_id=mentor_id,
            guidance=guidance.strip(),
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError(
            "extract_reviewed_knowledge_from_guidance: LLM returned "
            "non-object JSON"
        )
    summary = str(payload.get("summary") or "").strip()
    if not summary:
        raise ValueError(
            "extract_reviewed_knowledge_from_guidance: LLM omitted summary"
        )
    detail = str(payload.get("detail") or "").strip() or summary
    return ReviewedKnowledgeDraft(
        knowledge_id=knowledge_id,
        summary=summary,
        detail=detail,
        confidence=_clamp_unit(payload.get("confidence", 0.7)),
        source_label=source_label,
        relevance_hint=str(payload.get("relevance_hint") or "").strip(),
        domain=str(payload.get("domain") or "").strip(),
    )


# ---------------------------------------------------------------------------
# case
# ---------------------------------------------------------------------------


def extract_signature_case_from_guidance(
    guidance: str,
    *,
    llm_client: LlmJsonClient,
    case_id: str,
    mentor_id: str = "anonymous-mentor",
) -> SignatureCase:
    """Convert a free-text worked example into a reviewed ``SignatureCase``.

    The returned case carries only review-time fields; runtime lifecycle
    state stays owned by ``case_memory``. ``SignatureCase.__post_init__``
    enforces a non-empty ``intervention_ordering``.
    """

    if not guidance or not guidance.strip():
        raise ValueError(
            "extract_signature_case_from_guidance: empty guidance"
        )
    payload = llm_client.complete_json(
        system_prompt=MENTOR_CASE_EXTRACTOR_SYSTEM_PROMPT,
        user_prompt=MENTOR_CASE_EXTRACTOR_USER_TEMPLATE.format(
            mentor_id=mentor_id,
            guidance=guidance.strip(),
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError(
            "extract_signature_case_from_guidance: LLM returned non-object JSON"
        )
    ordering = _str_tuple(payload.get("intervention_ordering"))
    if not ordering:
        raise ValueError(
            "extract_signature_case_from_guidance: LLM omitted a non-empty "
            "intervention_ordering (case has no retrievable content)"
        )
    return SignatureCase(
        case_id=case_id,
        domain=str(payload.get("domain") or "").strip() or "general",
        problem_pattern=str(payload.get("problem_pattern") or "").strip(),
        user_state_pattern=str(payload.get("user_state_pattern") or "").strip(),
        risk_markers=_str_tuple(payload.get("risk_markers")),
        track_tags=(),
        regime_tags=(),
        intervention_ordering=ordering,
        outcome_label=str(payload.get("outcome_label") or "").strip() or "unknown",
        confidence=_clamp_unit(payload.get("confidence", 0.7)),
        description=str(payload.get("description") or "").strip(),
    )


# ---------------------------------------------------------------------------
# protocol_revision
# ---------------------------------------------------------------------------

# The set of (target_field, change_kind) combinations that
# ``ProtocolRegistry._apply_change`` actually implements. Anything else is
# rejected here (fail-loud) rather than surfacing a NotImplementedError from
# deep inside the apply path.
_SUPPORTED_REVISION_OPS: dict[
    ProtocolRevisionTargetField, frozenset[ProtocolRevisionChangeKind]
] = {
    ProtocolRevisionTargetField.STRATEGY_PRIOR: frozenset(
        {
            ProtocolRevisionChangeKind.WEIGHT_DECAY,
            ProtocolRevisionChangeKind.DEACTIVATE,
            ProtocolRevisionChangeKind.WEIGHT_REINFORCE,
            ProtocolRevisionChangeKind.ADD_STRATEGY,
        }
    ),
    ProtocolRevisionTargetField.KNOWLEDGE_SEED: frozenset(
        {ProtocolRevisionChangeKind.ARCHIVE}
    ),
    ProtocolRevisionTargetField.SIGNATURE_CASE: frozenset(
        {ProtocolRevisionChangeKind.ARCHIVE}
    ),
    ProtocolRevisionTargetField.BOUNDARY_CONTRACT: frozenset(
        {ProtocolRevisionChangeKind.BOUNDARY_REFINEMENT}
    ),
    ProtocolRevisionTargetField.IDENTITY_ASSERTION: frozenset(
        {ProtocolRevisionChangeKind.IDENTITY_CLARIFICATION}
    ),
}


def build_protocol_revision_proposal(
    *,
    proposal_id: str,
    target_protocol_id: str,
    target_field: str | ProtocolRevisionTargetField,
    target_entry_id: str,
    change_kind: str | ProtocolRevisionChangeKind,
    summary: str,
    observation_window_turns: int = 1,
    pe_signature: str = "mentor-intake",
    proposed_payload: dict | None = None,
    required_review_level: ReviewLevel = ReviewLevel.L3,
) -> ProtocolRevisionProposal:
    """Build a typed revision proposal from mentor-supplied targets.

    The mentor must name which loaded protocol + entry to revise and the
    mutation kind; we never synthesise those from free text. Rejects any
    ``(target_field, change_kind)`` combo the apply layer does not support
    (plus the field-agnostic ``PROTOCOL_RETIREMENT``).
    """

    field = _resolve_revision_field(target_field)
    kind = _resolve_revision_kind(change_kind)

    if kind is not ProtocolRevisionChangeKind.PROTOCOL_RETIREMENT:
        supported = _SUPPORTED_REVISION_OPS.get(field, frozenset())
        if kind not in supported:
            raise ValueError(
                "build_protocol_revision_proposal: unsupported revision "
                f"change_kind={kind.value!r} on target_field={field.value!r}. "
                "Supported: "
                + ", ".join(
                    f"{f.value}->[{', '.join(sorted(k.value for k in ks))}]"
                    for f, ks in _SUPPORTED_REVISION_OPS.items()
                )
                + "; plus field-agnostic protocol_retirement."
            )

    evidence = ProposalEvidence(
        observation_window_turns=max(1, int(observation_window_turns)),
        pe_signature=pe_signature or "mentor-intake",
        summary=summary.strip() or "mentor-intake revision",
    )
    return ProtocolRevisionProposal(
        proposal_id=proposal_id,
        target_protocol_id=target_protocol_id,
        target_field=field,
        target_entry_id=target_entry_id,
        change_kind=kind,
        evidence=evidence,
        proposed_payload=proposed_payload,
        required_review_level=required_review_level,
    )


# ---------------------------------------------------------------------------
# experience
# ---------------------------------------------------------------------------


def resolve_experience_outcome_kind(
    raw: str | DialogueExternalOutcomeKind,
) -> DialogueExternalOutcomeKind:
    """Validate a mentor-supplied outcome against the closed vocabulary.

    ``DialogueExternalOutcomeKind`` is a closed vocabulary that "requires a
    typed evidence source; free-text inference is not a typed source"
    (see ``vz-contracts`` dialogue_trace). The mentor (a human-review
    source) therefore picks the outcome explicitly; this never infers it.
    """

    if isinstance(raw, DialogueExternalOutcomeKind):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            "resolve_experience_outcome_kind: experience intake requires an "
            "explicit outcome_kind (free-text inference is not a typed source)"
        )
    try:
        return DialogueExternalOutcomeKind(raw.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(k.value for k in DialogueExternalOutcomeKind)
        raise ValueError(
            f"resolve_experience_outcome_kind: unknown outcome_kind "
            f"{raw!r}; allowed: {allowed}"
        ) from exc


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_revision_field(
    raw: str | ProtocolRevisionTargetField,
) -> ProtocolRevisionTargetField:
    if isinstance(raw, ProtocolRevisionTargetField):
        return raw
    try:
        return ProtocolRevisionTargetField(str(raw).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(f.value for f in ProtocolRevisionTargetField)
        raise ValueError(
            f"build_protocol_revision_proposal: unknown target_field "
            f"{raw!r}; allowed: {allowed}"
        ) from exc


def _resolve_revision_kind(
    raw: str | ProtocolRevisionChangeKind,
) -> ProtocolRevisionChangeKind:
    if isinstance(raw, ProtocolRevisionChangeKind):
        return raw
    try:
        return ProtocolRevisionChangeKind(str(raw).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(k.value for k in ProtocolRevisionChangeKind)
        raise ValueError(
            f"build_protocol_revision_proposal: unknown change_kind "
            f"{raw!r}; allowed: {allowed}"
        ) from exc


def _clamp_unit(raw: object) -> float:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.7
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _str_tuple(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in raw if str(item).strip())


__all__ = [
    "ReviewedKnowledgeDraft",
    "build_protocol_revision_proposal",
    "extract_reviewed_knowledge_from_guidance",
    "extract_signature_case_from_guidance",
    "resolve_experience_outcome_kind",
]
