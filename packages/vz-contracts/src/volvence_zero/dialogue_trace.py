"""Dialogue trace contracts.

The dialogue trace layer records dialogue actions and their later observable
outcomes for replay/evidence. It does not own prediction-error semantics and
does not classify user text.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from volvence_zero.environment import EnvironmentFrame


class DialogueActionKind(str, Enum):
    """Structural dialogue action surface, not a semantic rule set."""

    ASSISTANT_RESPONSE = "assistant_response"


class DialogueOutcomeKind(str, Enum):
    """Conservative outcome taxonomy for dialogue replay."""

    UNKNOWN = "unknown"
    CONTINUED = "continued"
    CLARIFIED = "clarified"
    CORRECTED = "corrected"
    REJECTED = "rejected"
    SCENE_CLOSED = "scene_closed"
    DEFERRED = "deferred"


class DialogueOutcomeEvidenceSource(str, Enum):
    """Typed source of dialogue outcome evidence."""

    OWNER_SNAPSHOT = "owner_snapshot"
    EVALUATION = "evaluation"
    SOCIAL_PREDICTION = "social_prediction"
    SCENE_EVENT = "scene_event"


class DialogueExternalOutcomeKind(str, Enum):
    """Closed v0 vocabulary for *externally*-produced dialogue outcomes.

    This vocabulary is distinct from :class:`DialogueOutcomeKind`:

    * ``DialogueOutcomeKind`` is the structural replay taxonomy used by the
      dialogue trace store (e.g. ``CONTINUED``, ``REJECTED``, ``SCENE_CLOSED``).
    * ``DialogueExternalOutcomeKind`` is the *external-signal* taxonomy used
      by rupture / repair evidence (e.g. ``MISSED``, ``DECISION_CLEARER``)
      and (since W3-A) by LTV / private-domain conversion-funnel evidence
      (e.g. ``PURCHASE_CONFIRMED``, ``CHURNED``).

    Adding a new value requires a typed evidence source capable of producing
    it; free-text inference is not a typed source.

    The conversion-funnel block (W3-A) is fed by external CRM / payments
    integrations through the typed feedback envelope; the platform never
    infers these labels from chat text. ``submit_dialogue_outcome`` accepts
    them, the four downstream mapping tables (PE bias, regime score,
    structural projection, rupture mapping) carry explicit semantics for
    each new value.
    """

    HELPED = "helped"
    FELT_HEARD = "felt_heard"
    MISSED = "missed"
    OVER_DIRECTIVE = "over_directive"
    DECISION_CLEARER = "decision_clearer"
    COME_BACK = "come_back"
    UNSAFE = "unsafe"
    ABANDONED = "abandoned"
    # ------------------------------------------------------------------
    # W3-A conversion / LTV vocabulary. Sourced from external CRM /
    # payments / human-review evidence; never inferred from chat text.
    # ------------------------------------------------------------------
    LEAD_QUALIFIED = "lead_qualified"
    RECOMMENDATION_MADE = "recommendation_made"
    PURCHASE_CONFIRMED = "purchase_confirmed"
    REPURCHASE = "repurchase"
    CHURNED = "churned"


class DialogueExternalOutcomeEvidenceSource(str, Enum):
    """Typed source of :class:`DialogueExternalOutcomeEvidence`.

    LLM proposal is present in the enum so the contract is stable, but
    runtime intake is gated behind an explicit ``BrainConfig`` flag in v0.
    """

    USER_EXPLICIT = "user_explicit"
    HUMAN_REVIEW = "human_review"
    ENVIRONMENT = "environment"
    LLM_PROPOSAL = "llm_proposal"


class DialogueResolutionStatus(str, Enum):
    """Resolution state for a previous dialogue trace."""

    PENDING = "pending"
    RESOLVED = "resolved"
    STALE = "stale"


@dataclass(frozen=True)
class DialogueOutcomeEvidence:
    """Structured evidence produced by an owner or evaluation readout.

    The trace layer may map this evidence to ``DialogueOutcomeKind``. It must
    not parse raw user text to produce this shape.
    """

    evidence_id: str
    source: DialogueOutcomeEvidenceSource
    source_owner: str
    outcome_kind: DialogueOutcomeKind
    confidence: float
    evidence_refs: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        _require_non_empty("evidence_id", self.evidence_id)
        _require_non_empty("source_owner", self.source_owner)
        _require_unit_interval("confidence", self.confidence)
        _require_unique_non_empty("evidence_refs", self.evidence_refs)


@dataclass(frozen=True)
class DialogueExternalOutcomeEvidence:
    """External outcome evidence for rupture / repair.

    Produced by :func:`submit_dialogue_outcome` adapters (wired in vz-runtime)
    and published on the ``dialogue_external_outcome`` snapshot slot. This
    type carries only external-signal provenance; it does not model the
    dialogue action being evaluated (that lives in :class:`DialogueActionTrace`).
    """

    evidence_id: str
    turn_index: int
    kind: DialogueExternalOutcomeKind
    source: DialogueExternalOutcomeEvidenceSource
    confidence: float
    evidence_ref: str
    description: str = ""

    def __post_init__(self) -> None:
        _require_non_empty("evidence_id", self.evidence_id)
        _require_non_empty("evidence_ref", self.evidence_ref)
        _require_non_negative_int("turn_index", self.turn_index)
        _require_unit_interval("confidence", self.confidence)


@dataclass(frozen=True)
class DialogueExternalOutcomeSnapshot:
    """Per-turn readout of external outcome evidence.

    Owned by ``DialogueExternalOutcomeModule`` (vz-runtime). Consumers
    (``PredictionErrorModule``, ``RegimeModule``, ``RuptureStateModule``,
    ``ReflectionEngine``) read this snapshot and integrate its entries
    inside their own ``process(...)`` paths — no external caller mutates
    those owners' internal state.
    """

    turn_index: int
    entries: tuple[DialogueExternalOutcomeEvidence, ...]
    description: str

    def __post_init__(self) -> None:
        _require_non_negative_int("turn_index", self.turn_index)
        evidence_ids = tuple(entry.evidence_id for entry in self.entries)
        _require_unique_non_empty("entries.evidence_id", evidence_ids)
        for entry in self.entries:
            if entry.turn_index > self.turn_index:
                raise ValueError(
                    "DialogueExternalOutcomeSnapshot.entries must not carry evidence "
                    "from a later turn than the snapshot's turn_index."
                )


@dataclass(frozen=True)
class DialogueOutcomeTrace:
    """Outcome evidence linked to a previous dialogue action."""

    outcome_id: str
    previous_trace_id: str
    observed_trace_id: str
    observed_turn_index: int
    kind: DialogueOutcomeKind
    evidence_refs: tuple[str, ...] = ()
    prediction_error_refs: tuple[str, ...] = ()
    structured_evidence: tuple[DialogueOutcomeEvidence, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        _require_non_empty("outcome_id", self.outcome_id)
        _require_non_empty("previous_trace_id", self.previous_trace_id)
        _require_non_empty("observed_trace_id", self.observed_trace_id)
        _require_non_negative_int("observed_turn_index", self.observed_turn_index)
        _require_unique_non_empty("evidence_refs", self.evidence_refs)
        _require_unique_non_empty("prediction_error_refs", self.prediction_error_refs)
        evidence_ids = tuple(evidence.evidence_id for evidence in self.structured_evidence)
        _require_unique_non_empty("structured_evidence.evidence_id", evidence_ids)


@dataclass(frozen=True)
class DialogueActionTrace:
    """Replay-safe record of one assistant dialogue action."""

    trace_id: str
    event_id: str
    wave_id: str
    turn_index: int
    action_kind: DialogueActionKind
    environment_frame: EnvironmentFrame
    environment_event_kind: str
    environment_trigger_kind: str
    active_regime: str | None
    active_abstract_action: str | None
    response_rationale: str
    prediction_id: str | None
    outcome: DialogueOutcomeTrace
    response_text_hash: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        _require_non_empty("trace_id", self.trace_id)
        _require_non_empty("event_id", self.event_id)
        _require_non_empty("wave_id", self.wave_id)
        _require_non_negative_int("turn_index", self.turn_index)
        _require_non_empty("environment_event_kind", self.environment_event_kind)
        _require_non_empty("environment_trigger_kind", self.environment_trigger_kind)
        if self.prediction_id is not None:
            _require_non_empty("prediction_id", self.prediction_id)
        if self.response_text_hash:
            _require_non_empty("response_text_hash", self.response_text_hash)


@dataclass(frozen=True)
class DialogueOutcomeResolution:
    """Resolution record emitted when a later turn settles prior evidence."""

    previous_trace_id: str
    observed_trace_id: str
    status: DialogueResolutionStatus
    outcome: DialogueOutcomeTrace
    description: str

    def __post_init__(self) -> None:
        _require_non_empty("previous_trace_id", self.previous_trace_id)
        _require_non_empty("observed_trace_id", self.observed_trace_id)
        _require_non_empty("description", self.description)


@dataclass(frozen=True)
class DialogueTraceSnapshot:
    """Session-local dialogue trace readout for replay and evidence."""

    traces: tuple[DialogueActionTrace, ...]
    unresolved_trace_ids: tuple[str, ...]
    resolved_outcomes: tuple[DialogueOutcomeTrace, ...]
    description: str

    def __post_init__(self) -> None:
        trace_ids = tuple(trace.trace_id for trace in self.traces)
        _require_unique_non_empty("traces.trace_id", trace_ids)
        _require_unique_non_empty("unresolved_trace_ids", self.unresolved_trace_ids)
        outcome_ids = tuple(outcome.outcome_id for outcome in self.resolved_outcomes)
        _require_unique_non_empty("resolved_outcomes.outcome_id", outcome_ids)
        _require_non_empty("description", self.description)


def build_unknown_dialogue_outcome(
    *,
    previous_trace_id: str,
    observed_trace_id: str,
    observed_turn_index: int,
    evidence_refs: tuple[str, ...] = (),
    prediction_error_refs: tuple[str, ...] = (),
    structured_evidence: tuple[DialogueOutcomeEvidence, ...] = (),
) -> DialogueOutcomeTrace:
    """Build the conservative default outcome without reading user text."""

    return DialogueOutcomeTrace(
        outcome_id=f"{previous_trace_id}:outcome:{observed_trace_id}",
        previous_trace_id=previous_trace_id,
        observed_trace_id=observed_trace_id,
        observed_turn_index=observed_turn_index,
        kind=DialogueOutcomeKind.UNKNOWN,
        evidence_refs=evidence_refs,
        prediction_error_refs=prediction_error_refs,
        structured_evidence=structured_evidence,
        description="Outcome is unresolved semantically; trace keeps PE linkage only.",
    )


def _require_non_empty(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_non_empty_items(field_name: str, values: tuple[str, ...]) -> None:
    for value in values:
        if not value.strip():
            raise ValueError(f"{field_name} entries must be non-empty")


def _require_unique_non_empty(field_name: str, values: tuple[str, ...]) -> None:
    _require_non_empty_items(field_name, values)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} entries must be unique")


def _require_non_negative_int(field_name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _require_unit_interval(field_name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {value!r}")


__all__ = [
    "DialogueActionKind",
    "DialogueActionTrace",
    "DialogueExternalOutcomeEvidence",
    "DialogueExternalOutcomeEvidenceSource",
    "DialogueExternalOutcomeKind",
    "DialogueExternalOutcomeSnapshot",
    "DialogueOutcomeEvidence",
    "DialogueOutcomeEvidenceSource",
    "DialogueOutcomeKind",
    "DialogueOutcomeResolution",
    "DialogueOutcomeTrace",
    "DialogueResolutionStatus",
    "DialogueTraceSnapshot",
    "build_unknown_dialogue_outcome",
]
