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
    # ------------------------------------------------------------------
    # Task-execution outcome vocabulary (coding-lab evidence lane).
    # Sourced from a deterministic environment oracle (test suites /
    # build gates) through the typed ENVIRONMENT evidence source; never
    # inferred from chat text. Deliberately relationship-neutral: a
    # settled task outcome is world-track evidence, so its PE bias rows
    # touch only the task / action axes (see _EXTERNAL_OUTCOME_AXIS_BIAS)
    # and neither value produces rupture evidence.
    # ------------------------------------------------------------------
    TASK_VERIFIED = "task_verified"
    TASK_REGRESSED = "task_regressed"


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
    # Join key for bank attribution, paired with ``action_turn_index``. Empty
    # means the submitting surface did not identify a session, which is
    # allowed -- the outcome still reaches its consumers -- but such an entry
    # cannot be attributed to a bank set, so it is excluded from credit
    # assignment rather than being guessed at.
    session_scope: str = ""
    # The turn whose *action* this outcome evaluates.
    #
    # Deliberately distinct from ``turn_index``, which the submitting surfaces
    # already use with two different meanings: the HTTP feedback endpoint binds
    # to the upcoming turn that will consume the evidence, while the runner
    # default binds to the turn that just finished. Both are defensible for
    # consumption scheduling, and neither is changed here -- but attribution
    # needs exactly one of them (the producing action), so it gets its own
    # field instead of overloading a value whose meaning depends on the caller.
    #
    # ``-1`` means "not declared"; such evidence is counted but never
    # attributed, because guessing which turn produced it would silently
    # assign credit to the wrong bank set.
    action_turn_index: int = -1

    def __post_init__(self) -> None:
        _require_non_empty("evidence_id", self.evidence_id)
        _require_non_empty("evidence_ref", self.evidence_ref)
        _require_non_negative_int("turn_index", self.turn_index)
        _require_unit_interval("confidence", self.confidence)
        if self.action_turn_index < -1:
            raise ValueError(
                "DialogueExternalOutcomeEvidence action_turn_index must be a "
                "non-negative turn or -1 for 'not declared'."
            )

    @property
    def is_attributable(self) -> bool:
        """Whether this evidence can be joined to a conditioning lineage.

        Both halves of the join key must be present. A partially-keyed entry
        is treated as unattributable rather than joined on session alone,
        which would spread one outcome across every turn of the session.
        """

        return bool(self.session_scope) and self.action_turn_index >= 0


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
class ConditioningLineage:
    """Which conditioning banks were live when a dialogue action was taken.

    This is the right-hand side of the external-outcome attribution join: an
    outcome reported later carries ``(session_scope, turn_index)``, and this
    record is what makes that pair resolvable to "these banks, at these
    versions, produced that action". Without it an outcome can be counted but
    not attributed, so bank-level credit cannot be computed at all.

    Fields are flat strings rather than the contract enums/dataclasses so a
    trace row stays trivially serialisable to the durable JSONL sink and stays
    readable after a bank vocabulary change. ``selected_bank_set`` therefore
    holds ``ConditioningBankType`` *values*, not members.

    The three artifact versions are empty until the corresponding component
    exists: a bank set can be recorded long before there is a learned encoder,
    and recording an empty version is honest, whereas omitting the field would
    make old and new rows indistinguishable.
    """

    session_scope: str
    selected_bank_set: tuple[str, ...] = ()
    bank_fingerprints: tuple[tuple[str, str], ...] = ()
    state_encoder_version: str = ""
    prefix_generator_version: str = ""
    router_version: str = ""
    # State KV P4-c: per-bank scores from the routing policy named by
    # ``router_version`` when that policy actually scored candidates (the
    # Top-K router). Empty for the deterministic select-all policy, which
    # has no scores by construction.
    router_scores: tuple[tuple[str, float], ...] = ()
    # State KV P4-c SHADOW audit: when the Top-K router runs report-only
    # (behaviour stays select-all), its would-be decision is recorded here
    # so negative-control evidence can verify that an irrelevant bank
    # scores low without flipping the policy live. Empty when no shadow
    # evaluation happened this turn.
    shadow_router_version: str = ""
    shadow_router_scores: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty("session_scope", self.session_scope)
        _require_unique_non_empty("selected_bank_set", self.selected_bank_set)
        fingerprinted = tuple(bank for bank, _ in self.bank_fingerprints)
        _require_unique_non_empty("bank_fingerprints.bank", fingerprinted)
        for bank, fingerprint in self.bank_fingerprints:
            if not fingerprint:
                raise ValueError(
                    "ConditioningLineage bank_fingerprints must carry a "
                    f"non-empty fingerprint for {bank!r}."
                )
        # A selected bank with no fingerprint cannot be attributed to a
        # specific state version, which defeats the point of recording it.
        missing = set(self.selected_bank_set) - set(fingerprinted)
        if missing:
            raise ValueError(
                "ConditioningLineage selected_bank_set entries must each have a "
                f"fingerprint; missing: {sorted(missing)}."
            )
        for label, scores in (
            ("router_scores", self.router_scores),
            ("shadow_router_scores", self.shadow_router_scores),
        ):
            banks_scored = tuple(bank for bank, _ in scores)
            if scores:
                _require_unique_non_empty(f"{label}.bank", banks_scored)
            for bank, score in scores:
                if not 0.0 <= score <= 1.0:
                    raise ValueError(
                        f"ConditioningLineage {label} must be in [0, 1]; "
                        f"got {score!r} for {bank!r}."
                    )
        if self.shadow_router_scores and not self.shadow_router_version:
            raise ValueError(
                "ConditioningLineage shadow_router_scores require a "
                "non-empty shadow_router_version naming the policy."
            )
        if self.router_scores and not self.router_version:
            raise ValueError(
                "ConditioningLineage router_scores require a non-empty "
                "router_version naming the policy."
            )


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
    conditioning_lineage: ConditioningLineage | None = None

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
    "ConditioningLineage",
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
