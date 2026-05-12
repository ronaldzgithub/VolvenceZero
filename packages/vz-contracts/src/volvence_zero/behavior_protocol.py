"""Behavior Protocol Runtime — schema (packet 1.0).

Cross-wheel immutable contracts for the Behavior Protocol Runtime
described in ``docs/specs/protocol-runtime.md``. These envelopes are
*data only* — no registry, no activation logic, no fixture
adapters. The kernel-side owner is
``volvence_zero.protocol_runtime.ProtocolRegistryModule`` (in
``vz-cognition``); per-vertical fixture adapters live in each
``lifeform-domain-*`` wheel.

Why these live in ``vz-contracts`` (not in ``vz-cognition`` or some
``lifeform-*`` wheel):

* Boundary owners (``boundary_policy``), metacontroller, vitals, and
  strategy_playbook are kernel modules. They will eventually consume
  ``ActiveMixtureSnapshot`` directly. If the schema lived in
  ``vz-cognition`` the cross-wheel imports would still work, but
  ``vz-contracts`` is the canonical home for cross-wheel snapshot
  shapes (mirrors ``thinking.py`` / ``dialogue_trace.py`` /
  ``social_cognition.py``).
* Lifeform-side adapters (FixtureUptake, future DocumentUptake) need
  to construct ``BehaviorProtocol`` instances. They cannot import
  from ``vz-cognition`` (``vz-* ↛ lifeform-*`` invariant + product
  layer separation), so the constructible types must be in a
  cross-tier home.

Packet 1.0 scope:

* All frozen dataclasses + closed enums declared.
* Validation in ``__post_init__`` is *light*: id non-empty,
  uniqueness inside collections, ``BehaviorProtocol`` requires at
  least one success and one failure signal unless
  ``legacy_fixture=True``.
* No runtime-state fields. ``BehaviorProtocol`` is fully reviewed
  data; the registry holds working state out-of-band.

Fields that are spec-declared but unused by packet 1.0 (e.g. PE
revision metadata on ``StrategyPrior``, full TemporalArc progression
machinery, identity_gate machinery) are present in the schema so the
shape is final, but downstream packets will be the first real
consumers. Schema growth is preferred to schema churn — declare it
once, light up consumption packet by packet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Closed enums
# ---------------------------------------------------------------------------


class BehaviorProtocolSignalSource(str, Enum):
    """Closed vocabulary for ``SuccessSignal.measurable_via`` /
    ``FailureSignal.measurable_via``.

    Each member must correspond to a typed signal source already
    owned somewhere in the system (or, for packet 1.0,
    ``DRIVE_HOMEOSTASIS_*`` which derive from
    ``lifeform_core.vitals.DriveLevel.out_of_band``). The closed-
    enum invariant prevents fragmentation: extending the vocabulary
    requires adding a member here AND wiring a typed source — same
    pattern as ``RuptureKind``.

    Packet 1.0 ships the six members below. Packet 1.5a' adds two
    more (regime / retrieval direct signals) and lights up the
    ``USER_DROPOUT_OBSERVED`` detector. Extensions in later packets
    follow the same nuclear rule: PR adds enum member + typed signal
    source + at least one consumer test.
    """

    DRIVE_HOMEOSTASIS_HOLD = "drive_homeostasis_hold"
    DRIVE_HOMEOSTASIS_BREACH = "drive_homeostasis_breach"
    BOUNDARY_VIOLATION_FIRED = "boundary_violation_fired"
    RUPTURE_KIND_FIRED = "rupture_kind_fired"
    INTERLOCUTOR_ZONE_TRANSITION = "interlocutor_zone_transition"
    USER_DROPOUT_OBSERVED = "user_dropout_observed"
    # Packet 1.5a' (REGIME_TRANSITION_RECENT): fires when the
    # active regime just changed this turn — i.e. the
    # ``RegimeSnapshot.turns_in_current_regime`` is at its post-
    # transition floor (1 by canonical convention; 0 also accepted
    # for cold-start tolerance). Useful for protocols that want to
    # weight more heavily during regime-change windows (e.g.
    # crisis/repair protocols become more relevant at transitions).
    REGIME_TRANSITION_RECENT = "regime_transition_recent"
    # Packet 1.5a' (RETRIEVAL_HITS_PRESENT): fires when the
    # ``RetrievalPolicySnapshot`` is requesting any knowledge
    # domains (i.e. the retrieval policy expects relevant memory
    # / case lookups this turn). Useful for protocols whose
    # strategies depend on retrieval-grounded answers.
    RETRIEVAL_HITS_PRESENT = "retrieval_hits_present"
    # Packet 7.0: dialogue trace-derived sources (latency / length /
    # initiative-question proxies). The detectors read these via
    # ``dialogue_trace`` and ``interlocutor_state`` fields on the
    # canonical kernel snapshots. New members require detector
    # impl (see ``activation._signal_is_firing``) + at least one
    # contract test.
    USER_REPLY_LATENCY = "user_reply_latency"
    USER_REPLY_LENGTH = "user_reply_length"
    USER_INITIATIVE_QUESTION = "user_initiative_question"
    # Packet 7.0: AAC commitment lifecycle — protocol-level signal
    # for "user committed to next step" / "user broke a prior
    # commitment". Reads from ``commitment`` snapshot.
    COMMITMENT_FULFILLED = "commitment_fulfilled"
    COMMITMENT_BROKEN = "commitment_broken"


class ReviewStatus(str, Enum):
    """BehaviorProtocol lifecycle state (R10 ModificationGate gating)."""

    DRAFT = "draft"
    SHADOW = "shadow"
    ACTIVE = "active"
    # Packet 6.1 / 6.4: terminal lifecycle. RETIRED protocols are
    # filtered from active_mixture but kept in registry for audit
    # and R15 rollback.
    RETIRED = "retired"


class ReviewLevel(str, Enum):
    """R10 ModificationGate review level required to mutate this field.

    L1 = automated; L4 = human + admin. Spec ``§TaskUptake / Review
    分级``. Packet 1.0 only honours these tags as audit metadata;
    actual gate enforcement lights up in later packets.
    """

    L1 = "l1"
    L2 = "l2"
    L3 = "l3"
    L4 = "l4"


class BoundarySeverity(str, Enum):
    """How a boundary contract enforces.

    ``HARD_BLOCK``: block the action and downgrade.
    ``SOFT_REMIND``: emit a reminder/disclaimer but allow.
    ``ESCALATE_HUMAN``: pause and escalate (used for identity-level
    violations).
    """

    SOFT_REMIND = "soft_remind"
    HARD_BLOCK = "hard_block"
    ESCALATE_HUMAN = "escalate_human"


class ProtocolSourceKind(str, Enum):
    """Where the BehaviorProtocol came from."""

    FIXTURE = "fixture"
    PDF_UPTAKE = "pdf_uptake"
    MARKDOWN_UPTAKE = "markdown_uptake"
    TASK_DESCRIPTION = "task_description"
    API_INJECTION = "api_injection"
    DIRECTORY_SCAN = "directory_scan"


class ActivationReasonKind(str, Enum):
    """Which signal contributed to a protocol's activation weight."""

    CONTEXT_MATCH = "context_match"
    PE_HISTORY = "pe_history"
    DRIVE_COUPLING = "drive_coupling"
    IDENTITY_GATE = "identity_gate"
    MINIMUM_FLOOR = "minimum_floor"
    EQUAL_WEIGHT_FALLBACK = "equal_weight_fallback"


# ---------------------------------------------------------------------------
# Identity / boundaries / strategies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentityAssertion:
    """Protocol's compatibility statement vs the lifeform Identity Core.

    Read by ``ActivationController`` to compute ``identity_gate``.
    Packet 1.0 ships the dataclass shape but ``identity_gate`` is
    hard-coded to 1.0 in ``vz-cognition.protocol_runtime``; real
    R7-Self / R14-regime cross-checks light up in later packets.
    """

    requires_self_traits: tuple[str, ...] = ()
    forbidden_self_traits: tuple[str, ...] = ()
    required_regime_compatibility: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoundaryContract:
    """A single anti-pattern / hard-rule the protocol asserts.

    Compiled (packet 1.2+) into the consumer-side ``BoundaryPolicy``
    via ``compile_protocol_to_application_artifacts`` →
    ``BoundaryPriorHint`` (in ``vz-application``).
    ``trigger_reasons`` must reference typed signal sources, not
    user-text keywords (no-keyword-matching-hacks invariant).

    Schema relationship to ``BoundaryPriorHint`` (vz-application):

    * Packet 1.0 fields shared 1:1: ``boundary_id`` (→ ``hint_id``
      with namespace prefix), ``description``, ``trigger_reasons``,
      ``blocked_topics``, ``required_disclaimers``,
      ``refer_out_required``, ``confidence``.
    * Packet 1.2 added (lossless conversion): ``regime_id``,
      ``answer_depth_limit_hint``, ``clarification_required`` —
      these mirror ``BoundaryPriorHint`` so reviewed verticals like
      ``GrowthAdvisorBoundaryPrior`` pass through unchanged.
    * Protocol-only metadata (NOT on ``BoundaryPriorHint``):
      ``severity`` (used for downstream rendering hints) and
      ``review_level`` (used by ``ModificationGate``); these stay
      out of the application-side hint.
    """

    boundary_id: str
    description: str
    trigger_reasons: tuple[str, ...]
    blocked_topics: tuple[str, ...] = ()
    required_disclaimers: tuple[str, ...] = ()
    refer_out_required: bool = False
    regime_id: str | None = None
    answer_depth_limit_hint: str = ""
    clarification_required: bool = False
    severity: BoundarySeverity = BoundarySeverity.HARD_BLOCK
    review_level: ReviewLevel = ReviewLevel.L3
    confidence: float = 0.9

    def __post_init__(self) -> None:
        if not self.boundary_id.strip():
            raise ValueError("BoundaryContract.boundary_id must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"BoundaryContract.confidence must be in [0, 1], "
                f"got {self.confidence!r}"
            )


@dataclass(frozen=True)
class StrategyPriorRevision:
    """Append-only entry recording one PE-driven weight change."""

    revision_id: str
    revised_at_tick: int
    delta: float
    reason: str

    def __post_init__(self) -> None:
        if not self.revision_id.strip():
            raise ValueError(
                "StrategyPriorRevision.revision_id must be non-empty"
            )


@dataclass(frozen=True)
class StrategyPrior:
    """A strategy / playbook prior with PE-revisable weight metadata.

    Compiled (packet 1.3b+) into the consumer-side ``strategy_playbook``
    via ``compile_protocol_to_application_artifacts`` →
    ``PlaybookRule`` (in ``vz-application``).

    Schema relationship to ``PlaybookRule`` (vz-application):

    * Packet 1.0 fields shared 1:1 (modulo naming):
      ``rule_id`` (→ namespaced ``PlaybookRule.rule_id``),
      ``problem_pattern``, ``recommended_ordering``,
      ``recommended_pacing``, ``avoid_patterns``, ``confidence``,
      ``description``, ``applicability_phase`` (→
      ``PlaybookRule.applicability_scope``; same role, different
      historical name).
    * Packet 1.3b added (lossless conversion to ``PlaybookRule``):
      ``recommended_regime``, ``knowledge_weight_hint``,
      ``experience_weight_hint`` — these mirror
      ``PlaybookRule`` so reviewed verticals like
      ``GrowthAdvisorStrategyPrior`` pass through unchanged.
    * Protocol-only metadata (NOT on ``PlaybookRule``):
      ``initial_weight`` / ``pe_decay_rate`` /
      ``pe_reinforce_rate`` / ``minimum_weight_floor`` /
      ``revision_history`` are PE-revision metadata consumed by
      the future activation controller (packet 1.5+);
      ``StrategyPlaybookModule`` does not read them, so they
      stay out of ``PlaybookRule``.
    * ``PlaybookRule`` has additional optional continuum fields
      (``continuum_band_id`` / ``mean_continuum_position``) that
      protocols don't seed; the compile uses ``PlaybookRule``
      defaults.
    """

    rule_id: str
    problem_pattern: str
    recommended_ordering: tuple[str, ...]
    recommended_pacing: str
    avoid_patterns: tuple[str, ...] = ()
    applicability_phase: tuple[str, ...] = ()
    recommended_regime: str | None = None
    knowledge_weight_hint: float = 0.45
    experience_weight_hint: float = 0.65
    initial_weight: float = 1.0
    pe_decay_rate: float = 0.0
    pe_reinforce_rate: float = 0.0
    minimum_weight_floor: float = 0.0
    revision_history: tuple[StrategyPriorRevision, ...] = ()
    confidence: float = 0.85
    description: str = ""

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise ValueError("StrategyPrior.rule_id must be non-empty")
        if not 0.0 <= self.knowledge_weight_hint <= 1.0:
            raise ValueError(
                f"StrategyPrior.knowledge_weight_hint must be in [0, 1], "
                f"got {self.knowledge_weight_hint!r}"
            )
        if not 0.0 <= self.experience_weight_hint <= 1.0:
            raise ValueError(
                f"StrategyPrior.experience_weight_hint must be in [0, 1], "
                f"got {self.experience_weight_hint!r}"
            )
        if not 0.0 <= self.initial_weight <= 1.0:
            raise ValueError(
                f"StrategyPrior.initial_weight must be in [0, 1], "
                f"got {self.initial_weight!r}"
            )
        if self.pe_decay_rate < 0.0:
            raise ValueError(
                f"StrategyPrior.pe_decay_rate must be >= 0, "
                f"got {self.pe_decay_rate!r}"
            )
        if self.pe_reinforce_rate < 0.0:
            raise ValueError(
                f"StrategyPrior.pe_reinforce_rate must be >= 0, "
                f"got {self.pe_reinforce_rate!r}"
            )
        if not 0.0 <= self.minimum_weight_floor <= self.initial_weight:
            raise ValueError(
                f"StrategyPrior.minimum_weight_floor must be in "
                f"[0, initial_weight={self.initial_weight}], "
                f"got {self.minimum_weight_floor!r}"
            )


# ---------------------------------------------------------------------------
# Temporal arc (relationship-state-driven, not calendar-driven)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressionSignal:
    """Signal that advances or retreats a TemporalArc phase pointer.

    Packet 1.0 ships an empty placeholder shape (no driving logic).
    Packet 1.4 will populate this with PE-driven entry/exit
    conditions.
    """

    signal_id: str
    measurable_via: BehaviorProtocolSignalSource
    threshold: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class DriveExpectation:
    """Expected homeostatic state of a named drive in a phase."""

    drive_name: str
    expected_band: tuple[float, float]


@dataclass(frozen=True)
class TemporalPhase:
    """One phase of the protocol's temporal arc."""

    phase_id: str
    description: str = ""
    entry_conditions: tuple[ProgressionSignal, ...] = ()
    exit_conditions: tuple[ProgressionSignal, ...] = ()
    expected_drives_state: tuple[DriveExpectation, ...] = ()


@dataclass(frozen=True)
class TemporalArc:
    """The protocol's full phase arc.

    Phase advancement is driven by typed
    :class:`ProgressionSignal` evidence (see packet 5.0
    ``ProtocolPhaseModule``); the calendar-tag fallback used in
    packet 1.0 fixture conversion is a transition-period
    placeholder, not the canonical mechanism.

    ``progression_signals`` (packet 5.0) declares arc-level
    signals applicable to any phase transition. Per-phase
    ``entry_conditions`` / ``exit_conditions`` on
    :class:`TemporalPhase` provide finer control.
    """

    phases: tuple[TemporalPhase, ...] = ()
    progression_signals: tuple[ProgressionSignal, ...] = ()

    def __post_init__(self) -> None:
        seen = set()
        for phase in self.phases:
            if phase.phase_id in seen:
                raise ValueError(
                    f"TemporalArc.phases.phase_id duplicate: "
                    f"{phase.phase_id!r}"
                )
            seen.add(phase.phase_id)
        signal_ids = set()
        for signal in self.progression_signals:
            if signal.signal_id in signal_ids:
                raise ValueError(
                    f"TemporalArc.progression_signals.signal_id "
                    f"duplicate: {signal.signal_id!r}"
                )
            signal_ids.add(signal.signal_id)


# ---------------------------------------------------------------------------
# Activation conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextMatchSignal:
    """One typed signal that contributes to ``context_match``.

    Packet 1.0 ships an empty placeholder shape.
    """

    signal_id: str
    measurable_via: BehaviorProtocolSignalSource
    weight: float = 1.0
    description: str = ""


@dataclass(frozen=True)
class ActivationConditions:
    """When this protocol matches and how it co-exists with others."""

    context_match_signals: tuple[ContextMatchSignal, ...] = ()
    co_activation_compatible: tuple[str, ...] = ()
    co_activation_incompatible: tuple[str, ...] = ()
    minimum_weight_floor: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_weight_floor <= 1.0:
            raise ValueError(
                f"ActivationConditions.minimum_weight_floor must be in "
                f"[0, 1], got {self.minimum_weight_floor!r}"
            )
        overlap = set(self.co_activation_compatible) & set(
            self.co_activation_incompatible
        )
        if overlap:
            raise ValueError(
                f"ActivationConditions: protocol ids cannot be both "
                f"compatible AND incompatible: {sorted(overlap)!r}"
            )


# ---------------------------------------------------------------------------
# PE signal definitions (R-PE entry; required unless legacy_fixture)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuccessSignal:
    """Protocol-declared 'this is what working looks like' readout."""

    signal_id: str
    description: str
    measurable_via: BehaviorProtocolSignalSource
    expected_value_range: tuple[float, float] = (0.0, 1.0)
    weight_in_pe: float = 1.0

    def __post_init__(self) -> None:
        if not self.signal_id.strip():
            raise ValueError("SuccessSignal.signal_id must be non-empty")
        low, high = self.expected_value_range
        if low > high:
            raise ValueError(
                f"SuccessSignal.expected_value_range must satisfy low<=high, "
                f"got {self.expected_value_range!r}"
            )
        if self.weight_in_pe < 0.0:
            raise ValueError(
                f"SuccessSignal.weight_in_pe must be >= 0, "
                f"got {self.weight_in_pe!r}"
            )


@dataclass(frozen=True)
class FailureSignal:
    """Protocol-declared 'this is what failing looks like' readout."""

    signal_id: str
    description: str
    measurable_via: BehaviorProtocolSignalSource
    threshold: float = 0.0
    weight_in_pe: float = 1.0

    def __post_init__(self) -> None:
        if not self.signal_id.strip():
            raise ValueError("FailureSignal.signal_id must be non-empty")
        if self.weight_in_pe < 0.0:
            raise ValueError(
                f"FailureSignal.weight_in_pe must be >= 0, "
                f"got {self.weight_in_pe!r}"
            )


# ---------------------------------------------------------------------------
# Signature cases (compile to vz-application CaseMemoryRecord; packet 1.4b)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignatureCase:
    """Reviewed (situation, action, outcome) episode carried by a protocol.

    Compiled (packet 1.4b+) into the consumer-side ``case_memory``
    store via ``compile_protocol_to_application_artifacts`` →
    ``CaseMemoryRecord`` (in ``vz-application``).

    Schema relationship to ``CaseMemoryRecord``:

    * 1:1 lossless review-time fields: ``case_id`` (→ namespaced
      ``case_id``), ``domain``, ``problem_pattern``,
      ``user_state_pattern``, ``risk_markers``, ``track_tags``,
      ``regime_tags``, ``intervention_ordering``, ``outcome_label``,
      ``escalation_observed``, ``repair_observed``, ``confidence``,
      ``relevance_score``, ``description``.
    * Derived / compiler-side: ``delayed_signal_count`` and
      ``reconstruction_source`` are protocol-level metadata
      (FixtureUptake sets these to mirror the vertical's
      hardcoded values).
    * NOT carried at the protocol level (CaseMemoryRecord defaults):
      ``continuum_*`` (continuum band positioning),
      ``lifecycle`` / ``ttl_seconds`` / ``expires_at_tick`` /
      ``provisional_origin`` (runtime-state-y; protocol seeds always
      land as ``CaseLifecycle.VALIDATED``). If a future protocol
      really needs to seed continuum-band cases, extend in a
      separate packet — current protocols don't.
    """

    case_id: str
    domain: str
    problem_pattern: str
    user_state_pattern: str
    risk_markers: tuple[str, ...]
    track_tags: tuple[str, ...]
    regime_tags: tuple[str, ...]
    intervention_ordering: tuple[str, ...]
    outcome_label: str
    confidence: float
    description: str
    relevance_score: float = 0.75
    escalation_observed: bool = False
    repair_observed: bool = False
    delayed_signal_count: int = 0
    reconstruction_source: str = "behavior-protocol"

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("SignatureCase.case_id must be non-empty")
        if not self.domain.strip():
            raise ValueError("SignatureCase.domain must be non-empty")
        if not self.intervention_ordering:
            # Mirrors the application-side validator: a case without an
            # intervention ordering has no semantic content for retrieval
            # / planner consumption.
            raise ValueError(
                f"SignatureCase.intervention_ordering must be non-empty "
                f"(case_id={self.case_id!r})"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"SignatureCase.confidence must be in [0, 1], "
                f"got {self.confidence!r}"
            )
        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError(
                f"SignatureCase.relevance_score must be in [0, 1], "
                f"got {self.relevance_score!r}"
            )
        if self.delayed_signal_count < 0:
            raise ValueError(
                f"SignatureCase.delayed_signal_count must be >= 0, "
                f"got {self.delayed_signal_count!r}"
            )


# ---------------------------------------------------------------------------
# Knowledge seeds (compile to vz-application DomainKnowledgeRecord; packet 1.4a)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgeSeed:
    """Reviewed knowledge / value / persona statement carried by a protocol.

    Compiled (packet 1.4a+) into the consumer-side
    ``domain_knowledge`` store via
    ``compile_protocol_to_application_artifacts`` →
    ``DomainKnowledgeRecord`` (in ``vz-application``).

    Schema relationship to ``DomainKnowledgeRecord``:

    * 1:1 lossless fields: ``domain``, ``title``, ``summary``,
      ``snippet``, ``confidence``, ``evidence_strength``,
      ``topic_tags``, ``source_type``, ``freshness_label``,
      ``jurisdiction_tags``, ``conflict_markers``.
    * Renames at compile time: ``seed_id`` → namespaced
      ``record_id`` (``protocol:{protocol_id}:knowledge:{seed_id}``);
      ``evidence_locator`` → ``locator`` (same role, vertical name).
    * ``url``: derived from ``BehaviorProtocol.source_locator`` at
      compile time (matches vertical's
      ``DomainKnowledgeRecord.url = profile.source_uri`` choice).
      ``KnowledgeSeed`` itself does NOT carry url; if a per-seed url
      is needed, extend later (no current need).

    Defaults align with ``GrowthAdvisorKnowledgeSeed`` so the
    fixture conversion is lossless.
    """

    seed_id: str
    domain: str
    title: str
    summary: str
    snippet: str
    evidence_locator: str
    confidence: float
    evidence_strength: str = "medium"
    topic_tags: tuple[str, ...] = ()
    source_type: str = "internal-guide"
    freshness_label: str = "reviewed"
    jurisdiction_tags: tuple[str, ...] = ()
    conflict_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.seed_id.strip():
            raise ValueError("KnowledgeSeed.seed_id must be non-empty")
        if not self.domain.strip():
            raise ValueError("KnowledgeSeed.domain must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"KnowledgeSeed.confidence must be in [0, 1], "
                f"got {self.confidence!r}"
            )


# ---------------------------------------------------------------------------
# Protocol revision log
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtocolRevision:
    """Append-only record of one mutation to the protocol."""

    revision_id: str
    revised_at_tick: int
    revised_by: str
    description: str
    affected_field: str

    def __post_init__(self) -> None:
        if not self.revision_id.strip():
            raise ValueError("ProtocolRevision.revision_id must be non-empty")


# ---------------------------------------------------------------------------
# Top-level: BehaviorProtocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BehaviorProtocol:
    """Reviewed, hot-loadable task configuration.

    See ``docs/specs/protocol-runtime.md`` §BehaviorProtocol Schema.

    Lifecycle: produced by a TaskUptake adapter (per-vertical
    FixtureUptake in packet 1.0; lifeform-protocol-runtime
    DocumentUptake in packet 1.1+); registered into
    ``ProtocolRegistryModule`` via ``load_protocol``; published into
    ``ActiveMixtureSnapshot`` each turn (SHADOW only in packet 1.0).

    PE-signal invariant: every protocol must declare at least one
    success and one failure signal so reflection / drives can learn
    from it. Pure legacy fixtures with no drives may opt out via
    ``legacy_fixture=True`` (packet 1.0 fixture conversion uses
    drive-synthesis so this opt-out is rarely needed).
    """

    protocol_id: str
    version: str
    advisor_name: str
    description: str
    source_kind: ProtocolSourceKind
    source_locator: str

    identity_assertion: IdentityAssertion
    boundary_contracts: tuple[BoundaryContract, ...]
    activation_conditions: ActivationConditions
    strategy_priors: tuple[StrategyPrior, ...]
    temporal_arc: TemporalArc
    success_signals: tuple[SuccessSignal, ...]
    failure_signals: tuple[FailureSignal, ...]

    knowledge_seeds: tuple[KnowledgeSeed, ...] = ()
    signature_cases: tuple[SignatureCase, ...] = ()
    parent_protocol_id: str | None = None
    review_status: ReviewStatus = ReviewStatus.DRAFT
    revision_log: tuple[ProtocolRevision, ...] = ()
    legacy_fixture: bool = False

    def __post_init__(self) -> None:
        if not self.protocol_id.strip():
            raise ValueError("BehaviorProtocol.protocol_id must be non-empty")
        if not self.version.strip():
            raise ValueError("BehaviorProtocol.version must be non-empty")
        if not self.advisor_name.strip():
            raise ValueError("BehaviorProtocol.advisor_name must be non-empty")
        if not self.source_locator.strip():
            raise ValueError(
                "BehaviorProtocol.source_locator must be non-empty"
            )
        _check_unique(
            "boundary_contracts.boundary_id",
            tuple(b.boundary_id for b in self.boundary_contracts),
        )
        _check_unique(
            "strategy_priors.rule_id",
            tuple(s.rule_id for s in self.strategy_priors),
        )
        _check_unique(
            "success_signals.signal_id",
            tuple(s.signal_id for s in self.success_signals),
        )
        _check_unique(
            "failure_signals.signal_id",
            tuple(s.signal_id for s in self.failure_signals),
        )
        _check_unique(
            "knowledge_seeds.seed_id",
            tuple(s.seed_id for s in self.knowledge_seeds),
        )
        _check_unique(
            "signature_cases.case_id",
            tuple(c.case_id for c in self.signature_cases),
        )
        _check_unique(
            "revision_log.revision_id",
            tuple(r.revision_id for r in self.revision_log),
        )
        if not self.legacy_fixture:
            if not self.success_signals:
                raise ValueError(
                    "BehaviorProtocol.success_signals must be non-empty "
                    "(set legacy_fixture=True to opt out for pure-legacy "
                    "fixtures with no drives / no PE entry)."
                )
            if not self.failure_signals:
                raise ValueError(
                    "BehaviorProtocol.failure_signals must be non-empty "
                    "(set legacy_fixture=True to opt out for pure-legacy "
                    "fixtures with no drives / no PE entry)."
                )


# ---------------------------------------------------------------------------
# Active mixture snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivationReason:
    """Per-protocol breakdown of why this activation_weight was chosen."""

    kind: ActivationReasonKind
    contribution: float
    detail: str = ""


@dataclass(frozen=True)
class ActiveProtocolEntry:
    """One row of the ``ActiveMixtureSnapshot.active_protocols`` tuple."""

    protocol_id: str
    activation_weight: float
    current_phase_id: str | None = None
    activation_reasons: tuple[ActivationReason, ...] = ()

    def __post_init__(self) -> None:
        if not self.protocol_id.strip():
            raise ValueError(
                "ActiveProtocolEntry.protocol_id must be non-empty"
            )
        if not 0.0 <= self.activation_weight <= 1.0:
            raise ValueError(
                f"ActiveProtocolEntry.activation_weight must be in [0, 1], "
                f"got {self.activation_weight!r}"
            )


@dataclass(frozen=True)
class ActiveMixtureSnapshot:
    """Per-turn published value of the ``active_mixture`` slot.

    Owner: ``volvence_zero.protocol_runtime.ProtocolRegistryModule``
    in ``vz-cognition``. Default wiring level: SHADOW (packet 1.0 —
    no consumer reads this slot yet).

    Content vs config boundary (packet 1.2+ Choice A — locked in):

    The snapshot publishes **IDs and weights only**, not boundary /
    strategy / case content bodies. Canonical content lives in the
    existing application owners (``boundary_policy`` /
    ``strategy_playbook`` / ``case_memory`` / ``domain_knowledge``)
    and is populated by the protocol load-time compile path
    (see ``volvence_zero.protocol_runtime.compiler``). Consumers
    treat ``active_mixture`` as a *configuration / weighting layer*
    and read execution content from those existing owners. This
    pins the R8 single-owner invariant: ProtocolRuntime is **not**
    a second owner of boundary / strategy / case content.

    Consumers (when wired in later packets):
    - ``boundary_policy``: reads its own ``ApplicationRareHeavyState
      .boundary_prior_hints`` (where the protocol compile path
      pushed entries) — does NOT read ``active_mixture`` directly
      for boundary content
    - ``metacontroller`` (vz-temporal): reads
      ``active_protocols[*].activation_weight`` → biases z_t
      selection alongside strategy_priors content from
      ``strategy_playbook``
    - ``strategy_playbook``: receives compiled ``PlaybookRule``
      entries (packet 1.3+); reads ``active_protocols[*]
      .activation_weight`` → weights retrieval mix
    - ``vitals``: reads ``active_protocols[*].current_phase_id`` →
      adjusts drive expected_band per phase
    """

    active_protocols: tuple[ActiveProtocolEntry, ...]
    boundary_union_ids: tuple[str, ...]
    identity_gate_traits: tuple[str, ...] = ()
    revision_fingerprint: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        _check_unique(
            "active_protocols.protocol_id",
            tuple(a.protocol_id for a in self.active_protocols),
        )
        _check_unique(
            "boundary_union_ids",
            self.boundary_union_ids,
        )


# ---------------------------------------------------------------------------
# Protocol revision proposals (packet 3.0)
# ---------------------------------------------------------------------------


class ProtocolRevisionTargetField(str, Enum):
    """Which family of protocol content a revision proposal touches."""

    STRATEGY_PRIOR = "strategy_prior"
    KNOWLEDGE_SEED = "knowledge_seed"
    SIGNATURE_CASE = "signature_case"
    BOUNDARY_CONTRACT = "boundary_contract"
    IDENTITY_ASSERTION = "identity_assertion"


class ProtocolRevisionChangeKind(str, Enum):
    """What kind of mutation the proposal is asking for."""

    WEIGHT_DECAY = "weight_decay"
    DEACTIVATE = "deactivate"
    REPLACE_TEXT = "replace_text"
    ARCHIVE = "archive"
    # Packet 5.2: reflection can propose adding a new strategy_prior
    # (or, future, a new knowledge_seed / signature_case) when
    # accumulated experience reveals a successful pattern not
    # covered by any existing protocol entry.
    ADD_STRATEGY = "add_strategy"
    # Packet 6.1: symmetric to WEIGHT_DECAY — multiplier > 1.
    WEIGHT_REINFORCE = "weight_reinforce"
    # Packet 6.1: change a BoundaryContract's trigger_reasons /
    # blocked_topics; review_level=L4 (admin sign-off required).
    BOUNDARY_REFINEMENT = "boundary_refinement"
    # Packet 6.1: edit identity_assertion (requires_self_traits /
    # forbidden_self_traits / required_regime_compatibility);
    # review_level=L4 fail-safe (ALWAYS queued).
    IDENTITY_CLARIFICATION = "identity_clarification"
    # Packet 6.1: mark protocol.review_status = RETIRED. Requires
    # packet 6.4 RETIRED state. review_level=L3-L4.
    PROTOCOL_RETIREMENT = "protocol_retirement"


@dataclass(frozen=True)
class ProposalEvidence:
    """Background-slow evidence accompanying a ProtocolRevisionProposal.

    Captures *why* the reflection layer wants this change so the
    R10 ModificationGate can audit and (optionally) automate the
    decision. All fields are owner-supplied summaries — the
    proposal does not carry raw history.
    """

    observation_window_turns: int
    pe_signature: str
    summary: str

    def __post_init__(self) -> None:
        if self.observation_window_turns < 1:
            raise ValueError(
                "ProposalEvidence.observation_window_turns must be >= 1; "
                f"got {self.observation_window_turns!r}"
            )
        if not self.summary.strip():
            raise ValueError(
                "ProposalEvidence.summary must be non-empty"
            )


@dataclass(frozen=True)
class ProtocolRegistryEntry:
    """Per-protocol summary published in
    :class:`ProtocolRegistrySnapshot`. Surface for audit / CLI tools.

    Packet 9.1 adds ``knowledge_lineage_ids`` and ``case_lineage_ids``
    — the fully-qualified ``protocol:{protocol_id}:knowledge:{seed_id}``
    / ``protocol:{protocol_id}:case:{case_id}`` strings that the
    compile path produces. Reflection rules (e.g. archival) need
    these to identify which retrieval-store record_ids belong to
    which protocol without re-running the compile pipeline.
    """

    protocol_id: str
    version: str
    advisor_name: str
    review_status: "ReviewStatus"
    parent_protocol_id: str | None
    boundary_count: int
    strategy_count: int
    knowledge_seed_count: int
    signature_case_count: int
    revision_count: int
    knowledge_lineage_ids: tuple[str, ...] = ()
    case_lineage_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProtocolRegistrySnapshot:
    """Per-turn published value of the ``protocol_registry`` slot
    (packet 6.8). Lets external auditors / CLI tools see the
    contents of the registry without poking at owner internals."""

    entries: tuple[ProtocolRegistryEntry, ...]
    active_count: int
    retired_count: int
    description: str = ""


@dataclass(frozen=True)
class ProtocolRevisionLogEntry:
    """Per-revision summary for the ``protocol_revision_log`` slot.

    Field names mirror :class:`ProtocolRevision` (audit transparency)."""

    protocol_id: str
    revision_id: str
    revised_at_tick: int
    revised_by: str
    description: str
    affected_field: str


@dataclass(frozen=True)
class ProtocolRevisionLogSnapshot:
    """Per-turn published view of all revisions across all protocols."""

    entries: tuple[ProtocolRevisionLogEntry, ...]
    description: str = ""


@dataclass(frozen=True)
class ProtocolRevisionQueueEntry:
    """One row in :class:`ProtocolRevisionQueueSnapshot`.

    ``outcome`` mirrors :class:`ApprovalOutcome` value — kept as
    a string here so vz-contracts has no dependency on
    vz-application.protocol_runtime.revision_queue.
    """

    proposal_id: str
    target_protocol_id: str
    change_kind: str
    outcome: str
    rationale: str


@dataclass(frozen=True)
class ProtocolRevisionQueueSnapshot:
    """Per-turn published view of the routing of new revision
    proposals (packet 9.0). Each turn the
    :class:`ProtocolRevisionQueueModule` consumes
    ``protocol_reflection`` upstream, runs each new proposal
    through the ModificationGate, submits to a shared
    ``RevisionQueue``, and optionally auto-applies AUTO_APPROVED
    ones via the registry.

    ``newly_routed`` lists the proposals processed *this turn*
    only. ``pending_count`` is the queue total (auto + manual).
    ``auto_applied_count`` is how many of ``newly_routed`` were
    AUTO_APPROVED *and* applied this turn (the closed PE→learning
    loop's heartbeat).
    """

    newly_routed: tuple[ProtocolRevisionQueueEntry, ...]
    pending_count: int
    auto_applied_count: int
    description: str = ""


@dataclass(frozen=True)
class ProtocolPhaseSnapshot:
    """Per-turn published value of the ``protocol_phase`` slot.

    Owner: ``vz-application.protocol_runtime.phase_engine.ProtocolPhaseModule``.
    The owner evaluates each loaded protocol's
    :class:`TemporalArc.progression_signals` against typed
    upstream snapshots (``prediction_error`` / ``interlocutor_state``
    / ``regime`` / ``rupture_state``) and advances or retreats
    the phase pointer accordingly. Protocols with empty
    ``progression_signals`` (e.g. cheng_laoshi default fixture)
    pin at the first phase forever — preserves backwards-compat.

    ``ProtocolRegistryModule`` consumes this snapshot to populate
    :attr:`ActiveProtocolEntry.current_phase_id`. Replaces the
    static ``_default_phase_id`` placeholder used by packet 1.0.

    Pinned R3 invariant (β_t learned, not calendar-tagged):
    phase advances only fire from typed signal evidence, never
    from "session_day == 3" string tags.
    """

    phase_by_protocol_id: tuple[tuple[str, str], ...]
    turns_in_current_phase: tuple[tuple[str, int], ...]
    description: str = ""

    def __post_init__(self) -> None:
        ids = tuple(pid for pid, _ in self.phase_by_protocol_id)
        _check_unique("phase_by_protocol_id.protocol_id", ids)
        turn_ids = tuple(pid for pid, _ in self.turns_in_current_phase)
        _check_unique("turns_in_current_phase.protocol_id", turn_ids)


@dataclass(frozen=True)
class ProtocolReflectionSnapshot:
    """Per-turn published value of the ``protocol_reflection`` slot.

    Owner: ``vz-cognition.reflection.engine.ProtocolReflectionEngine``
    (background-slow timescale; runs reflection rules every N turns
    and otherwise re-publishes the previous snapshot).

    Why a separate slot from ``reflection``: the existing
    :class:`ReflectionModule` owns memory / policy consolidation
    snapshots and has a different cadence + writeback pipeline.
    Protocol-content reflection is its own R8 owner to keep the
    SSOT clean.

    Empty proposals tuple is the default ("no proposals this
    cycle"); consumers must NOT treat empty as a failure mode.
    """

    protocol_revision_proposals: tuple["ProtocolRevisionProposal", ...]
    observation_window_turns: int
    turns_since_last_scan: int
    description: str = ""

    def __post_init__(self) -> None:
        if self.observation_window_turns < 0:
            raise ValueError(
                "ProtocolReflectionSnapshot.observation_window_turns "
                f"must be >= 0; got {self.observation_window_turns!r}"
            )
        if self.turns_since_last_scan < 0:
            raise ValueError(
                "ProtocolReflectionSnapshot.turns_since_last_scan "
                f"must be >= 0; got {self.turns_since_last_scan!r}"
            )
        ids = tuple(p.proposal_id for p in self.protocol_revision_proposals)
        _check_unique("protocol_revision_proposals.proposal_id", ids)


@dataclass(frozen=True)
class ProtocolRevisionProposal:
    """A reflection-driven proposal to mutate one protocol entry.

    Lifecycle:

    * ReflectionEngine emits the proposal in
      ``ReflectionSnapshot.protocol_revision_proposals``.
    * R10 ModificationGate evaluates it (auto-approve at L1/L2,
      gated at L3/L4).
    * Approved proposals are applied by
      ``ProtocolRegistry.apply_revision`` — this produces a
      new ``BehaviorProtocol`` with the change recorded in
      ``revision_log`` (R15 rollback path).

    Note: the ``proposed_payload`` is intentionally a plain
    dict so the schema can carry diverse change kinds. The
    apply layer interprets the payload per ``change_kind``;
    bad payloads fail the apply step (not the proposal step).
    """

    proposal_id: str
    target_protocol_id: str
    target_field: ProtocolRevisionTargetField
    target_entry_id: str
    change_kind: ProtocolRevisionChangeKind
    evidence: ProposalEvidence
    proposed_payload: dict | None = None
    required_review_level: ReviewLevel = ReviewLevel.L3

    def __post_init__(self) -> None:
        if not self.proposal_id.strip():
            raise ValueError(
                "ProtocolRevisionProposal.proposal_id must be non-empty"
            )
        if not self.target_protocol_id.strip():
            raise ValueError(
                "ProtocolRevisionProposal.target_protocol_id must be non-empty"
            )
        if not self.target_entry_id.strip():
            raise ValueError(
                "ProtocolRevisionProposal.target_entry_id must be non-empty"
            )


# ---------------------------------------------------------------------------
# DocumentUptake: candidate protocol + provenance (packet 2.0)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtocolProvenance:
    """Audit trail for a candidate protocol's origin.

    Carries enough information for ``R10 ModificationGate`` review
    + post-hoc audit: where did this protocol come from, what
    extracted it, when, and how confident is the extractor in the
    structured output.
    """

    source_kind: ProtocolSourceKind
    source_locator: str
    extracted_at_iso: str
    extractor_id: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.source_locator.strip():
            raise ValueError(
                "ProtocolProvenance.source_locator must be non-empty"
            )
        if not self.extractor_id.strip():
            raise ValueError(
                "ProtocolProvenance.extractor_id must be non-empty"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "ProtocolProvenance.confidence must be in [0, 1], "
                f"got {self.confidence!r}"
            )


@dataclass(frozen=True)
class BehaviorProtocolCandidate:
    """A protocol freshly extracted from a non-fixture source.

    Wraps a ``BehaviorProtocol`` plus the provenance + a
    ``requires_review`` flag that the load path consults
    (``ProtocolRegistryModule.load_protocol_candidate``):

    * ``requires_review=True`` (default for any LLM/document-derived
      candidate) → loaded into ``review_status=SHADOW`` and
      flagged for human / automated review by R10 ModificationGate.
      Cannot be promoted to ACTIVE without explicit approval.
    * ``requires_review=False`` is reserved for already-reviewed
      candidates (e.g. APIInjection from a trusted upstream
      that itself ran review). Setting it manually for an
      LLM-extracted candidate is a contract violation that the
      review CLI / approval helpers MUST catch.

    Frozen by construction; reviewers produce a *new* approved
    ``BehaviorProtocol`` via the approval helper rather than
    mutating the candidate in place (R15 audit trail).
    """

    protocol: "BehaviorProtocol"
    provenance: ProtocolProvenance
    requires_review: bool = True
    review_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Sanity: provenance.source_kind must agree with the inner
        # protocol's source_kind. If the candidate was extracted
        # from a PDF, both fields must say PDF_UPTAKE; this
        # prevents silent provenance drift between the wrapper
        # and the inner field.
        if self.protocol.source_kind is not self.provenance.source_kind:
            raise ValueError(
                "BehaviorProtocolCandidate.provenance.source_kind "
                f"({self.provenance.source_kind!r}) must match the "
                "inner BehaviorProtocol.source_kind "
                f"({self.protocol.source_kind!r})"
            )
        if self.protocol.source_locator != self.provenance.source_locator:
            raise ValueError(
                "BehaviorProtocolCandidate.provenance.source_locator "
                f"({self.provenance.source_locator!r}) must match the "
                "inner BehaviorProtocol.source_locator "
                f"({self.protocol.source_locator!r})"
            )


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


def _check_unique(field_name: str, values: tuple[str, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError(
            f"{field_name} values must be unique, got {values!r}"
        )


__all__ = [
    "ActiveMixtureSnapshot",
    "ActiveProtocolEntry",
    "ActivationConditions",
    "ActivationReason",
    "ActivationReasonKind",
    "BehaviorProtocol",
    "BehaviorProtocolCandidate",
    "BehaviorProtocolSignalSource",
    "BoundaryContract",
    "BoundarySeverity",
    "ContextMatchSignal",
    "DriveExpectation",
    "FailureSignal",
    "IdentityAssertion",
    "KnowledgeSeed",
    "ProgressionSignal",
    "ProposalEvidence",
    "ProtocolPhaseSnapshot",
    "ProtocolProvenance",
    "ProtocolReflectionSnapshot",
    "ProtocolRegistryEntry",
    "ProtocolRegistrySnapshot",
    "ProtocolRevisionLogEntry",
    "ProtocolRevisionLogSnapshot",
    "ProtocolRevisionQueueEntry",
    "ProtocolRevisionQueueSnapshot",
    "ProtocolRevision",
    "ProtocolRevisionChangeKind",
    "ProtocolRevisionProposal",
    "ProtocolRevisionTargetField",
    "ProtocolSourceKind",
    "ReviewLevel",
    "ReviewStatus",
    "SignatureCase",
    "StrategyPrior",
    "StrategyPriorRevision",
    "SuccessSignal",
    "TemporalArc",
    "TemporalPhase",
]
