"""BehaviorProtocol → application owners compile path (packet 1.2 / 1.3b / 1.4a / 1.4b).

This module implements the core R8 SSOT discipline declared in
``docs/specs/protocol-runtime.md`` §ProtocolRuntime 与 application
owners 的内容边界:

    ProtocolRuntime is NOT the canonical owner of boundary /
    strategy / case / domain knowledge. ``BehaviorProtocol`` carries
    those as protocol-level *declarations*; at load time, the
    declarations are compiled into the existing application owners
    (boundary_policy / strategy_playbook / case_memory /
    domain_knowledge), which remain the single writers of
    canonical content.

Compile path coverage (one consumer per packet, mechanical mirror):

* Packet 1.2: ``BoundaryContract`` → ``BoundaryPriorHint`` →
  ``ApplicationRareHeavyState.upsert_boundary_prior_hints``
* Packet 1.3b: ``StrategyPrior`` → ``PlaybookRule`` →
  ``ApplicationRareHeavyState.upsert_distilled_playbook_rules``
* Packet 1.4a: ``KnowledgeSeed`` → ``DomainKnowledgeRecord`` →
  ``ApplicationDomainKnowledgeStore.upsert_records``
* Packet 1.4b: ``SignatureCase`` → ``CaseMemoryRecord`` →
  ``ApplicationCaseMemoryStore.upsert_records``
* Packet 1.4'+: ``temporal_arc.phases`` → PE-driven progression
  signals (no direct application owner; feeds metacontroller).

Why a *new* compile module instead of folding into the existing
``volvence_zero.application.domain_experience.compile_domain_experience_package``:

* Provenance: the compile output must be traceable back to the
  source ``protocol_id`` so a future unload path can identify
  protocol-driven entries vs reviewed-vertical-driven ones.
  Encoding lineage in ``hint_id`` namespace keeps the existing
  ``BoundaryPriorHint`` schema unchanged.
* Source type: the existing function consumes
  ``DomainExperiencePackage`` (with embedded
  ``BoundaryPriorHint`` records). The protocol path consumes
  ``BehaviorProtocol`` (with ``BoundaryContract`` records that
  must be field-mapped). Mixing the two in one function would
  blur ownership.
* Audit surface: protocol load is gated by
  ``ModificationGate`` (R10) at higher review levels than
  vertical seeding; keeping the compile separate makes that
  gate easy to wire later.

What this module is NOT:

* It is not a runtime owner. The output is a frozen artifact
  (``ProtocolApplicationArtifacts``) consumed by the owner side
  of the application stores. Application owners remain the
  R8 single writers of their own state.
* It does not call upsert directly. The owner module
  (``ProtocolRegistryModule.load_protocol``) decides whether to
  apply (depends on whether a rare-heavy state was injected).
* Unload semantics are deferred (see owner ``unload_protocol``).
"""

from __future__ import annotations

from dataclasses import replace as _replace
from typing import Callable

from dataclasses import dataclass

from volvence_zero.application.storage import (
    CaseMemoryRecord,
    DomainKnowledgeRecord,
)
from volvence_zero.application.types import BoundaryPriorHint, PlaybookRule
from volvence_zero.behavior_protocol import (
    BehaviorProtocol,
    BoundaryContract,
    KnowledgeSeed,
    SignatureCase,
    StrategyPrior,
)


_HINT_ID_PROTOCOL_NAMESPACE = "protocol"


@dataclass(frozen=True)
class ProtocolApplicationArtifacts:
    """Compiled artifacts ready to be applied into application owners.

    Returned by :func:`compile_protocol_to_application_artifacts`.

    Packet 1.2 populated ``boundary_prior_hints``; packet 1.3b adds
    ``playbook_rules``. Future packets will add fields for compiled
    ``CaseMemoryRecord`` / ``DomainKnowledgeRecord`` entries — each
    behind its own packet so consumer changes are isolated.

    The artifact carries the ``protocol_id`` so callers can
    associate the application-owner side-effects with the protocol
    that produced them (audit / future unload).
    """

    protocol_id: str
    boundary_prior_hints: tuple[BoundaryPriorHint, ...]
    playbook_rules: tuple[PlaybookRule, ...] = ()
    domain_knowledge_records: tuple[DomainKnowledgeRecord, ...] = ()
    case_memory_records: tuple[CaseMemoryRecord, ...] = ()


def merge_protocol_chain(
    child: BehaviorProtocol,
    *,
    lookup: "Callable[[str], BehaviorProtocol | None]",
) -> BehaviorProtocol:
    """Packet 6.5: walk parent_protocol_id chain and merge content.

    Merge semantics (Open Q1 resolve):

    * ``boundary_contracts`` / ``strategy_priors`` / ``knowledge_seeds``
      / ``signature_cases``: union by id; child entry overrides parent
      with the same id.
    * ``identity_assertion``: child overrides (treated as opaque).
    * ``activation_conditions``: child overrides.
    * ``temporal_arc``: child overrides (phase content is structural).
    * ``success_signals`` / ``failure_signals``: union by signal_id;
      child overrides.
    * ``revision_log``: per-protocol per-lifeform — NOT merged
      across chain. Child's own revision_log returned unchanged.
    * Top-level metadata (protocol_id / version / advisor_name /
      description / source_*): child wins.

    Cycle protection: walks chain via ``lookup`` callable; if a
    chain visits the same id twice, raises ValueError. Missing
    parent (``lookup`` returns None) raises ValueError.

    Returns a *new* ``BehaviorProtocol`` instance with the merged
    content. Caller is expected to compile this merged protocol
    (or load it as the canonical compiled form).
    """

    visited: set[str] = set()
    chain: list[BehaviorProtocol] = []
    cursor: BehaviorProtocol | None = child
    while cursor is not None:
        if cursor.protocol_id in visited:
            raise ValueError(
                f"protocol chain has a cycle at {cursor.protocol_id!r}"
            )
        visited.add(cursor.protocol_id)
        chain.append(cursor)
        if cursor.parent_protocol_id is None:
            break
        parent = lookup(cursor.parent_protocol_id)
        if parent is None:
            raise ValueError(
                f"protocol {cursor.protocol_id!r} declares "
                f"parent_protocol_id={cursor.parent_protocol_id!r} "
                "but parent is not loaded in the registry"
            )
        cursor = parent

    # chain is [child, parent, grandparent, ...] — merge from
    # oldest to newest so child wins on conflicts.
    chain_oldest_first = list(reversed(chain))

    def _merge_by_id(
        existing: tuple, incoming: tuple, key: str
    ) -> tuple:
        bucket: dict = {}
        for entry in existing:
            bucket[getattr(entry, key)] = entry
        for entry in incoming:
            bucket[getattr(entry, key)] = entry
        return tuple(bucket.values())

    base = chain_oldest_first[0]
    boundary_contracts = base.boundary_contracts
    strategy_priors = base.strategy_priors
    knowledge_seeds = base.knowledge_seeds
    signature_cases = base.signature_cases
    success_signals = base.success_signals
    failure_signals = base.failure_signals

    for protocol in chain_oldest_first[1:]:
        boundary_contracts = _merge_by_id(
            boundary_contracts, protocol.boundary_contracts, "boundary_id"
        )
        strategy_priors = _merge_by_id(
            strategy_priors, protocol.strategy_priors, "rule_id"
        )
        knowledge_seeds = _merge_by_id(
            knowledge_seeds, protocol.knowledge_seeds, "seed_id"
        )
        signature_cases = _merge_by_id(
            signature_cases, protocol.signature_cases, "case_id"
        )
        success_signals = _merge_by_id(
            success_signals, protocol.success_signals, "signal_id"
        )
        failure_signals = _merge_by_id(
            failure_signals, protocol.failure_signals, "signal_id"
        )

    return _replace(
        child,
        boundary_contracts=boundary_contracts,
        strategy_priors=strategy_priors,
        knowledge_seeds=knowledge_seeds,
        signature_cases=signature_cases,
        success_signals=success_signals,
        failure_signals=failure_signals,
    )


def compile_protocol_to_application_artifacts(
    protocol: BehaviorProtocol,
) -> ProtocolApplicationArtifacts:
    """Compile a ``BehaviorProtocol`` into application-owner artifacts.

    Pure function: no I/O, no global state, no side effects on
    ``protocol``. Idempotent — same input ⇒ identical output (frozen
    tuple of frozen dataclasses).

    Packet 1.2 scope: ``BoundaryContract`` → ``BoundaryPriorHint``.
    Packet 1.3b scope: ``StrategyPrior`` → ``PlaybookRule``.
    Packet 1.4a scope: ``KnowledgeSeed`` → ``DomainKnowledgeRecord``.
    Packet 1.4b scope: ``SignatureCase`` → ``CaseMemoryRecord``.
    All four kinds compile in one pass; the function signature is
    stable for future additions.
    """

    if not isinstance(protocol, BehaviorProtocol):
        raise TypeError(
            f"compile_protocol_to_application_artifacts expects "
            f"BehaviorProtocol, got {type(protocol).__name__}"
        )

    boundary_hints = tuple(
        _boundary_contract_to_prior_hint(protocol.protocol_id, contract)
        for contract in protocol.boundary_contracts
    )
    playbook_rules = tuple(
        _strategy_prior_to_playbook_rule(protocol.protocol_id, prior)
        for prior in protocol.strategy_priors
    )
    domain_knowledge_records = tuple(
        _knowledge_seed_to_domain_record(protocol, seed)
        for seed in protocol.knowledge_seeds
    )
    case_memory_records = tuple(
        _signature_case_to_case_record(protocol.protocol_id, case)
        for case in protocol.signature_cases
    )
    return ProtocolApplicationArtifacts(
        protocol_id=protocol.protocol_id,
        boundary_prior_hints=boundary_hints,
        playbook_rules=playbook_rules,
        domain_knowledge_records=domain_knowledge_records,
        case_memory_records=case_memory_records,
    )


# ---------------------------------------------------------------------------
# BoundaryContract → BoundaryPriorHint
# ---------------------------------------------------------------------------


def _boundary_contract_to_prior_hint(
    protocol_id: str,
    contract: BoundaryContract,
) -> BoundaryPriorHint:
    """Map one ``BoundaryContract`` to one ``BoundaryPriorHint``.

    Lineage: the produced ``hint_id`` is namespaced with the
    ``protocol_id`` so application-owner audit can trace which
    protocol owns which entry. Format::

        protocol:{protocol_id}:boundary:{boundary_id}

    All BoundaryContract fields that ``BoundaryPriorHint`` carries
    pass through unchanged. ``severity`` and ``review_level`` are
    protocol-only metadata and stay out of the hint
    (``BoundaryPriorHint`` doesn't carry them today; if downstream
    boundary execution needs them, ``BoundaryPriorHint`` schema
    must be extended in vz-application separately and this mapping
    updated). ``BoundaryPriorHint`` validation in
    ``volvence_zero.application.domain_experience`` requires
    non-empty ``trigger_reasons``; a protocol that ships an empty
    trigger list produces a hint that the application validator
    will reject on ingest — fail-loud is intended.
    """

    return BoundaryPriorHint(
        hint_id=_protocol_hint_id(protocol_id, contract.boundary_id),
        regime_id=contract.regime_id,
        trigger_reasons=contract.trigger_reasons,
        answer_depth_limit_hint=contract.answer_depth_limit_hint,
        clarification_required=contract.clarification_required,
        refer_out_required=contract.refer_out_required,
        blocked_topics=contract.blocked_topics,
        required_disclaimers=contract.required_disclaimers,
        confidence=contract.confidence,
        description=contract.description,
    )


def _protocol_hint_id(protocol_id: str, boundary_id: str) -> str:
    """Stable namespaced hint id for a protocol-driven boundary entry.

    Public so tests can construct expected ids without re-encoding
    the format string. Lineage prefix lets a future unload path
    identify protocol-driven entries by id-prefix scan, since
    ``BoundaryPriorHint`` itself has no provenance field.
    """

    return (
        f"{_HINT_ID_PROTOCOL_NAMESPACE}:{protocol_id}:boundary:{boundary_id}"
    )


# ---------------------------------------------------------------------------
# StrategyPrior → PlaybookRule (packet 1.3b)
# ---------------------------------------------------------------------------


def _strategy_prior_to_playbook_rule(
    protocol_id: str,
    prior: StrategyPrior,
) -> PlaybookRule:
    """Map one ``StrategyPrior`` to one ``PlaybookRule``.

    Lineage: the produced ``rule_id`` is namespaced with the
    ``protocol_id`` so application-owner audit can trace which
    protocol owns which entry. Format::

        protocol:{protocol_id}:playbook:{rule_id}

    Field mapping (lossless for fields ``PlaybookRule`` carries;
    PE-revision metadata stays on the protocol side):

    * ``rule_id``                  → namespaced
    * ``problem_pattern``          → 1:1
    * ``recommended_regime``       → 1:1
    * ``recommended_ordering``     → 1:1
    * ``recommended_pacing``       → 1:1
    * ``avoid_patterns``           → 1:1
    * ``knowledge_weight_hint``    → 1:1
    * ``experience_weight_hint``   → 1:1
    * ``applicability_phase``      → ``applicability_scope``
      (rename only; same role; ``StrategyPrior`` historical name
      vs ``PlaybookRule`` schema name)
    * ``confidence``               → 1:1
    * ``description``              → 1:1
    * ``initial_weight`` / ``pe_decay_rate`` /
      ``pe_reinforce_rate`` / ``minimum_weight_floor`` /
      ``revision_history``        → **dropped**: PE-revision
      metadata is consumed by the future activation controller
      (packet 1.5+); ``StrategyPlaybookModule`` doesn't read it.
    * ``continuum_band_id`` / ``mean_continuum_position``
      → ``PlaybookRule`` defaults; protocols don't seed continuum
      fields today.

    Vertical compile parity: this function produces a
    ``PlaybookRule`` byte-equivalent to
    ``lifeform_domain_growth_advisor.compiler._playbook_rule``
    output, except for the ``rule_id`` lineage prefix. The
    matched-control test
    (``tests/contracts/test_protocol_strategy_matched_control.py``)
    pins this invariant.
    """

    return PlaybookRule(
        rule_id=_protocol_rule_id(protocol_id, prior.rule_id),
        problem_pattern=prior.problem_pattern,
        recommended_regime=prior.recommended_regime,
        recommended_ordering=prior.recommended_ordering,
        recommended_pacing=prior.recommended_pacing,
        avoid_patterns=prior.avoid_patterns,
        knowledge_weight_hint=prior.knowledge_weight_hint,
        experience_weight_hint=prior.experience_weight_hint,
        applicability_scope=prior.applicability_phase,
        confidence=prior.confidence,
        description=prior.description,
    )


def _protocol_rule_id(protocol_id: str, rule_id: str) -> str:
    """Stable namespaced rule id for a protocol-driven playbook entry.

    Public-by-convention (private API; tests reach through directly
    when needed). Format mirrors ``_protocol_hint_id``::

        protocol:{protocol_id}:playbook:{rule_id}

    Lineage prefix lets future unload identify protocol-driven
    entries by prefix; ``PlaybookRule`` has no provenance field
    just like ``BoundaryPriorHint``.
    """

    return (
        f"{_HINT_ID_PROTOCOL_NAMESPACE}:{protocol_id}:playbook:{rule_id}"
    )


# ---------------------------------------------------------------------------
# KnowledgeSeed → DomainKnowledgeRecord (packet 1.4a)
# ---------------------------------------------------------------------------


def _knowledge_seed_to_domain_record(
    protocol: BehaviorProtocol,
    seed: KnowledgeSeed,
) -> DomainKnowledgeRecord:
    """Map one ``KnowledgeSeed`` to one ``DomainKnowledgeRecord``.

    Lineage: the produced ``record_id`` is namespaced with the
    ``protocol_id`` so application-owner audit can trace which
    protocol owns which entry. Format::

        protocol:{protocol_id}:knowledge:{seed_id}

    Field mapping (lossless):

    * ``seed_id``               → namespaced ``record_id``
    * ``domain``                → 1:1
    * ``topic_tags``            → 1:1
    * ``jurisdiction_tags``     → 1:1 (default empty tuple if seed
      doesn't specify; vertical bookmarks ``private-domain-companion``
      via fixture uptake)
    * ``source_type``           → 1:1
    * ``title``                 → 1:1
    * ``evidence_locator``      → ``locator`` (rename only; mirrors
      ``GrowthAdvisorKnowledgeSeed.evidence_locator`` →
      ``DomainKnowledgeRecord.locator``)
    * ``summary``               → 1:1
    * ``snippet``               → 1:1
    * ``freshness_label``       → 1:1
    * ``confidence``            → 1:1
    * ``evidence_strength``     → 1:1
    * ``conflict_markers``      → 1:1
    * ``url``                   → derived from
      ``BehaviorProtocol.source_locator`` (mirrors vertical's
      ``DomainKnowledgeRecord.url = profile.source_uri`` choice).
      ``KnowledgeSeed`` itself does not carry url.

    Vertical compile parity: this function produces a
    ``DomainKnowledgeRecord`` byte-equivalent to
    ``lifeform_domain_growth_advisor.compiler._knowledge_record``
    output (modulo ``record_id`` lineage prefix). The
    matched-control test
    (``tests/contracts/test_protocol_knowledge_matched_control.py``)
    pins this invariant.
    """

    return DomainKnowledgeRecord(
        record_id=_protocol_record_id(protocol.protocol_id, seed.seed_id),
        domain=seed.domain,
        topic_tags=seed.topic_tags,
        jurisdiction_tags=seed.jurisdiction_tags,
        source_type=seed.source_type,
        title=seed.title,
        locator=seed.evidence_locator,
        summary=seed.summary,
        snippet=seed.snippet,
        freshness_label=seed.freshness_label,
        confidence=seed.confidence,
        evidence_strength=seed.evidence_strength,
        conflict_markers=seed.conflict_markers,
        url=protocol.source_locator,
    )


def _protocol_record_id(protocol_id: str, seed_id: str) -> str:
    """Stable namespaced record id for a protocol-driven knowledge entry.

    Format mirrors ``_protocol_hint_id`` / ``_protocol_rule_id``::

        protocol:{protocol_id}:knowledge:{seed_id}

    Lineage prefix lets future unload identify protocol-driven
    entries by prefix scan; ``DomainKnowledgeRecord`` itself has
    no provenance field beyond ``url``.
    """

    return (
        f"{_HINT_ID_PROTOCOL_NAMESPACE}:{protocol_id}:knowledge:{seed_id}"
    )


# ---------------------------------------------------------------------------
# SignatureCase → CaseMemoryRecord (packet 1.4b)
# ---------------------------------------------------------------------------


def _signature_case_to_case_record(
    protocol_id: str,
    case: SignatureCase,
) -> CaseMemoryRecord:
    """Map one ``SignatureCase`` to one ``CaseMemoryRecord``.

    Lineage: the produced ``case_id`` is namespaced with the
    ``protocol_id`` so application-owner audit can trace which
    protocol owns which entry. Format::

        protocol:{protocol_id}:case:{case_id}

    Field mapping (lossless on review-time fields):

    * ``case_id``                → namespaced
    * 1:1 fields: ``domain``, ``problem_pattern``,
      ``user_state_pattern``, ``risk_markers``, ``track_tags``,
      ``regime_tags``, ``intervention_ordering``, ``outcome_label``,
      ``delayed_signal_count``, ``escalation_observed``,
      ``repair_observed``, ``confidence``, ``relevance_score``,
      ``description``, ``reconstruction_source``
    * Defaults from ``CaseMemoryRecord`` (NOT carried at protocol
      level; protocol seeds always land as VALIDATED with no ttl):
      ``continuum_profile_id=None`` / ``continuum_band_id=None`` /
      ``continuum_position=0.0`` / ``continuum_update_frequency=0.0`` /
      ``lifecycle=CaseLifecycle.VALIDATED`` / ``ttl_seconds=None`` /
      ``expires_at_tick=None`` / ``provisional_origin=""``.

    Vertical compile parity: this function produces a
    ``CaseMemoryRecord`` byte-equivalent to
    ``lifeform_domain_growth_advisor.compiler._case_record``
    output (modulo ``case_id`` lineage prefix), provided the
    ``SignatureCase`` was constructed via FixtureUptake which sets
    ``delayed_signal_count=1`` and
    ``reconstruction_source="reviewed-growth-advisor-profile"`` to
    match the vertical's hardcoded values. Pinned by
    ``tests/contracts/test_protocol_case_matched_control.py``.
    """

    return CaseMemoryRecord(
        case_id=_protocol_case_id(protocol_id, case.case_id),
        domain=case.domain,
        problem_pattern=case.problem_pattern,
        user_state_pattern=case.user_state_pattern,
        risk_markers=case.risk_markers,
        track_tags=case.track_tags,
        regime_tags=case.regime_tags,
        intervention_ordering=case.intervention_ordering,
        outcome_label=case.outcome_label,
        delayed_signal_count=case.delayed_signal_count,
        escalation_observed=case.escalation_observed,
        repair_observed=case.repair_observed,
        confidence=case.confidence,
        relevance_score=case.relevance_score,
        description=case.description,
        reconstruction_source=case.reconstruction_source,
    )


def _protocol_case_id(protocol_id: str, case_id: str) -> str:
    """Stable namespaced case id for a protocol-driven case_memory entry.

    Format mirrors ``_protocol_hint_id`` / ``_protocol_rule_id`` /
    ``_protocol_record_id``::

        protocol:{protocol_id}:case:{case_id}

    Lineage prefix lets future unload identify protocol-driven
    entries by prefix scan; ``CaseMemoryRecord`` has no provenance
    field beyond ``reconstruction_source``.
    """

    return (
        f"{_HINT_ID_PROTOCOL_NAMESPACE}:{protocol_id}:case:{case_id}"
    )


__all__ = [
    "ProtocolApplicationArtifacts",
    "compile_protocol_to_application_artifacts",
]
