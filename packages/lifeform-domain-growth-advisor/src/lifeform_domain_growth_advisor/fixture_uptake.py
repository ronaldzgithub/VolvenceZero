"""FixtureUptake: ``GrowthAdvisorProfile`` → ``BehaviorProtocol`` (packet 1.0).

Per-vertical adapter that converts the reviewed ``GrowthAdvisorProfile``
fixture (``profiles/cheng_laoshi.py``) into a ``BehaviorProtocol``
suitable for loading into ``volvence_zero.protocol_runtime
.ProtocolRegistryModule``.

This adapter is *additive*: the existing
``DomainExperiencePackage`` / ``VitalsBootstrap`` /
``IngestionEnvelope`` compilation paths still produce their products.
Packet 1.0 wires the BehaviorProtocol path SHADOW alongside the
existing path; later packets will move ownership of strategy / boundary
weighting from the existing application owners to the
``ActiveMixtureSnapshot`` consumer flow.

PE-signal synthesis (spec §协议 → PE 映射 option ii):

* For each ``GrowthAdvisorDrivePrior`` D, derive:
    - one ``SuccessSignal`` keyed to D *holding* in
      ``homeostatic_band`` — source ``DRIVE_HOMEOSTASIS_HOLD``
    - one ``FailureSignal`` keyed to D *breaching* the band —
      source ``DRIVE_HOMEOSTASIS_BREACH``
* ``weight_in_pe`` passes through ``D.pe_weight`` so the protocol's
  PE readout is exactly the underlying vitals deviation, just
  re-keyed by signal_id for downstream consumption.
* 4 drives → 8 signals satisfies the spec's "≥1 success ∧ ≥1
  failure" invariant without ``legacy_fixture=True`` opt-out.

Things this adapter does NOT do (deferred):

* Synthesise PE-driven ``TemporalArc.progression_signals`` from the
  reviewed onboarding arc (icebreaker → baseline → empathy → pain
  mining → rapport → targeted advice → summary). The previous
  calendar-day routing tags (``growth_advisor:day{1..7}``) were
  removed on 2026-05-14 — relationship phase routing is now owned
  by ``BehaviorProtocol.TemporalArc.progression_signals``
  (PE-driven), to be wired in protocol-runtime once the phase
  ids are agreed. Packet 1.0 still uses a single placeholder phase
  and passes ``StrategyPrior.applicability_scope`` (now
  funnel/regime-only) through unchanged in
  ``StrategyPrior.applicability_phase``; the field is intentionally
  kept as a transparent passthrough until protocol-runtime ACTIVE
  re-keys those scopes to TemporalArc phase ids.
* Translate ``GrowthAdvisorKnowledgeSeed`` / ``signature_cases``
  into ``BehaviorProtocol`` fields: those continue to flow through
  the existing ``compiler.build_growth_advisor_package`` path into
  ``domain_knowledge`` / ``case_memory``. The BehaviorProtocol
  layer only carries protocol *metadata* (identity / boundary /
  strategy / temporal / drives), not domain content.
* Set ``identity_assertion.required_regime_compatibility`` —
  packet 1.3+ wires R14 regime cross-check.
"""

from __future__ import annotations

from volvence_zero.behavior_protocol import (
    ActivationConditions,
    BehaviorProtocol,
    BehaviorProtocolSignalSource,
    BoundaryContract,
    BoundarySeverity,
    FailureSignal,
    IdentityAssertion,
    KnowledgeSeed,
    ProtocolSourceKind,
    ReviewLevel,
    ReviewStatus,
    SignatureCase,
    StrategyPrior,
    SuccessSignal,
    TemporalArc,
    TemporalPhase,
)

from lifeform_domain_growth_advisor.profile import (
    GrowthAdvisorBoundaryPrior,
    GrowthAdvisorDrivePrior,
    GrowthAdvisorKnowledgeSeed,
    GrowthAdvisorProfile,
    GrowthAdvisorSignatureCase,
    GrowthAdvisorStrategyPrior,
)


_GROWTH_ADVISOR_JURISDICTION_TAGS: tuple[str, ...] = ("private-domain-companion",)
# Mirror the vertical's hardcoded values (see
# ``lifeform_domain_growth_advisor.compiler._case_record``) so the
# protocol compile path produces ``CaseMemoryRecord`` records
# byte-equivalent to the legacy vertical compile (modulo case_id
# lineage prefix). Pinned by
# ``tests/contracts/test_protocol_case_matched_control.py``.
_GROWTH_ADVISOR_CASE_DELAYED_SIGNAL_COUNT: int = 1
_GROWTH_ADVISOR_CASE_RECONSTRUCTION_SOURCE: str = "reviewed-growth-advisor-profile"


_PROTOCOL_ID_PREFIX = "growth_advisor"
_PLACEHOLDER_PHASE_ID = "long_term_companion"


def growth_advisor_profile_to_behavior_protocol(
    profile: GrowthAdvisorProfile,
) -> BehaviorProtocol:
    """Convert a reviewed ``GrowthAdvisorProfile`` into a ``BehaviorProtocol``.

    Round-trip invariant: counts of boundary / strategy / drive
    items survive unchanged; PE signals are synthesised 1:1 from
    drives (4 success + 4 failure for the canonical Cheng Laoshi
    profile).

    The returned protocol has ``review_status=ACTIVE`` because the
    source ``GrowthAdvisorProfile`` is itself reviewed (it ships in
    a vetted fixture module). Packet 1.5+ ModificationGate flow may
    require freshly-uptaken protocols to land at ``DRAFT`` first;
    fixture conversions skip that gate.
    """

    boundary_contracts = tuple(
        _convert_boundary(b) for b in profile.boundary_priors
    )
    strategy_priors = tuple(
        _convert_strategy(s) for s in profile.strategy_priors
    )
    knowledge_seeds = tuple(
        _convert_knowledge_seed(seed) for seed in profile.knowledge_seeds
    )
    signature_cases = tuple(
        _convert_signature_case(case) for case in profile.signature_cases
    )
    success_signals, failure_signals = _synthesize_pe_signals_from_drives(
        profile.drive_priors
    )
    identity_assertion = _synthesize_identity_assertion()
    activation_conditions = _synthesize_activation_conditions(profile)
    temporal_arc = _placeholder_temporal_arc(profile)

    return BehaviorProtocol(
        protocol_id=f"{_PROTOCOL_ID_PREFIX}:{profile.profile_id}",
        version=profile.version,
        advisor_name=profile.advisor_name,
        description=(
            f"Behavior protocol compiled from reviewed fixture "
            f"{profile.profile_id!r}. "
            f"{profile.description}"
        ),
        source_kind=ProtocolSourceKind.FIXTURE,
        source_locator=profile.source_uri,
        identity_assertion=identity_assertion,
        boundary_contracts=boundary_contracts,
        activation_conditions=activation_conditions,
        strategy_priors=strategy_priors,
        temporal_arc=temporal_arc,
        success_signals=success_signals,
        failure_signals=failure_signals,
        knowledge_seeds=knowledge_seeds,
        signature_cases=signature_cases,
        parent_protocol_id=None,
        review_status=ReviewStatus.ACTIVE,
        revision_log=(),
        legacy_fixture=False,
    )


# ---------------------------------------------------------------------------
# Private converters
# ---------------------------------------------------------------------------


def _convert_boundary(prior: GrowthAdvisorBoundaryPrior) -> BoundaryContract:
    """Lossless `GrowthAdvisorBoundaryPrior` → `BoundaryContract`.

    Packet 1.2 added ``regime_id`` / ``answer_depth_limit_hint`` /
    ``clarification_required`` to ``BoundaryContract`` so the
    forward path matches the existing ``GrowthAdvisorBoundaryPrior``
    surface 1:1 — and so the protocol-side compile path
    (``compile_protocol_to_application_artifacts``) produces a
    ``BoundaryPriorHint`` indistinguishable from the legacy
    ``compiler._boundary_hint`` output.
    """

    return BoundaryContract(
        boundary_id=prior.boundary_id,
        description=prior.description,
        trigger_reasons=prior.trigger_reasons,
        blocked_topics=prior.blocked_topics,
        required_disclaimers=prior.required_disclaimers,
        refer_out_required=prior.refer_out_required,
        regime_id=prior.regime_id,
        answer_depth_limit_hint=prior.answer_depth_limit_hint,
        clarification_required=prior.clarification_required,
        severity=BoundarySeverity.HARD_BLOCK,
        review_level=ReviewLevel.L3,
        confidence=prior.confidence,
    )


def _convert_strategy(prior: GrowthAdvisorStrategyPrior) -> StrategyPrior:
    """Lossless `GrowthAdvisorStrategyPrior` → `StrategyPrior`.

    Packet 1.3b added ``recommended_regime`` /
    ``knowledge_weight_hint`` / ``experience_weight_hint`` to
    ``StrategyPrior`` so this conversion captures every field
    `PlaybookRule` carries (the eventual compile target). Result:
    ``compile_protocol_to_application_artifacts(...)`` produces a
    ``PlaybookRule`` indistinguishable (modulo lineage prefix on
    ``rule_id``) from the legacy
    ``compiler._playbook_rule(...)`` output. Pinned by
    ``tests/contracts/test_protocol_strategy_matched_control.py``.
    """

    return StrategyPrior(
        rule_id=prior.rule_id,
        problem_pattern=prior.problem_pattern,
        recommended_ordering=prior.recommended_ordering,
        recommended_pacing=prior.recommended_pacing,
        avoid_patterns=prior.avoid_patterns,
        applicability_phase=prior.applicability_scope,
        recommended_regime=prior.recommended_regime,
        knowledge_weight_hint=prior.knowledge_weight_hint,
        experience_weight_hint=prior.experience_weight_hint,
        initial_weight=1.0,
        pe_decay_rate=0.0,
        pe_reinforce_rate=0.0,
        minimum_weight_floor=0.0,
        revision_history=(),
        confidence=prior.confidence,
        description=prior.description,
    )


def _convert_signature_case(case: GrowthAdvisorSignatureCase) -> SignatureCase:
    """Lossless `GrowthAdvisorSignatureCase` → `SignatureCase`.

    Packet 1.4b added ``BehaviorProtocol.signature_cases`` so the
    fixture-side case corpus flows through the protocol path into
    ``ApplicationCaseMemoryStore`` (mirroring the boundary /
    strategy / knowledge paths). All semantic fields are preserved
    1:1; ``delayed_signal_count`` and ``reconstruction_source`` are
    set to the vertical's hardcoded values so the protocol →
    ``CaseMemoryRecord`` compile is byte-equivalent to
    ``compiler._case_record`` (modulo the lineage prefix on
    ``case_id``).
    Pinned by ``tests/contracts/test_protocol_case_matched_control.py``.
    """

    return SignatureCase(
        case_id=case.case_id,
        domain=case.domain,
        problem_pattern=case.problem_pattern,
        user_state_pattern=case.user_state_pattern,
        risk_markers=case.risk_markers,
        track_tags=case.track_tags,
        regime_tags=case.regime_tags,
        intervention_ordering=case.intervention_ordering,
        outcome_label=case.outcome_label,
        confidence=case.confidence,
        description=case.description,
        relevance_score=case.relevance_score,
        escalation_observed=case.escalation_observed,
        repair_observed=case.repair_observed,
        delayed_signal_count=_GROWTH_ADVISOR_CASE_DELAYED_SIGNAL_COUNT,
        reconstruction_source=_GROWTH_ADVISOR_CASE_RECONSTRUCTION_SOURCE,
    )


def _convert_knowledge_seed(seed: GrowthAdvisorKnowledgeSeed) -> KnowledgeSeed:
    """Lossless `GrowthAdvisorKnowledgeSeed` → `KnowledgeSeed`.

    Packet 1.4a added ``BehaviorProtocol.knowledge_seeds`` so the
    fixture-side knowledge corpus flows through the protocol path
    into ``ApplicationDomainKnowledgeStore`` (mirroring the
    boundary / strategy paths). All semantic fields are preserved
    1:1; ``jurisdiction_tags`` defaults to the growth-advisor
    vertical's hard-coded ``("private-domain-companion",)`` to
    match the legacy ``compiler._knowledge_record`` output.
    Pinned by ``tests/contracts/test_protocol_knowledge_matched_control.py``.
    """

    return KnowledgeSeed(
        seed_id=seed.seed_id,
        domain=seed.domain,
        title=seed.title,
        summary=seed.summary,
        snippet=seed.snippet,
        evidence_locator=seed.evidence_locator,
        confidence=seed.confidence,
        evidence_strength=seed.evidence_strength,
        topic_tags=seed.topic_tags,
        source_type=seed.source_type,
        freshness_label=seed.freshness_label,
        jurisdiction_tags=_GROWTH_ADVISOR_JURISDICTION_TAGS,
        conflict_markers=(),
    )


def _synthesize_pe_signals_from_drives(
    drive_priors: tuple[GrowthAdvisorDrivePrior, ...],
) -> tuple[tuple[SuccessSignal, ...], tuple[FailureSignal, ...]]:
    """Synthesize success/failure PE signals from drive priors.

    Mapping (spec §协议 → PE 映射 option ii):
    - in-band → SuccessSignal sourced by ``DRIVE_HOMEOSTASIS_HOLD``
    - out-of-band → FailureSignal sourced by
      ``DRIVE_HOMEOSTASIS_BREACH``

    Both signals carry ``weight_in_pe = drive.pe_weight`` so the
    protocol's PE readout aggregates exactly the underlying vitals
    deviation.

    For a profile with N drives, this returns N success + N failure
    signals. cheng_laoshi has 4 drives → 4 + 4 = 8 signals.
    """

    successes = tuple(
        SuccessSignal(
            signal_id=_drive_signal_id(d, "hold"),
            description=(
                f"drive {d.name!r} stays within homeostatic band "
                f"{d.homeostatic_band!r}; vitals contributes 0 PE."
            ),
            measurable_via=BehaviorProtocolSignalSource.DRIVE_HOMEOSTASIS_HOLD,
            expected_value_range=d.homeostatic_band,
            weight_in_pe=d.pe_weight,
        )
        for d in drive_priors
    )
    failures = tuple(
        FailureSignal(
            signal_id=_drive_signal_id(d, "breach"),
            description=(
                f"drive {d.name!r} deviates outside homeostatic band "
                f"{d.homeostatic_band!r}; vitals reports nonzero PE."
            ),
            measurable_via=BehaviorProtocolSignalSource.DRIVE_HOMEOSTASIS_BREACH,
            threshold=0.0,
            weight_in_pe=d.pe_weight,
        )
        for d in drive_priors
    )
    return successes, failures


def _drive_signal_id(drive: GrowthAdvisorDrivePrior, suffix: str) -> str:
    return f"drive:{drive.name}:{suffix}"


def _synthesize_identity_assertion() -> IdentityAssertion:
    """Hard-coded identity stance for the growth-advisor archetype.

    These traits are documented in the cheng_laoshi profile module
    docstring (warm peer-mom register, long horizon, anti-pitch).
    Packet 1.3+ will read these from R7 dual-track Self snapshot
    instead of synthesising them; until then, the fixture supplies
    the canonical archetype stance.
    """

    return IdentityAssertion(
        requires_self_traits=("warm_peer_register", "long_horizon"),
        forbidden_self_traits=("high_pressure_sales",),
        required_regime_compatibility=(),
    )


def _synthesize_activation_conditions(
    profile: GrowthAdvisorProfile,
) -> ActivationConditions:
    """Minimal activation conditions for fixture-uptake protocols.

    Packet 1.0 ships:
    - empty ``context_match_signals`` (real signals land in packet
      1.5+ when ``ActivationController`` consumes context)
    - empty co-activation lists (no known sibling protocols to
      conflict with at packet 1.0 — only one fixture is loaded
      per session typically)
    - ``minimum_weight_floor=0.0`` (no floor; equal-weight fallback
      handles solo or sibling cases uniformly)
    """

    del profile  # reserved for future signal mining
    return ActivationConditions(
        context_match_signals=(),
        co_activation_compatible=(),
        co_activation_incompatible=(),
        minimum_weight_floor=0.0,
    )


def _placeholder_temporal_arc(profile: GrowthAdvisorProfile) -> TemporalArc:
    """Single placeholder phase covering the full fixture lifetime.

    The reviewed ``cheng_laoshi`` profile carries an onboarding arc
    intent (icebreaker → baseline → empathy → pain mining → rapport
    → targeted advice → summary) across 7 ``playbook-day*``
    ``StrategyPrior``s. Calendar-day routing was removed on
    2026-05-14; PE-driven ``ProgressionSignal`` synthesis from the
    reviewed onboarding intent is the protocol-runtime ACTIVE work.
    Packet 1.0 keeps a single ``long_term_companion`` phase so the
    schema is non-empty without claiming a phase semantics it doesn't
    yet implement.
    """

    del profile  # reserved for future phase translation
    return TemporalArc(
        phases=(
            TemporalPhase(
                phase_id=_PLACEHOLDER_PHASE_ID,
                description=(
                    "Placeholder phase for packet 1.0 SHADOW. "
                    "Real PE-driven phase progression lands when "
                    "protocol-runtime synthesises ProgressionSignals "
                    "from the reviewed onboarding arc intent."
                ),
            ),
        ),
    )


__all__ = ["growth_advisor_profile_to_behavior_protocol"]
