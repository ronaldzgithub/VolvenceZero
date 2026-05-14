"""FixtureUptake conversion tests: GrowthAdvisorProfile -> BehaviorProtocol.

Pins the lossless-on-counts invariant for the cheng_laoshi fixture
plus the PE-signal synthesis rule from
``docs/specs/protocol-runtime.md`` §协议 → PE 映射 option ii.

The conversion is *additive*: the existing
``DomainExperiencePackage`` / ``VitalsBootstrap`` /
``IngestionEnvelope`` paths still produce their products. These
tests only assert the BehaviorProtocol-side invariants.
"""

from __future__ import annotations

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.behavior_protocol import (
    BehaviorProtocol,
    BehaviorProtocolSignalSource,
    ProtocolSourceKind,
    ReviewStatus,
)


def _converted() -> BehaviorProtocol:
    return growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())


def test_returns_behavior_protocol_instance() -> None:
    assert isinstance(_converted(), BehaviorProtocol)


def test_protocol_id_namespaced_under_growth_advisor() -> None:
    bp = _converted()
    assert bp.protocol_id.startswith("growth_advisor:"), bp.protocol_id
    assert bp.protocol_id == "growth_advisor:cheng-laoshi"


def test_version_carried_through() -> None:
    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)
    assert bp.version == profile.version


def test_advisor_name_carried_through() -> None:
    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)
    assert bp.advisor_name == profile.advisor_name


def test_source_kind_is_fixture() -> None:
    assert _converted().source_kind is ProtocolSourceKind.FIXTURE


def test_source_locator_carries_profile_uri() -> None:
    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)
    assert bp.source_locator == profile.source_uri


def test_review_status_active_for_reviewed_fixture() -> None:
    """Reviewed fixture → ACTIVE; we trust the on-disk review."""
    assert _converted().review_status is ReviewStatus.ACTIVE


def test_legacy_fixture_flag_false_after_drive_synthesis() -> None:
    """4 drives → 8 PE signals satisfies the invariant; no opt-out."""
    assert _converted().legacy_fixture is False


# ---------------------------------------------------------------------------
# Lossless-on-counts invariants
# ---------------------------------------------------------------------------


def test_boundary_count_matches_profile() -> None:
    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)
    assert len(bp.boundary_contracts) == len(profile.boundary_priors)


def test_strategy_count_matches_profile() -> None:
    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)
    assert len(bp.strategy_priors) == len(profile.strategy_priors)


def test_pe_signals_synthesized_one_pair_per_drive() -> None:
    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)
    expected = len(profile.drive_priors)
    assert len(bp.success_signals) == expected
    assert len(bp.failure_signals) == expected
    # cheng_laoshi ships 4 drives → 8 PE signals.
    assert expected == 4


# ---------------------------------------------------------------------------
# PE-signal source vocabulary (closed enum guard)
# ---------------------------------------------------------------------------


def test_success_signals_use_drive_homeostasis_hold_source() -> None:
    bp = _converted()
    sources = {s.measurable_via for s in bp.success_signals}
    assert sources == {BehaviorProtocolSignalSource.DRIVE_HOMEOSTASIS_HOLD}, sources


def test_failure_signals_use_drive_homeostasis_breach_source() -> None:
    bp = _converted()
    sources = {s.measurable_via for s in bp.failure_signals}
    assert sources == {BehaviorProtocolSignalSource.DRIVE_HOMEOSTASIS_BREACH}, sources


# ---------------------------------------------------------------------------
# Drive → signal_id round-trip is reversible
# ---------------------------------------------------------------------------


def test_each_drive_has_paired_hold_and_breach_signals() -> None:
    """One success + one failure signal per drive, paired by drive name."""
    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)

    drive_names = {d.name for d in profile.drive_priors}
    success_ids = {s.signal_id for s in bp.success_signals}
    failure_ids = {f.signal_id for f in bp.failure_signals}

    expected_success = {f"drive:{name}:hold" for name in drive_names}
    expected_failure = {f"drive:{name}:breach" for name in drive_names}

    assert success_ids == expected_success
    assert failure_ids == expected_failure


def test_pe_signal_weights_pass_through_drive_pe_weight() -> None:
    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)

    drive_pe_by_name = {d.name: d.pe_weight for d in profile.drive_priors}
    for signal in bp.success_signals:
        # signal_id format: drive:<name>:hold
        drive_name = signal.signal_id.split(":")[1]
        assert signal.weight_in_pe == drive_pe_by_name[drive_name]
    for signal in bp.failure_signals:
        drive_name = signal.signal_id.split(":")[1]
        assert signal.weight_in_pe == drive_pe_by_name[drive_name]


def test_success_signal_expected_value_range_matches_drive_band() -> None:
    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)
    drive_band_by_name = {d.name: d.homeostatic_band for d in profile.drive_priors}
    for signal in bp.success_signals:
        drive_name = signal.signal_id.split(":")[1]
        assert signal.expected_value_range == drive_band_by_name[drive_name]


# ---------------------------------------------------------------------------
# Strategy applicability_phase carries day-tag scopes through unchanged
# ---------------------------------------------------------------------------


def test_strategy_applicability_phase_carries_through_day_tag_scopes() -> None:
    """Packet 1.0 is intentionally a transparent passthrough.

    The BehaviorProtocol's ``applicability_phase`` mirrors the
    ``GrowthAdvisorStrategyPrior.applicability_scope`` tuple 1:1 so
    the fixture round-trips losslessly. Calendar-day routing
    (``growth_advisor:day{1..7}``) was removed on 2026-05-14; this
    transparent-passthrough invariant is independent of which tags
    the upstream profile chooses to ship (currently funnel/regime
    only). Future protocol-runtime ACTIVE work will rewrite these
    scopes into TemporalArc phase ids; that change rides this same
    1:1 pass-through invariant.
    """

    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)
    by_id = {s.rule_id: s for s in bp.strategy_priors}
    for prior in profile.strategy_priors:
        converted = by_id[prior.rule_id]
        assert converted.applicability_phase == prior.applicability_scope


def test_strategy_packet_1_3b_fields_carry_through() -> None:
    """``recommended_regime`` / ``knowledge_weight_hint`` /
    ``experience_weight_hint`` were added to ``StrategyPrior`` in
    packet 1.3b so the protocol → ``PlaybookRule`` compile is
    lossless. Verify they pass through from
    ``GrowthAdvisorStrategyPrior``.
    """

    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)
    by_id = {s.rule_id: s for s in bp.strategy_priors}
    for prior in profile.strategy_priors:
        converted = by_id[prior.rule_id]
        assert converted.recommended_regime == prior.recommended_regime
        assert converted.knowledge_weight_hint == prior.knowledge_weight_hint
        assert converted.experience_weight_hint == prior.experience_weight_hint


# ---------------------------------------------------------------------------
# Packet 1.4a: knowledge_seeds round-trip
# ---------------------------------------------------------------------------


def test_knowledge_seed_count_matches_profile() -> None:
    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)
    assert len(bp.knowledge_seeds) == len(profile.knowledge_seeds)


def test_knowledge_seed_fields_carry_through() -> None:
    """Lossless conversion of ``GrowthAdvisorKnowledgeSeed`` →
    ``BehaviorProtocol.KnowledgeSeed``.
    """

    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)
    by_id = {s.seed_id: s for s in bp.knowledge_seeds}
    for source_seed in profile.knowledge_seeds:
        converted = by_id[source_seed.seed_id]
        assert converted.domain == source_seed.domain
        assert converted.title == source_seed.title
        assert converted.summary == source_seed.summary
        assert converted.snippet == source_seed.snippet
        assert converted.evidence_locator == source_seed.evidence_locator
        assert converted.confidence == source_seed.confidence
        assert converted.evidence_strength == source_seed.evidence_strength
        assert converted.topic_tags == source_seed.topic_tags
        assert converted.source_type == source_seed.source_type
        assert converted.freshness_label == source_seed.freshness_label


def test_knowledge_seed_jurisdiction_tags_match_vertical_default() -> None:
    """Fixture uptake hard-codes ``("private-domain-companion",)``
    on every ``KnowledgeSeed`` so the protocol → DomainKnowledgeRecord
    compile produces records byte-equivalent to the legacy
    ``compiler._knowledge_record`` output.
    """

    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)
    for seed in bp.knowledge_seeds:
        assert seed.jurisdiction_tags == ("private-domain-companion",)


# ---------------------------------------------------------------------------
# Packet 1.4b: signature_cases round-trip
# ---------------------------------------------------------------------------


def test_signature_case_count_matches_profile() -> None:
    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)
    assert len(bp.signature_cases) == len(profile.signature_cases)


def test_signature_case_fields_carry_through() -> None:
    """Lossless conversion of ``GrowthAdvisorSignatureCase`` →
    ``BehaviorProtocol.SignatureCase``.
    """

    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)
    by_id = {c.case_id: c for c in bp.signature_cases}
    for source_case in profile.signature_cases:
        converted = by_id[source_case.case_id]
        assert converted.domain == source_case.domain
        assert converted.problem_pattern == source_case.problem_pattern
        assert converted.user_state_pattern == source_case.user_state_pattern
        assert converted.risk_markers == source_case.risk_markers
        assert converted.track_tags == source_case.track_tags
        assert converted.regime_tags == source_case.regime_tags
        assert converted.intervention_ordering == source_case.intervention_ordering
        assert converted.outcome_label == source_case.outcome_label
        assert converted.confidence == source_case.confidence
        assert converted.relevance_score == source_case.relevance_score
        assert converted.escalation_observed == source_case.escalation_observed
        assert converted.repair_observed == source_case.repair_observed
        assert converted.description == source_case.description


def test_signature_case_carries_vertical_hardcoded_metadata() -> None:
    """FixtureUptake sets ``delayed_signal_count=1`` and
    ``reconstruction_source="reviewed-growth-advisor-profile"`` on
    every ``SignatureCase`` so the protocol → CaseMemoryRecord
    compile is byte-equivalent to the legacy ``compiler._case_record``
    output.
    """

    profile = build_cheng_laoshi_profile()
    bp = growth_advisor_profile_to_behavior_protocol(profile)
    for case in bp.signature_cases:
        assert case.delayed_signal_count == 1
        assert case.reconstruction_source == "reviewed-growth-advisor-profile"


# ---------------------------------------------------------------------------
# Identity assertion synthesised
# ---------------------------------------------------------------------------


def test_identity_assertion_includes_required_traits() -> None:
    bp = _converted()
    assert "warm_peer_register" in bp.identity_assertion.requires_self_traits
    assert "long_horizon" in bp.identity_assertion.requires_self_traits


def test_identity_assertion_forbids_high_pressure_sales() -> None:
    bp = _converted()
    assert (
        "high_pressure_sales" in bp.identity_assertion.forbidden_self_traits
    )


# ---------------------------------------------------------------------------
# Temporal arc placeholder (packet 1.0)
# ---------------------------------------------------------------------------


def test_temporal_arc_has_single_placeholder_phase() -> None:
    """Packet 1.0 intentionally ships one placeholder phase.

    Packet 1.4+ replaces this with PE-driven phase progression.
    """

    bp = _converted()
    assert len(bp.temporal_arc.phases) == 1
    assert bp.temporal_arc.phases[0].phase_id == "long_term_companion"


# ---------------------------------------------------------------------------
# Cheng Laoshi original fixture remains untouched (packet 1.0 invariant)
# ---------------------------------------------------------------------------


def test_cheng_laoshi_profile_unchanged_after_conversion() -> None:
    """Conversion must not mutate the original ``GrowthAdvisorProfile``.

    Packet 1.0's ``cheng_laoshi.py`` is preserved unchanged as a
    regression baseline.
    """

    profile_a = build_cheng_laoshi_profile()
    pre_strategy_count = len(profile_a.strategy_priors)
    pre_boundary_count = len(profile_a.boundary_priors)
    pre_drive_count = len(profile_a.drive_priors)

    growth_advisor_profile_to_behavior_protocol(profile_a)

    profile_b = build_cheng_laoshi_profile()
    assert len(profile_a.strategy_priors) == pre_strategy_count
    assert len(profile_a.boundary_priors) == pre_boundary_count
    assert len(profile_a.drive_priors) == pre_drive_count
    # Profiles built from the same source should be structurally
    # identical pre and post conversion (frozen dataclasses; no
    # global state mutation).
    assert profile_a == profile_b
