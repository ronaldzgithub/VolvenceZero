"""Compile path: ``BehaviorProtocol`` → ``ProtocolApplicationArtifacts``.

Pins the lossless conversion of:

* ``BoundaryContract`` → ``BoundaryPriorHint`` (packet 1.2)
* ``StrategyPrior`` → ``PlaybookRule`` (packet 1.3b)
* ``KnowledgeSeed`` → ``DomainKnowledgeRecord`` (packet 1.4a)
* ``SignatureCase`` → ``CaseMemoryRecord`` (packet 1.4b)

All four kinds compile in one pass; this file covers the
field-level invariants for each. Matched-control parity with the
vertical compile path is in
``tests/contracts/test_protocol_*_matched_control.py``.

Why these tests matter: ``ProtocolRegistryModule.load_protocol``
calls the compiler when an ``ApplicationRareHeavyState`` is
injected, so any silent drift in the compile output would
silently corrupt boundary execution. The tests guard:

* All boundary fields shared between contract and hint pass
  through unchanged.
* The ``hint_id`` is namespaced with the protocol id so future
  unload paths can identify protocol-driven entries.
* The compile function is pure and idempotent.
* The output type is ``ProtocolApplicationArtifacts`` (frozen
  dataclass).
"""

from __future__ import annotations

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.application.storage import (
    CaseLifecycle,
    CaseMemoryRecord,
    DomainKnowledgeRecord,
)
from volvence_zero.application.types import BoundaryPriorHint, PlaybookRule
from volvence_zero.behavior_protocol import BehaviorProtocol
from volvence_zero.protocol_runtime import (
    ProtocolApplicationArtifacts,
    compile_protocol_to_application_artifacts,
)
from volvence_zero.protocol_runtime.compiler import (
    _protocol_case_id,
    _protocol_hint_id,
    _protocol_record_id,
    _protocol_rule_id,
)


def _cheng_laoshi_protocol() -> BehaviorProtocol:
    return growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


def test_returns_protocol_application_artifacts() -> None:
    artifacts = compile_protocol_to_application_artifacts(_cheng_laoshi_protocol())
    assert isinstance(artifacts, ProtocolApplicationArtifacts)


def test_artifacts_carry_protocol_id() -> None:
    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    assert artifacts.protocol_id == bp.protocol_id


# ---------------------------------------------------------------------------
# Boundary count parity
# ---------------------------------------------------------------------------


def test_boundary_hint_count_matches_protocol() -> None:
    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    assert len(artifacts.boundary_prior_hints) == len(bp.boundary_contracts)


# ---------------------------------------------------------------------------
# Lossless field passthrough
# ---------------------------------------------------------------------------


def test_each_hint_is_boundary_prior_hint_instance() -> None:
    artifacts = compile_protocol_to_application_artifacts(_cheng_laoshi_protocol())
    for hint in artifacts.boundary_prior_hints:
        assert isinstance(hint, BoundaryPriorHint)


def test_hint_id_namespaced_with_protocol_id() -> None:
    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    expected = {
        f"protocol:{bp.protocol_id}:boundary:{c.boundary_id}"
        for c in bp.boundary_contracts
    }
    actual = {h.hint_id for h in artifacts.boundary_prior_hints}
    assert actual == expected


def test_protocol_hint_id_helper_format() -> None:
    """The helper function is the SSOT for the namespace scheme."""
    assert (
        _protocol_hint_id("growth_advisor:cheng-laoshi", "bp-no-hard-sell")
        == "protocol:growth_advisor:cheng-laoshi:boundary:bp-no-hard-sell"
    )


def test_hint_trigger_reasons_carry_through() -> None:
    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    by_boundary_id = {c.boundary_id: c for c in bp.boundary_contracts}
    for hint in artifacts.boundary_prior_hints:
        # Recover boundary_id from namespaced hint_id
        boundary_id = hint.hint_id.rsplit(":", 1)[-1]
        contract = by_boundary_id[boundary_id]
        assert hint.trigger_reasons == contract.trigger_reasons


def test_hint_blocked_topics_and_disclaimers_carry_through() -> None:
    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    by_boundary_id = {c.boundary_id: c for c in bp.boundary_contracts}
    for hint in artifacts.boundary_prior_hints:
        boundary_id = hint.hint_id.rsplit(":", 1)[-1]
        contract = by_boundary_id[boundary_id]
        assert hint.blocked_topics == contract.blocked_topics
        assert hint.required_disclaimers == contract.required_disclaimers


def test_hint_carries_three_packet_1_2_fields_from_contract() -> None:
    """``regime_id`` / ``answer_depth_limit_hint`` /
    ``clarification_required`` were added to ``BoundaryContract`` in
    packet 1.2 specifically so the compile path is lossless. Verify
    they pass through.
    """

    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    by_boundary_id = {c.boundary_id: c for c in bp.boundary_contracts}
    for hint in artifacts.boundary_prior_hints:
        boundary_id = hint.hint_id.rsplit(":", 1)[-1]
        contract = by_boundary_id[boundary_id]
        assert hint.regime_id == contract.regime_id
        assert hint.answer_depth_limit_hint == contract.answer_depth_limit_hint
        assert hint.clarification_required == contract.clarification_required


def test_hint_confidence_and_description_carry_through() -> None:
    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    by_boundary_id = {c.boundary_id: c for c in bp.boundary_contracts}
    for hint in artifacts.boundary_prior_hints:
        boundary_id = hint.hint_id.rsplit(":", 1)[-1]
        contract = by_boundary_id[boundary_id]
        assert hint.confidence == contract.confidence
        assert hint.description == contract.description


def test_hint_refer_out_required_carries_through() -> None:
    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    by_boundary_id = {c.boundary_id: c for c in bp.boundary_contracts}
    for hint in artifacts.boundary_prior_hints:
        boundary_id = hint.hint_id.rsplit(":", 1)[-1]
        contract = by_boundary_id[boundary_id]
        assert hint.refer_out_required == contract.refer_out_required


# ---------------------------------------------------------------------------
# Idempotency / purity
# ---------------------------------------------------------------------------


def test_compile_is_pure_and_does_not_mutate_protocol() -> None:
    bp = _cheng_laoshi_protocol()
    pre_repr = repr(bp)
    compile_protocol_to_application_artifacts(bp)
    compile_protocol_to_application_artifacts(bp)
    assert repr(bp) == pre_repr


def test_compile_is_idempotent_same_input_same_output() -> None:
    bp = _cheng_laoshi_protocol()
    a = compile_protocol_to_application_artifacts(bp)
    b = compile_protocol_to_application_artifacts(bp)
    assert a == b


# ---------------------------------------------------------------------------
# Type guard
# ---------------------------------------------------------------------------


def test_compile_rejects_non_behavior_protocol() -> None:
    import pytest

    with pytest.raises(TypeError):
        compile_protocol_to_application_artifacts("not a protocol")


# ---------------------------------------------------------------------------
# Packet 1.3b: StrategyPrior → PlaybookRule
# ---------------------------------------------------------------------------


def test_artifacts_include_playbook_rules_field() -> None:
    artifacts = compile_protocol_to_application_artifacts(_cheng_laoshi_protocol())
    assert hasattr(artifacts, "playbook_rules")
    assert isinstance(artifacts.playbook_rules, tuple)


def test_playbook_rule_count_matches_protocol() -> None:
    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    assert len(artifacts.playbook_rules) == len(bp.strategy_priors)


def test_each_playbook_rule_is_playbook_rule_instance() -> None:
    artifacts = compile_protocol_to_application_artifacts(_cheng_laoshi_protocol())
    for rule in artifacts.playbook_rules:
        assert isinstance(rule, PlaybookRule)


def test_playbook_rule_id_namespaced_with_protocol_id() -> None:
    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    expected = {
        f"protocol:{bp.protocol_id}:playbook:{p.rule_id}"
        for p in bp.strategy_priors
    }
    actual = {r.rule_id for r in artifacts.playbook_rules}
    assert actual == expected


def test_protocol_rule_id_helper_format() -> None:
    """The helper function is the SSOT for the namespace scheme."""
    assert (
        _protocol_rule_id("growth_advisor:cheng-laoshi", "funnel-height")
        == "protocol:growth_advisor:cheng-laoshi:playbook:funnel-height"
    )


def test_playbook_rule_field_passthrough() -> None:
    """Per-field passthrough for shared fields between StrategyPrior
    and PlaybookRule.
    """

    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    by_rule_id = {p.rule_id: p for p in bp.strategy_priors}
    for rule in artifacts.playbook_rules:
        # Recover original rule_id from namespaced PlaybookRule id
        original_rule_id = rule.rule_id.rsplit(":", 1)[-1]
        prior = by_rule_id[original_rule_id]
        assert rule.problem_pattern == prior.problem_pattern
        assert rule.recommended_regime == prior.recommended_regime
        assert rule.recommended_ordering == prior.recommended_ordering
        assert rule.recommended_pacing == prior.recommended_pacing
        assert rule.avoid_patterns == prior.avoid_patterns
        assert rule.knowledge_weight_hint == prior.knowledge_weight_hint
        assert rule.experience_weight_hint == prior.experience_weight_hint
        # applicability_phase ↔ applicability_scope rename
        assert rule.applicability_scope == prior.applicability_phase
        assert rule.confidence == prior.confidence
        assert rule.description == prior.description


def test_playbook_rule_uses_default_continuum_fields() -> None:
    """``StrategyPrior`` doesn't carry continuum fields; compile uses
    ``PlaybookRule`` defaults (``None`` / ``0.0``).
    """

    artifacts = compile_protocol_to_application_artifacts(_cheng_laoshi_protocol())
    for rule in artifacts.playbook_rules:
        assert rule.continuum_band_id is None
        assert rule.mean_continuum_position == 0.0


def test_playbook_compile_pe_revision_metadata_dropped_silently() -> None:
    """PE-revision metadata on ``StrategyPrior``
    (``initial_weight`` / ``pe_decay_rate`` / ``pe_reinforce_rate`` /
    ``minimum_weight_floor`` / ``revision_history``) is intentionally
    not carried into ``PlaybookRule``; ``StrategyPlaybookModule``
    doesn't read it. Future activation controller (packet 1.5+)
    consumes those fields directly off ``BehaviorProtocol``.
    """

    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    # PlaybookRule has no fields named like the PE-revision metadata
    rule_field_names = {f.name for f in PlaybookRule.__dataclass_fields__.values()}
    pe_revision_fields = {
        "initial_weight",
        "pe_decay_rate",
        "pe_reinforce_rate",
        "minimum_weight_floor",
        "revision_history",
    }
    overlap = rule_field_names & pe_revision_fields
    assert overlap == set(), (
        f"PlaybookRule unexpectedly carries PE-revision fields: {overlap}; "
        "compile path needs review (these belong on protocol side only)"
    )


# ---------------------------------------------------------------------------
# Packet 1.4a: KnowledgeSeed → DomainKnowledgeRecord
# ---------------------------------------------------------------------------


def test_artifacts_include_domain_knowledge_records_field() -> None:
    artifacts = compile_protocol_to_application_artifacts(_cheng_laoshi_protocol())
    assert hasattr(artifacts, "domain_knowledge_records")
    assert isinstance(artifacts.domain_knowledge_records, tuple)


def test_domain_knowledge_record_count_matches_protocol() -> None:
    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    assert len(artifacts.domain_knowledge_records) == len(bp.knowledge_seeds)


def test_each_knowledge_record_is_domain_knowledge_record_instance() -> None:
    artifacts = compile_protocol_to_application_artifacts(_cheng_laoshi_protocol())
    for record in artifacts.domain_knowledge_records:
        assert isinstance(record, DomainKnowledgeRecord)


def test_knowledge_record_id_namespaced_with_protocol_id() -> None:
    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    expected = {
        f"protocol:{bp.protocol_id}:knowledge:{seed.seed_id}"
        for seed in bp.knowledge_seeds
    }
    actual = {r.record_id for r in artifacts.domain_knowledge_records}
    assert actual == expected


def test_protocol_record_id_helper_format() -> None:
    """The helper function is the SSOT for the namespace scheme."""
    assert (
        _protocol_record_id("growth_advisor:cheng-laoshi", "persona-identity")
        == "protocol:growth_advisor:cheng-laoshi:knowledge:persona-identity"
    )


def test_knowledge_record_field_passthrough() -> None:
    """Per-field passthrough for shared fields between KnowledgeSeed
    and DomainKnowledgeRecord (modulo evidence_locator → locator
    rename and url derivation from protocol.source_locator).
    """

    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    by_seed_id = {s.seed_id: s for s in bp.knowledge_seeds}
    for record in artifacts.domain_knowledge_records:
        # Recover original seed_id from namespaced record_id
        seed_id = record.record_id.rsplit(":", 1)[-1]
        seed = by_seed_id[seed_id]
        assert record.domain == seed.domain
        assert record.title == seed.title
        assert record.summary == seed.summary
        assert record.snippet == seed.snippet
        # evidence_locator → locator rename
        assert record.locator == seed.evidence_locator
        assert record.confidence == seed.confidence
        assert record.evidence_strength == seed.evidence_strength
        assert record.topic_tags == seed.topic_tags
        assert record.source_type == seed.source_type
        assert record.freshness_label == seed.freshness_label
        assert record.jurisdiction_tags == seed.jurisdiction_tags
        assert record.conflict_markers == seed.conflict_markers


def test_knowledge_record_url_derived_from_protocol_source_locator() -> None:
    """``DomainKnowledgeRecord.url`` is derived from
    ``BehaviorProtocol.source_locator``, mirroring the vertical's
    ``DomainKnowledgeRecord.url = profile.source_uri`` choice.
    ``KnowledgeSeed`` itself has no ``url`` field.
    """

    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    for record in artifacts.domain_knowledge_records:
        assert record.url == bp.source_locator


# ---------------------------------------------------------------------------
# Packet 1.4b: SignatureCase → CaseMemoryRecord
# ---------------------------------------------------------------------------


def test_artifacts_include_case_memory_records_field() -> None:
    artifacts = compile_protocol_to_application_artifacts(_cheng_laoshi_protocol())
    assert hasattr(artifacts, "case_memory_records")
    assert isinstance(artifacts.case_memory_records, tuple)


def test_case_memory_record_count_matches_protocol() -> None:
    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    assert len(artifacts.case_memory_records) == len(bp.signature_cases)


def test_each_case_record_is_case_memory_record_instance() -> None:
    artifacts = compile_protocol_to_application_artifacts(_cheng_laoshi_protocol())
    for record in artifacts.case_memory_records:
        assert isinstance(record, CaseMemoryRecord)


def test_case_id_namespaced_with_protocol_id() -> None:
    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    expected = {
        f"protocol:{bp.protocol_id}:case:{c.case_id}"
        for c in bp.signature_cases
    }
    actual = {r.case_id for r in artifacts.case_memory_records}
    assert actual == expected


def test_protocol_case_id_helper_format() -> None:
    """The helper function is the SSOT for the namespace scheme."""
    assert (
        _protocol_case_id("growth_advisor:cheng-laoshi", "day1-icebreaker")
        == "protocol:growth_advisor:cheng-laoshi:case:day1-icebreaker"
    )


def test_case_record_field_passthrough() -> None:
    """Per-field passthrough for shared fields between SignatureCase
    and CaseMemoryRecord.
    """

    bp = _cheng_laoshi_protocol()
    artifacts = compile_protocol_to_application_artifacts(bp)
    by_case_id = {c.case_id: c for c in bp.signature_cases}
    for record in artifacts.case_memory_records:
        # Recover original case_id from namespaced record case_id
        original_case_id = record.case_id.rsplit(":", 1)[-1]
        case = by_case_id[original_case_id]
        assert record.domain == case.domain
        assert record.problem_pattern == case.problem_pattern
        assert record.user_state_pattern == case.user_state_pattern
        assert record.risk_markers == case.risk_markers
        assert record.track_tags == case.track_tags
        assert record.regime_tags == case.regime_tags
        assert record.intervention_ordering == case.intervention_ordering
        assert record.outcome_label == case.outcome_label
        assert record.delayed_signal_count == case.delayed_signal_count
        assert record.escalation_observed == case.escalation_observed
        assert record.repair_observed == case.repair_observed
        assert record.confidence == case.confidence
        assert record.relevance_score == case.relevance_score
        assert record.description == case.description
        assert record.reconstruction_source == case.reconstruction_source


def test_case_record_uses_default_continuum_fields() -> None:
    """``SignatureCase`` doesn't carry continuum fields; compile
    uses ``CaseMemoryRecord`` defaults.
    """

    artifacts = compile_protocol_to_application_artifacts(_cheng_laoshi_protocol())
    for record in artifacts.case_memory_records:
        assert record.continuum_profile_id is None
        assert record.continuum_band_id is None
        assert record.continuum_position == 0.0
        assert record.continuum_update_frequency == 0.0


def test_case_record_uses_default_lifecycle_validated() -> None:
    """All protocol-driven cases land at ``CaseLifecycle.VALIDATED``
    (never CANDIDATE / PROVISIONAL / RETIRED). ttl/expires/origin
    are also default-none. Protocols are not the surface for
    lifecycle-stage management.
    """

    artifacts = compile_protocol_to_application_artifacts(_cheng_laoshi_protocol())
    for record in artifacts.case_memory_records:
        assert record.lifecycle is CaseLifecycle.VALIDATED
        assert record.ttl_seconds is None
        assert record.expires_at_tick is None
        assert record.provisional_origin == ""
