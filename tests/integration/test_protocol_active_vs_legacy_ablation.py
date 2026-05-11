"""Packet 4.1: protocol-driven vs legacy ablation harness.

Asserts the matched-control invariant for the post-1.5a' / 4.0
production path:

* Loading a ``BehaviorProtocol`` (cheng_laoshi) into the
  application-tier owner adds ``protocol:`` lineage-prefixed
  entries to the canonical application stores
  (``ApplicationRareHeavyState`` for boundary / playbook;
  ``ApplicationDomainKnowledgeStore`` for domain knowledge;
  ``ApplicationCaseMemoryStore`` for case memory).
* NOT loading any protocol leaves those stores without ``protocol:``
  lineage entries.
* The ``protocol:cheng-laoshi:*`` lineage IS the in-band evidence
  that the 1.x compile path is producing artifacts that downstream
  ACTIVE consumers will pick up.

This is the production-path version of the matched-control test
that ``tests/contracts/test_protocol_boundary_matched_control.py``
already pins for the boundary slice. Packet 4.1 generalizes the
guarantee across all four artifact families and ties it to the
ablation contract that motivates the ACTIVE-default flip in 4.0.
"""

from __future__ import annotations

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState
from volvence_zero.application.storage import (
    ApplicationCaseMemoryStore,
    ApplicationDomainKnowledgeStore,
)
from volvence_zero.protocol_runtime import ProtocolRegistryModule


# ---------------------------------------------------------------------------
# Ablation builders
# ---------------------------------------------------------------------------


def _build_loaded_state() -> tuple[
    ApplicationRareHeavyState,
    ApplicationDomainKnowledgeStore,
    ApplicationCaseMemoryStore,
]:
    """Production-shape: load cheng_laoshi protocol via the registry."""
    rare = ApplicationRareHeavyState()
    knowledge = ApplicationDomainKnowledgeStore()
    case_memory = ApplicationCaseMemoryStore()
    module = ProtocolRegistryModule(
        application_rare_heavy_state=rare,
        domain_knowledge_store=knowledge,
        case_memory_store=case_memory,
    )
    bp = growth_advisor_profile_to_behavior_protocol(build_cheng_laoshi_profile())
    module.load_protocol(bp)
    return rare, knowledge, case_memory


def _build_unloaded_state() -> tuple[
    ApplicationRareHeavyState,
    ApplicationDomainKnowledgeStore,
    ApplicationCaseMemoryStore,
]:
    """Ablation: registry exists but no protocol ever loaded."""
    rare = ApplicationRareHeavyState()
    knowledge = ApplicationDomainKnowledgeStore()
    case_memory = ApplicationCaseMemoryStore()
    _module = ProtocolRegistryModule(
        application_rare_heavy_state=rare,
        domain_knowledge_store=knowledge,
        case_memory_store=case_memory,
    )
    return rare, knowledge, case_memory


# ---------------------------------------------------------------------------
# Boundary lineage
# ---------------------------------------------------------------------------


def test_loaded_protocol_introduces_protocol_lineage_in_boundary_hints() -> None:
    rare_loaded, _, _ = _build_loaded_state()

    protocol_prefixed = [
        h
        for h in rare_loaded.boundary_prior_hints
        if h.hint_id.startswith("protocol:growth_advisor:cheng-laoshi:boundary:")
    ]
    assert len(protocol_prefixed) == 4, [
        h.hint_id for h in rare_loaded.boundary_prior_hints
    ]


def test_unloaded_state_has_no_protocol_lineage_in_boundary_hints() -> None:
    rare_unloaded, _, _ = _build_unloaded_state()

    protocol_prefixed = [
        h
        for h in rare_unloaded.boundary_prior_hints
        if h.hint_id.startswith("protocol:")
    ]
    assert protocol_prefixed == []


# ---------------------------------------------------------------------------
# Strategy / playbook lineage
# ---------------------------------------------------------------------------


def test_loaded_protocol_introduces_protocol_lineage_in_playbook_rules() -> None:
    rare_loaded, _, _ = _build_loaded_state()

    protocol_prefixed = [
        r
        for r in rare_loaded.distilled_playbook_rules
        if r.rule_id.startswith("protocol:growth_advisor:cheng-laoshi:playbook:")
    ]
    assert len(protocol_prefixed) >= 1, [
        r.rule_id for r in rare_loaded.distilled_playbook_rules
    ]


def test_unloaded_state_has_no_protocol_lineage_in_playbook_rules() -> None:
    rare_unloaded, _, _ = _build_unloaded_state()

    protocol_prefixed = [
        r
        for r in rare_unloaded.distilled_playbook_rules
        if r.rule_id.startswith("protocol:")
    ]
    assert protocol_prefixed == []


# ---------------------------------------------------------------------------
# Knowledge lineage
# ---------------------------------------------------------------------------


def test_loaded_protocol_introduces_protocol_lineage_in_knowledge_records() -> None:
    _, knowledge_loaded, _ = _build_loaded_state()

    records = knowledge_loaded.records
    protocol_prefixed = [
        r for r in records
        if r.record_id.startswith("protocol:growth_advisor:cheng-laoshi:knowledge:")
    ]
    assert len(protocol_prefixed) >= 1, [r.record_id for r in records]


def test_unloaded_state_has_no_protocol_lineage_in_knowledge_records() -> None:
    _, knowledge_unloaded, _ = _build_unloaded_state()

    protocol_prefixed = [
        r for r in knowledge_unloaded.records
        if r.record_id.startswith("protocol:")
    ]
    assert protocol_prefixed == []


# ---------------------------------------------------------------------------
# Case lineage
# ---------------------------------------------------------------------------


def test_loaded_protocol_introduces_protocol_lineage_in_case_records() -> None:
    _, _, case_loaded = _build_loaded_state()

    records = case_loaded.records
    protocol_prefixed = [
        r for r in records
        if r.case_id.startswith("protocol:growth_advisor:cheng-laoshi:case:")
    ]
    assert len(protocol_prefixed) >= 1, [r.case_id for r in records]


def test_unloaded_state_has_no_protocol_lineage_in_case_records() -> None:
    _, _, case_unloaded = _build_unloaded_state()

    protocol_prefixed = [
        r for r in case_unloaded.records
        if r.case_id.startswith("protocol:")
    ]
    assert protocol_prefixed == []


# ---------------------------------------------------------------------------
# Aggregate ablation: every artifact family flipped together
# ---------------------------------------------------------------------------


def test_loaded_vs_unloaded_aggregate_ablation() -> None:
    """Aggregate sanity: loaded state has protocol lineage in
    every artifact family that ``compile_protocol_to_application_artifacts``
    populates; unloaded state has none. This pins the contract
    that loading a protocol is the SOLE entry point for
    ``protocol:`` lineage into application owners.
    """

    rare_loaded, knowledge_loaded, case_loaded = _build_loaded_state()
    rare_un, knowledge_un, case_un = _build_unloaded_state()

    boundary_ids_loaded = [h.hint_id for h in rare_loaded.boundary_prior_hints]
    playbook_ids_loaded = [r.rule_id for r in rare_loaded.distilled_playbook_rules]
    knowledge_ids_loaded = [r.record_id for r in knowledge_loaded.records]
    case_ids_loaded = [r.case_id for r in case_loaded.records]

    assert any(
        i.startswith("protocol:growth_advisor:cheng-laoshi:boundary:")
        for i in boundary_ids_loaded
    ), boundary_ids_loaded
    assert any(
        i.startswith("protocol:growth_advisor:cheng-laoshi:playbook:")
        for i in playbook_ids_loaded
    ), playbook_ids_loaded
    assert any(
        i.startswith("protocol:growth_advisor:cheng-laoshi:knowledge:")
        for i in knowledge_ids_loaded
    ), knowledge_ids_loaded
    assert any(
        i.startswith("protocol:growth_advisor:cheng-laoshi:case:")
        for i in case_ids_loaded
    ), case_ids_loaded

    boundary_ids_un = [h.hint_id for h in rare_un.boundary_prior_hints]
    playbook_ids_un = [r.rule_id for r in rare_un.distilled_playbook_rules]
    knowledge_ids_un = [r.record_id for r in knowledge_un.records]
    case_ids_un = [r.case_id for r in case_un.records]

    for label, ids in [
        ("boundary", boundary_ids_un),
        ("playbook", playbook_ids_un),
        ("knowledge", knowledge_ids_un),
        ("case", case_ids_un),
    ]:
        assert all(not i.startswith("protocol:") for i in ids), (
            f"unloaded state contains {label} lineage; got {ids}"
        )
