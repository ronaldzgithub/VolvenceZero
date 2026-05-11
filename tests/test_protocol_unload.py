"""Packet 6.9: full unload_protocol with prefix-based store cleanup."""

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


def _build_full_module():
    rare = ApplicationRareHeavyState()
    knowledge = ApplicationDomainKnowledgeStore()
    case_memory = ApplicationCaseMemoryStore()
    module = ProtocolRegistryModule(
        application_rare_heavy_state=rare,
        domain_knowledge_store=knowledge,
        case_memory_store=case_memory,
    )
    return module, rare, knowledge, case_memory


def test_unload_removes_protocol_boundary_hints() -> None:
    module, rare, _, _ = _build_full_module()
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    module.load_protocol(bp)
    pre = len(
        [h for h in rare.boundary_prior_hints if bp.protocol_id in h.hint_id]
    )
    assert pre > 0

    assert module.unload_protocol(bp.protocol_id) is True
    post = len(
        [h for h in rare.boundary_prior_hints if bp.protocol_id in h.hint_id]
    )
    assert post == 0


def test_unload_removes_protocol_playbook_rules() -> None:
    module, rare, _, _ = _build_full_module()
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    module.load_protocol(bp)
    pre = len(
        [r for r in rare.distilled_playbook_rules if bp.protocol_id in r.rule_id]
    )
    assert pre > 0
    module.unload_protocol(bp.protocol_id)
    post = len(
        [r for r in rare.distilled_playbook_rules if bp.protocol_id in r.rule_id]
    )
    assert post == 0


def test_unload_removes_protocol_knowledge_records() -> None:
    module, _, knowledge, _ = _build_full_module()
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    module.load_protocol(bp)
    pre = len(
        [r for r in knowledge.records if bp.protocol_id in r.record_id]
    )
    assert pre > 0
    module.unload_protocol(bp.protocol_id)
    post = len(
        [r for r in knowledge.records if bp.protocol_id in r.record_id]
    )
    assert post == 0


def test_unload_removes_protocol_case_records() -> None:
    module, _, _, case_memory = _build_full_module()
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    module.load_protocol(bp)
    pre = len(
        [r for r in case_memory.records if bp.protocol_id in r.case_id]
    )
    assert pre > 0
    module.unload_protocol(bp.protocol_id)
    post = len(
        [r for r in case_memory.records if bp.protocol_id in r.case_id]
    )
    assert post == 0


def test_unload_removes_registry_entry() -> None:
    module, _, _, _ = _build_full_module()
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    module.load_protocol(bp)
    assert module.registry.get(bp.protocol_id) is not None
    module.unload_protocol(bp.protocol_id)
    assert module.registry.get(bp.protocol_id) is None


def test_unload_unloaded_returns_false() -> None:
    module, _, _, _ = _build_full_module()
    assert module.unload_protocol("growth_advisor:not-loaded") is False
