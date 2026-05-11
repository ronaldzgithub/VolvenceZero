"""Packet 6.8: ProtocolRegistryIntrospectionModule + ProtocolRevisionLogModule tests."""

from __future__ import annotations

import asyncio
from dataclasses import replace as _replace

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState
from volvence_zero.behavior_protocol import (
    ProposalEvidence,
    ProtocolRevisionChangeKind,
    ProtocolRevisionProposal,
    ProtocolRevisionTargetField,
    ReviewStatus,
)
from volvence_zero.protocol_runtime import (
    ProtocolRegistry,
    ProtocolRegistryIntrospectionModule,
    ProtocolRegistryModule,
    ProtocolRevisionLogModule,
)
from volvence_zero.runtime import WiringLevel


def test_registry_introspection_empty() -> None:
    registry = ProtocolRegistry()
    module = ProtocolRegistryIntrospectionModule(registry=registry)
    snap = asyncio.run(module.process({}))
    assert snap.value.entries == ()
    assert snap.value.active_count == 0
    assert snap.value.retired_count == 0


def test_registry_introspection_summarizes_loaded() -> None:
    registry = ProtocolRegistry()
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    registry.load(bp)
    module = ProtocolRegistryIntrospectionModule(registry=registry)
    snap = asyncio.run(module.process({}))
    assert len(snap.value.entries) == 1
    e = snap.value.entries[0]
    assert e.protocol_id == bp.protocol_id
    assert e.boundary_count == len(bp.boundary_contracts)
    assert e.strategy_count == len(bp.strategy_priors)


def test_registry_introspection_counts_active_vs_retired() -> None:
    registry = ProtocolRegistry()
    bp_a = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    bp_b = _replace(
        bp_a,
        protocol_id="growth_advisor:retired-clone",
        review_status=ReviewStatus.RETIRED,
    )
    registry.load(bp_a)
    registry.load(bp_b)
    module = ProtocolRegistryIntrospectionModule(registry=registry)
    snap = asyncio.run(module.process({}))
    assert snap.value.active_count == 1
    assert snap.value.retired_count == 1


def test_revision_log_module_empty() -> None:
    registry = ProtocolRegistry()
    module = ProtocolRevisionLogModule(registry=registry)
    snap = asyncio.run(module.process({}))
    assert snap.value.entries == ()


def test_revision_log_module_lists_revisions() -> None:
    rare = ApplicationRareHeavyState()
    registry_module = ProtocolRegistryModule(
        application_rare_heavy_state=rare
    )
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    registry_module.load_protocol(bp)
    proposal = ProtocolRevisionProposal(
        proposal_id="prop:audit",
        target_protocol_id=bp.protocol_id,
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id=bp.protocol_id,
        change_kind=ProtocolRevisionChangeKind.WEIGHT_DECAY,
        evidence=ProposalEvidence(
            observation_window_turns=8,
            pe_signature="test",
            summary="audit test",
        ),
    )
    registry_module.apply_revision(proposal)

    log_module = ProtocolRevisionLogModule(
        registry=registry_module.registry
    )
    snap = asyncio.run(log_module.process({}))
    assert len(snap.value.entries) >= 1
    entry = snap.value.entries[0]
    assert entry.protocol_id == bp.protocol_id


def test_introspection_modules_in_final_wiring() -> None:
    from volvence_zero.integration.final_wiring import (
        FinalRolloutConfig,
        build_final_runtime_modules,
    )
    from volvence_zero.substrate.adapter import PlaceholderSubstrateAdapter

    config = FinalRolloutConfig()
    modules = build_final_runtime_modules(
        config=config,
        substrate_adapter=PlaceholderSubstrateAdapter(model_id="introspection"),
    )
    slots = {m.slot_name for m in modules}
    assert "protocol_registry" in slots
    assert "protocol_revision_log" in slots
    # Both should default SHADOW.
    for m in modules:
        if m.slot_name in ("protocol_registry", "protocol_revision_log"):
            assert m.wiring_level is WiringLevel.SHADOW
