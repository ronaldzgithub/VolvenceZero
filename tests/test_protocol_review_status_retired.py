"""Packet 6.4: ReviewStatus.RETIRED + lifecycle state machine tests."""

from __future__ import annotations

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
    ReviewLevel,
    ReviewStatus,
)
from volvence_zero.protocol_runtime import ProtocolRegistry, ProtocolRegistryModule


def test_review_status_enum_includes_retired() -> None:
    assert ReviewStatus.RETIRED.value == "retired"


def test_loaded_filters_out_retired_protocols() -> None:
    """``ProtocolRegistry.loaded()`` skips RETIRED entries; ``loaded_all()`` keeps them."""
    registry = ProtocolRegistry()
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    registry.load(bp)

    # Mark RETIRED via mark_status (lifecycle API).
    registry.mark_status(bp.protocol_id, ReviewStatus.RETIRED)

    assert registry.loaded() == ()
    all_loaded = registry.loaded_all()
    assert len(all_loaded) == 1
    assert all_loaded[0].review_status is ReviewStatus.RETIRED


def test_retired_protocol_excluded_from_active_mixture() -> None:
    """ProtocolRegistryModule.process skips RETIRED in active_mixture."""
    import asyncio

    rare = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=rare)
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    module.load_protocol(bp)
    module.registry.mark_status(bp.protocol_id, ReviewStatus.RETIRED)

    snapshot = asyncio.run(module.process({}))
    assert snapshot.value.active_protocols == ()
    assert snapshot.value.boundary_union_ids == ()


def test_protocol_retirement_change_kind_marks_status() -> None:
    """PROTOCOL_RETIREMENT proposal applied → review_status=RETIRED."""
    rare = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=rare)
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    module.load_protocol(bp)

    proposal = ProtocolRevisionProposal(
        proposal_id="prop:retire-via-revision",
        target_protocol_id=bp.protocol_id,
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id=bp.protocol_id,
        change_kind=ProtocolRevisionChangeKind.PROTOCOL_RETIREMENT,
        evidence=ProposalEvidence(
            observation_window_turns=15,
            pe_signature="sustained_failure",
            summary="protocol retired via reflection",
        ),
        required_review_level=ReviewLevel.L4,
    )
    module.apply_revision(proposal)

    bp_after = module.registry.get(bp.protocol_id)
    assert bp_after.review_status is ReviewStatus.RETIRED
    # And it's filtered from loaded() now.
    assert bp.protocol_id not in {p.protocol_id for p in module.registry.loaded()}


def test_mark_status_round_trip_through_lifecycle() -> None:
    """DRAFT → SHADOW → ACTIVE → RETIRED transitions all work."""
    registry = ProtocolRegistry()
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    registry.load(bp)

    for status in (ReviewStatus.SHADOW, ReviewStatus.ACTIVE, ReviewStatus.RETIRED):
        registry.mark_status(bp.protocol_id, status)
        assert registry.get(bp.protocol_id).review_status is status
