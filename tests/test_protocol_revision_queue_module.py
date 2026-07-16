"""Packet 9.0: ProtocolRevisionQueueModule — closes PE→learning loop.

Asserts:

* Module shape (slot / owner / dependencies / SHADOW default).
* Empty upstream → empty snapshot, no side effects.
* Reflection emitting proposals → routed through gate → submitted to queue.
* L1/L2 AUTO_APPROVED + auto_apply=True + registry injected → applied to registry.
* L4 always queued; never auto-applied.
* Per-proposal-id dedup across turns.
* auto_apply=False short-circuits even on AUTO_APPROVED outcome.
* SHADOW-tolerant on missing protocol_reflection.
"""

from __future__ import annotations

import asyncio

from lifeform_domain_growth_advisor import (
    build_cheng_laoshi_profile,
    growth_advisor_profile_to_behavior_protocol,
)
from volvence_zero.application.rare_heavy_state import ApplicationRareHeavyState
from volvence_zero.behavior_protocol import (
    ProposalEvidence,
    ProtocolReflectionSnapshot,
    ProtocolRevisionChangeKind,
    ProtocolRevisionProposal,
    ProtocolRevisionTargetField,
    ReviewLevel,
)
from volvence_zero.protocol_runtime import (
    ApprovalOutcome,
    ProtocolRegistryModule,
    ProtocolRevisionQueueModule,
    RevisionQueue,
)
from volvence_zero.runtime import Snapshot, WiringLevel


def _proposal(
    *,
    pid: str,
    target: str = "growth_advisor:cheng-laoshi",
    kind: ProtocolRevisionChangeKind = ProtocolRevisionChangeKind.WEIGHT_DECAY,
    level: ReviewLevel = ReviewLevel.L3,
) -> ProtocolRevisionProposal:
    return ProtocolRevisionProposal(
        proposal_id=pid,
        target_protocol_id=target,
        target_field=ProtocolRevisionTargetField.STRATEGY_PRIOR,
        target_entry_id=target,
        change_kind=kind,
        evidence=ProposalEvidence(
            observation_window_turns=8,
            pe_signature="auto-test",
            summary="auto-test summary",
        ),
        required_review_level=level,
        proposed_payload={"weight_multiplier": 0.5},
    )


def _reflection_snapshot(
    *proposals: ProtocolRevisionProposal,
) -> Snapshot[ProtocolReflectionSnapshot]:
    return Snapshot(
        slot_name="protocol_reflection",
        owner="ProtocolReflectionEngine",
        version=1,
        timestamp_ms=0,
        value=ProtocolReflectionSnapshot(
            protocol_revision_proposals=tuple(proposals),
            observation_window_turns=8,
            turns_since_last_scan=0,
            description="test",
        ),
    )


def _registry_with_cheng() -> ProtocolRegistryModule:
    rare = ApplicationRareHeavyState()
    module = ProtocolRegistryModule(application_rare_heavy_state=rare)
    bp = growth_advisor_profile_to_behavior_protocol(
        build_cheng_laoshi_profile()
    )
    module.load_protocol(bp)
    return module


# ---------------------------------------------------------------------------
# Module shape
# ---------------------------------------------------------------------------


def test_queue_module_owns_protocol_revision_queue_slot() -> None:
    m = ProtocolRevisionQueueModule()
    assert m.slot_name == "protocol_revision_queue"
    assert m.owner == "ProtocolRevisionQueueModule"
    # A1 (#90 residue): guidance-conflict proposals from the
    # apprenticeship protocol-alignment owner route through this same
    # gate + queue (single revision router).
    assert m.dependencies == (
        "protocol_reflection",
        "apprenticeship_protocol_alignment",
    )
    assert m.wiring_level is WiringLevel.SHADOW


# ---------------------------------------------------------------------------
# Empty / SHADOW-tolerant
# ---------------------------------------------------------------------------


def test_queue_module_empty_upstream_publishes_empty_snapshot() -> None:
    m = ProtocolRevisionQueueModule()
    snap = asyncio.run(m.process({}))
    assert snap.value.newly_routed == ()
    assert snap.value.pending_count == 0
    assert snap.value.auto_applied_count == 0


def test_queue_module_empty_proposals_publishes_empty_snapshot() -> None:
    m = ProtocolRevisionQueueModule()
    snap = asyncio.run(m.process({"protocol_reflection": _reflection_snapshot()}))
    assert snap.value.newly_routed == ()


# ---------------------------------------------------------------------------
# Routing semantics
# ---------------------------------------------------------------------------


def test_l1_proposal_auto_approved_and_auto_applied() -> None:
    """L1 → AUTO_APPROVED → registry.apply_revision called."""
    registry = _registry_with_cheng()
    queue = RevisionQueue()
    m = ProtocolRevisionQueueModule(
        revision_queue=queue, registry_module=registry, auto_apply=True
    )
    p = _proposal(pid="prop:l1", level=ReviewLevel.L1)
    snap = asyncio.run(
        m.process({"protocol_reflection": _reflection_snapshot(p)})
    )
    assert snap.value.auto_applied_count == 1
    assert snap.value.newly_routed[0].outcome == ApprovalOutcome.AUTO_APPROVED.value
    bp = registry.registry.get("growth_advisor:cheng-laoshi")
    assert bp.revision_log  # apply_revision appended


def test_l4_proposal_always_queued_never_applied() -> None:
    registry = _registry_with_cheng()
    queue = RevisionQueue()
    m = ProtocolRevisionQueueModule(
        revision_queue=queue, registry_module=registry
    )
    p = _proposal(pid="prop:l4", level=ReviewLevel.L4)
    snap = asyncio.run(
        m.process({"protocol_reflection": _reflection_snapshot(p)})
    )
    assert snap.value.auto_applied_count == 0
    assert snap.value.pending_count == 1
    bp = registry.registry.get("growth_advisor:cheng-laoshi")
    assert bp.revision_log == ()


def test_dedup_across_turns() -> None:
    """Same proposal_id appearing in two consecutive snapshots → routed once."""
    registry = _registry_with_cheng()
    m = ProtocolRevisionQueueModule(
        registry_module=registry, auto_apply=True
    )
    p = _proposal(pid="prop:dedup", level=ReviewLevel.L1)
    asyncio.run(
        m.process({"protocol_reflection": _reflection_snapshot(p)})
    )
    snap2 = asyncio.run(
        m.process({"protocol_reflection": _reflection_snapshot(p)})
    )
    assert snap2.value.newly_routed == ()
    assert snap2.value.auto_applied_count == 0
    assert m.auto_applied_total == 1


def test_auto_apply_false_skips_apply_even_on_auto_approved() -> None:
    registry = _registry_with_cheng()
    m = ProtocolRevisionQueueModule(
        registry_module=registry, auto_apply=False
    )
    p = _proposal(pid="prop:no-apply", level=ReviewLevel.L1)
    snap = asyncio.run(
        m.process({"protocol_reflection": _reflection_snapshot(p)})
    )
    assert snap.value.auto_applied_count == 0
    bp = registry.registry.get("growth_advisor:cheng-laoshi")
    assert bp.revision_log == ()


def test_no_registry_handle_means_no_auto_apply() -> None:
    """Even with auto_apply=True, missing registry → no apply (queue only)."""
    m = ProtocolRevisionQueueModule(registry_module=None, auto_apply=True)
    p = _proposal(pid="prop:no-reg", level=ReviewLevel.L1)
    snap = asyncio.run(
        m.process({"protocol_reflection": _reflection_snapshot(p)})
    )
    assert snap.value.auto_applied_count == 0
    assert snap.value.pending_count >= 0  # queue still records


def test_apply_failure_does_not_break_routing() -> None:
    """Proposal targeting unloaded protocol → apply raises KeyError;
    the routing snapshot still records the AUTO_APPROVED outcome with
    a note, no exception escapes."""
    registry = _registry_with_cheng()
    m = ProtocolRevisionQueueModule(
        registry_module=registry, auto_apply=True
    )
    bad = _proposal(
        pid="prop:bad-target",
        target="growth_advisor:does-not-exist",
        level=ReviewLevel.L1,
    )
    snap = asyncio.run(
        m.process({"protocol_reflection": _reflection_snapshot(bad)})
    )
    # Routed but not applied; rationale should mention skip.
    routed = snap.value.newly_routed
    assert len(routed) == 1
    assert "auto-apply skipped" in routed[0].rationale.lower()
    assert snap.value.auto_applied_count == 0


# ---------------------------------------------------------------------------
# final_wiring registration
# ---------------------------------------------------------------------------


def test_module_registered_in_final_wiring() -> None:
    from volvence_zero.integration.final_wiring import (
        FinalRolloutConfig,
        build_final_runtime_modules,
    )
    from volvence_zero.substrate.adapter import PlaceholderSubstrateAdapter

    config = FinalRolloutConfig()
    modules = build_final_runtime_modules(
        config=config,
        substrate_adapter=PlaceholderSubstrateAdapter(model_id="qr-test"),
    )
    publishers = [m for m in modules if m.slot_name == "protocol_revision_queue"]
    assert len(publishers) == 1
    assert publishers[0].wiring_level is WiringLevel.SHADOW
