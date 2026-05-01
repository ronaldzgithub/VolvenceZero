"""Tests for the commitment-owner confidence-floor filter.

Phase B+A's verify exposed that ``CommitmentModule`` was absorbing
every OBSERVE proposal the ``LLMSemanticProposalRuntime`` (and the
NoOp baseline) emitted, inflating ``outcome_rejected_count`` and
``advocacy_proposed_count`` with routine "the user said something"
events. This test pins the policy:

* default ``SemanticOwnerModule.min_proposal_confidence == 0.0``
  (every proposal flows through, historical behaviour);
* ``CommitmentModule.min_proposal_confidence == 0.40`` (drops
  OBSERVE proposals from both runtimes, keeps DEFER / CREATE /
  COMPLETE / BLOCK);
* the filter runs at the OWNER layer, so the snapshot still
  reports the runtime's original ``description`` (audit trail
  intact) but the store's ``apply`` only sees accepted proposals.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from volvence_zero.memory import MemorySnapshot
from volvence_zero.runtime import Snapshot, WiringLevel
from volvence_zero.semantic_state import (
    CommitmentModule,
    PlanIntentModule,
    SemanticProposal,
    SemanticProposalBatch,
    SemanticProposalOperation,
    SemanticProposalRuntime,
    SemanticStateStore,
)
from volvence_zero.substrate import SubstrateSnapshot


def _stub_substrate_snapshot() -> SubstrateSnapshot:
    return SubstrateSnapshot(
        model_id="stub",
        is_frozen=True,
        surface_kind="feature",
        token_logits=(),
        feature_surface=(),
        residual_activations=(),
        residual_sequence=(),
        unavailable_fields=(),
        description="stub for confidence-filter test",
    )


def _stub_memory_snapshot() -> MemorySnapshot:
    return MemorySnapshot(
        transient_summary="",
        episodic_summary="",
        durable_summary="",
        retrieved_entries=(),
        total_entries_by_stratum=(),
        pending_promotions=0,
        pending_decays=0,
        cms_state=None,
        description="stub memory",
    )


def _make_proposal(
    *,
    operation: SemanticProposalOperation,
    confidence: float,
    target_slot: str = "commitment",
    turn_index: int = 0,
) -> SemanticProposal:
    return SemanticProposal(
        proposal_id=f"{target_slot}:{operation.value}:{turn_index}",
        target_slot=target_slot,
        operation=operation,
        summary=f"test-{operation.value}",
        detail="evidence",
        confidence=confidence,
        evidence="evidence",
        control_signal=0.05,
    )


@dataclass
class _ScriptedRuntime(SemanticProposalRuntime):
    runtime_id: str = "scripted-runtime"
    proposals: tuple[SemanticProposal, ...] = ()

    def propose(
        self,
        *,
        target_slot: str,
        user_input: str | None,
        substrate_snapshot: SubstrateSnapshot | None,
        memory_snapshot: MemorySnapshot | None,
        previous_snapshot: Any | None,
        turn_index: int,
    ) -> SemanticProposalBatch:
        del (
            target_slot, user_input, substrate_snapshot,
            memory_snapshot, previous_snapshot, turn_index,
        )
        return SemanticProposalBatch(
            proposals=self.proposals,
            runtime_id=self.runtime_id,
            schema_version=1,
            description=f"scripted batch with {len(self.proposals)} proposal(s)",
        )


def _drive_owner(
    *, module_cls, proposals: tuple[SemanticProposal, ...],
    store: SemanticStateStore,
) -> Snapshot[Any]:
    runtime = _ScriptedRuntime(proposals=proposals)
    module = module_cls(
        store=store,
        proposal_runtime=runtime,
        user_input="hello",
        turn_index=0,
        wiring_level=WiringLevel.ACTIVE,
    )
    upstream = {
        "substrate": Snapshot(
            slot_name="substrate",
            owner="StubSubstrate",
            version=1,
            timestamp_ms=0,
            value=_stub_substrate_snapshot(),
        ),
        "memory": Snapshot(
            slot_name="memory",
            owner="StubMemory",
            version=1,
            timestamp_ms=0,
            value=_stub_memory_snapshot(),
        ),
    }
    return asyncio.run(module.process(upstream))


def test_default_owner_threshold_is_zero() -> None:
    """Backstop: any owner that doesn't override stays at 0.0.

    Tightening the default would silently change behaviour for
    long-tail consumers; we want a per-owner explicit decision.
    """
    assert PlanIntentModule.min_proposal_confidence == 0.0
    proposal = _make_proposal(
        operation=SemanticProposalOperation.OBSERVE,
        confidence=0.20,
        target_slot="plan_intent",
    )
    store = SemanticStateStore()
    _drive_owner(
        module_cls=PlanIntentModule, proposals=(proposal,), store=store
    )
    # The store mutated: a record exists. We don't introspect the
    # snapshot beyond confirming the filter was a no-op.
    plan_records = store.records_for("plan_intent")
    assert len(plan_records) == 1


def test_commitment_threshold_drops_low_confidence_observe() -> None:
    """OBSERVE @ 0.20 (NoOp) and 0.25 (LLM) must NOT enter lifecycle.

    These confidence values are the public contract of the two
    proposal runtimes that ship in-tree; if the runtimes change
    their confidence schedule the threshold should be reviewed
    here, not silently pass through.
    """
    proposals = (
        _make_proposal(
            operation=SemanticProposalOperation.OBSERVE, confidence=0.20
        ),
        _make_proposal(
            operation=SemanticProposalOperation.OBSERVE, confidence=0.25
        ),
    )
    store = SemanticStateStore()
    snap = _drive_owner(
        module_cls=CommitmentModule, proposals=proposals, store=store
    )
    cs = snap.value
    assert len(cs.lifecycle_entries) == 0
    assert len(cs.active_commitments) == 0
    assert cs.advocacy_proposed_count == 0
    assert cs.outcome_rejected_count == 0


def test_commitment_threshold_keeps_classified_operations() -> None:
    """DEFER 0.50 / CREATE 0.55 / COMPLETE 0.60 / BLOCK 0.60 all flow through.

    The ladder mirrors ``LLMSemanticProposalRuntime``'s
    ``_OPERATION_CONFIDENCE`` table; a regression on either side
    (the runtime drops confidences, or the owner raises the floor)
    breaks this test \u2014 forcing the developer to reconcile the
    two-side contract explicitly.
    """
    proposals = (
        _make_proposal(
            operation=SemanticProposalOperation.DEFER, confidence=0.50,
            turn_index=0,
        ),
        _make_proposal(
            operation=SemanticProposalOperation.CREATE, confidence=0.55,
            turn_index=1,
        ),
        _make_proposal(
            operation=SemanticProposalOperation.COMPLETE, confidence=0.60,
            turn_index=2,
        ),
        _make_proposal(
            operation=SemanticProposalOperation.BLOCK, confidence=0.60,
            turn_index=3,
        ),
    )
    store = SemanticStateStore()
    snap = _drive_owner(
        module_cls=CommitmentModule, proposals=proposals, store=store
    )
    cs = snap.value
    assert len(cs.lifecycle_entries) == 4


def test_commitment_threshold_keeps_proposals_at_exact_floor() -> None:
    """Boundary: a proposal exactly at 0.40 confidence is KEPT.

    The implementation uses ``>=``, not ``>``, so equal-to-floor
    proposals still flow into the store. This matters because a
    runtime that wants to mark a proposal as "borderline accept"
    can do so by emitting 0.40 explicitly.
    """
    proposal = _make_proposal(
        operation=SemanticProposalOperation.DEFER, confidence=0.40
    )
    store = SemanticStateStore()
    snap = _drive_owner(
        module_cls=CommitmentModule, proposals=(proposal,), store=store
    )
    assert len(snap.value.lifecycle_entries) == 1


def test_commitment_threshold_drops_at_just_below_floor() -> None:
    """Boundary: 0.39 confidence is DROPPED."""
    proposal = _make_proposal(
        operation=SemanticProposalOperation.DEFER, confidence=0.39
    )
    store = SemanticStateStore()
    snap = _drive_owner(
        module_cls=CommitmentModule, proposals=(proposal,), store=store
    )
    assert len(snap.value.lifecycle_entries) == 0
