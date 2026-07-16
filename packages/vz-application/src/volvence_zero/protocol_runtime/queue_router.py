"""Packet 9.0 — ProtocolRevisionQueueModule.

Closes the architectural learning loop:

1. ``ProtocolReflectionEngine`` publishes ``protocol_reflection``
   each background-slow tick with proposed revisions.
2. **This module** consumes that snapshot, dedupes proposals it
   has already routed, runs each new one through
   :func:`evaluate_protocol_revision` (the R10 ModificationGate),
   submits to a shared :class:`RevisionQueue`, AND — if the
   gate decision is ``AUTO_APPROVED`` — calls
   :meth:`ProtocolRegistryModule.apply_revision` so the change
   takes effect immediately.
3. ``L4`` / queued-for-human proposals stay in the queue for
   the operator CLI (packet 7.4) or external review service.

Without this module the ProtocolReflectionEngine output was a
dead-end snapshot — proposals were generated but never applied.
This is the single piece that makes the system "actually learn
from PE without human intervention" for safe ChangeKinds.

Owner placement: ``vz-application.protocol_runtime`` — reads
the cognition-side reflection snapshot but writes registry/queue
state owned by this wheel.

Wiring level: SHADOW by default. Production must opt-in via
``FinalRolloutConfig.level_for("protocol_revision_queue", ACTIVE)``.
The auto_apply branch only fires when the module is wired ACTIVE
*and* has a registry handle injected (so SHADOW dual-runs are
truly side-effect free).
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from volvence_zero.behavior_protocol import (
    ProtocolReflectionSnapshot,
    ProtocolRevisionProposal,
    ProtocolRevisionQueueEntry,
    ProtocolRevisionQueueSnapshot,
)
from volvence_zero.runtime import RuntimeModule, Snapshot, WiringLevel

from volvence_zero.application.types import (
    ApprenticeshipProtocolAlignmentSnapshot,
)

from volvence_zero.protocol_runtime.owner import ProtocolRegistryModule
from volvence_zero.protocol_runtime.revision_queue import (
    ApprovalOutcome,
    RevisionQueue,
    evaluate_protocol_revision,
)


class ProtocolRevisionQueueModule(
    RuntimeModule[ProtocolRevisionQueueSnapshot]
):
    """Routes reflection-proposed revisions through the gate +
    queue + (optionally) registry."""

    slot_name: ClassVar[str] = "protocol_revision_queue"
    owner: ClassVar[str] = "ProtocolRevisionQueueModule"
    value_type: ClassVar[type[Any]] = ProtocolRevisionQueueSnapshot
    # A1 (#90 residue): in addition to background-slow reflection
    # proposals, route guidance-conflict proposals published by the
    # apprenticeship protocol-alignment owner through the SAME gate +
    # queue + dedup path (single revision router, R8).
    dependencies: ClassVar[tuple[str, ...]] = (
        "protocol_reflection",
        "apprenticeship_protocol_alignment",
    )
    default_wiring_level: ClassVar[WiringLevel] = WiringLevel.SHADOW

    def __init__(
        self,
        *,
        wiring_level: WiringLevel | None = None,
        revision_queue: RevisionQueue | None = None,
        registry_module: ProtocolRegistryModule | None = None,
        auto_apply: bool = True,
    ) -> None:
        """Create the routing module.

        Args:
            wiring_level: SHADOW by default; ACTIVE when production
                opts in.
            revision_queue: shared queue. If None, an in-memory
                queue is allocated (visible via :attr:`queue`).
            registry_module: registry handle for auto-applying
                AUTO_APPROVED proposals. If None, auto-apply is
                disabled regardless of ``auto_apply``.
            auto_apply: if True (default) and a registry is injected,
                AUTO_APPROVED proposals are immediately applied.
                Set False to make the module pure-routing (queue
                only) for testing or staged rollouts.
        """

        super().__init__(wiring_level=wiring_level)
        self._queue = revision_queue if revision_queue is not None else RevisionQueue()
        self._registry_module = registry_module
        self._auto_apply = auto_apply
        # Per-proposal-id dedup: a proposal may keep appearing in
        # protocol_reflection's snapshot until it gets revised
        # / applied; we route each id only once.
        self._routed_proposal_ids: set[str] = set()
        self._auto_applied_total: int = 0

    @property
    def queue(self) -> RevisionQueue:
        return self._queue

    @property
    def auto_applied_total(self) -> int:
        """Cumulative count of AUTO_APPROVED proposals applied
        (across all turns this module has run)."""
        return self._auto_applied_total

    async def process(
        self, upstream: Mapping[str, Snapshot[Any]]
    ) -> Snapshot[ProtocolRevisionQueueSnapshot]:
        reflection = upstream.get("protocol_reflection")
        proposals: tuple[ProtocolRevisionProposal, ...] = ()
        if reflection is not None and isinstance(
            reflection.value, ProtocolReflectionSnapshot
        ):
            proposals = reflection.value.protocol_revision_proposals
        alignment = upstream.get("apprenticeship_protocol_alignment")
        if alignment is not None and isinstance(
            alignment.value, ApprenticeshipProtocolAlignmentSnapshot
        ):
            proposals = proposals + alignment.value.revision_proposals

        newly_routed: list[ProtocolRevisionQueueEntry] = []
        auto_applied_this_turn = 0

        for proposal in proposals:
            if proposal.proposal_id in self._routed_proposal_ids:
                continue
            decision = evaluate_protocol_revision(proposal)
            self._routed_proposal_ids.add(proposal.proposal_id)

            entry_rationale = decision.rationale

            if decision.outcome is ApprovalOutcome.QUEUED_FOR_HUMAN:
                # Only QUEUED_FOR_HUMAN decisions are accepted by
                # RevisionQueue.submit (queue is the human-review
                # backlog, not an audit log).
                self._queue.submit(proposal, decision)
            elif decision.outcome is ApprovalOutcome.AUTO_APPROVED:
                if (
                    self._auto_apply
                    and self._registry_module is not None
                ):
                    try:
                        self._registry_module.apply_revision(proposal)
                        auto_applied_this_turn += 1
                    except (
                        ValueError,
                        KeyError,
                        NotImplementedError,
                    ) as exc:
                        entry_rationale = (
                            f"auto-apply skipped after AUTO_APPROVED: "
                            f"{type(exc).__name__}: {exc}"
                        )
                else:
                    # auto_apply disabled or no registry — record
                    # outcome but no action taken.
                    pass
            # REJECTED: nothing to do beyond audit record below.

            newly_routed.append(
                ProtocolRevisionQueueEntry(
                    proposal_id=proposal.proposal_id,
                    target_protocol_id=proposal.target_protocol_id,
                    change_kind=proposal.change_kind.value,
                    outcome=decision.outcome.value,
                    rationale=entry_rationale,
                )
            )

        self._auto_applied_total += auto_applied_this_turn
        snapshot = ProtocolRevisionQueueSnapshot(
            newly_routed=tuple(newly_routed),
            pending_count=len(self._queue.list_pending()),
            auto_applied_count=auto_applied_this_turn,
            description=(
                f"protocol_revision_queue: {len(newly_routed)} routed, "
                f"{auto_applied_this_turn} auto-applied, "
                f"{len(self._queue.list_pending())} pending"
            ),
        )
        return self.publish(snapshot)


__all__ = ["ProtocolRevisionQueueModule"]
