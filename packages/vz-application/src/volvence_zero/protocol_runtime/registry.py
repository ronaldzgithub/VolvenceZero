"""In-memory store for loaded BehaviorProtocols (packet 1.0).

Per-module store backing ``ProtocolRegistryModule``. Does NOT
publish snapshots itself; the module reads ``loaded()`` each turn and
hands the tuple to ``compute_active_mixture``.

Lifecycle ops are synchronous mutations on the registry (``load`` /
``unload`` / ``mark_status``). Adapters in ``lifeform-domain-*``
wheels (and future ``lifeform-protocol-runtime``) call ``load(...)``
to register protocols; nothing else mutates the registry.

Packet 1.0 keeps this minimal:

* No revision_log mutation API (``ProtocolRevision`` records are
  declared in vz-contracts but no PE-driven writeback exists yet).
* No persistence; in-memory only — module rebuild = empty registry.
* No cross-session sharing; one ``ProtocolRegistryModule`` instance
  per lifeform.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import replace as _replace
from threading import RLock

from volvence_zero.behavior_protocol import (
    BehaviorProtocol,
    ProtocolRevision,
    ProtocolRevisionChangeKind,
    ProtocolRevisionProposal,
    ProtocolRevisionTargetField,
    ReviewStatus,
    StrategyPriorRevision,
)


class ProtocolRegistry:
    """Mutable, threadsafe in-memory registry of loaded BehaviorProtocols.

    Holds the canonical protocol identity → ``BehaviorProtocol``
    mapping. Snapshot publication and activation weighting are NOT
    its concern; ``ProtocolRegistryModule.process`` reads
    ``loaded()`` and constructs the snapshot via
    ``compute_active_mixture``.

    Threadsafety: an ``RLock`` guards the dict so that adapter calls
    (``load_protocol`` from session-side code) cannot race with the
    kernel ``process()`` call. Reads return tuples (immutable
    snapshots of the dict at read-time) so consumers don't see
    in-flight mutations.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._loaded: dict[str, BehaviorProtocol] = {}

    def load(self, protocol: BehaviorProtocol) -> None:
        """Register a protocol. Replaces an existing entry of the same id.

        Replacement is idempotent for FixtureUptake (same protocol
        loaded twice = no-op semantically). Future packets will
        gate replacement behind ``ModificationGate`` + version
        bookkeeping; packet 1.0 trusts the caller.
        """

        if not isinstance(protocol, BehaviorProtocol):
            raise TypeError(
                f"ProtocolRegistry.load expects BehaviorProtocol, got "
                f"{type(protocol).__name__}"
            )
        with self._lock:
            self._loaded[protocol.protocol_id] = protocol

    def unload(self, protocol_id: str) -> bool:
        """Remove a protocol; return True if it was present."""

        with self._lock:
            return self._loaded.pop(protocol_id, None) is not None

    def mark_status(self, protocol_id: str, status: ReviewStatus) -> None:
        """Replace a loaded protocol with one carrying a new review status.

        ``BehaviorProtocol`` is frozen; transitioning lifecycle state
        means swapping the dataclass instance. Packet 1.0 callers are
        expected to use this directly; later packets will route through
        ``ModificationGate``.
        """

        with self._lock:
            existing = self._loaded.get(protocol_id)
            if existing is None:
                raise KeyError(
                    f"ProtocolRegistry: no loaded protocol with id "
                    f"{protocol_id!r}"
                )
            self._loaded[protocol_id] = _replace(
                existing, review_status=status
            )

    def get(self, protocol_id: str) -> BehaviorProtocol | None:
        with self._lock:
            return self._loaded.get(protocol_id)

    def apply_revision(
        self,
        proposal: ProtocolRevisionProposal,
        *,
        revised_by: str = "ProtocolReflectionEngine",
        revised_at_tick: int = 0,
    ) -> BehaviorProtocol:
        """Packet 3.3: apply a reviewed revision proposal.

        Mutation rules (per change_kind):

        * ``WEIGHT_DECAY`` on ``STRATEGY_PRIOR``: multiply each
          ``initial_weight`` by ``proposed_payload['weight_multiplier']``
          (defaults to 0.5). When ``target_entry_id`` matches a
          specific ``rule_id``, only that strategy decays; when
          it matches the ``protocol_id`` (the "protocol-granular"
          path used by :func:`propose_strategy_decay`), all
          strategies decay.
        * ``DEACTIVATE`` on ``STRATEGY_PRIOR``: set
          ``initial_weight=0`` for the matched strategies.
        * ``ARCHIVE`` on ``KNOWLEDGE_SEED`` / ``SIGNATURE_CASE``:
          remove the targeted entry from the protocol's
          collection.
        * Any other combination raises NotImplementedError —
          keeps the surface narrow until a specific rule needs
          it (failing loud beats silent no-op).

        Side effects:

        * Appends a :class:`ProtocolRevision` entry to
          ``revision_log``.
        * Replaces the registry slot for the protocol with the
          new (frozen) ``BehaviorProtocol`` instance.

        Returns the new protocol instance. Caller is expected to
        re-run the compile path against application owners
        (handled at the owner level for clean injection).
        """

        with self._lock:
            existing = self._loaded.get(proposal.target_protocol_id)
            if existing is None:
                raise KeyError(
                    f"ProtocolRegistry.apply_revision: no loaded "
                    f"protocol with id {proposal.target_protocol_id!r}"
                )

            mutated = _apply_change(existing, proposal)

            revision = ProtocolRevision(
                revision_id=(
                    f"revision:{proposal.proposal_id}:"
                    f"{_now_iso()}"
                ),
                revised_at_tick=revised_at_tick,
                revised_by=revised_by,
                description=(
                    f"{proposal.change_kind.value} on "
                    f"{proposal.target_field.value} entry "
                    f"{proposal.target_entry_id!r}: "
                    f"{proposal.evidence.summary}"
                ),
                affected_field=proposal.target_field.value,
            )
            mutated = _replace(
                mutated,
                revision_log=mutated.revision_log + (revision,),
            )
            self._loaded[proposal.target_protocol_id] = mutated
            return mutated

    def checkout_revision(
        self,
        protocol_id: str,
        revision_id: str | None = None,
    ) -> BehaviorProtocol:
        """R15 rollback: restore a protocol from its revision history.

        ``revision_id=None`` rolls back to before any revisions
        (the original protocol shape); a specific ``revision_id``
        rolls back to the state that existed *immediately after*
        that revision was applied.

        This is a forward-looking API: a full implementation
        replays revisions from a checkpoint snapshot. Packet 3.3
        ships a minimal version: it can only undo the LAST
        revision (truncate ``revision_log[-1]``), which is enough
        for "oops, that proposal was bad, roll back" emergency
        flows. Full replay-from-checkpoint lands in a follow-up.
        """

        with self._lock:
            existing = self._loaded.get(protocol_id)
            if existing is None:
                raise KeyError(
                    f"ProtocolRegistry.checkout_revision: no loaded "
                    f"protocol with id {protocol_id!r}"
                )
            if not existing.revision_log:
                raise ValueError(
                    f"protocol {protocol_id!r} has no revisions to roll back"
                )
            if revision_id is not None:
                # Truncate revision_log down to the one identified.
                idx = next(
                    (
                        i for i, r in enumerate(existing.revision_log)
                        if r.revision_id == revision_id
                    ),
                    None,
                )
                if idx is None:
                    raise KeyError(
                        f"protocol {protocol_id!r} has no revision "
                        f"with id {revision_id!r}"
                    )
                if idx != len(existing.revision_log) - 1:
                    # Multi-step rollback would need replay-from-checkpoint;
                    # packet 3.3 only supports last-revision rollback.
                    raise NotImplementedError(
                        "multi-step rollback (non-last revision) is "
                        "deferred; current packet supports only the "
                        "most recent revision via revision_id=None"
                    )
            # Undo the last revision: truncate revision_log.
            # NOTE: this preserves the *content* of the latest
            # revision (the apply already mutated content
            # in place via dataclass replace; the revision_log
            # entry is only an audit record). A full content
            # rollback requires the apply step to also store the
            # pre-mutation content snapshot; this is captured
            # inside the revision entry's ``description`` only at
            # packet 3.3 — full content rollback is the deferred
            # follow-up. Loud-fail any caller relying on content
            # rollback today.
            raise NotImplementedError(
                "content rollback (replay-from-checkpoint) is "
                "deferred to a follow-up packet. checkout_revision "
                "validates the revision_id but cannot yet revert "
                "content changes."
            )

    def loaded(self) -> tuple[BehaviorProtocol, ...]:
        """Return all currently-loaded protocols as an immutable tuple.

        Ordering is by ``protocol_id`` so snapshots are
        deterministic across runs (important for stable
        ``revision_fingerprint``).
        """

        with self._lock:
            return tuple(
                self._loaded[pid] for pid in sorted(self._loaded)
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._loaded)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _apply_change(
    protocol: BehaviorProtocol,
    proposal: ProtocolRevisionProposal,
) -> BehaviorProtocol:
    """Pure mutation: produce a new BehaviorProtocol per the change kind."""

    field = proposal.target_field
    kind = proposal.change_kind
    target_entry_id = proposal.target_entry_id

    if field is ProtocolRevisionTargetField.STRATEGY_PRIOR:
        if kind in (
            ProtocolRevisionChangeKind.WEIGHT_DECAY,
            ProtocolRevisionChangeKind.DEACTIVATE,
        ):
            multiplier = (
                0.0
                if kind is ProtocolRevisionChangeKind.DEACTIVATE
                else (
                    float((proposal.proposed_payload or {}).get(
                        "weight_multiplier", 0.5
                    ))
                )
            )
            new_strategies = []
            for strategy in protocol.strategy_priors:
                if (
                    target_entry_id == protocol.protocol_id
                    or target_entry_id == strategy.rule_id
                ):
                    history_entry = StrategyPriorRevision(
                        revision_id=f"{proposal.proposal_id}:{strategy.rule_id}",
                        revised_at_tick=0,
                        delta=(multiplier - 1.0) * strategy.initial_weight,
                        reason=(
                            f"applied {kind.value} via reflection "
                            f"proposal {proposal.proposal_id}"
                        ),
                    )
                    new_strategies.append(
                        _replace(
                            strategy,
                            initial_weight=strategy.initial_weight * multiplier,
                            revision_history=strategy.revision_history
                            + (history_entry,),
                        )
                    )
                else:
                    new_strategies.append(strategy)
            return _replace(
                protocol, strategy_priors=tuple(new_strategies)
            )

    if field is ProtocolRevisionTargetField.KNOWLEDGE_SEED:
        if kind is ProtocolRevisionChangeKind.ARCHIVE:
            kept = tuple(
                seed
                for seed in protocol.knowledge_seeds
                if seed.seed_id != target_entry_id
            )
            return _replace(protocol, knowledge_seeds=kept)

    if field is ProtocolRevisionTargetField.SIGNATURE_CASE:
        if kind is ProtocolRevisionChangeKind.ARCHIVE:
            kept = tuple(
                case
                for case in protocol.signature_cases
                if case.case_id != target_entry_id
            )
            return _replace(protocol, signature_cases=kept)

    raise NotImplementedError(
        f"apply_revision: change_kind={kind.value!r} on "
        f"target_field={field.value!r} is not yet supported"
    )


__all__ = ["ProtocolRegistry"]
