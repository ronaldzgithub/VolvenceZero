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

from threading import RLock

from volvence_zero.behavior_protocol import (
    BehaviorProtocol,
    ReviewStatus,
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
            from dataclasses import replace as _replace

            self._loaded[protocol_id] = _replace(
                existing, review_status=status
            )

    def get(self, protocol_id: str) -> BehaviorProtocol | None:
        with self._lock:
            return self._loaded.get(protocol_id)

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


__all__ = ["ProtocolRegistry"]
