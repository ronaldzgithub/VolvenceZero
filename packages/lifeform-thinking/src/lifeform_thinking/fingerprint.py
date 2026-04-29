"""Fingerprint computation for ``ThinkingTask`` staleness guard.

A fingerprint is a stable string digest of the ``(slot_name, version,
value-hash)`` triples for a *declared set* of upstream snapshots. The
scheduler computes it when a task is queued; the owner recomputes it
when it goes to apply an artifact. Mismatch => the task is ``STALE``
and its payload must not be applied.

Two hard invariants:

1. **Declared fingerprint scope.** Every worker declares which slots
   it depends on (``FingerprintScope``). The scheduler never sees
   "all snapshots"; wildcard scopes are explicitly forbidden. Each
   worker's declared scope is part of its registration and a
   contract test can grep for unauthorised expansions.
2. **Stable ordering.** The hash is over a JSON-serialised
   dictionary with *sorted* slot names. Python dict iteration order
   is deterministic since 3.7 but SORTING guarantees the digest is
   identical across processes / versions of the same snapshots, so
   a scheduler in process A and an owner in process B (e.g. future
   multi-tenant service) produce identical fingerprints.

Fingerprints are strings, not bytes, so they round-trip through JSON
logging and through the stdlib ``dataclasses.asdict`` used by debug
dumps without needing extra encoding.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FingerprintScope:
    """Declared set of upstream slots a worker reads.

    ``slot_names`` must be a non-empty, deterministic tuple. The
    ordering in the tuple is preserved for readability but the
    fingerprint itself sorts internally so consumers can declare the
    scope in "natural" order.

    Why a dataclass rather than a raw tuple: we want scope objects to
    be loggable + diffable + surfaceable as an audit trail in
    ``ThinkingTask``'s construction context. Future expansion (e.g.
    adding a ``value_path_selector`` for field-level fingerprinting)
    will benefit from being inside a named dataclass.
    """

    slot_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.slot_names:
            raise ValueError(
                "FingerprintScope.slot_names must be non-empty; a worker "
                "declaring no upstream scope cannot detect staleness."
            )
        if len(set(self.slot_names)) != len(self.slot_names):
            raise ValueError(
                f"FingerprintScope.slot_names must be unique, "
                f"got {self.slot_names!r}"
            )

    @property
    def sorted_slot_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.slot_names))


def _snapshot_digest_payload(snapshot: Any) -> dict[str, Any]:
    """Extract the three stable fields from a kernel ``Snapshot``.

    Uses ``getattr`` with explicit defaults so the function works on
    anything shaped like a snapshot, including the test stubs used by
    scheduler unit tests. Nothing here inspects internal owner state.
    """
    return {
        "slot_name": str(getattr(snapshot, "slot_name", "")),
        "version": int(getattr(snapshot, "version", 0)),
        # Value hash: we hash the repr so mutable payloads can't sneak
        # through. ``repr`` on a frozen dataclass is stable; snapshot
        # values are frozen by contract (see
        # docs/specs/contract-runtime.md).
        "value_repr_hash": hashlib.sha256(
            repr(getattr(snapshot, "value", None)).encode("utf-8")
        ).hexdigest(),
    }


def compute_fingerprint(
    *,
    snapshots: Mapping[str, Any],
    scope: FingerprintScope,
) -> str:
    """Compute a SHA256 fingerprint over the declared-scope slots.

    ``snapshots`` must contain every slot named in ``scope.slot_names``.
    Missing slots fail loudly (``KeyError`` bubbled from the lookup);
    the scheduler relies on this to detect misconfiguration before a
    task starts running.

    The digest is stable: sorting slot names + JSON-serialising with
    ``sort_keys=True`` means the same upstream state always yields
    the same fingerprint, independent of Python dict insertion order
    or scope declaration order.
    """
    ordered_slots = scope.sorted_slot_names
    # Fail-loudly key lookup: a missing slot is a bug in the caller,
    # not a silent staleness signal.
    payload = {
        slot: _snapshot_digest_payload(snapshots[slot]) for slot in ordered_slots
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def fingerprints_match(*, task_fingerprint: str, current_fingerprint: str) -> bool:
    """Return True iff the two fingerprints are byte-for-byte equal.

    Wrapping the equality check in a named function makes the apply
    path self-documenting and gives contract tests a single choke
    point to grep for owner-side fingerprint guards.
    """
    return (
        bool(task_fingerprint)
        and bool(current_fingerprint)
        and task_fingerprint == current_fingerprint
    )


__all__ = [
    "FingerprintScope",
    "compute_fingerprint",
    "fingerprints_match",
]
