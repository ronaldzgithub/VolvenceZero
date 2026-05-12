"""Packet D (long-horizon-closure) — cross-session owner hydration contract.

This module ships in ``vz-contracts`` because the protocol is cross-wheel
shared: ``SemanticStateStore`` lives in ``vz-cognition``, ``FollowupManager`` /
``VitalsModule`` live in ``lifeform-core``, and the orchestrator that
binds them lives in ``vz-runtime``. None of these can take a hard
dependency on each other, so the protocol type lives at the contract
foundation.

See ``docs/specs/owner-hydration.md`` for the full design (invariants,
storage layout, migration path, rollback). The summary is:

- Each owner that wants cross-session continuity implements
  ``HydratableOwnerProtocol`` (``export_persistence_snapshot`` /
  ``hydrate_from_persistence``).
- The runtime orchestrator (``OwnerHydrationStore`` in vz-runtime)
  reads / writes those snapshots through the same
  ``PersistenceBackend`` the ``MemoryStore`` already uses, with key
  prefix ``owner_hydration/<owner_name>``.
- Hydration failures MUST be typed exceptions, never silent fallbacks
  (``no-swallow-errors-no-hasattr-abuse`` rule).
- Rollback is by ``BrainConfig.owner_hydration_wiring`` flipping back
  to ``DISABLED``; old call sites and stored payloads stay untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class OwnerPersistenceSnapshot:
    """Versioned, owner-published state usable for cross-session hydration.

    Attributes:
        owner_name: Stable owner identifier. The hydration store uses
            this as part of the persistence key (``owner_hydration/<name>``).
            Receiving owner MUST verify the name matches its own slot
            before applying the payload (``HydrationOwnerMismatchError``).
        schema_version: OWNER-internal schema version. Bumping it
            requires either a forward-compatible migration in the
            owner's ``hydrate_from_persistence`` or a fail-loud
            ``HydrationVersionMismatchError`` so the operator can
            decide whether to start from scratch.
        payload: JSON-serialisable mapping. Owner decides shape;
            consumers MUST NOT inspect it (this is owner-internal
            detail crossing a serialisation boundary).
        description: Human-readable line for logs / debugging.
            Never read programmatically.
    """

    owner_name: str
    schema_version: int
    payload: Mapping[str, Any]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.owner_name or not self.owner_name.strip():
            raise ValueError(
                f"OwnerPersistenceSnapshot.owner_name must be non-empty"
            )
        if self.schema_version < 1:
            raise ValueError(
                f"OwnerPersistenceSnapshot.schema_version must be >= 1; "
                f"got {self.schema_version!r} for owner_name="
                f"{self.owner_name!r}"
            )
        if not isinstance(self.payload, Mapping):
            raise TypeError(
                f"OwnerPersistenceSnapshot.payload must be a Mapping; "
                f"got {type(self.payload).__name__} for owner_name="
                f"{self.owner_name!r}"
            )


class HydratableOwnerProtocol(Protocol):
    """Optional owner ability: dump + restore for cross-session continuity.

    Owners that do NOT implement this protocol simply do not
    participate in owner hydration; the kernel does not require it.

    Implementations MUST satisfy:

    - ``hydrate(export()) == export()`` (round-trip stability), per the
      ``test_owner_hydration_protocol.py`` contract test.
    - ``export_persistence_snapshot()`` is read-only on owner state
      (no side effects, no version bump).
    - ``hydrate_from_persistence(...)`` is idempotent: applying the
      same snapshot twice yields the same internal state.
    - On any structurally invalid input it raises a typed
      ``HydrationError`` subclass (never bare exceptions / silent
      fallbacks).
    """

    def export_persistence_snapshot(self) -> OwnerPersistenceSnapshot:
        ...

    def hydrate_from_persistence(
        self, snapshot: OwnerPersistenceSnapshot
    ) -> None:
        ...


class HydrationError(Exception):
    """Base class for cross-session owner hydration failures.

    Catch this when the caller wants to handle "any hydration error";
    catch the more specific subclasses when behavior should differ
    by failure mode.
    """


class HydrationVersionMismatchError(HydrationError):
    """``schema_version`` is unknown to the receiving owner.

    Typical fix: either bump the owner's understanding of the version
    (write a migration), or delete the stale persistence key so the
    next session starts clean.
    """


class HydrationPayloadInvalidError(HydrationError):
    """Snapshot payload is structurally invalid (missing required keys,
    wrong types, etc).

    Distinct from ``HydrationVersionMismatchError`` because the
    operator response differs: an invalid payload at a known version
    indicates a bug in the producer (or backend corruption), not an
    upgrade situation.
    """


class HydrationOwnerMismatchError(HydrationError):
    """``snapshot.owner_name`` does not match the receiving owner's
    own slot identifier.

    Indicates a wiring bug — the hydration store routed a snapshot
    to the wrong owner. Always fail loud.
    """


__all__ = [
    "HydratableOwnerProtocol",
    "HydrationError",
    "HydrationOwnerMismatchError",
    "HydrationPayloadInvalidError",
    "HydrationVersionMismatchError",
    "OwnerPersistenceSnapshot",
]
