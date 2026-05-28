"""AffordanceRegistry \u2014 startup atomic write, runtime O(1) read.

Why a separate class instead of a module-level dict:

* The registry is session-scoped (different verticals register
  different affordances). A module-level dict would leak state across
  tests and across multi-tenant service deployments.
* Runtime reads are O(1); startup writes go through
  ``register_all`` / ``register`` which validate uniqueness.
* ``seal()`` locks the registry so post-seal writes raise. Callers
  that want fully static registries call ``seal()`` after bootstrap
  and rely on the exception to catch accidental mid-turn mutation.

Design decisions that avoid traps:

* ``register`` takes an already-constructed ``AffordanceDescriptor``
  so all invariants have already been enforced by post_init; the
  registry just checks global (``name``) uniqueness.
* ``name`` is the only key we index on; the registry preserves
  registration order so the renderers produce deterministic output.
* ``by_kind`` / ``by_tag`` return tuples (not generators) so
  callers can iterate multiple times without surprises.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass

from volvence_zero.affordance import (
    AffordanceDescriptor,
    AffordanceKind,
)


_FAIL_CLOSED_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _env_fail_closed_default() -> bool:
    """Read the platform-wide tool-policy fail-closed default.

    SHADOW (default, env unset/false): a ``contract_id`` with no
    registered policy is allow-all (legacy back-compat).
    ACTIVE (``DLAAS_TOOL_POLICY_FAIL_CLOSED=true``): a ``contract_id``
    with no registered policy is deny-all — a contract that was
    supposed to carry a ``tool_policy_snapshot`` but whose policy never
    reached the registry exposes ZERO affordances instead of silently
    exposing everything.
    """
    raw = (os.environ.get("DLAAS_TOOL_POLICY_FAIL_CLOSED", "") or "").strip().lower()
    return raw in _FAIL_CLOSED_TRUTHY


class AffordanceRegistryError(RuntimeError):
    """Base class for registry errors. Subclass so callers can
    distinguish "bug in my code" from "deployment misconfiguration".
    """


class AffordanceAlreadyRegisteredError(AffordanceRegistryError):
    """Two descriptors share the same ``name``; the registry is
    single-owner per name, no shadowing.
    """


class AffordanceRegistrySealedError(AffordanceRegistryError):
    """A mutation was attempted after ``seal()``."""


@dataclass(frozen=True)
class AffordanceLintWarning:
    """One warning from registration-time lint.

    Slice 1 lints nothing beyond what ``AffordanceDescriptor.__post_init__``
    already enforces (post_init raises, so "violating" is never
    surfaced as a warning). Keeping this dataclass now so future
    slices can add soft lints (e.g. "examples is empty") without
    changing the API.
    """

    descriptor_name: str
    severity: str  # "info" / "warn"
    message: str


class AffordanceRegistry:
    """Session-scoped affordance registry.

    Usage:

        registry = AffordanceRegistry()
        registry.register_all(vertical.affordances())
        registry.seal()   # optional: lock the registry
        # Runtime reads:
        descriptor = registry.get("read_file")
        for d in registry.by_kind(AffordanceKind.TOOL):
            ...
    """

    def __init__(self, *, fail_closed: bool | None = None) -> None:
        self._descriptors: dict[str, AffordanceDescriptor] = {}
        self._registration_order: list[str] = []
        self._sealed: bool = False
        self._lint_warnings: list[AffordanceLintWarning] = []
        # debt #16: per-contract tool policy snapshots indexed by
        # contract_id. Each entry is a frozenset of allowed
        # affordance names (whitelist semantics; absence = allow-all
        # legacy back-compat). Set via :meth:`set_contract_policy`;
        # consumed by :meth:`list_for_contract`. Never mutated
        # in-place after `seal()`.
        self._contract_policies: dict[str, frozenset[str]] = {}
        # Tool-policy fail-closed gate (R10 / debt #16). Explicit arg
        # wins; otherwise the platform-wide env default. When True, a
        # ``contract_id`` with no registered policy is deny-all rather
        # than allow-all. Default-OFF keeps existing behaviour (SHADOW).
        self._fail_closed: bool = (
            _env_fail_closed_default() if fail_closed is None else fail_closed
        )

    @property
    def fail_closed(self) -> bool:
        return self._fail_closed

    # ------------------------------------------------------------------
    # Write path (startup only)
    # ------------------------------------------------------------------

    def register(self, descriptor: AffordanceDescriptor) -> None:
        """Add one descriptor. Raises on duplicate name or after seal."""
        if self._sealed:
            raise AffordanceRegistrySealedError(
                f"Cannot register {descriptor.name!r}: registry is sealed. "
                f"Register all affordances before calling seal()."
            )
        existing = self._descriptors.get(descriptor.name)
        if existing is not None:
            raise AffordanceAlreadyRegisteredError(
                f"Affordance {descriptor.name!r} is already registered "
                f"(kind={existing.kind.value}, version={existing.version}). "
                f"Names must be globally unique; pick a different name or "
                f"use a versioned variant (e.g. {descriptor.name}_v2)."
            )
        self._descriptors[descriptor.name] = descriptor
        self._registration_order.append(descriptor.name)

    def register_all(self, descriptors: Iterable[AffordanceDescriptor]) -> None:
        """Bulk registration. Atomic: if any descriptor fails, none are
        registered.
        """
        if self._sealed:
            raise AffordanceRegistrySealedError(
                "Cannot register_all: registry is sealed."
            )
        to_add = list(descriptors)
        # Pre-check uniqueness against existing AND within the batch.
        seen: set[str] = set()
        for d in to_add:
            if d.name in self._descriptors:
                raise AffordanceAlreadyRegisteredError(
                    f"Batch registration: {d.name!r} is already registered."
                )
            if d.name in seen:
                raise AffordanceAlreadyRegisteredError(
                    f"Batch registration contains duplicate name {d.name!r}."
                )
            seen.add(d.name)
        # Now commit.
        for d in to_add:
            self._descriptors[d.name] = d
            self._registration_order.append(d.name)

    def seal(self) -> None:
        """Lock the registry. Post-seal registrations raise."""
        self._sealed = True

    # ------------------------------------------------------------------
    # Read path (runtime)
    # ------------------------------------------------------------------

    @property
    def sealed(self) -> bool:
        return self._sealed

    def __len__(self) -> int:
        return len(self._descriptors)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._descriptors

    def get(self, name: str) -> AffordanceDescriptor:
        """Fetch by name. Raises ``KeyError`` when unknown.

        Deliberately raises rather than returning ``None`` so a typo
        in a vertical's affordance name fails loudly instead of
        silently yielding an empty candidate set.
        """
        descriptor = self._descriptors.get(name)
        if descriptor is None:
            raise KeyError(
                f"No affordance registered with name {name!r}. "
                f"Registered: {sorted(self._descriptors)!r}."
            )
        return descriptor

    def try_get(self, name: str) -> AffordanceDescriptor | None:
        """Non-raising variant for callers that legitimately may
        encounter unknown names (e.g. stale persistence hits).
        """
        return self._descriptors.get(name)

    def all_descriptors(self) -> tuple[AffordanceDescriptor, ...]:
        """Return every registered descriptor, in registration order.

        Deterministic ordering matters for the renderers: the
        markdown / compact list output must be stable across
        processes so cache hits work.
        """
        return tuple(self._descriptors[n] for n in self._registration_order)

    def by_kind(self, kind: AffordanceKind) -> tuple[AffordanceDescriptor, ...]:
        return tuple(
            self._descriptors[n]
            for n in self._registration_order
            if self._descriptors[n].kind is kind
        )

    def by_tag(self, tag: str) -> tuple[AffordanceDescriptor, ...]:
        return tuple(
            self._descriptors[n]
            for n in self._registration_order
            if tag in self._descriptors[n].affordance_tags
        )

    def names(self) -> tuple[str, ...]:
        return tuple(self._registration_order)

    # ------------------------------------------------------------------
    # debt #16: per-contract tool policy enforcement
    # ------------------------------------------------------------------

    def set_contract_policy(
        self,
        *,
        contract_id: str,
        allowed_affordance_names: Iterable[str],
    ) -> None:
        """Push a contract's ``tool_policy_snapshot`` whitelist into the registry.

        The DLaaS control plane computes the whitelist from
        ``contract.tool_policy_snapshot`` and calls this method so the
        runtime read path can filter by it. Per debt #16, this used to
        be computed but never wired — sessions saw every globally
        registered affordance regardless of the contract policy.

        ``contract_id`` is opaque to the registry (the registry does
        not validate it against any contract store; that's the
        caller's job). ``allowed_affordance_names`` is a frozen
        whitelist; unknown names are tolerated (registry doesn't
        eagerly cross-reference) so a stale policy doesn't crash the
        runtime — :meth:`list_for_contract` simply returns the
        intersection with currently-registered affordances.
        """

        if not contract_id.strip():
            raise ValueError("set_contract_policy: contract_id must be non-empty")
        self._contract_policies[contract_id] = frozenset(
            n for n in allowed_affordance_names if n.strip()
        )

    def has_contract_policy(self, contract_id: str) -> bool:
        return contract_id in self._contract_policies

    def list_for_contract(
        self, contract_id: str | None
    ) -> tuple[AffordanceDescriptor, ...]:
        """Return descriptors visible to ``contract_id`` (debt #16).

        Behaviour:

        * ``contract_id is None`` → return :meth:`all_descriptors`
          (legacy back-compat for sites that don't yet plumb a
          contract id through).
        * ``contract_id`` has no registered policy →
          - fail-closed OFF (default / SHADOW): return
            :meth:`all_descriptors` (legacy contract created before
            tool_policy_snapshot was wired; default-allow keeps
            existing behaviour observable).
          - fail-closed ON (``DLAAS_TOOL_POLICY_FAIL_CLOSED=true``):
            return an empty tuple (deny-all). A contract that should
            have published a policy but didn't exposes ZERO
            affordances instead of silently exposing everything.
        * ``contract_id`` has a registered policy → return only the
          intersection with the whitelist, preserving registration
          order so the rendered tool list stays deterministic.
        """

        if contract_id is None:
            return self.all_descriptors()
        if contract_id not in self._contract_policies:
            if self._fail_closed:
                return ()
            return self.all_descriptors()
        allow = self._contract_policies[contract_id]
        return tuple(
            self._descriptors[n]
            for n in self._registration_order
            if n in allow
        )

    def list_for_session(
        self, *, contract_id: str | None = None
    ) -> tuple[AffordanceDescriptor, ...]:
        """Session-time read alias for :meth:`list_for_contract`.

        Most session callers don't think in "contract" terms — they
        just have a session that maps to one. We expose this thin
        alias so the call site reads naturally:

            for d in registry.list_for_session(contract_id=ctx.contract_id):
                ...
        """

        return self.list_for_contract(contract_id)

    # ------------------------------------------------------------------
    # Lint surface (slice 2 will grow this)
    # ------------------------------------------------------------------

    def lint_warnings(self) -> tuple[AffordanceLintWarning, ...]:
        """Return accumulated registration-time lint warnings.

        Empty in slice 1: all currently-declared invariants are
        enforced by ``AffordanceDescriptor.__post_init__`` (raises),
        so "violating" never manifests as a warning. Slice 2 may add
        soft lints such as ``examples=()`` or overly-short
        ``description`` and these will flow here.
        """
        return tuple(self._lint_warnings)


__all__ = [
    "AffordanceAlreadyRegisteredError",
    "AffordanceLintWarning",
    "AffordanceRegistry",
    "AffordanceRegistryError",
    "AffordanceRegistrySealedError",
]
