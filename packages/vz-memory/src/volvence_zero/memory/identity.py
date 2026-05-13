"""Per-user identity and scoped memory-store lifecycle.

This module is the v0 contract surface for binding an ``AgentSessionRunner``
to a specific user identity (Rupture-and-Repair M4). It owns:

* :class:`UserIdentity` — frozen identity dataclass.
* :class:`IdentityProvider` — Protocol for resolving session_id -> identity.
* :class:`AnonymousIdentityProvider` — default no-op provider.
* :class:`StaticIdentityProvider` — trivial one-identity provider for tests
  and for hosts that want to bind a lifeform session to a known user.
* :func:`build_scoped_memory_store` — builds a ``MemoryStore`` (optionally
  filesystem-backed) for a given identity.
* :func:`list_durable_entries_for_scope` / :func:`delete_entries_for_scope`
  — helpers that inspect and delete rupture-repair entries tagged with
  the given ``user_scope``.

Design rationale: ``vz-memory`` is the SSOT for memory-store lifecycle, so
the contract lives here in v0. If a second kernel package later consumes
identity (e.g. a per-user regime prior owner), the Protocol should be
promoted to ``vz-contracts`` with only ``build_scoped_memory_store``
staying here. See ``docs/specs/rupture-and-repair.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from volvence_zero.memory.contracts import MemoryEntry, MemoryStratum
from volvence_zero.memory.persistence import FileSystemPersistenceBackend
from volvence_zero.memory.store import MemoryStore, build_default_memory_store


ANONYMOUS_USER_SCOPE = "anonymous"


# ---------------------------------------------------------------------------
# Two-layer scope (debt #46 / commercialisation cross-cutting F-B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TenantIdentity:
    """B2B tenant identity (e.g. one paying customer / institution).

    Closed-alpha derives ``tenant_id == "alpha"`` automatically; B2B
    private-domain growth-advisor clients each get a real
    ``tenant_id``. The kernel reads only ``tenant_id``; ``display_name``
    is a product-layer concern.

    See `cross-cutting-foundation-packet.md` §2.2 + debt #46.
    """

    tenant_id: str
    display_name: str = ""

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.tenant_id.strip():
            raise ValueError("TenantIdentity.tenant_id must be non-empty")


@dataclass(frozen=True)
class EndUserIdentity:
    """End-user inside a tenant scope.

    The pair ``(tenant_id, end_user_id)`` is globally unique; the same
    ``end_user_id`` value may legitimately exist under different
    tenants (e.g. ``alice`` under tenant ``brand_a`` and ``brand_b``
    must not share state).
    """

    tenant_id: str
    end_user_id: str
    display_name: str = ""

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.tenant_id.strip():
            raise ValueError("EndUserIdentity.tenant_id must be non-empty")
        if not self.end_user_id or not self.end_user_id.strip():
            raise ValueError("EndUserIdentity.end_user_id must be non-empty")


def derive_scope_key(
    tenant_identity: TenantIdentity | None,
    end_user_identity: EndUserIdentity | None,
) -> str:
    """SSOT for ``scope_key`` derivation under the two-layer model.

    * Two-layer (preferred for new code)::

        TenantIdentity("brand_a"), EndUserIdentity("brand_a", "alice")
            -> "brand_a:alice"

    * Single-layer fallback (legacy closed-alpha): when only an
      ``end_user_identity`` is supplied with ``tenant_id == "alpha"``,
      the derived key is ``"alpha:<end_user_id>"``.

    Returns ``ANONYMOUS_USER_SCOPE`` if both are ``None``.
    """

    if end_user_identity is None and tenant_identity is None:
        return ANONYMOUS_USER_SCOPE
    if end_user_identity is None:
        # Only tenant scope: rare but legal (tenant-level admin actions)
        return f"{tenant_identity.tenant_id}:_admin_"  # type: ignore[union-attr]
    if tenant_identity is not None and tenant_identity.tenant_id != end_user_identity.tenant_id:
        raise ValueError(
            "derive_scope_key: tenant_identity.tenant_id "
            f"({tenant_identity.tenant_id!r}) does not match "
            f"end_user_identity.tenant_id ({end_user_identity.tenant_id!r})"
        )
    return f"{end_user_identity.tenant_id}:{end_user_identity.end_user_id}"


@dataclass(frozen=True)
class UserIdentity:
    """Frozen user identity surface.

    * ``user_id`` is the stable external identifier (chosen by the host).
    * ``scope_key`` is what memory/regime/rupture-repair will tag with;
      in v0 it equals ``user_id`` unless the host explicitly wants a
      narrower sub-scope (for example a per-device separator).
    * ``display_name`` is product-layer concern only; the kernel does
      not read it.
    * ``permissions`` is a small typed tuple of named capabilities the
      host has granted. v0 reserves ``{"persist", "inspect", "delete"}``;
      missing capabilities mean the helper APIs raise ``PermissionError``.
    * ``tenant_identity`` / ``end_user_identity`` (debt #46): optional
      two-layer scope. When supplied, ``scope_key`` should be derived
      via :func:`derive_scope_key`. Closed-alpha auto-fills these as
      ``TenantIdentity("alpha")`` + ``EndUserIdentity("alpha", user_id)``
      so the new schema is backward-compatible with the single-layer
      ``user_id == scope_key`` legacy.
    """

    user_id: str
    scope_key: str
    display_name: str = ""
    permissions: tuple[str, ...] = ("persist", "inspect", "delete")
    tenant_identity: TenantIdentity | None = None
    end_user_identity: EndUserIdentity | None = None

    def __post_init__(self) -> None:
        if not self.user_id or not self.user_id.strip():
            raise ValueError("UserIdentity.user_id must be non-empty")
        if not self.scope_key or not self.scope_key.strip():
            raise ValueError("UserIdentity.scope_key must be non-empty")
        if any(perm.strip() == "" for perm in self.permissions):
            raise ValueError("UserIdentity.permissions must not contain empty entries")
        # When both layers are supplied, scope_key must equal the SSOT derivation.
        if self.tenant_identity is not None and self.end_user_identity is not None:
            expected = derive_scope_key(self.tenant_identity, self.end_user_identity)
            if expected != self.scope_key:
                raise ValueError(
                    f"UserIdentity.scope_key={self.scope_key!r} does not match "
                    f"derive_scope_key(tenant, end_user)={expected!r}"
                )

    def has_permission(self, name: str) -> bool:
        return name in self.permissions


class IdentityProvider(Protocol):
    """Resolve a session_id to the user identity (if any)."""

    def resolve(self, session_id: str) -> UserIdentity | None:
        """Return the user identity bound to this session, or ``None``."""


class AnonymousIdentityProvider:
    """Default provider: every session is anonymous (scoped store disabled)."""

    def resolve(self, session_id: str) -> UserIdentity | None:  # noqa: ARG002
        return None


@dataclass(frozen=True)
class StaticIdentityProvider:
    """Provider that always returns the same identity.

    Useful for tests and for lifeform adapters that bind a single
    product-level user to one or more sessions. Hosts that serve
    multiple users should implement their own ``IdentityProvider``.
    """

    identity: UserIdentity

    def resolve(self, session_id: str) -> UserIdentity | None:  # noqa: ARG002
        return self.identity


def scope_key_for(identity: UserIdentity | None) -> str:
    """Return the ``user_scope`` tag value for a given identity.

    Anonymous sessions use :data:`ANONYMOUS_USER_SCOPE` so rupture-repair
    entries from an anonymous session are still traceable but never
    collide with an identified user's scope.
    """

    if identity is None:
        return ANONYMOUS_USER_SCOPE
    return identity.scope_key


def scoped_memory_dir(*, root_dir: str | os.PathLike[str], user_id: str) -> Path:
    """Compute the filesystem directory for a scoped store."""

    return Path(root_dir) / user_id


def build_scoped_memory_store(
    *,
    identity: UserIdentity | None,
    root_dir: str | os.PathLike[str] | None = None,
    latent_dim: int = 8,
    nested_profile: bool = True,
) -> MemoryStore:
    """Build a ``MemoryStore`` for the given identity.

    * If ``identity`` is ``None`` or lacks the ``"persist"`` permission,
      returns a fresh in-memory store with no filesystem backend
      (current default behavior — no cross-session memory).
    * If ``identity`` is present, ``root_dir`` is required and the
      store is wired with a filesystem persistence backend rooted at
      ``<root_dir>/<user_id>/memory`` plus a parallel eager-load so
      previous sessions' durable entries are available to the new
      session.
    """

    if identity is None or not identity.has_permission("persist"):
        return build_default_memory_store(
            latent_dim=latent_dim, nested_profile=nested_profile
        )
    if root_dir is None:
        raise ValueError(
            "root_dir is required when building a scoped MemoryStore for a known identity."
        )
    user_dir = scoped_memory_dir(root_dir=root_dir, user_id=identity.user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    backend = FileSystemPersistenceBackend(base_dir=str(user_dir))
    store = build_default_memory_store(
        latent_dim=latent_dim, nested_profile=nested_profile
    )
    store._persistence_backend = backend  # noqa: SLF001
    # Eagerly load so the session starts with any prior durable content.
    # load_from_backend returns False when no checkpoint exists; that is
    # the expected first-session case and is not an error.
    store.load_from_backend()
    return store


def _durable_entries_iter(store: MemoryStore) -> Iterable[MemoryEntry]:
    """Iterate the durable stratum without going through retrieval ranking.

    Lives here (inside vz-memory) so other wheels never access
    ``_entries_for`` directly. Consumers should treat it as an admin /
    inspection surface; normal runtime reads still flow through the
    memory snapshot.
    """

    return store._entries_for(MemoryStratum.DURABLE)  # noqa: SLF001


def list_durable_entries_for_scope(
    store: MemoryStore,
    *,
    user_scope: str,
) -> tuple[MemoryEntry, ...]:
    """Return DURABLE rupture-repair entries tagged with ``user_scope``.

    v0 convention: entries tagged ``user_scope:<scope>`` are scoped to
    that user. Entries missing the tag are not scope-attributable and
    are excluded from this helper.
    """

    tag = f"user_scope:{user_scope}"
    return tuple(
        entry for entry in _durable_entries_iter(store) if tag in entry.tags
    )


def delete_entries_for_scope(
    store: MemoryStore,
    *,
    user_scope: str,
) -> tuple[str, ...]:
    """Delete DURABLE entries tagged with ``user_scope``.

    Returns the deleted entry ids. The derived retrieval index is
    *not* rebuilt here — it is a projection that re-derives on next
    retrieval (``docs/specs/continuum-memory.md``). Transient and
    episodic entries are session-local and left untouched.
    """

    tag = f"user_scope:{user_scope}"
    artifact_store = store._artifact_store  # noqa: SLF001
    targets = tuple(
        entry.entry_id
        for entry in artifact_store.entries_for(MemoryStratum.DURABLE)
        if tag in entry.tags
    )
    deleted: list[str] = []
    for entry_id in targets:
        removed = artifact_store.delete_entry(entry_id)
        if removed is not None:
            deleted.append(entry_id)
    return tuple(deleted)


__all__ = [
    "ANONYMOUS_USER_SCOPE",
    "AnonymousIdentityProvider",
    "EndUserIdentity",
    "IdentityProvider",
    "StaticIdentityProvider",
    "TenantIdentity",
    "UserIdentity",
    "build_scoped_memory_store",
    "delete_entries_for_scope",
    "derive_scope_key",
    "list_durable_entries_for_scope",
    "scope_key_for",
    "scoped_memory_dir",
]
