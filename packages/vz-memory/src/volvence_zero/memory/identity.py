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

from volvence_zero.memory.contracts import (
    MemoryEntry,
    MemoryStoreCheckpoint,
    MemoryStratum,
)
from volvence_zero.memory.persistence import (
    FileSystemPersistenceBackend,
    PersistenceBackend,
    resolve_persistence_backend,
)
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


def legacy_single_layer_scope(identity: UserIdentity | None) -> str | None:
    """Return the pre-two-layer single-layer ``user_scope`` alias, if any.

    Before debt #46 flipped two-layer scope to the default, durable
    memory entries were tagged ``user_scope:<user_id>`` (the bare
    end-user id). After the flip the same end user is tagged
    ``user_scope:<tenant>:<end_user>``. The on-disk *checkpoint files*
    are keyed by ``user_id`` directory and so are byte-identical across
    the flip, but the ``user_scope:<scope>`` *tags* embedded on durable
    entries differ. Hosts that need to keep finding / deleting memory
    written under the old single-layer key pass this alias to
    :func:`list_durable_entries_for_scope` /
    :func:`delete_entries_for_scope` via ``extra_scopes`` so the
    migration never silently orphans on-disk memory (the kernel
    constraint: do not silently re-key existing memory).

    Returns ``None`` when the identity is already single-layer
    (``scope_key == user_id``) or anonymous — there is no distinct
    legacy alias to add.
    """

    if identity is None:
        return None
    if identity.scope_key == identity.user_id:
        return None
    return identity.user_id


def _safe_scope_dirname(user_id: str) -> str:
    """Return a filesystem-safe directory name for a logical scope key.

    Scope keys intentionally contain semantic separators such as
    ``company:<id>`` or ``tenant:user``. Those are legal logical ids but
    not legal path components on Windows (``:`` is forbidden). Keep the
    logical key unchanged everywhere else; only encode the path segment.
    The prefix makes the mapping explicit and avoids collisions with
    pre-existing plain directory names.
    """

    import base64

    encoded = base64.urlsafe_b64encode(user_id.encode("utf-8")).decode("ascii")
    return "scope_" + encoded.rstrip("=")


def scoped_memory_dir(*, root_dir: str | os.PathLike[str], user_id: str) -> Path:
    """Compute the filesystem directory for a scoped store."""

    return Path(root_dir) / _safe_scope_dirname(user_id)


def build_scoped_memory_store(
    *,
    identity: UserIdentity | None,
    root_dir: str | os.PathLike[str] | None = None,
    latent_dim: int = 8,
    nested_profile: bool = True,
    seed_checkpoint: MemoryStoreCheckpoint | None = None,
    persistence_backend: PersistenceBackend | None = None,
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

    D2 (pluggable scoped-memory backend): ``persistence_backend`` lets a
    host inject any :class:`PersistenceBackend` (e.g. a Postgres-backed
    backend) instead of the on-disk JSON default. When ``None`` the
    backend is resolved via
    :func:`volvence_zero.memory.persistence.resolve_persistence_backend`
    (env ``VZ_MEMORY_BACKEND`` / ``VZ_MEMORY_PG_DSN``), which defaults to
    the historical :class:`FileSystemPersistenceBackend` so existing
    deployments are byte-for-byte unchanged. The two-layer ``scope_key``
    is used as the backend ``namespace`` so a shared Postgres instance
    partitions every tenant/end-user without filesystem fan-out.

    NW10: ``seed_checkpoint`` lets a baked LifeformTemplate's canonical
    ``memory_checkpoint`` seed the scoped store **exactly once** — on the
    first session for this scope (when ``load_from_backend()`` reports no
    prior content). On every subsequent session the accumulated on-disk
    state wins and the seed is ignored, so the character starts from its
    trained self and then durably co-evolves with the player rather than
    resetting to the checkpoint every session. The seed is materialised
    to disk immediately so a crash before the first ``end_scene`` drain
    does not lose the canonical baseline.
    """

    if identity is None or not identity.has_permission("persist"):
        store = build_default_memory_store(
            latent_dim=latent_dim, nested_profile=nested_profile
        )
        # No durable backend → at least give the (ephemeral) session the
        # canonical baseline so an unscoped fallback isn't a blank slate.
        if seed_checkpoint is not None:
            store.restore_checkpoint(seed_checkpoint)
        return store
    if persistence_backend is not None:
        backend: PersistenceBackend = persistence_backend
    else:
        # Resolve the backend through the SSOT funnel. The filesystem
        # default still requires (and mkdir-creates) the per-scope
        # directory so existing on-disk deployments are unchanged; a
        # non-filesystem backend (memory / postgres) namespaces by the
        # logical scope_key instead.
        backend_choice = os.environ.get("VZ_MEMORY_BACKEND", "").strip().lower()
        if backend_choice in ("", "filesystem", "file", "fs"):
            if root_dir is None:
                raise ValueError(
                    "root_dir is required when building a scoped MemoryStore "
                    "for a known identity (filesystem backend)."
                )
            user_dir = scoped_memory_dir(root_dir=root_dir, user_id=identity.user_id)
            user_dir.mkdir(parents=True, exist_ok=True)
            backend = FileSystemPersistenceBackend(base_dir=str(user_dir))
        else:
            backend = resolve_persistence_backend(
                base_dir=root_dir,
                namespace=identity.scope_key,
                backend=backend_choice,
            )
    store = build_default_memory_store(
        latent_dim=latent_dim, nested_profile=nested_profile
    )
    store._persistence_backend = backend  # noqa: SLF001
    # Eagerly load so the session starts with any prior durable content.
    # load_from_backend returns False when no checkpoint exists; that is
    # the expected first-session case and is not an error.
    loaded = store.load_from_backend()
    # NW10: seed-once from the canonical checkpoint. Only when there is no
    # prior durable content (first session for this scope). Never re-seed
    # over accumulated memory — that would wipe the relationship history.
    if not loaded and seed_checkpoint is not None:
        store.restore_checkpoint(seed_checkpoint)
        store.save_to_backend()
    return store


def _durable_entries_iter(store: MemoryStore) -> Iterable[MemoryEntry]:
    """Iterate the durable stratum without going through retrieval ranking.

    Lives here (inside vz-memory) so other wheels never access
    ``_entries_for`` directly. Consumers should treat it as an admin /
    inspection surface; normal runtime reads still flow through the
    memory snapshot.
    """

    return store._entries_for(MemoryStratum.DURABLE)  # noqa: SLF001


def _scope_tags(user_scope: str, extra_scopes: tuple[str, ...]) -> frozenset[str]:
    """Build the set of ``user_scope:<scope>`` tags to match.

    ``extra_scopes`` lets callers add backward-compat aliases (e.g. the
    pre-two-layer single-layer key from :func:`legacy_single_layer_scope`)
    so durable entries written under an older scope convention are still
    found after the two-layer-default flip. Empty / whitespace aliases
    are dropped so an absent legacy alias is a no-op.
    """

    scopes = [user_scope, *extra_scopes]
    return frozenset(f"user_scope:{scope}" for scope in scopes if scope and scope.strip())


def list_durable_entries_for_scope(
    store: MemoryStore,
    *,
    user_scope: str,
    extra_scopes: tuple[str, ...] = (),
) -> tuple[MemoryEntry, ...]:
    """Return DURABLE rupture-repair entries tagged with ``user_scope``.

    v0 convention: entries tagged ``user_scope:<scope>`` are scoped to
    that user. Entries missing the tag are not scope-attributable and
    are excluded from this helper.

    ``extra_scopes`` adds backward-compat alias scopes (e.g. the
    pre-two-layer single-layer key) so a host migrating to the
    two-layer-default scope can still enumerate memory written under
    the old single-layer tag. Defaults to no aliases (legacy behaviour).
    """

    tags = _scope_tags(user_scope, extra_scopes)
    return tuple(
        entry
        for entry in _durable_entries_iter(store)
        if any(tag in entry.tags for tag in tags)
    )


def delete_entries_for_scope(
    store: MemoryStore,
    *,
    user_scope: str,
    extra_scopes: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Delete DURABLE entries tagged with ``user_scope``.

    Returns the deleted entry ids. The derived retrieval index is
    *not* rebuilt here — it is a projection that re-derives on next
    retrieval (``docs/specs/continuum-memory.md``). Transient and
    episodic entries are session-local and left untouched.

    ``extra_scopes`` adds backward-compat alias scopes (see
    :func:`list_durable_entries_for_scope`) so a two-layer-default
    deletion still reaches memory tagged under the old single-layer
    key — closing the GDPR / PIPL "right to be forgotten" hole that a
    silent re-key would otherwise open.
    """

    tags = _scope_tags(user_scope, extra_scopes)
    artifact_store = store._artifact_store  # noqa: SLF001
    targets = tuple(
        entry.entry_id
        for entry in artifact_store.entries_for(MemoryStratum.DURABLE)
        if any(tag in entry.tags for tag in tags)
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
    "legacy_single_layer_scope",
    "list_durable_entries_for_scope",
    "scope_key_for",
    "scoped_memory_dir",
]
