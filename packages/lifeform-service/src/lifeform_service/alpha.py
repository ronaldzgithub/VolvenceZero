"""Closed-alpha service configuration and identity binding.

This module is product/service owned. It maps authenticated alpha users to
``UserIdentity`` objects consumed by ``Brain``; it does not interpret user
text or mutate kernel state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from volvence_zero.memory import (
    EndUserIdentity,
    TenantIdentity,
    UserIdentity,
    derive_scope_key,
)


# Two-layer scope is the default since debt #46 ACTIVE (2026-05-14).
# ``bind_session`` derives ``scope_key == f"{tenant_id}:{end_user_id}"``
# (using ``DEFAULT_ALPHA_TENANT_ID`` when no explicit tenant is
# supplied) and wires the typed tenant + end-user identity onto
# ``UserIdentity``. Hosts that must preserve the previous SHADOW
# default ``scope_key == user_id`` contract (e.g. closed-alpha
# evidence / memory files written before the upgrade) call
# ``bind_session_legacy_alias(...)`` explicitly so the migration is
# observable rather than silent.
DEFAULT_ALPHA_TENANT_ID = "alpha"


ALPHA_DISCLAIMER = (
    "Volvence Zero closed alpha is a research prototype for companionship "
    "and continuity. It is not therapy, medical care, legal advice, or an "
    "emergency service. Do not use it for minors or crisis situations."
)


@dataclass(frozen=True)
class AlphaServiceConfig:
    enabled: bool = False
    memory_scope_root_dir: str | None = None
    evidence_root_dir: str | None = None
    service_version: str = "closed-alpha-v0"
    policy_version: str = "alpha-policy-v0"
    alpha_users: frozenset[str] = frozenset()
    # D6 (#alpha-reload): the file the allow-list was loaded from, kept
    # so the running service can hot-reload it (reload endpoint / SIGHUP)
    # when a new sign-in is appended — no platform restart required.
    # ``None`` means the allow-list was supplied inline (not file-backed)
    # and therefore cannot be reloaded from disk.
    alpha_users_path: str | None = None

    def is_allowed(self, user_id: str) -> bool:
        return not self.alpha_users or user_id in self.alpha_users

    def with_alpha_users(self, alpha_users: frozenset[str]) -> "AlphaServiceConfig":
        """Return a copy with a refreshed allow-list (frozen-safe).

        Used by the hot-reload path: :class:`AlphaServiceConfig` is
        frozen, so reloading swaps in a new instance rather than
        mutating in place.
        """

        return replace(self, alpha_users=alpha_users)


class AlphaIdentityProvider:
    """Session-scoped identity provider for the HTTP alpha service."""

    def __init__(self, *, allowed_users: frozenset[str] = frozenset()) -> None:
        self._allowed_users = allowed_users
        self._session_to_identity: dict[str, UserIdentity] = {}

    @property
    def allowed_users(self) -> frozenset[str]:
        return self._allowed_users

    def set_allowed_users(self, allowed_users: frozenset[str]) -> None:
        """Hot-swap the allow-list without dropping live sessions (D6).

        New sign-ins take effect immediately for *future* binds; already
        bound sessions are intentionally left intact (revoking a user
        mid-session is a separate, explicit action — closing the
        session — not a side-effect of an allow-list edit).
        """

        self._allowed_users = frozenset(allowed_users)

    def reload_allowed_users_from_file(self, path: str | None) -> frozenset[str]:
        """Re-read the allow-list file and apply it. Returns the new set.

        ``path is None`` means the allow-list was supplied inline and
        there is nothing on disk to reload; we leave the current set
        untouched and return it. Any malformed file raises (the caller
        maps it to a 4xx / keeps the old set) so a bad edit never
        silently empties the allow-list.
        """

        if path is None:
            return self._allowed_users
        refreshed = load_alpha_users(path)
        self._allowed_users = refreshed
        return refreshed

    def bind_session(
        self,
        *,
        session_id: str,
        user_id: str | None = None,
        end_user_id: str | None = None,
        tenant_id: str = DEFAULT_ALPHA_TENANT_ID,
    ) -> UserIdentity:
        """Default two-layer scope binding (debt #46 ACTIVE).

        Accepts either ``end_user_id=`` (preferred new contract) or
        legacy positional ``user_id=`` (preserved so closed-alpha
        callers that pre-date the upgrade keep compiling). Both names
        are routed through :meth:`bind_session_two_layer` so the
        derived ``scope_key == f"{tenant_id}:{end_user_id}"``.

        For sites that must preserve the previous SHADOW
        ``scope_key == user_id`` contract (because their evidence /
        memory files on disk were written under it), call
        :meth:`bind_session_legacy_alias` explicitly so the migration
        is observable.
        """

        effective_id = end_user_id if end_user_id is not None else user_id
        if effective_id is None:
            raise ValueError(
                "AlphaIdentityProvider.bind_session requires end_user_id "
                "(or legacy user_id)"
            )
        return self.bind_session_two_layer(
            session_id=session_id,
            end_user_id=effective_id,
            tenant_id=tenant_id,
        )

    def bind_session_legacy_alias(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> UserIdentity:
        """Single-layer legacy migration shim (debt #46 backward-compat).

        Closed-alpha bindings under the previous SHADOW default kept
        ``scope_key == user_id`` so on-disk evidence + scoped memory
        files were keyed by the bare user_id. Hosts that still need to
        load those files call this method explicitly. New sites
        should prefer :meth:`bind_session` (default two-layer) so
        admin / end-user separation is automatic.
        """

        user_id = user_id.strip()
        if not user_id:
            raise ValueError(
                "AlphaIdentityProvider.bind_session_legacy_alias: "
                "user_id must be non-empty"
            )
        if self._allowed_users and user_id not in self._allowed_users:
            raise PermissionError(f"alpha user {user_id!r} is not allowed")
        identity = UserIdentity(user_id=user_id, scope_key=user_id)
        self._session_to_identity[session_id] = identity
        return identity

    def bind_session_two_layer(
        self,
        *,
        session_id: str,
        end_user_id: str,
        tenant_id: str = DEFAULT_ALPHA_TENANT_ID,
    ) -> UserIdentity:
        """Explicit two-layer scope binding (debt #46 ACTIVE).

        Derives ``scope_key == f"{tenant_id}:{end_user_id}"`` via
        :func:`volvence_zero.memory.derive_scope_key`. This is what
        :meth:`bind_session` calls internally; named entry kept for
        sites that want the intent visible at the call-site.
        """

        end_user_id = end_user_id.strip()
        tenant_id = tenant_id.strip()
        if not end_user_id:
            raise ValueError("alpha two-layer: end_user_id must be non-empty")
        if not tenant_id:
            raise ValueError("alpha two-layer: tenant_id must be non-empty")
        if self._allowed_users and end_user_id not in self._allowed_users:
            raise PermissionError(f"alpha user {end_user_id!r} is not allowed")
        tenant = TenantIdentity(tenant_id=tenant_id)
        end_user = EndUserIdentity(tenant_id=tenant_id, end_user_id=end_user_id)
        scope_key = derive_scope_key(tenant, end_user)
        identity = UserIdentity(
            user_id=end_user_id,
            scope_key=scope_key,
            tenant_identity=tenant,
            end_user_identity=end_user,
        )
        self._session_to_identity[session_id] = identity
        return identity

    def resolve(self, session_id: str) -> UserIdentity | None:
        return self._session_to_identity.get(session_id)

    def user_for_session(self, session_id: str) -> str | None:
        identity = self.resolve(session_id)
        return identity.user_id if identity is not None else None

    def unbind_session(self, session_id: str) -> bool:
        """Drop a session->identity mapping. Returns True if removed.

        SessionManager calls this on session close so the
        ``session_id -> UserIdentity`` map does not grow unboundedly
        across long-lived processes (matters when the substrate
        provider closes all sessions on a model swap).
        """
        return self._session_to_identity.pop(session_id, None) is not None

    def clear_all_sessions(self) -> int:
        """Remove every session->identity mapping. Returns the count.

        Used by ``SessionManager.close_all_sessions_sync`` during a
        substrate swap.
        """
        count = len(self._session_to_identity)
        self._session_to_identity.clear()
        return count


def load_alpha_users(path: str | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    raw = Path(path).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if isinstance(payload, list):
        users = payload
    elif isinstance(payload, dict):
        users = payload.get("users", ())
    else:
        raise ValueError("alpha users file must be a JSON list or object with users")
    out: set[str] = set()
    for item in users:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("alpha users must be non-empty strings")
        out.add(item.strip())
    return frozenset(out)


def alpha_config_to_json(config: AlphaServiceConfig) -> Mapping[str, object]:
    return {
        "enabled": config.enabled,
        "memory_scope_root_dir": config.memory_scope_root_dir,
        "evidence_root_dir": config.evidence_root_dir,
        "service_version": config.service_version,
        "policy_version": config.policy_version,
        "allowed_user_count": len(config.alpha_users),
        "disclaimer": ALPHA_DISCLAIMER,
    }


__all__ = (
    "ALPHA_DISCLAIMER",
    "AlphaIdentityProvider",
    "AlphaServiceConfig",
    "alpha_config_to_json",
    "load_alpha_users",
)
