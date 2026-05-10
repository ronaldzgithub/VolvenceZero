"""Closed-alpha service configuration and identity binding.

This module is product/service owned. It maps authenticated alpha users to
``UserIdentity`` objects consumed by ``Brain``; it does not interpret user
text or mutate kernel state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from volvence_zero.memory import UserIdentity


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

    def is_allowed(self, user_id: str) -> bool:
        return not self.alpha_users or user_id in self.alpha_users


class AlphaIdentityProvider:
    """Session-scoped identity provider for the HTTP alpha service."""

    def __init__(self, *, allowed_users: frozenset[str] = frozenset()) -> None:
        self._allowed_users = allowed_users
        self._session_to_identity: dict[str, UserIdentity] = {}

    def bind_session(self, *, session_id: str, user_id: str) -> UserIdentity:
        user_id = user_id.strip()
        if not user_id:
            raise ValueError("alpha user_id must be non-empty")
        if self._allowed_users and user_id not in self._allowed_users:
            raise PermissionError(f"alpha user {user_id!r} is not allowed")
        # Closed alpha intentionally keeps path scope and memory tag scope
        # identical to avoid user_id/scope_key divergence.
        identity = UserIdentity(user_id=user_id, scope_key=user_id)
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
