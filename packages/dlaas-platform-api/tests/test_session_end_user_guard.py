"""Tests for the P0 session/end-user consistency guard."""

from __future__ import annotations

import pytest

from dlaas_platform_api.app import (
    _SessionEndUserMismatch,
    _get_or_create_session,
    _session_end_user_remap_allowed,
)


class _FakeSessionManager:
    """Minimal manager exposing the get/create/session_end_user surface."""

    class _NotFound(Exception):
        pass

    def __init__(self, *, existing: dict[str, str] | None = None) -> None:
        # session_id -> end_user_ref bound at create
        self._sessions = dict(existing or {})
        self.created: list[tuple[str, str | None]] = []

    async def get_session(self, session_id: str):
        from lifeform_service import SessionNotFoundError

        if session_id not in self._sessions:
            raise SessionNotFoundError(session_id)
        return f"session:{session_id}"

    async def create_session(self, *, session_id: str, user_id: str | None = None):
        self._sessions[session_id] = user_id or ""
        self.created.append((session_id, user_id))
        return f"session:{session_id}"

    def session_end_user(self, session_id: str) -> str | None:
        return self._sessions.get(session_id) or None


async def test_create_when_absent() -> None:
    mgr = _FakeSessionManager()
    out = await _get_or_create_session(mgr, "s1", user_id="alice")
    assert out == "session:s1"
    assert mgr.created == [("s1", "alice")]


async def test_reuse_same_end_user_ok() -> None:
    mgr = _FakeSessionManager(existing={"s1": "alice"})
    out = await _get_or_create_session(mgr, "s1", user_id="alice")
    assert out == "session:s1"


async def test_reuse_different_end_user_raises() -> None:
    mgr = _FakeSessionManager(existing={"s1": "alice"})
    with pytest.raises(_SessionEndUserMismatch):
        await _get_or_create_session(mgr, "s1", user_id="bob")


async def test_remap_allowed_via_env(monkeypatch) -> None:
    monkeypatch.setenv("VZ_ALLOW_SESSION_END_USER_REMAP", "1")
    assert _session_end_user_remap_allowed() is True
    mgr = _FakeSessionManager(existing={"s1": "alice"})
    # No raise when remap is explicitly allowed.
    out = await _get_or_create_session(mgr, "s1", user_id="bob")
    assert out == "session:s1"


async def test_guard_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("VZ_ALLOW_SESSION_END_USER_REMAP", raising=False)
    assert _session_end_user_remap_allowed() is False
