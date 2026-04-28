"""SessionManager \u2014 multi-tenant lifecycle for ``LifeformSession``s.

Hard rule: each ``session_id`` maps to exactly one in-process
``LifeformSession``. Sessions are NOT shared across IDs (the kernel's
single-owner contract requires this), so a single-process service can
host as many concurrent users as memory allows.

Substrate sharing: the manager is given a ``substrate_runtime`` (or
``None`` for synthetic / per-session-runtime mode) and passes that same
instance to every ``lifeform_factory(runtime)`` call. So one
``TransformersOpenWeightResidualRuntime`` in memory backs every
session's Brain. This is the "one Qwen on one GPU, many concurrent
chats" deployment model.

Eviction:

* ``max_sessions`` caps the total live session count. When a new
  ``create_session`` would push over the cap, the *least-recently-used*
  session is closed.
* ``idle_eviction_seconds`` (optional) is checked on every operation; any
  session that has been idle longer than the threshold is closed.

Thread / async safety: all methods are ``async`` and hold an internal
``asyncio.Lock`` while mutating ``_sessions``. The substrate runtime
itself is consumed under the asyncio event loop's single-thread
guarantee \u2014 ``runtime.generate(...)`` is sync and blocks the loop, so
concurrent sessions naturally serialise. Do NOT run ``run_in_executor``
on the runtime without re-introducing a ``threading.Lock`` here.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lifeform_core import Lifeform, LifeformSession

if TYPE_CHECKING:
    from volvence_zero.substrate import OpenWeightResidualRuntime


@dataclass
class _SessionEntry:
    session: LifeformSession
    last_active_at: float


class SessionManager:
    """Owns the live ``{session_id -> LifeformSession}`` map."""

    def __init__(
        self,
        *,
        lifeform_factory: Callable[["OpenWeightResidualRuntime | None"], Lifeform],
        vertical_name: str,
        max_sessions: int = 256,
        idle_eviction_seconds: float | None = 60 * 30,
        clock: Callable[[], float] = time.monotonic,
        substrate_runtime: "OpenWeightResidualRuntime | None" = None,
    ) -> None:
        self._factory = lifeform_factory
        self._vertical_name = vertical_name
        self._max_sessions = max_sessions
        self._idle_eviction_seconds = idle_eviction_seconds
        self._clock = clock
        self._substrate_runtime = substrate_runtime
        self._sessions: dict[str, _SessionEntry] = {}
        self._lock = asyncio.Lock()

    @property
    def substrate_runtime(self) -> "OpenWeightResidualRuntime | None":
        return self._substrate_runtime

    @property
    def vertical_name(self) -> str:
        return self._vertical_name

    @property
    def max_sessions(self) -> int:
        return self._max_sessions

    def session_count(self) -> int:
        return len(self._sessions)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def create_session(self, *, session_id: str | None = None) -> LifeformSession:
        async with self._lock:
            self._evict_idle_locked()
            self._evict_lru_to_capacity_locked(needed=1)

            sid = session_id or self._fresh_session_id()
            if sid in self._sessions:
                raise SessionAlreadyExistsError(sid)
            life = self._factory(self._substrate_runtime)
            session = life.create_session(session_id=sid)
            self._sessions[sid] = _SessionEntry(
                session=session,
                last_active_at=self._clock(),
            )
            return session

    async def get_session(self, session_id: str) -> LifeformSession:
        async with self._lock:
            self._evict_idle_locked()
            entry = self._sessions.get(session_id)
            if entry is None:
                raise SessionNotFoundError(session_id)
            entry.last_active_at = self._clock()
            return entry.session

    async def close_session(self, session_id: str) -> bool:
        async with self._lock:
            entry = self._sessions.pop(session_id, None)
            return entry is not None

    async def has_session(self, session_id: str) -> bool:
        async with self._lock:
            return session_id in self._sessions

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fresh_session_id(self) -> str:
        return f"sess-{uuid.uuid4().hex[:12]}"

    def _evict_idle_locked(self) -> int:
        if self._idle_eviction_seconds is None:
            return 0
        cutoff = self._clock() - self._idle_eviction_seconds
        stale = [
            sid for sid, entry in self._sessions.items() if entry.last_active_at < cutoff
        ]
        for sid in stale:
            self._sessions.pop(sid, None)
        return len(stale)

    def _evict_lru_to_capacity_locked(self, *, needed: int) -> int:
        target = max(0, self._max_sessions - needed)
        if len(self._sessions) <= target:
            return 0
        # Sort ascending by last_active_at; evict oldest until under cap.
        ordered = sorted(
            self._sessions.items(), key=lambda kv: kv[1].last_active_at
        )
        evictions = 0
        while len(self._sessions) > target and ordered:
            sid, _entry = ordered.pop(0)
            self._sessions.pop(sid, None)
            evictions += 1
        return evictions


class SessionNotFoundError(LookupError):
    """Raised when a session_id is not in the manager."""


class SessionAlreadyExistsError(ValueError):
    """Raised when create_session is called with an explicit ID already in use."""
