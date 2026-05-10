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
import pathlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lifeform_core import Lifeform, LifeformSession
from lifeform_service.alpha import AlphaIdentityProvider
from lifeform_service.templates import (
    TemplateContext,
    VerticalTemplateAdapter,
)

if TYPE_CHECKING:
    from volvence_zero.memory import IdentityProvider
    from volvence_zero.substrate import OpenWeightResidualRuntime


@dataclass
class _SessionEntry:
    session: LifeformSession
    last_active_at: float
    # Optional per-session payload produced by the vertical's template
    # adapter when this session was created via ``build_*_session_context``.
    # Sessions created via the legacy ``factory`` / ``alpha_factory`` path
    # have ``template_context=None`` and cannot be saved as templates.
    template_context: TemplateContext | None = None


class SessionManager:
    """Owns the live ``{session_id -> LifeformSession}`` map."""

    def __init__(
        self,
        *,
        lifeform_factory: Callable[["OpenWeightResidualRuntime | None"], Lifeform],
        alpha_lifeform_factory: Callable[
            ["OpenWeightResidualRuntime | None", "IdentityProvider", str | None],
            Lifeform,
        ]
        | None = None,
        alpha_identity_provider: AlphaIdentityProvider | None = None,
        alpha_memory_scope_root_dir: str | None = None,
        vertical_name: str,
        max_sessions: int = 256,
        idle_eviction_seconds: float | None = 60 * 30,
        clock: Callable[[], float] = time.monotonic,
        substrate_runtime: "OpenWeightResidualRuntime | None" = None,
        template_adapter: VerticalTemplateAdapter | None = None,
        templates_root_dir: pathlib.Path | None = None,
    ) -> None:
        self._factory = lifeform_factory
        self._alpha_factory = alpha_lifeform_factory
        self._alpha_identity_provider = alpha_identity_provider
        self._alpha_memory_scope_root_dir = alpha_memory_scope_root_dir
        self._vertical_name = vertical_name
        self._max_sessions = max_sessions
        self._idle_eviction_seconds = idle_eviction_seconds
        self._clock = clock
        self._substrate_runtime = substrate_runtime
        self._template_adapter = template_adapter
        self._templates_root_dir = templates_root_dir
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

    async def session_summaries(self) -> tuple[dict[str, object], ...]:
        async with self._lock:
            self._evict_idle_locked()
            return tuple(
                {
                    "session_id": sid,
                    "turn_count": len(entry.session.turn_summaries),
                    "user_id": (
                        self._alpha_identity_provider.user_for_session(sid)
                        if self._alpha_identity_provider is not None
                        else None
                    ),
                    "last_active_at": entry.last_active_at,
                }
                for sid, entry in self._sessions.items()
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def alpha_identity_provider(self) -> AlphaIdentityProvider | None:
        return self._alpha_identity_provider

    @property
    def template_adapter(self) -> VerticalTemplateAdapter | None:
        return self._template_adapter

    @property
    def templates_root_dir(self) -> pathlib.Path | None:
        return self._templates_root_dir

    def template_context_for(self, session_id: str) -> TemplateContext | None:
        """Return the per-session template context, if any.

        ``None`` means the session was created via the legacy
        factory path and is not save-as-template eligible.
        """
        entry = self._sessions.get(session_id)
        if entry is None:
            raise SessionNotFoundError(session_id)
        return entry.template_context

    async def create_session(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        template_id: str | None = None,
    ) -> LifeformSession:
        """Mint a new live session.

        ``template_id`` selects how the underlying Lifeform is built:

        * ``None`` (default) — use the vertical's legacy
          ``factory`` / ``alpha_factory``. No save-as-template
          context is captured for this session.
        * non-empty + adapter present — route through
          ``adapter.build_session_context_from_template`` so the
          session inherits the saved profile / drives, and a
          :class:`TemplateContext` is stashed for save-back.

        Sessions whose vertical declares a ``template_adapter`` AND
        no ``template_id`` was passed go through
        ``adapter.build_default_session_context`` so save-as-template
        works even for "fresh start" sessions. Verticals without an
        adapter always go through the legacy factory path.
        """
        async with self._lock:
            self._evict_idle_locked()
            self._evict_lru_to_capacity_locked(needed=1)

            sid = session_id or self._fresh_session_id()
            if sid in self._sessions:
                raise SessionAlreadyExistsError(sid)
            alpha_enabled = self._alpha_identity_provider is not None
            identity_provider: "IdentityProvider | None" = None
            if alpha_enabled:
                if user_id is None:
                    raise ValueError("alpha sessions require user_id")
                self._alpha_identity_provider.bind_session(
                    session_id=sid,
                    user_id=user_id,
                )
                identity_provider = self._alpha_identity_provider
                if self._alpha_factory is None and self._template_adapter is None:
                    raise ValueError("vertical does not support alpha identity")

            template_context: TemplateContext | None = None
            if template_id is not None and template_id.strip():
                if self._template_adapter is None:
                    raise TemplatesNotSupportedError(
                        f"vertical {self._vertical_name!r} does not support "
                        "templates (no adapter registered)"
                    )
                if self._templates_root_dir is None:
                    raise TemplatesNotSupportedError(
                        "service has no templates_root_dir configured"
                    )
                life, template_context = self._template_adapter.build_session_context_from_template(
                    root_dir=self._templates_root_dir,
                    template_id=template_id.strip(),
                    runtime=self._substrate_runtime,
                    identity_provider=identity_provider,
                    memory_scope_root_dir=self._alpha_memory_scope_root_dir,
                    alpha_enabled=alpha_enabled,
                )
            elif self._template_adapter is not None and self._templates_root_dir is not None:
                # Adapter-aware default path so save-as-template can capture
                # this session even though it started from the vertical
                # default profile.
                life, template_context = self._template_adapter.build_default_session_context(
                    runtime=self._substrate_runtime,
                    identity_provider=identity_provider,
                    memory_scope_root_dir=self._alpha_memory_scope_root_dir,
                    alpha_enabled=alpha_enabled,
                )
            elif alpha_enabled:
                life = self._alpha_factory(
                    self._substrate_runtime,
                    self._alpha_identity_provider,
                    self._alpha_memory_scope_root_dir,
                )
            else:
                life = self._factory(self._substrate_runtime)
            session = life.create_session(session_id=sid)
            self._sessions[sid] = _SessionEntry(
                session=session,
                last_active_at=self._clock(),
                template_context=template_context,
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


class TemplatesNotSupportedError(LookupError):
    """Raised when a template operation is requested on a vertical that
    does not register a :class:`VerticalTemplateAdapter`."""
