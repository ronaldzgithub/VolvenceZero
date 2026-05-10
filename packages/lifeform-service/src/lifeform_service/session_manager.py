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
from lifeform_service.substrate_registry import (
    SubstrateRuntimeProvider,
    fixed_provider_from_runtime,
)
from lifeform_service.templates import (
    TemplateContext,
    VerticalTemplateAdapter,
)
from lifeform_service.vertical_registry import (
    UnknownVerticalError,
    VerticalNotAlphaCapableError,
    VerticalRegistry,
)
from lifeform_service.verticals import VerticalSpec

if TYPE_CHECKING:
    from volvence_zero.memory import IdentityProvider
    from volvence_zero.substrate import OpenWeightResidualRuntime


@dataclass
class _SessionEntry:
    session: LifeformSession
    last_active_at: float
    # Records which vertical built this session's lifeform. Save-as-
    # template, listing handlers, and ops dashboards consult this so
    # they route through the right vertical's template adapter rather
    # than the service-level default. Empty string means a session
    # created via the legacy single-factory ctor path (DLaaS launcher),
    # in which case the manager's default vertical applies.
    vertical_name: str = ""
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
        lifeform_factory: Callable[["OpenWeightResidualRuntime | None"], Lifeform]
        | None = None,
        alpha_lifeform_factory: Callable[
            ["OpenWeightResidualRuntime | None", "IdentityProvider", str | None],
            Lifeform,
        ]
        | None = None,
        alpha_identity_provider: AlphaIdentityProvider | None = None,
        alpha_memory_scope_root_dir: str | None = None,
        vertical_name: str | None = None,
        max_sessions: int = 256,
        idle_eviction_seconds: float | None = 60 * 30,
        clock: Callable[[], float] = time.monotonic,
        substrate_runtime: "OpenWeightResidualRuntime | None" = None,
        substrate_provider: SubstrateRuntimeProvider | None = None,
        template_adapter: VerticalTemplateAdapter | None = None,
        templates_root_dir: pathlib.Path | None = None,
        vertical_registry: VerticalRegistry | None = None,
    ) -> None:
        """Construct a multi-vertical session manager.

        Two construction shapes are accepted:

        * **Multi-vertical (preferred)**: pass ``vertical_registry``.
          ``create_session(vertical_name=...)`` then resolves the
          factory per-call from the registry. The ``templates_root_dir``
          is the *service-level* root (e.g. ``artifacts/lifeform-templates``);
          per-vertical sub-directories are computed at call time
          from each vertical's ``template_subdir`` field.
        * **Single-vertical (legacy)**: pass ``lifeform_factory``,
          ``alpha_lifeform_factory``, ``vertical_name``,
          ``template_adapter`` and ``templates_root_dir`` (where
          the latter is already vertical-resolved). This is the
          shape ``dlaas-platform-launcher.InstanceManager`` uses;
          the constructor synthesises a one-entry registry under
          the hood so the rest of the class stays uniform.
        """
        if substrate_runtime is not None and substrate_provider is not None:
            raise ValueError(
                "SessionManager: pass substrate_runtime OR substrate_provider, not both"
            )
        if substrate_provider is None and substrate_runtime is not None:
            # Back-compat shim for callers that still pass a single
            # frozen runtime (notably ``dlaas-platform-launcher``'s
            # ``InstanceManager``). Wraps the runtime in a fixed
            # (non-swappable) provider so the rest of this class can
            # uniformly consume a provider.
            substrate_provider = fixed_provider_from_runtime(substrate_runtime)
        self._registry, self._templates_root_dir = _resolve_registry_and_templates_root(
            vertical_registry=vertical_registry,
            lifeform_factory=lifeform_factory,
            alpha_lifeform_factory=alpha_lifeform_factory,
            vertical_name=vertical_name,
            template_adapter=template_adapter,
            templates_root_dir=templates_root_dir,
            alpha_enabled=alpha_identity_provider is not None,
        )
        self._alpha_identity_provider = alpha_identity_provider
        self._alpha_memory_scope_root_dir = alpha_memory_scope_root_dir
        self._max_sessions = max_sessions
        self._idle_eviction_seconds = idle_eviction_seconds
        self._clock = clock
        self._substrate_provider = substrate_provider
        self._sessions: dict[str, _SessionEntry] = {}
        self._lock = asyncio.Lock()

    @property
    def vertical_registry(self) -> VerticalRegistry:
        return self._registry

    @property
    def substrate_runtime(self) -> "OpenWeightResidualRuntime | None":
        """Return the *current* shared substrate runtime, if any.

        Reads through the substrate provider on every access so
        callers (notably ``dlaas-platform-launcher.InstanceManager``)
        always see the post-swap runtime. We deliberately do NOT
        cache the runtime locally — caching would re-introduce the
        "second owner" anti-pattern the provider was built to avoid.
        """
        if self._substrate_provider is None:
            return None
        return self._substrate_provider.current_runtime

    @property
    def substrate_provider(self) -> SubstrateRuntimeProvider | None:
        return self._substrate_provider

    @property
    def vertical_name(self) -> str:
        """Default vertical name. Per-session vertical lives on
        :class:`_SessionEntry.vertical_name`; this property is only
        the service-level default for routes / clients that don't
        carry a session id."""
        return self._registry.default_name

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
        """Default vertical's template adapter (or None).

        Per-session lookups should go through
        :meth:`template_adapter_for` so save-as-template lands in
        the right vertical's adapter even when sessions were
        created against a non-default vertical.
        """
        return self._registry.default.template_adapter

    @property
    def templates_root_dir(self) -> pathlib.Path | None:
        """Default vertical's resolved templates directory (or None).

        Returns the *per-vertical* path (service root + vertical
        subdir). Per-session lookups should call
        :meth:`templates_dir_for` instead.
        """
        return self.templates_dir_for(self._registry.default_name)

    def template_adapter_for(
        self, vertical_name: str
    ) -> VerticalTemplateAdapter | None:
        """Return the template adapter for ``vertical_name``, if any."""
        spec = self._registry.get(vertical_name)
        return spec.template_adapter if spec is not None else None

    def templates_dir_for(self, vertical_name: str) -> pathlib.Path | None:
        """Return the per-vertical templates directory.

        Layout: ``<service_root>/<vertical.template_subdir or vertical.name>``.
        Returns ``None`` when either the service has no templates
        root configured or the vertical does not register a
        template adapter.

        Special case for legacy single-vertical construction: when
        ``vertical.template_subdir`` is the empty string (set by
        :func:`_resolve_registry_and_templates_root` when the caller
        passed an already-resolved per-vertical root), this returns
        the service-level root verbatim instead of appending an
        empty path component.
        """
        if self._templates_root_dir is None:
            return None
        spec = self._registry.get(vertical_name)
        if spec is None or spec.template_adapter is None:
            return None
        if spec.template_subdir == "":
            # Legacy resolved-root path; return as-is.
            return self._templates_root_dir
        subdir = (spec.template_subdir or spec.name).strip()
        if not subdir:
            return None
        return self._templates_root_dir / subdir

    def template_context_for(self, session_id: str) -> TemplateContext | None:
        """Return the per-session template context, if any.

        ``None`` means the session was created via the legacy
        factory path and is not save-as-template eligible.
        """
        entry = self._sessions.get(session_id)
        if entry is None:
            raise SessionNotFoundError(session_id)
        return entry.template_context

    def vertical_name_for(self, session_id: str) -> str:
        """Return which vertical built the session's lifeform.

        Falls back to the registry's default for legacy sessions
        whose entry didn't record a vertical (DLaaS path).
        """
        entry = self._sessions.get(session_id)
        if entry is None:
            raise SessionNotFoundError(session_id)
        return entry.vertical_name or self._registry.default_name

    async def create_session(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        template_id: str | None = None,
        vertical_name: str | None = None,
    ) -> LifeformSession:
        """Mint a new live session.

        ``vertical_name`` selects which vertical's factory builds
        the underlying Lifeform; ``None`` means use the registry's
        default. Unknown / alpha-incompatible verticals raise
        typed exceptions the route layer maps to HTTP 422.

        ``template_id`` selects how the Lifeform is built:

        * ``None`` (default) — adapter-aware default path when the
          chosen vertical has a template adapter (so save-as-template
          remains possible); otherwise the vertical's legacy factory.
        * non-empty + adapter present — route through
          ``adapter.build_session_context_from_template`` so the
          session inherits the saved profile / drives.
        """
        async with self._lock:
            self._evict_idle_locked()
            self._evict_lru_to_capacity_locked(needed=1)

            sid = session_id or self._fresh_session_id()
            if sid in self._sessions:
                raise SessionAlreadyExistsError(sid)

            chosen_name = (vertical_name or self._registry.default_name).strip()
            chosen_spec = self._registry.require(chosen_name)
            alpha_enabled = self._alpha_identity_provider is not None
            identity_provider: "IdentityProvider | None" = None
            if alpha_enabled:
                if user_id is None:
                    raise ValueError("alpha sessions require user_id")
                if (
                    chosen_spec.alpha_factory is None
                    and chosen_spec.template_adapter is None
                ):
                    raise VerticalNotAlphaCapableError(
                        f"vertical {chosen_name!r} has no alpha_factory or "
                        "template_adapter; pick a different vertical or "
                        "disable alpha mode"
                    )
                self._alpha_identity_provider.bind_session(
                    session_id=sid,
                    user_id=user_id,
                )
                identity_provider = self._alpha_identity_provider

            # Resolve runtime via the provider so post-swap session
            # creation always lands on the current model. Reading inside
            # the lock keeps concurrent swap attempts ordered relative
            # to session creation: a swap that holds the provider's
            # internal lock blocks here only if it was already
            # mid-flight.
            runtime = (
                self._substrate_provider.current_runtime
                if self._substrate_provider is not None
                else None
            )

            adapter = chosen_spec.template_adapter
            adapter_dir = self.templates_dir_for(chosen_name)

            template_context: TemplateContext | None = None
            if template_id is not None and template_id.strip():
                if adapter is None or adapter_dir is None:
                    raise TemplatesNotSupportedError(
                        f"vertical {chosen_name!r} does not support "
                        "templates (no adapter or no templates_root_dir)"
                    )
                life, template_context = adapter.build_session_context_from_template(
                    root_dir=adapter_dir,
                    template_id=template_id.strip(),
                    runtime=runtime,
                    identity_provider=identity_provider,
                    memory_scope_root_dir=self._alpha_memory_scope_root_dir,
                    alpha_enabled=alpha_enabled,
                )
            elif adapter is not None and adapter_dir is not None:
                # Adapter-aware default path so save-as-template can capture
                # this session even though it started from the vertical
                # default profile.
                life, template_context = adapter.build_default_session_context(
                    runtime=runtime,
                    identity_provider=identity_provider,
                    memory_scope_root_dir=self._alpha_memory_scope_root_dir,
                    alpha_enabled=alpha_enabled,
                )
            elif alpha_enabled:
                if chosen_spec.alpha_factory is None:
                    raise VerticalNotAlphaCapableError(
                        f"vertical {chosen_name!r} has no alpha_factory"
                    )
                life = chosen_spec.alpha_factory(
                    runtime,
                    self._alpha_identity_provider,
                    self._alpha_memory_scope_root_dir,
                )
            else:
                life = chosen_spec.factory(runtime)
            session = life.create_session(session_id=sid)
            self._sessions[sid] = _SessionEntry(
                session=session,
                last_active_at=self._clock(),
                vertical_name=chosen_name,
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

    def close_all_sessions_sync(self) -> int:
        """Drop every live session, returning the count.

        Wired as the substrate provider's pre-swap callback. Must be
        synchronous because :meth:`SubstrateRuntimeProvider.swap`
        runs the callback inside its own ``asyncio.Lock``-guarded
        critical section and does not await it.

        Important: we deliberately do **not** acquire ``self._lock``
        here. The provider's swap path serialises swaps, so we are
        already in a single-flight context; acquiring the manager
        lock from a non-async function would deadlock if any
        concurrent ``create_session`` is mid-flight on the same
        loop. The session dict mutation is a constant-time Python
        operation that releases the GIL only at well-defined
        boundaries — safe to perform from the swap caller.

        In-flight HTTP turns that already hold a reference to a
        session continue running against the (now-unreferenced)
        session object until they complete; their substrate
        reference points at the OLD runtime which Python's GC keeps
        alive until the last reference drops. Subsequent requests
        for a closed ``session_id`` raise ``SessionNotFoundError``.
        """
        count = len(self._sessions)
        if self._alpha_identity_provider is not None:
            self._alpha_identity_provider.clear_all_sessions()
        self._sessions.clear()
        return count

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


def _resolve_registry_and_templates_root(
    *,
    vertical_registry: VerticalRegistry | None,
    lifeform_factory: Callable[..., Lifeform] | None,
    alpha_lifeform_factory: Callable[..., Lifeform] | None,
    vertical_name: str | None,
    template_adapter: VerticalTemplateAdapter | None,
    templates_root_dir: pathlib.Path | None,
    alpha_enabled: bool,
) -> tuple[VerticalRegistry, pathlib.Path | None]:
    """Resolve the constructor's two shapes into a registry + root.

    The new shape (preferred) passes ``vertical_registry`` directly
    plus an *unresolved* service-level ``templates_root_dir``;
    per-vertical sub-dir computation lives in
    :meth:`SessionManager.templates_dir_for`.

    The legacy shape (DLaaS launcher / single-vertical tests) passes
    explicit ``lifeform_factory`` / ``alpha_lifeform_factory`` /
    ``vertical_name`` / ``template_adapter`` / ``templates_root_dir``
    where ``templates_root_dir`` may already be the per-vertical
    resolved path. We synthesise a one-entry registry with an empty
    ``template_subdir`` so :meth:`templates_dir_for` returns the
    legacy path verbatim instead of double-appending.
    """
    if vertical_registry is not None:
        if any(
            arg is not None
            for arg in (
                lifeform_factory,
                alpha_lifeform_factory,
                vertical_name,
                template_adapter,
            )
        ):
            raise ValueError(
                "SessionManager: vertical_registry is mutually exclusive "
                "with lifeform_factory / alpha_lifeform_factory / "
                "vertical_name / template_adapter (the legacy single-"
                "vertical args)"
            )
        return vertical_registry, templates_root_dir
    if lifeform_factory is None or vertical_name is None:
        raise ValueError(
            "SessionManager requires either vertical_registry OR "
            "(lifeform_factory + vertical_name)"
        )
    legacy_spec = VerticalSpec(
        name=vertical_name,
        factory=lifeform_factory,
        has_temporal_bootstrap=False,
        has_regime_bootstrap=False,
        alpha_factory=alpha_lifeform_factory,
        template_adapter=template_adapter,
        # Empty subdir: the caller already resolved the per-vertical
        # path into ``templates_root_dir`` (this is how DLaaS launcher
        # and single-vertical tests have always passed it). Empty
        # subdir tells :meth:`templates_dir_for` to return the root
        # verbatim instead of joining a sub-directory.
        template_subdir="",
    )
    return (
        VerticalRegistry.single(legacy_spec, alpha_enabled=alpha_enabled),
        templates_root_dir,
    )
