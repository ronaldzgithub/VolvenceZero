"""``InstanceManager`` — multi-ai_id lifecycle on a shared substrate.

Slice 3.5 introduces the launcher: one Python process can host many
``ai_id`` instances simultaneously, each backed by its own
``LifeformSession``-managing :class:`SessionManager` but sharing a
single :class:`OpenWeightResidualRuntime` (one Qwen on one GPU).

Why a launcher rather than reusing ``lifeform-service``'s single
SessionManager:

* DLaaS public clients address the runtime by ``ai_id`` (not by
  ``session_id``). Multiple ai_ids may share the same vertical, but
  each one owns its own kernel state — sessions belonging to ai_A
  must not see memories or commitments belonging to ai_B.
* The launcher resolves ``runtime_template_id`` to a registered
  ``lifeform-service.verticals`` entry so multi-tenant DLaaS keeps
  using the same vertical builders the standalone service uses.
* ``_enforce_frozen_for_sharing`` (in
  ``lifeform-service.app``) is honoured at process startup; the
  launcher passes the already-validated runtime into every
  ``SessionManager`` it creates.

Resolution flow:

1. Adoption arrives carrying ``contract``.
2. The launcher reads ``contract.template_id`` from the registry,
   pulls the typed :class:`TemplateSpec`, and resolves the
   ``runtime_template_id`` to a :class:`VerticalSpec` via the
   ``vertical_resolver`` callback.
3. A fresh :class:`SessionManager` is constructed with the vertical's
   factory + the shared substrate runtime, and stored under
   ``{ai_id -> SessionManager}``.
4. On future ``/dlaas/instances/{ai_id}/interactions`` traffic, the
   api wheel pulls the SessionManager from the launcher.

Slice 5.x uses the launcher's snapshot to drive ops decisions
(pause / handoff). The cognitive state itself stays in the
SessionManager (i.e. in the kernel) — the launcher holds the
mapping only.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dlaas_platform_contracts import InstanceLifecycleState, InstanceStatus
from lifeform_service.session_manager import SessionManager
from lifeform_service.verticals import VerticalSpec

INSTANCE_MANAGER_APP_KEY = "dlaas_instance_manager"
"""``app[INSTANCE_MANAGER_APP_KEY]`` — the running InstanceManager."""


if TYPE_CHECKING:
    from volvence_zero.substrate import OpenWeightResidualRuntime


VerticalResolver = Callable[[str], VerticalSpec | None]
"""Maps a ``runtime_template_id`` string to a :class:`VerticalSpec`.

Returning ``None`` means "no vertical with that name is registered";
the launcher converts ``None`` into a typed ``LookupError``. The
default resolver in :func:`default_vertical_resolver` looks up
``runtime_template_id`` exactly against
``lifeform_service.verticals.discover_verticals()``.
"""


class InstanceNotFound(LookupError):
    """Raised when ``ai_id`` has no registered SessionManager."""


def default_vertical_resolver() -> VerticalResolver:
    """Return a resolver that maps ``runtime_template_id`` to a vertical
    by exact name match.

    Because ``runtime_template_id`` is the platform's bridge to the
    registered ``lifeform-service.verticals`` entries, this default
    works for every vertical that ships a stable name (``companion``,
    ``coding``, ``zhang_wuji``, …). Custom resolvers can layer alias
    tables in front of the default — e.g. mapping
    ``"dongfang_growth_advisor__job_seed_v1"`` to ``"companion"``
    until the vertical registry catches up.
    """

    from lifeform_service.verticals import discover_verticals

    cache: dict[str, VerticalSpec] | None = None

    def resolve(runtime_template_id: str) -> VerticalSpec | None:
        nonlocal cache
        if cache is None:
            cache = discover_verticals()
        return cache.get(runtime_template_id)

    return resolve


class InstanceManager:
    """``{ai_id -> SessionManager}`` registry shared by all ai_ids."""

    def __init__(
        self,
        *,
        vertical_resolver: VerticalResolver,
        substrate_runtime: "OpenWeightResidualRuntime | None" = None,
        max_sessions_per_instance: int = 256,
        idle_eviction_seconds: float | None = 60 * 30,
    ) -> None:
        self._vertical_resolver = vertical_resolver
        self._substrate_runtime = substrate_runtime
        self._instances: dict[str, SessionManager] = {}
        self._verticals: dict[str, str] = {}  # ai_id -> vertical_name
        self._lifecycle: dict[str, InstanceLifecycleState] = {}
        self._last_interaction_at_ms: dict[str, int] = {}
        self._last_wake_reason: dict[str, str] = {}
        self._last_sleep_reason: dict[str, str] = {}
        self._failure_reason: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._max_sessions_per_instance = max_sessions_per_instance
        self._idle_eviction_seconds = idle_eviction_seconds

    @property
    def substrate_runtime(self) -> "OpenWeightResidualRuntime | None":
        return self._substrate_runtime

    def has(self, ai_id: str) -> bool:
        return ai_id in self._instances

    def list_ai_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._instances.keys()))

    def vertical_for(self, ai_id: str) -> str:
        try:
            return self._verticals[ai_id]
        except KeyError as exc:
            raise InstanceNotFound(ai_id) from exc

    def get(self, ai_id: str) -> SessionManager:
        try:
            self.record_interaction(ai_id)
            return self._instances[ai_id]
        except KeyError as exc:
            raise InstanceNotFound(ai_id) from exc

    def record_interaction(self, ai_id: str) -> None:
        """Record runtime activity for status readouts.

        This is platform lifecycle telemetry only. It intentionally
        does not feed PE or memory; the kernel receives the actual
        event through the normal SessionManager dispatch path.
        """

        self._last_interaction_at_ms[ai_id] = _now_ms()

    async def acquire(
        self,
        *,
        ai_id: str,
        runtime_template_id: str,
    ) -> SessionManager:
        """Idempotently register ``ai_id`` with the matching vertical.

        If the ``ai_id`` already exists, the existing manager is
        returned unchanged (so adoption is replayable on retry). If
        the vertical cannot be resolved, raises ``LookupError`` so
        the caller can return a typed 422 / 503.
        """
        async with self._lock:
            if ai_id in self._instances:
                return self._instances[ai_id]
            spec = self._vertical_resolver(runtime_template_id)
            if spec is None:
                raise LookupError(
                    f"runtime_template_id={runtime_template_id!r} does not "
                    f"resolve to any registered lifeform-service vertical."
                )
            manager = SessionManager(
                lifeform_factory=spec.factory,
                alpha_lifeform_factory=spec.alpha_factory,
                vertical_name=spec.name,
                max_sessions=self._max_sessions_per_instance,
                idle_eviction_seconds=self._idle_eviction_seconds,
                substrate_runtime=self._substrate_runtime,
            )
            self._instances[ai_id] = manager
            self._verticals[ai_id] = spec.name
            self._lifecycle[ai_id] = InstanceLifecycleState.AWAKE
            self._last_wake_reason[ai_id] = "acquire"
            self._failure_reason.pop(ai_id, None)
            return manager

    async def wake(
        self,
        *,
        ai_id: str,
        runtime_template_id: str = "",
        reason: str = "on_demand",
    ) -> InstanceStatus:
        """Wake an adopted instance, optionally acquiring it first."""

        async with self._lock:
            if ai_id not in self._instances:
                if not runtime_template_id.strip():
                    raise InstanceNotFound(ai_id)
                spec = self._vertical_resolver(runtime_template_id)
                if spec is None:
                    self._lifecycle[ai_id] = InstanceLifecycleState.FAILED
                    self._failure_reason[ai_id] = (
                        f"runtime_template_id={runtime_template_id!r} does not resolve"
                    )
                    raise LookupError(self._failure_reason[ai_id])
                manager = SessionManager(
                    lifeform_factory=spec.factory,
                    alpha_lifeform_factory=spec.alpha_factory,
                    vertical_name=spec.name,
                    max_sessions=self._max_sessions_per_instance,
                    idle_eviction_seconds=self._idle_eviction_seconds,
                    substrate_runtime=self._substrate_runtime,
                )
                self._instances[ai_id] = manager
                self._verticals[ai_id] = spec.name
            self._lifecycle[ai_id] = InstanceLifecycleState.AWAKE
            self._last_wake_reason[ai_id] = reason
            self._failure_reason.pop(ai_id, None)
            return self._status_unlocked(ai_id)

    async def sleep(
        self,
        *,
        ai_id: str,
        reason: str = "idle_timeout",
        release_instance: bool = False,
    ) -> InstanceStatus:
        """Mark an instance asleep and optionally release its manager."""

        async with self._lock:
            if ai_id not in self._instances and ai_id not in self._lifecycle:
                raise InstanceNotFound(ai_id)
            status = self._status_unlocked(ai_id)
            self._lifecycle[ai_id] = InstanceLifecycleState.ASLEEP
            self._last_sleep_reason[ai_id] = reason
            if release_instance:
                self._instances.pop(ai_id, None)
            return InstanceStatus(
                ai_id=ai_id,
                lifecycle_state=InstanceLifecycleState.ASLEEP,
                vertical=status.vertical,
                session_count=0 if release_instance else status.session_count,
                max_sessions=status.max_sessions,
                last_interaction_at_ms=status.last_interaction_at_ms,
                last_wake_reason=status.last_wake_reason,
                last_sleep_reason=reason,
                failure_reason=status.failure_reason,
            )

    def status(self, ai_id: str) -> InstanceStatus:
        if ai_id not in self._instances and ai_id not in self._lifecycle:
            raise InstanceNotFound(ai_id)
        return self._status_unlocked(ai_id)

    async def release(self, ai_id: str) -> bool:
        """Drop the ``ai_id`` mapping; returns True if present.

        Active sessions are released by the SessionManager's own
        teardown path (close on every session). The launcher does
        NOT explicitly close substrate runtimes — those are owned
        by the api wheel at process scope.
        """
        async with self._lock:
            removed = self._instances.pop(ai_id, None)
            self._verticals.pop(ai_id, None)
            self._lifecycle[ai_id] = InstanceLifecycleState.ASLEEP
            return removed is not None

    def overview(self) -> tuple[dict[str, Any], ...]:
        """Read-only snapshot of every registered ai_id."""
        ids = set(self._instances) | set(self._lifecycle)
        return tuple(self._status_unlocked(ai_id).to_json() for ai_id in sorted(ids))

    def _status_unlocked(self, ai_id: str) -> InstanceStatus:
        manager = self._instances.get(ai_id)
        state = self._lifecycle.get(
            ai_id,
            InstanceLifecycleState.AWAKE if manager is not None else InstanceLifecycleState.ASLEEP,
        )
        return InstanceStatus(
            ai_id=ai_id,
            lifecycle_state=state,
            vertical=self._verticals.get(ai_id, "unknown"),
            session_count=manager.session_count() if manager is not None else 0,
            max_sessions=manager.max_sessions if manager is not None else self._max_sessions_per_instance,
            last_interaction_at_ms=self._last_interaction_at_ms.get(ai_id, 0),
            last_wake_reason=self._last_wake_reason.get(ai_id, ""),
            last_sleep_reason=self._last_sleep_reason.get(ai_id, ""),
            failure_reason=self._failure_reason.get(ai_id, ""),
        )


def _now_ms() -> int:
    return int(time.time() * 1000)


__all__ = [
    "INSTANCE_MANAGER_APP_KEY",
    "InstanceManager",
    "InstanceNotFound",
    "VerticalResolver",
    "default_vertical_resolver",
]
