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
import os
import pathlib
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dlaas_platform_contracts import (
    InstanceLifecycleState,
    InstanceStatus,
    PluginManifest,
)
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

    aliases = {
        # Back-compat only. lifeform-service now registers
        # digital-employee.org.v0 / .twin.v0 as first-class vertical
        # names that wrap the companion v0 factory, so healthy
        # deployments resolve by exact name. The alias keeps older
        # installed wheels working during a rolling upgrade.
        "digital-employee.org.v0": "companion",
        "digital-employee.twin.v0": "companion",
        # novel-worlds BFFs speak in product runtime_template_id form
        # while the lifeform-service registry entry is named by the
        # vertical factory. Keep this alias explicit so wake/adoption
        # requests with runtime_template_id="novel-worlds.character.v0"
        # resolve to the CharacterTemplateAdapter-backed vertical that
        # scans /data/novel-bundles/novel-worlds.
        "novel-worlds.character.v0": "novel-worlds-character",
    }

    def resolve(runtime_template_id: str) -> VerticalSpec | None:
        nonlocal cache
        if cache is None:
            cache = discover_verticals()
        return cache.get(runtime_template_id) or cache.get(
            aliases.get(runtime_template_id, "")
        )

    return resolve


class InstanceManager:
    """``{ai_id -> SessionManager}`` registry shared by all ai_ids."""

    def __init__(
        self,
        *,
        vertical_resolver: VerticalResolver,
        substrate_runtime: "OpenWeightResidualRuntime | None" = None,
        alpha_identity_provider: Any | None = None,
        alpha_memory_scope_root_dir: str | None = None,
        attach_default_mcp_bundle: bool = False,
        max_sessions_per_instance: int = 256,
        idle_eviction_seconds: float | None = 60 * 30,
        templates_root_dir: "pathlib.Path | str | None" = None,
    ) -> None:
        self._vertical_resolver = vertical_resolver
        self._substrate_runtime = substrate_runtime
        self._alpha_identity_provider = alpha_identity_provider
        self._alpha_memory_scope_root_dir = alpha_memory_scope_root_dir
        self._attach_default_mcp_bundle = attach_default_mcp_bundle
        # NW9: service-level templates root (e.g. /data/novel-bundles).
        # Per-vertical SessionManagers built in acquire()/wake() resolve
        # the per-vertical subdir off this root so baked LifeformTemplate
        # JSON is loadable via create_session(template_id=...). When None,
        # template loading is unavailable and chat uses default profiles.
        self._templates_root_dir: pathlib.Path | None = (
            pathlib.Path(templates_root_dir)
            if templates_root_dir is not None
            else None
        )
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

    def _resolved_templates_dir_for(
        self, spec: VerticalSpec
    ) -> pathlib.Path | None:
        """NW9: per-vertical templates dir for the legacy single-vertical
        SessionManager built per ai_id.

        The legacy SessionManager constructor synthesises a one-entry
        registry with an empty ``template_subdir``, so its
        ``templates_dir_for`` returns the passed ``templates_root_dir``
        verbatim. We therefore resolve the per-vertical subdir HERE off
        the service-level root (e.g. ``/data/novel-bundles`` +
        ``novel-worlds`` -> ``/data/novel-bundles/novel-worlds``).

        Returns ``None`` when no service root is configured or the
        vertical does not register a template adapter, in which case the
        SessionManager is built without template support (default
        profiles only).
        """
        if self._templates_root_dir is None or spec.template_adapter is None:
            return None
        subdir = (spec.template_subdir or spec.name or "").strip()
        if not subdir:
            return self._templates_root_dir
        return self._templates_root_dir / subdir

    def _resolve_memory_root_for(self, *, ai_id: str, tenant_id: str) -> str | None:
        """Compute the per-ai_id memory root when opted in.

        Returns ``{base}/{tenant_id}/{ai_id}`` when
        ``VZ_PER_AI_MEMORY_ROOT=1`` and a base root is configured, so
        two ai_ids in one process never share on-disk memory. Otherwise
        returns the shared base root (legacy behaviour).
        """

        base = self._alpha_memory_scope_root_dir
        if base is None:
            return None
        opt_in = os.environ.get("VZ_PER_AI_MEMORY_ROOT", "").strip() in (
            "1",
            "true",
            "True",
        )
        if not opt_in:
            return base
        tenant_segment = tenant_id.strip() or "default-tenant"
        return str(pathlib.Path(base) / tenant_segment / ai_id)

    async def acquire(
        self,
        *,
        ai_id: str,
        runtime_template_id: str,
        plugins: tuple[PluginManifest, ...] = (),
        contract_id: str = "",
        tool_policy_snapshot: dict[str, Any] | None = None,
        tenant_id: str = "",
        scope_strategy: str = "",
        substrate_profile: str = "",
    ) -> SessionManager:
        """Idempotently register ``ai_id`` with the matching vertical.

        If the ``ai_id`` already exists, the existing manager is
        returned unchanged (so adoption is replayable on retry). If
        the vertical cannot be resolved, raises ``LookupError`` so
        the caller can return a typed 422 / 503.

        ``plugins`` is the contract-resolved plugin manifest set the
        DLaaS adopt path passes from
        :attr:`ContractSpec.plugins`; each freshly created
        :class:`SessionManager` carries it forward to every lifeform
        the manager later builds (debt #PluginFoundation).
        """
        async with self._lock:
            if ai_id in self._instances:
                # Adopt is replayable. Refresh the tool policy / plugin
                # set on the existing manager so a re-adopt with a
                # changed tool_policy_snapshot is NOT silently dropped
                # (debt #16: policy must not be frozen at first adopt).
                existing = self._instances[ai_id]
                if contract_id or tool_policy_snapshot is not None or plugins:
                    existing.update_contract_policy(
                        contract_id=contract_id,
                        plugins=plugins,
                        tool_policy_snapshot=tool_policy_snapshot,
                    )
                return existing
            spec = self._vertical_resolver(runtime_template_id)
            if spec is None:
                raise LookupError(
                    f"runtime_template_id={runtime_template_id!r} does not "
                    f"resolve to any registered lifeform-service vertical."
                )
            manager = SessionManager(
                lifeform_factory=spec.factory,
                alpha_lifeform_factory=spec.alpha_factory,
                alpha_identity_provider=self._alpha_identity_provider,
                alpha_memory_scope_root_dir=self._resolve_memory_root_for(
                    ai_id=ai_id, tenant_id=tenant_id
                ),
                vertical_name=spec.name,
                # NW9: wire template loading so create_session(template_id=)
                # reincarnates a baked LifeformTemplate (memory_checkpoint)
                # instead of falling back to the vertical default profile.
                template_adapter=spec.template_adapter,
                templates_root_dir=self._resolved_templates_dir_for(spec),
                max_sessions=self._max_sessions_per_instance,
                idle_eviction_seconds=self._idle_eviction_seconds,
                substrate_runtime=self._substrate_runtime,
                attach_default_mcp_bundle=self._attach_default_mcp_bundle,
                contract_plugins=plugins,
                contract_id=contract_id,
                tool_policy_snapshot=tool_policy_snapshot,
                tenant_id=tenant_id,
                scope_strategy=scope_strategy,
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
        plugins: tuple[PluginManifest, ...] = (),
        contract_id: str = "",
        tool_policy_snapshot: dict[str, Any] | None = None,
        tenant_id: str = "",
        scope_strategy: str = "",
        substrate_profile: str = "",
    ) -> InstanceStatus:
        """Wake an adopted instance, optionally acquiring it first.

        ``plugins`` is forwarded to a fresh :class:`SessionManager`
        when a wake call has to re-acquire an evicted instance (the
        DLaaS adopt path always passes the current
        :attr:`ContractSpec.plugins` so the rehydrated instance keeps
        the same plugin set the original adopt installed).
        """

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
                    alpha_identity_provider=self._alpha_identity_provider,
                    alpha_memory_scope_root_dir=self._resolve_memory_root_for(
                        ai_id=ai_id, tenant_id=tenant_id
                    ),
                    vertical_name=spec.name,
                    # NW9: same template wiring as acquire() so a wake that
                    # re-acquires an evicted instance keeps template loading.
                    template_adapter=spec.template_adapter,
                    templates_root_dir=self._resolved_templates_dir_for(spec),
                    max_sessions=self._max_sessions_per_instance,
                    idle_eviction_seconds=self._idle_eviction_seconds,
                    substrate_runtime=self._substrate_runtime,
                    attach_default_mcp_bundle=self._attach_default_mcp_bundle,
                    contract_plugins=plugins,
                    contract_id=contract_id,
                    tool_policy_snapshot=tool_policy_snapshot,
                    tenant_id=tenant_id,
                    scope_strategy=scope_strategy,
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
