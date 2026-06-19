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

Serial-decode assumption (backend-specific): the above holds for the
``TransformersOpenWeightResidualRuntime`` (one blocking decode + one
context-managed LoRA activation at a time). The
``VLLMOpenWeightResidualRuntime`` is the concurrent backend: it batches
requests and attaches a per-request ``LoRARequest``, tracking the active
adapter in a ``contextvars.ContextVar`` so concurrent tenant turns route
their own persona without colliding. When deploying the vLLM backend the
serial-decode constraint above is relaxed (the per-request path is
``generate_for_request``); the persona-LoRA pool's nested-activation
guard is per-task, not process-global, on that backend.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lifeform_core import Lifeform, LifeformSession
from lifeform_service.alpha import (
    DEFAULT_ALPHA_TENANT_ID,
    AlphaIdentityProvider,
)
from lifeform_service.substrate_registry import (
    SubstrateRuntimeProvider,
    fixed_provider_from_runtime,
)
from lifeform_service.templates import (
    TemplateContext,
    VerticalTemplateAdapter,
)
from lifeform_service.default_mcp_bundle import with_default_mcp_bundle
from lifeform_service.plugin_attach import (
    apply_contract_policy_for_plugins,
    apply_plugins_to_lifeform_config,
    register_http_plugins_after_start,
)
from lifeform_service.vertical_registry import (
    VerticalNotAlphaCapableError,
    VerticalRegistry,
)
from lifeform_service.verticals import VerticalSpec

if TYPE_CHECKING:
    from lifeform_service.protocol_uptake import ProtocolUptakeService
    from volvence_zero.memory import IdentityProvider
    from volvence_zero.substrate import OpenWeightResidualRuntime


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() in ("1", "true", "True", "yes", "on")


def _legacy_single_layer_scope_opt_out() -> bool:
    """Whether the operator has explicitly opted out of two-layer scope.

    Two-layer ``{tenant}:{end_user}`` scope is the default (debt #46 /
    D3 / D22). An operator restores the previous single-layer
    ``scope_key == user_id`` behaviour with either:

    * ``VZ_LEGACY_SINGLE_LAYER_SCOPE=1`` (preferred, explicit), or
    * ``VZ_TWO_LAYER_SCOPE=0`` / ``=false`` (negating the legacy opt-in).

    A bare / unset / truthy ``VZ_TWO_LAYER_SCOPE`` does NOT opt out — the
    default is two-layer regardless of that flag now.
    """

    if _env_flag("VZ_LEGACY_SINGLE_LAYER_SCOPE"):
        return True
    raw = os.environ.get("VZ_TWO_LAYER_SCOPE")
    if raw is not None and raw.strip() in ("0", "false", "False", "no", "off"):
        return True
    return False


@dataclass
class TimeNodeSnapshot:
    """Restorable timepoint checkpoint metadata published by SessionManager."""

    time_node_id: str
    ai_id: str
    scope_key: str
    source_session_id: str
    as_of_ms: int
    captured_at_ms: int
    snapshot_version: str
    restore_status: str
    owner_slots: tuple[str, ...]
    evidence: dict[str, object]

    def to_json(self) -> dict[str, object]:
        return {
            "time_node_id": self.time_node_id,
            "ai_id": self.ai_id,
            "scope_key": self.scope_key,
            "source_session_id": self.source_session_id,
            "as_of_ms": self.as_of_ms,
            "captured_at_ms": self.captured_at_ms,
            "snapshot_version": self.snapshot_version,
            "restore_status": self.restore_status,
            "owner_slots": list(self.owner_slots),
            "evidence": dict(self.evidence),
        }


@dataclass
class _SessionEntry:
    session: LifeformSession
    lifeform: Lifeform
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
    # The end-user this session was created for (``user_id`` at create
    # time). Used by the dispatch layer to fail loud when a caller reuses
    # one ``session_id`` for a different end-user (which would otherwise
    # serve the first user's kernel state). Empty string = unknown /
    # legacy session created without a user_id.
    end_user_ref: str = ""
    historical_readonly: bool = False
    source_session_id: str = ""
    fork_as_of_ms: int | None = None
    time_node_id: str = ""
    fork_metadata: dict[str, object] | None = None


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
        protocol_uptake_service: "ProtocolUptakeService | None" = None,
        attach_default_mcp_bundle: bool = False,
        contract_plugins: tuple = (),
        contract_id: str = "",
        tool_policy_snapshot: dict | None = None,
        tenant_id: str = "",
        scope_strategy: str = "",
        instance_ai_id: str = "",
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
        # Tenant + memory scope strategy (DLaaS adopt). ``tenant_id`` is
        # the adopting tenant; ``scope_strategy`` (e.g.
        # ``"tenant_ai_end_user"``) plus env ``VZ_TWO_LAYER_SCOPE`` opt
        # the manager into two-layer ``{tenant}:{end_user}`` memory
        # scope keys instead of the legacy single-layer
        # ``scope_key == user_id``. Empty defaults keep the legacy
        # single-layer behaviour for standalone / closed-alpha lanes.
        self._tenant_id: str = tenant_id
        self._scope_strategy: str = scope_strategy
        # Optional figure-vertical bundle bound by the DLaaS adopt
        # path (``manager.bind_figure_bundle(bundle)``). When set,
        # every lifeform created by this manager has the bundle
        # attached via :meth:`Lifeform.bind_figure_bundle` right
        # after the vertical factory returns; the per-session
        # synthesizer clone then carries the bundle through to L1
        # / L3 / L4 enforcement (debt #22 closure).
        self._figure_bundle: object | None = None
        # Substrate adapter_policy gate (R10). The DLaaS adopt path
        # binds this from the contract's resolved substrate profile so
        # every lifeform this manager builds honours the policy at the
        # persona-LoRA activation site. Defaults True (permissive) so
        # the standalone service path keeps the additive behaviour.
        self._persona_lora_enabled: bool = True
        # Per-ai_id scoped persona-LoRA pool. Each SessionManager owns
        # one so two tenants that adopt different bundles for the same
        # figure_id never collide in the process-wide default pool
        # (which is last-register-wins). The bound bundle's LoRA is
        # registered here at bind time and the pool is handed to every
        # lifeform this manager builds. ``None`` until a figure bundle
        # is bound, so non-figure managers keep the default-pool path.
        self._persona_lora_pool: object | None = None
        # ``protocol-online-learning-active`` packet: when set,
        # every freshly-built ``Lifeform`` (regardless of which
        # vertical / template / alpha branch produced it) is
        # rebuilt via ``Lifeform.with_seed_protocols(...)`` with
        # the currently-approved ``BehaviorProtocol`` snapshot.
        # The kernel session's stable
        # ``ProtocolRegistryModule.load_protocol`` then auto-applies
        # each protocol's compiled hint / rule / knowledge / case
        # to the application owners AND keeps the protocol available
        # for online α/β PE-driven mixing across turns. This is the
        # load-bearing wiring that makes "upload PDF → approve → AI
        # behaves accordingly on the next session, and continues
        # learning from PE during the session" hold end-to-end.
        # Existing sessions are NOT mutated — the contract is
        # "approval applies to NEW sessions only", which the chat
        # UI surfaces verbatim.
        self._protocol_uptake_service = protocol_uptake_service
        # Service/product default: attach external/vz-bundle to freshly
        # created lifeforms unless the vertical already supplied explicit
        # MCP specs. Direct SessionManager tests / DLaaS legacy callers
        # leave this False and opt in at their construction boundary.
        self._attach_default_mcp_bundle = attach_default_mcp_bundle
        # debt #PluginFoundation: per-contract plugin manifests applied
        # to every fresh lifeform built by this manager. The DLaaS
        # launcher seeds this from ``ContractSpec.plugins`` so each
        # ai_id's sessions see exactly the plugins its tenant approved.
        # Empty tuple = legacy contract (default), in which case
        # ``create_session`` skips the plugin attach pass entirely.
        self._contract_plugins = tuple(contract_plugins)
        self._contract_id = contract_id
        self._tool_policy_snapshot = dict(tool_policy_snapshot or {})
        # The platform ai_id this manager serves (DLaaS launcher path;
        # empty for standalone service lanes). Injected into every HTTP
        # plugin backend as `X-DLaaS-AI-ID` so multi-tenant act
        # surfaces can attribute tool calls — identity travels on the
        # transport, never as an LLM-proposed parameter.
        self._instance_ai_id = instance_ai_id.strip()
        # Restorable Moonlight checkpoints owned by this runtime manager.
        # BFF apps may mirror the JSON, but do not own the checkpoint.
        self._time_nodes: dict[str, TimeNodeSnapshot] = {}

    def update_contract_policy(
        self,
        *,
        contract_id: str = "",
        plugins: tuple = (),
        tool_policy_snapshot: dict | None = None,
    ) -> None:
        """Refresh contract plugin set + tool policy for this manager.

        Used by the DLaaS adopt-retry / contract-update path (debt #16)
        so an already-acquired ``ai_id`` is NOT frozen at its
        first-adopt policy. New sessions always pick up the refreshed
        values via :meth:`create_session`; live sessions get the
        contract policy re-applied to their affordance registry here.

        Empty / None args are treated as "leave unchanged" so callers
        can refresh just the policy snapshot without resupplying the
        plugin set.
        """
        if contract_id:
            self._contract_id = contract_id
        if plugins:
            self._contract_plugins = tuple(plugins)
        if tool_policy_snapshot is not None:
            self._tool_policy_snapshot = dict(tool_policy_snapshot)
        if not self._contract_id:
            return
        contract_ids = [self._contract_id]
        if self._contract_id != "digital-employee":
            # The digital-employee BFF still tags runtime envelopes with
            # the stable compatibility contract_id; keep both in sync.
            contract_ids.append("digital-employee")
        for entry in self._sessions.values():
            for cid in contract_ids:
                apply_contract_policy_for_plugins(
                    entry.lifeform,
                    contract_id=cid,
                    plugins=self._contract_plugins,
                    tool_policy_snapshot=self._tool_policy_snapshot,
                )

    @property
    def vertical_registry(self) -> VerticalRegistry:
        return self._registry

    @property
    def figure_bundle(self) -> object | None:
        """The figure-vertical bundle bound to this manager (or None)."""
        return self._figure_bundle

    @property
    def persona_lora_pool(self) -> object | None:
        """The per-ai_id scoped persona-LoRA pool (None until a bundle
        with adapters is bound). Diagnostics / tests only."""
        return self._persona_lora_pool

    def bind_figure_bundle(
        self,
        bundle: object | None,
        *,
        persona_lora_enabled: bool = True,
    ) -> None:
        """Bind a figure-vertical artifact bundle for new sessions.

        Called from the DLaaS adopt path after
        ``lookup_figure_bundle(bundle_id=template.figure_artifact_id)``
        resolves a bundle. The manager records it; every subsequent
        lifeform created by :meth:`create_session` calls
        :meth:`lifeform_core.Lifeform.bind_figure_bundle` so the
        per-session synthesizer picks the bundle up at clone time.

        ``persona_lora_enabled`` carries the adopting contract's
        resolved substrate ``adapter_policy``: when ``False`` the
        bound lifeforms skip persona-LoRA activation at synthesis time
        (R10 — adapter usage is policy-gated). Defaults True.

        Pass ``None`` to clear the binding (e.g. on contract
        revocation). Existing sessions retain the bundle they were
        created with — change-of-mind invalidates the entire
        session, not the bundle on a live session.
        """
        self._figure_bundle = bundle
        self._persona_lora_enabled = persona_lora_enabled
        if bundle is None:
            self._persona_lora_pool = None
            return
        # Register the bundle's baked LoRA into this manager's scoped
        # pool (isolated per ai_id). When the policy forbids adapters
        # ``register_bundle_persona_lora`` is a no-op, so the scoped
        # pool stays empty and activation never fires.
        from volvence_zero.substrate import PersonaLoRAPool

        from lifeform_service.figure_bundle_store import (
            register_bundle_persona_lora,
        )

        scoped_pool = PersonaLoRAPool()
        register_bundle_persona_lora(
            bundle,
            pool=scoped_pool,
            persona_lora_enabled=persona_lora_enabled,
        )
        self._persona_lora_pool = scoped_pool

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
            evicted = self._evict_idle_locked()
            summaries = tuple(
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
        await self._shutdown_entries(evicted)
        return summaries

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def alpha_identity_provider(self) -> AlphaIdentityProvider | None:
        return self._alpha_identity_provider

    @property
    def alpha_memory_scope_root_dir(self) -> str | None:
        """Return the filesystem root used for alpha scoped memory, if enabled.

        DLaaS launcher-created per-``ai_id`` managers inherit this from the
        root service manager so multi-instance traffic does not silently lose
        the same per-user scoped-memory contract that the OpenAI-compatible
        single-manager path already has.
        """

        return self._alpha_memory_scope_root_dir

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

    def session_end_user(self, session_id: str) -> str | None:
        """Return the end-user this session was created for, or None.

        Returns ``None`` when the session does not exist or was created
        without a ``user_id`` (legacy). The dispatch layer uses this to
        reject reusing one ``session_id`` for a different end-user.
        """
        entry = self._sessions.get(session_id)
        if entry is None:
            return None
        return entry.end_user_ref or None

    def _two_layer_scope_enabled(self) -> bool:
        """Whether this manager binds two-layer ``{tenant}:{end_user}`` scope.

        Debt #46 / D3 / D22: two-layer scope is now the **default** for
        any manager whose adopting contract selects
        ``scope_strategy == "tenant_ai_end_user"`` (which is itself the
        default for :class:`MemoryPolicySelection`). The historical
        ``VZ_TWO_LAYER_SCOPE`` opt-in env is still honoured, but the gate
        is no longer required to turn two-layer *on*.

        Safety / no-silent-rekey: managers that carry an empty
        ``scope_strategy`` — notably the standalone closed-alpha service,
        which keeps single-layer ``scope_key == user_id`` on-disk memory —
        stay single-layer untouched. Operators with a two-layer-adopting
        contract that must keep the previous single-layer behaviour
        (because their durable entries were tagged under the old scope)
        set ``VZ_LEGACY_SINGLE_LAYER_SCOPE=1`` (or ``VZ_TWO_LAYER_SCOPE=0``)
        to opt out explicitly, so the migration is observable rather than
        silent. When they do, the legacy single-layer scope alias remains
        available via :func:`volvence_zero.memory.legacy_single_layer_scope`
        for read/delete reconciliation.
        """

        if self._scope_strategy != "tenant_ai_end_user":
            return False
        return not _legacy_single_layer_scope_opt_out()

    async def create_session(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        template_id: str | None = None,
        vertical_name: str | None = None,
        tenant_id: str | None = None,
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

        ``tenant_id`` (D22): optional per-call tenant override for the
        two-layer scope binding. When two-layer scope is active it takes
        precedence over the manager's adopting ``tenant_id`` and the
        closed-alpha default, letting a single multi-tenant front door
        (e.g. the OpenAI-compat bridge) partition end-user memory per
        tenant. Ignored in single-layer mode.
        """
        evicted: tuple[_SessionEntry, ...] = ()
        async with self._lock:
            evicted = (
                *self._evict_idle_locked(),
                *self._evict_lru_to_capacity_locked(needed=1),
            )

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
                # SessionManager defaults to the legacy single-layer
                # contract (``scope_key == user_id``) so closed-alpha
                # evidence / scoped-memory files on disk are not
                # silently re-keyed. When the adopting contract opts in
                # (``scope_strategy == "tenant_ai_end_user"``) AND the
                # operator sets ``VZ_TWO_LAYER_SCOPE=1``, bind through
                # the two-layer path so memory partitions per
                # ``{tenant}:{end_user}`` (debt #46 / #69).
                if self._two_layer_scope_enabled():
                    # Per-call ``tenant_id`` (e.g. plumbed from the
                    # OpenAI-compat bridge ``metadata.tenant_id``) wins;
                    # otherwise fall back to the manager's adopting
                    # tenant, then the closed-alpha default tenant.
                    effective_tenant = (
                        (tenant_id or "").strip()
                        or self._tenant_id
                        or DEFAULT_ALPHA_TENANT_ID
                    )
                    self._alpha_identity_provider.bind_session(
                        session_id=sid,
                        end_user_id=user_id,
                        tenant_id=effective_tenant,
                    )
                else:
                    self._alpha_identity_provider.bind_session_legacy_alias(
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
            if self._attach_default_mcp_bundle:
                life = with_default_mcp_bundle(life)
            if self._contract_plugins:
                life = apply_plugins_to_lifeform_config(
                    life, self._contract_plugins
                )
            life = self._inject_uptake_seed_protocols(life)
            if self._figure_bundle is not None:
                bind = getattr(life, "bind_figure_bundle", None)
                if callable(bind):
                    bind(self._figure_bundle)
            if not self._persona_lora_enabled:
                set_policy = getattr(life, "set_persona_lora_enabled", None)
                if callable(set_policy):
                    set_policy(False)
            if self._persona_lora_pool is not None:
                set_pool = getattr(life, "set_persona_lora_pool", None)
                if callable(set_pool):
                    set_pool(self._persona_lora_pool)
            await life.start()
            if self._contract_plugins:
                instance_headers: dict[str, str] = {}
                if self._instance_ai_id:
                    instance_headers["X-DLaaS-AI-ID"] = self._instance_ai_id
                if sid:
                    instance_headers["X-DLaaS-Session-ID"] = sid
                register_http_plugins_after_start(
                    life,
                    self._contract_plugins,
                    instance_headers=instance_headers or None,
                )
            contract_ids = [self._contract_id] if self._contract_id else []
            if self._contract_id and self._contract_id != "digital-employee":
                # The digital-employee BFF still sends the stable
                # compatibility contract_id on runtime envelopes while
                # adopt stores the generated platform contract_id.
                contract_ids.append("digital-employee")
            for contract_id in contract_ids:
                apply_contract_policy_for_plugins(
                    life,
                    contract_id=contract_id,
                    plugins=self._contract_plugins,
                    tool_policy_snapshot=self._tool_policy_snapshot,
                )
            session = life.create_session(session_id=sid)
            self._sessions[sid] = _SessionEntry(
                session=session,
                lifeform=life,
                last_active_at=self._clock(),
                vertical_name=chosen_name,
                template_context=template_context,
                end_user_ref=(user_id or ""),
            )
        await self._shutdown_entries(evicted)
        return session

    async def list_time_nodes(
        self,
        *,
        scope_key: str = "",
        session_id: str = "",
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int = 200,
    ) -> tuple[TimeNodeSnapshot, ...]:
        """Return restorable time-node readiness for this runtime."""
        if limit <= 0:
            return ()
        async with self._lock:
            if session_id and session_id in self._sessions:
                self._ensure_time_node_locked(
                    session_id=session_id,
                    scope_key=scope_key,
                    as_of_ms=until_ms or _epoch_ms(),
                )
            nodes = tuple(self._time_nodes.values())
        filtered = [
            node
            for node in nodes
            if (not scope_key or node.scope_key == scope_key)
            and (not session_id or node.source_session_id == session_id)
            and (since_ms is None or node.as_of_ms >= since_ms)
            and (until_ms is None or node.as_of_ms <= until_ms)
        ]
        filtered.sort(key=lambda node: node.as_of_ms, reverse=True)
        return tuple(filtered[:limit])

    async def get_time_node(self, time_node_id: str) -> TimeNodeSnapshot:
        async with self._lock:
            try:
                return self._time_nodes[time_node_id]
            except KeyError as exc:
                raise TimeNodeNotFoundError(time_node_id) from exc

    async def fork_session(
        self,
        *,
        source_session_id: str,
        fork_session_id: str,
        time_node_id: str,
        scope_key: str,
        mode: str,
        metadata: dict[str, object] | None = None,
        user_id: str | None = None,
    ) -> dict[str, object]:
        """Create a distinct historical-readonly session from a time node."""
        if mode != "historical_readonly":
            raise InvalidTemporalForkError("mode must be 'historical_readonly'")
        async with self._lock:
            source_entry = self._sessions.get(source_session_id)
            if source_entry is None:
                raise SessionNotFoundError(source_session_id)
            node = self._time_nodes.get(time_node_id)
            if node is None:
                node = self._ensure_time_node_locked(
                    session_id=source_session_id,
                    scope_key=scope_key,
                    as_of_ms=_coerce_as_of_ms(metadata),
                )
            if node.restore_status != "ready":
                raise SnapshotNotRestorableError(time_node_id, node.restore_status)
            if scope_key and node.scope_key != scope_key:
                raise ScopeNotAuthorizedError(scope_key)
            vertical_name = source_entry.vertical_name or self._registry.default_name
        fork_user_id = user_id or f"{node.scope_key}:moonlight:{node.as_of_ms}"
        session = await self.create_session(
            session_id=fork_session_id,
            user_id=fork_user_id,
            vertical_name=vertical_name,
        )
        async with self._lock:
            fork_entry = self._sessions[fork_session_id]
            fork_entry.historical_readonly = True
            fork_entry.source_session_id = source_session_id
            fork_entry.fork_as_of_ms = node.as_of_ms
            fork_entry.time_node_id = node.time_node_id
            fork_entry.fork_metadata = dict(metadata or {})
        marker = getattr(session, "set_historical_readonly", None)
        if callable(marker):
            marker(
                source_session_id=source_session_id,
                as_of_ms=node.as_of_ms,
                time_node_id=node.time_node_id,
            )
        return {
            "status": "ok",
            "ai_id": self._instance_ai_id,
            "source_session_id": source_session_id,
            "fork_session_id": fork_session_id,
            "time_node_id": node.time_node_id,
            "snapshot_version": node.snapshot_version,
            "mode": mode,
        }

    def is_historical_readonly(self, session_id: str) -> bool:
        entry = self._sessions.get(session_id)
        return bool(entry and entry.historical_readonly)

    async def get_session(self, session_id: str) -> LifeformSession:
        async with self._lock:
            evicted = self._evict_idle_locked()
            entry = self._sessions.get(session_id)
            if entry is None:
                raise SessionNotFoundError(session_id)
            entry.last_active_at = self._clock()
            session = entry.session
        await self._shutdown_entries(evicted)
        return session

    async def close_session(self, session_id: str) -> bool:
        async with self._lock:
            entry = self._sessions.pop(session_id, None)
        if entry is None:
            return False
        await entry.lifeform.shutdown()
        return True

    async def has_session(self, session_id: str) -> bool:
        async with self._lock:
            evicted = self._evict_idle_locked()
            found = session_id in self._sessions
        await self._shutdown_entries(evicted)
        return found

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

    def _ensure_time_node_locked(
        self,
        *,
        session_id: str,
        scope_key: str,
        as_of_ms: int,
    ) -> TimeNodeSnapshot:
        entry = self._sessions.get(session_id)
        if entry is None:
            raise SessionNotFoundError(session_id)
        normalized_scope = scope_key.strip() or entry.end_user_ref or session_id
        node_id = _time_node_id(
            session_id=session_id, scope_key=normalized_scope, as_of_ms=as_of_ms
        )
        existing = self._time_nodes.get(node_id)
        if existing is not None:
            return existing
        persisted = entry.session.persist_owners()
        active_slots = tuple(sorted(entry.session.latest_active_snapshots))
        owner_slots = tuple(dict.fromkeys((*active_slots, *persisted)))
        node = TimeNodeSnapshot(
            time_node_id=node_id,
            ai_id=self._instance_ai_id,
            scope_key=normalized_scope,
            source_session_id=session_id,
            as_of_ms=as_of_ms,
            captured_at_ms=_epoch_ms(),
            snapshot_version="tn.v1",
            restore_status="ready" if owner_slots else "not_restorable",
            owner_slots=owner_slots,
            evidence={
                "source_count": len(entry.session.turn_summaries),
                "latest_source_captured_at_ms": as_of_ms,
                "watermark": f"sha256:{uuid.uuid5(uuid.NAMESPACE_URL, node_id).hex}",
            },
        )
        self._time_nodes[node_id] = node
        return node

    def _inject_uptake_seed_protocols(self, life: Lifeform) -> Lifeform:
        """Append uptake-approved seed protocols onto ``life``.

        Returns ``life`` unchanged when no
        :class:`ProtocolUptakeService` is wired OR no protocol
        is currently approved. Otherwise returns a new
        :class:`Lifeform` whose ``LifeformConfig`` carries the
        vertical's own ``seed_protocols`` plus the approved
        ``BehaviorProtocol`` tuple.

        ``Lifeform.with_seed_protocols`` reuses the same
        ``substrate_runtime`` instance (carried in
        ``_init_kwargs``), so this does NOT trigger a second HF
        weight load — the only cost is reconstructing the
        controller-layer modules at session creation. Once the
        kernel session is built, its stable
        :class:`ProtocolRegistryModule.load_protocol` auto-applies
        the protocol's compiled artifacts into the application
        owner stores AND tracks each protocol for online α/β PE-
        driven mixing.
        """
        if self._protocol_uptake_service is None:
            return life
        approved = self._protocol_uptake_service.loaded_approved_snapshot()
        if not approved:
            return life
        return life.with_seed_protocols(approved)

    def _fresh_session_id(self) -> str:
        return f"sess-{uuid.uuid4().hex[:12]}"

    async def _shutdown_entries(self, entries: tuple[_SessionEntry, ...]) -> None:
        for entry in entries:
            await entry.lifeform.shutdown()

    def _evict_idle_locked(self) -> tuple[_SessionEntry, ...]:
        if self._idle_eviction_seconds is None:
            return ()
        cutoff = self._clock() - self._idle_eviction_seconds
        stale = [
            sid for sid, entry in self._sessions.items() if entry.last_active_at < cutoff
        ]
        evicted: list[_SessionEntry] = []
        for sid in stale:
            entry = self._sessions.pop(sid, None)
            if entry is not None:
                evicted.append(entry)
        return tuple(evicted)

    def _evict_lru_to_capacity_locked(self, *, needed: int) -> tuple[_SessionEntry, ...]:
        target = max(0, self._max_sessions - needed)
        if len(self._sessions) <= target:
            return ()
        # Sort ascending by last_active_at; evict oldest until under cap.
        ordered = sorted(
            self._sessions.items(), key=lambda kv: kv[1].last_active_at
        )
        evicted: list[_SessionEntry] = []
        while len(self._sessions) > target and ordered:
            sid, _entry = ordered.pop(0)
            entry = self._sessions.pop(sid, None)
            if entry is not None:
                evicted.append(entry)
        return tuple(evicted)


class SessionNotFoundError(LookupError):
    """Raised when a session_id is not in the manager."""


class TimeNodeNotFoundError(LookupError):
    """Raised when a time_node_id is not published by this manager."""


class SnapshotNotRestorableError(RuntimeError):
    """Raised when a visible time node cannot hydrate a fork."""

    def __init__(self, time_node_id: str, restore_status: str) -> None:
        super().__init__(
            f"time_node_id={time_node_id!r} restore_status={restore_status!r}"
        )
        self.time_node_id = time_node_id
        self.restore_status = restore_status


class ScopeNotAuthorizedError(PermissionError):
    """Raised when a fork asks for a scope outside the node boundary."""


class InvalidTemporalForkError(ValueError):
    """Raised for malformed historical fork requests."""


class SessionAlreadyExistsError(ValueError):
    """Raised when create_session is called with an explicit ID already in use."""


class TemplatesNotSupportedError(LookupError):
    """Raised when a template operation is requested on a vertical that
    does not register a :class:`VerticalTemplateAdapter`."""


def _epoch_ms() -> int:
    return int(time.time() * 1000)


def _coerce_as_of_ms(metadata: dict[str, object] | None) -> int:
    if not metadata:
        return _epoch_ms()
    raw = metadata.get("moonlight.as_of_ms") or metadata.get("as_of_ms")
    if raw is None:
        return _epoch_ms()
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise InvalidTemporalForkError(
            "metadata moonlight.as_of_ms must be an integer"
        ) from exc


def _time_node_id(*, session_id: str, scope_key: str, as_of_ms: int) -> str:
    raw = f"{scope_key}:{session_id}:{as_of_ms}"
    return f"tn_{uuid.uuid5(uuid.NAMESPACE_URL, raw).hex[:16]}"


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
