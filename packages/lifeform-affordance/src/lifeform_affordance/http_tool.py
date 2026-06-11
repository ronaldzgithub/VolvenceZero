"""HTTP tool plugin adapter — turn ``HttpPluginSpec``-shaped data into
runtime ``AffordanceDescriptor`` + ``AffordanceBackend`` pairs.

This module is the in-kernel counterpart to the ``http`` variant of
:class:`dlaas_platform_contracts.PluginManifest`. The contracts wheel
holds the wire-format schema; this module holds the runtime
representation. Conversion happens at the launcher / service boundary
(``lifeform-service``), keeping
``lifeform-affordance`` free of any ``dlaas-platform-*`` imports.

Two halves:

* **Blueprint shape** — small dataclasses
  :class:`HttpToolEndpoint` / :class:`HttpToolBlueprint` that mirror
  the fields the affordance layer actually needs. They deliberately
  drop transport-only fields (``timeout_seconds`` is per-endpoint,
  there is no ``call_timeout_seconds`` etc.); see
  :func:`blueprint_from_http_plugin` for the lossless converter.
* **Builder** — :func:`build_http_tool_descriptors` /
  :func:`build_http_tool_backends` produce one
  :class:`AffordanceDescriptor` / :class:`AffordanceBackend` per
  endpoint and :func:`register_http_blueprints` wires them into the
  given registry + invoker.

R10 carries through: every plugin endpoint MUST resolve a
:class:`HttpToolSafetyEntry` (loaded from the plugin's
``.vzbridge.yaml`` by :mod:`lifeform_service.plugin_attach`).
Endpoints without a matching manifest entry trip
:class:`MissingHttpPluginManifestEntryError` — safety is never
defaulted to "audit_required and pray".

The "entry-like" abstraction is a :class:`typing.Protocol`
(:class:`HttpToolSafetyEntry`) so the affordance wheel does not
import :mod:`lifeform_mcp_bridge` (which would create a cycle —
the bridge already depends on this wheel). The protocol matches
``lifeform_mcp_bridge.SafetyManifestEntry`` by structural typing,
so the launcher hands those instances straight through.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from volvence_zero.affordance import (
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceKind,
    AffordanceSafety,
)

from lifeform_affordance.invoker import AffordanceBackend, AffordanceInvoker
from lifeform_affordance.registry import AffordanceRegistry


_LOG = logging.getLogger("lifeform_affordance.http_tool")


_VALID_HTTP_METHODS: frozenset[str] = frozenset(
    {"GET", "POST", "PUT", "PATCH", "DELETE"}
)


class MissingHttpPluginManifestEntryError(KeyError):
    """An HTTP plugin endpoint has no matching ``.vzbridge.yaml`` entry.

    R10: safety must be reviewed per-tool. Building an
    :class:`AffordanceDescriptor` without a manifest-supplied
    ``safety_model`` / ``cost_model`` would silently default to
    "audit_required, no consent, no regime block"; we refuse instead.

    The caller (the launcher) catches this and surfaces a typed 503
    so the operator fixes the manifest before the contract goes
    live.
    """


@runtime_checkable
class HttpToolSafetyEntry(Protocol):
    """Structural shape of one per-tool manifest entry.

    Mirrors :class:`lifeform_mcp_bridge.SafetyManifestEntry`. We use
    ``Protocol`` instead of importing the bridge class because
    ``lifeform-mcp-bridge`` already depends on ``lifeform-affordance``
    — a reverse import here would create a cycle that breaks the
    wheel import-boundary contract test.

    Callers in ``lifeform-service`` hand
    ``SafetyManifestEntry`` instances straight through and they
    satisfy this protocol by structural typing.
    """

    @property
    def when_to_use(self) -> str: ...

    @property
    def when_not_to_use(self) -> str: ...

    @property
    def cost_model(self) -> AffordanceCost: ...

    @property
    def safety_model(self) -> AffordanceSafety: ...

    @property
    def affordance_tags(self) -> tuple[str, ...]: ...

    @property
    def excluded(self) -> bool: ...


# ---------------------------------------------------------------------------
# Blueprint dataclasses (in-kernel representation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HttpToolEndpoint:
    """One HTTP operation = one affordance.

    Fields mirror :class:`dlaas_platform_contracts.HttpEndpoint`
    except that the in-kernel layer enforces the same minimum
    selection-hint length as every other ``AffordanceDescriptor`` —
    short or empty hints get padded by the builder so descriptor
    construction does not crash.
    """

    name: str
    method: str
    path: str
    description: str = ""
    when_to_use: str = ""
    when_not_to_use: str = ""
    parameters_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("HttpToolEndpoint.name must be a non-empty string")
        if self.method.upper() not in _VALID_HTTP_METHODS:
            raise ValueError(
                f"HttpToolEndpoint.method must be one of "
                f"{sorted(_VALID_HTTP_METHODS)!r}; got {self.method!r}."
            )
        if not self.path.startswith("/"):
            raise ValueError(
                f"HttpToolEndpoint.path must start with '/'; "
                f"got {self.path!r}."
            )
        if self.timeout_seconds <= 0:
            raise ValueError(
                "HttpToolEndpoint.timeout_seconds must be > 0; got "
                f"{self.timeout_seconds!r}."
            )


@dataclass(frozen=True)
class HttpToolBlueprint:
    """All info needed to attach one HTTP plugin to a lifeform."""

    plugin_name: str
    plugin_version: str
    base_url: str
    endpoints: tuple[HttpToolEndpoint, ...]
    safety_manifest_path: str
    auth_header_templates: Mapping[str, str] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.plugin_name or not self.plugin_name.strip():
            raise ValueError("HttpToolBlueprint.plugin_name required")
        if not self.plugin_version or not self.plugin_version.strip():
            raise ValueError("HttpToolBlueprint.plugin_version required")
        if not self.base_url or not (
            self.base_url.startswith("http://")
            or self.base_url.startswith("https://")
        ):
            raise ValueError(
                f"HttpToolBlueprint.base_url must start with http:// or "
                f"https://; got {self.base_url!r}."
            )
        if not self.endpoints:
            raise ValueError(
                "HttpToolBlueprint.endpoints must contain at least one entry"
            )
        if not self.safety_manifest_path or not self.safety_manifest_path.strip():
            raise ValueError(
                f"HttpToolBlueprint(plugin_name={self.plugin_name!r}): "
                "safety_manifest_path required (R10 invariant)."
            )


def affordance_name_for(plugin_name: str, endpoint_name: str) -> str:
    """Stable affordance naming: ``<plugin>.<endpoint>``."""

    if not plugin_name.strip() or not endpoint_name.strip():
        raise ValueError(
            f"affordance_name_for: plugin_name={plugin_name!r}, "
            f"endpoint_name={endpoint_name!r} must both be non-empty."
        )
    return f"{plugin_name}.{endpoint_name}"


# ---------------------------------------------------------------------------
# Descriptor + backend construction
# ---------------------------------------------------------------------------


def _parameters_schema_with_default(
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Ensure the schema satisfies ``AffordanceDescriptor`` invariants.

    The descriptor requires a ``type`` key. Empty / missing
    parameters_schema becomes ``{"type": "object", "properties": {}}``.
    """

    schema_dict = dict(schema)
    if "type" not in schema_dict:
        schema_dict["type"] = "object"
        schema_dict.setdefault("properties", {})
    return schema_dict


def _output_schema_with_default(
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    schema_dict = dict(schema)
    if not schema_dict:
        return {"type": "object"}
    return schema_dict


@dataclass(frozen=True)
class HttpDescriptorBuild:
    """Result of building one HTTP plugin endpoint's descriptor.

    ``backend_eligible`` is ``False`` when the manifest entry declared
    ``excluded: true`` — the descriptor still appears in the registry
    (so audits / catalog renders see it) but no backend is bound,
    matching the MCP bridge's "register-but-skip-backend" pattern.
    """

    descriptor: AffordanceDescriptor
    entry: HttpToolSafetyEntry
    endpoint: HttpToolEndpoint
    backend_eligible: bool


def build_http_tool_descriptors(
    blueprint: HttpToolBlueprint,
    *,
    entries: Mapping[str, HttpToolSafetyEntry],
) -> tuple[HttpDescriptorBuild, ...]:
    """One ``AffordanceDescriptor`` per endpoint, deterministic order.

    ``entries`` is the per-endpoint manifest map (keyed by endpoint
    name = MCP-style tool_name). Every endpoint MUST have a matching
    entry; missing keys raise
    :class:`MissingHttpPluginManifestEntryError` so safety drift
    fails loud (R10).

    Returns a tuple of :class:`HttpDescriptorBuild` so
    :func:`register_http_blueprints` knows which descriptors should
    get a backend (``backend_eligible``) and which are excluded.
    """

    builds: list[HttpDescriptorBuild] = []
    for endpoint in blueprint.endpoints:
        entry = entries.get(endpoint.name)
        if entry is None:
            raise MissingHttpPluginManifestEntryError(
                f"HTTP plugin {blueprint.plugin_name!r}: endpoint "
                f"{endpoint.name!r} has no matching entry in the safety "
                f"manifest at {blueprint.safety_manifest_path!r}. "
                "Every endpoint must be reviewed; add a tools[] entry "
                "with name == endpoint name."
            )
        name = affordance_name_for(blueprint.plugin_name, endpoint.name)
        description = (
            endpoint.description
            or blueprint.description
            or f"HTTP plugin endpoint {endpoint.method} {blueprint.base_url}{endpoint.path}."
        )
        merged_tags = ("http_plugin", blueprint.plugin_name) + tuple(
            tag for tag in entry.affordance_tags
            if tag not in {"http_plugin", blueprint.plugin_name}
        )
        descriptor = AffordanceDescriptor(
            name=name,
            kind=AffordanceKind.TOOL,
            version=blueprint.plugin_version,
            display_name=f"{blueprint.plugin_name}.{endpoint.name}",
            description=description,
            when_to_use=entry.when_to_use,
            when_not_to_use=entry.when_not_to_use,
            parameters_schema=_parameters_schema_with_default(
                endpoint.parameters_schema
            ),
            output_schema=_output_schema_with_default(endpoint.output_schema),
            cost_model=entry.cost_model,
            safety_model=entry.safety_model,
            affordance_tags=merged_tags,
            source_path=blueprint.safety_manifest_path,
            excluded_from_runtime_selection=entry.excluded,
        )
        builds.append(
            HttpDescriptorBuild(
                descriptor=descriptor,
                entry=entry,
                endpoint=endpoint,
                backend_eligible=not entry.excluded,
            )
        )
    return tuple(builds)


_DEFAULT_ENV_RESOLVER: Callable[[str], str | None] = os.environ.get


def _resolve_header_templates(
    templates: Mapping[str, str],
    env_resolver: Callable[[str], str | None],
) -> dict[str, str]:
    """Expand ``${env:VAR}`` placeholders in header templates.

    Unresolved placeholders are dropped; the affordance backend
    surfaces a missing-secret error at call time rather than failing
    silently. Plain (non-templated) values pass through unchanged.
    """

    resolved: dict[str, str] = {}
    for key, raw in templates.items():
        value = raw
        for var in _extract_env_vars(raw):
            replacement = env_resolver(var)
            if replacement is None:
                _LOG.warning(
                    "http_tool: auth header %r references missing env var %r",
                    key,
                    var,
                )
                value = ""
                break
            value = value.replace(f"${{env:{var}}}", replacement)
        if value:
            resolved[key] = value
    return resolved


def _extract_env_vars(value: str) -> list[str]:
    """Find every ``${env:VAR}`` placeholder; returns ordered list."""

    out: list[str] = []
    idx = 0
    while True:
        start = value.find("${env:", idx)
        if start == -1:
            return out
        end = value.find("}", start + 6)
        if end == -1:
            return out
        out.append(value[start + 6 : end])
        idx = end + 1


def build_http_tool_backend(
    blueprint: HttpToolBlueprint,
    endpoint: HttpToolEndpoint,
    *,
    env_resolver: Callable[[str], str | None] = _DEFAULT_ENV_RESOLVER,
    http_client_factory: Callable[[], Any] | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> AffordanceBackend:
    """Build the async backend for one endpoint.

    The default uses ``aiohttp.ClientSession`` per call (created
    fresh, closed inside) so the backend has no long-lived resources
    to manage. Tests inject ``http_client_factory`` to swap the
    HTTP transport for a stub.

    ``extra_headers`` are caller-identity headers injected by the
    host at registration time (e.g. ``X-DLaaS-AI-ID`` /
    ``X-DLaaS-Session-ID`` from ``lifeform-service``). HTTP act
    surfaces are a trust boundary: a multi-tenant BFF cannot route a
    tool call to the right tenant from LLM-proposed parameters
    alone, so the transport carries WHO is acting. They merge over
    the manifest's ``auth_header_templates`` (identity wins on key
    collision — the host knows better than the manifest).
    """

    headers = _resolve_header_templates(
        blueprint.auth_header_templates, env_resolver
    )
    if extra_headers:
        headers.update(extra_headers)
    method = endpoint.method.upper()
    url = blueprint.base_url.rstrip("/") + endpoint.path
    timeout = endpoint.timeout_seconds

    async def _invoke(parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
        params_dict = dict(parameters)
        # Lazy import keeps ``aiohttp`` an optional dep at install
        # time even though the bridge wheel typically pulls it.
        if http_client_factory is None:
            import aiohttp  # noqa: PLC0415

            session_factory = lambda: aiohttp.ClientSession(  # noqa: E731
                timeout=aiohttp.ClientTimeout(total=timeout)
            )
        else:
            session_factory = http_client_factory
        session = session_factory()
        try:
            request_kwargs: dict[str, Any] = {"headers": headers}
            if method == "GET":
                request_kwargs["params"] = params_dict
            else:
                request_kwargs["json"] = params_dict
            async with session as client:
                async with client.request(method, url, **request_kwargs) as resp:
                    text = await resp.text()
                    try:
                        body: Any = await resp.json(content_type=None)
                    except Exception:  # pragma: no cover - non-JSON tools
                        body = text
                    return {
                        "status": resp.status,
                        "body": body,
                        "headers": dict(resp.headers),
                    }
        except Exception:
            # Re-raise: the invoker turns the exception into
            # ``AffordanceInvocationStatus.BACKEND_FAILED`` so the
            # session sees a structured failure record rather than
            # a Python traceback.
            raise

    return _invoke


def register_http_blueprints(
    *,
    registry: AffordanceRegistry,
    invoker: AffordanceInvoker,
    blueprints: Sequence[HttpToolBlueprint],
    entries_by_plugin: Mapping[str, Mapping[str, HttpToolSafetyEntry]],
    env_resolver: Callable[[str], str | None] = _DEFAULT_ENV_RESOLVER,
    http_client_factory: Callable[[], Any] | None = None,
    extra_headers: Mapping[str, str] | None = None,
) -> tuple[AffordanceDescriptor, ...]:
    """Register every blueprint's descriptors + backends.

    ``entries_by_plugin`` is the manifest payload keyed by
    ``plugin_name → {endpoint_name → HttpToolSafetyEntry}``. The
    caller (typically :mod:`lifeform_service.plugin_attach`) loads
    each plugin's ``.vzbridge.yaml`` and assembles this map.

    ``extra_headers`` (optional) is forwarded verbatim to every
    backend built here — see :func:`build_http_tool_backend`.

    Returns the flattened tuple of registered descriptors so callers
    can log / audit what was attached.

    Raises:

    * :class:`MissingHttpPluginManifestEntryError` when a plugin /
      endpoint has no manifest entry (R10).
    * :class:`AffordanceAlreadyRegisteredError` when an HTTP plugin's
      affordance name collides with an existing one.
    """

    all_descriptors: list[AffordanceDescriptor] = []
    for blueprint in blueprints:
        entries = entries_by_plugin.get(blueprint.plugin_name)
        if entries is None:
            raise MissingHttpPluginManifestEntryError(
                f"HTTP plugin {blueprint.plugin_name!r}: no manifest "
                "entries supplied; the launcher must call "
                "load_manifest(...) first and pass the result via "
                "entries_by_plugin."
            )
        builds = build_http_tool_descriptors(blueprint, entries=entries)
        registry.register_all(build.descriptor for build in builds)
        for build in builds:
            if not build.backend_eligible:
                _LOG.info(
                    "http_tool: skipping backend for %s (manifest entry "
                    "excluded=true)",
                    build.descriptor.name,
                )
                continue
            backend = build_http_tool_backend(
                blueprint,
                build.endpoint,
                env_resolver=env_resolver,
                http_client_factory=http_client_factory,
                extra_headers=extra_headers,
            )
            invoker.register_backend(build.descriptor.name, backend)
        all_descriptors.extend(build.descriptor for build in builds)
    return tuple(all_descriptors)


__all__ = [
    "HttpDescriptorBuild",
    "HttpToolBlueprint",
    "HttpToolEndpoint",
    "HttpToolSafetyEntry",
    "MissingHttpPluginManifestEntryError",
    "affordance_name_for",
    "build_http_tool_backend",
    "build_http_tool_descriptors",
    "register_http_blueprints",
]
