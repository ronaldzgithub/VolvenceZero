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

R10 carries through: each blueprint carries its
``safety_manifest_path`` and the resulting affordances inherit the
safety envelope a manifest-reader pass derives from it. Until that
reader is wired in (debt #PluginFoundation), the default safety model
is conservative (``audit_required=True``, no consent grants
auto-allowed); operators must approve loosening per-affordance.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from volvence_zero.affordance import (
    MIN_SELECTION_HINT_CHARS,
    AffordanceCost,
    AffordanceDescriptor,
    AffordanceKind,
    AffordanceLatencyClass,
    AffordanceMonetaryClass,
    AffordanceSafety,
)

from lifeform_affordance.invoker import AffordanceBackend, AffordanceInvoker
from lifeform_affordance.registry import AffordanceRegistry


_LOG = logging.getLogger("lifeform_affordance.http_tool")


_VALID_HTTP_METHODS: frozenset[str] = frozenset(
    {"GET", "POST", "PUT", "PATCH", "DELETE"}
)


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


def _pad_to_min_hint(text: str, fallback: str) -> str:
    if len(text) >= MIN_SELECTION_HINT_CHARS:
        return text
    if text:
        merged = f"{text} {fallback}"
    else:
        merged = fallback
    while len(merged) < MIN_SELECTION_HINT_CHARS:
        merged += " (HTTP plugin endpoint)"
    return merged


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


def build_http_tool_descriptors(
    blueprint: HttpToolBlueprint,
) -> tuple[AffordanceDescriptor, ...]:
    """One ``AffordanceDescriptor`` per endpoint, deterministic order."""

    descriptors: list[AffordanceDescriptor] = []
    for endpoint in blueprint.endpoints:
        name = affordance_name_for(blueprint.plugin_name, endpoint.name)
        description = (
            endpoint.description
            or blueprint.description
            or f"HTTP plugin endpoint {endpoint.method} {blueprint.base_url}{endpoint.path}."
        )
        when_to_use = _pad_to_min_hint(
            endpoint.when_to_use,
            (
                f"Call the {blueprint.plugin_name} plugin's "
                f"{endpoint.name!r} endpoint when the task requires "
                f"the external {endpoint.method} {endpoint.path} API."
            ),
        )
        when_not_to_use = _pad_to_min_hint(
            endpoint.when_not_to_use,
            (
                f"Do not call {blueprint.plugin_name}.{endpoint.name} for "
                "internal-only tasks, for sensitive operations without "
                "user confirmation, or when the boundary policy gates this plugin."
            ),
        )
        descriptors.append(
            AffordanceDescriptor(
                name=name,
                kind=AffordanceKind.TOOL,
                version=blueprint.plugin_version,
                display_name=f"{blueprint.plugin_name}.{endpoint.name}",
                description=description,
                when_to_use=when_to_use,
                when_not_to_use=when_not_to_use,
                parameters_schema=_parameters_schema_with_default(
                    endpoint.parameters_schema
                ),
                output_schema=_output_schema_with_default(endpoint.output_schema),
                cost_model=AffordanceCost(
                    latency_class=AffordanceLatencyClass.SLOW,
                    monetary_class=AffordanceMonetaryClass.LOW,
                ),
                safety_model=AffordanceSafety(audit_required=True),
                affordance_tags=("http_plugin", blueprint.plugin_name),
                source_path=blueprint.safety_manifest_path,
            )
        )
    return tuple(descriptors)


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
) -> AffordanceBackend:
    """Build the async backend for one endpoint.

    The default uses ``aiohttp.ClientSession`` per call (created
    fresh, closed inside) so the backend has no long-lived resources
    to manage. Tests inject ``http_client_factory`` to swap the
    HTTP transport for a stub.
    """

    headers = _resolve_header_templates(
        blueprint.auth_header_templates, env_resolver
    )
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
    env_resolver: Callable[[str], str | None] = _DEFAULT_ENV_RESOLVER,
    http_client_factory: Callable[[], Any] | None = None,
) -> tuple[AffordanceDescriptor, ...]:
    """Register every blueprint's descriptors + backends.

    Returns the flattened tuple of registered descriptors so callers
    can log / audit what was attached.

    Raises :class:`AffordanceAlreadyRegisteredError` when an HTTP
    plugin's affordance name collides with an existing one — that
    must surface loudly rather than silently shadow, because
    duplicated names break the contract whitelist.
    """

    all_descriptors: list[AffordanceDescriptor] = []
    for blueprint in blueprints:
        descriptors = build_http_tool_descriptors(blueprint)
        registry.register_all(descriptors)
        for descriptor, endpoint in zip(
            descriptors, blueprint.endpoints, strict=True
        ):
            backend = build_http_tool_backend(
                blueprint,
                endpoint,
                env_resolver=env_resolver,
                http_client_factory=http_client_factory,
            )
            invoker.register_backend(descriptor.name, backend)
        all_descriptors.extend(descriptors)
    return tuple(all_descriptors)


__all__ = [
    "HttpToolBlueprint",
    "HttpToolEndpoint",
    "affordance_name_for",
    "build_http_tool_backend",
    "build_http_tool_descriptors",
    "register_http_blueprints",
]
