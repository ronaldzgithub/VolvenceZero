"""Plugin manifest — the cross-layer schema for "application as tool bundle".

A :class:`PluginManifest` is the wire-format used end-to-end:

* business apps (``apps/growth-advisor`` etc.) declare them via the
  ``@volvence/dlaas-foundation`` TypeScript SDK on boot;
* the DLaaS control plane stores them per application and merges them
  into :class:`~dlaas_platform_contracts.ContractSpec.plugins` at adopt
  time;
* the launcher converts the ``mcp`` variant to an
  :class:`MCPServerSpec` for the lifeform pool, and the ``http``
  variant into runtime ``AffordanceDescriptor`` factories.

This module lives in the contracts wheel (the lowest tier) and is
**not allowed to import** any ``lifeform-*`` package. The conversion
helpers return plain dicts; the launcher / affordance side is
responsible for hydrating them into runtime classes. Keeping that
boundary clean is what lets the SDK / portal / control plane reason
about plugins without pulling the kernel into their import graph.

R10 carries through: HTTP plugins MUST declare a
``safety_manifest_path`` so safety is never derived from the external
capability surface alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


PluginKind = Literal["mcp", "http"]


_VALID_PLUGIN_KINDS: frozenset[str] = frozenset({"mcp", "http"})
_VALID_MCP_TRANSPORTS: frozenset[str] = frozenset({"stdio", "http"})
_VALID_RESTART_POLICIES: frozenset[str] = frozenset(
    {"never", "on_crash", "always"}
)
_VALID_HTTP_METHODS: frozenset[str] = frozenset(
    {"GET", "POST", "PUT", "PATCH", "DELETE"}
)


def _is_safe_identifier(value: str) -> bool:
    """Stable identifier rule shared with ``MCPServerSpec`` server names.

    Allowed: ASCII letters / digits / underscore / hyphen. Dot is
    deliberately forbidden because we use ``"<plugin>.<endpoint>"`` as
    the affordance name boundary later in the pipeline.
    """

    if not value:
        return False
    return all(ch.isalnum() or ch in {"_", "-"} for ch in value)


@dataclass(frozen=True)
class MCPPluginSpec:
    """Configuration for one external MCP server attached as a plugin.

    Field semantics mirror
    :class:`lifeform_mcp_bridge.MCPServerSpec` so the launcher can
    hydrate one directly via :meth:`to_mcp_server_kwargs`. The
    contracts wheel does not import the bridge — values are validated
    here and the bridge revalidates on construction, so a stale spec
    fails loudly on both sides.
    """

    transport: Literal["stdio", "http"] = "stdio"
    command: tuple[str, ...] = ()
    url: str = ""
    env: Mapping[str, str] = field(default_factory=dict)
    autostart: bool = True
    restart_policy: Literal["never", "on_crash", "always"] = "on_crash"
    call_timeout_seconds: float = 30.0
    enable_resources: bool = True
    enable_prompts: bool = False

    def __post_init__(self) -> None:
        if self.transport not in _VALID_MCP_TRANSPORTS:
            raise ValueError(
                f"MCPPluginSpec.transport must be one of "
                f"{sorted(_VALID_MCP_TRANSPORTS)!r}; got {self.transport!r}."
            )
        if self.transport == "stdio" and not self.command:
            raise ValueError(
                "MCPPluginSpec: stdio transport requires a non-empty "
                "'command' tuple."
            )
        if self.transport == "http" and not self.url.strip():
            raise ValueError(
                "MCPPluginSpec: http transport requires a non-empty 'url'."
            )
        if self.restart_policy not in _VALID_RESTART_POLICIES:
            raise ValueError(
                f"MCPPluginSpec.restart_policy must be one of "
                f"{sorted(_VALID_RESTART_POLICIES)!r}; got "
                f"{self.restart_policy!r}."
            )
        if self.call_timeout_seconds <= 0:
            raise ValueError(
                "MCPPluginSpec.call_timeout_seconds must be > 0; got "
                f"{self.call_timeout_seconds!r}."
            )
        if not isinstance(self.env, Mapping):
            raise TypeError(
                "MCPPluginSpec.env must be a Mapping; got "
                f"{type(self.env).__name__}."
            )
        for key, value in dict(self.env).items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(
                    "MCPPluginSpec.env keys and values must be strings; "
                    f"got key={key!r} value={value!r}."
                )

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "MCPPluginSpec":
        if not isinstance(data, Mapping):
            raise ValueError("MCPPluginSpec payload must be a JSON object")
        command_raw = data.get("command", ()) or ()
        if not isinstance(command_raw, (list, tuple)):
            raise ValueError("MCPPluginSpec.command must be a list of strings")
        command = tuple(str(item) for item in command_raw)
        env_raw = data.get("env", {}) or {}
        if not isinstance(env_raw, Mapping):
            raise ValueError("MCPPluginSpec.env must be a JSON object")
        env = {str(k): str(v) for k, v in env_raw.items()}
        return cls(
            transport=str(data.get("transport", "stdio") or "stdio"),  # type: ignore[arg-type]
            command=command,
            url=str(data.get("url", "") or ""),
            env=env,
            autostart=bool(data.get("autostart", True)),
            restart_policy=str(  # type: ignore[arg-type]
                data.get("restart_policy", "on_crash") or "on_crash"
            ),
            call_timeout_seconds=float(
                data.get("call_timeout_seconds", 30.0) or 30.0
            ),
            enable_resources=bool(data.get("enable_resources", True)),
            enable_prompts=bool(data.get("enable_prompts", False)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "transport": self.transport,
            "command": list(self.command),
            "url": self.url,
            "env": dict(self.env),
            "autostart": self.autostart,
            "restart_policy": self.restart_policy,
            "call_timeout_seconds": self.call_timeout_seconds,
            "enable_resources": self.enable_resources,
            "enable_prompts": self.enable_prompts,
        }


@dataclass(frozen=True)
class HttpEndpoint:
    """One HTTP operation = one affordance produced by the plugin.

    ``name`` becomes the leaf in the affordance name
    ``"<plugin>.<endpoint>"`` (must follow the same identifier rule as
    plugin / MCP server names). ``parameters_schema`` is the JSON
    Schema that gates affordance invocation; the launcher uses it both
    to drive the LLM tool descriptor and to validate inbound argument
    objects.
    """

    name: str
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "POST"
    path: str = ""
    description: str = ""
    when_to_use: str = ""
    parameters_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("HttpEndpoint.name must be a non-empty string")
        if not _is_safe_identifier(self.name):
            raise ValueError(
                f"HttpEndpoint.name={self.name!r}: only ASCII letters / "
                "digits / underscore / hyphen are allowed; '.' is "
                "reserved as the plugin/endpoint separator."
            )
        if self.method not in _VALID_HTTP_METHODS:
            raise ValueError(
                f"HttpEndpoint.method must be one of "
                f"{sorted(_VALID_HTTP_METHODS)!r}; got {self.method!r}."
            )
        if not isinstance(self.path, str) or not self.path.startswith("/"):
            raise ValueError(
                f"HttpEndpoint.path must start with '/'; got {self.path!r}."
            )
        if self.timeout_seconds <= 0:
            raise ValueError(
                "HttpEndpoint.timeout_seconds must be > 0; got "
                f"{self.timeout_seconds!r}."
            )
        if not isinstance(self.parameters_schema, Mapping):
            raise TypeError(
                "HttpEndpoint.parameters_schema must be a JSON Schema object"
            )
        if not isinstance(self.output_schema, Mapping):
            raise TypeError(
                "HttpEndpoint.output_schema must be a JSON Schema object"
            )

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "HttpEndpoint":
        if not isinstance(data, Mapping):
            raise ValueError("HttpEndpoint payload must be a JSON object")
        return cls(
            name=str(data.get("name", "") or ""),
            method=str(data.get("method", "POST") or "POST").upper(),  # type: ignore[arg-type]
            path=str(data.get("path", "/") or "/"),
            description=str(data.get("description", "") or ""),
            when_to_use=str(data.get("when_to_use", "") or ""),
            parameters_schema=dict(data.get("parameters_schema") or {}),
            output_schema=dict(data.get("output_schema") or {}),
            timeout_seconds=float(
                data.get("timeout_seconds", 30.0) or 30.0
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "path": self.path,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "parameters_schema": dict(self.parameters_schema),
            "output_schema": dict(self.output_schema),
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class HttpPluginSpec:
    """Plain HTTP/Webhook tool plugin spec.

    A plugin can expose one or many endpoints; each materialises as an
    :class:`AffordanceDescriptor` at runtime. Auth is declarative
    (header templates with ``${env:VAR}`` placeholders resolved by the
    launcher); the wire surface here does not embed secrets.
    """

    base_url: str
    endpoints: tuple[HttpEndpoint, ...]
    auth_header_templates: Mapping[str, str] = field(default_factory=dict)
    default_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not self.base_url or not self.base_url.strip():
            raise ValueError("HttpPluginSpec.base_url must be a non-empty string")
        if not (
            self.base_url.startswith("http://")
            or self.base_url.startswith("https://")
        ):
            raise ValueError(
                "HttpPluginSpec.base_url must start with http:// or "
                f"https://; got {self.base_url!r}."
            )
        if not self.endpoints:
            raise ValueError(
                "HttpPluginSpec.endpoints must contain at least one entry; "
                "a plugin with zero endpoints exposes no affordances."
            )
        seen: set[str] = set()
        for endpoint in self.endpoints:
            if not isinstance(endpoint, HttpEndpoint):
                raise TypeError(
                    "HttpPluginSpec.endpoints must contain HttpEndpoint "
                    f"instances; got {type(endpoint).__name__}."
                )
            if endpoint.name in seen:
                raise ValueError(
                    f"HttpPluginSpec.endpoints contains duplicate "
                    f"endpoint name {endpoint.name!r}."
                )
            seen.add(endpoint.name)
        if not isinstance(self.auth_header_templates, Mapping):
            raise TypeError(
                "HttpPluginSpec.auth_header_templates must be a Mapping"
            )
        for key, value in dict(self.auth_header_templates).items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(
                    "HttpPluginSpec.auth_header_templates keys and values "
                    f"must be strings; got key={key!r} value={value!r}."
                )
        if self.default_timeout_seconds <= 0:
            raise ValueError(
                "HttpPluginSpec.default_timeout_seconds must be > 0; got "
                f"{self.default_timeout_seconds!r}."
            )

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "HttpPluginSpec":
        if not isinstance(data, Mapping):
            raise ValueError("HttpPluginSpec payload must be a JSON object")
        endpoints_raw = data.get("endpoints") or ()
        if not isinstance(endpoints_raw, (list, tuple)):
            raise ValueError(
                "HttpPluginSpec.endpoints must be a list of endpoint objects"
            )
        endpoints = tuple(HttpEndpoint.from_json(item) for item in endpoints_raw)
        auth_raw = data.get("auth_header_templates", {}) or {}
        if not isinstance(auth_raw, Mapping):
            raise ValueError(
                "HttpPluginSpec.auth_header_templates must be a JSON object"
            )
        auth = {str(k): str(v) for k, v in auth_raw.items()}
        return cls(
            base_url=str(data.get("base_url", "") or ""),
            endpoints=endpoints,
            auth_header_templates=auth,
            default_timeout_seconds=float(
                data.get("default_timeout_seconds", 30.0) or 30.0
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "endpoints": [endpoint.to_json() for endpoint in self.endpoints],
            "auth_header_templates": dict(self.auth_header_templates),
            "default_timeout_seconds": self.default_timeout_seconds,
        }


@dataclass(frozen=True)
class PluginManifest:
    """Application-declared plugin: MCP server OR HTTP endpoint bundle.

    Exactly one of :attr:`mcp` / :attr:`http` is populated, selected by
    :attr:`kind`. The control plane stores manifests verbatim; the
    launcher hydrates them into :class:`MCPServerSpec` /
    :class:`AffordanceDescriptor` instances at adopt time.

    :attr:`declared_capabilities` is the set of *capability names*
    that the manifest contributes to the contract's
    ``tool_policy_snapshot.enabled_capabilities`` allowlist. For an
    MCP plugin this is what the bundle advertises ahead of
    ``tools/list``; for an HTTP plugin this is the
    ``"<name>.<endpoint>"`` set. Either way the contract's whitelist
    stays the SSOT, so the registry filter at session time still
    works exactly as today.

    Frozen / hashable so manifests can flow through caches and
    multi-step adopt flows without defensive copies.
    """

    name: str
    version: str
    kind: PluginKind
    safety_manifest_path: str
    description: str = ""
    declared_capabilities: tuple[str, ...] = ()
    mcp: MCPPluginSpec | None = None
    http: HttpPluginSpec | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("PluginManifest.name must be a non-empty string")
        if not _is_safe_identifier(self.name):
            raise ValueError(
                f"PluginManifest.name={self.name!r}: only ASCII letters / "
                "digits / underscore / hyphen are allowed; '.' is "
                "reserved as the plugin/endpoint separator."
            )
        if not self.version or not self.version.strip():
            raise ValueError("PluginManifest.version must be a non-empty string")
        if self.kind not in _VALID_PLUGIN_KINDS:
            raise ValueError(
                f"PluginManifest.kind must be one of "
                f"{sorted(_VALID_PLUGIN_KINDS)!r}; got {self.kind!r}."
            )
        if not self.safety_manifest_path or not self.safety_manifest_path.strip():
            raise ValueError(
                f"PluginManifest(name={self.name!r}): safety_manifest_path "
                "is required (R10: safety is never derived from the "
                "external capability surface)."
            )
        if self.kind == "mcp":
            if self.mcp is None:
                raise ValueError(
                    f"PluginManifest(name={self.name!r}, kind=mcp): the "
                    "'mcp' field is required."
                )
            if self.http is not None:
                raise ValueError(
                    f"PluginManifest(name={self.name!r}, kind=mcp): the "
                    "'http' field must be None."
                )
        elif self.kind == "http":
            if self.http is None:
                raise ValueError(
                    f"PluginManifest(name={self.name!r}, kind=http): the "
                    "'http' field is required."
                )
            if self.mcp is not None:
                raise ValueError(
                    f"PluginManifest(name={self.name!r}, kind=http): the "
                    "'mcp' field must be None."
                )
        for cap in self.declared_capabilities:
            if not isinstance(cap, str) or not cap.strip():
                raise ValueError(
                    "PluginManifest.declared_capabilities must contain "
                    f"non-empty strings; got {cap!r}."
                )

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "PluginManifest":
        if not isinstance(data, Mapping):
            raise ValueError("PluginManifest payload must be a JSON object")
        kind = str(data.get("kind", "") or "")
        mcp_raw = data.get("mcp")
        http_raw = data.get("http")
        mcp_spec = (
            MCPPluginSpec.from_json(mcp_raw)
            if isinstance(mcp_raw, Mapping)
            else None
        )
        http_spec = (
            HttpPluginSpec.from_json(http_raw)
            if isinstance(http_raw, Mapping)
            else None
        )
        capabilities_raw = data.get("declared_capabilities", ()) or ()
        if not isinstance(capabilities_raw, (list, tuple)):
            raise ValueError(
                "PluginManifest.declared_capabilities must be a list of strings"
            )
        capabilities = tuple(str(item) for item in capabilities_raw)
        return cls(
            name=str(data.get("name", "") or ""),
            version=str(data.get("version", "") or ""),
            kind=kind,  # type: ignore[arg-type]
            safety_manifest_path=str(
                data.get("safety_manifest_path", "") or ""
            ),
            description=str(data.get("description", "") or ""),
            declared_capabilities=capabilities,
            mcp=mcp_spec,
            http=http_spec,
        )

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "safety_manifest_path": self.safety_manifest_path,
            "description": self.description,
            "declared_capabilities": list(self.declared_capabilities),
        }
        if self.mcp is not None:
            payload["mcp"] = self.mcp.to_json()
        if self.http is not None:
            payload["http"] = self.http.to_json()
        return payload

    def to_mcp_server_kwargs(self) -> dict[str, Any]:
        """Return kwargs for ``MCPServerSpec(...)`` (kind == 'mcp' only).

        The launcher passes these into
        ``lifeform_mcp_bridge.MCPServerSpec`` directly. Wrapping the
        construction in the bridge keeps the contracts wheel free of
        the bridge import while still giving callers a single helper.
        """

        if self.kind != "mcp" or self.mcp is None:
            raise ValueError(
                f"PluginManifest(name={self.name!r}): to_mcp_server_kwargs "
                "is only valid for kind == 'mcp' manifests."
            )
        return {
            "name": self.name,
            "transport": self.mcp.transport,
            "command": tuple(self.mcp.command),
            "url": self.mcp.url,
            "env": dict(self.mcp.env),
            "safety_manifest_path": self.safety_manifest_path,
            "autostart": self.mcp.autostart,
            "restart_policy": self.mcp.restart_policy,
            "call_timeout_seconds": self.mcp.call_timeout_seconds,
            "enable_resources": self.mcp.enable_resources,
            "enable_prompts": self.mcp.enable_prompts,
        }


def merge_capability_whitelist(
    base_engine_tools: Mapping[str, Any],
    plugins: Sequence[PluginManifest],
) -> list[str]:
    """Merge ``engine_tools`` bool flags + plugin capabilities into the
    flat ``enabled_capabilities`` allowlist used by
    :class:`AffordanceRegistry.set_contract_policy`.

    Mirrors the existing ``_compute_tool_policy_snapshot`` semantics:

    * a bool ``engine_tools[name] == True`` enables ``name``;
    * a ``Mapping`` with ``{"enabled": True}`` also enables ``name``;
    * each plugin contributes every entry in
      :attr:`PluginManifest.declared_capabilities` unconditionally.

    The result preserves first-seen insertion order, deduplicated, so
    the rendered tool list stays deterministic across processes.
    """

    seen: dict[str, None] = {}
    for name, value in base_engine_tools.items():
        if isinstance(value, bool) and value:
            seen[name] = None
            continue
        if isinstance(value, Mapping) and bool(value.get("enabled", False)):
            seen[name] = None
    for plugin in plugins:
        for cap in plugin.declared_capabilities:
            seen[cap] = None
    return list(seen.keys())


def compute_plugin_tool_policy_snapshot(
    engine_tools: Mapping[str, Any],
    plugins: Sequence[PluginManifest],
) -> dict[str, Any]:
    """Same shape as ``_compute_tool_policy_snapshot`` but plugin-aware.

    Carries the legacy bool flags forward verbatim (so the existing
    ``engine_tools`` UI doesn't regress), unions the
    ``enabled_capabilities`` list with every plugin's
    :attr:`declared_capabilities`, and adds a ``plugins`` array with
    each plugin's name + version + kind for downstream diagnostics.
    """

    snapshot = dict(engine_tools)
    snapshot["enabled_capabilities"] = merge_capability_whitelist(
        engine_tools, plugins
    )
    snapshot["plugins"] = [
        {
            "name": plugin.name,
            "version": plugin.version,
            "kind": plugin.kind,
        }
        for plugin in plugins
    ]
    return snapshot


__all__ = [
    "HttpEndpoint",
    "HttpPluginSpec",
    "MCPPluginSpec",
    "PluginKind",
    "PluginManifest",
    "compute_plugin_tool_policy_snapshot",
    "merge_capability_whitelist",
]
