"""Wire ``ContractSpec.plugins`` (DLaaS contracts wheel) into a Lifeform.

This is the single conversion point between the platform-tier
:class:`dlaas_platform_contracts.PluginManifest` schema and the
in-kernel objects the lifeform / affordance / MCP-bridge layers
consume. Putting it here keeps both downstream sides clean:

* ``lifeform-affordance`` knows nothing about
  ``dlaas-platform-contracts``;
* ``lifeform-core`` exposes a generic
  :meth:`Lifeform.ensure_affordance_registry` so MCP-only and
  HTTP-only and mixed lifeforms all converge on the same registry +
  invoker.

Two halves:

* :func:`apply_plugins_to_lifeform_config` runs *before*
  ``Lifeform.start()``: kind == ``mcp`` plugins get rolled into
  ``LifeformConfig.mcp_server_specs`` via
  :meth:`Lifeform.with_mcp_server_specs`.
* :func:`register_http_plugins_after_start` runs *after*
  ``Lifeform.start()``: kind == ``http`` plugins get translated into
  :class:`HttpToolBlueprint` and attached to the lifeform-scoped
  affordance registry + invoker.

Used by :class:`lifeform_service.session_manager.SessionManager` and
:class:`dlaas_platform_launcher.InstanceManager` so every adopted ai_id
ends up with the contract's plugin set live on its lifeform instance.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from lifeform_affordance.http_tool import (
    HttpToolBlueprint,
    HttpToolEndpoint,
    HttpToolSafetyEntry,
    register_http_blueprints,
)
from lifeform_core import Lifeform
from lifeform_mcp_bridge import MCPServerSpec
from lifeform_mcp_bridge.safety_manifest import (
    SafetyManifestEntry,
    load_manifest,
)


if TYPE_CHECKING:
    from dlaas_platform_contracts import PluginManifest


_LOG = logging.getLogger("lifeform_service.plugin_attach")


def load_http_plugin_manifest_entries(
    plugin: "PluginManifest",
) -> dict[str, SafetyManifestEntry]:
    """Load + validate one HTTP plugin's ``.vzbridge.yaml``.

    The manifest must declare ``server.name == plugin.name`` (the
    existing :func:`load_manifest` enforces that). Returns the
    flat ``{tool_name -> SafetyManifestEntry}`` map used by
    :func:`build_http_tool_descriptors`.

    Raises :class:`MCPSafetyManifestSchemaError` on any of the loader's
    failures (missing file / bad schema / server.name mismatch /
    short hints / illegal enums). The launcher catches the typed
    exception and surfaces a 503 so the operator fixes the
    manifest before the contract goes live.
    """

    manifest = load_manifest(
        path=plugin.safety_manifest_path,
        expected_server_name=plugin.name,
    )
    return dict(manifest.tool_entries)


def _entries_by_plugin(
    plugins: Sequence["PluginManifest"],
) -> dict[str, Mapping[str, HttpToolSafetyEntry]]:
    """Load each HTTP plugin's manifest entries into one keyed map."""

    out: dict[str, Mapping[str, HttpToolSafetyEntry]] = {}
    for plugin in plugins:
        if plugin.kind != "http":
            continue
        out[plugin.name] = load_http_plugin_manifest_entries(plugin)
    return out


def apply_plugins_to_lifeform_config(
    lifeform: Lifeform,
    plugins: Sequence["PluginManifest"],
) -> Lifeform:
    """Return a clone of ``lifeform`` with MCP plugin specs attached.

    Pure: returns a new ``Lifeform`` instance via
    :meth:`Lifeform.with_mcp_server_specs`. HTTP plugins are not
    touched here (they need the registry, which only exists after
    ``start()``).
    """

    mcp_specs = mcp_server_specs_from_plugins(plugins)
    if not mcp_specs:
        return lifeform
    return lifeform.with_mcp_server_specs(mcp_specs)


def mcp_server_specs_from_plugins(
    plugins: Sequence["PluginManifest"],
) -> tuple[MCPServerSpec, ...]:
    """Translate kind == 'mcp' plugins into bridge specs.

    Plugins of other kinds are skipped silently. Each manifest is
    revalidated by :class:`MCPServerSpec.__post_init__` so a bogus
    plugin fails before it can crash the pool.
    """

    out: list[MCPServerSpec] = []
    for manifest in plugins:
        if manifest.kind != "mcp" or manifest.mcp is None:
            continue
        kwargs = manifest.to_mcp_server_kwargs()
        out.append(MCPServerSpec(**kwargs))
    return tuple(out)


def http_blueprints_from_plugins(
    plugins: Sequence["PluginManifest"],
) -> tuple[HttpToolBlueprint, ...]:
    """Translate kind == 'http' plugins into in-kernel blueprints."""

    out: list[HttpToolBlueprint] = []
    for manifest in plugins:
        if manifest.kind != "http" or manifest.http is None:
            continue
        endpoints = tuple(
            HttpToolEndpoint(
                name=endpoint.name,
                method=endpoint.method,
                path=endpoint.path,
                description=endpoint.description,
                when_to_use=endpoint.when_to_use,
                when_not_to_use="",  # surface defaults via http_tool padding
                parameters_schema=endpoint.parameters_schema,
                output_schema=endpoint.output_schema,
                timeout_seconds=endpoint.timeout_seconds,
            )
            for endpoint in manifest.http.endpoints
        )
        out.append(
            HttpToolBlueprint(
                plugin_name=manifest.name,
                plugin_version=manifest.version,
                base_url=manifest.http.base_url,
                endpoints=endpoints,
                safety_manifest_path=manifest.safety_manifest_path,
                auth_header_templates=manifest.http.auth_header_templates,
                description=manifest.description,
            )
        )
    return tuple(out)


def register_http_plugins_after_start(
    lifeform: Lifeform,
    plugins: Sequence["PluginManifest"],
) -> None:
    """Register every HTTP plugin's affordances on a started lifeform.

    No-op when ``plugins`` contains zero HTTP-kind manifests.
    Otherwise lazily creates the affordance registry + invoker via
    :meth:`Lifeform.ensure_affordance_registry` and attaches each
    blueprint.

    Idempotency: re-registering the same plugin against the same
    lifeform raises ``AffordanceAlreadyRegisteredError`` (deliberate
    fail-loud, matches the contract policy invariants). Callers
    that need hot-reload semantics should reconstruct the lifeform.
    """

    blueprints = http_blueprints_from_plugins(plugins)
    if not blueprints:
        return
    entries_by_plugin = _entries_by_plugin(plugins)
    registry, invoker = lifeform.ensure_affordance_registry()
    descriptors = register_http_blueprints(
        registry=registry,
        invoker=invoker,
        blueprints=blueprints,
        entries_by_plugin=entries_by_plugin,
    )
    _LOG.info(
        "plugin_attach: registered %d HTTP affordance(s) from %d plugin(s)",
        len(descriptors),
        len(blueprints),
    )


def apply_contract_policy_for_plugins(
    lifeform: Lifeform,
    *,
    contract_id: str,
    plugins: Sequence["PluginManifest"],
    extra_allowed_names: Sequence[str] = (),
) -> None:
    """Push the union of plugin-contributed names + extras into the
    lifeform's affordance registry as a contract whitelist.

    Mirrors what
    :class:`AffordanceRegistry.set_contract_policy` does for
    legacy ``engine_tools``-only contracts, with plugin endpoints
    folded in so the session-time filter respects them.
    """

    if not contract_id.strip():
        return
    allowed: list[str] = list(extra_allowed_names)
    for manifest in plugins:
        allowed.extend(manifest.declared_capabilities)
    if not allowed:
        return
    registry, _invoker = lifeform.ensure_affordance_registry()
    registry.set_contract_policy(
        contract_id=contract_id,
        allowed_affordance_names=allowed,
    )


__all__ = [
    "apply_contract_policy_for_plugins",
    "apply_plugins_to_lifeform_config",
    "http_blueprints_from_plugins",
    "load_http_plugin_manifest_entries",
    "mcp_server_specs_from_plugins",
    "register_http_plugins_after_start",
]
