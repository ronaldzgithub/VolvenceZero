"""Plugin "Preview" endpoint — dry-run rendering of one plugin tool.

Packet 7a of the DLaaS plugin foundation: org admins (typically via
the ``apps/dlaas-portal`` ``/security/plugins`` UI) can pick a
plugin endpoint, plug in test parameters, and see what the platform
*would* dispatch if a real session invoked it — without firing any
external HTTP request or starting any MCP subprocess.

Preview output by plugin kind:

* **HTTP**: ``method`` + composed ``url`` + redacted resolved
  ``headers`` + serialized ``body`` (query string for GET,
  ``application/json`` body otherwise).
* **MCP**: server-spec snapshot + the JSON-RPC ``tools/call`` frame
  that would be sent over stdio / HTTP transport.

R10 invariants stay intact:

* Tenant must have approved the application
  (:meth:`ApplicationStore.get_approval`).
* The manifest entry for the requested tool MUST exist (the load
  path is the same :func:`load_safety_manifest` the runtime uses, so a
  missing entry trips :class:`SafetyManifestSchemaError`).
* Secrets pulled from the platform's environment are redacted to
  ``first 4 chars + "***"`` so the operator can confirm the env var
  resolved without leaking the value into a portal UI.

The handler is intentionally read-only: it never writes audit rows
and never builds a real backend. A test asserts that
``aiohttp.ClientSession`` is not constructed during a preview call.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from aiohttp import web

from dlaas_platform_contracts import (
    HttpEndpoint as HttpEndpointSpec,
    PluginManifest,
)
from dlaas_platform_registry import (
    ApplicationNotFound,
    assert_tenant_id_matches,
    require_tenant_auth,
)
from volvence_zero.mcp_safety_manifest import (
    SafetyManifest,
    SafetyManifestEntry,
    SafetyManifestSchemaError,
    load_safety_manifest,
)


REDACT_SUFFIX = "***"
REDACT_VISIBLE_CHARS = 4


def _redact(value: str) -> str:
    """Return a leak-safe rendering of ``value`` for portal display.

    Strategy: keep the first ``REDACT_VISIBLE_CHARS`` characters so an
    operator can confirm the right secret resolved, then append
    ``REDACT_SUFFIX``. Short values are entirely masked.
    """

    if not value:
        return ""
    if len(value) <= REDACT_VISIBLE_CHARS:
        return REDACT_SUFFIX
    return f"{value[:REDACT_VISIBLE_CHARS]}{REDACT_SUFFIX}"


def _extract_env_vars(value: str) -> list[str]:
    """Find every ``${env:VAR}`` placeholder; ordered list."""

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


def _resolve_template_with_redaction(
    template: str,
    env_resolver,
) -> tuple[str, bool]:
    """Expand ``${env:VAR}`` placeholders, redacting resolved values.

    Returns ``(rendered, fully_resolved)``: ``fully_resolved`` is
    ``False`` when at least one env var was missing — the caller
    surfaces that as a warning rather than a hard error (preview is
    informational).
    """

    value = template
    fully_resolved = True
    for var in _extract_env_vars(template):
        raw = env_resolver(var)
        if raw is None:
            fully_resolved = False
            value = value.replace(f"${{env:{var}}}", f"<<missing:{var}>>")
            continue
        value = value.replace(f"${{env:{var}}}", _redact(raw))
    return value, fully_resolved


def _find_plugin(application_plugins, plugin_name: str) -> PluginManifest | None:
    for plugin in application_plugins:
        if plugin.name == plugin_name:
            return plugin
    return None


def _find_http_endpoint(
    plugin: PluginManifest, endpoint_name: str
) -> HttpEndpointSpec | None:
    if plugin.http is None:
        return None
    for endpoint in plugin.http.endpoints:
        if endpoint.name == endpoint_name:
            return endpoint
    return None


def _validate_against_schema(
    schema: Mapping[str, Any], parameters: Mapping[str, Any]
) -> list[str]:
    """Return a list of validation error strings (empty = valid).

    Same narrow JSON Schema subset as
    :func:`lifeform_affordance.invoker.validate_parameters` but
    operates directly on the schema dict so we don't need to build
    a full :class:`AffordanceDescriptor` just for preview. Keeps the
    error messages caller-friendly.
    """

    errors: list[str] = []
    if not isinstance(schema, Mapping):
        return ["parameters_schema must be a JSON object"]
    if schema.get("type") not in (None, "object"):
        return [f"parameters_schema.type must be 'object'; got {schema.get('type')!r}"]
    required = tuple(schema.get("required", ()) or ())
    properties: Mapping[str, Any] = schema.get("properties", {}) or {}
    for key in required:
        if key not in parameters:
            errors.append(f"missing required parameter {key!r}")
    if schema.get("additionalProperties") is False:
        for key in parameters.keys():
            if key not in properties:
                errors.append(
                    f"unknown parameter {key!r}; additionalProperties=False"
                )
    for key, value in parameters.items():
        prop_schema = properties.get(key)
        if not isinstance(prop_schema, Mapping):
            continue
        prop_type = prop_schema.get("type")
        if prop_type is None:
            continue
        if not _matches_type(value, prop_type):
            errors.append(
                f"parameter {key!r} expected type {prop_type!r}, got "
                f"{type(value).__name__}"
            )
    return errors


def _matches_type(value: Any, json_type: str) -> bool:
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, Mapping)
    if json_type == "null":
        return value is None
    return True  # unknown types pass; the affordance layer rejects elsewhere


def _safety_model_payload(entry: SafetyManifestEntry) -> dict[str, Any]:
    safety = entry.safety_model
    return {
        "requires_user_confirmation": safety.requires_user_confirmation,
        "irreversible": safety.irreversible,
        "requires_consent_grant": list(safety.requires_consent_grant),
        "blocked_in_regimes": list(safety.blocked_in_regimes),
        "audit_required": safety.audit_required,
    }


def _cost_model_payload(entry: SafetyManifestEntry) -> dict[str, Any]:
    cost = entry.cost_model
    return {
        "latency_class": cost.latency_class.value,
        "monetary_class": cost.monetary_class.value,
        "rate_limit_per_minute": cost.rate_limit_per_minute,
    }


def _http_preview(
    plugin: PluginManifest,
    endpoint: HttpEndpointSpec,
    parameters: Mapping[str, Any],
    env_resolver,
) -> dict[str, Any]:
    assert plugin.http is not None
    method = endpoint.method.upper()
    url = plugin.http.base_url.rstrip("/") + endpoint.path
    headers_redacted: dict[str, str] = {}
    missing_vars: list[str] = []
    for header_name, template in plugin.http.auth_header_templates.items():
        rendered, ok = _resolve_template_with_redaction(template, env_resolver)
        headers_redacted[header_name] = rendered
        if not ok:
            missing_vars.append(header_name)
    body: Any = None
    if method == "GET":
        body = {"params": dict(parameters)}
    else:
        body = {"json": dict(parameters)}
    return {
        "kind": "http",
        "method": method,
        "url": url,
        "headers": headers_redacted,
        "missing_env_vars_for_headers": missing_vars,
        "body": body,
    }


def _mcp_preview(
    plugin: PluginManifest,
    tool_name: str,
    parameters: Mapping[str, Any],
    env_resolver,
) -> dict[str, Any]:
    assert plugin.mcp is not None
    env_redacted: dict[str, str] = {}
    for env_key, env_template in plugin.mcp.env.items():
        rendered, _ok = _resolve_template_with_redaction(
            env_template, env_resolver
        )
        env_redacted[env_key] = rendered
    return {
        "kind": "mcp",
        "server_name": plugin.name,
        "transport": plugin.mcp.transport,
        "command": list(plugin.mcp.command),
        "url": plugin.mcp.url,
        "env": env_redacted,
        "jsonrpc_call": {
            "jsonrpc": "2.0",
            "id": "preview",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": dict(parameters),
            },
        },
    }


def _success(payload: dict[str, Any]) -> web.Response:
    return web.json_response({"status": "ok", **payload})


def _error(status: int, code: str, detail: str) -> web.Response:
    return web.json_response(
        {"status": "error", "error": code, "detail": detail},
        status=status,
    )


async def _handle_preview(request: web.Request) -> web.Response:
    """``POST /dlaas/applications/{application_id}/plugins/{plugin_name}/tools/{tool_name}:preview``.

    Auth: tenant. The tenant MUST have approved the application.
    Body: ``{"parameters": {...}}``.
    """

    # Local import avoids a cycle while still keeping the BFF facade
    # discoverable at module load time.
    from dlaas_platform_api.control_plane import CONTROL_PLANE_STORES_KEY

    tenant = await require_tenant_auth(request)
    application_id = request.match_info["application_id"]
    plugin_name = request.match_info["plugin_name"]
    tool_name = request.match_info["tool_name"]
    body_tenant_id = ""
    try:
        raw_body = await request.text()
    except Exception as exc:  # pragma: no cover - aiohttp internal
        return _error(400, "invalid_body", f"Could not read body: {exc}")
    if raw_body.strip():
        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            return _error(400, "invalid_json", f"Body is not valid JSON: {exc}")
        if not isinstance(parsed, dict):
            return _error(
                400, "invalid_envelope", "Top-level body must be a JSON object"
            )
        body_tenant_id = str(parsed.get("tenant_id", "") or "")
        parameters_raw = parsed.get("parameters", {}) or {}
    else:
        parameters_raw = {}
    if body_tenant_id:
        assert_tenant_id_matches(tenant, body_tenant_id)
    if not isinstance(parameters_raw, Mapping):
        return _error(
            400,
            "invalid_parameters",
            "'parameters' must be a JSON object",
        )
    parameters = dict(parameters_raw)

    stores = request.app[CONTROL_PLANE_STORES_KEY]
    try:
        application = await stores.applications.get(application_id)
    except ApplicationNotFound:
        return _error(404, "application_not_found", application_id)
    approval = await stores.applications.get_approval(
        tenant_id=tenant.tenant_id, application_id=application_id
    )
    if approval is None:
        return _error(
            403,
            "application_not_approved",
            (
                f"tenant_id={tenant.tenant_id!r} has not approved "
                f"application_id={application_id!r}; approve it first."
            ),
        )

    plugin = _find_plugin(application.plugins, plugin_name)
    if plugin is None:
        return _error(404, "plugin_not_found", plugin_name)

    # Resolve manifest entry for the requested tool.
    try:
        manifest: SafetyManifest = load_safety_manifest(
            path=plugin.safety_manifest_path,
            expected_server_name=plugin.name,
        )
    except SafetyManifestSchemaError as exc:
        return _error(503, "safety_manifest_invalid", str(exc))
    entry = manifest.lookup(tool_name)
    if entry is None:
        return _error(
            404,
            "tool_not_found",
            (
                f"plugin {plugin_name!r}: tool {tool_name!r} has no entry "
                f"in the safety manifest at {manifest.manifest_path!r}."
            ),
        )

    # Schema validation depends on plugin kind: HTTP plugins carry the
    # endpoint's parameters_schema; MCP plugins do not (the schema
    # comes from the server's tools/list, which we are not calling).
    validation_errors: list[str] = []
    parameters_schema: Mapping[str, Any] | None = None
    env_resolver = os.environ.get

    if plugin.kind == "http":
        endpoint = _find_http_endpoint(plugin, tool_name)
        if endpoint is None:
            return _error(
                404,
                "endpoint_not_found",
                f"plugin {plugin_name!r}: HTTP endpoint {tool_name!r} not declared.",
            )
        parameters_schema = endpoint.parameters_schema
        validation_errors = _validate_against_schema(
            parameters_schema, parameters
        )
        preview_payload = _http_preview(plugin, endpoint, parameters, env_resolver)
    elif plugin.kind == "mcp":
        # For MCP we don't know the tool's parameters_schema without
        # spawning the server (which would break dry-run). We surface
        # the manifest's entry tags + safety model and a JSON-RPC
        # frame; no schema validation possible.
        preview_payload = _mcp_preview(plugin, tool_name, parameters, env_resolver)
    else:  # pragma: no cover - schema enforced upstream
        return _error(
            400,
            "unsupported_plugin_kind",
            f"plugin {plugin_name!r}: unsupported kind {plugin.kind!r}.",
        )

    return _success(
        {
            "application_id": application_id,
            "plugin_name": plugin_name,
            "tool_name": tool_name,
            "parameters_valid": len(validation_errors) == 0,
            "validation_errors": validation_errors,
            "parameters_schema_available": parameters_schema is not None,
            "safety_model": _safety_model_payload(entry),
            "cost_model": _cost_model_payload(entry),
            "affordance_tags": list(entry.affordance_tags),
            "preview": preview_payload,
        }
    )


def attach_plugin_preview_routes(app: web.Application) -> None:
    """Register the preview route on the given aiohttp app.

    Idempotent: callers can invoke this from ``build_dlaas_app`` next
    to ``attach_control_plane_routes`` without worrying about
    duplicate registration (router state is initialised once per app
    in aiohttp). The route is intentionally separate from the
    control-plane router so a deployment that wants to disable
    preview (compliance reasons) can simply skip this call.
    """

    app.router.add_post(
        r"/dlaas/applications/{application_id}/plugins/{plugin_name}/tools/{tool_name}:preview",
        _handle_preview,
    )


__all__ = [
    "attach_plugin_preview_routes",
]
