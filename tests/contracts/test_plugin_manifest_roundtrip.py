"""Contract test: PluginManifest schema + ContractSpec.plugins.

Validates the Packet 1 contract surface of the DLaaS plugin
foundation rollout:

1. ``PluginManifest`` enforces R10 (``safety_manifest_path`` is
   required) and the kind ↔ payload one-of invariant.
2. ``MCPPluginSpec`` / ``HttpPluginSpec`` field-level validation
   matches the corresponding ``MCPServerSpec`` rules in the bridge.
3. ``PluginManifest.to_mcp_server_kwargs`` produces kwargs that
   ``MCPServerSpec`` can be instantiated from directly.
4. ``ContractSpec`` carries ``plugins=()`` for legacy JSON payloads
   so prior data deserialises unchanged (rollback contract).
5. ``compute_plugin_tool_policy_snapshot`` unions engine_tools bool
   flags + plugin ``declared_capabilities`` into a deduplicated
   ``enabled_capabilities`` allowlist and appends a ``plugins``
   diagnostic array.
"""

from __future__ import annotations

import pytest

from dlaas_platform_contracts import (
    ContractSpec,
    HttpEndpoint,
    HttpPluginSpec,
    MCPPluginSpec,
    PluginManifest,
    compute_plugin_tool_policy_snapshot,
)


# ---------------------------------------------------------------------------
# PluginManifest core invariants
# ---------------------------------------------------------------------------


def _mcp_plugin(name: str = "wechat_readonly") -> PluginManifest:
    return PluginManifest(
        name=name,
        version="1.0.0",
        kind="mcp",
        safety_manifest_path="manifests/wechat.vzbridge.yaml",
        declared_capabilities=(f"{name}.read_messages",),
        mcp=MCPPluginSpec(
            transport="stdio",
            command=("python", "-m", "wechat_bundle.server"),
        ),
    )


def _http_plugin(name: str = "weather") -> PluginManifest:
    return PluginManifest(
        name=name,
        version="0.3.1",
        kind="http",
        safety_manifest_path="manifests/weather.vzbridge.yaml",
        declared_capabilities=(f"{name}.current", f"{name}.forecast"),
        http=HttpPluginSpec(
            base_url="https://api.weather.example.com",
            endpoints=(
                HttpEndpoint(
                    name="current",
                    method="GET",
                    path="/v1/current",
                    description="Look up current weather for a city.",
                    parameters_schema={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
                HttpEndpoint(
                    name="forecast",
                    method="POST",
                    path="/v1/forecast",
                    parameters_schema={"type": "object"},
                ),
            ),
            auth_header_templates={"x-api-key": "${env:WEATHER_API_KEY}"},
        ),
    )


def test_mcp_plugin_manifest_roundtrip_preserves_fields() -> None:
    manifest = _mcp_plugin()
    payload = manifest.to_json()
    revived = PluginManifest.from_json(payload)
    assert revived == manifest


def test_http_plugin_manifest_roundtrip_preserves_endpoints() -> None:
    manifest = _http_plugin()
    payload = manifest.to_json()
    revived = PluginManifest.from_json(payload)
    assert revived == manifest


def test_manifest_requires_safety_manifest_path() -> None:
    """R10: safety is never derived from the capability surface alone."""

    with pytest.raises(ValueError, match="safety_manifest_path"):
        PluginManifest(
            name="wechat_readonly",
            version="1.0.0",
            kind="mcp",
            safety_manifest_path="",
            mcp=MCPPluginSpec(command=("python", "-m", "x")),
        )


def test_manifest_mcp_kind_rejects_http_payload() -> None:
    with pytest.raises(ValueError, match="'http' field must be None"):
        PluginManifest(
            name="bad",
            version="1.0.0",
            kind="mcp",
            safety_manifest_path="x.yaml",
            mcp=MCPPluginSpec(command=("python", "-m", "x")),
            http=HttpPluginSpec(
                base_url="https://example.com",
                endpoints=(HttpEndpoint(name="ping", path="/p"),),
            ),
        )


def test_manifest_http_kind_rejects_missing_http_payload() -> None:
    with pytest.raises(ValueError, match="'http' field is required"):
        PluginManifest(
            name="bad",
            version="1.0.0",
            kind="http",
            safety_manifest_path="x.yaml",
        )


def test_manifest_rejects_dot_in_name() -> None:
    """``.`` is reserved as the plugin/endpoint separator."""

    with pytest.raises(ValueError, match="ASCII letters / digits"):
        PluginManifest(
            name="plugin.invalid",
            version="1.0.0",
            kind="mcp",
            safety_manifest_path="x.yaml",
            mcp=MCPPluginSpec(command=("python", "-m", "x")),
        )


# ---------------------------------------------------------------------------
# MCP plugin spec field rules
# ---------------------------------------------------------------------------


def test_mcp_plugin_spec_stdio_requires_command() -> None:
    with pytest.raises(ValueError, match="non-empty 'command' tuple"):
        MCPPluginSpec(transport="stdio", command=())


def test_mcp_plugin_spec_http_requires_url() -> None:
    with pytest.raises(ValueError, match="non-empty 'url'"):
        MCPPluginSpec(transport="http", url="")


def test_mcp_plugin_spec_rejects_invalid_restart_policy() -> None:
    with pytest.raises(ValueError, match="restart_policy"):
        MCPPluginSpec(
            transport="stdio",
            command=("x",),
            restart_policy="reboot_world",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# HTTP plugin spec rules
# ---------------------------------------------------------------------------


def test_http_plugin_spec_rejects_empty_endpoints() -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        HttpPluginSpec(base_url="https://x", endpoints=())


def test_http_plugin_spec_rejects_non_http_base_url() -> None:
    with pytest.raises(ValueError, match="http:// or https://"):
        HttpPluginSpec(
            base_url="ftp://x",
            endpoints=(HttpEndpoint(name="a", path="/a"),),
        )


def test_http_plugin_spec_rejects_duplicate_endpoint_names() -> None:
    with pytest.raises(ValueError, match="duplicate endpoint name"):
        HttpPluginSpec(
            base_url="https://x",
            endpoints=(
                HttpEndpoint(name="ping", path="/a"),
                HttpEndpoint(name="ping", path="/b"),
            ),
        )


def test_http_endpoint_rejects_dotted_name() -> None:
    with pytest.raises(ValueError, match="ASCII letters"):
        HttpEndpoint(name="weather.current", path="/x")


def test_http_endpoint_rejects_non_root_path() -> None:
    with pytest.raises(ValueError, match="must start with '/'"):
        HttpEndpoint(name="x", path="weather")


# ---------------------------------------------------------------------------
# MCPServerSpec hydration helper
# ---------------------------------------------------------------------------


def test_to_mcp_server_kwargs_yields_valid_mcp_server_spec_kwargs() -> None:
    pytest.importorskip("lifeform_mcp_bridge")
    from lifeform_mcp_bridge import MCPServerSpec  # noqa: PLC0415

    manifest = _mcp_plugin()
    kwargs = manifest.to_mcp_server_kwargs()
    spec = MCPServerSpec(**kwargs)
    assert spec.name == manifest.name
    assert spec.command == manifest.mcp.command
    assert spec.safety_manifest_path == manifest.safety_manifest_path


def test_to_mcp_server_kwargs_raises_for_http_plugin() -> None:
    manifest = _http_plugin()
    with pytest.raises(ValueError, match="kind == 'mcp'"):
        manifest.to_mcp_server_kwargs()


# ---------------------------------------------------------------------------
# ContractSpec extension + legacy back-compat
# ---------------------------------------------------------------------------


def test_contract_spec_plugins_defaults_to_empty() -> None:
    spec = ContractSpec(
        contract_id="ctr_legacy",
        tenant_id="ten_legacy",
        template_id="tpl_legacy",
        template_version=1,
        shell_id="shl_legacy",
    )
    assert spec.plugins == ()


def test_contract_spec_from_json_legacy_payload_back_compat() -> None:
    """Existing DB rows (no ``plugins`` key) deserialise to an empty tuple."""

    legacy_payload = {
        "contract_id": "ctr_legacy",
        "tenant_id": "ten_legacy",
        "template_id": "tpl_legacy",
        "shell_id": "shl_legacy",
        "engine_tools": {"text": True},
    }
    spec = ContractSpec.from_json(legacy_payload)
    assert spec.plugins == ()
    assert spec.engine_tools == {"text": True}


def test_contract_spec_to_from_json_with_plugins_roundtrip() -> None:
    spec = ContractSpec(
        contract_id="ctr1",
        tenant_id="ten1",
        template_id="tpl1",
        template_version=2,
        shell_id="shl1",
        engine_tools={"text": True},
        plugins=(_mcp_plugin(), _http_plugin()),
    )
    revived = ContractSpec.from_json(spec.to_json())
    assert revived.plugins == spec.plugins
    assert [p.name for p in revived.plugins] == [
        "wechat_readonly",
        "weather",
    ]


# ---------------------------------------------------------------------------
# compute_plugin_tool_policy_snapshot
# ---------------------------------------------------------------------------


def test_tool_policy_snapshot_unions_legacy_and_plugin_capabilities() -> None:
    snapshot = compute_plugin_tool_policy_snapshot(
        engine_tools={"text": True, "voice": False, "code": {"enabled": True}},
        plugins=(_mcp_plugin(), _http_plugin()),
    )
    assert snapshot["enabled_capabilities"] == [
        "text",
        "code",
        "wechat_readonly.read_messages",
        "weather.current",
        "weather.forecast",
    ]
    assert snapshot["plugins"] == [
        {"name": "wechat_readonly", "version": "1.0.0", "kind": "mcp"},
        {"name": "weather", "version": "0.3.1", "kind": "http"},
    ]


def test_tool_policy_snapshot_no_plugins_matches_legacy_behaviour() -> None:
    snapshot = compute_plugin_tool_policy_snapshot(
        engine_tools={"text": True, "voice": False},
        plugins=(),
    )
    assert snapshot["enabled_capabilities"] == ["text"]
    assert snapshot["plugins"] == []
