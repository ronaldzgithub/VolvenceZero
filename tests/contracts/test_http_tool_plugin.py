"""Contract test: HTTP plugin → AffordanceDescriptor + invocation.

Validates Packet 2 of the DLaaS plugin foundation rollout:

1. ``build_http_tool_descriptors`` generates one descriptor per
   endpoint, padding short hints to satisfy ``MIN_SELECTION_HINT_CHARS``.
2. ``build_http_tool_backend`` consults the resolved env-var
   header templates and shapes the request properly per HTTP method.
3. ``register_http_blueprints`` writes both registry and invoker.
4. ``HttpToolBlueprint`` raises when ``safety_manifest_path`` is
   missing (R10 invariant).
5. ``lifeform_service.plugin_attach`` correctly converts a tuple of
   :class:`dlaas_platform_contracts.PluginManifest` into MCP specs
   and HTTP blueprints.
"""

from __future__ import annotations

from typing import Any

import pytest

from dlaas_platform_contracts import (
    HttpEndpoint,
    HttpPluginSpec,
    MCPPluginSpec,
    PluginManifest,
)
from lifeform_affordance import (
    AffordanceInvoker,
    AffordanceRegistry,
    HttpToolBlueprint,
    HttpToolEndpoint,
    build_http_tool_backend,
    build_http_tool_descriptors,
    register_http_blueprints,
)
from lifeform_service.plugin_attach import (
    http_blueprints_from_plugins,
    mcp_server_specs_from_plugins,
)


# ---------------------------------------------------------------------------
# Blueprint construction
# ---------------------------------------------------------------------------


def _blueprint(plugin_name: str = "weather") -> HttpToolBlueprint:
    return HttpToolBlueprint(
        plugin_name=plugin_name,
        plugin_version="1.0.0",
        base_url="https://api.weather.example.com",
        endpoints=(
            HttpToolEndpoint(
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
            HttpToolEndpoint(
                name="forecast",
                method="POST",
                path="/v1/forecast",
                parameters_schema={"type": "object"},
            ),
        ),
        safety_manifest_path="manifests/weather.vzbridge.yaml",
        auth_header_templates={"x-api-key": "${env:WEATHER_API_KEY}"},
    )


def test_blueprint_requires_safety_manifest_path() -> None:
    """R10: safety is never derived from the capability surface alone."""

    with pytest.raises(ValueError, match="safety_manifest_path"):
        HttpToolBlueprint(
            plugin_name="weather",
            plugin_version="1.0.0",
            base_url="https://api.weather.example.com",
            endpoints=(HttpToolEndpoint(name="current", method="GET", path="/c"),),
            safety_manifest_path="",
        )


def test_descriptors_pad_short_hints_to_min_chars() -> None:
    blueprint = _blueprint()
    descriptors = build_http_tool_descriptors(blueprint)
    assert len(descriptors) == 2
    for descriptor in descriptors:
        assert len(descriptor.when_to_use) >= 50
        assert len(descriptor.when_not_to_use) >= 50
    assert [d.name for d in descriptors] == ["weather.current", "weather.forecast"]


def test_descriptors_carry_plugin_tag_and_safety_path() -> None:
    blueprint = _blueprint()
    descriptors = build_http_tool_descriptors(blueprint)
    for descriptor in descriptors:
        assert "http_plugin" in descriptor.affordance_tags
        assert blueprint.plugin_name in descriptor.affordance_tags
        assert descriptor.source_path == blueprint.safety_manifest_path
        assert descriptor.safety_model.audit_required


# ---------------------------------------------------------------------------
# Backend invocation (stubbed HTTP client)
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, status: int, payload: Any, headers: dict[str, str]) -> None:
        self.status = status
        self._payload = payload
        self.headers = headers

    async def text(self) -> str:
        return str(self._payload)

    async def json(self, content_type: Any = None) -> Any:
        return self._payload

    async def __aenter__(self) -> "_StubResponse":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass


class _StubSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _StubResponse:
        self.calls.append((method, url, kwargs))
        return _StubResponse(
            status=200,
            payload={"echo": kwargs},
            headers={"content-type": "application/json"},
        )

    async def __aenter__(self) -> "_StubSession":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass


@pytest.mark.asyncio
async def test_get_backend_passes_parameters_as_query_string() -> None:
    blueprint = _blueprint()
    endpoint = blueprint.endpoints[0]  # GET /v1/current
    session = _StubSession()
    backend = build_http_tool_backend(
        blueprint,
        endpoint,
        env_resolver=lambda var: "secret-key" if var == "WEATHER_API_KEY" else None,
        http_client_factory=lambda: session,
    )
    result = await backend({"city": "shanghai"})
    assert result is not None
    assert result["status"] == 200
    method, url, kwargs = session.calls[0]
    assert method == "GET"
    assert url == "https://api.weather.example.com/v1/current"
    assert kwargs["params"] == {"city": "shanghai"}
    assert kwargs["headers"]["x-api-key"] == "secret-key"


@pytest.mark.asyncio
async def test_post_backend_serialises_parameters_as_json_body() -> None:
    blueprint = _blueprint()
    endpoint = blueprint.endpoints[1]  # POST /v1/forecast
    session = _StubSession()
    backend = build_http_tool_backend(
        blueprint,
        endpoint,
        env_resolver=lambda var: "secret-key" if var == "WEATHER_API_KEY" else None,
        http_client_factory=lambda: session,
    )
    await backend({"city": "shanghai", "days": 5})
    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url == "https://api.weather.example.com/v1/forecast"
    assert kwargs["json"] == {"city": "shanghai", "days": 5}


@pytest.mark.asyncio
async def test_backend_drops_unresolved_header_when_env_missing() -> None:
    blueprint = _blueprint()
    endpoint = blueprint.endpoints[0]
    session = _StubSession()
    backend = build_http_tool_backend(
        blueprint,
        endpoint,
        env_resolver=lambda var: None,
        http_client_factory=lambda: session,
    )
    await backend({"city": "x"})
    method, url, kwargs = session.calls[0]
    assert "x-api-key" not in kwargs["headers"]


# ---------------------------------------------------------------------------
# Registry + invoker wiring
# ---------------------------------------------------------------------------


def test_register_http_blueprints_populates_registry_and_invoker() -> None:
    blueprint = _blueprint()
    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    descriptors = register_http_blueprints(
        registry=registry,
        invoker=invoker,
        blueprints=(blueprint,),
        env_resolver=lambda var: "stub",
    )
    assert tuple(d.name for d in descriptors) == (
        "weather.current",
        "weather.forecast",
    )
    assert set(invoker.backend_names()) == {
        "weather.current",
        "weather.forecast",
    }


def test_register_http_blueprints_rejects_duplicate_plugin_names() -> None:
    """Two plugins claiming the same endpoint name must fail loudly."""

    from lifeform_affordance.registry import AffordanceAlreadyRegisteredError

    blueprint_a = _blueprint()
    blueprint_b = HttpToolBlueprint(
        plugin_name="weather",
        plugin_version="2.0.0",
        base_url="https://other.example.com",
        endpoints=(
            HttpToolEndpoint(name="current", method="GET", path="/c"),
        ),
        safety_manifest_path="other.yaml",
    )
    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    register_http_blueprints(
        registry=registry, invoker=invoker, blueprints=(blueprint_a,)
    )
    with pytest.raises(AffordanceAlreadyRegisteredError):
        register_http_blueprints(
            registry=registry, invoker=invoker, blueprints=(blueprint_b,)
        )


# ---------------------------------------------------------------------------
# plugin_attach converters (the platform <-> lifeform boundary)
# ---------------------------------------------------------------------------


def _mcp_manifest() -> PluginManifest:
    return PluginManifest(
        name="wechat_readonly",
        version="1.0.0",
        kind="mcp",
        safety_manifest_path="manifests/wechat.vzbridge.yaml",
        declared_capabilities=("wechat_readonly.read_messages",),
        mcp=MCPPluginSpec(
            transport="stdio",
            command=("python", "-m", "wechat_bundle.server"),
        ),
    )


def _http_manifest() -> PluginManifest:
    return PluginManifest(
        name="weather",
        version="0.3.1",
        kind="http",
        safety_manifest_path="manifests/weather.vzbridge.yaml",
        declared_capabilities=("weather.current",),
        http=HttpPluginSpec(
            base_url="https://api.weather.example.com",
            endpoints=(
                HttpEndpoint(
                    name="current",
                    method="GET",
                    path="/v1/current",
                    parameters_schema={"type": "object"},
                ),
            ),
        ),
    )


def test_mcp_server_specs_from_plugins_round_trips_into_bridge_spec() -> None:
    pytest.importorskip("lifeform_mcp_bridge")
    specs = mcp_server_specs_from_plugins((_mcp_manifest(), _http_manifest()))
    assert len(specs) == 1
    assert specs[0].name == "wechat_readonly"
    assert specs[0].command == ("python", "-m", "wechat_bundle.server")


def test_http_blueprints_from_plugins_skips_mcp_kind() -> None:
    blueprints = http_blueprints_from_plugins(
        (_mcp_manifest(), _http_manifest())
    )
    assert len(blueprints) == 1
    assert blueprints[0].plugin_name == "weather"
    assert blueprints[0].endpoints[0].name == "current"
