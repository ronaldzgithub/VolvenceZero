"""Contract test: HTTP plugin → AffordanceDescriptor + invocation.

Validates Packet 2 + Packet 6 of the DLaaS plugin foundation rollout:

1. ``build_http_tool_descriptors`` requires per-endpoint manifest
   entries; without them it raises
   :class:`MissingHttpPluginManifestEntryError`.
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
    MissingHttpPluginManifestEntryError,
    build_http_tool_backend,
    build_http_tool_descriptors,
    register_http_blueprints,
)
from lifeform_mcp_bridge.safety_manifest import SafetyManifestEntry
from lifeform_service.plugin_attach import (
    http_blueprints_from_plugins,
    mcp_server_specs_from_plugins,
)
from volvence_zero.affordance import (
    AffordanceCost,
    AffordanceLatencyClass,
    AffordanceMonetaryClass,
    AffordanceSafety,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _entry(
    tool_name: str,
    *,
    excluded: bool = False,
    affordance_tags: tuple[str, ...] = (),
    safety: AffordanceSafety | None = None,
    cost: AffordanceCost | None = None,
) -> SafetyManifestEntry:
    """Build a ``SafetyManifestEntry`` good enough for ``http_tool`` tests."""

    return SafetyManifestEntry(
        tool_name=tool_name,
        when_to_use=(
            f"Call {tool_name!r} when the test scenario explicitly asks "
            "for the weather lookup; this hint is intentionally long enough "
            "to satisfy MIN_SELECTION_HINT_CHARS."
        ),
        when_not_to_use=(
            f"Do not call {tool_name!r} for sensitive data exfiltration or "
            "for tasks that do not need a fresh external lookup; long hint "
            "again for MIN_SELECTION_HINT_CHARS."
        ),
        cost_model=cost
        or AffordanceCost(
            latency_class=AffordanceLatencyClass.FAST,
            monetary_class=AffordanceMonetaryClass.FREE,
        ),
        safety_model=safety or AffordanceSafety(audit_required=False),
        affordance_tags=affordance_tags,
        excluded=excluded,
    )


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


def _entries() -> dict[str, SafetyManifestEntry]:
    return {
        "current": _entry("current"),
        "forecast": _entry("forecast"),
    }


# ---------------------------------------------------------------------------
# Blueprint invariants
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# build_http_tool_descriptors with manifest entries
# ---------------------------------------------------------------------------


def test_descriptors_inherit_when_to_use_from_manifest() -> None:
    builds = build_http_tool_descriptors(_blueprint(), entries=_entries())
    assert len(builds) == 2
    for build in builds:
        assert "Call" in build.descriptor.when_to_use
        assert "Do not call" in build.descriptor.when_not_to_use
    assert [b.descriptor.name for b in builds] == [
        "weather.current",
        "weather.forecast",
    ]


def test_descriptors_carry_plugin_tag_and_safety_path() -> None:
    builds = build_http_tool_descriptors(_blueprint(), entries=_entries())
    for build in builds:
        descriptor = build.descriptor
        assert "http_plugin" in descriptor.affordance_tags
        assert "weather" in descriptor.affordance_tags
        assert descriptor.source_path == "manifests/weather.vzbridge.yaml"


def test_missing_entry_raises_typed_error() -> None:
    """Endpoint without a manifest entry must NOT silently default."""

    incomplete: dict[str, SafetyManifestEntry] = {"current": _entry("current")}
    with pytest.raises(MissingHttpPluginManifestEntryError):
        build_http_tool_descriptors(_blueprint(), entries=incomplete)


def test_descriptor_inherits_safety_from_entry() -> None:
    safety = AffordanceSafety(
        requires_user_confirmation=True,
        irreversible=False,
        requires_consent_grant=("weather_read",),
        blocked_in_regimes=("emotional_support",),
        audit_required=True,
    )
    entries = {
        "current": _entry("current", safety=safety),
        "forecast": _entry("forecast"),
    }
    builds = build_http_tool_descriptors(_blueprint(), entries=entries)
    current_build = builds[0]
    assert current_build.descriptor.safety_model == safety


def test_descriptor_excluded_entry_disables_backend() -> None:
    entries = {
        "current": _entry("current", excluded=True),
        "forecast": _entry("forecast"),
    }
    builds = build_http_tool_descriptors(_blueprint(), entries=entries)
    excluded_build = builds[0]
    backend_eligible_build = builds[1]
    assert excluded_build.descriptor.excluded_from_runtime_selection is True
    assert excluded_build.backend_eligible is False
    assert backend_eligible_build.backend_eligible is True


def test_entry_tags_merged_after_default_plugin_tags() -> None:
    """Manifest tags ride along after the standard ``http_plugin`` tags."""

    entries = {
        "current": _entry("current", affordance_tags=("weather_lookup",)),
        "forecast": _entry("forecast", affordance_tags=("http_plugin",)),
    }
    builds = build_http_tool_descriptors(_blueprint(), entries=entries)
    current_tags = builds[0].descriptor.affordance_tags
    forecast_tags = builds[1].descriptor.affordance_tags
    assert current_tags == ("http_plugin", "weather", "weather_lookup")
    assert forecast_tags == ("http_plugin", "weather")  # duplicates deduped


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
        entries_by_plugin={"weather": _entries()},
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


def test_register_http_blueprints_skips_backend_for_excluded_entries() -> None:
    blueprint = _blueprint()
    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    entries: dict[str, SafetyManifestEntry] = {
        "current": _entry("current", excluded=True),
        "forecast": _entry("forecast"),
    }
    descriptors = register_http_blueprints(
        registry=registry,
        invoker=invoker,
        blueprints=(blueprint,),
        entries_by_plugin={"weather": entries},
        env_resolver=lambda var: "stub",
    )
    # Both descriptors land in the registry (audit visibility), but
    # only the non-excluded one gets a backend.
    assert tuple(d.name for d in descriptors) == (
        "weather.current",
        "weather.forecast",
    )
    assert set(invoker.backend_names()) == {"weather.forecast"}


def test_register_http_blueprints_missing_plugin_entries_raises() -> None:
    blueprint = _blueprint()
    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    with pytest.raises(MissingHttpPluginManifestEntryError):
        register_http_blueprints(
            registry=registry,
            invoker=invoker,
            blueprints=(blueprint,),
            entries_by_plugin={},  # no weather entries
            env_resolver=lambda var: "stub",
        )


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
        registry=registry,
        invoker=invoker,
        blueprints=(blueprint_a,),
        entries_by_plugin={"weather": _entries()},
    )
    with pytest.raises(AffordanceAlreadyRegisteredError):
        register_http_blueprints(
            registry=registry,
            invoker=invoker,
            blueprints=(blueprint_b,),
            entries_by_plugin={"weather": {"current": _entry("current")}},
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
