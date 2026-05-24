"""Contract test: HTTP plugin reads safety_model from .vzbridge.yaml.

Packet 6 of the DLaaS plugin foundation: the HTTP plugin no longer
defaults to a conservative placeholder safety model — every endpoint
MUST be backed by a reviewed
:class:`lifeform_mcp_bridge.SafetyManifestEntry`. The launcher's
``plugin_attach.register_http_plugins_after_start`` is the wiring
point that loads each manifest and applies it.

Cases covered:

1. **Happy path** — a valid ``.vzbridge.yaml`` whose ``server.name``
   matches the plugin name and whose ``tools[]`` entries cover every
   endpoint produces affordances with the manifest's
   ``safety_model`` / ``cost_model`` / ``when_to_use`` /
   ``when_not_to_use`` / ``affordance_tags``.
2. **Missing entry** — manifest exists but does not cover one of the
   plugin's endpoints → :class:`MissingHttpPluginManifestEntryError`.
3. **excluded: true** — manifest entry marks the endpoint excluded;
   the descriptor is still registered but no backend is bound.
4. **Schema-version mismatch** — manifest with wrong
   ``schema_version`` → :class:`MCPSafetyManifestSchemaError`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dlaas_platform_contracts import (
    HttpEndpoint,
    HttpPluginSpec,
    PluginManifest,
)
from lifeform_affordance import (
    AffordanceInvoker,
    AffordanceRegistry,
    MissingHttpPluginManifestEntryError,
    register_http_blueprints,
)
from lifeform_mcp_bridge.errors import MCPSafetyManifestSchemaError
from lifeform_service.plugin_attach import (
    http_blueprints_from_plugins,
    load_http_plugin_manifest_entries,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_manifest(path: Path, document: dict) -> None:
    path.write_text(yaml.safe_dump(document), encoding="utf-8")


def _valid_manifest(server_name: str) -> dict:
    return {
        "schema_version": 1,
        "server": {
            "name": server_name,
            "description": "Weather plugin",
        },
        "tools": [
            {
                "name": "current",
                "when_to_use": (
                    "Call weather.current when the user asks for "
                    "real-time conditions in a named city; long enough "
                    "to satisfy MIN_SELECTION_HINT_CHARS."
                ),
                "when_not_to_use": (
                    "Do not call weather.current for historical lookups "
                    "or for cities the user has not named; long enough "
                    "to satisfy MIN_SELECTION_HINT_CHARS."
                ),
                "cost_model": {
                    "latency_class": "fast",
                    "monetary_class": "low",
                    "rate_limit_per_minute": 60,
                },
                "safety_model": {
                    "requires_user_confirmation": False,
                    "irreversible": False,
                    "requires_consent_grant": ["weather_read"],
                    "blocked_in_regimes": ["emotional_support"],
                    "audit_required": True,
                },
                "affordance_tags": ["weather_lookup"],
            },
            {
                "name": "forecast",
                "when_to_use": (
                    "Call weather.forecast when the user wants a "
                    "multi-day forecast; long enough to satisfy "
                    "MIN_SELECTION_HINT_CHARS."
                ),
                "when_not_to_use": (
                    "Do not call weather.forecast for current conditions; "
                    "use current instead. Long enough to satisfy "
                    "MIN_SELECTION_HINT_CHARS."
                ),
                "cost_model": {
                    "latency_class": "slow",
                    "monetary_class": "medium",
                },
                "safety_model": {
                    "audit_required": False,
                },
                "affordance_tags": ["weather_forecast"],
            },
        ],
        "resources": {"default_compliance_profile": "forced"},
        "prompts": {"enabled": False},
    }


def _plugin(safety_manifest_path: str) -> PluginManifest:
    return PluginManifest(
        name="weather",
        version="1.0.0",
        kind="http",
        safety_manifest_path=safety_manifest_path,
        declared_capabilities=("weather.current", "weather.forecast"),
        http=HttpPluginSpec(
            base_url="https://api.weather.example.com",
            endpoints=(
                HttpEndpoint(
                    name="current",
                    method="GET",
                    path="/v1/current",
                    parameters_schema={"type": "object"},
                ),
                HttpEndpoint(
                    name="forecast",
                    method="POST",
                    path="/v1/forecast",
                    parameters_schema={"type": "object"},
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_descriptors_inherit_real_safety_model_from_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "weather.vzbridge.yaml"
    _write_manifest(manifest_path, _valid_manifest("weather"))
    plugin = _plugin(str(manifest_path))

    entries = load_http_plugin_manifest_entries(plugin)
    assert set(entries.keys()) == {"current", "forecast"}

    blueprint = http_blueprints_from_plugins((plugin,))[0]
    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    descriptors = register_http_blueprints(
        registry=registry,
        invoker=invoker,
        blueprints=(blueprint,),
        entries_by_plugin={"weather": entries},
    )
    by_name = {d.name: d for d in descriptors}

    current = by_name["weather.current"]
    assert current.safety_model.requires_consent_grant == ("weather_read",)
    assert current.safety_model.blocked_in_regimes == ("emotional_support",)
    assert current.safety_model.audit_required is True
    assert current.cost_model.rate_limit_per_minute == 60
    assert "weather_lookup" in current.affordance_tags
    assert "Call weather.current" in current.when_to_use

    forecast = by_name["weather.forecast"]
    assert forecast.safety_model.audit_required is False
    assert "weather_forecast" in forecast.affordance_tags


# ---------------------------------------------------------------------------
# Missing manifest entry
# ---------------------------------------------------------------------------


def test_missing_manifest_entry_raises_typed_error(tmp_path: Path) -> None:
    """Manifest exists but lacks an entry for one of the endpoints."""

    partial = _valid_manifest("weather")
    partial["tools"] = [partial["tools"][0]]  # drop the "forecast" entry
    manifest_path = tmp_path / "weather.vzbridge.yaml"
    _write_manifest(manifest_path, partial)

    plugin = _plugin(str(manifest_path))
    entries = load_http_plugin_manifest_entries(plugin)
    blueprint = http_blueprints_from_plugins((plugin,))[0]

    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    with pytest.raises(MissingHttpPluginManifestEntryError, match="forecast"):
        register_http_blueprints(
            registry=registry,
            invoker=invoker,
            blueprints=(blueprint,),
            entries_by_plugin={"weather": entries},
        )


# ---------------------------------------------------------------------------
# excluded: true ⇒ descriptor in registry, backend skipped
# ---------------------------------------------------------------------------


def test_excluded_entry_registers_descriptor_without_backend(
    tmp_path: Path,
) -> None:
    document = _valid_manifest("weather")
    document["tools"][1]["excluded"] = True  # exclude forecast
    manifest_path = tmp_path / "weather.vzbridge.yaml"
    _write_manifest(manifest_path, document)

    plugin = _plugin(str(manifest_path))
    entries = load_http_plugin_manifest_entries(plugin)
    blueprint = http_blueprints_from_plugins((plugin,))[0]
    registry = AffordanceRegistry()
    invoker = AffordanceInvoker(registry=registry)
    register_http_blueprints(
        registry=registry,
        invoker=invoker,
        blueprints=(blueprint,),
        entries_by_plugin={"weather": entries},
    )

    # Both descriptors registered for audit visibility...
    assert {"weather.current", "weather.forecast"} <= set(registry.names())
    forecast_descriptor = registry.get("weather.forecast")
    assert forecast_descriptor.excluded_from_runtime_selection is True
    # ...but only the non-excluded one has a backend.
    assert set(invoker.backend_names()) == {"weather.current"}


# ---------------------------------------------------------------------------
# Schema-version mismatch ⇒ MCPSafetyManifestSchemaError
# ---------------------------------------------------------------------------


def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    document = _valid_manifest("weather")
    document["schema_version"] = 9999
    manifest_path = tmp_path / "weather.vzbridge.yaml"
    _write_manifest(manifest_path, document)

    plugin = _plugin(str(manifest_path))
    with pytest.raises(MCPSafetyManifestSchemaError):
        load_http_plugin_manifest_entries(plugin)


def test_server_name_mismatch_raises(tmp_path: Path) -> None:
    document = _valid_manifest("totally_different")
    manifest_path = tmp_path / "weather.vzbridge.yaml"
    _write_manifest(manifest_path, document)

    plugin = _plugin(str(manifest_path))
    with pytest.raises(MCPSafetyManifestSchemaError, match="server.name"):
        load_http_plugin_manifest_entries(plugin)
