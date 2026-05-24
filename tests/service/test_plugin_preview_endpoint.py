"""HTTP-level test for the plugin Preview (dry-run) endpoint.

Packet 7a routes exercised:

* ``POST /dlaas/applications/{id}/plugins/{name}/tools/{tool}:preview``
  for both ``http`` and ``mcp`` plugin kinds.

The most important invariant is **dry-run** — preview MUST never
construct an :class:`aiohttp.ClientSession` (would mean we are about
to call an external API). The fixture monkey-patches the constructor
and the test asserts it was not touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dlaas_platform_api import build_dlaas_app


CONTROL_PLANE_SECRET = "cp_secret_plugin_preview"


# ---------------------------------------------------------------------------
# Manifest + payload helpers
# ---------------------------------------------------------------------------


def _http_manifest_yaml(server_name: str) -> dict:
    return {
        "schema_version": 1,
        "server": {"name": server_name, "description": "weather"},
        "tools": [
            {
                "name": "current",
                "when_to_use": (
                    "Call weather.current when the user asks for real-time "
                    "conditions in a named city; long enough to satisfy "
                    "MIN_SELECTION_HINT_CHARS."
                ),
                "when_not_to_use": (
                    "Do not call weather.current for historical lookups; "
                    "use a different tool. Long enough to satisfy "
                    "MIN_SELECTION_HINT_CHARS."
                ),
                "cost_model": {"latency_class": "fast", "monetary_class": "low"},
                "safety_model": {
                    "requires_consent_grant": ["weather_read"],
                    "audit_required": True,
                },
                "affordance_tags": ["weather_lookup"],
            }
        ],
        "resources": {"default_compliance_profile": "forced"},
        "prompts": {"enabled": False},
    }


def _mcp_manifest_yaml(server_name: str) -> dict:
    return {
        "schema_version": 1,
        "server": {"name": server_name, "description": "wechat"},
        "tools": [
            {
                "name": "read_messages",
                "when_to_use": (
                    "Call wechat.read_messages when the user wants to "
                    "review recent group messages; long enough to satisfy "
                    "MIN_SELECTION_HINT_CHARS."
                ),
                "when_not_to_use": (
                    "Do not call wechat.read_messages when the user has "
                    "not authorised social-graph access; long enough to "
                    "satisfy MIN_SELECTION_HINT_CHARS."
                ),
                "cost_model": {"latency_class": "slow", "monetary_class": "free"},
                "safety_model": {
                    "irreversible": False,
                    "audit_required": True,
                },
                "affordance_tags": ["wechat"],
            }
        ],
        "resources": {"default_compliance_profile": "forced"},
        "prompts": {"enabled": False},
    }


def _http_plugin_payload(safety_manifest_path: str) -> dict:
    return {
        "name": "weather",
        "version": "1.0.0",
        "kind": "http",
        "safety_manifest_path": safety_manifest_path,
        "declared_capabilities": ["weather.current"],
        "http": {
            "base_url": "https://api.weather.example.com",
            "endpoints": [
                {
                    "name": "current",
                    "method": "GET",
                    "path": "/v1/current",
                    "parameters_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            "auth_header_templates": {
                "x-api-key": "${env:PREVIEW_WEATHER_API_KEY}"
            },
        },
    }


def _mcp_plugin_payload(safety_manifest_path: str) -> dict:
    return {
        "name": "wechat",
        "version": "1.0.0",
        "kind": "mcp",
        "safety_manifest_path": safety_manifest_path,
        "declared_capabilities": ["wechat.read_messages"],
        "mcp": {
            "transport": "stdio",
            "command": ["python", "-m", "wechat_bundle.server"],
            "env": {"WECHAT_BOT_TOKEN": "${env:PREVIEW_WECHAT_TOKEN}"},
        },
    }


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


async def _build_app(tmp_path: Path):
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    return build_dlaas_app(
        db_path=str(tmp_path / "plugin_preview.sqlite"),
        control_plane_secret=CONTROL_PLANE_SECRET,
        vertical=spec,
        max_sessions=4,
        idle_eviction_seconds=None,
    )


@pytest.fixture
async def http_client(aiohttp_client, tmp_path: Path):
    return await aiohttp_client(await _build_app(tmp_path))


@pytest.fixture
def no_external_calls(monkeypatch: pytest.MonkeyPatch):
    """Assert ``aiohttp.ClientSession`` is never constructed in the test.

    Importable at session start so the platform's other infra (e.g.
    test client) is wired BEFORE we install the trap; we only sentry
    new ``ClientSession`` instances after the platform is built. Any
    construction during the preview call would mean the dry-run
    contract broke.
    """

    import aiohttp

    original = aiohttp.ClientSession
    constructions: list[tuple] = []

    def _trap(*args, **kwargs):
        constructions.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(aiohttp, "ClientSession", _trap)
    yield constructions


async def _bootstrap_tenant(http_client) -> tuple[dict[str, str], str]:
    cp = {"X-Control-Plane-Secret": CONTROL_PLANE_SECRET}
    resp = await http_client.post(
        "/dlaas/tenants",
        headers=cp,
        json={"tenant_name": "Plugin preview", "contact_email": "p@example.com"},
    )
    body = await resp.json()
    tenant_id = body["tenant_id"]
    return (
        {
            "X-Tenant-Api-Key": body["api_key"],
            "X-Tenant-Api-Secret": body["api_secret"],
        },
        tenant_id,
    )


async def _register_application(http_client, plugin_payload: dict) -> dict:
    cp = {"X-Control-Plane-Secret": CONTROL_PLANE_SECRET}
    resp = await http_client.post(
        "/dlaas/applications",
        headers=cp,
        json={
            "name": "Preview app",
            "version": "1.0.0",
            "plugins": [plugin_payload],
        },
    )
    return await resp.json()


# ---------------------------------------------------------------------------
# HTTP plugin preview
# ---------------------------------------------------------------------------


async def test_http_preview_renders_url_headers_body(
    http_client, tmp_path: Path, no_external_calls, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "weather.vzbridge.yaml"
    manifest_path.write_text(
        yaml.safe_dump(_http_manifest_yaml("weather")), encoding="utf-8"
    )
    monkeypatch.setenv("PREVIEW_WEATHER_API_KEY", "super-secret-1234")

    tenant_headers, tenant_id = await _bootstrap_tenant(http_client)
    app = await _register_application(
        http_client, _http_plugin_payload(str(manifest_path))
    )
    application_id = app["application_id"]
    resp = await http_client.post(
        f"/dlaas/tenants/{tenant_id}/applications/{application_id}/approve",
        headers=tenant_headers,
        json={},
    )
    assert resp.status == 200, await resp.text()

    resp = await http_client.post(
        f"/dlaas/applications/{application_id}/plugins/weather/tools/current:preview",
        headers=tenant_headers,
        json={"parameters": {"city": "shanghai"}},
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["parameters_valid"] is True
    assert body["validation_errors"] == []
    assert body["safety_model"]["requires_consent_grant"] == ["weather_read"]
    assert body["cost_model"]["latency_class"] == "fast"
    preview = body["preview"]
    assert preview["kind"] == "http"
    assert preview["method"] == "GET"
    assert preview["url"] == "https://api.weather.example.com/v1/current"
    # Header redacted: first 4 chars + ***
    assert preview["headers"]["x-api-key"].startswith("supe")
    assert preview["headers"]["x-api-key"].endswith("***")
    assert "super-secret-1234" not in preview["headers"]["x-api-key"]
    assert preview["body"] == {"params": {"city": "shanghai"}}
    # Dry-run: no external HTTP calls ever made.
    assert no_external_calls == []


async def test_http_preview_reports_validation_errors(
    http_client, tmp_path: Path, no_external_calls
) -> None:
    manifest_path = tmp_path / "weather.vzbridge.yaml"
    manifest_path.write_text(
        yaml.safe_dump(_http_manifest_yaml("weather")), encoding="utf-8"
    )
    tenant_headers, tenant_id = await _bootstrap_tenant(http_client)
    app = await _register_application(
        http_client, _http_plugin_payload(str(manifest_path))
    )
    application_id = app["application_id"]
    await http_client.post(
        f"/dlaas/tenants/{tenant_id}/applications/{application_id}/approve",
        headers=tenant_headers,
        json={},
    )

    resp = await http_client.post(
        f"/dlaas/applications/{application_id}/plugins/weather/tools/current:preview",
        headers=tenant_headers,
        json={"parameters": {}},  # missing required "city"
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["parameters_valid"] is False
    assert any("city" in err for err in body["validation_errors"])
    assert no_external_calls == []


# ---------------------------------------------------------------------------
# MCP plugin preview
# ---------------------------------------------------------------------------


async def test_mcp_preview_renders_jsonrpc_frame(
    http_client, tmp_path: Path, no_external_calls, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "wechat.vzbridge.yaml"
    manifest_path.write_text(
        yaml.safe_dump(_mcp_manifest_yaml("wechat")), encoding="utf-8"
    )
    monkeypatch.setenv("PREVIEW_WECHAT_TOKEN", "wxtoken-abcdef")

    tenant_headers, tenant_id = await _bootstrap_tenant(http_client)
    app = await _register_application(
        http_client, _mcp_plugin_payload(str(manifest_path))
    )
    application_id = app["application_id"]
    await http_client.post(
        f"/dlaas/tenants/{tenant_id}/applications/{application_id}/approve",
        headers=tenant_headers,
        json={},
    )

    resp = await http_client.post(
        f"/dlaas/applications/{application_id}/plugins/wechat/tools/read_messages:preview",
        headers=tenant_headers,
        json={"parameters": {"group_id": "x", "limit": 5}},
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    # MCP has no parameters_schema available without spawning the server.
    assert body["parameters_schema_available"] is False
    preview = body["preview"]
    assert preview["kind"] == "mcp"
    assert preview["transport"] == "stdio"
    assert preview["command"] == ["python", "-m", "wechat_bundle.server"]
    assert preview["env"]["WECHAT_BOT_TOKEN"].startswith("wxto")
    assert preview["env"]["WECHAT_BOT_TOKEN"].endswith("***")
    assert "wxtoken-abcdef" not in preview["env"]["WECHAT_BOT_TOKEN"]
    assert preview["jsonrpc_call"] == {
        "jsonrpc": "2.0",
        "id": "preview",
        "method": "tools/call",
        "params": {
            "name": "read_messages",
            "arguments": {"group_id": "x", "limit": 5},
        },
    }
    assert no_external_calls == []


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


async def test_preview_requires_application_approval(
    http_client, tmp_path: Path, no_external_calls
) -> None:
    """A tenant who has not approved the application gets 403."""

    manifest_path = tmp_path / "weather.vzbridge.yaml"
    manifest_path.write_text(
        yaml.safe_dump(_http_manifest_yaml("weather")), encoding="utf-8"
    )
    tenant_headers, _tenant_id = await _bootstrap_tenant(http_client)
    app = await _register_application(
        http_client, _http_plugin_payload(str(manifest_path))
    )
    application_id = app["application_id"]
    # No approval call.
    resp = await http_client.post(
        f"/dlaas/applications/{application_id}/plugins/weather/tools/current:preview",
        headers=tenant_headers,
        json={"parameters": {"city": "x"}},
    )
    assert resp.status == 403
    body = await resp.json()
    assert body["error"] == "application_not_approved"
    assert no_external_calls == []


async def test_preview_unknown_tool_returns_404(
    http_client, tmp_path: Path, no_external_calls
) -> None:
    manifest_path = tmp_path / "weather.vzbridge.yaml"
    manifest_path.write_text(
        yaml.safe_dump(_http_manifest_yaml("weather")), encoding="utf-8"
    )
    tenant_headers, tenant_id = await _bootstrap_tenant(http_client)
    app = await _register_application(
        http_client, _http_plugin_payload(str(manifest_path))
    )
    application_id = app["application_id"]
    await http_client.post(
        f"/dlaas/tenants/{tenant_id}/applications/{application_id}/approve",
        headers=tenant_headers,
        json={},
    )
    resp = await http_client.post(
        f"/dlaas/applications/{application_id}/plugins/weather/tools/ghost:preview",
        headers=tenant_headers,
        json={"parameters": {}},
    )
    assert resp.status == 404
    body = await resp.json()
    assert body["error"] == "tool_not_found"
    assert no_external_calls == []
