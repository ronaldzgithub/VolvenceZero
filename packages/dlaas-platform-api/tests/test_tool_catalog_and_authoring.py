"""Tests for the built-in tool catalog endpoint and the portal-driven
inline-safety registration path (HTTP-plugin auto-authoring)."""

from __future__ import annotations

import asyncio
import json

import pytest

_VALID_HINT = (
    "Use this only when the requirement clearly calls for it and the "
    "caller has supplied every required field up front."
)
_VALID_NOT = (
    "Do not use this for unrelated requests, status checks, or whenever "
    "a cheaper in-bundle tool already covers the need."
)


def _safety_doc(server_name: str, tool_name: str) -> dict:
    return {
        "schema_version": 1,
        "server": {"name": server_name, "description": "test plugin"},
        "tools": [
            {
                "name": tool_name,
                "when_to_use": _VALID_HINT,
                "when_not_to_use": _VALID_NOT,
                "cost_model": {
                    "latency_class": "fast",
                    "monetary_class": "low",
                    "rate_limit_per_minute": 30,
                },
                "safety_model": {
                    "requires_user_confirmation": True,
                    "irreversible": True,
                    "requires_consent_grant": [],
                    "blocked_in_regimes": [],
                    "audit_required": True,
                },
                "affordance_tags": ["write"],
                "excluded": False,
            }
        ],
        "resources": {"default_compliance_profile": "forced"},
        "prompts": {"enabled": False},
    }


# ---------------------------------------------------------------------------
# builtin-tools catalog
# ---------------------------------------------------------------------------


def test_builtin_tools_parses_manifest(tmp_path, monkeypatch) -> None:
    from dlaas_platform_api.app import _handle_catalog_builtin_tools

    manifest = tmp_path / ".vzbridge.yaml"
    manifest.write_text(json.dumps(_safety_doc("vz-bundle", "read_file")), encoding="utf-8")
    monkeypatch.setenv("VZ_BUNDLE_MANIFEST_PATH", str(manifest))

    resp = asyncio.run(_handle_catalog_builtin_tools(None))
    payload = json.loads(resp.text)
    assert payload["status"] == "ok"
    names = [t["name"] for t in payload["tools"]]
    assert "read_file" in names
    tool = payload["tools"][0]
    assert tool["cost_model"]["latency_class"] == "fast"
    assert tool["safety_model"]["irreversible"] is True


def test_builtin_tools_degrades_when_missing(tmp_path, monkeypatch) -> None:
    from dlaas_platform_api.app import _handle_catalog_builtin_tools

    monkeypatch.setenv(
        "VZ_BUNDLE_MANIFEST_PATH", str(tmp_path / "does-not-exist.yaml")
    )
    resp = asyncio.run(_handle_catalog_builtin_tools(None))
    payload = json.loads(resp.text)
    assert payload["status"] == "ok"
    assert payload["tools"] == []


# ---------------------------------------------------------------------------
# inline-safety materialization
# ---------------------------------------------------------------------------


def _http_plugin(safety_yaml: str) -> dict:
    return {
        "name": "helpdesk_ticket",
        "version": "0.1.0",
        "kind": "http",
        "safety_manifest_path": "managed://pending",
        "description": "open tickets",
        "declared_capabilities": ["helpdesk_ticket.create"],
        "http": {
            "base_url": "https://helpdesk.example.com",
            "auth_header_templates": {},
            "default_timeout_seconds": 30,
            "endpoints": [
                {
                    "name": "create",
                    "method": "POST",
                    "path": "/v1/tickets",
                    "description": "create",
                    "parameters_schema": {"type": "object", "properties": {}},
                    "output_schema": {},
                    "timeout_seconds": 30,
                }
            ],
        },
        "safety_manifest_yaml": safety_yaml,
    }


def test_materialize_inline_safety_writes_file(tmp_path, monkeypatch) -> None:
    from dlaas_platform_api.control_plane import (
        _materialize_inline_safety,
        _parse_inline_plugins,
    )

    monkeypatch.setenv("DLAAS_MANAGED_SAFETY_DIR", str(tmp_path))
    safety = json.dumps(_safety_doc("helpdesk_ticket", "create"))
    out = _materialize_inline_safety([_http_plugin(safety)], application_id="app_x")

    expected = tmp_path / "app_x" / "helpdesk_ticket.vzbridge.yaml"
    assert out[0]["safety_manifest_path"] == str(expected)
    assert expected.is_file()
    assert "safety_manifest_yaml" not in out[0]
    # The rewritten plugin must now parse into a PluginManifest.
    plugins = _parse_inline_plugins(out)
    assert plugins[0].name == "helpdesk_ticket"
    assert plugins[0].kind == "http"


def test_materialize_inline_safety_rejects_endpoint_mismatch(
    tmp_path, monkeypatch
) -> None:
    from dlaas_platform_api.control_plane import _materialize_inline_safety

    monkeypatch.setenv("DLAAS_MANAGED_SAFETY_DIR", str(tmp_path))
    # Safety declares tool 'other' but the endpoint is 'create'.
    safety = json.dumps(_safety_doc("helpdesk_ticket", "other"))
    with pytest.raises(ValueError, match="no matching tool entry"):
        _materialize_inline_safety([_http_plugin(safety)], application_id="app_y")


def test_materialize_inline_safety_passthrough_without_yaml(
    tmp_path, monkeypatch
) -> None:
    from dlaas_platform_api.control_plane import _materialize_inline_safety

    monkeypatch.setenv("DLAAS_MANAGED_SAFETY_DIR", str(tmp_path))
    plugin = {
        "name": "shipped",
        "version": "0.1.0",
        "kind": "http",
        "safety_manifest_path": "/some/file.vzbridge.yaml",
        "http": {
            "base_url": "https://x.example.com",
            "endpoints": [{"name": "e", "path": "/e"}],
        },
    }
    out = _materialize_inline_safety([plugin], application_id="app_z")
    assert out[0]["safety_manifest_path"] == "/some/file.vzbridge.yaml"
