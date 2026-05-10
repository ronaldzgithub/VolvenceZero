"""Verify ``lifeform-serve --enable-openai-compat`` wires up the routes.

This is the **single integration test** that crosses the
lifeform-service / lifeform-openai-compat seam. Everything else is
unit-tested with fakes; this test runs the actual ``cli._build_parser``
+ ``add_openai_routes`` path against a real ``create_app(...)`` to
prove that:

1. The new ``--enable-openai-compat`` flag is recognised.
2. With the flag set + ``add_openai_routes`` mounted, the OpenAI
   surfaces (``/v1/chat/completions``, ``/v1/models``) are reachable.
3. Without the flag, the OpenAI surfaces are NOT mounted (404), so
   the default deployment is unchanged.

We do NOT exercise the real Qwen substrate here — the test app uses
the synthetic vertical (no shared runtime), so chat-completions
route in ``mode=lifeform`` succeeds (lifeform synth path works) and
``mode=raw`` returns the documented 503 (no shared runtime).
"""

from __future__ import annotations

import pytest


@pytest.fixture
async def client_with_openai_compat(aiohttp_client):
    """Build a real lifeform-service app + mount the OpenAI compat router."""
    from lifeform_service.app import create_app
    from lifeform_service.verticals import discover_verticals

    from lifeform_openai_compat import add_openai_routes

    spec = discover_verticals()["companion"]
    app = create_app(vertical=spec, max_sessions=4, idle_eviction_seconds=None)
    add_openai_routes(app)
    return await aiohttp_client(app)


@pytest.fixture
async def client_without_openai_compat(aiohttp_client):
    """Same app but without the OpenAI compat router — surfaces should 404."""
    from lifeform_service.app import create_app
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    app = create_app(vertical=spec, max_sessions=4, idle_eviction_seconds=None)
    return await aiohttp_client(app)


# ---------------------------------------------------------------------------
# CLI argparse
# ---------------------------------------------------------------------------


def test_cli_parser_recognises_enable_openai_compat_flag() -> None:
    from lifeform_service.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["--enable-openai-compat"])
    assert args.enable_openai_compat is True

    args = parser.parse_args([])
    assert args.enable_openai_compat is False


def test_cli_help_documents_openai_compat_flag() -> None:
    from lifeform_service.cli import _build_parser

    parser = _build_parser()
    help_text = parser.format_help()
    assert "--enable-openai-compat" in help_text
    assert "OpenAI" in help_text


# ---------------------------------------------------------------------------
# Mounted routes are reachable on the real lifeform-service app
# ---------------------------------------------------------------------------


async def test_existing_lifeform_service_models_route_still_works_after_mount(
    client_with_openai_compat,
) -> None:
    """lifeform-service owns /v1/models; mounting our adapter must not break it.

    The adapter intentionally mounts only ``/v1/chat/completions``,
    so the lifeform-service-native ``/v1/models`` route should
    continue serving on the same app.
    """
    resp = await client_with_openai_compat.get("/v1/models")
    assert resp.status == 200
    body = await resp.json()
    # lifeform-service shape: {"vertical": ..., "swap_supported": bool, ...}
    assert "vertical" in body


async def test_chat_completions_lifeform_mode_smoke(client_with_openai_compat) -> None:
    """Default mode (lifeform) runs through synth substrate end-to-end."""

    resp = await client_with_openai_compat.post(
        "/v1/chat/completions",
        json={
            "model": "lifeform-companion",
            "messages": [{"role": "user", "content": "Hello there."}],
        },
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    # Synth substrate produces SOME assistant text — exact content varies
    # but it must not be empty.
    assert body["choices"][0]["message"]["content"]


async def test_chat_completions_raw_mode_503_without_substrate(client_with_openai_compat) -> None:
    """Synthetic-only deployment has no substrate runtime → mode=raw 503."""

    resp = await client_with_openai_compat.post(
        "/v1/chat/completions?mode=raw",
        json={
            "model": "lifeform-companion-raw",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert resp.status == 503
    body = await resp.json()
    assert body["error"]["code"] == "raw_substrate_unavailable"


# ---------------------------------------------------------------------------
# Without the flag, surfaces are NOT mounted (default deployment unchanged)
# ---------------------------------------------------------------------------


async def test_default_app_does_not_mount_openai_chat_completions(
    client_without_openai_compat,
) -> None:
    """Without the flag, /v1/chat/completions is NOT mounted (404).

    lifeform-service's own /v1/models is unaffected by the flag —
    that route belongs to lifeform-service. We only assert that the
    OpenAI-shape chat-completions route is gated behind the flag.
    """
    resp_chat = await client_without_openai_compat.post(
        "/v1/chat/completions",
        json={
            "model": "lifeform-companion",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp_chat.status == 404


async def test_default_app_existing_routes_still_work(client_without_openai_compat) -> None:
    """Sanity: --enable-openai-compat off → existing /v1/sessions API unchanged."""

    health = await client_without_openai_compat.get("/v1/health")
    assert health.status == 200
    create = await client_without_openai_compat.post("/v1/sessions", json={})
    assert create.status == 201
