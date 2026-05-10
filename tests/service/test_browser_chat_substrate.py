"""Browser-chat substrate hot-swap — e2e contract tests.

Covers the two new HTTP routes (``GET /v1/models``,
``POST /v1/admin/substrate``), the swap lifecycle (close sessions →
unload → load), and the explicit refusal contracts (DLaaS-style
fixed runtime rejects swap; unknown / mutation-capable runtimes
fail loudly).

We use :class:`SyntheticOpenWeightResidualRuntime` as the loaded
runtime — it carries the same ``model_id`` /
``supports_live_substrate_mutation`` interface as the HF runtime,
so the swap path's R2 enforcement is exercised faithfully without
the cost of loading a real Qwen.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def companion_spec():
    from lifeform_service.verticals import discover_verticals

    return discover_verticals()["companion"]


# ---------------------------------------------------------------------------
# Provider-mode fixture (swap supported)
# ---------------------------------------------------------------------------


def _make_synthetic_runtime(model_id: str):
    from volvence_zero.substrate import SyntheticOpenWeightResidualRuntime

    return SyntheticOpenWeightResidualRuntime(model_id=model_id)


@pytest.fixture
async def swap_client(aiohttp_client, companion_spec):
    """Browser-chat-style fixture: real provider with a counted loader.

    The loader closure records every load so swap tests can assert
    "the new runtime was actually built" without poking at provider
    internals.
    """
    from lifeform_service.app import create_app
    from lifeform_service.substrate_registry import (
        SubstrateModelSpec,
        SubstrateRuntimeProvider,
    )

    initial = _make_synthetic_runtime("synthetic-initial")
    load_log: list[str] = []

    def loader(model_id: str):
        load_log.append(model_id)
        return _make_synthetic_runtime(model_id)

    provider = SubstrateRuntimeProvider(
        initial_runtime=initial,
        initial_model_id="synthetic-initial",
        available=(
            SubstrateModelSpec(
                model_id="synthetic-initial",
                display_name="Synthetic Initial",
                family="synthetic",
                size_label="0B",
            ),
            SubstrateModelSpec(
                model_id="synthetic-alt",
                display_name="Synthetic Alt",
                family="synthetic",
                size_label="0B",
            ),
        ),
        runtime_loader=loader,
        swap_supported=True,
    )
    app = create_app(
        vertical=companion_spec,
        substrate_provider=provider,
        idle_eviction_seconds=None,
    )
    client = await aiohttp_client(app)
    client._test_load_log = load_log  # type: ignore[attr-defined]
    client._test_provider = provider  # type: ignore[attr-defined]
    return client


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


async def test_list_models_reports_provider_state(swap_client):
    resp = await swap_client.get("/v1/models")
    assert resp.status == 200
    body = await resp.json()
    assert body["swap_supported"] is True
    assert body["current_model_id"] == "synthetic-initial"
    assert body["swap_count"] == 0
    ids = [item["model_id"] for item in body["models"]]
    assert ids == ["synthetic-initial", "synthetic-alt"]


async def test_list_models_returns_unsupported_for_fixed_runtime(
    aiohttp_client, companion_spec
):
    """DLaaS-equivalent path: fixed runtime → swap_supported=False.

    The fixed-runtime wrapper still synthesises a 1-entry
    allowlist so the UI can render the active model id, but
    /v1/admin/substrate refuses swap calls with 503.
    """
    from lifeform_service.app import create_app

    runtime = _make_synthetic_runtime("synthetic-fixed")
    app = create_app(
        vertical=companion_spec,
        substrate_runtime=runtime,
        idle_eviction_seconds=None,
    )
    client = await aiohttp_client(app)
    body = await (await client.get("/v1/models")).json()
    assert body["swap_supported"] is False
    assert body["current_model_id"] == "synthetic-fixed"
    assert [item["model_id"] for item in body["models"]] == ["synthetic-fixed"]


async def test_list_models_returns_no_provider_when_runtime_unset(
    aiohttp_client, companion_spec
):
    """No substrate at all (synthetic per-session factory mode)."""
    from lifeform_service.app import create_app

    app = create_app(vertical=companion_spec, idle_eviction_seconds=None)
    client = await aiohttp_client(app)
    body = await (await client.get("/v1/models")).json()
    assert body["swap_supported"] is False
    assert body["current_model_id"] is None
    assert body["models"] == []


# ---------------------------------------------------------------------------
# POST /v1/admin/substrate
# ---------------------------------------------------------------------------


async def test_swap_substrate_loads_new_runtime_and_closes_sessions(swap_client):
    # Create a session that will be closed by the swap.
    create = await (await swap_client.post("/v1/sessions", json={})).json()
    sid = create["session_id"]
    assert (await swap_client.get(f"/v1/sessions/{sid}/state")).status == 200

    resp = await swap_client.post(
        "/v1/admin/substrate", json={"model_id": "synthetic-alt"}
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["swapped"] is True
    assert body["model_id"] == "synthetic-alt"
    assert body["previous_model_id"] == "synthetic-initial"
    assert body["closed_session_count"] == 1
    # Loader was actually invoked.
    assert "synthetic-alt" in swap_client._test_load_log

    # Old session is gone.
    state = await swap_client.get(f"/v1/sessions/{sid}/state")
    assert state.status == 404

    # New session lands on the new runtime.
    create2 = await (await swap_client.post("/v1/sessions", json={})).json()
    sid2 = create2["session_id"]
    manager = swap_client.app["session_manager"]
    new_session = await manager.get_session(sid2)
    runtime = new_session.brain_session.runner._default_residual_runtime
    assert runtime.model_id == "synthetic-alt"


async def test_swap_substrate_idempotent_for_same_model_id(swap_client):
    resp = await swap_client.post(
        "/v1/admin/substrate", json={"model_id": "synthetic-initial"}
    )
    assert resp.status == 200
    body = await resp.json()
    assert body["model_id"] == "synthetic-initial"
    assert body["closed_session_count"] == 0
    assert body["duration_seconds"] == 0.0
    # Loader was NOT invoked for an idempotent re-selection.
    assert "synthetic-initial" not in swap_client._test_load_log


async def test_swap_substrate_rejects_unknown_model_id(swap_client):
    resp = await swap_client.post(
        "/v1/admin/substrate", json={"model_id": "no-such-model"}
    )
    assert resp.status == 400
    body = await resp.json()
    assert body["error"] == "unknown_model_id"


async def test_swap_substrate_rejects_missing_model_id(swap_client):
    resp = await swap_client.post("/v1/admin/substrate", json={})
    assert resp.status == 400
    body = await resp.json()
    assert body["error"] == "invalid_model_id"


async def test_swap_substrate_503s_for_fixed_runtime(
    aiohttp_client, companion_spec
):
    from lifeform_service.app import create_app

    runtime = _make_synthetic_runtime("synthetic-fixed")
    app = create_app(
        vertical=companion_spec,
        substrate_runtime=runtime,
        idle_eviction_seconds=None,
    )
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/admin/substrate", json={"model_id": "synthetic-fixed"}
    )
    assert resp.status == 503
    body = await resp.json()
    assert body["error"] == "substrate_swap_not_supported"


async def test_swap_loader_failure_clears_runtime_and_returns_503(
    aiohttp_client, companion_spec
):
    """Loader raising an exception leaves the provider in a clean
    no-runtime state and surfaces ``substrate_load_failed``."""
    from lifeform_service.app import create_app
    from lifeform_service.substrate_registry import (
        SubstrateModelSpec,
        SubstrateRuntimeProvider,
    )

    initial = _make_synthetic_runtime("synthetic-initial")

    def broken_loader(model_id: str):
        raise RuntimeError(f"simulated load failure for {model_id}")

    provider = SubstrateRuntimeProvider(
        initial_runtime=initial,
        initial_model_id="synthetic-initial",
        available=(
            SubstrateModelSpec(
                model_id="synthetic-initial",
                display_name="initial",
                family="synthetic",
                size_label="0B",
            ),
            SubstrateModelSpec(
                model_id="synthetic-broken",
                display_name="broken",
                family="synthetic",
                size_label="0B",
            ),
        ),
        runtime_loader=broken_loader,
        swap_supported=True,
    )
    app = create_app(
        vertical=companion_spec,
        substrate_provider=provider,
        idle_eviction_seconds=None,
    )
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/admin/substrate", json={"model_id": "synthetic-broken"}
    )
    assert resp.status == 503
    body = await resp.json()
    assert body["error"] == "substrate_load_failed"
    assert "synthetic-broken" in body["target_model_id"]
    # Provider cleared the runtime; subsequent /v1/models surfaces the
    # broken state without crashing.
    listing = await (await client.get("/v1/models")).json()
    assert listing["current_model_id"] in (None, "")
    assert "synthetic-broken" in listing["last_swap_error"]


# ---------------------------------------------------------------------------
# create_app validation
# ---------------------------------------------------------------------------


def test_create_app_rejects_both_runtime_and_provider(companion_spec):
    from lifeform_service.app import create_app
    from lifeform_service.substrate_registry import fixed_provider_from_runtime

    runtime = _make_synthetic_runtime("synthetic-x")
    provider = fixed_provider_from_runtime(runtime)
    with pytest.raises(ValueError, match="not both"):
        create_app(
            vertical=companion_spec,
            substrate_runtime=runtime,
            substrate_provider=provider,
            idle_eviction_seconds=None,
        )


# ---------------------------------------------------------------------------
# Allowlist parsing
# ---------------------------------------------------------------------------


def test_parse_allowlist_falls_back_to_default_when_unset():
    from lifeform_service.substrate_registry import (
        DEFAULT_QWEN_MODEL_SPECS,
        parse_model_id_allowlist,
    )

    assert parse_model_id_allowlist(None) == DEFAULT_QWEN_MODEL_SPECS
    assert parse_model_id_allowlist("") == DEFAULT_QWEN_MODEL_SPECS
    assert parse_model_id_allowlist("   ") == DEFAULT_QWEN_MODEL_SPECS


def test_parse_allowlist_synthesizes_specs_for_unknown_ids():
    from lifeform_service.substrate_registry import parse_model_id_allowlist

    specs = parse_model_id_allowlist("Qwen/Qwen2.5-3B-Instruct,custom-org/custom-model")
    by_id = {spec.model_id: spec for spec in specs}
    assert "Qwen/Qwen2.5-3B-Instruct" in by_id
    assert by_id["Qwen/Qwen2.5-3B-Instruct"].family == "qwen2.5"
    assert "custom-org/custom-model" in by_id
    assert by_id["custom-org/custom-model"].family == "unknown"


def test_merge_initial_into_allowlist_prepends_initial():
    from lifeform_service.substrate_registry import (
        SubstrateModelSpec,
        merge_initial_into_allowlist,
    )

    base = (
        SubstrateModelSpec(
            model_id="b", display_name="b", family="x", size_label=""
        ),
    )
    merged = merge_initial_into_allowlist(initial_model_id="a", allowlist=base)
    assert [spec.model_id for spec in merged] == ["a", "b"]
    # Idempotent when initial is already in the list.
    merged2 = merge_initial_into_allowlist(initial_model_id="b", allowlist=base)
    assert [spec.model_id for spec in merged2] == ["b"]
