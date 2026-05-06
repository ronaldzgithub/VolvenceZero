"""Tests for shared-substrate sharing across sessions.

These tests exercise the "one substrate runtime serving N concurrent
sessions" path that the lifeform service uses when deployed on a single
GPU server. We use ``SyntheticOpenWeightResidualRuntime`` for speed and
determinism; the real HF runtime path is the same wiring \u2014 just with a
heavier ``runtime`` instance.

Three things are pinned:

1. ``create_app(substrate_runtime=...)`` propagates the runtime so every
   ``LifeformSession`` ends up with the same Python object as its
   ``_default_residual_runtime``.
2. Concurrent turns on different sessions all complete successfully and
   return non-empty responses (no race / no shared mutable state).
3. ``create_app`` raises when handed a runtime that advertises
   ``supports_live_substrate_mutation=True`` \u2014 sharing such a runtime
   would let one session's adapter-delta updates leak into every other
   session.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture
def shared_synthetic_runtime():
    from volvence_zero.substrate import SyntheticOpenWeightResidualRuntime

    return SyntheticOpenWeightResidualRuntime(model_id="shared-test-runtime")


@pytest.fixture
def companion_spec():
    from lifeform_service.verticals import discover_verticals

    return discover_verticals()["companion"]


# ---------------------------------------------------------------------------
# Wiring: shared runtime reaches every session
# ---------------------------------------------------------------------------


async def test_shared_substrate_propagates_to_every_session(
    shared_synthetic_runtime, companion_spec
):
    from lifeform_service.app import create_app

    app = create_app(vertical=companion_spec, substrate_runtime=shared_synthetic_runtime)
    manager = app["session_manager"]

    s1 = await manager.create_session(session_id="tenant-a")
    s2 = await manager.create_session(session_id="tenant-b")

    rt_a = s1.brain_session.runner._default_residual_runtime
    rt_b = s2.brain_session.runner._default_residual_runtime
    assert rt_a is shared_synthetic_runtime
    assert rt_b is shared_synthetic_runtime
    # /v1/info should reflect that sharing is active.
    assert app.get("substrate_runtime") is shared_synthetic_runtime


# ---------------------------------------------------------------------------
# Concurrency: parallel turns complete safely
# ---------------------------------------------------------------------------


async def test_concurrent_turns_on_shared_runtime_complete_independently(
    shared_synthetic_runtime, companion_spec
):
    from lifeform_service.app import create_app

    app = create_app(vertical=companion_spec, substrate_runtime=shared_synthetic_runtime)
    manager = app["session_manager"]

    # Sequential create (each one must observe a unique id and the same
    # shared runtime); concurrency is exercised on the run_turn calls.
    sessions = [
        await manager.create_session(session_id=f"sess-{i}") for i in range(4)
    ]
    assert len(sessions) == 4

    user_inputs = (
        "I have been feeling really stuck lately and I do not know why.",
        "Just saying hi, hope your day is going alright.",
        "Can you help me think through whether to leave my job?",
        "It is mostly that work feels heavy and home is also tense.",
    )
    results = await asyncio.gather(
        *(s.run_turn(text) for s, text in zip(sessions, user_inputs, strict=True))
    )
    assert len(results) == 4
    for result in results:
        assert result.response.text.strip(), "empty response on shared runtime"
        assert "model=shared-test-runtime" in result.response.rationale
        assert result.active_regime, "regime not selected"

    # Each session has independent turn bookkeeping.
    for s in sessions:
        assert len(s.turn_summaries) == 1


# ---------------------------------------------------------------------------
# Frozen-substrate guard
# ---------------------------------------------------------------------------


def test_create_app_rejects_mutation_capable_runtime_when_shared(
    shared_synthetic_runtime, companion_spec
):
    from lifeform_service.app import create_app

    # Synthetic runtime declares its frozen flag the same way the HF
    # runtime does; we flip it to simulate a misconfigured deployment.
    shared_synthetic_runtime.supports_live_substrate_mutation = True
    with pytest.raises(ValueError, match="supports_live_substrate_mutation=True"):
        create_app(vertical=companion_spec, substrate_runtime=shared_synthetic_runtime)


# ---------------------------------------------------------------------------
# /v1/info reports sharing state
# ---------------------------------------------------------------------------


@pytest.fixture
async def shared_client(aiohttp_client, companion_spec, shared_synthetic_runtime):
    from lifeform_service.app import create_app

    app = create_app(
        vertical=companion_spec,
        substrate_runtime=shared_synthetic_runtime,
        idle_eviction_seconds=None,
    )
    return await aiohttp_client(app)


async def test_info_reports_shared_substrate_when_runtime_is_supplied(shared_client):
    body = await (await shared_client.get("/v1/info")).json()
    assert body["substrate_shared"] is True
    assert body["substrate_model_id"] == "shared-test-runtime"
    assert body["substrate_runtime_origin"] is not None


@pytest.fixture
async def unshared_client(aiohttp_client, companion_spec):
    from lifeform_service.app import create_app

    app = create_app(vertical=companion_spec, idle_eviction_seconds=None)
    return await aiohttp_client(app)


async def test_info_reports_no_sharing_when_runtime_is_none(unshared_client):
    body = await (await unshared_client.get("/v1/info")).json()
    assert body["substrate_shared"] is False
    assert body["substrate_model_id"] is None
