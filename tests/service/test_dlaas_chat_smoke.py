"""Slice 1 smoke check for DLaaS InteractionEnvelope dispatch.

Per the agreed test discipline (`docs/moving forward/dlaas-platform-rollout.md`),
this is the only behavioral test we ship before Slice 7. It exists to prove
that the architecture wires up end-to-end: a typed envelope flows through
``dlaas-platform-api`` into the existing kernel and produces an OutputAct
without any vz-* code change.

What we deliberately do NOT test here (deferred to Slice 7):

* Multi-tenant isolation, persistence round-trip, full lifecycle.
* The other six interaction_type values (those land in Slice 2.x and get
  contract tests in Slice 7.1).
* Operator pause / handoff / SSE.

If this single smoke ever flakes, the architecture failed earlier — the
import_boundaries tests should have caught it before we even got here.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def event_loop():
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def dlaas_client(aiohttp_client):
    """A lifeform-service app with the Slice 1 DLaaS routes attached."""
    from dlaas_platform_api import attach_dlaas_routes
    from lifeform_service.app import create_app
    from lifeform_service.verticals import discover_verticals

    spec = discover_verticals()["companion"]
    app = create_app(vertical=spec, max_sessions=4, idle_eviction_seconds=None)
    attach_dlaas_routes(app, default_ai_id="ai_smoke")
    return await aiohttp_client(app)


@pytest.fixture
async def dlaas_client_growth_advisor(aiohttp_client):
    """A lifeform-service app with the growth_advisor vertical wired in.

    Mirrors the companion fixture but routes
    ``runtime_template_id`` -> the LTV / private-domain growth-advisor
    vertical (``lifeform-domain-growth-advisor``). The smoke proves the
    vertical is reachable end-to-end through the DLaaS chat envelope
    path with no kernel diff.
    """
    from dlaas_platform_api import attach_dlaas_routes
    from lifeform_service.app import create_app
    from lifeform_service.verticals import discover_verticals

    discovered = discover_verticals()
    if "growth_advisor" not in discovered:
        pytest.skip("lifeform-domain-growth-advisor wheel not installed")
    spec = discovered["growth_advisor"]
    app = create_app(vertical=spec, max_sessions=4, idle_eviction_seconds=None)
    attach_dlaas_routes(app, default_ai_id="ai_smoke_growth_advisor")
    return await aiohttp_client(app)


async def test_chat_envelope_dispatches_to_kernel_and_returns_output_act(
    dlaas_client,
):
    """A chat envelope produces a structured OutputAct + active_regime.

    This is the Slice 1 Done check end-to-end:

    * The envelope is parsed by ``dlaas-platform-contracts``.
    * The router in ``dlaas-platform-api`` dispatches on ``interaction_type``.
    * ``LifeformSession.run_turn`` executes a real kernel turn.
    * The response body uses the DLaaS wire format
      (``output_acts[0].act_type == "text"``, ``capability == "text_streaming"``).
    """
    body = {
        "contract_id": "ctr_smoke_001",
        "protocol_version": "dlaas/v1",
        "session_id": "sess_smoke_001",
        "end_user_ref": "user_smoke",
        "interaction_type": "chat",
        "mode": "live",
        "human_brief": "你好，今天我们来聊聊学习。",
        "structured_context": {"target_person_ids": []},
        "lang": "cn",
    }
    resp = await dlaas_client.post(
        "/dlaas/instances/ai_smoke/interactions",
        json=body,
    )
    assert resp.status == 200, await resp.text()
    payload = await resp.json()
    assert payload["status"] == "ok"
    assert payload["ai_id"] == "ai_smoke"
    assert payload["contract_id"] == "ctr_smoke_001"
    assert payload["session_id"] == "sess_smoke_001"
    assert payload["interaction_type"] == "chat"
    assert payload["protocol_version"] == "dlaas/v1"
    assert isinstance(payload["output_acts"], list) and payload["output_acts"]
    primary = payload["output_acts"][0]
    assert primary["act_type"] == "text"
    assert primary["capability"] == "text_streaming"
    assert "content" in primary["payload"]
    assert isinstance(primary["payload"]["content"], str)


async def test_chat_envelope_rejects_missing_human_brief(dlaas_client):
    """A chat envelope without ``human_brief`` returns a typed 400 error.

    The point is to confirm the dispatch handler does NOT silently fall
    through to ``LifeformSession.run_turn("")`` — the no-empty-input
    invariant is part of the typed contract surface.
    """
    body = {
        "contract_id": "ctr_smoke_002",
        "session_id": "sess_smoke_002",
        "end_user_ref": "user_smoke",
        "interaction_type": "chat",
    }
    resp = await dlaas_client.post(
        "/dlaas/instances/ai_smoke/interactions",
        json=body,
    )
    assert resp.status == 400
    payload = await resp.json()
    assert payload["error"] == "invalid_human_brief"


async def test_unknown_interaction_type_returns_typed_400(dlaas_client):
    """Unknown interaction_type values are rejected before reaching the kernel.

    The platform must NEVER fall back to keyword-guess dispatch — the
    contract surface is a typed enum and this is how we prove it.
    """
    body = {
        "contract_id": "ctr_smoke_003",
        "session_id": "sess_smoke_003",
        "end_user_ref": "user_smoke",
        "interaction_type": "telepathy",
    }
    resp = await dlaas_client.post(
        "/dlaas/instances/ai_smoke/interactions",
        json=body,
    )
    assert resp.status == 400
    payload = await resp.json()
    assert payload["error"] == "invalid_envelope"


async def test_chat_envelope_dispatches_to_growth_advisor_vertical(
    dlaas_client_growth_advisor,
):
    """A chat envelope reaches the growth-advisor vertical end-to-end.

    Asserts the LTV vertical is wired through the same DLaaS dispatch
    path as the companion vertical: the typed envelope is parsed, the
    router dispatches on ``interaction_type``, ``LifeformSession.run_turn``
    executes a real kernel turn, and the response uses the canonical
    DLaaS wire format.

    Together with the companion smoke this is the structural guarantee
    that "DLaaS adopt -> growth_advisor template -> chat" is reachable
    once the platform-tier ``runtime_template_id`` -> vertical mapping
    is wired in (Wave 2).
    """
    body = {
        "contract_id": "ctr_smoke_growth_001",
        "protocol_version": "dlaas/v1",
        "session_id": "sess_smoke_growth_001",
        "end_user_ref": "user_smoke_mom",
        "interaction_type": "chat",
        "mode": "live",
        "human_brief": "你好，我家娃今年5岁，男孩。",
        "structured_context": {"target_person_ids": []},
        "lang": "cn",
    }
    resp = await dlaas_client_growth_advisor.post(
        "/dlaas/instances/ai_smoke_growth_advisor/interactions",
        json=body,
    )
    assert resp.status == 200, await resp.text()
    payload = await resp.json()
    assert payload["status"] == "ok"
    assert payload["ai_id"] == "ai_smoke_growth_advisor"
    assert payload["interaction_type"] == "chat"
    assert isinstance(payload["output_acts"], list) and payload["output_acts"]
    primary = payload["output_acts"][0]
    assert primary["act_type"] == "text"
    assert isinstance(primary["payload"]["content"], str)


async def test_non_chat_types_validate_their_typed_contract(dlaas_client):
    """After Slice 2 every non-chat ``interaction_type`` is wired but
    each one validates its own typed contract surface. Sending a bare
    ``human_brief`` is no longer enough — the dispatcher refuses to
    fall through to the kernel without the typed payload.

    This is the canonical Slice 2 invariant: typed dispatch never
    silently uses defaults, so missing fields surface as typed 400s.
    """
    body_template = {
        "contract_id": "ctr_smoke_004",
        "session_id": "sess_smoke_004",
        "end_user_ref": "user_smoke",
        "human_brief": "x",
    }
    expected_codes = {
        "feedback": "missing_feedback_payload",
        "observe": "missing_observation_type",
        "command": "invalid_command",
    }
    for kind, code in expected_codes.items():
        payload_in = dict(body_template, interaction_type=kind)
        resp = await dlaas_client.post(
            "/dlaas/instances/ai_smoke/interactions",
            json=payload_in,
        )
        assert resp.status == 400, await resp.text()
        payload = await resp.json()
        assert payload["error"] == code, payload
    # ``teach`` / ``task`` accept the same ``human_brief`` as chat — they
    # reach the kernel and produce an OutputAct under the apprentice
    # trigger. ``report`` accepts an empty body and returns the drained
    # scaffold. We don't exercise those here; the Slice 7 contract
    # suite covers them with typed assertions.
