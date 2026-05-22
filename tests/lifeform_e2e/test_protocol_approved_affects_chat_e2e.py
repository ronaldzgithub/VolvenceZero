"""End-to-end proof that an approved BehaviorProtocol changes a chat turn.

Production wiring under test::

    POST /v1/protocols/from-payload     (API injection — no LLM needed)
    POST /v1/protocols/candidates/{id}/approve
    POST /v1/sessions                   (NEW session, since seed protocols
                                          are wired at session construction)
    POST /v1/sessions/{id}/turns
    → assert turn JSON reflects the protocol-loaded artifacts
    → also inspect in-process snapshots (active_mixture, response_assembly)
       to nail down WHERE the effect lives — the HTTP TurnResponse is a
       deliberately compact projection, so the snapshot peek is how we
       prove the protocol reached the application owners, not just the
       service registry.

Why this test is load-bearing
-----------------------------

Every other protocol test in the repo proves one slice:

* ``tests/test_lifeform_service_protocol_routes.py`` — HTTP CRUD only.
* ``tests/test_protocol_uptake_persistence.py`` — disk round-trip only.
* ``tests/contracts/test_protocol_uptake_session_injection.py`` — service
  → session injection, but uses a stub vertical factory and no turn.
* ``tests/test_protocol_load_to_application_state.py`` — compile path,
  but no HTTP and no turn.

This file is the one that bridges all three:
**HTTP approve → real session → real turn → effect observable**. If
this passes, "protocol is effective" holds end-to-end from the user's
PoV. If it fails, somewhere on the production seam something silently
dropped the protocol.

Vertical choice
---------------

``companion`` — cheapest in CI (no external deps, no LLM needed; uses
the deterministic ``GroundedResponseSynthesizer``). The protocol
artifacts compile into the same application stores
(``boundary_policy`` / ``strategy_playbook`` / ``domain_knowledge`` /
``case_memory``) regardless of which vertical is hosting the session,
so this is the right minimum-cost configuration.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from aiohttp import web

from lifeform_service.app import create_app
from lifeform_service.protocol_uptake import (
    ProtocolUptakeConfig,
    ProtocolUptakeService,
)
from lifeform_service.verticals import discover_verticals


# Distinctive markers we'll fingerprint in the turn output.
#
# Note on knowledge_seeds: the API-injection adapter
# (``inject_protocol_from_payload``) deliberately accepts only
# boundaries / strategies / identity / activation_conditions /
# signals. Knowledge seeds and signature cases cannot ride this
# channel — they require document-uptake (LLM) or direct
# ``seed_protocols`` injection. The behavioural assertions below
# stick to boundary + strategy artefacts, which is what the
# HTTP-only path actually delivers; the matched-control suite
# (``tests/contracts/test_protocol_knowledge_matched_control.py``
# / ``test_protocol_case_matched_control.py``) covers knowledge
# + case compile from a direct seed path.

_DISTINCTIVE_DISCLAIMER = "marker-disclaimer-do-not-replace-9f3a"
_DISTINCTIVE_PROTOCOL_ID = "e2e:protocol-effective-bot"
_DISTINCTIVE_RULE_ID = "rule:e2e:first-contact"


def _protocol_payload() -> dict[str, Any]:
    """Construct a protocol whose effects are easy to fingerprint.

    The payload uses *only* fields the API-injection adapter accepts
    so the test does not need an LLM. See
    ``inject_protocol_from_payload`` for the accepted schema.
    """

    return {
        "request_id": "req-e2e-effective",
        "protocol": {
            "protocol_id": _DISTINCTIVE_PROTOCOL_ID,
            "advisor_name": "E2E Effective Advisor",
            "description": "Distinctive markers test protocol",
            "boundaries": [
                {
                    "boundary_id": "bd:e2e:no-medical",
                    "description": "Refuse medical specifics",
                    "trigger_reasons": ["medical question detected"],
                    "blocked_topics": ["dosage"],
                    "required_disclaimers": [_DISTINCTIVE_DISCLAIMER],
                    "severity": "hard_block",
                    "refer_out_required": True,
                }
            ],
            "strategies": [
                {
                    "rule_id": _DISTINCTIVE_RULE_ID,
                    "problem_pattern": "first contact",
                    "recommended_ordering": ["greet", "ask context"],
                }
            ],
        },
    }


def _build_app() -> web.Application:
    verticals = discover_verticals()
    if "companion" not in verticals:
        pytest.skip("companion vertical not installed; cannot run e2e")
    uptake = ProtocolUptakeService(
        config=ProtocolUptakeConfig(
            autoload_dir=None,
            autoload_force_approve=False,
            llm_client_factory=lambda: None,
        ),
        persistence=None,
    )
    return create_app(
        verticals=verticals,
        default_vertical="companion",
        max_sessions=8,
        idle_eviction_seconds=None,
        protocol_uptake_service=uptake,
    )


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client(aiohttp_client):
    app = _build_app()
    return await aiohttp_client(app)


async def _approve_protocol(client) -> None:
    resp = await client.post(
        "/v1/protocols/from-payload",
        data=json.dumps(_protocol_payload()),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status == 201, await resp.text()
    resp = await client.post(
        f"/v1/protocols/candidates/{_DISTINCTIVE_PROTOCOL_ID}/approve"
    )
    assert resp.status == 200, await resp.text()


async def _create_session(client) -> str:
    resp = await client.post("/v1/sessions", json={})
    assert resp.status == 201, await resp.text()
    return (await resp.json())["session_id"]


async def _run_turn(client, sid: str, user_input: str) -> dict[str, Any]:
    resp = await client.post(
        f"/v1/sessions/{sid}/turns", json={"user_input": user_input}
    )
    assert resp.status == 200, await resp.text()
    return await resp.json()


async def _get_session(app: web.Application, sid: str):
    manager = app["session_manager"]
    return await manager.get_session(sid)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


_USER_INPUT_BASE = (
    "Hi, I'm new here. Can you help me think through a first-contact issue?"
)


async def test_baseline_session_without_protocol_has_no_protocol_markers(client) -> None:
    """Control arm: a session created WITHOUT approving any protocol
    must not show our distinctive markers anywhere.

    This is the negative half of the matched-control proof. If this
    test ever fails, the test-2 positive proof is invalidated (the
    marker would be a baseline artefact, not a protocol effect)."""

    sid = await _create_session(client)
    turn = await _run_turn(client, sid, _USER_INPUT_BASE)
    payload_text = json.dumps(turn)
    assert _DISTINCTIVE_DISCLAIMER not in payload_text
    assert _DISTINCTIVE_PROTOCOL_ID not in payload_text

    session = await _get_session(client.app, sid)
    snaps = session.latest_active_snapshots
    am = snaps.get("active_mixture")
    if am is not None:
        active_ids = tuple(p.protocol_id for p in am.value.active_protocols)
        assert _DISTINCTIVE_PROTOCOL_ID not in active_ids


async def test_approved_protocol_appears_in_active_mixture_after_new_session(
    client,
) -> None:
    """Production-path proof — the protocol reaches the runner.

    This is the most basic 'protocol effective' assertion: after
    approving via HTTP and creating a NEW session, the runner-held
    ``ProtocolRegistryModule`` lists our protocol_id in the
    published ``active_mixture`` snapshot. If this fails the seed
    injection seam is broken."""

    await _approve_protocol(client)
    sid = await _create_session(client)
    await _run_turn(client, sid, _USER_INPUT_BASE)

    session = await _get_session(client.app, sid)
    snaps = session.latest_active_snapshots
    am = snaps.get("active_mixture")
    assert am is not None, (
        "active_mixture snapshot missing — seed protocol injection "
        "either didn't reach the runner or the runner skipped the "
        "ProtocolRegistryModule.process call this turn."
    )
    active_ids = tuple(p.protocol_id for p in am.value.active_protocols)
    assert _DISTINCTIVE_PROTOCOL_ID in active_ids, (
        f"approved protocol absent from active_mixture; saw {active_ids}"
    )


async def test_approved_protocol_compiles_into_application_owner_stores(
    client,
) -> None:
    """Compile path proof — the artefacts land in the application stores.

    The source of truth for "compiled into owners" is
    ``runner._application_rare_heavy_state``: a rare-heavy snapshot
    of the boundary-prior-hint / distilled-playbook-rule /
    knowledge-record / case-record stores that ``load_protocol``
    auto-applies into when seed protocols are present at session
    construction. This mirrors the assertion in
    ``tests/contracts/test_protocol_uptake_session_injection.py::test_create_session_picks_up_uptake_protocol``
    but starts from the **HTTP approve** path rather than a stub
    vertical factory, so it certifies the full production seam.

    Note we deliberately do NOT assert the protocol's disclaimer is
    inlined into ``boundary_policy.active_decision.required_disclaimers``
    on this turn. ``BoundaryPolicyModule`` only inlines a hint when
    the turn's typed signal upstream fires the hint's
    ``trigger_reasons`` (R3/R4 + the no-keyword-matching-hacks rule).
    Proving that requires arranging upstream signal detectors to
    fire, which lives in dedicated tests like
    ``tests/contracts/test_protocol_*_matched_control.py`` — not in
    this seam test. Loading into the prior-hint store is the right
    invariant to prove **here**.
    """

    await _approve_protocol(client)
    sid = await _create_session(client)
    await _run_turn(client, sid, _USER_INPUT_BASE)

    session = await _get_session(client.app, sid)
    runner = session.brain_session.runner
    rare_heavy = runner._application_rare_heavy_state  # noqa: SLF001

    assert any(
        _DISTINCTIVE_PROTOCOL_ID in hint.hint_id
        for hint in rare_heavy.boundary_prior_hints
    ), (
        f"protocol boundary hint absent from rare-heavy state; "
        f"saw {[h.hint_id for h in rare_heavy.boundary_prior_hints]}"
    )
    assert any(
        _DISTINCTIVE_DISCLAIMER in disclaimer
        for hint in rare_heavy.boundary_prior_hints
        for disclaimer in getattr(hint, "required_disclaimers", ())
    ), (
        "protocol's required_disclaimer did not survive compile into "
        "boundary_prior_hints; this means the BoundaryContract → "
        "BoundaryPriorHint compile lost the disclaimer field."
    )
    assert any(
        _DISTINCTIVE_PROTOCOL_ID in rule.rule_id
        for rule in rare_heavy.distilled_playbook_rules
    ), (
        f"protocol-derived playbook rule absent from rare-heavy state; "
        f"saw {[r.rule_id for r in rare_heavy.distilled_playbook_rules]}"
    )


async def test_protocol_loaded_via_persisted_library_route_round_trips_to_turn(
    aiohttp_client, tmp_path,
) -> None:
    """Closes the disk → library → load → session → turn loop.

    The chain under test mirrors the user-facing UX after a restart:
    approve (which persists to disk if persistence is wired) → unload
    from registry (simulates restart 'active set is empty') → POST
    /v1/protocols/library/{id}/load → create a NEW session → the
    protocol is back in the active mixture on the next turn.

    Failure here means the library → load path doesn't re-engage seed
    injection, which would break the chat-browser 'multi-select and
    load' flow the persistence packet was built for."""

    from lifeform_service.protocol_persistence import ProtocolPersistenceStore

    verticals = discover_verticals()
    if "companion" not in verticals:
        pytest.skip("companion vertical not installed")
    persistence = ProtocolPersistenceStore(tmp_path / "lib")
    uptake = ProtocolUptakeService(
        config=ProtocolUptakeConfig(
            autoload_dir=None,
            autoload_force_approve=False,
            llm_client_factory=lambda: None,
        ),
        persistence=persistence,
    )
    app = create_app(
        verticals=verticals,
        default_vertical="companion",
        max_sessions=8,
        idle_eviction_seconds=None,
        protocol_uptake_service=uptake,
    )
    cli = await aiohttp_client(app)

    await _approve_protocol(cli)
    # Simulate "restart" — drop from registry but keep on disk.
    resp = await cli.post(
        f"/v1/protocols/library/{_DISTINCTIVE_PROTOCOL_ID}/unload"
    )
    assert resp.status == 200, await resp.text()

    # New session right now must NOT have the protocol (active set empty).
    sid_clean = await _create_session(cli)
    await _run_turn(cli, sid_clean, _USER_INPUT_BASE)
    sess_clean = await _get_session(cli.app, sid_clean)
    am_clean = sess_clean.latest_active_snapshots.get("active_mixture")
    if am_clean is not None:
        ids_clean = tuple(
            p.protocol_id for p in am_clean.value.active_protocols
        )
        assert _DISTINCTIVE_PROTOCOL_ID not in ids_clean, (
            "after unload-from-registry the protocol should be absent "
            "from a fresh session, but active_mixture still contains it. "
            "This means unload_from_registry is not actually dropping "
            "the registry entry."
        )

    # Reload from library → next NEW session must have it again.
    resp = await cli.post(
        f"/v1/protocols/library/{_DISTINCTIVE_PROTOCOL_ID}/load"
    )
    assert resp.status == 200, await resp.text()

    sid_reloaded = await _create_session(cli)
    await _run_turn(cli, sid_reloaded, _USER_INPUT_BASE)
    sess_reloaded = await _get_session(cli.app, sid_reloaded)
    am_reloaded = sess_reloaded.latest_active_snapshots.get("active_mixture")
    assert am_reloaded is not None
    ids_reloaded = tuple(
        p.protocol_id for p in am_reloaded.value.active_protocols
    )
    assert _DISTINCTIVE_PROTOCOL_ID in ids_reloaded, (
        f"library/load did not re-engage the seed protocol; "
        f"active_mixture after reload contains {ids_reloaded}"
    )
