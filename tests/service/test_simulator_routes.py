"""HTTP smoke tests for the governance-demo simulator routes.

Covers the three endpoints registered by
``lifeform_service.simulator_routes``:

* ``GET /v1/scenarios`` — lists public companion-bench scenarios.
* ``POST /v1/sessions/{sid}/simulator/init`` — binds a scenario.
* ``POST /v1/sessions/{sid}/simulator/next-user-turn`` — advances
  the cached simulator one tick.

Plus the cache-eviction guarantee:

* ``DELETE /v1/sessions/{sid}`` evicts the simulator state so a stale
  POST to ``.../simulator/next-user-turn`` returns 404
  ``simulator_not_bound``.

The companion-bench utterance backend is forced to the deterministic
fake so the tests never reach an external LLM endpoint and stay
byte-stable across runs.
"""

from __future__ import annotations

import asyncio
import pathlib

import pytest

from companion_bench.user_simulator import DeterministicFakeUtteranceClient


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def all_verticals():
    from lifeform_service.verticals import discover_verticals

    return discover_verticals()


@pytest.fixture
async def sim_client(aiohttp_client, all_verticals, tmp_path):
    """Service with the fake utterance backend forced on.

    We pass ``utterance_backend=DeterministicFakeUtteranceClient()``
    explicitly so the test never depends on ``PROTOCOL_LLM_*`` env
    state and never hits an external network. Companion is used as the
    default vertical because every install ships it.
    """

    from lifeform_service.app import create_app

    if "companion" not in all_verticals:
        pytest.skip("companion vertical not installed")
    app = create_app(
        verticals=all_verticals,
        default_vertical="companion",
        max_sessions=8,
        idle_eviction_seconds=None,
        templates_root_dir=str(tmp_path / "templates"),
        utterance_backend=DeterministicFakeUtteranceClient(),
    )
    return await aiohttp_client(app)


# ---------------------------------------------------------------------------
# GET /v1/scenarios
# ---------------------------------------------------------------------------


async def test_list_scenarios_returns_public_scenarios(sim_client):
    resp = await sim_client.get("/v1/scenarios")
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert isinstance(body["scenarios"], list)
    assert body["scenarios"], "expected at least one public scenario"
    by_id = {row["scenario_id"]: row for row in body["scenarios"]}
    # F1-continuity-001 is the canonical first public scenario; if the
    # wheel ever drops it the suite needs to know.
    assert "F1-continuity-001" in by_id
    row = by_id["F1-continuity-001"]
    assert row["family"] == "F1"
    assert row["language"] == "en"
    assert row["arc_length_sessions"] >= 2
    assert row["paraphrase_seed_count"] >= 1
    assert isinstance(row["session_turn_range"], list)


async def test_list_scenarios_family_filter(sim_client):
    resp = await sim_client.get("/v1/scenarios?family=F2")
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["scenarios"], "expected at least one F2 scenario"
    for row in body["scenarios"]:
        assert row["family"] == "F2"


async def test_list_scenarios_language_filter(sim_client):
    resp = await sim_client.get("/v1/scenarios?language=zh")
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    for row in body["scenarios"]:
        assert row["language"] == "zh"


# ---------------------------------------------------------------------------
# POST /v1/sessions/{sid}/simulator/init
# ---------------------------------------------------------------------------


async def _create_session(sim_client) -> str:
    resp = await sim_client.post("/v1/sessions", json={})
    assert resp.status == 201, await resp.text()
    body = await resp.json()
    return body["session_id"]


async def test_simulator_init_binds_schedule(sim_client):
    sid = await _create_session(sim_client)
    resp = await sim_client.post(
        f"/v1/sessions/{sid}/simulator/init",
        json={"scenario_id": "F1-continuity-001", "paraphrase_seed": 0},
    )
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert body["scenario_id"] == "F1-continuity-001"
    assert body["paraphrase_seed"] == 0
    assert body["arc_length_sessions"] >= 2
    assert len(body["schedule"]) == body["arc_length_sessions"]
    assert body["cursor"] == 0
    assert body["resumed_from_session_id"] is None
    assert body["identity"]["name"]
    # Schedule rows must declare the correct gap_days for each session.
    schedule = body["schedule"]
    assert schedule[0]["gap_days"] == 0


async def test_simulator_init_404s_on_unknown_scenario(sim_client):
    sid = await _create_session(sim_client)
    resp = await sim_client.post(
        f"/v1/sessions/{sid}/simulator/init",
        json={"scenario_id": "no-such-scenario", "paraphrase_seed": 0},
    )
    assert resp.status == 404, await resp.text()
    body = await resp.json()
    assert body["error"] == "unknown_scenario_or_session"


async def test_simulator_init_400s_on_invalid_seed(sim_client):
    sid = await _create_session(sim_client)
    resp = await sim_client.post(
        f"/v1/sessions/{sid}/simulator/init",
        json={"scenario_id": "F1-continuity-001", "paraphrase_seed": 9999},
    )
    assert resp.status == 400, await resp.text()
    body = await resp.json()
    assert body["error"] == "invalid_simulator_init"


# ---------------------------------------------------------------------------
# POST /v1/sessions/{sid}/simulator/next-user-turn
# ---------------------------------------------------------------------------


async def test_next_user_turn_emits_fsm_action_at_scripted_coordinate(
    sim_client,
):
    sid = await _create_session(sim_client)
    init_body = await (
        await sim_client.post(
            f"/v1/sessions/{sid}/simulator/init",
            json={
                "scenario_id": "F1-continuity-001",
                "paraphrase_seed": 0,
            },
        )
    ).json()
    # Walk the first session's worth of turns; F1-continuity-001 fires
    # ``establish_pattern`` at (session=1, turn=2) per the public YAML.
    first_session_count = init_body["schedule"][0]["turn_count"]
    seen_fsm_actions: list[str] = []
    for i in range(first_session_count):
        body_in = (
            {"recent_assistant_text": "[fake bot reply]"}
            if i > 0
            else {}
        )
        resp = await sim_client.post(
            f"/v1/sessions/{sid}/simulator/next-user-turn",
            json=body_in,
        )
        assert resp.status == 200, await resp.text()
        body = await resp.json()
        assert body["user_text"].strip()
        if body["fsm_step"] is not None:
            seen_fsm_actions.append(body["fsm_step"]["action"])
        assert body["session_index"] == 1
        assert body["turn_index"] == i + 1
    assert "establish_pattern" in seen_fsm_actions


async def test_next_user_turn_404s_when_not_bound(sim_client):
    sid = await _create_session(sim_client)
    resp = await sim_client.post(
        f"/v1/sessions/{sid}/simulator/next-user-turn",
        json={},
    )
    assert resp.status == 404, await resp.text()
    body = await resp.json()
    assert body["error"] == "simulator_not_bound"


async def test_arc_position_flips_at_session_boundary(sim_client):
    sid = await _create_session(sim_client)
    init_body = await (
        await sim_client.post(
            f"/v1/sessions/{sid}/simulator/init",
            json={
                "scenario_id": "F1-continuity-001",
                "paraphrase_seed": 0,
            },
        )
    ).json()
    schedule = init_body["schedule"]
    first_session_count = schedule[0]["turn_count"]
    last_arc_position = None
    for i in range(first_session_count):
        body_in = (
            {"recent_assistant_text": "[fake bot reply]"}
            if i > 0
            else {}
        )
        resp = await sim_client.post(
            f"/v1/sessions/{sid}/simulator/next-user-turn",
            json=body_in,
        )
        body = await resp.json()
        last_arc_position = body["arc_position"]
    if init_body["arc_length_sessions"] > 1:
        assert last_arc_position == "session_end"
        # The reported gap should match the spec for the next session.
        # We don't hard-code the value (scenarios vary) but assert it
        # is positive: by spec session 2's gap_days >= 1.
        assert (
            body["next_gap_days"] == schedule[1]["gap_days"]  # type: ignore[possibly-undefined]
        )


# ---------------------------------------------------------------------------
# Cache eviction
# ---------------------------------------------------------------------------


async def test_close_session_evicts_simulator_state(sim_client):
    sid = await _create_session(sim_client)
    resp = await sim_client.post(
        f"/v1/sessions/{sid}/simulator/init",
        json={"scenario_id": "F1-continuity-001", "paraphrase_seed": 0},
    )
    assert resp.status == 200
    delete_resp = await sim_client.delete(f"/v1/sessions/{sid}")
    assert delete_resp.status == 200, await delete_resp.text()
    # Subsequent /next-user-turn must report "not bound" — even though
    # the session_id technically existed earlier. The cache is the
    # only owner; deleting the session is the only release path.
    follow = await sim_client.post(
        f"/v1/sessions/{sid}/simulator/next-user-turn",
        json={},
    )
    # After delete the session itself is also gone; the route uses
    # the cache lookup first, which yields 404 simulator_not_bound.
    # If the session-state lookup happens to win the race, we still
    # accept its 404 session_not_found because the meaning is the
    # same operator-facing signal: this simulator is unavailable.
    assert follow.status == 404, await follow.text()
    body = await follow.json()
    assert body["error"] in {"simulator_not_bound", "session_not_found"}


# ---------------------------------------------------------------------------
# resume_from_session_id (cross arc-session transfer)
# ---------------------------------------------------------------------------


async def test_resume_from_session_id_transfers_simulator_state(sim_client):
    sid_a = await _create_session(sim_client)
    init_a = await (
        await sim_client.post(
            f"/v1/sessions/{sid_a}/simulator/init",
            json={
                "scenario_id": "F1-continuity-001",
                "paraphrase_seed": 0,
            },
        )
    ).json()
    # Advance the simulator by one turn under sid_a.
    advance = await sim_client.post(
        f"/v1/sessions/{sid_a}/simulator/next-user-turn",
        json={},
    )
    assert advance.status == 200
    cursor_after_a = (await advance.json())["turn_index"]
    assert cursor_after_a == 1

    sid_b = await _create_session(sim_client)
    init_b_resp = await sim_client.post(
        f"/v1/sessions/{sid_b}/simulator/init",
        json={
            "scenario_id": "F1-continuity-001",
            "paraphrase_seed": 0,
            "resume_from_session_id": sid_a,
            "recent_assistant_text": "[fake bot reply for sid_a turn 1]",
        },
    )
    assert init_b_resp.status == 200, await init_b_resp.text()
    init_b = await init_b_resp.json()
    assert init_b["resumed_from_session_id"] == sid_a
    # Cursor was 1 on sid_a; it must remain 1 on sid_b (state moved,
    # not duplicated).
    assert init_b["cursor"] == 1

    # sid_a's cache binding was dropped, so a follow-up next-user-turn
    # on sid_a must report unbound.
    follow = await sim_client.post(
        f"/v1/sessions/{sid_a}/simulator/next-user-turn",
        json={},
    )
    assert follow.status == 404, await follow.text()
    follow_body = await follow.json()
    assert follow_body["error"] == "simulator_not_bound"


# ---------------------------------------------------------------------------
# Static guardrail: scenarios package data ships with the wheel
# ---------------------------------------------------------------------------


def test_companion_bench_ships_public_scenarios_resource():
    """If the package_data stanza is dropped the route returns []."""

    from lifeform_service.simulator_routes import _public_scenarios_dir

    d = _public_scenarios_dir()
    assert isinstance(d, pathlib.Path)
    assert d.exists(), f"scenarios resource missing on disk: {d}"
    yamls = list(d.rglob("*.yaml"))
    assert yamls, "no scenario YAMLs found inside companion-bench wheel"
