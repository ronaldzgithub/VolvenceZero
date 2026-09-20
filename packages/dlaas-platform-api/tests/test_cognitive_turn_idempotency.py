from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import time
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

from dlaas_platform_api import app as app_module
from dlaas_platform_api.app import attach_dlaas_routes
from dlaas_platform_contracts import InteractionEnvelope, OutputContract
from dlaas_platform_launcher import INSTANCE_MANAGER_APP_KEY
from dlaas_platform_registry import (
    CognitiveTurnLedgerStore,
    CognitiveTurnLedgerTransitionError,
    Registry,
)
from lifeform_service import SessionManager
from volvence_zero.substrate import (
    OpenWeightResidualStreamSubstrateAdapter,
    SyntheticOpenWeightResidualRuntime,
)


class _BlockingCaptureRuntime:
    model_id = "blocking-capture-runtime"
    is_frozen = True
    runtime_origin = "test"
    fallback_active = False
    capture_source = "blocking-test"

    def __init__(self, *, delay_seconds: float, failure: Exception | None = None) -> None:
        self._delegate = SyntheticOpenWeightResidualRuntime(model_id=self.model_id)
        self._delay_seconds = delay_seconds
        self._failure = failure
        self.calls = 0

    def capture(self, *, source_text: str):
        self.calls += 1
        time.sleep(self._delay_seconds)
        if self._failure is not None:
            raise self._failure
        return self._delegate.capture(source_text=source_text)


class _BlockingSessionLifeform:
    def __init__(self, *, session_delay_seconds: float) -> None:
        self._session_delay_seconds = session_delay_seconds

    async def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    def create_session(self, *, session_id: str) -> object:
        time.sleep(self._session_delay_seconds)
        return object()


def _payload(*, perception: str = "I saw the traveller cut the rope.") -> dict[str, Any]:
    return {
        "contract_id": "contract-1",
        "session_id": "lifeform-qiao",
        "end_user_ref": "player-1",
        "interaction_type": "cognitive_turn",
        "perceived_event": {
            "action_id": "action-17",
            "perception": perception,
            "frame": {
                "actor": {
                    "actor_id": "player-1",
                    "actor_kind": "player_character",
                    "display_name": "Traveller",
                },
                "active_speaker_id": "player-1",
                "addressee_ids": ["qiao"],
                "subject_ids": ["player-1"],
                "audience_ids": ["qiao"],
            },
            "provenance": "novel-worlds:observer:qiao:action-17",
        },
        "cognition_task": {
            "kind": "choose_observable_action",
            "required_readouts": ["response_action_realization"],
        },
        "expression_contract": {
            "schema_name": "lifeform_intent",
            "schema": {
                "type": "object",
                "properties": {"intended_action": {"type": "string"}},
                "required": ["intended_action"],
            },
            "strict": True,
            "exact_bindings": [
                {
                    "json_pointer": "/intended_action",
                    "source": "response_action_realization.action_statement",
                }
            ],
        },
    }


def _envelope(*, perception: str = "I saw the traveller cut the rope.") -> InteractionEnvelope:
    return InteractionEnvelope.from_json(_payload(perception=perception))


def _request(store: CognitiveTurnLedgerStore | None) -> web.Request:
    app = web.Application()
    if store is not None:
        app[app_module._COGNITIVE_TURN_LEDGER_STORE_KEY] = store
    return make_mocked_request("POST", "/dlaas/v1/instances/ai-qiao/interactions", app=app)


def _json(response: web.StreamResponse) -> dict[str, Any]:
    assert isinstance(response, web.Response)
    assert response.body is not None
    return json.loads(response.body)


async def test_single_execution_replay_and_payload_conflict(monkeypatch) -> None:
    store = CognitiveTurnLedgerStore(Registry())
    request = _request(store)
    run_turn_calls = 0

    async def fake_dispatch(*_args, **_kwargs):
        nonlocal run_turn_calls
        run_turn_calls += 1
        return web.json_response({"status": "ok", "output_acts": [{"payload": {"content": "hold"}}]})

    monkeypatch.setattr(app_module, "_dispatch_envelope_to_instance", fake_dispatch)
    first = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-1",
    )
    replay = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=replace(
            _envelope(),
            output_contract=OutputContract(delivery_channel="wechat", format="markdown", stream=False),
        ),
        idempotency_key="turn-1",
    )
    conflict = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=_envelope(perception="I heard the rope snap."),
        idempotency_key="turn-1",
    )

    assert first.status == replay.status == 200
    assert _json(first) == _json(replay)
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert conflict.status == 409
    assert _json(conflict)["error"] == "idempotency_key_payload_conflict"
    assert run_turn_calls == 1


async def test_active_lease_returns_in_progress_while_heartbeat_runs(
    monkeypatch,
) -> None:
    store = CognitiveTurnLedgerStore(Registry(), lease_ms=30)
    request = _request(store)
    run_turn_calls = 0

    async def slow_dispatch(*_args, **_kwargs):
        nonlocal run_turn_calls
        run_turn_calls += 1
        await asyncio.sleep(0.09)
        return web.json_response({"status": "ok"})

    monkeypatch.setattr(app_module, "_dispatch_envelope_to_instance", slow_dispatch)
    first_task = asyncio.create_task(
        app_module._dispatch_keyed_cognitive_turn(
            request,
            ai_id="ai-qiao",
            envelope=_envelope(),
            idempotency_key="turn-slow",
        )
    )
    await asyncio.sleep(0.055)
    concurrent = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-slow",
    )
    assert concurrent.status == 409
    assert _json(concurrent)["error"] == "cognitive_turn_in_progress"
    assert (await first_task).status == 200
    assert run_turn_calls == 1


async def test_blocking_capture_is_offloaded_while_heartbeat_completes_and_replays(
    monkeypatch,
) -> None:
    store = CognitiveTurnLedgerStore(Registry(), lease_ms=60)
    request = _request(store)
    runtime = _BlockingCaptureRuntime(delay_seconds=0.22)
    adapter = OpenWeightResidualStreamSubstrateAdapter(runtime=runtime)
    heartbeat_count = 0
    original_refresh = store.refresh_lease

    async def counted_refresh(**kwargs):
        nonlocal heartbeat_count
        heartbeat_count += 1
        return await original_refresh(**kwargs)

    async def blocking_dispatch(*_args, **_kwargs):
        await adapter.capture(source_text="a scene event that blocks inference")
        return web.json_response({"status": "ok"})

    monkeypatch.setattr(store, "refresh_lease", counted_refresh)
    monkeypatch.setattr(app_module, "_dispatch_envelope_to_instance", blocking_dispatch)
    first = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-blocking-capture",
    )
    replay = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-blocking-capture",
    )

    record = store.get(
        contract_id="contract-1",
        ai_id="ai-qiao",
        idempotency_key="turn-blocking-capture",
    )
    assert first.status == replay.status == 200
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert heartbeat_count >= 2
    assert runtime.calls == 1
    assert record is not None
    assert record.status.value == "COMPLETED"
    assert record.lease_expires_at_ms - record.created_at_ms > 60


async def test_fresh_session_construction_keeps_ledger_heartbeat_alive(
    monkeypatch,
) -> None:
    store = CognitiveTurnLedgerStore(Registry(), lease_ms=60)
    request = _request(store)
    heartbeat_count = 0
    factory_calls = 0
    original_refresh = store.refresh_lease

    def blocking_factory(_runtime) -> _BlockingSessionLifeform:
        nonlocal factory_calls
        factory_calls += 1
        time.sleep(0.12)
        return _BlockingSessionLifeform(session_delay_seconds=0.12)

    manager = SessionManager(
        lifeform_factory=blocking_factory,
        vertical_name="blocking-test",
        idle_eviction_seconds=None,
    )

    async def counted_refresh(**kwargs):
        nonlocal heartbeat_count
        heartbeat_count += 1
        return await original_refresh(**kwargs)

    async def dispatch_with_fresh_session(*_args, **_kwargs):
        await app_module._get_or_create_session(
            manager,
            "fresh-cognitive-session",
            user_id="player-1",
        )
        return web.json_response({"status": "ok"})

    monkeypatch.setattr(store, "refresh_lease", counted_refresh)
    monkeypatch.setattr(
        app_module,
        "_dispatch_envelope_to_instance",
        dispatch_with_fresh_session,
    )
    first = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-fresh-session",
    )
    replay = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-fresh-session",
    )

    record = store.get(
        contract_id="contract-1",
        ai_id="ai-qiao",
        idempotency_key="turn-fresh-session",
    )
    assert first.status == replay.status == 200
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert heartbeat_count >= 2
    assert factory_calls == 1
    assert record is not None
    assert record.status.value == "COMPLETED"


async def test_late_heartbeat_recovers_after_synchronous_event_loop_stall(
    monkeypatch,
) -> None:
    store = CognitiveTurnLedgerStore(Registry(), lease_ms=60)
    request = _request(store)
    dispatch_calls = 0

    async def blocking_dispatch(*_args, **_kwargs):
        nonlocal dispatch_calls
        dispatch_calls += 1
        time.sleep(0.12)
        return web.json_response({"status": "ok", "turn": "persisted-once"})

    monkeypatch.setattr(
        app_module,
        "_dispatch_envelope_to_instance",
        blocking_dispatch,
    )
    first = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-event-loop-stall",
    )
    replay = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-event-loop-stall",
    )

    assert first.status == replay.status == 200
    assert _json(first) == {"status": "ok", "turn": "persisted-once"}
    assert _json(replay) == _json(first)
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert dispatch_calls == 1


async def test_offloaded_capture_exception_is_unknown_logs_type_and_never_retries(
    monkeypatch,
    caplog,
) -> None:
    store = CognitiveTurnLedgerStore(Registry(), lease_ms=60)
    request = _request(store)
    runtime = _BlockingCaptureRuntime(
        delay_seconds=0.08,
        failure=RuntimeError("provider-secret=must-not-enter-log"),
    )
    adapter = OpenWeightResidualStreamSubstrateAdapter(runtime=runtime)

    async def failed_dispatch(*_args, **_kwargs):
        await adapter.capture(source_text="failing scene event")
        raise AssertionError("capture failure should have propagated")

    monkeypatch.setattr(app_module, "_dispatch_envelope_to_instance", failed_dispatch)
    first = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-offloaded-exception",
    )
    retry = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-offloaded-exception",
    )

    assert first.status == retry.status == 409
    assert _json(first)["unknown_reason"] == "dispatch_exception_or_lease_failure"
    assert runtime.calls == 1
    assert "cause_type=RuntimeError" in caplog.text
    assert "provider-secret" not in caplog.text


async def test_exception_is_immediately_unknown_and_retry_never_runs_turn(
    monkeypatch,
) -> None:
    store = CognitiveTurnLedgerStore(Registry())
    request = _request(store)
    run_turn_calls = 0

    async def failed_dispatch(*_args, **_kwargs):
        nonlocal run_turn_calls
        run_turn_calls += 1
        raise RuntimeError("provider failed after turn start")

    monkeypatch.setattr(app_module, "_dispatch_envelope_to_instance", failed_dispatch)
    first = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-exception",
    )
    retry = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-exception",
    )
    assert first.status == retry.status == 409
    assert _json(first) == {
        "status": "error",
        "error": "cognitive_turn_outcome_unknown",
        "detail": "automatic cognitive-turn replay is permanently disabled",
        "unknown_reason": "dispatch_exception_or_lease_failure",
    }
    assert _json(retry)["error"] == "cognitive_turn_outcome_unknown"
    assert run_turn_calls == 1


async def test_upstream_5xx_and_unrecordable_response_become_unknown(
    monkeypatch,
) -> None:
    store = CognitiveTurnLedgerStore(Registry())
    request = _request(store)
    responses = [
        web.json_response({"status": "error"}, status=502),
        web.Response(body=b"not-json", content_type="text/plain"),
    ]

    for index, dispatched in enumerate(responses):

        async def fake_dispatch(*_args, _response=dispatched, **_kwargs):
            return _response

        monkeypatch.setattr(app_module, "_dispatch_envelope_to_instance", fake_dispatch)
        response = await app_module._dispatch_keyed_cognitive_turn(
            request,
            ai_id="ai-qiao",
            envelope=_envelope(perception=f"perception-{index}"),
            idempotency_key=f"turn-failure-{index}",
        )
        assert response.status == 409
        assert _json(response)["error"] == "cognitive_turn_outcome_unknown"
        expected = "nondeterministic_http_status_502" if index == 0 else "response_unrecordable"
        assert _json(response)["unknown_reason"] == expected


async def test_deterministic_4xx_is_persisted_and_replayed(monkeypatch) -> None:
    store = CognitiveTurnLedgerStore(Registry())
    request = _request(store)
    calls = 0

    async def deterministic_failure(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return web.json_response(
            {"status": "error", "error": "session_template_binding_mismatch"},
            status=409,
        )

    monkeypatch.setattr(app_module, "_dispatch_envelope_to_instance", deterministic_failure)
    first = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-4xx",
    )
    replay = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-4xx",
    )
    assert first.status == replay.status == 409
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert calls == 1


async def test_missing_ledger_returns_503_before_dispatch(monkeypatch) -> None:
    called = False

    async def fake_dispatch(*_args, **_kwargs):
        nonlocal called
        called = True
        return web.json_response({"status": "ok"})

    monkeypatch.setattr(app_module, "_dispatch_envelope_to_instance", fake_dispatch)
    response = await app_module._dispatch_keyed_cognitive_turn(
        _request(None),
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-no-ledger",
    )
    assert response.status == 503
    assert _json(response)["error"] == "cognitive_turn_ledger_unavailable"
    assert called is False


async def test_public_route_requires_valid_idempotency_key() -> None:
    app = web.Application()
    app["session_manager"] = object()
    attach_dlaas_routes(app)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        missing = await client.post("/dlaas/v1/instances/ai-qiao/interactions", json=_payload())
        invalid = await client.post(
            "/dlaas/v1/instances/ai-qiao/interactions",
            json=_payload(),
            headers={"Idempotency-Key": " "},
        )
        assert missing.status == invalid.status == 400
        assert (await missing.json())["error"] == "invalid_idempotency_key"
        assert (await invalid.json())["error"] == "invalid_idempotency_key"
    finally:
        await client.close()


async def test_completion_cas_failure_becomes_unknown_or_finalize_503(
    monkeypatch,
) -> None:
    store = CognitiveTurnLedgerStore(Registry())
    request = _request(store)

    async def success(*_args, **_kwargs):
        return web.json_response({"status": "ok"})

    async def fail_complete(**_kwargs):
        raise CognitiveTurnLedgerTransitionError("lost CAS")

    monkeypatch.setattr(app_module, "_dispatch_envelope_to_instance", success)
    monkeypatch.setattr(store, "complete", fail_complete)
    response = await app_module._dispatch_keyed_cognitive_turn(
        request,
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-cas",
    )
    assert response.status == 409
    assert _json(response)["unknown_reason"] == "completion_compare_and_set_failed"

    failed_store = CognitiveTurnLedgerStore(Registry())

    async def fail_unknown(**_kwargs):
        raise CognitiveTurnLedgerTransitionError("cannot persist UNKNOWN")

    monkeypatch.setattr(failed_store, "complete", fail_complete)
    monkeypatch.setattr(failed_store, "mark_outcome_unknown", fail_unknown)
    failed = await app_module._dispatch_keyed_cognitive_turn(
        _request(failed_store),
        ai_id="ai-qiao",
        envelope=_envelope(),
        idempotency_key="turn-cas-failed",
    )
    assert failed.status == 503
    assert _json(failed)["error"] == "cognitive_turn_ledger_finalize_failed"


async def test_multi_pod_uses_trusted_cognitive_forward() -> None:
    class _Launcher:
        trusted_calls = 0

        async def forward_interaction(self, **_kwargs):
            raise AssertionError("reserved cognitive turn must not use public forwarding")

        async def forward_cognitive_turn(self, **_kwargs):
            self.trusted_calls += 1
            return 409, {"status": "error", "error": "deterministic"}

    launcher = _Launcher()
    app = web.Application()
    app[INSTANCE_MANAGER_APP_KEY] = launcher
    app_module._ensure_shadow_intake_stores(app)
    request = make_mocked_request("POST", "/dlaas/v1/instances/ai-qiao/interactions", app=app)
    response = await app_module._dispatch_envelope_to_instance(
        request,
        "ai-qiao",
        _envelope(),
        require_cognitive_turn_idempotency=True,
    )
    assert response.status == 409
    assert launcher.trusted_calls == 1
