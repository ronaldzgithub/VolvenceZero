from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from typing import Any

from aiohttp import web
from aiohttp.test_utils import make_mocked_request
import pytest

from dlaas_platform_api import app as app_module
from dlaas_platform_api.dispatch import DispatchError, dispatch_envelope
from dlaas_platform_contracts import (
    InteractionEnvelope,
    InteractionType,
    OutputContract,
)
from dlaas_platform_launcher import INSTANCE_MANAGER_APP_KEY
from dlaas_platform_registry import Registry, SceneEndLedgerStore


def _envelope(*, brief: str = "close") -> InteractionEnvelope:
    return InteractionEnvelope(
        contract_id="contract-1",
        session_id="session-1",
        end_user_ref="player-1",
        interaction_type=InteractionType.REPORT,
        human_brief=brief,
        structured_context={"scene_id": "gate"},
    )


def _receipt_payload() -> dict[str, Any]:
    return {
        "schema_id": "volvence.memory.checkpoint-persistence-receipt",
        "schema_version": 1,
        "operation": "save",
        "checkpoint_id": "persist-memory/store",
        "checkpoint_key": "memory/store",
        "checkpoint_version": 1,
        "payload_sha256": "a" * 64,
        "payload_bytes": 42,
        "entry_count": 3,
        "durability": "restart_durable",
        "completed_at_ms": 1_800_000_000_000,
        "restored_payload_sha256": None,
        "restored_matches_persisted": None,
    }


def _success_response() -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "scene_id": "gate",
            "memory_checkpoint_receipt": _receipt_payload(),
        }
    )


def _request(store: SceneEndLedgerStore) -> web.Request:
    app = web.Application()
    app[app_module._SCENE_END_LEDGER_STORE_KEY] = store
    return make_mocked_request(
        "POST", "/dlaas/v1/instances/ai-1/interactions", app=app
    )


def _json(response: web.StreamResponse) -> dict[str, Any]:
    assert isinstance(response, web.Response)
    assert response.body is not None
    return json.loads(response.body)


async def test_duplicate_replays_original_and_dispatches_scene_end_once(monkeypatch) -> None:
    store = SceneEndLedgerStore(Registry())
    request = _request(store)
    calls: list[InteractionEnvelope] = []

    async def fake_dispatch(
        _request, _ai_id, envelope, *, require_report_persistence=False
    ):
        calls.append(envelope)
        assert require_report_persistence is True
        return _success_response()

    monkeypatch.setattr(app_module, "_dispatch_envelope_to_instance", fake_dispatch)
    first = await app_module._dispatch_keyed_report(
        request,
        ai_id="ai-1",
        envelope=_envelope(),
        idempotency_key="key-1",
    )
    second = await app_module._dispatch_keyed_report(
        request,
        ai_id="ai-1",
        envelope=_envelope(),
        idempotency_key="key-1",
    )
    assert first.status == second.status == 200
    assert _json(first) == _json(second)
    assert second.headers["Idempotency-Replayed"] == "true"
    assert len(calls) == 1

    conflict = await app_module._dispatch_keyed_report(
        request,
        ai_id="ai-1",
        envelope=_envelope(brief="different"),
        idempotency_key="key-1",
    )
    assert conflict.status == 409
    assert _json(conflict)["error"] == "idempotency_key_payload_conflict"
    assert len(calls) == 1


async def test_keyed_stream_is_rejected_before_reservation_or_dispatch(
    monkeypatch,
) -> None:
    store = SceneEndLedgerStore(Registry())
    request = _request(store)
    called = False

    async def fake_dispatch(*_args, **_kwargs):
        nonlocal called
        called = True
        return _success_response()

    monkeypatch.setattr(app_module, "_dispatch_envelope_to_instance", fake_dispatch)
    response = await app_module._dispatch_keyed_report(
        request,
        ai_id="ai-1",
        envelope=replace(_envelope(), output_contract=OutputContract(stream=True)),
        idempotency_key="key-stream",
    )

    assert response.status == 400
    assert _json(response)["error"] == "keyed_report_stream_not_supported"
    assert called is False
    assert store.get(
        contract_id="contract-1",
        ai_id="ai-1",
        idempotency_key="key-stream",
    ) is None


async def test_heartbeat_keeps_slow_dispatch_in_progress(monkeypatch) -> None:
    store = SceneEndLedgerStore(Registry(), lease_ms=30)
    request = _request(store)
    call_count = 0

    async def slow_dispatch(
        _request, _ai_id, _envelope, *, require_report_persistence=False
    ):
        nonlocal call_count
        assert require_report_persistence is True
        call_count += 1
        await asyncio.sleep(0.09)
        return _success_response()

    monkeypatch.setattr(app_module, "_dispatch_envelope_to_instance", slow_dispatch)
    first_task = asyncio.create_task(
        app_module._dispatch_keyed_report(
            request,
            ai_id="ai-1",
            envelope=_envelope(),
            idempotency_key="key-slow",
        )
    )
    await asyncio.sleep(0.055)
    concurrent = await app_module._dispatch_keyed_report(
        request,
        ai_id="ai-1",
        envelope=_envelope(),
        idempotency_key="key-slow",
    )
    assert concurrent.status == 409
    assert _json(concurrent)["error"] == "scene_end_in_progress"
    assert (await first_task).status == 200
    assert call_count == 1


async def test_transport_5xx_becomes_durable_unknown_and_never_retries(monkeypatch) -> None:
    store = SceneEndLedgerStore(Registry())
    request = _request(store)
    calls = 0

    async def failed_dispatch(
        _request, _ai_id, _envelope, *, require_report_persistence=False
    ):
        nonlocal calls
        assert require_report_persistence is True
        calls += 1
        return web.json_response({"status": "error"}, status=502)

    monkeypatch.setattr(app_module, "_dispatch_envelope_to_instance", failed_dispatch)
    first = await app_module._dispatch_keyed_report(
        request,
        ai_id="ai-1",
        envelope=_envelope(),
        idempotency_key="key-unknown",
    )
    retry = await app_module._dispatch_keyed_report(
        request,
        ai_id="ai-1",
        envelope=_envelope(),
        idempotency_key="key-unknown",
    )
    assert first.status == 502
    assert retry.status == 409
    assert _json(retry)["error"] == "scene_end_outcome_unknown"
    assert calls == 1
    record = store.get(
        contract_id="contract-1", ai_id="ai-1", idempotency_key="key-unknown"
    )
    assert record is not None
    assert record.unknown_reason == "nondeterministic_http_status_502"


async def test_transport_exception_becomes_unknown_without_second_dispatch(
    monkeypatch,
) -> None:
    store = SceneEndLedgerStore(Registry())
    request = _request(store)
    calls = 0

    async def timeout_dispatch(
        _request, _ai_id, _envelope, *, require_report_persistence=False
    ):
        nonlocal calls
        calls += 1
        raise TimeoutError("pod timed out after scene closure")

    monkeypatch.setattr(app_module, "_dispatch_envelope_to_instance", timeout_dispatch)
    with pytest.raises(TimeoutError):
        await app_module._dispatch_keyed_report(
            request,
            ai_id="ai-1",
            envelope=_envelope(),
            idempotency_key="key-timeout",
        )
    retry = await app_module._dispatch_keyed_report(
        request,
        ai_id="ai-1",
        envelope=_envelope(),
        idempotency_key="key-timeout",
    )
    assert retry.status == 409
    assert _json(retry)["unknown_reason"] == "dispatch_exception_or_lease_failure"
    assert calls == 1


async def test_multi_pod_keyed_report_uses_trusted_forward_not_public_forward() -> None:
    class _Launcher:
        trusted_calls = 0

        async def forward_interaction(self, **_kwargs):
            raise AssertionError("keyed report must not use public forwarding")

        async def forward_scene_end_report(self, **_kwargs):
            self.trusted_calls += 1
            return 200, _json(_success_response())

    launcher = _Launcher()
    app = web.Application()
    app[INSTANCE_MANAGER_APP_KEY] = launcher
    app_module._ensure_shadow_intake_stores(app)
    request = make_mocked_request(
        "POST", "/dlaas/v1/instances/ai-1/interactions", app=app
    )
    response = await app_module._dispatch_envelope_to_instance(
        request,
        "ai-1",
        _envelope(),
        require_report_persistence=True,
    )
    assert response.status == 200
    assert launcher.trusted_calls == 1


class _ClosedScene:
    scene_id = "gate"


class _Receipt:
    def __init__(self, durability: str) -> None:
        self.is_restart_durable = durability == "restart_durable"
        self._payload = _receipt_payload()
        self._payload["durability"] = durability

    def to_json(self) -> dict[str, Any]:
        return dict(self._payload)


class _ReportSession:
    def __init__(self, durability: str) -> None:
        self.events: list[str] = []
        self._receipt = _Receipt(durability)

    async def end_scene(self, **_kwargs: Any) -> _ClosedScene:
        self.events.append("end_scene")
        return _ClosedScene()

    def persist_memory_with_receipt(self) -> _Receipt:
        self.events.append("persist_memory")
        return self._receipt


async def test_report_orders_end_scene_before_receipt_and_rejects_process_local() -> None:
    durable = _ReportSession("restart_durable")
    body = await dispatch_envelope(
        envelope=InteractionEnvelope.from_json(
            _envelope().to_json()
        ),
        session=durable,
        ai_id="ai-1",
        require_report_persistence=True,
    )
    assert durable.events == ["end_scene", "persist_memory"]
    assert body["memory_checkpoint_receipt"]["durability"] == "restart_durable"

    process_local = _ReportSession("process_local")
    try:
        await dispatch_envelope(
            envelope=InteractionEnvelope.from_json(
                _envelope().to_json()
            ),
            session=process_local,
            ai_id="ai-1",
            require_report_persistence=True,
        )
    except DispatchError as exc:
        assert exc.status == 503
        assert exc.code == "memory_checkpoint_not_restart_durable"
    else:  # pragma: no cover - fail-loud assertion
        raise AssertionError("process-local receipt must reject keyed report")
    assert process_local.events == ["end_scene", "persist_memory"]
