from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dlaas_platform_contracts import (
    InteractionEnvelope,
    InteractionType,
    ReportSceneEndRequest,
    SceneEndLedgerStatus,
    SceneEndReserveOutcome,
)
from dlaas_platform_registry import (
    Registry,
    SceneEndLedgerStore,
    SceneEndLedgerTransitionError,
)


def _request(*, brief: str = "close") -> ReportSceneEndRequest:
    return ReportSceneEndRequest.from_envelope(
        ai_id="ai-1",
        envelope=InteractionEnvelope(
            contract_id="contract-1",
            session_id="session-1",
            end_user_ref="player-1",
            interaction_type=InteractionType.REPORT,
            human_brief=brief,
            structured_context={"scene_id": "gate"},
        ),
    )


async def test_concurrent_reserve_conflict_and_cas_completion(tmp_path: Path) -> None:
    db_path = tmp_path / "concurrent.sqlite3"
    registry = Registry(db_path=db_path)
    peer_registry = Registry(db_path=db_path)
    store = SceneEndLedgerStore(registry, lease_ms=1_000)
    peer_store = SceneEndLedgerStore(peer_registry, lease_ms=1_000)
    first, second = await asyncio.gather(
        peer_store.reserve(
            request=_request(), idempotency_key="report-1", now_ms=100
        ),
        store.reserve(request=_request(), idempotency_key="report-1", now_ms=100),
    )
    assert {first.outcome, second.outcome} == {
        SceneEndReserveOutcome.ACQUIRED,
        SceneEndReserveOutcome.IN_PROGRESS,
    }
    acquired = first if first.acquired else second
    conflict = await store.reserve(
        request=_request(brief="different"),
        idempotency_key="report-1",
        now_ms=101,
    )
    assert conflict.outcome is SceneEndReserveOutcome.CONFLICT
    completed = await store.complete(
        reservation=acquired.record,
        response_status=200,
        response_body={"status": "ok", "scene_id": "gate"},
        now_ms=110,
    )
    assert completed.status is SceneEndLedgerStatus.COMPLETED
    replay = await store.reserve(
        request=_request(), idempotency_key="report-1", now_ms=120
    )
    assert replay.outcome is SceneEndReserveOutcome.REPLAY_COMPLETED
    assert replay.record.response_body == {"status": "ok", "scene_id": "gate"}
    with pytest.raises(SceneEndLedgerTransitionError):
        await store.complete(
            reservation=acquired.record,
            response_status=200,
            response_body={"status": "second completion"},
            now_ms=130,
        )
    registry.close()
    peer_registry.close()


async def test_heartbeat_prevents_live_lease_expiry_then_crash_expires_unknown() -> None:
    registry = Registry()
    store = SceneEndLedgerStore(registry, lease_ms=100)
    acquired = await store.reserve(
        request=_request(), idempotency_key="report-slow", now_ms=1_000
    )
    refreshed = await store.refresh_lease(
        reservation=acquired.record, now_ms=1_050
    )
    assert refreshed.lease_expires_at_ms == 1_150
    still_live = await store.reserve(
        request=_request(), idempotency_key="report-slow", now_ms=1_101
    )
    assert still_live.outcome is SceneEndReserveOutcome.IN_PROGRESS
    expired = await store.reserve(
        request=_request(), idempotency_key="report-slow", now_ms=1_151
    )
    assert expired.outcome is SceneEndReserveOutcome.OUTCOME_UNKNOWN
    assert expired.record.unknown_reason == "lease_expired"
    retry = await store.reserve(
        request=_request(), idempotency_key="report-slow", now_ms=2_000
    )
    assert retry.outcome is SceneEndReserveOutcome.OUTCOME_UNKNOWN
    with pytest.raises(SceneEndLedgerTransitionError):
        await store.refresh_lease(reservation=acquired.record, now_ms=2_001)


async def test_sqlite_reopen_replays_original_response(tmp_path: Path) -> None:
    db_path = tmp_path / "registry.sqlite3"
    first_registry = Registry(db_path=db_path)
    first_store = SceneEndLedgerStore(first_registry)
    acquired = await first_store.reserve(
        request=_request(), idempotency_key="report-reopen", now_ms=1_000
    )
    await first_store.complete(
        reservation=acquired.record,
        response_status=409,
        response_body={"status": "error", "error": "deterministic"},
        now_ms=1_001,
    )
    first_registry.close()
    reopened_registry = Registry(db_path=db_path)
    reopened_store = SceneEndLedgerStore(reopened_registry)
    replay = await reopened_store.reserve(
        request=_request(), idempotency_key="report-reopen", now_ms=2_000
    )
    assert replay.outcome is SceneEndReserveOutcome.REPLAY_COMPLETED
    assert replay.record.response_status == 409
    assert replay.record.response_body == {
        "status": "error",
        "error": "deterministic",
    }
    reopened_registry.close()


async def test_explicit_unknown_reason_is_durable_terminal() -> None:
    registry = Registry()
    store = SceneEndLedgerStore(registry)
    acquired = await store.reserve(
        request=_request(), idempotency_key="report-timeout", now_ms=1_000
    )
    unknown = await store.mark_outcome_unknown(
        reservation=acquired.record,
        reason="upstream_transport_timeout",
        now_ms=1_001,
    )
    assert unknown.status is SceneEndLedgerStatus.OUTCOME_UNKNOWN
    assert unknown.unknown_reason == "upstream_transport_timeout"
    retry = await store.reserve(
        request=_request(), idempotency_key="report-timeout", now_ms=9_000
    )
    assert retry.outcome is SceneEndReserveOutcome.OUTCOME_UNKNOWN
