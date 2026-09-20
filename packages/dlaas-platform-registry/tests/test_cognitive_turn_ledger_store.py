from __future__ import annotations

import asyncio
from pathlib import Path
import sqlite3

import pytest

from dlaas_platform_contracts import (
    CognitiveTurnLedgerStatus,
    CognitiveTurnRequest,
    CognitiveTurnReserveOutcome,
    InteractionEnvelope,
)
from dlaas_platform_registry import (
    CognitiveTurnLedgerStore,
    CognitiveTurnLedgerTransitionError,
    Registry,
)
from dlaas_platform_registry.db import SCHEMA_VERSION


def _envelope(*, perception: str = "I saw the traveller cut the rope.") -> InteractionEnvelope:
    return InteractionEnvelope.from_json(
        {
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
    )


def _request(*, perception: str = "I saw the traveller cut the rope.") -> CognitiveTurnRequest:
    return CognitiveTurnRequest.from_envelope(ai_id="ai-qiao", envelope=_envelope(perception=perception))


async def test_concurrent_reserve_conflict_and_single_completion(tmp_path: Path) -> None:
    db_path = tmp_path / "cognitive.sqlite3"
    registry = Registry(db_path=db_path)
    peer_registry = Registry(db_path=db_path)
    store = CognitiveTurnLedgerStore(registry, lease_ms=1_000)
    peer_store = CognitiveTurnLedgerStore(peer_registry, lease_ms=1_000)
    first, second = await asyncio.gather(
        store.reserve(request=_request(), idempotency_key="turn-1", now_ms=100),
        peer_store.reserve(request=_request(), idempotency_key="turn-1", now_ms=100),
    )
    assert {first.outcome, second.outcome} == {
        CognitiveTurnReserveOutcome.ACQUIRED,
        CognitiveTurnReserveOutcome.IN_PROGRESS,
    }
    acquired = first if first.acquired else second
    conflict = await store.reserve(
        request=_request(perception="I heard the traveller cut the rope."),
        idempotency_key="turn-1",
        now_ms=101,
    )
    assert conflict.outcome is CognitiveTurnReserveOutcome.CONFLICT

    completed = await store.complete(
        reservation=acquired.record,
        response_status=200,
        response_body={"status": "ok", "output_acts": []},
        now_ms=110,
    )
    assert completed.status is CognitiveTurnLedgerStatus.COMPLETED
    replay = await peer_store.reserve(request=_request(), idempotency_key="turn-1", now_ms=120)
    assert replay.outcome is CognitiveTurnReserveOutcome.REPLAY_COMPLETED
    assert replay.record.response_body == {"status": "ok", "output_acts": []}
    with pytest.raises(CognitiveTurnLedgerTransitionError):
        await store.complete(
            reservation=acquired.record,
            response_status=200,
            response_body={"status": "second"},
            now_ms=130,
        )
    registry.close()
    peer_registry.close()


async def test_expired_lease_is_permanently_unknown_and_cannot_complete() -> None:
    registry = Registry()
    store = CognitiveTurnLedgerStore(registry, lease_ms=100)
    acquired = await store.reserve(request=_request(), idempotency_key="turn-expired", now_ms=1_000)
    with pytest.raises(CognitiveTurnLedgerTransitionError):
        await store.complete(
            reservation=acquired.record,
            response_status=200,
            response_body={"status": "too-late"},
            now_ms=1_101,
        )
    expired = await store.reserve(request=_request(), idempotency_key="turn-expired", now_ms=1_101)
    assert expired.outcome is CognitiveTurnReserveOutcome.OUTCOME_UNKNOWN
    assert expired.record.unknown_reason == "lease_expired"
    retry = await store.reserve(request=_request(), idempotency_key="turn-expired", now_ms=9_000)
    assert retry.outcome is CognitiveTurnReserveOutcome.OUTCOME_UNKNOWN


async def test_heartbeat_renews_live_lease() -> None:
    registry = Registry()
    store = CognitiveTurnLedgerStore(registry, lease_ms=100)
    acquired = await store.reserve(request=_request(), idempotency_key="turn-slow", now_ms=1_000)
    refreshed = await store.refresh_lease(reservation=acquired.record, now_ms=1_050)
    assert refreshed.lease_expires_at_ms == 1_150
    live = await store.reserve(request=_request(), idempotency_key="turn-slow", now_ms=1_101)
    assert live.outcome is CognitiveTurnReserveOutcome.IN_PROGRESS


async def test_sqlite_restart_replays_deterministic_response(tmp_path: Path) -> None:
    db_path = tmp_path / "registry.sqlite3"
    first_registry = Registry(db_path=db_path)
    first_store = CognitiveTurnLedgerStore(first_registry)
    acquired = await first_store.reserve(request=_request(), idempotency_key="turn-restart", now_ms=1_000)
    await first_store.complete(
        reservation=acquired.record,
        response_status=409,
        response_body={"status": "error", "error": "deterministic"},
        now_ms=1_001,
    )
    first_registry.close()

    reopened_registry = Registry(db_path=db_path)
    reopened_store = CognitiveTurnLedgerStore(reopened_registry)
    replay = await reopened_store.reserve(request=_request(), idempotency_key="turn-restart", now_ms=2_000)
    assert replay.outcome is CognitiveTurnReserveOutcome.REPLAY_COMPLETED
    assert replay.record.response_status == 409
    assert replay.record.response_body == {
        "status": "error",
        "error": "deterministic",
    }
    reopened_registry.close()


def test_schema_v14_registry_migrates_to_separate_cognitive_table(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-v14.sqlite3"
    legacy = sqlite3.connect(db_path)
    legacy.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    legacy.execute(
        "INSERT INTO schema_meta (key, value) VALUES (?, ?)",
        ("schema_version", "14"),
    )
    legacy.commit()
    legacy.close()

    registry = Registry(db_path=db_path)
    table = registry.conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        ("cognitive_turn_ledger",),
    ).fetchone()
    version = registry.conn.execute("SELECT value FROM schema_meta WHERE key = ?", ("schema_version",)).fetchone()
    assert table is not None
    assert version["value"] == str(SCHEMA_VERSION)
    registry.close()
