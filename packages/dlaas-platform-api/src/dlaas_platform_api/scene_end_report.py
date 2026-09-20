"""Parent-owned orchestration for durable keyed ``report`` interactions."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from aiohttp import web

from dlaas_platform_contracts import (
    InteractionEnvelope,
    ReportSceneEndRequest,
    SceneEndReserveOutcome,
)
from dlaas_platform_registry import (
    SceneEndLedgerStore,
    SceneEndLedgerTransitionError,
)


_LOG = logging.getLogger("dlaas_platform_api.scene_end_report")


def valid_idempotency_key(value: str) -> bool:
    return bool(
        value
        and value.strip()
        and len(value) <= 256
        and all(character.isprintable() for character in value)
    )


def _json_error(
    *, status: int, error: str, detail: str, extra: Mapping[str, Any] | None = None
) -> web.Response:
    body: dict[str, Any] = {"status": "error", "error": error, "detail": detail}
    if extra is not None:
        body.update(extra)
    return web.json_response(body, status=status)


def _response_json_object(response: web.StreamResponse) -> dict[str, Any]:
    if not isinstance(response, web.Response) or response.body is None:
        raise ValueError("keyed report dispatch did not return a JSON response")
    decoded = json.loads(response.body.decode(response.charset or "utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("keyed report response body must be a JSON object")
    return decoded


def _has_restart_durable_memory_receipt(body: Mapping[str, Any]) -> bool:
    receipt = body.get("memory_checkpoint_receipt")
    if not isinstance(receipt, Mapping):
        return False
    required = {
        "schema_id",
        "schema_version",
        "operation",
        "checkpoint_id",
        "checkpoint_key",
        "checkpoint_version",
        "payload_sha256",
        "payload_bytes",
        "entry_count",
        "durability",
        "completed_at_ms",
        "restored_payload_sha256",
        "restored_matches_persisted",
    }
    if not required.issubset(receipt):
        return False
    payload_sha256 = receipt.get("payload_sha256")
    integer_fields = (
        receipt.get("schema_version"),
        receipt.get("checkpoint_version"),
        receipt.get("payload_bytes"),
        receipt.get("entry_count"),
        receipt.get("completed_at_ms"),
    )
    return (
        receipt.get("schema_id")
        == "volvence.memory.checkpoint-persistence-receipt"
        and integer_fields[0] == 1
        and all(isinstance(value, int) and not isinstance(value, bool) for value in integer_fields)
        and integer_fields[1] >= 0
        and integer_fields[2] > 0
        and integer_fields[3] >= 0
        and integer_fields[4] > 0
        and isinstance(receipt.get("checkpoint_id"), str)
        and bool(receipt["checkpoint_id"].strip())
        and isinstance(receipt.get("checkpoint_key"), str)
        and bool(receipt["checkpoint_key"].strip())
        and isinstance(payload_sha256, str)
        and len(payload_sha256) == 64
        and all(character in "0123456789abcdefABCDEF" for character in payload_sha256)
        and receipt.get("durability") == "restart_durable"
        and receipt.get("operation") == "save"
        and receipt.get("restored_payload_sha256") is None
        and receipt.get("restored_matches_persisted") is None
    )


async def _mark_unknown(
    *, store: SceneEndLedgerStore, reservation: Any, reason: str
) -> None:
    try:
        await store.mark_outcome_unknown(reservation=reservation, reason=reason)
    except SceneEndLedgerTransitionError:
        _LOG.exception("scene-end ledger lost reservation while marking unknown")


async def _run_with_heartbeat(
    *,
    run_dispatch: Callable[[], Awaitable[web.StreamResponse]],
    store: SceneEndLedgerStore,
    reservation: Any,
) -> web.StreamResponse:
    stop = asyncio.Event()

    async def _heartbeat() -> None:
        while True:
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=store.heartbeat_interval_seconds
                )
                return
            except TimeoutError:
                await store.refresh_lease(reservation=reservation)

    dispatch_task = asyncio.create_task(run_dispatch())
    heartbeat_task = asyncio.create_task(_heartbeat())
    try:
        done, _pending = await asyncio.wait(
            {dispatch_task, heartbeat_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if heartbeat_task in done and not dispatch_task.done():
            await heartbeat_task
            raise RuntimeError("scene-end lease heartbeat stopped unexpectedly")
        response = await dispatch_task
        stop.set()
        await heartbeat_task
        return response
    finally:
        stop.set()
        if not heartbeat_task.done():
            heartbeat_task.cancel()
        if not dispatch_task.done():
            dispatch_task.cancel()
        await asyncio.gather(heartbeat_task, dispatch_task, return_exceptions=True)


async def dispatch_keyed_report(
    *,
    ai_id: str,
    envelope: InteractionEnvelope,
    idempotency_key: str,
    store: SceneEndLedgerStore | None,
    run_dispatch: Callable[[], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """Reserve, execute once, and durably finalize a destructive report."""

    if store is None:
        return _json_error(
            status=503,
            error="scene_end_ledger_unavailable",
            detail="keyed report requires the registry-owned durable ledger",
        )
    request_spec = ReportSceneEndRequest.from_envelope(ai_id=ai_id, envelope=envelope)
    reservation = await store.reserve(
        request=request_spec, idempotency_key=idempotency_key
    )
    if reservation.outcome is SceneEndReserveOutcome.REPLAY_COMPLETED:
        status = reservation.record.response_status
        body = reservation.record.response_body
        if status is None or body is None:  # pragma: no cover - typed invariant
            raise RuntimeError("completed scene-end record is missing its response")
        response = web.json_response(dict(body), status=status)
        response.headers["Idempotency-Replayed"] = "true"
        return response
    if reservation.outcome is SceneEndReserveOutcome.CONFLICT:
        return _json_error(
            status=409,
            error="idempotency_key_payload_conflict",
            detail="Idempotency-Key is bound to a different canonical request",
        )
    if reservation.outcome is SceneEndReserveOutcome.IN_PROGRESS:
        return _json_error(
            status=409,
            error="scene_end_in_progress",
            detail="the same keyed report holds an active execution lease",
        )
    if reservation.outcome is SceneEndReserveOutcome.OUTCOME_UNKNOWN:
        return _json_error(
            status=409,
            error="scene_end_outcome_unknown",
            detail="automatic replay is permanently disabled",
            extra={"unknown_reason": reservation.record.unknown_reason},
        )

    try:
        response = await _run_with_heartbeat(
            run_dispatch=run_dispatch,
            store=store,
            reservation=reservation.record,
        )
    except BaseException:
        await _mark_unknown(
            store=store,
            reservation=reservation.record,
            reason="dispatch_exception_or_lease_failure",
        )
        raise
    try:
        body = _response_json_object(response)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        await _mark_unknown(
            store=store,
            reservation=reservation.record,
            reason="response_unrecordable",
        )
        return _json_error(
            status=503,
            error="scene_end_response_unrecordable",
            detail="scene-end response was not a replayable JSON object",
        )
    if 200 <= response.status < 300 and not _has_restart_durable_memory_receipt(body):
        await _mark_unknown(
            store=store,
            reservation=reservation.record,
            reason="memory_receipt_missing_or_not_restart_durable",
        )
        return _json_error(
            status=503,
            error="memory_checkpoint_receipt_required",
            detail="successful keyed report lacked restart_durable Memory proof",
        )
    if 200 <= response.status < 300 or 400 <= response.status < 500:
        try:
            await store.complete(
                reservation=reservation.record,
                response_status=response.status,
                response_body=body,
            )
        except SceneEndLedgerTransitionError:
            await _mark_unknown(
                store=store,
                reservation=reservation.record,
                reason="completion_compare_and_set_failed",
            )
            return _json_error(
                status=503,
                error="scene_end_ledger_finalize_failed",
                detail="completion lost its durable reservation lease",
            )
        return response
    await _mark_unknown(
        store=store,
        reservation=reservation.record,
        reason=f"nondeterministic_http_status_{response.status}",
    )
    return response


__all__ = ["dispatch_keyed_report", "valid_idempotency_key"]
