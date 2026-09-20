"""Parent-owned orchestration for durable keyed cognitive turns."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from aiohttp import web

from dlaas_platform_contracts import (
    CognitiveTurnRequest,
    CognitiveTurnReserveOutcome,
    InteractionEnvelope,
)
from dlaas_platform_registry import (
    CognitiveTurnLedgerStore,
)


_LOG = logging.getLogger("dlaas_platform_api.cognitive_turn_idempotency")


def _json_error(*, status: int, error: str, detail: str, extra: Mapping[str, Any] | None = None) -> web.Response:
    body: dict[str, Any] = {"status": "error", "error": error, "detail": detail}
    if extra is not None:
        body.update(extra)
    return web.json_response(body, status=status)


def _response_json_object(response: web.StreamResponse) -> dict[str, Any]:
    if not isinstance(response, web.Response) or response.body is None:
        raise ValueError("cognitive-turn dispatch did not return a JSON response")
    decoded = json.loads(response.body.decode(response.charset or "utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("cognitive-turn response body must be a JSON object")
    return decoded


async def _outcome_unknown_response(*, store: CognitiveTurnLedgerStore, reservation: Any, reason: str) -> web.Response:
    try:
        await store.mark_outcome_unknown(reservation=reservation, reason=reason)
    except Exception:  # request boundary: terminal persistence itself failed
        _LOG.exception("cognitive-turn ledger failed while marking outcome unknown")
        return _json_error(
            status=503,
            error="cognitive_turn_ledger_finalize_failed",
            detail=("the cognitive-turn outcome could not be durably classified; automatic retry is not safe"),
        )
    return _json_error(
        status=409,
        error="cognitive_turn_outcome_unknown",
        detail="automatic cognitive-turn replay is permanently disabled",
        extra={"unknown_reason": reason},
    )


async def _run_with_heartbeat(
    *,
    run_dispatch: Callable[[], Awaitable[web.StreamResponse]],
    store: CognitiveTurnLedgerStore,
    reservation: Any,
) -> web.StreamResponse:
    stop = asyncio.Event()

    async def _heartbeat() -> None:
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=store.heartbeat_interval_seconds)
                return
            except TimeoutError:
                await store.refresh_lease(reservation=reservation)

    dispatch_task = asyncio.create_task(run_dispatch())
    heartbeat_task = asyncio.create_task(_heartbeat())
    try:
        done, _pending = await asyncio.wait({dispatch_task, heartbeat_task}, return_when=asyncio.FIRST_COMPLETED)
        if heartbeat_task in done and not dispatch_task.done():
            await heartbeat_task
            raise RuntimeError("cognitive-turn lease heartbeat stopped unexpectedly")
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


async def dispatch_keyed_cognitive_turn(
    *,
    ai_id: str,
    envelope: InteractionEnvelope,
    idempotency_key: str,
    store: CognitiveTurnLedgerStore | None,
    run_dispatch: Callable[[], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """Reserve, execute at most once, and durably finalize one cognitive turn."""

    if store is None:
        return _json_error(
            status=503,
            error="cognitive_turn_ledger_unavailable",
            detail=("cognitive_turn requires the registry-owned durable ledger before run_turn can start"),
        )
    request_spec = CognitiveTurnRequest.from_envelope(ai_id=ai_id, envelope=envelope)
    try:
        reservation = await store.reserve(request=request_spec, idempotency_key=idempotency_key)
    except Exception:  # request boundary: execution has not started
        _LOG.exception("cognitive-turn ledger reservation failed")
        return _json_error(
            status=503,
            error="cognitive_turn_ledger_unavailable",
            detail="cognitive_turn was not dispatched because reservation failed",
        )
    if reservation.outcome is CognitiveTurnReserveOutcome.REPLAY_COMPLETED:
        status = reservation.record.response_status
        body = reservation.record.response_body
        if status is None or body is None:  # pragma: no cover - typed invariant
            raise RuntimeError("completed cognitive-turn record is missing its response")
        response = web.json_response(dict(body), status=status)
        response.headers["Idempotency-Replayed"] = "true"
        return response
    if reservation.outcome is CognitiveTurnReserveOutcome.CONFLICT:
        return _json_error(
            status=409,
            error="idempotency_key_payload_conflict",
            detail="Idempotency-Key is bound to a different canonical request",
        )
    if reservation.outcome is CognitiveTurnReserveOutcome.IN_PROGRESS:
        return _json_error(
            status=409,
            error="cognitive_turn_in_progress",
            detail="the same keyed cognitive turn holds an active execution lease",
        )
    if reservation.outcome is CognitiveTurnReserveOutcome.OUTCOME_UNKNOWN:
        return _json_error(
            status=409,
            error="cognitive_turn_outcome_unknown",
            detail="automatic cognitive-turn replay is permanently disabled",
            extra={"unknown_reason": reservation.record.unknown_reason},
        )

    try:
        response = await _run_with_heartbeat(
            run_dispatch=run_dispatch,
            store=store,
            reservation=reservation.record,
        )
    except Exception as exc:  # request boundary: convert to a durable UNKNOWN receipt
        cause = exc.__cause__ if exc.__cause__ is not None else exc
        _LOG.error(
            "cognitive-turn dispatch or lease heartbeat failed; cause_type=%s",
            type(cause).__name__,
        )
        return await _outcome_unknown_response(
            store=store,
            reservation=reservation.record,
            reason="dispatch_exception_or_lease_failure",
        )
    try:
        body = _response_json_object(response)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return await _outcome_unknown_response(
            store=store,
            reservation=reservation.record,
            reason="response_unrecordable",
        )
    if 200 <= response.status < 300 or 400 <= response.status < 500:
        try:
            await store.complete(
                reservation=reservation.record,
                response_status=response.status,
                response_body=body,
            )
        except Exception:  # request boundary: completion was not durable
            _LOG.exception("cognitive-turn ledger completion failed")
            return await _outcome_unknown_response(
                store=store,
                reservation=reservation.record,
                reason="completion_compare_and_set_failed",
            )
        return response
    return await _outcome_unknown_response(
        store=store,
        reservation=reservation.record,
        reason=f"nondeterministic_http_status_{response.status}",
    )


__all__ = ["dispatch_keyed_cognitive_turn"]
