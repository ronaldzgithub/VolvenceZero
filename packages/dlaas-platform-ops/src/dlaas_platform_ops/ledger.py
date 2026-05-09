"""Conversation ledger + admin SSE event broker.

The ledger records every notable event the platform sees:

* ``turn`` — user / AI turn appended to a session.
* ``pause`` / ``resume`` — operator-takeover state changes.
* ``operator_message`` — operator turn injected during takeover.
* ``handoff_open`` / ``handoff_resolved`` — ticket lifecycle events.

Subscribers (the admin SSE stream) receive every event from the
moment they connect; the ledger does not replay history. A small
in-memory ring buffer is kept for diagnostics only and is not part
of the public ledger contract.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any


_RING_BUFFER_CAP = 256


@dataclass(frozen=True)
class LedgerEvent:
    """One event entry. ``event_type`` is a typed string slug."""

    event_type: str
    payload: Mapping[str, Any]
    created_at_ms: int

    def to_json(self) -> dict[str, Any]:
        body = {
            "type": self.event_type,
            "created_at_ms": self.created_at_ms,
            **self.payload,
        }
        return body


class LedgerBroker:
    """Pub-sub for ledger events, used by the admin SSE stream."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[LedgerEvent]] = []
        self._lock = asyncio.Lock()
        self._ring: list[LedgerEvent] = []

    async def publish(
        self,
        *,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> LedgerEvent:
        event = LedgerEvent(
            event_type=event_type,
            payload=dict(payload or {}),
            created_at_ms=int(time.time() * 1000.0),
        )
        async with self._lock:
            self._ring.append(event)
            if len(self._ring) > _RING_BUFFER_CAP:
                self._ring = self._ring[-_RING_BUFFER_CAP:]
            queues = list(self._subscribers)
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - defensive
                pass
        return event

    async def subscribe(self) -> "_LedgerSubscription":
        q: asyncio.Queue[LedgerEvent] = asyncio.Queue(maxsize=512)
        async with self._lock:
            self._subscribers.append(q)
        return _LedgerSubscription(broker=self, queue=q)

    async def unsubscribe(self, queue: asyncio.Queue[LedgerEvent]) -> None:
        async with self._lock:
            try:
                self._subscribers.remove(queue)
            except ValueError:
                pass

    async def recent(self, limit: int = 50) -> tuple[LedgerEvent, ...]:
        """Read the in-memory tail; intended for diagnostics only."""
        async with self._lock:
            tail = self._ring[-max(0, limit) :]
            return tuple(tail)


class _LedgerSubscription:
    """Async-iterator handle the SSE writer pulls from."""

    def __init__(
        self, *, broker: LedgerBroker, queue: asyncio.Queue[LedgerEvent]
    ) -> None:
        self._broker = broker
        self._queue = queue
        self._closed = False

    def __aiter__(self) -> AsyncIterator[LedgerEvent]:
        return self

    async def __anext__(self) -> LedgerEvent:
        if self._closed:
            raise StopAsyncIteration
        return await self._queue.get()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._broker.unsubscribe(self._queue)


__all__ = ["LedgerBroker", "LedgerEvent"]
