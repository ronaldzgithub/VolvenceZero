"""Lifeform metabolic tick engine.

Independent of the kernel — the kernel runs purely turn-driven. This engine
gives the lifeform its own clock so that "alive between turns" behaviour
(scene staleness, follow-up due time, slow reflection budgeting) has a
real basis.

Design choices:

* **Monotonic tick index, not wall-clock**: deterministic for replay /
  benchmark / regression tests. ``elapsed_seconds`` is computed from the
  configured tick period, not ``time.monotonic()``.
* **Configurable cadence per kind**: SYSTEM is the base; ENERGY and CONTEXT
  fire on multiples of SYSTEM.
* **Hookable**: callers register handlers. Handlers are awaited in
  registration order. A handler raising propagates — we do **not** swallow
  errors here (R8 / no-swallow rule).

The kernel does not import this module.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from lifeform_core.types import TickEvent, TickKind


TickHandler = Callable[[TickEvent], Awaitable[None] | None]


@dataclass(frozen=True)
class TickEngineConfig:
    """Tick cadence config.

    Defaults are conservative enough for tests and short demos. Production
    services should tune ``system_tick_seconds`` to e.g. 1.0–5.0s.
    """

    system_tick_seconds: float = 1.0
    energy_every_n_system_ticks: int = 5
    context_every_n_system_ticks: int = 30


class TickEngine:
    """Async, deterministic metabolic clock.

    Use either as an explicit pump (call ``await advance(n)``) for tests, or
    as an autonomous task (call ``await start()`` then ``await stop()``).
    """

    def __init__(self, config: TickEngineConfig | None = None) -> None:
        self._config = config or TickEngineConfig()
        self._handlers: list[TickHandler] = []
        self._tick_index = 0
        self._task: asyncio.Task[None] | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def config(self) -> TickEngineConfig:
        return self._config

    @property
    def tick_index(self) -> int:
        return self._tick_index

    @property
    def elapsed_seconds(self) -> float:
        return self._tick_index * self._config.system_tick_seconds

    def register(self, handler: TickHandler) -> None:
        self._handlers.append(handler)

    # ------------------------------------------------------------------
    # Pump-style API (preferred for tests / replay)
    # ------------------------------------------------------------------

    async def advance(self, system_ticks: int = 1, *, reason: str = "") -> tuple[TickEvent, ...]:
        """Advance the clock by ``system_ticks`` SYSTEM ticks.

        Returns every TickEvent fired in order (SYSTEM events plus any ENERGY
        and CONTEXT events that align with the multipliers).
        """
        if system_ticks <= 0:
            raise ValueError("system_ticks must be positive")

        events: list[TickEvent] = []
        for _ in range(system_ticks):
            self._tick_index += 1
            events.extend(await self._fire_for_index(self._tick_index, reason=reason))
        return tuple(events)

    async def _fire_for_index(self, index: int, *, reason: str) -> list[TickEvent]:
        out: list[TickEvent] = []
        out.append(
            TickEvent(
                tick_index=index,
                kind=TickKind.SYSTEM,
                elapsed_seconds=index * self._config.system_tick_seconds,
                reason=reason,
            )
        )
        if index % self._config.energy_every_n_system_ticks == 0:
            out.append(
                TickEvent(
                    tick_index=index,
                    kind=TickKind.ENERGY,
                    elapsed_seconds=index * self._config.system_tick_seconds,
                    reason=reason,
                )
            )
        if index % self._config.context_every_n_system_ticks == 0:
            out.append(
                TickEvent(
                    tick_index=index,
                    kind=TickKind.CONTEXT,
                    elapsed_seconds=index * self._config.system_tick_seconds,
                    reason=reason,
                )
            )
        for ev in out:
            await self._dispatch(ev)
        return out

    async def _dispatch(self, event: TickEvent) -> None:
        for handler in self._handlers:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result

    # ------------------------------------------------------------------
    # Autonomous loop API (for long-running services)
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            raise RuntimeError("TickEngine is already running")
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="lifeform-tick")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._config.system_tick_seconds)
            except asyncio.CancelledError:
                break
            if not self._running:
                break
            self._tick_index += 1
            await self._fire_for_index(self._tick_index, reason="auto")
