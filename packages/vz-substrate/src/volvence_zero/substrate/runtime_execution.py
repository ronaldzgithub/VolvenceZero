"""Async execution boundary for synchronous substrate runtime calls.

Transformers inference is synchronous and a single shared runtime cannot run
two capture/generate operations concurrently.  This module keeps those calls
off the asyncio event loop while preserving one process-local lock per runtime
execution owner. Distinct provider wrappers may publish the same explicit
owner so every call into one loaded model shares that lock. Runtimes that
explicitly publish concurrent-call support (vLLM) are still offloaded, but do
not pass through the serial lock.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import threading
from typing import Protocol, TypeVar, runtime_checkable
from weakref import WeakKeyDictionary

from volvence_zero.substrate.residual_interfaces import OpenWeightResidualRuntime


_ResultT = TypeVar("_ResultT")
_LOCKS_GUARD = threading.Lock()
_SERIAL_LOCKS: WeakKeyDictionary[object, threading.Lock] = WeakKeyDictionary()


@runtime_checkable
class _RuntimeExecutionOwnerBinding(Protocol):
    @property
    def runtime_execution_owner(self) -> object:
        """Return the shared runtime whose model execution this call uses."""


def _execution_owner(runtime: object) -> object:
    if isinstance(runtime, _RuntimeExecutionOwnerBinding):
        owner = runtime.runtime_execution_owner
        if owner is None:
            raise TypeError("runtime_execution_owner must not be None")
        return owner
    return runtime


def _serial_lock_for(runtime: object) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _SERIAL_LOCKS.get(runtime)
        if lock is None:
            lock = threading.Lock()
            _SERIAL_LOCKS[runtime] = lock
        return lock


def _call_serialized(
    lock: threading.Lock,
    operation: Callable[[], _ResultT],
) -> _ResultT:
    with lock:
        return operation()


async def run_runtime_call(
    *,
    runtime: object,
    operation: Callable[[], _ResultT],
) -> _ResultT:
    """Run one synchronous runtime operation without blocking asyncio.

    The open-weight runtime contract defaults to serial execution.  Non-contract
    test doubles are also conservatively serialized, so they exercise the same
    safety boundary instead of silently receiving concurrent calls.
    """

    owner = _execution_owner(runtime)
    if (
        isinstance(owner, OpenWeightResidualRuntime)
        and owner.supports_concurrent_runtime_calls
    ):
        return await asyncio.to_thread(operation)
    lock = _serial_lock_for(owner)
    return await asyncio.to_thread(_call_serialized, lock, operation)


__all__ = ["run_runtime_call"]
