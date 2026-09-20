"""Async execution boundary for synchronous substrate runtime calls.

Transformers inference is synchronous and a single shared runtime cannot run
two capture/generate operations concurrently.  This module keeps those calls
off the asyncio event loop while preserving one process-local, single-threaded
executor per runtime execution owner. Distinct provider wrappers may publish
the same explicit owner so every call into one loaded model shares both the
serial queue and its worker-thread identity. Runtimes that explicitly publish
concurrent-call support (vLLM) are still offloaded through the default executor.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import contextvars
import hashlib
import logging
import re
import threading
from typing import Protocol, TypeVar, runtime_checkable
from weakref import WeakKeyDictionary

from volvence_zero.substrate.residual_interfaces import OpenWeightResidualRuntime


_ResultT = TypeVar("_ResultT")
_EXECUTORS_GUARD = threading.Lock()
_SERIAL_EXECUTORS: WeakKeyDictionary[object, ThreadPoolExecutor] = (
    WeakKeyDictionary()
)
_SAFE_OPERATION_KIND = re.compile(r"^[a-zA-Z0-9_.:-]{1,80}$")
_LOG = logging.getLogger("volvence_zero.substrate.runtime_execution")


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


def _serial_executor_for(runtime: object) -> ThreadPoolExecutor:
    with _EXECUTORS_GUARD:
        executor = _SERIAL_EXECUTORS.get(runtime)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="vz-runtime-owner",
            )
            _SERIAL_EXECUTORS[runtime] = executor
        return executor


def _safe_operation_label(operation_kind: str) -> str:
    return (
        operation_kind
        if _SAFE_OPERATION_KIND.fullmatch(operation_kind)
        else "runtime_call"
    )


def _failure_origin(exc: BaseException) -> tuple[str, str]:
    traceback = exc.__traceback__
    if traceback is None:
        return ("unknown", "unknown")
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    frame = traceback.tb_frame
    module = str(frame.f_globals.get("__name__", "unknown"))
    function = frame.f_code.co_name
    return (f"{module}.{function}:{traceback.tb_lineno}", type(exc).__name__)


def _log_runtime_failure(*, operation_kind: str, exc: BaseException) -> None:
    cause = exc.__cause__ if exc.__cause__ is not None else exc
    origin, cause_type = _failure_origin(cause)
    fingerprint_payload = (
        f"{operation_kind}\0{cause_type}\0{origin}\0{str(cause)}"
    ).encode("utf-8", errors="replace")
    fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()[:16]
    _LOG.error(
        "runtime call failed; operation_kind=%s cause_type=%s "
        "origin=%s failure_fingerprint=%s",
        _safe_operation_label(operation_kind),
        cause_type,
        origin,
        fingerprint,
    )


async def run_runtime_call(
    *,
    runtime: object,
    operation: Callable[[], _ResultT],
    operation_kind: str = "runtime_call",
) -> _ResultT:
    """Run one synchronous runtime operation without blocking asyncio.

    The open-weight runtime contract defaults to serial execution.  Non-contract
    test doubles are also conservatively serialized, so they exercise the same
    safety boundary instead of silently receiving concurrent calls.
    """

    owner = _execution_owner(runtime)
    try:
        if (
            isinstance(owner, OpenWeightResidualRuntime)
            and owner.supports_concurrent_runtime_calls
        ):
            return await asyncio.to_thread(operation)
        executor = _serial_executor_for(owner)
        context = contextvars.copy_context()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, context.run, operation)
    except Exception as exc:
        _log_runtime_failure(operation_kind=operation_kind, exc=exc)
        raise


__all__ = ["run_runtime_call"]
