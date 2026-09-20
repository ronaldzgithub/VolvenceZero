"""Async execution boundary for synchronous substrate runtime calls.

Transformers inference is synchronous and a single shared runtime cannot run
two capture/generate operations concurrently.  This module keeps those calls
off the asyncio event loop while preserving one process-local, single-threaded
executor per runtime execution owner. Distinct provider wrappers may publish
the same explicit owner so every call into one loaded model shares both the
serial queue and its worker-thread identity. Owner bindings are resolved
recursively: a proposal provider can point at a residual runtime, which in turn
points at the actual loaded model. A separate tokenizer-identity gate spans the
whole operation, including chat-template rendering, tokenization, generation,
and decoding. Runtimes that explicitly publish concurrent-call support (vLLM)
are still offloaded through the default executor.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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
_EXECUTION_LOCKS: WeakKeyDictionary[object, threading.RLock] = (
    WeakKeyDictionary()
)
_TOKENIZER_LOCKS: WeakKeyDictionary[object, threading.RLock] = (
    WeakKeyDictionary()
)
# A few built-in/tokenizer test doubles are deliberately unhashable dataclasses,
# which WeakKeyDictionary cannot index. They are process-long runtime resources,
# so retaining their identity alongside the fallback lock is both bounded by the
# loaded runtimes and prevents Python id reuse from joining unrelated resources.
_UNHASHABLE_TOKENIZER_LOCKS: dict[
    int, tuple[object, threading.RLock]
] = {}
_SAFE_OPERATION_KIND = re.compile(r"^[a-zA-Z0-9_.:-]{1,80}$")
_LOG = logging.getLogger("volvence_zero.substrate.runtime_execution")


@runtime_checkable
class _RuntimeExecutionOwnerBinding(Protocol):
    @property
    def runtime_execution_owner(self) -> object:
        """Return the shared runtime whose model execution this call uses."""


@runtime_checkable
class _RuntimeTokenizerOwnerBinding(Protocol):
    @property
    def runtime_tokenizer_owner(self) -> object:
        """Return the tokenizer shared by this runtime/provider wrapper."""


def _execution_owner(runtime: object) -> object:
    current = runtime
    seen: set[int] = set()
    while isinstance(current, _RuntimeExecutionOwnerBinding):
        current_id = id(current)
        if current_id in seen:
            raise TypeError("runtime_execution_owner bindings must not cycle")
        seen.add(current_id)
        owner = current.runtime_execution_owner
        if owner is None:
            raise TypeError("runtime_execution_owner must not be None")
        if owner is current:
            return current
        current = owner
    return current


def _tokenizer_owner(runtime: object) -> object | None:
    """Resolve a tokenizer identity through nested provider/runtime wrappers."""

    current = runtime
    seen: set[int] = set()
    while True:
        current_id = id(current)
        if current_id in seen:
            raise TypeError("runtime owner bindings must not cycle")
        seen.add(current_id)
        if isinstance(current, _RuntimeTokenizerOwnerBinding):
            tokenizer = current.runtime_tokenizer_owner
            if tokenizer is None:
                raise TypeError("runtime_tokenizer_owner must not be None")
            return tokenizer
        if not isinstance(current, _RuntimeExecutionOwnerBinding):
            return None
        owner = current.runtime_execution_owner
        if owner is None:
            raise TypeError("runtime_execution_owner must not be None")
        if owner is current:
            return None
        current = owner


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


def _tokenizer_lock_for(tokenizer: object) -> threading.RLock:
    with _EXECUTORS_GUARD:
        try:
            lock = _TOKENIZER_LOCKS.get(tokenizer)
        except TypeError:
            identity = id(tokenizer)
            entry = _UNHASHABLE_TOKENIZER_LOCKS.get(identity)
            if entry is None or entry[0] is not tokenizer:
                entry = (tokenizer, threading.RLock())
                _UNHASHABLE_TOKENIZER_LOCKS[identity] = entry
            return entry[1]
        if lock is None:
            lock = threading.RLock()
            _TOKENIZER_LOCKS[tokenizer] = lock
        return lock


def _execution_lock_for(owner: object) -> threading.RLock:
    with _EXECUTORS_GUARD:
        lock = _EXECUTION_LOCKS.get(owner)
        if lock is None:
            lock = threading.RLock()
            _EXECUTION_LOCKS[owner] = lock
        return lock


@contextmanager
def runtime_resource_guard(
    runtime: object,
    *,
    include_execution_owner: bool = True,
) -> Iterator[None]:
    """Serialize access to the model and tokenizer behind ``runtime``.

    Locks are keyed by the canonical resource identities, ordered by identity,
    and reentrant. Consequently a provider can use this guard for direct
    synchronous callers while ``run_runtime_call`` safely nests the same guard
    on its dedicated owner thread.
    """

    resources: list[tuple[object, threading.RLock]] = []
    if include_execution_owner:
        owner = _execution_owner(runtime)
        resources.append((owner, _execution_lock_for(owner)))
    tokenizer_owner = _tokenizer_owner(runtime)
    if tokenizer_owner is not None and all(
        tokenizer_owner is not resource for resource, _ in resources
    ):
        resources.append(
            (tokenizer_owner, _tokenizer_lock_for(tokenizer_owner))
        )
    acquired: list[threading.RLock] = []
    try:
        for _, lock in sorted(resources, key=lambda item: id(item[0])):
            lock.acquire()
            acquired.append(lock)
        yield
    finally:
        for lock in reversed(acquired):
            lock.release()


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
    supports_concurrent_calls = (
        isinstance(owner, OpenWeightResidualRuntime)
        and owner.supports_concurrent_runtime_calls
    )

    def guarded_operation() -> _ResultT:
        with runtime_resource_guard(
            runtime,
            include_execution_owner=not supports_concurrent_calls,
        ):
            return operation()

    try:
        if supports_concurrent_calls:
            return await asyncio.to_thread(guarded_operation)
        executor = _serial_executor_for(owner)
        context = contextvars.copy_context()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            context.run,
            guarded_operation,
        )
    except Exception as exc:
        _log_runtime_failure(operation_kind=operation_kind, exc=exc)
        raise


__all__ = ["run_runtime_call", "runtime_resource_guard"]
